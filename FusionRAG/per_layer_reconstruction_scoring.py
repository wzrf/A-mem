#!/usr/bin/env python3
"""
Per-Layer Token Selection using Reconstruction-based Scoring (KVzip style)

核心思想：
1. 不使用 query attention 来评分，而是使用"重复上下文"任务
2. 让模型执行 "Repeat the previous context exactly." 任务
3. 在重复过程中，模型高度关注的 tokens 就是重要的 tokens
4. 这是 query-agnostic 的评分方式

实现：
- 分层选择（每层选不同 tokens）
- 同层各 head 选相同 tokens（先验证这个基线）
"""

import json
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from typing import List, Dict, Tuple, Optional
import math

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache


def compute_reconstruction_scores(
    model,
    tokenizer,
    doc_text: str,
    prefix_cache_k: torch.Tensor,  # [num_layers, 1, num_kv_heads, prefix_len, head_dim]
    prefix_cache_v: torch.Tensor,
    prefix_len: int,
    device: str = "cuda:0",
    chunk_size: int = 512,
) -> Dict[int, np.ndarray]:
    """
    使用文本重构任务计算每层每个 token 的重要性分数

    方法：
    1. 构建重复任务: "Repeat the previous context exactly." + document_chunk
    2. 执行 forward，计算 attention weights
    3. 取每个 token 被 attend 到的最大值作为其重要性分数

    Args:
        model: 语言模型
        tokenizer: tokenizer
        doc_text: 完整文档文本
        prefix_cache_k/v: system prompt 的 KV cache
        prefix_len: system prompt 长度
        device: 设备
        chunk_size: 分块大小

    Returns:
        layer_scores: {layer_idx: np.ndarray of shape [doc_len]}
    """

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_key_value_groups = num_heads // num_kv_heads

    # Tokenize document
    doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
    doc_len = len(doc_tokens)

    print(f"\n{'='*80}")
    print("Computing Reconstruction-based Scores (KVzip style)")
    print(f"{'='*80}")
    print(f"Document length: {doc_len} tokens")
    print(f"Prefix length: {prefix_len} tokens")
    print(f"Chunk size: {chunk_size}")

    # Initialize scores: [num_layers, doc_len]
    layer_scores = {layer_idx: np.zeros(doc_len) for layer_idx in range(num_layers)}

    # 构建重复任务的 prompt
    repeat_prompt = "\n\nRepeat the previous context exactly."
    repeat_prompt_ids = tokenizer.encode(repeat_prompt, add_special_tokens=False)

    # 分块处理文档
    num_chunks = (doc_len + chunk_size - 1) // chunk_size
    print(f"Number of chunks: {num_chunks}")

    # 创建完整的输入序列: prefix + doc + repeat_prompt + doc_chunk (target)
    # 但我们这里简化：直接用 prefix + doc 作为 context，然后让模型 attend

    # 对于每个 chunk，计算其 tokens 的重要性
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, doc_len)
        chunk_tokens = doc_tokens[chunk_start:chunk_end]
        chunk_len = len(chunk_tokens)

        print(f"\n  Chunk {chunk_idx + 1}/{num_chunks}: tokens {chunk_start}-{chunk_end}")

        # 构建输入: [repeat_prompt] + [chunk_tokens] (作为要重复的目标)
        # 这里我们计算：当模型尝试生成 chunk_tokens 时，它会 attend 到 doc 的哪些位置

        # 构建完整序列：prefix + doc[0:chunk_end] + repeat_prompt + chunk_tokens
        context_tokens = doc_tokens[:chunk_end]
        input_tokens = repeat_prompt_ids + chunk_tokens

        # 创建 input_ids
        full_input = context_tokens + input_tokens
        input_ids = torch.tensor([full_input], dtype=torch.long, device=device)

        # 创建 position_ids
        # prefix 部分: 0 ~ prefix_len-1
        # doc 部分: prefix_len ~ prefix_len + chunk_end - 1
        # repeat_prompt + chunk: prefix_len + chunk_end ~ end
        total_len = len(full_input)
        position_ids = torch.arange(prefix_len, prefix_len + total_len, device=device).unsqueeze(0)

        # Forward pass，收集 attention weights
        with torch.no_grad():
            # 使用 output_attentions=True 获取 attention weights
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        # outputs.attentions: tuple of [batch, num_heads, seq_len, seq_len]
        attentions = outputs.attentions

        # 计算每个 context token 的重要性分数
        # 只关注 repeat_prompt + chunk_tokens 对 context 的 attention
        query_start = len(context_tokens)  # repeat_prompt 开始的位置
        query_end = total_len
        context_end = len(context_tokens)

        for layer_idx, attn in enumerate(attentions):
            # attn: [1, num_heads, seq_len, seq_len]
            # 提取 query (repeat部分) 对 context (doc部分) 的 attention
            # query positions: query_start ~ query_end
            # key positions: 0 ~ context_end (doc tokens in this input)

            attn_to_context = attn[0, :, query_start:query_end, :context_end]  # [num_heads, q_len, context_len]

            # 对每个 context position，取所有 query 和 head 的最大 attention
            max_attn_per_pos = attn_to_context.amax(dim=(0, 1))  # [context_len]
            max_attn_per_pos = max_attn_per_pos.cpu().numpy()

            # 更新分数（取最大值）
            layer_scores[layer_idx][:chunk_end] = np.maximum(
                layer_scores[layer_idx][:chunk_end],
                max_attn_per_pos
            )

        # 清理显存
        del outputs, attentions
        torch.cuda.empty_cache()

    # 打印统计信息
    print(f"\n{'='*80}")
    print("Reconstruction Score Statistics")
    print(f"{'='*80}")
    print(f"{'Layer':<6} {'Mean':<12} {'Std':<12} {'Max':<12} {'P99':<12} {'P95':<12}")
    print("-" * 70)

    for layer_idx in [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]:
        scores = layer_scores[layer_idx]
        print(f"{layer_idx:<6} {np.mean(scores):<12.6f} {np.std(scores):<12.6f} "
              f"{np.max(scores):<12.6f} {np.percentile(scores, 99):<12.6f} "
              f"{np.percentile(scores, 95):<12.6f}")

    return layer_scores


def select_tokens_per_layer(
    layer_scores: Dict[int, np.ndarray],
    total_ratio: float,
    doc_len: int,
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    基于重构分数进行分层 token 选择（每层选不同 tokens，但同层各 head 相同）

    Args:
        layer_scores: {layer_idx: scores array}
        total_ratio: 目标选择比例
        doc_len: 文档长度

    Returns:
        layer_selections: {layer_idx: [selected_positions]}
        layer_thresholds: {layer_idx: threshold}
    """
    num_layers = len(layer_scores)
    target_count = max(int(doc_len * total_ratio), 20)

    print(f"\n{'='*80}")
    print(f"Layer-wise Token Selection (ratio={total_ratio}, target={target_count} per layer)")
    print(f"{'='*80}")

    layer_selections = {}
    layer_thresholds = {}

    for layer_idx in range(num_layers):
        scores = layer_scores[layer_idx]

        # 选择 top-k
        sorted_indices = np.argsort(scores)[::-1]
        selected_positions = sorted_indices[:target_count].tolist()

        # 计算阈值
        threshold = scores[sorted_indices[min(target_count-1, len(sorted_indices)-1)]]

        layer_selections[layer_idx] = selected_positions
        layer_thresholds[layer_idx] = threshold

        if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx:2d}: selected {len(selected_positions):4d} tokens, "
                  f"threshold={threshold:.6f}")

    return layer_selections, layer_thresholds


def sparse_prefill_layer_wise(
    model,
    past_key_values,
    layer_selections: Dict[int, List[int]],
    position_to_token: Dict[int, int],
    query_tensor: torch.Tensor,
    prefix_len: int,
    doc_len: int,
    device: str = "cuda:0"
):
    """
    执行分层稀疏 prefill（同层各 head 选相同 tokens）

    与 per-head 版本的区别：
    - 每层只有一个选择列表（不是每个 head 一个）
    - 同层的 4 个 kv_head 更新相同位置的 cache
    """

    print(f"\n{'='*80}")
    print("Sparse Prefill (Layer-wise, same tokens per head)")
    print(f"{'='*80}")

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_key_value_groups = num_heads // num_kv_heads
    hidden_size = model.config.hidden_size

    # Layer 0 的选择决定了 sparse 序列中的 critical tokens
    critical_positions = sorted(layer_selections[0])
    critical_token_ids = [position_to_token[pos] for pos in critical_positions]

    query_len = query_tensor.shape[1]
    num_critical = len(critical_positions)

    print(f"Critical tokens (Layer 0): {num_critical}")
    print(f"Query tokens: {query_len}")
    print(f"Total sparse sequence: {num_critical + query_len}")

    # 构建 sparse 序列
    sparse_token_ids = critical_token_ids + query_tensor[0].tolist()
    sparse_input = torch.tensor([sparse_token_ids], dtype=torch.long, device=device)

    # 计算 position_ids
    critical_abs_positions = [prefix_len + pos for pos in critical_positions]
    query_start_pos = prefix_len + doc_len
    query_positions = list(range(query_start_pos, query_start_pos + query_len))

    all_positions = critical_abs_positions + query_positions
    position_ids = torch.tensor([all_positions], dtype=torch.long, device=device)

    print(f"\nPosition mapping:")
    print(f"  Critical positions: {min(critical_abs_positions)} - {max(critical_abs_positions)}")
    print(f"  Query positions: {min(query_positions)} - {max(query_positions)}")

    # 获取 RoPE 参数
    rotary_emb = model.model.layers[0].self_attn.rotary_emb

    # 逐层处理
    print(f"\nProcessing {num_layers} layers...")

    # 统计信息
    layer_stats = []

    with torch.no_grad():
        # Embedding
        hidden_states = model.model.embed_tokens(sparse_input)

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            # 获取该层的选择
            layer_selected_positions = set(layer_selections[layer_idx])

            # LayerNorm
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, seq_len, _ = hidden_states.shape

            # Q/K/V projection
            q_states = layer.self_attn.q_proj(hidden_states)
            k_states = layer.self_attn.k_proj(hidden_states)
            v_states = layer.self_attn.v_proj(hidden_states)

            q_states = q_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
            k_states = k_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v_states = v_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            cos, sin = rotary_emb(v_states, position_ids)

            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
                cos = cos.unsqueeze(unsqueeze_dim)
                sin = sin.unsqueeze(unsqueeze_dim)
                q_embed = (q * cos) + (rotate_half(q) * sin)
                k_embed = (k * cos) + (rotate_half(k) * sin)
                return q_embed, k_embed

            q_states, k_states_rope = apply_rotary_pos_emb(q_states, k_states, cos, sin)

            # 更新 cache：只更新该层选中的 positions
            # 由于是 layer-wise，所有 heads 更新相同的 positions
            update_count = 0
            for sparse_idx in range(num_critical):
                doc_pos = critical_positions[sparse_idx]
                abs_pos = critical_abs_positions[sparse_idx]

                if doc_pos in layer_selected_positions:
                    # 更新所有 kv_heads
                    past_key_values.key_cache[layer_idx][:, :, abs_pos, :] = k_states_rope[:, :, sparse_idx, :]
                    past_key_values.value_cache[layer_idx][:, :, abs_pos, :] = v_states[:, :, sparse_idx, :]
                    update_count += 1

            layer_stats.append({
                'layer_idx': layer_idx,
                'selected_count': len(layer_selected_positions),
                'update_count': update_count,
                'ratio': update_count / doc_len if doc_len > 0 else 0
            })

            # Attention using full cache
            full_k = past_key_values.key_cache[layer_idx][:, :, :query_start_pos + query_len, :]
            full_v = past_key_values.value_cache[layer_idx][:, :, :query_start_pos + query_len, :]

            # Expand for GQA
            full_k = full_k.repeat_interleave(num_key_value_groups, dim=1)
            full_v = full_v.repeat_interleave(num_key_value_groups, dim=1)

            # 构建 q_idx for causal attention
            sparse_len = num_critical + query_len
            q_idx = torch.zeros(1, sparse_len, dtype=torch.long, device=device)

            for i in range(num_critical):
                q_idx[0, i] = critical_abs_positions[i]
            for i in range(query_len):
                q_idx[0, num_critical + i] = query_positions[i]

            # Standard attention with causal mask
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_states, full_k, full_v, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            # Residual + MLP
            hidden_states = residual + attn_output
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                print(f"  Layer {layer_idx:2d}: updated {update_count:4d} positions ({update_count/doc_len*100:.1f}%)")

    # 更新 query 部分的 cache
    # Query tokens 需要完整的 forward
    query_hidden = hidden_states[:, -query_len:, :]

    # Update cache length
    past_key_values._seen_tokens = query_start_pos + query_len

    print(f"\nCache updated: total length = {past_key_values._seen_tokens}")

    # 计算总体重算比例
    total_updates = sum(s['update_count'] for s in layer_stats)
    total_possible = doc_len * num_layers
    overall_ratio = total_updates / total_possible if total_possible > 0 else 0

    print(f"\n{'='*80}")
    print("Recomputation Statistics")
    print(f"{'='*80}")
    print(f"Total document tokens: {doc_len}")
    print(f"Total layers: {num_layers}")
    print(f"Total possible recomputations: {total_possible}")
    print(f"Actual recomputations: {total_updates}")
    print(f"Overall recomputation ratio: {overall_ratio*100:.2f}%")

    return layer_stats, overall_ratio


def main(
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    example_idx=4,
    sub_question_idx=1,
    total_ratio=0.3,
    max_new_tokens=100,
    device="cuda:0",
    output_path='./reconstruction_scoring_results.json'
):
    """
    Main function: 基于重构评分的分层 token 选择
    """

    print(f"\n{'='*100}")
    print("Reconstruction-based Scoring for Layer-wise Token Selection")
    print(f"{'='*100}\n")

    # Load model
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, device_map = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Update device
    if device_map and isinstance(device_map, dict):
        device = f"cuda:{list(device_map.values())[0]}" if 'cuda' not in str(list(device_map.values())[0]) else str(list(device_map.values())[0])

    # Load data
    print("\nLoading data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen', 10,
        max_main_questions=example_idx + 1,
        preprocess=False
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"\n{'='*80}")
    print(f"Example {example_idx}, Sub-question {sub_question_idx}")
    print(f"{'='*80}")
    print(f"Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")

    # Get document info
    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    passages = q_data['docs']

    # Build document text (same format as cache generation)
    doc_texts = []
    for chunk_id in doc_chunk_ids:
        if chunk_id == 0:
            continue  # Skip system prompt
        passage = passages[chunk_id - 1]
        doc_text = f"Document: {passage}\n"
        doc_texts.append(doc_text)

    full_doc_text = "".join(doc_texts)
    doc_tokens = tokenizer.encode(full_doc_text, add_special_tokens=False)
    doc_len = len(doc_tokens)

    print(f"\nDocument: {len(doc_texts)} passages, {doc_len} tokens")

    # Build position_to_token mapping
    position_to_token = {}
    for i, token in enumerate(doc_tokens):
        position_to_token[i] = token

    # Load prefix cache (system prompt)
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')
    prefix_key_path = f'{full_cache_path}/{example_idx}_0_key.pt'
    prefix_value_path = f'{full_cache_path}/{example_idx}_0_value.pt'

    prefix_k = torch.load(prefix_key_path, weights_only=True)
    prefix_v = torch.load(prefix_value_path, weights_only=True)
    prefix_len = prefix_k[0].shape[2]

    print(f"Prefix (system prompt) length: {prefix_len}")

    # Step 1: Compute reconstruction-based scores
    layer_scores = compute_reconstruction_scores(
        model, tokenizer, full_doc_text,
        prefix_k, prefix_v, prefix_len,
        device=device, chunk_size=512
    )

    # Step 2: Select tokens per layer
    layer_selections, layer_thresholds = select_tokens_per_layer(
        layer_scores, total_ratio, doc_len
    )

    # Step 3: Load full cache and do sparse prefill
    print(f"\n{'='*80}")
    print("Loading full cache...")
    print(f"{'='*80}")

    max_cache_len = 32768
    cache_device = device_map if device_map else device
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=cache_device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Load and concatenate caches
    total_cache_len = 0
    for chunk_id in doc_chunk_ids:
        key_path = f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt'
        value_path = f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt'

        if not os.path.exists(key_path):
            print(f"  Warning: Cache not found for chunk {chunk_id}")
            continue

        key_cache = torch.load(key_path, weights_only=True)
        value_cache = torch.load(value_path, weights_only=True)

        chunk_len = key_cache[0].shape[2]

        for layer_idx in range(len(key_cache)):
            k = key_cache[layer_idx].to(device)
            v = value_cache[layer_idx].to(device)
            past_key_values.key_cache[layer_idx][:, :, total_cache_len:total_cache_len + chunk_len, :] = k
            past_key_values.value_cache[layer_idx][:, :, total_cache_len:total_cache_len + chunk_len, :] = v

        total_cache_len += chunk_len
        print(f"  Loaded chunk {chunk_id}: {chunk_len} tokens")

    past_key_values._seen_tokens = total_cache_len
    print(f"\nTotal cache length: {total_cache_len}")

    # Build query tensor
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor([question_tokens], dtype=torch.long, device=device)

    # Step 4: Sparse prefill (layer-wise)
    layer_stats, overall_ratio = sparse_prefill_layer_wise(
        model, past_key_values, layer_selections, position_to_token,
        query_tensor, prefix_len, doc_len, device
    )

    # Step 5: Generate
    print(f"\n{'='*80}")
    print("Generating...")
    print(f"{'='*80}")

    # Re-forward query to get correct hidden states
    query_start_pos = prefix_len + doc_len
    query_positions = torch.arange(query_start_pos, query_start_pos + query_tensor.shape[1], device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=query_tensor,
            position_ids=query_positions,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    # Generate tokens
    generated_ids = []
    next_token_logits = outputs.logits[:, -1, :]

    for step in range(max_new_tokens):
        next_token = torch.argmax(next_token_logits, dim=-1)
        generated_ids.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            print(f"  EOS at step {step}")
            break

        if step < 10 or step % 10 == 0:
            token_text = tokenizer.decode([next_token.item()])
            print(f"  Step {step}: {token_text}")

        # Next step
        next_pos = torch.tensor([[past_key_values._seen_tokens]], device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=next_token.unsqueeze(0),
                position_ids=next_pos,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        next_token_logits = outputs.logits[:, -1, :]

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Print results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"\nGround Truth:\n{sub_q_info['answer']}")
    print(f"\nGenerated Answer:\n{generated_text}")
    print(f"\n{'='*80}")

    # Save results
    results = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'generated_answer': generated_text,
        'total_ratio': total_ratio,
        'overall_recompute_ratio': overall_ratio,
        'doc_len': doc_len,
        'scoring_method': 'reconstruction',
        'selection_method': 'layer_wise',
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    results = main(
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        example_idx=4,
        sub_question_idx=1,
        total_ratio=0.4,
        max_new_tokens=100,
        device='cuda:0',
        output_path='./reconstruction_scoring_results.json'
    )

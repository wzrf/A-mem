#!/usr/bin/env python3
"""
Compare layer-wise token selection vs uniform selection

New approach: Each layer selects different tokens based on its own attention distribution
"""

import json
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoConfig

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import (
    prefill_and_save_kv_cache,
    prefill_and_generate,
    rotate_half,
)
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect import load_model, prepare_reflect_data, PreprocessScope


def compute_layer_wise_attention_selection(
    model,
    tokenizer,
    past_key_values,
    passages,
    chunk_ids,
    query_tensor,
    example_id,
    full_cache_path,
    ratio=0.3,
    device="cuda:0",
    device_map=None
):
    """
    对每层计算attention分布，选择该层最重要的tokens

    Args:
        model: 模型
        tokenizer: tokenizer
        past_key_values: KV cache
        passages: 文档列表（不包括query）
        chunk_ids: chunk IDs
        query_tensor: query tokens
        example_id: 样本ID
        full_cache_path: 完整KV cache的路径（rate=1.0）
        ratio: 每层保留的token比例
        device: 设备
        device_map: 设备映射

    Returns:
        layer_wise_selections: List[np.ndarray], 每层选择的token indices
        layer_wise_scores: List[np.ndarray], 每层的attention scores
    """
    input_device = "cuda:0" if device_map is not None else device
    num_layers = len(model.model.layers)

    print(f"\n{'='*100}")
    print("Computing layer-wise attention-based token selection")
    print(f"{'='*100}")
    print(f"Number of layers: {num_layers}")
    print(f"Selection ratio: {ratio}")
    print(f"Loading from: {full_cache_path}")

    # 第一步：加载完整的document KV caches（rate=1.0）
    system_len = passages[0].shape[0]
    past_len = 0

    # 加载所有document caches
    doc_key_caches = []
    doc_value_caches = []
    doc_lengths = []

    for idx, passage in enumerate(passages):  # passages已经不包括query了
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]

        # 加载该chunk的KV cache（使用rate=1.0的完整cache）
        chunk_key_cache = torch.load(
            f'{full_cache_path}/{example_id}_{chunk_id}_key.pt',
            weights_only=True
        ).to(input_device)
        chunk_value_cache = torch.load(
            f'{full_cache_path}/{example_id}_{chunk_id}_value.pt',
            weights_only=True
        ).to(input_device)

        doc_key_caches.append(chunk_key_cache)
        doc_value_caches.append(chunk_value_cache)
        doc_lengths.append(passage_len)
        past_len += passage_len

    print(f"Loaded {len(doc_key_caches)} document caches, total length: {past_len}")

    # 将所有doc caches拼接到past_key_values
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] = 0
        current_pos = 0

        for doc_idx in range(len(doc_key_caches)):
            doc_len = doc_lengths[doc_idx]
            past_key_values.key_cache[layer_idx].narrow(2, current_pos, doc_len).copy_(
                doc_key_caches[doc_idx][layer_idx]
            )
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, doc_len).copy_(
                doc_value_caches[doc_idx][layer_idx]
            )
            past_key_values.past_tokens[layer_idx] += doc_len
            current_pos += doc_len

    # 第二步：使用query做一次forward，获取每层的attention
    query_len = query_tensor.shape[0]
    cache_position = torch.arange(past_len, past_len + query_len, device=input_device)
    inputs = query_tensor.unsqueeze(0).to(input_device)

    print(f"\nForwarding query (length={query_len}) to compute layer-wise attention...")

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
        outputs = model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True,
            output_attentions=False  # sdpa模式不支持
        )

    # 第三步：对每层计算attention分数（使用K-based similarity）
    num_key_value_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    num_attention_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_attention_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads

    layer_wise_selections = []
    layer_wise_scores = []

    print(f"\nComputing attention scores for each layer...")

    for layer_idx in range(num_layers):
        # 获取该层的key cache（包括documents + query）
        key_cache_layer = past_key_values.key_cache[layer_idx][:, :, :past_len + query_len, :]

        # 使用query的最后一个token的K来计算与所有历史tokens的相似度
        # Shape: [batch, num_kv_heads, seq_len, head_dim]
        query_k = key_cache_layer[:, :, past_len:past_len + query_len, :]  # [1, num_kv_heads, query_len, head_dim]
        doc_k = key_cache_layer[:, :, :past_len, :]  # [1, num_kv_heads, past_len, head_dim]

        # 计算相似度：query_k @ doc_k^T
        # 取query的平均K来代表整个query的关注点
        query_k_avg = query_k.mean(dim=2, keepdim=True)  # [1, num_kv_heads, 1, head_dim]

        similarities = torch.matmul(query_k_avg, doc_k.transpose(2, 3))  # [1, num_kv_heads, 1, past_len]
        similarities = similarities / (head_dim ** 0.5)

        # 对每个kv head计算attention weights
        attention_weights = torch.nn.functional.softmax(similarities, dim=-1)  # [1, num_kv_heads, 1, past_len]

        # 平均所有heads的attention
        attention_avg = attention_weights[0, :, 0, :].mean(dim=0)  # [past_len]
        attention_scores = attention_avg.float().cpu().numpy()  # 转换为float32再转numpy

        # 根据attention分数选择top-k tokens
        k = int(past_len * ratio)
        top_k_indices = np.argsort(attention_scores)[-k:]
        top_k_indices = np.sort(top_k_indices)  # 保持顺序

        layer_wise_selections.append(top_k_indices)
        layer_wise_scores.append(attention_scores)

        actual_ratio = len(top_k_indices) / past_len

        if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx:2d}: selected {len(top_k_indices)}/{past_len} tokens "
                  f"({actual_ratio*100:.1f}%, target={ratio*100:.0f}%) "
                  f"| scores: min={attention_scores[top_k_indices].min():.6f}, "
                  f"max={attention_scores[top_k_indices].max():.6f}, "
                  f"mean={attention_scores[top_k_indices].mean():.6f}")

    # 统计所有层的平均重算比例
    all_ratios = [len(sel) / past_len for sel in layer_wise_selections]
    print(f"\nRecomputation ratio statistics:")
    print(f"  Target ratio: {ratio*100:.0f}%")
    print(f"  Actual ratio: min={min(all_ratios)*100:.1f}%, max={max(all_ratios)*100:.1f}%, "
          f"mean={np.mean(all_ratios)*100:.1f}%, std={np.std(all_ratios)*100:.1f}%")
    print(f"{'='*100}\n")

    return layer_wise_selections, layer_wise_scores


def save_layer_wise_selection_cache(
    model,
    passages,
    chunk_ids,
    layer_wise_selections,
    example_id,
    full_cache_path,
    save_path,
    device="cuda:0"
):
    """
    保存每层选择后的KV cache

    Args:
        model: 模型
        passages: 文档列表
        chunk_ids: chunk IDs
        layer_wise_selections: 每层选择的token indices
        example_id: 样本ID
        full_cache_path: 完整KV cache的路径
        save_path: 保存路径
        device: 设备
    """
    os.makedirs(save_path, exist_ok=True)

    num_layers = len(model.model.layers)

    print(f"\n{'='*100}")
    print(f"Saving layer-wise selection caches to {save_path}")
    print(f"{'='*100}")

    # 对每个chunk，保存每层选择后的KV cache
    system_len = passages[0].shape[0]
    doc_start_pos = system_len

    for doc_idx, passage in enumerate(passages):
        chunk_id = chunk_ids[doc_idx]
        passage_len = passage.shape[0]

        # 加载该chunk的完整KV cache
        chunk_key_cache = torch.load(
            f'{full_cache_path}/{example_id}_{chunk_id}_key.pt',
            weights_only=True
        ).to(device)
        chunk_value_cache = torch.load(
            f'{full_cache_path}/{example_id}_{chunk_id}_value.pt',
            weights_only=True
        ).to(device)

        # 为每层创建选择后的cache
        selected_key_cache = []
        selected_value_cache = []

        for layer_idx in range(num_layers):
            # 获取该层选择的indices
            selected_indices = layer_wise_selections[layer_idx]

            # 筛选出属于当前document的indices
            doc_end_pos = doc_start_pos + passage_len
            doc_mask = (selected_indices >= doc_start_pos) & (selected_indices < doc_end_pos)
            doc_selected_indices = selected_indices[doc_mask] - doc_start_pos

            # 如果该层没有选择该document的任何token，至少保留第一个和最后一个
            if len(doc_selected_indices) == 0:
                doc_selected_indices = np.array([0, passage_len - 1])

            # 从完整的chunk cache中提取选中的KV
            selected_key = chunk_key_cache[layer_idx][:, :, doc_selected_indices, :]
            selected_value = chunk_value_cache[layer_idx][:, :, doc_selected_indices, :]

            selected_key_cache.append(selected_key.cpu())
            selected_value_cache.append(selected_value.cpu())

        # 保存每层选择的原始indices（相对于当前chunk）
        selected_indices_per_layer = []
        for layer_idx in range(num_layers):
            selected_indices = layer_wise_selections[layer_idx]
            doc_end_pos = doc_start_pos + passage_len
            doc_mask = (selected_indices >= doc_start_pos) & (selected_indices < doc_end_pos)
            doc_selected_indices = selected_indices[doc_mask] - doc_start_pos

            if len(doc_selected_indices) == 0:
                doc_selected_indices = np.array([0, passage_len - 1])

            selected_indices_per_layer.append(doc_selected_indices)

        # 保存
        torch.save(
            selected_key_cache,
            f'{save_path}/{example_id}_{chunk_id}_key.pt'
        )
        torch.save(
            selected_value_cache,
            f'{save_path}/{example_id}_{chunk_id}_value.pt'
        )
        torch.save(
            selected_indices_per_layer,  # 保存原始indices
            f'{save_path}/{example_id}_{chunk_id}_indices.pt'
        )

        # 统计信息
        avg_selected = np.mean([len(indices) for indices in selected_indices_per_layer])

        print(f"  Chunk {chunk_id}: original_len={passage_len}, "
              f"avg_selected={avg_selected:.1f} ({avg_selected/passage_len*100:.1f}%)")

        doc_start_pos = doc_end_pos

    print(f"{'='*100}\n")


def generate_with_layer_wise_cache(
    model,
    tokenizer,
    past_key_values,
    passages,
    load_path,
    example_id,
    chunk_ids,
    max_new_tokens=100,
    device="cuda:0",
    device_map=None
):
    """
    使用layer-wise selection的cache生成答案

    与原版generate_with_kvcache的区别：
    - 加载的cache每层可能有不同的长度
    - 需要对每层分别处理
    """
    from ktransformers.util.utils import rotate_half

    input_device = "cuda:0" if device_map is not None else device
    num_layers = len(model.model.layers)

    # 重置cache
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] = 0

    # 加载每个document的layer-wise cache
    print(f"\n{'='*100}")
    print(f"Loading layer-wise selection caches from {load_path}")
    print(f"{'='*100}")

    # 追踪所有原始token的最大位置
    max_original_position = 0
    chunk_start_pos = 0  # 追踪每个chunk在原始序列中的起始位置

    for doc_idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[doc_idx]
        chunk_len = passage.shape[0]

        # 加载该chunk的layer-wise cache和indices
        chunk_key_cache = torch.load(
            f'{load_path}/{example_id}_{chunk_id}_key.pt',
            weights_only=True
        )
        chunk_value_cache = torch.load(
            f'{load_path}/{example_id}_{chunk_id}_value.pt',
            weights_only=True
        )
        chunk_indices = torch.load(
            f'{load_path}/{example_id}_{chunk_id}_indices.pt',
            weights_only=False  # indices是numpy array，不能用weights_only=True
        )

        # 对每层分别处理
        for layer_idx in range(num_layers):
            layer_key = chunk_key_cache[layer_idx].to(input_device)
            layer_value = chunk_value_cache[layer_idx].to(input_device)

            layer_len = layer_key.shape[2]
            current_pos = past_key_values.past_tokens[layer_idx]

            # 复制到cache
            past_key_values.key_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len

            # 更新最大原始位置
            if len(chunk_indices[layer_idx]) > 0:
                layer_max_pos = chunk_start_pos + int(chunk_indices[layer_idx].max())
                max_original_position = max(max_original_position, layer_max_pos)

        # 统计每层的长度
        layer_lens = [chunk_key_cache[i].shape[2] for i in range(num_layers)]
        print(f"  Chunk {chunk_id}: layer_lens min={min(layer_lens)}, max={max(layer_lens)}, "
              f"mean={np.mean(layer_lens):.1f}, std={np.std(layer_lens):.1f}")

        chunk_start_pos += chunk_len

    # 输出每层的总长度和重算比例
    print(f"\nPer-layer cache statistics:")
    print(f"{'Layer':<8} {'Cached tokens':<15} {'Recompute ratio':<20} {'vs original':<15}")
    print(f"{'-'*60}")

    # 计算原始总长度（所有documents）
    original_total_len = chunk_start_pos

    for layer_idx in range(num_layers):
        cached_len = past_key_values.past_tokens[layer_idx]
        recompute_ratio = cached_len / original_total_len * 100

        if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
            print(f"Layer {layer_idx:<2d}  {cached_len:<15d} {recompute_ratio:>6.1f}%              {original_total_len}")

    avg_cached = np.mean([past_key_values.past_tokens[i] for i in range(num_layers)])
    avg_ratio = avg_cached / original_total_len * 100

    print(f"{'-'*60}")
    print(f"Average: {avg_cached:<15.1f} {avg_ratio:>6.1f}%              {original_total_len}")
    print(f"\nMax original position: {max_original_position}")
    print(f"{'='*100}\n")

    # 获取query tokens
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
    if query_prefix_len >= len(passages[-1]):
        query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0])) + 1

    query_tokens = passages[-1][query_prefix_len:]
    query_len = query_tokens.shape[0]

    print(f"Query length: {query_len} tokens")
    print(f"Generating up to {max_new_tokens} tokens...\n")

    # Prefill query
    inputs = query_tokens.unsqueeze(0).to(input_device)
    inputs_embeds = model.model.embed_tokens(inputs).to(input_device)

    # 关键修改：使用原始序列的逻辑位置，而不是物理cache长度
    # query紧接在documents后面，所以从max_original_position+1开始
    logical_position_start = max_original_position + 1
    cache_position = torch.arange(logical_position_start, logical_position_start + query_len, device=input_device)

    print(f"Cache position for query: {cache_position[0].item()} - {cache_position[-1].item()}")

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True
        )
        logits = outputs.logits[0, -1, :]

    # 更新每层的past_tokens
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] += query_len

    # 更新逻辑位置
    current_logical_position = logical_position_start + query_len

    # Autoregressive generation
    generated_tokens = []
    eos_tokens = [tokenizer.eos_token_id]
    if hasattr(tokenizer, 'im_end_id'):
        eos_tokens.append(tokenizer.im_end_id)
    im_end_token = tokenizer.encode('<|im_end|>', add_special_tokens=False)
    if len(im_end_token) == 1:
        eos_tokens.append(im_end_token[0])

    for step in range(max_new_tokens):
        next_token = torch.argmax(logits).item()
        generated_tokens.append(next_token)

        if (step + 1) % 10 == 0 or step < 10:
            token_text = tokenizer.decode([next_token])
            print(f"  Step {step + 1}: token_id={next_token}, text='{token_text}'")

        if next_token in eos_tokens:
            print(f"\nStopped: EOS token (id={next_token})")
            break

        # 继续生成，使用逻辑位置而不是物理past_len
        cache_position = torch.tensor([current_logical_position], device=input_device)
        input_token = torch.tensor([[next_token]], dtype=torch.long, device=input_device)

        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(input_token).to(input_device)
            outputs = model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True
            )
            logits = outputs.logits[0, 0, :]

        # 更新物理和逻辑位置
        for layer_idx in range(num_layers):
            past_key_values.past_tokens[layer_idx] += 1
        current_logical_position += 1

    print(f"Total generated: {len(generated_tokens)} tokens")
    print(f"Generated tokens: {generated_tokens}\n")

    return generated_tokens, query_len


def analyze_layer_wise_selection_overlap(
    layer_wise_selections,
    save_path=None
):
    """
    分析不同层之间选择的重叠度

    Args:
        layer_wise_selections: List[np.ndarray], 每层选择的indices
        save_path: 保存分析结果的路径
    """
    num_layers = len(layer_wise_selections)

    print(f"\n{'='*100}")
    print("Analyzing layer-wise selection overlap")
    print(f"{'='*100}")

    # 计算每对相邻层之间的重叠
    overlap_stats = []

    for layer_idx in range(num_layers - 1):
        curr_selection = set(layer_wise_selections[layer_idx])
        next_selection = set(layer_wise_selections[layer_idx + 1])

        intersection = curr_selection & next_selection
        union = curr_selection | next_selection

        overlap_ratio = len(intersection) / len(union) if len(union) > 0 else 0

        overlap_stats.append({
            'layer_pair': f"{layer_idx}-{layer_idx+1}",
            'overlap_count': len(intersection),
            'overlap_ratio': overlap_ratio,
            'curr_size': len(curr_selection),
            'next_size': len(next_selection)
        })

    # 输出统计信息
    avg_overlap = np.mean([s['overlap_ratio'] for s in overlap_stats])
    print(f"Average overlap between adjacent layers: {avg_overlap:.2%}\n")

    # 显示几个代表性的层
    print("Sample layer pairs:")
    for i in [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-2]:
        if i < len(overlap_stats):
            s = overlap_stats[i]
            print(f"  Layers {s['layer_pair']}: "
                  f"{s['overlap_count']}/{s['curr_size']} tokens overlap "
                  f"({s['overlap_ratio']:.2%})")

    # 计算所有层的交集（所有层都选择的tokens）
    common_tokens = set(layer_wise_selections[0])
    for layer_idx in range(1, num_layers):
        common_tokens &= set(layer_wise_selections[layer_idx])

    print(f"\nTokens selected by ALL layers: {len(common_tokens)}")
    print(f"Percentage: {len(common_tokens) / len(layer_wise_selections[0]) * 100:.1f}%")

    # 计算每个token被多少层选择
    all_tokens = set()
    for selection in layer_wise_selections:
        all_tokens |= set(selection)

    token_selection_counts = {}
    for token_idx in all_tokens:
        count = sum(1 for selection in layer_wise_selections if token_idx in selection)
        token_selection_counts[token_idx] = count

    # 统计分布
    count_distribution = {}
    for count in token_selection_counts.values():
        count_distribution[count] = count_distribution.get(count, 0) + 1

    print(f"\nDistribution of how many layers select each token:")
    for count in sorted(count_distribution.keys(), reverse=True)[:10]:
        print(f"  Selected by {count:2d} layers: {count_distribution[count]:4d} tokens "
              f"({count_distribution[count] / len(all_tokens) * 100:.1f}%)")

    print(f"{'='*100}\n")

    # 保存结果
    if save_path:
        analysis_result = {
            'overlap_stats': overlap_stats,
            'avg_overlap': avg_overlap,
            'common_tokens': len(common_tokens),
            'token_selection_counts': {int(k): int(v) for k, v in token_selection_counts.items()},
            'count_distribution': {int(k): int(v) for k, v in count_distribution.items()}
        }

        with open(save_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)

        print(f"Analysis saved to {save_path}\n")


def main(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    max_cache_len=32768,
    topk=10,
    preprocess=True,
    preprocess_scope=PreprocessScope.GLOBAL,
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    device="cuda:0",
    use_multi_gpu=False,
    example_idx=4,
    sub_question_idx=1,
    ratio=0.3,
    max_new_tokens=100,
    output_dir='./kvcache_layerwise_analysis'
):
    """
    主函数：实现layer-wise token selection并对比uniform selection
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading tokenizer and config from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    print(f"Loading {model_type} model...")
    model, device_map = load_model(model_type, model_path, config, device, use_multi_gpu)

    # Prepare data
    print("Preparing data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, model_type, topk,
        max_main_questions=example_idx + 1,
        preprocess=preprocess,
        preprocess_scope=preprocess_scope
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print("\n" + "="*100)
    print(f"ANALYZING EXAMPLE {example_idx}, SUB-QUESTION {sub_question_idx}")
    print("="*100)
    print(f"Main Question: {q_data['main_question']}")
    print(f"Sub Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print("="*100)

    # Build query
    if model_type == 'qwen3':
        question_text = f"<|im_end|>\n<|im_start|>user\n/no_think\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    else:
        question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "

    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    # Get documents
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    kv_chunk_ids = [0] + doc_chunk_ids

    input_device = "cuda:0" if use_multi_gpu else device

    # Initialize cache
    cache_device = device_map if use_multi_gpu else device
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=cache_device,
        dtype=model.dtype,
        passage_len=32768
    )

    # ================================================================================
    # STEP 1: 计算layer-wise attention selection
    # ================================================================================
    # 完整KV cache路径（rate=1.0）
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    layer_wise_selections, layer_wise_scores = compute_layer_wise_attention_selection(
        model, tokenizer, past_key_values,
        iter_tokens[:-1],  # documents only (不包括query)
        kv_chunk_ids,
        question_tensor,
        example_idx,
        full_cache_path,
        ratio=ratio,
        device=input_device,
        device_map=device_map
    )

    # ================================================================================
    # STEP 2: 分析layer-wise selection的重叠度
    # ================================================================================
    analyze_layer_wise_selection_overlap(
        layer_wise_selections,
        save_path=os.path.join(output_dir, 'layer_overlap_analysis.json')
    )

    # ================================================================================
    # STEP 3: 保存layer-wise selection cache
    # ================================================================================
    layerwise_cache_path = os.path.join(cache_path, model_name, 'layerwise_kv_cache')
    save_layer_wise_selection_cache(
        model,
        iter_tokens[:-1],  # documents only
        kv_chunk_ids,
        layer_wise_selections,
        example_idx,
        full_cache_path,
        layerwise_cache_path,
        device=input_device
    )

    # ================================================================================
    # STEP 4: 使用layer-wise cache生成答案
    # ================================================================================
    print("\n" + "="*100)
    print("GENERATING with layer-wise selection cache")
    print("="*100)

    # 重置cache
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    answer_tokens_layerwise, query_len = generate_with_layer_wise_cache(
        model, tokenizer, past_key_values,
        iter_tokens,
        layerwise_cache_path,
        example_idx,
        kv_chunk_ids,
        max_new_tokens=max_new_tokens,
        device=input_device,
        device_map=device_map
    )

    answer_text_layerwise = tokenizer.decode(answer_tokens_layerwise, skip_special_tokens=True)

    print("\n" + "="*100)
    print("RESULTS")
    print("="*100)
    print(f"Layer-wise answer: {answer_text_layerwise}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print("="*100)

    # 保存结果
    summary = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'main_question': q_data['main_question'],
        'sub_question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'answer_layerwise': answer_text_layerwise,
        'ratio': ratio,
        'num_layers': len(layer_wise_selections),
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        topk=10,
        preprocess=True,
        preprocess_scope=PreprocessScope.GLOBAL,
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        device="cuda:0",
        use_multi_gpu=False,
        example_idx=4,
        sub_question_idx=1,
        ratio=0.3,
        max_new_tokens=100,
        output_dir='./kvcache_layerwise_analysis'
    )

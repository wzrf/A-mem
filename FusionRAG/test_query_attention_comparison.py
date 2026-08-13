#!/usr/bin/env python3
"""
对比测试：比较 per_head_generation.py 和 test_fusionrag_reflect.py 中 QueryAttention 的实现

目标：找出为什么两者产生不同结果
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from transformers import AutoTokenizer, AutoConfig
from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import rotate_half


def compute_query_attention_per_head_style(model, past_key_values, query_tensor, total_cache_len, prefix_len, doc_len, device):
    """
    per_head_generation.py 风格的 attention 计算
    - 对完整 cache (prefix + doc) 计算 attention
    - softmax 后提取 doc 部分
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(query_tensor)
        hidden_states = inputs_embeds

        layer_attention_scores = {}

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = past_key_values.key_cache[layer_idx][:, :, :total_cache_len, :]

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)

            # 对完整 cache 计算 attention
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)

            # 提取 doc 部分
            doc_attn = attn_weights[0, :, :, prefix_len:prefix_len + doc_len]
            # 对 query tokens 和 heads 平均
            doc_attn_avg = doc_attn.mean(dim=(0, 1)).cpu().float().numpy()

            layer_attention_scores[layer_idx] = doc_attn_avg

            # 继续前向传播
            value_states = past_key_values.value_cache[layer_idx][:, :, :total_cache_len, :]
            value_states = value_states.repeat_interleave(n_rep, dim=1)

            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

    return layer_attention_scores


def compute_query_attention_modeling_style(model, past_key_values, query_tensor, passages_len, device):
    """
    modeling_qwen2.py 风格的 attention 计算（通过 importance_cache）
    - 使用模型前向传播
    - 通过 importance_cache 存储 attention
    """
    import math

    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    system_len = passages_len[0]
    doc_len = sum(passages_len[1:-1])
    query_len = passages_len[-1]
    total_context_len = system_len + doc_len
    past_len = sum(passages_len[:-1])

    # 初始化 importance_cache
    for layer_idx in range(num_layers):
        past_key_values.importance_cache[layer_idx].zero_()

    cache_position = torch.arange(past_len, past_len + query_len, device=device)

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(query_tensor)
        hidden_states = inputs_embeds

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            query_states = layer.self_attn.q_proj(hidden_states)

            # 从 cache 获取 key_states
            key_states = past_key_values.key_cache[layer_idx][:, :, :past_len, :]
            value_states = past_key_values.value_cache[layer_idx][:, :, :past_len, :]

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

            # GQA: repeat key/value
            n_rep = num_heads // num_kv_heads
            key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
            value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)

            # 只对后 1/4 层计算 QueryAttention
            start_layer = num_layers * 3 // 4
            if layer_idx >= start_layer:
                # 关键：对完整上下文计算 attention
                context_key = key_states_expanded[:, :, :total_context_len, :]

                attn_weights = torch.matmul(query_states, context_key.transpose(-1, -2))
                attn_weights = attn_weights / math.sqrt(head_dim)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

                # 提取 doc 部分
                doc_attn = attn_weights[:, :, :, system_len:system_len + doc_len]
                doc_attn = doc_attn.mean(dim=2)  # 对 query 取平均

                # 存储到 importance_cache
                # importance_cache shape: [num_attention_heads, passage_len]
                # doc_attn shape: [1, num_heads, doc_len]
                past_key_values.importance_cache[layer_idx].narrow(1, system_len, doc_len).copy_(
                    doc_attn.squeeze(0).to(past_key_values.importance_cache[layer_idx].dtype)
                )

            # 继续前向传播
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states_expanded, value_states_expanded,
                attn_mask=None, dropout_p=0.0
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

    # 提取 importance_cache 中的结果
    layer_attention_scores = {}
    for layer_idx in range(num_layers):
        # importance_cache[layer_idx] shape: [num_kv_heads, passage_len]
        attn = past_key_values.importance_cache[layer_idx][:, system_len:system_len + doc_len]
        # 对 heads 平均
        attn_avg = attn.mean(dim=0).cpu().float().numpy()
        layer_attention_scores[layer_idx] = attn_avg

    return layer_attention_scores


def main():
    print("=" * 100)
    print("QueryAttention Implementation Comparison Test")
    print("=" * 100)

    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    cache_path = '/mnt/data/reflect/'
    model_name = 'Qwen2.5-7B-Instruct'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    device = 'cuda:0'

    example_idx = 4
    sub_question_idx = 1

    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, device_map = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load data
    print("\nLoading data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen', 10,
        max_main_questions=example_idx + 1,
        preprocess=False
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")

    # Build query tensor
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Prepare cache
    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    # Initialize cache
    max_cache_len = 32768
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Load KV cache
    print("\nLoading KV cache...")
    passages_len = []
    prefix_len = 0

    for chunk_id in doc_chunk_ids:
        chunk_key = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt', weights_only=True)
        chunk_value = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt', weights_only=True)

        cache_len = chunk_key[0].shape[2]
        passages_len.append(cache_len)

        for layer_idx in range(model.config.num_hidden_layers):
            current_pos = past_key_values.past_tokens[layer_idx]
            past_key_values.key_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(chunk_key[layer_idx].to(device))
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(chunk_value[layer_idx].to(device))
            past_key_values.past_tokens[layer_idx] += cache_len

        if chunk_id == 0:
            prefix_len = cache_len

        print(f"  Loaded chunk {chunk_id}: {cache_len} tokens")

    # Add query length to passages_len
    passages_len.append(len(question_tokens))

    total_cache_len = sum(passages_len[:-1])
    doc_len = sum(passages_len[1:-1])

    print(f"\nPrefix (system): {prefix_len} tokens")
    print(f"Document: {doc_len} tokens")
    print(f"Query: {len(question_tokens)} tokens")
    print(f"Total cache: {total_cache_len} tokens")
    print(f"passages_len: {passages_len}")

    # ========================================================================
    # Method 1: per_head_generation.py style
    # ========================================================================
    print("\n" + "=" * 100)
    print("Method 1: per_head_generation.py style")
    print("=" * 100)

    attn_scores_1 = compute_query_attention_per_head_style(
        model, past_key_values, query_tensor,
        total_cache_len, prefix_len, doc_len, device
    )

    # ========================================================================
    # Method 2: modeling_qwen2.py style (通过 importance_cache)
    # ========================================================================
    print("\n" + "=" * 100)
    print("Method 2: modeling_qwen2.py style (via importance_cache)")
    print("=" * 100)

    attn_scores_2 = compute_query_attention_modeling_style(
        model, past_key_values, query_tensor,
        passages_len, device
    )

    # ========================================================================
    # Compare results
    # ========================================================================
    print("\n" + "=" * 100)
    print("Comparison Results")
    print("=" * 100)

    num_layers = model.config.num_hidden_layers
    start_layer = num_layers * 3 // 4

    print(f"\nComparing layers {start_layer} to {num_layers - 1} (后 1/4 层)")

    for layer_idx in range(start_layer, num_layers):
        scores1 = attn_scores_1[layer_idx]
        scores2 = attn_scores_2[layer_idx]

        # 计算相关性
        correlation = np.corrcoef(scores1, scores2)[0, 1]

        # 计算 top-k 重叠
        k = int(len(scores1) * 0.3)
        top_k_1 = set(np.argsort(scores1)[-k:])
        top_k_2 = set(np.argsort(scores2)[-k:])
        overlap = len(top_k_1 & top_k_2) / k

        print(f"\nLayer {layer_idx}:")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Top-30% overlap: {overlap*100:.1f}%")
        print(f"  Method 1 - mean: {scores1.mean():.6f}, std: {scores1.std():.6f}, max: {scores1.max():.6f}")
        print(f"  Method 2 - mean: {scores2.mean():.6f}, std: {scores2.std():.6f}, max: {scores2.max():.6f}")

    # 聚合后 1/4 层的结果并比较最终选择
    print("\n" + "=" * 100)
    print("Final Selection Comparison (聚合后 1/4 层)")
    print("=" * 100)

    layers_to_use = list(range(start_layer, num_layers))

    multi_layer_1 = np.stack([attn_scores_1[i] for i in layers_to_use]).mean(axis=0)
    multi_layer_2 = np.stack([attn_scores_2[i] for i in layers_to_use]).mean(axis=0)

    correlation = np.corrcoef(multi_layer_1, multi_layer_2)[0, 1]

    k = int(len(multi_layer_1) * 0.3)
    top_k_1 = set(np.argsort(multi_layer_1)[-k:])
    top_k_2 = set(np.argsort(multi_layer_2)[-k:])
    overlap = len(top_k_1 & top_k_2) / k

    print(f"\nAggregated attention (layers {start_layer}-{num_layers-1}):")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Top-30% overlap: {overlap*100:.1f}%")

    # 显示 top-10 positions
    top_10_1 = np.argsort(multi_layer_1)[-10:][::-1]
    top_10_2 = np.argsort(multi_layer_2)[-10:][::-1]

    print(f"\n  Method 1 top-10 positions: {top_10_1.tolist()}")
    print(f"  Method 2 top-10 positions: {top_10_2.tolist()}")

    print("\n" + "=" * 100)
    print("Test Complete")
    print("=" * 100)


if __name__ == '__main__':
    main()

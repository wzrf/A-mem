#!/usr/bin/env python3
"""
测试混合选择策略：
1. 使用 ratio=0.9 作为基础选择
2. 从 missing tokens (0.95 - 0.9) 中补充 30%
3. 测试能否恢复正确答案
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
sys.path.insert(0, '/mnt/data/wjh/FusionRAG/ktransformers')

from per_head_generation import compute_layerwise_token_selection, sparse_prefill_per_head, generate_with_sparse_prefill
from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache


def run_hybrid_test():
    device = "cuda:0"
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    cache_path = '/mnt/data/reflect/'
    model_name = 'Qwen2.5-7B-Instruct'
    example_idx = 4
    sub_question_idx = 1

    print("=" * 100)
    print("Hybrid Selection Test: ratio=0.3 + 30% of (0.95 - 0.9) missing tokens")
    print("=" * 100)

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

    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    passages = q_data['docs']

    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)

    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    # Step 1: Get selections for both ratios
    print("\n" + "=" * 80)
    print("Step 1: Computing selections for ratio=0.95, 0.9, and 0.3")
    print("=" * 80)

    selections = {}
    for ratio in [0.95, 0.9, 0.3]:
        layer_head_selections, layer_union_positions, doc_len, position_to_token = compute_layerwise_token_selection(
            model, tokenizer, passages, doc_chunk_ids, query_tensor,
            example_idx, full_cache_path,
            total_ratio=ratio,
            device=device,
            per_head_selection=True,
            selection_strategy="layer_wise",
            scoring_method="reconstruction"
        )
        selections[ratio] = {
            'layer_head_selections': layer_head_selections,
            'layer_union_positions': layer_union_positions,
            'doc_len': doc_len,
            'position_to_token': position_to_token
        }

    # Step 2: Compute missing tokens per layer
    print("\n" + "=" * 80)
    print("Step 2: Computing missing tokens and adding 30%")
    print("=" * 80)

    doc_len = selections[0.95]['doc_len']
    position_to_token = selections[0.95]['position_to_token']
    num_layers = 28
    num_kv_heads = 4

    # 创建混合选择
    hybrid_layer_union = {}
    hybrid_layer_head_selections = {}

    for layer_idx in range(num_layers):
        pos_095 = set(selections[0.95]['layer_union_positions'].get(layer_idx, []))
        pos_090 = set(selections[0.9]['layer_union_positions'].get(layer_idx, []))
        pos_030 = set(selections[0.3]['layer_union_positions'].get(layer_idx, []))

        # missing = 0.95 选了但 0.9 没选的 tokens
        missing = pos_095 - pos_090

        # 补充 30% 的 missing tokens
        missing_list = sorted(list(missing))
        num_to_add = max(1, int(len(missing_list) * 0.3))

        # 按原始分数排序选择最重要的 30%
        # 这里简化处理：取前 30%
        tokens_to_add = missing_list[:num_to_add]

        # 合并选择: ratio=0.3 的基础 + missing 的 30%
        hybrid_positions = sorted(list(pos_030) + tokens_to_add)
        hybrid_layer_union[layer_idx] = hybrid_positions

        # 为所有 heads 设置相同的选择（layer_wise 策略）
        hybrid_layer_head_selections[layer_idx] = {}
        for kv_head_idx in range(num_kv_heads):
            hybrid_layer_head_selections[layer_idx][kv_head_idx] = {
                'positions': hybrid_positions.copy(),
                'count': len(hybrid_positions)
            }

        if layer_idx % 7 == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx:2d}: base(0.3)={len(pos_030):4d} + missing(0.95-0.9)={len(missing):3d} * 30% = {num_to_add:3d} => {len(hybrid_positions):4d} tokens ({len(hybrid_positions)/doc_len*100:.1f}%)")

    # Step 3: Run generation with hybrid selection
    print("\n" + "=" * 80)
    print("Step 3: Loading FULL cache and running sparse prefill")
    print("=" * 80)

    # Load full cache
    max_cache_len = 32768
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Cache file naming: {example_idx}_{chunk_id}_key.pt (contains all layers)
    prefix_len = 0

    for chunk_idx, chunk_id in enumerate(doc_chunk_ids):
        key_path = os.path.join(full_cache_path, f'{example_idx}_{chunk_id}_key.pt')
        value_path = os.path.join(full_cache_path, f'{example_idx}_{chunk_id}_value.pt')

        chunk_key_cache = torch.load(key_path, map_location=device, weights_only=True)
        chunk_value_cache = torch.load(value_path, map_location=device, weights_only=True)

        # Copy to past_key_values for each layer
        for layer_idx in range(num_layers):
            layer_key = chunk_key_cache[layer_idx].to(device)
            layer_value = chunk_value_cache[layer_idx].to(device)

            layer_len = layer_key.shape[2]
            current_pos = past_key_values.past_tokens[layer_idx]

            past_key_values.key_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len

        if chunk_idx == 0:
            prefix_len = layer_len
            print(f"  Loaded prefix (chunk {chunk_id}): {layer_len} tokens")
        else:
            print(f"  Loaded chunk {chunk_id}: {layer_len} tokens")

    total_cache_len = past_key_values.past_tokens[0]
    print(f"  Document length: {doc_len}")
    print(f"  Total cache length: {total_cache_len}")

    # Run sparse prefill
    print("\n" + "=" * 80)
    print("Step 4: Sparse Prefill with Hybrid Selection")
    print("=" * 80)

    layer_recompute_stats, overall_ratio = sparse_prefill_per_head(
        model,
        past_key_values,
        hybrid_layer_head_selections,
        hybrid_layer_union,
        position_to_token,
        query_tensor,
        prefix_len,
        doc_len,
        device
    )

    # 重置 past_tokens（不包含 query，因为 query 会重新 forward）
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] = prefix_len + doc_len

    # Generate
    print("\n" + "=" * 80)
    print("Step 5: Generating Answer")
    print("=" * 80)

    generated_text = generate_with_sparse_prefill(
        model,
        tokenizer,
        past_key_values,
        query_tensor,
        prefix_len,
        doc_len,
        max_new_tokens=100,
        device=device
    )

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"\nGround Truth:\n{sub_q_info['answer']}")
    print(f"\nGenerated Answer:\n{generated_text}")
    print("=" * 100)

    # 计算实际的 ratio
    total_selected = sum(len(v) for v in hybrid_layer_union.values())
    actual_ratio = total_selected / (num_layers * doc_len)
    print(f"\nActual selection ratio: {actual_ratio*100:.2f}%")

    return generated_text


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    run_hybrid_test()

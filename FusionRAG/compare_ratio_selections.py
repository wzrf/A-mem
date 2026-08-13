#!/usr/bin/env python3
"""
对比不同 ratio 下的 token 选择差异
分析 ratio=0.95 vs ratio=0.9 哪些 token 被少选了
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# 添加项目路径
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
sys.path.insert(0, '/mnt/data/wjh/FusionRAG/ktransformers')

from per_head_generation import compute_layerwise_token_selection
from test_fusionrag_reflect import load_model, prepare_reflect_data

def run_comparison():
    device = "cuda:0"
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    cache_path = '/mnt/data/reflect/'
    model_name = 'Qwen2.5-7B-Instruct'
    example_idx = 4
    sub_question_idx = 1

    print("=" * 100)
    print("Token Selection Comparison: ratio=0.95 vs ratio=0.9")
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

    # Get data
    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    passages = q_data['docs']

    # Build query tensor
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)

    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    results = {}

    for ratio in [0.95, 0.9]:
        print(f"\n{'=' * 80}")
        print(f"Computing selections for ratio={ratio}")
        print(f"{'=' * 80}")

        layer_head_selections, layer_union_positions, doc_len, position_to_token = compute_layerwise_token_selection(
            model, tokenizer, passages, doc_chunk_ids, query_tensor,
            example_idx, full_cache_path,
            total_ratio=ratio,
            device=device,
            per_head_selection=True,
            selection_strategy="layer_wise",
            scoring_method="reconstruction"
        )

        results[ratio] = {
            'layer_union_positions': {k: set(v) for k, v in layer_union_positions.items()},
            'doc_len': doc_len,
            'position_to_token': position_to_token
        }

    # Compare selections
    print("\n" + "=" * 100)
    print("COMPARISON ANALYSIS")
    print("=" * 100)

    doc_len = results[0.95]['doc_len']
    position_to_token = results[0.95]['position_to_token']

    # 分析每层的差异
    print("\nPer-layer difference (tokens in 0.95 but not in 0.9):")
    print("-" * 80)

    all_missing_positions = set()
    layer_missing = {}

    for layer_idx in range(28):
        pos_095 = results[0.95]['layer_union_positions'].get(layer_idx, set())
        pos_090 = results[0.9]['layer_union_positions'].get(layer_idx, set())

        missing = pos_095 - pos_090
        layer_missing[layer_idx] = missing
        all_missing_positions.update(missing)

        if len(missing) > 0:
            print(f"Layer {layer_idx:2d}: {len(pos_095):4d} -> {len(pos_090):4d} (missing {len(missing):4d} tokens)")

    print(f"\n总共有 {len(all_missing_positions)} 个位置在 ratio=0.95 中选中但在 ratio=0.9 中未选中")

    # 分析 missing tokens 的内容
    print("\n" + "=" * 100)
    print("MISSING TOKENS CONTENT ANALYSIS")
    print("=" * 100)

    # 按位置排序显示 missing tokens
    sorted_missing = sorted(all_missing_positions)

    print(f"\nMissing token positions and content (first 50):")
    print("-" * 80)

    for i, pos in enumerate(sorted_missing[:50]):
        token_id = position_to_token.get(pos, None)
        if token_id is not None:
            token_text = tokenizer.decode([token_id])
            print(f"  Position {pos:4d}: token_id={token_id:6d}, text='{token_text}'")

    if len(sorted_missing) > 50:
        print(f"  ... and {len(sorted_missing) - 50} more positions")

    # 找出包含关键信息的 tokens
    print("\n" + "=" * 100)
    print("SEARCHING FOR CRITICAL TOKENS IN MISSING SET")
    print("=" * 100)

    # 关键词：1216, 1220, Gloucester, Westminster, crowned, coronation
    keywords = ['1216', '1220', 'Gloucester', 'Westminster', 'crowned', 'coronation', 'May', '17']

    for keyword in keywords:
        keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
        print(f"\nKeyword '{keyword}' -> token_ids: {keyword_tokens}")

        for pos in sorted_missing:
            token_id = position_to_token.get(pos, None)
            if token_id in keyword_tokens:
                token_text = tokenizer.decode([token_id])
                print(f"  Found at position {pos}: '{token_text}'")

    # 保存结果
    save_data = {
        'ratio_095_count': {str(k): len(v) for k, v in results[0.95]['layer_union_positions'].items()},
        'ratio_090_count': {str(k): len(v) for k, v in results[0.9]['layer_union_positions'].items()},
        'missing_positions': sorted_missing,
        'doc_len': doc_len
    }

    with open('./ratio_comparison_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to ./ratio_comparison_results.json")

    return results, all_missing_positions, position_to_token


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    run_comparison()

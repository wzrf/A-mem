#!/usr/bin/env python3
"""
分析不同 ratio 下选到的 token 位置，确认是否包含关键信息
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
sys.path.insert(0, '/mnt/data/wjh/FusionRAG/ktransformers')

from per_head_generation import compute_layerwise_token_selection
from test_fusionrag_reflect import load_model, prepare_reflect_data


def analyze():
    device = "cuda:0"
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    cache_path = '/mnt/data/reflect/'
    model_name = 'Qwen2.5-7B-Instruct'
    example_idx = 4
    sub_question_idx = 1

    print("=" * 100)
    print("Token Position Analysis")
    print("=" * 100)

    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, _ = load_model('qwen', model_path, config, device, use_multi_gpu=False)
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

    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    passages = q_data['docs']

    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)

    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    # 计算三种 ratio 的选择
    print("\n" + "=" * 80)
    print("Computing selections for ratio=0.95, 0.9, 0.3")
    print("=" * 80)

    selections = {}
    for ratio in [0.95, 0.9, 0.3]:
        print(f"\n--- Computing ratio={ratio} ---")
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
            'layer_union_positions': layer_union_positions,
            'doc_len': doc_len,
            'position_to_token': position_to_token
        }

    doc_len = selections[0.95]['doc_len']
    position_to_token = selections[0.95]['position_to_token']

    # 找出关键 tokens 的位置
    print("\n" + "=" * 100)
    print("FINDING KEY TOKENS IN DOCUMENT")
    print("=" * 100)

    # 关键词
    keywords = ['1216', '1220', 'Gloucester', 'Westminster', 'crowned', 'coronation']

    keyword_positions = {}
    for keyword in keywords:
        keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
        print(f"\nKeyword '{keyword}' -> token_ids: {keyword_tokens}")

        # 找到这些 token 在文档中的位置
        positions = []
        for pos, token_id in position_to_token.items():
            if token_id in keyword_tokens:
                token_text = tokenizer.decode([token_id])
                positions.append((pos, token_id, token_text))

        keyword_positions[keyword] = positions
        for pos, tid, txt in sorted(positions):
            print(f"  Position {pos:4d}: token_id={tid:6d}, text='{txt}'")

    # 分析每种 ratio 是否选到了关键 tokens
    print("\n" + "=" * 100)
    print("CHECKING IF KEY TOKENS ARE SELECTED")
    print("=" * 100)

    # 使用 Layer 0 作为代表（layer_wise 策略下所有层选择相同）
    layer_idx = 0

    pos_095 = set(selections[0.95]['layer_union_positions'].get(layer_idx, []))
    pos_090 = set(selections[0.9]['layer_union_positions'].get(layer_idx, []))
    pos_030 = set(selections[0.3]['layer_union_positions'].get(layer_idx, []))
    missing = pos_095 - pos_090  # 0.95 选了但 0.9 没选

    print(f"\nLayer {layer_idx} selection sizes:")
    print(f"  ratio=0.95: {len(pos_095)} tokens")
    print(f"  ratio=0.9:  {len(pos_090)} tokens")
    print(f"  ratio=0.3:  {len(pos_030)} tokens")
    print(f"  missing (0.95-0.9): {len(missing)} tokens")

    for keyword in keywords:
        positions = keyword_positions[keyword]
        if not positions:
            continue

        print(f"\n--- Keyword: '{keyword}' ---")
        for pos, tid, txt in sorted(positions):
            in_095 = pos in pos_095
            in_090 = pos in pos_090
            in_030 = pos in pos_030
            in_missing = pos in missing

            status = []
            if in_030:
                status.append("0.3✓")
            else:
                status.append("0.3✗")
            if in_090:
                status.append("0.9✓")
            else:
                status.append("0.9✗")
            if in_095:
                status.append("0.95✓")
            else:
                status.append("0.95✗")
            if in_missing:
                status.append("MISSING✓")

            print(f"  Position {pos:4d} '{txt}': {' | '.join(status)}")

    # 详细分析 1216 和 1220
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS: 1216 vs 1220")
    print("=" * 100)

    for year in ['1216', '1220']:
        print(f"\n=== {year} ===")
        positions = keyword_positions.get(year, [])

        all_in_030 = all(pos in pos_030 for pos, _, _ in positions)
        all_in_090 = all(pos in pos_090 for pos, _, _ in positions)
        all_in_095 = all(pos in pos_095 for pos, _, _ in positions)
        any_in_missing = any(pos in missing for pos, _, _ in positions)

        print(f"  All tokens in ratio=0.3:  {all_in_030}")
        print(f"  All tokens in ratio=0.9:  {all_in_090}")
        print(f"  All tokens in ratio=0.95: {all_in_095}")
        print(f"  Any token in missing (0.95-0.9): {any_in_missing}")

        for pos, tid, txt in sorted(positions):
            print(f"    Position {pos}: '{txt}' -> 0.3:{pos in pos_030} | 0.9:{pos in pos_090} | 0.95:{pos in pos_095} | missing:{pos in missing}")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    analyze()

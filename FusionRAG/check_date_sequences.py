#!/usr/bin/env python3
"""
检查per-head selection是否保留了date sequences
"""

import json
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import prepare_reflect_data, PreprocessScope


def check_date_sequence_in_selection(all_tokens, selected_indices, tokenizer):
    """
    检查selected tokens中是否包含连续的日期序列

    返回找到的日期模式
    """
    # 目标序列: "1216" = ['1', '2', '1', '6']
    # 目标序列: "1220" = ['1', '2', '2', '0']

    target_1216 = ['1', '2', '1', '6']
    target_1220 = ['1', '2', '2', '0']

    # 将selected_indices转换为set方便查找
    selected_set = set(selected_indices)

    # 检查所有tokens中的日期序列
    found_dates = []

    for i in range(len(all_tokens) - 3):
        # 获取连续4个tokens
        seq_tokens = [all_tokens[i+j] for j in range(4)]
        seq_text = [tokenizer.decode([t]).strip() for t in seq_tokens]

        # 检查是否匹配日期模式
        if seq_text == target_1216:
            # 检查这4个token是否都被选中
            positions = [i+j for j in range(4)]
            selected_count = sum(1 for pos in positions if pos in selected_set)
            found_dates.append({
                'date': '1216',
                'positions': positions,
                'selected_count': selected_count,
                'fully_selected': selected_count == 4,
                'tokens': seq_tokens
            })
        elif seq_text == target_1220:
            positions = [i+j for j in range(4)]
            selected_count = sum(1 for pos in positions if pos in selected_set)
            found_dates.append({
                'date': '1220',
                'positions': positions,
                'selected_count': selected_count,
                'fully_selected': selected_count == 4,
                'tokens': seq_tokens
            })

    return found_dates


def main():
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    cache_path = '/mnt/data/reflect/Qwen2.5-7B-Instruct/headwise_kv_cache'
    example_idx = 4
    sub_question_idx = 1

    print(f"\n{'='*100}")
    print("Checking if per-head selections preserved date sequences")
    print(f"{'='*100}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load data
    print("Loading data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, '/mnt/data/models/bge-m3-FP16', 'qwen', 10,
        max_main_questions=example_idx + 1,
        preprocess=False
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}\\n")

    # Get document tokens (skip chunks 0 and 1)
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']

    all_doc_tokens = []
    for chunk_id in doc_chunk_ids[2:]:  # Skip first two
        chunk_tokens = doc_tensors[chunk_id - 1].tolist()
        all_doc_tokens.extend(chunk_tokens)

    print(f"Total recomputable tokens: {len(all_doc_tokens)}")

    # First, find all date sequences in the original document
    print(f"\n{'='*80}")
    print("Date sequences in original document:")
    print(f"{'='*80}")

    all_dates = check_date_sequence_in_selection(all_doc_tokens, range(len(all_doc_tokens)), tokenizer)
    for date_info in all_dates:
        print(f"  {date_info['date']} at positions {date_info['positions']}")

    # Load per-head selections
    num_layers = 28
    num_kv_heads = 4
    focus_layers = [0, 5, 14, 20, 27]

    print(f"\n{'='*80}")
    print("Checking per-head selections:")
    print(f"{'='*80}\n")

    for layer_idx in focus_layers:
        print(f"{'#'*80}")
        print(f"LAYER {layer_idx}")
        print(f"{'#'*80}\n")

        for head_idx in range(num_kv_heads):
            # Load this head's selection indices
            # Note: We saved layer-wise selections, need to reconstruct
            # For now, let's load the saved JSON analysis
            pass

        # Instead, let's recompute the selections
        # Load document KV caches
        doc_key_caches = []
        for chunk_id in doc_chunk_ids[2:]:
            try:
                chunk_key_cache = torch.load(
                    f'/mnt/data/reflect/Qwen2.5-7B-Instruct/kv_cache/{example_idx}_{chunk_id}_key.pt',
                    weights_only=True
                ).to('cpu')
                doc_key_caches.append(chunk_key_cache)
            except FileNotFoundError:
                print(f"Warning: key cache for chunk {chunk_id} not found")
                continue

        # Concatenate
        if layer_idx < len(doc_key_caches[0]):
            layer_keys = [cache[layer_idx] for cache in doc_key_caches]
            all_keys = torch.cat(layer_keys, dim=2)

            # Compute per-head selections
            for head_idx in range(num_kv_heads):
                head_keys = all_keys[:, head_idx:head_idx+1, :, :]
                head_dim = head_keys.shape[-1]

                # Compute attention scores
                avg_key = head_keys.mean(dim=2, keepdim=True)
                similarities = torch.matmul(avg_key, head_keys.transpose(2, 3))
                similarities = similarities / (head_dim ** 0.5)
                attention_scores = torch.nn.functional.softmax(similarities, dim=-1)
                attention_scores = attention_scores[0, 0, 0, :].float().cpu().numpy()

                # Select top-30%
                k = int(len(all_doc_tokens) * 0.3)
                top_k_indices = np.argsort(attention_scores)[-k:]

                # Check date sequences
                date_results = check_date_sequence_in_selection(
                    all_doc_tokens, top_k_indices, tokenizer
                )

                print(f"  Head {head_idx}:")
                if date_results:
                    for date_info in date_results:
                        status = "✅ FULLY SELECTED" if date_info['fully_selected'] else f"⚠️  PARTIAL ({date_info['selected_count']}/4 tokens)"
                        print(f"    {date_info['date']}: {status}")
                        print(f"      Positions: {date_info['positions']}")
                else:
                    print(f"    ❌ No date sequences found")

        print()


if __name__ == '__main__':
    main()

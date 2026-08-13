#!/usr/bin/env python3
"""
分析每个head选择的具体tokens
特别关注深层的聚焦模式
"""

import json
import os
import sys
import torch
import numpy as np
from collections import Counter, defaultdict
from transformers import AutoTokenizer

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data, PreprocessScope


def load_head_selections(
    output_dir='./kvcache_headwise_analysis',
    cache_path='/mnt/data/reflect/Qwen2.5-7B-Instruct/headwise_kv_cache',
    example_id=4,
    chunk_ids=None
):
    """
    加载per-head selection的indices和scores

    Returns:
        head_selections: [layer_idx][head_idx] -> global token indices
        chunk_info: 每个chunk的信息
    """
    # 从之前的运行中加载
    # 需要重新计算，因为我们保存时是按chunk分的

    # 先加载保存的chunk-wise indices
    chunk_selections = {}

    # 跳过chunk 0和1（prefix cache命中）
    for chunk_id in chunk_ids[2:]:
        try:
            indices_file = f'{cache_path}/{example_id}_{chunk_id}_indices.pt'
            chunk_indices = torch.load(indices_file, weights_only=False)
            chunk_selections[chunk_id] = chunk_indices
        except FileNotFoundError:
            print(f"Warning: {indices_file} not found, skipping")
            continue

    return chunk_selections


def analyze_token_types(tokens, tokenizer):
    """
    分析tokens的类型分布

    Returns:
        类型统计字典
    """
    types = {
        'numbers': 0,
        'dates': 0,
        'entities': 0,  # 大写开头
        'punctuation': 0,
        'common_words': 0,
        'special': 0,
    }

    entities = []
    numbers = []

    for token in tokens:
        text = tokenizer.decode([token]).strip()

        if not text:
            types['special'] += 1
        elif text[0].isupper() and len(text) > 1:
            types['entities'] += 1
            entities.append(text)
        elif any(c.isdigit() for c in text):
            types['numbers'] += 1
            numbers.append(text)
            # 检查是否是日期格式
            if '12' in text or '16' in text or '20' in text:
                types['dates'] += 1
        elif text in '.,;:!?()[]{}':
            types['punctuation'] += 1
        else:
            types['common_words'] += 1

    return types, entities, numbers


def analyze_attention_score_distribution(scores):
    """
    分析attention score的分布
    """
    scores = np.array(scores)

    return {
        'min': float(scores.min()),
        'max': float(scores.max()),
        'mean': float(scores.mean()),
        'median': float(np.median(scores)),
        'std': float(scores.std()),
        'q25': float(np.percentile(scores, 25)),
        'q75': float(np.percentile(scores, 75)),
    }


def extract_key_phrases(tokens, tokenizer, window=5):
    """
    提取关键短语（连续的tokens）
    """
    phrases = []

    for i in range(len(tokens) - window + 1):
        phrase_tokens = tokens[i:i+window]
        phrase_text = tokenizer.decode(phrase_tokens).strip()
        phrases.append(phrase_text)

    # 返回最常见的短语
    phrase_counter = Counter(phrases)
    return phrase_counter.most_common(10)


def main(
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    example_idx=4,
    sub_question_idx=1,
    output_dir='./head_token_analysis'
):
    """
    主分析函数
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*100}")
    print("Per-Head Token Selection Analysis")
    print(f"{'='*100}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load data
    print("Loading data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen', 10,
        max_main_questions=example_idx + 1,
        preprocess=True,
        preprocess_scope=PreprocessScope.GLOBAL
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}\n")

    # Get document tokens
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']

    # Build full document sequence (skip chunks 0 and 1)
    all_doc_tokens = []
    chunk_boundaries = []
    current_pos = 0

    for idx, chunk_id in enumerate(doc_chunk_ids[2:], start=2):  # 从chunk 2开始
        chunk_tokens = doc_tensors[chunk_id - 1].tolist()
        chunk_boundaries.append({
            'chunk_id': chunk_id,
            'start': current_pos,
            'end': current_pos + len(chunk_tokens),
            'length': len(chunk_tokens)
        })
        all_doc_tokens.extend(chunk_tokens)
        current_pos += len(chunk_tokens)

    print(f"Total recomputable tokens: {len(all_doc_tokens)}")
    print(f"Chunk boundaries:")
    for cb in chunk_boundaries:
        chunk_text = tokenizer.decode(doc_tensors[cb['chunk_id']-1][:50])
        print(f"  Chunk {cb['chunk_id']}: pos {cb['start']}-{cb['end']} (len={cb['length']})")
        print(f"    Preview: {chunk_text[:100]}...")

    # 重新运行selection来获取详细信息
    print(f"\n{'='*100}")
    print("Re-computing head selections to get attention scores...")
    print(f"{'='*100}\n")

    # 加载完整的KV caches
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    num_layers = 28
    num_kv_heads = 4
    ratio = 0.3

    # 加载documents的KV caches
    doc_key_caches = []
    for idx, chunk_id in enumerate(doc_chunk_ids[2:], start=2):
        chunk_key_cache = torch.load(
            f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt',
            weights_only=True
        ).to('cpu')  # 用CPU节省内存
        doc_key_caches.append(chunk_key_cache)

    # 拼接
    all_doc_keys = []
    for layer_idx in range(num_layers):
        layer_keys = [doc_key_caches[i][layer_idx] for i in range(len(doc_key_caches))]
        layer_key = torch.cat(layer_keys, dim=2)
        all_doc_keys.append(layer_key)

    # 对每层的每个head计算selection
    print("Computing selections with scores...")

    layer_head_selections = []  # [layer][head] -> {'indices': [], 'scores': []}

    for layer_idx in range(num_layers):
        layer_selections = []

        doc_keys = all_doc_keys[layer_idx]  # [1, num_kv_heads, seq_len, head_dim]
        head_dim = doc_keys.shape[-1]

        for head_idx in range(num_kv_heads):
            head_keys = doc_keys[:, head_idx:head_idx+1, :, :]

            # 计算attention scores
            avg_key = head_keys.mean(dim=2, keepdim=True)
            similarities = torch.matmul(avg_key, head_keys.transpose(2, 3))
            similarities = similarities / (head_dim ** 0.5)
            attention_weights = torch.nn.functional.softmax(similarities, dim=-1)
            attention_scores = attention_weights[0, 0, 0, :].float().numpy()

            # 选择top-k
            k = int(len(all_doc_tokens) * ratio)
            top_k_indices = np.argsort(attention_scores)[-k:]
            top_k_scores = attention_scores[top_k_indices]

            # 排序保持顺序
            sorted_order = np.argsort(top_k_indices)
            top_k_indices = top_k_indices[sorted_order]
            top_k_scores = top_k_scores[sorted_order]

            layer_selections.append({
                'indices': top_k_indices,
                'scores': top_k_scores,
                'all_scores': attention_scores
            })

        layer_head_selections.append(layer_selections)

        if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx:2d} done")

    # 现在分析每一层的每个head
    print(f"\n{'='*100}")
    print("Analyzing selected tokens per head")
    print(f"{'='*100}\n")

    analysis_results = []

    # 关注的层：浅层、中层、深层
    focus_layers = [0, 5, 14, 20, 27]

    for layer_idx in focus_layers:
        print(f"\n{'#'*100}")
        print(f"LAYER {layer_idx} Analysis")
        print(f"{'#'*100}\n")

        layer_result = {
            'layer': layer_idx,
            'heads': []
        }

        for head_idx in range(num_kv_heads):
            head_data = layer_head_selections[layer_idx][head_idx]
            selected_indices = head_data['indices']
            selected_scores = head_data['scores']

            # 获取选中的tokens
            selected_tokens = [all_doc_tokens[i] for i in selected_indices]

            print(f"{'='*80}")
            print(f"Layer {layer_idx}, Head {head_idx}")
            print(f"{'='*80}")

            # 1. Token类型分析
            types, entities, numbers = analyze_token_types(selected_tokens, tokenizer)

            print(f"\nToken Type Distribution:")
            print(f"  Numbers: {types['numbers']} ({types['numbers']/len(selected_tokens)*100:.1f}%)")
            print(f"  Dates: {types['dates']} ({types['dates']/len(selected_tokens)*100:.1f}%)")
            print(f"  Entities (capitalized): {types['entities']} ({types['entities']/len(selected_tokens)*100:.1f}%)")
            print(f"  Punctuation: {types['punctuation']} ({types['punctuation']/len(selected_tokens)*100:.1f}%)")
            print(f"  Common words: {types['common_words']} ({types['common_words']/len(selected_tokens)*100:.1f}%)")

            # 2. 显示包含关键数字的tokens
            print(f"\nTokens containing '1216' or '1220':")
            key_date_tokens = []
            for idx, token in zip(selected_indices, selected_tokens):
                text = tokenizer.decode([token]).strip()
                if '1216' in text or '1220' in text or '12' in text or '16' in text or '20' in text:
                    score = selected_scores[list(selected_indices).index(idx)]
                    key_date_tokens.append((idx, token, text, score))
                    print(f"  Position {idx}: '{text}' (score={score:.6f})")

            if not key_date_tokens:
                print(f"  ⚠️ No date-related tokens found!")

            # 3. Top-10 高分tokens
            print(f"\nTop 10 highest attention tokens:")
            top10_idx = np.argsort(selected_scores)[-10:][::-1]
            for rank, idx in enumerate(top10_idx, 1):
                global_idx = selected_indices[idx]
                token = selected_tokens[idx]
                score = selected_scores[idx]
                text = tokenizer.decode([token]).strip()
                print(f"  {rank:2d}. Position {global_idx:4d}: '{text:20s}' (score={score:.6f})")

            # 4. Attention score分布
            score_dist = analyze_attention_score_distribution(selected_scores)
            print(f"\nAttention Score Distribution:")
            print(f"  Min: {score_dist['min']:.6f}, Max: {score_dist['max']:.6f}")
            print(f"  Mean: {score_dist['mean']:.6f}, Median: {score_dist['median']:.6f}")
            print(f"  Std: {score_dist['std']:.6f}")
            print(f"  Q25-Q75: {score_dist['q25']:.6f} - {score_dist['q75']:.6f}")

            # 5. 关键实体
            print(f"\nTop Entities (capitalized words):")
            entity_counter = Counter(entities)
            for entity, count in entity_counter.most_common(10):
                print(f"  '{entity}': {count}")

            # 6. 数字tokens
            print(f"\nAll Number Tokens:")
            number_counter = Counter(numbers)
            for num, count in number_counter.most_common(20):
                print(f"  '{num}': {count}")

            # 保存结果
            head_result = {
                'head': head_idx,
                'num_selected': len(selected_tokens),
                'types': types,
                'key_date_tokens': [(int(idx), text, float(score))
                                   for idx, _, text, score in key_date_tokens],
                'score_distribution': score_dist,
                'top_entities': entity_counter.most_common(10),
                'top_numbers': number_counter.most_common(20),
            }

            layer_result['heads'].append(head_result)

        analysis_results.append(layer_result)

        # 层间对比
        print(f"\n{'='*80}")
        print(f"Layer {layer_idx} Cross-Head Comparison")
        print(f"{'='*80}")

        print(f"\nDate tokens ('1216'/'1220') per head:")
        for head_idx in range(num_kv_heads):
            head_res = layer_result['heads'][head_idx]
            num_dates = len(head_res['key_date_tokens'])
            print(f"  Head {head_idx}: {num_dates} date-related tokens")
            if num_dates > 0:
                print(f"    Tokens: {[text for _, text, _ in head_res['key_date_tokens'][:5]]}")

        print(f"\nEntity focus per head:")
        for head_idx in range(num_kv_heads):
            head_res = layer_result['heads'][head_idx]
            entity_ratio = head_res['types']['entities'] / head_res['num_selected'] * 100
            print(f"  Head {head_idx}: {entity_ratio:.1f}% entities")

    # 保存结果
    output_file = os.path.join(output_dir, 'head_token_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"Analysis saved to {output_file}")
    print(f"{'='*100}")

    # 生成总结
    print(f"\n{'='*100}")
    print("SUMMARY: Cross-Layer Head Behavior")
    print(f"{'='*100}\n")

    for head_idx in range(num_kv_heads):
        print(f"Head {head_idx} Evolution:")
        for layer_result in analysis_results:
            layer_idx = layer_result['layer']
            head_res = layer_result['heads'][head_idx]
            num_dates = len(head_res['key_date_tokens'])
            entity_ratio = head_res['types']['entities'] / head_res['num_selected'] * 100

            print(f"  Layer {layer_idx:2d}: {num_dates:2d} date tokens, "
                  f"{entity_ratio:5.1f}% entities, "
                  f"score range: {head_res['score_distribution']['min']:.6f}-"
                  f"{head_res['score_distribution']['max']:.6f}")
        print()


if __name__ == '__main__':
    main(
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        example_idx=4,
        sub_question_idx=1,
        output_dir='./head_token_analysis'
    )

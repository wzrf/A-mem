#!/usr/bin/env python3
"""
手动选择包含正确答案的 token，测试生成质量
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
sys.path.insert(0, '/mnt/data/wjh/FusionRAG/ktransformers')

from per_head_generation import sparse_prefill_per_head, generate_with_sparse_prefill
from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache


def find_answer_positions(position_to_token, tokenizer):
    """找到文档中包含正确答案的位置"""

    # 将 position_to_token 转换为文本
    positions = sorted(position_to_token.keys())

    # 构建完整文本和位置映射
    tokens_list = [position_to_token[p] for p in positions]
    full_text = tokenizer.decode(tokens_list)

    print("=" * 100)
    print("DOCUMENT CONTENT (first 3000 chars)")
    print("=" * 100)
    print(full_text[:3000])
    print("..." if len(full_text) > 3000 else "")

    # 找关键短语
    print("\n" + "=" * 100)
    print("SEARCHING FOR KEY PHRASES")
    print("=" * 100)

    key_phrases = [
        "1216",
        "1220",
        "crowned",
        "coronation",
        "Gloucester",
        "Westminster",
        "17 May",
        "May 1220"
    ]

    # 逐 token 重建，找到每个 token 对应的文本位置
    token_to_text_pos = {}
    current_text = ""
    for pos in positions:
        token_id = position_to_token[pos]
        token_text = tokenizer.decode([token_id])
        start_pos = len(current_text)
        current_text += token_text
        token_to_text_pos[pos] = (start_pos, len(current_text), token_text)

    # 找到包含关键短语的 token 范围
    answer_positions = set()

    for phrase in key_phrases:
        idx = 0
        while True:
            idx = full_text.find(phrase, idx)
            if idx == -1:
                break
            end_idx = idx + len(phrase)

            # 找到覆盖这个范围的所有 token
            for pos in positions:
                start, end, txt = token_to_text_pos[pos]
                if start < end_idx and end > idx:
                    answer_positions.add(pos)

            print(f"  Found '{phrase}' at text position {idx}")
            idx += 1

    # 扩展选择范围：选择每个关键 token 前后 N 个 token 作为上下文
    extended_positions = set()
    context_window = 5  # 缩小到 ±5

    for pos in answer_positions:
        for offset in range(-context_window, context_window + 1):
            new_pos = pos + offset
            if new_pos in position_to_token:
                extended_positions.add(new_pos)

    print(f"\nKey phrase tokens: {len(answer_positions)}")
    print(f"Extended with context (±{context_window}): {len(extended_positions)}")

    return extended_positions, answer_positions


def main():
    device = "cuda:0"
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    cache_path = '/mnt/data/reflect/'
    model_name = 'Qwen2.5-7B-Instruct'
    example_idx = 4
    sub_question_idx = 1

    print("=" * 100)
    print("Manual Token Selection Test")
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

    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")

    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    passages = q_data['docs']

    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)

    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    # 首先加载文档获取 position_to_token
    print("\n" + "=" * 80)
    print("Loading document caches to build position mapping")
    print("=" * 80)

    num_layers = 28
    num_kv_heads = 4

    # 构建 position_to_token 映射
    position_to_token = {}
    current_pos = 0
    prefix_len = 0

    for chunk_idx, chunk_id in enumerate(doc_chunk_ids):
        # 获取这个 chunk 的 tokens
        if chunk_id == 0:
            # System prompt - 从 cache 文件推断长度
            key_path = os.path.join(full_cache_path, f'{example_idx}_{chunk_id}_key.pt')
            chunk_key = torch.load(key_path, map_location='cpu', weights_only=True)
            chunk_len = chunk_key[0].shape[2]
            prefix_len = chunk_len
        else:
            passage = passages[chunk_id - 1]
            doc_text = f"Document: {passage}\n"
            tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            chunk_len = len(tokens)

            # 记录每个位置对应的 token
            for i, token_id in enumerate(tokens):
                position_to_token[current_pos - prefix_len + i] = token_id

        current_pos += chunk_len

    doc_len = current_pos - prefix_len
    print(f"Prefix length: {prefix_len}")
    print(f"Document length: {doc_len}")
    print(f"Position to token mapping size: {len(position_to_token)}")

    # 找到正确答案的位置
    selected_positions, key_positions = find_answer_positions(position_to_token, tokenizer)

    print(f"\n" + "=" * 80)
    print(f"MANUAL SELECTION: {len(selected_positions)} tokens ({len(selected_positions)/doc_len*100:.1f}%)")
    print("=" * 80)

    # 显示选中的 tokens
    print("\nSelected token preview (first 100):")
    sorted_selected = sorted(selected_positions)[:100]
    for pos in sorted_selected:
        token_id = position_to_token.get(pos, 0)
        token_text = tokenizer.decode([token_id])
        is_key = "★" if pos in key_positions else " "
        print(f"  {is_key} Position {pos:4d}: '{token_text}'")

    # 构建 layer_head_selections 和 layer_union_positions
    layer_head_selections = {}
    layer_union_positions = {}

    selected_list = sorted(list(selected_positions))

    for layer_idx in range(num_layers):
        layer_union_positions[layer_idx] = selected_list
        layer_head_selections[layer_idx] = {}
        for kv_head_idx in range(num_kv_heads):
            layer_head_selections[layer_idx][kv_head_idx] = {
                'positions': selected_list.copy(),
                'num_selected': len(selected_list)
            }

    # 加载完整 cache
    print("\n" + "=" * 80)
    print("Loading FULL cache")
    print("=" * 80)

    max_cache_len = 32768
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    for chunk_idx, chunk_id in enumerate(doc_chunk_ids):
        key_path = os.path.join(full_cache_path, f'{example_idx}_{chunk_id}_key.pt')
        value_path = os.path.join(full_cache_path, f'{example_idx}_{chunk_id}_value.pt')

        chunk_key_cache = torch.load(key_path, map_location=device, weights_only=True)
        chunk_value_cache = torch.load(value_path, map_location=device, weights_only=True)

        for layer_idx in range(num_layers):
            layer_key = chunk_key_cache[layer_idx].to(device)
            layer_value = chunk_value_cache[layer_idx].to(device)
            layer_len = layer_key.shape[2]
            current_pos = past_key_values.past_tokens[layer_idx]

            past_key_values.key_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len

        print(f"  Loaded chunk {chunk_id}: {layer_len} tokens")

    total_cache_len = past_key_values.past_tokens[0]
    print(f"Total cache length: {total_cache_len}")

    # Sparse prefill
    print("\n" + "=" * 80)
    print("Sparse Prefill with Manual Selection")
    print("=" * 80)

    layer_recompute_stats, overall_ratio = sparse_prefill_per_head(
        model,
        past_key_values,
        layer_head_selections,
        layer_union_positions,
        position_to_token,
        query_tensor,
        prefix_len,
        doc_len,
        device
    )

    # 重置 past_tokens
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] = prefix_len + doc_len

    # Generate
    print("\n" + "=" * 80)
    print("Generating Answer")
    print("=" * 80)

    generated_text = generate_with_sparse_prefill(
        model,
        tokenizer,
        past_key_values,
        query_tensor,
        prefix_len,
        doc_len,
        max_new_tokens=150,
        device=device
    )

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"\nGround Truth:\n{sub_q_info['answer']}")
    print(f"\nGenerated Answer:\n{generated_text}")
    print(f"\nSelection ratio: {len(selected_positions)/doc_len*100:.1f}%")
    print("=" * 100)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    main()

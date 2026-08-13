#!/usr/bin/env python3
"""
调试：比较 full text generation 和 per_head_generation 使用的输入是否一致
"""

import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data, load_system_prompt


def compare_inputs(
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    example_idx=4,
    sub_question_idx=1,
    device="cuda:0"
):
    """
    比较 full text 和 per_head_generation 使用的输入
    """
    print(f"\n{'='*100}")
    print("Debug: Compare Inputs for Full Text vs Per-Head Generation")
    print(f"{'='*100}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load data
    print("Loading data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen', 10,
        max_main_questions=example_idx + 1,
        preprocess=False
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"\n{'='*80}")
    print(f"Example {example_idx}, Sub-question {sub_question_idx}")
    print(f"Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print(f"{'='*80}\n")

    # ===== Full Text 方式的输入 =====
    print("="*80)
    print("Full Text Input (test_full_text_generation.py 方式)")
    print("="*80)

    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    print(f"\nSystem tensor shape: {system_tensor.shape}")
    print(f"System text (first 200 chars):")
    print(f"  '{tokenizer.decode(system_tensor[:50])}'...")

    print(f"\nDocument chunk_ids: {doc_chunk_ids}")
    print(f"Number of documents: {len(sub_q_doc_tensors)}")
    for i, doc_tensor in enumerate(sub_q_doc_tensors):
        doc_text = tokenizer.decode(doc_tensor)
        print(f"\n  Doc {i+1} (chunk_id={doc_chunk_ids[i]}): {doc_tensor.shape[0]} tokens")
        print(f"    Text (first 100 chars): '{doc_text[:100]}'...")

    print(f"\nQuestion tensor shape: {question_tensor.shape}")
    print(f"Question text: '{question_text}'")

    # 拼接完整输入
    iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    full_input = torch.cat(iter_tokens)

    print(f"\nFull input length: {full_input.shape[0]} tokens")

    # ===== Per-Head Generation 方式的输入 =====
    print("\n" + "="*80)
    print("Per-Head Generation Input (per_head_generation.py 方式)")
    print("="*80)

    # per_head_generation.py 使用的是 q_data['docs'] (passages)
    passages = q_data['docs']

    print(f"\nTotal passages in q_data['docs']: {len(passages)}")
    print(f"doc_chunk_ids for this sub_question: {doc_chunk_ids}")

    # per_head_generation 加载 cache 的方式
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')
    print(f"\nCache path: {full_cache_path}")

    # 检查 cache 文件
    print(f"\nCache files for example {example_idx}:")
    for chunk_id in doc_chunk_ids:
        key_path = f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt'
        value_path = f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt'
        key_exists = os.path.exists(key_path)
        value_exists = os.path.exists(value_path)
        print(f"  Chunk {chunk_id}: key={key_exists}, value={value_exists}")

    # per_head_generation 使用 passages[2:] 作为 document
    # 注意：passages 的索引从 0 开始，而 chunk_id 从 0/1 开始
    # chunks 0, 1 是 system prompt 相关的
    # chunks 2+ 是 documents

    print(f"\n--- Per-head generation 的文档加载逻辑 ---")
    print(f"doc_chunk_ids[2:] (文档 chunks): {doc_chunk_ids[2:] if len(doc_chunk_ids) > 2 else doc_chunk_ids}")

    # 让我们看看 per_head_generation.py 中的 passages[2:] 对应什么
    print(f"\npassages[2:] 的文档内容:")
    for i, passage in enumerate(passages[2:]):
        print(f"  Doc {i} (passages[{i+2}]): {passage[:100]}...")

    # 比较 doc_tensors 和 passages
    print(f"\n--- 比较 doc_tensors vs passages ---")
    print(f"len(doc_tensors): {len(doc_tensors)}")
    print(f"len(passages): {len(passages)}")

    for i in range(min(len(doc_tensors), len(passages))):
        doc_text_from_tensor = tokenizer.decode(doc_tensors[i])
        print(f"\nIndex {i}:")
        print(f"  doc_tensors[{i}] (first 80 chars): '{doc_text_from_tensor[:80]}'")
        print(f"  passages[{i}] (first 80 chars): '{passages[i][:80] if isinstance(passages[i], str) else 'Tensor'}'")

    # ===== 关键差异分析 =====
    print("\n" + "="*80)
    print("关键差异分析")
    print("="*80)

    print("""
1. Full Text 方式:
   - 使用 system_tensor + [doc_tensors[chunk_id-1] for chunk_id in chunk_ids] + question_tensor
   - chunk_id 是从 1 开始的，所以 chunk_id=1 对应 doc_tensors[0]

2. Per-Head Generation 方式:
   - 加载 cache 时使用 doc_chunk_ids[2:] (跳过 chunks 0, 1)
   - 文档内容来自 passages[2:] (跳过 system 相关的前两项)
   - 但构建 query tensor 时使用相同的 question_text
""")

    # 让我检查一下 per_head_generation 中 cache 的实际内容
    print("\n" + "="*80)
    print("检查 KV Cache 内容")
    print("="*80)

    # 加载第一个文档的 cache，检查其长度
    for chunk_id in doc_chunk_ids[:3]:  # 只看前 3 个
        key_path = f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt'
        if os.path.exists(key_path):
            key_cache = torch.load(key_path, weights_only=True)
            # key_cache 的形状应该是 [num_layers, batch, num_kv_heads, seq_len, head_dim]
            # 或 [num_layers][batch, num_kv_heads, seq_len, head_dim]
            if isinstance(key_cache, list):
                print(f"\nChunk {chunk_id}: {len(key_cache)} layers")
                print(f"  Layer 0 shape: {key_cache[0].shape}")
                seq_len = key_cache[0].shape[2]
                print(f"  Sequence length: {seq_len}")
            else:
                print(f"\nChunk {chunk_id} shape: {key_cache.shape}")

    return {
        'full_text_input_len': full_input.shape[0],
        'doc_chunk_ids': doc_chunk_ids,
        'num_doc_tensors': len(doc_tensors),
        'num_passages': len(passages)
    }


if __name__ == '__main__':
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    results = compare_inputs(
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        example_idx=4,
        sub_question_idx=1,
        device='cuda:0'
    )

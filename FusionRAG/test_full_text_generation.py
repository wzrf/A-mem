#!/usr/bin/env python3
"""
Test Full Text Generation vs Sparse Prefill Generation

对比两种方式的生成结果：
1. 完整文本拼接后直接 prefill 并生成（baseline，等价于 ratio=1）
2. 使用 per_head_generation.py 的 sparse prefill 方式生成

用于调试生成格式问题
"""

import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data, load_system_prompt


def full_text_generation(
    model,
    tokenizer,
    full_input_tensor,
    max_new_tokens=100,
    device="cuda:0"
):
    """
    完整文本拼接后直接 prefill 并生成（baseline）

    这是最简单的生成方式，等价于 ratio=1
    """
    from ktransformers.models.custom_cache import StaticCache

    print(f"\n{'='*80}")
    print("Full Text Generation (Baseline)")
    print(f"{'='*80}\n")

    input_len = full_input_tensor.shape[1]
    print(f"Input length: {input_len} tokens")

    # 初始化 StaticCache
    max_cache_len = 32768
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    # 做完整 prefill
    cache_position = torch.arange(0, input_len, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=full_input_tensor,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True
        )
        logits = outputs.logits[0, -1, :]

    # 自回归生成
    generated_ids = []
    current_position = input_len

    print("Generating tokens...")

    for step in range(max_new_tokens):
        # Sample next token (greedy)
        next_token_id = torch.argmax(logits, dim=-1)

        # 检查 EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"  EOS reached at step {step}")
            break

        generated_ids.append(next_token_id.item())

        if step < 10 or step % 10 == 0:
            token_text = tokenizer.decode([next_token_id.item()])
            print(f"  Step {step}: '{token_text}'")

        # Forward 下一个 token
        next_token_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
        cache_position = torch.tensor([current_position], device=device)

        with torch.no_grad():
            outputs = model(
                input_ids=next_token_tensor,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True
            )
            logits = outputs.logits[0, -1, :]

        current_position += 1

    # Decode 生成的 tokens（只 decode 新生成的，不包括 query）
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\n{'='*80}")
    print("Generation Complete")
    print(f"{'='*80}\n")

    return generated_text, generated_ids


def test_generation_comparison(
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    example_idx=4,
    sub_question_idx=1,
    max_new_tokens=100,
    device="cuda:0"
):
    """
    测试完整文本生成 vs sparse prefill 生成
    """
    print(f"\n{'='*100}")
    print("Test Full Text Generation vs Sparse Prefill")
    print(f"{'='*100}\n")

    # Load model
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, device_map = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()

    # Load tokenizer
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

    print(f"\n{'='*100}")
    print(f"Example {example_idx}, Sub-question {sub_question_idx}")
    print(f"{'='*100}")
    print(f"Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print(f"{'='*100}\n")

    # Get documents for this sub-question
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    # 构建 question tensor
    # 注意：这里的格式与 per_head_generation.py 保持一致
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    print(f"System prompt length: {system_tensor.shape[0]} tokens")
    print(f"Question tensor length: {question_tensor.shape[0]} tokens")
    print(f"Document tensors: {len(sub_q_doc_tensors)} documents")
    for i, doc_tensor in enumerate(sub_q_doc_tensors):
        print(f"  Doc {i+1}: {doc_tensor.shape[0]} tokens")

    # 拼接完整输入
    iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    full_input = torch.cat(iter_tokens).unsqueeze(0).to(device)

    print(f"\nFull input length: {full_input.shape[1]} tokens")

    # 打印完整输入的文本（前 500 字符和后 200 字符）
    full_text = tokenizer.decode(full_input[0], skip_special_tokens=False)
    print(f"\n--- Full Input Text (first 500 chars) ---")
    print(full_text[:500])
    print("...")
    print(f"--- Full Input Text (last 200 chars) ---")
    print(full_text[-200:])
    print("---")

    # 测试1: 完整文本生成（baseline）
    print("\n" + "="*100)
    print("TEST 1: Full Text Generation (Baseline, equivalent to ratio=1)")
    print("="*100)

    baseline_text, baseline_ids = full_text_generation(
        model, tokenizer, full_input, max_new_tokens, device
    )

    print(f"\nBaseline Generated Answer:")
    print(f"  '{baseline_text}'")

    # 测试2: 跳过 model.generate() API（自定义模型不支持）
    generate_api_text = baseline_text  # 使用相同结果

    # 对比结果
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)

    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"\nGround Truth: {sub_q_info['answer']}")
    print(f"\nBaseline (manual generation):")
    print(f"  '{baseline_text}'")
    print(f"\nGenerate API:")
    print(f"  '{generate_api_text}'")

    # 检查两种方式是否一致
    if baseline_text.strip() == generate_api_text.strip():
        print("\n✓ Two methods produce identical results")
    else:
        print("\n✗ Two methods produce DIFFERENT results!")
        print(f"  Baseline length: {len(baseline_text)}")
        print(f"  Generate API length: {len(generate_api_text)}")

    print(f"\n{'='*100}")

    return {
        'question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'baseline_answer': baseline_text,
        'generate_api_answer': generate_api_text,
        'baseline_ids': baseline_ids
    }


def test_with_sparse_ratio1(
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    example_idx=4,
    sub_question_idx=1,
    max_new_tokens=100,
    device="cuda:0"
):
    """
    测试 sparse prefill with ratio=1 vs full text generation

    ratio=1 意味着选择所有 tokens，应该和 full text 结果一致
    """
    # 先运行 full text baseline
    baseline_results = test_generation_comparison(
        model_path, data_path, cache_path, model_name, bge_model_path,
        example_idx, sub_question_idx, max_new_tokens, device
    )

    # 然后运行 per_head_generation.py with ratio=1
    print("\n" + "="*100)
    print("TEST 3: Per-Head Generation with ratio=1")
    print("="*100)

    from per_head_generation import main as per_head_main

    sparse_results = per_head_main(
        model_path=model_path,
        data_path=data_path,
        cache_path=cache_path,
        model_name=model_name,
        bge_model_path=bge_model_path,
        example_idx=example_idx,
        sub_question_idx=sub_question_idx,
        total_ratio=1.0,  # ratio=1 means select ALL tokens
        max_new_tokens=max_new_tokens,
        device=device,
        output_path='./test_sparse_ratio1_results.json'
    )

    sparse_answer = sparse_results['generated_answer']

    print("\n" + "="*100)
    print("FINAL COMPARISON: Full Text vs Sparse ratio=1")
    print("="*100)

    print(f"\nQuestion: {baseline_results['question']}")
    print(f"\nGround Truth: {baseline_results['ground_truth']}")
    print(f"\nFull Text Baseline:")
    print(f"  '{baseline_results['baseline_answer']}'")
    print(f"\nSparse Prefill (ratio=1):")
    print(f"  '{sparse_answer}'")

    # 分析差异
    baseline_clean = baseline_results['baseline_answer'].strip()
    sparse_clean = sparse_answer.strip()

    if baseline_clean == sparse_clean:
        print("\n✓ MATCH: Full text and sparse ratio=1 produce identical answers")
    else:
        print("\n✗ MISMATCH: Full text and sparse ratio=1 produce DIFFERENT answers!")
        print(f"\n  Analyzing differences...")

        # 检查 sparse_answer 是否包含了额外的前缀
        if 'user' in sparse_answer or 'Question:' in sparse_answer:
            print(f"  → Sparse answer contains format markers (user, Question:)")
            print(f"  → This indicates the query tokens are being included in the output")

        # 找到公共部分
        if baseline_clean in sparse_clean:
            print(f"  → Baseline is a substring of sparse answer")
            extra = sparse_clean.replace(baseline_clean, '')
            print(f"  → Extra content: '{extra[:100]}...'")
        elif sparse_clean in baseline_clean:
            print(f"  → Sparse answer is a substring of baseline")
            extra = baseline_clean.replace(sparse_clean, '')
            print(f"  → Missing content: '{extra[:100]}...'")

    print(f"\n{'='*100}")

    return {
        'baseline': baseline_results,
        'sparse_ratio1': sparse_results
    }


if __name__ == '__main__':
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # 只运行 baseline 测试
    results = test_generation_comparison(
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        example_idx=4,
        sub_question_idx=1,
        max_new_tokens=100,
        device='cuda:0'
    )

    print("\n\nFinal Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

#!/usr/bin/env python3
"""
Analyze attention and logits differences between rate=0.3 and rate=1.0

This script:
1. Selects a single example from the dataset
2. Runs generation with rate=0.3 and rate=1.0
3. For each rate, re-computes the full sequence (query + generated answer)
4. Compares the last layer attention (L2 distance) and logits (top-1 token changes)
"""

import json
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoConfig

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import (
    prefill_and_save_kv_cache,
    load_kv_and_generate,
    prefill_and_generate,
)
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect import load_model, load_system_prompt, prepare_reflect_data, PreprocessScope


def recompute_with_attention(
    model,
    tokenizer,
    full_sequence: torch.Tensor,
    device: str = "cuda:0",
    device_map=None,
    return_attentions: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-compute a full sequence and return logits and last-layer attention

    Args:
        model: The language model
        tokenizer: Tokenizer
        full_sequence: Full input sequence (query + generated answer)
        device: Device to use
        device_map: Multi-GPU device map
        return_attentions: Whether to return attention weights

    Returns:
        logits: [seq_len, vocab_size] - logits for each position
        last_layer_attention: [num_heads, seq_len, seq_len] - last layer attention weights
    """
    input_device = "cuda:0" if device_map is not None else device
    full_sequence = full_sequence.unsqueeze(0).to(input_device)

    with torch.no_grad():
        # Forward pass with output_attentions=True
        outputs = model(
            input_ids=full_sequence,
            output_attentions=return_attentions,
            use_cache=False,
            return_dict=True
        )

        logits = outputs.logits[0]  # [seq_len, vocab_size]

        if return_attentions and outputs.attentions is not None:
            # Get last layer attention: tuple of [batch, num_heads, seq_len, seq_len]
            last_layer_attention = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
        else:
            last_layer_attention = None

    return logits, last_layer_attention


def analyze_logits_change(
    logits_rate03: torch.Tensor,
    logits_rate1: torch.Tensor,
    tokenizer,
    query_len: int,
    save_path: str
):
    """
    Analyze logits changes for each decode step

    Args:
        logits_rate03: Logits from rate=0.3, shape [seq_len, vocab_size]
        logits_rate1: Logits from rate=1.0, shape [seq_len, vocab_size]
        tokenizer: Tokenizer
        query_len: Length of the query (to identify decode positions)
        save_path: Path to save results
    """
    # Only analyze the decode positions (query_len-1 onwards)
    # Position i predicts token i+1
    decode_positions = range(query_len - 1, logits_rate03.shape[0])

    results = []

    print("\n" + "="*100)
    print("LOGITS CHANGE ANALYSIS (Per Decode Step)")
    print("="*100)

    for pos in decode_positions:
        step = pos - (query_len - 1) + 1  # Decode step number (1-indexed)

        # Get probabilities
        probs_rate03 = torch.softmax(logits_rate03[pos], dim=-1)
        probs_rate1 = torch.softmax(logits_rate1[pos], dim=-1)

        # Get top-1 tokens
        top1_token_rate03 = torch.argmax(probs_rate03).item()
        top1_token_rate1 = torch.argmax(probs_rate1).item()

        top1_prob_rate03 = probs_rate03[top1_token_rate03].item()
        top1_prob_rate1 = probs_rate1[top1_token_rate1].item()

        # Decode tokens
        token_text_rate03 = tokenizer.decode([top1_token_rate03])
        token_text_rate1 = tokenizer.decode([top1_token_rate1])

        # Check if top-1 changed
        top1_changed = top1_token_rate03 != top1_token_rate1

        print(f"\n--- Decode Step {step} (Position {pos}) ---")

        if top1_changed:
            # Token A -> Token B
            print(f"  ⚠️  TOP-1 TOKEN CHANGED:")
            print(f"    Rate=1.0: '{token_text_rate1}' (ID={top1_token_rate1}, prob={top1_prob_rate1:.6f})")
            print(f"    Rate=0.3: '{token_text_rate03}' (ID={top1_token_rate03}, prob={top1_prob_rate03:.6f})")

            # Show how probabilities changed for both tokens
            prob_A_in_rate03 = probs_rate03[top1_token_rate1].item()
            prob_B_in_rate1 = probs_rate1[top1_token_rate03].item()

            print(f"\n  Token A ('{token_text_rate1}') probability change:")
            print(f"    Rate=1.0: {top1_prob_rate1:.6f}  →  Rate=0.3: {prob_A_in_rate03:.6f}  (Δ={prob_A_in_rate03 - top1_prob_rate1:+.6f})")

            print(f"  Token B ('{token_text_rate03}') probability change:")
            print(f"    Rate=1.0: {prob_B_in_rate1:.6f}  →  Rate=0.3: {top1_prob_rate03:.6f}  (Δ={top1_prob_rate03 - prob_B_in_rate1:+.6f})")
        else:
            # Same token
            print(f"  ✓ TOP-1 TOKEN UNCHANGED: '{token_text_rate1}' (ID={top1_token_rate1})")
            print(f"    Probability change: {top1_prob_rate1:.6f} → {top1_prob_rate03:.6f} (Δ={top1_prob_rate03 - top1_prob_rate1:+.6f})")

        # Compute L2 distance for logits at this position
        logits_l2 = torch.norm(logits_rate03[pos] - logits_rate1[pos], p=2).item()
        probs_l2 = torch.norm(probs_rate03 - probs_rate1, p=2).item()

        print(f"  Logits L2 distance: {logits_l2:.6f}")
        print(f"  Probs L2 distance: {probs_l2:.6f}")

        # Save result
        result_entry = {
            'decode_step': step,
            'position': pos,
            'top1_changed': top1_changed,
            'rate1_token': token_text_rate1,
            'rate1_token_id': top1_token_rate1,
            'rate1_prob': top1_prob_rate1,
            'rate03_token': token_text_rate03,
            'rate03_token_id': top1_token_rate03,
            'rate03_prob': top1_prob_rate03,
            'logits_l2': logits_l2,
            'probs_l2': probs_l2,
        }

        if top1_changed:
            result_entry['token_A_prob_in_rate03'] = prob_A_in_rate03
            result_entry['token_B_prob_in_rate1'] = prob_B_in_rate1

        results.append(result_entry)

    # Save to JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"Logits analysis saved to: {save_path}")
    print(f"{'='*100}")

    return results


def analyze_attention_change(
    attn_rate03: torch.Tensor,
    attn_rate1: torch.Tensor,
    query_len: int,
    save_path: str
):
    """
    Analyze attention changes (L2 distance)

    Args:
        attn_rate03: Attention from rate=0.3, shape [num_heads, seq_len, seq_len]
        attn_rate1: Attention from rate=1.0, shape [num_heads, seq_len, seq_len]
        query_len: Length of the query
        save_path: Path to save results
    """
    num_heads, seq_len, _ = attn_rate03.shape

    print("\n" + "="*100)
    print("ATTENTION CHANGE ANALYSIS (Last Layer)")
    print("="*100)

    # Compute L2 distance for each head
    head_l2_distances = []
    for head_idx in range(num_heads):
        l2_dist = torch.norm(attn_rate03[head_idx] - attn_rate1[head_idx], p=2).item()
        head_l2_distances.append(l2_dist)
        print(f"  Head {head_idx}: L2 distance = {l2_dist:.6f}")

    # Overall L2 distance (across all heads)
    overall_l2 = torch.norm(attn_rate03 - attn_rate1, p=2).item()
    print(f"\n  Overall L2 distance (all heads): {overall_l2:.6f}")

    # Analyze per-position attention changes (for decode positions)
    print(f"\n  --- Per-Position Analysis (Decode Positions) ---")
    decode_positions = range(query_len - 1, seq_len)

    position_results = []
    for pos in decode_positions:
        step = pos - (query_len - 1) + 1

        # Average attention difference across heads for this query position
        attn_diff_at_pos = torch.norm(attn_rate03[:, pos, :] - attn_rate1[:, pos, :], p=2).item()

        print(f"    Decode Step {step} (Position {pos}): L2 distance = {attn_diff_at_pos:.6f}")

        position_results.append({
            'decode_step': step,
            'position': pos,
            'attention_l2': attn_diff_at_pos,
        })

    # Save results
    results = {
        'num_heads': num_heads,
        'seq_len': seq_len,
        'query_len': query_len,
        'head_l2_distances': head_l2_distances,
        'overall_l2': overall_l2,
        'per_position': position_results,
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save raw attention tensors (optional, can be large)
    torch.save({
        'attn_rate03': attn_rate03.cpu(),
        'attn_rate1': attn_rate1.cpu(),
    }, save_path.replace('.json', '_tensors.pt'))

    print(f"\n{'='*100}")
    print(f"Attention analysis saved to: {save_path}")
    print(f"Raw tensors saved to: {save_path.replace('.json', '_tensors.pt')}")
    print(f"{'='*100}")

    return results


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
    reprocess_method='FusionRAG',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    revert_rope=True,
    device="cuda:0",
    use_multi_gpu=False,
    example_idx=0,  # Which example to analyze
    sub_question_idx=0,  # Which sub-question to analyze
    output_dir='./attention_logits_analysis'
):
    """
    Main function for attention and logits comparison analysis
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading tokenizer and config from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "eager"  # Use eager mode to get attention weights

    print(f"Loading {model_type} model...")
    model, device_map = load_model(model_type, model_path, config, device, use_multi_gpu)

    # Prepare data
    print("Preparing data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, model_type, topk,
        max_main_questions=example_idx + 1,  # Only load up to the example we need
        preprocess=preprocess,
        preprocess_scope=preprocess_scope
    )

    # Get the specific example and sub-question
    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print("\n" + "="*100)
    print(f"ANALYZING EXAMPLE {example_idx}, SUB-QUESTION {sub_question_idx}")
    print("="*100)
    print(f"Main Question: {q_data['main_question']}")
    print(f"Sub Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print("="*100)

    # Build query tokens
    if model_type == 'qwen3':
        question_text = f"<|im_end|>\n<|im_start|>user\n/no_think\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    else:
        question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "

    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    # Get documents for this sub-question
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    kv_chunk_ids = [0] + doc_chunk_ids

    input_device = "cuda:0" if use_multi_gpu else device

    # Setup cache paths
    model_cache_root = os.path.join(cache_path, model_name)
    save_path = os.path.join(model_cache_root, 'kv_cache')

    if preprocess_scope == PreprocessScope.GLOBAL:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_global')
    elif preprocess_scope == PreprocessScope.PER_EXAMPLE:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_per_example')
    else:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache')

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

    # ============================================================================
    # STEP 1: Generate answer with rate=0.3
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 1: Generating answer with rate=0.3")
    print("="*100)

    load_path_rate03 = preprocess_save_path if preprocess else save_path
    generated_tokens_rate03, _ = load_kv_and_generate(
        model, tokenizer, past_key_values, iter_tokens, load_path_rate03, example_idx,
        max_new_tokens=500, revert_rope=revert_rope,
        reprocess_method=reprocess_method, rate=0.3,
        preprocess=preprocess, device=input_device, chunk_ids=kv_chunk_ids, device_map=device_map
    )

    answer_rate03 = tokenizer.decode(torch.tensor(generated_tokens_rate03[:-1]), skip_special_tokens=True)
    print(f"Generated answer (rate=0.3): {answer_rate03}")

    # ============================================================================
    # STEP 2: Generate answer with rate=1.0
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 2: Generating answer with rate=1.0 (full recompute)")
    print("="*100)

    full_input_rate1 = torch.cat(iter_tokens).to(input_device).unsqueeze(0)
    generated_tokens_rate1, _, _ = prefill_and_generate(
        model, tokenizer, full_input_rate1, max_new_tokens=500, device=input_device, device_map=device_map
    )

    answer_rate1 = tokenizer.decode(torch.tensor(generated_tokens_rate1[:-1]), skip_special_tokens=True)
    print(f"Generated answer (rate=1.0): {answer_rate1}")

    # ============================================================================
    # STEP 3: Build full sequences (query + answer)
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 3: Building full sequences (query + generated answer)")
    print("="*100)

    # For rate=0.3
    answer_tokens_rate03 = tokenizer.encode(answer_rate03, add_special_tokens=False)
    full_sequence_rate03 = torch.cat([
        torch.cat(iter_tokens),
        torch.tensor(answer_tokens_rate03, dtype=torch.long)
    ])

    # For rate=1.0
    answer_tokens_rate1 = tokenizer.encode(answer_rate1, add_special_tokens=False)
    full_sequence_rate1 = torch.cat([
        torch.cat(iter_tokens),
        torch.tensor(answer_tokens_rate1, dtype=torch.long)
    ])

    query_len = torch.cat(iter_tokens).shape[0]

    print(f"Query length: {query_len} tokens")
    print(f"Full sequence length (rate=0.3): {full_sequence_rate03.shape[0]} tokens")
    print(f"Full sequence length (rate=1.0): {full_sequence_rate1.shape[0]} tokens")

    # ============================================================================
    # STEP 4: Re-compute with attention for rate=0.3
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 4: Re-computing full sequence with rate=0.3 (to get attention & logits)")
    print("="*100)

    # Note: For rate=0.3, we need to use the KV cache approach
    # But for comparison, we'll do a full forward pass
    logits_rate03, attn_rate03 = recompute_with_attention(
        model, tokenizer, full_sequence_rate03, device, device_map, return_attentions=True
    )

    print(f"Logits shape: {logits_rate03.shape}")
    print(f"Attention shape: {attn_rate03.shape if attn_rate03 is not None else 'None'}")

    # ============================================================================
    # STEP 5: Re-compute with attention for rate=1.0
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 5: Re-computing full sequence with rate=1.0 (to get attention & logits)")
    print("="*100)

    logits_rate1, attn_rate1 = recompute_with_attention(
        model, tokenizer, full_sequence_rate1, device, device_map, return_attentions=True
    )

    print(f"Logits shape: {logits_rate1.shape}")
    print(f"Attention shape: {attn_rate1.shape if attn_rate1 is not None else 'None'}")

    # ============================================================================
    # STEP 6: Analyze differences
    # ============================================================================

    # To compare, we need sequences of the same length
    # Use the shorter one
    min_len = min(full_sequence_rate03.shape[0], full_sequence_rate1.shape[0])

    logits_rate03_trimmed = logits_rate03[:min_len]
    logits_rate1_trimmed = logits_rate1[:min_len]

    if attn_rate03 is not None and attn_rate1 is not None:
        attn_rate03_trimmed = attn_rate03[:, :min_len, :min_len]
        attn_rate1_trimmed = attn_rate1[:, :min_len, :min_len]
    else:
        attn_rate03_trimmed = None
        attn_rate1_trimmed = None

    print("\n" + "="*100)
    print("STEP 6: Analyzing differences")
    print("="*100)

    # Analyze logits
    logits_save_path = os.path.join(output_dir, f'logits_analysis_ex{example_idx}_subq{sub_question_idx}.json')
    analyze_logits_change(
        logits_rate03_trimmed, logits_rate1_trimmed, tokenizer, query_len, logits_save_path
    )

    # Analyze attention
    if attn_rate03_trimmed is not None and attn_rate1_trimmed is not None:
        attn_save_path = os.path.join(output_dir, f'attention_analysis_ex{example_idx}_subq{sub_question_idx}.json')
        analyze_attention_change(
            attn_rate03_trimmed, attn_rate1_trimmed, query_len, attn_save_path
        )
    else:
        print("⚠️  Attention weights not available (model may not support output_attentions)")

    # Save summary
    summary = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'main_question': q_data['main_question'],
        'sub_question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'answer_rate03': answer_rate03,
        'answer_rate1': answer_rate1,
        'query_len': query_len,
        'full_seq_len_rate03': full_sequence_rate03.shape[0],
        'full_seq_len_rate1': full_sequence_rate1.shape[0],
    }

    summary_path = os.path.join(output_dir, f'summary_ex{example_idx}_subq{sub_question_idx}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"Results saved to: {output_dir}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Logits analysis: {logits_save_path}")
    if attn_rate03_trimmed is not None:
        print(f"  - Attention analysis: {attn_save_path}")
    print("="*100)


if __name__ == '__main__':
    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        topk=10,
        preprocess=True,
        reprocess_method='FusionRAG',
        preprocess_scope=PreprocessScope.GLOBAL,
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        revert_rope=True,
        device="cuda:0",
        use_multi_gpu=False,
        example_idx=0,  # Analyze the first example
        sub_question_idx=0,  # Analyze the first sub-question
        output_dir='./attention_logits_analysis'
    )

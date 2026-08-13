#!/usr/bin/env python3
"""
Compare attention and logits between rate=0.3 and rate=1.0 KV cache

Correct workflow:
1. Prepare two types of KV cache (system + documents)
2. Use rate=0.3 KV cache to generate answer A (autoregressive)
3. Use rate=1.0 KV cache to generate answer B (autoregressive)
4. Re-compute with rate=0.3: prefill "query + answer A", get attention/logits
5. Re-compute with rate=1.0: prefill "query + answer B", get attention/logits
6. Compare overlapping parts (query is same, answer compare min length)
"""

import json
import os
import sys
import torch
import shutil
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoConfig

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import (
    prefill_and_save_kv_cache,
    prefill_and_generate,
    prefill_with_cache_and_save_preprocess,
    find_group_and_index,
)
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect import load_model, prepare_reflect_data, PreprocessScope


def generate_with_kvcache(
    model,
    tokenizer,
    past_key_values,
    passages,
    load_path,
    example_id,
    chunk_ids,
    max_new_tokens=100,
    revert_rope=False,
    device="cuda:0",
    device_map=None
):
    """
    Generate answer autoregressively using KV cache

    Returns:
        generated_tokens: List of generated token IDs
        query_len: Length of the query part
    """
    from ktransformers.util.utils import rotate_half

    input_device = "cuda:0" if device_map is not None else device

    passages_len = [passage.shape[0] for passage in passages]
    system_len = passages[0].shape[0]

    # Load KV cache for documents
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    past_len = 0
    key_cache = []
    value_cache = []

    # Load all document caches
    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]

        chunk_key_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_key.pt', weights_only=True).to('cpu')
        chunk_value_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_value.pt', weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)

    # Copy to past_key_values
    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]
        key_cache[idx] = key_cache[idx].to(input_device)
        chunk_key_cache = key_cache[idx]
        chunk_value_cache = value_cache[idx].to(input_device)

        if revert_rope and chunk_id > 0:
            position_ids = torch.full((1, chunk_key_cache[0].shape[2]), past_len - system_len, device=input_device)
            try:
                cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[0], position_ids)
            except:
                cos, sin = model.model.rotary_emb(chunk_key_cache[0], position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)

        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_key_cache[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_value_cache[layer_idx])
            past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len

    print(f"  Loaded {len(passages)-1} document caches, total past_len={past_len}")

    # Prefill query
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
    if query_prefix_len >= len(passages[-1]):
        query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0])) + 1

    query_tokens = passages[-1][query_prefix_len:]
    query_len = query_tokens.shape[0]

    print(f"  Query length: {query_len} tokens")

    # Prefill query
    cache_position = torch.arange(past_len, past_len + query_len, device=input_device)
    inputs = query_tokens.unsqueeze(0).to(input_device)

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
        outputs = model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True
        )
        logits = outputs.logits[0, -1, :]

    past_len += query_len

    # Autoregressive generation
    generated_tokens = []
    print(f"  Generating up to {max_new_tokens} tokens...")

    # Get all possible EOS tokens for Qwen models
    eos_tokens = [tokenizer.eos_token_id]
    if hasattr(tokenizer, 'im_end_id'):
        eos_tokens.append(tokenizer.im_end_id)
    # For chat models, also check for <|im_end|>
    im_end_token = tokenizer.encode('<|im_end|>', add_special_tokens=False)
    if len(im_end_token) == 1:
        eos_tokens.append(im_end_token[0])

    print(f"  EOS tokens: {eos_tokens}")

    stop_reason = None

    for step in range(max_new_tokens):
        # Sample next token (greedy)
        next_token = torch.argmax(logits).item()
        generated_tokens.append(next_token)

        token_text = tokenizer.decode([next_token])

        if (step + 1) % 10 == 0 or step < 10:
            print(f"    Step {step + 1}: token_id={next_token}, text='{token_text}'")

        # Check for EOS and stop
        if next_token in eos_tokens:
            stop_reason = f"EOS token (id={next_token})"
            break

        # Prepare next input
        cache_position = torch.tensor([past_len], device=input_device)
        input_token = torch.tensor([[next_token]], dtype=torch.long, device=input_device)

        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(input_token).to(input_device)
            outputs = model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True
            )
            logits = outputs.logits[0, 0, :]

        past_len += 1

    if stop_reason:
        print(f"  Stopped: {stop_reason}")
    else:
        print(f"  Stopped: reached max_new_tokens ({max_new_tokens})")

    print(f"  Total generated: {len(generated_tokens)} tokens")
    print(f"  Generated tokens: {generated_tokens[:20]}")  # Show first 20

    return generated_tokens, query_len


def prefill_and_extract(
    model,
    tokenizer,
    past_key_values,
    passages,
    load_path,
    example_id,
    chunk_ids,
    answer_tokens: List[int],
    revert_rope=False,
    device="cuda:0",
    device_map=None
):
    """
    Prefill query + answer and extract logits/attention at each position

    Returns:
        logits_list: List of logits at each position [seq_len, vocab_size]
        attention_list: List of attention at each position (K-based similarity)
    """
    from ktransformers.util.utils import rotate_half

    input_device = "cuda:0" if device_map is not None else device

    system_len = passages[0].shape[0]

    # Load KV cache
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    past_len = 0
    key_cache = []
    value_cache = []

    # Load document caches
    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]

        chunk_key_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_key.pt', weights_only=True).to('cpu')
        chunk_value_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_value.pt', weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)

    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]
        key_cache[idx] = key_cache[idx].to(input_device)
        chunk_key_cache = key_cache[idx]
        chunk_value_cache = value_cache[idx].to(input_device)

        if revert_rope and chunk_id > 0:
            position_ids = torch.full((1, chunk_key_cache[0].shape[2]), past_len - system_len, device=input_device)
            try:
                cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[0], position_ids)
            except:
                cos, sin = model.model.rotary_emb(chunk_key_cache[0], position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)

        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_key_cache[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_value_cache[layer_idx])
            past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len

    # Get query tokens
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
    if query_prefix_len >= len(passages[-1]):
        query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0])) + 1

    query_tokens = passages[-1][query_prefix_len:].tolist()

    # Combine query + answer
    full_sequence = query_tokens + answer_tokens

    print(f"  Prefilling {len(full_sequence)} tokens (query={len(query_tokens)}, answer={len(answer_tokens)})")

    # Get model config
    num_layers = len(model.model.layers)
    num_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    head_dim = model.config.hidden_size // num_heads
    num_key_value_groups = num_heads // num_key_value_heads

    logits_list = []
    attention_list = []

    # Process token by token to get logits and attention at each position
    for pos, token_id in enumerate(full_sequence):
        cache_position = torch.tensor([past_len], device=input_device)
        input_token = torch.tensor([[token_id]], dtype=torch.long, device=input_device)

        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(input_token).to(input_device)
            outputs = model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True
            )

            logits = outputs.logits[0, 0, :]
            logits_list.append(logits.cpu())

            # Extract K-based attention from last layer
            last_layer_idx = num_layers - 1
            key_cache_layer = past_key_values.key_cache[last_layer_idx][:, :, :past_len+1, :]

            current_k = key_cache_layer[:, :, past_len:past_len+1, :]
            similarities = torch.matmul(current_k, key_cache_layer.transpose(2, 3)) / (head_dim ** 0.5)
            pseudo_attn = torch.nn.functional.softmax(similarities, dim=-1)

            if num_key_value_groups > 1:
                pseudo_attn = pseudo_attn.repeat_interleave(num_key_value_groups, dim=1)

            attention_list.append(pseudo_attn[0, :, 0, :].cpu())

        past_len += 1

        if (pos + 1) % 20 == 0:
            print(f"    Processed {pos + 1}/{len(full_sequence)} tokens")

    return logits_list, attention_list, len(query_tokens)


def compare_results(
    logits_rate03: List[torch.Tensor],
    logits_rate1: List[torch.Tensor],
    attn_rate03: List[torch.Tensor],
    attn_rate1: List[torch.Tensor],
    tokens_rate03: List[int],
    tokens_rate1: List[int],
    query_len: int,
    tokenizer,
    save_dir: str
):
    """
    Compare overlapping parts of logits and attention
    """
    # Find overlapping length
    min_len = min(len(logits_rate03), len(logits_rate1))

    print("\n" + "="*100)
    print(f"COMPARISON (Query len={query_len}, Rate0.3 total={len(logits_rate03)}, Rate1.0 total={len(logits_rate1)})")
    print(f"Comparing overlapping {min_len} positions")
    print("="*100)

    results = []

    for pos in range(min_len):
        logits_03 = logits_rate03[pos]
        logits_1 = logits_rate1[pos]

        # Get top-1 tokens
        probs_03 = torch.softmax(logits_03, dim=-1)
        probs_1 = torch.softmax(logits_1, dim=-1)

        top1_token_03 = torch.argmax(probs_03).item()
        top1_token_1 = torch.argmax(probs_1).item()

        top1_prob_03 = probs_03[top1_token_03].item()
        top1_prob_1 = probs_1[top1_token_1].item()

        token_text_03 = tokenizer.decode([top1_token_03])
        token_text_1 = tokenizer.decode([top1_token_1])

        # Get actual tokens at this position
        actual_token_03 = tokens_rate03[pos] if pos < len(tokens_rate03) else None
        actual_token_1 = tokens_rate1[pos] if pos < len(tokens_rate1) else None

        actual_text_03 = tokenizer.decode([actual_token_03]) if actual_token_03 is not None else "N/A"
        actual_text_1 = tokenizer.decode([actual_token_1]) if actual_token_1 is not None else "N/A"

        # Position type
        if pos < query_len:
            pos_type = "QUERY"
        else:
            pos_type = f"ANSWER (step {pos - query_len + 1})"

        top1_changed = top1_token_03 != top1_token_1

        print(f"\n--- Position {pos + 1} ({pos_type}) ---")
        print(f"  Actual tokens:")
        print(f"    Rate=0.3: '{actual_text_03}' (ID={actual_token_03})")
        print(f"    Rate=1.0: '{actual_text_1}' (ID={actual_token_1})")

        if top1_changed:
            print(f"  ⚠️  PREDICTED TOP-1 DIFFERENT:")
            print(f"    Rate=1.0 predicts: '{token_text_1}' (prob={top1_prob_1:.6f})")
            print(f"    Rate=0.3 predicts: '{token_text_03}' (prob={top1_prob_03:.6f})")

            prob_A_in_03 = probs_03[top1_token_1].item()
            prob_B_in_1 = probs_1[top1_token_03].item()

            print(f"  Probability changes:")
            print(f"    '{token_text_1}': {top1_prob_1:.6f} (rate=1.0) → {prob_A_in_03:.6f} (rate=0.3), Δ={prob_A_in_03 - top1_prob_1:+.6f}")
            print(f"    '{token_text_03}': {prob_B_in_1:.6f} (rate=1.0) → {top1_prob_03:.6f} (rate=0.3), Δ={top1_prob_03 - prob_B_in_1:+.6f}")
        else:
            print(f"  ✓ PREDICTED TOP-1 SAME: '{token_text_1}' (prob: {top1_prob_1:.6f} → {top1_prob_03:.6f}, Δ={top1_prob_03 - top1_prob_1:+.6f})")

        # Logits L2
        logits_l2 = torch.norm(logits_03 - logits_1, p=2).item()
        probs_l2 = torch.norm(probs_03 - probs_1, p=2).item()
        print(f"  Logits L2: {logits_l2:.4f}, Probs L2: {probs_l2:.6f}")

        # Attention L2
        if pos < len(attn_rate03) and pos < len(attn_rate1):
            attn_03 = attn_rate03[pos]
            attn_1 = attn_rate1[pos]

            min_seq = min(attn_03.shape[-1], attn_1.shape[-1])
            attn_l2 = torch.norm(attn_03[:, :min_seq] - attn_1[:, :min_seq], p=2).item()
            print(f"  Attention L2: {attn_l2:.4f}")

        results.append({
            'position': pos + 1,
            'position_type': pos_type,
            'actual_token_rate03': actual_text_03,
            'actual_token_rate1': actual_text_1,
            'predicted_token_rate03': token_text_03,
            'predicted_token_rate1': token_text_1,
            'top1_changed': top1_changed,
            'prob_rate03': top1_prob_03,
            'prob_rate1': top1_prob_1,
            'logits_l2': logits_l2,
            'probs_l2': probs_l2,
        })

    # Save results
    with open(os.path.join(save_dir, 'comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save detailed attention analysis
    print("\n" + "="*100)
    print("ATTENTION DETAILED ANALYSIS")
    print("="*100)

    attention_results = {
        'query_len': query_len,
        'total_positions': min_len,
        'positions': []
    }

    # Separate query and answer parts
    query_attn_l2 = []
    answer_attn_l2 = []

    for pos in range(min_len):
        if pos < len(attn_rate03) and pos < len(attn_rate1):
            attn_03 = attn_rate03[pos]
            attn_1 = attn_rate1[pos]

            min_seq = min(attn_03.shape[-1], attn_1.shape[-1])
            attn_03_trimmed = attn_03[:, :min_seq]
            attn_1_trimmed = attn_1[:, :min_seq]

            overall_l2 = torch.norm(attn_03_trimmed - attn_1_trimmed, p=2).item()

            # Per-head L2
            num_heads = attn_03_trimmed.shape[0]
            head_l2_distances = []
            for head_idx in range(num_heads):
                l2_dist = torch.norm(attn_03_trimmed[head_idx] - attn_1_trimmed[head_idx], p=2).item()
                head_l2_distances.append(l2_dist)

            pos_type = "QUERY" if pos < query_len else f"ANSWER_{pos - query_len + 1}"

            attention_results['positions'].append({
                'position': pos + 1,
                'type': pos_type,
                'overall_l2': overall_l2,
                'head_l2_distances': head_l2_distances,
            })

            if pos < query_len:
                query_attn_l2.append(overall_l2)
            else:
                answer_attn_l2.append(overall_l2)

    # Summary statistics
    if query_attn_l2:
        print(f"\nQuery part attention L2 statistics:")
        print(f"  Mean: {np.mean(query_attn_l2):.4f}")
        print(f"  Std: {np.std(query_attn_l2):.4f}")
        print(f"  Min: {np.min(query_attn_l2):.4f}")
        print(f"  Max: {np.max(query_attn_l2):.4f}")

    if answer_attn_l2:
        print(f"\nAnswer part attention L2 statistics:")
        print(f"  Mean: {np.mean(answer_attn_l2):.4f}")
        print(f"  Std: {np.std(answer_attn_l2):.4f}")
        print(f"  Min: {np.min(answer_attn_l2):.4f}")
        print(f"  Max: {np.max(answer_attn_l2):.4f}")

    attention_results['summary'] = {
        'query_attention_l2': {
            'mean': float(np.mean(query_attn_l2)) if query_attn_l2 else 0,
            'std': float(np.std(query_attn_l2)) if query_attn_l2 else 0,
            'min': float(np.min(query_attn_l2)) if query_attn_l2 else 0,
            'max': float(np.max(query_attn_l2)) if query_attn_l2 else 0,
        },
        'answer_attention_l2': {
            'mean': float(np.mean(answer_attn_l2)) if answer_attn_l2 else 0,
            'std': float(np.std(answer_attn_l2)) if answer_attn_l2 else 0,
            'min': float(np.min(answer_attn_l2)) if answer_attn_l2 else 0,
            'max': float(np.max(answer_attn_l2)) if answer_attn_l2 else 0,
        }
    }

    with open(os.path.join(save_dir, 'attention_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(attention_results, f, indent=2, ensure_ascii=False)

    # Save raw attention tensors
    torch.save({
        'attn_rate03': [a.cpu() if a is not None else None for a in attn_rate03],
        'attn_rate1': [a.cpu() if a is not None else None for a in attn_rate1],
    }, os.path.join(save_dir, 'attention_tensors.pt'))

    print(f"\n{'='*100}")
    print(f"Results saved to {save_dir}/")
    print(f"  - comparison.json: Per-position logits and attention comparison")
    print(f"  - attention_analysis.json: Detailed attention statistics")
    print(f"  - attention_tensors.pt: Raw attention tensors for visualization")
    print(f"{'='*100}")


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
    example_idx=0,
    sub_question_idx=0,
    max_new_tokens=100,
    output_dir='./kvcache_final_analysis'
):
    """
    Main function for correct KV cache comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading tokenizer and config from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    print(f"Loading {model_type} model...")
    model, device_map = load_model(model_type, model_path, config, device, use_multi_gpu)

    # Prepare data
    print("Preparing data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, model_type, topk,
        max_main_questions=example_idx + 1,
        preprocess=preprocess,
        preprocess_scope=preprocess_scope
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print("\n" + "="*100)
    print(f"ANALYZING EXAMPLE {example_idx}, SUB-QUESTION {sub_question_idx}")
    print("="*100)
    print(f"Main Question: {q_data['main_question']}")
    print(f"Sub Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print("="*100)

    # Build query
    if model_type == 'qwen3':
        question_text = f"<|im_end|>\n<|im_start|>user\n/no_think\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    else:
        question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "

    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    # Get documents
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    kv_chunk_ids = [0] + doc_chunk_ids

    input_device = "cuda:0" if use_multi_gpu else device

    # Setup paths
    model_cache_root = os.path.join(cache_path, model_name)
    save_path = os.path.join(model_cache_root, 'kv_cache')

    if preprocess_scope == PreprocessScope.GLOBAL:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_global')
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

    # STEP 1: Generate with rate=0.3
    print("\n" + "="*100)
    print("STEP 1: Generating answer with rate=0.3 KV cache")
    print("="*100)

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    answer_tokens_rate03, query_len = generate_with_kvcache(
        model, tokenizer, past_key_values, iter_tokens,
        preprocess_save_path, example_idx, kv_chunk_ids,
        max_new_tokens=max_new_tokens,
        revert_rope=revert_rope,
        device=input_device,
        device_map=device_map
    )

    answer_text_rate03 = tokenizer.decode(answer_tokens_rate03, skip_special_tokens=True)
    print(f"\n  Generated answer (rate=0.3): {answer_text_rate03}")

    # STEP 2: Generate with rate=1.0 (use prefill_and_generate for consistency)
    print("\n" + "="*100)
    print("STEP 2: Generating answer with rate=1.0 (full forward)")
    print("="*100)

    full_input = torch.cat(iter_tokens).to(input_device).unsqueeze(0)
    generated_tokens_full, _, _ = prefill_and_generate(
        model, tokenizer, full_input, max_new_tokens=max_new_tokens, device=input_device, device_map=device_map
    )

    # Convert list of tensors to list of integers
    if isinstance(generated_tokens_full, list) and len(generated_tokens_full) > 0:
        if isinstance(generated_tokens_full[0], torch.Tensor):
            generated_tokens_full = [t.item() for t in generated_tokens_full]
    elif isinstance(generated_tokens_full, torch.Tensor):
        generated_tokens_full = generated_tokens_full.squeeze().tolist()

    answer_tokens_rate1 = generated_tokens_full[:-1] if len(generated_tokens_full) > 0 else []
    answer_text_rate1 = tokenizer.decode(answer_tokens_rate1, skip_special_tokens=True)
    print(f"\n  Generated answer (rate=1.0): {answer_text_rate1}")
    print(f"  Generated {len(answer_tokens_rate1)} tokens")

    # STEP 3: Re-compute with rate=0.3
    print("\n" + "="*100)
    print("STEP 3: Re-computing with rate=0.3 (query + answer_rate03)")
    print("="*100)

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    logits_rate03, attn_rate03, _ = prefill_and_extract(
        model, tokenizer, past_key_values, iter_tokens,
        preprocess_save_path, example_idx, kv_chunk_ids,
        answer_tokens_rate03,
        revert_rope=revert_rope,
        device=input_device,
        device_map=device_map
    )

    # STEP 4: Re-compute with rate=1.0
    print("\n" + "="*100)
    print("STEP 4: Re-computing with rate=1.0 (query + answer_rate1)")
    print("="*100)

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    logits_rate1, attn_rate1, _ = prefill_and_extract(
        model, tokenizer, past_key_values, iter_tokens,
        save_path, example_idx, kv_chunk_ids,
        answer_tokens_rate1,
        revert_rope=False,
        device=input_device,
        device_map=device_map
    )

    # STEP 5: Compare
    print("\n" + "="*100)
    print("STEP 5: Comparing overlapping parts")
    print("="*100)

    # Get query tokens for reference
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(iter_tokens[-1]).split('Question: ')[0]))
    if query_prefix_len >= len(iter_tokens[-1]):
        query_prefix_len = len(tokenizer.encode(tokenizer.decode(iter_tokens[-1]).split('Question：')[0])) + 1
    query_tokens_list = iter_tokens[-1][query_prefix_len:].tolist()

    # Full token sequences
    full_tokens_rate03 = query_tokens_list + answer_tokens_rate03
    full_tokens_rate1 = query_tokens_list + answer_tokens_rate1

    compare_results(
        logits_rate03, logits_rate1,
        attn_rate03, attn_rate1,
        full_tokens_rate03, full_tokens_rate1,
        len(query_tokens_list),
        tokenizer,
        output_dir
    )

    # Save summary
    summary = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'main_question': q_data['main_question'],
        'sub_question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'answer_rate03': answer_text_rate03,
        'answer_rate1': answer_text_rate1,
        'query_len': len(query_tokens_list),
        'answer_len_rate03': len(answer_tokens_rate03),
        'answer_len_rate1': len(answer_tokens_rate1),
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nGenerated Answers:")
    print(f"  Rate=0.3: {answer_text_rate03}")
    print(f"  Rate=1.0: {answer_text_rate1}")
    print(f"  Ground Truth: {sub_q_info['answer']}")
    print(f"\nResults saved to: {output_dir}")
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
        example_idx=4,
        sub_question_idx=1,
        max_new_tokens=100,
        output_dir='./kvcache_final_analysis'
    )

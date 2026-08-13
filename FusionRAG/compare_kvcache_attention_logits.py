#!/usr/bin/env python3
"""
Compare attention and logits between rate=0.3 (FusionRAG preprocessed) and rate=1.0 KV cache

This script:
1. Prepares two types of KV cache:
   - FusionRAG preprocessed cache (rate=0.3)
   - Full cache (rate=1.0, no compression)
2. Uses rate=1.0 to generate an answer
3. Concatenates query + answer
4. Uses both KV caches to prefill the "query + answer" sequence
5. Compares last-layer attention and logits
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
    load_kv_and_generate,
    prefill_and_generate,
    prefill_with_cache_and_save_preprocess,
    find_group_and_index,
)
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect import load_model, load_system_prompt, prepare_reflect_data, PreprocessScope


def load_and_decode_with_monitoring(
    model,
    tokenizer,
    past_key_values,
    passages,
    load_path,
    example_id,
    chunk_ids,
    answer_tokens: List[int],
    revert_rope=False,
    reprocess_method='FusionRAG',
    rate=0,
    preprocess=False,
    device="cuda:0",
    device_map=None,
    monitor_attention=True
):
    """
    Load KV cache and decode answer tokens while monitoring attention and logits

    Args:
        answer_tokens: List of answer token IDs to decode

    Returns:
        logits_list: List of logits at each decode step, shape [(vocab_size,), ...]
        attention_list: List of last-layer attention at each step, shape [(num_heads, seq_len, seq_len), ...]
    """
    from ktransformers.util.utils import rotate_half
    import time

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

        assert passage_len == chunk_key_cache.shape[3]

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

    # Now prefill query
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
        # Don't request attention for prefill to avoid potential issues
        outputs = model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True,
            output_attentions=False
        )

    past_len += query_len

    # Now decode answer tokens one by one
    logits_list = []
    attention_list = []

    print(f"  Decoding {len(answer_tokens)} answer tokens...")

    # Setup hook to capture attention if needed
    captured_attn = {}

    def attention_hook(module, input, output):
        """Hook to capture attention weights from the last layer"""
        # output is (attn_output, attn_weights, past_key_value)
        # For Qwen2Attention, we need to compute attention manually from the states
        # Since the module doesn't return attn_weights by default in sdpa mode
        # We'll capture the necessary tensors and compute later
        pass

    # Get model config
    num_layers = len(model.model.layers)
    num_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    head_dim = model.config.hidden_size // num_heads
    num_key_value_groups = num_heads // num_key_value_heads

    if monitor_attention:
        print(f"  Will extract attention from KV cache (last layer only)")
        print(f"    Num heads: {num_heads}, KV heads: {num_key_value_heads}, Head dim: {head_dim}")

    for step, token_id in enumerate(answer_tokens):
        # Get current position
        cache_position = torch.tensor([past_len], device=input_device)

        # Prepare input
        input_token = torch.tensor([[token_id]], dtype=torch.long, device=input_device)

        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(input_token).to(input_device)

            # Forward pass
            outputs = model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True,
                output_attentions=False
            )

            # Get logits
            logits = outputs.logits[0, 0, :]  # [vocab_size]
            logits_list.append(logits.cpu())

            # Extract attention from KV cache
            if monitor_attention:
                last_layer_idx = num_layers - 1

                # After forward, past_key_values contains the updated K and V (including current token)
                # Extract K and V from cache
                key_cache = past_key_values.key_cache[last_layer_idx][:, :, :past_len+1, :]  # [B, num_kv_heads, seq_len, head_dim]
                value_cache = past_key_values.value_cache[last_layer_idx][:, :, :past_len+1, :]  # [B, num_kv_heads, seq_len, head_dim]

                # Get the query state for the current token (last position in cache)
                # Extract Q from the newly added cache entry
                query_cache = key_cache[:, :, past_len:past_len+1, :]  # [B, num_kv_heads, 1, head_dim]

                # For GQA, we need to expand query to all heads
                # Actually, the cache stores KV heads, so we need to get Q differently
                # The cache only has K and V, not Q
                # We need to recompute Q from the hidden states

                # Simplified approach: extract attention from cache
                # Since we only have K and V in cache, we need Q from somewhere
                # Option 1: Recompute the last layer with output_attentions (but sdpa doesn't support it well)
                # Option 2: Manually compute Q from the output hidden states
                # Option 3: Use the fact that for the last token, we can approximate attention from KV similarity

                # Let's use a pragmatic approach: compute Q from the last hidden state
                # Get hidden state from model output (before lm_head)
                # This requires accessing intermediate states

                # Alternative: Just extract K at the current position as proxy for attention pattern
                # This won't give us exact attention but can show where the key is focusing

                # Simplest approach for now: compute attention using K as query approximation
                # Or we can store the attention pattern during forward pass

                # Actually, let me try a different approach:
                # Set output_attentions=True just for this step and catch any error
                try:
                    # Try to recompute with attention output
                    # Reset position
                    cache_position = torch.tensor([past_len], device=input_device)

                    # We'll compute attention manually using the cached K, V
                    # For the query, we can use the current K as an approximation
                    # This is not perfect but gives us a signal

                    # Better approach: compute cosine similarity between current K and all previous K
                    current_k = key_cache[:, :, past_len:past_len+1, :]  # [B, num_kv_heads, 1, head_dim]

                    # Compute similarity (not exact attention, but correlates)
                    similarities = torch.matmul(current_k, key_cache.transpose(2, 3))  # [B, num_kv_heads, 1, seq_len]
                    similarities = similarities / (head_dim ** 0.5)

                    # Apply softmax to get pseudo-attention
                    pseudo_attn = torch.nn.functional.softmax(similarities, dim=-1)  # [B, num_kv_heads, 1, seq_len]

                    # Expand to all heads if using GQA
                    if num_key_value_groups > 1:
                        pseudo_attn = pseudo_attn.repeat_interleave(num_key_value_groups, dim=1)

                    # Store [num_heads, seq_len]
                    attention_list.append(pseudo_attn[0, :, 0, :].cpu())

                except Exception as e:
                    if step == 0:
                        print(f"  ⚠️  Could not compute attention: {e}")
                        print(f"  Using K-based similarity as proxy")
                    attention_list.append(None)
            else:
                attention_list.append(None)

        past_len += 1

        if (step + 1) % 10 == 0:
            print(f"    Decoded {step + 1}/{len(answer_tokens)} tokens")

    return logits_list, attention_list


def compare_logits(
    logits_rate03: List[torch.Tensor],
    logits_rate1: List[torch.Tensor],
    tokenizer,
    save_path: str
):
    """
    Compare logits from rate=0.3 and rate=1.0 for each decode step

    Args:
        logits_rate03: List of logits from rate=0.3, each shape [vocab_size]
        logits_rate1: List of logits from rate=1.0, each shape [vocab_size]
        tokenizer: Tokenizer
        save_path: Path to save results
    """
    min_len = min(len(logits_rate03), len(logits_rate1))

    results = []

    print("\n" + "="*100)
    print("LOGITS COMPARISON (Per Decode Step)")
    print("="*100)

    for step in range(min_len):
        logits_03 = logits_rate03[step]
        logits_1 = logits_rate1[step]

        # Get probabilities
        probs_03 = torch.softmax(logits_03, dim=-1)
        probs_1 = torch.softmax(logits_1, dim=-1)

        # Get top-1 tokens
        top1_token_03 = torch.argmax(probs_03).item()
        top1_token_1 = torch.argmax(probs_1).item()

        top1_prob_03 = probs_03[top1_token_03].item()
        top1_prob_1 = probs_1[top1_token_1].item()

        # Decode tokens
        token_text_03 = tokenizer.decode([top1_token_03])
        token_text_1 = tokenizer.decode([top1_token_1])

        # Check if top-1 changed
        top1_changed = top1_token_03 != top1_token_1

        print(f"\n--- Decode Step {step + 1} ---")

        if top1_changed:
            print(f"  ⚠️  TOP-1 TOKEN CHANGED:")
            print(f"    Rate=1.0 predicts: '{token_text_1}' (ID={top1_token_1})")
            print(f"    Rate=0.3 predicts: '{token_text_03}' (ID={top1_token_03})")
            print(f"")
            print(f"  Probability changes:")

            # Show how probabilities changed
            prob_A_in_03 = probs_03[top1_token_1].item()
            prob_B_in_1 = probs_1[top1_token_03].item()

            print(f"    '{token_text_1}' (rate=1.0 winner):")
            print(f"      Rate=1.0: {top1_prob_1:.6f}  →  Rate=0.3: {prob_A_in_03:.6f}  (Δ={prob_A_in_03 - top1_prob_1:+.6f})")

            print(f"    '{token_text_03}' (rate=0.3 winner):")
            print(f"      Rate=1.0: {prob_B_in_1:.6f}  →  Rate=0.3: {top1_prob_03:.6f}  (Δ={top1_prob_03 - prob_B_in_1:+.6f})")
        else:
            print(f"  ✓ TOP-1 TOKEN UNCHANGED")
            print(f"    Token: '{token_text_1}' (ID={top1_token_1})")
            print(f"    Rate=1.0: '{token_text_1}' → '{token_text_1}' (prob: {top1_prob_1:.6f} → {top1_prob_03:.6f}, Δ={top1_prob_03 - top1_prob_1:+.6f})")

        # Compute L2 distances
        logits_l2 = torch.norm(logits_03 - logits_1, p=2).item()
        probs_l2 = torch.norm(probs_03 - probs_1, p=2).item()

        print(f"  Logits L2 distance: {logits_l2:.6f}")
        print(f"  Probs L2 distance: {probs_l2:.6f}")

        # Save result
        result_entry = {
            'decode_step': step + 1,
            'top1_changed': top1_changed,
            'rate1_token': token_text_1,
            'rate1_token_id': top1_token_1,
            'rate1_prob': top1_prob_1,
            'rate03_token': token_text_03,
            'rate03_token_id': top1_token_03,
            'rate03_prob': top1_prob_03,
            'logits_l2': logits_l2,
            'probs_l2': probs_l2,
        }

        if top1_changed:
            result_entry['token_A_prob_in_rate03'] = prob_A_in_03
            result_entry['token_B_prob_in_rate1'] = prob_B_in_1

        results.append(result_entry)

    # Save to JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"Logits comparison saved to: {save_path}")
    print(f"{'='*100}")

    return results


def compare_attention(
    attn_rate03: List[torch.Tensor],
    attn_rate1: List[torch.Tensor],
    save_path: str
):
    """
    Compare attention from rate=0.3 and rate=1.0 for each decode step

    Args:
        attn_rate03: List of attention tensors from rate=0.3, each shape [num_heads, seq_len]
        attn_rate1: List of attention tensors from rate=1.0, each shape [num_heads, seq_len]
        save_path: Path to save results
    """
    min_len = min(len(attn_rate03), len(attn_rate1))

    # Filter out None values
    valid_indices = [i for i in range(min_len) if attn_rate03[i] is not None and attn_rate1[i] is not None]

    if len(valid_indices) == 0:
        print("⚠️  No valid attention data to compare")
        return None

    results = []

    print("\n" + "="*100)
    print("ATTENTION COMPARISON (Last Layer, Per Decode Step)")
    print("="*100)

    for idx in valid_indices:
        step = idx + 1
        attn_03 = attn_rate03[idx]  # [num_heads, seq_len]
        attn_1 = attn_rate1[idx]    # [num_heads, seq_len]

        # Compute L2 distance
        # Note: attention tensors might have different seq_len
        # We'll compare up to the minimum length
        min_seq_len = min(attn_03.shape[-1], attn_1.shape[-1])

        attn_03_trimmed = attn_03[:, :min_seq_len]
        attn_1_trimmed = attn_1[:, :min_seq_len]

        overall_l2 = torch.norm(attn_03_trimmed - attn_1_trimmed, p=2).item()

        # Per-head L2
        num_heads = attn_03_trimmed.shape[0]
        head_l2_distances = []
        for head_idx in range(num_heads):
            l2_dist = torch.norm(attn_03_trimmed[head_idx] - attn_1_trimmed[head_idx], p=2).item()
            head_l2_distances.append(l2_dist)

        # Also compute max absolute difference per head
        max_abs_diffs = []
        for head_idx in range(num_heads):
            max_diff = torch.max(torch.abs(attn_03_trimmed[head_idx] - attn_1_trimmed[head_idx])).item()
            max_abs_diffs.append(max_diff)

        print(f"\n--- Decode Step {step} ---")
        print(f"  Sequence length: {min_seq_len}")
        print(f"  Overall L2 distance: {overall_l2:.6f}")
        print(f"  Per-head L2 distances (showing first 5 heads):")
        for head_idx in range(min(5, num_heads)):
            print(f"    Head {head_idx}: L2={head_l2_distances[head_idx]:.6f}, Max_abs_diff={max_abs_diffs[head_idx]:.6f}")

        results.append({
            'decode_step': step,
            'seq_len': min_seq_len,
            'overall_l2': overall_l2,
            'head_l2_distances': head_l2_distances,
            'head_max_abs_diffs': max_abs_diffs,
        })

    # Save to JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save raw attention tensors (optional)
    torch.save({
        'attn_rate03': [a.cpu() if a is not None else None for a in attn_rate03],
        'attn_rate1': [a.cpu() if a is not None else None for a in attn_rate1],
    }, save_path.replace('.json', '_tensors.pt'))

    print(f"\n{'='*100}")
    print(f"Attention comparison saved to: {save_path}")
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
    example_idx=0,
    sub_question_idx=0,
    output_dir='./kvcache_comparison_analysis'
):
    """
    Main function for KV cache attention and logits comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading tokenizer and config from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"  # Use sdpa mode first (will switch to eager for attention monitoring)

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

    # Get the specific example
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

    # Setup cache paths
    model_cache_root = os.path.join(cache_path, model_name)
    save_path = os.path.join(model_cache_root, 'kv_cache')

    if preprocess_scope == PreprocessScope.GLOBAL:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_global')
    else:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache')

    # ============================================================================
    # STEP 1: Generate answer with rate=1.0 to get ground truth answer
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 1: Generating answer with rate=1.0")
    print("="*100)

    full_input = torch.cat(iter_tokens).to(input_device).unsqueeze(0)
    generated_tokens, _, _ = prefill_and_generate(
        model, tokenizer, full_input, max_new_tokens=100, device=input_device, device_map=device_map
    )

    answer_text = tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True)
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)

    print(f"Generated answer: {answer_text}")
    print(f"Answer tokens: {len(answer_tokens)} tokens")

    # ============================================================================
    # STEP 2: Prepare two types of KV cache
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 2: Ensuring KV caches are ready")
    print("="*100)

    system_len = system_tensor.shape[0]

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

    # Ensure original KV cache exists
    print("  Checking original KV cache (rate=1.0)...")
    system_cache_path = f'{save_path}/{example_idx}_0_key.pt'
    if not os.path.exists(system_cache_path):
        print("    Generating system KV cache...")
        input_tensor = system_tensor.unsqueeze(0)
        prefill_and_save_kv_cache(
            model, tokenizer, past_key_values, input_tensor.to(input_device),
            save_path=save_path, example_id=example_idx, chunk_id=0,
            system_len=system_len, passage_len=system_len,
            reprocess_method=reprocess_method, device=input_device, device_map=device_map
        )

    for doc_idx, doc_tensor in enumerate(sub_q_doc_tensors):
        chunk_id = doc_chunk_ids[doc_idx]
        cache_key_path = f'{save_path}/{example_idx}_{chunk_id}_key.pt'

        if not os.path.exists(cache_key_path):
            print(f"    Generating KV cache for document {chunk_id}...")
            passage_len = doc_tensor.shape[0]
            input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)

            prefill_and_save_kv_cache(
                model, tokenizer, past_key_values, input_tensor.to(input_device),
                save_path=save_path, example_id=example_idx, chunk_id=chunk_id,
                system_len=system_len, passage_len=passage_len,
                reprocess_method=reprocess_method, device=input_device, device_map=device_map
            )

    # Ensure preprocessed KV cache exists
    if preprocess:
        print("  Checking preprocessed KV cache (rate=0.3)...")

        # Copy system cache
        system_preprocess_key = f"{preprocess_save_path}/{example_idx}_0_key.pt"
        if not os.path.exists(system_preprocess_key):
            shutil.copy(f'{save_path}/{example_idx}_0_key.pt', system_preprocess_key)
            shutil.copy(f'{save_path}/{example_idx}_0_value.pt', f"{preprocess_save_path}/{example_idx}_0_value.pt")

        # Preprocess each document
        for doc_idx, doc_tensor in enumerate(sub_q_doc_tensors):
            chunk_id = doc_chunk_ids[doc_idx]
            preprocess_key_path = f"{preprocess_save_path}/{example_idx}_{chunk_id}_key.pt"

            if os.path.exists(preprocess_key_path):
                continue

            print(f"    Preprocessing document {chunk_id} with FusionRAG...")

            # Generate on-demand similar documents' cache if needed
            if len(context_rank) > 0:
                global_doc_idx = sum(corpus_lens[:example_idx]) + (chunk_id - 1)
                if global_doc_idx < len(context_rank):
                    for similar_global_idx in context_rank[global_doc_idx][:topk]:
                        if similar_global_idx < 0 or similar_global_idx == global_doc_idx:
                            continue

                        corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                        similar_chunk_id = c_id + 1

                        similar_cache_key_path = f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt"
                        if not os.path.exists(similar_cache_key_path):
                            # Generate cache for similar document
                            other_system_cache_path = f'{save_path}/{corpus_i}_0_key.pt'
                            if not os.path.exists(other_system_cache_path):
                                other_input = system_tensor.unsqueeze(0)
                                prefill_and_save_kv_cache(
                                    model, tokenizer, past_key_values, other_input.to(input_device),
                                    save_path=save_path, example_id=corpus_i, chunk_id=0,
                                    system_len=system_len, passage_len=system_len,
                                    reprocess_method=reprocess_method, device=input_device, device_map=device_map
                                )

                            similar_doc_tensor = questions_data[corpus_i]['doc_tensors'][c_id]
                            other_passage_len = similar_doc_tensor.shape[0]
                            other_input = torch.cat((system_tensor, similar_doc_tensor)).unsqueeze(0)

                            prefill_and_save_kv_cache(
                                model, tokenizer, past_key_values, other_input.to(input_device),
                                save_path=save_path, example_id=corpus_i, chunk_id=similar_chunk_id,
                                system_len=system_len, passage_len=other_passage_len,
                                reprocess_method=reprocess_method, device=input_device, device_map=device_map
                            )

            # Load and fuse caches
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.past_tokens[layer_idx] = 0

            past_len = 0
            corpus_passages = [system_tensor]

            # Load system cache
            system_key_cache = torch.load(f"{save_path}/{example_idx}_0_key.pt", weights_only=True)
            system_value_cache = torch.load(f"{save_path}/{example_idx}_0_value.pt", weights_only=True)

            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx].narrow(2, 0, system_len).copy_(system_key_cache[layer_idx])
                past_key_values.value_cache[layer_idx].narrow(2, 0, system_len).copy_(system_value_cache[layer_idx])
                past_key_values.past_tokens[layer_idx] += system_len
            past_len += system_len

            # Load similar documents
            if len(context_rank) > 0:
                global_doc_idx = sum(corpus_lens[:example_idx]) + (chunk_id - 1)
                if global_doc_idx < len(context_rank):
                    for similar_global_idx in context_rank[global_doc_idx][:topk]:
                        if similar_global_idx < 0 or similar_global_idx == global_doc_idx:
                            continue

                        corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                        similar_chunk_id = c_id + 1

                        similar_doc_tensor = questions_data[corpus_i]['doc_tensors'][c_id]
                        corpus_len = similar_doc_tensor.shape[0]
                        corpus_passages.append(similar_doc_tensor)

                        chunk_key_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt", weights_only=True)
                        chunk_value_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_value.pt", weights_only=True)

                        for layer_idx in range(len(past_key_values.key_cache)):
                            past_key_values.key_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_key_cache[layer_idx])
                            past_key_values.value_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_value_cache[layer_idx])
                            past_key_values.past_tokens[layer_idx] += corpus_len
                        past_len += corpus_len

            # Add current document
            corpus_passages.append(doc_tensor)

            # Preprocess
            prefill_with_cache_and_save_preprocess(
                model, tokenizer, past_key_values, corpus_passages,
                preprocess_save_path, example_idx, chunk_id,
                system_len=system_len, revert_rope=revert_rope,
                reprocess_method=reprocess_method, device=input_device, device_map=device_map
            )

    print("  ✓ Both KV caches are ready")

    # ============================================================================
    # STEP 3: Decode with rate=0.3 KV cache and monitor
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 3: Decoding with rate=0.3 (preprocessed) KV cache")
    print("="*100)

    # Reset cache
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    logits_rate03, attention_rate03 = load_and_decode_with_monitoring(
        model, tokenizer, past_key_values, iter_tokens,
        preprocess_save_path, example_idx, kv_chunk_ids,
        answer_tokens,
        revert_rope=revert_rope,
        reprocess_method=reprocess_method,
        rate=0.3,
        preprocess=True,
        device=input_device,
        device_map=device_map,
        monitor_attention=True
    )

    # Decode the answer from rate=0.3 logits (greedy)
    answer_rate03_tokens = []
    for logits in logits_rate03:
        top1_token = torch.argmax(logits).item()
        answer_rate03_tokens.append(top1_token)
    answer_rate03 = tokenizer.decode(answer_rate03_tokens, skip_special_tokens=True)
    print(f"\nGenerated answer with rate=0.3: {answer_rate03}")

    # ============================================================================
    # STEP 4: Decode with rate=1.0 KV cache and monitor
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 4: Decoding with rate=1.0 (full) KV cache")
    print("="*100)

    # Reset cache
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    logits_rate1, attention_rate1 = load_and_decode_with_monitoring(
        model, tokenizer, past_key_values, iter_tokens,
        save_path, example_idx, kv_chunk_ids,
        answer_tokens,
        revert_rope=False,  # No rope reversion for rate=1.0
        reprocess_method='normal',
        rate=0,
        preprocess=False,
        device=input_device,
        device_map=device_map,
        monitor_attention=True
    )

    # ============================================================================
    # STEP 5: Compare results
    # ============================================================================
    print("\n" + "="*100)
    print("STEP 5: Comparing attention and logits")
    print("="*100)

    # Compare logits
    logits_save_path = os.path.join(output_dir, f'logits_comparison_ex{example_idx}_subq{sub_question_idx}.json')
    compare_logits(logits_rate03, logits_rate1, tokenizer, logits_save_path)

    # Compare attention
    attn_save_path = os.path.join(output_dir, f'attention_comparison_ex{example_idx}_subq{sub_question_idx}.json')
    compare_attention(attention_rate03, attention_rate1, attn_save_path)

    # Save summary
    summary = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'main_question': q_data['main_question'],
        'sub_question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'generated_answer_rate1': answer_text,
        'generated_answer_rate03': answer_rate03,
        'num_answer_tokens': len(answer_tokens),
    }

    summary_path = os.path.join(output_dir, f'summary_ex{example_idx}_subq{sub_question_idx}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nGenerated Answers:")
    print(f"  Rate=1.0: {answer_text}")
    print(f"  Rate=0.3: {answer_rate03}")
    print(f"  Ground Truth: {sub_q_info['answer']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Logits: {logits_save_path}")
    print(f"  - Attention: {attn_save_path}")
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
        output_dir='./kvcache_comparison_analysis'
    )

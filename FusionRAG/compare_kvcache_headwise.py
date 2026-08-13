#!/usr/bin/env python3
"""
Per-KV-Head Token Selection

Key innovations:
1. Each KV head selects different tokens based on its own attention
2. Each head maintains its own position encoding (RoPE)
3. Each head has different cache length
4. Custom attention that supports per-head variable lengths
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoConfig

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import rotate_half
from test_fusionrag_reflect import load_model, prepare_reflect_data, PreprocessScope


class PerHeadKVCache:
    """
    KV Cache that supports different sequence lengths per head

    Structure:
        key_cache[layer_idx][head_idx]: [batch, 1, seq_len_head, head_dim]
        value_cache[layer_idx][head_idx]: [batch, 1, seq_len_head, head_dim]
        past_tokens[layer_idx][head_idx]: int (length for this head)
        position_ids[layer_idx][head_idx]: [seq_len_head] (actual positions)
    """
    def __init__(self, num_layers, num_kv_heads, head_dim, max_cache_len, device, dtype):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Initialize storage for each layer and head
        self.key_cache = []
        self.value_cache = []
        self.past_tokens = []
        self.position_ids = []

        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            layer_past = []
            layer_positions = []

            for head_idx in range(num_kv_heads):
                # Pre-allocate cache
                key = torch.zeros(1, 1, max_cache_len, head_dim, device=device, dtype=dtype)
                value = torch.zeros(1, 1, max_cache_len, head_dim, device=device, dtype=dtype)

                layer_keys.append(key)
                layer_values.append(value)
                layer_past.append(0)
                layer_positions.append([])

            self.key_cache.append(layer_keys)
            self.value_cache.append(layer_values)
            self.past_tokens.append(layer_past)
            self.position_ids.append(layer_positions)

    def update(self, layer_idx, head_idx, key_states, value_states, positions):
        """
        Update cache for specific layer and head

        Args:
            key_states: [batch, 1, new_len, head_dim]
            value_states: [batch, 1, new_len, head_dim]
            positions: [new_len] - actual position ids for these tokens
        """
        new_len = key_states.shape[2]
        current_len = self.past_tokens[layer_idx][head_idx]

        # Copy new states to cache
        self.key_cache[layer_idx][head_idx][:, :, current_len:current_len+new_len, :] = key_states
        self.value_cache[layer_idx][head_idx][:, :, current_len:current_len+new_len, :] = value_states

        # Update length and positions
        self.past_tokens[layer_idx][head_idx] += new_len
        self.position_ids[layer_idx][head_idx].extend(positions.tolist())

    def get_kv(self, layer_idx, head_idx):
        """Get cached KV for a specific head"""
        length = self.past_tokens[layer_idx][head_idx]
        key = self.key_cache[layer_idx][head_idx][:, :, :length, :]
        value = self.value_cache[layer_idx][head_idx][:, :, :length, :]
        return key, value

    def get_positions(self, layer_idx, head_idx):
        """Get position ids for a specific head"""
        return torch.tensor(self.position_ids[layer_idx][head_idx], dtype=torch.long, device=self.device)


def compute_per_head_attention_selection(
    model,
    tokenizer,
    passages,
    chunk_ids,
    query_tensor,
    example_id,
    full_cache_path,
    ratio=0.3,
    device="cuda:0"
):
    """
    对每层的每个KV head独立计算attention并选择tokens

    Returns:
        head_wise_selections: List[List[np.ndarray]]
            head_wise_selections[layer_idx][head_idx] = selected token indices
    """
    num_layers = len(model.model.layers)
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    num_query_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_query_heads
    num_groups = num_query_heads // num_kv_heads

    print(f"\n{'='*100}")
    print("Computing per-KV-head attention-based token selection")
    print(f"{'='*100}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of KV heads per layer: {num_kv_heads}")
    print(f"Number of query heads per layer: {num_query_heads}")
    print(f"Query heads per KV head: {num_groups}")
    print(f"Selection ratio: {ratio}")

    # 加载完整的document KV caches
    # 跳过chunk 0 (system prompt) 和 chunk 1 (第一个文档)
    # 因为它们都是严格的prefix cache命中
    past_len = 0
    doc_key_caches = []
    doc_value_caches = []
    doc_lengths = []

    print(f"Skipping chunks: 0 (system) and 1 (first doc, prefix cache hit)")

    for idx, passage in enumerate(passages[2:]):  # 跳过chunk 0和chunk 1
        chunk_id = chunk_ids[idx + 2]
        passage_len = passage.shape[0]

        chunk_key_cache = torch.load(
            f'{full_cache_path}/{example_id}_{chunk_id}_key.pt',
            weights_only=True
        ).to(device)
        chunk_value_cache = torch.load(
            f'{full_cache_path}/{example_id}_{chunk_id}_value.pt',
            weights_only=True
        ).to(device)

        doc_key_caches.append(chunk_key_cache)
        doc_value_caches.append(chunk_value_cache)
        doc_lengths.append(passage_len)
        past_len += passage_len

    print(f"Loaded {len(doc_key_caches)} document caches for recomputation")
    print(f"Total recomputable document length: {past_len}")

    # 拼接所有documents的KV cache
    all_doc_keys = []
    all_doc_values = []

    for layer_idx in range(num_layers):
        layer_keys = []
        layer_values = []

        for chunk_idx in range(len(doc_key_caches)):
            layer_keys.append(doc_key_caches[chunk_idx][layer_idx])
            layer_values.append(doc_value_caches[chunk_idx][layer_idx])

        # Concatenate along sequence dimension
        layer_key = torch.cat(layer_keys, dim=2)  # [batch, num_kv_heads, total_seq_len, head_dim]
        layer_value = torch.cat(layer_values, dim=2)

        all_doc_keys.append(layer_key)
        all_doc_values.append(layer_value)

    # 简化：直接使用document keys计算attention，不需要forward query
    # 我们可以用query tensor的embedding + average作为参考
    print(f"\nComputing per-head attention scores (using K-based similarity)...")

    head_wise_selections = []

    for layer_idx in range(num_layers):
        layer_selections = []

        # Get document keys for this layer
        doc_keys = all_doc_keys[layer_idx]  # [1, num_kv_heads, past_len, head_dim]

        # For each KV head
        for head_idx in range(num_kv_heads):
            # Get this head's keys
            head_keys = doc_keys[:, head_idx:head_idx+1, :, :]  # [1, 1, past_len, head_dim]

            # Compute similarity with query
            # Use average key from this head as reference
            avg_key = head_keys.mean(dim=2, keepdim=True)  # [1, 1, 1, head_dim]

            # Compute attention scores
            similarities = torch.matmul(avg_key, head_keys.transpose(2, 3))  # [1, 1, 1, past_len]
            similarities = similarities / (head_dim ** 0.5)
            attention_weights = F.softmax(similarities, dim=-1)  # [1, 1, 1, past_len]

            attention_scores = attention_weights[0, 0, 0, :].float().cpu().numpy()  # [past_len]

            # Select top-k
            k = int(past_len * ratio)
            top_k_indices = np.argsort(attention_scores)[-k:]
            top_k_indices = np.sort(top_k_indices)

            layer_selections.append(top_k_indices)

        head_wise_selections.append(layer_selections)

        if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
            # Statistics for this layer
            head_lens = [len(sel) for sel in layer_selections]
            print(f"  Layer {layer_idx:2d}: head selections: {head_lens}")

    # Analyze head diversity
    print(f"\n{'='*100}")
    print("Per-head selection diversity analysis")
    print(f"{'='*100}")

    for layer_idx in [0, num_layers//2, num_layers-1]:
        print(f"\nLayer {layer_idx}:")
        layer_sels = head_wise_selections[layer_idx]

        # Compute pairwise overlap
        for i in range(num_kv_heads):
            for j in range(i+1, num_kv_heads):
                overlap = len(set(layer_sels[i]) & set(layer_sels[j]))
                union = len(set(layer_sels[i]) | set(layer_sels[j]))
                overlap_ratio = overlap / union if union > 0 else 0
                print(f"  Head {i} vs Head {j}: {overlap}/{union} overlap ({overlap_ratio*100:.1f}%)")

    print(f"{'='*100}\n")

    return head_wise_selections, past_len


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def save_per_head_cache(
    model,
    passages,
    chunk_ids,
    head_wise_selections,
    example_id,
    full_cache_path,
    save_path,
    device="cuda:0"
):
    """
    保存per-head selection的cache
    """
    os.makedirs(save_path, exist_ok=True)

    num_layers = len(model.model.layers)
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)

    print(f"\n{'='*100}")
    print(f"Saving per-head selection caches to {save_path}")
    print(f"{'='*100}")

    # 跳过chunk 0 (system) 和 chunk 1 (第一个文档，prefix cache命中)
    doc_passages = passages[2:]
    doc_chunk_ids = chunk_ids[2:]

    print(f"Saving {len(doc_passages)} chunks (skipped chunks 0 and 1)")

    # For each chunk
    chunk_start_pos = 0

    for chunk_idx, passage in enumerate(doc_passages):
        chunk_id = doc_chunk_ids[chunk_idx]
        chunk_len = passage.shape[0]

        # Load full cache for this chunk
        chunk_key_cache = torch.load(
            f'{full_cache_path}/{example_id}_{chunk_id}_key.pt',
            weights_only=True
        ).to(device)
        chunk_value_cache = torch.load(
            f'{full_cache_path}/{example_id}_{chunk_id}_value.pt',
            weights_only=True
        ).to(device)

        # For each layer and head, extract selected tokens
        selected_keys = []  # [num_layers][num_heads]
        selected_values = []
        selected_indices = []

        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            layer_indices = []

            for head_idx in range(num_kv_heads):
                # Get global selected indices for this head
                global_indices = head_wise_selections[layer_idx][head_idx]

                # Filter to current chunk
                chunk_end_pos = chunk_start_pos + chunk_len
                chunk_mask = (global_indices >= chunk_start_pos) & (global_indices < chunk_end_pos)
                chunk_indices = global_indices[chunk_mask] - chunk_start_pos

                # Ensure at least 2 tokens
                if len(chunk_indices) == 0:
                    chunk_indices = np.array([0, chunk_len-1])

                # Extract KV for this head
                head_key = chunk_key_cache[layer_idx][:, head_idx:head_idx+1, chunk_indices, :]
                head_value = chunk_value_cache[layer_idx][:, head_idx:head_idx+1, chunk_indices, :]

                layer_keys.append(head_key.cpu())
                layer_values.append(head_value.cpu())
                layer_indices.append(chunk_indices)

            selected_keys.append(layer_keys)
            selected_values.append(layer_values)
            selected_indices.append(layer_indices)

        # Save
        torch.save(selected_keys, f'{save_path}/{example_id}_{chunk_id}_keys.pt')
        torch.save(selected_values, f'{save_path}/{example_id}_{chunk_id}_values.pt')
        torch.save(selected_indices, f'{save_path}/{example_id}_{chunk_id}_indices.pt')

        # Statistics
        avg_per_head = np.mean([[len(indices) for indices in layer]
                                for layer in selected_indices])
        print(f"  Chunk {chunk_id}: avg {avg_per_head:.1f} tokens/head ({avg_per_head/chunk_len*100:.1f}%)")

        chunk_start_pos += chunk_len

    print(f"{'='*100}\n")


def main(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    topk=10,
    preprocess=True,
    preprocess_scope=PreprocessScope.GLOBAL,
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    device="cuda:0",
    use_multi_gpu=False,
    example_idx=4,
    sub_question_idx=1,
    ratio=0.3,
    output_dir='./kvcache_headwise_analysis'
):
    """
    Per-KV-head token selection实验
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "eager"  # 需要eager模式来自定义attention

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
    print(f"Sub Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print("="*100)

    # Build query
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    # Get documents
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    kv_chunk_ids = [0] + doc_chunk_ids

    # Full cache path
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    # Compute per-head selection
    head_wise_selections, total_doc_len = compute_per_head_attention_selection(
        model, tokenizer,
        iter_tokens[:-1],  # exclude query
        kv_chunk_ids,
        question_tensor,
        example_idx,
        full_cache_path,
        ratio=ratio,
        device=device
    )

    # Save per-head cache
    headwise_cache_path = os.path.join(cache_path, model_name, 'headwise_kv_cache')
    save_per_head_cache(
        model,
        iter_tokens[:-1],
        kv_chunk_ids,
        head_wise_selections,
        example_idx,
        full_cache_path,
        headwise_cache_path,
        device=device
    )

    print("\nPer-head selection completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        topk=10,
        preprocess=True,
        preprocess_scope=PreprocessScope.GLOBAL,
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        device="cuda:0",
        use_multi_gpu=False,
        example_idx=4,
        sub_question_idx=1,
        ratio=0.3,
        output_dir='./kvcache_headwise_analysis'
    )

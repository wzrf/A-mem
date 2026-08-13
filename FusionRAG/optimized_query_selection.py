#!/usr/bin/env python3
"""
优化的 Query Attention 选择方法
目标：在 30% 选择率下实现正确输出
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
sys.path.insert(0, '/mnt/data/wjh/FusionRAG/ktransformers')

from per_head_generation import sparse_prefill_per_head, generate_with_sparse_prefill
from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def compute_query_attention_scores(model, past_key_values, query_tensor, total_cache_len, prefix_len, doc_len, device):
    """计算 query 对文档每个位置的 attention 分数"""
    
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(query_tensor)
        hidden_states = inputs_embeds
        
        layer_attention_scores = {}
        
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            bsz, q_len, _ = hidden_states.size()
            
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = past_key_values.key_cache[layer_idx][:, :, :total_cache_len, :]
            
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            
            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            doc_attn = attn_weights[0, :, :, prefix_len:prefix_len + doc_len]
            doc_attn_avg = doc_attn.mean(dim=(0, 1)).cpu().float().numpy()
            
            layer_attention_scores[layer_idx] = doc_attn_avg
            
            value_states = past_key_values.value_cache[layer_idx][:, :, :total_cache_len, :]
            value_states = value_states.repeat_interleave(n_rep, dim=1)
            
            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)
            
            hidden_states = residual + attn_output
            
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
    
    return layer_attention_scores


def optimized_selection(attention_scores, doc_len, target_ratio):
    """
    优化的选择策略：
    1. 找到高 attention 的峰值区域
    2. 只扩展峰值附近的上下文
    3. 保持总选择数在目标范围内
    """
    
    # 使用后几层的 attention
    layers_to_use = [16, 20, 24, 27]
    multi_layer_attn = np.stack([attention_scores[i] for i in layers_to_use]).mean(axis=0)
    
    target_count = int(doc_len * target_ratio)
    
    # Step 1: 找到显著高于平均的位置（峰值）
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    
    # 使用不同的阈值找峰值
    threshold_high = mean_attn + 2 * std_attn  # 非常高的 attention
    threshold_medium = mean_attn + 1 * std_attn  # 中等高的 attention
    
    peaks_high = set(np.where(multi_layer_attn > threshold_high)[0])
    peaks_medium = set(np.where(multi_layer_attn > threshold_medium)[0])
    
    print(f"  High peaks (>mean+2*std): {len(peaks_high)}")
    print(f"  Medium peaks (>mean+1*std): {len(peaks_medium)}")
    
    # Step 2: 为高峰值添加较大上下文，中等峰值添加较小上下文
    selected = set()
    
    # 高峰值：±3 上下文
    for pos in peaks_high:
        for offset in range(-3, 4):
            new_pos = pos + offset
            if 0 <= new_pos < doc_len:
                selected.add(new_pos)
    
    # 中等峰值：±1 上下文 (如果还没选中)
    for pos in peaks_medium:
        if pos not in selected:
            for offset in range(-1, 2):
                new_pos = pos + offset
                if 0 <= new_pos < doc_len:
                    selected.add(new_pos)
    
    print(f"  After context expansion: {len(selected)}")
    
    # Step 3: 如果还没达到目标，继续添加高分 token
    if len(selected) < target_count:
        remaining = target_count - len(selected)
        sorted_indices = np.argsort(multi_layer_attn)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(pos)
                if len(selected) >= target_count:
                    break
    
    # Step 4: 如果超过目标，移除一些低分 token
    if len(selected) > target_count:
        # 计算选中 token 的分数，移除分数最低的
        selected_list = list(selected)
        scores = [multi_layer_attn[p] for p in selected_list]
        sorted_pairs = sorted(zip(selected_list, scores), key=lambda x: x[1], reverse=True)
        selected = set([p for p, _ in sorted_pairs[:target_count]])
    
    return sorted(list(selected))


def layer_wise_optimized_selection(attention_scores, doc_len, target_ratio):
    """
    Layer-wise 优化选择：每层可以选不同的 tokens
    """
    layer_selections = {}
    
    for layer_idx in attention_scores.keys():
        layer_attn = attention_scores[layer_idx]
        target_count = int(doc_len * target_ratio)
        
        # 找峰值
        mean_attn = np.mean(layer_attn)
        std_attn = np.std(layer_attn)
        threshold = mean_attn + 1.5 * std_attn
        
        peaks = set(np.where(layer_attn > threshold)[0])
        selected = set()
        
        # 扩展峰值上下文
        for pos in peaks:
            for offset in range(-2, 3):
                new_pos = pos + offset
                if 0 <= new_pos < doc_len:
                    selected.add(new_pos)
        
        # 补充到目标数量
        if len(selected) < target_count:
            sorted_indices = np.argsort(layer_attn)[::-1]
            for pos in sorted_indices:
                if pos not in selected:
                    selected.add(pos)
                    if len(selected) >= target_count:
                        break
        elif len(selected) > target_count:
            selected_list = list(selected)
            scores = [layer_attn[p] for p in selected_list]
            sorted_pairs = sorted(zip(selected_list, scores), key=lambda x: x[1], reverse=True)
            selected = set([p for p, _ in sorted_pairs[:target_count]])
        
        layer_selections[layer_idx] = sorted(list(selected))
    
    return layer_selections


def test_selection_strategy(model, tokenizer, past_key_values, position_to_token, 
                           selected_positions, query_tensor, prefix_len, doc_len, 
                           num_layers, num_kv_heads, device, full_cache_path, 
                           doc_chunk_ids, passages, example_idx, strategy_name):
    """测试一个选择策略"""
    
    print(f"\n{'=' * 80}")
    print(f"Testing: {strategy_name}")
    print("=" * 80)
    
    # 重新加载 cache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=32768,
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
            current_pos_cache = past_key_values.past_tokens[layer_idx]
            
            past_key_values.key_cache[layer_idx].narrow(2, current_pos_cache, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos_cache, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len
    
    actual_ratio = len(selected_positions) / doc_len
    print(f"Selected {len(selected_positions)} tokens ({actual_ratio*100:.1f}%)")
    
    # Check key token coverage
    positions = sorted(position_to_token.keys())
    tokens_list = [position_to_token[p] for p in positions]
    full_text = tokenizer.decode(tokens_list)
    
    token_to_text_pos = {}
    current_text = ""
    for pos in positions:
        token_id = position_to_token[pos]
        token_text = tokenizer.decode([token_id])
        start_pos = len(current_text)
        current_text += token_text
        token_to_text_pos[pos] = (start_pos, len(current_text), token_text)
    
    for phrase in ["1216", "1220"]:
        phrase_positions = []
        idx = 0
        while True:
            idx = full_text.find(phrase, idx)
            if idx == -1:
                break
            end_idx = idx + len(phrase)
            for pos in positions:
                start, end, _ = token_to_text_pos[pos]
                if start < end_idx and end > idx:
                    phrase_positions.append(pos)
            idx += 1
        
        selected_count = sum(1 for p in phrase_positions if p in selected_positions)
        print(f"  '{phrase}': {selected_count}/{len(phrase_positions)} selected")
    
    # Build layer_head_selections
    layer_head_selections = {}
    layer_union_positions = {}
    
    for layer_idx in range(num_layers):
        layer_union_positions[layer_idx] = selected_positions
        layer_head_selections[layer_idx] = {}
        for kv_head_idx in range(num_kv_heads):
            layer_head_selections[layer_idx][kv_head_idx] = {
                'positions': selected_positions.copy(),
                'num_selected': len(selected_positions)
            }
    
    # Sparse prefill
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
    
    # Reset past_tokens
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] = prefix_len + doc_len
    
    # Generate
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
    
    print(f"\nGenerated: {generated_text}")
    
    return {
        "strategy": strategy_name,
        "ratio": actual_ratio,
        "output": generated_text
    }


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
    print("Optimized Query Attention Selection Test")
    print("=" * 100)
    
    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, _ = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    
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
    
    # Build position_to_token
    position_to_token = {}
    current_pos = 0
    prefix_len = 0
    
    for chunk_idx, chunk_id in enumerate(doc_chunk_ids):
        if chunk_id == 0:
            key_path = os.path.join(full_cache_path, f'{example_idx}_{chunk_id}_key.pt')
            chunk_key = torch.load(key_path, map_location='cpu', weights_only=True)
            chunk_len = chunk_key[0].shape[2]
            prefix_len = chunk_len
        else:
            passage = passages[chunk_id - 1]
            doc_text = f"Document: {passage}\n"
            tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            chunk_len = len(tokens)
            
            for i, token_id in enumerate(tokens):
                position_to_token[current_pos - prefix_len + i] = token_id
        
        current_pos += chunk_len
    
    doc_len = current_pos - prefix_len
    print(f"\nPrefix length: {prefix_len}")
    print(f"Document length: {doc_len}")
    
    # Load cache for attention computation
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
            current_pos_cache = past_key_values.past_tokens[layer_idx]
            
            past_key_values.key_cache[layer_idx].narrow(2, current_pos_cache, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos_cache, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len
    
    total_cache_len = past_key_values.past_tokens[0]
    
    # Compute query attention
    print("\nComputing Query Attention Scores...")
    attention_scores = compute_query_attention_scores(
        model, past_key_values, query_tensor, total_cache_len, prefix_len, doc_len, device
    )
    
    results = []
    
    # Test optimized selection strategies
    for target_ratio in [0.30, 0.35]:
        print(f"\n{'#' * 80}")
        print(f"TARGET RATIO: {target_ratio*100:.0f}%")
        print("#" * 80)
        
        selected = optimized_selection(attention_scores, doc_len, target_ratio)
        
        result = test_selection_strategy(
            model, tokenizer, past_key_values, position_to_token,
            selected, query_tensor, prefix_len, doc_len,
            num_layers, num_kv_heads, device, full_cache_path,
            doc_chunk_ids, passages, example_idx,
            f"Optimized Query Selection ({target_ratio*100:.0f}%)"
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print("\nResults:")
    for r in results:
        status = "✓" if "1216" in r['output'] and "1220" in r['output'] else "✗"
        print(f"  {status} [{r['ratio']*100:.1f}%] {r['strategy']}: {r['output']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
使用 Query Attention 选择的 tokens 测试生成质量
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
            
            # 提取文档部分
            doc_attn = attn_weights[0, :, :, prefix_len:prefix_len + doc_len]
            doc_attn_avg = doc_attn.mean(dim=(0, 1)).cpu().float().numpy()
            
            layer_attention_scores[layer_idx] = doc_attn_avg
            
            # 继续前向传播
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


def select_with_context_expansion(attention_scores, doc_len, target_ratio, context_window=3):
    """选择 top-k 并扩展上下文窗口"""
    
    # 聚合多层 attention
    layers_to_use = [16, 20, 24, 27]  # 使用后面几层
    multi_layer_attn = np.stack([attention_scores[i] for i in layers_to_use]).mean(axis=0)
    
    target_count = int(doc_len * target_ratio)
    
    # 先选择 top tokens
    top_indices = set(np.argsort(multi_layer_attn)[::-1][:target_count // 2])
    
    # 扩展上下文
    expanded = set()
    for pos in top_indices:
        for offset in range(-context_window, context_window + 1):
            new_pos = pos + offset
            if 0 <= new_pos < doc_len:
                expanded.add(new_pos)
    
    # 补充更多高分 token 直到达到目标
    remaining = target_count - len(expanded)
    if remaining > 0:
        for pos in np.argsort(multi_layer_attn)[::-1]:
            if pos not in expanded:
                expanded.add(pos)
                if len(expanded) >= target_count:
                    break
    
    return sorted(list(expanded))


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
    print("Query Attention Selection Test")
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
    
    # Load cache
    print("\n" + "=" * 80)
    print("Loading cache")
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
            current_pos_cache = past_key_values.past_tokens[layer_idx]
            
            past_key_values.key_cache[layer_idx].narrow(2, current_pos_cache, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos_cache, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len
    
    total_cache_len = past_key_values.past_tokens[0]
    print(f"Total cache length: {total_cache_len}")
    
    # Compute query attention scores
    print("\n" + "=" * 80)
    print("Computing Query Attention Scores")
    print("=" * 80)
    
    attention_scores = compute_query_attention_scores(
        model, past_key_values, query_tensor, total_cache_len, prefix_len, doc_len, device
    )
    
    # Test different strategies
    strategies = [
        ("Pure Query Attention (30%)", 0.3, 0),
        ("Query Attention + Context (30%)", 0.3, 3),
        ("Query Attention + Context (25%)", 0.25, 3),
    ]
    
    results = []
    
    for strategy_name, target_ratio, context_window in strategies:
        print(f"\n{'=' * 80}")
        print(f"Testing: {strategy_name}")
        print("=" * 80)
        
        # 重新加载 cache
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
        
        # Select tokens
        if context_window > 0:
            selected_positions = select_with_context_expansion(
                attention_scores, doc_len, target_ratio, context_window
            )
        else:
            # Pure top-k
            layers_to_use = [16, 20, 24, 27]
            multi_layer_attn = np.stack([attention_scores[i] for i in layers_to_use]).mean(axis=0)
            target_count = int(doc_len * target_ratio)
            selected_positions = sorted(np.argsort(multi_layer_attn)[::-1][:target_count].tolist())
        
        actual_ratio = len(selected_positions) / doc_len
        print(f"Selected {len(selected_positions)} tokens ({actual_ratio*100:.1f}%)")
        
        # Check key token coverage
        positions = sorted(position_to_token.keys())
        tokens_list = [position_to_token[p] for p in positions]
        full_text = tokenizer.decode(tokens_list)
        
        key_phrases = ["1216", "1220"]
        token_to_text_pos = {}
        current_text = ""
        for pos in positions:
            token_id = position_to_token[pos]
            token_text = tokenizer.decode([token_id])
            start_pos = len(current_text)
            current_text += token_text
            token_to_text_pos[pos] = (start_pos, len(current_text), token_text)
        
        for phrase in key_phrases:
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
        
        results.append({
            "strategy": strategy_name,
            "ratio": actual_ratio,
            "output": generated_text
        })
        
        print(f"\nGenerated: {generated_text}")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print("\nResults:")
    for r in results:
        print(f"  [{r['ratio']*100:.1f}%] {r['strategy']}: {r['output']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Smart Query Selection: 使用连通性分析确保相关 token 群组被完整选中
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


def find_connected_components(positions, max_gap=3):
    """找到位置列表中的连通分量（相邻 token 群组）"""
    if not positions:
        return []
    
    positions = sorted(positions)
    components = []
    current_component = [positions[0]]
    
    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= max_gap:
            current_component.append(positions[i])
        else:
            components.append(current_component)
            current_component = [positions[i]]
    
    components.append(current_component)
    return components


def smart_selection(attention_scores, doc_len, target_ratio):
    """
    Smart 选择策略：
    1. 找到高 attention 的连通群组
    2. 按群组的平均 attention 排序
    3. 选择最重要的群组，并确保群组完整性
    """
    
    layers_to_use = [16, 20, 24, 27]
    multi_layer_attn = np.stack([attention_scores[i] for i in layers_to_use]).mean(axis=0)
    
    target_count = int(doc_len * target_ratio)
    
    # Step 1: 找到显著高于平均的位置
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    threshold = mean_attn + 0.5 * std_attn  # 较低阈值以捕获更多候选
    
    high_attn_positions = set(np.where(multi_layer_attn > threshold)[0])
    
    print(f"  High attention positions (>mean+0.5*std): {len(high_attn_positions)}")
    
    # Step 2: 找到连通群组
    components = find_connected_components(list(high_attn_positions), max_gap=2)
    
    print(f"  Connected components: {len(components)}")
    
    # Step 3: 计算每个群组的总 attention 和平均 attention
    component_scores = []
    for comp in components:
        total_score = sum(multi_layer_attn[p] for p in comp)
        avg_score = total_score / len(comp)
        component_scores.append((comp, total_score, avg_score))
    
    # Step 4: 按总 attention 排序
    component_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Step 5: 贪心选择群组，直到达到目标
    selected = set()
    selected_components = []
    
    for comp, total_score, avg_score in component_scores:
        # 扩展群组边界以包含上下文
        extended_comp = set()
        for p in comp:
            for offset in range(-1, 2):  # ±1 上下文
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)
        
        # 检查是否会超过目标
        new_positions = extended_comp - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:  # 允许10%的余量
            selected.update(extended_comp)
            selected_components.append((comp, total_score))
    
    print(f"  Selected {len(selected_components)} components")
    
    # Step 6: 如果还没达到目标，添加更多高分位置
    if len(selected) < target_count:
        remaining = target_count - len(selected)
        sorted_indices = np.argsort(multi_layer_attn)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(pos)
                if len(selected) >= target_count:
                    break
    
    # Step 7: 如果超过目标，移除最低分的位置
    while len(selected) > target_count:
        # 找到分数最低的位置
        min_pos = min(selected, key=lambda p: multi_layer_attn[p])
        selected.remove(min_pos)
    
    return sorted(list(selected))


def test_selection(model, tokenizer, past_key_values, position_to_token, 
                  selected_positions, query_tensor, prefix_len, doc_len, 
                  num_layers, num_kv_heads, device, full_cache_path, 
                  doc_chunk_ids, passages, example_idx, strategy_name):
    """测试选择策略"""
    
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
    
    for phrase in ["1216", "1220", "crowned", "Westminster"]:
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
    
    # Reset
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
    print("Smart Query Selection Test")
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
    
    # Test different ratios
    for target_ratio in [0.30, 0.32, 0.35]:
        print(f"\n{'#' * 80}")
        print(f"TARGET RATIO: {target_ratio*100:.0f}%")
        print("#" * 80)
        
        selected = smart_selection(attention_scores, doc_len, target_ratio)
        
        result = test_selection(
            model, tokenizer, past_key_values, position_to_token,
            selected, query_tensor, prefix_len, doc_len,
            num_layers, num_kv_heads, device, full_cache_path,
            doc_chunk_ids, passages, example_idx,
            f"Smart Query Selection ({target_ratio*100:.0f}%)"
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")
    print("\nResults:")
    for r in results:
        has_1216 = "1216" in r['output']
        has_1220 = "1220" in r['output']
        status = "✓" if has_1216 and has_1220 else "✗"
        print(f"  {status} [{r['ratio']*100:.1f}%] {r['strategy']}: {r['output']}")
    
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    print("""
基于 Query Attention 的选择方法分析:

1. Query Attention 能够识别与问题相关的 tokens
2. 但需要确保关键 token 群组（如年份数字）被完整选中
3. 使用连通分量分析可以帮助保持 token 群组的完整性
4. 需要在选择率和完整性之间权衡

建议的最终方案:
- 使用 Query Attention 作为主要评分依据
- 采用连通分量分析确保群组完整性  
- 目标选择率设为 32-35% 以确保关键信息完整
""")


if __name__ == "__main__":
    main()

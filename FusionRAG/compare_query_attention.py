#!/usr/bin/env python3
"""
Compare QueryAttention implementations between per_head_generation.py and test_fusionrag_reflect.py

This script isolates and compares:
1. per_head_generation.py: Uses compute_query_attention_scores() with a separate forward pass
2. test_fusionrag_reflect.py: Uses model forward with reprocess_method='QueryAttention'

Target: example_idx=4, sub_question_idx=1
Expected answer: "1216 and 1220"
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import (
    rotate_half,
    smart_query_selection as utils_smart_query_selection,
    compute_query_attention_scores_direct,
    smart_query_selection_v2
)


def per_head_compute_query_attention_scores(model, past_key_values, query_tensor, total_cache_len, prefix_len, doc_len, device):
    """
    [per_head_generation.py 版本]
    计算 query 对文档每个位置的 attention 分数

    特点:
    - 手动做 forward pass
    - query 没有应用 RoPE (但 cache 中的 key 有 RoPE)
    - 返回每层的 attention scores (dict)
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    print(f"\n[per_head 方法] Computing attention scores...")
    print(f"  total_cache_len={total_cache_len}, prefix_len={prefix_len}, doc_len={doc_len}")

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(query_tensor)
        hidden_states = inputs_embeds

        layer_attention_scores = {}

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            # Query projection (NO RoPE applied here!)
            query_states = layer.self_attn.q_proj(hidden_states)
            # Get keys from cache (keys already have RoPE from when they were cached)
            key_states = past_key_values.key_cache[layer_idx][:, :, :total_cache_len, :]

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)

            # Compute attention: Q @ K^T (note: Q has no RoPE, K has RoPE!)
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Extract document portion's attention
            doc_attn = attn_weights[0, :, :, prefix_len:prefix_len + doc_len]
            # Average over query tokens (dim 1) and heads (dim 0)
            doc_attn_avg = doc_attn.mean(dim=(0, 1)).cpu().float().numpy()

            layer_attention_scores[layer_idx] = doc_attn_avg

            # Continue forward pass for next layer
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


def per_head_smart_query_selection(attention_scores, doc_len, target_ratio, num_layers):
    """
    [per_head_generation.py 版本]
    Smart Query Selection: 使用连通性分析确保相关 token 群组被完整选中

    特点:
    - 输入是 dict: {layer_idx: numpy array of shape [doc_len]}
    - 使用固定的层: [16, 20, 24, 27]
    - 返回位置列表 (不加 system_len 偏移)
    """
    # 使用后几层的 attention
    layers_to_use = [l for l in [16, 20, 24, 27] if l < num_layers]
    if not layers_to_use:
        layers_to_use = [num_layers - 1]

    print(f"\n[per_head smart_query_selection] Using layers: {layers_to_use}")

    multi_layer_attn = np.stack([attention_scores[i] for i in layers_to_use]).mean(axis=0)

    target_count = int(doc_len * target_ratio)

    # Step 1: 找到高 attention 位置
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    threshold = mean_attn + 0.5 * std_attn

    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])

    print(f"  High attention positions (>mean+0.5*std): {len(high_attn_positions)}")
    print(f"  Target count: {target_count}")

    # Step 2: 找到连通群组
    def find_connected_components(positions, max_gap=2):
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

    components = find_connected_components(high_attn_positions, max_gap=2)
    print(f"  Connected components: {len(components)}")

    # Step 3: 计算每个群组的总 attention
    component_scores = []
    for comp in components:
        total_score = sum(multi_layer_attn[p] for p in comp)
        component_scores.append((comp, total_score))

    # Step 4: 按总 attention 排序
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 5: 贪心选择群组 + 上下文扩展
    selected = set()

    for comp, total_score in component_scores:
        # 扩展群组边界 (±1)
        extended_comp = set()
        for p in comp:
            for offset in range(-1, 2):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)

        # 检查是否会超过目标 (允许 10% 余量)
        new_positions = extended_comp - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended_comp)

    print(f"  After component selection: {len(selected)}")

    # Step 6: 补充到目标数量
    if len(selected) < target_count:
        sorted_indices = np.argsort(multi_layer_attn)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(pos)
                if len(selected) >= target_count:
                    break

    # Step 7: 如果超过目标，移除最低分的位置
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: multi_layer_attn[p])
        selected.remove(min_pos)

    selected_list = sorted(list(selected))
    return selected_list


def utils_compute_query_attention(model, past_key_values, query_tensor, system_len, doc_len, passages_len, device):
    """
    [test_fusionrag_reflect.py + utils.py 版本]
    使用 model forward 计算 attention, 模拟 load_kv_and_generate 中的流程

    特点:
    - 调用 model forward 并设置 reprocess_method='QueryAttention'
    - query_states 在 model forward 中已经应用了 RoPE
    - attention 存储到 importance_cache
    """
    print(f"\n[utils 方法] Computing attention scores via model forward...")
    print(f"  system_len={system_len}, doc_len={doc_len}")

    seq_length = query_tensor.shape[1]
    past_len = system_len + doc_len

    cache_position = torch.arange(past_len, past_len + seq_length, device=device)

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(query_tensor).to(device)

        # 调用 model forward，触发 QueryAttention 分支
        model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            reprocess_method='QueryAttention',
            return_dict=False,
            use_cache=True,
            passages_len=passages_len
        )

        # 聚合多层 attention (后 1/4 的层)
        num_layers = len(past_key_values.importance_cache)
        start_layer = num_layers * 3 // 4
        active_layers = list(range(start_layer, num_layers))

        print(f"  Using layers: {active_layers}")

        # 收集多层的 attention 并平均
        layer_attentions = []
        for layer_idx in active_layers:
            layer_attn = past_key_values.importance_cache[layer_idx][:, system_len:system_len + doc_len]
            # 对所有 heads 取平均
            layer_attn_avg = layer_attn.mean(dim=0).to(device)  # [doc_len]
            layer_attentions.append(layer_attn_avg)

        # 聚合多层
        multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]

    return multi_layer_attn


def main():
    print("="*100)
    print("Comparing QueryAttention Implementations")
    print("="*100)

    # Configuration
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    cache_path = '/mnt/data/reflect/'
    model_name = 'Qwen2.5-7B-Instruct'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    example_idx = 4
    sub_question_idx = 1
    rate = 0.3
    device = "cuda:0"

    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model, device_map = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load data
    print("\nLoading data...")
    questions_data, system_tensor, _, _ = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen', 10,
        max_main_questions=example_idx + 1,
        preprocess=False
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"\nExample {example_idx}, Sub-question {sub_question_idx}")
    print(f"Question: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")

    # Get data
    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    passages = q_data['docs']
    doc_tensors = q_data['doc_tensors']

    # Build query tensor
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Prepare cache path
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    # ==========================================================================
    # Method 1: per_head_generation.py approach
    # ==========================================================================
    print("\n" + "="*100)
    print("METHOD 1: per_head_generation.py approach")
    print("="*100)

    # Load cache for Method 1
    max_cache_len = 32768
    past_key_values_1 = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    prefix_len = 0
    total_doc_len = 0
    for chunk_id in doc_chunk_ids:
        chunk_key = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt', weights_only=True).to(device)
        chunk_value = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt', weights_only=True).to(device)
        cache_len = chunk_key[0].shape[2]

        for layer_idx in range(model.config.num_hidden_layers):
            layer_key = chunk_key[layer_idx]
            layer_value = chunk_value[layer_idx]
            current_pos = past_key_values_1.past_tokens[layer_idx]
            past_key_values_1.key_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(layer_key)
            past_key_values_1.value_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(layer_value)
            past_key_values_1.past_tokens[layer_idx] += cache_len

        if chunk_id == 0:
            prefix_len = cache_len
        else:
            total_doc_len += cache_len

        print(f"  Loaded chunk {chunk_id}: {cache_len} tokens")

    total_cache_len = prefix_len + total_doc_len
    print(f"\nprefix_len={prefix_len}, doc_len={total_doc_len}, total={total_cache_len}")

    # Compute attention scores using per_head method
    layer_attention_scores_1 = per_head_compute_query_attention_scores(
        model, past_key_values_1, query_tensor, total_cache_len, prefix_len, total_doc_len, device
    )

    # Apply smart selection (per_head version)
    selected_1 = per_head_smart_query_selection(
        layer_attention_scores_1, total_doc_len, rate, model.config.num_hidden_layers
    )
    # Convert to global indices (add prefix_len offset)
    selected_1_global = [p + prefix_len for p in selected_1]

    print(f"\n[Method 1] Selected {len(selected_1)} positions (local to doc)")
    print(f"  First 20 local positions: {selected_1[:20]}")

    # ==========================================================================
    # Method 2: test_fusionrag_reflect.py + utils.py approach
    # ==========================================================================
    print("\n" + "="*100)
    print("METHOD 2: test_fusionrag_reflect.py + utils.py approach")
    print("="*100)

    # Load cache for Method 2 (fresh cache)
    past_key_values_2 = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Load cache same as method 1
    for chunk_id in doc_chunk_ids:
        chunk_key = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt', weights_only=True).to(device)
        chunk_value = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt', weights_only=True).to(device)
        cache_len = chunk_key[0].shape[2]

        for layer_idx in range(model.config.num_hidden_layers):
            layer_key = chunk_key[layer_idx]
            layer_value = chunk_value[layer_idx]
            current_pos = past_key_values_2.past_tokens[layer_idx]
            past_key_values_2.key_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(layer_key)
            past_key_values_2.value_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(layer_value)
            past_key_values_2.past_tokens[layer_idx] += cache_len

    # Build passages_len for utils method
    system_len = prefix_len
    passages_len = [system_len]
    for chunk_id in doc_chunk_ids:
        if chunk_id == 0:
            continue
        chunk_key = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt', weights_only=True)
        passages_len.append(chunk_key[0].shape[2])
    passages_len.append(query_tensor.shape[1])  # query length

    print(f"passages_len = {passages_len}")

    # Compute attention scores using utils method
    multi_layer_attn_2 = utils_compute_query_attention(
        model, past_key_values_2, query_tensor, system_len, total_doc_len, passages_len, device
    )

    # Apply smart selection (utils version)
    selected_2 = utils_smart_query_selection(
        attention_scores=multi_layer_attn_2,
        doc_len=total_doc_len,
        target_ratio=rate,
        system_len=system_len,
        device=device
    )

    print(f"\n[Method 2] Selected {len(selected_2)} positions (global, includes system_len offset)")
    print(f"  First 20 global positions: {selected_2[:20]}")

    # Convert to local positions for comparison
    selected_2_local = [p - system_len for p in selected_2]
    print(f"  First 20 local positions: {selected_2_local[:20]}")

    # ==========================================================================
    # Method 3: Fixed implementation (compute_query_attention_scores_direct + smart_query_selection_v2)
    # ==========================================================================
    print("\n" + "="*100)
    print("METHOD 3: Fixed implementation (using compute_query_attention_scores_direct)")
    print("="*100)

    # Load cache for Method 3 (fresh cache)
    past_key_values_3 = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Load cache same as method 1 and 2
    for chunk_id in doc_chunk_ids:
        chunk_key = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt', weights_only=True).to(device)
        chunk_value = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt', weights_only=True).to(device)
        cache_len = chunk_key[0].shape[2]

        for layer_idx in range(model.config.num_hidden_layers):
            layer_key = chunk_key[layer_idx]
            layer_value = chunk_value[layer_idx]
            current_pos = past_key_values_3.past_tokens[layer_idx]
            past_key_values_3.key_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(layer_key)
            past_key_values_3.value_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(layer_value)
            past_key_values_3.past_tokens[layer_idx] += cache_len

    # Use the fixed implementation
    layer_attention_scores_3 = compute_query_attention_scores_direct(
        model=model,
        past_key_values=past_key_values_3,
        query_tensor=query_tensor,
        total_cache_len=total_cache_len,
        prefix_len=prefix_len,
        doc_len=total_doc_len,
        device=device
    )

    # Apply the fixed smart selection
    selected_3 = smart_query_selection_v2(
        attention_scores_dict=layer_attention_scores_3,
        doc_len=total_doc_len,
        target_ratio=rate,
        system_len=prefix_len,
        num_layers=model.config.num_hidden_layers
    )

    # Convert to local positions for comparison
    selected_3_local = [p - prefix_len for p in selected_3]

    print(f"\n[Method 3] Selected {len(selected_3)} positions (global)")
    print(f"  First 20 local positions: {selected_3_local[:20]}")

    # ==========================================================================
    # Compare results
    # ==========================================================================
    print("\n" + "="*100)
    print("COMPARISON")
    print("="*100)

    set_1 = set(selected_1)
    set_2 = set(selected_2_local)
    set_3 = set(selected_3_local)

    intersection_1_2 = set_1 & set_2
    intersection_1_3 = set_1 & set_3
    intersection_2_3 = set_2 & set_3

    print(f"\nMethod 1 (per_head original): {len(set_1)} positions")
    print(f"Method 2 (old utils.py): {len(set_2)} positions")
    print(f"Method 3 (fixed utils.py): {len(set_3)} positions")
    print(f"\nOverlap M1 & M2: {len(intersection_1_2)} ({len(intersection_1_2)/len(set_1)*100:.1f}%)")
    print(f"Overlap M1 & M3 (should be ~100%): {len(intersection_1_3)} ({len(intersection_1_3)/len(set_1)*100:.1f}%)")
    print(f"Overlap M2 & M3: {len(intersection_2_3)} ({len(intersection_2_3)/len(set_2)*100:.1f}%)")

    # Compare attention score distributions
    print("\n" + "-"*50)
    print("Attention Score Statistics (aggregated across layers)")
    print("-"*50)

    # Method 1: aggregate layers [16, 20, 24, 27]
    layers_to_use = [l for l in [16, 20, 24, 27] if l < model.config.num_hidden_layers]
    attn_1_agg = np.stack([layer_attention_scores_1[i] for i in layers_to_use]).mean(axis=0)

    # Method 2: already aggregated
    attn_2_agg = multi_layer_attn_2.float().cpu().numpy()

    print(f"\nMethod 1 attention (layers {layers_to_use}):")
    print(f"  Mean: {np.mean(attn_1_agg):.6f}")
    print(f"  Std:  {np.std(attn_1_agg):.6f}")
    print(f"  Max:  {np.max(attn_1_agg):.6f}")
    print(f"  Min:  {np.min(attn_1_agg):.6f}")

    num_layers = model.config.num_hidden_layers
    start_layer = num_layers * 3 // 4
    active_layers = list(range(start_layer, num_layers))
    print(f"\nMethod 2 attention (layers {active_layers}):")
    print(f"  Mean: {np.mean(attn_2_agg):.6f}")
    print(f"  Std:  {np.std(attn_2_agg):.6f}")
    print(f"  Max:  {np.max(attn_2_agg):.6f}")
    print(f"  Min:  {np.min(attn_2_agg):.6f}")

    # Check correlation
    correlation = np.corrcoef(attn_1_agg, attn_2_agg)[0, 1]
    print(f"\nCorrelation between Method 1 and Method 2 attention: {correlation:.4f}")

    # Analyze which tokens contain the answer "1216 and 1220"
    print("\n" + "-"*50)
    print("Analyzing answer tokens")
    print("-"*50)

    # Build token list from loaded documents
    all_tokens = []
    for chunk_id in doc_chunk_ids:
        if chunk_id == 0:
            continue
        passage = passages[chunk_id - 1]
        doc_text = f"Document: {passage}\n"
        tokens = tokenizer.encode(doc_text, add_special_tokens=False)
        all_tokens.extend(tokens)

    # Find answer patterns (digit by digit tokenization)
    pattern_1216 = tokenizer.encode("1216", add_special_tokens=False)
    pattern_1220 = tokenizer.encode("1220", add_special_tokens=False)

    print(f"  Pattern for '1216': {pattern_1216} = {[tokenizer.decode([t]) for t in pattern_1216]}")
    print(f"  Pattern for '1220': {pattern_1220} = {[tokenizer.decode([t]) for t in pattern_1220]}")

    def find_pattern(tokens, pattern):
        positions = []
        for i in range(len(tokens) - len(pattern) + 1):
            if tokens[i:i+len(pattern)] == pattern:
                positions.append(i)
        return positions

    positions_1216 = find_pattern(all_tokens, pattern_1216)
    positions_1220 = find_pattern(all_tokens, pattern_1220)

    print(f"\n  Found '1216' at positions: {positions_1216}")
    print(f"  Found '1220' at positions: {positions_1220}")

    # Critical positions: all 4 digits of each year
    critical_positions_1216 = []
    for start_pos in positions_1216:
        for offset in range(4):
            critical_positions_1216.append(start_pos + offset)
    critical_positions_1220 = []
    for start_pos in positions_1220:
        for offset in range(4):
            critical_positions_1220.append(start_pos + offset)

    all_critical = set(critical_positions_1216 + critical_positions_1220)
    print(f"\n  Total critical positions (both years, all digits): {len(all_critical)}")

    # Check coverage in each method
    critical_in_1 = all_critical & set_1
    critical_in_2 = all_critical & set_2
    critical_in_3 = all_critical & set_3

    print(f"\n  Critical positions in Method 1 (original): {len(critical_in_1)}/{len(all_critical)}")
    print(f"  Critical positions in Method 2 (old): {len(critical_in_2)}/{len(all_critical)}")
    print(f"  Critical positions in Method 3 (fixed): {len(critical_in_3)}/{len(all_critical)}")

    # Detailed breakdown by year
    positions_1216_in_1 = set(critical_positions_1216) & set_1
    positions_1216_in_2 = set(critical_positions_1216) & set_2
    positions_1216_in_3 = set(critical_positions_1216) & set_3
    positions_1220_in_1 = set(critical_positions_1220) & set_1
    positions_1220_in_2 = set(critical_positions_1220) & set_2
    positions_1220_in_3 = set(critical_positions_1220) & set_3

    print(f"\n  '1216' positions in Method 1: {len(positions_1216_in_1)}/{len(critical_positions_1216)}")
    print(f"  '1216' positions in Method 2: {len(positions_1216_in_2)}/{len(critical_positions_1216)}")
    print(f"  '1216' positions in Method 3 (fixed): {len(positions_1216_in_3)}/{len(critical_positions_1216)}")
    print(f"  '1220' positions in Method 1: {len(positions_1220_in_1)}/{len(critical_positions_1220)}")
    print(f"  '1220' positions in Method 2: {len(positions_1220_in_2)}/{len(critical_positions_1220)}")
    print(f"  '1220' positions in Method 3 (fixed): {len(positions_1220_in_3)}/{len(critical_positions_1220)}")

    # Show which specific positions are missing
    if len(positions_1216_in_3) < len(critical_positions_1216):
        missing = set(critical_positions_1216) - positions_1216_in_3
        print(f"\n  Method 3 (fixed) missing '1216' positions: {sorted(missing)}")
    if len(positions_1220_in_3) < len(critical_positions_1220):
        missing = set(critical_positions_1220) - positions_1220_in_3
        print(f"  Method 3 (fixed) missing '1220' positions: {sorted(missing)}")

    # Check surrounding context (±10 tokens around each answer occurrence)
    print("\n  Checking context coverage (±10 tokens around each year):")
    context_range = 10
    for year, positions in [("1216", positions_1216), ("1220", positions_1220)]:
        for start_pos in positions:
            context_start = max(0, start_pos - context_range)
            context_end = min(total_doc_len, start_pos + 4 + context_range)
            context_set = set(range(context_start, context_end))

            in_1 = len(context_set & set_1)
            in_2 = len(context_set & set_2)
            total_context = len(context_set)

            context_tokens = all_tokens[context_start:context_end]
            context_text = tokenizer.decode(context_tokens)[:60]

            print(f"    {year} at pos {start_pos}: M1={in_1}/{total_context}, M2={in_2}/{total_context}"
                  f" | '{context_text}...'")

    # Attention scores at answer positions
    print("\n  Attention scores at answer positions:")
    for year, positions in [("1216", positions_1216), ("1220", positions_1220)]:
        for start_pos in positions:
            attn_1 = [attn_1_agg[p] for p in range(start_pos, min(start_pos+4, len(attn_1_agg)))]
            attn_2 = [attn_2_agg[p] for p in range(start_pos, min(start_pos+4, len(attn_2_agg)))]
            avg_1 = np.mean(attn_1)
            avg_2 = np.mean(attn_2)
            print(f"    {year} at pos {start_pos}: M1 avg={avg_1:.6f}, M2 avg={avg_2:.6f}"
                  f" (M1 in set: {start_pos in set_1}, M2 in set: {start_pos in set_2})")

    print("\n" + "="*100)
    print("Analysis Complete")
    print("="*100)


if __name__ == '__main__':
    main()

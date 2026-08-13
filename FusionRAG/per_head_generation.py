#!/usr/bin/env python3
"""
Per-Layer Token Selection and Recomputation

实现逐层 token 选择 + 手动重算的完整流程
- 版本1：分层选择（所有 KV heads 共享）
- 保留接口支持未来扩展到分层分头选择
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import math

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import rotate_half
from ktransformers.operators.per_head_sparse_attention import per_head_sparse_attention, prepare_per_head_kv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def analyze_attention_distribution(layer_head_attention_scores, total_doc_len, output_dir='./attention_analysis'):
    """
    分析 attention scores 的分布特征

    输出：
    1. 每层每个 head 的统计信息（mean, std, max, percentiles）
    2. 可视化图表
    3. 推荐的选择策略
    """

    os.makedirs(output_dir, exist_ok=True)

    num_layers = len(layer_head_attention_scores)
    num_kv_heads = len(layer_head_attention_scores[0])

    print(f"\n{'='*100}")
    print("Attention Distribution Analysis")
    print(f"{'='*100}\n")

    # 收集统计信息
    layer_stats = {}

    for layer_idx in range(num_layers):
        layer_scores_all_heads = []
        head_stats = {}

        for kv_head_idx in range(num_kv_heads):
            scores = layer_head_attention_scores[layer_idx][kv_head_idx]
            layer_scores_all_heads.extend(scores)

            # 计算统计量
            head_stats[kv_head_idx] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'min': np.min(scores),
                'p99': np.percentile(scores, 99),
                'p95': np.percentile(scores, 95),
                'p90': np.percentile(scores, 90),
                'p75': np.percentile(scores, 75),
                'p50': np.percentile(scores, 50),
            }

        # 整层统计
        layer_scores_all_heads = np.array(layer_scores_all_heads)
        layer_stats[layer_idx] = {
            'heads': head_stats,
            'layer_mean': np.mean(layer_scores_all_heads),
            'layer_std': np.std(layer_scores_all_heads),
            'layer_max': np.max(layer_scores_all_heads),
            'layer_p99': np.percentile(layer_scores_all_heads, 99),
            'layer_p95': np.percentile(layer_scores_all_heads, 95),
            'layer_p90': np.percentile(layer_scores_all_heads, 90),
        }

    # 打印统计信息
    print("Layer-wise Statistics:")
    print(f"{'Layer':<6} {'Mean':<10} {'Std':<10} {'Max':<10} {'P99':<10} {'P95':<10} {'P90':<10}")
    print("-" * 70)

    for layer_idx in range(num_layers):
        stats = layer_stats[layer_idx]
        if layer_idx % 2 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx:<6} {stats['layer_mean']:<10.6f} {stats['layer_std']:<10.6f} "
                  f"{stats['layer_max']:<10.6f} {stats['layer_p99']:<10.6f} "
                  f"{stats['layer_p95']:<10.6f} {stats['layer_p90']:<10.6f}")

    print()

    # 可视化：每层的分布特征
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 每层的平均注意力分数
    layer_means = [layer_stats[i]['layer_mean'] for i in range(num_layers)]
    axes[0, 0].plot(range(num_layers), layer_means, 'o-')
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Mean Attention Score')
    axes[0, 0].set_title('Average Attention Score per Layer')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 每层的标准差
    layer_stds = [layer_stats[i]['layer_std'] for i in range(num_layers)]
    axes[0, 1].plot(range(num_layers), layer_stds, 'o-', color='orange')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Std of Attention Score')
    axes[0, 1].set_title('Std of Attention Score per Layer')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 每层的 Percentiles
    layer_p99 = [layer_stats[i]['layer_p99'] for i in range(num_layers)]
    layer_p95 = [layer_stats[i]['layer_p95'] for i in range(num_layers)]
    layer_p90 = [layer_stats[i]['layer_p90'] for i in range(num_layers)]

    axes[1, 0].plot(range(num_layers), layer_p99, 'o-', label='P99', alpha=0.7)
    axes[1, 0].plot(range(num_layers), layer_p95, 's-', label='P95', alpha=0.7)
    axes[1, 0].plot(range(num_layers), layer_p90, '^-', label='P90', alpha=0.7)
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Percentile Value')
    axes[1, 0].set_title('Attention Score Percentiles per Layer')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 选择几层的分布直方图
    selected_layers = [0, num_layers // 4, num_layers // 2, num_layers - 1]
    colors = ['blue', 'green', 'orange', 'red']

    for idx, layer_idx in enumerate(selected_layers):
        scores = []
        for kv_head_idx in range(num_kv_heads):
            scores.extend(layer_head_attention_scores[layer_idx][kv_head_idx])

        axes[1, 1].hist(scores, bins=50, alpha=0.5, label=f'Layer {layer_idx}', color=colors[idx])

    axes[1, 1].set_xlabel('Attention Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Attention Score Distribution (Selected Layers)')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'attention_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to {plot_path}\n")
    plt.close()

    # 分析选择策略
    print("="*100)
    print("Selection Strategy Analysis")
    print("="*100)
    print()

    # 计算如果用不同的 threshold，每层会选择多少 tokens
    thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]

    print("If using ABSOLUTE threshold (# tokens selected per layer):")
    print(f"{'Layer':<6}", end='')
    for th in thresholds:
        print(f"{'th=' + str(th):<12}", end='')
    print()
    print("-" * (6 + 12 * len(thresholds)))

    threshold_selections = {th: [] for th in thresholds}

    for layer_idx in range(num_layers):
        if layer_idx % 2 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx:<6}", end='')

            for th in thresholds:
                # Union of all heads
                union_set = set()
                for kv_head_idx in range(num_kv_heads):
                    scores = layer_head_attention_scores[layer_idx][kv_head_idx]
                    selected = np.where(scores > th)[0]
                    union_set.update(selected)

                num_selected = len(union_set)
                threshold_selections[th].append(num_selected)
                print(f"{num_selected:<12}", end='')

            print()

    print()

    # 计算如果用 percentile threshold
    print("If using PERCENTILE threshold (# tokens selected per layer):")
    percentiles = [99, 95, 90, 85, 80, 75]

    print(f"{'Layer':<6}", end='')
    for p in percentiles:
        print(f"{'P' + str(p):<12}", end='')
    print()
    print("-" * (6 + 12 * len(percentiles)))

    percentile_selections = {p: [] for p in percentiles}

    for layer_idx in range(num_layers):
        if layer_idx % 2 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx:<6}", end='')

            for p in percentiles:
                # Union of all heads
                union_set = set()
                for kv_head_idx in range(num_kv_heads):
                    scores = layer_head_attention_scores[layer_idx][kv_head_idx]
                    threshold = np.percentile(scores, p)
                    selected = np.where(scores >= threshold)[0]
                    union_set.update(selected)

                num_selected = len(union_set)
                percentile_selections[p].append(num_selected)
                print(f"{num_selected:<12}", end='')

            print()

    print()
    print("="*100)
    print()

    # 保存统计数据
    stats_path = os.path.join(output_dir, 'attention_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'layer_stats': {
                str(k): {
                    'layer_mean': float(v['layer_mean']),
                    'layer_std': float(v['layer_std']),
                    'layer_max': float(v['layer_max']),
                    'layer_p99': float(v['layer_p99']),
                    'layer_p95': float(v['layer_p95']),
                    'layer_p90': float(v['layer_p90']),
                }
                for k, v in layer_stats.items()
            },
            'threshold_selections': {str(k): [int(x) for x in v] for k, v in threshold_selections.items()},
            'percentile_selections': {str(k): [int(x) for x in v] for k, v in percentile_selections.items()},
        }, f, indent=2)

    print(f"Saved statistics to {stats_path}")

    return layer_stats


def compute_query_attention_scores(model, past_key_values, query_tensor, total_cache_len, prefix_len, doc_len, device):
    """
    计算 query 对文档每个位置的 attention 分数

    返回: {layer_idx: attention_scores} 其中 attention_scores 是 numpy array, shape=(doc_len,)
    """
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

            # 提取文档部分的 attention
            doc_attn = attn_weights[0, :, :, prefix_len:prefix_len + doc_len]
            # 对 query tokens 和 heads 平均
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


def find_connected_components(positions, max_gap=2):
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


def smart_query_selection(attention_scores, doc_len, target_ratio, num_layers):
    """
    Smart Query Selection: 使用连通性分析确保相关 token 群组被完整选中

    核心思想：
    1. 聚合多层 attention (Layer 16, 20, 24, 27)
    2. 找到高 attention 位置 (> mean + 0.5 * std)
    3. 使用连通分量分析找到 token 群组
    4. 按群组总 attention 排序，贪心选择
    5. 扩展上下文 (±1)

    返回: {layer_idx: selected_positions}
    """
    # 使用后几层的 attention
    layers_to_use = [l for l in [16, 20, 24, 27] if l < num_layers]
    if not layers_to_use:
        layers_to_use = [num_layers - 1]

    multi_layer_attn = np.stack([attention_scores[i] for i in layers_to_use]).mean(axis=0)

    target_count = int(doc_len * target_ratio)

    # Step 1: 找到高 attention 位置
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    threshold = mean_attn + 0.5 * std_attn

    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])

    print(f"  High attention positions (>mean+0.5*std): {len(high_attn_positions)}")

    # Step 2: 找到连通群组
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

    # 所有层使用相同的选择
    layer_selections = {}
    for layer_idx in range(num_layers):
        layer_selections[layer_idx] = selected_list

    return layer_selections


def compute_layerwise_token_selection(
    model,
    tokenizer,
    passages,
    chunk_ids,
    query_tensor,
    example_id,
    full_cache_path,
    total_ratio=0.3,
    device="cuda:0",
    per_head_selection=True,  # 是否返回逐头选择结果
    selection_strategy="independent",  # 选择策略: "independent", "union_constrained", "greedy_union", "layer_wise", "nested", "threshold"
    scoring_method="query"  # 评分方法: "query", "reconstruction", 或 "query_attention" (Smart Query Selection)
):
    """
    计算每层每头需要重算的 token 选择（逐层逐头版本）

    流程：
    1. 加载 document KV caches，forward query/reconstruction task
    2. 计算每层每个 KV head 的 attention scores
    3. 为每层每头设计选择策略
    4. 返回逐层逐头的选择结果

    选择策略：
    - "independent": 每个 head 独立选择 top-k，不同层/头可以完全不同
    - "layer_wise": 每层选择不同 tokens，但同层各 head 选相同 tokens
    - "nested": 嵌套策略，深层选择是浅层的子集
    - "threshold": 基于 attention 阈值选择，不固定比例

    评分方法：
    - "query": 使用用户问题的 attention 来评分（query-dependent）
    - "reconstruction": 使用文本重构任务的 attention 来评分（query-agnostic，KVzip 风格）
    - "query_attention": Smart Query Selection - 使用 query attention + 连通分量分析（推荐）

    参数：
        total_ratio: 总体平均重算比例（默认 0.3）
        per_head_selection: 是否返回逐头选择结果（默认 True）
        scoring_method: 评分方法

    返回：
        layer_head_selections: {
            layer_idx: {
                kv_head_idx: {
                    'positions': [...],      # 该层该头需要重算的位置
                    'scores': [...],         # 对应的 attention scores
                }
            }
        }
        layer_union_positions: {
            layer_idx: [...]  # 该层所有头的并集（用于构建 sparse 序列）
        }
    """

    print(f"\n{'='*100}")
    print("Step 1: Computing per-layer token selections")
    print(f"{'='*100}\n")

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_key_value_groups = num_heads // num_kv_heads

    print(f"Model config:")
    print(f"  Layers: {num_layers}")
    print(f"  Attention heads: {num_heads}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Total ratio: {total_ratio}\n")

    # Special case: if total_ratio >= 1, select ALL tokens in ALL layers for ALL heads
    if total_ratio >= 1.0:
        print("Special case: total_ratio >= 1.0, selecting ALL tokens in ALL layers for ALL heads\n")

        # Build position to token mapping
        # 注意：必须按照 doc_chunk_ids 的顺序来遍历文档，因为 KV cache 是按这个顺序加载的
        # chunk_id=0 是 system prompt，其他 chunk_id 是文档
        # chunk_id - 1 就是 passages 中的索引（chunk_id 从 1 开始）
        position_to_token = {}
        current_pos = 0
        total_doc_len = 0
        # 跳过 chunk_id=0 (system prompt)，只处理文档 chunks
        doc_chunk_ids_docs = [cid for cid in chunk_ids if cid != 0]
        for chunk_id in doc_chunk_ids_docs:
            passage = passages[chunk_id - 1]  # chunk_id 从 1 开始，所以 -1
            # 注意：cache 是用 "Document: {doc}\n" 格式生成的，这里要保持一致
            doc_text = f"Document: {passage}\n"
            tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            for i, token in enumerate(tokens):
                position_to_token[current_pos + i] = token
            current_pos += len(tokens)
            total_doc_len += len(tokens)

        print(f"Total document length: {total_doc_len} tokens\n")

        # Create selection: all positions for all layers and all heads
        all_positions = list(range(total_doc_len))

        layer_head_selections = {}
        layer_union_positions = {}

        for layer_idx in range(num_layers):
            layer_head_selections[layer_idx] = {}
            for kv_head_idx in range(num_kv_heads):
                layer_head_selections[layer_idx][kv_head_idx] = {
                    'positions': all_positions,
                    'scores': [1.0] * total_doc_len,  # Dummy scores
                    'threshold': 0.0,
                    'num_selected': total_doc_len
                }
            layer_union_positions[layer_idx] = all_positions

        print(f"{'='*100}")
        print("Selection Statistics Summary")
        print(f"{'='*100}")

        total_recompute_tokens = total_doc_len * num_layers * num_kv_heads
        total_possible = total_doc_len * num_layers * num_kv_heads
        overall_ratio = 1.0

        print(f"\nTotal document tokens: {total_doc_len}")
        print(f"Total layers: {num_layers}")
        print(f"Total KV heads per layer: {num_kv_heads}")
        print(f"Total possible (layer × head × token): {total_possible}")
        print(f"Actual per-head selections: {total_recompute_tokens}")
        print(f"Overall selection ratio: {overall_ratio*100:.2f}%")

        print(f"\n{'='*100}")
        print("Selection complete")
        print(f"{'='*100}\n")

        return layer_head_selections, layer_union_positions, total_doc_len, position_to_token

    # Load document KV caches (skip chunk_id=0 which is system prompt)
    doc_key_caches = []
    doc_value_caches = []
    doc_chunk_ids = [cid for cid in chunk_ids if cid != 0]

    for chunk_id in doc_chunk_ids:
        try:
            chunk_key = torch.load(
                f'{full_cache_path}/{example_id}_{chunk_id}_key.pt',
                weights_only=True
            ).to(device)
            chunk_value = torch.load(
                f'{full_cache_path}/{example_id}_{chunk_id}_value.pt',
                weights_only=True
            ).to(device)
            doc_key_caches.append(chunk_key)
            doc_value_caches.append(chunk_value)
        except FileNotFoundError:
            print(f"Warning: Cache file not found for chunk {chunk_id}")
            continue

    total_doc_len = sum([cache[0].shape[2] for cache in doc_key_caches])
    print(f"Loaded {len(doc_key_caches)} document caches")
    print(f"Total document length: {total_doc_len} tokens\n")

    # Initialize cache and load document caches
    max_cache_len = 32768
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Load document caches
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] = 0
        current_pos = 0

        for cache_idx in range(len(doc_key_caches)):
            layer_key = doc_key_caches[cache_idx][layer_idx]
            layer_value = doc_value_caches[cache_idx][layer_idx]
            layer_len = layer_key.shape[2]

            past_key_values.key_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len
            current_pos += layer_len

    # Forward and compute attention scores per layer per head
    print(f"Computing attention scores using '{scoring_method}' method...\n")

    if scoring_method == "reconstruction":
        # ====================================================================
        # KVzip 风格：使用文本重构任务来评分
        # ====================================================================
        # 核心思想：让模型"重复"文档内容，看它需要关注哪些 tokens
        # 这是 query-agnostic 的评分方式
        #
        # 输入格式：[repeat_prompt] + [document_chunk]
        # 评分：只取 repeat_prompt 部分对 document 的 attention (不包括 target_tokens)
        # ====================================================================

        print("Using RECONSTRUCTION scoring (KVzip style)")
        print("  Scoring is query-agnostic (based on document structure)")

        # 构建重复任务的输入
        repeat_prompt = "\n\nRepeat the previous context exactly:\n"
        repeat_prompt_ids = tokenizer.encode(repeat_prompt, add_special_tokens=False)

        # 取文档的一部分作为重复目标（避免序列过长）
        chunk_size = min(512, total_doc_len)

        # 跳过 chunk_id=0 (system prompt)，只处理文档 chunks
        doc_chunk_ids_list = [cid for cid in chunk_ids if cid != 0]
        all_doc_tokens = []
        for chunk_id in doc_chunk_ids_list:
            passage = passages[chunk_id - 1]
            doc_text = f"Document: {passage}\n"
            tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            all_doc_tokens.extend(tokens)

        # 取前 chunk_size 个 tokens 作为重复目标
        target_tokens = all_doc_tokens[:chunk_size]

        # 完整输入：repeat_prompt + target_tokens
        scoring_input_ids = repeat_prompt_ids + target_tokens
        scoring_tensor = torch.tensor([scoring_input_ids], dtype=torch.long, device=device)
        scoring_len = len(scoring_input_ids)

        # Position IDs：从文档末尾开始
        cache_position = torch.arange(total_doc_len, total_doc_len + scoring_len, device=device)

        print(f"  Repeat prompt: {len(repeat_prompt_ids)} tokens")
        print(f"  Target chunk: {len(target_tokens)} tokens")
        print(f"  Total scoring input: {scoring_len} tokens")

        # 用于计算 attention 的范围
        # 我们关心的是 repeat_prompt 对 document 的 attention
        query_start = 0  # repeat_prompt 开始
        query_end = len(repeat_prompt_ids)  # repeat_prompt 结束
        print(f"  Scoring query range: [{query_start}, {query_end})")

    elif scoring_method == "query_attention":
        # ====================================================================
        # Smart Query Selection: 使用 query attention + 连通分量分析
        # ====================================================================
        print("Using SMART QUERY ATTENTION selection")
        print("  Query-aware selection with connected component analysis")

        # 需要加载 system prompt cache 以保持位置一致性
        # 加载 chunk 0 (system prompt)
        prefix_len = 0
        if 0 in chunk_ids:
            try:
                sys_key = torch.load(f'{full_cache_path}/{example_id}_0_key.pt', weights_only=True).to(device)
                sys_value = torch.load(f'{full_cache_path}/{example_id}_0_value.pt', weights_only=True).to(device)
                sys_len = sys_key[0].shape[2]
                prefix_len = sys_len

                # 重新加载 cache：先加载 system prompt，再加载 documents
                for layer_idx in range(num_layers):
                    past_key_values.past_tokens[layer_idx] = 0

                    # 先加载 system prompt
                    past_key_values.key_cache[layer_idx].narrow(2, 0, sys_len).copy_(sys_key[layer_idx])
                    past_key_values.value_cache[layer_idx].narrow(2, 0, sys_len).copy_(sys_value[layer_idx])
                    past_key_values.past_tokens[layer_idx] = sys_len

                    # 再加载 documents
                    current_pos = sys_len
                    for cache_idx in range(len(doc_key_caches)):
                        layer_key = doc_key_caches[cache_idx][layer_idx]
                        layer_value = doc_value_caches[cache_idx][layer_idx]
                        layer_len = layer_key.shape[2]

                        past_key_values.key_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_key)
                        past_key_values.value_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_value)
                        past_key_values.past_tokens[layer_idx] += layer_len
                        current_pos += layer_len

                print(f"  Loaded system prompt: {prefix_len} tokens")
            except FileNotFoundError:
                print("  Warning: System prompt cache not found, using prefix_len=0")
                prefix_len = 0

        total_cache_len = prefix_len + total_doc_len
        print(f"  Total cache length: {total_cache_len} (prefix={prefix_len}, doc={total_doc_len})")

        # 计算 query attention scores
        print("\n  Computing query attention scores...")
        query_attention_scores = compute_query_attention_scores(
            model, past_key_values, query_tensor, total_cache_len, prefix_len, total_doc_len, device
        )

        # 使用 smart selection 算法
        print("\n  Applying smart selection algorithm...")
        layer_smart_selections = smart_query_selection(
            query_attention_scores, total_doc_len, total_ratio, num_layers
        )

        # Build position to token mapping
        position_to_token = {}
        current_pos = 0
        doc_chunk_ids_docs = [cid for cid in chunk_ids if cid != 0]
        for chunk_id in doc_chunk_ids_docs:
            passage = passages[chunk_id - 1]
            doc_text = f"Document: {passage}\n"
            tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            for i, token in enumerate(tokens):
                position_to_token[current_pos + i] = token
            current_pos += len(tokens)

        # 构建 layer_head_selections 和 layer_union_positions
        layer_head_selections = {}
        layer_union_positions = {}

        for layer_idx in range(num_layers):
            selected_positions = layer_smart_selections[layer_idx]
            layer_union_positions[layer_idx] = selected_positions

            layer_head_selections[layer_idx] = {}
            for kv_head_idx in range(num_kv_heads):
                layer_head_selections[layer_idx][kv_head_idx] = {
                    'positions': selected_positions.copy(),
                    'scores': [query_attention_scores[layer_idx][p] if p < len(query_attention_scores[layer_idx]) else 0 for p in selected_positions],
                    'threshold': 0.0,
                    'num_selected': len(selected_positions)
                }

        # 打印统计信息
        print(f"\n{'='*100}")
        print("Smart Query Selection Statistics")
        print(f"{'='*100}")
        print(f"\nTotal document tokens: {total_doc_len}")
        print(f"Target ratio: {total_ratio*100:.1f}%")
        print(f"Selected tokens: {len(layer_union_positions[0])} ({len(layer_union_positions[0])/total_doc_len*100:.1f}%)")
        print(f"\n{'='*100}\n")

        return layer_head_selections, layer_union_positions, total_doc_len, position_to_token

    else:  # scoring_method == "query"
        print("Using QUERY-BASED scoring")
        print("  Scoring is based on user question attention to document")
        # ====================================================================
        # 默认方法：使用用户问题来评分
        # ====================================================================
        scoring_tensor = query_tensor
        scoring_len = query_tensor.shape[1]
        cache_position = torch.arange(total_doc_len, total_doc_len + scoring_len, device=device)
        query_start = 0
        query_end = scoring_len

    inputs_embeds = model.model.embed_tokens(scoring_tensor).to(device)
    hidden_states = inputs_embeds
    position_ids = cache_position.unsqueeze(0)

    # Store attention scores for each layer and head
    layer_head_attention_scores = {}

    with torch.no_grad():
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            # Get document K, V from cache
            doc_keys = past_key_values.key_cache[layer_idx][:, :, :total_doc_len, :]
            doc_values = past_key_values.value_cache[layer_idx][:, :, :total_doc_len, :]

            # Compute query Q, K, V projections
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)

            # Reshape
            bsz, q_len, _ = hidden_states.size()
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)

            def apply_rotary_pos_emb(q, k, cos, sin):
                q_embed = (q * cos) + (rotate_half(q) * sin)
                k_embed = (k * cos) + (rotate_half(k) * sin)
                return q_embed, k_embed

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Compute per-head attention scores
            layer_head_scores = {}

            for kv_head_idx in range(num_kv_heads):
                # Get document keys for this KV head
                doc_k_head = doc_keys[:, kv_head_idx, :, :]  # [1, doc_len, head_dim]

                # Get query states for heads that use this KV head (GQA)
                start_q_head = kv_head_idx * num_key_value_groups
                end_q_head = start_q_head + num_key_value_groups
                query_heads = query_states[:, start_q_head:end_q_head, :, :]  # [1, num_groups, q_len, head_dim]

                # 只使用指定范围的 query tokens 来计算 attention
                # - query 方法: 使用所有用户问题 tokens
                # - reconstruction 方法: 只使用 repeat_prompt tokens (不包括 target_tokens)
                query_heads_subset = query_heads[:, :, query_start:query_end, :]  # [1, num_groups, subset_len, head_dim]

                # 计算 attention: Q @ K^T
                # query_heads_subset: [1, num_groups, subset_len, head_dim]
                # doc_k_head: [1, doc_len, head_dim]
                # 结果: [1, num_groups, subset_len, doc_len]
                attn_weights = torch.matmul(
                    query_heads_subset,
                    doc_k_head.unsqueeze(1).transpose(2, 3)
                )
                attn_weights = attn_weights / math.sqrt(head_dim)
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

                # KVzip 风格: 对每个 document token，取所有 query positions 的最大 attention
                # 这代表了该 token 被任意 query token 关注的最大程度
                # 先在 query positions 维度取 max，再在 query heads 维度取 max
                attn_max_over_queries = attn_weights.max(dim=2)[0]  # [1, num_groups, doc_len]
                attn_max_over_heads = attn_max_over_queries.max(dim=1)[0]  # [1, doc_len]

                attention_scores = attn_max_over_heads[0, :].float().cpu().numpy()
                layer_head_scores[kv_head_idx] = attention_scores

            layer_head_attention_scores[layer_idx] = layer_head_scores

            # Continue forward pass for next layer
            key_states_full = torch.cat([doc_keys, key_states], dim=2)
            value_states_full = torch.cat([doc_values, value_states], dim=2)

            # Repeat KV for GQA
            key_states_full = key_states_full.repeat_interleave(num_key_value_groups, dim=1)
            value_states_full = value_states_full.repeat_interleave(num_key_value_groups, dim=1)

            # Compute attention output
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states_full, value_states_full, attn_mask=None, dropout_p=0.0
            )

            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            # Residual + LayerNorm + MLP
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = hidden_states + attn_output
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

            if layer_idx % 5 == 0 or layer_idx == num_layers - 1:
                print(f"  Layer {layer_idx:2d} done")

    # 分析 attention 分布
    analyze_attention_distribution(layer_head_attention_scores, total_doc_len)

    print(f"\nComputing layer-wise PER-HEAD selections using {selection_strategy.upper()} strategy...\n")

    # Build position to token mapping
    # 注意：必须按照 doc_chunk_ids 的顺序来遍历文档，因为 KV cache 是按这个顺序加载的
    # chunk_id=0 是 system prompt，其他 chunk_id 是文档
    # chunk_id - 1 就是 passages 中的索引（chunk_id 从 1 开始）
    position_to_token = {}
    current_pos = 0
    # 跳过 chunk_id=0 (system prompt)，只处理文档 chunks
    doc_chunk_ids_docs = [cid for cid in chunk_ids if cid != 0]
    for chunk_id in doc_chunk_ids_docs:
        passage = passages[chunk_id - 1]  # chunk_id 从 1 开始，所以 -1
        # 注意：cache 是用 "Document: {doc}\n" 格式生成的，这里要保持一致
        doc_text = f"Document: {passage}\n"
        tokens = tokenizer.encode(doc_text, add_special_tokens=False)
        for i, token in enumerate(tokens):
            position_to_token[current_pos + i] = token
        current_pos += len(tokens)

    # ========================================================================
    # 选择策略分支
    # ========================================================================

    if selection_strategy == "independent":
        # ====================================================================
        # 独立选择策略（Independent Per-Head Selection）
        # ====================================================================
        # 核心思想：
        # 1. 每个 head 在每层独立选择 top-k tokens，基于自己的 attention 分布
        # 2. 不同层、不同 head 可以选择完全不同的 tokens
        # 3. 所有层使用相同的目标比例 total_ratio
        # ====================================================================

        print("Using INDEPENDENT selection strategy")
        print("  - Each head independently selects based on its own attention scores")
        print("  - No nested constraint between layers")
        print(f"  - Target ratio per head: {total_ratio*100:.1f}%\n")

        target_per_head = max(int(total_doc_len * total_ratio), 20)

        layer_head_selections = {}
        layer_union_positions = {}

        # 收集每层每头的 attention scores
        layer_all_scores = {}
        for layer_idx in range(num_layers):
            layer_all_scores[layer_idx] = []
            for kv_head_idx in range(num_kv_heads):
                scores = layer_head_attention_scores[layer_idx][kv_head_idx]
                layer_all_scores[layer_idx].append(scores)

        # 每层每头独立选择
        for layer_idx in range(num_layers):
            all_head_scores = layer_all_scores[layer_idx]
            head_selections = {}
            layer_union = set()

            for kv_head_idx in range(num_kv_heads):
                scores = all_head_scores[kv_head_idx]

                # 按该 head 的 attention 排序，选 top-k
                sorted_positions = np.argsort(scores)[::-1]
                selected_positions = sorted_positions[:target_per_head].tolist()
                selected_scores = [scores[p] for p in selected_positions]

                head_selections[kv_head_idx] = {
                    'positions': selected_positions,
                    'scores': selected_scores,
                    'threshold': scores[sorted_positions[min(target_per_head-1, len(sorted_positions)-1)]] if len(sorted_positions) > 0 else 0,
                    'num_selected': len(selected_positions)
                }

                layer_union.update(selected_positions)

            layer_head_selections[layer_idx] = head_selections
            layer_union_positions[layer_idx] = sorted(list(layer_union))

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                head_counts = [len(layer_head_selections[layer_idx][h]['positions']) for h in range(num_kv_heads)]
                min_count, max_count = min(head_counts), max(head_counts)
                avg_count = sum(head_counts) / len(head_counts)
                union_count = len(layer_union_positions[layer_idx])
                print(f"  Layer {layer_idx:2d}: per-head={target_per_head:4d}, "
                      f"union={union_count:4d} ({union_count/total_doc_len*100:5.1f}%)")

    elif selection_strategy == "union_constrained":
        # ====================================================================
        # Union 约束选择策略（Union-Constrained Per-Head Selection）
        # ====================================================================
        # 核心思想：
        # 1. 控制 union 大小 = doc_len × ratio（而不是 per-head 大小）
        # 2. 先确定 union pool（按所有 heads 的 max attention 选择）
        # 3. 每个 head 在 union pool 内按自己的 attention 排序
        # 4. 这样 union 精确等于 ratio，但 heads 内部排序不同
        # ====================================================================

        print("Using UNION_CONSTRAINED selection strategy")
        print("  - Union size is constrained to ratio%")
        print("  - Each head selects ALL tokens in union, but with different priority")
        print(f"  - Target union ratio: {total_ratio*100:.1f}%\n")

        target_union_size = max(int(total_doc_len * total_ratio), 20)

        layer_head_selections = {}
        layer_union_positions = {}

        # 收集每层每头的 attention scores
        layer_all_scores = {}
        for layer_idx in range(num_layers):
            layer_all_scores[layer_idx] = []
            for kv_head_idx in range(num_kv_heads):
                scores = layer_head_attention_scores[layer_idx][kv_head_idx]
                layer_all_scores[layer_idx].append(scores)

        for layer_idx in range(num_layers):
            all_head_scores = layer_all_scores[layer_idx]

            # Step 1: 计算每个位置的 max attention（跨所有 heads）
            position_max_scores = np.zeros(total_doc_len)
            for kv_head_idx in range(num_kv_heads):
                position_max_scores = np.maximum(position_max_scores, all_head_scores[kv_head_idx])

            # Step 2: 选择 top-k 构成 union pool
            sorted_positions = np.argsort(position_max_scores)[::-1]
            union_positions = sorted_positions[:target_union_size].tolist()
            union_set = set(union_positions)

            # Step 3: 每个 head 在 union 内按自己的 attention 排序
            head_selections = {}
            for kv_head_idx in range(num_kv_heads):
                scores = all_head_scores[kv_head_idx]

                # 在 union 内按该 head 的 attention 排序
                head_ranked = [(p, scores[p]) for p in union_positions]
                head_ranked.sort(key=lambda x: x[1], reverse=True)

                selected_positions = [p for p, _ in head_ranked]
                selected_scores = [scores[p] for p in selected_positions]

                head_selections[kv_head_idx] = {
                    'positions': selected_positions,
                    'scores': selected_scores,
                    'threshold': min(selected_scores) if selected_scores else 0,
                    'num_selected': len(selected_positions)
                }

            layer_head_selections[layer_idx] = head_selections
            layer_union_positions[layer_idx] = sorted(union_positions)

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                head_counts = [len(layer_head_selections[layer_idx][h]['positions']) for h in range(num_kv_heads)]
                union_count = len(layer_union_positions[layer_idx])
                print(f"  Layer {layer_idx:2d}: per-head={head_counts[0]:4d}, "
                      f"union={union_count:4d} ({union_count/total_doc_len*100:5.1f}%)")

    elif selection_strategy == "greedy_union":
        # ====================================================================
        # 贪心 Union 选择策略（Greedy Union Selection）
        # ====================================================================
        # 核心思想：
        # 1. 控制 union 大小 = doc_len × ratio
        # 2. 贪心地选择 tokens：优先选择对多个 heads 都重要的 tokens
        # 3. 每个 head 可以选择 union 的子集（不一定全选）
        # ====================================================================

        print("Using GREEDY_UNION selection strategy")
        print("  - Greedily select tokens that benefit most heads")
        print(f"  - Target union ratio: {total_ratio*100:.1f}%\n")

        target_union_size = max(int(total_doc_len * total_ratio), 20)
        # 每个 head 选择 union 的 80%
        per_head_ratio_of_union = 0.8

        layer_head_selections = {}
        layer_union_positions = {}

        for layer_idx in range(num_layers):
            all_head_scores = layer_head_attention_scores[layer_idx]

            # Step 1: 计算每个位置的综合得分（所有 heads 的 attention 之和）
            # 这样选择的 tokens 是对多个 heads 都相对重要的
            position_sum_scores = np.zeros(total_doc_len)
            for kv_head_idx in range(num_kv_heads):
                # 归一化每个 head 的 scores
                scores = np.array(all_head_scores[kv_head_idx])
                max_score = scores.max()
                if max_score > 0:
                    scores = scores / max_score
                position_sum_scores += scores

            # Step 2: 选择综合得分最高的 tokens 构成 union
            sorted_positions = np.argsort(position_sum_scores)[::-1]
            union_positions = sorted_positions[:target_union_size].tolist()
            union_set = set(union_positions)

            # Step 3: 每个 head 从 union 中选择子集
            per_head_target = max(int(len(union_positions) * per_head_ratio_of_union), 20)

            head_selections = {}
            for kv_head_idx in range(num_kv_heads):
                scores = all_head_scores[kv_head_idx]

                # 在 union 内按该 head 的 attention 排序
                head_ranked = [(p, scores[p]) for p in union_positions]
                head_ranked.sort(key=lambda x: x[1], reverse=True)

                # 选择 top per_head_target
                selected_positions = [p for p, _ in head_ranked[:per_head_target]]
                selected_scores = [scores[p] for p in selected_positions]

                head_selections[kv_head_idx] = {
                    'positions': selected_positions,
                    'scores': selected_scores,
                    'threshold': selected_scores[-1] if selected_scores else 0,
                    'num_selected': len(selected_positions)
                }

            layer_head_selections[layer_idx] = head_selections
            layer_union_positions[layer_idx] = sorted(union_positions)

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                head_counts = [len(layer_head_selections[layer_idx][h]['positions']) for h in range(num_kv_heads)]
                min_count, max_count = min(head_counts), max(head_counts)
                union_count = len(layer_union_positions[layer_idx])
                print(f"  Layer {layer_idx:2d}: per-head={min_count:4d}-{max_count:4d}, "
                      f"union={union_count:4d} ({union_count/total_doc_len*100:5.1f}%)")

    elif selection_strategy == "layer_wise":
        # ====================================================================
        # 分层选择策略（Layer-wise Selection）
        # ====================================================================
        # 核心思想：
        # 1. 每层选择不同的 tokens（基于该层的 attention 分布）
        # 2. 同层内所有 heads 选择相同的 tokens
        # 3. 使用所有 heads 的 max attention 来决定 token 重要性
        # ====================================================================

        print("Using LAYER_WISE selection strategy")
        print("  - Different layers select different tokens")
        print("  - All heads within a layer select the SAME tokens")
        print(f"  - Target ratio: {total_ratio*100:.1f}%\n")

        target_per_layer = max(int(total_doc_len * total_ratio), 20)

        layer_head_selections = {}
        layer_union_positions = {}

        for layer_idx in range(num_layers):
            # 聚合该层所有 heads 的 attention scores（取 max）
            layer_max_scores = np.zeros(total_doc_len)
            for kv_head_idx in range(num_kv_heads):
                scores = layer_head_attention_scores[layer_idx][kv_head_idx]
                layer_max_scores = np.maximum(layer_max_scores, scores)

            # 选择 top-k tokens
            sorted_positions = np.argsort(layer_max_scores)[::-1]
            selected_positions = sorted_positions[:target_per_layer].tolist()
            selected_set = set(selected_positions)

            # 所有 heads 使用相同的选择
            head_selections = {}
            for kv_head_idx in range(num_kv_heads):
                scores = layer_head_attention_scores[layer_idx][kv_head_idx]
                selected_scores = [scores[p] for p in selected_positions]

                head_selections[kv_head_idx] = {
                    'positions': selected_positions.copy(),  # 所有 head 相同
                    'scores': selected_scores,
                    'threshold': layer_max_scores[sorted_positions[min(target_per_layer-1, len(sorted_positions)-1)]],
                    'num_selected': len(selected_positions)
                }

            layer_head_selections[layer_idx] = head_selections
            layer_union_positions[layer_idx] = sorted(selected_positions)

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                union_count = len(layer_union_positions[layer_idx])
                print(f"  Layer {layer_idx:2d}: selected={target_per_layer:4d}, "
                      f"union={union_count:4d} ({union_count/total_doc_len*100:5.1f}%)")

    elif selection_strategy == "threshold":
        # ====================================================================
        # 阈值选择策略（Threshold-Based Selection）
        # ====================================================================
        # 核心思想：
        # 1. 使用 attention percentile 阈值（如 95th percentile）
        # 2. 选择超过阈值的 tokens，而非固定数量
        # 3. 让每层每头自然决定需要多少 tokens
        # ====================================================================

        print("Using THRESHOLD selection strategy")
        # 将 total_ratio 转换为 percentile 阈值
        # ratio=0.3 → 选择 top 30% → percentile=70
        percentile_threshold = (1.0 - total_ratio) * 100
        print(f"  - Attention percentile threshold: {percentile_threshold:.1f}")
        print("  - Each head selects tokens above its own threshold\n")

        layer_head_selections = {}
        layer_union_positions = {}

        for layer_idx in range(num_layers):
            head_selections = {}
            layer_union = set()

            for kv_head_idx in range(num_kv_heads):
                scores = layer_head_attention_scores[layer_idx][kv_head_idx]

                # 计算该 head 的 attention 阈值
                threshold = np.percentile(scores, percentile_threshold)

                # 选择超过阈值的 tokens
                selected_positions = [i for i, s in enumerate(scores) if s >= threshold]
                selected_scores = [scores[p] for p in selected_positions]

                head_selections[kv_head_idx] = {
                    'positions': selected_positions,
                    'scores': selected_scores,
                    'threshold': threshold,
                    'num_selected': len(selected_positions)
                }

                layer_union.update(selected_positions)

            layer_head_selections[layer_idx] = head_selections
            layer_union_positions[layer_idx] = sorted(list(layer_union))

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                head_counts = [len(layer_head_selections[layer_idx][h]['positions']) for h in range(num_kv_heads)]
                min_count, max_count = min(head_counts), max(head_counts)
                union_count = len(layer_union_positions[layer_idx])
                print(f"  Layer {layer_idx:2d}: per-head min={min_count:4d}, max={max_count:4d}, "
                      f"union={union_count:4d} ({union_count/total_doc_len*100:5.1f}%)")

    else:  # selection_strategy == "nested"
        # ====================================================================
        # 嵌套选择策略（Nested Selection Strategy）
        # ====================================================================
        # 核心思想：
        # 1. 由于层间依赖性，深层选择的 token 必须在所有浅层也被选中
        # 2. 因此采用嵌套策略：Layer(i+1) 的选择 ⊆ Layer(i) 的选择
        # 3. 浅层选择更多 token（信息分布广），深层选择更少 token（信息汇聚）
        # ====================================================================

        print("Using NESTED selection strategy (shallow layers more, deep layers less)\n")

        # Step 1: 确定每层的目标选择比例
        # 使用递减策略：layer 0 最多，layer N-1 最少
        # 总体平均 = total_ratio
        #
        # 设计：
        # - 前 20% 层：选 shallow_ratio（较高）
        # - 中间 60% 层：线性递减
        # - 后 20% 层：选 core_ratio（较低）

        core_ratio = max(0.05, total_ratio * 0.2)  # 最深层至少选 total_ratio 的 20%
        shallow_ratio = min(total_ratio * 2.0, 0.8)  # 最浅层最多选 total_ratio 的 2 倍

        # 计算每层的目标比例
        layer_ratios = []
        for layer_idx in range(num_layers):
            t = layer_idx / (num_layers - 1) if num_layers > 1 else 0  # 0 到 1

            if t < 0.2:
                # 前 20% 层：保持高比例
                ratio = shallow_ratio
            elif t > 0.8:
                # 后 20% 层：保持低比例
                ratio = core_ratio
            else:
                # 中间 60% 层：线性递减
                mid_t = (t - 0.2) / 0.6  # 0 到 1
                ratio = shallow_ratio - (shallow_ratio - core_ratio) * mid_t

            layer_ratios.append(ratio)

        # 归一化使得平均值等于 total_ratio
        avg_ratio = sum(layer_ratios) / num_layers
        scale_factor = total_ratio / avg_ratio if avg_ratio > 0 else 1.0
        layer_ratios = [r * scale_factor for r in layer_ratios]

        print(f"Layer ratio distribution (target avg = {total_ratio*100:.0f}%):")
        print(f"  Layer  0: {layer_ratios[0]*100:.1f}%")
        print(f"  Layer {num_layers//4:2d}: {layer_ratios[num_layers//4]*100:.1f}%")
        print(f"  Layer {num_layers//2:2d}: {layer_ratios[num_layers//2]*100:.1f}%")
        print(f"  Layer {num_layers-1:2d}: {layer_ratios[num_layers-1]*100:.1f}%")
        print(f"  Actual avg: {sum(layer_ratios)/num_layers*100:.1f}%\n")

        # Step 2: 从深层到浅层，逐层构建嵌套选择
        layer_head_selections = {}
        layer_union_positions = {}

        # 收集每层每头的 attention scores
        layer_all_scores = {}
        for layer_idx in range(num_layers):
            layer_all_scores[layer_idx] = []
            for kv_head_idx in range(num_kv_heads):
                scores = layer_head_attention_scores[layer_idx][kv_head_idx]
                layer_all_scores[layer_idx].append(scores)

        # 从最深层开始，向浅层扩展（嵌套构建）
        previous_union = None

        for layer_idx in range(num_layers - 1, -1, -1):
            target_count = max(int(total_doc_len * layer_ratios[layer_idx]), 20)
            all_head_scores = layer_all_scores[layer_idx]

            # 计算该层每个位置的最大 attention
            layer_position_max = np.zeros(total_doc_len)
            for kv_head_idx in range(num_kv_heads):
                layer_position_max = np.maximum(layer_position_max, all_head_scores[kv_head_idx])

            if previous_union is None:
                # 最深层：直接选 top-k
                sorted_positions = np.argsort(layer_position_max)[::-1]
                current_union = set(sorted_positions[:target_count].tolist())
            else:
                # 浅层：必须包含深层的选择，然后扩展
                current_union = previous_union.copy()

                if len(current_union) < target_count:
                    remaining_count = target_count - len(current_union)
                    # 按该层 attention 排序，选择未被选中的 top-k
                    candidates = [(p, layer_position_max[p]) for p in range(total_doc_len) if p not in current_union]
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    for p, _ in candidates[:remaining_count]:
                        current_union.add(p)

            # 为每个 head 分配选择
            # 每个 head 选择与 layer ratio 一致的比例（而不是 union 的 50%）
            head_selections = {}
            per_head_target = max(int(total_doc_len * layer_ratios[layer_idx]), 20)

            for kv_head_idx in range(num_kv_heads):
                scores = all_head_scores[kv_head_idx]
                # 在 current_union 内按该 head 的 attention 排序
                head_candidates = [(p, scores[p]) for p in current_union]
                head_candidates.sort(key=lambda x: x[1], reverse=True)

                selected_positions = [p for p, _ in head_candidates[:per_head_target]]
                selected_scores = [scores[p] for p in selected_positions]

                head_selections[kv_head_idx] = {
                    'positions': selected_positions,
                    'scores': selected_scores,
                    'threshold': scores[head_candidates[min(per_head_target-1, len(head_candidates)-1)][0]] if head_candidates else 0,
                    'num_selected': len(selected_positions)
                }

            layer_head_selections[layer_idx] = head_selections
            layer_union_positions[layer_idx] = sorted(list(current_union))
            previous_union = current_union

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                # 统计该层各 head 的选择情况
                head_counts = [len(layer_head_selections[layer_idx][h]['positions']) for h in range(num_kv_heads)]
                min_count, max_count = min(head_counts), max(head_counts)
                avg_count = sum(head_counts) / len(head_counts)
                union_count = len(layer_union_positions[layer_idx])
                print(f"  Layer {layer_idx:2d}: per-head min={min_count:4d}, max={max_count:4d}, "
                      f"avg={avg_count:6.1f}, union={union_count:4d} ({union_count/total_doc_len*100:5.1f}%)")

    # ========================================================================
    # 统计信息（所有策略共享）
    # ========================================================================
    print(f"\n{'='*100}")
    print("Per-Head Selection Statistics")
    print(f"{'='*100}")

    # 统计总体情况
    total_head_selections = 0
    for layer_idx in range(num_layers):
        for kv_head_idx in range(num_kv_heads):
            total_head_selections += len(layer_head_selections[layer_idx][kv_head_idx]['positions'])

    total_possible = total_doc_len * num_layers * num_kv_heads
    overall_ratio = total_head_selections / total_possible if total_possible > 0 else 0

    print(f"\nTotal document tokens: {total_doc_len}")
    print(f"Total layers: {num_layers}")
    print(f"Total KV heads per layer: {num_kv_heads}")
    print(f"Total possible (layer × head × token): {total_possible}")
    print(f"Actual per-head selections: {total_head_selections}")
    print(f"Overall selection ratio: {overall_ratio*100:.2f}%")

    # Layer 0 的并集（用于构建 sparse 序列的基础）
    layer0_union = layer_union_positions[0]
    print(f"\nLayer 0 union size: {len(layer0_union)} tokens ({len(layer0_union)/total_doc_len*100:.1f}%)")

    # 打印每层的 head 间差异
    print(f"\nPer-layer head diversity (Jaccard similarity between heads):")
    for layer_idx in [0, num_layers//4, num_layers//2, num_layers-1]:
        head_sets = [set(layer_head_selections[layer_idx][h]['positions']) for h in range(num_kv_heads)]
        # 计算平均 Jaccard 相似度
        jaccard_sum = 0
        count = 0
        for i in range(num_kv_heads):
            for j in range(i+1, num_kv_heads):
                intersection = len(head_sets[i] & head_sets[j])
                union = len(head_sets[i] | head_sets[j])
                if union > 0:
                    jaccard_sum += intersection / union
                    count += 1
        avg_jaccard = jaccard_sum / count if count > 0 else 0
        print(f"  Layer {layer_idx:2d}: avg Jaccard similarity = {avg_jaccard:.3f}")

    print(f"\n{'='*100}")
    print("Selection complete")
    print(f"{'='*100}\n")

    return layer_head_selections, layer_union_positions, total_doc_len, position_to_token


def sparse_prefill_per_head(
    model,
    past_key_values,
    layer_head_selections,
    layer_union_positions,
    position_to_token,
    query_tensor,
    prefix_len,
    doc_len,
    device="cuda:0"
):
    """
    实现逐层逐头的 Sparse Prefill（使用 Triton kernel 加速）

    核心思想：
    1. 构建 sparse 序列：critical tokens (Layer 0 union) + query tokens
    2. 逐层处理，每层：
       a) 计算 Q/K/V 投影和 RoPE
       b) 对每个 kv_head，只更新该 head 选中的 positions 的 K/V cache
       c) 使用 Triton kernel 计算逐头不同的 sparse attention
    3. 最终生成时，使用更新后的 cache
    """

    print(f"\n{'='*100}")
    print("Step 2: Sparse Prefill (Per-Head with Triton)")
    print(f"{'='*100}\n")

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_key_value_groups = num_heads // num_kv_heads
    hidden_size = model.config.hidden_size

    # Layer 0 的 union 决定了 sparse 序列中包含哪些 critical tokens
    critical_positions = layer_union_positions[0]  # 相对于 doc 起始的位置
    critical_token_ids = [position_to_token[pos] for pos in critical_positions]

    query_len = query_tensor.shape[1]
    num_critical = len(critical_positions)

    print(f"Critical tokens (Layer 0 union): {num_critical}")
    print(f"Query tokens: {query_len}")
    print(f"Total sparse sequence: {num_critical + query_len}")

    # 构建 sparse 序列的 token tensor
    critical_tensor = torch.tensor([critical_token_ids], dtype=torch.long, device=device)
    sparse_tensor = torch.cat([critical_tensor, query_tensor], dim=1)  # [1, num_critical + query_len]
    sparse_len = sparse_tensor.shape[1]

    # 构建 position ids（保持原始位置信息用于 RoPE）
    critical_abs_positions = [prefix_len + pos for pos in critical_positions]
    query_abs_positions = list(range(prefix_len + doc_len, prefix_len + doc_len + query_len))
    sparse_positions = critical_abs_positions + query_abs_positions
    position_ids = torch.tensor([sparse_positions], dtype=torch.long, device=device)

    print(f"\nPosition mapping:")
    print(f"  Critical positions: {critical_abs_positions[0]} - {critical_abs_positions[-1] if critical_abs_positions else 'N/A'}")
    print(f"  Query positions: {query_abs_positions[0]} - {query_abs_positions[-1]}")

    # Token embedding
    sparse_embeds = model.model.embed_tokens(sparse_tensor)  # [1, sparse_len, hidden_size]
    hidden_states = sparse_embeds

    # ========================================================================
    # 统计每层重算比例
    # ========================================================================
    print(f"\n{'='*100}")
    print("Per-Layer Per-Head Recomputation Statistics")
    print(f"{'='*100}")
    print(f"\n{'Layer':<6} ", end='')
    for h in range(num_kv_heads):
        print(f"Head{h:<6}", end='')
    print(f"{'Union':<8} {'Ratio':<8}")
    print("-" * (6 + 10 * num_kv_heads + 16))

    total_recompute = 0
    layer_recompute_stats = []

    for layer_idx in range(num_layers):
        layer_selections = layer_head_selections[layer_idx]
        head_counts = []
        for kv_head_idx in range(num_kv_heads):
            count = len(layer_selections[kv_head_idx]['positions'])
            head_counts.append(count)
        union_count = len(layer_union_positions[layer_idx])
        layer_ratio = union_count / doc_len if doc_len > 0 else 0

        # 计算该层真正重算的 token 数（使用 union）
        total_recompute += union_count
        layer_recompute_stats.append({
            'head_counts': head_counts,
            'union_count': union_count,
            'ratio': layer_ratio
        })

        if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx:<6} ", end='')
            for h in range(num_kv_heads):
                print(f"{head_counts[h]:<10}", end='')
            print(f"{union_count:<8} {layer_ratio*100:>6.1f}%")

    # 总体重算比例
    total_possible = doc_len * num_layers
    overall_ratio = total_recompute / total_possible if total_possible > 0 else 0

    print("-" * (6 + 10 * num_kv_heads + 16))
    print(f"\nTotal document tokens: {doc_len}")
    print(f"Total layers: {num_layers}")
    print(f"Total possible recomputations: {total_possible}")
    print(f"Actual recomputations (union): {total_recompute}")
    print(f"Overall recomputation ratio: {overall_ratio*100:.2f}%")
    print(f"{'='*100}\n")

    print(f"Processing {num_layers} layers with per-head sparse attention (Triton)...\n")

    with torch.no_grad():
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            # 1. LayerNorm
            normed_hidden = layer.input_layernorm(hidden_states)

            # 2. Q/K/V projections
            q_states = layer.self_attn.q_proj(normed_hidden)
            k_states = layer.self_attn.k_proj(normed_hidden)
            v_states = layer.self_attn.v_proj(normed_hidden)

            # 3. Reshape
            bsz = 1
            q_states = q_states.view(bsz, sparse_len, num_heads, head_dim).transpose(1, 2)
            k_states = k_states.view(bsz, sparse_len, num_kv_heads, head_dim).transpose(1, 2)
            v_states = v_states.view(bsz, sparse_len, num_kv_heads, head_dim).transpose(1, 2)

            # 4. Apply RoPE
            cos, sin = layer.self_attn.rotary_emb(v_states, position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q_states = (q_states * cos) + (rotate_half(q_states) * sin)
            k_states_rope = (k_states * cos) + (rotate_half(k_states) * sin)

            # 5. 以 kv_head 为粒度更新 cache（被选中的 tokens）
            full_cache_len = prefix_len + doc_len
            layer_selections = layer_head_selections[layer_idx]

            for kv_head_idx in range(num_kv_heads):
                head_selected_positions = set(layer_selections[kv_head_idx]['positions'])

                # 遍历 critical positions，只更新被该 head 选中的
                for sparse_idx, (doc_pos, abs_pos) in enumerate(zip(critical_positions, critical_abs_positions)):
                    if doc_pos in head_selected_positions:
                        past_key_values.key_cache[layer_idx][:, kv_head_idx, abs_pos, :] = \
                            k_states_rope[:, kv_head_idx, sparse_idx, :]
                        past_key_values.value_cache[layer_idx][:, kv_head_idx, abs_pos, :] = \
                            v_states[:, kv_head_idx, sparse_idx, :]

            # 6. 使用更新后的 cache 作为 K/V
            # Query 部分的 K/V
            query_k = k_states_rope[:, :, num_critical:, :]
            query_v = v_states[:, :, num_critical:, :]

            # 获取更新后的 cache（每个 kv_head 内容不同）
            past_k = past_key_values.key_cache[layer_idx][:, :, :full_cache_len, :]
            past_v = past_key_values.value_cache[layer_idx][:, :, :full_cache_len, :]

            # 构建完整的 K/V 序列：更新后的 cache + query K/V
            full_k = torch.cat([past_k, query_k], dim=2)  # [1, num_kv_heads, full_cache_len + query_len, head_dim]
            full_v = torch.cat([past_v, query_v], dim=2)
            full_kv_len = full_k.shape[2]

            # 7. 构建 q_idx 用于 sparse attention
            # q_idx[b, h, i] = sparse 序列中第 i 个 token 可以 attend 到的最大位置
            # 对于 critical token: 它原始位置之前的所有 tokens
            # 对于 query token: prefix + doc + 之前的 query tokens
            q_idx = torch.zeros(1, num_heads, sparse_len, dtype=torch.int32, device=device)

            for sparse_idx in range(num_critical):
                # Critical token: 原始位置是 prefix_len + critical_positions[sparse_idx]
                orig_pos = prefix_len + critical_positions[sparse_idx]
                for h in range(num_heads):
                    q_idx[0, h, sparse_idx] = orig_pos

            for q_i in range(query_len):
                # Query token: 原始位置是 prefix_len + doc_len + q_i
                orig_pos = prefix_len + doc_len + q_i
                sparse_idx = num_critical + q_i
                for h in range(num_heads):
                    q_idx[0, h, sparse_idx] = orig_pos

            # 8. 使用 Triton kernel 计算 sparse attention (GQA)
            try:
                from ktransformers.operators.sparse_attention import selected_query_sparse_attention_gqa
                attn_output = selected_query_sparse_attention_gqa(
                    q_states,   # [1, num_heads, sparse_len, head_dim]
                    full_k,     # [1, num_kv_heads, full_kv_len, head_dim]
                    full_v,     # [1, num_kv_heads, full_kv_len, head_dim]
                    q_idx,      # [1, num_heads, sparse_len]
                    block_size_M=64,
                    block_size_N=64
                )
            except Exception as e:
                # Fallback to PyTorch implementation
                if layer_idx == 0:
                    print(f"  Triton kernel failed, using PyTorch fallback: {e}")

                scale = 1.0 / math.sqrt(head_dim)

                # Expand K/V for GQA
                full_k_expanded = full_k.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1, -1)
                full_k_expanded = full_k_expanded.reshape(1, num_heads, full_kv_len, head_dim)
                full_v_expanded = full_v.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1, -1)
                full_v_expanded = full_v_expanded.reshape(1, num_heads, full_kv_len, head_dim)

                # 构建 causal mask
                attn_mask = torch.zeros(sparse_len, full_kv_len, device=device, dtype=hidden_states.dtype)
                for sparse_idx in range(sparse_len):
                    max_pos = q_idx[0, 0, sparse_idx].item()  # 所有 head 相同
                    attn_mask[sparse_idx, :max_pos + 1] = 1.0

                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min

                attn_weights = torch.matmul(q_states, full_k_expanded.transpose(-2, -1)) * scale
                attn_weights = attn_weights + attn_mask
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, full_v_expanded)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, sparse_len, hidden_size)
            attn_output = layer.self_attn.o_proj(attn_output)

            # 8. 将 query 的 K/V 更新到 cache
            for q_idx in range(query_len):
                query_abs_pos = query_abs_positions[q_idx]
                sparse_idx = num_critical + q_idx
                past_key_values.key_cache[layer_idx][:, :, query_abs_pos, :] = \
                    k_states_rope[:, :, sparse_idx, :]
                past_key_values.value_cache[layer_idx][:, :, query_abs_pos, :] = \
                    v_states[:, :, sparse_idx, :]

            # 9. Residual + MLP
            hidden_states = hidden_states + attn_output
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                print(f"  Layer {layer_idx:2d} done")

    # 更新 cache 的 past_tokens 计数
    total_cache_len = prefix_len + doc_len + query_len
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] = total_cache_len

    print(f"\nCache updated: total length = {total_cache_len}")

    print(f"\n{'='*100}")
    print("Sparse Prefill complete")
    print(f"{'='*100}\n")

    # 不再返回 query_hidden_states，因为会在 generate 阶段重新 forward query
    return layer_recompute_stats, overall_ratio


def generate_with_sparse_prefill(
    model,
    tokenizer,
    past_key_values,
    query_tensor,
    prefix_len,
    doc_len,
    max_new_tokens=100,
    device="cuda:0"
):
    """
    在更新好的 KV cache 基础上，用 query tokens 重新做完整 forward pass 并生成

    流程：
    1. 用 query_tensor 在已更新的 KV cache 上做完整 forward pass
    2. 获取第一个 token 的 logits
    3. 进行自回归生成

    Cache 中已包含：
    - Prefix (chunks 0, 1): 完整的 cache
    - Document (chunks 2+): 已通过 sparse prefill 更新（逐层逐头不同）
    """

    print(f"\n{'='*100}")
    print("Step 3: Generating with query re-forward")
    print(f"{'='*100}\n")

    query_len = query_tensor.shape[1]

    print(f"Prefix length: {prefix_len}")
    print(f"Document length: {doc_len}")
    print(f"Query length: {query_len}")

    # 在更新后的 KV cache 上，用 query tokens 重新做完整的 forward pass
    # 这样 query 可以 attend 到完整的 prefix + document cache
    cache_position = torch.arange(prefix_len + doc_len, prefix_len + doc_len + query_len, device=device)

    print(f"Query positions: {prefix_len + doc_len} - {prefix_len + doc_len + query_len - 1}")
    print(f"Re-forwarding query tokens on updated KV cache...\n")

    with torch.no_grad():
        outputs = model(
            input_ids=query_tensor,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True
        )
        logits = outputs.logits[0, -1, :]  # 取最后一个 token 的 logits

    total_cache_len = prefix_len + doc_len + query_len
    current_position = total_cache_len

    print(f"Total cache length after query forward: {total_cache_len}")

    # Generation loop
    # 注意：只追加新生成的 tokens，不包括 query tokens
    generated_ids = []

    print("Generating tokens...\n")

    for step in range(max_new_tokens):
        # Sample next token (greedy)
        next_token_id = torch.argmax(logits, dim=-1)

        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"  EOS reached at step {step}")
            break

        generated_ids.append(next_token_id.item())

        # Print progress
        if step < 10 or step % 10 == 0:
            token_text = tokenizer.decode([next_token_id.item()])
            print(f"  Step {step}: {token_text}")

        # Forward next token using standard model forward
        next_token_embeds = model.model.embed_tokens(next_token_id.unsqueeze(0).unsqueeze(0)).to(device)
        cache_position = torch.tensor([current_position], device=device)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=next_token_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True
            )
            logits = outputs.logits[0, -1, :]

        current_position += 1

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\n{'='*100}")
    print("Generation completed")
    print(f"{'='*100}\n")

    return generated_text


def main(
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    example_idx=4,
    sub_question_idx=1,
    total_ratio=0.3,
    selection_strategy="layer_wise",  # 选择策略: "independent", "union_constrained", "greedy_union", "layer_wise", "nested", "threshold"
    scoring_method="query_attention",  # 评分方法: "query", "reconstruction", 或 "query_attention" (推荐)
    max_new_tokens=100,
    device="cuda:0",
    output_path='./per_head_generation_results.json'
):
    """
    Main function for per-layer generation with recomputation
    """

    print(f"\n{'='*100}")
    print("Per-Layer Token Selection and Recomputation")
    print(f"{'='*100}\n")

    # Load model
    print("Loading model...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model, device_map = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Update device
    if device_map and isinstance(device_map, dict):
        device = f"cuda:{list(device_map.values())[0]}" if 'cuda' not in str(list(device_map.values())[0]) else str(list(device_map.values())[0])
        print(f"Using device: {device}")

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

    # Get data
    # 注意：需要在 doc_chunk_ids 前面加上 chunk_id=0（system prompt cache）
    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    passages = q_data['docs']

    # Build query tensor
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Prepare cache path
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    # Step 1: Compute per-layer per-head token selection
    layer_head_selections, layer_union_positions, doc_len, position_to_token = compute_layerwise_token_selection(
        model, tokenizer, passages, doc_chunk_ids, query_tensor,
        example_idx, full_cache_path, total_ratio, device,
        per_head_selection=True, selection_strategy=selection_strategy,
        scoring_method=scoring_method
    )

    # Step 2: Load FULL cache (prefix + complete document)
    print(f"\n{'='*100}")
    print("Loading FULL cache (prefix + document)")
    print(f"{'='*100}\n")

    max_cache_len = 32768
    cache_device = device_map if device_map else device
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=cache_device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Load all chunks (0, 1, 2+)
    doc_tensors = q_data['doc_tensors']
    prefix_len = 0

    for doc_idx, chunk_id in enumerate(doc_chunk_ids):
        # Load chunk KV cache first to get actual cache length
        chunk_key_cache = torch.load(
            f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt',
            weights_only=True
        )
        chunk_value_cache = torch.load(
            f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt',
            weights_only=True
        )

        # Get actual cache length from the loaded cache (this is the ground truth)
        cache_len = chunk_key_cache[0].shape[2]

        # Copy to past_key_values for each layer
        for layer_idx in range(model.config.num_hidden_layers):
            target_device = past_key_values.key_cache[layer_idx].device
            layer_key = chunk_key_cache[layer_idx].to(target_device)
            layer_value = chunk_value_cache[layer_idx].to(target_device)

            layer_len = layer_key.shape[2]
            current_pos = past_key_values.past_tokens[layer_idx]

            past_key_values.key_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len

        # Chunk 0 is system prompt (prefix)
        if chunk_id == 0:
            prefix_len = cache_len

        print(f"  Loaded chunk {chunk_id}: {cache_len} tokens")

    print(f"\nPrefix length (chunk 0 = system prompt): {prefix_len}")
    print(f"Document length (chunks 2+): {doc_len}")
    print(f"Total cache length: {prefix_len + doc_len}\n")

    # Step 3: Sparse prefill to update KV cache (per-head selection)
    layer_recompute_stats, overall_recompute_ratio = sparse_prefill_per_head(
        model, past_key_values, layer_head_selections, layer_union_positions,
        position_to_token, query_tensor, prefix_len, doc_len, device
    )

    # 重置 past_tokens 为 prefix_len + doc_len（不包含 query，因为 query 会重新 forward）
    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        past_key_values.past_tokens[layer_idx] = prefix_len + doc_len
    # Step 4: Re-forward query on updated cache and generate
    generated_text = generate_with_sparse_prefill(
        model, tokenizer, past_key_values, query_tensor,
        prefix_len, doc_len, max_new_tokens, device
    )

    # Print results
    print(f"\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}\n")
    print(f"Question: {sub_q_info['query']}\n")
    print(f"Ground Truth:\n{sub_q_info['answer']}\n")
    print(f"Generated Answer:\n{generated_text}\n")
    print(f"{'='*100}\n")

    # Compute statistics
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads

    layer_stats = {}
    for layer_idx in range(num_layers):
        head_stats = {}
        for kv_head_idx in range(num_kv_heads):
            head_info = layer_head_selections[layer_idx][kv_head_idx]
            head_stats[kv_head_idx] = {
                'num_tokens': head_info['num_selected'],
                'ratio': head_info['num_selected'] / doc_len if doc_len > 0 else 0
            }
        layer_stats[layer_idx] = {
            'per_head': head_stats,
            'union_size': len(layer_union_positions[layer_idx]),
            'union_ratio': len(layer_union_positions[layer_idx]) / doc_len if doc_len > 0 else 0
        }

    # Save results
    results = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'generated_answer': generated_text,
        'total_ratio': total_ratio,
        'selection_strategy': selection_strategy,
        'overall_recompute_ratio': overall_recompute_ratio,
        'doc_len': doc_len,
        'num_layers': num_layers,
        'num_kv_heads': num_kv_heads,
        'layer_recompute_stats': [
            {
                'head_counts': stats['head_counts'],
                'union_count': stats['union_count'],
                'ratio': stats['ratio']
            }
            for stats in layer_recompute_stats
        ],
        'layer_stats': {
            str(layer_idx): {
                'union_size': info['union_size'],
                'union_ratio': info['union_ratio']
            }
            for layer_idx, info in layer_stats.items()
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")

    return results


def compute_draft_model_attention(
    draft_model,
    tokenizer,
    input_ids,
    device="cuda:0"
):
    """
    使用小模型（draft model）运行完整 prefill，获得完整的 attention 分布

    Args:
        draft_model: 小模型 (e.g., Qwen2.5-3B-Instruct)
        tokenizer: tokenizer
        input_ids: 完整输入的 token ids [1, seq_len]
        device: 计算设备

    Returns:
        layer_attention_scores: {layer_idx: attention_matrix}
            attention_matrix shape: [num_heads, seq_len, seq_len]
    """
    config = draft_model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    seq_len = input_ids.shape[1]

    print(f"\n{'='*100}")
    print("Computing Draft Model Full Prefill Attention")
    print(f"{'='*100}")
    print(f"  Draft model layers: {num_layers}")
    print(f"  Draft model heads: {num_heads} (KV heads: {num_kv_heads})")
    print(f"  Sequence length: {seq_len}")

    # 存储每层的 attention
    layer_attention_scores = {}

    with torch.no_grad():
        # 获取 embeddings
        inputs_embeds = draft_model.model.embed_tokens(input_ids.to(device))
        hidden_states = inputs_embeds

        # 构建 position_ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # 逐层计算
        for layer_idx in range(num_layers):
            layer = draft_model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            # Q, K, V projections
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)

            # Reshape
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            # For GQA models, both query and key have the same head_dim, so apply full cos/sin
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)

            # Expand K, V for GQA
            n_rep = num_heads // num_kv_heads
            key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
            value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)

            # Compute attention weights
            attn_weights = torch.matmul(query_states.float(), key_states_expanded.float().transpose(2, 3)) / (head_dim ** 0.5)

            # Apply causal mask
            causal_mask = torch.triu(torch.ones(q_len, q_len, device=device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)

            # 保存后半部分层的 attention 矩阵（用于动态选层）
            # 只保存后 50% 的层，节省内存
            if layer_idx >= num_layers // 2:
                # Shape: [num_heads, seq_len, seq_len] - 完整 attention 矩阵
                layer_attention_scores[layer_idx] = attn_weights[0].cpu().float().numpy()

            # Continue forward pass
            attn_output = torch.matmul(attn_weights.to(value_states_expanded.dtype), value_states_expanded)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                print(f"  Layer {layer_idx} done")

    print(f"\nDraft model prefill complete")

    return layer_attention_scores


def select_tokens_from_draft_attention(
    draft_attention_scores,
    system_len,
    doc_len,
    query_len,
    target_ratio,
    draft_num_layers,
    target_num_layers
):
    """
    根据 draft model 的 attention 分布选择重要 tokens

    使用 query 对 document 的 attention 来选择重要位置

    Args:
        draft_attention_scores: {layer_idx: attention_matrix [num_heads, seq_len]}
        system_len: system prompt 长度
        doc_len: 文档长度
        query_len: query 长度
        target_ratio: 目标选择比例
        draft_num_layers: draft model 层数
        target_num_layers: target model 层数

    Returns:
        selected_positions: 选中的文档位置列表 (相对于文档起始的位置)
    """
    print(f"\n{'='*100}")
    print("Selecting Tokens from Draft Model Attention (Entropy-based Layer Selection)")
    print(f"{'='*100}")
    print(f"  System len: {system_len}, Doc len: {doc_len}, Query len: {query_len}")
    print(f"  Target ratio: {target_ratio} ({int(doc_len * target_ratio)} tokens)")

    # 位置信息
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    # =========================================================================
    # Step 1: 计算每层的 query→doc attention 和对应的熵
    # =========================================================================
    layer_entropy = {}
    layer_attention = {}

    for layer_idx in sorted(draft_attention_scores.keys()):
        layer_attn = draft_attention_scores[layer_idx]  # [num_heads, seq_len, seq_len]

        # 提取 query→doc attention
        query_to_doc = layer_attn[:, query_start:total_len, doc_start:doc_end]  # [num_heads, query_len, doc_len]

        # 对所有 heads 和 query positions 平均，得到每个 doc position 的 attention
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
        layer_attention[layer_idx] = doc_attention_avg

        # 计算熵: H = -sum(p * log(p))
        # 归一化 attention 分布
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        # 避免 log(0)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropy[layer_idx] = entropy

    # =========================================================================
    # Step 2: 选择熵最低的 top-k 层（attention 更集中的层）
    # =========================================================================
    top_k = 4  # 选择 4 层
    sorted_layers = sorted(layer_entropy.items(), key=lambda x: x[1])
    layers_to_use = [layer_idx for layer_idx, _ in sorted_layers[:top_k]]

    print(f"\n  Layer entropy analysis:")
    for layer_idx, entropy in sorted_layers:
        marker = " <-- selected" if layer_idx in layers_to_use else ""
        print(f"    Layer {layer_idx}: entropy={entropy:.4f}{marker}")

    print(f"\n  Selected layers (lowest entropy): {layers_to_use}")

    # =========================================================================
    # Step 3: 聚合选中层的 attention
    # =========================================================================
    query_to_doc_attention = [layer_attention[layer_idx] for layer_idx in layers_to_use]
    multi_layer_attn = np.stack(query_to_doc_attention).mean(axis=0)  # [doc_len]

    print(f"  Attention stats: mean={multi_layer_attn.mean():.6f}, std={multi_layer_attn.std():.6f}, "
          f"max={multi_layer_attn.max():.6f}")

    # 使用 smart selection 算法
    target_count = int(doc_len * target_ratio)

    # Step 1: 找到高 attention 位置 (> mean + 0.5 * std)
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    threshold = mean_attn + 0.5 * std_attn

    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])
    print(f"  High attention positions (>mean+0.5*std): {len(high_attn_positions)}")

    # Step 2: 连通分量分析
    components = find_connected_components(high_attn_positions, max_gap=2)
    print(f"  Connected components: {len(components)}")

    # Step 3: 计算每个分量的总 attention
    component_scores = []
    for comp in components:
        total_score = sum(multi_layer_attn[p] for p in comp)
        component_scores.append((comp, total_score))

    # Step 4: 按总 attention 排序
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 5: 贪心选择分量 + 上下文扩展 (±1)
    selected = set()

    for comp, total_score in component_scores:
        extended_comp = set()
        for p in comp:
            for offset in range(-1, 2):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)

        new_positions = extended_comp - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended_comp)

    print(f"  After component selection: {len(selected)}")

    # Step 6: 补充到目标数量
    if len(selected) < target_count:
        sorted_indices = np.argsort(multi_layer_attn)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(int(pos))
                if len(selected) >= target_count:
                    break

    # Step 7: 如果超过目标，移除最低分的位置
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: multi_layer_attn[p])
        selected.remove(min_pos)

    selected_list = sorted(list(selected))

    print(f"  Final selected: {len(selected_list)} positions ({len(selected_list)/doc_len*100:.1f}%)")

    return selected_list, multi_layer_attn


def analyze_draft_model_selection(
    draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
    data_path='./result_reflect.json',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    example_idx=4,
    sub_question_idx=1,
    total_ratio=0.3,
    device="cuda:0"
):
    """
    分析 DraftModel 方法选择了哪些 tokens

    输出:
    1. 选中的 token 位置和对应的文本
    2. 按文档分组展示选中的片段
    3. 可视化 attention 分布和选择结果
    """
    from transformers import AutoConfig

    print(f"\n{'='*100}")
    print("DraftModel Token Selection Analysis")
    print(f"{'='*100}")
    print(f"  Draft model: {draft_model_path}")
    print(f"  Target ratio: {total_ratio}")
    print(f"{'='*100}\n")

    # Load draft model
    print("[Step 1] Loading draft model...")
    draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
    draft_model, _ = load_model('qwen', draft_model_path, draft_config, device, use_multi_gpu=False)
    draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(draft_model_path, trust_remote_code=True)

    # Load data
    print("\n[Step 2] Loading data...")
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

    # Build full input
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    docs = q_data['docs']  # 原始文本
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]
    sub_q_docs = [docs[chunk_id - 1] for chunk_id in doc_chunk_ids]

    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

    system_len = system_tensor.shape[0]
    doc_len = sum(t.shape[0] for t in sub_q_doc_tensors)
    query_len = question_tensor.shape[0]

    print(f"[Step 3] Input structure:")
    print(f"  System prompt: {system_len} tokens")
    print(f"  Documents: {doc_len} tokens ({len(sub_q_doc_tensors)} docs)")
    print(f"  Query: {query_len} tokens")
    print(f"  Total: {full_input.shape[1]} tokens\n")

    # Compute draft model attention
    print("[Step 4] Computing draft model attention...")
    draft_attention = compute_draft_model_attention(
        draft_model, tokenizer, full_input, device
    )

    # Select tokens
    print("\n[Step 5] Selecting tokens...")
    selected_positions, attention_scores = select_tokens_from_draft_attention(
        draft_attention,
        system_len=system_len,
        doc_len=doc_len,
        query_len=query_len,
        target_ratio=total_ratio,
        draft_num_layers=draft_model.config.num_hidden_layers,
        target_num_layers=28
    )

    # =========================================================================
    # 分析选中的 tokens
    # =========================================================================
    print(f"\n{'='*100}")
    print("Selected Tokens Analysis")
    print(f"{'='*100}\n")

    # 构建文档 token 到文本的映射
    doc_tokens_flat = torch.cat(sub_q_doc_tensors).tolist()

    # 统计每个文档被选中的 token 数
    doc_boundaries = []
    current_pos = 0
    for i, t in enumerate(sub_q_doc_tensors):
        doc_boundaries.append((current_pos, current_pos + t.shape[0], i))
        current_pos += t.shape[0]

    print(f"Total selected: {len(selected_positions)} / {doc_len} tokens ({len(selected_positions)/doc_len*100:.1f}%)\n")

    # 按文档分组展示
    for doc_start, doc_end, doc_idx in doc_boundaries:
        doc_selected = [p for p in selected_positions if doc_start <= p < doc_end]
        doc_total = doc_end - doc_start

        print(f"\n{'─'*80}")
        print(f"Document {doc_idx + 1} (chunk_id={doc_chunk_ids[doc_idx]}): {len(doc_selected)}/{doc_total} tokens selected ({len(doc_selected)/doc_total*100:.1f}%)")
        print(f"{'─'*80}")

        if len(doc_selected) == 0:
            print("  (No tokens selected)")
            continue

        # 找出连续的片段
        doc_selected_local = [p - doc_start for p in doc_selected]
        segments = []
        current_segment = [doc_selected_local[0]]

        for pos in doc_selected_local[1:]:
            if pos == current_segment[-1] + 1:
                current_segment.append(pos)
            else:
                segments.append(current_segment)
                current_segment = [pos]
        segments.append(current_segment)

        print(f"  Selected segments: {len(segments)}")

        # 获取文档的 tokens
        doc_tokens = sub_q_doc_tensors[doc_idx].tolist()

        # 展示每个片段
        for seg_idx, segment in enumerate(segments[:10]):  # 最多展示前 10 个片段
            start_pos = segment[0]
            end_pos = segment[-1]

            # 获取片段文本（带上下文）
            context_start = max(0, start_pos - 3)
            context_end = min(len(doc_tokens), end_pos + 4)

            context_tokens = doc_tokens[context_start:context_end]
            context_text = tokenizer.decode(context_tokens)

            # 标记选中的部分
            selected_tokens = doc_tokens[start_pos:end_pos + 1]
            selected_text = tokenizer.decode(selected_tokens)

            # 计算片段的平均 attention
            seg_attention = [attention_scores[p] for p in segment]
            avg_attention = np.mean(seg_attention)

            print(f"\n  Segment {seg_idx + 1}: positions {start_pos}-{end_pos} (len={len(segment)}, avg_attn={avg_attention:.6f})")
            print(f"    Context: ...{context_text}...")
            print(f"    Selected: 【{selected_text}】")

        if len(segments) > 10:
            print(f"\n  ... and {len(segments) - 10} more segments")

    # =========================================================================
    # 可视化 attention 分布
    # =========================================================================
    print(f"\n{'='*100}")
    print("Saving attention visualization...")
    print(f"{'='*100}\n")

    output_dir = './attention_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制 attention 分布图
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 上图：完整 attention 分布
    ax1 = axes[0]
    ax1.bar(range(doc_len), attention_scores, alpha=0.7, width=1.0)
    ax1.axhline(y=np.mean(attention_scores), color='r', linestyle='--', label=f'Mean={np.mean(attention_scores):.6f}')
    ax1.axhline(y=np.mean(attention_scores) + 0.5*np.std(attention_scores), color='g', linestyle='--',
                label=f'Threshold (μ+0.5σ)={np.mean(attention_scores) + 0.5*np.std(attention_scores):.6f}')
    ax1.set_xlabel('Document Position')
    ax1.set_ylabel('Attention Score')
    ax1.set_title(f'DraftModel Attention Distribution (ratio={total_ratio})')
    ax1.legend()

    # 标记文档边界
    for doc_start, doc_end, doc_idx in doc_boundaries:
        ax1.axvline(x=doc_start, color='gray', linestyle=':', alpha=0.5)

    # 下图：选中的位置
    ax2 = axes[1]
    selection_mask = np.zeros(doc_len)
    for p in selected_positions:
        selection_mask[p] = 1
    ax2.bar(range(doc_len), selection_mask, alpha=0.7, width=1.0, color='green')
    ax2.set_xlabel('Document Position')
    ax2.set_ylabel('Selected (1/0)')
    ax2.set_title(f'Selected Tokens ({len(selected_positions)}/{doc_len} = {len(selected_positions)/doc_len*100:.1f}%)')

    # 标记文档边界
    for doc_start, doc_end, doc_idx in doc_boundaries:
        ax2.axvline(x=doc_start, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()

    output_file = f'{output_dir}/draft_model_selection_example{example_idx}_sub{sub_question_idx}_ratio{total_ratio}.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_file}")

    # 清理 draft model
    del draft_model
    torch.cuda.empty_cache()

    return {
        'selected_positions': selected_positions,
        'attention_scores': attention_scores.tolist(),
        'doc_len': doc_len,
        'num_docs': len(sub_q_doc_tensors),
        'question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer']
    }


def main_with_draft_model(
    target_model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    example_idx=4,
    sub_question_idx=1,
    total_ratio=0.3,
    max_new_tokens=100,
    device="cuda:0",
    output_path='./draft_guided_generation_results.json'
):
    """
    使用小模型 (draft model) 指导大模型生成的方法

    流程:
    1. 加载小模型，运行完整 prefill 获得 attention 分布
    2. 分析 attention 分布，选择重要 tokens
    3. 加载大模型的 KV cache
    4. 只对选中的 tokens 重算，然后生成
    """

    print(f"\n{'='*100}")
    print("Draft Model Guided Generation")
    print(f"{'='*100}")
    print(f"  Target model: {target_model_path}")
    print(f"  Draft model: {draft_model_path}")
    print(f"  Target ratio: {total_ratio}")
    print(f"{'='*100}\n")

    from transformers import AutoConfig

    # =========================================================================
    # Step 1: Load draft model and compute attention
    # =========================================================================
    print("\n[Step 1] Loading draft model...")

    draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
    draft_model, draft_device_map = load_model('qwen', draft_model_path, draft_config, device, use_multi_gpu=False)
    draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(draft_model_path, trust_remote_code=True)

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

    # Build full input for draft model
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    # Build question tensor
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    # Concatenate all tokens for draft model prefill
    all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

    system_len = system_tensor.shape[0]
    doc_len = sum(t.shape[0] for t in sub_q_doc_tensors)
    query_len = question_tensor.shape[0]

    print(f"Input lengths: system={system_len}, doc={doc_len}, query={query_len}, total={full_input.shape[1]}")

    # Compute draft model attention
    draft_attention = compute_draft_model_attention(
        draft_model, tokenizer, full_input, device
    )

    # Select tokens based on draft attention
    selected_positions, attention_scores = select_tokens_from_draft_attention(
        draft_attention,
        system_len=system_len,
        doc_len=doc_len,
        query_len=query_len,
        target_ratio=total_ratio,
        draft_num_layers=draft_model.config.num_hidden_layers,
        target_num_layers=28  # Qwen2.5-7B has 28 layers
    )

    # Free draft model memory
    del draft_model
    if "cuda" in device:
        torch.cuda.empty_cache()
    print("\nDraft model unloaded, memory freed")

    # =========================================================================
    # Step 2: Load target model and generate
    # =========================================================================
    print(f"\n{'='*100}")
    print("[Step 2] Loading target model...")
    print(f"{'='*100}")

    target_config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    target_model, target_device_map = load_model('qwen', target_model_path, target_config, device, use_multi_gpu=False)
    target_model.eval()

    # Prepare cache
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    max_cache_len = 32768
    cache_device = target_device_map if target_device_map else device
    past_key_values = StaticCache(
        config=target_model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=cache_device,
        dtype=target_model.dtype,
        passage_len=32768
    )

    # Load all KV cache chunks
    print("\nLoading KV cache...")
    kv_chunk_ids = [0] + doc_chunk_ids
    prefix_len = 0

    for chunk_id in kv_chunk_ids:
        chunk_key_cache = torch.load(
            f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt',
            weights_only=True
        )
        chunk_value_cache = torch.load(
            f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt',
            weights_only=True
        )

        cache_len = chunk_key_cache[0].shape[2]

        for layer_idx in range(target_model.config.num_hidden_layers):
            target_device = past_key_values.key_cache[layer_idx].device
            layer_key = chunk_key_cache[layer_idx].to(target_device)
            layer_value = chunk_value_cache[layer_idx].to(target_device)

            current_pos = past_key_values.past_tokens[layer_idx]
            past_key_values.key_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += cache_len

        if chunk_id == 0:
            prefix_len = cache_len

        print(f"  Loaded chunk {chunk_id}: {cache_len} tokens")

    total_cache_len = prefix_len + doc_len
    print(f"\nTotal cache: prefix={prefix_len}, doc={doc_len}, total={total_cache_len}")

    # =========================================================================
    # Step 3: Recompute selected tokens and generate
    # =========================================================================
    print(f"\n{'='*100}")
    print("[Step 3] Recomputing selected tokens and generating")
    print(f"{'='*100}")

    # Convert selected positions to global indices (add prefix_len)
    k_need_index = [p + prefix_len for p in selected_positions]
    # Add query positions
    query_positions = list(range(total_cache_len, total_cache_len + query_len))
    k_need_index.extend(query_positions)

    print(f"  Selected doc tokens: {len(selected_positions)} ({len(selected_positions)/doc_len*100:.1f}%)")
    print(f"  Query tokens: {query_len}")
    print(f"  Total recompute: {len(k_need_index)}")

    # Build recompute input
    recompute_tokens = []
    for pos in selected_positions:
        # Find which document this position belongs to
        cumsum = 0
        for doc_tensor in sub_q_doc_tensors:
            if cumsum + doc_tensor.shape[0] > pos:
                local_pos = pos - cumsum
                recompute_tokens.append(doc_tensor[local_pos].item())
                break
            cumsum += doc_tensor.shape[0]

    # Add query tokens
    recompute_tokens.extend(question_tokens)

    recompute_input = torch.tensor(recompute_tokens, dtype=torch.long).unsqueeze(0).to(device)
    cache_position = torch.tensor(k_need_index, device=device)

    print(f"  Recompute input shape: {recompute_input.shape}")

    # Recompute
    with torch.no_grad():
        inputs_embeds = target_model.model.embed_tokens(recompute_input)

        outputs = target_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True
        )

        logits = outputs.logits[0, -1, :]

    # =========================================================================
    # Step 4: Generate
    # =========================================================================
    print(f"\n{'='*100}")
    print("[Step 4] Generating...")
    print(f"{'='*100}")

    generated_ids = []
    current_position = total_cache_len + query_len

    for step in range(max_new_tokens):
        next_token_id = torch.argmax(logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"  EOS at step {step}")
            break

        # Check for <|im_end|>
        if tokenizer.decode([next_token_id.item()]) == '<|im_end|>':
            print(f"  <|im_end|> at step {step}")
            break

        generated_ids.append(next_token_id.item())

        # Forward next token
        next_token_embeds = target_model.model.embed_tokens(next_token_id.unsqueeze(0).unsqueeze(0))
        cache_position = torch.tensor([current_position], device=device)

        with torch.no_grad():
            outputs = target_model(
                inputs_embeds=next_token_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True
            )
            logits = outputs.logits[0, -1, :]

        current_position += 1

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # =========================================================================
    # Results
    # =========================================================================
    print(f"\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}")
    print(f"Question: {sub_q_info['query']}\n")
    print(f"Ground Truth:\n{sub_q_info['answer']}\n")
    print(f"Generated Answer:\n{generated_text}\n")
    print(f"{'='*100}")

    # Check if correct
    if '1216' in generated_text and '1220' in generated_text:
        print("\n✓ SUCCESS: Answer contains both '1216' and '1220'!")
    else:
        print("\n✗ FAILURE: Answer does NOT contain both '1216' and '1220'")
        if '1216' in generated_text:
            print("  - '1216' found")
        else:
            print("  - '1216' NOT found")
        if '1220' in generated_text:
            print("  - '1220' found")
        else:
            print("  - '1220' NOT found")

    # Save results
    results = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'generated_answer': generated_text,
        'total_ratio': total_ratio,
        'method': 'draft_model_guided',
        'draft_model': draft_model_path,
        'target_model': target_model_path,
        'doc_len': doc_len,
        'selected_tokens': len(selected_positions),
        'selection_ratio': len(selected_positions) / doc_len
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    import os
    # 如果环境变量未设置，默认使用 GPU 0
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 分析 DraftModel 选择了哪些 tokens
    results = analyze_draft_model_selection(
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        data_path='./result_reflect.json',
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        example_idx=4,
        sub_question_idx=1,
        total_ratio=0.3,
        device='cuda:0'
    )

    # # 使用小模型指导生成（注释掉）
    # results = main_with_draft_model(
    #     target_model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    #     draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
    #     data_path='./result_reflect.json',
    #     cache_path='/mnt/data/reflect/',
    #     model_name='Qwen2.5-7B-Instruct',
    #     bge_model_path='/mnt/data/models/bge-m3-FP16',
    #     example_idx=4,
    #     sub_question_idx=1,
    #     total_ratio=0.3,
    #     max_new_tokens=100,
    #     device='cuda:0',
    #     output_path='./draft_guided_generation_results.json'
    # )

    # # 原方法（注释掉）
    # # 选择策略: "independent", "union_constrained", "greedy_union", "layer_wise", "nested", "threshold"
    # strategy = "layer_wise"  # 使用分层选择策略（同层各 head 相同）
    # ratio = 0.3
    #
    # # 评分方法: "query", "reconstruction", 或 "query_attention" (推荐)
    # # - query: 使用用户问题的 attention
    # # - reconstruction: KVzip 风格的文本重构评分
    # # - query_attention: Smart Query Selection (query attention + 连通分量分析) - 推荐！
    # scoring = "query_attention"  # 使用 Smart Query Selection
    #
    # results = main(
    #     model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    #     data_path='./result_reflect.json',
    #     cache_path='/mnt/data/reflect/',
    #     model_name='Qwen2.5-7B-Instruct',
    #     bge_model_path='/mnt/data/models/bge-m3-FP16',
    #     example_idx=4,
    #     sub_question_idx=1,
    #     total_ratio=ratio,
    #     selection_strategy=strategy,
    #     scoring_method=scoring,
    #     max_new_tokens=100,
    #     device='cuda:0',
    #     output_path=f'./per_head_generation_results_{strategy}_{scoring}_{ratio}.json'
    # )

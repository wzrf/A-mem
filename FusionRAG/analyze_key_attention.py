#!/usr/bin/env python3
"""
分析关键 token 的 attention 分数，手动计算 attention weights
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

from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def analyze_attention_for_key_tokens():
    device = "cuda:0"
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    cache_path = '/mnt/data/reflect/'
    model_name = 'Qwen2.5-7B-Instruct'
    example_idx = 4
    sub_question_idx = 1
    
    print("=" * 100)
    print("Attention Analysis for Key Tokens")
    print("=" * 100)
    
    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, _ = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 模型配置
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    
    print(f"\nModel config: {num_layers} layers, {num_heads} heads, {num_kv_heads} KV heads, {head_dim} head_dim")
    
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
    
    # Build position_to_token mapping
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')
    
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
    
    # 找到关键 token 的位置
    positions = sorted(position_to_token.keys())
    tokens_list = [position_to_token[p] for p in positions]
    full_text = tokenizer.decode(tokens_list)
    
    # 构建 token 到文本位置的映射
    token_to_text_pos = {}
    current_text = ""
    for pos in positions:
        token_id = position_to_token[pos]
        token_text = tokenizer.decode([token_id])
        start_pos = len(current_text)
        current_text += token_text
        token_to_text_pos[pos] = (start_pos, len(current_text), token_text)
    
    # 关键短语
    key_phrases = {
        "1216": "critical",
        "1220": "critical",
        "17 May": "important",
        "crowned": "important",
        "coronation": "moderate",
        "Gloucester": "important",
        "Westminster": "important",
        "Henry III": "moderate",
    }
    
    # 找到关键 token
    key_token_positions = {}
    for phrase, importance in key_phrases.items():
        key_token_positions[phrase] = {"importance": importance, "positions": []}
        idx = 0
        while True:
            idx = full_text.find(phrase, idx)
            if idx == -1:
                break
            end_idx = idx + len(phrase)
            
            for pos in positions:
                start, end, txt = token_to_text_pos[pos]
                if start < end_idx and end > idx:
                    key_token_positions[phrase]["positions"].append(pos)
            idx += 1
    
    print("\n" + "=" * 80)
    print("KEY TOKEN POSITIONS")
    print("=" * 80)
    for phrase, info in key_token_positions.items():
        pos_list = sorted(set(info["positions"]))
        print(f"  '{phrase}' ({info['importance']}): positions {pos_list[:10]}{'...' if len(pos_list) > 10 else ''}")
    
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
            current_pos = past_key_values.past_tokens[layer_idx]
            
            past_key_values.key_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_key)
            past_key_values.value_cache[layer_idx].narrow(2, current_pos, layer_len).copy_(layer_value)
            past_key_values.past_tokens[layer_idx] += layer_len
    
    total_cache_len = past_key_values.past_tokens[0]
    print(f"Total cache length: {total_cache_len}")
    
    # 构造查询 tokens
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)
    query_len = query_tensor.shape[1]
    
    print(f"\nQuery: {question_text[:100]}...")
    print(f"Query tokens: {query_len}")
    
    # 手动计算 query 的 attention weights
    print("\n" + "=" * 80)
    print("Computing Query Attention Weights Manually")
    print("=" * 80)
    
    # 获取 query 的 hidden states
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(query_tensor)
        
        layer_attention_scores = {}
        hidden_states = inputs_embeds
        
        # 只处理几个关键层
        key_layers = [0, 4, 8, 12, 16, 20, 24, 27]
        
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            
            # 获取 residual
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # 获取 Q, K, V projections
            bsz, q_len, _ = hidden_states.size()
            
            query_states = layer.self_attn.q_proj(hidden_states)
            # 使用 cache 中的 key
            key_states = past_key_values.key_cache[layer_idx][:, :, :total_cache_len, :]
            
            # Reshape
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            # key_states shape: (1, num_kv_heads, kv_len, head_dim)
            
            # GQA: repeat KV heads
            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            
            # Compute attention scores
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / (head_dim ** 0.5)
            # attn_weights: (1, num_heads, q_len, kv_len)
            
            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            if layer_idx in key_layers:
                # 保存文档部分的 attention
                doc_attn = attn_weights[0, :, :, prefix_len:prefix_len + doc_len]  # (num_heads, q_len, doc_len)
                # 对 query tokens 和 heads 平均
                doc_attn_avg = doc_attn.mean(dim=(0, 1))  # (doc_len,)
                layer_attention_scores[layer_idx] = {
                    "doc_attn": doc_attn_avg.cpu().float().numpy(),
                    "per_head_attn": doc_attn.mean(dim=1).cpu().float().numpy()  # (num_heads, doc_len)
                }
            
            # 继续前向传播
            value_states = past_key_values.value_cache[layer_idx][:, :, :total_cache_len, :]
            value_states = value_states.repeat_interleave(n_rep, dim=1)
            
            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)
            
            hidden_states = residual + attn_output
            
            # MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
    
    print("\nAttention computed for layers:", list(layer_attention_scores.keys()))
    
    # 分析关键 token 的 attention 分数
    print("\n" + "=" * 80)
    print("ATTENTION SCORES FOR KEY TOKENS (Query-based)")
    print("=" * 80)
    
    for phrase, info in key_phrases.items():
        if info == "critical" or info == "important":
            pos_list = sorted(set(key_token_positions[phrase]["positions"]))
            if len(pos_list) == 0:
                continue
            
            print(f"\n  '{phrase}' ({info}):")
            
            for layer_idx in key_layers:
                doc_attn = layer_attention_scores[layer_idx]["doc_attn"]
                
                key_attns = [doc_attn[pos] for pos in pos_list if pos < len(doc_attn)]
                if key_attns:
                    mean_attn = np.mean(key_attns)
                    max_attn = np.max(key_attns)
                    percentile = (doc_attn < mean_attn).sum() / len(doc_attn) * 100
                    
                    print(f"    Layer {layer_idx:2d}: mean={mean_attn:.6f}, max={max_attn:.6f}, rank={percentile:.1f}%")
    
    # 加载 reconstruction scores 对比
    print("\n" + "=" * 80)
    print("COMPARISON: Query Attention vs Reconstruction Scoring")
    print("=" * 80)
    
    score_path = os.path.join(full_cache_path, f'{example_idx}_scores.pt')
    recon_scores = None
    if os.path.exists(score_path):
        recon_scores = torch.load(score_path, map_location='cpu', weights_only=True)
        print(f"\nReconstruction scores shape: {recon_scores.shape}")
        
        if len(recon_scores.shape) == 2:
            recon_avg = recon_scores.mean(dim=0).numpy()
        else:
            recon_avg = recon_scores.numpy()
        
        print("\nReconstruction scores for key tokens:")
        for phrase, info in key_phrases.items():
            if info == "critical":
                pos_list = sorted(set(key_token_positions[phrase]["positions"]))
                if len(pos_list) == 0:
                    continue
                
                key_scores = [recon_avg[pos] for pos in pos_list if pos < len(recon_avg)]
                if key_scores:
                    mean_score = np.mean(key_scores)
                    max_score = np.max(key_scores)
                    percentile = (recon_avg < mean_score).sum() / len(recon_avg) * 100
                    print(f"  '{phrase}': mean={mean_score:.4f}, max={max_score:.4f}, rank={percentile:.1f}%")
    else:
        print("Reconstruction scores not found")
    
    # 设计新的选择策略
    print("\n" + "=" * 80)
    print("PROPOSED SELECTION STRATEGIES (30% target)")
    print("=" * 80)
    
    target_ratio = 0.3
    target_count = int(doc_len * target_ratio)
    
    # 策略1: Query-based attention (last layer)
    print("\n1. Query Attention (Layer 27):")
    layer27_attn = layer_attention_scores[27]["doc_attn"]
    top_k_query = np.argsort(layer27_attn)[::-1][:target_count]
    
    for phrase, info in key_phrases.items():
        if info == "critical":
            pos_list = sorted(set(key_token_positions[phrase]["positions"]))
            selected_count = sum(1 for p in pos_list if p in top_k_query)
            print(f"    '{phrase}': {selected_count}/{len(pos_list)} selected")
    
    # 策略2: Query-based attention (多层聚合)
    print("\n2. Query Attention (Layer 20-27 avg):")
    multi_layer_attn = np.stack([
        layer_attention_scores[i]["doc_attn"] for i in [20, 24, 27]
    ]).mean(axis=0)
    top_k_multi = np.argsort(multi_layer_attn)[::-1][:target_count]
    
    for phrase, info in key_phrases.items():
        if info == "critical":
            pos_list = sorted(set(key_token_positions[phrase]["positions"]))
            selected_count = sum(1 for p in pos_list if p in top_k_multi)
            print(f"    '{phrase}': {selected_count}/{len(pos_list)} selected")
    
    # 策略3: 混合策略
    if recon_scores is not None:
        print("\n3. Hybrid (0.5 * Recon + 0.5 * QueryAttn):")
        # 归一化
        recon_norm = (recon_avg - recon_avg.min()) / (recon_avg.max() - recon_avg.min() + 1e-8)
        query_norm = (multi_layer_attn - multi_layer_attn.min()) / (multi_layer_attn.max() - multi_layer_attn.min() + 1e-8)
        
        hybrid_score = 0.5 * recon_norm + 0.5 * query_norm
        top_k_hybrid = np.argsort(hybrid_score)[::-1][:target_count]
        
        for phrase, info in key_phrases.items():
            if info == "critical":
                pos_list = sorted(set(key_token_positions[phrase]["positions"]))
                selected_count = sum(1 for p in pos_list if p in top_k_hybrid)
                print(f"    '{phrase}': {selected_count}/{len(pos_list)} selected")
    
    # 策略4: 对比原始 reconstruction scoring
    if recon_scores is not None:
        print("\n4. Pure Reconstruction Scoring (for comparison):")
        top_k_recon = np.argsort(recon_avg)[::-1][:target_count]
        
        for phrase, info in key_phrases.items():
            if info == "critical":
                pos_list = sorted(set(key_token_positions[phrase]["positions"]))
                selected_count = sum(1 for p in pos_list if p in top_k_recon)
                print(f"    '{phrase}': {selected_count}/{len(pos_list)} selected")
    
    # 保存最佳策略选择的 tokens
    print("\n" + "=" * 80)
    print("SAVING BEST SELECTION FOR TESTING")
    print("=" * 80)
    
    # 使用 query attention 策略的选择
    best_selection = {
        "strategy": "query_attention_multi_layer",
        "target_ratio": target_ratio,
        "selected_positions": [int(p) for p in top_k_multi],
        "key_token_coverage": {}
    }
    
    for phrase, info in key_phrases.items():
        pos_list = sorted(set(key_token_positions[phrase]["positions"]))
        selected_count = sum(1 for p in pos_list if p in top_k_multi)
        best_selection["key_token_coverage"][phrase] = {
            "total": len(pos_list),
            "selected": selected_count
        }
    
    os.makedirs("./attention_analysis", exist_ok=True)
    with open("./attention_analysis/query_attention_selection.json", "w") as f:
        json.dump(best_selection, f, indent=2)
    
    print(f"\nSaved selection to ./attention_analysis/query_attention_selection.json")
    print(f"Selected {len(top_k_multi)} tokens ({len(top_k_multi)/doc_len*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)
    
    print("""
发现:
1. Reconstruction Scoring 的问题:
   - 它是 query-agnostic 的，只考虑 token 对上下文重建的贡献
   - 关键年份 (1216, 1220) 在重建任务中可能不那么重要
   
2. Query Attention 的优势:
   - 它是 query-aware 的，知道问题在问什么
   - 能够更准确地定位与问题相关的 tokens
   
3. 建议的改进:
   - 使用 Query Attention 作为主要选择依据
   - 或者使用 Hybrid 策略 (Reconstruction + Query Attention)
""")

    print("=" * 80)


if __name__ == "__main__":
    analyze_attention_for_key_tokens()

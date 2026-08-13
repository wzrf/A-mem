import torch
import time

from numpy import ndarray

from ktransformers.util.utils import load_kv, revert_key_cache, analyze_print_and_return_max_mse_map
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compare_kv_cache_tokens(kv_tensor: torch.Tensor) -> dict:
    """
    高效量化 KV Cache 中第三维度（Token 维度）上任意两个 Token 之间的 KV 差异。

    参数:
        kv_tensor: torch.Tensor, 形状必须为 [1, 2, x, 128]
                  其中 dim=1 的 0 位代表 Key, 1 位代表 Value

    返回:
        dict: 包含 K 相似度矩阵和 V 差异性矩阵的字典，形状均为 [x, x]
    """
    assert kv_tensor.shape[0] == 1, "Batch size 必须为 1"
    assert kv_tensor.shape[1] == 2, "第二维度必须为 2 (分别代表 K 和 V)"

    # 1. 剥离 Batch 维度，并分离 K 和 V -> 形状变为 [x, 128]
    K = kv_tensor[0, 0, :, :].float()  # 强转 float 避免半精度溢出
    V = kv_tensor[0, 1, :, :].float()

    x = K.shape[0]

    # ----------------------------------------------------
    # 2. 高效对比 K: 利用广播并行计算所有 Token 两两之间的余弦相似度
    # ----------------------------------------------------
    # K 的形状: [x, 128] -> 扩展为 [x, 1, 128] 和 [1, x, 128]
    K_expanded1 = K.unsqueeze(1)
    K_expanded2 = K.unsqueeze(0)

    # 利用 PyTorch 内置的余弦相似度函数，在最后一个维度上规约
    # 结果 k_cosine_matrix 的形状为 [x, x]
    # 矩阵中 (i, j) 位置的值代表第 i 个 Token 和第 j 个 Token 的 K 向量夹角余弦值
    k_cosine_matrix = torch.cosine_similarity(K_expanded1, K_expanded2, dim=-1)

    # ----------------------------------------------------
    # 3. 高效对比 V: 利用广播并行计算所有 Token 两两之间的相对范数差异
    # ----------------------------------------------------
    # 计算 V1 - V2 的差值矩阵，形状为 [x, x, 128]
    V_diff = V.unsqueeze(1) - V.unsqueeze(0)

    # 计算元素级的平方和，并在最后一个维度规约（相当于计算 Frobenius 范数）
    # v_diff_norm 形状为 [x, x]
    v_diff_norm = torch.norm(V_diff, p='fro', dim=-1)

    # 计算分母：||V_i|| + ||V_j|| 用于归一化，规避绝对数值大小的影响
    v_norms = torch.norm(V, p='fro', dim=-1)  # [x]
    v_norms_matrix = v_norms.unsqueeze(1) + v_norms.unsqueeze(0)  # 广播得到 [x, x]

    # 相对差异矩阵 = ||V_i - V_j|| / (||V_i|| + ||V_j|| + epsilon)
    v_diff_matrix = v_diff_norm / (v_norms_matrix + 1e-8)

    return {
        "k_cosine_similarity": k_cosine_matrix,  # 值域 [-1, 1]，越接近 1 越相似
        "v_relative_difference": v_diff_matrix  # 值域 [0, 1]，越接近 0 越相似
    }

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def compute_gqa_attention_inplace(query: torch.Tensor, kv_tensor: torch.Tensor, model):
    # rotary_emb = model.model.layers[0].self_attn.rotary_emb
    # if hasattr(rotary_emb, 'inv_freq') and rotary_emb.inv_freq is not None:
    #     rotary_device = rotary_emb.inv_freq.device
    # else:
    #     # Fallback: use the device of the first layer
    #     rotary_device = next(model.model.layers[0].parameters()).device
    #
    # query = query.to(rotary_device)
    # kv_tensor = kv_tensor.to(rotary_device)
    #
    # position_ids = torch.full((1, kv_tensor.shape[2]), kv_tensor.shape[2], device=rotary_device)
    # chunk_key_for_rope = kv_tensor.to(rotary_device)
    # cos, sin = rotary_emb(chunk_key_for_rope, position_ids)
    # # mistral 限定
    # cos = cos.unsqueeze(1).to(rotary_device)
    # sin = sin.unsqueeze(1).to(rotary_device)
    # kv_tensor = (kv_tensor * cos) + (rotate_half(kv_tensor) * sin)

    # 1. 自动推导相关的维度
    batch_size, q_head, q_len, head_dim = query.shape
    _, kv_head, kv_len, _ = kv_tensor.shape

    # 计算分组比例 (16 / 2 = 8)
    num_queries_per_kv = q_head // kv_head
    k_v_shuffled = kv_tensor.repeat_interleave(num_queries_per_kv, dim=1)
    scale = 1.0 / (head_dim ** 0.5)
    attn_weights = torch.matmul(query, k_v_shuffled.transpose(-2, -1)) * scale
    output = torch.mean(attn_weights)
    return output


def find_closest_tensor_idx(query_tensor: torch.Tensor, tensor_list: list[torch.Tensor]) -> int:
    if not tensor_list:
        raise ValueError("输入的 tensor_list 不能为空！")

    best_idx = -1
    min_mse = float('inf')  # 初始化为一个无穷大的数

    for idx, t in enumerate(tensor_list):
        # 计算均方误差：(t1 - t2) 平方后的平均值
        # 使用 .item() 将单元素 Tensor 转换为 Python 的 float 类型，方便比较与提升性能
        mse = torch.mean((query_tensor - t) ** 2).item()

        if mse < min_mse:
            min_mse = mse
            best_idx = idx

    return best_idx


def calculate_mse(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    """
    计算两个 Tensor 之间的均方误差 (MSE)，并返回一个 float。
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Tensor 形状不匹配: {tensor_a.shape} vs {tensor_b.shape}")
    mse_tensor = torch.nn.functional.mse_loss(tensor_a.float(), tensor_b.float()).cpu()
    return mse_tensor.item()

def draft_model_find_most_similar_copy(
        model,
        passages: list[torch.Tensor],
        past_key_values,
        raw_load_path: str,
        preprocess_load_path: str,
        device: str,
        is_preprocess_list: list[bool],
        preprocess_cache_keys: list[list[str]],
        all_preprocess_doc_prefix_lens: list[list[int]],
        hash_keys: list[str],
        device_map=None,
):
    system_len = passages[0].shape[0]
    passages_len_wo_query = [passage.shape[0] for passage in passages[:-1]] ##不算query
    total_len_wo_query = sum(passages_len_wo_query)
    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = f"cuda:{device_map['model.embed_tokens']}" if device_map is not None else device
    key_cache_copies_list = []
    value_cache_copies_list = []
    for idx, passage in enumerate(passages[:-1]):
        key_cache = []
        value_cache = []
        if is_preprocess_list[idx]:
            load_path = preprocess_load_path
            for cache_copy_idx, preprocess_cache_key_ in enumerate(preprocess_cache_keys[idx]):
                if preprocess_cache_key_ != "":
                    preprocess_cache_key = preprocess_cache_key_ + "_"
                else:
                    preprocess_cache_key = ""
                cache_copy_prefix_len = all_preprocess_doc_prefix_lens[idx][cache_copy_idx]
                cache_copy_use_len = sum([x.shape[0] for x in passages[:idx]])
                chunk_key_cache = torch.load(f'{load_path}/{preprocess_cache_key}{hash_keys[idx]}_key.pt', weights_only=True).to(device)
                ## rope
                chunk_key_cache = revert_key_cache(model, chunk_key_cache,
                                                   cache_save_idx=cache_copy_prefix_len,
                                                   cache_use_idx=cache_copy_use_len)
                chunk_value_cache = torch.load(f'{load_path}/{preprocess_cache_key}{hash_keys[idx]}_value.pt', weights_only=True).to(device)
                key_cache.append(chunk_key_cache)
                value_cache.append(chunk_value_cache)
        else:
            load_path = raw_load_path
            chunk_key_cache = torch.load(f'{load_path}/{hash_keys[idx]}_key.pt', weights_only=True).to(device)
            chunk_value_cache = torch.load(f'{load_path}/{hash_keys[idx]}_value.pt', weights_only=True).to(device)
            key_cache.append(chunk_key_cache)
            value_cache.append(chunk_value_cache)
        key_cache_copies_list.append(key_cache)
        value_cache_copies_list.append(value_cache)

    with torch.no_grad():
        passages_len = [passage.shape[0] for passage in passages]
        passages_len_sum = sum(passages_len)
        k_need_index = [i for i in range(passages_len_sum)]
        cache_position = torch.tensor(k_need_index, device=input_device)
        reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(input_device)
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(input_device)
        model_output = model(
            inputs_embeds=inputs_embeds, cache_position=cache_position,
            past_key_values=past_key_values, return_dict=False, use_cache=True,
            use_sparse_attention=False,
        )[0]

        draft_model_prefilled_key_cache = torch.stack(past_key_values.key_cache)[:, :, :, :total_len_wo_query, :]
        draft_model_prefilled_value_cache = torch.stack(past_key_values.value_cache)[:, :, :, :total_len_wo_query, :]

        return key_cache_copies_list, value_cache_copies_list, draft_model_prefilled_key_cache, draft_model_prefilled_value_cache

        chosen_md5 = []
        chosen_md5_idx = []
        mean_key_before_list = [None for i in range(len(passages)-2)]
        mean_value_before_list = [None for i in range(len(passages)-2)]

        for psg_idx in range(len(passages_len_wo_query)):
            if not is_preprocess_list[psg_idx]:
                chosen_md5.append("")
                chosen_md5_idx.append(-1) ## no preprocess
                if psg_idx > 0:
                    mean_key_before_list[psg_idx - 1] = key_cache_copies_list[psg_idx][0]
                    mean_value_before_list[psg_idx - 1] = value_cache_copies_list[psg_idx][0]
            else:
                start_idx = sum(passages_len_wo_query[:psg_idx])
                end_idx = sum(passages_len_wo_query[:psg_idx+1])
                passage_key = draft_model_prefilled_key_cache[:, :, :, start_idx:end_idx, :]
                passage_value = draft_model_prefilled_value_cache[:, :, :, start_idx:end_idx, :]
                key_cache_list = key_cache_copies_list[psg_idx]
                value_cache_list = value_cache_copies_list[psg_idx]
                mse_min = 10e10
                min_hash_idx = -1
                for idx in range(len(key_cache_list)):
                    k_mse = calculate_mse(passage_key, key_cache_list[idx])
                    v_mse = calculate_mse(passage_value, value_cache_list[idx])
                    if k_mse + v_mse < mse_min:
                        mse_min = k_mse + v_mse
                        min_hash_idx = idx
                        mean_key_before_list[psg_idx-1] = key_cache_list[idx] ## -1是为了减去system prompt
                        mean_value_before_list[psg_idx-1] = value_cache_list[idx]
                chosen_md5.append(preprocess_cache_keys[psg_idx][min_hash_idx])
                chosen_md5_idx.append(min_hash_idx)

        mean_key_after = torch.stack(past_key_values.key_cache).mean(dim=0)[:, :, system_len:total_len_wo_query, :]
        mean_value_after = torch.stack(past_key_values.value_cache).mean(dim=0)[:, :, system_len:total_len_wo_query, :]
        mean_key_before = torch.cat(mean_key_before_list, dim=3).mean(dim=0)
        mean_value_before = torch.cat(mean_value_before_list, dim=3).mean(dim=0)

        compare_sim = analyze_print_and_return_max_mse_map(
            mean_key_before,
            mean_value_before,
            mean_key_after,
            mean_value_after,
            top_n=50
        )
        return chosen_md5, chosen_md5_idx, compare_sim



def draft_model_compare_kv_similarity(
        model,
        past_key_values,
        past_key_values_compare,
        passages,
        load_path='',
        preprocess_load_path='',
        revert_rope=False,
        device="cuda",
        device_map=None,
        hash_keys=None,
        query_states:list=None,
        query="",
        keyword="",
        preprocess=False
):
    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = f"cuda:{device_map['model.embed_tokens']}" if device_map is not None else device

    passages_len = [passage.shape[0] for passage in passages]
    passages_len_sum = sum(passages_len)
    passages_len_sum_without_query = sum([passage.shape[0] for passage in passages[:-1]])

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    system_len = passages[0].shape[0]

    key_cache = []
    value_cache = []

    chunk_ids = list(range(len(passages) - 1))
    mean_attn_weights = []

    for idx, passage in enumerate(passages[:-1]):
        ##fixme： mengyao_debug，第一个(system prompt)和第二个（首个文档）都是不需要preprocess的
        if idx >=2 and preprocess:
            chunk_key_cache = torch.load(f'{preprocess_load_path}/{hash_keys[idx]}_key.pt', weights_only=True).to('cpu')
            chunk_value_cache = torch.load(f'{preprocess_load_path}/{hash_keys[idx]}_value.pt', weights_only=True).to('cpu')
        else:
            chunk_key_cache = torch.load(f'{load_path}/{hash_keys[idx]}_key.pt', weights_only=True).to('cpu')
            print(load_path)
            chunk_value_cache = torch.load(f'{load_path}/{hash_keys[idx]}_value.pt', weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)
        if idx>=1:
            mean_attn_weight = compute_gqa_attention_inplace(
                query=query_states[0].to('cpu'),
                kv_tensor=chunk_key_cache[0],
                model=model
            )
            mean_attn_weights.append(mean_attn_weight)
    ## load kv cache
    past_len = load_kv(model, passages, chunk_ids, key_cache, value_cache, input_device, past_key_values, revert_rope, system_len, query=query)
    past_len = load_kv(model, passages, chunk_ids, key_cache, value_cache, input_device, past_key_values_compare, revert_rope, system_len, query=query)

    k_need_index = [i for i in range(passages_len_sum)]

    use_sparse_attention = False
    reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(input_device)
    cache_position = torch.tensor(k_need_index, device=input_device)
    with torch.no_grad():
        without_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(input_device)

        # Don't force move to input_device - keep on the device where model output is
        # This avoids cross-GPU transfer deadlock in PP mode
        start_time = time.time()
        model_output = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position,
            past_key_values=past_key_values, return_dict=False, use_cache=True, use_sparse_attention=use_sparse_attention,
        )[0]

        mean_key_after = torch.stack(past_key_values.key_cache).mean(dim=0)[:, :, system_len:passages_len_sum_without_query, :]
        mean_value_after = torch.stack(past_key_values.value_cache).mean(dim=0)[:, :, system_len:passages_len_sum_without_query, :]
        mean_key_before = torch.stack(past_key_values_compare.key_cache).mean(dim=0)[:, :, system_len:passages_len_sum_without_query, :]
        mean_value_before = torch.stack(past_key_values_compare.value_cache).mean(dim=0)[:, :, system_len:passages_len_sum_without_query, :]

        if "mse" in keyword:
            ## mse，越小越相似
            result = analyze_print_and_return_max_mse_map(
                mean_key_before,
                mean_value_before,
                mean_key_after,
                mean_value_after,
                top_n=50
            )
        else:
            ## 余弦相似度，越大越相似
            result = analyze_print_and_return_min_sim_map(
                mean_key_before,
                mean_value_before,
                mean_key_after,
                mean_value_after,
                top_n=50
            )


        return result, mean_attn_weights


def analyze_print_and_return_min_sim_map(
        mean_key_before: torch.Tensor,
        mean_value_before: torch.Tensor,
        mean_key_after: torch.Tensor,
        mean_value_after: torch.Tensor,
        top_n: int = 10
) -> dict:
    """
    全量计算每个位置(Index)的 KV 拯救权重。
    kv_combined_weight_map 直接计算 KV 拼接矩阵 [seq_len, 256] 的 Cos-Sim 差异。
    """
    assert mean_key_before.shape == mean_key_after.shape, "Before 和 After 的 Shape 必须一致"

    # 1. 提取特征向量维度 -> [seq_len, 128]
    k_before = mean_key_before[0].transpose(0, 1).flatten(1).float()
    v_before = mean_value_before[0].transpose(0, 1).flatten(1).float()
    k_after = mean_key_after[0].transpose(0, 1).flatten(1).float()
    v_after = mean_value_after[0].transpose(0, 1).flatten(1).float()

    seq_len = k_before.shape[0]

    # 2. 🚀【核心修改】：在 dim=-1 (128维度) 上将 K 和 V 拼接成 [seq_len, 256] 的联合向量
    kv_before_cat = torch.cat([k_before, v_before], dim=-1)
    kv_after_cat = torch.cat([k_after, v_after], dim=-1)

    # 3. 向量化并行计算各部分的余弦相似度
    time_start = time.time()

    # 分别算 K 和 V 用于满足原输出格式的分支
    k_cos = torch.cosine_similarity(k_before, k_after, dim=-1).cpu().numpy()
    v_cos = torch.cosine_similarity(v_before, v_after, dim=-1).cpu().numpy()

    # 直接计算拼接后 256 维联合特征的相似度
    kv_combined_cos = torch.cosine_similarity(kv_before_cat, kv_after_cat, dim=-1).cpu().numpy()

    print(f"time to compute similarity: {time.time() - time_start:.6f} s")

    # 4. 统一转换成 1 - cos 的失真权重
    k_weights = 1.0 - k_cos
    v_weights = 1.0 - v_cos
    kv_combined_weight = 1.0 - kv_combined_cos  # 联合整体差异

    # 5. 保持原格式顺序返回
    result = {
        "key_min_sim_map": {i: float(k_weights[i]) for i in range(seq_len)},
        "value_min_sim_map": {i: float(v_weights[i]) for i in range(seq_len)},
        "kv_combined_weight_map": {i: float(kv_combined_weight[i]) for i in range(seq_len)}
    }
    # plot_and_save_weight_distribution(result)
    return result



def plot_and_save_weight_distribution(sim_maps: dict, save_path: str = "weight_distribution.png"):
    """
    接收分析函数的返回字典，绘制 K、V 及 KV 拼接联合权重的全量分布图，并保存到本地。

    参数:
        sim_maps: 包含 "key_min_sim_map", "value_min_sim_map", "kv_combined_weight_map" 的字典
        save_path: 本地图片保存路径
    """
    # 1. 提取数据并按 Index 排序（确保 X 轴顺序正确）
    k_dict = sim_maps["key_min_sim_map"]
    v_dict = sim_maps["value_min_sim_map"]
    kv_dict = sim_maps["kv_combined_weight_map"]

    indices = sorted(k_dict.keys())
    seq_len = len(indices)

    k_vals = [k_dict[i] for i in indices]
    v_vals = [v_dict[i] for i in indices]
    kv_vals = [kv_dict[i] for i in indices]

    # 2. 设置学术图表风格
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax_main, ax_top) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 2]})

    # =========================================================================
    # 子图 1: 全量 Token 权重分布走势图（折线 + 面积）
    # =========================================================================
    ax_main.plot(indices, k_vals, color='#1f77b4', alpha=0.8, linewidth=1.5, label='Key Weight (1-Cos)')
    ax_main.fill_between(indices, k_vals, color='#1f77b4', alpha=0.1)

    ax_main.plot(indices, v_vals, color='#2ca02c', alpha=0.8, linewidth=1.5, label='Value Weight (1-Cos)')
    ax_main.fill_between(indices, v_vals, color='#2ca02c', alpha=0.1)

    # 联合拼接特征由于是 256 维夹角，用醒目的紫色粗实线绘制
    ax_main.plot(indices, kv_vals, color='#9467bd', alpha=0.9, linewidth=2.5, label='KV Combined Weight (1-Cat_Cos)')
    ax_main.fill_between(indices, kv_vals, color='#9467bd', alpha=0.15)

    ax_main.set_title(f"KV Cache Distortion Weight Distribution Across Sequence (Seq_Len: {seq_len})",
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_ylabel("Distortion Weight (Higher = More Drift)", fontsize=12, fontweight='bold')
    ax_main.set_xlabel("Token Sequence Index", fontsize=12, fontweight='bold')
    ax_main.set_xlim(0, seq_len - 1)
    ax_main.set_ylim(-0.02, max(max(k_vals), max(v_vals), max(kv_vals)) * 1.1)
    ax_main.legend(loc='upper right', frameon=True, fontsize=11)

    # =========================================================================
    # 子图 2: Top-15 异常（失真最严重）Token 的横向对比柱状图
    # =========================================================================
    # 找出联合失真权重最大的 Top-15 个位置
    top_n = min(15, seq_len)
    top_indices = sorted(indices, key=lambda i: kv_dict[i], reverse=True)[:top_n]

    x_labels = [f"Idx {i}" for i in top_indices]
    top_k = [k_dict[i] for i in top_indices]
    top_v = [v_dict[i] for i in top_indices]
    top_kv = [kv_dict[i] for i in top_indices]

    x = np.arange(top_n)
    width = 0.25

    ax_top.bar(x - width, top_k, width, label='Key Weight', color='#1f77b4', alpha=0.7)
    ax_top.bar(x, top_v, width, label='Value Weight', color='#2ca02c', alpha=0.7)
    ax_top.bar(x + width, top_kv, width, label='KV Combined', color='#9467bd', alpha=0.9)

    ax_top.set_title(f"Top-{top_n} Most Distorted Tokens Analysis", fontsize=13, fontweight='bold', pad=10)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax_top.set_ylabel("Weight Value", fontsize=12, fontweight='bold')
    ax_top.legend(loc='upper right', frameon=True)

    # 3. 规整布局并保存
    plt.tight_layout()

    # 自动创建不存在的父级目录
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 [📊 可视化成功] 权重分布图已成功保存至本地: {os.path.abspath(save_path)}")
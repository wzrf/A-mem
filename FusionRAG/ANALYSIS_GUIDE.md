# FusionRAG 失败案例分析指南

## 目标

分析 FusionRAG 在 rate=0.3 时失败但 rate=1.0 时成功的案例，从 attention 分数和 logits 变化中找出改进方案。

## 分析工具

### 1. `analyze_failure_cases.py` - 失败案例识别

**功能**：
- 对比 rate=1.0 和 rate=0.3 的测试结果
- 识别失败案例（rate=1正确，rate=0.3错误）
- 生成失败案例列表供详细分析

**使用方法**：
```bash
# 首先运行测试生成结果
python test_fusionrag_reflect.py --rate 1.0 --max-samples 100
python test_fusionrag_reflect.py --rate 0.3 --max-samples 100

# 然后分析失败案例
python analyze_failure_cases.py
```

**输出**：
- `failure_cases_analysis.json`: 包含所有失败案例的详细信息

### 2. `deep_analysis_tool.py` - 深度分析工具

**功能**：
- 追踪每层的 attention weights
- 记录每层的 hidden states
- 对比 logits 分布
- 识别分歧点
- 生成改进建议

**核心分析指标**：

1. **Attention KL Divergence**
   ```
   KL(Attn_r1 || Attn_r0.3) = Σ P(r1) * log(P(r1) / P(r0.3))
   ```
   - 衡量 attention 分布的变化
   - 高值表示 attention 模式显著不同

2. **Hidden State Cosine Similarity**
   ```
   sim = (H_r1 · H_r0.3) / (||H_r1|| * ||H_r0.3||)
   ```
   - 衡量隐藏层状态的相似度
   - 低值表示表征已经发散

3. **Logits KL Divergence**
   ```
   KL(P_r1 || P_r0.3) at each position
   ```
   - 衡量预测分布的差异
   - 找出首次分歧的位置

## 可能的问题根源

### 问题1: Attention 分布偏移

**症状**：
- Attention KL divergence 在某些层特别高
- Query 对关键 tokens 的 attention 权重下降

**原因**：
- Reprocess 时部分 tokens 的 KV cache 被重新计算
- 新计算的 KV 与预处理的 KV 在数值上有微小差异
- RoPE 位置编码的累积误差

**诊断方法**：
```python
# 在 deep_analysis_tool.py 中
layer_analysis = analyzer.analyze_layer(layer_idx)
if layer_analysis.attention_kl_div > 1.0:
    print(f"Layer {layer_idx} has high attention divergence!")
    # 进一步分析哪些 attention heads 变化最大
```

### 问题2: 累积误差放大

**症状**：
- 浅层 KL divergence 较小
- 深层 KL divergence 急剧增大
- Hidden state similarity 随层数递减

**原因**：
- 早期层的小误差在后续层中放大
- 非线性激活函数放大差异
- 残差连接传播误差

**诊断方法**：
```python
# 检查每层的 divergence 是否递增
divergences = [la['attention_kl_div'] for la in layer_analysis]
if np.gradient(divergences).mean() > 0.1:
    print("Error accumulation detected!")
```

### 问题3: 关键 Token 的 KV 被近似

**症状**：
- 某些特定位置的 logits 差异特别大
- 这些位置通常对应答案的关键词
- Attention 在这些位置的权重在 rate=0.3 时下降

**原因**：
- FusionRAG 的 reprocess 策略没有识别出关键 tokens
- 关键信息被压缩或近似

**诊断方法**：
```python
# 分析divergent positions是否集中在答案关键词
pred_analysis = analyzer.analyze_final_predictions()
for detail in pred_analysis['position_details']:
    # 检查这个位置是否在ground truth中
    # 如果是，说明关键token被错误处理
```

## 改进方案

### 方案1: 自适应 Recomputation

**思路**：根据 token 重要性动态调整 recomputation ratio

**实现**：
```python
def adaptive_recomputation(attention_scores, base_ratio=0.3):
    """
    根据attention分数自适应调整recomputation ratio

    Args:
        attention_scores: [num_heads, seq_len, seq_len]
        base_ratio: 基础recomputation ratio

    Returns:
        per_token_ratio: [seq_len] 每个token的recomputation ratio
    """
    # 计算每个token的平均attention weight（被其他token attend的程度）
    importance = attention_scores.mean(dim=(0, 1))  # [seq_len]

    # 归一化
    importance = (importance - importance.min()) / (importance.max() - importance.min())

    # 重要的token使用更高的ratio
    per_token_ratio = base_ratio + (1 - base_ratio) * importance

    return per_token_ratio
```

**预期效果**：
- 关键 tokens 完全 recompute
- 不重要的 tokens 使用更低的 ratio
- 整体 ratio 仍保持在 30% 左右

### 方案2: 分层 Recomputation 策略

**思路**：不同层使用不同的 recomputation ratio

**实现**：
```python
def layer_wise_recomputation(num_layers):
    """
    为每层分配不同的recomputation ratio

    浅层：低ratio（表征稳定）
    深层：高ratio（误差累积大）
    """
    ratios = []

    for layer_idx in range(num_layers):
        if layer_idx < num_layers // 3:
            # 浅层：使用低ratio
            ratio = 0.2
        elif layer_idx < 2 * num_layers // 3:
            # 中层：使用中等ratio
            ratio = 0.3
        else:
            # 深层：使用高ratio
            ratio = 0.5

        ratios.append(ratio)

    return ratios
```

**预期效果**：
- 在相同平均 ratio 下提高准确率
- 减少深层的误差累积

### 方案3: Attention-Guided Recomputation

**思路**：使用 attention score 作为选择 recompute tokens 的依据

**实现**：
```python
def attention_guided_selection(query_embedding, kv_cache, ratio=0.3):
    """
    基于query与KV的attention score选择需要recompute的tokens

    Args:
        query_embedding: Query的embedding [d_model]
        kv_cache: 预处理的KV cache
        ratio: recomputation ratio

    Returns:
        tokens_to_recompute: 需要recompute的token indices
    """
    # 计算query与所有key的attention score
    keys = kv_cache['keys']  # [seq_len, d_model]
    scores = torch.matmul(keys, query_embedding)  # [seq_len]
    scores = F.softmax(scores, dim=0)

    # 选择top-k最相关的tokens进行recompute
    k = int(len(scores) * ratio)
    top_k_indices = torch.topk(scores, k=k).indices

    return top_k_indices
```

**预期效果**：
- 根据 query 动态选择需要 recompute 的 tokens
- 确保与 query 最相关的 KV 被准确计算

### 方案4: 分层 KL 正则化

**思路**：在训练/微调时加入 KL divergence 约束

**实现**：
```python
def kl_regularization_loss(
    logits_full,
    logits_approx,
    alpha=0.1
):
    """
    KL divergence正则化损失

    鼓励approximate版本的输出接近full版本

    Args:
        logits_full: Full recomputation的logits
        logits_approx: Approximate的logits
        alpha: 正则化系数

    Returns:
        loss: KL正则化损失
    """
    p = F.softmax(logits_full, dim=-1)
    q = F.softmax(logits_approx, dim=-1)

    kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

    return alpha * kl_loss
```

**预期效果**：
- 模型学会在部分 recomputation 下仍保持准确性
- 提高对 KV approximation 的鲁棒性

### 方案5: 混合精度 KV Cache

**思路**：关键 tokens 使用 FP32，其他使用 FP16/INT8

**实现**：
```python
def mixed_precision_kv_cache(kv_cache, importance_scores, high_precision_ratio=0.3):
    """
    根据重要性使用混合精度存储KV cache

    Args:
        kv_cache: 原始KV cache (FP32)
        importance_scores: Token重要性分数
        high_precision_ratio: 使用高精度的比例

    Returns:
        mixed_kv_cache: 混合精度的KV cache
    """
    k = int(len(importance_scores) * high_precision_ratio)
    high_precision_indices = torch.topk(importance_scores, k=k).indices

    mixed_kv = {}

    for i, kv in enumerate(kv_cache):
        if i in high_precision_indices:
            mixed_kv[i] = kv.to(torch.float32)
        else:
            mixed_kv[i] = kv.to(torch.float16)  # or int8

    return mixed_kv
```

**预期效果**：
- 减少数值误差
- 保持内存效率

## 实验方案

### 阶段1: 诊断分析（当前）

1. 运行基准测试
   ```bash
   python test_fusionrag_reflect.py --rate 1.0 --max-samples 100
   python test_fusionrag_reflect.py --rate 0.3 --max-samples 100
   ```

2. 识别失败案例
   ```bash
   python analyze_failure_cases.py
   ```

3. 深度分析失败案例
   ```python
   # 修改test脚本，集成deep_analysis_tool
   # 对失败案例进行逐层分析
   ```

### 阶段2: 试验改进方案

1. **方案1测试**: 自适应 recomputation
   - 修改 FusionRAG reprocess logic
   - 对比 baseline vs adaptive
   - 记录准确率和性能变化

2. **方案2测试**: 分层策略
   - 实现layer-wise ratio
   - A/B测试不同配置

3. **方案3测试**: Attention-guided
   - 实现query-dependent selection
   - 对比与random selection

### 阶段3: 综合评估

对比所有方案：
- 准确率提升
- 延迟增加
- 内存使用
- 适用场景

## 预期分析结果示例

```json
{
  "summary": {
    "num_layers": 28,
    "avg_attention_kl": 0.342,
    "max_attention_kl": 1.856,
    "layer_with_max_kl": 23,
    "avg_hidden_sim": 0.912,
    "min_hidden_sim": 0.776,
    "layer_with_min_sim": 24
  },
  "prediction_analysis": {
    "divergence_rate": 0.187,
    "avg_kl_divergence": 0.445,
    "first_divergence_position": 42
  },
  "recommendations": [
    "⚠️  Layer 23-24 show high divergence. Consider increasing recomputation ratio for these layers.",
    "⚠️  First divergence at position 42 (middle of answer). Implement attention-guided selection."
  ]
}
```

## 下一步

1. **立即执行**：运行 analyze_failure_cases.py 获取失败案例
2. **深度分析**：选择 3-5 个典型失败案例进行逐层分析
3. **方案选择**：基于分析结果选择 1-2 个最有希望的改进方案
4. **原型实现**：快速实现选定方案的原型
5. **效果验证**：在失败案例上验证改进效果

---

**注意**：这些工具提供了系统化的分析框架。实际改进需要根据具体的失败模式调整策略。

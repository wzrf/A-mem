# Per-KV-Head Token Selection: Diversity Pattern Analysis

## 实验配置

- **模型**: Qwen2.5-7B-Instruct (28 layers, 4 KV heads/layer, 7 query heads/KV head)
- **数据**: Example 4, Sub-question 1 - "When was King Henry III of England crowned?"
- **Selection ratio**: 30%
- **Recomputable tokens**: 1927 (跳过了system prompt和chunk 1，严格prefix cache命中)
- **每个head选择**: 578 tokens

---

## 关键发现：Head Clustering现象

### 发现1：4个KV Heads形成了明显的两个Cluster

**Cluster A: {Head 0, Head 2, Head 3}** - 在深层高度相似
**Cluster B: {Head 1}** - 独立，与其他heads差异巨大

### 发现2：层间差异演化模式

| Layer | Head 0 vs 1 | Head 0 vs 2 | Head 0 vs 3 | Head 1 vs 2 | Head 1 vs 3 | Head 2 vs 3 |
|-------|-------------|-------------|-------------|-------------|-------------|-------------|
| **Layer 0** (浅层) | **1.4%** ⚠️ | 53.5% | 16.1% | **7.3%** ⚠️ | 16.9% | 25.7% |
| **Layer 14** (中层) | 24.6% | 40.8% | 34.0% | 37.1% | 31.5% | 40.8% |
| **Layer 27** (深层) | **14.0%** ⚠️ | **81.8%** ✓ | **69.0%** ✓ | **14.5%** ⚠️ | 15.3% | **72.0%** ✓ |

**图示演化**：
```
浅层(L0):   H0━━━━━H2     H1(独立)     H3
            53.5%       极低overlap

中层(L14):  H0━━H2━━H3━━H1
            所有pairs都在25-40%范围，相对均匀

深层(L27):  H0━━━━━━━━H2━━━━━━H3     H1(极度独立)
            81.8%     72.0%        ≈14% with all
```

---

## 详细模式分析

### Pattern 1: Head 1的"独立性"

**现象**：Head 1在所有层都与其他heads有极低的overlap

| Layer | H1 vs H0 | H1 vs H2 | H1 vs H3 | 平均 |
|-------|----------|----------|----------|------|
| 0     | 1.4%     | 7.3%     | 16.9%    | 8.5% |
| 14    | 24.6%    | 37.1%    | 31.5%    | 31.1% |
| 27    | 14.0%    | 14.5%    | 15.3%    | 14.6% |

**解读**：
- Head 1关注的是**完全不同的信息模式**
- 在浅层（L0），Head 1与Head 0几乎没有共同关注的tokens（1.4%）
- 在深层（L27），这种独立性反而增强（14.0%）
- 可能的角色：**Head 1负责捕获其他heads忽略的"长尾信息"或"反向模式"**

### Pattern 2: Head 0和Head 2的"融合"

**现象**：Head 0和Head 2从中等相似（53.5%）演化到高度相似（81.8%）

| Layer | H0 vs H2 Overlap |
|-------|------------------|
| 0     | 53.5%            |
| 14    | 40.8%            |
| 27    | 81.8% ✓          |

**演化趋势**：U型曲线
- L0: 53.5% - 初始有一定相似性
- L14: 40.8% - 中层分化，各自探索
- L27: 81.8% - 深层收敛，达成共识

**解读**：
- 浅层：Head 0和2有初步的共同关注点（可能是显著特征）
- 中层：分化探索不同的语义维度
- **深层：收敛到高度一致的"核心语义信息"**
- 可能的角色：**Head 0和2负责识别和强化核心语义**

### Pattern 3: Head 2和Head 3的"协同"

**现象**：Head 2和Head 3在深层也高度相似（72.0%）

| Layer | H2 vs H3 Overlap |
|-------|------------------|
| 0     | 25.7%            |
| 14    | 40.8%            |
| 27    | 72.0% ✓          |

**演化趋势**：单调递增
- L0: 25.7% - 初期较低
- L14: 40.8% - 逐步接近
- L27: 72.0% - 深层高度协同

**解读**：
- Head 2和3在深层形成"协同处理单元"
- 与Head 0形成三角形结构：H0↔H2↔H3，彼此高度重叠
- 可能的角色：**三个heads共同强化最终的语义表示**

### Pattern 4: 中层的"探索期"

**现象**：Layer 14所有head pairs的overlap都在25-40%范围内，相对均匀

| Head Pair | L0 Overlap | L14 Overlap | L27 Overlap | L14特征 |
|-----------|------------|-------------|-------------|---------|
| H0 vs H1  | 1.4%       | 24.6%       | 14.0%       | 峰值！  |
| H0 vs H2  | 53.5%      | 40.8%       | 81.8%       | 谷值    |
| H0 vs H3  | 16.1%      | 34.0%       | 69.0%       | 中间    |
| H1 vs H2  | 7.3%       | 37.1%       | 14.5%       | 峰值！  |

**解读**：
- L14是一个"重组期"
- 原本极低overlap的pairs（H0-H1, H1-H2）在中层达到峰值
- 原本高overlap的pairs（H0-H2）在中层降到谷值
- **中层是heads进行"信息交换和重新定位"的关键阶段**

---

## 4个KV Heads的角色假设

基于overlap patterns，推测每个head的功能分工：

### Head 0: "主干语义提取器"
- 与Head 2高度协同（深层81.8%）
- 与Head 3中度协同（深层69.0%）
- 与Head 1低度交互（深层14.0%）
- **角色**：提取核心语义特征，作为主要的信息聚合节点

### Head 1: "长尾/反向信息捕获器"
- 与所有其他heads保持独立（平均14.6% overlap）
- 在所有层都保持这种独立性
- **角色**：
  - 捕获其他heads忽略的"边缘信息"
  - 可能关注反向模式（如"what NOT to attend to"）
  - 提供多样性，防止过度聚焦
  - 类似"探索者"角色，寻找非主流但可能关键的信息

### Head 2: "核心语义强化器"
- 与Head 0深层高度融合（81.8%）
- 与Head 3深层高度协同（72.0%）
- 与Head 1完全分离（14.5%）
- **角色**：
  - 强化和验证Head 0识别的核心语义
  - 形成主流语义共识
  - 可能负责"确信度"评估

### Head 3: "语义支持器"
- 与Head 0、2形成三角协同（69-72%）
- 与Head 1低交互（15.3%）
- **角色**：
  - 支持Head 0和2的核心语义
  - 提供额外的语义验证
  - 可能处理语义的某个特定维度（如时间、实体等）

---

## 信息流动假设

### 浅层（Layer 0）: 初步分化
```
H0: 关注显著特征A
H1: 关注边缘特征B（完全不同）
H2: 关注显著特征A'（与H0部分重叠53.5%）
H3: 关注中间特征C
```

### 中层（Layer 14）: 信息交换和重组
```
所有heads进行"信息交换"
H1临时增加与其他heads的交互（探索主流语义）
H0-H2减少overlap（分化探索）
→ 为深层的收敛做准备
```

### 深层（Layer 27）: 收敛和分工明确
```
主流共识：H0 ━━━━━ H2 ━━━━ H3  (81.8%, 72.0% overlap)
          ↓强化核心语义

独立探索：H1 (14% overlap with all)
          ↓保持多样性
```

---

## 对"1216 vs 1220"问题的启示

### 当前问题的context
- Ground truth: "1216 at Gloucester + 1220 at Westminster"
- Uniform selection (rate=0.3): 只生成"1220"
- Layer-wise selection: 生成"1220"但后续解释提到了"1216"

### Head-wise可能的优势

基于上述分析，per-head selection可能：

1. **Head 1捕获"1216"相关的长尾信息**
   - 其他heads聚焦于"1220"（主流、更正式的加冕）
   - Head 1独立关注"1216"（临时加冕、Gloucester）
   - 两个信息都被保留

2. **Head 0/2/3强化"1220"的核心语义**
   - 三个heads共识："1220 at Westminster Abbey"是主要答案
   - 高overlap确保这个信息被强烈保留

3. **中层的信息交换**
   - Layer 14的"探索期"可能帮助整合"1216"和"1220"
   - 不同heads临时交互，发现两个日期的关联

### 预期效果

如果实现per-head generation：
- **可能生成**: "1216 and 1220" 或 "first in 1216, then 1220"
- **原因**:
  - Head 1保留了"1216"信息
  - Head 0/2/3保留了"1220"信息
  - 综合所有heads的cache，两个信息都可用

---

## Union vs Intersection 策略

### Option 1: Union（任一head选中就保留）

**优势**：
- 保留所有heads关注的信息
- 最大化信息覆盖
- 578×4 = 2312个unique tokens（去重后可能1200-1400）

**劣势**：
- 实际重算比例可能超过30%
- 可能引入noise（某个head误判的tokens）

### Option 2: Intersection（所有heads共选才保留）

**优势**：
- 只保留"高共识"tokens
- 实际重算比例远低于30%（可能5-10%）
- 聚焦核心信息

**劣势**：
- 丢失大量可能有用的信息
- Head 1的独特信息会被完全忽略

### Option 3: Weighted Union（推荐）

**策略**：
```python
# 每个token的权重 = 选择它的heads数量
weight[token] = sum([1 for head in heads if token in head.selected])

# 选择高权重的tokens（至少2个heads选中）
selected = [token for token in all_tokens if weight[token] >= 2]
```

**优势**：
- 平衡覆盖和精度
- 保留"至少有一定共识"的tokens
- 可能包含Head 0-Head 2的共同关注（core）+ Head 1的部分独特信息

---

## 后续实验建议

### 1. 分析实际选择的tokens内容

看看不同heads到底选了什么：
```python
# Head 0选择的top 10 tokens（按attention score）
# Head 1选择的top 10 tokens
# Head 2选择的top 10 tokens
# Head 3选择的top 10 tokens

# 分析：
# - Head 1是否真的选择了"1216"相关tokens？
# - Head 0/2/3是否都选择了"1220"相关tokens？
# - 差异主要在哪些类型的信息上？
```

### 2. Visualize head attention heatmap

为每个head生成attention heatmap：
```
             Token1  Token2  Token3  ...  Token1927
Head 0:        0.8     0.2     0.1   ...    0.5
Head 1:        0.1     0.9     0.3   ...    0.2
Head 2:        0.7     0.3     0.2   ...    0.6
Head 3:        0.6     0.4     0.1   ...    0.4
```

找出：
- Head 1独有的高attention tokens
- 所有heads共同高attention的tokens

### 3. 测试不同的Union策略

实现并对比：
- Pure union (all heads)
- Weighted union (≥2 heads)
- Majority union (≥3 heads)
- Core only (all 4 heads)

看哪个策略生成的答案最完整。

### 4. 跨层分析

不只看单层的4个heads，还要看：
- 浅层（L0-9）的heads聚焦什么？
- 中层（L10-19）的heads聚焦什么？
- 深层（L20-27）的heads聚焦什么？
- 是否存在"浅层关注局部，深层关注全局"的模式？

---

## 总结

**核心洞察**：

1. ✅ **Per-head selection确实有效** - 不同heads关注不同信息，overlap低至1.4%

2. ✅ **存在明显的head clustering** - {H0, H2, H3}形成主流共识，H1独立探索

3. ✅ **层间演化模式清晰** - 浅层分化 → 中层交换 → 深层收敛

4. ⭐ **潜在优势** - Head 1可能保留了被uniform/layer-wise selection忽略的关键信息（如"1216"）

5. ⚠️ **需要验证** - 必须实现生成部分才能确认是否真的提升了答案完整性

**下一步**：
1. 实现per-head的custom attention和generation
2. 验证是否能生成"1216 and 1220"
3. 分析实际选择的tokens内容
4. 对比不同union策略的效果

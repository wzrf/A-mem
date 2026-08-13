# 重新评估脚本使用指南

## 修改内容

### 1. test_fusionrag_reflect.py 的改进

修改了 `judge_answer_with_openai` 函数，新增以下功能：

- ✅ **返回详细原因**：不仅返回对错判断，还返回详细的判断理由
- ✅ **中文 Prompt**：使用中文提示词，更适合中文模型
- ✅ **格式化输出**：要求模型按照 "判断: [正确/错误]" 和 "原因: [详细说明]" 格式回答
- ✅ **CSV 新增列**：在结果 CSV 中新增 `Reason` 列，保存判断原因

#### 使用方法

```python
# 直接运行修改后的脚本即可
python test_fusionrag_reflect.py
```

脚本会自动：
1. 对每个子问题进行判断
2. 打印判断结果和原因
3. 将结果保存到 CSV（包含 Reason 列）

#### CSV 输出格式

| Main Question | Sub Question | Ground Truth | Predicted | Correct | Reason |
|--------------|--------------|--------------|-----------|---------|--------|
| 主问题 | 子问题 | 标准答案 | 预测答案 | True/False | 详细判断原因 |

---

## 2. reevaluate_results.py - 重新评估已有结果

这个脚本用于重新评估 `/mnt/data/junshi/` 和 `/mnt/data/reflect` 目录下已有的 CSV 结果文件。

### 主要功能

1. **批量重新评估**：自动查找指定目录下的所有 CSV 文件
2. **保留旧结果**：在新 CSV 中保留旧的判断结果（`Old Correct` 列）
3. **对比变化**：显示判断结果的变化
4. **详细原因**：为每个答案提供详细的判断原因
5. **统计分析**：生成主问题和子问题的准确率统计

### 使用方法

#### 方式 1：直接运行（默认配置）

```bash
cd /mnt/data/wjh/FusionRAG
python reevaluate_results.py
```

默认会评估：
- `/mnt/data/junshi/` 下的所有 CSV 文件
- `/mnt/data/reflect` 下的所有 CSV 文件

#### 方式 2：自定义配置

```python
from reevaluate_results import main

main(
    directories=[
        '/mnt/data/junshi',
        '/mnt/data/reflect',
        '/path/to/other/directory'  # 可以添加更多目录
    ],
    openai_api_key="your-api-key",
    openai_base_url="https://api.deepseek.com/v1",
    openai_model="deepseek-chat"
)
```

#### 方式 3：只评估单个文件

```python
from reevaluate_results import reevaluate_csv, OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1"
)

reevaluate_csv(
    csv_path="/path/to/your/result.csv",
    openai_client=client,
    openai_model="deepseek-chat"
)
```

### 输出文件

对于每个输入的 CSV 文件 `result.csv`，会生成：

1. **`result_reevaluated.csv`**：重新评估的详细结果
   - 包含原始的所有列
   - 新增 `Old Correct` 列（保留旧的判断）
   - 新增 `Reason` 列（详细判断原因）

2. **`result_reevaluated_stats.txt`**：统计摘要
   - 主问题准确率
   - 子问题准确率
   - 每个主问题的详细统计

### 示例输出

#### 控制台输出

```
================================================================================
重新评估: /mnt/data/junshi/results/FusionRAG_global_topk_10_rate_0.2.csv
================================================================================
共 50 条记录需要重新评估

[1/50] 评估中...
子问题: 土耳其接收的俄罗斯防空导弹系统具体是什么型号？...
  结果: ✓ CORRECT
  原因: 预测答案"S-400防空系统"与标准答案"S-400"在语义上完全一致...

[2/50] 评估中...
子问题: S-400防空导弹系统击落的战机是什么...
  ⚠️  判断变化: True -> False
  结果: ✗ INCORRECT
  原因: 预测答案提到"F-16战斗机"但缺少关键信息"飞行员姓名"...

================================================================================
评估完成！
总计: 50 条
正确: 38 条
准确率: 76.00%
结果已保存至: /mnt/data/junshi/results/FusionRAG_global_topk_10_rate_0.2_reevaluated.csv
================================================================================
```

#### stats.txt 输出示例

```
================================================================================
重新评估统计结果
================================================================================

主问题准确率: 8/13 (0.6154)
子问题准确率: 38/50 (0.7600)

================================================================================
主问题详细统计:
================================================================================

1. ✓ 土耳其接收了某型号的俄罗斯防空导弹系统...
   子问题正确率: 3/3

2. ✗ What is the network which National Cycle Route 57...
   子问题正确率: 1/2

...
```

---

## 重新评估的意义

1. **更一致的评估标准**：使用相同的 prompt 和模型重新评估所有结果
2. **发现评估差异**：对比新旧判断结果，发现可能的评估问题
3. **详细的错误分析**：通过判断原因了解模型为什么答错
4. **便于改进**：基于详细原因优化模型或数据集

---

## 注意事项

1. **API 费用**：重新评估会调用 OpenAI API，请注意费用
2. **速率限制**：如果文件很多，注意 API 的速率限制
3. **备份数据**：重新评估前建议备份原始 CSV 文件
4. **环境变量**：可以设置 `OPENAI_API_KEY` 环境变量而不是在代码中硬编码

---

## 常见问题

### Q1: 如何修改评估标准？

修改 `judge_answer_with_openai` 函数中的 `judge_prompt`，调整判断标准。

### Q2: 如何使用其他模型？

修改 `openai_model` 参数，例如：
- `"gpt-4"`
- `"gpt-3.5-turbo"`
- `"deepseek-chat"`

### Q3: 重新评估需要多久？

取决于：
- CSV 文件数量和大小
- API 响应速度
- 每个答案约需 1-3 秒

预估：100 条记录约需 3-5 分钟

### Q4: 可以并行评估吗？

当前版本是串行评估。如需并行，可以使用 `concurrent.futures` 或 `asyncio`。

---

## 联系方式

如有问题，请查看代码注释或联系开发者。

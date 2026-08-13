# Multi-GPU Parallel Testing Guide

本指南说明如何使用多 GPU 并行测试功能加速 FusionRAG 测试。

## 前置要求

### 1. 安装依赖
```bash
pip install filelock
```

### 2. 确认 GPU 可用性
```bash
nvidia-smi
```

## 使用方法

### 快速开始

在 `test_fusionrag_reflect.py` 的 `__main__` 部分修改配置：

```python
if __name__ == '__main__':
    # 配置
    USE_PARALLEL = True  # 设置为 True 启用多 GPU 并行
    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  # 要使用的 GPU ID 列表
```

### 配置选项

#### 并行模式 (推荐)
```python
USE_PARALLEL = True
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  # 使用 8 张卡
```

**优势：**
- 8 张卡并行测试 200 个样本
- 每张卡处理约 25 个样本
- 显著加快测试速度

#### 单 GPU 模式
```python
USE_PARALLEL = False
```

**适用场景：**
- 调试
- GPU 资源有限

## 工作原理

### 1. 任务分配
- 200 个测试样本平均分配到 8 张卡
- 每个进程独立运行，互不干扰
- GPU 0: 样本 0-24
- GPU 1: 样本 25-49
- GPU 2: 样本 50-74
- ...以此类推

### 2. KV Cache 文件锁
- 使用 `filelock` 防止多进程同时写入相同的 KV cache 文件
- 如果文件已存在，跳过重复生成
- 确保线程安全

### 3. 结果汇总
测试完成后，自动合并所有进程的结果：

**生成的文件：**
- `FusionRAG_global_topk_10_rate_0.3_process_0.csv` (GPU 0 的结果)
- `FusionRAG_global_topk_10_rate_0.3_process_1.csv` (GPU 1 的结果)
- ...
- `FusionRAG_global_topk_10_rate_0.3.csv` (合并后的最终结果)
- `FusionRAG_global_topk_10_rate_0.3.txt` (最终统计)

## 性能对比

| 模式 | GPU 数量 | 预计时间 | 加速比 |
|------|----------|----------|--------|
| 单卡 | 1 | 基准时间 | 1x |
| 并行 | 2 | ~50% | 2x |
| 并行 | 4 | ~25% | 4x |
| 并行 | 8 | ~12.5% | 8x |

*实际加速比取决于样本复杂度、模型大小和硬件性能*

## 自定义配置

### 使用不同的 GPU
```python
GPU_IDS = [0, 2, 4, 6]  # 只使用 4 张卡
```

### 调整样本数量
```python
common_params = dict(
    ...
    max_samples=500  # 测试 500 个样本
)
```

### 修改输出路径
```python
common_params = dict(
    ...
    cache_path='/your/custom/path/',
)
```

## 注意事项

1. **内存要求**：每个进程会加载一个完整的模型，确保有足够的 GPU 显存
2. **文件 I/O**：多个进程可能同时访问磁盘，建议使用高速 SSD
3. **OpenAI API 限流**：如果遇到 API 限流，可以减少并行 GPU 数量
4. **错误处理**：如果某个进程失败，其他进程会继续运行，最终合并时会跳过失败的进程

## 故障排查

### 问题 1: `ModuleNotFoundError: No module named 'filelock'`
**解决方案：**
```bash
pip install filelock
```

### 问题 2: CUDA Out of Memory
**解决方案：**
- 减少并行 GPU 数量
- 或者使用 `use_multi_gpu=True` 让单个进程使用模型并行

### 问题 3: 结果文件缺失
**检查：**
- 查看进程是否正常完成
- 检查日志中是否有错误信息
- 确认 `csv_path` 目录权限

## 示例输出

```
================================================================================
PARALLEL EXECUTION SETUP
================================================================================
Total samples: 200
Number of GPUs: 8
GPU IDs: [0, 1, 2, 3, 4, 5, 6, 7]
Samples per GPU: ~25
================================================================================

Starting process 0 on GPU 0: samples [0, 25)
Starting process 1 on GPU 1: samples [25, 50)
Starting process 2 on GPU 2: samples [50, 75)
...

Waiting for all 8 processes to complete...
✓ Process 0 completed
✓ Process 1 completed
...

================================================================================
MERGING RESULTS FROM ALL PROCESSES
================================================================================
✓ Merged FusionRAG_global_topk_10_rate_0.3_process_0.csv (25 rows)
✓ Merged FusionRAG_global_topk_10_rate_0.3_process_1.csv (25 rows)
...

================================================================================
MERGED FINAL RESULTS
================================================================================
Main Questions: 150/200 (75.00%)
Sub Questions: 180/200 (90.00%)
================================================================================
```

## 高级用法

### 编程接口

如果需要在代码中动态调用：

```python
from test_fusionrag_reflect import run_parallel, PreprocessScope

run_parallel(
    gpu_ids=[0, 1, 2, 3],
    model_type='qwen',
    model_path='/path/to/model',
    data_path='./result_reflect.json',
    cache_path='/path/to/cache',
    model_name='Qwen2.5-7B-Instruct',
    rate=0.3,
    topk=10,
    preprocess=True,
    preprocess_scope=PreprocessScope.GLOBAL,
    max_samples=200
)
```

## 贡献

如有问题或建议，请提交 Issue。

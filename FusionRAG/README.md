# 简介

## 数据集

数据集位于 `/mnt/data/ktransformers-dev/data/` 目录下，包含以下文件：
- `2wikimqa-200.jsonl` - 2WikiMQA 数据集（200个样本）
- `hotpotqa-260-100-10-doc.jsonl` - HotpotQA 数据集
- `musique-200.jsonl` - MuSiQue 数据集（200个样本）
- `triviaqa-270-100-10-doc.jsonl` - TriviaQA 数据集

每个样本包含多个与给定问题相关的文本段落。这些段落在数据加载时会被打乱顺序。

## 代码

在 `ktransformers` 目录下提供了以下脚本：

### 旧版脚本（模型特定）
- `process_cache.py` - 支持 Mistral 系列模型
- `qwen_process_cache.py` - 支持 Qwen 系列模型
- `pangu_process_cache.py` - 支持 PanGu 系列模型
- `llama_process_cache.py` - 支持 Llama 系列模型

### 统一脚本（推荐使用）
- `unified_process_cache.py` - **新的统一脚本，支持所有模型类型**（Mistral、Qwen、PanGu、Llama）

## 统一 Process Cache 参数说明

`unified_process_cache.py` 脚本为所有模型提供了统一的接口。以下是所有可用参数：

### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | str | `'mistral'` | 模型架构类型。可选值：`'mistral'`、`'pangu'`、`'qwen'`、`'llama'` |
| `model_path` | str | `'/mnt/data/models/Mistral-7B-Instruct-v0.3'` | 主模型路径 |
| `model_name` | str | `'Mistral-7B-Instruct-v0.3'` | 模型名称（用于日志和输出文件命名） |
| `draft_model_path` | str | `None` | draft 模型路径，用于 speculative_prefill 方法（可选，所有模型类型都可使用任意 draft 模型） |

### 数据配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_name` | str | `'musique-200.jsonl'` | 数据集文件名。可选值：`'musique-200.jsonl'`、`'2wikimqa-200.jsonl'`、`'hotpotqa-260-100-10-doc.jsonl'`、`'triviaqa-270-100-10-doc.jsonl'` |
| `data_path` | str | `'/mnt/data/ktransformers-dev/data/'` | 包含数据集文件的目录 |

### 缓存配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cache_path` | str | `'/mnt/data/processCache/'` | 存储缓存文件和结果的目录 |
| `max_cache_len` | int | `32768` | 最大缓存长度 |

### 重处理配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rate` | float | `0.2` | 重计算比例（0.0 = 无重计算，1.0 = 完全重计算） |
| `revert_rope` | bool | `False` | 是否还原 RoPE（旋转位置编码） |
| `reprocess_method` | str | `'cacheBlend'` | 重处理方法。可选值：<br>- `'processCache'`：查询引导的重处理（论文方法）<br>- `'cacheBlend'`：SOTA CacheBlend 基线<br>- `'Cache-Craft'`：SOTA Cache-Craft 基线<br>- `'speculative_prefill'`：使用 draft 模型的推测预填充<br>- `'frontRow'`：前排基线 |

### 预处理配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `preprocess` | bool | `True` | 是否应用 KVCache 预处理（论文中的相似度引导方法） |
| `topk` | int | `10` | 预处理阶段选择的相似段落数量（仅在 `preprocess=True` 时有效） |
| `bge_model_path` | str | `'/mnt/data/models/bge-m3-FP16'` | 用于相似度计算的 BGE 嵌入模型路径 |

## 使用示例

### 示例 1：Mistral + 查询引导的重处理
```python
from ktransformers.unified_process_cache import main

main(
    model_type='mistral',
    model_path='/mnt/data/models/Mistral-7B-Instruct-v0.3',
    model_name='Mistral-7B-Instruct-v0.3',
    data_name='musique-200.jsonl',
    rate=0.15,
    preprocess=True,
    topk=10,
    revert_rope=True,
    reprocess_method='processCache'
)
```

### 示例 2：Qwen + CacheBlend
```python
main(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    model_name='Qwen2.5-7B-Instruct',
    cache_path='/mnt/data3/processCache/',
    data_name='hotpotqa-260-100-10-doc.jsonl',
    rate=0.2,
    preprocess=False,
    reprocess_method='cacheBlend'
)
```

### 示例 3：PanGu + 推测预填充
```python
main(
    model_type='pangu',
    model_path='/mnt/data/models/openPangu-Embedded-1B-V1.1',
    model_name='openPangu-Embedded-1B-V1.1',
    draft_model_path='/mnt/data/models/Qwen2.5-1.5B-Instruct',  # 任意模型都可作为 draft 模型
    data_name='triviaqa-270-100-10-doc.jsonl',
    rate=0.1,
    reprocess_method='speculative_prefill'
)
```

### 示例 4：Llama + 完全缓存重计算
```python
main(
    model_type='llama',
    model_path='/mnt/data/model/Llama-3.1-8B-Instruct',
    model_name='Llama-3.1-8B-Instruct',
    data_name='2wikimqa-200.jsonl',
    rate=1.0,  # 完全重计算
    preprocess=False,
    reprocess_method='processCache'
)
```

## 输出文件

结果将以以下结构保存在 `cache_path` 目录下：
```
{cache_path}/
├── {data_name}/
│   └── {model_name}/
│       ├── reprocess_method_{method}_rate_{rate}_revert_rope_{rope}_topk_{topk}.csv
│       ├── reprocess_method_{method}_rate_{rate}_revert_rope_{rope}_topk_{topk}.txt
│       └── ...
└── data/
    ├── {data_name}/
    │   └── {model_name}/
    │       ├── {example_id}_{chunk_id}_key.pt
    │       ├── {example_id}_{chunk_id}_value.pt
    │       └── ...
    └── {data_name}-preprocess-{topk}-revert_rope-{rope}/
        └── {model_name}/
            └── ...
```

## 方法说明

### processCache（查询引导的重处理）
论文中描述的主要贡献。使用查询引导的注意力机制来选择重要的 KV 缓存条目进行重计算。

### cacheBlend
SOTA 基线方法，混合完全注意力和流式注意力的缓存。

### Cache-Craft
SOTA 基线方法，使用精心设计的缓存选择策略。

### speculative_prefill
使用较小的 draft 模型通过推测执行加速预填充。需要设置 `draft_model_path` 参数。

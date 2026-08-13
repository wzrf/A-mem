import pandas as pd
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from langchain_openai import ChatOpenAI
from dataclasses import dataclass, field
import typing as t
from ragas.callbacks import Callbacks
from ragas.metrics.base import MetricType
import json
import re
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import time

# 添加项目路径
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from transformers import AutoTokenizer, AutoConfig

# =========================================================================
# 【修改区域】: 使用 LangChain 包装器连接 OpenAI
# =========================================================================
# 使用你的 OpenAI API Key 替换
OPENAI_API_KEY = "xxxx"
OPENAI_MODEL = "deepseek-chat"  # 推荐使用 gpt-3.5-turbo 或 gpt-4o-mini 进行评估

print(f"Using OpenAI Model: {OPENAI_MODEL} for RAGAS evaluation.")

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        openai_api_base="https://api.deepseek.com",
        model=OPENAI_MODEL, 
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        max_retries=2,
        timeout=60
    )
)
# =========================================================================


def extract_retrieved_contexts_from_tokens(tokens_list, tokenizer):
    """
    从tokens_data中提取召回的文本块
    tokens_list格式: [system_tokens, passage1_tokens, passage2_tokens, ..., query_tokens]
    """
    # 中间的所有passage tokens (排除第一个system和最后一个query)
    passage_tokens = tokens_list[1:-1]
    
    retrieved_contexts = []
    for passage_token in passage_tokens:
        # 将tokens解码为文本
        passage_text = tokenizer.decode(passage_token, skip_special_tokens=True)
        
        # 移除"Passage N:"前缀
        if passage_text.startswith('Passage'):
            content = re.sub(r'^Passage \d+:\n', '', passage_text).strip()
        else:
            content = passage_text.strip()
        
        if content:
            retrieved_contexts.append(content)
    
    return retrieved_contexts

def prepare_evaluation_data(
    model_path='/mnt/data/models/Qwen2.5-14B-Instruct',
    data_name='2wikimqa-200.jsonl',
    data_path='/mnt/data/benchmark/data/',
    cache_path='/mnt/data/processCache/',
    model_name='Qwen2.5-14B-Instruct',
    rate=1,
    topk=10,
    revert_rope=True,
    preprocess=False,
    reprocess_method='cacheBlend'
):
    """
    准备评估数据，复用原始main函数的数据准备逻辑
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载配置
    prompt_config = json.load(open('/mnt/data/benchmark/config/dataset2prompt_few-shot.json'))
    data_path_full = data_path + data_name
    
    if data_name in ['2wikimqa.jsonl', 'samsum.jsonl', 'multi_news.jsonl', 'musique.jsonl', 'hotpotqa.jsonl', 'triviaqa.jsonl', 'bamboogle.jsonl']:
        data_name_prefix = data_name.split('.')[0]
    else:
        data_name_prefix = data_name.split('-')[0]
    
    # 读取数据
    data_file = open(data_path_full, 'r', encoding='utf-8')
    data = []
    for line in data_file.readlines():
        data.append(json.loads(line))
    
    # 处理数据格式
    if data_name_prefix in ['hotpotqa', 'triviaqa', 'bamboogle'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
        if data_name_prefix in ['hotpotqa', 'triviaqa']:
            data = data[0]
        for i in range(len(data)):
            import random
            random.seed(1)
            random.shuffle(data[i]['output'][0]['document'])
            data[i]['passage'] = data[i]['output'][0]['document']
    else:
        if data_name == 'samsum.jsonl':
            split_mark = 'Dialogue:'
        else:
            split_mark = 'Passage'
        for i in range(len(data)):
            data[i]['passage'] = re.findall(f'({split_mark} \\d+.*?)(?={split_mark} \\d+|$)', data[i]['context'], re.DOTALL)
        if data_name == 'musique-140.jsonl':
            for i in range(len(data)):
                data[i]['passage'] = re.findall(f'Passage \\d+:\\n(.*?)(?=Passage \\d+:|$)', data[i]['context'], re.DOTALL)
                data[i]['passage'] = ['\n\n' + text for text in data[i]['passage']]
                data[i]['passage'][-1] = data[i]['passage'][-1] + '\n'
    
    # 构建 prompt 和 tokens
    system_prompt = prompt_config['system_prompt'][model_name.split('-')[0]][data_name_prefix]
    query_task = prompt_config['query_prompt'][model_name.split('-')[0]][data_name_prefix]
    
    import torch
    system_tokens = torch.tensor(tokenizer.encode(system_prompt, add_special_tokens=False), dtype=torch.int)
    
    batch_tokens = []
    question_list = []
    
    N = len(data)
    for query_id, query in enumerate(data[:N]):
        query_prompt = query_task.format(input=data[query_id]['input'])
        query_tokens = torch.tensor(tokenizer.encode(query_prompt, add_special_tokens=False), dtype=torch.int)
        question_list.append(data[query_id]['input'])
        
        passage_tokens = [system_tokens]
        index = 0
        
        for bn in range(len(query['passage'])):
            if data_name_prefix in ['hotpotqa', 'triviaqa', 'bamboogle'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
                passage_text = f'Passage {index+1}:\n' + query['passage'][index] + '\n'
            else:
                passage_text = query['passage'][index] + '\n'
            
            passage_tokens.append(torch.tensor(tokenizer.encode(passage_text, add_special_tokens=False), dtype=torch.int))
            index += 1
            if index >= len(query['passage']):
                break
        
        passage_tokens.append(query_tokens)
        batch_tokens.append(passage_tokens)
    
    # 确定CSV文件路径
    csv_path = f"{cache_path}{data_name.split('.')[0]}/{model_name}"
    if preprocess:
        csv_file = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_topk_{topk}.csv"
    else:
        csv_file = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}.csv"
    
    # 返回 jsonl 数据总数
    return csv_file, batch_tokens, question_list, tokenizer, data_name_prefix, N

def load_samples_from_csv_and_tokens(csv_path, batch_tokens, tokenizer, expected_count):
    """
    从CSV和tokens数据构建评估样本
    确保CSV中的数据条数与jsonl一致
    """
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 检查数据条数
    csv_count = len(df)
    if csv_count != expected_count:
        print(f"Warning: CSV has {csv_count} rows but expected {expected_count} rows from jsonl")
        print(f"Will only process the minimum of the two: {min(csv_count, expected_count)}")
    
    # 使用最小值作为实际处理数量
    actual_count = min(csv_count, expected_count, len(batch_tokens))
    
    samples = []
    for idx in range(actual_count):
        row = df.iloc[idx]
        # 从tokens提取retrieved_contexts
        retrieved_contexts = extract_retrieved_contexts_from_tokens(batch_tokens[idx], tokenizer)
        
        sample = SingleTurnSample(
            user_input=row['Question'],
            response=str(row['Pred Answer']),  # 确保是字符串
            retrieved_contexts=retrieved_contexts
        )
        samples.append(sample)
    
    print(f"Loaded {len(samples)} samples (expected: {expected_count})")
    if expected_count != len(samples):
        # ⚠️ 注释掉，以便在调试时发现问题
        # raise ValueError
        pass
    return samples

async def evaluate_single_sample(faithfulness_metric, sample, idx, semaphore):
    """
    评估单个样本，使用信号量限制并发数
    """
    async with semaphore:
        try:
            score = await faithfulness_metric.single_turn_ascore(sample)
            return idx, score, None
        except Exception as e:
            # 打印详细错误信息有助于调试
            print(f"Sample {idx+1} failed with error: {e}", file=sys.stderr)
            return idx, 0.0, str(e)

async def evaluate_faithfulness_truly_concurrent(samples, faithfulness_metric, max_concurrent=20):
    """
    真正的并发评估 - 使用信号量控制并发数
    """
    total_samples = len(samples)
    scores = [0.0] * total_samples
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"Starting evaluation with max {max_concurrent} concurrent requests...")
    start_time = time.time()
    
    # 创建所有任务
    tasks = [
        evaluate_single_sample(faithfulness_metric, sample, idx, semaphore)
        for idx, sample in enumerate(samples)
    ]
    
    # 使用 as_completed 来实时显示进度
    completed = 0
    # 由于 scores 列表是预先分配的，需要一个列表来存储所有结果
    results = [None] * total_samples
    
    for coro in asyncio.as_completed(tasks):
        idx, score, error = await coro
        results[idx] = (score, error)
        completed += 1
        
        if error:
            print(f"[{completed}/{total_samples}] Sample {idx+1}: ERROR - {error}")
            scores[idx] = 0.0 # 错误时分数记为0
        else:
            print(f"[{completed}/{total_samples}] Sample {idx+1}: Score = {score:.4f}")
            scores[idx] = score
        
        # 每完成10个显示一次平均分
        if completed % 10 == 0:
            # 只计算已完成的有效分数
            valid_scores = [s for s in scores if s > 0 or s == 0 and s != 0.0] # 包含所有已处理的分数
            current_avg = sum(scores[:completed]) / completed
            elapsed = time.time() - start_time
            rate = completed / elapsed
            eta = (total_samples - completed) / rate if rate > 0 else 0
            print(f"  Progress: {completed}/{total_samples} ({completed/total_samples*100:.1f}%) | "
                  f"Avg: {current_avg:.4f} | "
                  f"Rate: {rate:.2f} samples/s | "
                  f"ETA: {eta:.0f}s")
    
    elapsed = time.time() - start_time
    # 最终结果是所有有效分数的列表
    final_scores = [r[0] for r in results if r is not None]

    print(f"\nCompleted all {total_samples} samples in {elapsed:.2f}s "
          f"({total_samples/elapsed:.2f} samples/s)")
    
    return final_scores

async def main_evaluation(
    model_path='/mnt/data/models/Qwen2.5-14B-Instruct',
    data_name='2wikimqa-200.jsonl',
    data_path='/mnt/data/benchmark/data/',
    cache_path='/mnt/data/processCache/',
    model_name='Qwen2.5-14B-Instruct',
    rate=1,
    topk=10,
    revert_rope=True,
    preprocess=False,
    reprocess_method='cacheBlend',
    max_concurrent=20  # 最大并发数
):
    """
    主评估函数
    """
    print("Preparing evaluation data...")
    csv_file, batch_tokens, question_list, tokenizer, data_name_prefix, expected_count = prepare_evaluation_data(
        model_path=model_path,
        data_name=data_name,
        data_path=data_path,
        cache_path=cache_path,
        model_name=model_name,
        rate=rate,
        topk=topk,
        revert_rope=revert_rope,
        preprocess=preprocess,
        reprocess_method=reprocess_method
    )
    
    # ========== 添加文件存在性检查 (RAGAS 评估结果) ==========
    summary_path = csv_file.replace('.csv', f'_faithfulness_{OPENAI_MODEL.replace(".", "_")}_summary.txt')
    # 可以根据需要决定是否启用结果检查跳过
    # if os.path.exists(summary_path):
    #     print(f"\n{'='*50}")
    #     print(f"⚠️  Summary file already exists, skipping evaluation:")
    #     print(f"   {summary_path}")
    #     # ... (读取并返回现有分数逻辑)
    #     print(f"{'='*50}\n")
    #     return None, None
    # =======================================================
    
    print(f"Expected {expected_count} samples from jsonl")
    print(f"Loading data from {csv_file}")
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return None, None
    
    samples = load_samples_from_csv_and_tokens(csv_file, batch_tokens, tokenizer, expected_count)
    
    # 直接使用 Faithfulness metric
    # evaluator_llm 在文件头部已经配置为 OpenAI
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    
    # 真正并发评估
    print(f"\nEvaluating faithfulness with max {max_concurrent} concurrent requests using {OPENAI_MODEL}...")
    scores = await evaluate_faithfulness_truly_concurrent(
        samples, faithfulness_metric, max_concurrent=max_concurrent
    )
    
    # 计算平均分数
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n{'='*50}")
    print(f"Average Faithfulness Score: {avg_score:.4f}")
    if scores:
        print(f"Min Score: {min(scores):.4f}")
        print(f"Max Score: {max(scores):.4f}")
    print(f"{'='*50}")
    
    # 保存结果
    output_path = csv_file.replace('.csv', f'_faithfulness_{OPENAI_MODEL.replace(".", "_")}.csv')
    results_df = pd.DataFrame({
        'Question': [s.user_input for s in samples],
        'Pred Answer': [s.response for s in samples],
        'Faithfulness Score': scores
    })
    # results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # 保存汇总统计
    with open(summary_path, 'w') as f:
        f.write(f"Evaluator Model: {OPENAI_MODEL}\n")
        f.write(f"Average Faithfulness Score: {avg_score:.4f}\n")
        if scores:
            f.write(f"Min Score: {min(scores):.4f}\n")
            f.write(f"Max Score: {max(scores):.4f}\n")
        f.write(f"Number of Samples: {len(scores)}\n")
        f.write(f"Expected Samples: {expected_count}\n")
    print(f"Summary saved to {summary_path}")
    
    return avg_score, scores

if __name__ == '__main__':
    # 确保 main_evaluation 仍然使用你的目标模型和数据集路径
    for dataset in ["musique-200.jsonl"]:

        for method in [ "FusionRAG"]:
        # for method in ["FusionRAG"]:
            avg_score, scores = asyncio.run(main_evaluation(
                model_path='/mnt/data/models/GLM-4-32B-0414',
                data_name=dataset,
                data_path='/mnt/data/benchmark/data/',
                cache_path='/mnt/data3/processCache/',
                model_name='GLM-4-32B-0414',
                rate=0.15,
                topk=10,
                revert_rope=True,
                preprocess=True,
                reprocess_method=method,
                max_concurrent=20  # 可以调高到30-50试试
            ))

            if avg_score is not None:
                print(f"\nFinal Average Faithfulness Score: {avg_score:.4f}")

        # # for method in ["CacheBlend", "Cache-Craft"]:
        # for method in ["FusionRAG"]:
        #     avg_score, scores = asyncio.run(main_evaluation(
        #         model_path='/mnt/data/models/GLM-4-32B-0414',
        #         data_name=dataset,
        #         data_path='/mnt/data/benchmark/data/',
        #         cache_path='/mnt/data3/processCache/',
        #         model_name='GLM-4-32B-0414',
        #         rate=0.15,
        #         topk=10,
        #         revert_rope=True,
        #         preprocess=True,
        #         reprocess_method=method,
        #         max_concurrent=20  # 可以调高到30-50试试
        #     ))

        #     if avg_score is not None:
        #         print(f"\nFinal Average Faithfulness Score: {avg_score:.4f}")

        # for method in ["FusionRAG"]:
        #     avg_score, scores = asyncio.run(main_evaluation(
        #         model_path='/mnt/data/models/GLM-4-32B-0414',
        #         data_name=dataset,
        #         data_path='/mnt/data/benchmark/data/',
        #         cache_path='/mnt/data3/processCache/',
        #         model_name='GLM-4-32B-0414',
        #         rate=1,
        #         topk=10,
        #         revert_rope=True,
        #         preprocess=True,
        #         reprocess_method=method,
        #         max_concurrent=20  # 可以调高到30-50试试
        #     ))

        #     if avg_score is not None:
        #         print(f"\nFinal Average Faithfulness Score: {avg_score:.4f}")
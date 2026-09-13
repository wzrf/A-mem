from pathlib import Path
import json
import re
import hashlib
import os
from openai import OpenAI
import concurrent.futures
from typing import Dict, Any, Optional
from dataclasses import dataclass
import threading
from tqdm import tqdm
import time

def run_dir(path: str, name: str):
    json_files = Path(path).glob("*.json")
    all_prompt_tokens = []
    all_completion_tokens = []
    for file in json_files:
        if name in file.name:
            with open(file, "r", encoding="utf-8") as f:
                datas = json.load(f)
                prompt_tokens = 0
                completion_tokens = 0
                for data in datas:
                    prompt_tokens += data["prompt_tokens"]
                    completion_tokens += data["completion_tokens"]

            if prompt_tokens > 0:
                all_prompt_tokens.append(prompt_tokens)
            if completion_tokens > 0:
                all_completion_tokens.append(completion_tokens)

    if len(all_prompt_tokens) > 0 and len(all_completion_tokens) > 0:
        print(f"path: {path}\n"
              f"name: {name}\n"
              f"average prompt_tokens: {sum(all_prompt_tokens)/len(all_prompt_tokens)} "
              f"average completion_tokens: {sum(all_completion_tokens)/len(all_completion_tokens)}")
        print("="*100)



import argparse
import json
from collections import defaultdict
from pathlib import Path
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import pandas as pd


# ============================================================================
# LLM Judge Functions
# ============================================================================

SYSTEM = """You are a strict, method-blind evaluator of question answering. Judge only whether the candidate answer is semantically correct
  according to the question and reference answer. Do not infer which system produced it."""

TEMPLATE = """Decide whether the candidate answer is correct.

  Rules:
  1. Accept concise paraphrases, equivalent names, equivalent date/number formats, and a correct answer embedded in harmless extra explanation.
  2. Reject a wrong person, entity, event, date, ordering, count, amount, or polarity; a contradiction; a refusal when the reference answers the question; or an answer missing a required list item, comparison, calculation, or event.
  3. Extra text is harmless only if it does not add a materially false answer claim.
  4. For open-ended preference or recommendation questions, the answer need not copy every example in the reference, but it must correctly use the core personal information required by the reference.
  5. Treat the reference as the scoring ground truth. Do not use outside knowledge.

  Question:
  {question}

  Reference answer:
  {reference}

  Candidate answer:
  {prediction}

  Do not REASON. JUST GIVE THE RESULT.
  Return exactly one JSON object with one boolean field and no other text:
  {{"correct": true}}
  or
  {{"correct": false}}"""

def text_sha256(text: str) -> str:
    """Return SHA256 hex digest of the input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

PROMPT_SHA256 = text_sha256(SYSTEM + "\n\0\n" + TEMPLATE)

def parse_correct(content: object) -> bool:
    text = str(content or "").strip()
    text = re.sub(r"^(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*$", "", text)
    try:
        value = json.loads(text)
    except Exception as e:
        print(f"parse error: {e} text: {text}")
        return False
    if not isinstance(value, dict) or set(value) != {"correct"} or not isinstance(value["correct"], bool):
        print(f"judge response is not strict correct:boolean JSON: {value}")
        return False
    return value["correct"]


class LLMJudgeCache:
    """Cache for LLM judge results to avoid redundant API calls."""

    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()  # 1. 引入线程锁
        if cache_file:
            self.load()

    def _make_key(self, question: str, reference: str, prediction: str) -> str:
        key_string = f"{question}|||{reference}|||{prediction}"
        return hashlib.sha256(key_string.encode("utf-8")).hexdigest()

    def get(self, question: str, reference: str, prediction: str) -> Optional[Dict[str, Any]]:
        key = self._make_key(question, reference, prediction)
        with self.lock:  # 2. 加锁读取
            return self.cache.get(key)

    def set(self, question: str, reference: str, prediction: str,
            correct: bool, usage: Dict[str, Any]) -> None:
        key = self._make_key(question, reference, prediction)
        with self.lock:  # 3. 加锁写入和保存，防止并发写和写入磁盘时的遍历冲突
            self.cache[key] = {
                "correct": correct,
                "usage": usage,
                "timestamp": os.times().user
            }
            if self.cache_file:
                self._save_unlocked()  # 避免死锁，在锁内部调用不加锁的 save

    def load(self) -> None:
        try:
            if self.cache_file and os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    with self.lock:
                        self.cache = json.load(f)
        except Exception as e:
            self.cache = {}
            print(f"Warning: Failed to load cache from {self.cache_file}: {e}")

    def _save_unlocked(self) -> None:
        """内部调用的保存逻辑（无锁版，由调用方加锁）"""
        try:
            dir_path = os.path.dirname(os.path.abspath(self.cache_file))
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                # 此时字典被 lock 保护，不会有其他线程在 dump 时修改 size
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache to {self.cache_file}: {e}")

    def save(self) -> None:
        """外部主动调用的保存逻辑"""
        with self.lock:
            self._save_unlocked()


def llm_judge_correctness(
    question: str,
    reference: str,
    prediction: str,
    model: str = "GLM-5.3",
    endpoint: str = "http://127.0.0.1:30002/v1",
    api_key: str = "sk-dummy",
    timeout: int = 30,
    max_tokens: int = 1024,
    cache: Optional[LLMJudgeCache] = None
) -> tuple[bool, dict]:
    """Call LLM to judge correctness of prediction against reference.

    Returns:
        (correct, usage_dict)
    """
    # Check cache first
    if cache is not None:
        cached = cache.get(question, reference, prediction)
        if cached is not None:
            # Return cached result
            return cached["correct"], cached["usage"]

    # Use default key if None or empty
    if not api_key:
        api_key = "sk-dummy"

    client = OpenAI(
        api_key=api_key,
        base_url=endpoint,
        timeout=timeout,
    )

    time_start = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(
                question=question,
                reference=reference,
                prediction=prediction
            )},
        ],
        max_tokens=max_tokens,
        stream=False,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}}
    )

    choice = completion.choices[0]
    message = choice.message
    content = message.content or ""

    if not content and hasattr(message, "reasoning_content"):
        content = message.reasoning_content or ""

    usage = completion.usage.model_dump() if completion.usage else {}

    print(f"request takes {time.time() - time_start} seconds, usage={usage}")
    time_start = time.time()
    correct = parse_correct(content)

    # Store in cache if cache is provided
    if cache is not None:
        cache.set(question, reference, prediction, correct, usage)
    print(f"save cache takes {time.time() - time_start} seconds")

    return correct, usage


def simple_tokenize(text: str) -> list[str]:
    """基于规则的简易文本分词与清洗"""
    text = str(text)
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def compute_f1(prediction: str, reference: str) -> float:
    """基于词重叠（Overlap Token Level）计算 F1 Score"""
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)

    if (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def calculate_bleu_scores(prediction: str, reference: str) -> dict[str, float]:
    """使用 NLTK 计算 BLEU 1-4 分数（带 Smoothing Method 1）"""
    try:
        pred_tokens = nltk.word_tokenize(str(prediction).lower())
        ref_tokens = [nltk.word_tokenize(str(reference).lower())]
    except Exception:
        pred_tokens = simple_tokenize(prediction)
        ref_tokens = [simple_tokenize(reference)]

    weights_list = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=weights,
                smoothing_function=smooth,
            )
        except Exception:
            score = 0.0
        scores[f"bleu{n}"] = score

    return scores


def extract_fields(item: dict) -> tuple[str, str, str, str, float | None, int, int]:
    """动态解析样本中的类别、问题、预测文本、参考文本与准确率判定"""
    # 提取 Category
    if "question_type" in item:
        category = str(item["question_type"])
    elif "category" in item:
        cat = item["category"]
        category = f"Category {cat}" if str(cat).isdigit() else str(cat)
    else:
        category = "Uncategorized"

    # 提取 Question
    question = item.get("question") or item.get("input") or item.get("query") or ""

    # 提取 Prediction / System Answer
    pred = item.get("system_answer") or item.get("prediction") or ""

    # 提取 Reference / Golden Answer
    ref = item.get("golden_answer") or item.get("original_answer") or item.get("reference") or ""
    prompt_tokens = item.get("prompt_tokens", 0)
    completion_tokens = item.get("completion_tokens", 0)

    # 提取 Correct / Accuracy 判定
    correct_val = item.get("correct")
    judge_score = None
    if correct_val is not None:
        if isinstance(correct_val, bool):
            judge_score = 1.0 if correct_val else 0.0
        else:
            judge_score = float(correct_val)

    return category, question, str(pred), str(ref), judge_score, prompt_tokens, completion_tokens


def evaluate_dataset(
    file_path: str,
    use_llm_judge: bool = True,
    llm_model: str = "GLM-5.3",
    llm_endpoint: str = "http://127.0.0.1:30002/v1",
    llm_api_key: str = "sk-dummy",
    llm_timeout: int = 30,
    llm_max_tokens: int = 1024,
    cache_file: Optional[str] = None,
    max_workers: int = 32,
) -> pd.DataFrame:
    """读取评测 JSON 文件并统一计算指标"""
    file_p = Path(file_path)
    if not file_p.exists():
        raise FileNotFoundError(f"文件未找到: {file_path}")

    with open(file_p, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Initialize cache if cache_file is provided
    cache = None
    if cache_file:
        cache = LLMJudgeCache(cache_file)

    metrics_by_category = defaultdict(list)

    all_prompt_tokens = []
    all_completion_tokens = []
    llm_prompt_tokens_all = []
    llm_completion_tokens_all = []

    # Prepare data structures for concurrent processing
    llm_tasks = []  # list of (index, cat, question, pred, ref, item)
    records = []  # list of records to be updated
    categories = []  # list of categories for each item

    # First pass: extract fields, compute basic metrics, collect LLM tasks
    for idx, item in enumerate(data_list):
        cat, question, pred, ref, judge_score, prompt_tokens, completion_tokens = extract_fields(item)
        if prompt_tokens != 0:
            all_prompt_tokens.append(prompt_tokens)
        if completion_tokens != 0:
            all_completion_tokens.append(completion_tokens)

        # Compute F1 and BLEU
        f1_val = compute_f1(pred, ref)
        bleu_dict = calculate_bleu_scores(pred, ref)

        # Create base record
        record = {
            "prompt_tokens": item.get("prompt_tokens"),
            "completion_tokens": item.get("completion_tokens"),
            "f1": f1_val,
            "bleu1": bleu_dict["bleu1"],
            "bleu2": bleu_dict["bleu2"],
            "bleu3": bleu_dict["bleu3"],
            "bleu4": bleu_dict["bleu4"],
        }

        # Preserve existing metrics
        if "metrics" in item and isinstance(item["metrics"], dict):
            for m_key, m_val in item["metrics"].items():
                if isinstance(m_val, (int, float)) and m_key not in record:
                    record[m_key] = m_val

        if judge_score is not None:
            record["accuracy"] = judge_score

        # If LLM judge is needed, add to task list
        if use_llm_judge and question:
            llm_tasks.append((idx, cat, question, pred, ref, item))
        else:
            # If no LLM judge needed, add to metrics directly
            metrics_by_category[cat].append(record)

        # Store record and category for later update
        records.append(record)
        categories.append(cat)

    # Concurrent LLM judging
    if use_llm_judge and llm_tasks:
        print(f"Processing {len(llm_tasks)} LLM judge tasks with {max_workers} workers...")

        def process_llm_task(task):
            idx, cat, question, pred, ref, item = task
            try:
                correct, llm_usage = llm_judge_correctness(
                    question=question,
                    reference=ref,
                    prediction=pred,
                    model=llm_model,
                    endpoint=llm_endpoint,
                    api_key=llm_api_key,
                    timeout=llm_timeout,
                    max_tokens=llm_max_tokens,
                    cache=cache,
                )
                judge_score = 1.0 if correct else 0.0
                return idx, cat, correct, llm_usage, judge_score, None
            except Exception as e:
                return idx, cat, False, {}, None, str(e)

        # Use ThreadPoolExecutor for concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in llm_tasks:
                future = executor.submit(process_llm_task, task)
                futures.append(future)

            # Collect results
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="LLM Judging",
                    unit="task"
            ):
                idx, cat, correct, llm_usage, judge_score, error = future.result()
                record = records[idx]

                if error:
                    print(f"LLM judge error for item {idx}: {error}")
                    # Keep original judge_score if any
                else:
                    record["llm_judge_correct"] = correct
                    if llm_usage:
                        llm_prompt = llm_usage.get("prompt_tokens", 0)
                        llm_completion = llm_usage.get("completion_tokens", 0)
                        record["llm_judge_prompt_tokens"] = llm_prompt
                        record["llm_judge_completion_tokens"] = llm_completion
                        if llm_prompt > 0:
                            llm_prompt_tokens_all.append(llm_prompt)
                        if llm_completion > 0:
                            llm_completion_tokens_all.append(llm_completion)

                    # Update accuracy based on LLM judge
                    if judge_score is not None:
                        record["accuracy"] = judge_score

                # Add to metrics_by_category (if not already added)
                metrics_by_category[cat].append(record)

    # For items without LLM judge, they were already added to metrics_by_category
    # Need to add those that were skipped (if any)
    # Actually, we already added non-LLM items in the first pass.
    # But we need to ensure all records are accounted for.
    # Let's just trust the logic above.

    # 汇总计算各 Group 均值
    summary_rows = []
    all_records = []

    question_prompt_tokens_avg = sum(all_prompt_tokens) / len(all_prompt_tokens)
    question_completion_tokens_avg = sum(all_completion_tokens) / len(all_completion_tokens)

    print(
        f"[QUESTION] Average Prompt Tokens    : {question_prompt_tokens_avg:.2f}\n"
        f"[QUESTION] Average Completion Tokens: {question_completion_tokens_avg:.2f}"
    )
    if use_llm_judge and llm_prompt_tokens_all:
        llm_prompt_avg = sum(llm_prompt_tokens_all) / len(llm_prompt_tokens_all)
        llm_completion_avg = sum(llm_completion_tokens_all) / len(llm_completion_tokens_all)
        print(
            f"[LLM JUDGE] Average Prompt Tokens    : {llm_prompt_avg:.2f}\n"
            f"[LLM JUDGE] Average Completion Tokens: {llm_completion_avg:.2f}"
        )

    for cat_key, records in metrics_by_category.items():
        all_records.extend(records)
        df_cat = pd.DataFrame(records)
        mean_dict = df_cat.mean().to_dict()
        mean_dict["category"] = cat_key
        mean_dict["count"] = len(records)
        summary_rows.append(mean_dict)

    # 计算 Overall 总平均
    df_all = pd.DataFrame(all_records)
    overall_dict = df_all.mean().to_dict()
    overall_dict["category"] = "OVERALL (Total Avg)"
    overall_dict["count"] = len(all_records)
    summary_rows.append(overall_dict)

    result_df = pd.DataFrame(summary_rows)

    # 整理列顺序
    primary_cols = ["category", "count", "f1", "bleu1", "bleu2", "bleu3", "bleu4"]
    if "accuracy" in result_df.columns:
        primary_cols.append("accuracy")
    if "prompt_tokens" in result_df.columns:
        primary_cols.append("prompt_tokens")
    if "completion_tokens" in result_df.columns:
        primary_cols.append("completion_tokens")
    if "llm_judge_prompt_tokens" in result_df.columns:
        primary_cols.append("llm_judge_prompt_tokens")
    if "llm_judge_completion_tokens" in result_df.columns:
        primary_cols.append("llm_judge_completion_tokens")
    if "llm_judge_correct" in result_df.columns:
        primary_cols.append("llm_judge_correct")

    other_cols = [c for c in result_df.columns if c not in primary_cols]
    final_cols = primary_cols + other_cols

    return result_df[final_cols]


def analyze_halumem_robust(folder_path: str = "./results_halumem_robust") -> pd.DataFrame:
    """分析 halumem_robust 结果文件夹中的 JSON 文件，按问题类型统计 token 消耗和准确率指标。"""
    import json
    from pathlib import Path
    import pandas as pd
    from collections import defaultdict

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    json_files = list(folder.glob("*.json"))
    if not json_files:
        raise ValueError(f"在文件夹中没有找到 JSON 文件: {folder_path}")

    all_records = []
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        for item in data_list:
            # 提取 token 字段
            turn_build_memory_prompt_tokens = item.get("turn_build_memory_prompt_tokens", 0)
            turn_build_memory_completion_tokens = item.get("turn_build_memory_completion_tokens", 0)
            answer_prompt_tokens = item.get("answer_prompt_tokens", 0)
            answer_completion_tokens = item.get("answer_completion_tokens", 0)
            # 提取问题类型
            question_type = item.get("question_type", "Uncategorized")
            # 提取指标
            metrics = item.get("metrics", {})
            f1 = metrics.get("f1", 0.0)
            exact_match = metrics.get("exact_match", 0.0)
            rouge1_f = metrics.get("rouge1_f", 0.0)
            rouge2_f = metrics.get("rouge2_f", 0.0)
            rougeL_f = metrics.get("rougeL_f", 0.0)
            bleu1 = metrics.get("bleu1", 0.0)
            bleu2 = metrics.get("bleu2", 0.0)
            bleu3 = metrics.get("bleu3", 0.0)
            bleu4 = metrics.get("bleu4", 0.0)
            sbert_similarity = metrics.get("sbert_similarity", 0.0)
            meteor = metrics.get("meteor", 0.0)

            record = {
                "question_type": question_type,
                "turn_build_memory_prompt_tokens": turn_build_memory_prompt_tokens,
                "turn_build_memory_completion_tokens": turn_build_memory_completion_tokens,
                "answer_prompt_tokens": answer_prompt_tokens,
                "answer_completion_tokens": answer_completion_tokens,
                "f1": f1,
                "exact_match": exact_match,
                "rouge1_f": rouge1_f,
                "rouge2_f": rouge2_f,
                "rougeL_f": rougeL_f,
                "bleu1": bleu1,
                "bleu2": bleu2,
                "bleu3": bleu3,
                "bleu4": bleu4,
                "sbert_similarity": sbert_similarity,
                "meteor": meteor,
            }
            all_records.append(record)

    df = pd.DataFrame(all_records)
    if df.empty:
        print("没有找到有效数据。")
        return pd.DataFrame()

    # 按问题类型分组计算平均值
    grouped = df.groupby("question_type").agg({
        "turn_build_memory_prompt_tokens": "mean",
        "turn_build_memory_completion_tokens": "mean",
        "answer_prompt_tokens": "mean",
        "answer_completion_tokens": "mean",
        "f1": "mean",
        "exact_match": "mean",
        "rouge1_f": "mean",
        "rouge2_f": "mean",
        "rougeL_f": "mean",
        "bleu1": "mean",
        "bleu2": "mean",
        "bleu3": "mean",
        "bleu4": "mean",
        "sbert_similarity": "mean",
        "meteor": "mean",
    }).round(4)
    grouped["count"] = df.groupby("question_type").size()

    # 重置索引，将 question_type 变为列
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={"question_type": "category"})

    # 计算总体平均值
    overall = df.mean(numeric_only=True).to_dict()
    overall["category"] = "OVERALL (Total Avg)"
    overall["count"] = len(df)
    overall_df = pd.DataFrame([overall])
    # 确保列顺序一致
    overall_df = overall_df[grouped.columns]

    # 合并
    result_df = pd.concat([grouped, overall_df], ignore_index=True)

    # 整理列顺序，将类别和计数放在前面
    cols = ["category", "count"]
    token_cols = ["turn_build_memory_prompt_tokens", "turn_build_memory_completion_tokens",
                  "answer_prompt_tokens", "answer_completion_tokens"]
    metric_cols = ["f1", "exact_match", "rouge1_f", "rouge2_f", "rougeL_f",
                   "bleu1", "bleu2", "bleu3", "bleu4", "sbert_similarity", "meteor"]
    # 只保留存在的列
    existing_cols = []
    for col in cols + token_cols + metric_cols:
        if col in result_df.columns:
            existing_cols.append(col)
    # 添加其他列（如果有）
    other_cols = [c for c in result_df.columns if c not in existing_cols]
    final_cols = existing_cols + other_cols
    result_df = result_df[final_cols]

    return result_df



def main():
    parser = argparse.ArgumentParser(description="使用基准对齐的 F1/BLEU 逻辑计算评测统计")
    parser.add_argument("--csv", type=str, default=None, help="保存统计结果 CSV 路径")
    parser.add_argument("--halumem_robust", action="store_true", help="分析 results_halumem_robust 文件夹")
    parser.add_argument("--folder", type=str, default="./results_halumem_robust", help="halumem_robust 文件夹路径（默认：./results_halumem_robust）")
    parser.add_argument("--run_dir", action="store_true", help="运行旧版 run_dir 分析 token_consumption 文件夹")
    parser.add_argument("--llm_model", type=str, default="GLM-5.3", help="LLM模型名称（默认: GLM-5.3）")
    parser.add_argument("--llm_endpoint", type=str, default="http://127.0.0.1:30002/v1", help="LLM API端点（默认: http://127.0.0.1:30002/v1）")
    parser.add_argument("--llm_api_key", type=str, default="sk-dummy", help="LLM API密钥（默认: sk-dummy）")
    parser.add_argument("--llm_timeout", type=int, default=30, help="LLM API超时秒数（默认: 30）")
    parser.add_argument("--llm_max_tokens", type=int, default=2048, help="LLM最大生成token数（默认: 1024）")
    parser.add_argument("--cache_file", type=str, default="./.llm_judge_cache.json", help="缓存文件路径，用于避免重复LLM调用")
    parser.add_argument("--max_workers", type=int, default=32, help="并发LLM判断的最大线程数（默认: 32）")

    args = parser.parse_args()

    # 在这里配置要依次评测的文件
    file_paths = [
        "results/result__0.3_Qwen2.5-3B-Instruct_qwen2.5-7B_retrieve_10.json", ## qwen+locomo
        "results_GLM_4.5_air/retrieve_10.json", ## glm+locomo
        "results_Kimi_k26/retrieve_10.json", ## kimi+locomo

        "results/result_longmemeval_qwen3-8b.json"
    ]

    # 在这里配置要依次运行的目录和对应 name
    run_dirs = [
        ("./token_consumption_qwen3-8b", "locomo"),
        ("./token_consumption_qwen3-8b", "longmemeval"),

        ("./token_consumption_GLM-4.5-Air", "locomo"),
        ("./token_consumption_GLM-4.5-Air", "longmemeval"),

        ("./token_consumption_Kimi-K2.6", "locomo"),
        ("./token_consumption_Kimi-K2.6", "longmemeval"),
    ]

    # 确保 NLTK punkt 分词数据可用
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    if args.halumem_robust:
        summary_df = analyze_halumem_robust(args.folder)
        print("\n" + "=" * 104)
        print(f" HALUMEM Robust 统计表: {args.folder}")
        print("=" * 104)
        print(summary_df.round(4).to_string(index=False))
        print("=" * 104 + "\n")
        if args.csv:
            summary_df.round(6).to_csv(args.csv, index=False)
            print(f"结果已成功导出至: {args.csv}")
        return

    for file_path in file_paths:
        summary_df = evaluate_dataset(
            file_path,
            llm_model=args.llm_model,
            llm_endpoint=args.llm_endpoint,
            llm_api_key=args.llm_api_key,
            llm_timeout=args.llm_timeout,
            llm_max_tokens=args.llm_max_tokens,
            cache_file=args.cache_file,
            max_workers=args.max_workers,
        )
        print("\n" + "=" * 90)
        print(f" 评测指标对齐统计表: {file_path}")
        print("=" * 90)
        print(summary_df.round(4).to_string(index=False))
        print("=" * 90 + "\n")
        if args.csv:
            summary_df.round(6).to_csv(args.csv, index=False)
            print(f"结果已成功导出至: {args.csv}")

    for run_path, name in run_dirs:
        run_dir(run_path, name)


if __name__ == "__main__":
    main()
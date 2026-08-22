from pathlib import Path
import json


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

    print(f"{name}: average prompt_tokens: {sum(all_prompt_tokens)/len(all_prompt_tokens)} "
          f"average completion_tokens: {sum(all_completion_tokens)/len(all_completion_tokens)}")


run_dir("./token_consumption", "locomo")
run_dir("./token_consumption", "longmemeval")

import argparse
import json
from collections import defaultdict
from pathlib import Path
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import pandas as pd


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


def extract_fields(item: dict) -> tuple[str, str, str, float | None]:
    """动态解析样本中的类别、预测文本、参考文本与准确率判定"""
    # 提取 Category
    if "question_type" in item:
        category = str(item["question_type"])
    elif "category" in item:
        cat = item["category"]
        category = f"Category {cat}" if str(cat).isdigit() else str(cat)
    else:
        category = "Uncategorized"

    # 提取 Prediction / System Answer
    pred = item.get("system_answer") or item.get("prediction") or ""

    # 提取 Reference / Golden Answer
    ref = item.get("golden_answer") or item.get("original_answer") or item.get("reference") or ""

    # 提取 Correct / Accuracy 判定
    correct_val = item.get("correct")
    judge_score = None
    if correct_val is not None:
        if isinstance(correct_val, bool):
            judge_score = 1.0 if correct_val else 0.0
        else:
            judge_score = float(correct_val)

    return category, str(pred), str(ref), judge_score


def evaluate_dataset(file_path: str) -> pd.DataFrame:
    """读取评测 JSON 文件并统一计算指标"""
    file_p = Path(file_path)
    if not file_p.exists():
        raise FileNotFoundError(f"文件未找到: {file_path}")

    with open(file_p, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    metrics_by_category = defaultdict(list)

    for item in data_list:
        cat, pred, ref, judge_score = extract_fields(item)

        # 实时按对齐的方法计算 F1 和 BLEU 1-4
        f1_val = compute_f1(pred, ref)
        bleu_dict = calculate_bleu_scores(pred, ref)

        record = {
            "prompt_tokens": item.get("prompt_tokens"),
            "completion_tokens": item.get("completion_tokens"),
            "f1": f1_val,
            "bleu1": bleu_dict["bleu1"],
            "bleu2": bleu_dict["bleu2"],
            "bleu3": bleu_dict["bleu3"],
            "bleu4": bleu_dict["bleu4"],
        }

        # 保留已有的其他评测指标（如 exact_match, sbert_similarity 等）
        if "metrics" in item and isinstance(item["metrics"], dict):
            for m_key, m_val in item["metrics"].items():
                if isinstance(m_val, (int, float)) and m_key not in record:
                    record[m_key] = m_val

        if judge_score is not None:
            record["accuracy"] = judge_score

        metrics_by_category[cat].append(record)

    # 汇总计算各 Group 均值
    summary_rows = []
    all_records = []

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

    other_cols = [c for c in result_df.columns if c not in primary_cols]
    final_cols = primary_cols + other_cols

    return result_df[final_cols]


def main():
    parser = argparse.ArgumentParser(description="使用基准对齐的 F1/BLEU 逻辑计算评测统计")
    parser.add_argument("--file_path", type=str, help="评估 JSON 结果文件路径")
    parser.add_argument("--csv", type=str, default=None, help="保存统计结果 CSV 路径")

    args = parser.parse_args()

    # 确保 NLTK punkt 分词数据可用
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    summary_df = evaluate_dataset(args.file_path)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n" + "=" * 90)
    print(f" 评测指标对齐统计表: {args.file_path}")
    print("=" * 90)
    print(summary_df.round(4).to_string(index=False))
    print("=" * 90 + "\n")

    if args.csv:
        summary_df.round(6).to_csv(args.csv, index=False)
        print(f"结果已成功导出至: {args.csv}")


if __name__ == "__main__":
    main()
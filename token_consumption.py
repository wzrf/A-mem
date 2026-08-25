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


def extract_fields(item: dict) -> tuple[str, str, str, float, int, int | None]:
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

    return category, str(pred), str(ref), judge_score, prompt_tokens, completion_tokens


def evaluate_dataset(file_path: str) -> pd.DataFrame:
    """读取评测 JSON 文件并统一计算指标"""
    file_p = Path(file_path)
    if not file_p.exists():
        raise FileNotFoundError(f"文件未找到: {file_path}")

    with open(file_p, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    metrics_by_category = defaultdict(list)

    all_prompt_tokens = []
    all_completion_tokens = []
    for item in data_list:
        cat, pred, ref, judge_score, prompt_tokens, completion_tokens = extract_fields(item)
        if prompt_tokens != 0:
            all_prompt_tokens.append(prompt_tokens)
        if completion_tokens != 0:
            all_completion_tokens.append(completion_tokens)

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

    question_prompt_tokens_avg = sum(all_prompt_tokens) / len(all_prompt_tokens)
    question_completion_tokens_avg = sum(all_completion_tokens) / len(all_completion_tokens)

    print(
        f"[QUESTION] Average Prompt Tokens    : {question_prompt_tokens_avg:.2f}\n"
        f"[QUESTION] Average Completion Tokens: {question_completion_tokens_avg:.2f}"
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
    parser.add_argument("--file_path", type=str, help="评估 JSON 结果文件路径")
    parser.add_argument("--csv", type=str, default=None, help="保存统计结果 CSV 路径")
    parser.add_argument("--halumem_robust", action="store_true", help="分析 results_halumem_robust 文件夹")
    parser.add_argument("--folder", type=str, default="./results_halumem_robust", help="halumem_robust 文件夹路径（默认：./results_halumem_robust）")
    parser.add_argument("--run_dir", action="store_true", help="运行旧版 run_dir 分析 token_consumption 文件夹")

    args = parser.parse_args()

    # 确保 NLTK punkt 分词数据可用
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    if args.halumem_robust:
        summary_df = analyze_halumem_robust(args.folder)
        print("\n" + "=" *104)
        print(f" HALUMEM Robust 统计表: {args.folder}")
        print("=" *104)
        print(summary_df.round(4).to_string(index=False))
        print("=" *104 + "\n")
        if args.csv:
            summary_df.round(6).to_csv(args.csv, index=False)
            print(f"结果已成功导出至: {args.csv}")
        return

    if args.file_path:
        summary_df = evaluate_dataset(args.file_path)
        print("\n" + "=" * 90)
        print(f" 评测指标对齐统计表: {args.file_path}")
        print("=" * 90)
        print(summary_df.round(4).to_string(index=False))
        print("=" * 90 + "\n")
        if args.csv:
            summary_df.round(6).to_csv(args.csv, index=False)
            print(f"结果已成功导出至: {args.csv}")
        return

    if args.run_dir:
        # 旧版 token_consumption 文件夹分析
        run_dir("./token_consumption", "locomo")
        run_dir("./token_consumption", "longmemeval")
        return

    # 默认行为：如果没有提供任何参数，则运行旧版分析（保持向后兼容）
    print("未指定参数，运行默认 token_consumption 分析...")
    run_dir("./token_consumption", "locomo")
    run_dir("./token_consumption", "longmemeval")


if __name__ == "__main__":
    main()
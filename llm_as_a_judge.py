import json
import os, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. 配置 阿里云 DashScope API (兼容 OpenAI 接口)
# ---------------------------------------------------------------------------
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-11ce7640e46049a6977c0d96ba855ffb")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3"  # 可选: deepseek-v3 或 deepseek-r1

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ---------------------------------------------------------------------------
# 2. Prompt 模版与 LLM 判定逻辑
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert grader that determines if an answer to a question matches the gold standard reference answer.
Compare the model's prediction with the reference answer.
You must return a valid JSON object with the key "label" which is strictly either "correct" or "incorrect", and "reason" for a brief explanation.

Example Output:
{"label": "correct", "reason": "The predicted date matches the reference date."}
"""

JUDGE_PROMPT_TEMPLATE = """[Question]: {question}
[Reference Answer]: {reference}
[Model Prediction]: {prediction}

Based on the question and reference answer, is the model's prediction correct? Respond ONLY with JSON format."""


def judge_single_item(item: Dict[str, Any], max_retries: int = 3) -> bool:
    """调用 DeepSeek 评估单条数据的准确度"""
    question = item.get("question", "")
    reference = item.get("reference", "")
    prediction = item.get("prediction", "")

    if not str(prediction).strip():
        return False

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference=reference,
        prediction=prediction
    )

    for attempt in range(max_retries):
        content = ""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            if "```" in content:
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
            parsed = json.loads(content)
            label = str(parsed.get("label", "")).strip().lower()

            return label == "correct"

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[Warning] API call failed for sample_id {item.get('sample_id')}: {e} content={content}")
                return False


# ---------------------------------------------------------------------------
# 3. 按文件和分类评估逻辑（并发数默认 8）
# ---------------------------------------------------------------------------
def evaluate_file(file_path: str, max_workers: int = 8) -> Dict[Any, Dict[str, float]]:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return {}

    print(f"\n>>> Loading and evaluating: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_data = defaultdict(list)
    for item in data:
        cat = item.get("category", "Unknown")
        category_data[cat].append(item)

    results_by_category = {}

    for cat, items in category_data.items():
        print(f"Evaluating Category [{cat}] ({len(items)} samples)...")
        correct_count = 0
        total_count = len(items)

        # 设置 8 个线程并发
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(judge_single_item, item): item for item in items}

            for future in tqdm(as_completed(future_to_item), total=total_count, desc=f"Cat {cat}"):
                if future.result():
                    correct_count += 1

        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
        results_by_category[cat] = {
            "correct": correct_count,
            "total": total_count,
            "accuracy": accuracy
        }

    return results_by_category


# ---------------------------------------------------------------------------
# 4. 打印格式化报告
# ---------------------------------------------------------------------------
def print_report(file_name: str, results: Dict[Any, Dict[str, float]]):
    print("\n" + "=" * 55)
    print(f" EVALUATION REPORT: {file_name}")
    print("=" * 55)
    print(f"{'Category':<15} | {'Correct':<8} | {'Total':<8} | {'Accuracy (%)':<12}")
    print("-" * 55)

    total_correct = 0
    total_samples = 0

    sorted_cats = sorted(results.keys(), key=lambda x: str(x))
    for cat in sorted_cats:
        stats = results[cat]
        correct = stats["correct"]
        total = stats["total"]
        acc = stats["accuracy"]

        total_correct += correct
        total_samples += total

        print(f"{str(cat):<15} | {correct:<8} | {total:<8} | {acc:.2f}%")

    print("-" * 55)
    overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    print(f"{'OVERALL':<15} | {total_correct:<8} | {total_samples:<8} | {overall_acc:.2f}%")
    print("=" * 55)


# ---------------------------------------------------------------------------
# 5. 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    FILE_1 = "./results/result_simplerag_fusion_rag_1.0_Qwen2.5-3B-Instruct_qwen2.5-7B_retrieve_5.json"
    FILE_2 = "./results/result_fusion_rag_1.0_Qwen2.5-3B-Instruct_qwen2.5-7B_retrieve_5.json"

    # 并发数设为 8
    MAX_WORKERS = 8

    res1 = evaluate_file(FILE_1, max_workers=MAX_WORKERS)
    print_report(FILE_1, res1)

    res2 = evaluate_file(FILE_2, max_workers=MAX_WORKERS)
    print_report(FILE_2, res2)
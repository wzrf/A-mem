import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# 复用 test_advanced_robust 中的核心模块
from test_advanced_robust import (
    RobustAdvancedMemAgent,
    parse_plain_text_answer,
    calculate_metrics,
    aggregate_metrics
)

# ============================================================================
# HaluMem 数据结构与 JSONL 加载器
# ============================================================================

@dataclass
class HaluMemSample:
    sample_id: str
    persona_info: str
    sessions: List[Dict]


def load_halumem_dataset(path_input: Union[str, Path]) -> List[HaluMemSample]:
    """
    加载 HaluMem 数据集 (.jsonl 格式或包含 jsonl 的目录)
    """
    path_input = Path(path_input)
    if not path_input.exists():
        raise FileNotFoundError(f"Path not found at {path_input}")

    files = []
    if path_input.is_dir():
        files = sorted(list(path_input.glob("*.jsonl")))
    else:
        files = [path_input]

    samples = []
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                sid = data.get("uuid", f"{fpath.stem}_{line_idx}")
                persona = data.get("persona_info", "")
                sessions = data.get("sessions", [])
                samples.append(HaluMemSample(sample_id=str(sid), persona_info=persona, sessions=sessions))

    print(f"Successfully loaded {len(samples)} samples from {path_input}")
    return samples


# ============================================================================
# HaluMem 增量测试器
# ============================================================================

class HaluMemRobustTester:
    def __init__(self, model: str, backend: str, retrieve_k: int, temperature_c5: float,
                 sglang_host: str, sglang_port: int):
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port

    def _get_expected_question_ids(self, sample: HaluMemSample) -> set:
        """获取样本中所有问题的预期 question_id 集合。"""
        safe_sample_id = sample.sample_id.replace(" ", "_")
        expected_ids = set()
        for s_idx, session in enumerate(sample.sessions):
            questions = session.get("questions", [])
            for q_idx, _ in enumerate(questions):
                question_id = f"{safe_sample_id}_s{s_idx}_q{q_idx}"
                expected_ids.add(question_id)
        return expected_ids

    def _get_existing_question_ids(self, file_path: str) -> set:
        """从结果文件中读取已有的 question_id 集合。"""
        if not os.path.exists(file_path):
            return set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                return set()
            existing_ids = set()
            for item in data:
                if 'question_id' in item:
                    existing_ids.add(item['question_id'])
            return existing_ids
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            return set()

    def run_single_sample(self, sample: HaluMemSample, sample_idx: int, save_dir: str = "./results_halumem_robust"):
        """
        处理单个 HaluMem 样本：
        按 Session 顺序执行：
          1. 将当前 Session 的 Dialogue 加入 Memory，并统计增加 Dialogues 的 Token 开销。
          2. 对当前 Session 的 Questions 进行检索与回答，统计 QA 的 Token 开销。
        """
        safe_sample_id = sample.sample_id.replace(" ", "_")
        token_consumption_file = f"./token_consumption/halumem_{safe_sample_id}.json"

        # 检查是否已经处理了所有问题
        result_file = os.path.join(save_dir, f"{safe_sample_id}.json")
        expected_ids = self._get_expected_question_ids(sample)
        existing_ids = self._get_existing_question_ids(result_file)

        if expected_ids.issubset(existing_ids):
            print(f"[{sample_idx}] Sample {safe_sample_id} 已完全处理，跳过。")
            # 读取现有结果并返回
            with open(result_file, 'r', encoding='utf-8') as f:
                sample_results = json.load(f)
            avg_f1 = sum(r['metrics'].get('f1', 0) for r in sample_results) / len(sample_results) if sample_results else 0
            print(f"[{sample_idx}] Sample ID: {safe_sample_id} | Total Queries: {len(sample_results)} | Avg F1: {avg_f1:.3f} (已存在)")
            return sample_results

        # 为当前 Sample 初始化专属的 Memory Agent 实例
        agent = RobustAdvancedMemAgent(
            model=self.model,
            backend=self.backend,
            retrieve_k=self.retrieve_k,
            temperature_c5=self.temperature_c5,
            sglang_host=self.sglang_host,
            sglang_port=self.sglang_port,
            token_consumption_file=token_consumption_file
        )

        sample_results = []
        accumulated_history_turns = 0
        total_build_prompt_tokens = 0
        total_build_completion_tokens = 0

        # 遍历每个 Session
        for s_idx, session in enumerate(sample.sessions):
            print(f"running sample {sample_idx} session {s_idx}/{len(sample.sessions)}")
            dialogue_turns = session.get("dialogue", session.get("messages", []))
            questions = session.get("questions", [])

            # ----------------------------------------------------------------
            # 步骤 1：将当前 Session 对话插入 Agent 记忆，统计 Build Token 消耗
            # ----------------------------------------------------------------

            session_build_prompt_tokens = 0
            session_build_completion_tokens = 0
            for turn_idx, turn in enumerate(dialogue_turns):
                speaker = turn.get("speaker", turn.get("role", "user"))
                text = turn.get("content", "")
                ts = turn.get("timestamp", turn.get("time_stamp", None))
                prompt_tokens, completion_tokens = agent.add_memory(f"Speaker {speaker} says : {text}", time=ts)
                session_build_prompt_tokens += prompt_tokens
                session_build_completion_tokens += completion_tokens

            accumulated_history_turns += len(dialogue_turns)


            total_build_prompt_tokens += session_build_prompt_tokens
            total_build_completion_tokens += session_build_completion_tokens

            # ----------------------------------------------------------------
            # 步骤 2：记忆更新完成后，回答当前 Session 内的问题
            # ----------------------------------------------------------------
            for q_idx, q_item in enumerate(questions):
                question = q_item.get("question", "")
                reference = str(q_item.get("answer", ""))
                question_type = q_item.get("question_type", "default")
                question_id = f"{safe_sample_id}_s{s_idx}_q{q_idx}"

                # 2.1 执行 Answer (Agent 内部完成了检索与 Prompt 生成)
                qa_start = time.time()
                prediction, user_prompt, raw_context, raw_context_list, answer_p_tokens, answer_c_tokens, query_prompt = agent.answer_question(
                    question, category=1, answer=reference
                )
                qa_time = time.time() - qa_start

                # 清洗模型输出并计算评估指标
                parsed_pred = parse_plain_text_answer(prediction)
                metrics = calculate_metrics(parsed_pred, reference) if reference else {}

                query_res = {
                    'sample_id': safe_sample_id,
                    'question_id': question_id,
                    'session_index': s_idx,
                    'question_type': question_type,
                    'difficulty': q_item.get("difficulty", "normal"),
                    'question': question,
                    'answer': parsed_pred,
                    'raw_prediction': prediction,
                    'reference': reference,

                    # 1. 本轮及累计增加 dialogues 的 memory token 消耗
                    'session_dialogue_turns_added': len(dialogue_turns),
                    'turn_build_memory_prompt_tokens': session_build_prompt_tokens,
                    'turn_build_memory_completion_tokens': session_build_completion_tokens,
                    'turn_build_memory_total_tokens': session_build_prompt_tokens + session_build_completion_tokens,

                    'accumulated_history_turns': accumulated_history_turns,
                    'total_build_memory_prompt_tokens': total_build_prompt_tokens,
                    'total_build_memory_completion_tokens': total_build_completion_tokens,

                    # 2. 回答生成阶段 Token 开销
                    'answer_prompt_tokens': answer_p_tokens,
                    'answer_completion_tokens': answer_c_tokens,
                    'answer_total_tokens': answer_p_tokens + answer_c_tokens,

                    # 3. QA 环节总 Token 消耗
                    'query_total_prompt_tokens': answer_p_tokens,
                    'query_total_completion_tokens': answer_c_tokens,
                    'query_total_tokens': answer_p_tokens + answer_c_tokens,

                    'total_time': qa_time,
                    'metrics': metrics
                }
                sample_results.append(query_res)

            # 保存单个 sample 结果
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/{safe_sample_id}.json", 'w', encoding='utf-8') as f:
                json.dump(sample_results, f, indent=2, ensure_ascii=False)

        avg_f1 = sum(r['metrics'].get('f1', 0) for r in sample_results) / len(sample_results) if sample_results else 0
        print(f"[{sample_idx}] Sample ID: {safe_sample_id} | Total Queries: {len(sample_results)} | Avg F1: {avg_f1:.3f}")
        return sample_results


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Robust Memory Agent on HaluMem dataset")
    parser.add_argument("--dataset", type=str, default="data/HaluMem-Medium.jsonl",
                        help="Path to HaluMem jsonl file or directory")
    parser.add_argument("--model", type=str, default="qwen3-8b")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--output-dir", type=str, default="./results_halumem_robust",
                        help="Directory to save evaluation results")
    args = parser.parse_args()

    # 1. 加载 HaluMem 数据集
    samples = load_halumem_dataset(args.dataset)

    max_workers = 16
    if os.environ.get('DEBUG') == "1":
        max_workers = 1

    # 工作线程函数
    def _worker(idx_sample):
        idx, sample = idx_sample
        tester = HaluMemRobustTester(
            model=args.model,
            backend=args.backend,
            retrieve_k=args.retrieve_k,
            temperature_c5=args.temperature_c5,
            sglang_host=args.sglang_host,
            sglang_port=args.sglang_port
        )
        return tester.run_single_sample(sample, idx, save_dir=args.output_dir)

    all_flattened_results = []
    print(f"Starting evaluation across {len(samples)} samples with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, (i, s)) for i, s in enumerate(samples)]
        for future in as_completed(futures):
            try:
                sample_res = future.result()
                all_flattened_results.extend(sample_res)
            except Exception as e:
                import traceback
                print(f"Sample execution failed: {e}")
                traceback.print_exc()

    # 汇总输出指标
    if all_flattened_results:
        metrics_list = [r['metrics'] for r in all_flattened_results if r.get('metrics')]
        categories = [r.get('question_type', 'default') for r in all_flattened_results if r.get('metrics')]

        aggregated = aggregate_metrics(metrics_list, categories)

        print("\n" + "=" * 80)
        print(" HaluMem Robust Agent Test Summary ".center(80, "="))
        print(f"Total Queries Evaluated Across All Samples: {len(all_flattened_results)}")

        overall = aggregated.get('overall', {})
        for metric_name in ['f1', 'rougeL_f', 'bert_f1', 'sbert_similarity', 'exact_match']:
            if metric_name in overall:
                print(f"  {metric_name:20s}: {overall[metric_name]['mean']:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
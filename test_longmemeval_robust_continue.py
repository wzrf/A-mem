import os
import time
import json
import queue
import pickle
import threading
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "4"

# 复用 test_advanced_robust 中的核心模块
from test_advanced_robust import (
    RobustAdvancedMemAgent,
    setup_logger,
    parse_plain_text_answer,
    calculate_metrics,
    aggregate_metrics
)


# ============================================================================
# LongMemEval 数据加载器
# ============================================================================

@dataclass
class LongMemQA:
    question_id: str
    question: str
    answer: str
    question_type: str
    haystack_sessions: List[List[Dict[str, str]]]
    haystack_dates: List[str]


def load_longmemeval_dataset(file_path: str) -> List[LongMemQA]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(LongMemQA(
            question_id=item["question_id"],
            question=item["question"],
            answer=str(item.get("answer", "")),
            question_type=item.get("question_type", "default"),
            haystack_sessions=item.get("haystack_sessions", []),
            haystack_dates=item.get("haystack_dates", [])
        ))
    return samples


# ============================================================================
# 评测与建库逻辑
# ============================================================================

def build_memory_longmem(samples: List[LongMemQA], model: str, backend: str,
                         retrieve_k: int, temperature_c5: float, sglang_host: str,
                         sglang_port: int, max_workers: int = 4):
    memories_dir = os.path.join(os.path.dirname(__file__), f"cached_memories_longmem_{backend}_{model}")
    os.makedirs(memories_dir, exist_ok=True)
    TOKEN_CONSUMPTION_DIR = f"./token_consumption_{model}"
    os.makedirs(TOKEN_CONSUMPTION_DIR, exist_ok=True)
    progress_dir = os.path.join(os.path.dirname(__file__), f"progress_longmemeval_{model}")
    os.makedirs(progress_dir, exist_ok=True)
    print(f"TOKEN_CONSUMPTION_DIR={TOKEN_CONSUMPTION_DIR}")
    print(f"memories_dir={memories_dir}")
    print(f"progress_dir={progress_dir}")

    def process_sample(sample_idx: int, sample: LongMemQA):
        memory_cache_file = os.path.join(memories_dir, f"memory_cache_{sample.question_id}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_{sample.question_id}.pkl")
        retriever_cache_emb = os.path.join(memories_dir, f"retriever_cache_{sample.question_id}.npy")
        progress_file = os.path.join(progress_dir, f"progress_{sample.question_id}.json")
        token_consumption_file = f"{TOKEN_CONSUMPTION_DIR}/longmemeval_{sample.question_id}.json"

        # 加载进度：记录已处理的session索引
        start_session_idx = 0
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                start_session_idx = progress_data.get("last_session_idx", 0)
                print(f"[sample_idx={sample.question_id}] resuming from session {start_session_idx}")
            except Exception as e:
                print(f"[sample_idx={sample.question_id}] error reading progress file: {e}, starting from 0")

        # 如果start_session_idx == 0，可能需要加载已存在的缓存（如果存在）
        # 如果start_session_idx > 0，则必须加载缓存
        load_cache = start_session_idx > 0 or os.path.exists(memory_cache_file)

        # 创建agent
        agent = RobustAdvancedMemAgent(
            model, backend, retrieve_k, temperature_c5, sglang_host, sglang_port,
            token_consumption_file=token_consumption_file
        )

        # 加载token consumption（如果文件存在）
        if os.path.exists(token_consumption_file):
            try:
                with open(token_consumption_file, 'r') as f:
                    agent.tokens_comsumption = json.load(f)
                print(
                    f"[sample_idx={sample.question_id}] loaded {len(agent.tokens_comsumption)} token consumption records")
            except Exception as e:
                print(f"[sample_idx={sample.question_id}] error loading token consumption: {e}")

        # 加载缓存（如果存在且需要）
        if load_cache and os.path.exists(memory_cache_file):
            try:
                with open(memory_cache_file, 'rb') as f:
                    agent.memory_system.memories = pickle.load(f)
                print(
                    f"[sample_idx={sample.question_id}] loaded {len(agent.memory_system.memories)} memories from cache")
                # 加载retriever
                if os.path.exists(retriever_cache_file):
                    agent.memory_system.retriever = agent.memory_system.retriever.load(retriever_cache_file,
                                                                                       retriever_cache_emb)
                    print(f"[sample_idx={sample.question_id}] loaded retriever cache")
                else:
                    agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                        agent.memory_system.memories, 'all-MiniLM-L6-v2'
                    )
            except Exception as e:
                print(f"[sample_idx={sample.question_id}] error loading cache: {e}, rebuilding from scratch")
                start_session_idx = 0
                agent.memory_system.memories = {}
                # 重建retriever
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    agent.memory_system.memories, 'all-MiniLM-L6-v2'
                )
        elif start_session_idx > 0 and not os.path.exists(memory_cache_file):
            print(
                f"[sample_idx={sample.question_id}] WARNING: progress indicates session {start_session_idx} but cache missing, resetting progress")
            start_session_idx = 0
            # 进度文件可能已损坏，删除它
            try:
                os.remove(progress_file)
            except:
                pass
        # 若不需要加载缓存（start_session_idx == 0 且无缓存文件），则 memory_system 已处于初始状态，无需操作

        # 遍历会话与时间戳入库，从start_session_idx开始
        total_sessions = len(sample.haystack_sessions)
        for s_idx in range(start_session_idx, total_sessions):
            session = sample.haystack_sessions[s_idx]
            session_date = sample.haystack_dates[s_idx] if s_idx < len(sample.haystack_dates) else None
            time_start = time.time()
            for turn_idx, turn in enumerate(session):
                speaker = turn.get("role", "user")
                text = turn.get("content", "")
                agent.add_memory(f"Speaker {speaker} says : {text}", time=session_date)
                # print(f"[sample_idx={sample_idx}] added turn {turn_idx}/{len(session)}")
            print(
                f"[sample_idx={sample_idx}] added session {s_idx}/{total_sessions}, takes {time.time() - time_start} seconds.")

            # 每个session处理完后，保存缓存文件和进度文件
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(agent.memory_system.memories, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_emb)

            # 更新进度文件
            progress_data = {"last_session_idx": s_idx + 1, "updated_at": time.time()}
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=4)
            print(f"[sample_idx={sample.question_id}] saved cache and progress after session {s_idx}")

        print(f"[sample_idx={sample.question_id}] finished all sessions")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, idx, sample) for idx, sample in enumerate(samples)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            future.result()


def evaluate_longmemeval(samples: List[LongMemQA], model: str, backend: str,
                         retrieve_k: int, temperature_c5: float, sglang_host: str,
                         sglang_port: int, devices: List[str]):
    memories_dir = os.path.join(os.path.dirname(__file__), f"cached_memories_longmem_{backend}_{model}")
    results_file = f"./results/result_longmemeval_{model}.json"
    os.makedirs("./results", exist_ok=True)

    qa_queue = queue.Queue()
    result_queue = queue.Queue()

    def worker_loop(agent: RobustAdvancedMemAgent):
        while True:
            sample = qa_queue.get()
            if sample is None:
                qa_queue.task_done()
                break

            # 加载缓存
            mem_file = os.path.join(memories_dir, f"memory_cache_{sample.question_id}.pkl")
            ret_file = os.path.join(memories_dir, f"retriever_cache_{sample.question_id}.pkl")
            ret_emb = os.path.join(memories_dir, f"retriever_cache_{sample.question_id}.npy")

            with open(mem_file, 'rb') as f:
                agent.memory_system.memories = pickle.load(f)

            if os.path.exists(ret_file):
                agent.memory_system.retriever = agent.memory_system.retriever.load(ret_file, ret_emb)
            else:
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    agent.memory_system.memories, 'all-MiniLM-L6-v2'
                )

            # 执行预测 (映射为 Category 1 标准问题)
            prediction, user_prompt, raw_context, raw_context_list, p_tokens, c_tokens = agent.answer_question(
                sample.question, category=1, answer=sample.answer
            )
            prediction = parse_plain_text_answer(prediction)
            metrics = calculate_metrics(prediction, sample.answer) if sample.answer else {}

            res = {
                "question_id": sample.question_id,
                "question_type": sample.question_type,
                "question": sample.question,
                "prediction": prediction,
                "reference": sample.answer,
                "metrics": metrics,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens
            }
            print(res)
            result_queue.put((res, metrics, sample.question_type))
            qa_queue.task_done()

    # 启动工作线程
    threads = []
    for dev in devices:
        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5, sglang_host, sglang_port)
        t = threading.Thread(target=worker_loop, args=(agent,))
        t.start()
        threads.append(t)

    # 推送任务
    for sample in samples:
        qa_queue.put(sample)

    qa_queue.join()

    # 停止线程
    for _ in devices:
        qa_queue.put(None)
    for t in threads:
        t.join()

    # 收集结果
    results, all_metrics, all_types = [], [], []
    while not result_queue.empty():
        res, metrics, q_type = result_queue.get()
        results.append(res)
        all_metrics.append(metrics)
        all_types.append(q_type)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印汇总信息
    aggregated = aggregate_metrics(all_metrics, all_types)
    print("\n" + "=" * 60)
    print("LongMemEval Aggregate Metrics Summary:")
    for split_name, m_dict in aggregated.items():
        print(f"[{split_name}]")
        for k, v in m_dict.items():
            if 'mean' in v:
                print(f"  {k}: {v['mean']:.4f}")
    print("=" * 60)


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate robust agent on LongMemEval dataset")
    parser.add_argument("--dataset", type=str, default="data/longmemeval_mixed.json")
    parser.add_argument("--model", type=str, default="qwen3-8b")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--max-workers", type=int, default=64)
    args = parser.parse_args()

    samples = load_longmemeval_dataset(args.dataset)
    devices = ["cuda:1" for i in range(32)]  ##mengyao_debug 测试的并发
    MAX_WORKERS = args.max_workers  ##mengyao_debug build的并发
    print(f"using {MAX_WORKERS} workers")
    if os.environ.get('DEBUG') == "1":
        MAX_WORKERS = 1

    if not args.skip_build:
        print("Building memories for LongMemEval...")
        build_memory_longmem(samples, args.model, args.backend, args.retrieve_k,
                             args.temperature_c5, args.sglang_host, args.sglang_port, max_workers=MAX_WORKERS)

    print("Evaluating LongMemEval...")
    evaluate_longmemeval(samples, args.model, args.backend, args.retrieve_k,
                         args.temperature_c5, args.sglang_host, args.sglang_port, devices)


if __name__ == "__main__":
    main()
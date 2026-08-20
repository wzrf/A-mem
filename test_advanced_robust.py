"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from sglang_kvcache import get_model_and_prompt, run_one_question_sglang, run_one_question_origin_sglang
from llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
import platform
import os
import time
import json
import argparse
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime
import os
import queue
import threading

# Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('wordnet')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('/mnt/qjhs-sh-lab-01/models/all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

logger = logging.getLogger("amem_robust")

import random
from collections import defaultdict
from typing import List, Optional, Any

def sample_qa_by_ratio(
    qa_list: List[Any],
    ratio: float = 0.1,
    allow_categories: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> List[Any]:
    """
    按照类别 (category) 的比例从 QA 列表中随机抽样。

    :param qa_list: 原始 QA 对象列表 (如 sample.qa)
    :param ratio: 抽样比例 (如 0.1 代表抽 10%)
    :param allow_categories: 允许选择的类别列表，如 [1, 2, 3, 4, 5]。若为 None 则使用全部类别
    :param seed: 随机种子 (传入整数可固定抽样结果，便于复现)
    :return: 抽样后的 QA 列表
    """
    if seed is not None:
        random.seed(seed)

    # 1. 过滤符合分类条件的 QA
    if allow_categories is not None:
        allow_set = set(allow_categories)
        valid_qas = [qa for qa in qa_list if int(qa.category) in allow_set]
    else:
        valid_qas = list(qa_list)

    total_valid = len(valid_qas)
    if total_valid == 0 or ratio <= 0:
        return []

    # 2. 计算目标抽取总数 (至少取 1 个，最多不超总数)
    target_count = min(total_valid, max(1, round(total_valid * ratio)))

    if target_count >= total_valid:
        return valid_qas

    # 3. 按 category 分组
    qa_by_cat = defaultdict(list)
    for qa in valid_qas:
        qa_by_cat[int(qa.category)].append(qa)

    # 4. 计算每个类别的配额（最大余数法）
    allocated_counts = {}
    remainders = []

    for cat, qas in qa_by_cat.items():
        exact_quota = target_count * (len(qas) / total_valid)
        floor_quota = int(exact_quota)
        allocated_counts[cat] = floor_quota
        # 记录小数余数和类别
        remainders.append((exact_quota - floor_quota, cat))

    # 补充因为取整丢失的名额
    remaining_slots = target_count - sum(allocated_counts.values())
    remainders.sort(reverse=True, key=lambda x: x[0])  # 余数从大到小排序

    for i in range(remaining_slots):
        cat = remainders[i][1]
        allocated_counts[cat] += 1

    # 5. 按分配好的名额在每个类别内随机抽取
    selected_qas = []
    for cat, count in allocated_counts.items():
        if count > 0:
            selected_qas.extend(random.sample(qa_by_cat[cat], count))

    # 再次打乱顺章（可选，避免同一类别的 QA 集中在一起）
    random.shuffle(selected_qas)

    return selected_qas

class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000,
                 fusion_rag_model=None,
                 model_sglang="",
                 recomputation_rate: float=-1,
                 method_keyword="",
                 preprocess=False,
                 use_weighted_diff_attention=True,
                 sglang_url="",
                 sglang_url_prefiller="",
                 use_fusion_rag=False,
                 encoder=None,
                 rag_indices=None,
                 rag_id2texts=None,
                 token_consumption_file=""
                 ):

        if use_fusion_rag:
            self.model_sglang = model_sglang
            self.fusion_rag_model = fusion_rag_model
            self.recomputation_rate = recomputation_rate
            self.method_keyword = method_keyword
            self.preprocess = preprocess
            self.use_weighted_diff_attention = use_weighted_diff_attention
            self.sglang_url = sglang_url
            self.sglang_url_prefiller = sglang_url_prefiller
        else:
            self.fusion_rag_model = None
        self.encoder = encoder
        self.rag_indices = rag_indices
        self.rag_id2texts = rag_id2texts
        self.tokens_comsumption = []
        self.token_consumption_file = token_consumption_file

        self.memory_system = RobustAgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        _, prompt_tokens, completion_tokens = self.memory_system.add_note(content, time=time)
        self.tokens_comsumption.append(
            {
                # "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
        if self.token_consumption_file != "":
            with open(self.token_consumption_file, "w") as f:
                json.dump(self.tokens_comsumption, f, indent=4)


    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        raw_context, raw_context_list = self.retrieve_memory(keywords, k=self.retrieve_k)
        context = raw_context

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context, raw_context_list, response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"]


    def answer_question_fusionrag(self, question: str, category: int, answer: str, use_rag: bool, sample_idx: int) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        if not use_rag:
            keywords = self.generate_query_llm(question)
            raw_context, raw_context_list = self.retrieve_memory(keywords, k=self.retrieve_k)
            context = raw_context
        else:
            from rag import retrieve_single_question_debug
            raw_context_list = retrieve_single_question_debug(
                question=question,
                index=self.rag_indices[sample_idx],
                id2text=self.rag_id2texts[sample_idx],
                encoder=self.encoder,
                topk=15,
            )
            raw_context = "\n".join(raw_context_list)

        _, DEFAULT_SYSTEM_PROMPT, _ = get_model_and_prompt(model=self.model_sglang)
        DEFAULT_SYSTEM_PROMPT += "Based on the context: "

        assert category in [1, 2, 3, 4]

        if category == 5:
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f""", answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
        elif category == 3:
            user_prompt = f""", write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
        else:
            user_prompt = f""", write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""


        query_draft = user_prompt.format(question=question)
        time_start = time.time()
        if self.recomputation_rate < 1.0:
            recompute_tokens, recompute_tokens_list, retrieved_docs, recompute_rate, sorted_doc_index, sorted_doc_index_before, recompute_indices = self.fusion_rag_model.draft_one_question(
                DEFAULT_SYSTEM_PROMPT,  ## DEFAULT_SYSTEM_PROMPT
                raw_context_list,
                query_draft,
                self.recomputation_rate,
                self.method_keyword,
                False,
                False,
                [],
                self.use_weighted_diff_attention,
                self.preprocess,  ## if do preprocess
                False,
                True
            )
            time_end = time.time()

            content, usage, top_logprobs, real_recomputation_rate = run_one_question_sglang(
                query_draft,
                raw_context_list,
                500,  ## max tokens.
                [],
                recompute_tokens,
                recompute_tokens_list,
                1,  ## max_workers.
                self.recomputation_rate,
                self.model_sglang,
                self.sglang_url,
                self.sglang_url_prefiller,
                self.method_keyword,
                recompute_indices=recompute_indices,
            )
            print(f"time_draft={time_end-time_start}, time_run={time.time()-time_end}")
        else:
            content, usage, top_logprobs, real_recomputation_rate = run_one_question_origin_sglang(
                query_prompt=query_draft,
                retrived_docs=raw_context_list,
                max_tokens=500,
                model_use=self.model_sglang,
                endpoint_url=self.sglang_url,
            )
            print(f"time_run={time.time() - time_start}")
        if "</think>" in content:
            content = content.split("</think>")[1].strip()
        return content, user_prompt, raw_context, raw_context_list, usage["prompt_tokens"], usage["completion_tokens"]


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def build_memory(dataset_path: str, model: str, output_path: Optional[str] = None,
                 ratio: float = 1.0, backend: str = "sglang",
                 temperature_c5: float = 0.5, retrieve_k: int = 10,
                 sglang_host: str = "http://localhost", sglang_port: int = 30000,
                 max_workers: int = 1):
    """Evaluate the robust agent on the LoComo dataset using multi-threading."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio * 100:.1f}% of dataset)")

    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_robust_{}_{}".format(backend, model),
    )
    os.makedirs(memories_dir, exist_ok=True)

    sapphire3_ip = "192.168.200.15"
    sapphire3_prefiller_port = 30003
    fusion_rag_model = None

    def process_sample(sample_idx: int, sample):
        """单样本处理函数（运行在独立线程中）"""
        prefix = f"[Sample {sample_idx + 1}/{len(samples)}]"

        token_consumption_file = f"./token_consumption/locomo_{sample_idx}.json"

        agent = RobustAdvancedMemAgent(
            model, backend, retrieve_k, temperature_c5,
            sglang_host, sglang_port,
            model_sglang="Qwen3-8B",
            method_keyword="",
            preprocess=False,
            use_weighted_diff_attention=True,
            sglang_url=f"http://{sapphire3_ip}:{sapphire3_prefiller_port}/v1/completions",
            sglang_url_prefiller=f"http://{sapphire3_ip}:{sapphire3_prefiller_port}/v1/completions",
            fusion_rag_model=fusion_rag_model,
            token_consumption_file=token_consumption_file
        )

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        if os.path.exists(memory_cache_file):
            eval_logger.info(f"{prefix} Loading cached memories")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                eval_logger.info(f"{prefix} Found retriever cache files")
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                eval_logger.info(f"{prefix} No retriever cache found, loading from memory")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, 'all-MiniLM-L6-v2'
                )
            eval_logger.info(f"{prefix} Successfully loaded {len(cached_memories)} memories")
        else:
            eval_logger.info(f"{prefix} No cached memories found. Creating new memories.")

            eval_logger.info(f"{prefix} total sessions: {len(sample.conversation.sessions.items())}")
            for session_idx, turns in sample.conversation.sessions.items():
                for turx_idx, turn in enumerate(turns.turns):
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    agent.add_memory(conversation_tmp, time=turn_datatime)
                    # print(f"{prefix} finish turn {turx_idx}/{len(turns.turns)}")
                print(f"{prefix} finish session {session_idx}")

            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            eval_logger.info(f"{prefix} Successfully cached {len(memories_to_cache)} memories")

        eval_logger.info(f"{prefix} Finished processing")

    # 使用 ThreadPoolExecutor 进行 16 线程并发处理
    eval_logger.info(f"Starting multi-threaded processing with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_sample, sample_idx, sample): sample_idx
            for sample_idx, sample in enumerate(samples)
        }

        for future in as_completed(futures):
            sample_idx = futures[future]
            try:
                future.result()
            except Exception as e:
                eval_logger.error(f"[Sample {sample_idx + 1}] Processing failed with error: {e}", exc_info=True)



def evaluate_dataset(
    dataset_path: str,
    model: str,
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    backend: str = "sglang",
    temperature_c5: float = 0.5,
    retrieve_k: int = 10,
    sglang_host: str = "http://localhost",
    sglang_port: int = 30000,
    use_fusion_rag=False,
    recomputation_rate=0.3,
    qa_ratio=1.0,
    devices: Optional[List[str]] = None,  # 设备列表，如 ["cuda:0", "cuda:1"]
    sglang_model="",
    sglang_url="",
    sglang_url_prefiller="",
    draft_model_path="",
    draft_model_type="",
    draft_model_name="",
    use_rag=False,
):
    """Evaluate the robust agent with fine-grained (per-QA) parallelism."""
    if devices is None:
        devices = ["cuda:4"]

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    # --- 检查并读取已有的结果文件 ---
    fusion_rag_tag = "fusion_rag" if use_fusion_rag else ""
    rag_tag = "_simplerag" if use_rag else ""
    os.makedirs("./results", exist_ok=True)
    results_file = f"./results/result{rag_tag}_{fusion_rag_tag}_{recomputation_rate}_{draft_model_name}_{sglang_model}_retrieve_{retrieve_k}.json"
    print(f"results_file={results_file}")
    processed_keys = set()

    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            for r in results:
                processed_keys.add((r["sample_id"], r["question"]))
                all_metrics.append(r["metrics"])
                all_categories.append(r["category"])
                total_questions += 1
                category_counts[r["category"]] += 1
            eval_logger.info(f"Loaded {len(results)} existing results from {results_file}, skipping them.")
        except Exception as e:
            eval_logger.warning(f"Failed to load existing results from {results_file}: {e}")
            results = []

    error_num = 0
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_robust_{}_{}".format(backend, model),
    )
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1, 2, 3, 4]

    SYSTEM_ = platform.system().lower()

    # --- 任务队列与结果队列 ---
    qa_queue = queue.Queue()
    result_queue = queue.Queue()

    # --- Worker 线程逻辑（仅接收已初始化的 agent 对象） ---
    def worker_loop(agent: RobustAdvancedMemAgent):
        """Worker 线程：直接使用传入的 Agent 处理 QA 任务"""
        current_sample_idx = -1  # 记录当前 Agent 加载的 Sample

        while True:
            task = qa_queue.get()
            if task is None:  # 结束信号
                qa_queue.task_done()
                break

            sample_idx, qa = task

            # 如果任务属于新的 Sample，则加载该 Sample 的 Memory 缓存
            if current_sample_idx != sample_idx:
                memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
                retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
                retriever_cache_embeddings_file = os.path.join(
                    memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
                )

                with open(memory_cache_file, 'rb') as f:
                    cached_memories = pickle.load(f)
                agent.memory_system.memories = cached_memories

                if os.path.exists(retriever_cache_file):
                    agent.memory_system.retriever = agent.memory_system.retriever.load(
                        retriever_cache_file, retriever_cache_embeddings_file
                    )
                else:
                    agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                        cached_memories, 'all-MiniLM-L6-v2'
                    )
                current_sample_idx = sample_idx

            # 评估单个 QA
            if use_fusion_rag:
                prediction, user_prompt, raw_context, raw_context_list, prompt_tokens, completion_tokens = agent.answer_question_fusionrag(
                    qa.question, qa.category, qa.final_answer, use_rag, sample_idx
                )
            else:
                prediction, user_prompt, raw_context, raw_context_list, prompt_tokens, completion_tokens = agent.answer_question(
                    qa.question, qa.category, qa.final_answer
                )
            print(f"prediction={prediction}")

            prediction = parse_plain_text_answer(prediction)
            metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
            }

            result = {
                "sample_id": sample_idx,
                "question": qa.question,
                "prediction": prediction,
                "reference": qa.final_answer,
                "category": qa.category,
                "metrics": metrics,
                "raw_context_len": len(raw_context_list),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }

            log_payload = {
                "question": qa.question,
                "prediction": prediction,
                "reference": qa.final_answer,
                "user_prompt": user_prompt,
                "category": qa.category,
                "raw_context": raw_context
            }

            result_queue.put((result, metrics, qa.category, log_payload))
            qa_queue.task_done()

    # --- 在主线程中初始化所有设备的 Agent 并启动工作线程 ---
    threads = []
    for dev in devices:
        eval_logger.info(f"Initializing agent on device: {dev}...")
        if SYSTEM_ == "linux" and use_fusion_rag:
            from FusionRAG.run_question import FusionRAGModel
            fusion_rag_model = FusionRAGModel(
                device=dev,
                draft_model_device=dev,
                draft_model_path=draft_model_path,
                draft_model_type=draft_model_type,
                draft_model_name=draft_model_name,
                model_path="",
                cache_path="",
                preprocess=False,
                apikey="xxx",
            )
        else:
            fusion_rag_model = None

        from rag import init_locomo_indeces, OnlineEncoder
        rag_indices, rag_id2texts = init_locomo_indeces()
        agent = RobustAdvancedMemAgent(
            model, backend, retrieve_k, temperature_c5,
            sglang_host, sglang_port,
            model_sglang="Qwen3-8B",
            recomputation_rate=recomputation_rate,
            method_keyword="",
            preprocess=False,
            use_weighted_diff_attention=True,
            sglang_url=sglang_url,
            sglang_url_prefiller=sglang_url_prefiller,
            fusion_rag_model=fusion_rag_model,
            use_fusion_rag=use_fusion_rag,
            encoder=OnlineEncoder(),
            rag_indices=rag_indices,
            rag_id2texts=rag_id2texts,
        )

        t = threading.Thread(target=worker_loop, args=(agent, ))
        t.start()
        threads.append(t)

    # --- 逐个 Sample 处理 ---
    for sample_idx, sample in enumerate(samples):
        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")

        # 1. 确保当前 Sample 的 Memory 缓存已建好
        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        if not os.path.exists(memory_cache_file):
            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories...")
            temp_agent = RobustAdvancedMemAgent(
                model, backend, retrieve_k, temperature_c5,
                sglang_host, sglang_port,
                model_sglang="Qwen3-8B",
                recomputation_rate=recomputation_rate,
                method_keyword="", preprocess=False,
                use_weighted_diff_attention=True,
                sglang_url=sglang_url,
                sglang_url_prefiller=sglang_url_prefiller,
                fusion_rag_model=None, use_fusion_rag=False,
            )
            for session_idx, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    temp_agent.add_memory(conversation_tmp, time=turns.date_time)

            memories_to_cache = temp_agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            temp_agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories for sample {sample_idx}")

        # 2. 筛选出当前 Sample 未评估的 QA 任务
        qa_sub_list = sample_qa_by_ratio(
            qa_list=sample.qa,
            ratio=qa_ratio,
            allow_categories=allow_categories,
            seed=42,
        )

        unprocessed_qas = []
        for qa in qa_sub_list:
            if int(qa.category) in allow_categories:
                if (sample_idx, qa.question) in processed_keys:
                    eval_logger.info(f"Skipping already evaluated question: {qa.question}")
                    continue
                unprocessed_qas.append(qa)

        if not unprocessed_qas:
            continue

        eval_logger.info(f"Submitting {len(unprocessed_qas)} QA tasks to workers for sample {sample_idx}")

        # 3. 将当前 Sample 的 QA 任务放入队列
        for qa in unprocessed_qas:
            qa_queue.put((sample_idx, qa))

        # 4. 等待当前 Sample 的所有 QA 被所有 worker 消耗完毕
        qa_queue.join()

        # 5. 主线程收集并写入当前 Sample 的所有评估结果
        while not result_queue.empty():
            result, metrics, category, log_payload = result_queue.get()
            total_questions += 1
            category_counts[category] += 1
            all_metrics.append(metrics)
            all_categories.append(category)
            results.append(result)

            eval_logger.info(f"Question {total_questions}: {log_payload['question']}")
            eval_logger.info(f"Prediction: {log_payload['prediction']}")
            eval_logger.info(f"Reference: {log_payload['reference']}")
            eval_logger.info(f"User Prompt: {log_payload['user_prompt']}")
            eval_logger.info(f"Category: {log_payload['category']}")
            eval_logger.info(f"Raw Context: {log_payload['raw_context']}")

            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)

            if total_questions % 10 == 0:
                eval_logger.info(f"Processed {total_questions} questions")

    # 停止所有 Worker 线程
    for _ in devices:
        qa_queue.put(None)
    for t in threads:
        t.join()

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
    }
    eval_logger.info(f"Error number: {error_num}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    return final_results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="qwen3-8b",
                        help="Model to use")
    parser.add_argument("--sglang_model", type=str, default="qwen2.5-7B",
                        help="Model to use for fusionrag")
    parser.add_argument("--draft_model", type=str, default="qwen2.5-3B",
                        help="Model to use draft")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--skip_build", type=bool, default=False,
                        help="skip parallel build")
    parser.add_argument("--use_fusion_rag", type=str, default="false",
                        help="use fusion rag or not")
    parser.add_argument("--use_rag", type=str, default="false",
                        help="use rag or not")
    parser.add_argument("--recomputation_rate", type=float, default=0.3,
                        help="recomputation rate for fusionrag")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--qa_ratio", type=float, default=1.0,
                        help="Ratio of qa to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, or vllm)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    args.use_fusion_rag = args.use_fusion_rag.lower() == "true"
    args.use_rag = args.use_rag.lower() == "true"
    print(f"use_fusion_rag= {args.use_fusion_rag}")

    sapphire3_ip = "192.168.200.15"
    sapphire3_port_qwen25_7b = 30003

    if "qwen2.5-7B".lower() in args.sglang_model.lower():
        sglang_url = sglang_url_prefiller = f"http://{sapphire3_ip}:{sapphire3_port_qwen25_7b}/v1/completions"
    else:
        print(f"unknown sglang model")
        exit(0)

    if "qwen2.5-3b".lower() in args.draft_model.lower():
        draft_model_path = '/mnt/data/models/Qwen2.5-3B-Instruct'
        draft_model_type = "qwen"
        draft_model_name = "Qwen2.5-3B-Instruct"
    elif "qwen2.5-1.5b".lower() in args.draft_model.lower():
        draft_model_path = '/mnt/data/models/Qwen2.5-1.5B-Instruct'
        draft_model_type = "qwen"
        draft_model_name = "Qwen2.5-1.5B-Instruct"
    else:
        print(f"unknown draft model")
        exit(0)

    ##mengyao_debug max_workers
    MAX_WORKERS = 10
    if os.environ.get('DEBUG') == "1":
        MAX_WORKERS = 1

    ## 是否只是build
    if not args.skip_build:
        build_memory(
            dataset_path, args.model, output_path, args.ratio,
            args.backend, args.temperature_c5, args.retrieve_k,
            args.sglang_host, args.sglang_port, max_workers=MAX_WORKERS
        )

    devices = ["cuda:1"]
    # devices = ["cuda:0"]
    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port, args.use_fusion_rag,
        args.recomputation_rate, qa_ratio=args.qa_ratio, devices=devices,
        sglang_model=args.sglang_model,
        sglang_url=sglang_url,
        sglang_url_prefiller=sglang_url_prefiller,
        draft_model_path=draft_model_path,
        draft_model_type=draft_model_type,
        draft_model_name=draft_model_name,
        use_rag=args.use_rag,
    )


if __name__ == "__main__":
    main()

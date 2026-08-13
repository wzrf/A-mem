#!/usr/bin/env python3
"""
Test FusionRAG on result_reflect.json dataset

Following the same workflow as unified_process_cache.py:
1. Load all documents from all sub-questions as independent text chunks
2. Generate independent KV cache for each document
3. Use BGE model to compute document similarity (context_rank)
4. If preprocess=True, perform FusionRAG preprocess (fuse related documents' KV cache)
5. For each sub-question, generate answer using preprocessed KV cache
6. Use OpenAI API to judge if answer is correct
7. A question is correct only if all sub-questions are correct
"""

import json
import os
import sys
import csv
import shutil
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from enum import Enum
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig
from FlagEmbedding import BGEM3FlagModel

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import (
    prefill_and_save_kv_cache,
    load_kv_and_generate,
    prefill_with_cache_and_save_preprocess,
    rotate_half,
    find_group_and_index,
    compute_f1,
    _exact_match_score
)
from ktransformers.models.custom_cache import StaticCache
import torch.nn.functional as F


class PreprocessScope(Enum):
    """
    Enum to control the scope of document retrieval during preprocessing

    GLOBAL: Retrieve similar documents from ALL examples globally (original behavior)
    PER_EXAMPLE: Retrieve similar documents only within each example's documents
    SKIP_UNTESTED: Skip retrieval for documents from untested examples (should_test=False)
    """
    GLOBAL = "global"
    PER_EXAMPLE = "per_example"
    SKIP_UNTESTED = "skip_untested"


def load_model(model_type, model_path, config, device="cuda:0", use_multi_gpu=False):
    """
    Load model based on model type (same as unified_process_cache.py)

    Args:
        model_type: Type of model ('qwen', 'qwen2', 'qwen3', 'mistral', 'llama', 'pangu')
        model_path: Path to the model
        config: Model configuration
        device: Device to load model on (single GPU)
        use_multi_gpu: If True, use device_map="auto" for multi-GPU

    Returns:
        model: Loaded model
        device_map: Device map if multi-GPU, else None
    """
    load_kwargs = {
        'config': config,
        'torch_dtype': config.torch_dtype
    }

    # Add device_map for multi-GPU
    if use_multi_gpu:
        from ktransformers.models.modeling_qwen3 import Qwen3ForCausalLM
        config = AutoConfig.from_pretrained(model_path)

        # 2. 创建一个空的模型（只有架构，没有权重）
        with torch.device("meta"):  # 使用meta设备，不占用实际内存
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(config)
        from accelerate import infer_auto_device_map
        # 3. 推断设备映射
        device_map = infer_auto_device_map(
            model,
            max_memory={
                0: "20GiB",
                1: "20GiB",
                2: "20GiB",
                3: "20GiB",
            },
            no_split_module_classes=model._no_split_modules  # 保持某些模块不被分割
        )
        from accelerate import infer_auto_device_map
        load_kwargs['device_map'] = device_map

    if model_type == 'mistral':
        from ktransformers.models.modeling_mistral import MistralForCausalLM
        with torch.no_grad():
            model = MistralForCausalLM.from_pretrained(model_path, **load_kwargs)
    elif model_type == 'pangu':
        from ktransformers.models.modeling_openpangu_dense import PanguEmbeddedForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = PanguEmbeddedForCausalLM.from_pretrained(model_path, **load_kwargs)
    elif model_type == 'qwen' or model_type == 'qwen2':
        from ktransformers.models.modeling_qwen2 import Qwen2ForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = Qwen2ForCausalLM.from_pretrained(model_path, **load_kwargs)
    elif model_type == 'qwen3':
        from ktransformers.models.modeling_qwen3 import Qwen3ForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = Qwen3ForCausalLM.from_pretrained(model_path, **load_kwargs)
    elif model_type == 'llama':
        from ktransformers.models.modeling_llama import LlamaForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = LlamaForCausalLM.from_pretrained(model_path, **load_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Get device_map if using multi-GPU
    device_map = None
    if use_multi_gpu:
        device_map = model.hf_device_map
        print(f"\nModel loaded with device_map across GPUs:")
        for name, dev in device_map.items():
            print(f"  {name}: {dev}")
    else:
        model = model.to(device)

    return model, device_map


def load_system_prompt(model_family: str, dataset_type: str = "2wikimqa") -> str:
    """
    Load system prompt from config file
    """
    config_path = "./config/dataset2prompt_few-shot.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Get system prompt
    if model_family in config["system_prompt"]:
        if dataset_type in config["system_prompt"][model_family]:
            return config["system_prompt"][model_family][dataset_type]

    # Default to Qwen2.5 2wikimqa
    return config["system_prompt"]["Qwen3"]["2wikimqa"]


def prepare_reflect_data(
    data_path: str,
    tokenizer,
    bge_model_path: str,
    model_type: str = 'qwen2',
    topk: int = 10,
    max_main_questions: int = None,
    preprocess: bool = True,
    preprocess_scope: PreprocessScope = PreprocessScope.GLOBAL
) -> Tuple[List, torch.Tensor, List, List]:
    """
    Prepare data from result_reflect.json with configurable document corpus scope

    Args:
        model_type: Type of model to determine system prompt
        topk: Top-k similar documents for each document
        preprocess: Whether to compute context_rank
        preprocess_scope: Scope of document retrieval
            - GLOBAL: All documents from all questions (original behavior)
            - PER_EXAMPLE: Only retrieve within each example's documents
            - SKIP_UNTESTED: Exclude documents from untested examples (should_test=False)

    Returns:
        questions_data: List of dicts for each main question
        system_tensor: Tokenized system prompt
        context_rank: [total_docs x topk] array of similar document indices
        corpus_lens: List of document counts per question
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_main_questions:
        dataset = dataset[:max_main_questions]
        print(f"Limited to first {max_main_questions} main questions")

    # Map model_type to model_family for system prompt
    model_family_map = {
        'qwen': 'Qwen2.5',
        'qwen2': 'Qwen2.5',
        'qwen3': 'Qwen3',
        'mistral': 'Mistral',
        'llama': 'Llama',
        'pangu': 'Pangu'
    }
    model_family = model_family_map.get(model_type, 'Qwen2.5')

    # Tokenize system prompt (shared across all questions)
    system_prompt = load_system_prompt(model_family, "2wikimqa")
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=True)
    system_tensor = torch.tensor(system_tokens, dtype=torch.long)

    # STEP 1: Build document corpus based on preprocess_scope
    print("\n" + "="*80)
    print(f"Building document corpus with scope: {preprocess_scope.value}")
    print("="*80)

    global_corpus = []  # Documents based on scope
    corpus_lens = []  # Number of docs per question
    questions_data = []

    # First pass: collect all documents globally and build question metadata
    for main_q_idx, data_item in enumerate(dataset):
        main_question = data_item["question"]
        main_answer = data_item["answer"]
        intermediate_context = data_item.get("intermediate_context", [])

        question_docs = []  # Documents for THIS question only
        doc_to_idx = {}  # Local doc -> chunk_id mapping for this question
        sub_questions_info = []

        # Check if this main question should be tested
        # Skip if main question's llm_judge is False
        should_test_main_question = True
        # if data_item.get('llm_judge', True) is False:
        #     should_test_main_question = False

        for sub_q_idx, sub_q in enumerate(intermediate_context):
            docs = sub_q.get("retrieve docs", [])
            doc_chunk_ids = []  # chunk_ids for this sub-question (local to this question)

            for doc in docs:
                if doc not in doc_to_idx:
                    # New document for this question
                    question_docs.append(doc)
                    chunk_id = len(question_docs)  # chunk_id starts from 1
                    doc_to_idx[doc] = chunk_id
                    doc_chunk_ids.append(chunk_id)
                else:
                    # Document already seen in this question
                    doc_chunk_ids.append(doc_to_idx[doc])

            # Remove "Intermediate queryXXX:" prefix from query
            query = sub_q['query']
            if query.startswith("Intermediate query"):
                # Find the colon and extract text after it
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()

            # Remove "Intermediate answerXXX:" prefix from answer
            answer = sub_q['answer']
            if answer.startswith("Intermediate answer"):
                # Find the colon and extract text after it
                colon_pos = answer.find(":")
                if colon_pos != -1:
                    answer = answer[colon_pos + 1:].strip()

            # Check if any sub-question has problematic answer
            # If so, skip the entire main question
            if "No relevant information found" in answer or "没有相关信息" in answer:
                should_test_main_question = False

            sub_questions_info.append({
                'query': query,
                'answer': answer,
                'chunk_ids': doc_chunk_ids,  # chunk_ids for docs used by this sub-question
            })

        print(f"  Main question {main_q_idx + 1}: {len(question_docs)} unique documents, {len(sub_questions_info)} sub-questions")

        # Tokenize documents for this main question
        doc_tensors = []
        for doc in question_docs:
            doc_text = f"Document: {doc}\n"
            doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
            doc_tensors.append(doc_tensor)

        # Add this question's docs to global corpus based on scope
        # For SKIP_UNTESTED, only add docs if should_test is True
        if preprocess_scope == PreprocessScope.SKIP_UNTESTED:
            if should_test_main_question:
                global_corpus.extend(question_docs)
                corpus_lens.append(len(question_docs))
            else:
                corpus_lens.append(0)  # No docs added for this question
        else:
            # GLOBAL and PER_EXAMPLE: add all docs
            global_corpus.extend(question_docs)
            corpus_lens.append(len(question_docs))

        questions_data.append({
            'main_question': main_question,
            'main_answer': main_answer,
            'sub_questions': sub_questions_info,
            'docs': question_docs,
            'doc_tensors': doc_tensors,
            'should_test': should_test_main_question,  # Whether to test this main question
        })

    # Statistics
    total_main_q = len(questions_data)
    testable_main_q = sum(1 for q in questions_data if q['should_test'])
    skipped_main_q = total_main_q - testable_main_q

    total_sub_q = sum(len(q['sub_questions']) for q in questions_data)
    testable_sub_q = sum(len(q['sub_questions']) for q in questions_data if q['should_test'])
    skipped_sub_q = total_sub_q - testable_sub_q

    total_docs = sum(len(q['docs']) for q in questions_data)

    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total main questions: {total_main_q}")
    print(f"  - Testable: {testable_main_q}")
    print(f"  - Skipped (llm_judge=False or problematic answers): {skipped_main_q}")
    print(f"\nTotal sub-questions: {total_sub_q}")
    print(f"  - Testable: {testable_sub_q}")
    print(f"  - Skipped: {skipped_sub_q}")
    print(f"\nTotal documents (across all questions): {total_docs}")
    print(f"{'='*80}")

    # STEP 2: Build FAISS index and compute context_rank based on scope
    context_rank = []
    if preprocess and len(global_corpus) > 0:
        print("\n" + "="*80)
        print(f"Computing document similarity with BGE + FAISS (scope: {preprocess_scope.value})...")
        print("="*80)

        import faiss
        from FlagEmbedding import FlagModel

        # Load BGE model
        print(f"Loading BGE model from {bge_model_path}...")
        bgem3 = FlagModel(bge_model_path, use_fp16=True)

        if preprocess_scope == PreprocessScope.PER_EXAMPLE:
            # Build separate FAISS index for EACH example
            print("Building per-example FAISS indices...")
            context_rank = []

            for q_idx, q_data in enumerate(questions_data):
                example_docs = q_data['docs']

                if len(example_docs) == 0:
                    continue

                print(f"  Example {q_idx + 1}: {len(example_docs)} documents")

                # Encode this example's documents
                example_embeddings = bgem3.encode(example_docs)
                example_embeddings = example_embeddings.astype(np.float32)

                # Build FAISS index for this example
                dim = example_embeddings.shape[-1]
                index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
                index.train(example_embeddings)
                index.add(example_embeddings)

                # Search within this example only
                example_embeddings_query = bgem3.encode_queries(example_docs)
                example_embeddings_query = example_embeddings_query.astype(np.float32)
                actual_k = min(topk, len(example_docs))
                score, idx = index.search(example_embeddings_query, k=actual_k)

                # Convert local indices to global indices
                global_offset = sum(corpus_lens[:q_idx])
                global_idx = idx + global_offset

                # Pad to topk if needed
                if actual_k < topk:
                    pad_width = ((0, 0), (0, topk - actual_k))
                    global_idx = np.pad(global_idx, pad_width, mode='constant', constant_values=-1)

                context_rank.append(global_idx)

            if len(context_rank) > 0:
                context_rank = np.vstack(context_rank)
                print(f"Per-example context rank computed: {context_rank.shape}")

        else:
            # GLOBAL or SKIP_UNTESTED: Build single FAISS index for all corpus
            print(f"Encoding {len(global_corpus)} documents for FAISS index...")
            corpus_embeddings = bgem3.encode(global_corpus)
            print(f"Corpus embeddings shape: {corpus_embeddings.shape}")

            # Build FAISS index
            dim = corpus_embeddings.shape[-1]
            index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
            corpus_embeddings = corpus_embeddings.astype(np.float32)
            index.train(corpus_embeddings)
            index.add(corpus_embeddings)
            print(f"FAISS index built with {index.ntotal} vectors")

            # Search for similar documents
            print(f"Searching for top-{topk} similar documents for each document...")
            corpus_embeddings_query = bgem3.encode_queries(global_corpus)
            corpus_embeddings_query = corpus_embeddings_query.astype(np.float32)
            score, idx = index.search(corpus_embeddings_query, k=topk)
            context_rank = idx  # Shape: [total_docs, topk]

            print(f"Context rank computed: {context_rank.shape}")

        bgem3 = None  # Free memory

    return questions_data, system_tensor, context_rank, corpus_lens


def judge_answer_with_openai(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    predicted_answer: str,
    ground_truth_answer: str
) -> Tuple[bool, str]:
    """
    Use OpenAI API to judge if the predicted answer is correct

    Returns:
        Tuple[bool, str]: (is_correct, reason)
    """
    judge_prompt = f"""你是一个答案评估专家。你的任务是判断预测答案是否正确地回答了问题。

问题: {question}

标准答案: {ground_truth_answer}

预测答案: {predicted_answer}

请判断预测答案是否正确回答了问题。判断标准：
1. 预测答案包含了标准答案的关键信息
2. 预测答案与标准答案在语义上等价
3. 允许措辞上的细微差异，只要意思保持一致即可

请按照以下格式回答：
判断: [正确/错误]
原因: [详细说明为什么正确或错误，至少30字]"""

    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "你是一个专业的答案评估专家。"},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        result = response.choices[0].message.content.strip()

        # 解析返回结果
        is_correct = False
        reason = result

        # 尝试解析格式化的回答
        lines = result.split('\n')
        for i, line in enumerate(lines):
            if '判断' in line or 'judgment' in line.lower():
                if '正确' in line or 'YES' in line.upper() or '对' in line:
                    is_correct = True
                elif '错误' in line or 'NO' in line.upper() or '错' in line:
                    is_correct = False
            if '原因' in line or 'reason' in line.lower():
                # 获取原因部分
                if ':' in line or '：' in line:
                    reason_start = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                    # 如果原因在下一行
                    if len(lines) > i + 1 and not reason_start:
                        reason = '\n'.join(lines[i+1:]).strip()
                    else:
                        reason = reason_start + '\n' + '\n'.join(lines[i+1:]).strip()
                    reason = reason.strip()
                    break

        # 如果没有找到格式化的原因，使用整个回答
        if not reason or len(reason) < 10:
            reason = result

        return is_correct, reason

    except Exception as e:
        error_msg = f"调用 OpenAI API 时出错: {e}"
        print(error_msg)
        return False, error_msg


def main(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    draft_model_path=None,  # Draft model path for DraftModel method
    data_path='/mnt/data/ktransformers-dev/result_reflect.json',
    cache_path='/mnt/data3/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    max_cache_len=32768,
    rate=0.2,
    topk=10,
    preprocess=True,
    preprocess_scope=PreprocessScope.GLOBAL,
    reprocess_method='FusionRAG',
    use_entropy_selection=False,  # 是否使用熵选层 (用于 QueryAttention 消融实验)
    entropy_top_k=4,  # 熵选层选择的层数
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    revert_rope=True,
    device="cuda:0",
    device_draft_model="",
    use_multi_gpu=True,
    openai_api_key=None,
    openai_base_url="https://api.openai.com/v1",
    openai_model="gpt-4",
    max_samples=None
):
    """
    Main function for FusionRAG testing on result_reflect.json

    Args:
        model_type: Type of model ('qwen', 'qwen2', 'qwen3', 'mistral', 'llama', 'pangu')
        model_path: Path to the language model
        data_path: Path to result_reflect.json
        cache_path: Path to save KV cache
        model_name: Model name for logging
        max_cache_len: Maximum cache length
        rate: Compression rate (0=no compression, 1=full recompute)
        topk: Top-k similar documents to fuse in preprocess
        preprocess: Whether to use FusionRAG preprocess
        preprocess_scope: Scope for document retrieval (GLOBAL, PER_EXAMPLE, SKIP_UNTESTED)
        reprocess_method: Method name ('FusionRAG')
        bge_model_path: Path to BGE model for computing similarity
        revert_rope: Whether to revert rope in preprocessing
        device: Device to use (for single GPU)
        use_multi_gpu: Whether to use multi-GPU with device_map='auto'
        openai_api_key: OpenAI API key for judging answers
        openai_base_url: OpenAI API base URL
        openai_model: OpenAI model for judging
        max_samples: Maximum number of main questions to test (None = all)
    """

    # Create cache directories with model-specific subdirectories
    # Different preprocess_scope uses different preprocess cache directories
    model_cache_root = os.path.join(cache_path, model_name)
    save_path = os.path.join(model_cache_root, 'kv_cache')

    # Separate preprocess cache for different scopes
    if preprocess_scope == PreprocessScope.GLOBAL:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_global')
    elif preprocess_scope == PreprocessScope.PER_EXAMPLE:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_per_example')
    elif preprocess_scope == PreprocessScope.SKIP_UNTESTED:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_skip_untested')
    else:
        preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache')

    csv_path = os.path.join(model_cache_root, 'results')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(preprocess_save_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    print(f"Cache directories created under: {model_cache_root}")
    print(f"  - KV cache: {save_path}")
    print(f"  - Preprocess cache ({preprocess_scope.value}): {preprocess_save_path}")
    print(f"  - Results: {csv_path}")

    # Load model and tokenizer
    print(f"Loading tokenizer and config from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    print(f"Loading {model_type} model...")
    if use_multi_gpu:
        print("Using multi-GPU with device_map='auto'")
    model, device_map = load_model(model_type, model_path, config, device, use_multi_gpu)

    # Load draft model if using DraftModel method
    draft_model = None
    if reprocess_method == 'DraftModel':
        if draft_model_path is None:
            raise ValueError("draft_model_path must be provided when using DraftModel method")
        print(f"\nLoading draft model from {draft_model_path}...")
        draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
        draft_config._attn_implementation = "sdpa"
        # Draft model always on single GPU
        draft_model, _ = load_model('qwen', draft_model_path, draft_config, device_draft_model, use_multi_gpu=False)
        draft_model.eval()
        print(f"Draft model loaded: {draft_model.config.num_hidden_layers} layers")

    # Prepare data organized by main questions
    print("Preparing data organized by main questions...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, model_type, topk, max_samples, preprocess, preprocess_scope
    )

    # Initialize OpenAI client
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    # CSV file for results (include preprocess_scope and revert_rope in filename)
    rope_suffix = "_revert_rope" if revert_rope else ""
    if preprocess:
        csv_file = f"{csv_path}/{reprocess_method}_{preprocess_scope.value}_topk_{topk}_rate_{rate}{rope_suffix}.csv"
        result_file = f"{csv_path}/{reprocess_method}_{preprocess_scope.value}_topk_{topk}_rate_{rate}{rope_suffix}.txt"
        rate1_csv_file = f"{csv_path}/{reprocess_method}_{preprocess_scope.value}_topk_{topk}_rate_1{rope_suffix}.csv"
    else:
        csv_file = f"{csv_path}/{reprocess_method}_rate_{rate}{rope_suffix}.csv"
        result_file = f"{csv_path}/{reprocess_method}_rate_{rate}{rope_suffix}.txt"
        rate1_csv_file = f"{csv_path}/{reprocess_method}_rate_1{rope_suffix}.csv"

    # Load rate=1 results for comparison if rate != 1
    rate1_results = {}
    if rate != 1:
        if os.path.exists(rate1_csv_file):

            print(f"\nLoading rate=1 baseline results from {rate1_csv_file}...")
            with open(rate1_csv_file, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    key = (row['Main Question'], row['Sub Question'])
                    rate1_results[key] = {
                        'predicted': row['Predicted'],
                        'correct': row['Correct'],
                        'f1': row['F1'],
                        'em': row['EM'],
                        'reason': row['Reason']
                    }
            print(f"Loaded {len(rate1_results)} rate=1 results for comparison")

    # Write CSV header
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if rate != 1:
            writer.writerow([
                'Main Question', 'Sub Question', 'Ground Truth',
                'Predicted', 'Correct', 'F1', 'EM', 'Reason',
                'Rate1_Predicted', 'Rate1_Correct', 'Rate1_F1', 'Rate1_EM', 'Rate1_Reason'
            ])
        else:
            writer.writerow(['Main Question', 'Sub Question', 'Ground Truth', 'Predicted', 'Correct', 'F1', 'EM', 'Reason'])

    # Initialize static cache
    # For multi-GPU, pass device_map; for single GPU, pass device string
    cache_device = device_map if use_multi_gpu else device
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=cache_device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Answer sub-questions with on-demand KV cache generation
    print(f"\n{'='*80}")
    print("Answering sub-questions with on-demand KV cache generation")
    print(f"{'='*80}")

    system_len = system_tensor.shape[0]

    # Determine the device for input tensors
    # For multi-GPU, use the first device; for single GPU, use the specified device
    if use_multi_gpu:
        input_device = "cuda:0"  # First GPU for inputs
    else:
        input_device = device

    # Track results
    total_main_questions = 0
    correct_main_questions = 0
    total_sub_questions = 0
    correct_sub_questions = 0
    total_f1 = 0.0
    total_em = 0.0

    # Process each main question (on-demand cache generation)
    for example_id, q_data in enumerate(questions_data):
        print(f"\n{'='*80}")
        print(f"Main Question {example_id+1}/{len(questions_data)}: {q_data['main_question']}")
        print(f"{'='*80}")

        # Skip main questions that should not be tested
        if not q_data.get('should_test', True):
            print("⊘ SKIPPED (llm_judge=False or contains problematic answers)")
            print("  Note: Documents still included in global corpus for similarity computation")
            continue

        doc_tensors = q_data['doc_tensors']

        # Step 1: Generate KV cache for THIS main question's documents
        if rate != 1:  # Skip if full recompute
            # Generate system KV cache (chunk_id=0)
            system_cache_path = f'{save_path}/{example_id}_0_key.pt'
            #fixme: mengyao_debug
            if not os.path.exists(system_cache_path):
                print(f"Generating system KV cache...")
                input_tensor = system_tensor.unsqueeze(0)
                prefill_and_save_kv_cache(
                    model, tokenizer, past_key_values, input_tensor.to(input_device),
                    save_path=save_path, example_id=example_id, chunk_id=0,
                    system_len=system_len, passage_len=system_len,
                    reprocess_method=reprocess_method, device=input_device, device_map=device_map
                )

            # Generate KV cache for each document in THIS main question
            for doc_idx, doc_tensor in enumerate(doc_tensors):
                chunk_id = doc_idx + 1
                cache_key_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'

                if not os.path.exists(cache_key_path):
                    passage_len = doc_tensor.shape[0]
                    input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)

                    prefill_and_save_kv_cache(
                        model, tokenizer, past_key_values, input_tensor.to(input_device),
                        save_path=save_path, example_id=example_id, chunk_id=chunk_id,
                        system_len=system_len, passage_len=passage_len,
                        reprocess_method=reprocess_method, device=input_device, device_map=device_map
                    )
                    print(f"  Generated KV cache for document {chunk_id}/{len(doc_tensors)}")

        # Step 2: FusionRAG preprocess (if enabled)
        if preprocess and rate != 1:
            # Copy system cache (chunk_id=0)
            system_preprocess_key = f"{preprocess_save_path}/{example_id}_0_key.pt"
            if not os.path.exists(system_preprocess_key):
                shutil.copy(f'{save_path}/{example_id}_0_key.pt', system_preprocess_key)
                shutil.copy(f'{save_path}/{example_id}_0_value.pt', f"{preprocess_save_path}/{example_id}_0_value.pt")

            # Preprocess each document
            for doc_idx in range(len(doc_tensors)):
                chunk_id = doc_idx + 1
                preprocess_key_path = f"{preprocess_save_path}/{example_id}_{chunk_id}_key.pt"

                if os.path.exists(preprocess_key_path):
                    continue

                print(f"  Preprocessing document {chunk_id}/{len(doc_tensors)} with FusionRAG...")

                # Show retrieved similar documents
                if len(context_rank) > 0:
                    global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx
                    if global_doc_idx < len(context_rank):
                        similar_docs_info = []
                        for similar_global_idx in context_rank[global_doc_idx][:topk]:
                            # Skip invalid indices (from padding in PER_EXAMPLE mode)
                            if similar_global_idx < 0:
                                continue
                            if similar_global_idx == global_doc_idx:
                                continue
                            corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                            similar_chunk_id = c_id + 1
                            similar_cache_key_path = f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt"
                            cache_exists = os.path.exists(similar_cache_key_path)
                            status = "✓ cached" if cache_exists else "✗ need generate"
                            similar_docs_info.append(f"Q{corpus_i+1}-Doc{similar_chunk_id} ({status})")

                        if similar_docs_info:
                            print(f"    Retrieved similar docs: {', '.join(similar_docs_info)}")

                # STEP 1: Check and generate all required similar documents' cache FIRST
                # (to avoid past_key_values corruption during on-demand generation)
                if len(context_rank) > 0:
                    global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx

                    if global_doc_idx < len(context_rank):
                        for similar_global_idx in context_rank[global_doc_idx][:topk]:
                            # Skip invalid indices (from padding in PER_EXAMPLE mode)
                            if similar_global_idx < 0:
                                continue
                            if similar_global_idx == global_doc_idx:
                                continue

                            corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                            similar_chunk_id = c_id + 1

                            # Check and generate if needed
                            similar_cache_key_path = f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt"
                            if not os.path.exists(similar_cache_key_path):
                                print(f"      → On-demand: Generating cache for Q{corpus_i+1}-Doc{similar_chunk_id}...")

                                # Generate system cache for that question if needed
                                other_system_cache_path = f'{save_path}/{corpus_i}_0_key.pt'
                                if not os.path.exists(other_system_cache_path):
                                    other_input = system_tensor.unsqueeze(0)
                                    prefill_and_save_kv_cache(
                                        model, tokenizer, past_key_values, other_input.to(input_device),
                                        save_path=save_path, example_id=corpus_i, chunk_id=0,
                                        system_len=system_len, passage_len=system_len,
                                        reprocess_method=reprocess_method, device=input_device, device_map=device_map
                                    )

                                # Generate the document cache
                                similar_doc_tensor = questions_data[corpus_i]['doc_tensors'][c_id]
                                other_passage_len = similar_doc_tensor.shape[0]
                                other_input = torch.cat((system_tensor, similar_doc_tensor)).unsqueeze(0)

                                prefill_and_save_kv_cache(
                                    model, tokenizer, past_key_values, other_input.to(input_device),
                                    save_path=save_path, example_id=corpus_i, chunk_id=similar_chunk_id,
                                    system_len=system_len, passage_len=other_passage_len,
                                    reprocess_method=reprocess_method, device=input_device, device_map=device_map
                                )

                # STEP 2: Now load all required cache into past_key_values
                # Reset cache
                past_len = 0
                for layer_idx in range(len(past_key_values.key_cache)):
                    past_key_values.past_tokens[layer_idx] = 0

                # Load system KV cache
                corpus_passages = [system_tensor]
                system_key_cache = torch.load(f"{save_path}/{example_id}_0_key.pt", weights_only=True)
                system_value_cache = torch.load(f"{save_path}/{example_id}_0_value.pt", weights_only=True)

                for layer_idx in range(len(past_key_values.key_cache)):
                    past_key_values.key_cache[layer_idx].narrow(2, 0, system_len).copy_(system_key_cache[layer_idx])
                    past_key_values.value_cache[layer_idx].narrow(2, 0, system_len).copy_(system_value_cache[layer_idx])
                    past_key_values.past_tokens[layer_idx] += system_len
                past_len += system_len

                # Load topk similar documents' KV cache
                if len(context_rank) > 0:
                    global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx

                    if global_doc_idx < len(context_rank):
                        for similar_global_idx in context_rank[global_doc_idx][:topk]:
                            # Skip invalid indices (from padding in PER_EXAMPLE mode)
                            if similar_global_idx < 0:
                                continue
                            if similar_global_idx == global_doc_idx:
                                continue

                            corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                            similar_chunk_id = c_id + 1

                            # Load the similar document's cache (now guaranteed to exist)
                            similar_doc_tensor = questions_data[corpus_i]['doc_tensors'][c_id]
                            corpus_len = similar_doc_tensor.shape[0]
                            corpus_passages.append(similar_doc_tensor)

                            chunk_key_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt", weights_only=True)
                            chunk_value_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_value.pt", weights_only=True)

                            # Copy to past_key_values
                            for layer_idx in range(len(past_key_values.key_cache)):
                                past_key_values.key_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_key_cache[layer_idx])
                                past_key_values.value_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_value_cache[layer_idx])
                                past_key_values.past_tokens[layer_idx] += corpus_len
                            past_len += corpus_len

                # Add current document
                corpus_passages.append(doc_tensors[doc_idx])

                # Preprocess with fused KV cache
                prefill_with_cache_and_save_preprocess(
                    model, tokenizer, past_key_values, corpus_passages,
                    preprocess_save_path, example_id, chunk_id,
                    system_len=system_len, revert_rope=revert_rope,
                    reprocess_method=reprocess_method, device=input_device, device_map=device_map
                )

        # Step 3: Answer sub-questions
        all_sub_correct = True

        for sub_q_idx, sub_q_info in enumerate(q_data['sub_questions']):
            print(f"\nSub-question {sub_q_idx+1}/{len(q_data['sub_questions'])}")
            print(f"Question: {sub_q_info['query']}")
            print(f"Ground Truth: {sub_q_info['answer']}")

            # Build tokens: system + docs + question
            # Add /no_think for Qwen3 models to disable chain-of-thought
            if model_type == 'qwen3':
                question_text = f"<|im_end|>\n<|im_start|>user\n/no_think\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
            else:
                question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            question_tensor = torch.tensor(question_tokens, dtype=torch.long)

            # Get documents for this sub-question using chunk_ids
            doc_chunk_ids = sub_q_info['chunk_ids']  # List of chunk_ids (1-indexed) for documents
            sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]  # Convert to 0-indexed

            iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]

            # Prepare chunk_ids for load_kv_and_generate: [0, chunk_id1, chunk_id2, ...]
            # chunk_id 0 is system, then the actual document chunk_ids
            kv_chunk_ids = [0] + doc_chunk_ids

            # Generate answer using this main question's KV cache
            if rate == 1:
                # Full recompute
                inputs = torch.cat(iter_tokens).to(input_device).unsqueeze(0)
                from ktransformers.util.utils import prefill_and_generate
                generated_tokens, _, _ = prefill_and_generate(
                    model, tokenizer, inputs, max_new_tokens=500, device=input_device, device_map=device_map
                )
            else:
                # Load preprocessed KV cache and generate (FusionRAG, QueryAttention, DraftModel, etc.)
                load_path = preprocess_save_path if preprocess else save_path
                generated_tokens, _ = load_kv_and_generate(
                    model, tokenizer, past_key_values, iter_tokens, load_path, example_id,
                    max_new_tokens=500, revert_rope=revert_rope,
                    reprocess_method=reprocess_method, rate=rate,
                    draft_model=draft_model,  # DraftModel 方法会用到
                    use_entropy_selection=use_entropy_selection,
                    entropy_top_k=entropy_top_k,
                    preprocess=preprocess, device=input_device, chunk_ids=kv_chunk_ids, device_map=device_map,
                    device_draft_model=device_draft_model
                )

            # Decode answer
            answer = tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True)
            print(f"Predicted: {answer}")

            # Judge
            is_correct, judge_reason = judge_answer_with_openai(
                openai_client, openai_model,
                sub_q_info['query'], answer, sub_q_info['answer']
            )

            # Compute F1 and EM
            f1_score = compute_f1(answer, sub_q_info['answer'], tokenizer)
            em_score = 1.0 if _exact_match_score(answer, sub_q_info['answer']) else 0.0

            print(f"Judgment: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            print(f"F1: {f1_score:.4f}, EM: {em_score:.4f}")
            print(f"Reason: {judge_reason}")

            total_sub_questions += 1
            total_f1 += f1_score
            total_em += em_score
            if is_correct:
                correct_sub_questions += 1
            else:
                all_sub_correct = False

            print(f"rate={rate} sub_question({total_sub_questions}) acc rate={correct_sub_questions / total_sub_questions if total_sub_questions > 0 else 0}")

            # Save to CSV
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if rate != 1:
                    # Get rate=1 results for comparison
                    key = (q_data['main_question'], sub_q_info['query'])
                    rate1_data = rate1_results.get(key, {
                        'predicted': 'N/A',
                        'correct': 'N/A',
                        'f1': 'N/A',
                        'em': 'N/A',
                        'reason': 'N/A'
                    })
                    writer.writerow([
                        q_data['main_question'], sub_q_info['query'],
                        sub_q_info['answer'], answer, is_correct, f1_score, em_score, judge_reason,
                        rate1_data['predicted'], rate1_data['correct'],
                        rate1_data['f1'], rate1_data['em'], rate1_data['reason']
                    ])
                else:
                    writer.writerow([
                        q_data['main_question'], sub_q_info['query'],
                        sub_q_info['answer'], answer, is_correct, f1_score, em_score, judge_reason
                    ])

            torch.cuda.empty_cache()

        # Main question result
        total_main_questions += 1
        if all_sub_correct:
            correct_main_questions += 1
            print(f"\n✓ Main question {example_id+1}: ALL {len(q_data['sub_questions'])} sub-questions CORRECT")
        else:
            print(f"\n✗ Main question {example_id+1}: Some sub-questions INCORRECT")

    # Final results
    main_q_acc = correct_main_questions / total_main_questions if total_main_questions > 0 else 0
    sub_q_acc = correct_sub_questions / total_sub_questions if total_sub_questions > 0 else 0
    avg_f1 = total_f1 / total_sub_questions if total_sub_questions > 0 else 0
    avg_em = total_em / total_sub_questions if total_sub_questions > 0 else 0

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Main Questions: {correct_main_questions}/{total_main_questions} ({main_q_acc:.2%})")
    print(f"Sub Questions: {correct_sub_questions}/{total_sub_questions} ({sub_q_acc:.2%})")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average EM: {avg_em:.4f}")

    # Show comparison with rate=1 if applicable
    if rate != 1 and len(rate1_results) > 0:
        # Calculate rate=1 statistics
        rate1_correct = sum(1 for v in rate1_results.values() if v['correct'].lower() == 'true')
        rate1_total = len(rate1_results)
        rate1_acc = rate1_correct / rate1_total if rate1_total > 0 else 0
        rate1_avg_f1 = sum(float(v['f1']) for v in rate1_results.values()) / rate1_total if rate1_total > 0 else 0
        rate1_avg_em = sum(float(v['em']) for v in rate1_results.values()) / rate1_total if rate1_total > 0 else 0

        print(f"\n{'='*80}")
        print("COMPARISON WITH RATE=1 BASELINE")
        print(f"{'='*80}")
        print(f"Current (rate={rate}):")
        print(f"  Sub Questions Accuracy: {sub_q_acc:.2%}, F1: {avg_f1:.4f}, EM: {avg_em:.4f}")
        print(f"Baseline (rate=1):")
        print(f"  Sub Questions Accuracy: {rate1_acc:.2%}, F1: {rate1_avg_f1:.4f}, EM: {rate1_avg_em:.4f}")
        print(f"Delta:")
        print(f"  Accuracy: {sub_q_acc - rate1_acc:+.2%}, F1: {avg_f1 - rate1_avg_f1:+.4f}, EM: {avg_em - rate1_avg_em:+.4f}")
        print(f"{'='*80}")

    print(f"{'='*80}")

    with open(result_file, 'w') as f:
        f.write(f"Main Questions Accuracy: {correct_main_questions}/{total_main_questions} ({main_q_acc:.4f})\n")
        f.write(f"Sub Questions Accuracy: {correct_sub_questions}/{total_sub_questions} ({sub_q_acc:.4f})\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"Average EM Score: {avg_em:.4f}\n")

        # Add rate=1 comparison to file
        if rate != 1 and len(rate1_results) > 0:
            rate1_correct = sum(1 for v in rate1_results.values() if v['correct'].lower() == 'true')
            rate1_total = len(rate1_results)
            rate1_acc = rate1_correct / rate1_total if rate1_total > 0 else 0
            rate1_avg_f1 = sum(float(v['f1']) for v in rate1_results.values()) / rate1_total if rate1_total > 0 else 0
            rate1_avg_em = sum(float(v['em']) for v in rate1_results.values()) / rate1_total if rate1_total > 0 else 0

            f.write(f"\n--- Comparison with Rate=1 Baseline ---\n")
            f.write(f"Rate=1 Sub Questions Accuracy: {rate1_acc:.4f}\n")
            f.write(f"Rate=1 Average F1 Score: {rate1_avg_f1:.4f}\n")
            f.write(f"Rate=1 Average EM Score: {rate1_avg_em:.4f}\n")
            f.write(f"Accuracy Delta: {sub_q_acc - rate1_acc:+.4f}\n")
            f.write(f"F1 Delta: {avg_f1 - rate1_avg_f1:+.4f}\n")
            f.write(f"EM Delta: {avg_em - rate1_avg_em:+.4f}\n")

    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
    # DraftModel 方法: 用小模型指导大模型的 token 选择
    main(
        model_type='qwen3',
        model_path='/data2/qy_tmp/xumengyao/Qwen3-32B',
        draft_model_path='/data2/qy_tmp/xumengyao/Qwen2.5-3B-Instruct',  # Draft model for guidance
        data_path='./result_reflect.json',
        cache_path='/data2/qy_tmp/xumengyao/fusionrag/',
        model_name='Qwen3-32B',
        rate=0.2,  # 30% token selection
        topk=10,
        preprocess=False,
        use_entropy_selection=True,
        reprocess_method='DraftModel',  # 使用 Draft Model 指导的方法
        preprocess_scope=PreprocessScope.GLOBAL,
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        revert_rope=True,
        device="cuda:0",
        device_draft_model="cuda:3",
        use_multi_gpu=True,
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
        max_samples=200
    )

    # # QueryAttention 方法
    # main(
    #     model_type='qwen3',
    #     model_path='/mnt/data/models/Qwen3-32B',
    #     data_path='./result_reflect.json',
    #     cache_path='/mnt/data/reflect/',
    #     model_name='Qwen3-32B',
    #     rate=1,  # 30% token selection
    #     topk=10,
    #     preprocess=True,  # 开启预处理
    #     reprocess_method='QueryAttention',  # 测试 QueryAttention 方法
    #     preprocess_scope=PreprocessScope.GLOBAL,
    #     bge_model_path='/mnt/data/models/bge-m3-FP16',
    #     revert_rope=True,
    #     device="cuda:0",
    #     use_multi_gpu=True,  # Multi-GPU mode
    #     openai_base_url="https://api.deepseek.com/v1",
    #     openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    #     openai_model="deepseek-chat",
    #     max_samples=200  # Test samples
    #     )
    # main(
    #     model_type='qwen3',
    #     model_path='/mnt/data/models/Qwen3-32B',
    #     data_path='./result.json',
    #     cache_path='/mnt/data/junshi/',
    #     model_name='Qwen3-32B',
    #     rate=0.3,
    #     topk=10,
    #     preprocess=True,
    #     reprocess_method='FusionRAG',
    #     preprocess_scope=PreprocessScope.GLOBAL,
    #     bge_model_path='/mnt/data/models/bge-m3-FP16',
    #     revert_rope=True,
    #     device="cuda:0",
    #     use_multi_gpu=True,  # Set to True for multi-GPU (e.g., Qwen3-32B)
    #     openai_base_url="https://api.deepseek.com/v1",
    #     openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    #     openai_model="deepseek-chat",
    #     max_samples=200  # Test first 2 MAIN questions
    # )

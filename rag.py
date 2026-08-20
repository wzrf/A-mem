import os
import os
import json
import hashlib, re
import time

import numpy as np
import faiss
# from .sglang_kvcache import run_one_question_sglang
from typing import List, Dict, Any, Union
from openai import OpenAI
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import openai
import random
import requests

class OnlineEncoder:
    def __init__(self):
        self.embedding_model_name = os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-v4")
        llm_base_url = os.getenv("LLM_EMBEDDING_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        llm_api_key = os.getenv("EMBED_MODEL_KEY", "sk-11ce7640e46049a6977c0d96ba855ffb")
        if llm_api_key == "":
            raise ValueError("环境变量 EMBED_MODEL_KEY 未设置")
        self.client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)

    def encode(self, text: Union[str, List[str]], batch_size=10, normalize_embeddings=False,
               query_type="", max_concurrent_requests=10) -> np.ndarray:
        """将文本编码为向量。支持单文本或文本列表。"""
        prompt_prefixes = {
            'passage': 'Given a question, retrieve relevant documents that best answer the question.',
            'entity': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
            'edge': 'Given a question, retrieve relevant triplet facts that matches this question.',
            'fill_in_edge': 'Given a triples with only head and relation, retrieve relevant triplet facts that best fill the atomic query.'
        }
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        if query_type in prompt_prefixes:
            prefix = f"Instruct: {prompt_prefixes[query_type]}\nQuery: "
            texts = [prefix + t for t in texts]

        all_embeddings = []
        batch_size = min(batch_size, 10)  # 阿里云限制
        total_batches = (len(texts) + batch_size - 1) // batch_size

        def log_retry_attempt(retry_state):
            exception = retry_state.outcome.exception()
            print(
                f"[⚠️ 触发重试] 第 {retry_state.attempt_number} 次调用失败！"
                f"原因: {exception.__class__.__name__}: {exception} | "
                f"将在 {retry_state.next_action.sleep:.2f} 秒后重试..."
            )

        @retry(
            wait=wait_random_exponential(min=1, max=20),
            stop=stop_after_attempt(50),
            retry=retry_if_exception_type(openai.RateLimitError),
            before_sleep=log_retry_attempt,
            reraise=True  # 如果重试 5 次都失败了，把最后的异常抛出来
        )
        def process_batch_with_retry(batch_texts):
            return self.client.embeddings.create(
                input=batch_texts,
                model=self.embedding_model_name
            )

        def process_batch(batch_idx):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            # 调用带重试机制的函数
            resp = process_batch_with_retry(batch_texts)

            return [item.embedding for item in resp.data], batch_idx

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_concurrent_requests, total_batches)) as executor:
            futures = {executor.submit(process_batch, i): i for i in range(total_batches)}
            results = [None] * total_batches
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                emb, _ = fut.result()
                results[idx] = emb
            for emb_list in results:
                if emb_list is not None:
                    all_embeddings.extend(emb_list)

        arr = np.array(all_embeddings, dtype=np.float32)
        arr = np.ascontiguousarray(arr)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            arr = arr / norms
        return arr[0] if is_single else arr

    def get_sentence_embedding_dimension(self):
        return 1024  # 根据模型实际调整


def load_or_build_index(docs_path: str, encoder: OnlineEncoder, keyword: str):
    """
    根据文档文件，加载或构建 FAISS 索引（IndexIDMap）和 id2text 映射。
    只处理新增的文档（id 不在已有映射中的）。
    返回：index (faiss.IndexIDMap), id2text (dict)
    """
    index_file = f"./data/simplerag/{keyword}/faiss_index_{keyword}.faiss"
    id2text_file = f"./data/simplerag/{keyword}/id2text_{keyword}.json"

    # 读取所有文档，构建 id->text 字典（从文件最新内容）
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)          # 列表，每个元素有 "id" 和 "text"
    all_docs = {doc["id"]: doc["text"] for doc in docs}

    # 尝试加载已有索引和映射
    index = None
    id2text = {}
    if os.path.exists(index_file) and os.path.exists(id2text_file):
        print("加载缓存的 FAISS 索引和 id2text...")
        index = faiss.read_index(index_file)          # 读取 IndexIDMap 包装的索引
        with open(id2text_file, 'r', encoding='utf-8') as f:
            id2text = {int(k): v for k, v in json.load(f).items()}
        print(f"已加载 {len(id2text)} 个文档的向量。")
    else:
        print("未找到缓存，将构建新索引。")

    # 找出需要新增的文档（id 不在 id2text 中）
    existing_ids = set(id2text.keys())
    new_docs = [(doc_id, text) for doc_id, text in all_docs.items() if doc_id not in existing_ids]
    if not new_docs:
        print("没有新文档，直接使用现有索引。")
        return index, id2text

    print(f"发现 {len(new_docs)} 个新文档，开始编码...")
    new_ids = [doc_id for doc_id, _ in new_docs]
    new_texts = [text for _, text in new_docs]

    # 编码新文档
    new_embeddings = encoder.encode(new_texts, batch_size=10, normalize_embeddings=True)
    new_embeddings = np.ascontiguousarray(new_embeddings).astype(np.float32)

    # 如果索引不存在，先创建底层索引并包装为 IndexIDMap
    if index is None:
        dim = new_embeddings.shape[1]
        base_index = faiss.IndexFlatIP(dim)          # 内积（余弦相似度，因向量已归一化）
        index = faiss.IndexIDMap(base_index)
        print(f"创建新索引，维度 {dim}。")

    # 添加新向量及其 ID
    index.add_with_ids(new_embeddings, np.array(new_ids, dtype=np.int64))
    # 更新 id2text 映射
    id2text.update({doc_id: text for doc_id, text in zip(new_ids, new_texts)})
    print(f"索引更新完成，当前总文档数：{index.ntotal}")

    # 保存索引和 id2text 到文件
    faiss.write_index(index, index_file)
    with open(id2text_file, 'w', encoding='utf-8') as f:
        json.dump(id2text, f, ensure_ascii=False, indent=2)
    print("索引和 id2text 已保存到本地缓存。")

    return index, id2text


def init_locomo_indeces():
    indeces = []
    id2texts = []
    encoder = OnlineEncoder()
    keywords = []
    docs_paths = []
    for i in range(10):
        keywords.append(f"locomo_input_{i}")
        docs_paths.append(f"./data/input/locomo_short/locomo_input_{i}.json")

    for idx, docs_path in enumerate(docs_paths):
        keyword = keywords[idx]
        os.makedirs(f"./data/simplerag/{keyword}", exist_ok=True)
        index, id2text = load_or_build_index(docs_path, encoder, keyword)
        indeces.append(index)
        id2texts.append(id2text)
    return indeces, id2texts


def retrieve_single_question_debug(question, index, id2text, encoder, topk=10):
    # 编码问题
    q_emb = encoder.encode(question, normalize_embeddings=True, query_type='passage')
    q_emb = q_emb.reshape(1, -1).astype(np.float32)

    # 检索 top10 文档（返回自定义 ID）
    scores, ids = index.search(q_emb, topk)  # ids shape (1, 10)
    retrieved_ids = ids[0].tolist()
    # 通过 id2text 获取文本（忽略可能无效的 ID）
    retrieved_texts = [id2text[doc_id] for doc_id in retrieved_ids if doc_id in id2text]

    must_choose_docs = []

    return retrieved_texts


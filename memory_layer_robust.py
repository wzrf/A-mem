"""
Robust A-MEM memory layer — drop-in replacement for memory_layer.py.

Key differences from the original:
  - No response_format / JSON schema dependency in LLM calls
  - Plain-text prompts with section-marker parsing (via llm_text_parsers)
  - Structured logging instead of print()
  - Retry wrapper for transient LLM failures
  - Connectivity check on controller init
  - Graceful degradation: evolution failure -> memory stored without evolution
"""

from typing import List, Dict, Optional, Literal, Any
import json
import re
import uuid
import os
import time
import logging
import functools
from datetime import datetime
from abc import ABC, abstractmethod
from fusionrag.sglang_kvcache import run_one_question_sglang, run_one_question_origin_sglang

from memory_layer import SimpleEmbeddingRetriever, simple_tokenize
from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT,
    EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
    validate_analysis_result,
    ANALYZE_CONTENT_PROMPT_PREFIX,
    ANALYZE_CONTENT_PROMPT_QUERY, EVOLUTION_DECISION_PROMPT_PREFIX, EVOLUTION_DECISION_PROMPT_QUERY,
    STRENGTHEN_DETAILS_PROMPT_PREFIX, STRENGTHEN_DETAILS_PROMPT_QUERY, UPDATE_NEIGHBORS_PROMPT_PREFIX,
    UPDATE_NEIGHBORS_PROMPT_QUERY
)

logger = logging.getLogger("amem_robust")

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM call %s failed after %d attempts: %s",
                         func.__name__, max_retries + 1, last_exc)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Robust LLM Controllers — no response_format parameter
# ---------------------------------------------------------------------------

class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """Get a plain-text completion from the LLM."""
        pass

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion("Reply with exactly one word: READY", temperature=0.0)
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


class RobustOpenAIController(RobustBaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, sglang_host: str="", sglang_port: int=0, fusion_rag_model=None) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        self.model = model
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        if sglang_host == "":
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            base_url = f"{sglang_host}:{sglang_port}/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.sglang_url = f"{base_url}/completions"
        self.sglang_url_prefiller = f"{base_url}/completions"
        self.fusion_rag_model = fusion_rag_model
        if os.environ.get("FUSIONRAG", "false").lower() == "true":
            if fusion_rag_model is None:
                raise

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "thinking": False
                }
            }
        )
        return response.choices[0].message.content

    @retry_llm_call(max_retries=2)
    def get_completion_with_token(self, prompt: str, temperature: float = 0.7) -> (str, int, int):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "thinking": False
                }
            }
        )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

    def generate_response_with_fusionrag(
        self,
        system_prompt: str,
        prefix: str,
        fusionrag_cache_list: list[str],
        query_prompt: str,
        model: str="qwen3-8b",
        max_tokens = 5000
    ) -> (str, dict, dict):

        system_prompt = self.SYSTEM_MESSAGE
        template = {
            "DEFAULT_SYSTEM_PROMPT": f"""<|im_start|>system\n{system_prompt}\n{prefix}""",
            "USER_PROMPT": f"""<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query_prompt}<|im_end|>\n<|im_start|>assistant\nAnswer: </think>"""
        }

        fusionrag_cache_list_text = "".join(fusionrag_cache_list)
        system_len = len(self.fusion_rag_model.draft_model_tokenizer.encode(template["DEFAULT_SYSTEM_PROMPT"]))
        query_len = len(self.fusion_rag_model.draft_model_tokenizer.encode(template["USER_PROMPT"]))
        origin_text_list_len = len(self.fusion_rag_model.draft_model_tokenizer.encode(fusionrag_cache_list_text))

        recompute_tokens, recompute_tokens_list, retrieved_docs, recompute_rate, sorted_doc_index, sorted_doc_index_before, selected_indices = self.fusion_rag_model.draft_one_question(
            template["DEFAULT_SYSTEM_PROMPT"],  ## DEFAULT_SYSTEM_PROMPT
            fusionrag_cache_list,
            template["USER_PROMPT"],
            float(os.getenv("recompute_rate", "0.3")),
            "",
            False,
            False,
            [],
            False,
            False,  ## if do preprocess
            False,
            True
        )

        try:
            content, usage, top_logprobs, real_recomputation_rate = run_one_question_sglang(
                DEFAULT_SYSTEM_PROMPT=template["DEFAULT_SYSTEM_PROMPT"],
                USER_PROMPT=template["USER_PROMPT"],
                MODEL=model,
                retrived_docs=fusionrag_cache_list,
                max_tokens=max_tokens,  ## max tokens.
                retrived_docs_relevant_docs=[],
                recompute_tokens=recompute_tokens,
                recompute_tokens_list=recompute_tokens_list,
                max_workers=1,  ## max_workers.
                recomputation_rate=float(os.getenv("recompute_rate", "0.3")),
                model_use=model,
                endpoint_url=self.sglang_url,
                prefiller_endpoint_url=self.sglang_url_prefiller,
                method_keyword="",
            )

            usage_info = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }

            return content, usage_info, {
                "system_len": system_len,
                "query_len": query_len,
                "origin_text_list_len": origin_text_list_len,
                "fusionrag_text_list_len":  len(selected_indices),
            }
        except Exception as e:
            print(e)


class RobustOllamaController(RobustBaseLLMController):
    """Direct Ollama library controller (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama2"):
        self.model = model

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("ollama package not found. Install it with: pip install ollama")
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class RobustSGLangController(RobustBaseLLMController):
    def __init__(self, model: str = "llama2",
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 1000,
            }
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(f"SGLang server returned status {response.status_code}: {response.text}")

    # @retry_llm_call(max_retries=2)
    # def get_completion_with_token(
    #         self, prompt: str, temperature: float = 0.7
    # ) -> tuple[str, int, int]:
    #     payload = {
    #         "text": prompt,
    #         "sampling_params": {
    #             "temperature": temperature,
    #             "max_new_tokens": 1000,
    #             "no_think": True,  # 禁用 SGLang / DeepSeek-R1 的思考链输出
    #         },
    #     }
    #     response = self._requests.post(
    #         f"{self.base_url}/generate",
    #         headers={"Content-Type": "application/json"},
    #         json=payload,
    #         timeout=60,
    #     )
    #     if response.status_code == 200:
    #         data = response.json()
    #         # SGLang 原生接口在根节点或 meta_info 中返回 token 统计
    #         prompt_tokens = data.get("prompt_tokens", 0) or data.get(
    #             "meta_info", {}
    #         ).get("prompt_tokens", 0)
    #         completion_tokens = data.get("completion_tokens", 0) or data.get(
    #             "meta_info", {}
    #         ).get("completion_tokens", 0)
    #
    #         return data.get("text", ""), prompt_tokens, completion_tokens
    #
    #     raise RuntimeError(
    #         f"SGLang server returned status {response.status_code}: {response.text}"
    #     )


class RobustVLLMController(RobustBaseLLMController):
    """Controller for vLLM's OpenAI-compatible API server."""

    def __init__(self, model: str = "llama2",
                 vllm_host: str = "http://localhost",
                 vllm_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{vllm_host}:{vllm_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1000,
        }
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")


class RobustLiteLLMController(RobustBaseLLMController):
    """LiteLLM controller for universal LLM access (Ollama, SGLang, etc.)."""

    def __init__(self, model: str, api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        from litellm import completion as _completion
        self._completion = _completion
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if self.api_base:
            completion_args["api_base"] = self.api_base
        if self.api_key:
            completion_args["api_key"] = self.api_key

        response = self._completion(**completion_args)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang", "vllm"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 fusion_rag_model=None):
        if backend == "openai":
            self.llm = RobustOpenAIController(model, api_key, sglang_host, sglang_port, fusion_rag_model)
        elif backend == "ollama":
            self.llm = RobustOllamaController(model)
        elif backend == "sglang":
            self.llm = RobustSGLangController(model, sglang_host, sglang_port)
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', 'sglang', or 'vllm'")

        if check_connection:
            self.llm.check_connectivity()


# ---------------------------------------------------------------------------
# RobustMemoryNote
# ---------------------------------------------------------------------------

class RobustMemoryNote:
    """Memory note that uses plain-text LLM calls for metadata extraction."""

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[RobustLLMController] = None):

        self.content = content

        self.init_prompt_tokens = 0
        self.init_completion_tokens = 0
        self.init_fusionrag_stats = []
        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            analysis, self.init_prompt_tokens, self.init_completion_tokens, fusionrag_stats = self.analyze_content(content, llm_controller)
            if fusionrag_stats is not None:
                self.init_fusionrag_stats.append(fusionrag_stats)
            logger.debug("analysis result: %s", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []


    @staticmethod
    def analyze_content(content: str, llm_controller: RobustLLMController) -> (Dict, int, int, dict):
        """Analyze content using plain-text prompt + section-marker parsing."""
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        fusionrag_stats = None
        try:
            if os.environ.get("FUSIONRAG", "false").lower() == "true":
                response, usage_info, fusionrag_stats = llm_controller.llm.generate_response_with_fusionrag(
                    system_prompt="",
                    prefix=ANALYZE_CONTENT_PROMPT_PREFIX,
                    fusionrag_cache_list=[content],
                    query_prompt=ANALYZE_CONTENT_PROMPT_QUERY
                )
                prompt_tokens = usage_info["prompt_tokens"]
                completion_tokens = usage_info["completion_tokens"]
                fusionrag_stats["reuse_type"] = "reuse_prefill"
            else:
                response, prompt_tokens, completion_tokens = llm_controller.llm.get_completion_with_token(prompt)
            analysis = parse_analyze_content(response, content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.llm.get_completion(retry_prompt, temperature=0.3)
                from llm_text_parsers import _parse_list_items
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, content)
            return analysis, prompt_tokens, completion_tokens, fusionrag_stats

        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            # Graceful degradation: heuristic keywords/context
            from llm_text_parsers import _heuristic_keywords, _heuristic_context
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }, 0, 0, fusionrag_stats


# ---------------------------------------------------------------------------
# RobustAgenticMemorySystem
# ---------------------------------------------------------------------------

class RobustAgenticMemorySystem:
    """Memory management system using plain-text LLM calls (no JSON schema)."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 fusion_rag_model=None):

        self.memories: Dict[str, RobustMemoryNote] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = RobustLLMController(
            llm_backend, llm_model, api_key, api_base,
            sglang_host, sglang_port, check_connection,
            fusion_rag_model=fusion_rag_model
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    # ---- public API (mirrors AgenticMemorySystem) ----

    def add_note(self, content: str, time: str = None, **kwargs) -> (str, int, int):
        """Add a new memory note."""
        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            **kwargs,
        )
        evo_label, note, prompt_tokens, completion_tokens, fusionrag_stats_list = self.process_memory(note)
        fusionrag_stats_list.extend(note.init_fusionrag_stats)
        self.memories[note.id] = note
        self.retriever.add_documents([
            "content:" + note.content +
            " context:" + note.context +
            " keywords: " + ", ".join(note.keywords) +
            " tags: " + ", ".join(note.tags)
        ])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id, note.init_prompt_tokens + prompt_tokens, note.init_completion_tokens + completion_tokens, fusionrag_stats_list

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        self.retriever = SimpleEmbeddingRetriever(model_name)
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def find_related_memories(self, query: str, k: int = 5) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> (str, list):
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        memory_str_list = []
        chosen = []
        for i in indices:
            j = 0
            if i not in chosen:
                memory_str_ = (
                    "talk start time:" + all_memories[i].timestamp +
                    "memory content: " + all_memories[i].content +
                    "memory context: " + all_memories[i].context +
                    "memory keywords: " + str(all_memories[i].keywords) +
                    "memory tags: " + str(all_memories[i].tags) + "\n"
                )
                memory_str += memory_str_
                memory_str_list.append(memory_str_)
                chosen.append(i)
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                if neighbor < len(all_memories):
                    if neighbor not in chosen:
                        memory_str_ = (
                            "talk start time:" + all_memories[neighbor].timestamp +
                            "memory content: " + all_memories[neighbor].content +
                            "memory context: " + all_memories[neighbor].context +
                            "memory keywords: " + str(all_memories[neighbor].keywords) +
                            "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                        )
                        memory_str += memory_str_
                        memory_str_list.append(memory_str_)
                        chosen.append(neighbor)
                    if j >= k:
                        break
                    j += 1
        return memory_str, memory_str_list

    # ---- evolution (3 sequential plain-text calls) ----

    def process_memory(self, note: RobustMemoryNote) -> tuple:
        """Process a memory note for evolution using plain-text LLM calls.

        Uses up to 3 sequential calls (conditional):
          1. Evolution decision
          2. Strengthen details (skip if no strengthen)
          3. Update neighbors (skip if no update)
        """
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        prompt_tokens = 0
        completion_tokens = 0
        fusionrag_stats_list = []

        if len(indices) == 0:
            return False, note, 0, 0, fusionrag_stats_list

        try:
            # ---- Call 1: Evolution decision ----
            if os.environ.get("FUSIONRAG", "false").lower() == "true":
                decision_response, usage, fusionrag_stats = self.llm_controller.llm.generate_response_with_fusionrag(
                    system_prompt="",
                    prefix=EVOLUTION_DECISION_PROMPT_PREFIX,
                    fusionrag_cache_list=[
                        note.context,
                        "Content: " + note.content,
                        "Keywords: " + "".join(note.keywords),
                        "Nearest neighbor memories: " + neighbor_memory
                    ],
                    query_prompt=EVOLUTION_DECISION_PROMPT_QUERY
                )
                fusionrag_stats["reuse_type"] = "reuse_mix"
                fusionrag_stats_list.append(fusionrag_stats)
                prompt_tokens_1 = usage["prompt_tokens"]
                completion_tokens_1 = usage["completion_tokens"]

            else:
                decision_prompt = EVOLUTION_DECISION_PROMPT.format(
                    context=note.context,
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                decision_response, prompt_tokens_1, completion_tokens_1 = self.llm_controller.llm.get_completion_with_token(decision_prompt)

            prompt_tokens += prompt_tokens_1
            completion_tokens += completion_tokens_1

            decision = parse_evolution_decision(decision_response)
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False, note, 0, 0, fusionrag_stats_list

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            # ---- Call 2: Strengthen details (conditional) ----
            if should_strengthen:
                if os.environ.get("FUSIONRAG", "false").lower() == "true":
                    strengthen_response, usage, fusionrag_stats = self.llm_controller.llm.generate_response_with_fusionrag(
                        system_prompt="",
                        prefix=STRENGTHEN_DETAILS_PROMPT_PREFIX,
                        fusionrag_cache_list=[
                            "Content: " + note.content,
                            "Keywords: " + "".join(note.keywords),
                            "Nearest neighbor memories:\n" + neighbor_memory
                        ],
                        query_prompt=STRENGTHEN_DETAILS_PROMPT_QUERY
                    )
                    fusionrag_stats["reuse_type"] = "reuse_mix"
                    fusionrag_stats_list.append(fusionrag_stats)
                    prompt_tokens_2 = usage["prompt_tokens"]
                    completion_tokens_2 = usage["completion_tokens"]

                else:
                    strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                        content=note.content,
                        keywords=note.keywords,
                        nearest_neighbors_memories=neighbor_memory,
                    )
                    strengthen_response, prompt_tokens_2, completion_tokens_2 = self.llm_controller.llm.get_completion_with_token(strengthen_prompt)

                prompt_tokens += prompt_tokens_2
                completion_tokens += completion_tokens_2
                strengthen = parse_strengthen_details(strengthen_response)
                logger.debug("Strengthen details: %s", strengthen)

                note.links.extend(strengthen["connections"])
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            # ---- Call 3: Update neighbors (conditional) ----
            if should_update:
                if os.environ.get("FUSIONRAG", "false").lower() == "true":
                    update_response, usage, fusionrag_stats = self.llm_controller.llm.generate_response_with_fusionrag(
                        system_prompt="",
                        prefix=UPDATE_NEIGHBORS_PROMPT_PREFIX,
                        fusionrag_cache_list=[
                            "Content: " + note.content,
                            "Context: " + note.context,
                            "Nearest neighbor memories:\n" + neighbor_memory
                        ],
                        query_prompt=UPDATE_NEIGHBORS_PROMPT_QUERY.format(
                            max_neighbor_idx=len(indices) - 1,
                            neighbor_count=len(indices),
                        )
                    )
                    fusionrag_stats["reuse_type"] = "reuse_mix"
                    fusionrag_stats_list.append(fusionrag_stats)
                    prompt_tokens_3 = usage["prompt_tokens"]
                    completion_tokens_3 = usage["completion_tokens"]

                else:
                    update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                        content=note.content,
                        context=note.context,
                        nearest_neighbors_memories=neighbor_memory,
                        max_neighbor_idx=len(indices) - 1,
                        neighbor_count=len(indices),
                    )
                    update_response, prompt_tokens_3, completion_tokens_3 = self.llm_controller.llm.get_completion_with_token(update_prompt)

                prompt_tokens += prompt_tokens_3
                completion_tokens += completion_tokens_3
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                logger.debug("Neighbor updates: %s", neighbor_updates)

                noteslist = list(self.memories.values())
                notes_id = list(self.memories.keys())
                for i in range(min(len(indices), len(neighbor_updates))):
                    upd = neighbor_updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(noteslist):
                        continue
                    notetmp = noteslist[memorytmp_idx]
                    if upd["tags"]:
                        notetmp.tags = upd["tags"]
                    if upd["context"]:
                        notetmp.context = upd["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note, prompt_tokens, completion_tokens, fusionrag_stats_list

        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note, prompt_tokens, completion_tokens, fusionrag_stats_list

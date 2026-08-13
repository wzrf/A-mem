import requests, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import time
import hashlib

# prompt_postfix = "<｜Assistant｜>"
PROMPTS = {
    "DeepSeek-V3.2": {
        "DEFAULT_SYSTEM_PROMPT": """<｜begin▁of▁sentence｜>system\nYou are a helpful assistant. Write a high-quality answer for the given question using only the provided search results. The answer process requires reference to the material content and step-by-step thinking.\n""",
        "USER_PROMPT": """<｜end▁of▁sentence｜>\n\n<｜User｜>{query_prompt}<｜Assistant｜>\n""",
        # "USER_PROMPT": """<｜end▁of▁sentence｜>\n\n<｜User｜>{query_prompt}\n"""
    },
    "Qwen2.5-7B-Instruct": {
        "DEFAULT_SYSTEM_PROMPT": """<|im_start|>system\nYou are a helpful assistant.\nWrite a brief and high-quality answer for the given question using only the provided search results.\n""",
        "USER_PROMPT": """<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query_prompt}<|im_end|>\n<|im_start|>assistant\nAnswer: """
    },
    "Qwen3-8B": {
        "DEFAULT_SYSTEM_PROMPT": """<|im_start|>system\nYou are a helpful assistant.\nWrite a brief and high-quality answer for the given question using only the provided search results.\n""",
        "USER_PROMPT": """<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query_prompt}<|im_end|>\n<|im_start|>assistant</think>\nAnswer: """
    },
    "Kimi-K2.5": {
        "DEFAULT_SYSTEM_PROMPT": """<|im_system|>system<|im_middle|>\nYou are a helpful assistant. Write a high-quality answer for the given question using only the provided search results. The answer process requires reference to the material content and step-by-step thinking.\n""",
        "USER_PROMPT": """<|im_end|><|im_user|>user<|im_middle|>{query_prompt}<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"""
    },
}

SHORT_PROMPTS = {
    "Qwen3-8B": {
        "DEFAULT_SYSTEM_PROMPT": "<|im_start|>system\nYou are a helpful assistant.\nWrite a high-quality answer for the given question using only the provided search results. You should come out the answer without any other words. The answer process requires reference to the material content and step-by-step thinking.\n\nFor example:\nDocument (Title: Ed Wood) Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.\nDocument (Title: Scott Derrickson) Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as 'Sinister', 'The Exorcism of Emily Rose', and 'Deliver Us From Evil', as well as the 2016 Marvel Cinematic Universe installment, 'Doctor Strange.'\nQuestion: Were Scott Derrickson and Ed Wood of the same nationality?\nAnswer: yes\n\nDocument (Title: Shirley Temple) Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nDocument (Title: Kiss and Tell (1945 film)) Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.\nQuestion: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nAnswer: Chief of Protocol\n\nDocument (Title: Animorphs) Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, innocence, leadership, freedom and growing up are the core themes of the series.\nDocument (Title: The Hork-Bajir Chronicles) The Hork-Bajir Chronicles is the second companion book to the 'Animorphs' series, written by K. A. Applegate. With respect to continuity within the series, it takes place before book #23, 'The Pretender', although the events told in the story occur between the time of 'The Ellimist Chronicles' and 'The Andalite Chronicles'. The book is introduced by Tobias, who flies to the valley of the free Hork-Bajir, where Jara Hamee tells him the story of how the Yeerks enslaved the Hork-Bajir, and how Aldrea, an Andalite, and her companion, Dak Hamee, a Hork-Bajir, tried to save their world from the invasion. Jara Hamee's story is narrated from the points of view of Aldrea, Dak Hamee, and Esplin 9466, alternating in similar fashion to the 'Megamorphs' books.\nQuestion: What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?\nAnswer: Animorphs\n\nThe following is reference material:\n\n\n",
        "USER_PROMPT": "\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {query_prompt}\nAnswer:\n<|im_end|>\n<|im_start|>assistant\n",
    },
}

SPECIAL_WORDS = {
    "DeepSeek-V3.2".lower(): "<｜begin▁of▁sentence｜>",
    "Qwen".lower(): "<|im_start|>",
}

proprocess_key_hash_exemption = "skip"


def replace_keywords(draftmodel: str, runmodel: str, recompute_tokens: list[str]):
    words_to_be_replace = SPECIAL_WORDS[draftmodel]
    words_to_replace = SPECIAL_WORDS[runmodel]
    recompute_tokens_new = []
    for recompute_token in recompute_tokens:
        recompute_token_ = recompute_token.replace(words_to_be_replace, words_to_replace)
        recompute_tokens_new.append(recompute_token_)


def get_model_and_prompt(model: str):
    print(f"model=\n{model}")
    if model.lower() == "DeepSeek-V3.2".lower():
        return ("DeepSeek-V3.2",
                PROMPTS["DeepSeek-V3.2"]["DEFAULT_SYSTEM_PROMPT"],
                PROMPTS["DeepSeek-V3.2"]["USER_PROMPT"])
    elif model.lower() == "Qwen2.5-7B-Instruct".lower():
        return ("Qwen2.5-7B-Instruct",
                PROMPTS["Qwen2.5-7B-Instruct"]["DEFAULT_SYSTEM_PROMPT"],
                PROMPTS["Qwen2.5-7B-Instruct"]["USER_PROMPT"])
    elif model.lower() == "Kimi-K2.5".lower():
        return ("Kimi-K2.5",
                PROMPTS["Kimi-K2.5"]["DEFAULT_SYSTEM_PROMPT"],
                PROMPTS["Kimi-K2.5"]["USER_PROMPT"])
    elif model.lower() == "Qwen3-8B".lower():
        return ("Qwen3-8B",
                PROMPTS["Qwen3-8B"]["DEFAULT_SYSTEM_PROMPT"],
                PROMPTS["Qwen3-8B"]["USER_PROMPT"])

def get_model_and_prompt_concise(model: str):
    print(f"model=\n{model}")
    return ("Qwen3-8B",
     SHORT_PROMPTS["Qwen3-8B"]["DEFAULT_SYSTEM_PROMPT"],
     SHORT_PROMPTS["Qwen3-8B"]["USER_PROMPT"])


def run_raw_cache(prompt: str, prefix_prompt: str, MODEL: str, prompt_list: list[str], prefix_prompt_list: list[str], endpoint_url: str):
    # API端点
    url = endpoint_url
    # print(f"[run_raw_cache] for prompt=\n{prompt} prefix_prompt=\n{prefix_prompt}")

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求数据
    data = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 0,
        "fusionrag_params": {
            "save_cache": True,
            "save_raw_cache": True,
            "save_preprocess_cache": False,
            "prefix_prompt": prefix_prompt,
            "prompt_list": prompt_list,
            "prefix_prompt_list": prefix_prompt_list
        },
        "temperature": 0.8
    }

    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, json=data)

        # 检查响应状态
        if response.status_code == 200:
            ""
            # print(f"raw cache 生成 请求成功, prompt={prompt[:15]}\nprefix_prompt={prefix_prompt[:15]}\n"
            #       f"prompt_list={len(prompt_list)}\nprefix_prompt_list={len(prefix_prompt_list)}")
        else:
            print(f"请求失败，状态: {response.json()}")
    except Exception as e:
        print(e)

def run_preprocess_cache(prompt: str, prefix_prompt: str, MODEL: str, prompt_list: list[str],
                         prefix_prompt_list: list[str], endpoint_url:str, preprocess_cache_key: str, recompute_tokens=None):
    # API端点
    url = endpoint_url
    # print(f"[run_preprocess_cache] for prompt=\n{prompt} prefix_prompt=\n{prefix_prompt}")

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求数据
    data = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 0,
        "fusionrag_params": {
            "save_cache": True,
            "save_raw_cache": False,
            "save_preprocess_cache": True,
            "prefix_prompt": prefix_prompt,
            "prompt_list": prompt_list,
            "prefix_prompt_list": prefix_prompt_list,
            "preprocess_cache_key": preprocess_cache_key
        },
        "temperature": 0.8
    }

    if recompute_tokens is not None:
        data["fusionrag_params"]["recompute_tokens"] = recompute_tokens

    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, json=data)

        # 检查响应状态
        if response.status_code == 200:
            ""
            # print(f"preprocess cache 生成 请求成功, prompt={prompt[:15]}\nprefix_prompt={prefix_prompt[:15]}\n"
            #       f"prompt_list={len(prompt_list)}\nprefix_prompt_list={len(prefix_prompt_list)}")
        else:
            print(f"请求失败，状态: {response.json()}")
    except Exception as e:
        print(e)

def run_gen_original(
        prompt: str,
        max_tokens:int,
        MODEL: str,
        endpoint_url: str
) :
    url = endpoint_url
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0
    }

    try:
        # 发送POST请求
        time_start = time.time()
        response = requests.post(url, headers=headers, json=data)
        time_end = time.time()

        # 检查响应状态
        if response.status_code == 200:
            ""
            # print("请求成功！")
            # print("响应内容:")
            # print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        else:
            print(f"请求失败，状态: {response.json()}")
        response = response.json()
        response["usage"]["generation_time"] = time_end - time_start
        return (response["choices"][0]["text"], response["usage"], [],
                response["choices"][0].get("metadata", {}).get("recomputation_rate", 1))
    except Exception as e:
        print(e)

def run_gen(prompt: str, prefix_prompt:str, max_tokens:int, rope:bool, use_fusion_rag:bool,
            load_preprocess_cache:bool, load_raw_cache:bool, recompute_tokens: list[str], MODEL: str,
            prompt_list: list[str], prefix_prompt_list: list[str], endpoint_url: str,
            preprocess_cache_key_list: list[str]=None,
            cache_is_preprocess_list: list[bool]=None) :
    url = endpoint_url
    headers = {
        "Content-Type": "application/json"
    }

    # print(f"[run_gen] prompt={prompt}\n")

    # 请求数据
    if use_fusion_rag:
        data = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "fusionrag_params":{
                "save_cache": False,
                "load_preprocess_cache": load_preprocess_cache,
                "load_raw_cache": load_raw_cache,
                "rope": rope,
                "use_fusion_rag": use_fusion_rag,
                "prefix_prompt": prefix_prompt,
                "recompute_tokens": recompute_tokens,
                "prompt_list": prompt_list,
                "prefix_prompt_list": prefix_prompt_list
                },
            "temperature": 0.0,
            "logprobs": 10
        }
        if len(recompute_tokens) == 0:
            del data["fusionrag_params"]["recompute_tokens"]
        if preprocess_cache_key_list is not None:
            data["fusionrag_params"]["preprocess_cache_key_list"] = preprocess_cache_key_list
        if cache_is_preprocess_list is not None:
            data["fusionrag_params"]["cache_is_preprocess_list"] = cache_is_preprocess_list
    else:
        data = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }

    try:
        # 发送POST请求
        time_start = time.time()
        response = requests.post(url, headers=headers, json=data)
        time_end = time.time()

        # 检查响应状态
        if response.status_code == 200:
            ""
            # print("请求成功！")
            # print("响应内容:")
            # print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        else:
            print(f"请求失败，状态: {response.json()}")
        response = response.json()
        response["usage"]["generation_time"] = time_end - time_start
        return (response["choices"][0]["text"], response["usage"], response["choices"][0]["logprobs"],
                response["choices"][0].get("metadata", {}).get("recomputation_rate", -1))
        # return response["choices"][0]["text"], response["usage"], []
    except Exception as e:
        print(e)

def run_one_question_sglang_preprocess(
        query_prompt: str,
        retrived_docs: list[str],
        max_tokens:int,
        retrived_docs_relevant_docs: list[list[str]],
        recompute_tokens: list[str],
        recompute_tokens_list: list[list[str]],
        max_workers:int,
        recomputation_rate:float,
        model_use:str,
        endpoint_url:str,
        prefiller_endpoint_url:str,
        method_keyword:str,
        preprocess_cache_key_list: list[str],
        cache_is_preprocess_list: list[bool],
        full_recompute_preprocess: bool,
):
    if not full_recompute_preprocess:
        assert all(x == "" for x in preprocess_cache_key_list)
    else:
        assert preprocess_cache_key_list[0] == ""
        assert preprocess_cache_key_list[1] == ""
        assert all(x != "" for x in preprocess_cache_key_list[2:])
    ##fixme: 如果是多副本的话，必然使用full_recompute_preprocess=True，也就是preprocess要完全重算所有kv，并且有md5前缀；
    ## 如果不是的话，就是老版本preprocess，只重算最后的doc的kvcache，并且无md5前缀
    assert len(cache_is_preprocess_list) == len(retrived_docs) + 1 ## 第一个是systemprompt
    assert len(cache_is_preprocess_list) == len(preprocess_cache_key_list)
    if max_tokens <= 100:
        MODEL, DEFAULT_SYSTEM_PROMPT, USER_PROMPT = get_model_and_prompt_concise(model_use)
    else:
        MODEL, DEFAULT_SYSTEM_PROMPT, USER_PROMPT = get_model_and_prompt(model_use)
    # 1. run raw cache for system prompt
    run_raw_cache(prompt=DEFAULT_SYSTEM_PROMPT, prefix_prompt="", MODEL=MODEL, prefix_prompt_list=[], prompt_list=[DEFAULT_SYSTEM_PROMPT], endpoint_url=prefiller_endpoint_url)
    # 2. run raw cache for system prompt
    run_preprocess_cache(prompt=DEFAULT_SYSTEM_PROMPT, prefix_prompt="", MODEL=MODEL, prefix_prompt_list=[], prompt_list=[DEFAULT_SYSTEM_PROMPT], endpoint_url=prefiller_endpoint_url,
                         preprocess_cache_key="")
    # 3. raw cache for each doc
    for doc in retrived_docs:
        run_raw_cache(
            prompt=DEFAULT_SYSTEM_PROMPT+doc,
            prefix_prompt=DEFAULT_SYSTEM_PROMPT,
            MODEL=MODEL,
            prefix_prompt_list=[DEFAULT_SYSTEM_PROMPT],
            prompt_list=[DEFAULT_SYSTEM_PROMPT, doc], ## bug
            endpoint_url=prefiller_endpoint_url
        )
    for idx, doc in enumerate(retrived_docs):
        if not cache_is_preprocess_list[idx + 1]:
            continue

        # 4. raw cache for each relevant doc
        for relevant_doc in retrived_docs_relevant_docs[idx]:
            run_raw_cache(
                prompt=DEFAULT_SYSTEM_PROMPT + relevant_doc,
                prefix_prompt=DEFAULT_SYSTEM_PROMPT,
                MODEL=MODEL,
                prefix_prompt_list=[DEFAULT_SYSTEM_PROMPT],
                prompt_list=[DEFAULT_SYSTEM_PROMPT, relevant_doc],
                endpoint_url=prefiller_endpoint_url,
            )

        relevant_docs = copy.deepcopy(retrived_docs_relevant_docs[idx])
        ## 去重复，保持顺序
        # relevant_docs = list(dict.fromkeys(relevant_docs))
        prefix_prompt_list = [DEFAULT_SYSTEM_PROMPT]
        prefix_prompt_list.extend(relevant_docs)
        prompt_list = copy.deepcopy(prefix_prompt_list)
        prompt_list.append(doc)
        ##fixme： preprocess_cache_key是前缀的hash。 recompute token第0、2、4.。。是不用重算的token，1、3、5.。。是需要重算的，所以这么写一下。
        recompute_tokens_ = None
        if full_recompute_preprocess:
            recompute_tokens_ = ["", "".join(prefix_prompt_list)]
        # print(f"generate preprocess kvcache\n prompt_list={prompt_list}\n prefix_prompt_list={prefix_prompt_list}")
        run_preprocess_cache(
            prompt="".join(prompt_list),
            prefix_prompt="".join(prefix_prompt_list),
            MODEL=MODEL,
            prompt_list=prompt_list,
            prefix_prompt_list=prefix_prompt_list,
            endpoint_url=endpoint_url,
            preprocess_cache_key=preprocess_cache_key_list[idx + 1],
            recompute_tokens=recompute_tokens_
        )

    prefix_prompt_list = [DEFAULT_SYSTEM_PROMPT]
    prefix_prompt_list.extend(retrived_docs)
    prompt_list = copy.deepcopy(prefix_prompt_list)
    user_prompt = USER_PROMPT.format(query_prompt=query_prompt)
    prompt_list.append(user_prompt)

    if recomputation_rate == 1.0:
        return run_gen(
            prompt="".join(prompt_list),
            prefix_prompt="",
            prefix_prompt_list=[],
            prompt_list=prompt_list,
            max_tokens=max_tokens,
            rope=True,
            use_fusion_rag=True,
            load_raw_cache=True,
            load_preprocess_cache=True, ## doesn't matter
            recompute_tokens=recompute_tokens,
            MODEL=MODEL,
            endpoint_url=endpoint_url
        )
    return run_gen(
        prompt="".join(prompt_list),
        prefix_prompt="".join(prefix_prompt_list),
        prefix_prompt_list=prefix_prompt_list,
        prompt_list=prompt_list,
        max_tokens=max_tokens,
        rope=True,
        use_fusion_rag=True,
        load_raw_cache=False,
        load_preprocess_cache=True,
        recompute_tokens=recompute_tokens,
        MODEL=MODEL,
        endpoint_url=endpoint_url,
        preprocess_cache_key_list=preprocess_cache_key_list,
        cache_is_preprocess_list=cache_is_preprocess_list
    )

def run_one_question_sglang(
        query_prompt: str,
        retrived_docs: list[str],
        max_tokens:int,
        retrived_docs_relevant_docs: list[list[str]],
        recompute_tokens: list[str],
        recompute_tokens_list: list[list[str]],
        max_workers=1,
        recomputation_rate=0.0,
        model_use="",
        endpoint_url="",
        prefiller_endpoint_url="",
        method_keyword=""
):
    question_up_front = False
    MODEL, DEFAULT_SYSTEM_PROMPT, USER_PROMPT = get_model_and_prompt(model_use)
    user_prompt = USER_PROMPT.format(query_prompt=query_prompt)
    run_raw_cache(
        prompt=DEFAULT_SYSTEM_PROMPT,
        prefix_prompt="", MODEL=MODEL,
        prefix_prompt_list=[],
        prompt_list=[DEFAULT_SYSTEM_PROMPT],
        endpoint_url=prefiller_endpoint_url
    )

    for doc in retrived_docs:
        prefix_prompt_list = [DEFAULT_SYSTEM_PROMPT]
        prompt_list = [DEFAULT_SYSTEM_PROMPT, doc]
        run_raw_cache(
            prompt="".join(prompt_list),
            prefix_prompt="".join(prefix_prompt_list),
            MODEL=MODEL,
            prefix_prompt_list=prefix_prompt_list,
            prompt_list=prompt_list,
            endpoint_url=prefiller_endpoint_url
        )

    print(f"cache 生成 请求成功！")

    if question_up_front:
        user_prompt = user_prompt

        ## generate user prompt cache.
        run_raw_cache(prompt=DEFAULT_SYSTEM_PROMPT + user_prompt,
                      prefix_prompt=DEFAULT_SYSTEM_PROMPT,
                      prompt_list=[DEFAULT_SYSTEM_PROMPT, user_prompt],
                      prefix_prompt_list=[DEFAULT_SYSTEM_PROMPT],
                      MODEL=MODEL,
                      endpoint_url=prefiller_endpoint_url
                      )

        prefix_prompt_list = [DEFAULT_SYSTEM_PROMPT]
        prefix_prompt_list.append(user_prompt)
        prefix_prompt_list.extend(retrived_docs)
        prompt_list = copy.deepcopy(prefix_prompt_list)
        # prompt_list.append(prompt_postfix)

    else:
        prefix_prompt_list = [DEFAULT_SYSTEM_PROMPT]
        prefix_prompt_list.extend(retrived_docs)
        prompt_list = copy.deepcopy(prefix_prompt_list)
        prompt_list.append(user_prompt)
        # prompt_list.append(prompt_postfix)

    ## if full recompute, set prefix_prompt = 0, so that full recompute.
    if recomputation_rate == 1.0:
        return run_gen(
            prompt="".join(prompt_list),
            prefix_prompt="",
            prefix_prompt_list=[], ## prefix is empty
            prompt_list=prompt_list,
            max_tokens=max_tokens,
            rope=True,
            use_fusion_rag=True,
            load_raw_cache=True,
            load_preprocess_cache=False,
            recompute_tokens=recompute_tokens,
            MODEL=MODEL,
            endpoint_url=endpoint_url
        )
    return run_gen(
        prompt="".join(prompt_list),
        prefix_prompt="".join(prefix_prompt_list),
        prefix_prompt_list=prefix_prompt_list,
        prompt_list=prompt_list,
        max_tokens=max_tokens,
        rope=True,
        use_fusion_rag=True,
        load_raw_cache=True,
        load_preprocess_cache=False,
        recompute_tokens=recompute_tokens,
        MODEL=MODEL,
        endpoint_url=endpoint_url
    )

def run_one_question_origin_sglang(
        query_prompt: str,
        retrived_docs: list[str],
        max_tokens:int,
        model_use="",
        endpoint_url="",
):
    if max_tokens <= 100:
        MODEL, DEFAULT_SYSTEM_PROMPT, USER_PROMPT = get_model_and_prompt_concise(model_use)
    else:
        MODEL, DEFAULT_SYSTEM_PROMPT, USER_PROMPT = get_model_and_prompt(model_use)
    user_prompt = USER_PROMPT.format(query_prompt=query_prompt)
    prompt_list = [DEFAULT_SYSTEM_PROMPT]
    prompt_list.extend(retrived_docs)
    prompt_list.append(user_prompt)
    return run_gen_original(
        prompt="".join(prompt_list),
        max_tokens=max_tokens,
        MODEL=MODEL,
        endpoint_url=endpoint_url
    )



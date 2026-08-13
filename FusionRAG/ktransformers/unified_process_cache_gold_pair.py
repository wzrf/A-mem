"""
Unified process cache script supporting multiple models (Mistral, PanGu, Qwen, Llama)
Special version for hotpotqa/triviaqa: first 2 passages mutually recall each other, remaining 8 passages recall nothing
"""
import shutil
import torch
import os
import csv
import sys
from transformers import (
    AutoTokenizer,
    AutoConfig,
)

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import (
    prefill_and_generate, prefill_and_save_kv_cache, load_kv_and_generate,
    rotate_half, prefill_with_cache_and_save_preprocess,
    prepare_data, _exact_match_score, _metric_max_over_ground_truths,
    find_group_and_index, remove_unused_tokens, compute_f1
)
from ktransformers.models.custom_cache import StaticCache


def load_model(model_type, model_path, config):
    """
    Load model based on model type

    Args:
        model_type: one of ['mistral', 'pangu', 'qwen', 'llama']
        model_path: path to model
        config: model config

    Returns:
        model instance
    """
    if model_type == 'mistral':
        from models.modeling_mistral import MistralForCausalLM
        with torch.no_grad():
            model = MistralForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
    elif model_type == 'pangu':
        from models.modeling_openpangu_dense import PanguEmbeddedForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = PanguEmbeddedForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
    elif model_type == 'qwen':
        from models.modeling_qwen2 import Qwen2ForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = Qwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
    elif model_type == 'llama':
        from models.modeling_llama import LlamaForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = LlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def post_process_answer(answer, model_type):
    """
    Post-process answer based on model type

    Args:
        answer: raw answer string
        model_type: one of ['mistral', 'pangu', 'qwen', 'llama']

    Returns:
        processed answer string
    """
    if model_type == 'pangu':
        answer = remove_unused_tokens(answer)
        if "Answer:" in answer:
            answer = answer.split('Answer:')[1]
    elif model_type == 'llama':
        answer = answer.split('[/INST]')[0]
    # mistral and qwen don't need special processing
    return answer


def compute_score(answer, real_answer_list, tokenizer, model_type):
    """
    Compute exact match score based on model type

    Args:
        answer: predicted answer
        real_answer_list: list of ground truth answers
        tokenizer: tokenizer instance
        model_type: one of ['mistral', 'pangu', 'qwen', 'llama']

    Returns:
        exact match score
    """
    if model_type == 'llama':
        return max([compute_f1(answer, real_answer, tokenizer) for real_answer in real_answer_list])
    else:
        return max([_exact_match_score(answer, real_answer) for real_answer in real_answer_list])


def prepare_data_gold_pair(model_name, data_path, data_name, cache_path, tokenizer: AutoTokenizer,
                 topk: int, revert_rope, preprocess, bge_model_path='/mnt/data/models/bge-m3-FP16'):
    """
    Modified prepare_data for gold pair recall:
    - Don't shuffle passages for hotpotqa/triviaqa
    - First 2 passages mutually recall each other
    - Remaining 8 passages recall nothing
    """
    import json
    import re
    import numpy as np
    from ktransformers.util.utils import _rouge1_score, _rougel_score

    prompt_config = json.load(open('./config/dataset2prompt_few-shot.json'))
    data_path = data_path+data_name
    if data_name in ['2wikimqa.jsonl', 'samsum.jsonl', 'multi_news.jsonl', 'musique.jsonl', 'hotpotqa.jsonl', 'triviaqa.jsonl']:
        data_name_prefix = data_name.split('.')[0]
    else:
        data_name_prefix = data_name.split('-')[0]
    if data_name_prefix in ['hotpotqa','triviaqa','2wikimqa','musique']:
        rouge_metrics = _rouge1_score
        max_tokens_length = 50
    elif data_name_prefix in ['samsum','multi_news']:
        rouge_metrics = _rougel_score
        max_tokens_length = 512
    system_prompt = prompt_config['system_prompt'][model_name.split('-')[0]][data_name_prefix]
    system_tokens = torch.tensor(tokenizer.encode(system_prompt, add_special_tokens = False),dtype=torch.int)
    query_task = prompt_config['query_prompt'][model_name.split('-')[0]][data_name_prefix]
    local_model_config = json.load(open('./config/model_config.json'))
    stop_token_id = local_model_config[model_name.split('-')[0]]['stop_token_id']

    # Create directories
    if not os.path.exists(f"{cache_path}{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}{data_name.split('.')[0]}/{model_name}")
    if not os.path.exists(f"{cache_path}data"):
        os.makedirs(f"{cache_path}data")
    if not os.path.exists(f"{cache_path}data/{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}data/{data_name.split('.')[0]}/{model_name}")
    if not os.path.exists(f"{cache_path}{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}{data_name.split('.')[0]}/{model_name}")
    if not os.path.exists(f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}"):
        os.makedirs(f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}")

    csv_path = f"{cache_path}{data_name.split('.')[0]}/{model_name}"
    reprocess_path = f"{cache_path}data/{data_name.split('.')[0]}/{model_name}"
    preprocess_path = f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}"

    data_file = open(data_path, 'r', encoding='utf-8')
    data = []
    for line in data_file.readlines():
        data.append(json.loads(line))

    # MODIFIED: Don't shuffle for hotpotqa/triviaqa - keep original order (first 2 are gold passages)
    if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
        data = data[0]
        for i in range(len(data)):
            # Don't shuffle! Keep first 2 passages as gold passages
            data[i]['passage'] = data[i]['output'][0]['document']
    else:
        if data_name == 'samsum.jsonl':
            split_mark = 'Dialogue:'
        else:
            split_mark = 'Passage'
        for i in range(len(data)):
            data[i]['passage'] = re.findall(f'({split_mark} \\d+.*?)(?={split_mark} \\d+|$)', data[i]['context'], re.DOTALL)
        if data_name == 'musique-140.jsonl':
            for i in range(len(data)):
                data[i]['passage'] = re.findall(f'Passage \\d+:\\n(.*?)(?=Passage \\d+:|$)', data[i]['context'], re.DOTALL)
                data[i]['passage'] = ['\n\n' + text for text in data[i]['passage']]
                data[i]['passage'][-1] = data[i]['passage'][-1] + '\n'

    N = len(data)
    batch_data = []
    batch_tokens = []
    question_list = []
    real_answer_list = []

    for query_id,query in enumerate(data[:N]):
        query_prompt = query_task.format(input=data[query_id]['input'])
        query_tokens = torch.tensor(tokenizer.encode(query_prompt, add_special_tokens = False),dtype=torch.int)
        question_list.append(data[query_id]['input'])
        tmp_list = []
        if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
            for i in range(len(data[query_id]['output'])):
                if 'answer' in data[query_id]['output'][i] and \
                    data[query_id]['output'][i]['answer'] not in tmp_list:
                    tmp_list.append(data[query_id]['output'][i]['answer'])
        else:
            for i in range(len(data[query_id]['answers'])):
                if data[query_id]['answers'][i] not in tmp_list:
                    tmp_list.append(data[query_id]['answers'][i])
        real_answer_list.append(tmp_list)
        index = 0

        passage = [system_prompt]
        passage_tokens = [system_tokens]
        for bn in range(len(query['passage'])):
            if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
                passage.append(f'Passage {index+1}:\n' + query['passage'][index] + '\n')
                passage_tokens.append(torch.tensor(tokenizer.encode(f'Passage {index+1}:\n' + query['passage'][index] + '\n', add_special_tokens = False),dtype=torch.int))
            else:
                passage.append(query['passage'][index] + '\n')
                passage_tokens.append(torch.tensor(tokenizer.encode(query['passage'][index] + '\n', add_special_tokens = False),dtype=torch.int))
            index += 1
            if index >=len(query['passage']):
                break
        passage.append(query_prompt)
        passage_tokens.append(query_tokens)
        batch_tokens.append(passage_tokens)
        batch_data.append(passage)

    # MODIFIED: Create gold pair context_rank for hotpotqa/triviaqa
    corpus_lens = []
    context_rank = []

    if preprocess == True:
        for batch_idx, batch in enumerate(batch_data):
            num_passages = len(batch[1:-1])  # Exclude system prompt and query
            corpus_lens.append(num_passages)

        # For hotpotqa/triviaqa gold pair setup
        if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
            # Build context_rank: chunk_id 1 and 2 (first two doc passages) recall each other, rest recall nothing
            # Note: chunk_id 0 is system prompt, chunk_id 1-10 are document passages
            for batch_idx in range(len(batch_data)):
                num_passages = corpus_lens[batch_idx]
                base_idx = sum(corpus_lens[:batch_idx])

                for passage_idx in range(num_passages):
                    global_idx = base_idx + passage_idx

                    # passage_idx corresponds to chunks 1-10 in batch_data
                    # passage_idx 0 = chunk_id 1 = first gold document
                    # passage_idx 1 = chunk_id 2 = second gold document
                    # passage_idx 2-9 = chunk_id 3-10 = distractor documents

                    if passage_idx == 0:
                        # First gold passage (chunk_id 1) recalls only second gold passage (chunk_id 2)
                        # passage_idx 1 is the second gold passage
                        recall_list = [base_idx + 1]
                        # Fill remaining slots with self-reference to avoid issues
                        recall_list.extend([global_idx] * (topk - 1))
                    elif passage_idx == 1:
                        # Second gold passage (chunk_id 2) recalls only first gold passage (chunk_id 1)
                        # passage_idx 0 is the first gold passage
                        recall_list = [base_idx + 0]
                        recall_list.extend([global_idx] * (topk - 1))
                    else:
                        # Distractor passages (chunk_id 3-10): recall nothing - only self
                        recall_list = [global_idx] * topk

                    context_rank.append(recall_list)

            context_rank = np.array(context_rank)
            print(f"Gold pair setup: passages 0 and 1 mutually recall, passages 2-9 recall nothing")
        else:
            # For other datasets, use original embedding-based retrieval
            import faiss
            from FlagEmbedding import FlagModel
            import time

            bgem3 = FlagModel(bge_model_path,
                          query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                          use_fp16=True)
            corpus = []
            for batch in batch_data:
                corpus.extend(batch[1:-1])

            path = f"{cache_path}data/{data_name.split('.')[0]}.bin"
            start_time = time.time()
            if os.path.exists(path):
                index = faiss.read_index(path)
            else:
                corpus_embeddings = bgem3.encode(corpus)
                print("shape of the corpus embeddings:", corpus_embeddings.shape)
                print("data type of the embeddings: ", corpus_embeddings.dtype)
                dim = corpus_embeddings.shape[-1]
                index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
                corpus_embeddings = corpus_embeddings.astype(np.float32)
                index.train(corpus_embeddings)
                index.add(corpus_embeddings)
                print(f"total number of vectors: {index.ntotal}")
                faiss.write_index(index, path)

            corpus = np.asarray(corpus)
            corpus_embeddings = bgem3.encode_queries(corpus)
            corpus_embeddings = corpus_embeddings[:].astype(np.float32)
            score, idx = index.search(corpus_embeddings, k=topk)
            context_rank = idx
            bgem3 = None
            duration_time = time.time() - start_time
            print(f"embedding time: {duration_time}")

    context_rank = context_rank if preprocess == True else []
    corpus_lens = corpus_lens if preprocess == True else []

    return batch_data, batch_tokens, question_list, real_answer_list, stop_token_id, \
        reprocess_path, preprocess_path, csv_path, \
        data_name_prefix, rouge_metrics, context_rank, corpus_lens


def main(model_type='mistral',
         model_path='/mnt/data/models/Mistral-7B-Instruct-v0.3',
         data_name='musique-200.jsonl',
         data_path='/mnt/data/ktransformers-dev/data/',
         cache_path='/mnt/data/processCache/',
         model_name='Mistral-7B-Instruct-v0.3',
         max_cache_len=32768,
         rate=0.2,
         dense=2,
         revert_rope=False,
         topk=10,
         preprocess=True,
         reprocess_method='cacheBlend',
         bge_model_path='/mnt/data/models/bge-m3-FP16',
         draft_model_path=None):
    """
    Main function for process cache experiments with gold pair recall

    Args:
        model_type: one of ['mistral', 'pangu', 'qwen', 'llama']
        model_path: path to the model
        data_name: name of the dataset file
        data_path: path to the data directory
        cache_path: path to cache directory
        model_name: name of the model (for logging)
        max_cache_len: maximum cache length
        rate: compression rate
        dense: dense parameter
        revert_rope: whether to revert rope
        topk: top-k for preprocessing (ignored for hotpotqa/triviaqa, uses gold pair)
        preprocess: whether to use preprocessing
        reprocess_method: reprocessing method ('cacheBlend', 'processCache', 'Cache-Craft', 'speculative_prefill')
        bge_model_path: path to BGE model for embedding
        draft_model_path: path to draft model for speculative_prefill (optional, any model can use any draft model)
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Prepare data with gold pair recall
    prompt_data, tokens_data, question_list, real_answer_list, stop_token_id, \
    save_path, preporcess_save_path, csv_path, data_name_prefix, rouge_metrics, \
    context_rank, corpus_lens = prepare_data_gold_pair(
        model_name, data_path, data_name, cache_path, tokenizer,
        topk, revert_rope, preprocess, bge_model_path
    )

    # Model-specific configuration
    config._attn_implementation = "sdpa"

    # Load model
    model = load_model(model_type, model_path, config)
    model = model.to('cuda')

    # Load draft model for speculative_prefill (all models can use any draft model)
    draft_model = None
    if reprocess_method == "speculative_prefill" and draft_model_path is not None:
        # Automatically detect draft model type from path or use AutoModelForCausalLM
        from transformers import AutoModelForCausalLM
        with torch.no_grad():
            draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
            draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_path, config=draft_config, torch_dtype=config.torch_dtype, trust_remote_code=True
            )
        draft_model = draft_model.to('cuda')

    answer_list = []
    rouge_score = 0
    normalized_em = 0

    # Generate preprocess kv cache
    if preprocess:
        csv_file = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_topk_{topk}_goldpair.csv"
    else:
        csv_file = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_goldpair.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Real Answer', 'Pred Answer'])

    # Adjust passage_len for different models
    passage_len_config = {
        'mistral': 32768,
        'pangu': 32768,
        'qwen': 32768,
        'llama': 32768
    }

    past_key_values = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=max_cache_len,
        device='cuda', dtype=model.dtype,
        passage_len=passage_len_config.get(model_type, 32768),
    )

    for i, iter in enumerate(tokens_data):
        system_len = iter[0].shape[0]

        if rate == 1:
            # Full Cache Recompute
            inputs = torch.cat(iter).to('cuda').unsqueeze(0)
            generated_tokens, _ = prefill_and_generate(model, tokenizer, inputs, max_new_tokens=50)
        else:
            # Cache Reuse
            # Generate KV Cache and importance
            for chunk_id, chunk in enumerate(iter[:-1]):
                if not os.path.exists(f'{save_path}/{i+1}_{chunk_id}_key.pt') or \
                   (reprocess_method == "Cache-Craft" and not os.path.exists(f'{save_path}/cachecraftattn_{i+1}_{chunk_id}.pt')):
                    passage_len = chunk.shape[0]
                    if chunk_id == 0:
                        input_tensor = chunk.unsqueeze(0)
                    else:
                        input_tensor = torch.cat((iter[0], chunk)).unsqueeze(0)
                    prefill_and_save_kv_cache(
                        model, tokenizer, past_key_values, input_tensor.cuda(), save_path=save_path,
                        example_id=i+1, chunk_id=chunk_id, system_len=iter[0].shape[0],
                        passage_len=passage_len, reprocess_method=reprocess_method
                    )

                if preprocess == True and chunk_id == 0:
                    if not os.path.exists(f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt"):
                        shutil.copy(f"{save_path}/{i+1}_{chunk_id}_key.pt", f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt")
                        shutil.copy(f"{save_path}/{i+1}_{chunk_id}_value.pt", f"{preporcess_save_path}/{i+1}_{chunk_id}_value.pt")

                elif preprocess == True and chunk_id > 0:
                    past_len = 0
                    for layer_idx in range(len(past_key_values.key_cache)):
                        past_key_values.past_tokens[layer_idx] = 0

                    # Skip if already preprocessed
                    if os.path.exists(f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt"):
                        continue

                    corpus_passages = [iter[0]]
                    system_key_cache = torch.load(f"{save_path}/{i+1}_{0}_key.pt", weights_only=True)
                    system_value_cache = torch.load(f"{save_path}/{i+1}_{0}_value.pt", weights_only=True)

                    for layer_idx in range(len(past_key_values.key_cache)):
                        past_key_values.key_cache[layer_idx].narrow(2, 0, system_len).copy_(system_key_cache[layer_idx])
                        past_key_values.value_cache[layer_idx].narrow(2, 0, system_len).copy_(system_value_cache[layer_idx])
                        past_key_values.past_tokens[layer_idx] += system_len
                    past_len += system_len
                    id = 1

                    # Check and prepare context topk
                    for corpus_id in context_rank[sum(corpus_lens[:i])+chunk_id-1]:
                        corpus_i, c_id = find_group_and_index(corpus_lens, corpus_id)
                        corpus_i += 1
                        c_id += 1
                        corpus_len = tokens_data[corpus_i-1][c_id].shape[0]

                        # Skip if self-reference (happens for passages 2-9 in gold pair setup)
                        if corpus_i - 1 == i and c_id == chunk_id:
                            continue

                        corpus_passages.append(tokens_data[corpus_i-1][c_id])

                        if os.path.exists(f"{save_path}/{corpus_i}_{c_id}_key.pt") and \
                           ((reprocess_method == "Cache-Craft" and os.path.exists(f'{save_path}/cachecraftattn_{corpus_i}_{c_id}.pt')) or \
                            reprocess_method != "Cache-Craft"):
                            chunk_key_cache = torch.load(f"{save_path}/{corpus_i}_{c_id}_key.pt", weights_only=True)
                            chunk_value_cache = torch.load(f"{save_path}/{corpus_i}_{c_id}_value.pt", weights_only=True)
                        else:
                            tmp_past_key_values = StaticCache(
                                config=model.config, max_batch_size=1,
                                max_cache_len=corpus_len+iter[0].shape[0]+5, device='cuda', dtype=model.dtype
                            )
                            input_tensor = torch.cat((iter[0], tokens_data[corpus_i-1][c_id])).unsqueeze(0)
                            chunk_key_cache, chunk_value_cache = prefill_and_save_kv_cache(
                                model, tokenizer, tmp_past_key_values, input_tensor.cuda(), save_path=save_path,
                                example_id=corpus_i, chunk_id=c_id, system_len=iter[0].shape[0],
                                passage_len=corpus_len, reprocess_method=reprocess_method,
                            )

                        # rope correction
                        if revert_rope and id > 1:
                            position_ids = torch.full((1, chunk_key_cache[0].shape[2]), past_len - system_len, device='cuda')
                            # Different models have different rotary_emb access patterns
                            if model_type in ['mistral', 'qwen']:
                                cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[0], position_ids)
                            else:  # pangu, llama
                                cos, sin = model.model.rotary_emb(chunk_key_cache[0], position_ids)
                            cos = cos.unsqueeze(1)
                            sin = sin.unsqueeze(1)
                            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)

                        for layer_idx in range(len(past_key_values.key_cache)):
                            past_key_values.key_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_key_cache[layer_idx])
                            past_key_values.value_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_value_cache[layer_idx])
                            past_key_values.past_tokens[layer_idx] += corpus_len
                        past_len += corpus_len
                        id += 1

                    corpus_passages.append(chunk)
                    prefill_with_cache_and_save_preprocess(
                        model, tokenizer, past_key_values,
                        corpus_passages, preporcess_save_path,
                        i+1, chunk_id, system_len=system_len, revert_rope=revert_rope,
                        reprocess_method=reprocess_method
                    )
                    print(f'preprocess batch: {i+1}, context_id: {chunk_id}')

            if preprocess:
                load_path = preporcess_save_path
            else:
                load_path = save_path

            generated_tokens,_ = load_kv_and_generate(
                model, tokenizer, past_key_values, iter, load_path, i+1,
                max_new_tokens=50, revert_rope=revert_rope, reprocess_method=reprocess_method,
                rate=rate, dense=dense, draft_model=draft_model, preprocess=preprocess
            )

        answer = tokenizer.decode(torch.tensor(generated_tokens[:-1]))

        # Post-process answer based on model type
        answer = post_process_answer(answer, model_type)

        print(model_name, data_name.split('.')[0], rate, topk, "GOLD_PAIR")
        if data_name_prefix != 'samsum':
            print("question: " + question_list[i])
        print(f'batch: {i+1} preprocess: {preprocess} reprocess_method: {reprocess_method}')
        print("real answer: " + real_answer_list[i][0])
        print("answer: " + answer)

        if answer == '':
            answer_list.append(' ')
        else:
            answer_list.append(answer)

        # Compute score based on model type
        local_em = compute_score(answer, real_answer_list[i], tokenizer, model_type)
        normalized_em += local_em

        local_rouge = _metric_max_over_ground_truths(
            rouge_metrics, answer, real_answer_list[i]
        )
        rouge_score += local_rouge

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([question_list[i], real_answer_list[i][0], answer])

        torch.cuda.empty_cache()

    # Print results
    print(rouge_score/len(tokens_data))
    print(f'em: {normalized_em/len(tokens_data)}')

    if preprocess:
        file_path = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_topk_{topk}_goldpair.txt"
    else:
        file_path = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_goldpair.txt"

    with open(file_path, 'w') as f:
        print(f'num_in_batch: {10}', file=f)
        print(rouge_score/len(tokens_data), file=f)
        print(f'em: {normalized_em/len(tokens_data)}', file=f)


if __name__ == '__main__':
    # Example usage for hotpotqa/triviaqa with gold pair recall

    # Example: Run experiments with gold pair recall (first 2 passages recall each other, rest recall nothing)
    for data_name in ['hotpotqa-260-100-10-doc.jsonl']:
        for topk in [10]:  # topk is ignored for hotpotqa/triviaqa, but kept for compatibility
            for rate in [0]:
                for method in ['FusionRAG']:
                    main(model_type='pangu',
                         model_path='/mnt/data/models/openPangu-Embedded-1B-V1.1/',
                         model_name='openPangu-Embedded-1B-V1.1',
                         rate=rate, preprocess=True, revert_rope=True,
                         cache_path='/mnt/data3/processCache/',
                         reprocess_method=method, data_name=data_name, topk=topk)

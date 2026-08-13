from models.modeling_qwen2 import Qwen2ForCausalLM
from FlagEmbedding import FlagModel
import faiss
import shutil
import numpy as np
import collections
# from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import random
import torch
import os
from rouge import Rouge
import re
import json
import string
import csv
import sys
import time
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
from ktransformers.util.utils import prefill_and_generate, prefill_and_save_kv_cache,load_kv_and_generate, rotate_half, prefill_with_cache_and_save_preprocess
from ktransformers.models.custom_cache import StaticCache
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def parse_generation(s):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s
def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    #gold_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_gold))])).tokens[4:-4]
    #pred_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_pred))])).tokens[4:-4]
    #pdb.set_trace()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
def find_group_and_index(sizes, idx):
    """
    找到list中的某个索引属于哪个组及该组中的索引
    :param sizes: 每个组的大小的列表
    :param idx: 要查找的索引
    :return: (组号, 组中的索引)
    """
    cumulative_size = 0
    for group_id, group_size in enumerate(sizes):
        if cumulative_size + group_size > idx:
            group_index = idx - cumulative_size
            return group_id, group_index
        cumulative_size += group_size
    return None, None  # 如果索引超出范围，返回None
def split_passages_by_title(text, title_marker):
    # 使用标题标记作为分割点，找到所有的位置
    titles = [i for i in range(len(text)) if text.startswith(title_marker, i)]
    # 根据标题位置分割文本为多个段落
    passages = [text[titles[i]:titles[i+1]].strip() for i in range(len(titles) - 1)]
    passages.append(text[titles[-1]:].strip())  # 添加最后一个段落
    return passages
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
def _rouge1_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        # scores = rouge.get_scores(prediction, ground_truth, avg=True)
        scores = rouge.get_scores(normalize_answer(prediction), normalize_answer(ground_truth), avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def prepare_data(model_name, data_path, data_name, cache_path, tokenizer: AutoTokenizer, topk: int, revert_rope, preprocess):

    prompt_config = json.load(open('/mnt/data/benchmark/config/dataset2prompt_few-shot.json'))
    data_path = data_path+data_name
    if data_name in ['2wikimqa.jsonl', 'samsum.jsonl', 'multi_news.jsonl', 'musique.jsonl', 'hotpotqa.jsonl', 'triviaqa.jsonl', 'bamboogle.jsonl']:
        data_name_prefix = data_name.split('.')[0]
    else:
        data_name_prefix = data_name.split('-')[0]
    if data_name_prefix in ['hotpotqa','triviaqa','2wikimqa','musique', 'bamboogle']:
        rouge_metrics = _rouge1_score
        max_tokens_length = 50
    elif data_name_prefix in ['samsum','multi_news']:
        rouge_metrics = _rougel_score
        max_tokens_length = 512
    system_prompt = prompt_config['system_prompt'][model_name.split('-')[0]][data_name_prefix]
    system_tokens = torch.tensor(tokenizer.encode(system_prompt, add_special_tokens = False),dtype=torch.int)
    query_task = prompt_config['query_prompt'][model_name.split('-')[0]][data_name_prefix]
    local_model_config = json.load(open('/mnt/data/benchmark/config/model_config.json'))
    stop_token_id = local_model_config[model_name.split('-')[0]]['stop_token_id']
    # 存报告
    if not os.path.exists(f"{cache_path}{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}{data_name.split('.')[0]}/{model_name}")
    # 存数据
    if not os.path.exists(f"{cache_path}data"):
        os.makedirs(f"{cache_path}data")
    # reprocess 数据
    if not os.path.exists(f"{cache_path}data/{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}data/{data_name.split('.')[0]}/{model_name}")
    if not os.path.exists(f"{cache_path}{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}{data_name.split('.')[0]}/{model_name}")
    # preprocesss 数据
    if not os.path.exists(f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}"):
        os.makedirs(f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}")
    
    csv_path = f"{cache_path}{data_name.split('.')[0]}/{model_name}"
    reprocess_path = f"{cache_path}data/{data_name.split('.')[0]}/{model_name}"
    preprocess_path = f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}"
    data_file = open(data_path, 'r', encoding='utf-8')
    data = []
    for line in data_file.readlines():
        data.append(json.loads(line))  
    if data_name_prefix in ['hotpotqa','triviaqa', 'bamboogle'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
        if data_name_prefix in ['hotpotqa','triviaqa']:
            data = data[0]
        for i in range(len(data)):
            # 打乱顺序
            random.seed(1)
            random.shuffle(data[i]['output'][0]['document'])
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
        if data_name_prefix in ['hotpotqa','triviaqa', 'bamboogle'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
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
            if data_name_prefix in ['hotpotqa','triviaqa', 'bamboogle'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
                passage.append(f'Passage {index+1}:\n' + query['passage'][index] + '\n') 
                passage_tokens.append(torch.tensor(tokenizer.encode(f'Passage {index+1}:\n' + query['passage'][index] + '\n', add_special_tokens = False),dtype=torch.int))
                # passage.append(tokenizer.apply_chat_template([{'role':'user','content':f'Passage {index+1}:\n' + query['passage'][index] + '\n'}], tokenize=False)[98:])
                # passage_tokens.append(torch.tensor(tokenizer.encode(tokenizer.apply_chat_template([{'role':'user','content':f'Passage {index+1}:\n' + query['passage'][index] + '\n'}], tokenize=False)[98:])))
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

    if preprocess == True:
        corpus = []
        corpus_lens = []
        for batch in batch_data:
            corpus.extend(batch[1:-1])
            corpus_lens.append(len(batch[1:-1]))
        path = f"{cache_path}data/{data_name.split('.')[0]}.bin"
        start_time = time.time()
        bgem3 = FlagModel('/mnt/data/models/bge-m3-FP16',
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=True)
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
    return batch_data, batch_tokens,question_list, real_answer_list, stop_token_id, \
        reprocess_path, preprocess_path, csv_path,\
            data_name_prefix, rouge_metrics, context_rank, corpus_lens

def main(model_path= '/mnt/data/models/Qwen2.5-7B-Instruct', 
         data_name='musique-200.jsonl', 
         data_path='/mnt/data/benchmark/data/',
         cache_path='/mnt/data3/processCache/', 
         model_name = 'Qwen2.5-7B-Instruct', 
         max_cache_len= 32768,
         rate=0.2,
         dense=2,
         revert_rope=False,
         topk = 10,
         preprocess=True,
         reprocess_method='cacheBlend'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # prepare data
    prompt_data, tokens_data, question_list, real_answer_list, stop_token_id, save_path, preporcess_save_path, csv_path, data_name_prefix, rouge_metrics, context_rank, corpus_lens  = prepare_data(model_name, data_path, data_name, cache_path, tokenizer, topk, revert_rope, preprocess)
   # preprocess preprare topk

        

    torch.set_default_dtype(config.torch_dtype)
    config._attn_implementation = "sdpa"
    # config.torch_dtype="float16"
    with torch.no_grad():
        model = Qwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    if reprocess_method == "speculative_prefill":
        with torch.no_grad():
            draft_config = AutoConfig.from_pretrained("/mnt/data/models/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
            draft_model = Qwen2ForCausalLM.from_pretrained("/mnt/data/models/Qwen2.5-1.5B-Instruct", config=draft_config, torch_dtype=config.torch_dtype)
        draft_model = model.to('cuda')
    else:
        draft_model = None
    model = model.to('cuda')
    answer_list = []
    rouge_score = 0
    normalized_em = 0

        
    # end preprocess prepare topk

    # 生成 preprocess kv cache
    if preprocess:
        csv_file = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_topk_{topk}.csv"
    else:
        csv_file = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Real Answer', 'Pred Answer'])
    past_key_values = StaticCache(
                config = model.config, max_batch_size = 1, max_cache_len = max_cache_len, device = 'cuda', dtype = model.dtype, passage_len=27000,
            )
    inputs = None
    for i,iter in enumerate(tokens_data):
        # if i != 1:
        #     continue
        system_len = iter[0].shape[0]
        if rate == 1:
        # # Full Cache Recompute 
            inputs = torch.cat(iter).to('cuda').unsqueeze(0)
            # if inputs == None:
            #     inputs = torch.cat(iter).to('cuda').unsqueeze(0)
            # else:
            #     tmp = torch.cat(iter).to('cuda').unsqueeze(0)
            #     inputs = torch.cat((inputs, torch.tensor(tokenizer.encode(f"\nAnswer:{real_answer_list[i][0]}<|im_end|>\n")).to('cuda:0').unsqueeze(0)),dim=1)
            #     inputs = torch.cat((inputs, tmp),dim=1)
            generated_tokens, _ = prefill_and_generate(model, tokenizer, inputs, max_new_tokens=50)
        else:
            # Cache Reuse
            # Generate KV Cache and importance
            for chunk_id, chunk in enumerate(iter[:-1]):
                if not os.path.exists(f'{save_path}/{i+1}_{chunk_id}_key.pt') or (reprocess_method == "Cache-Craft" and not os.path.exists(f'{save_path}/cachecraftattn_{i+1}_{chunk_id}.pt')):
                    preprocess_false_start_time = time.time()
                    passage_len = chunk.shape[0]
                    if chunk_id == 0:
                        input_tensor = chunk.unsqueeze(0)
                    else:
                        input_tensor = torch.cat((iter[0], chunk)).unsqueeze(0)
                    prefill_and_save_kv_cache(
                    model, tokenizer, past_key_values, input_tensor.cuda(), save_path=save_path, 
                    example_id = i+1, chunk_id = chunk_id, system_len = iter[0].shape[0], 
                    passage_len=passage_len,  reprocess_method=reprocess_method,
                    )
                    preprocess_false_duration_time = time.time() - preprocess_false_start_time
                preprocess_false_start_time = time.time()
                if preprocess == True and chunk_id == 0:
                    if not os.path.exists(f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt"):
                        shutil.copy(f"{save_path}/{i+1}_{chunk_id}_key.pt", f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt")
                        shutil.copy(f"{save_path}/{i+1}_{chunk_id}_value.pt", f"{preporcess_save_path}/{i+1}_{chunk_id}_value.pt")
                elif preprocess == True and chunk_id > 0:
                    past_len = 0
                    for layer_idx in range(len(past_key_values.key_cache)):
                        past_key_values.past_tokens[layer_idx] = 0
                    # 不需要生成 preprocess
                    if os.path.exists(f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt"):
                        continue
                    corpus_passages = [iter[0]]
                    system_key_cache = torch.load(f"{save_path}/{i+1}_{0}_key.pt",weights_only=True)
                    system_value_cache = torch.load(f"{save_path}/{i+1}_{0}_value.pt",weights_only=True)
                    for layer_idx in range(len(past_key_values.key_cache)):
                        past_key_values.key_cache[layer_idx].narrow(2,0,system_len).copy_(system_key_cache[layer_idx])
                        past_key_values.value_cache[layer_idx].narrow(2,0,system_len).copy_(system_value_cache[layer_idx])
                        past_key_values.past_tokens[layer_idx] += system_len
                    past_len += system_len
                    id = 1
                    # 检查下 context 的 topk 有没有准备好，没有的现场生成
                    for corpus_id in context_rank[sum(corpus_lens[:i])+chunk_id-1]:
                        corpus_i, c_id = find_group_and_index(corpus_lens, corpus_id)
                        corpus_i += 1
                        c_id += 1
                        corpus_len = tokens_data[corpus_i-1][c_id].shape[0]
                        # 存在，更新到 past_key_value 中
                        if corpus_i - 1 == i and c_id == chunk_id:
                            continue
                        corpus_passages.append(tokens_data[corpus_i-1][c_id])
                        if os.path.exists(f"{save_path}/{corpus_i}_{c_id}_key.pt") and ((reprocess_method == "Cache-Craft" and os.path.exists(f'{save_path}/cachecraftattn_{corpus_i}_{c_id}.pt')) or reprocess_method != "Cache-Craft"):
                            chunk_key_cache = torch.load(f"{save_path}/{corpus_i}_{c_id}_key.pt",weights_only=True)
                            chunk_value_cache = torch.load(f"{save_path}/{corpus_i}_{c_id}_value.pt",weights_only=True)
                        else:
                            tmp_past_key_values = StaticCache(
                                config = model.config, max_batch_size = 1, max_cache_len = corpus_len+iter[0].shape[0]+5 , device = 'cuda', dtype = model.dtype
                            )
                            input_tensor = torch.cat((iter[0], tokens_data[corpus_i-1][c_id])).unsqueeze(0)
                            chunk_key_cache, chunk_value_cache =  prefill_and_save_kv_cache(
                            model, tokenizer, tmp_past_key_values, input_tensor.cuda(), save_path=save_path, 
                            example_id = corpus_i, chunk_id = c_id, system_len = iter[0].shape[0], 
                            passage_len=corpus_len, reprocess_method=reprocess_method,
                            )
                        # rope 修正
                        if revert_rope and id > 1:
                            position_ids = torch.full((1, chunk_key_cache[layer_idx].shape[2]), past_len - system_len, device='cuda')
                            cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[layer_idx], position_ids)
                            # mistral 限定
                            cos = cos.unsqueeze(1)
                            sin = sin.unsqueeze(1)
                            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)
                        for layer_idx in range(len(past_key_values.key_cache)):
                            past_key_values.key_cache[layer_idx].narrow(2,past_len,corpus_len).copy_(chunk_key_cache[layer_idx])
                            past_key_values.value_cache[layer_idx].narrow(2,past_len,corpus_len).copy_(chunk_value_cache[layer_idx])
                            past_key_values.past_tokens[layer_idx] += corpus_len
                        past_len += corpus_len
                        id += 1
                    corpus_passages.append(chunk)
                    preprocess_start_time = time.time()
                    prefill_with_cache_and_save_preprocess(model, tokenizer, past_key_values, 
                                                           corpus_passages, preporcess_save_path, 
                                                           i+1, chunk_id, system_len=system_len, revert_rope=revert_rope, reprocess_method=reprocess_method,)
                    print(f'preprocess batch: {i+1}, context_id: {chunk_id}')
                    preprocess_false_duration_time = time.time() - preprocess_start_time
            if preprocess:
                load_path = preporcess_save_path
            else:
                load_path = save_path
            generated_tokens = load_kv_and_generate(model, tokenizer, past_key_values, iter, load_path, i+1, 
                                                    max_new_tokens=50, revert_rope=revert_rope, reprocess_method=reprocess_method,
                                                    rate=rate, dense=dense, draft_model=draft_model)
        answer = tokenizer.decode(torch.tensor(generated_tokens[:-1]))
        print(model_name,data_name.split('.')[0],rate, topk)
        if data_name_prefix != 'samsum':
            print("question: " + question_list[i])
        print(f'batch: {i+1} preprocess: {preprocess} reprocess_method: {reprocess_method}')
        print("real answer: " + real_answer_list[i][0])
        print("answer: " + answer)
        if answer == '':
            answer_list.append(' ')
        else:
            answer_list.append(answer)
        local_em = max([_exact_match_score(answer, real_answer) for real_answer in real_answer_list[i]])
        normalized_em += local_em
        local_rouge = _metric_max_over_ground_truths(
            rouge_metrics, answer, real_answer_list[i]
        )
        rouge_score += local_rouge
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([question_list[i], real_answer_list[i][0], answer])
        torch.cuda.empty_cache()
        # break
    # rouge_score = rouge.get_scores(hyps=answer_list, refs=real_answer_list, avg=True)
    print(rouge_score/len(tokens_data))
    print(f'em: {normalized_em/len(tokens_data)}')
    if preprocess:
        file_path = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_topk_{topk}.txt"
    else:
        file_path = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}.txt"
    with open(file_path,  'w') as f:
        print(f'num_in_batch: {10}', file=f)
        print(rouge_score/len(tokens_data), file=f)
        print(f'em: {normalized_em/len(tokens_data)}', file=f)
        
# for rate in [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,1]:
#     main(rate = rate, preprocess=False, revert_rope=False, reprocess_method='processCache') 
# for rate in [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,1]:
#     main(rate = rate, preprocess=False, revert_rope=False, reprocess_method='cacheBlend')
# for rate in [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,1]:
#     main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='processCache') 
# for rate in [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,1]:
#     main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='processCache') 
# data_name = 'hotpotqa-260-100-10-doc.jsonl'
# for rate in [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1]:
#     # main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='processCache',data_name=data_name) 
#     # main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='cacheBlend', data_name=data_name)
#     main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='processCache', data_name=data_name,topk = 10) 
# for data_name in ['hotpotqa-254-500-10-doc.jsonl', 'triviaqa-285-500-10-doc.jsonl']:
# # for data_name in ['triviaqa-282-1000-10-doc.jsonl']:
#     for rate in [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,1]:
#         main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='processCache', data_name=data_name)
#         main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='cacheBlend', data_name=data_name)
#         main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='processCache', data_name=data_name)
if __name__ == '__main__':
    # for data_name in ['triviaqa-270-100-10-doc.jsonl', "musique-200.jsonl", "2wikimqa-200.jsonl"]:
    #     for topk in [10]:
    #         for rate in [0, 1, 0.15, 0.05, 0.1]:
    #             # main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='Cache-Craft', data_name=data_name, topk=topk)
    #             # main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='Cache-Craft', data_name=data_name, topk=topk)
    #             main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='cacheBlend', data_name=data_name, topk=topk)
    #             main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='processCache', data_name=data_name, topk=topk)
    for data_name in ["musique-200.jsonl", 'triviaqa-270-100-10-doc.jsonl', 'hotpotqa-260-100-10-doc.jsonl', "2wikimqa-200.jsonl"]:
        for topk in [10]:
            for rate in [0, 1, 0.15, 0.05, 0.1]:
                # main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='Cache-Craft', data_name=data_name, topk=topk)
                # main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='Cache-Craft', data_name=data_name, topk=topk)
                # main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='cacheBlend', data_name=data_name, topk=topk)
                main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='processCache', data_name=data_name, topk=topk)
                # main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='processCache', data_name=data_name, topk=topk)
        # for rate in[0,0.05,0.1,0.15]:
        #     main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='processCache', data_name=data_name, topk=10)
    # for data_name in ['triviaqa-270-100-10-doc.jsonl', '2wikimqa-200.jsonl', 'musique-200.jsonl', 'hotpotqa-260-100-10-doc.jsonl',]:
    #     for rate in[0, 0.05, 0.1, 0.15, 1]:
    #         for topk in [10]:
    #             main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='Cache-Craft', data_name=data_name, topk=topk)
    #             main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='Cache-Craft', data_name=data_name, topk=topk)
    #             main(rate = rate, preprocess=True, revert_rope=True, reprocess_method='processCache', data_name=data_name, topk=topk)
    #             main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='cacheBlend', data_name=data_name, topk=topk)
    # main(rate = rate, preprocess=False, revert_rope=True, reprocess_method='cacheBlend', data_name=data_name)  
# for rate in [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,1]:
#     main(rate = rate, preprocess=True, revert_rope=False, reprocess_method='processCache',data_name=data_name) 
# for rate in [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,1]:
#     main(rate = rate, preprocess=True, revert_rope=False, reprocess_method='processCache',data_name=data_name) 
import copy
import json
import os
import time
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from run_question import FusionRAGModel
import multiprocessing
from multiprocessing import Process, Lock, Manager
import threading

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
        max_main_questions=None,
        start_idx=0,
        end_idx=None
):
    """准备数据，可以指定起始和结束索引"""
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_main_questions:
        dataset = dataset[:max_main_questions]
        print(f"Limited to first {max_main_questions} main questions")

    if end_idx is None or end_idx > len(dataset):
        end_idx = len(dataset)

    dataset = dataset[start_idx:end_idx]
    print(f"Processing questions {start_idx} to {end_idx}")

    all_questions = []

    for main_q_idx, data_item in enumerate(dataset):
        main_question = data_item["question"]
        main_answer = data_item["answer"]
        intermediate_context = data_item.get("intermediate_context", [])

        question_docs = []  # Documents for THIS question only
        doc_to_idx = {}  # Local doc -> chunk_id mapping for this question
        sub_questions_info = []

        should_test_main_question = True
        if data_item.get('llm_judge', True) is False:
            should_test_main_question = False

        for sub_q_idx, sub_q in enumerate(intermediate_context):
            docs = sub_q.get("retrieve docs", [])
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

            if "No relevant information found".lower() in answer.lower() or "没有相关信息" in answer:
                ""
            else:
                sub_questions_info.append({
                    'query': query,
                    'answer': answer,
                    'gold_docs': docs,  # chunk_ids for docs used by this sub-question
                })
        if should_test_main_question:
            all_questions.extend(sub_questions_info)
    return all_questions


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
{{
"reason": "",
"answer": "", ## output right or wrong
}}
"""

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
        print(result)

        res_json = json.loads(result)

        return res_json["answer"] == "right", res_json["reason"]

    except Exception as e:
        error_msg = f"调用 OpenAI API 时出错: {e}"
        print(error_msg)
        return False, error_msg

def write_result_to_individual_file(result, individual_file_path):
    """将结果写入单个进程的独立文件"""
    # 读取现有结果
    existing_results = []
    if os.path.exists(individual_file_path):
        try:
            with open(individual_file_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            # 如果文件为空或格式错误，重新开始
            existing_results = []

    # 添加新结果
    existing_results.append(result)

    # 写入文件
    with open(individual_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=4)

def prepare_locomo_data(category=2, start_idx=-1, end_index=-1):
    conversation = []
    for i in range(10):
        filename = f"./data/locomo/locomo_input_{i}.json"
        with open(filename, 'r') as f:
            data = json.load(f)
            conversation.append(
                "".join([x["text"] for x in data])
            )
    question_filename = f"./data/locomo/locomo_questions_category_{category}.json"
    with open(question_filename, 'r') as f:
        questions = json.load(f)
        for question in questions:
            question["gold_docs"] = [conversation[question["conversation_index"]]]
            question["query"] = question["question"]
    if start_idx!=-1 and end_index!=-1:
        return questions[start_idx: end_index]
    return questions

def run_test_process(
        process_id,
        gpu_ids,
        start_idx,
        end_idx,
        total_run=200,
        rate=0.3,
        reprocess_method="",
        file_lock=None,
        result_queue=None,
        preprocess_method="",
        questions_to_run=[],
        last_time_result_file="",
        max_memory=None,
        dataset="musique",
        preprocess=True,
        test_last_wrong=False,
        test_last_keep=False,
        category=2,
        cache_path="",
):
    last_questions = []
    try:
        if last_time_result_file != "":
            print(f"rerun last_time_result_file={last_time_result_file}")
            with open(last_time_result_file, 'r') as f:
                last_questions = json.load(f)
    except Exception as E:
        print(f"[run_test_process] fail to load last time questions. E={E}")
    """单个测试进程的运行函数"""
    print(f"Process {process_id}: Starting with GPUs {gpu_ids}")

    # 设置当前进程可见的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # 根据GPU分配情况设置设备
    if len(gpu_ids) > 0:
        main_device = f"cuda:0"
        draft_device = "cuda:0"
    else:
        main_device = "cuda:0"
        draft_device = "cuda:0"

    draft_model_path = "/data2/qy_tmp/xumengyao/Qwen2.5-3B-Instruct"
    if "DraftModel" not in reprocess_method:
        draft_model_path = ""
    if reprocess_method == "DraftModel_origin":
        draft_model_path = ""

    if dataset == "musique":
        file_input = "/home/qy_tmp/xumengyao/work/DATASET/musique_data/musique_input.json"
    elif dataset == "locomo":
        file_input = "/home/qy_tmp/xumengyao/work/DATASET/locomo/locomo_input.json"
    elif dataset == "2wiki":
        file_input = "/home/qy_tmp/xumengyao/work/DATASET/2wikiQA/2wiki_input.json"

    fusion_rag_model = FusionRAGModel(
        model_path='/data2/qy_tmp/xumengyao/Qwen3-32B',
        use_multi_gpu=True,
        model_type="qwen3",
        model_name="Qwen3-32B",
        device=main_device,
        cache_path=cache_path,
        draft_model_device=draft_device,
        draft_model_path=draft_model_path,
        draft_model_type="qwen",
        file_input=file_input,
        preprocess_model_path="/data2/qy_tmp/xumengyao/bge-m3",
        preprocess=preprocess,
        preprocess_method=preprocess_method,
        max_memory=max_memory,
        use_origin_draft_model=reprocess_method=="DraftModel_with_answer",
        apikey="sk-27b5e2809a7148aaba768b6ea0de76b5"
    )


    openai_client = OpenAI(api_key="sk-27b5e2809a7148aaba768b6ea0de76b5",
                           base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/")
    all_questions = []
    if dataset == "musique":
        # 获取该进程需要处理的问题
        all_questions = prepare_reflect_data(
            data_path=f"./result_reflect.json",
            max_main_questions=total_run,
            start_idx=start_idx,
            end_idx=end_idx
        )
    elif dataset == "2wiki":
        all_questions = prepare_reflect_data(
            data_path=f"./data/2wiki/result_reflect.json",
            max_main_questions=total_run,
            start_idx=start_idx,
            end_idx=end_idx
        )

        print(f"Process {process_id}: Processing {len(all_questions)} questions (main questions {start_idx} to {end_idx})")
    elif dataset == "locomo":
        all_questions = prepare_reflect_data(
            data_path=f"./data/locomo/result_locomo_category_{category}_no_detail.json",
            max_main_questions=total_run,
            start_idx=start_idx,
            end_idx=end_idx
        )
        # all_questions = prepare_locomo_data(category=2, start_idx=start_idx, end_index=end_idx)

    if test_last_wrong:
        if len(last_questions) > 0:
            print(f"running the last time wrong questions.")
            all_questions_wrong = []
            for q in all_questions:
                for last_q in last_questions:
                    if q["query"] == last_q["query"] and q["answer"] == last_q["answer"] and last_q[
                        "llm_judge"] is False:
                        all_questions_wrong.append(q)
            all_questions = all_questions_wrong
    elif test_last_keep:
        if len(last_questions) > 0:
            last_questions = [q["query"] for q in last_questions]
            all_questions = [q for q in all_questions if q["query"] not in last_questions]
        print(f"keep running the last time questions. all_questions_len={len(all_questions)}")

    revert_rope = True
    if fusion_rag_model.preprocess == True and fusion_rag_model.preprocess_method == "space":
        revert_rope = False
    if rate == 0.0:
        revert_rope = False
    print(f"Process {process_id}: revert_rope={revert_rope}")

    for question in all_questions:
        if question["query"] == "":
            continue
        ## debug
        if len(questions_to_run)>0:
            if question["query"] not in questions_to_run:
                continue

        gold_docs = []
        ## keep the suquence
        for doc in question["gold_docs"]:
            if doc not in gold_docs:
                gold_docs.append(doc)
        prefix = ""
        format_postfix = """
        output using json format:
        {
        "reason": "",
        "answer": ""
        }
        """

        system_len, doc_tensors_total_length, query_len, decode_len, answer, docs_lens, eigenvalue = fusion_rag_model.run_one_question(
            query=f'question is {question["query"]}',
            question_prefix=f'Given these paragraphs, please first output a short piece of reason, and then '
                  f'generate an appropriate answer for the query.'
                  f'{prefix} {format_postfix}',
            # query=question["query"],
            retrieved_docs=gold_docs,
            model_type='qwen3',
            rate=rate,
            reprocess_method=reprocess_method,
            revert_rope=revert_rope,
            max_new_tokens=500,
        )
        print(f"rate={rate} answer=\n{answer}\n")
        if "</think>" in answer:
            answer = answer.split("</think>")[1].replace("\n\n", "")
        answer = answer.strip("\n")

        try:
            response_json = json.loads(answer.replace("```json", "").replace("```", "").strip())
            reason = response_json["reason"]
            answer = response_json["answer"]
        except:
            reason = ""
        print(f"answer={answer}")
        print(f"GTanswer={question['answer']}")
        print(f"reason={reason}")

        is_correct, judge_reason = judge_answer_with_openai(
            openai_client=openai_client,
            openai_model="deepseek-v3.2",
            question=question["query"],
            ground_truth_answer=question["answer"],
            predicted_answer=f"{reason} {answer}",
        )
        if "调用 OpenAI API 时出错" in judge_reason:
            continue

        question_copy = copy.deepcopy(question)
        question_copy["llm_answer"] = answer
        question_copy["eigenvalue"] = eigenvalue
        question_copy["llm_reason"] = reason
        question_copy["llm_judge"] = is_correct
        question_copy["llm_judge_reason"] = judge_reason
        question_copy["process_id"] = process_id
        question_copy["gpu_ids"] = gpu_ids
        question_copy["timestamp"] = time.time()

        print(f"Process {process_id} - Judgment: {'✓ CORRECT' if is_correct else '✗ INCORRECT'} eigenvalue={eigenvalue}")
        print(f"Process {process_id} - Question: {question['query']}")
        print(f"Process {process_id} - Reason: {reason}")
        print(f"Process {process_id} - Answer: {question['answer']}")
        print(f"Process {process_id} - Fusionrag answer: {answer}")
        print("=" * 80)


        # 将结果放入队列（如果需要进一步处理）
        if result_queue:
            result_queue.put(question_copy)

    print(f"Process {process_id}: Completed processing all questions")

    # 将完成信号放入队列
    if result_queue:
        result_queue.put({"process_id": process_id, "status": "completed"})


def real_time_monitor(result_queue, total_processes, keyword_base, test_last_keep=False):
    """实时监控队列并更新统计信息"""
    completed_processes = 0
    all_results = []
    if test_last_keep is True:
        try:
            summary_file = f"./results/summary_{keyword_base}_interim.json"
            with open(summary_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except Exception as E:
            print(f"fail to load last time questions E={E}")


    while completed_processes < total_processes:
        try:
            result = result_queue.get(timeout=10)  # 10秒超时

            if isinstance(result, dict) and result.get("status") == "completed":
                completed_processes += 1
                print(
                    f"Monitor: Process {result['process_id']} completed. Total completed: {completed_processes}/{total_processes}")
            else:
                # 这是一个结果
                all_results.append(result)

                # 更新统计信息
                correct_count = sum(1 for item in all_results if item.get("llm_judge", False))
                total_count = len(all_results)
                accuracy = correct_count / total_count if total_count > 0 else 0

                print(f"\nMonitor: New result received from Process {result.get('process_id', 'unknown')}")
                print(f"Monitor: Current stats - Correct: {correct_count}/{total_count} ({accuracy:.2%})")

                # 每10个结果保存一次汇总文件
                if len(all_results) % 1 == 0:
                    summary_file = f"./results/summary_{keyword_base}_interim.json"
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=4)
                    print(f"Monitor: Interim summary saved to {summary_file}")

        except Exception as e:
            # 超时或其他错误，继续等待
            continue

    # 所有进程完成，保存最终汇总
    print("\nMonitor: All processes completed!")

    # 最终统计
    correct_count = sum(1 for item in all_results if item.get("llm_judge", False))
    total_count = len(all_results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"Final stats - Correct: {correct_count}/{total_count} ({accuracy:.2%})")

    # 保存最终汇总文件
    # final_summary_file = f"./results/summary_{keyword_base}_final.json"
    # with open(final_summary_file, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=4)
    #
    # print(f"Final summary saved to {final_summary_file}")

    return all_results


def get_data_length(dataset="musique", category=2) -> int:
    if dataset == "musique":
        return len(prepare_reflect_data(
            data_path=f"./result_reflect.json",
        ))
    elif dataset == "locomo":
        return len(prepare_locomo_data(category=category))

def test_question_multiprocess(total_run=-1,
                               rate=0.3,
                               reprocess_method="",
                               preprocess_method="",
                               last_rate=0.3,
                               last_reprocess_method="",
                               last_preprocess_method="",
                               questions_to_run=[],
                               sep=2,
                               dataset="musique",
                               test_last_wrong=False,
                               test_last_keep=False,
                               preprocess=True,
                               gpu_configs=None,
                               max_memories=None,
                               category=2,
                               cache_path=""):
    """多进程测试主函数（改进版：支持动态GPU分配）

    Args:
        sep: GPU分割方式
            1: 8张卡一起用，启动1个进程
            2: 4+4两张卡，启动2个进程
            4: 2+2+2+2四张卡，启动4个进程
            8: 1*8八张卡，启动8个进程
    """
    print(f"Starting multiprocess testing with sep={sep}")

    # 确保结果目录存在
    os.makedirs("./results", exist_ok=True)

    # 根据sep参数确定进程数量和GPU分配
    total_gpus = 8  # 总GPU数量

    if sep <= 0 or sep > total_gpus:
        raise ValueError(f"sep参数必须为1-8之间的整数，当前sep={sep}")

    if total_gpus % sep != 0 and sep!=3: ## 3 is allowed
        raise ValueError(f"sep参数必须能整除8，当前sep={sep}")

    num_processes = sep  # 进程数等于sep
    gpus_per_process = total_gpus // sep  # 每个进程分配的GPU数量

    print(f"进程数: {num_processes}, 每个进程GPU数: {gpus_per_process}")

    # 生成GPU配置
    if max_memories == None:
        max_memories = [None for i in range(1000)]


    if gpu_configs is None:
        gpu_configs = []
        if num_processes == 3:
            "special case"
            gpu_configs = [
                ([1, 2, 3], 0),
                ([1, 4, 5], 1),
                ([1, 6, 7], 2)
            ]
            if dataset == "musique":
                max_memories=[
                    {0: "0GiB", 1: "40GiB", 2: "40GiB"},
                    {0: "0GiB", 1: "40GiB", 2: "40GiB"},
                    {0: "0GiB", 1: "40GiB", 2: "40GiB"},
                ]
            elif dataset == "locomo":
                max_memories=[
                    {0: "20GiB", 1: "40GiB", 2: "40GiB"},
                    {0: "20GiB", 1: "40GiB", 2: "40GiB"},
                    {0: "20GiB", 1: "40GiB", 2: "40GiB"},
                ]
        else:
            for i in range(num_processes):
                start_gpu = i * gpus_per_process
                end_gpu = (i + 1) * gpus_per_process
                gpu_ids = list(range(start_gpu, end_gpu))
                gpu_configs.append((gpu_ids, i))

        print(f"GPU配置: {gpu_configs}")


    # 计算每个进程处理的数据范围
    if total_run < 0:
        total_run = get_data_length(dataset=dataset, category=category)
    print(f"总运行问题={total_run}")


    data_ranges = []
    chunk_size = total_run // num_processes

    for i in range(num_processes):
        start_idx = i * chunk_size
        if i == num_processes - 1:  # 最后一个进程处理剩余数据
            end_idx = total_run
        else:
            end_idx = (i + 1) * chunk_size
        data_ranges.append((start_idx, end_idx))

    print(f"数据划分: {data_ranges}")


    dataset_postfix = ""
    if dataset == "locomo":
        dataset_postfix = f"category_{category}"
    # 创建共享结果文件路径
    keyword_base = f"dataset_{dataset}_{dataset_postfix}_model_Qwen3-32B_rate_{rate}_reprocess_method_{reprocess_method}_preprocess_{preprocess}_{preprocess_method}"
    keyword_base_last_time = f"dataset_{dataset}_{dataset_postfix}_model_Qwen3-32B_rate_{last_rate}_reprocess_method_{last_reprocess_method}_preprocess_{preprocess}_{last_preprocess_method}"
    if test_last_wrong:
        keyword_base = f"{keyword_base}_last_wrong"
        last_time_result_file = f"./results/summary_{keyword_base_last_time}_interim.json"
    elif test_last_keep:
        last_time_result_file = f"./results/summary_{keyword_base}_interim.json"
    else:
        last_time_result_file = ""

    # 创建管理器和锁
    manager = Manager()
    file_lock = manager.Lock()
    result_queue = manager.Queue()

    # 创建进程列表
    processes = []

    # 启动所有进程
    for (gpu_ids, process_id), (start_idx, end_idx), max_memory in zip(gpu_configs, data_ranges, max_memories):
        p = Process(
            target=run_test_process,
            args=(
                process_id,
                gpu_ids,
                start_idx,
                end_idx,
                total_run,
                rate,
                reprocess_method,
                file_lock,
                result_queue,
                preprocess_method,
                questions_to_run,
                last_time_result_file,
                max_memory,
                dataset,
                preprocess,
                test_last_wrong,
                test_last_keep,
                category,
                cache_path
            )
        )
        processes.append(p)
        p.start()

    # 启动实时监控线程
    monitor_thread = threading.Thread(
        target=real_time_monitor,
        args=(result_queue, num_processes, keyword_base, test_last_keep)
    )
    monitor_thread.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 发送结束信号给监控线程
    result_queue.put(None)  # 添加结束信号

    # 等待监控线程完成
    monitor_thread.join()


    return []

if __name__ == '__main__':
    print(f"start testing run_question with multiprocess")

    questions_to_run = [
        "When was King Henry III of England crowned?"
    ]
    questions_to_run = []

    max_memories = [{0: "0GiB", 1: "35GiB", 2: "35GiB"}]
    total_run = 200
    all_gpu_configs = [[([0,1,2], 0)], [([3,4,5], 0)], [([3,6,7], 0)]]
    reprocess_methods = ["DraftModel_smarter"]
    rates = [0.3]
    cache_path="/data1/qy_tmp/xumengyao/fusionrag/" ## 10024 is data1, 10026 is data2

    run_index = 1

    if run_index ==0:
        for idx in range(len(rates)):
            test_question_multiprocess(
                total_run=total_run,  ## -1 means run all
                rate=rates[idx],  ## change this
                reprocess_method=reprocess_methods[idx],
                ## change this  1. DraftModel 2. DraftModel_ppr 3. average 4.DraftModel_with_answer
                preprocess_method="default",  ## change this  1. space 2. default
                questions_to_run=questions_to_run,
                sep=1,  ## 1/2/3/4
                dataset="2wiki",  ## 1. locomo 2. musique
                preprocess=True,  ## if locomo then false, otherwise True
                test_last_keep=True,  ## set=True if keep running
                gpu_configs=all_gpu_configs[run_index],  ## personalize if need
                max_memories=max_memories,
                cache_path=cache_path
            )
    #
    elif run_index ==1:
        for idx in range(len(rates)):
            test_question_multiprocess(
                total_run=total_run,  ## -1 means run all
                rate=rates[idx],  ## change this
                reprocess_method=reprocess_methods[idx],
                ## change this  1. DraftModel 2. DraftModel_ppr 3. average 4.DraftModel_with_answer
                preprocess_method="default",  ## change this  1. space 2. default
                questions_to_run=questions_to_run,
                sep=1,  ## 1/2/3/4
                dataset="musique",  ## 1. locomo 2. musique
                preprocess=True,  ## if locomo then false, otherwise True
                test_last_keep=True,  ## set=True if keep running
                gpu_configs=all_gpu_configs[run_index],  ## personalize if need
                max_memories=max_memories,
                cache_path=cache_path
            )

    elif run_index == 2:
        for idx in range(len(rates)):
            for category in [1,2,3]:
                all_results = test_question_multiprocess(
                    total_run=total_run, ## -1 means run all
                    rate=rates[idx], ## change this
                    reprocess_method=reprocess_methods[idx], ## change this  1. DraftModel 2. DraftModel_ppr 3. average 4.DraftModel_with_answer
                    preprocess_method="default", ## change this  1. space 2. default
                    questions_to_run=questions_to_run,
                    sep=1, ## 1/2/3/4
                    dataset= "locomo", ## 1. locomo 2. musique
                    preprocess=True,  ## if locomo then false, otherwise True
                    test_last_keep=True,  ## set=True if keep running
                    gpu_configs=all_gpu_configs[run_index],  ## personalize if need
                    category=category,
                    max_memories=max_memories,
                    cache_path=cache_path,

                    test_last_wrong=True,
                    last_rate=rates[idx],
                    last_preprocess_method="default",
                    last_reprocess_method=reprocess_methods[idx]
                )
    #
    #




    print("All tests completed!")

test_last_wrong = True,
last_rate = rates[idx],
last_preprocess_method = "default",
last_reprocess_method = reprocess_methods[idx]
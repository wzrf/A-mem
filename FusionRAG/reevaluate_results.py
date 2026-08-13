#!/usr/bin/env python3
"""
重新评估已有的测试结果

读取 /mnt/data/junshi/ 和 /mnt/data/reflect 目录下的 CSV 文件，
使用改进的判断模块重新评估所有答案，并保存新的结果。
"""

import os
import csv
import glob
from typing import Tuple, List, Dict
from openai import OpenAI


def judge_answer_with_openai(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    predicted_answer: str,
    ground_truth_answer: str
) -> Tuple[bool, str]:
    """
    使用 OpenAI API 判断预测答案是否正确

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


def read_csv_results(csv_path: str) -> List[Dict]:
    """
    读取 CSV 文件中的测试结果

    Args:
        csv_path: CSV 文件路径

    Returns:
        List[Dict]: 每一行的数据字典列表
    """
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def reevaluate_csv(
    csv_path: str,
    openai_client: OpenAI,
    openai_model: str,
    output_path: str = None
):
    """
    重新评估一个 CSV 文件中的所有结果

    Args:
        csv_path: 输入的 CSV 文件路径
        openai_client: OpenAI 客户端
        openai_model: 使用的模型名称
        output_path: 输出文件路径（默认在原文件名后加 _reevaluated）
    """
    print(f"\n{'='*80}")
    print(f"重新评估: {csv_path}")
    print(f"{'='*80}")

    # 读取原始结果
    results = read_csv_results(csv_path)
    total = len(results)
    print(f"共 {total} 条记录需要重新评估")

    # 重新评估每一条
    reevaluated_results = []
    correct_count = 0

    for i, row in enumerate(results, 1):
        # 获取字段（兼容不同的列名）
        main_question = row.get('Main Question', '')
        sub_question = row.get('Sub Question', '')
        ground_truth = row.get('Ground Truth', '')
        predicted = row.get('Predicted', '')
        old_correct = row.get('Correct', '')

        print(f"\n[{i}/{total}] 评估中...")
        print(f"子问题: {sub_question[:50]}...")

        # 重新判断
        is_correct, reason = judge_answer_with_openai(
            openai_client, openai_model,
            sub_question, predicted, ground_truth
        )

        # 更新结果
        new_row = {
            'Main Question': main_question,
            'Sub Question': sub_question,
            'Ground Truth': ground_truth,
            'Predicted': predicted,
            'Old Correct': old_correct,  # 保留旧的判断结果
            'Correct': is_correct,
            'Reason': reason
        }
        reevaluated_results.append(new_row)

        if is_correct:
            correct_count += 1

        # 显示变化
        old_str = str(old_correct).lower()
        old_is_correct = old_str in ['true', '1', 'yes']
        if old_is_correct != is_correct:
            print(f"  ⚠️  判断变化: {old_correct} -> {is_correct}")
        print(f"  结果: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        print(f"  原因: {reason[:100]}...")

    # 计算子问题准确率
    sub_q_accuracy = correct_count / total if total > 0 else 0

    # 计算主问题准确率
    main_questions = {}
    for row in reevaluated_results:
        main_q = row['Main Question']
        is_correct = row['Correct']

        if main_q not in main_questions:
            main_questions[main_q] = {'total': 0, 'correct': 0}

        main_questions[main_q]['total'] += 1
        if is_correct:
            main_questions[main_q]['correct'] += 1

    # 统计主问题准确率（所有子问题都正确才算主问题正确）
    total_main_q = len(main_questions)
    correct_main_q = sum(1 for stats in main_questions.values() if stats['correct'] == stats['total'])
    main_q_accuracy = correct_main_q / total_main_q if total_main_q > 0 else 0

    # 保存结果
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_reevaluated.csv"

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Main Question', 'Sub Question', 'Ground Truth', 'Predicted',
                      'Old Correct', 'Correct', 'Reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reevaluated_results)

    print(f"\n{'='*80}")
    print(f"评估完成！")
    print(f"{'='*80}")
    print(f"子问题统计:")
    print(f"  总计: {total} 条")
    print(f"  正确: {correct_count} 条")
    print(f"  准确率: {sub_q_accuracy:.2%}")
    print(f"\n主问题统计:")
    print(f"  总计: {total_main_q} 个")
    print(f"  完全正确: {correct_main_q} 个")
    print(f"  准确率: {main_q_accuracy:.2%}")
    print(f"\n结果已保存至: {output_path}")
    print(f"{'='*80}")

    return {
        'sub_q_accuracy': sub_q_accuracy,
        'sub_q_correct': correct_count,
        'sub_q_total': total,
        'main_q_accuracy': main_q_accuracy,
        'main_q_correct': correct_main_q,
        'main_q_total': total_main_q
    }


def aggregate_main_question_results(csv_path: str, output_txt_path: str = None):
    """
    统计主问题级别的准确率

    Args:
        csv_path: 重新评估后的 CSV 文件路径
        output_txt_path: 输出统计结果的文本文件路径
    """
    results = read_csv_results(csv_path)

    # 按主问题分组
    main_questions = {}
    for row in results:
        main_q = row['Main Question']
        is_correct = str(row['Correct']).lower() in ['true', '1', 'yes']

        if main_q not in main_questions:
            main_questions[main_q] = {'total': 0, 'correct': 0}

        main_questions[main_q]['total'] += 1
        if is_correct:
            main_questions[main_q]['correct'] += 1

    # 统计主问题准确率
    total_main_q = len(main_questions)
    correct_main_q = 0

    for main_q, stats in main_questions.items():
        # 只有所有子问题都正确，主问题才算正确
        if stats['correct'] == stats['total']:
            correct_main_q += 1

    main_q_accuracy = correct_main_q / total_main_q if total_main_q > 0 else 0

    # 计算子问题准确率
    total_sub_q = len(results)
    correct_sub_q = sum(1 for row in results if str(row['Correct']).lower() in ['true', '1', 'yes'])
    sub_q_accuracy = correct_sub_q / total_sub_q if total_sub_q > 0 else 0

    # 保存统计结果
    if output_txt_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_txt_path = f"{base_name}_stats.txt"

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"重新评估统计结果\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"主问题准确率: {correct_main_q}/{total_main_q} ({main_q_accuracy:.4f})\n")
        f.write(f"子问题准确率: {correct_sub_q}/{total_sub_q} ({sub_q_accuracy:.4f})\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"主问题详细统计:\n")
        f.write(f"{'='*80}\n\n")

        for i, (main_q, stats) in enumerate(main_questions.items(), 1):
            all_correct = stats['correct'] == stats['total']
            f.write(f"{i}. {'✓' if all_correct else '✗'} {main_q[:100]}\n")
            f.write(f"   子问题正确率: {stats['correct']}/{stats['total']}\n\n")

    print(f"\n统计结果已保存至: {output_txt_path}")

    return main_q_accuracy, sub_q_accuracy


def main(
    directories: List[str] = None,
    openai_api_key: str = None,
    openai_base_url: str = "https://api.deepseek.com/v1",
    openai_model: str = "deepseek-chat"
):
    """
    重新评估多个目录下的所有 CSV 结果文件

    Args:
        directories: 要评估的目录列表
        openai_api_key: OpenAI API key
        openai_base_url: API base URL
        openai_model: 使用的模型
    """
    if directories is None:
        directories = [
            '/mnt/data/junshi',
            '/mnt/data/reflect'
        ]

    # 初始化 OpenAI 客户端
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    print(f"\n{'='*80}")
    print(f"开始重新评估测试结果")
    print(f"{'='*80}")
    print(f"目录列表: {directories}")
    print(f"模型: {openai_model}")
    print(f"{'='*80}\n")

    # 处理每个目录
    all_results = []

    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️  目录不存在，跳过: {directory}")
            continue

        print(f"\n{'='*80}")
        print(f"处理目录: {directory}")
        print(f"{'='*80}")

        # 查找所有 CSV 文件（排除已经重新评估的文件）
        csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
        csv_files = [f for f in csv_files if '_reevaluated' not in f]

        if not csv_files:
            print(f"  未找到 CSV 文件")
            continue

        print(f"  找到 {len(csv_files)} 个 CSV 文件")

        # 重新评估每个 CSV 文件
        for csv_file in csv_files:
            try:
                result = reevaluate_csv(
                    csv_file, openai_client, openai_model
                )

                all_results.append({
                    'file': csv_file,
                    'sub_q_accuracy': result['sub_q_accuracy'],
                    'sub_q_correct': result['sub_q_correct'],
                    'sub_q_total': result['sub_q_total'],
                    'main_q_accuracy': result['main_q_accuracy'],
                    'main_q_correct': result['main_q_correct'],
                    'main_q_total': result['main_q_total']
                })

                # 生成主问题统计文件
                reevaluated_csv = csv_file.replace('.csv', '_reevaluated.csv')
                if os.path.exists(reevaluated_csv):
                    aggregate_main_question_results(reevaluated_csv)

            except Exception as e:
                print(f"\n❌ 处理文件 {csv_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 打印总结
    print(f"\n\n{'='*80}")
    print(f"全部评估完成！")
    print(f"{'='*80}")

    if all_results:
        print(f"\n总结:")
        for result in all_results:
            print(f"\n文件: {result['file']}")
            print(f"  子问题准确率: {result['sub_q_accuracy']:.2%} ({result['sub_q_correct']}/{result['sub_q_total']})")
            print(f"  主问题准确率: {result['main_q_accuracy']:.2%} ({result['main_q_correct']}/{result['main_q_total']})")

        # 计算总体准确率
        total_sub_q_correct = sum(r['sub_q_correct'] for r in all_results)
        total_sub_q = sum(r['sub_q_total'] for r in all_results)
        overall_sub_q_accuracy = total_sub_q_correct / total_sub_q if total_sub_q > 0 else 0

        total_main_q_correct = sum(r['main_q_correct'] for r in all_results)
        total_main_q = sum(r['main_q_total'] for r in all_results)
        overall_main_q_accuracy = total_main_q_correct / total_main_q if total_main_q > 0 else 0

        print(f"\n{'='*80}")
        print(f"总体统计:")
        print(f"  子问题准确率: {overall_sub_q_accuracy:.2%} ({total_sub_q_correct}/{total_sub_q})")
        print(f"  主问题准确率: {overall_main_q_accuracy:.2%} ({total_main_q_correct}/{total_main_q})")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    # 使用示例
    main(
        directories=[
            '/mnt/data/junshi',
            '/mnt/data/reflect'
        ],
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_base_url="https://api.deepseek.com/v1",
        openai_model="deepseek-chat"
    )

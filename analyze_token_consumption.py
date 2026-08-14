import json


def analyze_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 假设 data 是一个列表，每个元素是字典
    if not isinstance(data, list):
        raise ValueError("JSON 文件根元素应为列表")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    count = len(data)

    for item in data:
        total_prompt_tokens += item.get('prompt_tokens', 0)
        total_completion_tokens += item.get('completion_tokens', 0)

    avg_prompt = total_prompt_tokens / count if count > 0 else 0
    avg_completion = total_completion_tokens / count if count > 0 else 0

    print(f"条目数量: {count}")
    print(f"平均 prompt_tokens: {avg_prompt:.2f}")
    print(f"平均 completion_tokens: {avg_completion:.2f}")
    print(f"总 completion_tokens: {avg_prompt+avg_completion:.2f}")


# 使用方法
analyze_tokens('./token_consumption/0.json')
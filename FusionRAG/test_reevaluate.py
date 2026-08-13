#!/usr/bin/env python3
"""
快速测试重新评估脚本
"""

import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from reevaluate_results import main

# 测试运行
main(
    directories=[
        '/mnt/data/junshi/Qwen3-32B/results',  # 指定具体的 results 目录
    ],
    openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    openai_base_url="https://api.deepseek.com/v1",
    openai_model="deepseek-chat"
)

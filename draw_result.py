import glob
import json
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt


def process_results(directory="."):
    # 匹配目录下所有符合 result_fusion_rag_*.json 的文件
    file_pattern = os.path.join(directory, "result_fusion_rag_*.json")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print("未找到匹配的文件，请检查文件路径或文件名格式！")
        return

    # 正则表达式匹配文件名结构：
    # result_fusion_rag_<fusion_param>_<model_name>_retrieve_<k>.json
    pattern = r"result_fusion_rag_([0-9.]+?)_(.+?)_retrieve_(\d+)\.json"

    # 使用字典按 (model, k) 分组保存折线数据
    # 结构: series_data[(model, k)] = [(param_val, f1_avg), ...]
    series_data = defaultdict(list)

    for path in file_paths:
        filename = os.path.basename(path)

        match = re.match(pattern, filename)
        if not match:
            # 如果不匹配完整规则，跳过或打印提示
            print(f"⚠️ 文件名不符合预期命名规则，已跳过: {filename}")
            continue

        param_str, model_name, k_str = match.groups()
        param_val = float(param_str)
        retrieve_k = int(k_str)

        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not samples:
            print(f"警告: 文件 {filename} 内容为空，已跳过。")
            continue

        totals = defaultdict(float)
        count = len(samples)

        # 累加 metrics 以及 raw_context_len
        for item in samples:
            metrics = item.get("metrics", {})
            for k, v in metrics.items():
                totals[k] += v

            if "raw_context_len" in item:
                totals["raw_context_len"] += item["raw_context_len"]

        # 计算各项均值
        averages = {k: round(v / count, 6) for k, v in totals.items()}

        # 终端控制台打印输出
        print("\n" + "=" * 60)
        print(f"📄 文件名: {filename}")
        print(
            f"🤖 模型: {model_name} | 🔍 Retrieve K: {retrieve_k} | 📊 参数 (横轴): {param_val}"
        )
        print(f"👥 样本总数: {count}")
        print("-" * 60)
        print("均值统计结果 (Average Metrics):")
        print(json.dumps(averages, indent=4, ensure_ascii=False))

        # 保存绘图数据
        f1_avg = averages.get("f1", 0.0)
        group_key = (model_name, retrieve_k)
        series_data[group_key].append((param_val, f1_avg))

    if not series_data:
        print("没有提取到可用于绘图的数据。")
        return

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 预设不同的标记符号，区分多条折线
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]
    marker_idx = 0

    # 所有出现过的横轴参数集合，用于统一 X 轴刻度
    all_x_vals = set()

    # 遍历每个 (model, retrieve_k) 组合，画一条折线
    for (model_name, retrieve_k), points in sorted(series_data.items()):
        # 按横轴参数从小到大排序
        points.sort(key=lambda x: x[0])

        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        all_x_vals.update(x_vals)

        label_name = f"{model_name} (k={retrieve_k})"
        current_marker = markers[marker_idx % len(markers)]
        marker_idx += 1

        # 绘制当前组的折线
        line = plt.plot(
            x_vals,
            y_vals,
            marker=current_marker,
            linewidth=2,
            markersize=7,
            label=label_name,
        )

        # 在折线点上方标记具体 F1 值
        color = line[0].get_color()
        for x, y in zip(x_vals, y_vals):
            plt.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=color,
            )

    plt.xlabel("Fusion Parameter (e.g., 0.3, 1.0)", fontsize=11)
    plt.ylabel("Average F1 Score", fontsize=11)
    plt.title("Average F1 Score vs Fusion Parameter Across Models/K", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)

    # 刻度及图例设置
    plt.xticks(sorted(list(all_x_vals)))
    plt.legend(title="Model & Retrieve K", fontsize=10, loc="best")
    plt.tight_layout()

    # 保存并显示图片
    output_image = "f1_score_trends_comparison.png"
    plt.savefig(output_image, dpi=300)
    print("\n" + "=" * 60)
    print(f"🎉 绘图完成！多线对比图已保存至: {os.path.abspath(output_image)}")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    process_results("results")
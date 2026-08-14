import glob
import json
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# ==================== 用户配置选项 ====================
# 是否包含 1.5B 的 Draft Model (True: 画出 1.5B; False: 过滤掉 1.5B，只保留 3B 等)
PLOT_1_5B = False

# 图表输出前缀
OUTPUT_PREFIX = "f1_score"
# ======================================================


def process_results(directory="."):
    file_pattern = os.path.join(directory, "result_*.json")
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print("未找到匹配的文件，请检查文件路径或文件名格式！")
        return

    # 正则表达式匹配文件名结构
    pattern = r"^result_(.+?)_([0-9.]+?)_(Qwen2\.5-.+?)_([^_]+)_retrieve_(\d+)\.json$"

    # 按 Category 存储绘图数据
    # 结构: category_series_data[cat_id][group_key] = [(param_val, f1_avg), ...]
    category_series_data = defaultdict(lambda: defaultdict(list))

    all_x_vals = set()

    for path in file_paths:
        filename = os.path.basename(path)

        match = re.match(pattern, filename)
        if not match:
            print(f"⚠️ 文件名不符合预期命名规则，已跳过: {filename}")
            continue

        rag_type, param_str, draft_model, target_model, k_str = match.groups()
        param_val = float(param_str)
        retrieve_k = int(k_str)

        # 控制选项：过滤 1.5B 模型
        if not PLOT_1_5B and "1.5B" in draft_model:
            print(f"🙈 已根据配置过滤 1.5B 模型文件: {filename}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not samples:
            print(f"⚠️ 警告: 文件 {filename} 内容为空，已跳过。")
            continue

        # 按 Category 统计指标
        # cat_totals[cat_id] = { "f1": sum, "exact_match": sum, ... }
        cat_totals = defaultdict(lambda: defaultdict(float))
        cat_counts = defaultdict(int)

        for item in samples:
            cat = item.get("category", "Unknown")
            metrics = item.get("metrics", {})

            for k, v in metrics.items():
                cat_totals[cat][k] += v

            if "raw_context_len" in item:
                cat_totals[cat]["raw_context_len"] += item["raw_context_len"]

            cat_counts[cat] += 1

        # Shell 终端控制台打印统计信息
        print("\n" + "=" * 75)
        print(f"📄 文件名: {filename}")
        print(f"⚙️  RAG 模式: {rag_type} | Draft: {draft_model}")
        print(
            f"🎯 Target: {target_model} | Retrieve K: {retrieve_k} | 横轴参数: {param_val}"
        )
        print("-" * 75)

        group_key = (rag_type, draft_model, target_model, retrieve_k)
        all_x_vals.add(param_val)

        # 针对每个 Category 计算平均值并保存绘图数据
        for cat, totals in cat_totals.items():
            count = cat_counts[cat]
            averages = {k: round(v / count, 6) for k, v in totals.items()}

            print(f"📌 [Category {cat}] (样本数: {count}):")
            print(f"   平均 F1: {averages.get('f1', 0.0):.4f} | EM: {averages.get('exact_match', 0.0):.4f} | ROUGE-L: {averages.get('rougeL_f', 0.0):.4f}")

            # 存入绘图字典
            f1_avg = averages.get("f1", 0.0)
            category_series_data[cat][group_key].append((param_val, f1_avg))

    if not category_series_data:
        print("\n未解析到任何可绘图的有效数据。")
        return

    # 按 Category 分类画图
    markers = ["o", "s", "^", "D", "v", "p", "*", "X"]
    line_styles = ["-", "--", "-."]

    categories = sorted(list(category_series_data.keys()), key=lambda x: str(x))

    print("\n" + "=" * 75)
    print(f"📊 开始按 Category 逐个生成折线图 (PLOT_1_5B = {PLOT_1_5B})...")

    for cat in categories:
        series_data = category_series_data[cat]

        plt.figure(figsize=(11, 6))
        marker_idx = 0

        # 遍历每一种配置路线
        for (rag_type, draft_model, target_model, retrieve_k), points in sorted(
            series_data.items()
        ):
            # 按横轴参数从小到大排序
            points.sort(key=lambda x: x[0])

            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]

            draft_short = draft_model.replace("Qwen2.5-", "").replace(
                "-Instruct", ""
            )
            label_name = f"[{rag_type}] Draft:{draft_short} (k={retrieve_k})"

            current_marker = markers[marker_idx % len(markers)]
            current_ls = line_styles[
                (marker_idx // len(markers)) % len(line_styles)
            ]
            marker_idx += 1

            line = plt.plot(
                x_vals,
                y_vals,
                marker=current_marker,
                linestyle=current_ls,
                linewidth=2,
                markersize=7,
                label=label_name,
            )

            # 在折线节点标注 F1 数值
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

        plt.xlabel("Recomputation Rate / Fusion Parameter", fontsize=11)
        plt.ylabel("Average F1 Score", fontsize=11)
        plt.title(f"Average F1 Score vs Recomputation Rate (Category {cat})", fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.xticks(sorted(list(all_x_vals)))
        plt.legend(title="Configurations", fontsize=9, loc="best", framealpha=0.8)
        plt.tight_layout()

        # 保存为分类图片，如 f1_score_category_4.png
        output_image = f"{OUTPUT_PREFIX}_category_{cat}.png"
        plt.savefig(output_image, dpi=300)
        print(f"  └─ Category {cat} 图表已保存至: {os.path.abspath(output_image)}")
        plt.close()  # 关闭当前 figure 释放内存

    print("=" * 75)
    print("🎉 所有 Category 分类图片绘制完毕！")


if __name__ == "__main__":
    process_results("results")
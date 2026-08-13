import os
from openai import OpenAI, AzureOpenAI
import concurrent.futures
import torch
import numpy as np
from typing import Dict, Union, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
import copy

LOQUACIOUS_TEXT = """
# The Art of Effective Reading: A Comprehensive Guide to Extracting Information from English Articles

Reading articles effectively—especially in a non-native language like English—is not a passive activity but an active process of engagement, analysis, and synthesis. It is a skill that combines strategy, critical thinking, and methodology. Whether you're a student, researcher, professional, or lifelong learner, mastering this skill will transform how you acquire knowledge. Here is a detailed, step-by-step guide to doing it right.

---

## **Phase 1: Pre-Reading – Setting the Stage**

**1. Clarify Your Purpose:**  
Before you read a single word, ask yourself: *Why am I reading this?*  
- Are you reading for **general understanding** of a topic?  
- Are you searching for **specific data or answers** to a question?  
- Are you **analyzing an argument** or evaluating evidence?  
- Are you **preparing for a discussion or writing a summary**?  
Your purpose dictates your strategy. Reading for a specific fact is different from reading to understand a complex theory.

**2. Survey the Article (The "Look-Over"):**  
Spend 5-10 minutes conducting a preliminary reconnaissance.  
- **Title & Subtitle:** They announce the core topic and often the author's angle.  
- **Abstract/Summary:** If present, this is the article's blueprint. It states the purpose, methodology, key findings, and conclusions.  
- **Headings & Subheadings:** These form the article's skeleton. They reveal the structure and logical flow.  
- **Visuals:** Examine graphs, charts, tables, and images. Their captions often contain condensed information.  
- **Introduction & Conclusion:** Read these carefully. The introduction sets the stage; the conclusion summarizes the key takeaways and implications.  
- **First & Last Sentences of Paragraphs:** In academic writing, these are often topic and concluding sentences.  
- **Bold, Italicized, or Highlighted Text:** Key terms and definitions are often emphasized.  
- **References/Bibliography:** This shows the intellectual context and can lead you to other valuable sources.

This survey gives you a mental map. You'll know what to expect and where to focus your attention.

**3. Activate Prior Knowledge:**  
Ask: *What do I already know about this topic?* Jot down a few points or questions. Connecting new information to existing knowledge (schema) dramatically improves comprehension and retention.

---

## **Phase 2: Active Reading – Deep Engagement with the Text**

**1. Annotate as You Read:**  
Do not just read—*interact*. Have a pen, highlighter, or digital annotation tool ready.  
- **Underline/Highlight Key Ideas:** Be selective. Don't turn the page into a rainbow. Highlight only the core thesis, main arguments, crucial evidence, and surprising claims.  
- **Write Marginal Notes (Marginilia):**  
  - Summarize a complex paragraph in 2-3 words in the margin.  
  - Write "!" for surprising information, "?" for confusion or questions.  
  - Draw arrows to connect related ideas across the text.  
  - Define difficult terms in your own words at the side.

**2. Employ Targeted Reading Strategies:**  
- **Skimming:** Move your eyes quickly over the text to get the gist. Use this when reviewing or when a section seems less relevant to your purpose.  
- **Scanning:** Look for specific words, numbers, or phrases (e.g., a date, a name, a statistic). Let your eyes dart rapidly until you find the target.  
- **Close Reading:** For critical sections (the thesis, methodology, key analysis), slow down. Read every word. Unpack complex sentences. Ask how each sentence relates to the one before and after.

**3. Tackle Vocabulary Strategically:**  
- **Do NOT stop for every unfamiliar word.** It disrupts flow and comprehension.  
- **Circle it and infer its meaning** from context (the surrounding words and sentences).  
- **Look it up only if:** it appears repeatedly (signaling a key concept), or if the sentence's meaning is completely opaque without it. Keep a vocabulary journal for recurrent technical terms.

**4. Ask Questions Constantly (The Q&A Method):**  
Turn headings into questions. For example, if a heading is "The Economic Impact of Climate Policy," ask: *What IS the economic impact according to this author?* Read to answer that question. Other powerful questions include:  
- What is the author's **main claim**?  
- What **evidence** are they using to support it? (Data, examples, logic, appeals to authority?)  
- What is the author's **purpose**? (To inform, persuade, critique, propose?)  
- Who is the intended **audience**? (Experts, general public, policymakers?)  
- What are the underlying **assumptions**?  
- Do I **agree** with the conclusions? Why or why not?  
- How does this **connect** to other things I've read or know?

**5. Paraphrase and Summarize Periodically:**  
After a key section or a difficult paragraph, pause. Look away from the text and explain it to yourself *in your own words, aloud or in writing*. This is the single best test of true understanding. If you can't rephrase it, you haven't grasped it yet. Reread.

**6. Visualize the Information:**  
Create a quick mental or physical diagram. For an argument, draw a flowchart of the logic. For a process, sketch the steps. For a comparison, make a Venn diagram. This engages a different part of your brain and solidifies understanding.

---

## **Phase 3: Post-Reading – Consolidation and Synthesis**

**1. Create a Structured Summary:**  
Now that you've finished the active read, write a summary **without looking at the article**. Use the "Who, What, When, Where, Why, and How" framework, or a simple outline:  
- **Main Thesis/Argument:** In one sentence.  
- **Key Supporting Points:** 3-5 bullet points.  
- **Primary Evidence Used:** Data, studies, historical examples.  
- **Conclusions and Implications.**  
After writing, check your summary against the text for accuracy.

**2. Organize the Extracted Information:**  
Based on your original purpose, organize what you found.  
- For **research:** Sort information into thematic categories in your notes or a digital document. Use quotes and page numbers.  
- For **answering specific questions:** Write the answers clearly, citing the relevant section of the article.  
- Use a system like **Cornell Notes**, a **graphic organizer**, or a digital tool like Notion or OneNote to keep information structured and retrievable.

**3. Reflect and Critically Evaluate:**  
This is where you move from *what the text says* to *what you think about it*.  
- **Evaluate the Source:** Is it credible? Peer-reviewed? From a reputable publisher? Is the author biased?  
- **Assess the Argument:** Was the evidence convincing? Were there logical fallacies? Were counter-arguments addressed?  
- **Synthesize:** How does this information fit with or challenge what you already know? What are the broader implications? What new questions does it raise?

**4. Discuss or Teach It:**  
Explain the article's content to a friend, colleague, or even an imaginary audience. Teaching forces you to clarify your thoughts, reveal gaps in your understanding, and solidify the material in your long-term memory.

---

## **Special Considerations for Reading in English**

- **Accept Ambiguity Temporarily:** It's okay not to understand 100% immediately. Focus on grasping the core idea first. Clarity often emerges as you read further.
- **Practice "Chunking":** Read in meaningful phrases (noun phrases, verb phrases) rather than word-by-word. This improves speed and comprehension. For example, read: "The rapid development / of artificial intelligence / has sparked / widespread ethical debates."
- **Leverage Digital Tools Wisely:** Use browser extensions for instant dictionary definitions, text-to-speech functions to hear difficult passages aloud, and translation tools for single words or phrases when truly stuck—but rely on them as a crutch, not a wheelchair.
- **Read Widely and Regularly:** The best way to get better at reading English articles is to read more of them. Start with well-edited magazines (The Economist, Scientific American) or reputable news sites before diving into dense academic journals.

### **The Mindset of a Proficient Reader**

Ultimately, the "right way" to read is to be **proactive, not reactive**. You are not a vessel to be filled by the text; you are a miner, a detective, and a critic actively extracting, interrogating, and evaluating information. You enter the text with a plan, you engage with it using disciplined techniques, and you leave it with organized knowledge and sharper critical faculties. This transformative approach turns reading from a task into a powerful tool for learning, thinking, and growing in any language, but especially in the global lingua franca of English.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def save_matrix_heatmap(v_2d, save_path="ppr_matrix_heatmap.png", cmap_name="viridis", vmin=0, vmax=0.01):
    """
        将二维的 (x, x) PPR 或 Attention 矩阵绘制为高清热力图，并保存到本地（支持手动控制颜色范围）。

        参数:
        v_2d (torch.Tensor 或 numpy.ndarray): 二维矩阵，形状为 (x, x)
        save_path (str): 本地保存图片的路径
        cmap_name (str): 颜色映射方案
        vmin (float, 可选): 颜色条对应的最小值。如果不传，自动设为矩阵的最小值。
        vmax (float, 可选): 颜色条对应的最大值。如果要放大细节，可以设为一个较小的值（如 0.05 或 0.1）。
        """
    if isinstance(v_2d, torch.Tensor):
        v_np = v_2d.detach().cpu().numpy()
    else:
        v_np = v_2d

    if len(v_np.shape) != 2:
        raise ValueError(f"输入矩阵必须是二维的，当前维度形状为: {v_np.shape}")

    x = v_np.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # 💡 核心改动：在 imshow 中显式传入 vmin 和 vmax
    im = ax.imshow(v_np, cmap=cmap_name, aspect='equal', origin='upper',
                   vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PPR / Attention Score", fontsize=11)

    ax.set_title(f"2D Attention Matrix Heatmap ({x}x{x})", fontsize=13, pad=15)
    ax.set_xlabel("Key Token Index", fontsize=11)
    ax.set_ylabel("Query Token Index", fontsize=11)

    if x > 20:
        ax.locator_params(nbins=10)

    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(
        f"🎉 二维热力图（范围: [{vmin if vmin is not None else 'Auto'}, {vmax if vmax is not None else 'Auto'}]) 已成功保存至: {save_path}")

def save_distribution_plot(v_1d, save_path="ppr_distribution.png", cmap_name="viridis"):
    """
    将一维 PPR Tensor 绘制为颜色随数值渐变的条形图，并保存到本地。

    参数:
    v_1d (torch.Tensor): 一维的 PyTorch Tensor, 形状为 (x,)
    save_path (str): 本地保存图片的路径 (支持 .png, .jpg, .pdf 等)
    cmap_name (str): 颜色映射方案, 常用有 'viridis', 'plasma', 'inferno', 'YlGnBu'
    """
    # 1. 确保安全转换到 CPU 上的 NumPy 数组
    if isinstance(v_1d, torch.Tensor):
        # 展平成真正的一维，脱离计算图，移至 CPU，转为 numpy
        v_np = v_1d.view(-1).detach().cpu().numpy()
    else:
        v_np = v_1d

    x = len(v_np)

    # 2. 创建画布 (宽 15 保持长序列的稀疏可读性，高 5)
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)  # 300 DPI 适合论文和报告

    # 3. 设置数值到颜色的映射
    norm = plt.Normalize(v_np.min(), v_np.max())
    # 兼容新旧版本 Matplotlib 的 cmap 获取方式
    try:
        cmap = cm.get_cmap(cmap_name)
    except AttributeError:
        cmap = plt.colormaps[cmap_name]

    colors = cmap(norm(v_np))

    # 4. 绘制条形图 (width=1.0 且 edgecolor='none' 可以让柱子无缝拼接，呈连续颜色带)
    bars = ax.bar(range(x), v_np, color=colors, width=1.0, edgecolor='none')

    # 5. 右侧添加颜色渐变指示杆 (Colorbar)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="PPR Weight / Score")

    # 6. 完善图表基础信息
    ax.set_title("Personalized PageRank (PPR) Global Weight Distribution", fontsize=14, pad=15)
    ax.set_xlabel("Token / Node Index", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)  # 仅开启横向网格线，辅助观察高度

    # 自动紧凑布局，防止标签切边
    plt.tight_layout()

    # 7. 创建本地目录并保存
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # 释放内存，防止在循环中绘图导致内存泄漏
    print(f"🎉 PPR 权重分布图已成功保存至: {save_path}")


def power_iteration_ppr_tensor(A, s, alpha=0.15, max_iter=100, tol=1e-6):
    """
    使用幂迭代法在 Tensor 上计算 PPR
    A: 邻接矩阵 (NxN Tensor), 可以是稠密或稀疏 Tensor
    s: 个性化向量 (Nx1 Tensor) 或 批量个性化矩阵 (NxB Tensor)
    alpha: 跳转概率
    """
    num_nodes = A.size(0)

    # 1. 计算出度并构建转移矩阵 P (P = A / out_degrees)
    # 注意：处理孤立节点（出度为0），防止除以0
    out_degrees = torch.sum(A, dim=0, keepdim=True)
    out_degrees[out_degrees == 0] = 1.0
    P = A / out_degrees

    # 2. 初始化收敛向量 v
    v = s.clone().float()

    # 3. 迭代计算
    for i in range(max_iter):
        v_next = (1 - alpha) * torch.matmul(P, v) + alpha * s

        # 检查是否收敛 (L1 范数)
        if torch.norm(v_next - v, p=1) < tol:
            return v_next
        v = v_next

    return v

def personalized_pagerank(attention_matrix, initial_scores, alpha=0.85, max_iter=150, tol=1e-6):
    """
    基于个性化PageRank算法计算token最终分数

    参数:
    attention_matrix: [x, x] 注意力权重矩阵 (应该已经过行归一化)
    initial_scores: [x] 每个token的初始分数
    alpha: 阻尼因子 (teleportation probability)，通常0.85
    max_iter: 最大迭代次数
    tol: 收敛容忍度

    返回:
    final_scores: [x] 每个token的最终分数
    """
    # 确保注意力矩阵是行归一化的（每行和为1）
    row_sums = attention_matrix.sum(dim=1, keepdim=True)
    # 避免除以零，给极小值
    row_sums = torch.where(row_sums == 0, torch.tensor(1e-10), row_sums)
    normalized_attention = attention_matrix / row_sums

    # 归一化初始分数（使其和为1，成为概率分布）
    initial_probs = initial_scores
    # initial_probs = initial_scores / (initial_scores.max() + 1e-10)
    # torch.set_printoptions(threshold=10000)
    # print(initial_scores)
    # print(torch.sum(initial_scores))
    n_tokens = attention_matrix.size(0)
    scores = torch.ones(n_tokens, device=attention_matrix.device, dtype=attention_matrix.dtype) / n_tokens

    # PPR迭代
    for i in range(max_iter):
        # PPR公式: (1-alpha) * A^T * scores + alpha * initial_probs
        # 注意：这里使用A^T是因为我们想要从其他token传递到当前token
        new_scores = (1 - alpha) * torch.matmul(normalized_attention.t(), scores) + alpha * initial_probs

        # 检查收敛
        diff = torch.norm(new_scores - scores)
        scores = new_scores

        if diff < tol:
            print(f"PPR在迭代 {i + 1} 次后收敛，差异: {diff:.6f}")
            break

        # if i == max_iter - 1:
        #     print(f"PPR达到最大迭代次数 {max_iter}，最终差异: {diff:.6f}")

    return scores


def get_top_tokens(scores, attention_matrix=None, top_n=10):
    """
    获取最重要的token及其分析
    """
    # 获取top-n token的索引和分数
    top_scores, top_indices = torch.topk(scores, k=top_n)

    # print(f"\nTop-{top_n} 重要token:")
    # for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
    #     print(f"  {i + 1}. Token {idx}: 分数={score:.4f}")

    # 如果有注意力矩阵，还可以分析连接关系
    if attention_matrix is not None:
        print(f"\n重要token之间的注意力连接:")
        for i in range(min(top_n, len(top_indices))):
            idx_i = top_indices[i]
            for j in range(i + 1, min(top_n, len(top_indices))):
                idx_j = top_indices[j]
                attn_ij = attention_matrix[idx_i, idx_j]
                attn_ji = attention_matrix[idx_j, idx_i]
                if attn_ij > 0.1 or attn_ji > 0.1:  # 只显示较强的连接
                    print(f"  Token {idx_i} -> Token {idx_j}: {attn_ij:.3f}")
                    print(f"  Token {idx_j} -> Token {idx_i}: {attn_ji:.3f}")

    return top_indices, top_scores


def highlight_tokens(k_need_index, passages, tokenizer):
    """
    将passages中的token解码为字符串，并高亮显示k_need_index位置的token

    参数:
        k_need_index: List[int] - 需要高亮的token索引位置
        passages: List[torch.Tensor] - token张量列表
        tokenizer: transformers tokenizer - 用于解码token
    """
    # 将passages拼接成一个完整的token序列
    full_passage = torch.cat(passages).squeeze()  # [seq_len] 或 [batch, seq_len] -> [seq_len]

    # 解码完整的token序列
    decoded_text = tokenizer.decode(full_passage, skip_special_tokens=True)

    # 如果需要高亮的位置为空，直接打印完整文本
    if not k_need_index:
        print(f"完整文本:\n{decoded_text}")
        return

    print("-" * 50)

    # 获取所有token的字符串表示
    tokens = []
    for token_id in full_passage:
        token_str = tokenizer.decode(token_id, skip_special_tokens=True)
        # 清理解码结果（移除特殊字符和空格）
        tokens.append(token_str.strip())

    # 构建带高亮的文本
    highlighted_tokens = []
    for i, token in enumerate(tokens):
        if i in k_need_index:
            highlighted_tokens.append(f"\033[1;31m{token}\033[0m")  # 红色高亮
        else:
            highlighted_tokens.append(token)

    # 打印高亮后的文本
    highlighted_text = " ".join(highlighted_tokens)
    print(highlighted_text)

    print("\n" + "=" * 50 + "\n")

def get_str_list_len(passages: list[str]):
    return sum(len(s) for s in passages)

def find_sub_strlist_overlap(start_len: int, finish_len: int, combine_tokens: list[str]):
    start_idx = -1
    start_cutoff = 0
    end_idx = -1
    end_keep = 0
    for i in range(len(combine_tokens)):
        if get_str_list_len(combine_tokens[:i]) <= start_len < get_str_list_len(combine_tokens[:i + 1]):
            start_idx = i
            start_cutoff = start_len - get_str_list_len(combine_tokens[:i])
        if get_str_list_len(combine_tokens[:i]) < finish_len <= get_str_list_len(combine_tokens[:i + 1]):
            end_idx = i
            end_keep = finish_len - get_str_list_len(combine_tokens[:i])
    combine_tokens_new = copy.deepcopy(combine_tokens[start_idx: end_idx+1])
    if len(combine_tokens_new) == 0:
        ""
    elif len(combine_tokens_new) == 1:
        combine_tokens_new[0] = combine_tokens_new[0][start_cutoff: end_keep]
    else:
        combine_tokens_new[0] = combine_tokens_new[0][start_cutoff:]
        combine_tokens_new[-1] = combine_tokens_new[-1][:end_keep]
    ## 默认第一个都是不用算的
    if start_idx % 2 == 1:
        combine_tokens_new.insert(0, "")
    return combine_tokens_new

def find_recompute_tokens_within_passages(passages: list[str], combine_tokens: list[str]):
    all_recompute_tokens = []
    for p_idx, passage in enumerate(passages):
        combine_tokens_new = find_sub_strlist_overlap(
            start_len=get_str_list_len(passages[:p_idx]),
            finish_len=get_str_list_len(passages[:p_idx+1]),
            combine_tokens=combine_tokens
        )
        all_recompute_tokens.append(combine_tokens_new)
    return all_recompute_tokens


def highlight_tokens_compare(k_need_index, passages, tokenizer, query="", passages_str=None) -> Tuple[List[str], List[List[str]]]:
    """
    将passages中的token解码为字符串，并高亮显示k_need_index位置的token

    参数:
        k_need_index: List[int] - 需要高亮的token索引位置
        passages: List[torch.Tensor] - token张量列表
        tokenizer: transformers tokenizer - 用于解码token
    """
    # 将passages拼接成一个完整的token序列
    # print(f"[highlight_tokens_compare] k_need_index={k_need_index}")
    if type(passages) == list:
        full_passage = torch.cat(passages).squeeze()  # [seq_len] 或 [batch, seq_len] -> [seq_len]
    else:
        full_passage = passages

    combine_tokens = []
    combined_passages = []
    last_chosen = False
    last_tokens = []
    for i, token in enumerate(full_passage):
        if (i in k_need_index) == last_chosen:
            last_tokens.append(int(token))
        else:
            ## mengyao_debug: 状态转换了，从不需要重计算-》需要重计算 / 需要重计算-〉不需要
            ## 这个情况下第一个字符串肯定是不需要重计算的，sglang里面对齐的也是这个逻辑。
            last_chosen = i in k_need_index
            combined_passages.append(last_tokens)
            last_tokens = [int(token)]
    ##mengyao_debug: append the last one
    combined_passages.append(last_tokens)

    for i, sub_tokens in enumerate(combined_passages):
        previous_text_list = combined_passages[:i]
        cur_text_list = combined_passages[:i+1]
        previous_text_list_combine = sum(previous_text_list, [])
        cur_text_list_combine = sum(cur_text_list, [])
        previous_text = tokenizer.decode(previous_text_list_combine, skip_special_tokens=False)
        cur_text = tokenizer.decode(cur_text_list_combine, skip_special_tokens=False)
        combine_tokens.append(cur_text[len(previous_text):])

    all_recompute_tokens = []
    if passages_str is not None:
        all_recompute_tokens = find_recompute_tokens_within_passages(
            combine_tokens=combine_tokens,
            passages=passages_str
        )


    print("-" * 50)
    # 获取所有token的字符串表示
    tokens = []
    for token_id in full_passage:
        token_str = tokenizer.decode(token_id, skip_special_tokens=False)
        # 注意：这里不要strip()，保留原始解码结果
        tokens.append(token_str)

    # 构建带高亮的文本（用空格连接）
    highlighted_tokens = []
    for i, token in enumerate(tokens):
        if i in k_need_index:
            highlighted_tokens.append(f"\033[1;31m{token}\033[0m")  # 红色高亮
        else:
            highlighted_tokens.append(token)

    highlighted_with_spaces = "".join(highlighted_tokens)

    if len(k_need_index) > 0:
        print(f"query={query}\n")
        print(f"highlighted_with_spaces={highlighted_with_spaces}")
    return combine_tokens, all_recompute_tokens



# 更简单直接的版本
def highlight_tokens_simple(k_need_index, passages, tokenizer):
    """
    简单直接的方法：逐个字符处理
    """
    # 将passages拼接成一个完整的token序列
    full_passage = torch.cat(passages).squeeze()

    # 解码完整的token序列
    decoded_text = tokenizer.decode(full_passage, skip_special_tokens=True)

    if not k_need_index:
        print(f"完整文本:\n{decoded_text}")
        return

    print("-" * 50)

    # 获取所有token及其在解码文本中的位置
    tokens = []
    token_positions = []

    current_pos = 0
    for token_id in full_passage:
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        tokens.append(token_str)

        # 在解码文本中找到这个token
        if current_pos < len(decoded_text):
            # 尝试在当前位置找到token
            found_pos = decoded_text.find(token_str, current_pos)
            if found_pos != -1:
                token_positions.append((found_pos, found_pos + len(token_str)))
                current_pos = found_pos + len(token_str)
            else:
                # 如果没找到，假设它紧接在前一个token之后
                token_positions.append((current_pos, current_pos + len(token_str)))
                current_pos += len(token_str)
        else:
            token_positions.append((current_pos, current_pos + len(token_str)))
            current_pos += len(token_str)

    # 构建高亮文本
    highlighted_chars = list(decoded_text)

    # 对于每个需要高亮的token，在其周围插入颜色代码
    # 从后向前处理，避免索引变化
    for idx in sorted(k_need_index, reverse=True):
        if idx < len(token_positions):
            start, end = token_positions[idx]

            # 在token结束位置插入结束颜色代码
            highlighted_chars.insert(end, '\033[0m')

            # 在token开始位置插入开始颜色代码
            highlighted_chars.insert(start, '\033[1;31m')

    final_highlighted = "".join(highlighted_chars)
    print(final_highlighted)

    # 显示位置信息
    print("\n" + "=" * 50)
    print(f"高亮位置: {sorted(k_need_index)}")
    for idx in sorted(k_need_index):
        if idx < len(tokens):
            print(f"位置 {idx}: '{tokens[idx]}'")


# 最可靠的版本：使用tokenizer的convert_tokens_to_string
def highlight_tokens_reliable(k_need_index, passages, tokenizer):
    """
    最可靠的方法：使用tokenizer的内部方法
    """
    # 将passages拼接成一个完整的token序列
    full_passage = torch.cat(passages).squeeze()

    # 将token IDs转换为token字符串
    token_strings = tokenizer.convert_ids_to_tokens(full_passage.tolist())

    # 创建高亮版本的token列表
    highlighted_tokens = []
    for i, token in enumerate(token_strings):
        if i in k_need_index:
            # 在token周围添加颜色代码
            highlighted_tokens.append(f"\033[1;31m{token}\033[0m")
        else:
            highlighted_tokens.append(token)

    # 使用tokenizer正确连接tokens
    highlighted_text = tokenizer.convert_tokens_to_string(highlighted_tokens)

    print(highlighted_text)

    # 显示详细信息
    print("\n" + "=" * 50)
    print(f"总token数: {len(full_passage)}")
    print(f"高亮token数: {len(k_need_index)}")

    # 对于每个高亮的token，显示其原始形式和解码后的形式
    for idx in sorted(k_need_index):
        if idx < len(token_strings):
            # 获取token的字符串表示
            token_str = token_strings[idx]

            # 解码单个token看看它是什么
            decoded_token = tokenizer.decode([full_passage[idx]], skip_special_tokens=True)

            print(f"位置 {idx}: token='{token_str}', decoded='{decoded_token}'")




def topk_position_dispersion(
        tensor: torch.Tensor,
        top_percent: float = 0.1,
        method: str = "simple",
        visualize: bool = False
) -> Dict[str, Union[float, str, np.ndarray]]:
    """
    计算top k%大值的位置分布散度

    参数:
        tensor: 一维张量
        top_percent: 考虑前百分之几的大值 (0-1)
        method: 计算方法 ("simple", "comprehensive", "all")
        visualize: 是否可视化位置分布

    返回:
        包含散度指标和解释的字典
    """
    assert tensor.dim() == 1, "输入必须是一维张量"
    assert 0 < top_percent <= 1, "top_percent必须在(0,1]范围内"

    n = len(tensor)
    k = max(1, int(n * top_percent))  # 至少取1个

    # 1. 获取top k%大值的值和位置
    if k == n:  # 如果取全部
        top_values = tensor
        top_indices = torch.arange(n)
    else:
        top_values, top_indices = torch.topk(tensor, k=k)

    # 排序位置（重要！）
    sorted_indices = torch.sort(top_indices).values

    result = {}
    result["n_total"] = n
    result["k_top"] = k
    result["top_percent"] = top_percent
    result["top_indices"] = sorted_indices.cpu().numpy()

    # 2. 如果只有一个位置，散度为0（最集中）
    if k <= 1:
        result["dispersion"] = 0.0
        result["interpretation"] = "只有一个位置，完全集中"
        return result

    # 3. 计算位置间距
    sorted_indices = sorted_indices.float()
    gaps = sorted_indices[1:] - sorted_indices[:-1]
    result["gaps"] = gaps.cpu().numpy()
    result["gaps_mean"] = gaps.mean().item()
    result["gaps_std"] = gaps.std().item()

    # 4. 计算归一化位置（0到1之间）
    normalized_indices = sorted_indices.float() / (n - 1)  # 归一化到[0,1]
    result["normalized_indices"] = normalized_indices.cpu().numpy()

    # 5. 方法1：基于间距的变异系数
    def dispersion_v1(gaps_tensor):
        """基于间距的变异系数"""
        if gaps_tensor.mean() == 0:
            return 0.0  # 所有间距为0（不可能，除非相邻位置相同）
        cv = gaps_tensor.std() / gaps_tensor.mean()
        # 散度与CV正相关，但用sigmoid压缩
        dispersion = torch.sigmoid(cv - 1.0).item()  # CV=1时散度约0.5
        return dispersion

    # 6. 方法2：基于位置的基尼系数
    def dispersion_v2(indices_tensor):
        """基于位置值的基尼系数（位置越不均匀，散度越大）"""
        # 将位置视为权重
        sorted_positions = torch.sort(indices_tensor.float()).values
        n_pos = len(sorted_positions)

        if sorted_positions.sum() == 0:
            return 0.0

        # 基尼系数（位置的不平等性）
        cumulative = torch.cumsum(sorted_positions, dim=0)
        gini = 1 - 2 * torch.sum(cumulative) / (n_pos * sorted_positions.sum())

        # 位置分布越不均匀（基尼系数越高），散度越大
        return gini.item()

    # 7. 方法3：基于归一化位置的标准差
    def dispersion_v3(norm_indices):
        """基于归一化位置的标准差"""
        std = norm_indices.std()
        # 理论最大标准差（当位置均匀分布在两端）
        max_std = 0.5  # 当一半在0，一半在1时
        dispersion = (std / max_std).item()
        return min(dispersion, 1.0)

    # 8. 方法4：覆盖范围比例
    def dispersion_v4(indices_tensor, total_n):
        """位置覆盖范围占总长度的比例"""
        span = indices_tensor[-1] - indices_tensor[0]  # 最大-最小位置
        if total_n <= 1:
            return 0.0
        coverage = span / (total_n - 1)
        return min(coverage.item(), 1.0)

    # 9. 方法5：间距的熵（信息论方法）
    def dispersion_v5(gaps_tensor, eps=1e-12):
        """基于间距分布的熵"""
        # 归一化为概率分布
        gaps_pos = gaps_tensor - gaps_tensor.min() + eps
        prob = gaps_pos / gaps_pos.sum()

        # 计算熵
        entropy = -torch.sum(prob * torch.log(prob))

        # 最大熵（均匀分布）
        max_entropy = torch.log(torch.tensor(len(gaps_tensor), dtype=torch.float32))

        # 归一化熵
        norm_entropy = entropy / max_entropy

        # 熵越高，间距分布越均匀，散度越大？不一定，需要结合
        # 这里我们用熵来衡量间距的不确定性
        return norm_entropy.item()

    # 10. 方法6：聚类指标（位置是否成簇）
    def dispersion_v6(indices_tensor, n_clusters=3):
        """基于聚类假设的散度指标"""
        # 计算位置密度：每单位长度的点数
        span = indices_tensor[-1] - indices_tensor[0]
        if span == 0:
            return 0.0

        density = len(indices_tensor) / span.item()

        # 最大可能密度（连续位置）
        max_density = 1.0  # 每个位置都有点

        # 归一化密度
        norm_density = min(density / max_density, 1.0)

        # 密度越高，越集中，散度越小
        dispersion = 1.0 - norm_density
        return dispersion

    # 计算各种散度指标
    result["dispersion_cv"] = dispersion_v1(gaps)
    result["dispersion_gini"] = dispersion_v2(sorted_indices)
    result["dispersion_std"] = dispersion_v3(normalized_indices)
    result["dispersion_coverage"] = dispersion_v4(sorted_indices, n)
    result["dispersion_entropy"] = dispersion_v5(gaps)
    result["dispersion_cluster"] = dispersion_v6(sorted_indices)

    # 11. 综合散度指标（推荐）
    # 给不同指标赋予不同权重
    weights = {
        "cv": 0.25,  # 间距变异系数
        "coverage": 0.25,  # 覆盖范围
        "gini": 0.20,  # 位置基尼系数
        "cluster": 0.15,  # 聚类密度
        "std": 0.10,  # 归一化标准差
        "entropy": 0.05  # 间距熵
    }

    comprehensive_dispersion = (
            weights["cv"] * result["dispersion_cv"] +
            weights["coverage"] * result["dispersion_coverage"] +
            weights["gini"] * result["dispersion_gini"] +
            weights["cluster"] * result["dispersion_cluster"] +
            weights["std"] * result["dispersion_std"] +
            weights["entropy"] * result["dispersion_entropy"]
    )

    result["comprehensive_dispersion"] = comprehensive_dispersion

    # 12. 解释散度值
    def interpret_dispersion(score):
        if score < 0.2:
            return "位置高度集中（成簇）"
        elif score < 0.4:
            return "位置比较集中"
        elif score < 0.6:
            return "位置分布中等"
        elif score < 0.8:
            return "位置比较分散"
        else:
            return "位置高度分散"

    result["interpretation"] = interpret_dispersion(comprehensive_dispersion)

    # 14. 根据method参数返回不同格式的结果
    if method == "simple":
        return {
            "dispersion": comprehensive_dispersion,
            "interpretation": result["interpretation"],
            "k_top": k,
            "span": (sorted_indices[-1] - sorted_indices[0]).item()
        }
    elif method == "comprehensive":
        return {
            "comprehensive_dispersion": comprehensive_dispersion,
            "interpretation": result["interpretation"],
            "k_top": k,
            "span": (sorted_indices[-1] - sorted_indices[0]).item(),
            "components": {
                "coverage": result["dispersion_coverage"],
                "cv": result["dispersion_cv"],
                "gini": result["dispersion_gini"],
                "cluster": result["dispersion_cluster"]
            }
        }
    else:  # "all"
        return result


class OnlineEncoder:
    def __init__(self, llm_api_key:str):
        self.embedding_model_name = os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-v4")
        llm_base_url = os.getenv("LLM_EMBEDDING_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if llm_api_key == "":
            print(f"mengyao_debug fail to find llm_api_key, abort.")
            exit(1)

        print(f"LLM_BASE_URL is {llm_base_url}")

        self.client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url,
        )

    def get_sentence_embedding_dimension(self):
        return 1024

    def encode(self, text: Union[str, List[str]], batch_size=10, convert_to_tensor=False, device=None,
               normalize_embeddings=False, query_type="", max_concurrent_requests=10):
        """
        对文本进行嵌入编码，支持单个字符串或字符串列表输入

        Args:
            text: 输入文本，可以是单个字符串或字符串列表
            batch_size: 批处理大小
            convert_to_tensor: 是否转换为张量
            device: 设备信息
            normalize_embeddings: 是否对返回的向量进行归一化
            max_concurrent_requests: 最大并发请求数
        """

        prompt_prefixes = {
            'passage': 'Given a question, retrieve relevant documents that best answer the question.',
            'entity': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
            'edge': 'Given a question, retrieve relevant triplet facts that matches this question.',
            'fill_in_edge': 'Given a triples with only head and relation, retrieve relevant triplet facts that best fill the atomic query.'
        }

        if query_type in prompt_prefixes:
            prompt_prefix = prompt_prefixes[query_type]
            query_prefix = f"Instruct: {prompt_prefix}\nQuery: "
            if isinstance(text, str):
                text = f"{query_prefix}{text}"
            elif isinstance(text, list):
                text = [f"{query_prefix}{t}" for t in text]


        # 检查输入类型并统一处理
        is_single_string = isinstance(text, str)

        if is_single_string:
            text = [text]  # 将单个字符串转换为列表


        all_embeddings = []

        # 如果batch_size大于10，限制单个批次大小
        single_batch_size = min(batch_size, 10)  # max for aliyun

        # 计算需要多少个批次
        total_batches = (len(text) + single_batch_size - 1) // single_batch_size
        print(f"mengyao_debug total batches: {total_batches}, single batch size: {single_batch_size}")

        # 分批处理函数
        def process_batch(batch_index):
            start_idx = batch_index * single_batch_size
            end_idx = min(start_idx + single_batch_size, len(text))
            batch_texts = text[start_idx:end_idx]

            # print(f"mengyao_debug processing batch {batch_index + 1}/{total_batches}: {batch_texts}")

            # 调用API获取嵌入
            response = self.client.embeddings.create(
                input=batch_texts,
                model=self.embedding_model_name
            )
            print(f"mengyao_debug batch {batch_index + 1} response received")

            # 提取嵌入向量
            batch_embeddings = [item.embedding for item in response.data]
            return batch_embeddings, batch_index

        # 使用线程池并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_concurrent_requests, total_batches)) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(process_batch, i): i
                for i in range(total_batches)
            }

            # 收集结果并保持顺序
            batch_results = [None] * total_batches

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    batch_embeddings, _ = future.result()
                    batch_results[batch_index] = batch_embeddings
                    print(f"mengyao_debug batch {batch_index + 1} processed successfully")
                except Exception as exc:
                    print(f"mengyao_debug batch {batch_index + 1} generated an exception: {exc}")
                    # 如果某个批次失败，可以在这里处理重试逻辑

            # 按顺序合并所有批次的嵌入向量
            for batch_embeddings in batch_results:
                if batch_embeddings is not None:
                    all_embeddings.extend(batch_embeddings)

        # 转换为numpy数组
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        embeddings_array = np.ascontiguousarray(embeddings_array)

        # 向量归一化
        if normalize_embeddings:
            print("mengyao_debug normalizing embeddings")
            # 计算每个向量的L2范数（模长）
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            # 避免除以零，将零范数替换为1
            norms = np.where(norms == 0, 1, norms)
            # 归一化：每个向量除以其模长
            embeddings_array = embeddings_array / norms
            print(f"mengyao_debug normalized embeddings shape: {embeddings_array.shape}")

        # 根据输入类型决定输出格式
        if is_single_string:
            # 如果是单个字符串输入，返回单个向量
            result = embeddings_array[0]
        else:
            # 如果是列表输入，返回所有向量
            result = embeddings_array

        # 转换为张量（如果需要）
        if convert_to_tensor:
            import torch
            tensor_result = torch.tensor(result).detach()
            return tensor_result
        else:
            print(f"mengyao_debug returning array with shape: {result.shape}")
            return result


def calculate_vector_set_similarity(vectors):
    """
    计算一组向量的整体相似度

    Args:
        vectors: ndarray of shape (10, 1024)

    Returns:
        scalar: 表示这组向量整体相似度的值
    """
    # 计算所有向量两两之间的余弦相似度
    sim_matrix = cosine_similarity(vectors)  # 形状 (10, 10)

    # 排除对角线（自身与自身的相似度，总是1）
    mask = 1 - np.eye(len(vectors))
    pairwise_similarities = sim_matrix[mask == 1]

    # 返回平均相似度
    return np.mean(pairwise_similarities)
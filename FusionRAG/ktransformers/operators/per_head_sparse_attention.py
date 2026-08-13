# coding=utf-8
"""
Per-Head Sparse Attention Triton Kernel

实现逐头不同 attention mask 的 sparse attention。
每个 kv_head 可以有不同的 selected positions，因此 attention pattern 不同。
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def per_head_sparse_attention_kernel(
    Q,          # [batch, num_q_heads, q_len, head_dim]
    K,          # [batch, num_kv_heads, max_kv_len, head_dim] (padded)
    V,          # [batch, num_kv_heads, max_kv_len, head_dim] (padded)
    Q_idx,      # [batch, num_kv_heads, q_len] - 每个 query 对每个 kv_head 的 causal boundary
    KV_len,     # [num_kv_heads] - 每个 kv_head 的实际 K/V 长度
    Out,        # [batch, num_q_heads, q_len, head_dim]
    batch_size: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    q_len: tl.constexpr,
    max_kv_len: tl.constexpr,
    head_dim: tl.constexpr,
    num_q_per_kv: tl.constexpr,  # GQA: num_q_heads // num_kv_heads
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_idxb, stride_idxh, stride_idxm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Per-head sparse attention kernel with FlashAttention-style computation.

    每个 kv_head 有不同的 Q_idx，控制每个 query 可以 attend 到的最大 position。
    对于 GQA，同一个 kv_head 对应的多个 q_heads 共享相同的 K/V 和 Q_idx。
    """
    # Program IDs
    pid_m = tl.program_id(0)  # Query block index
    pid_bh = tl.program_id(1)  # batch * num_q_heads

    batch_idx = pid_bh // num_q_heads
    q_head_idx = pid_bh % num_q_heads
    kv_head_idx = q_head_idx // num_q_per_kv  # GQA mapping

    # Early exit if out of bounds
    if pid_m * BLOCK_M >= q_len:
        return

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Get actual KV length for this head
    kv_len = tl.load(KV_len + kv_head_idx)

    # Pointers
    q_ptrs = Q + batch_idx * stride_qb + q_head_idx * stride_qh + \
             offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + batch_idx * stride_kb + kv_head_idx * stride_kh + \
             offs_d[:, None] * stride_kk  # Will add N offset in loop
    v_ptrs = V + batch_idx * stride_vb + kv_head_idx * stride_vh + \
             offs_d[None, :] * stride_vk  # Will add N offset in loop
    o_ptrs = Out + batch_idx * stride_ob + q_head_idx * stride_oh + \
             offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # Load Q_idx for this head and these query positions
    idx_ptrs = Q_idx + batch_idx * stride_idxb + kv_head_idx * stride_idxh + \
               offs_m * stride_idxm
    q_mask = offs_m < q_len
    q_boundary = tl.load(idx_ptrs, mask=q_mask, other=0)  # [BLOCK_M]
    max_boundary = tl.max(q_boundary)

    # Load Q and cast to float32 for computation
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim), other=0.0)
    q = q.to(tl.float32)

    # Scale Q
    scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))
    q = q * scale

    # Initialize accumulators for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over K/V blocks
    for start_n in range(0, tl.cdiv(max_boundary + 1, BLOCK_N) * BLOCK_N, BLOCK_N):
        cols = start_n + offs_n

        # Load K block and cast to float32
        k_block_ptrs = k_ptrs + cols[None, :] * stride_kn
        k = tl.load(k_block_ptrs,
                    mask=(cols[None, :] < kv_len) & (offs_d[:, None] < head_dim),
                    other=0.0)
        k = k.to(tl.float32)

        # Compute QK^T
        qk = tl.dot(q, k)

        # Apply causal mask using q_boundary
        # Each query position can only attend to positions <= its boundary
        qk = tl.where(cols[None, :] <= q_boundary[:, None], qk, float("-inf"))

        # Also mask out positions beyond kv_len
        qk = tl.where(cols[None, :] < kv_len, qk, float("-inf"))

        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # Update accumulator
        acc = acc * alpha[:, None]

        # Load V block and cast to float32
        v_block_ptrs = v_ptrs + cols[:, None] * stride_vn
        v = tl.load(v_block_ptrs,
                    mask=(cols[:, None] < kv_len) & (offs_d[None, :] < head_dim),
                    other=0.0)
        v = v.to(tl.float32)

        # Accumulate weighted values
        acc += tl.dot(p, v)

        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim))


def per_head_sparse_attention(
    q: torch.Tensor,        # [batch, num_q_heads, q_len, head_dim]
    k: torch.Tensor,        # [batch, num_kv_heads, max_kv_len, head_dim]
    v: torch.Tensor,        # [batch, num_kv_heads, max_kv_len, head_dim]
    q_idx: torch.Tensor,    # [batch, num_kv_heads, q_len] - causal boundaries
    kv_len: torch.Tensor,   # [num_kv_heads] - actual lengths
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    Per-head sparse attention entry function.

    Args:
        q: Query tensor [batch, num_q_heads, q_len, head_dim]
        k: Key tensor [batch, num_kv_heads, max_kv_len, head_dim] (padded)
        v: Value tensor [batch, num_kv_heads, max_kv_len, head_dim] (padded)
        q_idx: Per-head causal boundaries [batch, num_kv_heads, q_len]
               q_idx[b, h, i] = max position that query i can attend to for head h
        kv_len: Actual K/V length per head [num_kv_heads]
        block_m: Block size for query dimension
        block_n: Block size for key dimension

    Returns:
        Output tensor [batch, num_q_heads, q_len, head_dim]
    """
    batch_size, num_q_heads, q_len, head_dim = q.shape
    _, num_kv_heads, max_kv_len, _ = k.shape

    assert num_q_heads % num_kv_heads == 0, "GQA: num_q_heads must be divisible by num_kv_heads"
    num_q_per_kv = num_q_heads // num_kv_heads

    # Ensure tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_idx = q_idx.contiguous().to(torch.int32)
    kv_len = kv_len.contiguous().to(torch.int32)

    # Output tensor
    out = torch.empty_like(q)

    # Pad head_dim to power of 2 if needed
    block_d = triton.next_power_of_2(head_dim)

    # Grid
    grid = (triton.cdiv(q_len, block_m), batch_size * num_q_heads)

    # Launch kernel
    per_head_sparse_attention_kernel[grid](
        q, k, v, q_idx, kv_len, out,
        batch_size, num_q_heads, num_kv_heads, q_len, max_kv_len, head_dim, num_q_per_kv,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        q_idx.stride(0), q_idx.stride(1), q_idx.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    return out


def prepare_per_head_kv(
    prefix_k: torch.Tensor,     # [1, num_kv_heads, prefix_len, head_dim]
    prefix_v: torch.Tensor,     # [1, num_kv_heads, prefix_len, head_dim]
    critical_k: torch.Tensor,   # [1, num_kv_heads, num_critical, head_dim]
    critical_v: torch.Tensor,   # [1, num_kv_heads, num_critical, head_dim]
    query_k: torch.Tensor,      # [1, num_kv_heads, query_len, head_dim]
    query_v: torch.Tensor,      # [1, num_kv_heads, query_len, head_dim]
    head_critical_masks: list,  # List of [num_critical] bool tensors, one per kv_head
    prefix_len: int,
    query_len: int,
    num_critical: int,
    device: torch.device,
):
    """
    Prepare per-head K/V tensors and q_idx for sparse attention.

    Each kv_head has different selected critical positions.
    We need to:
    1. Build per-head K/V sequences (pad to max length)
    2. Build per-head q_idx (causal boundaries)
    """
    num_kv_heads = prefix_k.shape[1]
    head_dim = prefix_k.shape[3]
    sparse_len = num_critical + query_len

    # Find max K/V length across all heads
    head_kv_lens = []
    for h in range(num_kv_heads):
        num_head_critical = head_critical_masks[h].sum().item()
        head_kv_len = prefix_len + num_head_critical + query_len
        head_kv_lens.append(head_kv_len)

    max_kv_len = max(head_kv_lens)

    # Allocate padded K/V tensors
    padded_k = torch.zeros(1, num_kv_heads, max_kv_len, head_dim, device=device, dtype=prefix_k.dtype)
    padded_v = torch.zeros(1, num_kv_heads, max_kv_len, head_dim, device=device, dtype=prefix_v.dtype)

    # Allocate q_idx: [1, num_kv_heads, sparse_len]
    q_idx = torch.zeros(1, num_kv_heads, sparse_len, device=device, dtype=torch.int32)

    # Allocate kv_len: [num_kv_heads]
    kv_len = torch.tensor(head_kv_lens, device=device, dtype=torch.int32)

    # Fill per-head data
    for h in range(num_kv_heads):
        mask = head_critical_masks[h]
        num_head_critical = mask.sum().item()
        head_kv_len = head_kv_lens[h]

        # Copy prefix
        padded_k[0, h, :prefix_len, :] = prefix_k[0, h, :, :]
        padded_v[0, h, :prefix_len, :] = prefix_v[0, h, :, :]

        # Copy selected critical
        if num_head_critical > 0:
            selected_indices = mask.nonzero(as_tuple=True)[0]
            padded_k[0, h, prefix_len:prefix_len + num_head_critical, :] = critical_k[0, h, selected_indices, :]
            padded_v[0, h, prefix_len:prefix_len + num_head_critical, :] = critical_v[0, h, selected_indices, :]

        # Copy query
        padded_k[0, h, prefix_len + num_head_critical:head_kv_len, :] = query_k[0, h, :, :]
        padded_v[0, h, prefix_len + num_head_critical:head_kv_len, :] = query_v[0, h, :, :]

        # Build q_idx for this head
        # Create mapping: critical position index -> head_critical index
        critical_to_head_idx = {}
        if num_head_critical > 0:
            selected_indices = mask.nonzero(as_tuple=True)[0].tolist()
            for idx, pos in enumerate(selected_indices):
                critical_to_head_idx[pos] = idx

        for sparse_idx in range(sparse_len):
            if sparse_idx < num_critical:
                # Critical token
                if mask[sparse_idx].item():
                    head_idx = critical_to_head_idx[sparse_idx]
                    q_idx[0, h, sparse_idx] = prefix_len + head_idx
                else:
                    # Not selected by this head, attend to prefix only
                    # (prefix_len - 1 means can attend to all prefix tokens)
                    q_idx[0, h, sparse_idx] = prefix_len - 1
            else:
                # Query token
                q_offset = sparse_idx - num_critical
                q_idx[0, h, sparse_idx] = prefix_len + num_head_critical + q_offset

    return padded_k, padded_v, q_idx, kv_len

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton ``moe_align_block_size`` (issue #4): sort/pad routed token-slots into
per-expert blocks, the vLLM/SGLang-style histogram + padded prefix-sum + scatter.

The fused-MoE block GEMM consumes ``(sorted_token_ids, expert_ids,
num_tokens_post_padded)`` so each ``BLOCK_SIZE_M`` row tile maps to exactly one
expert. The reference (``moe_align_block_size_ref``) is a correctness-first torch
loop (argsort + per-expert python padding); this is the on-device port.

Algorithm (mirrors vLLM's ``moe_align_block_size_triton`` 4-stage kernel, with a
vectorized ``expert_ids`` pass instead of a data-dependent fill loop so it also
runs unchanged under ``TRITON_INTERPRET=1``):

The flattened ``M*top_k`` token-slots are split into ``num_experts`` contiguous
chunks, one per program. ``tokens_cnts`` is a ``[num_experts+1, num_experts]``
matrix: row ``p+1`` = histogram of program ``p``'s chunk over experts.

* **stage1 (count)**  each program tallies its chunk into row ``pid+1``.
* **stage2 (cumsum down programs)**  column ``e`` is prefix-summed over the
  program rows, so ``tokens_cnts[p, e]`` becomes the count of expert ``e`` in
  chunks ``0..p-1`` — i.e. program ``p``'s base offset inside expert ``e``'s run.
* **stage3 (padded prefix-sum)**  a single program rounds each expert's total up
  to a ``block_size`` multiple into ``cumsum[e+1]`` and writes
  ``num_tokens_post_padded``.
* **expert_ids**  one program per block ``b`` finds its owning expert as
  ``#{e in [1, num_experts] : cumsum[e] <= b*block_size}`` (a vectorized scan of
  the monotone ``cumsum``).
* **stage4 (scatter)**  each program walks its chunk again and writes token-slot
  ``t`` to ``cumsum[e] + tokens_cnts[pid, e]`` (post-incremented).

Because chunks are contiguous and walked in order, within-expert ordering is the
same stable order as the reference's ``argsort(stable=True)``, so the full output
triple matches the reference bit-for-bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["moe_align_block_size_triton"]

# Sentinel for masked-out cumsum lanes. Must be a `tl.constexpr` instance (not a
# plain module global) to be readable from inside a @triton.jit function — the
# real compiler rejects bare globals even though the interpreter tolerates them.
_I32_MAX = tl.constexpr(2147483647)


@triton.jit
def _align_stage1_count(
    topk_ids_ptr,
    tokens_cnts_ptr,  # [num_experts+1, num_experts] int32, zero-initialised
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    """Per-program histogram of its token chunk into ``tokens_cnts[pid+1, :]``."""
    pid = tl.program_id(0)
    start = pid * tokens_per_thread
    row = (pid + 1) * num_experts
    for i in range(tokens_per_thread):
        if start + i < numel:
            e = tl.load(topk_ids_ptr + start + i)
            cur = tl.load(tokens_cnts_ptr + row + e)
            tl.store(tokens_cnts_ptr + row + e, cur + 1)


@triton.jit
def _align_stage2_cumsum(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    """Prefix-sum expert column ``pid`` down the program rows (in place)."""
    pid = tl.program_id(0)  # expert column
    last = 0
    for i in range(1, num_experts + 1):
        cur = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last = last + cur
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last)


@triton.jit
def _align_stage3_pad(
    num_tokens_post_padded_ptr,  # [1] int32
    tokens_cnts_ptr,
    cumsum_ptr,  # [num_experts+1] int32, zero-initialised
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    """Round each expert's total up to a ``block_size`` multiple -> ``cumsum``."""
    last = 0
    final_row = num_experts * num_experts  # row `num_experts` holds the totals
    for i in range(1, num_experts + 1):
        cnt = tl.load(tokens_cnts_ptr + final_row + i - 1)
        last = last + tl.cdiv(cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last)
    tl.store(num_tokens_post_padded_ptr, last)


@triton.jit
def _align_expert_ids(
    cumsum_ptr,
    expert_ids_ptr,  # [num_m_blocks] int32
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    EXPERTS_P2: tl.constexpr,  # next_pow2(num_experts + 1)
):
    """Block ``b`` belongs to ``#{e in [1, num_experts] : cumsum[e] <= b*block}``.

    Vectorized scan of the monotone ``cumsum``. ``EXPERTS_P2`` is
    ``next_pow2(num_experts + 1)`` because ``tl.arange`` needs a power-of-2 length
    and the lanes must reach index ``num_experts`` (we read ``cumsum[1..num_experts]``);
    masked / out-of-range lanes load the ``_I32_MAX`` sentinel so they never count.
    """
    b = tl.program_id(0)
    off = b * block_size
    e = tl.arange(0, EXPERTS_P2)
    valid = (e >= 1) & (e <= num_experts)
    cs = tl.load(cumsum_ptr + e, mask=valid, other=_I32_MAX)
    expert = tl.sum(((cs <= off) & valid).to(tl.int32), axis=0)
    tl.store(expert_ids_ptr + b, expert)


@triton.jit
def _align_stage4_scatter(
    topk_ids_ptr,
    sorted_token_ids_ptr,  # [max_pad] int32, pre-filled with pad_id
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    """Scatter each token-slot to ``cumsum[e] + tokens_cnts[pid, e]`` (post-inc)."""
    pid = tl.program_id(0)
    start = pid * tokens_per_thread
    row = pid * num_experts  # this program's base offsets (from stage2)
    for i in range(tokens_per_thread):
        if start + i < numel:
            e = tl.load(topk_ids_ptr + start + i)
            cnt = tl.load(tokens_cnts_ptr + row + e)
            rank = cnt + tl.load(cumsum_ptr + e)
            tl.store(sorted_token_ids_ptr + rank, start + i)
            tl.store(tokens_cnts_ptr + row + e, cnt + 1)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort/pad routed token-slots into per-expert blocks (issue #4, Triton).

    Drop-in for ``moe_align_block_size_ref`` — identical signature and outputs.

    Args:
        topk_ids: ``[M, top_k]`` int32 expert id per token-slot.
        block_size: GEMM block size each expert run is padded up to.
        num_experts: number of experts.

    Returns:
        ``(sorted_token_ids [max_pad], expert_ids [n // block_size],
        num_tokens_post_padded [1])`` where ``max_pad = M*top_k +
        (num_experts+1)*(block_size-1)``, ``n = num_tokens_post_padded`` and
        unused ``sorted_token_ids`` slots hold ``pad_id = M*top_k``.
    """
    flat = topk_ids.reshape(-1).contiguous()
    numel = flat.numel()
    pad_id = numel
    device = topk_ids.device

    max_pad = numel + (num_experts + 1) * (block_size - 1)
    max_blocks = triton.cdiv(max_pad, block_size)

    sorted_ids = torch.full((max_pad,), pad_id, dtype=torch.int32, device=device)
    expert_ids = torch.zeros((max_blocks,), dtype=torch.int32, device=device)
    num_post = torch.zeros((1,), dtype=torch.int32, device=device)
    tokens_cnts = torch.zeros((num_experts + 1, num_experts), dtype=torch.int32, device=device)
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=device)

    tokens_per_thread = triton.cdiv(numel, num_experts)
    grid = (num_experts,)

    _align_stage1_count[grid](flat, tokens_cnts, num_experts, numel, tokens_per_thread)
    _align_stage2_cumsum[grid](tokens_cnts, num_experts)
    _align_stage3_pad[(1,)](num_post, tokens_cnts, cumsum, num_experts, block_size)
    _align_expert_ids[(max_blocks,)](
        cumsum,
        expert_ids,
        num_experts,
        block_size,
        EXPERTS_P2=triton.next_power_of_2(num_experts + 1),
    )
    _align_stage4_scatter[grid](
        flat, sorted_ids, tokens_cnts, cumsum, num_experts, numel, tokens_per_thread
    )

    # Reference returns expert_ids truncated to the blocks actually used.
    n = int(num_post.item())
    return sorted_ids, expert_ids[: n // block_size], num_post


register("moe_align_block_size", Backend.TRITON)(moe_align_block_size_triton)

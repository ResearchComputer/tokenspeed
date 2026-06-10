# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Optimized INT4 W4A16 grouped fused-MoE GEMM for AMD MI300A (gfx942, CDNA3).

This is a standalone, autotuned + microbenchmarked version of the in-kernel
INT4-dequant fused-MoE GEMM that ships inside tokenspeed
(``ops/moe/triton.py::fused_moe_kernel``, ``use_int4_w4a16`` branch). It tracks
issue #1 of ResearchComputer/kernels.

Why Triton (not Gluon) here
---------------------------
tokenspeed's ``AGENTS.md`` prefers Triton Gluon for AMD kernels, but it also
states Triton is the right choice for *portable* solutions and that vendor /
backend-specific solutions are a consolidation target rather than a hard
requirement for a first cut. This kernel is intentionally portable Triton so it:

* runs under ``TRITON_INTERPRET=1`` for CPU correctness checking with no GPU,
* compiles unchanged on both NVIDIA and AMD for parity testing,
* exposes the CDNA3 tuning knobs (``waves_per_eu``, ``matrix_instr_nonkdim``,
  ``kpack``) through the autotune ``extra`` kwargs path used elsewhere in
  tokenspeed (see ``ops/attention/triton/mha_decode.py``).

A Gluon rewrite (explicit ``BlockedLayout`` + ``amd_mfma`` + LDS double-buffer,
mirroring ``ops/moe/gluon.py``) is the follow-up once the config space below has
been validated on real gfx942 hardware.

Compute pattern (W4A16, compressed-tensors ``pack-quantized``)
--------------------------------------------------------------
* Per expert ``e``, weight ``W_e`` is INT4, symmetric group-quantized along K
  (``group_size=32``), stored ``uint4b8`` (unsigned nibble, subtract 8 -> signed
  ``[-8, 7]``).
* Packed ``B``: ``[E, N, K // 8]`` int32, 8 nibbles per int32, low nibble = lowest
  K index. Scales ``S``: ``[E, N, K // group]`` bf16.
* ``W_e[n, k] = (((B[e, n, k // 8] >> (4 * (k % 8))) & 0xF) - 8) * S[e, n, k // group]``.
* Fused MoE: activations ``A [M, K]`` bf16 routed to top-k experts; grouped GEMM
  ``A @ W_e^T`` driven by ``sorted_token_ids`` / ``expert_ids``, unpacking and
  scaling inline, accumulating in fp32.

Key optimizations vs the in-tree kernel
----------------------------------------
1. **Autotuning** with a CDNA3-reasoned config space (see ``configs.py``) keyed on
   the GEMM shape; the in-tree kernel uses a single hardcoded small-M config.
2. **K packed along the contraction without per-element shifts in the hot path.**
   We load one ``int32`` per 8 logical-K elements (``BLOCK_SIZE_K // 8`` int32 per
   row of the tile) and unpack all 8 nibbles with a single broadcasted shift over a
   length-8 ``constexpr`` vector, so the unpack is amortized over 8 MACs and the
   weight read is one coalesced int32 load (4x fewer bytes than bf16).
3. **Group-scale reloaded once per K-group, not per K-element.** With
   ``BLOCK_SIZE_K`` a multiple of ``group_size`` the scale tile is ``[N, BLOCK_K //
   group]`` and is broadcast across the 32 K within a group.
4. **MFMA-friendly tiles**: ``BLOCK_SIZE_K`` is a multiple of 8 (pack factor) and of
   the group size (32); ``matrix_instr_nonkdim=16`` selects the 16x16 MFMA on CDNA3;
   ``waves_per_eu`` is tuned so the small-M decode tiles do not over-occupy and
   spill VGPRs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register
from ..w4a16 import moe_align_block_size_ref
from .configs import get_autotune_configs, prune_configs

__all__ = ["int4_w4a16_moe_gemm", "fused_moe_int4_kernel"]


@triton.autotune(
    configs=get_autotune_configs(),
    key=["N", "K", "EM", "num_valid_tokens"],
    prune_configs_by={"early_config_prune": prune_configs},
)
@triton.heuristics(
    {
        # Computed AFTER autotune picks BLOCK_SIZE_K (heuristics is the inner
        # decorator). K is the logical (unpacked) contraction length and is
        # normally a multiple of pack(8) and group(32), but guard it anyway so
        # K masking is only emitted when actually needed.
        "EVEN_K": lambda a: a["K"] % a["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def fused_moe_int4_kernel(
    a_ptr,  # [M, K] bf16 activations (token rows, pre-permute)
    b_ptr,  # [E, N, K // 8] int32 packed uint4b8 weights
    c_ptr,  # [EM, N] or [M, top_k, N] output
    b_scale_ptr,  # [E, N, K // group_k] bf16 group scales
    topk_weights_ptr,  # [num_valid_tokens] fp32 routing weights (sorted order)
    sorted_token_ids_ptr,  # [EM] int32 token-slot ids grouped by expert
    expert_ids_ptr,  # [num_m_blocks] int32 expert per M-block (-1 = filtered)
    num_tokens_post_padded_ptr,  # [1] int32
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,  # stride over the packed-K (int32) dim
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsn,
    stride_bsk,  # stride over the group dim of the scale tensor
    group_k: tl.constexpr,  # quant group size along K (e.g. 32)
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    FILTER_EXPERT: tl.constexpr,
    # AMD/CDNA3 lowering knobs. Declared as (unused) constexpr so the same
    # autotune ``Config.kwargs`` work on both stock Triton (which forwards
    # Config kwargs as kernel args -> accepted here, ignored in codegen) and the
    # tokenspeed_triton ROCm fork (whose AMD backend reads matrix_instr_nonkdim /
    # waves_per_eu / kpack from the kernel specialization to pick the MFMA shape,
    # occupancy hint, and K-packing). They have no effect on the math.
    waves_per_eu: tl.constexpr = 0,
    matrix_instr_nonkdim: tl.constexpr = 16,
    kpack: tl.constexpr = 2,
):
    """Grouped per-expert INT4 W4A16 GEMM with inline dequant.

    Each program computes one ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` output tile for a
    single expert. The packed int4 weights are unpacked and group-dequantized in
    the K loop and the dequantized rhs is cast to the activation dtype before
    ``tl.dot`` (matching the in-tree kernel: ``tl.dot`` asserts equal operand
    dtypes, and the in-tensor dtype is the safe choice).
    """
    PACK: tl.constexpr = 8  # nibbles packed per int32
    BLOCK_K_PACK: tl.constexpr = BLOCK_SIZE_K // PACK  # int32 per K-block row

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if FILTER_EXPERT and off_experts == -1:
        # Filtered (EP) block: write zeros and exit. At EP=8 most blocks for a
        # given rank are *not* filtered, but routed tokens whose expert lives on
        # another rank still produce a -1 block that must zero its output slot.
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    # A pointers: gather token rows (de-duplicated by top_k for gate_up; the
    # caller passes the already-permuted activation for the down GEMM).
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    # Packed-B pointers: one int32 per (packed-K, N). We index the packed-K dim
    # by ``offs_kp`` (BLOCK_K_PACK entries) and broadcast the 8 nibbles inside the
    # loop, so the global load is a single coalesced int32 tile per step.
    offs_kp = tl.arange(0, BLOCK_K_PACK)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_kp[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    # Nibble shift LUT: shifts[i] = 4 * i, i in [0, 8). Applied to the broadcast
    # int32 to extract the 8 packed values; this is the only per-element unpack
    # work and it is shared across the whole N tile.
    shifts = (tl.arange(0, PACK) * 4).to(tl.int32)  # [PACK]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // group_k
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for kb in range(0, num_k_blocks):
        k0 = kb * BLOCK_SIZE_K
        # ---- load activation tile [BLOCK_M, BLOCK_K] ----
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k0),
                other=0.0,
            )

        # ---- load packed weight tile [BLOCK_K_PACK, BLOCK_N] int32 ----
        kp0 = k0 // PACK
        if EVEN_K:
            bpack = tl.load(b_ptrs)
        else:
            kp_mask = (kp0 + offs_kp) < (K // PACK)
            bpack = tl.load(b_ptrs, mask=kp_mask[:, None], other=0)
        bpack = bpack.to(tl.int32)

        # ---- unpack 8 nibbles -> [BLOCK_K_PACK, PACK, BLOCK_N] -> [BLOCK_K, BLOCK_N] ----
        # bpack: [KP, N]; shifts: [PACK]. Broadcast to [KP, PACK, N].
        nib = (bpack[:, None, :] >> shifts[None, :, None]) & 0xF  # [KP, PACK, N]
        b_int = nib.to(tl.float32) - 8.0  # uint4b8 -> signed [-8, 7]
        # Reshape so K is contiguous low-first: logical k = kp * 8 + nibble.
        b_int = tl.reshape(b_int, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        # ---- group-scale tile [GROUPS_PER_BLOCK, BLOCK_N], reloaded once / group ----
        g0 = k0 // group_k
        offs_g = g0 + tl.arange(0, GROUPS_PER_BLOCK)
        bscl = tl.load(
            b_scale_ptr
            + off_experts * stride_bse
            + offs_g[:, None] * stride_bsk
            + offs_bn[None, :] * stride_bsn,
            mask=(offs_g[:, None] < tl.cdiv(K, group_k)),
            other=0.0,
        ).to(tl.float32)  # [GPB, N]
        # Expand each group scale across its ``group_k`` K rows: [GPB, N] ->
        # [GPB, group_k, N] -> [BLOCK_K, N]. One scale fetch per group, not per K.
        bscl = tl.broadcast_to(
            bscl[:, None, :], (GROUPS_PER_BLOCK, group_k, BLOCK_SIZE_N)
        )
        bscl = tl.reshape(bscl, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b_deq = (b_int * bscl).to(a.dtype)
        accumulator = tl.dot(a, b_deq, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_K_PACK * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def int4_w4a16_moe_gemm(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    top_k: int,
    group_size: int,
    mul_routed_weight: bool,
    compute_type: tl.dtype = tl.bfloat16,
    filter_expert: bool = True,
) -> torch.Tensor:
    """Launch the grouped INT4 W4A16 fused-MoE GEMM.

    Args:
        a: ``[num_valid_tokens // top_k, K]`` bf16 activations. The kernel gathers
            rows via ``sorted_token_ids // top_k`` so this is the *un-repeated*
            token tensor (the standard fused-MoE convention).
        b_packed: ``[E, N, K // 8]`` int32 packed ``uint4b8`` weights (8 nibbles per
            int32, low nibble = lowest K index).
        b_scale: ``[E, N, K // group_size]`` bf16 symmetric group scales.
        c: output, ``[num_valid_tokens, N]`` == ``[M * top_k, N]``, caller-allocated.
            The kernel writes **token-indexed** rows (``c[offs_token]``), matching
            the in-tree fused-MoE convention: row ``t = m * top_k + j`` holds the
            ``j``-th expert result for token ``m``, so the downstream reduce is a
            plain ``c.view(M, top_k, N).sum(1)`` (with the routing weight already
            folded for the down GEMM). No un-permute by ``sorted_token_ids`` is
            needed.
        topk_weights: ``[num_valid_tokens]`` fp32 routing weights in sorted order
            (only read when ``mul_routed_weight``).
        sorted_token_ids: ``[EM]`` int32, token-slot ids grouped/padded per expert.
        expert_ids: ``[num_m_blocks]`` int32 expert per M-block (``-1`` = filtered).
        num_tokens_post_padded: ``[1]`` int32, padded token count.
        top_k: experts per token (gate_up GEMM uses real top_k; down GEMM uses 1).
        group_size: quant group size along K.
        mul_routed_weight: fold routing weight into the output (down GEMM).
        compute_type: output / accumulate-cast dtype.
        filter_expert: honor ``-1`` expert ids (EP). Set False when no filtering.

    Returns:
        ``c`` (written in place).
    """
    assert b_packed.dtype == torch.int32
    assert sorted_token_ids.stride(0) == 1
    E, N, kp = b_packed.shape
    K = kp * 8
    assert K % group_size == 0
    assert b_scale.shape == (E, N, K // group_size)

    num_valid_tokens = topk_weights.numel() if topk_weights is not None else a.shape[0] * top_k

    def grid(meta):
        return (
            triton.cdiv(sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    fused_moe_int4_kernel[grid](
        a,
        b_packed,
        c,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        c.stride(0),
        c.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        group_k=group_size,
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=compute_type,
        FILTER_EXPERT=filter_expert,
    )
    return c


# --- xkernels backend registration -----------------------------------------
# Thin adapter exposing the same backend-agnostic [M, N] signature as the
# reference (``moe_w4a16_ref``): build the per-expert dispatch, launch into a
# token-indexed scratch buffer, then reduce ``view(M, top_k, N).sum(1)``.


def _moe_int4_w4a16_triton(
    A: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    group_size: int = 32,
    mul_routed_weight: bool = True,
) -> torch.Tensor:
    M, top_k = topk_ids.shape
    E, N, _ = packed.shape
    block_m = 16 if M <= 32 else 64
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    c = torch.zeros((M * top_k, N), dtype=A.dtype, device=A.device)
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float32
    int4_w4a16_moe_gemm(
        A,
        packed,
        scale,
        c,
        topk_w.reshape(-1).float(),
        sorted_ids,
        expert_ids,
        num_post,
        top_k=top_k,
        group_size=group_size,
        mul_routed_weight=mul_routed_weight,
        compute_type=compute_type,
        filter_expert=False,
    )
    return c.view(M, top_k, N).sum(dim=1)


register("moe_int4_w4a16", Backend.TRITON)(_moe_int4_w4a16_triton)

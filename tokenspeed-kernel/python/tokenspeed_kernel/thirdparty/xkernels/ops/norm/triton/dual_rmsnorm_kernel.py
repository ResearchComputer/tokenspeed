# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Single-launch fused parallel dual RMSNorm (issue #2) for AMD MI300A (gfx942).

NVIDIA has a single fused kernel for the MLA ``q_a`` / ``kv_a`` latents; AMD
falls back to two sequential RMSNorm launches. This kernel does both latents in
one launch: one program per token row normalizes ``x1`` over ``d1`` and ``x2``
over ``d2`` (each loaded as a single masked tile of ``next_pow2(d)`` elements),
reducing in fp32. The win is one kernel / one pass instead of two.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["dual_rmsnorm_triton", "dual_rmsnorm_kernel"]


@triton.jit
def dual_rmsnorm_kernel(
    x1_ptr,
    w1_ptr,
    o1_ptr,
    x2_ptr,
    w2_ptr,
    o2_ptr,
    x1_row_stride,
    o1_row_stride,
    x2_row_stride,
    o2_row_stride,
    d1,
    d2,
    eps,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    row = tl.program_id(axis=0)

    # ---- latent 1: rmsnorm over d1 ----
    cols1 = tl.arange(0, BLOCK_D1)
    m1 = cols1 < d1
    x1 = tl.load(x1_ptr + row * x1_row_stride + cols1, mask=m1, other=0.0).to(tl.float32)
    inv1 = tl.rsqrt(tl.sum(x1 * x1, axis=0) / d1 + eps)
    w1 = tl.load(w1_ptr + cols1, mask=m1, other=0.0).to(tl.float32)
    tl.store(o1_ptr + row * o1_row_stride + cols1, x1 * inv1 * w1, mask=m1)

    # ---- latent 2: rmsnorm over d2 ----
    cols2 = tl.arange(0, BLOCK_D2)
    m2 = cols2 < d2
    x2 = tl.load(x2_ptr + row * x2_row_stride + cols2, mask=m2, other=0.0).to(tl.float32)
    inv2 = tl.rsqrt(tl.sum(x2 * x2, axis=0) / d2 + eps)
    w2 = tl.load(w2_ptr + cols2, mask=m2, other=0.0).to(tl.float32)
    tl.store(o2_ptr + row * o2_row_stride + cols2, x2 * inv2 * w2, mask=m2)


def dual_rmsnorm_triton(x1, w1, x2, w2, eps: float = 1e-6):
    assert x1.shape[0] == x2.shape[0], "x1 and x2 must share the token dim T"
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    T, d1 = x1.shape
    d2 = x2.shape[1]
    o1 = torch.empty_like(x1)
    o2 = torch.empty_like(x2)
    block_d1 = triton.next_power_of_2(d1)
    block_d2 = triton.next_power_of_2(d2)
    num_warps = max(1, min(16, max(block_d1, block_d2) // 256))
    dual_rmsnorm_kernel[(T,)](
        x1,
        w1,
        o1,
        x2,
        w2,
        o2,
        x1.stride(0),
        o1.stride(0),
        x2.stride(0),
        o2.stride(0),
        d1,
        d2,
        eps,
        BLOCK_D1=block_d1,
        BLOCK_D2=block_d2,
        num_warps=num_warps,
    )
    return o1, o2


register("dual_rmsnorm", Backend.TRITON)(dual_rmsnorm_triton)

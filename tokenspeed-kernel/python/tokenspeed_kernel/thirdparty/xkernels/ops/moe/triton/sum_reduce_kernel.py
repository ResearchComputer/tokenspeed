# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Weighted top-k reduction (issue #5) for AMD MI300A (gfx942).

The torch.compile combine path is unstable at high rank on AMD; this is the
dedicated kernel. One program per ``(m, H-tile)`` accumulates the ``top_k``
partials (optionally times the routing weight) in fp32, then applies the
routed-scaling factor.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["moe_sum_reduce_triton", "moe_sum_reduce_kernel"]


@triton.jit
def moe_sum_reduce_kernel(
    y_ptr,
    w_ptr,
    out_ptr,
    stride_ym,
    stride_yk,
    stride_yh,
    stride_wm,
    stride_wk,
    stride_om,
    stride_oh,
    H,
    scaling,
    TOP_K: tl.constexpr,
    HAS_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    m = tl.program_id(axis=0)
    hb = tl.program_id(axis=1)
    cols = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in range(TOP_K):
        yk = tl.load(
            y_ptr + m * stride_ym + k * stride_yk + cols * stride_yh,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_W:
            wk = tl.load(w_ptr + m * stride_wm + k * stride_wk).to(tl.float32)
            yk = yk * wk
        acc += yk

    acc = acc * scaling
    tl.store(out_ptr + m * stride_om + cols * stride_oh, acc, mask=mask)


def moe_sum_reduce_triton(y, w=None, routed_scaling_factor: float = 1.0):
    y = y.contiguous()
    M, top_k, H = y.shape
    out = torch.empty((M, H), dtype=y.dtype, device=y.device)

    has_w = w is not None
    if has_w:
        w = w.contiguous()
        sw_m, sw_k, w_ptr = w.stride(0), w.stride(1), w
    else:
        sw_m, sw_k, w_ptr = 0, 0, y  # dummy ptr; not read when HAS_W is False

    block_h = min(triton.next_power_of_2(H), 1024)
    grid = (M, triton.cdiv(H, block_h))
    moe_sum_reduce_kernel[grid](
        y,
        w_ptr,
        out,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        sw_m,
        sw_k,
        out.stride(0),
        out.stride(1),
        H,
        float(routed_scaling_factor),
        TOP_K=top_k,
        HAS_W=has_w,
        BLOCK_H=block_h,
    )
    return out


register("moe_sum_reduce", Backend.TRITON)(moe_sum_reduce_triton)

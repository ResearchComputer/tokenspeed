# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Fused residual-add + RMSNorm epilogue (issue #12) for AMD MI300A (gfx942).

The trailing ``residual += x; x = rmsnorm(residual)`` after the MoE all-reduce, in
one pass: one program per token row adds the residual, writes back the new
residual, and stores the normalized output (fp32 reduction). This is the compute
half of the planned ``allreduce_residual_rmsnorm`` fusion — composed after the
hierarchical all-gather; fusing it *into* the all-gather epilogue (overlapping
the xGMI gather with the norm) is a further refinement.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["add_rmsnorm_triton", "add_rmsnorm_kernel"]


@triton.jit
def add_rmsnorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    new_res_ptr,
    row_stride,
    H,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + res
    tl.store(new_res_ptr + row * row_stride + cols, h, mask=mask)  # new residual

    inv = tl.rsqrt(tl.sum(h * h, axis=0) / H + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * row_stride + cols, h * inv * w, mask=mask)


def add_rmsnorm_triton(x, residual, weight, eps: float = 1e-6):
    x = x.contiguous()
    residual = residual.contiguous()
    *lead, H = x.shape
    x2d = x.view(-1, H)
    res2d = residual.view(-1, H)
    out = torch.empty_like(x2d)
    new_res = torch.empty_like(x2d)
    n_rows = x2d.shape[0]
    block_h = triton.next_power_of_2(H)
    num_warps = max(1, min(16, block_h // 256))
    add_rmsnorm_kernel[(n_rows,)](
        x2d, res2d, weight, out, new_res, x2d.stride(0), H, eps,
        BLOCK_H=block_h, num_warps=num_warps,
    )
    return out.view(*lead, H), new_res.view(*lead, H)

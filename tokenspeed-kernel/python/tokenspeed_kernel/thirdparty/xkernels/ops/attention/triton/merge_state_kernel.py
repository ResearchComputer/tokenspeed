# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Online-softmax merge of attention partials (issue #3) for AMD MI300A (gfx942).

Portable Triton replacement for the CUDA-only ``merge_state`` (which crashed on
AMD). One program per ``(T, H)`` row combines the two partials in the log2
domain (``exp2`` / ``log2``, the flashinfer convention): with ``s = log2(e)``,

    la, lb = lse_a*s, lse_b*s ; m = max(la, lb)
    wa, wb = 2**(la-m), 2**(lb-m) ; denom = wa + wb
    out = (out_a*wa + out_b*wb) / denom ; lse = (m + log2(denom)) / s

which is algebraically identical to the natural-log reference (the ``s`` cancels)
but maps onto the hardware ``exp2`` unit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["mha_merge_state_triton", "merge_state_kernel"]

_LOG2E = 1.4426950408889634  # log2(e)


@triton.jit
def merge_state_kernel(
    out_a_ptr,
    lse_a_ptr,
    out_b_ptr,
    lse_b_ptr,
    out_ptr,
    lse_ptr,
    D,
    LOG2E,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(axis=0)

    la = tl.load(lse_a_ptr + row).to(tl.float32) * LOG2E
    lb = tl.load(lse_b_ptr + row).to(tl.float32) * LOG2E
    m = tl.maximum(la, lb)
    wa = tl.exp2(la - m)
    wb = tl.exp2(lb - m)
    denom = wa + wb

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    oa = tl.load(out_a_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    ob = tl.load(out_b_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    out = (oa * wa + ob * wb) / denom
    tl.store(out_ptr + row * D + cols, out, mask=mask)

    # lse back in the natural-log basis: (m + log2(denom)) / s.
    tl.store(lse_ptr + row, (m + tl.log2(denom)) / LOG2E)


def mha_merge_state_triton(out_a, lse_a, out_b, lse_b):
    out_a = out_a.contiguous()
    out_b = out_b.contiguous()
    lse_a = lse_a.contiguous()
    lse_b = lse_b.contiguous()
    D = out_a.shape[-1]
    n_rows = lse_a.numel()  # T * H
    out = torch.empty_like(out_a)
    lse = torch.empty(out_a.shape[:-1], dtype=torch.float32, device=out_a.device)
    block_d = triton.next_power_of_2(D)
    merge_state_kernel[(n_rows,)](
        out_a, lse_a, out_b, lse_b, out, lse, D, _LOG2E, BLOCK_D=block_d
    )
    return out, lse


register("mha_merge_state", Backend.TRITON)(mha_merge_state_triton)

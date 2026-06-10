"""Triton FFN backend: torch matmuls for the projections, a fused Triton
kernel for the elementwise SwiGLU activation `silu(g) * u`."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register
from .._activation import SwigluAct


@triton.jit
def _swiglu_kernel(g_ptr, u_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(g_ptr + offs, mask=mask)
    u = tl.load(u_ptr + offs, mask=mask)
    out = (g * tl.sigmoid(g.to(tl.float32)).to(g.dtype)) * u
    tl.store(out_ptr + offs, out, mask=mask)


def _swiglu_triton(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    g = g.contiguous()
    u = u.contiguous()
    out = torch.empty_like(g)
    n = g.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_kernel[grid](g, u, out, n, BLOCK=1024)
    return out


def ffn_triton(x, w_gate, w_up, w_down):
    g = x @ w_gate
    u = x @ w_up
    h = SwigluAct.apply(g, u, _swiglu_triton)
    return h @ w_down


register("ffn", Backend.TRITON)(ffn_triton)

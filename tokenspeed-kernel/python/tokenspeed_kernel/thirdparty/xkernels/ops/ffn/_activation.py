"""Autograd wrapper for the fused SwiGLU activation `silu(g) * u`.

Forward runs a backend-provided elementwise kernel; backward is computed in
torch so every backend (triton/cuda) is differentiable without a custom
backward kernel. d/dg[silu(g)] = sigmoid(g) * (1 + g * (1 - sigmoid(g))).
"""
from __future__ import annotations

from collections.abc import Callable

import torch


class SwigluAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor, kernel: Callable):
        ctx.save_for_backward(g, u)
        return kernel(g, u)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        g, u = ctx.saved_tensors
        sig = torch.sigmoid(g)
        silu = g * sig
        dsilu_dg = sig * (1 + g * (1 - sig))
        grad_g = grad_out * u * dsilu_dg
        grad_u = grad_out * silu
        return grad_g, grad_u, None  # None for the non-tensor `kernel` arg

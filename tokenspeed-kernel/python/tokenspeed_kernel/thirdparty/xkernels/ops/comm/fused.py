# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Fused residual-add + RMSNorm epilogue + its composition with the all-reduce
(issue #12). Pure-torch reference oracle, a backend-selecting public op, and the
``hierarchical_all_reduce`` + fused-epilogue wrapper.
"""

from __future__ import annotations

import torch

from .hierarchical import hierarchical_all_reduce

__all__ = [
    "add_rmsnorm_ref",
    "residual_rmsnorm",
    "hierarchical_all_reduce_residual_rmsnorm",
]


def add_rmsnorm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: ``new_residual = x + residual; out = rmsnorm(new_residual, weight)``.

    Returns ``(out, new_residual)`` both in ``x.dtype`` (fp32 reduction).
    """
    hf = x.float() + residual.float()
    new_residual = hf.to(x.dtype)
    inv = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)
    out = ((hf * inv).to(weight.dtype) * weight).to(x.dtype)
    return out, new_residual


def residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_triton: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm. Uses the Triton kernel when available on GPU,
    else the torch reference. ``use_triton`` forces the choice."""
    if use_triton is None:
        use_triton = x.is_cuda
    if use_triton:
        try:
            from .triton.add_rmsnorm_kernel import add_rmsnorm_triton

            return add_rmsnorm_triton(x, residual, weight, eps)
        except Exception:
            pass
    return add_rmsnorm_ref(x, residual, weight, eps)


def hierarchical_all_reduce_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    intra_group,
    cross_group,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical all-reduce of ``x`` followed by the fused residual+RMSNorm epilogue.

    Returns ``(normed_output, new_residual)``.
    """
    reduced = hierarchical_all_reduce(x, intra_group, cross_group)
    return residual_rmsnorm(reduced, residual, weight, eps)

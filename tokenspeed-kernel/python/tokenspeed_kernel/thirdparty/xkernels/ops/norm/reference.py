# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the fused dual RMSNorm (issue #2) — numerical oracle
and default (CPU / no-Triton) backend.

MLA attention normalizes two independent latent projections per layer (``q_a``,
``kv_a``); this is the slow two-launch baseline the single-launch Triton kernel
is checked against.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["rmsnorm", "dual_rmsnorm_ref"]


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Single RMSNorm: ``x * rsqrt(mean(x^2, -1) + eps) * w`` (fp32 reduction)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(w.dtype) * w


def dual_rmsnorm_ref(
    x1: torch.Tensor,
    w1: torch.Tensor,
    x2: torch.Tensor,
    w2: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two independent RMSNorms: ``(rmsnorm(x1, w1), rmsnorm(x2, w2))``.

    Args:
        x1: ``[T, d1]`` activations, x2: ``[T, d2]`` activations (bf16 or fp32).
        w1: ``[d1]`` weight, w2: ``[d2]`` weight.
        eps: variance epsilon.
    """
    return rmsnorm(x1, w1, eps), rmsnorm(x2, w2, eps)


register("dual_rmsnorm", Backend.REFERENCE)(dual_rmsnorm_ref)

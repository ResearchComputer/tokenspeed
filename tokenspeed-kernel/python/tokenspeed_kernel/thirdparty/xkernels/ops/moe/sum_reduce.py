# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Weighted reduction of top-k expert outputs (issue #5).

Final step of the fused MoE: reduce each token's ``top_k`` partial down-proj
outputs into one hidden vector (routing weight + optional routed-scaling
factor). Public op + pure-torch reference (REFERENCE backend).
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch, register

__all__ = ["moe_sum_reduce", "moe_sum_reduce_ref"]


def moe_sum_reduce_ref(
    y: torch.Tensor,
    w: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Reference: ``out[m] = routed_scaling_factor * sum_k w[m,k] * y[m,k,:]``.

    Args:
        y: ``[M, top_k, H]`` per-expert partial outputs (bf16 or fp32).
        w: ``[M, top_k]`` routing weights, or None (plain sum — the in-stack
            case where the weight is already folded into the down-proj GEMM).
        routed_scaling_factor: scalar applied to the reduced output.

    Returns:
        ``[M, H]`` output in ``y.dtype``.
    """
    yf = y.float()
    if w is not None:
        yf = yf * w[..., None].float()
    return (yf.sum(dim=1) * routed_scaling_factor).to(y.dtype)


register("moe_sum_reduce", Backend.REFERENCE)(moe_sum_reduce_ref)


def moe_sum_reduce(
    y: torch.Tensor,
    w: torch.Tensor | None = None,
    *,
    routed_scaling_factor: float = 1.0,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Weighted reduction of top-k expert outputs. See ``moe_sum_reduce_ref``."""
    return dispatch(
        "moe_sum_reduce",
        y,
        w,
        routed_scaling_factor=routed_scaling_factor,
        backend=backend,
    )

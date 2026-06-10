# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the INT4 W4A16 fused-MoE GEMM — the numerical oracle
and the default (CPU / no-Triton) backend for ``moe_int4_w4a16``.

Intentionally written for clarity, not speed: explicit unpack -> dequant ->
grouped GEMM (acceptance, issue #1: match dequant-then-matmul within
``atol/rtol ~ 2e-2`` bf16).
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register
from .w4a16 import dequant_w4a16

__all__ = ["moe_w4a16_ref"]


def moe_w4a16_ref(
    A: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    group_size: int = 32,
    mul_routed_weight: bool = True,
) -> torch.Tensor:
    """Reference grouped MoE GEMM: ``out[m] = sum_j w[m,j] * (A[m] @ W[e]^T)``.

    Args:
        A: ``[M, K]`` activations (bf16 or fp32).
        packed: ``[E, N, K // 8]`` int32 packed weights.
        scale: ``[E, N, K // group_size]`` group scales.
        topk_ids: ``[M, top_k]`` int32 expert ids.
        topk_w: ``[M, top_k]`` fp32 routing weights.
        group_size: quant group size.
        mul_routed_weight: fold routing weights into the sum (matches down GEMM).

    Returns:
        ``[M, N]`` output in ``A.dtype`` (fp32 accumulation).
    """
    W = dequant_w4a16(packed, scale, group_size)  # [E, N, K] bf16
    M, topk = topk_ids.shape
    out = torch.zeros(M, W.shape[1], dtype=torch.float32, device=A.device)
    for m in range(M):
        for j in range(topk):
            e = int(topk_ids[m, j])
            contrib = A[m].float() @ W[e].float().T
            if mul_routed_weight:
                contrib = topk_w[m, j].float() * contrib
            out[m] += contrib
    return out.to(A.dtype)


register("moe_int4_w4a16", Backend.REFERENCE)(moe_w4a16_ref)

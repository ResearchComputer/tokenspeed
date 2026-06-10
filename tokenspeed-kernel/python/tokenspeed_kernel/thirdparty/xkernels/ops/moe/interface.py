# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public ``fused_moe_int4_w4a16`` op: dispatches to a registered backend.

The op signature is backend-agnostic ``[M, N]`` in / out; the Triton backend
hides the ``moe_align_block_size`` dispatch build, the token-indexed
``[M*top_k, N]`` scratch buffer, and the ``view(M, top_k, N).sum(1)`` reduce.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def fused_moe_int4_w4a16(
    A: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    *,
    group_size: int = 32,
    mul_routed_weight: bool = True,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """INT4 W4A16 grouped fused-MoE GEMM ``out[m] = sum_j w[m,j] * (A[m] @ W[e]^T)``.

    Args:
        A: ``[M, K]`` activations (bf16, or fp32 under the Triton interpreter).
        packed: ``[E, N, K // 8]`` int32 ``uint4b8`` packed weights.
        scale: ``[E, N, K // group_size]`` bf16 symmetric group scales.
        topk_ids: ``[M, top_k]`` int32 expert ids.
        topk_w: ``[M, top_k]`` fp32 routing weights.
        group_size: quant group size along K (default 32).
        mul_routed_weight: fold routing weights into the output (down GEMM).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``[M, N]`` output in ``A.dtype``.
    """
    return dispatch(
        "moe_int4_w4a16",
        A,
        packed,
        scale,
        topk_ids,
        topk_w,
        group_size=group_size,
        mul_routed_weight=mul_routed_weight,
        backend=backend,
    )

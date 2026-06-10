# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""moe_align_block_size (issue #4): sort/pad routed tokens into per-expert blocks.

The fused-MoE block GEMM needs tokens grouped by expert with each expert's run
padded to a multiple of ``block_size``, so every block maps to exactly one
expert. This exposes the operation as a dispatched op with the correctness-first
torch reference (``moe_align_block_size_ref``) as the REFERENCE backend.

**Status:** reference backend only. A Triton perf kernel (vLLM/SGLang-style:
per-expert histogram + padded prefix-sum + scatter) is the tracked follow-up for
issue #4 — it relies on device atomics whose behavior and speedup must be
validated on real gfx942 hardware, so it is intentionally not landed unverified.
Until then ``backend="auto"`` resolves to the reference everywhere.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch, register
from .w4a16 import moe_align_block_size_ref

__all__ = ["moe_align_block_size"]

# The reference (in w4a16.py, shared with the INT4 GEMM launcher) is the oracle
# and the only backend for now.
register("moe_align_block_size", Backend.REFERENCE)(moe_align_block_size_ref)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort/pad routed token-slots into per-expert blocks.

    Args:
        topk_ids: ``[M, top_k]`` int32 expert id per token-slot.
        block_size: GEMM block size each expert run is padded up to.
        num_experts: number of experts.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(sorted_token_ids [max_pad], expert_ids [max_blocks], num_tokens_post_padded [1])``
        where ``max_pad = M*top_k + (num_experts+1)*(block_size-1)`` and unused
        slots hold ``pad_id = M*top_k``.
    """
    return dispatch("moe_align_block_size", topk_ids, block_size, num_experts, backend=backend)

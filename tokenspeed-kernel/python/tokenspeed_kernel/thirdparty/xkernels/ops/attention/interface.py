# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public ``mha_merge_state`` op: dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def mha_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention partials by their log-sum-exp (online softmax).

    For chunked-prefill / split-KV MLA: combine per-KV-chunk partial outputs and
    LSEs into a single output + merged LSE.

    Args:
        out_a, out_b: ``[T, H, D]`` partial outputs (bf16 or fp32).
        lse_a, lse_b: ``[T, H]`` fp32 log-sum-exp (natural-log basis).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(out [T, H, D] in out_a.dtype, lse [T, H] fp32)``.
    """
    return dispatch("mha_merge_state", out_a, lse_a, out_b, lse_b, backend=backend)

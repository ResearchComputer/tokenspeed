# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public ``dual_rmsnorm`` op: dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def dual_rmsnorm(
    x1: torch.Tensor,
    w1: torch.Tensor,
    x2: torch.Tensor,
    w2: torch.Tensor,
    *,
    eps: float = 1e-6,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused parallel dual RMSNorm of two MLA latents (``q_a`` / ``kv_a``).

    Computes ``(rmsnorm(x1, w1), rmsnorm(x2, w2))`` in a single launch on the
    Triton backend (rows are independent; the win is one kernel/one pass vs two
    sequential RMSNorm launches).

    Args:
        x1: ``[T, d1]`` activations, x2: ``[T, d2]`` activations (must share T).
        w1: ``[d1]`` per-feature weight, w2: ``[d2]`` per-feature weight.
        eps: variance epsilon.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(out1 [T, d1], out2 [T, d2])`` in the input dtypes.
    """
    return dispatch("dual_rmsnorm", x1, w1, x2, w2, eps=eps, backend=backend)

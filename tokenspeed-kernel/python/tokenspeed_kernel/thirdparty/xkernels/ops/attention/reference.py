# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for ``mha_merge_state`` (issue #3) — numerical oracle and
default (CPU / no-Triton) backend.

Merges two attention partial outputs by their log-sum-exp (natural-log basis), a
numerically-stable online-softmax combine. Parity target: flashinfer
``MergeStateKernel``.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["merge_state_ref"]


def merge_state_ref(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax merge of two attention partials.

    Args:
        out_a, out_b: ``[T, H, D]`` partial outputs (bf16 or fp32).
        lse_a, lse_b: ``[T, H]`` log-sum-exp of each partial (fp32, natural-log).

    Returns:
        ``(out [T, H, D] in out_a.dtype, lse [T, H] fp32)``.
    """
    la, lb = lse_a.float(), lse_b.float()
    m = torch.maximum(la, lb)
    wa, wb = (la - m).exp(), (lb - m).exp()
    denom = wa + wb
    out = (out_a.float() * wa[..., None] + out_b.float() * wb[..., None]) / denom[..., None]
    return out.to(out_a.dtype), (m + denom.log())


register("mha_merge_state", Backend.REFERENCE)(merge_state_ref)

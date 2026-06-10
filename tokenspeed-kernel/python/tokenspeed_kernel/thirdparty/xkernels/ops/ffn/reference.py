"""Pure-torch SwiGLU FFN — the correctness oracle and default backend."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._backends import Backend
from ..._dispatch import register


def ffn_reference(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Compute (silu(x @ w_gate) * (x @ w_up)) @ w_down."""
    return (F.silu(x @ w_gate) * (x @ w_up)) @ w_down


register("ffn", Backend.REFERENCE)(ffn_reference)

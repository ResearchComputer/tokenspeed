"""Public `fused_ffn` op: normalizes leading dims, then dispatches."""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def fused_ffn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """SwiGLU FFN: (silu(x @ w_gate) * (x @ w_up)) @ w_down.

    `x` may have any number of leading dims; only the last (feature) dim must
    match `w_gate`/`w_up`. `backend` is "auto" or a `Backend` / its string value.
    """
    *lead, d_model = x.shape
    x2d = x.reshape(-1, d_model)
    out = dispatch("ffn", x2d, w_gate, w_up, w_down, backend=backend)
    return out.reshape(*lead, out.shape[-1])

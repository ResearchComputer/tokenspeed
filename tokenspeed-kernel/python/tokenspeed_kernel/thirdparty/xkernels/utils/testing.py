"""Correctness-test helpers with per-dtype tolerance presets."""
from __future__ import annotations

import torch

_TOL: dict[torch.dtype, dict[str, float]] = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 1e-3, "atol": 1e-3},
    torch.bfloat16: {"rtol": 1.6e-2, "atol": 1e-2},
}


def tolerance(dtype: torch.dtype) -> dict[str, float]:
    """Return {'rtol', 'atol'} for a dtype (defaults to float32 tolerances)."""
    return _TOL.get(dtype, _TOL[torch.float32])


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """torch.testing.assert_close with tolerances keyed off `expected.dtype`."""
    tol = tolerance(expected.dtype)
    torch.testing.assert_close(actual, expected, rtol=tol["rtol"], atol=tol["atol"])

"""CUDA/HIP FFN backend. Registers only if the compiled extension imports.

The extension is built by `setup.py` (one per kernel type) when a CUDA/ROCm
toolkit is present. On NVIDIA it registers as Backend.CUDA; on AMD as
Backend.HIP. If the extension is absent, importing this module raises and the
backend is simply not registered.
"""
from __future__ import annotations

import torch

from ...._backends import Backend, detect_vendor
from ...._dispatch import register
from .._activation import SwigluAct
from . import _cuda  # compiled extension; ImportError if not built


def _swiglu_cuda(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return _cuda.swiglu_act(g, u)


def ffn_cuda(x, w_gate, w_up, w_down):
    g = x @ w_gate
    u = x @ w_up
    h = SwigluAct.apply(g, u, _swiglu_cuda)
    return h @ w_down


# Register under whichever vendor this torch build targets (default to CUDA).
_backend = Backend.HIP if detect_vendor() == "amd" else Backend.CUDA
register("ffn", _backend)(ffn_cuda)

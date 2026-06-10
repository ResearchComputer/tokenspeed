"""Backend identifiers and hardware/vendor detection."""
from __future__ import annotations

import enum

import torch


class Backend(enum.Enum):
    REFERENCE = "reference"
    TRITON = "triton"
    CUDA = "cuda"
    HIP = "hip"


def detect_vendor() -> str:
    """Return the GPU vendor of the current torch build: 'nvidia', 'amd', or 'none'."""
    if getattr(torch.version, "hip", None):
        return "amd"
    if getattr(torch.version, "cuda", None) and torch.cuda.is_available():
        return "nvidia"
    return "none"


def device_of(tensor: torch.Tensor) -> str:
    """Return 'cpu' or 'cuda' for a tensor's device type (ROCm reports 'cuda')."""
    return tensor.device.type

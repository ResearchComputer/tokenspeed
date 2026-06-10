# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Flat all-reduce — the correctness oracle for the hierarchical schedule (issue #12).

A plain ``dist.all_reduce`` (SUM) over the full group. The hierarchical all-reduce
must match this bit-for-bit up to floating-point accumulation order.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["flat_all_reduce"]


def flat_all_reduce(x: torch.Tensor, group=None) -> torch.Tensor:
    """Sum-reduce ``x`` across all ranks in ``group``; result replicated to all.

    Returns a new tensor (does not modify ``x``).
    """
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
    return y

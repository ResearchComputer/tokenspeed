"""Pure-torch reference comm ops (test oracle / default backend).

TODO: add real backends (e.g. NCCL/RCCL custom all-reduce, fused reduce-scatter).
The reference models the single-process semantics: a sum over the shard list.
"""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register


def all_reduce_reference(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """Sum-allreduce semantics: every shard becomes the elementwise sum."""
    total = torch.stack(shards, dim=0).sum(dim=0)
    return [total.clone() for _ in shards]


register("comm", Backend.REFERENCE)(all_reduce_reference)

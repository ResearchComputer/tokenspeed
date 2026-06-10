"""Public `all_reduce` op (stub — dispatches to reference until kernels land)."""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def all_reduce(
    shards: list[torch.Tensor],
    *,
    backend: Backend | str = "auto",
) -> list[torch.Tensor]:
    """Sum-allreduce over a list of shards. See `all_reduce_reference`."""
    return dispatch("comm", shards, backend=backend)

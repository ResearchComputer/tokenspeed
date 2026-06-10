# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Topology-aware (2-level) hierarchical all-reduce (issue #12).

Replaces a flat 8-way all-reduce that crosses the slow CXI fabric with:

    intra-node reduce-scatter (xGMI, ranks_per_node) -> each rank holds 1/rpp
    cross-node all-reduce of that 1/rpp partial (CXI, one hop, small payload)
    intra-node all-gather (xGMI) -> full reduced tensor on every rank

For the MoE decode reduce ([bs, 7168] bf16, bs tiny) this keeps the cross-node leg
to a single hop over a 1/ranks_per_node-size payload, instead of a flat all-reduce
whose every step crosses CXI. Numerically equal to ``flat_all_reduce`` up to
bf16 accumulation order.

The tensor is flattened before scatter, so it works for any shape whose total
element count is divisible by ``ranks_per_node`` (``7168 % 4 == 0``, so any ``bs``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

__all__ = ["hierarchical_all_reduce"]


def hierarchical_all_reduce(x: torch.Tensor, intra_group, cross_group) -> torch.Tensor:
    """Sum-reduce ``x`` across all ranks via the 2-level schedule.

    Args:
        x: tensor to reduce (any shape; ``x.numel()`` must be divisible by the
            intra-node group size). Result is replicated to every rank.
        intra_group: the intra-node process group (xGMI peers).
        cross_group: the cross-node process group (same local rank across nodes).

    Returns:
        A new tensor, same shape/dtype as ``x``, holding the global sum.
    """
    rpp = dist.get_world_size(intra_group)
    shape, dtype, device = x.shape, x.dtype, x.device
    flat = x.contiguous().view(-1)
    n = flat.numel()
    if n % rpp != 0:
        raise ValueError(f"numel {n} not divisible by intra-node group size {rpp}")

    # 1) intra-node reduce-scatter (xGMI): rank lr ends with the node-sum of chunk lr.
    rs = torch.empty(n // rpp, dtype=dtype, device=device)
    dist.reduce_scatter_tensor(rs, flat, op=dist.ReduceOp.SUM, group=intra_group)

    # 2) cross-node all-reduce (CXI, 1/rpp payload): chunk lr summed across nodes.
    dist.all_reduce(rs, op=dist.ReduceOp.SUM, group=cross_group)

    # 3) intra-node all-gather (xGMI): reassemble the full global-sum tensor.
    out_flat = torch.empty(n, dtype=dtype, device=device)
    dist.all_gather_into_tensor(out_flat, rs, group=intra_group)
    return out_flat.view(shape)

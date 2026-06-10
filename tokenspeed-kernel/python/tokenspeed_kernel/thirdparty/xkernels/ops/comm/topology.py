# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Process-group construction for topology-aware hierarchical collectives (issue #12).

Ranks are assumed contiguously placed: node ``n`` owns ranks
``[n*ranks_per_node, (n+1)*ranks_per_node)`` (the default SLURM block layout). Two
group families are built so a flat all-reduce can be split into a fast intra-node
leg (xGMI / Infinity Fabric) and a slow cross-node leg (Slingshot/CXI):

* **intra-node group** — the ``ranks_per_node`` GPUs sharing a node (xGMI).
* **cross-node group** — ranks with the *same* local rank across nodes (CXI); after
  an intra-node reduce-scatter, rank ``lr`` holds chunk ``lr``, and the partials
  for chunk ``lr`` live on exactly these ranks.

``build_topology_groups`` is collective: **every** rank must call it (all groups
are created on all ranks, as ``torch.distributed`` requires).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist

__all__ = ["TopologyInfo", "build_topology_groups"]


@dataclass(frozen=True)
class TopologyInfo:
    rank: int
    world_size: int
    ranks_per_node: int
    num_nodes: int
    local_rank: int  # position within the node, in [0, ranks_per_node)
    node: int  # node index, in [0, num_nodes)


def build_topology_groups(ranks_per_node: int = 4, world_size: int | None = None):
    """Build the intra-node and cross-node process groups for this rank.

    Args:
        ranks_per_node: GPUs per node (4 on MI300A). Must divide the world size.
        world_size: override (defaults to ``dist.get_world_size()``).

    Returns:
        ``(intra_group, cross_group, info)`` where ``info`` is a ``TopologyInfo``.
    """
    if world_size is None:
        world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % ranks_per_node != 0:
        raise ValueError(
            f"world_size {world_size} not divisible by ranks_per_node {ranks_per_node}"
        )
    num_nodes = world_size // ranks_per_node
    local_rank = rank % ranks_per_node
    node = rank // ranks_per_node

    # All intra-node groups must be created on every rank (collective new_group).
    intra_group = None
    for n in range(num_nodes):
        ranks = list(range(n * ranks_per_node, (n + 1) * ranks_per_node))
        g = dist.new_group(ranks)
        if n == node:
            intra_group = g

    # Cross-node groups: same local rank across nodes.
    cross_group = None
    for lr in range(ranks_per_node):
        ranks = list(range(lr, world_size, ranks_per_node))
        g = dist.new_group(ranks)
        if lr == local_rank:
            cross_group = g

    info = TopologyInfo(
        rank=rank,
        world_size=world_size,
        ranks_per_node=ranks_per_node,
        num_nodes=num_nodes,
        local_rank=local_rank,
        node=node,
    )
    return intra_group, cross_group, info

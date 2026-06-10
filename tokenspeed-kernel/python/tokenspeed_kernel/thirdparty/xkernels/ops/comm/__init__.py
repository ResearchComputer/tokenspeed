"""Communication kernels.

Topology-aware hierarchical all-reduce for DP-attention MoE serving on 2-node
MI300A (issue #12): split a flat cross-fabric all-reduce into a fast intra-node
leg (xGMI) + a small cross-node leg (CXI), plus the optional fused residual-add +
RMSNorm epilogue. These are distributed collectives (``torch.distributed`` /
RCCL), not single-GPU dispatched kernels, so they are exposed as functions taking
process groups rather than through the ``Backend`` registry.
"""
from .fused import (
    add_rmsnorm_ref,
    hierarchical_all_reduce_residual_rmsnorm,
    residual_rmsnorm,
)
from .hierarchical import hierarchical_all_reduce
from .reference import flat_all_reduce
from .topology import TopologyInfo, build_topology_groups

__all__ = [
    "build_topology_groups",
    "TopologyInfo",
    "flat_all_reduce",
    "hierarchical_all_reduce",
    "residual_rmsnorm",
    "add_rmsnorm_ref",
    "hierarchical_all_reduce_residual_rmsnorm",
]

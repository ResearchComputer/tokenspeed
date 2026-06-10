"""xkernels — customized compute kernels across vendors and kernel types."""

from .ops.attention import mha_merge_state
from .ops.comm import (
    build_topology_groups,
    flat_all_reduce,
    hierarchical_all_reduce,
    residual_rmsnorm,
)
from .ops.ffn import fused_ffn
from .ops.moe import fused_moe_int4_w4a16, moe_align_block_size, moe_sum_reduce
from .ops.norm import dual_rmsnorm

__version__ = "0.0.1"
__all__ = [
    "fused_ffn",
    "fused_moe_int4_w4a16",
    "moe_align_block_size",
    "moe_sum_reduce",
    "mha_merge_state",
    "dual_rmsnorm",
    "build_topology_groups",
    "flat_all_reduce",
    "hierarchical_all_reduce",
    "residual_rmsnorm",
    "__version__",
]

"""Mixture-of-Experts kernels.

Ships the INT4 W4A16 grouped fused-MoE GEMM (issue #1), the weighted top-k
reduction (issue #5), and the block-align dispatch builder (issue #4). Each
public op dispatches across a pure-torch reference (default on CPU / no Triton)
and, where available, an autotuned Triton backend.
"""
from .align import moe_align_block_size
from .interface import fused_moe_int4_w4a16
from .sum_reduce import moe_sum_reduce
from .w4a16 import dequant_w4a16, make_w4a16_weights, moe_align_block_size_ref

# Import Triton backends for their registration side effects. Optional — guard
# each so the package imports without Triton installed. Routed through the
# vendored triton-compat shim so the kernels bind ``tokenspeed_triton`` (not
# stock ``triton``) inside tokenspeed; see ``thirdparty/xkernels/_triton_compat.py``.
# This matters for ``moe_int4_kernel`` in particular: its ``tl.dot`` asserts that
# both operands share the same dtype *object*, which fails across packages.
try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import moe_int4_kernel  # noqa: F401
except Exception:
    pass

try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import sum_reduce_kernel  # noqa: F401
except Exception:
    pass

__all__ = [
    "fused_moe_int4_w4a16",
    "moe_align_block_size",
    "moe_sum_reduce",
    "dequant_w4a16",
    "make_w4a16_weights",
    "moe_align_block_size_ref",
]

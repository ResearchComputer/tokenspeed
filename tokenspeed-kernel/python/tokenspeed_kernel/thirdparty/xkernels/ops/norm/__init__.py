"""Normalization kernels.

Ships the fused parallel dual RMSNorm (MLA ``q_a`` / ``kv_a`` latents, issue #2):
two independent RMSNorms over differently-sized feature dims in a single launch.
"""
from .interface import dual_rmsnorm

# Import the Triton backend for its registration side effect. Optional.
# The import is routed through the vendored triton-compat shim so the kernel
# binds ``tokenspeed_triton`` (not stock ``triton``) when running inside
# tokenspeed; see ``thirdparty/xkernels/_triton_compat.py``.
try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import dual_rmsnorm_kernel  # noqa: F401
except Exception:
    pass

__all__ = ["dual_rmsnorm"]

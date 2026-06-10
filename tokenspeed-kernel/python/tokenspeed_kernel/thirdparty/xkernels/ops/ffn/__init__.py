"""Fused FFN kernels."""
from .interface import fused_ffn

# Import backend modules for their registration side effects. Triton/CUDA are
# optional — guard so the package imports on any machine. The Triton import is
# routed through the vendored triton-compat shim so the kernel binds
# ``tokenspeed_triton`` (not stock ``triton``) inside tokenspeed; see
# ``thirdparty/xkernels/_triton_compat.py``.
try:  # pragma: no cover - hardware dependent
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import ffn_kernel  # noqa: F401
except Exception:
    pass

try:  # pragma: no cover - requires compiled extension
    from . import cuda  # noqa: F401
except Exception:
    pass

__all__ = ["fused_ffn"]

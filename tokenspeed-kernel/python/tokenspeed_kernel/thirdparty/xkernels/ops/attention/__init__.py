"""Attention kernels.

Ships ``mha_merge_state`` (issue #3): the numerically-stable online-softmax
merge of two attention partials by their log-sum-exp, used by chunked-prefill /
split-KV MLA on AMD MI300A.
"""
from .interface import mha_merge_state

# Import the Triton backend for its registration side effect. Optional. Routed
# through the vendored triton-compat shim so the kernel binds ``tokenspeed_triton``
# (not stock ``triton``) inside tokenspeed; see
# ``thirdparty/xkernels/_triton_compat.py``.
try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import merge_state_kernel  # noqa: F401
except Exception:
    pass

__all__ = ["mha_merge_state"]

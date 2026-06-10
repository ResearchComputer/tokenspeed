# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton-package compatibility shim for the vendored ``xkernels`` copy.

Upstream ``xkernels`` Triton backends are written against the *stock* ``triton``
distribution (``import triton`` / ``import triton.language as tl``). Inside
tokenspeed, however, the runtime kernels are built against the vendored
``tokenspeed_triton`` package (imported via ``tokenspeed_kernel._triton``), and
the two are **separate Python packages**: ``triton.bfloat16`` is *not*
``tokenspeed_triton.bfloat16``, and ``tl.dot`` asserts that both operands share
the *same* dtype object, so a dtype coming from the stock package is rejected at
runtime ("Unsupported rhs dtype"). See ``tokenspeed_kernel/_triton.py``.

To keep the vendored source byte-identical to upstream (only the package
``__init__`` files route their Triton-backend imports through here), this module
exposes :func:`triton_import_ctx`: a context manager that makes ``triton[.x.y]``
resolve to ``tokenspeed_triton[.x.y]`` while a Triton backend module is being
imported. When tokenspeed's redirect is unavailable (e.g. the package is used
standalone with stock Triton), it degrades to a no-op so the kernels still
import and compile against whatever ``triton`` is installed.
"""

from __future__ import annotations

import contextlib

__all__ = ["triton_import_ctx"]


def triton_import_ctx():
    """Return a context manager that aliases ``triton`` -> ``tokenspeed_triton``.

    Use around the import of any vendored Triton backend module so its
    module-level ``import triton`` / ``import triton.language as tl`` bind the
    ``tokenspeed_triton`` objects that the rest of the tokenspeed kernel stack
    uses. No-op when tokenspeed's redirect helper is not importable.
    """
    try:
        from tokenspeed_kernel._triton import (
            redirect_triton_to_tokenspeed_triton,
        )
    except Exception:
        return contextlib.nullcontext()
    return redirect_triton_to_tokenspeed_triton()

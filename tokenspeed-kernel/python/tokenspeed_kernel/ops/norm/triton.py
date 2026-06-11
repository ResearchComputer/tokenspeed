# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fused dual RMSNorm (norm.dual_rmsnorm) backed by ``xkernels``.

This normalizes the two MLA latents (``q_a`` / ``kv_a``) in a single Triton
launch instead of two sequential RMSNorm launches. The math comes from the
``xkernels`` kernel (``xkernels/ops/norm/triton``); this
module is the thin tokenspeed-side launcher + registry registration.

The ``@triton.jit`` kernel is imported through the third-party
boundary, which routes the import through ``tokenspeed_kernel._triton`` so the
kernel binds the ``tokenspeed_triton`` package (not stock ``triton``) — required
for the kernel to compile/run inside the tokenspeed serving process.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Importing the xkernels norm package registers its backends and, as a side
# effect, imports its ``@triton.jit`` kernel under the tokenspeed_triton redirect
# (xkernels' ops/norm/__init__.py routes the import through its _triton_compat
# hook). We then reach in for the compiled-kernel symbol so we can launch it into
# caller-provided output buffers (the upstream launcher always allocates fresh
# outputs).
from xkernels.ops.norm.triton.dual_rmsnorm_kernel import (
    dual_rmsnorm_kernel as _xkernels_dual_rmsnorm_kernel,
)

__all__ = ["dual_rmsnorm"]


@register_kernel(
    "norm",
    "dual_rmsnorm",
    name="triton_dual_rmsnorm",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures(
        ("x1", "x2"), "dense", {torch.float16, torch.bfloat16, torch.float32}
    ),
    priority=Priority.PERFORMANT + 2,
    tags={"portability"},
)
def dual_rmsnorm(
    x1: torch.Tensor,
    w1: torch.Tensor,
    x2: torch.Tensor,
    w2: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float | None = None,
    out1: torch.Tensor | None = None,
    out2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused parallel dual RMSNorm of two independent latents in one launch.

    Computes ``(rmsnorm(x1, w1, eps1), rmsnorm(x2, w2, eps2))`` where each row is
    normalized independently (variance reduced in fp32), numerically identical to
    two separate RMSNorms. Built for the DeepSeek MLA ``q_a`` / ``kv_a`` latents
    on AMD, replacing two sequential RMSNorm launches.

    Args:
        x1: ``[T, d1]`` activations for the first latent.
        w1: ``[d1]`` per-feature RMSNorm weight for the first latent.
        x2: ``[T, d2]`` activations for the second latent (must share ``T``).
        w2: ``[d2]`` per-feature RMSNorm weight for the second latent.
        eps1: variance epsilon for the first latent.
        eps2: variance epsilon for the second latent. Defaults to ``eps1``. The
            fused kernel uses a single epsilon, so ``eps2`` must equal ``eps1``
            (they are identical at every tokenspeed call site, both being
            ``config.rms_norm_eps``); a mismatch raises so callers fall back to
            two sequential RMSNorms rather than silently using the wrong eps.
        out1: optional output buffer for the first latent. When ``None`` the
            first input is normalized in place. Must be contiguous if given.
        out2: optional output buffer for the second latent. When ``None`` the
            second input is normalized in place. Must be contiguous if given.

    Returns:
        ``(out1, out2)`` — the two normed latents in the input dtypes. When
        ``out*`` was ``None`` the corresponding input tensor is returned
        (normalized in place).
    """
    if eps2 is None:
        eps2 = eps1
    if eps1 != eps2:
        raise ValueError(
            "triton_dual_rmsnorm uses a single epsilon for both latents; "
            f"got eps1={eps1} != eps2={eps2}. Fall back to two RMSNorm launches."
        )

    assert x1.shape[0] == x2.shape[0], "x1 and x2 must share the token dim T"

    o1 = x1 if out1 is None else out1
    o2 = x2 if out2 is None else out2

    # The kernel addresses each row as ``base + row * row_stride + cols`` (cols
    # added directly), so it requires only that the *feature* (last) dim is
    # contiguous; the row stride is honored explicitly. This lets us write
    # in place into a non-contiguous latent view (e.g. a slice of the MLA
    # latent cache, or a sub-slice of a packed norm-output buffer) without an
    # extra contiguous copy + copy-back. We only fall back to .contiguous() on
    # inputs whose feature dim is strided (rare).
    if x1.stride(-1) != 1:
        x1 = x1.contiguous()
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    assert (
        o1.stride(-1) == 1 and o2.stride(-1) == 1
    ), "dual_rmsnorm output buffers must be contiguous along the feature dim"

    T, d1 = x1.shape
    d2 = x2.shape[1]
    block_d1 = triton.next_power_of_2(d1)
    block_d2 = triton.next_power_of_2(d2)
    num_warps = max(1, min(16, max(block_d1, block_d2) // 256))
    _xkernels_dual_rmsnorm_kernel[(T,)](
        x1,
        w1,
        o1,
        x2,
        w2,
        o2,
        x1.stride(0),
        o1.stride(0),
        x2.stride(0),
        o2.stride(0),
        d1,
        d2,
        eps1,
        BLOCK_D1=block_d1,
        BLOCK_D2=block_d2,
        num_warps=num_warps,
    )
    return o1, o2

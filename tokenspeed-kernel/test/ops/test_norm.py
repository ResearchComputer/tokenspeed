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

"""Parity tests for the fused dual RMSNorm (norm.dual_rmsnorm).

Runs on an NVIDIA or AMD GPU (validated on MI300A / gfx942). Checks the fused
single-launch kernel against (a) the vendored xkernels pure-torch reference and
(b) the existing two-launch ``triton_rmsnorm`` path it replaces in
``FusedRMSNorm.forward``.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm
from tokenspeed_kernel.ops.norm.triton import dual_rmsnorm
from tokenspeed_kernel.platform import current_platform
from xkernels.ops.norm.reference import (
    dual_rmsnorm_ref,
)

platform = current_platform()
torch.manual_seed(0)

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton dual_rmsnorm tests require an NVIDIA or AMD GPU.",
)


# DeepSeek MLA latents: q_a_lora_rank=1536, kv_lora_rank=512 are the production
# shapes; the smaller/odd dims exercise next_pow2 masking and non-multiple-of-256.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "d1,d2",
    [(1536, 512), (512, 512), (128, 384), (2048, 576), (127, 65)],
)
def test_dual_rmsnorm_matches_reference(
    dtype: torch.dtype, d1: int, d2: int, device: str
) -> None:
    num_tokens = 19
    eps = 1e-6
    x1 = torch.randn(num_tokens, d1, device=device, dtype=dtype)
    x2 = torch.randn(num_tokens, d2, device=device, dtype=dtype)
    w1 = torch.randn(d1, device=device, dtype=dtype)
    w2 = torch.randn(d2, device=device, dtype=dtype)

    # Fused kernel writes into fresh output buffers (no in-place) so we can
    # compare against the untouched inputs.
    out1 = torch.empty_like(x1)
    out2 = torch.empty_like(x2)
    o1, o2 = dual_rmsnorm(x1, w1, x2, w2, eps1=eps, out1=out1, out2=out2)

    ref1, ref2 = dual_rmsnorm_ref(x1, w1, x2, w2, eps)
    torch.testing.assert_close(o1, ref1, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(o2, ref2, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("d1,d2", [(1536, 512), (128, 384)])
def test_dual_rmsnorm_matches_sequential_triton(
    dtype: torch.dtype, d1: int, d2: int, device: str
) -> None:
    """Fused kernel must match the two-launch ``triton_rmsnorm`` path it replaces."""
    num_tokens = 23
    eps = 1e-6
    x1 = torch.randn(num_tokens, d1, device=device, dtype=dtype)
    x2 = torch.randn(num_tokens, d2, device=device, dtype=dtype)
    w1 = torch.randn(d1, device=device, dtype=torch.float32)
    w2 = torch.randn(d2, device=device, dtype=torch.float32)

    seq1 = triton_rmsnorm(x1.clone(), w1, eps)
    seq2 = triton_rmsnorm(x2.clone(), w2, eps)

    out1 = torch.empty_like(x1)
    out2 = torch.empty_like(x2)
    fused1, fused2 = dual_rmsnorm(x1, w1, x2, w2, eps1=eps, out1=out1, out2=out2)

    torch.testing.assert_close(fused1, seq1, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fused2, seq2, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_dual_rmsnorm_inplace(dtype: torch.dtype, device: str) -> None:
    """With no out buffers, the inputs are normalized in place (the production
    in-place call path) and the returned tensors alias the inputs."""
    num_tokens = 7
    d1, d2 = 1536, 512
    eps = 1e-6
    x1 = torch.randn(num_tokens, d1, device=device, dtype=dtype)
    x2 = torch.randn(num_tokens, d2, device=device, dtype=dtype)
    w1 = torch.randn(d1, device=device, dtype=dtype)
    w2 = torch.randn(d2, device=device, dtype=dtype)

    ref1, ref2 = dual_rmsnorm_ref(x1, w1, x2, w2, eps)

    o1, o2 = dual_rmsnorm(x1, w1, x2, w2, eps1=eps)
    assert o1.data_ptr() == x1.data_ptr()
    assert o2.data_ptr() == x2.data_ptr()
    torch.testing.assert_close(o1, ref1, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(o2, ref2, atol=2e-2, rtol=2e-2)


def test_dual_rmsnorm_rejects_mismatched_eps(device: str) -> None:
    """The fused kernel uses one epsilon; mismatched epsilons must raise so the
    caller falls back to two sequential RMSNorms."""
    x1 = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    x2 = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    w1 = torch.ones(128, device=device, dtype=torch.bfloat16)
    w2 = torch.ones(64, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="single epsilon"):
        dual_rmsnorm(x1, w1, x2, w2, eps1=1e-6, eps2=1e-5)

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

"""Op-parity for the opt-in xkernels INT4 W4A16 fused-MoE experts GEMM on MI300A.

Runs the vendored xkernels kernel (``xkernels_moe_int4_w4a16``) and compares its
token-indexed grouped-GEMM output against:

  (a) the in-tree ``triton_moe_fused_experts`` INT4 path it would replace, fed
      the same packed weights / routing / dispatch, and
  (b) the xkernels pure-torch reference (``moe_w4a16_ref``).

This is a *kernel-equivalence* test for the opt-in path; it does not change
which kernel production selects. Runs on a single NVIDIA/AMD GPU (validated on
MI300A / gfx942); skipif no CUDA — do not submit a SLURM job.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU (MI300A) / CUDA required"
)

import tokenspeed_kernel  # noqa: E402
from tokenspeed_kernel._triton import tl  # noqa: E402
from tokenspeed_kernel.ops.moe.xkernels import (  # noqa: E402
    invoke_xkernels_moe_int4_w4a16,
)
from tokenspeed_kernel.thirdparty.xkernels.ops.moe.reference import (  # noqa: E402
    moe_w4a16_ref,
)
from tokenspeed_kernel.thirdparty.xkernels.ops.moe.w4a16 import (  # noqa: E402
    make_w4a16_weights,
)

GROUP = 32


def _dispatch(topk_ids, block_m, E):
    return tokenspeed_kernel.moe_dispatch(
        topk_ids,
        block_m,
        E,
        dtype=torch.int32,
        expected_kernel_name="triton_moe_align_block_size",
    )


def _run_xkernels(A, packed, scale, C, topk_weights, topk_ids, disp, config, top_k):
    sorted_ids, expert_ids, num_pad = disp
    invoke_xkernels_moe_int4_w4a16(
        A=A,
        B=packed,
        bias=None,
        C=C,
        A_scale=None,
        B_scale=scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_pad,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        per_channel_quant=False,
        block_shape=(0, GROUP),
        filter_expert=False,
    )


def _run_in_tree(A, packed, scale, C, topk_weights, topk_ids, disp, config, top_k):
    sorted_ids, expert_ids, num_pad = disp
    tokenspeed_kernel.moe_experts(
        A=A,
        B=packed,
        bias=None,
        C=C,
        A_scale=None,
        B_scale=scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_pad,
        config=config,
        a_use_tma=False,
        b_use_tma=False,
        c_sorted=False,
        mul_routed_weight=False,
        top_k=top_k,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        per_channel_quant=False,
        block_shape=(0, GROUP),
        filter_expert=False,
        dtype=torch.bfloat16,
        features={"dispatch_sorted"},
        expected_kernel_name="triton_moe_fused_experts",
    )


# DeepSeek-V2-Lite / Kimi-K2 routed-expert shapes (K multiple of 8 and GROUP=32;
# N is the gate_up (2*inter) or down (hidden) dim). Small M exercises decode.
@pytest.mark.parametrize(
    "E,N,K,M,top_k,block_m",
    [
        (8, 256, 128, 16, 2, 16),
        (8, 512, 256, 8, 2, 16),
        (16, 256, 512, 4, 4, 16),
        (8, 128, 256, 64, 2, 64),
    ],
)
def test_xkernels_int4_matches_in_tree(E, N, K, M, top_k, block_m):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    A = torch.randn(M, K, device=dev, dtype=dt) * 0.5
    packed, scale, _w_ref = make_w4a16_weights(E, N, K, GROUP, device=dev, seed=1)

    topk_ids = torch.randint(0, E, (M, top_k), device=dev, dtype=torch.int32)
    topk_weights = torch.rand(M, top_k, device=dev, dtype=dt)

    config = dict(
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,  # multiple of pack(8) and GROUP(32)
        GROUP_SIZE_M=1,
        num_warps=4,
        num_stages=2,
    )
    disp = _dispatch(topk_ids, config["BLOCK_SIZE_M"], E)

    C_xk = torch.empty(M * top_k, N, device=dev, dtype=dt)
    C_in = torch.empty(M * top_k, N, device=dev, dtype=dt)
    _run_xkernels(A, packed, scale, C_xk, topk_weights, topk_ids, disp, config, top_k)
    _run_in_tree(A, packed, scale, C_in, topk_weights, topk_ids, disp, config, top_k)
    torch.cuda.synchronize()

    # Same math (unpack + per-group dequant + grouped GEMM), different unpack
    # micro-strategy -> identical up to fp accumulation order.
    torch.testing.assert_close(C_xk.float(), C_in.float(), atol=2e-2, rtol=2e-2)


def test_xkernels_int4_matches_reference():
    """The token-indexed kernel output, reduced over top_k, matches the xkernels
    pure-torch reference ``out[m] = sum_j w[m,j] * (A[m] @ W[e]^T)``."""
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    E, N, K, M, top_k = 8, 256, 128, 16, 2
    A = torch.randn(M, K, device=dev, dtype=dt) * 0.5
    packed, scale, _w_ref = make_w4a16_weights(E, N, K, GROUP, device=dev, seed=2)

    topk_ids = torch.randint(0, E, (M, top_k), device=dev, dtype=torch.int32)
    topk_weights = torch.rand(M, top_k, device=dev, dtype=dt)

    config = dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1,
        num_warps=4,
        num_stages=2,
    )
    disp = _dispatch(topk_ids, config["BLOCK_SIZE_M"], E)

    # mul_routed_weight=False in the kernel (gate_up convention); fold the routing
    # weight in the reduce to match moe_w4a16_ref(..., mul_routed_weight=True).
    C = torch.empty(M * top_k, N, device=dev, dtype=dt)
    _run_xkernels(A, packed, scale, C, topk_weights, topk_ids, disp, config, top_k)
    torch.cuda.synchronize()
    out = (C.view(M, top_k, N).float() * topk_weights.float()[..., None]).sum(dim=1)

    ref = moe_w4a16_ref(
        A, packed, scale, topk_ids, topk_weights, GROUP, mul_routed_weight=True
    )
    torch.testing.assert_close(out, ref.float(), atol=3e-2, rtol=3e-2)

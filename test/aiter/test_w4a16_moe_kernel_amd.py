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

"""Op-parity for the in-kernel INT4 W4A16 fused-MoE GEMM on MI300A.

Runs the fused expert GEMM with packed INT4 weights (use_int4_w4a16=True) and
compares against the same kernel fed the bf16-dequantized weights. The dequant
reference is exact (q*scale), so the only thing under test is the kernel's
unpack + per-group-scale path. Validates the kernel committed in 6c72a32.
"""

import pytest
import torch
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU (MI300A) required"
)

import tokenspeed_kernel  # noqa: E402

from tokenspeed.runtime.layers.quantization.compressed_tensors.w4a16_dequant import (  # noqa: E402
    dequantize_w4a16,
)

GROUP = 32


def _quantize_pack(w: torch.Tensor):
    """[E, N, K] bf16 -> packed [E, N, K//8] int32 + scale [E, N, K//GROUP]."""
    e, n, k = w.shape
    ng = k // GROUP
    wg = w.float().reshape(e, n, ng, GROUP)
    scale = (wg.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8)
    q = torch.round(wg / scale).clamp(-8, 7).reshape(e, n, k)
    u = (q + 8).to(torch.int32).reshape(e, n, k // 8, 8)
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    packed = (u << shifts).sum(dim=-1).to(torch.int32)
    return packed, scale.reshape(e, n, ng).to(w.dtype)


def _run_experts(
    A,
    B,
    C,
    B_scale,
    topk_weights,
    topk_ids,
    sorted_ids,
    expert_ids,
    num_pad,
    config,
    top_k,
    use_int4,
    block_shape,
):
    tokenspeed_kernel.moe_experts(
        A=A,
        B=B,
        bias=None,
        C=C,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_pad,
        config=config,
        a_use_tma=False,
        b_use_tma=False,
        c_sorted=False,
        A_scale=None,
        B_scale=B_scale,
        mul_routed_weight=False,
        top_k=top_k,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=use_int4,
        per_channel_quant=False,
        block_shape=block_shape,
        filter_expert=False,
        dtype=torch.bfloat16,
        features={"dispatch_sorted"},
        expected_kernel_name="triton_moe_fused_experts",
    )


def test_w4a16_moe_kernel_matches_bf16_dequant():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    E, N, K, M, top_k = 8, 256, 128, 16, 2
    A = (torch.randn(M, K, device=dev, dtype=dt)) * 0.5
    W = (torch.randn(E, N, K, device=dev, dtype=dt)) * 0.2

    packed, scale = _quantize_pack(W)
    Wdq = torch.stack(
        [dequantize_w4a16(packed[e], scale[e], GROUP, out_dtype=dt) for e in range(E)]
    )  # [E, N, K] bf16 — exact q*scale reference weights

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
    sorted_ids, expert_ids, num_pad = tokenspeed_kernel.moe_dispatch(
        topk_ids,
        config["BLOCK_SIZE_M"],
        E,
        dtype=torch.int32,
        expected_kernel_name="triton_moe_align_block_size",
    )

    C_int4 = torch.empty(M * top_k, N, device=dev, dtype=dt)
    C_ref = torch.empty(M * top_k, N, device=dev, dtype=dt)
    _run_experts(
        A,
        packed,
        C_int4,
        scale,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_pad,
        config,
        top_k,
        True,
        (0, GROUP),
    )
    _run_experts(
        A,
        Wdq,
        C_ref,
        None,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_pad,
        config,
        top_k,
        False,
        None,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(C_int4.float(), C_ref.float(), atol=2e-2, rtol=2e-2)

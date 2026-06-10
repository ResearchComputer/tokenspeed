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

"""Opt-in INT4 W4A16 fused-MoE experts GEMM backed by vendored ``xkernels``.

This adapts the vendored xkernels INT4 W4A16 grouped-GEMM kernel
(``thirdparty/xkernels/ops/moe/triton/moe_int4_kernel.py``) to the *exact*
``invoke_fused_moe_kernel`` call signature used by
``tokenspeed.runtime.layers.moe.backends.triton_common.triton_forward``, and
registers it as ``xkernels_moe_int4_w4a16``.

**Not selected by default.** The W4A16 backend pins the in-tree kernel via
``expected_kernel_name="triton_moe_fused_experts"``; this kernel is only used
when a caller explicitly requests ``expected_kernel_name="xkernels_moe_int4_w4a16"``.
It exists so the autotuned xkernels variant can be A/B'd against the in-tree
default-config kernel without touching the production path. See
``docs/xkernels-int4-moe-integration-plan.md``.

Triton-package note: the vendored kernel is imported through the third-party
boundary, which routes it through ``tokenspeed_kernel._triton`` so it binds the
``tokenspeed_triton`` package. ``compute_type`` is taken from the same ``tl`` so
``tl.dot`` does not see a cross-package dtype (the kernel additionally casts the
dequantized rhs to ``a.dtype`` before ``tl.dot``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Importing the vendored moe package registers the xkernels backends under the
# tokenspeed_triton redirect (see thirdparty/xkernels/ops/moe/__init__.py). We
# reach in for the kernel object so we can launch it with the caller-supplied
# dispatch block size (the dispatch stage already padded sorted_token_ids to
# config["BLOCK_SIZE_M"], so BLOCK_SIZE_M must match — see the integration plan).
from tokenspeed_kernel.thirdparty.xkernels.ops.moe.triton.moe_int4_kernel import (
    fused_moe_int4_kernel as _xkernels_fused_moe_int4_kernel,
)

__all__ = ["invoke_xkernels_moe_int4_w4a16"]


def _raw_kernel():
    """Return the underlying ``@triton.jit`` kernel (unwrap the autotuner).

    The vendored kernel is wrapped by ``@triton.autotune``; for the opt-in
    in-flow path we launch it with the caller's fixed ``BLOCK_SIZE_M`` (the one
    the dispatch stage padded to) rather than letting autotune pick its own M
    tile, so we unwrap to the raw JIT function and pass an explicit config.
    """
    k = _xkernels_fused_moe_int4_kernel
    # triton.autotune -> Autotuner with .fn = (heuristics ->) JITFunction.
    fn = getattr(k, "fn", k)
    fn = getattr(fn, "fn", fn)  # unwrap a possible @heuristics layer too
    return fn


@register_kernel(
    "moe",
    "experts",
    name="xkernels_moe_int4_w4a16",
    features={"dispatch_sorted"},
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    # One band below the in-tree triton_moe_fused_experts (PERFORMANT + 2) so it
    # never wins auto-selection; reachable only via expected_kernel_name.
    priority=Priority.PERFORMANT + 1,
    tags={"experimental"},
)
def invoke_xkernels_moe_int4_w4a16(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    a_use_tma: bool = False,
    b_use_tma: bool = False,
    c_sorted: bool = False,
    filter_expert: bool = True,
) -> None:
    """Launch the xkernels INT4 W4A16 grouped fused-MoE GEMM in place into ``C``.

    Signature-compatible with ``invoke_fused_moe_kernel`` so it slots into
    ``triton_forward`` unchanged. Only the W4A16 args are honored; TMA / bias /
    activation-scale / sorted-C options are not supported by this kernel.

    Args:
        A: ``[num_valid_tokens // top_k, K]`` bf16 activations (un-repeated; the
            kernel gathers rows via ``sorted_token_ids // top_k``).
        B: ``[E, N, K // 8]`` int32 packed ``uint4b8`` weights.
        bias: must be ``None`` (unsupported).
        C: ``[num_valid_tokens, N]`` output, written token-indexed in place.
        A_scale: must be ``None`` (W4A16 has no activation scale).
        B_scale: ``[E, N, K // group_size]`` bf16 group scales.
        topk_weights: ``[M, top_k]`` fp32 routing weights.
        topk_ids: ``[M, top_k]`` int32 expert ids (used only for its numel).
        sorted_token_ids: ``[EM]`` int32 token-slot ids grouped per expert,
            padded to ``config["BLOCK_SIZE_M"]`` by the dispatch stage.
        expert_ids: ``[num_m_blocks]`` int32 expert per M-block (``-1`` filtered).
        num_tokens_post_padded: ``[1]`` int32.
        mul_routed_weight: fold routing weight into the output (down GEMM).
        top_k: experts per token (gate_up uses real top_k; down uses 1).
        config: dispatch config dict; ``BLOCK_SIZE_M``/``N``/``K``/``GROUP_SIZE_M``
            are used directly so the M tile matches the padded dispatch.
        compute_type: tokenspeed_triton output dtype (e.g. ``tl.bfloat16``).
        use_fp8_w8a8 / use_int8_w8a16: must be ``False``.
        use_int4_w4a16: must be ``True``.
        per_channel_quant: ignored.
        block_shape: ``(0, group_size)`` — group size read from ``block_shape[1]``.
        a_use_tma / b_use_tma / c_sorted: must be falsy (unsupported).
        filter_expert: honor ``-1`` expert ids (EP).

    Returns:
        ``None`` (writes into ``C``).
    """
    assert (
        use_int4_w4a16 and not use_fp8_w8a8 and not use_int8_w8a16
    ), "xkernels_moe_int4_w4a16 supports only the INT4 W4A16 path"
    assert A_scale is None, "W4A16 has no activation scale"
    assert bias is None, "xkernels INT4 MoE GEMM does not support bias"
    assert not (
        a_use_tma or b_use_tma or c_sorted
    ), "xkernels INT4 MoE GEMM does not support TMA / sorted-C output"
    assert B.dtype == torch.int32, "packed W4A16 weights must be int32"
    assert sorted_token_ids.stride(0) == 1
    assert topk_weights.stride(1) == 1
    assert (
        block_shape is not None and block_shape[1] > 0
    ), "group_size must be provided via block_shape=(0, group_size)"
    group_size = block_shape[1]

    E, N, kp = B.shape
    K = kp * 8
    assert K % group_size == 0
    assert B_scale is not None and B_scale.shape == (E, N, K // group_size)

    num_valid_tokens = topk_ids.numel()
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config["BLOCK_SIZE_K"]
    GROUP_SIZE_M = config["GROUP_SIZE_M"]
    assert (
        BLOCK_SIZE_K % 8 == 0 and BLOCK_SIZE_K % group_size == 0
    ), "BLOCK_SIZE_K must be a multiple of the pack factor (8) and group_size"
    EVEN_K = K % BLOCK_SIZE_K == 0

    def grid(meta):
        return (
            triton.cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M)
            * triton.cdiv(N, BLOCK_SIZE_N),
        )

    raw = _raw_kernel()
    raw[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights.reshape(-1).to(torch.float32),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        num_valid_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        group_k=group_size,
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=compute_type,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        EVEN_K=EVEN_K,
        FILTER_EXPERT=filter_expert,
    )

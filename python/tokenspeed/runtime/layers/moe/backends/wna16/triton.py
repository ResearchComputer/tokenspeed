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

"""AMD W4A16 MoE backend with the in-kernel INT4 dequant Triton fused MoE.

Loads compressed-tensors INT4 (``pack-quantized``, group, symmetric) routed-expert
weights exactly like the dequant backend, but keeps them packed: the fused MoE GEMM
unpacks 4-bit weights and applies the per-group scale inside the kernel
(``use_int4_w4a16=True``). This preserves the ~4x memory saving of the quantized
checkpoint (no bf16 materialization), which is what lets large INT4 MoE models fit
on a bounded number of GPUs. Numerically validated against the dequant oracle by
``test/aiter/test_w4a16_moe_kernel_amd.py``.

Restricted to 4-bit group-quantized, symmetric, non act-order checkpoints (what the
in-kernel path implements); the selector falls back to ``wna16/triton_dequant`` for
anything else.
"""

from __future__ import annotations

import logging

import torch
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.triton_common import (
    build_triton_gemms,
    triton_forward,
)
from tokenspeed.runtime.layers.moe.backends.wna16.weights import attach_marlin_weights
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import CompressedTensorsConfig

logger = logging.getLogger(__name__)


def _to_kernel_layout(t: torch.Tensor) -> torch.Tensor:
    """Checkpoint per-expert ``[E, *, N]`` -> fused-MoE ``[E, N, *]`` (B / B_scale).

    ``attach_marlin_weights`` stores the (``is_transposed``) checkpoint layout:
    packed weights as ``[E, K // pack, N]`` and group scales as ``[E, K // group, N]``.
    The fused MoE GEMM indexes both as ``[E, N, K//pack]`` / ``[E, N, K//group]``
    using explicit strides, so a (zero-copy) transpose of the last two dims is all
    that is needed. We deliberately do **not** call ``.contiguous()``: the kernel
    reads via strides and never builds a TMA descriptor on AMD, and materializing a
    contiguous copy would transiently double the (large) packed-weight footprint.
    """
    return t.transpose(1, 2)


class Wna16TritonBackend(MoEBackend):
    """W4A16 routed-expert MoE on AMD with the in-kernel INT4 Triton fused MoE."""

    supported_arches = frozenset({"any"})

    def __init__(self, key, spec: MoELayerSpec, quant_config, routing_config=None):
        super().__init__(key, spec, quant_config, routing_config)
        config = quant_config.target_scheme_map["Linear"].get("weights")
        self._num_bits = config.num_bits
        self._packed_factor = 32 // config.num_bits
        self._strategy = config.strategy
        self._group_size = config.group_size
        self._actorder = config.actorder
        assert config.num_bits == 4, "in-kernel INT4 path supports 4-bit only"
        assert config.symmetric, "Only symmetric W4A16 is supported"
        assert not self._actorder, "act-order (g_idx) W4A16 is not supported on AMD yet"
        assert (
            isinstance(self._group_size, int) and self._group_size > 0
        ), "in-kernel INT4 path requires group quantization (positive group_size)"

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not isinstance(quant_config, CompressedTensorsConfig):
            return False
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        if not (
            current_platform().is_amd
            and spec.activation in {"silu", "swiglu"}
            and quant_config._is_wNa16_group_channel(weight_quant, input_quant)
        ):
            return False
        # The in-kernel path implements 4-bit, symmetric, group-quantized (positive
        # group_size), non act-order. Everything else falls back to dequant-on-load.
        return (
            weight_quant is not None
            and weight_quant.num_bits == 4
            and bool(weight_quant.symmetric)
            and not weight_quant.actorder
            and isinstance(weight_quant.group_size, int)
            and weight_quant.group_size > 0
        )

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        del with_bias
        # Same packed int32 weight + group-scale params (and loaders) as the dequant
        # backend, so the checkpoint loads unchanged. We keep them packed.
        attach_marlin_weights(self, layer)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Reinterpret the packed weights / scales into the fused-MoE B / B_scale
        # layout (zero-copy transpose) and keep them INT4 — no dequant.
        w13_packed = _to_kernel_layout(layer.w13_weight_packed.data)  # [E, N, K//8]
        w2_packed = _to_kernel_layout(layer.w2_weight_packed.data)
        w13_scale = _to_kernel_layout(layer.w13_weight_scale.data)  # [E, N, K//group]
        w2_scale = _to_kernel_layout(layer.w2_weight_scale.data)
        for name in (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_shape",
            "w2_weight_shape",
            "w13_weight_g_idx",
            "w2_weight_g_idx",
        ):
            if hasattr(layer, name):
                delattr(layer, name)
        layer.register_parameter(
            "w13_weight", nn.Parameter(w13_packed, requires_grad=False)
        )
        layer.register_parameter(
            "w2_weight", nn.Parameter(w2_packed, requires_grad=False)
        )
        logger.info(
            "wna16/triton (in-kernel INT4) active: %d-bit group=%d, w13=%s %s, "
            "w2=%s %s (weights kept packed, no bf16 dequant)",
            self._num_bits,
            self._group_size,
            tuple(w13_packed.shape),
            w13_packed.dtype,
            tuple(w2_packed.shape),
            w2_packed.dtype,
        )
        self._gate_up_gemm, self._down_gemm, self._get_config_func = build_triton_gemms(
            layer,
            self.spec,
            use_int4_w4a16=True,
            pack_factor=self._packed_factor,
            block_shape=(0, self._group_size),
            gate_up_B_scale=w13_scale,
            down_B_scale=w2_scale,
            dtype_tag="bf16",
        )

    def forward(
        self,
        layer: nn.Module,
        hidden_states,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ):
        del num_global_tokens, max_num_tokens_per_gpu
        return triton_forward(
            self._gate_up_gemm,
            self._down_gemm,
            self._get_config_func,
            layer.activation,
            layer,
            hidden_states,
            topk_output,
        )

    @property
    def apply_routed_scaling_factor_on_output(self) -> bool:
        return False


__all__ = ["Wna16TritonBackend"]

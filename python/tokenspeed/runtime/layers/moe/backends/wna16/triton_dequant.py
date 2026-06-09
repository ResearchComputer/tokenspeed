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

"""AMD W4A16 MoE backend via dequant-on-load.

Loads compressed-tensors INT4 (``pack-quantized``, group, symmetric) routed-expert
weights exactly like the NVIDIA Marlin backend, then dequantizes them to bf16 in
``process_weights_after_loading`` and serves through the standard bf16 Triton fused
MoE. No INT4 kernel required — this is the portable AMD baseline (and the numerical
oracle for the in-kernel INT4 path). Trades 4x weight memory for not needing a
quantized kernel; see ``wna16/triton.py`` for the footprint-preserving variant.
"""

from __future__ import annotations

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
from tokenspeed.runtime.layers.quantization.compressed_tensors.w4a16_dequant import (
    dequantize_w4a16,
)


def _dequant_experts(
    packed: torch.Tensor, scale: torch.Tensor, group_size: int, num_bits: int
) -> torch.Tensor:
    """Dequant transposed per-expert packed weights to dense bf16.

    ``packed`` is ``[E, K // pack, N]`` (``is_transposed`` checkpoint layout) and
    ``scale`` is ``[E, K // group_size, N]``. Returns dense ``[E, N, K]`` bf16,
    matching the bf16 fused-MoE expert layout (``B`` is ``[E, N, K]``).
    """
    num_experts = packed.shape[0]
    dense = []
    for e in range(num_experts):
        pe = packed[e].transpose(0, 1).contiguous()  # [N, K // pack]
        se = scale[e].transpose(0, 1).contiguous()  # [N, K // group_size]
        dense.append(
            dequantize_w4a16(
                pe, se, group_size, num_bits=num_bits, out_dtype=torch.bfloat16
            )
        )
    return torch.stack(dense, dim=0)  # [E, N, K]


class Wna16DequantBackend(MoEBackend):
    """W4A16 routed-expert MoE on AMD by dequantizing INT4 -> bf16 at load."""

    supported_arches = frozenset({"any"})

    def __init__(self, key, spec: MoELayerSpec, quant_config, routing_config=None):
        super().__init__(key, spec, quant_config, routing_config)
        config = quant_config.target_scheme_map["Linear"].get("weights")
        self._num_bits = config.num_bits
        self._packed_factor = 32 // config.num_bits
        self._strategy = config.strategy
        self._group_size = config.group_size
        self._actorder = config.actorder
        assert config.symmetric, "Only symmetric W4A16 is supported"
        assert not self._actorder, "act-order (g_idx) W4A16 is not supported on AMD yet"

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not isinstance(quant_config, CompressedTensorsConfig):
            return False
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        return (
            current_platform().is_amd
            and spec.activation in {"silu", "swiglu"}
            and quant_config._is_wNa16_group_channel(weight_quant, input_quant)
        )

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        del with_bias
        # Same packed int32 weight + group-scale params (and loaders) as Marlin, so
        # the checkpoint loads unchanged. Dequant happens after loading.
        attach_marlin_weights(self, layer)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        gs = self._group_size
        w13 = _dequant_experts(
            layer.w13_weight_packed.data,
            layer.w13_weight_scale.data,
            gs,
            self._num_bits,
        )
        w2 = _dequant_experts(
            layer.w2_weight_packed.data,
            layer.w2_weight_scale.data,
            gs,
            self._num_bits,
        )
        # Register dense bf16 expert weights and drop the packed/quant params.
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
        layer.register_parameter("w13_weight", nn.Parameter(w13, requires_grad=False))
        layer.register_parameter("w2_weight", nn.Parameter(w2, requires_grad=False))
        # Now that dense bf16 weights exist, build the standard bf16 Triton gemms.
        self._gate_up_gemm, self._down_gemm, self._get_config_func = build_triton_gemms(
            layer, self.spec, dtype_tag="bf16"
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


__all__ = ["Wna16DequantBackend"]

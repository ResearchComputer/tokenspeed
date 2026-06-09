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

"""Round-trip test for compressed-tensors W4A16 dequant (CPU; no GPU required)."""

import pytest
import torch

from tokenspeed.runtime.layers.quantization.compressed_tensors.w4a16_dequant import (
    dequantize_w4a16,
    unpack_w4a16,
)


def _pack_w4a16(q_uint4: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Pack ``[out, in]`` uint4 values (0..15) into ``[out, in // 8]`` int32."""
    pack = 32 // num_bits
    out, in_features = q_uint4.shape
    q = q_uint4.to(torch.int32).reshape(out, in_features // pack, pack)
    shifts = torch.arange(pack, dtype=torch.int32) * num_bits
    return (q << shifts).sum(dim=-1).to(torch.int32)


def _quantize_w4a16_symmetric(w: torch.Tensor, group_size: int):
    """Quantize ``[out, in]`` -> (packed int32, scales, dequant reference)."""
    out, in_features = w.shape
    num_groups = in_features // group_size
    wg = w.float().reshape(out, num_groups, group_size)
    scales = (wg.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8)
    q = torch.round(wg / scales).clamp(-8, 7)  # signed int4 [-8, 7]
    w_ref = (q * scales).reshape(out, in_features)
    q_uint = (q + 8).to(torch.int32).reshape(out, in_features)  # uint4b8 0..15
    packed = _pack_w4a16(q_uint)
    return packed, scales.reshape(out, num_groups), w_ref


@pytest.mark.parametrize(
    "out_features,in_features,group_size", [(64, 128, 32), (16, 256, 32)]
)
def test_unpack_recovers_signed_values(out_features, in_features, group_size):
    torch.manual_seed(0)
    q_signed = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int32)
    packed = _pack_w4a16((q_signed + 8).to(torch.int32))
    got = unpack_w4a16(packed)
    assert torch.equal(got, q_signed), "unpack must recover the signed int4 values"


@pytest.mark.parametrize(
    "out_features,in_features,group_size", [(64, 128, 32), (16, 256, 32)]
)
def test_dequantize_w4a16_matches_reference(out_features, in_features, group_size):
    torch.manual_seed(0)
    w = torch.randn(out_features, in_features)
    packed, scales, w_ref = _quantize_w4a16_symmetric(w, group_size)
    w_got = dequantize_w4a16(packed, scales, group_size, out_dtype=torch.float32)
    torch.testing.assert_close(w_got, w_ref, atol=1e-5, rtol=1e-5)
    # And the dequant should track the original weight to within INT4 group
    # rounding (~step/4 ≈ 0.09 mean-abs-error for unit-variance randn, group 32).
    assert (w_got - w).abs().mean().item() < 0.15

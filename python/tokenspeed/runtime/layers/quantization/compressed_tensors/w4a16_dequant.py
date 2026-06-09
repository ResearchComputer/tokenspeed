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

"""Dequantize compressed-tensors W4A16 ``pack-quantized`` weights to a dense
floating-point tensor.

Layout (compressed-tensors, ``num_bits=4``, symmetric / ``uint4b8``):

* ``qweight_packed``: ``int32`` of shape ``[out, in // 8]``. Each int32 packs 8
  consecutive ``in`` values; nibble ``i`` (bits ``4*i .. 4*i+3``) holds the
  quantized value for input index ``col * 8 + i``. Values are stored unsigned
  with a bias of 8 (``uint4b8``): stored ``u`` represents signed ``u - 8`` in
  ``[-8, 7]``.
* ``scales``: shape ``[out, in // group_size]`` — one scale per (row, K-group).
* No zero-points (symmetric).

This is the reference path used by the dequant-on-load MoE backend and as the
numerical oracle for the in-kernel INT4 path.
"""

from __future__ import annotations

import torch

__all__ = ["dequantize_w4a16", "unpack_w4a16"]


def unpack_w4a16(qweight_packed: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Unpack ``[out, in // pack]`` int32 into signed ``[out, in]`` int values.

    Args:
        qweight_packed: ``int32`` packed weights ``[out, in // pack]`` where
            ``pack = 32 // num_bits`` values share each int32 along the input dim.
        num_bits: bits per quantized value (only 4 is exercised here).

    Returns:
        ``int32`` tensor ``[out, in]`` with values in ``[-8, 7]`` (uint4b8 → signed).
    """
    pack = 32 // num_bits
    mask = (1 << num_bits) - 1
    out, in_packed = qweight_packed.shape
    shifts = (
        torch.arange(pack, device=qweight_packed.device, dtype=torch.int32) * num_bits
    )
    # [out, in_packed, pack] — low nibble first (col*pack + i ordering).
    u = (qweight_packed.unsqueeze(-1) >> shifts) & mask
    u = u.reshape(out, in_packed * pack)
    return u.to(torch.int32) - (1 << (num_bits - 1))


def dequantize_w4a16(
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    *,
    num_bits: int = 4,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed W4A16 weights to a dense ``out_dtype`` tensor ``[out, in]``.

    Args:
        qweight_packed: ``int32`` ``[out, in // (32 // num_bits)]`` packed weights.
        scales: ``[out, in // group_size]`` per-group scales.
        group_size: number of input elements sharing a scale (e.g. 32).
        num_bits: bits per value (4).
        out_dtype: dtype of the returned dense weight.

    Returns:
        Dense weight ``[out, in]`` in ``out_dtype`` (``signed_int4 * group_scale``).
    """
    q = unpack_w4a16(qweight_packed, num_bits=num_bits)  # [out, in] signed int
    out, in_features = q.shape
    num_groups = in_features // group_size
    assert scales.shape[0] == out, (scales.shape, out)
    assert scales.shape[1] == num_groups, (scales.shape, num_groups)
    q = q.reshape(out, num_groups, group_size).to(torch.float32)
    w = q * scales.reshape(out, num_groups, 1).to(torch.float32)
    return w.reshape(out, in_features).to(out_dtype)

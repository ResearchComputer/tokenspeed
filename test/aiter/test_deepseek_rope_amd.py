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

"""DeepseekScalingRotaryEmbedding AMD side-effect contract (MI300A only).

The DeepSeek MLA model calls rotary_emb as a *statement*: it relies on RoPE being
written into ``output_q_rope`` and applied to ``key`` in-place, and discards the
return value (flattened [tokens, heads, rotary_dim] layout, 1-D positions). This
test asserts that contract on the AMD path and checks correctness vs a reference.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU (MI300A) required"
)

from tokenspeed.runtime.layers.rotary_embedding import (  # noqa: E402
    DeepseekScalingRotaryEmbedding,
)


def _ref_neox(x, cos, sin, rd):
    x1, x2 = x[..., : rd // 2], x[..., rd // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def test_deepseek_rope_amd_writes_in_place_and_matches_reference():
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    rd, H, T = 64, 8, 5
    rope = DeepseekScalingRotaryEmbedding(rd, rd, 4096, 10000, True, 1.0, dt).to(dev)

    positions = torch.arange(T, device=dev)
    q = torch.randn(T, H, rd, device=dev, dtype=dt)
    k = torch.randn(T, 1, rd, device=dev, dtype=dt)
    out_q = torch.zeros(T, H, rd, device=dev, dtype=dt)
    q_in, k_in = q.clone(), k.clone()

    rope(positions, q, k, output_q_rope=out_q)
    torch.cuda.synchronize()

    # Contract: output_q_rope must be written (RoPE applied), key roped in-place.
    assert out_q.abs().sum().item() > 0, "output_q_rope not written: RoPE is a no-op"
    assert not torch.equal(k, k_in), "key not roped in-place"

    # Correctness vs an independent neox reference using the module's cache.
    cs = rope.cos_sin_cache[positions]
    cos, sin = cs.chunk(2, dim=-1)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2).float()
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2).float()
    ref_q = _ref_neox(q_in.float(), cos, sin, rd)
    ref_k = _ref_neox(k_in.float(), cos, sin, rd)
    torch.testing.assert_close(out_q.float(), ref_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.float(), ref_k, atol=2e-2, rtol=2e-2)

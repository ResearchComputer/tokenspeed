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

"""Round-trip test for the MLA KV-cache scatter write (MI300A only).

This is the prefill cache-write path used by DeepSeek MLA models
(`set_mla_kv_buffer` -> Triton scatter). It writes a per-token latent
``[nope | rope]`` into the paged KV buffer at scattered locations; the decode
kernel later reads it back. Validates placement/correctness on gfx942.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU (MI300A) required"
)

from tokenspeed.runtime.cache.utils import set_mla_kv_buffer_triton  # noqa: E402


@pytest.mark.parametrize("n", [1, 8, 37, 128])
def test_set_mla_kv_buffer_roundtrip(n):
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    kv_lora, qk_rope = 512, 64
    dim = kv_lora + qk_rope
    N = 256

    kv_buffer = torch.zeros(N, 1, dim, device=dev, dtype=dt)
    loc = torch.randperm(N, device=dev)[:n].to(torch.int64)
    nope = torch.randn(n, 1, kv_lora, device=dev, dtype=dt)
    rope = torch.randn(n, 1, qk_rope, device=dev, dtype=dt)

    set_mla_kv_buffer_triton(kv_buffer, loc, nope, rope, enable_pdl=False)
    torch.cuda.synchronize()

    # Each written location must hold [nope | rope]; unwritten rows stay zero.
    for i in range(n):
        row = kv_buffer[loc[i].item(), 0]
        torch.testing.assert_close(row[:kv_lora], nope[i, 0], atol=0, rtol=0)
        torch.testing.assert_close(row[kv_lora:], rope[i, 0], atol=0, rtol=0)

    written = torch.zeros(N, dtype=torch.bool, device=dev)
    written[loc] = True
    if (~written).any():
        assert kv_buffer[~written].abs().sum().item() == 0.0, "unwritten rows changed"

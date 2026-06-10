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

"""AMD moe_align_block_size parity vs the torch reference (MI300A only)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU (MI300A) required"
)

from tokenspeed_kernel.numerics.moe import canonicalize_align_block_size  # noqa: E402
from tokenspeed_kernel.numerics.reference.moe import (  # noqa: E402
    torch_moe_align_block_size,
)
from tokenspeed_kernel.ops.moe.triton import moe_align_block_size_amd  # noqa: E402

CASES = [
    (128, 6, 64, 16),  # DeepSeek-V2-Lite-ish: 64 experts, top-6
    (1, 8, 64, 16),  # single token (decode)
    (37, 2, 8, 16),  # uneven, small
    (256, 8, 64, 32),  # larger block
    (64, 1, 4, 16),  # few experts
    (1, 8, 48, 16),  # Kimi-K2.6 decode: 48 local experts (EP=8), top-8
    (512, 8, 48, 16),  # Kimi-K2.6 prefill-scale: many tokens, sparse per-expert
    (256, 8, 384, 32),  # many experts, most empty per call (sparse blocks)
]


@pytest.mark.parametrize("m, top_k, num_experts, block_size", CASES)
def test_amd_align_matches_reference(m, top_k, num_experts, block_size):
    torch.manual_seed(0)
    topk_ids = torch.randint(
        0, num_experts, (m, top_k), device="cuda", dtype=torch.int32
    )
    s, e, n = moe_align_block_size_amd(topk_ids, block_size, num_experts)

    # canonicalize requires sorted_ids.numel() == expert_ids.numel()*block_size.
    n_blocks = e.numel()
    pad_id = topk_ids.numel()
    padded = torch.full(
        (n_blocks * block_size,), pad_id, dtype=torch.int32, device=s.device
    )
    padded[: s.numel()] = s
    mine = canonicalize_align_block_size(padded, e, n, block_size)
    ref = torch_moe_align_block_size(topk_ids, block_size, num_experts)
    assert torch.equal(mine, ref), f"mismatch for {(m, top_k, num_experts, block_size)}"


def test_amd_align_selected_on_amd():
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_amd:
        pytest.skip("AMD only")

    from tokenspeed_kernel.ops.moe import moe_dispatch

    topk_ids = torch.randint(0, 64, (128, 6), device="cuda", dtype=torch.int32)
    # Must not raise "Kernel implementation not found": AMD selects the torch impl.
    s, e, n = moe_dispatch(
        topk_ids,
        16,
        64,
        dtype=torch.int32,
        expected_kernel_name="triton_moe_align_block_size",
    )
    assert s.dtype == torch.int32 and n.item() > 0

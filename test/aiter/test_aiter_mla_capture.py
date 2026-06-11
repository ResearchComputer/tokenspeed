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

"""HIP-graph capturability test for AITER mla_decode_fwd (AMD MI300A only).

Guards the assumption AiterMLABackend's HIP-graph decode capture relies on:
mla_decode_fwd CAPTURED at seq_len==1 (the cuda-graph seq-len fill value)
REPLAYS correctly for arbitrary/larger seq_lens, because the asm kernel reads
its paged metadata at runtime rather than baking the grid/kv-splits at capture.
If this regresses, decode graph capture would silently truncate long sequences.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU (MI300A) required"
)

from tokenspeed_kernel.ops.attention import aiter_mla as kops  # noqa: E402

H, KV_LORA, QK_ROPE = 8, 512, 64  # Kimi-K2.6 per-rank: attn_tp=8 -> gqa=8
DQ = KV_LORA + QK_ROPE  # 576
PAGE = 1
BS = 4
MAXP = 4096  # max pages/req (page_size=1, max_model_len=4096)


def _ref(kv_pool, seq_lens, q_vals, sm_scale):
    outs = []
    for i, length in enumerate(seq_lens):
        kv = kv_pool[i * MAXP : i * MAXP + length, 0, 0, :].float()
        qi = q_vals[i].float()
        scores = torch.einsum("hd,sd->hs", qi, kv) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hs,sd->hd", probs, kv[:, :KV_LORA]))
    return torch.stack(outs, 0)


def test_mla_decode_hip_graph_capture_replay():
    assert kops.is_available(), "aiter.mla not importable"
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    sm_scale = 1.0 / (DQ**0.5)
    npool = BS * MAXP

    # Persistent (graph-static) buffers — the addresses the capture records.
    q_buf = torch.zeros(BS, H, DQ, device=dev, dtype=dt)
    kv_pool = torch.randn(npool, PAGE, 1, DQ, device=dev, dtype=dt)
    o_buf = torch.empty(BS, H, KV_LORA, device=dev, dtype=dt)
    qo_indptr = torch.arange(0, BS + 1, device=dev, dtype=torch.int32)
    kv_indptr = torch.zeros(BS + 1, device=dev, dtype=torch.int32)
    kv_indices = torch.zeros(npool, device=dev, dtype=torch.int32)
    kv_last = torch.ones(BS, device=dev, dtype=torch.int32)

    def fill(seq_lens, q_vals):
        sl = torch.tensor(seq_lens, device=dev, dtype=torch.int64)
        numpages = (sl + PAGE - 1) // PAGE
        kv_indptr.zero_()
        kv_indptr[1:] = torch.cumsum(numpages, 0).to(torch.int32)
        kv_indices.zero_()
        pos = 0
        for i, np_i in enumerate(numpages.tolist()):
            kv_indices[pos : pos + np_i] = torch.arange(
                i * MAXP, i * MAXP + np_i, device=dev, dtype=torch.int32
            )
            pos += np_i
        kv_last.copy_(((sl - 1) % PAGE + 1).to(torch.int32))
        q_buf.copy_(q_vals)

    def run():
        kops.mla_decode_fwd(
            q_buf.view(-1, H, DQ),
            kv_pool,
            o_buf,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last,
            max_seqlen_q=1,
            page_size=PAGE,
            sm_scale=sm_scale,
        )

    # Capture at seq_len==1 (mirrors get_cuda_graph_seq_len_fill_value()).
    q_cap = torch.randn(BS, H, DQ, device=dev, dtype=dt)
    fill([1, 1, 1, 1], q_cap)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()

    # Replay at large / varied seq_lens — must match eager (no capture-time freeze).
    for seq_lens in ([4096, 2000, 777, 13], [128, 4096, 1, 999]):
        q_vals = torch.randn(BS, H, DQ, device=dev, dtype=dt)
        fill(seq_lens, q_vals)
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            o_buf.float(),
            _ref(kv_pool, seq_lens, q_vals, sm_scale),
            atol=2e-2,
            rtol=2e-2,
        )

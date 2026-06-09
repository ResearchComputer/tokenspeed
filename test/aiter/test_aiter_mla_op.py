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

"""Numerical-parity test for the AITER MLA op wrapper (AMD MI300A only).

Run inside the ROCm AITER image on an MI300A node, e.g. via the cluster harness
``jobs/test-aiter.sbatch``:
    python -m pytest test/aiter/test_aiter_mla_op.py -v
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU (MI300A) required"
)

from tokenspeed_kernel.ops.attention import aiter_mla as kops  # noqa: E402


def _ref_mla_decode(q, kv_per_req, sm_scale):
    """Reference absorbed-MLA decode: 1 query/req attends all cached kv tokens.

    q: [B, H, Dq]; kv_per_req: list of [S_i, Dq]; returns [B, H, kv_lora_rank].
    Scores over the full Dq dim; value = latent slice [:kv_lora_rank].
    """
    Dq = q.shape[-1]
    kv_lora = Dq - 64  # qk_rope_head_dim = 64
    outs = []
    for i, kv in enumerate(kv_per_req):
        k = kv.float()  # [S, Dq]
        qi = q[i].float()  # [H, Dq]
        scores = torch.einsum("hd,sd->hs", qi, k) * sm_scale  # [H, S]
        p = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hs,sd->hd", p, k[:, :kv_lora]))  # [H, kv_lora]
    return torch.stack(outs, dim=0)  # [B, H, kv_lora]


def test_mla_decode_matches_reference():
    assert kops.is_available(), "aiter.mla not importable"
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    B, H, kv_lora, qk_rope = 2, 16, 512, 64
    Dq = kv_lora + qk_rope  # 576
    page = 1
    S = 8  # kv tokens per request
    sm_scale = 1.0 / (Dq**0.5)

    q = torch.randn(B, H, Dq, device=dev, dtype=dt)
    # decode kv_buffer is 4-D [num_page, page_size, nhead_kv=1, Dq]; page_size=1
    # -> one token per page (tokenspeed's [N,1,Dq] pool viewed as [N,1,1,Dq]).
    kv_buffer = torch.randn(B * S, page, 1, Dq, device=dev, dtype=dt)
    o = torch.empty(B, H, kv_lora, device=dev, dtype=dt)

    qo_indptr = torch.arange(0, B + 1, device=dev, dtype=torch.int32)  # 1 q/req
    kv_indptr = torch.arange(0, (B + 1) * S, S, device=dev, dtype=torch.int32)
    kv_indices = torch.arange(0, B * S, device=dev, dtype=torch.int32)  # seq pages
    kv_last_page_lens = torch.ones(B, device=dev, dtype=torch.int32)

    kops.mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q=1,
        page_size=page,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()

    kv_per_req = [kv_buffer[i * S : (i + 1) * S, 0, 0, :] for i in range(B)]
    ref = _ref_mla_decode(q, kv_per_req, sm_scale)
    torch.testing.assert_close(o.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_flash_attn_varlen_causal_matches_reference():
    """Prefill uses full-MHA flash_attn_varlen_func (decompressed q/k/v)."""
    assert kops.is_available()
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    B, H, qk_head_dim, v_head_dim = 2, 16, 192, 128  # DeepSeek MLA prefill dims
    L = 8  # prompt length per request (no prefix -> causal self-attention)
    sm_scale = 1.0 / (qk_head_dim**0.5)
    total = B * L

    q = torch.randn(total, H, qk_head_dim, device=dev, dtype=dt)
    k = torch.randn(total, H, qk_head_dim, device=dev, dtype=dt)
    v = torch.randn(total, H, v_head_dim, device=dev, dtype=dt)
    cu = torch.arange(0, (B + 1) * L, L, device=dev, dtype=torch.int32)

    out = kops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu,
        cu,
        L,
        L,
        softmax_scale=sm_scale,
        causal=True,
    )
    out = out[0] if isinstance(out, tuple) else out
    torch.cuda.synchronize()

    # Reference: per-request causal MHA.
    refs = []
    for i in range(B):
        qi = q[i * L : (i + 1) * L].float().transpose(0, 1)  # [H,L,d]
        ki = k[i * L : (i + 1) * L].float().transpose(0, 1)
        vi = v[i * L : (i + 1) * L].float().transpose(0, 1)
        s = torch.matmul(qi, ki.transpose(-1, -2)) * sm_scale  # [H,L,L]
        mask = torch.ones(L, L, dtype=torch.bool, device=dev).tril()
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        refs.append(torch.matmul(p, vi).transpose(0, 1))  # [L,H,d]
    ref = torch.cat(refs, dim=0)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_aiter_mla_registered_on_amd():
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_amd:
        pytest.skip("AMD only")
    import tokenspeed.runtime.layers.attention.backends  # noqa: F401  (registers)
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.layers.attention.registry import (
        _BACKEND_REGISTRY,
        _get_default_backend_name,
    )

    assert "aiter_mla" in _BACKEND_REGISTRY
    assert AttentionArch.MLA in _BACKEND_REGISTRY["aiter_mla"][0]
    assert _get_default_backend_name(AttentionArch.MLA) == "aiter_mla"


def test_decode_metadata_indptr_math():
    """The page-count / last-page-len math the backend uses to build AITER metadata."""
    page = 16
    seq_lens = torch.tensor([1, 16, 17, 40])
    kv_indptr = torch.zeros(5, dtype=torch.int32)
    for i, L in enumerate(seq_lens.tolist()):
        kv_indptr[i + 1] = kv_indptr[i] + (L + page - 1) // page
    last = ((seq_lens - 1) % page + 1).to(torch.int32)
    assert kv_indptr.tolist() == [0, 1, 2, 4, 7]
    assert last.tolist() == [1, 16, 1, 8]

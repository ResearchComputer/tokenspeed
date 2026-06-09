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

"""AMD MLA attention backend (ROCm AITER), eager-only v1.

Wraps AITER's paged absorbed-MLA kernels so DeepSeek-V3 / Kimi-style models run
on AMD CDNA3/CDNA4 (gfx942/gfx950). Both decode and extend use the same paged
metadata over the absorbed latent KV pool; there is no separate ragged/normal
path (unlike the NVIDIA ``FlashMLABackend``).

v1 constraints (see docs/superpowers/specs/2026-06-09-aiter-mla-backend-design.md):
- eager only — CUDA/HIP-graph capture is not supported; run with ``--enforce-eager``.
- bf16 KV (no fp8).
- KV is written by the model (this backend is in ``_MLA_KERNEL_BACKENDS``), so
  ``forward_*`` only read the pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed_kernel.ops.attention import aiter_mla as kops

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class AiterMLAMetadata:
    """Paged metadata for one forward pass (decode or extend)."""

    qo_indptr: torch.Tensor          # [bs+1] int32, cumsum of query lengths
    kv_indptr: torch.Tensor          # [bs+1] int32, cumsum of pages per request
    kv_indices: torch.Tensor         # [total_pages] int32, page ids per request
    kv_last_page_lens: torch.Tensor  # [bs]   int32, fill of each request's last page
    max_seqlen_q: int


class AiterMLABackend(AttentionBackend):
    """Eager MLA backend for AMD via AITER. No graph capture in v1."""

    def __init__(self, config: MLAConfig):
        super().__init__(config)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        self.num_q_heads = config.num_attention_heads
        self.page_size = config.page_size
        self.forward_decode_metadata: AiterMLAMetadata | None = None
        self.forward_extend_metadata: AiterMLAMetadata | None = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _build_kv_paging(self, req_pool_indices, seq_lens, req_to_page):
        """Build AITER paged KV metadata from a block table + cache lengths.

        ``block_table = req_to_page[req_pool_indices]`` is ``[bs, max_pages]`` of
        physical page ids. For each request we keep the first
        ``ceil(seq_len/page_size)`` pages (row-major), producing the flat
        ``kv_indices`` whose request boundaries are ``kv_indptr``.
        """
        block_table = req_to_page[req_pool_indices]  # [bs, max_pages]
        bs, max_pages = block_table.shape
        device = block_table.device
        seq_lens = seq_lens.to(device=device, dtype=torch.int64)
        num_pages = (seq_lens + self.page_size - 1) // self.page_size  # [bs]

        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(num_pages, dim=0).to(torch.int32)

        page_pos = torch.arange(max_pages, device=device)
        mask = page_pos[None, :] < num_pages[:, None]  # [bs, max_pages]
        kv_indices = block_table[mask].to(torch.int32)  # flat, row-major
        kv_last_page_lens = ((seq_lens - 1) % self.page_size + 1).to(torch.int32)
        return kv_indptr, kv_indices, kv_last_page_lens

    def init_forward_metadata(self, bs, num_extends, req_pool_indices, seq_lens,
                              forward_mode, req_to_page=None, extend_seq_lens=None,
                              **kwargs):
        device = seq_lens.device
        if forward_mode.is_extend_or_mixed():
            # Extend/prefill (Task 5): qo covers the new tokens per request.
            n = num_extends
            kv_indptr, kv_indices, kv_last = self._build_kv_paging(
                req_pool_indices[:n], seq_lens[:n], req_to_page)
            ext = extend_seq_lens[:n].to(device=device, dtype=torch.int64)
            qo_indptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
            qo_indptr[1:] = torch.cumsum(ext, dim=0).to(torch.int32)
            self.forward_extend_metadata = AiterMLAMetadata(
                qo_indptr, kv_indptr, kv_indices, kv_last,
                max_seqlen_q=int(ext.max().item()) if n > 0 else 1)
        else:
            # Decode: 1 query per request (qo_indptr filled in forward_decode to
            # honour the actual per-request query length, e.g. spec decode).
            kv_indptr, kv_indices, kv_last = self._build_kv_paging(
                req_pool_indices[:bs], seq_lens[:bs], req_to_page)
            self.forward_decode_metadata = AiterMLAMetadata(
                qo_indptr=None, kv_indptr=kv_indptr, kv_indices=kv_indices,
                kv_last_page_lens=kv_last, max_seqlen_q=1)

    # ------------------------------------------------------------------
    # CUDA/HIP graph: not supported in v1 (eager only)
    # ------------------------------------------------------------------
    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        raise NotImplementedError(
            "AiterMLABackend v1 is eager-only; launch the server with "
            "--enforce-eager (HIP-graph capture is planned for v2)."
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    # ------------------------------------------------------------------
    # Forward (Task 4/5)
    # ------------------------------------------------------------------
    def _paged_kv_buffer(self, token_to_kv_pool, layer):
        """The MLA pool's latent buffer viewed as 4-D pages for the AITER kernel."""
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        return k_cache.view(-1, self.page_size, 1, self.kv_cache_dim)

    def forward_decode(self, q, k, v, layer, out_cache_loc, token_to_kv_pool,
                       bs, save_kv_cache=True, **kwargs):
        # KV is written by the model (aiter_mla is in _MLA_KERNEL_BACKENDS); the
        # backend only reads. Guard for completeness if ever called otherwise.
        if save_kv_cache and k is not None:
            token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)

        md = self.forward_decode_metadata
        q = q.view(-1, self.num_q_heads, self.kv_cache_dim)
        total_q = q.shape[0]
        q_len_per_req = total_q // bs if bs > 0 else 1
        qo_indptr = torch.arange(
            0, total_q + 1, q_len_per_req, dtype=torch.int32, device=q.device)

        o = torch.empty(total_q, self.num_q_heads, self.kv_lora_rank,
                        dtype=q.dtype, device=q.device)
        kops.mla_decode_fwd(
            q, self._paged_kv_buffer(token_to_kv_pool, layer), o,
            qo_indptr, md.kv_indptr, md.kv_indices, md.kv_last_page_lens,
            max_seqlen_q=q_len_per_req, page_size=self.page_size,
            sm_scale=layer.scaling,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(self, q, k, v, layer, out_cache_loc, token_to_kv_pool,
                       bs, save_kv_cache=True, **kwargs):
        raise NotImplementedError("Implemented in Task 5")


register_backend("aiter_mla", {AttentionArch.MLA}, AiterMLABackend)

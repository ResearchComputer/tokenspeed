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

Mirrors the NVIDIA ``FlashMLABackend`` integration on AMD CDNA3/CDNA4
(gfx942/gfx950):

- **Decode** runs in the *absorbed* MLA space via AITER ``mla_decode_fwd`` over
  the paged latent KV pool.
- **Prefill** runs as full multi-head attention over the *decompressed* q/k/v
  via AITER ``flash_attn_varlen_func`` (the model's chunked-prefill core merges
  cached-prefix chunks with ``merge_state``).

``aiter_mla`` is registered in ``DeepseekV3AttentionMLA._MLA_KERNEL_BACKENDS`` so
the model writes the KV cache (``set_mla_kv_buffer``) and produces the absorbed
decode query; the backend therefore only reads the cache.

v1 constraints (docs/superpowers/specs/2026-06-09-aiter-mla-backend-design.md):
- eager only — CUDA/HIP-graph capture unsupported; run with ``--enforce-eager``.
- bf16 KV (no fp8); no speculative decode; single node.
- validated with ``--block-size 1`` (page_size == 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed_kernel.ops.attention import aiter_mla as kops

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class AiterMLADecodeMetadata:
    """Paged latent-KV metadata for an absorbed decode pass."""

    kv_indptr: torch.Tensor          # [bs+1] int32, cumsum of pages per request
    kv_indices: torch.Tensor         # [total_pages] int32, page ids per request
    kv_last_page_lens: torch.Tensor  # [bs]   int32, fill of each request's last page


@dataclass
class AiterChunkedPrefillMetadata:
    """Chunked-prefill metadata consumed by the model's forward_normal_chunked."""

    extend_seq_lens: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    max_extend_seq_len: int
    chunked_loop_num: int
    chunk_kv_indices_list: list
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list


class AiterMLABackend(AttentionBackend):
    """Eager MLA backend for AMD via AITER. No graph capture in v1."""

    def __init__(self, config: MLAConfig):
        super().__init__(config)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        self.num_local_heads = config.num_attention_heads // config.attn_tp_size
        self.num_q_heads = self.num_local_heads
        self.page_size = config.page_size

        self.forward_decode_metadata: AiterMLADecodeMetadata | None = None
        self.chunked_prefill_metadata: AiterChunkedPrefillMetadata | None = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _build_kv_paging(self, req_pool_indices, seq_lens, req_to_page):
        """Build AITER paged KV metadata from a block table + cache lengths.

        ``block_table = req_to_page[req_pool_indices]`` is ``[bs, max_pages]`` of
        physical page ids; we keep the first ``ceil(seq_len/page_size)`` pages of
        each row (row-major) to form the flat ``kv_indices`` whose per-request
        boundaries are ``kv_indptr``.
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
                              forward_mode, req_to_page=None,
                              extend_prefix_lens=None, **kwargs):
        if forward_mode.is_extend_or_mixed():
            n = num_extends
            extend_seq_lens = kwargs.pop("extend_seq_lens")
            extend_seq_lens_cpu = kwargs.pop("extend_seq_lens_cpu")
            extend_prefix_lens_cpu = kwargs.pop("extend_prefix_lens_cpu")
            device = seq_lens.device

            cum = torch.zeros(n + 1, dtype=torch.int32, device=device)
            torch.cumsum(extend_seq_lens, dim=0, out=cum[1:])
            max_ext = int(extend_seq_lens_cpu.max().item()) if n > 0 else 1
            (
                chunked_loop_num,
                chunk_kv_indices_list,
                chunked_seq_len,
                cu_chunked_seq_len,
                max_chunk_len_per_loop,
            ) = build_chunked_prefill_metadata_arrays(
                extend_prefix_lens,
                extend_prefix_lens_cpu,
                req_to_page,
                req_pool_indices[:n],
                self.page_size,
            )
            self.chunked_prefill_metadata = AiterChunkedPrefillMetadata(
                extend_seq_lens=extend_seq_lens,
                cum_extend_seq_lens=cum,
                max_extend_seq_len=max_ext,
                chunked_loop_num=chunked_loop_num,
                chunk_kv_indices_list=chunk_kv_indices_list,
                chunked_seq_len=chunked_seq_len,
                cu_chunked_seq_len=cu_chunked_seq_len,
                max_chunk_len_per_loop=max_chunk_len_per_loop,
            )
        if forward_mode.is_decode_or_idle() or forward_mode.is_mixed():
            kv_indptr, kv_indices, kv_last = self._build_kv_paging(
                req_pool_indices[:bs], seq_lens[:bs], req_to_page)
            self.forward_decode_metadata = AiterMLADecodeMetadata(
                kv_indptr=kv_indptr, kv_indices=kv_indices,
                kv_last_page_lens=kv_last)

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
    # Forward
    # ------------------------------------------------------------------
    def forward_decode(self, q, k, v, layer, out_cache_loc, token_to_kv_pool,
                       bs, save_kv_cache=True, **kwargs):
        # KV is written by the model (aiter_mla is in _MLA_KERNEL_BACKENDS); the
        # backend only reads. Guard kept for completeness.
        if save_kv_cache and k is not None:
            token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)

        md = self.forward_decode_metadata
        q = q.view(-1, self.num_q_heads, self.kv_cache_dim)
        total_q = q.shape[0]
        q_len_per_req = total_q // bs if bs > 0 else 1
        qo_indptr = torch.arange(
            0, total_q + 1, q_len_per_req, dtype=torch.int32, device=q.device)

        # MLA latent pool [N, 1, kv_cache_dim] viewed as 4-D pages for the kernel.
        kv_buffer = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, self.page_size, 1, self.kv_cache_dim)
        o = torch.empty(total_q, self.num_q_heads, self.kv_lora_rank,
                        dtype=q.dtype, device=q.device)
        kops.mla_decode_fwd(
            q, kv_buffer, o, qo_indptr, md.kv_indptr, md.kv_indices,
            md.kv_last_page_lens, max_seqlen_q=q_len_per_req,
            page_size=self.page_size, sm_scale=layer.scaling,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend_chunked(self, q, k, v, scaling, logits_soft_cap=None, *,
                               cum_seq_lens_q, cum_seq_lens_kv, max_q_len,
                               max_kv_len, seq_lens, batch_size, causal,
                               out=None):
        """Full-MHA ragged attention over decompressed q/k/v (prefill).

        Returns ``(output, lse)``; ``out`` (when given) receives the output. The
        lse is consumed by the model's ``merge_state`` only when cached-prefix
        chunks exist (disabled in v1 via ``--no-enable-prefix-caching``).
        """
        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        result = kops.flash_attn_varlen_func(
            q.view(-1, self.num_local_heads, head_dim),
            k.view(-1, self.num_local_heads, head_dim).to(q.dtype),
            v.view(-1, self.num_local_heads, self.v_head_dim).to(q.dtype),
            cum_seq_lens_q,
            cum_seq_lens_kv,
            max_q_len,
            max_kv_len,
            softmax_scale=scaling,
            causal=causal,
            return_lse=True,
            out=out,
        )
        output, lse = (result[0], result[1]) if isinstance(result, tuple) else (result, None)
        if out is not None and output is not out:
            out.copy_(output.view(out.shape))
            output = out
        return output, lse

    def forward_extend(self, q, k, v, layer, out_cache_loc, token_to_kv_pool,
                       bs, save_kv_cache=True, **kwargs):
        # DeepSeek MLA drives prefill through forward_extend_chunked, not this
        # path. (Plain ragged extend / spec-decode verify is a v2 concern.)
        raise NotImplementedError(
            "AiterMLABackend prefill goes through forward_extend_chunked; "
            "the plain forward_extend path is not implemented in v1."
        )


register_backend("aiter_mla", {AttentionArch.MLA}, AiterMLABackend)

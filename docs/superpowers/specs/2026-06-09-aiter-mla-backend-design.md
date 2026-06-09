# AITER MLA attention backend (`aiter_mla`) — design

**Date:** 2026-06-09
**Status:** approved (design)
**Target hardware:** AMD Instinct MI300A (gfx942 / CDNA3) on CSCS beverin; also applicable to gfx950/MI350.

## Goal

Add an AMD MLA (Multi-head Latent Attention) backend so tokenspeed can serve
DeepSeek-V3 / Kimi-style MLA models on AMD GPUs. Today every MLA backend
(`trtllm_mla`, `flashmla`, `tokenspeed_mla`, `deepseek_v4`) is registered only
`if platform.is_nvidia`, so on AMD the attention registry holds MHA-only
backends and any MLA model fails with `No backend supports arch MLA`.

The backend wraps **ROCm AITER** (`github.com/ROCm/aiter`, `aiter/mla.py`) —
the AMD FlashMLA equivalent. AITER's MLA decode (prebuilt gfx942 ASM) and
prefill (composable-kernel) kernels have been validated to build against
torch 2.11+rocm7.2 and run correctly on MI300A (AITER's own `op_tests/test_mla.py`
passed its allclose checks; a direct `mla_decode_fwd` call ran).

**v1 scope:** eager decode + prefill (no HIP-graph capture), bf16, single node.
First success metric: a correct end-to-end serve of a small MLA model
(DeepSeek-V2-Lite) on one MI300A node.

## Non-goals (v1)

- HIP/CUDA-graph capture (run with `--enforce-eager`); add in v2.
- fp8 KV cache (bf16 first).
- Speculative / MTP draft paths.
- Multi-node.
- Full Kimi-K2.6 serve (still blocked by MoE-at-384-expert-scale + ~2 TB / multi-node memory; this backend unblocks the *attention* piece only).

## Components (changes)

Respecting the AGENTS.md kernel boundary (third-party kernels live under
`tokenspeed-kernel`, wrapped in `ops/`; runtime imports only the kernel package).

1. **Kernel op wrapper** — `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/aiter_mla/`
   (`__init__.py` + impl) exposing thin `mla_decode_fwd` / `mla_prefill_fwd`
   (and prefill-ps if needed) that import the third-party `aiter` package. Add
   `aiter` to `tokenspeed-kernel/python/requirements/rocm-thirdparty.txt`.
   (AITER is built from source against the image's torch/ROCm; pinned commit
   `26aecec` for now.)
2. **Backend** — `python/tokenspeed/runtime/layers/attention/backends/aiter_mla.py`:
   `AiterMLABackend(AttentionBackend)`, constructed from `MLAConfig`, mirroring
   `FlashMLABackend`:
   - `__init__`: dims (`kv_lora_rank`, `qk_rope_head_dim`, `v_head_dim`,
     `kv_cache_dim = kv_lora_rank + qk_rope_head_dim`, head counts), `page_size`,
     pre-allocated indptr/indices scratch.
   - `init_forward_metadata(...)`: build AITER paging metadata (decode and
     extend modes).
   - `forward_decode(q,k,v,layer,...)`: reshape q → `[T, H, kv_cache_dim]`, view
     KV pool buffer as pages, call `kops.mla_decode_fwd(...)`, return
     `[T, H*v_head_dim]`.
   - `forward_extend(...)`: normal extend (no prefix) vs absorbed extend
     (prefix present), mapping to AITER's prefill entry points.
   - cuda-graph hooks: guard so v1 uses the eager path (raise a clear error if
     graph capture is attempted; `get_cuda_graph_seq_len_fill_value` → 1).
   - module-level `register_backend("aiter_mla", {AttentionArch.MLA}, AiterMLABackend)`.
3. **Registration / gating**
   - `backends/__init__.py`: add `if platform.is_amd: _try_import_optional_backend("aiter_mla", "aiter")`.
   - `registry._get_default_backend_name`: `elif platform.is_amd: return "aiter_mla"` for `AttentionArch.MLA`.
4. **Model wiring** — add `"aiter_mla"` to `_MLA_KERNEL_BACKENDS` in
   `runtime/models/deepseek_v3.py` so the model writes KV itself
   (`set_mla_kv_buffer`) and passes `save_kv_cache=False` (backend only reads).
5. **Model registry** — alias `DeepseekV2ForCausalLM` → `DeepseekV3ForCausalLM`
   (DeepSeek-V2-Lite's HF arch isn't registered; its MLA attention path is
   identical, only dim values differ).

## Core problem: paging-metadata conversion

`FlashMLABackend` passes `block_table (bs, max_pages)` + `cache_seqlens` to its
kernel. AITER's `mla_decode_fwd` / `mla_prefill_fwd` use flashinfer-style ragged
paging: `qo_indptr`, `kv_indptr`, `kv_indices`, `kv_last_page_lens`,
`max_seqlen_q`, `page_size`. The backend builds these from
`req_to_page[req_pool_indices]` + `seq_lens` (decode) / extend lens (prefill).

**Page size** must agree between the KV pool (`server_args.block_size`, the MLA
pool buffer is `[size + page_size, 1, kv_cache_dim]` per layer) and what AITER's
kernels accept (decode default `page_size=1`). v1: pin `block_size` to a value
AITER supports and build `kv_indices` accordingly; validate empirically.

## Data flow (decode)

model computes absorbed `Q (T,H,kv_lora_rank+qk_rope)` and latent
`K (T,1,kv_lora_rank+qk_rope)` → writes K to the MLA pool via `set_mla_kv_buffer`
→ `attn_mqa` (`PagedAttention`) → `PagedAttention.forward` →
`AiterMLABackend.forward_decode` → `kops.mla_decode_fwd(q, kv_view, o, indptrs…, sm_scale=layer.scaling)`
→ `o (T,H,v_head_dim=kv_lora_rank)` → reshape `[T, H*v_head_dim]`.

## Validation ladder

1. **Op parity** — standalone script comparing `AiterMLABackend` decode + extend
   output against a reference (torch MLA or AITER's golden) on a tiny config, run
   in the `tokenspeed-rocm-aiter` image on an MI300A node.
2. **End-to-end serve** — DeepSeek-V2-Lite (after the arch alias), one MI300A
   node, `--enforce-eager`, `--attention-backend aiter_mla` (or AMD default).
   Curl a completion; short correctness sanity (e.g. a few gsm8k items).
   Reuses the validated build/serve loop and `~/.edf/tokenspeed-rocm-aiter.toml`.

**Dev loop:** edit locally → `rcc`/rsync to `/capstor/.../tokenspeed-amd` →
editable-install or `PYTHONPATH` override in the aiter image (no 33 GB rebuild
per iteration) → serve.

## Dependencies / risks

- **AMD MoE backend** must work for the V2-Lite e2e serve (V2-Lite has a 64-expert
  MoE). Verify the AMD MoE path (triton/unquantized) early; it's required for
  end-to-end even though it's orthogonal to MLA attention.
- Metadata conversion correctness (block_table ↔ indptr/indices) — main bug surface.
- page_size alignment (pool ↔ AITER kernel).
- prefill path mapping: `mla_prefill_fwd` (paged) vs `mla_prefill_ps_fwd`
  (explicit K/V + reduction); absorbed vs normal extend.
- AITER API drift — pin commit `26aecec`.
- git is broken inside the ROCm container images (libcurl-gnutls/nghttp2 symbol
  clash); clone AITER before the torch wheel and/or `LD_PRELOAD` the system
  libnghttp2 (already solved in the image build).

## Test / artifacts already in place

- Image: `/capstor/.../tokenspeed-beverin/images/tokenspeed-rocm-aiter.x86_64.sqsh` (33 GB), EDF `~/.edf/tokenspeed-rocm-aiter.toml`.
- Build/serve sbatches under `/capstor/.../tokenspeed-beverin/jobs/`.

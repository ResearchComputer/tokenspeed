# AITER MLA kernel API — locked convention (commit 26aecec, MI300A/gfx942)

Captured from `aiter/mla.py` signatures + `op_tests/test_mla.py` on the cluster
(job 379495). This drives the `aiter_mla` backend code (plan Tasks 2/4/5).

## Decode — `aiter.mla.mla_decode_fwd`

```
mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
               kv_last_page_lens, max_seqlen_q, page_size=1, nhead_kv=1,
               sm_scale=None, logit_cap=0.0, num_kv_splits=None, ...)
```

- `q`: `[total_q, nhead, kv_lora_rank + qk_rope_head_dim]` (absorbed query; `total_q = sum(qo_lens)`, =bs for plain decode).
- `kv_buffer` (decode): **4-D `[num_page, page_size, nhead_kv=1, kv_lora_rank + qk_rope_head_dim]`** (bf16). `mla.py:186` does `_, _, _, qk_head_dim = kv_buffer.shape`. tokenspeed's MLA pool buffer is 3-D `(size+page_size, 1, kv_cache_dim)`; **view it as `(-1, page_size, 1, kv_cache_dim)`** before the call (exactly what FlashMLA does: `k_cache.view(-1, PAGE_SIZE, 1, kv_cache_dim)`). With `page_size=1` → `[total_tokens, 1, 1, Dq]`.
- `o`: `[total_q, nhead, kv_lora_rank]` (output is the latent/value dim = kv_lora_rank; the model up-projects afterward).
- `qo_indptr`: `[bs+1]` int32, cumsum of query lengths (plain decode: `arange(bs+1)`).
- `kv_indptr`: `[bs+1]` int32, cumsum of **pages per request** = `ceil(seq_len/page_size)`.
- `kv_indices`: int32, flat concat of page ids per request (from `req_to_page[req_pool_indices]`). With `page_size=1`, page id == token cache location.
- `kv_last_page_lens`: `[bs]` int32 = `((seq_len-1) % page_size) + 1` (all 1s when page_size=1).
- `sm_scale`: `layer.scaling`. `num_kv_splits=None` → auto.

## Prefill (absorbed, paged) — `aiter.mla.mla_prefill_fwd`

```
mla_prefill_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
                kv_last_page_lens, max_seqlen_q, sm_scale=None, ...)
```
Same metadata/shapes as decode; `qo_indptr` = cumsum of extend lengths, `max_seqlen_q` = max extend len. Causal is implied (no `is_causal` arg). The reference (`torch_mla_extend`) confirms absorbed MLA: scores over the full 576-d, value = latent[:kv_lora_rank], causal `tril(diagonal=s_k-s_q)`.

## Other entry points (not needed for v1)

- `mla_prefill_ps_fwd(Q, K, V, output, qo_indptr, kv_indptr, kv_page_indices, work_indptr, work_info_set, max_seqlen_q, is_causal, ...)` — explicit (un-absorbed) K/V prefill + persistent-kernel work metadata. More complex; only needed if the absorbed prefill path proves insufficient.
- `aiter.flash_attn_varlen_func(q,k,v,qo_indptr,kv_indptr,max_q,max_kv,softmax_scale,causal)` — the "normal" (non-absorbed) prefill the test uses for MHA reference.

## v1 mapping decision (simplification)

Both decode and extend run the **paged absorbed** AITER kernels on the latent KV
pool, sharing one metadata builder (`qo_indptr/kv_indptr/kv_indices/kv_last_page_lens`).
This avoids FlashMLA's normal/absorbed/ragged split entirely.

- decode → `mla_decode_fwd`, extend → `mla_prefill_fwd`.
- The metadata builder is general over `page_size`; **start validation with `--block-size 1`** (page_size=1 → `kv_indices` are per-token locations, `kv_last_page_lens` all 1) to minimize risk, and `--no-enable-prefix-caching` so every extend is a fresh full prefill (`s_q==s_k`) until prefix/chunked is validated.
- KV is written by the model (`set_mla_kv_buffer`) since `aiter_mla` will be in `_MLA_KERNEL_BACKENDS`; backend reads only (`save_kv_cache=False`).

## fp8 / dtype

bf16 q + bf16 kv supported (our v1). fp8 paths exist (`q_scale`/`kv_scale`) — out of scope for v1.

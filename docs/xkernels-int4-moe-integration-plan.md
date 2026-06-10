# INT4 W4A16 fused-MoE GEMM: xkernels integration plan

Status: **analysis + opt-in registration wired; not selected by default.** The
existing in-tree kernel (`ops/moe/triton.py::fused_moe_kernel`, `use_int4_w4a16`
branch, selected as `triton_moe_fused_experts`) is unchanged and remains the
production path. This document is the plan for adopting the **autotuned**
xkernels variant.

Provenance of the xkernels source: vendored under
`tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/xkernels/`
(ResearchComputer/kernels @ `696740a7c39dd643596a04efe2b9f9e72458d2f2`).

---

## 1. Why

The in-tree INT4 path is numerically validated (`test/aiter/test_w4a16_moe_kernel_amd.py`)
but runs with `get_default_config` (see `triton_config.py:160` —
`"Using default MoE kernel config. Performance might be sub-optimal!"`): a single
hardcoded tile (`BLOCK_M=64,N=64,K=32,GROUP_M=8`, or the small-M
`16/32/64/1`). The xkernels kernel
(`thirdparty/xkernels/ops/moe/triton/moe_int4_kernel.py::fused_moe_int4_kernel`)
is the **autotuned** version of the same math, with a CDNA3-reasoned config space
(`thirdparty/xkernels/ops/moe/triton/configs.py`) keyed on `(N, K, EM,
num_valid_tokens)` and exposing the AMD lowering knobs (`waves_per_eu`,
`matrix_instr_nonkdim`, `kpack`).

## 2. The math/layout contracts MATCH

Both kernels implement compressed-tensors `pack-quantized` W4A16, and the
weight/scale layouts are **identical**:

| Contract | in-tree (`fused_moe_kernel`) | xkernels (`fused_moe_int4_kernel`) | match |
|---|---|---|---|
| Packed B | `[E, N, K//8]` int32, 8 `uint4b8`/int32, low nibble = lowest K | same | ✅ |
| Dequant | `(uint4b8 - 8) * scale[g]` | `(uint4b8 - 8) * scale[g]` | ✅ |
| Scale | `[E, N, K//group]` bf16, `group_size=32` | same | ✅ |
| A gather | `a_ptr + offs_token//top_k * stride_am` | same | ✅ |
| Routing | `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded` | same | ✅ |
| EP filter | `expert_ids == -1` → write zeros (`filter_expert`) | `FILTER_EXPERT` + `-1` → write zeros | ✅ |
| Output | token-indexed `c[stride_cm*offs_token]` | token-indexed `c[stride_cm*offs_token]` | ✅ |
| `mul_routed_weight` | fold routing weight (down GEMM) | `MUL_ROUTED_WEIGHT` | ✅ |
| `tl.dot` rhs dtype | dequant in fp32 → `.to(a.dtype)` before dot | dequant in fp32 → `.to(a.dtype)` before dot | ✅ |

So tokenspeed's two-GEMM flow (`triton_common.py::triton_forward`: gate_up GEMM
with `top_k=spec.top_k`, `mul_routed_weight=False`; SiLU; down GEMM with
`top_k=1`, `mul_routed_weight=True`; `moe_combine` reduce) maps directly onto the
xkernels **launcher** `int4_w4a16_moe_gemm(...)`, which takes exactly
`(a, b_packed, b_scale, c, topk_weights, sorted_token_ids, expert_ids,
num_tokens_post_padded, top_k, group_size, mul_routed_weight, compute_type,
filter_expert)`.

### Deltas (none blocking)

1. **Unpack micro-strategy.** In-tree loads one int32 per *logical* K and shifts
   by `(k%8)*4`; xkernels loads one int32 per *8* K (`BLOCK_K_PACK = BLOCK_K//8`)
   and broadcasts an 8-wide shift LUT (4x fewer weight bytes, unpack amortized
   over 8 MACs). Same numeric result; xkernels requires `BLOCK_SIZE_K % 8 == 0`
   and `BLOCK_SIZE_K % group_k == 0` (its `prune_configs` enforces this; its
   config space uses K ∈ {64,128,256}, all valid).
2. **Config source.** In-tree: `config` dict passed in from
   `try_get_optimal_moe_config` (one config, applied to BOTH gate_up and down).
   xkernels: `@triton.autotune` picks per shape; the caller-supplied
   `BLOCK_SIZE_M`-driven `moe_align_block_size` block size must agree with the
   autotuned `BLOCK_SIZE_M` — see §4.1 (the one real integration subtlety).
3. **API surface.** xkernels also ships a high-level `[M,N]`-in/out op
   (`fused_moe_int4_w4a16` / `_moe_int4_w4a16_triton`) that builds its own align
   + `view(M,top_k,N).sum(1)` reduce. That is the wrong seam for tokenspeed (we
   already have dispatch + combine kernels); we wire the **launcher**, not the op.

## 3. The two-triton-package handling (the #1 runtime breakage)

The serving process imports **stock** `triton`; tokenspeed kernels are built
against **`tokenspeed_triton`** (`tokenspeed_kernel._triton`). `tl.dot` asserts
both operands share the *same* dtype object, so a stock-triton dtype reaching it
is rejected ("Unsupported rhs dtype"). Handling, already in place:

- The vendored Triton backends are imported through
  `thirdparty/xkernels/_triton_compat.py::triton_import_ctx()`, which aliases
  `triton[.x.y]` → `tokenspeed_triton[.x.y]` during import (see the edited
  `thirdparty/xkernels/ops/moe/__init__.py`). So `fused_moe_int4_kernel`'s
  module-level `import triton.language as tl` binds **tokenspeed_triton**, and
  `@triton.autotune`/`@triton.heuristics`/`@triton.jit` are the tokenspeed_triton
  decorators.
- `compute_type` must likewise be a tokenspeed_triton dtype. The tokenspeed-side
  wrapper (`ops/moe/xkernels.py`) builds it from `tokenspeed_kernel._triton`'s
  `tl` (NOT stock triton), and the kernel itself already does the safe
  `b_deq.to(a.dtype)` before `tl.dot` (same trick as the in-tree kernel,
  documented at `ops/moe/triton.py:704`).
- The AMD knob constexprs (`waves_per_eu`, `matrix_instr_nonkdim`, `kpack`) in
  the xkernels `Config.kwargs` are accepted as (unused) kernel args under stock
  Triton and read by the AMD backend under the tokenspeed_triton ROCm fork — no
  code change needed, they're declared as kernel params.

## 4. Where it registers / how the backend would select it

### 4.1 Wiring (done, opt-in)

`ops/moe/xkernels.py` adapts the xkernels launcher to the **exact**
`invoke_fused_moe_kernel` call signature used by `triton_common.py::triton_forward`
and registers it as:

```
register_kernel("moe", "experts", name="xkernels_moe_int4_w4a16",
                features={"dispatch_sorted"}, solution="triton",
                capability=CapabilityRequirement(vendors=frozenset({"amd"})),
                signatures=format_signatures("x","dense",{bf16}),
                priority=Priority.PERFORMANT + 1)   # below in-tree (+2)
```

Because `build_triton_gemms` pins `expected_kernel_name="triton_moe_fused_experts"`,
the selector returns the in-tree kernel unless a caller explicitly asks for
`expected_kernel_name="xkernels_moe_int4_w4a16"`. The registration is therefore
inert for production until opted in. The wrapper:

- ignores `A_scale`/`a_use_tma`/`b_use_tma`/`c_sorted`/`bias` (W4A16 has no
  activation scale, no TMA on AMD, writes token-indexed unsorted C, no bias);
- asserts `use_int4_w4a16 and not (use_fp8_w8a8 or use_int8_w8a16)`;
- derives `group_size` from `block_shape[1]` (the backend passes
  `block_shape=(0, group_size)`);
- **autotune-vs-align block-size subtlety:** the caller's
  `moe_align_block_size` uses `config["BLOCK_SIZE_M"]` to pad per-expert blocks,
  but xkernels autotunes its own `BLOCK_SIZE_M`. If the autotuned `BLOCK_SIZE_M`
  is **larger** than the align block size, `expert_ids[pid_m]` no longer maps a
  contiguous single-expert run and results are wrong. **Resolution:** constrain
  the autotune space (in the wrapper) so `BLOCK_SIZE_M == config["BLOCK_SIZE_M"]`
  (filter the config list to the caller's block_m), OR re-run
  `moe_align_block_size` at the autotuned block_m. The wrapper takes the first
  approach (pins `BLOCK_SIZE_M` to the dispatch block_m, autotunes only
  `N/K/warps/stages/waves/kpack`), which keeps the existing dispatch stage
  unchanged. This is the single most important correctness gate to verify
  on-device.

### 4.2 Eventual production switch (not done here)

Once validated + tuned on gfx942, flip `Wna16TritonBackend.process_weights_after_loading`
(`backends/wna16/triton.py:148`) `build_triton_gemms(...)` to request
`expected_kernel_name="xkernels_moe_int4_w4a16"`. That is a one-line change in
`triton_common.py::build_triton_gemms` (`_experts_common["expected_kernel_name"]`)
gated behind a server arg / env so it can be A/B'd. **Do not** change the
dispatch (`moe_align_block_size_amd`) or combine kernels — they are reused as-is.

## 5. On-device validation + tuning steps (human, MI300A / gfx942)

All op-tests are written to run on a single MI300A; **do not** submit SLURM jobs.

1. **Parity (added):**
   ```
   pytest tokenspeed-kernel/test/ops/test_moe_int4_xkernels.py -q
   ```
   Checks the xkernels launcher against (a) the in-tree `invoke_fused_moe_kernel`
   INT4 path and (b) the xkernels pure-torch reference, for the
   DeepSeek-V2-Lite / Kimi-K2 shapes (E, N, K, top_k, group=32).
2. **Autotune block-M gate:** confirm the wrapper pins `BLOCK_SIZE_M` to the
   dispatch block size (assert in the test); flip an env to let it free-tune and
   confirm parity FAILS (proves the gate is real), then re-pin.
3. **Microbench** gate_up and down GEMM shapes at decode (M=1·top_k…8·top_k) and
   prefill (M=512,2048) vs the in-tree default config; expect the win at
   small-M decode (weight-read bound) where the default tile is least tuned.
4. **End-to-end:** serve DeepSeek-V2-Lite (single GPU) and Kimi-K2 (2 nodes / 8
   GPUs, EP=8 — `filter_expert=True` path) with the env flag from §4.2 on;
   compare GSM8K / coherence + tokens/s against the in-tree path.
5. **Persist tuned configs** the same way the in-tree path loads them
   (`get_moe_configs` JSON keyed by `E, N, K, device_name`) if we later prefer a
   static config over JIT autotune at serve start.

## 6. Risks

- **Block-M mismatch (high if unhandled):** see §4.1 — mitigated by pinning.
- **Autotune at serve start:** first-call latency spike while the autotuner
  sweeps; acceptable for a warmup, but consider a frozen config for production.
- **`num_warps` counts 64-lane wavefronts on AMD:** the config space assumes
  CDNA3; on a non-gfx942 AMD part re-tune.
- **bf16-only:** wrapper asserts bf16 activations (the W4A16 production dtype);
  fp32 is interpreter-only.
```

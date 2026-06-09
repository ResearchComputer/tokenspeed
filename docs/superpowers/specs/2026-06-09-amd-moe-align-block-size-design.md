# AMD `moe_align_block_size` (Triton) — design

**Date:** 2026-06-09
**Status:** approved (design)
**Branch:** `MI300x`
**Target hardware:** AMD Instinct MI300A (gfx942 / CDNA3) on CSCS beverin.

## Goal

Make the unquantized **bf16 Triton MoE** path (`Bf16TritonBackend`) run on AMD
gfx942 so DeepSeek-style MoE models (e.g. DeepSeek-V2-Lite) serve end-to-end on
MI300A — together with the `aiter_mla` attention backend already on this branch.

## Background — the single blocking gap

A code-explorer trace of the full `triton_forward` call sequence
(`python/tokenspeed/runtime/layers/moe/backends/triton_common.py`) found that
every kernel except one is already AMD-portable:

| MoE op | registry key | AMD status |
|---|---|---|
| `moe_dispatch` → **`moe_align_block_size`** | `moe/dispatch` | ❌ the registered `triton_moe_align_block_size` is vendor-neutral but its body calls `_moe_align_block_size` imported from `tokenspeed_kernel.thirdparty.trtllm` (NVIDIA-only) → `error_fn` on AMD (`RuntimeError: Kernel implementation not found`) |
| `moe_experts` (fused gate-up / down GEMM, bf16) | `moe/experts` | ✅ pure Triton (`fused_moe_kernel`); TMA path off for bf16 non-TMA |
| `moe_combine` (`moe_sum_reduce`) | `moe/combine` | ✅ pure Triton / torch-compile, vendor-neutral |
| `silu_and_mul` | (activation) | ✅ AMD pure-torch fallback already added |

The gfx950 gluon MoE kernels (`ops/moe/gluon.py`) are CDNA4-only
(`min/max_arch_version = 9.5`) and quantized (MXFP4/FP8) — not reusable for the
bf16 CDNA3 path.

`moe_align_block_size` permutes tokens into expert-sorted, block-padded order:
count tokens per expert (histogram) → prefix sums → scatter token ids into
`sorted_token_ids` with per-expert block padding, producing
`(sorted_token_ids, expert_ids, num_tokens_post_padded)`. A pure-torch
reference with the exact semantics exists at
`tokenspeed-kernel/.../numerics/reference/moe.py:391` (the correctness oracle).

## Design

**Add one tokenspeed-owned pure-Triton `moe_align_block_size`** and wire it via
the kernel registry so AMD selects it while NVIDIA keeps the fast trtllm path.

1. **Triton kernel + wrapper** in `tokenspeed-kernel/.../ops/moe/triton.py`
   (tokenspeed's own Triton — lives in `ops/moe/`, not `thirdparty/`):
   implement the histogram → prefix-sum → scatter algorithm producing the same
   `(sorted_token_ids, expert_ids, num_tokens_post_padded)` as the reference.
   Register it for `moe/dispatch`, `solution="triton"`,
   `signatures=format_signatures("indices", "dense", {torch.int32})`,
   `traits={"comm_strategy": {"local"}}`, `tags={"portability"}`.
2. **Constrain the existing trtllm-backed registration to NVIDIA**
   (`capability=CapabilityRequirement(vendors=frozenset({"nvidia"}))`) so it is
   filtered out on AMD. NVIDIA keeps trtllm (it remains selectable and higher
   priority); AMD selects the new Triton kernel.
3. **Selection invariants:** on NVIDIA the trtllm kernel still wins (vendor +
   priority); on AMD only the Triton kernel is eligible. Verify both via the
   registry.

## Components / files

| File | Change |
|---|---|
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton.py` | add `@triton.jit` align kernel(s) + a `moe_align_block_size_triton` wrapper; register for `moe/dispatch`; add `vendors={"nvidia"}` capability to the existing trtllm-backed registration |
| `tokenspeed-kernel/test/ops/moe/test_moe_align_amd.py` (or extend existing) | op-parity vs `numerics/reference/moe.py` align reference |

## Data flow (unchanged contract)

`triton_forward` → `tokenspeed_kernel.moe_dispatch(topk_ids, block_size, num_experts, dtype=int32, expected_kernel_name="triton_moe_align_block_size")` → `select_kernel("moe","dispatch", ...)` → (AMD) the new Triton kernel → `(sorted_token_ids, expert_ids, num_tokens_post_padded)`.

## Validation

1. **Op-parity (MI300A):** new Triton `moe_align_block_size` vs the pure-torch
   reference, over a range of `(num_tokens, top_k, num_experts, block_size)`
   incl. uneven expert loads and empty experts. Run in the
   `tokenspeed-rocm-aiter` image via the existing cluster harness.
2. **End-to-end (MI300A):** re-run the DeepSeek-V2-Lite serve
   (`jobs/serve-dsv2lite.sbatch`: `--enforce-eager --block-size 1
   --no-enable-prefix-caching --disable-kvstore`, MLA default `aiter_mla`).
   Success = `server READY` + coherent completions (capital-of-France,
   arithmetic), confirming attention (`aiter_mla`) + bf16 Triton MoE end-to-end.

## Non-goals (v1)

- No MoE perf tuning / autotune configs for AMD (correctness first).
- No quantized (MXFP4/FP8) MoE on AMD (that's the gfx950 gluon path).
- No TMA path (bf16 non-TMA only).
- TMA-config check: confirm `try_get_optimal_moe_config` on AMD does not return
  a TMA config for `moe_experts` (explorer flagged as low-risk); address only if it does.

## Risks

- **Further runtime gaps after align:** static analysis says align is the only
  gap (all other kernels confirmed AMD-portable); if the e2e serve surfaces
  another missing kernel, it becomes the next small fix.
- **Triton kernel correctness:** the block-padding / `num_tokens_post_padded`
  semantics must exactly match the reference (the fused expert GEMM indexes by
  `sorted_token_ids`); the op-parity test guards this.

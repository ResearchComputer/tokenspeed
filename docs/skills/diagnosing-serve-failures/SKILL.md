---
name: diagnosing-serve-failures
description: >-
  Use when a tokenspeed serve crashes, hangs, or produces garbage. Provides a failure
  taxonomy keyed to the engine init stage (import → backend → KV pool → comm → warmup
  → forward), the exact signatures and fixes seen on AMD MI300A, and the working
  diagnostic tools (engine log, the offline Engine profiler, py-spy) vs the dead ends.
---

# Diagnosing a serve failure

## Step 0: find which stage died

Read the **engine log** (not just the job stdout). Locate the *last* INFO line and
the *first* ERROR/Traceback. The last successful line tells you the stage:

| Last good line | Stage reached | Look for the failure class below |
|---|---|---|
| `server_args=...` only | import/registration | §A |
| `Loading ... shards N/M` | weight load | §B |
| `Initialized ... KV pool` | comm/warmup next | §C/§D/§E |
| `Scheduler config: ...` | warmup forward next | §E |
| `*** READY ***` + garbage output | numerics | §F |

A serve that **OOM-kills** shows `slurmstepd: ... oom_kill ... Out Of Memory` in the
*job stdout* with little in the engine log — that's §C.

## §A — "Unknown attention backend: 'X'. Available: [...]"

Registration, not the kernel. Your backend module was **not imported for this
platform**. Fix the platform branch in
`layers/attention/backends/__init__.py` (e.g. add `deepseek_v4` to the `is_amd`
block). Distinguish from `No backend supports arch K` (= default-backend selection
has no candidate for that arch) and `Backend 'X' does not support arch K` (= registered
but for the wrong arch set).

## §B — weight load errors

- `AttributeError` in a fused-proj loader (e.g. reading `weight_block_size` off a
  `CompressedTensorsConfig`) → guard with `getattr(..., None)`. Fp8-only attrs don't
  exist on other quant configs.
- Garbage later but load is clean → not a load bug; keep going.

## §C — OOM at KV pool init (unified-memory APU)

`oom_kill` right after/within `Initialized ... KV pool`. On MI300A the node's ~512 GB
is **shared across 4 ranks**. Causes & fixes, in order:
1. `--gpu-memory-utilization` too high: 0.9 × 4 ranks overflows. Use **0.5**, lower if
   needed. (The KV pool `num_device_pages` scales directly with this.)
2. The pinned host **KVStore** pool is `kvstore_ratio × device_pool` per rank and
   `hipHostRegister`-pinned → 4× overflows. Use `--kvstore-ratio 0.1`.
3. Still tight → reduce `--max-model-len`.

## §D — comm / NCCL failures

- `Failed to initialize any NET plugin` at the first all-reduce → the EDF forces
  `NCCL_NET="AWS Libfabric"` but the aws-ofi-rccl plugin isn't on `LD_LIBRARY_PATH`.
  Add `.../aws-ofi-rccl/lib`. Needed even single-node.
- Multi-node warmup **hangs** (GPUs 0%, no error) → a sub-group collective deadlock.
  Known cause class: the symmetric-memory / triton all-reduce `can_run()` enters a
  rendezvous that diverges across AMD ranks. AMD must use NCCL all-reduce
  (`comm_backend` gating returns False on non-NVIDIA). A *flat* 8-rank all-reduce
  passing while the *serve* hangs ⇒ it's a sub-group communicator, not the transport.
- Diagnose hangs with **py-spy** dumping all ranks on a timer (raise
  `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` so the watchdog doesn't kill before capture).

## §E — warmup/forward: missing accelerator kernel (the common one)

A traceback ending in `error_fn`, `... is unavailable`, `Kernel implementation not
found`, `No backend supports arch`, or a `thirdparty/cuda` / `deep_gemm` import. This
is an NVIDIA-only op with no portable path **at the call site the forward just
reached**. Walk it:
1. Read the *bottom* frames — they name the layer and op (e.g.
   `deepseek_v4_mhc.py:mhc_pre → deep_gemm.tf32_hc_prenorm_gemm is unavailable`).
2. Classify per `serving-engine-bringup` (gated kernel / direct thirdparty call /
   registration) and fix per `tokenspeed-kernel-boundary`.
3. Re-serve; the **next** op in the chain surfaces. This is expected — bring-up is a
   chain (V4 chained: DSA indexer → o_proj fp8 → sparse-MLA compute → MHC prenorm GEMM).

Same bug class, recurring: a runtime caller importing `thirdparty.cuda.*` directly
(passes a spurious CC gate on gfx942 because ROCm reports `device.type == "cuda"`,
`major == 9`) and then crashing on the missing `.so`. Fix = route through the registry
op and gate `is_supported` on `is_nvidia`. (Historical: `lm_head_gemm`, `merge_state`.)

## §F — READY but garbage output

Numerics, not a crash. The classic AMD trap: a per-forward op that **returns** a
result the fused caller expects it to **write in place** (e.g. RoPE writing
`output_q_rope` / rotating `key` in place). Positions silently never encoded → garbage.

> **Debugging lesson: size-gate any per-forward diagnostic.** The first forwards are
> *warmup* (size 1-2), not the real request. Ungated prints fire on warmup and mask
> the real-shape bug. Gate diagnostics to real sizes.

## Tools: what works, what's a dead end

- **Works:** the engine log + tracebacks; the **offline Engine profiler**
  (`Engine.start_profile()/stop_profile()` with `TOKENSPEED_PROFILER_DIR`, run in an
  `if __name__=="__main__"` driver) — it runs `torch.profiler` *inside* the scheduler
  subprocess where the kernels are and exports a chrome trace on `stop`. py-spy for hangs.
- **Dead ends:** the HTTP `/start_profile` (proxied to the Rust gateway which has no
  such route → profiler never arms); external `rocprofv3` wrapping the 2-node serve
  (blows the gRPC warmup deadline; never finalizes its `.dat` on a killed process).

## Red flags

| Thought | Reality |
|---|---|
| "Unknown backend = missing kernel" | It's registration. Different stage, different fix. |
| "Nondeterministic hang" | Usually a deterministic crash on a code path only some requests hit (e.g. chunked-prefill `merge_state`). Provoke it; py-spy it. |
| "Garbage = kernel numerics bug" | Often a return-vs-write-in-place contract bug (RoPE). Check the caller's expectation. |
| "I'll profile over HTTP" | Dead end here. Use the offline Engine profiler. |
| "My warmup diagnostic shows it's fine" | Warmup is size 1-2, not the real request. Size-gate. |

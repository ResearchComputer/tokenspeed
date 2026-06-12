---
name: serving-engine-bringup
description: >-
  Use when bringing a new model, attention backend, or MoE backend online on a new
  accelerator (e.g. AMD gfx942) in tokenspeed — or when a model "loads but won't
  serve". Encodes the incremental blocker-chain methodology, the staged-init mental
  model, and the correctness-first discipline that makes bring-up converge instead
  of thrash.
---

# Bringing a model/backend online (the blocker-chain method)

## The core idea

A new model on a new accelerator does **not** fail in one place. It fails as a
**chain**: you fix one missing capability, the forward advances, and the *next*
missing capability surfaces. The job is to walk the chain **one blocker at a time,
in execution order**, and never guess at a blocker you have not yet reached.

> The single most expensive mistake is theorizing the whole chain up front from a
> *load-only* probe. A model that imports and loads tells you almost nothing about
> the forward. **Run the forward to discover the real chain.** (This repo's own V4
> history has two corrections where a load-only assessment named the wrong "sole
> blocker".)

## The staged-init mental model

Every serve walks these stages in order. Knowing which stage you died in tells you
the *class* of fix before you read a single line of the traceback:

1. **Import / registration** — module import side effects register backends/kernels.
   Failure here = a `register_*` call gated off for your platform, or an import that
   raises. Symptom: `Unknown attention backend: 'X'. Available: [...]` (your backend
   is absent from the list) or an ImportError at startup.
2. **Weight load** — sharded safetensors → device. Slow but usually
   platform-agnostic. Failure = dtype/quant-method mismatch, or a fused-proj loader
   reading an attribute the quant config does not have.
3. **KV pool / scheduler init** — allocates the paged cache. Failure = **OOM**
   (see below) or a cache-layout assertion.
4. **Communicator setup** — process groups, all-reduce backend. Failure = NCCL/RCCL
   transport (`Failed to initialize any NET plugin`, sub-group deadlocks).
5. **Warmup forward** — the first real forward. **This is where most accelerator
   gaps surface**: a kernel bound only on `is_nvidia`, a `thirdparty.cuda`/`deep_gemm`
   op with no portable path, a layout the asm kernel reinterprets differently.
6. **Decode forward** — only reached once prefill works. Decode often has its *own*
   gaps (paged-cache gather, graph capture) that prefill never exercises.

A serve that prints `*** READY ***` has passed 1–5 for the warmup shapes. Getting a
**coherent completion** is the real bar (warmup can pass on garbage).

## The loop

```
pick the next blocker (the current traceback) ──► find the call site ──►
classify (config? registration? missing kernel?) ──► apply the smallest fix ──►
re-serve ──► read the NEW failure ──► repeat
```

Validate prefill before decode. The serve recipe already splits them:
`max_tokens=1` exercises **prefill only**; `max_tokens=48` exercises **decode**.
Use `max_tokens=1` as the prefill gate, then add decode.

## Classifying a warmup/forward blocker

When the forward dies on a missing capability, it is almost always one of:

- **A kernel gated `if platform.is_nvidia` / `is_hopper_plus`** that falls through to
  `error_fn` or `raise`. Fix: provide a portable (Triton/torch) path behind the same
  op name. See `tokenspeed-kernel-boundary`.
- **A direct `thirdparty.cuda.*` or `deep_gemm.*` call** in runtime code (an
  AGENTS.md boundary violation that only bites off-NVIDIA). Fix: route through the
  kernel registry op, or add a bf16/fp32 fallback at the call site (mirror the
  existing o_proj fp8 fallback in `layers/dense/fp8.py`).
- **A registration gated off** for your platform in
  `layers/attention/backends/__init__.py`. Fix: add your backend to the platform
  branch (this is what makes `deepseek_v4` reachable on AMD).
- **A config/memory knob** (not code at all): OOM, NCCL plugin path, an env var.
  See `diagnosing-serve-failures` and `beverin-dev-loop`.

## Correctness-first, then performance

- **First** get a coherent completion with the simplest correct path. It is fine to
  materialize a bf16 workspace, gather with torch, or run eager. The repo's whole AMD
  stack started this way (eager, block-size 1, dequant-on-load) and was *then*
  optimized (HIP graphs, in-kernel INT4, sync-free kernels) — see
  `decode-throughput-optimization`.
- **Reuse a validated path before writing a new one.** If prefill already builds a
  bf16 latent workspace and calls a validated compute op, make decode build the same
  workspace rather than inventing a paged fast-path. Optimize after it is correct.
- **Pin every layout empirically.** Before writing a cache adapter, read the *writer*
  (e.g. `kv_cache/deepseek_v4.py::_move_fp8_ds_mla_rows`) and confirm the byte layout
  — do not infer it from a sibling view that the asm kernel reinterprets internally.

## Red flags (stop and reconsider)

| Thought | Reality |
|---|---|
| "I'll map the whole blocker chain from the load log" | Load ≠ forward. Run the forward. |
| "The backend isn't registered, so the kernel is missing" | Registration and the kernel are *different* stages. Check which one. |
| "It reached READY, so it works" | READY ≠ coherent. Check an actual completion. |
| "Prefill works, decode will too" | Decode has its own gather/graph gaps. Test `max_tokens=48`. |
| "I'll write the fast paged path now" | Correctness-first. Reuse the prefill path; optimize later. |
| "This OOM means the model is too big" | On unified-memory APUs it usually means `--gpu-memory-utilization` × ranks. Tune first. |

## Where things live (V4 / DeepSeek family, this repo)

- Model forward: `python/tokenspeed/runtime/models/deepseek_v4.py`
- Attention backend: `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`
- Backend registration: `.../attention/backends/__init__.py` (platform branches)
- KV cache pool + layout: `.../attention/kv_cache/deepseek_v4.py`
- Kernel boundary for the NVIDIA-only ops: `tokenspeed-kernel/.../ops/attention/...`
  (`flash_mla/`, `triton/deepseek_v4.py`)
- The portable AMD kernels are pulled from `xkernels` (pinned in
  `tokenspeed-kernel/python/requirements/common.txt`).

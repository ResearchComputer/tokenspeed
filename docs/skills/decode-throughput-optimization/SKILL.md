---
name: decode-throughput-optimization
description: >-
  Use when decode tokens/sec is low and you want to improve it (especially on AMD
  MI300A). Encodes the host-bound mental model, the measured decode kernel-time
  breakdown method (the offline profiler), the levers that actually moved the needle
  here (HIP-graph capture, sync-free kernels, tuned/in-kernel quant GEMM), and the
  things that look like levers but are not (torch.compile, hierarchical all-reduce).
---

# Optimizing decode throughput

## The mental model: decode is host-bound, not GPU-bound

At batch 1, single-stream decode on this stack is **death-by-a-thousand-small-kernels**
(~2300 kernels/token, ~50%+ of GPU time in tiny elementwise/pointwise ops). The GPUs
idle waiting on the CPU to launch the next kernel. Two consequences:

- **Per-step fixed overhead (launches + collectives) dominates.** Batching amortizes
  it: aggregate decode throughput scales ~linearly with batch (bs1≈12 → bs32≈167 tok/s
  on 2×MI300A for Kimi) even though per-stream rate degrades.
- **The biggest single-stream lever is removing launch overhead, not making any one
  kernel faster.** That is what HIP graphs do.

## The measured progression (single-stream, Kimi-K2.6, 2×MI300A)

`4.3 → 5.88 → 12.25 tok/s`, by:
1. **Sync-free kernels** (4.3→5.88): a per-forward kernel doing a device→host
   `.item()` per expert (`moe_align_block_size`) stalls ~thousands of times/token.
   Rewrote vectorized/sync-free (scatter_add histogram + cumsum + searchsorted). This
   also *unblocks graph capture* (fixed shapes, no host sync).
2. **HIP-graph decode capture** (5.88→12.25, ~2×): the single biggest lever. Capture
   removes per-token launch overhead. Requires the decode metadata to be **graph-static**
   (persistent buffers, sync-free index build). Mirror `backends/flashmla.py`'s
   `init_cuda_graph_state` / capture / replay.

## How to find the real bottleneck: the offline profiler breakdown

Do **not** guess. Measure with the offline Engine profiler (the HTTP one is a dead end
— see `diagnosing-serve-failures`):

```python
# in an `if __name__ == "__main__":` driver (Engine spawns the scheduler via mp 'spawn')
engine = Engine(**server_args); warmup(); engine.start_profile()
generate(...); engine.stop_profile()   # exports .trace.json.gz from inside the scheduler
```
Set `TOKENSPEED_PROFILER_DIR`. Then aggregate `cat=="kernel"` events by name. The
measured DSV2-Lite decode breakdown that ranked the work:
- elementwise/pointwise swarm ~37–53% (RoPE/residual/casts) — needs forward-fusion,
  a feature, not a flag.
- `moe_align` (argsort + cumsum + fills, miscategorized across Activation/Other) **~28%**
  — the single biggest *addressable* sink; a graph-safe Triton `moe_align` removes it.
- dense bf16 GEMM ~20–28% — misses MFMA/hipBLASLt on torch2.11+rocm7.2.
- INT4 MoE GEMM ~13–18% — needs tuned configs for the serving shapes.

## Levers, ranked

1. **HIP-graph capture** (if not already on) — removes launch overhead. Biggest win.
   Cap the capture size (`--max-cudagraph-capture-size`, validated ceiling 16; 32 is
   memory-fragile). `NCCL_GRAPH_MIXING_SUPPORT=0` is required to capture >8 buckets
   (default 1 lazily `hipMalloc`s during capture → `invalid device pointer`).
2. **Sync-free, fixed-shape kernels on the hot path** — any `.item()`/data-dependent
   shape blocks capture and stalls eager. Make the op graph-safe.
3. **Graph-safe Triton `moe_align`** — replaces the torch argsort/cumsum/fills (~28%).
4. **Tuned / in-kernel quant GEMM** — tuned INT4 W4A16 configs for the serving shapes;
   in-kernel INT4 (weights stay packed, ~4× footprint saving vs dequant-on-load).
5. **Dense bf16 GEMM** — try `TORCH_BLAS_PREFER_HIPBLASLT=0` (routes through rocBLAS;
   the bf16 path misses MFMA/hipBLASLt and is far slower than fp16 untuned).
6. **Comms layout** — at batch 1 the cross-node all-reduces/token are latency-bound;
   a DP-attention layout (`--attn-tp-size N --data-parallel-size M`) moves attn
   all-reduces onto intra-node xGMI. (Blocked here on capturability — verify before
   committing.)

## Things that look like levers but are NOT (verified)

- **torch.compile** — a **net pessimization** (~−19% tok/s) on AMD eager decode:
  dynamo guard/recompile CPU overhead on the launch-bound critical path, and inductor
  *de-optimizes* (disables online softmax). The model forward is **not** broadly
  torch.compiled here; production sets `TORCH_COMPILE_DISABLE=1` and should stay there.
- **Hierarchical all-reduce** — does not beat RCCL's flat collective on 2-node/4-NIC
  MI300A. Keep the flat collective (and keep graphs).
- **"Reduce kernel launch overhead"** as a separate project — HIP graphs already did
  that; it was the 4.3→12.25 win. Post-graphs, the lever is *fusing the pointwise
  swarm* (a forward feature) or the specific GEMM/align kernels above.

## Discipline

- **Measure before and after every change**, with the offline profiler, on real
  shapes. Eager inflates kernel *count* (graphs remove launch, not per-kernel cost),
  so an eager A/B can show "flat tok/s" while the change is a big win *under graphs* —
  judge by GPU-time/dispatch counts, not just eager tok/s.
- Note any silent cap (top-N, sampling). Don't let an untuned-config warning
  ("Using default MoE kernel config…") pass unexamined — it's often double-digit %.

## Red flags

| Thought | Reality |
|---|---|
| "Make the hot kernel faster" | At bs1 it's launch-bound. Remove launches (graphs) first. |
| "torch.compile will fuse it" | Net slower on AMD eager decode. Keep it disabled. |
| "Eager A/B shows no change, drop it" | Graphs change the verdict. Compare GPU-time/dispatches. |
| "I'll add `.item()` for a quick check" | It blocks capture and stalls eager. Keep the hot path sync-free. |
| "Hierarchical all-reduce will cut comms" | It doesn't beat flat RCCL here. Measure before adopting. |

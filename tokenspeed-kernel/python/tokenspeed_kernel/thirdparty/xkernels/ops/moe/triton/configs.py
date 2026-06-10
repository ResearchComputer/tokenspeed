# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Autotune config space for the INT4 W4A16 fused-MoE GEMM, reasoned for CDNA3.

Target: AMD MI300A (gfx942, CDNA3). Notes on the architecture that drive the
choices below:

* Wavefront = 64 lanes. ``num_warps`` here counts 64-lane wavefronts; a CU has
  4 SIMD32 units, and the MFMA pipe wants enough wavefronts to hide the
  global-load latency of the packed-int4 weights.
* MFMA: CDNA3 has ``v_mfma_*_16x16x*`` and ``32x32x*`` instructions. We pin
  ``matrix_instr_nonkdim=16`` (16x16) because the decode regime has tiny M
  (1 token x top_k), so a 32x32 MFMA would waste most of its M lanes; 16x16
  packs the work better and keeps register pressure low.
* LDS = 64 KB/CU. With double/triple buffering the A and dequantized-B tiles
  must fit; large BLOCK_N x BLOCK_K bf16 B tiles + 2-3 stages can overflow LDS,
  so big tiles are paired with ``num_stages<=2``.
* ``waves_per_eu`` (occupancy hint): higher values let more wavefronts hide
  memory latency, but cost VGPRs. Small-M tiles have spare VGPRs (M is tiny) so
  they can afford ``waves_per_eu=2-4``; large prefill tiles set it to 0/1 so the
  big accumulator does not spill.
* ``kpack=2`` packs two K elements per VGPR for the MFMA feed on CDNA3,
  improving the ds_read/MFMA ratio — relevant since our effective K tile is wide
  after unpack.

Memory-bandwidth framing (decode). At M=1xtop_k the GEMM is *weight-read bound*:
each expert weight is read once and barely reused (M ~ 1-8 rows). Packed int4 is
~4x smaller than bf16, so the design goal is to (a) keep the weight read one
coalesced int32 load per 8 K, (b) minimize per-element dequant overhead, and
(c) pick BLOCK_N large enough to amortize the per-tile scale fetch and launch
overhead while keeping BLOCK_M tiny. BLOCK_K is a multiple of both the pack
factor (8) and the group size (32) so unpack and scale-broadcast are clean.
"""

from __future__ import annotations

import triton

__all__ = ["get_autotune_configs", "prune_configs"]


def _cfg(bm, bn, bk, gm, *, num_warps, num_stages, waves_per_eu, kpack=2, nonkdim=16):
    """Build a Triton ``Config``.

    AMD-only lowering knobs (``waves_per_eu``, ``matrix_instr_nonkdim``,
    ``kpack``) are passed through the kwargs dict; the Triton AMD backend reads
    them and they are ignored on other backends, so the same config list is
    portable. ``matrix_instr_nonkdim=16`` selects the 16x16 MFMA.
    """
    return triton.Config(
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gm,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": nonkdim,
            "kpack": kpack,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )


def get_autotune_configs():
    """Return the candidate configs for the INT4 W4A16 fused-MoE GEMM on gfx942.

    The space spans three regimes:

    * **Decode (tiny M)** — ``BLOCK_M in {16, 32}``: minimal M, wide-ish N to
      amortize scale + launch overhead, ``waves_per_eu`` high to hide the int4
      weight read, modest ``num_warps`` (2-4) since the M dim is too small to
      feed 8 wavefronts. ``GROUP_SIZE_M=1`` (no L2 super-grouping needed — at
      tiny M there is little weight reuse to exploit).
    * **Light prefill (M ~ 64-256)** — balanced square-ish tiles.
    * **Heavy prefill (M >= 512)** — large ``BLOCK_M``/``BLOCK_N``, more warps,
      ``waves_per_eu`` low to avoid VGPR spills on the big accumulator,
      ``GROUP_SIZE_M`` larger for L2 reuse across the now-shared weights.

    ``BLOCK_SIZE_K`` candidates are multiples of 32 (group size) and 8 (pack):
    64/128/256. 256 maximizes weight-read coalescing (32 int32 per row) but
    needs ``num_stages<=2`` to fit LDS; 64 keeps register/LDS pressure low for
    the smallest tiles.
    """
    configs = [
        # ---- decode / tiny-M: weight-read bound, maximize occupancy ----
        _cfg(16, 64, 128, 1, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(16, 64, 256, 1, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(16, 128, 128, 1, num_warps=4, num_stages=2, waves_per_eu=3),
        _cfg(16, 128, 256, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(16, 256, 128, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(32, 64, 128, 1, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(32, 128, 128, 1, num_warps=4, num_stages=2, waves_per_eu=3),
        _cfg(32, 128, 256, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(32, 256, 128, 1, num_warps=8, num_stages=2, waves_per_eu=2),
        # ---- light prefill: balanced tiles ----
        _cfg(64, 64, 128, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(64, 128, 128, 8, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(64, 128, 64, 8, num_warps=4, num_stages=3, waves_per_eu=2),
        _cfg(64, 256, 64, 8, num_warps=8, num_stages=2, waves_per_eu=1),
        _cfg(128, 128, 64, 8, num_warps=8, num_stages=2, waves_per_eu=1),
        # ---- heavy prefill: large tiles, low occupancy to avoid VGPR spill ----
        _cfg(128, 256, 64, 8, num_warps=8, num_stages=2, waves_per_eu=0),
        _cfg(256, 128, 64, 8, num_warps=8, num_stages=2, waves_per_eu=0),
        _cfg(128, 128, 128, 4, num_warps=8, num_stages=2, waves_per_eu=1),
    ]
    return configs


def prune_configs(configs, named_args, **kwargs):
    """Drop configs that cannot run for the given problem before benchmarking.

    Removes configs whose ``BLOCK_SIZE_K`` is not a multiple of the quant group
    size (the scale-broadcast reshape requires ``BLOCK_K % group_k == 0``) and
    over-large tiles for tiny problems (e.g. ``BLOCK_N > N``). Cuts autotune time
    and avoids compiling configs that would fail the constexpr asserts.
    """
    group_k = named_args.get("group_k", 32)
    N = named_args.get("N")
    K = named_args.get("K")
    pruned = []
    for c in configs:
        bk = c.kwargs["BLOCK_SIZE_K"]
        bn = c.kwargs["BLOCK_SIZE_N"]
        if bk % group_k != 0:
            continue
        if bk % 8 != 0:  # pack factor
            continue
        if K is not None and bk > max(K, group_k):
            continue
        if N is not None and bn > 2 * N:  # allow one over-tile for masking
            continue
        pruned.append(c)
    return pruned or list(configs)

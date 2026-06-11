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

"""Production AMD ``moe/dispatch`` backend: the vendored xkernels Triton
``moe_align_block_size`` in its sync-free / fixed-shape mode (kernels#18).

This replaces the in-tree torch ``amd_moe_align_block_size`` (argsort + cumsum +
scatter_add + buffer fills) as the selected AMD dispatch builder. The torch
implementation is already sync-free, but it spends ~28% of decode GPU time across
an ``argsort`` plus a swarm of elementwise kernels; the Triton kernel does the
same work in five small launches (histogram + padded prefix-sum + scatter), which
measured **−28.5% total decode GPU time, bit-identical output** in the offline
DSV2-Lite A/B (kernels#18 / issue #18).

**Graph-capturable.** Called with ``truncate=False``, the kernel performs no
``.item()`` device→host sync and returns a fixed-shape ``expert_ids`` of length
``max_blocks = cdiv(max_pad, block_size)`` (unused trailing blocks = sentinel 0),
matching the in-tree contract that the fused-MoE block GEMM consumes (it reads
only ``num_tokens_post_padded // block_size`` blocks). Both buffers are
fixed-shape per batch bucket, so the op captures into a HIP graph unchanged.

**Selection (opt-in).** Registered at ``Priority.PERFORMANT + 3`` (one band above
the in-tree ``amd_moe_align_block_size`` at ``+2``) so it wins auto-selection on
AMD **only when ``TS_XKERNELS_ALIGN=1``**; unset/``0`` falls back to the in-tree
torch kernel. Kept opt-in (not default-ON) because, being a Triton kernel, it
JIT-compiles per prefill shape — under a multi-rank serve that requires a
**node-local, execable** ``TRITON_CACHE_DIR`` (e.g. ``/tmp``; *not* the shared
``~/.triton``, which races to ``OSError: Errno 116`` on cold compile, nor noexec
``/dev/shm``). Validated correct + graph-stable on the 2-node Kimi serve; flip to
default-ON once a clean decode-tok/s win is measured and the cache-dir setting is
baked into the serve recipe.

**EP safety.** This serves only the ``comm_strategy="local"`` dispatch (the path
``moe.backends.triton_common.triton_forward`` takes). Under expert parallelism
that path remaps non-local tokens to **local expert 0 with weight 0** *before*
dispatch and passes the **local** expert count, so ``topk_ids`` is always in
``[0, num_local_experts - 1]`` — the histogram column index ``e`` never reaches
``num_experts`` (which would overflow the ``[num_experts+1, num_experts]``
``tokens_cnts`` buffer). The DeepEP dispatch uses a different op
(``deepep_moe_scatter``) and is unaffected.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Importing this kernel triggers the xkernels moe package __init__, which binds
# the Triton backend to ``tokenspeed_triton`` via xkernels' _triton_compat hook.
from xkernels.ops.moe.triton.align_kernel import (
    moe_align_block_size_triton as _xk_align,
)

__all__ = ["xkernels_moe_align_block_size"]

# Opt-in: register only when ``TS_XKERNELS_ALIGN=1`` (see module docstring for
# why this is not default-ON yet — multi-rank serves need a node-local execable
# TRITON_CACHE_DIR, and the decode-tok/s win is not yet cleanly measured e2e).
if os.environ.get("TS_XKERNELS_ALIGN") == "1":

    @register_kernel(
        "moe",
        "dispatch",
        name="xkernels_moe_align_block_size",
        solution="triton",
        signatures=format_signatures("indices", "dense", {torch.int32}),
        traits={"comm_strategy": frozenset({"local"})},
        # One band ABOVE the in-tree amd_moe_align_block_size (PERFORMANT+2) so it
        # is auto-selected on AMD. Graph-capturable (truncate=False, no sync).
        priority=Priority.PERFORMANT + 3,
        tags={"portability"},
        capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    )
    def xkernels_moe_align_block_size(
        topk_ids: torch.Tensor, block_size: int, num_experts: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """vLLM/SGLang-style Triton histogram+prefix-sum+scatter align.

        Same ``(sorted_token_ids, expert_ids, num_tokens_post_padded)`` contract
        as the in-tree ``amd_moe_align_block_size``: ``pad_id = topk_ids.numel()``,
        ``expert_ids`` is the full ``cdiv(max_pad, block_size)`` length with unused
        trailing blocks = sentinel 0 (``truncate=False``, no device→host sync), so
        it is HIP-graph-capturable.

        Args:
            topk_ids: ``[M, top_k]`` int32 expert id per token-slot (values in
                ``[0, num_experts - 1]``; EP remaps non-local tokens before here).
            block_size: GEMM block size each expert run is padded up to.
            num_experts: number of (local) experts.

        Returns:
            ``(sorted_token_ids [max_pad], expert_ids [max_blocks],
            num_tokens_post_padded [1])``.
        """
        return _xk_align(topk_ids, block_size, num_experts, truncate=False)

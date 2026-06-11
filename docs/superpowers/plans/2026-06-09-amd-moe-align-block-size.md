# AMD moe_align_block_size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the bf16 Triton MoE path run on AMD gfx942 by providing an AMD-selectable `moe_align_block_size`, so DeepSeek-V2-Lite serves end-to-end on MI300A (with the `aiter_mla` attention backend already on this branch).

**Architecture:** The registered `moe/dispatch` kernel `triton_moe_align_block_size` delegates to `thirdparty.trtllm` (NVIDIA-only → `error_fn` on AMD). Add a second `moe/dispatch` registration — a correct **torch** implementation gated to `vendors={"amd"}` — and constrain the trtllm one to `vendors={"nvidia"}`. The kernel registry then selects trtllm on NVIDIA and the torch impl on AMD. (Refinement vs the spec's "pure-Triton": align operates on the small routing tensor, not the GEMMs, and the spec mandates correctness-first/no perf-tuning, so v1 is torch, validated against the proven reference; a Triton perf kernel is a follow-up.)

**Tech Stack:** PyTorch 2.11+rocm7.2, tokenspeed-kernel registry (`register_kernel` + `CapabilityRequirement`), ROCm AITER image on CSCS beverin (MI300A / gfx942).

---

## Conventions

- Branch: `MI300x`. Local repo: `/home/xiayao/Documents/projects/researchcomputer/swissai/tokenspeed-amd`. Cluster code: `/capstor/store/cscs/swissai/infra02/xyao/code/tokenspeed-amd`.
- **No local AMD GPU** — run all tests on MI300A via the harness `jobs/test-aiter.sbatch` (image `tokenspeed-rocm-aiter`, sets `PYTHONPATH=$CODE/python:$CODE/tokenspeed-kernel/python`, `HSA_NO_SCRATCH_RECLAIM=1`). After each edit: `rsync` (command below), write `jobs/_test_cmd.sh`, `sbatch jobs/test-aiter.sbatch`, poll the log.
- Sync: `rsync -avh --delete --exclude '.git/' --exclude 'docs/node_modules/' --exclude '**/__pycache__/' -e ssh /home/xiayao/Documents/projects/researchcomputer/swissai/tokenspeed-amd/ beverin:/capstor/store/cscs/swissai/infra02/xyao/code/tokenspeed-amd/`
- Commits: `-s` sign-off + `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File structure

| File | Responsibility |
|---|---|
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton.py` | add `CapabilityRequirement` import; add `vendors={"nvidia"}` to the existing trtllm-backed `triton_moe_align_block_size` registration; add `amd_moe_align_block_size` (torch impl) registered for `moe/dispatch` with `vendors={"amd"}` |
| `tokenspeed-kernel/test/ops/moe/test_moe_align_amd.py` | op-parity of the AMD kernel vs the torch reference (canonicalized) + registry-selection assertion on AMD |

## Reference facts (verified)

- Existing dispatch wrapper: `moe_align_block_size(topk_ids, block_size, num_experts) -> (sorted_ids, expert_ids, num_tokens_post_pad)` at `ops/moe/triton.py:1057`; registered `name="triton_moe_align_block_size", solution="triton", signatures=format_signatures("indices","dense",{torch.int32}), traits={"comm_strategy": frozenset({"local"})}, priority=Priority.PERFORMANT+2, tags={"portability"}` at line 1045. Body buffers: `max_num_tokens_padded = topk_ids.numel() + (num_experts+1)*(block_size-1)`, `max_num_m_blocks = ceil(max_num_tokens_padded/block_size)`, pad id = `topk_ids.numel()`.
- Reference oracle: `torch_moe_align_block_size(topk_ids, block_size, num_experts)` (`numerics/reference/moe.py:391`) returns a canonical packed tensor via `canonicalize_align_block_size`.
- `canonicalize_align_block_size(sorted_ids, expert_ids, num_tokens_post_pad, block_size)` (`numerics/moe.py:117`) requires `sorted_ids.numel() == expert_ids.numel()*block_size`; sorts within each block (within-block order is non-deterministic) then packs `[num_tokens_post_pad, expert_ids, blocks_sorted]`.
- `compute_align_block_size_buffer_dims(pad_id, num_experts, block_size)` → `(num_blocks, num_blocks*block_size)`; `num_blocks` equals the wrapper's `max_num_m_blocks` (same formula, `pad_id == numel`).
- `register_kernel(..., capability=CapabilityRequirement(vendors=frozenset({...})))` is the gating mechanism (see `ops/moe/gluon.py:4383` for the pattern). `CapabilityRequirement` lives in `tokenspeed_kernel.platform`.
- `expected_kernel_name` in `select_kernel` is a debug-only warning (`selection.py:685`), not a hard filter — a differently-named AMD kernel is selected fine.

---

## Task 1: AMD torch `moe_align_block_size` + registry gating

**Files:**
- Modify: `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton.py`
- Test: `tokenspeed-kernel/test/ops/moe/test_moe_align_amd.py`

- [ ] **Step 1: Write the failing op-parity + selection test.** Create `tokenspeed-kernel/test/ops/moe/test_moe_align_amd.py`:

```python
# Copyright (c) 2026 LightSeek Foundation  (license header as in sibling tests)
"""AMD moe_align_block_size parity vs the torch reference (MI300A only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU (MI300A) required"
)

from tokenspeed_kernel.numerics.moe import canonicalize_align_block_size  # noqa: E402
from tokenspeed_kernel.numerics.reference.moe import (  # noqa: E402
    torch_moe_align_block_size,
)
from tokenspeed_kernel.ops.moe.triton import moe_align_block_size_amd  # noqa: E402

CASES = [
    (128, 6, 64, 16),   # DeepSeek-V2-Lite-ish: 64 experts, top-6
    (1, 8, 64, 16),     # single token (decode)
    (37, 2, 8, 16),     # uneven, small
    (256, 8, 64, 32),   # larger block
    (64, 1, 4, 16),     # few experts
]


@pytest.mark.parametrize("m, top_k, num_experts, block_size", CASES)
def test_amd_align_matches_reference(m, top_k, num_experts, block_size):
    torch.manual_seed(0)
    topk_ids = torch.randint(
        0, num_experts, (m, top_k), device="cuda", dtype=torch.int32
    )
    s, e, n = moe_align_block_size_amd(topk_ids, block_size, num_experts)

    # Pad sorted_ids to expert_ids.numel()*block_size for canonicalization.
    n_blocks = e.numel()
    pad_id = topk_ids.numel()
    padded = torch.full(
        (n_blocks * block_size,), pad_id, dtype=torch.int32, device=s.device
    )
    padded[: s.numel()] = s
    mine = canonicalize_align_block_size(padded, e, n, block_size)
    ref = torch_moe_align_block_size(topk_ids, block_size, num_experts)
    assert torch.equal(mine, ref), f"mismatch for {(m, top_k, num_experts, block_size)}"


def test_amd_align_selected_on_amd():
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_amd:
        pytest.skip("AMD only")
    import torch as _t

    from tokenspeed_kernel.ops.moe import moe_dispatch

    topk_ids = _t.randint(0, 64, (128, 6), device="cuda", dtype=_t.int32)
    # Must not raise "Kernel implementation not found": AMD selects the torch impl.
    s, e, n = moe_dispatch(
        topk_ids, 16, 64, dtype=_t.int32,
        expected_kernel_name="triton_moe_align_block_size",
    )
    assert s.dtype == _t.int32 and n.item() > 0
```

- [ ] **Step 2: Run — verify it fails.** Sync, then via harness `TS_TEST_CMD`-equivalent (`jobs/_test_cmd.sh` = `python -m pytest tokenspeed-kernel/test/ops/moe/test_moe_align_amd.py -v`):
Expected: FAIL — `ImportError: cannot import name 'moe_align_block_size_amd'`.

- [ ] **Step 3: Add the `CapabilityRequirement` import** to `ops/moe/triton.py` (near the other `tokenspeed_kernel` imports, ~line 35):

```python
from tokenspeed_kernel.platform import CapabilityRequirement
```

- [ ] **Step 4: Constrain the existing trtllm-backed registration to NVIDIA.** In the `@register_kernel(...)` decorator for `triton_moe_align_block_size` (`ops/moe/triton.py:1045`), add a `capability` argument:

```python
@register_kernel(
    "moe",
    "dispatch",
    name="triton_moe_align_block_size",
    solution="triton",
    signatures=format_signatures("indices", "dense", {torch.int32}),
    traits={
        "comm_strategy": frozenset({"local"}),
    },
    priority=Priority.PERFORMANT + 2,
    tags={"portability"},
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
)
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...  # unchanged body (calls trtllm _moe_align_block_size)
```

- [ ] **Step 5: Add the AMD torch implementation** (place after the `moe_align_block_size` function in `ops/moe/triton.py`). It returns the same `(sorted_ids, expert_ids, num_tokens_post_pad)` contract/buffer dims as the trtllm wrapper, filling the expert-sorted, block-padded layout with `pad_id = topk_ids.numel()`:

```python
@register_kernel(
    "moe",
    "dispatch",
    name="amd_moe_align_block_size",
    solution="triton",
    signatures=format_signatures("indices", "dense", {torch.int32}),
    traits={
        "comm_strategy": frozenset({"local"}),
    },
    priority=Priority.PERFORMANT + 2,
    tags={"portability"},
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
)
def moe_align_block_size_amd(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AMD torch implementation of moe_align_block_size.

    Produces the same (sorted_token_ids, expert_ids, num_tokens_post_padded)
    layout as the trtllm kernel: token-k flat ids (0..numel-1) grouped by expert,
    each expert's run padded up to a multiple of block_size with pad_id; unused
    tail = pad_id. Operates on the small routing tensor (correctness-first; a
    Triton perf kernel is a follow-up). Filtered/EP tokens (expert == num_experts)
    are dropped (treated as padding) — single-node bf16 path has none.
    """
    device = topk_ids.device
    total = topk_ids.numel()
    pad_id = total
    max_num_tokens_padded = total + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    sorted_ids = torch.full(
        (max_num_tokens_padded,), pad_id, dtype=torch.int32, device=device
    )
    expert_ids = torch.zeros((max_num_m_blocks,), dtype=torch.int32, device=device)

    flat_expert = topk_ids.flatten().to(torch.int64)
    flat_token = torch.arange(total, device=device, dtype=torch.int32)
    # Group token ids by expert (stable so within-expert order is deterministic;
    # canonicalization sorts within blocks anyway).
    order = torch.argsort(flat_expert, stable=True)
    tokens_by_expert = flat_token[order]
    counts = torch.bincount(flat_expert, minlength=num_experts + 1)

    write_pos = 0
    block_idx = 0
    offset = 0
    for e in range(num_experts):  # filtered slot (e == num_experts) is dropped
        n = int(counts[e].item())
        grp = tokens_by_expert[offset : offset + n]
        n_blocks = (n + block_size - 1) // block_size
        for b in range(n_blocks):
            cnt = min(block_size, n - b * block_size)
            start = write_pos + b * block_size
            sorted_ids[start : start + cnt] = grp[b * block_size : b * block_size + cnt]
            expert_ids[block_idx + b] = e
        write_pos += n_blocks * block_size
        block_idx += n_blocks
        offset += n

    num_tokens_post_pad = torch.tensor([write_pos], dtype=torch.int32, device=device)
    return sorted_ids, expert_ids, num_tokens_post_pad
```

- [ ] **Step 6: Run — verify it passes.** Sync, run via harness:
`python -m pytest tokenspeed-kernel/test/ops/moe/test_moe_align_amd.py -v`
Expected: all parametrized parity cases + `test_amd_align_selected_on_amd` PASS on MI300A.

- [ ] **Step 7: Commit**

```bash
git add tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton.py \
        tokenspeed-kernel/test/ops/moe/test_moe_align_amd.py
git commit -s -m "feat(kernel): AMD moe_align_block_size (torch); gate trtllm to NVIDIA

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: End-to-end DeepSeek-V2-Lite serve on MI300A

**Files:** reuse `/capstor/.../tokenspeed-beverin/jobs/serve-dsv2lite.sbatch` (already configured: `--enforce-eager --block-size 1 --no-enable-prefix-caching --disable-kvstore`, MLA default `aiter_mla`).

- [ ] **Step 1: Sync and submit the serve job.**
`rsync ...` then `ssh beverin 'cd <deploy> && sbatch jobs/serve-dsv2lite.sbatch'`. Poll the log.

- [ ] **Step 2: Verify readiness + completions.** Expected: `server READY`, then coherent completions for the two prompts (capital-of-France → "Paris"; `17 + 25` → "42"). This confirms attention (`aiter_mla`) + bf16 Triton MoE (with the new align kernel) end-to-end on MI300A.

- [ ] **Step 3: If a different missing-kernel error appears** (low probability — static trace says align was the only gap): capture it, and treat it as the next small fix (e.g. a TMA-config that hard-codes `device="cuda"` in `moe_experts`; if so, ensure `try_get_optimal_moe_config` returns a non-TMA config on AMD). Re-run.

- [ ] **Step 4: Record the result** in `docs/superpowers/specs/aiter-mla-api-notes.md` (serve command + sample output) and commit.

```bash
git add docs/superpowers/specs/aiter-mla-api-notes.md
git commit -s -m "docs: DeepSeek-V2-Lite end-to-end serve on MI300A (aiter_mla + Triton MoE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** new AMD `moe_align_block_size` (Task 1, Step 5); trtllm gated to NVIDIA (Step 4); registry selection invariant (Step 1 `test_amd_align_selected_on_amd`); op-parity vs reference (Step 1 parity cases); e2e DeepSeek-V2-Lite serve (Task 2). All spec sections mapped.
- **Refinement flagged:** spec said "pure-Triton"; plan uses torch for v1 (correctness-first; align is not the bottleneck) — surfaced in the header and Step 5 docstring; Triton perf kernel is an explicit follow-up.
- **Placeholder scan:** none — full kernel + test code provided.
- **Consistency:** `moe_align_block_size_amd` name + `(sorted_ids, expert_ids, num_tokens_post_pad)` contract + `pad_id = topk_ids.numel()` used consistently in impl and test; `canonicalize_align_block_size` sizing (`expert_ids.numel()*block_size`) handled by padding in the test.
- **EP note:** filtered tokens (expert == num_experts) dropped; valid for the single-node bf16 path (test uses `topk_ids ∈ [0, num_experts)`), matching the reference when `counts[num_experts] == 0`.

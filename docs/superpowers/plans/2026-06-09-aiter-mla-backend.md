# AITER MLA Attention Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `aiter_mla` attention backend that wraps ROCm AITER's MLA kernels so tokenspeed can serve DeepSeek-V3/Kimi-style MLA models on AMD MI300A (gfx942), validated end-to-end on DeepSeek-V2-Lite.

**Architecture:** A kernel-package op wrapper exposes AITER's `mla_decode_fwd`/`mla_prefill_fwd`; a runtime `AiterMLABackend(AttentionBackend)` mirrors `FlashMLABackend` (eager decode+prefill, no graph capture in v1), converting tokenspeed's paged KV metadata to AITER's flashinfer-style `indptr/indices` form; registration is gated on `platform.is_amd`.

**Tech Stack:** Python, PyTorch 2.11+rocm7.2, ROCm AITER (commit `26aecec`), tokenspeed runtime, enroot/pyxis on CSCS beverin (MI300A, gfx942).

---

## Conventions for this plan

- **Repo (local):** `/home/xiayao/Documents/projects/researchcomputer/swissai/tokenspeed-amd`, branch `feat/aiter-mla-backend`.
- **Cluster code dir:** `/capstor/store/cscs/swissai/infra02/xyao/code/tokenspeed-amd` (kept in sync via `rsync`).
- **Image/EDF:** `tokenspeed-rocm-aiter.x86_64.sqsh` + `~/.edf/tokenspeed-rocm-aiter.toml` (already built; AITER installed at `/opt/aiter`).
- **No local AMD GPU** — all tests run on an MI300A node. Use the test harness from Task 0. After every local edit: `rsync` then run via the harness.
- **Sync command** (run locally):
  ```bash
  rsync -avh --delete --exclude '.git/' --exclude 'docs/node_modules/' --exclude '**/__pycache__/' \
    -e ssh /home/xiayao/Documents/projects/researchcomputer/swissai/tokenspeed-amd/ \
    beverin:/capstor/store/cscs/swissai/infra02/xyao/code/tokenspeed-amd/
  ```
- **PYTHONPATH override:** the image has tokenspeed installed in site-packages; the harness prepends the synced source so edits are live:
  `export PYTHONPATH=$CODE/python:$CODE/tokenspeed-kernel/python:$PYTHONPATH` (CODE = cluster code dir).
- **Commits:** sign off (`-s`) and end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File structure

| File | Responsibility |
|---|---|
| `tokenspeed-kernel/python/requirements/rocm-thirdparty.txt` | add `aiter` dependency (built from source in the image) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/aiter_mla/__init__.py` | thin op wrapper: `mla_decode_fwd`, `mla_prefill_fwd` re-exported from third-party `aiter` |
| `python/tokenspeed/runtime/layers/attention/backends/aiter_mla.py` | `AiterMLABackend(AttentionBackend)` — decode/extend + metadata conversion + cuda-graph guard + `register_backend` |
| `python/tokenspeed/runtime/layers/attention/backends/__init__.py:49-58` | add `if platform.is_amd:` import of `aiter_mla` |
| `python/tokenspeed/runtime/layers/attention/registry.py:107-116` | `_get_default_backend_name(MLA)` → `aiter_mla` on AMD |
| `python/tokenspeed/runtime/models/deepseek_v3.py:450` | add `"aiter_mla"` to `_MLA_KERNEL_BACKENDS` |
| `python/tokenspeed/runtime/models/registry.py` | alias `DeepseekV2ForCausalLM` → `DeepseekV3ForCausalLM` |
| `test/aiter/test_aiter_mla_op.py` | numerical parity: op wrapper vs torch-reference MLA decode |
| `meta/beverin/jobs/test-aiter.sbatch` (cluster) | reusable harness: run pytest/scripts in the aiter image on MI300A |

---

## Task 0: Cluster test harness + lock AITER's exact MLA API

**Files:**
- Create (cluster): `/capstor/.../tokenspeed-beverin/jobs/test-aiter.sbatch`
- Create (local, notes): `docs/superpowers/specs/aiter-mla-api-notes.md`

- [ ] **Step 1: Write the test-harness sbatch** (on the cluster) that runs an arbitrary command in the aiter image on an MI300A node with the synced source on `PYTHONPATH`.

```bash
cat > /capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/jobs/test-aiter.sbatch <<'SB'
#!/bin/bash
#SBATCH -J ts-aiter-test
#SBATCH -p mi300
#SBATCH -A a-infra02
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --output=/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/logs/aiter-test-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/logs/aiter-test-%j.out
set -uo pipefail
CMD="${TS_TEST_CMD:?set TS_TEST_CMD}"
srun --environment=tokenspeed-rocm-aiter bash -c '
  set -uo pipefail
  CODE=/capstor/store/cscs/swissai/infra02/xyao/code/tokenspeed-amd
  export PYTHONPATH=$CODE/python:$CODE/tokenspeed-kernel/python:${PYTHONPATH:-}
  export HSA_NO_SCRATCH_RECLAIM=1
  cd $CODE
  echo "RUN: '"$CMD"'"; '"$CMD"'
'
echo "TEST_JOB_DONE $?"
SB
```
Invoke it with: `ssh beverin "cd <deploy> && TS_TEST_CMD='<cmd>' sbatch --export=ALL,TS_TEST_CMD jobs/test-aiter.sbatch"`.

- [ ] **Step 2: Lock the exact AITER API.** Submit a harness job that prints the real signatures + the calling convention used by AITER's own passing test (pinned commit `26aecec`):

Run (via harness `TS_TEST_CMD`):
```
python -c "import inspect,aiter.mla as m; print(inspect.signature(m.mla_decode_fwd)); print(inspect.signature(m.mla_prefill_fwd))" && sed -n '1,200p' /opt/aiter/op_tests/test_mla.py
```
Expected: prints both signatures and the test's construction of `q/kv_buffer/o/qo_indptr/kv_indptr/kv_indices/kv_last_page_lens/page_size/sm_scale`.

- [ ] **Step 3: Record the convention** in `docs/superpowers/specs/aiter-mla-api-notes.md`: exact arg order, tensor shapes/dtypes, whether `o` is written in place, how `kv_indptr/kv_indices/kv_last_page_lens` index the paged buffer, the supported `page_size`(s), and the decode-vs-prefill differences. This file is the authority for Tasks 2/4/5.

- [ ] **Step 4: Commit**
```bash
git add docs/superpowers/specs/aiter-mla-api-notes.md
git commit -s -m "docs: lock AITER MLA kernel calling convention (commit 26aecec)"
```

---

## Task 1: Verify the AMD MoE path serves (dependency de-risk)

DeepSeek-V2-Lite has a 64-expert MoE; the e2e serve needs a working AMD MoE backend. Confirm this independently of MLA before investing in the backend.

**Files:** none (cluster smoke only).

- [ ] **Step 1: Serve a tiny AMD MoE model** with the existing (MHA) path to confirm the AMD MoE backend works. Use `Qwen/Qwen3-30B-A3B`? Too big. Use a small MoE that uses MHA (not MLA) so MLA isn't required — e.g. `Qwen/Qwen1.5-MoE-A2.7B` (MHA + 60-expert MoE, ~14 GB bf16, fits one node).

Run (via harness): a serve smoke (reuse `smoke.sbatch` pattern, model `Qwen/Qwen1.5-MoE-A2.7B`, `--enforce-eager --tensor-parallel-size 1 --max-model-len 4096`), then curl `/v1/chat/completions`.
Expected: server READY + a coherent completion → AMD MoE path works.

- [ ] **Step 2:** If it fails, STOP and resolve the MoE backend before continuing (record findings); MLA attention alone won't yield an e2e serve. If it passes, proceed.

(No commit — this is a runtime verification.)

---

## Task 2: Kernel op wrapper for AITER MLA

**Files:**
- Modify: `tokenspeed-kernel/python/requirements/rocm-thirdparty.txt`
- Create: `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/aiter_mla/__init__.py`
- Test: `test/aiter/test_aiter_mla_op.py`

- [ ] **Step 1: Add the dependency.** Append to `tokenspeed-kernel/python/requirements/rocm-thirdparty.txt`:
```
# AITER provides AMD MLA kernels; built from source in the ROCm image (commit pinned in the image build).
aiter
```

- [ ] **Step 2: Write the op wrapper.** Create `ops/attention/aiter_mla/__init__.py` — re-export AITER's MLA entry points so runtime never imports `aiter` directly (AGENTS.md boundary). Use the exact names confirmed in Task 0.
```python
"""AMD MLA kernels (ROCm AITER) exposed through the tokenspeed-kernel boundary."""
from __future__ import annotations

try:
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    _AVAILABLE = True
except ImportError:  # AITER only present on AMD images
    mla_decode_fwd = None
    mla_prefill_fwd = None
    _AVAILABLE = False


def is_available() -> bool:
    return _AVAILABLE


__all__ = ["mla_decode_fwd", "mla_prefill_fwd", "is_available"]
```

- [ ] **Step 3: Write the failing numerical-parity test.** Create `test/aiter/test_aiter_mla_op.py` — compare the wrapper's decode output to a small torch-reference MLA decode. (Reference: expand absorbed q/kv to standard attention and compute softmax(QK/√d)V on the latent dims; tolerance bf16-appropriate.) Use the shapes/convention from Task 0.
```python
import pytest, torch
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AMD GPU required")

from tokenspeed_kernel.ops.attention import aiter_mla as k


def _ref_mla_decode(q, kvb, sm_scale):
    # q: [B,H,Dq], kvb: [B,S,1,Dq]; latent attention over Dq, output over Dv=kv_lora
    B, H, Dq = q.shape
    S = kvb.shape[1]
    kv = kvb[:, :, 0, :].float()                     # [B,S,Dq]
    qf = q.float()                                   # [B,H,Dq]
    scores = torch.einsum("bhd,bsd->bhs", qf, kv) * sm_scale
    p = torch.softmax(scores, dim=-1)                # [B,H,S]
    Dv = Dq - 64                                     # kv_lora_rank
    out = torch.einsum("bhs,bsd->bhd", p, kv[..., :Dv])
    return out                                       # [B,H,Dv]


def test_mla_decode_matches_reference():
    assert k.is_available()
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    B, H, kv_lora, qk_rope = 2, 16, 512, 64
    Dq, Dv, page, S = kv_lora + qk_rope, kv_lora, 1, 8
    q = torch.randn(B, H, Dq, device=dev, dtype=dt)
    kvb = torch.randn(B * S, page, 1, Dq, device=dev, dtype=dt)
    o = torch.empty(B, H, Dv, device=dev, dtype=dt)
    qo_indptr = torch.arange(0, B + 1, device=dev, dtype=torch.int32)
    kv_indptr = torch.arange(0, (B + 1) * S, S, device=dev, dtype=torch.int32)
    kv_indices = torch.arange(0, B * S, device=dev, dtype=torch.int32)
    kv_last = torch.full((B,), page, device=dev, dtype=torch.int32)
    k.mla_decode_fwd(q, kvb, o, qo_indptr, kv_indptr, kv_indices, kv_last,
                     max_seqlen_q=1, page_size=page, sm_scale=1.0 / (Dq ** 0.5))
    torch.cuda.synchronize()
    ref = _ref_mla_decode(q, kvb.view(B, S, 1, Dq), 1.0 / (Dq ** 0.5))
    torch.testing.assert_close(o.float(), ref, atol=2e-2, rtol=2e-2)
```
(If Task 0 shows a different arg order/shape, adjust this call to match before running.)

- [ ] **Step 4: Run the test (fails first — wrapper import or shape).** Locally edit → rsync → run via harness:
```
TS_TEST_CMD='python -m pytest test/aiter/test_aiter_mla_op.py -v'
```
Expected initial: FAIL (e.g. mismatch or signature) — iterate the call to match Task 0's convention until PASS.

- [ ] **Step 5: Iterate to PASS.** Adjust the wrapper call args/shape per Task 0 until `test_mla_decode_matches_reference` PASSES on MI300A.

- [ ] **Step 6: Commit**
```bash
git add tokenspeed-kernel/python/requirements/rocm-thirdparty.txt \
        tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/aiter_mla/__init__.py \
        test/aiter/test_aiter_mla_op.py
git commit -s -m "feat(kernel): AITER MLA op wrapper + decode parity test (AMD)"
```

---

## Task 3: `AiterMLABackend` skeleton + registration

**Files:**
- Create: `python/tokenspeed/runtime/layers/attention/backends/aiter_mla.py`
- Test: extend `test/aiter/test_aiter_mla_op.py` with a registration test

- [ ] **Step 1: Write the failing registration test.** Append:
```python
def test_aiter_mla_registered_on_amd():
    from tokenspeed_kernel.platform import current_platform
    if not current_platform().is_amd:
        pytest.skip("AMD only")
    import tokenspeed.runtime.layers.attention.backends  # triggers registration
    from tokenspeed.runtime.layers.attention.registry import (
        _BACKEND_REGISTRY, _get_default_backend_name)
    from tokenspeed.runtime.configs.model_config import AttentionArch
    assert "aiter_mla" in _BACKEND_REGISTRY
    assert AttentionArch.MLA in _BACKEND_REGISTRY["aiter_mla"][0]
    assert _get_default_backend_name(AttentionArch.MLA) == "aiter_mla"
```

- [ ] **Step 2: Run — verify it fails** (backend not yet defined / not registered).
```
TS_TEST_CMD='python -m pytest test/aiter/test_aiter_mla_op.py::test_aiter_mla_registered_on_amd -v'
```
Expected: FAIL (`aiter_mla` not in registry).

- [ ] **Step 3: Write the backend skeleton.** Create `backends/aiter_mla.py` with `__init__` (mirroring `FlashMLABackend.__init__` dims from `MLAConfig`), cuda-graph guard (eager-only v1), and registration. Decode/extend raise `NotImplementedError` for now.
```python
"""AMD MLA attention backend wrapping ROCm AITER kernels (eager, v1)."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from tokenspeed_kernel.ops.attention import aiter_mla as kops
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


class AiterMLABackend(AttentionBackend):
    """Eager MLA backend for AMD (gfx942/gfx950) via AITER. No graph capture in v1."""

    def __init__(self, config: MLAConfig):
        super().__init__(config)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
        self.num_q_heads = config.num_attention_heads
        self.page_size = config.page_size
        self.forward_decode_metadata = None
        self.forward_prefill_metadata = None

    def init_forward_metadata(self, *args, **kwargs):
        raise NotImplementedError("Task 4/5")

    def init_cuda_graph_state(self, max_bs, seq_lens_buf):
        raise NotImplementedError(
            "AiterMLABackend v1 is eager-only; run with --enforce-eager")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, q, k, v, layer, out_cache_loc, token_to_kv_pool,
                       bs, save_kv_cache=True, **kwargs):
        raise NotImplementedError("Task 4")

    def forward_extend(self, q, k, v, layer, out_cache_loc, token_to_kv_pool,
                       bs, save_kv_cache=True, **kwargs):
        raise NotImplementedError("Task 5")


register_backend("aiter_mla", {AttentionArch.MLA}, AiterMLABackend)
```

- [ ] **Step 4: Wire registration + default (so the test can import/register).** Modify `backends/__init__.py` (after the `mha` import, line ~58):
```python
if platform.is_amd:
    _try_import_optional_backend("aiter_mla", "aiter")
```
And `registry.py:_get_default_backend_name` MLA branch:
```python
    if arch == AttentionArch.MLA:
        if platform.is_blackwell:
            return "trtllm_mla"
        if platform.is_hopper:
            return "flashmla"
        if platform.is_amd:
            return "aiter_mla"
        return "trtllm_mla"
```

- [ ] **Step 5: Run — verify it passes.**
```
TS_TEST_CMD='python -m pytest test/aiter/test_aiter_mla_op.py::test_aiter_mla_registered_on_amd -v'
```
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add python/tokenspeed/runtime/layers/attention/backends/aiter_mla.py \
        python/tokenspeed/runtime/layers/attention/backends/__init__.py \
        python/tokenspeed/runtime/layers/attention/registry.py \
        test/aiter/test_aiter_mla_op.py
git commit -s -m "feat(attn): AiterMLABackend skeleton + AMD registration/default"
```

---

## Task 4: `forward_decode` (metadata conversion + AITER decode)

**Files:** Modify `python/tokenspeed/runtime/layers/attention/backends/aiter_mla.py`

Reference `FlashMLABackend.forward_decode` (`flashmla.py:595`) and `_init_decode_metadata` (`flashmla.py:224`). FlashMLA builds `block_table = req_to_page[req_pool_indices]` + `cache_seqlens`; AITER needs `qo_indptr/kv_indptr/kv_indices/kv_last_page_lens`. Build them from `block_table` + `seq_lens` using `page_size`.

- [ ] **Step 1: Implement `init_forward_metadata` (decode path)** — build a small dataclass holding the AITER paging tensors. For decode (1 query/seq):
  - `qo_indptr = arange(bs+1)`
  - `kv_indptr[i+1] = kv_indptr[i] + ceil(seq_lens[i]/page_size)`
  - `kv_indices` = concatenation of `block_table[i, :num_pages_i]`
  - `kv_last_page_lens[i] = ((seq_lens[i]-1) % page_size) + 1`
  Store on `self.forward_decode_metadata`. (Exact tensor dtypes/int32 per Task 0.)

- [ ] **Step 2: Implement `forward_decode`** — mirror FlashMLA's reshape, then call the wrapper:
```python
def forward_decode(self, q, k, v, layer, out_cache_loc, token_to_kv_pool,
                   bs, save_kv_cache=True, **kwargs):
    if save_kv_cache and k is not None:            # only if not in _MLA_KERNEL_BACKENDS
        token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)
    md = self.forward_decode_metadata
    kv_buffer = token_to_kv_pool.get_key_buffer(layer.layer_id)\
        .view(-1, self.page_size, 1, self.kv_cache_dim)
    q = q.view(-1, self.num_q_heads, self.kv_cache_dim)
    o = torch.empty(q.shape[0], self.num_q_heads, self.v_head_dim,
                    dtype=q.dtype, device=q.device)
    kops.mla_decode_fwd(q, kv_buffer, o, md.qo_indptr, md.kv_indptr,
                        md.kv_indices, md.kv_last_page_lens,
                        max_seqlen_q=1, page_size=self.page_size,
                        sm_scale=layer.scaling)
    return o.view(-1, layer.tp_q_head_num * self.v_head_dim)
```
(Adjust arg order/shapes to Task 0's convention.)

- [ ] **Step 3: Test via end-to-end decode** — there is no isolated unit harness for the full backend metadata; correctness is validated by the e2e serve in Task 8. As an interim check, add a metadata unit test asserting the indptr/indices math on a hand-built `block_table`+`seq_lens` (pure tensor logic, no kernel):
```python
def test_decode_metadata_indptr_math():
    import torch
    page = 16
    seq_lens = torch.tensor([1, 16, 17, 40])
    # expected pages per seq: ceil(L/page) = [1,1,2,3]; last page lens = [1,16,1,8]
    kv_indptr = torch.zeros(5, dtype=torch.int32)
    for i, L in enumerate(seq_lens.tolist()):
        kv_indptr[i + 1] = kv_indptr[i] + (L + page - 1) // page
    last = ((seq_lens - 1) % page + 1).to(torch.int32)
    assert kv_indptr.tolist() == [0, 1, 2, 4, 7]
    assert last.tolist() == [1, 16, 1, 8]
```
Run: `TS_TEST_CMD='python -m pytest test/aiter/test_aiter_mla_op.py::test_decode_metadata_indptr_math -v'` → PASS. (Mirror this exact math in `init_forward_metadata`.)

- [ ] **Step 4: Commit**
```bash
git add python/tokenspeed/runtime/layers/attention/backends/aiter_mla.py test/aiter/test_aiter_mla_op.py
git commit -s -m "feat(attn): AITER MLA forward_decode + paging-metadata conversion"
```

---

## Task 5: `forward_extend` (prefill)

**Files:** Modify `backends/aiter_mla.py`

Reference `FlashMLABackend.forward_extend`/`_forward_normal_extend` (`flashmla.py:477,663`) and `_forward_absorbed_extend` (`flashmla.py:683`). Map: no-prefix extend → AITER prefill (explicit K/V or paged per Task 0); prefix-present → paged `mla_prefill_fwd`.

- [ ] **Step 1: Implement extend metadata** in `init_forward_metadata` for `forward_mode.is_extend_or_mixed()` (build `qo_indptr` from `extend_seq_lens`, `kv_indptr/indices/last` from prefix+extend lens).

- [ ] **Step 2: Implement `forward_extend`** calling `kops.mla_prefill_fwd` with the paged KV buffer view and the extend metadata (q reshaped to `[T,H,kv_cache_dim]`, output `[T,H,v_head_dim]`). Follow the exact convention from Task 0; handle the absorbed (`save_kv_cache=False`, model pre-wrote KV) vs normal path.

- [ ] **Step 3: Validate via e2e** (Task 8 — a single-prompt completion exercises prefill then decode). No isolated harness; rely on Task 8 correctness.

- [ ] **Step 4: Commit**
```bash
git add python/tokenspeed/runtime/layers/attention/backends/aiter_mla.py
git commit -s -m "feat(attn): AITER MLA forward_extend (prefill)"
```

---

## Task 6: Model wiring (KV ownership + V2-Lite arch alias)

**Files:**
- Modify: `python/tokenspeed/runtime/models/deepseek_v3.py:450`
- Modify: `python/tokenspeed/runtime/models/registry.py`

- [ ] **Step 1: Add `aiter_mla` to `_MLA_KERNEL_BACKENDS`** so the model writes KV (`set_mla_kv_buffer`) and passes `save_kv_cache=False`:
```python
_MLA_KERNEL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "aiter_mla")
```

- [ ] **Step 2: Alias the V2-Lite architecture.** In `models/registry.py`, register `DeepseekV2ForCausalLM` to resolve to `DeepseekV3ForCausalLM` (find the architecture→class map and add the alias; the attention/MoE code already reads `getattr(config, ...)` defaults that cover V2-Lite).

- [ ] **Step 3: Verify the alias resolves** (config-only, fast):
```
TS_TEST_CMD='python -c "from tokenspeed.runtime.models.registry import ModelRegistry as R; print(R.resolve_model_cls([\"DeepseekV2ForCausalLM\"]))"'
```
Expected: prints the DeepseekV3 class (no KeyError). (Adjust to the actual registry API found in the file.)

- [ ] **Step 4: Commit**
```bash
git add python/tokenspeed/runtime/models/deepseek_v3.py python/tokenspeed/runtime/models/registry.py
git commit -s -m "feat(model): route aiter_mla KV writes; alias DeepseekV2ForCausalLM"
```

---

## Task 7: End-to-end serve DeepSeek-V2-Lite on MI300A

**Files:** Create (cluster): `/capstor/.../tokenspeed-beverin/jobs/serve-dsv2lite.sbatch`

- [ ] **Step 1: Write the serve job** (reuse `smoke.sbatch` shape, image `tokenspeed-rocm-aiter`, model `deepseek-ai/DeepSeek-V2-Lite-Chat`):
```
tokenspeed serve deepseek-ai/DeepSeek-V2-Lite-Chat \
  --served-model-name dsv2lite \
  --trust-remote-code \
  --attention-backend aiter_mla \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --host 127.0.0.1 --port 8000
```
Wait for `/v1/models` 200, then curl `/v1/chat/completions` with a simple prompt.

- [ ] **Step 2: Run and capture.** Submit via sbatch; poll the log.
Expected: `server READY`; a coherent completion (not garbage), confirming decode+prefill correctness through `aiter_mla`.

- [ ] **Step 3: Correctness sanity.** Send 3–5 short factual/arithmetic prompts at `temperature=0`; verify sensible answers (a quick proxy for numerical correctness of the MLA path). If outputs are garbled, debug metadata conversion (Task 4/5) — likely `kv_indices`/`page_size`/scale.

- [ ] **Step 4: Record result** in `docs/superpowers/specs/aiter-mla-api-notes.md` (serve command, model, sample output) and commit.
```bash
git add docs/superpowers/specs/aiter-mla-api-notes.md
git commit -s -m "docs: DeepSeek-V2-Lite e2e serve via aiter_mla on MI300A"
```

---

## Self-review (completed)

- **Spec coverage:** op wrapper (T2), backend decode (T4)/extend (T5), registration+default+gating (T3), model wiring + V2-Lite alias (T6), MoE dependency (T1), e2e validation (T7), AITER-API lock + harness (T0). All spec sections mapped.
- **Eager-only v1:** enforced via `--enforce-eager` (T7) + `init_cuda_graph_state` raising (T3). Graph capture explicitly deferred.
- **Names consistent:** `aiter_mla` backend name, `AiterMLABackend`, `kops.mla_decode_fwd/mla_prefill_fwd`, `_MLA_KERNEL_BACKENDS` used identically across tasks.
- **Empirical kernel details:** Task 0 locks the exact AITER signature/convention before any kernel-calling code (T2/T4/T5), reducing blind-code risk; the metadata math has an isolated unit test (T4 Step 3).
- **No local-GPU assumption:** all kernel/serve tests run via the MI300A harness (T0).

## Open items the implementer must resolve on-hardware (flagged, not placeholders)

- Exact AITER arg order/shapes & supported `page_size` → Task 0 output drives T2/T4/T5 code.
- `ModelRegistry` alias API shape → confirm method name in `registry.py` during T6.
- AITER prefill entry choice (`mla_prefill_fwd` paged vs `mla_prefill_ps_fwd`) → decide from Task 0 notes in T5.

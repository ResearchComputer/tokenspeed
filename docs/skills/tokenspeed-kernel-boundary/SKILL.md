---
name: tokenspeed-kernel-boundary
description: >-
  Use when adding, binding, or fixing a kernel/op for tokenspeed — especially adding
  an AMD (gfx942) path for an op that is NVIDIA-only. Encodes the AGENTS.md kernel
  package boundary, the register_kernel / error_fn binding pattern, platform gating,
  vendored-vs-pinned-pip choice, and the Triton two-package tl.dot pitfall that has
  bitten this repo repeatedly.
---

# The tokenspeed-kernel boundary (and how to add an AMD kernel path)

## The boundary rules (these are invariants, from AGENTS.md)

1. **Runtime `tokenspeed` uses `tokenspeed-kernel` as its only kernel boundary.**
   Runtime code must **never** `import tokenspeed_kernel.thirdparty.cuda...`,
   `deep_gemm...`, `flash_mla...`, `xkernels...` directly. It imports a registry op.
   - This rule is invisible on NVIDIA and *fatal* off-NVIDIA: a direct
     `thirdparty.cuda` import is a `.so` that was never built for ROCm. Most AMD
     "missing kernel" crashes are this violation (e.g. the historical `lm_head_gemm`
     and `merge_state` crashes). The fix is always: route through the registry.
2. **Third-party kernel code is integrated two ways, both *inside* `tokenspeed-kernel`:**
   - (a) **pinned pip dependency** (preferred when upstream is self-contained and
     git-SHA-pinnable — e.g. `xkernels`, pinned in
     `tokenspeed-kernel/python/requirements/common.txt`), imported from `ops/`; or
   - (b) **vendored** under `thirdparty/` with a `NOTICE.md` recording the upstream
     commit (use only when local patches are needed). Prefer (a); upstream patches so
     a vendor copy can be dropped.
3. **All direct `tokenspeed-triton` imports happen in `_triton.py`**, then re-import
   elsewhere.
4. **Backend choice:** CuteDSL for NVIDIA, **Triton Gluon for AMD**, plain Triton for
   portable. Vendor libs (incl. TileLang) stay **optional**.
5. **`ops/` layout is `<family>/<solution>`** — e.g. `gemm/trtllm.py`,
   `attention/triton/`. New public APIs document args/returns.

## The error_fn binding pattern (how NVIDIA-only ops degrade)

An op that only exists on NVIDIA is bound to a sentinel so the *import* never fails;
the *call site* checks the sentinel and either dispatches a portable path or raises a
clear error. Example shape (`ops/attention/flash_mla/__init__.py`):

```python
from tokenspeed_kernel.registry import error_fn
flash_mla_sparse_fwd = error_fn          # default: "unavailable"
if platform.is_nvidia and platform.is_hopper_plus:
    from flash_mla import flash_mla_sparse_fwd        # NVIDIA path
elif platform.is_amd:
    from xkernels import flash_mla_sparse_fwd          # portable Triton path
```

Then the runtime backend keeps `if op is error_fn: raise RuntimeError(...)` guards so
an un-bound op gives a *named* failure, not an `AttributeError`. **To add an AMD
path you add the `elif platform.is_amd` branch here** — you do not touch the runtime
caller (it already calls the op by name).

## Registering a kernel/backend

- Kernels register via `register_kernel` from `ops/`; attention backends via
  `register_backend` (module-level, e.g.
  `register_backend("deepseek_v4", {AttentionArch.MLA}, DeepseekV4AttentionBackend)`).
- Registration is **triggered by import**. If your backend is "Unknown" at runtime,
  the *module was never imported for your platform* — check the platform branch in
  `layers/attention/backends/__init__.py`, not the `register_*` call.

## The two-package `tl.dot` pitfall (READ THIS before writing a Triton kernel here)

The runtime imports stock `triton.language`, but tokenspeed kernels use
`tokenspeed_triton`. **`triton.bfloat16` is not identical to
`tokenspeed_triton.bfloat16`** (two separate packages). A `tl.dot` with one operand
cast to the *foreign* bf16 fails its same-dtype assert: `Unsupported rhs dtype bf16`.

- **Fix:** cast the dequantized rhs to `a.dtype` (the in-tensor's own dtype), never
  to a `compute_type` constant from the wrong package. This is what made the in-kernel
  INT4 MoE compile.
- Vendored third-party Triton ops route their `triton` imports through a
  `_triton_compat` shim that rebinds `triton → tokenspeed_triton`. When vendoring or
  pinning a Triton kernel lib, ensure that shim is in place (it is a no-op standalone,
  so a pristine pip install still works).

## Choosing vendored vs pinned-pip (decision)

- Upstream is a clean Python package, no local patches needed, git-SHA pinnable →
  **pinned pip** (bump the SHA in `requirements/common.txt`). This is `xkernels`.
- Upstream needs local patches, or isn't packaged → **vendor** under `thirdparty/`
  with `NOTICE.md`. Plan to upstream the patches and drop the vendor copy.
- A heavyweight from-source build with no current payoff (e.g. a DSL that loses to
  Triton at your shapes) → keep it **opt-in / out of "auto"**, do not make it a hard
  dep. (This is why the TileLang sparse-MLA backend is held, not merged.)

## Iterating on a kernel WITHOUT rebuilding the image

Because `xkernels` is pure Python + Triton (JIT), you can shadow the pip-installed
copy with a source tree on `PYTHONPATH` and edit it live — no image rebuild per
change. See `beverin-dev-loop` ("PYTHONPATH-shadow"). Land the change as a proper
SHA bump + a pinned-pip update only once it is validated.

## Red flags

| Thought | Reality |
|---|---|
| "I'll just import deep_gemm/thirdparty.cuda in the model" | Boundary violation; crashes off-NVIDIA. Use the registry op. |
| "I'll cast to triton.bfloat16 for the dot" | Wrong package → `Unsupported rhs dtype`. Cast to `a.dtype`. |
| "The backend class exists, so it's registered" | Registration needs the *import* to run for your platform. |
| "I'll vendor this DSL to get a kernel" | Vendor libs stay optional/opt-in; prefer Triton-Gluon; don't add a hard from-source dep for no win. |

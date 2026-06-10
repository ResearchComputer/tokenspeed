# Vendored: `xkernels`

This directory is a vendored, self-contained copy of the **`xkernels`** kernel
library, integrated under tokenspeed's third-party kernel boundary
(`tokenspeed_kernel/thirdparty/`) per `AGENTS.md`.

## Provenance

- **Upstream**: https://github.com/ResearchComputer/kernels
- **Branch**: `main`
- **Commit**: `9c403de275a8db108d69383a387a7edff9f764c4`
- **Package path upstream**: `src/xkernels/`
- **License**: MIT (see `LICENSE` in this directory)
- **Previously vendored at**: `696740a` (refreshed to pick up the Triton
  `moe_align_block_size` kernel (#15/#4), the `ops/comm` topology-aware
  hierarchical all-reduce + fused residual-add/RMSNorm epilogue (#12), and the
  speedup-vs-PyTorch bench harness (#14)).

## What was vendored

The pure-Python + Triton parts of the package, importable as:

```python
from tokenspeed_kernel.thirdparty.xkernels import (
    dual_rmsnorm,            # ops/norm
    fused_moe_int4_w4a16,    # ops/moe
    moe_align_block_size,    # ops/moe
    moe_sum_reduce,          # ops/moe
    mha_merge_state,         # ops/attention
    fused_ffn,               # ops/ffn
    # ops/comm (distributed collectives, take process groups not the Backend
    # registry): build_topology_groups, flat_all_reduce, hierarchical_all_reduce,
    # residual_rmsnorm (fused residual-add + RMSNorm epilogue).
)
```

Each op auto-dispatches across `reference` (pure torch) and `triton` backends
via the package's own `_dispatch` / `_backends` (`detect_vendor()` → amd → HIP,
TRITON, REFERENCE). Inside tokenspeed we generally bypass this auto-dispatch and
call the concrete backend (or register it through tokenspeed's own
`register_kernel`) so kernel selection stays under tokenspeed's `registry`.

## Modifications vs upstream

The vendored source is kept as close to upstream as possible. The only changes:

1. **CUDA/HIP C++ extension dropped.** The optional compiled extension
   (`ops/**/*.cu`, `*.cpp`, `*.h`, built by upstream `setup.py`) was **not**
   vendored — AMD uses the Triton backend. `ops/ffn/cuda/__init__.py` already
   guards its `from . import _cuda` import behind `try/except`, so the missing
   extension simply means the `Backend.HIP`/`Backend.CUDA` FFN backend is not
   registered and `fused_ffn` falls back to Triton/reference.

2. **Triton-package routing (the two-triton-package pitfall).** Upstream Triton
   backends are written against the *stock* `triton` package. Inside tokenspeed,
   runtime kernels must instead bind the vendored `tokenspeed_triton` (imported
   via `tokenspeed_kernel._triton`); the two are separate packages and
   `tl.dot` rejects cross-package dtype objects ("Unsupported rhs dtype").
   To fix this without touching the kernel source, the package-level
   `__init__.py` files (`ops/norm`, `ops/moe`, `ops/attention`, `ops/ffn`) now
   route their Triton-backend imports through `_triton_compat.triton_import_ctx()`,
   a small added shim (`_triton_compat.py`) that aliases `triton[.x.y]` →
   `tokenspeed_triton[.x.y]` during the backend import (no-op when used
   standalone with stock Triton).

3. **`ops/comm/interface.py` not present.** Upstream removed it (the comm ops are
   plain distributed functions, not registry-dispatched); the stale vendored copy
   was dropped on this refresh.

No other files were modified; the kernels, references, interfaces, configs, and
the `_dispatch`/`_backends` machinery are byte-identical to upstream. The four
shimmed `__init__.py` are `ops/{norm,moe,attention,ffn}`; `ops/moe/__init__.py`
additionally routes the new `align_kernel` import through the shim.

## Updating

To refresh, re-clone upstream at the new commit, re-copy `src/xkernels/.` here,
re-drop the C++/CUDA sources, re-apply the four `__init__.py` Triton-import edits
(and keep `_triton_compat.py` / `LICENSE` / this `NOTICE.md`), then bump the
commit SHA above.

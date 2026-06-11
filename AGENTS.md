# General Agent Guidelines

> If a `AGENTS.local.md` file exists alongside this file, read and respect it--
> it contains developer-specific overrides that supplement this shared guidance.

## Development environment

* Before any work, check local Python venv and activate if one exists.
* Don't install pip packages outside the local Python venv if one exists.

## Code changes

* Add tests and update docs for the changed code.
* Before creating commits, run `pre-commit run --all-files` to format.
* When creating commits, perform sign off on behalf of the author.

## Dependency boundaries

* `tokenspeed` runtime dependencies should stay vendor-neutral.
* Runtime code should use `tokenspeed-kernel` as its only kernel package
  boundary.
* Third-party kernel libraries belong under `tokenspeed-kernel`; avoid direct
  runtime dependencies or imports that bypass it.
* If a dependency repeatedly breaks during version upgrades or slows project
  progress, consider removing it entirely or at least making it optional.

## tokenspeed-kernel

Inside the root tokenspeed-kernel/ directory:

* All direct tokenspeed-triton imports should happen in `_triton.py` and then
  re-import to other places.
* Third-party kernel code is integrated one of two ways, and either is fine as
  long as it is imported only inside `tokenspeed-kernel` (never from runtime
  `tokenspeed`) and registered via `register_kernel` from `ops/`:
  (a) a **pinned pip dependency** declared in `requirements/` and imported from
  `ops/` (preferred when the upstream package is self-contained and version-
  pinnable — e.g. `xkernels`, pinned by git SHA); or
  (b) **vendored** under `thirdparty/` (use when upstream needs local patches or
  isn't packaged). Vendor copies must record provenance (a `NOTICE.md` with the
  upstream commit). Prefer (a) and upstream any local patches so the vendor copy
  can be dropped.
* Prefer CuteDSL for NVIDIA GPU kernels and Triton Gluon for AMD GPU kernels.
  Use Triton for portable solutions across vendors. Vendor libraries should
  stay optional, and other solutions may be used as temporary transitions, but
  new work should consolidate toward these backend choices.
* Files under `ops/` should follow `<family>/<solution>` structure, like
  `gemm/trtllm.py` or `attention/triton/`.
* When defining new public APIs, explain arguments and returns in docstring.

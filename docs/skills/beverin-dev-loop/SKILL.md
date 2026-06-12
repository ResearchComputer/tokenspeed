---
name: beverin-dev-loop
description: >-
  Use when building, serving, or iterating on tokenspeed on the CSCS beverin cluster
  (AMD MI300A / gfx942). Encodes the enroot overlay image build, the PYTHONPATH-shadow
  fast-iteration trick, the code-sync (and stale-snapshot trap), the EDF/srun serve
  recipe, and the specific env gotchas (HSA_NO_SCRATCH_RECLAIM, gpu-mem-util for
  unified memory, the NCCL aws-ofi-rccl plugin path, Triton cache dir, the 24h cert).
---

# The beverin (MI300A) dev loop

beverin = CSCS Alps vCluster, AMD **MI300A APU (gfx942, CDNA3)**, 4 GPUs/node,
**unified CPU+GPU memory (~128 GB/GPU, ~512 GB/node shared)**. `srun -A a-infra02
-p mi300`. Containers are **enroot + pyxis** (`.sqsh` + EDF), not Apptainer. See the
project memory `beverin-cluster-facts` and `tokenspeed-beverin-deploy` for the full
deployment history.

Paths:
- deploy root: `/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/`
  (`jobs/ images/ logs/ aws-ofi-rccl/ libs/`)
- code (rsync snapshot, run from source): `/capstor/.../infra02/xyao/code/tokenspeed-amd`
- EDFs: `~/.edf/tokenspeed-rocm-aiter-myofi-xk*.toml`

## The fast iteration loop

The cluster runs tokenspeed **from source via `PYTHONPATH`**, not an installed wheel
(Triton kernels JIT; only the heavy thirdparty CUDA libs are baked into the image).
So the loop is:

```
edit locally ──► rsync the changed tree ──► sbatch a serve/parity job ──► read the log
```

1. **Sync code** (no `git push`; the cluster is an rsync snapshot):
   ```bash
   rsync -az -e ssh --include='*/' --include='*.py' --exclude='*' \
     python/ beverin:/capstor/.../code/tokenspeed-amd/python/
   rsync -az ... tokenspeed-kernel/python/ beverin:.../tokenspeed-kernel/python/
   ```
   > **Stale-snapshot trap (this has cost real serve cycles):** the cluster snapshot
   > can be *older than your local HEAD*. A single-file sync leaves the rest stale —
   > you will hit "feature added in commit X is missing" (e.g. an AMD backend
   > registration that exists locally but not on the cluster → `Unknown attention
   > backend`). When in doubt, **sync the whole `python/` + `tokenspeed-kernel/python/`
   > trees**, and `grep` the cluster file to confirm your change landed.

2. **Kernel iteration without an image rebuild — the PYTHONPATH-shadow.** `xkernels`
   is pure Python + Triton. Extract a clean tree of the SHA you want and put its
   `src/` *first* on `PYTHONPATH` so `import xkernels` resolves to it, shadowing the
   pip-installed copy in the image:
   ```bash
   git -C <kernels> archive <sha> src | ssh beverin 'tar -x -C .../code/kernels-<sha>'
   # in the serve sbatch:
   export PYTHONPATH=.../code/kernels-<sha>/src:$CODE/python:$CODE/tokenspeed-kernel/python
   ```
   This turns a ~15-min image rebuild into a ~3-min edit→serve. Bake the SHA into a
   pinned-pip `requirements/common.txt` bump + a rebuilt image only once validated.

## Building an enroot overlay image (when you DO need a new dep in the image)

Use the overlay pattern (`jobs/build-xkernels-overlay-xk*.sbatch`): `enroot create`
a writable rootfs from a base `.sqsh`, `pip install` the new dep, `enroot export` to a
**new** `.sqsh` (never clobber the last-known-good image — bump the suffix, e.g.
`-xk2`→`-xk3`). Key points baked into those scripts:
- Put all enroot data paths on **`/dev/shm`** (compute nodes are diskless).
- In-container **git is broken** (stale `libnghttp2`): `LD_PRELOAD=/lib/x86_64-linux-gnu/libnghttp2.so.14`
  for the git/pip step.
- `pip install --no-deps --no-build-isolation` and verify `torch.version.hip` is
  **unchanged** after (a pip resolver can silently clobber the ROCm torch).
- A new image needs a matching EDF (`~/.edf/...-xkN.toml`) pointing at it.

## The serve recipe and its non-obvious env (single-node TP, MI300A)

A correctness-first V4/DeepSeek serve (`jobs/serve-v4flash-tp4.sbatch` is the
template) needs, in the `srun --environment=<edf>` body:

| Setting | Why |
|---|---|
| `HSA_NO_SCRATCH_RECLAIM=1` | MI300A + ROCm 7.2 **aborts at kernel launch** without it. Non-negotiable. |
| `--gpu-memory-utilization 0.5` (4 ranks) | **Unified memory**: 0.9 × 4 ranks → host-cgroup OOM allocating the KV pool. Use ~0.5; lower if still OOM. |
| `LD_LIBRARY_PATH=.../aws-ofi-rccl/lib` | The myofi EDF forces `NCCL_NET="AWS Libfabric"`; without the plugin on the path RCCL fails the first all-reduce: `Failed to initialize any NET plugin`. Needed **even single-node**. |
| `TRITON_CACHE_DIR=/tmp/triton-cache` | A shared-FS Triton cache → `Errno 116`. Use node-local `/tmp` (NOT `/dev/shm` — noexec). |
| `--enforce-eager --block-size 1 --no-enable-prefix-caching --disable-kvstore` | Correctness-first AMD path; drop these only once a graph/paged path is validated. |
| `TORCH_COMPILE_DISABLE=1` | torch.compile is a net pessimization on AMD eager decode and triggers an 8-rank inductor abort. Keep off. |
| `--kvstore-ratio 0.1` | The pinned host KVStore pool is `ratio × device_pool` per rank; 4× overflows. |

A serve job runs the engine in the background, polls `/v1/models` for READY, then
fires curls: **`max_tokens=1` = prefill-only**, `max_tokens=48` = decode. Use this
split to gate prefill before decode.

## Operational gotchas

- **24h SSH cert.** `ela.cscs.ch: Permission denied (publickey)` = expired cert, not a
  bug; renew via MFA at https://sshservice.cscs.ch/ (the *user* does this; an agent
  cannot). **Running jobs are unaffected** — only new ssh/scp/sbatch fail.
- **Don't trust a single `squeue` miss as "job done"** in a poll loop — a transient
  ssh blip returns empty and a naive `until ! squeue ...` exits early. Grep the job
  stdout for an explicit terminal marker (`JOB_DONE` / `PROCESS EXITED`) instead.
- **V4-Flash load is ~15-20 min** (149 GB over capstor). Budget for it; warm-cache
  reloads are only modestly faster.
- **Account flag is mandatory:** `-A a-infra02` (note the `a-` prefix), partition
  `mi300`.

## Red flags

| Thought | Reality |
|---|---|
| "I synced the one file I changed" | The rest of the snapshot may be stale. Sync the whole tree; grep to confirm. |
| "I'll rebuild the image to test a kernel edit" | Use the PYTHONPATH-shadow; rebuild only to ship a pinned SHA. |
| "OOM → reduce the model" | On unified memory it's `gpu-mem-util × ranks`. Tune the knob first. |
| "It's single-node, NCCL is fine" | The myofi EDF still forces the OFI plugin; set the plugin `LD_LIBRARY_PATH`. |
| "squeue shows it gone, it finished" | Could be an ssh blip. Confirm via a stdout terminal marker. |

# Agent-native skills: developing & improving the serving engine

These are **skills** — structured, agent-followable knowledge documents — for working
on `tokenspeed` (the serving engine) and `tokenspeed-kernel` (its kernel package),
with a focus on **bringing models up on non-NVIDIA hardware** (AMD MI300A / gfx942)
and **improving serving throughput**.

They encode hard-won methodology and gotchas that are *not* derivable from the code:
the order to attack blockers, the kernel-package boundary rules, the cluster dev loop,
the failure taxonomy, and the profiling levers.

## How an agent should use these

1. Read `serving-engine-bringup` first — it is the spine; the others are its limbs.
2. When a step needs detail, jump to the specific skill (it tells you which).
3. Prefer the concrete commands / file paths here over guessing. Verify a cited
   `file:line` still exists before acting on it — the engine moves fast.
4. These are **flexible** skills (adapt the principle to the case), except where a
   section says "rule" — those are invariants (e.g. the kernel-boundary rules).

## The skills

| Skill | Use when |
|-------|----------|
| [serving-engine-bringup](serving-engine-bringup/SKILL.md) | Bringing a new model or attention/MoE backend online on a new accelerator. The methodology: incremental blocker-chain, staged-init mental model, correctness-first. |
| [tokenspeed-kernel-boundary](tokenspeed-kernel-boundary/SKILL.md) | Adding/binding a kernel or op; deciding vendored-vs-pinned-pip; the `register_kernel`/`error_fn` pattern; the Triton two-package `tl.dot` pitfall. |
| [beverin-dev-loop](beverin-dev-loop/SKILL.md) | The MI300A cluster dev loop: enroot overlay image build, the PYTHONPATH-shadow fast-iteration trick, code sync, EDF/srun, the serve recipe and its env gotchas. |
| [diagnosing-serve-failures](diagnosing-serve-failures/SKILL.md) | A serve crashes/hangs. The failure taxonomy keyed to the init stage (import → backend → KV pool → comm → warmup → forward) and how to read each. |
| [decode-throughput-optimization](decode-throughput-optimization/SKILL.md) | Decode tok/s is low. The host-bound model, HIP-graph capture, sync-free kernels, the offline profiler, and which levers actually move the needle. |

## Relationship to the rest of the repo

- `AGENTS.md` — the authoritative dependency-boundary rules; these skills operationalize them.
- `docs/superpowers/specs/` & `docs/superpowers/plans/` — per-feature design+TDD artifacts.
- `tokenspeed-kernel/` — the only kernel-package boundary for runtime code.

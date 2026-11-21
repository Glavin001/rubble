---
name: gpu-ab-on-modal
description: Judge a GPU performance change by A/B on Modal CUDA. Use it when an optimization, kernel rewrite or architecture change needs a verdict (faster or slower, and still bit for bit), when comparing two commits or branches, or when deciding between a runtime switch and two builds.
user-invocable: false
---

# A/B performance changes on Modal (CUDA)

The default way to tell whether a change makes the GPU frame faster without changing
results. Kernels are built locally with nvcc (free); Modal only runs the binaries. Each
verdict covers all 12 scenario-matrix rows, takes about 30 s to 2 minutes on Modal, and
costs half a cent to 2 cents on a T4. There is no GPU lock and no risk to the Mac. Full reference:
`engine/tools/modal/README.md` ("Two builds, head to head" and "The scenario matrix on
CUDA").

## Pick the comparison

| The change | How |
|---|---|
| Anything you can commit (the usual case, and the only one for architecture changes: a new buffer layout, a new schedule, rewritten kernels) | **Two builds.** Commit it, then `run.sh --cuda --base BASE_REF [--head HEAD_REF]` |
| A setting meant to stay switchable at runtime (a real feature flag) | **One build, a switch.** `run.sh --cuda SWITCH` |

Prefer two builds for performance experiments. A switch adds branches you must later
remove. A branch inside a hot kernel can change register use and spill long-lived values,
so you end up measuring the branch. Two builds measure exactly the code you would ship.

## Two builds, step by step (from `engine/`)

1. Commit the experiment on its branch. The working tree can be the head, but it is not
   kept: only commits enter the library.
2. Run:
   ```bash
   ENGINE_CUDA_FMAD=0 tools/scenario-matrix/run.sh --cuda --base main --head my-branch
   ```
   - `ENGINE_CUDA_FMAD=0` (strict floats) keeps bit for bit meaningful across builds that
     launch different modules. It costs nothing measurable.
   - Without `--head`, the head is the working tree's build.
   - Pick rows by name after the refs (`... --head X pile8 pile32`).
3. Each build is made once in local Docker (`build_ref.sh`: about 8–10 minutes the first
   time), then kept in the library `engine/target/gpu-run-refs/` (gitignored), keyed by
   commit, `ENGINE_*` settings and every row's saved state. Later comparisons involving
   that commit cost no build, whichever rows they name.
   - `tools/modal/build_ref.sh --list` shows what is kept and what each binary is
     (`info.txt`).
   - `--prune N` keeps the newest N.
4. Read the table:
   - `head ms` / `base ms`: the median of each build's processes. Base and head run
     alternately, A B, B A, A B (`PROCS=3`), in one container on one GPU.
   - `delta`: the head against the base.
   - `noise`: the spread between the base's own processes. Trust a delta well above it.
   - `bit4bit`: the end states word for word. A changed layout compares the bindings both
     builds have and names the ones that differ.
   - Verdict: PASS only if every row is bit for bit and none is slower than 5 % and
     0.05 ms.
5. A change that reorders arithmetic on purpose: `ALLOW_DIFF=1` reports state differences
   without failing. Equivalence then needs its own evidence (the CPU-reference suites,
   rung 3). Never wave a difference through for a change meant to be lossless.
6. To see why a change is faster or slower, add `COUNTERS=1` and name a few rows. Each
   kernel then gets NVIDIA's hardware counters for both builds: time, registers, achieved
   occupancy, SM and DRAM throughput, L1 and L2 hit rates. The base's value is shown in
   brackets where it moved.
   - Clocks cannot be locked on Modal, so ratios can be off. Use the counters to explain;
     the timed table decides.
   - If the kernels' total barely moves, or drops, while the frame slows, the cost is
     between launches or on the host: run the switch matrix with `PROFILE_SPLIT=1`, or
     look at launches and waits.
   - Expensive: about 6.5 minutes and 11 cents a row (8 cores, given automatically).
     Name one or two rows. Its timing table is noisier; judge speed by a run without
     `COUNTERS`.

Proof it works (2026-11-21, T4): base 58a0cfe1 against a demo commit that makes the host
read indirect counts back. 11 of 12 rows were flagged slower by 13–42 %, every row stayed
bit for bit, and noise was ≤ 0.5 %. The same commit against itself passed with deltas
within 0.7 %.

## How reliable it is (measured 2026-11-21, T4)

- Within one container the halves of a row agree to 0.0–0.2 % on the multi-car rows and
  usually 0.2 % on single-car rows. Between separate processes, the single-car rows can
  shift a few percent, which is why each build gets several processes and the median.
- Between separate runs (different containers, possibly different hosts and drivers),
  absolute times move: drive 3.64–3.90 ms, truss 1.26 vs 2.18 ms. **Never compare a
  head against a base number from an earlier run.** Always run both in the same run.
- One run per candidate is enough when the delta clears the noise. Repeat (or raise
  `FRAMES`, `ROUNDS`, `PROCS`) only to claim a small win (under about 2 %) on single-car
  rows. Repeats cost half a cent each.

## What it does not cover

- **Apple and Chrome.** A T4 win is an NVIDIA win. Before keeping a change, run the Mac
  matrix once (`tools/scenario-matrix/run.sh SWITCH` under the lock, rung 1.5). A
  default-on change also needs Chrome's A/B (rung 3, AGENTS.md).
- **Absolute numbers for docs.** Those come from rung 4 on a dedicated GPU.
- **Code without a CUDA host.** The matrix rows cover the explicit soft-body frame. The
  World's coupling, fracture and terrain are not on CUDA yet
  (`engine/docs/world-on-cuda.md`).
- **States the base cannot read.** Both builds replay the same portable fixtures
  (`out/scenario-matrix/portable/`). If the head changes what a fixture holds, make the
  fixtures again from the head (`run.sh --make`, `matrix.py --make-portable`). The base
  must be able to read them too, or you need a fixture per build.

## Costs

| Item | Cost |
|---|---|
| A two-build verdict, both builds in the library | about 110 s on Modal, about $0.02 |
| A switch verdict | about 30 s on Modal, about $0.005 |
| A new commit's build | local Docker, free, minutes once |
| $30 a month | about 1,500 two-build verdicts or 6,000 switch verdicts. The limit is build time, not money |

# The stress-solver baseline: goal, rules, milestones

This is the work's charter (2026-10-18). Every change on this track serves one of the milestones
below, in order. Anything else is a follow-up, recorded and not started.

- Status and history: `stress-qs-baseline-plan.md`.
- What we are measured against: `vibe-land-physx-review.md`.

## Goal

Make **rubble capable of replacing PhysX GPU and its native destruction stage** as vibe-land's
destruction physics engine.
- Integrating rubble into vibe-land is **out of scope**. The work is rubble's own.

That means a quasi-static stress solver built into rubble as a **first-class GPU feature, in
Slang**, which is:
- **resident:** one submission and one status read per tick, and no other host communication;
- **forward-only:** fracture and splits take effect between rubble's substeps, the moment bonds
  fail. No rewind and no correction re-solve;
- **deterministic:** identical runs give identical results;
- at least at **parity** with the PhysX native stage in features, accuracy and speed, on
  vibe-land's own town bombard.

Higher-fidelity dynamic impact handling comes after this baseline.

## Rules

1. **No deviation.** Work that does not advance the current milestone is written down as a
   follow-up, not done.
2. **Capabilities are options.** The stress solver will grow more capabilities than Blast has:
   - impact islands;
   - explicit dynamics;
   - rate effects, fatigue, crushing, multirate;
   - the oracle's model switches.

   Each one is a **named capability flag**: `SolverConfig`, `Methods` or `SimFeatures`, with an
   environment override for measurement.
3. **The baseline runs as one named profile, `blast_parity`.** It turns off every capability
   Blast's quasi-static solver does not have, and sets Blast's behaviour:
   - Blast's model (`blast_model`);
   - authored fatal limits and the pack's masses;
   - a budget of 1 Newton step × 16 iterations, tolerance 1e-3;
   - unconverged answers committed;
   - at most 2 cascade passes per tick;
   - the repeat gate and certified repeats.

   A capability is enabled in a profile only once it is measured fast enough to keep that
   profile's targets.
4. **Nothing is deleted.** Existing code paths are kept, for example:
   - the explicit and adaptive solvers;
   - impact islands;
   - the host split and rebuild paths;
   - the oracle's methods.

   A refactor makes them selectable by a flag; it does not remove them. A path that the GPU
   replaces stays as the reference it is tested against, and as a fallback.
5. **Lossless by default.**
   - A change must be bit-identical, or pass the full reference tests.
   - Lossy options are flags, measured and reported; turning one on is your decision (AGENTS.md).
6. **Determinism is a gate.** Every milestone's acceptance includes two identical runs agreeing
   bit for bit on both suite scenes, plus the guard (`scripts/destruction_peak.py guard`).
7. **Slang only** for GPU code. Rust holds host and CPU logic. JS is a thin layer.

## The benchmark and targets

- **Scenes:** `stress_bench --pack <vibe-land suite pack> --only vibe-town_suite_quick` and
  `vibe-town_suite_bombard`.
  - Pack: `vibe-land/target/perf-suite/packs/runtime/town/vibe-town.json`.
  - These are vibe-land's own events: 30 settle frames, then 60 or 90 recorded frames.
- **Machine:** quiet M3 Max, nothing else on the GPU.

| Tier | Median | p95 | Worst |
|---|---|---|---|
| Standard (90 ticks, 5 shots) | ≤ 11.46 ms | ≤ 37.88 ms | ≤ 48.24 ms |
| Quick (60 ticks, 2 shots) | ≤ 11.38 ms | ≤ 15.32 ms | ≤ 35.5 ms |

**Accuracy:**
- breakage within vibe-land's run-to-run variation on the same events;
- per-iteration stress accuracy at least Blast's (`blast_forces_gpu` against `blast_ladder_bench`);
- **at least as faithful as the native stage's correction-limit-1 pass** on the free-wall ram
  and the wall row (`findings/rewind-scaling.md` scenes).

## Milestones

Each milestone's acceptance criteria are all required. Its measurements go into this file's log
(below) when it closes.

### M0. One profile: `blast_parity`

- **Steps:**
  - Gather today's scattered Blast-matching switches into one named profile:
    - in stress-ref `SolverConfig` and Scene;
    - in `stress_bench --config blast_parity`;
    - in rubble's API.
  - Every other capability is explicitly off in it.
  - Document which flag means what.
- **Accept when:**
  - the profile reproduces the current matched configuration's results bit for bit;
  - no code path was removed;
  - `--config blast` and the environment overrides still work.

### M1. Deterministic baseline

- **Steps:** find and fix the race in the first static solve after a split.
  - It shows in both the rebuild and the in-place paths.
  - Use `STRESS_GPU_SPLIT_HASH=1` and `STRESS_GPU_STATE_DUMP` to locate the first diverging
    chunk.
- **Accept when:**
  - two identical runs agree bit for bit on both suite scenes;
  - the guard is 24/24;
  - stress-gpu and stress-rubble tests pass.

### M2. Fracture topology on the GPU, in Slang

M2 and M3 may proceed in parallel.

- **Steps:** port the native stage's algorithms against our layout:
  - connected components of broken clusters, as union-find plus scans
    (`PhysX/physx/source/gpudestruction/src/PxgDestructionTopology.cu`,
    `PxgDestructionTransaction.cuh`);
  - each child's mass, centre of mass, inertia, principal frame and velocity
    (`prepareCandidateBodies`, `checkUnchangedMotionCommit` in `PxgDestructionRuntime.cu`);
  - incremental: untouched clusters keep their records.
- **Accept when:**
  - on every split of both suite scenes, the GPU's children (membership and order), masses
    and inertias equal the host's `split_cluster`, exactly or within a stated f32 tolerance;
  - deterministic;
  - the host path is kept as the reference.

### M3. New rigid bodies created on the GPU, in rubble

- **Steps:**
  - a rubble kernel (`engine-webgpu`) creates `BodyState` records from candidate records;
  - chunk colliders are re-pointed to their new body (the chunk is already one collider);
  - retired bodies are released;
  - slots are allocated by scans (deterministic, no atomic appends);
  - counts stay on the GPU;
  - the new-body delta rides on the tick's one status read.
- **Accept when:**
  - a split's bodies equal those `sync_bodies` creates from the host (same keys, poses and
    velocities);
  - no host call is involved;
  - deterministic.

### M4. Split layout on the GPU

- **Steps:**
  - a split parent's island tables (chunks, bonds, incidence, island records, gate entries,
    static-solve slots) are re-laid out by kernels;
  - this replaces `split_resident`'s host work and the mirror sync plus rebuild;
  - the host mirror learns about the split as a delta.
- **Accept when:**
  - a split frame does zero host work beyond the one status read;
  - the results equal the host rebuild path;
  - deterministic;
  - the suite's p95 and worst are set by GPU time (reported).

### M5. Forward splits inside rubble's step

- **Steps:**
  - the stress stage joins rubble's step graph (`engine-exec/src/rigid_plan.rs`): after each
    substep, run the contact ingest, the cascade, the fracture commit and the GPU split, and
    the next substep sees the separated bodies;
  - every dispatch is sized on the GPU (including rubble's colour count);
  - a tick is one submission and one read.
- **Accept when:**
  - one submission and one status read per tick (counted);
  - the free-wall ram and wall row are at least as close to the oracle as a correction-limit-1
    pass (ram speed, fragment count);
  - deterministic.

### M6. Speed to the targets

- **Steps, lossless first:**
  - rubble's substep: dispatch count and colour headroom, broadphase;
  - the cascade only on structures a substep loads;
  - the CCD and iteration policy, decided with you and reported.
- **Accept when:** both suite tiers meet the targets, with breakage, fidelity and determinism
  reported.

### After the baseline

Enable the higher-fidelity capabilities (impact islands, explicit dynamics), one flag at a time,
each when it keeps the profile's targets. They stay on the same resident, forward foundation.

## Log

| Date | Milestone | Result |
|---|---|---|
| 2026-10-18 | **M2 done** | **`shaders/split.slang` and `src/gpu_split.rs`.** `split_components` runs one group per parent island: minimum-label connected components over still-connected bonds, then the children in order of their first chunk with their member lists. `split_children` runs one thread per child: mass, centre of mass, inertia, anchoring, and rigid velocity from the chunks' exact world velocities, given each parent's rigid frame. Bond types moved to the shared `bonds.slang` (generated `world.metal` byte-identical). **Check** (`STRESS_GPU_SPLIT_CHECK=1`, every split, both suite tiers): on the standard tier, 64 splits and 639 children with 0 topology and 0 anchoring mismatches; mass 5e-7 relative, centre of mass 5.5e-5 m, inertia 7e-6, velocity 2.4e-4, angular velocity 1e-2 (near-zero spins) against the host's f64. **Bug found and fixed:** the host's rebuild split path split with stale chunk states, because `sync` brings bonds only. `sync_chunk_state` now reads the parents' state first. Quick tier after the fix: digest 1797fb47 both runs, median 15.2 ms, p95 107–109, worst 189–192, 572 bonds broken (vibe-land 638). Guard 24/24; resident and ingest tests pass |
| 2026-10-18 | **M1 done** | **Cause:** rubble appends contact rows at atomic slots (`engine/slang/kernels/world_step.slang:1879, 2323`), so their order varies run to run. The resident ingest summed each chunk's records in row-slot order (`coupling.slang` `ingest_merge`). **Fix:** each record carries a content key (substep, own collider, other collider, feature × 2 + side), and the merge orders by it. The fix is in the shared ingest, so it covers both split paths. **Results, two identical runs each:** `vibe-town_suite_quick` digest 0ffcf577 both runs (median 14.0 ms, p95 102–105, worst 188–190, 513 bonds broken); `vibe-town_suite_bombard` digest 41c25930 both runs (median 15.8–15.9 ms, p95 105–106, worst 188, 2,848 bonds); in-place splits (`STRESS_QS_SPLIT_IN_PLACE=1`) digest c12e91b3 both runs (median 16.3, p95 49–50, worst 108–109, 2,989 bonds). Guard 24/24; stress-gpu tests pass; stress-rubble tests pass after one test fix: `ingest_gpu` now compares manifolds by other body rather than in row order, which is the nondeterministic quantity |
| 2026-10-18 | **M0 done** | **`stress_ref::capability::Profile::BlastParity`** (`crates/stress-ref/src/capability.rs`). Applied through `stress_bench --config blast_parity`, `GpuWorld::apply_profile`, and the scene's `sim.static_tolerance` (new; it reaches the setup solves). Bit-identical to the old flags (`--config blast --authored-fatal --pack-masses --method blast_model=on` with `STRESS_RESIDENT=1 STRESS_QS_BUDGET=1,16 STRESS_GPU_STATICS_TOL=1e-3 STRESS_QS_UNCONVERGED=commit STRESS_QS_PASSES=2`): Vibe Town idle digest b8d04bba both ways, and the suite scene's 31 settle-frame digests identical. The old flags and environment overrides still work (the environment overrides the profile); nothing removed |
| 2026-10-18 | baseline before M0 | quick tier: median 16.6–16.8 ms, p95 120–134, worst 222–277; **not deterministic** (2,920 vs 513 bonds broken) |

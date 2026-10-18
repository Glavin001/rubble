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
| 2026-10-18 | **M4, part 4** | **The split pass copies only what its layout reads**, into buffers kept between splits. It used to allocate and copy all 86 MB of tables per split (chunks 13, state 3, bonds 32, bond records 21, index 1, islands 2, ctab 13.5); allocation and zero fill cost more than the copy. Now it copies each parent's chunk and bond ranges and the per-chunk ctab entries, plus the whole incidence, the islands and the ctab island entries (a moved island can be any island). **Measured:** digests unchanged (63f0bf02 and b89f3650, two runs each). Quick tier p95 26 → 21–22 ms and worst 58 → 54–55 ms. The standard tier was measured while another process (vibe-land's `web-fps-server`, 160–300% CPU) loaded the machine: median 20.4–20.7, p95 42–43, worst 67–70 (from 21, 40–50 and 65–69 under the same load). **Diagnostics:** `STRESS_GPU_TIMING=1` prints each split-pass kernel's GPU time. On the standard tier, `split_components` and `split_children` take 0.6–1.8 ms each for 500–1,100 chunks, `spawn_bodies` 1–1.7, and thread 0 in `layout_parent` up to 1.1. `STRESS_GPU_DET_TRACE=1` prints each cascade window's status read, to find where two runs part. **Open: intermittent divergence under heavy outside GPU load.** While UnrealEditor ran, 2 of 12 standard-tier runs diverged (8a2de626 with 4,234 bonds; a1a16eee with 3,008), with and without the buffer reuse. It has not recurred since: 30 runs, 8 of them two at a time, all 63f0bf02. Metal shader validation finds no out-of-bounds access. Guard 24/24; all 39 suites pass |
| 2026-10-18 | **M4, part 3** | **The coupling's collider table kept on the GPU** (`shaders/colliders.slang`, `src/colliders.rs`, with `STRESS_GPU_LAYOUT=1`), so the host no longer rebuilds it after a split the GPU laid out. The split pass ends with three kernels. `collider_keys` gives each new collider its stable chunk key and marks the parents' colliders retired. `collider_slots` maps each stable chunk to its layout index. `collider_refresh` writes each chunk collider's entry from its chunk's island and that island's first chunk. That entry holds the layout chunk, the body code from the cluster id, mass or infinity when anchored, modulus and radius. Retired colliders become static geometry, as `Owner::Retired` does. The cluster id rides in the island record's spare word (`done_pad`), written by `island_record` and by `layout_parent`. **Check** (`STRESS_GPU_COLLIDER_CHECK=1`): after each GPU-laid-out split, the GPU's table against the host's `collider_table`. Standard tier: 36 tables and 2.04 M entries; quick tier: 13 tables and 0.69 M entries. Every integer word is identical; the props are within 2.3e-7 relative (f32 mass). **Live**, two runs each, with digests unchanged: standard tier 63f0bf02, median 16.7 / p95 36.4–36.8 / worst 60 ms (from 17.0 / 44 / 65). Quick tier b89f3650, 15.6 / 26 / 58–60 ms (from 15.8 / 30 / 63–67). Guard 24/24; all 39 suites pass |
| 2026-10-18 | **M4, part 2** | **One GPU split pass** (`GpuSplitter::pass`, `STRESS_GPU_LAYOUT=1`): `split_components`, `split_children`, `split_frames`, `spawn_offsets`/`spawn_bodies`/`spawn_retire` and `layout_plan`/`layout_moved`/`layout_parent` in one submission with one read. `spawn_offsets` checks capacity and writes an abort flag (`control[0]`); the host then falls back to its in-place split, so no input is silently dropped. `layout_parent` sorts each parent's children by island once (`child_order_at`), so the order lookup is O(1) instead of O(z²) (the meteor split at frame 102 had taken 4.3 s). Bond placement is now count, place, then a parallel copy. Island headroom is `max(2 × clusters, 4096)`, so large splits stay on the GPU. **Supporting changes:** rubble's per-chunk part snapshot is cached; impactor refresh is deferred while resident; the QS tables are not rebuilt when the generation is unchanged; motion sync is skipped in layout mode. **Check** (`STRESS_GPU_SPLIT_CHECK=1`, in-place): 35 (standard) and 12 (quick) GPU layouts against the host's in-place split. Every integer word is identical; island floats are within 1.2e-2 (compensated pairs compared by their sums). **Live**, two runs each: standard tier digest 63f0bf02, median 17.0 / p95 44 / worst 64–65 ms, 2,993 bonds (host path: 0452850b, 17.0 / 200 / 205, 2,998). Quick tier digest b89f3650, 15.8 / 30 / 63–67 ms, 573 bonds (host path: f46991a0, 15.5 / 116 / 209, 572). Guard 24/24; all 39 stress-gpu and stress-rubble suites pass. **Still to do for M4 acceptance:** the per-split host bookkeeping and its read, and the full `ingest.set_colliders` rebuild on engine changes |
| 2026-10-18 | **M4, part 1** | **`shaders/layout.slang` (`layout_plan`, `layout_moved`, `layout_parent`).** The GPU re-lays out a split's tables as the host's `split_resident` does: island order through the host's `swap_remove` sequence, children's chunks, chunk states, bonds, bond records and incidence (cut bonds as padding), static-solve and cut-bond slots cleared, children's island records, the moved island renamed, the resident tables' chunk and island entries, gate flags. **Check** (`STRESS_GPU_SPLIT_CHECK=1` with in-place splits): after the host's in-place split, every chunk, chunk-state, bond, bond-record and incidence word is identical (0 differing) and every child island's integer word is identical. The float fields agree within 1.9e-3 relative (a child's angular velocity, f32 inverse inertia), with compensated pairs compared by their sums. **Live** (`STRESS_GPU_LAYOUT=1`, with the spawn): the GPU writes the live tables from copies of them, and the host keeps only its own layout and mirror in step, splitting the mirror into the GPU's components (`split_cluster_into`): no table read or written. Resident tables and the gate are sized by island capacity (`ResidentQs::adopt_layout`, nothing rebuilt). Quick tier digest 65b62add both runs, 574 bonds broken (host in-place: 574); median 15.7–16.2 ms, p95 31 (host in-place 38–43, rebuild about 106), worst 100–106. Standard tier digest 6bd6b3a9 both runs, 2,989 bonds broken (host in-place: 2,989); median 16.9–17.0 ms, p95 59–62, worst 121–127. Host bookkeeping of the in-place split went from about 13 to about 7 ms (range copies instead of maps; bit-identical: host in-place digest d6738d6e unchanged). Guard 24/24; resident, ingest and catalogue tests pass. **Left for M4:** the split's two read-backs (topology and new bodies) and the host bookkeeping still run inside the frame; they move to the frame's one status read |
| 2026-10-18 | **M3 done** | **`shaders/spawn.slang`.** `spawn_offsets` gives each child its next body and collider slots in child order; parents left whole get none. `spawn_bodies` builds each child's body from the per-chunk part table: mass, centre of mass, inertia, rubble's Jacobi principal axes (`convex::principal_axes` ported), its `BodyState` in both regions, and its colliders. `spawn_retire` retires the parent body and colliders as `remove_body` does. **Supporting code:** `stress-rubble` `cook_with` records each collider's shape as a `Part` (structure frame), cached per chunk. The GPU `PartTable` is indexed by (structure, chunk) and built once. `WgpuSim::adopt_appended`, `mirror_bodies`, `mirror_colliders` and `mirror_retire` keep rubble's host mirrors following GPU-written records with no upload. `ResidentEngine::adopt_spawned` handles rubble's bookkeeping. `cluster_signature` is shared with `world_coupling`. **Check** (`STRESS_GPU_SPLIT_CHECK=1`, standard tier): 635 GPU-made bodies against the host's `create_body`. Collider counts and records are bit-identical; body position within 5.3e-5 m; collider world position within 1.5e-5 m and rotation within 4e-7; inverse mass within 1.8e-7 and world inverse inertia within 4e-5 relative; velocity within 2e-3 m/s. Body orientations differ only where principal axes are degenerate; world quantities agree. Slots follow GPU child order, which the engine adopts. **Live** (`STRESS_GPU_SPAWN=1`): quick tier digest ac16c357 both runs (573 bonds, against 572 when the host creates bodies); standard tier digest 087b035a both runs (2,984 bonds, against 2,998). Guard 24/24; resident tests pass. Split frames still pay the host's mirror sync (18–30 ms) and rebuild (about 57 ms) per pass; that is M4 |
| 2026-10-18 | **M2 done** | **`shaders/split.slang` and `src/gpu_split.rs`.** `split_components` runs one group per parent island: minimum-label connected components over still-connected bonds, then the children in order of their first chunk with their member lists. `split_children` runs one thread per child: mass, centre of mass, inertia, anchoring, and rigid velocity from the chunks' exact world velocities, given each parent's rigid frame. Bond types moved to the shared `bonds.slang` (generated `world.metal` byte-identical). **Check** (`STRESS_GPU_SPLIT_CHECK=1`, every split, both suite tiers): on the standard tier, 64 splits and 639 children with 0 topology and 0 anchoring mismatches; mass 5e-7 relative, centre of mass 5.5e-5 m, inertia 7e-6, velocity 2.4e-4, angular velocity 1e-2 (near-zero spins) against the host's f64. **Bug found and fixed:** the host's rebuild split path split with stale chunk states, because `sync` brings bonds only. `sync_chunk_state` now reads the parents' state first. Quick tier after the fix: digest 1797fb47 both runs, median 15.2 ms, p95 107–109, worst 189–192, 572 bonds broken (vibe-land 638). Guard 24/24; resident and ingest tests pass |
| 2026-10-18 | **M1 done** | **Cause:** rubble appends contact rows at atomic slots (`engine/slang/kernels/world_step.slang:1879, 2323`), so their order varies run to run. The resident ingest summed each chunk's records in row-slot order (`coupling.slang` `ingest_merge`). **Fix:** each record carries a content key (substep, own collider, other collider, feature × 2 + side), and the merge orders by it. The fix is in the shared ingest, so it covers both split paths. **Results, two identical runs each:** `vibe-town_suite_quick` digest 0ffcf577 both runs (median 14.0 ms, p95 102–105, worst 188–190, 513 bonds broken); `vibe-town_suite_bombard` digest 41c25930 both runs (median 15.8–15.9 ms, p95 105–106, worst 188, 2,848 bonds); in-place splits (`STRESS_QS_SPLIT_IN_PLACE=1`) digest c12e91b3 both runs (median 16.3, p95 49–50, worst 108–109, 2,989 bonds). Guard 24/24; stress-gpu tests pass; stress-rubble tests pass after one test fix: `ingest_gpu` now compares manifolds by other body rather than in row order, which is the nondeterministic quantity |
| 2026-10-18 | **M0 done** | **`stress_ref::capability::Profile::BlastParity`** (`crates/stress-ref/src/capability.rs`). Applied through `stress_bench --config blast_parity`, `GpuWorld::apply_profile`, and the scene's `sim.static_tolerance` (new; it reaches the setup solves). Bit-identical to the old flags (`--config blast --authored-fatal --pack-masses --method blast_model=on` with `STRESS_RESIDENT=1 STRESS_QS_BUDGET=1,16 STRESS_GPU_STATICS_TOL=1e-3 STRESS_QS_UNCONVERGED=commit STRESS_QS_PASSES=2`): Vibe Town idle digest b8d04bba both ways, and the suite scene's 31 settle-frame digests identical. The old flags and environment overrides still work (the environment overrides the profile); nothing removed |
| 2026-10-18 | baseline before M0 | quick tier: median 16.6–16.8 ms, p95 120–134, worst 222–277; **not deterministic** (2,920 vs 513 bonds broken) |

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

#### M4: the remaining design (a split frame with no host work beyond its one read)

What still needs the host on a split frame (2026-10-18), and how each piece moves to the GPU:

1. **Dispatch sizes.** The cascade, `resident.slang` and the world kernels are sized by the
   host's island count. They move to island capacity; records past the live count are blank,
   and every kernel checks a live count kept on the GPU.
2. **The cascade's copy of the island records** (`cascade_islands`) is uploaded from the
   host's layout. The split pass's layout keeps it current itself: moved records and the
   children's new ones.
3. **The gate after a split** is re-created, with the host carrying entries by cluster id.
   The GPU layout already moves the gate's island flags and keeps the per-chunk loads where
   the chunks stay, so the gate stays as it is.
4. **The split pass's inputs:** the parents and their ranges from the host's layout, the
   engine frame and retire records, the engine's next slots, the next cluster id. Instead:
   - a kernel builds the parent table from the cascade's status (disconnected islands in
     ascending order, with bases);
   - the ctab engine entries carry each island's body, collider range and frame;
   - GPU counters hold the engine's body and collider counts and the next cluster id;
   - every split kernel is sized by capacity and exits past the live counts.
5. **The world kernels' `halt_index`** (the sentinel, the island count) is copied into their
   params from the GPU count.
6. **One submission per frame:** every pass, each followed by a conditional split pass (a
   no-op when nothing disconnected). The host applies the frame's splits from the one read:
   the mirror's `split_cluster_into` and the engine's `adopt_spawned`, in order.
7. **Results:** the children's island records become the GPU's (f32), not the host's
   `island_record` (f64, then rounded). That needs the same check against the host as M2:
   integers equal, floats within a stated tolerance. Any loss is to be measured; inverses go
   in compensated arithmetic where f32 falls short.

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

#### M5: the design, by your decision of 2026-10-18 (routing on the GPU)

Forward quasi-static splits cannot reach the fidelity bar under `blast_parity` in any form
measured (log, "M5 investigation"). Rubble's rigid contact exchanges the ram's whole momentum
with the intact wall in the substep of contact. The coupled frame's routing reaches the oracle
in one forward pass: it puts the struck structure in an impact island before the engine steps.
The profile is `blast_parity_routed`: `blast_parity` plus routing and impact islands, under
the tiered coupling. The pieces:

1. **Promotion is its own flag** (`promote_fractures`, `Profile::promote_fractures`,
   `STRESS_PROMOTE`). Promotion turns every cluster of a structure that fractured under the
   engine's loads into an island seed for six frames. It is off in the routed profile: only an
   impactor's contact routes. With promotion on, the town settle put every cluster in islands
   for six frames of 65–121 s each.
2. **The routing check on the GPU** (`shaders/routing.slang`, `src/routing.rs`). At the end of
   every resident frame, in the same submission and the same read, the check asks whether the
   next frame may route an island. It uses the engine's bodies where the frame left them, and
   a superset of the host's seed test:
   - an impactor's swept sphere against each bonded island's swept bounds (its chunks at its
     pose);
   - a closing-speed bound above the tiered impact speed.
3. **Frames switch between the resident and the coupled path.** A frame runs resident unless
   the last frame's check is set, or a host prediction is set after a coupled frame
   (`World::predict_islands`, exact). The coupled frame takes the engine's motion first
   (`World::take_engine_motion`), then routes exactly on the host. Because the check is a
   superset, every frame the host's routing would give an island takes the coupled path.
   `STRESS_ROUTED_RESIDENT=0` makes every routed frame coupled.
4. **Island frames in the resident submission.** Today an island frame is the coupled path:
   host-orchestrated substeps at the explicit stress step, several submissions and reads.
   M5's acceptance counts them (`coupled_frames`). Moving them into one submission, and
   making them fast, is next:
   - the island's substeps as a GPU loop over many substeps;
   - the island bounded to the impact region.

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
| 2026-10-18 | **M5, part 3: what an island frame costs** | Free-wall impact frame (routed, 277 ms): about 5,350 substeps at the wall's certified stress step (6.7e-6 s, then 1.2e-6 s once the ram's 236 contact pairs engage), in 8 host-planned segments with 2 host splits. That is about 50 µs per substep. Real time (16 ms) needs about 3 µs. REALTIME_ROADMAP.md A6 measured the floors on this M3 Max: the full island model about 24 µs per substep whatever the size (latency, not work); an intact-only kernel running many substeps per dispatch, 2.4–10 µs; a persistent contact pipeline in one threadgroup was slower (entry 15). So even the fastest measured kernel puts this frame at 13–54 ms before its contact substeps. One submission per island frame also means moving the segment loop, its contact plans, re-plans and host splits onto the GPU. **Decision needed** on how far to take island frames |
| 2026-10-18 | **M5, part 2: the island's extent, measured** (your decisions of 2026-10-18: the town pack is to be re-exported at its source; a fast-join rule behind a flag, measured before adoption) | `World::tiered_fast_joins` (`STRESS_TIERED_JOIN=fast`, off by default): under the tiered coupling only a pair with an impactor, or one closing faster than the tiered impact speed, joins islands; resting contacts stay the engine's. **Measured:** free wall unchanged (34.999 m/s, 3 fragments, digest 7e6ced7f identical); wall row same ram, fragments and bonds (31.65 m/s, 41, 199), digest cdae23c8 against 5f69613e. Town quick tier, first shot (frame 31): 55 clusters in the island against 321, but its contact plan holds 467,609 chunk pairs (183 cluster pairs, 51,921 chunks in the tables), and `contact_forces` needs 236,335 groups, past wgpu's 65,535; with the assets' stress step (about 1e-7 s near the debris) the frame would also need about 10^5 substeps. The substep loop now fails with that dispatch named and its counts, in place of wgpu's validation panic. **Not adopted** (your decision). Routed island frames in the town wait for the re-exported pack |
| 2026-10-18 | **M5, part 1: routed frames run resident between impacts** | **Design** (above, "M5: the design"). Promotion is a flag of its own, off in `blast_parity_routed`. A GPU check at the end of each resident frame, in its submission and its read (`routing.slang`), asks whether the next frame may route an island. Its test is a superset of the host's seed test: an impactor's swept sphere against each bonded island's swept bounds, with a closing-speed bound. A flagged frame takes the coupled path and routes exactly on the host; after it, the host predicts the next frame exactly (`predict_islands`). **Fidelity** (ram exit speed / fragments; oracle; correction-limit-1): free wall (`wall_ram_late`) 34.999 m/s / 3 (35.0 / 6; 10.9 / 23); wall row of two walls at 40 m/s (`wall_row2_late`, new) 31.65 / 41 (23.6 / 133; 4.9 / 240). Both are closer to the oracle than correction-limit-1 on both measures. `blast_parity` on the wall row: 10.17 / 43, also closer. Same ram, fragments and bonds as every routed frame coupled (`STRESS_ROUTED_RESIDENT=0`); the digests differ, as a resident frame's f32 solve does from a coupled one. **Deterministic:** two runs each, identical digests (7e6ced7f, 5f69613e). **Cost:** quiet routed frames 2.5–6 ms (the town's were about 220 ms coupled). Island frames stay host-coupled: free wall 277, 108, 54 ms; wall row 291, 610, 317 ms. **Town:** frames 1–30 now run resident. The first shot (frame 31) routes 321 clusters into one island, since routing joins every cluster whose swept bounds touch it, and a dispatch then exceeds wgpu's 65,535 groups. The town's assets set its stress step: 159 structures certify below 1 µs (2 mm road markings 3e-7 s, mailbox flags 1e-7 s, timber posts 7e-7 s), so any island there needs 20k–100k substeps per frame. **Left for M5:** island frames in one submission and fast; the town needs a decision on its assets (re-export from Shardwright) and on the island's extent |
| 2026-10-18 | **M6 done** | **Your decision of 2026-10-18:** `blast_parity` runs vibe-land's PhysX policy, one engine step a frame (no CCD substeps) and 4 solver iterations (`Profile::engine_policy`, `RubbleEngine::set_policy`). The profile also turns on the GPU's split path by itself, with no flags needed (`GPU_FRAME_PROFILE`; `STRESS_GPU_LAYOUT` and `STRESS_GPU_FRAME` still override, 1 on, 0 off). **Lossless:** the ingest's merge runs over only the substeps run, and in each only over its rows (`row_counts`, `record_live`). It had cleared and scanned every slot for every substep it had room for, which is four now. Digests unchanged. **Results**, two runs each, deterministic: standard tier median 10.4, p95 15.9–16.7, worst 17.6–18.8 ms (targets 11.46 / 37.88 / 48.24), digest 37cd230c, 6,289 bonds (vibe-land 6,707). Quick tier 7.8 / 11.0–11.1 / 14.7–15.3 ms (targets 11.38 / 15.32 / 35.5), digest 46728bf1, 1,329 bonds. Every target is met. A per-island backup kernel instead of the whole-table copies was measured 0.8 ms slower and stays opt-in (`STRESS_QS_ISLAND_BACKUP=1`). Guard 26/26; all 39 suites pass. **Next:** M5, by your decision: impact-island routing in the GPU-resident frame. `blast_parity_routed` (routing plus impact islands on the host-coupled path) already gives the free-wall ram 34.99 m/s and 3 fragments (oracle 35.0 and 6), but costs 270 ms on its impact frame |
| 2026-10-18 | **M6: more impact-spike work** | (a) An empty `set_motion` no longer reads the bodies: the deferred flush's write-back had been waiting on the engine's queued substeps (8–12 ms), which undid the overlap. Digests unchanged. (b) The ingest starts with room for 4 substeps. Its growth on the first 3-substep frame (the first shot) had zero-filled its buffers in that frame's ingest, 12 ms; quick worst 32 → 21 ms, digests unchanged. (c) `split_children_group` and `spawn_bodies_group` (`STRESS_GPU_SPLIT_TREE`, on in the frame's split pass and the M3 check): one group per child slot, with the sums in a fixed tree instead of one thread per child. `split_children` 1.0 → under 0.05 ms per pass, digests unchanged. `spawn_bodies` 1.3 ms → under 0.05; f32 rounding changes, so the digests move: standard tier a69ab23b (2,988 bonds), quick d0b1d964 (572), deterministic. Against the host's `create_body` the bodies are as close or closer (position 2.2e-5 m against 5.3e-5 for the one-thread sums). **Results:** standard tier median 17.1–17.2, p95 22.3–22.8, worst 29.8–31.5 ms; quick tier 16.0 / 18.7 / 20.9 ms. Guard 26/26; all 39 suites pass. The worst frame is now GPU-bound: rubble's three substeps 17.5 ms, two split passes about 3 ms each, the cascade 3.8 |
| 2026-10-18 | **M6: impact spikes, the split pass made parallel** (lossless: identical digests) | `STRESS_GPU_TIMING=1` now times each step of a frame-mode split pass. On the worst standard-tier frame (109) two passes took 10.5 ms of GPU each, against rubble's three substeps at 5.8 ms. Now parallel, with the same integer results: (a) `layout_parent` (4.0 → 1.5 ms, 256 threads): the new chunk positions (each child's start in island order, then every member), the bonds' new indices (ranks by class in blocks), and the incidence offsets (a scan). (b) `spawn_offsets` is one group of scans over the members' collider counts and the valid child slots, instead of one thread. The colliders go to `spawn_colliders` (per member), and `spawn_retire` runs one group per parent. (c) `collider_keys` runs per member. (d) `pass_snapshot` uses 256 threads. The child bodies' float sums in `spawn_bodies` keep their order. **Check fix:** a frame log's collider check now runs after its last split, or after the deferred flush. Every integer word is identical in both modes. **Results**, two runs each with the digests unchanged (66c9e746 and 0ea38b6f): standard tier median 17.2–17.3, p95 25.1–25.7, worst 34.5–35.2 ms (from 45.5–46.8); quick tier 16.2–16.3 / 18.6–18.7 / 32.0–32.9 ms (from 36.6–37.2). Quick worst now meets 35.5. Guard 26/26; all 39 suites pass. Still short under today's policy: the medians, and quick p95 (18.6 against 15.32), which rubble's CCD substeps set (see the policy row) |
| 2026-10-18 | **M6: the CCD and iteration policy, measured (not adopted)** | Frame mode (`STRESS_GPU_LAYOUT=1 STRESS_GPU_FRAME=1`), two runs each, each deterministic. A rubble substep is about 4.3 ms: 10 primal iterations at about 0.23 ms each, broadphase 0.6, narrowphase 0.26. **Today's policy** (CCD substeps, 10 iterations): standard 17.1 / 27.0 / 46.6 ms, 2,995 bonds; quick 16.0 / 18.7 / 37.2 ms. **No CCD** (`STRESS_RUBBLE_MAX_SUBSTEPS=1`, one substep a frame as vibe-land's PhysX runs): standard 8.8–9.0 / 22.9–23.7 / 38.6–38.7 ms, 6,193 bonds; quick 7.9–8.0 / 12.7–13.0 / 22.2–22.3 ms, 1,028 bonds. Every target met. **No CCD and 4 iterations** (vibe-land's PhysX settings): standard 10.6–10.9 / 20.0–20.2 / 22.6–22.7 ms, 6,298 bonds; quick 7.7–8.0 / 12.7–12.8 / 18.9–19.5 ms, 1,324 bonds. Every target met, and the breakage is near vibe-land's 6,707 on the standard tier. **4 iterations alone:** standard 14.0 / 26.8 / 35.6 ms; quick 13.1 / 17.0 / 36.9 ms. **Speculative contacts** (`STRESS_RUBBLE_SPECULATIVE=1`): the standard tier overflows rubble's capacity, and the quick tier is slower (25 ms median). Dropping CCD lets thin walls tunnel at high speed (section 4 of the plan). **This is your decision:** neither policy is adopted |
| 2026-10-18 | **M5 investigation: forward splits and the free-wall ram** | **Setup:** the bench's `wall_ram` strikes during frame 0, which is never resident (`fast_qs_conditions` requires `frame > 0`), so its impact ran the host's coupled pass. `wall_ram_late` starts the ram 1.1 m back, so it strikes in resident frame 1 (14 CCD substeps). **Measured** (ram exit speed and fragments; oracle 35.0 m/s and 6; correction-limit-1 10.9 m/s and 23): (a) `blast_parity` frame mode: 8.6 m/s, 1 fragment, no bond broken. The impact frame's load is the impulse averaged over the frame (J/dt, about 300 kN), and the free wall's inertia relief keeps every bond under strength, even with converged solves (budgets up to 60 Newton and 20,000 CG). (b) Each impact at its Hertz peak force (`STRESS_QS_IMPACT_PEAK=1`, a measurement): 413 bonds, 131 fragments, still 8.6 m/s. (c) Frames of 1/240 s (`STRESS_BENCH_FRAME_DT`: a split, and the engine's adoption of it, after every quarter of the impact frame): 94 bonds and 2 fragments at 8.74 m/s (peak loads: 131 fragments, 8.73 m/s). **Finding:** splitting between substeps barely moves the ram. Rubble's rigid contact exchanges the whole momentum against the intact wall in the substep of contact, and a forward split can only follow that substep. This is `findings/rewind-scaling.md`'s rule, that the struck body exchanges momentum only with the piece that breaks free. The default configuration's coupled frame, which routes the wall into an impact island before the engine steps, reaches the oracle (35.08 m/s, 6 fragments) in one forward pass with no redo. So M5's fidelity bar is out of reach for forward quasi-static splits under `blast_parity` in any form measured here. It needs impact islands (routing, the higher-fidelity capability) in the resident frame, or another load model; that is a decision for you. **Diagnostics added:** `wall_ram_late`, `STRESS_BENCH_FRAME_DT`, `STRESS_QS_IMPACT_PEAK`, `STRESS_GPU_QS_LOADS` |
| 2026-10-18 | **M4, part 8: the host's bookkeeping off the GPU's path** | After a frame's read, the engine adopts the GPU's bodies at once, named by the GPU's cluster ids (`layout_plan`'s order) with signatures from the logged chunks. The mirror's `split_cluster_into` and the layout bookkeeping wait for the next frame, which runs them after submitting its engine substeps, while those run (`flush_pending`, `STRESS_GPU_FRAME_DEFER=0` to follow them at once). The engine write-back waits with them. Anything else that reads the mirror flushes first: `sync`, `add_impactor`, any non-resident frame. The log now describes each entry by itself, from the tables before its split: each parent's cluster id and structure, and each member as its chunk within its structure. The host can then read every entry of a frame before following any. The flush checks that the mirror names the children as the GPU did. **Results:** digests unchanged (66c9e746 and 0ea38b6f, three runs each). **Standard tier:** median 17.1–17.3 ms, p95 27.0–27.3 (target 37.88, met), worst 45.5–46.8 (target 48.24, met). **Quick tier:** 16.0–16.1 / 18.7–19.0 / 36.6–37.2 (targets 11.38 / 15.32 / 35.5). Each frame now records its GPU wait (`frame_wait_ms`); the rest is host time. Worst standard-tier frame 109: 46.8 ms, of which 42.3 ms waiting on the GPU. Median host time 1.85 ms. Guard 24/24; all 39 suites pass. **M4 status:** a split frame's cascade and splits are one submission and one read; the host's work after it is off the GPU's path; results are deterministic and equal the per-split GPU path. Left for M5: the engine's substeps are their own submissions, and splits happen between frames rather than between substeps. Left for M6: the medians, which rubble's three CCD substeps set at about 4.3 ms each |
| 2026-10-18 | **M4, part 7: the frame's cascade and splits in one submission** (`STRESS_GPU_FRAME=1`, with `STRESS_GPU_LAYOUT=1`) | **Design steps 4–6.** (4) The split pass's counts are GPU control words (`CTL_*`, common.slang) rather than uniforms. Its inputs are made on the GPU (`shaders/pass.slang`, `src/gpu_pass.rs`): `pass_parents` builds the parent table, bases, engine records and frame data from the cascade's status and the ctab engine entries. The ctab entries now also carry each island's engine body, fixed ones included, and its collider range. `pass_snapshot` and `pass_whole` copy what the layout reads, and only when something splits. Every size-dependent dispatch is indirect, and `pass_finish` advances GPU counters (live islands, engine bodies and colliders, next cluster id) and the cascade's island arguments. With `STRESS_GPU_PASS=1` in the per-split flow the digests are identical to the host-made inputs. (6) `resident_cascade_with` records a split pass after every cascade pass: one compute pass of prepared bind groups, a no-op unless that pass disconnected an island. `passlog.slang` appends each split to a frame log, which the frame's one read takes along with the status (the first 256 KB; a longer log is read separately). The host then follows the splits in order (`apply_frame_log`: the mirror's `split_cluster_into`, the layout bookkeeping, `adopt_spawned`). A split that does not fit stops the frame's GPU cascade, and the per-window path takes over. **Fixes found on the way:** the persistent child slots of a parent left whole are cleared, and `spawn_bodies` honours the abort flag (stale slots could have made bodies). **Results:** digests are identical to the per-split GPU path on both tiers (66c9e746 and 0ea38b6f) and repeat run to run. Standard tier median 17.1–17.2, p95 29.8–30.7, worst 51.7–51.8 ms; quick tier 16.0–16.1 / 19.8–19.9 / 38.8 ms. Before frame mode: 17.1 / 31.6 / 54 and 15.8 / 21.3–22.3 / 41. **Host work on a split frame now:** the one read, then the bookkeeping that follows the splits (the mirror and the engine's host tables), off the GPU's path. The engine's substeps are still their own submissions; that is M5. Guard 24/24; all 39 suites pass |
| 2026-10-18 | **M4, part 6** | **Steps 1–3 of the remaining design.** (1) With `STRESS_GPU_LAYOUT=1` the cascade, `resident.slang` and the world kernels dispatch indirectly from `island_args`, a GPU buffer of live-island group counts that the host writes for now. Their params and buffers span the island capacity; blank islands past the live ones are passed over, and their ctab engine entries are NONE. `STRESS_QS_CAPACITY=0` gives the old dispatches. Digests are unchanged, and the cost stays at the baseline (dispatching the capacity directly had cost about 0.5 ms a frame). (2) `layout_copy` keeps the cascade's island copy (`cascade_islands`) on the GPU through a split. Kept and moved records come from the copy as it was, children's from `layout_parent`, and the sentinel is blank; the host no longer re-uploads it. (3) The gate is kept through a GPU-laid-out split; the layout moves its island flags. **Results change**, because the cascade now restores the GPU's island records rather than records the host rebuilt from its mirror. `STRESS_GPU_ISLAND_CHECK=1` compares the two and shows where they differ: a free child's position (the host folds the mean hidden displacement with chunk states its mirror has not synced in layout mode; the GPU folds the live ones); a disconnected parent left whole (the host rebuilds it as a child, dropping its quiet flag and probes; the GPU keeps it); and motion fields that every pass retakes from the engine. **Fidelity fix:** an anchored parent's split frame now comes from its island record (its pose, as `split_cluster` takes it), not from its fixed engine body; that frame could sit millimetres off. `STRESS_GPU_ANCHORED_ENGINE_FRAME=1` gives the old frame. **Live, two runs each:** standard tier digest 66c9e746, 2,995 bonds broken (from 2,993), median 17.1 / p95 31.6–31.8 / worst 54 ms. Quick tier digest 0ea38b6f, 574 bonds (from 573), 15.8–15.9 / 21.3–22.3 / 41 ms. Guard 24/24; all 39 suites pass |
| 2026-10-18 | **M4, part 5** | **Engine-only changes no longer rebuild the frame's tables.** Rubble logs each collider range it changes (`colliders_changed`, `ResidentEngine::collider_changes_since`): added, retired or adopted. Under an unchanged layout, the resident frame then rewrites only those entries of the collider table and their keys (`patch_collider_table`, with lookups cached per layout generation). A frame needing more substeps than the ingest holds now grows only its per-substep buffers (`EngineIngest::grow_substeps`). It used to make the ingest and the QS tables anew: 14 ms on the first shot frame. `STRESS_COLLIDER_PATCH=0` and `STRESS_INGEST_GROW=0` give the old behaviour. **Check:** `STRESS_GPU_COLLIDER_CHECK=1` compares every patched table with a full rebuild: 1 patch on the quick tier and 4 on the standard tier, every integer word identical. Digests are unchanged on both paths (GPU path 63f0bf02 and b89f3650; host path 0452850b and f46991a0, also with both switches off). **Live, quiet machine, two runs each:** standard tier median 16.8–17.0, p95 31.2–31.4 (target ≤ 37.88: met), worst 51–55 ms. Quick tier 15.7–15.9 / 22.3–22.6 / 41.5 ms. **Determinism toolkit** (env-gated diagnostics, reads of their own unless noted): `STRESS_GPU_DET_TRACE` (each window's status, no read), `STRESS_GPU_DET_END` (ingest output and engine bodies, in the frame's own read), `STRESS_GPU_DET_BODIES`, `STRESS_GPU_DET_STAGE=<frame>`, `STRESS_GPU_DET_REPLAY=<frame>` (the cascade's first window re-run from saved buffers) and `STRESS_GPU_DET_INGEST=<frame>` (the ingest re-run). **Open, intermittent divergence:** it recurred only while UnrealEditor kept the GPU at 99% (about 1 in 4 to 1 in 10 runs, on both split paths; once at frame 3, once at 4, once at 87). In a diverged run the engine's bodies were identical through the diverging frame and that frame's cascade verdicts differed, so the rubble step is ruled out. In isolation every stage repeats bit for bit: 40 cascade replays, 60 ingest replays, unique merge keys, no out-of-bounds access under Metal validation. 70 runs without the outside load, 40 of them two at a time, all agree. Still to find: the race between the ingest and the cascade that only outside GPU load exposes. Guard 24/24; all 39 suites pass |
| 2026-10-18 | **M4, part 4** | **The split pass copies only what its layout reads**, into buffers kept between splits. It used to allocate and copy all 86 MB of tables per split (chunks 13, state 3, bonds 32, bond records 21, index 1, islands 2, ctab 13.5); allocation and zero fill cost more than the copy. Now it copies each parent's chunk and bond ranges and the per-chunk ctab entries, plus the whole incidence, the islands and the ctab island entries (a moved island can be any island). **Measured:** digests unchanged (63f0bf02 and b89f3650, two runs each). Quick tier p95 26 → 21–22 ms and worst 58 → 54–55 ms. The standard tier was measured while another process (vibe-land's `web-fps-server`, 160–300% CPU) loaded the machine: median 20.4–20.7, p95 42–43, worst 67–70 (from 21, 40–50 and 65–69 under the same load). **Diagnostics:** `STRESS_GPU_TIMING=1` prints each split-pass kernel's GPU time. On the standard tier, `split_components` and `split_children` take 0.6–1.8 ms each for 500–1,100 chunks, `spawn_bodies` 1–1.7, and thread 0 in `layout_parent` up to 1.1. `STRESS_GPU_DET_TRACE=1` prints each cascade window's status read, to find where two runs part. **Open: intermittent divergence under heavy outside GPU load.** While UnrealEditor ran, 2 of 12 standard-tier runs diverged (8a2de626 with 4,234 bonds; a1a16eee with 3,008), with and without the buffer reuse. It has not recurred since: 30 runs, 8 of them two at a time, all 63f0bf02. Metal shader validation finds no out-of-bounds access. Guard 24/24; all 39 suites pass |
| 2026-10-18 | **M4, part 3** | **The coupling's collider table kept on the GPU** (`shaders/colliders.slang`, `src/colliders.rs`, with `STRESS_GPU_LAYOUT=1`), so the host no longer rebuilds it after a split the GPU laid out. The split pass ends with three kernels. `collider_keys` gives each new collider its stable chunk key and marks the parents' colliders retired. `collider_slots` maps each stable chunk to its layout index. `collider_refresh` writes each chunk collider's entry from its chunk's island and that island's first chunk. That entry holds the layout chunk, the body code from the cluster id, mass or infinity when anchored, modulus and radius. Retired colliders become static geometry, as `Owner::Retired` does. The cluster id rides in the island record's spare word (`done_pad`), written by `island_record` and by `layout_parent`. **Check** (`STRESS_GPU_COLLIDER_CHECK=1`): after each GPU-laid-out split, the GPU's table against the host's `collider_table`. Standard tier: 36 tables and 2.04 M entries; quick tier: 13 tables and 0.69 M entries. Every integer word is identical; the props are within 2.3e-7 relative (f32 mass). **Live**, two runs each, with digests unchanged: standard tier 63f0bf02, median 16.7 / p95 36.4–36.8 / worst 60 ms (from 17.0 / 44 / 65). Quick tier b89f3650, 15.6 / 26 / 58–60 ms (from 15.8 / 30 / 63–67). Guard 24/24; all 39 suites pass |
| 2026-10-18 | **M4, part 2** | **One GPU split pass** (`GpuSplitter::pass`, `STRESS_GPU_LAYOUT=1`): `split_components`, `split_children`, `split_frames`, `spawn_offsets`/`spawn_bodies`/`spawn_retire` and `layout_plan`/`layout_moved`/`layout_parent` in one submission with one read. `spawn_offsets` checks capacity and writes an abort flag (`control[0]`); the host then falls back to its in-place split, so no input is silently dropped. `layout_parent` sorts each parent's children by island once (`child_order_at`), so the order lookup is O(1) instead of O(z²) (the meteor split at frame 102 had taken 4.3 s). Bond placement is now count, place, then a parallel copy. Island headroom is `max(2 × clusters, 4096)`, so large splits stay on the GPU. **Supporting changes:** rubble's per-chunk part snapshot is cached; impactor refresh is deferred while resident; the QS tables are not rebuilt when the generation is unchanged; motion sync is skipped in layout mode. **Check** (`STRESS_GPU_SPLIT_CHECK=1`, in-place): 35 (standard) and 12 (quick) GPU layouts against the host's in-place split. Every integer word is identical; island floats are within 1.2e-2 (compensated pairs compared by their sums). **Live**, two runs each: standard tier digest 63f0bf02, median 17.0 / p95 44 / worst 64–65 ms, 2,993 bonds (host path: 0452850b, 17.0 / 200 / 205, 2,998). Quick tier digest b89f3650, 15.8 / 30 / 63–67 ms, 573 bonds (host path: f46991a0, 15.5 / 116 / 209, 572). Guard 24/24; all 39 stress-gpu and stress-rubble suites pass. **Still to do for M4 acceptance:** the per-split host bookkeeping and its read, and the full `ingest.set_colliders` rebuild on engine changes |
| 2026-10-18 | **M4, part 1** | **`shaders/layout.slang` (`layout_plan`, `layout_moved`, `layout_parent`).** The GPU re-lays out a split's tables as the host's `split_resident` does: island order through the host's `swap_remove` sequence, children's chunks, chunk states, bonds, bond records and incidence (cut bonds as padding), static-solve and cut-bond slots cleared, children's island records, the moved island renamed, the resident tables' chunk and island entries, gate flags. **Check** (`STRESS_GPU_SPLIT_CHECK=1` with in-place splits): after the host's in-place split, every chunk, chunk-state, bond, bond-record and incidence word is identical (0 differing) and every child island's integer word is identical. The float fields agree within 1.9e-3 relative (a child's angular velocity, f32 inverse inertia), with compensated pairs compared by their sums. **Live** (`STRESS_GPU_LAYOUT=1`, with the spawn): the GPU writes the live tables from copies of them, and the host keeps only its own layout and mirror in step, splitting the mirror into the GPU's components (`split_cluster_into`): no table read or written. Resident tables and the gate are sized by island capacity (`ResidentQs::adopt_layout`, nothing rebuilt). Quick tier digest 65b62add both runs, 574 bonds broken (host in-place: 574); median 15.7–16.2 ms, p95 31 (host in-place 38–43, rebuild about 106), worst 100–106. Standard tier digest 6bd6b3a9 both runs, 2,989 bonds broken (host in-place: 2,989); median 16.9–17.0 ms, p95 59–62, worst 121–127. Host bookkeeping of the in-place split went from about 13 to about 7 ms (range copies instead of maps; bit-identical: host in-place digest d6738d6e unchanged). Guard 24/24; resident, ingest and catalogue tests pass. **Left for M4:** the split's two read-backs (topology and new bodies) and the host bookkeeping still run inside the frame; they move to the frame's one status read |
| 2026-10-18 | **M3 done** | **`shaders/spawn.slang`.** `spawn_offsets` gives each child its next body and collider slots in child order; parents left whole get none. `spawn_bodies` builds each child's body from the per-chunk part table: mass, centre of mass, inertia, rubble's Jacobi principal axes (`convex::principal_axes` ported), its `BodyState` in both regions, and its colliders. `spawn_retire` retires the parent body and colliders as `remove_body` does. **Supporting code:** `stress-rubble` `cook_with` records each collider's shape as a `Part` (structure frame), cached per chunk. The GPU `PartTable` is indexed by (structure, chunk) and built once. `WgpuSim::adopt_appended`, `mirror_bodies`, `mirror_colliders` and `mirror_retire` keep rubble's host mirrors following GPU-written records with no upload. `ResidentEngine::adopt_spawned` handles rubble's bookkeeping. `cluster_signature` is shared with `world_coupling`. **Check** (`STRESS_GPU_SPLIT_CHECK=1`, standard tier): 635 GPU-made bodies against the host's `create_body`. Collider counts and records are bit-identical; body position within 5.3e-5 m; collider world position within 1.5e-5 m and rotation within 4e-7; inverse mass within 1.8e-7 and world inverse inertia within 4e-5 relative; velocity within 2e-3 m/s. Body orientations differ only where principal axes are degenerate; world quantities agree. Slots follow GPU child order, which the engine adopts. **Live** (`STRESS_GPU_SPAWN=1`): quick tier digest ac16c357 both runs (573 bonds, against 572 when the host creates bodies); standard tier digest 087b035a both runs (2,984 bonds, against 2,998). Guard 24/24; resident tests pass. Split frames still pay the host's mirror sync (18–30 ms) and rebuild (about 57 ms) per pass; that is M4 |
| 2026-10-18 | **M2 done** | **`shaders/split.slang` and `src/gpu_split.rs`.** `split_components` runs one group per parent island: minimum-label connected components over still-connected bonds, then the children in order of their first chunk with their member lists. `split_children` runs one thread per child: mass, centre of mass, inertia, anchoring, and rigid velocity from the chunks' exact world velocities, given each parent's rigid frame. Bond types moved to the shared `bonds.slang` (generated `world.metal` byte-identical). **Check** (`STRESS_GPU_SPLIT_CHECK=1`, every split, both suite tiers): on the standard tier, 64 splits and 639 children with 0 topology and 0 anchoring mismatches; mass 5e-7 relative, centre of mass 5.5e-5 m, inertia 7e-6, velocity 2.4e-4, angular velocity 1e-2 (near-zero spins) against the host's f64. **Bug found and fixed:** the host's rebuild split path split with stale chunk states, because `sync` brings bonds only. `sync_chunk_state` now reads the parents' state first. Quick tier after the fix: digest 1797fb47 both runs, median 15.2 ms, p95 107–109, worst 189–192, 572 bonds broken (vibe-land 638). Guard 24/24; resident and ingest tests pass |
| 2026-10-18 | **M1 done** | **Cause:** rubble appends contact rows at atomic slots (`engine/slang/kernels/world_step.slang:1879, 2323`), so their order varies run to run. The resident ingest summed each chunk's records in row-slot order (`coupling.slang` `ingest_merge`). **Fix:** each record carries a content key (substep, own collider, other collider, feature × 2 + side), and the merge orders by it. The fix is in the shared ingest, so it covers both split paths. **Results, two identical runs each:** `vibe-town_suite_quick` digest 0ffcf577 both runs (median 14.0 ms, p95 102–105, worst 188–190, 513 bonds broken); `vibe-town_suite_bombard` digest 41c25930 both runs (median 15.8–15.9 ms, p95 105–106, worst 188, 2,848 bonds); in-place splits (`STRESS_QS_SPLIT_IN_PLACE=1`) digest c12e91b3 both runs (median 16.3, p95 49–50, worst 108–109, 2,989 bonds). Guard 24/24; stress-gpu tests pass; stress-rubble tests pass after one test fix: `ingest_gpu` now compares manifolds by other body rather than in row order, which is the nondeterministic quantity |
| 2026-10-18 | **M0 done** | **`stress_ref::capability::Profile::BlastParity`** (`crates/stress-ref/src/capability.rs`). Applied through `stress_bench --config blast_parity`, `GpuWorld::apply_profile`, and the scene's `sim.static_tolerance` (new; it reaches the setup solves). Bit-identical to the old flags (`--config blast --authored-fatal --pack-masses --method blast_model=on` with `STRESS_RESIDENT=1 STRESS_QS_BUDGET=1,16 STRESS_GPU_STATICS_TOL=1e-3 STRESS_QS_UNCONVERGED=commit STRESS_QS_PASSES=2`): Vibe Town idle digest b8d04bba both ways, and the suite scene's 31 settle-frame digests identical. The old flags and environment overrides still work (the environment overrides the profile); nothing removed |
| 2026-10-18 | baseline before M0 | quick tier: median 16.6–16.8 ms, p95 120–134, worst 222–277; **not deterministic** (2,920 vs 513 bonds broken) |

# Rubble performance audit: close the three-avbd gap, then pass everyone

## Context

**Goal:** rubble (`engine/`, one Slang source → WGSL for WebGPU, MSL for Metal, CUDA later) must beat
every open GPU engine and hold ≥120 FPS (≤8.33 ms/step) with 100,000s of active rigid bodies.
- **Hardware:** M3 Max (Metal and Chrome WebGPU) now; RTX 4090 (CUDA) later.
- **Scope:** spheres, capsules, boxes, hulls, planes, compounds, joints (incl. ropes) and motors. No soft bodies.

**Source of truth for this audit:** branch `integrate/all-worktrees` @ 26b269c. Exception: the D6 joint
rewrite, which is still uncommitted on `feat/avbd-attempt-2`.
**Evidence:**
- the joint-parity benchmark (`bench/results/joints-all`, `bench/results/calib`)
- three-avbd local baselines (`engine/docs/baselines/three-avbd-oracle-baseline-2026-10-13.md`)
- rubble's design log (`engine/DECISIONS.md`, about 60 rejected trials)

**What was audited:**
- three-avbd (pinned at upstream HEAD 2bb1a31), kernel by kernel: its hand-written WGSL against
  rubble's *generated* WGSL (`target/.../out/wgsl/*.wgsl`, plus the `engine-webgpu/src/lib.rs` post-pass)
- the host orchestration of both engines
- Box3D, PhysX 5 GPU, Jolt and MuJoCo/MJX/MJWarp, for the structural roadmap (P2)

### Where rubble stands vs three-avbd (M3 Max, same scene files)

| scene | three-avbd (Dawn) | rubble Metal (bench) | rubble native wgpu (bench) | rubble Chrome host |
|---|---:|---:|---:|---:|
| ring110k @4 it | 10.8 ms | 16.2 | 16.6 | ~10.2 |
| gables506k @3 it | 26.6 ms | ~49 | ~50 | ~27 |
| ring 9k @4 it | 1.75 ms | — | — | ~2.3 |
| chain mail 1.6k + 3.1k joints @10 it | 1.13–1.30 ms | 1.57 | (no joint support) | (no joint scene) |

**The gap comes from four sources, in order of size:**
1. **Host/config parity (P0).** The fast paths exist only in the browser host, and only for the two big scenes.
2. **Dispatch count (P1-A).** Rubble encodes 2–3× more dispatches per step, most of them empty.
3. **Small/medium scenes don't get the big-scene data paths (P1-B).**
4. **Slang feature creep and codegen artefacts in hot kernels (P1-C).**

Rubble is already better on:
- the broadphase reject path, split half-stencil grid and two-pass narrowphase
- convex hulls (cooperative SAT, feature cache)
- incremental colouring with local repair, CSR partner array, stable colour tiles
- lower-triangle 6×6 accumulation, sleep and rest carry
- GPU-driven sizing with no truncation

## P0 — One fast configuration on every host (largest and cheapest)

1. **Native wgpu: flip the defaults.** Every October optimization is opt-in via `ENGINE_*` env vars and off
   by default (`engine-webgpu/src/gpu.rs:155-462`):
   - persistent contacts
   - `REUSE=copy`, `NP=split`, `PERSIST=list`
   - `REPAIR=local`, `REST=carry`
   - `HASH_INSERT=record`, `GRID=split`

   Flip the defaults and keep the env vars as controls.
2. **Browser host: one default for all scenes.** The 110k collision/contact modes are enabled only for
   `ring110k`/`gables506k` (`demo.html:103-181`); ring 9k still runs 128 B canonical contacts,
   per-contact hash insert, a fused narrowphase and the legacy two-sided grid. Make them the defaults
   for all scenes, after the existing gates pass at 9k, wall smash and columns.
3. **Metal host: port to the same kernel sequence.** `engine-metal/src/world.rs:779-990` is the old pipeline:
   - canonical narrowphase
   - per-contact hash insert
   - 32 global JP rounds, with no local repair (HOST_S(27) slot conflict, DECISIONS:241)
   - serial colour-tile prefix
   - waits after every step

   Port it to the WebGPU sequence, so one dispatch graph is shared across hosts. Ideally the graph is
   generated from one description; `engine-fracture/src/plan.rs` is the closest existing model.
4. **Pipeline submission.**
   - Metal and `scene_fps`: step with `step_gpu_ex(false)` ×N, then `wait_gpu` (`world.rs:708,1029`);
     double-buffer `params_buf`.
   - Native wgpu: replace the blocking `read_counters` on every step (`gpu.rs:1263-1288`) with a readback ring.
   - Bench adapter: set `pipelined: true` and stop running rubble synchronously
     (`bench/adapters/rubble/src/main.rs:256`).
5. **Host CPU per step.**
   - Metal: remove `env::var` reads and HashMap/string pipeline lookups per dispatch
     (`world.rs:530-592,713-777`; `gpu.rs:1039-1091`); resolve pipelines into a dense array once.
6. **Native joint support.**
   - Bug: `gpu.rs:937-947` never dispatches `csr_degree_joints`/`csr_scatter_joints`, so joints would
     be missing from adjacency, colouring and the primal.
   - Add a chain-mail scene to the browser and native wgpu benches so joints are measured on WebGPU too.
7. **Re-bench** ring110k / gables506k / ring9k / chain_mail / wall_smash / columns with `bench/run.py` in
   a quiet window. Only the gap that remains after P0 is a kernel problem.

## P1 — Remaining three-avbd parity items (kernel and dispatch level)

### P1-A: Dispatch count (biggest small-scene cost)

Dispatches per step:

| scene | three-avbd | rubble |
|---|---:|---:|
| ring 9k | ~105 | 205 |
| chain mail (rubble on Metal) | 138 | 405 |

At ~1.6 µs per dependent dispatch (more for indirect), that is ≈0.43 ms on chain mail and 0.16–0.35 ms on ring 9k, which is most of each gap.

1. **Adaptive colour cap with a completeness tail.**
   - Dispatch last step's colour count + 4, then `solve_primal_sparse` from there to the GPU's real count.
   - For colouring: last round count + 2, then `color_bodies_tail` (`world_step.slang:3761-3794`, which exists but is unused).
   - Already implemented for fracture in `engine-fracture/src/plan.rs:1081-1137`; port it to `demo.html:1132-1162`, `gpu.rs:957-996` and `world.rs:930-956`.
   - Keeps DECISIONS 8cl (no unsolved colour) and stays bit-identical for colours <4096 bodies.
   - Today rubble always encodes 32 primal colours per iteration and 32 JP rounds; three-avbd encodes cap ≈ used+3 (`solver.ts:152,1680-1706`).
2. **Skip global JP rounds** when the repair list ≤4096 (local repair already finished).
3. **Skip whole passes that can't do work:**
   - the sleep gate, commit and args (10 dispatches) when sleep can't trigger (`demo.html:1168-1172`, `gpu.rs:1008-1015`, `world.rs:971-983`);
   - `carry_rest_pairs` when nothing sleeps (it loads every previous manifold, both colliders and both bodies, then emits nothing; timed inside "grid");
   - `record_color_degrees` when its output is unused;
   - the joint hash rebuild when joints didn't change.
4. Add an indirect-dispatch case to `dispatch-diagnostic.html` to pin the per-dispatch cost model.

### P1-B: Bytes per contact, joint and body

5. **Persistent 32 B pair + 64 B point records** for every scene (P0-2 covers the browser; the native defaults also need it).
6. **Body hot prefix.**
   - Today: `BodyState` (112 B) spreads contact-read fields over bytes 0–32, 44–48 and 80–112, about 1.75 cache lines per visit.
   - Change: reorder so pos+invMass, rot, initialPos+flags and initialRot come first, as three-avbd's `layout.ts:9-25` does, or pad to 128 B.
   - Not yet tried (8dx and 8br didn't change the layout); 8dw names it as the remaining dual lever.
   - Also move the inertial target out of `gAabbs` into the body record, and drop the re-normalization and 4 branchy reciprocals per iteration (`world_step.slang:4260-4279`).
7. **Joint writes: changed fields only.**
   - Today: the dual and `warmstart_joints` store the whole 192 B joint, 31 field stores each, so every field must also be loaded (`world_step.slang:2756,2776,4920,4934`).
   - three-avbd stores 4–6 vec4.
   - Estimated about −0.3 to −0.6 ms at 71k joints × 10 iterations.
8. **Colouring awake bit.** Pack the dynamic/awake bit into the CSR partner array, removing the per-neighbour `gBodies` loads in `compactPreviousColor`/`init_colors`/`jpTryColor`. DECISIONS 8er names this as the lead.

### P1-C: Slang feature creep and codegen

The generated primal is about 2,300 lines versus three-avbd's ~300, and the generated narrowphase about 3,400–4,000 versus ~560.

9. **Specialize hot kernels at pipeline creation** with WGSL `override` constants or Slang `#define` variants.
   - Primal and dual: carry only what the scene uses. Candidates are:
     - armature (flag 8388608, a collider loop) and world inertia (flag 64)
     - slide Hessian and rolling resistance (2097152)
     - bond, fixed, hinge, D6 and slider joints
     - post-stabilization and the pivot-floor atomics

     three-avbd lazily compiles hull code for exactly this reason (`solver.ts:721-725`).
   - Narrowphase: split box-box (+sphere/plane) from the other pairs, and remove the full shape fallback from the reuse-only defer pass. Guarantee its tail capacity with an overflow error instead; this frees 2 bindings.
   - Verify the gain with pipeline statistics (registers and occupancy) before and after.
10. **`BoxManifold` (272 B, dynamically indexed `pts[8]`).**
    - Today it is passed and returned by value about 25 times in the narrowphase module.
    - Emit through `inout` and drop `mids[8]`, computing midpoints inline as three-avbd's `addFound` does.
    - Precedent: by-value clip arrays cost 4.52 → 2.0 ms (8de).
11. **Loop guards** (`engine-webgpu/src/lib.rs:304-330`).
    - Apply them only to data-dependent loops.
    - Force-unroll fixed-count loops (SAT 3 and 3×3, `mat6Zero`, `vec6Zero`, the 3-row joint loops).
12. **Primal prologue: 7 barriers.**
    - Today: `GROUP_SYNC` plus 3 rewritten `workgroupUniformLoad`s.
    - Read `gsUniformLanes` once, or compute lanes per thread as three-avbd does.
    - Also: mark read-only bindings `read`; use the `inverseSqrt` builtin instead of `1/sqrt`.

### P1-D: Collision passes

13. **Per-step clears.**
    - Today the warm hash is cleared every step: 16 MB at 110k and about 32 MB at 506k.
    - Replace this with step-tagged claim words, or use 0 = empty plus `clearBuffer`.
    - Skip clearing the empty second split-grid table.
14. **Warm-start path.**
    - Hash lookups stop at the match.
    - Persistence uses the record's run count instead of re-probing per point.
    - Use plain (non-atomic) read views for heads and records.
    - Fuse `pack_solver_pairs` into manifold emission.
15. **Remove dependent loads.**
    - Put body ids (plus a sphere bit) in the candidate record, removing the collider→body hop.
    - Shrink the world box record 80 → 64 B, with friction stored in `half.w`, so the narrowphase skips `gMaterials`.

**Don't redo** (rejected in DECISIONS):
- fused primal+dual (8dz) and per-pair dual (8dw)
- field-level body loads (8dx), counting-sort grid (8di), shrinking the grid table (8bi)
- persistent colours without repair (8bn), `setPipeline` hoisting (8bu)
- warm start inside the narrowphase on the current oversized kernel (8du/8dy); revisit only after items 9–10
- subgroup primal reduction (8dt) and the primal micro-arithmetic variants (8az–8bh, 8dk–8dp)

## P2 — Structural roadmap to pass everyone at 100k–500k (from Box3D, PhysX, Jolt, MuJoCo)

These follow P0/P1 and are ranked by expected impact on ring110k/gables506k. Estimates are per item vs today and are not additive.
1. **≤4-point area-max manifold reduction.**
   - Today rubble keeps up to 8 points (`box_collide.slang:234`); brick pairs average 4.4–5.6.
   - Pick the deepest point, then the farthest, then the max-area triangle, then the point adding most area outside it (Box3D `convex_manifold.c:792-952`), not "first 4" (8cj).
   - Estimate: ring −1.0 to −1.4 ms, gables −2.5 to −3.5 ms.
2. **Manifold-level friction for AVBD.**
   - A friction centre with 2 tangent rows plus twist, instead of 2 rows per point (Box3D `contact_solver_wide.inl:774-839`, PhysX patch anchors).
   - Rows per 4-point manifold drop from 12 to 7.
   - Needs new f64 oracle tests. Estimate: −0.8 to −1.5 / −2.5 to −4 ms.
3. **Persistent pair slots with relative-pose recycling, in place.**
   - Box3D recycling (10° plus 5 cm conservative bound), Jolt body-pair cache, PhysX PCM thresholds (`cudaNpCommon.h:340-411`).
   - No hash and no copy: three-avbd's hash+copy version gained only 2% (FINDINGS 8h).
   - Estimate: −1.0 to −1.6 / −3.5 to −5 ms.
4. **Per-body convergence skip within iterations.**
   - Jolt early exit, MuJoCo zero-iteration exit, PhysX freeze.
   - Must not hide sleeping; the 8da workload checks still apply.
5. **Moved-only broadphase.**
   - Fat AABBs with margin `min(5 cm, 0.125·r)`, a persistent pair set and a change bitmap (Box3D, PhysX ABP).
   - Fat margins rather than a 2 mm tolerance; that tolerance is why three-avbd 8k failed.
6. **Box/hull SAT feature cache** (8 B per pair, Box3D `b3SATCache`).
7. **Compact records:** extent-scaled snorm16 anchors and f16 penalties; best bundled with item 2.
8. **Gravity-ordered colour schedule** to try one fewer iteration (three-avbd 2D finding: top-down holds at 4 iterations, where coloured order needs about 10).
9. **Awake-set compaction (solver sets)** for mostly-sleeping worlds; add a benchmark for that case.
10. **Persistent CSR adjacency plus incremental colouring** driven by touch deltas; depends on items 3 and 5.
11. **Scope and backends:**
    - full shape coverage: segment capsules, core-shape GJK, hulls in the main path; capsule–capsule today uses a sphere approximation (`world_step.slang:1423-1438`)
    - compounds as one proxy per body, with manifold merge to ≤4
    - one workgroup per small island
    - two-level grid for mixed sizes
    - speculative contacts plus TOI against statics
    - CUDA host plus graph capture; Metal concurrent dispatch
    - collide once, substep the solve

## Execution order (after approval)

0. **Work in a new worktree copy.** Create a fresh git worktree from `integrate/all-worktrees`
   (26b269c), for example branch `perf/three-avbd-parity` at `.claude/worktrees/perf-three-avbd-parity`.
   - All edits, builds, tests and benchmarks run there.
   - The main checkout (`feat/avbd-attempt-2`, with other sessions' uncommitted joint and bench work) and the
     other worktrees are not touched.
   - Nothing is pushed without asking.
1. **Save the audit.** Write this plan plus the seven audit catalogs (four engines, three three-avbd
   comparisons) to `engine/docs/ENGINE_AUDIT_2026-10-16.md` in that worktree, and link it from
   `engine/docs/PINNED_BEST.md`.
2. **P0** items 1–7, then re-bench and record the result as a DECISIONS entry (next free id 8et) with JSON evidence.
3. **P1-A**, then P1-B, P1-C and P1-D: one A/B trial per item, following the existing protocol (interleaved runs, same session, DECISIONS entry plus evidence JSON plus patch on rejection).
4. **P2**, in order, starting with items 1, 6 and 8 (cheap and high-signal), then 5 → 3 → 10, then 2 and 4.

## Verification (every item)

- **Correctness:**
  - `cargo test -p engine-metal` and `-p engine-oracles` (Metal accuracy 18/18)
  - `engine-webgpu` GPU tests (`gpu_tests.rs`), and the Metal↔wgpu joint parity test
  - browser `ring480`/`gables240`/`columns180` gates, and native `ring_hold`/`gables_hold`
  - new unit tests per item; the colour-tail coverage test must fail if a colour is skipped
- **Speed:**
  - `python3 bench/run.py --group webgpu|metal --scenes small,steady,paper,large` in a quiet window (load < 3)
  - the browser `three-avbd-2x5x10 bench=only`, with `profile=colors` / `profile=np` phase splits
  - `scene_fps` on Metal (pipelined after P0)
  - the 8da workload validation must pass: no sleeping bodies and no dropped work in timed windows
- **Portability:** every change runs on the Metal and WGSL hosts. CUDA joins once its host exists.

## Appendix — reference catalogs (condensed)

Sources: Box3D `inspiration/box3d` @ efca0c1, Jolt v5.3.0, MuJoCo @ 89bb1a0, MJWarp @ c384ac5,
PhysX `../PhysX` (merge-base 4f2103c3a), three-avbd 2bb1a31.

- **PhysX 5 GPU:**
  - Broadphase:
    - integer-snapped AABBs (`broadphase.cu:101-165`)
    - ballot change bitmap (`updateTransformAndBoundArray.cu:209-231`)
    - incremental SAP that only reports found/lost pairs (`broadphase.cu:1528-1722`)
  - Narrowphase:
    - pair-type buckets
    - 4-lane box–box that reduces to 4 points (`cudaBox.cu:509-1098`)
    - PCM refresh/invalidation thresholds (`cudaNpCommon.h:340-411`)
  - Solver and partitioning:
    - AoSoA-32 constraint blocks
    - patch friction with 2 anchors (`contactConstraintBlockPrep.cuh:75-307`)
    - velocity forwarding (`solverMultiBlockTGS.cu:993-1125`)
    - at most 8 partitions plus mass splitting
    - statics excluded from colouring
  - Don't copy: CPU colouring, CPU islands, readbacks.
- **Box3D:**
  - Broadphase:
    - fat AABBs with margin `min(5 cm, 0.125·r)`
    - moved-only pair finding (`broad_phase.c:357-482`)
    - contact recycling (`physics_world.c:653-737`)
  - Narrowphase:
    - SAT cache (`convex_manifold.c:1271-1596`)
    - 4-point reduction (`:792-952`)
  - Solver:
    - 24 colours plus overflow, statics excluded (`constraint_graph.c`)
    - soft step with relax
    - friction centre plus twist (`contact_solver_wide.inl:774-839`)
    - solver sets
- **Jolt:**
  - body-pair cache at 1 mm / 2° (`ContactConstraintManager.cpp:795-995`)
  - manifold merge across sub-shapes, pruned to 4 (`PhysicsSystem.cpp:1045-1173`)
  - large-island splitter
  - union-find islands
- **MuJoCo/MJWarp:**
  - Most of their speed comes from many small worlds.
  - Transferable:
    - core-shape GJK (`collision_gjk.py:2591-2629`)
    - box–box face/edge hysteresis
    - count, scan and binary-search load balancing
    - refsafe stiffness cap
    - zero-iteration solver exit
    - implicit gyroscopic and damping terms
- **three-avbd FINDINGS:**
  - Rejected:
    - relative reuse with hash plus copy (8h)
    - f16 anchors (8d)
    - still-pair carry (8k)
    - fused dual (8i)
  - Unported: manifold reuse across substeps (#15), two-level grid (#17), top-down colour order.

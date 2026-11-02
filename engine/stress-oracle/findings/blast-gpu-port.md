# Blast's GPU stress solver in Slang (2026-10-25)

**Goal (user):** rubble replaces PhysX+Blast in vibe-land. That means matching vibe-land's stress solver in accuracy, then in speed, then beating it.

**What was ported:** the model vibe-land destroys structures with. That is Blast's `ExtStressSolver` from the PhysX fork, solved on the GPU by `stressgpu/NvBlastExtStressGpu.cu`. It is reimplemented in Slang as a new capability profile, `blast_gpu`.

**The plan:** `~/.claude/plans/in-rubble-we-have-cheeky-feather.md`.

## Parity with PhysX's native stage, 2026-11-01

**Target:** the GPU stress solver of the PhysX fork's native destruction stage, `Glavin001/PhysX` branch `integration/oracle-stress-gpu`, commit `ec8118341` (2026-10-16). It was read to understand the algorithm, never built against. The record is `reference/blast-gpu-port/provenance.json`.

**vibe-land's native configuration** (`sim-native/src/city.rs`, `scripts/perf/garage-vehicle-server.sh`, `destruction/src/native_runtime.rs`):
- 16 iterations, tolerance 1e-3, force tolerance 1e-3, warm start, damage rate 2, bend gain cap 3, fibre bending.
- **On:** incremental motion forest, incremental topology, the bond-balanced operator, unconverged steps published, and (always, on CuMetal) the float-expansion motion forest.
- **Off:** Krylov carry (on by default only with section rotational stiffness), section bending and rotational stiffness, shear stiffness, Mohr–Coulomb shear, rebearing, bearing joints. These stay opt-in flags here too, off in `blast_gpu`.

### Step 1: rigid modes as the target builds them

**Before:** the modes were taken from the authored node positions about their weighted centre, always six, and the Gram matrix was solved by Gaussian elimination in f32 every solve.

**Now** (`shaders/blast_motion.slang`; reference `stress-ref blast_bond/motion.rs`, following `StressMotionForest.cuh`): the frame is built once per island rebuild, on the GPU.
- **Positions** come from the live bond equations: exact float-expansion sums of each tree bond's `o0 - o1` from the island's root.
- **Closures:** a cycle that closes on more than 2^-17 of its coordinate scale is a real moment. One such closure leaves one rotation about it; two that are not exactly collinear leave none.
- **The frame:** the centre, the Gram matrix and its diagonally scaled Cholesky factor are built in float pairs and stored in f32. Each node's offset from the centre replaces its position for the projection.

**Deviations, on purpose:**
- **The tree.** The target takes each tree edge from whichever union wins a lock-free race. Here the tree is deterministic: levels from the root, each node taking its lowest-index live bond to the level before. Since the sums are exact, positions differ from any other tree only by the offsets' rounding residue, and reruns stay bit-identical.
- **The collinearity test.** The target's test sums binary64 products. With no f64 in WGSL, it grows float two-products here, up to 64 terms. Overflow reports the frame inexact (an error, as the target's expansion overflow is), never a silent answer.

**Fast math.** WGSL and Metal source are compiled by the driver with fast math. That folds TwoSum's error term to 0: a test measured `1 + 1e-8` giving error 0 on both paths. Multiplying intermediates by an unseen 1.0 does not help, because the compiler factors it out of a sum and reassociates the rest. Each intermediate's bits are XORed with an unseen 0 (`params.opaque_zero`) instead, which float algebra cannot see through. The precompiled Metal library uses precise math either way.

**Tests** (`stress-gpu tests/blast_motion.rs`), on the Metal library, Metal source and WGSL paths:
- **Arithmetic:** two-sum and two-product are exact on every path.
- **house-1story and villa-savoye, standing and freed from their supports:** the frames match the reference within 1e-5: anchoring, rotations, scale, factor and every node's offset.
- **Dyadic offsets:**
  - with consistent offsets, every cycle closes exactly and all three rotations stay;
  - one bond bent leaves one rotation, about the bend;
  - two non-collinear bends leave none.

All earlier suites still pass, on Metal and on WGSL at 10 storage buffers.

### Head-to-head on PhysX's own captured systems

**Tool:** `stress-gpu examples/blast_capture.rs`. It reads solves the native stage recorded with its diagnostic capture runtime (`StressProblemCapture.cuh` v2): each node's `D` and `b`, each bond's ends, offsets, health, column scale and warm forces, each component's decisions, and the forces the solve ended with.
- **The SDK:** vibe-land's `scripts/perf/build-stress-capture-sdk.sh`, PhysX `8bbfaac` (dirty), built 2026-10-07. That is near the target commit, not the target itself.
- **The runs:** vibe-land's vehicle lab, 2026-10-01, `target/stress-capture/*`, read in place.
- **The method:** rubble solves each system again from the same warm start on the GPU (Metal library), and both results are compared with the converged f64 answer. The captures are run-time input, not test fixtures.

| Set | Solves | Settings | Iterations equal | Convergence equal | Frames (anchored, rotations) equal | Worst force difference | rubble no farther from exact |
|---|---|---|---|---|---|---|---|
| m1 | 40 | 64 iterations | 40 | 40 | all | 1.1e-5 | 40 |
| c2 | 182 | 64 iterations | 182 | 182 | all | 3.9e-5 | 180 |
| c3 | 82 | 64 iterations | 82 | 82 | all | 1.5e-5 | 82 |
| f2 | 688 | force tolerance 1e-3 | 688 | 688 | all | 4.2e-5 | 688 |
| f1 (older force rule) | 688 | force tolerance 1e-3 | 366 | 688 | all | 3.5e-1 | 688 |
| rot/veneer-nocarry | 5 | rotational stiffness, 16 iterations | 5 | 5 | all | 1.7e-5 | 5 |
| rot/mailbox-1 | 6 | rotational stiffness, 16 iterations | 6 | 6 | all | 7.3e-6 | 6 |
| rot/tree-101 | 6 | rotational stiffness, 16 iterations | 6 | 6 | all | 4.0e-2 | 6 |

The force difference is `|rubble - physx| / |physx|` over live bonds. Exactness counts solves where rubble is within 1% of PhysX's own distance from the exact answer.

**f1 against f2: the force test's first step.** At the target commit, the force test judges only a preconditioned step. The projected steepest-descent step of iteration 0 is never judged, unless the polynomial runs first. The target's own comment gives the reason: judged from iteration 0, captured car solves stopped after one step with up to 42% force error.
- rubble had judged iteration 0, and so matched f1, an older build.
- rubble now follows the target. f2, recorded ten minutes later with the same cars, then matches in every solve.
- f1's iteration counts are expected to differ under the target's rule. rubble stays as close to the exact forces as PhysX was on them.

**Rotational stiffness on PhysX's systems:** the `rot` captures store each bond's `A`. rubble's solve with it matches iterations, convergence and frames. tree-101's force difference of 4e-2 comes with both 0.4 to 0.7 from the exact answer at 16 iterations (an unconverged tree), and rubble no farther.

**Krylov carry** (`BLAST_STRESS_CARRY_KRYLOV`, `BlastSettings::carry_krylov`; on by default with rotational stiffness, as in the target) is ported to the reference and both GPU paths.
- An unconverged solve leaves its direction, gamma and recurrence count for the next solve of the same island.
- It continues under the target's conditions: the same node count, fewer than `6 x nodes` recurrence iterations, and a load moved by at most the tolerance. A 16-fold gamma jump restarts.
- An island rebuild clears it, with the rigid-mode frame.

**What the carry does:**
- On a capped frame at rest it converges in 5 frames instead of 19 (reference test), with the same converged answer.
- The device tracks the f32 reference, carried or not, in one group and many.
- A race was found and fixed. `group_uniform4` synchronises only on WGSL. On Metal, threads that read the carry words after thread 0 cleared them disagreed on whether the solve was carried.

**Against the captured sequences:** `STRESS_CAPTURE_SEQUENCE=1` runs a capture's solves in order on one solver, warm from rubble's own last solve.
- **Without carry (veneer-nocarry, 5 solves):** PhysX's forces within 1e-6, every solve.
- **With carry (veneer-carry, 30 solves):**
  - PhysX's solve 1 equals its own no-carry result. Its carry shows only from solve 2, consistent with one more topology-generation change voiding the first carry.
  - From solve 2, PhysX's carried frames gain less than rubble's: at solve 5, 7.6e-2 from the exact forces against rubble's 1.7e-2.
  - rubble is never farther from the exact answer.

rubble follows the target's carry as written. These captures were recorded 2026-10-07 by a build nine days older than the target commit, so the difference is left to fresh captures from the target build.

**`BLAST_STRESS_FIRST_PRECONDITIONER`** (the target's A/B switch) is now `BlastSettings::first_preconditioner`: identity (the default), polynomial or warm. It is ported to the reference and both GPU paths, and tested against the f32 reference.

## Update: the native (resident) solve, 2026-10-25

`blast_gpu` now runs PhysX's native destruction solver (`componentStressSolve`), the one vibe-land runs on the Mac: `shaders/blast_native.slang`, with the reference in `stress-ref/src/blast_bond/native.rs`. The host-path CG stays available as `STRESS_BLAST_SOLVER=host`.

**Reference, house-1story:**
- Native reaches the host path's converged forces (within 1e-8) in 165 iterations, against 2,925.
- At vibe-land's budgets (native 16 iterations a frame, host 32), after 30 frames native is 1e-6 from the converged forces. The host path is 38% off.

**GPU, Metal and WGSL:**
- Converged forces are within 3e-6 of f64.
- Capped warm frames are within 1e-3 of the f32 reference; that reference's one-ulp spread is 3e-3.
- The certificate, zero loads, free pieces and bit-identical reruns behave as in the reference.

**f32 residual floor:** in f32 the residual test has a floor on the house far above the tolerance, while the forces are within 1e-5. This is the CUDA README's own observation. The force tolerance (`STRESS_BLAST_FORCE_TOL=1e-3`, off by default, as in vibe-land) stops such solves within 0.28% of the converged forces.

**Gravity only, 240 frames:**

| Pack | Native | Host |
|---|---|---|
| house-1story | 0 bonds broken | 0 |
| villa-savoye | 2 bonds broken (still one body) | 0 |

**villa-savoye, 4 shots** (frame times with the wait included):

| Solver | Frame median | Bonds broken | Bodies | Split frames |
|---|---|---|---|---|
| native | 5.8 ms | 1,106 | 313 | 115 |
| host | 3.6 ms | 208 | 63 | 5 |

The native solve carries impact loads through converged forces rather than a stalled iterate. Which breakage is right is for the vibe-land head-to-head to settle.

**Solve alone, villa-savoye** (submission included): 2.3 ms at 16 iterations. With the force tolerance on it is 1.0 ms, settling to 1 iteration a frame.

## Update: GPU-built events, and the city scale, 2026-10-29

**Events.** World's D1b destruction events now come from a GPU change log (`stress-gpu` `fracture_log.slang`), not a host diff of full arrays.
- After each break frame's split, the GPU appends broken bonds, chunks that changed body, and re-framed body slots. Each entry is stamped with its step, into a ring the host acknowledges.
- Readback is at most about 60 KB a step. The full arrays (about 460 KB at 10k chunks) are read only after an overflow.
- Fracture no longer forces `Backpressure::Block`. A World on `Drop` that dropped 115 of 120 readbacks reported the same 47 bonds, 4 splits and 104 bodies as one that waited every frame.

**City scale.** vibe-land's production scene `skyline-stable.json` has 29,592 chunks and 92,421 bonds.
- Its Blast islands are only 5, of about 6,000 nodes each. One workgroup per island leaves the GPU almost idle: the native solve takes 35 ms a frame (median; p95 41 ms; measured with the Unreal Editor also on the GPU).
- vibe-land's CuMetal build has the same one-threadgroup-per-component shape up to 8,192 nodes. CUDA switches to a grid-wide cooperative kernel above 1,024.
- Next: spread a large island over many workgroups, per-iteration dispatches with two-level fixed-order sums per island, the same recurrence.

## What exists now

| Piece | Where | What it does |
|---|---|---|
| f64/f32 reference | `stress-ref/src/blast_bond.rs` | The CUDA host path's math, written from its definitions. It covers centred offsets, length and mass scaling, equalised or native mass weighting, and the compliance column scale. Islands are cut at supports, each with its own tolerance, and the solve is a node-space CGLS (numerator \|B^T rho\|^2, each bond once) with a warm start and a settled-island skip. Stresses use the shared formula (section bending capped at 3, fibres) and the damage law (fatal or 2/s). The CGLS is generic: `device_precision` runs it in f32. |
| Blast's pack inputs | `scene_pack::ImportOptions::blast_authoring` → `Scene::blast` | The pack's node centroids, masses and volumes; its bond centroids, normals and areas; and both limit sets of each material. Bonds map one to one onto the scene's bonds. |
| GPU solve | `stress-gpu/shaders/blast_cgls.slang` | One 1024-thread group per island, using fixed-order group sums (no float atomics). |
| GPU stresses and damage | `blast_material.slang` | The formula with fused operations, as the CUDA code uses them. |
| GPU islands | `blast_topology.slang` | Rebuilt from bond health only after a break, gated on the GPU. It uses a lock-free union-find, scans, scatter, and a per-island bitonic sort, so lists are ascending and deterministic. |
| GPU fracture frame | `blast_fracture.slang`, `GpuFracture::set_blast` | Loads go in as vibe-land feeds them: gravity, contact force over node mass, and the centrifugal load, in the structure frame. Breaks go into the fracture's split. |
| Profile | `Profile::BlastGpu` (`blast_gpu`) | `ThresholdWorld::with_profile`; stress-web; the Fracture lab's "Stress solver · Blast" mode; `fracture_glb STRESS_PROFILE=blast_gpu`. |

Nodes and bonds keep their indices: an island is a list of each. A break therefore rewrites lists and keeps every bond's warm start, which is the CUDA host path's incremental default. A frame reads nothing back.

## Accuracy

These results come from `stress-gpu/tests/blast_host_vs_reference.rs` and `stress-rubble/tests/blast_gpu_world.rs`. They pass on Metal, and on WGSL at 10 storage buffers.

- **Converged solves** (tolerance 1e-5): the GPU is within 4e-5 (equalised) and 8e-5 (native) of the f64 reference on house-1story.
- **Capped solves** (vibe-land's 32 iterations a frame) are compared against the f32 reference.
  - The CUDA host path itself runs this recurrence in f32. At iterations 24-32 on house-1story, a one-ulp nudge to the inputs moves the iterate by up to about 3e-3. Any f32 implementation has that spread, CUDA's included, since its float atomics reorder sums from run to run.
  - The GPU stays within about 1.2× that spread of the f32 reference. It is up to 1.4e-2 from the f64 one: at capped counts, f32 CG drifts from f64 CG.
  - `difference_by_iteration`, an ignored diagnostic, shows the drift growing smoothly. It is 1e-7 at iteration 1 and 4e-6 at 16. Then it jumps where CG loses orthogonality in f32, which is not the signature of a logic error.
- **Stresses and damage** are checked by applying the formula to the device's own impulses: stresses agree within 1e-5, and all 877 of 877 damage verdicts agree. Health after damage is exact.
- **Islands** are identical to the reference's after 693 bonds break.
- **In the world:** a house under gravity tracks the f32 reference frame for frame. The difference is 5e-4 at the first frame and falls to 2e-5 over 20 frames.
- **Determinism:**
  - Reruns are bit-identical, both the standalone solve and the struck-house world.
  - Villa-savoye's outcome (206 bonds, 62 bodies) is the same at 256 and at 1024 threads a group.

**Not done yet:**
- The one-off cross-check against Blast's own solver through the PhysX repo's `blast_ladder_bench` JSON.
- vibe-land's break sets for the same events, which need the P0a baseline run.

## Speed (M3 Max, one engine on the GPU at a time)

| Run | Frame (waited for) | Of which |
|---|---|---|
| villa-savoye (1045 chunks, 2464 bonds), 4 shots, `threshold` | about 2.7 ms | rubble 2.27, ingest 0.41, fracture 0.44 |
| the same, `blast_gpu` | 3.58 ms median, p95 3.81 | rubble 2.05, ingest 0.40, fracture with Blast 1.83 |
| Blast solve alone on villa-savoye (`blast_bench`) | 1.76 ms median, submission and wait included | one island, 32 iterations (Blast stalls on the hull packs, as the 2026-10-16 ladder found, so nothing settles) |

**Change made:** one group per island was latency-bound at 256 threads (4.4 ms a solve). At 1024 threads it is 1.8 ms. The recurrence is the same and the result is still repeatable.

**Discarded measurements:** runs earlier the same day read 23-29 ms a frame while other processes loaded the machine.

**vibe-land's numbers are not comparable yet:**
- Its server-alone figures (cannonball p50 7.6-9.0 ms, meteors about 9 ms; `performance-targets-status-2026-09-25.md`) are whole-city ticks.
- Matching them needs the same scenes here. Towns are multi-structure, and the Blast authoring covers one structure so far.

## Deviations from vibe-land, on purpose

- **No re-simulation.** vibe-land replays a frame after a split (`VIBE_CITY_RESIM_PASSES`); AGENTS.md forbids it. Strikes are predicted instead, as `threshold` predicts them.
- **Gravity loads every piece every frame.** vibe-land skips gravity on sleeping bodies untouched by contact. Here, a settled island's input repeats and the skip makes it free.
- **Islands are numbered by their lowest node, not by first appearance over the bonds.** Only the ids change.

## Next

1. **P0a:** measure vibe-land's stress stage and break sets on the same events. The `vibe-baseline` session brief is in the plan.
2. **Multi-structure Blast authoring (towns),** then 10k+ bodies.
3. **The resident path (P2):** Chebyshev block-Jacobi PCG, rigid-mode projection, native weighting. It converges in fewer iterations, so islands settle and skip.
4. **The V-cycle for large components (P3).**
5. **Guard digests** for `blast_gpu`, and the lab checked in the browser.
6. **The one-off Blast CPU cross-check.**

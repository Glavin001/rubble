# Blast vs our stress solver: head to head, then a fidelity ladder (2026-10-16)

## The question

The earlier Blast-based GPU stress solver in the PhysX repository was fast but
quasi-static. Ours (stress-gpu, with rubble as the rigid engine) has explicit impact
windows and a damageable joint law. This note asks three things:
- Is ours faster at Blast's own, simpler task?
- Starting from that task, what does each of our features add in accuracy and cost?
- Where is the right balance between them?

Machine: an M3 Max, shared all day with other sessions (load average 25-170). Every
timing here was taken under that load, so medians are the robust figures. Each table
states how many repeats it is over.

## Summary

- **Blast computes a quasi-static force balance per frame and stops early.**
  - It solves one equilibrium of bond forces per frame with CGNR: 25 iterations
    (32 in the Rust crate), warm-started, stopping at a normal-equation residual of 1e-3.
    A frame that hits the cap keeps its partial answer.
  - Loads are gravity, centrifugal acceleration, and contact impulse / dt at the
    nearest node.
  - A bond breaks at once above its fatal limit, and is damaged at 2/s above its elastic
    limit.
  - There is no inertia and no impact dynamics: the rigid engine decides how bodies
    move, and fracture follows from the loads.
- **We built the same task into our solver and matched Blast's answer where inputs
  match.** `--config blast` runs quasi-static equilibrium every frame with a same-step
  cascade, threshold bonds, and the engine driving every body. The only missing switch
  was turning impact islands off, which is now added; defaults are bit-identical. Fed
  the same masses and supports, the two answers agree on brick-building, the one
  well-conditioned pack.
- **On the hull packs the disagreement is Blast's, not ours.**
  - Blast's CGNR stalls, and its converged answer delivers at most 56 % (rig-portal)
    and 79 % (villa-savoye) of the weight to the supports.
  - Ours satisfies equilibrium exactly on every pack.
  - Under gravity alone, Blast needs 16 to 326 capped frames to reach even its own fixed
    point.
- **We are not faster at Blast's task, and the reason is the pipeline, not our features.**
  - Blast CPU (Rust crate with Rapier): an idle house costs about 0.03 ms a frame (it
    skips the solve when nothing changed), a rammed house 1-5 ms.
  - Ours in the Blast configuration: an idle house costs 3-7 ms, impacts tens to
    hundreds of ms, and a structure that cannot stand (villa-savoye under our materials)
    about 1.9 s.
  - The idle floor is per-frame host↔GPU round trips and engine coupling. That is
    Stage 2 of GPU-resident coupling, in the parent session.
  - The impact cost comes from solving every cluster to convergence (60 Newton steps ×
    up to 20,000 CG steps) with a same-step cascade. Blast stops at 25 iterations and
    accepts its iterate.
  - On this machine our Blast configuration is slower than our full real-time
    configuration (sum of per-scene median frames: 2,246 vs 1,621 ms). Adaptive mode
    does no work for steady clusters and handles impacts explicitly.
  - The Blast GPU path does not run here: the CuMetal build spent 40 minutes in setup
    without reaching frame 0. It would solve these packs on the CPU anyway, since all
    are under 4,096 bonds.
- **Lossless fixes and engine bugs fixed along the way** (results identical, digests
  checked; details below):
  - an exact one-substep frame average;
  - skipping solves whose inputs repeat bit for bit;
  - no rebuild without damage;
  - two engine bugs: a broadphase that lost static pairs (this changes b8_sudden, for
    the worse; see §4), and hulls rubble rejected (now feat/stress-fracture's fix).
- **The ladder: two features carry nearly all the accuracy.**
  - Impact islands cut exit-speed error from 64-69 % to 3 %.
  - Crack contact halves broken-bond error, and is cheaper than the rung before it.
  - Weibull, static fatigue and buckling are inert on this set: the materials and scenes
    provide no data for them, and the digests are identical. Rate effects and rebar
    cost nothing measurable.
  - Bond dashpots on the GPU move b5_v20 away from the oracle; that is a follow-up.
- **Recommendation:** see [the last section](#recommendation).

## 1. The Blast solver (PhysX repository)

Read-only investigation of `blast/` (stress solver, Rust crate, Bevy demo) and
`demos/blast-stress-demo`.

| aspect | Blast |
|---|---|
| unknowns | one 6-vector (force, torque) per bond (`stress.h`) |
| nodes | mass, volume, centroid; mass 0 = support; sphere inertia unless set |
| equation | `(C^T I^-1 C) J = -C^T a`: node accelerations in, bond forces out (`stress.cpp:314-341`); per-bond compliance weight `sqrt((E/30 GPa) A / L)`; mass and length scaled by their means |
| solver | CGNR (`cgnr.h`), stop at `|A^T r|^2 <= tol^2 |b|^2`, tol 1e-3; 25 iterations per frame (PhysX adapter, GPU), 32 (Rust crate), 64 (`native_wall_capture`); warm start; a capped frame keeps its iterate; settled islands skipped |
| GPU | CUDA (`NvBlastExtStressGpu.cu`, node-space CGLS), compiled to Metal by CuMetal on macOS; FP32; graphs under 4,096 bonds are solved on the CPU (`gpuStressMinimumBondCount`) |
| loads | gravity per actor, centrifugal `-w x (w x r)`, contact impulse / dt as a force at the nearest node (no torque) |
| stress | `n = F.n / A` (tension +), shear `|F_perp| / A + twist min(4.81 / sqrt A, 3)`, bend `|M_perp| / A min(6 / sqrt A, 3)`, fibres: tension `max(0, n + bend)`, compression `max(0, bend - n)` (`NvBlastExtStressFormula.h`) |
| fracture | excess `(sigma - elastic) / (fatal - elastic)` per mode; >= 1 breaks at once, else damage `health * min(m dt rate, 1)`, rate 2/s; health is the remaining area; no per-frame cap; resim passes optional |
| on this M3 Max | Rust crate (CPU solver + Rapier) builds in about 1 min and runs. `native_wall_capture` (CuMetal GPU) is prebuilt but spent 40 min in setup (house-1story, 3 frames) without reaching frame 0; `docs/CUMETAL_COMPATIBILITY.md` lists it as awaiting qualification; `record_stress_benchmark.sh` is Linux/NVIDIA only |

The packs in `engine/stress-oracle/packs` are byte-identical to Blast's (SHA-1). One
caveat: the 2026-10-16 three-way benchmark's "physx-gpu" column was stress-gpu on PhysX,
not Blast.

The Blast harness is `blast/blast-stress-solver-rs/examples/blast_ladder_bench.rs`, an
untracked file in PhysX; no tracked PhysX file was changed. It has three modes:
- `forces`: the solver alone, under gravity;
- `bench`: the full pipeline, timed;
- `column`: a check against known forces.

Two harness caveats:
- **Bond order.** Blast reports per-bond stresses in its asset's bond order, which
  `NvBlastCreateAsset` sorts by (lower node, higher node), not in the order the bonds
  were given. `column` checks this on a stacked column with known forces: the values
  are exact, but reordered.
- **Materials.** The Rust crate's `Sim` flattens a pack to one material, so its rams
  shatter whole houses. Its outcomes below are therefore not comparable with ours; its
  costs are.

## 2. The Blast-equivalent configuration of our solver

`stress_bench --config blast` sets:

| switch | value | how it matches Blast |
|---|---|---|
| `sim.solve_mode` | `QuasiStatic` | equilibrium every frame, no inertia in the stress solve |
| impact islands | off (new: `World::impact_islands`, `GpuWorld::set_impact_islands`; quasi-static always runs without) | the engine drives every body; its contacts reach the stress solve only as loads, as Blast takes PhysX's contact impulses |
| `Features` | `softening`, `weibull`, `crack_contact`, `rate_effects`, `static_fatigue`, `buckling`, `rebar`, `damping` off | threshold bonds that break when they reach strength, with nothing carried by a cracked share |
| `Features` kept | `crushing`, `rigid_motion_loads` | Blast fails bonds in compression, and loads actors with gravity and centrifugal acceleration |
| `Methods` | established, `true_step` | as the real-time configuration (they only matter for explicit substeps) |
| engine loads | frame average of the contact filter's pulses (exact) | Blast's impulse / dt |

What stays different, by design:
- **Solved to convergence.** Ours goes to a relative residual of 1e-5 (f32 GPU,
  60 Newton steps); a cluster with no equilibrium keeps its last one.
- **Same-step cascade.** Ours re-solves after breaks within the frame; Blast cascades
  over frames.
- **Stiffness model.** Ours is each joint's elastic stiffness (axial, shear, bending,
  torsion from the patch); Blast uses one compliance weight for all six directions.
- **Load point.** A contact load acts at its point, with torque, not at the nearest node.

Pack import differences (our importer, `scene_pack.rs`, not the solver):
- calibrated materials instead of the assets' gameplay limits;
- code-minimum rebar;
- supports at grade;
- bond patches recomputed from shared faces.

These change loads and strengths. `examples/blast_forces --blast-inputs` removes the
first and third (pack masses, pack supports) for the force comparison.

## 3. Agreement of the two quasi-static answers

Bond stresses under gravity, fracture off, ours on the f64 oracle:
- ours: `examples/blast_forces`;
- Blast: `blast_ladder_bench forces PACK 3000 25`;
- comparison: `scripts/blast_force_agreement.py`.

Blast's own stress formula is applied to our bond forces, and the two are compared bond
by bond. Columns: peak fibre stress (the larger of compression and tension) and shear,
over bonds Blast loads above 1 % of its largest.

**As imported** (our materials and supports):

| pack | Blast frames to converge | matched bonds | peak fibre: Pearson / median ours/Blast / within 2x | shear: Pearson / median / within 2x | first Blast frame vs converged Blast: peak Pearson / median |
|---|---|---|---|---|---|
| house-1story | 16 | 877 | 0.00 / 0.22 / 19 % | -0.03 / 0.15 / 16 % | 0.90 / 0.70 |
| house-2story | 67 | 1652 | -0.04 / 0.33 / 26 % | 0.01 / 0.31 / 24 % | 0.78 / 0.22 |
| villa-savoye | 326 | 2464 | 0.08 / 0.89 / 30 % | 0.02 / 0.69 / 33 % | 0.12 / 0.00 |
| brick-building | 20 | 494 | 0.98 / 3.94 / 9 % | 0.80 / 4.23 / 11 % | 0.96 / 0.72 |
| rig-portal | 4 | 39 | 0.26 / 1.21 / 31 % | 0.49 / 2.95 / 22 % | 1.00 / 1.00 |
| rig-toppled | 33 | 42 | 0.26 / 0.74 / 48 % | 0.70 / 0.68 / 60 % | 0.89 / 0.13 |
| comp-wall-bay | 12 | 109 | 0.02 / 0.08 / 17 % | 0.08 / 0.01 / 0 % | 0.88 / 0.89 |

**With Blast's inputs** (pack masses and supports, `--blast-inputs`):

| pack | Blast frames to converge | matched bonds | peak fibre: Pearson / median ours/Blast / within 2x | shear: Pearson / median / within 2x | first Blast frame vs converged Blast: peak Pearson / median |
|---|---|---|---|---|---|
| house-1story | 16 | 877 | -0.01 / 0.31 / 26 % | 0.06 / 0.20 / 20 % | 0.90 / 0.70 |
| house-2story | 67 | 1652 | -0.02 / 0.39 / 29 % | 0.01 / 0.38 / 27 % | 0.78 / 0.22 |
| villa-savoye | 326 | 2464 | 0.08 / 1.19 / 37 % | 0.01 / 0.91 / 40 % | 0.12 / 0.00 |
| brick-building | 20 | 494 | 0.98 / 0.99 / 85 % | 0.80 / 1.06 / 80 % | 0.96 / 0.72 |
| rig-portal | 4 | 39 | 0.23 / 1.42 / 37 % | -0.16 / 2.69 / 28 % | 1.00 / 1.00 |
| rig-toppled | 33 | 42 | 0.50 / 0.85 / 48 % | 0.71 / 0.72 / 63 % | 0.89 / 0.13 |
| comp-wall-bay | 12 | 109 | 0.18 / 0.53 / 27 % | 0.04 / 0.02 / 0 % | 0.88 / 0.89 |

Two results:
- With the same inputs, brick-building agrees: Pearson 0.98, median ratio 0.98, 85 %
  of bonds within 2×. The as-imported 3.9× there is our calibrated density
  (107,520 kg vs the pack's 25,000 kg).
- On the hull packs there is no agreement, and the equilibrium check below says why.

**Equilibrium check** (`scripts/blast_support_check.py`). Blast gives only
(compression, tension) per bond, which bounds each support bond's axial force by
`A max(c, t)`. Summing that bound over the bonds into the supports gives an upper bound
on the load Blast delivers there:

| pack | weight (kN) | ours: vertical load into supports (kN) | Blast: upper bound (kN) | Blast bound / weight |
|---|---|---|---|---|
| house-1story | 875 | 875 | 1283 | 1.47 |
| house-2story | 2088 | 2088 | 4208 | 2.02 |
| villa-savoye | 9692 | 9692 | 7671 | 0.79 |
| brick-building | 245 | 245 | 257 | 1.05 |
| rig-portal | 208 | 208 | 116 | 0.56 |
| rig-toppled | 784 | 784 | 8078 | 10.30 |
| comp-wall-bay | 845 | 845 | 983 | 1.16 |

- Ours delivers the weight exactly on every pack.
- Blast's bound falls below the weight on rig-portal (56 %) and villa-savoye (79 %), so
  its converged answer is not an equilibrium there.
- It is a fixed point of Blast's own solve: 5,000 iterations change it by under 2 %,
  and its linear error stays at 0.02-0.3.
- Elsewhere the bound is loose (above the weight) and decides nothing.
- Blast's first frame is also far from that fixed point: it needs 16 (house-1story),
  67 (house-2story) and 326 (villa-savoye) frames at 25 iterations to get there. For
  the villa, the first frame correlates 0.12 with the converged answer.

Blast's solve is cheap: 0.04 ms (house-1story) to 0.21 ms (villa) per warm frame, and
0.4-1.6 ms for the cold first frame.

## 4. Head to head on the packs

Each pack idle and rammed, 15 frames (0.25 s) × 3 repeats:
- ours: `stress_bench --set packs`;
- Blast: `blast_ladder_bench bench PACK idle|ram 15 1 SURFACE_X`.

The ram is a 400 kg body at 12 m/s into the middle of the -x face: a box for us, a
ball for Blast's harness. Its front starts 0.2 m from the hull surface.

#### Frame time (ms: mean / median / p95 / max; 15 frames x 3 repeats)

| scene | Blast CPU (resim 1), whole frame | Blast CPU, stress step only | ours, blast config | ours, realtime |
|---|---|---|---|---|
| house-1story_idle | 0.0 / 0.0 / 0.3 / 0.4 | 0.0 / 0.0 / 0.0 / 0.0 | 10.6 / 6.9 / 27.9 / 64.5 | 6.5 / 3.3 / 9.4 / 86.7 |
| house-1story_ram | 1.3 / 1.1 / 3.2 / 3.2 | 0.1 / 0.0 / 1.0 / 1.1 | 52.0 / 12.5 / 291.5 / 297.4 | 159.4 / 143.7 / 253.9 / 318.9 |
| house-2story_idle | 0.1 / 0.0 / 0.5 / 0.5 | 0.0 / 0.0 / 0.0 / 0.0 | 49.9 / 28.0 / 208.8 / 266.3 | 80.7 / 79.1 / 106.1 / 139.5 |
| house-2story_ram | 2.6 / 2.3 / 5.9 / 6.1 | 0.2 / 0.0 / 2.0 / 2.0 | 529.7 / 687.8 / 695.2 / 698.2 | 184.6 / 172.0 / 302.8 / 340.4 |
| villa-savoye_idle | 0.1 / 0.0 / 0.8 / 1.0 | 0.0 / 0.0 / 0.0 / 0.0 | 1895.5 / 1703.7 / 3292.6 / 3294.2 | 82.7 / 67.2 / 166.9 / 168.6 |
| villa-savoye_ram | 4.9 / 4.2 / 11.9 / 12.2 | 0.3 / 0.0 / 3.4 / 3.6 | 1810.7 / 1702.0 / 3197.8 / 3256.3 | 107.5 / 94.1 / 175.4 / 176.6 |
| brick-building_idle | 0.0 / 0.0 / 0.2 / 0.2 | 0.0 / 0.0 / 0.0 / 0.0 | 3.6 / 2.7 / 10.2 / 11.4 | 2.8 / 2.8 / 3.6 / 3.9 |
| brick-building_ram | 0.2 / 0.2 / 0.4 / 0.4 | 0.1 / 0.1 / 0.2 / 0.2 | 12.8 / 9.7 / 24.9 / 25.1 | 33.9 / 36.1 / 48.0 / 51.2 |
| rig-portal_idle | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 | 3.7 / 3.5 / 6.3 / 11.0 | 2.3 / 2.2 / 2.8 / 3.5 |
| rig-portal_ram | 0.1 / 0.1 / 0.2 / 0.2 | 0.0 / 0.0 / 0.1 / 0.1 | 10.7 / 8.4 / 24.0 / 26.9 | 24.0 / 22.2 / 32.5 / 50.4 |
| rig-toppled_idle | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 | 7.5 / 6.1 / 21.7 / 30.3 | 12.1 / 11.3 / 18.9 / 19.5 |
| rig-toppled_ram | 0.1 / 0.1 / 0.2 / 0.2 | 0.0 / 0.0 / 0.1 / 0.1 | 9.6 / 6.4 / 31.8 / 32.7 | 26.7 / 27.0 / 31.5 / 33.6 |
| comp-wall-bay_idle | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 | 2.6 / 2.1 / 4.0 / 9.9 | 2.2 / 2.2 / 3.1 / 3.4 |
| comp-wall-bay_ram | 0.2 / 0.2 / 0.5 / 0.5 | 0.0 / 0.0 / 0.1 / 0.2 | 8.4 / 6.1 / 21.6 / 22.8 | 39.3 / 36.3 / 54.0 / 58.7 |

#### Outcome (exit speed m/s / fragments / broken bonds)

| scene | Blast CPU | ours, blast config | ours, realtime |
|---|---|---|---|
| house-1story_idle | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 |
| house-1story_ram | 4.85 / 316 / 319 | 2.49 / 1 / 25 | 2.70 / 37 / 198 |
| house-2story_idle | 0.00 / 1 / 0 | 0.00 / 1 / 7 | 0.00 / 1 / 4 |
| house-2story_ram | 2.83 / 596 / 599 | 2.50 / 2 / 34 | 1.65 / 22 / 155 |
| villa-savoye_idle | 0.00 / 1 / 0 | 0.00 / 4 / 40 | 0.00 / 4 / 33 |
| villa-savoye_ram | 3.86 / 1045 / 1045 | 1.95 / 4 / 40 | 0.52 / 4 / 92 |
| brick-building_idle | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 |
| brick-building_ram | 6.28 / 1 / 0 | 6.39 / 1 / 0 | 6.70 / 1 / 0 |
| rig-portal_idle | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 |
| rig-portal_ram | 1.68 / 18 / 18 | 2.48 / 1 / 0 | 2.27 / 1 / 0 |
| rig-toppled_idle | 0.00 / 1 / 0 | 0.00 / 2 / 4 | 0.00 / 2 / 4 |
| rig-toppled_ram | 2.31 / 20 / 20 | 2.67 / 2 / 4 | 4.48 / 2 / 4 |
| comp-wall-bay_idle | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 |
| comp-wall-bay_ram | 3.84 / 35 / 35 | 2.48 / 1 / 0 | 2.29 / 13 / 64 |

Reading the table:
- **Cost.** Blast's whole frame is 1-3 orders of magnitude cheaper: 0.03 ms idle,
  0.1-5 ms rammed. Its stress step is at most 3.6 ms on the villa's impact frame.
- **Idle floor.** Ours costs 2-7 ms idle. On house-1story that is mostly engine
  coupling (sync_bodies, step, motion, set_motion: about 3.6 ms) plus GPU downloads,
  readbacks and the frame tail (about 3.5 ms).
- **Quasi-static tail.** Our Blast configuration's idle frame is no cheaper than the
  real-time one, and on houses it costs more. Each frame the tail syncs the mirror,
  rebuilds solve inputs, commits damage and uploads bond history, where adaptive mode's
  settled clusters do nothing.
- **Structures that cannot stand.** villa-savoye cracks under its own weight with our
  calibrated materials (40 bonds; 33 in the real-time configuration too). Its clusters
  then have no static equilibrium, and each frame repeats 60 failed Newton steps
  (about 36,000 CG steps, 1.7 s). Blast has no notion of "no equilibrium": it caps the
  iterations and moves on.
- **Measured, not adopted:** `STRESS_GPU_STATICS_NEWTON=10` (a lossy per-frame budget)
  takes the villa from 1.9 s to 0.65 s a frame with an identical digest. But it changes
  house-2story's rammed outcome (7 broken bonds instead of 34). Adopting a per-frame
  budget like Blast's is your decision.
- **Outcomes.** Blast's harness shatters the rammed houses completely (all 319 bonds of
  house-1story). That is its single flattened material, not a modelling claim, so these
  outcomes are not comparable with ours.

### Why we are slower, and what was fixed

Before the fixes, the coupled quasi-static frame did needless work. Fixes, all in the
new coupled quasi-static path, so no existing default changed:

| issue | fix | effect (Blast configuration) |
|---|---|---|
| about 2,800 empty substeps per frame at the explicit stress step, only to average the loads | engine pulses averaged exactly over the frame (`LoadFilter::frame_average_terms`), one substep | house_idle 43 → 21 ms mean, digest unchanged |
| every frame re-solved clusters whose inputs had not changed, including a mechanism re-proving 60 failed Newton steps (house_ram after impact: 225 ms a frame) | solves whose inputs repeat bit for bit a solve that left the state unchanged are skipped (`static_inputs.rs`) | house_ram 436 → 42 ms, b5_v20 880 → 372 ms, digests unchanged (`STRESS_GPU_QS_EVERY_SOLVE=1` A/B) |
| full rebuild after every static solve | upload of the solved bonds when nothing broke | part of the above |

Engine bugs found and fixed (rubble):

| bug | symptom | fix | check |
|---|---|---|---|
| grid broadphase lost static pairs below the querier's index (above 512 colliders) | b5's ram passed through its anchored 840-chunk wall: no contact at all with islands off; fragments fell through anchored structures in the engine | statics kept at any index in both grid paths (`world_step.slang`); new test `broadphase_grid_keeps_static_pairs_below_the_querier_index` fails before, passes after | lib suite, columns_10k, ring_hold, render_abi pass; realtime digests unchanged; gables_506k fails the same way before and after (pre-existing) |
| "hull not accepted by rubble: non-convex collision hull" panic when house hull chunks form a fragment | half the ladder's rungs (and destruction-view's house_ram) crashed | measured here with this branch's fallback (own-frame hull turned by the collider, ee15edc); after merging feat/stress-fracture, its fix 4b79c32 (hulls cooked once in their own frame, box fallback with a report) replaces it | realtime and accuracy digests equal feat/stress-fracture's (except b8, see below) |

Open items found along the way:
- **Adaptive without islands overflows rubble on house_car.** `Overflow(128)` is a grid
  hash slot holding more than 4,096 colliders, most likely bodies collapsing onto one
  cell. Only the island-less adaptive rung hits it.
- **The Blast configuration's b5 is not bit-reproducible.** Two digests appear across
  runs, with the same counts. Most likely the engine with about 86 fragment bodies.
- **`scenes::rammed` places its ram from chunk centres.** On house-1story (the `house_ram`
  regime of the benchmark) the ram starts 0.3 m inside the house; on rig-portal and
  comp-wall-bay, ram placement explodes in the real-time configuration (54 and 63 m/s
  exit speeds). The packs set here places it from the hull surface. The shared scene is
  unchanged, because its digests are the benchmark's.
- **The broadphase fix changes one real-time outcome: b8_sudden, for the worse.**
  - With the fix alone on top of feat/stress-fracture (checked bit for bit, twice),
    b8_sudden breaks 36 bonds into 35 fragments, where it broke 10 into 10. The oracle
    breaks 11.
  - Debris that used to fall through the anchored remainder of the frame now hits it in
    the engine. The engine's contact-to-load path then over-breaks: the oracle resolves
    that contact with its own penalty contact.
  - The fix is correct (the engine was dropping contacts), but it exposes that error, so
    it is a decision for the engine owners. It is its own commit (8352765).
  - The other 11 accuracy scenes are bit-identical with and without it. b5_v30's 95
    broken bonds, where the benchmark doc has 91, come from feat/stress-fracture itself,
    not from this fix.

## 5. The ladder

`stress_bench --set accuracy --config RUNG`, three round-robin passes (15 frames each, so
45 frames per scene and rung). Accuracy is against the stress-ref oracle (CPU f64, adaptive,
the same real-time switches). Rungs are cumulative, one switch each:

1. **blast**: §2's configuration.
2. **adaptive**: explicit dynamics while loaded, still no islands.
3. **islands**: impact islands routed before the step.
4. **softening, weibull, crack_contact, rate_effects, static_fatigue, buckling, rebar,
   damping**: each model feature in turn.
5. **realtime**: the full configuration (adaptive, established methods, true step,
   stress safety 0.5, contact safety 0.35, islands before the step,
   `redo_fractured_frames = false`).

`damping` and `realtime` are the same configuration: their digests are identical.

#### Cost per rung (ms per frame: mean / median / p95 / max)

| scene | blast | adaptive | islands | softening | weibull | crack_contact | rate_effects | static_fatigue | buckling | rebar | damping | realtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| house_idle | 4.9 / 4.3 / 9.4 / 13.6 | 5.6 / 4.0 / 10.7 / 32.5 | 6.9 / 3.7 / 14.3 / 112.4 | 4.6 / 3.4 / 10.0 / 23.9 | 3.4 / 3.3 / 5.5 / 6.1 | 3.3 / 3.2 / 5.3 / 5.7 | 3.2 / 3.2 / 5.1 / 5.6 | 7.6 / 3.3 / 14.0 / 127.5 | 4.2 / 3.2 / 8.3 / 15.0 | 6.0 / 3.3 / 9.0 / 86.7 | 5.0 / 3.1 / 5.7 / 76.9 | 4.6 / 3.3 / 8.9 / 13.5 |
| wall_ram | 64.0 / 56.9 / 136.4 / 162.6 | 104.5 / 60.6 / 495.3 / 882.7 | 123.6 / 103.0 / 306.7 / 538.4 | 362.6 / 308.3 / 910.0 / 1309.8 | 284.7 / 234.0 / 798.2 / 851.9 | 109.5 / 95.4 / 266.4 / 267.7 | 122.6 / 104.8 / 255.7 / 404.2 | 110.5 / 95.6 / 259.7 / 309.6 | 112.6 / 98.4 / 263.9 / 282.6 | 125.3 / 103.5 / 261.8 / 580.4 | 117.5 / 101.8 / 275.6 / 434.0 | 120.8 / 101.5 / 276.9 / 416.6 |
| house_ram | 39.4 / 6.3 / 254.6 / 277.8 | 11.4 / 9.0 / 22.5 / 28.3 | 224.0 / 194.3 / 381.9 / 587.8 | 182.6 / 152.4 / 275.4 / 427.4 | 161.2 / 146.4 / 278.1 / 304.8 | 152.3 / 151.2 / 275.7 / 283.1 | 194.1 / 164.4 / 296.7 / 392.5 | 156.4 / 150.7 / 272.3 / 364.0 | 163.1 / 151.3 / 278.2 / 351.8 | 157.1 / 135.4 / 284.6 / 339.5 | 176.3 / 145.4 / 333.7 / 406.7 | 164.4 / 146.1 / 252.5 / 337.7 |
| b5_v20 | 357.9 / 347.5 / 479.7 / 500.3 | 129.7 / 113.2 / 401.2 / 501.4 | 274.7 / 297.4 / 468.4 / 627.6 | 528.2 / 415.6 / 844.5 / 1060.0 | 445.9 / 398.6 / 733.9 / 1055.5 | 500.7 / 398.2 / 1538.5 / 2099.0 | 572.4 / 471.8 / 1454.5 / 1865.9 | 498.5 / 385.4 / 1453.9 / 2508.9 | 498.8 / 390.5 / 1479.1 / 2342.7 | 498.9 / 393.3 / 1440.8 / 2466.2 | 424.3 / 333.1 / 887.7 / 1435.7 | 374.6 / 320.9 / 881.4 / 1008.5 |
| b5_v10 | 503.5 / 477.2 / 646.9 / 649.0 | 131.4 / 111.3 / 381.9 / 448.4 | 930.6 / 814.9 / 2475.3 / 2521.1 | 489.0 / 470.0 / 766.8 / 918.1 | 466.1 / 427.1 / 769.8 / 853.5 | 343.2 / 300.7 / 669.2 / 829.3 | 388.0 / 321.0 / 730.9 / 1094.2 | 337.8 / 292.1 / 660.8 / 974.6 | 343.7 / 306.2 / 665.6 / 744.5 | 335.7 / 296.1 / 657.3 / 792.7 | 475.8 / 341.4 / 967.9 / 3572.0 | 395.1 / 331.6 / 958.0 / 1436.5 |
| b5_v15 | 335.9 / 323.3 / 451.6 / 453.3 | 251.3 / 111.2 / 1750.4 / 1994.7 | 1203.5 / 990.5 / 3223.5 / 3761.7 | 603.7 / 483.1 / 1158.3 / 1560.5 | 584.1 / 465.6 / 1234.9 / 2426.0 | 500.8 / 410.1 / 1418.7 / 2336.1 | 430.5 / 339.3 / 1324.6 / 1409.0 | 564.3 / 376.3 / 1285.4 / 5271.1 | 486.3 / 377.0 / 1296.6 / 2329.5 | 442.2 / 337.5 / 1288.0 / 2415.5 | 512.0 / 386.9 / 1829.6 / 2478.1 | 494.9 / 372.6 / 1590.2 / 2286.1 |
| b5_v30 | 377.7 / 377.7 / 448.0 / 643.7 | 130.5 / 114.0 / 357.1 / 494.5 | 176.1 / 119.0 / 528.7 / 638.7 | 434.9 / 376.4 / 602.2 / 790.5 | 410.6 / 392.8 / 548.6 / 744.3 | 186.5 / 157.3 / 381.8 / 581.4 | 156.3 / 126.5 / 378.3 / 467.2 | 167.6 / 127.0 / 380.8 / 988.1 | 160.1 / 127.4 / 380.5 / 584.7 | 154.0 / 126.9 / 378.7 / 517.9 | 155.5 / 138.0 / 319.5 / 470.9 | 154.9 / 138.7 / 351.0 / 501.1 |
| b5_v40 | 405.7 / 388.9 / 532.6 / 602.5 | 328.3 / 179.6 / 1060.4 / 2480.6 | 145.9 / 116.9 / 299.3 / 465.3 | 870.4 / 665.4 / 2687.8 / 3331.2 | 873.4 / 675.9 / 3038.9 / 3941.1 | 264.5 / 247.8 / 441.5 / 629.1 | 207.5 / 178.3 / 442.5 / 533.4 | 211.7 / 179.5 / 442.8 / 692.6 | 218.1 / 185.7 / 441.5 / 583.9 | 211.9 / 179.2 / 446.1 / 631.1 | 167.3 / 144.7 / 337.7 / 633.5 | 166.1 / 145.9 / 367.6 / 499.5 |
| b8_sudden | 16.3 / 13.5 / 37.0 / 184.0 | 17.6 / 11.7 / 52.8 / 201.6 | 10.7 / 10.3 / 18.2 / 30.6 | 9.9 / 10.1 / 17.1 / 117.2 | 10.6 / 14.4 / 16.2 / 19.2 | 20.6 / 14.0 / 69.3 / 240.2 | 12.6 / 12.4 / 22.8 / 117.9 | 14.3 / 15.4 / 23.0 / 162.7 | 13.5 / 14.8 / 26.7 / 68.8 | 13.4 / 15.2 / 23.6 / 148.7 | 17.5 / 16.2 / 37.4 / 308.7 | 12.2 / 12.3 / 19.3 / 123.9 |
| b4_sudden | 4.8 / 3.9 / 10.4 / 100.4 | 3.6 / 2.6 / 6.4 / 72.3 | 2.8 / 2.6 / 5.1 / 11.4 | 2.5 / 2.4 / 3.2 / 8.5 | 2.6 / 2.4 / 2.8 / 6.3 | 2.9 / 2.5 / 5.1 / 23.1 | 3.2 / 2.4 / 6.3 / 25.5 | 3.6 / 2.5 / 5.9 / 74.7 | 3.6 / 2.9 / 6.1 / 19.4 | 3.4 / 2.7 / 6.5 / 28.6 | 5.7 / 4.5 / 10.5 / 204.3 | 2.9 / 2.5 / 5.2 / 10.2 |
| house_car | 233.9 / 230.7 / 260.2 / 382.2 | 55.0 / 49.4 / 85.5 / 160.1 | 61.1 / 59.9 / 90.4 / 126.0 | 26.7 / 28.0 / 38.0 / 170.3 | 31.9 / 31.2 / 38.4 / 67.3 | 35.6 / 31.4 / 60.2 / 245.4 | 36.1 / 30.5 / 89.6 / 181.6 | 54.4 / 32.0 / 215.8 / 480.4 | 38.1 / 31.5 / 84.7 / 206.1 | 34.9 / 30.8 / 65.4 / 147.3 | 34.9 / 30.8 / 61.7 / 137.9 | 38.7 / 33.9 / 72.9 / 290.1 |
| floor_drop | 15.9 / 15.8 / 22.2 / 43.8 | 8.8 / 8.1 / 12.2 / 127.9 | 8.4 / 7.8 / 10.9 / 37.7 | 7.4 / 6.2 / 11.5 / 14.1 | 9.4 / 9.2 / 11.2 / 15.0 | 7.6 / 7.7 / 11.0 / 15.1 | 10.0 / 9.2 / 11.8 / 151.6 | 9.9 / 9.3 / 13.5 / 70.2 | 9.8 / 9.5 / 12.3 / 79.7 | 10.0 / 9.6 / 12.8 / 105.7 | 9.9 / 9.6 / 12.5 / 22.4 | 13.4 / 12.0 / 18.9 / 182.7 |
| **median frame, all scenes** | **16.0** | **9.4** | **9.7** | **9.2** | **10.3** | **9.7** | **10.4** | **11.3** | **11.0** | **11.1** | **11.6** | **13.1** |
| **sum of per-scene medians** | **2246** | **775** | **2720** | **2921** | **2801** | **1819** | **1764** | **1669** | **1699** | **1634** | **1656** | **1621** |

Failed runs: adaptive / house_car: rubble step failed: Overflow(128) (capacity overflow)

#### Outcome per rung (exit speed m/s / fragments / broken bonds, b5 failure mode)

| scene | oracle | blast | adaptive | islands | softening | weibull | crack_contact | rate_effects | static_fatigue | buckling | rebar | damping | realtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| house_idle | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 |
| wall_ram | 35.03 / 6 / 31 | 7.55 / 11 / 139 | 8.49 / 93 / 306 | 34.98 / 7 / 33 | 33.20 / 88 / 298 | 33.20 / 88 / 298 | 35.13 / 6 / 24 | 35.13 / 6 / 24 | 35.13 / 6 / 24 | 35.13 / 6 / 24 | 35.13 / 6 / 24 | 35.07 / 6 / 24 | 35.07 / 6 / 24 |
| house_ram | 3.03 / 31 / 187 | 2.49 / 1 / 25 | 2.49 / 1 / 0 | 3.83 / 50 / 234 | 3.85 / 31 / 143 | 3.85 / 31 / 143 | 3.22 / 29 / 177 | 3.20 / 29 / 177 | 3.20 / 29 / 177 | 3.20 / 29 / 177 | 3.11 / 38 / 194 | 3.10 / 33 / 188 | 3.10 / 33 / 188 |
| b5_v20 | 17.13 / 106 / 513 push_over | 1.37 / 86 / 731 push_over | 1.92 / 64 / 247 hole | 18.05 / 10 / 89 hole | 18.07 / 36 / 162 hole | 18.07 / 36 / 162 hole | 17.15 / 112 / 513 push_over | 17.15 / 112 / 513 push_over | 17.15 / 112 / 513 push_over | 17.15 / 112 / 513 push_over | 17.15 / 112 / 513 push_over | 17.66 / 32 / 350 push_over | 17.66 / 32 / 350 push_over |
| b5_v10 | 8.40 / 45 / 318 push_over | 0.08 / 4 / 427 push_over | 2.01 / 61 / 194 hole | 8.85 / 216 / 812 push_over | 8.79 / 36 / 196 push_over | 8.79 / 36 / 196 push_over | 8.62 / 22 / 246 push_over | 8.62 / 22 / 246 push_over | 8.62 / 22 / 246 push_over | 8.62 / 22 / 246 push_over | 8.62 / 22 / 246 push_over | 8.47 / 54 / 370 push_over | 8.47 / 54 / 370 push_over |
| b5_v15 | 12.56 / 77 / 367 push_over | 0.23 / 7 / 541 push_over | 1.24 / 283 / 844 push_over | 13.16 / 289 / 992 push_over | 12.82 / 112 / 388 push_over | 12.82 / 112 / 388 push_over | 12.59 / 92 / 421 push_over | 12.59 / 92 / 421 push_over | 12.59 / 92 / 421 push_over | 12.59 / 92 / 421 push_over | 12.59 / 92 / 421 push_over | 12.54 / 94 / 424 push_over | 12.54 / 94 / 424 push_over |
| b5_v30 | 26.25 / 107 / 599 push_over | 1.58 / 201 / 978 push_over | 2.21 / 60 / 225 hole | 27.01 / 5 / 65 hole | 27.13 / 15 / 96 hole | 27.13 / 15 / 96 hole | 27.15 / 4 / 80 hole | 27.15 / 4 / 80 hole | 27.15 / 4 / 80 hole | 27.15 / 4 / 80 hole | 27.15 / 4 / 80 hole | 27.10 / 3 / 95 hole | 27.10 / 3 / 95 hole |
| b5_v40 | 35.55 / 78 / 437 push_over | 2.00 / 304 / 1189 push_over | 2.31 / 244 / 724 push_over | 35.97 / 3 / 56 hole | 35.31 / 195 / 769 push_over | 35.31 / 195 / 769 push_over | 36.13 / 6 / 90 hole | 36.13 / 6 / 90 hole | 36.13 / 6 / 90 hole | 36.13 / 6 / 90 hole | 36.13 / 6 / 90 hole | 36.09 / 3 / 74 hole | 36.09 / 3 / 74 hole |
| b8_sudden | 0.00 / 11 / 11 | 0.00 / 2 / 0 | 0.00 / 53 / 54 | 0.00 / 34 / 35 | 0.00 / 13 / 13 | 0.00 / 13 / 13 | 0.00 / 28 / 29 | 0.00 / 28 / 29 | 0.00 / 28 / 29 | 0.00 / 28 / 29 | 0.00 / 28 / 29 | 0.00 / 35 / 36 | 0.00 / 35 / 36 |
| b4_sudden | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 | 0.00 / 1 / 0 |
| house_car | 9.05 / 33 / 135 | 2.44 / 396 / 947 | 1.82 / 620 / 1252 | 9.09 / 332 / 790 | 9.11 / 31 / 71 | 9.11 / 31 / 71 | 9.05 / 31 / 138 | 9.05 / 31 / 138 | 9.05 / 31 / 138 | 9.05 / 31 / 138 | 9.05 / 31 / 138 | 9.03 / 33 / 137 | 9.03 / 33 / 137 |
| floor_drop | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 | 0.00 / 2 / 0 |

#### Accuracy against the oracle per rung

Exit speed error is |ours - oracle| / impact speed over the struck scenes; bond and fragment errors are |log2(ours / oracle)| (1 = a factor of two), over scenes where the oracle breaks bonds; failure mode over b5's five speeds.

| rung | mean exit-speed error | mean broken-bond error (factor 2^x) | mean fragment error (2^x) | b5 failure mode matches | scenes within 25 % on broken bonds |
|---|---|---|---|---|---|
| blast | 69.1 % | 1.66 | 2.44 | 5 / 5 | 0 / 9 |
| adaptive | 64.2 % | 2.39 | 2.33 | 2 / 5 | 0 / 9 |
| islands | 3.3 % | 1.79 | 2.51 | 2 / 5 | 1 / 9 |
| softening | 3.6 % | 1.19 | 1.20 | 3 / 5 | 3 / 9 |
| weibull | 3.6 % | 1.19 | 1.20 | 3 / 5 | 3 / 9 |
| crack_contact | 1.3 % | 0.85 | 1.26 | 3 / 5 | 6 / 9 |
| rate_effects | 1.2 % | 0.85 | 1.26 | 3 / 5 | 6 / 9 |
| static_fatigue | 1.2 % | 0.85 | 1.26 | 3 / 5 | 6 / 9 |
| buckling | 1.2 % | 0.85 | 1.26 | 3 / 5 | 6 / 9 |
| rebar | 1.1 % | 0.85 | 1.28 | 3 / 5 | 6 / 9 |
| damping | 1.2 % | 0.92 | 1.54 | 3 / 5 | 5 / 9 |
| realtime | 1.2 % | 0.92 | 1.54 | 3 / 5 | 5 / 9 |

b5's failure mode is scored at 0.25 s, by which time most of the wall has detached in
the oracle at every speed (push_over). Punch-through shows in the exit speed instead:
the ram keeps 84-89 % of its speed.

### Reading the ladder

- **blast → adaptive: no accuracy, cheaper.**
  - Without islands, both quasi-static and explicit stress solves leave the ram to the
    rigid engine. It stops dead against b5's anchored wall (exit 0.1-2 m/s against the
    oracle's 8-36) and against the free wall (8 against 35).
  - The wall then fractures under the rigid stop's impulse: 2-3× too many bonds.
  - This is the limitation the user named. A quasi-static coupling cannot produce
    punch-through, because the engine has already decided the motion.
  - Quasi-static costs more than adaptive on every impact scene (median frame 16.0 vs
    9.4 ms), because of the solve-to-convergence and mechanism cost of §4.
- **adaptive → islands: the step that matters for impacts.**
  - Exit-speed error drops from 64 % to 3.3 %, with b5 at 8.9-36 m/s against the
    oracle's 8.4-35.6.
  - Broken bonds are still wrong, by a factor of 3.5 on average: threshold bonds over-break
    at low speed (b5_v10: 812 against 318; house_car 790 against 135).
  - Cost: island frames are explicit substeps, so the sum of medians goes from 775 to
    2,720 ms.
- **islands → softening:**
  - Fracture energy halves the bond error (1.79 → 1.19) and the fragment error
    (2.51 → 1.20).
  - It costs about 7 % more in the sum of medians. Some scenes get dearer (wall_ram
    103 → 308 ms median), others cheaper (b5_v15 990 → 483 ms, b5_v10 815 → 470 ms).
- **weibull: inert.** No material carries a Weibull modulus; digests are identical on
  all 12 scenes.
- **softening → crack_contact: the step that matters for damage.**
  - Bond error drops from 1.19 to 0.85, and 6 of 9 breaking scenes come within 25 % of
    the oracle (from 3).
  - Exit-speed error drops from 3.6 to 1.3 %.
  - It is also cheaper: 2,801 → 1,819 ms in the sum of medians. Cracked joints that keep
    carrying compression stop the over-fragmentation that made islands large and long.
- **rate_effects, static_fatigue, buckling, rebar: no measurable effect.**
  - rate_effects changes 1 of 12 scenes; static_fatigue and buckling change none (no
    fatigue data inside 0.25 s, no bond with a buckling length); rebar changes
    house_ram (177 → 194 broken bonds against the oracle's 187).
  - Their cost differences are within noise: the digests are identical on the inert
    rungs, so the work is identical.
- **rebar → damping (= realtime): a small step away from the oracle.**
  - b5_v20 goes from 513 broken bonds (equal to the oracle's 513) to 350. Over the set,
    the bond error goes from 0.85 to 0.92.
  - The oracle runs damping too, so this is a GPU-pipeline difference in the dashpot
    path. The certified step uses the bonds' own dashpots (`own_dashpot_step`) on the
    GPU, not the oracle's. Worth a follow-up.

## Recommendation

1. **Keep the real-time configuration.** The Blast-equivalent configuration is not a
   cheaper fallback on this pipeline:
   - It costs more than the real-time configuration on the accuracy set (sum of medians
     2,246 vs 1,621 ms).
   - It is wrong wherever an impact matters: no punch-through, and 2-3× over-fracture.
   - The balance point is the top of the ladder. Islands and crack contact buy nearly
     all the accuracy, and crack contact pays for itself.
   - The remaining features cost nothing measurable.
2. **To beat Blast's cost, attack the per-frame floor, not the physics.**
   - An idle house costs 3-7 ms here against Blast's 0.03 ms. Even a 1-5 ms rammed
     frame for Blast is below our floor.
   - The gap is per-frame host↔GPU round trips and engine coupling (several readbacks a
     frame), which is Stage 2 of GPU-resident coupling (parent session).
   - Second, skip whole frames whose inputs do not change: adaptive mode already does
     this for settled clusters, and Blast does it for idle actors.
3. **For quasi-static use, should we adopt a per-frame solve budget like Blast's?**
   This is your call. Measured: `STRESS_GPU_STATICS_NEWTON=10` takes the villa from
   1.9 s to 0.65 s a frame with an identical outcome, but changes house-2story's rammed
   outcome (7 broken bonds instead of 34). If it is adopted, it should be an explicit
   engine-wide setting.
4. **Follow-ups:**
   - the GPU dashpot difference (b5_v20 513 → 350);
   - rubble's grid overflow on house_car without islands;
   - `scenes::rammed` placement (from chunk centres);
   - the Blast harness's single-material flattening, if Blast outcomes are to be
     compared.

## Reproduce

```bash
cd engine/stress-oracle
export CARGO_TARGET_DIR=<scratch> STRESS_GPU_FIXED_HORIZON=1
cargo build --release -p stress-rubble --features gpu --example stress_bench
for c in blast adaptive islands softening weibull crack_contact rate_effects static_fatigue buckling rebar damping realtime; do
  $CARGO_TARGET_DIR/release/examples/stress_bench --set accuracy --config $c >> runs/ladder_$c.jsonl; done
$CARGO_TARGET_DIR/release/examples/stress_bench --solver oracle --set accuracy > runs/oracle_accuracy.jsonl
for c in blast realtime; do $CARGO_TARGET_DIR/release/examples/stress_bench --set packs --config $c --repeats 3 > runs/packs_$c.jsonl; done
python3 scripts/blast_ladder_report.py ladder runs; python3 scripts/blast_ladder_report.py packs runs
cargo run --release -p stress-ref --example blast_forces -- packs/house-1story.json --blast-inputs > forces/ours_house-1story.json
# PhysX: cd blast/blast-stress-solver-rs; cargo build --release --features bench-support --example blast_ladder_bench
#   blast_ladder_bench forces house-1story 3000 25 > forces/blast_house-1story.json
#   blast_ladder_bench bench house-1story ram 15 1 -5.5 >> runs/blast_packs.jsonl  (surface x from the pack's hulls)
python3 scripts/blast_force_agreement.py forces house-1story; python3 scripts/blast_support_check.py forces house-1story
```

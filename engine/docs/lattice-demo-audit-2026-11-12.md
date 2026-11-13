# Audit of the WebGPU lattice demo, and the activity-cost plan it led to

Written 2026-11-12. The subject is a 5,400-line WebGPU destruction demo (WGSL, no physics
library), called **the lattice demo** here. It was read in full: its design notes, its four
shader files and its frame loop. Nothing in this repository reads or depends on it.

The audit asked two questions: what in it is a real win for Rubble, and what is a loss we
must not import by accident.

## Verdict

- **Its physics is not a win.** Rubble is ahead on every solver capability (table below).
- **Its architecture is the lesson.** Every pass runs over a GPU-built list of what is
  active, dispatched indirectly, with no readback in the loop. Rubble does not do this yet in
  worlds with destructibles ("Where Rubble pays for the whole world").
- **Its scenes and failure cases are worth having**, written again as our own.

## What the demo is

- **Model.** Every chunk is a box and a rigid body. Every shared face is a bond sampled at
  4 Gauss points with normal and shear springs. One explicit loop at 16 substeps per 60 Hz
  frame gives static loads (as the steady state), impact stress and post-failure hinging.
- **Softening.** Explicit stability at that step allows a modulus near 6·10⁷ Pa, about 450×
  softer than concrete. Chunks still joined to the foundation are therefore drawn and
  collided at their rest pose, and their lattice is geometrically linear so it cannot buckle.
- **Run-ahead clock.** The hidden lattice takes 4 steps per real substep so that softness
  does not delay collapse; velocities are divided by 4 when a chunk is released.
- **Impact.** An impedance term `ρ·c·Δv` on the impact-velocity jump across a bond restores
  the stress the softening removed.
- **After failure.** Cracked interfaces (compression and friction only) and reinforcement
  ties with a plastic stretch and a plastic moment.
- **Contacts.** Box–box SAT with clipping, one Jacobi pass per substep with mass splitting,
  no warm start.
- **Connectivity.** FastSV-style minimum-label hooking with `atomicMin`, driven by a small
  GPU state machine and indirect dispatch.
- **Large parts.** 40 or more chunks released together become one rigid body (one per
  structure), with no stress inside.
- **Activity.** Active chunks, bonds and manifolds are compacted into lists each frame; every
  substep pass is dispatched indirectly over them. The standing structure's broadphase grid
  is sorted once at load. The only readback is a 128-byte counter block, never blocking.

Its own measurements (an iPhone, before its broadphase rewrite): 12.7k chunks 0.65 ms idle and
6.1 ms at the peak of a two-building demolition; 91k chunks 1.95 ms and 13.0 ms. The rewrite
was timed only on a software adapter (narrow phase 359 ms to 32 ms).

## Comparison

| Area | Lattice demo | Rubble | Take? |
|---|---|---|---|
| Stress model | Explicit lattice, modulus softened about 450× | Quasi-static solve (`blast_parity`), LoadPath, Threshold, explicit oracle at true stiffness | No: lossy |
| Response lag | Lattice clock runs 4× ahead | None to fix: the static solve has no lag | No: an approximation by its own account |
| Impact stress | Impedance term added back | Hertz peak (`impact.slang`), impact islands, GPU punch | No |
| Bond law | 4 Gauss points, cracked state, tie with plastic hinge | Damage joint with softening, rebar tie and dowel (`stress-ref/src/joint.rs`) | No: ours is the fuller law |
| Contacts | One Jacobi pass, no warm start; millimetres of sink per layer | AVBD, coloured Gauss–Seidel, persistent warm-started contacts | No |
| Shapes | Boxes, spheres | Hulls to 64 vertices, capsules, heightfield | No |
| Fallen parts | One rigid body per structure, no stress inside | Every cluster is a rigid compound with stress inside | No |
| Connectivity | FastSV on GPU | Union-find and minimum-label on GPU (M2–M4) | Have it |
| Allocation | `atomicAdd` slot claims | Deterministic scans, reserved at placement | No: breaks determinism |
| Overflow | Silent drops (96 candidates, 48 hits, 1,024 per bucket, 4 blasts) | The step fails with an error | No: against AGENTS.md |
| Mass | Fireball deletes chunks at random; rubble keeps 61 % of the volume | Crushed mass conserved by default | No |
| Blast shielding | Solid thickness binned 15° × 32 shells | Exact line of sight (`blast_threshold.rs`) | No |
| Whole world asleep | Frame skipped | Step skipped (`gpu.rs:2679`) | Have it |
| Cost when a little is active | Follows the active lists | Follows the whole world (next section) | **Yes: the plan** |
| Structure broadphase | Standing grid sorted once; 16-byte records | Each chunk collider inserted separately | **Yes: S6** |
| Hanging-structure cases | Explicit wake rules, each from a bug | Wake on touch | **Yes: tests, S1** |
| Scenes | Bearing-wall, framed, tower, chimney, mixed city | Lab has houses, glass tower, packs | **Yes: S1** |
| Strength sizing | One pass from settled utilisation | Authored in Shardwright | An idea for Shardwright |

## Where Rubble pays for the whole world

Found by reading the step and fracture encoders (2026-11-12).

| # | Cost that follows total size | Where |
|---|---|---|
| 1 | The first GPU fracture placed in a world turns the awake-only broadphase off: every collider is bounded, inserted and queried every step | `engine-world/src/threshold.rs:1211-1216` |
| 2 | `fr_loads` and `fr_breaks` run over every chunk and bond of every destructible whenever anything is awake | `stress-gpu/src/fracture.rs:1835,1843` |
| 3 | The fracture ingest copies full body capacity and scans contact and chunk capacity | `stress-gpu/src/ingest.rs:247,273,297` |
| 4 | Strike prediction runs over all chunks, bonds and body slots while a projectile exists | `fracture.rs:1865-1900` |
| 5 | One break sizes the split over the whole shared fracture | `fracture.slang:405-411` |
| 6 | Per-body passes: awake-list build, CSR, colour tiles, pose pack, contact-hash clear | `engine-exec/src/rigid_plan.rs:746-764,792-794,1043,653` |
| 7 | The pose readback copies every body; fracture event lists are capacity-sized | `engine-world/src/readback.rs:120,153` |
| 8 | Blast frames block on a whole-world body download, submit once per slot, and rebuild the occluder grid once per slot | `engine-world/src/blast_threshold.rs:346,471,576-591` |
| 9 | No structure-level broadphase bound; about 39 M bucket visits for 586 k accepted pairs at the 155k-body peak | `world_step.slang:988-995`, `fracture-grid-performance-2026-10-15.md:211` |
| 10 | A 2^20 chunk ceiling and no shared prototypes | `fracture.rs:1174-1177` |

## The plan

**Measure: idle tax** = time for (N sleeping structures + active set A) − time for A alone.
Target: flat in N and near zero, at N = 64, 256 and 1,024.

Every stage is lossless: identical digests, or exact pair-set equality.

| Stage | What |
|---|---|
| S0 | This document; baseline idle tax; broadphase visits classified by rejection reason |
| S1 | District scenes and blast weapons in the fracture lab; unsupported-sleeper tests |
| S2 | Fracture passes dispatched over loaded chunks, their bonds, and active destructibles |
| S3 | GPU splits mark their colliders as moved, so the awake broadphase stays on |
| S4 | Per-body passes sized by the awake set |
| S5 | Readbacks sized by what moved; blasts without the blocking whole-world download |
| S6 | Structure-level broadphase: a body-frame chunk grid per compound, or sorted buckets |
| S7 | Shared prototypes and a higher chunk ceiling |

### Why S2 can be exact

Under Threshold a bond breaks when `max(load[a], load[b]) > capacity`, and load comes only
from contact rows, blasts and host breaks. A chunk with no load contributes exactly zero. So
the passes can run over loaded chunks and their incident bonds alone and give the same
result. LoadPath, strike prediction and the split work on a region, so they take a
per-destructible active bit instead.

### Why S6 can beat the demo

An intact piece's chunks never move relative to its body. Their grid can be built once in the
body frame, at placement and again for the children of a split. The piece then inserts one
bound per step whether it sleeps or falls, and a mover that overlaps it walks the local grid
in the body's frame. The demo only gets this for the standing structure, at rest.

Broadphase ideas already measured here with no gain, not to be repeated: a larger hash,
Morton-ordered threads, a moving flag in the chain word, a forward-cell split grid, pair
batching (`fracture-grid-performance-2026-10-15.md:204-246`).

## Not adopted

Softened modulus; the run-ahead lattice clock; Jacobi contacts without warm start; atomic
slot claims; silent caps; velocity clamps (400 m/s bodies, 140 m/s blast push); random chunk
deletion and reduced-volume rubble; binned blast shielding; one rigid body per structure.

## An idea for Shardwright

The demo sizes joints in one pass: with the structure settled under its own weight it reads
each joint's utilisation and raises the strength of any above 1/5 until it sits there (72
joints in its test building). Settled stress does not depend on strength, so one pass is
exact. It changes authored strengths, so it belongs in the exporter, by the user's decision.

## Measurements

### S0 baseline (2026-11-12, M3 Max, one GPU job at a time)

`CITY_SLEEP=1 CITY_POOL=32768 CITY_K=1,64,256,1024 cargo test -p engine-world --release
--test city_scale -- --ignored --nocapture --test-threads=1`. Wall time per step, median ms
(step, then waiting for its frame). K houses of 715 chunks; `demolish` strikes one house,
`war` strikes 20, `drive` moves one kinematic car that touches nothing.

| K | chunks | drive | demolish | war | war: host part |
|---|---|---|---|---|---|
| 1 | 715 | 1.75 | 2.77 | 2.77 | 0.32 |
| 64 | 45,760 | 1.45 | 3.25 | 4.21 | 0.40 |
| 256 | 183,040 | 1.87 | 4.33 | 5.48 | 0.58 |
| 1,024 | 732,160 | 3.53 | 9.80 | 17.11 | 6.83 |

**Idle tax at K = 1,024:** 7.0 ms on `demolish` (9.80 against 2.77), with the same 31 to 51
bodies awake.

**What it is made of.** Timestamp marks could not attribute it: per-phase marks put the growth
in `coloring` (+3.1 ms), per-dispatch marks put it in `ccd_pick_key` (+2.6 ms), and neither
kernel's work grows with K. Switching one thing at a time and reading wall time did:

| K = 1,024, wall median ms | drive | demolish | war |
|---|---|---|---|
| Full broadphase (today, forced in fracture worlds) | 3.86 to 3.95 | 8.31 to 8.36 | 15.20 to 15.42 |
| Awake-only broadphase (`RUBBLE_FRACTURE_AWAKE_BROADPHASE=1`, measurement only) | 2.70 to 2.82 | 5.09 to 5.15 | 11.27 to 11.88 |

Two runs each; bonds broken and bodies awake were equal across the pair. **The forced full
broadphase (row 1) is 3.2 to 3.9 ms of the idle tax**, the largest single part.

Corrections to the table above, from these numbers:

- **Rows 2 to 5 (fracture passes over every chunk and bond) are small at this scale:** the
  `fracture` mark is 0.3 to 0.6 ms at both K = 1 and K = 1,024. They are whole-world passes,
  but cheap ones. S2 moves down the order.
- **`war` has a host cost that grows with K:** `World::step` alone is 6.8 ms at K = 1,024
  against 0.3 ms at K = 1. Not yet traced.
- About 2.3 ms of `demolish`'s idle tax remains after the broadphase and is not attributed.

**Order after measuring:** S3 (broadphase) first, then the host cost in `war`, then the
remainder; S2 only if a larger map shows it.

### S3: the awake broadphase stays on in fracture worlds

**Cause, found by comparing the two passes step by step.** A split re-frames the struck
building's static remainder: a new centre of mass and principal axes, and every collider's
local pose rewritten to match. The pieces do not move, but bounds rebuilt from the new frame
differ from the stored ones in the last bits (9,945 colliders in a 16-house war), and a cell
can differ with them.

**Fix.** The fracture already stamps every body it writes with the engine's next step as its
generation, and the engine never stamps a static body. So `list_moving_colliders` (which
reads each collider's body anyway) sees a body at rest whose generation is this step's and
asks for what a host write would get: every collider bounded and the resting table rebuilt,
on that step only (`StepModes::gpu_reframes`, `world_step.slang` `noteReframed`). No new
buffer, binding or host signal.

**A second, older inexactness, in the awake pass on hierarchical grids.** A coarse mover's
scan of finer resting colliders (`hierScanFiner`) ran the face-axis test with the mover as
the probe, where the full pass runs it with the resting collider as the probe. The axes are
the same; the order of operations is not, and pieces sharing a face plane sit on the test's
threshold. It now runs as the full pass runs it (`finerSeparatedOnFaceAxis`). This changes
which pairs the awake pass accepts at that threshold in any hull world, to the full pass's.

**Gate.** `tests/awake_broadphase_scenes.rs`: the city at war, sleep on and off, 150 steps
each, every step: candidate pairs equal as sets and every collider's bounds bit-identical
between the awake pass and the full pass on the same GPU state. The grid and ring scenes
still pass.

### Host: the grid's sizing kept between steps

Every step that adopted a split dropped the cached grid sizing and recomputed it: a median
over every collider's diameter, 4.9 ms at 732,160 colliders (`refresh_cell`). The part that
depends only on shapes and sizes is now kept (`HierGridGeometry`: base cell, colliders per
level) until a collider is added or its shape or size changes; a split changes neither. With
a reserved movable radius at or above every collider's level, the levels follow without a
scan. Checked against the one-pass sizing on 2,000 random cases.

### S1 in the browser

`fracture/?scene=district&bonds=threshold&support=1` (Chrome WebGPU, 2026-11-12): the district
loads (9,033 chunks, 22,015 bonds); three cannonballs broke 295 bonds into 75 bodies; the
physics step was 2.03 ms GPU; no console errors.

**The rocket is native only for now.** A step a blast reaches waits for the GPU
(`blast_threshold.rs`), and the lab page's step is a plain call that cannot wait: the wait
trapped and left the page's world unusable. The page's Rocket option is disabled until blasts
stop blocking (S5). `rubble::lab`'s `LabAction::BlastRay` works natively
(`lab_api::a_rocket_breaks_a_blast_loaded_building`).

### Finding and fix: sleepers left in the air

The engine wakes a sleeper on a touch from an awake body. Removing a support is not a touch,
so nothing woke what rested on it. `tests/unsupported_sleepers.rs`, before and after:

| Case | Before | After |
|---|---|---|
| A cube asleep on a static pedestal; the pedestal is destroyed (`World::destroy_body`) | stays at 1.492 m | lands at 0.455 m |
| Three cubes stacked and asleep; the bottom one is destroyed | stay at 1.489 m and 2.483 m | land at 0.459 m and 1.453 m |
| A plank asleep across two static posts; one post is destroyed | stays level at 2.092 m | tips, centre at 1.169 m |

**Fix (2026-11-12).** `destroy_body` wakes the sleeping pile on the body it removes
(`World::sleepers_reaching`): the dynamic sleepers whose bounding spheres reach the removed
body's, and in turn those that reach theirs, from the host's last polled poses (a sleeper's
pose is exact). This follows what a heightfield edit already does
(`sleepers_over_height_field`). The pile, not the first layer: waking only the middle cube
left the top one hanging, because a body that drops away under gravity separates too slowly
to wake its neighbour by contact. Bodies that only stand near wake too and sleep again.

**Supports removed on the GPU (2026-11-12).** Checked each way a support can go without a
`destroy_body` call from the host:

| Removal | Result | Test |
|---|---|---|
| A chunk crushed to dust (`CrushMass::Remove`) under a sleeping cube | Hung at 3.720 m; now drops to 3.310 m | `a_sleeper_drops_when_the_chunk_under_it_is_crushed_to_dust` |
| A building blown up under a cube asleep on its roof | Comes down (15.90 m to 3.59 m), with and without wake-on-touch | `a_sleeper_on_a_roof_comes_down_with_the_building` |
| A body past the kill bounds | Goes through `destroy_body` on the host, so it is covered | (by construction) |

The crushed chunk was the gap: it is gone, not moved, so no awake body is left where it was.
`apply_crush` now wakes the sleeping pile over each crushed chunk (`World::sleeping_pile`), a
frame or two after the crush, when its event is read back. A split needed nothing: its pieces
are written awake and, in the case tested, wake what they touch.

**Still open.**
- A sleeper whose supporter wakes and falls away from rest, where nothing was removed and
  nothing hit it (the middle of the stack, before the pile was woken as a whole). The blown
  building does not show it, because a blast throws the pieces. A piece freed by support loss
  alone, under a sleeper, is the case to try; no test of it yet.
- The lab's stress worlds never sleep (the threshold solver's own settings), so none of this
  shows there. It matters in worlds that sleep: large maps
  (`DestructibleAssetDesc::sleeping_solver_for`).
- The cubes come to rest about 4.5 cm into the ground plane after a 1 m drop (0.455 m for a
  0.5 m half extent). Seen while checking; not looked into.

### Pre-existing test failures seen while checking (not from this work)

(Also: the recorded destruction digests are out of date on main, see the quiet gate below.)

- `engine-world` `threshold::cuboid_house_breaks_and_splits_as_in_the_lab` fails the same way
  with these changes set aside: first break at frame 95 in `World` against frame 2 in the lab.
- Tests that load `fracture/assets/*.glb` fail in a worktree without the LFS files.

### After S3 and the host fix (K = 1,024, wall median ms)

| | drive | demolish | demolish p95 | war | war: host part |
|---|---|---|---|---|---|
| Before | 3.53 to 3.92 | 8.07 to 8.36 | 14.0 to 14.7 | 14.79 to 15.42 | 5.35 to 5.89 |
| S3 | 2.50 to 2.52 | 4.88 to 5.05 | 11.8 to 12.0 | 12.46 to 12.71 | 5.38 to 5.42 |
| S3 and host fix | 2.47 | 4.76 | 7.11 | 8.08 | 1.20 |
| K = 1 (the active set alone) | 1.08 | 2.93 | 3.56 | 2.75 | 0.37 |

Bonds broken and bodies awake are the same in every row (80 and 51; 2,312 and 1,058).

**Idle tax now:** 1.8 ms on `demolish` (was 5.2 to 7.0), 5.3 ms on `war` (was 12.1). `war` at
K = 1 strikes one house, so its row is not the same active set; a fair `war` baseline needs
20 houses alone.

### What remains of the idle tax (2026-11-12, after S3 and the host fix)

`drive` (one kinematic car, nothing touched), wall median ms, with the fracture frame on and
off (`RUBBLE_AB_SKIP_FRACTURE=1`, a measurement switch for scenes where nothing breaks):

| K | chunks | fracture frame on | off | the frame's cost |
|---|---|---|---|---|
| 1 | 715 | 1.21 | 0.75 | 0.46 |
| 64 | 45,760 | 1.50 | 0.84 | 0.66 |
| 1,024 | 732,160 | 2.64 | 1.27 | 1.37 |

So at K = 1,024 the remaining tax over one house (1.4 ms) is:

- **0.9 ms: the fracture frame's growth with the world.** Skipping one part at a time put
  about 0.35 ms in the ingest's per-chunk passes (clear, scan, merge) and about 0.7 ms in the
  fracture's own per-chunk and per-bond passes (`fr_loads`, `fr_breaks`); the body copy is
  free. This corrects the S0 note above: the `fracture` mark undercounted.
- **0.5 ms: the rigid step's growth** (the per-collider mover list and the per-body passes).
- The frame also costs 0.46 ms in a one-house world, and 0.2 ms of that is on the host.

**Tried and dropped: sizing the ingest's per-row and per-record passes by the live contact
count** (an indirect dispatch from a one-thread kernel). Results were identical and so was the
time (2.56 to 2.60 ms against 2.64 to 2.66 ms, two runs each): those threads return on their
first test without touching memory. The cost is in the passes that read and write a word per
chunk or per bond, which is memory traffic, not thread count. The change was reverted.

**What would remove the 0.9 ms.** The passes must not run over chunks that carry nothing:
S2 as planned. The cheap first step is a gate on the GPU: when a frame's contact rows touch
no chunk, no pulse is carried and no shot or host break is pending, the per-chunk and
per-bond passes dispatch nothing. That covers a map being driven or walked through with
nothing breaking. A frame where anything touches a chunk still pays in full until the passes
run over lists of loaded chunks. Both touch `coupling.slang` and `fracture.slang`, whose
regenerated Metal changes for every kernel in the file, so each needs the full digest guard.


### The quiet gate (2026-11-13)

Built: `GpuFracture::encode_frame_merged` (stress-gpu), used by the World for the shared and
the own fractures. Three one-thread or early-exit kernels decide on the GPU, with nothing
read back, whether a frame is quiet:

- `fr_quiet` (fracture.slang): no chunk carries a pulse into this frame (the pool half it
  reads had none claimed), and the sizes of the loads and the threshold breaks;
- `ingest_any` (coupling.slang): whether any live contact record is a chunk's;
- `ingest_gate`: the streak of such frames. From the third in a row, the ingest's seven merge
  passes, `fr_loads` and `fr_breaks` dispatch zero groups (indirect sizes).

**Why it is exact, and why two frames.** With no record and no carried pulse, the merge
writes zero counts and starts, `fr_loads` writes a zero load and an empty pulse run per live
chunk, and `fr_breaks` breaks nothing (a bond below zero capacity broke in the first of the
frames). The pulse pool has two halves, written on alternate frames, so after two such frames
every word those passes write already holds what they would write. Every other pass (crush,
load path, blast, strikes, the split's gate) runs as before and reads the same state it would
have, so the gate needs no exception for them. The host starts the streak over whenever it
changes what the skipped passes read (chunks or bonds placed or retired, a rebind) or runs a
frame without the gate. `STRESS_QUIET_GATE=0` runs without it.

**Checked.** `engine-world/tests/quiet_gate.rs` reads back every skipped frame of a small
city (a drive past sleeping houses, a war, the quiet after it) and counts the words the
skipped passes would have changed: zero in all 410 skipped frames; the war's frames ran in
full. `city_scale` gives the same awake and broken counts with the gate on and off (demolish
35 / 107 and 51 / 80, war 1,017 / 2,329 and 1,058 / 2,312 at K = 64 and 1,024).

The destruction digest guard (`stress-oracle/scripts/destruction_peak.py guard`, 32 scenes)
gives the same 32 digests on main, on this branch before the gate, and with it (the gate's
regenerated Metal included). 18 of them differ from the recorded file, on main too: the record
is out of date there, which is not from this work and is not fixed here.

**Measured** (`drive`, wall median ms, four runs each, gate off / on):

| K | chunks | gate off | gate on |
|---|---|---|---|
| 1 | 715 | 1.22 to 1.31 | 1.28 to 1.32 |
| 64 | 45,760 | 1.56 to 1.64 | 1.51 to 1.68 |
| 1,024 | 732,160 | 2.74 to 2.81 | 2.01 to 2.08 |

So the idle tax at K = 1,024 over one house is now about 0.75 ms, down from 1.5. `demolish`
and `war` do not change (5.6 and 9.1 ms): a frame in which anything touches a chunk still pays
in full, which is what the lists of loaded chunks (S2 proper) are for. Note that a world with
every body asleep already ran no fracture frame at all; the gate is for the common case of
something moving that touches no chunk.

**The 2^20 chunk ceiling, met in the browser.** `city.html?k=4096` stopped at house 1,269 with
a message that named the wrong remedy (`max_bodies`). The limit is the shared fracture's
(row 10: its scans hold 2^20 chunks, 1,466 of these houses). The message now says so, and
`CityDef::capacity` refuses a city that cannot fit before a world is made. Lifting it is S7.

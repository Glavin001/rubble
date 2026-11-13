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

Added as each stage closes.

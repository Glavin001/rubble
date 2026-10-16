# Stress-driven fracture in rubble

How rubble decides fracture with a stress solver at the physically right time, without
the trial-then-rewind loop of the earlier PhysX integration, and what it costs.

## The parts

| Part | Where | Role |
|---|---|---|
| stress-ref | `engine/stress-oracle/crates/stress-ref` | CPU f64 reference ("the oracle"): chunks with hidden 6-DOF small-strain states, the damageable joint law, explicit substeps at the certified step. The truth every pipeline is gated against. |
| stress-gpu | `engine/stress-oracle/crates/stress-gpu` | The same physics in Slang on the GPU (`shaders/world.slang`, `joint.slang`), with the coupled frame (`GpuWorld::with_engine`). |
| stress-layout | `engine/stress-oracle/crates/stress-layout` | The GPU record layout and its packing from a stress-ref solver (shared by stress-gpu and rubble's cooker). |
| stress-rubble | `engine/stress-oracle/crates/stress-rubble` | rubble's AVBD engine as the stress world's rigid-body engine (`RigidEngine`), on the stress solver's own device. |
| engine-fracture | `engine/crates/engine-fracture` | Shardwright GLB reading, the oracle input (`fracture_stress_export`), and the cooker (`fracture_cook`: a GLB level through the oracle into GPU records). |

The oracle moved here from the PhysX fork (`integration/oracle-stress-gpu` at
`ec8118341`, without history, MIT); `engine/stress-oracle/README.md` says what was left
behind. Lab assets are rebuilt reproducibly from a pinned Shardwright commit by
`engine/scripts/bake-fracture-assets.sh` (`--verify` against `scripts/fracture-assets.lock`).

## Timing: why there is no rewind

A rigid engine resolves a contact against the whole rigid body. For a fast impact on
something that breaks, that is the wrong load: a 200 kg ram at 40 m/s into a free wall
leaves at 10.9 m/s against the reference's 35 m/s (REALTIME_ROADMAP A10), and a stress
pass fed that impulse either never separates the plug or shatters the wall. The PhysX
integration therefore rewound frames and re-simulated them.

rubble's pipeline (`GpuWorld` with `redo_fractured_frames = false`, rubble as the engine)
decides before the engine step, using the reference's own rules:

1. **Routing (before the step).** Swept bounds find the pairs that may touch this frame. A
   pair is owned by the stress solve when the rigid assumption fails for it: the Hertz
   contact time is shorter than the struck body's wave round trip, or the predicted peak
   force exceeds its weakest bond (`world_coupling.rs` `rigid_assumption_fails`). Owned
   pairs and every movable body whose swept bounds touch them form an impact island.
2. **The island's frame.** The engine holds island bodies still; the stress solver
   simulates them chunk by chunk (its own contact, microsecond substeps), bonds fail at
   the substep they fail, and the pieces carry on from there. The engine then receives
   their end states. The engine never computes a rigid contact the island owns, so there
   is nothing to undo.
3. **Everything else** is the engine's; the stress solve follows it and takes its contact
   impulses as loads (quasi-static in the adaptive mode).
4. **No rewind; promotion instead.** A structure whose scripted support removal starts in
   a frame (its event time), or that fractured under the engine's loads, is promoted: its
   clusters are island seeds from the next frame (from the event's own frame for a
   removal) for as long as it keeps breaking (`PROMOTE_FRAMES` quiet frames), so the
   release dynamics are the stress solver's. The few frames where a fracture under engine
   loads stands are counted (`unverified_fractures`).

A rejected island segment (a contact plan made within it needs a smaller contact step) is
solved again from the segment's start: a local numerical retry of the stress solve, not a
frame rewind (`segment_retries`).

## Accuracy (fixed horizons, established methods, the real-time configuration)

Twelve scenes (`stress_bench --set accuracy`, 0.25 s): the A12 real-time regimes (idle
house, the free-wall ram, house-1story rammed at 12 m/s), the anchored b5 wall hit at
10–40 m/s, b8's sudden column removal, b4's sudden support loss, the house car and the
floor drop.

- rubble without rewinds gives the same outcome as PhysX GPU (with its rewinds) in 11 of
  12 scenes within the run-to-run spread of the GPU stress solve: both pipelines vary by 1-2
  fragments between repeats on b5 at 10-15 m/s and the free-wall ram, because stress-gpu's
  contact pipeline is not bit-deterministic. The one stable difference outside b8 is one
  fragment and one bond on house-1story rammed (33 / 188 against 32 / 187).
- b8's sudden column removal: rubble 10 fragments / 10 broken bonds, the oracle 11 / 11,
  PhysX GPU 14 / 14, rubble with rewinds 13 / 13.
- The anchored wall (the bouncing-projectile case): push-over at 10–15 m/s, punch-through
  at 20–40 m/s, the ram leaving at about its speed less the plug's momentum; no frame
  rewound.
- Both coupled engines differ from the standalone oracle in the number of debris
  fragments after a punch-through (e.g. b5 at 30 m/s: 3 against 107), because the engine
  owns debris contact once the island ends; exit speeds agree within 0.9 m/s.

`crates/stress-rubble/tests/integrated.rs` keeps these properties as tests.

## Accuracy against the oracle (2026-10-16 rerun)

`docs/stress-fracture-benchmark.md`, "Agreement with the oracle", has each solver's
outcomes over three repeats.

**rubble against PhysX GPU (same coupled frame, different engine).** They agree within
the GPU stress solve's own run-to-run spread on every regime except b8's sudden column
removal: rubble 10/10, PhysX 14/14, oracle 11/11, so rubble is closer.

**Both coupled pipelines against the oracle:**
- **Exit speeds.**
  - Within 0.07 m/s on the free-wall ram, the rammed house, the house car and b5 at 10–15 m/s.
  - 0.5–0.85 m/s high (2–3 %) on b5's punch-through at 20–40 m/s.
- **Debris counts differ.** After a punch-through the coupled frames keep the plug in a few large pieces: 3 fragments against the oracle's 107 at 30 m/s, and 32 against 106 at 20 m/s. At 10–15 m/s they produce more: 51–95 against 45–77.
  - The cause: once an impact island ends, the engine owns debris contact, while the oracle keeps every chunk in its own explicit contact until the end.
  - The fix would be to keep debris in its island until it separates. That changes behaviour and costs time, so it is in the recommendations.
- **Exact matches.** Idle, settling, support-loss (b4) and floor-drop outcomes match exactly.

## Cost

`engine/stress-oracle/scripts/stress_bench_report.py` turns `stress_bench` runs into the
three-way table (oracle CPU, PhysX GPU, rubble); the measured tables are in
`docs/stress-fracture-benchmark.md`. That is the 2026-10-16 rerun after the lossless host
work (`reference/bench-2026-10-16b`, M3 Max, three repeats; other sessions shared the GPU
at times, so single cells are noisy). Medians in rubble's production configuration (no
rewinds) against PhysX GPU:

| regime | PhysX GPU | rubble | |
|---|---|---|---|
| house idle | 5.2 ms | 1.5 ms | rubble 3.5x faster |
| 16-house town idle | 68 ms | 5.2 ms | 13x |
| 64-house town idle | 176 ms | 12.9 ms | 14x |
| 16 houses, one rammed | 415 ms | 302 ms | 1.4x |
| house-1story rammed | 150 ms | 150 ms | equal (GPU stress solve) |
| b5 wall, 10-40 m/s | 121-382 ms | 131-381 ms | equal within 9 % |
| house hit by a car | 40 ms | 30 ms | 1.4x |
| b8 sudden support loss | 20 ms | 10 ms | 2x |
| floor drop | 7.0 ms | 5.6 ms | 1.25x |
| free-wall ram, 40 m/s | 67 ms | 100 ms | PhysX 1.5x faster (rubble sub-steps its rigid scene for the fast ram; no CCD in rubble's Rust host) |

The CPU oracle runs 1.9-4.9 s per destruction frame.

At scale (`stress_bench --set scale`, 256 copies of house-1story, rubble only):

| regime | frame time | setup |
|---|---|---|
| 256 houses idle, after the first second | 6.8 ms median, 19.7 ms p95 | about 15 s |
| 256 houses, four rammed (first 0.5 s) | 514 ms mean | about 15 s |

In the four-rammed town the GPU waits are under a fifth of the main thread. The rest is
host work per segment that scales with the active houses and their debris: the plan
and pack of every build, snapshots, readbacks and the contact step bound. That is what
stage 3 of `docs/stress-fracture-gpu-resident.md` moves onto the GPU.

Where the time goes now:
- Idle and resting scenes: settled anchored clusters are left out of the GPU tables
  (`GpuSolver::quiet_clusters`, bit-identical), and the rest of the host work was cut
  bit for bit (see `docs/stress-fracture-recommendations.md`).
- Destruction frames are GPU-bound. On house-1story rammed, 89 % of the main thread
  waits for the stress substeps, about 44 µs each uncontended. Three dependent
  dispatches make up a substep: contact islands, contact forces, and sums with bonds.
  Three trivial dispatches cost 6 µs (`examples/dispatch_floor.rs`), so the rest is
  kernel execution. These trials found no lossless gain beyond about 5 %:
  - 512-thread islands: -5 % on two scenes, +2 % on the third, and not bit-identical;
  - a blocked bond-load gather: 0 %;
  - narrower barriers: 0 %.

  The number of substeps is the multiplier.

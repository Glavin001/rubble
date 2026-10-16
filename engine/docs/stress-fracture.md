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

## Cost

`engine/stress-oracle/scripts/stress_bench_report.py` turns `stress_bench` runs into the
three-way table (oracle CPU, PhysX GPU, rubble); the measured tables are in
`docs/stress-fracture-benchmark.md` (M3 Max, 2026-10-16, three repeats; other GPU jobs
were running, so single cells are noisy by about 10 %).

- Destruction frames cost the same on both GPU pipelines, because both spend them in the
  same stress solve: b5 and house-1story rammed within 0-10 % (median house ram 153 ms
  against 145 ms), against 1.6-5.5 s per frame for the CPU oracle.
- rubble is faster at town scale: 16 houses idle 35 ms against 37 ms median (44 against
  71 mean), 64 houses idle 140 against 160 ms, 16 houses with one rammed 408 against
  472 ms.
- rubble is slower where its rigid steps dominate: each AVBD step costs 2.4-4.7 ms even
  with a handful of bodies (dispatch and readback latency, not work), so the floor drop
  (one rigid step per frame) runs 11.8 against 5.0 ms median, and the free-wall ram, whose
  40 m/s ram makes the coupling sub-step the engine 11 times a frame, 97 against 67 ms.

Where the time goes in rubble's pipeline (house rammed, 15 frames): the stress solver's GPU
substeps about 73 % (≈44 µs per substep, three kernels: contact islands, contact forces,
sums and bonds), host transfers and splits about 8 %, syncing bodies with rubble 3 %,
rubble's rigid steps about 1 %, and the rest (about 14 %) host routing and contact planning.

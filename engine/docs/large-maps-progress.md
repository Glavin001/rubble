# Large maps: progress log

Design: [large-maps.md](large-maps.md). Benchmark: `engine-world/tests/city_scale.rs`.
M3 Max, native Metal, threshold cuboid houses (715 chunks, 1,549 bonds each) on a 2.2 km map.
Median wall ms per step (step + readback), 120 measured steps after 120 settling steps.

## L0 baseline (2026-10-30, `feat/large-maps` @ 50fced09)

Houses never asleep (the threshold solver's suggested policy):

| K | still | drive | demolish (1 house) | war (20 shots) |
|---|---|---|---|---|
| 1 | 5.0 | 5.5 | 6.1 | 5.8 |
| 4 | 5.9 | 5.8 | 6.2 | 6.5 |
| 16 | 19.1 | 22.2 | 17.7 | 19.7 |
| 64 | 74.4 | 86.6 | 74.0 | 72.5 |
| 256 | does not run: per-house GPU buffers sized to the whole world exhaust GPU memory (stalled 48 min) | | | |

Houses allowed to sleep (`CITY_SLEEP=1`):

| K | still | drive | demolish | war |
|---|---|---|---|---|
| 1 | 0.01 | 3.5 | 3.6 | 3.4 |
| 4 | 0.02 | 4.1 | 7.3 | 7.1 |
| 16 | 0.04 | 11.4 | 14.5 | 16.0 |
| 64 | 0.17 | 110.8 | 92.4 | 165.5 |

`still` with sleep is the whole-step skip. One awake body anywhere (`drive`: a car touching
nothing) runs every pass over every house.

### Why

- Each threshold destructible has its own contact ingest and fracture system. Every step
  each of them copies the world's whole body buffer (`begin_frame`), scans the world's
  whole contact capacity (`ingest_rows`, `max_rows`) and merges twice that, and allocates
  record buffers and a collider table sized to the world: cost and memory K × world.
- The rigid step dispatches most passes over all bodies and colliders.
- `profile_ms()` misses most of this (0.1 ms reported against ~12 ms of GPU work at K=16).

### Fixed on the way

- Adding a collider mid-simulation zeroed the counter block the pipelined step forwards
  its previous contact and manifold counts from: every resting contact restarted cold
  (a resting tower 1.3 mm/s → 0.16 m/s; awake threshold houses broke bonds after any
  spawn). Commit 043e8d99. Also on origin/main.

## L1/L2 so far (2026-10-30)

One fracture per world (99949fe8), placement uploads in runs (2771053a), change-only
fracture readback (a610ab80). Houses never asleep:

| K | still | drive | demolish | war |
|---|---|---|---|---|
| 1 | 3.0 | 3.1 | 3.9 | 3.0 |
| 16 | 3.6 | 4.1 | 4.0 | 4.3 |
| 64 | 5.7 | 6.1 | 6.2 | 6.6 |
| 256 | 10.5 | 10.8 | 12.6 | 12.3 |

(Other sessions ran GPU tests during these runs; ±2 ms noise.) What remains with houses
never asleep is real solver work: ~35 clusters per house stay awake.

`RUBBLE_GPU_TIMING=1`, one car among 64 sleeping houses: ~1.0 ms GPU per step over
~12 phases; host encode ~1 ms, frame build ~0.5 ms.

## Threshold sleep (2026-10-31)

### The shot is chaotic

The lab-parity cuboid-house shot (`tests/threshold.rs`) is deterministic run to run but
chaotic: moving the shot's origin 0.1 mm changes never-asleep's 47 broken bonds to 20
(nudges of ±0.1-10 mm give 20-56). Single-shot counts cannot judge a sleep policy; the
gate is an ensemble (`sleep_policy_ensemble`: 21 shots nudged -10..10 mm, house settled
240 steps first).

| Policy | Bonds broken | Bodies |
|---|---|---|
| never asleep (the reference) | 38.8 ± 16.5 [12, 56] | 101.4 ± 5.8 [92, 107] |
| sleep + archive + wake on touch (`sleeping_solver_for`) | 38.7 ± 11.6 [20, 50] | 101.1 ± 3.9 [95, 105] |
| sleep + archive | 35.2 ± 6.6 [27, 43] | 98.8 ± 3.4 [95, 102] |
| start asleep, never re-sleep (Unreal's town) | 46.3 ± 17.7 [21, 65] | 102.7 ± 6.2 [95, 109] |

None differs significantly from the reference at n = 21 (means within ~1.4 standard
errors; variance ratios under F's 5% critical value). The large-map policy is the second
row: it matches the reference's mean and spread best, and each part removes a real
mechanism by which sleep changes what awake bodies feel.

### What sleep changed, and the fixes

- **Cold restarts.** A sleeping pair produces no contact rows, so a woken body restarted
  its resting contacts from λ = 0, k = k_start. The **sleep archive**
  (`WgpuSim::enable_sleep_archive`, world_step.slang archive_*) keeps a pair's canonical
  rows when its body falls asleep (blocks from a GPU free list, past the contact
  buffer's halves) and a warm-start miss finds them. Rows are tagged with their
  colliders' local poses, so a split's re-parenting invalidates them.
- **Missing supports on the wake step.** A sleeper woken by a contact in the narrowphase
  had no resting pairs that step (the broadphase saw it asleep), so a strike's first
  frame met an unsupported cluster. A moving collider's pair with a sleeper with an
  archive now lists it, and `archive_emit` (after the broadphase) emits its archived
  resting pairs. The impact frame then matches the reference (30 bonds at the impact
  frame, as never-asleep breaks).
- **Sleepers as walls.** A sleeper woke only above the sleep speeds, so slow debris
  pressed on it as on static ground. `FLAG_WAKE_ON_TOUCH` wakes it on any touch from an
  awake body.

Cost while asleep: nothing per step. Archiving costs work over the step's contact rows
only on steps where something fell asleep; waking costs the woken bodies' archives.

### Open

- Start-asleep placement (Unreal's town) can split a sleeping cluster whose new pieces
  never report a pose (`ENSEMBLE_LENIENT` skips that check in the probe). To fix.
- The ensemble covers one asset. The town bombard (stress charter) is the next check.

## Sleep policy in the city (2026-10-31)

`CITY_SLEEP=1` (houses placed awake, settle, sleep with their contacts archived), after
`push_damping` walks only bodies that carry damping or a gravity scale (it walked every
slot on the host each step). Median ms per step; other sessions ran GPU tests meanwhile,
so wall times are noisy; host is `World::step` alone, GPU every pass from the step's
marks (`RUBBLE_GPU_TIMING=1`, a separate run):

| K | still: host | drive: host | drive: GPU |
|---|---|---|---|
| 1 | 0.01 | 0.69 | 1.5 |
| 16 | 0.01 | 0.75 | 2.5 |
| 64 | 0.01 | 0.77 | 1.4 |
| 256 | 0.07 | 1.14 | 3.1 |

A still map costs nothing. One car among sleeping houses is roughly flat in K; its GPU
time is mostly a fixed cost: a world past the sparse tail's body cap (`tail_ok`) dispatches
all 32 colours every iteration. The bench's wall column is latency (it blocks on every
frame's readback).

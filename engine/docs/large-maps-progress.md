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

## Threshold sleep is not lossless yet

The lab-parity cuboid-house shot (`tests/threshold.rs`), deterministic over 5 runs each:

| Policy | Bonds broken | Bodies |
|---|---|---|
| never asleep (the threshold solver's policy; the reference) | 47 | 104 |
| sleep at the engine's default thresholds (`THRESHOLD_SLEEP=1`) | 40 | 100 |
| start asleep, never re-sleep (`DestructibleDef::asleep`, `THRESHOLD_START_ASLEEP=1`) | 25 | 95 |

The masonry arch is the same (87 / 17) under all three. The Unreal client places the
town with `start_asleep = true`, so it runs the third row today.

Why: D1b breaks on total load, the structure's own weight plus the impact. A sleeping
cluster has no resting contacts; when woken its contacts restart cold (AVBD's penalty
ramps from `k_start`), so the first frames of a strike carry less load than on a cluster
that was awake. Rest-carry (`FLAG_CARRY_REST`/`ENGINE_REST`), which keeps resting pairs'
warm starts through sleep, walks the compact manifold store; threshold worlds use the
canonical rows the fracture ingest reads, so it is inert there (ENGINE_REST=drop gives the
same 40 / 100).

Plan (lossless, no decision needed): archive a pair's canonical contact rows (λ, k) when
it falls asleep and restore them as its warm start when the pair forms again; buildings
that start asleep get archived rows from a settle at placement. Zero cost per step while
asleep. Gate: the three rows above all give 47 / 104.

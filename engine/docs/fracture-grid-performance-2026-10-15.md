# Fracture grid performance (2026-10-15)

The goal: 60-120 physics ticks/s on the fracture lab's 49-house grid
(`veneer-house.glb` level 2, 7×7, ~155k fragment bodies) when the row shot
destroys a subset of houses, with no lossy physics changes. Measured on an
M3 Max in Chrome. The goal was stopped at ~33 ticks/s; this records why.

## Result

| | Start | End |
|---|---|---|
| Physics step at peak (~122k awake, ~950k contacts) | 36.2 ms | 30.2-30.5 ms |
| Physics step over a 60 s run | — | 30-33 ms (~130-135k awake) |
| Render | ~65 FPS (sliced steps) | ~65 FPS |

Physics runs at about a quarter of real time (dt = 1/120).

## Lossless changes that helped

| Commit | Change | Peak effect |
|---|---|---|
| `71a6c0b` | Hierarchical-grid chain links in the AABB spare word (one load per hop); overlap test before the level/cell checks | broadphase 5.0 → 3.8 ms |
| `40dc2ab` | `convex_bucket`: test the SAT's cached separating axis before bucketing (44% of hull pairs proven separated never reach the SAT) | SAT large 6.5 → 3.1 ms, manifold 4.1 → 3.0 ms |
| `f03a246` | Drop the SAT's now-redundant in-kernel cache phases | SAT large 3.1 → 2.6 ms, small 1.1 → 0.97 ms |

Each gives identical contacts (checked with the cache on and off at the same
state), and the GPU conformance suite and CPU oracles pass.

Peak per-kernel time now: primal ~11 ms, broadphase 3.8, manifold 2.9,
dual 2.8, SAT large 2.6, colouring 1.8, SAT small 0.97, classify + bucket 0.69.

## Tried and reverted (no gain)

- Capping the colour palette (empty colour dispatches cost nothing).
- Subgroup-shuffle primal reduction (bit-identical, 10.0 vs 9.6 ms).
- Field-wise body loads in primal and dual.
- Publishing each body's step displacement for neighbouring contact rows.

The primal's arithmetic is ~0.2 ms of its ~11 ms; it is bound by latency and
occupancy, and lane counts were already tuned (8 best).

## Correctness fixes found on the way

- `b46620a`: hill-climbing supports started at vertex 0, which on some cooked
  hulls lies on no face; two roof plates per house sank 8-10 cm into the floor.
  Also adds `FLAG_CONTACT_GAP` (32, opt-in) and `FLAG_WORLD_INERTIA` (64, on in
  the fracture scenes).
- `1b66f77`: speculative convex contacts under `FLAG_CONTACT_GAP`.
- `f0552fc`: near-ties in convex contact reduction go to the lower feature id,
  so resting pieces stop cycling through equivalent support points.

## Why physics stays near 33 ticks/s

The shot hits one column, but 43 of 49 houses have collapsed by 10 s (45 by
20 s). An unbonded house is a stack of fragments that stays up only while
asleep: debris wakes a neighbour and it crumbles under its own weight, so the
collapse cascades across the grid. About 130k pieces stay awake for 60 s and
more; awake count does not fall.

Measured and ruled out as fixes for the grid (all lossy, none adopted):

| Variant | Awake at ~20 s |
|---|---|
| Current rule (8-hop island gate) | ~128k |
| Size-scaled sleep (`max(v, ω·extent)`) | ~133k |
| 2-hop gate / no gate | ~124k / ~137k |
| Movers-only 1-hop gate | ~139k (still pieces sleep, but ~320 re-wake per step) |
| α = 0.99 + movers-only gate, 60 s | ~139k |

The ~6k pieces that keep moving are mostly slow slides in jammed piles (11% of
their contacts stick vs 41% for still pieces), driven by the α push-out of
overlaps formed during the collapse. Friction stiffness, contact persistence and
the asset are ruled out: intact houses have no touching hull pairs.

## Remaining route (not taken)

Simulate each intact bonded house as one compound rigid body and break it with
a stress solve over the bake's bond capacities (Shardwright's reference
approach; see the fracture README, "Breakable bonds"). Only hit houses would
become debris (~7 × 3,158 pieces instead of ~130k). It changes house behaviour,
so it was left for a later decision.

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

Each gave identical contacts when checked with the cache on and off at one
state, and the GPU conformance suite and CPU oracles pass. Correction (later):
`40dc2ab` is not exact on every state; see "Run-to-run nondeterminism".

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

## Making sleeping bodies free (later the same day)

Target use case: hundreds of hull-model houses, most asleep, plus scenes with
hundreds of thousands of bodies. Per-tick work should scale with *awake*
bodies, as in mature engines; until now almost every pass ran one thread
per collider or body.

### What other engines do

From the local sources (Jolt, PhysX incl. GPU, Newton, beat-box) and from
memory for Box2D v3, Box3D and Bullet (not available locally):

- Jolt: only the active-body list issues broadphase queries; the awake side
  owns awake-vs-sleeping pairs; sleeping is a swap-remove from the active list.
- PhysX: the GPU SAP updates only changed handles (active, non-frozen bodies);
  aggregates give a compound one broadphase entry; persistent island graph.
- Box2D v3: one tree per body type, fat AABBs and a move buffer; sleeping
  islands move to their own solver sets and cost nothing per step; persistent
  graph colours.
- Bullet: separate dynamic and fixed trees, but the island union-find runs
  over every object each step (the anti-pattern).
- Pitfalls they handle: canonical pair order, a deterministic pair order
  (not atomic-append order), tombstones in persistent structures, exact AABB
  retests.

Not adopted (each changes results; the user's decision): persistent graph
colours (Box2D, PhysX), persistent islands with deferred splitting, freeze
damping (PhysX), Jolt's per-island sleep rule.

### Lossless changes

| Commit | Change | Idle hull grid (ball rolling clear of the houses) |
|---|---|---|
| (before) | | 1.04 ms, 963 ticks/s |
| `9cf3d25` | Bounds rebuilt only for last step's awake bodies | |
| `6c37206` | Predict, velocity/sleep and sleep commit over the awake list | |
| `7641433` | Only awake colliders query the broadphase; a coarse mover finds finer resting colliders itself, its finer cells split into 64-cell chunks (one workgroup each) | 0.915 ms, 1093 |
| `de3b6a1` | Persistent resting broadphase table, rebuilt on the GPU only after sleep events; movers fill and empty an awake table | ~0.77 ms, ~1300 |
| `b46a66f` | `update_refs` over the awake list; tick-tagged warm hash (cleared once per 2046 steps instead of 2^21 words per step) | 0.62 ms, ~1590 |

These paths run on moved-only steps (under a quarter of bodies awake). At
the peak (~117k of 155k awake), forcing them was slower (32.7 vs 27.4 ms),
so the full path stays in use there.

Verification:

- Pair sets match the full broadphase pair for pair on 84 states (settle,
  hull-cuboid shot, hull grid up to 135k awake and 748k candidates).
- With `budget.deterministic` (below) on in both runs, everything off vs
  everything on is bit-identical over 1500 steps of the 3×3 hull-cuboid shot
  and 700 steps of the hull grid (up to 129k awake).

Fixes found on the way:

- Moved-only bounds rebuilt bodies that fell asleep in the last step; the
  full pass keeps their previous bounds.
- Work slot 67 is reused by the sleep pass, so the moved-only passes now
  dispatch from slot 78.
- `inspectCollisions` left `last.awake` at 0, after which `step()` skipped
  every step.

### Run-to-run nondeterminism: found

Two identical runs used to diverge after ~130-200 steps. The cause is
`sleep_island_gate`, which revokes sleep candidates in place. A revocation
written earlier in the same dispatch can already be read, so how far it
travels in 8 rounds depends on GPU scheduling, and with it which bodies fall
asleep.

`budget.deterministicSleepGate` (opt-in, default off) moves a revocation
exactly one hop per round: a body revoked in round r is marked and still
counts as a candidate for the rest of round r. With it on, identical runs are
bit-identical: three trials over 600 steps of a settling house.

It changes which bodies sleep relative to the current racy gate, which can
travel further than 8 hops, so making it the default is the user's decision.
It is also what makes long lockstep proofs of lossless changes possible.

On the hull grid there is a second source: the SAT separating-feature cache
(`40dc2ab`). It is direct-mapped, so pairs that share a slot overwrite each
other in scheduling order. On rare near pairs, the cached axis proves the
hulls separated while the full SAT, which skips near-parallel edge pairs,
finds no separating axis and emits a support-point fallback contact. First
seen at step 123 of the hull-grid shot: one pair, one contact. With cache
lookups off (params flag 8), identical runs stay bit-identical through 320
steps.

`budget.deterministic` (opt-in) turns on both the exact gate and flag 8.
Making either the default changes results. The options for the cache are:

- a deterministic cache: one atomic word per slot, with a fixed rule for
  which pair wins a slot;
- a complete SAT, so the cache can only agree with it;
- dropping the cache (~0.5 ms at peak).

### Still scaling with total bodies at idle

`build_awake_list`, `csr_clear` and the CSR prefix passes (one thread per
body), plus fixed per-colour and per-iteration dispatch counts. Next
candidates: an event-driven awake list and a CSR over awake bodies only.

### Peak: what was measured after the idle work

The hull-grid shot at ~3.5 s (113k awake, 872k contacts, 586k candidates)
takes 26.5 ms per tick (~38 ticks/s). Per kernel:

| Kernel | ms |
|---|---|
| primal (8 iterations × 12 colour dispatches) | 8.6 |
| broadphase query | 3.6 |
| convex manifold | 2.7 |
| SAT large | 2.5 |
| dual (8 iterations) | 2.5 |
| colouring | 0.87 |
| SAT small | 0.87 |
| persist | 0.87 |

Tried at the peak with `budget.deterministic` (so every run has the same
state; counts identical); none helped:

- Grid hash 2-16× colliders instead of 2× (alias walks): 3.93 → 3.69-3.82 ms
  for the query, within noise.
- Broadphase threads in Morton order of the start positions: no gain (the
  scene's collider order is already spatially coherent).
- A moving flag in each chain word, so ownership is decided before the
  collider/body loads: no gain.

The query is bound by the AABB loads of the chain entries themselves (about
39M entry visits per tick).

The primal runs at about 0.65 ns per body-contact visit, loading ~110 bytes
each, which is close to memory bandwidth. Its colours take 0.33, 0.26, 0.20,
0.13, 0.13, 0.07, 0.07 ms per iteration; the empty headroom dispatches are
free.

With SAT-cache lookups off, SAT large rises from 2.5 to 7.3 ms. A
deterministic cache, not dropping it, is the right fix for its race.

60 ticks/s at this peak needs roughly 10 ms less per tick. No lossless change
found so far comes close; the remaining levers change behaviour (fewer awake
pieces, as in "Remaining route" above).

### The lab's frame loop was starving physics (`ebfe128`)

In the interactive lab, the grid ran only 2-4 ticks/s during the shot, and
even at idle, while the GPU sat mostly idle. The slice controller aimed for
16.0 ms frames, but 60 Hz vsync paces frames at 16.7 ms, so every frame read
as over budget and the rate sank to its floor.

The rate now follows whether the GPU finished the work submitted two frames
earlier. In the lab during the row shot it runs at 22-30 ticks/s and 50-58
FPS. Rendering takes ~2.4 ms per frame (16.7M non-indexed vertices); the
rest of the GPU goes to physics. The ceiling remains the ~26.5 ms per tick
physics cost at the peak.

### Headless A/B results (`fracture/headless.mjs`, deterministic, tick 480)

| Change | Counts vs HEAD | Kernel effect |
|---|---|---|
| Split grid: movers/resting tables, mover pairs visited once (forward cells) | identical | broadphase query 3.85 → 4.04 ms (no gain) |
| Per-thread pair batching (one atomic per 8 pairs) | identical | 3.85 → 3.82 ms (no gain) |
| `[ForceUnroll]` of the solver's small loops | differ | primal 8.8 → 8.1 ms at diverged states; in default mode 8.2 → 9.0 ms (no gain) |
| Unrolling only the zeroing and negation loops | differ | no gain |
| Skip island-gate rounds after one revokes nothing (a fixed point) | identical | none at the peak: revocations spread through all 8 rounds |
| Fewer spare colour dispatches and colouring rounds (budget) | (not checked) | ~0.1 ms per tick |
| Chrome `--enable-dawn-features=disable_robustness` (no bounds checks) | differ | 32.7 → 30.5 ms per tick |

The broadphase query is bound neither by chain-entry visits nor by atomic
contention.

Any edit to the solver kernels changes results in the last bits: Metal
fuses floating-point operations differently once the code shape changes.
This happens even for an edit that cannot change any value. The unchanged
shaders reproduce exactly (614002 candidates on repeat runs). Accepting such
rounding-only changes is the user's decision.

Turning off Dawn's bounds checks (no shader source change) also changes the
results, which confirms the cause: on this stack (Chrome, Dawn, Metal), any
change to the generated kernel code can change floating-point rounding.
Under a bit-identical rule, only changes that leave the arithmetic kernels'
generated code untouched qualify.

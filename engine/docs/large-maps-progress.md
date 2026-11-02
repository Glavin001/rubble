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

## 1,024 houses (2026-10-31, 4911eb28)

`CITY_POOL=32768 CITY_SLEEP=1 CITY_K=1024`: 732,160 chunks on the 2.2 km map, one
shared piece pool. The GPU was not idle (another app reported 100% utilisation), so
treat the wall column as an upper bound until a quiet rerun.

| scenario | wall median ms | wall p95 ms | host median ms | awake | bonds broken |
|---|---|---|---|---|---|
| still | 0.02 | 0.13 | 0.02 | 0 | 0 |
| drive | 14.21 | 14.48 | 0.62 | 0 | 0 |
| war | 20.64 | 21.39 | 0.68 | 1001 | 2410 |

- Still costs nothing and the host stays under 0.7 ms at every size.
- Drive moves one kinematic car past sleeping houses, waking nothing. Its wall time
  grows with the map (about 3 ms at 256 houses, 14 ms at 1,024), so some GPU pass
  still runs over every collider or body once anything is awake. That is the next
  L1 target: awake-indexed passes, a persistent broadphase for sleepers and
  statics, and the fixed per-colour dispatches.

Two "hangs" on the way were not engine cost:
- The first step after a shader change took about 2 minutes. That was the one-time
  Metal pipeline compile, which I kept killing before it was cached.
- Placement deadlocked because each house's uploads were never waited on (fixed in
  4911eb28).

## Awake-only passes (2026-11-01, `claude/awake-only-passes`)

The next L1 target above. What drive's cost was made of, measured on `World` at 256 and
1,024 houses (`CITY_SLEEP=1 CITY_POOL=32768`; A/B by switching single kernels or
features off):

- **The resting-pair carry: about half.** `carry_rest_pairs` re-emits every manifold of a
  pair at rest (both bodies asleep or static) every step, and each one goes through the
  convex narrowphase (SAT and clipping), the warm hash, persistence, the CSR passes and
  four dual iterations: 9,708 pairs and 26,254 contact rows at 64 houses, about 150,000
  and 400,000 at 1,024. Leaving out `carry_rest_pairs` alone took K=1,024 drive from
  13.3 to 7.4 ms.
- **The broadphase over every collider** (732,160 at K=1,024): bounds, insert and query
  each ran one thread per collider, about 1 ms.
- **wgpu's indirect-call validation:** a validation pass before each of the step's
  dozens of indirect dispatches, about 0.5 ms a step at any size.
- The empty colour dispatches (four colours plus the sparse tail, per iteration) cost
  about 0.6 ms at any size. The per-body passes cost about 0.1 ms at K=256.

### Done (lossless)

- **No indirect-call validation on native devices** (`gpu::instance_flags`). Every
  indirect argument the plans write is within the device limits already; the terrain's
  MPM dispatches are checked against the workgroup limit at `terrain_enable`, with the
  limit named. `WGPU_VALIDATION_INDIRECT_CALL=1` restores it. Pure dispatch overhead.
- **Awake-indexed broadphase** (`EngineFeatures::awake_broadphase`, hierarchical grids;
  `ENGINE_BROADPHASE=full` or the lab's `?broadphase=full` turn it off). Static and
  sleeping colliders stay in the fracture lab's persistent resting table; a step lists the
  colliders of its moving bodies (`list_moving_colliders`, one cheap scan reading each
  collider's body flags) and bounds, inserts and queries only those; coarse movers scan
  finer resting cells. The table is rebuilt on the GPU after a step with sleep events, and
  on the host's request after any write to bodies, colliders, materials, hulls or
  params, after commands, and after any step that did not keep it. Characters and soft
  bodies keep the full pass. `tests/awake_broadphase.rs` runs both on the same GPU state
  every sixth step of a 480-step scene (piles that settle and sleep, a coarse mover over
  them, moving and stopped kinematics, thrown cubes, a CCD bullet): the same candidate
  pairs and bit-identical collider bounds every time.
- **`broadphase_fast` sized from the fast list** (`write_fast_args`): it launched all
  16,384 workgroups whenever the fast list was scheduled; now one per listed collider,
  numbered as before (compared against the full-size dispatch in the same test).

City, native, wall ms per step (median of 120 waited steps; GPU of other sessions was
busy at times, ±0.3 ms). "Before" is this branch with `ENGINE_BROADPHASE=full
WGPU_VALIDATION_INDIRECT_CALL=1`, run back to back with "after":

| K | scenario | before | after | after, carry off (decision below) |
|---|---|---|---|---|
| 64 | drive | 2.38 | 1.96 | 1.60 |
| 64 | war | 4.96 | 4.96 | 4.64 |
| 256 | drive | 4.91 | 4.26 | 2.67 |
| 256 | war | 8.35 | 8.11 | 6.62 |
| 1,024 | drive | 13.82 | 12.39 | 6.35 |
| 1,024 | war | 20.49 | 16.85 | 10.94 |

War at K=64 breaks the same 2,630 bonds with 1,034 bodies awake before and after; at the
larger K the shots are chaotic (see the threshold notes above) and the counts move by a
few percent either way.

GPU time of drive from the step's pass marks (`RUBBLE_GPU_TIMING=1`): K=1,024 6.1 ms
after (the carry's rows dominate it); with the carry off, 1.0 ms at K=256 and 1.3 ms at
K=1,024, roughly flat. The waited wall time is still not flat with the carry off (2.7 →
6.4 ms): `poll_blocking` waits 2.2 ms at K=256 and 6.0 ms at K=1,024 for work the marks
do not see. That is outside the rigid plan: the threshold fracture's
`fr_loads` / `fr_breaks` still run over every chunk and bond each frame (L2 §4), and the
readback copies a pose buffer sized to every body (L2 §2).

Fracture lab grid (ball in flight, one CCD ball among 49 sleeping houses, 64 bodies
awake): GPU 1.93 → 1.71 ms per step, pipelined 3.13 → 2.62 ms (same build,
`?broadphase=full` vs default). The lab plan's 1.0 ms is not reached: what is left is
the carry and fixed per-step work (eight solver iterations of colour dispatches and the
sparse tail, the stabilization pass, the per-body passes).

### Decision: the resting-pair carry

`carry_rest` (on by default) keeps a resting pair's contacts alive through sleep by
re-emitting its manifold every step. That was written for the JS host, where contact
reuse (`FLAG_REUSE_CONTACTS`) copied the rows with their warm λ. `World`'s large-map
policy runs canonical contacts without contact reuse, so each carried pair is
re-detected from scratch every step: fresh rows with λ = 0 and k = 1 (verified on the
GPU: all 6,524 carried rows at K=16 have λN = 0), which persistence leaves as they are
and the dual leaves alone (both ends at rest). The rows do nothing except:

1. **They mask the sleep archive.** When a body wakes, its pairs find last step's
   carried row in the warm hash (λ = 0), so `archiveFind` never runs for them: the woken
   body restarts its resting contacts cold, which is what the archive was built to avoid
   (the threshold-sleep notes above describe the archive as if sleeping pairs had no
   rows).
2. **They give a body woken inside the narrowphase its supports that same step**, and
   wake propagates through them within the step (deterministically one hop when a sphere
   pair wakes the body in `convex_classify`, racily further in `convex_manifold`).

So removing the carry's cost cannot be made bit-for-bit lossless cheaply. Two options:

- **A. Turn the carry off where the sleep archive is on** (`ENGINE_REST=drop` today).
  Sleeping pairs then have no rows, as the archive design assumed: a woken body's pairs
  warm-start from the archive, and `archive_emit` gives a touched sleeper its resting
  pairs on the wake step. This changes results: woken pairs start warm instead of cold,
  and wakes no longer chain through resting pairs within a step. It is the large win in
  the table above (K=1,024 drive 12.4 → 6.4 ms, war 16.9 → 10.9 ms). It needs the
  `sleep_policy_ensemble` gate rerun with it before adoption. **Recommended.**
- **B. Keep today's behaviour and make it cheap**: a persistent store of the resting
  pairs (no per-step re-emission), the "virtual hit" (a pair that was carried last step
  skips `archiveFind`, as its λ = 0 row would have made it), and the woken bodies' pairs
  emitted or replayed in the dispatch order that reproduces today's wake propagation.
  Lossless up to which of today's racy outcomes it picks, but a large change for
  behaviour that is itself an accident of the two features meeting.

Not adopted: either changes code paths whose results the user signs off.

**Option A's gate (2026-11-01): failed, not adopted.** The user chose A subject to the
`sleep_policy_ensemble` gate (21 shots nudged -10..10 mm, 120 frames; bonds broken and
bodies, mean ± standard deviation [min, max]; `THRESHOLD_CARRY=on|off` forces the carry):

| Policy | Bonds broken | Bodies |
|---|---|---|
| never asleep (the reference) | 57.3 ± 2.5 [53, 62] | 107.5 ± 0.6 [107, 109] |
| sleep + archive + wake on touch, carry on (today) | 56.3 ± 1.2 [55, 60] | 108.0 ± 0.2 [108, 109] |
| sleep + archive + wake on touch, carry off (A) | 60.9 ± 2.3 [57, 66] | 107.7 ± 0.7 [106, 109] |

Bodies match. Bonds do not: carry off breaks 3.6 more on average than the reference,
about 4.9 standard errors at n = 21 (today's carry-on is 1.0 below, about 1.6), and its
range reaches 66 against the reference's 62. By the gate's rule (the breakage
distribution within the reference's spread, mean and range) it fails, so the carry
stays on by default. The numbers are identical with `ENGINE_BROADPHASE=full
WGPU_VALIDATION_INDIRECT_CALL=1`, so this branch's other changes do not move them. (They
differ from the October ensemble above because the house and shot have changed since;
the reference was rerun alongside.)

**Why carry-off breaks more (2026-11-01).** The two suspects, each a measurement variant
behind a switch with today's default untouched (`EngineFeatures::archive_cold`,
`ENGINE_ARCHIVE=cold`: a woken pair skips the archive and starts cold;
`archive_spread_rounds`, `ENGINE_ARCHIVE_SPREAD=n`: before `archive_emit`, each touched
sleeper lists the sleepers in its archive, one hop a round, so their archived pairs are
emitted too and a wake can chain through them within the step), plus a third check
(`RUBBLE_SKIP_KERNELS=archive_emit`: no archived pairs emitted on a touch at all). Same
ensemble:

| Policy | Bonds broken | Bodies |
|---|---|---|
| never asleep (the reference) | 57.3 ± 2.5 [53, 62] | 107.5 ± 0.6 [107, 109] |
| carry on (today) | 56.3 ± 1.2 [55, 60] | 108.0 ± 0.2 [108, 109] |
| carry off | 60.9 ± 2.3 [57, 66] | 107.7 ± 0.7 [106, 109] |
| carry off, cold wake (a) | 61.5 ± 2.6 [57, 66] | 107.8 ± 0.9 [106, 109] |
| carry off, spread wake, 8 rounds (b) | 65.7 ± 1.6 [63, 69] | 108.8 ± 0.7 [108, 110] |
| carry off, cold and spread wake | 67.4 ± 2.4 [62, 72] | 109.4 ± 1.1 [108, 111] |
| carry on, no `archive_emit` | 55.5 ± 2.4 [53, 62] | 106.4 ± 1.0 [105, 109] |
| carry off, no `archive_emit` | 61.3 ± 2.5 [59, 70] | 107.9 ± 0.6 [107, 109] |

Neither variant passes the gate. What this shows:

- **Not the warm start.** Starting woken pairs cold changes nothing measurable (61.5 vs
  60.9).
- **Not the missing wake chain, as far as the archived adjacency reaches.** Restoring
  chained wakes and their resting rows through the archive makes it worse (65.7). The
  rows `archive_emit` adds hardly matter either way (removing them moves carry on by
  −0.8 and carry off by +0.4).
- So the carried λ = 0 rows must matter through something other than the rigid
  solve's warm start and wake. The remaining candidate is the threshold fracture: its
  contact ingest (stress-gpu `ingest.rs`) reads the step's contact rows, and resting pairs
  that keep rows (carry on) or lose them (carry off) may change what it treats as
  supported or loaded, even at λ = 0. **Ruled out by the code:** `ingest_rows`
  (stress-gpu `shaders/coupling.slang`) turns a row into chunk records only through its
  impulse `−(n·λN + t1·λT1 + t2·λT2)·h`, and returns before writing any record when that
  is zero. A carried row (λ = 0) gives the fracture no support, load or contact.

The mechanism is still open: on the rigid side the carried rows only (1) mask the
archive (a), (2) give same-step rows and wake chaining to bodies touched or woken in
the narrowphase (b covers the archived part of that), and (3) are inert otherwise; the
fracture ignores them. What (b) does not reproduce is chaining over resting pairs that
are in no archive (a pair whose partner fell asleep later is in only that partner's
archive) and the racy in-dispatch order of today's chain.

Until that is understood, option A cannot be made to pass by fixing the rigid side, and
option B (keep the carry's behaviour, make it cheap) is the safer route to the cost.

### Pre-existing test failures (checked 2026-11-01)

- `engine-webgpu` `heightfield::bodies_rest_on_uneven_terrain_without_sinking_in` passes
  on `origin/main` (3169cde2, 3 runs) and fails on this branch's base a56c45e5 (3 runs,
  deterministic: body 38 at 0.0678 m/s against the 0.05 m/s limit). Bisected to
  8ff688ab, "Sleep that threshold fracture cannot tell from staying awake" (the sleep
  archive). Its parent leaves body 38 at 0.0472 m/s, already just under the limit. The
  commit's new kernel paths are all gated (`sleepArchive()`, `wakeOnTouch()`) and the
  test enables neither, so its logic is unchanged for the test; the edits to hot kernels
  (`emitCandidate`, `wakeFromContact`, persistence) shift their shader codegen, and the
  marginal body ends a little faster. Not fixed yet.
- The three WGSL lib tests (`char_move`, `mpm_wheel_loads` codegen) fail on `origin/main`
  too: the local toolchain's Slang output, not this branch.

### Left for the next step

- The decision above.
- Fixed solver cost when few bodies are awake: per iteration, four colour dispatches and
  the sparse tail even with nothing to colour (about 0.6 ms natively, more in Dawn). The
  lab's single-dispatch sparse solve uses 8 lanes per body where the colour kernel picks
  lanes by degree, so switching to it changes the float summation order; a lossless
  version needs a sparse kernel with the colour kernel's lane choice.
- The moving-collider list is one cheap scan over every collider (one word of each
  collider record and its body's flags): about 0.1 ms at 732,160 colliders. A
  body-to-collider index (colliders are not sorted by body once threshold splits
  re-parent them) would make it proportional to the awake bodies.
- The resting table is rebuilt over every collider after any step with sleep events
  (war: often). Incremental insert of the new sleepers needs the stale R entries of
  re-sleeping bodies removed first.
- Per-body passes (predict, extract, the colour tiles and the CSR) still run over every
  body slot. `predict_bodies` has an awake-list mode, but kinematic bodies are not on the
  awake list.
- Outside the rigid plan: the threshold fracture's per-frame passes over every chunk and
  bond, and the pose readback sized to every body.

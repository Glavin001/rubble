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

## On top of #92's carry-off (2026-11-01, merge of #94 @ 47372a0b and 067504b2)

#92 (`f9bd3887` on `feat/large-maps`) turned the resting-pair carry off wherever the sleep
archive is on, and spread the coupling ingest scan over the GPU; `7e79e634` moved the
archive's warm start out of `persistContact`'s hot block (the heightfield regression
bisected above). This branch keeps both with the awake-indexed broadphase.
`EngineFeatures::carry_with_archive` (`ENGINE_REST=keep`, `THRESHOLD_CARRY=on` in the
ensemble) keeps the carry under the archive for A/B runs. The ensemble's default is now
the carry-off row: 60.9 ± 2.3 [57, 66] bonds, 107.7 ± 0.7 bodies (never asleep 57.3 ±
2.5); `awake_broadphase.rs` stays bit-identical and the heightfield test passes.

City, native, quiet GPU, wall ms per step (median of 120 waited steps). "#94" is this
branch with `ENGINE_BROADPHASE=full WGPU_VALIDATION_INDIRECT_CALL=1` (#94's head
plus only this branch's fast-list sizing); drive at K=256 and 1,024 is the median of
three runs each (the others were noisy: another session's CPU load showed in the host
column):

| K | scenario | #94 | this branch |
|---|---|---|---|
| 64 | drive | 1.77 | 1.42 |
| 64 | war | 4.76 | 4.44 |
| 256 | drive | 2.40 | 1.60 |
| 256 | war | 6.78 | 5.98 |
| 1,024 | drive | 4.19 | 2.52 |
| 1,024 | war | 10.72 | 7.28 |

Same breaks before and after at each K (K=64: 1,041 awake, 2,391 bonds; K=256 1,139 /
2,389; K=1,024 1,070 / 2,251).

Drive is now 1.4 → 2.5 ms from 64 to 1,024 houses. A per-dispatch profile
(`RUBBLE_GPU_TIMING_DISPATCH=1`, K=64 against K=1,024, drive) finds no rigid pass that
grows by more than about 0.1 ms: the largest are the awake insert and query and
`archive_emit` (+0.12, +0.06, +0.05 ms, within the marks' noise). What still grows sits
outside the rigid plan's marks: the frame wait (`poll_blocking`) for the threshold
fracture's per-frame passes over every chunk and bond (`fr_loads`, `fr_breaks`), and the
pose readback sized to every body. Those are the next targets (L2 §2, §4).

Web, same build (`?broadphase=full` for the left column): fracture lab grid, ball in
flight, GPU 2.09 → 1.95 ms, pipelined 3.37 → 2.96 ms (medians of three; the lab grid has
no sleep archive, so the carry still runs there). Busy pages, ms per step: ring110k
10.46, gables506k 29.88, softbody traffic (20 cars) 2.44, vehicle driving 1.09, vehicle
traffic driving 2.38, all at or below the table's earlier values.

## Carry off + archive as the default: the touch query (2026-11-02)

**Why carry-off broke more, found.** Deterministic traces of the house's nudge-0 shot
(`THRESHOLD_TRACE`, every body's GPU record each step; `THRESHOLD_TRACE_CONTACTS`, the
step's contact rows; `THRESHOLD_TRACE_ARCHIVE`, a body's archived rows) show that at the
impact step (96) the clusters the hit wakes had no resting contacts in carry-off: they
fell freely (vy −0.14 m/s, vz 0) where never-asleep and carry-on push them (body 14:
vz −2.09 / −1.97), and the next step they were slammed (6 extra breaks). Three causes:

1. `archiveEmitResting` reads only the touched sleeper's own archive, but `archiveOwner`
   files a pair under one body (the first that fell asleep with a resting partner), so a
   touched sleeper's pairs filed under its partners never formed.
2. Only sleepers a moving body's *candidate* touched were listed. In the house the ball's
   pairs wake a cluster in the narrowphase's sphere pass, and that cluster wakes the
   clusters it rests on in the hull pass of the same step: those were never listed, so
   they were solved without supports (carry-on has every resting pair already).
3. A pair at rest stored its fresh row (λ = 0) even with an archive behind it, and the
   warm start then found that row first when the pair woke.

**The touch query** (`EngineFeatures::archive_touch_query`, on; `ENGINE_ARCHIVE_WAKE=emit`
selects the earlier `archive_emit` path for A/B runs), world_step.slang:

- Every sleeper a moving body's candidate pair includes is listed (`archiveMark`, archive
  or not), and so is a sleeper a projectile hits after the broadphase (`apply_impulses`'
  late pass).
- `archive_touch_spread` lists the sleepers the listed ones overlap (one hop, from a
  snapshot of the list so the hop count does not depend on thread order).
- `archive_touch` queries each listed sleeper's colliders as if awake, against the
  colliders at rest only (a mover's pairs are its own query's), on hierarchical grids
  (resting table or full grid, own and coarser levels plus the finer scan), uniform grids
  (split or one table) and the small pairwise path. Each pair is emitted once: between
  two listed sleepers at one level the lower collider index emits it, across levels the
  finer one.
- Persistence warms each such pair from either body's archive (`archiveFind`), at rest
  too (`archiveRestRow`), and a row whose collider a split re-parented (key mismatch)
  still lends a same-feature match its λ and k, with its old-frame anchors dropped
  (`archiveStaleRow`), as the previous step's row does without the archive.
- Not with the carry (`rigid_plan::archive_touch`): the carried pairs already include
  a touched sleeper's.

Measured and not kept: more spread hops (`ENGINE_ARCHIVE_SPREAD=1|3`: 53.9 ± 1.7, the
same for both), marking every sleeper within reach of a fast CCD body's sweep (bit
identical on the 61 shots: the ball's own candidates list them), and searching the archive
of the body a split took a collider from (identical).

The nudge-0 trace now matches carry-on at the impact step (body 14 vz −1.97 against
carry-on's −1.97 and never's −2.09; bodies 3–19 within a few cm/s of carry-on).

**Gate**, 61 shots nudged −30..30 mm, 120 frames, bonds broken, mean ± sd [min, max]
(`sleep_policy_power`; paired: the same shot under carry-on):

| Policy | House | Paired vs carry-on | Masonry arch |
|---|---|---|---|
| never asleep (reference) | 57.1 ± 2.4 [53, 62] | −0.90 ± 2.79 | 127.5 ± 5.0 [116, 139] |
| carry on | 56.2 ± 1.7 [52, 60] | | 127.5 ± 5.0 [116, 139] |
| carry off, archive, `archive_emit` (before) | 60.6 ± 2.7 [56, 70] | | 127.5 ± 5.0 [116, 139] |
| carry off, archive, touch query (now) | 55.1 ± 2.1 [52, 65] | −1.08 ± 2.25 | 127.5 ± 5.0 [116, 139] |

The arch never sleeps before its hit, so all four agree bit for bit there. On the house
the touch query sits 1.1 bonds below carry-on (as carry-on sits 0.9 below never asleep),
inside the reference spread except one shot (nudge 14: 65). The pair sets of the woken
bodies match carry-on's at steps 96–98 on the most divergent shots (only pairs with
λ = 0 differ); what remains is the solve diverging after the hit.

City (`CITY_SLEEP=1 CITY_POOL=32768`, one run each; wall median ms, awake, bonds broken):

| K | scenario | carry (`ENGINE_REST=keep`) | archive_emit (before) | touch query (now) |
|---|---|---|---|---|
| 64 | war | 4.65, 1,034, 2,630 | 4.21, 1,041, 2,391 | 4.45, 1,090, 2,516 |
| 256 | war | 6.75, 1,055, 2,425 | 5.39, 1,139, 2,389 | 6.06, 1,016, 2,368 |
| 64 | drive | 1.85 | 1.36 | 1.39 |
| 256 | drive | 3.17 | 1.58 | 1.62 |

**Default.** `EngineFeatures::resting_contacts` (`RestingContacts::Archive`, `Carry`,
`Drop`; `ENGINE_REST=carry|drop`): `World::new` enables the archive in every world with
sleep on canonical or persistent contacts (compact contacts keep the carry), sized
`WorldDef::sleep_archive_rows` or the contact capacity. The archive now runs on
persistent contacts too (rows past the two halves of bindings 4 and 27,
`archive_count_persistent` / `archive_scatter_persistent`; WGSL only, so Metal and CUDA
codegen is unchanged). The wasm `features.restingContacts`, the demo's `?rest=` (archive
by default; the "Carry resting pairs" checkbox switches the carry live) and the C ABI's
`RbWorldDef::resting_contacts` / `sleep_archive_rows` follow; the Unreal plugin takes
`rbDefaultWorldDef`'s default. `tests/world.rs struck_sleeping_pile_moves_as_with_the_carry`
strikes a sleeping pile on canonical and persistent contacts with the archive and the
carry.

Demo recordings (headless Chrome, persistent contacts, the default `?rest=archive`;
`~/.cache/rubble-probe/{grid4,ring4,gables4}` against the earlier `{grid3,ring2,gables3}`),
awake / contact rows:

| Scene, step | carry | carry dropped | archive_emit | touch query (default) |
|---|---|---|---|---|
| legacyGrid100k, 460 (ball at 300) | 204 / 405,224 | 740 / 4,104 | 104 / 785 | 74 / 758 |
| legacyGrid100k, 900 | 278 / 405,650 | 13,812 / 73,217 | 1 / 1 | 1 / 1 |
| ring110k, 1,020 | 14,989 / 1,001,535 | 78,931 / 674,009 | 12,681 / 104,993 | 8,681 / 101,924 |
| ring110k, 1,500 | 465 / 1,007,698 | 73,587 / 657,002 | 0 / 0 | 0 / 0 |
| gables506k, 630 | 23,290 / 3,925,735 | 26,671 / 130,525 | 23,795 / 122,128 | 24,314 / 129,991 |
| gables506k, 1,200 | 9,579 / 3,900,238 | 31,700 / 187,487 | 8,566 / 51,178 | 9,962 / 64,201 |

No cascade on the grid, the ring settles, and gables tracks the carry's awake count with
2% of its contact rows.

### Cost (demo, `__profileNow()`: plan-phase GPU ms, median of 10 profiled steps)

Before: the carry (`?rest=carry`, the earlier default). After: the archive (default).

| Phase | gables506k, step 1,200, ~9k awake: carry | archive | ring110k, step 780, ~22k awake: carry | archive |
|---|---|---|---|---|
| broadphaseGrid | 1.44 | 1.42 | 0.65 | 0.72 |
| broadphase (archive passes) | 0.01 | 0.24 | 0.01 | 0.27 |
| ccdToi | 3.26 | 0.29 | 0.31 | 0.09 |
| narrowphaseMain | 5.25 | 0.51 | 1.43 | 0.70 |
| warm | 4.93 | 0.04 | 0.98 | 0.13 |
| graph | 1.53 | 0.29 | 0.27 | 0.17 |
| topology | 0.56 | 0.56 | 0.11 | 0.11 |
| primal | 0.63 | 0.60 | 1.60 | 1.43 |
| dual | 3.39 | 0.05 | 0.93 | 0.15 |
| sleepMain | 6.69 | 0.56 | 0.58 | 0.33 |
| pose | 0.17 | 0.17 | 0.02 | 0.03 |
| GPU span | 28.05 | 4.81 | 6.99 | 4.24 |

The touch query's own passes went 0.51 → 0.13 ms on gables (settled) when a touched
sleeper is listed only if the touch can wake it (the narrowphase's wake rule on the same
velocities: one it cannot wake stays asleep with its pairs at rest, which nothing reads that
step) and each pass runs a thread per listed sleeper; scanning the colliders instead of a
body's run (needed once threshold splits leave colliders non-contiguous) brought them back
to 0.24 ms. Both bit-identical on the gate.

Where gables506k's 6.7 ms goes with 26k bricks awake (step 420): broadphaseGrid 1.45,
narrowphase 1.07, primal 1.15 (8 iterations), sleepMain 0.68, topology 0.58, graph 0.57,
ccdToi 0.31, archive passes 0.30, warm 0.16, pose 0.16. About 3 ms of it does not scale with
the awake count: the uniform grid's per-step clear, insert and query over every collider
(1.4 ms; `build_aabbs` already skips sleepers), the per-body passes over every body slot
(topology, sleep, pose) and `ccd_wake` / `ccd_unhold` (a collider binary search per sleeping
body while a fast CCD body exists). The awake-indexed broadphase exists only for hierarchical
grids; `?gridlevels=on` puts gables on it, but there the resting table is rebuilt over every
collider after each step with sleep events (every step while a pile settles: 1.40 ms). Next:
a persistent resting table for the uniform split grid (its "other" table kept across steps,
movers in the awake table only), and incremental resting-table inserts for steps whose new
sleepers have not moved cell (their tombstone entries are valid again); both lossless.

## The struck grid: woken pairs start cold (2026-11-02)

**Report.** `demo.html?scene=legacyGrid100k&sleep=on`, F at step 400: with the carry the
ball knocks bottom boxes out and the column above falls (awake ~200 at step 560, rising to
~550); with the archive the ball barely dents the face and everything is asleep again ~100
steps later. The 61-shot house gate missed it.

**Gate.** `tests/grid_strike.rs` (`grid_strike_probe`, ignored): the "100k Grid" scene
(`engine_scenes::build_host`, the demo's snapshot) imported into a `World`, settled asleep
(sleep time 0.5 s, as the demo), the demo's ball (8 kg, r 0.35 m, 30 m/s from the camera
towards the look point) fired at step 400; awake count every 50 steps, boxes moved. Never
asleep is no reference here: the grid collapses by itself without sleep (98k boxes moved).
Native, ball without CCD (`STRIKE_NO_CCD=1`), two runs each (native runs are not
bit-reproducible), awake at step 550 / 700 / 1,000, boxes moved > 0.1 m:

| Policy | Run 1 | Run 2 |
|---|---|---|
| carry | 175 / 552 / ~800, 312 | 172 / 432 / ~750, 309 |
| archive, woken pairs warm (the default before) | 58 / 198 / ~400, 116 | 65 / 228 / ~420, 275 |
| archive, woken pairs cold (now) | 174 / 470 / ~500, 285 | 174 / 527 / ~700, 515 |

**Root cause: the warm start.** A woken box's pairs restarted from the archive with their
resting λ and stiffness: from its first step the box is gripped by its full resting load
(friction μ·λN against the sleeping boxes around it, which act as static walls), so it
barely moves, wakes few neighbours and the stack falls asleep again. The carry's rows are
re-detected at rest every step (λ = 0, stiffness at its floor: `World` runs without contact
reuse), so a box it wakes restarts cold, slides, and wakes the boxes it disturbs. Starting
woken pairs cold (`EngineFeatures::archive_cold`, now the default; `ENGINE_ARCHIVE=warm`
for the earlier behaviour) matches the carry. With the touch query a cold archive needs no
rows, so `archive_free` .. `archive_seal` no longer run (`rigid_plan::archive_rows`) and
`World` sizes the archive to one block.

**And a lost pair.** Comparing the house's contact rows with carry-on at the impact step
found pairs between two listed sleepers at different grid levels emitted by neither: the
finer one returned on a touched partner at a coarser level, and the coarser one's finer
scan skips touched colliders by design. The finer one now emits them (`touchEmit`). House
gate after the fix (61 shots): archive warm 54.5 ± 1.2 [52, 58], archive cold 54.3 ± 1.5
[51, 60], against carry-on 56.2 ± 1.7 [52, 60] and never asleep 57.1 ± 2.4 [53, 62]
(paired against carry-on −1.7 and −1.9; carry-on against never −0.9). Both ranges now sit
inside the references'. The carry itself emits some pairs twice: at the house's impact step
body 5's pairs with its neighbours appear as two identical manifolds (`(5, 8)`, colliders
646/649, three rows each, twice), so carry-on is not a clean reference either.

Browser check (same build, `?rest=archive` with cold wakes against `?rest=carry`, F at step
400, awake count at steps 580 / 700 / 760): archive 205 / 309 / 275, 225 / 232 / 228,
177 / 249 / 296; carry 201 / 264 / 377, 153 / 213 / 255, 214 / 268 / 207. The column above
the struck boxes falls in all six. One earlier archive run (of five) dented the face
without a collapse (awake back to 1 at step 620); every carry run so far collapsed.

Cost: no archived rows also drops the five row passes, so `sleepMain` on gables506k goes
0.56 → 0.39 ms (settled, ~9k awake) and 0.68 → 0.46 ms (~26k awake). GPU span 4.79 and
6.41 ms.

`tests/floor_probe.rs` (ignored) imports an engine-scenes scene, settles it asleep and keeps
`FLOOR_WAKE` bodies nudged awake, printing each dispatch's GPU time with
`RUBBLE_GPU_TIMING_DISPATCH=1`. "100k Grid" with 200 awake: 173 dispatches a step, of which
82 primal colour dispatches, 12 sparse and 12 dual (12 iterations, colours dispatched =
last step's + 4). This is the next step's starting point (the cost floor).

## Archive against carry, settled (2026-11-02, later)

**The carry reference doubled contact rows.** With the carry and the archive both on
(`EngineFeatures::carry_with_archive`, the threshold probe's `carry_on` and `ENGINE_REST=keep`),
`archive_emit` still emitted a touched sleeper's archived pairs although the carry already
re-emits every resting pair: the house's struck body had every pair with its neighbours
twice (`RUBBLE_SKIP_KERNELS=archive_emit` removed the copy). `archive_emit` now emits
nothing while the carry runs, and no archived rows are kept when nothing reads them
(`rigid_plan::archive_rows`). With that, the 61-shot house gate is **bit-identical**
between carry-on and the archive default: 54.3 ± 1.5 [51, 60] both, paired 0.00 on every
shot (never asleep 57.1 ± 2.4; the 1.9-bond gap to carry-on reported earlier was this
duplication). The arch is identical across all policies as before.

**legacyGrid100k, 10 browser runs each** (headless, live playback, F at step 400, awake at
step ~740 > 100 counted as a collapse): archive 6 of 10, carry 6 of 10. The "collapses" are
not what they look like, though (next paragraph): with `?reuse=pass` both policies dent the
face and fall asleep again.

**CCD bullets push boxes through the ground (pre-existing).** `grid_strike_probe` with the
ball a bullet (the demo's default): about 2,300 boxes end below the ground (lowest −360 m by
step 1,000), under carry and archive alike, and on `origin/main` 09647583 (carry) too. The
bottom boxes next to the ball's impact (bottom layer, front rows) sink below 0.1 m in the
impact step and their columns follow them down, endlessly (they never sleep). It needs all
of: persistent contacts, reuse-copy with the split narrowphase (`EngineFeatures::np_split`,
the defaults), and a CCD body. Each of these avoids it: `ENGINE_NP=fused` (lowest box
0.08 m), `ENGINE_REUSE=pass` (0.19 m), canonical contacts (0.19 m), the ball without CCD
(0.15 m); `ENGINE_PERSIST=scan` does not. In the browser the same configuration makes the
struck column "collapse" (boxes vanish under the floor and the awake count keeps rising);
`demo.html?...&reuse=pass` struck runs settle back to sleep under both policies. **Fixed (root cause below).**

Side-by-side recordings (headless Chrome, archive default left, `?rest=carry` right;
`~/.cache/rubble-probe/side/`): `ring-archive-vs-carry.mp4` (ring110k to step 1,500: awake
46,830 / 44,557 at step 316, 14,236 / 14,322 at 915, 0 / 0 at 1,500), `gables-archive-vs-carry.mp4`
(gables506k to 1,200: 26,272 / 26,560 at 495, 8,729 / 8,962 at 1,200; the carry steps at
27.9 ms GPU against 6.6), `grid-archive-vs-carry.mp4` (legacyGrid100k, F at 400; one run
each, both bimodal through the CCD fall-through above), `city-archive-vs-carry.mp4`
(city.html k=64 war, 600 steps: 188 / 301 awake, 2,995 / 2,784 bonds at the end; one run)
and `gauntlet-archive-vs-carry.mp4` (vehicle.html?scene=gauntlet, monster truck cruising
with the cannon firing). No visible difference beyond run-to-run spread. `city.html` and
`vehicle.html` take `?rest=carry` for this.


## CCD fall-through: root cause and fix (2026-11-02)

Traced with `grid_strike_probe`'s diagnostics (`STRIKE_ROWS`: a box's persistent contact rows
and its candidate pairs each step, `WgpuSim::last_persistent_rows` / `last_candidates`):

- A woken bottom box had pairs with the boxes above it but none with the ground, and fell.
- Its ground pair was lost earlier, in the step the CCD body first exists: that step every
  resting box-plane pair the carry re-emitted came out of `narrowphase_fresh` as a manifold
  and rows with colliders (0, 0) (2,482 manifolds, 9,928 rows), though the deferred tail held
  the right pair and `narrowphase_fresh` read it correctly (checked by writing what it read
  back into the tail). The next step the carry re-emitted (0, 0), which `emitCandidateCore`
  drops (i == j), so the box kept no ground pair while asleep, and had none when woken.
- Needed: the split narrowphase (the fused one emits the same pairs correctly), persistent
  contacts, and look-ahead on (a CCD body in the scene turns it on), i.e. the
  `closingPoints` call in `emitContactManifold`: skipping it removed the (0, 0) manifolds,
  and so does inlining it. The code is correct as written (the WGSL reads the pair once and
  passes the struct by pointer); the call, compiled natively (wgpu on Metal), clobbers the
  caller's pair. Fix: `[ForceInline]` on `closingPoints` (no other
  change). Rebuilding the kept manifold point by point instead of copying it did not help.

After the fix, `grid_strike_probe` with a CCD ball leaves no box below the ground (lowest
0.16-0.19 m; carry and archive, the probe's ball and the demo's parked projectile with the
demo's solver).

**The browser never had the fall-through.** Counting bodies below the ground after a
legacyGrid100k shot (`record-below.mjs`: `__world.readBodies` over every slot at step 900, F at
step 400): 0 in every run, on this branch (5 archive, 5 carry) and on `origin/main` 09647583
(5 runs). What the browser shows is a real collapse: the ball knocks bottom boxes out and the
column above falls into the hole, sometimes, in all builds. Runs whose awake count was still
above 100 at step ~880-900 (the column still falling): this branch after the fix, archive
4 of 10 and carry 2 of 10 (live playback, `g10`), then archive 3 of 5, carry 2 of 5 and
`origin/main` 5 of 5 (`g11`). So main with sleep on collapses too, at least as often; whether
a shot collapses the column depends on where and when the ball lands (live playback timing),
not on archive vs carry. The earlier `?reuse=pass` runs that settled (2) were within that
spread. Native fall-through: present on `origin/main` and before this fix, gone after it.

## Cost floor (2026-11-02, `claude/cost-floor`)

The per-step work that did not shrink with the awake count (about 3 ms of gables506k's
4.8 ms with ~9k awake): the uniform grid's clear, insert and query over every collider,
per-body passes over every body slot, `ccd_wake`'s collider search per sleeping body,
and the primal's dispatch per colour. Every change below gives the same results as the
pass it replaces, and each has a switch back.

### Done

- **Uniform-grid rest tables** (`EngineFeatures::awake_uniform`, on; `ENGINE_BROADPHASE=hier`
  or `?broadphase=hier` keep the full pass on uniform grids; world_step.slang
  `restUniform`). The split grid's other table (sleepers and small statics) is kept across
  steps as the resting table R (the hierarchical grid's `restHeads` / `restLinks`); each
  step only the moving colliders (`list_moving_colliders`) are bounded, inserted into W
  (`awakeHeads`, links in the AABB spare word, as the full pass's awake table) and query
  (`broadphase_query_awake_uniform`: `broadphase_query`'s awake branch with the other
  table read from R). A collider in W this step is a tombstone in R: its stamp
  (`restStampBase`, this step's index) is tested by every R walk (the awake query, the
  oversize query, `broadphase_fast`, `sceneWalk`, the touch query), so a body woken after
  the inserts (a projectile's late impulse) is still found in R by the walks that run
  later, as the full pass's other table held it. Planes, oversized statics and resting
  large dynamics are listed in R's header and published into the scratch lists the full
  pass fills (overflows flagged every step, as it flags them). The sleepers' pairs with
  the large-dynamic list (the only pairs a sleeper's own thread emits in the full pass)
  come from `broadphase_sleeper_large_walk`: a lane per R cell that can hold an entry
  overlapping a large dynamic (bounded by the largest half extent of R's entries, header
  word 29), each entry at its true cell only, plus the listed large ones; a thread per
  collider (`broadphase_sleeper_large`) only if a large box spans more than 32,768 such
  cells. Not with kinematic bodies (they sit in the other table while they move) or at
  or below the pairwise path's 512 colliders. Taken only while at most 10% of the
  bodies are awake, until more than 15% are (`uniform_rest_wanted`): the pass costs a
  scan over every collider plus the awake ones' work, and at ring110k's 22% awake it was
  slower than the full pass (0.8 → 1.0 ms). `set_uniform_rest_always` for tests.
- **Incremental rest inserts** (`EngineFeatures::rest_incremental`, on;
  `ENGINE_REST_TABLE=rebuild`), both grids, so also `?gridlevels=on`: instead of
  rebuilding R after every step with sleep events, the colliders that came to rest this
  step (slot 81 still lists them) join R at the end of it (`rest_sleepers_mark`,
  `rest_unlink`, `rest_push`). A collider already in R at its bucket (a tombstone of a
  body that fell asleep again in the same cell) is valid again as it is; one in another
  bucket is spliced out of that chain first, one thread per claimed bucket (its W head,
  empty after the step's queries, is the claim). A rebuild only when a hierarchical
  collider moves between a cell and the oversized list (header word 31). Each collider's
  R bucket is kept (`restAtBase`); `rest_table_words` grows by three words a collider.
- **`ccd_wake` by collider** (`EngineFeatures::ccd_by_collider`, on; `ENGINE_CCD_WAKE=body`):
  a body lies within reach × r + own of a sweep iff one of its colliders c lies within
  reach × r + extent(c) (own is the largest extent; rounding is monotone), so
  `ccd_wake_colliders` tests a thread per collider, with each CCD slot's r taken once
  (`ccd_slot_reach`, or `ccd_slot_reach_listed` over the listed movers on awake-indexed
  steps: a registered CCD body was moving when it registered). Work slot 87 is empty
  unless some CCD body moves more than four contact offsets a step (`ccdUnholds`), when
  `ccd_wake` could wake nothing; the earlier pass searched the colliders of every
  sleeping body whenever a CCD body existed. It also drops `ccd_wake`'s assumption that
  colliders are sorted by body (a binary search, which threshold splits break). Bodies
  without colliders are never woken by it, so a host whose mirror has a dynamic body
  without colliders keeps `ccd_wake`. `ccd_unhold` stays per body: it searches colliders
  only for the few bodies whose wake is deferred, and a per-collider version cost 0.1 ms
  more on gables while the ball flies (so the sorted assumption remains there, for those
  bodies only).
- **Awake-list body passes** (`EngineFeatures::awake_body_passes`, on; `ENGINE_BODY_PASSES=all`):
  `predict_bodies` and `extract_velocity_sleep` in their existing PASS_U = 1 mode, a
  thread per awake-list entry (slot 67), where nothing changes a body between
  `build_awake_list` and them: not with vehicles, characters, soft bodies or directed
  debris, and prediction not with kinematic bodies.
- **Velocity-only commands** no longer bound every collider: only commands that may move
  a pose or a collider that does not move after (teleport, reparent, mass frame,
  kinematic target) bound every collider, and they now also rebuild R, which they did
  not on hierarchical grids (a teleported static stayed in its old cell)
  (`commands_move_poses`).
- **The primal tail** (`EngineFeatures::primal_tail`, opt-in: `ENGINE_PRIMAL=tail`):
  the headroom colours (from last step's colour count to the sparse tail's first) solved
  in the sparse tail's single workgroup with the colour dispatches' lanes, accumulation
  and reduction (`solve_primal_sparse_tail`), then the sparse colours as before: one
  dispatch an iteration instead of one per headroom colour. Bit-identical bodies, but
  not adopted: in Chrome the empty headroom dispatches it saves cost next to nothing
  (legacyGrid100k, 12 iterations, 48 fewer dispatches a step: primal within noise of
  the per-colour dispatches), and a step whose colour count grew solves the new colours
  in one workgroup. Natively, on "100k Grid" with 200 awake (`floor_probe`, per-dispatch
  marks), it takes the step from 145 to 110 dispatches (colour dispatches 79 → 34, plus
  12 tails); the native totals were within the noise of the background load.
- Demo switches for A/B timing in one build: `?broadphase=full|hier`, and `?costfloor=off`
  (or a list of the parts to turn off: `uniform`, `incremental`, `ccd`, `body`, `tail`).

### Correctness

- `tests/awake_broadphase.rs`: a new uniform-grid scene (a plane, small and wide
  statics, 576 cubes in piles that settle and sleep, two 2.5 m spheres in the
  large-dynamic list, one resting against a pile and one rolling into another, thrown
  cubes, a CCD bullet, a moving kinematic late): the same candidate pairs and
  bit-identical bounds as the full pass at 78 of 86 compared steps (the kinematic turns
  the awake pass off after), with incremental inserts and with rebuilds
  (`ENGINE_REST_TABLE=rebuild`). The hierarchical test passes with incremental inserts.
- `tests/ccd_wake_colliders.rs` (new): a brick wall and compound sleepers struck by fast
  CCD balls, every step of the strikes run from the same saved state (`debug_step_until`)
  with each variant and with the reference twice: `ccd_wake` by collider gives the same
  body flags and timers (220 comparisons, 286 wakes); the awake-list passes and the
  primal tail give bit-identical bodies before the sleep gate and the same poses and
  sleep timers after it (the in-place gate's revocations travel as far as scheduling
  lets them, so which bodies fall asleep can differ between two runs of the same passes;
  one reference run in 220 differed from itself).
- The 61-shot gate (house and masonry arch, `sleep_policy_power`, carry off + archive):
  every shot and per-step break timeline bit-identical to the base (a28112a4).
- `tests/grid_strike.rs`, two runs each (native runs are not bit-reproducible); boxes
  moved > 0.1 m, awake at step 1,000, without CCD: base 628, 618 / 788, 735; this branch
  458, 449 / 634, 641 (one build earlier 367, 551 / 584, 812), the spread of the cold
  archive runs above. With the ball's CCD on (the probe's default) base and this branch
  alike dropped 2,300-2,700 boxes through the ground (lowest y ≈ −360 m at step 1,000):
  the native `closingPoints` miscompile ("CCD fall-through" above), fixed on
  `claude/awake-only-passes` and merged here (see "Merged, then the floor again" below).
- Suites: engine-world (all but the known `cuboid_house_breaks_and_splits_as_in_the_lab`,
  the same failure as the base), engine-webgpu broadphase_oracle, awake_broadphase,
  gables_hold, ring_hold, columns_hold, static_pairs, pipelined_gate, heightfield,
  ccd_wake_colliders, engine-ffi, engine-metal `vehicle_parity`: pass.

### Cost (demo, `__profileNow()`: plan-phase GPU ms, median of 10 profiled steps)

Headless Chrome, `?sleep=on`, persistent contacts, the default `?rest=archive`. "Base" is
a28112a4's build, "off" this build with `?costfloor=off`, "now" this build; each cell the
median of two interleaved rounds. The machine was not quiet: an orphaned headless Chrome
from this session's own profiling kept running ring110k throughout (it could not be
stopped from here), and another session's builds and tests ran in between, so whole
rounds moved by 20-40% together (gables420's GPU span read 6.2 ms in one round and 7.5 ms
in another, for the same build). Within a round the phases this work touches moved
consistently; the others (narrowphase, primal) are noise between columns.

| Phase | gables506k step 420 (~26k awake): base | off | now | step 1,200 (~9k awake): base | off | now |
|---|---|---|---|---|---|---|
| broadphaseGrid | 1.58 | 1.56 | 0.99 | 1.90 | 1.72 | 1.19 |
| broadphase (archive passes) | 0.34 | 0.34 | 0.46 | 0.42 | 0.45 | 0.42 |
| ccdToi | 0.34 | 0.34 | 0.37 | 0.41 | 0.44 | 0.33 |
| topology | 0.62 | 0.64 | 0.52 | 0.69 | 0.65 | 0.51 |
| sleepMain + sleep | 0.50 | 0.52 | 0.47 | 0.54 | 0.50 | 0.42 |
| primal | 1.32 | 1.62 | 1.73 | 1.17 | 1.29 | 1.23 |
| GPU span | 7.29 | 8.94 | 7.49 | 7.29 | 6.97 | 5.79 |

| Phase | ring110k step 780 (~26k of 110k awake): base | off | now | legacyGrid100k step 560, F at 400 (~150 awake): base | off | now |
|---|---|---|---|---|---|---|
| broadphaseGrid | 1.44 | 1.45 | 1.49 | 0.47 | 0.44 | 0.78 |
| broadphase | 0.50 | 0.49 | 0.59 | 0.29 | 0.15 | 0.05 |
| ccdToi | 0.18 | 0.18 | 0.23 | 0.11 | 0.11 | 0.08 |
| topology | 0.18 | 0.20 | 0.21 | 0.15 | 0.14 | 0.14 |
| sleepMain + sleep | 0.43 | 0.40 | 0.41 | 0.19 | 0.18 | 0.21 |
| primal | 3.40 | 3.61 | 3.81 | 3.12 | 2.48 | 2.98 |
| GPU span | 8.75 | 8.84 | 10.00 | 4.98 | 4.30 | 5.64 |

The earlier, quieter rounds (an earlier build: the walk in one workgroup, the per-collider
`ccd_unhold`, the primal tail on, no awake-fraction switch) agree on gables: broadphaseGrid
1.79 → 0.93 ms (step 420) and 1.51 → 0.93 ms (step 1,200), sleepMain 0.55 → 0.35 and
0.40 → 0.26, topology 0.64 → 0.56 and 0.59 → 0.54, GPU span 7.58 → 6.77 and 5.33 → 4.42
(off → now). The gables floor that this work set out to cut (broadphaseGrid 1.4-1.6,
sleepMain 0.4-0.5, topology 0.6, ccdToi 0.3) is down by about 0.6-0.8 ms of GPU time a
step: the uniform broadphase's part roughly halves, the sleep pass loses about a third
(the extraction over the awake list), the topology pass about a sixth (prediction over the
awake list). On ring110k (22% awake) the uniform rest pass is off by the awake-fraction
switch, so it runs the full pass as before; the differences there are noise. On
legacyGrid100k with ~150 awake the uniform rest pass did not measure faster in the
browser (broadphaseGrid + broadphase 0.76 base, 0.59 off, 0.83 now, with the same
spread between rounds); the per-step work it saves on 100k colliders (the clear, the
inserts and the queries over every collider, about 0.2 ms) is of the size of the
noise here, and natively (`floor_probe`, "100k Grid", 200 awake) the GPU totals of the
two were within each other's run-to-run spread.

`?gridlevels=on` (gables506k on the hierarchical grid, step 420): incremental rest inserts
against the rebuild after every step with sleep events (`?costfloor=incremental`),
broadphaseGrid 1.93 → 0.96 ms, GPU span 9.89 → 7.87 ms.

### What is left in the floor

On gables506k the uniform broadphase's fixed part is now the scan over every collider
that lists the movers (`list_moving_colliders`, one collider word and one body word each)
and `ccd_register`'s scan over every body; the rest is the awake colliders' own work.
Other passes still run over every body slot, and are not proportional to the awake count
without changing what they produce or the order they produce it in:

- **`list_moving_colliders`**: without an index from bodies to their colliders it has to
  read every collider's body. Threshold splits re-parent colliders on the GPU
  (`fr_reparent`) with no host write, so such an index would have to be kept on the GPU.
- **`build_awake_list`, the CSR (`csr_clear`, the prefix passes) and the colour tiles
  (`count_color_tiles`, `prefix_color_tiles_parallel`, `scatter_color_tiles`)**: indexed by
  body. The colour lists keep each colour's bodies in body-index order, which the solve's
  order (and so its float sums) follows; built from the awake list (atomic order) they
  would need a sort to come out the same.
- **`pose_pack`**: must see every body whose pose or sleep state changed, including ones
  changed outside the awake list (bodies `ccd_respond_apply` wakes at the end of the step,
  host writes, a threshold split's re-framed bodies between steps).
- **The primal's colour dispatches**: one per colour of last step's count per iteration;
  their cost in Chrome is their work, not the empty headroom dispatches (see the tail
  above). Moving colours to the sparse kernel changes the lanes per body.
- **The contact hash clear**: the record hash is sized at least four slots a body, so its
  clear scales with the bodies; a smaller table changes probe sequences and its overflow
  behaviour, so it is not shown lossless.
- **The sleep island gate**: eight rounds over the manifolds (contact-proportional).
- **`build_aabbs` in moved-only mode** still finds a body's colliders by binary search.
  Only engine-fracture's planner enables that mode (HOST_S(19); World and Metal leave it
  0), and only for scenes whose colliders it found sorted at load (`bounds_sortable`);
  a planner whose splits re-parent colliders after load would have to clear that flag.
  The kernel is shared with Metal, so it was left.
- **The uniform rest pass at high awake fractions**: below 10% awake it saves the clear,
  insert and query over every collider; above, its per-collider scan plus the awake
  colliders' R walks (a link and a stamp load an entry, where the full pass's other table
  keeps its link in the AABB) cost more than the full pass, so the host switches.

## Memory: what PR #97 added, and what fits on a phone (2026-11-03)

Measured in headless Chrome (`mem.mjs`: `GPUDevice.createBuffer` / `destroy` wrapped before the
page loads, the wasm memories, `performance.memory`), `demo.html?scene=<s>&sleep=on` after ~200
steps; MB, peak:

| Scene | Commit | GPU | largest buffer | wasm | JS heap | total |
|---|---|---|---|---|---|---|
| gables506k | main 09647583 | 2,390 | 768 | 825 | 259 | 3,475 |
| gables506k | e0319a91 (#97 before the archive) | 2,414 | 768 | 825 | 259 | 3,499 |
| gables506k | cb061d16 (archive) | 3,587 | 768 | 826 | 259 | 4,672 |
| gables506k | a84adf59 (archive + cost floor) | 3,600 | 768 | 827 | 218 | 4,645 |
| gables506k | this fix | 2,446 | 768 | 767 | 259 | 3,473 |
| gables | main | 151 | 32 | 24 | 7 | 182 |
| gables | a84adf59 | 204 | 32 | 25 | 8 | 238 |
| gables | this fix | 155 | 32 | 25 | 8 | 189 |
| ring110k | main | 840 | 256 | 195 | 62 | 1,099 |
| ring110k | a84adf59 | 1,241 | 256 | 205 | 59 | 1,506 |
| ring110k | this fix | 856 | 256 | 205 | 59 | 1,121 |

The growth was the sleep archive's: `WgpuSim::enable_sleep_archive` re-created the contact
buffers (bindings 4 and 27, persistent contacts) a few rows larger and dropped the old ones
without `destroy()`, so the browser held both until it collected them (b27 768 -> 1,536 MB,
b4 384 -> 768 MB on gables506k). With cold wakes and the touch query (the default) nothing
reads archived rows, so the contact buffers are no longer grown at all, and any buffer the
archive does replace is destroyed once the copy is submitted. What remains over main is the
awake-indexed broadphase's tables (binding 14, +36 MB on gables506k) and the archive's
per-body scratch (binding 21, +15 MB). House and arch gates bit-identical.

**gables506k does not fit a phone on main either.** iOS gives a tab about 1.5 GB; the scene
needs ~3.5 GB on main (2.4 GB of GPU buffers, its largest 768 MB, plus 0.8 GB of wasm heap),
because its capacity (6M contacts, 4M candidates, sized for the all-awake start) is fixed at
creation. A mobile budget would mean smaller capacities that fail loudly when exceeded, or a
smaller scene: a product decision, not done here.

## Equivalence: PR #97 against origin/main (2026-11-03)

`engine/crates/engine-world/tests/equivalence.rs` (ignored; `EQUIV_OUT` records one scene and
sleep mode per run, `EQUIV_COMPARE=dirA,dirB` compares two recordings, `EQUIV_ENV=1` lets a
recording read the `ENGINE_*` switches; threshold scenes then need `ENGINE_CONTACTS=canonical`).
It uses only API that main has, so the same file runs on both builds (main needs the
`engine-scenes` dev-dependency). Reference: origin/main 0ab6a4dc, run twice (A, B) for the
noise floor. Candidate: this branch with threshold destructibles on main's archive (below).
Native wgpu/Metal, each build's default, sleep on and off. Per-dump metric: every body slot's
position, linear velocity and awake flag (parked slots excluded), plus contacts, awake bodies
and bonds broken per step (per shot for house and arch).

Scenes: house and arch (the 61 lab shots, 120 frames each), grid (the demo's 100k Grid, a
CCD ball at step 400, 900 steps), ring (Brick Ring 28k with its rolling ball, plus a 4 t ball
at step 200, 400 steps), gables (12 brick gables, two balls at step 150, 600 steps), city (16
threshold houses, the war scenario, 240 steps), vehicle (engine-vehicle's rigid sedan at full
throttle into a 160-brick wall, 600 steps).

**Threshold destructibles keep main's archive.** Main runs two resting-contact designs: the
carry in non-threshold worlds, and in threshold worlds (canonical contacts) a sleep archive the
first threshold destructible enables, whose woken pairs warm-start from their archived rows
through `archive_emit`. The PR had moved threshold worlds to its persistent worlds' archive
(the touch query, cold wakes): the house broke 54.3 bonds a shot against main's 60.6 (warm wakes
alone gave 54.5, so the touch query is most of it) and the city 6.5% more bonds. Canonical
worlds with `RestingContacts::Archive` (the default) now do what main does;
`RestingContacts::Touch` (`ENGINE_REST=touch`, `RB_RESTING_TOUCH`, `rest=touch` on the demo and
city pages) opts in to the touch query there.

| scene | floor (main A vs B) | main vs PR | verdict |
|---|---|---|---|
| house, off / on | identical | identical: 57.1 / 60.6 bonds a shot, 61/61 shots | identical |
| arch, off / on | identical | identical: 127.5 bonds a shot, 61/61 | identical |
| city, sleep off | identical | parts at step 78 (one candidate pair, one contact fewer); same 2,214 bonds; mean 0.1 mm, max 0.157 m | differs: the awake broadphase (below) |
| city, sleep on | identical | 2,030 -> 2,127 bonds, 1,395 awake-state differences | differs: the awake broadphase and `ccd_by_collider` (below) |
| gables, sleep off | identical | identical | identical |
| gables, sleep on | mean 0.349 m, 1,806 awake diffs | mean 0.378 m, 2,242 | within noise |
| grid, sleep off | identical | main: 100,000 bodies below the ground; PR: 0. Main with only `[ForceInline]` on `closingPoints` is bit-identical to the PR | differs: main's CCD fall-through (fixed) |
| grid, sleep on | mean 0.063 m, 2,021 | main: 2,353 below the ground; PR: 0 | differs: main's CCD fall-through (fixed) |
| ring, sleep off | identical | identical | identical |
| ring, sleep on | mean 0.207 m, 8,720 | mean 0.242 m, 11,981 | within noise |
| vehicle, off / on | identical | identical (positions, velocities, sleep) | identical |

Hard checks, every scene and mode: no NaNs, body counts equal, no body below the ground on the
PR. The ring's one or two bodies below y = -1 (both builds) are bricks knocked past the edge of
the scene's 200 m ground slab (z 111-125 m), not fall-throughs. Final states at rest: identical
wherever the table says identical.

**The city's two causes.** A bisect of city sleep-off over main..62a93fbc (harness at every
commit) finds 99a4c6cc (rigid_plan: awake-indexed broadphase on hierarchical grids) first;
its parent 165e32e8 is bit-identical to main. With `ENGINE_BROADPHASE=full` the PR's city
sleep-off run is bit-identical to main; `ENGINE_BROADPHASE=hier` is not. The bodies are
bit-identical through step 77; at step 78 the awake broadphase emits 4,102 candidate pairs
against the full pass's 4,103, and the missing pair is one manifold and one contact (9,416
against 9,417). At step 74 it emitted one more than the full pass (no contact). So it does not
form "the same pairs" as documented: it is a different pair set at the margins, not a reorder
or rounding, and one of its misses was a real contact. Making it identical is a fix to the
awake path's pair test (lossless, the full pass is the reference), not a summation-order change.
With sleep on, `ENGINE_BROADPHASE=full` alone still leaves 2,104 bonds; adding
`ENGINE_CCD_WAKE=body` (main's per-body `ccd_wake`) brings the city to main's 2,030 bonds, 0
awake-state differences, max 0.1 mm. The per-collider pass wakes 38 bodies on the first shell
step where main's wakes 32, and that is main's bug, as the CCD fall-through was. The city's
colliders are not in body order (its houses take pooled body slots: body 87's one collider is
collider 587), and main's per-body pass finds a body's colliders by `firstColliderOfBody` (a
binary search) and a contiguous run. From the state before that step (`tests/ccd_wake_city.rs`,
ignored), both rules on the CPU give exactly the GPU's two answers: 16 sleepers woken by main's
rule, 22 by the per-collider rule, the extra six bodies 87-92, each a house piece whose single
hull collider (extent 3.95 m) lies outside main's run (`firstColliderOfBody` returns 827 for all
six, a run of none, so main counts their extent as zero). Each is 4.30 m from its shell's sweep
against the rule's 1.25 x 0.6 + 3.95 = 4.70 m: inside the wake reach, with the hull's bounding
sphere 0.25 m into the shell's swept sphere. No split is needed for this: the slots are
out of order from placement. `tests/ccd_wake_unsorted.rs` pins it: a sleeper whose collider
follows a later body's, 2 m off a bullet's sweep, wakes in that step; with main's per-body
rule it stays asleep and the test fails.

**Memory** (headless Chrome, `mem.mjs`, MB peak; `demo.html?scene=<s>&sleep=on` 200 steps,
`city.html` 1,024 houses 60 s):

| page | main 0ab6a4dc GPU / wasm | 62a93fbc (touch, cold) | this branch |
|---|---|---|---|
| gables506k | 2,391 / 826 | 2,447 / 795 | 2,447 / 795 |
| ring110k | 841 / 203 | 856 / 197 | 856 / 197 |
| city | 2,653 / 1,408 | 2,045 / 1,328 | 2,561 / 1,411 |

The city's archive keeps main's rows again (b4 640 MB, as on main), and the page is 92 MB under
main. gables506k and ring110k are unchanged by this: +56 MB and +15 MB of GPU buffers over main
(the awake broadphase's tables, binding 14, and the archive's per-body scratch, binding 21),
offset by less wasm heap (-31 MB, -6 MB).

Notes:

- **Contact counts with sleep on are lower on the PR** where it archives persistent contacts
  (gables 38k -> 11.6k, grid 404k -> 6, vehicle 1,440 -> 105): sleeping pairs are archived
  rather than re-emitted, so only awake contacts are counted.
- **city-on's per-slot position difference (max 2,337 m)** is not a motion difference: once
  bonds break differently, split pieces take different body slots.
- **Main's sleep-on runs of gables, grid and ring are not reproducible** on native wgpu (the
  floor column); house, arch, city and vehicle are, and so is every sleep-off run.
- **The ring overflows the record warm hash** (`OVERFLOW_WARM_HASH`, 64 probes) from about step
  440 of the collapse, on main and on the PR alike, so the ring run stops at 400. An engine
  limit that fails loudly; more contact capacity does not help (the hash is sized from the
  previous contact count).
- **grid_strike**, sleep on, persistent contacts: no box below 0.1 m from the strike to step
  1,000, lowest box centre 0.17 m. With `STRIKE_CANONICAL=1` (now the carry, as on main): no box
  below the ground, lowest centre 0.06 m, one box below 0.1 m at steps 850 and 1,000.

## The city's pair set, and PR #97's memory (2026-11-03, later)

**The pair.** City, sleep off, the war scenario: from the first shell steps the awake pass and
the full pass disagree before the broadphase even runs. Up to 9,790 colliders' bounds differ
(steps 20-23 and 38-41), and pairs go missing, for example (10486, 10562): a static 0.28 m
chunk of body 9913 against a 2 m piece of body 9921 that moves; (12810, 13055) is the same
case. The harness's step-78 contact is one of these. **Cause:** the threshold fracture changes
colliders and bodies on the GPU between steps with no host write (fr_commit, fr_reparent,
retire, re-frame), while the awake path rebuilds bounds only for this step's listed movers and
keeps its resting table until the host marks it stale. A re-framed static chunk keeps its old
bounds and its old cell. **Fix:** the first GPU fracture turns the awake pass off for its
world (threshold.rs), so threshold worlds take the full pass, as on main. City sleep off is
bit-identical to main again. Sleep on differs only by `ccd_by_collider` (2,104 bonds against
main's 2,030; a0b23796 is on that). Making the awake pass follow the fracture would need a GPU
signal from the fracture (its move and frame counters) that rebuilds bounds and the table; not
done.

**The test.** `engine-world/tests/awake_broadphase_scenes.rs` compares the awake pass with the
full pass before each compared step, on the same GPU state, as exact sets of candidate pairs
plus bit-identical collider bounds:
- the demo's 100k Grid with its CCD ball, under the demo's planning (136 steps) and at every
  awake fraction (150);
- Brick Ring 28k at every step (400);
- the city, which must take the full pass.

The two runs must start from the same binding 21. The sleep archive's touch query lists
touched sleepers there per step, and a second run spread one hop further from the first run's
list. Without that restore the comparison reported pairs (two just-asleep ring bricks, a grid
brick and the ground) that only the second run found. These were artifacts of running twice,
not misses.

**Memory.** Headless Chrome, `mem.mjs`, `demo.html?scene=<s>&sleep=on`, ~200 steps, MB of GPU
buffers at peak, against main 0ab6a4dc:

| page | main | PR before | this | what is over main |
|---|---|---|---|---|
| ring110k | 881.6 | 898.1 (+16.5) | 871.5 (-10.1) | below main: b14 -12.0, b21 +1.6, b22 +0.4 |
| gables506k | 2,506.7 | 2,565.7 (+59.0) | 2,532.3 (+25.6) | b14 +18.9 (the awake pass's tables, in use), b21 +6.3 (the touch query's records), b22 +0.4 (pass table) |

(Peak equals live in both: the grown binding 14 is created after the old one is freed. A first
build that created it before freeing the old one peaked 21 MB higher on gables.)

Three changes, each lossless:
- **The pre-CSR region holds what its tables can reach.** `constraint_id_cap` keeps the power
  of two every table is sized within, so the grid heads, the warm hash and the ids are
  unchanged. `constraint_id_words` allocates only the most of them, and the rounding none of
  them reaches is gone. 2 × (2^21 contacts + 64 joints) used to take 2^23 words (ring110k: 16.8
  MB). Gables keeps its region: its record warm hash can reach all of it.
- **The awake pass's tables are allocated on use.** `WgpuSim::ensure_rest_tables` grows
  binding 14 the first time a step takes the awake pass, sized for the colliders present
  (rounded up to a power of two, at most the capacity). It fails the step if the device cannot
  hold it. Nothing else in binding 14 outlives a step, so the old buffer is freed first and
  nothing is copied. Worlds that never take the awake pass spend nothing on it: fracture worlds,
  and uniform grids that never drop below their awake fraction. The awake bitmap moves ahead of
  the tables, because the awake-list body passes read it without them.
- **An archive that keeps no rows is laid out without them** (`archiveCompact`, host word 30
  bit 9). Cold wakes with the touch query are the default for persistent contacts, and they
  never read rows, blocks or per-body row records. That leaves 3 words a body instead of 7. The
  reset snapshot's copy of binding 21 shrinks with it (snapshot-b21 5.7 MB -> 0), and snapshots
  skip the constant pass table.

Gables506k stays above main because its uniform grid falls below 10% awake within the 200
steps, so the awake pass runs and its tables are live: 2^20 heads twice (8.4 MB), and five
words a collider for 2^19 colliders (10.5 MB). The touch query keeps three words a body. Going
below main there would mean dropping work these paths do, or aliasing W into the pre-CSR region
with a full clear every step.

Gates on e9c6ae76..611507e9:
- harness against mainA: house and arch identical (61/61, off and on); city off identical, on
  as above; gables, ring and vehicle off identical, on within main's noise (gables mean 0.365 m,
  ring 0.243 m); grid only main's fall-through;
- grid_strike x2: no box below 0 (lowest 0.12 and 0.16 m);
- engine-world (all but the known `cuboid_house_breaks_and_splits_as_in_the_lab`), the
  engine-webgpu integration suites, engine-ffi and engine-metal `vehicle_parity` pass.

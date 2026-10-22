# P1: the full-state recorder, its completeness gate, sizes and the viability dataset plan (2026-10-22)

## The contract (recorder revision 9)

The state contract is mechanical. It is every field that `stress_ref::world::World`
serializes (serde derives on the world, its solver and every state type), shredded into one
typed column per field path:

- `solver.bonds[][].joint.kappa` is a list element's struct field;
- `sticks{k}` / `sticks{v}` are the persistent contact table, keyed by pair and sorted by key;
- every occurrence's kind is kept, so options and enums round-trip.

The columns are written at every 1/60 s boundary to `full_state/<boundary>.fs.zst`.
Every 10th boundary is self-contained; the others are zstd deltas of the previous one. The
code is `stress_learned::full_state`, and FRAMES_SCHEMA.md documents the layout.

What is left out:

- the static data: the scene, and the structures rebuilt from it;
- the engine coupling (none in the standalone oracle);
- one pure cache, `solver.step_bounds`: the stable-step bound keyed by cluster contents.
  The routing prediction fills it between boundaries.

**The completeness gate passes.** A World rebuilt from boundary f (plus `scene.json`) and
stepped one oracle frame equals boundary f + 1 bit for bit, in every column, with f64
compared by bits. It was checked:

- through the impact on the M5 free wall (20 frames) and the M5 wall row (20 frames);
- on an anchored wall under gravity with several impactors (30 frames);
- on a free wall with several impactors (30 frames);
- on a settling anchored wall (40 frames);
- on every boundary of a recorded event (`tests/outcome.rs`), and through a nine-delta
  chain (`tests/outcome_calls.rs`).

## Field classification (P2's request)

`learned_impact_full_state_classify` takes each of the 508 paths at five mid-impact or
later boundaries (the two M5 walls, anchored t2 at contact and 12 frames later, free t2 at
contact). It zeroes that field (0, false, empty), rebuilds the world, steps one frame and
compares the result with the untouched step:

- **state (244 paths):** another field's next value changes, or the rebuild or step fails;
- **derived (156 paths):** nothing changes;
- **diagnostic (108 paths):** only the field's own next value changes.

The table is `runs/dev/full-state-classify/classification.{json,md}`. Two caveats:

- **"Derived" means "had no effect in one frame on these five trials".** It includes the
  config switches of features these scenes don't exercise (rebar, implicit, multirate,
  blast).
- **"Diagnostic" can hide state that only matters later.** Examples are running maxima
  such as `joint.kappa_c` and `joint.ductility`. Resetting them changed nothing within the
  frame, but they would change a later compression failure.

So the recorded contract stays the whole serialization. The classification is a guide for
choosing model inputs and outputs, not a licence to drop fields. A multi-frame
classification (reset, then step 10 frames) would separate delayed state from diagnostics;
that is a follow-up.

Fields that are state for certain:

- chunks: u, θ, v, w, active, cluster;
- clusters: pose, velocity, activity and timers;
- bonds: `joint.kappa`, `damage`, `crush`, `plastic`, `mode`, `alive` and the multirate
  state;
- contacts: `sticks`, `pair_offsets`, `ground_offsets`;
- impactors: pose, velocity, angular velocity;
- time, `next_cluster_id`, `pending_split`, `unchanged`, `true_steps`, `mass_damping`.

Not state, as the classification found: per-bond `force`, `elastic` and `measures`
(recomputed every substep), the energy ledgers, `events`, `neighbours` (rebuilt),
`initial_positions` and `step_start`.

## The retention spec: what each item is stored as

| Spec item | Stored | Notes |
|---|---|---|
| Every frame, full state | `full_state/` + the f32 `state_*` arrays | The arrays are a convenience copy the call builder reads (about 30% of the arrays' size) |
| Per break: time, order, mode | `break_i32` (frame, bond, order, mode) and `break_f64` (time) | |
| Peaks > 0.2 or broken: stress components and time of peak | `peak_i32` / `peak_f64` | Generalised force (shear t1/t2, axial, bending t1/t2, torsion) at the substep end with the largest failure index seen |
| Energy split per frame | `frame_energy` (14 cumulative terms) | |
| Energy split per island | fracture and damping from `frame_bond_dissipated` | **Gap:** contact and friction work per island needs per-pair work instrumentation in stress-ref, which doesn't exist yet |
| Impactor impulse, peak force, 8 sub-frame states | `frame_impactor_impulse`, `_peak_force`, `_subframe` [F,I,8,14] | |
| Support reaction per anchored structure | `frame_structure_support_impulse` [F,S,6] | Momentum change of the structure's chunks less its external-load impulse (`frame_structure_load_impulse`, from a stress-ref accumulator: off by default, no physics change). About 0 for a free wall (tested) |
| Mid-frame chunk position and velocity (contact, secondary and separation frames) | `mid_frame`, `mid_time`, `mid_chunk_pos`, `mid_chunk_vel` (f64) | |
| Cost per frame | `frame_cost_i32` (substeps, contact-limited substeps), `frame_cost_f64` (dt range), `frame_oracle_seconds` | Wall-clock, not CPU seconds; not reproducible, so `regenerate` skips it |
| Provenance | `context.json` `provenance` | Source commit (`git describe --dirty` at build), binary FNV-64, revisions; seeds and perturbations in the recipe |
| Blast face loads | – | No blast events yet |

## Sizes (pilot `runs/size-r9/`: 8 training events per family, bulk-like pilot ranges)

MB per event, compressed:

| family | frames | non-settling frames | chunks | arrays + JSON | + full state | full state on non-settling frames + every 10th only | wall s at 8 parallel (overhead) |
|---|---:|---:|---:|---:|---:|---:|---:|
| free_wall t1 | 78 | 9 | 131 | 2.02 | **6.49** | 3.31 | 22 (5) |
| anchored_wall t1 | 100 | 77 | 140 | 4.22 | **15.1** | 13.7 | 58 (10) |
| free_wall t2 | 54 | 12 | 132 | 1.66 | **4.65** | 2.86 | 25 (4) |
| anchored_wall t2 | 105 | 99 | 131 | 2.62 | **6.17** | 6.02 | 73 (9) |
| ballistic bracket (102 events) | | | | 1.8 | **5.1** | | |

- **The full state roughly triples an event.** A frame is about 100 KB compressed for a
  130-chunk wall, from about 0.5 MB raw.
- **This is close to the lossless floor.** XOR against the previous frame plus byte-shuffle
  saves only 3% more: the dynamic f64 content (about 45 values per bond) is near
  incompressible.
- **Anchored walls don't settle quickly.** Ground contact keeps them in "secondary" and
  "contact" phases, so thinning full state to active frames saves little there.

## Ballistic-limit sweeps

There are six families in the `p1-ballistic` catalog:

- training: `bl_free_thin`, `bl_anchored_base`, `bl_standing`, `bl_coarse_heavy`;
- validation: `bl_free_thick`;
- test: `bl_anchored_sides`.

A bracket pilot (`runs/bl-bracket/`, 0.02–8× each family's punch threshold) found the
penetration transition far below the punch model's threshold in every family. The oracle's
walls give way at 0.1–0.7× the punch speed.

| family | punch threshold | transition (rebound/arrest → perforate) |
|---|---:|---|
| bl_free_thin | 12.5 m/s | 1.9–2.7 m/s |
| bl_anchored_base | 14.0 | 1.5–2.1 (an arrest at 3.0: noisy) |
| bl_standing | 12.3 | 1.9–2.6 |
| bl_coarse_heavy | 12.5 | 6.3–8.9 |
| bl_free_thick (val) | 20.0 | 3.0–6.0 |
| bl_anchored_sides (test) | 21.3 | 7.5–10.6 |

Planned sweep: 12 log-spaced speeds per family over 0.5–2× the bracket's midpoint, each an
8-member ensemble. That is 6 × 96 = 576 events, about 2.9 GB. Command per family:
`--catalog p1-ballistic --family F --speeds-abs LO:HI:12`.

## The viability dataset plan, and the budget conflict

Disk now: `runs/` holds **32 GB**, above the 30 GB budget.

- **Calls: 15.8 GB.** `calls-v4-eval` 8.4, `calls-v4-train` 3.3, and `calls-v3-eval` 4.1,
  which P2 still holds for its /3 gate.
- **Recorded data: 16 GB.** This includes the 2.3 GB revision-8 pilot and the 0.6 GB
  bracket.

The full plan, all revision 9 with full state, standalone CPU oracle:

| set | events | MB/event | GB |
|---|---:|---:|---:|
| train singles, distinct configurations: free t1 1,000; free t2 500; anchored t1 700; anchored t2 500; standing t1 600; standing t2 300 | 3,600 | about 7.5 | 27 |
| ballistic sweeps (6 families × 12 speeds × 8) | 576 | 5 | 2.9 |
| standing-wall val and test ensembles (10 + 10 configurations × 8) | 160 | 10 | 1.6 |
| **total new** | **4,336** | | **about 32** |

That takes about 6–8 h at 8 threads.

Options:

- **(A) Raise the budget to about 70 GB (recommended).** Disk has 474 GB free. The full
  plan fits, and the calls can be rebuilt freely.
- **(B) Keep 30 GB; full state on evaluation, ensembles and ballistic events only.** After
  evicting the calls (13.8 GB free), training singles are arrays only, at about 2.6 MB:
  - 3,500 training singles: 9.1 GB;
  - ballistic sweeps and standing evaluation: 4.5 GB.

  Training events stay exactly regenerable from their recipes, and a regenerated event
  carries its full state.
- **(C) Keep 30 GB with full state everywhere.** After evicting the calls: ballistic and
  evaluation as planned, plus about 1,200 training singles.

## Not done yet

- **An offline re-derivation tool.** It would rebuild the World from a recorded boundary and
  recompute routing, contacts, pairs, QS prior and baselines under a given routing config.
  The builder's tiered routing needs it, and the full state makes it possible without
  re-simulation.
- **GPU-parity derived inputs:** deferred by planning.
- **Irregular-chunk and materials pilots:** pilot-only, after viability.
- **Regenerating the 828 thinned bulk events, and the evalplus queue:** both paused. The
  queue sits at test anchored t2 56/120, and the PAUSE file is in place.

## Viability dataset v1 (overnight, planning's plan of 2026-10-22; decided without sign-off)

**What is recorded.**
- **Oracle:** the standalone CPU oracle (a12_realtime), recorder revision 9: every frame
  kept, full state at every boundary, all the direct labels above.
- **Gravity:** on everywhere. The families are `anchored_wall` and the new
  `standing_wall` (no supports, resting on the ground), tiers 1 and 2, with pilot
  ranges, matching the measured sizes.
- **Ballistic sweeps:** the six ballistic families are all grounded.
  `bl_free_thin` and `bl_free_thick` became `bl_ground_thin` and `bl_ground_thick`,
  re-bracketed (`runs/viability-pilot/`).

**Pilot.** `runs/viability-pilot/`, 8 events per standing tier:
- standing walls stand still before the impact (no breaks; kinetic energy at most 1e-7 J);
- standing t1 averages 11 MB per event (median 5.5 MB; one slow settle took 46 MB and
  237 frames); standing t2 averages 8.9 MB (median 5.7 MB).

**Splits.** By layout group, hashed with one shared `--split-seed 777` so a layout has
one split whatever job records it. `--fractions 0.98,0.01,0.01`:
- training configurations are single runs (`--ensemble-train-share 0`);
- validation and test configurations are 8-member ensembles;
- the ballistic families' splits are fixed by family (4 train, 1 val, 1 test), every
  speed an 8-member ensemble.

The jobs run in this order, so a first chunk is ready early. MB per event are means from
the pilots:

| order | jobs | configurations | events (about) | MB/event | GB |
|---|---|---:|---:|---:|---:|
| A | anchored t1 / t2, standing t1 / t2, 80 configurations each (seeds 21001-21004) | 320 | 310 train + about 80 val/test | 8 | 3.1 |
| ballistic | 6 families × 10 speeds × 8 members (`--speeds-abs` 0.5–2× each bracket's midpoint) | 60 | 480 | 5 | 2.4 |
| B | anchored t1 120, anchored t2 120, standing t1 170, standing t2 90 (seeds 21101-21104) | 500 | 490 train + about 80 val/test | 8 | 4.6 |
| total | | 880 | about 1,440 | | **about 10** |

The ballistic speed ranges, in m/s (10 speeds each):

| family | split | range |
|---|---|---|
| bl_ground_thin | train | 1.25–5 |
| bl_anchored_base | train | 0.9–3.6 |
| bl_standing | train | 1.1–4.5 |
| bl_coarse_heavy | train | 3.75–15 |
| bl_ground_thick | val | 2–11 (wider: its bracket is noisy) |
| bl_anchored_sides | test | 4.5–18 |

**The budget, and why this size.** The user's 30 GB covers all data under
`learned-impact/runs`, with derived call caches first to evict. Tonight 15.8 GB of caches
are in use: `calls-v3-eval` (P2's overfit series and P0's harness A) and
`calls-v4-eval` / `calls-v4-train` (P2's overnight viability training). Evicting them now
would break those runs.

The most principled reading: recorded ground truth plus the `/5` call cache must fit in
30 GB on their own. The in-use caches are evicted as soon as their readers release them:
- recorded data: 16.6 GB before tonight;
- the `/5` cache: bounded at about 3 GB;
- that leaves about 10 GB for new events.

The generator's guard reads its budget from `runs/viability-v1.budget`:
- **45.8 GB tonight:** 30 GB plus the 15.8 GB of caches in use;
- **lowered to 30 GB** when the old caches are deleted.

So the excess on disk overnight is only caches that are already scheduled for eviction.
Nothing recorded is ever deleted.

**Calls.** `/5` (`learned-impact-outcome/5`) is the `/4` arrays plus the island's complete
oracle state at the frame's start and end:
- every serialized field of its chunks, of every bond inside the island (live or broken),
  of its clusters and of its impactors, flattened by field path into named f64 columns;
- the persistent contact slots (`pair_offsets`) as a keyed table;
- static columns once per event (`event_arrays`).

About 3.8 MB per anchored t2 event (86 calls). The `/5` calls of chunk A go to
`runs/calls-v5/` as soon as it is recorded; the rest are built as recorded, within the
bounded cache.

### Overnight log (2026-10-22)

**The budget.** Planning confirmed the user's limit is 30 GB for the whole `runs` tree,
derived caches included. Changes:
- **The queue was restarted** (`runs/viability-v1b.sh`, resuming; finished events
  skipped) with every job under a whole-tree guard of 29 GB. It had started at 45.8 GB.
- **Deleted, each when its readers released it:**
  - `size-r8` (findings committed in 03194672);
  - the sealed and test splits of `calls-v4-eval` (P2's 32 test members were rebuilt for
    its overfit, then released);
  - `calls-v3-eval`;
  - then all of `calls-v4-*`, once P2 moved viability to `/5`.
- **Recorded frames:** none touched. The tree went from 33 to 17.1 GB.

**Extra validation for chunk A.** Chunk A's own hash split gave 2 val configurations.
`--only-split val` keeps the recipes the same split hash (seed 777) sends to val, so no
layout leaves its split. It adds 5 configurations × 8 members for each of anchored t1,
anchored t2, standing t1 and standing t2 (`viability-v1-*-val`, 160 events).
`calls-v5/.complete` is touched once chunk A's calls and these are built: P2's overnight
chain waits on it.

**Ballistic ensembles: a bug, and the fix within the budget.** The queue's
`--ensemble-train-share 0` overrode the sweep's every-speed ensembles. The four
training-split ballistic families were therefore recorded as single runs (10 speeds each,
40 events). The validation (`bl_ground_thick`) and test (`bl_anchored_sides`) families
have full 8-member ensembles at every speed (80 events each), and the acceptance's
ballistic-limit curve is scored on those.

Full ensembles for the four training families (320 events, about 2.2 GB with calls) don't
fit beside chunk B in the budget. So `viability-v1d.sh` records them at 5 speeds across
each family's measured transition, 8 members each (160 events, about 1.1 GB with calls):

| family | speeds (m/s) |
|---|---|
| bl_ground_thin | 1.8–3.5 |
| bl_anchored_base | 1.3–3.2 |
| bl_standing | 1.6–3.0 |
| bl_coarse_heavy | 5.5–10 |

It runs after chunk B, under a 28.3 GB guard. Chunk B (`viability-v1c.sh`) records under
a 27.3 GB guard so its own `/5` calls fit. The whole tree stays under 29 GB.

**End state at the budget (2026-10-22, about 08:40 ADT).**

The guard had undercounted. The generator summed file lengths (26.2 GB) while the disk
held 28.45 GB (`du`, allocated blocks), because a recording's many small files take about
8% more than their lengths. The guard now counts allocated blocks (183ae3d3). Chunk B was
paused cleanly at that point, and its `/5` calls were built in batches, validation and test
first, while the tree stayed under 28.95 GB. The tree is now 28.96 GB, measured as decimal
GB on allocated blocks. (`du -h` shows binary GiB, about 27.0.)

Recorded (982 viability events, every frame, full state):
- chunk A: 314 train + 26 val and 2 test ensembles, including the extra validation;
- ballistic: 4 training families as single runs (10 speeds each), plus the val
  (`bl_ground_thick`) and test (`bl_anchored_sides`) families as 8-member ensembles at
  10 speeds;
- chunk B: anchored t1 (134 events) and anchored t2 (110 events).

`/5` calls in `calls-v5` (5.3 GB, 850 events): 420 train; 310 val in 39 ensembles; 120
test in 15 ensembles. 132 chunk B training events are recorded without calls, which are
rebuildable when space allows.

Not recorded (the budget):
- chunk B's standing-wall jobs, and the rest of anchored t2 B;
- the training ballistic families' ensembles (`viability-v1d.sh` is ready to run).

**Where more space could come from, losslessly.**
- The calls repeat each frame's state: call f's `fs_*_out` is call f+1's `fs_*_in` for
  the same island. Referencing the previous call instead would save about a third of
  `calls-v5`, about 1.7 GB.
- If the user's 30 GB means `du -h` units (GiB), the budget is 32.2 GB.

**The calls dedup frees no disk (measured).** In 85% of consecutive calls, a call's
frame-start tables are byte-identical to the previous call's frame-end tables (279 of 328
in a standing t1 event). The writer now stores identical arrays once, at a shared offset
(64fbf30f, round-trip test). That cuts the raw arrays by 29% (38 → 27 MB). But zstd level
19 had already found those repeats: the files are the same size (10.73 vs 10.75 MB), and a
long-distance window (window_log 27) gains nothing. `calls-v5` was therefore not rebuilt.
The remaining levers are planning's call:
- `/5` calls with only P2's frozen field set (derived; the recorded full state keeps
  everything);
- on-demand calls for training events;
- a larger budget.

## Dataset audit, viability v1 (for the morning)

**The audit** (`findings/p1-viability-v1-audit-2026-10-22.{md,json}`, `outcome_audit`):
- 982 events, 0 failures, 0 integrity problems;
- 0 events still breaking in their last 0.5 s; 7 (0.7%) truncated at the 10 s cap;
- splits: train 552, val 310, test 120; every frame kept (99,674);
- ensembles: 54 groups (432 members), all complete. Chunk B's stop had left
  `p1_anchored_wall_t2_00069_s21102` (val) at 6 of 8 members; e0 and e1 were recorded from
  the job's recipes, and the ensemble's `/5` calls rebuilt.

**Completeness, on recorded events** (`learned_impact_full_state_check`,
`runs/dev/full-state-check-v1.jsonl`). On 32 events, two drawn at random from each of the
16 jobs, every boundary was rebuilt from its full state and stepped one oracle frame: all
3,374 boundaries equal the next recorded boundary bit for bit.

**Disk.** The tree is at 28.99 GB (block-counted). The test cache from the `CallStream`
throughput check (0.13 GB) was removed.

**Decision (planning, accepted).** Training calls stay resident until tonight's viability
runs finish. Then P2 switches to building them once per run from the cache and drops its
clone; P1 deletes the training single-run calls (2.25 GB) and records the training
ballistic ensembles in that space.

## Reclaiming the superseded bulk-v1 recordings (user decision, 2026-10-22)

**Why.** The bulk-v1 recordings are revision 7: no full state, the pre-contract recorder.
The user approved reclaiming them for full-state training events. The evaluation sets
(evalplus, sealed, bl-bracket) are kept whole, as references.

**Step (a), done.** `learned_impact.outcome_reclaim --splits train --apply` removed the
frames of every bulk-v1 training event:
- events: 1,987 (`p1-bulk-v1-free-t1` 1,371, `anchored-t1` 404, `free-t2` 212), thinned
  single runs and training ensemble members alike;
- removed per event: `arrays/`, `catalog.json` and the duplicate copies of the scene
  (`scene.json`, `replay.json`);
- kept per event: `context.json`, `outcome.json`, `frames.json` and the job summaries; each
  job's `recipes.json` is now `recipes.json.zst`, compressed losslessly and verified;
- freed: **4.34 GB**; the tree went from 26.99 to 22.66 GB.

Each reclaimed event has a `reclaimed.json` with what went and its regenerate command.
Regeneration uses the current recorder, so it comes back with full state:

    zstd -dc p1-bulk-v1-X/recipes.json.zst > /tmp/r.json
    learned_impact_outcome_record --recipes /tmp/r.json --event EVENT_ID --out DIR

Nobody read these frames:
- P0's harness A reads only val and test ensembles;
- P2's trainer reads viability only, and its /4 overfit events are copied into its own runs.

**Step (b), pending.** The bulk-v1 val and test ensembles: 2,344 members, about 4.5 GB.
P0 extracted compact references for harness A (its worktree,
`runs/refs/p1-bulk-v1-*`, 869 MB, commit aac22d01) and verified them: identical reports,
and the recorded fingerprints re-simulate. The frames go once P2 confirms its n10 DAgger
jobs point at those references.

**The guard counts the whole footprint.** `viability-v1f.sh` reads its guard from
`runs/viability-guard.gb`: 29 GB less the learned-impact data outside this tree.
- Now 24.1 GB: 29 less P0's references (0.87 GB) less P2's call cache cap (4 GB, in P2's
  worktree).
- Lowered further if more data lands outside this tree.

The queue records:
1. the training ballistic ensembles;
2. chunk B's remainder;
3. new distinct full-state training configurations (chunk C): anchored t1/t2 and
   standing t1/t2, 60 configurations each per round, rounds interleaved, so a budget stop
   leaves the families balanced.

## Viability v1: end state (2026-10-22, about 12:50 ADT)

**Recorded:** 1,252 events, all recorder revision 9 (every frame, full state). Splits are
by layout group (split seed 777). Counts are events (configurations or ensembles):

| family | train | val | test |
|---|---:|---:|---:|
| anchored_wall t1 | 197 (197) | 64 (8) | – |
| anchored_wall t2 | 196 (196) | 40 (5) | 24 (3) |
| standing_wall t1 | 126 (126) | 77 (10) | 14 (3) |
| standing_wall t2 | 89 (89) | 56 (7) | 9 (2) |
| bl_ground_thin | 50 (10 single + 5×8) | – | – |
| bl_anchored_base | 50 (10 + 5×8) | – | – |
| bl_standing | 50 (10 + 5×8) | – | – |
| bl_coarse_heavy | 50 (10 + 5×8) | – | – |
| bl_ground_thick | – | 80 (10×8) | – |
| bl_anchored_sides | – | – | 80 (10×8) |
| **total** | **808** | **317** | **127** |

- **Incomplete ensembles:** 74 of 78 are complete. The budget stopped chunk B in the
  middle of four standing-wall ensembles: `p1_standing_wall_t1_00134_s21103` and
  `_00041_s21103` (5 of 8), `p1_standing_wall_t1_00046_s21103` and
  `p1_standing_wall_t2_00021_s21104` (1 of 8). Filter on `ensemble_size`. Completing them
  needs 20 events.
- **Chunk B, stopped by the budget:** standing t1 at 59 of 205 events, standing t2 at
  13 of 118.
- **Chunk C:** recorded nothing. Each round job stopped on the budget at once.

**Calls.** `calls-v5` is resident for val/test/ensembles (604 events, 4.08 GB, rev 8, with
`y_bond_break_time`). The 648 training single runs are built on demand into P2's
`runs/calls-cache` (in this tree, capped at 1.5 GB).

**Disk.** This tree is 26.78 GB block-counted (the cache's 1.24 GB included). P0's
references outside it add 0.91 GB, so the learned-impact total is about 27.7 GB, at most
28.0 GB with the cache full.

**Queue bug, recorded.** After the guard was reached, each chunk C job still wrote its
plan (`recipes.json`, about 20 MB) before stopping, adding 0.53 GB. The empty plan
directories (no recorded events) were removed. A future queue should check the guard
before starting a job.

**Still waiting on the user.** Deleting the remaining bulk-v1 val/test frames (2,344
members, about 4.5 GB) was blocked by this session's permission checker. P0's verified
references replace them for harness A (aac22d01), and P2 confirms nothing reads them. If
approved, the space would go first to completing the four ensembles, then to chunk B's
standing walls and chunk C.

## Contact energy in the calls (builder revision 9, planning 2026-10-22)

**What.** Each recorded call's `energy` now carries `E_contact_start` and `E_contact_end`:
the island's penalty contact energy (the oracle's `k δ²/2` per point, net of pair offsets)
at the frame's start and end. It is computed from the recorded full state, with no
re-simulation:
- restore the World at the boundary;
- take `World::island_contact_energy_of` over the call's chunks and impactors.

The code is P0's `island_contacts` (08bb117f), ported. The probe hunks in
`apply_contacts` are inert unless `probe_start` is called, so the goldens are unchanged.
P0's `handed_ground_overlaps` hunk is not ported: this branch's World has no handed overlaps.

**Verified.** In `recorded_contact_energy_equals_the_live_worlds`, at every call boundary
the recorded values equal the live stepping World's, bit for bit, nonzero cases included.
P0 cross-checked with its own implementation: harness A with the exact oracle, probing
the routed island's start World and merged end World. All 1,284 of 1,284 values over
642 calls in 6 events agree bit for bit, nearly all of them nonzero. The events cover
anchored t1/t2 (box impactors included), standing t1/t2, and the val and test ballistic
families.

**Rebuilt.** All 604 resident events in `calls-v5` (40,713 calls; 35,817 with nonzero
contact energy), 24 min at 4 threads. Restoring two boundaries per call roughly doubles the
build time. Training calls get the fields on demand through P2's cache.

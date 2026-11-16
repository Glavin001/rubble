# Cars in contact: a contact group across workgroups (C2), design and proof (2026-11-14)

The design for stepping touching cars at the physical cost, chosen by measurement, written
before the implementation was finished (branch `perf/contact-groups`). Owner's constraints:
the same results as today's group path bit for bit (and so the CPU reference == GPU
contract), car-to-car contact never on a partner's state a substep old, no CPU in the
runtime path, scaling to hundreds of cars, most possibly in one pile-up.

Machine: M3 Max (40 GPU cores), native wgpu 30 on Metal, Chrome (headless, WebGPU on the
same GPU). Car: the generated sedan as `fleet.sh` runs it (`sedan_new`, JBeam mode, 804
nodes, 6,427 beams, 528 collision triangles).

## 1. Today: where a contact group's frame goes

`fleet.sh town` (`explicit_vehicles_bench`, 3 s, GPU ms of a 1/60 s World step from
submit to completion) with `EV_GROUPS` (the groups every 60 steps), then
`EV_AB_AT=150 EV_AB_MASKS=…` (one snapshot at World step 150, then 20 frames of 34 steps
of the explicit frame alone under each ablation mask of the `EB_ABLATE` build, time
only). The machine was shared with other agents' GPU jobs: utilisation sampled before
each run read 82–100 %, and the same count's median moved by up to 1.8× between runs
(32 cars: 72.6 and 128.0 ms). The ablations compare masks from one snapshot in one run,
so their differences are the reliable part.

**Town, GPU ms a World step (median), the old path** (this run; the idle-GPU figures of
`fleet-benchmark.md` in brackets): 1 car 17.9 (12.2), 2 cars 17.2, 4 cars 31.2 (30.0),
8 cars 58.1 (53.6), 16 cars 63.5 (58.8), 32 cars 72.6–128.0 (103.3). The World alone
is 2.2–4.5 ms of each.

**Groups** (`EV_GROUPS`, steps 60 / 120 / 180): 2 cars none; 4 cars one group of 2 from
between steps 60 and 120; 8 cars [2] then [2, 2]; 16 cars [2 × 4] then [2 × 5]; 32 cars
[2 × 10] throughout. Every group in the town is two cars meeting head on in a lane.

**The explicit frame alone at step 150, ms, by what is ablated:**

| Cars (groups) | Frame | − contact pass (512) | − its triangle tests (131072) | − its levels (524288) | − its candidates (262144) | − members' self-collision tests (1048576) | − both (1572864) | − coupling passes (57344) | − surface pass (2^27) |
|---|---|---|---|---|---|---|---|---|---|
| 4 ([2]) | 40.0 | 39.3 | 40.0 | 33.5 | 29.5 | 16.3 | 12.4 | 26.7 | 17.8 |
| 8 ([2, 2]) | 54.9 | 29.7 | 29.6 | 41.1 | 31.4 | 37.4 | 24.7 | 54.4 | 42.7 |
| 16 ([2 × 5]) | 72.8 | 37.0 | 38.3 | 55.2 | 40.6 | 51.0 | 36.0 | 71.1 | 57.3 |
| 32 ([2 × 10]) | 101.2 | 49.4 | 45.4 | 74.2 | 60.5 | 55.3 | 43.3 | 92.5 | 81.1 |

(2 cars: the ablation run failed to load its module, a build was writing the output
directory at that moment; the town forms no group at 2 cars, so it is one car's frame.)
A frame waits for its slowest workgroup, so these are the costs of the worst group, and
shares overlap (a step waits on its slowest phase).

**By cause, a group of two head-on (8–32 cars):**
- **The contact pass's triangle tests: 25–56 ms a frame** (0.7–1.6 ms a substep), the
  largest single cause and the one that grows with the crash: the owners' triangles
  walk the candidates' grid and test the pairs on the 512 lanes of the group's one
  workgroup.
- **The members' self-collision tests: 17–46 ms.** A member's step in a group walks its
  grid and tests inline every step (no reused pair list), and both members' walks share
  one workgroup's lanes.
- **The levels (applying the hits in the specification's order): 14–27 ms.**
- **The surface pass: 12–20 ms**, the members' passes one after the other on all lanes.
- The coupling: 0.5–9 ms. The rest (beams, nodes, drive): the frame less all of the above.

**The 4-car anomaly (30 ms with this car, 12.5 ms with the earlier one).** With the
re-exported sedan two of the four cars meet head on and form a group from about 1.5 s
(step 60–120); with the earlier car the four struck houses and formed none
(`fleet-benchmark.md`). The group's frame is the cost: at step 150 its members'
self-collision tests are 24 of its 40 ms, its surface pass (two members serially) 22 ms
and its coupling 13 ms (both cars against the house's debris); its contact pass is 1 ms
(they barely touch yet). So it is not the contact pass but **two cars' worth of solo work
on one workgroup without the solo frame's reused pair list**, which a group of two forces.


## 2. Dispatch and barrier costs (measured)

`examples/contact_dispatch_cost.rs` (native) and `tools/contact-groups/dispatch.mjs`
(headless Chrome) run one WGSL (`tools/contact-groups/dispatch_cost.wgsl`): a 512-lane
kernel bound as the explicit frame is (a uniform and ten storage buffers), every
dispatch reading and writing what the last wrote, with a 512-word or a vehicle-sized
(7,936-word) workgroup array. A dispatch's cost is (pass of 341 − pass of 1) / 340, a
sync's (k = 4,000 − k = 0) / 4,000; medians of 9, pass timestamps and wall time agree.

| | Native (wgpu 30, Metal) | Chrome (headless, WebGPU) |
|---|---|---|
| Dependent dispatch, 512 lanes, 1 / 40 / 128 workgroups, small workgroup array | 1.9 / 1.3 / 2.4 µs | 1.4 / 1.6 / 2.2 µs |
| The same, vehicle-sized workgroup array (7,936 words), zeroed by the translator | **803 / 677 / 1,773 µs** (naga's MSL zeroes it from one lane) | 1.9 / 2.1 / 5.0 µs (Tint zeroes it across the lanes) |
| The same, not zeroed (`zero_initialize_workgroup_memory: false`) | 1.4 / 1.6 / 2.2 µs | (not available in WebGPU) |
| One `sync()` (storage + workgroup barrier), 512 lanes | 0.19–0.25 µs | 0.18–0.19 µs |

So:
- a dependent dispatch costs about **2 µs** in both, ten times a bare barrier (0.2 µs);
- natively, a kernel with the vehicle kernels' workgroup memory pays **0.7–1.8 ms a
  dispatch** for naga's zeroing (the audit's 0.16 ms was a smaller array) unless the
  pipeline turns it off; the current one-dispatch frame pays it once a frame
  (a finding for the solo frame too: §7);
- Chrome's zeroing is parallel and costs a few µs.


## 3. What exactness requires

The vehicle-contact specification (`docs/explicit-beams-vehicle-contact-spec.md` §3.1,
§8.3) orders a world step as: every vehicle's own step (no vehicle reads another's data);
the bounding boxes; then the vehicle-contact pass, **owners strictly in vehicle order**,
each owner's hits on one contact point in triangle list order, each hit reading the
point's force with every earlier contribution added. Positions and velocities do not
change during the pass; only accumulators do.

Today's group path (`contactGroup`) runs exactly that per 2 kHz substep inside one
workgroup: the members' steps side by side (`contactMembersStep`, lanes partitioned, or
one after another past four members), a barrier, the contact pass (`vvPassSmall`: the
owners' scans, candidates and tests concurrently, then the hits applied in levels, an
owner after every earlier owner whose touched vehicles meet its own; `vvPass` past four
members: owner after owner), a barrier, the next substep.

So per substep there are exactly two points where every member must have finished
something before any member continues:

1. the contact pass needs every member's step of this substep (positions, velocities,
   box keys, forces after beams and self-collision);
2. the next substep's member step needs the contact pass's accumulator adds.

Across workgroups WebGPU has no barrier (no forward-progress guarantee between
workgroups), so each of those two points is a dispatch boundary. **Two dependent
dispatches per substep is the minimum for an exact schedule across workgroups**; any
design with fewer must read a partner's state a substep old (the lossy two-phase
substep, not allowed).

The owner order inside the pass is a second constraint: an owner's hits read forces the
earlier owners added. Owners whose touched vehicle sets are disjoint commute
(fixed-point sums), so the pass needs as many ordered rounds as the longest chain of
meeting owners with hits (`vvLevel`). Two cars: at most 2. A column of n cars nose to
tail: up to n. That depth is the specification's (it is RoR's sequential vehicle loop);
the design cannot remove it without changing results, only make each round cheap.

## 4. The candidates

The cost model per 2 kHz substep of a frame with groups is its critical path: the
slowest vehicle's step, then the slowest group's pass, plus a dispatch boundary
(≈2 µs, §2) or a barrier (≈0.2 µs) at each point where everything must have finished.

**(a) Two dispatches per substep** (every vehicle's step, then every group's pass):
68 dependent dispatches a frame, ≈0.14 ms of boundaries natively and in Chrome (no
zeroing natively: 68 × 0.7–1.8 ms with it, §2). Each member gets its own workgroup, so a
group of k no longer puts k members' self-collision, surface pass and coupling on one
core; the pass still runs in one workgroup a group. Expected from §1 at 8 cars (a frame
of 55 ms in the ablation build): the members' self-collision share halves (−8 ms),
the serial surface pass halves (−6 ms), the pass is unchanged (25 ms of tests, 14 ms
of levels). Necessary (it is the minimum for exactness, §3) but not sufficient.

**(b) Phase-split dispatch across all cars** (a dispatch per phase per substep, every
car in each, as the wide passes): every one of a step's 9–16 member barriers and the
pass's 15–25 becomes a dispatch boundary, 10× a barrier: ≈30 × 2 µs × 34 ≈ 2 ms a frame
of boundaries before any work, and every phase's workgroup state (grid scans, tested
lists, hit flags) moved to storage. A car's phase is 1–3 % of the GPU, so spreading a
phase over more cores buys nothing; the solo frame's measurement agrees (the wide
passes: 0.63 ms against 0.44 ms for one sedan, `vehicle-perf-first-principles.md`).
Rejected.

**(c) A global broadphase and per-pair work lists sized by indirect dispatch.** The
broadphase already exists and is global (the plan passes: frame boxes, a hash grid, the
connected components, once a frame), and each substep's pass rechecks the k² member
boxes. What (c) adds that pays is the per-pair part: the owners' triangle tests (the
largest cause, 25–56 ms) spread over many workgroups, with their counters in global
memory, between the pass's front (scans, candidates, grid) and its levels. Two more
dispatches a substep (the walk listing pairs, then the pairs' tests) and the levels in a
third: 170 a frame, ≈0.35 ms of boundaries. The levels stay in one workgroup a group:
their depth is the specification's (§3) and a level's barrier (0.2 µs) is ten times
cheaper than a dispatch.

**(d) Persistent workgroups with a global barrier in the kernel** (spin on an atomic
counter): no forward-progress guarantee between workgroups in WebGPU (a workgroup that is
not resident never arrives: a hang, which on this machine needs a reboot). Rejected.
**(d′) Each car's contact on its partners' state a substep old** (no barrier inside a
substep, groups dissolve into solo frames): changes results; excluded by the owner.

**(e) Solo cars beside the chain.** One queue runs dispatches in order, so with (a) a
solo car's whole-frame dispatch would run before or after the group chain, adding its
≈9–12 ms to the frame. Instead, in a frame with a group every solo car steps in the
same per-substep vehicles' dispatch, its frame cut at the step boundaries
(`EB_SOLO_STEP`, its pair list's state kept in its records between steps). Without a
group nothing changes (the solo frame whole, one dispatch).

**Chosen: (a) + (c) for the tests + (e)** (§5). The measured split of the work (§6)
decides the rest.


## 5. The design and its proof of exactness

### 5.1 The schedule

A frame whose plan formed a contact group (`vvArgs` writes the indirect arguments; with
no group every one of these dispatches has no workgroups and the solo frame runs whole,
as before):

| Dispatch | Workgroups | Does |
|---|---|---|
| groups' start (copy `steps + 1`, `SPLIT_BEGIN`) | one a group | `contactGroup`'s prologue: members in vehicle order, run prefixes, each member's start (rebase, wing check) |
| per substep s: **vehicles' step** (copy s) | one a vehicle | a member: `splitMember` (its coupling, drive, `contactMembersStepAt` for that member only on its partition's lanes, or `contactMemberStep` past four members, its surface pass); a solo vehicle: step s of its own frame (`explicitBeamsFrameStep`, `EB_SOLO_STEP`) |
| per substep: **groups' front** | one a group | the pass's scans, throttles, candidates and their grid (`vvPassSmallFront`), exported to the group's scratch; past four members the whole pass (`vvPass`) |
| per substep: **tests, walk** | 16 a group | each listed triangle's grid cells: its pairs listed (or, without the pair list, tested inline) |
| per substep: **tests, pairs** (copy `steps + 2 + s`) | 16 a group | each listed pair tested; an owner past the list's room re-walks inline |
| per substep: **groups' levels** | one a group | the hits applied level by level (`vvPassSmallLevels`); after the last step the group's frame end |

170 dependent dispatches for a 34-step frame (68 with the tests left in the group's
workgroup, `set_group_split_tests(false)`). The old path stays selectable
(`set_group_split(false)`, `RUBBLE_GROUP_SPLIT=0`).

With self-collision split and the members' kept pair lists (`PERF_MEMBER_LIST`,
vehicle-perf-first-principles.md §8.19) a substep is: the vehicles' step (a member's cut
at its self-collision tests, as a solo vehicle's), the self-collision walk and pairs
dispatches over every vehicle, **the members' hits** (`memberRest`: a workgroup a
vehicle when there is a group, a solo vehicle's returning at once), then the groups'
four dispatches. The pass's scratch between its dispatches (per owner counters, the
grid's offsets) and its listed pairs are in vehicle contact's own regions (`vvScratch`,
`vvScratchOff`, `vvTriPairs`), not the members' self-collision words: those hold a
member's pair list and its state from one substep to the next.

### 5.2 Why the results are the same, bit for bit

The new path runs the same functions on the same values in an order the old path's
results do not depend on:

1. **A member's step** is `contactMembersStep` restricted to one member
   (`contactMembersStepAt(…, only)`): the same code with the other partitions idle, the
   member on lanes 0..P−1 with P = `vvPartLanes(k)` of its group, so every per-lane
   decision (the grid's `listed · cands > BRUTE_PAIRS_PER_LANE · P`, the early pair list's
   room, the lanes per triangle) is the one the group's workgroup made. The two
   group-wide decisions it took over all members are kept or are inert: whether
   `selfApply` runs (any member with triangles: the same static test over the group) and
   whether the grid scan's barriers run (only the gridded member's own scan writes
   anything). Past four members, `contactMemberStep` is already one member at a time on
   every lane.
2. **The coupling and the drive** of a member are the per-vehicle functions the group
   called (`cplReactBy/IntegrateBy/SpheresBy/CountBy` on the partition, `driveWheelsSplit`
   with the member's wheels); the coupling's workgroup table (`cplGsAdd`, which decides
   whether a member's spheres and node ranges are in workgroup memory) is rebuilt for
   the whole group in the same order, so every member gets the same bases and counts.
3. **Nothing crosses a dispatch in workgroup memory.** What `contactGroup` kept there
   across substeps: the coupling's header copy and table (rebuilt each dispatch from the
   buffers, which the frame does not change), the drive's on word (re-read), the
   contact pass's counters (cleared at the pass's start, as each member step cleared
   them), the hit flags `gsUniformHits/Open` (reset where the member step reset them). The
   coupling's per-step values (spheres, node ranges, reaction terms) are written and read
   within one step's coupling and node phase, inside one dispatch.
4. **The contact pass** is `vvPassSmall` cut at its barriers into `vvPassSmallFront`,
   the tests and `vvPassSmallLevels`: the old path calls the three in a row. Between the
   cuts the values the next part reads move through the group's scratch (its first
   member's wide-pass grid words, dead between that member's steps): per owner the
   offered, listed, candidate, reset and fix words, the partner bits, the fastest
   candidate's speed and the grid's offsets; the tests' counters (pairs, hits) are global
   atomics instead of workgroup atomics. The tests themselves are the same functions
   (`vvTriangleCandsList/Grid/Small`, `vvPairTest`, through the counter accessors). The
   lists they fill were already in atomic order in the group's workgroup: the pair list's
   order, the hit list's order and the visit counts never decided anything (the hits are
   applied in the specification's order, triangle then node, by `vvApplyOrderedBuckets`,
   whatever order they were found in; the marks are idempotent stores; forces are
   fixed-point sums).
5. **A solo vehicle's frame a step a dispatch** (`EB_SOLO_STEP`): its start (wing check,
   rebase) in step 0, its end (coupling end, pair-list statistics, triangle finish, slide
   settle, group drops, bounds check, step count) after the last; what its frame keeps in
   workgroup memory across steps is saved after each step and loaded before the next:
   the hit flag (its segment record) and the reused pair list's state (`REUSE_WORDS` past
   the grid buckets, into its wide-pass grid counts). Every other workgroup value the
   frame reads is written earlier in the same step.
6. **Workgroup memory is not zeroed natively** (`zero_initialize_workgroup_memory:
   false`). By 3 and 5 every workgroup value a split dispatch reads it wrote earlier in
   the same dispatch; the one exception is inert (a member's lane 0 does not read the idle
   partitions' tested counts: `contactMembersStepAt`). Checked bit for bit against the
   zeroing build (`set_group_split_zero(true)`). Browsers always zero (Tint, in parallel:
   §2).

The ordering constraint of §3 is unchanged: the levels are the old path's, one
workgroup a group; the members' steps and the tests are what spreads.

### 5.3 What it does not change

- Groups are formed by the same plan passes (`vv_bounds`, `vv_pairs`, `vv_group`), a
  connected component of frame-box overlaps, at most `VV_MAX_GROUP` = 64 members (a
  larger component's vehicles run alone and the frame reports it: the capacity error).
- Groups past four members keep the owner-after-owner pass (`vvPass`) in one workgroup:
  their members' steps spread, their pass does not. A pile-up of more than four cars in
  one component pays it (§6).
- Vehicle sleep, the test kernel variants and the diagnostic pipelines keep the group's
  workgroup (the host's `group_split_on`).
- WebGPU only (native wgpu and browsers). Metal and CUDA run the old group path
  (known-bugs P1's table).


## 6. Results

### 6.1 Exactness (verified)

- `group_split_gpu` (native, M3 Max): two generated sedans head on (40 frames of 17
  steps, a group in all 40) and a column of six (30 frames, groups of up to six: the
  larger groups' pass): every state word the same as the group's workgroup, for split
  with the tests across workgroups, split with each group's pass whole, and split with
  workgroup memory zeroed.
- The crash town, `EV_SPLIT_AT=60,120,150`: at 2 cars bit for bit (no group forms).
  The 8- and 32-car checks did not complete (§6.3).
- `explicit_beams_bitwise`: 12 passed with the split path on (then the default).
- `explicit_plan_modules`: passes with the split modules in the WebGPU-only table.

### 6.2 Compile cost (measured)

Native pipeline creation (`examples/contact_compile_time.rs`, Metal, wall s, zeroing
on / off): the old contact module (nosurf) 26.7 / 26.6; the split members' module
54.3 / 44.3 (the solo frame and the member step in one kernel, with surface contact);
the groups' module 8.8 / 8.6; the tests' module 0.4 / 0.0. The split pipelines are
created at the first frame with a group (natively) and are in `compute_specs` as
deferred (a browser compiles them after the first frame's). On NVIDIA's Vulkan driver
(Modal) each large module takes minutes cold (engine/tools/modal/README.md).

### 6.3 Fleet timings: not established

The Mac's GPU was shared with jobs outside the GPU lock for the whole session
(utilisation 82–100 % sampled before every locked run, the same count varying 1.8×
between runs), and from about 20:46 it stayed at 100 % with locked jobs starved (the
old path stalled the same way as the split one), until the coordinator paused every
stream. The split frame's timings at 8/16/32 cars, apart and town, split against old,
and 64/128 were therefore not measured. The 2-car town (no group) ran: the explicit
frame alone 10.7–16.2 ms either way, the same within noise.

### 6.3a The pinned GPU and the split path

The coordinator suspected a runaway kernel in the split path. Two of my runs on the
pinned GPU were killed at the 600 s limit (`explicit_contact_gpu`'s
`a_group_over_the_limit_runs_alone_and_is_reported`, about 21:06–21:16, split on; the
8-car town, about 20:46–20:56, split on). The utilisation sampled inside the lock just
before the 8-car run already read 100 / 100 / 100, and that test with the split off
stalled the same way at 21:57. Checked since, off the Mac's GPU:
- every split dispatch count comes from the plan's `vvArgs` (groups ≤ vehicles,
  vehicles, groups × 16, now bounded by the fleet size), from a zeroed buffer;
- every loop bound in the split path is a constant or a count written earlier in the
  same substep in global memory; no workgroup value is read before it is written;
- on Modal (T4), the over-limit test with the split on ran 3 × 1,750 s at **GPU 0 %**,
  one core compiling the cold kernels (the old path passed it in 1,038 s, nearly all
  compile); no dispatch was reached, so the test itself is not yet run there.
Not established: whether a split frame ever ran away on the Mac.

### 6.4 The default

`RUBBLE_GROUP_SPLIT` defaults to **off** until the suites and the fleet runs are clean
on an idle GPU; `set_group_split(true)` / `RUBBLE_GROUP_SPLIT=1` turns it on. The split
path also stands down for a fleet past 4,095 vehicles (its tests' dispatch, 16
workgroups a group, must stay within WebGPU's 65,535 a dimension).

## 7. Findings on the way

- **Native workgroup zeroing costs 0.7–1.8 ms a dispatch** for the vehicle kernels'
  workgroup arrays (§2). The existing one-dispatch frame pays it once a frame (about 4 %
  of one car's frame). Lossless to remove after the same audit (the split pipelines do).
- **The old contact module with surface contact does not build on native Metal** (more
  than 31 threadgroup variables; `examples/contact_compile_time.rs` fails on
  `explicit_beams_contact_wide`). Native hosts run the `_nosurf` twin when no vehicle
  has surface contact; a native fleet with surface contact and vehicle contact would
  fail at pipeline creation. Not reproduced in a scene; reported, not fixed.
- **The members' self-collision walks every step in a group** (no reused pair list):
  17–46 ms of the frames in §1, the second cause. The member pair list was not used
  by this design (the member's step is today's, bit for bit), so its one-unit mismatch
  was not investigated here.

## 8. Contact without groups: fleet-wide lists (design, 2026-11-16)

Why: §5's pass is per group. A group past eight members takes the larger groups' pass on
one core, and 32 cars side by side are one group: 93–96 % of a 125–452 ms frame with or
without a hit (`vehicle-perf-first-principles.md` §8.17). A group is only a connected
component of overlapping boxes; nothing in the contact rule needs it. What the rule
needs is per owner: its partners this step, its candidates, its listed triangles, its
hits, and the owners' order.

### 8.1 What stays exactly as it is

The leaf rules, unchanged and reused: the step scans' box and approach tests, a
triangle's throttle step and near-box test, the candidate's box test, `vvInBox`,
`vvFind`, `vvClip`, `vvPairTest`, `vvTry`, the hits' canonical order (an owner's hits by
contact point, a point's hits by triangle), `vvThrottleEnd`, the two-body counts, the
snapshot and the residual phases. So the same hits, applied in the same order.

### 8.2 Keyed by vehicle, not by a group's slot

Today an owner's state is addressed by (group offset, the owner's index a in the group):
counters in workgroup arrays of `VV_PART`, runs and hit lists at `runOff + a · points`,
grids in an eighth of the workgroup's buckets, partner bits in a `VV_PART²` table. In
this design every one of those is the owner vehicle's own:

| state | today | here |
|---|---|---|
| counters (offered, listed, candidates, pairs, hits, reset, fix, fastest candidate) | workgroup arrays, exported to the group's scratch | a record a vehicle in `gAcc` |
| partners this step | `gsVvPair[a · VV_PART + m]` | the broadphase's pair list (`vv_pairs`), a bit a pair |
| candidate run, its sorted copy, ranks | `vvRunSmall`, `vvSortAt`, `vvRankAt` at the group's run offset | one fleet-wide run, each owner's part at its prefix of the offered counts |
| candidate grid | `GRID_BUCKETS / VV_PART` workgroup buckets | the owner's own grid region (`gridCountBase`, `gridOffBase`: 1,024 buckets, dead between its steps) |
| listed triangles, listed pairs | already the vehicle's (`vvTestedAt`, `triPairsAt`) | the same |
| hit list, its order | `vvHitSmall`, `vvOrderAt` at the group's offset | one fleet-wide list, an owner's part at its prefix |
| two-body point counts | slot a of `VV_PART` a node | the owner's rank among the point's vehicle's partners |

Explicit limits (a clear error flag when passed, never a silent drop): the fleet-wide
candidate run and hit list hold `VV_PART` times the fleet's contact points, and a
vehicle has at most `VV_PART` partners with two-body contact at once.

### 8.3 The owners' order without a group

Owner v's level is 1 + the highest level of an owner u < v with hits whose touched
vehicles ({u} and its partners) meet v's; 0 without hits (`vvLevelsSet`'s rule, which
never used the group beyond listing the owners). Over the fleet it is a relaxation on
the pair list: a lane a vehicle in one workgroup, a round a barrier, until no level
changes (the rounds are bounded by the longest chain of touching owners).

### 8.4 The schedule of a substep

Each a dispatch over every vehicle, pair or element (empty when its count is 0, and
then about 2 µs):

1. the pairs' step scans: partners, offered counts, resets;
2. every offered owner's triangles: throttle step, near-box test, listed;
3. every pair's contact points into the owner's part of the run;
4. each owner's grid: counts with 3, offsets (a workgroup an owner), sorted copy;
5. the walk (today's `splitTests`, over every owner);
6. the listed pairs' tests (the same);
7. the levels: the plan of 8.3, then per level in use the hits applied (a thread a hit;
   a point's chain found by scan for a short list, by buckets for a long one), then
   the residual phases where two-body contact is on.

Cars' own steps are not part of this: a workgroup a car or the fleet-wide passes, as
measured for the fleet.

### 8.5 Stages, each bit for bit against the path before it

1. Per-owner state keyed by vehicle (8.2) under the existing per-group dispatches.
2. Steps 1–4 as fleet-wide dispatches; the front's workgroup goes.
3. Step 7 fleet-wide; the group's workgroup, `VV_PART` as a group limit and the larger
   groups' pass go from the default path.

### 8.6 What is built (2026-11-16): stages 1 and 2

Kernel `engine/slang/kernels/explicit_beams_fleet.slang`; addresses `fl*` in
`explicit_beams.slang`; perf flags 4096 (`PERF_FLEET_STATE`) and 8192
(`PERF_FLEET_FRONT`), both on by default; measurements in
`vehicle-perf-first-principles.md` §8.25. Stage 3 (the levels) is not built: §8.7.

**Stage 1, the state by vehicle (flag 4096), under the groups' own dispatches.**

| state | where |
|---|---|
| counters (offered, listed, candidates, pairs, hits, reset, fix, fastest candidate) | gAcc `flOwn(v)`, 8 words a vehicle, zero between passes (`flOwnerEnd`) |
| the plan's pairs by vehicle | gNext `flAdjFirst`, `flAdjAt`: built by the grouping kernel (`vvAdjBuild`), an entry a pair and side |
| partners this step | gNext `flBitsAt(d)`, a word a pair's side, written by the pair's scan (`flScanPair`) |
| candidate run, sorted copy, ranks, hit list | the regions as before (`vvRun`, `vvHits`), an owner's part at `flRun(v)`, its prefix of the offered counts |
| candidate grid | the owner's own 1,024 counts (gAcc `flGridCount`) and offsets (`vvScratchOff`), 128 buckets an owner before |
| two-body point counts | slot = the owner's rank among the point's vehicle's two-body partners of the step (`flSlotAt`, set by `flOwnerPlan`) |
| per owner, a step | gBeams `flBoxAt`: its box grown by its range and reach, its partners' boxes' union |

The leaf functions are the small pass's own (`vvTriangleCandsList`, `vvPairTest`,
`vvTry`, `vvPassSmallLevels`, …); the address helpers (`vvRunSmall`, `vvHitSmall`,
`vvSortAt`, `vvGridBucket`, `vvOffAt`, `vvHitAdd`, `vvPairAdd`, `vvSpeedAt`,
`vvSplitPoint`) follow a thread-local mode (`gFl`) in the modules built with
`EB_VV_FLEET` (the groups', the tests', the fleet's). The levels still run in the
group's workgroup: lane 0 imports the owners' records into the workgroup arrays
(`flImport`), and their hit buckets stay the workgroup's (stage 3's to replace).

Limits, each raised through the contact status' overflow word
(`explicit_plan::contact_error`), never a silent drop: the owners' offered counts
together past `VV_PART` times the fleet's contact points (the pass of that step is cut
short); an owner's hits past twice its offered count; a vehicle with more than
`VV_PART` two-body partners at once.

**A finding on the way: the plan's pair list can hold a pair twice.** A vehicle sits
in a vehicle-grid bucket once for each of its cells that hash there, and
`vvFindPairs` then meets it twice in one cell. The grouping does not mind. Partners
read from the list do: the partner's contact points became candidates twice and its
hits were applied twice (four cars, 18 frames after first contact; found by the
harness, not by the suites). `vvAdjBuild` marks a pair's later copies (bit 4 of its
bits words), and the scans leave them.

**Stage 2, the front and the tests as dispatches over the fleet (flag 8192).** A
substep's contact pass, where a group's front (one workgroup) and its tests (64
workgroups, twice) were:

| dispatch | module, step word | threads | sized by |
|---|---|---|---|
| the pairs' scans | `fleet_front`, step | a pair | the contact plan (`vvArgs` 32) |
| the owners' parts: prefix of the offered counts, each owner's records, the sizes of what follows | `fleet_plan`, step | one workgroup | the contact plan (36) |
| every triangle's throttle step, every contact point's gather with its grid count | `fleet_front`, `SPLIT_TESTS` | a triangle or point | the first plan |
| each owner's grid offsets | `split_tests`, step | a workgroup a vehicle | the first plan |
| the candidates' sorted copy | `fleet_front`, `SPLIT_LEVELS` | a place of the run | the first plan |
| the listed triangles' prefix, the walk's size | `fleet_plan`, `SPLIT_TESTS` | one workgroup | the first plan |
| the walk | `split_tests`, `SPLIT_TESTS` | a part of a listed triangle (128 workgroups' worth aimed at) | the second plan |
| the listed pairs' prefix, the pairs' size | `fleet_plan`, `SPLIT_LEVELS` | one workgroup | the first plan |
| the listed pairs | `split_tests`, `SPLIT_LEVELS` | a pair | the third plan |
| the levels | `split_group`, `SPLIT_LEVELS` | a workgroup a group | the contact plan |

Sizes travel in two small argument buffers of the fleet's own (`gArgs` at binding 10,
`gArgs2` at 11): a dispatch may not bind the buffer it runs from, so the first plan
(run from the contact plan's arguments) writes the first buffer and the later plans
(run from the first) write the second. With no owner offered a partner the first plan
writes zeros: a group whose members have not met costs the scans and one workgroup a
substep. The groups' levels zero both buffers after the pass, so with no group every
one of these is an empty dispatch. Threads find their owner by a binary search of a
prefix over the vehicles (`flFind`); the contact points' prefix is static.

A group past `VV_PART` members still takes the larger groups' pass (`vvPass`) in its
own workgroup (a dispatch that has workgroups only when such a group exists), and its
pairs are no one's partners in the fleet-wide passes (`flVehicleOn`).

### 8.7 Stage 3, not built: what it needs

The levels in the group's workgroup are now the largest part of contact after the
walk (3.2 ms a frame of eight cars at rest), and a group past `VV_PART` is unchanged
(32 cars: 139–270 ms a frame). What stage 3 has to add, from building the first two:

- the owners' levels by relaxation over the pairs (one workgroup, a round a barrier)
  and the hits' prefix (`flPrefixAt(3, v)` is reserved);
- a dispatch a level and phase (application, three residual phases) that knows its
  level: the step's three parameter copies are used up as phases, so levels need
  copies of their own (past `PARAM_COPIES`, a level and phase each) or a module a
  phase; with L level slots that is 4 L dispatches a substep, empty past the highest
  level in use, and one workgroup that loops the levels past L (lossless, slow, rare);
- the hits' chains without the workgroup's buckets (a scan of the owner's hits for a
  short list, a counting sort in global memory for a long one);
- the snapshot, the throttle updates and the two-body clears as threads over hits and
  listed triangles (independent of the levels: these could go first);
- then `flVehicleOn` for every group, the two-body slot limit as the only bound on
  partners, and `vvPass` out of the default path.

### 8.8 The candidates kept across substeps (2026-11-16)

Perf flag 32768 (`PERF_FLEET_KEEP`, with 8192; a checkbox on the lab page), kernel
functions `fk*` in `explicit_beams_fleet.slang`; measurements and the default in
`vehicle-perf-first-principles.md` §8.26. Steps 2 to 5 of §8.4 find, every substep, the
(triangle, contact point) pairs whose box test passes; in a pile that is over 10,000
pairs from 4,000 walked triangles for 0 to 20 hits, and almost nothing has moved since
the substep before. With the flag the pairs are a Verlet list with a conservative skin:

- **The build** lists every (triangle, contact point) of two vehicles the plan pairs
  whose point is inside the triangle's box grown by its range, `CLIP_DEPTH` and a margin
  (`FK_MARGIN`, 0.1 m), whatever the throttle, the step's partner bits or the owner's box
  say in that substep. It is the front's own machinery with the padding: the gather (every
  plan partner's points inside the owner's box grown by the same amount), the owners'
  grids, the sorted copy, and a walk over every triangle. The pairs are (triangle, node)
  in the owner's pair region (`vvTriPairsAt`, after a two-word signature). Each tracked
  node's position and each vehicle's origin are recorded.
- **Every pass** tests the kept pairs with what the front and the walk would have applied
  to them, on that pass's values (`fkPairTry`): the triangle was listed by its throttle
  step this pass (`flTriangle`, unchanged, sets a word a triangle); the two vehicles are
  partners this pass (the scans' bits); the point is in the owner's box (`flPoint`'s
  test); it is in the triangle's box (`vvInBox`, which sets the had-candidates mark);
  then `vvFind`, the two-body counts, the hit listed, the hit mark, as `vvPairTest`.
- **The bound.** A thread a triangle and contact point (in the scans' dispatch) takes, per
  vehicle, the farthest any tracked node has moved since the build relative to the
  vehicle's first node, and the fastest tracked node's speed. For each pair of vehicles
  that are partners this pass, the first plan adds the two bounds, the first nodes'
  relative motion (in the owner's frame as the tests compute it: local positions plus the
  origins' difference, so a rebase cancels), both fastest nodes' travel over a step
  (anti-clip's pads, which grow with speed) and room for rounding (2 mm and 10⁻⁵ of the
  coordinates). Over the margin for any pair, or a pair the build did not cover, or a
  missing signature: the list is built again in that pass.

Why the same hits: a pair today's walk lists has its point within the triangle's box
grown by the range and at most `CLIP_DEPTH` plus the two nodes' travel. At the build the
point was nearer to that box than this distance plus the pair's relative motion since,
which the bound holds under the margin: the kept list has the pair. The other conditions
(listed, partners, owner's box, `vvInBox`) are evaluated exactly, so the pairs that reach
`vvFind` are today's set, each once; the hit list's order does not matter (the levels
apply hits in the canonical order).

What forces a build or is handled exactly:

| | |
|---|---|
| partners of the step, the approach test, `reset` / `fix` | the scans and the first plan run unchanged every pass; a kept pair is tested only for partners of the pass; a pair of vehicles the build did not cover (the plan's list changed, a group formed) forces a build |
| the triangle throttle, its pending flag | `flTriangle` runs unchanged every pass on every triangle; the kept list holds pairs of throttled triangles and tests them only in a pass that lists the triangle |
| broken or retired triangles and nodes | not used by the build (geometry alone); `vvTriangle`, `vvInBox` and `vvFind` decide in the pass |
| anti-clip's pads | the bound includes both vehicles' fastest tracked nodes' travel over a step; too fast for the margin at all: the pass takes the front's own path |
| the marks | had-candidates from `vvInBox` passing for a listed triangle, hit from `vvFind`: the conditions the walk and `vvPairTest` set them on |
| two-body counts | `vvSplitCount` at the same hits |
| the frame boundary | the list, the records and the bound's inputs are in the buffers (snapshots hold them); the plan's new pair list is compared with the build's covered partners by vehicle |
| another path writing the pair region (the flag off for a frame) | the signature in each owner's first two words is gone: a build |
| a build past an owner's room (32,767 pairs) or the run's | that pass tests in the padded walk; the front's own path for the next 136 passes |
| NaN or infinite positions and speeds | the bounds are compared as bit patterns, anything not finite is over the margin: the front's own path |

The dispatches do not change: a pass that reuses the list runs its tests in the grids'
dispatch (the tests' module, the step word alone, sized by the first plan), and the
sorted copy, both later plans, the walk and the listed pairs' dispatch are empty; a build
runs every dispatch, its tests in the listed pairs'.

Limits. The bound is one number a vehicle, so a wheel spinning or a piece flying off
rebuilds the list of every owner in the fleet; the build is fleet-wide.

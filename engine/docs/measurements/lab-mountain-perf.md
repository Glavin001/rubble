# The vehicle lab's mountain: where a World step's time goes

The report: `vehicle-lab.html?mode=mountain&car=gen&…&rockslide=1&steepness=55&roughness=0.25&launch=12&surface=1`
in Chrome, generated sedan, surface contact on: 13 World steps/s (21 % of real time),
"GPU not ready" 78 %, about 75 ms of GPU per World step, 6 shapes in the coupling window.

Everything below is native (wgpu on Metal, M3 Max, `examples/vehicle_lab.rs --measure
mountain-perf`), wall time of a World step that is waited on. The generated sedan is the
live export (`sedan_new`: 548 nodes, 4,253 beams, 528 collision triangles). Scenario:
seed 1, 55°, roughness 0.25, launch 12 m/s, rockslide on, surface contact on. GPU
utilisation before each run is given; the owner's page was open for most runs (70–99 %),
so absolute numbers are high, but every comparison is between runs of the same
trajectory (the speeds match to 0.1 km/h at every second) or, where marked, on the same
state.

```bash
cargo build --release -j 4 -p engine-world --example vehicle_lab
python3 engine/scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 RUBBLE_SURFACE_SPLIT=0 \
  engine/target/release/examples/vehicle_lab --mode gpu --measure mountain-perf --secs 30 \
  --car ~/Development/rubble-artifacts/sedan-live/mods/sedan_new --surface 1 \
  --steepness 55 --roughness 0.25 --launch 12 --rockslide 1 [--ab 4,8,12] [--alternate culls]
```

## 1. Where the time goes

| Configuration | ms per World step | GPU use before |
|---|---|---|
| Free fall (t 2–3 s), surface on, native default (split frames) | 67–69 | 82–99 % |
| Free fall, surface on, `RUBBLE_SURFACE_SPLIT=0` | 12.6–12.9 | 94 % |
| Free fall, surface off | 11.6–12.4 | 82 % |
| Flat arena, parked, no proxies: split / unsplit / surface off | 80 / 12.6 / 10.2 | 99 % |
| Old car (`sedan_new_prev`, 363 triangles) vs new, surface on, free fall | 66–68 vs 68–71 | 92–99 % |
| Authored showcase sedan, surface on, free fall | 63–70 | 97 % |
| Mountain in contact (t 4–13 s), unsplit, surface on | 19–49 | 69 % |
| Same, surface pass ablated on alternate windows | 11–16 | 25 % |
| Same, only the narrow phase ablated (box cull kept) | 11–16 | 97 % |
| Same, the pair path off on alternate windows | no faster (−5 to +10 %) | 89 % |

Two separate costs:

**(a) Native split frames: 55 ms per World step of workgroup-memory zeroing.** With the
World's coupling and surface contact, a frame runs as segments with the surface pass as
nine small pipelines between them (`3e0aa017`, made so Chrome compiles the surface pass
in seconds): 34 × 10 dispatches of one workgroup per World step. Each costs about
0.16 ms whatever it does: the surface pair kernels that return at once cost the same,
duplicating them adds 0.16 ms each, giving them zero workgroups removes it. naga's MSL
writer zero-initialises every workgroup variable a kernel uses from one thread
(`if (local_invocation_index == 0u) { … }`, naga 30 `back/msl/writer.rs`
`write_workgroup_variables_initialization`); the surface pipelines use `gsCpl`
(2,080 words). `examples/dispatch_cost.rs`: 340 dependent dispatches of a trivial kernel
take 1.8 ms; with a used 2,080-word workgroup array 54 ms; with 8,192 words 211 ms.
`examples/explicit_dispatch_cost.rs`: 340 dispatches of the shipped
`explicit_surface_find_pair` with no vehicle (it returns at once) take 54 ms of wall.
Chrome's Tint spreads the zeroing over the workgroup's lanes, so this is a native cost
only: the page's own numbers with split frames (`vehicle-lab.md`, Shoot idle 3.7–4.0 ms
of GPU per step) show it does not pay it. It does make native timings of the lab
meaningless unless `RUBBLE_SURFACE_SPLIT=0` (bit for bit the same results: the split
commit's suites hold both paths to the CPU twin).

**(b) The surface pass's narrow phase on the mountain: 5–35 ms per World step.** The
mountain is slabs 3 m thick turned 55° plus rocks. A turned slab's bounding box is about
14 × 20 × 36 m, so it holds the whole car: every triangle, beam edge and node passes the
box cull against it. The pair path (`surfPairList`) listed 3,400–3,800 (element, shape)
pairs a step for 0–90 hits, past its 2,048 room, so the step fell back to the
per-element loops: each lane takes elements and runs the full closest-feature query
(`fqContact`, `fqSegContact`) against every shape whose box it meets. Ablating only
those queries (the box cull kept) removes the whole surface cost. Self-collision is not
the cost (the old export with 363 triangles costs the same; surface off removes it).

The rest of a step (11–16 ms here, 33–38 ms after 24 s when 40+ shapes and rockslide
boulders are in the window, 150–175 ms in one run where flung nodes made the car's box
hundreds of metres tall and the window held 100 shapes and 700 hull planes) is the
explicit frame and the coupling; it was not part of this fix.

## 2. The fix: a separation cull (lossless)

`engine_schema::feature_query::separated` and its Slang twin `fqSeparated`
(`slang/physics/feature_query.slang`): an element (a triangle's three points, a beam
edge's two) is skipped when it lies wholly outside one of the shape's outward face planes
by more than the band `r` plus a rounding allowance (1 mm and 1e-4 of the largest
coordinate read): a box's six planes, a hull's planes, a cylinder's ends; for a sphere, a
capsule and a cylinder's side, the gap from the element's box to the shape's core. Such
a plane holds every point of the element farther than `r` from every point of the shape,
so the narrow phase, whose computed gaps are within 1e-5 of the largest coordinate and
1e-4 m of the true ones (the same bound the existing edge culls `fqFar` rely on), finds
nothing. Nodes (point contact) are not culled.

Where: the CPU twin's triangle and segment loops (`explicit_surface.rs`, switch
`ExplicitBeams::surface_separation`, counter `surface_queries`); on the GPU every
triangle and segment narrow-phase site of `explicit_beams_surface.slang` (static slots,
proxies, the casts' gathering, the pair path's contact) and, the part that matters on
the GPU, the pair path's listing: a separated pair is not listed. The listing then fits
its room and each listed pair gets its own lane. The GPU culls are under the existing
`fqCullsOn()` switch (`ExplicitBeamsGpu::set_surface_culls`).

### Before → after (same trajectory, unsplit, native)

| t (s) | shapes | pairs listed before → after | ms per World step before → after |
|---|---|---|---|
| 4 | 4 | 3,833 → 553 | 19.5 → 14.5 |
| 5 | 9 | 2,735 → 379 | 25.9 → 17.3 |
| 8 | 30 | 3,413 → 668 | 29.8 → 20.9 |
| 9 | 36 | 3,825 → 591 | 42.9 → 21.7 |
| 10 | 33 | 1,865 → 251 | 49.3 → 22.1 |
| 13 | 26 | 3,778 → 541 | 46.3 → 21.6 |
| 19 | 36 | 3,694 → 533 | 47.2 → 23.9 |
| 22 | 33 | 3,705 → 540 | 39.9 → 25.4 |
| 25 | 36 | 1,278 → 171 | 51.9 → 36.2 |
| 29 | 23 | 2,136 → 296 | 49.9 → 36.8 |

GPU use before the runs: 69 % (before), 0 % (after). Free fall is unchanged (12.6 →
12.2 ms). The cull at the narrow-phase sites alone (without the listing) gained only
3–6 %: on the GPU a step waits for its slowest lane, and the few elements that do meet a
shape still ran their queries serially; listing fewer pairs puts them back on the pair
path, one pair a lane.

CPU twin (`--measure mountain`, 12 s): box narrow-phase queries 37.2 M → 0.29 M, wall
20.8 s → 12.0 s, the run's summary identical.

### Proof it is lossless

- **Same state, GPU, bit for bit:** `--ab T…` snapshots the car's solver, runs 20 frames
  of 34 steps with the culls on and again with them off (the coupling's proxies as the
  World last wrote them), and compares every state word (`state_words`: nodes, beams,
  accumulators, groups, misc, trigger keys; `coupling_words`: proxy records and sums).
  Identical at t = 4, 6, 8, 10, 12, 16, 20, 24, 28, 29.5 s (on 17–39 ms a frame, off
  19–70 ms).
- **Whole run:** before and after give the same speed, drop and coupling counts at every
  second of 30 s.
- **CPU:** `tests/explicit_surface_cull.rs`: a sedan pushed onto a 55° slab steps 400 steps
  bit for bit with the cull on and off (12,695 queries against 771,200; 1,816 hits), and
  every element the cull skips is checked to have no contact; across the 12 s CPU
  mountain run every separated pair's query was evaluated and none had a contact.
- GPU suites (native wgpu on Metal, CPU == GPU bit for bit): `explicit_loop_cap` (12),
  `explicit_surface_proxy_gpu` (12), `explicit_contact_gpu` (5), `explicit_beams_bitwise`
  (4), `surface_pairs`, `vehicle_lab_reset` (3), `surface_contact_gpu` (3); CPU:
  `explicit_surface`, `_cast`, `_friction`, `_cull`, `explicit_thin_box`.
- **Not verified: `engine-metal` `vehicle_parity` and `explicit_surface_proxy_parity`.**
  Their kernel's cold Metal compile (Slang to MSL, the frame with the surface pass in
  it) ran past a 25 and a 20 minute kill timer (`MTLCompilerService` at 100 %, tests
  waiting on it); the change invalidated the cached build. The added code is +0.4–1.0 %
  of each module's inlined size (`scripts/wgsl-inline-size.py`: `explicit_surface_prep`
  +944 lines of 90,248, `explicit_surface_find_tri` +374 of 65,058, `explicit_beams_wide`
  +2,057 of 499,054, `explicit_beams_contact_wide` +4,114 of 1,036,117), so the compile
  was presumably as long before; that was not measured.

### The gate

`tests/explicit_surface_cull.rs` `a_slab_on_a_slope_queries_only_the_elements_along_its_face`
(committed failing first): a sedan resting on a slab turned 55°; narrow-phase queries must
not exceed the elements within `r` + 0.1 m of the slab's face. Before: 1,928 queries (every
element) for 43; after: 5.

## 3. Not done, and the owner's decisions

1. **Chrome was not measured** (no browser in this task). The page's surface kernels are
   the same Slang, so the pair-path overflow and its fix apply there; the 75 ms itself and
   how much of it this removes need a page measurement.
2. **Native split frames** (§1a): native hosts split by default (`RUBBLE_SURFACE_SPLIT`
   unset). Turning the split off natively by default would take 55 ms off every native
   World step with surface contact (bit for bit the same results), but native tests would
   then no longer run the browser's split path unless they ask for it. Alternatively
   native shader modules could skip workgroup zeroing (`create_shader_module_trusted`),
   which needs every kernel checked for reads before writes. Neither was adopted.
3. **The coupling's cost with many shapes** (33–38 ms a step with 40 shapes; 150+ ms when
   flung nodes spread the window over 100 shapes and 700 hull planes) is the next cost on
   the mountain; rockslide boulders accumulate around a car that has stopped.
4. **The lab's arena fixtures** ride along as the car's static shapes on the mountain (far
   overhead, for one layout per template), so the per-element surface loops always run
   against them (cheap: the box cull rejects them).

## 4. Surface contact off: the self-collision pair list (2026-11-14)

The report: `vehicle-lab.html?mode=mountain&car=gen&steepness=55&roughness=0.25&launch=12&surface=0&drag=0`
"Mountain (hard)": smooth at first, then slow. Headless Chrome (`vehicle-lab-record.mjs
--probe`, 60 Hz, GPU idle before each run), GPU time of a World step: 7.6 ms in free fall,
11–12.6 ms on the slope (t 5–16 s), 13–14 ms to 19 s, then 26–28 ms (30 World steps/s) from
t ≈ 21 s on. Air drag on gives the same.

**Where it goes (Chrome, the frame's WGSL with one function stubbed at t = 24 s,
`--eval` + `debugSetExplicitFrameWgsl`):** all of it is `explicit frame (solo)`; without
`triangleGridWalk` (the self-collision search) the step is 7.1 ms instead of 28.2 ms, with
120 proxy shapes in the window. Stubbing `selfApply` and `driveStep` as well changes nothing
(6.8, 7.2 ms). The coupling switched off (`setExplicitKnob('coupling')`) takes 28 to 20–25 ms.

**Why (native GPU World, `--tri-census 1`: the walk's load each second).** With anti-clip
every triangle's box is padded by r + (its own fastest node + the car's fastest node) × h
+ `CLIP_DEPTH`. Detached nodes left behind fall off the mountain's recycled chunks and
fall for ever: the car's fastest node gains 9.8 m/s every second (111 m/s at 17 s, 6 nodes;
228 m/s at 29 s, 14 nodes). The padded boxes then hold more (triangle, candidate) pairs:
5,200 at the start, 7,800 at 17 s, 9,500 at 23 s, 11,200 at 29 s; with each candidate's own
travel in place of the fastest node's they would hold 5,200–6,700 throughout. Past the
pair list's room (`TRI_PAIR_CAP`, 8,192) the frame walks again with its tests in the
walk's lanes (uneven), and the step waits for the slowest. The collision triangles
themselves stay small (longest box side 1–2.3 m, none past the grid's 64 cells).

**Fix applied (lossless): `TRI_PAIR_CAP` 8,192 → 32,768** (196 KB more of `gNext` a
vehicle). The listed pairs' tests and results are the walk's (the same hits, applied in
the specification's order), so only the path changes.

| t (s) | native ms per World step, before → after | Chrome GPU ms, before → after |
|---|---|---|
| 17 | 15.0 → 15.1 | 13.0 → 12.6 |
| 21 | 17.5 (worst 28.3) → 15.9 | 14.3–17 → 17.2 |
| 23 | 26.2 → 16.8 | 26.4 → 17.0 |
| 25 | 28.5 → 17.4 | 28.4 → 18.2 |
| 29 | 33.1 → 20.0 | 23.0–28.0 → 18.4 |

The native trajectories are the same to every figure printed (speed, drop, box, proxies,
pair counts each second). `engine-webgpu explicit_beams_bitwise`: 12 passed.

**Not real time yet:** 17–20 ms a step from 20 s on (40 World steps/s in Chrome). The
rest is the same cause, the fastest node's padding, which the room does not remove:
more pairs to test and more cells for every triangle's walk. Two changes would, both
the owner's decision (neither is lossless):
1. *Each candidate tested against the box padded by its own travel* (the triangle's cells
   by the triangle's travel, a fast candidate entered in the cells its travel covers). It
   changes the specification's box (§36): a contact the rule only found because some
   other node was fast is no longer found. Pairs stay at 5,200–6,700.
2. *The lab retires debris that has left the mountain* (a node of a detached part more
   than some distance below the lowest live chunk), as a kill plane does. Lab scene only.

For 120 Hz (8.3 ms) the frame's base cost is the limit: 7.6 ms in free fall, 11–12.6 ms
in contact, for one car at 34 steps a World step.

Tools: `vehicle-lab-record.mjs --probe --segments` (the GPU split by pass every 5 s),
`--eval file.js` (a script in the page, its `window.__abLog` printed at the end);
`examples/vehicle_lab --tri-census N` (CPU and `mountain-perf`).

## 5. The pair list reused over steps, and anti-clip padded by each node's own travel (2026-11-14)

Two changes to the self-collision search of the explicit frame, each its own commit.

**A. The grid walk's pair list built once and reused (lossless, on by default).** A
building step lists every triangle's (triangle, contacter) pairs against its box padded
by a margin M = 0.05 m more; later steps test that list again while no tracked node (the
contacters and the collision triangles' nodes, the only nodes a box test reads) has moved,
relative to the others, by more than M less the boxes' growth: the spread of the nodes'
displacements since the build, per axis, plus h · (2 · the largest growth of a node's
speed + the growth of the fastest node's speed) plus a rounding allowance must stay
within M. Else the step builds again (and so does each frame's first step). A listed pair
is tested as the walk tests it (a waiting triangle skipped, the walk's cells, buckets and
box on the step's own geometry), then `tryContact`, so the hits are the walk's, applied in
the specification's order. A list never reused before it is invalidated sends the next 8
grid steps to the walk (it never happened on the mountain natively). Knobs:
`set_pair_reuse` (wasm `setExplicitKnob('pairReuse')`, `RUBBLE_PAIR_REUSE=0`),
`set_pair_reuse_margin`, `set_pair_reuse_streak`, `take_pair_reuse_stats`;
`vehicle_lab --pair-reuse 0|1 --pair-margin M --pair-streak N --ab-switch reuse`.
Metal had no threadgroup slot left for another workgroup variable: the state lives past
`gsCount`'s buckets, the step's path in `gsUniformLanes`' bits 1..2.

**B. Anti-clip padded by each contacter's own travel (an explicit setting, on in the lab
only).** `ExplicitBeams::own_travel_reach` (default off): a contacter is tested against
the triangle's box padded by r + (the triangle's fastest node + the contacter's own speed,
at most the vehicle's fastest) · h + `CLIP_DEPTH` instead of the vehicle's fastest node's.
The grid still gathers the cells of the fastest node's box, so a fast contacter is still
found. CPU and GPU use the same arithmetic. Lab: `LabOptions::own_travel_reach` (on), the
page's "own-travel anti-clip boxes" checkbox (live), `vehicle_lab --own-travel 0|1`. It is
not lossless in principle (a contact only the fastest node's padding reached is no longer
found); on the mountain the trajectory was the same to every printed figure.

### Results

Native (`vehicle_lab --measure mountain-perf`, the command of §4 with `--surface 0`), ms
per World step:

| t (s) | before (`--pair-reuse 0 --own-travel 0`) | A (M 0.05) | A + B (the defaults) |
|---|---|---|---|
| 3 (free fall) | 9.57 | 6.30 | 6.41 |
| 9 | 13.25 | 9.59 | 9.63 |
| 13 | 13.83 | 11.66 | 11.91 |
| 17 | 15.33 | 12.56 | 12.64 |
| 21 | 16.24 | 12.16 | 11.86 |
| 25 | 17.83 | 15.04 | 14.49 |
| 29 | 20.37 | 16.47 | 15.49 |

Native parked (`--scene flat`): 5.9 ms, the list reused on 97 % of grid steps.

The list (native, A): reused on 97 % of grid steps in free fall (4,987 pairs a build), 91 %
at 9 s, 66 % at 17 s, 57–62 % from 25 s (10,751–12,645 pairs a build). The late builds are
caused by detached nodes falling off the mountain (the tracked nodes' spread). With B the
pairs a build fall to 5,283 at 9 s, 6,753 at 17 s and 6,415–7,041 from 25 s, but the time
hardly changes: the late cost is no longer the pair tests.

The margin (native, A): M 0.025 m loses late (reuse 19–20 % from 25 s, 17.2 / 19.9 ms at
25 / 29 s); 0.05 and 0.1 m are equal over the run (0.1: 6.62, 9.98, 11.47, 12.45, 12.26,
14.59, 15.92 ms; more pairs a build, 16,124 at 29 s). 0.05 m kept: shorter lists.

Headless Chrome (`vehicle-lab-record.mjs --probe`, GPU ms per World step, the page's
mountain with `surface=0&drag=0`; "before" is A's build with `pairReuse` off and
`ownTravel=0`):

| sim t (s) | before | A + B (defaults) |
|---|---|---|
| 2.8 | 7.9 | 5.0 |
| 5.8 | 10.5 | 7.8 |
| 8.8 | 12.1 | 10.7 |
| 11.8 | 12.7 | 10.1 |
| 14.8 | 12.6 | 9.7 |
| 17.6–17.8 | 13.6 | 10.2 |
| 19.8–20.8 | 15.4 | 11.3 |
| 23–24 | 17.3 | 14.3 |
| 25–26 | 18.4 | 17.6 |
| 27–30 | 18.3–18.6 | 18.8 |
| 31–36 | 20.3–20.9 | 21.1–22.1 |

Parked (`mode=shoot&car=gen&surface=0&drag=0`): 7.0 ms → 4.4–4.9 ms (one earlier run
read 6.2 ms from 30 s on; not reproduced in two later runs, likely another GPU user then).

**Not a gain late in Chrome.** From t ≈ 25 s the list is no faster than the walk in Chrome,
and an in-page A/B toggling `pairReuse` every 2 s (one run) read it 1.5–2.5 ms slower from
26 s on; with the backoff, a later sweep of thresholds 1–4 against the walk read them all
within ±0.5–1 ms of the walk from 25 s on (and 2–3.5 ms faster before). Natively the list
still saves 3.6 ms a frame at 29 s. Why a build costs more in Chrome than the walk it
replaces is not established. Threshold 2 lost 0.5–0.6 ms natively late, so 1 is kept.

### Bit-identity of A

* `engine-webgpu --test explicit_beams_bitwise`: 12 passed (the reuse is on by default:
  the current kernel against the frozen reference, every frame).
* `--test explicit_pair_reuse_gpu` (new): reuse on against off from the same state every
  frame, every state word: the procedural sedan into a wall (31 builds, 48 reused steps),
  with anti-clip (pairs a build 1,635), and dropped tumbling at 4 rad/s (46 builds, 17
  reused): all bit for bit.
* The mountain (native, `--ab 2,8,14,20,26,29 --ab-switch reuse --own-travel 0`): 20 frames
  of 34 steps each way from one snapshot at each time, every state word the same at every
  time; on 5.78 / 9.08 / 12.03 / 11.21 / 14.90 / 15.85 ms a frame, off 9.04 / 12.76 /
  14.10 / 15.50 / 19.18 / 19.50.
* The native mountain runs with the list on and off: the same speed, drop, box and proxies
  every second for 30 s.

### Other tests

`--test showcase_car_gpu` 6 passed (1 ignored, as before), `--test explicit_contact_gpu`
13 passed (with the new `anti_clip_with_own_travel_steps_like_the_cpu`: side memories word
for word, positions bit for bit), `engine-softbody --test own_travel_reach` passed (a node at
300 m/s in no triangle and no contacter: 577 → 3,437 pairs tested with the fastest node's
padding, 577 → 577 with own travel), `engine-world --features stress --test
vehicle_lab_reset` 3 passed. Chrome: the mountain, the parked car and the authored sedan's
90 km/h crash (312 beams broken) with no console or WGSL validation error.

**Not verified:** the `engine-metal` suites (not run); contact groups, the wide and big
modules keep the walk every step (the reuse is the solo frame's only); B on a vehicle
contact (`explicit_contact.rs`, out of scope).

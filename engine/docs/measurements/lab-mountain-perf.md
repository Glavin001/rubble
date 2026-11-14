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

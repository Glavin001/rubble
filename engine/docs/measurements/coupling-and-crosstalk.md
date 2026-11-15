# The rigid coupling's per-step cost, and what crosses CPU↔GPU for vehicles (2026-11-14)

Scope: audit item 3 (`vehicle-perf-first-principles.md` §3.2, the coupling), the host and
readback questions it left unverified (§2.5, §6), and a report (not built) on audit item 5
for the coupling's phases. Everything here is lossless: the same contacts, counts, sums and
state, bit for bit, against the path it replaces, which stays selectable.

## 1. The coupling before

Per 2 kHz step, for every vehicle with proxy shapes (`explicit_beams_rigid.slang`):

| Phase | Work | Barrier after |
|---|---|---|
| `cplReactBy` | a lane a shape: last step's sums → window impulse, terms into workgroup memory | sync |
| `cplIntegrate` | a lane a body: sum its shapes, integrate | sync |
| `cplSpheresBy` | a lane a shape: cull sphere and step frame | sync |
| `cplCountBy` | a lane a node: **every shape** (sphere test, face test), count on its body | sync |
| node phase (`cplContacts`) | each node again against its shape range (native) or **every shape** (browsers: node ranges off) | (the node phase's own) |

So 4 syncs (8 barriers) a step, and O(nodes × shapes) a step in the count, and in Chrome
in the node phase too.

**Contention warning.** Every GPU number in this document was taken on 2026-11-14 evening
while the Mac's GPU read 90–100 % busy outside the lock (`ioreg` before several runs; a
runaway dispatch from another stream is suspected). Absolute ms are about 40 % above
earlier documents (fleet apart, 1 car: 15 ms here against 9 ms in `lab-mountain-perf.md`
§7), whether the list is on or off. Only **interleaved A/Bs from one snapshot** (the two
paths a few seconds apart) are used for conclusions, and even those are only good to about
±0.3 ms. Repeat the timing on an idle GPU.

**What it cost, measured** (native, M3 Max, the ablation kernel with a warm-up mask first,
20 frames of 34 steps from one snapshot each way, mountain query of the brief;
`out/cpl/m3-fold.txt`; the ablated variants run the old four-phase structure minus the
work, so these are the cost of the coupling's work, not of its barriers):

| Scene (native) | Coupling per frame | How |
|---|---|---|
| Mountain t 20 s, 11 shapes | ≈0.6 ms passes, ≈0.7 ms with the node contacts | ablation kernel, warm: 12.30 all, 11.71 without the per-step passes (57344), 11.58 without them and the node contacts (122880) |
| Mountain t 30 s, 15 shapes | ≈0.36 ms passes, ≈0.44 ms with contacts | 16.31 / 15.95 / 15.87 |
| Mountain t 35–41 s, 5–18 shapes | ≈0.15–0.6 ms | 14.64 / 13.91 (122880) at 35; 16.49 / 15.86 at 41 |

**The audit's 1.8–5.7 ms came mostly from a measurement artefact.** The `--ab-ablate`
harness (vehicle_lab, and so `lab-mountain-perf.md` §7) times each mask for 20 frames right
after switching; the first mask after the switch to the ablation kernel pays that
pipeline's first use. Measured twice in a row from the same snapshot, the same mask
(2147483648, nothing skipped): 14.53 then 12.30 ms at t 20 s, 17.58 then 16.31 at t 30 s
(`out/cpl/m3-fold.txt`). §7's "everything 19.6" was such a first mask, so its "coupling
5.7 ms" (19.6 − 13.9) holds ≈1.3–2.2 ms of pipeline warm-up, and was taken before the
window per piece cut the shapes from 110–135 to 14–73. Any further `--ab-ablate` list
should start with a warm-up mask (the bench, `EV_AB_MASKS`, already runs a frame first).

## 2. The candidate list (built)

A Verlet list with a checked skin, per node, built on the GPU inside the frame
(`cplCountListed`; CPU reference and proof test `engine_softbody::explicit_rigid::candidate_list`):

- **Build** (a frame's first step, or a step the node's check fails): the node scans
  every shape exactly as the old count does, and lists the shapes within `r + m` of their
  cull spheres (`r` the grown cull radius, `m` its skin), in shape order (16 at most;
  more: no list, every shape every step, as before). It keeps `x0` and `limit = m − E`.
- **Check** (every later step): the list holds while `|x − x0| + E < limit`. `E` is the
  frame's running maximum, over every shape of the vehicle, of the shape centre's distance
  from its centre at the frame's first step (one atomic max a shape a step, after a racy
  read so most shapes skip it).
- **Proof.** For an unlisted shape k at the build step b, `|x_b − p_k(b)| ≥ r_k(b) + m`.
  At step s, by the triangle inequality, `|x_s − p_k(s)| ≥ r_k(b) + m − |x_s − x_b| −
  |p_k(s) − p_k(b)|`, and `|p_k(s) − p_k(b)| ≤ E(s) + E(b)`. The check gives
  `|x_s − x_b| + E(s) + E(b) < m`, so `|x_s − p_k(s)| > r_k(b)`, and `r` is grown past the
  shape's true bounding radius by at least 1 cm and 0.1 % (`cplSpheresBy`), far beyond the
  rounding of these float sums at any coordinate below kilometres. So `cplFace` rejects the
  shape: no count, no force, no sum. Listed shapes are tested exactly as before, in shape
  order. Nothing about motion is assumed: an impact that changes a node's speed mid-frame, a
  body thrown through the car, a shape that enters reach mid-frame only make the check fail
  sooner, and the node rebuilds its own list (no barrier, no other lane involved).
- **The skin** only sets the rebuild rate: `1.25 · h · steps left · (node speed + the
  shapes' fastest point speed) + 2 cm`, at most 4 m.
- **The node phase** uses the same list (the count pass left it valid for this step), so in
  browsers too a node tests only its listed shapes.

Selectable: `ExplicitBeamsGpu::set_coupling_candidate_list`, `RUBBLE_COUPLING_LIST=0`,
coupling header bit 128 (off = every node against every shape, the old code untouched).

**Tests** (all bit for bit against the list off, every state word, every proxy record and
sum): `engine-world tests/explicit_vehicles.rs`
`the_coupling_candidate_list_is_bit_exact` (four cars into a house at 120 Hz, 20–79 chunk
shapes a car, contact groups) and `the_coupling_candidate_list_meets_fast_and_thin_bodies`
(a car at 30 m/s into a 2 cm free plate and a 2 cm static wall while a ball crosses it at
150 m/s from out of reach), each also with two test skins: without the shapes' speed (only
the check can catch the ball) and at the 4 m ceiling (nodes keep lists through hard
decelerations). CPU: `candidate_list` tests, 60 adversarial seeds (nodes jumping to 300 m/s
mid-frame, bodies at 200 m/s, 1 cm shapes, shapes starting 15 m out, positions 2 km from
the origin, skins at 0, the floor and the ceiling): no node is ever inside an unlisted
shape's true sphere.

## 3. The folded step (built): 4 syncs → 2

`cplStepFolded`, the solo frame's usual case:

- **B:** each shape's lane integrates its own body from the step's sums (read, not
  cleared; the arithmetic and order of `cplReactBy` + `cplIntegrateBodySplit`, so the same
  bits) and writes its sphere and step frame to gNext, plus the list maxima; each body's
  lane computes the same integration into registers and clears its contact count.
- **C:** the bodies' lanes write their records; the shapes' lanes do `cplReactBy`'s writes
  (window impulse and moment, sums cleared) and copy their spheres to workgroup memory; the
  count reads the spheres from gNext (their workgroup copies are being written).

Nothing in B writes what B reads; nothing in C reads what C writes; the node phase after
C's barrier sees the same records, sums, counts and spheres as after the four phases. A
first step is 2 syncs (was 3). Taken when the list is on, every sphere fits in workgroup
memory and bodies ≤ lanes; contact groups keep the four phases (their members' paths must
agree on barriers). Off: `set_coupling_fold`, `RUBBLE_COUPLING_FOLD=0`, header bit 1024.
Tested in the same two A/B tests (list on with the step unfolded, and folded, both equal to
the old path).

## 4. Results

All bitwise checks: identical (state words, proxy records and sums) at every comparison.

**Mountain, one car** (`vehicle_lab --ab T --ab-switch coupling|fold`, 20 frames each way
from one snapshot, ms a frame; contended GPU):

| t (s) | shapes | old (no list, 4 phases) | list + fold | bitwise | fold off / on (list on) |
|---|---|---|---|---|---|
| 20 | 11 | 17.26 | 17.24 | yes | 17.38 / 17.35 |
| 30 | 15 | 21.44 | 21.56 | yes | 22.53 / 21.97 |
| 35 | 5 | 27.34 | 27.66 | yes | 22.53 / 23.43 |
| 38 | 6 | 29.43 | 30.01 | yes | 22.95 / 22.98 |
| 41 | 18 | 29.81 | 30.04 | yes | 23.04 / 23.02 |

The coupling's whole cost at one car on the mountain is 0.4–0.7 ms a frame (§1), under the
1 ms target before and after. With 5–18 shapes the list saves nothing measurable here; late
it may cost up to ≈0.3–0.6 ms (always on-then-off order in this harness, so an order effect
is not ruled out).

**Crash town, explicit frame alone** (`explicit_vehicles_bench`, `EV_AB_AT`,
`EV_AB_COUPLING=1`: old / list / list + fold, three interleaved rounds, median ms a frame):

| cars | step 100 | step 150 |
|---|---|---|
| 1 (20–79 shapes) | 15.33 / 15.39 / 15.35 | 14.86 / 14.90 / 14.84 |
| 4 | 27.53 / 28.48 / 27.65 | 26.58 / 26.63 / 26.50 |
| 8 | 54.55 / 54.51 / 54.49 | 54.39 / 53.99 / 54.33 |

**Fleets, World GPU span median over 180 steps** (`explicit_vehicles_bench`, list on via
the default, off via `RUBBLE_COUPLING_LIST=0`; `out/cpl/measure1.txt`, `m2.txt`, `m4.txt`):

| | 1 | 4 | 8 | 32 |
|---|---|---|---|---|
| apart, on | 14.99 | – | 15.96 | 17.53 |
| apart, off | 14.78 | – | 15.93 | 17.46 |
| town, on (two rounds) | 16.92, 17.49 | 28.20, 28.24 | 55.37, 54.62 | 103.02 |
| town, off (two rounds) | 17.35, 17.35 | 30.34, 30.10 | 55.07, 53.81 | 104.22 (a run under CPU load) |

**Reading.** Natively the coupling is no longer where the time goes: the list and the fold
are neutral within noise, except the 4-car town, where the list is about 2 ms (7 %) faster in
two rounds. The list's value is where shapes are many (houses, rubble, debris windows) and
in browsers, whose node phase scans every shape (node ranges off there). **Chrome is not
measured**: both probes timed out loading the page on the saturated GPU (§7). Both paths
stay selectable; whether the list stays on by default for few-shape cars needs an
uncontended A/B, native and in Chrome.

## 5. CPU↔GPU crosstalk for vehicles

Inventory from the code (World: `engine-world/src/explicit_vehicles.rs`, `lib.rs`,
`readback.rs`; page: `vehicle-lab.html`, `engine-world-wasm/src/lib.rs`, `stability.rs`),
per World step (1/60 s, 33–34 explicit steps):

| # | What | Direction, bytes | How often | Gates the next step? | Grows with cars? |
|---|---|---|---|---|---|
| W1 | coupling `Params` | up, ≈144 B | every step | no | no |
| W2 | explicit frame parameters | up, 512 B × (steps + 1) = 17.9 KB → **0.5 KB** (fixed, `4238fd5a`) | every step | no | no |
| W3 | sleep wakes (sleep on) | up, 16 B a throttled car, one write each → **one write a run of cars** (fixed, `4c25864a`) | every step | no | was O(cars) writes |
| W4 | 3 bind groups created, 2 buffer clears | host objects / GPU only | every step | no | no |
| W5 | the World's readback slot: counters, event ring 16.4 KB, moved records `bodies × 32 B` (copied on the GPU for every body; only the moved ones are viewed), explicit loop-cap 4 B, coupling status 4 B, sleep mask 4·⌈cars/32⌉ B | down, ≈16.5 KB + 32 B × bodies | every step, one `mapAsync` | **yes** (below) | sleep mask only |
| W6 | coupling status | down, 16 B | every 8 windows | no | no |
| W7 | frame status (contact counters, loop cap) | down, 20 B | every 16 frames | no | no |
| P1 | lab monitor: nodes N × 48 B, origin, **coupling records 114.7 KB** (the whole region: 1 + 256·5 + 256·7 + 4096 records), beams B × 32 B every 250 ms | down | every step while the lab is busy, else every 4th; skipped when its 3 buffers are out | no | car 0 only |
| P2 | lab host work on P1 (rigid fit, penetration, union-finds, piece labels) | host | as P1 | no | car 0 only |
| P3 | `VehicleHealth.check` | down, 16 B + N × 48 B for each of the first min(cars, 8), a new staging buffer each time | every drawn frame | no | **yes, to 8 cars** |
| P4 | renderer uniforms | up, ≈1.4 KB | every drawn frame | no | no |

Drawing reads the solver's buffers directly (skin, normals, LOD as compute passes,
indirect draws); only the camera comes from P1's last reading.

**What gates stepping** (`WebWorld.canStep`): the World's readback ring has a free slot,
the device is alive, and fewer than `EXPLICIT_STEPS_IN_FLIGHT` = 2 World readbacks (W5) are
out. So step N + 2 waits for step N's W5 map to resolve and a `poll()`. W5 is the rigid
World's own output channel (moved bodies, events, fracture states) plus the explicit
runaway guards; the 2-in-flight rule is deliberate pacing (`docs/stability-stop.md`): a GPU
that falls behind runs the simulation slower instead of queueing work. Natively (the World
API), `step()` waits on the oldest readback (`wait_oldest`) when the ring is full and the
step needs its readback (`Backpressure::Block`, ray batches, contact events, **fracture**).

**Measured.** The per-step traffic above is from the code; `tools/explicit-2khz/xtalk.js`
(counts every `writeBuffer`, readback copy and `mapAsync`, with its resolve time, per
second in the running page) is written and was not run (§7).

**Fixed here:**
- W2: the frame uploads only the parameter copies its commands read (one for per-vehicle
  frames, all of them for the wide passes and split frames). That is 17.9 → 0.5 KB a World
  step (`4238fd5a`).
- W3: the sleep wakes are one write for each run of consecutive cars, not one write a car.
  A fleet with every car under throttle now costs one write a step (`4c25864a`).

**Defects at hundreds of cars, or things that gate stepping. Reported, not changed:**
- **canStep (W5).** Stepping waits on the rigid World's readback map (2 in flight). A
  pacing signal that reads nothing would decouple stepping from readback latency. One
  option is `queue.onSubmittedWorkDone` natively and in browsers, counting steps
  submitted against steps completed, with W5 made droppable when its ring is full. The
  ring has 8 slots in the lab; W5 is mapped every step. That is a World design change
  (`stability-stop.md`).
- **Native `wait_oldest`.** It blocks the host on a readback for every step with fracture
  (the town) or contact events once the ring is full. Fracture states are adopted on the
  host ("a split must be adopted in the step it happened"), which is CPU in the runtime
  path. The fix is the GPU-side split adoption the rigid stream owns.
- **W5 copies the event ring (16.4 KB) and every body's moved record every step**, and the
  wasm copies the ring into JS memory every step, even though the vehicle lab uses no
  events. The fix is to read the count first or make the events opt-in.
- **P3 `VehicleHealth.check`.** It reads the first 8 cars' whole node state every drawn
  frame (8 × 26 KB for the sedan) through a new staging buffer each time. A GPU reduction
  to one flag word a car would make it O(1) bytes.
- **P1.** The lab reads the whole coupling region (114.7 KB) with each monitor reading,
  every step while busy. Only the used records are needed (header counts), and only for
  the shot/plate readouts; the camera needs only the origin and a centre.
- **W6/W7.** These are small status reads that duplicate W5's per-step loop-cap and status
  words, but W5 can be dropped when its ring is full, so they are kept as backups.

## 6. Audit item 5 for the coupling's phases (report, not built)

Syncs (`sync()` = a device barrier then a group barrier) a step that the coupling and its
neighbours cost, the solo coupled car with raycast wheels:

| Site | Before | Now | Possible next (identical results) |
|---|---|---|---|
| coupling (`coupleStep`) | 4 (3 on a frame's first step) | **2** (B, C) | 1: run B in the previous step's last phase (beside `slideNode`, before the end-of-step barrier) once the surface pass's sums are complete there; B reads only the step's sums and body records and writes gNext, which nothing in that phase reads |
| `driveStep` (`driveWheelsSplit`: rays, bars, tyre, then its barrier) | 3 | 3 | 0 extra: rays in B, bars in C, tyre and end before the node phase's barrier would need a third phase; folding rays and bars into B and C leaves the coupling + drive at 3 syncs instead of 5. The wheels read static boxes and the ground, never the proxies (`drvRays`), and write node accumulators the coupling does not touch |
| `coupleEnd` (frame end) | 2 | 2 | 1, by the same fold as B (once a frame: negligible) |

**Measured barrier cost** (`EB_ABLATE` 64: one barrier a sync instead of two; 128: none;
time only, results meaningless; mountain, ablation kernel, warm-up mask first):

Not measured. The run that included masks 64 and 128 (17:40, `out/cpl/base.txt`) stalled
at its first `--ab` and was killed at the 600 s limit. The cause, 128 (no barriers) or the
contention, is not established. One mask-64 sample under contention, after a warm-up mask:
t 20 s 12.30 → 12.33 ms, t 35 s 14.64 → 14.63 ms, t 41 s 16.49 → 16.56 ms. Halving the
barriers changed nothing measurable, so the device-memory half of a sync costs ≈0 here.
The 128 (no barriers) run was cancelled when the GPU was paused.

What this says for item 5: barriers alone are not where the coupling's time goes on this
GPU. The fold's measured effect (§4) is within noise, consistent with that. The coupling's
remaining cost is the dependent-load latency of each phase (a body's record, its shapes'
records and sums, the node's record) and its share of the node phase; subgroup reductions
for `E` and the shape speed (one atomic a subgroup instead of a racy read a shape) and
loading a node's list beside its position (one dependent load fewer) are the lossless
next steps, each small.

## 7. Not verified / left

- **Chrome:** the 46 s mountain probe was not completed. Four attempts: two failed the
  WGSL validation of `f75c65de`/`d849f8c3`, fixed in `d849f8c3` and `5b5d942a`, and the
  last two timed out at page load on the saturated GPU. Not measured: the list's effect
  where it should matter most (node ranges off), and the page's traffic counters
  (`xtalk.js`). web-dist was rebuilt
  (`build-web.sh`) at 19:25 and its World wasm rebuilt twice by hand (wasm-pack, copied into
  `web-dist/generated/world-wasm`) for the two uniformity fixes; the Asyncify (Safari) world
  wasm in web-dist is older.
- **After `5b5d942a`** (the fold's uniform decision, final form): the GPU suites and the
  bitwise A/Bs were not re-run. Everything listed as passing ran on `f75c65de` (list +
  fold, per-step decision): explicit_beams_bitwise 12, explicit_contact_gpu 13,
  showcase_car_gpu 6 (+1 ignored), explicit_loop_cap 5, vv_grid_gpu 2, explicit_sleep_gpu 3;
  engine-world explicit_vehicles 10, explicit_sleep, explicit_drive_world,
  explicit_loop_budget, explicit_vehicle_broadphase 3, vehicle_lab_piece_windows,
  vehicle_lab_pose_time 2, vehicle_lab_reset 3. That covers mountain bitwise at 5 times
  for the list and for the fold, and the town and fast/thin tests. `d849f8c3` produced
  non-finite nodes on the mountain natively (a `workgroupUniformLoad` on lane 0);
  `5b5d942a` fixes it by construction (the WGSL now only writes the word on lane 0). Its
  native run was queued and cancelled by the GPU pause.
- **TDD order:** the list's failing test (`b93c14c1`) was committed before its kernel. It
  was not run red on the GPU: its "lists in use" assertion cannot pass without the kernel.
  The fold's test variants came in the same commit as the fold.
- **Not built:** item 5's further folds (§6); the crosstalk defects listed in §5; the
  fold for contact groups.
- **An uncontended timing** of everything in §4, and Chrome.

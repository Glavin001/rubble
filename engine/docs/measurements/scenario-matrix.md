# The scenario matrix: one command that gates a perf change across use cases

One World holds rigid bodies, soft structures and soft vehicles. A perf change must improve
that whole picture, never one case at the cost of another. The matrix replays a dozen saved
states of different use cases with the switch under test on and off, from the same state
in the same build, and says PASS only if every row is bit for bit and none is slower.

```sh
cd engine
tools/scenario-matrix/run.sh perf=8388608      # any vehicle_lab --ab-switch value
tools/scenario-matrix/run.sh --baseline        # each row's ms a frame against its target
tools/scenario-matrix/run.sh --make            # (re)save the states (once; after a layout change)
tools/scenario-matrix/run.sh --world           # the World's own cost a step on the scenario rows
PROFILE_SPLIT=1 tools/scenario-matrix/run.sh perf=MASK pile8   # with the frame by dispatch kind
```

`run.sh` builds `vehicle_lab` with `--profile fast` (as `fast-check.sh`), then runs the
whole matrix under one hold of the GPU lock (`scripts/with-gpu-lock.py`). Rows can be
picked by name (`run.sh perf=MASK pile wall`).

## Rows

| row | soft bodies | what | state |
|---|---|---|---|
| drive | 1 | one car driving on flat ground, 60 km/h, throttle on | `scenes/drive.json`, t 1.0 |
| tumble | 1 | one car in free flight, tumbling down the mountain | the mountain, 1 car, t 2.5 |
| wall | 1 | one car into a rigid wall at 56 km/h, mid-crush | `scenes/wall.json` (crash preset), t 0.1 |
| pile2 | 2 | 2 cars, first collisions on the mountain | the mountain, 2 cars, t 3.7 |
| pile8 | 8 | 8 cars piling up on the mountain | the mountain, 8 cars, t 8 |
| pile32 | 32 | 32 cars piling up on the mountain | the mountain, 32 cars, t 4 |
| headon | 2 | 2 cars head-on, 50 km/h each, mid-crush | `scenes/headon.json`, t 0.7 |
| side | 2 | a car at 50 km/h into a parked car's side | `scenes/side.json`, t 1.0 |
| bricks | 1 | a car at 40 km/h into a wall of 60 rigid bricks | `scenes/bricks.json`, t 0.7 |
| rocks | 1 | 4 rigid rocks (0.5 to 1 t) falling on a parked car | `scenes/rocks.json`, t 1.0 |
| barrier | 2 | a car at 50 km/h into a node-beam steel barrier | `scenes/barrier.json` + `barrier.truck`, t 0.65 |
| truss | 1 | a node-beam bridge truss sagging under 6 t, no vehicle | `--car scenes/truss.truck`, t 2.0 |

The car is the page's generated sedan (`CAR`, as `fast-check.sh`: 548 nodes, 4,251 beams).
The mountain rows are the lab's multi-car run at the fast-check fixtures' moments.

### Node-beam structures (rows barrier and truss)

The explicit solver and its loaders take a body with no wheels as they are:
`ExplicitBeams::from_truck` of a `.truck` with only `nodes`, `beams`, `cab` and `fixes`
spawns it without tyres or a drive, and the World takes a fleet of different bodies. No
solver or loader change was needed. The lab gained two options (lab and harness only; with
neither set, the World is built as before):

- `LabOptions::structures` (`structures: [{ truck | truckFile, x, z, angle, lift }]`): node-beam
  bodies placed after the cars, its anchored nodes still.
- `LabOptions::others` (`others: [{ x, z, angle, kmh }]`): where the flat arena's other cars
  start (the head-on row).

`make_structures.py` writes both `.truck` files (committed):

- `barrier.truck`: a 6 x 1.2 x 0.4 m lattice box truss, 104 nodes, every edge and face
  diagonal a beam (spring 2 MN/m, deform 40 kN, break 400 kN), its bottom row anchored,
  both faces covered with collision triangles. The car at 50 km/h bends it.
- `truss.truck`: a 10 m Pratt bridge truss on two 2 m piers, two planes, two 3 t loads at
  mid-span, the piers' feet and deck ends anchored. It sags and yields under its load.

## How the states are made

`run.sh --make` drives each row's scene (`vehicle_lab --measure mountain-perf`, with
`--scenario scenes/ROW.json` for the flat rows: `restart` start or crash preset, `events`,
`structures`, `others`) to its moment and saves the explicit solver's state there
(`--save-fixtures`) as `out/scenario-matrix/ROW.fx` (gitignored; 0.3 to 107 MB, pile32 the
largest). About 2.5 minutes the first time after a build (pipeline compiles), seconds per
row after. Re-make after any change of the buffers' layout (a replay of an old layout
fails loudly, `snapshot of another layout`).

What a replay covers: the explicit vehicles' frame (34 steps of 0.5 ms), including the
coupling's proxies of the rigid bodies as the World last wrote them (the bricks, the rocks,
the wall), vehicle–vehicle contact and the structures. The rigid solver's own step is not
replayed (the World has no snapshot of it, by design: destruction is forward-only), so a
change to the rigid solver is not gated by this matrix.

## The table and the PASS rule

Each row: `vehicle_lab --from-fixture ROW.fx --ab-switch SWITCH --ab-frames 10` with
`AB_WARM_MS=150` (the GPU run from the state for 150 ms first: a small row's first half
otherwise ran at the idle clock, up to +80 % on the truss) and `AB_ROUNDS=3` (on, off, on,
off, on, off, each from the state). Frame k is the same work in every round, so its
fastest round is its time without the noise; the row's ms a frame is the mean of those
(a crash's frames differ, so the mix is kept). Every state word is compared as
`fast-check.sh` does (the last round's).

PASS only if every row is bit for bit and no row's on half is slower than its off half by
more than **THRESHOLD % and NOISE_MS ms** (both must be exceeded; defaults below).
A `faster` mark is information, not a requirement.

## Noise: how the threshold was calibrated

Measured by running the matrix with `perf=0`: a mask of no bits, so both halves run the
same code (same vs same). Any delta is noise.

- **Before the warm-up and the rounds** (one on half then one off half, the median of 10
  frames each): most rows within ±2 %, but the small rows' first half was slow whenever the
  GPU had idled: the truss +12 to +81 %, the tumble +14 %, the wall −8 % (the GPU's clock
  ramping during the first half's 20 ms of frames). Unusable as a gate.
- **With `AB_WARM_MS=150` and `AB_ROUNDS=3`, each frame's best round** (8 matrices, 96 row
  samples over two sessions, other agents' builds running on the CPU): median |delta|
  0.4 %; 90 % of rows within ±3 %; 3 of 96 past +5 % (truss +7.3 and +7.6 %, i.e.
  0.12 ms, tumble +10.2 %), 3 past −5 %. The outliers were alone: the same row in the next
  matrix was within ±2 %.

So: **THRESHOLD 5 %, NOISE_MS 0.05 ms, and a row past both runs twice more (RETRIES 2), the
median of the three deciding.** A real slowdown repeats; at about 3 % of single runs past
+5 %, a false FAIL needs two of three, about 0.3 % a row, a few % a matrix. A change that
must be judged finer than 5 % on one row is a rung-4 measurement (whole runs, an idle GPU).
`THRESHOLD`, `NOISE_MS`, `RETRIES`, `ROUNDS`, `FRAMES` and `WARM_MS` are environment
overrides.

## Baseline (main at 366bc068, Apple M-series Metal, native, `--profile fast`)

`run.sh --baseline` (both halves the same path; the faster is the row's time). ms per
frame = 34 explicit steps (1/60 s of simulated time), GPU wall time with the host's
submission, as the A/B times it.

| row | soft bodies | ms/frame | target | |
|---|---|---|---|---|
| drive | 1 | 3.22 | 1.00 | over |
| tumble | 1 | 3.20 | 1.00 | over |
| wall | 1 | 4.11 | 1.00 | over |
| pile2 | 2 | 8.19 | ~1.29 | over |
| pile8 | 8 | 14.43 | 3.00 | over |
| pile32 | 32 | 56.39 | ~9.86 | over |
| headon | 2 | 15.41 | ~1.29 | over |
| side | 2 | 9.30 | ~1.29 | over |
| bricks | 1 | 3.40 | 1.00 | over |
| rocks | 1 | 3.81 | 1.00 | over |
| barrier | 2 | 7.27 | ~1.29 | over |
| truss | 1 | 1.56 | 1.00 | over |

Read across the rows: two cars in contact cost 2.5 to 5 times one car (pile2 8.2, headon
15.4, side 9.3 against drive 3.2): the contact between vehicles, not the second body, is the
cost. A node-beam barrier of 104 nodes beside a car costs 4 ms more than the car alone.
Rigid bricks and rocks add little to the car's frame (3.4 and 3.8 against 3.2).

Targets: the owner's, one soft body 1 ms a frame (worst) and eight bodies in contact 3 ms;
between and past them a straight line through both (marked ~: an extrapolation, not a
stated target). Every row is over today: the scoreboard.

## The known switch: `perf=8388608` (PERF_FLEET_HITLAT, on by default, bit for bit)

`run.sh perf=8388608`, twice: every row bit for bit, PASS. The switch is worth 46 % on
pile32 (56.0 against 104.3 ms) and 13.6 % on the head-on crash (15.4 against 17.8), and
nothing on the other rows (all within ±4 %; side −3.8 % once, within the noise):

```
row      bit4bit    on ms   off ms    delta         run s  what
drive        yes    3.206    3.188    +0.6%           0.8  one car driving on flat ground, 60 km/h, throttle on
tumble       yes    3.176    3.178    -0.1%           0.8  one car in free flight, tumbling down the mountain
wall         yes    4.062    4.050    +0.3%           0.8  one car into a rigid wall at 56 km/h, mid-crush
pile2        yes    8.179    8.122    +0.7%           1.3  2 cars, first collisions on the mountain
pile8        yes   14.448   14.440    +0.1%           2.9  8 cars piling up on the mountain
pile32       yes   56.021  104.339   -46.3%  faster   6.7  32 cars piling up on the mountain
headon       yes   15.418   17.847   -13.6%  faster   1.9  2 cars head-on, 50 km/h each, mid-crush
side         yes    9.295    9.661    -3.8%           1.5  a car at 50 km/h into a parked car's side
bricks       yes    3.460    3.445    +0.4%           2.1  a car at 40 km/h into a wall of 60 rigid bricks
rocks        yes    3.835    3.822    +0.3%           0.8  4 rigid rocks (0.5 to 1 t) falling on a parked car
barrier      yes    7.298    7.212    +1.2%           1.3  a car at 50 km/h into a node-beam steel barrier
truss        yes    1.553    1.550    +0.2%           0.6  a node-beam bridge truss sagging under 6 t, no vehicle
verdict: PASS; 12 rows in 21.5 s
```

## Time

Measured wall time of the whole matrix under the lock, after the build (`run.sh`'s cargo
step is seconds after an edit, as `fast-check.sh`):

- **Warm: 20 to 30 s** (21.5, 24.7, 26.9 s in the last session; 16 to 21 s in a quieter one).
  Per row 0.6 to 0.8 s for a one-car row (process start, the World's build, the state's
  load: about 0.5 s of it), pile8 3 s, pile32 5 to 7 s (60 frames of 56 to 104 ms: 3.4 to
  6.3 s of frames alone), bricks 1.5 to 4 s, barrier 1.3 to 3 s.
- **Cold: 94 to 155 s**, the first run after the Metal shader cache has lost the kernels
  (after a kernel edit; also after other jobs on the machine compiled other kernels). The
  pipelines compile in the first row that uses them: drive 27 s (the solo frame), pile2
  60 s (vehicle contact), bricks 19 to 27 s (the coupling's rigid proxies).

What would bring it to 30 s every time, and to well under it warm:

1. **Cold:** compile every pipeline the matrix uses up front, in parallel (the engine has
   `gpu_adapter --compile --jobs N` for the page's modules, kernel-compile-time.md), or
   warm the shader cache right after the build instead of in the first row of each kind.
   The compiles are 60 to 110 s one after another today.
2. **One process for all rows** (`vehicle_lab --matrix`): the device, pipelines and lab
   built once, each row's World rebuilt in it: about 0.5 s a row, 6 s a matrix.
3. **pile32** at `FRAMES=5` (the halves' frames 30 instead of 60): 2.5 to 3 s less, with a
   per-row frame count so the other rows keep 10.

Not done here: each is a change to the harness's process model, measured first.

## Findings on the way

- **A node-beam structure in the World does not make the World's step slow** (checked
  after this change; it was first reported here as a 50 ms `cpl_share` pass and a 22 ms
  median step against 5 ms for the car alone). By wall clock the barrier's World step is
  the explicit frame plus what every row pays: see "The World's own cost" below. The
  numbers that looked slow are GPU timestamps, which do not hold on Metal here (known bug
  C4): `cpl_share` (a loop of a few records) reads 0.06 to 7.4 ms from one second to the
  next on the car alone as well; one barrier step's span read 70 ms while its wall time was
  8 ms; one car-alone step's span read 2,536 ms against an 11 ms wall. The first two steps
  of every run are the pipelines' compiles (2.2 s and 0.43 s, car alone or not), and a cold
  shader cache compiles vehicle contact's pipelines when the car first meets the barrier.
  What the barrier does cost is in the explicit frame, which the matrix already times:
  with two bodies in vehicle contact the frame runs a workgroup per body instead of the
  wide passes (`is_wide` false; the wide passes do not run vehicle contact), 7.1 ms
  against the car alone's 3.2 ms.

## The World's own cost (`run.sh --world`)

The A/B rows replay the explicit frame alone, so a cost the World adds around it (the
rigid step, the coupling's passes, the host's encoding and poll) is invisible to them.
`run.sh --world` runs each scenario row's scene in the World up to the row's moment, then
`vehicle_lab --end-world 20`: 20 World steps timed one by one (the step and its poll, wall
clock), the vehicles put back to the state before them, and the same explicit steps (33,
33, 34, ...) as 20 frames of the vehicles' solver alone; 3 rounds from that same vehicle
state, every step pooled. The difference of the medians is what the World adds a step.
A row fails past `WORLD_MS` (default 2.5 ms). No saved states are needed; about 30 s.
(The restore puts back the vehicles, not the rigid bodies: a time measurement only.)

Two runs (`--profile fast`, Apple M-series Metal, 2026-11-18; step and frame ms of the first):

```text
row       step ms  frame ms  World ms        what
drive       3.859     3.249    +0.610 / +0.526  one car driving on flat ground
wall        4.775     4.155    +0.620 / +0.518  one car into a rigid wall
headon     16.081    15.171    +0.910 / +0.636  2 cars head-on
side        7.922     7.232    +0.690 / +0.726  a car into a parked car's side
bricks      5.038     3.389    +1.649 / +1.606  a car into 60 rigid bricks
rocks       4.827     3.832    +0.994 / +1.206  4 rigid rocks on a parked car
barrier     7.716     7.124    +0.592 / +0.594  a car into a node-beam barrier
truss       2.065     1.585    +0.480 / +0.448  a node-beam truss, no vehicle
```

The World adds 0.5 to 0.9 ms a step on every row, and more where the rigid step has real
bodies (bricks 1.6 ms, rocks 1.0 to 1.2 ms). Runs agree within about 0.2 ms. The barrier
adds no more than the car alone.

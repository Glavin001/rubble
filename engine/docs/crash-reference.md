# Crash reference suite

Ten real crash tests, each shown beside BeamNG in one video
(<https://www.youtube.com/watch?v=Lqx2KKWI8aM>), staged in Rubble on the explicit 2 kHz
node-beam solver with generated vehicles, and judged by acceptance criteria derived from
first principles; and four US government agency test configurations (S11–S14, section 4)
compared curve against curve with NHTSA's measured data. This page holds the scenarios, how
a crash is quantified, each scenario's criteria and what currently fails.

## How to run one

A scenario is defined once, in `crates/engine-softbody/src/crash_reference/` (one file a
scenario, one line in `all()`), and builds a fleet of `ExplicitBeams`, which every executor
takes. The same definition, sampling, metrics and criteria run everywhere:

| Where | Command |
|---|---|
| CPU reference | `cargo run --release -p engine-softbody-lab --example crash_ref -- S02 [--runs 8 \| --run K] [--contact discrete\|edge\|continuous] [--json] [--out DIR]` |
| CUDA on Modal | `tools/modal/gpu_run.sh --build modal [--fanout-tests] -- -p engine-cuda --test crash_reference` (`CRASH_CONTACT`, `CRASH_RUNS`, `CRASH_RUN_FROM`, `CRASH_SKIP`, `CRASH_REPORTS=DIR`, `CRASH_REPORT_LINES=1`) |
| The whole suite, both executors | `python3 tools/crash-reference/sweep.py --cpu [--per-run] [--ids ..] [--contacts ..]`, `--cuda [--build local] [--ids S06 --run-from 0 --runs 3] [--skip S06]`, `--summarize` (to `out/crash-reference/sweep/`: each run's report, `summary-cpu.md`, `summary-cuda.md`, `compare.md`, `known-cpu.md`; "Results" below) |
| Browser (WebGPU crash lab) | `softbody.html?solver=explicit&crashRef=S02[&run=0][&contact=edge][&cam=side]` (scene picker: "Reference crashes") |
| Browser, stills and report | from `crates/engine-webgpu`: `python3 ../../scripts/with-gpu-lock.py --limit 300 -- node crash-ref-record.mjs --crashRef S02 [--cam side] [--contact edge] [--video] [--all-cams]` (to `out/crash-reference/<id>-<contact>-run<k>/`; `--all-cams`: a still from every camera at every moment) |
| Page against CPU | `python3 tools/crash-reference/compare.py CPU.json report.json [--all]` (the asserted criteria side by side) |
| Curves against NHTSA's | `crash_ref -- S11 --json > s11.jsonl`, then `python3 -I tools/crash-reference/overlay.py docs/crash-reference/nhtsa out/crash-reference/overlays s11.jsonl` (section 4) |

`--json` reports carry the run's time histories (`series`, section 4.1) beside the metrics.

The crash lab's panels (`crash-lab-ui.mjs`, one responsibility each): **Scene** (Shift+S:
the scene picker, a reference crash's contact and run), **View** (Shift+V: the scenario's
cameras with the lab's, drawing options, grab), **Playback** (Shift+P: time) and **Results**
(Shift+R: the criteria, collapsed to a one-line summary; values arrive when the run ends).
H (or the bar's "focus") hides them all: the recorders' `clean=1` frame. Their layout at
800 to 1920 px is checked by `crash-lab-ui-check.mjs --case S05|S02|wall56` (screenshots
and overlap checks to `out/lab-ui/`).

In the browser the page sends the id; `engine-world-wasm` (`src/crash_ref.rs`) builds the
fleet with `CrashScenario::build`, reads the GPU state at every sample step and judges the
`CrashReport` with the same recorder. The generated designs' templates come from their mod
folders (`__mods/<design>_new`, which the site build generates with the same rigger:
`crash_templates::template_from_files`, the constructor `template` uses on the rigger's
output; a native test holds the two equal), so the body drawn is the physics' own: each
generated design its own skinned body (S09's wagon and 4x4 both). Bodies without a skin of
their own (the FMVSS 214 barrier) are drawn as their beams and their collision face.

Each run prints every criterion with its band and measured value; tier 0 and tier 1
failures fail the run unless recorded as known failures ("Currently failing"). The executor provides only `Runner` (step, read state); the metrics
are host code (`crash_reference/metrics.rs`), so CPU, CUDA and WebGPU are measured by the
same code.

## Reference frames

Third-party footage is not committed. The reference frames live in the **Reference Studio**
(`apps/reference-studio/`, its README): a static web app over one workspace folder
(`apps/reference-studio/workspace/`, its `library/` of images gitignored) where each frame
belongs to a *shot* (one crash run in the video) and is annotated, linked to a scenario and
pinned to a time and camera of the recorded run, and compared. Its points-only data is
committed in the workspace (`shots/`, `frames/`: scenario, video time, view, moment, what to
look at, regions, tracings, pins and cameras; `structures/`: the structure schemas;
`scenarios/`: recreated scenarios as data, below). The originals stay in `~/Development/rubble-artifacts/crash-reference/`;
A one-off import (removed; at commit 4773e6f1b) copied them into the library and migrated the old
`index.json` and tracings. `tools/crash-reference/sheet.py
[ID ...]` lays each linked frame beside Rubble's still from its camera at its moment (the
frame's own choice, else the rule in `crash_reference::shape::link`; the recorder's
`--all-cams` stills), one PNG per scenario and contact setting in `out/crash-reference/sheets/`
(`--copy-to DIR`); frames without an image (S10's) are skipped.

The cameras are framed for a 60° vertical field of view at 16:9 (the page's): side and top
views put the car at impact across about two thirds of the width, wide enough for where the
wreck ends (S07 tumbles back off the block, S10 slides on past it).

## Scenarios as data (recreated crashes)

A scenario can also be data: `crash_reference::ScenarioDesc` (`desc.rs`), written by the
Reference Studio's Scene view into `apps/reference-studio/workspace/scenarios/<id>.json`:
bodies (a generated design, where it stands on the ground, yaw, speed in km/h or a velocity,
on a carrier or not), static boxes (centre, half sizes, yaw: a turned box is an oriented box
shape), cameras, moments, seconds after first contact and the contact setting.
`ScenarioDesc::scenario()` is a `CrashScenario` whose layout is `LayoutSource::Data`: the same
`build`, `run`, metrics and recorder as the Rust scenarios, on every executor (the CPU
reference, CUDA, the page's WebGPU). The run's jitter is applied as the Rust scenarios apply
it (every speed, and the first moving body's lateral and yaw).

- **First contact is computed**, not stated (`desc::first_contact`): each body's spawn box
  (its rest extent's corners turned by its yaw, stood on its wheels) swept along its velocity
  against every static box's bound and every other body's box at their relative velocity;
  the earliest touch. Boxes, not surfaces, so it may come a little early, never late (S02
  as data: 0.099999994 s against S02's stated 0.1 s).
- **Criteria are the laws only** (`desc::law_criteria`): finite, energy created at most
  0.1 % of the start's kinetic energy (`all.energy_gain_frac`), nothing more than 5 cm into
  a static box, and without boxes the horizontal momentum's change within what friction
  (μ ≤ 1.2) and drag (C_d·A ≤ 8 m²) can give over the run (`all.momentum_friction_ratio`).
  A recreated crash that earns cited bands becomes a Rust file.
- `CrashScenario::to_desc` exports a Rust scenario as data (run 0, absolute positions);
  `tests/crash_scenario_data.rs` holds S02 as data equal to S02 as code word for word on the
  CPU reference through impact, and runs every committed data scenario.
- `crash_ref -- --scenario FILE.json`, `crash_ref -- --export S02`, `crash_outlines --
  --scenario FILE.json`; the registry pages read lists data scenarios after the built-in
  ones, marked `recreated` (`registry.rs`; the crash lab's `crashReferenceRegistry(data)`,
  `crashRef: { desc }`).

## Outlines: shape differences per part (prototype, S01 and S02)

Side-by-side stills hide shape differences (a hood that folds into a tent against one that
stays a plate). Outlines make them visible, per part, in four steps:

1. **Rubble's outlines, from Rust.** `cargo run --release -p engine-softbody-lab --example
   crash_outlines -- S02` runs the scenario on the CPU reference to each moment's exact step
   and writes `out/crash-reference/outlines/S02-edge/outlines.json`
   (`crash_reference::outline`): per part group (hood, fenders, bumpers, the four doors,
   trunk, greenhouse, roof, sills, wheels as centre and radius) the boundary of its
   collision triangles as each scenario camera sees it (the page's projection: look-at,
   60° vertical field of view, 1280 × 720), the same for the ghost (the rest shape carried
   rigidly with the cabin and rear, `metrics::robust_fit` over cabin and rear nodes), the
   hood's centreline (its nodes nearest the centre plane, front to rear), and all of it in
   the car's own side frame (x along the car, y up, origin the rear axle on the ground). It
   also renders each moment's still of the collision surface from each camera, so the
   outlines line up exactly; on the recorder's WebGPU stills they line up too, to within the
   GPU run's own difference. `--census DESIGN` lists a design's groups.
2. **Overlays.** `python3 tools/crash-reference/outlines.py all S01 S02` writes SVGs (outlines
   on the recorded and the CPU stills, the ghost dashed, onion skins of before and each
   moment), annotated reference frames, common-frame and change figures and
   `shape-numbers.md`; `python3 scripts/with-gpu-lock.py --limit 300 -- python3
   tools/crash-reference/outlines.py rasterise out/crash-reference/outlines/S01 ...` turns
   them into PNGs (headless Chrome over http, without the GPU).
3. **Tracings.** In the Reference Studio (`cd apps/reference-studio && npm run dev`,
   `http://127.0.0.1:8770/`): per frame, regions (the real and BeamNG halves' crops) each
   traced as an instance of a structure schema (`apps/reference-studio/workspace/structures/`:
   `vehicle-side` has the wheels as circles, hood, fender top, wheel arch, front face,
   windscreen, roofline, front door edge and its corner at the sill, sill and rear face;
   `vehicle-top` the outline, corners, centreline, hood, windscreen base, roof and side line),
   points in the region's normalised coordinates with a visibility flag (clear, blurred,
   hidden). The migrated S01/S02 tracings are a first pass placed by eye: approximate.
4. **One frame, shape numbers.** The schema names its frames: S01's frames show the whole
   car: origin the rear wheel, unit the wheelbase in each source's own before frame. S02's
   show only the front half: origin the front door's corner at the sill (Rubble: that
   corner carried with the cabin), unit the front wheel's radius. Hood fold height, chord
   shortening, creases (turns over 20°), front wheel setback, windscreen angle and ride
   height are computed by one Rust implementation for every source
   (`crash_reference::shape`, the example `reference_shape`), which the studio's Compare view
   and `outlines.py` both call; Rubble's elements are read off its outlines by rules in the
   schema. They are reported comparisons, not criteria. `crash_outlines -- ID --cameras FILE`
   also projects through explicit cameras (a frame's own camera, set by hand in the studio).

The generated designs' hoods have 4 rows of nodes front to rear, so a Rubble hood can bend at
two lines at most.

## What may be asserted

A criterion can fail a run only when its band has a basis, recorded with it
(`Criterion::law`, `Criterion::cited`):

- **a law**: conservation or a closed form (momentum, energy never created, nothing passes
  through, the velocity split from the mass ratio, a symmetric head-on's centre of mass at
  rest);
- **cited data under matching conditions**: a named source, the same speed and
  configuration, the same definition of the quantity, the band set to the data's whole
  spread and not its middle.

Every other band (an estimate, a reading of the reference footage) is measured and
reported and asserts nothing, whatever its tier: a wrong bound would be a wrong goal. Where
only a direction is certain (the struck side absorbs more), the direction is asserted (a
ratio above 1), not a size. The bands in sections 2 and 3 below are estimates unless a
scenario's file gives them a basis.

## Currently failing (asserted criteria)

Known failures are data: each is recorded on its criterion in the scenario's file
(`Criterion::known_failure` for one that fails in every run of a contact setting,
`Criterion::known_intermittent` for one that fails in some), with the measured value, the
date and a pointer here. A known failure that fails is reported as `KNOWN` (the page:
"known failure") and does not fail the run; an asserted criterion that fails without an entry
fails it. The ratchet (`crash_reference::verdict`, which `crash_ref` and the CUDA tests
apply to a scenario's runs in one setting): an entry recorded for every run that passes in
all of them fails with "known failure now passes: remove the entry", so the list only
shrinks; an intermittent one that does not occur is only noted. Bands, tiers and bases are
never changed by an entry. Each is a finding, not a band to widen.

The table is what `python3 tools/crash-reference/sweep.py --cpu` writes to
`out/crash-reference/sweep/known-cpu.md` from the entries and the CPU reference's 8 runs a
setting (2026-11-27). Rows marked new were found by that sweep and were not in this page's
earlier list: they are recorded so the suite runs green, and wait for review.

| Scenario | Criterion | Contact | Recorded as | Failed (CPU) | Value over the runs | Band | New, for review |
|---|---|---|---|---|---|---|---|
| S03 | no energy created | edge | every run | 8 of 8 | 14,112 to 211,378 |  .. 132.0 |  |
| S03 | no energy created | continuous | every run | 8 of 8 | 6,719 to 29,661 |  .. 132.0 |  |
| S04 | no energy created | discrete | some runs | 1 of 8 | 0.000 to 221.4 |  .. 182.0 |  |
| S04 | no energy created | edge | every run | 8 of 8 | 2,242 to 43,160 |  .. 182.0 |  |
| S04 | no energy created | continuous | some runs | 3 of 8 | 0.000 to 516.1 |  .. 182.0 |  |
| S06 | no energy created | edge | every run | 8 of 8 | 4,107 to 39,865 |  .. 610.0 |  |
| S06 | no energy created | continuous | some runs | 1 of 8 | 23.6 to 991.4 |  .. 610.0 |  |
| S07 | nothing passes into the block | continuous | some runs | 1 of 8 | 0.009 to 0.050 |  .. 0.050 | **yes** |
| S08 | the common centre of mass stays still | discrete | some runs | 2 of 8 | 0.060 to 0.461 |  .. 0.300 | **yes** |
| S08 | momentum along the road | edge | some runs | 1 of 8 | -133.6 to 642.8 | -450.0 .. 450.0 | **yes** |
| S08 | momentum across the road | edge | some runs | 1 of 8 | -283.7 to 519.0 | -450.0 .. 450.0 | **yes** |
| S08 | the common centre of mass stays still | edge | some runs | 2 of 8 | 0.144 to 0.409 |  .. 0.300 | **yes** |
| S08 | no energy created | continuous | some runs | 1 of 8 | 8.197 to 442.9 |  .. 340.0 |  |
| S08 | momentum along the road | continuous | some runs | 2 of 8 | -410.7 to 568.1 | -450.0 .. 450.0 | **yes** |
| S08 | the common centre of mass stays still | continuous | every run | 8 of 8 | 0.399 to 0.661 |  .. 0.300 |  |
| S09 | no energy created | discrete | some runs | 1 of 8 | 8.191 to 531.5 |  .. 400.0 |  |
| S09 | no energy created | edge | some runs | 5 of 8 | 1.796 to 2,510 |  .. 400.0 |  |
| S10 | nothing passes into the block | continuous | some runs | 4 of 8 | 0.001 to 0.147 |  .. 0.050 | **yes** |
| S11 | restitution within the 56 km/h sedan fleet's | discrete | every run | 8 of 8 | 0.302 to 0.346 |  .. 0.250 |  |
| S11 | restitution within the 56 km/h sedan fleet's | edge | every run | 8 of 8 | 0.302 to 0.346 |  .. 0.250 |  |
| S11 | restitution within the 56 km/h sedan fleet's | continuous | every run | 8 of 8 | 0.295 to 0.356 |  .. 0.250 |  |
| S14 | no energy created | edge | every run | 8 of 8 | 4,703 to 1,006,835 |  .. 203.0 |  |
| S14 | no energy created | continuous | every run | 8 of 8 | 6,271 to 13,083 |  .. 203.0 |  |

S03 (the barrier into the sedan's side), localised:

With edge contact the same run also shows what created energy looks like: plastic work of
419 kJ (three times the energy the scene starts with), a car node at 138 m/s when the barrier
arrives at 13.9 m/s. Discrete contact has none of this (38 kJ of plastic work, fastest node
16 m/s).

Localised 2026-11-27 (`crates/engine-softbody-lab/tests/crash_metrics_s03_energy.rs`, an
ignored probe printing the solver's own work ledger every 4 ms):

- The ledger is right for the barrier's beam kinds: its columns, densified blocks and shear
  beams driven through the solver's force functions give work = plastic + stored to 0.5 %.
- The energy comes from the position step of edge-contact hits. With that step off for edge
  hits (`EDGE_NO_SHIFT=1`, a measurement switch) the run creates 9 J, plastic work is
  37.6 kJ and the fastest car node 16 m/s, as with discrete contact. With edge contact on
  outer-skin edges only (`EDGE_OUTER_ONLY=1`): 9 J.
- It starts at first contact, in car-to-barrier contact, and pumps about 10 kJ every 4 ms
  while the barrier stays pressed; the fast nodes are the door's inner-panel nodes.
- The barrier's bumper element is not the source (removed: still 8,575 J). It does go under
  the sill: barrier nodes end 3.3-3.7 m past the car's centre of mass in every contact mode.
- Hypothesis, not established: edge hits on inner-face edges take their side from the
  partner, and their position step pushes inner door-panel nodes against the beams tying
  them to the outer skin every step, storing energy no force paid for.

S14's created energy is S03's (above): edge-contact hits between the barrier and the door.

Collision-mesh crossings at any sample (the repo's hard rule is 0), nominal run:

| Scenario | Discrete | Edge | Edge + continuous |
|---|---|---|---|
| S03 car and barrier | 94 | 315 | 0 |
| S08 two sedans | 154 | 31 | 0 |

On CUDA, continuous contact does not engage for one vehicle against static boxes: it runs
only on the split group path, which needs vehicle contact between two vehicles or more
(`FramePlanner::contact_active`). The executor now says so: `all.continuous_engaged` (0/1,
`Runner::continuous_engaged`, from `CudaExplicitBeams::continuous_ran`) and the report's note
"continuous contact did not engage on this executor for this scene". Measured on a T4
(`engine-cuda/tests/crash_reference.rs`, `continuous_engagement`, S11 run 0): S11 with the
continuous setting has `continuous_engaged` 0 and is the edge run with the same passes, all
198 metrics bit for bit; S08 (two cars) has 1. The CUDA runner takes the lab's continuous
setting (`vehicle_lab::options::continuous_on`: group split on, self split and the wide passes
off); **with the wide passes off a one-car CUDA run is not the edge run**: S11 run 0's
rebound ratio 0.235 and end speed 3.67 m/s against 0.323 and 5.05 with them on (152 of 198
metrics differ), and the CPU reference gives 0.323 and 5.05. The wide passes are meant to
give the same results either way, so this is an open finding: the CPU reference matches
the run with them on, so the path without them is the suspect for one vehicle with surface
contact (not established). With two vehicles (S08, in
Chrome) continuous contact engages and diverged from the CPU reference (momentum along the
road -1,024 kg m/s against -13).

## Reported outside its band

Measured on the generated sedan, edge contact, nominal run. Not failures: these are
estimates' bands, listed so the gaps are visible.

| Scenario | Measure | Band (basis) | Measured | Since |
|---|---|---|---|---|
| S02 (sedan stand-in for the SUV) | rebound, share of approach speed | at most 0.2 (estimate: NHTSA restitution 0.13-0.20 is the p10-p90 of sedans at 56 km/h; this run is 50 km/h) | 0.35 on the CPU reference and on CUDA | 2026-11-27 |

## 1. The ten scenarios

Speeds marked (video) are shown on screen; the others are the published protocol of the
test shown in the real half.

| # | Scenario | Video time | Real test shown | Speed | Camera angles |
|---|---|---|---|---|---|
| S1 | Small car, 40 % offset barrier | 1:52, 2:06 | Euro NCAP offset deformable barrier | 64 km/h | side, top |
| S2 | SUV, full-width rigid wall | 2:10 | Euro NCAP full width | 50 km/h (video) | side, top |
| S3 | Barrier sled into car's side | 2:26, 2:31 | IIHS/Euro NCAP moving barrier | 50–60 km/h | front 3/4 |
| S4 | Car under trailer rear | 2:59 | IIHS underride, full overlap | 56 km/h | side |
| S5 | Heavy truck T-bones car | 3:23, 3:29 | TV test, truck into stationary car | ~70–80 km/h | front of car, low |
| S6 | Car crushed between moving and parked truck | 4:06 | DEKRA | truck ~40–50 km/h, closing | side, aftermath |
| S7 | Car into concrete block at very high speed | 5:32 | TV test | 190 km/h (video) | side, aftermath ×2 |
| S8 | Car vs same car, head-on, partial overlap | 5:57 | DSK test 156 | ~50–55 km/h each | side, top |
| S9 | Car vs heavier, taller SUV, head-on | 6:15 | DSK test 157 | ~50–55 km/h each | side |
| S10 | Wagon into a narrow rigid block, very high speed, partial overlap | 6:33 | DTC test (car marked "200") | ~200 km/h | side close, side wide, aftermath |

## 2. How a crash is quantified (the measuring kit)

The owner's idea, looking at which bonds bend and break where, is the core of it. Three
kinds of measurement, all computed from solver state, none from pixels:

**A. Beam state by zone ("where did the metal work").** Every beam belongs to a part of
the generated car; parts are grouped into zones: front crush zone (bumper, rails ahead of
the firewall, hood, fenders), cabin cage (A/B/C pillars, sills, roof rails, floor), rear
(trunk, rear rails), each split left/right, plus doors, wheels/suspension. Per zone:
fraction of beams yielded, fraction broken, plastic work absorbed (J). Criteria are then
statements such as "front-left absorbs more than front-right", "the cabin cage absorbs
less than 15 % of the total", "at least one wheel or door separates".

**B. Gauges ("tape measures").** Distances between named node pairs, as crash labs do:
overall length, wheelbase left and right, door opening length (A-pillar to B-pillar at
the beltline) left and right, cabin width at the B-pillar, roof height above the sill,
front overhang. Reported as residual (at rest after the crash) and peak change.

**C. Kinematics.** Centre-of-mass velocity before/after of each body, pitch/yaw/roll
peak and final, deceleration pulse (peak, mean, duration), total momentum, energy ledger
(kinetic, elastic, plastic work, broken-beam loss, damping, contact friction).

Criteria come in three tiers, so that tests are not needlessly strict:

- **Tier 0, invariants (hard, every scenario).** Finite state; total momentum changes only
  by the impulse from ground and fixed obstacles; mechanical energy never rises; no body
  passes through another (the wall, the trailer, the other car) and the existing
  interpenetration acceptance holds; mass of detached parts is conserved; deformation is
  permanent: residual crush ≥ 70 % of peak crush (metal does not spring back).
- **Tier 1, first-principles bands (hard, wide).** Things physics fixes regardless of the
  car: post-impact velocities from mass ratio, crush order of magnitude from speed,
  which side/zone must take the damage, what must stay intact. Bands are wide
  (typically ×0.5 to ×2 of the estimate).
- **Tier 2, shape descriptors (reported, warn only).** What the frames show but which
  depends on the particular car: the hood tenting upward, the yaw angle, which panels
  come off. Printed in the report and used in the side-by-side; promoted to tier 1 only
  by the owner's decision.

Each scenario runs 8 times with small jitter (speed ±2 %, lateral offset ±2 cm, yaw
±0.5°). A tier-1 criterion must hold on all 8. This is what stops a criterion from being
tuned to one lucky run.

Useful first-principles numbers (m = car, M = heavy partner, V = closing speed, plastic
impact, e ≈ 0.1):

- Crush length s ≈ v² / (2·a_mean); passenger cars are built for a_mean of 20–35 g.
- Specific energy: 50 km/h 96 J/kg, 64 km/h 158 J/kg, 190 km/h 1394 J/kg (14× the 50).
- Common velocity after a plastic hit: v = (M·V_M + m·V_m) / (M + m).

## 3. Acceptance criteria per scenario

### S1. Small car, 40 % offset barrier, 64 km/h
What the frames show: the struck (driver's) front corner collapses to the front wheel; the
wheel is pushed back to the sill; the hood buckles upward; doors, pillars, roof and the
whole rear stay in shape; top view: the car rotates about the struck corner (about 20–30°)
and the rear swings out; the rear lifts a little.
First principles: 17.8 m/s, only 40 % of the front structure is loaded, so crush is deeper
than a full-width hit and the off-centre force yaws the car toward the barrier.
- Tier 1: car stops and rebounds at < 20 % of approach speed; peak crush on the struck
  side 0.4–1.1 m; struck-side front zone absorbs ≥ 2× the other front side; front zones
  together absorb ≥ 60 % of the plastic work, cabin cage ≤ 15 %; door opening length on the
  struck side shrinks ≤ 10 %, other side ≤ 3 %; rear zone essentially undamaged (< 2 % of
  beams yielded); yaw is toward the struck side and non-zero (≥ 5°); no rollover.
- Tier 2: yaw 10–35°; struck-side wheelbase shortens 5–25 cm; hood peak rises above its
  rest height; rear wheels unload; mean deceleration 15–40 g over 80–160 ms.

### S2. SUV, full-width rigid wall, 50 km/h
Frames: front crushed evenly across the width up to about the wheel arch; hood folds into a
tent; top view stays straight and symmetric; cabin intact; windscreen cracked.
First principles: 13.9 m/s, s ≈ 0.3–0.6 m at 17–33 g; symmetric load means no yaw.
- Tier 1: crush 0.25–0.8 m; left/right front plastic work within 25 % of each other;
  |yaw| < 5°, lateral drift < 0.15 m; rebound speed < 20 % of approach; door openings
  shrink ≤ 5 %; cabin ≤ 10 % of plastic work; rear undamaged; wall does not move and
  nothing passes through it.
- Tier 2: hood peak rises (tent); rear pitches up 1–8° at peak; both front wheels stay
  attached; pulse 80–140 ms.

### S3. Barrier sled into the side of a stationary car
Frames: sled face buried in both doors between the axles; B-pillar and doors pushed in;
roof rail bowed; glass gone on the struck side; nose and tail of the car unharmed; car
shoved sideways, still on its wheels.
First principles: plastic hit, struck car leaves at v = M·V/(M+m) (sled 1.5 t at 50 km/h
into a 1.5 t car: about 25 km/h sideways); a car's side has 15–30 cm of structure, so
intrusion is large relative to the energy.
- Tier 1: car's lateral speed after contact within ±25 % of M·V/(M+m), sled slowed by the
  matching momentum; cabin width at the B-pillar shrinks 0.10–0.45 m (residual);
  struck-side doors/B-pillar/sill absorb ≥ 60 % of plastic work; far-side doors and both
  ends of the car < 5 % of beams yielded; no rollover (peak roll < 25°); sled face does
  not pass the car's centreline.
- Tier 2: roof rail on the struck side bows inward; struck-side wheels keep contact;
  car yaws little if the sled is centred between the axles (< 10°).

### S4. Car under the rear of a semi trailer, 56 km/h
Frames: the trailer's underride guard meets the car at hood height; the hood and front end
are crushed back to the windscreen base; front wheels pushed back; the cabin stays out
from under the box (guard held); the trailer hardly moves.
First principles: trailer ≥ 10× the car's mass, so it is close to a wall, but one that
starts above the bumper: the load path is high, so the upper front (hood, fenders, cowl)
does the work and the lower rails do less than in S2.
- Tier 1: trailer speed after < 10 % of the car's approach speed; car stops (rebound
  < 20 %); front crush 0.5–1.3 m; upper front zone (hood/fenders/cowl) absorbs at least as
  much as the lower rails (the reverse of S2: assert the ratio is higher than S2's); no
  part of the cabin cage passes through the trailer body; rear undamaged.
- Tier 2: guard-held outcome: roof and A-pillars keep ≥ 90 % of roof height. If the guard
  model fails instead, the car goes under and the greenhouse is sheared: reported as the
  other valid outcome, not as a failure, with the guard's load at failure printed.

### S5. Heavy truck T-bones a stationary car
Frames: the car is wrapped around the truck's nose: the struck side is caved in past the
car's middle, the nose and tail fold toward the truck (a V in plan), the car tilts and is
carried along; hood and screen destroyed.
First principles: M/m ≥ 8, so the car is accelerated to nearly the truck's speed
(v ≥ 0.89·V) in about 0.1 s while the truck barely slows; the car's side cannot absorb that,
so the whole section collapses.
- Tier 1: car's speed along the truck's heading after contact ≥ 80 % of truck speed; truck
  loses ≤ 20 %; cabin width at the B-pillar shrinks ≥ 35 %; plan-view bend: the nose and
  tail end up ≥ 0.3 m ahead of the struck mid-section along the truck's heading; cabin cage
  absorbs ≥ 40 % of plastic work (the opposite of S1/S2); the car stays in front of the
  truck (no pass-through, no ejection sideways faster than the truck's speed).
- Tier 2: car rolls toward or away by 10–45°; struck-side wheels fold under; ≥ 1 door or
  panel separates.

### S6. Car crushed between a moving truck and a parked truck
Frames: truck drives the car's nose under the parked trailer's rear, then the cab closes
the gap; the car all but disappears between them; aftermath: car about half length, roof
flattened to the beltline, everything folded.
First principles: both partners are ≥ 10× the car and far stiffer. The car is the only
thing that can shorten, so its final length is set by geometry (the gap left between the
two trucks), not by its strength.
- Tier 1: car's residual overall length ≤ 60 % of original; roof height above sill ≤ 60 %
  of original over the cabin; ≥ 50 % of all the car's beams yielded and every zone
  (front, cabin, rear) has yielded beams; parked truck is pushed forward by the momentum
  the moving truck lost (ledger closes within 15 % after ground friction); car's mass
  stays between/under the trucks (centre of mass between the two truck bodies in plan);
  no explosion: no node leaves faster than 1.5× the truck's speed.
- Tier 2: wheels still visible near ground level; nose under the trailer deck; some
  panels detached.

### S7. Car into a concrete block at 190 km/h
Frames: the front vanishes in the first metre; the crush runs through the cabin; the rear
rises until the car stands on its nose; aftermath: a wreck about half the car's length,
rear axle on top, roof folded, doors and a wheel on the ground.
First principles: 14× the energy of the 50 km/h test. The crush zone can absorb roughly a
60–70 km/h impact; everything beyond must be absorbed by the cabin and rear. Stopping
52.8 m/s within about 2 m of crush is 60–70 g for about 80 ms. The centre of mass sits
above the line of the crush force, so the rear pitches up hard.
- Tier 1: nothing passes through the block (this is the continuous-collision test);
  residual overall length ≤ 60 % of original; cabin cage absorbs ≥ 30 % of plastic work
  and door openings shrink ≥ 30 %; all three zones have yielded beams, rear included;
  ≥ 1 break group separates (door, wheel, hood or bumper); rebound speed < 15 % of
  approach; peak pitch (rear up) ≥ 20°; energy ledger closes (no energy created at
  4× the usual contact speeds).
- Tier 2: pitch reaches 60–90°; rear axle ends above the front axle; ≥ 3 parts on the
  ground; wreck rests within 3 m of the block.

### S8. Two equal cars head-on, partial overlap
Frames: both fronts crushed to the wheel on the overlapping side, hoods folded, both
rears lifted off the ground, cabins intact; top view: the cars stay nearly in line.
First principles: equal masses and speeds, so the centre of mass stays still and each car
sees what it would see against a wall at its own speed. Symmetry is the test.
- Tier 1: combined centre-of-mass velocity stays within 5 % of approach speed of zero;
  both cars end at < 20 % of approach speed; the two cars' front plastic work within
  30 % of each other and peak crush within 30 %; each car's front zone ≥ 60 % of its
  plastic work, cabin ≤ 15 %; door openings shrink ≤ 10 %; the cars do not pass through
  each other (overlap of the two undeformed cabin volumes is zero).
- Tier 2: both rears pitch up 3–15° at peak; hoods rise; both cars yaw the same rotational
  sense (offset) by < 20°.

### S9. Car vs heavier, taller SUV, head-on
Frames: the SUV rides over the car's hood; the car's front is crushed back to the screen
and its rear is thrown up; the SUV's front is damaged much less and its hood pops; the
pair moves in the SUV's direction.
First principles: with mass ratio r = M/m ≈ 1.6 the car takes r/(1+r) ≈ 62 % of the
closing speed as its velocity change and the SUV 38 %; the SUV's rails are higher than the
car's, so they miss each other and the car's softer upper structure takes the load.
- Tier 1: velocity changes split by mass ratio within ±25 % (car's larger); pair's common
  velocity points in the SUV's direction; car's plastic work ≥ 1.3× the SUV's; car's peak
  crush > SUV's; SUV's cabin door openings shrink ≤ 5 %; car's rear pitch peak > SUV's;
  no pass-through.
- Tier 2: SUV's front bumper ends above the car's hood line (override); car's door opening
  shrinks more than in S8; SUV hood rises.

### S10. Wagon into a narrow rigid block at about 200 km/h, partial overlap
Frames: the block is narrower than the car and meets one side of the front; that side is
destroyed back through the front door; the rear rises to 40–60° and the car swings round
the block; BeamNG aftermath: everything ahead of the B-pillar is wreckage, the rear half
(rear doors, roof over the back, rear axle) keeps its shape, a wheel and a panel lie on
the ground.
First principles: same energy class as S7 (55 m/s), but only part of the width is loaded.
The loaded side cannot stop the car, so it is cut open, while the unloaded side carries
on: the car yaws and pitches about the block and much of the energy leaves as rotation and
residual speed instead of crush. That is why the rear survives here and not in S7.
- Tier 1: nothing passes through the block; struck-side door opening shrinks ≥ 40 %,
  the other side's shrinks less than the struck side's; struck-side front and cabin zones
  absorb ≥ 2× the other side's; rear zone damage is lower than in S7 (compare the two
  runs: yielded fraction S10 < S7); residual length on the struck side ≤ 75 % of
  original; ≥ 1 break group separates; peak pitch ≥ 20° and yaw ≠ 0 (≥ 10°); energy
  ledger closes.
- Tier 2: pitch 40–70°; a front wheel detaches; rear axle and rear doors stay attached;
  wreck comes to rest beside or past the block, not in front of it.

### Cross-scenario checks (cheap and strong)
- Overlap matters at high speed: S10's rear survives, S7's does not.
- Crush grows with speed: S7 residual shortening > S1 > S2-equivalent at equal car.
- S8 per-car crush ≈ the same car into a rigid full-width wall at the same speed, same
  overlap (within 30 %).
- Damage follows the hit: zone with the highest plastic work is the struck zone in every
  scenario.

## 4. Agency tests with measured data (S11–S14)

The four NHTSA configurations the CCSA 2012 Camry finite-element model was validated
against, staged with the generated sedan, and NHTSA's measured curves for two Camrys each:

| # | Configuration | Speed | NHTSA tests (Camry, test mass) | File |
|---|---|---|---|---|
| S11 | NCAP frontal, full-width rigid wall | 56.3 km/h | 8545 (2014, 1,664 kg), 7520 (2012, 1,662 kg) | `s11_ncap_frontal_56.rs` |
| S12 | Full-width rigid wall (FMVSS 208 rigid barrier) | 40 km/h | 9692 (2015, 1,690 kg), 7793 (2011 Hybrid, 1,798 kg) | `s12_frontal_40.rs` |
| S13 | NCAP side pole: carried on a carrier at 75°, 254 mm pole at the driver's head | 32 km/h | 7523 (2012, 1,566 kg), 8558 (2014, 1,579 kg) | `s13_side_pole.rs` |
| S14 | NCAP side MDB: FMVSS 214 barrier crabbed 27° into the driver's side | 38.5 mph (61.96 km/h) | 7517 (2012, 1,641 kg), 8560 (2014, 1,649 kg) | `s14_ncap_side_mdb.rs` |

**The generated sedan is not a Camry.** It is 1,448 kg as spawned (no dummies or ballast),
12–19 % lighter than the tests' 1,566–1,798 kg test masses, and a different structure. A
Camry's curve is therefore a comparison, never a band: only laws and fleet-wide ranges at the
same speed and configuration are asserted (section "What may be asserted"). Everything
curve-shaped, and every single-Camry number, is reported.

**Reference data.** `docs/crash-reference/nhtsa/<test>.json`: our own numbers derived from
NHTSA's public-domain test records (the downloads stay in the gitignored
`inspiration/ccsa-camry-2012/`) by `tools/crash-reference/nhtsa_reference.py camry <data>
<out>`: per test the channels used, the vehicle (test mass, curb mass, C1–C6 crush profile,
maximum crush, length loss), the scalars, the side tests' per-level maximum exterior crush
(read from the report with `pdftotext`), and the CFC 60 curves at 1 kHz from −10 to 250 ms.
No test or build reads them; the overlay tool does.

### 4.1 Time histories (`metrics::series`)

Per body, on the run's 4 ms samples, in SAE axes of its first frame (x forward, y right),
from the layout's first contact: the acceleration, velocity and displacement of its
**reference location** (the rear-seat crossmember's place: the mean of the cabin-floor nodes
behind the B-pillar and ahead of the rear axle, chosen by position at rest,
`series::reference_nodes`; 14 nodes on the generated sedan) and of its centre of mass; for
the scene, the **force on the static boxes**, −d(Σ m v)/dt over every node along body 0's
initial velocity. Every executor's samples go through this host code; the web runner and
CUDA report the same `series`.

- **Velocity** is the samples' own (point samples). **Displacement** is from positions
  (Hermite on the velocity when read between samples). **Acceleration** is the central
  difference of the velocity samples, exactly the mean acceleration over the 8 ms around each
  sample.
- **CFC 60 cannot be applied to 4 ms samples**: SAE J211's design frequency for CFC 60
  (2.0775 × 60 = 124.7 Hz) is the sampling's 125 Hz Nyquist frequency (`signal::cfc` refuses
  it). So the CPU reference also **traces every step** (`Runner::start_trace`): each body's
  reference location's and centre of mass's velocity after every 2 kHz step, through central
  differences and CFC 60 (`series::Fine`, `{ref,com}.cfc60_*`). The trace replays
  `ExplicitBeamsWorld::frame` and is held bit for bit to an untraced run
  (`engine-softbody-lab/tests/crash_series_rate.rs`). CUDA and the browser do not trace; their
  series is the 4 ms one.
- **Force on the boxes**: exact for vehicles against fixed boxes except for the other
  horizontal forces, tyre and ground friction and drag. Measured over the 40 ms before contact
  (`all.box_force_pre_n`) and bounded during it by μ m g (about 14 kN on the sedan, 2 % of a
  660 kN peak). The solver does not count the static boxes' contact force itself (it acts in
  the node pass, the surface pass and the edge pass), so momentum is the cheaper exact route.
- **Filter and integration, closed forms** (`metrics/signal.rs`, `metrics/tests.rs`): a 40 g
  half-sine of 80 ms keeps its peak to 0.001 % and its Δv to 1e-6 through CFC 60 at 10 kHz;
  the J211 filter is −3 dB at 100 Hz (|H| = 0.707 ± 0.02), 0.05 at 400 Hz; trapezoidal Δv and
  displacement of a half-sine within (π Δt / T)² / 4 of the closed form. Through the recorder,
  a rigid body with a 40 g, 100 ms half-sine against a wall, sampled at 4 ms: peak 39.82 g
  against the box-car's exact 39.82 (the 8 ms mean of the sine at the sample nearest the
  peak, 0.46 % under 40 g), t_m 58.04 ms against 58.02, dynamic crush to 1 µm, rebound and
  wall impulse to 1e-5, ∫F dt over the force curve to 0.1 %.
- **What the 4 ms estimator loses on real pulses.** A crash pulse has short spikes; an 8 ms
  mean keeps less of them than CFC 60:

| Record | CFC 60 peak | 4 ms estimator on the CFC 60 velocity | on the raw samples |
|---|---|---|---|
| S11 generated sedan (CPU, 2 kHz trace) | 59.1 g at 15.5 ms | 49.1 g | 50.0 g (the report) |
| S12 generated sedan | 52.7 g at 31.0 ms | 43.8 g | 46.2 g |
| NHTSA 8545 / 7520 (56 km/h) | 39.8 / 45.0 g | 36.8 / 40.7 g | — |
| NHTSA 9692 / 7793 (40 km/h) | 34.1 / 31.5 g | 27.1 / 26.1 g | — |
| NHTSA §8 sedans (56 km/h) | 35.5–47.3 g | 31.9–44.5 g | — |
| NHTSA 7523 / 8558 pole, CG Y | 16.5 / 17.9 g | 13.2 / 13.2 g | — |
| NHTSA 7517 / 8560 side MDB, CG Y | 31.8 / 35.6 g | 23.6 / 27.4 g | — |

  So peaks are compared like with like: Rubble's 4 ms peak against the tests' "as Rubble
  samples" (their CFC 60 velocity sampled every 4 ms and differenced), and Rubble's CPU CFC 60
  peak against the tests' CFC 60. The aliasing of the raw samples (raw against CFC 60
  velocity, both every 4 ms) is 2–5 % on the sedan. Velocity and displacement are not
  affected (the integrated CFC 60 velocity is within 0.3 m/s of the raw one).

### 4.2 Criteria and results

CPU reference, 8 runs (jitter as every scenario), discrete / edge / edge + continuous. For
S11–S13 discrete and edge give the same numbers (a single car against static boxes has no
edge-pair contact).

**S11, asserted:** finite, no energy created (≤ 177 J: measured 0 in every run and setting),
nothing into the wall (0 m), no rebound past elastic, the wall stops the car (law), and the
fleet's restitution (cited, ≤ 0.25): **fails, 0.29–0.36** in every run and setting. Reported
(nine sedans at 56 km/h, targets §8.1): dynamic crush 0.385–0.403 m (fleet 0.617–0.753), t_m
43–47 ms (62.7–76.3), mean deceleration to t_m 34–37 g (21.0–25.4), 4 ms peak 50–53 g (as
the fleet is sampled: 31.9–44.5), residual length loss 0.19–0.23 m (82 sedans p10–p90
0.311–0.602), wall force peak 656–675 kN (Camry 830, 921).

**S12, asserted:** the same laws (≤ 89 J: measured 0–2.8 J); all hold. Reported against the
two Camrys only (no 40 km/h fleet is held): dynamic crush 0.27–0.28 m (0.475, 0.489), t_m
39 ms (69, 75), restitution 0.46–0.50 (0.107, 0.149), 4 ms peak 43–49 g (as sampled 26.1,
27.1), residual length loss 0.09–0.10 m (0.279, 0.304).

**S13, asserted:** finite, no energy created (≤ 57 J: 9–15 J), nothing into the pole (0 m),
the pole throws it back no faster than it came and the pole and carrier only slow it (the
COM's velocity along the travel at the end, −1.0 to −0.46 m/s), the struck side absorbs more
(left/right work 10–18). All hold. Reported: residual crush of the struck door band
0.215–0.306 m (116 sedans' min–max 0.169–0.537; Camry 0.394, 0.379: not asserted, the
definitions differ), pole force peak 236–330 kN (Camry 191, 197), CG Δv_y by 150 ms
9.7–10.4 m/s (10.26, 10.82).

**S14, asserted:** finite; no energy created (≤ 203 J): **fails with edge contact (4.7 kJ to
1.0 MJ) and continuous (6.3–13.1 kJ)**, holds with discrete (4–5 J); momentum along the
barrier's travel stays in [0, 24,033] N s (17,400–18,700); the car leaves no faster than an
elastic hit (8.3–9.9 m/s ≤ 17.05); the struck side absorbs more (2.9–60). Reported below.

### 4.3 Beside the real tests

Nominal run (run 0), edge contact; Rubble's car is the 1,448 kg generated sedan in every row.
The overlays (`out/crash-reference/overlays/<id>-<contact>.png`, made by
`tools/crash-reference/overlay.py`) draw the same curves on the same axes.

#### S11, edge contact, run 0

Rubble is the generated sedan as generated by default (every front rule off, as before). "Rule 5" and "all six rules" are the same sedan generated with `Physics::front_rules` (`hood_fold_row` alone; every rule), measured the same way (2026-11-27, `crash-ref/front-fixes`; `sedan-front-audit.md` §8, `generated-vehicles-lessons.md` §25). Neither is the default: each breaks a gate or a hard rule elsewhere (lessons §25.8). With rule 5 the S11 run 0 meets the cited restitution (0.247 against at most 0.25; 6 of 8 runs in each contact setting). "Box cell": the crash box as a closed four-corner cell (`crash_box_cell`, lessons §26), alone and with every rule; also not the default.

| quantity | Rubble | Rubble, rule 5 | Rubble, all six rules | Rubble, box cell | Rubble, box cell and all six | test 8545 | test 7520 |
|---|---|---|---|---|---|---|---|
| mass (kg) | 1448 | 1448 | 1448 | 1448 | 1448 | 1664 | 1662 |
| speed (km/h) | 56.3 | 56.3 | 56.3 | 56.3 | 56.3 | 56.7 | 56.3 |
| peak decel, 4 ms estimator (g) | 50.0 | 48.5 | 46.7 | 50.7 | 45.8 | 36.8 | 40.7 |
| peak decel, CFC 60 (g) | 59.1 | 60.8 | 54.0 | 61.3 | 51.6 | 39.8 | 45.0 |
| time of peak, CFC 60 (ms) | 15.5 | 15.5 | 44.0 | 16.0 | 53.5 | 44.1 | 43.9 |
| t_m, v = 0 (ms) | 44.4 | 48.6 | 54.0 | 44.6 | 54.5 | 70.9 | 75.9 |
| dynamic crush (m) | 0.394 | 0.415 | 0.492 | 0.378 | 0.507 | 0.663 | 0.685 |
| mean decel to t_m (g) | 35.9 | 32.7 | 29.5 | 35.8 | 29.2 | 22.7 | 21.0 |
| rebound, t_m-200 ms (m/s) | 5.34 | 3.85 | 4.35 | 5.41 | 4.91 | 1.50 | 0.89 |
| restitution | 0.342 | 0.247 | 0.278 | 0.346 | 0.315 | 0.095 | 0.057 |
| residual crush (m) | 0.199 | 0.252 | 0.344 | 0.181 | 0.307 | 0.554 (length), 0.529 (max) | 0.579 (length), 0.579 (max) |
| peak wall force (kN) | 661 | 620 | 560 | 704 | 535 | 921 | 829 |
| time of peak force (ms) | 20.0 | 16.0 | 44.0 | 20.0 | 40.0 | 29.5 | 30.1 |


#### S12, edge contact, run 0

| quantity | Rubble | Rubble, rule 5 | Rubble, all six rules | Rubble, box cell | Rubble, box cell and all six | test 9692 | test 7793 |
|---|---|---|---|---|---|---|---|
| mass (kg) | 1448 | 1448 | 1448 | 1448 | 1448 | 1690 | 1798 |
| speed (km/h) | 40.0 | 40.0 | 40.0 | 40.0 | 40.0 | 39.6 | 39.6 |
| peak decel, 4 ms estimator (g) | 46.2 | 41.9 | 31.1 | 45.3 | 26.6 | 27.1 | 26.1 |
| peak decel, CFC 60 (g) | 52.7 | 44.5 | 33.8 | 48.8 | 28.2 | 34.1 | 31.5 |
| time of peak, CFC 60 (ms) | 31.0 | 30.5 | 20.5 | 31.0 | 27.0 | 55.1 | 43.2 |
| t_m, v = 0 (ms) | 39.5 | 40.7 | 68.5 | 38.6 | 66.1 | 69.3 | 74.7 |
| dynamic crush (m) | 0.275 | 0.283 | 0.378 | 0.263 | 0.370 | 0.475 | 0.489 |
| mean decel to t_m (g) | 28.6 | 27.8 | 16.5 | 29.4 | 17.1 | 16.2 | 15.0 |
| rebound, t_m-200 ms (m/s) | 5.37 | 5.01 | 2.95 | 5.47 | 3.34 | 1.64 | 1.17 |
| restitution | 0.484 | 0.452 | 0.265 | 0.493 | 0.301 | 0.149 | 0.107 |
| residual crush (m) | 0.099 | 0.111 | 0.232 | 0.095 | 0.192 | 0.304 (length), 0.326 (max) | 0.279 (length), 0.279 (max) |
| peak wall force (kN) | 576 | 538 | 431 | 606 | 400 | - | - |
| time of peak force (ms) | 32.0 | 32.0 | 24.0 | 28.0 | 24.0 | - | - |


#### S13, edge contact, run 0

| quantity | Rubble | test 8558 | test 7523 |
|---|---|---|---|
| struck car mass (kg) | 1448 | 1579 | 1566 |
| speed (km/h) | 32.0 | 31.9 | 32.1 |
| CG lateral peak, 4 ms estimator (g) | 21.0 | 13.2 | 13.2 |
| CG lateral peak, CFC 60 (g) | 21.9 | 17.9 | 16.5 |
| CG dv_y by 150 ms (m/s) | 9.75 | 10.82 | 10.26 |
| CG dv_y largest (m/s) | 9.75 | 10.89 | 10.38 |
| peak pole force (kN) | 307 | 197 | 191 |
| time of peak force (ms) | 44.0 | 46.0 | 43.9 |
| residual side crush (m) | 0.236 | 0.379 | 0.394 |
| peak side crush, dynamic (m) | 0.321 | - | - |
| CG travel at stop (m) | - | 0.455 | 0.510 |


#### S14, edge contact, run 0

| quantity | Rubble | test 8560 | test 7517 |
|---|---|---|---|
| struck car mass (kg) | 1448 | 1649 | 1641 |
| barrier mass (kg) | 1368 | 1362 | 1362 |
| barrier speed (km/h) | 62.0 | 61.9 | 62.3 |
| CG lateral peak, 4 ms estimator (g) | 13.6 | 27.4 | 23.6 |
| CG lateral peak, CFC 60 (g) | 15.2 | 35.6 | 31.8 |
| CG dv_y largest (m/s) | 8.37 | 6.79 | 6.60 |
|   when (ms) | 96 | 110 | 88 |
| CG dv_y by 150 ms (m/s) | 8.25 | 6.35 | 6.22 |
| barrier dv_x by 150 ms (m/s) | -10.67 | -8.15 | -8.56 |
| residual side crush (m) | 0.368 | 0.258 | 0.275 |
| peak side crush, dynamic (m) | 0.431 | - | - |

S14 with discrete contact (every law holds there):

#### S14, discrete contact, run 0

| quantity | Rubble | test 8560 | test 7517 |
|---|---|---|---|
| struck car mass (kg) | 1448 | 1649 | 1641 |
| barrier mass (kg) | 1368 | 1362 | 1362 |
| barrier speed (km/h) | 62.0 | 61.9 | 62.3 |
| CG lateral peak, 4 ms estimator (g) | 16.8 | 27.4 | 23.6 |
| CG lateral peak, CFC 60 (g) | 17.0 | 35.6 | 31.8 |
| CG dv_y largest (m/s) | 9.27 | 6.79 | 6.60 |
|   when (ms) | 96 | 110 | 88 |
| CG dv_y by 150 ms (m/s) | 9.15 | 6.35 | 6.22 |
| barrier dv_x by 150 ms (m/s) | -11.19 | -8.15 | -8.56 |
| residual side crush (m) | 0.051 | 0.258 | 0.275 |
| peak side crush, dynamic (m) | 0.156 | - | - |

In words (what the overlays show):
- **S11, S12 (frontal): too stiff early, too short, too bouncy.** Rubble's wall force rises
  at once to about 650 kN by 15–20 ms (the Camrys ramp through 200–300 kN and peak at 830–920 kN
  at 30 ms, when the engine reaches the firewall, with a second peak at 55 ms); the generated
  sedan stops at 44 ms (Camry 71–76) after 0.39 m of crush (0.66–0.69) and springs back at
  5.3 m/s (Camry 0.9–1.5): restitution 0.34 against 0.06–0.10, three to six times the Camrys'.
  Its residual crush, 0.20 m, is a third of theirs. At 40 km/h the same: t_m 40 ms (69–75),
  crush 0.28 m (0.48–0.49), restitution 0.48 (0.11–0.15). The front holds too hard and gives
  too much back.
- **S13 (pole): the right impulse, delivered too fast.** The lateral Δv by 150 ms is within
  10 % of the Camrys' (9.8 against 10.3–10.8 m/s), but the pole force peaks at 307 kN against
  191–197 and the pulse is over by about 100 ms (the Camrys' lasts 150 ms); the car's centre
  of mass moves 0.32 m sideways before it stops (Camry 0.44–0.50) and residual door crush
  is 0.24 m (Camry 0.38–0.39). Stiffer and shallower than a Camry's side.
- **S14 (side MDB):** with edge contact the run creates energy (section "Currently failing"),
  so its numbers are not physics; with discrete contact (which holds every law) the struck car
  leaves at 9.3 m/s across (Camry 6.6–6.8; a plastic hit on a 1,448 kg car gives 7.45), the
  barrier loses 11.2 m/s (Camry 8.2–8.6), and the door band springs back to 0.05 m of
  residual crush (Camry 0.26–0.28): an elastic side. The CG's lateral pulse lacks the Camrys'
  first 30–35 g spike at 6–7 ms (their door and sill meeting the barrier's bumper element).

## Results (2026-11-27)

Every scenario (S01–S14) in each contact setting, 8 jittered runs (`Jitter::run(0..8)`), on
the CPU reference (this Mac, M3 Max, `crash_ref`) and on CUDA (Modal, Tesla T4, driver
620.70.06, `engine-cuda`'s `crash_reference` tests, one container per test), at commit
349b82a35 (`crash-ref/cuda-fanout`, on `claude/crash-reference-suite` cb3211864). Vehicles:
the generated designs built from the current generator code by `crash_templates::template`
(`sedan`, `wagon`, `suv_4x4`, `truck_cab`, `box_trailer`) and the FMVSS 214 barrier
`mdb214` (JBeam from code); no authored car. Raw reports and logs:
`~/Development/rubble-artifacts/crash-reference/sweep-2026-11-27/`.

What it cost: the CPU sweep about 20–25 min of wall time in batches of 11 to 16 processes
(S04–S06 one process a run: their 8 runs take longer than the 5-minute bound). The CUDA
sweep about 5,500 container-seconds, about $1.15 (T4, 2 cores, 8 GiB): one invocation a
contact setting for every test but S06, S06 in chunks of 2–3 runs (a run of it is about
100 s on CUDA and host-bound: the GPU 12–28 % busy, the process at one full core; the
host's metrics, three pairs' mesh-crossing tests every 4 ms, are the likely cost, not
measured), S09's and S12–S14's continuous runs again after a cut; 31 minutes of wall time for
the invocations, most of them at once.

Verdicts: the CPU reference passes every scenario in every setting with the known failures
above. CUDA passes every scenario in discrete and edge contact; in continuous contact it
fails two, both with continuous contact engaged (two vehicles or more), on criteria the CPU
passes:

- S08 continuous: momentum across the road outside ±450 kg m/s in 3 of 8 runs (−511, 536,
  −638), and "turns the same sense as the other" in run 5 (−7.5°).
- S06 continuous: momentum along the road 16,545 kg m/s in run 4, below the band's 18,000.

These are findings, not known failures: they are not recorded as such. One-car "continuous"
rows on CUDA are edge runs with the wide passes off (`all.continuous_engaged` 0; "Currently
failing").

### CPU: every scenario x contact setting, 8 jittered runs

| Scenario | Contact | Runs | Asserted criteria | Headline metrics (min to max over the runs) |
|---|---|---|---|---|
| S01 | discrete | 8 | 5 pass | all.energy_gain_j 0.000 to 12.8; all.plastic_work_j 124,649 to 136,668; car.crush_peak 0.101 to 0.117; car.rebound_ratio 0.214 to 0.239; car.speed_end 3.882 to 4.246; all.continuous_engaged 0.000 |
| S01 | edge | 8 | 5 pass | all.energy_gain_j 0.000 to 12.8; all.plastic_work_j 124,649 to 136,668; car.crush_peak 0.101 to 0.117; car.rebound_ratio 0.214 to 0.239; car.speed_end 3.882 to 4.246; all.continuous_engaged 0.000 |
| S01 | continuous | 8 | 5 pass | all.energy_gain_j 0.000 to 90.2; all.plastic_work_j 125,641 to 137,371; car.crush_peak 0.102 to 0.120; car.rebound_ratio 0.221 to 0.258; car.speed_end 4.001 to 4.590; all.continuous_engaged 1.000 |
| S02 | discrete | 8 | 1 pass | all.energy_gain_j 0.000; all.plastic_work_j 100,621 to 111,242; car.crush_peak 0.263 to 0.280; car.rebound_ratio 0.253 to 0.277; car.speed_end 3.567 to 3.794; all.continuous_engaged 0.000 |
| S02 | edge | 8 | 1 pass | all.energy_gain_j 0.000; all.plastic_work_j 100,621 to 111,242; car.crush_peak 0.263 to 0.280; car.rebound_ratio 0.253 to 0.277; car.speed_end 3.567 to 3.794; all.continuous_engaged 0.000 |
| S02 | continuous | 8 | 1 pass | all.energy_gain_j 0.000; all.plastic_work_j 100,042 to 112,237; car.crush_peak 0.260 to 0.279; car.rebound_ratio 0.280 to 0.308; car.speed_end 3.968 to 4.211; all.continuous_engaged 1.000 |
| S03 | discrete | 8 | 7 pass | all.energy_gain_j 8.722 to 9.554; all.plastic_work_j 36,163 to 40,327; car.crush_peak 0.007 to 0.009; car.rebound_ratio 0.000; car.speed_end 3.501 to 3.675; pair.car_barrier.crossings_max 81.0 to 110.0; all.continuous_engaged 0.000 |
| S03 | edge | 8 | no energy created: known failure (8 of 8); 6 pass | all.energy_gain_j 14,112 to 211,378; all.plastic_work_j 235,195 to 2,387,892; car.crush_peak 0.010 to 0.017; car.rebound_ratio 0.000; car.speed_end 3.754 to 4.557; pair.car_barrier.crossings_max 238.0 to 439.0; all.continuous_engaged 0.000 |
| S03 | continuous | 8 | no energy created: known failure (8 of 8); 6 pass | all.energy_gain_j 6,719 to 29,661; all.plastic_work_j 140,734 to 268,067; car.crush_peak 0.007 to 0.014; car.rebound_ratio 0.000; car.speed_end 3.986 to 4.638; pair.car_barrier.crossings_max 0.000; all.continuous_engaged 1.000 |
| S04 | discrete | 8 | no energy created: known failure (1 of 8); 4 pass | all.energy_gain_j 0.000 to 221.4; all.plastic_work_j 50,502 to 62,075; car.crush_peak 0.085 to 0.107; car.rebound_ratio -0.053 to 0.002; car.speed_end 0.123 to 0.821; pair.car_trailer.crossings_max 106.0 to 146.0; all.continuous_engaged 0.000 |
| S04 | edge | 8 | no energy created: known failure (8 of 8); 4 pass | all.energy_gain_j 2,242 to 43,160; all.plastic_work_j 81,761 to 173,724; car.crush_peak 0.111 to 0.115; car.rebound_ratio -0.036 to 0.034; car.speed_end 0.084 to 0.663; pair.car_trailer.crossings_max 36.0 to 92.0; all.continuous_engaged 0.000 |
| S04 | continuous | 8 | no energy created: known failure (3 of 8); 4 pass | all.energy_gain_j 0.000 to 516.1; all.plastic_work_j 59,875 to 67,647; car.crush_peak 0.123 to 0.134; car.rebound_ratio 0.026 to 0.209; car.speed_end 0.406 to 3.222; pair.car_trailer.crossings_max 0.000; all.continuous_engaged 1.000 |
| S05 | discrete | 8 | 7 pass | all.energy_gain_j 0.000; all.plastic_work_j 115,777 to 125,866; car.crush_peak 0.057 to 0.063; car.rebound_ratio 0.000; car.speed_end 13.5 to 14.2; pair.truck_car.crossings_max 97.0 to 107.0; all.continuous_engaged 0.000 |
| S05 | edge | 8 | 7 pass | all.energy_gain_j 0.000 to 145.3; all.plastic_work_j 120,475 to 133,926; car.crush_peak 0.042 to 0.056; car.rebound_ratio 0.000; car.speed_end 13.3 to 14.3; pair.truck_car.crossings_max 4.000 to 11.0; all.continuous_engaged 0.000 |
| S05 | continuous | 8 | 7 pass | all.energy_gain_j 0.000; all.plastic_work_j 119,229 to 131,718; car.crush_peak 0.046 to 0.049; car.rebound_ratio 0.000; car.speed_end 13.2 to 14.4; pair.truck_car.crossings_max 0.000; all.continuous_engaged 1.000 |
| S06 | discrete | 8 | 5 pass | all.energy_gain_j 0.000 to 133.8; all.plastic_work_j 177,213 to 201,731; car.crush_peak 0.190 to 0.355; car.rebound_ratio 0.000; car.speed_end 0.446 to 2.673; pair.car_trailer.crossings_max 114.0 to 172.0; pair.truck_car.crossings_max 64.0 to 104.0; pair.truck_trailer.crossings_max 0.000; all.continuous_engaged 0.000 |
| S06 | edge | 8 | no energy created: known failure (8 of 8); 4 pass | all.energy_gain_j 4,107 to 39,865; all.plastic_work_j 246,913 to 357,702; car.crush_peak 0.203 to 0.620; car.rebound_ratio 0.000; car.speed_end 1.249 to 2.818; pair.car_trailer.crossings_max 55.0 to 129.0; pair.truck_car.crossings_max 9.000 to 17.0; pair.truck_trailer.crossings_max 0.000; all.continuous_engaged 0.000 |
| S06 | continuous | 8 | no energy created: known failure (1 of 8); 4 pass | all.energy_gain_j 23.6 to 991.4; all.plastic_work_j 219,178 to 247,879; car.crush_peak 0.531 to 0.782; car.rebound_ratio 0.000; car.speed_end 1.384 to 2.818; pair.car_trailer.crossings_max 0.000; pair.truck_car.crossings_max 0.000; pair.truck_trailer.crossings_max 0.000; all.continuous_engaged 1.000 |
| S07 | discrete | 8 | 3 pass | all.energy_gain_j 101.7 to 349.4; all.plastic_work_j 998,767 to 1,073,270; car.crush_peak 1.506 to 1.562; car.rebound_ratio 0.005 to 0.021; car.speed_end 0.516 to 1.840; all.continuous_engaged 0.000 |
| S07 | edge | 8 | 3 pass | all.energy_gain_j 101.7 to 349.4; all.plastic_work_j 998,767 to 1,073,270; car.crush_peak 1.506 to 1.562; car.rebound_ratio 0.005 to 0.021; car.speed_end 0.516 to 1.840; all.continuous_engaged 0.000 |
| S07 | continuous | 8 | nothing passes into the block: known failure (1 of 8); 2 pass | all.energy_gain_j 114.2 to 309.7; all.plastic_work_j 1,010,649 to 1,099,491; car.crush_peak 1.450 to 1.512; car.rebound_ratio -0.001 to 0.000; car.speed_end 0.014 to 0.116; all.continuous_engaged 1.000 |
| S08 | discrete | 8 | the common centre of mass stays still: known failure (2 of 8); 6 pass | all.energy_gain_j 44.3 to 153.6; all.plastic_work_j 122,903 to 142,224; car_a.crush_peak 0.056 to 0.089; car_a.rebound_ratio -0.003 to 0.005; car_a.speed_end 0.075 to 0.470; pair.car_a_car_b.crossings_max 127.0 to 169.0; all.continuous_engaged 0.000 |
| S08 | edge | 8 | momentum along the road: known failure (1 of 8); momentum across the road: known failure (1 of 8); the common centre of mass stays still: known failure (2 of 8); 4 pass | all.energy_gain_j 0.000 to 202.6; all.plastic_work_j 152,052 to 171,867; car_a.crush_peak 0.075 to 0.096; car_a.rebound_ratio 0.074 to 0.179; car_a.speed_end 1.193 to 2.821; pair.car_a_car_b.crossings_max 4.000 to 31.0; all.continuous_engaged 0.000 |
| S08 | continuous | 8 | no energy created: known failure (1 of 8); momentum along the road: known failure (2 of 8); the common centre of mass stays still: known failure (8 of 8); 4 pass | all.energy_gain_j 8.197 to 442.9; all.plastic_work_j 153,982 to 168,496; car_a.crush_peak 0.082 to 0.104; car_a.rebound_ratio 0.004 to 0.128; car_a.speed_end 0.661 to 2.008; pair.car_a_car_b.crossings_max 0.000; all.continuous_engaged 1.000 |
| S09 | discrete | 8 | no energy created: known failure (1 of 8); 6 pass | all.energy_gain_j 8.191 to 531.5; all.plastic_work_j 124,704 to 147,755; car.crush_peak 0.040 to 0.053; car.rebound_ratio 0.096 to 0.125; car.speed_end 1.475 to 1.944; pair.car_suv.crossings_max 101.0 to 134.0; all.continuous_engaged 0.000 |
| S09 | edge | 8 | no energy created: known failure (5 of 8); 6 pass | all.energy_gain_j 1.796 to 2,510; all.plastic_work_j 158,281 to 194,661; car.crush_peak 0.046 to 0.056; car.rebound_ratio 0.116 to 0.288; car.speed_end 1.788 to 4.330; pair.car_suv.crossings_max 43.0 to 66.0; all.continuous_engaged 0.000 |
| S09 | continuous | 8 | 7 pass | all.energy_gain_j 0.000 to 115.8; all.plastic_work_j 152,633 to 172,972; car.crush_peak 0.043 to 0.051; car.rebound_ratio 0.219 to 0.284; car.speed_end 3.319 to 4.388; pair.car_suv.crossings_max 0.000; all.continuous_engaged 1.000 |
| S10 | discrete | 8 | 5 pass | all.energy_gain_j 0.000 to 288.7; all.plastic_work_j 945,581 to 1,084,144; car.crush_peak 0.564 to 0.694; car.rebound_ratio -0.093 to 0.003; car.speed_end 0.182 to 5.283; all.continuous_engaged 0.000 |
| S10 | edge | 8 | 5 pass | all.energy_gain_j 0.000 to 288.7; all.plastic_work_j 945,581 to 1,084,144; car.crush_peak 0.564 to 0.694; car.rebound_ratio -0.093 to 0.003; car.speed_end 0.182 to 5.283; all.continuous_engaged 0.000 |
| S10 | continuous | 8 | nothing passes into the block: known failure (4 of 8); 4 pass | all.energy_gain_j 0.000 to 260.0; all.plastic_work_j 942,487 to 1,122,465; car.crush_peak 0.388 to 0.643; car.rebound_ratio -0.056 to 0.002; car.speed_end 0.142 to 3.700; all.continuous_engaged 1.000 |
| S11 | discrete | 8 | restitution within the 56 km/h sedan fleet's: known failure (8 of 8); 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 96,673 to 106,352; car.crush_peak 0.413 to 0.435; car.rebound_ratio 0.291 to 0.330; car.speed_end 4.597 to 5.065; all.continuous_engaged 0.000 |
| S11 | edge | 8 | restitution within the 56 km/h sedan fleet's: known failure (8 of 8); 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 96,673 to 106,352; car.crush_peak 0.413 to 0.435; car.rebound_ratio 0.291 to 0.330; car.speed_end 4.597 to 5.065; all.continuous_engaged 0.000 |
| S11 | continuous | 8 | restitution within the 56 km/h sedan fleet's: known failure (8 of 8); 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 97,147 to 108,587; car.crush_peak 0.409 to 0.432; car.rebound_ratio 0.282 to 0.334; car.speed_end 4.477 to 5.125; all.continuous_engaged 1.000 |
| S12 | discrete | 8 | 5 pass | all.energy_gain_j 0.000 to 2.841; all.plastic_work_j 38,181 to 43,074; car.crush_peak 0.285 to 0.302; car.rebound_ratio 0.427 to 0.451; car.speed_end 4.791 to 4.932; all.continuous_engaged 0.000 |
| S12 | edge | 8 | 5 pass | all.energy_gain_j 0.000 to 2.841; all.plastic_work_j 38,181 to 43,074; car.crush_peak 0.285 to 0.302; car.rebound_ratio 0.427 to 0.451; car.speed_end 4.791 to 4.932; all.continuous_engaged 0.000 |
| S12 | continuous | 8 | 5 pass | all.energy_gain_j 0.000 to 2.841; all.plastic_work_j 37,730 to 42,846; car.crush_peak 0.282 to 0.300; car.rebound_ratio 0.434 to 0.461; car.speed_end 4.875 to 5.034; all.continuous_engaged 1.000 |
| S13 | discrete | 8 | 6 pass | all.energy_gain_j 9.184 to 9.664; all.plastic_work_j 20,397 to 21,929; car.crush_peak 0.020 to 0.023; car.rebound_ratio -0.303 to -0.281; car.speed_end 2.893 to 3.128; all.continuous_engaged 0.000 |
| S13 | edge | 8 | 6 pass | all.energy_gain_j 9.184 to 9.664; all.plastic_work_j 20,397 to 21,929; car.crush_peak 0.020 to 0.023; car.rebound_ratio -0.303 to -0.281; car.speed_end 2.893 to 3.128; all.continuous_engaged 0.000 |
| S13 | continuous | 8 | 6 pass | all.energy_gain_j 9.184 to 15.2; all.plastic_work_j 23,221 to 24,575; car.crush_peak 0.017 to 0.023; car.rebound_ratio -0.277 to -0.246; car.speed_end 2.708 to 2.916; all.continuous_engaged 1.000 |
| S14 | discrete | 8 | 5 pass | all.energy_gain_j 3.654 to 5.031; all.plastic_work_j 46,473 to 51,378; car.crush_peak 0.013 to 0.018; car.rebound_ratio 0.000; car.speed_end 5.806 to 6.988; pair.car_barrier.crossings_max 46.0 to 76.0; all.continuous_engaged 0.000 |
| S14 | edge | 8 | no energy created: known failure (8 of 8); 4 pass | all.energy_gain_j 4,703 to 1,006,835; all.plastic_work_j 211,372 to 5,981,264; car.crush_peak 0.006 to 0.034; car.rebound_ratio 0.000; car.speed_end 5.354 to 5.921; pair.car_barrier.crossings_max 322.0 to 690.0; all.continuous_engaged 0.000 |
| S14 | continuous | 8 | no energy created: known failure (8 of 8); 4 pass | all.energy_gain_j 6,271 to 13,083; all.plastic_work_j 111,302 to 152,084; car.crush_peak 0.013 to 0.019; car.rebound_ratio 0.000; car.speed_end 5.674 to 6.492; pair.car_barrier.crossings_max 0.000; all.continuous_engaged 1.000 |

### CUDA: every scenario x contact setting, 8 jittered runs

| Scenario | Contact | Runs | Asserted criteria | Headline metrics (min to max over the runs) |
|---|---|---|---|---|
| S01 | discrete | 8 | 5 pass | all.energy_gain_j 0.000 to 12.8; all.plastic_work_j 124,649 to 136,668; car.crush_peak 0.101 to 0.117; car.rebound_ratio 0.214 to 0.239; car.speed_end 3.882 to 4.246; all.continuous_engaged 0.000 |
| S01 | edge | 8 | 5 pass | all.energy_gain_j 0.000 to 12.8; all.plastic_work_j 124,649 to 136,668; car.crush_peak 0.101 to 0.117; car.rebound_ratio 0.214 to 0.239; car.speed_end 3.882 to 4.246; all.continuous_engaged 0.000 |
| S01 | continuous | 8 | 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 131,464 to 143,289; car.crush_peak 0.108 to 0.138; car.rebound_ratio 0.217 to 0.238; car.speed_end 3.869 to 4.156; all.continuous_engaged 0.000 |
| S02 | discrete | 8 | 1 pass | all.energy_gain_j 0.000 to 1.859; all.plastic_work_j 101,139 to 111,111; car.crush_peak 0.265 to 0.278; car.rebound_ratio 0.257 to 0.275; car.speed_end 3.529 to 3.783; all.continuous_engaged 0.000 |
| S02 | edge | 8 | 1 pass | all.energy_gain_j 0.000 to 1.859; all.plastic_work_j 101,139 to 111,111; car.crush_peak 0.265 to 0.278; car.rebound_ratio 0.257 to 0.275; car.speed_end 3.529 to 3.783; all.continuous_engaged 0.000 |
| S02 | continuous | 8 | 1 pass | all.energy_gain_j 0.000; all.plastic_work_j 109,252 to 119,981; car.crush_peak 0.293 to 0.311; car.rebound_ratio 0.222 to 0.252; car.speed_end 3.152 to 3.447; all.continuous_engaged 0.000 |
| S03 | discrete | 8 | 7 pass | all.energy_gain_j 0.000; all.plastic_work_j 36,094 to 40,673; car.crush_peak 0.009 to 0.011; car.rebound_ratio 0.000; car.speed_end 3.411 to 3.721; pair.car_barrier.crossings_max 84.0 to 108.0; all.continuous_engaged 0.000 |
| S03 | edge | 8 | no energy created: known failure (8 of 8); 6 pass | all.energy_gain_j 12,426 to 27,738; all.plastic_work_j 209,844 to 574,708; car.crush_peak 0.012 to 0.023; car.rebound_ratio 0.000; car.speed_end 3.886 to 4.609; pair.car_barrier.crossings_max 208.0 to 340.0; all.continuous_engaged 0.000 |
| S03 | continuous | 8 | no energy created: known failure (8 of 8); 6 pass | all.energy_gain_j 9,628 to 22,314; all.plastic_work_j 137,333 to 208,708; car.crush_peak 0.008 to 0.013; car.rebound_ratio 0.000; car.speed_end 4.036 to 4.447; pair.car_barrier.crossings_max 0.000; all.continuous_engaged 1.000 |
| S04 | discrete | 8 | no energy created: known failure (3 of 8); 4 pass | all.energy_gain_j 0.000 to 679.2; all.plastic_work_j 52,081 to 65,218; car.crush_peak 0.094 to 0.112; car.rebound_ratio -0.012 to 0.004; car.speed_end 0.078 to 0.189; pair.car_trailer.crossings_max 105.0 to 150.0; all.continuous_engaged 0.000 |
| S04 | edge | 8 | no energy created: known failure (8 of 8); 4 pass | all.energy_gain_j 2,174 to 25,985; all.plastic_work_j 85,990 to 142,799; car.crush_peak 0.110 to 0.116; car.rebound_ratio -0.020 to 0.116; car.speed_end 0.040 to 1.818; pair.car_trailer.crossings_max 48.0 to 99.0; all.continuous_engaged 0.000 |
| S04 | continuous | 8 | no energy created: known failure (2 of 8); 4 pass | all.energy_gain_j 0.000 to 421.7; all.plastic_work_j 61,122 to 70,056; car.crush_peak 0.123 to 0.135; car.rebound_ratio 0.054 to 0.199; car.speed_end 0.856 to 3.116; pair.car_trailer.crossings_max 0.000; all.continuous_engaged 1.000 |
| S05 | discrete | 8 | 7 pass | all.energy_gain_j 0.000; all.plastic_work_j 125,882 to 138,883; car.crush_peak 0.056 to 0.062; car.rebound_ratio 0.000; car.speed_end 13.6 to 14.2; pair.truck_car.crossings_max 97.0 to 105.0; all.continuous_engaged 0.000 |
| S05 | edge | 8 | 7 pass | all.energy_gain_j 0.000; all.plastic_work_j 124,053 to 143,797; car.crush_peak 0.044 to 0.055; car.rebound_ratio 0.000; car.speed_end 13.2 to 14.1; pair.truck_car.crossings_max 0.000 to 17.0; all.continuous_engaged 0.000 |
| S05 | continuous | 8 | 7 pass | all.energy_gain_j 0.000; all.plastic_work_j 119,356 to 131,957; car.crush_peak 0.046 to 0.050; car.rebound_ratio 0.000; car.speed_end 13.7 to 14.8; pair.truck_car.crossings_max 0.000; all.continuous_engaged 1.000 |
| S06 | discrete | 8 | 5 pass | all.energy_gain_j 0.000 to 87.5; all.plastic_work_j 170,424 to 194,968; car.crush_peak 0.154 to 0.270; car.rebound_ratio 0.000; car.speed_end 1.574 to 2.516; pair.car_trailer.crossings_max 119.0 to 152.0; pair.truck_car.crossings_max 30.0 to 47.0; pair.truck_trailer.crossings_max 0.000; all.continuous_engaged 0.000 |
| S06 | edge | 8 | no energy created: known failure (8 of 8); 4 pass | all.energy_gain_j 2,506 to 28,673; all.plastic_work_j 227,508 to 879,256; car.crush_peak 0.172 to 0.226; car.rebound_ratio 0.000; car.speed_end 1.330 to 2.461; pair.car_trailer.crossings_max 52.0 to 158.0; pair.truck_car.crossings_max 9.000 to 12.0; pair.truck_trailer.crossings_max 0.000; all.continuous_engaged 0.000 |
| S06 | continuous | 8 | **momentum along the road: FAIL (1 of 8)**; no energy created: known failure (3 of 8); 3 pass | all.energy_gain_j 0.000 to 4,938; all.plastic_work_j 144,801 to 182,338; car.crush_peak 0.185 to 0.213; car.rebound_ratio 0.000; car.speed_end 0.406 to 1.847; pair.car_trailer.crossings_max 0.000; pair.truck_car.crossings_max 0.000; pair.truck_trailer.crossings_max 0.000; all.continuous_engaged 1.000 |
| S07 | discrete | 8 | 3 pass | all.energy_gain_j 159.4 to 311.4; all.plastic_work_j 993,619 to 1,074,863; car.crush_peak 1.503 to 1.564; car.rebound_ratio 0.004 to 0.016; car.speed_end 0.279 to 1.441; all.continuous_engaged 0.000 |
| S07 | edge | 8 | 3 pass | all.energy_gain_j 159.4 to 311.4; all.plastic_work_j 993,619 to 1,074,863; car.crush_peak 1.503 to 1.564; car.rebound_ratio 0.004 to 0.016; car.speed_end 0.279 to 1.441; all.continuous_engaged 0.000 |
| S07 | continuous | 8 | 3 pass | all.energy_gain_j 0.000 to 172.1; all.plastic_work_j 1,068,718 to 1,155,917; car.crush_peak 1.535 to 1.625; car.rebound_ratio -0.001 to 0.007; car.speed_end 0.132 to 0.646; all.continuous_engaged 0.000 |
| S08 | discrete | 8 | 7 pass | all.energy_gain_j 0.000 to 129.3; all.plastic_work_j 118,084 to 146,787; car_a.crush_peak 0.055 to 0.090; car_a.rebound_ratio -0.002 to 0.010; car_a.speed_end 0.029 to 0.331; pair.car_a_car_b.crossings_max 124.0 to 160.0; all.continuous_engaged 0.000 |
| S08 | edge | 8 | momentum along the road: known failure (2 of 8); the common centre of mass stays still: known failure (6 of 8); 5 pass | all.energy_gain_j 0.000 to 170.2; all.plastic_work_j 154,238 to 171,147; car_a.crush_peak 0.075 to 0.106; car_a.rebound_ratio 0.045 to 0.139; car_a.speed_end 0.840 to 2.243; pair.car_a_car_b.crossings_max 4.000 to 30.0; all.continuous_engaged 0.000 |
| S08 | continuous | 8 | **momentum across the road: FAIL (3 of 8)**; **turns the same sense as the other: FAIL (1 of 8)**; no energy created: known failure (1 of 8); momentum along the road: known failure (2 of 8); the common centre of mass stays still: known failure (8 of 8); 2 pass | all.energy_gain_j 0.000 to 349.5; all.plastic_work_j 153,601 to 171,829; car_a.crush_peak 0.084 to 0.104; car_a.rebound_ratio 0.003 to 0.111; car_a.speed_end 0.602 to 1.815; pair.car_a_car_b.crossings_max 0.000; all.continuous_engaged 1.000 |
| S09 | discrete | 8 | 7 pass | all.energy_gain_j 14.8 to 144.5; all.plastic_work_j 110,954 to 143,060; car.crush_peak 0.035 to 0.051; car.rebound_ratio 0.087 to 0.139; car.speed_end 1.336 to 2.168; pair.car_suv.crossings_max 75.0 to 120.0; all.continuous_engaged 0.000 |
| S09 | edge | 8 | no energy created: known failure (6 of 8); 6 pass | all.energy_gain_j 0.000 to 2,910; all.plastic_work_j 169,101 to 209,121; car.crush_peak 0.046 to 0.065; car.rebound_ratio 0.094 to 0.254; car.speed_end 1.405 to 3.942; pair.car_suv.crossings_max 35.0 to 72.0; all.continuous_engaged 0.000 |
| S09 | continuous | 8 | 7 pass | all.energy_gain_j 0.000 to 5.888; all.plastic_work_j 150,317 to 175,114; car.crush_peak 0.044 to 0.055; car.rebound_ratio 0.223 to 0.331; car.speed_end 3.398 to 5.036; pair.car_suv.crossings_max 0.000; all.continuous_engaged 1.000 |
| S10 | discrete | 8 | 5 pass | all.energy_gain_j 0.000 to 302.9; all.plastic_work_j 949,321 to 1,076,519; car.crush_peak 0.597 to 0.792; car.rebound_ratio -0.081 to -0.001; car.speed_end 0.087 to 4.566; all.continuous_engaged 0.000 |
| S10 | edge | 8 | 5 pass | all.energy_gain_j 0.000 to 302.9; all.plastic_work_j 949,321 to 1,076,519; car.crush_peak 0.597 to 0.792; car.rebound_ratio -0.081 to -0.001; car.speed_end 0.087 to 4.566; all.continuous_engaged 0.000 |
| S10 | continuous | 8 | 5 pass | all.energy_gain_j 0.000 to 13.0; all.plastic_work_j 1,059,585 to 1,228,207; car.crush_peak 0.797 to 0.884; car.rebound_ratio -0.060 to 0.004; car.speed_end 0.260 to 4.196; all.continuous_engaged 0.000 |
| S11 | discrete | 8 | restitution within the 56 km/h sedan fleet's: known failure (8 of 8); 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 96,673 to 106,352; car.crush_peak 0.413 to 0.435; car.rebound_ratio 0.291 to 0.330; car.speed_end 4.597 to 5.065; all.continuous_engaged 0.000 |
| S11 | edge | 8 | restitution within the 56 km/h sedan fleet's: known failure (8 of 8); 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 96,673 to 106,352; car.crush_peak 0.413 to 0.435; car.rebound_ratio 0.291 to 0.330; car.speed_end 4.597 to 5.065; all.continuous_engaged 0.000 |
| S11 | continuous | 8 | restitution within the 56 km/h sedan fleet's: known failure (6 of 8); 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 101,275 to 114,053; car.crush_peak 0.426 to 0.450; car.rebound_ratio 0.218 to 0.283; car.speed_end 3.474 to 4.369; all.continuous_engaged 0.000 |
| S12 | discrete | 8 | 5 pass | all.energy_gain_j 0.000 to 2.841; all.plastic_work_j 38,181 to 43,074; car.crush_peak 0.285 to 0.302; car.rebound_ratio 0.427 to 0.451; car.speed_end 4.791 to 4.932; all.continuous_engaged 0.000 |
| S12 | edge | 8 | 5 pass | all.energy_gain_j 0.000 to 2.841; all.plastic_work_j 38,181 to 43,074; car.crush_peak 0.285 to 0.302; car.rebound_ratio 0.427 to 0.451; car.speed_end 4.791 to 4.932; all.continuous_engaged 0.000 |
| S12 | continuous | 8 | 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 39,834 to 45,435; car.crush_peak 0.298 to 0.312; car.rebound_ratio 0.392 to 0.455; car.speed_end 4.405 to 4.981; all.continuous_engaged 0.000 |
| S13 | discrete | 8 | 6 pass | all.energy_gain_j 9.184 to 9.664; all.plastic_work_j 20,397 to 21,929; car.crush_peak 0.020 to 0.023; car.rebound_ratio -0.303 to -0.281; car.speed_end 2.893 to 3.128; all.continuous_engaged 0.000 |
| S13 | edge | 8 | 6 pass | all.energy_gain_j 9.184 to 9.664; all.plastic_work_j 20,397 to 21,929; car.crush_peak 0.020 to 0.023; car.rebound_ratio -0.303 to -0.281; car.speed_end 2.893 to 3.128; all.continuous_engaged 0.000 |
| S13 | continuous | 8 | 6 pass | all.energy_gain_j 0.000; all.plastic_work_j 22,618 to 24,607; car.crush_peak 0.021 to 0.022; car.rebound_ratio -0.292 to -0.273; car.speed_end 2.826 to 3.014; all.continuous_engaged 0.000 |
| S14 | discrete | 8 | 5 pass | all.energy_gain_j 0.000; all.plastic_work_j 47,088 to 51,299; car.crush_peak 0.012 to 0.018; car.rebound_ratio 0.000; car.speed_end 5.876 to 7.278; pair.car_barrier.crossings_max 48.0 to 75.0; all.continuous_engaged 0.000 |
| S14 | edge | 8 | no energy created: known failure (8 of 8); 4 pass | all.energy_gain_j 6,668 to 20,389; all.plastic_work_j 231,302 to 475,575; car.crush_peak 0.010 to 0.034; car.rebound_ratio 0.000; car.speed_end 4.771 to 6.084; pair.car_barrier.crossings_max 305.0 to 394.0; all.continuous_engaged 0.000 |
| S14 | continuous | 8 | no energy created: known failure (8 of 8); 4 pass | all.energy_gain_j 7,491 to 13,325; all.plastic_work_j 109,548 to 146,049; car.crush_peak 0.014 to 0.018; car.rebound_ratio 0.000; car.speed_end 6.235 to 6.810; pair.car_barrier.crossings_max 0.000; all.continuous_engaged 1.000 |

### CUDA against the CPU reference (min to max over the runs: CPU / CUDA)

| Scenario | Contact | Verdicts | Asserted criteria's values (CPU / CUDA) | Headline metrics (CPU / CUDA) |
|---|---|---|---|---|
| S01 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 12.8 / 0.000 to 12.8; nothing passes into the block: 0.000 / 0.000; the struck side takes more: 4.879 to 6.301 / 4.879 to 6.301; the car turns toward the block: 2.017 to 4.218 / 2.017 to 4.218 | all.energy_gain_j 0.000 to 12.8 / 0.000 to 12.8; all.plastic_work_j 124,649 to 136,668 / 124,649 to 136,668; car.crush_peak 0.101 to 0.117 / 0.101 to 0.117; car.rebound_ratio 0.214 to 0.239 / 0.214 to 0.239; car.speed_end 3.882 to 4.246 / 3.882 to 4.246; all.continuous_engaged 0.000 / 0.000 |
| S01 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 12.8 / 0.000 to 12.8; nothing passes into the block: 0.000 / 0.000; the struck side takes more: 4.879 to 6.301 / 4.879 to 6.301; the car turns toward the block: 2.017 to 4.218 / 2.017 to 4.218 | all.energy_gain_j 0.000 to 12.8 / 0.000 to 12.8; all.plastic_work_j 124,649 to 136,668 / 124,649 to 136,668; car.crush_peak 0.101 to 0.117 / 0.101 to 0.117; car.rebound_ratio 0.214 to 0.239 / 0.214 to 0.239; car.speed_end 3.882 to 4.246 / 3.882 to 4.246; all.continuous_engaged 0.000 / 0.000 |
| S01 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 90.2 / 0.000; nothing passes into the block: 0.000 / 0.000; the struck side takes more: 5.128 to 6.007 / 4.526 to 5.104; the car turns toward the block: 2.172 to 5.112 / 3.040 to 4.203 | all.energy_gain_j 0.000 to 90.2 / 0.000; all.plastic_work_j 125,641 to 137,371 / 131,464 to 143,289; car.crush_peak 0.102 to 0.120 / 0.108 to 0.138; car.rebound_ratio 0.221 to 0.258 / 0.217 to 0.238; car.speed_end 4.001 to 4.590 / 3.869 to 4.156; all.continuous_engaged 1.000 / 0.000 |
| S02 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000 | all.energy_gain_j 0.000 / 0.000 to 1.859; all.plastic_work_j 100,621 to 111,242 / 101,139 to 111,111; car.crush_peak 0.263 to 0.280 / 0.265 to 0.278; car.rebound_ratio 0.253 to 0.277 / 0.257 to 0.275; car.speed_end 3.567 to 3.794 / 3.529 to 3.783; all.continuous_engaged 0.000 / 0.000 |
| S02 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000 | all.energy_gain_j 0.000 / 0.000 to 1.859; all.plastic_work_j 100,621 to 111,242 / 101,139 to 111,111; car.crush_peak 0.263 to 0.280 / 0.265 to 0.278; car.rebound_ratio 0.253 to 0.277 / 0.257 to 0.275; car.speed_end 3.567 to 3.794 / 3.529 to 3.783; all.continuous_engaged 0.000 / 0.000 |
| S02 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000 | all.energy_gain_j 0.000 / 0.000; all.plastic_work_j 100,042 to 112,237 / 109,252 to 119,981; car.crush_peak 0.260 to 0.279 / 0.293 to 0.311; car.rebound_ratio 0.280 to 0.308 / 0.222 to 0.252; car.speed_end 3.968 to 4.211 / 3.152 to 3.447; all.continuous_engaged 1.000 / 0.000 |
| S03 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 8.722 to 9.554 / 0.000; momentum along the barrier's travel: -10,168 to -9,486 / -10,206 to -9,494; the car leaves no faster than an elastic hit: 8.329 to 8.621 / 8.332 to 8.601; the car ends moving the barrier's way: -3.633 to -3.479 / -3.695 to -3.394; the barrier slows, never speeds up or rebounds past elastic: -10.6 to -10.4 / -10.6 to -10.3; the struck side absorbs more than the far side: 2.900 to 3.442 / 2.617 to 3.231 | all.energy_gain_j 8.722 to 9.554 / 0.000; all.plastic_work_j 36,163 to 40,327 / 36,094 to 40,673; car.crush_peak 0.007 to 0.009 / 0.009 to 0.011; car.rebound_ratio 0.000 / 0.000; car.speed_end 3.501 to 3.675 / 3.411 to 3.721; pair.car_barrier.crossings_max 81.0 to 110.0 / 84.0 to 108.0; all.continuous_engaged 0.000 / 0.000 |
| S03 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 14,112 to 211,378 / 12,426 to 27,738; momentum along the barrier's travel: -10,126 to -8,876 / -9,960 to -9,120; the car leaves no faster than an elastic hit: 6.856 to 8.268 / 6.922 to 8.426; the car ends moving the barrier's way: -4.499 to -3.716 / -4.593 to -3.875; the barrier slows, never speeds up or rebounds past elastic: -11.8 to -11.0 / -11.9 to -11.0; the struck side absorbs more than the far side: 41.0 to 87.0 / 36.8 to 81.0 | all.energy_gain_j 14,112 to 211,378 / 12,426 to 27,738; all.plastic_work_j 235,195 to 2,387,892 / 209,844 to 574,708; car.crush_peak 0.010 to 0.017 / 0.012 to 0.023; car.rebound_ratio 0.000 / 0.000; car.speed_end 3.754 to 4.557 / 3.886 to 4.609; pair.car_barrier.crossings_max 238.0 to 439.0 / 208.0 to 340.0; all.continuous_engaged 0.000 / 0.000 |
| S03 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 6,719 to 29,661 / 9,628 to 22,314; momentum along the barrier's travel: -10,572 to -9,831 / -10,448 to -10,074; the car leaves no faster than an elastic hit: 7.654 to 8.513 / 7.872 to 8.448; the car ends moving the barrier's way: -4.632 to -3.982 / -4.332 to -4.027; the barrier slows, never speeds up or rebounds past elastic: -11.3 to -10.6 / -11.3 to -10.4; the struck side absorbs more than the far side: 27.6 to 45.5 / 27.0 to 41.5 | all.energy_gain_j 6,719 to 29,661 / 9,628 to 22,314; all.plastic_work_j 140,734 to 268,067 / 137,333 to 208,708; car.crush_peak 0.007 to 0.014 / 0.008 to 0.013; car.rebound_ratio 0.000 / 0.000; car.speed_end 3.986 to 4.638 / 4.036 to 4.447; pair.car_barrier.crossings_max 0.000 / 0.000; all.continuous_engaged 1.000 / 1.000 |
| S04 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 221.4 / 0.000 to 679.2; momentum along the road: 1,205 to 2,809 / 826.7 to 2,210; the trailer leaves no faster than an elastic hit sends it: 1.016 to 1.311 / 0.920 to 1.216; the car ends no faster forward than it came: -0.036 to 0.820 / -0.063 to 0.185 | all.energy_gain_j 0.000 to 221.4 / 0.000 to 679.2; all.plastic_work_j 50,502 to 62,075 / 52,081 to 65,218; car.crush_peak 0.085 to 0.107 / 0.094 to 0.112; car.rebound_ratio -0.053 to 0.002 / -0.012 to 0.004; car.speed_end 0.123 to 0.821 / 0.078 to 0.189; pair.car_trailer.crossings_max 106.0 to 146.0 / 105.0 to 150.0; all.continuous_engaged 0.000 / 0.000 |
| S04 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 2,242 to 43,160 / 2,174 to 25,985; momentum along the road: 1,129 to 3,342 / 1,759 to 2,775; the trailer leaves no faster than an elastic hit sends it: 1.388 to 1.738 / 1.411 to 1.810; the car ends no faster forward than it came: -0.530 to 0.572 / -1.807 to 0.309 | all.energy_gain_j 2,242 to 43,160 / 2,174 to 25,985; all.plastic_work_j 81,761 to 173,724 / 85,990 to 142,799; car.crush_peak 0.111 to 0.115 / 0.110 to 0.116; car.rebound_ratio -0.036 to 0.034 / -0.020 to 0.116; car.speed_end 0.084 to 0.663 / 0.040 to 1.818; pair.car_trailer.crossings_max 36.0 to 92.0 / 48.0 to 99.0; all.continuous_engaged 0.000 / 0.000 |
| S04 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 516.1 / 0.000 to 421.7; momentum along the road: -244.2 to 2,248 / -356.4 to 1,660; the trailer leaves no faster than an elastic hit sends it: 1.758 to 1.848 / 1.762 to 1.835; the car ends no faster forward than it came: -3.221 to -0.396 / -3.115 to -0.856 | all.energy_gain_j 0.000 to 516.1 / 0.000 to 421.7; all.plastic_work_j 59,875 to 67,647 / 61,122 to 70,056; car.crush_peak 0.123 to 0.134 / 0.123 to 0.135; car.rebound_ratio 0.026 to 0.209 / 0.054 to 0.199; car.speed_end 0.406 to 3.222 / 0.856 to 3.116; pair.car_trailer.crossings_max 0.000 / 0.000; all.continuous_engaged 1.000 / 1.000 |
| S05 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 / 0.000; momentum along the truck's travel: -130,249 to -122,641 / -126,564 to -121,007; the car ends moving the truck's way: -14.2 to -13.5 / -14.2 to -13.6; the truck never reverses: 13.7 to 14.7 / 13.5 to 14.1; the car leaves no faster than an elastic hit: 18.5 to 19.2 / 16.2 to 17.1; the struck side absorbs more than the far side: 3.641 to 3.933 / 3.528 to 3.895 | all.energy_gain_j 0.000 / 0.000; all.plastic_work_j 115,777 to 125,866 / 125,882 to 138,883; car.crush_peak 0.057 to 0.063 / 0.056 to 0.062; car.rebound_ratio 0.000 / 0.000; car.speed_end 13.5 to 14.2 / 13.6 to 14.2; pair.truck_car.crossings_max 97.0 to 107.0 / 97.0 to 105.0; all.continuous_engaged 0.000 / 0.000 |
| S05 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 145.3 / 0.000; momentum along the truck's travel: -126,769 to -120,562 / -126,041 to -116,586; the car ends moving the truck's way: -14.3 to -13.3 / -14.1 to -13.2; the truck never reverses: 13.4 to 14.2 / 13.0 to 14.1; the car leaves no faster than an elastic hit: 19.2 to 19.9 / 16.8 to 19.9; the struck side absorbs more than the far side: 4.519 to 5.131 / 4.265 to 4.948 | all.energy_gain_j 0.000 to 145.3 / 0.000; all.plastic_work_j 120,475 to 133,926 / 124,053 to 143,797; car.crush_peak 0.042 to 0.056 / 0.044 to 0.055; car.rebound_ratio 0.000 / 0.000; car.speed_end 13.3 to 14.3 / 13.2 to 14.1; pair.truck_car.crossings_max 4.000 to 11.0 / 0.000 to 17.0; all.continuous_engaged 0.000 / 0.000 |
| S05 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 / 0.000; momentum along the truck's travel: -128,283 to -123,980 / -128,342 to -123,205; the car ends moving the truck's way: -14.4 to -13.2 / -14.8 to -13.7; the truck never reverses: 13.9 to 14.4 / 13.8 to 14.4; the car leaves no faster than an elastic hit: 19.6 to 20.5 / 19.5 to 20.5; the struck side absorbs more than the far side: 4.419 to 5.087 / 4.254 to 5.014 | all.energy_gain_j 0.000 / 0.000; all.plastic_work_j 119,229 to 131,718 / 119,356 to 131,957; car.crush_peak 0.046 to 0.049 / 0.046 to 0.050; car.rebound_ratio 0.000 / 0.000; car.speed_end 13.2 to 14.4 / 13.7 to 14.8; pair.truck_car.crossings_max 0.000 / 0.000; all.continuous_engaged 1.000 / 1.000 |
| S06 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 133.8 / 0.000 to 87.5; momentum along the road: 22,894 to 59,053 / 36,771 to 50,685; the car stays ahead of the truck: 4.170 to 4.432 / 3.467 to 4.002; the car stays behind the trailer: 7.526 to 8.545 / 8.363 to 8.749 | all.energy_gain_j 0.000 to 133.8 / 0.000 to 87.5; all.plastic_work_j 177,213 to 201,731 / 170,424 to 194,968; car.crush_peak 0.190 to 0.355 / 0.154 to 0.270; car.rebound_ratio 0.000 / 0.000; car.speed_end 0.446 to 2.673 / 1.574 to 2.516; pair.car_trailer.crossings_max 114.0 to 172.0 / 119.0 to 152.0; pair.truck_car.crossings_max 64.0 to 104.0 / 30.0 to 47.0; pair.truck_trailer.crossings_max 0.000 / 0.000; all.continuous_engaged 0.000 / 0.000 |
| S06 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 4,107 to 39,865 / 2,506 to 28,673; momentum along the road: 36,672 to 56,879 / 24,105 to 47,018; the car stays ahead of the truck: 4.342 to 4.614 / 3.863 to 4.190; the car stays behind the trailer: 7.806 to 8.893 / 7.739 to 8.854 | all.energy_gain_j 4,107 to 39,865 / 2,506 to 28,673; all.plastic_work_j 246,913 to 357,702 / 227,508 to 879,256; car.crush_peak 0.203 to 0.620 / 0.172 to 0.226; car.rebound_ratio 0.000 / 0.000; car.speed_end 1.249 to 2.818 / 1.330 to 2.461; pair.car_trailer.crossings_max 55.0 to 129.0 / 52.0 to 158.0; pair.truck_car.crossings_max 9.000 to 17.0 / 9.000 to 12.0; pair.truck_trailer.crossings_max 0.000 / 0.000; all.continuous_engaged 0.000 / 0.000 |
| S06 | continuous | **disagree on 1**; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 23.6 to 991.4 / 0.000 to 4,938; **momentum along the road: 53,437 to 60,238 / 16,545 to 46,570**; the car stays ahead of the truck: 4.146 to 4.326 / 4.567 to 4.720; the car stays behind the trailer: 9.032 to 9.176 / 7.408 to 7.925 | all.energy_gain_j 23.6 to 991.4 / 0.000 to 4,938; all.plastic_work_j 219,178 to 247,879 / 144,801 to 182,338; car.crush_peak 0.531 to 0.782 / 0.185 to 0.213; car.rebound_ratio 0.000 / 0.000; car.speed_end 1.384 to 2.818 / 0.406 to 1.847; pair.car_trailer.crossings_max 0.000 / 0.000; pair.truck_car.crossings_max 0.000 / 0.000; pair.truck_trailer.crossings_max 0.000 / 0.000; all.continuous_engaged 1.000 / 1.000 |
| S07 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 101.7 to 349.4 / 159.4 to 311.4; nothing passes into the block: 0.000 to 0.014 / 0.000 to 0.014 | all.energy_gain_j 101.7 to 349.4 / 159.4 to 311.4; all.plastic_work_j 998,767 to 1,073,270 / 993,619 to 1,074,863; car.crush_peak 1.506 to 1.562 / 1.503 to 1.564; car.rebound_ratio 0.005 to 0.021 / 0.004 to 0.016; car.speed_end 0.516 to 1.840 / 0.279 to 1.441; all.continuous_engaged 0.000 / 0.000 |
| S07 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 101.7 to 349.4 / 159.4 to 311.4; nothing passes into the block: 0.000 to 0.014 / 0.000 to 0.014 | all.energy_gain_j 101.7 to 349.4 / 159.4 to 311.4; all.plastic_work_j 998,767 to 1,073,270 / 993,619 to 1,074,863; car.crush_peak 1.506 to 1.562 / 1.503 to 1.564; car.rebound_ratio 0.005 to 0.021 / 0.004 to 0.016; car.speed_end 0.516 to 1.840 / 0.279 to 1.441; all.continuous_engaged 0.000 / 0.000 |
| S07 | continuous | agree (1 known failure(s) on one only); 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 114.2 to 309.7 / 0.000 to 172.1; nothing passes into the block: 0.009 to 0.050 / 0.000 to 0.023 | all.energy_gain_j 114.2 to 309.7 / 0.000 to 172.1; all.plastic_work_j 1,010,649 to 1,099,491 / 1,068,718 to 1,155,917; car.crush_peak 1.450 to 1.512 / 1.535 to 1.625; car.rebound_ratio -0.001 to 0.000 / -0.001 to 0.007; car.speed_end 0.014 to 0.116 / 0.132 to 0.646; all.continuous_engaged 1.000 / 0.000 |
| S08 | discrete | agree (1 known failure(s) on one only); 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 44.3 to 153.6 / 0.000 to 129.3; momentum along the road: -202.8 to 114.7 / -59.0 to 153.7; momentum across the road: -75.4 to 38.7 / -26.3 to 120.7; the overlapping side takes more: 9.786 to 19.9 / 8.986 to 19.7; turns the same sense as the other: 4.884 to 9.607 / 6.287 to 11.0; the overlapping side takes more: 11.5 to 24.7 / 9.919 to 29.3; turns the same sense as the other: 5.849 to 11.2 / 4.175 to 11.1; the common centre of mass stays still: 0.060 to 0.461 / 0.014 to 0.288 | all.energy_gain_j 44.3 to 153.6 / 0.000 to 129.3; all.plastic_work_j 122,903 to 142,224 / 118,084 to 146,787; car_a.crush_peak 0.056 to 0.089 / 0.055 to 0.090; car_a.rebound_ratio -0.003 to 0.005 / -0.002 to 0.010; car_a.speed_end 0.075 to 0.470 / 0.029 to 0.331; pair.car_a_car_b.crossings_max 127.0 to 169.0 / 124.0 to 160.0; all.continuous_engaged 0.000 / 0.000 |
| S08 | edge | agree (1 known failure(s) on one only); 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 202.6 / 0.000 to 170.2; momentum along the road: -133.6 to 642.8 / -595.1 to 676.0; momentum across the road: -283.7 to 519.0 / -235.7 to 362.7; the overlapping side takes more: 9.761 to 11.2 / 8.979 to 10.7; turns the same sense as the other: 4.969 to 18.0 / 6.053 to 16.3; the overlapping side takes more: 9.679 to 10.9 / 8.408 to 10.3; turns the same sense as the other: 4.414 to 15.8 / 8.194 to 17.2; the common centre of mass stays still: 0.144 to 0.409 / 0.212 to 0.470 | all.energy_gain_j 0.000 to 202.6 / 0.000 to 170.2; all.plastic_work_j 152,052 to 171,867 / 154,238 to 171,147; car_a.crush_peak 0.075 to 0.096 / 0.075 to 0.106; car_a.rebound_ratio 0.074 to 0.179 / 0.045 to 0.139; car_a.speed_end 1.193 to 2.821 / 0.840 to 2.243; pair.car_a_car_b.crossings_max 4.000 to 31.0 / 4.000 to 30.0; all.continuous_engaged 0.000 / 0.000 |
| S08 | continuous | **disagree on 2**; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 8.197 to 442.9 / 0.000 to 349.5; momentum along the road: -410.7 to 568.1 / -462.5 to 581.9; **momentum across the road: -289.0 to 188.9 / -638.4 to 535.6**; the overlapping side takes more: 5.761 to 7.822 / 5.295 to 7.457; **turns the same sense as the other: 3.324 to 14.0 / 3.568 to 12.2**; the overlapping side takes more: 5.952 to 8.471 / 5.585 to 7.732; **turns the same sense as the other: 4.470 to 15.3 / -7.491 to 10.7**; the common centre of mass stays still: 0.399 to 0.661 / 0.402 to 0.622 | all.energy_gain_j 8.197 to 442.9 / 0.000 to 349.5; all.plastic_work_j 153,982 to 168,496 / 153,601 to 171,829; car_a.crush_peak 0.082 to 0.104 / 0.084 to 0.104; car_a.rebound_ratio 0.004 to 0.128 / 0.003 to 0.111; car_a.speed_end 0.661 to 2.008 / 0.602 to 1.815; pair.car_a_car_b.crossings_max 0.000 / 0.000; all.continuous_engaged 1.000 / 1.000 |
| S09 | discrete | agree (1 known failure(s) on one only); 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 8.191 to 531.5 / 14.8 to 144.5; momentum along the road: -6,517 to -4,601 / -7,164 to -4,199; momentum across the road: -1,677 to -1,009 / -1,657 to -1,148; the car is driven back: -1.915 to -1.450 / -2.138 to -1.300; the car's overlapping side takes more: 6.909 to 10.6 / 6.337 to 10.9; the SUV's overlapping side takes more: 37.9 to 182.5 / 33.4 to 84.5 | all.energy_gain_j 8.191 to 531.5 / 14.8 to 144.5; all.plastic_work_j 124,704 to 147,755 / 110,954 to 143,060; car.crush_peak 0.040 to 0.053 / 0.035 to 0.051; car.rebound_ratio 0.096 to 0.125 / 0.087 to 0.139; car.speed_end 1.475 to 1.944 / 1.336 to 2.168; pair.car_suv.crossings_max 101.0 to 134.0 / 75.0 to 120.0; all.continuous_engaged 0.000 / 0.000 |
| S09 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 1.796 to 2,510 / 0.000 to 2,910; momentum along the road: -6,896 to -5,664 / -7,085 to -5,761; momentum across the road: -1,573 to -26.9 / -1,242 to -726.4; the car is driven back: -4.330 to -1.774 / -3.905 to -1.403; the car's overlapping side takes more: 6.892 to 9.759 / 7.510 to 11.2; the SUV's overlapping side takes more: 17.9 to 58.4 / 16.2 to 77.1 | all.energy_gain_j 1.796 to 2,510 / 0.000 to 2,910; all.plastic_work_j 158,281 to 194,661 / 169,101 to 209,121; car.crush_peak 0.046 to 0.056 / 0.046 to 0.065; car.rebound_ratio 0.116 to 0.288 / 0.094 to 0.254; car.speed_end 1.788 to 4.330 / 1.405 to 3.942; pair.car_suv.crossings_max 43.0 to 66.0 / 35.0 to 72.0; all.continuous_engaged 0.000 / 0.000 |
| S09 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 115.8 / 0.000 to 5.888; momentum along the road: -7,078 to -6,273 / -7,437 to -6,374; momentum across the road: -1,239 to 83.8 / -1,190 to 12.8; the car is driven back: -4.387 to -3.289 / -5.020 to -3.385; the car's overlapping side takes more: 5.248 to 7.354 / 5.492 to 7.732; the SUV's overlapping side takes more: 20.2 to 31.9 / 18.9 to 39.0 | all.energy_gain_j 0.000 to 115.8 / 0.000 to 5.888; all.plastic_work_j 152,633 to 172,972 / 150,317 to 175,114; car.crush_peak 0.043 to 0.051 / 0.044 to 0.055; car.rebound_ratio 0.219 to 0.284 / 0.223 to 0.331; car.speed_end 3.319 to 4.388 / 3.398 to 5.036; pair.car_suv.crossings_max 0.000 / 0.000; all.continuous_engaged 1.000 / 1.000 |
| S10 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 288.7 / 0.000 to 302.9; nothing passes into the block: 0.000 to 0.009 / 0.000 to 0.008; the struck side takes more: 2.622 to 2.866 / 2.620 to 2.814; the car turns toward the block: 233.8 to 320.7 / 241.4 to 294.0 | all.energy_gain_j 0.000 to 288.7 / 0.000 to 302.9; all.plastic_work_j 945,581 to 1,084,144 / 949,321 to 1,076,519; car.crush_peak 0.564 to 0.694 / 0.597 to 0.792; car.rebound_ratio -0.093 to 0.003 / -0.081 to -0.001; car.speed_end 0.182 to 5.283 / 0.087 to 4.566; all.continuous_engaged 0.000 / 0.000 |
| S10 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 288.7 / 0.000 to 302.9; nothing passes into the block: 0.000 to 0.009 / 0.000 to 0.008; the struck side takes more: 2.622 to 2.866 / 2.620 to 2.814; the car turns toward the block: 233.8 to 320.7 / 241.4 to 294.0 | all.energy_gain_j 0.000 to 288.7 / 0.000 to 302.9; all.plastic_work_j 945,581 to 1,084,144 / 949,321 to 1,076,519; car.crush_peak 0.564 to 0.694 / 0.597 to 0.792; car.rebound_ratio -0.093 to 0.003 / -0.081 to -0.001; car.speed_end 0.182 to 5.283 / 0.087 to 4.566; all.continuous_engaged 0.000 / 0.000 |
| S10 | continuous | agree (1 known failure(s) on one only); 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 260.0 / 0.000 to 13.0; nothing passes into the block: 0.001 to 0.147 / 0.000; the struck side takes more: 2.621 to 3.133 / 2.743 to 2.992; the car turns toward the block: 188.3 to 377.1 / 93.3 to 332.3 | all.energy_gain_j 0.000 to 260.0 / 0.000 to 13.0; all.plastic_work_j 942,487 to 1,122,465 / 1,059,585 to 1,228,207; car.crush_peak 0.388 to 0.643 / 0.797 to 0.884; car.rebound_ratio -0.056 to 0.002 / -0.060 to 0.004; car.speed_end 0.142 to 3.700 / 0.260 to 4.196; all.continuous_engaged 1.000 / 0.000 |
| S11 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 / 0.000; nothing passes into the wall: 0.000 / 0.000; no rebound past an elastic one: 0.291 to 0.330 / 0.291 to 0.330; the wall stops the car: 44.1 to 46.1 / 44.1 to 46.1; restitution within the 56 km/h sedan fleet's: 0.302 to 0.346 / 0.302 to 0.346 | all.energy_gain_j 0.000 / 0.000; all.plastic_work_j 96,673 to 106,352 / 96,673 to 106,352; car.crush_peak 0.413 to 0.435 / 0.413 to 0.435; car.rebound_ratio 0.291 to 0.330 / 0.291 to 0.330; car.speed_end 4.597 to 5.065 / 4.597 to 5.065; all.continuous_engaged 0.000 / 0.000 |
| S11 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 / 0.000; nothing passes into the wall: 0.000 / 0.000; no rebound past an elastic one: 0.291 to 0.330 / 0.291 to 0.330; the wall stops the car: 44.1 to 46.1 / 44.1 to 46.1; restitution within the 56 km/h sedan fleet's: 0.302 to 0.346 / 0.302 to 0.346 | all.energy_gain_j 0.000 / 0.000; all.plastic_work_j 96,673 to 106,352 / 96,673 to 106,352; car.crush_peak 0.413 to 0.435 / 0.413 to 0.435; car.rebound_ratio 0.291 to 0.330 / 0.291 to 0.330; car.speed_end 4.597 to 5.065 / 4.597 to 5.065; all.continuous_engaged 0.000 / 0.000 |
| S11 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 / 0.000; nothing passes into the wall: 0.000 / 0.000; no rebound past an elastic one: 0.282 to 0.334 / 0.218 to 0.283; the wall stops the car: 43.3 to 47.2 / 45.5 to 52.2; restitution within the 56 km/h sedan fleet's: 0.295 to 0.356 / 0.224 to 0.297 | all.energy_gain_j 0.000 / 0.000; all.plastic_work_j 97,147 to 108,587 / 101,275 to 114,053; car.crush_peak 0.409 to 0.432 / 0.426 to 0.450; car.rebound_ratio 0.282 to 0.334 / 0.218 to 0.283; car.speed_end 4.477 to 5.125 / 3.474 to 4.369; all.continuous_engaged 1.000 / 0.000 |
| S12 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 2.841 / 0.000 to 2.841; nothing passes into the wall: 0.000 / 0.000; no rebound past an elastic one: 0.427 to 0.451 / 0.427 to 0.451; the wall stops the car: 39.3 to 39.7 / 39.3 to 39.7 | all.energy_gain_j 0.000 to 2.841 / 0.000 to 2.841; all.plastic_work_j 38,181 to 43,074 / 38,181 to 43,074; car.crush_peak 0.285 to 0.302 / 0.285 to 0.302; car.rebound_ratio 0.427 to 0.451 / 0.427 to 0.451; car.speed_end 4.791 to 4.932 / 4.791 to 4.932; all.continuous_engaged 0.000 / 0.000 |
| S12 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 2.841 / 0.000 to 2.841; nothing passes into the wall: 0.000 / 0.000; no rebound past an elastic one: 0.427 to 0.451 / 0.427 to 0.451; the wall stops the car: 39.3 to 39.7 / 39.3 to 39.7 | all.energy_gain_j 0.000 to 2.841 / 0.000 to 2.841; all.plastic_work_j 38,181 to 43,074 / 38,181 to 43,074; car.crush_peak 0.285 to 0.302 / 0.285 to 0.302; car.rebound_ratio 0.427 to 0.451 / 0.427 to 0.451; car.speed_end 4.791 to 4.932 / 4.791 to 4.932; all.continuous_engaged 0.000 / 0.000 |
| S12 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 0.000 to 2.841 / 0.000; nothing passes into the wall: 0.000 / 0.000; no rebound past an elastic one: 0.434 to 0.461 / 0.392 to 0.455; the wall stops the car: 38.7 to 38.8 / 40.0 to 40.3 | all.energy_gain_j 0.000 to 2.841 / 0.000; all.plastic_work_j 37,730 to 42,846 / 39,834 to 45,435; car.crush_peak 0.282 to 0.300 / 0.298 to 0.312; car.rebound_ratio 0.434 to 0.461 / 0.392 to 0.455; car.speed_end 4.875 to 5.034 / 4.405 to 4.981; all.continuous_engaged 1.000 / 0.000 |
| S13 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 9.184 to 9.664 / 9.184 to 9.664; nothing passes into the pole: 0.000 / 0.000; the pole throws it back no faster than it came: -1.013 to -0.458 / -1.013 to -0.458; the pole and the carrier only slow it: -1.013 to -0.458 / -1.013 to -0.458; the struck side absorbs more than the far side: 10.2 to 12.2 / 10.2 to 12.2 | all.energy_gain_j 9.184 to 9.664 / 9.184 to 9.664; all.plastic_work_j 20,397 to 21,929 / 20,397 to 21,929; car.crush_peak 0.020 to 0.023 / 0.020 to 0.023; car.rebound_ratio -0.303 to -0.281 / -0.303 to -0.281; car.speed_end 2.893 to 3.128 / 2.893 to 3.128; all.continuous_engaged 0.000 / 0.000 |
| S13 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 9.184 to 9.664 / 9.184 to 9.664; nothing passes into the pole: 0.000 / 0.000; the pole throws it back no faster than it came: -1.013 to -0.458 / -1.013 to -0.458; the pole and the carrier only slow it: -1.013 to -0.458 / -1.013 to -0.458; the struck side absorbs more than the far side: 10.2 to 12.2 / 10.2 to 12.2 | all.energy_gain_j 9.184 to 9.664 / 9.184 to 9.664; all.plastic_work_j 20,397 to 21,929 / 20,397 to 21,929; car.crush_peak 0.020 to 0.023 / 0.020 to 0.023; car.rebound_ratio -0.303 to -0.281 / -0.303 to -0.281; car.speed_end 2.893 to 3.128 / 2.893 to 3.128; all.continuous_engaged 0.000 / 0.000 |
| S13 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 9.184 to 15.2 / 0.000; nothing passes into the pole: 0.000 / 0.000; the pole throws it back no faster than it came: -0.977 to -0.766 / -0.845 to -0.476; the pole and the carrier only slow it: -0.977 to -0.766 / -0.845 to -0.476; the struck side absorbs more than the far side: 12.5 to 18.0 / 8.833 to 10.6 | all.energy_gain_j 9.184 to 15.2 / 0.000; all.plastic_work_j 23,221 to 24,575 / 22,618 to 24,607; car.crush_peak 0.017 to 0.023 / 0.021 to 0.022; car.rebound_ratio -0.277 to -0.246 / -0.292 to -0.273; car.speed_end 2.708 to 2.916 / 2.826 to 3.014; all.continuous_engaged 1.000 / 0.000 |
| S14 | discrete | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 3.654 to 5.031 / 0.000; momentum along the barrier's travel: 17,387 to 18,688 / 17,633 to 18,701; the car leaves no faster than an elastic hit: 9.327 to 9.909 / 9.219 to 9.734; the struck side absorbs more than the far side: 2.905 to 3.475 / 3.056 to 3.756 | all.energy_gain_j 3.654 to 5.031 / 0.000; all.plastic_work_j 46,473 to 51,378 / 47,088 to 51,299; car.crush_peak 0.013 to 0.018 / 0.012 to 0.018; car.rebound_ratio 0.000 / 0.000; car.speed_end 5.806 to 6.988 / 5.876 to 7.278; pair.car_barrier.crossings_max 46.0 to 76.0 / 48.0 to 75.0; all.continuous_engaged 0.000 / 0.000 |
| S14 | edge | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 4,703 to 1,006,835 / 6,668 to 20,389; momentum along the barrier's travel: 16,750 to 17,954 / 15,731 to 17,874; the car leaves no faster than an elastic hit: 8.273 to 9.156 / 7.817 to 8.735; the struck side absorbs more than the far side: 23.5 to 60.1 / 36.2 to 77.6 | all.energy_gain_j 4,703 to 1,006,835 / 6,668 to 20,389; all.plastic_work_j 211,372 to 5,981,264 / 231,302 to 475,575; car.crush_peak 0.006 to 0.034 / 0.010 to 0.034; car.rebound_ratio 0.000 / 0.000; car.speed_end 5.354 to 5.921 / 4.771 to 6.084; pair.car_barrier.crossings_max 322.0 to 690.0 / 305.0 to 394.0; all.continuous_engaged 0.000 / 0.000 |
| S14 | continuous | agree; 8 / 8 runs | state stays finite: 1.000 / 1.000; no energy created: 6,271 to 13,083 / 7,491 to 13,325; momentum along the barrier's travel: 17,600 to 18,687 / 17,674 to 18,947; the car leaves no faster than an elastic hit: 8.351 to 9.401 / 8.942 to 9.450; the struck side absorbs more than the far side: 21.7 to 28.6 / 15.0 to 28.4 | all.energy_gain_j 6,271 to 13,083 / 7,491 to 13,325; all.plastic_work_j 111,302 to 152,084 / 109,548 to 146,049; car.crush_peak 0.013 to 0.019 / 0.014 to 0.018; car.rebound_ratio 0.000 / 0.000; car.speed_end 5.674 to 6.492 / 6.235 to 6.810; pair.car_barrier.crossings_max 0.000 / 0.000; all.continuous_engaged 1.000 / 1.000 |

Verdict disagreements (the criterion fails the setting on one executor only):

- S06 continuous, momentum along the road: CPU pass, CUDA FAIL (1 of 8)
- S08 continuous, momentum across the road: CPU pass, CUDA FAIL (3 of 8)
- S08 continuous, turns the same sense as the other: CPU pass, CUDA FAIL (1 of 8)

Known failures that occurred on one executor only (neither fails the setting):

- S07 continuous, nothing passes into the block: CPU known failure (1 of 8), CUDA pass (known intermittent, did not occur)
- S08 discrete, the common centre of mass stays still: CPU known failure (2 of 8), CUDA pass (known intermittent, did not occur)
- S08 edge, momentum across the road: CPU known failure (1 of 8), CUDA pass (known intermittent, did not occur)
- S09 discrete, no energy created: CPU known failure (1 of 8), CUDA pass (known intermittent, did not occur)
- S10 continuous, nothing passes into the block: CPU known failure (4 of 8), CUDA pass (known intermittent, did not occur)

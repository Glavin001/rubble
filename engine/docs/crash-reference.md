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
| CPU reference | `cargo run --release -p engine-softbody-lab --example crash_ref -- S02 [--runs 8] [--contact discrete\|edge\|continuous] [--json]` |
| CUDA on Modal | `tools/modal/gpu_run.sh --build modal [--fanout-tests] -- -p engine-cuda --test crash_reference` (`CRASH_CONTACT`, `CRASH_RUNS`, `CRASH_REPORTS=DIR`) |
| Browser (WebGPU crash lab) | `softbody.html?solver=explicit&crashRef=S02[&run=0][&contact=edge][&cam=side]` (scene picker: "Reference crashes") |
| Browser, stills and report | from `crates/engine-webgpu`: `python3 ../../scripts/with-gpu-lock.py --limit 300 -- node crash-ref-record.mjs --crashRef S02 [--cam side] [--contact edge] [--video] [--all-cams]` (to `out/crash-reference/<id>-<contact>-run<k>/`; `--all-cams`: a still from every camera at every moment) |
| Page against CPU | `python3 tools/crash-reference/compare.py CPU.json report.json [--all]` (the asserted criteria side by side) |
| Curves against NHTSA's | `crash_ref -- S11 --json > s11.jsonl`, then `python3 -I tools/crash-reference/overlay.py docs/crash-reference/nhtsa out/crash-reference/overlays s11.jsonl` (section 4) |

`--json` reports carry the run's time histories (`series`, section 4.1) beside the metrics.

In the browser the page sends the id; `engine-world-wasm` (`src/crash_ref.rs`) builds the
fleet with `CrashScenario::build`, reads the GPU state at every sample step and judges the
`CrashReport` with the same recorder. The generated designs' templates come from their mod
folders (`__mods/<design>_new`, which the site build generates with the same rigger:
`crash_templates::template_from_files`, the constructor `template` uses on the rigger's
output; a native test holds the two equal), so the body drawn is the physics' own: each
generated design its own skinned body (S09's wagon and 4x4 both). Bodies without a skin of
their own (the FMVSS 214 barrier) are drawn as their beams and their collision face.

Each run prints every criterion with its band and measured value; tier 0 and tier 1
failures fail the run. The executor provides only `Runner` (step, read state); the metrics
are host code (`crash_reference/metrics.rs`), so CPU, CUDA and WebGPU are measured by the
same code.

## Reference frames

Third-party footage is not committed. `docs/crash-reference/index.json` lists each frame
(scenario, video time, camera, moment, what to look at); the frames themselves are in
`~/Development/rubble-artifacts/crash-reference/`. `tools/crash-reference/sheet.py [ID ...]`
lays each frame beside Rubble's still from the matching camera at the nearest moment (the
recorder's `--all-cams` stills), one PNG per scenario and contact setting in
`out/crash-reference/sheets/` (`--copy-to DIR`); it says "no reference frames" where the
frames are absent (S10), and nothing else reads that folder.

The cameras are framed for a 60° vertical field of view at 16:9 (the page's): side and top
views put the car at impact across about two thirds of the width, wide enough for where the
wreck ends (S07 tumbles back off the block, S10 slides on past it).

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
3. **Tracings.** `docs/crash-reference/tracings/*.json`: per reference frame, the real and
   BeamNG halves' crop and polylines in normalised image coordinates (hood, fender top,
   wheel arch, bumper, windscreen, roofline, front door's front edge, sill, wheels; for the
   top view the visible outline, hood, windscreen base, side line). The committed ones are
   a first pass placed by eye: approximate. Correct them with `python3
   tools/crash-reference/trace_server.py` and `http://127.0.0.1:8770/` (drag, shift-click to
   add, alt-click to delete, save writes the JSON back).
4. **One frame, shape numbers.** S01's frames show the whole car: origin the rear wheel,
   unit the wheelbase in each source's own before frame. S02's show only the front half:
   origin the front door's front edge at the sill (Rubble: that corner carried with the
   cabin), unit the front wheel's radius. Hood fold height, chord shortening, creases (turns
   over 20°), front wheel setback, windscreen angle and ride height are computed by the same
   code for the three sources; they are reported comparisons, not criteria.

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

CPU reference, nominal run, measured 2026-11-27. Each is a finding, not a band to widen.

| Scenario | Criterion (basis) | Band | Discrete | Edge | Edge + continuous |
|---|---|---|---|---|---|
| S03 barrier into the sedan's side, 50 km/h | no energy created (law): the largest rise of kinetic + gravitational + elastic energy | at most 132 J (0.1 % of the 131.9 kJ at the start) | 9 J, holds | **38,186 J** | **19,679 J** |

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

S11–S14 (added 2026-11-27; CPU reference, 8 runs each in each contact setting):

| Scenario | Criterion (basis) | Band | Discrete | Edge | Edge + continuous |
|---|---|---|---|---|---|
| S11 sedan into the rigid wall, 56.3 km/h | restitution at the rear-seat crossmember (cited: nine sedans at 56 km/h, 0.057–0.194, + 0.05) | at most 0.25 | **0.30–0.35**, 8 of 8 fail | **0.30–0.35**, 8 of 8 | **0.29–0.36**, 8 of 8 |
| S14 MDB crabbed 27° into the sedan, 62 km/h | no energy created (law) | at most 203 J | 4–5 J, holds | **4.7 kJ–1.0 MJ**, 8 of 8 fail | **6.3–13.1 kJ**, 8 of 8 fail |

S14's created energy is S03's (below): edge-contact hits between the barrier and the door.

Also failing with continuous contact, CPU reference, nominal run: S08 creates 443 J (at most
340) and its common centre of mass ends at 0.454 m/s (at most 0.3); discrete: 48 J and 0.106.

Collision-mesh crossings at any sample (the repo's hard rule is 0), nominal run:

| Scenario | Discrete | Edge | Edge + continuous |
|---|---|---|---|
| S03 car and barrier | 94 | 315 | 0 |
| S08 two sedans | 154 | 31 | 0 |

On the GPU, continuous contact does not engage for a single vehicle against static boxes (it
runs only on the split group path): a "continuous" run of S01, S02, S07 or S10 there is an
edge run. With two vehicles (S08, in Chrome) it engages and diverges from the CPU reference
(momentum along the road -1,024 kg m/s against -13).

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

Rubble is the generated sedan as generated by default (every front rule off, as before). "Rule 5" and "all six rules" are the same sedan generated with `Physics::front_rules` (`hood_fold_row` alone; every rule), measured the same way (2026-11-27, `crash-ref/front-fixes`; `sedan-front-audit.md` §8, `generated-vehicles-lessons.md` §25). Neither is the default: each breaks a gate or a hard rule elsewhere (lessons §25.8). With rule 5 the S11 run 0 meets the cited restitution (0.247 against at most 0.25; 6 of 8 runs in each contact setting).

| quantity | Rubble | Rubble, rule 5 | Rubble, all six rules | test 8545 | test 7520 |
|---|---|---|---|---|---|
| mass (kg) | 1448 | 1448 | 1448 | 1664 | 1662 |
| speed (km/h) | 56.3 | 56.3 | 56.3 | 56.7 | 56.3 |
| peak decel, 4 ms estimator (g) | 50.0 | 48.5 | 46.7 | 36.8 | 40.7 |
| peak decel, CFC 60 (g) | 59.1 | 60.8 | 54.0 | 39.8 | 45.0 |
| time of peak, CFC 60 (ms) | 15.5 | 15.5 | 44.0 | 44.1 | 43.9 |
| t_m, v = 0 (ms) | 44.4 | 48.6 | 54.0 | 70.9 | 75.9 |
| dynamic crush (m) | 0.394 | 0.415 | 0.492 | 0.663 | 0.685 |
| mean decel to t_m (g) | 35.9 | 32.7 | 29.5 | 22.7 | 21.0 |
| rebound, t_m-200 ms (m/s) | 5.34 | 3.85 | 4.35 | 1.50 | 0.89 |
| restitution | 0.342 | 0.247 | 0.278 | 0.095 | 0.057 |
| residual crush (m) | 0.199 | 0.252 | 0.344 | 0.554 (length), 0.529 (max) | 0.579 (length), 0.579 (max) |
| peak wall force (kN) | 661 | 620 | 560 | 921 | 829 |
| time of peak force (ms) | 20.0 | 16.0 | 44.0 | 29.5 | 30.1 |


#### S12, edge contact, run 0

| quantity | Rubble | Rubble, rule 5 | Rubble, all six rules | test 9692 | test 7793 |
|---|---|---|---|---|---|
| mass (kg) | 1448 | 1448 | 1448 | 1690 | 1798 |
| speed (km/h) | 40.0 | 40.0 | 40.0 | 39.6 | 39.6 |
| peak decel, 4 ms estimator (g) | 46.2 | 41.9 | 31.1 | 27.1 | 26.1 |
| peak decel, CFC 60 (g) | 52.7 | 44.5 | 33.8 | 34.1 | 31.5 |
| time of peak, CFC 60 (ms) | 31.0 | 30.5 | 20.5 | 55.1 | 43.2 |
| t_m, v = 0 (ms) | 39.5 | 40.7 | 68.5 | 69.3 | 74.7 |
| dynamic crush (m) | 0.275 | 0.283 | 0.378 | 0.475 | 0.489 |
| mean decel to t_m (g) | 28.6 | 27.8 | 16.5 | 16.2 | 15.0 |
| rebound, t_m-200 ms (m/s) | 5.37 | 5.01 | 2.95 | 1.64 | 1.17 |
| restitution | 0.484 | 0.452 | 0.265 | 0.149 | 0.107 |
| residual crush (m) | 0.099 | 0.111 | 0.232 | 0.304 (length), 0.326 (max) | 0.279 (length), 0.279 (max) |
| peak wall force (kN) | 576 | 538 | 431 | - | - |
| time of peak force (ms) | 32.0 | 32.0 | 24.0 | - | - |


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



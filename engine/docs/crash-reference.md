# Crash reference suite

Ten real crash tests, each shown beside BeamNG in one video
(<https://www.youtube.com/watch?v=Lqx2KKWI8aM>), staged in Rubble on the explicit 2 kHz
node-beam solver with generated vehicles, and judged by acceptance criteria derived from
first principles. This page holds the scenarios, how a crash is quantified, each scenario's
criteria and what currently fails.

## How to run one

A scenario is defined once, in `crates/engine-softbody/src/crash_reference/` (one file a
scenario, one line in `all()`), and builds a fleet of `ExplicitBeams`, which every executor
takes. The same definition, sampling, metrics and criteria run everywhere:

| Where | Command |
|---|---|
| CPU reference | `cargo run --release -p engine-softbody-lab --example crash_ref -- S02 [--runs 8] [--contact discrete\|edge\|continuous] [--json]` |
| CUDA on Modal | `tools/modal/gpu_run.sh --build modal [--fanout-tests] -- -p engine-cuda --test crash_reference` (`CRASH_CONTACT`, `CRASH_RUNS`, `CRASH_REPORTS=DIR`) |

Each run prints every criterion with its band and measured value; tier 0 and tier 1
failures fail the run. The executor provides only `Runner` (step, read state); the metrics
are host code (`crash_reference/metrics.rs`), so CPU, CUDA and WebGPU are measured by the
same code.

## Reference frames

Third-party footage is not committed. `docs/crash-reference/index.json` lists each frame
(scenario, video time, camera, moment, what to look at); the frames themselves are in
`~/Development/rubble-artifacts/crash-reference/`.

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
arrives at 13.9 m/s, and, with continuous contact, 1,195 kJ counted as removed by the contact
response. Discrete contact has none of this (38 kJ of plastic work, fastest node 23 m/s). Not
yet diagnosed. Two things in the scene to look at first: the barrier's bumper element is held
to its main block by push-only beams and folds under the car's sill in every run
(`crash_rigs.rs`), and the barrier's honeycomb is made of push-only support beams.

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


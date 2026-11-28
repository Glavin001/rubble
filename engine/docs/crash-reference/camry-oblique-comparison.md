# Camry oblique tests (NHTSA 9124, 8790, 9121) against Rubble

**Which car this was measured on (added 2026-11-27).** Every Rubble number in this document is
the generated sedan in its default crash character, `beamng_feel` (`Physics.crash`,
`engine/docs/generated-vehicles.md` "Crash character"): front members at 1.8 times their defined
crush force, towers and cabin at 1.5, panels at twice their yield strain, set to match the ETK,
not a real car. The car as defined (`real_crash`) is the one to hold against a Camry, and
since later on 2026-11-27 it is the default (with the front seats moved beside the front door:
`generated-vehicles-lessons.md` §29); its 8-run ranges are in the paragraph below the table. CPU
reference, run 0 (`out/crash-character/`; S11 and S13 edge contact, S14 discrete; S13 at its
moved pole):

| | `beamng_feel` (default) | `real_crash` | Camry |
|---|---|---|---|
| S11 dynamic crush (m) | 0.39 | 0.48 | 0.66-0.69 |
| S11 restitution | 0.34 | 0.24 | 0.06-0.10 |
| S11 stop (ms) | 44 | 57 | 71-76 |
| S11 wall force peak (kN) | 661 at 20 ms | 508 at 32 ms | 829-921 |
| S13 pole force (kN) | 275 | 252 | 191-197 |
| S13 crush, peak / residual (mm) | 341 / 110 | 364 / 164 | 379-394 residual |
| S13 thrown back (m/s) | 4.0 | 3.3 | 1.7-2.2 |
| S14 crush, peak / residual (mm) | 156 / 51 | 276 / 165 | 258-275 residual |
| S14 struck car, lateral speed change (m/s) | 9.27 | 8.98 | 6.6-6.8 (7.0-7.2 mass-scaled) |

The car as it is now (as defined, seats moved, S13 re-aimed on its centre of mass), CPU
reference, 8 jittered runs (`out/crash-reference/sweep/cpu/`): S11 dynamic crush 0.46-0.50 m,
restitution 0.23-0.26, maximum crush at 57-58 ms, wall force 505-530 kN at 32-36 ms; S13 pole
force 258-275 kN, crush 348-397 mm peak and 165-223 mm residual; S14 crush 171-293 mm peak and
79-181 mm residual, struck car's lateral speed change 8.83-9.26 m/s. The timelines, sheets and
detailed tables below are still the `beamng_feel` car's.

The three NHTSA research oblique tests of the Toyota Camry: a heavy moving deformable barrier
strikes the standing car's front corner at 15 degrees. What happened (films, photos, sensors,
reports), what a Rubble scenario needs to reproduce it, the closest thing Rubble runs today, and
where and why it differs. Measured 2026-11-27 on the CPU reference, generated sedan.

Data: NHTSA Vehicle Crash Test Database (public domain, US government), downloaded locally to
`inspiration/ccsa-camry-2012/nhtsa-<test>/` (gitignored; nothing here depends on it). Committed:

| What | Where |
|---|---|
| Numbers and curves from the sensors and the report's tables | `nhtsa/9124.json`, `nhtsa/8790.json`, `nhtsa/9121.json` (`tools/crash-reference/nhtsa_reference.py oblique DATA_ROOT docs/crash-reference/nhtsa`) |
| Annotated media index: every film's view, film rate, first-contact frame, moments; photo groups, final-shape and measurement photos | `nhtsa/media/9124.json`, `8790.json`, `9121.json` (`tools/crash-reference/nhtsa_footage.py`) |
| The stand-in scenarios (data) | `scenarios/camry-oblique-left-mdb214.json`, `scenarios/camry-oblique-right-mdb214.json` |
| Key stills | Reference Studio shots `NHTSA9124`, `NHTSA8790`, `NHTSA9121` (27 frames; images in the gitignored library) |

Contact sheets, frame grids and the pulse plots (third-party stills: never committed) are in
`~/Development/rubble-artifacts/crash-reference/camry-oblique/` (`sheets/`, `frames/`,
`pulse-left.png`, `pulse-right.png`).

Confidence marks used below: **measured** (a sensor, a report table, a Rubble metric),
**film** (seen on a film or photo, timed by its burned-in caption), **by eye** (an angle or
size estimated from a film still), **hypothesis** (an explanation not yet tested).

## 1. The tests

All three: Calspan, contract DTNH22-10-D-00155, "RMDB into front 15 degree stationary vehicle,
overlap = 35 percent"; the RMDB (NHTSA research MDB 2010B) towed "in a full forward direction,
without any crabbing" into the standing car, whose axis is turned 15 degrees to the RMDB's path
(report section 2). Track: concrete, recorded as wet in all three (API test record).

| | 9124 | 8790 | 9121 |
|---|---|---|---|
| Car | 2012 Camry, 1,758.5 kg | 2014 Camry, 1,734 kg | 2012 Camry, 1,751.5 kg |
| Struck side | left (driver) | left | right |
| RMDB | 2,490.7 kg at 89.68 km/h | 2,490.2 kg at 89.69 km/h | 2,490.7 kg at 90.46 km/h |
| Offset from target point (report) | +12 mm down, +10 mm left | -13 mm, 0 | -2 mm, -25 mm |
| Report pages | p.6, p.8 | R002 p.7, p.9 | p.6, p.8 |

**The barrier** (8790 report p.37 drawing and table; 9124 p.36; API vehicle 1): 3,995 mm long,
track 1,930 mm, wheelbase 2,449-2,451 mm, CG 904-907 mm behind the front axle, axle loads
1,605.7-1,609.4 kg front and 881.3-884.5 kg rear; a honeycomb face **2,205 mm wide and 605 mm
deep in two blocks** (the drawing's G and D), about 1,024 mm high (the drawing's M: inference
from the drawing; the RMDB crush grids cover heights from 25 or 50 mm to 910-925 mm). The
honeycomb's crush strengths, the face's ground clearance and the barrier's yaw inertia are **not
in the reports**. API "RESEARCH MDB 2010 LOWERED", width field 1,250 mm (meaning not stated).

**Film timing.** Every high-speed film carries a burned-in caption ("1,000 fps / T0: 52 / Frame:
51 / -0.08 ms"); the times were read off each film at frame 50 (0-based) and the frame whose
time is within 0.5 ms of 0 is recorded as `t0_frame` in the media index:

- 9124: off-board 1000 fps, first contact at frame 50; onboard (cameras 2-5) 500 fps, frame 25;
  cameras 10 and 11 carry no time caption (rate from the report, t0 not pinned); camera 1 real
  time 30 fps.
- 8790: every camera 1000 fps (the report lists onboard at 1000 fps too); first contact at frame
  48 (cameras 10, 11, 14, 15: "T0: 50", 2.32 or 2.00 ms at frame 50), 49 (camera 7), 50
  (cameras 2-4, 8, 12, 13; a plain "0.000 s 1000 fps" caption) or 51 (cameras 5, 6, 9).
- 9121: off-board 1000 fps, frame 50; onboard 500 fps, frame 25; camera 9's caption says
  **100 fps** (T0: 49, 26.86 ms at frame 50) against 1000 in the report's table: the caption is
  used, and that film has only 101 frames. No real-time film was published.

So frame 50 is first contact, as on frontal test 8545, except 8790's cameras listed above and the
500 fps onboard films (frame 25).

## 2. What happens, over time

Times are ms after first contact. Each event names its film and frame (0-based); sensor
numbers are CFC 60 (processing in the JSONs' `processing` and `frame` fields).

### 9124 (2012 Camry, left)

| ms | What | Evidence |
|---|---|---|
| 0 | the face's edge meets the left front corner | film: cam 15 (C007) frame 50 |
| 10 | bumper cover and left fender start to crush | film: cam 12 (C004) frame 60 |
| 14-22 | hood's left corner buckles, then folds in one large crease | film: cam 14 (C006) frames 64, 72 |
| 21 | airbags and pretensioners fire | measured: report p.30 |
| 25 | fender and hood crushed, hood tenting up; front wheel still in place | film: C004 frame 75 |
| 40 | face at the front wheel; wheel being pushed rearward; airbags out | film: C004 frame 90 |
| 45 | peak resultant acceleration at the CG accelerometer, 61.5 g | measured |
| 55-60 | the car's lateral velocity peaks (5.6 m/s, right) then settles to 4.4 | measured (`car_dvy_mps`) |
| 60 | face at the A-pillar base; wheel driven rearward under the RMDB; hood folded against the windscreen | film: C004 frame 110 |
| 75 | common velocity along the RMDB's travel (14.85 m/s); approach 1.02 m | measured |
| 80 | RMDB at the front door's leading edge; corner wrapped around the face; the car's heading changed little | film: C004 frame 130, C007 frame 130 (by eye) |
| 110 | car moving rearward and to its right, away from the RMDB | film: C004 frame 160 |
| 150 | RMDB pitched nose-down, sliding past the car's front, turned about 10 deg | film: C004 frame 200, C007 frame 200 (by eye) |
| 150 | car at 15.4 m/s along the RMDB's travel, RMDB at 14.1: separating at 1.4 m/s | measured |
| 300 | RMDB turned about 30 deg, past the car's front | film: C007 frame 350 (by eye) |

Final shape (report p.7, p.39, p.42-43; photos No.006/008/010/027/156/159): static crush 551 mm
at C2 (C1 unrecoverable), 405 / 250 / 90 / -67 mm at C3-C6; bumper beam crush 613 mm at B2 down to
3 mm at B9; left wheelbase -150 mm, right -39 mm; left door opening -26 mm (upper), driver door
jammed; A- and B-pillar slightly buckled (separation at the door top, photo No.061); windscreen
79.9 % retained on the left; RMDB face crush up to 308.7 mm at its outboard lower corner.

### 8790 (2014 Camry, left)

| ms | What | Evidence |
|---|---|---|
| 0 | first contact at the left front corner | film: cam 14 (C006) frame 48 |
| 7 | airbags, curtain and pretensioners fire (second driver stage 37 ms, seat bags 48 ms) | measured: channels' "time to fire" |
| 10-25 | face at the bumper corner; fender crushed, hood folding up | film: cam 8 (C014) frames 60, 75 |
| 40 | face at the front wheel; debris thrown up | film: C014 frame 90 |
| 42 | yaw rate peaks at +224 deg/s (nose right) | measured (angular rate sensor) |
| 45 | peak resultant acceleration 55.3 g | measured |
| 50 | yaw +2.9 deg (nose right), then the car turns back | measured |
| 60 | front hidden behind the RMDB | film: C014 frame 110 |
| 90-93 | closing speed along the RMDB's travel at its least, 0.4 m/s (does not cross zero in the integrated record) | measured |
| 100 | hood and corner crushed, windscreen cracked; yaw -1.0 deg | film: C006 frame 148; measured |
| 150 | yaw -4.9 deg (nose left) | measured |
| 200 | RMDB turned about 25 deg; pieces of the bumper cover on the floor | film: C006 frame 248 (by eye) |
| 290 | yaw -14.8 deg (nose left); car leaving the overhead frame | measured; film: C006 frame 348 |

Final shape (R002 p.8, p.41, p.44-45; photos P075-P089, P148-P158, P202-P215): crush 561 mm at C1,
560 at C2; bumper beam 538 mm at B2; left wheelbase -170 mm; A- and B-pillar buckled; the lower
bumper cover lies detached on the floor (photo P076, post-test); RMDB 277 mm.

### 9121 (2012 Camry, right)

| ms | What | Evidence |
|---|---|---|
| 0 | first contact at the right front corner | film: cam 15 (C006) frame 50 |
| 10-25 | face at the corner; hood and fender crushing, hood rising | film: cam 11 (C002) frames 60, 75 |
| 22 | driver airbag stage 1 fires (stage 2 32 ms) | measured: report p.30 |
| 40 | hood tented, bumper cover torn and pushed | film: C002 frame 90 |
| 43 | peak resultant acceleration 74.6 g | measured |
| 60 | right front heavily crushed, hood folded up | film: C002 frame 110 |
| 80 | bumper beam and cover swing out ahead of the car | film: C002 frame 130 |
| 87 | common velocity along the RMDB's travel (14.74 m/s); approach 1.09 m | measured |
| 100 | RMDB turned about 10 deg | film: C006 frame 150 (by eye) |
| 110-150 | the front end (bumper beam, radiator support) bent forward and toward the car's left, protruding far ahead | film: C002 frames 160, 200 |
| 150 | car's velocity change 15.9 m/s at 28 deg off its -X (body axes) | measured |
| 200-290 | RMDB turned about 35-40 deg; a large front-end assembly swings ahead of the car (still attached: post-test photo No.008 shows the beam protruding forward-left) | film: C006 frames 250, 340 (by eye); C002 frame 340 |

Final shape (report p.7, p.42-43; photos No.006/008/020/027/143-160): crush 579 mm at C5, 524 at
C6, -110 mm at C1 (the far corner pushed out); bumper beam 635 mm at B9, and B1 **+294 mm**
(forward); right wheelbase -80 mm, left +34 mm; right front door jammed; right A-pillar buckled
rearward (photo No.105); RMDB 289 mm at its outboard lower corner.

### The sensor numbers (all CFC 60, body axes unless stated)

| | 9124 | 8790 | 9121 |
|---|---|---|---|
| car dv at 150 ms (x, y), m/s | -14.80, +4.42 | -13.30, +3.18 | -14.04, -7.52 |
| resultant, direction off -X | 15.44, 16.6 deg | 13.67, 13.4 deg | 15.92, 28.2 deg |
| car along the RMDB's travel at 150 ms | 15.44 | 13.68 (yaw-corrected) | 15.50 |
| RMDB vx at 150 ms | 14.09 | 15.25 | 13.64 |
| common velocity: time, speed | 74.6 ms, 14.85 | none (closing least 0.38 at 93 ms) | 86.9 ms, 14.74 |
| perfectly plastic common velocity m_b V / (m_b + m_c) | 14.60 | 14.69 | 14.75 |
| momentum balance at 150 ms (car gained / RMDB lost) | 1.007 | 0.986 | 0.949 |
| peak resultant acceleration (CG accelerometer) | 61.5 g at 45 ms | 55.3 g at 45 ms | 74.6 g at 43 ms |
| yaw | no sensor; small to 150 ms (film, by eye) | +2.9 deg at 50 ms, -4.9 at 150, -14.8 at 290 | no sensor |

The momentum balance (0.95-1.01) is the check on the integration: two independent
accelerometers on two bodies agree on the impulse. The RMDB's velocities past about 100 ms are
in its own turning axes (the films show it turning 10-40 deg), so 8790's rising RMDB vx after
100 ms is that frame, not a speed-up. Yaw for 9124 and 9121 could not be derived from the
lateral accelerometers' difference (CG against rear sills): on 8790, where the angular rate is
measured, that estimate gave -57 deg at 290 ms against the measured -14.8 deg (0.2 m between the
sensors), so it is not published.

The NHTSA API's maximum crush for 9124 (3,326 mm) is C1's pre-test distance; the report's is
551 mm (recorded in `9124.json` as `api_errata`).

## 3. What reproducing it needs

The exact setup: a moving barrier of 2,490.5 kg (axle loads and CG above), heading straight at
89.7-90.5 km/h, free to roll and turn after release (towed to impact, "without any crabbing");
its face 2,205 mm wide, 605 mm deep, about 1,024 mm high, flat across; the standing car turned
15 deg to the barrier's path so that the barrier strikes its front corner, 35 % of the car's
width overlapped (left for 9124 and 8790, right for 9121; the report's offsets from the target
point are 0-25 mm); car at its test mass (1,734-1,758.5 kg, with two 50th male THOR dummies and
cargo ballast; 9124: 963 kg front axle, 795.5 kg rear, CG 1,252 mm behind the front axle), wheels
on a wet concrete floor; record to at least 300 ms.

**Closest runnable today, without new Rust**: the moving FMVSS 214 barrier (`mdb214`) placed by
hand as data, `docs/crash-reference/scenarios/camry-oblique-left-mdb214.json` (and `-right-`, its
mirror):

```
cd engine && cargo run --release -p engine-softbody-lab --example crash_ref -- \
  --scenario docs/crash-reference/scenarios/camry-oblique-left-mdb214.json --contact discrete --run 0 --json
```

The barrier heads -165 deg (15 deg off straight on) at 89.68 km/h; the inboard edge of its 1.676 m
face, carried along its travel, passes 35 % of the sedan's width in from the sedan's left side at
the sedan's front; first contact is computed. (`--family head_on --set car_a=sedan --set
car_b=mdb214 --set speed=0 --set speed_b=89.68 --set overlap=0.35 --set angle=15` gives nearly the
same crash, but turns the barrier about its own origin after placing the overlap, so the overlap
at contact is not 35 %.) The barrier's extents were read off `crash_ref --export S08` and `S03`.

How the stand-in differs from the real setup (masses computed):

| | Real | Stand-in | Difference |
|---|---|---|---|
| Barrier mass | 2,490.2-2,490.7 kg | 1,367.6 kg (49 CFR 587) | -1,123 kg, -45.1 % |
| Barrier face | 2,205 x ~1,024 mm, 605 mm deep | 1,676 x 559 mm at 279 mm, 381 mm deep + 102 mm bumper | 529 mm narrower, ~465 mm lower, 122-224 mm shallower |
| Face stiffness | not in the reports | FMVSS 214 honeycomb (`MDB_CRUSH`: 310 / 1,689 kPa) | unknown |
| Car | Camry, 1,734-1,758.5 kg, 4,804-4,808 mm, wheelbase 2,767-2,768 mm | generated sedan, 1,448.4 kg, 4.886 m, wheelbase 2.83 m | -285.6 to -310.1 kg (-16.5 to -17.6 %) |
| Mass ratio barrier / car | 1.416 (9124), 1.436 (8790), 1.422 (9121) | 0.944 | |
| Plastic common velocity | 14.60-14.75 m/s | 12.10 m/s | -17 % |
| Kinetic energy at contact | 772.8 kJ (9124) | 424.3 kJ | -45 % |

A second variant puts the `suv_4x4` (1,974.7 kg; ratio 1.363, plastic common velocity 14.37 m/s)
in the barrier's place, to separate the mass from the rest (files in `~/Development/rubble-artifacts/crash-reference/camry-oblique/desc/`,
written by `scripts/make_desc.py` there; working files, not committed). Its front is a
vehicle's, not a flat honeycomb, and its origin is not at its face, so its overlap is only
approximate.

**Yaw over time.** The run reports only the end yaw, so each setup was also run cut at 50, 100,
150, 200, 250, 300 ms and 1 s (the same run: the 150 ms dv is identical, -13.250 m/s, in every
cut).

## 4. Results (CPU reference, run 0)

Left oblique, discrete contact unless stated:

| | NHTSA 9124 | NHTSA 8790 | mdb214 stand-in | suv_4x4 variant |
|---|---|---|---|---|
| car dv at 150 ms (x, y), m/s | -14.80, +4.42 | -13.30, +3.18 | -13.25, +4.28 | -15.67, +4.17 |
| car along the travel at 150 ms | 15.44 | 13.68 | 13.91 | 16.22 |
| car along / plastic common velocity | 1.06 | 0.93 | 1.15 | 1.13 |
| barrier vx at 150 ms | 14.09 | 15.25 | 9.68 | 12.67 |
| separation speed at 150 ms (car along - barrier) | 1.35 | -1.57 | 4.23 | 3.55 |
| time to common velocity | 75 ms | ~90 ms (least closing) | 115 ms | 119 ms |
| approach along the travel until then | 1.02 m | | 1.83 m | 1.85 m |
| half the car's peak speed along the travel reached at | ~44 ms | ~45 ms | 83 ms | 79 ms |
| car yaw at 150 / 250 / 300 ms | small (film) | -4.9 / -12.2 / -14.8 deg nose LEFT (290 ms) | 7.6 / 18.4 / 21.5 deg nose RIGHT | 7.3 / 17.4 deg nose right (250 ms) |
| barrier turn at 150 / 300 ms | ~10 / ~30 deg (by eye) | 25 deg at 200 ms (by eye) | 13 / 35 deg | 8.6 deg at 150 ms |
| struck-side crush | 551 mm static (C2) | 561 mm (C1) | left length 0.116 m peak, 0.080 m at 300 ms; front overhang 0.235 m peak | left length 0.101 m peak |
| struck-side wheelbase | -150 mm | -170 mm | -47 mm | -49 mm |
| plastic work at 300 ms, car / barrier | | | 51.0 / 47.1 kJ | 36.6 / 41.3 kJ (250 ms) |
| car's plastic work in the left front zone | | | 42.2 kJ of 51.0 (front right 0.85 kJ) | 28.9 kJ |
| barrier's length growth (peak) | face crush 309 mm (static) | 277 mm | 0.81 m | 0.11 m |
| detached | none reported (9124); bumper cover pieces (8790, film and photo) | | 2 parts, 5.9 kg | none |

Right oblique (9121 against the mirrored stand-in): car dv at 150 ms -14.04, -7.52 m/s against
-13.56, -4.43; along the travel 15.50 against 14.25; barrier 13.64 against 9.49 m/s; common
velocity at 87 against 108 ms; the stand-in's car yaws 5.7 deg nose left at 150 ms, 18.9 at 300
ms (the left run's mirror); right wheelbase -80 mm against -69 mm at 300 ms; right length 0.099 m.

**Contact settings and energy** (left stand-in, 300 ms):

| Contact | Energy created | Collision-mesh crossings (max) | KE lost | Plastic work | KE lost not booked as plastic | car dv at 150 ms |
|---|---|---|---|---|---|---|
| discrete | 0 J | 62 | 227.8 kJ | 98.1 kJ | 129.7 kJ | -13.25, +4.28 |
| edge | 7,441 J (fails "no energy created") | 197 | 220.4 kJ | 166.8 kJ | | -12.28, +3.71 |
| continuous | 5,795 J (fails) | 0 | 221.3 kJ | 111.9 kJ | | -12.46, +3.83 |

At 150 ms (discrete) the run has lost 190.3 kJ of its 424.3 kJ, 91.9 kJ of it as plastic work;
the other 98.4 kJ is not attributed by the suite's metrics (elastic energy stored in beams, beam
damping, contact, the ground). What contact alone removes is therefore **not measured** here:
the solver's work ledger (`tests/crash_metrics_s03_energy.rs`, an S03 probe) is the tool to split
it, and was not run on this scenario. The edge and continuous runs create energy at
barrier-to-car contact, as S03 and S14 do (`crash-reference.md`, "Currently failing"); the
discrete run creates none but its collision meshes cross (62), against the repo's rule of none.

## 5. Differences, ranked

1. **The barrier is a different, lighter, smaller and softer one** (cause of most of the rest).
   Measured: 1,367.6 against 2,490.5 kg (-45 %), face 1.676 x 0.559 m against 2.205 x ~1.02 m,
   424 against 773 kJ at contact. The stand-in's own plastic work is as large as the car's (47
   against 51 kJ) and its length grows 0.81 m, where the real RMDB's face crushes 277-309 mm.
   Confidence: measured (masses, sizes); the stiffness comparison is a hypothesis until the
   OMDB's honeycomb strengths are known. With the 1,975 kg SUV in its place the car's dv
   reaches 16.2 m/s along the travel and the barrier stops stretching (0.11 m), so the mass
   explains the dv shortfall but not the shape (item 2).

2. **The car crushes far less: 0.08-0.12 m at its struck side against 0.55-0.56 m.** Measured
   (left length gauge against the report's C1/C2; wheelbase -47 against -150/-170 mm). In the
   SUV variant, with more mass and a stiffer striker, the crush is no larger (0.10 m), so a
   softer barrier is not the only cause. Hypothesis: the generated sedan's front corner (the
   rail's end, the bumper beam's end, the wheel and the A-pillar base) is too stiff in an
   offset, angled load, or the load path around the corner is missing (the real wheel is driven
   back into the sill and the A-pillar base, film C004 frame 110, wheelbase -150 mm). Evidence
   to collect next: the sedan's beams by zone through the crush (which break, which bend) and
   the rail's end-on and bending capacity against the Camry's (CCSA model).

3. **The pulse is slower and longer: common velocity at 108-119 ms against 75-87 ms; half the
   speed along the travel at 79-83 ms against 44-45 ms; 1.83 m of approach against 1.02-1.09 m.** Measured. Follows
   from 1 and 2: the engagement is carried by the barrier's soft face and by sliding, not by
   the car's structure. The real pulse's steep rise at 35-55 ms is where the face meets the
   wheel and the rail (film C004 frames 85-105); nothing in the stand-in's trace matches it.

4. **More rebound.** Measured: the car leaves 4.2 m/s faster than the barrier at 150 ms against
   1.4 (9124) and -1.6 (8790); the car's speed along the travel is 1.15 times the plastic common
   velocity against 1.06 and 0.93. Hypothesis: the same as S11's known restitution failure
   (0.30-0.35 against a fleet band of 0.25, `crash-reference.md`): elastic energy stored and
   returned by the front structure. The unbooked 98 kJ (section 4) is where to look.

5. **The car turns the other way.** Measured on 8790 only: nose right +2.9 deg at 50 ms, then
   nose left, -4.9 deg at 150 ms and -14.8 at 290 ms. The stand-in turns nose right throughout:
   7.6 deg at 150 ms, 21.5 at 300 ms. In this crash the two moments about the car's CG nearly
   cancel (rearward force at the corner, 0.6 m left of centre, turns the nose left; the
   rightward force 2.2 m ahead of the CG turns it right; dv ratio y/x 0.24-0.30), so the sign
   follows where along the front the force acts and how far rearward the corner collapses.
   Hypothesis: with the stiff corner (item 2) the force stays at the bumper's front, ahead of
   the CG, and the lateral moment wins; in the real car the corner collapses back to the wheel
   and A-pillar and the rearward force's moment wins. The barrier's turn agrees roughly (13 /
   35 deg at 150 / 300 ms against about 10 / 30, by eye).

6. **The right oblique's lateral dv is smaller: -4.4 against -7.5 m/s.** Measured, but the real
   number is in the car's turning body axes and 9121 has no yaw sensor; the overhead film shows
   the car pushed rearward and to its left. Partly frame, partly items 2 and 5 (hypothesis).
   The real right oblique also differs from the left ones (dv direction 28 against 13-17 deg;
   B1 +294 mm, the far end of the bumper beam pushed forward): the Camry is not symmetric
   (steering, engine and battery placement), the stand-in is a mirror.

7. **What breaks off.** Film and photo: 8790 loses pieces of the lower bumper cover (P076);
   9121's bumper beam assembly swings far forward but stays on (No.008). The stand-in detaches
   2 parts (5.9 kg) on the left, 1 (0.3 kg) on the right. Not compared further (part
   identities not matched).

8. **The car.** Measured: 285.6-310.1 kg lighter (-16.5 to -17.6 %), 78 mm longer, wheelbase
   62 mm longer; no dummies or ballast. The test mass's 45 % rear axle share (9124, with dummies
   and cargo) is not reproduced.

9. **Contact.** Measured: the only setting that creates no energy (discrete) lets the collision
   meshes cross (62); the settings without crossings create 5.8-7.4 kJ. Not a property of the
   real crash, but it bounds what the stand-in's numbers mean; the known S03/S14 finding.

## 6. Missing: an oblique family and the research barrier

To run these tests as reference scenarios, Rust needs (not added here):

- **A barrier design `omdb` (NHTSA research MDB 2010B)**: 2,490.5 kg; axle loads 1,605.7-1,609.4 /
  881.3-884.5 kg, CG 904-907 mm behind the front axle (reports p.36-37); length 3,995 mm, track
  1,930 mm, wheelbase 2,449-2,451 mm; face 2,205 mm wide, 605 mm deep in two honeycomb blocks,
  about 1,024 mm high; free-rolling wheels. Still to source: the two blocks' crush strengths and
  depths, the face's ground clearance and the barrier's yaw and pitch inertia (NHTSA's OMDB
  drawings and specification; not fetched, not in these reports). Its face model must be checked
  in an oblique load (the FMVSS 214 face's 0.81 m stretch here).
- **A family `oblique`** (or an `overlap_at` parameter for `head_on`): the barrier straight at a
  speed; the car standing, turned by an angle (15 deg); overlap as a share of the car's width at
  its front, measured from the struck side along the barrier's travel (35 %); the struck side
  (left/right); run-up. Presets: 9124 and 8790 (left, 89.68 and 89.69 km/h), 9121 (right, 90.46).
- **Metrics**: the car's yaw over time (the tests' only rotation sensor is 8790's), the car's
  velocity along the barrier's travel and the closing speed (the comparison above was done
  outside the suite by `scripts/analyze.py` in the artifacts folder from the run's series), and the
  report's crush stations C1-C6 and B1-B9 on the car's front (0.551-0.635 m maxima) instead of
  the length gauges.
- **Criteria with a basis**: the momentum balance is a law; the common-velocity speed, the dv
  direction and the crush profile are cited data from three tests of one configuration (two of
  one side), so any band would be their spread only.

## 7. Not verified

- Film angles and sizes marked "by eye"; the films' t0 for 9124 cameras 1, 10, 11 and 8790's
  camera 1 (no caption).
- The face height of the RMDB (read from the drawing) and the meaning of the API's 1,250 mm width.
- Which parts detach in the real tests beyond the bumper-cover pieces seen.
- Velocities of 9124 and 9121 after ~150 ms (body axes, no yaw sensor); the RMDB's after ~100 ms.
- What contact removes in the stand-in (section 4).
- The stand-in was run once (run 0) per setting; the eight jittered runs were not run.

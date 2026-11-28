# Camry side tests beside Rubble (S14 side MDB, S13 side pole)

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

The four real 2012–2014 Toyota Camry side tests from NHTSA's public database, read from their
films, photos, reports and sensor curves, set beside Rubble's S14 and S13 on the CPU
reference, with where Rubble differs over time and the most likely reason for each. Measured
2026-11-27 at `crash-ref/camry-side-footage` (cut from `claude/crash-reference-suite`
a81e0dfc7). Nothing was tuned: this is measurement only.

**Revised 2026-11-27 (S13's setup).** The first measurement found S13's pole 0.9 m behind where
the tests put it. S13 was moved (`s13_side_pole.rs`, "Where the pole strikes") and every S13
number, sheet and curve below is from the moved scenario; the earlier figures are kept in §2
for the record. The first version's second setup finding, that S14's barrier lands 0.6 m ahead
of the doors, was wrong (it read a door outline that starts at the window opening): S14 was
right and is unchanged.

- Real tests: 7517 (2012 Camry, side MDB), 8560 (2014.5 Camry, side MDB), 7523 (2012, pole),
  8558 (2014.5, pole). Media index with every film's timing and moments, every photo's group,
  and the report page sources: `nhtsa/media/<test>.json` (hand notes kept in
  `tools/crash-reference/camry-side-notes.json`). Numbers derived from the curves:
  `nhtsa/<test>.json` (`nhtsa_reference.py`) and `side_compare.py sensors` (below).
- Rubble: the **generated sedan** (1,448 kg) in every row unless "ETK" (the imported BeamNG ETK
  K-series `local:etkk_tex/etkk/kc6.pc`, 1,453 kg). Run 0 (nominal). The contact setting is
  named on every number and picture: S14 with edge or continuous contact **creates energy**
  (a known fault, `crash-reference.md` "Currently failing": here 28.4 kJ edge, 12.0 kJ
  continuous, 4 J discrete), so S14's physics is read from **discrete** contact and the others
  are shown only for comparison. S13 is the same with edge and discrete (a car against a static
  box has no edge pairs); continuous differs slightly.
- Pictures (not committed; third-party footage): `~/Development/rubble-artifacts/crash-reference/camry-side/`
  (`S14-overhead.png`, `S14-rear-of-struck-side.png`, `S14-front-oblique.png`,
  `S13-overhead.png`, `S13-front-ground-level.png`, `S13-rear-ground-level.png`: real | real |
  Rubble columns at -10, 0, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300 ms; `real-<test>.png`:
  five films of one test at the same times; `S14-lateral.png`, `S13-lateral.png`: curves).
  Reference Studio: shots `7517_S14_camry`, `8560_S14_camry`, `7523_S13_camry`,
  `8558_S13_camry`, 48 frames (-10, 40, 80, 300 ms from the overhead and two along-the-car
  films of each).

## 1. Film timing

Every high-speed film of the four tests carries a burned-in caption, "<t> ms ● 1,000 fps ●
Frame: <n>", and the reports' camera tables give 1,000 fps for cameras 1–9 (side MDB) and
2–10 (pole); the real-time cameras are 30 fps. **0.00 ms is Frame 51 on every film used**
(-50 ms at Frame 1); on 8560's impact-point film (C003, on the MDB) the face's edge reaches the
car's impact target between Frames 50 and 52, so the caption's zero is first contact to about
±2 ms (supported by film). The 2012 transcodes (`.mp4` of 7517, 7523) repeat each film frame
three times (1,082 file frames for 361 film frames); the 2014 ones do not (361). Film time:
`ms = Frame − 51`; file frame `(Frame − 1) × 3` (2012) or `Frame − 1` (2014). Real-time films
have no caption: their first contact was not established.

## 2. Setup: the tests against the scenarios

| | 7517 | 8560 | S14 (Rubble) |
|---|---|---|---|
| struck car, as tested | 1,641 kg (curb 1,452, two dummies, ballast) | 1,649 kg | 1,448 kg (generated sedan, no dummies) |
| wheelbase | 2,774 mm | 2,768 mm | 2,830 mm |
| barrier | 1,361.5 kg, face 1,676 mm | 1,361.6 kg | `mdb214` 1,367.6 kg |
| speed | 62.3 km/h | 61.9 km/h | 61.96 km/h |
| crab; MDB axis to car | 26.9°; 90.1° | 26.8°; 90.3° | 27°; 90° |
| face's forward edge behind the front axle | 463 mm (intended 447: 940 mm ahead of mid-wheelbase) | 455 (444) | 475 mm (the same rule on its 2,830 mm wheelbase) |
| car's CG behind the front axle | 1,188 mm, 38 mm left of centre | 1,156 mm, 37 mm left | not read (see §6) |

| | 7523 | 8558 | S13 (Rubble) |
|---|---|---|---|
| car, as tested | 1,566 kg | 1,579 kg | 1,448 kg |
| speed, angle | 32.1 km/h, 75° | 31.95 km/h, 75° | 32 km/h, 75° |
| pole | rigid round, 254 mm | same | square post, 254 mm (stated choice) |
| **impact line on the side, behind the front axle** | **1,101 mm** (front door) | **1,125 mm** | **1,355 mm** (front door); was 1,996 mm (rear door) |
| the line ahead of the car's centre of mass | about 0.06–0.09 m | same | 0.075 m; was 0.57 m behind it |
| the line ahead of the B-pillar | about 0.37 m | same | 0.28 m |

Setup findings:

1. **S13's pole was 0.9 m too far back; fixed.** The scenario had aimed at the generated
   sedan's driver seat, and the generator stands that seat behind the B-pillar, beside the rear
   door (seat 1.64–2.25 m behind the front axle, B-pillar 1.635 m, front door 0.57–1.60 m: node
   positions in `out/gen/sedan_new`). A real front seat is ahead of the B-pillar; this is a
   fault of the generated sedan (reported on PR 163), not of the procedure. The tests' line by
   the axle (1.113 m) does not transfer either: the sedan is rear-wheel drive with its centre of
   mass 1.43 m behind the front axle (0.51 of the wheelbase; the Camrys 1.16–1.19 m, 0.42), so
   that line is 0.31 m ahead of its centre of mass and the car swings the other way. Measured,
   edge contact, run 0:

   | impact line behind the front axle | relative to the centre of mass | yaw at 300 ms | pole force |
   |---|---|---|---|
   | 1.996 m (the old aim, rear door) | 0.57 m behind | 38.7° nose-left | 307 kN |
   | 1.113 m (the tests' line by the axle) | 0.31 m ahead | 19.9° nose-right | 342 kN |
   | **1.355 m (S13 now)** | 0.075 m ahead, as the tests | 3.6–4.1° nose-right | 275 kN |
   | tests | 0.06–0.09 m ahead | 8–10° nose-left (±30 %) | 191–197 kN |

   S13 now keeps the tests' lever arm about the centre of mass and strikes the front door.
2. **S14's setup is right.** The face's forward edge is on FMVSS 214's line by the same rule as
   the tests (0.475 m behind the front axle; the tests 0.455–0.463 m). The generated sedan's
   front door begins 0.57 m behind the front axle (a Camry's 0.41 m), so the face's first 0.10 m
   is on the fender's rear edge instead of 0.05 m inside the door: a small difference of the
   car, not a setup fault.
3. Mass: the generated sedan is 12 % lighter than the MDB Camrys and 8 % lighter than the pole
   Camrys (§6 computes what this alone changes). The barrier mass and speeds match within 0.5 %.
4. The tests' cars carry ES-2re / SID-IIs dummies and ballast; the scenario has none. Their mass
   is in the test mass; their load on the door is not modelled (they load the door from inside
   late in the event, inferred, not measured here).
5. The pole is a square post in S13 (a stated choice): the tests' cylinder wraps; this changes
   the contact shape after a few centimetres of crush (inferred, not isolated).
6. S14 lets the struck car roll on its tyres; the tests did too (inferred: the reports
   photograph a disengaged parking brake, photos "Pre-Test View of Disengaged Parking Brake").

## 3. What each real crash does over time

Times in ms after first contact; film and frame given (Frame = burned-in number). Sensor
numbers from `side_compare.py sensors` (CFC 60, integrated from contact, +Y right; table in
§5).

### Side MDB, 7517 and 8560 (they agree closely)

| ms | what happens | evidence |
|---|---|---|
| 0 | the MDB face's edge at the impact target | 8560 C003 Frames 50–52 |
| 4–7 | **sharp first spike**: CG 31.8 / 35.6 g at 6.1 / 7.4 ms, left rear sill 65 / 53 g at 4–6 ms, floor and right sills 35–50 g at 5–7 ms: the barrier's bumper element (432 mm centre) meets the sill and floor directly | EV5 curves |
| 10 | CG already +2.0–2.1 m/s across; side airbag inflating | curves; 8560 C004 Frame 61 |
| 15 | curtain airbag across both side windows | 8560 C004 Frame 66 |
| 20 | belt line visibly bowed in at the front door; CG +3.2 m/s (half its final change by 21–22 ms) | 8560 C004 Frame 71; curves |
| 25–30 | front door glass crazed (25), buckling outward (30) | 8560 C004 Frames 76, 81; 7517 C004 Frame 81 |
| 30–50 | second, broader pulse 13–16 g; glass bursts and sprays above the roof (40); MDB peak decel 18–19 g at 27–35 ms | C004 Frame 91; C005 Frame 81; curves |
| 50 | CG +6.3–6.4 m/s: the momentum exchange is essentially over; MDB −7.4 m/s | curves |
| 60–100 | the face buried in the front door and B-pillar foot; the car moves across, the MDB stays against it (the closing speed along the MDB axis is still +0.6 m/s at 150 ms: no separation); front fender undisturbed; no wheel lift seen; roof rail line stays straight | C001/C002 Frames 111–151; C006 Frame 111; C005 Frame 131; curves |
| 88 / 110 | CG lateral change at its largest, 6.60 / 6.79 m/s | curves |
| 100–300 | the car slides across on its tyres, losing about 1.6 m/s of lateral speed by 300 ms (sliding friction about 0.6–0.8 g; part of the drop is the accelerometer axis turning with the yaw, inferred); it turns **nose-left (toward the barrier) about 8° by 100 ms, 14–15° by 150 ms, 30–33° by 300 ms**; it has moved 1.65–1.70 m across by 300 ms | curves (yaw from the engine and trunk-floor accelerometers, ±30 %); C001 Frames 151–351 |
| after | doors jammed shut, latches held, no pillar or sill separation, front door glass broken; maximum static crush 275 / 258 mm at mid-door, 450 / 1,350 mm behind the impact line; ≥ 200 mm of crush at hip and mid-door levels from 150 to 1,950 mm behind the line (7517); window top only 32 mm; B-post width 197 / 352 mm less; MDB honeycomb crushed 152–179 mm at the bumper, 55–105 mm above | reports p. 22–29 / 25–27; photos P008, P010, P016 |

### Side pole, 7523 and 8558

| ms | what happens | evidence |
|---|---|---|
| 0 | the pole at the front door on the roof's head-CG cross tape | C001 Frame 51; 8558 C004 Frame 51 |
| 0–10 | door skin takes it alone: CG +0.05–0.12 m/s by 10 ms | curves |
| 13–15 | 8558: left front sill 41 g at 13 ms, CG 17.9 g at 15 ms (the sill reaches the pole); 7523 rises more slowly (peak 16.5 g at 34 ms) | curves |
| 20 | door glass crazed | 8558 C004 Frame 71 |
| 30–40 | glass broken, the pole past the window line, the roof cross tape bending round it; A- and B-post sills at 23–29 g (7523, 30 ms) | C004 Frame 91; curves |
| 44–46 | pole force at its peak, 191 / 197 kN | `nhtsa/<test>.json` |
| 60 | door, roof rail and roof cross tape in a V round the pole; CG half of its change (60–62 ms) | C001, C004 Frame 111; curves |
| 97–112 | the car stops along its travel after 0.46–0.51 m of CG travel (dynamic crush) | `nhtsa/<test>.json` |
| 100–150 | deepest wrap (the head-CG target near the pole, about 100 ms); the pulse runs to about 150 ms; engine and trunk floor catch up late (half-dv at 74–85 ms) | C004 Frame 151; curves |
| 150–300 | rebound 1.7–2.2 m/s back off the pole; the car stays square: **nose-left about 4.5° by 150 ms, 8–10° by 300 ms**; no roll seen on the front and rear films | curves; C002 / C006 Frame 351 |
| after | maximum crush 394 / 379 mm at mid-door at the pole (8558: sill 340, window sill 340); crush falls to 0 about ±1 m from the pole; wheelbase 129 / 106 mm and left side 135 / 131 mm shorter; B-pillar width 111 / 257 mm less; windshield cracked; front door glass broken; doors jammed | reports p. 20–26 / 22–26; photos P008, P016, P022 |

## 4. Rubble over the same times

From the curves (`S14-lateral.png`, `S13-lateral.png`), the stills, and the crash_ref reports
(`out/camry-side/runs/`).

**S14, discrete contact (the setting that holds every law):** no first spike; the CG's lateral
acceleration ramps from about 10 g at 10 ms to 17 g at 40–50 ms and ends at 75 ms. Its lateral
change is 0.58 m/s at 10 ms (real 2.0–2.1), 1.69 at 20 (3.2), 6.31 at 50 (6.3–6.4), and keeps
rising to **9.27 m/s at 96 ms** (6.6–6.8); half by 40 ms (21–22). The barrier loses 11.2 m/s
(8.1–8.6). At 150 ms the car moves across at 9.15 m/s while the barrier's speed along its own
axis is 4.15 m/s: **the car has left the barrier at 5.0 m/s** (real: still closing at 0.6 m/s).
The door band peaks at 156 mm of crush and springs back to 51 mm (real 258–275 static); the
cabin-width gauge across the B-pillar changes by 15 mm at most and 2 mm at the end (real 197–352
mm). Plastic work: car 13.0 kJ, barrier 35.6 kJ, of 128 kJ of kinetic energy lost (the rest
goes to ground friction, damping and contact; not separated here). Yaw nose-left 4° at 100 ms,
8° at 150 ms, 18° at 300 ms, 31.6° at 600 ms; roll peak 3°; no part detaches. The overhead
stills show a shallow even dent along the door band, nothing like the face buried in the door.
- edge contact (creates 28.4 kJ): the door band goes to 431 mm and stays at 368 mm, the car
  leaves at 8.37 m/s; continuous (creates 12.0 kJ): 236 / 144 mm, 8.69 m/s. Neither is physics
  while it creates energy; they show only that a softer-looking side is what the contact faults
  produce.
- ETK, discrete: no early spike either (it starts later still: 0.65 m/s at 20 ms), 9.47 m/s
  across, residual door-band crush 245 mm, yaw 47° by 600 ms, one part detaches; 110 J created
  (0.05 % of the energy).

Plan-view crush along the side (`side_compare.py profile`: each left-side group's boundary in
the top camera against its rigidly carried rest shape, deepest point per 150 mm at any height;
a proxy for the reports' grid, mm, + behind the impact line):

| behind the line (mm) | 7517 level 3 (mid-door) static | 8560 level 3 | S14 discrete, 80 ms | S14 discrete, 300 ms | S14 edge, 300 ms |
|---|---|---|---|---|---|
| 0 | 76 | −27 | −10 | 4 | 12 |
| 300 | 254 | 174 | 128 | 123 | −9 |
| 600 | 264 | 221 | 95 | 16 | 104 |
| 750 | 263 | 213 | 97 | 89 | 68 |
| 1,200 | 239 | 248 | 110 | 112 | 234 |
| 1,350 | 241 | 258 | 145 | −9 | 384 |
| 1,500 | 238 | 225 | 113 | 90 | 275 |
| 1,800 | 227 | 206 | −11 | −13 | 59 |

**S13, edge contact (the moved pole):** the CG starts at once (0.87 m/s at 10 ms against
0.05–0.12; 2.5 at 20, 7.6 at 50), has half its change by 43 ms (60–62) and **stops along the
travel at 58 ms after 0.27 m** (97–112 ms, 0.46–0.51 m); it is then **thrown back at 4.0 m/s**
(1.7–2.2), a lateral change of 12.7 m/s (10.3–10.9). Pole force **275 kN at 24 ms** (191–197 kN
at 44–46). Crush peak 341 mm, residual 110 mm (379–394 static). Yaw: nose-right 2° from 40 to
150 ms, 4° at 300 ms (real nose-left 4.5° at 150 ms, 8–10° at 300): small on both sides now. In
the overhead stills the post meets the front door just ahead of the B-pillar, the side dents
round it to 60–100 ms, and by 200 ms the car has left the post with most of the dent gone; the
real cars stay wrapped round the pole. Plan-view crush: 330 mm at 60 ms and 123 mm at 300 ms,
0.3 m behind the line (8558: 340–379 mm at the pole, ≥ 200 mm from −225 to +225, about 0 at ±1 m).
All of S13's asserted rules hold on 8 jittered runs with edge and with continuous contact.
- continuous: 271 kN at 20 ms, thrown back at 3.7 m/s, crush 355 / 296 mm; its cabin-width
  gauge moves 291 mm at the peak and 257 mm at the end (edge: 100 / 48 mm): unexplained (a
  B-pillar gauge node near the post, inferred).
- ETK, edge, struck 1.355 m behind its own front axle: 427 kN at 28 ms, thrown back at 2.2 m/s,
  crush 342 / 144 mm, yaw 25°. Continuous contact refuses to start on the ETK (skin pairs
  closer than 2 mm at spawn: the import's authoring).

## 5. Sensor detail (the real tests)

`side_compare.py sensors` output, dv in m/s from contact (+Y right):

| channel | 7517 peak (g) at ms | 7517 dv 10 / 20 / 50 / 150 | 8560 peak at ms | 8560 dv 10 / 20 / 50 / 150 |
|---|---|---|---|---|
| CG Y | 31.8 @ 6.1 | 2.13 / 3.15 / 6.34 / 6.22 | 35.6 @ 7.4 | 1.96 / 3.17 / 6.40 / 6.35 |
| left rear sill Y | 65.5 @ 4.1 | 2.42 / 2.96 / 6.99 / 7.37 | 53.5 @ 5.7 | 2.15 / 2.44 / 6.75 / 7.00 |
| right front sill Y | 35.4 @ 6.1 | 2.26 / 3.16 / 6.45 / 6.03 | 45.6 @ 7.3 | 2.22 / 3.32 / 6.95 / 6.67 |
| trunk floor Y | 32.2 @ 6.1 | 1.76 / 3.52 / 9.78 / 10.35 | 36.2 @ 7.9 | 1.69 / 3.71 / 9.91 / 10.68 |
| engine Y | 25.0 @ 39.7 | −0.03 / −0.20 / 4.35 / 4.06 | 27.0 @ 42.8 | −0.01 / −0.17 / 4.32 / 4.26 |
| MDB CG X | −18.3 @ 34.5 | −0.93 / −2.51 / −7.41 / −8.56 | −19.4 @ 26.6 | −0.94 / −2.60 / −7.47 / −8.15 |
| yaw (deg, + nose right) | | −0.1 / −0.6 / −3.3 / −13.6 (−30.2 at 300) | | −0.1 / −0.6 / −3.7 / −15.1 (−32.6 at 300) |

| channel | 7523 peak at ms | 7523 dv 20 / 50 / 100 / 150 | 8558 peak at ms | 8558 dv 20 / 50 / 100 / 150 |
|---|---|---|---|---|
| CG Y | 16.5 @ 33.7 | 0.92 / 3.74 / 7.64 / 10.26 | 17.9 @ 14.7 | 1.09 / 4.31 / 8.71 / 10.82 |
| left B-post at sill / mid B-post Y | 29.0 @ 30.2 | 1.67 / 4.71 / 6.52 / 7.96 | 21.9 @ 15.2 (mid) | 1.59 / 4.79 / 6.76 / 7.89 |
| engine Y | 20.3 @ 55.0 | −0.03 / 1.02 / 6.40 / 10.32 | 22.2 @ 55.1 | −0.03 / 0.93 / 6.46 / 9.97 |
| trunk floor Y | 14.0 @ 70.6 | 0.58 / 3.20 / 8.84 / 11.89 | 17.8 @ 50.4 | 0.32 / 3.09 / 9.64 / 12.97 |
| yaw (deg) | | −0.07 / −0.95 / −2.6 / −4.4 (−8.0 at 300) | | −0.02 / −0.66 / −2.2 / −4.6 (−9.7 at 300) |

Yaw is the engine's and the trunk floor's lateral velocities' difference over their distance
along the car (3.25, 3.11, 3.14, 3.86 m from the reports' accelerometer tables), integrated: the
engine sits on mounts and lags (its dv is near zero until 20–30 ms), so the yaw is good to about
±30 % (the right sills' front-to-rear difference over 0.98 m gives the same sign and a similar
rate). The 2012 left mid A-post channel is not usable (79 g at 0.1 ms, its dv drifts to −3 m/s);
7517's left lower A-post and left lower and mid B-post were void (report p. 7).

## 6. What mass alone changes

The generated sedan is 1,448 kg against 1,641–1,649 kg (MDB) and 1,566–1,579 kg (pole).

- **S14.** In a perfectly plastic hit the struck car leaves across at
  M·V·cos 27°/(M + m): 6.99 m/s for a 1,641 kg Camry at 62.3 km/h and 7.45 m/s for the 1,448 kg
  sedan (+6.6 %). The Camrys' 6.60–6.79 m/s, scaled by that ratio, would be **7.0–7.2 m/s** on
  the sedan's mass. Rubble's 9.27 m/s (discrete) is 29–32 % more than mass explains, and more
  than a plastic hit can give: the excess is rebound between the side and the barrier. The
  barrier's plastic loss for the sedan's mass is 7.9 m/s (8.4 for a Camry); the Camrys' 8.1–8.6
  scale to **7.6–8.0 m/s**; Rubble's 11.2 m/s is 40 % more. Mass explains none of the missing
  first spike or the 20 ms lag (it would make the sedan's response slightly faster, not slower).
- **S13.** Against a rigid pole the velocity change does not depend on mass. For the same side
  stiffness a lighter car's peak force scales with √m (a linear spring: F = v√(k m)): the
  Camrys' 191–197 kN become **184–189 kN** at 1,448 kg, and their pulse 4 % shorter. Rubble's
  275 kN is 1.5 times that, and it comes 20 ms earlier. Residual crush for the same energy per
  mass would be unchanged by mass.

## 7. Differences, ranked, with causes

Confidence: **measured** (numbers from both sides), **film** (seen on the films and the
stills), **hypothesis** (a likely cause, not isolated by an experiment here).

### S14 (side MDB), against discrete contact

1. **The side springs back instead of crushing: the car leaves 30 % faster than mass explains
   and separates from the barrier** (9.27 vs 7.0–7.2 mass-scaled; barrier −11.2 vs −7.6–8.0;
   5.0 m/s separation at 150 ms vs still in contact; door band 51 mm residual vs 258–275 mm;
   B-pillar width change 2 mm vs 197–352 mm). Measured. Most likely cause: the generated
   sedan's side (door, B-pillar, sill) yields too little and stores the load elastically, and
   the barrier face (`mdb214`) gives back what it takes (35.6 kJ of its plastic work, but the
   pair still rebounds). This is the doc's existing "elastic side" observation, now with the
   timeline: the energy goes into the car's motion after 50 ms instead of into the door.
   Authoring (side structure strength and yield) first; the barrier's honeycomb law second
   (its 152–179 mm bumper crush in the tests is a check on `mdb214`, not done here). Measured
   (effect), hypothesis (split between car and barrier).
2. **No first spike and a 20 ms lag at the start** (CG 0.6 vs 2.0–2.1 m/s at 10 ms; peak 17 g at
   40 ms vs 32–36 g at 6–7 ms then 13–16 g at 30–50). Measured. In the tests the barrier's
   bumper element (centre 432 mm up, 152–179 mm of crush) hits the sill and floor directly and
   the whole floor answers within 4–7 ms (left rear sill 54–66 g at 4–6 ms, right sills at 5–7
   ms: the load crosses the car through the floor at once). Rubble's load enters through door
   skin and reaches the floor late. Likely causes: (a) the barrier's lower element or the
   sedan's sill not meeting at the same height and stiffness (authoring of `mdb214` and the
   sill: the sill's top is 295–302 mm in the tests; not measured on the sedan here); (b) not the
   impact line, which is right (setup finding 2). Hypothesis.
3. **Too little yaw early, then yaw keeps growing** (4° vs 8° at 100 ms, 8° vs 14–15° at 150 ms,
   18° vs 30–33° at 300 ms). Measured (both from different methods: Rubble from the rigid fit
   of cabin and rear, the tests from two accelerometers, ±30 %). In the tests the rear is
   pushed harder (trunk floor 10.4–10.7 m/s vs engine 4.1–4.3 at 150 ms) because the
   barrier's 1.68 m face sits mostly behind the CG and keeps pushing; in Rubble the car leaves
   the face early (difference 1), so the turning impulse stops. Hypothesis (follows from 1).
4. **Where the crush is**: Rubble's dent is shallow and broken into patches (95–145 mm at 80
   ms; edge contact puts it at 1.2–1.5 m behind the line); the tests' is a continuous trough
   200–275 mm deep from 150 to 1,950 mm behind the line, deepest at the front door's rear and
   the B-pillar foot. Film and measured (proxy). Cause: 1.
5. **What breaks and detaches**: the tests break the front door glass (25–40 ms) and jam both
   left doors; nothing detaches from the generated sedan (glass is visual-only by design). Film;
   by design, not a defect.
6. Contact settings: edge and continuous create 28.4 and 12.0 kJ, and with them the door band
   crushes 3–7 times more. The doc's known fault; any agreement in crush depth under edge
   contact is an artefact. Measured.

### S13 (side pole), edge contact, the moved pole

1. **The side springs back: the car is thrown off the pole at 4.0 m/s** (real 1.7–2.2), and its
   dent recovers from 341 mm to 110 mm (real 379–394 mm left in the car). Measured and film.
   45 % of the approach speed comes back, 20 % of the energy. Cause: the door, sill and
   B-pillar yield too little and store the load; the same fault as S14-1, now seen without a
   barrier in the way. Authoring of the side structure first; the solver's limit on members
   folding (beam-yield-law.md §9) may share it. Hypothesis on the split.
2. **Too stiff and too early**: 275 kN at 24 ms vs 184–189 kN (mass-scaled) at 44–46 ms; stopped
   at 58 ms after 0.27 m vs 97–112 ms after 0.46–0.51 m. Measured. Causes: as 1; the square
   post spreading the load (a stated choice; small). The ETK is stiffer still (427 kN), so this
   is not one generator's choice.
3. **The start**: Rubble's CG moves at once (0.87 vs 0.05–0.12 m/s at 10 ms): the door skin and
   the structure behind it engage together in Rubble; on the Camrys the door skin crushes alone
   for 10–15 ms before the sill / B-post sill reach the pole (13–15 ms on 8558, 30 ms on 7523).
   Measured. Cause: little stand-off and little skin softness between the outer surface and
   the sill / pillar in the generated sedan's side (authoring). Hypothesis.
4. **Wrap**: the tests' side folds into a V around the round pole (wheelbase and left side
   106–135 mm shorter, roof rail bent, windshield cracked); Rubble's dent is a shallow trough
   130–240 mm deep spread from 0.3 m ahead to 0.6 m behind the post, with little length lost
   by its gauges (`wheelbase_left` changes 35 mm at the end). Film and measured. Causes: 1 and
   2; the square post (cannot wrap as a cylinder does).
5. Yaw: 4° nose-right against 8–10° nose-left (±30 %). Both small; the remaining difference is
   within what the pole's friction along the side and the wrap would turn (not isolated).

## 8. What would have to change to reproduce each real result

Ranked by how much each would close the gaps above. Nothing here was tried.

**Scenario setup**
1. S13: done (§2). What remains is the car: the generated sedan's front seats belong ahead of
   the B-pillar, and its centre of mass is 0.27 m further back than a Camry's.
2. S14: nothing to change (§2).
3. Report both at the Camry mass (ballast to 1,641 / 1,566 kg) when a like-for-like number is
   wanted; §6 shows mass explains 6–7 % of the S14 exit speed and none of the shape.

**Authoring (generator / imported cars)**
4. The generated sedan's side: door, B-pillar and sill strength and yield so the side crushes
   250–400 mm and stays crushed instead of springing back (S14-1, S13-2, S13-4; the residual
   B-pillar width loss of 110–350 mm is a target). The ETK shows the same stiffness, so this is
   not specific to one generator choice (hypothesis).
5. The front seats' position (behind the B-pillar; they belong ahead of it) and the front door's
   start (0.57 m behind the front axle; a Camry's 0.41 m).
6. The sill height and the barrier bumper's height match, so that the first spike appears
   (S14-2): measure the sedan's sill top against 295–302 mm.
7. ETK: its collision skin has pairs closer than 2 mm at spawn, so continuous contact refuses it.

**Solver / contact**
8. Edge and continuous contact creating energy between the barrier and the door (known fault):
   until fixed, S14's crush depth can only be read from discrete contact. Fixing it does not by
   itself make the side plastic (discrete contact, which holds every law, is the elastic one).
9. The barrier `mdb214`: its face's crush law against the tests' honeycomb crush (152–179 mm at
   the bumper, 55–105 mm above) and whether it rebounds; not measured here (unknown).

**Unknown**
10. How much of the real car's late lateral slow-down (6.6 to 5.0 m/s by 300 ms) is tyre
    sliding and how much the accelerometer turning with the yaw; and what Rubble's contact
    removes in each run (contact work is not reported apart from friction and damping by the
    suite's metrics).

## 9. What could not be determined, and what is unverified

- First contact on the real-time films (no caption); the documentation and web-cut videos were
  not viewed (their notes say so).
- Real yaw is from two accelerometers (±30 %), not from the overhead film (the overhead cameras
  are close and the car leaves the frame); Rubble's yaw is the rigid fit of cabin and rear.
- Rubble's crush profile is a plan-view proxy (deepest at any height per 150 mm slice of the
  outline groups' boundaries), not the reports' five levels; its door-band crush
  (`side_crush_left_*`) is a single number. The ETK has no outline groups, so no profile or yaw
  curve for it (its end yaw is from the report: 47° S14, 60° S13).
- The two B-post width measurements differ in pre-test value (1,821 vs 1,974 mm): probably not
  the same points; both are quoted.
- The Rubble cameras for the along-the-car views are placed from the camera tables and aimed by
  eye; the overhead ones are oriented by the barrier's travel / the car's travel as the films
  are. Close, not exact; `crash_outlines` draws only collision surfaces (the barrier is its face
  only, the sedan's rear panel is open).
- What contact removes in each crash is not separated (the suite reports kinetic energy lost and
  plastic work; the remainder mixes friction, damping and contact).
- The 2012 tests' A-post channels and some B-post channels are void or questionable (report p. 7).

## 10. Regenerating

From `engine/`, after `cargo build --release -p engine-softbody-lab --example crash_ref --example crash_outlines`:

```
tools/crash-reference/camry_side.sh index runs render real sheets sensors profile studio
PY=<python with matplotlib> tools/crash-reference/camry_side.sh curves
```

(`ROOT` defaults to `~/Development/rubble/inspiration/ccsa-camry-2012`; `RUBBLE_LOCAL_MODS` to
the crash-physics worktree's `__mods` for the ETK.) Each step is a subcommand of
`tools/crash-reference/side_compare.py` (its docstring lists them). Real stills are cut by the
committed index's film timing; Rubble's by `crash_outlines --scenario` on a copy of the
scenario's export with the films' times as moments and `--cameras` for the films' views (the
layout, and so the physics, is the export's own). `crash_ref --export ID --set car=...` now
exports with the car replaced (the change in this branch).

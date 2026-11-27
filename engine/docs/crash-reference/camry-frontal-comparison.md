# The real Camry frontal tests beside Rubble (S11)

The NHTSA NCAP full-frontal tests of the 2012–2014 Toyota Camry (test 8545: 2014.5 car,
Calspan, 56.71 km/h; test 7520: 2012 car, MGA, 56.3 km/h; rigid load-cell wall) read from their
high-speed films, photos, reports and sensor curves, and set beside Rubble's S11 at the same
times and views. This page measures and explains; nothing was tuned.

Sources: the annotated indexes `nhtsa/media/8545.json` and `nhtsa/media/7520.json` (every film's
view, time base, the moments read frame by frame, every photo's subject, the report pages), the
sensor references `nhtsa/8545.json` and `nhtsa/7520.json` (`nhtsa_reference.py`: rear-seat
crossmember X, CFC 60; wall force = sum of the load cells), and Rubble's CPU reference
(`crash_ref`, `crash_outlines`, run 0, edge contact). The footage and photos are local only
(`inspiration/ccsa-camry-2012/`, public domain US government work, never committed).

Images (not committed): `~/Development/rubble-artifacts/crash-reference/camry-frontal/`:
`sheets/side-left.jpg` (real 8545 and 7520 camera 3 | Rubble default | load_path,crush_law |
ETK), `sheets/side-right.jpg` (camera 9 | default | load_path | load_path,crush_law | ETK),
`sheets/pit.jpg` (camera 14 from below | three Rubble cars from below), `sheets/overall-right.jpg`
(camera 7 | three Rubble cars), `sheets/pulse.png` (speed, travel, wall force), `strips/` (each
film at 12–24 times), `measure/` (the gridded crops the numbers below were read from).
Rows of every sheet: 0, 10, 20, 30, 40, 55, 70, 85, 100, 121, 150, 290 ms after first contact.

Confidence tags: **measured** (a number from a sensor, the report or pixels with a stated
scale), **film** (seen in the films at the stated frame, not measured), **hypothesis**
(the most likely reading; what would settle it is stated).

## 1. Time base: how film frames become ms

- 8545: 14 high-speed cameras at 1000 fps (report Data Sheet 6, PDF p. 18), 351 frames each.
  First contact is file frame 50 in every one: six films' captions read the frame's time
  (−0.03 to −0.76 ms at frame 50), five read "T0: 50", and in camera 3 the bumper touches the
  wall at frame 50 (checked at 50 and 52). **Measured** cross-check: the door targets' travel
  in camera 3 (1.885 mm/px, set from the wheel rim and the sensor at 20 ms) is 0.30, 0.43, 0.53,
  0.62, 0.65 m at 20, 30, 40, 55, 70 ms against the sensor's 0.296, 0.423, 0.528, 0.632,
  0.663 m: the film's clock and the sensor's agree to about 1 ms.
- 7520: 1000 fps cameras, each camera frame written 3 times (1082 file frames); every film's
  caption reads "0,00 ms … Frame: 51" at file frames 150–152. So ms = (file frame − 150) / 3.
  The bumper is at the wall by eye at caption 0 (within 2 ms; `measure/t0_7520.png`).
- `camry_frontal.py` applies these from the index notes (`t0_frame`, `fps_film`,
  `file_frames_per_film_frame`).

## 2. What the real car does, in time (8545; 7520 the same sequence by eye)

| ms | what happens | read from | sensor at that time (8545) |
|---|---|---|---|
| 0 | bumper fascia on the wall | camera 3, frame 50 | — |
| 0–13 | fascia, foam and bumper beam flatten | camera 3 frames 50–60 (film) | first force rise: 258 kN at 9 ms (7520: 304 kN at 13 ms) |
| 15–20 | hood's front edge lifts off the latch line; by 20 ms the front third is buckled up with a gap under it | camera 3 frames 65, 70 (film) | 240–290 kN plateau (crash zones crushing: `front-load-path.md` §1.4) |
| 20–30 | airbags out (camera 11 at frame 80) | film | — |
| 29–30 | **first force peak** 916 kN (7520: 829 kN at 30 ms). The hood is a tent, its ridge about a third of the hood back from the front (camera 3 frame 80, camera 9 frame 80–90) | film; travel 0.42 m (measured) | `front-load-path.md` §1.4 reads this as the engine stopped against the wall through the radiator stack (block 0.337 m behind the beam) — **hypothesis** consistent with the travel; the films cannot see the engine at this moment (the pit view shows only the subframe band, camera 14 frame 80) |
| 30–45 | force falls to 276 kN at 40 ms; the tent rises; the fender crumples behind the headlamp; the front wheel still moves with the body (hub to door-corner target unchanged, 250 px) | camera 3 frames 80–90 (**measured**, `measure/c009_wheel.png`) | — |
| 45 | **front hub stops** (its image x 268–285 px from 45 to 75 ms) while the body goes on | camera 3 frames 95–125, every 5 (**measured**, `measure/c009_setback.png`) | force rising again |
| 53–56 | **second force peak** 628 kN (7520: 583 kN at 56 ms). Wheel pushed back: hub to door target −35 px (−66 mm) at 55 ms | camera 3 frame 105 (**measured**) | `front-load-path.md` reads the engine reaching the firewall/rack; the film adds the wheel, stopped since 45 ms, being driven into the hinge pillar and sill (**hypothesis**: both are likely in the same peak; the front tyre's rear is 80–110 mm from the rocker's front, `front-load-path.md` §1.1) |
| 65–75 | largest wheel setback: hub to door target −60 px = **−113 mm** (left), held to 75 ms | camera 3 frames 115–125 (**measured**) | — |
| 71 | **maximum travel 0.663 m**, speed 0 (7520: 0.685 m at 76 ms); door targets 0.65 m from their start (measured) | sensor; camera 3 frame 120 | peak deceleration 39.8 g at 44 ms (7520: 45.0 g) |
| 85 | hood folded into two panels, the front one steep; steam or dust out of the engine bay; debris thrown up | camera 3 frame 135, camera 9 frame 135 (film) | force 91 kN |
| 100–150 | **rear lifts**: belt line about 3° nose down at 100 ms, 3.5–4° at 121–150 ms (±1°, by eye from the belt-line targets); the rear tyre stays on the floor by eye, the rear suspension extends. At 70 ms the belt line is still level within 1° | camera 7 frames 120–200 (`measure/c013_pitch.png`, `strips/8545-C013-rear.jpg`) | — |
| 121 | lab's time of separation (7520: 88.3 ms); hub to door target back to −15 px (−28 mm) | report Data Sheet 13; camera 3 | rebound 1.44 m/s |
| 290 | car 0.34 m back from its deepest (door targets 162 px from start: 0.31 m from contact); a hood panel stands nearly vertical; debris on the floor | camera 3 frame 340 (**measured**) | sensor rebound 1.1–1.5 m/s integrates to about 0.2–0.3 m by 250 ms |
| after | crush C1..C6 445, 494, 529, 523, 468, 409 mm (max 554 mm at the centreline); wheelbase **−66 mm left, −30 mm right**; doors open and close (7520: 364, 478, 531, 535, 489, 400; max 579; wheelbase −56 / −52) | reports, Data Sheets 13–14 | — |

What detaches (film and photos): headlamp and grille pieces and small debris from about 85 ms;
the engine undertray torn and hanging (post-test underbody photo P024); no wheel, door or panel
leaves the car. The windshield stays in (report: 100 % retention).

The pit view (camera 14, from below, wall at the left) shows the engine's underside, the blue
reservoir and the red tow-eye brackets passing toward the wall in the first 20 ms, the black
subframe as a short band between the pit's crossbar and the front floor from 30 ms, and the
front floor still by eye from about 55–70 ms; the films do not show where the engine stops
relative to the wall (the wall plane is not identifiable in this view).

## 3. Rubble beside it

Cars: the generated sedan (1448 kg) as built by default; with `RUBBLE_FRONT_RULES=load_path`;
with `RUBBLE_FRONT_RULES=load_path,crush_law` (CPU only); the imported BeamNG ETK
(`local:etkk_tex/etkk/kc6.pc`, 1453 kg, length 3.92 m: a smaller car than the Camry, 4.81 m).

| | real 8545 | real 7520 | Rubble default | load_path | load_path, crush_law | ETK |
|---|---|---|---|---|---|---|
| mass (kg) | 1664 | 1662 | 1448 | 1448 | 1448 | 1453 |
| first force peak (kN at ms) | 916 at 29 | 829 at 30 | 661 at 20 (the only one) | 190–230 plateau 20–38, then 491 at 44 | 175–240 plateau 16–40, then 493 at 48 | 713 at 28, held ~690 to 44 |
| second force peak (kN at ms) | 628 at 53 | 583 at 56 | none | ~450 at 60–64 | ~360 at 64 | none |
| peak deceleration, CFC 60 (g at ms) | 39.8 at 44 | 45.0 at 44 | 59.1 at 15.5 | 55.0 at 53.5 | 50.6 at 54.5 | 50.8 at 27 |
| speed 0 (ms) | 71 | 76 | 44 | 73 | 77 | 45 |
| dynamic crush (m) | 0.663 | 0.685 | 0.394 | 0.700 | 0.702 | 0.437 |
| rebound (m/s) / restitution | 1.50 / 0.095 | 0.89 / 0.057 | 5.10 / 0.33 | 3.28 / 0.21 | 2.56 / 0.16 | 4.56 / 0.29 |
| kinetic energy left at the end (% of start) | 0.9 | 0.3 | 10.8 | 4.6 | 2.8 | 8.6 |
| residual length loss (mm) | 554 (centre) | 579 | 199 | 539 | 560 | 253 |
| springback, dynamic → residual | 0.663 → 0.554 (16 %) | 0.685 → 0.579 (15 %) | 0.42 → 0.20 (53 %) | 0.70 → 0.54 (23 %) | 0.70 → 0.56 (20 %) | 0.47 → 0.25 (46 %) |
| wheelbase change, dynamic (mm) | −113 left (film, 65–75 ms) | — | 56 / 57 (largest change, either sign) | 110 / 97 | 61 / 33 | not measured |
| front hub moved back vs the cabin (mm, left) at 55 / 70 ms; negative = forward | 66 / 109 (film, measured) | — | −21 / −1 | 32 / 74 | 31 / 20 | not valid |
| wheelbase change, residual (mm) | −66 L / −30 R | −56 / −52 | −28 / −11 | −41 / −16 | −9 / −16 | not measured |
| hood ridge above its rest line | ≈ 0.1 m at 20 ms; 0.3–0.4 m at 30–55 ms (film, ±0.1 m) | same by eye | 39 / 113 / 212 mm at 20 / 30 / 55 ms; 80–115 later | 53 / 129 / 240; 324 at 85 | 53 / 130 / 243; 344 at 85; 263 at rest | not measured |
| pitch (+ nose down) | ≈ 0 at 70 ms; +3 to +4° at 100–150 ms | — | +2.6° at 55, then −0.8 to −1.1° | −1.1° at 55, −1.4 to −1.9° at 121–150 | −1.1° at 55, −1.4 to −2.1° at 121–150 | +4.6° peak (crash_ref) |
| detached | debris, undertray hanging | — | nothing | nothing | 2 parts, 0.6 kg | nothing |
| front plastic work in the upper front (hood, fenders, aprons) | — | — | 22 % | 54 % | 62 % | 8 % |

Rubble's numbers: `crash_ref -- S11 --run 0 --contact edge --json` (metrics `car.ref.*` for the
rear-seat location, `all.box_force_*`, `car.gauge.*`, `car.zone.*`), and per moment
`camry_frontal.py shape` over `crash_outlines`' outlines (hub setback and hood rise relative to
the cabin-and-rear ghost; `rear_pitch_deg`). The wall force is Rubble's `box_force_n` (momentum
change, 4 ms samples) against the load-cell sum. The ETK has no part groups in the outline census,
so its ghost does not follow the cabin: its hub and front-setback numbers from outlines are not
valid and are left out.

### 3.1 By phase

**0–13 ms (bumper).** All four rise to 250–350 kN by 10–12 ms, like the tests (258/304 kN).
*Measured.* Close enough.

**13–30 ms (crash zones, then the first peak).** Real: a 240–290 kN plateau, then a sharp
rise to 916 kN at 29 ms while the hood tents. Default: a single 661 kN peak at 16–20 ms and a
slow fall — the front is too stiff early (`front-load-path.md` §4: SUPPORT stops and the box
bay's braces, 1.6 MN of springs that never yield) and stops the car at 44 ms with 0.39 m of crush.
load_path variants: the plateau is right in level (190–240 kN) but **there is no 30 ms peak**;
their peak comes at 44–48 ms. *Measured* (pulse). Most likely cause: in the generated car the
engine's front is 0.579 m behind the nose against the Camry's 0.491 m (scaled 0.499) and the
firewall 0.24 m further back (`front-load-path.md` §2), so the powertrain reaches the wall about
0.08–0.1 m later in travel; at the lp cars' travel curve 0.42 m (the real peak's travel) comes at
30 ms and 0.5–0.55 m at 40–45 ms, where their peak sits. *Hypothesis, supported by the geometry*;
settled by logging the engine block's nodes' distance to the wall against time in S11. ETK:
one broad 700 kN block from 24 to 44 ms; it is a shorter car with its own layout.

**Hood.** Real: buckles at the front by 15–20 ms, a tent with its ridge a third back by 30 ms,
0.3–0.4 m high by 30–55 ms, folded into two panels by 85 ms, a panel standing almost vertical at
290 ms (and in every post-test photo the hood is a high ridge across the front). Rubble: the
hood rises later and less (default 0.11 m at 30 ms, 0.21 m at 55 ms, then **springs back** to
0.08–0.12 m; load_path,crush_law 0.13 m at 30, 0.24 at 55, 0.34 at 85, 0.26 at rest), with its
ridge at the hood's middle row. *Measured (Rubble), film (real, ±0.1 m).* Causes: (1) the
generated hood has 4 rows of nodes (5 with load_path), so it can bend at two lines at most and
the ridge falls on a row, not a third of the way back (`crash-reference.md`, Outlines) —
*authoring, measured*; (2) in the default car the hood and fenders carry 272 + 2 × 155 kN of
yield across the box bay (`front-load-path.md` §2, "parallel paths"), so they push rather than
buckle — *measured in that doc*; (3) the default hood's springback says its beams stay largely
elastic (see "rebound") — *hypothesis*.

**45–75 ms (wheel and second peak).** Real: the front hub stops at 45 ms, the wheel is driven
back 113 mm toward the door by 65–75 ms, the second peak 628 kN at 53 ms; the left wheelbase is
66 mm short after the test. Rubble (hub against the cabin-and-rear ghost, `camry_frontal.py
shape`; + back): the default's hub does not go back, it sits 21–25 mm *forward* of its place at
55 ms, and the car has no second peak; load_path drives the hub back 74–91 mm at 70–85 ms,
closest to the film; load_path,crush_law back 20–31 mm on one side only, then 60 mm forward on
the left at 100 ms. (`crash_ref`'s wheelbase gauges give 56/57, 110/97 and 61/33 mm as their
largest change, of either sign.) *Measured.* Most likely cause: **the
tyres have no collision shape** (raycast wheels, `generated-vehicles-lessons.md` "raycast tyres
have no collision shape"), so the real path — tyre and rim pinched between the crushed stack at
the wall and the hinge pillar / sill — does not exist; the hub moves back only as far as the
crushed body carries it. *Hypothesis, supported by the film's stopped hub and by Rubble's small
setback*; settled by measuring in Rubble the wheel's distance to the wall stack (it should reach
zero near 45 ms in a 0.66 m crush) and the force through the hub. The second peak in the
load_path cars (~360–450 kN near 60–64 ms) is lower than the real 628 kN, consistent with one
of its two paths (wheel; engine to firewall) missing.

**Stop and maximum crush.** Real stops at 71/76 ms with 0.66/0.69 m. load_path cars: 73/77 ms,
0.70 m — *measured*, close, but at 13 % less mass (below). Default and ETK: 44–45 ms, 0.39–0.44 m:
too stiff. Peak deceleration: real 40–45 g at 44 ms; load_path cars 51–55 g at 54 ms (their
second rise, later); default 59 g at 16 ms.

**Rebound and springback.** The largest difference after the force shape. Real: rebound
1.5 / 0.9 m/s (restitution 0.06–0.10), 0.9 % / 0.3 % of the kinetic energy left; the car comes
back 0.34 m by 290 ms (film); crush springs back 15–16 % (0.66 → 0.55 m). Rubble: rebound
2.6–5.1 m/s (restitution 0.16–0.33), 2.8–10.8 % of the energy returned; the default springs back
53 % of its crush. *Measured.* Most likely cause: energy stored elastically in beams that do not
yield or yield late, released after the stop. `front-load-path.md` §4 measures it: elastic
energy at the stop 47.3 kJ (default) and 25.8 kJ (load_path), and the cabin returns the largest
share (16.6 / 12.7 kJ) — a cabin that should stay rigid but is a spring. The real crumpled
structure folds plastically (little elastic strain to recover) and absorbs further energy in
tearing and friction that Rubble does not model. *Hypothesis for the split between those
mechanisms; the stored elastic energy itself is measured.* Contact is not the cause: it removes
only closing (`crash-reference.md`, energy criteria, 0 J created in every run here).

**Pitch.** Real: level to 70 ms, then the rear lifts to 3–4° nose down at 100–150 ms. Default:
+2.6° nose down at 55 ms then a little nose up; the load_path cars pitch **nose up** (−1° at
55 ms, −1.4 to −2.1° at 121–150 ms); the ETK pitches nose down (+4.6° peak). *Measured (Rubble),
film ±1° (real).* Most likely cause: where the force acts. In the load_path cars 54–62 % of the
front's plastic work is in the upper front (hood, fenders, aprons) against 22 % in the default and
8 % in the ETK (`car.zone.front_upper.work_frac`): a wall force carried high, above the centre of
mass, turns the nose up; the real car's force goes mainly through the rails and subframe, below
it. *Hypothesis, supported by the work split*; settled by logging the wall force's height (centre
of pressure on the wall box) over time and comparing with the test's 36 load cells (rows by
height are in the ISO channels: 8545 has 36 cells, 7520 128).

**Crush profile across the width.** Real: deepest at the centre (529/523 mm) and less at the
corners (445/409 mm, 8545). Rubble's gauges are the side lines and the centreline: load_path,
crush_law 560/560/560 mm left/centre/right; load_path 540; default 200. Rubble's front loses
the same at the corners as at the centre, where the real car's corners lose 80–120 mm less.
*Measured.* Likely cause: the real bumper beam is curved and the corners (headlamps, fender
ends) sit behind the centre; how the generated nose is shaped in plan was not checked here
(*unknown*).

**Pit view (from below).** Rubble draws its collision skin, so from below it shows the floor and
the front's outline, not the engine, subframe or rails the film shows (`sheets/pit.jpg`). The
outline shortens evenly; nothing else can be compared there without drawing the beams.

### 3.2 What the mass alone changes

The test cars weigh 1664 / 1662 kg; Rubble's 1448 kg (−13 %). At the same 15.64 m/s the real
car brings 203.5 kJ against Rubble's 177.1 kJ (×1.149) and 26.0 against 22.6 kN·s of momentum
(×1.149). With the same front (same force against crush):

- The real cars' mean force over their crush is 206.4 kJ / 0.663 m = **311 kN** (8545) and
  203.3 / 0.685 = **297 kN** (7520); the load_path cars' 177.1 / 0.70 = **253 kN**; the default's
  450 kN; the ETK's 407 kN.
- Carrying the real mass, the load_path cars would need about 0.70 × 1.149 ≈ **0.80 m** at their
  mean force (an upper bound: more crush reaches stiffer parts). Their 0.70 m at 1448 kg is
  therefore not a match: their front is about **19 % too soft** in mean force (253 against
  297–311 kN). Conversely the real car at 1448 kg would crush about 0.663 / 1.149 ≈ 0.58 m.
- The default would reach about 0.45 m at 1664 kg: still far short of 0.66 m. Mass does not
  explain its stiffness.
- Deceleration at the same force scales 1/m: the real car's 40 g would be about 46 g at
  1448 kg, so the load_path cars' 51–55 g is closer than it looks, but late.
- Time to stop at the same mean force scales with m: the real 71 ms becomes about 62 ms at
  1448 kg; the load_path cars' 73–77 ms is again consistent with a softer front.
- Restitution, springback and pitch are ratios or shapes; the mass does not explain them.

## 4. What would have to change to reproduce the real result (ranked)

1. **Hold energy plastically: rebound 2.6–5.1 m/s against 0.9–1.5, springback 20–53 % against
   15–16 %.** Find where the elastic energy is stored at the stop (cabin first, then the upper
   front: `front-load-path.md` §4) and why those beams do not yield or fold. *Authoring* (members
   that should yield are built as springs; SUPPORT stops with infinite yield) and possibly
   *solver* (whether a folded section keeps its elastic energy: `beam-yield-law.md` §5, §9);
   *unknown* split.
2. **The powertrain where the Camry has it, so the first peak comes at 0.42 m and 30 ms.** The
   engine is 0.08 m further back and the firewall 0.24 m further back than the Camry's
   proportion; the drawn engine fixes it (`front-load-path.md` rule 11). *Authoring.*
3. **A tyre that can be pinched.** The wheel blocks against the crushed stack at 45 ms and loads
   the hinge pillar and sill (second peak, wheelbase −66 mm). Rubble's raycast tyres have no
   collision shape. *Solver/model* (a collision shape for the tyre, or node–beam tyres in crash
   scenarios) and *authoring* (the hinge pillar and sill's front as the path it loads).
4. **Front stiffness after the crash zones (mean force 253 → ~300 kN) and the real mass
   (1448 → 1664 kg).** With the mass alone the load_path cars would over-crush by ~0.14 m.
   *Authoring.* Which members are short of the §1.3 estimates is in `front-load-path.md` §4.1
   (the rail cells turn instead of crushing).
5. **The force carried low: the rear must lift (3–4° nose down), not the nose.** The load_path
   cars put 54–62 % of the front's work into the upper front. *Authoring* (`front-load-path.md`
   rules 4–5: the hood and fenders' share), checked by the wall's centre of pressure.
6. **Hood fold where the real one folds (a ridge a third back, by 30 ms, 0.3–0.4 m high).**
   Needs node rows at the fold lines and a hood that buckles at tens of kN, not hundreds.
   *Authoring* (gen2 rulebook: nodes on the panel's crests and character lines, not more nodes).
7. **Crush profile across the width (corners 80–120 mm less than the centre).** *Unknown*:
   the generated nose's plan shape was not compared.

The default car fails on 1, 2 and 4 together (one 661 kN peak at 20 ms, stops at 44 ms, 0.39 m);
the ETK behaves like the default (a 700 kN block, 45 ms, 0.44 m, restitution 0.29) but is a
different and smaller car, so it says how BeamNG-style content behaves in Rubble, not how a
Camry should.

## 5. Not determined, and why

- **What causes each real force peak** is read from geometry (`front-load-path.md` §1.4) and the
  films' timing; the films cannot see the engine or rails at those moments (the hood and fenders
  hide them; the pit view has no identifiable wall plane). The wheel's part in the second peak is
  inferred from the hub stopping at 45 ms.
- **Hood ridge height** is read by eye (±0.1 m): the hood is 0.2–0.9 m further from the camera
  than the side plane the scale was set on.
- **Pitch** is read by eye from the belt-line targets (±1°); 7520's pitch was not read.
- **Rubble's real-mass run** was not made (no `--set mass` exists for S11's car without changing
  the design); §3.2 is a calculation.
- **C1–C6 in Rubble**: no gauge at the six stations; only the sides and centreline.
- **Pitch sign of `crash_ref`'s `pitch_peak_deg`** is the largest-magnitude sample at any time;
  the per-moment values above are from the outlines (`rear_pitch_deg`).
- 8545 photo P079 ("As Delivered FOR WEB") could not be fetched whole (the server sends 37 391 of
  the listed 43 200 bytes); it is a pre-test exterior photo.
- **Unverified**: the 7520 films were annotated from the left camera only (sequence by eye, no
  pixel measurements); the ETK's outline-based numbers are invalid (no part groups); the ETK car's
  identity as a model (its size against a Camry) was taken from its collision length.

## 6. Regenerate

From `engine/` (ROOT = `/Users/glavin/Development/rubble/inspiration/ccsa-camry-2012`, local):

```sh
cargo build --release -p engine-softbody-lab --example crash_ref --example crash_outlines
python3 tools/crash-reference/nhtsa_footage.py $ROOT 8545 7520 --sheets      # probes, sheets, index
M=0,10,20,30,40,55,70,85,100,121,150,290
for v in v08545C009 v08545C015 v08545C006 v08545C013; do
  python3 tools/crash-reference/camry_frontal.py stills $ROOT 8545 $v.avi --ms=$M --stamp; done
python3 tools/crash-reference/camry_frontal.py stills $ROOT 7520 v07520C008.avi --ms=$M --stamp
python3 tools/crash-reference/camry_frontal.py strip $ROOT 8545 v08545C009.avi --ms=0,10,20,30,40,55,70,85,100,121,170,290 --cols 6 --width 512
python3 tools/crash-reference/camry_frontal.py scenario --ms $M > out/camry-frontal/rubble/S11-moments.json
python3 tools/crash-reference/camry_frontal.py scenario --ms $M --car local:etkk_tex/etkk/kc6.pc > out/camry-frontal/rubble/S11-etk.json
python3 tools/crash-reference/camry_frontal.py cameras > out/camry-frontal/rubble/cameras.json
export RUBBLE_LOCAL_MODS=<main worktree>/engine/crates/engine-webgpu/__mods
O=target/release/examples/crash_outlines; CF=out/camry-frontal/rubble/cameras.json
$O --scenario out/camry-frontal/rubble/S11-moments.json --cameras $CF --out out/camry-frontal/rubble/default
RUBBLE_FRONT_RULES=load_path $O --scenario out/camry-frontal/rubble/S11-moments.json --cameras $CF --out out/camry-frontal/rubble/lp
RUBBLE_FRONT_RULES=load_path,crush_law $O --scenario out/camry-frontal/rubble/S11-moments.json --cameras $CF --out out/camry-frontal/rubble/lp_cl
$O --scenario out/camry-frontal/rubble/S11-etk.json --cameras $CF --out out/camry-frontal/rubble/etk
B=out/camry-frontal/rubble
python3 tools/crash-reference/camry_frontal.py sheet --rows $M --cell 560 380 \
  --col "Real 2014 Camry (8545) Left Front Half=out/camry-frontal/real/8545/v08545C009/{ms}.jpg" \
  --col "Real 2012 Camry (7520) Left Front Half=out/camry-frontal/real/7520/v07520C008/{ms}.jpg" \
  --col "Rubble gen sedan (default)=$B/default/{ms}ms-left.png" \
  --col "Rubble sedan load_path,crush_law=$B/lp_cl/{ms}ms-left.png" \
  --col "Rubble ETK (BeamNG import)=$B/etk/{ms}ms-left.png" --out out/camry-frontal/sheets/side-left.jpg
#   (side-right: v08545C015 with -right.png; pit: v08545C006 with -pit.png; overall-right: v08545C013 with -overall_r.png)
python3 tools/crash-reference/camry_frontal.py shape default=$B/default lp=$B/lp lp_cl=$B/lp_cl   # per-moment shape table
for v in default lp lp_cl; do ...; done   # crash_ref S11 --run 0 --contact edge --json > out/camry-frontal/runs/<v>.json
#   (RUBBLE_FRONT_RULES as above; ETK: --set car=local:etkk_tex/etkk/kc6.pc)
python tools/crash-reference/pulse_plot.py --out out/camry-frontal/sheets/pulse.png \
  --ref docs/crash-reference/nhtsa/8545.json --ref docs/crash-reference/nhtsa/7520.json \
  --run "Gen sedan default=out/camry-frontal/runs/default.json" ...   # needs matplotlib
python3 tools/crash-reference/camry_frontal.py studio $ROOT   # Reference Studio shots S11_nhtsa8545/7520 and frames
```

In zsh, pass `--cameras` and its file as two words (a variable holding both is one word).
The Reference Studio shots: `apps/reference-studio/workspace/shots/S11_nhtsa8545.json`,
`S11_nhtsa7520.json` and their 26 frames (images in the gitignored `workspace/library/`).

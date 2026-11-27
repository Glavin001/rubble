# FE-import spike, stage 1: the CCSA 2012 Toyota Camry model

This spike asks whether a public, validated LS-DYNA crash model of a real car can be the input for a deterministic converter to Rubble's node-beam vehicles. Stage 1 covered:
- the licence;
- the download;
- parsing the deck;
- a mass report, part statistics and load-path sections;
- the measured test data the model was validated against;
- pictures of both.

No converter was built. All numbers below are our own, computed from the deck and the NHTSA files with the scripts in `engine/tools/fe-import/`.

## Where things are (all local, nothing committed from downloads)

**Downloads.** Everything downloaded sits under `inspiration/ccsa-camry-2012/` (gitignored):
- the zip, the extracted deck, the two CCSA PDFs and the model page;
- `SOURCE.md` with URLs, sizes and SHA-256;
- one `nhtsa-<test>/` folder per crash test, each with its own `SOURCE.md` and a `channels.json` describing every curve.

**Outputs.** Derived outputs sit under `engine/out/fe-import/` (gitignored):
- `report.json`, `parts.json` (every part with its mass, section, material and centroid) and `members.json`;
- `view/`: GLBs, `.blend` files and stills;
- `plots/`: per-test SVG/PNG and `nhtsa_metrics.json`.

**Scripts.** They use the Python standard library only, except the Blender one, and run with `python3 -I`:

| Script | Does | Time |
|---|---|---|
| `lsdyna_deck.py` | parse to a pickle cache | 10 s |
| `fe_mass.py` | element and lumped masses, mass-scaling estimate | 11 s |
| `fe_report.py` | counts, mass properties, classes, connections | 6 s |
| `fe_sections.py` | plane cuts of members, crush-force formula | 13 s |
| `fe_mesh.py` | per-part surface triangles | 3 s |
| `blender_fe_view.py` | build, decimate, export, render | about 5 s per step |
| `nhtsa_plots.py` | filter, integrate and plot the NHTSA curves | — |

Re-run in order:

```
python3 -I engine/tools/fe-import/lsdyna_deck.py inspiration/ccsa-camry-2012/model/2012-toyota-camry-detailed-v5a/combine.key engine/out/fe-import/cache
python3 -I engine/tools/fe-import/fe_mass.py engine/out/fe-import/cache
python3 -I engine/tools/fe-import/fe_report.py engine/out/fe-import/cache engine/out/fe-import
python3 -I engine/tools/fe-import/fe_sections.py engine/out/fe-import/cache engine/out/fe-import
python3 -I engine/tools/fe-import/fe_mesh.py engine/out/fe-import/cache engine/out/fe-import/part_sections.json engine/out/fe-import/mesh
Blender -b --factory-startup -P engine/tools/fe-import/blender_fe_view.py -- build engine/out/fe-import/mesh engine/out/fe-import/view   # then: decimate <dir> 400000; render <dir> camry_decimated.blend
python3 -I engine/tools/fe-import/nhtsa_plots.py inspiration/ccsa-camry-2012 engine/out/fe-import/plots
```

Nothing here is a test or build input. AGENTS.md forbids depending on `inspiration/`.

## 1. Licence and terms

The model page, the PDFs and the DOI metadata (DataCite, empty rights list) state no licence. The only terms are in the zip README and the deck header:
- users must verify their own results;
- CCSA, FHWA, GMU and NHTSA take no responsibility for results;
- CCSA at GMU and FHWA should be acknowledged in publications.

The page offers a plain download with no registration.

| Use | Status |
|---|---|
| Downloading and using the model for research and validation | Clearly the intent: the page offers it "for use in crash simulations", the CCSA calls it publicly available, and it is downloaded with no conditions |
| Distributing a derivative in an open repo | Not addressed, so not granted. The owner's decision is that files stay local and are never shared. Our code is ours |
| Publishing comparison results | Not restricted; acknowledgement requested |

The NHTSA data are US government works.

## 2. What the zip holds

The zip holds the input deck only. There are no results: no d3plot, binout, ASCII time histories or curves.

| File | Content |
|---|---|
| `camry-detailed-v5a.key` | 248 MB, the vehicle |
| `set-camry-detailed-v5a.key` | Added masses, NCAP dummies and cargo, initial velocity, part and node sets, accelerometers |
| `wall.key` | The NCAP rigid wall |
| `combine.key` | The NCAP 56 km/h wall run (control cards, contacts, output requests), which includes the other three |

**Units** are t, mm, s, N. The README and deck header state them, and the numbers are consistent:
- steel density 7.85e-9 t/mm³;
- E = 210,000 MPa;
- gravity 9806 mm/s²;
- initial velocity 15,646 mm/s (35 mph).

**Axes.** x points forward (the bumper is at x ≈ −66 mm, the car lies at negative x), y to the left, z up, with the ground at z ≈ 0. This was checked against the wheel and part positions.
- Front wheel centre: x = −945. Rear wheel centre: x = −3736.
- Wheelbase: 2790 mm, equal to the tech summary's value.

**Format.** The deck is standard fixed width: I8 fields for nodes and elements, I10 for other cards. It has no long (`+`) or free format. The parser supports both anyway, but that path is untested on a real deck. It does use `*PARAMETER`/`&NAME` substitution (dummy masses, velocity) and `_TITLE`/`_ID` variants.

**Counts against the tech summary** (vehicle only):

| Item | Parsed | Published |
|---|---|---|
| Shells | 2,032,594 | 2,032,594 (exact) |
| Solids | 218,794 | 218,785 (+9 solids in the set file) |
| Beams | 5,886 | 5,901 (−15, not reconciled) |
| Parts | 1,088 (1,086 in the main deck) | 1,086 |
| Nodes | 2,255,413 | 2,255,361 (+52 instrument nodes in the set file) |
| Elements | 2,257,274 | 2,257,280 |

The beam difference is likely the counting convention, for example the 9 seatbelt accelerometers.

**Element size.** The mean shell area is 49.5 mm², a 7.0 mm equivalent edge. The published size is 6–8 mm.

**Material models**, by number of parts:

| Model | Parts | Mass (kg) |
|---|---|---|
| `MAT_PIECEWISE_LINEAR_PLASTICITY` | 950 | 754 |
| `MAT_ELASTIC` (engine, transmission, battery, tyres…) | 24 | 263 |
| `MAT_RIGID` | 36 | 40 |
| `MAT_MODIFIED_PIECEWISE_LINEAR_PLASTICITY` (glass) | 10 | 37 |
| `MAT_LOW_DENSITY_FOAM` | 23 | 25 |
| `MAT_BLATZ-KO_RUBBER` | 19 | 7 |
| `MAT_NULL` | 11 | 1.4 |
| `MAT_SPOTWELD` | 1 | — |
| Springs and dampers | 14 | — |

**Steel grades.** There are 13 stress–strain curves. The grades by initial yield (MPa) are 180, 220, 250, 300, 360, 400, 450, 500, 675, 800 and 1300:
- 250 MPa: 270 parts, 228 kg;
- 300 MPa: 312 parts, 171 kg;
- 360 MPa: 167 parts, 154 kg;
- 675–1300 MPa: 19 parts, 24 kg, among them the B-pillar inner and the front bumper beam (1300).

Every steel uses Cowper–Symonds rate scaling with C = 8000 s⁻¹ and p = 8.

## 3. Mass report

Mass is computed the way an explicit code lumps it:
- shells: area × thickness × density, with the area as half the cross product of the diagonals;
- solids: trilinear-hex volume × density;
- beams: ρAL;
- `*ELEMENT_MASS` adds point masses;
- `*ELEMENT_MASS_PART` spreads added mass over the part's nodes in proportion to their element mass.

Section non-structural mass is zero everywhere.

| Layer | kg |
|---|---|
| Elements | 1128.6 |
| Added masses for missing components (`*ELEMENT_MASS_PART`, 45 parts: shocks, brake discs, doors, roof, floors, pillars…) | 291.7 |
| Point masses in the main deck (127 × 0.1–0.5 kg) | 15.9 |
| Fuel (tank parts, 30 + 12 kg) | 42.0 |
| NCAP dummies (78.5 + 62.2 kg) and cargo (69.8 kg) | 210.5 |
| Mass scaling (estimate, see below) | about 7.8 |

**Mass scaling.** `DT2MS = −0.778 µs`, applied to elements whose 0.9·Lc/c falls below 0.7 µs. We count about 52,600 such elements. The 7.8 kg is our estimate; the code itself was not run.

| Configuration | Mass (kg) | CG aft of front axle (mm) | CG height (mm) | CG y (mm) | Roll / pitch / yaw (kg·m²) |
|---|---|---|---|---|---|
| Elements + added + point (no fuel) | 1436.2 | 1075 | 568 | −1.6 | 569 / 2526 / 2810 |
| + fuel | 1478.2 | 1110 | 559 | −0.7 | 577 / 2596 / 2879 |
| + dummies and cargo (NCAP) | 1688.7 | 1214 | 541 | +3.1 | 617 / 2928 / 3230 |
| Published FE model (tech summary, Table 1) | 1462 | 1086 | 560 | −1 | 572 / 2524 / 2807 |
| Published physical car (SEAS) | 1452 | 1063 | 561 | −9 | 560 / 2519 / 2796 |
| Published FE in NCAP trim (Table 2) and test 8545 | 1674 / 1664 | 1196 / 1194 | — | — | — |

The products of inertia (no fuel) are Ixz = 77 and Ixy = −4.6 kg·m².

**What reconciles:**
- The inertias and CG of the no-fuel configuration match the published FE values within 0.1–0.5 %.
- The CG is 11 mm forward and 8 mm higher than published.

**What does not:**
- **Mass.** We get 1436 kg against the published 1462 kg (−1.8 %). Our 7.8 kg mass-scaling estimate closes only part of the gap. With fuel we get 1478 kg (+1.1 %), but then the CG moves 35 mm aft and the pitch inertia rises 70 kg·m², and the published numbers fit the no-fuel case.
- The likely explanation is the solver's own reported mass (added mass from scaling, and rigid-body or joint bookkeeping) or a slightly different deck version from the one the tables were made with. Neither was checked: we did not run the solver.
- **NCAP trim.** We get 1688.7 kg against 1674 published (+0.9 %); the published figure equals 1462 + 210.5 with no fuel. That suggests the tech summary's FE masses exclude the fuel the set file adds.
- **Odd parts.** Two "rear seat side airbag" solid blocks of 6.4 kg each carry steel density, probably a modelling stand-in.

**Part statistics** (element mass only, 1,074 parts with elements):

| Share of element mass | Heaviest parts needed |
|---|---|
| 50 % | 60 |
| 80 % | 234 |
| 90 % | 400 |
| 95 % | 560 |
| 99 % | 838 |

The cumulative share after the heaviest 1/10/50/100/200/500 parts is 11 / 26 / 47 / 61 / 76 / 93.5 %.

**Heaviest parts.** The engine is 126.7 kg, the transmission 60.5 kg and the battery 20.8 kg, all elastic solids. Then come:
- body-side outers, 16.5 kg each, 0.9 mm, 180 MPa;
- roof, 13.1 kg;
- front floor, 12.1 kg;
- hood outer, 10.4 kg;
- firewall, 10.3 kg;
- intake manifold, 7.9 kg;
- windshield, two layers of 7.6 kg;
- brake rotors, 7.4 kg each (rigid);
- exhaust, trunk floor and lid, tyres, rims, door outers.

The full ranked list is in `report.json` (`parts_top40`) and `parts.json`.

**Crash-section grouping.** The rule is transparent: ordered regular expressions on the part name, with a position test (x < −2900 mm counts as rear) for names that occur front and rear (subframe, floor, under-floor rails, wheelhouse). The rules are listed in `fe_report.py`.

| Section | Parts | Element mass (kg) |
|---|---|---|
| front_lower (bumper beam and cover, crash boxes, front rails "midrail*", radiator support, front subframe) | 96 | 69.6 |
| front_upper (hood, fenders, cowl, upper rails/aprons, strut towers) | 45 | 38.1 |
| cabin (pillars, sills "rocker*", roof, floors, under-floor rails, firewall, body sides, IP beam) | 144 | 198.2 |
| door (doors incl. inner structure, hinges, latches; door glass excluded) | 174 | 100.6 |
| rear (rear bumper, trunk, shelf, quarter inners, rear rails, rear subframe, rear floor) | 96 | 106.9 |
| wheel_suspension (tyres, rims, hubs, rotors, knuckles, arms, struts, springs, anti-roll bars, steering rack) | 138 | 169.1 |
| other (powertrain, cooling, exhaust, fuel tank, seats, dash trim, airbags, glass, lamps, pedals, spot-weld beams) | 381 | 446.0 |
| unclassified | 0 | 0 |

Notes on the grouping:
- The plastic dash trim is put in "other" (interior), not "cabin". The structural dash members (firewall, cowl, IP beam) are in cabin.
- The CCSA names the front longitudinal "midrail", the upper load path "upperrail" and the under-floor longitudinals "shotgun".

**Connections:**

| Kind | Count |
|---|---|
| `*CONSTRAINED_SPOTWELD` (node pairs) | 7,209 |
| Spot-weld beams (5 mm, `MAT_SPOTWELD`, elform 9), tied by one `*CONTACT_TIED_SHELL_EDGE_TO_SURFACE` | 5,102 |
| `*CONSTRAINED_NODAL_RIGID_BODY` (3–98 nodes, median 8, 13,235 nodes in total; bolts and clinches) | 1,166 |
| `*CONSTRAINED_EXTRA_NODES_SET` | 30 |
| `*CONSTRAINED_RIGID_BODIES` merges | 2 |
| Joints: revolute | 21 |
| Joints: spherical | 6 |
| Joints: cylindrical | 4 |
| Joints: translational | 1 |
| Discrete springs and dampers (suspension, exhaust hangers) | 31 |
| `*AIRBAG_SIMPLE_AIRBAG_MODEL` (tyre air volumes) | 4 |

Contacts are one automatic single surface and one interior contact, in `combine.key`.

## 4. Load-path members (left side, cut by planes through the shell mesh)

Each cut intersects the member's shells with the plane. The wall length in the section comes per part; the outer bounding box comes from all of them.

The flow stress σ0 is the mean of the part's stress–plastic-strain curve over plastic strain 0–0.2. The first segment is extrapolated back to zero strain, where it meets the stated yield.

**Front rail ("midrail").** σy = 360 MPa for every part, σ0 = 499 MPa.

| Cut | Parts (t mm) | Box (mm) | t_eq (mm) | Steel area (mm²) |
|---|---|---|---|---|
| x = −300 (crash-box zone) | midrailupperL 1.75, midrailL 1.6 | 92 × 166 | 1.69 | 781 |
| x = −600 | midrailml 2.08, midrailouterL 2.04, cradle supports 2.03/2.10 | 92 × 142 | 2.06 | 1076 |
| x = −900 | midrailrearL 2.65, midrailouterL 2.04, brackets 1.29/2.01 | 95 × 180 | 2.13 | 1228 |
| x = −1100 | midrailrearL 2.65, midrailouterL 2.04 | 93 × 184 | 2.41 | 1082 |

**Mean axial crush force of one rail.** We use the Abramowicz–Jones square-tube form (1984) of Wierzbicki–Abramowicz's superfolding element (1983): Pm = 13.06 σ0 t² (C/t)^(1/3), with C = (b + h)/2 for a rectangle.

| Cut | Static, σ = σy (kN) | Static, σ = σ0 (kN) | Dynamic (kN) |
|---|---|---|---|
| x = −300 | 57 | 79 | 120 |
| x = −600 | 77 | 107 | 162 |
| x = −900 | 86 | 119 | 180 |
| x = −1100 | 105 | 146 | 220 |

How the dynamic column is formed:
- It applies the deck's own Cowper–Symonds factor 1 + (ε̇/8000)^(1/8) = 1.51–1.52, at ε̇ = 0.33 V/C ≈ 40 s⁻¹ and V = 15.6 m/s.
- The classic mild-steel constants (D = 40.4, q = 5) would give a factor of about 2.0 instead.

**Observation, not proof.** Two crash-box sections at 120 kN each (240 kN) are close to the first plateau of the measured wall force in test 8545, about 250 kN at 10–20 ms. That is the size of yield a node-beam rail must carry.

**B-pillar.** The section takes in the body-side outer (0.9 mm, 180 MPa) and four inner parts.

| Cut | Parts (t mm / σy MPa) | Box, x × y (mm) | t_eq (mm) | σ0 eq (MPa) | Area (mm²) |
|---|---|---|---|---|---|
| z = 600 | sidepanel 0.9/180, B-pillar lower 1.5/500, bpillarinner 1.6/500 | 201 × 97 | 1.30 | 617 | 1134 |
| z = 900 | sidepanel 0.9/180, bpillarinnermain 1.5/675, bpillarinner 1.6/500, supports 1.85/500 and 1.6/360 | 168 × 94 | 1.43 | 702 | 1424 |
| z = 1200 | same parts | 127 × 70 | 1.42 | 706 | 968 |

**Sill ("rocker").**

| Cut | Parts (t mm / σy MPa) | Box, y × z (mm) | t_eq (mm) | σ0 eq (MPa) | Area (mm²) |
|---|---|---|---|---|---|
| x = −1800 | rockerpanel 1.1/400, rocker inner 1.45/400, sidepanel 0.9/180, horizontal supports 1.17/400 and 2.51/400 | 141 × 155 | 1.24 | 584 | 949 |
| x = −2200 | same | 140 × 157 | 1.26 | 586 | 990 |
| x = −2700 | rockermid 1.65/500, rocker inner 1.45/400, sidepanel 0.9/180 | 140 × 167 | 1.38 | 618 | 970 |

**Welds between member parts.** By `*CONSTRAINED_SPOTWELD` node pairs: rail 79, B-pillar 4, sill 4. Most body welds are the 5,102 beam welds, which attach through tied contact by proximity, so per-member counts of those need a proximity search (not done).

## 5. What the validation covers, and the measured data now held

Validations named by the CCSA deck and tech summary, with the NHTSA tests that match them:

| Validation (CCSA) | Plotted test vs simulation (CCSA) | NHTSA test(s) held as numbers |
|---|---|---|
| NCAP frontal rigid wall, 56 km/h | Rear-seat crossmember accelerations L/R and velocities; engine top/bottom accelerations; barrier force vs displacement (about 900 kN, 650 mm); energy balance; deformation pictures; CORA scores | **8545** (2014 Camry, the cited test), **7520** (2012 Camry) |
| Frontal full wall, 40 km/h (test car a Hybrid) | Vehicle pulse | **7793** (2011 Camry Hybrid, previous body: best available match); 9692 (2015, non-hybrid) |
| Frontal left oblique, 90 km/h | Vehicle kinematics and pulse; barrier pulse | **9124** (2012); 8790 (2014) |
| Frontal right oblique, 90 km/h | Same | **9121** (2012) |
| Side NCAP moving barrier, 62 km/h | Vehicle pulse; post-test intrusion at 5 heights | **7517** (2012); 8560 (2014.5) |
| Side NCAP pole, 32 km/h | Post-test intrusion at 5 heights | **7523** (2012); 8558 (2014.5) |
| IIHS roof strength (quasi-static) | Strength-to-weight | not in the NHTSA DB; not fetched |

Each test folder holds the following (the per-curve TSV URLs served wrong or truncated files for 8 of 11 tests, so the zips were used instead):
- the full ISO-MME zip and the EV5 zip, all channels as numbers;
- the API summary JSON (masses, speeds, the C1–C6 crush profile, intrusion);
- the report PDF, except for 7793 and 9692, which have none.

Films are listed with their sizes but not fetched (88–454 MB per test). About 394 MB were downloaded in total.

### Pulse metrics from the numbers

Rear-seat crossmember X (mean of L/R), bias removed, CFC 60 (SAE J211 two-pass Butterworth), trapezoidal integration from the test speed:

| Test | Speed (km/h) | Rate | Peak (g) | t_m (ms) | Dynamic crush (m) | Mean decel to t_m (g) | Rebound at 200 ms (m/s) | Peak wall force (kN, load cells) |
|---|---|---|---|---|---|---|---|---|
| 8545 | 56.7 | 20 kHz | 39.8 | 71 | 0.663 | 22.6 | 1.5 | 921 (36 cells) |
| 7520 | 56.3 | 10 kHz | 45.0 | 76 | 0.685 | 21.0 | 0.9 | 830 (128 cells) |
| 7793 | 39.6 | 10 kHz | 31.5 | 75 | 0.489 | 15.0 | 1.2 | — |
| 9692 | 39.6 | 10 kHz | 34.1 | 69 | 0.475 | 16.2 | 1.6 | — |

8545 matches the tech summary's description of its own test: about 900 kN and 650 mm.

**Impulse check.** The wall impulse ∫F dt is 31.8 kN·s for 8545 and 30.4 kN·s for 7520. That is 11–17 % more than the test mass × the integrated Δv. This is not reconciled. It is probably the wall-face mass in front of the load cells, or calibration; the sign and units are right.

**Oblique and side tests.** The car's CG is integrated from rest, except in the pole test.

| Test | Peak X (g) | Peak Y (g) | Δv at 150 ms | Other |
|---|---|---|---|---|
| 9124 | −45.9 | 41.0 | −14.8 / +4.4 m/s | barrier CG −25.8 g |
| 8790 | −48.2 | 52.5 | — | — |
| 9121 | −65.1 | −37.4 | — | — |
| 7517 side MDB | — | 31.8 (CG Y) | 6.2 m/s | — |
| 8560 side MDB | — | 35.6 (CG Y) | 6.35 m/s | — |
| 7523 pole | — | 16.5 | — | pole force peak 191 kN, 8 cells |
| 8558 pole | — | 17.9 | — | pole force peak 197 kN |

### Does this give enough ground truth that we never run the FE model?

**(a) Full-vehicle crashes: yes, for what the tests measured.** We now hold, as numbers:
- the frontal wall pulse (two 56 km/h tests and two 40 km/h ones), from which come velocity, displacement, dynamic crush, t_m and rebound;
- the total wall force, and with it force against displacement, in the two 56 km/h tests;
- engine accelerations;
- oblique-test car and barrier pulses (CG, sills, seat tracks, floor-pan string pot);
- side MDB and pole pulses, with A-/B-pillar, sill, roof and firewall accelerations and the pole force;
- static crush profiles and intrusion tables in the API JSON and reports.

The validation deck's CCSA simulation curves exist only as pictures in the PDF. We do not need them: the tests are the ground truth.

What tests cannot give: internal quantities such as section forces in a rail, member-by-member energy, and the time history of intrusion. Only a simulation (or our own model) gives those.

**(b) Component level: nothing is provided.** No single rail, door or pillar was crushed alone, either physically or as a published simulation. The CCSA gives component information only as the deck itself (geometry, thickness, material curves). Two routes remain:
- a section formula (above: Abramowicz–Jones for the rail; bending or plastic-hinge formulas for pillar and sill, not yet done);
- running a solver on an extracted sub-model.

**Solver.** The README of OpenCourant, the AGPL-3.0 community continuation of OpenRadioss, states that the Starter reads `.k`/`.key` LS-DYNA files natively (checked 2026-11-27 via the GitHub API, without installing anything). Its repository holds keyword-reader definitions for `MAT_024`, `CONSTRAINED_SPOTWELD`, `CONTACT_SPOTWELD` and others.
- Whether every card this deck uses is accepted, and gives the same response, was not verified: for example `CONTACT_TIED_SHELL_EDGE_TO_SURFACE`, `AIRBAG_SIMPLE_AIRBAG_MODEL`, `MAT_MODIFIED_PIECEWISE_LINEAR_PLASTICITY`, Blatz-Ko rubber, and the `*PARAMETER` expressions.
- The original `OpenRadioss/OpenRadioss` repository now returns 404 on the GitHub API.

## 6. Pictures

**Mesh.** Each part is one named mesh, `"<pid> <name>"`, coloured by crash section (7 materials):
- `engine/out/fe-import/view/camry_full.glb`: 4.24 M triangles, 52 MB;
- `camry_decimated.glb`: 448 k triangles, 6.2 MB. Blender collapse decimation, ratio 0.094, at least 40 triangles per part, target 400 k.

**Stills** (Blender Workbench, orthographic), in `engine/out/fe-import/view/`:
- `camry_side_left.png`
- `camry_top.png`
- `camry_front_three_quarter.png`
- `camry_structure_three_quarter.png`
- `camry_structure_side.png`

The two structure views hide the doors, glass, hood, fenders, trunk lid, bumper covers and foams, the wheels and suspension, and everything in "other".

**Test plots.** `engine/out/fe-import/plots/nhtsa_<test>.png`, one per test, four panels: acceleration, velocity, barrier force, and force against displacement.

**No Rubble overlay.** The Rubble suite row S02 reports scalars only, with no time history, so it is not overlaid. S02 is a different car and speed: the generated sedan, 1448 kg, at 50 km/h. Its figures are peak 43.4 g, mean 21.5 g over 88 ms, peak crush 0.373 m, residual crush 0.171 m and rebound ratio 0.35 (`engine/out/fe-import/rubble_S02_summary.json`).

## 7. Not done or not verified

- The FE model was not run: there is no LS-DYNA, and no open solver was installed.
- The 26 kg mass gap and the 15-beam count gap are not reconciled. The 7.8 kg mass-scaling figure is our estimate, not the solver's.
- Identifying the 40 km/h hybrid test (7793, previous body) and choosing among the oblique, side and pole test pairs is our judgement. The validation deck gives only the 8545 number.
- The CORA scores and the CCSA simulation curves are pictures in the PDF and were not digitised.
- The B-pillar and sill have no strength formula yet, only sections.
- Spot-weld beams were not attributed per member.
- The per-curve TSV server (`nrd-static.nhtsa.dot.gov/tsv/...`) returned wrong or truncated files for 8 of 11 tests on 2026-11-27. The 2026-11-04 pulses in `crash-reference-targets.md` §8 came from that server and may deserve a re-check against the zips (not done here).

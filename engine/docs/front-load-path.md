# The front load path: the 2012 Camry's, the generated sedan's, and a design for it

The sedan's front, after seven switches of its lattice (`sedan-front-audit.md` §8–9, lessons
§25–26), still did not fold: each round found a geometric defect instead. The owner's decision:
read how a real car of this class connects its front, compare with what the generator builds,
and design the load path as generator rules behind one switch. This page is that work
(2026-11-27, branch `crash-ref/front-load-path`). Facts, measurements and estimates are marked;
every number is our own.

Reproduce (from `engine/`; outputs in the gitignored `out/`, copies in
`~/Development/rubble-artifacts/sections/load-path/`):

```
python3 -I tools/fe-import/lsdyna_deck.py <deck>/combine.key out/fe-import/cache     # local deck only
python3 -I tools/fe-import/fe_load_path.py out/fe-import/cache out/fe-import          # parts, joins, cuts
cargo run --release -p engine-vehiclegen --example front_load_path                     # the calculator table
tools/sections/front_ab.sh off; tools/sections/load_path_ab.sh                          # the ladder
python3 -I tools/sections/front_ab.py off lp_c lp_cp lp_cpm lp_cpmu lp_cpmup lp        # tables (--plot DIR)
python3 -I tools/sections/load_path_spans.py off lp ...                                # bay by bay in the push
tools/sections/load_path_hard.sh lp load_path; target/release/examples/crash_verdict out/gen/hard_lp
```

## 1. The Camry's front, read off the deck (facts)

Source: the CCSA 2012 Camry deck (local, `fe-import-spike.md`), read with
`tools/fe-import/fe_load_path.py` (part boxes, plane cuts, joins: `*CONSTRAINED_SPOTWELD` pairs,
`*CONSTRAINED_NODAL_RIGID_BODY` sets spanning two parts (bolts, clinches), merged meshes, and the
5 mm weld beams each end given to the part with a shell node within 4 mm: an approximation of the
tied contact). Deck axes: x forward with the car at negative x, the bumper at x ≈ −66 mm, the
front axle at x = −945; left side. Stresses: σ_y and the flow stress σ₀ (mean of the part's curve
over plastic strain 0–0.2); every steel has Cowper–Symonds C = 8,000 s⁻¹, p = 8.

### 1.1 The members, front to rear (left side)

```
 x (mm)  0   -66  -198 -266        -445/-479             -760                -1210  -1300  -1400
         |cover|beam|brkt|crash zone |  mid rail (64+180)  |  rear rail (65+180)  |kick-down|sill
                     ^bolted (3)      ^gauge steps 1.7→2.06            2.65 ^welded to firewall
 upper:        radiator support top -291 ── tie bar 0.96 ── upper rail 1.02 ── -1363 A-pillar
 lower:        radiator lower x-member -196..-344 ── bracket (bolted) ── subframe front bushing -483..-570
               subframe side members -450 ... -1370 ── rear bushing -1308 (under toe pan, on the kick-down)
 powertrain:   engine -491..-914, transmission -553..-1051 (transverse); radiator -285..-317, fan to -408
```

| member (parts) | x range (mm) | section (mm), t (mm) | σ_y / σ₀ (MPa) | joined to (by) |
|---|---|---|---|---|
| bumper beam `24_bumperfront` | −66 (centre) … −198 (ends); 1,180 wide | B-section 61 deep × 116, two cells, 1.16 | 1300 / 1428 | the bracket by **3 rigid bodies a side (bolts)**; foam and cover |
| bumper bracket `31_bumpermidrail` | −176 … −266 | 69 × 116 box stub, 2.06 | 360 / 499 | the crash zone by 12 weld beams |
| crash zone `43_midrailupperL` + `17_midrailL` (+ plate `16`) | −206/−237 … −446/−479 | 92 × 163, t_eq 1.70 | 360 / 499 | the mid rail (mesh merged, 50 shared nodes); radiator support pieces (5 + 3 + 2 welds) |
| mid rail `64_midrailml` + `180_midrailouterL` | −440 … −763 (180 to −1222) | 83–92 × 140–155, 2.06 | 360 / 499 | rear rail (merged); subframe hanger `63` (5 welds); upper-rail bridges `25`, `28` |
| rear rail `65_midrailrearL` + `180` | −760 … −1210 | 57–95 × 168–193, 2.1–2.65 | 360 / 499 | strut housing `77` (17 welds); firewall (3); the under-floor rail `220` (13 + 9) |
| kick-down / under-floor rail `220_shotgununder`, `238` | −1024 … −2214, z from 500 down to 150 | 100–114 × 109–158, 1.8 | 360 / 499 | the rocker and A-pillar lower from −1350 |
| upper rail `23` tie bar, `28`/`30`, `07` | −291 … −1363 | tie bar 0.96 open; top ~75 × 75 hat on the apron, 1.02; rear 1.46 | 250–300 / 302–362 | radiator support top; strut housing; A-pillar inner, side panel |
| radiator support (`36`, `40`, `44`, `08`) | −196 … −376 | sheet 0.9–2.1 | 300 / 362 | welded to the crash zone's outer and lower faces |
| front subframe (`75`, `401`, `467`, `451`, `463`, `400`) | −448 … −1374 | side members ~70 × 48, 2.1–2.5 | 360 / 499 | front bushings (rubber solids in tubes) **bolted to hangers under the rail** at −483…−570, just behind the crash zone; a bracket `81` bolted forward to the radiator's lower corner; rear bushings bolted under the toe pan to the kick-down and the under-floor rail at −1308 |
| strut towers `77`, `84` | −804 … −1251 | sheet 1.08, top 2.9 | 250 / 302 | on the rail's outer side (17 welds) and the upper rail |
| firewall `54` (+ supports) | −1164 (top) … −1548 (toe pan), front face at z 300–600 −1148…−1191 | 1.11 | 250 / 302 | rail ends, under-floor rails, cowl, A-pillars |
| hinge pillar, sill | A-pillar lower from −1298, rocker from −1360/−1389 | sill 141 × 155, 1.24 | 400 / 584 | — |

**Free distances** (least over 50 mm bins where both parts lie, measured): bumper beam's rear
(−160 at the centre) to the radiator 128 mm; to the engine block 337 mm (to its front
accessories 273); radiator to block 129; engine block to firewall **249 mm**; transmission to
firewall 190 mm with the steering rack 35 mm behind the transmission and 102 mm ahead of the
firewall; transmission to the subframe's rear cross member 83 mm. Masses (elements): engine
126.6 kg, transmission 60.5 kg, battery 20.8 kg (all elastic solids). The front tyre's rear (−1282)
is 80–110 mm from the rocker's front end and the A-pillar's foot.

### 1.2 How the parts join

- The **bumper beam is bolted** (three rigid bodies a side) to a short 2.06 mm box bracket, which
  is welded to the rail's crash zone: a plate joint that carries the box's force, shear and moment.
- The **crash box is the rail's own front**, lighter (1.6–1.75 mm against 2.06 behind): one member
  with a gauge step, not a separate can.
- The **subframe hangs under the rail**: its front bushings are bolted to hangers welded under the
  rail just behind the crash zone; its rear bushings under the toe pan on the rail's kick-down.
  Nothing joins it to the crash zone itself (its forward bracket goes to the radiator's lower
  corner, a sheet path).
- The **upper rail** is its own thin member from the radiator support's top to the A-pillar,
  welded to the strut housing; it is joined to the main rail only by thin bridges.
- The **rail ends at the firewall**, welded to it and continued down and back under the floor
  (the kick-down), into the rocker and A-pillar foot: the survival cell's front.

### 1.3 Each member through the calculator (`examples/front_load_path.rs`)

Abramowicz–Jones mean crush force P_m (`thin_wall::crush`), static at σ₀ and at crash rate (the
deck's Cowper–Symonds at 15.6 m/s, `dynamic_factor`, its 0.33 V/C *to be verified*); M_p = σ_y·Z
of the section's box (the calculator's exact region); stroke 0.73 L. Sections as boxes of the cut's
outer size and equivalent gauge (an estimate where the cut is two stampings or a hat).

| car | member | section (mm) | t (mm) | σ_y / σ₀ (MPa) | L (m) | steel (kg) | squash (kN) | P_m static (kN) | rate | P_m crash rate (kN) | M_p strong / weak (kN·m) | stroke (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Camry | bumper beam (across the car) | 116 × 61 | 1.16 | 1300 / 1428 | 1.180 | 3.75 | 527 | 106.4 | 1.54 | 163.9 | 20.3 / 13.1 | – |
| Camry | bumper bracket | 69 × 116 | 2.06 | 360 / 499 | 0.090 | 0.53 | 268 | 98.3 | 1.54 | 151.1 | 10.5 / 7.3 | 0.066 |
| Camry | crash zone | 92 × 163 | 1.70 | 360 / 499 | 0.205 | 1.38 | 308 | 79.4 | 1.52 | 120.4 | 16.9 / 11.4 | 0.150 |
| Camry | mid rail | 83 × 145 | 2.06 | 360 / 499 | 0.315 | 2.28 | 332 | 105.4 | 1.52 | 160.6 | 16.2 / 11.0 | 0.230 |
| Camry | rear rail | 75 × 182 | 2.30 | 360 / 499 | 0.450 | 4.10 | 418 | 131.8 | 1.52 | 199.8 | 24.2 / 13.0 | 0.329 |
| Camry | upper rail (hat on the apron) | 75 × 75 | 1.02 | 250 / 302 | 0.550 | 1.30 | 75 | 17.2 | 1.55 | 26.7 | 2.1 / 2.1 | 0.402 |
| Camry | subframe side member (a side) | 70 × 48 | 2.30 | 360 / 499 | 0.920 | 3.77 | 188 | 101.7 | 1.57 | 159.5 | 4.5 / 3.4 | 0.672 |
| Camry | under-floor rail (kick-down) | 100 × 110 | 1.80 | 360 / 499 | 0.300 | 1.75 | 267 | 81.9 | 1.53 | 125.2 | 10.7 / 10.0 | 0.219 |
| Camry | sill | 141 × 155 | 1.24 | 400 / 584 | 1.100 | 6.29 | 291 | 57.7 | 1.51 | 87.0 | 16.5 / 15.5 | – |

### 1.4 The force against crush these members should give (estimate) against the tests

Both sides, crash rate, members engaging in the order the free distances give:

| crush (m) | what is crushing | estimate (kN) | Camry 8545 / 7520 measured (kN) |
|---|---|---|---|
| 0–0.07 | cover and foam | 20–50 | 53 / 30 at 0.05 |
| 0.07–0.13 | the beam pressed flat (its centre first, its ends at 0.13) | rising to the boxes' | 142 / 60 at 0.10 |
| 0.13–0.28 | two crash zones 2 × 120, upper rails 2 × 27, radiator support | **≈ 295** | 200–300 (249 / 219 at 0.15, 210 / 304 at 0.20) |
| 0.28–0.40 | mid rails 2 × 161, upper rails, the subframe's front reaching the stack | 375, rising | 294 / 210 at 0.30, rising |
| ≈ 0.40–0.43 | the engine (127 + 61 kg with its accessories) stopped against the wall through the radiator stack: beam-to-block 0.337 m + beam 0.066 m less the crushed stack | a peak set by the powertrain's momentum | **811 / 708 at 0.39; 921 / 829 peak at 30 ms** |
| 0.43–0.62 | rear rails 2 × 200 and the subframe's sides 2 × 160 behind the engine | 400–700 | 345–512 |
| ≈ 0.62–0.67 | the engine pushed back its 0.19–0.25 m to the rack and firewall | a second peak | 580–630 at 0.62 (≈ 55 ms) |

The reading reproduces the tests' sequence: the first plateau from the two crash zones (as the
spike found), the first peak where the engine meets the wall, the second where it meets the
firewall. **This is the check that the reading is right**; the rest of the plateau is not a fit.

## 2. What the generator builds (facts from the definitions and the rigged JBeam)

Generated sedan (1,448 kg; Camry test masses 1,662–1,664 kg). JBeam frame: y along the car, the
front axle at 0. Length 4.886 m (Camry 4.805); nose ahead of the axle 0.964 m (Camry 0.945).

| item | generated | Camry (as a share of length → this car) | difference and its consequence (evidence) |
|---|---|---|---|
| bumper beam | planar ladder: 5 node pairs across, one vertical plane (no depth along the car), 8 kN internal yield; 8 kg | B-section 61 deep, M_p 13–20 kN·m | no horizontal bending: it cannot spread a load or hold the boxes square; **mounted by 8 beams a side at 20 kN yield to the box's two lane nodes** (Camry: bolted plate) → the beam slides off the boxes' ends (lessons §26.2: +28 → −80 mm) |
| crash box | 70 × 110, 1.5 mm, 0.18 m, 4 nodes in one plane (or the closed cell, §26) | 92 × 163, 1.7 mm, 0.205 m + 0.09 m bracket | 0.18 m locks at `MIN_REST` after 0.08 m (44 %; 0.131 m at 73 %): **0.05 m of stroke a box lost** (lessons §26.4) |
| rail | one chord from the box's bottom node to `a4` (0.38 m) and `a5` (0.45 m), braces to the radiator support, apron, tower, upper rail | closed box to the firewall | a single-plane lattice; its bay carries 4.6–9 × its section with the braces (audit §1) |
| subframe | 0.45 m long, mounted by 16 beams a side at 157 kN to `a4`, `a5`, **the crash box's rear `a2`**, the apron `a7`, the radiator support's lower centre `a3_dc`, the floor, the firewall | 0.92 m, front bushings under the rail behind the crash zone, rear under the toe pan | its mounts spanned the rail's first bay (277 kN a side, audit §1, rule 1) |
| upper rail | `a3_u` – `a8_f` – `a8_m` – `b2_o` at the zone's 18.9 kN, braced to the box's nodes | ~27 kN at crash rate, its own member | braces from the boxes carry the upper rail's load into the crash box bay |
| engine | longitudinal block 0.6 m long, front 0.579 m behind the nose, rear 0.25 m ahead of the firewall (`sh_b1`), 152 kg on 8 + 2 nodes | front 0.491 m behind the nose (0.499 m scaled), block to firewall 0.249 m (0.253 scaled) | front **0.08 m further back** than the Camry's proportion (the drawn engine, fixed); rear gap the same |
| engine stops | 40 SUPPORT beams (35 kN yield, k 1e6) from the block's corners to the rails, radiator support, apron, subframe, firewall, engaging after **10 % of their length** (32 mm for `eng_1`–`a2`) | contact when the gap closes | the block is loaded through the crash zone long before anything touches it |
| firewall | `sh_b1` 0.465 m behind the axle (1.43 m from the nose: 29 % of length) | 0.22–0.24 m behind the axle (24 %) | the generated front is 0.24 m longer to the firewall |
| parallel paths | hood 272 kN and fenders 2 × 155 kN of yield across the box bay; SUPPORT stops (∞ yield) 1.6 MN of springs (audit §4) | hood and fenders buckle at tens of kN | rules 4, 5 |

Stroke of each generated member against `MIN_REST` (0.1 m): crash box 0.18 m → 0.080 m (44 %;
binding); rail bays 0.38 and 0.453 m → 0.277 and 0.331 m (73 %); upper-rail links 0.33–0.49 m →
73 %.

## 3. The design (switch `Physics::front_rules.load_path`, off by default)

Code: `engine-vehiclegen/src/front_rules.rs` (`impact_beam_bolted_plate`,
`subframe_on_rail_stations`, `BUMPER_PLATE_STRENGTH_N`), `physics.rs` (`FrontRules::resolved`,
`load_path_step`), the rigger's `gen2/load_path.rs`. It calls rules 1, 4, 5 (and 6's rate in
`real_crash`) and supersedes 2, 3 and the box cell: `FrontRules::resolved` turns them on and off.
`LOAD_PATH_STEPS` (a measurement switch: `cells`, `plate`, `mounts`, `upper`, `panels`, `engine`)
builds only some steps.

### 3.1 The rules, each with its derivation

1. **Stations and cells** (`cells`). Each side's front is one member from the bumper beam to the
   firewall: stations at the box's front and rear (`a1`, `a2`) and the rail's two stations (`a4`,
   `a5`), each a node at every corner of its section (the definitions' node at one corner, keeping
   its name; the others share its mass: car mass 1,447.7 kg and centre of mass unchanged within
   1 mm). Between stations a closed cell: four corner chords, a diagonal on each long face, both
   end faces closed. Source: the Camry's rail is a closed box from bracket to firewall (§1.2);
   section-matching.md §2 (chords at the section's corners).
2. **Each bay's force is its section's** (`cells`): axial yield = `Member::crush_n` × the
   character's front multiple (× the crash-rate factor in `real_crash`), flat (`deformLimitStress`
   = the yield, `deformLimit` 0.73) on chords and diagonals. Source: the calculator; section-matching
   rule 1–2. Sedan (`beamng_feel`): box 81.8 kN, rail bays 110.8 kN a side.
3. **The long faces' diagonals carry at most a quarter** (`DIAGONAL_SHARE`, *estimate*): with the
   zone's yield a 0.38 m rail bay's diagonals took 74 of 111 kN and the chords 9 kN (measured: the
   cell had a third of its bending). Chords now 15.3 (box) and 20.8 kN (rail).
4. **End faces held in shape**: a thin-walled section keeps its cross-section until its walls fold
   (P_m assumes it), so the face ties yield at no less than the bays' force. Measured: with the
   zone's 9 kN ties the box pivoted about its rear face.
5. **Nothing else across a bay** beyond `CRASH_BOX_BAY_BRACE_SHARE` (0.10, the estimate of rule 2)
   of the bay's force, along the car, together: braces from a station to the radiator support,
   apron, tower and upper rail, and the chain's skip beams. Measured on the sedan: 101 → 8 kN
   (box bay), 61 → 11 kN, 365 → 11 kN (rail bays) a side.
6. **The bumper beam's bolted plate** (`plate`): a beam node on its centreline in front of each of
   the box's four front corners (its 8 kg spread over 18 nodes), all 16 pairs to the corners at the
   box's own force (it does not fold first), each breaking at three M10 10.9 bolts' proof load, 144
   kN (*estimate of the bolt size*). Source: the Camry's three bolts a side (§1.2).
7. **The subframe under the rail stations** (`mounts`, with rule 1): its mounts on the four
   corners of `a4` (over its front corners) and `a5` (over its rear ones), the firewall's and the
   floor's nodes behind; not the crash box, the radiator support or the apron. Its beams' values
   are the definitions' (157 kN, k 1.5e6). Source: the Camry's hangers (§1.2).
8. **The upper rail its own member** (`upper`): its chain `a3_u`–`a8_f`–`a8_m`–`b2_o` at its own
   section's force (25.8 kN × 1.8 = 46.4 kN), flat; its skip beams at the brace share. Source: §1.1.
9. **Panels and stops** (`panels`): rules 4 and 5 (yielding stops, the hood row and the panels'
   buckling force), called.
10. **The engine's stops engage where the block meets what they bear on** (`engine`): rest
    length = the offset across the car with the gap along it closed to `pt_clearance` (0.03 m),
    never earlier than before and never under 0.12 m (`MIN_REST` + margin, so a yielding stop is
    not pushed back out). Source: the Camry's free distances are contact distances.
11. **The powertrain is not moved**: the drawn engine (GLB) fixes it. Its front is 0.08 m further
    back than the Camry's proportion; its rear gap matches (0.25 m against 0.253 m scaled).

**Look**: with the switch on, every design's `.glb` is byte for byte the same (0 of 465 files
changed); 109 files change on the six car designs (JBeam, `.rskin`); `truck_cab` and
`box_trailer` are untouched. Off: all 465 files byte for byte those of `33b57f525` (checked against
`~/Development/rubble-artifacts/sections/gen-before-33b57f525/SHA256SUMS`).

## 4. Measurements, one step at a time

CPU reference, generated sedan (`beamng_feel`), S11/S12 run 0 with edge contact, the audit's slow
push. Tags: `off` the car before; `lp_c` cells (rules 1–5 of §3.1); `lp_cp` + the plate;
`lp_cpm` + the subframe's mounts; `lp_cpmu` + the upper rail; `lp_cpmup` + rules 4 and 5; `lp`
every step. Plots and stills: `out/sections/load-path/` (and the artifacts copy).

![push and S11](../out/sections/load-path/front_ab_off_lp_c_lp_cp_lp_cpm_lp_cpmu_lp_cpmup_lp.png)

| | off | lp_c | lp_cp | lp_cpm | lp_cpmu | lp_cpmup | lp |
|---|---|---|---|---|---|---|---|
| cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN) | 1010 / 1606 | 783 / 1507 | 787 / 1507 | 787 / 1507 | 787 / 1507 | 520 / 130 | 520 / 130 |
|   of which crash boxes / box–other braces / hood + fenders (kN) | 194 / 224 / 401 | 164 / 16 / 414 | 164 / 16 / 414 | 164 / 16 / 414 | 164 / 16 / 414 | 164 / 16 / 149 | 164 / 16 / 149 |
| cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN) | 2236 / 1334 | 1866 / 1235 | 1869 / 1235 | 1062 / 1235 | 1112 / 1235 | 858 / 30 | 858 / 30 |
|   of which rail chords and braces / subframe–crash box (kN) | 348 / 553 | 222 / 544 | 222 / 544 | 222 / 0 | 222 / 0 | 222 / 0 | 222 / 0 |
| push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 162 / 394 / 538 / 509 | 127 / 192 / 172 / 229 | 154 / 155 / 196 / 225 | 151 / 202 / 159 / 213 | 147 / 187 / 145 / 193 | 126 / 157 / 121 / 101 | 126 / 143 / 153 / 111 |
| push at 0.39 m: crash-box chord / rail chords (m) | 0.128 / 0.371, 0.434 | 0.194 / 0.369, 0.447 | 0.184 / 0.380, 0.451 | 0.169 / 0.369, 0.448 | 0.170 / 0.370, 0.447 | 0.177 / 0.380, 0.449 | 0.180 / 0.375, 0.451 |
| push at 0.39 m: elastic / plastic (kJ) | 40.5 / 78.0 | 19.0 / 37.8 | 18.2 / 38.6 | 20.5 / 34.7 | 19.8 / 32.3 | 8.9 / 30.2 | 9.5 / 33.1 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 255 / 538 / 644 / – | 243 / 376 / 375 / 320 | 265 / 377 / 356 / 342 | 251 / 343 / 335 / 258 | 248 / 331 / 323 / 265 | 229 / 321 / 228 / 168 | 228 / 309 / 228 / 187 |
| S11 t_m (ms) | 44.4 | 73.6 | 73.6 | 74.4 | 73.6 | 71.9 | 73.3 |
| S11 dynamic crush (m) | 0.394 | 0.608 | 0.599 | 0.656 | 0.664 | 0.694 | 0.700 |
| S11 rebound (m/s) / restitution | 5.34 / 0.342 | 3.92 / 0.251 | 3.90 / 0.250 | 3.80 / 0.243 | 3.89 / 0.249 | 4.36 / 0.279 | 3.84 / 0.246 |
| S11 residual crush (m) | 0.199 | 0.369 | 0.388 | 0.512 | 0.516 | 0.523 | 0.539 |
| S11 peak wall force (kN) at (ms) | 661 at 20.0 | 409 at 16.0 | 384 at 16.0 | 419 at 44.0 | 434 at 44.0 | 504 at 60.0 | 491 at 44.0 |
| S11 4 ms / CFC 60 peak (g) | 50.0 / 59.1 | 31.0 / 32.8 | 28.1 / 31.4 | 31.2 / 40.9 | 37.4 / 44.4 | 46.4 / 48.7 | 42.8 / 55.0 |
| S11 cabin: door opening change peak / residual (mm) | 151 / -24 | 124 / -25 | 122 / -23 | 113 / -24 | 107 / -24 | 114 / -16 | 99 / -15 |
| S11 plastic work: cabin share | 0.061 | 0.020 | 0.022 | 0.014 | 0.017 | 0.027 | 0.027 |
| S11 elastic at the stop / 0.3 s later (kJ) | 47.3 / 9.7 | 30.7 / 7.9 | 29.9 / 7.1 | 27.1 / 8.1 | 26.9 / 8.0 | 27.9 / 3.9 | 25.8 / 4.4 |
| S11 released by zone (kJ) | cabin 16.6, front lower 8.2, front lower+upper 5.8, front upper 4.9, other 2.1 | cabin 12.8, front upper 5.2, front lower+upper 1.9, front lower 1.7, other 1.2 | cabin 11.9, front upper 4.4, front lower+upper 2.4, front lower 2.2, other 1.9 | cabin 8.0, front upper 4.3, front lower 3.3, front lower+upper 2.0, other 1.4 | cabin 8.5, front upper 4.7, front lower 2.7, front lower+upper 1.6, other 1.4 | cabin 15.5, front lower 3.9, other 2.0, front upper 1.4, front lower+upper 1.3 | cabin 12.7, front lower 4.1, other 2.0, front lower+upper 1.5, front upper 1.2 |
| S11 engine rear to firewall: least (m) at (ms); within 3 cm | 0.227 at 166; None | 0.215 at 68; None | 0.222 at 73; None | 0.196 at 73; None | 0.187 at 69; None | 0.157 at 74; None | 0.137 at 79; None |
| S12 t_m (ms) / crush (m) / restitution / residual (m) | 39.5 / 0.275 / 0.484 / 0.099 | 74.6 / 0.409 / 0.265 / 0.276 | 73.1 / 0.403 / 0.294 / 0.278 | 79.7 / 0.433 / 0.270 / 0.287 | 82.7 / 0.450 / 0.262 / 0.289 | 97.3 / 0.552 / 0.195 / 0.430 | 98.6 / 0.553 / 0.186 / 0.437 |
| S12 peak wall force (kN) at (ms) | 576 at 32.0 | 357 at 24.0 | 320 at 24.0 | 307 at 20.0 | 296 at 20.0 | 289 at 20.0 | 268 at 20.0 |
| hard: energy gain S11 / S12 (J); into the wall (m) | 0.0 / 1.4; 0.000 | 0.0 / 4384.5; 0.001 | 227.0 / 8832.0; 0.002 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.002 |

### 4.1 Does the front fold in sequence? (the push, bay by bay)

Push depth 0.2 / 0.4 / 0.6 m; Δz each bay's shortening along the car, chord its outboard bottom (rail) or top (box) chord's length (rest 0.180, 0.380, 0.453 m):

| variant | push (m) | box a1–a2: Δz / chord (m) | rail a2–a4: Δz / chord | rail a4–a5: Δz / chord | a5–firewall Δz | firewall–cabin Δz | beam to box front (m) | box front drop (m) | force (kN) |
|---|---|---|---|---|---|---|---|---|---|
| off | 0.2 | 0.007 / 0.173 | 0.004 / 0.377 | 0.012 / 0.442 | 0.002 | 0.024 | -0.063 | 0.018 | 400 |
| off | 0.4 | 0.102 / 0.129 | 0.032 / 0.370 | 0.024 / 0.433 | -0.023 | 0.116 | -0.060 | 0.016 | 508 |
| off | 0.6 | 0.180 / 0.143 | 0.058 / 0.368 | 0.039 / 0.423 | -0.025 | 0.195 | -0.063 | -0.004 | 710 |
| lp_c | 0.2 | 0.029 / 0.178 | 0.006 / 0.376 | 0.002 / 0.451 | -0.002 | 0.013 | -0.061 | 0.019 | 192 |
| lp_c | 0.4 | 0.327 / 0.194 | 0.029 / 0.373 | 0.008 / 0.448 | 0.003 | 0.029 | -0.004 | -0.051 | 226 |
| lp_c | 0.6 | 0.295 / 0.195 | 0.201 / 0.375 | 0.022 / 0.451 | -0.001 | 0.043 | -0.003 | -0.035 | 355 |
| lp_cp | 0.2 | 0.100 / 0.177 | 0.010 / 0.377 | 0.003 / 0.452 | -0.004 | 0.013 | -0.009 | 0.024 | 150 |
| lp_cp | 0.4 | 0.300 / 0.184 | 0.053 / 0.380 | 0.004 / 0.451 | -0.003 | 0.030 | 0.021 | -0.051 | 226 |
| lp_cp | 0.6 | 0.269 / 0.187 | 0.200 / 0.380 | 0.019 / 0.454 | 0.004 | 0.043 | 0.024 | -0.053 | 348 |
| lp | 0.2 | 0.006 / 0.178 | 0.015 / 0.379 | 0.033 / 0.449 | 0.023 | 0.011 | 0.008 | 0.013 | 143 |
| lp | 0.4 | 0.017 / 0.180 | 0.104 / 0.375 | 0.132 / 0.451 | 0.012 | 0.011 | 0.014 | -0.014 | 111 |
| lp | 0.6 | 0.112 / 0.179 | 0.347 / 0.377 | 0.082 / 0.455 | -0.007 | 0.021 | 0.039 | 0.016 | 267 |

**No, not by axial folding.** The bays now shorten in order (box first, then the rail's first
bay, then its second) and the bumper beam stays on the boxes (beam to box front within 0.04 m
throughout; before, −0.06 m), but each bay loses its span along the car while its chords keep
their length: it **turns** (a bending collapse of the cell), it does not crush. In `lp` the rail's
first bay pitches (its front 0.09–0.32 m above its rear) while `a4`, the station the subframe hangs
from, drops 0.15–0.31 m below `a5` (push columns `y_a2_a4`, `y_a4_a5`). The cause is the lattice,
not a connection: **a pin-jointed four-chord cell crushing at its section's P_m has no bending
reserve** (every chord already at its yield; the real section, folding at P_m ≈ 0.3 of its squash,
keeps most of σ_y·Z: the Camry crash zone 16.9 kN·m against our cell's ≈ 81.8 × 0.055 ≈ 4.5
kN·m). Any eccentricity, the subframe's mounts at `a4`, the kick-down's reaction at `a5`, makes
the cell turn first. Section-matching.md §2 says corner chords at P_m cannot carry M_p; this is
the dynamic consequence.

### 4.2 S11 and S12 beside the Camry tests

S11, edge contact, run 0 (Camry columns: `crash-reference.md` §4.3):

| quantity | Rubble before (off) | Rubble, load path | Rubble, all seven earlier switches | test 8545 | test 7520 |
|---|---|---|---|---|---|
| mass (kg) | 1448 | 1448 | 1448 | 1664 | 1662 |
| peak decel, 4 ms estimator / CFC 60 (g) | 50.0 / 59.1 | 42.8 / 55.0 | 45.8 / 51.6 | 36.8 / 39.8 | 40.7 / 45.0 |
| time of CFC 60 peak (ms) | 15.5 | 53.5 | 53.5 | 44.1 | 43.9 |
| t_m (ms) | 44.4 | **73.3** | 54.5 | 70.9 | 75.9 |
| dynamic crush (m) | 0.394 | **0.700** | 0.507 | 0.663 | 0.685 |
| mean decel to t_m (g) | 35.9 | **21.7** | 29.2 | 22.7 | 21.0 |
| rebound (m/s) / restitution | 5.34 / 0.342 | 3.84 / 0.246 | 4.91 / 0.315 | 1.50 / 0.095 | 0.89 / 0.057 |
| residual crush (m) | 0.199 | **0.539** | 0.307 | 0.554 | 0.579 |
| peak wall force (kN) at (ms) | 661 at 20 | 491 at 44 | 535 at 40 | 921 at 29.5 | 829 at 30.1 |
| elastic at the stop / 0.3 s later (kJ) | 47.3 / 9.7 | 25.8 / 4.4 | 34.6 / 5.9 | – | – |
| engine rear to firewall, least (m) | 0.227 | 0.137 | 0.227 | reaches it (second peak ≈ 55 ms) | |
| cabin: door opening change peak / residual (mm); plastic work share | 151 / −24; 0.061 | 99 / −15; 0.027 | 116 / −9; 0.030 | – | – |

S12, edge contact, run 0:

| quantity | Rubble before (off) | Rubble, load path | Rubble, all seven | test 9692 | test 7793 |
|---|---|---|---|---|---|
| mass (kg) | 1448 | 1448 | 1448 | 1690 | 1798 |
| peak decel, 4 ms / CFC 60 (g) | 46.2 / 52.7 | 21.8 / 22.5 | 26.6 / 28.2 | 27.1 / 34.1 | 26.1 / 31.5 |
| t_m (ms) | 39.5 | 98.6 | 66.1 | 69.3 | 74.7 |
| dynamic crush (m) | 0.275 | 0.553 | 0.370 | 0.475 | 0.489 |
| restitution | 0.484 | 0.186 | 0.301 | 0.149 | 0.107 |
| residual crush (m) | 0.099 | 0.437 | 0.192 | 0.304 | 0.279 |
| peak wall force (kN) at (ms) | 576 at 32 | 268 at 20 | 400 at 24 | – | – |

Over 8 runs × 3 contact settings (§5): S11 restitution 0.233–0.253 (discrete, edge) and
0.267–0.281 (continuous), crush 0.688–0.710 m, residual 0.498–0.546 m; S12 restitution
0.154–0.210, crush 0.499–0.564 m.

## 5. Hard rules and gates

8 runs × discrete/edge/continuous, CPU, `crash_reference::verdict` (`front_verdicts.py`; the car
before, `off`, is ok in all 15 settings with its known failures: its files are byte for byte
those of `33b57f525`, audit §9.6):

| variant | settings judged | new failures (scenario, setting: criterion, runs) | known failures that now pass in every run |
|---|---|---|---|
| lp | 15 | S08 edge: no energy created 1/8, S12 continuous: no energy created 1/8 | S08 continuous: the common centre of mass stays still |

| variant | scenario | contact | runs | energy created, max (J) | its band | into the wall, max (m) | finite | crossings, max | asserted failures | restitution | dyn. crush (m) | residual (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lp | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.002 | 1 | – | restitution within the 56 km/h sedan fleet's 2/8 | 0.233–0.253 | 0.693–0.710 | 0.534–0.546 |
| lp | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.002 | 1 | – | restitution within the 56 km/h sedan fleet's 2/8 | 0.233–0.253 | 0.693–0.710 | 0.534–0.546 |
| lp | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.013 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.267–0.281 | 0.688–0.708 | 0.498–0.513 |
| lp | S12 | discrete | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.172–0.186 | 0.540–0.564 | 0.429–0.445 |
| lp | S12 | edge | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.172–0.186 | 0.540–0.564 | 0.429–0.445 |
| lp | S12 | continuous | 8 | 106.9 | ≤ 89.0 | 0.006 | 1 | – | no energy created 1/8 | 0.154–0.210 | 0.499–0.537 | 0.382–0.410 |
| lp | S01 | discrete | 8 | 16.8 | ≤ 230.0 | 0.017 | 1 | – | none | 0.176–0.190 | 0.919–0.948 | 0.067–0.147 |
| lp | S01 | edge | 8 | 16.8 | ≤ 230.0 | 0.017 | 1 | – | none | 0.176–0.190 | 0.919–0.948 | 0.067–0.147 |
| lp | S01 | continuous | 8 | 11.8 | ≤ 230.0 | 0.013 | 1 | – | none | 0.181–0.200 | 0.914–0.938 | 0.076–0.129 |
| lp | S02 | discrete | 8 | 0.6 |  | 0.004 | 1 | – | none | 0.177–0.209 | 0.473–0.492 | 0.065–0.294 |
| lp | S02 | edge | 8 | 0.6 |  | 0.004 | 1 | – | none | 0.177–0.209 | 0.473–0.492 | 0.065–0.294 |
| lp | S02 | continuous | 8 | 0.0 |  | 0.027 | 1 | – | none | 0.183–0.214 | 0.471–0.487 | 0.136–0.300 |
| lp | S08 | discrete | 8 | 125.5 | ≤ 340.0 | 0.000 | 1 | 99.0 | none | 0.095–0.122 | 1.092–1.126 | -0.035–-0.001 |
| lp | S08 | edge | 8 | 1451.0 | ≤ 340.0 | 0.000 | 1 | 40.0 | momentum across the road 1/8, no energy created 1/8 | 0.094–0.151 | 1.071–1.096 | 0.020–0.079 |
| lp | S08 | continuous | 8 | 962.5 | ≤ 340.0 | 0.000 | 1 | 0.0 | no energy created 1/8 | 0.039–0.147 | 1.034–1.110 | -0.007–0.009 |

S08 edge's 1.45 kJ (one run) is the edge contact's known position step that every deeper-folding
front has reached (rules 2–4, lessons §25.8); S12 continuous's 107 J (one run, bar 89 J) is new.
Nothing enters the wall beyond 0.027 m (S02 continuous; 0.002 m before).

Gates (`front_gates.sh lp load_path`: the switch on as every design's default), against the car
before (audit §9.7):

| test | before | load path |
|---|---|---|
| `engine-vehiclegen` lib (63 + 3 new), `crash_zones`, `mechanical`; authoring `crash_zones`, `truck_cab`, `box_trailer`, `crash_invariants` (9), `rig_export` (2), `skin_lint` (3), `paint_binds_skin` (1), the new `load_path` test | pass | pass |
| `every_design` (≤ 5 mm parked) | pass (4x4 4.9 mm) | **fails**: Mach 1 `subf_rr` 11.6 mm (the subframe's new mounts) |
| `sedan_energy` | 6/6 | **fails** `every_part_torn_off_and_spun_adds_no_energy`: the hub cap FL spun at 5 m/s gains 21 J (its kinetic 39 J); not investigated |
| `generated_vehicle_lint` | the two listed | the same two; inside them the counts of rule 5's hood row (Mach 1 winding 4 → 6, gasser facing 54 → 56, 4x4 contacter pairs 21 → 23), as all seven switches gave |
| `gen3_rig::rest_fidelity` (fails before) | parked 1.090° | parked 1.367° |
| `gen_sedan_contact`, `continuous_contact` | the six / three listed | the same |
| `gen_sedan_paint::accept_head_60` (50 mm) | pass | **fails**, 51 mm |
| `gen_sedan_paint::accept_head_60_pressed` (50 mm) | pass, 21 mm | **fails**, 70 mm |
| `gen_sedan_paint::accept_side_90`, `accept_side_60_pressed`, `accept_head_100` (listed failing) | fail | **pass** (would leave `tests-expected-failures.txt`) |
| `gen_sedan_paint::own_paint_ends_apart` (listed) | fails | fails |

Not edited: no band, known failure or expected failure. Would change if it were switched on: the
three paint entries above leave the expected-failures list; S08 continuous's known "the common
centre of mass stays still" passes in every run; the matrix's car, fixtures and §4.3 rows are
regenerated.

## 6. What remains wrong, and why

- **The cells turn instead of folding** (§4.1): a lattice limit (corner chords at P_m have no
  bending reserve while crushing). What would fix it is a beam law the solver does not have: a
  compressive yield (the folding force) below a tensile one (the wall's squash share), or a peak
  then a lower plateau; or many more nodes along each member. Both are solver or cost decisions
  for the owner; nothing was changed in the solver.
- **The first plateau is low** (push 130–200 kN against the Camry's 200–300): the generated
  sections are 0.6 of the Camry's P_m statically (crash box 45 against 79 kN, rail 62 against
  105–132), so even at `beamng_feel`'s 1.8 the boxes are 2 × 82 kN against 2 × 120; and the turning
  bays fall below their section's force.
- **The engine stack is small and late**: the first peak at 0.53–0.56 m and 490 kN (Camry
  0.40–0.43 m and 830–920 kN). The engine's front is 0.08 m further back than the Camry's
  proportion (drawn, not moved), and the generated car is 13 % lighter (177 against 203 kJ). The
  engine now comes within 0.137 m of the firewall in S11 (0.227 m before), not yet the Camry's
  second peak.
- **S12 is now too soft**: t_m 99 ms and 0.55 m against 69–75 ms and 0.48–0.49 m: the turning
  bays' plateau is lower than the sections' at 40 km/h.
- **Restitution** 0.246 (S11) against 0.06–0.10: the cabin still releases 12.7 kJ (the largest
  share of what comes back).
- `MIN_REST`: the crash box's 0.18 m reaches 0.08 m of stroke (0.131 m at 73 %), 0.05 m short.
- Estimates: the brace share 0.10, the diagonal share 0.25, the bolt size, the Camry sections
  taken as boxes of the cut's outer size.

## 7. Recommendation

**Keep it off by default** (the standard: on only if, alone, it holds every hard rule and gate
that held before). With the switch on, alone (it is not meant to be combined with the seven: it
calls rules 1, 4, 5 and 6's rate and supersedes 2, 3 and the box cell, which become redundant
under it):

- **hard rule 1 breaks** in two settings that held before: S08 edge 1.45 kJ in 1 of 8 runs (bar
  340 J; the edge contact's known energy fault, which the folding fronts reach, as rules 2–4 did),
  S12 continuous 107 J in 1 of 8 (bar 89 J);
- **gates that held break**: `every_design` (the Mach 1's subframe rear corner 11.6 mm parked, bar
  5 mm), `sedan_energy` (a spun hub cap gains 21 J), the paint guard at head 60 (51 mm) and head 60
  pressed (70 mm, bar 50 mm); three listed paint failures pass.

What it gains against the real tests is the largest of any round: S11 t_m, dynamic crush,
residual crush and mean deceleration inside or at the Camry tests' range (73 ms, 0.70 m, 0.54 m,
21.7 g against 71–76 ms, 0.66–0.69 m, 0.55–0.58 m, 21–23 g), restitution 0.342 → 0.246, the
cabin's plastic share 6 % → 3 %, the engine 0.09 m nearer the firewall. But it gets there by the
cells turning, not folding (§4.1), its peak is small and late, and S12 is now too soft.

**Next, for the owner:** (1) decide whether the solver may have a compressive yield separate from
the tensile one (or a peak-then-plateau law): without it no four-chord cell at P_m folds instead of
turning, so no generator rule can make this front fold axially; (2) the edge contact's energy fault
(S03/S14, S08) keeps catching every front that crushes further; (3) the Mach 1's subframe mount
(drift 11.6 mm) is a fix in the rule's mount pairs, not yet made; (4) the drawn engine sits 0.08 m
further back than the Camry's proportion: a look change if wanted.

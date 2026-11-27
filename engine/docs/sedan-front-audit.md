# The generated sedan's front structure, audited against its own sections

The generated sedan stops in 44 ms where the Camry tests take 71–76 ms, crushes 0.39 m against
0.66–0.69 m and springs back at 5.3 m/s against 0.9–1.5 m/s (`crash-reference.md` §4.3, S11).
This audit asks why, member by member, with the section calculator and the beam law of
[section-matching.md](section-matching.md). **Nothing in the generator or the solver was
changed.** 2026-11-27, CPU reference, the generated `sedan` (default `beamng_feel`), 1,448 kg.

Reproduce (from `engine/`; outputs in the gitignored `out/sections/`, copies in
`~/Development/rubble-artifacts/sections/`):

```
cargo test --release -p engine-softbody-lab --test sedan_front_audit -- --ignored --nocapture --test-threads 1
cargo run --release -p engine-softbody-lab --example crash_ref -- S11 --json > out/sections/s11.jsonl
out/sections/venv/bin/python tools/sections/plot_front.py      # matplotlib, local
```

`sedan_front_members` (the definitions and the calculator), `sedan_front_cuts` (the rigged
lattice cut by planes), `sedan_front_push` (quasi-static push), `sedan_s11_stored_energy` (S11 with
the elastic energy by member). Facts, measurements and hypotheses are marked.

## 1. Headline

| | Stated section (generator) | Calculator: squash σ_y·A / P_m static σ₀ / P_m at 15.6 m/s (deck; mild) | `crush_n` today | Rigged: axial yield across the member's bay, per side | × stated P_m static / at crash rate | Travel before its beams lock | Hardening at the lock | Elastic at first yield |
|---|---|---|---|---|---|---|---|---|
| Crash box | 70 × 110, 1.5 mm, HSLA (σ_y 340, σ₀ 395 MPa), 0.18 m | 181 / **45.4** / **70** (94) kN | 45.4 kN × 1.8 (`beamng_feel`) = 81.8, as two chords of 40.9 | box: 2 chords 81.8 + its diagonals 15.4 = **97 kN**; bay (+ braces from its nodes to the radiator support 51, rail 32, upper rail 28): **207 kN** | box 2.1 / 1.4; bay **4.6 / 3.0** | **44 %** of 0.18 m (MIN_REST 0.1 m); stated 73 % | 1.8 × | 206 J a chord |
| Front rail | 70 × 110, 1.8 mm, HSLA, 1.04 m | 216 / **61.6** / **95** (128) kN | 61.6 × 1.8 = 110.8 on one chord (`sh_a2_d`–`sh_a4`; `sh_a4`–`sh_a5` 92.4) | ahead of the engine (z 0.47 m): rail chord and braces 174 + **subframe mounts to the crash box's rear nodes 277** + radiator support–subframe 131 (half) ≈ **580 kN** | **9.4 / 6.1** | 74–78 % | 3.8–4.5 × | 1.5 kJ a chord; 8.2 kJ a subframe mount beam |
| Upper rail | 40 × 60, 1.2 mm, HSLA, 0.97 m | 80 / 25.8 / 41 kN | not used: its chord is the zone's 10.5 × 1.8 = 18.9 kN | 18.9 kN chord; fender bolts (25 kN, elastic) on it | 0.73 / 0.46 | 79 % | 4.9 × | 110 J |
| Radiator support | uprights 40 × 60, 1.0 mild; upper and lower 60 × 40, 1.0 mild | 35 / 11.5 / 18 kN | not used | own beams 17.8 kN mean; 24 braces of 18.9 kN to the crash boxes | – | 77–81 % | 4.6–5.8 × | – |
| Impact (bumper) beam | no section stated: a mechanical part, 8 kg, 50 × 110 visual | – | – | internal 27 beams 8 kN; mounts 8 beams a side, 20 kN yield, 60 kN strength, k 1e6, 0.13–0.23 m | – | **41 %** (MIN_REST) | 1.8 × | 200 J a mount beam |
| Bumper cover's stops | – | – | – | SUPPORT beams to the crash boxes and fenders, **deform ∞**, strength 150 kN, k 6e5: up to 634–706 kN elastic ahead of the crash boxes | – | never yield | – | 3.2 kJ held at t_m in S11 |
| Hood | 4 node rows, 0.45 m apart | – | – | 45 beams of ~4.3 kN across the crash-box bay: **194 kN**; SUPPORT stops to the radiator support and fenders (deform ∞, 150 kN): 807 + 298 kN elastic | – | 78 % | 5.6 × | – |
| Fenders | – | – | – | 200 kN across the crash-box bay | – | 72 % | 4.5 × | – |
| Front subframe | literal values (no section) | – | – | internal 88 kN; **16 mount beams a side at 157 kN yield, 400 kN strength**, to the rail, crash box rear, apron, firewall, floor, radiator support | – | 60–78 % | 3–5 × | 8.2 kJ a beam |
| Firewall / toe pan | panel, 0.9 mm mild, 0.71 m² | – | – | the cut behind the engine (z −0.49): ≥ 2,000 kN that yields (subframe rear mounts 551 + 327) | – | – | – | – |
| Engine mounts | bounded 32 kN/m, 20 kN yield, 90 kN strength; engine stops (SUPPORT, 35 kN, k 1e6) to the rails, radiator support, firewall | – | – | – | – | – | – | – |

**Fact** (from the definitions and the rigged JBeam): the crush members' yields come from
`Member::crush_n` (Abramowicz–Jones at the grade's flow stress (σ_y + σ_u)/2, quasi-static) times
`CrashFeel::front_crush` = 1.8 (`beamng_feel`, the default), split over the member's lengthwise
edges (`gen2/shell.rs` `lattice_defined`: two crash-box chords a side, one rail chord). Every
other front beam takes its zone's value (`Zone::FrontRail` 10.5 kN × 1.8 = 18.9 kN, `Tower` 27.5 ×
1.5 = 41.2, `Cabin` 22 × 1.5 = 33), and defined parts keep their literals (subframe, impact beam,
engine, radiator). The shell beams carry no `deformLimit` (the rigger clears it on defined beams)
and no `deformLimitStress`, so every one of them hardens as 1/L until MIN_REST.

**In words:** the crash boxes are 2.1 × their stated section's static fold force (1.4 × at crash
rate) but their bay carries 4.6 × once the braces the shell lattice runs across it are counted;
the rails' bay is 6–9 × their section, because the front subframe's mounts reach forward to the
crash boxes' rear nodes at 157 kN a beam; the crash-box chords can crush 44 % of their length,
not 73 %; the hood and fenders add about 400 kN that a real hood and fender (which buckle at tens
of kN) do not; and SUPPORT stops that never yield sit across the bumper, hood and fender seams.

## 2. The push: force against displacement

`sedan_front_push`: the rigged sedan at 1 m/s into the rigid wall (5 cm gap), its body shell
behind z = −1.5 m (the B-pillar and rearward) ballasted to 2 × 10⁴ times its mass with the extra
weight cancelled by an upward force each step (the cabin slows from 1.000 to 0.94 m/s over 0.74 m),
the wall force −d(Σ m v_z)/dt of the whole car. (Holding those nodes' velocity instead was
unstable: light door and sill nodes beside the held ones reached 450 m/s in 25 steps: a
measurement-setup finding, not a vehicle one.)

![force against displacement](../out/sections/front_force_displacement.png)
(`~/Development/rubble-artifacts/sections/front_force_displacement.png`)

| d (m) | push (kN) | S11 56 km/h (kN) | Camry 8545 / 7520 (kN) | crash-box chord (m, from 0.180) | rail chords (from 0.380 / 0.453) | elastic / plastic (kJ) |
|---|---|---|---|---|---|---|
| 0.05 | 77 | 173 | 53 / 30 | 0.178 | 0.380 / 0.451 | 2.2 / 0.0 |
| 0.10 | 166 | 306 | 142 / 60 | 0.177 | 0.380 / 0.449 | 6.9 / 1.0 |
| 0.15 | 243 | 306 | 249 / 219 | 0.177 | 0.381 / 0.448 | 13.7 / 2.9 |
| 0.20 | 400 | 478 | 210 / 304 | 0.173 | 0.377 / 0.442 | 22.4 / 8.4 |
| 0.25 | 499 | 645 | 259 / 200 | 0.156 | 0.374 / 0.436 | 29.8 / 22.2 |
| 0.30 | 537 | 632 | 294 / 210 | 0.138 | 0.373 / 0.431 | 35.0 / 42.7 |
| 0.39 | 500 | 549 (t_m) | 811 / 708 | 0.128 | 0.371 / 0.434 | 40.7 / 78.2 |
| 0.50 | 381 | – | 345 / 364 | 0.141 | 0.376 / 0.433 | 43.7 / 117.9 |
| 0.60 | 710 | – | 512 / 430 | 0.143 | 0.368 / 0.423 | 57.9 / 154.3 |
| 0.70 | 975 | – | 273 / 221 | 0.144 | 0.364 / 0.396 | 76.7 / 203.0 |

The stated sections' prediction (green: two crash boxes folding over 73 % of 0.18 m from the
impact beam's contact at 0.10 m, then two rails): 91 then 123 kN static, 140 then 190 kN at crash
rate; with the upper rails and radiator uprights in parallel 258 then 308 kN; × 1.8 (`beamng_feel`)
252 then 341 kN.

**Measurements:**
- To 0.17 m the quasi-static push follows the Camry tests; then it climbs to **500–540 kN from
  0.25 to 0.45 m**, twice the Camry's 200–300 kN plateau and 3–4 × the stated sections at crash
  rate. It dips to 380 kN at 0.50 m and climbs to 1 MN at 0.74 m.
- S11 (dynamic, 56.3 km/h) sits 20–85 % above the push from the start (306 against 166 kN at
  0.10 m): the front's inertia and the beams' damping; the solver has no strain-rate law.
- **The first 0.2 m is mostly elastic.** At 0.20 m the beams store 22.4 kJ and have yielded 8.4
  kJ (73 % of the work is elastic); at 0.39 m 40.7 kJ elastic against 78 kJ plastic (34 %). Work
  to 0.39 m: 127 kJ quasi-static; S11 brings 177 kJ.
- **The rails do not crush.** At 0.39 m the rail chords have shortened 9 mm and 19 mm; the
  crash-box chord 52 mm (29 %), and then it lengthens to 0.144 m (the bay distorts rather than folds). Where the 0.39
  m went (push): nose to impact beam 0.10 (the cover), impact beam to crash boxes 0.09, crash
  boxes' front to firewall 0.13, **firewall to cabin rear 0.11** (the cabin's front half
  shortening under 500 kN: 18 kJ of cabin yield at 0.39 m).
- **The engine never reaches the firewall:** its rear stays 0.23–0.26 m from it through 0.74 m of
  push (and in S11: least 0.227 m). Its front closes on the radiator from 0.25 to 0.08 m by 0.74 m.
  The Camry's peak of 830–920 kN at 0.42 m (the engine stacking) has no counterpart at that depth.

![push internals](../out/sections/push_internals.png)

## 3. Where the rebound comes from (S11)

`sedan_s11_stored_energy`: S11 run 0, edge contact, the elastic energy of every beam by member at
the moment the car's centre of mass stops (45 ms), and 0.3 s later.

- Impact kinetic energy 177.1 kJ. **Elastic energy at the stop 47.3 kJ (27 %)**, largest 47.9 kJ
  at 41 ms; 9.7 kJ left 0.3 s later; **37.7 kJ released**. The car leaves at 5.10 m/s (centre of
  mass; 5.34 at the rear-seat crossmember): **18.8 kJ = 10.6 %** of the impact's energy, the rest
  of the released energy going to vibration (5 kJ internal kinetic at the stop) and damping.
- Released, by zone: **cabin 11.1 kJ (29 %)** (+ the seats on its floor 3.2), front lower
  (crash boxes, rails, impact beam, subframe, radiator support) 8.2, front lower with upper
  (braces and stops between them) 5.8, front upper (hood, fenders, upper rails, towers, bumper
  cover) 4.9, the rest 4.4.
- Largest single classes released: cabin beams 5.8 kJ, crash-box beams 3.4, cabin–seats 3.2,
  **bumper cover's stops on the crash boxes 3.1 (of 3.2 held: they never yield)**, bumper cover
  1.9, cabin–cowl 1.7, fenders 1.4, hood 1.1.
- Where the dynamic crush went and what came back (S11, at t_m → at the end): nose to impact beam
  0.079 → 0.015 m, impact beam to crash boxes 0.086 → 0.014 m, crash boxes to firewall 0.190 →
  0.155 m, **firewall to cabin rear 0.064 → 0.009 m**. So 0.064 + 0.072 + 0.035 + 0.055 = **0.23 m
  of the 0.42 m comes back**: the bumper cover and the impact beam's span almost wholly, the
  cabin's front half wholly, the front structure a fifth.

**Hypothesis, supported:** the rebound is the elastic energy a softened lattice holds at the force
the over-strong front sustains (≈ 550 kN at t_m against the Camry's ≈ 200 kN). Stored energy at a
given compliance goes as F², so halving the force at t_m quarters it; the remainder is in stops
that never yield (bumper cover, hood, fenders) and in the cabin, which carries the whole force
elastically.

## 4. The lattice cut by planes

`sedan_front_cuts` sums, for every plane z = const from the nose back, the axial capacity of
every beam crossing it: Σ threshold × cos for beams that yield, Σ strength × cos for beams that
never yield (springs until they break). This is the bay's axial collapse load if every crossing
beam shortened together: an upper bound on what the push meets (the hood and fenders, for one,
lift and bend rather than shorten).

![cuts](../out/sections/front_cuts.png)

| Plane | Yielding (kN) | of which | Never yielding (kN) |
|---|---|---|---|
| z 0.70 (crash-box bay) | 1,009 | crash boxes 194, hood 194, fenders 200, crash box–radiator support 103, crash box–rail 63, crash box–upper rail 55, bumper cover and beam 192 | 1,606: hood–radiator support stops 807, fender–hood 298, cover–fender 272, … |
| z 0.47 (rails ahead of the engine) | 2,068 | crash box–subframe mounts 553, rail chords and braces 348, radiator support–subframe 263, fenders 200, hood 194, … | 1,334 |
| z 0.30 (beside the engine) | 2,148 | as above plus fenders 292, hood 231 | 575 |

The two Camry crash boxes at crash rate carry 240 kN; the stated sections' 140 kN.

## 5. The hood's four rows

**Fact:** the hood's nodes stand in four rows across the car at z −0.50, −0.05, 0.39 and 0.84 m,
0.45 m apart (`hd_o0*`–`hd_o3*`, an inner node per row). It can bend only at the two inner rows.
**Consequence (measured in the cut):** its front 0.45 m is one bay of 45 beams of 4–8 kN, 194 kN
of axial yield in parallel with the crash boxes, which can shorten only by those beams crushing
(hardening to 5.6 × at their lock) or by the whole panel pivoting at the second row. A real
hood is built to buckle in its first 0.2–0.3 m at a few tens of kN (crush beads). It also rests
on SUPPORT stops to the radiator support and fenders (deform ∞, 150 kN, k 4.5–6e5): 1.1 MN of
elastic capacity across the same bay. In S11 the hood holds 1.5 kJ at t_m and its stops 0.5 kJ:
small, because it lifts rather than crushes; in the push the hood and cowl yield 4 kJ by 0.39 m.

## 6. Proposed generator changes, ranked

None changes the approved look: each is nodes and beams inside the skin, or a value. Each one
re-runs `every_design`, the lint (`generated_vehicle_lint.rs`) and the rest fidelity, and is
measured with the push and S11 (8 runs), as `deciding-tradeoffs` asks.

1. **Take the subframe's forward mounts off the crash boxes' rear nodes** (`mechanical.rs`
   `subframe_front` mounts `to: sh_a2_d, sh_a2_u` at 157 kN a beam), or give the beams that
   span the rail's crush bay a yield inside the rail's share. *Evidence:* 277 kN a side in the
   rails' bay beside a rail chord of 111 kN; the rail chords shorten 9–19 mm by 0.39 m of push.
   *Expected:* the rails fold; the plateau after the crash boxes falls from ~500 kN toward
   200–300 kN; dynamic crush deepens toward 0.6 m; t_m lengthens; the engine meets the radiator
   stack and the second peak appears near the Camry's depth. *Measure:* push force at 0.25–0.45
   m, the rail chords' shortening, S11 dynamic crush, t_m, restitution.
2. **Make the crash-box bay the crash box only**: the shell's bracing beams across the bay
   (crash box to radiator support, rail, upper rail, towers: 110 kN a side at the zone's 18.9 kN)
   at a small share of the box's force or routed outside the bay. *Expected:* the first plateau
   (0.10–0.25 m) at 2 × 70 kN plus the impact beam and cover, near the Camry's 200–250 kN.
   *Measure:* push force at 0.10–0.25 m; crash-box chord shortening.
3. **Give the crash box its stroke**: chords at least 0.37 m (one bay from the impact beam's
   mount to the rail, or the box's nodes so placed), `deformLimit` 0.73 and `deformLimitStress`
   = the chord's yield (no 1/L hardening, a flat fold), as `vehicle.rs` already does for its crush
   members (`CRUSH_STROKE`). *Evidence:* the 0.18 m chords lock at 44 % and 1.8 × (section 3 of
   section-matching.md). *Expected:* 0.13 m of crash-box travel at a flat force instead of 0.08 m
   rising; less elastic energy at the lock (½F²/k at 1.8 × the force is 3.2 × the energy).
4. **SUPPORT stops in the crush zone that yield**: the bumper cover's stops to the crash boxes
   and fenders, the hood's to the radiator support and fenders (deform ∞, 150 kN today) at a clip's
   or a rubber stop's few kN. *Evidence:* 3.1 kJ of the bumper's stops returned in S11; the nose to
   impact beam span returns 6.4 of 7.9 cm. *Expected:* rebound down by about 3–5 kJ (0.6 m/s);
   front force down 0–100 kN early. *Measure:* restitution, elastic by class at t_m.
5. **The hood and fenders: a fold row near the front and axial yields of a sheet's buckling**: a
   hood row 0.15–0.25 m behind its front edge (a crush bead's line) and the hood's and fenders'
   lengthwise beams at their panel's buckling force. *Evidence:* 194 + 200 kN of yield across the
   crash-box bay. *Expected:* 100–300 kN less in the cut; little in S11 (the hood lifts), more in
   offset and pole scenes. Adds binding nodes: check rest fidelity and the skin.
6. **The front members at crash rate, not × 1.8**: in `real_crash`, the crush members' yields at
   P_m × the deck's 1.5 (`thin_wall::crush::dynamic_factor`), so a 56 km/h scene sees the section's
   crash-rate force; `beamng_feel`'s 1.8 is an ETK-matching choice the owner made
   (`generated-vehicles-lessons.md` §13) and stays its own preset. *Expected:* crash boxes 97 →
   76 kN a side (with 2.); rails 111 → 92 kN. *Measure:* S11 with each preset.
7. **The cabin carries the whole force elastically** (0.064 m of the firewall–B-pillar span in
   S11, 0.055 m returned, 11 kJ released). Fixed mostly by 1–3 (the force at t_m); if not, the
   cabin's springs are under the node budget and could be stiffer where its nodes are heavy.
   *Measure:* the span and the cabin's elastic energy at t_m.
8. **The engine's place** (rear 0.25 m from the firewall, front 0.58 m behind the nose):
   checked only after 1–3; the Camry's own clearances were not measured here.

## 7. Not verified

- The formula coefficients marked "to be verified" in section-matching.md §1 (9.56 against 9.53,
  0.33 V/C, the stroke use of 0.73, the box collapse estimate in place of Kecman's formula). No
  top-hat formula is implemented.
- The push is one run at 1 m/s with a ballasted cabin (inertia of the front small but not zero,
  the tyres on the ground); S11 here is run 0 only, edge contact, not the 8 runs a decision needs.
- The cut's capacities assume every crossing beam shortens together: an upper bound.
- The member-by-member bay sums use the cut at one plane per member and assign a brace to the
  member whose nodes it joins; a brace crossing two bays counts in both.
- The stated sections are the generator's (the study's starting values), not the Camry's; the
  Camry rail (92 × 166, 1.69–2.65 mm) is larger than ours (70 × 110, 1.8 mm).

## 8. Fixes applied (2026-11-27, `crash-ref/front-fixes`)

The six changes of §6 (1–6; 7 and 8 were not changes) are rules of the generator, each a switch
of the design's physics block (`Physics::front_rules`; derivations, confidence and findings in
[generated-vehicles-lessons.md](generated-vehicles-lessons.md) §25). **The car before is kept**:
with every switch off the generated files are byte for byte those of commit `33b57f525` (8
designs, 465 files), kept in `~/Development/rubble-artifacts/sections/gen-before-33b57f525/`
(`SHA256SUMS`, every file). SHA-256 of each design's JBeam set (its `.jbeam` files and `.pc`,
concatenated in name order) at that commit:

| design | JBeam set SHA-256 |
|---|---|
| sedan | 5932630e77a6d90bdd949b9423e21f284e0ea2d7d7a963f6e72f5f44a2423533 |
| police_sedan | 505b05d23324ad189db4136390880504df71249517bc977cefad9e4edba9a631 |
| wagon | 3ff174b71bede14697cbd3b304d68730661fdc90073048d7a47bad0a9cb4e069 |
| mach1_1969 | a8f56cf838f8859e3006314eb38324c19ceb097eb9b0ef3bf9028f5b775f36f0 |
| mach1_gasser | ae0dc5008a1697551d8826a2436e07c9cad5acbc53c03fd2f513bbb20f15b37c |
| suv_4x4 | 57b3323a5d7314bc5fa85051dec8d38e768c61ccd09d219f36789519ece837e1 |
| truck_cab | c75ad80119930bf70cd172135634acd857d8775e0d0b13b834c0b2507e970525 |
| box_trailer | 5c827e032edf0e9bd532d60426b4459aae246d4edc0885c0b1d157c3d0117a61 |

Reproduce (from `engine/`): `tools/sections/front_ab_all.sh` (every variant below into
`out/gen/sedan_<tag>/`), `python3 -I tools/sections/front_ab.py TAG ...` (the tables, `--plot`),
`tools/sections/front_hard_rules.sh TAG rule ...` and `front_hard_rules.py` (8 runs × 3 contact
settings), `front_spans.py`, `front_gates.sh`, `front_files.sh`. Plots in `out/sections/fixes/`,
copies in `~/Development/rubble-artifacts/sections/fixes/`.

Tags: `off` every rule off (the car before); `r1`–`r5` one rule alone (`r6` changes nothing in
`beamng_feel`: byte for byte the car before); `c12`, `c123`, … the rules cumulative in §6's
order; `c145`/`c1456` rules 1, 4, 5 (and 6); `all`; `rc_*` the `real_crash` character. CPU
reference, generated sedan 1,448 kg, S11 and S12 run 0 with edge contact, the push of §2. The
cut rows are the lattice cut of §4 (both sides): yielding beams at their threshold, never-yielding
ones at their strength.

### 8.1 Each rule alone

| | off | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|---|
| cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN) | 1010 / 1606 | 1007 / 1606 | 804 / 1606 | 1010 / 1606 | 1077 / 229 | 701 / 1600 |
|   of which crash boxes / box–other braces / hood + fenders (kN) | 194 / 224 / 401 | 194 / 224 / 398 | 194 / 18 / 401 | 194 / 224 / 401 | 194 / 224 / 469 | 194 / 224 / 92 |
| cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN) | 2236 / 1334 | 1890 / 1334 | 2126 / 1334 | 2236 / 1334 | 2291 / 129 | 1959 / 1035 |
|   of which rail chords and braces / subframe–crash box (kN) | 348 / 553 | 348 / 0 | 290 / 553 | 348 / 553 | 348 / 553 | 348 / 553 |
| push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 162 / 394 / 538 / 509 | 165 / 362 / 523 / 531 | 141 / 254 / 235 / 375 | 162 / 394 / 514 / 518 | 127 / 396 / 517 / 414 | 162 / 379 / 515 / 341 |
| push at 0.39 m: crash-box chord / rail chords (m) | 0.128 / 0.371, 0.434 | 0.157 / 0.368, 0.386 | 0.193 / 0.373, 0.449 | 0.115 / 0.372, 0.438 | 0.131 / 0.370, 0.438 | 0.119 / 0.373, 0.430 |
| push at 0.39 m: elastic / plastic (kJ) | 40.5 / 78.0 | 41.2 / 77.4 | 27.4 / 35.3 | 41.4 / 75.4 | 34.0 / 79.5 | 28.8 / 81.9 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 255 / 538 / 644 / – | 253 / 519 / 706 / – | 243 / 470 / 471 / 419 | 255 / 535 / 623 / – | 229 / 539 / 645 / 406 | 240 / 516 / 580 / 529 |
| S11 t_m (ms) | 44.4 | 44.1 | 52.9 | 44.1 | 49.6 | 48.6 |
| S11 dynamic crush (m) | 0.394 | 0.398 | 0.471 | 0.397 | 0.406 | 0.415 |
| S11 rebound (m/s) / restitution | 5.34 / 0.342 | 5.02 / 0.322 | 4.77 / 0.306 | 5.54 / 0.355 | 4.40 / 0.282 | 3.85 / 0.247 |
| S11 residual crush (m) | 0.199 | 0.210 | 0.288 | 0.196 | 0.239 | 0.252 |
| S11 peak wall force (kN) at (ms) | 661 at 20.0 | 721 at 24.0 | 549 at 16.0 | 644 at 20.0 | 672 at 20.0 | 620 at 16.0 |
| S11 4 ms / CFC 60 peak (g) | 50.0 / 59.1 | 51.6 / 57.4 | 44.6 / 53.7 | 47.9 / 59.0 | 55.1 / 60.9 | 48.5 / 60.8 |
| S11 cabin: door opening change peak / residual (mm) | 151 / -24 | 170 / -32 | 166 / -27 | 150 / -26 | 151 / -24 | 97 / 0 |
| S11 plastic work: cabin share | 0.061 | 0.087 | 0.107 | 0.062 | 0.058 | 0.015 |
| S11 elastic at the stop / 0.3 s later (kJ) | 47.3 / 9.7 | 42.6 / 7.4 | 45.0 / 11.4 | 50.2 / 9.0 | 35.3 / 7.6 | 30.1 / 7.7 |
| S11 released by zone (kJ) | cabin 16.6, front lower 8.2, front lower+upper 5.8, front upper 4.9, other 2.1 | cabin 18.6, front lower+upper 6.4, front lower 5.3, front upper 4.2, other 0.8 | cabin 18.1, front lower 5.8, front upper 4.8, front lower+upper 3.4, other 1.5 | cabin 17.1, front lower 9.6, front lower+upper 6.7, front upper 5.5, other 2.3 | cabin 14.8, front upper 4.3, front lower+upper 4.2, front lower 2.9, other 1.5 | cabin 8.8, front lower+upper 6.4, front lower 3.9, front upper 2.8, other 0.6 |
| S11 engine rear to firewall: least (m) at (ms); within 3 cm | 0.227 at 166; None | 0.228 at 156; None | 0.250 at -94; None | 0.227 at 180; None | 0.228 at 173; None | 0.244 at 266; None |
| S12 t_m (ms) / crush (m) / restitution / residual (m) | 39.5 / 0.275 / 0.484 / 0.099 | 41.0 / 0.284 / 0.489 / 0.104 | 60.8 / 0.348 / 0.330 / 0.172 | 39.9 / 0.276 / 0.466 / 0.104 | 39.6 / 0.281 / 0.445 / 0.091 | 40.7 / 0.283 / 0.452 / 0.111 |
| S12 peak wall force (kN) at (ms) | 576 at 32.0 | 546 at 32.0 | 449 at 24.0 | 562 at 32.0 | 593 at 32.0 | 538 at 32.0 |
| hard: energy gain S11 / S12 (J); into the wall (m) | 0.0 / 1.4; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 1.4; 0.000 | 0.0 / 1.4; 0.000 | 0.0 / 1.4; 0.000 | 0.0 / 0.0; 0.000 |

![each rule alone](../out/sections/fixes/front_ab_off_r1_r2_r3_r4_r5.png)

### 8.2 Cumulative, in the audit's order

| | off | r1 | c12 | c123 | c1234 | c12345 | all |
|---|---|---|---|---|---|---|---|
| cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN) | 1010 / 1606 | 1007 / 1606 | 801 / 1606 | 801 / 1606 | 868 / 229 | 552 / 229 | 552 / 229 |
|   of which crash boxes / box–other braces / hood + fenders (kN) | 194 / 224 / 401 | 194 / 224 / 398 | 194 / 18 / 398 | 194 / 18 / 398 | 194 / 18 / 465 | 194 / 18 / 148 | 194 / 18 / 148 |
| cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN) | 2236 / 1334 | 1890 / 1334 | 1780 / 1334 | 1780 / 1334 | 1835 / 129 | 1542 / 129 | 1542 / 129 |
|   of which rail chords and braces / subframe–crash box (kN) | 348 / 553 | 348 / 0 | 290 / 0 | 290 / 0 | 290 / 0 | 290 / 0 | 290 / 0 |
| push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 162 / 394 / 538 / 509 | 165 / 362 / 523 / 531 | 144 / 242 / 234 / 376 | 144 / 242 / 234 / 377 | 118 / 250 / 212 / 380 | 115 / 251 / 143 / 374 | 115 / 251 / 143 / 374 |
| push at 0.39 m: crash-box chord / rail chords (m) | 0.128 / 0.371, 0.434 | 0.157 / 0.368, 0.386 | 0.193 / 0.375, 0.450 | 0.193 / 0.375, 0.450 | 0.187 / 0.373, 0.449 | 0.188 / 0.371, 0.447 | 0.188 / 0.371, 0.447 |
| push at 0.39 m: elastic / plastic (kJ) | 40.5 / 78.0 | 41.2 / 77.4 | 27.1 / 34.8 | 27.1 / 34.8 | 23.4 / 35.4 | 17.9 / 31.0 | 17.9 / 31.0 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 255 / 538 / 644 / – | 253 / 519 / 706 / – | 243 / 463 / 505 / 355 | 243 / 461 / 498 / 375 | 219 / 457 / 524 / 369 | 208 / 433 / 454 / 365 | 208 / 433 / 454 / 365 |
| S11 t_m (ms) | 44.4 | 44.1 | 53.5 | 53.4 | 52.9 | 54.0 | 54.0 |
| S11 dynamic crush (m) | 0.394 | 0.398 | 0.475 | 0.473 | 0.473 | 0.492 | 0.492 |
| S11 rebound (m/s) / restitution | 5.34 / 0.342 | 5.02 / 0.322 | 4.69 / 0.300 | 4.61 / 0.295 | 4.68 / 0.299 | 4.35 / 0.278 | 4.35 / 0.278 |
| S11 residual crush (m) | 0.199 | 0.210 | 0.291 | 0.286 | 0.292 | 0.344 | 0.344 |
| S11 peak wall force (kN) at (ms) | 661 at 20.0 | 721 at 24.0 | 562 at 44.0 | 556 at 44.0 | 558 at 44.0 | 560 at 44.0 | 560 at 44.0 |
| S11 4 ms / CFC 60 peak (g) | 50.0 / 59.1 | 51.6 / 57.4 | 43.4 / 50.9 | 44.5 / 50.8 | 44.8 / 51.8 | 46.7 / 54.0 | 46.7 / 54.0 |
| S11 cabin: door opening change peak / residual (mm) | 151 / -24 | 170 / -32 | 159 / -18 | 161 / -19 | 160 / -21 | 106 / -1 | 106 / -1 |
| S11 plastic work: cabin share | 0.061 | 0.087 | 0.090 | 0.092 | 0.087 | 0.024 | 0.024 |
| S11 elastic at the stop / 0.3 s later (kJ) | 47.3 / 9.7 | 42.6 / 7.4 | 41.0 / 10.4 | 41.1 / 10.0 | 38.9 / 7.6 | 30.1 / 5.7 | 30.1 / 5.7 |
| S11 released by zone (kJ) | cabin 16.6, front lower 8.2, front lower+upper 5.8, front upper 4.9, other 2.1 | cabin 18.6, front lower+upper 6.4, front lower 5.3, front upper 4.2, other 0.8 | cabin 17.2, front lower 5.0, front upper 3.9, front lower+upper 3.1, other 1.3 | cabin 17.7, front lower 5.3, front upper 4.0, front lower+upper 2.8, other 1.3 | cabin 17.0, front lower 6.1, front upper 4.2, front lower+upper 2.2, other 1.8 | cabin 11.0, front lower 7.1, other 2.6, front lower+upper 2.3, front upper 1.5 | cabin 11.0, front lower 7.1, other 2.6, front lower+upper 2.3, front upper 1.5 |
| S11 engine rear to firewall: least (m) at (ms); within 3 cm | 0.227 at 166; None | 0.228 at 156; None | 0.250 at -94; None | 0.250 at -94; None | 0.247 at 79; None | 0.226 at 78; None | 0.226 at 78; None |
| S12 t_m (ms) / crush (m) / restitution / residual (m) | 39.5 / 0.275 / 0.484 / 0.099 | 41.0 / 0.284 / 0.489 / 0.104 | 60.6 / 0.347 / 0.342 / 0.149 | 60.2 / 0.347 / 0.344 / 0.166 | 60.1 / 0.350 / 0.347 / 0.180 | 68.5 / 0.378 / 0.265 / 0.232 | 68.5 / 0.378 / 0.265 / 0.232 |
| S12 peak wall force (kN) at (ms) | 576 at 32.0 | 546 at 32.0 | 449 at 24.0 | 444 at 24.0 | 467 at 24.0 | 431 at 24.0 | 431 at 24.0 |
| hard: energy gain S11 / S12 (J); into the wall (m) | 0.0 / 1.4; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.005 | 0.0 / 0.0; 0.002 | 0.0 / 0.0; 0.001 | 0.0 / 0.0; 0.001 |

![cumulative](../out/sections/fixes/front_ab_off_r1_c12_c123_c1234_c12345_all.png)

### 8.3 `real_crash`, rule 6, and the rules that hold together

| | rc_off | rc_r6 | rc_all |
|---|---|---|---|
| cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN) | 627 / 1606 | 676 / 1606 | 447 / 229 |
|   of which crash boxes / box–other braces / hood + fenders (kN) | 108 / 125 / 204 | 157 / 125 / 204 | 157 / 16 / 84 |
| cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN) | 1756 / 1334 | 1823 / 1334 | 1308 / 129 |
|   of which rail chords and braces / subframe–crash box (kN) | 193 / 553 | 260 / 553 | 229 / 0 |
| push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 157 / 299 / 330 / 301 | 157 / 313 / 305 / 312 | 106 / 138 / 113 / 261 |
| push at 0.39 m: crash-box chord / rail chords (m) | 0.160 / 0.377, 0.435 | 0.163 / 0.376, 0.431 | 0.185 / 0.375, 0.449 |
| push at 0.39 m: elastic / plastic (kJ) | 25.8 / 60.3 | 25.8 / 60.3 | 10.9 / 28.0 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 245 / 436 / 477 / 501 | 245 / 454 / 497 / 482 | 194 / 372 / 380 / 380 |
| S11 t_m (ms) | 57.3 | 57.5 | 61.4 |
| S11 dynamic crush (m) | 0.480 | 0.477 | 0.553 |
| S11 rebound (m/s) / restitution | 3.67 / 0.235 | 3.51 / 0.225 | 3.16 / 0.203 |
| S11 residual crush (m) | 0.333 | 0.324 | 0.421 |
| S11 peak wall force (kN) at (ms) | 508 at 32.0 | 519 at 16.0 | 431 at 16.0 |
| S11 4 ms / CFC 60 peak (g) | 39.4 / 41.4 | 40.0 / 43.2 | 35.4 / 42.0 |
| S11 cabin: door opening change peak / residual (mm) | 131 / -23 | 135 / -26 | 74 / -4 |
| S11 plastic work: cabin share | 0.031 | 0.035 | 0.007 |
| S11 elastic at the stop / 0.3 s later (kJ) | 28.0 / 7.3 | 29.0 / 6.9 | 21.9 / 4.0 |
| S11 released by zone (kJ) | cabin 11.0, front lower+upper 3.8, front upper 3.4, front lower 1.7, other 1.0 | cabin 11.4, front lower+upper 4.3, front upper 3.8, front lower 1.8, other 0.8 | cabin 7.0, front lower 4.1, other 3.7, front lower+upper 2.5, front upper 0.7 |
| S11 engine rear to firewall: least (m) at (ms); within 3 cm | 0.234 at 77; None | 0.241 at 85; None | 0.224 at 55; None |
| S12 t_m (ms) / crush (m) / restitution / residual (m) | 50.3 / 0.317 / 0.366 / 0.156 | 48.7 / 0.311 / 0.345 / 0.159 | 77.5 / 0.430 / 0.244 / 0.311 |
| S12 peak wall force (kN) at (ms) | 427 at 24.0 | 449 at 24.0 | 351 at 24.0 |
| hard: energy gain S11 / S12 (J); into the wall (m) | 0.0 / 1.3; 0.000 | 0.0 / 1.3; 0.000 | 0.0 / 0.0; 0.000 |

| | off | c145 | rc_off | rc_c1456 |
|---|---|---|---|---|
| cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN) | 1010 / 1606 | 757 / 229 | 627 / 1606 | 556 / 229 |
|   of which crash boxes / box–other braces / hood + fenders (kN) | 194 / 224 / 401 | 194 / 224 / 148 | 108 / 125 / 204 | 157 / 125 / 84 |
| cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN) | 2236 / 1334 | 1652 / 129 | 1756 / 1334 | 1366 / 129 |
|   of which rail chords and braces / subframe–crash box (kN) | 348 / 553 | 348 / 0 | 193 / 553 | 260 / 0 |
| push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 162 / 394 / 538 / 509 | 123 / 363 / 468 / 527 | 157 / 299 / 330 / 301 | 116 / 262 / 309 / 372 |
| push at 0.39 m: crash-box chord / rail chords (m) | 0.128 / 0.371, 0.434 | 0.150 / 0.367, 0.384 | 0.160 / 0.377, 0.435 | 0.174 / 0.370, 0.375 |
| push at 0.39 m: elastic / plastic (kJ) | 40.5 / 78.0 | 30.0 / 79.0 | 25.8 / 60.3 | 17.1 / 59.3 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 255 / 538 / 644 / – | 214 / 497 / 602 / 502 | 245 / 436 / 477 / 501 | 200 / 410 / 471 / 420 |
| S11 t_m (ms) | 44.4 | 48.6 | 57.3 | 56.6 |
| S11 dynamic crush (m) | 0.394 | 0.428 | 0.480 | 0.507 |
| S11 rebound (m/s) / restitution | 5.34 / 0.342 | 4.24 / 0.271 | 3.67 / 0.235 | 3.84 / 0.246 |
| S11 residual crush (m) | 0.199 | 0.274 | 0.333 | 0.347 |
| S11 peak wall force (kN) at (ms) | 661 at 20.0 | 622 at 20.0 | 508 at 32.0 | 483 at 16.0 |
| S11 4 ms / CFC 60 peak (g) | 50.0 / 59.1 | 47.3 / 58.1 | 39.4 / 41.4 | 37.4 / 42.3 |
| S11 cabin: door opening change peak / residual (mm) | 151 / -24 | 115 / -4 | 131 / -23 | 81 / -7 |
| S11 plastic work: cabin share | 0.061 | 0.019 | 0.031 | 0.009 |
| S11 elastic at the stop / 0.3 s later (kJ) | 47.3 / 9.7 | 29.8 / 7.4 | 28.0 / 7.3 | 22.7 / 5.1 |
| S11 released by zone (kJ) | cabin 16.6, front lower 8.2, front lower+upper 5.8, front upper 4.9, other 2.1 | cabin 10.2, front lower 5.3, front lower+upper 4.1, front upper 1.9, other 0.8 | cabin 11.0, front lower+upper 3.8, front upper 3.4, front lower 1.7, other 1.0 | cabin 7.0, front lower+upper 4.2, front lower 4.1, other 1.2, front upper 1.1 |
| S11 engine rear to firewall: least (m) at (ms); within 3 cm | 0.227 at 166; None | 0.250 at -94; None | 0.234 at 77; None | 0.223 at 181; None |
| S12 t_m (ms) / crush (m) / restitution / residual (m) | 39.5 / 0.275 / 0.484 / 0.099 | 42.0 / 0.294 / 0.442 / 0.124 | 50.3 / 0.317 / 0.366 / 0.156 | 53.0 / 0.344 / 0.326 / 0.210 |
| S12 peak wall force (kN) at (ms) | 576 at 32.0 | 530 at 24.0 | 427 at 24.0 | 405 at 32.0 |
| hard: energy gain S11 / S12 (J); into the wall (m) | 0.0 / 1.4; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 1.3; 0.000 | 3.8 / 0.0; 0.000 |

### 8.4 Where the push's 0.39 m goes (z shortening of each bay, m)

| variant | crash box a1–a2: z shortening (m) | chords' length upper / lower (m, from 0.180) | rail a2–a4 (z) | rail a4–a5 (z) | a5 to firewall (z) | firewall to cabin rear | crash box front's rise (m) |
|---|---|---|---|---|---|---|---|
| off | 0.100 | 0.128 / 0.111 | 0.029 | 0.023 | -0.023 | 0.109 | -0.017 |
| r1 | 0.054 | 0.157 / 0.126 | 0.020 | 0.081 | -0.010 | 0.093 | -0.018 |
| r2 | 0.251 | 0.193 / 0.182 | 0.024 | 0.007 | -0.021 | 0.047 | 0.041 |
| r2a | 0.172 | 0.137 / 0.124 | 0.033 | 0.009 | -0.021 | 0.065 | 0.012 |
| r3 | 0.106 | 0.115 / 0.097 | 0.025 | 0.019 | -0.022 | 0.110 | -0.017 |
| r4 | 0.107 | 0.131 / 0.123 | 0.030 | 0.019 | -0.024 | 0.105 | 0.004 |
| r5 | 0.134 | 0.119 / 0.118 | 0.029 | 0.034 | 0.004 | 0.036 | -0.018 |
| c12 | 0.244 | 0.193 / 0.182 | 0.028 | 0.006 | -0.019 | 0.044 | 0.046 |
| c123 | 0.244 | 0.193 / 0.182 | 0.028 | 0.006 | -0.020 | 0.044 | 0.047 |
| c1234 | 0.229 | 0.187 / 0.182 | 0.022 | 0.008 | -0.020 | 0.048 | 0.026 |
| c12345 | 0.236 | 0.188 / 0.184 | 0.026 | 0.008 | -0.013 | 0.037 | 0.008 |
| all | 0.236 | 0.188 / 0.184 | 0.026 | 0.008 | -0.013 | 0.037 | 0.008 |
| rc_off | 0.160 | 0.161 / 0.155 | 0.044 | 0.022 | -0.013 | 0.052 | -0.000 |
| rc_r6 | 0.163 | 0.163 / 0.149 | 0.043 | 0.025 | -0.012 | 0.051 | 0.002 |
| rc_all | 0.216 | 0.185 / 0.184 | 0.039 | 0.005 | -0.006 | 0.021 | -0.001 |

The crash box's span along the car shortens by more than its chords: **it swings about its rear
nodes** (its lattice cell is four nodes in one vertical plane, no depth across the car), in the
car before too (its span reaches 0 at 0.55 m of push) and from 0.25 m once its braces are
weakened (rule 2). `r2a`: rule 2 on the braces along the car only (`FRONT_RULE2_ALONG=0.7`, a
measurement): the swing comes at 0.45 m, not never. Lessons §25.2.

### 8.5 Hard rules, 8 runs in each contact setting

| variant | scenario | contact | runs | energy created, max (J) | its band | into the wall, max (m) | finite | crossings, max | asserted failures | restitution | dyn. crush (m) | residual (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.302–0.346 | 0.385–0.403 | 0.188–0.221 |
| off | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.302–0.346 | 0.385–0.403 | 0.188–0.221 |
| off | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.295–0.356 | 0.381–0.403 | 0.190–0.228 |
| off | S12 | discrete | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.466–0.497 | 0.270–0.280 | 0.092–0.101 |
| off | S12 | edge | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.466–0.497 | 0.270–0.280 | 0.092–0.101 |
| off | S12 | continuous | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.463–0.496 | 0.269–0.278 | 0.091–0.099 |
| off | S01 | discrete | 8 | 12.8 | ≤ 230.0 | 0.000 | 1 | – | none | 0.231–0.261 | 0.569–0.596 | 0.062–0.071 |
| off | S01 | edge | 8 | 12.8 | ≤ 230.0 | 0.000 | 1 | – | none | 0.231–0.261 | 0.569–0.596 | 0.062–0.071 |
| off | S01 | continuous | 8 | 90.2 | ≤ 230.0 | 0.000 | 1 | – | none | 0.242–0.284 | 0.560–0.585 | 0.060–0.067 |
| off | S02 | discrete | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.275–0.300 | 0.285–0.296 | -0.158–-0.090 |
| off | S02 | edge | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.275–0.300 | 0.285–0.296 | -0.158–-0.090 |
| off | S02 | continuous | 8 | 0.0 |  | 0.000 | 1 | – | none | 0.303–0.328 | 0.279–0.290 | -0.185–-0.131 |
| off | S08 | discrete | 8 | 153.6 | ≤ 340.0 | 0.000 | 1 | 169.0 | the common centre of mass stays still 2/8 | 0.055–0.132 | 0.773–0.819 | -0.011–0.056 |
| off | S08 | edge | 8 | 202.6 | ≤ 340.0 | 0.000 | 1 | 31.0 | momentum across the road 1/8, momentum along the road 1/8, the common centre of mass stays still 2/8 | 0.121–0.205 | 0.696–0.737 | -0.009–0.059 |
| off | S08 | continuous | 8 | 442.9 | ≤ 340.0 | 0.000 | 1 | 0.0 | momentum along the road 2/8, no energy created 1/8, the common centre of mass stays still 8/8 | 0.125–0.173 | 0.649–0.730 | -0.012–0.048 |
| r1 | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.314–0.328 | 0.391–0.407 | 0.201–0.216 |
| r1 | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.314–0.328 | 0.391–0.407 | 0.201–0.216 |
| r1 | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.331–0.348 | 0.385–0.400 | 0.202–0.218 |
| r1 | S12 | discrete | 8 | 3.4 | ≤ 89.0 | 0.000 | 1 | – | none | 0.478–0.496 | 0.278–0.288 | 0.098–0.109 |
| r1 | S12 | edge | 8 | 3.4 | ≤ 89.0 | 0.000 | 1 | – | none | 0.478–0.496 | 0.278–0.288 | 0.098–0.109 |
| r1 | S12 | continuous | 8 | 3.4 | ≤ 89.0 | 0.000 | 1 | – | none | 0.466–0.497 | 0.276–0.285 | 0.097–0.108 |
| r1 | S01 | discrete | 8 | 4.7 | ≤ 230.0 | 0.000 | 1 | – | none | 0.261–0.281 | 0.577–0.601 | 0.067–0.073 |
| r1 | S01 | edge | 8 | 4.7 | ≤ 230.0 | 0.000 | 1 | – | none | 0.261–0.281 | 0.577–0.601 | 0.067–0.073 |
| r1 | S01 | continuous | 8 | 56.5 | ≤ 230.0 | 0.000 | 1 | – | none | 0.285–0.298 | 0.571–0.589 | 0.064–0.070 |
| r1 | S02 | discrete | 8 | 0.0 |  | 0.000 | 1 | – | none | 0.278–0.294 | 0.275–0.288 | -0.151–-0.032 |
| r1 | S02 | edge | 8 | 0.0 |  | 0.000 | 1 | – | none | 0.278–0.294 | 0.275–0.288 | -0.151–-0.032 |
| r1 | S02 | continuous | 8 | 0.0 |  | 0.000 | 1 | – | none | 0.305–0.332 | 0.270–0.284 | -0.193–0.014 |
| r1 | S08 | discrete | 8 | 183.4 | ≤ 340.0 | 0.000 | 1 | 159.0 | the common centre of mass stays still 1/8 | 0.073–0.166 | 0.761–0.813 | 0.003–0.055 |
| r1 | S08 | edge | 8 | 121.7 | ≤ 340.0 | 0.000 | 1 | 31.0 | momentum across the road 2/8, momentum along the road 1/8, the common centre of mass stays still 4/8 | 0.148–0.208 | 0.700–0.741 | -0.022–0.055 |
| r1 | S08 | continuous | 8 | 282.8 | ≤ 340.0 | 0.000 | 1 | 0.0 | momentum along the road 3/8, the common centre of mass stays still 8/8, turns the same sense as the other 3/8 | 0.119–0.178 | 0.660–0.734 | -0.021–0.053 |
| r4 | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.271–0.298 | 0.394–0.417 | 0.224–0.253 |
| r4 | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.271–0.298 | 0.394–0.417 | 0.224–0.253 |
| r4 | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.269–0.301 | 0.390–0.411 | 0.223–0.259 |
| r4 | S12 | discrete | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.432–0.453 | 0.277–0.286 | 0.084–0.099 |
| r4 | S12 | edge | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.432–0.453 | 0.277–0.286 | 0.084–0.099 |
| r4 | S12 | continuous | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.432–0.450 | 0.276–0.285 | 0.084–0.097 |
| r4 | S01 | discrete | 8 | 0.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.258–0.277 | 0.574–0.601 | 0.085–0.093 |
| r4 | S01 | edge | 8 | 0.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.258–0.277 | 0.574–0.601 | 0.085–0.093 |
| r4 | S01 | continuous | 8 | 3.7 | ≤ 230.0 | 0.000 | 1 | – | none | 0.271–0.293 | 0.570–0.597 | 0.081–0.089 |
| r4 | S02 | discrete | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.235–0.285 | 0.298–0.310 | -0.126–-0.015 |
| r4 | S02 | edge | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.235–0.285 | 0.298–0.310 | -0.126–-0.015 |
| r4 | S02 | continuous | 8 | 0.0 |  | 0.001 | 1 | – | none | 0.282–0.302 | 0.293–0.305 | -0.149–-0.047 |
| r4 | S08 | discrete | 8 | 169.7 | ≤ 340.0 | 0.000 | 1 | 152.0 | the common centre of mass stays still 4/8 | 0.071–0.138 | 0.751–0.817 | -0.006–0.067 |
| r4 | S08 | edge | 8 | 272.4 | ≤ 340.0 | 0.000 | 1 | 78.0 | none | 0.143–0.202 | 0.737–0.777 | 0.003–0.037 |
| r4 | S08 | continuous | 8 | 126.8 | ≤ 340.0 | 0.000 | 1 | 0.0 | momentum along the road 1/8 | 0.164–0.231 | 0.713–0.751 | 0.013–0.051 |
| r5 | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 2/8 | 0.227–0.276 | 0.404–0.425 | 0.235–0.266 |
| r5 | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 2/8 | 0.227–0.276 | 0.404–0.425 | 0.235–0.266 |
| r5 | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.004 | 1 | – | restitution within the 56 km/h sedan fleet's 2/8 | 0.209–0.277 | 0.404–0.425 | 0.235–0.264 |
| r5 | S12 | discrete | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.444–0.461 | 0.278–0.288 | 0.106–0.117 |
| r5 | S12 | edge | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.444–0.461 | 0.278–0.288 | 0.106–0.117 |
| r5 | S12 | continuous | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.443–0.463 | 0.278–0.288 | 0.106–0.117 |
| r5 | S01 | discrete | 8 | 81.9 | ≤ 230.0 | 0.000 | 1 | – | none | 0.208–0.231 | 0.595–0.618 | 0.033–0.042 |
| r5 | S01 | edge | 8 | 81.9 | ≤ 230.0 | 0.000 | 1 | – | none | 0.208–0.231 | 0.595–0.618 | 0.033–0.042 |
| r5 | S01 | continuous | 8 | 77.1 | ≤ 230.0 | 0.000 | 1 | – | none | 0.216–0.225 | 0.593–0.617 | 0.031–0.039 |
| r5 | S02 | discrete | 8 | 8.2 |  | 0.002 | 1 | – | none | 0.219–0.255 | 0.304–0.316 | -0.009–0.138 |
| r5 | S02 | edge | 8 | 8.2 |  | 0.002 | 1 | – | none | 0.219–0.255 | 0.304–0.316 | -0.009–0.138 |
| r5 | S02 | continuous | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.233–0.275 | 0.304–0.317 | -0.015–0.053 |
| r5 | S08 | discrete | 8 | 233.5 | ≤ 340.0 | 0.000 | 1 | 200.0 | the common centre of mass stays still 3/8 | 0.057–0.113 | 0.838–0.898 | -0.050–0.003 |
| r5 | S08 | edge | 8 | 76.0 | ≤ 340.0 | 0.000 | 1 | 36.0 | the common centre of mass stays still 1/8 | 0.116–0.155 | 0.767–0.814 | -0.034–0.049 |
| r5 | S08 | continuous | 8 | 3.0 | ≤ 340.0 | 0.000 | 1 | 0.0 | turns the same sense as the other 1/8 | 0.101–0.161 | 0.764–0.802 | 0.001–0.040 |
| rc_off | S11 | discrete | 8 | 717.1 | ≤ 177.0 | 0.001 | 1 | – | no energy created 1/8 | 0.223–0.249 | 0.466–0.496 | 0.310–0.346 |
| rc_off | S11 | edge | 8 | 717.1 | ≤ 177.0 | 0.001 | 1 | – | no energy created 1/8 | 0.223–0.249 | 0.466–0.496 | 0.310–0.346 |
| rc_off | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 4/8 | 0.228–0.259 | 0.462–0.490 | 0.307–0.328 |
| rc_off | S12 | discrete | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.362–0.374 | 0.309–0.325 | 0.153–0.165 |
| rc_off | S12 | edge | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.362–0.374 | 0.309–0.325 | 0.153–0.165 |
| rc_off | S12 | continuous | 8 | 322.9 | ≤ 89.0 | 0.000 | 1 | – | no energy created 1/8 | 0.367–0.384 | 0.306–0.320 | 0.153–0.161 |
| rc_off | S01 | discrete | 8 | 9.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.170–0.183 | 0.639–0.670 | -0.018–0.069 |
| rc_off | S01 | edge | 8 | 9.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.170–0.183 | 0.639–0.670 | -0.018–0.069 |
| rc_off | S01 | continuous | 8 | 11.5 | ≤ 230.0 | 0.000 | 1 | – | none | 0.190–0.206 | 0.631–0.658 | 0.060–0.072 |
| rc_off | S02 | discrete | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.205–0.244 | 0.340–0.360 | -0.057–0.221 |
| rc_off | S02 | edge | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.205–0.244 | 0.340–0.360 | -0.057–0.221 |
| rc_off | S02 | continuous | 8 | 907.6 |  | 0.003 | 1 | – | none | 0.236–0.255 | 0.333–0.356 | -0.043–0.163 |
| rc_off | S08 | discrete | 8 | 387.0 | ≤ 340.0 | 0.000 | 1 | 152.0 | momentum along the road 1/8, no energy created 2/8, the common centre of mass stays still 2/8 | 0.108–0.143 | 0.848–0.901 | -0.050–0.077 |
| rc_off | S08 | edge | 8 | 0.0 | ≤ 340.0 | 0.000 | 1 | 53.0 | none | 0.107–0.136 | 0.823–0.898 | -0.067–-0.039 |
| rc_off | S08 | continuous | 8 | 105.3 | ≤ 340.0 | 0.000 | 1 | 0.0 | momentum across the road 2/8, momentum along the road 1/8, the common centre of mass stays still 4/8, turns the same sense as the other 5/8 | 0.093–0.149 | 0.794–0.850 | -0.059–0.053 |
| rc_r6 | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 1/8 | 0.224–0.255 | 0.465–0.490 | 0.307–0.336 |
| rc_r6 | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 1/8 | 0.224–0.255 | 0.465–0.490 | 0.307–0.336 |
| rc_r6 | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.001 | 1 | – | restitution within the 56 km/h sedan fleet's 2/8 | 0.237–0.255 | 0.461–0.486 | 0.306–0.327 |
| rc_r6 | S12 | discrete | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.337–0.370 | 0.303–0.318 | 0.149–0.167 |
| rc_r6 | S12 | edge | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.337–0.370 | 0.303–0.318 | 0.149–0.167 |
| rc_r6 | S12 | continuous | 8 | 2.8 | ≤ 89.0 | 0.000 | 1 | – | none | 0.358–0.387 | 0.300–0.314 | 0.144–0.162 |
| rc_r6 | S01 | discrete | 8 | 3.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.171–0.190 | 0.635–0.665 | 0.060–0.067 |
| rc_r6 | S01 | edge | 8 | 3.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.171–0.190 | 0.635–0.665 | 0.060–0.067 |
| rc_r6 | S01 | continuous | 8 | 14.9 | ≤ 230.0 | 0.000 | 1 | – | none | 0.190–0.204 | 0.625–0.652 | 0.057–0.072 |
| rc_r6 | S02 | discrete | 8 | 659.4 |  | 0.003 | 1 | – | none | 0.211–0.249 | 0.342–0.359 | -0.066–0.215 |
| rc_r6 | S02 | edge | 8 | 659.4 |  | 0.003 | 1 | – | none | 0.211–0.249 | 0.342–0.359 | -0.066–0.215 |
| rc_r6 | S02 | continuous | 8 | 7.7 |  | 0.002 | 1 | – | none | 0.229–0.258 | 0.333–0.353 | -0.097–0.227 |
| rc_r6 | S08 | discrete | 8 | 96.4 | ≤ 340.0 | 0.000 | 1 | 178.0 | the common centre of mass stays still 4/8 | 0.092–0.131 | 0.857–0.907 | -0.058–0.076 |
| rc_r6 | S08 | edge | 8 | 1.4 | ≤ 340.0 | 0.000 | 1 | 32.0 | momentum across the road 1/8, momentum along the road 1/8, the common centre of mass stays still 1/8, turns the same sense as the other 4/8 | 0.110–0.145 | 0.820–0.894 | -0.069–-0.051 |
| rc_r6 | S08 | continuous | 8 | 290.2 | ≤ 340.0 | 0.000 | 1 | 0.0 | momentum across the road 1/8, the common centre of mass stays still 7/8, turns the same sense as the other 7/8 | 0.079–0.145 | 0.798–0.858 | -0.054–0.068 |
| all | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.001 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.278–0.283 | 0.485–0.500 | 0.333–0.349 |
| all | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.001 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.278–0.283 | 0.485–0.500 | 0.333–0.349 |
| all | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.002 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.276–0.285 | 0.485–0.499 | 0.336–0.347 |
| all | S12 | discrete | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.249–0.285 | 0.369–0.388 | 0.227–0.243 |
| all | S12 | edge | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.249–0.285 | 0.369–0.388 | 0.227–0.243 |
| all | S12 | continuous | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.222–0.287 | 0.369–0.387 | 0.226–0.243 |
| all | S01 | discrete | 8 | 2.7 | ≤ 230.0 | 0.000 | 1 | – | none | 0.174–0.204 | 0.637–0.659 | -0.025–0.067 |
| all | S01 | edge | 8 | 2.7 | ≤ 230.0 | 0.000 | 1 | – | none | 0.174–0.204 | 0.637–0.659 | -0.025–0.067 |
| all | S01 | continuous | 8 | 0.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.181–0.223 | 0.634–0.657 | 0.062–0.067 |
| all | S02 | discrete | 8 | 4.9 |  | 0.002 | 1 | – | none | 0.209–0.221 | 0.394–0.402 | -0.016–0.134 |
| all | S02 | edge | 8 | 4.9 |  | 0.002 | 1 | – | none | 0.209–0.221 | 0.394–0.402 | -0.016–0.134 |
| all | S02 | continuous | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.210–0.232 | 0.393–0.402 | 0.054–0.193 |
| all | S08 | discrete | 8 | 219.8 | ≤ 340.0 | 0.000 | 1 | 199.0 | none | 0.059–0.137 | 0.860–0.910 | -0.053–-0.006 |
| all | S08 | edge | 8 | 466.5 | ≤ 340.0 | 0.000 | 1 | 40.0 | no energy created 1/8 | 0.114–0.168 | 0.888–0.926 | -0.011–0.033 |
| all | S08 | continuous | 8 | 30.4 | ≤ 340.0 | 0.000 | 1 | 0.0 | none | 0.102–0.140 | 0.879–0.930 | -0.009–0.011 |

S08 with edge contact, energy created in each of the 8 runs (J; the bar 340 J): before 0–203
(holds); rule 1 alone 4–122; rule 2 alone 10–2,818 (**5 of 8 fail**); rule 3 alone 0–674 (**1
of 8 fails**); rule 4 alone 1–272; rule 5 alone 0–76; 1+2 16–1,415 (5 of 8); 1+2+3 585–2,262 (8
of 8); 1–4 0–2,686 (3 of 8); 1–5 and all six 0–467 (1 of 8). It is the edge contact's known
position step (crash-reference.md, S03/S14), which the swinging crash boxes' edges reach.

### 8.6 Gates and tests whose result changed

Every gate on the car before (`RUBBLE_FRONT_RULES=none`) and with each set (`front_gates.sh`;
`gen3_rig` with `GEN3_RIG_DESIGN=sedan`). Unchanged everywhere: `engine-vehiclegen` (lib 63,
`crash_zones`, `mechanical`), authoring `crash_zones`, `truck_cab`, `box_trailer` (their files
byte for byte with every rule), `sedan_energy` 6/6, `crash_invariants` 9/9, `gen_sedan_contact`
(the six listed failures), `continuous_contact` (the three listed), `generated_vehicle_lint` (the
two listed), `gen3_rig` (`rest_fidelity_at_spawn_and_parked` fails before and with every rule on
the generated sedan: parked normal 1.090° before, 1.089° with rule 5, 1.102° with all six; spawn
0.05° → 0.08°, vertex 0.105 mm; bar 1°, not in the expected-failures list because CI runs it on
the gen2 car).

| test (bar) | before | changes with |
|---|---|---|
| `every_design` (every part ≤ 5 mm parked) | passes; 4x4 `bf_g2` 4.9 mm | rule 1: **fails**, 5.5 mm (rules 2–5 alone 4.7–4.9; rules 1+4+5+6 5.8, all six 5.8) |
| `gen_sedan_paint::accept_head_30_pressed` (10 mm from 0.3 s) | passes, 8 mm | rule 5: **fails**, 21 mm |
| `gen_sedan_paint::accept_head_60_pressed` (50 mm) | passes, 21 mm | rule 4: **fails**, 53 mm; rule 5 18 mm; rules 1+4+5+6: fails |
| `gen_sedan_paint::accept_side_90` (50 mm; listed failing) | fails, 55 mm | rule 5: passes, 22 mm (would need removing from `tests-expected-failures.txt`) |
| `gen_sedan_paint::accept_side_60_pressed` (10 mm; listed failing) | fails, 16 mm | rule 4: passes, 2 mm; rule 5 17 mm |
| `gen_sedan_paint::accept_head_100` (listed failing) | fails, 61 mm | rule 4 117 mm; rule 5 54 mm |
| S11 restitution (cited ≤ 0.25), 8 runs × 3 settings | 0 of 24 pass | rule 4 0; rule 5 6 of 8 a setting; `real_crash` alone 8 of 8 (edge; it creates 717 J in one run), with rule 6 7 of 8 and no energy created |

Bands were not edited. The paint bars are the owner's (2026-11-25, independent targets, not
ratchets); the parked drift's 5 mm is the gate's own bar.

### 8.7 Shape (S02, the 4x4 into the wall; `crash_outlines` + `outlines.py`, peak crush)

| | real test | BeamNG | before | rules 1, 4, 5, 6 | all six |
|---|---|---|---|---|---|
| hood fold height (/ hood length) | 0.135 | 0.208 | 0.015 | 0.120 | 0.149 |
| hood length after / before | 0.833 | 0.832 | 1.001 | 0.935 | 0.879 |
| hood creases (turn > 20°) | 1 | 2 | 0 | 1 | 1 |

Stills of the collision surface (CPU, `crash_outlines S11`): before the hit, at the peak, at
maximum crush and at rest, rules off beside all six: `out/sections/fixes/stills_S11_side_off_vs_all.png`
and `stills_S11_top_off_vs_all.png` (copies in the artifacts). The pose before the hit is the
same; at rest the hood is tented and the fenders folded with the rules. The drawn car (GLB) is
byte for byte the same with every rule; the skins (`.rskin`) change with rule 5 only.

### 8.8 Against the audit's predictions

| audit §6 item | predicted | measured |
|---|---|---|
| 1 subframe mounts | rails fold; plateau ~500 → 200–300 kN; crush toward 0.6 m; t_m longer | rails' bay −346 kN, second rail chord −48 mm (was −19); **plateau unchanged** (520–530 kN), crush 0.394 → 0.398 m, t_m unchanged: not as predicted (the bay's other members hold it) |
| 2 bay alone | first plateau ≈ 2 × 70 kN + beam and cover, near 200–250 kN | 254 / 235 kN at 0.2 / 0.3 m, **but by the box swinging**, and S08 edge creates energy |
| 3 stroke | 0.13 m of travel at a flat force; less elastic energy | flat, but the chord still locks at `MIN_REST` (0.08 m, the 0.37 m chord not buildable: a deviation, lessons §25.3); elastic at the stop 47 → 50 kJ (not lower) |
| 4 stops yield | rebound −3–5 kJ (0.6 m/s); force −0–100 kN early | elastic at the stop −12 kJ, rebound 5.34 → 4.40 m/s; push −35 kN at 0.1 m; as predicted |
| 5 hood and fenders | 100–300 kN less in the cut; little in S11 | −309 kN in the cut; **much in S11** (restitution 0.342 → 0.247, cabin's plastic share 6 % → 1.5 %) |
| 6 crash rate | boxes 97 → 76 kN a side (with 2), rails 111 → 92 kN | `real_crash`: boxes 54 → 79 kN a side (with diagonals), the rails' chords and braces 193 → 260 kN (both sides, cut); in `beamng_feel` nothing (byte for byte) |
| 7 cabin carries the force elastically | fixed by 1–3 | the cabin's released energy 16.6 → 8.8 kJ with rule 5 alone, 11.0 kJ with all six; firewall-to-cabin span 0.109 → 0.036 m (rule 5) |
| (§2) engine stacks | second peak near the Camry's depth | **the engine never reaches the firewall** in any variant (0.22–0.25 m left) |

**Defaults: every rule stays off** (lessons §25.8: each breaks a hard rule or a gate that held
before). Rules 4 and 5 together come nearest (S11 restitution 0.271, run 0); rule 5 is the
single largest improvement against the real tests.

## 9. The crash box as a closed cell, and the six rules again (2026-11-27, `crash-ref/crash-box-cell`)

The owner's decision on §8: rebuild the crash box's lattice cell as a box (a node at each corner
of its section at both ends, every face braced), then re-measure. The rule, its derivation and
the stroke options: [generated-vehicles-lessons.md](generated-vehicles-lessons.md) §26. Switch
`crash_box_cell`; off, every design's files are byte for byte those of `33b57f525` (the kept
`gen-before-33b57f525`). With it on: every design's `.glb` and `crash.json` byte for byte the same,
`truck_cab` and `box_trailer` untouched; on the six car designs the body, closures', engine's,
impact beam's (and on the Mach 1s and the 4x4 the subframe's and windscreen's) JBeam and the
`.rskin`/`.lite.rskin` change (`out/gen-rules/changed_cell.txt`). Tags as §8 with `c` for the
cell: `cell` the cell alone, `cr1`…`cr5` the cell with one rule, `cc12`… cumulative, `call` all
seven; `cellshear` the cell with its long faces' diagonals at the walls' shear yield
(`CELL_DIAG_SHEAR=1`, a measurement).

### 9.1 Does the box fold? The slow push: the box's span along the car / its two outboard chords (m)

| variant | 0.1 m: span / chords (m) | 0.2 m: span / chords (m) | 0.3 m: span / chords (m) | 0.4 m: span / chords (m) | 0.5 m: span / chords (m) | 0.6 m: span / chords (m) | 0.7 m: span / chords (m) |
|---|---|---|---|---|---|---|---|
| off | 0.179 / 0.177, 0.179 | 0.173 / 0.173, 0.173 | 0.123 / 0.138, 0.125 | 0.078 / 0.129, 0.113 | 0.010 / 0.142, 0.147 | -0.000 / 0.143, 0.151 | -0.000 / 0.144, 0.151 |
| cell | 0.178 / 0.178, 0.178 | 0.169 / 0.173, 0.169 | 0.119 / 0.145, 0.126 | 0.055 / 0.129, 0.119 | 0.000 / 0.135, 0.125 | -0.000 / 0.137, 0.125 | -0.000 / 0.146, 0.141 |
| cellshear | 0.176 / 0.179, 0.176 | 0.167 / 0.169, 0.167 | 0.133 / 0.147, 0.137 | 0.095 / 0.125, 0.109 | 0.006 / 0.088, 0.093 | 0.001 / 0.092, 0.099 | 0.003 / 0.090, 0.100 |
| cr1 | 0.178 / 0.178, 0.179 | 0.172 / 0.174, 0.172 | 0.136 / 0.151, 0.138 | 0.096 / 0.137, 0.119 | 0.001 / 0.148, 0.133 | 0.000 / 0.165, 0.127 | 0.000 / 0.167, 0.127 |
| cr2 | 0.178 / 0.178, 0.178 | 0.130 / 0.180, 0.176 | -0.091 / 0.184, 0.189 | -0.093 / 0.190, 0.185 | -0.101 / 0.189, 0.184 | -0.100 / 0.185, 0.192 | -0.095 / 0.184, 0.194 |
| cr3 | 0.178 / 0.178, 0.178 | 0.169 / 0.173, 0.169 | 0.117 / 0.143, 0.124 | 0.051 / 0.116, 0.107 | -0.002 / 0.120, 0.110 | -0.002 / 0.122, 0.110 | -0.002 / 0.122, 0.111 |
| cr4 | 0.178 / 0.178, 0.178 | 0.165 / 0.169, 0.166 | 0.100 / 0.136, 0.124 | 0.060 / 0.137, 0.132 | -0.000 / 0.143, 0.133 | -0.000 / 0.144, 0.133 | -0.000 / 0.147, 0.139 |
| cr5 | 0.178 / 0.178, 0.178 | 0.168 / 0.169, 0.168 | 0.098 / 0.135, 0.124 | 0.000 / 0.134, 0.127 | 0.000 / 0.134, 0.127 | -0.000 / 0.136, 0.127 | -0.000 / 0.139, 0.141 |
| cc12 | 0.178 / 0.178, 0.178 | 0.132 / 0.179, 0.176 | -0.089 / 0.181, 0.188 | -0.099 / 0.188, 0.183 | -0.112 / 0.189, 0.182 | -0.115 / 0.187, 0.184 | -0.135 / 0.185, 0.187 |
| cc123 | 0.178 / 0.178, 0.178 | 0.134 / 0.179, 0.176 | -0.085 / 0.183, 0.190 | -0.098 / 0.189, 0.186 | -0.103 / 0.187, 0.185 | -0.103 / 0.188, 0.185 | -0.109 / 0.188, 0.185 |
| cc1234 | 0.178 / 0.178, 0.178 | 0.140 / 0.180, 0.176 | -0.078 / 0.183, 0.189 | -0.098 / 0.185, 0.187 | -0.103 / 0.184, 0.186 | -0.099 / 0.183, 0.185 | -0.099 / 0.182, 0.183 |
| cc12345 | 0.178 / 0.178, 0.178 | 0.148 / 0.180, 0.176 | -0.078 / 0.185, 0.183 | -0.093 / 0.185, 0.183 | -0.096 / 0.184, 0.182 | -0.096 / 0.184, 0.182 | -0.095 / 0.183, 0.182 |
| call | 0.178 / 0.178, 0.178 | 0.148 / 0.180, 0.176 | -0.078 / 0.185, 0.183 | -0.093 / 0.185, 0.183 | -0.096 / 0.184, 0.182 | -0.096 / 0.184, 0.182 | -0.095 / 0.183, 0.182 |
| rc_cell | 0.179 / 0.178, 0.179 | 0.160 / 0.166, 0.160 | 0.084 / 0.115, 0.103 | 0.037 / 0.107, 0.103 | -0.000 / 0.113, 0.120 | -0.000 / 0.121, 0.122 | -0.007 / 0.129, 0.147 |
| rc_cr6 | 0.178 / 0.178, 0.178 | 0.163 / 0.171, 0.164 | 0.098 / 0.150, 0.131 | -0.011 / 0.163, 0.146 | -0.028 / 0.168, 0.158 | -0.029 / 0.172, 0.167 | -0.027 / 0.173, 0.169 |
| rc_call | 0.178 / 0.179, 0.178 | 0.127 / 0.180, 0.177 | -0.069 / 0.181, 0.181 | -0.091 / 0.183, 0.181 | -0.093 / 0.183, 0.180 | -0.087 / 0.183, 0.180 | -0.085 / 0.183, 0.179 |

It does not: in every variant the box ends with its front and rear plates both on the wall
(span 0) and its chords at 0.12–0.15 m. The cell keeps its section and turns down and outboard
as a whole (front corners 0.11 m below and 0.06 m outboard of its rear corners at 0.5 m of push):
a bending collapse of box and rail end (lessons §26.2), driven by the impact beam sliding off the
boxes' ends (beam to box front +28 mm → −80 mm). With shear-matched diagonals it turns up.

### 9.2 The cell with each rule

| | off | cell | cr1 | cr2 | cr3 | cr4 | cr5 |
|---|---|---|---|---|---|---|---|
| cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN) | 1010 / 1606 | 1068 / 1507 | 1068 / 1507 | 781 / 1507 | 1068 / 1507 | 1136 / 130 | 798 / 1501 |
|   of which crash boxes / box–other braces / hood + fenders (kN) | 194 / 224 / 401 | 164 / 298 / 415 | 164 / 298 / 415 | 164 / 12 / 415 | 164 / 298 / 415 | 164 / 298 / 483 | 164 / 298 / 146 |
| cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN) | 2236 / 1334 | 2297 / 1235 | 1964 / 1235 | 2171 / 1235 | 2297 / 1235 | 2352 / 30 | 2091 / 936 |
|   of which rail chords and braces / subframe–crash box (kN) | 348 / 553 | 312 / 544 | 312 / 0 | 250 / 544 | 312 / 544 | 312 / 544 | 312 / 544 |
| push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 162 / 394 / 538 / 509 | 180 / 428 / 489 / 444 | 178 / 413 / 524 / 503 | 155 / 252 / 199 / 380 | 178 / 428 / 481 / 472 | 143 / 376 / 511 / 485 | 178 / 410 / 508 / 366 |
| push at 0.39 m: crash-box chord / rail chords (m) | 0.128 / 0.371, 0.434 | 0.129 / 0.379, 0.443 | 0.138 / 0.375, 0.419 | 0.189 / 0.371, 0.448 | 0.117 / 0.379, 0.444 | 0.136 / 0.375, 0.441 | 0.132 / 0.383, 0.446 |
| push at 0.39 m: elastic / plastic (kJ) | 40.5 / 78.0 | 44.0 / 78.6 | 45.5 / 79.7 | 30.9 / 38.4 | 42.8 / 78.8 | 40.8 / 75.5 | 39.5 / 75.2 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 255 / 538 / 644 / – | 282 / 573 / 687 / – | 281 / 563 / 672 / – | 267 / 477 / 499 / 470 | 281 / 569 / 694 / – | 247 / 536 / 714 / – | 273 / 539 / 640 / – |
| S11 t_m (ms) | 44.4 | 44.6 | 44.1 | 52.1 | 43.1 | 47.5 | 47.1 |
| S11 dynamic crush (m) | 0.394 | 0.378 | 0.382 | 0.465 | 0.376 | 0.400 | 0.397 |
| S11 rebound (m/s) / restitution | 5.34 / 0.342 | 5.41 / 0.346 | 5.65 / 0.362 | 4.81 / 0.308 | 5.56 / 0.356 | 5.52 / 0.353 | 5.59 / 0.358 |
| S11 residual crush (m) | 0.199 | 0.181 | 0.184 | 0.295 | 0.167 | 0.204 | 0.192 |
| S11 peak wall force (kN) at (ms) | 661 at 20.0 | 704 at 20.0 | 692 at 20.0 | 543 at 20.0 | 702 at 20.0 | 729 at 20.0 | 653 at 20.0 |
| S11 4 ms / CFC 60 peak (g) | 50.0 / 59.1 | 50.7 / 61.3 | 50.9 / 58.1 | 49.8 / 54.5 | 51.0 / 61.2 | 52.0 / 60.4 | 47.0 / 60.5 |
| S11 cabin: door opening change peak / residual (mm) | 151 / -24 | 166 / -33 | 171 / -33 | 214 / -77 | 172 / -39 | 172 / -41 | 136 / -13 |
| S11 plastic work: cabin share | 0.061 | 0.096 | 0.101 | 0.159 | 0.099 | 0.093 | 0.046 |
| S11 elastic at the stop / 0.3 s later (kJ) | 47.3 / 9.7 | 51.2 / 9.1 | 51.7 / 8.8 | 45.9 / 8.6 | 50.5 / 8.1 | 45.7 / 7.6 | 48.1 / 8.8 |
| S11 released by zone (kJ) | cabin 16.6, front lower 8.2, front lower+upper 5.8, front upper 4.9, other 2.1 | cabin 17.7, front lower+upper 11.1, front lower 5.2, front upper 4.9, other 3.2 | cabin 18.7, front lower+upper 10.1, front lower 5.6, front upper 4.9, other 3.8 | cabin 20.9, front lower+upper 5.6, front lower 5.2, front upper 3.0, other 2.8 | cabin 18.0, front lower+upper 9.7, front lower 7.0, front upper 4.4, other 3.3 | cabin 18.7, front lower+upper 7.3, front lower 6.3, front upper 4.6, other 1.2 | cabin 15.4, front lower+upper 12.1, front lower 5.2, front upper 3.5, other 3.1 |
| S11 engine rear to firewall: least (m) at (ms); within 3 cm | 0.227 at 166; None | 0.226 at 66; None | 0.224 at 65; None | 0.231 at 71; None | 0.224 at 69; None | 0.226 at 175; None | 0.222 at 157; None |
| S12 t_m (ms) / crush (m) / restitution / residual (m) | 39.5 / 0.275 / 0.484 / 0.099 | 38.6 / 0.263 / 0.493 / 0.095 | 39.3 / 0.266 / 0.496 / 0.101 | 60.1 / 0.335 / 0.306 / 0.217 | 39.0 / 0.264 / 0.484 / 0.098 | 40.7 / 0.279 / 0.472 / 0.111 | 39.7 / 0.269 / 0.484 / 0.104 |
| S12 peak wall force (kN) at (ms) | 576 at 32.0 | 606 at 28.0 | 596 at 28.0 | 459 at 24.0 | 596 at 28.0 | 601 at 28.0 | 578 at 32.0 |
| hard: energy gain S11 / S12 (J); into the wall (m) | 0.0 / 1.4; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 |

![cell, each rule](../out/sections/fixes/front_ab_off_cell_cr1_cr2_cr3_cr4_cr5.png)

### 9.3 Cumulative with the cell

| | off | cell | cr1 | cc12 | cc123 | cc1234 | cc12345 | call |
|---|---|---|---|---|---|---|---|---|
| cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN) | 1010 / 1606 | 1068 / 1507 | 1068 / 1507 | 781 / 1507 | 781 / 1507 | 849 / 130 | 515 / 130 | 515 / 130 |
|   of which crash boxes / box–other braces / hood + fenders (kN) | 194 / 224 / 401 | 164 / 298 / 415 | 164 / 298 / 415 | 164 / 12 / 415 | 164 / 12 / 415 | 164 / 12 / 483 | 164 / 12 / 149 | 164 / 12 / 149 |
| cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN) | 2236 / 1334 | 2297 / 1235 | 1964 / 1235 | 1837 / 1235 | 1837 / 1235 | 1893 / 30 | 1583 / 30 | 1583 / 30 |
|   of which rail chords and braces / subframe–crash box (kN) | 348 / 553 | 312 / 544 | 312 / 0 | 250 / 0 | 250 / 0 | 250 / 0 | 250 / 0 | 250 / 0 |
| push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 162 / 394 / 538 / 509 | 180 / 428 / 489 / 444 | 178 / 413 / 524 / 503 | 153 / 255 / 220 / 375 | 151 / 259 / 217 / 315 | 138 / 226 / 174 / 352 | 134 / 198 / 115 / 365 | 134 / 198 / 115 / 365 |
| push at 0.39 m: crash-box chord / rail chords (m) | 0.128 / 0.371, 0.434 | 0.129 / 0.379, 0.443 | 0.138 / 0.375, 0.419 | 0.187 / 0.371, 0.449 | 0.188 / 0.364, 0.448 | 0.185 / 0.364, 0.448 | 0.186 / 0.360, 0.445 | 0.186 / 0.360, 0.445 |
| push at 0.39 m: elastic / plastic (kJ) | 40.5 / 78.0 | 44.0 / 78.6 | 45.5 / 79.7 | 30.5 / 35.1 | 21.8 / 49.2 | 24.1 / 44.0 | 16.7 / 30.8 | 16.7 / 30.8 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 255 / 538 / 644 / – | 282 / 573 / 687 / – | 281 / 563 / 672 / – | 266 / 474 / 515 / 425 | 266 / 470 / 515 / 445 | 238 / 448 / 524 / 417 | 225 / 410 / 412 / 414 | 225 / 410 / 412 / 414 |
| S11 t_m (ms) | 44.4 | 44.6 | 44.1 | 53.8 | 53.4 | 53.0 | 54.5 | 54.5 |
| S11 dynamic crush (m) | 0.394 | 0.378 | 0.382 | 0.471 | 0.467 | 0.479 | 0.507 | 0.507 |
| S11 rebound (m/s) / restitution | 5.34 / 0.342 | 5.41 / 0.346 | 5.65 / 0.362 | 4.88 / 0.312 | 4.78 / 0.306 | 5.13 / 0.328 | 4.91 / 0.315 | 4.91 / 0.315 |
| S11 residual crush (m) | 0.199 | 0.181 | 0.184 | 0.299 | 0.296 | 0.280 | 0.307 | 0.307 |
| S11 peak wall force (kN) at (ms) | 661 at 20.0 | 704 at 20.0 | 692 at 20.0 | 555 at 20.0 | 550 at 20.0 | 559 at 20.0 | 535 at 40.0 | 535 at 40.0 |
| S11 4 ms / CFC 60 peak (g) | 50.0 / 59.1 | 50.7 / 61.3 | 50.9 / 58.1 | 40.3 / 46.5 | 39.6 / 46.4 | 45.1 / 51.0 | 45.8 / 51.6 | 45.8 / 51.6 |
| S11 cabin: door opening change peak / residual (mm) | 151 / -24 | 166 / -33 | 171 / -33 | 212 / -67 | 211 / -66 | 186 / -46 | 116 / -9 | 116 / -9 |
| S11 plastic work: cabin share | 0.061 | 0.096 | 0.101 | 0.150 | 0.147 | 0.127 | 0.030 | 0.030 |
| S11 elastic at the stop / 0.3 s later (kJ) | 47.3 / 9.7 | 51.2 / 9.1 | 51.7 / 8.8 | 44.9 / 8.3 | 44.8 / 8.5 | 42.1 / 5.6 | 34.6 / 5.9 | 34.6 / 5.9 |
| S11 released by zone (kJ) | cabin 16.6, front lower 8.2, front lower+upper 5.8, front upper 4.9, other 2.1 | cabin 17.7, front lower+upper 11.1, front lower 5.2, front upper 4.9, other 3.2 | cabin 18.7, front lower+upper 10.1, front lower 5.6, front upper 4.9, other 3.8 | cabin 21.2, front lower+upper 5.4, front lower 3.7, front upper 3.5, other 3.0 | cabin 21.2, front lower+upper 5.0, front upper 3.5, front lower 3.4, other 3.2 | cabin 20.6, front lower 5.5, front upper 4.8, front lower+upper 3.6, other 2.1 | cabin 12.2, front lower 7.3, front lower+upper 4.2, other 3.0, front upper 2.1 | cabin 12.2, front lower 7.3, front lower+upper 4.2, other 3.0, front upper 2.1 |
| S11 engine rear to firewall: least (m) at (ms); within 3 cm | 0.227 at 166; None | 0.226 at 66; None | 0.224 at 65; None | 0.224 at 68; None | 0.224 at 69; None | 0.229 at 175; None | 0.227 at 77; None | 0.227 at 77; None |
| S12 t_m (ms) / crush (m) / restitution / residual (m) | 39.5 / 0.275 / 0.484 / 0.099 | 38.6 / 0.263 / 0.493 / 0.095 | 39.3 / 0.266 / 0.496 / 0.101 | 60.4 / 0.337 / 0.275 / 0.228 | 59.2 / 0.332 / 0.302 / 0.209 | 57.6 / 0.336 / 0.346 / 0.175 | 66.1 / 0.370 / 0.301 / 0.192 | 66.1 / 0.370 / 0.301 / 0.192 |
| S12 peak wall force (kN) at (ms) | 576 at 32.0 | 606 at 28.0 | 596 at 28.0 | 462 at 24.0 | 455 at 24.0 | 459 at 24.0 | 400 at 24.0 | 400 at 24.0 |
| hard: energy gain S11 / S12 (J); into the wall (m) | 0.0 / 1.4; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 4859.5; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 |

![cell, cumulative](../out/sections/fixes/front_ab_off_cell_cr1_cc12_cc123_cc1234_cc12345_call.png)

### 9.4 `real_crash`

| | rc_off | rc_cell | rc_cr6 | rc_call |
|---|---|---|---|---|
| cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN) | 627 / 1606 | 659 / 1507 | 708 / 1507 | 427 / 130 |
|   of which crash boxes / box–other braces / hood + fenders (kN) | 108 / 125 / 204 | 91 / 166 / 211 | 140 / 166 / 211 | 140 / 12 / 84 |
| cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN) | 1756 / 1334 | 1805 / 1235 | 1871 / 1235 | 1349 / 30 |
|   of which rail chords and braces / subframe–crash box (kN) | 193 / 553 | 174 / 544 | 240 / 544 | 207 / 0 |
| push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 157 / 299 / 330 / 301 | 152 / 329 / 385 / 340 | 160 / 346 / 333 / 303 | 125 / 146 / 90 / 259 |
| push at 0.39 m: crash-box chord / rail chords (m) | 0.160 / 0.377, 0.435 | 0.108 / 0.379, 0.444 | 0.161 / 0.382, 0.439 | 0.183 / 0.357, 0.446 |
| push at 0.39 m: elastic / plastic (kJ) | 25.8 / 60.3 | 27.9 / 67.2 | 25.8 / 66.5 | 11.9 / 27.3 |
| S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN) | 245 / 436 / 477 / 501 | 264 / 464 / 526 / 429 | 267 / 485 / 541 / 410 | 210 / 362 / 354 / 349 |
| S11 t_m (ms) | 57.3 | 53.4 | 54.3 | 61.9 |
| S11 dynamic crush (m) | 0.480 | 0.457 | 0.455 | 0.565 |
| S11 rebound (m/s) / restitution | 3.67 / 0.235 | 3.95 / 0.253 | 3.90 / 0.250 | 3.77 / 0.241 |
| S11 residual crush (m) | 0.333 | 0.293 | 0.320 | 0.373 |
| S11 peak wall force (kN) at (ms) | 508 at 32.0 | 527 at 20.0 | 549 at 20.0 | 446 at 36.0 |
| S11 4 ms / CFC 60 peak (g) | 39.4 / 41.4 | 39.3 / 42.6 | 39.2 / 45.0 | 40.0 / 44.9 |
| S11 cabin: door opening change peak / residual (mm) | 131 / -23 | 154 / -41 | 149 / -38 | 82 / -12 |
| S11 plastic work: cabin share | 0.031 | 0.052 | 0.049 | 0.018 |
| S11 elastic at the stop / 0.3 s later (kJ) | 28.0 / 7.3 | 33.3 / 10.0 | 31.5 / 9.6 | 25.4 / 4.6 |
| S11 released by zone (kJ) | cabin 11.0, front lower+upper 3.8, front upper 3.4, front lower 1.7, other 1.0 | cabin 12.7, front lower+upper 4.4, front lower 3.1, front upper 2.5, other 0.7 | cabin 12.4, front lower+upper 3.5, front lower 2.6, front upper 2.5, other 1.0 | cabin 6.7, other 4.8, front lower 4.3, front lower+upper 3.1, front upper 1.9 |
| S11 engine rear to firewall: least (m) at (ms); within 3 cm | 0.234 at 77; None | 0.246 at 184; None | 0.239 at 186; None | 0.222 at 55; None |
| S12 t_m (ms) / crush (m) / restitution / residual (m) | 50.3 / 0.317 / 0.366 / 0.156 | 48.0 / 0.304 / 0.395 / 0.152 | 47.3 / 0.297 / 0.383 / 0.146 | 69.6 / 0.427 / 0.309 / 0.252 |
| S12 peak wall force (kN) at (ms) | 427 at 24.0 | 462 at 28.0 | 483 at 28.0 | 340 at 24.0 |
| hard: energy gain S11 / S12 (J); into the wall (m) | 0.0 / 1.3; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 | 0.0 / 0.0; 0.000 |

### 9.5 Bay by bay at 0.39 m of push (z shortening, m)

| variant | crash box a1–a2: z shortening (m) | chords' length upper / lower (m, from 0.180) | rail a2–a4 (z) | rail a4–a5 (z) | a5 to firewall (z) | firewall to cabin rear | crash box front's rise (m) |
|---|---|---|---|---|---|---|---|
| off | 0.100 | 0.128 / 0.111 | 0.029 | 0.023 | -0.023 | 0.109 | -0.017 |
| cell | 0.111 | 0.129 / 0.119 | 0.021 | 0.015 | -0.028 | 0.118 | 0.001 |
| cr1 | 0.076 | 0.138 / 0.119 | 0.021 | 0.041 | -0.027 | 0.126 | -0.006 |
| cr2 | 0.276 | 0.190 / 0.186 | 0.019 | 0.009 | -0.020 | 0.050 | 0.028 |
| cr3 | 0.122 | 0.117 / 0.107 | 0.016 | 0.014 | -0.027 | 0.112 | 0.001 |
| cr4 | 0.112 | 0.136 / 0.130 | 0.021 | 0.017 | -0.028 | 0.115 | 0.006 |
| cr5 | 0.168 | 0.132 / 0.125 | 0.011 | 0.010 | -0.020 | 0.067 | -0.001 |
| cc12 | 0.277 | 0.187 / 0.184 | 0.021 | 0.009 | -0.023 | 0.050 | 0.055 |
| cc123 | 0.276 | 0.188 / 0.186 | 0.027 | 0.008 | -0.017 | 0.040 | 0.024 |
| cc1234 | 0.273 | 0.185 / 0.187 | 0.024 | 0.008 | -0.021 | 0.047 | 0.013 |
| cc12345 | 0.272 | 0.186 / 0.183 | 0.024 | 0.009 | -0.010 | 0.035 | -0.038 |
| call | 0.272 | 0.186 / 0.183 | 0.024 | 0.009 | -0.010 | 0.035 | -0.038 |
| rc_cell | 0.132 | 0.107 / 0.102 | 0.014 | 0.014 | -0.025 | 0.104 | 0.006 |
| rc_cr6 | 0.184 | 0.162 / 0.143 | 0.022 | 0.017 | -0.027 | 0.073 | 0.025 |
| rc_call | 0.270 | 0.183 / 0.181 | 0.032 | 0.008 | -0.008 | 0.026 | -0.026 |

### 9.6 Hard rules, 8 runs × 3 contact settings, judged by `crash_reference::verdict`

`crash_verdict` (`engine-softbody-lab/examples/crash_verdict.rs`) judges each file's 8 reports
again by the scenarios' current criteria and known failures, so a known failure (listed in the
scenario, e.g. S11's restitution, S08's centre of mass) does not fail it and a new one does.
Before the rules (`off`), every scenario and setting is ok with its known failures.

Hard rules on 8 runs × discrete/edge/continuous × S11, S12, S01, S02 (the 4x4), S08, judged by
`crash_reference::verdict` (`crash_verdict`; known failures listed in the scenarios do not fail,
new ones do; `off` is ok everywhere). Variants with one file are S08 edge only, the setting where
the edge contact's energy fault shows:

| variant | settings judged | new failures (scenario, setting: criterion, runs) | known failures that now pass in every run |
|---|---|---|---|
| off | 15 | none | – |
| r1 | 15 | S08 continuous: turns the same sense as the other 2/8 | – |
| r2 | 1 | S08 edge: no energy created 5/8 | – |
| r3 | 1 | S08 edge: no energy created 1/8 | – |
| r4 | 15 | none | S08 continuous: the common centre of mass stays still |
| r5 | 15 | S08 continuous: turns the same sense as the other 1/8 | S08 continuous: the common centre of mass stays still |
| all | 15 | S08 edge: no energy created 1/8 | S08 continuous: the common centre of mass stays still |
| c12 | 1 | S08 edge: no energy created 5/8 | – |
| c123 | 1 | S08 edge: no energy created 8/8 | – |
| c1234 | 1 | S08 edge: no energy created 3/8 | – |
| c12345 | 1 | S08 edge: no energy created 1/8 | – |
| rc_off | 15 | S08 continuous: momentum across the road 2/8, S08 continuous: turns the same sense as the other 5/8, S08 discrete: momentum along the road 1/8, S08 discrete: no energy created 2/8, S11 discrete: no energy created 1/8, S11 edge: no energy created 1/8, S12 continuous: no energy created 1/8 | S11 discrete: restitution within the 56 km/h sedan fleet's; S11 edge: restitution within the 56 km/h sedan fleet's |
| rc_r6 | 15 | S08 continuous: momentum across the road 1/8, S08 continuous: turns the same sense as the other 5/8, S08 edge: turns the same sense as the other 3/8 | – |
| cell | 15 | S08 continuous: momentum across the road 2/8, S08 continuous: turns the same sense as the other 2/8, S08 edge: turns the same sense as the other 1/8 | – |
| cr1 | 1 | none | – |
| cr2 | 1 | S08 edge: turns the same sense as the other 2/8 | – |
| cr3 | 1 | none | – |
| cr4 | 1 | S08 edge: no energy created 2/8 | – |
| cr5 | 15 | none | S08 continuous: the common centre of mass stays still |
| call | 15 | none | S08 continuous: the common centre of mass stays still |

With the cell, rules 2 and 3 no longer create energy in S08 edge (worst 296 and 69 J against 2.8
kJ and 674 J without it); rule 4 still does (2 of 8). The cell alone fails S08's own laws in 2 of
8 continuous runs (momentum across the road, 540 N·s against the 450 bar) and its "turns the same
sense" in 3 runs; with rule 5 or with every rule on, S08 holds every criterion in every setting.
S08's laws are chaotic in our runs (the same variant's run-by-run spread decides them), so the
one-rule-at-a-time standard reads them run by run.

| variant | scenario | contact | runs | energy created, max (J) | its band | into the wall, max (m) | finite | crossings, max | asserted failures | restitution | dyn. crush (m) | residual (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cell | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.341–0.357 | 0.367–0.384 | 0.161–0.192 |
| cell | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.341–0.357 | 0.367–0.384 | 0.161–0.192 |
| cell | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.000 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.348–0.379 | 0.359–0.379 | 0.174–0.196 |
| cell | S12 | discrete | 8 | 3.4 | ≤ 89.0 | 0.000 | 1 | – | none | 0.480–0.505 | 0.259–0.268 | 0.092–0.098 |
| cell | S12 | edge | 8 | 3.4 | ≤ 89.0 | 0.000 | 1 | – | none | 0.480–0.505 | 0.259–0.268 | 0.092–0.098 |
| cell | S12 | continuous | 8 | 3.4 | ≤ 89.0 | 0.000 | 1 | – | none | 0.497–0.516 | 0.258–0.267 | 0.091–0.101 |
| cell | S01 | discrete | 8 | 1.4 | ≤ 230.0 | 0.000 | 1 | – | none | 0.250–0.275 | 0.548–0.573 | 0.055–0.064 |
| cell | S01 | edge | 8 | 1.4 | ≤ 230.0 | 0.000 | 1 | – | none | 0.250–0.275 | 0.548–0.573 | 0.055–0.064 |
| cell | S01 | continuous | 8 | 63.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.265–0.310 | 0.540–0.566 | -0.002–0.063 |
| cell | S02 | discrete | 8 | 0.0 |  | 0.000 | 1 | – | none | 0.303–0.330 | 0.295–0.306 | -0.264–-0.078 |
| cell | S02 | edge | 8 | 0.0 |  | 0.000 | 1 | – | none | 0.303–0.330 | 0.295–0.306 | -0.264–-0.078 |
| cell | S02 | continuous | 8 | 0.0 |  | 0.009 | 1 | – | none | 0.316–0.337 | 0.291–0.303 | -0.097–0.060 |
| cell | S08 | discrete | 8 | 206.1 | ≤ 340.0 | 0.000 | 1 | 204.0 | the common centre of mass stays still 1/8 | 0.089–0.211 | 0.710–0.795 | -0.017–0.051 |
| cell | S08 | edge | 8 | 153.4 | ≤ 340.0 | 0.000 | 1 | 16.0 | momentum across the road 3/8, momentum along the road 2/8, the common centre of mass stays still 6/8, turns the same sense as the other 1/8 | 0.163–0.225 | 0.682–0.736 | -0.048–0.050 |
| cell | S08 | continuous | 8 | 177.3 | ≤ 340.0 | 0.000 | 1 | 0.0 | momentum across the road 2/8, momentum along the road 1/8, the common centre of mass stays still 6/8, turns the same sense as the other 2/8 | 0.134–0.174 | 0.654–0.727 | -0.011–0.038 |
| cr5 | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.013 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.344–0.376 | 0.387–0.420 | 0.173–0.220 |
| cr5 | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.013 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.344–0.376 | 0.387–0.420 | 0.173–0.220 |
| cr5 | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.013 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.349–0.379 | 0.384–0.419 | 0.187–0.223 |
| cr5 | S12 | discrete | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.438–0.486 | 0.265–0.276 | 0.095–0.111 |
| cr5 | S12 | edge | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.438–0.486 | 0.265–0.276 | 0.095–0.111 |
| cr5 | S12 | continuous | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.444–0.491 | 0.264–0.276 | 0.096–0.112 |
| cr5 | S01 | discrete | 8 | 122.6 | ≤ 230.0 | 0.000 | 1 | – | none | 0.184–0.243 | 0.570–0.600 | 0.017–0.033 |
| cr5 | S01 | edge | 8 | 122.6 | ≤ 230.0 | 0.000 | 1 | – | none | 0.184–0.243 | 0.570–0.600 | 0.017–0.033 |
| cr5 | S01 | continuous | 8 | 0.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.212–0.256 | 0.572–0.601 | 0.025–0.059 |
| cr5 | S02 | discrete | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.240–0.311 | 0.328–0.350 | 0.062–0.157 |
| cr5 | S02 | edge | 8 | 0.0 |  | 0.002 | 1 | – | none | 0.240–0.311 | 0.328–0.350 | 0.062–0.157 |
| cr5 | S02 | continuous | 8 | 0.0 |  | 0.003 | 1 | – | none | 0.268–0.306 | 0.327–0.348 | -0.011–0.073 |
| cr5 | S08 | discrete | 8 | 104.2 | ≤ 340.0 | 0.000 | 1 | 107.0 | none | 0.117–0.206 | 0.753–0.781 | -0.050–0.012 |
| cr5 | S08 | edge | 8 | 58.0 | ≤ 340.0 | 0.000 | 1 | 26.0 | momentum across the road 3/8 | 0.159–0.225 | 0.704–0.760 | -0.050–0.022 |
| cr5 | S08 | continuous | 8 | 42.3 | ≤ 340.0 | 0.000 | 1 | 0.0 | none | 0.128–0.182 | 0.729–0.790 | -0.024–0.017 |
| call | S11 | discrete | 8 | 0.0 | ≤ 177.0 | 0.005 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.300–0.318 | 0.499–0.514 | 0.296–0.310 |
| call | S11 | edge | 8 | 0.0 | ≤ 177.0 | 0.005 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.300–0.318 | 0.499–0.514 | 0.296–0.310 |
| call | S11 | continuous | 8 | 0.0 | ≤ 177.0 | 0.001 | 1 | – | restitution within the 56 km/h sedan fleet's 8/8 | 0.302–0.321 | 0.501–0.513 | 0.299–0.310 |
| call | S12 | discrete | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.263–0.337 | 0.361–0.386 | 0.185–0.223 |
| call | S12 | edge | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.263–0.337 | 0.361–0.386 | 0.185–0.223 |
| call | S12 | continuous | 8 | 0.0 | ≤ 89.0 | 0.000 | 1 | – | none | 0.259–0.335 | 0.360–0.386 | 0.184–0.222 |
| call | S01 | discrete | 8 | 18.6 | ≤ 230.0 | 0.000 | 1 | – | none | 0.137–0.153 | 0.640–0.664 | 0.050–0.065 |
| call | S01 | edge | 8 | 18.6 | ≤ 230.0 | 0.000 | 1 | – | none | 0.137–0.153 | 0.640–0.664 | 0.050–0.065 |
| call | S01 | continuous | 8 | 10.0 | ≤ 230.0 | 0.000 | 1 | – | none | 0.146–0.175 | 0.640–0.660 | 0.057–0.080 |
| call | S02 | discrete | 8 | 0.0 |  | 0.001 | 1 | – | none | 0.261–0.273 | 0.408–0.416 | 0.099–0.213 |
| call | S02 | edge | 8 | 0.0 |  | 0.001 | 1 | – | none | 0.261–0.273 | 0.408–0.416 | 0.099–0.213 |
| call | S02 | continuous | 8 | 0.0 |  | 0.001 | 1 | – | none | 0.272–0.280 | 0.408–0.416 | 0.122–0.208 |
| call | S08 | discrete | 8 | 232.3 | ≤ 340.0 | 0.000 | 1 | 99.0 | none | 0.100–0.145 | 0.886–0.939 | -0.062–-0.011 |
| call | S08 | edge | 8 | 16.1 | ≤ 340.0 | 0.000 | 1 | 22.0 | none | 0.107–0.175 | 0.856–0.899 | -0.040–-0.014 |
| call | S08 | continuous | 8 | 4.3 | ≤ 340.0 | 0.000 | 1 | 0.0 | none | 0.125–0.163 | 0.853–0.912 | -0.025–-0.016 |

### 9.7 Gates

Gates (`front_gates.sh`, `RUBBLE_FRONT_RULES`; `gen3_rig` on the generated sedan), before / the
cell alone / all seven switches:

| test | before | cell | all seven |
|---|---|---|---|
| `engine-vehiclegen` lib (63), `crash_zones`, `mechanical`; authoring `crash_zones`, `truck_cab`, `box_trailer`, `sedan_energy` (6), `crash_invariants` (9), `rig_export` (2), `skin_lint` (3), `paint_binds_skin` (1) | pass | pass | pass |
| `every_design` (≤ 5 mm parked, nothing bent) | pass (4x4 `bf_g2` 4.9 mm) | pass | pass |
| `generated_vehicle_lint` | the two listed failures | the same rules on the same designs, every line identical | the same two tests fail; inside them (rule 5's hood row and triangles): Mach 1 skin winding 4 → 6 edges (`bb_o10-11`, `bb_o15-16`), Mach 1 gasser inward-facing triangles 54 → 56, the 4x4's contacter pairs inside the contact range 21 → 23, more open edges on the Mach 1s and the 4x4 (holes 1 each, as before) |
| `gen3_rig::rest_fidelity_at_spawn_and_parked` (≤ 1 mm, ≤ 1°; fails before) | spawn 0.105 mm / 0.05°, parked 0.065 mm / 1.090° | 0.105 mm / 0.05°, 0.065 mm / 1.110° | parked 0.065 mm / 1.097° |
| `gen_sedan_contact` | the six listed failures | the same six | the same six |
| `continuous_contact` | the three listed | the same three | the same three |
| `gen_sedan_paint::accept_head_30_pressed` (10 mm from 0.3 s) | pass, 8 mm | pass, 2 mm | pass |
| `gen_sedan_paint::accept_head_60_pressed` (50 mm) | pass, 21 mm | pass, 14 mm | pass |
| `gen_sedan_paint::accept_side_90` (listed failing) | fails, 55 mm | **passes**, 48 mm | **passes** |
| `gen_sedan_paint::accept_side_60_pressed` (listed failing) | fails, 16 mm | **passes**, 0 mm | **passes** |
| `gen_sedan_paint::accept_head_100` (listed failing) | fails, 61 mm | **passes**, 32 mm | fails |
| `gen_sedan_paint::own_paint_ends_apart` (listed) | fails | fails | fails |

The visual mesh: every design's `.glb` byte for byte the same with the cell (and with every
rule); the collision surface at rest is the same in the stills (`stills_S11_side_off_vs_cell.png`);
the skins change (`.rskin`: the body binds the new nodes) and the rest fidelity holds as before.

### 9.8 Defaults

**Every switch stays off by default, the cell included.** The standard (the owner's): on only if,
alone and together, it holds every hard rule and gate that held before.

| switch | alone: what stops it | category |
|---|---|---|
| `crash_box_cell` | S08 continuous: momentum across the road 2/8 (540 N·s, bar 450) and "turns the same sense" 2/8; S08 edge "turns the same sense" 1/8. And it does not do what it is for: the box still turns (§26.2) | (c) S08's own laws (momentum, the yaw symmetry), not the edge energy fault |
| 1 subframe mounts | `every_design`: the 4x4's bumper row 5.5 mm (bar 5); S08 continuous "turns the same sense" 2/8 | (c) |
| 2 bay alone | S08 edge no energy created 5/8 (2.8 kJ) | (a) the edge contact's energy fault |
| 3 stroke | S08 edge no energy created 1/8 (674 J) | (a) |
| 4 yielding stops | paint, head 60 pressed 53 mm (bar 50) | (b) the owner's paint limits |
| 5 hood row, panels | paint, head 30 pressed 21 mm (bar 10); S08 continuous "turns the same sense" 1/8 | (b) and (c) |
| 6 crash rate | `real_crash` only: S08 continuous momentum across 1/8, "turns the same sense" (edge 3/8) | (c) (`real_crash` before the rule already fails S08's and S11's laws: momentum, energy) |

**Together:** the cell with rule 5, and all seven switches on, hold every hard rule in all 15
settings and every gate that held before (and three listed paint failures pass with the cell;
side 90 and side 60 pressed with all seven). The one-at-a-time standard still keeps them off,
because the cell alone fails S08's momentum in 2 of 8 continuous runs: those laws are decided run
by run by chaotic two-car contact, so a set can pass where its members fail. That is a conflict
the rule does not order: the owner's call. **All seven together** measure (S11 run 0): t_m 54.5
ms, crush 0.507 m, restitution 0.315, residual 0.307 m, peak 535 kN at 40 ms; S12 t_m 66 ms,
restitution 0.301 (Camry 69–75 ms, 0.11–0.15). The cell is necessary for rules 2 and 3 (their
energy creation goes away with it) but not sufficient for the box to fold.

**If the owner switched a set on, what would have to change** (none was changed here):
`tests-expected-failures.txt` loses `gen_sedan_paint::accept_side_90` and `accept_side_60_pressed`
(they pass; with the cell alone `accept_head_100` too); S08's known failure "the common centre
of mass stays still" in continuous contact passes in every run with rules 4 or 5 and with all
seven (`crash_verdict`: "remove the entry"); in `real_crash` without rule 6, S11's discrete and edge
restitution known failures pass; the scenario matrix's car (`out/gen/sedan_new`), its saved
states and the portable fixtures, the site's `__mods/*_new`, and the crash reference's
`crash-reference.md` §4.3 rows are regenerated; the generated vehicles' JBeam and `.rskin` files
change (the `.glb` does not).

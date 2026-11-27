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

# Generated vehicles: design in, crash-ready car out

> Lessons, conclusions and decisions from this work: [generated-vehicles-lessons.md](generated-vehicles-lessons.md).

A living doc. The vehicle generator draws a car; the **vehicle rigger** turns
it into a deformable node–beam car that Rubble's explicit solver, the GPU
pages and the World load like any BeamNG JBeam vehicle, with Rubble's own
features on top. One programmatic path, no hand placement:

```
design (Rust: engine-vehiclegen, parameters)      art .glb (68 parts, 42 markers)
   or (Python: engine/tools/vehicle-gen)      ──▶  + rig data (key lines, sections,
                                                    end faces, skin loops, markers,
                                                    PHYSICS block)
                                                         │
                                         vehicle rigger  ▼  engine_softbody::authoring::rig
     lattice on the render model ─ masses, springs, yields (census rules + PHYSICS)
     ─ hardware at the markers ─ stability bound ─ collision skin from the render skin
     ─ triangle aerodynamics ─ weighted skins binding every render vertex
                                                         │
                                                         ▼
     mod folder: <design>/<design>.jbeam + parts, <design>.pc, <design>.glb, <design>.rskin
```

(History: the lattice code started as the rig of an earlier hand-written Rust
render model; it now rigs any design through one input trait.)

## Run it

```bash
# Rig the Rust sedan in memory and write the mod (default __mods/sedan, gitignored)
cargo run --release -p engine-softbody --example vehicle_gen_rig -- sedan [--params p.json]
# Checks
cargo run --release -p engine-softbody --example vehicle_lint -- sedan
GEN3_RIG_DESIGN=sedan cargo test --release -p engine-softbody --test gen3_rig -- rest_
cargo test --release -p engine-softbody --test vehicle_lint
# Drive and crash on the CPU reference
cargo run --release -p engine-softbody --example vehicle_drive -- sedan 40
cargo run --release -p engine-softbody --example vehicle_c2c -- sedan 50 0.6
SURFACE_CONTACT=1 cargo run --release -p engine-softbody --example vehicle_compare -- --cars sedan
# Clips: CPU reference simulation, skinned mesh rendered in Blender (frames in out/film)
sh engine/tools/vehicle-gen/tools/film_all.sh sedan
# The page
softbody.html?solver=explicit&preset=wall56&mod=/__mods/sedan&pc=sedan/sedan.pc&wheels=raycast
```

In code:

```rust
use engine_softbody::authoring::rig;
let car = rig::rig_vehicle("sedan", Some(r#"{"wheelbase": 2.9}"#), None)?;
for (path, bytes) in rig::mod_files(&car) { /* write */ }
```

Python designs (not yet ported to Rust) build their `.glb` and
`<design>.rig.json` with Blender (`build.py` writes both;
`tools/build_rigs.sh <designs>` builds them without renders into
`engine/tools/vehicle-gen/out/`), and the rigger reads those files
(`rig::rig_vehicle_from`).

## What the rigger builds

- **Lattice.** Members along the body's real structure (sills, floor rails,
  tunnel, hinge pillar, A-, B- and C-pillars, roof rails and bows, front and
  rear rails, aprons, towers, firewall, cowl, radiator support, shelf, tail
  panel), closures with depth (inner frames 6–8 cm under the skin, door
  intrusion trusses, window frames), bumper covers on beams with crush cans.
  Beams are a neighbourhood graph per part that never crosses an opening (door
  apertures from the door meshes, the cabin, the engine bay, the boot).
- **Shut lines.** Structural edge nodes on both sides of every shut line,
  every 22 cm, with SUPPORT stops across. For generated designs the shut
  lines come straight from the design (the welded skins' shared boundary
  points), not from guessing at mesh gaps.
- **Hardware from the markers.** Door hinges as break groups, latches as
  couplers, hood and boot hinges and latches, fender and bumper bolts (bumper
  covers on clips), engine mounts as four break groups, cooling and wheel
  stacks (BOUNDED stops), push links (SUPPORT) front to back.
- **Mass and stiffness.** Part masses from the PHYSICS block, spread by member
  kg/m and render surface; springs and yields by the census rules and the zone
  yields; every spring checked against the explicit step's bound (ω·h ≤ 1.92,
  centred damping).
- **Collision skin.** The render skin's restricted Delaunay triangulation over
  each part's skin nodes: every outer render vertex goes to its nearest node,
  each render triangle on three nodes gives that triangle. It covers glass,
  pillars, floor and tail face, and never spans an opening. Sliver and
  off-skin triangles are dropped.
  - The pool is the nodes on the skin (|sdf| < 3 cm: no bumper beam or inner
    frames); a triangle is judged by its three cells' area-weighted normal,
    not the one render triangle at their junction; a quad of four meeting
    cells keeps its shorter diagonal.
  - The front cover carries a row on the grille's top edge; seam bridges
    (the same triangulation over all parts' skin nodes and the grille, plus
    a zipper from the open edges) close the strips between parts' patches
    (cover, grille, hood, lamps, fenders), owned by the moving part, with no
    break group (`handoff/known-bugs.md` S2); doors and the boot lid get none.
  - Each wheel has a cone from its rim ring to the outer hub node.
  - Head on, the collision skin is within 15 cm of the first render surface
    over 92 % of the frontal area (was 68 %; `vehicle_c2c_overlap`).
- **Aerodynamics.** Triangle drag only (no per-node drag): `dragCoef` sized so
  the skin drags ½ρ·Cd·A·v² head on, A the model's rasterised frontal area.
- **Visual binding.** Every render vertex is bound by construction (moving
  least squares over its part's nodes, `.rskin`), rigid pieces (lamps,
  mirrors, seats, wheels, wheel-well liners) ride one frame. Glass and lamps
  are deform groups.

## The PHYSICS block

Every field has the census / 2011 Accord default, so `{}` is a valid car.
Rust designs carry it in their parameters (`SedanParams::physics`, so any
designer parameter set has its physics); Python designs set `PHYSICS = {...}`
in their design file and variants inherit it through `derive()`.
Unknown keys are an error.

| Group | Fields (defaults) |
|---|---|
| Mass (kg) | `kerb` 1450, `biw` 339, `door_f` 29.5, `door_r` 23.7, `fender` 5.1, `hood` 17.9, `decklid` 12.4, `engine` 169.9, `transaxle` 96.7, `bumper_f` 10.5, `bumper_r` 9.1, `named` {name: kg} |
| Powertrain (m) | `pt_ahead` 0.39, `pt_behind` 0.12, `dash_behind` [0.27, 0.30], `pt_clearance` 0.03 |
| Zone yields (× census 1.15 %) | `crush_yield` 0.23 (DP600 rails), `tower_yield` 0.4, `footwell_yield` 3.0, `greenhouse_yield` 0.38, `apillar_yield` 3.0 (22MnB5), `cross_member_yield` 3.0, `floor_pan_yield` 0.3 (0.7 mm HC220B), `bpillar_upper_yield` 3.0, `bpillar_lower_yield` 0.5, `intrusion_yield` 1.0, `side_yield` 1.0, `door_yield` 1.0 |
| Hardware (N per beam) | `door_hinge` 40e3, `door_latch` 50e3, `hood_hinge` 40e3, `hood_latch` 30e3, `trunk_hinge` 25e3, `trunk_latch` 45e3, `fender_bolt` 25e3, `fender_row_bolt` 16e3 (M6 8.8), `bumper_bolt` 20e3, `bumper_side_bolt` 15e3, `engine_mount` 60e3 |
| Stacks, lattice (m) | `cooling_stack` 0.08, `wheel_stack` 0.23, `reach_body` 0.90, `reach_panel` 0.66, `edge_spacing` 0.22 |
| Aero | `drag_cd` 0.30 |

Steel grades are recorded as what each zone yield stands for; the rigger
consumes yield multiples, not `Grades`.

## The lint (`authoring::lint`, `vehicle_lint`)

Hard rules: **free mechanisms** (every node held in three directions by
bilateral beams, one piece), **stability** (ω·h ≤ 1.92; fail past 2),
**mass** (±10 % of design), **hardware** (every hinge, latch, bolt and wheel
centre marker has its hardware within 15 cm), **skin coverage** (p99 of outer
render vertices within 127 mm, half the FMVSS 214 pole, of a collision
triangle). Bands (warnings) from the BeamNG census: node mass, k·L, yield
strain and damping per part, node counts, spring ends, SUPPORT share, hinge,
latch and seal patterns, body mass share, collision triangle count and edge,
sliver triangles, node distance to skin, nodes per m², vertex reach. It reads
any JBeam car; the ETK is linted only to check the bands' self-consistency.

## Results: the Rust sedan (2026-11-08, CPU reference, surface contact on)

| Check | Result | Target |
|---|---|---|
| Lattice | 744 nodes, 629 collision triangles, ω·h 1.924, 1,450 kg, CoM 1,176 mm behind the front axle | |
| Rest fidelity at spawn / parked 10 s | 0.26 mm, 0.06° / 0.83 mm, 0.74° | ≤ 1 mm, 1° |
| Rest drift, every part / broken, bent | ≤ 4.8 mm (rear rail; 5.6 mm without self-collision before the change too) / 0, 0 | ≤ 5 mm / 0, 0 |
| Lint hard rules | mechanism, mass, hardware pass; stability 1.926 (warn, bound 1.92 without the wheels' hub mass); skin coverage p99 0.116 m (end faces now checked) | |
| Drive, raycast wheels | 213 km/h top speed (drive setting 237), 0–100 km/h about 7 s; coast-down 1.9 kN at 209 km/h | not drag-capped |
| Wall 56 km/h: dynamic crush, t_m | 713 mm, 77 ms | 620–750, 63–76 |
| peak g (100 Hz, rear cabin), residual max / avg | 33.2 g, 636 / 553 mm (the nose's middle zones now on the skin; before they had no skin node and read 0) | 34–47, 380–580 / 261–512 |
| Campbell b1, restitution, feet intrusion | 24.1 /s, 0.256, 173 mm | 22.9–39.2, 0.13–0.20, 1–67 |
| broken beams, energy ledger | 1 (a hinge group), closes (largest rise 12 J) | |
| Mesh after the wall: stretch p99, fold p99, seams > 5 cm | 1.37, 6.3°, 81 of 2,770 pairs | ≤ 1.6, ≤ 31° |
| Side pole 32 km/h 75°: residual / dynamic | 64 / 158 mm (the car slides along the pole) | 297–445 residual |
| Pole sweep 90°, 24 offsets | 1 passed through | 0 |
| MDB 214 residual | 376 mm | 126–206 |
| Roof FMVSS 216a | 4.84 × weight | 4.48–5.32 |
| Door FMVSS 214 initial / intermediate / peak | 12.5 / 15.6 / 53.5 kN | ≥ 10.0 / 15.6 / 28.5 |
| Head on, sedan into sedan, 50 + 50 km/h | dynamic crush 544 / 519 mm, residual 416 / 402 mm (2 s), cabin peak 27.9 / 28.4 g, 9 / 6 beams broken, both stop | (≈ the 50 km/h wall) |
| Head on: the other car inside the collision hull, peak / at 0.6 s (`vehicle_c2c_overlap`) | contact nodes 29 / 34, deepest 302 / 331 mm; paint vertices p99 220 / 196 mm; 98 L / 8 / 10 nodes, paint p99 67 / 59 mm | 0 (was 147 / 130 nodes, 450 mm, 1,256 L) |

**On the GPU, in the World** (`engine-world` example `coupling_wall_gpu`,
`CW_MOD=crates/engine-webgpu/__mods/sedan CW_PC=sedan/sedan.pc`, 60 Hz World,
raycast wheels, 50 km/h): it loads and runs the generated car unchanged.
Into a free 1,500 kg box the box leaves at 3.9 m/s; into a wall of 192
loose 25 kg bricks the car goes through at 6.8 m/s with 98 bricks moving
and 0 beams broken.

Clips (CPU reference, frame-locked; crashes at 7.5× slow motion):
`~/Development/rubble-artifacts/vehicle-gen-physics/sedan-{drive,wall,pole,c2c,wall-orbit,c2c-orbit}.mp4`,
stills in `stills/`. After the wall the hood buckles, the fender tears at the
arch, the front doors gap at the A-pillar and the cabin stays whole. At the
pole the struck front door's leading edge stands open from the A-pillar (a
visible follow-up).

## The census lattice (2026-11-12, `feat/sedan-stiff`): measured against the ETK

BeamNG's node counts and masses on the closures (`RIG_OLD_CENSUS=1` builds
the earlier lattice): doors 17 nodes as a closed box, hood 24, lid 18,
fenders 11, bumpers 39 / 34; no light edge nodes; 3 × 3 hinges; every closure
node bears on two body members (SUPPORT), the latch side preloaded; bumpers at
ζ 0.16. `sedan_stiffness` (CPU reference; logs in
`~/Development/rubble-artifacts/sedan-stiffness/v3/`):

| | before | now | ETK |
|---|---|---|---|
| Closure Σk kept by the stability pass: door / hood / lid / fender | 0.35 / 0.55 / 0.46 / 0.56 | 0.92 / 0.79 / 0.82 / 0.89 | |
| SUPPORT share of beams (lint) | 3.0 % | 5.6 % | 11 % |
| First mode, held at neighbours (Hz): door F / door R / hood / lid / fender | 7.8 / 5.1 / 8.0 / 7.9 / 19.5 | 14.7 / 16.3 / 8.7 / 15.1 / 12.7 | 34.9 / – / 12.5 / 27.0 / 41.8 |
| 5 cm kerb at 30 km/h, raycast wheels, peak about the body (mm): body / door / hood / lid / fender / bumper | 0.87 / 1.18 / 1.75 / 0.82 / 1.36 / 1.35 | 0.41 / 0.55 / 0.64 / 0.44 / 0.51 / 0.70 | 0.09 / 0.14 / 0.23 / 0.21 / 0.09 / 0.17 |
| the same, settle under 0.2 mm (s) | 1.4–2.9 | 0.7–1.4 | 0–0.6 |
| the same on pressure wheels (mm) | 8.1 / 9.3 / 15.2 / 9.9 / 11.4 / 11.3 | 4.0 / 4.5 / 9.2 / 3.2 / 6.2 / 7.2 | 1.0 / 1.2 / 2.4 / 3.0 / 1.6 / 3.5 |
| beams the kerb bent (pressure wheels) | 42 (the aprons) | 0 | |
| Rest drift / parked fidelity | 4.8 mm / 0.83 mm, 0.74° | 3.1 mm / 0.41 mm, 0.30° | |
| Wall 56: crush, t_m, peak, residual max / avg, b1, restitution, feet | 713 mm, 77 ms, 33.2 g, 632 / 557 mm, 23.9, 0.256, 173 mm | 579 mm, 64.5 ms, 37.8 g, 438 / 329 mm, 40.5, 0.221, 83 mm | |
| Pole 32 km/h residual / sweep through / MDB / roof / door static | 128 mm / 2 / 367 mm / 4.9 / 12.4, 15.6, 44.1 kN | 170 mm / 1 / 407 mm / 6.2 / 16.9, 20.5, 58.1 kN | |
| Head on: peak nodes inside, overlap, front hubs' peak | 29 / 34, 98 L, 0.93 m | 24 / 24, 86 L, 0.46 m | |

**Not at the ETK's level** (the bar): the kerb's wobble is 2–6 × the ETK's on
raycast wheels and 1–4 × on pressure wheels, the doors' first mode is under
half of it. What limits it: the body (346 nodes at 1.6 kg against 124 at
4.0; the stability pass still keeps only 72 % of its springs, and its own
ripple, 0.41 mm, is under every closure's), SUPPORT at 5.6 % (the body has no
internal stops) and the hood's and fenders' few depth members. The body's
lattice is the next step; the hubs, strut tops and mounts must stay explicit
named beams when it is rebuilt. Not done: the visual binding's reach (the
kernel radii are shared by every skin; at 0.3–0.5 m node spacing the reach
cannot go under about 0.25 m). Head-on coverage rays more than 15 cm behind
the render surface rose from 9 % to 15 % with the coarser closures (the lint's
coverage p99 is 0.118 m, passing). The nose lift: `handoff/known-bugs.md` S2.

## Other designs

The rigger takes any design through the same input trait. What each needs:

| Design | Status | Needs |
|---|---|---|
| sedan (Rust) | rigged, checked, filmed | |
| police_sedan, wagon (Python) | rig and rest-check (drift 5.2 / 6.1 mm at the rear rail; wagon parked fidelity 1.35 mm); wall 56 in band like the sedan | port to Rust; equipment masses as named masses with positions; the wagon's long roof needs roof stations from its key lines |
| mach1_1969 | not rigged | two doors: the rigger assumes four (rear doors optional) |
| mach1_gasser | not rigged | a posed body (`post` re-poses the parts): the rig data must carry the pose |
| suv_4x4 | not rigged | the lattice's member stations are sedan heights and stations; they must come from the design's key lines (floor, sill, belt) and wheelbase |

## Follow-ups

- **Crash bands** (reported, not tuned): restitution 0.26, feet intrusion
  176 mm and Campbell b1 44 high; peak 31 g low; the side is weak (MDB
  376 mm) and the pole slides (as every car does on this harness).
- **Surface contact and light nodes:** the pole lanes' 0.15 kg contact-only
  nodes as collision triangle corners made the side pole inject energy
  (3.7e29 J) with surface contact on; generated cars have no lanes. An engine
  question for the user (`handoff/known-bugs.md`).
- **Collision skin:** one pole-sweep pass-through; below the bumper's lower
  row (under 0.29 m) and the tread beyond the rim ring stay open; the front
  corners (hood, fender, lamp, cover) still take most of the remaining
  head-on entries. Engine-side remainder: `handoff/known-bugs.md` S2.
- **Lattice from the design.** Member stations and heights are still sedan
  numbers (rails, floor, roof stations): derive them from the key lines so
  any proportion (the designer's sliders, the SUV) rigs right.
- **Census bands** the lattice misses by design: nodes per part and node mass
  (about 4× the census's nodes), k·L at the step's bound, body mass share
  (the named masses ride the body).
- **Mod size:** the Rust sedan's mod is about 85 MB, mostly the `.rskin`
  (308k render vertices).
- **On the GPU page:** record `softbody.html?mod=/__mods/sedan` in Chrome (the
  GPU World run is native here; the clips are CPU reference rendered in
  Blender).

## Solid parts (the Rust sedan's art)

The Rust generator builds every panel, pane, lens, door card and liner as a
closed solid: outer skin, edge walls, inner face
([engine-vehiclegen, "Solid parts"](../crates/engine-vehiclegen/README.md#solid-parts)).
`VgInput::from_rust` hands the rigger two things:

- **The flexbody** is the solid `.glb` (442k triangles; the sheets were
  289k), so the `.rskin` binds the edge walls and inner faces too. Every
  vertex is tagged outer / edge / inner: `_SURFACE` in the `.glb`,
  `RigSpec::surfaces` in memory.
- **The rigger's model** (lattice, edge nodes, areas, collision skin) is the
  same design built as the legacy sheets (`BuildOpts::solid = false`), so the
  rig is the one the sheet car had. `VG_RIG_OUTER=1` gives it the solids'
  bare outer skins instead: the hood's edge nodes then drift 16 mm parked
  (`gen3_rig`), because the rigger's edge nodes were built on the sheets'
  hems and inner panels. `VG_SHEET=1` builds the whole car as sheets, to
  compare a binding against.

`GEN3_RIG_DESIGN=sedan gen3_rig` with the solid flexbody on the unchanged
binding: spawn 0.26 mm / 0.06°, parked 10 s 0.83 mm / 0.71° (the sheets:
0.83 mm / 0.74°); after the 56 km/h wall, stretch p99 1.51 and fold p99 7.6°
(the sheets: 1.36 and 6.3°). Binding the inner faces and walls to their own
part's nodes, with thickness kept, is the next step.

## Body structure and mechanical parts (Step 2d, 2026-11-13, `feat/sedan-body`)

The specification is [beamng-structure-study.md](beamng-structure-study.md) §10. **State: the definitions and the shell's lattice are built and tested; the rigger does not use them yet,** so the rigged car is still the census lattice above and nothing below is a crash or stiffness measurement of the new body.

**One definition per part** (the Rust generator, solid build only; the legacy sheet build is unchanged):

| Module | What it defines | Tests |
|---|---|---|
| `engine-vehiclegen::structure` | 50 body-in-white members of the study's 22 types: section, gauge, grade, a closed solid (`structure_front`, `_floor`, `_cabin`, `_rear`), and the shell's 131 named nodes on them (24 groups, 487 kg). Steel mass, squash load and mean crush force per member. | `tests/structure.rs` |
| `engine-vehiclegen::mechanical` | 48 mechanical and interior parts: nodes and masses, internal beams, mounts as named break groups, beam-only members (arm legs, struts, tie rods, half-shafts), and a closed solid through the same points. | `tests/mechanical.rs` |
| `engine-softbody` `gen2::shell` | The shell's beams on those nodes, braced to the study's rules, with springs fitted to the node budget; the rules measured. | `tests/shell_lattice.rs` |

`RigSpec.structure` and `RigSpec.mechanical` carry the first two to the rigger.

**Structure.** Every member is closed, one piece, at least 8 mm inside the skin, and cuts through no other part or member. All 131 nodes are within 2 cm of their member's surface. The upper A-pillars, C-pillars, roof rails, headers and roof bow have no solid of their own: the shell's solid there is already a closed section 30 mm deep, with the headliner straight below it and the door frames beside it. Their nodes sit inside that section. Modelled steel is 158 kg; the rest of the shell's 487 kg is the outer skin's own solids and the fixed trim.

| Member | Section (mm), gauge, grade | Mean crush force |
|---|---|---|
| Crash box | 70 × 110, 1.5, HSLA | 45 kN |
| Front rail | 70 × 110, 1.8, HSLA | 62 kN |
| Rear rail | 60 × 90, 1.6, HSLA | 48 kN |
| Sill (inner box) | 65 × 140, 1.4, HSLA | 42 kN |
| Hinge pillar | 120 × 89, 1.6, DP600 | 66 kN |

**Mechanical parts** (kg by class): engine 170, driveline 88 (transmission 45, differential 42, shaft 1), suspension 180.5 (subframes 46.5 and 42, hubs 17 and 24, rack 10), wheels 78.4 (4 × 19 plus caps), exhaust 18.2, fuel tank 40, radiator 5.6, impact beams 14, interior 106.2 (seats 2 × 18.9, bench 18.4, instrument panel 45, console 5). Each hub is four nodes held by exactly six links. The failure ladder is ordered: toe link bends (85 kN) < arm bends (155) < wheel leaves (168) < strut breaks (180) < toe link breaks (250) < arm breaks (300) < front subframe leaves (400 per beam).

**Shell lattice** (reference sedan; bands from the study's §10.7):

| Measure | Ours | Band |
|---|---|---|
| Nodes / beams / beams per node | 131 / 699 / 5.34 | 110–140 / 650–760 / 5.0–5.8 |
| Neighbours per node, median / least | 11 / 6 | 9–11 / ≥ 5 |
| Nodes not in a tetrahedron | 0 | 0 |
| Least beam-direction spread | 0.028 | ≥ 0.02 |
| Skip beams / cross-car beams | 12.4 % / 6.4 % | 12–28 % / 8–10 % (cross-car is under its band) |
| Braces over 0.9 m through the cabin | 6 | ≤ 6 |
| Floor torsion bars | 12 | 10–20 |
| Σk/m per node, median | 8.5e6 s⁻² | 7–9.5e6 (ETK shell) |
| Shell alone, ω·h (power iteration on k + 2c/h) | 1.73 | ≤ 1.90; whole car to be measured |
| Σ yield of longitudinal beams by station (kN): nose / front axle / firewall / cabin front / cabin rear / rear axle / tail | 132 / 523 / 1,374 / 1,272 / 1,618 / 782 / 365 | rising nose to firewall; cabin ≥ 2 × the front's peak |
| Front yield under 0.45 m / over 0.75 m (shell and mechanical parts) | 59 % / 2 % | ≥ 55 % / ≤ 6 % |

**Triangles:** 465,998 in 103 parts (structure 17.3k, mechanical 13.3k). The art spec gained structure (40k) and mechanical (60k) budgets and its total rose from 500k to 600k.

**Not done:** the rigger on these definitions (shell, mechanical parts, wheels on axle nodes, glass triggers, engine stops, SUPPORT between parts), lint rules 2, 8, 11, 14–24 on the rigged car, every measurement of the plan's step D (parked fidelity, rest drift, modes, kerb wobble, settle, head-on overlap and pitch, wall, pole, MDB, roof, door), and the `-v4` clips.

**Stills** (`~/Development/rubble-artifacts/sedan-body/stills/`, Blender, all inspected): `structure_top`, `structure_under`, `structure_mechanical_top`, `structure_mechanical_under`, `mechanical_only`, `front_end`, `interior_on_floor`, `explode_all`.

## The complete sedan: equipment, hardware and trim (2026-11-13, `feat/sedan-complete`)

The question was: does the car have everything a real one has, so that nothing is hollow or empty when pieces break off? This step adds what was missing as parts, each defined once in `engine-vehiclegen::equipment` in the style of `mechanical` (a closed solid, nodes with real masses, internal beams, mounts in named break groups with finite strengths), carried to the rigger in `RigSpec.mechanical`. **State: generator only.** The rigger does not read the new fields yet, and nothing here is a crash or stiffness measurement.

**The sedan is 171 render parts and 116 part definitions** (was 103 and 48); 502,942 triangles.

**What was added**

| Item | Parts |
|---|---|
| Engine bay | Engine detail as sub-solids of the engine (head, cam cover with coils and filler cap, sump, intake plenum with runners and throttle body, exhaust manifold, crank and pump pulleys, alternator, compressor, belt). Battery on its tray, air box with duct and snorkel, fan with shroud, condenser with receiver-drier, coolant and washer reservoirs, brake booster with master cylinder, fuse box, strut-tower brace. |
| Body hardware | Eight door hinges (pillar leaf, pin, door leaf), four door latches with strikers, hood hinges, hood latch, two gas struts, boot hinges (goosenecks) and latch, foam absorbers between each bumper cover and its beam, an intrusion beam and a window regulator inside each door, fuel filler neck and flap, undertray, number plates. |
| Running gear | Brake calipers astride the discs (the disc lost its block), coil springs on the four struts, front and rear anti-roll bars with bushes and drop links, wheel bearings through the discs' hats. |
| Exhaust | The same 13-node chain drawn as sections: flanged downpipe, catalytic converter, centre pipe, resonator, pipe over the axle, rear silencer, tailpipe with tip, four rubber hangers at the hanger mounts. |
| Controls | Steering column from under the instrument panel through the firewall, two joints and the intermediate shaft to a pinion on the rack; pedal box with brake and accelerator pedals and the push rod through the firewall to the booster. The firewall has an opening for each. |
| Interior | Moulded carpet over the floor, toe board, tunnel and longitudinals (cut out round the seat cross members and the console); A, B and C pillar trims; sun visors; interior mirror; front seat belts (reel, webbing, ring, buckle); parcel shelf trim with speaker grilles; boot floor lining and side trims; dials, centre screen, vents and glovebox door on the instrument panel. |

**New fields the rigger must honour** (`mechanical.rs`):

- `Mount.to_part`: the mount's far ends are nodes of a body panel's own lattice (`door_FL`, `hood`, `trunk_lid`), the nearest to `from`; `to` is empty. A hinge, a latch or a gas strut is a part mounted to the shell with one group and to its panel with another.
- `MechPart.host` and `carried_kg`: a part with no nodes that is fixed inside or onto a body panel (door beam, window regulator, fuel flap, number plate). Its solid follows the host's nodes and its mass is part of the host's listed mass. A caliper has no host: it binds to its hub's nodes, and its mass is in the hub's.
- `MechPart.from_trim`: its mass came out of the shell's fixed-trim allowance.
- `Structure.trim_removed`, `Structure::take_trim`, `TRIM_GROUPS`: the shell's floor, tunnel, firewall, boot-floor and quarter nodes (groups C1, C3, B1, E3, E5) are lighter by the mass that became parts, in proportion to their mass (each is 53 % lighter). The study's §10.1 puts the fixed trim on these nodes.

**Mass by class** (kg; `Sedan::mass_table`, `tests/equipment.rs`). Kerb mass stays 1,450 kg.

| Class | Before | After | Study's band (§10.1) |
|---|---|---|---|
| Shell: body-in-white | 339.0 | 339.0 | shell 430–500 with its trim |
| Shell: fixed trim (no part of its own) | 148.1 | 46.0 | (100–150 before trim was parts) |
| Bumper beams and absorbers | 14.0 | 15.4 | |
| Suspension, subframes, hubs, steering | 180.5 | 191.9 | 240–270 |
| Wheels | 78.4 | 78.4 | 72–80 |
| Engine | 170.0 | 170.0 | 150–190 |
| Driveline | 88.0 | 88.0 | 60–70 |
| Exhaust, fuel, cooling | 63.8 | 73.5 | 70–80 |
| Interior | 106.2 | 130.6 | 80–110 (seats, panel, console only) |
| Electrical, engine bay, controls | 0 | 36.4 | |
| Hardware and body (hinges, latches, struts, undertray) | 0 | 18.8 | |
| Panels, glass, lamps, driver | 262.0 | 262.0 | |

102.1 kg moved from the shell's nodes to 53 parts with nodes of their own (157 nodes, none lighter than 0.1 kg); 15 more parts have no nodes and ride a panel or a hub. The 46 kg left on the shell is what still has no part: wiring, the heater unit, sealer and sound deadening, paint, fluids.

**Parts inventory** (every part definition; a family's row gives its first member: rear hubs are 24 kg, rear calipers carry 3.2 kg). Regenerate with `cargo test --release -p engine-vehiclegen --test equipment inventory -- --nocapture --ignored`.

| System | Part | kg each (nodes) | Held by (beams × break force per beam) |
|---|---|---|---|
| bumper | `impact_beam_front` | 8.00 (10) | shell A1 8×60.0 kN |
| bumper | `impact_beam_rear` | 6.00 (10) | shell E1 8×40.0 kN |
| bumper | `absorber_front` | 0.70 (4) | impbf 8×0.8 kN |
| bumper | `absorber_rear` | 0.70 (4) | impbr 8×0.8 kN |
| suspension | `subframe_front` | 46.50 (7) | shell A4 16×400.0 kN |
| suspension | `subframe_rear` | 42.00 (6) | shell E1 10×150.0 kN |
| suspension | `steering_rack` | 10.00 (2) | subf 4×inf kN |
| suspension | `hub` ×4 | 17.00 (4) | follows its joints |
| suspension | `arm_lower` ×4 | in its joints' parts | link 300 kN |
| suspension | `strut` ×4 | in its joints' parts | link 180 kN; link 300 kN |
| suspension | `toe_link` ×4 | in its joints' parts | link 250 kN |
| suspension | `arm_upper` ×2 | in its joints' parts | link 300 kN |
| suspension | `tower_brace` | 2.40 (2) | shell A6 4×32.0 kN |
| suspension | `brake_caliper` ×4 | 4.50 in its host | follows its joints |
| suspension | `antiroll_bar_front` | 5.50 (4) | subf 3×12.0 kN; hub 1×8.0 kN |
| suspension | `antiroll_bar_rear` | 3.50 (4) | subr 3×12.0 kN; hub 1×8.0 kN |
| wheel | `wheel` ×4 | 19.00 (5) | hub 8×168.0 kN |
| wheel | `hub_cap` ×4 | 0.60 (3) | whe 6×0.6 kN |
| engine | `engine` | 170.00 (10) | subf 3×90.0 kN |
| driveline | `transmission` | 45.00 (2) | eng 8×inf kN; shell C1 9×160.0 kN |
| driveline | `driveshaft` | 1.00 (2) | tra 2×3.5 kN; dif 3×3.5 kN |
| driveline | `differential` | 42.00 (3) | subr 18×200.0 kN |
| driveline | `halfshaft` ×2 | in its joints' parts | link 10 kN |
| exhaust | `exhaust` | 18.20 (13) | eng 4×65.0 kN; shell C1 8×15.0 kN; shell E1 8×15.0 kN |
| fuel | `fuel_tank` | 40.00 (8) | shell E1 28×35.0 kN |
| fuel | `fuel_filler` | 1.30 (3) | fuet 4×4.0 kN; shell E5 4×4.0 kN |
| cooling | `radiator` | 5.60 (4) | shell A3 12×20.0 kN |
| cooling | `fan_shroud` | 4.20 (4) | rad 8×3.0 kN |
| cooling | `condenser` | 2.60 (4) | rad 8×3.0 kN |
| cooling | `coolant_reservoir` | 1.60 (4) | shell A7 6×1.5 kN |
| interior | `seat` ×2 | 18.90 (7) | shell C1 6×30.0 kN |
| interior | `seat_rear` | 18.40 (8) | shell C1 14×30.0 kN; shell E7 10×30.0 kN |
| interior | `dash` | 45.00 (7) | shell B2 10×60.0 kN; shell C3 10×60.0 kN |
| interior | `console` | 5.00 (4) | shell C3 12×10.0 kN |
| interior | `carpet` | 9.00 (6) | shell C1 12×0.6 kN |
| interior | `pillar_trim_a` ×2 | 0.60 (3) | shell D1 5×0.4 kN |
| interior | `pillar_trim_c` ×2 | 0.60 (3) | shell D3 5×0.4 kN |
| interior | `pillar_trim_b` ×2 | 0.60 (3) | shell D2 5×0.4 kN |
| interior | `sun_visor` ×2 | 0.35 (3) | shell D4 4×0.2 kN |
| interior | `seat_belt` ×2 | 1.50 (3) | shell C2 3×15.0 kN; shell D2 2×15.0 kN; shell C3 3×15.0 kN |
| interior | `boot_trim` ×2 | 1.40 (3) | shell E2 6×0.5 kN |
| interior | `interior_mirror` | 0.30 (3) | shell D4 3×0.2 kN |
| interior | `parcel_shelf_trim` | 2.00 (4) | shell E6 8×0.5 kN |
| interior | `boot_lining` | 3.00 (4) | shell E3 8×0.5 kN |
| electrical | `battery` | 17.00 (4) | shell A8 8×9.0 kN |
| electrical | `fuse_box` | 1.40 (4) | shell B1 6×1.2 kN |
| engine_bay | `airbox` | 2.40 (4) | shell A8 8×2.5 kN |
| engine_bay | `washer_reservoir` | 3.00 (4) | shell A7 6×2.2 kN |
| controls | `brake_booster` | 4.40 (4) | shell B1 8×6.0 kN |
| controls | `steering_column` | 5.00 (3) | das 3×8.0 kN; shell B1 3×6.0 kN; ster 3×10.0 kN |
| controls | `pedal_box` | 3.20 (4) | shell B1 6×5.0 kN |
| hardware | `hinge_door_upper` ×4 | 0.80 (2) | shell B2 4×16.0 kN; door 4×16.0 kN |
| hardware | `hinge_door_lower` ×4 | 0.80 (2) | shell C2 4×16.0 kN; door 4×16.0 kN |
| hardware | `latch_door` ×4 | 0.90 (2) | shell D2 3×11.0 kN; door 3×11.0 kN |
| hardware | `door_beam` ×4 | 2.20 in its host | rides door |
| hardware | `window_regulator` ×4 | 1.80 in its host | rides door |
| hardware | `hinge_hood` ×2 | 1.10 (2) | shell B2 4×10.0 kN; hood 4×10.0 kN |
| hardware | `gas_strut` ×2 | 0.35 (2) | shell A8 3×2.0 kN; hood 3×2.0 kN |
| hardware | `hinge_trunk` ×2 | 0.90 (2) | shell E6 4×9.0 kN; trunk_lid 4×9.0 kN |
| hardware | `latch_hood` | 0.80 (2) | shell A3 3×9.0 kN; hood 3×9.0 kN |
| hardware | `latch_trunk` | 0.70 (2) | shell E4 3×9.0 kN; trunk_lid 3×9.0 kN |
| body | `undertray` | 2.60 (4) | shell A1 10×1.5 kN |
| body | `fuel_flap` | 0.25 in its host | rides body_shell |
| body | `plate_front` | 0.20 in its host | rides bumper_front |
| body | `plate_rear` | 0.20 in its host | rides body_shell |

**Checks** (`tests/mechanical.rs`, `tests/equipment.rs`, `tests/build.rs`): every new solid is closed with positive volume, above the road in every preset, and cuts through no other part or member (the pair test runs over all 116 definitions against all 171 render parts: 0 crossings). Every mount lands on real nodes or names a real panel, and every part with nodes can break loose. The four presets build the same parts. The Khronos validator reports 0 errors and 0 warnings.

**Triangle budgets** (unchanged, `docs/vehicle-art-spec.md`): body 280,762 of 300,000; wheel 14,420 of 15,000; interior 62,316 of 80,000; structure 17,292 of 40,000; mechanical 29,592 of 60,000; lamps, glass and trim 54,532 of 60,000; total 502,942 of 600,000.

**Build time:** 0.30 s native without panel UVs (was 0.28 s) and 0.65 s with them; in wasm (Node) 380 ms without UVs and 474 ms with them.

**Not added, and why**

- **Spare wheel:** the structure has no room for a well. The tank (to y 0.44 m) and the rear silencer are directly under the boot floor (y 0.47 m).
- **Radiator hoses, wiring, heater unit, horn, lamp back housings:** not modelled. Behind each front bumper corner there is only the wheel-well liner.
- **Rear seat belts.**
- **The door's inner parts are inside a closed door solid**, so they show only in a cutaway.

**Stills** (`~/Development/rubble-artifacts/sedan-complete/`, all inspected; made by `tools/vehicle-gen/tools/render_complete.sh`): `exploded_systems`, `exploded_systems_rear`, `hood_off`, `door_FL_off`, `door_RL_off`, `door_FR_off`, `door_RR_off`, `trunk_off`, `bumper_front_off`, `bumper_rear_off`, `fender_FL_off`, `wheel_FL_off`, `wheel_RL_off`, `roof_cutaway`, `underside`. `render_parts.py` gained `all` with `-part` exclusions, `systems` (the exploded view by system) and `cut@y<1.02` cutaways.

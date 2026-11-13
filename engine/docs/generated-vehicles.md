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

**Energy bombs** (hard, 2026-11-13; known-bugs S3). A car must not be able to
create energy, whatever hits it.

| Check | What fails | Where |
|---|---|---|
| `bound_wall` | A BOUNDED beam whose bound the step cannot integrate. Its limit spring acts on the whole extension once a bound is crossed, so the force steps by ΔF = \|k_limit − k\| × bound × L across the bound zone z. Fails when ΔF·h²/(m·z) > 1 for the lighter end (the force step moves the node more than the zone in one step, and ½·ΔF·bound·L appears from nothing). | `lint::energy_bombs` |
| `tether` | A beam between two parts that holds nothing at rest (no base spring), can pull, and never breaks, where every beam that does hold the two parts can break: once they part it pulls on a free part. A stop is push-only (SUPPORT, rest length = the length it engages at) or lets go. | `lint::energy_bombs` |
| Stops, statically | Every gapped stop of the rigged sedan: no force however far its ends part (to 1,000 lengths), and no step of 1 mm changes its force by more than 5 kN. | `tests/sedan_energy.rs` |
| Tear-off and fling | Every part that can leave is torn off the standing car, flung at 15 m/s and spun at 5 and 20 m/s at its rim, 1 s: the ledger never rises (2 % of the part's own kinetic energy). A wheel thrown 90 m. | `tests/sedan_energy.rs` |
| Whole-car scenes | Side pole 186 km/h, 42° wall 106 km/h, a 25 km/h bump between two sedans: never above the starting energy. | `tests/sedan_energy.rs` |

On the cars (2026-11-13): the sedan as exported at 32cbc954 fails both rules
(84 bounds over, the worst 1,592 zones a step, 6.5e9 J; 12 tethers: the front
wheels' stops). The sedan now passes (worst 0.26: the engine mounts; largest
energy step 1.3 kJ). The five community cars pass both (ETK 0.67, Vertex 0.22,
CCF 0.25, full-size SUV 0.75, boathauler 0.03 zones a step; none has a
spring-less bounded beam between separable parts). The whole-car scenes and
the fling run with the objective damping form on
(`ExplicitBeamsSettings::centred_objective`); with it off a spinning part
gains energy from its own beams' damping (the solver's, not the car's).

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

## Crash character and the frontal crashes against the ETK (2026-11-13, `feat/sedan-body`)

**State: the rigger builds the car on the body definitions** (554 nodes, 3,467 JBeam beams, 131-node shell, 48 mechanical parts; shell and parts at built / rule stiffness 1.000, ω·h 1.85; 1,447.7 kg against a kerb mass of 1,450). The "not done" list of the section above is superseded by this one.

**The setting.** `Physics.crash` (`engine_vehiclegen::physics::CrashFeel`) is the car's crash character: multiples on the definitions' mean crush forces by zone and on the panels' yield strain, applied before the lattice is built (never beam by beam). A multiple `s` on a box section's crush force is the same section in a gauge `s^0.6` times as thick.

| Preset | Front crush | Towers | Cabin | Rear | Panel yield strain | What it is |
|---|---|---|---|---|---|---|
| `beamng_feel` (default) | 1.8 (crash box 1.5 → 2.1 mm, rail 1.8 → 2.6 mm, or DP980-class steel) | 1.5 | 1.5 | 1.0 (not measured yet) | 2.0 (hood, fenders 6 %; doors 2.9 %) | the ETK's level in the same scenes |
| `real_crash` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | the members as defined; nearest the NHTSA full-frontal bands |

**Side by side** (`examples/frontal_compare`, CPU reference, surface contact on, pressure wheels, 0.6 s after contact; ETK K-series `kc6`). There is no offset or centre-pole reference in `crash-reference-targets.md`; the full wall's NHTSA bands are crush 620–750 mm, peak 34–47 g, restitution 0.13–0.20.

| Scene | Measure | Sedan as defined (`real_crash`) | Sedan `beamng_feel` | ETK (target) | Delta to the ETK |
|---|---|---|---|---|---|
| Wall 56 km/h | dynamic crush (mm) | 578 | 474 | 499 | −25 |
| | peak g / restitution | 33.4 / 0.22 | 47.7 / 0.30 | 71.9 / 0.38 | −24 g / −0.08 |
| | residual crush max / mean (mm) | 396 / 296 | 296 / 212 | 228 / 192 | +68 / +20 |
| | beams bent / broken | 438 / 0 | 373 / 16 | 335 / 4 | |
| | beams yielded; of those over 1.2 m behind the front | 15.6 % ; 6.8 % | 13.2 % ; 7.0 % | 13.7 % ; 6.4 % | |
| | mesh: stretch p99, fold p99, flipped triangles | 1.44, 12.6°, 65 | 1.32, 10.6°, 1,116 | 1.46, 27.9°, 178 | flipped: worse |
| Offset 40 %, 64 km/h | dynamic crush (mm) | 788 | 682 | 659 | +23 |
| | peak g / restitution | 39.2 / 0.17 | 46.3 / 0.26 | 47.6 / 0.18 | −1 g / +0.08 |
| | residual crush max / mean (mm) | 626 / 304 | 522 / 242 | 430 / 250 | +92 / −8 |
| | beams bent / broken | 540 / 21 | 428 / 10 | 322 / 18 | |
| | yielded; far | 19.2 % ; 10.5 % | 15.1 % ; 8.0 % | 12.8 % ; 5.0 % | far: worse |
| | mesh: stretch p99, fold p99, flipped | 1.66, 15.4°, 147 | 1.45, 15.5°, 1,063 | 1.56, 31.9°, 167 | flipped: worse |
| Centre pole 50 km/h | dynamic crush (mm) | 599 | 542 | 601 | −59 |
| | peak g / restitution | 38.1 / 0.29 | 49.9 / 0.42 | 56.6 / 0.33 | −7 g / +0.08 |
| | residual crush max / mean (mm) | 505 / 226 | 347 / 148 | 364 / 138 | −17 / +10 |
| | beams bent / broken | 407 / 20 | 336 / 4 | 287 / 3 | |
| | mesh: stretch p99, fold p99, flipped | 1.40, 9.8°, 124 | 1.36, 8.1°, 0 | 1.47, 26.9°, 110 | |

(The "as defined" column is the export before this work; it also had the 0.6 kg hub caps, the driver on the seat's back and the 6 kN seat frame.)

- **Parts leaving.** As defined, both front hub caps left in the offset and the pole. Their mass was a full wheel cover's (0.6 kg) on a 136 mm centre cap; from its geometry it is 0.12 kg and they stay. Nothing detaches on the ETK in these scenes; it breaks its driveshaft, one fender bolt group and a door glass.
- **Worse than the ETK now:** both fenders' front bolt groups break in the full wall and the left one in the offset (the ETK loses one fender group, in the offset only), which is where the flipped triangles come from; residual crush at the struck corner in the offset (+92 mm); the fenders' ripple at the end (17–46 mm; the ETK's worst parts are suspension arms at 10–45 mm); peak g in the wall (48 against 72: the ETK's front stack is stiffer than ours and than real cars).
- **Which is nearer the real bands.** In the wall the sedan as defined (578 mm, 33 g, 0.22) is nearer NHTSA's than either the ETK (499 mm, 72 g, 0.38) or `beamng_feel`.
- **Where the offset's load goes** (`examples/sedan_frontal offset`, as defined, raycast wheels, in the rear floor's own frame): cabin travel 720 mm with surface contact and 900 mm with node contact only (the barrier's edge passes between nodes; the page now turns surface contact on for generated cars). Toe board and sill front 30–38 mm, firewall 102 mm peak / 40 residual, A-pillar base 171 / 75, strut tower 176. The struck rail folds (its front 236 mm up, the subframe mount 103 mm down) more than it shortens. Plastic work: front structure 65 %, hood, fenders and bumper cover 13 %, cabin 8 %.
- **Seats.** The driver's 75 kg sits on the cushion node (a belted driver loads the lap belt's anchors, not the seat's back), and the frame and feet are sized for FMVSS 207 / 210 anchorage loads (frame 15 kN a beam, feet 12 kN yield and 45 kN break). At the study's 6 kN the driver's seat slid 18 cm forward in the wall.
- **Mass.** The raycast wheel carries the defined wheel less the wheel part's own nodes (`raycastWheelMass` 9.5 kg on the pressure-wheel row; the loader's default stays 20 kg): 1,447.7 kg. Its spin inertia follows that 9.5 kg (0.52 kg·m², about half a real wheel's): open.
- **Rest.** Parked 10 s: 0 bent, 0 broken, drift within 5 mm, vertex 0.53 mm; the parked normal is 1.72° (limit 1°) at a hub cap, not the exhaust: the cap's fit takes wheel nodes, and the wheel's frame nodes lean against its axle under the car's weight. `rest_fidelity_at_spawn_and_parked` still fails on it.
- **Side pole on raycast wheels.** A car launched sideways rides a carrier (`crash_rigs::on_carrier`, tyre grip 0.01): 31.7 km/h at the pole from a 32 km/h launch; before, its own tyres stopped it.

**Not done:** stiffness against the ETK on the new body (`sedan_stiffness`: modes, kerb wobble, settle), the head-on (`vehicle_c2c_overlap` needs the new hub names and placement by the tyre's bottom), lint rules 2, 8, 11, 14–24 and the "nothing faked" rules, matched stills against the ETK, the rear's crash character, the fender bolts.

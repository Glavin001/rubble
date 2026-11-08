# Generated vehicles: design in, crash-ready car out

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
| Lattice | 739 nodes, 491 collision triangles, ω·h 1.918, 1,450 kg, CoM 1,176 mm behind the front axle | |
| Rest fidelity at spawn / parked 10 s | 0.26 mm, 0.06° / 0.52 mm, 0.71° | ≤ 1 mm, 1° |
| Rest drift, every part / broken, bent | ≤ 3.0 mm / 0, 0 | ≤ 5 mm / 0, 0 |
| Lint hard rules | mechanism, mass, hardware pass; stability 1.926 (warn, bound 1.92 without the wheels' hub mass); skin coverage p99 0.136 m (just over 0.127) | |
| Drive, raycast wheels | 213 km/h top speed (drive setting 237), 0–100 km/h about 7 s; coast-down 1.9 kN at 209 km/h | not drag-capped |
| Wall 56 km/h: dynamic crush, t_m | 707 mm, 76.5 ms | 620–750, 63–76 |
| peak g (100 Hz, rear cabin), residual max / avg | 31.3 g, 515 / 301 mm | 34–47, 380–580 / 261–512 |
| Campbell b1, restitution, feet intrusion | 44.3 /s, 0.255, 176 mm | 22.9–39.2, 0.13–0.20, 1–67 |
| broken beams, energy ledger | 1 (a hinge group), closes (largest rise 12 J) | |
| Mesh after the wall: stretch p99, fold p99, seams > 5 cm | 1.37, 6.3°, 81 of 2,770 pairs | ≤ 1.6, ≤ 31° |
| Side pole 32 km/h 75°: residual / dynamic | 64 / 158 mm (the car slides along the pole) | 297–445 residual |
| Pole sweep 90°, 24 offsets | 1 passed through | 0 |
| MDB 214 residual | 376 mm | 126–206 |
| Roof FMVSS 216a | 4.84 × weight | 4.48–5.32 |
| Door FMVSS 214 initial / intermediate / peak | 12.5 / 15.6 / 53.5 kN | ≥ 10.0 / 15.6 / 28.5 |
| Head on, sedan into sedan, 50 + 50 km/h | dynamic crush 586 / 451 mm, residual 438 / 390 mm, 13 / 9 beams broken, both stop | (≈ the 50 km/h wall) |

Clips (CPU reference, frame-locked; crashes at 7.5× slow motion):
`~/Development/rubble-artifacts/vehicle-gen-physics/sedan-{drive,wall,pole,c2c,wall-orbit,c2c-orbit}.mp4`,
stills in `stills/`. After the wall the hood buckles, the fender tears at the
arch, the front doors gap at the A-pillar and the cabin stays whole.

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
- **Collision skin:** p99 just over the coverage rule at the bumper corners by
  the arches and the fender's lower rear corner; one pole-sweep pass-through.
- **Lattice from the design.** Member stations and heights are still sedan
  numbers (rails, floor, roof stations): derive them from the key lines so
  any proportion (the designer's sliders, the SUV) rigs right.
- **Census bands** the lattice misses by design: nodes per part and node mass
  (about 4× the census's nodes), k·L at the step's bound, body mass share
  (the named masses ride the body).
- **Mod size:** the Rust sedan's mod is about 85 MB, mostly the `.rskin`
  (308k render vertices).
- **On the GPU page:** load and record `softbody.html?mod=/__mods/sedan` and a
  World scene with rigid bodies (not done here; the clips are CPU reference).

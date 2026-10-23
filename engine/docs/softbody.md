# Soft-body vehicles (node–beam, Rigs of Rods / BeamNG import)

Deforming, breakable vehicles in the AVBD solver. A vehicle is a node–beam network:
3-DOF particles joined by plastic, breakable beams. Rigs of Rods `.truck` files and
BeamNG `.jbeam` vehicles import directly. Particles and beams are solved on the GPU
in the same step as rigid bodies, joints, contacts and raycast vehicles, from one
Slang source (`slang/kernels/soft_step.slang`), on native wgpu and in Chrome WebGPU.

```bash
cd engine
cargo test -p engine-softbody                                  # importers, model, colouring, CPU reference
cargo test -p engine-webgpu --release --test softbody          # GPU: rest, sleep, CPU parity, drive, crashes
cargo run -p engine-webgpu --release --bin wgsl_export         # WGSL + wasm for the browser
python3 -m http.server 8791 --directory crates/engine-webgpu   # Chrome → http://127.0.0.1:8791/softbody.html
node --experimental-websocket crates/engine-webgpu/softbody-check.mjs   # headless Chrome end-to-end check + screenshots
cargo run -p engine-webgpu --release --bin soft_bench          # scale benchmark
cargo run -p engine-webgpu --release --bin soft_calibrate       # calibration harness
```

The demo (`softbody.html`) offers a drivable crash yard and four crash tests: a rigid
wall at 54 km/h, a 20 cm pole at 40 km/h, a 1,500 kg side impact at 40 km/h, and a
head-on collision at 2 × 29 km/h. You can load your own `.truck` file or a BeamNG
vehicle folder (`.jbeam` parts plus an optional `.pc`). Keys: `V` shows the beams
coloured by load (orange once yielded), `M` switches between the flexbody skin and
the collision skin, `B` fires a ball, `T` launches the side ram.

## Layout

| Path | Role |
|---|---|
| `crates/engine-softbody` | Host side (no GPU): `truck` (RoR loader), `jbeam` (JBeam loader), `desc` (the engine-neutral `SoftVehicleDesc`), `model` (beam energy, plasticity, breaking), `color` (spawn colouring), `pack` (GPU records, rigid core), `host` (`SoftHost`: arena allocation and uploads), `flexbody` (skinning), `cpu` (reference solver), `calibrate` (metrics), `procedural` (original test content) |
| `slang/kernels/soft_step.slang` | The kernels (below) |
| `crates/engine-exec/src/rigid_plan.rs` | `SoftCounts` and the soft passes in `encode_step` |
| `crates/engine-webgpu` | `WgpuSim::new_with_soft`, `add_soft_vehicle`, `read_soft_beams`, `poll_beam_breaks`; `tests/softbody.rs`; `softbody.html`, `soft.mjs`, `softbody-check.mjs`; `soft_bench`, `soft_calibrate` |
| `crates/engine-vehicle-wasm/src/soft.rs` | `WasmSoftHost`, `soft_load_truck`, `soft_load_jbeam`, flexbody meshes for the browser |

## Architecture

**Particles live in `gBodies`.** A node is a `BodyState` with `BODY_PARTICLE` (64),
zero inverse inertia and an identity orientation. It keeps every rigid stage:
prediction, broadphase, node-sphere contacts, CSR, velocity extraction and sleep. It
is left out of the rigid colour lists (`count_color_tiles`, `scatter_color_tiles`,
`pack_color_lists`), so the rigid primal never solves it.

**One extra binding.** Beams, colour lists, adjacency, triangles, contact slots and
settings live in one `float4` arena, `gSoftData`, at binding 32 (`layout.rs` documents
every region). The particle kernel binds 9 storage buffers, under Chrome's limit of 10.
The rigid kernels are unchanged.

**Particle colours are computed at spawn.** A vehicle's beam graph only loses edges
(breaks), so a greedy colouring at spawn stays valid. Colour k of every vehicle is
dispatched together, so dispatches per iteration equal the largest vehicle's colour
count (8 for the test car). Particle–rigid contacts and joints are evaluated from both
sides, Newton-style. Each iteration sweeps the rigid colours, then the particle colours.

**The particle block reuses the rigid accumulation.** `soft_primal` calls
`primalAccumulate` for the particle's inertia, contacts and joints, keeps the linear
3×3 block (rotation is frozen), adds its beams and soft contacts, and solves in closed
form. Near-singular blocks are skipped, as in VBD, and the same trust region as rigid
bodies applies.

**Rigid core (stage 1).** Each vehicle with wheels gets a rigid core: a `VehicleDesc`
with Rubble's raycast wheels, drivetrain, anti-roll bars, speed-sensitive steering and
traction control. It is attached by D6 ball joints to the wheels' axle nodes only, so
all the bodywork stays free to crumple. The core takes a share of every node's mass
(25%), so its inertia follows the vehicle's mass distribution. The wheel queries
ignore the vehicle's own node spheres: the core and its nodes share a negative
collision group.

### Step placement

```text
vehicle_input → soft_wake → … broadphase, narrowphase, CSR, colouring …
→ predict_bodies → vehicle_prepare → soft_begin → [soft_bounds → soft_collide_nodes → soft_collide_tris]
→ per iteration { rigid colours → soft_primal_fused (or soft_primal × colours → soft_chebyshev)
                  → solve_dual → [soft_dual] → vehicle_iterate }
→ soft_plastic → soft_break_groups → extract_velocity_sleep → vehicle_post
→ soft_sleep → sleep gate → soft_sleep → sleep commit
```

| Kernel | Does |
|---|---|
| `soft_wake` / `soft_sleep` | A soft body (and its core) sleeps and wakes as a whole |
| `soft_begin` | Hydro rest lengths from the steering input, break-group flags cleared, each particle's initial guess (VBD adaptive: x⁻ + h v + h² g ã) |
| `soft_bounds`, `soft_collide_nodes`, `soft_collide_tris` | Soft contact detection at the initial guess (below) |
| `soft_primal` | One particle colour: the 3×3 VBD block solve |
| `soft_primal_fused` | Every colour plus the Chebyshev step in one 256-lane workgroup, when each colour fits one pass. Same Gauss–Seidel order and bit-identical results (`ENGINE_SOFT_UNFUSED=1` forces the per-colour path); saves one dispatch per colour per iteration |
| `soft_chebyshev` | Chebyshev acceleration of the sweep (ρ = 0.9), skipped for particles with any contact or joint |
| `soft_dual` | AVBD finite-stiffness ramp (and the multipliers of beams flagged hard; no importer sets the flag). Planned only when some beam ramps |
| `soft_plastic`, `soft_break_groups` | Post-step plasticity and breaking at the converged state, then detacher groups |

## Beam model

With d = xᵢ − xⱼ, ℓ = |d| and d̂ = d/ℓ, a beam stores E = ½k(ℓ − L)². Its gradient is
k(ℓ − L)d̂. Its Hessian is k d̂d̂ᵀ + max(0, k(ℓ − L)/ℓ)(I − d̂d̂ᵀ); the clamp keeps it
positive semi-definite in compression. The axial dashpot is (c/h)(ℓ − ℓ⁻) with Hessian
(c/h) d̂d̂ᵀ, where ℓ⁻ comes from both particles' step-start positions, so no state is
stored. Kinds:

| Kind | Behaviour | Sources |
|---|---|---|
| Normal | Elastic, yields and breaks (RoR rules below) | RoR beams; JBeam NORMAL, ANISOTROPIC and PRESSURED (the last two simulated as NORMAL) |
| Support | No energy when stretched (damping × 0.1); breaks past L·factor | RoR `s`, JBeam SUPPORT |
| Rope | No energy when compressed | RoR `r` |
| Bounded | Spring-damper plus a stiff limit spring outside [L(1 − short), L(1 + long)]; never deforms | RoR shocks / shocks2 / shocks3 (in-rates), JBeam BOUNDED (`beamLimitSpring`) |
| Hydro | L = L₀(1 − steer·factor), never deforms | RoR hydros (commands as fixed hydros), JBeam HYDRO with `inputSource: steering` |

Plasticity and breaking follow Rigs of Rods' rules. Compression past the threshold
moves L so the beam sits on it; the threshold hardens by L_old/L_new. Tension is
symmetric and weakens the break strength by the plastic stretch. |F| past the strength
breaks the beam. A broken beam with a detacher group breaks its whole group and writes
`EVENT_BEAM_BREAK` (5) records to the shared event ring (`WgpuSim::poll_beam_breaks`,
`world.pollBeamBreaks`).

## Collision

* **Node spheres** (radius 0.06 m) go through the existing broadphase, narrowphase and
  contact persistence against rigid bodies, terrain and other vehicles' nodes. The
  node's block takes the linear part of the contact rows.
* **Node–triangle** contacts between soft bodies. Each contacter node finds the closest
  collision triangle of another vehicle that it is in front of (prism test, step-start
  side). Each triangle collects up to four nodes that chose it. With two, a crushed
  front pressing many nodes into one panel lost most of the triangle side, and the cars
  overlapped visibly. Normals and
  barycentrics are frozen for the step; the gap is re-evaluated every iteration.
* **Triangle–obstacle** contacts against thin static spheres and capsules registered
  with `add_soft_obstacle`. These are the poles and bollards narrower than the node
  spacing that would otherwise pass between nodes.
* Soft contacts are penalties (10⁷ N/m) with lagged Coulomb friction, stabilised like
  AVBD's rigid contacts: only (1 − α) of an overlap present at the step start is
  corrected per step. Without that, a node freed inside a crushed panel was fired out
  at about 35 m/s. A per-particle flag skips the slot reads for particles with no soft
  contact this step.
* Not done: rigid boxes and hulls against soft triangles (walls are wide enough for
  node spheres), self-collision, edge–edge contacts, and conservative displacement
  bounds or substeps for highway-speed tunnelling.

**Soft-scene parameters.** `soft_scene_params(hz, iterations)` raises the contact
penalty ramp `beta_lin` from 10⁴ to 10⁷. A light node's contact must reach its own
m/h² within an iteration or two. At the rigid default, a dropped lattice sank into the
ground and bounced back to most of its height; at 10⁷ it lands and settles.

## Importers

**Rigs of Rods `.truck`** (`truck.rs`, written from the documented format) parses:

* sections: `globals`, `nodes`/`nodes2`, `beams`, `shocks`/`shocks2`/`shocks3`, `hydros`, `commands`/`commands2`, `cab`, `contacters`, `fixes`, `minimass`, `cameras`, `wheels`/`wheels2`/`meshwheels`/`meshwheels2`/`flexbodywheels` (axle, radius, width, braking, propulsion, mass), `engine`, `brakes`;
* directives: `set_beam_defaults`, `set_beam_defaults_scale`, `set_node_defaults`, `set_default_minimass`, `enable_advanced_deformation`, `detacher_group`, `set_collision_range`, `fileformatversion`.

It reproduces the sequential importer's implicit node numbering (wheels and cinecams
consume numbers). It also reproduces RoR's mass rules: cargo on loaded nodes, dry mass
spread by beam length including wheel spokes, then the minimass floor. Known sections
it does not simulate are skipped with one warning each.

* Steering wheels are found from steering hydros on their axle or arm nodes, or one
  beam from them (the Agora's steering links).
* In stage 1 the raycast wheel does the steering. A hydro on an axle node keeps its
  length rather than fight the rigid core that holds that node.
* The vehicle frame comes from `cameras`, resolved after the whole file, because files
  may list cameras before nodes.
* Rubble's gearbox holds 7 forward ratios with reverse. Longer gearboxes (the DAF's 13)
  keep 7 spread over the range, with a warning.

**Real Rigs of Rods content** (RigsOfRods/content, GPLv3, local only, so not in the
repo): the Agora S and L buses, the DAF semi and both its trailers import with no
unknown nodes. They settle on the ground with the CPU reference
(`examples/truck_probe.rs`). The opt-in GPU test `ror_bus_and_semi_settle_drive_and_steer`
(`RUBBLE_ROR_DIR=<content-master>`) settles the Agora S (98 nodes, 12.2 t), the Agora L
(151 nodes, 19.2 t) and the DAF semi (79 nodes, 10.1 t) on every wheel. They drive 20,
13 and 21 m in 4 s and turn, with no beam bent or broken.

**BeamNG JBeam** (`jbeam.rs`, written from the JBeam documentation, with BeamNG's
MIT-licensed editors as references):

* relaxed JSON: comments, commas as whitespace, unquoted keys, `=`, `FLT_MAX`, stray closing brackets;
* parts and slots: `slots` and `slots2`, `.pc` configurations, slot `nodeOffset`/`nodeMove`, 50 levels;
* variables, with `$=` expressions in Lua semantics;
* table expansion with modifier rows and trailing option dictionaries;
* sections: nodes, beams, triangles, quads, `pressureWheels`, `refNodes`;
* the powertrain graph, to find driven wheels;
* the frame: JBeam (+X left, +Y rear, +Z up) becomes Rubble (+X left, +Y up, +Z forward).

LBEAM, torsion bars, rails, slidenodes, flexbodies and props are counted and skipped.

**Licensing.** The design doc lists BeamNG's `agenty_legocar` and `legotrailer` as CC0.
They are not. The `tests/LICENSE.txt` of Blender-JBeam-Editor is CC0, but the vehicles
directory has its own `LICENSE.txt`: "The files in this directory are not subject to
CC0 1.0 Universal terms" (AgentY's mod, used with permission). They are therefore not
in this repo. The loader is tested on an original JBeam vehicle written in the unit
tests. The legocar runs only as opt-in local checks (`RUBBLE_JBEAM_DIR`): it loads 751
files into 757 nodes, 5,476 beams, 694 triangles and 4 wheels (2 driven) in 66 ms,
settles on its wheels and drives on the GPU and in Chrome. The procedural hatchback
(`procedural.rs`) is original Rubble content, written as `.truck` text so the fixtures
go through the real importer. It has 102 nodes, 574 beams and 186 skin triangles. Its
crumple zones (bonnet and boot) and door panels use the base thresholds, while the
passenger cell's floor, sills, pillars and roof have 3× the deform threshold and 2×
the strength, as RoR and BeamNG authors zone their beams. A crash therefore folds the
ends first and leaves the cell nearly intact, where uniform beams spread yield through
the whole car. It also has a bumper that comes off in one piece (only its mounts are on
the detacher group) and a cargo floor that keeps the centre of mass low.

## Visual mesh

The demo draws either the collision skin or a **flexbody** mesh (`flexbody.rs`). The
flexbody mesh is the skin subdivided once with its edges pushed out into rounded
panels. Each vertex is bound at spawn to a frame of three nodes, and the vertex shader
rebuilds it every frame from the particle positions:
p = c + a·(x − c) + b·(y − c) + d·((x − c) × (y − c)).

* `bind` is for an arbitrary mesh, such as a converted RoR or BeamNG flexbody: the
  closest node plus the two whose edges are closest to perpendicular.
* `bind_to_triangles` is for the skin-derived mesh. Each vertex binds to a skin triangle
  it lies on, and a shared edge midpoint is bound once, so both panels rebuild the same
  point and the skin stays closed.

A nearest-node frame that spanned a bumper and the body sheared when the bumper
tore off and threw a vertex out as a spike. A triangle with a frame stretched past 1.8×
drops out, which reads as a tear. Panels darken with strain. The beam view colours
intact beams by load and deformed ones by plastic strain |L/L₀ − 1| (yellow, then red at
8%), so a millimetre of yield does not look like wreckage.
Mesh-file flexbodies (RoR OGRE `.mesh`, BeamNG `.dae`) need an offline converter to
glTF that is not built yet; the binding and the skinning shader are what it would feed.

## Validation

| Test | Shows |
|---|---|
| `engine-softbody` unit tests (28) | Section parsing, defaults and RoR mass rules; numbering; gradient and Hessian against finite differences; PSD in compression; yield, hardening, weakening, supports, ropes, bounded; colouring validity; relaxed JSON, Lua expressions, slots, variables, wheels; JSON round trip of infinite thresholds; flexbody exactness under rigid motion; calibration metrics |
| `tests/cpu_reference.rs` | Car rests without yielding; convergence check (sag error 150% → 2% → 0% with iterations, and closer at 240 Hz than at 60 Hz); a 54 km/h wall crash crumples and sheds the bumper |
| `tests/softbody.rs` (GPU) | Lattice lands, keeps its shape and sleeps as a whole; cantilever sag on the GPU matches the CPU reference (0.01678 vs 0.01674 m); car settles on four raycast wheels with the suspension carrying its weight; drives 9.7 m in 3 s and steers, with no yielding; wall crash crumples the front (12 bumper-mount breaks, matching ring events); wall, pole and side impact crush differently; head-on cars crumple without passing through |
| `softbody-check.mjs` (Chrome) | Every demo scene steps, checks and saves a screenshot to `recordings/softbody/`. The browser matches the native run: wall 229–236 beams bent with every mount break reported, head-on symmetric |

**Damage character.** Front residual crush, inward fold ("wrap") of the front ring, and
side intrusion, in cm; raw file thresholds at 240 Hz × 10 iterations:

| Impact | Bent beams | Front crush | Front wrap | Struck / far side intrusion | Mid-car crush |
|---|---|---|---|---|---|
| Rigid wall, 54 km/h | 209 | 11.0 | −0.1 | 0.5 / 0.5 | 0.0 |
| 20 cm pole, 40 km/h | 90 | 0.6 | 3.6 | 0 / 0 | 0.0 |
| 1,500 kg block into the side, 40 km/h | 108 | 0.2 | 0 | 1.9 / −0.1 | — |

The demo applies the calibrated threshold scale (below), so the same impacts bend more
and crush deeper there. The head-on at 2 × 29 km/h, for example, bends 242 beams per car.

## Calibration

`calibrate.rs` measures:

* static sag (RMS residual after a rigid best fit);
* the tap-test frequency (DFT of a structural length after a kick);
* residual crush (RMS and max);
* the broken-beam Jaccard index;
* the share of kinetic energy absorbed;
* a golden-section fit of one global setting.

`soft_calibrate` runs the standard tests (rest, tap, 1 m drop, wall at 30 and 60 km/h)
at candidate settings against a reference. No Rigs of Rods recording is available, so
the reference is Rubble at 960 Hz × 30 iterations, a stand-in for a 2,000 Hz explicit
engine; this is the design's convergence check. Results on the procedural car with crumple zones (M3 Max):

| Setting | Tap test (Hz) | Crush 30 km/h RMS / max (cm) | Crush 60 km/h RMS / max (cm) |
|---|---|---|---|
| reference 960 Hz × 30 | 44 | 1.7 / 5.1 | 8.4 / 22.6 |
| 120 Hz × 10 | 15 | 2.3 / 5.8 | 5.6 / 14.3 |
| 240 Hz × 10 | 23 | 1.4 / 4.4 | 4.5 / 12.0 |
| 240 Hz × 20 | 25 | 1.3 / 4.2 | 4.0 / 11.3 |
| 480 Hz × 10 | 36 | 1.7 / 4.7 | 6.4 / 17.1 |

At every setting:

* static sag is 0.65 mm RMS;
* a 1 m drop bends at most one beam;
* the broken-beam set matches the reference (Jaccard 1.0);
* the car stops (100% of kinetic energy absorbed).

The tap test shows backward Euler lowering a stiff structure's frequency at coarse
steps. The discrete frequency is atan(ωh)/h: about 22 Hz for a 44 Hz mode at 120 Hz,
before under-convergence. Only step rate (and iterations) close that gap; no threshold
setting does.

Two effects meet in crush depth. Under-convergence makes a coarse step softer;
backward Euler's dissipation and lower peak forces make it crush less. The second
wins, as the design predicted, so thresholds need scaling down at coarse steps. One
global threshold scale per step rate brings the 60 km/h crush back to the reference:
0.45 at 120 Hz (23.3 cm) and 0.30 at 240 Hz (20.6 cm), unchanged by the crumple zones. `SoftSettings::calibrated_for(hz)`
interpolates these; the demo uses it, and `?threshold=1` keeps raw file values. With
it, everyday driving still bends no beams. These values come from one vehicle; the
design's 15-vehicle set and Rigs of Rods reference runs remain to do.

## Performance

`soft_bench` drives N procedural cars at 240 Hz × 10 iterations, all awake (Apple M3
Max, wgpu/Metal, pipelined submission). `HZ`, `ITERS`, `SWEEPS`, `CHEB` and `CONTACTS`
override the solver settings, and `RUBBLE_GPU_TIMING=1` prints GPU time per phase.

| Cars | Particles | Beams | ms / step | ms / 60 fps frame (4 steps) | Rigid-only baseline, same cars |
|---|---|---|---|---|---|
| 1 | 102 | 572 | 3.0 | 12.1 | 1.9 ms / step |
| 10 | 1,020 | 5,720 | 3.6 | 14.4 | |
| 50 | 5,100 | 28,600 | 4.0 | 16.0 | 2.0 ms / step |

GPU time per phase, 10 cars, 240 Hz × 10 (ms):

| Phase | ms | Per step |
|---|---|---|
| broadphase, narrowphase, colouring | 0.28 | once |
| `soft_begin` (initial guess, hydros; soft contacts when present) | 0.20 | once |
| rigid primal | 0.05 | per iteration |
| particle sweep (`soft_primal_fused`) | 0.17 | per iteration |
| rigid dual and vehicle solve | 0.12 | per iteration |
| `soft_plastic` and break groups | 0.01 | once |

Each solver pass waits on the one before, so the cost is per-dispatch latency rather
than arithmetic. One car's sweep costs 0.135 ms and ten cars' 0.17 ms, so 50 cars cost
only 1.0 ms more per step than one. Chrome WebGPU runs the yard (3 cars and 3 crates:
387 particles, 2,190 beams) at 5.2 ms per 240 Hz step.

### Accuracy against cost

`soft_calibrate` with `SETTINGS=HZxITERS,...` and `SWEEPS`: metrics as in
Calibration, raw thresholds, against 960 Hz × 30. The cost is `soft_bench` at 10 cars.

| Setting | Tap (Hz) | 1 m drop: beams bent | Crush max 30 / 60 km/h (cm) | ms / step | ms / 60 fps frame |
|---|---|---|---|---|---|
| reference 960 Hz × 30 | 44 | 0 | 5.1 / 22.6 | | |
| 60 Hz × 10 | 4 | 16 | 12.6 / 42.3 | | |
| 60 Hz × 20 | 6 | 0 | 3.4 / 15.7 | 6.34 | 6.3 |
| 120 Hz × 5 | 3 | 33 | 11.8 / 30.1 | | |
| 120 Hz × 10 | 15 | 1 | 5.8 / 14.3 | 3.45 | 6.9 |
| 120 Hz × 20 | 15 | 0 | 3.2 / 7.1 | 6.31 | 12.6 |
| 240 Hz × 5 | 23 | 0 | 6.6 / 17.3 | 2.06 | 8.2 |
| 240 Hz × 10 (default) | 23 | 0 | 4.4 / 12.0 | 3.58 | 14.3 |
| 240 Hz × 20 | 25 | 0 | 4.2 / 11.3 | 6.62 | 26.5 |
| 480 Hz × 5 | 35 | 0 | 4.4 / 17.0 | 2.10 | 16.8 |
| 120 Hz × 5, 2 sweeps | 14 | 4 | 6.5 / 16.2 | | |
| 120 Hz × 10, 2 sweeps | 15 | 0 | 3.6 / 7.9 | | |
| 240 Hz × 5, 2 sweeps | 22 | 0 | 4.8 / 13.1 | 2.83 | 11.3 |
| 240 Hz × 3, 4 sweeps | 22 | 0 | 4.4 / 13.5 | 2.83 | 11.3 |

What the sweep shows:

* **Step rate sets the dynamics.** The tap frequency depends almost only on the step
  rate: backward Euler's atan(ωh)/h. 60 Hz is out: the structure barely rings
  (4–6 Hz against 44 Hz), and at 10 iterations a 1 m drop bends 16 beams.
* **Too few iterations reads as phantom damage.** 120 Hz × 5 bends 33 beams in a
  1 m drop; 10 iterations are the floor at 120 Hz.
* **Past 10 iterations at 240 Hz little changes.** 240 Hz × 20 is within 6% of × 10
  at almost twice the cost.
* **Extra particle sweeps buy convergence at half the price of iterations**
  (`SoftSettings::sweeps`, decision s9).
  * 240 Hz × 5 with 2 sweeps tracks 240 Hz × 10 (13.1 against 12.0 cm, 22 against
    23 Hz) at 21% less cost natively.
  * In Chrome the same setting runs the yard at 3.53 instead of 5.18 ms per step
    (−32%), and every `softbody-check` scene passes
    (`SOFT_QUERY='iters=5&sweeps=2'`).
  * The default stays 1 sweep because it halves the rigid iterations; the demo takes
    `?iters=5&sweeps=2`.

Negative results (decisions s10 and s11):

* **Hard AVBD beams fail.** Augmented-Lagrangian beams with λ carried across steps
  wind up while the sweep is unconverged, and the car flies apart at rest.
* **A multi-lane particle solve is slower.** Eight lanes per particle with a
  groupshared reduction took 3.0 → 4.85 ms per step for one car.

Levers in place, all lossless:

* the fused sweep;
* the contact-flag skip;
* skipping the no-op dual pass.

Not implemented yet:

* rigid mode with promotion;
* per-island step rates;
* per-state iteration budgets.

Against the design's milestone-2 targets (about 2 ms per frame for one car, 10 ms for
ten), the soft part alone is about 4.6 ms per frame for one car and 6.4 ms for ten. That
comes on top of the rigid step's fixed 7.5 ms per frame at 240 Hz.

## Deviations from the design

* Beam word 14 holds the tension yield threshold, not ℓ⁻. ℓ⁻ comes from the step-start
  positions, and word 15 holds the force readout for rendering.
* Soft contacts are stabilised penalties, not AVBD inequalities with λ. The design
  allows this as its fallback; the per-step frozen normal and barycentrics are as
  designed.
* The rigid core attaches at the axle nodes only, and takes a share of every node's
  mass rather than the frame nodes' summed mass.
* A triangle records at most four nodes for the triangle side of node–triangle
  contacts; further nodes still get their own side.
* The RoR cab-node exception (keep the last beams on a collision node) is not
  implemented. LBEAM, torsion bars, pressured beams and node–beam tyres are not
  simulated (JBeam PRESSURED and ANISOTROPIC run as NORMAL).
* Only the native wgpu host and the browser run soft bodies. The Metal (`MetalSim`)
  and CUDA hosts compile and plan no soft passes (`SoftCounts::default`).
* Stage-2 wheels (mounted on particles), rigid-mode demotion, crash-island substeps and
  the 15-vehicle Rigs of Rods calibration are next.

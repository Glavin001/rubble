# Soft-body vehicles: BeamNG-parity plan and progress

Goal: match BeamNG.drive's destruction quality (crush shape, folding, tearing, no
clipping) in Rubble's GPU soft-body vehicles, at higher performance and with many
vehicles at once. Tyres are out of scope: the stage-1 raycast wheels on the rigid
core stay. Builds on `engine/docs/softbody.md` (PR #54 and later decisions s1–s25).

Order: 0 → 4 → 1 → 2 → 3 → 8 → 5 → 9, with 6 and 7 as time-boxed spikes.

| # | Work item | Status |
|---|---|---|
| 0 | Measurement harness and explicit reference oracle | see §0 |
| 1 | Self-collision parity and persistent anti-clip | see §1 |
| 2 | Penetration-free contact (OGC displacement bounds) | see §2 |
| 3 | Missing JBeam beam types and rules | see §3 |
| 4 | Float-precision safety (vehicle-local frames) | see §4 |
| 5 | Crash islands at a high step rate | see §5 |
| 6 | Lower-dissipation integrator spike (BDF2 / TR-BDF2) | see §6 |
| 7 | JGS2 solver spike | see §7 |
| 8 | BeamNG-resolution vehicles and scale | see §8 |
| 9 | Calibration against BeamNG / the oracle | see §9 |

Where the plan and the existing engine already overlap:

* Self-collision exists (s18): a node against its own body's triangles, excluded by
  rest distance (12 cm). Phase 1 adds BeamNG's per-node flags, group exclusion,
  persistent anti-clip, triangle activation, thickness and bias, and breaking.
* BDF2 exists as an experimental setting (s23): tap 36 Hz at 240 Hz but the 60 km/h
  crush blew up. Phase 6 starts from that result.

## Acceptance criteria (from the plan)

* §0: `soft_calibrate` prints every metric for every setting against the explicit
  oracle; CI runs the CPU-only parts (oracle, metrics) on the small car.
* §1: penetration count 0 at rest, driving, and the 56 km/h wall on both test cars;
  self-intersection near 0 in wall, pole, T-bone and head-on; the engine block does
  not pass the firewall; self-collision ≤ 25% of the soft step on the high-res car.
* §2: 100 km/h wall, 2 × 80 km/h head-on and 60 km/h side pole: penetration count 0;
  overhead ≤ 15% of the soft step.
* §3: finite-difference gradient/Hessian and PSD tests for every new element; oracle
  and GPU agree per element; the opt-in legocar skips only tyres, flexbodies, props.
* §4: the same wall crash at the origin and at (10 km, 0, 10 km) crushes within 1 cm.
* §5: crash islands ≥ 38 Hz tap; crush profile within 2 cm of the oracle for wall
  56/100, pole and head-on; non-crashing cars cost unchanged.
* §8 (Chrome WebGPU, mid-range GPU): 1 high-res car crashing at 60 fps; 20 driving +
  2–4 crashing at 60 fps; 100 mostly rigid cars real time.

Rules that bind every phase (AGENTS.md): no lossy change on our own initiative; new
limits are explicit engine-wide settings that fail loudly; third-party vehicles stay
local (`RUBBLE_ROR_DIR`, `RUBBLE_JBEAM_DIR`); every new pass fuses into an existing
kernel where possible (dispatch latency dominates).

## Progress log

Entries are appended as each step lands, with measured numbers.

### Phase 0 — measurement harness and reference (done)

* `explicit_ref` (oracle), `metrics` (penetration and self-intersection counts with a
  side cache that also catches crossings between samples, intrusion, crush profile,
  engine-block travel, time to stop, peak deceleration), `hires` (an original 1,014-node,
  5,774-beam, 1,600-triangle JBeam car with groups), and `soft_calibrate` on all of them
  (`CAR=sedan|default|hires`, `OFFSET=` for distance from the origin). CI runs the oracle
  and metrics tests (`cargo test -p engine-softbody`).
* Sedan, no rigid core (DECISIONS s26):

| setting | tap (Hz) | crush 30 / 56 / 100 km/h max (cm) | 56 km/h profile (cm) | peak (g) | self-penetrations |
|---|---|---|---|---|---|
| oracle (explicit) | 46 | 10.2 / 25.6 / 73.4 | 20 15 19 | 49 | 2 |
| 240 Hz × 10 (before) | 25 | 8.8 / 40.6 / 115.7 | 32 34 32 | 36 | 44 |
| 960 Hz × 30 (before) | 43 | 10.1 / 24.0 / 73.8 | 19 16 20 | 47 | 24 |

### Phase 4 — float precision (done, DECISIONS s28)

* 240 Hz × 10 at (10 km, 0, 10 km), before: crush 18.2 / 58.2 / 146.7 cm against
  8.8 / 40.6 / 115.7 at the origin. After: 9.5 / 37.8 / 105.9 against 8.6 / 36.5 / 107.4.
* `tests/soft_precision.rs`: RMS crush within 0.1 mm, deepest node within 1.3 cm (world
  f32 contacts with the ground and wall).

### Phase 1 — collision parity (in progress, DECISIONS s27)

Done: lone-vehicle self-collision, persistent anti-clip side cache with two-sided
triangles, BeamNG node groups and `selfCollision`, triangle bias, `NONCOLLIDABLE`,
triangle break on edge beam or `breakGroup`. Sedan after phases 1 and 4:

| setting | crush 30 / 56 / 100 km/h max (cm) | 56 km/h profile (cm) | peak (g) | self-penetrations |
|---|---|---|---|---|
| 240 Hz × 10 | 8.6 / 36.5 / 107.4 | 31 29 31 | 38 | 19 |
| 960 Hz × 30 | 10.3 / 22.8 / 68.8 | 20 15 20 | 48 | 5 |

Then (DECISIONS s29): four contacts per node from the side cache, pairs cached on
approach, overlap past the face pushed out at ≥ 1 m/s:

| setting | crush 30 / 56 / 100 km/h max (cm) | 56 km/h profile (cm) | rebound (m/s) | intrusion (cm) | self-penetrations |
|---|---|---|---|---|---|
| oracle | 10.2 / 25.6 / 73.4 | 20 15 19 | 5.3 | 6.1 | 2 |
| 240 Hz × 10 | 9.6 / 36.6 / 113.9 | 20 23 22 | 4.6 | 4.2 | 5 |
| 960 Hz × 30 | 10.3 / 22.5 / 74.8 | 20 20 22 | 5.8 | 5.1 | 7 |

Open: triangle activation pre-pass, rigid hulls against soft triangles, `staticCollision`,
visual tearing in the skinning kernel matching collision tearing, self-collision cost on
the high-res car.

### Phase 2 — penetration-free contact (implemented, off by default; s29)

`SoftSettings::displacement_bound` (γ): OGC-style bounds along each contact's normal,
from the detection position. Penetrations, max over the run (`soft_anticlip.rs`
probes):

| scene | oracle (6.3 kHz) | 240 Hz × 10, γ = 0 | 240 Hz × 10, γ = 0.45 | 960 Hz × 30, γ = 0 |
|---|---|---|---|---|
| wall 56 km/h | 2 self | 5 self | 13 self | 7 self |
| wall 100 km/h | 7 self | 33 self | 25 self | 23 self |
| head-on 2 × 80 km/h | 14 ext + 8 self | 1 ext + 35 self | 12 ext + 40 self | 0 ext + 33 self |
| side pole 60 km/h | — | 5 self | 5 self | 4 self |

The plan's "penetration count = 0" is not met by the penalty contact model, the
explicit reference included. The remaining cases are nodes already behind a face at
the step start under crash loads.

### High-resolution car (phase 8 input)

`CAR=hires` (1,014 nodes, no rigid core):

| setting | crush 30 / 56 / 100 km/h max (cm) | 56 km/h profile (cm) | self-penetrations | 1 m drop: bent |
|---|---|---|---|---|
| oracle (10.4 kHz) | 8.3 / 53.0* / 78.7 | 15 15 20 14 18 | 3 | 115 |
| 240 Hz × 10 | 71.6 / 150 / 215 | ~190 | 251 | 517 |
| 960 Hz × 30 | 7.4 / 20.5 / 77.2 | 15 13 21 15 18 | 96 | 63 |

\* one torn shell node; the profile agrees with 960 Hz × 30. At production settings a
BeamNG-resolution car is badly under-converged (it collapses), so phases 5, 7 and 8
carry the high-res target.

### Phase 3 — JBeam features (DECISIONS s30)

Done: LBEAM, torsion bars, rails and slidenodes (capped, looped, tolerance, strength,
cap strength, coming off), ANISOTROPIC (`springExpansion`, `dampExpansion`), and the
fix that lets `deformLimit` / `deformLimitExpansion` reach the solver. Each element has
finite-difference gradient and PSD tests (`elements.rs`), oracle behaviour tests
(`elements_oracle.rs`) and GPU-vs-oracle static tests (`soft_elements.rs`, < 0.1 mm).

Then: `deformLimitStress` (ceiling on hardening), `breakGroupType` 1 (breaks with
its group without setting it off) and `precompressionRange` (spawn length change in
metres), on the CPU, oracle and GPU; anisotropic beams' expansion rates now read back.

Open: `beamPrecompressionTime` (a gradual spawn change; needs a per-step rest ramp),
the 2 cm minimum deformed length (RoR's 0.1 m floor stays), the anisotropic
`transitionZone`, and checking the legocar's skipped sections (needs
`RUBBLE_JBEAM_DIR`, not available here).

### Phase 8 — BeamNG resolution and scale (measured; DECISIONS s31)

`soft_bench` (M3 Max, native, 240 Hz × 10, all cars soft and driving):

| car | 1 car | 10 cars | 40 cars |
|---|---|---|---|
| sedan (161 nodes) | 2.11 ms/step | 2.11 | 2.79 |
| high-res (1,014 nodes) | 4.83 | 6.19 | 10.03 |

Self-collision on the high-res car: detection 0.96 → 0.50 ms per step after the
lane-parallel `soft_collide_tris`; with the solve's contact terms it is about +1.4 ms
on a 3.65 ms soft step without soft contacts (the plan's target is ≤ 25%).
At these settings the high-res car does not converge (see the phase 0 table); that,
not throughput, blocks the plan's high-res targets.

### Phase 5 — crash islands (implemented, off by default; DECISIONS s31)

`SoftSettings::crash_substeps`. Sedan without a core, 240 Hz × 10:

| setting | tap (Hz) | crush 30 / 56 / 100 km/h max (cm) | 56 km/h profile (cm) | peak (g) | rebound (m/s) |
|---|---|---|---|---|---|
| oracle | 46 | 10.2 / 25.6 / 73.4 | 20 15 19 | 49 | 5.3 |
| 1 substep | 25 | 9.6 / 36.6 / 113.9 | 20 23 22 | 40 | 4.6 |
| 2 substeps | 35 | 6.9 / 39.9 / 68.4 | 19 19 15 | 48 | 5.1 |
| 2 substeps, 12 iterations | 37 | 6.0 / 34.4 / 80.5 | 20 19 5 | 49 | 5.3 |

The core coupling is fixed (DECISIONS s32: joint and contact rows read both bodies
at the same fraction of the step), so islands now work on cars with a rigid core.
Still off by default: with 2 substeps the sedan loses its tuned 20–35 g crumple
plateau (13 g), so the procedural cars need re-tuning first.

### Phase 6 — BDF2 against the oracle (spike, s23 confirmed)

Sedan without a core, 240 Hz × 10 with `BDF2=1`: tap 35 Hz (25 with backward Euler,
oracle 46), crush 8.9 / 35.0 / 200 cm (oracle 10.2 / 25.6 / 73.4), stop 1.40 s. The
100 km/h crash still blows up, as s23 found against the old reference. TR-BDF2 and
JGS2 (phase 7) are not started.

### Phase 9 — calibration against the oracle (fit recorded, not applied)

`FIT=1`: one global threshold scale per rate to match the oracle's 56 km/h crush
depth: 240 Hz × 10 → 1.36 (25.1 cm against 25.6); 120 Hz × 10 → 2.35 still crushes
57 cm. `SoftSettings::calibrated_for` is unchanged (1.0 from 240 Hz): scaling every
threshold changes the material, a trade-off for the user to choose. BeamNG.tech
recordings and the 15-vehicle set are not available here.

### Matching Rigs of Rods: RoR's step on the GPU (2026-11-01, DECISIONS s34)

The implicit solver's gap to RoR is in its model (converging it further moves it
away), so Rubble now runs RoR's own step: `engine_softbody::ror` on the CPU (exact
twin of RoR's code, the CI reference) and `RorGpu` (`slang/kernels/ror_step.slang`)
on the GPU. `RorSim::from_truck` spawns a published `.truck` as RoR does, soft tyres
included. Published Agora buses on their tyres, 56 km/h into a wall
(`soft_calibrate REF=ror SETTINGS=ror,240x10 CAR=truck:<file>`):

| bus | solver | sag (mm) | crush 30 / 56 / 100 km/h max (cm) | 56 km/h profile (cm) | peak (g) |
|---|---|---|---|---|---|
| Agora S | Rigs of Rods | 17.64 | 5.6 / 13.1 / 141.4 | 11 0 4 11 | 44 |
| | Rubble GPU (RoR step) | 17.65 | 5.6 / 13.1 / 144.5 | 11 0 4 11 | 44 |
| | Rubble implicit 240 Hz × 10 | 11.95 | 3.3 / 22.7 / 184.0 | 19 0 7 19 | 33 |
| Agora L | Rigs of Rods | 23.38 | 12.7 / 32.3 / 186.7 | 31 12 0 4 14 | 21 |
| | Rubble GPU (RoR step) | 23.93 | 12.4 / 30.6 / 236.8 | 28 11 0 4 16 | 21 |
| | Rubble implicit 240 Hz × 10 | 13.88 | 21.0 / 44.9 / 250.8 | 26 0 0 12 26 | 21 |

(The 100 km/h maximum is one node in a chaotic rebound; the profiles agree.)

Cost per 60 fps frame (33 steps), M3 Max (`ror_gpu_check bench`):

| | 1 | 10 | 40 | 100 | 400 | 1000 |
|---|---|---|---|---|---|---|
| Agora S, RoR's thread pool / 1 thread | 1.78 / 0.27 | 1.90 / 2.81 | 4.03 / 12.7 | | | |
| Agora S, Rubble GPU | 0.57 | 0.56 | 0.56 | 0.88 | 3.4 | 8.3 |
| Agora L, RoR's thread pool / 1 thread | 1.80 / 0.39 | 2.10 / 4.09 | 5.21 / 18.7 | | | |
| Agora L, Rubble GPU | 0.64 | 0.64 | 0.65 | 1.29 | 5.2 | 13.0 |

Videos (side by side, RoR's code and Rubble's GPU run): `examples/ror_video.rs` from
`ror_gpu_check trace=` and the harness's `trace=`.

Open: wheel torque and engine (`CalcWheels`), vehicle–vehicle collisions, coupling
`RorGpu` vehicles with Rubble's rigid world (the implicit path keeps that role), and
the single-vehicle latency floor (0.57 ms against RoR's single-threaded 0.27 ms).

### Against Rigs of Rods' own solver, on published RoR content (2026-11-01)

Supersedes the "Rigs of Rods-style solver" section below as the reference: that one
re-implemented RoR with our own beam rules. Now RoR's unmodified code is the oracle
(local harness `inspiration/ror-harness`: its parser compiled as is, its spawner,
mass, beam, node and collision functions copied verbatim at build time; graphics and
sound stubbed; fixed 0.5 ms step). `soft_calibrate REF=ror ROR_BIN=... CAR=truck:<file>`
runs both on the same unmodified `.truck` (RigsOfRods/content, GPLv3, local only).
Rubble runs its production model (rigid core with 25% of the mass, raycast wheels);
RoR its soft tyres. Coasting into a 1 m thick wall, M3 Max:

| vehicle | solver | crush 30 / 56 / 100 km/h max (cm) | 56 km/h profile (cm) | peak (g) | rebound (m/s) |
|---|---|---|---|---|---|
| Agora S (98 nodes + RoR tyres) | Rigs of Rods | 5.5 / 13.2 / 141.8 | 11 0 0 11 | 44 | 7.6 |
| | Rubble 240 Hz × 10 | 3.2 / 22.7 / 183.1 | 19 0 0 19 | 33 | 7.1 |
| | Rubble 960 Hz × 30 | 4.0 / 27.4 / 218.1 | 23 0 0 23 | 35 | 7.5 |
| Agora L (151 nodes) | Rigs of Rods | 12.9 / 32.5 / 186.7 | 31 12 0 6 15 | 21 | 7.1 |
| | Rubble 240 Hz × 10 | 19.5 / 45.0 / 243.6 | 26 0 0 0 26 | 21 | 5.5 |
| | Rubble 960 Hz × 30 | 20.7 / 52.9 / 239.7 | 28 0 0 0 28 | 21 | 6.0 |

Cost per 60 fps frame (RoR: its step on the CPU, 33 steps per frame, without
inter-vehicle collision, so a lower bound; "pool" = RoR's per-step thread pool on 16
threads. Rubble: `soft_bench CAR=truck:`, pipelined, machine under other load):

| | 1 bus | 10 buses | 40 buses |
|---|---|---|---|
| Agora S, RoR 1 thread / pool | 0.27 / 1.78 | 2.81 / 1.90 | 12.7 / 4.03 |
| Agora S, Rubble 240 Hz × 10 | 7.8 | 9.0 | 11.1 |
| Agora L, RoR 1 thread / pool | 0.39 / 1.80 | 4.09 / 2.10 | 18.7 / 5.21 |
| Agora L, Rubble 240 Hz × 10 | 11.0 | 12.8 | — |

Reading: Rubble is slower than RoR at every count measured, and differs from RoR's
crash more than convergence explains (converged Rubble, 960 × 30, is further from RoR
than 240 × 10). The gap is in the model, not only the solver: candidates are the rigid
core holding 25% of the mass, the wall contact (RoR's `primitiveCollision` against
penalty/AVBD contacts), node spheres, and yield inside the solve. Spawn data matches
exactly (node masses, beam parameters; after the fixes below).

Fixes found by the comparison: `minimass` is a block keyword in RoR (our cars wrote
it inline, which RoR reads as a `globals` line); our wheel-spoke mass used the
declared width instead of the axle length and left out cinecam beams (Agora S was
12,170.8 kg against RoR's 12,058.1; now equal node by node). The procedural sedan
is not valid RoR content: its 5 kg minimum node mass blows up at RoR's 2 kHz.

### Against a Rigs of Rods-style solver (2026-11-01, superseded)

`examples/ror_baseline.rs` solves our cars the way Rigs of Rods and BeamNG do:
explicit symplectic Euler in f32 at a fixed rate, the same beam rules (`model`),
nodes against ground and wall by projection, no node–triangle contacts (so a lower
bound on RoR's cost), one car per thread as RoR runs actors. M3 Max, CPU:

| car | beam-only stable rate | 56 km/h crush profile, 2 kHz | at 4 kHz | at 8 kHz |
|---|---|---|---|---|
| sedan (367 nodes, 2,170 beams) | 4.6 kHz | 23 19 17 cm, 92 broken | 18 17 16, 43 | 17 17 17, 39 |
| high-res (1,014 nodes, 5,774 beams) | 6.3 kHz | blows up (2,230 broken) | 19 17 22 21 22, 135 | 19 17 21 18 19, 110 |

Cost per 60 fps frame (ms), CPU explicit at a converged rate against Rubble's GPU at
240 Hz × 10 (`soft_bench`, with rigid core, wheels and self-collision; the machine was
under heavy load from other builds, so Rubble's figures are upper bounds; the clean
phase-8 figures are 4.83 / 6.19 / 10.03 ms per step for the high-res car):

| | 1 car | 10 cars | 40 cars |
|---|---|---|---|
| sedan, RoR-style 8 kHz, 1 thread / 16 threads | 2.8 / 2.8 | 27.7 / 3.1 | 112 / 11.7 |
| sedan, Rubble 240 Hz × 10 | 10.8–13.2 | 17–24 | 21–32 |
| high-res, RoR-style 4 kHz, 1 / 16 threads | 3.7 / 3.8 | 37 / 5.8 | 151 / 20 |
| high-res, Rubble 240 Hz × 10 | 19.3 (clean) – 23.5 | 24.8 (clean) – 39 | 40 (clean) – 62 |
| high-res, Rubble 240 Hz × 10, no soft contacts | 19.4 | | 22.6 |
| high-res, Rubble 960 Hz × 30 (converged) | 234 | 281 | |

Reading: today Rubble is neither faster nor more accurate than the RoR method on a
16-core CPU. The explicit solver at its stable rate is converged (it is the oracle's
method), while Rubble at 240 Hz × 10 over-crushes the sedan by ~40% and collapses the
high-res car; at 960 Hz × 30 Rubble converges but costs ~60× the CPU. Rubble's cost
is dispatch latency (about 4–5 ms per step for one car, flat in car count), so the GPU
only gains with car count: at 40 high-res cars without soft contacts it is level with
16 CPU threads. Rubble's content also needs 4–6 kHz explicitly; RoR's fixed 2 kHz
cannot run the high-res car.

## Status summary

| Phase | State |
|---|---|
| 0 harness and oracle | done |
| 1 collision parity | mostly done (activation pre-pass, hull obstacles, staticCollision, skin tearing open) |
| 2 OGC bounds | implemented, off by default (mixed results; zero penetrations not reached, the oracle has them too) |
| 3 JBeam features | done for LBEAM, torsion bars, rails, slidenodes, anisotropic, deform limits; smaller beam options open |
| 4 float precision | done |
| 5 crash islands | implemented, off by default (core-joint coupling) |
| 6 integrator spike | BDF2 re-measured; TR-BDF2 not started |
| 7 JGS2 spike | not started; now the main blocker for the high-res car |
| 8 resolution and scale | benchmarked; collide pass 4× faster; high-res car unconverged at real-time settings |
| 9 calibration | oracle fit recorded, not applied |

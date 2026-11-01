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

Open: triangle activation pre-pass, rigid hulls against soft triangles, the remaining
penetrations (phase 2's displacement bounds), `staticCollision`, visual tearing in the
skinning kernel matching collision tearing, self-collision cost on the high-res car.


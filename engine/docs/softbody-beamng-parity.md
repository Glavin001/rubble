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

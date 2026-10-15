# Vehicles

Raycast-wheel vehicles solved **inside** the AVBD solver, on Metal, native wgpu and
Chrome WebGPU from one Slang source (`slang/kernels/vehicle_step.slang`).

```bash
cd engine
cargo run -p engine-webgpu --release --bin wgsl_export     # WGSL + vehicle presets/golden
python3 -m http.server 8791 --directory crates/engine-webgpu
# Chrome → http://127.0.0.1:8791/vehicle.html            (?scene=playground|destruction|traffic&preset=sedan)
cargo test -p engine-webgpu --test vehicle                 # physical envelopes (wgpu)
cargo test -p engine-metal --test vehicle_parity           # Metal behaviour + Metal↔wgpu parity
node --test crates/engine-webgpu/tests/vehicle_layout.test.mjs   # JS packer == Rust packer
```

## Model

A vehicle is **one dynamic chassis body** plus per-wheel records in `gVehicleData`.
Attached wheels are not bodies:

| Row (per wheel) | Kind | Notes |
|---|---|---|
| Suspension | soft, one-sided, implicit spring-damper | `K = m_eff ω²`, `D = 2 m_eff ζ ω` (Jolt), folded as `K + D/dt` (newton VBD); `1/cos` slope correction; preload |
| Bump stop | compliant (default 6 Hz) or rigid | compliant avoids the full-compression load spikes that break mounts |
| Anti-roll bar | pair row | active only when both wheels touch (Jolt) |
| Longitudinal / lateral friction | hard AVBD rows with λ | bounded by a friction ellipse `μ(slip)·Fn` (or Jolt's independent box limits per wheel); stick anchors persist across steps so parked cars do not creep |
| Wheel spin `y` | extra 1-DoF per spin group | solved **exactly** each iteration (monotone 1-D root incl. tyre and brake limits), folded into the chassis with the Schur complement of its regime; a group is one wheel, or every wheel of a track |
| Brake / handbrake / rolling resistance | bounded servo row on `y` toward `y*` (0, or the steering-brake target of a track) | locked wheels hold positionally |
| Motorcycle lean | soft angular row about forward | Jolt `MotorcycleController` spring/damper/integral toward the ground-force direction, plus the linear force that keeps the tyres from being dragged sideways |
| Pitch/roll limit | one-sided angular row | Jolt `mMaxPitchRollAngle` |
| Aero | explicit | drag and downforce at an aero centre |

All rows act only on the chassis, so vehicles add **no graph edges and no colours**.
`vehicle_prepare`/`vehicle_iterate` write a 6×6 block into the `gAabbs` tail and
`primalAccumulate` adds it for bodies flagged `BODY_VEHICLE` (block index in
`BodyState.initialPad`). The ground is moving-but-given during a step; the reaction
reaches dynamic grounds as a velocity change for the next step.

Step placement: `vehicle_input → build_aabbs → wheel_query → … → predict → vehicle_prepare →
{primal colours → solve_dual → vehicle_iterate}×N → extract → vehicle_post → sleep`.

Drivetrain: a port of Jolt `WheeledVehicleController::PostCollide` — engine torque curve,
clutch, implicit engine/wheel coupling, auto/manual gearbox (Jolt shift gating),
open / limited-slip differentials, inter-differential LSD, plus a direct-drive mode.

Wheel queries: one 64-lane workgroup per wheel casts a ray, or sweeps a sphere of the
wheel radius (`wheel_query: Sphere`, catches kerbs and box edges), along the suspension at
`x⁻`. It runs right after `broadphase_query` and walks the broadphase grid cells the
segment crosses (padded by one cell), plus planes, large dynamics and listed large statics,
so cost no longer grows with the collider count. `VehicleHost::full_scan_queries` /
`WgpuSim::set_vehicle_full_scan` switches to scanning every collider; that is the test
oracle, and grid and full scan give bit-identical trajectories. `query_interval = N` probes
each wheel every N steps, staggered by wheel. Steps in between slide the contact along the
last hit plane, stored in ground-body space so it follows a moving ground (Jolt
`mNumStepsBetweenCollisionTest`). `raycast_batch` exposes the ray query.

Tracked vehicles (`tracks: Some(TracksDesc)`, a port of Jolt `TrackedVehicleController`):
each track's wheels form one spin group led by its driven wheel, which carries the track
inertia, damping, brake and differential ratio. The engine is tied to the faster track,
and each track gets `diff·ratio·transmission` torque up to the speed the engine allows.
The left/right ratio sync conserves momentum. Steer with no throttle below `pivot_speed`
counter-rotates the tracks to pivot. Two additions keep Jolt's ratio steering working
under implicit tyre rows, which would otherwise pull a synced track straight back to
ground speed:
* While driving, the inner track's brake holds it at `ratio ×` the outer track's speed.
  The brake fades in up to 1 m/s, and the torque it takes is handed to the outer track,
  like a geared steering differential.
* `turn_grip_g` caps the ratio so the turn needs at most that lateral acceleration.
  Jolt's fixed 0.6 ratio spins a 20 m/s tank out.

Tracks use `FrictionModel::Box`, so full drive force does not eat the sideways scrub
budget a pivot needs.

Motorcycles (`motorcycle: Some(MotorcycleDesc)`, a port of Jolt `MotorcycleController`):
* The target lean follows the direction of last step's ground force (normal plus
  lateral). It is smoothed, sideways only, and clamped to `max_lean_deg`.
* An implicit angular row with spring, damping and integral terms pulls the chassis
  toward the target while every wheel is loaded.
* Like Jolt, the lean torque comes with the linear force that cancels the acceleration it
  would give the contact patches. Without it the tyres get dragged sideways, the friction
  feeds back into the target lean, and the bike oscillates into a slide.
* Steering is capped to `asin(wheelbase·tan(max_lean)·g / (v²·cos caster))`.
* `suspension_force_point` (Jolt `mSuspensionForcePoint`) applies a wheel's suspension
  force at a chassis point, for example along a raked fork.

## Parts, functional state and severing

A vehicle is built from **parts** (collider groups on the chassis) with roles:
`body`, `engine`, `drive_shaft{wheels}`, `steering{wheels}`, `brake{wheels}`.

* `detach_wheel` / `detach_part` — the piece becomes a free rigid body (pre-allocated,
  parked far away) with the point velocity of the chassis; a wheel also keeps its spin.
  Chassis mass and inertia are reduced analytically; a short collision grace keeps the
  piece from exploding out of the chassis it overlapped.
* `disable_wheel` / `disable_part` — remove without spawning anything.
* **Ownership is the source of truth:** a part is attached while its colliders belong to
  the chassis. A future fracture system can sever a part with only `CMD_REPARENT`; the
  vehicle notices next step and updates `wheel_mask` / `drive_mask` / `steer_mask` /
  `brake_mask` / `engine_connected`.
* `set_functional_state` — overrides, ANDed with ownership (cannot resurrect a lost part).
* `add_wheel` — attach a wheel into a spare slot (`VehicleDesc::spare_wheels`) at runtime.
* `set_wheel_params` — retune a wheel in place.
* `step_loads` — per-wheel mount impulse wrench for a stress/fracture solver.
* `poll_vehicle_events` — `WheelDetached`, `WheelRemoved`, `PartLost`,
  `FunctionalStateChanged`, `WheelContactBegin/End`, `WheelSlideBegin/End`
  (`VehicleState::event_seq` also increments on every functional change).
* `snapshot` / `restore` (`WgpuSim`) — copies every GPU buffer plus host state; replay
  from a snapshot is bit-identical on the same device (rollback netcode, debugging).

`MountDesc { break_force, break_torque }` is reserved on parts; no automatic fracture
reads it yet.

## API

Rust (`engine-vehicle`; `MetalWorld` and `WgpuSim` expose the same methods):

```rust
let v = sim.add_vehicle(&presets::sedan(), [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])?;
sim.set_vehicle_input(v, VehicleInput { forward: 1.0, steer: -0.3, ..Default::default() });
sim.step()?;
let st = sim.vehicle_state(v);          // speed, rpm, gear, masks, per-wheel pose/forces/slip
sim.detach_wheel(v, 0);                  // or detach_part / disable_* / set_functional_state
let hits = sim.raycast_batch(&rays);
```

JS (`crates/engine-webgpu/vehicle.mjs` + `world-host.mjs`): `VehicleHost` with
`addVehicle / setInput / detachWheel / detachPart / disableWheel / disablePart /
setFunctionalState / addWheel / readState`, driven by `createWorld(device, …)`.

Presets (`engine_vehicle::presets`, exported to `generated/vehicle_presets.json`):
`jolt_default`, `sedan`, `destructible_sedan`, `sports`, `monster_truck`,
`six_wheel_truck`, `kart`, `tank` (two 7-wheel tracks, Jolt `TankTest` drivetrain),
`motorcycle` (Jolt `MotorcycleTest`: 30° caster fork, chain-driven rear wheel, lean
controller).

## Configuration

| Group | Options |
|---|---|
| Chassis | parts (shape, position, rotation, material, mass, role), inertia override, auto centre of mass, centre-of-mass offset, gravity scale, can sleep, spare wheels |
| Wheel | position, suspension dir, steer axis (caster/kingpin), forward (toe), up (camber, visual), radius, width, mass, inertia, angular damping, suspension force point, rim spheres/material of the free wheel |
| Suspension | min/max length, preload, spring (frequency+ζ or k+c), bump stop (spring / rigid / none, max force), roll influence |
| Tyre | longitudinal μ(slip ratio) and lateral μ(slip angle °) curves, μ scales, friction model (ellipse / box), rolling resistance; surface friction combines as √(tyre·surface) |
| Steering | max steer per wheel, Ackermann strength, wheelbase/track (auto), slew rate, steer-vs-speed curve |
| Brakes | max brake / handbrake torque, handbrake wheels, response rates |
| Drive | drivetrain {engine curve, min/max RPM, inertia, damping; gearbox auto/manual, ratios, reverse, switch/release/latency, shift RPMs, clutch strength; differentials {wheels, ratio, split, LSD, torque share}; inter-diff LSD} or direct {torque, top speed} |
| Stability | anti-roll bars, max pitch/roll, max slope for contacts, reverse-brake behaviour |
| Aero | drag, downforce, aero centre |
| Assists (opt-in) | air control (pitch/yaw per input while airborne), upright stiffness/damping |
| Queries | ray / sphere cast, query interval (probe every N steps, extrapolate between) |
| Tracks | per track: wheels, driven wheel, inertia, damping, brake torque, differential ratio; turn ratio, pivot speed, turn grip (g) |
| Motorcycle | max lean, lean spring/damping/integral + decay, target smoothing, lean-limited steering |

## Limits and next steps

* Wheel casts are rays or spheres; a cylinder cast that follows steer and camber is not
  implemented. A sweep longer than `RAY_GRID_MAX_CELLS` (512) grid cells stops early.
* At most 16 wheel slots per vehicle (tank tracks use 7 wheels a side, Jolt's sample uses 9).
* The ground reaction on dynamic bodies lags one step and is applied without atomics
  (two vehicles on the same box in one step can lose one update).
* Wheels are not colliders while attached. Each wheel's load is capped (`bump_max_force`,
  else 8× an even share of the weight). The reaction on a light dynamic body is capped at
  the relative speed plus 2 m/s. Hits on dynamic bodies higher than the wheel could climb
  are ignored, so the chassis meets that crate instead (DECISIONS v9).
* Not yet: cylinder casts, an inactive (sleeping) query rate, and automatic load-based
  fracture (its inputs already exist via `step_loads`).

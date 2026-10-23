# Vehicles

Raycast-wheel vehicles solved **inside** the AVBD solver, on Metal, native wgpu and
Chrome WebGPU from one Slang source (`slang/kernels/vehicle_step.slang`).

```bash
cd engine
cargo run -p engine-webgpu --release --bin wgsl_export     # WGSL + vehicle presets + engine-vehicle wasm (wasm-pack)
python3 -m http.server 8791 --directory crates/engine-webgpu
# Chrome → http://127.0.0.1:8791/vehicle.html            (?scene=playground|destruction|buggies|traffic|bridge&preset=sedan)
cargo test -p engine-webgpu --test vehicle                 # physical envelopes (wgpu)
cargo test -p engine-webgpu --test vehicle_mounts          # mounts that break under load (wgpu)
cargo test -p engine-metal --test vehicle_parity           # Metal behaviour + Metal↔wgpu parity
node --test crates/engine-webgpu/tests/vehicle_layout.test.mjs   # JS wrapper over the wasm host
```

## Model

A vehicle is **one dynamic chassis body** plus per-wheel records in `gVehicleData`.
Attached wheels are not bodies:

| Row (per wheel) | Kind | Notes |
|---|---|---|
| Suspension | soft, one-sided, implicit spring-damper | `K = m_eff ω²`, `D = 2 m_eff ζ ω` (Jolt), folded as `K + D/dt` (newton VBD); `1/cos` slope correction; preload |
| Bump stop | compliant (default 6 Hz) or rigid | compliant avoids the full-compression load spikes that break mounts |
| Anti-roll bar | pair row | active only when both wheels touch (Jolt) |
| Longitudinal / lateral friction | hard AVBD rows with λ | bounded by a friction ellipse `μ(slip)·Fn`, shared along the slip velocity once saturated (combined slip), or by Jolt's independent box limits per wheel; stick anchors persist across steps so parked cars do not creep |
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
`x⁻`. Each collider's hit counts only inside the tyre cone (35° from the suspension axis for
sphere casts, and for a hit above full bump the cone is measured from the full-bump wheel centre).
Anything else is the wheel shell's, so a crate face beside the tyre never hides the ground below
it. A cast that starts inside a collider (a mount overlapping a crate) reports where the line
entered it (t ≤ 0) and the real normal there, never an invented full-bump hit along the axis.
It runs right after `broadphase_query` and walks the broadphase grid cells the
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
The left/right ratio sync conserves momentum. Input is an arcade skid-steer mix: left
track = throttle + k·steer, right = throttle − k·steer (`steer_strength` k). So steer
always rotates the hull the same way: a pivot from rest, a track-ratio turn while driving,
and the same rotation sense in reverse. Two additions keep Jolt's ratio steering working
under implicit tyre rows, which would otherwise pull a synced track straight back to
ground speed:
* While driving, the inner track's brake holds it at `ratio ×` the outer track's speed.
  The brake fades in up to 1 m/s, and the torque it takes is handed to the outer track,
  like a geared steering differential.
* `turn_grip_g` caps the ratio so the turn needs at most that lateral acceleration.
  Jolt's fixed 0.6 ratio spins a 20 m/s tank out.

Tracks use `FrictionModel::Box`, so full drive force does not eat the sideways scrub
budget a pivot needs.

Track pods (`TrackDesc::pod`) make the track a solid belt instead of a row of rays.
The ground probes still carry the tank on terrain, but each track also gets up to four
belt boxes on the chassis, built by `pack::track_pod_colliders`:
* an upper run over every wheel, from the end wheels' axles up to `top`;
* a lower run between the first and last road-wheel axles, down to the road wheels'
  bottomed-out tread line plus `clearance`, so on flat ground only the rays touch;
* a ramp at each end, from that tread line up to the sprocket's or idler's rim. An
  edge met on the ramp pushes the nose up even when the track is stalled.

The pod closes the gaps between road wheels and replaces their shells. Its belt runs
at the sprocket's rim speed: `vehicle_post` writes ω·R into each pod collider's
`density`, and world_step's friction rows target the surface velocity speed·(axle ×
n_out). The surface moves backwards along the bottom, down the nose and forwards
along the top. So debris under a moving track stays put on the ground while the tank
rolls over it, and a step against the nose is pulled under. At rest the belt is
plain friction. Contacts that don't involve a belt cost one flag test. The tank preset
uses 0.6 m pods, twice its road-wheel width, and a belly plate giving about 0.5 m
ground clearance.

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

## Two-way ground coupling

A wheel acts on whatever it stands on, inside the same solve. A crate under a tyre is
pushed down and sideways, a bridge deck feels the truck's real (dynamic) load in the step
it happens, and a car can drive on another car. This is the Jolt model
(`VehicleConstraint::BuildIslands` links the vehicle and each wheel's dynamic contact body
into one island; suspension, max-up stop and friction are two-body constraint parts),
expressed in AVBD's per-body block solve:

* **Static or kinematic ground** keeps the fast path. The tyre rows act on the chassis
  only, and the ground's motion enters as a given displacement.
* **Dynamic ground** turns each touching wheel into a two-body constraint.
  * `vehicle_link` runs after `wheel_query`, before the constraint graph is built, and
    points the wheel's reserved tyre joint (a D6 record flagged `JOINT_FLAG_TYRE`, its
    ground block index in `motion`) at (chassis, ground). A sleeping ground
    body is woken, as in Jolt. The joint only creates the graph edge, so the two bodies get
    different colours and the ground body's primal gathers the wheel. It is skipped by
    the joint pair filter (the chassis still collides with the crate), by joint
    warm-starting and by the joint dual update. With no dynamic ground, the joint points
    at the vehicle's static anchor body and adds no edge.
  * Each iteration, the vehicle kernels evaluate every tyre row (suspension and bump stop,
    anti-roll, longitudinal, lateral) at the current iterates of both bodies. The ground's
    displacement at the contact point replaces the explicit `v_g·dt`. Each row adds its
    chassis Jacobian to the chassis block (as before, with wheel spin condensed). It also
    adds its ground Jacobian `J_g = −[e; r_g × e]` to that wheel's **ground block**: a
    6×6 + 6 in the `gAabbs` tail after the chassis blocks. The ground body's primal adds
    the block through the tyre joint. Both blocks come from the same rows with the same
    multipliers, so the forces are equal and opposite.
  * The lagged explicit ground reaction is gone, along with its velocity cap.
* **Wheel shells.** Each wheel has a collider on the chassis body (a sphere, or a capsule
  along the axle, sized slightly inside the tyre). `vehicle_post` moves it to the wheel's
  suspended position every step. It handles everything except the tyre's own ground
  contact: sidewalls against debris, kerb faces, other cars. Because it sits inside the
  tyre, it never carries the car on the ground; the tyre rows do that. This is a simpler
  variant of the PhysX vehicle pattern of wheel shapes plus contact modification that
  drops the ground contact. Detaching or removing a wheel disables its shell.
* **Removed workarounds.** The default wheel-load cap is gone (`bump_max_force` stays as
  an explicit, physical option), and so is the "ignore unclimbable dynamic hits" ray
  filter. With two-body rows and shells, the solver resolves crates and debris.

The bridge case follows directly: the deck body's block carries the truck's load, so a
deck held by breakable joints gives way under a heavy vehicle, and the truck goes with it.

### Breakable joints and the plank bridge

`JointDesc::with_break(force, torque)` makes any D6 joint breakable (`JOINT_BIG` =
never; see [JOINTS.md](JOINTS.md)). On the final solver iteration `solve_dual` compares the
converged linear force and angular torque with the thresholds. A joint over either one gets
`JOINT_FLAG_BROKEN` and an `EVENT_BREAK` record `[3, joint, force, torque]` in the event
ring of the scratch atomics (binding 21, from word `EVENT_RING_BASE` = 96). From the next
step it is out of the graph (CSR, warm-start, dual, sleep gate)
and out of the pair filter, so its bodies collide again. Hosts read new events with
`WgpuSim::poll_joint_breaks`, `MetalWorld::poll_joint_breaks` and `world.pollJointBreaks()`
(JS). Tyre joints never break.

`?scene=bridge` (and the `bridge` / `bridge_fast` recorder takes) mirrors
`tests/vehicle.rs`: five 2.6 m planks (200 kg/m) span 12 m between abutment seats, 3 m
up. Each seam has four ball-joint bolts at the top and bottom corners, so the deck
carries bending as a bolt couple, rated 255 kN. Bolt forces at 20 solver iterations:

| Load (2.5 m/s) | Midspan bolt force |
|---|---|
| Deck alone | 102 kN (107 kN analytic: M = w·x(L−x)/2 over a 0.32 m couple) |
| 1.3 t sedan | ~205 kN, holds |
| 3 t monster truck | ~320 kN if unbreakable: the bolts break and it falls through |
| 4.6 t tank | ~450 kN |

The monster truck always breaks the bolts, but speed decides where it ends up:

| Truck speed | Outcome |
|---|---|
| ≤ 8 m/s | Falls through with the planks |
| 9 m/s | Front wheels reach the far road, rear caught on the edge |
| ≥ 10 m/s | The deck folds behind it and it lands on the far road |

The scene runs 20 iterations: at 10, colored Gauss–Seidel needs about 3 s to converge
the deck's bending, so a truck faster than ~9 m/s crossed before the bolts felt its
weight. The bolts start at a 1e8 N/m penalty. From the default 1e4 the β·|C| ramp is
far too slow for 520 kg planks (m/h² ≈ 2e6 N/m), and the span free-falls. Tab switches
the driven vehicle (sedan first, then `?heavy=monster_truck`; `heavy=none` for one).

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

### Mounts that break under load

Every part except the root (part 0) and every wheel has a `MountDesc { break_force,
break_torque, anchor }`. A limit ≤ 0 is unbreakable (the default: every preset but
`dune_buggy` keeps its parts until something severs them).

`vehicle_loads` runs once per step after the final dual and `extract_velocity_sleep`.
It uses one thread per vehicle and walks the chassis's contacts in its sorted CSR order,
so the sums are deterministic and use no float atomics. A part is rigid with the chassis,
so its mount carries what the part's own motion needs minus what its contacts supply:

    F = m (a_p − g) − Σ F_contact
    τ_anchor = I α + ω × I ω − (anchor − com) × F − Σ (x_i − com) × F_i

* `a_p` comes from the chassis's linear and angular acceleration over the step and the
  part's centre of mass.
* The contact forces are the final λ of the contacts on the part's colliders (n = A→B; B
  gets −λ).
* A wheel's mount carries its tyre force, its shell contacts and its own inertia. The
  moment about the spin axis is left out (the drive and brakes react it).
* `anchor` defaults to the part's centre of mass, or a wheel's hardpoint.

Past either limit, the part's (or wheel's) bit is set in `V_BRK` and an
`EVENT_MOUNT_BREAK` record `[6, vehicle << 8 | is_wheel << 7 | index, force, torque]`
goes on the shared event ring. The next step's `vehicle_input` severs it exactly as
`detach_part` / `detach_wheel` would. The break is decided on the converged step and the
split happens before the next step solves: forward-only, the same rule joints break by.

Readouts:

* `VehicleState::parts_broken` / `wheels_broken`: latched masks.
* `VehicleEvent::PartBroken` / `WheelBroken`.
* `read_mount_loads` / `WgpuSim::vehicle_mount_loads`: each mount's last force and moment
  against its limits, and the peak ratio since spawn.
* `poll_mount_breaks` (Rust); `world.pollMountBreaks` / `VehicleHost.readMountLoads` (JS).

Contacts are read as each host stores them:

* Canonical and compact hosts (the browser, Metal) publish final λ to `gContacts`.
* The persistent-contact host compiles `vehicle_loads_persistent`
  (`slang/kernels/vehicle_step_persistent.slang`), which reads its solver pairs and the
  contacts' cold records.

The check runs only for vehicles flagged `VF_BREAKABLE` (some finite limit) and awake.
A sleeping vehicle carries its resting load and is skipped.

One level of mounting: every part hangs off the chassis. A headlamp rated against the
nose stays on the car when the nose comes off.

**Discrete impacts.** A projectile that moves more than a thin part's depth in one step
lands its blow on whatever is behind it (`tests/vehicle_mounts.rs` uses impacts under
0.2 m per step). A 45 m/s cannonball goes through a 3 cm lamp bracket and meets the cage.

### The dune buggy

`dune_buggy` is vibe-land's sand rail (`engine/scenes/dune-buggy`, baked from its recipe).

* Its 17 parts carry the recipe's masses.
* Each part's mount is rated from the measured joints between it and the frame. Each
  joint gives interface area × the weaker material's fatal strength, bounded by the
  smaller member's section (vibe-land `real-joint-capacity.mjs`).
* A wheel's mount is its suspension arms, tie rod and damper where they meet the frame.

`vehicle.html?scene=buggies` draws it with the recipe's full mesh and a posed suspension
(`buggy-render.mjs`, `buggy-rig.mjs`). `?strength=0.5` (the *Mounts* menu) scales every
limit.

At the rated strength (native wgpu, 10 iterations, `dune_buggy_crash_probe`):

* Two buggies meeting head-on tear off each other's front bumper, nose and headlamps.
* A side hit by a 400 kg ball at 25 m/s snaps the seat it strikes.
* Jumps and full-throttle driving stay under 11% of every limit.

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
setFunctionalState / addWheel / setWheelParams / readState`, driven by
`createWorld(device, …)`. It wraps the Rust `VehicleHost` and packer compiled to wasm
(`crates/engine-vehicle-wasm`): each call returns the record writes as an op stream
that `applyOps` replays into the GPU buffers, so the browser uploads the same bits
as the native hosts.

Presets (`engine_vehicle::presets`, exported to `generated/vehicle_presets.json`):
`jolt_default`, `sedan`, `destructible_sedan`, `sports`, `monster_truck`,
`six_wheel_truck`, `kart`, `tank` (two 7-wheel tracks, Jolt `TankTest` drivetrain),
`motorcycle` (Jolt `MotorcycleTest`: 30° caster fork, chain-driven rear wheel, lean
controller), `dune_buggy` (vibe-land's sand rail; parts and wheels on breakable mounts).

## Configuration

| Group | Options |
|---|---|
| Chassis | parts (shape, position, rotation, material, mass, role, mount limits), inertia override, auto centre of mass, centre-of-mass offset, gravity scale, can sleep, spare wheels |
| Wheel | position, suspension dir, steer axis (caster/kingpin), forward (toe), up (camber, visual), radius, width, mass, inertia, angular damping, suspension force point, rim spheres/material of the free wheel, mount limits |
| Suspension | min/max length, preload, spring (frequency+ζ or k+c), bump stop (spring / rigid / none, max force), roll influence |
| Tyre | longitudinal μ(slip ratio) and lateral μ(slip angle °) curves, μ scales, friction model (ellipse / box), rolling resistance; surface friction combines as √(tyre·surface) |
| Steering | max steer per wheel, Ackermann strength, wheelbase/track (auto), slew rate, steer-vs-speed curve |
| Brakes | max brake / handbrake torque, handbrake wheels, response rates |
| Drive | drivetrain {engine curve, min/max RPM, inertia, damping; gearbox auto/manual, ratios, reverse, switch/release/latency, shift RPMs, clutch strength; differentials {wheels, ratio, split, LSD, torque share}; inter-diff LSD} or direct {torque, top speed} |
| Stability | anti-roll bars, max pitch/roll, max slope for contacts, reverse-brake behaviour |
| Aero | drag, downforce, aero centre |
| Assists (opt-in) | air control (pitch/yaw per input while airborne), upright stiffness/damping, traction control |
| Queries | ray / sphere cast, query interval (probe every N steps, extrapolate between) |
| Tracks | per track: wheels, driven wheel, inertia, damping, brake torque, differential ratio, pod (width, clearance, top, material); steer strength, turn grip (g) |
| Motorcycle | max lean, lean spring/damping/integral + decay, target smoothing, lean-limited steering |

## Limits and next steps

* Wheel casts are rays or spheres; a cylinder cast that follows steer and camber is not
  implemented. A sweep longer than `RAY_GRID_MAX_CELLS` (512) grid cells stops early.
* At most 16 wheel slots per vehicle (tank tracks use 7 wheels a side, Jolt's sample uses 9).
* The ground reaction on dynamic bodies lags one step and is applied without atomics
  (two vehicles on the same box in one step can lose one update).
* Wheel shells are spheres (the world narrowphase has no cylinder or capsule–box pair), so
  sidewalls are a five-sphere approximation of the tyre outline.
* A ground block holds one wheel's rows. When both ends of an anti-roll bar stand on the
  same dynamic body, the cross term between the two blocks is dropped (the gradient stays
  exact; the Hessian is block-diagonal).
* A track belt's speed is set by the sprocket (one-way). Friction the pod meets acts on
  the hull but not back on the track spin, so climbing does not load the engine.
* Mounts are one level deep (every part on the chassis) and the root part never breaks.
* Not yet: cylinder casts, an inactive (sleeping) query rate.

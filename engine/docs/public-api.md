# Rubble public API

Status: Phase 1 shipped on `claude/public-api-world` (crate `engine-world`,
library name `rubble`). This document is the proposal, the record of what
Phase 1 delivered, and the plan for the phases after it.

## Why

Before this work, a host (Unreal, Godot, Unity, Bevy, a web page or a server)
had to drive `WgpuSim` directly:

- Every velocity change, impulse, wake or teleport was a **blocking** GPU map:
  `read_body`, then edit, then `write_bodies`. vibe-land's `rubble-bridge`
  does this for every gameplay write.
- Bodies could not be removed, so hosts "parked" them far away and tracked the
  slot themselves.
- Poses came back as the whole 112 B `BodyState` for every body, sleeping or
  not.
- Only `with_device` took the host's GPU device, and it was fixed to
  canonical contacts.
- About 30 `ENGINE_*` environment variables chose internal paths.
- Events were spread across several APIs: joint breaks, soft-body events and
  vehicle events each had their own polling call and cursor. There were no
  sleep or wake events.

## What other engines do

The research covered Box2D v3 (and v2.4), Jolt, PhysX 5, Rapier, Bepu v2,
Bullet, Unity (PhysX and DOTS), Unreal Chaos, MuJoCo/MJX/MJWarp, Newton,
Genesis and AVBD.

| Concern | Common practice | Rubble Phase 1 |
|---|---|---|
| Handles | An index plus a generation, passed by value. Examples: Box2D v3 `b2BodyId`, Jolt `BodyID` (sequence number), Rapier arena handles. Paired with `IsValid` checks. | `BodyId` and `JointId`: u64 values (index, generation), packable with `to_bits`/`from_bits`. Stale handles return `Err`. |
| Creation | Definition structs with defaults, copied on create: Box2D `b2DefaultBodyDef`, Jolt `BodyCreationSettings`. | `BodyDef`, `ShapeDef`, `MaterialDef`, `JointDef`, all with `Default`. |
| Stepping | Fixed `dt` and explicit sub-steps, with no internal accumulator (Box2D, Bepu, Newton, Jolt). Bullet's `btMotionState` interpolation is the exception. | Fixed `dt` (`WorldConfig::solver`). The host accumulates real time and interpolates `previous → pose`. |
| Transform sync | Report only the bodies that moved: Box2D `b2BodyMoveEvent` (with `fellAsleep`), PhysX active actors. | `Frame::moved`: only bodies whose pose bits or sleep state changed, each with its previous pose. |
| Events | Arrays read after the step and valid until the next one (Box2D, Unity DOTS, PhysX `fetchResults`). Opt-in per shape. Hit events gated by a threshold. | `Frame::events` per step: joint, mount and beam breaks, soft-body pieces, sleep and wake. A lost count is always reported. |
| Capacity | Fixed at init and failing loudly: Jolt `Init`, MJWarp `naconmax`, Box2D `capacity`. | `WorldCapacity` errors on creation (checked against device limits) and on overflow. |
| GPU sharing | PhysX Direct GPU API, Newton/Warp arrays, MJX device arrays. | `GpuContext::Shared` takes the host's device. `WgpuSim::pose_buffer` holds poses that can be drawn in place. |
| Determinism | Jolt: same binary and call order; `CROSS_PLATFORM_DETERMINISTIC` costs about 8%. Rapier: `enhanced-determinism`. | Same build and same GPU give the same results run to run. Command order per body is the submission order. |
| Destruction | Chaos break, collision, trailing and removal events behind listener filters. AVBD constraint rows carry a `fracture` threshold. | Joint, beam, mount and piece events now. GPU fracture splits in World come in Phase 4. |

The patterns adopted are:

- id+generation handles;
- definition structs with defaults;
- per-step event arrays that report losses explicitly;
- move events only for bodies that moved, including sleep transitions;
- loud capacities;
- fixed `dt` with interpolation left to the host;
- the host's device accepted.

Two patterns are not adopted yet:

- per-shape opt-in contact and sensor events, and threshold hit events (Phase 2);
- a debug-draw callback struct (Phase 5).

## Phase 1: what shipped

Every item is **lossless**: a world that does not use a feature computes
exactly what it computed before. The tests compare `download_bodies()`
byte for byte.

### 1. One world handle: `rubble::World`

```rust
use rubble::*;
let mut world = World::new(GpuContext::Default, WorldConfig {
    capacity: WorldCapacity::small(),
    gravity: [0.0, -9.81, 0.0],
    ..WorldConfig::default()
})?;
let ground = world.create_body(
    &BodyDef { kind: BodyKind::Static, position: [0.0, -0.5, 0.0], ..Default::default() },
    &[ShapeDef::new(Shape::Box { half: [50.0, 0.5, 50.0] })],
)?;
let crate_ = world.create_body(
    &BodyDef { position: [0.0, 4.0, 0.0], user_data: 42, ..Default::default() },
    &[ShapeDef { density: 500.0, ..ShapeDef::new(Shape::Box { half: [0.5; 3] }) }],
)?;
world.apply_impulse(crate_, [0.0, 0.0, 200.0], [0.0, 4.2, 0.0])?;
loop {
    world.step()?;                       // never waits on a readback
    for frame in world.poll()? {         // finished steps, oldest first
        for m in &frame.moved { /* render: lerp m.previous -> m.pose */ }
        for e in &frame.events { /* audio, VFX, gameplay */ }
    }
}
```

- Construction: `World::new(GpuContext::{Default, Shared{info, device, queue}}, WorldConfig)`.
- `WorldConfig` fields:
  - `capacity`, `contacts`;
  - `vehicles` and `soft` pools;
  - `gravity` (Y only for now; any other axis is refused);
  - `solver` (`AvbdParams`) and `features`;
  - `env_overrides`, which defaults to off;
  - `readback_frames` and `backpressure`.
- Bodies:
  - `create_body` and `remove_body`;
  - checks: `is_valid`, `user_data`, `pose`;
  - commands: `set_velocity`, `apply_impulse`, `add_force` (one step), `teleport`, `wake`.
- Joints:
  - `create_joint(JointDef { body_a, body_b, desc })` takes any `JointDesc` preset (ball, hinge, fixed, d6, bond…), with limits, drives and break thresholds;
  - `remove_joint`, `set_joint_drive`, `is_joint_valid`.
- Vehicles and soft bodies:
  - `add_vehicle`, `set_vehicle_input`, `vehicle_state`;
  - `add_soft_vehicle`;
  - `owned_body`, which gives a handle to a chassis or wheel body.
- Queries: `raycast(&[Ray]) -> Vec<Option<RayHit { body, point, normal, distance }>>` (blocking).
- Stepping:
  - `step()`;
  - `poll()` (non-blocking) and `poll_blocking()`;
  - `read_poses()`, which resynchronises every pose.
- `sim()` gives read access to the `WgpuSim` underneath.

### 2. Every configuration takes the caller's device

- `WgpuSim::with_device_options(capacity, SimOptions, info, device, queue)` is
  the one constructor. Every `new_with_*` wraps it.
- `SimOptions` holds:
  - `contacts: ContactStorage::{Canonical, Compact, Persistent}`;
  - the vehicle pool;
  - `soft: Option<SoftCapacity>` (requires canonical contacts, or creation fails);
  - `features: EngineFeatures`;
  - `pose_output`.
- The device's limits are checked against every binding before allocation,
  with a clear error.
- `EngineFeatures` replaces the scattered `ENGINE_*` reads.
  `SimOptions::from_env()` resolves them exactly as before, so binaries and
  benchmarks are unchanged.

### 3. Body commands without readback

- `WgpuSim::push_body_commands(&[GpuCommand])` takes:
  - `wake`, `teleport`, `set_velocity`;
  - `apply_impulse` (new): an impulse at a world point;
  - `add_force` (the `AttachForce` opcode): force and torque for one step.
- They run through the existing Slang `apply_commands` kernel, which the plan
  now emits first in a step whenever commands are queued.
- The host stable-sorts commands by body. One thread applies each body's run in
  submission order and stores the result into both body rows. That reproduces
  exactly what a host `read_body`/`write_bodies` would leave, and the tests
  check it bit for bit for wake, teleport and set_velocity.
- Going past `max_commands` is an error.
- Metal, CUDA and the JS host replay the same plan. They pass 0 commands, so
  their command lists do not change.

### 4. Render poses and the moved list: `pose_pack.slang`

- The kernel runs after the sleep commit when `SimOptions::pose_output` is on.
  `World` always turns it on.
- It reads `gBodies` and writes only its own buffers (bindings 33–35, which
  sit after the shared plan's 33):
  - `pose_buffer` holds every body's pose, 32 B each, for renderers that draw in place;
  - the moved list holds a 32 B record for each body whose pose bits or sleep
    state changed this step, with `SLEPT`, `WOKE` and `FIRST` bits;
  - `FIRST` marks a new body or a reused slot, which has no pose to interpolate from.
- Sleeping bodies cost one read and no write. A host reads 32 B per moved body,
  instead of 112 B for every body.
- The list's order is atomic and therefore nondeterministic, so hosts sort it
  by body. `World` does this.

### 5. Unified events and lifecycle

- `WgpuSim::ring_events(words)` decodes one copy of the event ring into joint,
  mount and beam breaks, plus the soft pieces those breaks cut loose. When
  records were lost, it resynchronises the soft bodies.
- Sleep and wake events come from the moved list's transition bits. That list
  is bounded by the body count, so it cannot overflow. A pile waking at once
  would overflow the 1024-record ring.
- `Frame::lost_events` counts records overwritten before they were read. There
  is a test that breaks 1100 joints in one step.
- Removal:
  - `World::remove_body` releases the body's joints;
  - it parks the body: static, at a lattice point far below the world;
  - it turns the body's colliders into tiny spheres that collide with nothing,
    so a removed large body cannot fill the oversized-static list;
  - it bumps the slot's generation;
  - the slot is reused after two steps, once neither contact history half can
    reference its old colliders.
- A reused slot is deterministic, but it is not identical to a fresh world.

### 6. Asynchronous results

- Each step copies its counters, the event ring and the moved list into one
  slot of a staging ring and maps it with `map_async`.
- `poll()` returns finished frames in step order.
- When every slot is in flight:
  - `Backpressure::Block` waits for the oldest;
  - `Drop` marks the frame `dropped`, and the host calls `read_poses`.
- Pipelined stepping was gated first. `step_submit` gives bodies byte-identical
  to `step()` (`engine-webgpu/tests/pipelined_gate.rs`).
- A step's counters also tell the simulation when it may skip an asleep step.
  This happens when the frame is the newest one, or when nothing was awake and
  nothing was written since that frame.

### Tests

| File | Covers |
|---|---|
| `engine-world/tests/world.rs` | World is bit-identical to raw `WgpuSim` over canonical, compact and persistent contacts, with and without a vehicle. Also: frame order and previous poses; one sleep per box and a wake on command; 500 remove/re-add cycles (stale handles refused, no NaN, no ghost contacts, deterministic); joint breaks with handles; ring overflow reported; config errors. |
| `engine-world/tests/bridge_acceptance.rs` | vibe-land `rubble-bridge` cases through World: central impulse, free fall, removal and slot reuse, a city of large statics, a sleeping wall, a ball knocking the wall loose, rays. |
| `engine-webgpu/tests/body_commands.rs` | Commands match host read-modify-write bit for bit; impulse and force math; per-body order; 1000 commands on one body; the capacity error. |
| `engine-webgpu/tests/pose_output.rs` | Bodies are the same with pose output on or off (all three contact modes); the moved list equals a host diff every step; one SLEPT and one WOKE per body; FIRST after `forget_pose`. |
| `engine-webgpu/tests/device_options.rs` | A caller's device matches the wrappers bit for bit; the soft/contacts error. |
| `engine-webgpu/tests/pipelined_gate.rs` | Pipelined steps match synchronous steps. |

### Known gaps found on the way

- **Asleep colliders and rays.** A collider added *asleep* to a world that has
  already stepped gets no ray bounds, so rays miss it. Raw `WgpuSim` on `main`
  misses it too.
  - The fix is `af24d153`, "WgpuSim: bounds for colliders added asleep to a
    running world", on vibe-land's `feat/vibe-land-backend` checkout. It is not
    merged into `main`.
  - `bodies_added_asleep_to_a_running_world_are_hit_by_rays` is `#[ignore]`d
    until it lands.
- **Large statics overflow.** A city of large statics beside small dynamic
  bodies overflows the oversized-static list on the uniform grid. It needs
  `EngineFeatures::grid_levels = Some(true)`, the same setting the bridge uses.
  The overflow is reported as an error, never dropped.

## Corrections to the original draft

- `engine-ffi` and `engine-demos` do not exist; there is no C ABI yet.
  `World` is the surface a C ABI will wrap (Phase 2).
- `apply_commands` existed in Slang, but the shared plan never emitted it.
  It does now, when commands are queued.
- `BODY_KINEMATIC`, `EVENT_SLEEP` and `EVENT_WAKE` were defined but never used.
  Sleep and wake now come from `pose_pack`. Kinematic bodies are Phase 2.
- Gravity is Y only (`AvbdParams::gravity_y`, part of the 80 B parameter
  layout every host shares). A full vector has to keep the Y-only arithmetic
  bit-identical, so it is Phase 2 and is refused in `WorldConfig` until then.
- Z-up conversion stays at the FFI boundary, as the draft proposed. It is exact
  and free, and the kernels stay unchanged.

## Next phases

1. **C ABI (`engine-ffi`).** A cbindgen header over `World`, with opaque
   handles as `uint64_t`, def structs with `rubble_default_*_def()`, and Z-up
   conversion at the boundary. Unreal is the first client, through RHI's
   `GetNativeDevice` together with item 5.
2. **Physics surface.**
   - Full gravity vector, with the Y-only path left unchanged.
   - Kinematic bodies with velocity-driven targets.
   - Opt-in contact begin/end and sensor events. These need a "matched last
     step" mark that works with `carry_rest` and manifold reuse.
   - GPU-filtered impact events above a threshold (point, normal, impulse,
     bodies, materials) for audio and VFX.
   - Debug-draw callback struct.
3. **Queries.** Asynchronous batched raycast, overlap and shape cast, with
   query filters (category and mask), outside the vehicle pool.
4. **Fracture in `World`.** `WorldConfig::fracture` drives the stress
   `ThresholdWorld` / `Planner`. GPU splits emit `BodiesSplit { parent,
   children }` and fragment maps, replacing the diagnostic `chunk_bodies()`.
   Destruction stays forward-only.
5. **Rendering export.**
   - HLSL target in `engine-slang-build`, so `soft_skin.slang` and `pose_pack`
     can run as Unreal `.usf` from the same source.
   - Render-asset export: fragment meshes with UVs, material ids and an
     interior-face flag, plus the fragment→body map. The GLB reader currently
     drops UVs.
   - The soft-body visual mesh with UV and material slots.
   - Native interop through `wgpu-hal`: zero-copy pose and skin buffers, and
     DX12.

## Performance at scale

Measured with `cargo run -p engine-webgpu --release --bin pose_bench -- 10000 300`:

- 10,000 boxes, persistent contacts, sleep off, so every body stays awake. This
  is the worst case for `pose_pack`.
- Timing is wall clock around a synchronous `step`.
- Off and on runs were interleaved over two rounds.
- Apple-silicon Mac, with a Chrome WebGPU page also running.

| Phase | pose_output off (median) | pose_output on (median) |
|---|---|---|
| falling (all moving) | 2.48 / 2.49 ms | 2.53 / 2.52 ms |
| resting, awake | 2.52 / 2.50 ms | 2.50 / 2.53 ms |

The cost is about 0.02–0.05 ms per step (≤ 2%), close to run-to-run noise. A
world that is asleep skips its steps, so it pays nothing. The host reads
32 B per moved body, against 112 B per body for `download_bodies`.

# Rubble public API

Status: Phase 1 and the P1.5 Box3D alignment have shipped on
`claude/public-api-world`, rebased on `main` `4a81b441`. It adds the crate `engine-world`, whose library name is `rubble`.
This document records what Phase 1 delivered, sets out the **target API**,
and plans the work that gets there.

The target is **Box3D-aligned**, with destruction designed after Blast.
The rationale for every choice is in [`api-review.md`](api-review.md). Box3D is Erin Catto's 3D successor to Box2D
v3: MIT licensed, written in C17, v0.1.0 tagged 2026-06-30. Rubble adopts its
vocabulary and data model wherever they fit. It differs only where a
GPU-resident, asynchronous engine has to.

## Why

Before this work, a host (Unreal, Godot, Unity, Bevy, a web page, a server)
had to drive `WgpuSim` directly:

- Every velocity change, impulse, wake or teleport was a **blocking** GPU
  readback (`read_body`) followed by a write (`write_bodies`). vibe-land's
  `rubble-bridge` did this for every gameplay write.
- Bodies could not be removed, so hosts "parked" them far away.
- Poses came back as the whole 112-byte `BodyState`, for every body.
- Only `with_device` took the host's GPU device, and it was fixed to
  canonical contacts.
- About 30 `ENGINE_*` environment variables chose internal paths.
- Events were split across separate polling APIs, each with its own cursor.
  There were no sleep or wake events.

## Reference engines

The research covered:

- **Box3D**, read from its headers: `include/box3d/types.h`, `box3d.h`, `id.h`.
- **Box2D**: v3 and v2.4.
- **Other rigid-body engines**: Jolt, PhysX 5, Rapier, Bepu v2 and Bullet.
- **Game-engine physics**: Unity (PhysX and DOTS) and Unreal Chaos.
- **GPU and research simulators**: MuJoCo, MJX and MJWarp, Newton, Genesis
  and AVBD.

Box3D is the primary reference for four reasons:

1. It is the newest design in the lineage.
2. It is 3D.
3. It was built to replace Chaos in a shipping Unreal game, which is the same
   job Rubble is aiming at.
4. Its API is the de-facto standard for C-ABI physics libraries.

The other engines supply what Box3D does not cover:

- GPU sharing comes from the PhysX Direct GPU API, Newton/Warp and MJX.
- Destruction events come from Chaos.
- Loud capacities come from Jolt and MJWarp.

| Concern | Box3D | Others | Rubble (target) |
|---|---|---|---|
| Handles | `b3BodyId`, `b3ShapeId` and `b3JointId` hold `{index1, world0, generation}`, are packable into a `uint64`, are null when zeroed, and come with `*_IsValid` checks. `b3ContactId` is transient. | Jolt `BodyID` (sequence number), Rapier arenas | `BodyId`, `ShapeId`, `JointId` and `ContactId` hold `{index, generation}` in a u64. Zero is null. Stale handles are refused. |
| Creation | `b3Default*Def()` copies, plus shapes attached after the body (`b3CreateHullShape(bodyId, &def, &hull)`) with `updateBodyMass`. | Jolt `BodyCreationSettings` | The same. Rubble adds batch creation so a scene uploads once. |
| Body frame | The body origin and the centre of mass are separate (`b3Body_GetLocalCenter`, `b3Body_SetMassData`). | All mainstream engines | The same: the host keeps the origin-to-centre offset and the principal rotation. The GPU solves in the centre-of-mass principal frame. |
| Body types | `b3_staticBody`, `b3_kinematicBody` (`SetTargetTransform`) and `b3_dynamicBody`. | PhysX, Jolt | The same. Kinematic needs kernel work. |
| Stepping | `b3World_Step(dt, subSteps)`, with no accumulator. Events are valid immediately after the step. | Bepu, Newton, Jolt | `step()` uses a fixed `dt` and returns without waiting. Results arrive through `poll()`, one or more frames later. `step_sync()` gives Box3D-style immediacy. |
| Transform sync | `b3BodyMoveEvent {transform, bodyId, userData, fellAsleep}` covers simulation moves only. | PhysX active actors | `BodyMoveEvent` adds the previous pose (for interpolation) and also covers moves the host makes, such as teleports. |
| Contact events | Begin, end and hit, opted into per shape (`enableContactEvents`, `enableHitEvents`). A hit carries `approachSpeed` and both `userMaterialId`s. | Rapier `ActiveEvents`, PhysX pair flags | The same shape, filtered on the GPU. |
| Sensors | `isSensor` and `enableSensorEvents`, giving begin and end events. | PhysX triggers | The same. |
| Joints | `b3JointDef` base (frames, `forceThreshold`, `torqueThreshold`, `collideConnected`, `userData`) plus revolute, prismatic, spherical, weld, distance, wheel, motor, filter and parallel defs. Threshold crossings raise joint events. | PhysX, Jolt | The same base and per-type defs. Rubble adds breakable joints (`break_force`) and bonds for destruction. |
| Materials | A `b3SurfaceMaterial` on each shape: friction, restitution, `rollingResistance`, `tangentVelocity`, `userMaterialId`; per triangle on meshes. Mixing callbacks. | — | The same struct. Rubble interns materials into its 64-slot GPU table and fails loudly when a 65th distinct one is added. No mixing callbacks: the GPU cannot call back into the host. |
| Filtering | `b3Filter {categoryBits, maskBits, groupIndex}` and `b3QueryFilter`. | Bullet, Unity DOTS | Identical (16-bit fields today). |
| Queries | Ray, closest ray, shape cast and overlap of an AABB or a shape, plus the mover (capsule cast and collide) for character controllers. | Jolt, PhysX | The same set, batched and asynchronous, with blocking variants. |
| Host hooks | Task system (`enqueueTask`), allocator, assert and log hooks; `preSolve` and custom-filter callbacks. | Jolt `JobSystem` | Not applicable: the GPU replaces the task system, and per-pair callbacks cannot run on the GPU. Rubble provides `GpuContext::Shared` instead. |
| Record and replay | `b3World_StartRecording`, a player with keyframes, `b3ValidateReplay` against state hashes. | Jolt `StateRecorder`, Rapier serde | The same: record inputs and a GPU state hash each frame; replay validates. |
| Debug, profiling | A `b3DebugDraw` callback struct, `b3Profile` and `b3Counters`. | Jolt `DebugRenderer` | The same. |
| Determinism | Cross-platform. | Jolt (opt-in), Rapier | Same build, same GPU, run to run. Stated plainly. |
| Capacity | Optional `b3Capacity` sizes for preallocation. | Jolt `Init`, MJWarp | Required `WorldCapacity`, checked against device limits, with loud overflow. |
| Large worlds | `b3Pos` positions, optional double precision. | PhysX `shiftOrigin` | f32. `World::shift_origin` is planned for city-scale worlds. |
| Destruction | Explosions only. | Chaos break, collision and trailing events | Joint and bond breaks, soft-body pieces, and GPU fracture splits (`BodiesSplit`). |

## Target API

The Rust API comes first, in `rubble::World`. The C ABI mirrors it almost one
to one, using Box3D's naming with an `rb` prefix: `rbCreateWorld`,
`rbCreateBody`, `rbCreateHullShape`, `rbBody_ApplyLinearImpulse`,
`rbWorld_GetContactEvents` and so on. Anyone who knows Box2D or Box3D should
recognise every call.

```rust
use rubble::*;
let mut world = World::new(GpuContext::Default, &WorldDef {
    gravity: [0.0, -9.81, 0.0],
    capacity: WorldCapacity::small(),
    ..WorldDef::default()
})?;

let ground = world.create_body(&BodyDef::default())?;            // static by default, as in Box3D
world.create_box_shape(ground, &ShapeDef::default(), [50.0, 0.5, 50.0])?;

let crate_ = world.create_body(&BodyDef {
    body_type: BodyType::Dynamic,
    position: [0.0, 4.0, 0.0],
    user_data: 42,
    ..BodyDef::default()
})?;
let shape = world.create_box_shape(crate_, &ShapeDef {
    density: 500.0,
    material: SurfaceMaterial { friction: 0.7, user_material_id: WOOD, ..Default::default() },
    enable_hit_events: true,
    ..ShapeDef::default()
}, [0.5; 3])?;

world.body_apply_linear_impulse(crate_, [0.0, 0.0, 200.0], [0.0, 4.2, 0.0], true)?;

loop {
    world.step()?;                                  // fixed dt, never waits
    for frame in world.poll()? {                    // finished steps, oldest first
        for m in &frame.body_events.moves { /* lerp m.previous -> m.transform */ }
        for h in &frame.contact_events.hits { /* audio by h.user_material_ids */ }
        for e in &frame.destruction_events { /* breaks, pieces, splits */ }
    }
}
```

### World

- **Creation:** `World::new(GpuContext::{Default, Shared{info, device, queue}}, &WorldDef)`.
- **`WorldDef` fields shared with Box3D:**
  - `gravity: [f32; 3]`;
  - `hit_event_threshold` and `restitution_threshold`;
  - `contact_hertz`, `contact_damping_ratio` and `contact_speed`, mapped onto the AVBD
    contact stiffness wherever there is an exact correspondence, and otherwise
    left to the advanced `solver` field;
  - `maximum_linear_speed`;
  - `enable_sleep` and `enable_continuous`;
  - `user_data`.
- **`WorldDef` fields Rubble adds:**
  - `capacity` (required);
  - `contacts` (storage layout);
  - `vehicles` and `soft` pools;
  - `solver: AvbdParams`;
  - `features: EngineFeatures`;
  - `readback_frames` and `backpressure`.
- **Stepping:**
  - `step()`;
  - `poll() -> Vec<Frame>` and `poll_blocking()`;
  - `step_sync() -> Frame`, which steps, waits and returns that step's frame.
- **Settings:** `set_gravity`, `enable_sleeping`, `enable_continuous`,
  `set_hit_event_threshold`, and the other Box3D `b3World_Set*` calls the
  solver supports.
- **Diagnostics:** `profile()` (per-phase GPU times), `counters()` (bodies,
  shapes, contacts, joints, awake count, overflow flags), and `draw(&mut impl
  DebugDraw)`.
- **Recording:** `start_recording(&mut Recording)` and `stop_recording()`;
  `Player` mirrors Box3D's `b3RecPlayer_*`.
- **Large worlds:** `shift_origin(offset)`, the PhysX-style answer to `b3Pos`.
- **Device access** (after the PhysX Direct GPU API), for hosts that render on
  the same device:
  - `pose_buffer()` holds 32-byte transforms for every body;
  - `encode_get(&mut encoder, Attribute, index_buffer, out_buffer)` and
    `encode_set(...)` do batched indexed access for `Pose`, `LinearVelocity`,
    `AngularVelocity` and `Force`, recorded into the host's own command
    encoder;
  - `gpu_index(BodyId)` gives the body's slot.

  These are additions: every feature keeps working alongside them, unlike
  PhysX, where turning direct access on turns other features off.
- **Observables** (after Newton): `observe(&[Observable])` turns on readbacks
  that cost something, namely contact forces, bond loads and utilization, and
  accelerations. Nothing is read back unless asked.
- **Statistics:** `high_water()` reports the peak usage of each capacity
  (contacts, pairs, colours, commands, pieces), so capacities can be tuned. It
  sits alongside the per-step overflow errors.

### Bodies

- **Lifecycle:**
  - `create_body(&BodyDef) -> BodyId`;
  - `create_bodies(&[BodyDef]) -> Vec<BodyId>` (batch, one upload);
  - `destroy_body` and `body_is_valid`;
  - `body_enable` and `body_disable`, which park the body and keep its handle.
- **`BodyDef`**, with Box3D's fields and defaults:

  | Field | Status |
  |---|---|
  | `body_type` (Static, Kinematic, Dynamic) | Static and Dynamic available; Kinematic is P3 |
  | `position`, `rotation` | Available |
  | `linear_velocity`, `angular_velocity` | Available |
  | `linear_damping`, `angular_damping` | P3 |
  | `gravity_scale` | P3 |
  | `sleep_threshold` | P3 (global today) |
  | `motion_locks` | P3 |
  | `is_bullet` | P3 (port the Metal CCD sweep) |
  | `enable_sleep` | Available |
  | `is_awake` | Available |
  | `is_enabled` | P1.5 (park) |
  | `allow_fast_rotation` | P3 |
  | `name` | Available |
  | `user_data` | Available |

  A field the solver does not support yet is accepted at its default value
  and refused with `Error::Unsupported` otherwise. It is never silently
  ignored.
- **Transform:** the body origin is the user frame. `body_set_transform` is a
  teleport command. `body_set_target_transform(target, dt)` drives a
  kinematic body (P3). `body_local_center` and `body_world_center` report the
  centre of mass.
- **Velocity, forces and impulses** (queued GPU commands, applied at the start
  of the next step in submission order per body, with a `wake` flag as in
  Box3D):
  - `body_set_linear_velocity` and `body_set_angular_velocity`;
  - `body_apply_force(force, point)` and `body_apply_force_to_center`;
  - `body_apply_torque`;
  - `body_apply_linear_impulse(j, point)` and
    `body_apply_linear_impulse_to_center`;
  - `body_apply_angular_impulse`;
  - `body_set_awake`.

  Forces last one step, as in Box3D, where they are cleared after each step.
- **Mass:** shapes update the mass as they are attached (`update_body_mass`,
  default true). There are also `body_apply_mass_from_shapes`,
  `body_set_mass_data(MassData {mass, center, rotational_inertia})` and
  `body_mass_data`. The host turns any inertia tensor into principal moments
  plus a rotation, so the GPU's diagonal-inertia layout stays exact.
- **Getters:** `body_transform`, `body_position`, `body_rotation` and
  `body_is_awake` return the **last polled frame** and document that latency.
  Velocities and contact data need a request:
  `request_body_states(&[BodyId]) -> Ticket`, with results in a later frame.
  A blocking `read_body_state` exists for tools and tests. Rubble has no
  synchronous getter that hides a GPU readback.

### Shapes

- **Creation:**
  - `create_sphere_shape`, `create_capsule_shape` (along +Y) and
    `create_box_shape` (a native fast path);
  - `create_hull_shape` and `create_transformed_hull_shape(hull, transform,
    scale)`;
  - later: `create_mesh_shape` and `create_height_field_shape` (static only,
    as in Box3D) and baked static compounds.

  Each takes `(BodyId, &ShapeDef, geometry)` and returns a `ShapeId`.
- **`ShapeDef` fields:**
  - `material: SurfaceMaterial`, interned into the GPU table;
  - `density`;
  - `filter: Filter`, `is_sensor` (P4) and `enable_sensor_events` (P4);
  - `enable_contact_events` and `enable_hit_events` (P4);
  - `update_body_mass`, `name` and `user_data`.

  Box3D's `enablePreSolveEvents` and custom filtering are not offered,
  because per-pair host callbacks cannot run on the GPU.
- **Editing:**
  - `destroy_shape(update_body_mass)`;
  - `shape_set_filter`, `shape_set_surface_material`, `shape_set_density`;
  - `shape_enable_*_events`;
  - `shape_body` and `shape_aabb`.

### Joints

- **Base:** `JointDef` holds `body_a`, `body_b`, `local_frame_a`,
  `local_frame_b`, `force_threshold`, `torque_threshold`,
  `collide_connected` and `user_data`. Frames are measured from each body's
  origin, as in Box3D.
- **Per-type definitions,** mapped onto Rubble's joint presets:

  | Definition | Rubble preset |
  |---|---|
  | `RevoluteJointDef` | hinge |
  | `PrismaticJointDef` | slider |
  | `SphericalJointDef` | ball or cone |
  | `WeldJointDef` | fixed |
  | `DistanceJointDef` | distance, spring or max-distance |
  | `WheelJointDef` | — |
  | `MotorJointDef` | — |
  | `FilterJointDef` | disables collision only |

- **Rubble-only definitions:** `D6JointDef` and `BondJointDef`, plus
  `break_force` and `break_torque` on any joint.
- **Thresholds and breaks:** a threshold crossing raises a `JointEvent`, as in
  Box3D. Exceeding a break limit breaks the joint and raises
  `JointBroke`.
- **Control:** `joint_set_motor_speed`, `joint_set_target_*`, `joint_set_limits`
  and `joint_constraint_force` / `joint_constraint_torque` (from the frame).

### Events

`Frame` holds one step's results. It is valid until the host drops it.

- **`body_events.moves`** is a list of `BodyMoveEvent`s: `body`, `transform`,
  `previous: Option<Transform>`, `fell_asleep`, `woke` and `user_data`.
- **`contact_events`**:
  - `begins`: `shape_a`, `shape_b` and `contact`;
  - `ends`: the same fields;
  - `hits`: `shape_a`, `shape_b`, `point`, `normal`, `approach_speed` and
    `user_material_ids`.
- **`sensor_events`**: `begins` and `ends`, each naming the sensor shape and
  the visitor shape.
- **`joint_events`**: threshold crossings.
- **`destruction_events`**:
  - `JointBroke`;
  - `BondBroke`;
  - `MountBroke`;
  - `BeamBroke`;
  - `PieceDetached`;
  - `BodiesSplit {parent, children}`.
- **Overall:**
  - `lost_events` counts records the GPU ring overwrote before they were read.
    It is never silently zero.
  - `counters` holds the step's counts.
  - `dropped` marks a frame whose readback was skipped under backpressure.

### Queries

- **Batched queries:**
  - `cast_ray`, `cast_ray_closest`, `cast_shape`;
  - `overlap_aabb`, `overlap_shape`.

  Each takes a `QueryFilter {category_bits, mask_bits}` and is submitted as a
  batch that returns a `Ticket`. Results arrive with a later frame. Each has a
  `_blocking` variant.
- **Character mover:** `cast_mover` and `collide_mover` (capsule), with
  `solve_planes` on the host, after Box3D's mover.
- **`explode(ExplosionDef {position, radius, falloff, impulse_per_area,
  mask_bits})`** is a GPU command.

### Destruction (first-class)

Destruction is Rubble's differentiator, so it gets its own part of the API,
designed after NVIDIA Blast. Chaos supplies anchors, debris policy, the root
proxy and event opt-ins. APEX supplies budgets and replication. The rationale
is in [`api-review.md`](api-review.md).

#### Asset and instance

**`DestructibleAsset`** is the immutable model, loaded from a Shardwright GLB:

```rust
let asset = world.load_destructible_asset(&glb_bytes)?;
```

It holds:

- the chunk hierarchy (levels);
- the support graph, including world bonds;
- each bond's patch geometry and strength capacities, derived from the material
  library;
- the materials;
- the render mapping: chunk → glTF node, interior material, root proxy.

Invalid assets are refused. They are never repaired silently.

**`DestructibleDef`** describes one placed instance:

| Field | Meaning |
|---|---|
| `asset` | The asset to place. |
| `transform` | Where to place it. |
| `support_level` | The hierarchy level that is simulated. |
| `model: StrengthModel` | How bonds fail (see below). |
| `strength_scale` | Multiplier on bond strengths. |
| `anchors: Anchors` | `AssetWorldBonds`, `None` or `Boxes(&[Aabb])`. |
| `initial_health` | Optional starting health. |
| `budget: DestructionBudget` | Limits on pieces, splits and debris. |
| `events: DestructionEventMask` | Which destruction events to report. |
| `user_data` | Echoed back in events. |

```rust
let id = world.create_destructible(&def)?;
```

This returns a `DestructibleId`. Each intact cluster is a body, an actor in
Blast's terms, reachable as a `BodyId`.

**`StrengthModel`** picks how bonds fail. The same API serves all three:

- **`Threshold`** is the default and fully GPU-resident (`GpuFracture`).
  Contact and impact load is compared against bond capacity, with strike
  prediction, and splits go into pre-allocated body slots.
- **`Stress(StressSettings)`** is Blast-parity: a quasi-static stress solve
  with `compression`, `tension` and `shear` limits, each with an elastic and a
  fatal value, plus an iteration budget. It is the charter baseline profile
  `blast_parity`. It is host-coupled until island frames move to the GPU.
- **`Bonds`** turns each bond into an AVBD joint, as in the fracture lab. It
  supports yield and bend limits, and every fragment is its own body.

#### Damage as data

```rust
world.apply_damage(id, &DamageDesc::Radial { position, min_radius, max_radius, damage })
```

- **Descriptors:** `Radial`, `Capsule`, `Shear{normal}`, `Impact{point,
  impulse, radius}` and `Cut{plane}`.
- Each descriptor is a GPU damage program, with Blast's shader set as the
  reference.
- Damage is a queued command applied before the next step, so fracture stays
  forward-only.
- `explode(ExplosionDef)` damages destructibles in range and also applies
  impulses to bodies.

#### The three stages, all on the GPU

Damage, stress and threshold loads all go through the same stages, as in
Blast:

1. **Generate:** loads are turned into fracture commands, of the form
   `{bond, health_delta}`.
2. **Apply:** bond health is updated, and bonds that break are recorded.
3. **Split:** connected components are found with union-find, the anchored
   component keeps the original body, and new pieces take pre-allocated slots.
   Mass, centre of mass and inertia are summed in fixed point, so the result is
   deterministic.

#### Fracture commands are replicable

- **Server side:** `Frame::fracture_commands` lists each step's applied
  commands as flat, compact records.
- **Client side:** `world.apply_fracture_commands(id, &cmds)` replays them.
  Applied to the same instance state, they produce the same breaks and the same
  pieces. That is the netcode unit for vibe-land: send commands, not chunk
  transforms. Body poses still need correction snapshots across different GPUs.
- `world.set_fracture_authority(id, Authority::Local | Replicated)`: a
  replicated instance only breaks through commands it is given.

#### Destruction events

These arrive in `Frame::destruction_events`, filtered at the source by
`DestructionEventMask`:

- **`BondsBroken`:** `{destructible, count, bonds: [BondId; ≤detail]}`. The
  count is exact; the list stops at the detail limit, after APEX.
- **`Split`:**
  - `destructible`;
  - `parent: ActorInfo`, passed by value: body, chunk count, mass, and whether
    it was anchored;
  - `children: [ActorInfo {body: BodyId, chunks, mass, com, linear_velocity,
    anchored}]`.
- **`DebrisRemoved`:** `{body, reason: Lifetime | Sleep | Budget}`. This only
  happens when a debris policy asks for it.
- **`BudgetExceeded`:** `{destructible, kind, dropped: 0}`. With the default
  `Fail` policy, the step that hits the limit returns an error instead.
- **Joints:** `JointBroke` and `BondBroke`, from the `Bonds` model and from any
  breakable joint.
- **Soft bodies and vehicles:** `BeamBroke`, `PieceDetached` and
  `MountBroke`.

#### Queries

These are frame-latent by default; blocking variants exist for tools.

- `chunk_body(id, chunk) -> BodyId`
- `actor_chunks(body) -> &[ChunkIndex]`
- `bond_health(id, bond)`
- `is_anchored(body)`
- `destructible_stats(id)`: intact bonds, pieces, free slots.

#### Anchors

- **At creation:** from the asset's world bonds.
- **At runtime:** `set_anchored_by_box(id, aabb, anchored)` and
  `remove_all_anchors(id)`, after Chaos. Removing an anchor triggers a normal
  forward split on the next step.

#### Budgets and debris

`DestructionBudget` has these fields:

| Field | Meaning |
|---|---|
| `max_pieces` | The pre-allocated slot pool. |
| `max_splits_per_step` | Cap on splits in a single step. |
| `over_budget: Fail \| Defer` | `Defer` carries the remaining splits to the next step, in order. |
| `debris: Option<DebrisPolicy>` | `{min_mass, lifetime, remove_on_sleep, one_way_below_mass}`. |

- **The defaults are lossless:** `Fail` loudly, and no debris removal.
- Removing debris is lossy, so it is opt-in, and every removal is reported.

#### Rendering

- **Pose buffer:** `destructible_pose_buffer(id)` holds a per-chunk transform
  buffer on the GPU, written by the split kernels (`fr_view_*`). It is meant
  for zero-copy rendering on a shared device.
- **Root proxy:** each intact cluster draws as its root proxy until it breaks.
- **Render assets:** `render_assets(asset)` returns, per chunk, the mesh with
  UVs, material ids and an interior-face flag. This comes in P8 and needs
  Shardwright export fields.

#### Observables (opt-in readback)

`world.observe(Observable::BondUtilization | BondLoads | ContactForces)`
enables a readback for heat maps and debugging. Computing these values is part
of the solve; only reading them back costs anything.

### Vehicles and soft bodies (Rubble-only)

- **Vehicles:** `add_vehicle`, `set_vehicle_input`, `vehicle_state`.
- **Soft bodies:** `add_soft_vehicle`, with skin output through
  `SoftSkinner`.
- Their bodies are reachable as `BodyId`s, but the vehicle or soft body owns
  them.

### Rubble-specific guarantees

1. **Lossless:** a world that does not use a feature computes exactly what it
   computed before. Every phase is tested byte for byte against raw
   `WgpuSim`.
2. **Asynchronous by default:** stepping never waits on a readback. Getters
   are frame-latent, and that is documented.
3. **Zero-copy option:** `pose_buffer()` holds 32-byte poses that renderers on
   a shared device can read in place.
4. **Loud limits:** capacities are checked at creation and on overflow.
5. **Determinism:** same build, same GPU, run to run. Command order per body
   is the submission order.
6. **Forward-only destruction:** fracture is never decided by re-simulating a
   step.
7. **Coordinates:** Rubble has no up axis; gravity is a vector. A Z-up option
   in the C ABI converts exactly at the boundary.

## Phase 1: shipped

Phase 1 is lossless throughout: it is byte-identical to raw `WgpuSim` across
all contact modes, with and without vehicles.

1. **`rubble::World`** with:
   - `GpuContext::{Default, Shared}` and `WorldConfig`;
   - generational `BodyId` and `JointId`;
   - `create_body(&BodyDef, &[ShapeDef])` and `remove_body`, which parks the
     body, turns its colliders inert and reuses the slot after two steps;
   - body commands;
   - joints from any `JointDesc`;
   - vehicles and soft bodies;
   - blocking ray casts;
   - `step` and `poll` returning `Frame`s with moved bodies and events.
2. **Every configuration takes the caller's device.**
   `WgpuSim::with_device_options` takes `SimOptions {contacts, vehicles, soft,
   features, pose_output}`. The device limits are checked. `EngineFeatures`
   replaces the `ENGINE_*` environment reads; `SimOptions::from_env()`
   resolves them as before.
3. **Body commands without readback.** Wake, teleport, set velocity, impulse
   at a point and one-step force go through `apply_commands`. One thread
   handles each body's run, so results match a host read-modify-write bit for
   bit.
4. **`pose_pack.slang`** writes 32-byte poses and the moved list with SLEPT,
   WOKE and FIRST bits. At 10k awake bodies it costs about 0.02–0.05 ms per
   step, which is 2% or less.
5. **Unified events.** One ring copy is decoded for joint, mount and beam
   breaks and soft-body pieces. Sleep and wake come from the moved list. Lost
   records are counted.
6. **Asynchronous results.** A staging ring with `Block` or `Drop`
   backpressure delivers results. Pipelined steps are byte-identical to
   synchronous ones (gate test).

The tests are in `engine-world/tests/{world,bridge_acceptance}.rs` and
`engine-webgpu/tests/{body_commands,pose_output,device_options,pipelined_gate}.rs`.
`pose_bench` measures the cost of `pose_pack`.

### Known gaps found on the way

- **Asleep colliders are invisible to rays.** A collider added asleep to a
  world that has already stepped gets no ray bounds. Raw `WgpuSim` misses it
  too. The fix is `af24d153` on vibe-land's `feat/vibe-land-backend` branch,
  and it is not merged. The matching test is `#[ignore]`d.
- **Large statics beside small bodies.** A city of large statics next to
  small dynamic bodies needs `EngineFeatures::grid_levels = Some(true)`. On the
  default grid the overflow is reported as an error.
- **Parallel 100k-body tests hang.** The two 100k-body `columns_hold` tests
  hang when they run in parallel. This also happens on `main`, and each passes
  alone.

## Plan

Each phase lands with byte-equality tests against raw `WgpuSim` for worlds
that do not use its features.

**P1.5: Box3D alignment (shipped).** `rubble::World` now has this shape:

- **Definitions:** `WorldDef`, and `BodyDef` with Box3D's fields and a static
  default.
- **First-class shapes:** `ShapeId` and `create_{sphere,capsule,box,
  transformed_box,hull,transformed_hull}_shape`; `destroy_shape` reuses
  colliders after a quarantine.
- **Surface materials:** `SurfaceMaterial`, interned into the 64-slot table,
  with a loud error on the 65th distinct one.
- **Mass and frames:** `MassData`, `body_set_mass_data` and
  `body_apply_mass_from_shapes`.
  - The body origin is separate from the centre of mass and principal axes.
    Transforms are reported at the origin.
  - A live body re-centres on the GPU through a new `SetMassFrame` command,
    with no readback.
- **Box3D commands:** separate linear and angular velocity, force at a point,
  torque, impulse at a point or at the centre, angular impulse, and a
  `wake = false` flag that leaves sleeping bodies alone.
- **Typed joints:** revolute, prismatic, spherical, weld, distance and filter
  definitions, with Box3D axes (revolute turns about frame z), plus `Custom`
  for Rubble's full joint model. Springs take physical stiffness.
- **Grouped events:** `Frame { body_events.moves, destruction_events }`.
- **Getters and helpers:** frame-latent getters, `read_body_state`,
  `step_sync`, `create_bodies` (batch), `cast_ray_blocking` with shape ids.
- **Errors:** `Unsupported` and `StaleHandle` can be matched with downcast.

Deferred from P1.5:

- joint threshold events (which need a GPU check);
- `body_enable` / `body_disable`;
- `request_body_states`;
- device access, observables and `high_water()`.

The original P1.5 list follows.

- Rename the types:
  - `WorldConfig` becomes `WorldDef`;
  - `BodyKind` becomes `BodyType`, with a Kinematic variant that returns
    `Unsupported` until P3;
  - `MaterialDef` becomes `SurfaceMaterial`, with `rolling_resistance`,
    `tangent_velocity` and `user_material_id`;
  - `Event` is split into `body_events`, `joint_events` and
    `destruction_events`.
- Make shapes first-class: `ShapeId`, `create_*_shape(body, &ShapeDef,
  geometry)`, `destroy_shape`, `update_body_mass`. Material interning replaces
  explicit material slots.
- Separate the body frame from the solver frame. The host keeps the
  origin-to-centre offset and the principal rotation, applies them to poses,
  commands and joint frames, and reports transforms at the body origin. This
  removes Phase 1's rule that shapes must be centred on the centre of mass.
- Add `MassData`, `body_set_mass_data` and `body_apply_mass_from_shapes`.
- Add the full `BodyDef` and `ShapeDef` field sets. Unsupported non-default
  values return `Error::Unsupported`. Add `body_enable` and `body_disable`
  (park), and `create_bodies` (batch).
- Add per-type joint definitions over `JointDesc`, with `force_threshold` and
  `torque_threshold`. Joint events need a GPU threshold check emitting a ring
  record.
- Add command variants with a `wake` flag, plus `body_apply_torque` and
  `body_apply_angular_impulse`.
- Add frame-latent getters, `request_body_states`, `step_sync`, `counters()`
  and `profile()`.

**D1: destructibles in `World`, threshold model.** This follows P1.5, before
the C ABI, because destruction is the reason to choose Rubble.

- `load_destructible_asset` (Shardwright GLB) and `create_destructible`, built
  on `GpuFracture` / `FractureTables`, with pre-allocated slots taken from
  `WorldCapacity`.
- Body handles for the pieces.
- `Split` and `BondsBroken` events read from the split log in the frame
  readback.
- `chunk_body`, `actor_chunks` and `destructible_stats`.
- Budgets with `Fail`.
- The `ThresholdWorld` demo is rebuilt on top of this.

**D2: damage descriptors and explosions.** GPU damage programs (radial,
capsule, shear, impact, cut) produce fracture commands. `explode` applies
impulses and damage.

**D3: fracture commands.** Record them per frame,
`apply_fracture_commands`, authority modes, and a two-world replay test
(server and client produce the same topology).

**D4: anchors, budgets and debris policy.** Runtime anchors; `Defer`;
`DebrisPolicy` with reported removals.

**D5: the stress model.** `StrengthModel::Stress` with `blast_parity`,
host-coupled until R5 and island frames run on the GPU. Bond-utilization
observables.

**P2: C ABI (`engine-ffi`).** A cbindgen header mirrors P1.5 with Box3D
naming (`rb` prefix). It has:

- `rbDefault*Def()` functions;
- `uint64_t` handles with `rbStore`/`rbLoad` and `RB_IS_NULL`;
- event arrays valid until the next poll;
- error results plus `rbGetLastError()`;
- `catch_unwind` at the boundary;
- the Z-up option.

Unreal is the first client.

**P3: Box3D body features (new Slang, opt-in, defaults lossless).**

- Kinematic bodies and `set_target_transform`, which must push dynamic bodies
  and carry friction.
- A full gravity vector, keeping the Y-only path bit-identical.
- `gravity_scale`, linear and angular damping, and per-body `sleep_threshold`.
- `motion_locks`.
- `is_bullet`: port the Metal CCD classify and sweep passes to the shared
  plan.
- `allow_fast_rotation`.

**P4: contact, hit and sensor events.**

- Opt-in per shape. Pairs are flagged on the GPU, and begin/end come from a
  "matched last step" bit that must work with `carry_rest` and manifold reuse.
- Hit events above `hit_event_threshold` carry the approach speed and the
  material ids, and are filtered on the GPU.
- Sensor shapes (no response) get begin and end overlap events.
- `explode`.

**P5: queries and mover.** Batched asynchronous ray, shape-cast and overlap
queries with `QueryFilter`, moved off the vehicle ray pool. Then capsule
`cast_mover` and `collide_mover`.

**P6: tooling.**

- Recording and replay of inputs plus a per-frame GPU state hash, validated on
  replay.
- A `DebugDraw` callback struct fed by one readback of colliders and poses.
- `shift_origin`.

**P8: rendering export.**

- An HLSL target in `engine-slang-build`, so `soft_skin` and `pose_pack` can
  run as Unreal `.usf`.
- Render assets: fragment meshes with UVs, material ids and an interior-face
  flag, and the soft-body visual mesh.
- Native-device interop through `wgpu-hal` (zero-copy, DX12).
- Static mesh and height-field shapes, if the kernels gain them.

### Mapping from the original draft

| Draft item | Where |
|---|---|
| 1. One world API | P1 (done) and P1.5 |
| 2. Device ownership | P1 (done) |
| 3. Render-pose output | P1 (done) |
| 4. Unified events and lifecycle | P1 (done); fracture splits in D1 |
| 5. Impact events | P4 (Box3D hit events) |
| 6. HLSL target | P8 |
| 7. Native interop | P8 |
| 8. Scene queries | P5 |
| 9. Render-asset export | P8 |
| (C ABI) | P2 |

## Performance at scale

Measured with `cargo run -p engine-webgpu --release --bin pose_bench -- 10000 300`:

- 10,000 boxes, persistent contacts, sleep off, so every body stays awake.
  This is the worst case.
- Timing is wall clock around a synchronous `step`, with off and on runs
  interleaved over two rounds.
- The machine was an Apple-silicon Mac, with a Chrome WebGPU page also
  running.

| Phase | pose_output off (median) | pose_output on (median) |
|---|---|---|
| falling (all moving) | 2.48 / 2.49 ms | 2.53 / 2.52 ms |
| resting, awake | 2.52 / 2.50 ms | 2.50 / 2.53 ms |

That is about 0.02–0.05 ms per step, 2% or less and close to noise. An
asleep world skips its steps entirely. The host reads 32 bytes per moved
body, against 112 bytes per body from `download_bodies`.

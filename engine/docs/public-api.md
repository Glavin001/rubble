# Rubble public API

Status: Phase 1, the P1.5 Box3D alignment and the P1.5b output fixes have
shipped on `claude/public-api`, rebased on `main` `4a81b441`. The plan is
revised from a prospective developer's review (see "Developer feedback"). It adds the crate `engine-world`, whose library name is `rubble`.
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
  | `is_bullet` | ✅ `BODY_CCD` (#52's continuous collision) |
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
  - `create_height_field_shape` (static only, as in Box3D; editable in place
    with `set_height_field_region`, see `deformable-terrain.md`); a field may
    carry a Bekker soil (`HeightField::with_soil`, `set_height_field_soil`, C
    ABI `rbHeightField_SetSoil`) that vehicle wheels sink into and rut
    (`deformable-terrain.md`, "T5 ruts"); and its ground can deform
    (`enable_terrain`, `terrain_impact` / `terrain_explode` / `terrain_open`,
    `terrain_watch_shape` for heavy-body strikes, `read_height_field`; C ABI
    `rbWorld_EnableTerrain`, `rbTerrain_*`, `rbHeightField_GetHeights`;
    `deformable-terrain.md`, "T4 status");
  - later: baked static compounds.
  - **No triangle-mesh shapes** (decided 2026-10-24). Level geometry comes in
    convex-decomposed at the source (several hull shapes on a static body),
    the way Unreal's own collision is authored. Nothing is decomposed at
    load time.

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

### Frame outputs and delivery (P1.5b)

**Per-step contents.** Every `Frame` also carries:

- `vehicles` (each vehicle's state);
- `soft` (node positions of soft bodies with output enabled);
- `skins` (skinned meshes).

**Dropped frames.** `Backpressure::Drop` is lossless. The frame after a
dropped one copies every published pose and reports each body that changed
since the last delivered frame, including bodies that fell asleep in between.
It sets `resynced`.

**Coalesced delivery.** `poll_coalesced()` folds several steps into one result
for hosts that step faster than they draw, such as 240 Hz soft scenes:

- the latest transform;
- the earliest previous transform;
- the net sleep change;
- events in order, each tagged with its step.

**Shape geometry.** `shape_geometry` and `shape_mesh` return a shape's
geometry, and a triangle mesh for box and hull shapes, in the body frame.

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

**`DestructibleAsset`** is the immutable model. It is built from a
**`DestructibleAssetDesc`**, after Blast's `NvBlastAssetDesc`, and that
descriptor is the API's one input. A desc holds:

- chunks: hierarchy, hulls and mass;
- bonds: patch geometry, capacities and world anchors;
- materials.

Loaders produce descriptors, so no source is special:

- `DestructibleAssetDesc::from_glb(&bytes)` reads a Shardwright GLB;
- `from_pack(&json)` reads the vibe-city and blast-stress packs;
- `cuboid`, `row` and `grid` build the fracture lab's procedural scenes;
- hosts can also fill a descriptor themselves.

```rust
let asset = world.create_destructible_asset(&DestructibleAssetDesc::from_glb(&glb)?)?;
```

It holds:

- the chunk hierarchy (levels);
- the support graph, including world bonds;
- each bond's patch geometry and strength capacities, derived from the material
  library;
- the materials;
- the render mapping: chunk → glTF node (in D1), with the interior material
  and root proxy to follow.

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

- **`Threshold`** is the fracture lab's `threshold` mode, fully GPU-resident
  (`GpuFracture`, shipped in D1b). Each bonded cluster is one body. Contact and
  impact loads are compared against bond capacity, with strike prediction for
  bodies marked as projectiles (`body_set_projectile`), and splits go into body
  slots reserved when the destructible is placed.
- **`Stress(StressSettings)`** is Blast-parity: a quasi-static stress solve
  with `compression`, `tension` and `shear` limits, each with an elastic and a
  fatal value, plus an iteration budget. It is the charter baseline profile
  `blast_parity`. It is host-coupled until island frames move to the GPU.
- **`Bonds`** turns each bond into a breakable AVBD bond joint, the fracture
  lab's default. It supports yield and bend limits, and every fragment is its
  own body. It shipped in D1a.

#### Render contract

- **Every piece body uses the destructible instance's frame as its body
  origin.** Its centre of mass and principal axes are the body's solver frame,
  which is the origin/centre-of-mass split P1.5 built.
- A chunk's world transform is therefore always `body_transform *
  chunk_local`, where `chunk_local` is fixed in the asset. A split never
  requires a renderer to rebuild meshes.
- Each child's centre of mass and axes come back in the split log.

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

#### Directed debris (opt-in gameplay force, shipped)

Authored zones that debris tends to land in: around an NPC, on a parked car,
in a third-floor window. Players should see only physics. **This is not
physics**: it adds momentum, so it is off unless the world asks for it.

- **Opt in twice.** `WorldDef::guidance: Some(GuidanceCapacity { max_zones,
  plan_budget })` reserves the buffers (`plan_budget` at most 256), and
  `set_guidance(&GuidanceSettings)` enables it. Then guide bodies with `body_set_guided(id, true)`,
  `body_set_guided_tags(id, mask)` or `destructible_set_guided(id, mask)`.
  A world that does neither encodes no guidance passes, and its steps are
  bitwise the same as before.
- **Zones** (`set_guidance_zones`): `Point`, `Sphere`, `Box`, `Ring` (an
  annulus: debris around a subject, not on it) or `Arc` (behind or beside it).
  - A zone sits at a world point or on a body. A body anchor moves the zone
    with that body and leads it by its velocity; `anchor_relative` turns the
    zone with the body's heading (vehicle +Z forward, +X left).
  - Each zone has a `weight`, an optional `quota`, an `active` time window and
    `tags` matched against each body's tag mask.
  - **Known motion** (`motion: ZoneMotion`). By default a zone keeps its
    anchor's current velocity, which misses a target that turns before the
    debris lands: re-plans are held to `max_jerk`, too slow to follow a
    reversal. `Accelerating { accel }` adds a known constant acceleration
    (a braking car). `Path { keys }` gives up to `GUIDE_PATH_KEYS` (8)
    (position, world time) keys, joined by straight segments and extended past
    both ends: a car's route, a patrol. The plan then aims at where the zone
    will be at the hit time, turns included, with the same kick and held
    acceleration limits. An anchored zone moves from its body as the path
    moves from where it is now, so it stays on the body if the body drifts
    off the path. Upload keys covering the next few seconds; refresh them as
    the route goes on. Anchor-relative zones keep the body's current heading.
    In `engine-world/tests/guidance.rs` a target that turns back 1.2 s into
    a 2.85 s flight is missed by 3.6 m with velocity extrapolation and hit
    within 0.04 m with its path.
  - More zones than `max_zones` is an error, and so are path keys out of
    time order or more than 8.
- **How it stays invisible.** A guided body is planned once, when it becomes
  free (no joints, nothing pushing it for `settle_steps`).
  - **Aim points:** per zone, a seeded point inside it, then points two
    thirds, a third and none of the way from the point of the zone nearest
    the body's unguided landing toward it. The body takes the first it can
    reach, so a burst spreads over the zone and a body that can only just
    reach it lands at its near edge. Each is met at the time that needs the
    least of the budget (for a zone on a fast car that is near the landing,
    not the arc's closest approach in mid-air).
  - **Zone choice:** each reachable zone scores `weight · (1 − s)²`, where `s`
    is the fraction of the budget its easiest aim point needs. The body then
    picks a zone by a seeded draw.
  - **Launch kick:** spread over `kick_steps` steps, at most `launch_kick` of
    the launch speed and `launch_cone` of direction change. It is applied only
    within `launch_window` of the break, where the break's chaos hides it.
  - **Held acceleration:** at most `accel_max_g · g`, constant for the whole
    flight. Gravity plus a constant acceleration is still an exact parabola,
    so the arc never bends or homes in. Only zones anchored to a moving body
    are re-planned, every `replan_interval`, within `max_jerk`.
  - **Defaults** (6 %, 3°, 0.1 s, 3 %·g) shift a 3 s flight's landing by
    about 4–5 m and a 0.5 s hop by a few centimetres. Hops shorter than
    `min_flight` are left alone.
- **One arc only.** A push is a step whose velocity differs from the
  ballistic update by more than `impact_dv`. Look-ahead contacts with
  neighbours flying alongside do not count.
  - The first push on a steered body ends its guidance for good.
  - A body still waiting for a plan that is bumped inside a debris cloud just
    waits for a clean flight. Pushed within 0.5 m of the ground, or left slower
    than `min_speed`, it has landed and is never guided.
  - Bounces, cascades and secondary breaks are pure physics.
- **Cost is bounded.** Per step:
  - one scan of the bodies (one word for an unguided body);
  - at most `plan_budget` plans, one 64-lane workgroup each, with the lanes
    splitting the zones (four aim points of 32 arc samples per zone);
  - a single-workgroup select and commit.

  A large burst is planned over several steps instead of spiking one. A body
  planned after its launch window gets the held acceleration only. Selection
  and quota claims resolve in body order, so results are deterministic.

  Measured with `guide_bench` on an M-series GPU, 256 zones, every guided body
  freed in the same step (`RUBBLE_GPU_TIMING=1`), the guide phase takes about
  0.13 ms per step at 10k bodies and 0.32–0.36 ms at 100k while the queue drains
  (256 plans a step, each weighing four aim points per zone). It was 0.03–0.05 ms
  with one seeded aim point per zone, which missed zones on fast-moving anchors.
  Zones without a path cost the same; paths cost only where used. With every
  one of the 256 zones on an 8-key path (`GUIDE_BENCH_PATHS=1`), the guide
  phase takes about 0.9 ms at 100k while the queue drains (0.07 ms at 10k).
- **Reads (blocking, for tools and tests).** `body_guidance(id)` returns a
  body's state, zone, held acceleration, hit time and why its guidance ended
  (`GuideEnd`); `guidance_claims()` returns each zone's claimed bodies.
- **Demos.** The fracture lab's Directed debris (Near Miss rings, city props);
  `vehicle.html?scene=gauntlet`, a drivable truck down a shelled street, with
  a ring around the truck, an arc behind it and the parked cars' roofs as
  zones; and `vehicle.html?scene=meteors`, the same volleys into a steered and
  an unsteered lane of target pads (`&targets=moving` puts the pads on carts
  that turn back every 6 s; *Known routes* gives the steered zones the carts'
  paths). All use the same passes through `RigidStep` / the lab planner.
- **C ABI:** `RbWorldDef::enable_guidance`, `rbWorld_SetGuidance`,
  `rbWorld_SetGuidanceZones` (with `RbGuidanceZone::accel` and `path`),
  `rbWorld_ResetGuidanceClaims`,
  `rbBody_SetGuided` and `rbDestructible_SetGuided`. Values are in the world's
  coordinates; Unreal yaw and arc angles convert exactly.
- **Code:** `slang/kernels/guide_step.slang`, `engine-world/src/guidance.rs`
  and `engine-world/tests/guidance.rs`. It runs on the wgpu hosts (`World`
  and `WgpuSim`); the Metal and CUDA benchmark hosts do not encode it.

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

### Assemblies and vehicles (Rubble-only, planned V1)

A vehicle is not a separate kind of object. It is an **assembly** of parts
with a controller on top.

**Parts.** Each part chooses a physical model:

- `Rigid`: bodies and shapes;
- `Soft`: a node-beam lattice that bends, yields and tears;
- `Destructible`: a D1 instance that fractures.

**Connections.** Each part also chooses how it attaches:

- `Core` joints;
- breakable `Mount`s;
- `Bonds`;
- any `JointDef`.

**Vehicles.** A `VehicleDef` adds wheels, drivetrain and steering to an
assembly whose rigid **core** carries them. That is how every vehicle kernel
already works, and the hybrid dune buggy is a soft frame with bolted rigid
parts:

```rust
world.add_vehicle(&VehicleDef {
    chassis: PartDef::rigid(body, shapes),
    parts: vec![
        PartDef { name: "frame".into(), model: PartModel::Soft(lattice), attach: Attach::Core },
        PartDef { name: "door_l".into(), model: PartModel::Rigid { .. }, attach: Attach::Mount { to: "frame".into(), break_force: 8e3, break_torque: 2e3 } },
        PartDef { name: "hood".into(),  model: PartModel::Destructible(def), attach: Attach::Bonds { to: "frame".into() } },
    ],
    wheels, drivetrain, steering,
})
```

- **Presets and imports:** the presets (`sedan`, `destructible_sedan`, the
  dune buggy) and the JBeam and Rigs of Rods imports become `VehicleDef`
  builders. `add_soft_vehicle` is folded in.
- **Assemblies without a controller:** `create_assembly(&AssemblyDef)` covers
  mechs, cranes, ragdolls and pre-rigged structures.
- **Breakage reporting:** everything that breaks reports through
  `destruction_events`, keyed by part.

**Available today, P1.5b:**

- `add_vehicle` (rigid) and `add_soft_vehicle`;
- `set_vehicle_input` and `set_soft_vehicle_input`;
- per-frame `Frame::vehicles` (wheel poses, suspension, spin, contact, rpm,
  gear);
- soft output through `soft_enable_output` (node positions per frame) and
  `soft_bind_skin` (GPU-skinned vertices and normals per frame), usable by
  hosts on any device;
- `soft_damage`.

Soft-body particles never appear as body moves.

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

### MVP: demo parity for Unreal (in progress)

The bar, set by the user on 2026-10-24: everything the shipped web pages do
(`demo.html`, `vehicle.html`, `softbody.html` and the fracture lab) can be
done from Unreal through the C ABI, as is. The pages drive raw GPU records
today; each feature below gets a World call and an ABI call instead. Status:
✅ shipped, ⏳ next (this order), 🔜 later in the MVP.

| Page feature | World / ABI | Status |
|---|---|---|
| Static and dynamic bodies, initial velocity, sleep, user data | `create_body`, `rbCreateBody` | ✅ |
| Sphere, capsule, box, hull and compound bodies; filters; friction and restitution | `create_*_shape`, `rbCreate*Shape` | ✅ |
| Ground plane (legacy scenes, fracture lab) | `Geometry::Plane` on static bodies, `rbCreatePlaneShape` (at most 8) | ✅ |
| Projectiles, cannons, rams (teleport plus velocity) | `rbBody_SetTransform`, `rbBody_SetLinearVelocity`, create and destroy | ✅ |
| Body velocity and awake state for HUDs and colouring | `rbBody_GetVelocity` (blocking), `rbBody_IsAwake` | ✅ |
| Hand-set mass and inertia from C | `body_set_mass_data`, `rbBody_SetMassData` / `rbBody_GetMassData` (tensor converted for Unreal) | ✅ |
| Runtime gravity | `rbWorld_SetGravity` | ✅ |
| Joints: ball, cone and twist limits, hinge limits, slider, weld, spring, distance and tether, motors, servo, breakable | typed joint defs | ✅ |
| Live motor speed and servo target (motor toggle, servo sweep) | `joint_set_drive`, `rbJoint_SetDrive` | ✅ |
| Plastic welds (`with_yield`) | `WeldJointDef::plastic`, `RbWeldJointDef::yield_torque` / `break_bend` | ✅ |
| Shape edits (friction, filter) | `rbShape_SetFriction`, `rbShape_SetFilter` | ✅ |
| Every vehicle preset; descriptors edited by the host (mount strength, track pods) | `rbCreateVehiclePreset`, `rbGetVehiclePresetJson`, `rbCreateVehicleFromJson` (the serde `VehicleDesc`) | ✅ |
| Vehicle input with gear requests | `RbVehicleInput::manual_gear` / `gear` | ✅ |
| Telemetry: speed, rpm, gear, clutch, parts lost, per-wheel contact, slip, suspension force, detached | `RbVehicleOutput`, `RbWheelState` (every frame) | ✅ |
| Vehicle damage: detach or disable wheels and parts, functional state, mount loads | `vehicle_damage`, `rbVehicle_Damage`, `rbVehicle_SetFunctionalState`, `rbVehicle_GetMountLoads` | ✅ |
| Soft vehicles: `.truck`, JBeam, procedural car, dune buggy, descriptors as JSON; spawn velocity and `rigid_core` | `rbCreateSoftVehicle`, `rbCreateSoftVehicleJBeam`, `rbCreateSoftVehicleTruck`, `rbGetSoftVehiclePresetJson` | ✅ |
| Soft settings (`rigid_mode`, threshold, sweeps) and soft obstacles | `rbGetCalibratedSoftSettingsJson`, `rbWorld_SetSoftSettingsJson`, `rbSoft_AddObstacle` | ✅ |
| Soft damage (yielded and broken beams, plastic work, rigid or soft mode) and the beam view | `soft_damage`, `rbSoft_GetDamage`; `soft_beams`, `rbSoft_GetBeams` (node ordinals, rest lengths, force, broken, yielded, visible) | ✅ |
| Soft skinning, node output, beam breaks and torn-off pieces | `rbSoft_BindSkin`, `RbFrame` | ✅ |
| Destructibles from GLB, pack and cuboid houses; strength scale; layouts | `rbCreateDestructibleAsset*` | ✅ |
| Bonds that settle under self-weight before they can break (the lab's preload) | `preload_destructible_asset`, `rbWorld_PreloadDestructibleAsset` (the lab's procedure, checked against the Metal lab: the masonry arch never sleeps within 600 steps in either, so both give up there) | ✅ |
| The lab's solver settings (120 Hz, 8 iterations, contact and sleep flags) | `DestructibleAssetDesc::solver`, `World::set_solver`, `rbWorld_UseDestructibleAssetSolver` | ✅ |
| Fracture lab `threshold` mode (GPU splitter) | `StrengthModel::Threshold`, `RB_STRENGTH_THRESHOLD`; `body_set_projectile`, `rbBody_SetProjectile`; `solver_for`, `rbWorld_UseDestructibleAssetSolverFor`; `rbWorld_GetDestructibleAssetGround` (D1b) | ✅ |
| Fracture lab `stress` mode (blast_parity) | D5, `StrengthModel::Stress` | 🔜 |
| Deformable terrain (`impact.html`, `terrain.html`): meteor and charge craters, heavy-body strikes, vehicle ruts, baked heights for the host's mesh | `enable_terrain`, `terrain_impact` / `terrain_explode` / `terrain_open`, `terrain_watch_shape`, `set_height_field_soil`; `rbWorld_EnableTerrain`, `rbTerrain_*`, `rbHeightField_SetSoil`, `rbHeightField_GetHeights` | ✅ (an Unreal sample project 🔜) |
| Picking (camera ray) | `rbWorld_CastRays` | ✅ |
| Grab and drag | a kinematic mouse body (`rbBody_SetTargetTransform` each frame) jointed to the grabbed body (spherical, or a weld with `linear_stiffness`); tested through the ABI | ✅ (a dedicated mouse joint later) |
| Scripted crusher, moving slabs | kinematic bodies: `BodyType::Kinematic`, `RB_KINEMATIC_BODY`, set velocity or `rbBody_SetTargetTransform` | ✅ |
| Collider debug view | `World::draw`; `rbWorld_DebugDraw` (C callbacks) | ✅ |
| Fixed rate, several steps per frame, pause, single step, sleep skip | `step`, `rbWorld_Step`, `rbWorld_Poll(coalesce)` | ✅ |
| Reset | destroy and recreate the world, or remove and re-add; snapshot and restore is P6 | ✅ (P6 later) |
| GPU step timings | `profile_ms`, `rbWorld_GetProfileMs` (per-kernel breakdown later) | ✅ |

Descriptors and settings cross the ABI as JSON in the engine's units and
design frame (metres, Y up), the format of `vehicle_presets.json` and the
soft descriptors; spawn transforms and velocities use the world's
coordinates. `tests/demo_parity.rs` covers each row through the ABI.

Next: D5 (`stress`).

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

**P1.5b: developer-feedback fixes (shipped).** Soft particles are kept out of
the moves, with soft node and skin output added in their place. Also shipped:

- soft-vehicle input;
- per-frame vehicle output;
- lossless resync after dropped frames;
- `poll_coalesced`;
- `shape_geometry` and `shape_mesh`.

**P2: C ABI (`engine-ffi`): next, then extended in every phase.** It is
versioned 0.x and unstable until D1 ships. It is generated from the Rust API.
Details are under P2 below.

**D1a: destructibles with the `Bonds` model (shipped).**

- **Assets.** `DestructibleAssetDesc` holds chunks (principal pose, mass,
  inertia, shapes, fragment, glTF node) and bonds (anchors, rest, frame,
  capacities, stiffness).
  - Loaders: `from_glb(bytes, level)`, `from_pack`, `cuboid_house` and
    `from_scene` for any fracture-lab scene. The lab's parked projectiles are
    excluded.
  - Lossless `ConvexHull::unpack` decodes the packed hulls the scenes carry.
  - `create_destructible_asset` validates the descriptor.
- **Placing.** `create_destructible` makes one body per chunk. Every body's
  origin is the instance frame and its solver frame is the chunk's exact
  principal frame. Bonds become breakable bond joints, and world bonds attach
  to a static anchor body.
- **Events.** `BondsBroken` and `Split` arrive in each frame. A split carries
  the parent actor by value, and the children partition it (chunks, bodies,
  anchored, mass).
- **Queries.** `destructible_chunk_body`, `destructible_chunk_local`
  (identity), `destructible_chunk_node`, `destructible_actor_of`,
  `destructible_stats`, `destroy_destructible`.
- **C ABI.** `rbCreateDestructibleAssetFromGlb` / `FromPack`,
  `rbCreateDestructible`, chunk body, node and count queries, and
  `RB_BONDS_BROKEN` and `RB_SPLIT` events with `RbActor` lists.
- **Finding: pack bonds don't break under `Bonds`.** Pack assets load and
  simulate, but their bond capacities are stress-solver limits, which the lab
  treats as unbreakable bond joints. They break only under the `Threshold` and
  `Stress` models. Converting the limits would be a lossy decision, so it is
  not made silently.

**D1b: the `Threshold` model in `World` (shipped).** The fracture lab's
`threshold` mode (stress-rubble `ThresholdWorld`) runs inside `World`, on the
world's own device and `WgpuSim`. It sits behind engine-world's `stress`
feature, which is on by default and adds path dependencies on the stress
oracle's `stress-gpu`, `stress-ref` and `stress-rubble`.

- **Assets.** `from_glb`, `cuboid_house` and `from_pack` also import the same
  source as the lab's stress modes do (`DestructibleAssetDesc::stress`, a
  `StressSource`). That import supplies each chunk's material, volume and
  support, each bond's tensile capacity and area, and the strike geometry.
  Nothing is converted at load. A threshold instance's chunks are the stress
  scene's chunks: the level's fragments in the asset's order (with their glTF
  nodes), then the ground anchors the import adds. A failed import is reported
  when a threshold destructible is placed, never at load.
- **Placing.** One body per bonded cluster is cooked exactly as the lab cooks
  it (stress-rubble `Cooker`), and its body record is written as the lab
  writes it. Every other piece a split can make gets a reserved body slot up
  front. The fracture tables and the contact ingest are built from World's
  colliders, and the plan hints are fixed as the lab fixes them.
- **Stepping.** Each step records `ingest.begin_frame`, World's own
  `step_encode`, then `ingest.substep`, `merge` and `fracture.encode_frame`,
  in one submission. The step's readback carries the fracture counters, chunk
  owners, bond states and body frames. Readbacks are never dropped while a
  threshold destructible exists.
- **Adoption.** When a step split a body, the pieces' reserved slots become
  live `BodyId`s. Every changed body takes the GPU's centre of mass and
  principal axes as its solver frame from the next step on, with mass and
  inertia summed from its chunks. The step then reports `BondsBroken` and
  `Split` (the children partition the parent; one body per actor). Every
  piece's origin stays the instance frame, so `chunk_local` is the identity
  and the render contract holds.
- **Projectiles.** `body_set_projectile(body, Some(Projectile))` adds a body
  with one sphere or box shape to strike prediction and gives its contacts
  the lab's shot loads. Other non-chunk colliders load chunks as static
  geometry does (the lab's ground).
- **Settings.** `solver_for(StrengthModel::Threshold)` gives the lab's
  threshold engine settings: never asleep, zero-gap contact, world inertia,
  4 iterations, 60 Hz. `StressSource::ground_height` gives the lab's ground
  plane.
- **Limits (fail loudly).**
  - The world must use `ContactStorage::Canonical`, the rows the ingest reads.
  - Reserved slots count against `max_bodies`.
  - At most 64 projectiles (the strike table).
  - At most 2^20 chunks or body slots (the fracture's scans).
  - The impact pulse pool is sized from `max_contacts`.
  - A fracture error bit fails the frame.
  - Pieces belong to their destructible: `destroy_body` on one is refused.
- **Cost.** Each step reads back 4 bytes per chunk, 4 per bond and 32 per
  body slot of each threshold destructible. That is about 0.5 MB per step at
  10k chunks.
- **Parity.** `engine-world/tests/threshold.rs` places the masonry arch and
  the cuboid house, fires the lab's shot, and runs the lab's
  `ThresholdWorld` on the same scene. The first break comes in the same frame
  with the same bond count in both. Afterwards the totals follow body-slot
  order (the lab parks 16 shots between its clusters and its spare slots).
  The arch breaks 87 bonds in World against the lab's 92, both ending at 17
  bodies; the house ends at 104 bodies against 95 (50 bonds against 21). Each
  new piece starts where its parent was. `engine-ffi/tests/threshold.rs` runs both
  scenes through the C ABI in Unreal coordinates.
- **Shaders.** The kernels run as WGSL on World's device, as in the browser.
  A device opened with `PASSTHROUGH_SHADERS` on Metal runs the Metal
  libraries instead.
- **Not yet.** Snapshots of threshold state; damage descriptors (D2);
  fracture-command replay (D3).

**D1 (original list):**

- `DestructibleAssetDesc` with loaders: GLB, packs, and the procedural
  cuboid, row and grid builders.
- `create_destructible`.
- Two strength models: `Threshold`, built on `GpuFracture` /
  `FractureTables` with pre-allocated slots taken from `WorldCapacity`, and
  `Bonds`, using breakable bond joints.
- The render contract: piece origin = instance frame, a fixed per-chunk
  transform, and a chunk → glTF node map.
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

**V1 (shipped, first cut).** These calls compose rigid, soft and
destructible parts.

**Calls.**

- `create_assembly(&AssemblyDef)` places parts and connections; it suits
  mechs, signs, doors and pre-rigged structures.
- `create_vehicle(&VehicleDef)` builds a vehicle from:
  - a core, `VehicleModel::Rigid(VehicleDesc)` or
    `VehicleModel::Soft(SoftVehicleDesc, SpawnOptions)`;
  - extra parts mounted onto `Chassis`, `VehiclePart(i)`, `SoftNode(i)`, any
    named part, or a destructible's `Chunk`.
- `assembly_part` and `assembly_connection` look up what was built.

**Parts.** A part is `Rigid` (a body with shapes) or `Destructible` (a D1
instance).

**Connections.**

- A connection is a breakable `Mount` or any typed `JointDef`.
- Its frame is given once, in assembly or design space. Each body's joint
  frame is derived from where that body starts.
- A connection that breaks raises `ConnectionBroke { assembly, connection,
  joint, force, torque }`, which is also `RB_CONNECTION_BROKE` in the ABI.

**Next.**

- Soft *parts* inside a rigid vehicle.
- Folding the presets and the JBeam / Rigs of Rods imports into `VehicleDef`
  builders.
- Assemblies in the C ABI.

**V1 (original plan): assemblies and composable vehicles.** `AssemblyDef` and `VehicleDef`
with rigid, soft and (after D1) destructible parts, and core, mount, bond and
joint connections. The presets become builders. `add_soft_vehicle` is folded
into `add_vehicle`.

**P2 detail: C ABI (`engine-ffi`).** A cbindgen header mirrors the Rust API
with Box3D naming (`rb` prefix). It has:

- `rbDefault*Def()` functions;
- `uint64_t` handles with `rbStore`/`rbLoad` and `RB_IS_NULL`;
- event arrays valid until the next poll;
- error results plus `rbGetLastError()`;
- `catch_unwind` at the boundary;
- flat array outputs (handles plus transforms) next to the object calls, for
  Unreal, Unity and ECS hosts;
- a **coordinate option** (Y-up right-handed metres, or Unreal's Z-up
  left-handed centimetres), converted exactly at the boundary, with these
  rules:
  - **Polar vectors** (position, velocity, force, impulse, points, normals)
    are mapped and scaled.
  - **Pseudovectors** (angular velocity, torque, angular impulse) pick up an
    extra sign flip under the left-handed reflection.
  - **Quaternions** transform as rotations; joint angle limits and motor
    directions flip sign.
  - **Units:** lengths and speeds ×100; force and impulse carry the cm factor;
    inertia ×10⁴; density ×10⁻⁶. Mass and angles are unchanged.
  - **Meshes:** exported meshes flip their triangle winding.
  - Each field's class is listed in the header, and a round-trip test covers
    every field.

Unreal is the first client.

**P3: Box3D body features (new Slang, opt-in, defaults lossless).**

- ✅ Kinematic bodies and `set_target_transform`, which push dynamic bodies
  and carry them by friction (shipped 2026-10-24). They carry
  `BODY_KINEMATIC` without `BODY_DYNAMIC`, with zero inverse mass (the CCD
  branch's model, `engine/docs/ccd.md`): prediction places them at exactly
  `x0 + v dt` and the solver never moves them. Contacts, bounds, the
  broadphase, wakes, CCD and characters treat a moving one as a mover with
  infinite mass, and a stopped one costs what a static costs. The target command computes the velocity on
  the GPU from the pose at the step start, so animation tracking never
  drifts. Without the flag every kernel path is unchanged.
- A full gravity vector, keeping the Y-only path bit-identical.
- ✅ `gravity_scale` and linear and angular damping (shipped 2026-10-24): a
  per-step command at the start of the step, `v' = v / (1 + h c) + h (s - 1) g`,
  which gives Box3D's inertial target exactly; only awake bodies that use
  them get one, and they neither wake bodies nor keep an asleep world
  stepping. Per-body `sleep_threshold` is still to do.
- ✅ `motion_locks` (shipped 2026-10-24): body-flag bits 9..14 (above
  `BODY_CCD`) read by two passes scheduled only while some body has locks:
  `lock_predict` after prediction and `lock_project` after the solve each
  remove the locked components (world axes) of the body's displacement and
  rotation since the step start, so the inertial target respects the locks
  and the locked axes are exact at the end of every step. The hot kernels
  (`predictOne`, `primalApply`) are untouched: an earlier in-kernel version
  shifted Metal and wgpu fast-math codegen for every body and failed the
  Metal parity tests. C: `RB_LOCK_*` bits, permuted for Unreal axes.
- ✅ `is_bullet` (2026-10-24): World sets `BODY_CCD`, #52's continuous
  collision (uncapped reach, the fast list, a time-of-impact contact and
  start pose); C `RbBodyDef::is_bullet`. Sensors take no time of impact and
  no post-solve sweep check, so bullets pass through them.
- ✅ Characters (2026-10-24): #52's capsule movers (`char_move`: walk,
  step up, stick to the floor, slopes, jump, moving support, push weight)
  through `World::create_character(CharacterDef)`, `character_set_input`
  (velocity, jump once), `character_teleport`, `destroy_character` (slot
  reused after quarantine); every frame reports `Frame::characters` and
  `character_events` (landed, support lost, crushed) from one readback copy
  of the records, no blocking read. The inner kinematic capsule is a World
  body of its own, left out of body moves. C: `rbWorld_CreateCharacter`,
  `rbCharacter_*`, `RbFrame::characters` / `character_events`.
- `allow_fast_rotation`.

**P4: contact, hit and sensor events.** Contact begin/end and hit events
shipped 2026-10-24 (World `Frame::contact_events`, C `RbFrame::contact_begin`,
`contact_end`, `hits`; tested in persistent and canonical storage and through
the ABI in Unreal units). Sensors shipped the same day: `is_sensor` shapes
have no response (the events pass zeroes their manifold and solver pair, so
the CSR graph and islands skip them) and, with `enable_sensor_events`, report
`SensorBegin` / `SensorEnd` (World `Frame::sensor_events`, C
`RbFrame::sensor_begin` / `sensor_end`). Sensors do not detect sensors.
Compact contact storage refuses sensors (its solver pairs are packed before
the pass).

The design (2026-10-24):

- **Flags.** Per shape, in the free bits 12..15 of `Collider.material`:
  contact events, hit events and sensor. Material indices stay below 64 and
  the filter group keeps bits 16..31, so the 64 B collider record does not
  grow. `materialIndex()` masks the low 12 bits; no existing record sets the
  new bits.
- **GPU pass.** An opt-in `contact_events` pass right after narrowphase,
  added to the step graph only when a world has event-enabled shapes, so
  other hosts and worlds replay the same commands. One thread per manifold:
  if either collider is flagged and its deepest contact is touching
  (`c0N <= 0`), it appends `{colliderA, colliderB, approach speed, point,
  normal}` to a fixed-capacity region after the large-static list in the
  scratch buffer (binding 21, `CONTACT_EVENT_CAP` records). The approach
  speed uses the bodies' step-start velocities at the deepest point, before
  the solve changes them. Overflow increments a counter that the frame
  reports (`lost_events`); it is never silent.
- **Host diff.** `World` reads the region with each frame's readback and
  diffs the sorted pair set against the last one: `ContactBegin`
  (shapes, point, normal, approach speed), `ContactEnd`, and `Hit` (a begin
  whose approach speed is at least `hit_event_threshold`, for shapes with
  hit events). A pair whose two bodies are both asleep stays in the set
  (no end on sleep, as Box3D), and destroying a touching shape ends its
  pairs. Exact for every contact storage mode, because it never depends on
  how contacts are matched across steps. Frames that carry contact events
  are never dropped.
- **Sensors (second step).** Sensor shapes keep their overlaps out of the
  solver: narrowphase marks their contacts inactive, the pass reports them
  as overlaps, and the host emits `SensorBegin` and `SensorEnd`.
- `explode` (D2's radial damage reuses it).

**P5: queries and mover.** Batched asynchronous ray, shape-cast and overlap
queries with `QueryFilter`, moved off the vehicle ray pool. Then capsule
`cast_mover` and `collide_mover`.

- *Shipped (2026-10-24):* `World::overlap_aabb(min, max, QueryFilter)`
  (C `rbWorld_OverlapAABB`), Box3D's bounds overlap: a pass over colliders
  after the step, with bounds refreshed to its final poses (as ray queries
  do; lossless, tested), up to 32 queries and 8192 results per step,
  sensors not reported; results in `Frame::overlap_results` /
  `RbFrame::overlaps`. And `World::explode(ExplosionDef)` (C
  `rbWorld_Explode`), Box3D's radial impulse: an early pass in steps that
  have one, impulse per area times each collider's cross-section, full
  inside the radius and fading over the falloff, waking what it reaches.

- *Shipped (2026-10-24):* `World::cast(CastDef)` (C `rbWorld_Cast`): ray,
  sphere and upright-capsule casts, closest hit, on #52's in-step
  `query_batch` (grid walk, exact shapes, category mask, an ignored body).
  It needs no vehicle pool. It runs before the step moves anything, so it
  sees the poses the step starts from. Up to 256 per step; results in
  `Frame::cast_results` / `RbFrame::casts`, and a frame with casts is never
  dropped. Projectiles: `World::spawn_projectile(ProjectileDef)` (C
  `rbWorld_SpawnProjectile`), #52's piercing ballistic casts. Energy pays
  for each shape's thickness, and hits push dynamic bodies. Stopped
  projectiles' slots are reused, and the pool is released once all have
  stopped, so a quiet world can sleep again. Results arrive as
  `Frame::projectiles` and `projectile_events` (passed with the thickness
  crossed, or stopped).
- *Shipped first cut:* `World::cast_rays` (C: `rbWorld_CastRays`) queues
  closest-hit rays for the next step. They trace on the GPU against that
  step's final poses (they share the vehicle ray pool, whose ray count only
  `raycast_batch` reads, so bodies stay byte-identical; tested). Hits arrive
  in `Frame::ray_results` / `RbFrame::rays` without a stall, and a frame
  carrying queries is never dropped. An asleep world runs one step to answer.

**P6: tooling.**

- Recording and replay of inputs plus a per-frame GPU state hash, validated on
  replay.
- A `DebugDraw` callback struct fed by one readback of colliders and poses.
  *Shipped first cut:* `World::draw(&mut impl DebugDraw, DebugDrawOptions)`
  draws every shape at its body's last polled pose, coloured by state or by
  `SurfaceMaterial::custom_color`, with no extra readback.
- `shift_origin`.

**P8: rendering export.**

- An HLSL target in `engine-slang-build`, so `soft_skin` and `pose_pack` can
  run as Unreal `.usf`.
- Render assets: fragment meshes with UVs, material ids and an interior-face
  flag, and the soft-body visual mesh.
- Native-device interop through `wgpu-hal` (zero-copy, DX12).
- Static mesh and height-field shapes, if the kernels gain them.

### Developer feedback (2026-10-23)

A prospective integrator raised nine points. All nine were accepted, some
with refinements:

1. **C ABI is too late.** It moved up to follow P1.5b and is extended in
   every phase.
2. **Soft nodes flood the moves.** Confirmed in the code: each particle is a
   body slot. Fixed in P1.5b.
3. **Wheels can't be drawn from the frame.** Fixed in P1.5b with per-frame
   vehicle output.
4. **`Drop` loses bodies that settle.** Fixed in P1.5b with a lossless
   resync.
5. **Several steps per display frame.** Fixed in P1.5b with
   `poll_coalesced`. Per-step `poll` stays, because audio and netcode need
   step precision.
6. **Destructible render contract.** Taken into D1: piece origin = instance
   frame, plus the chunk → node map.
7. **The `Bonds` model and non-GLB sources are unscheduled.** Both are in
   D1, with descriptor-first assets.
8. **`shape_geometry`.** Shipped in P1.5b.
9. **The Z-up spec needs detail.** Added to P2, including the pseudovector
   rule the review missed.

Separately, the review asked for one composable `add_vehicle`. That became
V1, assemblies.

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

//! Rubble's public world API: rigid bodies, shapes, joints, vehicles and soft
//! bodies simulated on the GPU. The object model follows Box3D (definitions
//! with defaults, generational ids, shapes attached to bodies, body origin
//! separate from the centre of mass, events grouped per step); where a
//! GPU-resident engine must differ, it does so explicitly:
//!
//! - **Asynchronous results.** [`World::step`] never waits on a readback;
//!   [`World::poll`] returns each finished step's [`Frame`]. Getters such as
//!   [`World::body_transform`] return the last polled frame; nothing hides a
//!   GPU readback. [`World::step_sync`] gives Box3D-style immediacy.
//! - **Commands, not writes.** Velocities, forces, impulses and teleports are
//!   queued and applied on the GPU at the start of the next step, in
//!   submission order per body.
//! - **Loud limits.** Capacities are fixed at creation and fail with clear
//!   errors; unsupported definition fields return [`Unsupported`].
//! - **Lossless.** A world computes exactly what the raw [`WgpuSim`] computes.
//! - **Deterministic** on the same build and GPU.
//!
//! See `engine/docs/public-api.md` and `engine/docs/api-review.md`.

mod assembly;
mod contacts;
mod debug_draw;
mod defs;
mod destruction;
mod error;
mod events;
mod handles;
mod joints;
mod math;
mod motion;
mod queries;
mod readback;
mod rig;
mod terrain;
mod threshold;

pub use assembly::{
    AssemblyDef, AssemblyId, ConnectionDef, ConnectionKind, PartDef, PartHandle, PartModel,
    PartRef, VehicleBuild, VehicleDef, VehicleModel,
};
pub use debug_draw::{
    DebugColor, DebugDraw, DebugDrawOptions, COLOR_ASLEEP, COLOR_AWAKE, COLOR_STATIC,
};
pub use defs::{BodyDef, BodyType, Geometry, MassData, MotionLocks, ShapeDef, SurfaceMaterial};
pub use destruction::{
    lab_solver, ActorInfo, Anchors, BondDesc, ChunkDesc, DestructibleAssetDesc,
    DestructibleAssetId, DestructibleDef, DestructibleId, DestructibleStats, Preload,
    StrengthModel,
};
pub use error::{StaleHandle, Unsupported};
pub use events::{
    coalesce, BodyEvents, BodyMoveEvent, CoalescedFrame, ContactBeginEvent, ContactEndEvent,
    ContactEvents, DestructionEvent, Frame, RayBatchId, RayBatchResult, SensorEvent, SensorEvents,
    SkinId, SkinOutput, SoftOutput, VehicleOutput,
};
pub use handles::{BodyId, JointId, ShapeId};
pub use joints::{
    DistanceJointDef, FilterJointDef, JointBase, JointDef, Plastic, PrismaticJointDef,
    RevoluteJointDef, SphericalJointDef, Spring, WeldJointDef,
};
pub use math::Transform;
pub use motion::{
    CastDef, CastHit, CastId, CastResult, CastShape, CharacterDef, CharacterDesc, CharacterEvent,
    CharacterEventKind, CharacterId, CharacterOutput, GroundState, ProjectileDef, ProjectileEvent,
    ProjectileEventKind, ProjectileId, ProjectileOutput,
};
pub use queries::{ExplosionDef, OverlapResult, QueryFilter, QueryId};
pub use readback::Backpressure;
pub use rig::{SoftBeam, VehicleDamage};
pub use terrain::{StrikeDef, TerrainDef};
pub use threshold::{Projectile, StressSource};
pub use engine_terrain::{
    CraterPrediction, CraterTarget, ExplosionSource, ImpactSource, InitialStress, RegionBoundary,
    RegionPlan, SandMaterial, TerrainConfig, TerrainError, YieldSurface,
};
pub use engine_webgpu::TerrainStepReport;

pub use engine_schema::convex::{CollisionFilter as Filter, ConvexHull};
pub use engine_schema::heightfield::{HeightField, HeightFieldError};
pub use engine_schema::soil::{Soil, SoilError};
pub use engine_schema::{
    AvbdParams, Dof, JointDesc, JointFrame, Motion, StepCounters, WorldCapacity,
};
pub use engine_softbody::{
    SoftCapacity, SoftDamage, SoftHandle, SoftSettings, SoftVehicleDesc, SpawnOptions,
};
pub use engine_vehicle::{
    FunctionalState, GearRequest, MountLoad, VehicleCapacity, VehicleDesc, VehicleHandle,
    VehicleInput, VehicleState, WheelState,
};
pub use engine_webgpu::{ContactStorage, EngineFeatures, WgpuSim};

use engine_schema::convex::HullArena;
use engine_schema::{BodyState, Collider, GpuCommand, ShapeType};
use engine_webgpu::SimOptions;
use error::unsupported;
use glam::{Quat, Vec3};
use math::SolverFrame;
use readback::{Pending, ReadbackRing};

/// The GPU a world runs on.
#[allow(clippy::large_enum_variant)] // Built once per world.
pub enum GpuContext {
    /// A high-performance adapter the world opens itself.
    Default,
    /// The host's device (a renderer's), so its passes can read the world's
    /// buffers in place.
    Shared {
        info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
    },
}

/// A world (Box3D `b3WorldDef`, plus what a GPU world fixes at creation).
#[derive(Clone, Debug)]
pub struct WorldDef {
    /// m/s². Only gravity along Y is supported for now (planned P3).
    pub gravity: [f32; 3],
    /// Bodies may fall asleep (Box3D `enableSleep`).
    pub enable_sleep: bool,
    pub user_data: u64,
    /// Rubble: fixed capacities; overflow fails the step.
    pub capacity: WorldCapacity,
    pub contacts: ContactStorage,
    pub vehicles: VehicleCapacity,
    /// A soft-body arena (needs [`ContactStorage::Canonical`]).
    pub soft: Option<SoftCapacity>,
    /// Step size, iterations, sleep thresholds and solver constants
    /// (`gravity_y` comes from `gravity`).
    pub solver: AvbdParams,
    pub features: EngineFeatures,
    /// Apply the `ENGINE_*` benchmark variables over `features` and `contacts`.
    pub env_overrides: bool,
    /// Steps whose results can be in flight at once.
    pub readback_frames: u32,
    pub backpressure: Backpressure,
    /// Contact begins at or above this approach speed (m/s) on shapes with
    /// `enable_hit_events` are also reported as hits (Box3D
    /// `hitEventThreshold`).
    pub hit_event_threshold: f32,
}

impl Default for WorldDef {
    fn default() -> Self {
        Self {
            gravity: [0.0, AvbdParams::default().gravity_y, 0.0],
            enable_sleep: true,
            user_data: 0,
            capacity: WorldCapacity::small(),
            contacts: ContactStorage::Persistent,
            vehicles: VehicleCapacity::default(),
            soft: None,
            solver: AvbdParams::default(),
            features: EngineFeatures::default(),
            env_overrides: false,
            readback_frames: 3,
            hit_event_threshold: 1.0,
            backpressure: Backpressure::Block,
        }
    }
}

/// A body's state read straight from the GPU ([`World::read_body_state`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyStateRead {
    pub transform: Transform,
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub awake: bool,
}

/// A shape's triangle mesh in its body's frame ([`World::shape_mesh`]).
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeMesh {
    pub vertices: Vec<[f32; 3]>,
    /// Counter-clockwise seen from outside.
    pub triangles: Vec<[u32; 3]>,
}

/// A ray hit ([`World::cast_ray_blocking`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayResult {
    /// Null for a vehicle's or soft body's own colliders.
    pub shape: ShapeId,
    pub body: BodyId,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub distance: f32,
}

/// A ray for [`World::cast_ray_blocking`]: it runs the full length of
/// `translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayInput {
    pub origin: [f32; 3],
    pub translation: [f32; 3],
    pub ignore_body: Option<BodyId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    User,
    /// A vehicle's chassis, wheel or part, or a soft body's core.
    Vehicle,
    /// A soft-body particle: reported through `Frame::soft`, not as moves.
    SoftNode,
    /// A threshold destructible's piece (or a slot reserved for one): its shapes and
    /// lifetime belong to the destructible.
    Piece,
    /// A character's inner capsule: reported through `Frame::characters`.
    Character,
}

/// What an extra readback copy holds.
#[derive(Clone, Debug)]
enum Extra {
    /// Every body's published pose: the frame after a dropped one diffs it.
    Resync,
    VehicleRecord(VehicleHandle),
    VehicleWheels(VehicleHandle),
    SoftNodes(SoftHandle),
    SkinPositions(SkinId),
    SkinNormals(SkinId),
    /// The ray-query pool after this step's traces: batches by `(id, first,
    /// count)` into it.
    RayHits(Vec<(RayBatchId, u32, u32)>),
    /// This step's touching pairs of event-enabled shapes.
    Contacts,
    /// The overlap query region, for these queries.
    Overlaps(Vec<QueryId>),
    /// A threshold destructible's state after this step.
    #[cfg(feature = "stress")]
    Fracture(threshold::FractureRead),
    /// Every character record.
    Characters,
    /// The scene-query records, for these casts.
    Casts(Vec<CastId>),
    /// Every projectile record.
    Projectiles,
    /// The terrain counters after this step's MPM substeps.
    TerrainCounters,
    /// A baked heightfield's samples, for the host mirror.
    TerrainHeights(ShapeId),
    /// The terrain look-ahead's pick after this step.
    TerrainWatch,
}

#[derive(Clone, Debug)]
struct BodySlot {
    generation: u32,
    alive: bool,
    owner: Owner,
    body_type: BodyType,
    user_data: u64,
    name: Option<String>,
    shapes: Vec<u32>,
    joints: Vec<u32>,
    removed_at: Option<u64>,
    /// Current mass properties (None: static, or dynamic without mass yet).
    mass: Option<MassData>,
    /// Principal moments the solver uses (with the frame's axes).
    principal: Vec3,
    /// Solver frames over time, `(first step, frame)`, so frames still in
    /// flight convert with the frame they were simulated in.
    frames: Vec<(u64, SolverFrame)>,
    /// The step the body was last fully uploaded before: while
    /// `written_step == world.step`, the GPU state is exactly that upload.
    written_step: u64,
    origin: Transform,
    velocity: ([f32; 3], [f32; 3]),
    asleep: bool,
    /// Latest frame-reported state.
    last_transform: Option<Transform>,
    last_asleep: bool,
    /// The solver pose last delivered, for the resync diff.
    last_solver: Option<Transform>,
    /// Damping (linear, angular) and gravity scale, sent each step as a
    /// command while they differ from (0, 0, 1).
    damping: (f32, f32),
    gravity_scale: f32,
    locks: u32,
    bullet: bool,
}

impl BodySlot {
    fn tuned(&self) -> bool {
        self.damping != (0.0, 0.0) || self.gravity_scale != 1.0
    }
    fn frame(&self) -> SolverFrame {
        self.frames.last().map_or(SolverFrame::IDENTITY, |f| f.1)
    }

    fn frame_at(&self, step: u64) -> SolverFrame {
        self.frames
            .iter()
            .rev()
            .find(|f| f.0 <= step)
            .map_or(SolverFrame::IDENTITY, |f| f.1)
    }
}

#[derive(Clone, Debug)]
struct ShapeSlot {
    generation: u32,
    alive: bool,
    body: u32,
    collider: u32,
    geometry: Geometry,
    def: ShapeDef,
}

#[derive(Clone, Copy, Debug)]
struct JointSlot {
    generation: u32,
    alive: bool,
    bodies: [u32; 2],
    user_data: u64,
    /// Both frames in the bodies' origin frames, to re-express them when a
    /// body's centre of mass moves.
    origin_frames: [JointFrame; 2],
}

/// Steps a released body or collider slot waits before reuse: neither contact
/// history half may still refer to its old colliders.
const QUARANTINE_STEPS: u64 = 2;

fn graveyard(slot: u32) -> [f32; 3] {
    [
        (slot % 1024) as f32 * 8.0,
        -10_000.0,
        (slot / 1024) as f32 * 8.0,
    ]
}

/// A collider that collides with nothing: a tiny filtered-out sphere.
fn retired_collider(body: u32) -> Collider {
    let mut c = Collider {
        body,
        shape: ShapeType::Sphere as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [1e-3, 0.0, 0.0, 1e-3],
    };
    Filter {
        categories: 0,
        mask: 1,
        group: 0,
    }
    .apply(&mut c);
    c
}

/// A simulated world.
pub struct World {
    sim: WgpuSim,
    hulls: HullArena,
    bodies: Vec<BodySlot>,
    free_bodies: Vec<u32>,
    shapes: Vec<ShapeSlot>,
    free_shapes: Vec<u32>,
    /// Retired colliders: `(collider, released at step)`.
    free_colliders: Vec<(u32, u64)>,
    /// Arena blocks of removed heightfields: `(base, released at step)`. Freed
    /// past quarantine, when no step in flight can still read them.
    retired_height_fields: Vec<(u32, u64)>,
    /// Collider index -> shape slot (u32::MAX: not a user shape).
    collider_shape: Vec<u32>,
    joints: Vec<JointSlot>,
    /// Interned surface materials (GPU slot = index).
    materials: Vec<[u32; 3]>,
    readback: ReadbackRing<Extra>,
    /// A frame was dropped: the next delivered one carries a full pose diff.
    resync: bool,
    vehicles: Vec<VehicleHandle>,
    /// Soft bodies whose nodes each frame reports.
    soft_outputs: Vec<SoftHandle>,
    skins: Vec<(SoftHandle, engine_webgpu::SoftSkinner)>,
    assets: Vec<std::sync::Arc<DestructibleAssetDesc>>,
    destructibles: Vec<destruction::DestructibleSlot>,
    /// Bond joint index -> (destructible slot, bond index).
    bond_owner: std::collections::HashMap<u32, (u32, u32)>,
    assemblies: Vec<assembly::AssemblySlot>,
    /// Connection joint index -> (assembly slot, connection index).
    connection_owner: std::collections::HashMap<u32, (u32, u32)>,
    /// Ray queries for the next step, and the batches they belong to.
    pending_rays: Vec<engine_vehicle::Ray>,
    pending_batches: Vec<(RayBatchId, u32, u32)>,
    next_ray_batch: u64,
    backpressure: Backpressure,
    user_data: u64,
    step: u64,
    newest_ran: Option<u64>,
    ready_frames: Vec<Frame>,
    dirty: bool,
    last_input_step: Option<u64>,
    last_counters: Option<StepCounters>,
    /// The adapter this world runs on (for helper worlds on its device).
    adapter: wgpu::AdapterInfo,
    contacts: contacts::ContactTracker,
    queries: queries::QueryQueue,
    hit_event_threshold: f32,
    contact_storage: ContactStorage,
    /// Threshold destructibles (`StrengthModel::Threshold`).
    fracture: threshold::Fracture,
    motion: motion::MotionState,
    /// Deformable terrain, once enabled.
    terrain: Option<terrain::TerrainHost>,
}

impl World {
    pub fn new(gpu: GpuContext, def: &WorldDef) -> anyhow::Result<Self> {
        if def.gravity[0] != 0.0 || def.gravity[2] != 0.0 {
            return unsupported("gravity off the Y axis (planned P3)");
        }
        let (contacts, features) = if def.env_overrides {
            let env = SimOptions::from_env();
            (env.contacts, env.features)
        } else {
            (def.contacts, def.features.clone())
        };
        let options = SimOptions {
            contacts,
            vehicles: def.vehicles,
            soft: def.soft,
            features,
            pose_output: true,
        };
        let (info, device, queue) = match gpu {
            GpuContext::Default => engine_webgpu::default_device()?,
            GpuContext::Shared {
                info,
                device,
                queue,
            } => (info, device, queue),
        };
        let adapter = info.clone();
        let mut sim =
            WgpuSim::with_device_options(def.capacity.clone(), options, info, device, queue)?;
        let mut params = def.solver;
        params.gravity_y = def.gravity[1];
        if !def.enable_sleep {
            params.sleep_time = f32::MAX;
        }
        sim.set_params(params);
        let readback = ReadbackRing::new(&sim, def.readback_frames);
        Ok(Self {
            sim,
            hulls: HullArena::new(),
            bodies: Vec::new(),
            free_bodies: Vec::new(),
            shapes: Vec::new(),
            free_shapes: Vec::new(),
            free_colliders: Vec::new(),
            retired_height_fields: Vec::new(),
            collider_shape: Vec::new(),
            joints: Vec::new(),
            // Slot 0 is the sim's default material (friction 0.5).
            materials: vec![SurfaceMaterial::default().key()],
            readback,
            backpressure: def.backpressure,
            user_data: def.user_data,
            step: 0,
            newest_ran: None,
            ready_frames: Vec::new(),
            dirty: false,
            last_input_step: None,
            last_counters: None,
            adapter,
            contacts: Default::default(),
            queries: Default::default(),
            hit_event_threshold: def.hit_event_threshold,
            contact_storage: contacts,
            motion: Default::default(),
            resync: false,
            vehicles: Vec::new(),
            soft_outputs: Vec::new(),
            skins: Vec::new(),
            assets: Vec::new(),
            destructibles: Vec::new(),
            bond_owner: std::collections::HashMap::new(),
            assemblies: Vec::new(),
            connection_owner: std::collections::HashMap::new(),
            pending_rays: Vec::new(),
            pending_batches: Vec::new(),
            next_ray_batch: 0,
            fracture: Default::default(),
            terrain: None,
        })
    }

    /// The simulation underneath, for reads this API does not cover yet.
    pub fn sim(&self) -> &WgpuSim {
        &self.sim
    }

    pub fn device(&self) -> &wgpu::Device {
        self.sim.device()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.sim.queue()
    }

    pub fn user_data(&self) -> u64 {
        self.user_data
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// The last polled step's counters.
    pub fn counters(&self) -> Option<StepCounters> {
        self.last_counters
    }

    /// GPU time of the last step the host waited for (ms).
    pub fn profile_ms(&self) -> f64 {
        self.sim.timings_ms()
    }

    /// Replace the solver settings (step size, iterations, contact and
    /// sleep tuning, flags), gravity included: for example a destructible
    /// asset's `solver`, for the fracture lab's behaviour. Hosts step at the
    /// new `dt`.
    pub fn set_solver(&mut self, solver: AvbdParams) {
        self.dirty = true;
        self.sim.set_params(solver);
    }

    pub fn solver(&self) -> AvbdParams {
        self.sim.params()
    }

    pub fn set_gravity(&mut self, gravity: [f32; 3]) -> anyhow::Result<()> {
        if gravity[0] != 0.0 || gravity[2] != 0.0 {
            return unsupported("gravity off the Y axis (planned P3)");
        }
        let mut p = self.sim.params();
        p.gravity_y = gravity[1];
        self.sim.set_params(p);
        self.dirty = true;
        Ok(())
    }

    // ----- bodies ----------------------------------------------------------

    /// A body without shapes (Box3D `b3CreateBody`); attach shapes with
    /// `create_*_shape`. A dynamic body without shapes has unit mass and
    /// inertia until shapes or `body_set_mass_data` give it more.
    pub fn create_body(&mut self, def: &BodyDef) -> anyhow::Result<BodyId> {
        Ok(self.create_bodies(std::slice::from_ref(def))?[0])
    }

    /// Many bodies in one upload (Jolt's batch add): prefer it at 10k+ bodies.
    pub fn create_bodies(&mut self, defs: &[BodyDef]) -> anyhow::Result<Vec<BodyId>> {
        for d in defs {
            d.validate()?;
        }
        self.dirty = true;
        let mut ids = vec![BodyId::NULL; defs.len()];
        let mut fresh: Vec<(usize, BodyState)> = Vec::new();
        for (k, d) in defs.iter().enumerate() {
            let rotation = Quat::from_array(d.rotation).normalize().to_array();
            let asleep = !d.is_awake && d.body_type != BodyType::Static;
            let slot = BodySlot {
                generation: 1,
                alive: true,
                owner: Owner::User,
                body_type: d.body_type,
                user_data: d.user_data,
                name: d.name.clone(),
                shapes: Vec::new(),
                joints: Vec::new(),
                removed_at: None,
                mass: None,
                principal: Vec3::ONE,
                frames: Vec::new(),
                written_step: self.step,
                origin: Transform::new(d.position, rotation),
                velocity: (d.linear_velocity, d.angular_velocity),
                asleep,
                last_transform: None,
                last_asleep: asleep,
                last_solver: None,
                damping: (d.linear_damping, d.angular_damping),
                gravity_scale: d.gravity_scale,
                locks: d.motion_locks.bits(),
                bullet: d.is_bullet && d.body_type == BodyType::Dynamic,
            };
            let state = body_state(&slot);
            match self.reusable_body_slot() {
                Some(i) => {
                    let generation = self.bodies[i as usize].generation;
                    self.bodies[i as usize] = BodySlot { generation, ..slot };
                    self.sim.write_bodies(i, &[state]);
                    self.sim.forget_pose(i);
                    ids[k] = BodyId::new(i, generation);
                }
                None => {
                    // `bodies` mirrors the sim's slots, so this push and the
                    // batch upload below land on the same index.
                    fresh.push((k, state));
                    self.bodies.push(slot);
                }
            }
        }
        if !fresh.is_empty() {
            let cap = self.sim.capacity().max_bodies;
            if self.sim.body_count() + fresh.len() as u32 > cap {
                self.bodies.truncate(self.sim.body_count() as usize);
                anyhow::bail!("WorldCapacity::max_bodies ({cap}) reached");
            }
            let states: Vec<BodyState> = fresh.iter().map(|f| f.1).collect();
            let range = self.sim.add_bodies(&states);
            for ((k, _), index) in fresh.iter().zip(range) {
                ids[*k] = BodyId::new(index, 1);
            }
        }
        if defs
            .iter()
            .any(|d| d.motion_locks.bits() != 0 && d.body_type == BodyType::Dynamic)
        {
            self.sim.set_motion_locks(true);
        }
        Ok(ids)
    }

    /// Destroy a body, its shapes and its joints (Box3D `b3DestroyBody`).
    /// Every copy of its handle becomes invalid at once; its slot is reused
    /// after two steps.
    pub fn destroy_body(&mut self, id: BodyId) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        if self.bodies[i].owner == Owner::Piece {
            anyhow::bail!(
                "body {} is a threshold destructible's piece; destroy the destructible instead",
                id.index()
            );
        }
        if self.bodies[i].owner != Owner::User {
            anyhow::bail!(
                "body {} belongs to a vehicle or soft body; remove that instead",
                id.index()
            );
        }
        self.dirty = true;
        for j in std::mem::take(&mut self.bodies[i].joints) {
            self.release_joint(j);
        }
        for s in std::mem::take(&mut self.bodies[i].shapes) {
            self.release_shape(s);
        }
        let index = id.index();
        self.fracture.body_destroyed(index);
        self.sim.cancel_body_commands(index);
        let p = graveyard(index);
        let mut parked = BodyState::static_plane_support();
        parked.position = p;
        parked.initial_pos = p;
        self.sim.write_bodies(index, &[parked]);
        let slot = &mut self.bodies[i];
        slot.alive = false;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        slot.removed_at = Some(self.step);
        self.free_bodies.push(index);
        Ok(())
    }

    pub fn body_is_valid(&self, id: BodyId) -> bool {
        self.live_body(id).is_ok()
    }

    pub fn body_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.alive).count()
    }

    pub fn body_type(&self, id: BodyId) -> anyhow::Result<BodyType> {
        Ok(self.bodies[self.live_body(id)?].body_type)
    }

    pub fn body_user_data(&self, id: BodyId) -> anyhow::Result<u64> {
        Ok(self.bodies[self.live_body(id)?].user_data)
    }

    pub fn body_name(&self, id: BodyId) -> anyhow::Result<Option<&str>> {
        Ok(self.bodies[self.live_body(id)?].name.as_deref())
    }

    pub fn body_shapes(&self, id: BodyId) -> anyhow::Result<Vec<ShapeId>> {
        let b = &self.bodies[self.live_body(id)?];
        Ok(b.shapes
            .iter()
            .map(|&s| ShapeId::new(s, self.shapes[s as usize].generation))
            .collect())
    }

    /// The body origin's pose as of the last polled frame (`None` before the
    /// body's first frame). Frame-latent by design: no hidden readback.
    pub fn body_transform(&self, id: BodyId) -> anyhow::Result<Option<Transform>> {
        Ok(self.bodies[self.live_body(id)?].last_transform)
    }

    /// Whether the body was awake as of the last polled frame.
    pub fn body_is_awake(&self, id: BodyId) -> anyhow::Result<bool> {
        Ok(!self.bodies[self.live_body(id)?].last_asleep)
    }

    /// Centre of mass in the body frame.
    pub fn body_local_center(&self, id: BodyId) -> anyhow::Result<[f32; 3]> {
        Ok(self.bodies[self.live_body(id)?].frame().center.to_array())
    }

    pub fn body_mass_data(&self, id: BodyId) -> anyhow::Result<Option<MassData>> {
        Ok(self.bodies[self.live_body(id)?].mass)
    }

    /// The body's state read straight from the GPU (blocking; for tools and
    /// tests, not per-frame use).
    pub fn read_body_state(&self, id: BodyId) -> anyhow::Result<BodyStateRead> {
        let i = self.live_body(id)?;
        let b = self.sim.read_body(id.index());
        let solver = Transform::new(b.position, b.orientation);
        Ok(BodyStateRead {
            transform: self.bodies[i].frame().to_origin(solver),
            linear_velocity: b.linear_vel,
            angular_velocity: b.angular_vel,
            awake: b.flags & engine_schema::BodyFlags::Sleeping as u32 == 0,
        })
    }

    /// Set the mass properties explicitly (Box3D `b3Body_SetMassData`); the
    /// body origin stays where it is.
    pub fn body_set_mass_data(&mut self, id: BodyId, mass: MassData) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        anyhow::ensure!(
            self.bodies[i].body_type == BodyType::Dynamic,
            "mass data on a non-dynamic body"
        );
        anyhow::ensure!(mass.mass > 0.0, "mass must be positive");
        self.set_mass(i, Some(mass))
    }

    /// Recompute the mass properties from the body's shapes.
    pub fn body_apply_mass_from_shapes(&mut self, id: BodyId) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        anyhow::ensure!(
            self.bodies[i].body_type == BodyType::Dynamic,
            "mass on a non-dynamic body"
        );
        let mass = self.shapes_mass(i);
        self.set_mass(i, mass)
    }

    /// Move a kinematic body to `transform` (its origin) over the next step
    /// (Box3D `b3Body_SetTargetTransform`): the velocity that gets it there is
    /// computed on the GPU from where it actually is when the step starts, so
    /// a body driven by animation every frame tracks it without drift. It
    /// pushes and carries dynamic bodies on the way.
    pub fn body_set_target_transform(
        &mut self,
        id: BodyId,
        transform: Transform,
    ) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        anyhow::ensure!(
            self.bodies[i].body_type == BodyType::Kinematic,
            "target transforms are for kinematic bodies"
        );
        let s = self.bodies[i].frame().to_solver(transform);
        self.command(
            id,
            GpuCommand::set_kinematic_target(id.index(), s.position, s.rotation),
        )
    }

    /// Box3D damping (1/s): velocities shrink by `1 / (1 + h c)` each step.
    pub fn body_set_damping(
        &mut self,
        id: BodyId,
        linear: f32,
        angular: f32,
    ) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        anyhow::ensure!(linear >= 0.0 && angular >= 0.0, "negative damping");
        self.bodies[i].damping = (linear, angular);
        Ok(())
    }

    pub fn body_damping(&self, id: BodyId) -> anyhow::Result<(f32, f32)> {
        Ok(self.bodies[self.live_body(id)?].damping)
    }

    /// Multiply gravity for this body (it wakes).
    pub fn body_set_gravity_scale(&mut self, id: BodyId, scale: f32) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        anyhow::ensure!(scale.is_finite(), "gravity_scale must be finite");
        if self.bodies[i].gravity_scale == scale {
            return Ok(());
        }
        self.bodies[i].gravity_scale = scale;
        // A body resting under the old scale may have slept: it moves again.
        self.command(id, GpuCommand::wake(id.index()))
    }

    pub fn body_gravity_scale(&self, id: BodyId) -> anyhow::Result<f32> {
        Ok(self.bodies[self.live_body(id)?].gravity_scale)
    }

    /// Lock axes (world frame) the body may not move along or turn about
    /// (Box3D `b3Body_SetMotionLocks`): its displacement and rotation since
    /// each step's start lose their locked components (the `lock_predict` and
    /// `lock_project` passes, scheduled only while some body has locks).
    pub fn body_set_motion_locks(&mut self, id: BodyId, locks: MotionLocks) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        self.bodies[i].locks = locks.bits();
        if locks.bits() != 0 {
            self.sim.set_motion_locks(true);
        }
        self.command(id, GpuCommand::set_locks(id.index(), locks.bits()))
    }

    pub fn body_motion_locks(&self, id: BodyId) -> anyhow::Result<MotionLocks> {
        Ok(MotionLocks::from_bits(
            self.bodies[self.live_body(id)?].locks,
        ))
    }

    /// This step's damping and gravity-scale commands: awake dynamic bodies
    /// whose values differ from (0, 0, 1). They neither wake bodies nor
    /// count as input, so an asleep world still skips its steps.
    fn push_damping(&mut self) -> anyhow::Result<()> {
        let p = self.sim.params();
        let h = p.dt;
        let mut cmds = Vec::new();
        for (i, b) in self.bodies.iter().enumerate() {
            if !b.alive || b.body_type != BodyType::Dynamic || b.last_asleep || !b.tuned() {
                continue;
            }
            let add = [0.0, h * (b.gravity_scale - 1.0) * p.gravity_y, 0.0];
            let cmd = GpuCommand::damp(
                i as u32,
                add,
                1.0 / (1.0 + h * b.damping.0),
                1.0 / (1.0 + h * b.damping.1),
            );
            cmds.push(cmd.without_wake());
        }
        if cmds.is_empty() {
            return Ok(());
        }
        self.sim.push_body_commands(&cmds)
    }

    /// Teleport the body origin (velocities kept; the body wakes).
    pub fn body_set_transform(&mut self, id: BodyId, transform: Transform) -> anyhow::Result<()> {
        let i = self.live_body(id)?;
        let s = self.bodies[i].frame().to_solver(transform);
        self.command(id, GpuCommand::teleport(id.index(), s.position, s.rotation))
    }

    /// Velocity of the centre of mass.
    pub fn body_set_linear_velocity(&mut self, id: BodyId, v: [f32; 3]) -> anyhow::Result<()> {
        self.command(id, GpuCommand::set_linear_velocity(id.index(), v))
    }

    pub fn body_set_angular_velocity(&mut self, id: BodyId, w: [f32; 3]) -> anyhow::Result<()> {
        self.command(id, GpuCommand::set_angular_velocity(id.index(), w))
    }

    /// Both velocities at once.
    pub fn body_set_velocity(
        &mut self,
        id: BodyId,
        v: [f32; 3],
        w: [f32; 3],
    ) -> anyhow::Result<()> {
        self.command(id, GpuCommand::set_velocity(id.index(), v, w))
    }

    /// A force (N) at a world point for the next step (Box3D clears forces
    /// after each step too). `wake = false` leaves a sleeping body alone.
    pub fn body_apply_force(
        &mut self,
        id: BodyId,
        force: [f32; 3],
        point: [f32; 3],
        wake: bool,
    ) -> anyhow::Result<()> {
        let cmd = GpuCommand::add_force_at(id.index(), force, point);
        self.command(id, wake_flag(cmd, wake))
    }

    pub fn body_apply_force_to_center(
        &mut self,
        id: BodyId,
        force: [f32; 3],
        wake: bool,
    ) -> anyhow::Result<()> {
        let cmd = GpuCommand::add_force(id.index(), force, [0.0; 3]);
        self.command(id, wake_flag(cmd, wake))
    }

    pub fn body_apply_torque(
        &mut self,
        id: BodyId,
        torque: [f32; 3],
        wake: bool,
    ) -> anyhow::Result<()> {
        let cmd = GpuCommand::add_force(id.index(), [0.0; 3], torque);
        self.command(id, wake_flag(cmd, wake))
    }

    /// An impulse (N s) at a world point.
    pub fn body_apply_linear_impulse(
        &mut self,
        id: BodyId,
        impulse: [f32; 3],
        point: [f32; 3],
        wake: bool,
    ) -> anyhow::Result<()> {
        let cmd = GpuCommand::apply_impulse(id.index(), impulse, point);
        self.command(id, wake_flag(cmd, wake))
    }

    pub fn body_apply_linear_impulse_to_center(
        &mut self,
        id: BodyId,
        impulse: [f32; 3],
        wake: bool,
    ) -> anyhow::Result<()> {
        let cmd = GpuCommand::apply_impulse_at_center(id.index(), impulse);
        self.command(id, wake_flag(cmd, wake))
    }

    pub fn body_apply_angular_impulse(
        &mut self,
        id: BodyId,
        impulse: [f32; 3],
        wake: bool,
    ) -> anyhow::Result<()> {
        let cmd = GpuCommand::angular_impulse(id.index(), impulse);
        self.command(id, wake_flag(cmd, wake))
    }

    /// Wake a body (and so its island). Putting a body to sleep on demand is
    /// not supported.
    pub fn body_set_awake(&mut self, id: BodyId, awake: bool) -> anyhow::Result<()> {
        if !awake {
            return unsupported("forcing a body to sleep");
        }
        self.command(id, GpuCommand::wake(id.index()))
    }

    // ----- shapes ----------------------------------------------------------

    pub fn create_sphere_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef,
        center: [f32; 3],
        radius: f32,
    ) -> anyhow::Result<ShapeId> {
        self.create_shape(body, def, Geometry::Sphere { center, radius })
    }

    /// Along local +Y between `center ± half_len`.
    pub fn create_capsule_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef,
        center: [f32; 3],
        radius: f32,
        half_len: f32,
    ) -> anyhow::Result<ShapeId> {
        self.create_shape(
            body,
            def,
            Geometry::Capsule {
                center,
                radius,
                half_len,
            },
        )
    }

    /// A box centred on the body origin (Box3D builds boxes as hulls; Rubble
    /// keeps a native box collider).
    pub fn create_box_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef,
        half: [f32; 3],
    ) -> anyhow::Result<ShapeId> {
        self.create_transformed_box_shape(body, def, half, Transform::IDENTITY)
    }

    pub fn create_transformed_box_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef,
        half: [f32; 3],
        transform: Transform,
    ) -> anyhow::Result<ShapeId> {
        self.create_shape(body, def, Geometry::Box { half, transform })
    }

    pub fn create_hull_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef,
        hull: &ConvexHull,
    ) -> anyhow::Result<ShapeId> {
        self.create_transformed_hull_shape(body, def, hull, Transform::IDENTITY)
    }

    pub fn create_transformed_hull_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef,
        hull: &ConvexHull,
        transform: Transform,
    ) -> anyhow::Result<ShapeId> {
        let geometry = Geometry::Hull {
            hull: hull.clone(),
            transform,
        };
        self.create_shape(body, def, geometry)
    }

    /// Terrain on a static body: `field` centred on `transform` (its +Y up).
    pub fn create_height_field_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef,
        field: &HeightField,
        transform: Transform,
    ) -> anyhow::Result<ShapeId> {
        let geometry = Geometry::HeightField {
            field: field.clone(),
            transform,
        };
        self.create_shape(body, def, geometry)
    }

    /// Overwrite the samples `[i0, i0 + w) × [j0, j0 + d)` of a heightfield
    /// shape (row-major `values`). Only the changed rows are uploaded; contacts
    /// are rebuilt on the next step, and bodies that were asleep (as of the
    /// last polled frame) over the edited cells are woken so they respond.
    /// Fails, changing nothing, if the edit is out of bounds or the wakes exceed
    /// `WorldCapacity::max_commands`.
    pub fn set_height_field_region(
        &mut self,
        id: ShapeId,
        i0: u32,
        j0: u32,
        w: u32,
        d: u32,
        values: &[f32],
    ) -> anyhow::Result<()> {
        let s = self.live_shape(id)?;
        self.terrain_before_edit(id)?;
        let Geometry::HeightField { field, transform } = &self.shapes[s].geometry else {
            anyhow::bail!("shape {} is not a heightfield", id.to_bits());
        };
        let (mut field, transform) = (field.clone(), *transform);
        field
            .set_region(i0, j0, w, d, values)
            .map_err(|e| anyhow::anyhow!("heightfield edit: {e}"))?;
        let wakes = self.sleepers_over_height_field(s, &field, transform, i0, j0, w, d);
        self.sim.push_body_commands(&wakes)?;
        let collider = self.shapes[s].collider;
        self.sim
            .update_height_field(&mut self.hulls, collider, &field, j0..j0 + d)
            .map_err(|e| anyhow::anyhow!("heightfield edit: {e}"))?;
        // The collider's filter bits live in `generation`: keep them.
        self.shapes[s].geometry = Geometry::HeightField { field, transform };
        self.dirty = true;
        Ok(())
    }

    /// Wake commands for the bodies asleep (as of the last polled frame) over
    /// samples `[i0, i0 + w) × [j0, j0 + d)` of heightfield shape `s`.
    #[allow(clippy::too_many_arguments)]
    fn sleepers_over_height_field(
        &self,
        s: usize,
        field: &HeightField,
        transform: Transform,
        i0: u32,
        j0: u32,
        w: u32,
        d: u32,
    ) -> Vec<GpuCommand> {
        // Sleeping bodies over the edited cells, from last frame's poses (the
        // terrain's static body: its origin until a frame reports it).
        let body = self.shapes[s].body as usize;
        let to_world = |p: Vec3| {
            let t = self.bodies[body].last_transform.unwrap_or(self.bodies[body].origin);
            t.p() + t.q() * (transform.p() + transform.q() * p)
        };
        let lo = field.vertex(i0, j0);
        let hi = field.vertex((i0 + w).saturating_sub(1), (j0 + d).saturating_sub(1));
        let (mut mn, mut mx) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for c in [Vec3::new(lo.x, -1e3, lo.z), Vec3::new(hi.x, 1e3, hi.z), Vec3::new(lo.x, 1e3, hi.z), Vec3::new(hi.x, -1e3, lo.z)] {
            let p = to_world(c);
            mn = mn.min(p);
            mx = mx.max(p);
        }
        let (dx, dz) = field.cell_size();
        let pad = Vec3::splat(dx.max(dz));
        self.bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| b.alive && b.last_asleep)
            .filter_map(|(i, b)| {
                let p = b.last_transform?.p();
                // Every collider's bounding sphere, about the body origin.
                let reach = Vec3::splat(
                    b.shapes
                        .iter()
                        .map(|&k| {
                            let c = self.sim.collider_record(self.shapes[k as usize].collider);
                            Vec3::from(c.local_pos).length() + c.params[3]
                        })
                        .fold(0.0, f32::max),
                );
                (p.cmpge(mn - pad - reach).all() && p.cmple(mx + pad + reach).all()).then(|| GpuCommand::wake(i as u32))
            })
            .collect()
    }

    /// Give a heightfield shape a soil (or take it away, `None`): vehicle
    /// wheels on it then sink to the rigid-wheel depth of their load, press
    /// ruts and pay compaction resistance (deformable terrain T5,
    /// `engine/docs/deformable-terrain.md`). A field can also be created with
    /// one (`HeightField::with_soil`). The field is placed again from this
    /// host's copy of its samples, which is fresh soil: ruts pressed so far
    /// are not kept, and bodies asleep over the field are woken. Fails,
    /// changing nothing, on invalid soil parameters or if the wakes exceed
    /// `WorldCapacity::max_commands`.
    pub fn set_height_field_soil(&mut self, id: ShapeId, soil: Option<Soil>) -> anyhow::Result<()> {
        let s = self.live_shape(id)?;
        let Geometry::HeightField { field, transform } = &self.shapes[s].geometry else {
            anyhow::bail!("shape {} is not a heightfield", id.to_bits());
        };
        let (mut field, transform) = (field.clone(), *transform);
        field
            .set_soil(soil)
            .map_err(|e| anyhow::anyhow!("heightfield soil: {e}"))?;
        // Ruts pressed so far are gone: wake what sleeps on the field.
        let (nx, nz) = field.samples();
        let wakes = self.sleepers_over_height_field(s, &field, transform, 0, 0, nx, nz);
        self.sim.push_body_commands(&wakes)?;
        let collider = self.shapes[s].collider;
        let mut col = self.sim.collider_record(collider);
        let old = engine_schema::convex::convex_params_base(col.params);
        let base = self
            .hulls
            .insert_height_field(&field)
            .map_err(|e| anyhow::anyhow!("heightfield: {e}"))?;
        col.params = engine_schema::heightfield::height_field_collider_params(base, &field);
        self.sim.set_collider(collider, col);
        // Rebuilds the terrain's contacts (and uploads the new block).
        self.sim
            .update_height_field(&mut self.hulls, collider, &field, 0..0)
            .map_err(|e| anyhow::anyhow!("heightfield: {e}"))?;
        // The old block is freed once no step in flight can read it.
        self.retired_height_fields.push((old, self.step));
        self.shapes[s].geometry = Geometry::HeightField { field, transform };
        self.dirty = true;
        Ok(())
    }

    /// Any geometry (the `create_*_shape` calls end here).
    pub fn create_shape(
        &mut self,
        body: BodyId,
        def: &ShapeDef,
        geometry: Geometry,
    ) -> anyhow::Result<ShapeId> {
        let b = self.live_body(body)?;
        anyhow::ensure!(
            self.bodies[b].owner == Owner::User,
            "shapes on vehicle-owned bodies"
        );
        def.validate()?;
        if matches!(geometry, Geometry::HeightField { .. }) {
            anyhow::ensure!(
                self.bodies[b].body_type == BodyType::Static,
                "heightfields go on static bodies"
            );
        }
        if def.is_sensor && self.contact_storage == ContactStorage::Compact {
            // Compact solver pairs are packed before the events pass runs.
            return unsupported("sensor shapes with ContactStorage::Compact");
        }
        if matches!(geometry, Geometry::Plane { .. }) {
            anyhow::ensure!(
                self.bodies[b].body_type == BodyType::Static,
                "planes go on static bodies"
            );
            let planes = self
                .shapes
                .iter()
                .filter(|s| s.alive && matches!(s.geometry, Geometry::Plane { .. }))
                .count();
            anyhow::ensure!(planes < 8, "at most 8 planes per world");
        }
        self.dirty = true;
        let material = self.intern_material(&def.material)?;
        let frame = self.bodies[b].frame();
        let mut collider =
            self.collider(body.index(), frame, &geometry, def.density, material, None)?;
        def.filter.apply(&mut collider);
        collider.material |= contacts::event_bits(def);
        self.contacts_shape_added(def);
        let collider_index = self.place_collider(collider)?;
        let slot = ShapeSlot {
            generation: 1,
            alive: true,
            body: body.index(),
            collider: collider_index,
            geometry,
            def: def.clone(),
        };
        let index = match self.free_shapes.pop() {
            Some(i) => {
                let generation = self.shapes[i as usize].generation;
                self.shapes[i as usize] = ShapeSlot { generation, ..slot };
                i
            }
            None => {
                self.shapes.push(slot);
                self.shapes.len() as u32 - 1
            }
        };
        if self.collider_shape.len() <= collider_index as usize {
            self.collider_shape
                .resize(collider_index as usize + 1, u32::MAX);
        }
        self.collider_shape[collider_index as usize] = index;
        self.bodies[b].shapes.push(index);
        if def.update_body_mass && self.bodies[b].body_type == BodyType::Dynamic {
            let mass = self.shapes_mass(b);
            self.set_mass(b, mass)?;
        }
        Ok(ShapeId::new(index, self.shapes[index as usize].generation))
    }

    /// A shape whose collider record the caller built (a threshold destructible's
    /// cooked chunk parts): placed as is, with `geometry` for queries and rewrites.
    /// Returns the shape slot and the collider.
    #[cfg_attr(not(feature = "stress"), allow(dead_code))]
    pub(crate) fn add_shape_record(
        &mut self,
        b: usize,
        def: &ShapeDef,
        geometry: Geometry,
        collider: Collider,
    ) -> anyhow::Result<(u32, u32)> {
        self.dirty = true;
        self.contacts_shape_added(def);
        let collider_index = self.place_collider(collider)?;
        let slot = ShapeSlot {
            generation: 1,
            alive: true,
            body: b as u32,
            collider: collider_index,
            geometry,
            def: def.clone(),
        };
        let index = match self.free_shapes.pop() {
            Some(i) => {
                let generation = self.shapes[i as usize].generation;
                self.shapes[i as usize] = ShapeSlot { generation, ..slot };
                i
            }
            None => {
                self.shapes.push(slot);
                self.shapes.len() as u32 - 1
            }
        };
        if self.collider_shape.len() <= collider_index as usize {
            self.collider_shape
                .resize(collider_index as usize + 1, u32::MAX);
        }
        self.collider_shape[collider_index as usize] = index;
        self.bodies[b].shapes.push(index);
        Ok((index, collider_index))
    }

    /// Destroy a shape (Box3D `b3DestroyShape`).
    pub fn destroy_shape(&mut self, id: ShapeId, update_body_mass: bool) -> anyhow::Result<()> {
        let s = self.live_shape(id)?;
        let b = self.shapes[s].body as usize;
        self.dirty = true;
        self.bodies[b].shapes.retain(|&x| x != s as u32);
        self.release_shape(s as u32);
        if update_body_mass && self.bodies[b].body_type == BodyType::Dynamic {
            let mass = self.shapes_mass(b);
            self.set_mass(b, mass)?;
        }
        Ok(())
    }

    pub fn shape_is_valid(&self, id: ShapeId) -> bool {
        self.live_shape(id).is_ok()
    }

    pub fn shape_body(&self, id: ShapeId) -> anyhow::Result<BodyId> {
        let b = self.shapes[self.live_shape(id)?].body;
        Ok(BodyId::new(b, self.bodies[b as usize].generation))
    }

    pub fn shape_user_data(&self, id: ShapeId) -> anyhow::Result<u64> {
        Ok(self.shapes[self.live_shape(id)?].def.user_data)
    }

    pub fn shape_surface_material(&self, id: ShapeId) -> anyhow::Result<SurfaceMaterial> {
        Ok(self.shapes[self.live_shape(id)?].def.material)
    }

    pub fn shape_filter(&self, id: ShapeId) -> anyhow::Result<Filter> {
        Ok(self.shapes[self.live_shape(id)?].def.filter)
    }

    /// The shape's geometry in its body's frame (from the body origin).
    pub fn shape_geometry(&self, id: ShapeId) -> anyhow::Result<&Geometry> {
        Ok(&self.shapes[self.live_shape(id)?].geometry)
    }

    /// A triangle mesh of a box, hull or heightfield shape in its body's frame:
    /// vertices and counter-clockwise (outward) triangles, for renderers.
    /// Spheres and capsules return `None` (draw them as primitives).
    pub fn shape_mesh(&self, id: ShapeId) -> anyhow::Result<Option<ShapeMesh>> {
        if let Geometry::HeightField { field, transform } = self.shape_geometry(id)? {
            return Ok(Some(height_field_mesh(field, *transform)));
        }
        let (hull, transform) = match self.shape_geometry(id)? {
            Geometry::Box { half, transform } => {
                (ConvexHull::cuboid(Vec3::from(*half)), *transform)
            }
            Geometry::Hull { hull, transform } => (hull.clone(), *transform),
            _ => return Ok(None),
        };
        let (q, p) = (transform.q(), transform.p());
        let vertices = hull
            .vertices
            .iter()
            .map(|v| (p + q * *v).to_array())
            .collect();
        let mut triangles = Vec::new();
        for f in &hull.faces {
            for k in 1..f.indices.len().saturating_sub(1) {
                let i = |n: usize| u32::from(f.indices[n]);
                triangles.push([i(0), i(k), i(k + 1)]);
            }
        }
        Ok(Some(ShapeMesh {
            vertices,
            triangles,
        }))
    }

    pub fn shape_set_filter(&mut self, id: ShapeId, filter: Filter) -> anyhow::Result<()> {
        let s = self.live_shape(id)?;
        self.shapes[s].def.filter = filter;
        self.rewrite_shape(s)
    }

    pub fn shape_set_surface_material(
        &mut self,
        id: ShapeId,
        material: SurfaceMaterial,
    ) -> anyhow::Result<()> {
        material.validate()?;
        let s = self.live_shape(id)?;
        self.shapes[s].def.material = material;
        self.rewrite_shape(s)
    }

    // ----- joints ----------------------------------------------------------

    pub fn create_joint(&mut self, def: &JointDef) -> anyhow::Result<JointId> {
        let (ida, idb) = def.bodies();
        let a = self.live_body(ida)?;
        let b = self.live_body(idb)?;
        let mut desc = def.desc()?;
        anyhow::ensure!(
            self.sim.joint_count() < self.sim.capacity().max_joints
                || self.joints.iter().any(|j| !j.alive),
            "WorldCapacity::max_joints reached"
        );
        desc.frame_a = joint_frame(self.bodies[a].frame(), desc.frame_a);
        desc.frame_b = joint_frame(self.bodies[b].frame(), desc.frame_b);
        Ok(self.add_joint_solver(desc, a, b, def.user_data()))
    }

    /// Register a joint whose frames are already in the bodies' solver frames
    /// (destructible bonds come that way from their asset).
    pub(crate) fn add_joint_solver(
        &mut self,
        mut desc: JointDesc,
        a: usize,
        b: usize,
        user_data: u64,
    ) -> JointId {
        self.dirty = true;
        desc.body_a = a as u32;
        desc.body_b = b as u32;
        let origin_frames = [
            joint_frame_origin(self.bodies[a].frame(), desc.frame_a),
            joint_frame_origin(self.bodies[b].frame(), desc.frame_b),
        ];
        let index = self.sim.add_joint_desc(&desc);
        let i = index as usize;
        if self.joints.len() <= i {
            let empty = JointSlot {
                generation: 1,
                alive: false,
                bodies: [0, 0],
                user_data: 0,
                origin_frames: [JointFrame::at(Vec3::ZERO); 2],
            };
            self.joints.resize(i + 1, empty);
        }
        let slot = &mut self.joints[i];
        debug_assert!(!slot.alive, "the sim reused a live joint slot");
        slot.alive = true;
        slot.bodies = [a as u32, b as u32];
        slot.user_data = user_data;
        slot.origin_frames = origin_frames;
        let id = JointId::new(index, slot.generation);
        self.bodies[a].joints.push(index);
        if b != a {
            self.bodies[b].joints.push(index);
        }
        id
    }

    /// Destroy a joint, broken or not (Box3D `b3DestroyJoint`).
    pub fn destroy_joint(&mut self, id: JointId) -> anyhow::Result<()> {
        let i = self.live_joint(id)?;
        self.dirty = true;
        let [a, b] = self.joints[i].bodies;
        for body in [a, b] {
            self.bodies[body as usize].joints.retain(|&j| j != i as u32);
        }
        self.release_joint(i as u32);
        Ok(())
    }

    pub fn joint_is_valid(&self, id: JointId) -> bool {
        self.live_joint(id).is_ok()
    }

    pub fn joint_user_data(&self, id: JointId) -> anyhow::Result<u64> {
        Ok(self.joints[self.live_joint(id)?].user_data)
    }

    /// Change a joint's drive (motor speed, spring target) in place.
    pub fn joint_set_drive(
        &mut self,
        id: JointId,
        velocity: f32,
        target: f32,
    ) -> anyhow::Result<()> {
        let i = self.live_joint(id)?;
        self.dirty = true;
        self.sim.set_joint_drive(i as u32, velocity, target);
        Ok(())
    }

    // ----- vehicles and soft bodies (Rubble) -------------------------------

    pub fn add_vehicle(
        &mut self,
        desc: &VehicleDesc,
        position: [f32; 3],
        rotation: [f32; 4],
    ) -> anyhow::Result<VehicleHandle> {
        self.dirty = true;
        let before = self.sim.body_count();
        let h = self
            .sim
            .add_vehicle(desc, position, rotation)
            .map_err(|e| anyhow::anyhow!("add_vehicle: {e}"))?;
        self.adopt_owned(before);
        self.vehicles.push(h);
        Ok(h)
    }

    pub fn set_vehicle_input(&mut self, vehicle: VehicleHandle, input: VehicleInput) {
        self.dirty = true;
        self.sim.set_vehicle_input(vehicle, input);
    }

    /// A vehicle's state (blocking readback).
    pub fn vehicle_state(&self, vehicle: VehicleHandle) -> VehicleState {
        self.sim.vehicle_state(vehicle)
    }

    pub fn add_soft_vehicle(
        &mut self,
        desc: &SoftVehicleDesc,
        position: [f32; 3],
        rotation: [f32; 4],
        options: &SpawnOptions,
    ) -> anyhow::Result<SoftHandle> {
        self.dirty = true;
        let before = self.sim.body_count();
        let h = self
            .sim
            .add_soft_vehicle(desc, position, rotation, options)
            .map_err(|e| anyhow::anyhow!("add_soft_vehicle: {e}"))?;
        self.adopt_owned(before);
        let e = &self.sim.soft_host().entries[h.0 as usize];
        let (first, n) = (e.first_body, e.particles);
        let core = e.core.map(|c| c.0);
        for i in first..first + n {
            self.bodies[i as usize].owner = Owner::SoftNode;
        }
        if let Some(v) = core {
            self.vehicles.push(v);
        }
        Ok(h)
    }

    /// Drive a soft vehicle: its rigid core's wheels and drivetrain take
    /// `input`, its steering hydros take `input.steer`.
    pub fn set_soft_vehicle_input(
        &mut self,
        soft: SoftHandle,
        input: VehicleInput,
    ) -> anyhow::Result<()> {
        let entry = self
            .sim
            .soft_host()
            .entries
            .get(soft.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("no soft body {}", soft.0))?;
        let core = entry.core.map(|c| c.0);
        self.dirty = true;
        if let Some(v) = core {
            self.sim.set_vehicle_input(v, input);
        }
        self.sim.set_soft_steer(soft, input.steer);
        Ok(())
    }

    /// The vehicle driving a soft body's rigid core, if it has wheels.
    pub fn soft_vehicle(&self, soft: SoftHandle) -> Option<VehicleHandle> {
        self.sim
            .soft_host()
            .entries
            .get(soft.0 as usize)?
            .core
            .map(|c| c.0)
    }

    /// Report this soft body's node positions in every frame (`Frame::soft`).
    pub fn soft_enable_output(&mut self, soft: SoftHandle, enabled: bool) -> anyhow::Result<()> {
        anyhow::ensure!(
            (soft.0 as usize) < self.sim.soft_host().entries.len(),
            "no soft body {}",
            soft.0
        );
        self.soft_outputs.retain(|&h| h != soft);
        if enabled {
            self.soft_outputs.push(soft);
        }
        Ok(())
    }

    /// Skin a render mesh onto a soft body (optionally one named part): each
    /// frame then carries its deformed vertices and normals (`Frame::skins`),
    /// computed on the GPU by the same Slang kernel the demos use.
    pub fn soft_bind_skin(
        &mut self,
        soft: SoftHandle,
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
        part: Option<&str>,
    ) -> anyhow::Result<SkinId> {
        let verts: Vec<Vec3> = vertices.iter().map(|v| Vec3::from(*v)).collect();
        let bindings = self
            .sim
            .soft_host()
            .bind_mesh(soft, &verts, part)
            .map_err(|e| anyhow::anyhow!("bind_mesh: {e}"))?;
        let skinner = engine_webgpu::SoftSkinner::new(&self.sim, &bindings, &verts, triangles);
        self.skins.push((soft, skinner));
        Ok(SkinId(self.skins.len() as u32 - 1))
    }

    /// A soft body's damage so far (blocking readback): beams bent and broken,
    /// plastic work, per-node deformation.
    pub fn soft_damage(&self, soft: SoftHandle) -> engine_softbody::SoftDamage {
        self.sim.read_soft_damage(soft)
    }

    /// The handle of a body a vehicle or soft body owns (chassis, wheels).
    pub fn owned_body(&self, index: u32) -> Option<BodyId> {
        let b = self.bodies.get(index as usize)?;
        (b.alive && b.owner == Owner::Vehicle).then(|| BodyId::new(index, b.generation))
    }

    // ----- queries ---------------------------------------------------------

    /// Batched ray casts, the closest hit of each (blocking). Asynchronous
    /// batched queries are planned (P5).
    pub fn cast_ray_blocking(&mut self, rays: &[RayInput]) -> Vec<Option<RayResult>> {
        let rays: Vec<engine_vehicle::Ray> = rays.iter().map(engine_ray).collect();
        let hits = self.sim.raycast_batch(&rays);
        hits.into_iter().map(|h| self.ray_result(h)).collect()
    }

    /// Queue ray casts (closest hit each) for the next step: they trace
    /// against that step's final poses on the GPU, and their results arrive
    /// with its frame (`Frame::ray_results`) without stalling. Batches queued
    /// before one step share its query pool (`VehicleCapacity::max_rays`).
    pub fn cast_rays(&mut self, rays: &[RayInput]) -> anyhow::Result<RayBatchId> {
        let first = self.pending_rays.len() as u32;
        let max = self.sim.vehicle_host().max_rays();
        anyhow::ensure!(
            first as usize + rays.len() <= max as usize,
            "{} ray queries this step exceed VehicleCapacity::max_rays ({max})",
            first as usize + rays.len()
        );
        self.next_ray_batch += 1;
        let id = RayBatchId(self.next_ray_batch);
        self.pending_rays.extend(rays.iter().map(engine_ray));
        self.pending_batches.push((id, first, rays.len() as u32));
        self.dirty = true;
        Ok(id)
    }

    fn ray_result(&self, hit: Option<engine_vehicle::RayHit>) -> Option<RayResult> {
        let h = hit?;
        let body = self.sim.collider_body(h.collider)?;
        let slot = self.bodies.get(body as usize)?;
        if !slot.alive {
            return None;
        }
        let shape = match self.collider_shape.get(h.collider as usize) {
            Some(&s) if s != u32::MAX => ShapeId::new(s, self.shapes[s as usize].generation),
            _ => ShapeId::NULL,
        };
        Some(RayResult {
            shape,
            body: BodyId::new(body, slot.generation),
            point: h.point,
            normal: h.normal,
            distance: h.distance,
        })
    }

    // ----- stepping --------------------------------------------------------

    /// Advance one fixed step (queued commands first). Never waits on a
    /// readback unless every readback slot is in flight and
    /// `Backpressure::Block`; collect results with [`World::poll`].
    pub fn step(&mut self) -> anyhow::Result<()> {
        let step = self.step;
        if std::mem::take(&mut self.dirty) {
            self.last_input_step = Some(step);
        }
        let rays = std::mem::take(&mut self.pending_rays);
        let batches = std::mem::take(&mut self.pending_batches);
        self.push_damping()?;
        let explosions = std::mem::take(&mut self.queries.explosions);
        if !explosions.is_empty() {
            self.sim.write_explosions(&explosions)?;
        }
        let boxes = std::mem::take(&mut self.queries.boxes);
        let query_ids = std::mem::take(&mut self.queries.ids);
        if !boxes.is_empty() {
            self.sim.write_overlap_queries(&boxes)?;
        }
        let contact_events = self.contacts.enabled_shapes > 0;
        if contact_events {
            self.sim.clear_contact_events();
        }
        let casts = self.motion_before_step();
        if !rays.is_empty() {
            // Forces the step to run, so its frame carries the hits.
            self.sim.write_ray_queries(&rays)?;
        }
        let fracture = self.fracture.active();
        #[cfg(feature = "stress")]
        if fracture && self.fracture_projectiles_dirty() {
            self.sync_projectiles()?;
        }
        let mut enc = self
            .sim
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("world step"),
            });
        // The terrain's substeps go first: the rigid step applies their reactions.
        self.terrain_open_strike()?;
        let terrain = self.terrain_encode(&mut enc)?;
        #[cfg(feature = "stress")]
        if fracture {
            self.encode_fracture_begin(&mut enc);
        }
        self.sim
            .step_encode(&mut enc)
            .map_err(|e| anyhow::anyhow!("step: {e:?}"))?;
        let ran = self.sim.last_step_ran();
        #[cfg(feature = "stress")]
        if fracture && ran {
            // Loads, breaks and splits from this step's contacts, before the next step.
            self.encode_fracture(&mut enc);
        }
        let mut mapped = None;
        if ran {
            // Ray results are never dropped: a frame with queries waits for a slot.
            // Nor are contact events: a lost frame would break begin/end pairing.
            // Nor are fracture states: a split must be adopted in the step it happened.
            let block = self.backpressure == Backpressure::Block
                || !batches.is_empty()
                || !query_ids.is_empty()
                || !casts.is_empty()
                || contact_events
                || fracture;
            if !self.readback.has_free() && block {
                self.wait_oldest()?;
            }
            if self.readback.has_free() {
                for (_, skin) in &self.skins {
                    skin.encode(&mut enc);
                }
                if !batches.is_empty() {
                    self.sim.encode_ray_queries(&mut enc);
                }
                let watched = self.terrain_watching();
                if watched {
                    // Reads the step's end state: after the rigid step.
                    self.sim.terrain_encode_watch(&mut enc);
                }
                let copies = self.terrain_extras(terrain, watched);
                let mut extras = extras(
                    &self.sim,
                    self.resync,
                    &self.vehicles,
                    &self.soft_outputs,
                    &self.skins,
                );
                if !batches.is_empty() {
                    let (first, count) = self.sim.ray_query_slots();
                    extras.push(readback::Extra {
                        kind: Extra::RayHits(batches),
                        buffer: self.sim.vehicle_pool_buffer(),
                        offset: u64::from(first) * 16,
                        size: u64::from(count) * 16,
                    });
                }
                extras.extend(Self::terrain_extra_copies(&self.sim, copies));
                #[cfg(feature = "stress")]
                if fracture {
                    extras.extend(self.fracture.extras());
                }
                if !query_ids.is_empty() {
                    self.sim.encode_overlap_queries(&mut enc);
                    let (buffer, offset, size) = self.sim.overlap_result_range();
                    extras.push(readback::Extra {
                        kind: Extra::Overlaps(query_ids),
                        buffer,
                        offset,
                        size,
                    });
                }
                if contact_events {
                    let (buffer, offset, size) = self.sim.contact_event_range();
                    extras.push(readback::Extra {
                        kind: Extra::Contacts,
                        buffer,
                        offset,
                        size,
                    });
                }
                extras.extend(motion::extras(&self.sim, &self.motion, casts));
                let slot = self.readback.encode(&self.sim, &mut enc, step, &extras);
                if terrain {
                    // Copied: the next frame's flags start clear.
                    self.sim.terrain_encode_flags_clear(&mut enc);
                }
                self.resync = false;
                mapped = Some(slot);
            } else {
                self.resync = true;
                let dropped = Pending::Empty {
                    step,
                    dropped: true,
                };
                self.readback.pending.push_back(dropped);
            }
            self.newest_ran = Some(step);
        } else {
            let skipped = Pending::Empty {
                step,
                dropped: false,
            };
            self.readback.pending.push_back(skipped);
        }
        self.sim.queue().submit([enc.finish()]);
        if let Some(slot) = mapped {
            self.readback.map(slot);
        }
        let _ = self.sim.device().poll(wgpu::PollType::Poll);
        self.step += 1;
        Ok(())
    }

    /// The frames of every step whose results are ready, oldest first. Never
    /// blocks. A capacity overflow in any step is an error.
    pub fn poll(&mut self) -> anyhow::Result<Vec<Frame>> {
        let _ = self.sim.device().poll(wgpu::PollType::Poll);
        self.collect(false)
    }

    /// Every ready frame folded into one (several steps per display frame):
    /// latest transforms, earliest previous transforms, all events in order.
    /// Never blocks.
    pub fn poll_coalesced(&mut self) -> anyhow::Result<CoalescedFrame> {
        Ok(coalesce(self.poll()?))
    }

    /// Wait for every step taken so far and return their frames.
    pub fn poll_blocking(&mut self) -> anyhow::Result<Vec<Frame>> {
        self.collect(true)
    }

    /// Step, wait, and return this step's frame (Box3D-style immediacy for
    /// tools, tests and simple hosts). Earlier unpolled frames are kept for
    /// the next `poll`.
    pub fn step_sync(&mut self) -> anyhow::Result<Frame> {
        self.step()?;
        let mut frames = self.poll_blocking()?;
        let last = frames.pop().expect("the step's frame");
        self.ready_frames = frames;
        Ok(last)
    }

    /// Every live body's current origin pose (blocking): to resynchronise after
    /// a dropped frame, or to initialise a renderer.
    pub fn read_transforms(&mut self) -> anyhow::Result<Vec<(BodyId, Transform)>> {
        let held = self.poll_blocking()?;
        self.ready_frames.extend(held);
        let records = self.sim.read_poses(0, self.sim.body_count());
        Ok(records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let slot = self.bodies.get(i)?;
                slot.alive.then(|| {
                    let solver = Transform::new(r.position, r.rotation);
                    let id = BodyId::new(i as u32, slot.generation);
                    (id, slot.frame().to_origin(solver))
                })
            })
            .collect())
    }

    // ----- internals -------------------------------------------------------

    fn collect(&mut self, wait: bool) -> anyhow::Result<Vec<Frame>> {
        let mut frames = std::mem::take(&mut self.ready_frames);
        while let Some(front) = self.readback.pending.front() {
            match *front {
                Pending::Empty { step, dropped } => {
                    self.readback.pending.pop_front();
                    frames.push(Frame {
                        step,
                        ran: dropped,
                        dropped,
                        ..Frame::default()
                    });
                }
                Pending::Mapped { step, slot } => {
                    if !self.readback.ready(self.sim.device(), slot, wait)? {
                        break;
                    }
                    self.readback.pending.pop_front();
                    let raw = self.readback.take(slot);
                    frames.push(self.frame(step, raw)?);
                }
            }
        }
        if let Some(last) = frames.last_mut() {
            let (ends, sensor_ends) = self.take_pending_contact_ends();
            last.contact_events.end.extend(ends);
            last.sensor_events.end.extend(sensor_ends);
        }
        Ok(frames)
    }

    fn wait_oldest(&mut self) -> anyhow::Result<()> {
        let mut held = Vec::new();
        while !self.readback.has_free() {
            let Some(front) = self.readback.pending.pop_front() else {
                break;
            };
            match front {
                Pending::Empty { .. } => held.push(front),
                Pending::Mapped { step, slot } => {
                    self.readback.ready(self.sim.device(), slot, true)?;
                    let raw = self.readback.take(slot);
                    let frame = self.frame(step, raw)?;
                    self.ready_frames.push(frame);
                }
            }
        }
        for p in held.into_iter().rev() {
            self.readback.pending.push_front(p);
        }
        Ok(())
    }

    fn frame(&mut self, step: u64, raw: readback::Readback<Extra>) -> anyhow::Result<Frame> {
        let c = raw.counters;
        // Counters describe the world as it is now when they are the newest
        // step's, or when nothing was awake and nothing was written since:
        // every later step left it unchanged, so the sim may skip asleep steps.
        let quiet_since = self.last_input_step.is_none_or(|s| s <= step);
        if self.newest_ran == Some(step) || (c.awake_count == 0 && quiet_since) {
            self.sim
                .drained(c)
                .map_err(|e| anyhow::anyhow!("step {step}: {e:?}"))?;
        } else {
            anyhow::ensure!(
                c.overflow_flags == 0,
                "step {step}: capacity overflow (flags {:#x}); raise the WorldCapacity",
                c.overflow_flags
            );
            self.sim.set_plan_hints(c.contact_count, c.color_count);
        }
        if let Some((contacts, colors)) = self.fracture.fixed_plan {
            self.sim.set_plan_hints(contacts, colors);
        }
        self.last_counters = Some(c);
        let ring = self.sim.ring_events(&raw.ring);
        let (character_events, projectile_events, lost_motion) =
            self.motion_events(step, &raw.ring);
        let mut destruction = Vec::new();
        for b in ring.joint_breaks {
            if let Some(slot) = self.joints.get(b.joint as usize) {
                destruction.push(DestructionEvent::JointBroke {
                    joint: JointId::new(b.joint, slot.generation),
                    force: b.force,
                    torque: b.torque,
                    user_data: slot.user_data,
                });
            }
        }
        destruction.extend(
            ring.mount_breaks
                .into_iter()
                .map(DestructionEvent::MountBroke),
        );
        destruction.extend(
            ring.soft
                .breaks
                .into_iter()
                .map(DestructionEvent::BeamBroke),
        );
        destruction.extend(
            ring.soft
                .pieces
                .into_iter()
                .map(DestructionEvent::PieceDetached),
        );
        let mut destruction = self.destruction_events(destruction);
        #[cfg(feature = "stress")]
        {
            let reads: Vec<(threshold::FractureRead, &[u8])> = raw
                .extras
                .iter()
                .filter_map(|(k, b)| match k {
                    Extra::Fracture(r) => Some((*r, b.as_slice())),
                    _ => None,
                })
                .collect();
            if !reads.is_empty() {
                destruction.extend(self.fracture_frame(step, &reads)?);
            }
        }
        let destruction = self.connection_events(destruction);
        let resync = raw
            .extras
            .iter()
            .find(|(k, _)| matches!(k, Extra::Resync))
            .map(|(_, b)| b.clone());
        let mut records: Vec<engine_webgpu::PoseRecord> = match &resync {
            // After a dropped frame: every body whose published pose or sleep
            // state differs from what was last delivered.
            Some(bytes) => bytes
                .chunks_exact(32)
                .enumerate()
                .filter_map(|(i, c)| self.resync_record(i as u32, &bytemuck::pod_read_unaligned(c)))
                .collect(),
            None => raw.moved,
        };
        records.sort_by_key(|r| r.body());
        let mut moves = Vec::with_capacity(records.len());
        for r in &records {
            let i = r.body() as usize;
            let Some(slot) = self.bodies.get_mut(i) else {
                continue;
            };
            if !slot.alive {
                continue;
            }
            let solver = Transform::new(r.position, r.rotation);
            let transform = slot.frame_at(step).to_origin(solver);
            let previous = if r.first() { None } else { slot.last_transform };
            slot.last_transform = Some(transform);
            slot.last_solver = Some(solver);
            if r.slept() {
                slot.last_asleep = true;
            }
            if r.woke() {
                slot.last_asleep = false;
            }
            if matches!(slot.owner, Owner::SoftNode | Owner::Character) {
                continue;
            }
            moves.push(BodyMoveEvent {
                body: BodyId::new(r.body(), slot.generation),
                transform,
                previous,
                fell_asleep: r.slept(),
                woke: r.woke(),
                user_data: slot.user_data,
            });
        }
        let mut vehicles = Vec::new();
        let mut soft = Vec::new();
        let mut skins: Vec<SkinOutput> = Vec::new();
        let mut record: Option<(VehicleHandle, Vec<[u32; 4]>)> = None;
        let mut ray_results = Vec::new();
        let mut terrain = None;
        let mut contact_events = ContactEvents::default();
        let mut sensor_events = SensorEvents::default();
        let mut overlap_results = Vec::new();
        let mut characters = Vec::new();
        let mut cast_results = Vec::new();
        let mut projectiles = Vec::new();
        let mut lost_contacts = 0;
        for (kind, bytes) in &raw.extras {
            match kind {
                Extra::Resync => {}
                #[cfg(feature = "stress")]
                Extra::Fracture(_) => {}
                Extra::Overlaps(ids) => {
                    let (results, lost) = self.decode_overlaps(ids, bytes);
                    overlap_results = results;
                    lost_contacts += lost;
                }
                Extra::Contacts => {
                    (contact_events, sensor_events, lost_contacts) = self.decode_contacts(bytes);
                }
                Extra::Characters => characters = self.decode_characters(step, bytes),
                Extra::Casts(ids) => cast_results = self.decode_casts(ids, bytes),
                Extra::Projectiles => projectiles = self.decode_projectiles(step, bytes),
                Extra::RayHits(batches) => {
                    let hits = self.sim.decode_ray_hits(&slots(bytes));
                    for (batch, first, count) in batches {
                        let range = *first as usize..(*first + *count) as usize;
                        ray_results.push(RayBatchResult {
                            batch: *batch,
                            hits: hits[range].iter().map(|h| self.ray_result(*h)).collect(),
                        });
                    }
                }
                Extra::VehicleRecord(v) => record = Some((*v, slots(bytes))),
                Extra::VehicleWheels(v) => {
                    if let Some((rv, r)) = record.take() {
                        debug_assert_eq!(rv, *v);
                        let state = self.sim.decode_vehicle_state(*v, &r, &slots(bytes));
                        vehicles.push(VehicleOutput { vehicle: *v, state });
                    }
                }
                Extra::SoftNodes(h) => soft.push(SoftOutput {
                    soft: *h,
                    nodes: bytes
                        .chunks_exact(32)
                        .map(|c| {
                            bytemuck::pod_read_unaligned::<engine_webgpu::PoseRecord>(c).position
                        })
                        .collect(),
                }),
                Extra::SkinPositions(id) => skins.push(SkinOutput {
                    skin: *id,
                    positions: xyz(bytes),
                    normals: Vec::new(),
                }),
                Extra::SkinNormals(id) => {
                    if let Some(s) = skins.iter_mut().find(|s| s.skin == *id) {
                        s.normals = xyz(bytes);
                    }
                }
                Extra::TerrainCounters => terrain = Some(self.terrain_counters(step, bytes)?),
                Extra::TerrainHeights(id) => self.terrain_heights(*id, bytes),
                Extra::TerrainWatch => self.terrain_strike(step, bytes),
            }
        }
        Ok(Frame {
            step,
            ran: true,
            body_events: BodyEvents { moves },
            destruction_events: destruction,
            vehicles,
            soft,
            skins,
            ray_results,
            contact_events,
            sensor_events,
            overlap_results,
            characters,
            character_events,
            cast_results,
            projectiles,
            projectile_events,
            // One ring: its overflow is counted once, however many readers.
            lost_events: ring.lost.max(lost_motion) + lost_contacts,
            dropped: false,
            resynced: resync.is_some(),
            counters: Some(c),
            terrain,
        })
    }

    /// The record a resync reports for body `i`, if it changed since the last
    /// delivered frame (or was never delivered).
    fn resync_record(
        &self,
        i: u32,
        r: &engine_webgpu::PoseRecord,
    ) -> Option<engine_webgpu::PoseRecord> {
        const SLEEPING: u32 = 1;
        const PUBLISHED: u32 = 2;
        let slot = self.bodies.get(i as usize)?;
        if !slot.alive || r.word & PUBLISHED == 0 {
            return None;
        }
        let asleep = r.word & SLEEPING != 0;
        let same_pose = slot.last_solver.is_some_and(|l| {
            bytemuck::bytes_of(&l.position) == bytemuck::bytes_of(&r.position)
                && bytemuck::bytes_of(&l.rotation) == bytemuck::bytes_of(&r.rotation)
        });
        if same_pose && asleep == slot.last_asleep {
            return None;
        }
        let mut word = i;
        if slot.last_transform.is_none() {
            word |= 1 << 30; // FIRST
        } else if asleep && !slot.last_asleep {
            word |= 1 << 28; // SLEPT
        } else if !asleep && slot.last_asleep {
            word |= 1 << 29; // WOKE
        }
        Some(engine_webgpu::PoseRecord { word, ..*r })
    }

    fn command(&mut self, id: BodyId, cmd: GpuCommand) -> anyhow::Result<()> {
        self.live_body(id)?;
        self.dirty = true;
        self.sim.push_body_commands(&[cmd])
    }

    fn live_body(&self, id: BodyId) -> anyhow::Result<usize> {
        let i = id.index() as usize;
        match self.bodies.get(i) {
            Some(b) if b.alive && b.generation == id.generation() => Ok(i),
            _ => Err(StaleHandle("body", id.to_bits()).into()),
        }
    }

    fn live_shape(&self, id: ShapeId) -> anyhow::Result<usize> {
        let i = id.index() as usize;
        match self.shapes.get(i) {
            Some(s) if s.alive && s.generation == id.generation() => Ok(i),
            _ => Err(StaleHandle("shape", id.to_bits()).into()),
        }
    }

    fn live_joint(&self, id: JointId) -> anyhow::Result<usize> {
        let i = id.index() as usize;
        match self.joints.get(i) {
            Some(j) if j.alive && j.generation == id.generation() => Ok(i),
            _ => Err(StaleHandle("joint", id.to_bits()).into()),
        }
    }

    fn release_joint(&mut self, index: u32) {
        self.sim.remove_joint(index);
        self.bond_owner.remove(&index);
        self.connection_owner.remove(&index);
        let slot = &mut self.joints[index as usize];
        slot.alive = false;
        slot.generation = slot.generation.wrapping_add(1).max(1);
    }

    fn release_shape(&mut self, index: u32) {
        self.contacts_shape_removed(index);
        let s = &mut self.shapes[index as usize];
        s.alive = false;
        s.generation = s.generation.wrapping_add(1).max(1);
        let (collider, body) = (s.collider, s.body);
        // Hull blocks stay cached for reuse (inserts deduplicate them); a
        // heightfield's block is its own, so it is freed once out of quarantine.
        if matches!(s.geometry, Geometry::HeightField { .. }) {
            let base = engine_schema::convex::convex_params_base(self.sim.collider_record(collider).params);
            self.retired_height_fields.push((base, self.step));
        }
        self.sim.set_collider(collider, retired_collider(body));
        self.collider_shape[collider as usize] = u32::MAX;
        self.free_colliders.push((collider, self.step));
        self.free_shapes.push(index);
    }

    fn reusable_body_slot(&mut self) -> Option<u32> {
        let now = self.step;
        let pos = self.free_bodies.iter().position(|&i| {
            self.bodies[i as usize]
                .removed_at
                .is_some_and(|t| now >= t + QUARANTINE_STEPS)
        })?;
        Some(self.free_bodies.swap_remove(pos))
    }

    /// Reuse a retired collider past quarantine, or append one.
    fn place_collider(&mut self, c: Collider) -> anyhow::Result<u32> {
        let now = self.step;
        let reusable = self
            .free_colliders
            .iter()
            .position(|&(_, t)| now >= t + QUARANTINE_STEPS);
        if let Some(pos) = reusable {
            let (index, _) = self.free_colliders.swap_remove(pos);
            self.sim.set_collider(index, c);
            return Ok(index);
        }
        let cap = self.sim.capacity().max_colliders;
        anyhow::ensure!(
            self.sim.collider_count() < cap,
            "WorldCapacity::max_colliders ({cap}) reached"
        );
        Ok(self.sim.add_collider(c))
    }

    fn intern_material(&mut self, m: &SurfaceMaterial) -> anyhow::Result<u32> {
        let key = m.key();
        if let Some(i) = self.materials.iter().position(|k| *k == key) {
            return Ok(i as u32);
        }
        anyhow::ensure!(
            self.materials.len() < 64,
            "more than 64 distinct surface materials (friction, restitution, contact offset)"
        );
        let i = self.materials.len() as u32;
        self.materials.push(key);
        self.sim.set_material(i, m.gpu());
        Ok(i)
    }

    /// The GPU collider for a shape. `hull_params` reuses an existing hull's
    /// arena entry instead of inserting the hull again.
    fn collider(
        &mut self,
        body: u32,
        frame: SolverFrame,
        g: &Geometry,
        density: f32,
        material: u32,
        hull_params: Option<[f32; 4]>,
    ) -> anyhow::Result<Collider> {
        let identity = [0.0, 0.0, 0.0, 1.0];
        let (shape, params, local) = match g {
            Geometry::Sphere { center, radius } => (
                ShapeType::Sphere,
                [*radius, 0.0, 0.0, *radius],
                Transform::new(*center, identity),
            ),
            Geometry::Capsule {
                center,
                radius,
                half_len,
            } => (
                ShapeType::Capsule,
                [*radius, *half_len, 0.0, *radius + *half_len],
                Transform::new(*center, identity),
            ),
            Geometry::Box { half, transform } => {
                let h = Vec3::from(*half);
                (ShapeType::Box, [h.x, h.y, h.z, h.length()], *transform)
            }
            Geometry::Hull { hull, transform } => {
                let params = match hull_params {
                    Some(p) => p,
                    None => {
                        let base = self
                            .hulls
                            .insert(hull)
                            .map_err(|e| anyhow::anyhow!("hull arena: {e:?}"))?;
                        self.sim.sync_hulls(&mut self.hulls);
                        engine_schema::convex::convex_collider_params(
                            base,
                            1.0,
                            hull.bounding_radius(),
                        )
                    }
                };
                (ShapeType::ConvexHull, params, *transform)
            }
            Geometry::HeightField { field, transform } => {
                let params = match hull_params {
                    Some(p) => p,
                    None => {
                        let now = self.step;
                        let hulls = &mut self.hulls;
                        self.retired_height_fields.retain(|&(base, t)| {
                            let due = now >= t + QUARANTINE_STEPS;
                            if due {
                                hulls.release(base);
                            }
                            !due
                        });
                        let base = self
                            .hulls
                            .insert_height_field(field)
                            .map_err(|e| anyhow::anyhow!("heightfield: {e}"))?;
                        self.sim.sync_hulls(&mut self.hulls);
                        engine_schema::heightfield::height_field_collider_params(base, field)
                    }
                };
                (ShapeType::HeightField, params, *transform)
            }
            Geometry::Plane { normal, offset } => {
                // The kernels read planes in world space: place this one by
                // the (static) body's pose now.
                let b = engine_softbody::SoftBackend::host_body(&self.sim, body);
                let q = Quat::from_array(b.orientation);
                let n = (q * Vec3::from(*normal)).normalize();
                let origin = Vec3::from(b.position) - q * frame.center;
                let d = *offset + n.dot(origin);
                return Ok(Collider {
                    body,
                    shape: ShapeType::Plane as u32,
                    material,
                    generation: 0,
                    local_pos: [0.0; 3],
                    density: 0.0,
                    local_rot: identity,
                    params: [n.x, n.y, n.z, d],
                });
            }
        };
        let local = frame.local_to_solver(local);
        Ok(Collider {
            body,
            shape: shape as u32,
            material,
            generation: 0,
            local_pos: local.position,
            density,
            local_rot: local.rotation,
            params,
        })
    }

    /// Rewrite a shape's collider (its material, filter or solver-frame pose).
    fn rewrite_shape(&mut self, s: usize) -> anyhow::Result<()> {
        self.dirty = true;
        let slot = self.shapes[s].clone();
        let frame = self.bodies[slot.body as usize].frame();
        let material = self.intern_material(&slot.def.material)?;
        let hull_params = match slot.geometry {
            Geometry::Hull { .. } | Geometry::HeightField { .. } => Some(self.sim.collider_record(slot.collider).params),
            _ => None,
        };
        let mut c = self.collider(
            slot.body,
            frame,
            &slot.geometry,
            slot.def.density,
            material,
            hull_params,
        )?;
        slot.def.filter.apply(&mut c);
        c.material |= contacts::event_bits(&slot.def);
        self.sim.set_collider(slot.collider, c);
        Ok(())
    }

    fn shapes_mass(&self, b: usize) -> Option<MassData> {
        let parts: Vec<_> = self.bodies[b]
            .shapes
            .iter()
            .map(|&s| {
                let s = &self.shapes[s as usize];
                defs::geometry_mass(&s.geometry, s.def.density)
            })
            .collect();
        defs::combine(&parts)
    }

    /// Give body `b` new mass properties, keeping its origin where it is.
    fn set_mass(&mut self, b: usize, mass: Option<MassData>) -> anyhow::Result<()> {
        let (next, moments) = match &mass {
            Some(m) => {
                let (moments, axes) = math::principal(m.tensor());
                anyhow::ensure!(
                    moments.min_element() >= 0.0,
                    "inertia tensor is not positive semi-definite"
                );
                let frame = SolverFrame {
                    center: Vec3::from(m.center),
                    axes,
                };
                (frame, moments)
            }
            None => (SolverFrame::IDENTITY, Vec3::ONE),
        };
        self.set_mass_frame(b, mass, next, moments)
    }

    /// Give body `b` mass properties with an exact solver frame (centre of
    /// mass and principal axes in the body frame) and principal moments, as an
    /// asset gives them; the body origin stays where it is.
    pub(crate) fn set_mass_frame(
        &mut self,
        b: usize,
        mass: Option<MassData>,
        next: SolverFrame,
        moments: Vec3,
    ) -> anyhow::Result<()> {
        self.dirty = true;
        let old = self.bodies[b].frame();
        let step = self.step;
        let slot = &mut self.bodies[b];
        slot.mass = mass;
        slot.principal = moments;
        if slot.frames.last().map(|f| f.0) == Some(step) {
            slot.frames.pop();
        }
        if next != slot.frame() {
            slot.frames.push((step, next));
        }
        if slot.written_step == step {
            // Nothing has simulated since the upload: rewrite it.
            let state = body_state(slot);
            self.sim.write_bodies(b as u32, &[state]);
        } else {
            // A live body re-centres on the GPU, without a readback.
            let (offset, rotation) = old.delta_to(&next);
            let (inv_mass, inv_inertia) = inverse_mass(slot);
            self.sim.push_body_commands(&[GpuCommand::set_mass_frame(
                b as u32,
                offset.to_array(),
                rotation.to_array(),
                inv_inertia,
                inv_mass,
            )])?;
        }
        if next != old {
            for s in self.bodies[b].shapes.clone() {
                self.rewrite_shape(s as usize)?;
            }
            // Joints are kept in solver frames: re-express this body's side.
            // (A plastic weld's accumulated bend restarts from its rest.)
            for j in self.bodies[b].joints.clone() {
                let slot = &self.joints[j as usize];
                if !slot.alive {
                    continue;
                }
                let (bodies, frames) = (slot.bodies, slot.origin_frames);
                for side in 0..2 {
                    if bodies[side] as usize == b {
                        let f = joint_frame(next, frames[side]);
                        self.sim.set_joint_frame(
                            j,
                            side == 1,
                            f.anchor.to_array(),
                            f.rot.to_array(),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn adopt_owned(&mut self, before: u32) {
        for _ in before..self.sim.body_count() {
            self.bodies.push(BodySlot {
                generation: 1,
                alive: true,
                owner: Owner::Vehicle,
                body_type: BodyType::Dynamic,
                user_data: 0,
                name: None,
                shapes: Vec::new(),
                joints: Vec::new(),
                removed_at: None,
                mass: None,
                principal: Vec3::ONE,
                frames: Vec::new(),
                written_step: 0,
                origin: Transform::IDENTITY,
                velocity: ([0.0; 3], [0.0; 3]),
                asleep: false,
                last_transform: None,
                last_asleep: false,
                last_solver: None,
                damping: (0.0, 0.0),
                gravity_scale: 1.0,
                locks: 0,
                bullet: false,
            });
        }
    }
}

fn wake_flag(cmd: GpuCommand, wake: bool) -> GpuCommand {
    if wake {
        cmd
    } else {
        cmd.without_wake()
    }
}

/// Inverse mass and inverse principal inertia for the GPU body record.
fn inverse_mass(slot: &BodySlot) -> (f32, [f32; 3]) {
    if slot.body_type != BodyType::Dynamic {
        return (0.0, [0.0; 3]);
    }
    let m = slot.mass.map_or(1.0, |m| m.mass);
    let moments = slot.principal;
    let inv = |x: f32| if x > 0.0 { 1.0 / x } else { 0.0 };
    (inv(m), [inv(moments.x), inv(moments.y), inv(moments.z)])
}

/// The GPU record for a body's upload.
fn body_state(slot: &BodySlot) -> BodyState {
    let solver = slot.frame().to_solver(slot.origin);
    let mut b = match slot.body_type {
        BodyType::Dynamic => {
            let mass = slot.mass.map_or(1.0, |m| m.mass);
            let inertia = slot.principal;
            BodyState::dynamic(
                Vec3::from(solver.position),
                Quat::from_array(solver.rotation),
                mass,
                inertia,
            )
        }
        BodyType::Kinematic => {
            // Infinite mass that moves at its set velocity (the kernels'
            // `bodyIsKinematic`: BODY_KINEMATIC without BODY_DYNAMIC).
            let mut s = BodyState::static_plane_support();
            s.position = solver.position;
            s.orientation = solver.rotation;
            s.flags =
                engine_schema::BodyFlags::Kinematic as u32 | engine_schema::BodyFlags::Awake as u32;
            s
        }
        BodyType::Static => {
            let mut s = BodyState::static_plane_support();
            s.position = solver.position;
            s.orientation = solver.rotation;
            s
        }
    };
    if slot.bullet {
        b.flags |= engine_schema::BODY_CCD;
    }
    if slot.body_type == BodyType::Dynamic {
        b.flags |= slot.locks << engine_schema::BODY_LOCK_SHIFT;
    }
    b.initial_pos = b.position;
    b.initial_rot = b.orientation;
    b.linear_vel = slot.velocity.0;
    b.angular_vel = slot.velocity.1;
    if slot.asleep {
        b.flags = (b.flags & !(engine_schema::BodyFlags::Awake as u32))
            | engine_schema::BodyFlags::Sleeping as u32;
    }
    b
}

/// The inverse of `joint_frame`: a solver-frame joint frame in the body's
/// origin frame.
fn joint_frame_origin(frame: SolverFrame, f: JointFrame) -> JointFrame {
    JointFrame {
        anchor: frame.center + frame.axes * f.anchor,
        rot: (frame.axes * f.rot).normalize(),
    }
}

fn joint_frame(frame: SolverFrame, f: JointFrame) -> JointFrame {
    let local = Transform::new(f.anchor.to_array(), f.rot.to_array());
    let t = frame.local_to_solver(local);
    JointFrame {
        anchor: Vec3::from(t.position),
        rot: Quat::from_array(t.rotation),
    }
}

fn slots(bytes: &[u8]) -> Vec<[u32; 4]> {
    bytes
        .chunks_exact(16)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

fn xyz(bytes: &[u8]) -> Vec<[f32; 3]> {
    bytes
        .chunks_exact(16)
        .map(|c| {
            let v: [f32; 4] = bytemuck::pod_read_unaligned(c);
            [v[0], v[1], v[2]]
        })
        .collect()
}

/// The extra copies a step's readback carries.
fn extras<'a>(
    sim: &'a WgpuSim,
    resync: bool,
    vehicles: &[VehicleHandle],
    soft: &[SoftHandle],
    skins: &'a [(SoftHandle, engine_webgpu::SoftSkinner)],
) -> Vec<readback::Extra<'a, Extra>> {
    let mut out = Vec::new();

    let pose = engine_webgpu::POSE_RECORD_BYTES;
    if resync {
        out.push(readback::Extra {
            kind: Extra::Resync,
            buffer: sim.pose_buffer(),
            offset: 0,
            size: u64::from(sim.body_count()) * pose,
        });
    }
    for &v in vehicles {
        let [(vs, vn), (ws, wn)] = sim.vehicle_state_slots(v);
        let pool = sim.vehicle_pool_buffer();
        out.push(readback::Extra {
            kind: Extra::VehicleRecord(v),
            buffer: pool,
            offset: u64::from(vs) * 16,
            size: u64::from(vn) * 16,
        });
        out.push(readback::Extra {
            kind: Extra::VehicleWheels(v),
            buffer: pool,
            offset: u64::from(ws) * 16,
            size: u64::from(wn) * 16,
        });
    }
    for &h in soft {
        let e = &sim.soft_host().entries[h.0 as usize];
        out.push(readback::Extra {
            kind: Extra::SoftNodes(h),
            buffer: sim.pose_buffer(),
            offset: u64::from(e.first_body) * pose,
            size: u64::from(e.particles) * pose,
        });
    }
    for (k, (_, skin)) in skins.iter().enumerate() {
        let size = u64::from(skin.vertex_count()) * 16;
        let id = SkinId(k as u32);
        out.push(readback::Extra {
            kind: Extra::SkinPositions(id),
            buffer: &skin.positions,
            offset: 0,
            size,
        });
        out.push(readback::Extra {
            kind: Extra::SkinNormals(id),
            buffer: &skin.normals,
            offset: 0,
            size,
        });
    }
    out
}

fn engine_ray(r: &RayInput) -> engine_vehicle::Ray {
    let t = Vec3::from(r.translation);
    engine_vehicle::Ray {
        origin: r.origin,
        dir: t.normalize_or_zero().to_array(),
        max_dist: t.length(),
        exclude_body: r.ignore_body.map(|b| b.index()),
    }
}

/// Two triangles per cell, counter-clockwise seen from +Y (the GPU's split).
fn height_field_mesh(field: &HeightField, transform: Transform) -> ShapeMesh {
    let (nx, nz) = field.samples();
    let (q, p) = (transform.q(), transform.p());
    let mut vertices = Vec::with_capacity((nx * nz) as usize);
    for j in 0..nz {
        for i in 0..nx {
            vertices.push((p + q * field.vertex(i, j)).to_array());
        }
    }
    let mut triangles = Vec::with_capacity(((nx - 1) * (nz - 1) * 2) as usize);
    for j in 0..nz - 1 {
        for i in 0..nx - 1 {
            let v = |di: u32, dj: u32| (j + dj) * nx + i + di;
            triangles.push([v(0, 0), v(1, 1), v(1, 0)]);
            triangles.push([v(0, 0), v(0, 1), v(1, 1)]);
        }
    }
    ShapeMesh { vertices, triangles }
}

#[cfg(test)]
mod height_field_arena_tests {
    use super::*;

    /// A removed heightfield's arena block is freed once out of quarantine, so
    /// replacing terrain again and again does not grow the arena.
    #[test]
    fn replacing_terrain_reuses_its_arena_block() {
        let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
        let field = HeightField::new(33, 33, 0.5, 0.5, vec![0.0; 33 * 33]).unwrap();
        let ground = w.create_body(&BodyDef::default()).unwrap();
        let mut len = None;
        for _ in 0..4 {
            let s = w
                .create_height_field_shape(ground, &ShapeDef::default(), &field, Transform::IDENTITY)
                .unwrap();
            let now = w.hulls.len();
            assert!(len.is_none_or(|l| now <= l), "the arena grew to {now} records (was {len:?})");
            len = Some(now);
            w.destroy_shape(s, false).unwrap();
            for _ in 0..QUARANTINE_STEPS + 1 {
                w.step().unwrap();
            }
        }
    }
}

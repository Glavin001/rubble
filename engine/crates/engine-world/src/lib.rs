//! Rubble's public world API: one handle for rigid bodies, joints, vehicles and
//! soft bodies simulated on the GPU, for game engines and simulations.
//!
//! - **Handles** ([`BodyId`], [`JointId`]) carry a generation: removing an
//!   object invalidates every copy of its handle.
//! - **Descriptors** ([`BodyDef`], [`ShapeDef`], [`JointDef`], [`MaterialDef`])
//!   have defaults and are copied on create.
//! - **Body commands** (`set_velocity`, `apply_impulse`, `add_force`, `teleport`,
//!   `wake`) are queued and applied on the GPU at the start of the next step,
//!   in submission order per body. Nothing is read back to apply them.
//! - **Results** come back asynchronously: [`World::step`] never waits on a
//!   readback, and [`World::poll`] returns each finished step's [`Frame`] (the
//!   bodies that moved, with their previous pose for interpolation, and the
//!   events: joint, beam and mount breaks, soft pieces, sleep and wake).
//! - **Capacities** are fixed at creation and fail loudly when exceeded.
//! - **Determinism:** a world gives the same results run to run on the same
//!   build and GPU. Its physics is exactly a [`WgpuSim`]'s: the pose output and
//!   readback only read state.
//!
//! The step is a fixed `dt` ([`WorldConfig::solver`]); hosts accumulate
//! real time and interpolate between [`MovedBody::previous`] and `pose`.

mod defs;
mod events;
mod handles;
mod readback;

pub use defs::{
    BodyDef, BodyKind, JointDef, MassSpec, MaterialDef, Shape, ShapeDef, MASS_ORIGIN_TOLERANCE,
    PRINCIPAL_TOLERANCE,
};
pub use events::{Event, Frame, MovedBody, Pose};
pub use handles::{BodyId, JointId, MaterialId};
pub use readback::Backpressure;

pub use engine_schema::convex::{CollisionFilter, ConvexHull};
pub use engine_schema::{AvbdParams, JointDesc, StepCounters, WorldCapacity};
pub use engine_softbody::{SoftCapacity, SoftHandle, SoftVehicleDesc, SpawnOptions};
pub use engine_vehicle::{
    Ray, VehicleCapacity, VehicleDesc, VehicleHandle, VehicleInput, VehicleState,
};
pub use engine_webgpu::{ContactStorage, EngineFeatures, WgpuSim};

use engine_schema::convex::HullArena;
use engine_schema::{BodyState, Collider, GpuCommand, ShapeType};
use engine_webgpu::SimOptions;
use glam::{Quat, Vec3};
use readback::{Pending, ReadbackRing};

/// The GPU a world runs on.
#[allow(clippy::large_enum_variant)] // Built once per world.
pub enum GpuContext {
    /// A high-performance adapter the world opens itself.
    Default,
    /// The host's device (a renderer's), so the world's buffers can be read in
    /// place by the host's own passes.
    Shared {
        info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
    },
}

/// Everything fixed at creation.
#[derive(Clone, Debug)]
pub struct WorldConfig {
    pub capacity: WorldCapacity,
    pub contacts: ContactStorage,
    pub vehicles: VehicleCapacity,
    /// A soft-body arena (needs [`ContactStorage::Canonical`]).
    pub soft: Option<SoftCapacity>,
    /// m/s². Only gravity along Y is supported for now; anything else is refused.
    pub gravity: [f32; 3],
    /// Step size, iterations, sleep thresholds and solver constants. Its
    /// `gravity_y` is replaced by `gravity`.
    pub solver: AvbdParams,
    pub features: EngineFeatures,
    /// Apply the `ENGINE_*` benchmark variables over `features` and `contacts`.
    pub env_overrides: bool,
    /// Steps whose results can be in flight at once.
    pub readback_frames: u32,
    pub backpressure: Backpressure,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            capacity: WorldCapacity::small(),
            contacts: ContactStorage::Persistent,
            vehicles: VehicleCapacity::default(),
            soft: None,
            gravity: [0.0, AvbdParams::default().gravity_y, 0.0],
            solver: AvbdParams::default(),
            features: EngineFeatures::default(),
            env_overrides: false,
            readback_frames: 3,
            backpressure: Backpressure::Block,
        }
    }
}

/// What a body slot holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    /// Made by `create_body`; removable.
    User,
    /// A vehicle's chassis, wheel or part, or a soft body's core: owned by it.
    Vehicle,
}

#[derive(Clone, Debug)]
struct BodySlot {
    generation: u32,
    alive: bool,
    owner: Owner,
    user_data: u64,
    colliders: Vec<u32>,
    joints: Vec<u32>,
    /// Removed at this step; the slot is reused two steps later, once neither
    /// contact history half can refer to its old colliders.
    removed_at: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct JointSlot {
    generation: u32,
    alive: bool,
    bodies: [u32; 2],
}

/// Steps a removed slot waits before reuse (the two contact ping-pong halves).
const QUARANTINE_STEPS: u64 = 2;

/// A parked (removed) body's pose: a sparse lattice far below the world.
fn graveyard(slot: u32) -> [f32; 3] {
    [
        (slot % 1024) as f32 * 8.0,
        -10_000.0,
        (slot / 1024) as f32 * 8.0,
    ]
}

/// A retired collider: a tiny sphere that collides with nothing.
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
    CollisionFilter {
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
    joints: Vec<JointSlot>,
    materials: u16,
    readback: ReadbackRing,
    backpressure: Backpressure,
    /// Last pose reported per body slot (interpolation source).
    last_pose: Vec<Option<Pose>>,
    step: u64,
    /// The newest step that ran on the GPU (its counters refresh the sim's).
    newest_ran: Option<u64>,
    /// Frames read while waiting for a free readback slot, for the next poll.
    ready_frames: Vec<Frame>,
    /// Something was written (a body, joint, command or vehicle input) since the
    /// last step.
    dirty: bool,
    /// The last step that applied host writes.
    last_input_step: Option<u64>,
}

impl World {
    pub fn new(gpu: GpuContext, config: WorldConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.gravity[0] == 0.0 && config.gravity[2] == 0.0,
            "gravity {:?}: only gravity along Y is supported for now",
            config.gravity
        );
        let (contacts, features) = if config.env_overrides {
            let env = SimOptions::from_env();
            (env.contacts, env.features)
        } else {
            (config.contacts, config.features)
        };
        let options = SimOptions {
            contacts,
            vehicles: config.vehicles,
            soft: config.soft,
            features,
            pose_output: true,
        };
        let mut sim = match gpu {
            GpuContext::Default => WgpuSim::new_with_options(config.capacity, options)?,
            GpuContext::Shared {
                info,
                device,
                queue,
            } => WgpuSim::with_device_options(config.capacity, options, info, device, queue)?,
        };
        let mut params = config.solver;
        params.gravity_y = config.gravity[1];
        sim.set_params(params);
        let readback = ReadbackRing::new(&sim, config.readback_frames);
        Ok(Self {
            sim,
            hulls: HullArena::new(),
            bodies: Vec::new(),
            free_bodies: Vec::new(),
            joints: Vec::new(),
            materials: 1,
            readback,
            backpressure: config.backpressure,
            last_pose: Vec::new(),
            step: 0,
            newest_ran: None,
            ready_frames: Vec::new(),
            dirty: false,
            last_input_step: None,
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

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    // ----- materials -------------------------------------------------------

    /// A new surface material (64 in all, the default included).
    pub fn create_material(&mut self, def: MaterialDef) -> anyhow::Result<MaterialId> {
        self.dirty = true;
        anyhow::ensure!(self.materials < 64, "all 64 material slots are in use");
        let id = MaterialId(self.materials);
        self.materials += 1;
        self.sim.set_material(u32::from(id.0), def.into());
        Ok(id)
    }

    /// Change a material (the default, `MaterialId::DEFAULT`, included).
    pub fn set_material(&mut self, id: MaterialId, def: MaterialDef) -> anyhow::Result<()> {
        self.dirty = true;
        anyhow::ensure!(id.0 < self.materials, "unknown material {}", id.0);
        self.sim.set_material(u32::from(id.0), def.into());
        Ok(())
    }

    // ----- bodies ----------------------------------------------------------

    /// A body made of `shapes` (at least one).
    pub fn create_body(&mut self, def: &BodyDef, shapes: &[ShapeDef]) -> anyhow::Result<BodyId> {
        self.dirty = true;
        anyhow::ensure!(!shapes.is_empty(), "a body needs at least one shape");
        for s in shapes {
            anyhow::ensure!(
                s.material.0 < self.materials,
                "unknown material {}",
                s.material.0
            );
        }
        let state = self.body_state(def, shapes)?;
        let slot = self.reusable_slot();
        let index = match slot {
            Some(i) => {
                self.sim.write_bodies(i, &[state]);
                self.sim.forget_pose(i);
                i
            }
            None => {
                anyhow::ensure!(
                    self.sim.body_count() < self.sim.capacity().max_bodies,
                    "WorldCapacity::max_bodies ({}) reached",
                    self.sim.capacity().max_bodies
                );
                self.sim.add_body(state)
            }
        };
        let mut colliders = Vec::with_capacity(shapes.len());
        for s in shapes {
            colliders.push(self.collider(index, s)?);
        }
        let mut ids = match slot {
            Some(i) => std::mem::take(&mut self.bodies[i as usize].colliders),
            None => Vec::new(),
        };
        // Reuse the slot's retired colliders, then append.
        let reused = ids.len().min(colliders.len());
        for (k, c) in colliders[..reused].iter().enumerate() {
            self.sim.set_collider(ids[k], *c);
        }
        if colliders.len() > reused {
            let free = self.sim.capacity().max_colliders - self.sim.collider_count();
            anyhow::ensure!(
                (colliders.len() - reused) as u32 <= free,
                "WorldCapacity::max_colliders ({}) reached",
                self.sim.capacity().max_colliders
            );
            let range = self.sim.add_colliders(&colliders[reused..]);
            ids.extend(range);
        }
        // Extra retired colliders stay with the slot (still retired) for later reuse.
        let generation = match slot {
            Some(i) => self.bodies[i as usize].generation,
            None => 1,
        };
        let entry = BodySlot {
            generation,
            alive: true,
            owner: Owner::User,
            user_data: def.user_data,
            colliders: ids,
            joints: Vec::new(),
            removed_at: None,
        };
        let i = index as usize;
        if i < self.bodies.len() {
            self.bodies[i] = entry;
        } else {
            debug_assert_eq!(i, self.bodies.len());
            self.bodies.push(entry);
        }
        if self.last_pose.len() <= i {
            self.last_pose.resize(i + 1, None);
        }
        self.last_pose[i] = None;
        Ok(BodyId::new(index, generation))
    }

    /// Remove a body and the joints attached to it. Its handle (and every copy)
    /// becomes invalid at once; its slot is reused after two steps.
    pub fn remove_body(&mut self, id: BodyId) -> anyhow::Result<()> {
        self.dirty = true;
        let i = self.live_body(id)?;
        anyhow::ensure!(
            self.bodies[i].owner == Owner::User,
            "body {} belongs to a vehicle or soft body; remove that instead",
            id.index()
        );
        for j in std::mem::take(&mut self.bodies[i].joints) {
            self.release_joint(j);
        }
        let index = id.index();
        self.sim.cancel_body_commands(index);
        let p = graveyard(index);
        let parked = BodyState {
            position: p,
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: p,
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        };
        self.sim.write_bodies(index, &[parked]);
        for &c in &self.bodies[i].colliders {
            self.sim.set_collider(c, retired_collider(index));
        }
        let slot = &mut self.bodies[i];
        slot.alive = false;
        slot.generation = slot.generation.wrapping_add(1).max(1);
        slot.removed_at = Some(self.step);
        slot.user_data = 0;
        self.free_bodies.push(index);
        self.last_pose[i] = None;
        Ok(())
    }

    /// Whether `id` refers to a live body.
    pub fn is_valid(&self, id: BodyId) -> bool {
        self.live_body(id).is_ok()
    }

    pub fn user_data(&self, id: BodyId) -> anyhow::Result<u64> {
        Ok(self.bodies[self.live_body(id)?].user_data)
    }

    /// The pose last reported for a body (`None` before its first frame).
    pub fn pose(&self, id: BodyId) -> anyhow::Result<Option<Pose>> {
        Ok(self.last_pose[self.live_body(id)?])
    }

    /// Bodies alive now.
    pub fn body_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.alive).count()
    }

    /// Overwrite a body's velocity (and wake it).
    pub fn set_velocity(
        &mut self,
        id: BodyId,
        linear: [f32; 3],
        angular: [f32; 3],
    ) -> anyhow::Result<()> {
        self.command(id, GpuCommand::set_velocity(id.index(), linear, angular))
    }

    /// An instantaneous impulse (N s) at a world point (and wake the body).
    pub fn apply_impulse(
        &mut self,
        id: BodyId,
        impulse: [f32; 3],
        point: [f32; 3],
    ) -> anyhow::Result<()> {
        self.command(id, GpuCommand::apply_impulse(id.index(), impulse, point))
    }

    /// A force (N) through the centre of mass and a torque (N m) for the next
    /// step only (and wake the body). Submit again every step it should act.
    pub fn add_force(
        &mut self,
        id: BodyId,
        force: [f32; 3],
        torque: [f32; 3],
    ) -> anyhow::Result<()> {
        self.command(id, GpuCommand::add_force(id.index(), force, torque))
    }

    /// Move a body without sweeping (respawn, pooling); velocities are kept.
    pub fn teleport(
        &mut self,
        id: BodyId,
        position: [f32; 3],
        rotation: [f32; 4],
    ) -> anyhow::Result<()> {
        self.command(id, GpuCommand::teleport(id.index(), position, rotation))
    }

    pub fn wake(&mut self, id: BodyId) -> anyhow::Result<()> {
        self.command(id, GpuCommand::wake(id.index()))
    }

    // ----- joints ----------------------------------------------------------

    pub fn create_joint(&mut self, def: &JointDef) -> anyhow::Result<JointId> {
        self.dirty = true;
        let a = self.live_body(def.body_a)?;
        let b = self.live_body(def.body_b)?;
        anyhow::ensure!(
            self.sim.joint_count() < self.sim.capacity().max_joints
                || self.joints.iter().any(|j| !j.alive),
            "WorldCapacity::max_joints reached"
        );
        let mut desc = def.desc;
        desc.body_a = a as u32;
        desc.body_b = b as u32;
        let index = self.sim.add_joint_desc(&desc);
        let i = index as usize;
        if self.joints.len() <= i {
            self.joints.resize(
                i + 1,
                JointSlot {
                    generation: 1,
                    alive: false,
                    bodies: [0, 0],
                },
            );
        }
        let slot = &mut self.joints[i];
        debug_assert!(!slot.alive, "the sim reused a live joint slot");
        slot.alive = true;
        slot.bodies = [a as u32, b as u32];
        let id = JointId::new(index, slot.generation);
        self.bodies[a].joints.push(index);
        if b != a {
            self.bodies[b].joints.push(index);
        }
        Ok(id)
    }

    /// Remove a joint (broken or not): the bodies collide again if it kept them apart.
    pub fn remove_joint(&mut self, id: JointId) -> anyhow::Result<()> {
        self.dirty = true;
        let i = self.live_joint(id)?;
        let [a, b] = self.joints[i].bodies;
        for body in [a, b] {
            self.bodies[body as usize].joints.retain(|&j| j != i as u32);
        }
        self.release_joint(i as u32);
        Ok(())
    }

    pub fn is_joint_valid(&self, id: JointId) -> bool {
        self.live_joint(id).is_ok()
    }

    /// Set a drive's target velocity and position in place.
    pub fn set_joint_drive(
        &mut self,
        id: JointId,
        velocity: f32,
        target: f32,
    ) -> anyhow::Result<()> {
        self.dirty = true;
        let i = self.live_joint(id)?;
        self.sim.set_joint_drive(i as u32, velocity, target);
        Ok(())
    }

    // ----- vehicles and soft bodies ---------------------------------------

    /// A raycast vehicle (chassis, wheels, breakable parts) at a chassis pose.
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

    /// A soft-body vehicle (needs `WorldConfig::soft`).
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
        Ok(h)
    }

    /// The body handle of a slot a vehicle or soft body owns (chassis, wheels).
    pub fn owned_body(&self, index: u32) -> Option<BodyId> {
        let b = self.bodies.get(index as usize)?;
        (b.alive && b.owner == Owner::Vehicle).then(|| BodyId::new(index, b.generation))
    }

    // ----- queries ---------------------------------------------------------

    /// Batched ray casts (blocking; directions need not be normalised). Each hit
    /// carries the body it struck.
    pub fn raycast(&mut self, rays: &[Ray]) -> Vec<Option<RayHit>> {
        let rays: Vec<Ray> = rays
            .iter()
            .map(|r| Ray {
                dir: Vec3::from(r.dir).normalize_or_zero().to_array(),
                ..*r
            })
            .collect();
        let hits = self.sim.raycast_batch(&rays);
        hits.into_iter()
            .map(|hit| {
                let h = hit?;
                let body = self.sim.collider_body(h.collider)?;
                let slot = self.bodies.get(body as usize)?;
                slot.alive.then(|| RayHit {
                    body: BodyId::new(body, slot.generation),
                    point: h.point,
                    normal: h.normal,
                    distance: h.distance,
                })
            })
            .collect()
    }

    // ----- stepping --------------------------------------------------------

    /// Advance one fixed step. Queued commands apply first. Never waits on a
    /// readback (unless every readback slot is in flight and
    /// `Backpressure::Block`); collect results with [`World::poll`].
    pub fn step(&mut self) -> anyhow::Result<()> {
        let step = self.step;
        if std::mem::take(&mut self.dirty) {
            self.last_input_step = Some(step);
        }
        let mut enc = self
            .sim
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("world step"),
            });
        self.sim
            .step_encode(&mut enc)
            .map_err(|e| anyhow::anyhow!("step: {e:?}"))?;
        let ran = self.sim.last_step_ran();
        let mut mapped = None;
        if ran {
            if !self.readback.has_free() && self.backpressure == Backpressure::Block {
                self.wait_oldest()?;
            }
            if self.readback.has_free() {
                let slot = self.readback.encode(&self.sim, &mut enc, step);
                mapped = Some((slot, self.sim.body_count()));
            } else {
                self.readback.pending.push_back(Pending::Empty {
                    step,
                    dropped: true,
                });
            }
            self.newest_ran = Some(step);
        } else {
            self.readback.pending.push_back(Pending::Empty {
                step,
                dropped: false,
            });
        }
        self.sim.queue().submit([enc.finish()]);
        if let Some((slot, bodies)) = mapped {
            self.readback.map(slot, bodies);
        }
        let _ = self.sim.device().poll(wgpu::PollType::Poll);
        self.step += 1;
        Ok(())
    }

    /// The frames of every step whose results are ready, oldest first. Never
    /// blocks. A capacity overflow in any step is returned as an error.
    pub fn poll(&mut self) -> anyhow::Result<Vec<Frame>> {
        let _ = self.sim.device().poll(wgpu::PollType::Poll);
        self.collect(false)
    }

    /// Wait for every step taken so far and return their frames.
    pub fn poll_blocking(&mut self) -> anyhow::Result<Vec<Frame>> {
        self.collect(true)
    }

    /// Every live body's current pose (blocking): to resynchronise after a
    /// dropped frame, or to initialise a renderer.
    pub fn read_poses(&mut self) -> Vec<(BodyId, Pose)> {
        self.poll_blocking().ok();
        let records = self.sim.read_poses(0, self.sim.body_count());
        records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let slot = self.bodies.get(i)?;
                slot.alive.then(|| {
                    (
                        BodyId::new(i as u32, slot.generation),
                        Pose {
                            position: r.position,
                            rotation: r.rotation,
                        },
                    )
                })
            })
            .collect()
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
        Ok(frames)
    }

    fn wait_oldest(&mut self) -> anyhow::Result<()> {
        // Frames read here are delivered by the next poll, in order.
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

    fn frame(&mut self, step: u64, raw: readback::Readback) -> anyhow::Result<Frame> {
        let c = raw.counters;
        // Counters describe the world as it is now when they are the newest step's,
        // or when nothing was awake and nothing was written since: then every
        // later step left it unchanged, and the sim may skip asleep steps.
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
        let ring = self.sim.ring_events(&raw.ring);
        let mut events = Vec::new();
        for b in ring.joint_breaks {
            if let Some(slot) = self.joints.get(b.joint as usize) {
                events.push(Event::JointBroke {
                    joint: JointId::new(b.joint, slot.generation),
                    force: b.force,
                    torque: b.torque,
                });
            }
        }
        events.extend(ring.mount_breaks.into_iter().map(Event::MountBroke));
        events.extend(ring.soft.breaks.into_iter().map(Event::BeamBroke));
        events.extend(ring.soft.pieces.into_iter().map(Event::PieceDetached));
        let mut records = raw.moved;
        records.sort_by_key(|r| r.body());
        let mut moved = Vec::with_capacity(records.len());
        for r in &records {
            let i = r.body() as usize;
            let Some(slot) = self.bodies.get(i) else {
                continue;
            };
            if !slot.alive {
                continue;
            }
            let body = BodyId::new(r.body(), slot.generation);
            let pose = Pose {
                position: r.position,
                rotation: r.rotation,
            };
            let previous = if r.first() { None } else { self.last_pose[i] };
            self.last_pose[i] = Some(pose);
            if r.slept() {
                events.push(Event::BodySlept {
                    body,
                    user_data: slot.user_data,
                });
            }
            if r.woke() {
                events.push(Event::BodyWoke {
                    body,
                    user_data: slot.user_data,
                });
            }
            moved.push(MovedBody {
                body,
                pose,
                previous,
                user_data: slot.user_data,
            });
        }
        Ok(Frame {
            step,
            ran: true,
            moved,
            events,
            lost_events: ring.lost,
            dropped: false,
            counters: Some(c),
        })
    }

    fn command(&mut self, id: BodyId, cmd: GpuCommand) -> anyhow::Result<()> {
        self.dirty = true;
        self.live_body(id)?;
        self.sim.push_body_commands(&[cmd])
    }

    fn live_body(&self, id: BodyId) -> anyhow::Result<usize> {
        let i = id.index() as usize;
        match self.bodies.get(i) {
            Some(b) if b.alive && b.generation == id.generation() => Ok(i),
            _ => anyhow::bail!("stale or invalid body handle {id:?}"),
        }
    }

    fn live_joint(&self, id: JointId) -> anyhow::Result<usize> {
        let i = id.index() as usize;
        match self.joints.get(i) {
            Some(j) if j.alive && j.generation == id.generation() => Ok(i),
            _ => anyhow::bail!("stale or invalid joint handle {id:?}"),
        }
    }

    fn release_joint(&mut self, index: u32) {
        self.sim.remove_joint(index);
        let slot = &mut self.joints[index as usize];
        slot.alive = false;
        slot.generation = slot.generation.wrapping_add(1).max(1);
    }

    /// A removed slot whose quarantine has passed.
    fn reusable_slot(&mut self) -> Option<u32> {
        let now = self.step;
        let pos = self.free_bodies.iter().position(|&i| {
            self.bodies[i as usize]
                .removed_at
                .is_some_and(|t| now >= t + QUARANTINE_STEPS)
        })?;
        Some(self.free_bodies.swap_remove(pos))
    }

    /// Register bodies a vehicle or soft body added (`before..body_count`).
    fn adopt_owned(&mut self, before: u32) {
        for i in before..self.sim.body_count() {
            debug_assert_eq!(i as usize, self.bodies.len());
            self.bodies.push(BodySlot {
                generation: 1,
                alive: true,
                owner: Owner::Vehicle,
                user_data: 0,
                colliders: Vec::new(),
                joints: Vec::new(),
                removed_at: None,
            });
        }
        self.last_pose.resize(self.bodies.len(), None);
    }

    fn body_state(&self, def: &BodyDef, shapes: &[ShapeDef]) -> anyhow::Result<BodyState> {
        let pos = Vec3::from(def.position);
        let rot = Quat::from_array(def.rotation).normalize();
        let mut b = match def.kind {
            BodyKind::Static => {
                let mut s = BodyState::static_plane_support();
                s.position = pos.to_array();
                s.orientation = rot.to_array();
                s
            }
            BodyKind::Dynamic => {
                let (mass, inertia) = match def.mass {
                    MassSpec::Explicit { mass, inertia } => (mass, Vec3::from(inertia)),
                    MassSpec::FromShapes => defs::shapes_mass(shapes)?,
                    MassSpec::Mass(m) => {
                        let (m0, i0) = defs::shapes_mass(shapes)?;
                        (m, i0 * (m / m0))
                    }
                };
                anyhow::ensure!(mass > 0.0, "a dynamic body needs a positive mass");
                BodyState::dynamic(pos, rot, mass, inertia)
            }
        };
        b.initial_pos = b.position;
        b.initial_rot = b.orientation;
        b.linear_vel = def.linear_velocity;
        b.angular_vel = def.angular_velocity;
        if def.asleep && def.kind == BodyKind::Dynamic {
            b.flags = (b.flags & !(engine_schema::BodyFlags::Awake as u32))
                | engine_schema::BodyFlags::Sleeping as u32;
        }
        Ok(b)
    }

    fn collider(&mut self, body: u32, s: &ShapeDef) -> anyhow::Result<Collider> {
        let (shape, params) = match &s.shape {
            Shape::Sphere { radius } => (ShapeType::Sphere, [*radius, 0.0, 0.0, *radius]),
            Shape::Box { half } => {
                let h = Vec3::from(*half);
                (ShapeType::Box, [h.x, h.y, h.z, h.length()])
            }
            Shape::Capsule { radius, half_len } => (
                ShapeType::Capsule,
                [*radius, *half_len, 0.0, *radius + *half_len],
            ),
            Shape::Convex(hull) => {
                let base = self
                    .hulls
                    .insert(hull)
                    .map_err(|e| anyhow::anyhow!("hull arena: {e:?}"))?;
                self.sim.sync_hulls(&mut self.hulls);
                (
                    ShapeType::ConvexHull,
                    engine_schema::convex::convex_collider_params(
                        base,
                        1.0,
                        hull.bounding_radius(),
                    ),
                )
            }
        };
        let mut c = Collider {
            body,
            shape: shape as u32,
            material: u32::from(s.material.0),
            generation: 0,
            local_pos: s.local_position,
            density: s.density,
            local_rot: s.local_rotation,
            params,
        };
        s.filter.apply(&mut c);
        Ok(c)
    }
}

/// A ray hit on a body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub body: BodyId,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub distance: f32,
}

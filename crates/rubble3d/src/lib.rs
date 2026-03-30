//! `rubble3d` — Public API facade for the rubble 3D physics engine.
//!
//! Orchestrates the full simulation pipeline:
//! broadphase (LBVH) → narrowphase → solver.
//!
//! Optionally dispatches the entire pipeline on the GPU via [`gpu::GpuPipeline`].

pub mod gpu;

use std::collections::HashSet;

use glam::{Mat3, Quat, Vec3, Vec4};
use rubble_broadphase3d::{find_plane_pairs, Lbvh};
use rubble_math::{
    Aabb3D, BodyHandle, CollisionEvent, Contact3D, RigidBodyProps3D, RigidBodyState3D, FLAG_STATIC,
    SHAPE_BOX, SHAPE_CAPSULE, SHAPE_SPHERE,
};
use rubble_narrowphase3d::{
    box_box, capsule_capsule, plane_box, plane_sphere, reduce_manifold, sphere_box, sphere_capsule,
    sphere_sphere, ContactPersistence,
};
use rubble_shapes3d::{
    compute_box_aabb, compute_capsule_aabb, compute_sphere_aabb, BoxData, CapsuleData, Plane,
    SphereData,
};
use rubble_solver3d::{Solver3D, SolverParams};

// ---------------------------------------------------------------------------
// SimConfig
// ---------------------------------------------------------------------------

/// Top-level simulation configuration.
pub struct SimConfig {
    pub gravity: Vec3,
    pub dt: f32,
    pub solver_iterations: u32,
    pub max_bodies: usize,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 65536,
        }
    }
}

// ---------------------------------------------------------------------------
// ShapeDesc
// ---------------------------------------------------------------------------

/// Describes a collision shape for body creation.
#[derive(Debug, Clone, Copy)]
pub enum ShapeDesc {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { half_height: f32, radius: f32 },
}

// ---------------------------------------------------------------------------
// RigidBodyDesc
// ---------------------------------------------------------------------------

/// Descriptor for creating a rigid body.
pub struct RigidBodyDesc {
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    /// Mass of the body. Use 0 for static bodies.
    pub mass: f32,
    pub friction: f32,
    pub shape: ShapeDesc,
}

impl Default for RigidBodyDesc {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
            shape: ShapeDesc::Sphere { radius: 0.5 },
        }
    }
}

// ---------------------------------------------------------------------------
// Inertia computation
// ---------------------------------------------------------------------------

/// Compute the inverse inertia tensor for a shape given its mass.
/// Returns `Mat3::ZERO` for static bodies (mass <= 0).
fn compute_inertia(shape: &ShapeDesc, mass: f32) -> Mat3 {
    if mass <= 0.0 {
        return Mat3::ZERO;
    }
    let diag = match shape {
        ShapeDesc::Sphere { radius } => {
            // I = 2/5 * m * r^2  (each diagonal)
            let i = (2.0 / 5.0) * mass * radius * radius;
            Vec3::splat(i)
        }
        ShapeDesc::Box { half_extents } => {
            // Full extents: w=2*hx, h=2*hy, d=2*hz
            let w = 2.0 * half_extents.x;
            let h = 2.0 * half_extents.y;
            let d = 2.0 * half_extents.z;
            // Ixx = m/12 * (h^2 + d^2), Iyy = m/12 * (w^2 + d^2), Izz = m/12 * (w^2 + h^2)
            Vec3::new(
                mass / 12.0 * (h * h + d * d),
                mass / 12.0 * (w * w + d * d),
                mass / 12.0 * (w * w + h * h),
            )
        }
        ShapeDesc::Capsule {
            half_height,
            radius,
        } => {
            // Approximate as cylinder: height = 2*half_height, radius = radius
            let r2 = radius * radius;
            let full_h = 2.0 * half_height;
            let h2 = full_h * full_h;
            // Ixx = Izz = m/12 * (3*r^2 + h^2), Iyy = m/2 * r^2
            Vec3::new(
                mass / 12.0 * (3.0 * r2 + h2),
                mass / 2.0 * r2,
                mass / 12.0 * (3.0 * r2 + h2),
            )
        }
    };

    // Inverse of a diagonal inertia tensor.
    Mat3::from_diagonal(Vec3::new(1.0 / diag.x, 1.0 / diag.y, 1.0 / diag.z))
}

// ---------------------------------------------------------------------------
// GenerationalIndexAllocator
// ---------------------------------------------------------------------------

/// Generational index allocator for stable body handles.
struct GenerationalIndexAllocator {
    generations: Vec<u32>,
    free_list: Vec<u32>,
}

impl GenerationalIndexAllocator {
    fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_list: Vec::new(),
        }
    }

    fn alloc(&mut self) -> BodyHandle {
        if let Some(index) = self.free_list.pop() {
            let gen = self.generations[index as usize];
            BodyHandle::new(index, gen)
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            BodyHandle::new(index, 0)
        }
    }

    fn dealloc(&mut self, handle: BodyHandle) -> bool {
        let idx = handle.index as usize;
        if idx >= self.generations.len() {
            return false;
        }
        if self.generations[idx] != handle.generation {
            return false;
        }
        self.generations[idx] += 1;
        self.free_list.push(handle.index);
        true
    }

    fn is_valid(&self, handle: BodyHandle) -> bool {
        let idx = handle.index as usize;
        idx < self.generations.len() && self.generations[idx] == handle.generation
    }
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

/// The main 3D physics world. Add bodies, step the simulation, query results.
///
/// When created with [`World::new_gpu`], the simulation step dispatches
/// compute shaders on the GPU. Otherwise it falls back to the CPU solver.
pub struct World {
    config: SimConfig,
    states: Vec<RigidBodyState3D>,
    props: Vec<RigidBodyProps3D>,
    inv_inertias: Vec<Mat3>,
    shapes: Vec<ShapeDesc>,
    spheres: Vec<SphereData>,
    boxes: Vec<BoxData>,
    capsules: Vec<CapsuleData>,
    planes: Vec<Plane>,
    allocator: GenerationalIndexAllocator,
    solver: Solver3D,
    persistence: ContactPersistence,
    prev_pairs: HashSet<(u32, u32)>,
    events: Vec<CollisionEvent>,
    /// Tracks which slot indices are alive (not removed).
    alive: Vec<bool>,
    /// Optional GPU pipeline. When `Some`, [`step`] dispatches on the GPU.
    gpu_pipeline: Option<gpu::GpuPipeline>,
}

impl World {
    /// Create a new physics world with the given configuration.
    pub fn new(config: SimConfig) -> Self {
        let solver = Solver3D::new(SolverParams {
            iterations: config.solver_iterations,
            ..Default::default()
        });
        let persistence = ContactPersistence::new(0.95);
        Self {
            config,
            states: Vec::new(),
            props: Vec::new(),
            inv_inertias: Vec::new(),
            shapes: Vec::new(),
            spheres: Vec::new(),
            boxes: Vec::new(),
            capsules: Vec::new(),
            planes: Vec::new(),
            allocator: GenerationalIndexAllocator::new(),
            solver,
            persistence,
            prev_pairs: HashSet::new(),
            events: Vec::new(),
            alive: Vec::new(),
            gpu_pipeline: None,
        }
    }

    /// Create a new physics world backed by GPU compute shaders.
    ///
    /// The GPU context is created synchronously (blocking). If no GPU adapter
    /// is available, returns an error.
    pub fn new_gpu(config: SimConfig) -> Result<Self, rubble_gpu::GpuError> {
        let ctx = pollster::block_on(rubble_gpu::GpuContext::new())?;
        let pipeline = gpu::GpuPipeline::new(ctx, config.max_bodies);

        let solver = Solver3D::new(SolverParams {
            iterations: config.solver_iterations,
            ..Default::default()
        });
        let persistence = ContactPersistence::new(0.95);
        Ok(Self {
            config,
            states: Vec::new(),
            props: Vec::new(),
            inv_inertias: Vec::new(),
            shapes: Vec::new(),
            spheres: Vec::new(),
            boxes: Vec::new(),
            capsules: Vec::new(),
            planes: Vec::new(),
            allocator: GenerationalIndexAllocator::new(),
            solver,
            persistence,
            prev_pairs: HashSet::new(),
            events: Vec::new(),
            alive: Vec::new(),
            gpu_pipeline: Some(pipeline),
        })
    }

    /// Returns `true` if this world has an active GPU pipeline.
    pub fn has_gpu(&self) -> bool {
        self.gpu_pipeline.is_some()
    }

    /// Add a rigid body to the world. Returns a stable handle.
    pub fn add_body(&mut self, desc: &RigidBodyDesc) -> BodyHandle {
        let handle = self.allocator.alloc();
        let idx = handle.index as usize;

        let inv_mass = if desc.mass > 0.0 {
            1.0 / desc.mass
        } else {
            0.0
        };

        let state = RigidBodyState3D::new(
            desc.position,
            inv_mass,
            desc.rotation,
            desc.linear_velocity,
            desc.angular_velocity,
        );

        let inv_inertia = compute_inertia(&desc.shape, desc.mass);

        let flags = if desc.mass <= 0.0 { FLAG_STATIC } else { 0 };

        let (shape_type, shape_index) = match &desc.shape {
            ShapeDesc::Sphere { radius } => {
                let si = self.spheres.len() as u32;
                self.spheres.push(SphereData {
                    radius: *radius,
                    _pad: [0.0; 3],
                });
                (SHAPE_SPHERE, si)
            }
            ShapeDesc::Box { half_extents } => {
                let si = self.boxes.len() as u32;
                self.boxes.push(BoxData {
                    half_extents: half_extents.extend(0.0),
                });
                (SHAPE_BOX, si)
            }
            ShapeDesc::Capsule {
                half_height,
                radius,
            } => {
                let si = self.capsules.len() as u32;
                self.capsules.push(CapsuleData {
                    half_height: *half_height,
                    radius: *radius,
                    _pad: [0.0; 2],
                });
                (SHAPE_CAPSULE, si)
            }
        };

        let prop =
            RigidBodyProps3D::new(inv_inertia, desc.friction, shape_type, shape_index, flags);

        // Ensure arrays are large enough for this index.
        if idx >= self.states.len() {
            self.states.resize(idx + 1, bytemuck::Zeroable::zeroed());
            self.props.resize(idx + 1, bytemuck::Zeroable::zeroed());
            self.inv_inertias.resize(idx + 1, Mat3::ZERO);
            self.shapes
                .resize(idx + 1, ShapeDesc::Sphere { radius: 0.5 });
            self.alive.resize(idx + 1, false);
        }

        self.states[idx] = state;
        self.props[idx] = prop;
        self.inv_inertias[idx] = inv_inertia;
        self.shapes[idx] = desc.shape;
        self.alive[idx] = true;

        handle
    }

    /// Remove a body from the world. Returns `true` if it was successfully removed.
    pub fn remove_body(&mut self, handle: BodyHandle) -> bool {
        if !self.allocator.is_valid(handle) {
            return false;
        }
        let idx = handle.index as usize;
        self.alive[idx] = false;
        self.states[idx] = bytemuck::Zeroable::zeroed();
        self.props[idx] = bytemuck::Zeroable::zeroed();
        self.inv_inertias[idx] = Mat3::ZERO;
        self.allocator.dealloc(handle);
        true
    }

    /// Number of currently alive bodies.
    pub fn body_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Get the position of a body, if the handle is valid.
    pub fn get_position(&self, handle: BodyHandle) -> Option<Vec3> {
        if !self.allocator.is_valid(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].position())
    }

    /// Get the orientation of a body, if the handle is valid.
    pub fn get_rotation(&self, handle: BodyHandle) -> Option<Quat> {
        if !self.allocator.is_valid(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].quat())
    }

    /// Get the linear velocity of a body, if the handle is valid.
    pub fn get_velocity(&self, handle: BodyHandle) -> Option<Vec3> {
        if !self.allocator.is_valid(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].linear_velocity())
    }

    /// Set the position of a body.
    pub fn set_position(&mut self, handle: BodyHandle, pos: Vec3) {
        if !self.allocator.is_valid(handle) {
            return;
        }
        let idx = handle.index as usize;
        let im = self.states[idx].inv_mass();
        self.states[idx].position_inv_mass = Vec4::new(pos.x, pos.y, pos.z, im);
    }

    /// Set the linear velocity of a body.
    pub fn set_velocity(&mut self, handle: BodyHandle, vel: Vec3) {
        if !self.allocator.is_valid(handle) {
            return;
        }
        let idx = handle.index as usize;
        self.states[idx].lin_vel = vel.extend(0.0);
    }

    /// Add an infinite half-space plane to the world.
    pub fn add_plane(&mut self, normal: Vec3, distance: f32) {
        self.planes.push(Plane {
            normal: normal.normalize(),
            distance,
        });
    }

    /// Advance the simulation by one time step.
    ///
    /// If a GPU pipeline is present, dispatches the physics step on the GPU.
    /// Otherwise falls back to the CPU solver.
    ///
    /// Pipeline: compute AABBs -> LBVH broadphase -> plane broadphase ->
    /// narrowphase contacts -> persistence (warm start) -> solver -> event detection.
    pub fn step(&mut self) {
        if self.gpu_pipeline.is_some() {
            self.step_gpu();
            return;
        }
        self.step_cpu();
    }

    /// GPU-accelerated simulation step.
    fn step_gpu(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        let alive_indices: Vec<usize> = (0..n).filter(|&i| self.alive[i]).collect();
        if alive_indices.is_empty() {
            return;
        }

        // Build compact arrays for GPU upload
        let compact_states: Vec<RigidBodyState3D> =
            alive_indices.iter().map(|&i| self.states[i]).collect();
        let compact_props: Vec<RigidBodyProps3D> =
            alive_indices.iter().map(|&i| self.props[i]).collect();
        let num_bodies = compact_states.len() as u32;

        let pipeline = self.gpu_pipeline.as_mut().unwrap();

        // Upload body states, props, and shape data to GPU.
        // AABBs are computed on the GPU via the AABB compute shader.
        pipeline.upload(
            &compact_states,
            &compact_props,
            &self.spheres,
            &self.boxes,
            self.config.gravity,
            self.config.dt,
            self.config.solver_iterations,
        );

        // Run GPU step
        let results = pipeline.step(num_bodies, self.config.solver_iterations);

        // Write results back to original arrays
        for (slot, &orig) in alive_indices.iter().enumerate() {
            if slot < results.len() {
                self.states[orig] = results[slot];
            }
        }
    }

    /// CPU-only simulation step (original path).
    fn step_cpu(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Collect indices of alive bodies.
        let alive_indices: Vec<usize> = (0..n).filter(|&i| self.alive[i]).collect();
        if alive_indices.is_empty() {
            return;
        }

        // Build mapping from original index -> compact slot for the solver.
        let mut index_to_slot: Vec<usize> = vec![usize::MAX; n];
        for (slot, &orig) in alive_indices.iter().enumerate() {
            index_to_slot[orig] = slot;
        }

        // Build compact arrays for the solver.
        // We append one extra "virtual static body" at the end for plane contacts.
        let mut compact_states: Vec<RigidBodyState3D> =
            alive_indices.iter().map(|&i| self.states[i]).collect();
        let mut compact_inv_inertias: Vec<Mat3> = alive_indices
            .iter()
            .map(|&i| self.inv_inertias[i])
            .collect();

        // Virtual static body for plane contacts (inv_mass = 0 means static).
        let plane_body_slot = compact_states.len() as u32;
        compact_states.push(RigidBodyState3D::new(
            Vec3::ZERO,
            0.0, // inv_mass = 0 => static
            Quat::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
        ));
        compact_inv_inertias.push(Mat3::ZERO);

        // 1. Compute AABBs for all alive bodies (in compact indexing).
        let aabbs: Vec<Aabb3D> = alive_indices
            .iter()
            .map(|&i| {
                let pos = self.states[i].position();
                let rot = self.states[i].quat();
                match &self.shapes[i] {
                    ShapeDesc::Sphere { radius } => compute_sphere_aabb(pos, *radius),
                    ShapeDesc::Box { half_extents } => compute_box_aabb(pos, rot, *half_extents),
                    ShapeDesc::Capsule {
                        half_height,
                        radius,
                    } => compute_capsule_aabb(pos, rot, *half_height, *radius),
                }
            })
            .collect();

        // 2. Build LBVH and find broad-phase pairs (slot indices).
        let lbvh = Lbvh::build(&aabbs);
        let broad_result = lbvh.find_overlapping_pairs(&aabbs);

        // 3. Find plane pairs (slot indices).
        let plane_pairs = find_plane_pairs(&self.planes, &aabbs);

        // 4. Generate narrow-phase contacts for body-body pairs.
        let mut all_contacts: Vec<Contact3D> = Vec::new();

        for &[slot_a, slot_b] in &broad_result.pairs {
            let orig_a = alive_indices[slot_a as usize];
            let orig_b = alive_indices[slot_b as usize];
            let contacts = self.generate_pair_contacts(orig_a, orig_b, slot_a, slot_b);
            all_contacts.extend(contacts);
        }

        // 5. Generate narrow-phase contacts for plane-body pairs.
        for &(plane_idx, slot_idx) in &plane_pairs {
            let orig = alive_indices[slot_idx as usize];
            // Skip static bodies for plane contacts.
            if self.states[orig].inv_mass() <= 0.0 {
                continue;
            }
            let plane = &self.planes[plane_idx];
            let pos = self.states[orig].position();
            let rot = self.states[orig].quat();

            // Use the virtual static body (plane_body_slot) as body_a,
            // and the dynamic body's slot as body_b. The solver sees
            // inv_mass=0 for the plane body, so only the dynamic body moves.
            let body_slot = slot_idx;
            let contacts = match &self.shapes[orig] {
                ShapeDesc::Sphere { radius } => plane_sphere(
                    plane.normal,
                    plane.distance,
                    pos,
                    *radius,
                    plane_body_slot,
                    body_slot,
                ),
                ShapeDesc::Box { half_extents } => {
                    let raw = plane_box(
                        plane.normal,
                        plane.distance,
                        pos,
                        rot,
                        *half_extents,
                        plane_body_slot,
                        body_slot,
                    );
                    reduce_manifold(&raw)
                }
                ShapeDesc::Capsule {
                    half_height,
                    radius,
                } => {
                    // Approximate capsule-plane as sphere-plane at endpoints + center.
                    let axis = rot * Vec3::Y;
                    let tip_lo = pos - axis * *half_height;
                    let tip_hi = pos + axis * *half_height;
                    let mut contacts = plane_sphere(
                        plane.normal,
                        plane.distance,
                        tip_lo,
                        *radius,
                        plane_body_slot,
                        body_slot,
                    );
                    contacts.extend(plane_sphere(
                        plane.normal,
                        plane.distance,
                        tip_hi,
                        *radius,
                        plane_body_slot,
                        body_slot,
                    ));
                    contacts.extend(plane_sphere(
                        plane.normal,
                        plane.distance,
                        pos,
                        *radius,
                        plane_body_slot,
                        body_slot,
                    ));
                    reduce_manifold(&contacts)
                }
            };
            all_contacts.extend(contacts);
        }

        // 6. Contact persistence (warm starting).
        let (warm_contacts, persistence_events) = self.persistence.update(all_contacts);
        let mut contacts = warm_contacts;

        // 7. Detect collision events via pair tracking.
        let curr_pairs: HashSet<(u32, u32)> = contacts
            .iter()
            .map(|c| (c.body_a.min(c.body_b), c.body_a.max(c.body_b)))
            .collect();

        // Emit Started/Ended events not already captured by persistence.
        // Persistence already handles this, so just forward those events.
        self.events.extend(persistence_events);

        self.prev_pairs = curr_pairs;

        // 8. Run solver.
        self.solver.solve(
            self.config.dt,
            self.config.gravity,
            &mut compact_states,
            &compact_inv_inertias,
            &mut contacts,
        );

        // Write compact states back to the original arrays.
        for (slot, &orig) in alive_indices.iter().enumerate() {
            self.states[orig] = compact_states[slot];
        }
    }

    /// Drain all collision events accumulated since the last drain.
    pub fn drain_events(&mut self) -> Vec<CollisionEvent> {
        std::mem::take(&mut self.events)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Generate narrow-phase contacts between two bodies.
    /// `orig_a`/`orig_b` are indices into self.shapes/states,
    /// `slot_a`/`slot_b` are compact indices used in the contact body_a/body_b fields.
    fn generate_pair_contacts(
        &self,
        orig_a: usize,
        orig_b: usize,
        slot_a: u32,
        slot_b: u32,
    ) -> Vec<Contact3D> {
        let pos_a = self.states[orig_a].position();
        let rot_a = self.states[orig_a].quat();
        let pos_b = self.states[orig_b].position();
        let rot_b = self.states[orig_b].quat();

        match (&self.shapes[orig_a], &self.shapes[orig_b]) {
            (ShapeDesc::Sphere { radius: ra }, ShapeDesc::Sphere { radius: rb }) => {
                sphere_sphere(pos_a, *ra, pos_b, *rb, slot_a, slot_b)
            }
            (ShapeDesc::Sphere { radius }, ShapeDesc::Box { half_extents }) => {
                sphere_box(pos_a, *radius, pos_b, rot_b, *half_extents, slot_a, slot_b)
            }
            (ShapeDesc::Box { half_extents }, ShapeDesc::Sphere { radius }) => {
                let mut contacts =
                    sphere_box(pos_b, *radius, pos_a, rot_a, *half_extents, slot_b, slot_a);
                for c in &mut contacts {
                    std::mem::swap(&mut c.body_a, &mut c.body_b);
                    c.normal = Vec4::new(-c.normal.x, -c.normal.y, -c.normal.z, 0.0);
                }
                contacts
            }
            (ShapeDesc::Box { half_extents: ha }, ShapeDesc::Box { half_extents: hb }) => {
                box_box(pos_a, rot_a, *ha, pos_b, rot_b, *hb, slot_a, slot_b)
            }
            (
                ShapeDesc::Sphere { radius },
                ShapeDesc::Capsule {
                    half_height,
                    radius: cr,
                },
            ) => sphere_capsule(
                pos_a,
                *radius,
                pos_b,
                rot_b,
                *half_height,
                *cr,
                slot_a,
                slot_b,
            ),
            (
                ShapeDesc::Capsule {
                    half_height,
                    radius: cr,
                },
                ShapeDesc::Sphere { radius },
            ) => {
                let mut contacts = sphere_capsule(
                    pos_b,
                    *radius,
                    pos_a,
                    rot_a,
                    *half_height,
                    *cr,
                    slot_b,
                    slot_a,
                );
                for c in &mut contacts {
                    std::mem::swap(&mut c.body_a, &mut c.body_b);
                    c.normal = Vec4::new(-c.normal.x, -c.normal.y, -c.normal.z, 0.0);
                }
                contacts
            }
            (
                ShapeDesc::Capsule {
                    half_height: hh_a,
                    radius: ra,
                },
                ShapeDesc::Capsule {
                    half_height: hh_b,
                    radius: rb,
                },
            ) => capsule_capsule(
                pos_a, rot_a, *hh_a, *ra, pos_b, rot_b, *hh_b, *rb, slot_a, slot_b,
            ),
            (
                ShapeDesc::Box { half_extents },
                ShapeDesc::Capsule {
                    half_height,
                    radius,
                },
            ) => {
                // Approximate box-capsule as sphere_box at capsule endpoints + center.
                let axis = rot_b * Vec3::Y;
                let tip_a = pos_b + axis * *half_height;
                let tip_b = pos_b - axis * *half_height;
                let mut contacts = Vec::new();
                contacts.extend(sphere_box(
                    tip_a,
                    *radius,
                    pos_a,
                    rot_a,
                    *half_extents,
                    slot_b,
                    slot_a,
                ));
                contacts.extend(sphere_box(
                    tip_b,
                    *radius,
                    pos_a,
                    rot_a,
                    *half_extents,
                    slot_b,
                    slot_a,
                ));
                contacts.extend(sphere_box(
                    pos_b,
                    *radius,
                    pos_a,
                    rot_a,
                    *half_extents,
                    slot_b,
                    slot_a,
                ));
                for c in &mut contacts {
                    std::mem::swap(&mut c.body_a, &mut c.body_b);
                    c.normal = Vec4::new(-c.normal.x, -c.normal.y, -c.normal.z, 0.0);
                }
                reduce_manifold(&contacts)
            }
            (
                ShapeDesc::Capsule {
                    half_height,
                    radius,
                },
                ShapeDesc::Box { half_extents },
            ) => {
                let axis = rot_a * Vec3::Y;
                let tip_a = pos_a + axis * *half_height;
                let tip_b = pos_a - axis * *half_height;
                let mut contacts = Vec::new();
                contacts.extend(sphere_box(
                    tip_a,
                    *radius,
                    pos_b,
                    rot_b,
                    *half_extents,
                    slot_a,
                    slot_b,
                ));
                contacts.extend(sphere_box(
                    tip_b,
                    *radius,
                    pos_b,
                    rot_b,
                    *half_extents,
                    slot_a,
                    slot_b,
                ));
                contacts.extend(sphere_box(
                    pos_a,
                    *radius,
                    pos_b,
                    rot_b,
                    *half_extents,
                    slot_a,
                    slot_b,
                ));
                reduce_manifold(&contacts)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_new() {
        let world = World::new(SimConfig::default());
        assert_eq!(world.body_count(), 0);
        assert!(world.planes.is_empty());
        assert!(world.events.is_empty());
    }

    #[test]
    fn test_add_remove_body() {
        let mut world = World::new(SimConfig::default());
        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::new(1.0, 2.0, 3.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        assert_eq!(world.body_count(), 1);
        assert_eq!(world.get_position(handle), Some(Vec3::new(1.0, 2.0, 3.0)));

        let removed = world.remove_body(handle);
        assert!(removed);
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.get_position(handle), None);

        // Double-remove should fail.
        assert!(!world.remove_body(handle));
    }

    #[test]
    fn test_handle_allocator() {
        let mut alloc = GenerationalIndexAllocator::new();

        let h1 = alloc.alloc();
        assert_eq!(h1.index, 0);
        assert_eq!(h1.generation, 0);
        assert!(alloc.is_valid(h1));

        let h2 = alloc.alloc();
        assert_eq!(h2.index, 1);
        assert_eq!(h2.generation, 0);
        assert!(alloc.is_valid(h2));

        // Dealloc h1.
        assert!(alloc.dealloc(h1));
        assert!(!alloc.is_valid(h1));

        // Re-alloc should reuse index 0 with bumped generation.
        let h3 = alloc.alloc();
        assert_eq!(h3.index, 0);
        assert_eq!(h3.generation, 1);
        assert!(alloc.is_valid(h3));

        // Old handle with generation 0 should be invalid.
        assert!(!alloc.is_valid(h1));
    }

    #[test]
    fn test_single_body_gravity() {
        let mut world = World::new(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 1024,
        });

        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        // Step ~60 times (approx 1 second).
        for _ in 0..60 {
            world.step();
        }

        let pos = world.get_position(handle).unwrap();
        // After 1 second of free-fall: y ~ 10 - 0.5 * 9.81 * 1^2 ~ 5.095
        assert!(pos.y < 10.0, "Body should have fallen: y = {}", pos.y);
        assert!(
            pos.y > 0.0,
            "Body should not have fallen too far: y = {}",
            pos.y
        );
        let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
        assert!(
            (pos.y - expected_y).abs() < 1.0,
            "Body y={} should be near expected y={} (tolerance 1.0)",
            pos.y,
            expected_y
        );
    }

    #[test]
    fn test_two_body_collision() {
        let mut world = World::new(SimConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
        });

        let h1 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(-2.0, 0.0, 0.0),
            linear_velocity: Vec3::new(5.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        let h2 = world.add_body(&RigidBodyDesc {
            position: Vec3::new(2.0, 0.0, 0.0),
            linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });

        // Step enough times for them to collide and resolve.
        for _ in 0..120 {
            world.step();
        }

        let p1 = world.get_position(h1).unwrap();
        let p2 = world.get_position(h2).unwrap();
        let dist = (p2 - p1).length();

        // Two spheres of radius 1.0 should not overlap significantly after collision.
        assert!(
            dist >= 1.5,
            "Bodies should not overlap significantly: distance = {}",
            dist
        );
    }

    #[test]
    fn test_body_count() {
        let mut world = World::new(SimConfig::default());

        let h1 = world.add_body(&RigidBodyDesc::default());
        let _h2 = world.add_body(&RigidBodyDesc::default());
        let h3 = world.add_body(&RigidBodyDesc::default());
        assert_eq!(world.body_count(), 3);

        world.remove_body(h1);
        assert_eq!(world.body_count(), 2);

        world.remove_body(h3);
        assert_eq!(world.body_count(), 1);
    }

    #[test]
    fn test_plane_collision() {
        let mut world = World::new(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
        });

        // Ground plane at y=0.
        world.add_plane(Vec3::Y, 0.0);

        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        // Run for several seconds of simulation time.
        for _ in 0..300 {
            world.step();
        }

        let pos = world.get_position(handle).unwrap();
        // The sphere (radius 0.5) should rest on the plane at approximately y >= 0.
        assert!(
            pos.y >= -0.5,
            "Sphere should not fall through the plane: y = {}",
            pos.y
        );
        assert!(
            pos.y < 5.0,
            "Sphere should have fallen from its starting position: y = {}",
            pos.y
        );
    }

    #[test]
    fn test_compute_inertia_sphere() {
        let shape = ShapeDesc::Sphere { radius: 1.0 };
        let inv_i = compute_inertia(&shape, 5.0);
        // I = 2/5 * 5 * 1 = 2.0 per diagonal, inv = 0.5
        let expected = 0.5;
        let cols = inv_i.to_cols_array_2d();
        assert!((cols[0][0] - expected).abs() < 1e-6);
        assert!((cols[1][1] - expected).abs() < 1e-6);
        assert!((cols[2][2] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_compute_inertia_static() {
        let shape = ShapeDesc::Sphere { radius: 1.0 };
        let inv_i = compute_inertia(&shape, 0.0);
        assert_eq!(inv_i, Mat3::ZERO);
    }

    #[test]
    fn test_set_position_and_velocity() {
        let mut world = World::new(SimConfig::default());
        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::ZERO,
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        world.set_position(handle, Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(
            world.get_position(handle),
            Some(Vec3::new(10.0, 20.0, 30.0))
        );

        world.set_velocity(handle, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(world.get_velocity(handle), Some(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn test_get_rotation() {
        let mut world = World::new(SimConfig::default());
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let handle = world.add_body(&RigidBodyDesc {
            rotation: rot,
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        let got = world.get_rotation(handle).unwrap();
        assert!((got.x - rot.x).abs() < 1e-6);
        assert!((got.y - rot.y).abs() < 1e-6);
        assert!((got.z - rot.z).abs() < 1e-6);
        assert!((got.w - rot.w).abs() < 1e-6);
    }
}

//! `rubble3d` — Public API facade for the rubble 3D GPU physics engine.
//!
//! All physics simulation runs entirely on the GPU via WGSL compute shaders.
//! The pipeline: predict → AABB → broadphase → narrowphase → AVBD solver → velocity extraction.

pub mod gpu;

use glam::{Mat3, Quat, Vec3, Vec4};
use rubble_math::{
    BodyHandle, CollisionEvent, RigidBodyProps3D, RigidBodyState3D, FLAG_STATIC, SHAPE_BOX,
    SHAPE_CAPSULE, SHAPE_CONVEX_HULL, SHAPE_PLANE, SHAPE_SPHERE,
};
use rubble_shapes3d::{BoxData, CapsuleData, ConvexHullData, ConvexVertex3D, SphereData};

// ---------------------------------------------------------------------------
// SimConfig
// ---------------------------------------------------------------------------

/// Top-level simulation configuration.
pub struct SimConfig {
    pub gravity: Vec3,
    pub dt: f32,
    pub solver_iterations: u32,
    pub max_bodies: usize,
    /// Augmented Lagrangian stiffness ramp rate.
    pub beta: f32,
    /// Initial penalty stiffness for contacts.
    pub k_start: f32,
    /// Decay factor for warm-started dual variables across steps.
    pub warmstart_decay: f32,
    /// Default friction coefficient for bodies without explicit friction.
    pub friction_default: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 65536,
            beta: 10.0,
            k_start: 1e4,
            warmstart_decay: 0.95,
            friction_default: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// ShapeDesc
// ---------------------------------------------------------------------------

/// Describes a collision shape for body creation.
#[derive(Debug, Clone)]
pub enum ShapeDesc {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vec3,
    },
    Capsule {
        half_height: f32,
        radius: f32,
    },
    /// Convex hull defined by up to 64 vertices in local space.
    ConvexHull {
        vertices: Vec<Vec3>,
    },
    /// Infinite plane (always static). Normal points into the free half-space.
    Plane {
        normal: Vec3,
        distance: f32,
    },
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
            let i = (2.0 / 5.0) * mass * radius * radius;
            Vec3::splat(i)
        }
        ShapeDesc::Box { half_extents } => {
            let w = 2.0 * half_extents.x;
            let h = 2.0 * half_extents.y;
            let d = 2.0 * half_extents.z;
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
            // Capsule inertia = cylinder + 2 hemispheres
            let h = 2.0 * half_height;
            let r2 = radius * radius;
            let cyl_mass = mass * h / (h + (4.0 / 3.0) * radius);
            let cap_mass = mass - cyl_mass;
            let iy = cyl_mass * r2 / 2.0 + cap_mass * 2.0 * r2 / 5.0;
            let ix = cyl_mass * (3.0 * r2 + h * h) / 12.0
                + cap_mass * (2.0 * r2 / 5.0 + h * h / 4.0 + 3.0 * h * radius / 8.0);
            Vec3::new(ix, iy, ix)
        }
        ShapeDesc::ConvexHull { vertices } => {
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for &v in vertices {
                min = min.min(v);
                max = max.max(v);
            }
            let size = max - min;
            Vec3::new(
                mass / 12.0 * (size.y * size.y + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.y * size.y),
            )
        }
        ShapeDesc::Plane { .. } => {
            // Planes are always static, so this shouldn't be called with mass > 0
            Vec3::splat(1.0)
        }
    };
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

/// The main 3D physics world. All simulation runs on the GPU.
pub struct World {
    config: SimConfig,
    states: Vec<RigidBodyState3D>,
    props: Vec<RigidBodyProps3D>,
    shapes: Vec<ShapeDesc>,
    spheres: Vec<SphereData>,
    boxes: Vec<BoxData>,
    capsules: Vec<CapsuleData>,
    convex_hulls: Vec<ConvexHullData>,
    convex_vertices: Vec<ConvexVertex3D>,
    /// Plane data stored as Vec4(nx, ny, nz, distance)
    planes: Vec<Vec4>,
    allocator: GenerationalIndexAllocator,
    alive: Vec<bool>,
    gpu_pipeline: gpu::GpuPipeline,
    contact_persistence: gpu::ContactPersistence3D,
    collision_events: Vec<CollisionEvent>,
}

impl World {
    /// Create a new 3D physics world backed by GPU compute shaders.
    ///
    /// Returns an error if no GPU adapter is available.
    pub fn new(config: SimConfig) -> Result<Self, rubble_gpu::GpuError> {
        let ctx = pollster::block_on(rubble_gpu::GpuContext::new())?;
        let pipeline = gpu::GpuPipeline::new(ctx, config.max_bodies);
        Ok(Self {
            config,
            states: Vec::new(),
            props: Vec::new(),
            shapes: Vec::new(),
            spheres: Vec::new(),
            boxes: Vec::new(),
            capsules: Vec::new(),
            convex_hulls: Vec::new(),
            convex_vertices: Vec::new(),
            planes: Vec::new(),
            allocator: GenerationalIndexAllocator::new(),
            alive: Vec::new(),
            gpu_pipeline: pipeline,
            contact_persistence: gpu::ContactPersistence3D::new(),
            collision_events: Vec::new(),
        })
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
            ShapeDesc::ConvexHull { vertices } => {
                let si = self.convex_hulls.len() as u32;
                let vertex_offset = self.convex_vertices.len() as u32;
                let vertex_count = vertices.len().min(64) as u32;
                for v in vertices.iter().take(64) {
                    self.convex_vertices.push(ConvexVertex3D {
                        x: v.x,
                        y: v.y,
                        z: v.z,
                        _pad: 0.0,
                    });
                }
                self.convex_hulls.push(ConvexHullData {
                    vertex_offset,
                    vertex_count,
                    face_offset: 0,
                    face_count: 0,
                    edge_offset: 0,
                    edge_count: 0,
                    gauss_map_offset: 0,
                    gauss_map_count: 0,
                });
                (SHAPE_CONVEX_HULL, si)
            }
            ShapeDesc::Plane { normal, distance } => {
                let si = self.planes.len() as u32;
                self.planes
                    .push(Vec4::new(normal.x, normal.y, normal.z, *distance));
                (SHAPE_PLANE, si)
            }
        };

        let prop =
            RigidBodyProps3D::new(inv_inertia, desc.friction, shape_type, shape_index, flags);

        if idx >= self.states.len() {
            self.states.resize(idx + 1, bytemuck::Zeroable::zeroed());
            self.props.resize(idx + 1, bytemuck::Zeroable::zeroed());
            self.shapes
                .resize(idx + 1, ShapeDesc::Sphere { radius: 0.5 });
            self.alive.resize(idx + 1, false);
        }

        self.states[idx] = state;
        self.props[idx] = prop;
        self.shapes[idx] = desc.shape.clone();
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

    /// Advance the simulation by one time step on the GPU.
    pub fn step(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        let alive_indices: Vec<usize> = (0..n).filter(|&i| self.alive[i]).collect();
        if alive_indices.is_empty() {
            return;
        }

        let compact_states: Vec<RigidBodyState3D> =
            alive_indices.iter().map(|&i| self.states[i]).collect();
        let compact_props: Vec<RigidBodyProps3D> =
            alive_indices.iter().map(|&i| self.props[i]).collect();
        let num_bodies = compact_states.len() as u32;

        self.gpu_pipeline.upload(
            &compact_states,
            &compact_props,
            &self.spheres,
            &self.boxes,
            &self.capsules,
            &self.convex_hulls,
            &self.convex_vertices,
            &self.planes,
            self.config.gravity,
            self.config.dt,
            self.config.solver_iterations,
        );

        let prev = self.contact_persistence.prev_contacts();
        let warm = if prev.is_empty() { None } else { Some(prev) };
        let (results, new_contacts) =
            self.gpu_pipeline
                .step_with_contacts(num_bodies, self.config.solver_iterations, warm);

        // Update persistence and generate collision events
        let events = self.contact_persistence.update(&new_contacts);
        self.collision_events.extend(events);

        for (slot, &orig) in alive_indices.iter().enumerate() {
            if slot < results.len() {
                self.states[orig] = results[slot];
            }
        }
    }

    /// Mark a body as kinematic (moves via set_position/set_velocity, not physics).
    pub fn set_body_kinematic(&mut self, handle: BodyHandle, kinematic: bool) {
        if !self.allocator.is_valid(handle) {
            return;
        }
        let idx = handle.index as usize;
        if kinematic {
            self.props[idx].flags |= rubble_math::FLAG_KINEMATIC;
        } else {
            self.props[idx].flags &= !rubble_math::FLAG_KINEMATIC;
        }
    }

    /// Drain collision events from the last step.
    pub fn drain_collision_events(&mut self) -> Vec<CollisionEvent> {
        std::mem::take(&mut self.collision_events)
    }

    /// Cast a ray and return the closest hit (handle, t parameter, hit normal).
    /// `origin` is the ray start, `direction` is the ray direction (need not be normalized).
    /// `max_t` is the maximum ray parameter to test.
    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_t: f32,
    ) -> Option<(BodyHandle, f32, Vec3)> {
        let dir_len = direction.length();
        if dir_len < 1e-12 {
            return None;
        }
        let dir = direction / dir_len;

        let mut best_t = max_t * dir_len;
        let mut best_handle = None;
        let mut best_normal = Vec3::ZERO;

        for (idx, alive) in self.alive.iter().enumerate() {
            if !*alive {
                continue;
            }
            let state = &self.states[idx];
            let pos = state.position();

            if let Some((t, normal)) = self.ray_shape_test(origin, dir, pos, state, idx) {
                if t >= 0.0 && t < best_t {
                    best_t = t;
                    let gen = self.allocator.generations[idx];
                    best_handle = Some(BodyHandle::new(idx as u32, gen));
                    best_normal = normal;
                }
            }
        }

        best_handle.map(|h| (h, best_t / dir_len, best_normal))
    }

    fn ray_shape_test(
        &self,
        origin: Vec3,
        dir: Vec3,
        pos: Vec3,
        _state: &RigidBodyState3D,
        idx: usize,
    ) -> Option<(f32, Vec3)> {
        match &self.shapes[idx] {
            ShapeDesc::Sphere { radius } => {
                // Ray-sphere intersection
                let oc = origin - pos;
                let b = oc.dot(dir);
                let c = oc.dot(oc) - radius * radius;
                let disc = b * b - c;
                if disc < 0.0 {
                    return None;
                }
                let t = -b - disc.sqrt();
                if t < 0.0 {
                    return None;
                }
                let hit = origin + dir * t;
                let normal = (hit - pos).normalize();
                Some((t, normal))
            }
            ShapeDesc::Box { half_extents } => {
                // Ray-AABB intersection (axis-aligned, ignoring rotation for simplicity)
                let q = _state.quat();
                let inv_q = q.conjugate();
                let local_origin = inv_q * (origin - pos);
                let local_dir = inv_q * dir;
                let he = *half_extents;

                let mut tmin = f32::NEG_INFINITY;
                let mut tmax = f32::INFINITY;
                let mut normal_idx = 0usize;
                let mut normal_sign = 1.0f32;

                for i in 0..3 {
                    let o = [local_origin.x, local_origin.y, local_origin.z][i];
                    let d = [local_dir.x, local_dir.y, local_dir.z][i];
                    let h = [he.x, he.y, he.z][i];

                    if d.abs() < 1e-12 {
                        if o < -h || o > h {
                            return None;
                        }
                        continue;
                    }
                    let t1 = (-h - o) / d;
                    let t2 = (h - o) / d;
                    let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                    if t_near > tmin {
                        tmin = t_near;
                        normal_idx = i;
                        normal_sign = if d > 0.0 { -1.0 } else { 1.0 };
                    }
                    tmax = tmax.min(t_far);
                    if tmin > tmax {
                        return None;
                    }
                }

                if tmin < 0.0 {
                    return None;
                }
                let mut local_normal = Vec3::ZERO;
                match normal_idx {
                    0 => local_normal.x = normal_sign,
                    1 => local_normal.y = normal_sign,
                    _ => local_normal.z = normal_sign,
                }
                let world_normal = q * local_normal;
                Some((tmin, world_normal))
            }
            ShapeDesc::Plane { normal, distance } => {
                let denom = normal.dot(dir);
                if denom.abs() < 1e-12 {
                    return None;
                }
                let t = (*distance - normal.dot(origin)) / denom;
                if t >= 0.0 {
                    Some((t, *normal))
                } else {
                    None
                }
            }
            _ => None, // Capsule and ConvexHull raycast: could be added later
        }
    }

    /// Query all bodies whose AABB overlaps the given axis-aligned bounding box.
    pub fn overlap_aabb(&self, query_min: Vec3, query_max: Vec3) -> Vec<BodyHandle> {
        let mut result = Vec::new();
        for (idx, alive) in self.alive.iter().enumerate() {
            if !*alive {
                continue;
            }
            let state = &self.states[idx];
            let pos = state.position();
            let (body_min, body_max) = self.compute_aabb(pos, state, idx);

            // AABB overlap test
            if body_min.x <= query_max.x
                && body_max.x >= query_min.x
                && body_min.y <= query_max.y
                && body_max.y >= query_min.y
                && body_min.z <= query_max.z
                && body_max.z >= query_min.z
            {
                let gen = self.allocator.generations[idx];
                result.push(BodyHandle::new(idx as u32, gen));
            }
        }
        result
    }

    fn compute_aabb(&self, pos: Vec3, state: &RigidBodyState3D, idx: usize) -> (Vec3, Vec3) {
        match &self.shapes[idx] {
            ShapeDesc::Sphere { radius } => {
                let r = Vec3::splat(*radius);
                (pos - r, pos + r)
            }
            ShapeDesc::Box { half_extents } => {
                let q = state.quat();
                let he = *half_extents;
                let ax = (q * Vec3::new(he.x, 0.0, 0.0)).abs();
                let ay = (q * Vec3::new(0.0, he.y, 0.0)).abs();
                let az = (q * Vec3::new(0.0, 0.0, he.z)).abs();
                let extent = ax + ay + az;
                (pos - extent, pos + extent)
            }
            ShapeDesc::Capsule {
                half_height,
                radius,
            } => {
                let q = state.quat();
                let axis = q * Vec3::new(0.0, *half_height, 0.0);
                let a = pos + axis;
                let b = pos - axis;
                let r = Vec3::splat(*radius);
                (a.min(b) - r, a.max(b) + r)
            }
            ShapeDesc::ConvexHull { vertices } => {
                let q = state.quat();
                let mut mn = Vec3::splat(f32::MAX);
                let mut mx = Vec3::splat(f32::NEG_INFINITY);
                for v in vertices {
                    let wv = pos + q * *v;
                    mn = mn.min(wv);
                    mx = mx.max(wv);
                }
                (mn, mx)
            }
            ShapeDesc::Plane { normal, distance } => {
                let center = *normal * *distance;
                let big = Vec3::splat(1e4);
                (center - big, center + big)
            }
        }
    }

    /// Access the GPU pipeline (for diagnostics like contact count).
    pub fn gpu_pipeline(&self) -> &gpu::GpuPipeline {
        &self.gpu_pipeline
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_world(config: SimConfig) -> World {
        World::new(config).expect("GPU required for tests")
    }

    #[test]
    fn test_world_new() {
        let world = gpu_world(SimConfig::default());
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_add_remove_body() {
        let mut world = gpu_world(SimConfig::default());
        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::new(1.0, 2.0, 3.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        assert_eq!(world.body_count(), 1);
        assert_eq!(world.get_position(handle), Some(Vec3::new(1.0, 2.0, 3.0)));

        assert!(world.remove_body(handle));
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.get_position(handle), None);
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
        assert!(alloc.is_valid(h2));

        assert!(alloc.dealloc(h1));
        assert!(!alloc.is_valid(h1));

        let h3 = alloc.alloc();
        assert_eq!(h3.index, 0);
        assert_eq!(h3.generation, 1);
        assert!(alloc.is_valid(h3));
        assert!(!alloc.is_valid(h1));
    }

    #[test]
    fn test_single_body_gravity() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 1024,
            ..Default::default()
        });

        let handle = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        for _ in 0..60 {
            world.step();
        }

        let pos = world.get_position(handle).unwrap();
        let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
        assert!(
            (pos.y - expected_y).abs() < 0.5,
            "Body y={} should be near expected y={}",
            pos.y,
            expected_y
        );
    }

    #[test]
    fn test_two_body_collision() {
        let mut world = gpu_world(SimConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
            ..Default::default()
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

        for _ in 0..120 {
            world.step();
        }

        let p1 = world.get_position(h1).unwrap();
        let p2 = world.get_position(h2).unwrap();
        let dist = (p2 - p1).length();
        assert!(
            dist >= 1.5,
            "Bodies should not overlap: distance = {}",
            dist
        );
    }

    #[test]
    fn test_body_count() {
        let mut world = gpu_world(SimConfig::default());

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
    fn test_compute_inertia_sphere() {
        let shape = ShapeDesc::Sphere { radius: 1.0 };
        let inv_i = compute_inertia(&shape, 5.0);
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
        let mut world = gpu_world(SimConfig::default());
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
        let mut world = gpu_world(SimConfig::default());
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

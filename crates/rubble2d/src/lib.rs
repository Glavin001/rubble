//! `rubble2d` — Public API facade for the rubble 2D GPU physics engine.
//!
//! All physics simulation runs entirely on the GPU via WGSL compute shaders.
//! The pipeline: predict → AABB → broadphase → narrowphase → AVBD solver → velocity extraction.

pub mod gpu;

use glam::{Vec2, Vec4};
use rubble_math::{BodyHandle, CollisionEvent, RigidBodyState2D};
use rubble_shapes2d::{CapsuleData2D, CircleData, ConvexPolygonData, ConvexVertex2D, RectData};

// ---------------------------------------------------------------------------
// SimConfig2D
// ---------------------------------------------------------------------------

/// Top-level simulation configuration.
pub struct SimConfig2D {
    pub gravity: Vec2,
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

impl Default for SimConfig2D {
    fn default() -> Self {
        Self {
            gravity: Vec2::new(0.0, -9.81),
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
// ShapeDesc2D
// ---------------------------------------------------------------------------

/// User-facing shape descriptor for adding bodies.
#[derive(Debug, Clone)]
pub enum ShapeDesc2D {
    Circle {
        radius: f32,
    },
    Rect {
        half_extents: Vec2,
    },
    /// Convex polygon defined by up to 64 vertices in local space (CCW winding).
    ConvexPolygon {
        vertices: Vec<Vec2>,
    },
    /// 2D capsule: a line segment with radius (oriented along local Y axis).
    Capsule {
        half_height: f32,
        radius: f32,
    },
}

// ---------------------------------------------------------------------------
// RigidBodyDesc2D
// ---------------------------------------------------------------------------

/// User-facing descriptor for creating a rigid body.
pub struct RigidBodyDesc2D {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub vx: f32,
    pub vy: f32,
    pub angular_velocity: f32,
    pub mass: f32,
    pub friction: f32,
    pub shape: ShapeDesc2D,
}

impl Default for RigidBodyDesc2D {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            angle: 0.0,
            vx: 0.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 1.0,
            friction: 0.5,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
        }
    }
}

// ---------------------------------------------------------------------------
// GenerationalIndexAllocator
// ---------------------------------------------------------------------------

/// Generational index allocator for stable body handles.
struct GenerationalIndexAllocator {
    generations: Vec<u32>,
    alive: Vec<bool>,
    free_list: Vec<u32>,
}

impl GenerationalIndexAllocator {
    fn new() -> Self {
        Self {
            generations: Vec::new(),
            alive: Vec::new(),
            free_list: Vec::new(),
        }
    }

    fn allocate(&mut self) -> BodyHandle {
        if let Some(index) = self.free_list.pop() {
            let i = index as usize;
            self.alive[i] = true;
            BodyHandle::new(index, self.generations[i])
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            self.alive.push(true);
            BodyHandle::new(index, 0)
        }
    }

    fn deallocate(&mut self, handle: BodyHandle) -> bool {
        let i = handle.index as usize;
        if i >= self.alive.len() || !self.alive[i] || self.generations[i] != handle.generation {
            return false;
        }
        self.alive[i] = false;
        self.generations[i] += 1;
        self.free_list.push(handle.index);
        true
    }

    fn is_alive(&self, handle: BodyHandle) -> bool {
        let i = handle.index as usize;
        i < self.alive.len() && self.alive[i] && self.generations[i] == handle.generation
    }

    fn live_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }
}

// ---------------------------------------------------------------------------
// 2D raycast helpers
// ---------------------------------------------------------------------------

/// Ray-capsule intersection in 2D. Capsule = segment(ep_a, ep_b) + radius.
fn ray_capsule_2d(
    origin: Vec2,
    dir: Vec2,
    ep_a: Vec2,
    ep_b: Vec2,
    radius: f32,
) -> Option<(f32, Vec2)> {
    // Test both endpoint circles and the rectangle between them
    let mut best: Option<(f32, Vec2)> = None;

    // Test circle at ep_a
    if let Some((t, n)) = ray_circle_2d(origin, dir, ep_a, radius) {
        if t >= 0.0 && best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, n));
        }
    }
    // Test circle at ep_b
    if let Some((t, n)) = ray_circle_2d(origin, dir, ep_b, radius) {
        if t >= 0.0 && best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, n));
        }
    }

    // Test the rectangle (slab) between endpoints
    let seg = ep_b - ep_a;
    let seg_len = seg.length();
    if seg_len > 1e-8 {
        let seg_dir = seg / seg_len;
        let seg_normal = Vec2::new(-seg_dir.y, seg_dir.x);

        // Two planes at distance ±radius from the segment line
        let d = seg_normal.dot(ep_a);
        for &sign in &[1.0f32, -1.0] {
            let plane_n = seg_normal * sign;
            let plane_d = d * sign + radius;
            let denom = plane_n.dot(dir);
            if denom.abs() < 1e-12 {
                continue;
            }
            let t = (plane_d - plane_n.dot(origin)) / denom;
            if t < 0.0 {
                continue;
            }
            // Check that hit point is between ep_a and ep_b along segment
            let hit = origin + dir * t;
            let proj = seg_dir.dot(hit - ep_a);
            if proj >= 0.0 && proj <= seg_len && best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, plane_n));
            }
        }
    }

    best
}

/// Ray-circle intersection in 2D.
fn ray_circle_2d(origin: Vec2, dir: Vec2, center: Vec2, radius: f32) -> Option<(f32, Vec2)> {
    let oc = origin - center;
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
    let normal = (hit - center).normalize();
    Some((t, normal))
}

/// Ray-convex polygon intersection in 2D using slab method on edges.
fn ray_convex_polygon_2d(origin: Vec2, dir: Vec2, verts: &[Vec2]) -> Option<(f32, Vec2)> {
    let n = verts.len();
    if n < 3 {
        return None;
    }

    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    let mut best_normal = Vec2::ZERO;

    for i in 0..n {
        let j = (i + 1) % n;
        let edge = verts[j] - verts[i];
        // Outward normal (for CCW winding)
        let normal = Vec2::new(edge.y, -edge.x);
        let normal_len = normal.length();
        if normal_len < 1e-12 {
            continue;
        }
        let normal = normal / normal_len;

        let denom = normal.dot(dir);
        let dist = normal.dot(verts[i] - origin);

        if denom.abs() < 1e-12 {
            // Ray parallel to edge
            if dist < 0.0 {
                return None; // Outside this edge
            }
            continue;
        }

        let t = dist / denom;
        if denom < 0.0 {
            // Entering
            if t > tmin {
                tmin = t;
                best_normal = normal;
            }
        } else {
            // Leaving
            tmax = tmax.min(t);
        }

        if tmin > tmax {
            return None;
        }
    }

    if tmin >= 0.0 && tmin <= tmax {
        Some((tmin, best_normal))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// World2D
// ---------------------------------------------------------------------------

/// The main 2D physics world. All simulation runs on the GPU.
pub struct World2D {
    config: SimConfig2D,
    states: Vec<RigidBodyState2D>,
    shapes: Vec<ShapeDesc2D>,
    frictions: Vec<f32>,
    flags: Vec<u32>,
    allocator: GenerationalIndexAllocator,
    gpu_pipeline: gpu::GpuPipeline2D,
    contact_persistence: gpu::ContactPersistence2D,
    collision_events: Vec<CollisionEvent>,
}

impl World2D {
    /// Create a new 2D physics world backed by GPU compute shaders (async).
    ///
    /// Use this constructor in WASM/browser environments where blocking is not allowed.
    pub async fn new_async(config: SimConfig2D) -> Result<Self, rubble_gpu::GpuError> {
        let ctx = rubble_gpu::GpuContext::new().await?;
        let pipeline = gpu::GpuPipeline2D::new(ctx, config.max_bodies);
        Ok(Self {
            config,
            states: Vec::new(),
            shapes: Vec::new(),
            frictions: Vec::new(),
            flags: Vec::new(),
            allocator: GenerationalIndexAllocator::new(),
            gpu_pipeline: pipeline,
            contact_persistence: gpu::ContactPersistence2D::new(),
            collision_events: Vec::new(),
        })
    }

    /// Create a new 2D physics world backed by GPU compute shaders.
    ///
    /// Returns an error if no GPU adapter is available.
    /// Not available on WASM targets — use [`new_async`](Self::new_async) instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(config: SimConfig2D) -> Result<Self, rubble_gpu::GpuError> {
        pollster::block_on(Self::new_async(config))
    }

    /// Add a rigid body to the world. Returns a stable handle.
    pub fn add_body(&mut self, desc: &RigidBodyDesc2D) -> BodyHandle {
        let handle = self.allocator.allocate();
        let inv_mass = if desc.mass <= 0.0 {
            0.0
        } else {
            1.0 / desc.mass
        };

        let state = RigidBodyState2D::new(
            desc.x,
            desc.y,
            desc.angle,
            inv_mass,
            desc.vx,
            desc.vy,
            desc.angular_velocity,
        );

        let idx = handle.index as usize;

        let body_flags = if desc.mass <= 0.0 {
            rubble_math::FLAG_STATIC
        } else {
            0u32
        };

        if idx >= self.states.len() {
            self.states.resize(
                idx + 1,
                RigidBodyState2D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            );
            self.shapes
                .resize(idx + 1, ShapeDesc2D::Circle { radius: 0.0 });
            self.frictions.resize(idx + 1, 0.5);
            self.flags.resize(idx + 1, 0);
        }

        self.states[idx] = state;
        self.shapes[idx] = desc.shape.clone();
        self.frictions[idx] = desc.friction;
        self.flags[idx] = body_flags;

        handle
    }

    /// Remove a body by handle. Returns `true` if the body existed and was removed.
    pub fn remove_body(&mut self, handle: BodyHandle) -> bool {
        if !self.allocator.is_alive(handle) {
            return false;
        }
        let idx = handle.index as usize;
        self.states[idx] = RigidBodyState2D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        self.shapes[idx] = ShapeDesc2D::Circle { radius: 0.0 };
        self.frictions[idx] = 0.0;
        self.flags[idx] = 0;
        self.allocator.deallocate(handle)
    }

    /// Number of live bodies in the world.
    pub fn body_count(&self) -> usize {
        self.allocator.live_count()
    }

    /// Get the position of a body, or `None` if the handle is stale/invalid.
    pub fn get_position(&self, handle: BodyHandle) -> Option<Vec2> {
        if !self.allocator.is_alive(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].position())
    }

    /// Get the rotation angle (radians) of a body.
    pub fn get_angle(&self, handle: BodyHandle) -> Option<f32> {
        if !self.allocator.is_alive(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].angle())
    }

    /// Get the linear velocity of a body.
    pub fn get_velocity(&self, handle: BodyHandle) -> Option<Vec2> {
        if !self.allocator.is_alive(handle) {
            return None;
        }
        Some(self.states[handle.index as usize].linear_velocity())
    }

    /// Set the position of a body (teleport).
    pub fn set_position(&mut self, handle: BodyHandle, pos: Vec2) {
        if !self.allocator.is_alive(handle) {
            return;
        }
        let idx = handle.index as usize;
        let s = &mut self.states[idx];
        let im = s.inv_mass();
        let angle = s.angle();
        let vel = s.linear_velocity();
        let omega = s.angular_velocity();
        *s = RigidBodyState2D::new(pos.x, pos.y, angle, im, vel.x, vel.y, omega);
    }

    /// Set the linear velocity of a body.
    pub fn set_velocity(&mut self, handle: BodyHandle, vel: Vec2) {
        if !self.allocator.is_alive(handle) {
            return;
        }
        let idx = handle.index as usize;
        let s = &mut self.states[idx];
        let omega = s.angular_velocity();
        s.lin_vel = Vec4::new(vel.x, vel.y, omega, 0.0);
    }

    /// Set the angular velocity of a body.
    pub fn set_angular_velocity(&mut self, handle: BodyHandle, omega: f32) {
        if !self.allocator.is_alive(handle) {
            return;
        }
        let idx = handle.index as usize;
        let s = &mut self.states[idx];
        let vel = s.linear_velocity();
        s.lin_vel = Vec4::new(vel.x, vel.y, omega, 0.0);
    }

    /// Mark a body as kinematic (moves via set_position/set_velocity, not physics).
    pub fn set_body_kinematic(&mut self, handle: BodyHandle, kinematic: bool) {
        if !self.allocator.is_alive(handle) {
            return;
        }
        let idx = handle.index as usize;
        if kinematic {
            self.flags[idx] |= rubble_math::FLAG_KINEMATIC;
            // Kinematic bodies have zero inverse mass for solver
            let s = &mut self.states[idx];
            let pos = s.position();
            let angle = s.angle();
            let vel = s.linear_velocity();
            let omega = s.angular_velocity();
            *s = RigidBodyState2D::new(pos.x, pos.y, angle, 0.0, vel.x, vel.y, omega);
        } else {
            self.flags[idx] &= !rubble_math::FLAG_KINEMATIC;
        }
    }

    /// Cast multiple rays and return results for each.
    pub fn raycast_batch(
        &self,
        rays: &[(Vec2, Vec2, f32)], // (origin, direction, max_t)
    ) -> Vec<Option<(BodyHandle, f32, Vec2)>> {
        rays.iter()
            .map(|&(origin, dir, max_t)| self.raycast(origin, dir, max_t))
            .collect()
    }

    /// Advance the simulation by one time step on the GPU.
    pub fn step(&mut self) {
        if self.states.is_empty() {
            return;
        }

        let alive_indices: Vec<usize> = (0..self.states.len())
            .filter(|&i| self.allocator.alive.get(i).copied().unwrap_or(false))
            .collect();

        if alive_indices.is_empty() {
            return;
        }

        // Build compact arrays for GPU upload.
        let compact_states: Vec<RigidBodyState2D> = alive_indices
            .iter()
            .map(|&i| {
                let mut s = self.states[i];
                // Pack friction into _pad0.x for the GPU solver
                s._pad0 = Vec4::new(self.frictions[i], 0.0, 0.0, 0.0);
                s
            })
            .collect();

        let mut shape_info_data: Vec<gpu::ShapeInfo> = Vec::with_capacity(alive_indices.len());
        let mut gpu_circles: Vec<CircleData> = Vec::new();
        let mut gpu_rects: Vec<RectData> = Vec::new();
        let mut gpu_convex_polys: Vec<ConvexPolygonData> = Vec::new();
        let mut gpu_convex_verts: Vec<ConvexVertex2D> = Vec::new();
        let mut gpu_capsules: Vec<CapsuleData2D> = Vec::new();

        for &i in &alive_indices {
            match &self.shapes[i] {
                ShapeDesc2D::Circle { radius } => {
                    shape_info_data.push(gpu::ShapeInfo {
                        shape_type: 0, // SHAPE_CIRCLE
                        shape_index: gpu_circles.len() as u32,
                    });
                    gpu_circles.push(CircleData {
                        radius: *radius,
                        _pad: [0.0; 3],
                    });
                }
                ShapeDesc2D::Rect { half_extents } => {
                    shape_info_data.push(gpu::ShapeInfo {
                        shape_type: 1, // SHAPE_RECT
                        shape_index: gpu_rects.len() as u32,
                    });
                    gpu_rects.push(RectData {
                        half_extents: Vec4::new(half_extents.x, half_extents.y, 0.0, 0.0),
                    });
                }
                ShapeDesc2D::ConvexPolygon { vertices } => {
                    shape_info_data.push(gpu::ShapeInfo {
                        shape_type: 2, // SHAPE_CONVEX_POLYGON
                        shape_index: gpu_convex_polys.len() as u32,
                    });
                    let vertex_offset = gpu_convex_verts.len() as u32;
                    let vertex_count = vertices.len().min(64) as u32;
                    for v in vertices.iter().take(64) {
                        gpu_convex_verts.push(ConvexVertex2D {
                            x: v.x,
                            y: v.y,
                            _pad: [0.0; 2],
                        });
                    }
                    gpu_convex_polys.push(ConvexPolygonData {
                        vertex_offset,
                        vertex_count,
                        _pad: [0; 2],
                    });
                }
                ShapeDesc2D::Capsule {
                    half_height,
                    radius,
                } => {
                    shape_info_data.push(gpu::ShapeInfo {
                        shape_type: 3, // SHAPE_CAPSULE
                        shape_index: gpu_capsules.len() as u32,
                    });
                    gpu_capsules.push(CapsuleData2D {
                        half_height: *half_height,
                        radius: *radius,
                        _pad: [0.0; 2],
                    });
                }
            }
        }

        // Ensure at least one element in shape buffers so GPU buffers are valid.
        if gpu_circles.is_empty() {
            gpu_circles.push(CircleData {
                radius: 0.0,
                _pad: [0.0; 3],
            });
        }
        if gpu_rects.is_empty() {
            gpu_rects.push(RectData {
                half_extents: Vec4::ZERO,
            });
        }
        if gpu_convex_polys.is_empty() {
            gpu_convex_polys.push(ConvexPolygonData {
                vertex_offset: 0,
                vertex_count: 0,
                _pad: [0; 2],
            });
        }
        if gpu_convex_verts.is_empty() {
            gpu_convex_verts.push(ConvexVertex2D {
                x: 0.0,
                y: 0.0,
                _pad: [0.0; 2],
            });
        }
        if gpu_capsules.is_empty() {
            gpu_capsules.push(CapsuleData2D {
                half_height: 0.0,
                radius: 0.0,
                _pad: [0.0; 2],
            });
        }

        let num_bodies = compact_states.len() as u32;

        self.gpu_pipeline.upload(
            &compact_states,
            &shape_info_data,
            &gpu_circles,
            &gpu_rects,
            &gpu_convex_polys,
            &gpu_convex_verts,
            &gpu_capsules,
            self.config.gravity,
            self.config.dt,
            self.config.solver_iterations,
        );

        let prev = self.contact_persistence.prev_contacts();
        let warm = if prev.is_empty() { None } else { Some(prev) };
        let (results, new_contacts) =
            self.gpu_pipeline
                .step_with_contacts(num_bodies, self.config.solver_iterations, warm);

        let events = self.contact_persistence.update(&new_contacts);
        self.collision_events.extend(events);

        for (slot, &orig) in alive_indices.iter().enumerate() {
            if slot < results.len() {
                self.states[orig] = results[slot];
            }
        }
    }

    /// Drain collision events from the last step.
    pub fn drain_collision_events(&mut self) -> Vec<CollisionEvent> {
        std::mem::take(&mut self.collision_events)
    }

    /// Cast a 2D ray and return the closest hit (handle, t parameter, hit normal).
    pub fn raycast(
        &self,
        origin: Vec2,
        direction: Vec2,
        max_t: f32,
    ) -> Option<(BodyHandle, f32, Vec2)> {
        let dir_len = direction.length();
        if dir_len < 1e-12 {
            return None;
        }
        let dir = direction / dir_len;

        let mut best_t = max_t * dir_len;
        let mut best_handle = None;
        let mut best_normal = Vec2::ZERO;

        for (idx, &alive) in self.allocator.alive.iter().enumerate() {
            if !alive {
                continue;
            }
            let state = &self.states[idx];
            let pos = state.position();

            if let Some((t, normal)) = self.ray_shape_test(origin, dir, pos, idx) {
                if t >= 0.0 && t < best_t {
                    best_t = t;
                    best_handle =
                        Some(BodyHandle::new(idx as u32, self.allocator.generations[idx]));
                    best_normal = normal;
                }
            }
        }

        best_handle.map(|h| (h, best_t / dir_len, best_normal))
    }

    fn ray_shape_test(
        &self,
        origin: Vec2,
        dir: Vec2,
        pos: Vec2,
        idx: usize,
    ) -> Option<(f32, Vec2)> {
        match &self.shapes[idx] {
            ShapeDesc2D::Circle { radius } => {
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
            ShapeDesc2D::Rect { half_extents } => {
                let angle = self.states[idx].angle();
                let (sin, cos) = (-angle).sin_cos();
                let local_origin = Vec2::new(
                    cos * (origin.x - pos.x) - sin * (origin.y - pos.y),
                    sin * (origin.x - pos.x) + cos * (origin.y - pos.y),
                );
                let local_dir = Vec2::new(cos * dir.x - sin * dir.y, sin * dir.x + cos * dir.y);
                let he = *half_extents;

                let mut tmin = f32::NEG_INFINITY;
                let mut tmax = f32::INFINITY;
                let mut normal_idx = 0usize;
                let mut normal_sign = 1.0f32;

                for i in 0..2 {
                    let o = [local_origin.x, local_origin.y][i];
                    let d = [local_dir.x, local_dir.y][i];
                    let h = [he.x, he.y][i];

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
                let local_normal = if normal_idx == 0 {
                    Vec2::new(normal_sign, 0.0)
                } else {
                    Vec2::new(0.0, normal_sign)
                };
                let (sin_fwd, cos_fwd) = angle.sin_cos();
                let world_normal = Vec2::new(
                    cos_fwd * local_normal.x - sin_fwd * local_normal.y,
                    sin_fwd * local_normal.x + cos_fwd * local_normal.y,
                );
                Some((tmin, world_normal))
            }
            ShapeDesc2D::Capsule {
                half_height,
                radius,
            } => {
                let angle = self.states[idx].angle();
                let (sin_a, cos_a) = angle.sin_cos();
                // Capsule axis is local Y, endpoints = pos ± half_height * local_Y_rotated
                let axis = Vec2::new(-sin_a * half_height, cos_a * half_height);
                let ep_a = pos + axis;
                let ep_b = pos - axis;
                // Find closest point on segment to ray, then do ray-circle test
                ray_capsule_2d(origin, dir, ep_a, ep_b, *radius)
            }
            ShapeDesc2D::ConvexPolygon { vertices } => {
                let angle = self.states[idx].angle();
                let (sin_a, cos_a) = angle.sin_cos();
                // Transform vertices to world space, then ray-polygon test
                let world_verts: Vec<Vec2> = vertices
                    .iter()
                    .map(|v| pos + Vec2::new(cos_a * v.x - sin_a * v.y, sin_a * v.x + cos_a * v.y))
                    .collect();
                ray_convex_polygon_2d(origin, dir, &world_verts)
            }
        }
    }

    /// Query all bodies whose AABB overlaps the given axis-aligned bounding box.
    pub fn overlap_aabb(&self, query_min: Vec2, query_max: Vec2) -> Vec<BodyHandle> {
        let mut result = Vec::new();
        for (idx, &alive) in self.allocator.alive.iter().enumerate() {
            if !alive {
                continue;
            }
            let state = &self.states[idx];
            let pos = state.position();
            let (body_min, body_max) = self.compute_body_aabb(pos, state, idx);

            if body_min.x <= query_max.x
                && body_max.x >= query_min.x
                && body_min.y <= query_max.y
                && body_max.y >= query_min.y
            {
                result.push(BodyHandle::new(idx as u32, self.allocator.generations[idx]));
            }
        }
        result
    }

    fn compute_body_aabb(&self, pos: Vec2, state: &RigidBodyState2D, idx: usize) -> (Vec2, Vec2) {
        let aabb = match &self.shapes[idx] {
            ShapeDesc2D::Circle { radius } => rubble_shapes2d::compute_circle_aabb(pos, *radius),
            ShapeDesc2D::Rect { half_extents } => {
                rubble_shapes2d::compute_rect_aabb(pos, state.angle(), *half_extents)
            }
            ShapeDesc2D::ConvexPolygon { vertices } => {
                rubble_shapes2d::compute_convex_polygon_aabb(pos, state.angle(), vertices)
            }
            ShapeDesc2D::Capsule {
                half_height,
                radius,
            } => rubble_shapes2d::compute_capsule2d_aabb(pos, state.angle(), *half_height, *radius),
        };
        (aabb.min_point(), aabb.max_point())
    }

    /// Access the GPU pipeline (for diagnostics like contact count).
    pub fn gpu_pipeline(&self) -> &gpu::GpuPipeline2D {
        &self.gpu_pipeline
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_world(config: SimConfig2D) -> World2D {
        World2D::new(config).expect("GPU required for tests")
    }

    fn gpu_world_default() -> World2D {
        gpu_world(SimConfig2D::default())
    }

    #[test]
    fn test_world2d_new() {
        let world = gpu_world_default();
        assert_eq!(world.body_count(), 0);
        assert!(world.states.is_empty());
        assert_eq!(world.config.gravity, Vec2::new(0.0, -9.81));
        assert!((world.config.dt - 1.0 / 60.0).abs() < 1e-6);
        assert_eq!(world.config.solver_iterations, 5);
        assert_eq!(world.config.max_bodies, 65536);
    }

    #[test]
    fn test_add_remove_body() {
        let mut world = gpu_world_default();

        let h = world.add_body(&RigidBodyDesc2D {
            x: 1.0,
            y: 2.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });

        assert_eq!(world.body_count(), 1);
        assert_eq!(world.get_position(h), Some(Vec2::new(1.0, 2.0)));

        let removed = world.remove_body(h);
        assert!(removed);
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.get_position(h), None);

        assert!(!world.remove_body(h));
    }

    #[test]
    fn test_handle_allocator() {
        let mut alloc = GenerationalIndexAllocator::new();

        let h0 = alloc.allocate();
        assert_eq!(h0.index, 0);
        assert_eq!(h0.generation, 0);
        assert!(alloc.is_alive(h0));

        let h1 = alloc.allocate();
        assert_eq!(h1.index, 1);

        assert!(alloc.deallocate(h0));
        assert!(!alloc.is_alive(h0));

        let h2 = alloc.allocate();
        assert_eq!(h2.index, 0);
        assert_eq!(h2.generation, 1);
        assert!(alloc.is_alive(h2));
        assert!(!alloc.is_alive(h0));
        assert_eq!(alloc.live_count(), 2);
    }

    #[test]
    fn test_gravity_fall_2d() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 1024,
            ..Default::default()
        });

        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 10.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });

        for _ in 0..60 {
            world.step();
        }

        let pos = world.get_position(h).unwrap();
        let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
        let error = (pos.y - expected_y).abs();
        assert!(
            error < 0.5,
            "Expected y ~ {expected_y}, got {}, error {error}",
            pos.y
        );
        assert!(pos.x.abs() < 1e-6, "X should remain ~0, got {}", pos.x);
    }

    #[test]
    fn test_two_body_collision_2d() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
            ..Default::default()
        });

        let h_a = world.add_body(&RigidBodyDesc2D {
            x: -2.0,
            y: 0.0,
            vx: 5.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });
        let h_b = world.add_body(&RigidBodyDesc2D {
            x: 2.0,
            y: 0.0,
            vx: -5.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });

        for _ in 0..120 {
            world.step();
        }

        let pos_a = world.get_position(h_a).unwrap();
        let pos_b = world.get_position(h_b).unwrap();
        let dist = (pos_b.x - pos_a.x).abs();
        assert!(
            dist >= 1.5,
            "Bodies should be separated (dist={dist}), not heavily overlapping"
        );
    }

    #[test]
    fn test_body_count() {
        let mut world = gpu_world_default();
        assert_eq!(world.body_count(), 0);

        let h1 = world.add_body(&RigidBodyDesc2D::default());
        assert_eq!(world.body_count(), 1);

        let h2 = world.add_body(&RigidBodyDesc2D::default());
        assert_eq!(world.body_count(), 2);

        let h3 = world.add_body(&RigidBodyDesc2D::default());
        assert_eq!(world.body_count(), 3);

        world.remove_body(h2);
        assert_eq!(world.body_count(), 2);

        world.remove_body(h1);
        assert_eq!(world.body_count(), 1);

        world.remove_body(h3);
        assert_eq!(world.body_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Raycast tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_raycast_circle_hit() {
        let mut world = gpu_world_default();
        world.add_body(&RigidBodyDesc2D {
            x: 5.0,
            y: 0.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });
        let hit = world.raycast(Vec2::ZERO, Vec2::X, 100.0);
        assert!(hit.is_some(), "Ray should hit circle");
        let (_, t, _) = hit.unwrap();
        assert!((t - 4.0).abs() < 0.1, "Expected t~4.0, got {t}");
    }

    #[test]
    fn test_raycast_circle_miss() {
        let mut world = gpu_world_default();
        world.add_body(&RigidBodyDesc2D {
            x: 5.0,
            y: 5.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });
        let hit = world.raycast(Vec2::ZERO, Vec2::X, 100.0);
        assert!(hit.is_none(), "Ray should miss circle above");
    }

    #[test]
    fn test_raycast_rect_hit() {
        let mut world = gpu_world_default();
        world.add_body(&RigidBodyDesc2D {
            x: 3.0,
            y: 0.0,
            mass: 1.0,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(1.0, 1.0),
            },
            ..Default::default()
        });
        let hit = world.raycast(Vec2::ZERO, Vec2::X, 100.0);
        assert!(hit.is_some(), "Ray should hit rect");
        let (_, t, _) = hit.unwrap();
        assert!((t - 2.0).abs() < 0.1, "Expected t~2.0, got {t}");
    }

    #[test]
    fn test_raycast_capsule_hit() {
        let mut world = gpu_world_default();
        world.add_body(&RigidBodyDesc2D {
            x: 4.0,
            y: 0.0,
            mass: 1.0,
            shape: ShapeDesc2D::Capsule {
                half_height: 1.0,
                radius: 0.5,
            },
            ..Default::default()
        });
        let hit = world.raycast(Vec2::ZERO, Vec2::X, 100.0);
        assert!(hit.is_some(), "Ray should hit capsule");
        let (_, t, _) = hit.unwrap();
        assert!(t > 2.0 && t < 5.0, "Expected reasonable t, got {t}");
    }

    #[test]
    fn test_raycast_convex_polygon_hit() {
        let mut world = gpu_world_default();
        // Triangle centered at (5,0)
        world.add_body(&RigidBodyDesc2D {
            x: 5.0,
            y: 0.0,
            mass: 1.0,
            shape: ShapeDesc2D::ConvexPolygon {
                vertices: vec![
                    Vec2::new(-1.0, -1.0),
                    Vec2::new(1.0, -1.0),
                    Vec2::new(0.0, 1.0),
                ],
            },
            ..Default::default()
        });
        let hit = world.raycast(Vec2::ZERO, Vec2::X, 100.0);
        assert!(hit.is_some(), "Ray should hit convex polygon");
    }

    #[test]
    fn test_raycast_batch() {
        let mut world = gpu_world_default();
        world.add_body(&RigidBodyDesc2D {
            x: 3.0,
            y: 0.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });
        world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 3.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });

        let results = world.raycast_batch(&[
            (Vec2::ZERO, Vec2::X, 100.0),
            (Vec2::ZERO, Vec2::Y, 100.0),
            (Vec2::ZERO, Vec2::new(1.0, 0.0), 0.5), // too short
        ]);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some(), "First ray should hit");
        assert!(results[1].is_some(), "Second ray should hit");
        assert!(results[2].is_none(), "Third ray too short");
    }

    // -----------------------------------------------------------------------
    // Overlap tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_overlap_aabb() {
        let mut world = gpu_world_default();
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 0.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });
        // Query that overlaps
        let hits = world.overlap_aabb(Vec2::new(-0.5, -0.5), Vec2::new(0.5, 0.5));
        assert!(hits.contains(&h), "Should find body in overlapping AABB");
        // Query that doesn't overlap
        let misses = world.overlap_aabb(Vec2::new(10.0, 10.0), Vec2::new(11.0, 11.0));
        assert!(misses.is_empty(), "Should find no body far away");
    }

    // -----------------------------------------------------------------------
    // Kinematic body tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kinematic_body_no_gravity() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 5.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });
        world.set_body_kinematic(h, true);

        for _ in 0..60 {
            world.step();
        }
        let pos = world.get_position(h).unwrap();
        assert!(
            (pos.y - 5.0).abs() < 1e-3,
            "Kinematic body should not fall, y={}",
            pos.y
        );
    }

    // -----------------------------------------------------------------------
    // Collision events
    // -----------------------------------------------------------------------

    #[test]
    fn test_collision_events_drain_once() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::ZERO,
            ..Default::default()
        });
        // Two bodies moving toward each other
        world.add_body(&RigidBodyDesc2D {
            x: -1.0,
            y: 0.0,
            vx: 10.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });
        world.add_body(&RigidBodyDesc2D {
            x: 1.0,
            y: 0.0,
            vx: -10.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });

        for _ in 0..10 {
            world.step();
        }
        let events = world.drain_collision_events();
        // Drain again — should be empty
        let events2 = world.drain_collision_events();
        assert!(
            events2.is_empty(),
            "Second drain should return empty, got {} events",
            events2.len()
        );
        // First drain may or may not have events (depends on CPU narrowphase), but API works
        let _ = events;
    }

    // -----------------------------------------------------------------------
    // Setter / getter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_position_and_velocity() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::ZERO,
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc2D {
            mass: 1.0,
            ..Default::default()
        });

        world.set_position(h, Vec2::new(10.0, 20.0));
        assert_eq!(world.get_position(h), Some(Vec2::new(10.0, 20.0)));

        world.set_velocity(h, Vec2::new(3.0, 4.0));
        assert_eq!(world.get_velocity(h), Some(Vec2::new(3.0, 4.0)));
    }

    #[test]
    fn test_set_angular_velocity() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::ZERO,
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc2D {
            mass: 1.0,
            ..Default::default()
        });
        world.set_angular_velocity(h, 2.0);
        // Step and check angle changed
        world.step();
        let angle = world.get_angle(h).unwrap();
        assert!(angle.abs() > 0.01, "Angle should have changed, got {angle}");
    }

    #[test]
    fn test_get_angle() {
        let mut world = gpu_world_default();
        let h = world.add_body(&RigidBodyDesc2D {
            angle: 1.5,
            mass: 1.0,
            ..Default::default()
        });
        let a = world.get_angle(h).unwrap();
        assert!(
            (a - 1.5).abs() < 1e-6,
            "Initial angle should be 1.5, got {a}"
        );
    }

    // -----------------------------------------------------------------------
    // Shape-specific tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rect_creation_and_step() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 10.0,
            mass: 1.0,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(1.0, 0.5),
            },
            ..Default::default()
        });
        for _ in 0..10 {
            world.step();
        }
        let pos = world.get_position(h).unwrap();
        assert!(pos.y < 10.0, "Rect should fall, y={}", pos.y);
    }

    #[test]
    fn test_capsule_creation_and_step() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 10.0,
            mass: 1.0,
            shape: ShapeDesc2D::Capsule {
                half_height: 1.0,
                radius: 0.5,
            },
            ..Default::default()
        });
        for _ in 0..10 {
            world.step();
        }
        let pos = world.get_position(h).unwrap();
        assert!(pos.y < 10.0, "Capsule should fall, y={}", pos.y);
    }

    #[test]
    fn test_convex_polygon_creation() {
        let mut world = gpu_world_default();
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 0.0,
            mass: 1.0,
            shape: ShapeDesc2D::ConvexPolygon {
                vertices: vec![
                    Vec2::new(-1.0, -1.0),
                    Vec2::new(1.0, -1.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(-1.0, 1.0),
                ],
            },
            ..Default::default()
        });
        assert_eq!(world.body_count(), 1);
        assert!(world.get_position(h).is_some());
    }

    // -----------------------------------------------------------------------
    // Static body tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_static_body_no_motion() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 5.0,
            mass: 0.0, // static
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });
        for _ in 0..60 {
            world.step();
        }
        let pos = world.get_position(h).unwrap();
        assert!(
            (pos.y - 5.0).abs() < 1e-6,
            "Static body should not move, y={}",
            pos.y
        );
    }

    // -----------------------------------------------------------------------
    // Zero gravity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_gravity_constant_velocity() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::ZERO,
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 0.0,
            vx: 1.0,
            vy: 2.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });
        let dt = world.config.dt;
        world.step();
        let pos = world.get_position(h).unwrap();
        assert!((pos.x - dt).abs() < 1e-4, "x should advance by vx*dt");
        assert!((pos.y - 2.0 * dt).abs() < 1e-4, "y should advance by vy*dt");
    }

    // -----------------------------------------------------------------------
    // Stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_many_bodies_no_crash() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            max_bodies: 1024,
            ..Default::default()
        });
        for i in 0..100 {
            world.add_body(&RigidBodyDesc2D {
                x: (i % 10) as f32 * 3.0,
                y: (i / 10) as f32 * 3.0,
                mass: 1.0,
                shape: ShapeDesc2D::Circle { radius: 0.5 },
                ..Default::default()
            });
        }
        assert_eq!(world.body_count(), 100);
        for _ in 0..10 {
            world.step();
        }
        assert_eq!(world.body_count(), 100);
    }

    #[test]
    fn test_add_remove_cycle() {
        let mut world = gpu_world_default();
        let mut handles = Vec::new();
        for i in 0..20 {
            handles.push(world.add_body(&RigidBodyDesc2D {
                x: i as f32,
                mass: 1.0,
                ..Default::default()
            }));
        }
        assert_eq!(world.body_count(), 20);
        // Remove even-indexed
        for i in (0..20).step_by(2) {
            world.remove_body(handles[i]);
        }
        assert_eq!(world.body_count(), 10);
        // Re-add
        for i in 0..10 {
            world.add_body(&RigidBodyDesc2D {
                x: (100 + i) as f32,
                mass: 1.0,
                ..Default::default()
            });
        }
        assert_eq!(world.body_count(), 20);
    }

    #[test]
    fn test_friction_coefficient_stored() {
        let mut world = gpu_world_default();
        let h = world.add_body(&RigidBodyDesc2D {
            mass: 1.0,
            friction: 0.8,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });
        // Verify friction is stored (accessible via internal state)
        let idx = h.index as usize;
        assert!((world.frictions[idx] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_shape_types_simulation() {
        let mut world = gpu_world(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        });

        world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 5.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });
        world.add_body(&RigidBodyDesc2D {
            x: 3.0,
            y: 5.0,
            mass: 1.0,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(0.5, 0.5),
            },
            ..Default::default()
        });
        world.add_body(&RigidBodyDesc2D {
            x: 6.0,
            y: 5.0,
            mass: 1.0,
            shape: ShapeDesc2D::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            ..Default::default()
        });
        world.add_body(&RigidBodyDesc2D {
            x: 9.0,
            y: 5.0,
            mass: 1.0,
            shape: ShapeDesc2D::ConvexPolygon {
                vertices: vec![
                    Vec2::new(-0.5, -0.5),
                    Vec2::new(0.5, -0.5),
                    Vec2::new(0.0, 0.5),
                ],
            },
            ..Default::default()
        });

        assert_eq!(world.body_count(), 4);
        for _ in 0..30 {
            world.step();
        }
        // All bodies should have fallen
        for i in 0..4 {
            let h = BodyHandle {
                index: i,
                generation: 0,
            };
            let pos = world.get_position(h).unwrap();
            assert!(pos.y < 5.0, "Body {i} should have fallen, y={}", pos.y);
        }
    }
}

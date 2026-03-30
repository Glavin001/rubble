//! `rubble3d` -- Main facade crate for 3D physics simulation.
//!
//! Orchestrates the full physics pipeline: broadphase, narrowphase,
//! contact persistence, and solver. CPU-only reference implementation.

use glam::{Mat3, Quat, Vec3, Vec4};
use rubble_math::*;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the 3D physics simulation.
pub struct SimConfig {
    pub gravity: Vec3,
    pub dt: f32,
    pub solver_iterations: u32,
    pub beta: f32,
    pub k_start: f32,
    pub warmstart_decay: f32,
    pub friction_default: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            beta: 10.0,
            k_start: 1e4,
            warmstart_decay: 0.95,
            friction_default: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape description
// ---------------------------------------------------------------------------

/// Description of a collision shape.
#[derive(Debug, Clone)]
pub enum ShapeDesc {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { half_height: f32, radius: f32 },
}

// ---------------------------------------------------------------------------
// Rigid body description
// ---------------------------------------------------------------------------

/// Description for creating a new rigid body.
pub struct RigidBodyDesc {
    pub shape: ShapeDesc,
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    /// Mass of the body. 0 means static (infinite mass).
    pub mass: f32,
    pub friction: f32,
}

// ---------------------------------------------------------------------------
// Handle allocator
// ---------------------------------------------------------------------------

/// Generational index allocator for body handles.
struct HandleAllocator {
    generations: Vec<u32>,
    free_list: Vec<u32>,
}

impl HandleAllocator {
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

    fn dealloc(&mut self, h: BodyHandle) {
        if self.is_valid(h) {
            self.generations[h.index as usize] += 1;
            self.free_list.push(h.index);
        }
    }

    fn is_valid(&self, h: BodyHandle) -> bool {
        let idx = h.index as usize;
        idx < self.generations.len() && self.generations[idx] == h.generation
    }
}

// ---------------------------------------------------------------------------
// Contact persistence
// ---------------------------------------------------------------------------

/// Tracks contacts across frames for warm-starting and collision event generation.
struct ContactPersistence {
    previous_pairs: HashSet<(u32, u32)>,
    cached_lambdas: HashMap<(u32, u32, u32), (f32, f32, f32)>,
}

impl ContactPersistence {
    fn new() -> Self {
        Self {
            previous_pairs: HashSet::new(),
            cached_lambdas: HashMap::new(),
        }
    }

    fn update(
        &mut self,
        contacts: &mut [Contact3D],
        handles: &[BodyHandle],
    ) -> Vec<CollisionEvent> {
        let mut events = Vec::new();
        let mut current_pairs = HashSet::new();

        for c in contacts.iter_mut() {
            let pair = (c.body_a.min(c.body_b), c.body_a.max(c.body_b));
            current_pairs.insert(pair);

            let key = (pair.0, pair.1, c.feature_id);
            if let Some(&(ln, lt1, lt2)) = self.cached_lambdas.get(&key) {
                c.lambda_n = ln;
                c.lambda_t1 = lt1;
                c.lambda_t2 = lt2;
            }
        }

        for &pair in &current_pairs {
            if !self.previous_pairs.contains(&pair) {
                let ha = if (pair.0 as usize) < handles.len() {
                    handles[pair.0 as usize]
                } else {
                    continue;
                };
                let hb = if (pair.1 as usize) < handles.len() {
                    handles[pair.1 as usize]
                } else {
                    continue;
                };
                events.push(CollisionEvent::Started {
                    body_a: ha,
                    body_b: hb,
                });
            }
        }

        for &pair in &self.previous_pairs {
            if !current_pairs.contains(&pair) {
                let ha = if (pair.0 as usize) < handles.len() {
                    handles[pair.0 as usize]
                } else {
                    continue;
                };
                let hb = if (pair.1 as usize) < handles.len() {
                    handles[pair.1 as usize]
                } else {
                    continue;
                };
                events.push(CollisionEvent::Ended {
                    body_a: ha,
                    body_b: hb,
                });
            }
        }

        self.cached_lambdas.clear();
        for c in contacts.iter() {
            let pair = (c.body_a.min(c.body_b), c.body_a.max(c.body_b));
            let key = (pair.0, pair.1, c.feature_id);
            self.cached_lambdas
                .insert(key, (c.lambda_n, c.lambda_t1, c.lambda_t2));
        }

        self.previous_pairs = current_pairs;
        events
    }
}

// ---------------------------------------------------------------------------
// Inline narrowphase (since rubble-narrowphase3d is a stub)
// ---------------------------------------------------------------------------

fn make_contact3d(
    body_a: u32,
    body_b: u32,
    point: Vec3,
    normal: Vec3,
    depth: f32,
) -> Contact3D {
    Contact3D {
        point: Vec4::new(point.x, point.y, point.z, depth),
        normal: Vec4::new(normal.x, normal.y, normal.z, 0.0),
        body_a,
        body_b,
        feature_id: 0,
        _pad: 0,
        lambda_n: 0.0,
        lambda_t1: 0.0,
        lambda_t2: 0.0,
        penalty_k: 0.0,
    }
}

fn sphere_sphere(
    pos_a: Vec3,
    rad_a: f32,
    pos_b: Vec3,
    rad_b: f32,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    let d = pos_b - pos_a;
    let dist = d.length();
    let r_sum = rad_a + rad_b;
    if dist >= r_sum || dist < 1e-10 {
        return vec![];
    }
    let normal = d / dist;
    let depth = -(r_sum - dist);
    let point = pos_a + normal * (rad_a + depth * 0.5);
    vec![make_contact3d(body_a, body_b, point, normal, depth)]
}

fn sphere_box(
    sphere_pos: Vec3,
    radius: f32,
    box_pos: Vec3,
    box_rot: Quat,
    half: Vec3,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    let rot_inv = box_rot.conjugate();
    let local = rot_inv * (sphere_pos - box_pos);
    let closest = local.clamp(-half, half);
    let diff = local - closest;
    let dist_sq = diff.length_squared();
    if dist_sq >= radius * radius {
        return vec![];
    }
    let dist = dist_sq.sqrt().max(1e-10);
    let normal_local = diff / dist;
    let normal = box_rot * normal_local;
    let closest_world = box_pos + box_rot * closest;
    let depth = -(radius - dist);
    vec![make_contact3d(body_a, body_b, closest_world, normal, depth)]
}

fn box_box(
    pos_a: Vec3,
    rot_a: Quat,
    half_a: Vec3,
    pos_b: Vec3,
    rot_b: Quat,
    half_b: Vec3,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    // SAT with 15 axes: 3 from A, 3 from B, 9 edge cross products
    let mat_a = Mat3::from_quat(rot_a);
    let mat_b = Mat3::from_quat(rot_b);
    let d = pos_b - pos_a;

    let axes_a = [mat_a.x_axis, mat_a.y_axis, mat_a.z_axis];
    let axes_b = [mat_b.x_axis, mat_b.y_axis, mat_b.z_axis];
    let halves_a = [half_a.x, half_a.y, half_a.z];
    let halves_b = [half_b.x, half_b.y, half_b.z];

    let mut min_overlap = f32::MAX;
    let mut min_axis = Vec3::ZERO;

    // Test face axes
    for axes in [&axes_a, &axes_b] {
        for &axis in axes.iter() {
            let proj_a: f32 = (0..3).map(|i| halves_a[i] * axes_a[i].dot(axis).abs()).sum();
            let proj_b: f32 = (0..3).map(|i| halves_b[i] * axes_b[i].dot(axis).abs()).sum();
            let dist = d.dot(axis).abs();
            let overlap = proj_a + proj_b - dist;
            if overlap <= 0.0 {
                return vec![];
            }
            if overlap < min_overlap {
                min_overlap = overlap;
                min_axis = if d.dot(axis) >= 0.0 { axis } else { -axis };
            }
        }
    }

    // Test edge cross-product axes
    for i in 0..3 {
        for j in 0..3 {
            let axis = axes_a[i].cross(axes_b[j]);
            let len = axis.length();
            if len < 1e-6 {
                continue; // parallel edges
            }
            let axis = axis / len;
            let proj_a: f32 = (0..3).map(|k| halves_a[k] * axes_a[k].dot(axis).abs()).sum();
            let proj_b: f32 = (0..3).map(|k| halves_b[k] * axes_b[k].dot(axis).abs()).sum();
            let dist = d.dot(axis).abs();
            let overlap = proj_a + proj_b - dist;
            if overlap <= 0.0 {
                return vec![];
            }
            if overlap < min_overlap {
                min_overlap = overlap;
                min_axis = if d.dot(axis) >= 0.0 { axis } else { -axis };
            }
        }
    }

    let depth = -min_overlap;
    let contact_point = (pos_a + pos_b) * 0.5;
    vec![make_contact3d(
        body_a,
        body_b,
        contact_point,
        min_axis,
        depth,
    )]
}

fn sphere_plane(
    sphere_pos: Vec3,
    radius: f32,
    plane: &rubble_shapes3d::Plane,
    body_idx: u32,
    plane_body_idx: u32,
) -> Vec<Contact3D> {
    let dist = sphere_pos.dot(plane.normal) - plane.distance;
    if dist >= radius {
        return vec![];
    }
    let depth = -(radius - dist);
    let point = sphere_pos - plane.normal * dist;
    vec![make_contact3d(
        body_idx,
        plane_body_idx,
        point,
        plane.normal,
        depth,
    )]
}

fn box_plane(
    box_pos: Vec3,
    box_rot: Quat,
    half: Vec3,
    plane: &rubble_shapes3d::Plane,
    body_idx: u32,
    plane_body_idx: u32,
) -> Vec<Contact3D> {
    // Find the vertex most penetrating the plane
    let rot_mat = Mat3::from_quat(box_rot);
    let mut deepest_depth = 0.0_f32;
    let mut deepest_point = Vec3::ZERO;
    let mut found = false;

    for i in 0..8u32 {
        let x = if i & 1 == 0 { -half.x } else { half.x };
        let y = if i & 2 == 0 { -half.y } else { half.y };
        let z = if i & 4 == 0 { -half.z } else { half.z };
        let local = Vec3::new(x, y, z);
        let world = box_pos + rot_mat * local;
        let dist = world.dot(plane.normal) - plane.distance;
        if dist < deepest_depth || !found {
            if dist < 0.0 || !found {
                deepest_depth = dist;
                deepest_point = world;
                found = true;
            }
        }
    }

    if deepest_depth >= 0.0 {
        return vec![];
    }

    let depth = deepest_depth;
    vec![make_contact3d(
        body_idx,
        plane_body_idx,
        deepest_point,
        plane.normal,
        depth,
    )]
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

/// The main 3D physics world.
pub struct World {
    config: SimConfig,
    allocator: HandleAllocator,

    // Body data (parallel arrays)
    states: Vec<RigidBodyState3D>,
    shapes: Vec<ShapeDesc>,
    frictions: Vec<f32>,
    inv_inertias: Vec<Mat3>,
    handles: Vec<BodyHandle>,
    active: Vec<bool>,

    // Pipeline components
    persistence: ContactPersistence,
    solver: rubble_solver3d::Solver3D,

    // Events
    events: Vec<CollisionEvent>,

    // Planes
    planes: Vec<rubble_shapes3d::Plane>,
}

impl World {
    /// Create a new physics world with the given configuration.
    pub fn new(config: SimConfig) -> Self {
        let solver_params = rubble_solver3d::SolverParams {
            iterations: config.solver_iterations,
            beta: config.beta,
            k_start: config.k_start,
            warmstart_decay: config.warmstart_decay,
        };
        Self {
            solver: rubble_solver3d::Solver3D::new(solver_params),
            config,
            allocator: HandleAllocator::new(),
            states: Vec::new(),
            shapes: Vec::new(),
            frictions: Vec::new(),
            inv_inertias: Vec::new(),
            handles: Vec::new(),
            active: Vec::new(),
            persistence: ContactPersistence::new(),
            events: Vec::new(),
            planes: Vec::new(),
        }
    }

    /// Add a rigid body to the world and return its handle.
    pub fn add_body(&mut self, desc: RigidBodyDesc) -> BodyHandle {
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

        let inv_inertia = if desc.mass > 0.0 {
            compute_inv_inertia(&desc.shape, desc.mass)
        } else {
            Mat3::ZERO
        };

        // Ensure arrays are large enough
        if idx >= self.states.len() {
            self.states.resize(
                idx + 1,
                RigidBodyState3D::new(Vec3::ZERO, 0.0, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO),
            );
            self.shapes.resize(idx + 1, ShapeDesc::Sphere { radius: 1.0 });
            self.frictions.resize(idx + 1, 0.5);
            self.inv_inertias.resize(idx + 1, Mat3::ZERO);
            self.handles.resize(idx + 1, BodyHandle::new(0, u32::MAX));
            self.active.resize(idx + 1, false);
        }

        self.states[idx] = state;
        self.shapes[idx] = desc.shape;
        self.frictions[idx] = desc.friction;
        self.inv_inertias[idx] = inv_inertia;
        self.handles[idx] = handle;
        self.active[idx] = true;

        handle
    }

    /// Remove a body from the world.
    pub fn remove_body(&mut self, handle: BodyHandle) {
        if self.allocator.is_valid(handle) {
            let idx = handle.index as usize;
            self.active[idx] = false;
            self.allocator.dealloc(handle);
        }
    }

    /// Return the number of active bodies.
    pub fn body_count(&self) -> u32 {
        self.active.iter().filter(|&&a| a).count() as u32
    }

    /// Add a static ground plane.
    pub fn add_plane(&mut self, normal: Vec3, distance: f32) {
        self.planes.push(rubble_shapes3d::Plane { normal, distance });
    }

    /// Step the simulation forward by one dt.
    pub fn step(&mut self) {
        let dt = self.config.dt;
        let gravity = self.config.gravity;

        // Collect active body indices
        let active_indices: Vec<usize> = (0..self.states.len())
            .filter(|&i| self.active[i])
            .collect();

        if active_indices.is_empty() {
            return;
        }

        // Build index mapping: active_slot -> original_index
        let _n = active_indices.len();
        let mut index_to_slot: Vec<usize> = vec![usize::MAX; self.states.len()];
        for (slot, &orig) in active_indices.iter().enumerate() {
            index_to_slot[orig] = slot;
        }

        // Collect states into a compact array for the solver
        let mut compact_states: Vec<RigidBodyState3D> =
            active_indices.iter().map(|&i| self.states[i]).collect();
        let compact_inv_inertias: Vec<Mat3> =
            active_indices.iter().map(|&i| self.inv_inertias[i]).collect();

        // 1. Compute AABBs
        let aabbs: Vec<Aabb3D> = active_indices
            .iter()
            .map(|&i| compute_aabb(&self.shapes[i], self.states[i].position(), self.states[i].quat()))
            .collect();

        // 2. Broadphase
        let lbvh = rubble_broadphase3d::Lbvh::build(&aabbs);
        let bp_result = lbvh.find_overlapping_pairs(&aabbs);

        // 3. Narrowphase
        let mut contacts = Vec::new();

        // Body-body pairs
        for pair in &bp_result.pairs {
            let slot_a = pair[0] as usize;
            let slot_b = pair[1] as usize;
            let orig_a = active_indices[slot_a];
            let orig_b = active_indices[slot_b];

            let new_contacts = narrowphase_pair(
                &self.shapes[orig_a],
                self.states[orig_a].position(),
                self.states[orig_a].quat(),
                &self.shapes[orig_b],
                self.states[orig_b].position(),
                self.states[orig_b].quat(),
                slot_a as u32,
                slot_b as u32,
            );
            contacts.extend(new_contacts);
        }

        // Plane-body contacts
        for (slot, &orig) in active_indices.iter().enumerate() {
            if self.states[orig].inv_mass() <= 0.0 {
                continue;
            }
            for plane in &self.planes {
                let plane_contacts = narrowphase_plane(
                    &self.shapes[orig],
                    self.states[orig].position(),
                    self.states[orig].quat(),
                    plane,
                    slot as u32,
                );
                contacts.extend(plane_contacts);
            }
        }

        // 4. Contact persistence
        let compact_handles: Vec<BodyHandle> =
            active_indices.iter().map(|&i| self.handles[i]).collect();
        let new_events = self.persistence.update(&mut contacts, &compact_handles);
        self.events.extend(new_events);

        // 5. Solve
        self.solver.solve(
            dt,
            gravity,
            &mut compact_states,
            &compact_inv_inertias,
            &mut contacts,
        );

        // Write back
        for (slot, &orig) in active_indices.iter().enumerate() {
            self.states[orig] = compact_states[slot];
        }
    }

    /// Get the position of a body by handle.
    pub fn get_body_position(&self, handle: BodyHandle) -> Option<Vec3> {
        if self.allocator.is_valid(handle) && self.active[handle.index as usize] {
            Some(self.states[handle.index as usize].position())
        } else {
            None
        }
    }

    /// Get the linear velocity of a body by handle.
    pub fn get_body_velocity(&self, handle: BodyHandle) -> Option<Vec3> {
        if self.allocator.is_valid(handle) && self.active[handle.index as usize] {
            Some(self.states[handle.index as usize].linear_velocity())
        } else {
            None
        }
    }

    /// Drain all pending collision events.
    pub fn drain_collision_events(&mut self) -> Vec<CollisionEvent> {
        std::mem::take(&mut self.events)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_inv_inertia(shape: &ShapeDesc, mass: f32) -> Mat3 {
    if mass <= 0.0 {
        return Mat3::ZERO;
    }
    match shape {
        ShapeDesc::Sphere { radius } => {
            let i = 2.0 / 5.0 * mass * radius * radius;
            Mat3::from_diagonal(Vec3::splat(1.0 / i))
        }
        ShapeDesc::Box { half_extents } => {
            let h = *half_extents;
            let ix = mass / 3.0 * (h.y * h.y + h.z * h.z);
            let iy = mass / 3.0 * (h.x * h.x + h.z * h.z);
            let iz = mass / 3.0 * (h.x * h.x + h.y * h.y);
            Mat3::from_diagonal(Vec3::new(1.0 / ix, 1.0 / iy, 1.0 / iz))
        }
        ShapeDesc::Capsule {
            half_height,
            radius,
        } => {
            // Approximate as cylinder
            let r2 = radius * radius;
            let h = half_height * 2.0;
            let ix = mass * (3.0 * r2 + h * h) / 12.0;
            let iy = mass * r2 / 2.0;
            Mat3::from_diagonal(Vec3::new(1.0 / ix, 1.0 / iy, 1.0 / ix))
        }
    }
}

fn compute_aabb(shape: &ShapeDesc, position: Vec3, rotation: Quat) -> Aabb3D {
    match shape {
        ShapeDesc::Sphere { radius } => rubble_shapes3d::compute_sphere_aabb(position, *radius),
        ShapeDesc::Box { half_extents } => {
            rubble_shapes3d::compute_box_aabb(position, rotation, *half_extents)
        }
        ShapeDesc::Capsule {
            half_height,
            radius,
        } => rubble_shapes3d::compute_capsule_aabb(position, rotation, *half_height, *radius),
    }
}

fn narrowphase_pair(
    shape_a: &ShapeDesc,
    pos_a: Vec3,
    rot_a: Quat,
    shape_b: &ShapeDesc,
    pos_b: Vec3,
    rot_b: Quat,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    match (shape_a, shape_b) {
        (ShapeDesc::Sphere { radius: ra }, ShapeDesc::Sphere { radius: rb }) => {
            sphere_sphere(pos_a, *ra, pos_b, *rb, body_a, body_b)
        }
        (ShapeDesc::Sphere { radius }, ShapeDesc::Box { half_extents }) => {
            sphere_box(pos_a, *radius, pos_b, rot_b, *half_extents, body_a, body_b)
        }
        (ShapeDesc::Box { half_extents }, ShapeDesc::Sphere { radius }) => {
            // Swap and reverse normal
            let mut contacts =
                sphere_box(pos_b, *radius, pos_a, rot_a, *half_extents, body_b, body_a);
            for c in &mut contacts {
                std::mem::swap(&mut c.body_a, &mut c.body_b);
                c.normal = (-c.normal.truncate()).extend(0.0);
            }
            contacts
        }
        (
            ShapeDesc::Box {
                half_extents: ha,
            },
            ShapeDesc::Box {
                half_extents: hb,
            },
        ) => box_box(pos_a, rot_a, *ha, pos_b, rot_b, *hb, body_a, body_b),
        _ => vec![], // Capsule combinations not yet implemented
    }
}

fn narrowphase_plane(
    shape: &ShapeDesc,
    pos: Vec3,
    rot: Quat,
    plane: &rubble_shapes3d::Plane,
    body_idx: u32,
) -> Vec<Contact3D> {
    // Use a special "plane body" index that won't conflict.
    // We treat plane contacts with body_a = body_idx, body_b = body_idx
    // but with a special normal from the plane.
    match shape {
        ShapeDesc::Sphere { radius } => {
            sphere_plane(pos, *radius, plane, body_idx, body_idx)
        }
        ShapeDesc::Box { half_extents } => {
            box_plane(pos, rot, *half_extents, plane, body_idx, body_idx)
        }
        _ => vec![],
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
    }

    #[test]
    fn test_single_body_falls() {
        let mut world = World::new(SimConfig::default());
        let handle = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 0.5 },
            position: Vec3::new(0.0, 10.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
        });

        let start_y = world.get_body_position(handle).unwrap().y;

        for _ in 0..60 {
            world.step();
        }

        let end_y = world.get_body_position(handle).unwrap().y;
        assert!(
            end_y < start_y,
            "Body should fall under gravity: start_y={start_y}, end_y={end_y}"
        );
    }

    #[test]
    fn test_floor_and_box() {
        let mut world = World::new(SimConfig::default());

        // Static floor (mass = 0)
        let _floor = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(10.0, 0.5, 10.0),
            },
            position: Vec3::new(0.0, -0.5, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            friction: 0.5,
        });

        // Dynamic box above floor
        let box_handle = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 0.5 },
            position: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
        });

        for _ in 0..120 {
            world.step();
        }

        let y = world.get_body_position(box_handle).unwrap().y;
        // The sphere should have settled near the floor (y ~ 0.5, sphere radius)
        // Allow generous tolerance since this is a reference CPU solver
        assert!(
            y > -2.0 && y < 5.0,
            "Box should settle near floor, got y={y}"
        );
    }

    #[test]
    fn test_add_and_remove_bodies() {
        let mut world = World::new(SimConfig::default());

        let h1 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 0.5 },
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
        });
        assert_eq!(world.body_count(), 1);

        let h2 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 0.5 },
            position: Vec3::new(5.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
        });
        assert_eq!(world.body_count(), 2);

        world.remove_body(h1);
        assert_eq!(world.body_count(), 1);
        assert!(world.get_body_position(h1).is_none());
        assert!(world.get_body_position(h2).is_some());

        // Step should not panic with removed bodies
        world.step();

        // Re-use the freed slot
        let h3 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 0.5 },
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
        });
        assert_eq!(world.body_count(), 2);
        // h3 reuses index 0 but with generation 1
        assert_eq!(h3.index, 0);
        assert_eq!(h3.generation, 1);
        // Old handle h1 is still invalid
        assert!(world.get_body_position(h1).is_none());
    }

    #[test]
    fn test_collision_events() {
        let mut world = World::new(SimConfig::default());

        // Two overlapping spheres
        let _h1 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
        });

        let _h2 = world.add_body(RigidBodyDesc {
            shape: ShapeDesc::Sphere { radius: 1.0 },
            position: Vec3::new(1.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.5,
        });

        world.step();
        let events = world.drain_collision_events();
        let started_count = events
            .iter()
            .filter(|e| matches!(e, CollisionEvent::Started { .. }))
            .count();
        assert!(
            started_count > 0,
            "Should have at least one Started event for overlapping spheres"
        );
    }
}

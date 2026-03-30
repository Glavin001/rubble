//! `rubble2d` — Public API facade for the rubble 2D physics engine.
//!
//! Orchestrates broadphase (LBVH) -> narrowphase (circle/rect/capsule) -> solver
//! to provide a simple `World2D` that users step forward each frame.

#![allow(dead_code)]

use std::collections::HashSet;

use glam::{Vec2, Vec4};
use rubble_broadphase2d::Lbvh;
use rubble_math::{Aabb2D, BodyHandle, CollisionEvent, Contact2D, RigidBodyState2D};
use rubble_narrowphase2d::{circle_circle, circle_rect, rect_rect, ContactPersistence2D};
use rubble_shapes2d::{
    compute_capsule2d_aabb, compute_circle_aabb, compute_rect_aabb, CapsuleData2D, CircleData,
    RectData,
};
use rubble_solver2d::{Solver2D, SolverParams};

// ---------------------------------------------------------------------------
// SimConfig2D
// ---------------------------------------------------------------------------

/// Top-level simulation configuration.
pub struct SimConfig2D {
    pub gravity: Vec2,
    pub dt: f32,
    pub solver_iterations: u32,
    pub max_bodies: usize,
}

impl Default for SimConfig2D {
    fn default() -> Self {
        Self {
            gravity: Vec2::new(0.0, -9.81),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 65536,
        }
    }
}

// ---------------------------------------------------------------------------
// ShapeDesc2D
// ---------------------------------------------------------------------------

/// User-facing shape descriptor for adding bodies.
#[derive(Debug, Clone, Copy)]
pub enum ShapeDesc2D {
    Circle { radius: f32 },
    Rect { half_extents: Vec2 },
    Capsule { half_height: f32, radius: f32 },
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
///
/// Each slot tracks a generation counter so that stale handles can be detected
/// after a body is removed and the slot is reused.
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
// World2D
// ---------------------------------------------------------------------------

/// The main 2D physics world. Add bodies, step the simulation, read results.
pub struct World2D {
    config: SimConfig2D,
    states: Vec<RigidBodyState2D>,
    shapes: Vec<ShapeDesc2D>,
    circles: Vec<CircleData>,
    rects: Vec<RectData>,
    capsules: Vec<CapsuleData2D>,
    allocator: GenerationalIndexAllocator,
    solver: Solver2D,
    persistence: ContactPersistence2D,
    prev_pairs: HashSet<(u32, u32)>,
    events: Vec<CollisionEvent>,
}

impl World2D {
    /// Create a new 2D physics world with the given configuration.
    pub fn new(config: SimConfig2D) -> Self {
        let solver = Solver2D::new(SolverParams {
            iterations: config.solver_iterations,
            ..SolverParams::default()
        });
        Self {
            config,
            states: Vec::new(),
            shapes: Vec::new(),
            circles: Vec::new(),
            rects: Vec::new(),
            capsules: Vec::new(),
            allocator: GenerationalIndexAllocator::new(),
            solver,
            persistence: ContactPersistence2D::new(),
            prev_pairs: HashSet::new(),
            events: Vec::new(),
        }
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

        // Grow storage if needed (slots may be reused via free list).
        if idx >= self.states.len() {
            self.states.resize(idx + 1, RigidBodyState2D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
            self.shapes.resize(idx + 1, ShapeDesc2D::Circle { radius: 0.0 });
        }

        self.states[idx] = state;
        self.shapes[idx] = desc.shape;

        handle
    }

    /// Remove a body by handle. Returns `true` if the body existed and was removed.
    pub fn remove_body(&mut self, handle: BodyHandle) -> bool {
        if !self.allocator.is_alive(handle) {
            return false;
        }
        let idx = handle.index as usize;
        // Zero out the state so it becomes a massless ghost.
        self.states[idx] = RigidBodyState2D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        self.shapes[idx] = ShapeDesc2D::Circle { radius: 0.0 };
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

    /// Advance the simulation by one time step (broadphase -> narrowphase -> solve).
    pub fn step(&mut self) {
        let dt = self.config.dt;
        let gravity = self.config.gravity;

        // Collect indices of live bodies.
        let live_indices: Vec<usize> = (0..self.states.len())
            .filter(|&i| self.allocator.alive.get(i).copied().unwrap_or(false))
            .collect();

        if live_indices.is_empty() {
            return;
        }

        // --- Broadphase: compute AABBs, build LBVH, find pairs ---
        let aabbs: Vec<Aabb2D> = live_indices
            .iter()
            .map(|&i| {
                let s = &self.states[i];
                let pos = s.position();
                let angle = s.angle();
                match &self.shapes[i] {
                    ShapeDesc2D::Circle { radius } => compute_circle_aabb(pos, *radius),
                    ShapeDesc2D::Rect { half_extents } => {
                        compute_rect_aabb(pos, angle, *half_extents)
                    }
                    ShapeDesc2D::Capsule {
                        half_height,
                        radius,
                    } => compute_capsule2d_aabb(pos, angle, *half_height, *radius),
                }
            })
            .collect();

        let lbvh = Lbvh::build(&aabbs);
        let broad_result = lbvh.find_overlapping_pairs(&aabbs);

        // --- Narrowphase: generate contacts for each broad pair ---
        let mut contacts: Vec<Contact2D> = Vec::new();

        for &[ai, bi] in &broad_result.pairs {
            let idx_a = live_indices[ai as usize];
            let idx_b = live_indices[bi as usize];
            let sa = &self.states[idx_a];
            let sb = &self.states[idx_b];
            let pos_a = sa.position();
            let pos_b = sb.position();
            let angle_a = sa.angle();
            let angle_b = sb.angle();

            let new_contacts = match (&self.shapes[idx_a], &self.shapes[idx_b]) {
                (ShapeDesc2D::Circle { radius: ra }, ShapeDesc2D::Circle { radius: rb }) => {
                    circle_circle(pos_a, *ra, pos_b, *rb, idx_a as u32, idx_b as u32)
                }
                (
                    ShapeDesc2D::Circle { radius },
                    ShapeDesc2D::Rect { half_extents },
                ) => circle_rect(
                    pos_a,
                    *radius,
                    pos_b,
                    angle_b,
                    *half_extents,
                    idx_a as u32,
                    idx_b as u32,
                ),
                (
                    ShapeDesc2D::Rect { half_extents },
                    ShapeDesc2D::Circle { radius },
                ) => circle_rect(
                    pos_b,
                    *radius,
                    pos_a,
                    angle_a,
                    *half_extents,
                    idx_b as u32,
                    idx_a as u32,
                ),
                (
                    ShapeDesc2D::Rect { half_extents: ha },
                    ShapeDesc2D::Rect { half_extents: hb },
                ) => rect_rect(
                    pos_a,
                    angle_a,
                    *ha,
                    pos_b,
                    angle_b,
                    *hb,
                    idx_a as u32,
                    idx_b as u32,
                ),
                // Capsule pairs: approximate with circle for now.
                (
                    ShapeDesc2D::Capsule { radius: ra, .. },
                    ShapeDesc2D::Capsule { radius: rb, .. },
                ) => circle_circle(pos_a, *ra, pos_b, *rb, idx_a as u32, idx_b as u32),
                (ShapeDesc2D::Circle { radius }, ShapeDesc2D::Capsule { radius: cr, .. }) => {
                    circle_circle(pos_a, *radius, pos_b, *cr, idx_a as u32, idx_b as u32)
                }
                (ShapeDesc2D::Capsule { radius: cr, .. }, ShapeDesc2D::Circle { radius }) => {
                    circle_circle(pos_a, *cr, pos_b, *radius, idx_a as u32, idx_b as u32)
                }
                (ShapeDesc2D::Rect { half_extents }, ShapeDesc2D::Capsule { radius, .. }) => {
                    circle_rect(
                        pos_b,
                        *radius,
                        pos_a,
                        angle_a,
                        *half_extents,
                        idx_b as u32,
                        idx_a as u32,
                    )
                }
                (ShapeDesc2D::Capsule { radius, .. }, ShapeDesc2D::Rect { half_extents }) => {
                    circle_rect(
                        pos_a,
                        *radius,
                        pos_b,
                        angle_b,
                        *half_extents,
                        idx_a as u32,
                        idx_b as u32,
                    )
                }
            };

            contacts.extend(new_contacts);
        }

        // --- Persistence: warm-start + collision events ---
        let handles: Vec<BodyHandle> = (0..self.states.len())
            .map(|i| {
                let gen = if i < self.allocator.generations.len() {
                    self.allocator.generations[i]
                } else {
                    0
                };
                BodyHandle::new(i as u32, gen)
            })
            .collect();

        let new_events = self.persistence.update(&mut contacts, &handles);
        self.events.extend(new_events);

        // --- Solver ---
        self.solver
            .solve(dt, gravity, &mut self.states, &mut contacts);
    }

    /// Drain all collision events accumulated since the last drain.
    pub fn drain_events(&mut self) -> Vec<CollisionEvent> {
        std::mem::take(&mut self.events)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world2d_new() {
        let world = World2D::new(SimConfig2D::default());
        assert_eq!(world.body_count(), 0);
        assert!(world.states.is_empty());
        assert_eq!(world.config.gravity, Vec2::new(0.0, -9.81));
        assert!((world.config.dt - 1.0 / 60.0).abs() < 1e-6);
        assert_eq!(world.config.solver_iterations, 5);
        assert_eq!(world.config.max_bodies, 65536);
    }

    #[test]
    fn test_add_remove_body() {
        let mut world = World2D::new(SimConfig2D::default());

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

        // Double remove should fail.
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

        // Deallocate h0, then reallocate. Should reuse index 0 with generation 1.
        assert!(alloc.deallocate(h0));
        assert!(!alloc.is_alive(h0));

        let h2 = alloc.allocate();
        assert_eq!(h2.index, 0);
        assert_eq!(h2.generation, 1);
        assert!(alloc.is_alive(h2));

        // Old handle h0 is stale.
        assert!(!alloc.is_alive(h0));

        assert_eq!(alloc.live_count(), 2);
    }

    #[test]
    fn test_gravity_fall_2d() {
        let mut world = World2D::new(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            max_bodies: 1024,
        });

        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 10.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });

        let steps = 60;
        for _ in 0..steps {
            world.step();
        }

        let pos = world.get_position(h).unwrap();
        // After 1 second of free fall: y = 10 - 0.5 * 9.81 * 1^2 = ~5.095
        let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
        let error = (pos.y - expected_y).abs();
        assert!(
            error < 0.5,
            "Expected y ~ {expected_y}, got {}, error {error}",
            pos.y
        );
        // X should stay at 0.
        assert!(pos.x.abs() < 1e-6, "X should remain ~0, got {}", pos.x);
    }

    #[test]
    fn test_two_body_collision_2d() {
        let mut world = World2D::new(SimConfig2D {
            gravity: Vec2::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
            max_bodies: 1024,
        });

        // Two circles approaching each other along x.
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

        // Step enough for them to meet and interact.
        for _ in 0..30 {
            world.step();
        }

        let pos_a = world.get_position(h_a).unwrap();
        let pos_b = world.get_position(h_b).unwrap();

        // After collision, they should not be overlapping heavily.
        let dist = (pos_b.x - pos_a.x).abs();
        assert!(
            dist >= 1.5,
            "Bodies should be separated (dist={dist}), not heavily overlapping"
        );
    }

    #[test]
    fn test_body_count() {
        let mut world = World2D::new(SimConfig2D::default());
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
}

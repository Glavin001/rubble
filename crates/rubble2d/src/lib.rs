//! `rubble2d` — Public API facade for the rubble 2D GPU physics engine.
//!
//! All physics simulation runs entirely on the GPU via WGSL compute shaders.
//! The pipeline: predict → AABB → broadphase → narrowphase → AVBD solver → velocity extraction.

pub mod gpu;

use glam::{Vec2, Vec4};
use rubble_math::{BodyHandle, RigidBodyState2D};
use rubble_shapes2d::{CircleData, ConvexPolygonData, ConvexVertex2D, RectData};

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
// World2D
// ---------------------------------------------------------------------------

/// The main 2D physics world. All simulation runs on the GPU.
pub struct World2D {
    config: SimConfig2D,
    states: Vec<RigidBodyState2D>,
    shapes: Vec<ShapeDesc2D>,
    allocator: GenerationalIndexAllocator,
    gpu_pipeline: gpu::GpuPipeline2D,
}

impl World2D {
    /// Create a new 2D physics world backed by GPU compute shaders.
    ///
    /// Returns an error if no GPU adapter is available.
    pub fn new(config: SimConfig2D) -> Result<Self, rubble_gpu::GpuError> {
        let ctx = pollster::block_on(rubble_gpu::GpuContext::new())?;
        let pipeline = gpu::GpuPipeline2D::new(ctx, config.max_bodies);
        Ok(Self {
            config,
            states: Vec::new(),
            shapes: Vec::new(),
            allocator: GenerationalIndexAllocator::new(),
            gpu_pipeline: pipeline,
        })
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

        if idx >= self.states.len() {
            self.states.resize(
                idx + 1,
                RigidBodyState2D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            );
            self.shapes
                .resize(idx + 1, ShapeDesc2D::Circle { radius: 0.0 });
        }

        self.states[idx] = state;
        self.shapes[idx] = desc.shape.clone();

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
        let compact_states: Vec<RigidBodyState2D> =
            alive_indices.iter().map(|&i| self.states[i]).collect();

        let mut shape_info_data: Vec<gpu::ShapeInfo> = Vec::with_capacity(alive_indices.len());
        let mut gpu_circles: Vec<CircleData> = Vec::new();
        let mut gpu_rects: Vec<RectData> = Vec::new();
        let mut gpu_convex_polys: Vec<ConvexPolygonData> = Vec::new();
        let mut gpu_convex_verts: Vec<ConvexVertex2D> = Vec::new();

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

        let num_bodies = compact_states.len() as u32;

        self.gpu_pipeline.upload(
            &compact_states,
            &shape_info_data,
            &gpu_circles,
            &gpu_rects,
            &gpu_convex_polys,
            &gpu_convex_verts,
            self.config.gravity,
            self.config.dt,
            self.config.solver_iterations,
        );

        let results = self
            .gpu_pipeline
            .step(num_bodies, self.config.solver_iterations);

        for (slot, &orig) in alive_indices.iter().enumerate() {
            if slot < results.len() {
                self.states[orig] = results[slot];
            }
        }
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
}

//! `rubble3d` — Public API facade for the rubble 3D GPU physics engine.
//!
//! All simulation runs on the GPU via WGSL compute shaders (AVBD solver):
//! predict → AABB compute → broadphase → narrowphase → solver → velocity extraction.

pub mod gpu;

use glam::{Mat3, Quat, Vec3, Vec4};
use rubble_math::{
    BodyHandle, RigidBodyProps3D, RigidBodyState3D, FLAG_STATIC, SHAPE_BOX, SHAPE_SPHERE,
};
use rubble_shapes3d::{BoxData, SphereData};

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
///
/// Create with [`World::new`] which initializes a GPU compute pipeline.
/// If no GPU adapter is available, returns an error.
pub struct World {
    config: SimConfig,
    states: Vec<RigidBodyState3D>,
    props: Vec<RigidBodyProps3D>,
    shapes: Vec<ShapeDesc>,
    spheres: Vec<SphereData>,
    boxes: Vec<BoxData>,
    allocator: GenerationalIndexAllocator,
    alive: Vec<bool>,
    gpu_pipeline: gpu::GpuPipeline,
}

impl World {
    /// Create a new GPU-accelerated physics world.
    ///
    /// Initializes a GPU context and compiles all compute shaders.
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
            allocator: GenerationalIndexAllocator::new(),
            alive: Vec::new(),
            gpu_pipeline: pipeline,
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
        };

        let prop =
            RigidBodyProps3D::new(inv_inertia, desc.friction, shape_type, shape_index, flags);

        // Ensure arrays are large enough for this index.
        if idx >= self.states.len() {
            self.states.resize(idx + 1, bytemuck::Zeroable::zeroed());
            self.props.resize(idx + 1, bytemuck::Zeroable::zeroed());
            self.shapes
                .resize(idx + 1, ShapeDesc::Sphere { radius: 0.5 });
            self.alive.resize(idx + 1, false);
        }

        self.states[idx] = state;
        self.props[idx] = prop;
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

        // Build compact arrays for GPU upload
        let compact_states: Vec<RigidBodyState3D> =
            alive_indices.iter().map(|&i| self.states[i]).collect();
        let compact_props: Vec<RigidBodyProps3D> =
            alive_indices.iter().map(|&i| self.props[i]).collect();
        let num_bodies = compact_states.len() as u32;

        // Upload body states, props, and shape data to GPU.
        self.gpu_pipeline.upload(
            &compact_states,
            &compact_props,
            &self.spheres,
            &self.boxes,
            self.config.gravity,
            self.config.dt,
            self.config.solver_iterations,
        );

        // Run GPU step
        let results = self
            .gpu_pipeline
            .step(num_bodies, self.config.solver_iterations);

        // Write results back to original arrays
        for (slot, &orig) in alive_indices.iter().enumerate() {
            if slot < results.len() {
                self.states[orig] = results[slot];
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

    fn gpu_world(config: SimConfig) -> World {
        World::new(config).expect("GPU adapter required for tests")
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
        let mut world = gpu_world(SimConfig {
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

        for _ in 0..60 {
            world.step();
        }

        let pos = world.get_position(handle).unwrap();
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
        let mut world = gpu_world(SimConfig {
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

        for _ in 0..120 {
            world.step();
        }

        let p1 = world.get_position(h1).unwrap();
        let p2 = world.get_position(h2).unwrap();
        let dist = (p2 - p1).length();

        assert!(
            dist >= 1.5,
            "Bodies should not overlap significantly: distance = {}",
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

//! WebAssembly bindings for the Rubble GPU physics engine.
//!
//! Exposes `PhysicsWorld2D` and `PhysicsWorld3D` to JavaScript via wasm-bindgen.
//! GPU compute shaders run via WebGPU in the browser.

use glam::{Quat, Vec2, Vec3};
use rubble_math::BodyHandle;
use wasm_bindgen::prelude::*;

// Pull in getrandom's wasm_js feature for WebGPU randomness support.
use getrandom as _;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// PhysicsWorld2D
// ---------------------------------------------------------------------------

/// 2D GPU physics world exposed to JavaScript.
#[wasm_bindgen]
pub struct PhysicsWorld2D {
    world: rubble2d::World2D,
    /// Ordered list of live body handles (index into world).
    handles: Vec<BodyHandle>,
    /// Shape type per handle: 0=circle, 1=rect, 2=polygon, 3=capsule
    shape_types: Vec<u32>,
    /// Shape size info per handle: [radius] for circles, [hw,hh] for rects, etc.
    shape_sizes: Vec<f32>,
    /// Offset into shape_sizes for each body.
    shape_size_offsets: Vec<usize>,
}

#[wasm_bindgen]
impl PhysicsWorld2D {
    /// Create a new 2D physics world with the given gravity and time step.
    /// Async because GPU initialization requires WebGPU adapter negotiation.
    /// Usage: `const world = await PhysicsWorld2D.create(0, -9.81, 1/60);`
    pub async fn create(
        gravity_x: f32,
        gravity_y: f32,
        dt: f32,
    ) -> Result<PhysicsWorld2D, JsError> {
        let config = rubble2d::SimConfig2D {
            gravity: Vec2::new(gravity_x, gravity_y),
            dt,
            ..rubble2d::SimConfig2D::default()
        };
        let world = rubble2d::World2D::new_async(config)
            .await
            .map_err(|e| JsError::new(&format!("GPU init failed: {e}")))?;
        Ok(Self {
            world,
            handles: Vec::new(),
            shape_types: Vec::new(),
            shape_sizes: Vec::new(),
            shape_size_offsets: Vec::new(),
        })
    }

    /// Add a dynamic circle body. Returns handle index.
    pub fn add_circle(&mut self, x: f32, y: f32, radius: f32, mass: f32) -> u32 {
        let handle = self.world.add_body(&rubble2d::RigidBodyDesc2D {
            x,
            y,
            mass,
            shape: rubble2d::ShapeDesc2D::Circle { radius },
            ..rubble2d::RigidBodyDesc2D::default()
        });
        let idx = self.handles.len() as u32;
        self.handles.push(handle);
        self.shape_types.push(0);
        self.shape_size_offsets.push(self.shape_sizes.len());
        self.shape_sizes.push(radius);
        idx
    }

    /// Add a dynamic rectangle body. Returns handle index.
    pub fn add_rect(
        &mut self,
        x: f32,
        y: f32,
        half_w: f32,
        half_h: f32,
        angle: f32,
        mass: f32,
    ) -> u32 {
        let handle = self.world.add_body(&rubble2d::RigidBodyDesc2D {
            x,
            y,
            angle,
            mass,
            shape: rubble2d::ShapeDesc2D::Rect {
                half_extents: Vec2::new(half_w, half_h),
            },
            ..rubble2d::RigidBodyDesc2D::default()
        });
        let idx = self.handles.len() as u32;
        self.handles.push(handle);
        self.shape_types.push(1);
        self.shape_size_offsets.push(self.shape_sizes.len());
        self.shape_sizes.push(half_w);
        self.shape_sizes.push(half_h);
        idx
    }

    /// Add a static (immovable) rectangle. Returns handle index.
    pub fn add_static_rect(&mut self, x: f32, y: f32, half_w: f32, half_h: f32, angle: f32) -> u32 {
        self.add_rect(x, y, half_w, half_h, angle, 0.0)
    }

    /// Add a dynamic capsule body. Returns handle index.
    pub fn add_capsule(&mut self, x: f32, y: f32, half_height: f32, radius: f32, mass: f32) -> u32 {
        let handle = self.world.add_body(&rubble2d::RigidBodyDesc2D {
            x,
            y,
            mass,
            shape: rubble2d::ShapeDesc2D::Capsule {
                half_height,
                radius,
            },
            ..rubble2d::RigidBodyDesc2D::default()
        });
        let idx = self.handles.len() as u32;
        self.handles.push(handle);
        self.shape_types.push(3);
        self.shape_size_offsets.push(self.shape_sizes.len());
        self.shape_sizes.push(half_height);
        self.shape_sizes.push(radius);
        idx
    }

    /// Remove a body by handle index.
    pub fn remove_body(&mut self, idx: u32) -> bool {
        let i = idx as usize;
        if i >= self.handles.len() {
            return false;
        }
        self.world.remove_body(self.handles[i])
    }

    /// Step the simulation forward by one time step.
    pub async fn step(&mut self) {
        #[cfg(target_arch = "wasm32")]
        self.world.step_async().await;
        #[cfg(not(target_arch = "wasm32"))]
        self.world.step();
    }

    /// Number of live bodies.
    pub fn body_count(&self) -> u32 {
        self.world.body_count() as u32
    }

    /// Get all body positions as a flat array: [x0, y0, x1, y1, ...].
    pub fn get_positions(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.handles.len() * 2);
        for h in &self.handles {
            if let Some(p) = self.world.get_position(*h) {
                out.push(p.x);
                out.push(p.y);
            } else {
                out.push(0.0);
                out.push(0.0);
            }
        }
        out
    }

    /// Get all body rotation angles as a flat array: [a0, a1, ...].
    pub fn get_angles(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.handles.len());
        for h in &self.handles {
            out.push(self.world.get_angle(*h).unwrap_or(0.0));
        }
        out
    }

    /// Get shape types for all bodies: 0=circle, 1=rect, 2=polygon, 3=capsule.
    pub fn get_shape_types(&self) -> Vec<u32> {
        self.shape_types.clone()
    }

    /// Get shape sizes as flat array. For circles: [radius], for rects: [hw, hh].
    /// Use `get_shape_size_offsets()` to index into this array per body.
    pub fn get_shape_sizes(&self) -> Vec<f32> {
        self.shape_sizes.clone()
    }

    /// Get the offset into shape_sizes for each body.
    pub fn get_shape_size_offsets(&self) -> Vec<u32> {
        self.shape_size_offsets.iter().map(|&o| o as u32).collect()
    }

    /// Total number of handles (including removed bodies — indices are stable).
    pub fn handle_count(&self) -> u32 {
        self.handles.len() as u32
    }

    /// Per-phase wall-clock timings (ms) from the last `step()` call.
    ///
    /// Returns 7 floats in fixed order:
    ///   [0] upload, [1] predict+aabb, [2] broadphase, [3] narrowphase,
    ///   [4] contact_fetch, [5] solve, [6] extract
    pub fn last_step_timings_ms(&self) -> Vec<f32> {
        self.world.last_step_timings().as_array().to_vec()
    }
}

// ---------------------------------------------------------------------------
// PhysicsWorld3D
// ---------------------------------------------------------------------------

/// 3D GPU physics world exposed to JavaScript.
#[wasm_bindgen]
pub struct PhysicsWorld3D {
    world: rubble3d::World,
    handles: Vec<BodyHandle>,
    /// Shape type per handle: 0=sphere, 1=box, 2=capsule
    shape_types: Vec<u32>,
    shape_sizes: Vec<f32>,
    shape_size_offsets: Vec<usize>,
}

#[wasm_bindgen]
impl PhysicsWorld3D {
    /// Create a new 3D physics world with the given gravity and time step.
    /// Async because GPU initialization requires WebGPU adapter negotiation.
    /// Usage: `const world = await PhysicsWorld3D.create(0, -9.81, 0, 1/60);`
    pub async fn create(
        gravity_x: f32,
        gravity_y: f32,
        gravity_z: f32,
        dt: f32,
    ) -> Result<PhysicsWorld3D, JsError> {
        let config = rubble3d::SimConfig {
            gravity: Vec3::new(gravity_x, gravity_y, gravity_z),
            dt,
            ..rubble3d::SimConfig::default()
        };
        let world = rubble3d::World::new_async(config)
            .await
            .map_err(|e| JsError::new(&format!("GPU init failed: {e}")))?;
        Ok(Self {
            world,
            handles: Vec::new(),
            shape_types: Vec::new(),
            shape_sizes: Vec::new(),
            shape_size_offsets: Vec::new(),
        })
    }

    /// Add a dynamic sphere. Returns handle index.
    pub fn add_sphere(&mut self, x: f32, y: f32, z: f32, radius: f32, mass: f32) -> u32 {
        let handle = self.world.add_body(&rubble3d::RigidBodyDesc {
            position: Vec3::new(x, y, z),
            mass,
            shape: rubble3d::ShapeDesc::Sphere { radius },
            ..rubble3d::RigidBodyDesc::default()
        });
        let idx = self.handles.len() as u32;
        self.handles.push(handle);
        self.shape_types.push(0);
        self.shape_size_offsets.push(self.shape_sizes.len());
        self.shape_sizes.push(radius);
        idx
    }

    /// Add a dynamic box. Returns handle index.
    #[allow(clippy::too_many_arguments)]
    pub fn add_box(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        half_w: f32,
        half_h: f32,
        half_d: f32,
        mass: f32,
    ) -> u32 {
        let handle = self.world.add_body(&rubble3d::RigidBodyDesc {
            position: Vec3::new(x, y, z),
            mass,
            shape: rubble3d::ShapeDesc::Box {
                half_extents: Vec3::new(half_w, half_h, half_d),
            },
            ..rubble3d::RigidBodyDesc::default()
        });
        let idx = self.handles.len() as u32;
        self.handles.push(handle);
        self.shape_types.push(1);
        self.shape_size_offsets.push(self.shape_sizes.len());
        self.shape_sizes.push(half_w);
        self.shape_sizes.push(half_h);
        self.shape_sizes.push(half_d);
        idx
    }

    /// Add a dynamic capsule. Returns handle index.
    pub fn add_capsule(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        half_height: f32,
        radius: f32,
        mass: f32,
    ) -> u32 {
        let handle = self.world.add_body(&rubble3d::RigidBodyDesc {
            position: Vec3::new(x, y, z),
            mass,
            shape: rubble3d::ShapeDesc::Capsule {
                half_height,
                radius,
            },
            ..rubble3d::RigidBodyDesc::default()
        });
        let idx = self.handles.len() as u32;
        self.handles.push(handle);
        self.shape_types.push(2);
        self.shape_size_offsets.push(self.shape_sizes.len());
        self.shape_sizes.push(half_height);
        self.shape_sizes.push(radius);
        idx
    }

    /// Add a static (immovable) box. Returns handle index.
    pub fn add_static_box(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        half_w: f32,
        half_h: f32,
        half_d: f32,
    ) -> u32 {
        self.add_box(x, y, z, half_w, half_h, half_d, 0.0)
    }

    /// Add a static ground plane.
    pub fn add_ground_plane(&mut self, y: f32) {
        let handle = self.world.add_body(&rubble3d::RigidBodyDesc {
            position: Vec3::ZERO,
            mass: 0.0,
            shape: rubble3d::ShapeDesc::Plane {
                normal: Vec3::Y,
                distance: y,
            },
            ..rubble3d::RigidBodyDesc::default()
        });
        self.handles.push(handle);
        self.shape_types.push(99); // sentinel for plane — not rendered
        self.shape_size_offsets.push(self.shape_sizes.len());
    }

    /// Remove a body by handle index.
    pub fn remove_body(&mut self, idx: u32) -> bool {
        let i = idx as usize;
        if i >= self.handles.len() {
            return false;
        }
        self.world.remove_body(self.handles[i])
    }

    /// Step the simulation forward by one time step.
    pub async fn step(&mut self) {
        #[cfg(target_arch = "wasm32")]
        self.world.step_async().await;
        #[cfg(not(target_arch = "wasm32"))]
        self.world.step();
    }

    /// Number of live bodies.
    pub fn body_count(&self) -> u32 {
        self.world.body_count() as u32
    }

    /// Get all body transforms as flat array:
    /// [x, y, z, qx, qy, qz, qw, x, y, z, qx, qy, qz, qw, ...] (7 floats per body).
    pub fn get_transforms(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.handles.len() * 7);
        for h in &self.handles {
            let pos = self.world.get_position(*h).unwrap_or(Vec3::ZERO);
            let rot = self.world.get_rotation(*h).unwrap_or(Quat::IDENTITY);
            out.push(pos.x);
            out.push(pos.y);
            out.push(pos.z);
            out.push(rot.x);
            out.push(rot.y);
            out.push(rot.z);
            out.push(rot.w);
        }
        out
    }

    /// Get shape types for all bodies: 0=sphere, 1=box, 2=capsule, 99=plane.
    pub fn get_shape_types(&self) -> Vec<u32> {
        self.shape_types.clone()
    }

    /// Get shape sizes as flat array. For spheres: [radius], for boxes: [hw, hh, hd].
    pub fn get_shape_sizes(&self) -> Vec<f32> {
        self.shape_sizes.clone()
    }

    /// Get the offset into shape_sizes for each body.
    pub fn get_shape_size_offsets(&self) -> Vec<u32> {
        self.shape_size_offsets.iter().map(|&o| o as u32).collect()
    }

    /// Total number of handles.
    pub fn handle_count(&self) -> u32 {
        self.handles.len() as u32
    }

    /// Per-phase wall-clock timings (ms) from the last `step()` call.
    ///
    /// Returns 7 floats in fixed order:
    ///   [0] upload, [1] predict+aabb, [2] broadphase, [3] narrowphase,
    ///   [4] contact_fetch, [5] solve, [6] extract
    pub fn last_step_timings_ms(&self) -> Vec<f32> {
        self.world.last_step_timings().as_array().to_vec()
    }
}

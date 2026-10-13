//! Shared host/GPU ABI for the AVBD engine.
//!
//! Units: meters, kilograms, seconds, radians.
//! Frame: right-handed Y-up. Quaternions are `(x, y, z, w)`.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

pub mod commands;
pub mod handles;
pub mod params;

pub use commands::*;
pub use handles::*;
pub use params::*;

/// ABI probe used by Gate 0 layout round-trips (must match `slang/kernels/abi_probe.slang`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AbiProbe {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub inertia_col0: [f32; 4],
    pub inertia_col1: [f32; 4],
    pub inertia_col2: [f32; 4],
    pub counter: u32,
    pub _pad1: [u32; 3],
}

impl AbiProbe {
    pub fn new(position: Vec3, inertia: Mat3, counter: u32) -> Self {
        let c0 = inertia.col(0);
        let c1 = inertia.col(1);
        let c2 = inertia.col(2);
        Self {
            position: position.to_array(),
            _pad0: 0.0,
            inertia_col0: [c0.x, c0.y, c0.z, 0.0],
            inertia_col1: [c1.x, c1.y, c1.z, 0.0],
            inertia_col2: [c2.x, c2.y, c2.z, 0.0],
            counter,
            _pad1: [0; 3],
        }
    }

    pub fn position(&self) -> Vec3 {
        Vec3::from_array(self.position)
    }

    pub fn inertia(&self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.inertia_col0[0], self.inertia_col0[1], self.inertia_col0[2]),
            Vec3::new(self.inertia_col1[0], self.inertia_col1[1], self.inertia_col1[2]),
            Vec3::new(self.inertia_col2[0], self.inertia_col2[1], self.inertia_col2[2]),
        )
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyFlags {
    None = 0,
    Dynamic = 1,
    Kinematic = 2,
    Sleeping = 4,
    Awake = 8,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeType {
    Sphere = 0,
    Capsule = 1,
    Box = 2,
    ConvexHull = 3,
    Plane = 4,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointType {
    Ball = 0,
    Fixed = 1,
    Hinge = 2,
    Slider = 3,
    D6 = 4,
    Distance = 5,
    MaxDistance = 6,
}

/// Hot body state (SoA-friendly AoS record, 80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BodyState {
    pub position: [f32; 3],
    pub inv_mass: f32,
    pub orientation: [f32; 4],
    pub inv_inertia: [f32; 3],
    pub flags: u32,
    pub linear_vel: [f32; 3],
    pub sleep_timer: f32,
    pub angular_vel: [f32; 3],
    pub generation: u32,
}

impl BodyState {
    pub fn static_plane_support() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0, 0.0, 0.0],
            flags: BodyFlags::None as u32,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            // 0 matches three-avbd; reuseContacts rejects when contact.generated < body.generation.
            generation: 0,
        }
    }

    pub fn dynamic(position: Vec3, orientation: Quat, mass: f32, inertia: Vec3) -> Self {
        let inv_m = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position: position.to_array(),
            inv_mass: inv_m,
            orientation: orientation.to_array(),
            inv_inertia: [
                if inertia.x > 0.0 { 1.0 / inertia.x } else { 0.0 },
                if inertia.y > 0.0 { 1.0 / inertia.y } else { 0.0 },
                if inertia.z > 0.0 { 1.0 / inertia.z } else { 0.0 },
            ],
            flags: BodyFlags::Dynamic as u32 | BodyFlags::Awake as u32,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
        }
    }

    pub fn is_dynamic(&self) -> bool {
        self.inv_mass > 0.0 && (self.flags & BodyFlags::Dynamic as u32) != 0
    }

    pub fn is_sleeping(&self) -> bool {
        (self.flags & BodyFlags::Sleeping as u32) != 0
    }

    pub fn quat(&self) -> Quat {
        Quat::from_array(self.orientation)
    }
}

/// Collider attached to a body (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Collider {
    pub body: u32,
    pub shape: u32,
    pub material: u32,
    pub generation: u32,
    pub local_pos: [f32; 3],
    pub density: f32,
    pub local_rot: [f32; 4],
    /// Sphere: radius; Capsule: radius, half_length; Box: half extents; Plane: normal.xyz + offset.w
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Material {
    pub friction: f32,
    pub restitution: f32,
    pub contact_offset: f32,
    pub _pad: f32,
}

/// Persistent contact row (128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Contact {
    pub collider_a: u32,
    pub collider_b: u32,
    pub feature_id: u32,
    pub alive: u32,
    pub normal: [f32; 3],
    pub penetration: f32,
    pub anchor_a: [f32; 3],
    pub c0_n: f32,
    pub anchor_b: [f32; 3],
    pub friction: f32,
    pub tangent1: [f32; 3],
    pub lambda_n: f32,
    pub tangent2: [f32; 3],
    pub lambda_t1: f32,
    pub k_n: f32,
    pub k_t: f32,
    pub lambda_t2: f32,
    pub c0_t1: f32,
    pub c0_t2: f32,
    pub _pad: [f32; 3],
}

/// Contact pair manifold (32 bytes) — three-avbd Stage 8b.
/// CSR / dual / primal adjacency key one edge per manifold, not per contact point.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Manifold {
    pub collider_a: u32,
    pub collider_b: u32,
    pub contact_start: u32,
    pub contact_count: u32,
    pub normal: [f32; 3],
    pub friction: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Joint {
    pub body_a: u32,
    pub body_b: u32,
    pub joint_type: u32,
    pub flags: u32,
    pub local_anchor_a: [f32; 3],
    pub stiffness: f32,
    pub local_anchor_b: [f32; 3],
    pub damping: f32,
    pub axis_a: [f32; 3],
    pub limit_min: f32,
    pub axis_b: [f32; 3],
    pub limit_max: f32,
    /// Constraint value at beginning of timestep (demo3d `C0Lin`).
    pub c0: [f32; 3],
    pub _pad_c0: f32,
    pub lambda: [f32; 6],
    pub k: [f32; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct StepCounters {
    pub body_count: u32,
    pub collider_count: u32,
    pub contact_count: u32,
    pub joint_count: u32,
    pub candidate_count: u32,
    pub awake_count: u32,
    pub color_count: u32,
    pub overflow_flags: u32,
    pub pivot_floor_hits: u32,
    pub sleep_events: u32,
    pub wake_events: u32,
    pub iteration: u32,
    pub manifold_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl Default for StepCounters {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CandidatePair {
    pub collider_a: u32,
    pub collider_b: u32,
    pub key: u32,
    pub _pad: u32,
}

pub const OVERFLOW_CONTACTS: u32 = 1;
pub const OVERFLOW_CANDIDATES: u32 = 2;
pub const OVERFLOW_JOINTS: u32 = 4;
pub const OVERFLOW_COLORS: u32 = 8;
pub const CCD_BUDGET_EXHAUSTED: u32 = 16;
pub const CCD_CANDIDATE_MISS: u32 = 32;
pub const OVERFLOW_MANIFOLDS: u32 = 64;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn abi_sizes() {
        assert_eq!(size_of::<AbiProbe>(), 80);
        assert_eq!(size_of::<BodyState>(), 80);
        assert_eq!(size_of::<Collider>(), 64);
        assert_eq!(size_of::<Contact>(), 128);
        assert_eq!(size_of::<Manifold>(), 32);
        assert_eq!(size_of::<StepCounters>(), 64);
        assert_eq!(align_of::<BodyState>(), 4);
    }
}

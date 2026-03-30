use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Quat, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RigidBodyState3D {
    pub position_inv_mass: Vec4,
    pub orientation: Quat,
    pub lin_vel: Vec4,
    pub ang_vel: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct RigidBodyState2D {
    pub position_inv_mass: Vec4,
    pub lin_vel: Vec4,
    pub pad0: Vec4,
    pub pad1: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct RigidBodyProps3D {
    pub inv_inertia_local_col0: Vec4,
    pub inv_inertia_local_col1: Vec4,
    pub inv_inertia_local_col2: Vec4,
    pub friction: f32,
    pub shape_type: u32,
    pub shape_index: u32,
    pub flags: u32,
}

impl Default for RigidBodyProps3D {
    fn default() -> Self {
        Self {
            inv_inertia_local_col0: Vec4::new(1.0, 0.0, 0.0, 0.0),
            inv_inertia_local_col1: Vec4::new(0.0, 1.0, 0.0, 0.0),
            inv_inertia_local_col2: Vec4::new(0.0, 0.0, 1.0, 0.0),
            friction: 0.5,
            shape_type: 0,
            shape_index: 0,
            flags: 0,
        }
    }
}

impl RigidBodyProps3D {
    pub fn from_inv_inertia(
        inv: Mat3,
        friction: f32,
        shape_type: u32,
        shape_index: u32,
        flags: u32,
    ) -> Self {
        let cols = inv.to_cols_array_2d();
        Self {
            inv_inertia_local_col0: Vec4::new(cols[0][0], cols[0][1], cols[0][2], 0.0),
            inv_inertia_local_col1: Vec4::new(cols[1][0], cols[1][1], cols[1][2], 0.0),
            inv_inertia_local_col2: Vec4::new(cols[2][0], cols[2][1], cols[2][2], 0.0),
            friction,
            shape_type,
            shape_index,
            flags,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Contact3D {
    pub point: Vec4,
    pub normal: Vec4,
    pub body_a: u32,
    pub body_b: u32,
    pub feature_id: u32,
    pub pad: u32,
    pub lambda_n: f32,
    pub lambda_t1: f32,
    pub lambda_t2: f32,
    pub penalty_k: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Aabb3D {
    pub min: Vec4,
    pub max: Vec4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyHandle {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionEvent {
    Started {
        body_a: BodyHandle,
        body_b: BodyHandle,
    },
    Ended {
        body_a: BodyHandle,
        body_b: BodyHandle,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_layout_contract() {
        assert_eq!(std::mem::size_of::<RigidBodyState3D>(), 64);
        assert_eq!(std::mem::size_of::<Contact3D>(), 64);
    }
}

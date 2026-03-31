//! Pure data definitions for the rubble GPU rigid body physics engine.
//!
//! All GPU-facing structs are `#[repr(C)]` and derive `bytemuck::Pod + bytemuck::Zeroable`.
//! Vec3 fields are padded to Vec4 for WGSL 16-byte alignment.

use bytemuck::{Pod, Zeroable};
use glam::Vec4;

// ---------------------------------------------------------------------------
// Body flags
// ---------------------------------------------------------------------------

pub const FLAG_STATIC: u32 = 1 << 0;
pub const FLAG_KINEMATIC: u32 = 1 << 1;

// ---------------------------------------------------------------------------
// Shape type constants
// ---------------------------------------------------------------------------

pub const SHAPE_SPHERE: u32 = 0;
pub const SHAPE_BOX: u32 = 1;
pub const SHAPE_CAPSULE: u32 = 2;
pub const SHAPE_CONVEX_HULL: u32 = 3;
pub const SHAPE_PLANE: u32 = 4;
pub const SHAPE_COMPOUND: u32 = 5;

// ---------------------------------------------------------------------------
// 3D rigid body state -- 4 x vec4 = 64 bytes
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RigidBodyState3D {
    /// (x, y, z, 1/m)
    pub position_inv_mass: Vec4,
    /// Orientation quaternion stored as (x, y, z, w)
    pub orientation: Vec4,
    /// (vx, vy, vz, 0)
    pub lin_vel: Vec4,
    /// (wx, wy, wz, 0)
    pub ang_vel: Vec4,
}

impl RigidBodyState3D {
    pub fn new(
        position: glam::Vec3,
        inv_mass: f32,
        orientation: glam::Quat,
        lin_vel: glam::Vec3,
        ang_vel: glam::Vec3,
    ) -> Self {
        Self {
            position_inv_mass: Vec4::new(position.x, position.y, position.z, inv_mass),
            orientation: Vec4::new(orientation.x, orientation.y, orientation.z, orientation.w),
            lin_vel: lin_vel.extend(0.0),
            ang_vel: ang_vel.extend(0.0),
        }
    }

    #[inline]
    pub fn position(&self) -> glam::Vec3 {
        self.position_inv_mass.truncate()
    }

    #[inline]
    pub fn inv_mass(&self) -> f32 {
        self.position_inv_mass.w
    }

    #[inline]
    pub fn quat(&self) -> glam::Quat {
        glam::Quat::from_xyzw(
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w,
        )
    }

    #[inline]
    pub fn linear_velocity(&self) -> glam::Vec3 {
        self.lin_vel.truncate()
    }

    #[inline]
    pub fn angular_velocity(&self) -> glam::Vec3 {
        self.ang_vel.truncate()
    }
}

// ---------------------------------------------------------------------------
// 2D rigid body state -- 4 x vec4 = 64 bytes (padded)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RigidBodyState2D {
    /// (x, y, angle, 1/m)
    pub position_inv_mass: Vec4,
    /// (vx, vy, angular_vel, 0)
    pub lin_vel: Vec4,
    pub _pad0: Vec4,
    pub _pad1: Vec4,
}

impl RigidBodyState2D {
    pub fn new(
        x: f32,
        y: f32,
        angle: f32,
        inv_mass: f32,
        vx: f32,
        vy: f32,
        angular_vel: f32,
    ) -> Self {
        Self {
            position_inv_mass: Vec4::new(x, y, angle, inv_mass),
            lin_vel: Vec4::new(vx, vy, angular_vel, 0.0),
            _pad0: Vec4::ZERO,
            _pad1: Vec4::ZERO,
        }
    }

    #[inline]
    pub fn position(&self) -> glam::Vec2 {
        glam::Vec2::new(self.position_inv_mass.x, self.position_inv_mass.y)
    }

    #[inline]
    pub fn angle(&self) -> f32 {
        self.position_inv_mass.z
    }

    #[inline]
    pub fn inv_mass(&self) -> f32 {
        self.position_inv_mass.w
    }

    #[inline]
    pub fn linear_velocity(&self) -> glam::Vec2 {
        glam::Vec2::new(self.lin_vel.x, self.lin_vel.y)
    }

    #[inline]
    pub fn angular_velocity(&self) -> f32 {
        self.lin_vel.z
    }
}

// ---------------------------------------------------------------------------
// Static properties (3D)
//
// The inverse inertia tensor is stored as 3 rows, each padded to vec4 for
// GPU alignment. Total: 3*16 + 4*4 + 4 = 68 -> rounds to 80 with padding.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RigidBodyProps3D {
    /// Row 0 of the 3x3 inverse inertia tensor (local frame), padded to vec4.
    pub inv_inertia_row0: Vec4,
    /// Row 1.
    pub inv_inertia_row1: Vec4,
    /// Row 2.
    pub inv_inertia_row2: Vec4,
    /// Friction coefficient.
    pub friction: f32,
    /// Shape type (see `SHAPE_*` constants).
    pub shape_type: u32,
    /// Index into the shape-specific storage buffer.
    pub shape_index: u32,
    /// Bitfield of `FLAG_*` constants.
    pub flags: u32,
}

impl RigidBodyProps3D {
    pub fn new(
        inv_inertia: glam::Mat3,
        friction: f32,
        shape_type: u32,
        shape_index: u32,
        flags: u32,
    ) -> Self {
        let cols = inv_inertia.to_cols_array_2d();
        // glam Mat3 is column-major; we store each column padded to vec4.
        Self {
            inv_inertia_row0: Vec4::new(cols[0][0], cols[0][1], cols[0][2], 0.0),
            inv_inertia_row1: Vec4::new(cols[1][0], cols[1][1], cols[1][2], 0.0),
            inv_inertia_row2: Vec4::new(cols[2][0], cols[2][1], cols[2][2], 0.0),
            friction,
            shape_type,
            shape_index,
            flags,
        }
    }

    /// Reconstruct the inverse inertia tensor as a `glam::Mat3`.
    pub fn inv_inertia(&self) -> glam::Mat3 {
        glam::Mat3::from_cols(
            self.inv_inertia_row0.truncate(),
            self.inv_inertia_row1.truncate(),
            self.inv_inertia_row2.truncate(),
        )
    }
}

// ---------------------------------------------------------------------------
// Contact3D -- 64 bytes
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Contact3D {
    /// (x, y, z, depth)
    pub point: Vec4,
    /// (nx, ny, nz, 0)
    pub normal: Vec4,
    pub body_a: u32,
    pub body_b: u32,
    pub feature_id: u32,
    pub _pad: u32,
    pub lambda_n: f32,
    pub lambda_t1: f32,
    pub lambda_t2: f32,
    pub penalty_k: f32,
}

impl Contact3D {
    #[inline]
    pub fn contact_point(&self) -> glam::Vec3 {
        self.point.truncate()
    }

    #[inline]
    pub fn depth(&self) -> f32 {
        self.point.w
    }

    #[inline]
    pub fn contact_normal(&self) -> glam::Vec3 {
        self.normal.truncate()
    }
}

// ---------------------------------------------------------------------------
// Contact2D
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Contact2D {
    /// (x, y, depth, 0)
    pub point: Vec4,
    /// (nx, ny, 0, 0)
    pub normal: Vec4,
    pub body_a: u32,
    pub body_b: u32,
    pub feature_id: u32,
    pub _pad: u32,
    pub lambda_n: f32,
    pub lambda_t: f32,
    pub penalty_k: f32,
    pub _pad2: f32,
}

impl Contact2D {
    #[inline]
    pub fn contact_point(&self) -> glam::Vec2 {
        glam::Vec2::new(self.point.x, self.point.y)
    }

    #[inline]
    pub fn depth(&self) -> f32 {
        self.point.z
    }

    #[inline]
    pub fn contact_normal(&self) -> glam::Vec2 {
        glam::Vec2::new(self.normal.x, self.normal.y)
    }
}

// ---------------------------------------------------------------------------
// AABB types
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Aabb3D {
    pub min: Vec4,
    pub max: Vec4,
}

impl Aabb3D {
    pub fn new(min: glam::Vec3, max: glam::Vec3) -> Self {
        Self {
            min: min.extend(0.0),
            max: max.extend(0.0),
        }
    }

    #[inline]
    pub fn min_point(&self) -> glam::Vec3 {
        self.min.truncate()
    }

    #[inline]
    pub fn max_point(&self) -> glam::Vec3 {
        self.max.truncate()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Aabb2D {
    pub min: Vec4,
    pub max: Vec4,
}

impl Aabb2D {
    pub fn new(min: glam::Vec2, max: glam::Vec2) -> Self {
        Self {
            min: Vec4::new(min.x, min.y, 0.0, 0.0),
            max: Vec4::new(max.x, max.y, 0.0, 0.0),
        }
    }

    #[inline]
    pub fn min_point(&self) -> glam::Vec2 {
        glam::Vec2::new(self.min.x, self.min.y)
    }

    #[inline]
    pub fn max_point(&self) -> glam::Vec2 {
        glam::Vec2::new(self.max.x, self.max.y)
    }
}

// ---------------------------------------------------------------------------
// BVH node -- 48 bytes
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BvhNode {
    pub aabb_min: Vec4,
    pub aabb_max: Vec4,
    /// Left child index (negative = leaf).
    pub left: i32,
    /// Right child index (negative = leaf).
    pub right: i32,
    pub _pad: [u32; 2],
}

impl BvhNode {
    pub fn leaf(aabb_min: glam::Vec3, aabb_max: glam::Vec3, object_index: i32) -> Self {
        Self {
            aabb_min: aabb_min.extend(0.0),
            aabb_max: aabb_max.extend(0.0),
            left: -1,
            right: object_index,
            _pad: [0; 2],
        }
    }

    pub fn internal(aabb_min: glam::Vec3, aabb_max: glam::Vec3, left: i32, right: i32) -> Self {
        Self {
            aabb_min: aabb_min.extend(0.0),
            aabb_max: aabb_max.extend(0.0),
            left,
            right,
            _pad: [0; 2],
        }
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left < 0
    }
}

// ---------------------------------------------------------------------------
// Collision events (CPU-readable, not GPU structs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle {
    pub index: u32,
    pub generation: u32,
}

impl BodyHandle {
    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
}

impl PartialOrd for BodyHandle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BodyHandle {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index
            .cmp(&other.index)
            .then(self.generation.cmp(&other.generation))
    }
}

// ---------------------------------------------------------------------------
// Graph coloring for solver ordering
// ---------------------------------------------------------------------------

/// Greedy graph coloring on body adjacency from contact pairs.
/// Returns `(color_per_body, num_colors)`. Bodies with the same color share no contacts.
///
/// This is used to partition solver iterations: contacts between bodies of the
/// same color can be solved in parallel without data races.
pub fn greedy_coloring(num_bodies: usize, contact_pairs: &[(u32, u32)]) -> (Vec<u32>, u32) {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_bodies];
    for &(a, b) in contact_pairs {
        let a = a as usize;
        let b = b as usize;
        if a < num_bodies && b < num_bodies && a != b {
            adj[a].push(b);
            adj[b].push(a);
        }
    }

    let mut colors: Vec<u32> = vec![u32::MAX; num_bodies];
    let mut num_colors: u32 = 0;

    for body in 0..num_bodies {
        let mut used = Vec::new();
        for &nb in &adj[body] {
            if colors[nb] != u32::MAX {
                used.push(colors[nb]);
            }
        }
        used.sort_unstable();
        used.dedup();

        let mut c = 0u32;
        for &u in &used {
            if c == u {
                c += 1;
            } else {
                break;
            }
        }
        colors[body] = c;
        if c + 1 > num_colors {
            num_colors = c + 1;
        }
    }

    if num_bodies > 0 && num_colors == 0 {
        num_colors = 1;
    }

    (colors, num_colors)
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{from_bytes, Pod, Zeroable};
    use std::mem::size_of;

    // Compile-time Pod + Zeroable checks (the derive already asserts this,
    // but these functions make the intent explicit).
    const _: fn() = || {
        fn assert_pod_zeroable<T: Pod + Zeroable>() {}
        assert_pod_zeroable::<RigidBodyState3D>();
        assert_pod_zeroable::<RigidBodyState2D>();
        assert_pod_zeroable::<RigidBodyProps3D>();
        assert_pod_zeroable::<Contact3D>();
        assert_pod_zeroable::<Contact2D>();
        assert_pod_zeroable::<Aabb3D>();
        assert_pod_zeroable::<Aabb2D>();
        assert_pod_zeroable::<BvhNode>();
    };

    #[test]
    fn size_rigid_body_state_3d() {
        assert_eq!(size_of::<RigidBodyState3D>(), 64);
    }

    #[test]
    fn size_rigid_body_state_2d() {
        assert_eq!(size_of::<RigidBodyState2D>(), 64);
    }

    #[test]
    fn size_contact_3d() {
        assert_eq!(size_of::<Contact3D>(), 64);
    }

    #[test]
    fn size_contact_2d() {
        assert_eq!(size_of::<Contact2D>(), 64);
    }

    #[test]
    fn size_bvh_node() {
        assert_eq!(size_of::<BvhNode>(), 48);
    }

    #[test]
    fn size_aabb_3d() {
        assert_eq!(size_of::<Aabb3D>(), 32);
    }

    #[test]
    fn size_aabb_2d() {
        assert_eq!(size_of::<Aabb2D>(), 32);
    }

    #[test]
    fn bytemuck_round_trip_rigid_body_state_3d() {
        let state = RigidBodyState3D::new(
            glam::Vec3::new(1.0, 2.0, 3.0),
            0.5,
            glam::Quat::IDENTITY,
            glam::Vec3::new(4.0, 5.0, 6.0),
            glam::Vec3::new(0.1, 0.2, 0.3),
        );
        let bytes: &[u8] = bytemuck::bytes_of(&state);
        let back: &RigidBodyState3D = from_bytes(bytes);
        assert_eq!(back.position(), glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(back.inv_mass(), 0.5);
        assert_eq!(back.quat(), glam::Quat::IDENTITY);
        assert_eq!(back.linear_velocity(), glam::Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(back.angular_velocity(), glam::Vec3::new(0.1, 0.2, 0.3));
    }

    #[test]
    fn bytemuck_round_trip_rigid_body_state_2d() {
        let state = RigidBodyState2D::new(10.0, 20.0, 1.57, 0.25, 3.0, 4.0, 0.5);
        let bytes: &[u8] = bytemuck::bytes_of(&state);
        let back: &RigidBodyState2D = from_bytes(bytes);
        assert_eq!(back.position(), glam::Vec2::new(10.0, 20.0));
        assert_eq!(back.angle(), 1.57);
        assert_eq!(back.inv_mass(), 0.25);
        assert_eq!(back.linear_velocity(), glam::Vec2::new(3.0, 4.0));
        assert_eq!(back.angular_velocity(), 0.5);
    }

    #[test]
    fn bytemuck_round_trip_contact_3d() {
        let c = Contact3D {
            point: Vec4::new(1.0, 2.0, 3.0, 0.05),
            normal: Vec4::new(0.0, 1.0, 0.0, 0.0),
            body_a: 7,
            body_b: 42,
            feature_id: 99,
            _pad: 0,
            lambda_n: 1.5,
            lambda_t1: 0.1,
            lambda_t2: 0.2,
            penalty_k: 1000.0,
        };
        let bytes = bytemuck::bytes_of(&c);
        let back: &Contact3D = from_bytes(bytes);
        assert_eq!(back.contact_point(), glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(back.depth(), 0.05);
        assert_eq!(back.contact_normal(), glam::Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(back.body_a, 7);
        assert_eq!(back.body_b, 42);
        assert_eq!(back.penalty_k, 1000.0);
    }

    #[test]
    fn bytemuck_round_trip_contact_2d() {
        let c = Contact2D {
            point: Vec4::new(5.0, 6.0, 0.01, 0.0),
            normal: Vec4::new(0.0, 1.0, 0.0, 0.0),
            body_a: 1,
            body_b: 2,
            feature_id: 10,
            _pad: 0,
            lambda_n: 0.5,
            lambda_t: 0.3,
            penalty_k: 500.0,
            _pad2: 0.0,
        };
        let bytes = bytemuck::bytes_of(&c);
        let back: &Contact2D = from_bytes(bytes);
        assert_eq!(back.contact_point(), glam::Vec2::new(5.0, 6.0));
        assert_eq!(back.depth(), 0.01);
        assert_eq!(back.contact_normal(), glam::Vec2::new(0.0, 1.0));
        assert_eq!(back.body_a, 1);
        assert_eq!(back.body_b, 2);
    }

    #[test]
    fn bytemuck_round_trip_bvh_node() {
        let node = BvhNode::internal(
            glam::Vec3::new(-1.0, -1.0, -1.0),
            glam::Vec3::new(1.0, 1.0, 1.0),
            2,
            3,
        );
        let bytes = bytemuck::bytes_of(&node);
        let back: &BvhNode = from_bytes(bytes);
        assert_eq!(back.left, 2);
        assert_eq!(back.right, 3);
        assert!(!back.is_leaf());
    }

    #[test]
    fn bvh_leaf_detection() {
        let leaf = BvhNode::leaf(glam::Vec3::ZERO, glam::Vec3::ONE, 5);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.left, -1);
        assert_eq!(leaf.right, 5);
    }

    #[test]
    fn bytemuck_round_trip_aabb() {
        let aabb = Aabb3D::new(
            glam::Vec3::new(-1.0, -2.0, -3.0),
            glam::Vec3::new(1.0, 2.0, 3.0),
        );
        let bytes = bytemuck::bytes_of(&aabb);
        let back: &Aabb3D = from_bytes(bytes);
        assert_eq!(back.min_point(), glam::Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(back.max_point(), glam::Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn body_handle_equality() {
        let a = BodyHandle::new(1, 0);
        let b = BodyHandle::new(1, 0);
        let c = BodyHandle::new(1, 1);
        let d = BodyHandle::new(2, 0);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn body_handle_ordering() {
        let a = BodyHandle::new(0, 0);
        let b = BodyHandle::new(0, 1);
        let c = BodyHandle::new(1, 0);
        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn collision_event_equality() {
        let h1 = BodyHandle::new(0, 0);
        let h2 = BodyHandle::new(1, 0);
        let started = CollisionEvent::Started {
            body_a: h1,
            body_b: h2,
        };
        let ended = CollisionEvent::Ended {
            body_a: h1,
            body_b: h2,
        };
        assert_ne!(started, ended);
        assert_eq!(
            started,
            CollisionEvent::Started {
                body_a: h1,
                body_b: h2
            }
        );
    }

    #[test]
    fn rigid_body_props_inertia_round_trip() {
        let mat = glam::Mat3::from_diagonal(glam::Vec3::new(0.1, 0.2, 0.3));
        let props = RigidBodyProps3D::new(mat, 0.5, SHAPE_BOX, 0, FLAG_STATIC);
        let recovered = props.inv_inertia();
        let cols_orig = mat.to_cols_array();
        let cols_back = recovered.to_cols_array();
        for (a, b) in cols_orig.iter().zip(cols_back.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
        assert_eq!(props.friction, 0.5);
        assert_eq!(props.shape_type, SHAPE_BOX);
        assert_eq!(props.shape_index, 0);
        assert_eq!(props.flags, FLAG_STATIC);
    }

    #[test]
    fn helper_3d_position_extraction() {
        let state = RigidBodyState3D::new(
            glam::Vec3::new(10.0, 20.0, 30.0),
            1.0,
            glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            glam::Vec3::ZERO,
            glam::Vec3::ZERO,
        );
        assert_eq!(state.position(), glam::Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(state.inv_mass(), 1.0);
        let q = state.quat();
        let expected = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert!((q.x - expected.x).abs() < 1e-6);
        assert!((q.y - expected.y).abs() < 1e-6);
        assert!((q.z - expected.z).abs() < 1e-6);
        assert!((q.w - expected.w).abs() < 1e-6);
    }

    #[test]
    fn flags_and_shape_constants() {
        assert_eq!(FLAG_STATIC, 1);
        assert_eq!(FLAG_KINEMATIC, 2);
        assert_eq!(SHAPE_SPHERE, 0);
        assert_eq!(SHAPE_BOX, 1);
        assert_eq!(SHAPE_CAPSULE, 2);
        assert_eq!(SHAPE_CONVEX_HULL, 3);
    }

    #[test]
    fn test_greedy_coloring_triangle() {
        // 3 bodies, all touching each other: need 3 colors.
        let pairs = vec![(0, 1), (1, 2), (0, 2)];
        let (colors, num) = greedy_coloring(3, &pairs);
        assert_eq!(num, 3);
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[0], colors[2]);
    }

    #[test]
    fn test_greedy_coloring_chain() {
        // A-B-C chain: A touches B, B touches C, but A and C don't touch.
        let pairs = vec![(0, 1), (1, 2)];
        let (colors, num) = greedy_coloring(3, &pairs);
        assert_eq!(num, 2);
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_eq!(colors[0], colors[2]); // A and C can share a color
    }

    #[test]
    fn test_greedy_coloring_no_contacts() {
        let (colors, num) = greedy_coloring(5, &[]);
        assert_eq!(num, 1);
        assert!(colors.iter().all(|&c| c == 0));
    }

    #[test]
    fn slice_cast_multiple_bodies() {
        let bodies = vec![
            RigidBodyState3D::new(
                glam::Vec3::X,
                1.0,
                glam::Quat::IDENTITY,
                glam::Vec3::ZERO,
                glam::Vec3::ZERO,
            ),
            RigidBodyState3D::new(
                glam::Vec3::Y,
                2.0,
                glam::Quat::IDENTITY,
                glam::Vec3::ZERO,
                glam::Vec3::ZERO,
            ),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&bodies);
        assert_eq!(bytes.len(), 128);
        let back: &[RigidBodyState3D] = bytemuck::cast_slice(bytes);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].position(), glam::Vec3::X);
        assert_eq!(back[1].position(), glam::Vec3::Y);
        assert_eq!(back[1].inv_mass(), 2.0);
    }
}

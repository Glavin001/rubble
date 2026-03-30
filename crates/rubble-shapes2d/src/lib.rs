//! 2D shape definitions and CPU-side AABB computation for the rubble physics engine.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec4};
use rubble_math::Aabb2D;

// ---------------------------------------------------------------------------
// Shape data types (GPU-compatible, #[repr(C)])
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CircleData {
    pub radius: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RectData {
    /// (hx, hy, 0, 0)
    pub half_extents: Vec4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CapsuleData2D {
    pub half_height: f32,
    pub radius: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ConvexPolygonData {
    pub vertex_offset: u32,
    pub vertex_count: u32, // max 64
    pub _pad: [u32; 2],
}

// ---------------------------------------------------------------------------
// AABB computation (CPU reference)
// ---------------------------------------------------------------------------

pub fn compute_circle_aabb(center: Vec2, radius: f32) -> Aabb2D {
    let r = Vec2::splat(radius);
    Aabb2D::new(center - r, center + r)
}

pub fn compute_rect_aabb(center: Vec2, angle: f32, half_extents: Vec2) -> Aabb2D {
    let (sin, cos) = angle.sin_cos();
    // The world-space half-extent on each axis is the sum of absolute
    // projections of each local half-extent onto that axis.
    let wx = (cos * half_extents.x).abs() + (sin * half_extents.y).abs();
    let wy = (sin * half_extents.x).abs() + (cos * half_extents.y).abs();
    let w = Vec2::new(wx, wy);
    Aabb2D::new(center - w, center + w)
}

pub fn compute_capsule2d_aabb(center: Vec2, angle: f32, half_height: f32, radius: f32) -> Aabb2D {
    // The capsule's local axis is Y. Rotate the local up vector.
    let (sin, cos) = angle.sin_cos();
    let axis = Vec2::new(-sin * half_height, cos * half_height);
    let a = center + axis;
    let b = center - axis;
    let r = Vec2::splat(radius);
    let min = a.min(b) - r;
    let max = a.max(b) + r;
    Aabb2D::new(min, max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_aabb_origin() {
        let aabb = compute_circle_aabb(Vec2::ZERO, 1.0);
        assert_eq!(aabb.min_point(), Vec2::new(-1.0, -1.0));
        assert_eq!(aabb.max_point(), Vec2::new(1.0, 1.0));
    }

    #[test]
    fn rect_aabb_axis_aligned() {
        let aabb = compute_rect_aabb(Vec2::ZERO, 0.0, Vec2::new(2.0, 1.0));
        let eps = 1e-5;
        assert!((aabb.min_point() - Vec2::new(-2.0, -1.0)).length() < eps);
        assert!((aabb.max_point() - Vec2::new(2.0, 1.0)).length() < eps);
    }

    #[test]
    fn rect_aabb_rotated_45() {
        let angle = std::f32::consts::FRAC_PI_4;
        let he = Vec2::new(1.0, 1.0);
        let aabb = compute_rect_aabb(Vec2::ZERO, angle, he);
        // Rotated 45 degrees: each axis extent = |cos45|*1 + |sin45|*1 = sqrt(2)
        let s = std::f32::consts::FRAC_1_SQRT_2 * 2.0; // sqrt(2)
        let eps = 1e-5;
        assert!((aabb.min_point() - Vec2::new(-s, -s)).length() < eps);
        assert!((aabb.max_point() - Vec2::new(s, s)).length() < eps);
    }

    #[test]
    fn test_rect_aabb_rotated_30() {
        let angle = std::f32::consts::FRAC_PI_6; // 30 degrees
        let he = Vec2::new(2.0, 0.5);
        let aabb = compute_rect_aabb(Vec2::ZERO, angle, he);

        let cos = angle.cos().abs();
        let sin = angle.sin().abs();
        // wx = |cos30|*hx + |sin30|*hy = cos*2 + sin*0.5
        // wy = |sin30|*hx + |cos30|*hy = sin*2 + cos*0.5
        let wx = cos * 2.0 + sin * 0.5;
        let wy = sin * 2.0 + cos * 0.5;
        let expected_min = Vec2::new(-wx, -wy);
        let expected_max = Vec2::new(wx, wy);

        let eps = 1e-4;
        assert!(
            (aabb.min_point() - expected_min).length() < eps,
            "min: {:?} vs expected {:?}",
            aabb.min_point(),
            expected_min
        );
        assert!(
            (aabb.max_point() - expected_max).length() < eps,
            "max: {:?} vs expected {:?}",
            aabb.max_point(),
            expected_max
        );
    }

    #[test]
    fn capsule2d_aabb() {
        // Vertical capsule (angle = 0).
        let aabb = compute_capsule2d_aabb(Vec2::ZERO, 0.0, 1.0, 0.5);
        let eps = 1e-5;
        let expected_min = Vec2::new(-0.5, -1.5);
        let expected_max = Vec2::new(0.5, 1.5);
        assert!((aabb.min_point() - expected_min).length() < eps);
        assert!((aabb.max_point() - expected_max).length() < eps);
    }
}

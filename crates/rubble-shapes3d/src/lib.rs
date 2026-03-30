//! 3D shape definitions and CPU-side AABB computation for the rubble physics engine.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3, Vec4};
use rubble_math::{Aabb3D, BvhNode};

// ---------------------------------------------------------------------------
// Shape data types (GPU-compatible, #[repr(C)])
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SphereData {
    pub radius: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BoxData {
    /// (hx, hy, hz, 0)
    pub half_extents: Vec4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CapsuleData {
    pub half_height: f32,
    pub radius: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ConvexHullData {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub face_offset: u32,
    pub face_count: u32,
    pub edge_offset: u32,
    pub edge_count: u32,
    pub gauss_map_offset: u32,
    pub gauss_map_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// Convex hull validation
// ---------------------------------------------------------------------------

pub const MAX_CONVEX_HULL_VERTICES: usize = 64;

#[derive(thiserror::Error, Debug)]
pub enum ConvexHullError {
    #[error(
        "Too many vertices: {count} (max {max})",
        max = MAX_CONVEX_HULL_VERTICES
    )]
    TooManyVertices { count: usize },
    #[error("Degenerate hull: fewer than 4 non-coplanar points")]
    Degenerate,
}

/// Validates that a set of vertices can form a valid 3D convex hull.
///
/// Checks that:
/// - The vertex count does not exceed `MAX_CONVEX_HULL_VERTICES`.
/// - At least 4 non-coplanar points exist (i.e. the points span 3D space).
pub fn validate_convex_hull(vertices: &[Vec3]) -> Result<(), ConvexHullError> {
    if vertices.len() > MAX_CONVEX_HULL_VERTICES {
        return Err(ConvexHullError::TooManyVertices {
            count: vertices.len(),
        });
    }
    if vertices.len() < 4 {
        return Err(ConvexHullError::Degenerate);
    }

    // Find the first non-degenerate tetrahedron among the vertices.
    let p0 = vertices[0];

    // Find a point not coincident with p0.
    let edge = vertices[1..]
        .iter()
        .find(|v| (**v - p0).length_squared() > 1e-10)
        .map(|v| *v - p0);
    let edge = match edge {
        Some(e) => e,
        None => return Err(ConvexHullError::Degenerate),
    };

    // Find a point not collinear with the first edge.
    let normal = vertices[1..].iter().find_map(|v| {
        let cross = edge.cross(*v - p0);
        if cross.length_squared() > 1e-10 {
            Some(cross.normalize())
        } else {
            None
        }
    });
    let normal = match normal {
        Some(n) => n,
        None => return Err(ConvexHullError::Degenerate),
    };

    // Find a point not coplanar with the triangle.
    let has_volume = vertices[1..]
        .iter()
        .any(|v| (*v - p0).dot(normal).abs() > 1e-6);

    if !has_volume {
        return Err(ConvexHullError::Degenerate);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// AABB computation (CPU reference)
// ---------------------------------------------------------------------------

pub fn compute_sphere_aabb(center: Vec3, radius: f32) -> Aabb3D {
    let r = Vec3::splat(radius);
    Aabb3D::new(center - r, center + r)
}

pub fn compute_box_aabb(center: Vec3, rotation: Quat, half_extents: Vec3) -> Aabb3D {
    // Rotate each local axis and take the absolute value to find the world-space extent.
    let rot_mat = glam::Mat3::from_quat(rotation);
    let abs_mat = glam::Mat3::from_cols(
        rot_mat.x_axis.abs(),
        rot_mat.y_axis.abs(),
        rot_mat.z_axis.abs(),
    );
    let world_half = abs_mat * half_extents;
    Aabb3D::new(center - world_half, center + world_half)
}

pub fn compute_capsule_aabb(
    center: Vec3,
    rotation: Quat,
    half_height: f32,
    radius: f32,
) -> Aabb3D {
    // The capsule's local axis is Y. Transform the local up vector to world space.
    let local_axis = Vec3::Y * half_height;
    let world_axis = rotation * local_axis;
    // The two sphere centers are at center +/- world_axis.
    let a = center + world_axis;
    let b = center - world_axis;
    let r = Vec3::splat(radius);
    let min = a.min(b) - r;
    let max = a.max(b) + r;
    Aabb3D::new(min, max)
}

// ---------------------------------------------------------------------------
// Compound shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompoundChild {
    pub shape_type: u32,
    pub shape_index: u32,
    pub local_position: Vec3,
    pub local_rotation: Quat,
    pub local_aabb: Aabb3D,
}

#[derive(Debug, Clone)]
pub struct CompoundShape {
    pub children: Vec<CompoundChild>,
    /// Precomputed local BVH over children's AABBs.
    pub bvh_nodes: Vec<BvhNode>,
}

impl CompoundShape {
    /// Create a compound shape from children, building a BVH over their local AABBs.
    pub fn new(children: Vec<CompoundChild>) -> Self {
        let bvh_nodes = build_bvh(&children);
        Self {
            children,
            bvh_nodes,
        }
    }

    /// Compute the world AABB by transforming each child's local AABB.
    pub fn world_aabb(&self, position: Vec3, rotation: Quat) -> Aabb3D {
        let mut total_min = Vec3::splat(f32::MAX);
        let mut total_max = Vec3::splat(f32::NEG_INFINITY);

        for child in &self.children {
            let child_min = child.local_aabb.min_point();
            let child_max = child.local_aabb.max_point();
            // Transform the 8 corners of the local AABB to world space.
            for i in 0..8u32 {
                let x = if i & 1 == 0 { child_min.x } else { child_max.x };
                let y = if i & 2 == 0 { child_min.y } else { child_max.y };
                let z = if i & 4 == 0 { child_min.z } else { child_max.z };
                let local = Vec3::new(x, y, z);
                let world = position + rotation * local;
                total_min = total_min.min(world);
                total_max = total_max.max(world);
            }
        }

        Aabb3D::new(total_min, total_max)
    }
}

/// Build a simple top-down median-split BVH over the children's AABBs.
/// Returns N-1 internal nodes for N children.
fn build_bvh(children: &[CompoundChild]) -> Vec<BvhNode> {
    if children.len() <= 1 {
        return vec![];
    }

    let mut nodes = Vec::new();
    let indices: Vec<usize> = (0..children.len()).collect();
    build_bvh_recursive(children, &indices, &mut nodes);
    nodes
}

/// Recursively builds BVH nodes. Returns the index into `nodes` for the created
/// node, or a negative leaf marker -(child_index + 1) for single-element subsets.
fn build_bvh_recursive(
    children: &[CompoundChild],
    indices: &[usize],
    nodes: &mut Vec<BvhNode>,
) -> i32 {
    assert!(!indices.is_empty());

    if indices.len() == 1 {
        // Leaf: encode as -(child_index + 1) so the parent can distinguish leaves.
        return -(indices[0] as i32 + 1);
    }

    // Compute the AABB of all children in this subset.
    let (aabb_min, aabb_max) = compute_combined_aabb(children, indices);

    // Find the longest axis of the combined AABB.
    let extent = aabb_max - aabb_min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    // Sort indices by the center of each child's AABB along the chosen axis.
    let mut sorted = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        let ca = center_on_axis(children, a, axis);
        let cb = center_on_axis(children, b, axis);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Median split.
    let mid = sorted.len() / 2;
    let left_indices = &sorted[..mid];
    let right_indices = &sorted[mid..];

    let left = build_bvh_recursive(children, left_indices, nodes);
    let right = build_bvh_recursive(children, right_indices, nodes);

    let node_index = nodes.len() as i32;
    nodes.push(BvhNode::internal(aabb_min, aabb_max, left, right));
    node_index
}

fn center_on_axis(children: &[CompoundChild], idx: usize, axis: usize) -> f32 {
    let c = &children[idx];
    let center = (c.local_aabb.min_point() + c.local_aabb.max_point()) * 0.5;
    match axis {
        0 => center.x,
        1 => center.y,
        _ => center.z,
    }
}

fn compute_combined_aabb(children: &[CompoundChild], indices: &[usize]) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &i in indices {
        min = min.min(children[i].local_aabb.min_point());
        max = max.max(children[i].local_aabb.max_point());
    }
    (min, max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_too_many_vertices() {
        let verts: Vec<Vec3> = (0..65).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        let result = validate_convex_hull(&verts);
        assert!(matches!(
            result,
            Err(ConvexHullError::TooManyVertices { count: 65 })
        ));
    }

    #[test]
    fn validate_valid_hull() {
        // 20 vertices forming a roughly spherical distribution.
        let mut verts = Vec::new();
        for i in 0..20 {
            let t = i as f32 / 20.0 * std::f32::consts::TAU;
            let phi = (i as f32 / 20.0 - 0.5) * std::f32::consts::PI;
            verts.push(Vec3::new(
                phi.cos() * t.cos(),
                phi.cos() * t.sin(),
                phi.sin(),
            ));
        }
        assert!(validate_convex_hull(&verts).is_ok());
    }

    #[test]
    fn validate_coplanar_degenerate() {
        // All points on the XY plane.
        let verts: Vec<Vec3> = (0..10)
            .map(|i| {
                let t = i as f32 / 10.0 * std::f32::consts::TAU;
                Vec3::new(t.cos(), t.sin(), 0.0)
            })
            .collect();
        let result = validate_convex_hull(&verts);
        assert!(matches!(result, Err(ConvexHullError::Degenerate)));
    }

    #[test]
    fn sphere_aabb_origin() {
        let aabb = compute_sphere_aabb(Vec3::ZERO, 1.0);
        assert_eq!(aabb.min_point(), Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(aabb.max_point(), Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn box_aabb_axis_aligned() {
        let aabb = compute_box_aabb(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.5));
        assert_eq!(aabb.min_point(), Vec3::new(-0.5, -0.5, -0.5));
        assert_eq!(aabb.max_point(), Vec3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn box_aabb_rotated_45_y() {
        let angle = std::f32::consts::FRAC_PI_4;
        let rot = Quat::from_rotation_y(angle);
        let he = Vec3::new(1.0, 1.0, 1.0);
        let aabb = compute_box_aabb(Vec3::ZERO, rot, he);

        // For a unit cube rotated 45 degrees around Y, the X and Z extents become
        // |cos(45)| * 1 + |sin(45)| * 1 = sqrt(2), Y stays 1.
        let s = angle.sin().abs() + angle.cos().abs(); // sqrt(2)
        let expected_min = Vec3::new(-s, -1.0, -s);
        let expected_max = Vec3::new(s, 1.0, s);

        let eps = 1e-5;
        assert!(
            (aabb.min_point() - expected_min).length() < eps,
            "min: {:?} vs {:?}",
            aabb.min_point(),
            expected_min
        );
        assert!(
            (aabb.max_point() - expected_max).length() < eps,
            "max: {:?} vs {:?}",
            aabb.max_point(),
            expected_max
        );
    }

    #[test]
    fn capsule_aabb_vertical() {
        // Vertical capsule (identity rotation, Y-axis aligned).
        let aabb = compute_capsule_aabb(Vec3::ZERO, Quat::IDENTITY, 1.0, 0.5);
        let expected_min = Vec3::new(-0.5, -1.5, -0.5);
        let expected_max = Vec3::new(0.5, 1.5, 0.5);
        let eps = 1e-5;
        assert!((aabb.min_point() - expected_min).length() < eps);
        assert!((aabb.max_point() - expected_max).length() < eps);
    }

    #[test]
    fn compound_world_aabb_union() {
        // 3 children: spheres at different positions.
        let children = vec![
            CompoundChild {
                shape_type: 0,
                shape_index: 0,
                local_position: Vec3::new(-2.0, 0.0, 0.0),
                local_rotation: Quat::IDENTITY,
                local_aabb: Aabb3D::new(Vec3::new(-3.0, -1.0, -1.0), Vec3::new(-1.0, 1.0, 1.0)),
            },
            CompoundChild {
                shape_type: 0,
                shape_index: 1,
                local_position: Vec3::ZERO,
                local_rotation: Quat::IDENTITY,
                local_aabb: Aabb3D::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            },
            CompoundChild {
                shape_type: 0,
                shape_index: 2,
                local_position: Vec3::new(2.0, 0.0, 0.0),
                local_rotation: Quat::IDENTITY,
                local_aabb: Aabb3D::new(Vec3::new(1.0, -1.0, -1.0), Vec3::new(3.0, 1.0, 1.0)),
            },
        ];
        let compound = CompoundShape::new(children);
        let world = compound.world_aabb(Vec3::ZERO, Quat::IDENTITY);
        let eps = 1e-5;
        assert!((world.min_point() - Vec3::new(-3.0, -1.0, -1.0)).length() < eps);
        assert!((world.max_point() - Vec3::new(3.0, 1.0, 1.0)).length() < eps);
    }

    #[test]
    fn compound_bvh_node_count() {
        // 20 children should produce 19 internal nodes.
        let children: Vec<CompoundChild> = (0..20)
            .map(|i| {
                let x = i as f32;
                CompoundChild {
                    shape_type: 0,
                    shape_index: i,
                    local_position: Vec3::new(x, 0.0, 0.0),
                    local_rotation: Quat::IDENTITY,
                    local_aabb: Aabb3D::new(
                        Vec3::new(x - 0.5, -0.5, -0.5),
                        Vec3::new(x + 0.5, 0.5, 0.5),
                    ),
                }
            })
            .collect();
        let compound = CompoundShape::new(children);
        assert_eq!(compound.bvh_nodes.len(), 19);
    }
}

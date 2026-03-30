use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Quat, Vec3, Vec4};
use rubble_math::Aabb3D;
use thiserror::Error;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeType {
    Sphere = 0,
    Box = 1,
    Capsule = 2,
    ConvexHull = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SphereData {
    pub radius: f32,
    pub pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BoxData {
    pub half_extents: Vec4,
}

#[derive(Clone, Debug)]
pub enum ShapeDesc {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
}

#[derive(Debug, Error, PartialEq)]
pub enum ConvexHullError {
    #[error("too many vertices: {count} > {max}")]
    TooManyVertices { count: usize, max: usize },
    #[error("degenerate hull")]
    Degenerate,
}

pub fn validate_convex_hull(vertices: &[Vec3]) -> Result<(), ConvexHullError> {
    if vertices.len() > 64 {
        return Err(ConvexHullError::TooManyVertices {
            count: vertices.len(),
            max: 64,
        });
    }
    if vertices.len() < 4 {
        return Err(ConvexHullError::Degenerate);
    }
    let a = vertices[0];
    let mut found = false;
    'outer: for i in 1..vertices.len() {
        for j in i + 1..vertices.len() {
            for k in j + 1..vertices.len() {
                let n = (vertices[j] - vertices[i]).cross(vertices[k] - vertices[i]);
                if n.length_squared() > 1e-8 && (a - vertices[i]).dot(n).abs() > 1e-5 {
                    found = true;
                    break 'outer;
                }
            }
        }
    }
    if !found {
        return Err(ConvexHullError::Degenerate);
    }
    Ok(())
}

pub fn compute_aabb(shape: &ShapeDesc, position: Vec3, rotation: Quat) -> Aabb3D {
    match shape {
        ShapeDesc::Sphere { radius } => Aabb3D {
            min: (position - Vec3::splat(*radius)).extend(0.0),
            max: (position + Vec3::splat(*radius)).extend(0.0),
        },
        ShapeDesc::Box { half_extents } => {
            let r = Mat3::from_quat(rotation);
            let abs_r = Mat3::from_cols(r.x_axis.abs(), r.y_axis.abs(), r.z_axis.abs());
            let ext = abs_r * *half_extents;
            Aabb3D {
                min: (position - ext).extend(0.0),
                max: (position + ext).extend(0.0),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hull_limit_enforced() {
        let verts = vec![Vec3::ZERO; 65];
        assert!(matches!(
            validate_convex_hull(&verts),
            Err(ConvexHullError::TooManyVertices { .. })
        ));
    }

    #[test]
    fn sphere_aabb_at_origin() {
        let aabb = compute_aabb(
            &ShapeDesc::Sphere { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        assert_eq!(aabb.min.truncate(), Vec3::splat(-1.0));
        assert_eq!(aabb.max.truncate(), Vec3::splat(1.0));
    }

    #[test]
    fn rotated_box_aabb_matches_analytic() {
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let aabb = compute_aabb(
            &ShapeDesc::Box {
                half_extents: Vec3::ONE,
            },
            Vec3::ZERO,
            rot,
        );
        let expected = std::f32::consts::SQRT_2;
        assert_relative_eq!(aabb.max.x, expected, epsilon = 1e-5);
        assert_relative_eq!(aabb.max.z, expected, epsilon = 1e-5);
    }
}

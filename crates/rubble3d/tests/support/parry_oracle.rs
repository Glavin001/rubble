#![allow(dead_code)]

use glam::{Quat, Vec3};
use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};
use parry3d::shape::{Ball, Capsule, ConvexPolyhedron, Cuboid, HalfSpace, SharedShape};
use rubble3d::ShapeDesc;

/// Result from a parry3d contact oracle query.
#[derive(Debug, Clone)]
pub struct ParryContactResult {
    /// World-space contact point on shape A.
    pub point_a: Vec3,
    /// World-space contact point on shape B.
    pub point_b: Vec3,
    /// Contact normal pointing from A toward B.
    pub normal: Vec3,
    /// Signed penetration depth (negative = penetrating).
    pub depth: f32,
}

/// Convert a glam position + quaternion to a parry3d Isometry.
pub fn to_isometry(position: Vec3, rotation: Quat) -> Isometry3<f32> {
    let translation = Vector3::new(position.x, position.y, position.z);
    let quat = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        rotation.w, rotation.x, rotation.y, rotation.z,
    ));
    Isometry3::from_parts(translation.into(), quat)
}

/// Convert a nalgebra Point3 to glam Vec3.
fn from_point(p: Point3<f32>) -> Vec3 {
    Vec3::new(p.x, p.y, p.z)
}

/// Convert a nalgebra Vector3 to glam Vec3.
fn from_vector(v: Vector3<f32>) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// Convert a Rubble ShapeDesc to a parry3d SharedShape.
///
/// Returns `None` for compound shapes (not directly translatable to a single parry shape).
pub fn to_parry_shape(shape: &ShapeDesc) -> Option<SharedShape> {
    match shape {
        ShapeDesc::Sphere { radius } => Some(SharedShape::new(Ball::new(*radius))),
        ShapeDesc::Box { half_extents } => Some(SharedShape::new(Cuboid::new(Vector3::new(
            half_extents.x,
            half_extents.y,
            half_extents.z,
        )))),
        ShapeDesc::Capsule {
            half_height,
            radius,
        } => {
            // Rubble capsule axis is local Y. Parry Capsule::new takes two endpoints.
            let a = Point3::new(0.0, -*half_height, 0.0);
            let b = Point3::new(0.0, *half_height, 0.0);
            Some(SharedShape::new(Capsule::new(a, b, *radius)))
        }
        ShapeDesc::ConvexHull { vertices } => {
            let points: Vec<Point3<f32>> = vertices
                .iter()
                .map(|v| Point3::new(v.x, v.y, v.z))
                .collect();
            // ConvexPolyhedron::from_convex_hull returns None if degenerate.
            ConvexPolyhedron::from_convex_hull(&points).map(SharedShape::new)
        }
        ShapeDesc::Plane { normal, distance } => {
            // Parry HalfSpace normal must be a unit vector. The shape is centered at origin;
            // the plane offset is encoded in the isometry instead.
            let _ = distance; // handled by caller via isometry translation
            let n = Vector3::new(normal.x, normal.y, normal.z);
            nalgebra::Unit::try_new(n, 1.0e-6)
                .map(|unit_n| SharedShape::new(HalfSpace::new(unit_n)))
        }
        ShapeDesc::Compound { .. } => None,
    }
}

/// Adjust the isometry for a plane shape (parry HalfSpace is at origin, so we encode
/// the plane distance as a translation along the normal).
pub fn plane_isometry(shape: &ShapeDesc, position: Vec3, rotation: Quat) -> Isometry3<f32> {
    if let ShapeDesc::Plane { normal, distance } = shape {
        // Plane equation: dot(normal, x) = distance
        // HalfSpace at origin: dot(normal, x) = 0
        // So translate by normal * distance
        let offset = *normal * *distance;
        to_isometry(position + offset, rotation)
    } else {
        to_isometry(position, rotation)
    }
}

/// Query parry3d for the closest contact between two shapes.
///
/// `prediction_distance` is the maximum separation at which a contact is still reported.
/// Returns `None` if no contact exists within the prediction distance.
pub fn parry_contact_query(
    shape_a: &ShapeDesc,
    pos_a: Vec3,
    rot_a: Quat,
    shape_b: &ShapeDesc,
    pos_b: Vec3,
    rot_b: Quat,
    prediction_distance: f32,
) -> Option<ParryContactResult> {
    let parry_a = to_parry_shape(shape_a)?;
    let parry_b = to_parry_shape(shape_b)?;

    let iso_a = if matches!(shape_a, ShapeDesc::Plane { .. }) {
        plane_isometry(shape_a, pos_a, rot_a)
    } else {
        to_isometry(pos_a, rot_a)
    };

    let iso_b = if matches!(shape_b, ShapeDesc::Plane { .. }) {
        plane_isometry(shape_b, pos_b, rot_b)
    } else {
        to_isometry(pos_b, rot_b)
    };

    let contact = parry3d::query::contact(
        &iso_a,
        parry_a.as_ref(),
        &iso_b,
        parry_b.as_ref(),
        prediction_distance,
    )
    .ok()
    .flatten()?;

    // parry returns dist > 0 for separated, dist < 0 for penetrating
    // Rubble uses depth < 0 for penetrating (same convention)
    Some(ParryContactResult {
        point_a: from_point(contact.point1),
        point_b: from_point(contact.point2),
        normal: from_vector(*contact.normal1),
        depth: -contact.dist, // flip sign: parry dist>0 = separated, rubble depth<0 = penetrating
    })
}

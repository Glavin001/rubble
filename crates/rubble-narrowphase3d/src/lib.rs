//! CPU reference narrowphase collision detection for the rubble physics engine.
//!
//! Provides collision detection routines for sphere, box, capsule, and plane
//! primitives, plus contact manifold reduction and persistence.

use glam::{Quat, Vec3, Vec4};
use rubble_math::{BodyHandle, CollisionEvent, Contact3D};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_contact(body_a: u32, body_b: u32, point: Vec3, normal: Vec3, depth: f32) -> Contact3D {
    Contact3D {
        point: Vec4::new(point.x, point.y, point.z, depth),
        normal: Vec4::new(normal.x, normal.y, normal.z, 0.0),
        body_a,
        body_b,
        feature_id: 0,
        _pad: 0,
        lambda_n: 0.0,
        lambda_t1: 0.0,
        lambda_t2: 0.0,
        penalty_k: 0.0,
    }
}

// ---------------------------------------------------------------------------
// Sphere vs Sphere
// ---------------------------------------------------------------------------

/// Sphere vs Sphere. Returns 0 or 1 contacts.
pub fn sphere_sphere(
    pos_a: Vec3,
    rad_a: f32,
    pos_b: Vec3,
    rad_b: f32,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    let d = pos_b - pos_a;
    let dist_sq = d.length_squared();
    let r_sum = rad_a + rad_b;
    if dist_sq >= r_sum * r_sum || dist_sq < 1e-12 {
        return vec![];
    }
    let dist = dist_sq.sqrt();
    let normal = d / dist;
    let depth = -(r_sum - dist); // negative = penetrating
    let point = pos_a + normal * (rad_a + depth * 0.5);
    vec![make_contact(body_a, body_b, point, normal, depth)]
}

// ---------------------------------------------------------------------------
// Sphere vs Box (closest point on OBB)
// ---------------------------------------------------------------------------

/// Sphere vs Box (closest point on OBB). Returns 0 or 1 contacts.
pub fn sphere_box(
    sphere_pos: Vec3,
    radius: f32,
    box_pos: Vec3,
    box_rot: Quat,
    half: Vec3,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    // Transform sphere center to box local space
    let inv_rot = box_rot.inverse();
    let local = inv_rot * (sphere_pos - box_pos);
    // Clamp to box extents
    let closest = local.clamp(-half, half);
    let diff = local - closest;
    let dist_sq = diff.length_squared();

    if dist_sq >= radius * radius && dist_sq > 1e-12 {
        return vec![];
    }

    if dist_sq < 1e-12 {
        // Sphere center is inside box -- find closest face for separation
        let face_dists = [
            half.x - local.x.abs(),
            half.y - local.y.abs(),
            half.z - local.z.abs(),
        ];
        let (min_axis, &min_dist) = face_dists
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let mut normal_local = Vec3::ZERO;
        match min_axis {
            0 => normal_local.x = if local.x >= 0.0 { 1.0 } else { -1.0 },
            1 => normal_local.y = if local.y >= 0.0 { 1.0 } else { -1.0 },
            _ => normal_local.z = if local.z >= 0.0 { 1.0 } else { -1.0 },
        }
        let normal = box_rot * normal_local;
        let point = box_pos + box_rot * closest;
        let depth = -(radius + min_dist);
        return vec![make_contact(body_a, body_b, point, normal, depth)];
    }

    let dist = dist_sq.sqrt();
    let normal_local = diff / dist;
    let normal = box_rot * normal_local;
    let point = box_pos + box_rot * closest;
    let depth = -(radius - dist);
    vec![make_contact(body_a, body_b, point, normal, depth)]
}

// ---------------------------------------------------------------------------
// Box vs Box (SAT, simplified)
// ---------------------------------------------------------------------------

/// Box vs Box using SAT (Separating Axis Theorem) with up to 15 axes.
/// Returns 0 or 1 contacts (simplified: single contact at penetration midpoint).
pub fn box_box(
    pos_a: Vec3,
    rot_a: Quat,
    half_a: Vec3,
    pos_b: Vec3,
    rot_b: Quat,
    half_b: Vec3,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    let axes_a = [
        rot_a * Vec3::X,
        rot_a * Vec3::Y,
        rot_a * Vec3::Z,
    ];
    let axes_b = [
        rot_b * Vec3::X,
        rot_b * Vec3::Y,
        rot_b * Vec3::Z,
    ];

    let d = pos_b - pos_a;
    let half_a_arr = [half_a.x, half_a.y, half_a.z];
    let half_b_arr = [half_b.x, half_b.y, half_b.z];

    let mut min_overlap = f32::MAX;
    let mut min_axis = Vec3::ZERO;

    // Helper: project half-extents of an OBB onto an axis
    let project_obb = |axes: &[Vec3; 3], halves: &[f32; 3], axis: Vec3| -> f32 {
        halves[0] * axes[0].dot(axis).abs()
            + halves[1] * axes[1].dot(axis).abs()
            + halves[2] * axes[2].dot(axis).abs()
    };

    let mut test_axis = |axis: Vec3| -> bool {
        let len_sq = axis.length_squared();
        if len_sq < 1e-8 {
            return true; // skip degenerate axis
        }
        let axis = axis / len_sq.sqrt();
        let proj_a = project_obb(&axes_a, &half_a_arr, axis);
        let proj_b = project_obb(&axes_b, &half_b_arr, axis);
        let dist = d.dot(axis).abs();
        let overlap = proj_a + proj_b - dist;
        if overlap < 0.0 {
            return false; // separating axis found
        }
        if overlap < min_overlap {
            min_overlap = overlap;
            // Ensure the normal points from A to B
            min_axis = if d.dot(axis) >= 0.0 { axis } else { -axis };
        }
        true
    };

    // 6 face normals
    for i in 0..3 {
        if !test_axis(axes_a[i]) {
            return vec![];
        }
        if !test_axis(axes_b[i]) {
            return vec![];
        }
    }

    // 9 edge-edge cross products
    for i in 0..3 {
        for j in 0..3 {
            if !test_axis(axes_a[i].cross(axes_b[j])) {
                return vec![];
            }
        }
    }

    if min_overlap <= 0.0 || min_axis.length_squared() < 1e-12 {
        return vec![];
    }

    // Generate contact point at the midpoint along the penetration axis
    let depth = -min_overlap;
    let point = pos_a + d * 0.5;
    vec![make_contact(body_a, body_b, point, min_axis, depth)]
}

// ---------------------------------------------------------------------------
// Sphere vs Capsule
// ---------------------------------------------------------------------------

/// Closest point on a line segment (a, b) to point p.
fn closest_point_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// Sphere vs Capsule. Returns 0 or 1 contacts.
pub fn sphere_capsule(
    sphere_pos: Vec3,
    radius: f32,
    cap_pos: Vec3,
    cap_rot: Quat,
    half_h: f32,
    cap_rad: f32,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    let axis = cap_rot * Vec3::Y;
    let seg_a = cap_pos - axis * half_h;
    let seg_b = cap_pos + axis * half_h;
    let closest = closest_point_on_segment(seg_a, seg_b, sphere_pos);
    sphere_sphere(sphere_pos, radius, closest, cap_rad, body_a, body_b)
}

// ---------------------------------------------------------------------------
// Capsule vs Capsule
// ---------------------------------------------------------------------------

/// Closest points between two line segments. Returns (point_on_seg1, point_on_seg2).
fn closest_points_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a < 1e-12 && e < 1e-12 {
        return (p1, p2);
    }
    if a < 1e-12 {
        let t = (f / e).clamp(0.0, 1.0);
        return (p1, p2 + d2 * t);
    }
    let c = d1.dot(r);
    if e < 1e-12 {
        let s = (-c / a).clamp(0.0, 1.0);
        return (p1 + d1 * s, p2);
    }

    let b = d1.dot(d2);
    let denom = a * e - b * b;

    let mut s = if denom.abs() > 1e-12 {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut t = (b * s + f) / e;

    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }

    (p1 + d1 * s, p2 + d2 * t)
}

/// Capsule vs Capsule. Returns 0 or 1 contacts.
pub fn capsule_capsule(
    pos_a: Vec3,
    rot_a: Quat,
    hh_a: f32,
    rad_a: f32,
    pos_b: Vec3,
    rot_b: Quat,
    hh_b: f32,
    rad_b: f32,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    let axis_a = rot_a * Vec3::Y;
    let axis_b = rot_b * Vec3::Y;
    let a1 = pos_a - axis_a * hh_a;
    let a2 = pos_a + axis_a * hh_a;
    let b1 = pos_b - axis_b * hh_b;
    let b2 = pos_b + axis_b * hh_b;

    let (ca, cb) = closest_points_segments(a1, a2, b1, b2);
    sphere_sphere(ca, rad_a, cb, rad_b, body_a, body_b)
}

// ---------------------------------------------------------------------------
// Plane vs Sphere
// ---------------------------------------------------------------------------

/// Plane vs Sphere. The plane is defined by normal and signed distance from origin.
/// Returns 0 or 1 contacts.
pub fn plane_sphere(
    normal: Vec3,
    dist: f32,
    sphere_pos: Vec3,
    radius: f32,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    let d = normal.dot(sphere_pos) - dist;
    if d >= radius {
        return vec![];
    }
    let depth = -(radius - d);
    let point = sphere_pos - normal * d;
    vec![make_contact(body_a, body_b, point, normal, depth)]
}

// ---------------------------------------------------------------------------
// Plane vs Box
// ---------------------------------------------------------------------------

/// Plane vs Box. Returns 0-8 contacts (one per vertex below the plane).
pub fn plane_box(
    normal: Vec3,
    dist: f32,
    box_pos: Vec3,
    box_rot: Quat,
    half: Vec3,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact3D> {
    // 8 box vertices in local space
    let signs: [(f32, f32, f32); 8] = [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ];

    let mut contacts = Vec::new();
    for (sx, sy, sz) in signs {
        let local = Vec3::new(sx * half.x, sy * half.y, sz * half.z);
        let world = box_pos + box_rot * local;
        let d = normal.dot(world) - dist;
        if d < 0.0 {
            let point = world - normal * d; // project onto plane
            let depth = d; // already negative
            contacts.push(make_contact(body_a, body_b, point, normal, depth));
        }
    }
    contacts
}

// ---------------------------------------------------------------------------
// Manifold reduction
// ---------------------------------------------------------------------------

/// Reduce a contact manifold to at most 4 contacts using area maximization.
///
/// Strategy:
/// 1. Keep the deepest contact.
/// 2. Keep the contact farthest from it.
/// 3. Keep the contact that maximizes triangle area with the first two.
/// 4. Keep the contact that maximizes the quadrilateral area.
pub fn reduce_manifold(contacts: &[Contact3D]) -> Vec<Contact3D> {
    if contacts.len() <= 4 {
        return contacts.to_vec();
    }

    let mut result = Vec::with_capacity(4);
    let mut used = vec![false; contacts.len()];

    // 1. Deepest (most negative depth)
    let idx0 = contacts
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.depth().partial_cmp(&b.1.depth()).unwrap())
        .unwrap()
        .0;
    result.push(contacts[idx0]);
    used[idx0] = true;

    // 2. Farthest from idx0
    let p0 = contacts[idx0].contact_point();
    let idx1 = contacts
        .iter()
        .enumerate()
        .filter(|(i, _)| !used[*i])
        .max_by(|a, b| {
            let da = (a.1.contact_point() - p0).length_squared();
            let db = (b.1.contact_point() - p0).length_squared();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap()
        .0;
    result.push(contacts[idx1]);
    used[idx1] = true;

    // 3. Maximize triangle area with p0, p1
    let p1 = contacts[idx1].contact_point();
    let edge = p1 - p0;
    let idx2 = contacts
        .iter()
        .enumerate()
        .filter(|(i, _)| !used[*i])
        .max_by(|a, b| {
            let area_a = edge.cross(a.1.contact_point() - p0).length_squared();
            let area_b = edge.cross(b.1.contact_point() - p0).length_squared();
            area_a.partial_cmp(&area_b).unwrap()
        })
        .unwrap()
        .0;
    result.push(contacts[idx2]);
    used[idx2] = true;

    // 4. Maximize quadrilateral area -- pick the point that maximizes the
    //    sum of triangle areas formed with each edge of the existing triangle.
    let p2 = contacts[idx2].contact_point();
    let idx3 = contacts
        .iter()
        .enumerate()
        .filter(|(i, _)| !used[*i])
        .max_by(|a, b| {
            let pa = a.1.contact_point();
            let pb = b.1.contact_point();
            let area_a = (p0 - pa).cross(p1 - pa).length()
                + (p1 - pa).cross(p2 - pa).length()
                + (p2 - pa).cross(p0 - pa).length();
            let area_b = (p0 - pb).cross(p1 - pb).length()
                + (p1 - pb).cross(p2 - pb).length()
                + (p2 - pb).cross(p0 - pb).length();
            area_a.partial_cmp(&area_b).unwrap()
        })
        .unwrap()
        .0;
    result.push(contacts[idx3]);

    result
}

// ---------------------------------------------------------------------------
// Contact persistence
// ---------------------------------------------------------------------------

/// Tracks contacts frame-to-frame for warm-starting and collision events.
pub struct ContactPersistence {
    prev_contacts: Vec<Contact3D>,
    warmstart_decay: f32,
}

impl ContactPersistence {
    /// Create a new persistence tracker with the given lambda decay factor (0..1).
    pub fn new(decay: f32) -> Self {
        Self {
            prev_contacts: Vec::new(),
            warmstart_decay: decay,
        }
    }

    /// Update with new contacts. Returns the contacts (with warm-started lambdas)
    /// and a list of collision events (Started / Ended).
    pub fn update(
        &mut self,
        mut new_contacts: Vec<Contact3D>,
    ) -> (Vec<Contact3D>, Vec<CollisionEvent>) {
        let mut events = Vec::new();

        // Collect active body pairs from previous and new frames
        let prev_pairs: HashSet<(u32, u32)> = self
            .prev_contacts
            .iter()
            .map(|c| (c.body_a.min(c.body_b), c.body_a.max(c.body_b)))
            .collect();

        let new_pairs: HashSet<(u32, u32)> = new_contacts
            .iter()
            .map(|c| (c.body_a.min(c.body_b), c.body_a.max(c.body_b)))
            .collect();

        // Started events: pairs in new but not in prev
        for &(a, b) in &new_pairs {
            if !prev_pairs.contains(&(a, b)) {
                events.push(CollisionEvent::Started {
                    body_a: BodyHandle::new(a, 0),
                    body_b: BodyHandle::new(b, 0),
                });
            }
        }

        // Ended events: pairs in prev but not in new
        for &(a, b) in &prev_pairs {
            if !new_pairs.contains(&(a, b)) {
                events.push(CollisionEvent::Ended {
                    body_a: BodyHandle::new(a, 0),
                    body_b: BodyHandle::new(b, 0),
                });
            }
        }

        // Warm-start: carry over lambdas from previous contacts to matching new contacts
        let dist_thresh_sq = 0.01 * 0.01; // 1cm matching threshold
        for nc in new_contacts.iter_mut() {
            let np = nc.contact_point();
            // Find closest previous contact with same body pair
            let mut best_dist_sq = f32::MAX;
            let mut best_idx: Option<usize> = None;
            for (i, pc) in self.prev_contacts.iter().enumerate() {
                if (pc.body_a == nc.body_a && pc.body_b == nc.body_b)
                    || (pc.body_a == nc.body_b && pc.body_b == nc.body_a)
                {
                    let d2 = (pc.contact_point() - np).length_squared();
                    if d2 < best_dist_sq {
                        best_dist_sq = d2;
                        best_idx = Some(i);
                    }
                }
            }
            if let Some(idx) = best_idx {
                if best_dist_sq < dist_thresh_sq {
                    let pc = &self.prev_contacts[idx];
                    nc.lambda_n = pc.lambda_n * self.warmstart_decay;
                    nc.lambda_t1 = pc.lambda_t1 * self.warmstart_decay;
                    nc.lambda_t2 = pc.lambda_t2 * self.warmstart_decay;
                }
            }
        }

        self.prev_contacts = new_contacts.clone();
        (new_contacts, events)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn approx_vec3(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // 1. sphere_sphere overlapping
    #[test]
    fn test_sphere_sphere_overlap() {
        let contacts = sphere_sphere(
            Vec3::ZERO,
            1.0,
            Vec3::new(1.5, 0.0, 0.0),
            1.0,
            0,
            1,
        );
        assert_eq!(contacts.len(), 1);
        let c = &contacts[0];
        assert!(approx_vec3(c.contact_normal(), Vec3::new(1.0, 0.0, 0.0)));
        assert!(approx_eq(c.depth(), -0.5));
    }

    // 2. sphere_sphere separated
    #[test]
    fn test_sphere_sphere_separated() {
        let contacts = sphere_sphere(
            Vec3::ZERO,
            1.0,
            Vec3::new(3.0, 0.0, 0.0),
            1.0,
            0,
            1,
        );
        assert_eq!(contacts.len(), 0);
    }

    // 3. sphere_box near face
    #[test]
    fn test_sphere_box_near_face() {
        // Sphere at x=1.3, radius=0.5. Box half=1. Closest point on box = (1,0,0).
        // Distance = 0.3, which is < 0.5, so there's a collision.
        let contacts = sphere_box(
            Vec3::new(1.3, 0.0, 0.0),
            0.5,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            0,
            1,
        );
        assert_eq!(contacts.len(), 1);
        let c = &contacts[0];
        // Normal should point roughly along +X (from box face toward sphere)
        assert!(c.contact_normal().x > 0.9);
        assert!(c.depth() < 0.0); // penetrating
    }

    // 4. box_box overlapping (axis-aligned)
    #[test]
    fn test_box_box_overlap() {
        let contacts = box_box(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            Vec3::new(1.5, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
            0,
            1,
        );
        assert!(!contacts.is_empty());
        let c = &contacts[0];
        assert!(c.depth() < 0.0); // penetrating
    }

    // 5. box_box separated
    #[test]
    fn test_box_box_separated() {
        let contacts = box_box(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            Vec3::new(3.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
            0,
            1,
        );
        assert_eq!(contacts.len(), 0);
    }

    // 6. sphere_capsule
    #[test]
    fn test_sphere_capsule() {
        // Sphere at (1.2, 0, 0) r=0.5. Capsule at origin, axis=Y, half_h=1, r=1.
        // Closest segment point to sphere = (0,0,0). Distance = 1.2. r_sum = 1.5. Overlap!
        let contacts = sphere_capsule(
            Vec3::new(1.2, 0.0, 0.0),
            0.5,
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
            1.0,
            0,
            1,
        );
        assert_eq!(contacts.len(), 1);
        let c = &contacts[0];
        assert!(c.depth() < 0.0);
    }

    // 7. capsule_capsule
    #[test]
    fn test_capsule_capsule() {
        // Two vertical capsules side by side. Both at Y axis, half_h=1, r=0.5.
        // Separated by 0.8 in X. Closest segment points are (0,0,0) and (0.8,0,0).
        // Distance=0.8, r_sum=1.0. Overlap!
        let contacts = capsule_capsule(
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
            0.5,
            Vec3::new(0.8, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            0.5,
            0,
            1,
        );
        assert!(!contacts.is_empty());
        let c = &contacts[0];
        assert!(c.depth() < 0.0);
    }

    // 8. plane_sphere on plane
    #[test]
    fn test_plane_sphere_on_plane() {
        let contacts = plane_sphere(
            Vec3::Y,
            0.0,
            Vec3::new(0.0, 0.5, 0.0),
            1.0,
            0,
            1,
        );
        assert_eq!(contacts.len(), 1);
        let c = &contacts[0];
        assert!(approx_eq(c.depth(), -0.5));
        assert!(approx_vec3(c.contact_normal(), Vec3::Y));
    }

    // 9. plane_sphere above
    #[test]
    fn test_plane_sphere_above() {
        let contacts = plane_sphere(
            Vec3::Y,
            0.0,
            Vec3::new(0.0, 2.0, 0.0),
            1.0,
            0,
            1,
        );
        assert_eq!(contacts.len(), 0);
    }

    // 10. plane_box: box resting on plane
    #[test]
    fn test_plane_box() {
        // Box with half-extent 1, centered at y=0.5 on a Y=0 plane.
        // Bottom 4 vertices are at y=-0.5, which are below plane y=0.
        let contacts = plane_box(
            Vec3::Y,
            0.0,
            Vec3::new(0.0, 0.5, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
            0,
            1,
        );
        // 4 vertices below the plane (y = 0.5 - 1.0 = -0.5)
        assert_eq!(contacts.len(), 4);
        for c in &contacts {
            assert!(c.depth() < 0.0);
        }
    }

    // 11. reduce_manifold: 8 contacts down to 4
    #[test]
    fn test_reduce_manifold() {
        // Generate 8 contacts in a ring pattern
        let mut contacts = Vec::new();
        for i in 0..8 {
            let angle = (i as f32) * std::f32::consts::TAU / 8.0;
            let p = Vec3::new(angle.cos(), 0.0, angle.sin());
            let depth = if i == 3 { -1.0 } else { -0.1 };
            contacts.push(make_contact(0, 1, p, Vec3::Y, depth));
        }
        let reduced = reduce_manifold(&contacts);
        assert!(reduced.len() <= 4);
        assert!(!reduced.is_empty());
        // The deepest contact (index 3) should be preserved
        let has_deepest = reduced.iter().any(|c| approx_eq(c.depth(), -1.0));
        assert!(has_deepest);
    }

    // 12. persistence: lambda carry-forward
    #[test]
    fn test_persistence_warmstart() {
        let mut persistence = ContactPersistence::new(0.9);

        // Frame 1: contact with accumulated lambda
        let mut c1 = make_contact(0, 1, Vec3::ZERO, Vec3::Y, -0.1);
        c1.lambda_n = 10.0;
        c1.lambda_t1 = 2.0;
        c1.lambda_t2 = 3.0;
        let (contacts, _events) = persistence.update(vec![c1]);
        // First frame: no previous, so lambdas stay at 0 from make_contact,
        // but we set them above, so update just stores them.
        assert_eq!(contacts.len(), 1);

        // Frame 2: same contact position, should get warm-started lambdas
        let c2 = make_contact(0, 1, Vec3::ZERO, Vec3::Y, -0.1);
        let (contacts2, _events2) = persistence.update(vec![c2]);
        assert_eq!(contacts2.len(), 1);
        assert!(approx_eq(contacts2[0].lambda_n, 10.0 * 0.9));
        assert!(approx_eq(contacts2[0].lambda_t1, 2.0 * 0.9));
        assert!(approx_eq(contacts2[0].lambda_t2, 3.0 * 0.9));
    }

    // 13. collision events: Started/Ended
    #[test]
    fn test_collision_events() {
        let mut persistence = ContactPersistence::new(0.9);

        // Frame 1: bodies 0,1 in contact
        let c1 = make_contact(0, 1, Vec3::ZERO, Vec3::Y, -0.1);
        let (_contacts, events1) = persistence.update(vec![c1]);
        assert_eq!(events1.len(), 1);
        assert!(matches!(
            &events1[0],
            CollisionEvent::Started { body_a, body_b }
            if body_a.index == 0 && body_b.index == 1
        ));

        // Frame 2: bodies 0,1 still in contact, bodies 2,3 start
        let c2a = make_contact(0, 1, Vec3::ZERO, Vec3::Y, -0.1);
        let c2b = make_contact(2, 3, Vec3::X, Vec3::Y, -0.2);
        let (_contacts, events2) = persistence.update(vec![c2a, c2b]);
        // Should have Started for (2,3) only
        assert!(events2.iter().any(|e| matches!(
            e,
            CollisionEvent::Started { body_a, body_b }
            if body_a.index == 2 && body_b.index == 3
        )));
        // No Ended events since (0,1) is still there
        assert!(!events2.iter().any(|e| matches!(e, CollisionEvent::Ended { .. })));

        // Frame 3: only bodies 2,3 remain; bodies 0,1 separated
        let c3 = make_contact(2, 3, Vec3::X, Vec3::Y, -0.2);
        let (_contacts, events3) = persistence.update(vec![c3]);
        // Should have Ended for (0,1)
        assert!(events3.iter().any(|e| matches!(
            e,
            CollisionEvent::Ended { body_a, body_b }
            if body_a.index == 0 && body_b.index == 1
        )));
    }
}

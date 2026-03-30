//! 2D narrowphase collision detection. CPU reference implementation.

use glam::{Vec2, Vec4};
use rubble_math::Contact2D;

fn make_contact2d(body_a: u32, body_b: u32, point: Vec2, normal: Vec2, depth: f32) -> Contact2D {
    Contact2D {
        point: Vec4::new(point.x, point.y, depth, 0.0),
        normal: Vec4::new(normal.x, normal.y, 0.0, 0.0),
        body_a,
        body_b,
        feature_id: 0,
        _pad: 0,
        lambda_n: 0.0,
        lambda_t: 0.0,
        penalty_k: 0.0,
        _pad2: 0.0,
    }
}

/// Circle vs Circle narrowphase collision test.
///
/// Returns a single contact if the circles overlap, or an empty vec if separated.
pub fn circle_circle(
    pos_a: Vec2,
    rad_a: f32,
    pos_b: Vec2,
    rad_b: f32,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact2D> {
    let d = pos_b - pos_a;
    let dist = d.length();
    let r_sum = rad_a + rad_b;
    if dist >= r_sum || dist < 1e-10 {
        return vec![];
    }
    let normal = d / dist;
    let depth = -(r_sum - dist);
    let point = pos_a + normal * (rad_a + depth * 0.5);
    vec![make_contact2d(body_a, body_b, point, normal, depth)]
}

/// Circle vs Rect (closest point on OBB) narrowphase collision test.
///
/// Returns a single contact if the circle overlaps the oriented rectangle.
pub fn circle_rect(
    circle_pos: Vec2,
    radius: f32,
    rect_pos: Vec2,
    rect_angle: f32,
    half: Vec2,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact2D> {
    let cos = rect_angle.cos();
    let sin = rect_angle.sin();
    let d = circle_pos - rect_pos;
    let local = Vec2::new(cos * d.x + sin * d.y, -sin * d.x + cos * d.y);
    let clamped = local.clamp(-half, half);
    let diff = local - clamped;
    let dist_sq = diff.length_squared();

    let (outward_local, depth, contact_local);
    if dist_sq > 1e-8 {
        // Circle center is outside the rect (standard case)
        let dist = dist_sq.sqrt();
        if dist >= radius {
            return vec![];
        }
        outward_local = diff / dist; // points from rect surface toward circle
        depth = -(radius - dist);
        contact_local = clamped;
    } else {
        // Circle center is inside the rect — find nearest face to push out.
        let dx_pos = half.x - local.x;
        let dx_neg = local.x + half.x;
        let dy_pos = half.y - local.y;
        let dy_neg = local.y + half.y;
        let min_dist = dx_pos.min(dx_neg).min(dy_pos).min(dy_neg);
        if min_dist == dx_pos {
            outward_local = Vec2::X;
            contact_local = Vec2::new(half.x, local.y);
        } else if min_dist == dx_neg {
            outward_local = Vec2::NEG_X;
            contact_local = Vec2::new(-half.x, local.y);
        } else if min_dist == dy_pos {
            outward_local = Vec2::Y;
            contact_local = Vec2::new(local.x, half.y);
        } else {
            outward_local = Vec2::NEG_Y;
            contact_local = Vec2::new(local.x, -half.y);
        }
        depth = -(radius + min_dist);
    }

    // Negate normal: solver convention is normal from body_a (circle) toward body_b (rect).
    // outward_local points from rect toward circle, so we negate it.
    let normal_local = -outward_local;
    let normal = Vec2::new(
        cos * normal_local.x - sin * normal_local.y,
        sin * normal_local.x + cos * normal_local.y,
    );
    let closest_world = rect_pos
        + Vec2::new(
            cos * contact_local.x - sin * contact_local.y,
            sin * contact_local.x + cos * contact_local.y,
        );
    vec![make_contact2d(body_a, body_b, closest_world, normal, depth)]
}

/// Rect vs Rect (SAT 4-axis) narrowphase collision test.
///
/// Tests all 4 separating axes (2 from each rectangle's orientation), finds the
/// axis of minimum penetration, and generates a contact point at the midpoint
/// between the two rectangle centers.
#[allow(clippy::too_many_arguments)]
pub fn rect_rect(
    pos_a: Vec2,
    angle_a: f32,
    half_a: Vec2,
    pos_b: Vec2,
    angle_b: f32,
    half_b: Vec2,
    body_a: u32,
    body_b: u32,
) -> Vec<Contact2D> {
    let cos_a = angle_a.cos();
    let sin_a = angle_a.sin();
    let cos_b = angle_b.cos();
    let sin_b = angle_b.sin();

    // Local axes for each rect
    let axes_a = [Vec2::new(cos_a, sin_a), Vec2::new(-sin_a, cos_a)];
    let axes_b = [Vec2::new(cos_b, sin_b), Vec2::new(-sin_b, cos_b)];

    let halves_a = [half_a.x, half_a.y];
    let halves_b = [half_b.x, half_b.y];

    let d = pos_b - pos_a;

    let mut min_overlap = f32::MAX;
    let mut min_axis = Vec2::ZERO;

    // Test axes from rect A
    for i in 0..2 {
        let axis = axes_a[i];
        let proj_a =
            halves_a[0] * axes_a[0].dot(axis).abs() + halves_a[1] * axes_a[1].dot(axis).abs();
        let proj_b =
            halves_b[0] * axes_b[0].dot(axis).abs() + halves_b[1] * axes_b[1].dot(axis).abs();
        let dist = d.dot(axis).abs();
        let overlap = proj_a + proj_b - dist;
        if overlap <= 0.0 {
            return vec![];
        }
        if overlap < min_overlap {
            min_overlap = overlap;
            // Ensure normal points from A to B
            min_axis = if d.dot(axis) >= 0.0 { axis } else { -axis };
        }
    }

    // Test axes from rect B
    for i in 0..2 {
        let axis = axes_b[i];
        let proj_a =
            halves_a[0] * axes_a[0].dot(axis).abs() + halves_a[1] * axes_a[1].dot(axis).abs();
        let proj_b =
            halves_b[0] * axes_b[0].dot(axis).abs() + halves_b[1] * axes_b[1].dot(axis).abs();
        let dist = d.dot(axis).abs();
        let overlap = proj_a + proj_b - dist;
        if overlap <= 0.0 {
            return vec![];
        }
        if overlap < min_overlap {
            min_overlap = overlap;
            min_axis = if d.dot(axis) >= 0.0 { axis } else { -axis };
        }
    }

    let depth = -min_overlap;
    // Contact point: midpoint between the two centers
    let contact_point = (pos_a + pos_b) * 0.5;

    vec![make_contact2d(
        body_a,
        body_b,
        contact_point,
        min_axis,
        depth,
    )]
}

// ---------------------------------------------------------------------------
// Contact persistence for warm-starting across frames
// ---------------------------------------------------------------------------

use rubble_math::{BodyHandle, CollisionEvent};
use std::collections::{HashMap, HashSet};

/// Tracks contacts across frames for warm-starting and collision event generation.
pub struct ContactPersistence2D {
    /// Previously active contact pairs (body_a_idx, body_b_idx).
    previous_pairs: HashSet<(u32, u32)>,
    /// Cached lambda values keyed by (body_a, body_b, feature_id).
    cached_lambdas: HashMap<(u32, u32, u32), (f32, f32)>,
}

impl ContactPersistence2D {
    pub fn new() -> Self {
        Self {
            previous_pairs: HashSet::new(),
            cached_lambdas: HashMap::new(),
        }
    }

    /// Update persistence with new contacts. Returns collision events.
    pub fn update(
        &mut self,
        contacts: &mut [Contact2D],
        handles: &[BodyHandle],
    ) -> Vec<CollisionEvent> {
        let mut events = Vec::new();
        let mut current_pairs = HashSet::new();

        for c in contacts.iter_mut() {
            let pair = (c.body_a.min(c.body_b), c.body_a.max(c.body_b));
            current_pairs.insert(pair);

            // Warm-start from cache
            let key = (pair.0, pair.1, c.feature_id);
            if let Some(&(ln, lt)) = self.cached_lambdas.get(&key) {
                c.lambda_n = ln;
                c.lambda_t = lt;
            }
        }

        // Started events: pairs in current but not in previous
        for &pair in &current_pairs {
            if !self.previous_pairs.contains(&pair) {
                let ha = if (pair.0 as usize) < handles.len() {
                    handles[pair.0 as usize]
                } else {
                    continue;
                };
                let hb = if (pair.1 as usize) < handles.len() {
                    handles[pair.1 as usize]
                } else {
                    continue;
                };
                events.push(CollisionEvent::Started {
                    body_a: ha,
                    body_b: hb,
                });
            }
        }

        // Ended events: pairs in previous but not in current
        for &pair in &self.previous_pairs {
            if !current_pairs.contains(&pair) {
                let ha = if (pair.0 as usize) < handles.len() {
                    handles[pair.0 as usize]
                } else {
                    continue;
                };
                let hb = if (pair.1 as usize) < handles.len() {
                    handles[pair.1 as usize]
                } else {
                    continue;
                };
                events.push(CollisionEvent::Ended {
                    body_a: ha,
                    body_b: hb,
                });
            }
        }

        // Cache current lambdas
        self.cached_lambdas.clear();
        for c in contacts.iter() {
            let pair = (c.body_a.min(c.body_b), c.body_a.max(c.body_b));
            let key = (pair.0, pair.1, c.feature_id);
            self.cached_lambdas.insert(key, (c.lambda_n, c.lambda_t));
        }

        self.previous_pairs = current_pairs;
        events
    }
}

impl Default for ContactPersistence2D {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn test_circle_circle_overlapping() {
        let contacts = circle_circle(Vec2::new(0.0, 0.0), 1.0, Vec2::new(1.5, 0.0), 1.0, 0, 1);
        assert_eq!(contacts.len(), 1);
        let c = &contacts[0];
        assert!(c.depth() < 0.0, "depth should be negative for penetration");
        assert_eq!(c.body_a, 0);
        assert_eq!(c.body_b, 1);
        // Normal should point roughly in +X direction (from A to B)
        assert!(c.contact_normal().x > 0.5);
    }

    #[test]
    fn test_circle_circle_separated() {
        let contacts = circle_circle(Vec2::new(0.0, 0.0), 1.0, Vec2::new(5.0, 0.0), 1.0, 0, 1);
        assert_eq!(contacts.len(), 0);
    }

    #[test]
    fn test_circle_rect_overlapping() {
        // Circle at (1.5, 0) radius 1, rect at origin axis-aligned half=(1,1)
        let contacts = circle_rect(
            Vec2::new(1.5, 0.0),
            1.0,
            Vec2::new(0.0, 0.0),
            0.0,
            Vec2::new(1.0, 1.0),
            0,
            1,
        );
        assert_eq!(contacts.len(), 1);
        let c = &contacts[0];
        assert!(c.depth() < 0.0);
    }

    #[test]
    fn test_circle_rect_separated() {
        let contacts = circle_rect(
            Vec2::new(5.0, 0.0),
            0.5,
            Vec2::new(0.0, 0.0),
            0.0,
            Vec2::new(1.0, 1.0),
            0,
            1,
        );
        assert_eq!(contacts.len(), 0);
    }

    #[test]
    fn test_rect_rect_overlapping() {
        // Two axis-aligned rects overlapping
        let contacts = rect_rect(
            Vec2::new(0.0, 0.0),
            0.0,
            Vec2::new(1.0, 1.0),
            Vec2::new(1.5, 0.0),
            0.0,
            Vec2::new(1.0, 1.0),
            0,
            1,
        );
        assert!(
            !contacts.is_empty(),
            "overlapping rects should produce contacts"
        );
        let c = &contacts[0];
        assert!(c.depth() < 0.0, "depth should be negative for overlap");
    }

    #[test]
    fn test_rect_rect_separated() {
        let contacts = rect_rect(
            Vec2::new(0.0, 0.0),
            0.0,
            Vec2::new(1.0, 1.0),
            Vec2::new(5.0, 0.0),
            0.0,
            Vec2::new(1.0, 1.0),
            0,
            1,
        );
        assert_eq!(contacts.len(), 0);
    }

    #[test]
    fn test_rect_rect_rotated_overlapping() {
        // One rect rotated 45 degrees, should still overlap with nearby rect
        let angle = std::f32::consts::FRAC_PI_4;
        let contacts = rect_rect(
            Vec2::new(0.0, 0.0),
            0.0,
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            angle,
            Vec2::new(1.0, 1.0),
            0,
            1,
        );
        assert!(!contacts.is_empty());
    }

    #[test]
    fn test_contact_persistence_events() {
        let mut persistence = ContactPersistence2D::new();
        let handles = vec![BodyHandle::new(0, 0), BodyHandle::new(1, 0)];

        // First frame: new contact
        let mut contacts = vec![make_contact2d(0, 1, Vec2::ZERO, Vec2::Y, -0.1)];
        let events = persistence.update(&mut contacts, &handles);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], CollisionEvent::Started { .. }));

        // Second frame: same contact, no new events
        let mut contacts2 = vec![make_contact2d(0, 1, Vec2::ZERO, Vec2::Y, -0.05)];
        let events2 = persistence.update(&mut contacts2, &handles);
        assert_eq!(events2.len(), 0);

        // Third frame: contact gone
        let events3 = persistence.update(&mut [], &handles);
        assert_eq!(events3.len(), 1);
        assert!(matches!(events3[0], CollisionEvent::Ended { .. }));
    }
}

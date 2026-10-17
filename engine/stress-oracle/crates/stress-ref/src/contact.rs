//! Contact geometry for the standalone world: oriented boxes, spheres and the ground.
//!
//! Penalty contact is resolved at the stress substep, so impact forces build up and
//! decay over the physical contact duration set by the contact stiffness and masses;
//! there is no impulse divided by a timestep anywhere.
//!
//! Chunks that were one cluster can overlap slightly when they separate (residual bond
//! deformation, rotation, crushing). Per sample point, the world treats a contact born
//! deeper than one substep of approach could produce as pre-existing overlap: it keeps
//! it as an offset that only ratchets down, so a split never converts it into kinetic
//! energy.
//!
//! Box contact uses sample points: the 8 corners pulled 10% towards the centre plus the
//! 6 face centres. Pulling corners inwards avoids double counting where the corners of
//! aligned, equal chunks meet exactly on an edge. Convex hull chunks are sampled the same
//! way (corners pulled in, face centroids) and resolved against their face planes.
//!
//! A compound chunk (a non-convex fragment as the union of non-overlapping convex parts,
//! e.g. a Shardwright fragment) samples every part, in part order, and a point inside it
//! is resolved against the part it lies deepest in. The pair's sample indices run over
//! all parts, so the pre-existing-overlap memory works per part point. `half` bounds the
//! union, which keeps the bounding sphere and the separating-axis rejection valid.

use std::sync::Arc;

use crate::hull::ConvexHull;
use crate::math::{Mat3, Vec3};

/// A chunk's shape in world space: an oriented box, a convex hull, or a compound of
/// convex parts (`half` then bounds the hull or the parts and serves as the thickness
/// for contact stiffness).
#[derive(Clone, Debug)]
pub struct OBox {
    pub center: Vec3,
    pub rotation: Mat3,
    pub half: Vec3,
    pub hull: Option<Arc<ConvexHull>>,
    pub parts: Option<Arc<Vec<Part>>>,
}

/// One convex part of a compound chunk: its hull, centred on the part's centroid, and
/// that centroid in the chunk frame.
#[derive(Clone, Debug)]
pub struct Part {
    pub offset: Vec3,
    pub hull: Arc<ConvexHull>,
}

impl Part {
    /// The part in world space as a hull shape, for a chunk at `center` turned by `rotation`.
    fn shape(&self, center: Vec3, rotation: Mat3) -> OBox {
        OBox { center: center + rotation * self.offset, rotation, half: self.hull.half_extents(), hull: Some(self.hull.clone()), parts: None }
    }

    fn sample_count(&self) -> usize {
        self.hull.vertices.len() + self.hull.faces.len()
    }

    /// Signed distance of `local` (chunk frame) to the part's nearest face, and that face.
    fn signed_distance(&self, local: Vec3) -> (f64, usize) {
        self.hull.signed_distance(local - self.offset)
    }
}

/// A penetrating contact: `normal` points from the first shape towards the second.
#[derive(Clone, Copy, Debug)]
pub struct ContactPoint {
    /// Sample-point index on the first shape (0 for spheres).
    pub index: usize,
    pub point: Vec3,
    pub normal: Vec3,
    pub depth: f64,
}

/// Sample points of a box.
pub const SAMPLE_POINTS: usize = 14;

impl OBox {
    pub fn bounding_radius(&self) -> f64 {
        self.half.norm()
    }

    /// Number of sample points (as `sample_points().len()`).
    pub fn sample_count(&self) -> usize {
        if let Some(parts) = &self.parts {
            return parts.iter().map(Part::sample_count).sum();
        }
        self.hull.as_ref().map_or(SAMPLE_POINTS, |h| h.vertices.len() + h.faces.len())
    }

    /// Sample points (world), in index order.
    pub fn sample_points(&self) -> Vec<Vec3> {
        let mut pts = Vec::with_capacity(self.sample_count());
        self.for_each_sample(|_, p| pts.push(p));
        pts
    }

    /// Calls `f(index, point)` for every sample point (world), in index order, without
    /// collecting them.
    pub fn for_each_sample(&self, mut f: impl FnMut(usize, Vec3)) {
        if let Some(parts) = &self.parts {
            let mut base = 0;
            for part in parts.iter() {
                for (i, p) in part.hull.samples().enumerate() {
                    f(base + i, self.center + self.rotation * (part.offset + p));
                }
                base += part.sample_count();
            }
            return;
        }
        if let Some(h) = &self.hull {
            for (i, p) in h.samples().enumerate() {
                f(i, self.center + self.rotation * p);
            }
            return;
        }
        let h = self.half * 0.9;
        for i in 0..8 {
            let s = Vec3::new(
                if i & 1 == 0 { -h.x } else { h.x },
                if i & 2 == 0 { -h.y } else { h.y },
                if i & 4 == 0 { -h.z } else { h.z },
            );
            f(i, self.center + self.rotation * s);
        }
        for axis in 0..3 {
            for (k, sign) in [(0usize, -1.0f64), (1usize, 1.0f64)] {
                let mut v = Vec3::ZERO;
                v[axis] = sign * self.half[axis];
                f(8 + 2 * axis + k, self.center + self.rotation * v);
            }
        }
    }

    /// If `p` is inside, the depth to the nearest face and that face's outward normal.
    pub fn penetration(&self, p: Vec3) -> Option<(f64, Vec3)> {
        // Outside the bounding sphere (`half` bounds hulls too), with a margin far above
        // the rounding of the rotation: no contact, without the change of frame.
        let r = p - self.center;
        if r.dot(r) > self.half.dot(self.half) * (1.0 + 1e-9) {
            return None;
        }
        let local = self.rotation.transpose() * r;
        if let Some(parts) = &self.parts {
            // The part the point lies deepest in (parts do not overlap: at most one).
            let mut best: Option<(f64, Vec3)> = None;
            for part in parts.iter() {
                let (d, f) = part.signed_distance(local);
                if d < 0.0 && best.is_none_or(|b| -d > b.0) {
                    best = Some((-d, self.rotation * part.hull.faces[f].normal));
                }
            }
            return best;
        }
        if let Some(h) = &self.hull {
            let (d, f) = h.signed_distance(local);
            return (d < 0.0).then(|| (-d, self.rotation * h.faces[f].normal));
        }
        let mut best = f64::INFINITY;
        let mut axis = 0;
        for k in 0..3 {
            let d = self.half[k] - local[k].abs();
            if d <= 0.0 {
                return None;
            }
            if d < best {
                best = d;
                axis = k;
            }
        }
        let mut n = Vec3::ZERO;
        n[axis] = if local[axis] >= 0.0 { 1.0 } else { -1.0 };
        Some((best, self.rotation * n))
    }

    /// Whether the two boxes may overlap (separating-axis test over the 15 face and
    /// edge axes, with a margin far above rounding). False means no sample point of
    /// either can lie inside the other (sample points lie in their box, and `half`
    /// bounds a hull), so their contact is empty.
    pub fn may_overlap(&self, other: &OBox) -> bool {
        let d = other.center - self.center;
        let margin = 1e-9 * (self.half.norm() + other.half.norm());
        let (ax, bx) = ([0, 1, 2].map(|k| self.rotation.col(k)), [0, 1, 2].map(|k| other.rotation.col(k)));
        let radius = |axes: &[Vec3; 3], half: Vec3, l: Vec3| (0..3).map(|k| half[k] * axes[k].dot(l).abs()).sum::<f64>();
        let separates = |l: Vec3| {
            let len = l.norm();
            len > 1e-9 && d.dot(l).abs() > radius(&ax, self.half, l) + radius(&bx, other.half, l) + margin * len
        };
        !(ax.iter().chain(&bx).any(|&l| separates(l)) || ax.iter().any(|a| bx.iter().any(|&b| separates(a.cross(b)))))
    }

    /// Sample points of `self` inside `other`. Normals point from `other` towards `self`
    /// reversed, i.e. from `self` to `other`'s interior is `-normal`: here `normal` is
    /// `other`'s outward face normal, which pushes `self` out of `other`.
    pub fn points_inside(&self, other: &OBox) -> Vec<ContactPoint> {
        let mut out = Vec::new();
        self.for_each_sample(|index, p| {
            if let Some((depth, normal)) = other.penetration(p) {
                out.push(ContactPoint { index, point: p, normal, depth });
            }
        });
        out
    }

    /// Sphere against this box: normal points from the box to the sphere centre.
    pub fn sphere_contact(&self, center: Vec3, radius: f64) -> Option<ContactPoint> {
        if let Some(parts) = &self.parts {
            // The deepest part contact.
            return parts
                .iter()
                .filter_map(|part| part.shape(self.center, self.rotation).sphere_contact(center, radius))
                .max_by(|a, b| a.depth.total_cmp(&b.depth));
        }
        let local = self.rotation.transpose() * (center - self.center);
        if let Some(h) = &self.hull {
            // Nearest face plane (exact over a face, conservative near edges and corners).
            let (d, f) = h.signed_distance(local);
            if d >= radius {
                return None;
            }
            let n = self.rotation * h.faces[f].normal;
            return Some(ContactPoint { index: 0, point: center - n * d.max(0.0), normal: n, depth: radius - d });
        }
        let q = Vec3::new(
            local.x.clamp(-self.half.x, self.half.x),
            local.y.clamp(-self.half.y, self.half.y),
            local.z.clamp(-self.half.z, self.half.z),
        );
        let d = local - q;
        let dist = d.norm();
        if dist > 1e-12 {
            if dist >= radius {
                return None;
            }
            let n = self.rotation * (d / dist);
            return Some(ContactPoint { index: 0, point: self.center + self.rotation * q, normal: n, depth: radius - dist });
        }
        // Centre inside the box: push out through the nearest face.
        let (inside, n) = self.penetration(center)?;
        Some(ContactPoint { index: 0, point: center - n * (radius.min(inside)), normal: n, depth: radius + inside })
    }

    /// Ray (segment `a + t (b - a)`, `t in [0, 1]`) against the box: entry parameter.
    pub fn segment_hit(&self, a: Vec3, b: Vec3) -> Option<f64> {
        if let Some(parts) = &self.parts {
            // The first part the segment enters.
            return parts.iter().filter_map(|part| part.shape(self.center, self.rotation).segment_hit(a, b)).min_by(f64::total_cmp);
        }
        let ro = self.rotation.transpose() * (a - self.center);
        let rd = self.rotation.transpose() * (b - a);
        if let Some(h) = &self.hull {
            // Clip the segment against every face's half-space (Cyrus-Beck).
            let (mut t0, mut t1) = (0.0f64, 1.0f64);
            for f in &h.faces {
                let (num, den) = (f.offset - f.normal.dot(ro), f.normal.dot(rd));
                if den.abs() < 1e-15 {
                    if num < 0.0 {
                        return None;
                    }
                } else if den < 0.0 {
                    t0 = t0.max(num / den);
                } else {
                    t1 = t1.min(num / den);
                }
                if t0 > t1 {
                    return None;
                }
            }
            return Some(t0);
        }
        let mut t0: f64 = 0.0;
        let mut t1: f64 = 1.0;
        for k in 0..3 {
            if rd[k].abs() < 1e-15 {
                if ro[k].abs() > self.half[k] {
                    return None;
                }
            } else {
                let inv = 1.0 / rd[k];
                let (mut ta, mut tb) = ((-self.half[k] - ro[k]) * inv, (self.half[k] - ro[k]) * inv);
                if ta > tb {
                    std::mem::swap(&mut ta, &mut tb);
                }
                t0 = t0.max(ta);
                t1 = t1.min(tb);
                if t0 > t1 {
                    return None;
                }
            }
        }
        Some(t0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box(c: Vec3) -> OBox {
        OBox { center: c, rotation: Mat3::IDENTITY, half: Vec3::splat(0.5), hull: None, parts: None }
    }

    #[test]
    fn sphere_touching_a_face() {
        let b = unit_box(Vec3::ZERO);
        let c = b.sphere_contact(Vec3::new(0.0, 0.0, 0.9), 0.5).unwrap();
        assert!((c.depth - 0.1).abs() < 1e-12);
        assert!((c.normal - Vec3::Z).norm() < 1e-12);
        assert!(b.sphere_contact(Vec3::new(0.0, 0.0, 1.1), 0.5).is_none());
    }

    #[test]
    fn stacked_boxes_overlap_once_per_point() {
        let a = unit_box(Vec3::new(0.0, 0.0, 0.95));
        let b = unit_box(Vec3::ZERO);
        let pts = a.points_inside(&b);
        // Four shrunk bottom corners and the bottom face centre are inside.
        assert_eq!(pts.len(), 5);
        assert!(pts.iter().all(|p| (p.normal - Vec3::Z).norm() < 1e-12));
    }

    /// The separating-axis test never rejects a pair that has a sample point inside the
    /// other box (random sizes, orientations and offsets around touching).
    #[test]
    fn separated_boxes_have_no_contact() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut rejected, mut touching) = (0, 0);
        for _ in 0..20_000 {
            let shape = |rnd: &mut dyn FnMut() -> f64, scale: f64| {
                let axis = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
                let rotation = crate::math::Quat::from_axis_angle(axis, axis.norm() * 6.0).to_mat3();
                let center = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * scale;
                OBox { center, rotation, half: Vec3::new(0.05 + rnd(), 0.05 + rnd(), 0.05 + rnd()) * 0.5, hull: None, parts: None }
            };
            let a = shape(&mut rnd, 0.0);
            let b = shape(&mut rnd, 3.0);
            let contact = !a.points_inside(&b).is_empty() || !b.points_inside(&a).is_empty();
            if !a.may_overlap(&b) {
                rejected += 1;
                assert!(!contact, "separated pair has a contact: {a:?} {b:?}");
            } else if contact {
                touching += 1;
            }
        }
        assert!(rejected > 1000 && touching > 1000, "rejected {rejected} touching {touching}");
    }

    #[test]
    fn segment_hits_box() {
        let b = unit_box(Vec3::ZERO);
        let t = b.segment_hit(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)).unwrap();
        assert!((t - 0.375).abs() < 1e-12);
        assert!(b.segment_hit(Vec3::new(-2.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0)).is_none());
    }
}

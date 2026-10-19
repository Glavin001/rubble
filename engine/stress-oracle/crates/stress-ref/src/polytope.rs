//! Convex polytopes in world space, their intersections and mass properties: the
//! geometry of the elastic-layer contact (`world.rs`, `sim.methods.layer_contact`).
//!
//! Two chunks in contact are rigid bodies separated by a thin elastic layer (the
//! elastic-foundation model of contact mechanics, Johnson, *Contact Mechanics* 4.3):
//! the layer's pressure at a point is `p = k'' delta`, with `delta` the local
//! interpenetration along the contact normal and `k'' = 1 / (h_a / E'_a + h_b / E'_b)`
//! the series stiffness per unit area of the two half-thicknesses along the normal
//! (`E' = E / (1 - nu^2)`). Integrated over the contact,
//!
//! * the normal force is `k'' V`, `V` the overlap volume, acting at the overlap's
//!   centroid (its first moment), and
//! * the stored energy is `k'' int s dV`, `s` the depth below the contact surface,
//!
//! exactly, whatever the overlap's shape, when the contact surface is a plane (a face
//! of the other body, the ground, the tangent plane of a sphere). Between two faces this
//! is the stiffness of the joint the chunks shared (`E A / L` axially, `E I / L` in
//! bending, `L = h_a + h_b`), so a cracked joint and a contact behave alike.
//!
//! Every face of an intersection remembers which body it came from (`owner`): the faces
//! on the other body's surface give the contact normal (their area-weighted outward
//! normal) and the contact area.

use crate::math::{Mat3, Vec3};

/// The body a face of a polytope belongs to.
pub type Owner = u8;

/// A planar convex face: vertices counter-clockwise seen from outside.
#[derive(Clone, Copy, Debug)]
pub struct Face<'a> {
    pub vertices: &'a [Vec3],
    /// Outward unit normal.
    pub normal: Vec3,
    pub owner: Owner,
}

/// Where a face's vertices lie in its polytope's vertex list.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: u32,
    len: u32,
    normal: Vec3,
    owner: Owner,
}

/// A convex polytope as its faces, their vertices in one list (face after face).
#[derive(Debug, Default)]
pub struct Polytope {
    vertices: Vec<Vec3>,
    spans: Vec<Span>,
}

// By hand: `clone_from` reuses the destination's buffers (the derived one reallocates).
impl Clone for Polytope {
    fn clone(&self) -> Self {
        Polytope { vertices: self.vertices.clone(), spans: self.spans.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.vertices.clone_from(&source.vertices);
        self.spans.clone_from(&source.spans);
    }
}

/// Reusable buffers of `Polytope::clip_into`.
#[derive(Default)]
pub struct ClipScratch {
    sides: Vec<f64>,
    kept: Vec<Vec3>,
    segments: Vec<(Vec3, Vec3)>,
    points: Vec<(f64, f64, Vec3)>,
    hull: Vec<(f64, f64, Vec3)>,
    /// The second buffer of `intersect`'s clips.
    spare: Polytope,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<ClipScratch> = std::cell::RefCell::new(ClipScratch::default());
    static FIELD: std::cell::RefCell<FieldScratch> = std::cell::RefCell::new(FieldScratch::default());
    static PAIR: std::cell::RefCell<PairBuffers> = std::cell::RefCell::new(PairBuffers::default());
}

/// Reusable polytopes of one chunk pair's contact (`with_pair_buffers`): the two chunks
/// and their overlap, rebuilt in place pair after pair instead of allocated.
#[derive(Default)]
pub struct PairBuffers {
    pub a: Polytope,
    pub b: Polytope,
    pub overlap: Polytope,
}

/// Runs `f` with this thread's pair buffers (taken out for the call, so a nested call
/// gets fresh ones rather than a borrow conflict).
pub fn with_pair_buffers<R>(f: impl FnOnce(&mut PairBuffers) -> R) -> R {
    let mut buffers = PAIR.with(|p| std::mem::take(&mut *p.borrow_mut()));
    let r = f(&mut buffers);
    PAIR.with(|p| *p.borrow_mut() = buffers);
    r
}

/// Reusable buffers of `field_cells`.
#[derive(Default)]
struct FieldScratch {
    planes: Vec<(Vec3, f64)>,
    range: Vec<(f64, f64)>,
    candidates: Vec<usize>,
    /// (face, (min, max) over the region of `s_f - s_g`) of the bisectors that cut.
    cuts: Vec<(usize, (f64, f64))>,
    region: Polytope,
    cell: Polytope,
    next: Polytope,
}

/// Volume, centroid and second moments about the centroid (`int r r^T dV`).
#[derive(Clone, Copy, Debug)]
pub struct MassProperties {
    pub volume: f64,
    pub centroid: Vec3,
    pub second_moment: Mat3,
}

impl Polytope {
    /// The faces, in order.
    pub fn faces(&self) -> impl ExactSizeIterator<Item = Face<'_>> + Clone + '_ {
        self.spans.iter().map(move |sp| self.face_of(sp))
    }

    fn face_of(&self, sp: &Span) -> Face<'_> {
        Face { vertices: &self.vertices[sp.start as usize..(sp.start + sp.len) as usize], normal: sp.normal, owner: sp.owner }
    }

    pub fn face_count(&self) -> usize {
        self.spans.len()
    }

    /// Every vertex, face after face (a vertex shared by faces appears once per face).
    pub fn all_vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// Appends a face.
    pub fn push_face(&mut self, vertices: &[Vec3], normal: Vec3, owner: Owner) {
        self.spans.push(Span { start: self.vertices.len() as u32, len: vertices.len() as u32, normal, owner });
        self.vertices.extend_from_slice(vertices);
    }

    /// Gives every face `owner`.
    pub fn set_owner(&mut self, owner: Owner) {
        self.spans.iter_mut().for_each(|sp| sp.owner = owner);
    }

    /// The same faces with every vertex mapped by `f`.
    pub fn map_vertices(&self, f: impl Fn(Vec3) -> Vec3) -> Polytope {
        Polytope { vertices: self.vertices.iter().map(|&v| f(v)).collect(), spans: self.spans.clone() }
    }

    fn clear(&mut self) {
        self.vertices.clear();
        self.spans.clear();
    }

    /// An oriented box (half extents `half`, rotation columns its axes, centre `center`).
    pub fn cuboid(center: Vec3, rotation: Mat3, half: Vec3, owner: Owner) -> Polytope {
        let mut out = Polytope { vertices: Vec::with_capacity(24), spans: Vec::with_capacity(6) };
        out.set_cuboid(center, rotation, half, owner);
        out
    }

    /// `cuboid` in place (this polytope's buffers reused).
    pub fn set_cuboid(&mut self, center: Vec3, rotation: Mat3, half: Vec3, owner: Owner) {
        let corner = |sx: f64, sy: f64, sz: f64| center + rotation * Vec3::new(sx * half.x, sy * half.y, sz * half.z);
        let out = self;
        out.clear();
        for axis in 0..3 {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            for sign in [-1.0, 1.0] {
                // Counter-clockwise about the outward normal `sign * e_axis`: (u, v) is
                // right-handed with e_axis, reversed on the negative side.
                let mut quad = [Vec3::ZERO; 4];
                for (k, (a, b)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].into_iter().enumerate() {
                    let mut s = [0.0; 3];
                    s[axis] = sign;
                    s[u] = a;
                    s[v] = b;
                    quad[k] = corner(s[0], s[1], s[2]);
                }
                if sign < 0.0 {
                    quad.reverse();
                }
                let mut n = Vec3::ZERO;
                n[axis] = sign;
                out.push_face(&quad, rotation * n, owner);
            }
        }
    }

    /// A convex hull (hull frame `vertices` and faces) placed at `center` with `rotation`.
    pub fn hull(hull: &crate::hull::ConvexHull, center: Vec3, rotation: Mat3, owner: Owner) -> Polytope {
        let mut out = Polytope::default();
        out.set_hull(hull, center, rotation, owner);
        out
    }

    /// `hull` in place (this polytope's buffers reused).
    pub fn set_hull(&mut self, hull: &crate::hull::ConvexHull, center: Vec3, rotation: Mat3, owner: Owner) {
        self.clear();
        for f in &hull.faces {
            self.spans.push(Span { start: self.vertices.len() as u32, len: f.vertices.len() as u32, normal: rotation * f.normal, owner });
            self.vertices.extend(f.vertices.iter().map(|&i| center + rotation * hull.vertices[i]));
        }
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The part on the side `normal . x <= offset`; the new face (on the plane) gets
    /// `owner`. Empty if nothing remains.
    ///
    /// Each face crossing the plane keeps its inside part, and contributes to the new
    /// face the edge along the plane from where its boundary re-enters to where it
    /// leaves; chained head to tail, those edges are the new face, counter-clockwise
    /// about `normal`. No sorting, tolerance or reference axis is involved, so the result
    /// does not depend on the orientation of the world axes.
    pub fn clip(&self, normal: Vec3, offset: f64, owner: Owner) -> Polytope {
        let mut out = Polytope::default();
        SCRATCH.with(|s| self.clip_into(normal, offset, owner, &mut out, &mut s.borrow_mut()));
        out
    }

    /// `clip` into `out` (its buffers reused), with `scratch` for the intermediates.
    pub fn clip_into(&self, normal: Vec3, offset: f64, owner: Owner, out: &mut Polytope, scratch: &mut ClipScratch) {
        crate::profile::count(crate::profile::Counter::PolytopeClips, 1);
        let side = |p: Vec3| normal.dot(p) - offset;
        out.clear();
        // (entry, exit) on the plane, one per face crossing it.
        scratch.segments.clear();
        let mut cut = false;
        for sp in &self.spans {
            let f = self.face_of(sp);
            let n = f.vertices.len();
            scratch.sides.clear();
            scratch.sides.extend(f.vertices.iter().map(|&p| side(p)));
            let s = &scratch.sides;
            if s.iter().all(|&x| x <= 0.0) {
                out.push_face(f.vertices, f.normal, f.owner);
                continue;
            }
            cut = true;
            let kept = &mut scratch.kept;
            kept.clear();
            let (mut entry, mut exit) = (None, None);
            for i in 0..n {
                let (p, q) = (f.vertices[i], f.vertices[(i + 1) % n]);
                let (sp, sq) = (s[i], s[(i + 1) % n]);
                if sp <= 0.0 {
                    kept.push(p);
                    if sq > 0.0 {
                        // Leaving: at p itself when p lies on the plane.
                        let x = if sp == 0.0 { p } else { p + (q - p) * (sp / (sp - sq)) };
                        if sp < 0.0 {
                            kept.push(x);
                        }
                        exit = Some(x);
                    }
                } else if sq <= 0.0 {
                    // Re-entering: at q itself when q lies on the plane (pushed next).
                    // Interpolated from the inside end, as where leaving: the two faces
                    // sharing an edge traverse it in opposite directions and must make
                    // the same point, bit for bit, for the result to stay watertight.
                    let x = if sq == 0.0 { q } else { q + (p - q) * (sq / (sq - sp)) };
                    if sq < 0.0 {
                        kept.push(x);
                    }
                    entry = Some(x);
                }
            }
            if let (Some(a), Some(b)) = (entry, exit) {
                scratch.segments.push((a, b));
            }
            if kept.len() >= 3 {
                out.push_face(kept, f.normal, f.owner);
            }
        }
        if !cut {
            // Every face was copied above, in order: `out` is already `self`.
            return;
        }
        let ClipScratch { segments, points, hull, kept, .. } = scratch;
        cap_polygon(segments, normal, points, hull, kept);
        if kept.len() >= 3 {
            out.push_face(kept, normal, owner);
        }
        // A remnant with no volume (touching only) is no polytope.
        if out.spans.len() < 4 {
            out.clear();
        }
    }

    /// The intersection with `other` (its faces keep `other`'s owners).
    pub fn intersect(&self, other: &Polytope) -> Polytope {
        let mut out = Polytope::default();
        self.intersect_into(other, &mut out);
        out
    }

    /// `intersect` into `out` (its buffers reused).
    pub fn intersect_into(&self, other: &Polytope, out: &mut Polytope) {
        out.clear();
        if self.outside_a_face_of(other) {
            // The clips would leave nothing (`outside_a_face_of`): skip them.
            return;
        }
        let mut clipped = false;
        SCRATCH.with(|s| {
            let scratch = &mut s.borrow_mut();
            let mut next = std::mem::take(&mut scratch.spare);
            for f in other.faces() {
                let offset = f.normal.dot(f.vertices[0]);
                // The first clip reads `self` directly (a copy of it would clip the same).
                if clipped {
                    out.clip_into(f.normal, offset, f.owner, &mut next, scratch);
                    std::mem::swap(out, &mut next);
                } else {
                    self.clip_into(f.normal, offset, f.owner, out, scratch);
                    clipped = true;
                }
                if out.is_empty() {
                    break;
                }
            }
            scratch.spare = next;
        });
        if !clipped {
            out.clone_from(self);
        }
    }

    /// Whether every vertex lies outside the plane of one face of `other` by more than the
    /// rounding of `intersect`'s clips can move a point: then `intersect` is empty, exactly.
    /// (Its clip by that face finds every vertex of the clipped polytope outside, keeps no
    /// face and leaves nothing.) The clips before that face make points on edges of convex
    /// hulls of this polytope's vertices, each within `12 u r` of the exact point
    /// (`p + (q - p) t`: the difference, the product and the sum each round by `u`
    /// relative, `|p|, |q| <= r`; `u` the unit roundoff, `r` the largest vertex norm),
    /// one level per earlier face; a point's computed side `n . x - o` errs by at most
    /// `3 u (|x| + |o|)` (`|n| = 1` to rounding). The test asks four times the sum of
    /// both. Erring only towards clipping, it never changes a result: false keeps the
    /// clips, true is exact.
    fn outside_a_face_of(&self, other: &Polytope) -> bool {
        if self.vertices.is_empty() || other.spans.is_empty() {
            return false;
        }
        let u = f64::EPSILON / 2.0;
        let r = self.vertices.iter().map(|v| v.norm()).fold(0.0, f64::max);
        let levels = other.spans.len() as f64;
        other.faces().any(|f| {
            let offset = f.normal.dot(f.vertices[0]);
            let margin = 4.0 * u * (12.0 * r * levels + 6.0 * (r + offset.abs()));
            self.vertices.iter().all(|&p| f.normal.dot(p) - offset > margin)
        })
    }

    /// Volume, centroid and second moments (tetrahedra from one vertex).
    pub fn mass_properties(&self) -> Option<MassProperties> {
        let origin = *self.vertices.first()?;
        // Accumulate unscaled determinants before dividing by six. Dividing each
        // signed term first can turn an exactly zero sum into a tiny positive
        // volume, amplifying a residual first moment into an unbounded centroid.
        let (mut six_vol, mut first, mut second) = (0.0, Vec3::ZERO, [[0.0f64; 3]; 3]);
        for f in self.faces() {
            for i in 1..f.vertices.len().saturating_sub(1) {
                let (a, b, c) = (f.vertices[0] - origin, f.vertices[i] - origin, f.vertices[i + 1] - origin);
                let determinant = a.dot(b.cross(c));
                six_vol += determinant;
                first += (a + b + c) * determinant;
                // int x x^T over the tetrahedron (0, a, b, c): v/20 (sum w w^T + s s^T).
                let s = a + b + c;
                for (p, q) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)] {
                    let w = a[p] * a[q] + b[p] * b[q] + c[p] * c[q] + s[p] * s[q];
                    second[p][q] += determinant * w;
                }
            }
        }
        if six_vol <= 0.0 {
            return None;
        }
        let vol = six_vol / 6.0;
        let c = first / (4.0 * six_vol);
        let mut m = [[0.0; 3]; 3];
        for p in 0..3 {
            for q in p..3 {
                // Parallel-axis shift from the origin vertex to the centroid.
                m[p][q] = second[p][q] / 120.0 - vol * c[p] * c[q];
                m[q][p] = m[p][q];
            }
        }
        Some(MassProperties { volume: vol, centroid: origin + c, second_moment: Mat3 { m } })
    }

    /// Sum of `area * outward normal` over the faces of `owner`, and their
    /// area-weighted centroid.
    pub fn owner_surface(&self, owner: Owner) -> (Vec3, Vec3, f64) {
        let (mut an, mut ac, mut a) = (Vec3::ZERO, Vec3::ZERO, 0.0);
        for f in self.faces().filter(|f| f.owner == owner) {
            let (area, centroid) = polygon_area_centroid(&f.vertices);
            an += f.normal * area;
            ac += centroid * area;
            a += area;
        }
        (an, if a > 0.0 { ac / a } else { Vec3::ZERO }, a)
    }

    /// Polar second moment `int |r_perp|^2 dA` of the faces of `owner`, projected on the
    /// plane normal to `axis`, about the axis through `point`.
    pub fn projected_polar_moment(&self, owner: Owner, point: Vec3, axis: Vec3) -> f64 {
        let flat = |x: Vec3| {
            let r = x - point;
            r - axis * r.dot(axis)
        };
        let mut j = 0.0;
        for f in self.faces().filter(|f| f.owner == owner) {
            let p0 = flat(f.vertices[0]);
            for i in 1..f.vertices.len().saturating_sub(1) {
                let (p1, p2) = (flat(f.vertices[i]), flat(f.vertices[i + 1]));
                let area = (p1 - p0).cross(p2 - p0).norm() * 0.5;
                // int |x|^2 over a triangle: A/6 (sum |p_i|^2 + sum_{i<j} p_i . p_j).
                j += area / 6.0 * (p0.dot(p0) + p1.dot(p1) + p2.dot(p2) + p0.dot(p1) + p1.dot(p2) + p2.dot(p0));
            }
        }
        j
    }

    /// The polytope seen along `normal`, about `point` in the plane normal to it: area,
    /// first and second moments of the projection of its faces of `owner` (those facing
    /// along the normal), or with `None` of its whole shadow (every point of which a
    /// convex body's boundary covers twice).
    pub fn shadow(&self, owner: Option<Owner>, normal: Vec3, point: Vec3) -> Shadow {
        let plane = Mat3::IDENTITY - Mat3::outer(normal, normal);
        let mut out = Shadow::default();
        for f in self.faces() {
            let w = match owner {
                Some(o) if f.owner == o => f.normal.dot(normal).max(0.0),
                Some(_) => continue,
                None => 0.5 * f.normal.dot(normal).abs(),
            };
            if w == 0.0 {
                continue;
            }
            // Over the face, y = P (x - point): each fan triangle from its vertices
            // already shifted to `point` and projected, so the second moment is a sum of
            // outer products with positive weights (a sliver's large, nearly edge-on faces
            // would otherwise cancel to round-off of the moment about the origin, and the
            // polar moment could come out negative).
            let y = |x: Vec3| plane * (x - point);
            let n = f.vertices.len();
            if n < 3 {
                continue;
            }
            let ya = y(f.vertices[0]);
            let mut yb = y(f.vertices[1]);
            for i in 1..n - 1 {
                let (a, b, d) = (f.vertices[0], f.vertices[i], f.vertices[i + 1]);
                let t = (b - a).cross(d - a).norm() * 0.5 * w;
                let yd = y(d);
                let sum = ya + yb + yd;
                out.area += t;
                out.first += sum * (t / 3.0);
                out.second += (Mat3::outer(ya, ya) + Mat3::outer(yb, yb) + Mat3::outer(yd, yd) + Mat3::outer(sum, sum)) * (t / 12.0);
                yb = yd;
            }
        }
        out
    }

    /// Mean distance from the axis `(point, axis)`, weighted by volume:
    /// `int |r_perp| dV / V` (tetrahedra from `point` over each face fanned from its
    /// centroid, each split in eight, four-point rule).
    pub fn mean_axis_distance(&self, point: Vec3, axis: Vec3) -> f64 {
        let dist = |x: Vec3| {
            let r = x - point;
            (r - axis * r.dot(axis)).norm()
        };
        let (mut num, mut den) = (0.0, 0.0);
        for f in self.faces() {
            // Fanned from the face's centroid, not a vertex: the rule's error must not
            // depend on where a face's vertex list happens to start.
            let n = f.vertices.len();
            let mid = f.vertices.iter().fold(Vec3::ZERO, |a, &v| a + v) / n as f64;
            for i in 0..n {
                let tet = [point, mid, f.vertices[i], f.vertices[(i + 1) % n]];
                for sub in subdivide(&tet) {
                    let v = (sub[1] - sub[0]).dot((sub[2] - sub[0]).cross(sub[3] - sub[0])).abs() / 6.0;
                    if v == 0.0 {
                        continue;
                    }
                    let mut s = 0.0;
                    for k in 0..4 {
                        // Four-point rule (degree 2): one vertex weighted alpha, the rest beta.
                        let x = sub[k] * TET_ALPHA + (sub[(k + 1) % 4] + sub[(k + 2) % 4] + sub[(k + 3) % 4]) * TET_BETA;
                        s += dist(x);
                    }
                    num += v * s / 4.0;
                    den += v;
                }
            }
        }
        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }
}

const TET_ALPHA: f64 = 0.585_410_196_624_968_5;
const TET_BETA: f64 = 0.138_196_601_125_010_5;

/// The eight tetrahedra of a tetrahedron split at its edge midpoints.
fn subdivide(t: &[Vec3; 4]) -> [[Vec3; 4]; 8] {
    let m = |i: usize, j: usize| (t[i] + t[j]) * 0.5;
    let (m01, m02, m03, m12, m13, m23) = (m(0, 1), m(0, 2), m(0, 3), m(1, 2), m(1, 3), m(2, 3));
    [
        [t[0], m01, m02, m03],
        [m01, t[1], m12, m13],
        [m02, m12, t[2], m23],
        [m03, m13, m23, t[3]],
        [m01, m02, m03, m13],
        [m01, m02, m12, m13],
        [m02, m03, m13, m23],
        [m02, m12, m13, m23],
    ]
}

/// Area and centroid of a planar convex polygon.
pub fn polygon_area_centroid(p: &[Vec3]) -> (f64, Vec3) {
    let (mut area, mut c) = (0.0, Vec3::ZERO);
    for i in 1..p.len().saturating_sub(1) {
        let a = (p[i] - p[0]).cross(p[i + 1] - p[0]).norm() * 0.5;
        area += a;
        c += (p[0] + p[i] + p[i + 1]) * (a / 3.0);
    }
    (area, if area > 0.0 { c / area } else { p.first().copied().unwrap_or(Vec3::ZERO) })
}

/// The new face a clip makes: the convex hull, in the plane, of the points where the
/// faces cross it (counter-clockwise about `normal`). A hull, not a chain of the
/// faces' segments: where the plane runs through vertices and along edges (a bisecting
/// plane through the edge of the two faces it bisects, coplanar faces of neighbouring
/// chunks), round-off can drop or duplicate a segment, and only a hull of the points is
/// immune. Its in-plane basis affects nothing but which of two points that coincide to
/// round-off is kept (the turn of the hull is invariant: its tests are 2D cross
/// products, which a reflection of the basis leaves unchanged).
fn cap_polygon(segments: &[(Vec3, Vec3)], normal: Vec3, pts: &mut Vec<(f64, f64, Vec3)>, hull: &mut Vec<(f64, f64, Vec3)>, ring: &mut Vec<Vec3>) {
    ring.clear();
    let u = normal.any_perpendicular().normalized();
    let v = normal.cross(u);
    pts.clear();
    // Each crossing point is usually both a face's exit and the next face's entry, bit
    // for bit; a repeat is left out, as the deduplication below would drop it (it sorts
    // next to the first, which is the one kept), unless its key is NaN (never equal).
    for p in segments.iter().flat_map(|&(a, b)| [a, b]) {
        let (pu, pv) = (p.dot(u), p.dot(v));
        let same = |q: &(f64, f64, Vec3)| q.2.x.to_bits() == p.x.to_bits() && q.2.y.to_bits() == p.y.to_bits() && q.2.z.to_bits() == p.z.to_bits();
        if pu.is_nan() || pv.is_nan() || !pts.iter().any(same) {
            pts.push((pu, pv, p));
        }
    }
    // Stable insertion sort by (u, v) in `total_cmp` order (as integer keys): a stable
    // sort's result is unique, so this is the order of `sort_by` with that comparator.
    let key = |x: f64| {
        let b = x.to_bits() as i64;
        b ^ (((b >> 63) as u64) >> 1) as i64
    };
    for i in 1..pts.len() {
        let x = pts[i];
        let kx = (key(x.0), key(x.1));
        let mut j = i;
        while j > 0 && (key(pts[j - 1].0), key(pts[j - 1].1)) > kx {
            pts[j] = pts[j - 1];
            j -= 1;
        }
        pts[j] = x;
    }
    pts.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    if pts.len() < 3 {
        return;
    }
    // Andrew's monotone chain, counter-clockwise in (u, v), i.e. about `normal`.
    let cross = |o: &(f64, f64, Vec3), a: &(f64, f64, Vec3), b: &(f64, f64, Vec3)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    hull.clear();
    for p in pts.iter() {
        while hull.len() >= 2 && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }
    let lower = hull.len() + 1;
    for p in pts.iter().rev().skip(1) {
        while hull.len() >= lower && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }
    hull.pop();
    // Start at the hull vertex met first among the segments (their order follows the
    // faces', which turns with the body), not at the extreme of the arbitrary in-plane
    // basis: a turned body then gets the same face, vertex for vertex, and every sum over
    // it rounds identically (half turns bit for bit, DECISIONS.md 17).
    let order = |p: Vec3| segments.iter().flat_map(|&(a, b)| [a, b]).position(|q| q == p).unwrap_or(usize::MAX);
    let start = (0..hull.len()).min_by_key(|&i| order(hull[i].2)).unwrap_or(0);
    hull.rotate_left(start);
    ring.extend(hull.iter().map(|p| p.2));
}

/// A body's share of an overlap that lies nearest one of its faces: the part of the
/// elastic-layer energy `k'' int s dV` (`s` the depth below the body's surface, the
/// distance to its nearest face) belonging to that face.
#[derive(Clone, Copy, Debug)]
pub struct FieldCell {
    /// Index of the body's face.
    pub face: usize,
    pub volume: f64,
    pub centroid: Vec3,
    /// The face's outward unit normal.
    pub normal: Vec3,
    /// `int s dV / V` over the cell.
    pub depth: f64,
    /// The largest depth `s` in the cell (at one of its vertices: `s` is linear there);
    /// zero for a ball's cells (impactors crush by their own law).
    pub max_depth: f64,
    /// Over the cell's boundaries with the cells of other faces (outward normal `m`):
    /// `int s m dA` and `int x x (s m) dA`. Where neighbouring cells have different
    /// stiffnesses, these carry the energy's gradient across the interface.
    pub interface_force: Vec3,
    pub interface_moment: Vec3,
}

/// The overlap `region` (inside `body`) split into the cells of `body`'s faces: where
/// each face is the nearest, `s = o_f - n_f . x` is the depth (the faces receded by
/// `indent`, the crushed surface, and the region clipped to it). The energy
/// `E = k'' int s dV` summed over cells is a potential of the two bodies' relative pose,
/// and by the divergence theorem (`s = 0` on the body's surface) its gradient is the
/// force `k'' V_f n_f` per cell at the cell's centroid, pushing the other body out:
/// exactly conservative, continuous, and between two faces the joint's stiffness. Only
/// faces that can be nearest somewhere in the region are clipped for: face `f` is the
/// nearest at some point only if `min_v s_f(v) <= min_g max_v s_g(v)` over the region's
/// vertices `v` (each `s_g` is linear); with one such face the cell is the region.
/// With a stiffness that differs from face to face, the gradient adds each cell's
/// interface integrals (`FieldCell::interface_force`).
pub fn field_cells(region: &Polytope, body: &Polytope, indent: f64) -> Vec<FieldCell> {
    crate::profile::count(crate::profile::Counter::FieldCells, 1);
    FIELD.with(|fs| SCRATCH.with(|sc| field_cells_in(region, body, indent, &mut fs.borrow_mut(), &mut sc.borrow_mut())))
}

fn field_cells_in(region: &Polytope, body: &Polytope, indent: f64, fs: &mut FieldScratch, sc: &mut ClipScratch) -> Vec<FieldCell> {
    let FieldScratch { planes, range, candidates, cuts, region: receded, cell, next } = fs;
    planes.clear();
    planes.extend(body.faces().map(|f| (f.normal, f.normal.dot(f.vertices[0]) - indent)));
    let region: &Polytope = if indent > 0.0 {
        receded.clone_from(region);
        for &(n, o) in planes.iter() {
            receded.clip_into(n, o, OWNER_B, next, sc);
            std::mem::swap(receded, next);
            if receded.is_empty() {
                return Vec::new();
            }
        }
        receded
    } else {
        region
    };
    let vertices: &[Vec3] = region.all_vertices();
    if vertices.is_empty() {
        return Vec::new();
    }
    let planes = &*planes;
    let depth = |k: usize, x: Vec3| planes[k].1 - planes[k].0.dot(x);
    range.clear();
    range.extend((0..planes.len()).map(|k| vertices.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(depth(k, v)), hi.max(depth(k, v))))));
    let bound = range.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    candidates.clear();
    candidates.extend((0..planes.len()).filter(|&k| range[k].0 <= bound));
    let mut cells = Vec::with_capacity(candidates.len());
    for &f in candidates.iter() {
        // A convex region is on one side of a plane exactly when its vertices are: a
        // bisector it lies wholly inside needs no cut, and one it lies wholly outside
        // leaves the cell empty.
        let gap = |g: usize| vertices.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            let x = depth(f, v) - depth(g, v);
            (lo.min(x), hi.max(x))
        });
        cuts.clear();
        cuts.extend(candidates.iter().copied().filter(|&g| g != f).map(|g| (g, gap(g))).filter(|c| c.1 .1 > 0.0));
        if cuts.iter().any(|c| c.1 .0 > 0.0) {
            continue;
        }
        cell.clone_from(region);
        for &(g, _) in cuts.iter() {
            // s_f <= s_g: (n_g - n_f) . x <= o_g - o_f.
            let d = planes[g].0 - planes[f].0;
            let len = d.norm();
            cell.clip_into(d / len, (planes[g].1 - planes[f].1) / len, INTERFACE + g as Owner, next, sc);
            std::mem::swap(cell, next);
            if cell.is_empty() {
                break;
            }
        }
        if let Some(mp) = cell.mass_properties() {
            if mp.volume > 0.0 {
                let (mut interface_force, mut interface_moment) = (Vec3::ZERO, Vec3::ZERO);
                for face in cell.faces().filter(|x| x.owner >= INTERFACE) {
                    // s = o_f - n_f . x is linear: int s dA = A s(c), int x s dA = o_f A c - S n_f.
                    let (area, c, second) = polygon_moments(&face.vertices);
                    let (n_f, o_f) = planes[f];
                    interface_force += face.normal * (area * (o_f - n_f.dot(c)));
                    let xs = c * (o_f * area) - second * n_f;
                    interface_moment += xs.cross(face.normal);
                }
                let max_depth = cell.all_vertices().iter().map(|&v| depth(f, v)).fold(0.0, f64::max);
                cells.push(FieldCell { face: f, volume: mp.volume, centroid: mp.centroid, normal: planes[f].0, depth: depth(f, mp.centroid), max_depth, interface_force, interface_moment });
            }
        }
    }
    cells
}

/// The elastic-layer contact of a body (`A`) pressed into another (`B`, or a plane):
/// everything the force law needs, from the overlap polytope.
#[derive(Clone, Copy, Debug)]
pub struct LayerContact {
    /// Overlap volume.
    pub volume: f64,
    /// Overlap centroid: the line of action of the normal force.
    pub centroid: Vec3,
    /// Unit contact normal, pushing `A` out of `B`.
    pub normal: Vec3,
    /// Contact area (the overlap's surface on `B`, projected on the normal).
    pub area: f64,
    /// Depth of the centroid below the contact surface (`int s dV = V * depth`).
    pub depth: f64,
    /// Volume-weighted mean distance of the overlap from the normal through the
    /// centroid (the lever arm of torsional friction at full slip).
    pub mean_radius: f64,
    /// Polar second moment of the contact area (projected on the plane normal to
    /// `normal`) about the normal through the centroid: the torsional stiffness of the
    /// shear layer is `k''_t` times it. The trace of `area_moment`.
    pub polar_moment: f64,
    /// The contact area's first and second moments about the centroid, in the plane
    /// normal to `normal`: `int r dA` and `int r r^T dA`, `r` the in-plane offset. The
    /// layer's springs and dashpots spread over the area resist rocking with them.
    pub area_first: Vec3,
    pub area_moment: Mat3,
}

/// Moments of a contact area seen along a normal (`Polytope::shadow`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Shadow {
    pub area: f64,
    pub first: Vec3,
    pub second: Mat3,
}

/// The overlap of two bodies: its depth along the contact normal, and the contact it
/// makes beyond a permanent indentation (`None` when the overlap lies within it).
#[derive(Clone, Copy, Debug)]
pub struct Overlap {
    pub contact: Option<LayerContact>,
    /// Largest thickness of the overlap along the normal.
    pub max_depth: f64,
}

impl LayerContact {
    /// The same contact with its forces acting at `point` (moments of the area shifted
    /// there by the parallel-axis theorem, in the contact plane).
    pub fn about(self, point: Vec3) -> LayerContact {
        let n = self.normal;
        let d = {
            let x = point - self.centroid;
            x - n * x.dot(n)
        };
        let (a, m) = (self.area, self.area_first);
        if a <= 0.0 {
            return LayerContact { centroid: point, ..self };
        }
        // About the area's own centroid (`m / a` from the old point) the moment is
        // central, then the shift adds `a e e^T` (`e` from that centroid to `point`):
        // the polar moment, `|m|^2 / a` below the trace exactly, is never negative, so a
        // round-off excess of `|m|^2 / a` is a zero central polar moment.
        let e = d - m / a;
        let central = self.area_moment - Mat3::outer(m, m) * (1.0 / a);
        let polar = (self.area_moment.trace() - m.norm2() / a).max(0.0) + a * e.norm2();
        let second = central + Mat3::outer(e, e) * a;
        LayerContact { centroid: point, area_first: m - d * a, area_moment: second, polar_moment: polar, ..self }
    }

    /// The contact of polytope `a` with polytope `b` (whose faces carry `OWNER_B`),
    /// beyond a permanent indentation `indent` of `b`'s surface (crushed material that
    /// no longer pushes back: the layer acts only deeper than `indent`). The normal is the
    /// area-weighted outward normal of `b`'s surface inside `a`: it varies continuously
    /// with the geometry (an edge sliver between aligned boxes pushes diagonally, not
    /// along whichever face axis round-off favours). Between two faces it is their
    /// normal, tilted by `delta / w` where one face overhangs the other's edge (`delta`
    /// the overlap depth, `w` the face width): 1e-5 at elastic contact depths.
    pub fn between(a: &Polytope, b: &Polytope, indent: f64) -> Option<Overlap> {
        let overlap = a.intersect(b);
        if overlap.is_empty() {
            return None;
        }
        let (an, _, _) = overlap.owner_surface(OWNER_B);
        let len = an.norm();
        if len == 0.0 {
            return None;
        }
        LayerContact::from_overlap(&overlap, an / len, indent)
    }

    /// The geometry of an overlap seen along a given contact normal, from the overlap
    /// alone (no face ownership, which round-off decides where the two bodies' faces
    /// are coplanar): volume, centroid, its shadow along the normal (area and polar
    /// moment about the normal through the centroid: every point of the shadow is
    /// covered twice by a convex body's boundary), depth below its extreme plane and
    /// the volume-weighted torsional lever arm.
    pub fn with_normal(overlap: &Polytope, normal: Vec3) -> Option<LayerContact> {
        let mp = overlap.mass_properties()?;
        let area = 0.5 * overlap.faces().map(|f| polygon_area_centroid(&f.vertices).0 * f.normal.dot(normal).abs()).sum::<f64>();
        if area <= 0.0 || mp.volume <= 0.0 {
            return None;
        }
        let hi = overlap.faces().flat_map(|f| f.vertices.iter()).map(|v| v.dot(normal)).fold(f64::NEG_INFINITY, f64::max);
        let shadow = overlap.shadow(None, normal, mp.centroid);
        Some(LayerContact {
            volume: mp.volume,
            centroid: mp.centroid,
            normal,
            area,
            depth: (hi - mp.centroid.dot(normal)).max(0.0),
            mean_radius: overlap.mean_axis_distance(mp.centroid, normal),
            polar_moment: shadow.second.trace(),
            area_first: shadow.first,
            area_moment: shadow.second,
        })
    }

    /// The contact of polytope `a` with the half-space `normal . x <= offset` (ground),
    /// beyond a permanent indentation `indent`.
    pub fn with_half_space(a: &Polytope, normal: Vec3, offset: f64, indent: f64) -> Option<Overlap> {
        let overlap = a.clip(normal, offset, OWNER_B);
        if overlap.is_empty() {
            return None;
        }
        LayerContact::from_overlap(&overlap, normal, indent)
    }

    fn from_overlap(overlap: &Polytope, normal: Vec3, indent: f64) -> Option<Overlap> {
        let (lo, hi) = overlap
            .faces()
            .flat_map(|f| f.vertices.iter())
            .map(|v| v.dot(normal))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(x), hi.max(x)));
        let max_depth = (hi - lo).max(0.0);
        // The contact surface is the overlap's extreme plane along the normal, receded by
        // the indentation: the layer acts on the part of the overlap deeper than it.
        let (engaged, top) = if indent > 0.0 { (overlap.clip(normal, hi - indent, OWNER_B), hi - indent) } else { (overlap.clone(), hi) };
        let contact = (|| {
            let mp = engaged.mass_properties()?;
            // Contact area: the engaged overlap's surface on the other body, projected.
            let area: f64 = engaged
                .faces()
                .filter(|f| f.owner == OWNER_B)
                .map(|f| polygon_area_centroid(&f.vertices).0 * f.normal.dot(normal).max(0.0))
                .sum();
            if area <= 0.0 {
                return None;
            }
            let depth = (top - mp.centroid.dot(normal)).max(0.0);
            let mean_radius = engaged.mean_axis_distance(mp.centroid, normal);
            let shadow = engaged.shadow(Some(OWNER_B), normal, mp.centroid);
            Some(LayerContact {
                volume: mp.volume,
                centroid: mp.centroid,
                normal,
                area,
                depth,
                mean_radius,
                polar_moment: shadow.second.trace(),
                area_first: shadow.first,
                area_moment: shadow.second,
            })
        })();
        Some(Overlap { contact, max_depth })
    }

    /// The contact of polytope `a` with a ball `b` (centre `center`, radius `radius`):
    /// the overlap `a ∩ b` in closed form, exact wherever the ball meets the polytope
    /// (a face, an edge, a corner, several chunks at once), so the force is the same
    /// function of the geometry whichever chunk the ball's surface crosses.
    ///
    /// By the divergence theorem over the overlap's boundary, which is the polytope's
    /// faces inside the ball (planar regions `R_f`: a polygon cut by a disc) and the
    /// ball's surface inside the polytope, with `x` from the centre and `h_f` the signed
    /// distance of face `f`'s plane (`x . n_f = h_f` on it):
    ///
    /// * `V = 4/3 pi r^3 [centre inside] + sum_f (h_f A_f - r^3 Omega_f) / 3`, from the
    ///   field `x (1 - r^3 / |x|^3) / 3` (divergence one, zero flux through the sphere),
    ///   `Omega_f = int_R h_f / |x|^3` the solid angle of `R_f`;
    /// * `int x dV = sum_f n_f (int_R |x|^2 / 2 - r^2 A_f / 2)`, from the gradient of
    ///   `|x|^2 / 2`, the sphere's part closed by `int n dA = 0`;
    /// * the ball's surface inside the polytope has area vector `-sum_f n_f A_f`: its
    ///   direction is the normal and its length the contact area; projected along the
    ///   normal it covers what the faces do, which gives its polar moment.
    ///
    /// Each region is summed over its polygon's edges from the foot of the
    /// perpendicular: triangles where an edge runs inside the disc, circular sectors
    /// where it runs outside. The depth is measured from the overlap's extreme plane on
    /// `a`'s side (its face, for a ball on a face: the spherical cap exactly). The
    /// torsional lever arm is the shallow cap's, `8 rho / 15` with `pi rho^2` the area.
    pub fn with_ball(a: &Polytope, center: Vec3, radius: f64) -> Option<LayerContact> {
        let r = radius;
        let mut inside = true;
        let mut through: Vec<Vec3> = Vec::new(); // normals of the faces whose plane holds the centre
        let mut volume = 0.0;
        let mut first = Vec3::ZERO; // int x dV, x from the centre
        let mut area_vector = Vec3::ZERO;
        let mut regions: Vec<(Vec3, Vec3, Vec3, Vec3, DiscRegion)> = Vec::new();
        let mut lo = f64::INFINITY;
        for f in a.faces() {
            let n = f.normal;
            let h = n.dot(f.vertices[0] - center);
            if h < 0.0 {
                inside = false;
            } else if h == 0.0 {
                through.push(n);
            }
            if h.abs() >= r || f.vertices.len() < 3 {
                continue;
            }
            let u = n.any_perpendicular().normalized();
            let v = n.cross(u);
            let foot = center + n * h;
            let poly: Vec<[f64; 2]> = f.vertices.iter().map(|&x| [(x - foot).dot(u), (x - foot).dot(v)]).collect();
            let reg = disc_region(&poly, h, r);
            if reg.area <= 0.0 {
                continue;
            }
            volume += reg.volume;
            first += n * reg.moment;
            area_vector += n * reg.area;
            regions.push((n, foot, u, v, reg));
        }
        if inside {
            // The field's singular point: its share of the ball, the solid angle of the
            // polytope at the centre over 4 pi (all of it inside; on a face, edge or
            // vertex, the face planes through the centre bound it).
            volume += r * r * r / 3.0 * interior_solid_angle(&through);
        }
        let len = area_vector.norm();
        if volume <= 0.0 || len == 0.0 {
            return None;
        }
        let normal = -area_vector / len;
        let centroid = center + first / volume;
        // The overlap's extreme along -normal: the ball's own extreme point if the
        // polytope holds it, else a point of a face region.
        let tip = center - normal * r;
        if a.faces().all(|f| f.normal.dot(tip - f.vertices[0]) <= 0.0) {
            lo = tip.dot(normal);
        }
        // The ball's surface inside the polytope seen along the normal covers what the
        // faces do (their area weighted by -n_f . normal): its moments about the
        // centroid, in the plane normal to `normal`.
        let plane = Mat3::IDENTITY - Mat3::outer(normal, normal);
        let (mut first, mut second) = (Vec3::ZERO, Mat3::default());
        for (n, foot, u, v, reg) in &regions {
            lo = lo.min(reg.min_along([normal.dot(*u), normal.dot(*v)]) + foot.dot(normal));
            let w = -n.dot(normal);
            let y0 = *foot - centroid;
            let m = *u * reg.first[0] + *v * reg.first[1];
            let s = &reg.second;
            let local = Mat3::outer(*u, *u) * s[0][0] + (Mat3::outer(*u, *v) + Mat3::outer(*v, *u)) * s[0][1] + Mat3::outer(*v, *v) * s[1][1];
            let raw = Mat3::outer(y0, y0) * reg.area + Mat3::outer(y0, m) + Mat3::outer(m, y0) + local;
            first += plane * ((y0 * reg.area + m) * w);
            second += plane * raw * plane * w;
        }
        let depth = (centroid.dot(normal) - lo).max(0.0);
        let rho = (len / std::f64::consts::PI).sqrt();
        let polar = second.trace();
        Some(LayerContact { volume, centroid, normal, area: len, depth, mean_radius: 8.0 * rho / 15.0, polar_moment: polar, area_first: first, area_moment: second })
    }

    /// A sphere (centre `center`, radius `radius`) whose surface dips `delta` below the
    /// plane through `surface_point` with outward normal `normal`: the spherical cap.
    pub fn sphere_cap(center: Vec3, radius: f64, normal: Vec3, delta: f64) -> Option<LayerContact> {
        if delta <= 0.0 {
            return None;
        }
        let d = delta.min(2.0 * radius);
        let volume = std::f64::consts::PI * d * d * (3.0 * radius - d) / 3.0;
        // Cap centroid, from the sphere's centre along -normal.
        let from_center = 3.0 * (2.0 * radius - d).powi(2) / (4.0 * (3.0 * radius - d));
        let centroid = center - normal * from_center;
        // The centroid's depth below the plane, `from_center - (radius - d)`, in closed
        // form: the difference of two values of order `radius` loses every digit for a
        // shallow cap.
        let depth = d * (4.0 * radius - d) / (4.0 * (3.0 * radius - d));
        let a = std::f64::consts::PI * (2.0 * radius * d - d * d);
        let rho = (2.0 * radius * d - d * d).sqrt(); // cap base radius
        // The contact area is the disc of radius rho. Over it the cap's thickness is
        // d (1 - (r / rho)^2) for shallow caps, whose volume-weighted mean radius is
        // 8 rho / 15 (exact as d / R -> 0, within a few percent at d = R).
        let mean_radius = 8.0 * rho / 15.0;
        let polar_moment = a * rho * rho / 2.0;
        // The disc, centred on the normal through the centroid: isotropic in its plane.
        let plane = Mat3::IDENTITY - Mat3::outer(normal, normal);
        Some(LayerContact { volume, centroid, normal, area: a, depth, mean_radius, polar_moment, area_first: Vec3::ZERO, area_moment: plane * (polar_moment / 2.0) })
    }
}

/// A convex polygon (counter-clockwise, coordinates from the foot of the perpendicular
/// from a ball's centre to its plane, at signed distance `h`, `|h| < r`) cut by the disc
/// the ball makes in the plane: the region's area, first and second moments about the
/// foot, its share of the overlap's volume and first moment (the divergence theorem's
/// terms, see `LayerContact::with_ball`), and the points where its boundary leaves the
/// polygon's (vertices inside the disc, edge crossings of the circle).
///
/// Summed over the polygon's edges from the foot: a triangle where an edge runs inside
/// the disc, a circular sector where it runs outside. The sectors carry the bulk of a
/// shallow overlap and are written without cancellation in `delta = r - |h|` (a cap of
/// depth `1e-6 r` exact to round-off); a triangle's volume is the difference of its
/// cone and pyramid, `(h A - r^3 Omega) / 3`, good to `1e-16 r / delta` relatively.
/// A centre on the plane (`h = 0`): the principal value, zero (`with_ball` adds the
/// singular point's share).
#[derive(Clone, Debug, Default)]
struct DiscRegion {
    area: f64,
    first: [f64; 2],
    second: [[f64; 2]; 2],
    /// `int_R (h - r^3 h / |x|^3) / 3`: the region's term of the overlap volume.
    volume: f64,
    /// `int_R (|x|^2 - r^2) / 2`: its term of the overlap's first moment (along `n`).
    moment: f64,
    points: Vec<[f64; 2]>,
    rho: f64,
    polygon: Vec<[f64; 2]>,
}

impl DiscRegion {
    /// Smallest `d . p` over the region (`d` a direction in the plane, any length).
    fn min_along(&self, d: [f64; 2]) -> f64 {
        if d == [0.0, 0.0] {
            // Every point of the (non-empty) region is as low as any other.
            return 0.0;
        }
        let mut lo = self.points.iter().map(|p| d[0] * p[0] + d[1] * p[1]).fold(f64::INFINITY, f64::min);
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if len > 0.0 {
            // The disc's own extreme point, where the polygon holds it.
            let e = [-self.rho * d[0] / len, -self.rho * d[1] / len];
            let n = self.polygon.len();
            let held = (0..n).all(|i| {
                let (p, q) = (self.polygon[i], self.polygon[(i + 1) % n]);
                (q[0] - p[0]) * (e[1] - p[1]) - (q[1] - p[1]) * (e[0] - p[0]) >= 0.0
            });
            if held {
                lo = lo.min(-self.rho * len);
            }
        }
        lo
    }
}

/// Solid angle of the cone `{y : n_i . y <= 0}` of the given outward unit normals (the
/// polytope seen from a point of its boundary; `4 pi` with none): a half-space, a wedge
/// (twice its dihedral angle), or a polyhedral cone (its dihedral angles less
/// `(k - 2) pi`, the normals taken in order around their mean).
fn interior_solid_angle(normals: &[Vec3]) -> f64 {
    use std::f64::consts::PI;
    match normals.len() {
        0 => 4.0 * PI,
        1 => 2.0 * PI,
        2 => 2.0 * (PI - normals[0].dot(normals[1]).clamp(-1.0, 1.0).acos()),
        k => {
            let axis = normals.iter().fold(Vec3::ZERO, |a, &n| a + n).normalized();
            let e1 = axis.any_perpendicular().normalized();
            let e2 = axis.cross(e1);
            let mut order: Vec<Vec3> = normals.to_vec();
            order.sort_by(|a, b| a.dot(e2).atan2(a.dot(e1)).total_cmp(&b.dot(e2).atan2(b.dot(e1))));
            let dihedrals: f64 = (0..k).map(|i| PI - order[i].dot(order[(i + 1) % k]).clamp(-1.0, 1.0).acos()).sum();
            dihedrals - (k as f64 - 2.0) * PI
        }
    }
}

fn disc_region(poly: &[[f64; 2]], h: f64, r: f64) -> DiscRegion {
    let delta = r - h.abs();
    let rho2 = delta * (2.0 * r - delta);
    let rho = rho2.sqrt();
    // The side of the plane the centre is on (on it: the principal value, zero).
    let side = if h > 0.0 {
        1.0
    } else if h < 0.0 {
        -1.0
    } else {
        0.0
    };
    let mut reg = DiscRegion { rho, polygon: poly.to_vec(), ..Default::default() };
    let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let dot = |a: [f64; 2], b: [f64; 2]| a[0] * b[0] + a[1] * b[1];
    let triangle = |reg: &mut DiscRegion, a: [f64; 2], b: [f64; 2]| {
        let area = 0.5 * cross(a, b);
        reg.area += area;
        reg.first[0] += area * (a[0] + b[0]) / 3.0;
        reg.first[1] += area * (a[1] + b[1]) / 3.0;
        let c = [a[0] + b[0], a[1] + b[1]];
        for i in 0..2 {
            for j in 0..2 {
                reg.second[i][j] += area / 12.0 * (a[i] * a[j] + b[i] * b[j] + c[i] * c[j]);
            }
        }
        let polar = area / 6.0 * (dot(a, a) + dot(b, b) + dot(a, b));
        reg.moment += 0.5 * (polar - rho2 * area);
        // Solid angle of the triangle (foot, a, b) seen from the centre, signed by
        // orientation and side (Van Oosterom and Strackee); zero on the plane.
        let omega = if h == 0.0 {
            0.0
        } else {
            let (la, lb, l0) = ((dot(a, a) + h * h).sqrt(), (dot(b, b) + h * h).sqrt(), h.abs());
            let den = l0 * la * lb + h * h * (la + lb) + (dot(a, b) + h * h) * l0;
            2.0 * (h * cross(a, b)).atan2(den)
        };
        reg.volume += (h * area - r * r * r * omega) / 3.0;
    };
    // The disc's sector from the unit direction `a` through the angle `dt` to the unit
    // direction `b`: its moments from the directions themselves (`(cos t, sin t)` at
    // either end), not from absolute angles, so they depend on the in-plane basis only
    // through the coordinates (a reflection of the basis maps them exactly; half turns
    // stay bit for bit, DECISIONS.md 17).
    let arc = |reg: &mut DiscRegion, a: [f64; 2], b: [f64; 2], dt: f64| {
        reg.area += 0.5 * rho2 * dt;
        reg.first[0] += rho2 * rho / 3.0 * (b[1] - a[1]);
        reg.first[1] += rho2 * rho / 3.0 * (a[0] - b[0]);
        // (sin 2t2 - sin 2t1) / 4 and (sin^2 t2 - sin^2 t1) / 2.
        let s2 = (b[0] * b[1] - a[0] * a[1]) / 2.0;
        let q = rho2 * rho2 / 4.0;
        reg.second[0][0] += q * (dt / 2.0 + s2);
        reg.second[1][1] += q * (dt / 2.0 - s2);
        let off = q * (b[1] * b[1] - a[1] * a[1]) / 2.0;
        reg.second[0][1] += off;
        reg.second[1][0] += off;
        // (h A - r^3 Omega) / 3 with A = rho^2 dt / 2, Omega = side dt (1 - |h| / r),
        // and (|x|^2 - r^2) / 2 integrated, both reduced exactly.
        reg.volume -= side * dt * delta * delta * (3.0 * r - delta) / 6.0;
        reg.moment -= rho2 * rho2 * dt / 8.0;
    };
    let unit = |a: [f64; 2]| {
        let l = dot(a, a).sqrt();
        [a[0] / l, a[1] / l]
    };
    let sector = |reg: &mut DiscRegion, a: [f64; 2], b: [f64; 2]| {
        let dt = cross(a, b).atan2(dot(a, b));
        // A sector of no angle (or with an end at the foot, where the angle is zero
        // too) contributes nothing.
        if dt != 0.0 {
            arc(reg, unit(a), unit(b), dt);
        }
    };
    let n = poly.len();
    // Where the polygon's boundary never meets the disc, the region is the whole disc
    // (the foot inside the polygon) or nothing: exactly, not as sectors cancelling.
    let meets = (0..n).any(|i| {
        let (p, q) = (poly[i], poly[(i + 1) % n]);
        let d = [q[0] - p[0], q[1] - p[1]];
        let t = if dot(d, d) > 0.0 { (-dot(p, d) / dot(d, d)).clamp(0.0, 1.0) } else { 0.0 };
        let c = [p[0] + d[0] * t, p[1] + d[1] * t];
        dot(c, c) < rho2
    });
    if !meets {
        // The foot inside the polygon (every edge's cross product non-negative) and,
        // since a polygon holding the disc covers it, an area of at least the disc's:
        // a degenerate sliver's cross products are zero or rounding-positive, and the
        // first test alone took the whole disc into it (a 1e-4 m^3 cell, DECISIONS.md 59).
        let crosses: Vec<f64> = (0..n).map(|i| cross(poly[i], poly[(i + 1) % n])).collect();
        let holds = crosses.iter().all(|&c| c >= 0.0) && 0.5 * crosses.iter().sum::<f64>() >= std::f64::consts::PI * rho2;
        if holds {
            arc(&mut reg, [1.0, 0.0], [1.0, 0.0], 2.0 * std::f64::consts::PI);
        }
        return reg;
    }
    for i in 0..n {
        let (p, q) = (poly[i], poly[(i + 1) % n]);
        if dot(p, p) <= rho2 {
            reg.points.push(p);
        }
        // |p + t (q - p)|^2 = rho^2.
        let d = [q[0] - p[0], q[1] - p[1]];
        let (aa, bb, cc) = (dot(d, d), 2.0 * dot(p, d), dot(p, p) - rho2);
        if aa == 0.0 {
            continue;
        }
        let disc = bb * bb - 4.0 * aa * cc;
        if disc <= 0.0 {
            sector(&mut reg, p, q);
            continue;
        }
        // Roots without cancellation: k / aa and cc / k.
        let k = -0.5 * (bb + if bb >= 0.0 { disc.sqrt() } else { -disc.sqrt() });
        let (mut t0, mut t1) = (k / aa, cc / k);
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        let (s0, s1) = (t0.max(0.0), t1.min(1.0));
        if s0 >= s1 {
            sector(&mut reg, p, q);
            continue;
        }
        let at = |t: f64| [p[0] + d[0] * t, p[1] + d[1] * t];
        let (e0, e1) = (if s0 > 0.0 { at(s0) } else { p }, if s1 < 1.0 { at(s1) } else { q });
        if s0 > 0.0 {
            sector(&mut reg, p, e0);
            reg.points.push(e0);
        }
        triangle(&mut reg, e0, e1);
        if s1 < 1.0 {
            reg.points.push(e1);
            sector(&mut reg, e1, q);
        }
    }
    reg
}

/// The overlap of a ball (centre `center`, radius `r`) with `body`, split into the cells
/// of `body`'s faces as in `field_cells`: each cell is the body cut by the bisecting
/// planes towards the other candidate faces, intersected with the ball exactly
/// (`LayerContact::with_ball`), and its interfaces are those planes' polygons cut by the
/// ball's discs. Candidates are found on a polytope holding the overlap: the body cut,
/// along each of its face normals, at the ball's far extent.
pub fn ball_cells(body: &Polytope, center: Vec3, r: f64) -> Vec<FieldCell> {
    let planes: Vec<(Vec3, f64)> = body.faces().map(|f| (f.normal, f.normal.dot(f.vertices[0]))).collect();
    let mut hull = body.clone();
    for &(n, _) in &planes {
        hull = hull.clip(-n, r - n.dot(center), OWNER_B);
        if hull.is_empty() {
            return Vec::new();
        }
    }
    let vertices: Vec<Vec3> = hull.all_vertices().to_vec();
    let depth = |k: usize, x: Vec3| planes[k].1 - planes[k].0.dot(x);
    let range: Vec<(f64, f64)> = (0..planes.len())
        .map(|k| vertices.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(depth(k, v)), hi.max(depth(k, v)))))
        .collect();
    let bound = range.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let candidates: Vec<usize> = (0..planes.len()).filter(|&k| range[k].0 <= bound).collect();
    let mut cells = Vec::with_capacity(candidates.len());
    for &f in &candidates {
        let mut cell = body.clone();
        for &g in candidates.iter().filter(|&&g| g != f) {
            let d = planes[g].0 - planes[f].0;
            let len = d.norm();
            cell = cell.clip(d / len, (planes[g].1 - planes[f].1) / len, INTERFACE + g as Owner);
            if cell.is_empty() {
                break;
            }
        }
        if cell.is_empty() {
            continue;
        }
        let Some(lc) = LayerContact::with_ball(&cell, center, r) else { continue };
        let (n_f, o_f) = planes[f];
        let (mut interface_force, mut interface_moment) = (Vec3::ZERO, Vec3::ZERO);
        for face in cell.faces().filter(|x| x.owner >= INTERFACE) {
            let Some((area, first, second)) = disc_moments(&face, center, r) else { continue };
            interface_force += face.normal * (o_f * area - n_f.dot(first));
            let xs = first * o_f - second * n_f;
            interface_moment += xs.cross(face.normal);
        }
        cells.push(FieldCell { face: f, volume: lc.volume, centroid: lc.centroid, normal: n_f, depth: o_f - n_f.dot(lc.centroid), max_depth: 0.0, interface_force, interface_moment });
    }
    cells
}

/// Area, first moment and second moment `int x x^T dA` (about the origin) of a planar
/// convex polygon cut by a ball (`None` when the ball misses its plane).
fn disc_moments(face: &Face, center: Vec3, r: f64) -> Option<(f64, Vec3, Mat3)> {
    let n = face.normal;
    let h = n.dot(face.vertices[0] - center);
    if h.abs() >= r || face.vertices.len() < 3 {
        return None;
    }
    let u = n.any_perpendicular().normalized();
    let v = n.cross(u);
    let foot = center + n * h;
    let poly: Vec<[f64; 2]> = face.vertices.iter().map(|&x| [(x - foot).dot(u), (x - foot).dot(v)]).collect();
    let reg = disc_region(&poly, h, r);
    if reg.area <= 0.0 {
        return None;
    }
    let m = u * reg.first[0] + v * reg.first[1];
    let sq = &reg.second;
    let local = Mat3::outer(u, u) * sq[0][0] + (Mat3::outer(u, v) + Mat3::outer(v, u)) * sq[0][1] + Mat3::outer(v, v) * sq[1][1];
    let second = Mat3::outer(foot, foot) * reg.area + Mat3::outer(foot, m) + Mat3::outer(m, foot) + local;
    Some((reg.area, foot * reg.area + m, second))
}

/// Area, centroid and second moment `int x x^T dA` (about the origin) of a planar
/// convex polygon.
fn polygon_moments(p: &[Vec3]) -> (f64, Vec3, Mat3) {
    let (mut area, mut c, mut second) = (0.0, Vec3::ZERO, Mat3::default());
    for i in 1..p.len().saturating_sub(1) {
        let (a, b, d) = (p[0], p[i], p[i + 1]);
        let t = (b - a).cross(d - a).norm() * 0.5;
        area += t;
        c += (a + b + d) * (t / 3.0);
        let sum = a + b + d;
        second += (Mat3::outer(a, a) + Mat3::outer(b, b) + Mat3::outer(d, d) + Mat3::outer(sum, sum)) * (t / 12.0);
    }
    (area, if area > 0.0 { c / area } else { Vec3::ZERO }, second)
}

/// Owner tags of the faces `field_cells` cuts between cells: `INTERFACE + g`, `g` the
/// neighbouring cell's face.
pub const INTERFACE: Owner = 2;

/// Owner tags: the body the force acts on and the body it presses into.
pub const OWNER_A: Owner = 0;
pub const OWNER_B: Owner = 1;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;

    fn cube(c: Vec3, h: f64) -> Polytope {
        Polytope::cuboid(c, Mat3::IDENTITY, Vec3::splat(h), OWNER_A)
    }

    use std::f64::consts::PI;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// A sliver (an edge or a slightly tilted face just touching another body, away from
    /// the origin) has a contact area whose second moment is positive semi-definite and
    /// whose polar moment is never negative, about its centroid and about any point
    /// (exact properties of `int r r^T dA`; computing them about the origin and shifting
    /// cancelled to the size of the true value, gave a negative torsional stiffness and
    /// a NaN moment).
    #[test]
    fn a_sliver_contact_has_a_positive_area_moment() {
        let b = Polytope::cuboid(Vec3::new(0.1, 0.1, -0.05), Mat3::IDENTITY, Vec3::splat(0.05), OWNER_B);
        for tilt in [0.7, 1e-2, 1e-4] {
            for depth in [1e-9, 1e-7, 1e-5] {
                let rot = Quat::from_axis_angle(Vec3::new(1.0, 0.3, 0.0).normalized(), tilt).to_mat3();
                let half = Vec3::splat(0.05);
                let low = (0..8)
                    .map(|i| (rot * Vec3::new(if i & 1 == 0 { -half.x } else { half.x }, if i & 2 == 0 { -half.y } else { half.y }, if i & 4 == 0 { -half.z } else { half.z })).z)
                    .fold(f64::INFINITY, f64::min);
                let a = Polytope::cuboid(Vec3::new(0.11, 0.09, -low - depth), rot, half, OWNER_A);
                let Some(lc) = LayerContact::between(&a, &b, 0.0).and_then(|o| o.contact) else { continue };
                let m = lc.area_moment;
                let label = format!("tilt {tilt}, depth {depth}");
                assert!(lc.polar_moment >= 0.0 && m.m[0][0] >= 0.0 && m.m[1][1] >= 0.0 && m.m[2][2] >= 0.0, "{label}: {lc:?}");
                for k in 0..3 {
                    for l in 0..3 {
                        assert!(m.m[k][l] * m.m[k][l] <= m.m[k][k] * m.m[l][l] * (1.0 + 8.0 * f64::EPSILON), "{label}: not semi-definite: {m:?}");
                    }
                }
                for shift in [Vec3::new(1e-6, -2e-6, 0.0), Vec3::new(0.01, 0.0, 0.003), Vec3::ZERO] {
                    let p = lc.about(lc.centroid + shift);
                    assert!(p.polar_moment >= 0.0, "{label}: shifted by {shift:?}: {}", p.polar_moment);
                }
            }
        }
    }

    #[test]
    fn a_ball_on_a_face_is_the_spherical_cap() {
        let rot = Quat::from_axis_angle(Vec3::new(0.3, -1.0, 0.4).normalized(), 0.7).to_mat3();
        let center = Vec3::new(0.2, -0.1, 0.3);
        let block = Polytope::cuboid(center, rot, Vec3::new(2.0, 1.5, 1.0), OWNER_A);
        let up = rot.col(2);
        for (delta, x, y) in [(1e-6, 0.0, 0.0), (1e-3, 0.3, -0.2), (0.05, -0.4, 0.1), (0.1, 0.0, 0.5)] {
            let r = 0.1;
            let ball = center + rot.col(0) * x + rot.col(1) * y + up * (1.0 + r - delta);
            let c = LayerContact::with_ball(&block, ball, r).unwrap();
            let cap = LayerContact::sphere_cap(ball, r, up, delta).unwrap();
            let cap = LayerContact { normal: -cap.normal, ..cap };
            assert!(rel(c.volume, cap.volume) < 1e-9, "{delta}: {} vs {}", c.volume, cap.volume);
            assert!((c.centroid - cap.centroid).norm() < 1e-12, "{delta}: {:?} vs {:?}", c.centroid, cap.centroid);
            assert!((c.normal - cap.normal).norm() < 1e-12);
            assert!(rel(c.area, cap.area) < 1e-9, "{delta}: {} vs {}", c.area, cap.area);
            assert!((c.depth - cap.depth).abs() < 1e-12 * r, "{delta}: {} vs {}", c.depth, cap.depth);
            assert!(rel(c.polar_moment, cap.polar_moment) < 1e-9, "{delta}: {} vs {}", c.polar_moment, cap.polar_moment);
        }
    }

    #[test]
    fn a_ball_at_a_face_edge_or_corner_takes_its_exact_share() {
        // Centred on a face, an edge, a corner: a half, quarter, eighth ball.
        let (r, h) = (0.3, 1.0);
        let block = cube(Vec3::ZERO, h);
        let v = 4.0 / 3.0 * PI * r * r * r;
        let cases = [
            (Vec3::new(0.0, 0.0, h), v / 2.0, Vec3::new(0.0, 0.0, -3.0 * r / 8.0)),
            // (each slice across the edge is cut symmetrically: the half ball's 3r/8)
            (Vec3::new(0.0, h, h), v / 4.0, Vec3::new(0.0, -1.0, -1.0) * (3.0 * r / 8.0)),
            (Vec3::new(h, h, h), v / 8.0, Vec3::new(-1.0, -1.0, -1.0) * (3.0 * r / 8.0)),
        ];
        for (at, vol, offset) in cases {
            let c = LayerContact::with_ball(&block, at, r).unwrap();
            assert!(rel(c.volume, vol) < 1e-12, "{at:?}: {} vs {vol}", c.volume);
            assert!((c.centroid - (at + offset)).norm() < 1e-12, "{at:?}: {:?}", c.centroid - at);
            assert!((c.normal - offset.normalized()).norm() < 1e-12, "{at:?}: {:?}", c.normal);
        }
        // Whole ball inside; whole cube inside.
        let c = LayerContact::with_ball(&block, Vec3::new(0.1, -0.2, 0.3), r);
        assert!(c.is_none(), "no surface of the ball in contact");
        let small = cube(Vec3::new(0.01, 0.02, -0.03), 0.1);
        let c = LayerContact::with_ball(&small, Vec3::ZERO, 1.0);
        assert!(c.is_none(), "a cube inside the ball has no ball surface inside it");
    }

    /// Gauss-Legendre nodes and weights on [-1, 1] (Newton on the Legendre polynomial).
    fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
                loop {
                    let (mut p0, mut p1) = (1.0, x);
                    for k in 2..=n {
                        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                        p0 = p1;
                        p1 = p2;
                    }
                    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                    let dx = p1 / dp;
                    x -= dx;
                    if dx.abs() <= 1e-16 {
                        return (x, 2.0 / ((1.0 - x * x) * dp * dp));
                    }
                }
            })
            .collect()
    }

    /// `int f(t) dt` over [a, b] split at `kinks`, `n`-point Gauss-Legendre on each piece,
    /// and the sum of `|f| w` (the scale of the sum's rounding).
    fn integrate(f: &dyn Fn(f64) -> [f64; 4], a: f64, b: f64, kinks: &[f64], n: usize) -> ([f64; 4], f64) {
        let mut cuts: Vec<f64> = kinks.iter().copied().filter(|&t| t > a && t < b).collect();
        cuts.push(a);
        cuts.push(b);
        cuts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let rule = gauss_legendre(n);
        let (mut out, mut scale) = ([0.0; 4], 0.0);
        for w in cuts.windows(2) {
            let (mid, half) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]));
            for &(x, wt) in &rule {
                let v = f(mid + half * x);
                for k in 0..4 {
                    out[k] += v[k] * wt * half;
                }
                scale += v[0].abs() * wt * half;
            }
        }
        (out, scale)
    }

    /// Volume and first moment `int x dV` of an axis-aligned box (half-sizes `half`, at
    /// the origin) cut by a ball, by nested quadrature over slices: `x = c_x - r cos t`
    /// and, in a slice of radius `rho`, `y = c_y - rho cos u` take the square roots out
    /// of the rims, and each integral is split where the integrand has a kink (the circle
    /// reaching a box edge or corner, a slice reaching a box face), so every piece is
    /// smooth and Gauss-Legendre converges exponentially. Returns `[V, int x, int y,
    /// int z]` and the scale of the sums (for their rounding).
    fn ball_box_reference(half: Vec3, c: Vec3, r: f64, n: usize) -> ([f64; 4], f64) {
        let arcs = |d: f64, rho: f64| -> Vec<f64> {
            if d > 0.0 && d < rho {
                let t = (d / rho).asin();
                vec![t, PI - t]
            } else {
                vec![]
            }
        };
        let slice = |rho: f64| -> [f64; 4] {
            if rho <= 0.0 {
                return [0.0; 4];
            }
            let mut kinks = Vec::new();
            for y in [-half.y, half.y] {
                let q = (c.y - y) / rho;
                if q.abs() < 1.0 {
                    kinks.push(q.acos());
                }
            }
            for d in [half.z - c.z, half.z + c.z] {
                kinks.extend(arcs(d.abs(), rho));
            }
            let f = |u: f64| -> [f64; 4] {
                let y = c.y - rho * u.cos();
                if y.abs() > half.y {
                    return [0.0; 4];
                }
                let s = rho * u.sin();
                let (lo, hi) = ((c.z - s).max(-half.z), (c.z + s).min(half.z));
                if hi <= lo {
                    return [0.0; 4];
                }
                let a = (hi - lo) * rho * u.sin();
                [a, 0.0, y * a, 0.5 * (lo + hi) * a]
            };
            integrate(&f, 0.0, PI, &kinks, n).0
        };
        let mut kinks = Vec::new();
        for x in [-half.x, half.x] {
            let q = (c.x - x) / r;
            if q.abs() < 1.0 {
                kinks.push(q.acos());
            }
        }
        for dy in [half.y - c.y, half.y + c.y] {
            for dz in [half.z - c.z, half.z + c.z] {
                kinks.extend(arcs(dy.abs(), r));
                kinks.extend(arcs(dz.abs(), r));
                kinks.extend(arcs((dy * dy + dz * dz).sqrt(), r));
            }
        }
        let f = |t: f64| -> [f64; 4] {
            let x = c.x - r * t.cos();
            if x.abs() > half.x {
                return [0.0; 4];
            }
            let s = slice(r * t.sin());
            let j = r * t.sin();
            [s[0] * j, x * s[0] * j, s[2] * j, s[3] * j]
        };
        integrate(&f, 0.0, PI, &kinks, n)
    }

    /// A ball over an edge or a corner of a box against an independent reference (nested
    /// Gauss-Legendre over the exact slices, `ball_box_reference`): volume and centroid
    /// agree within the reference's own uncertainty (64 against 128 points per piece, and
    /// the worst-case rounding of its sums, `n eps` of their scale) plus `with_ball`'s
    /// rounding (measured by moving the whole scene: 19 fixed shifts).
    #[test]
    fn a_ball_over_an_edge_matches_an_independent_integration() {
        let half = Vec3::new(0.4, 0.3, 0.25);
        let block = |x: Vec3| {
            let b = cube(Vec3::ZERO, 1.0);
            b.map_vertices(|v| v.mul_elem(half) + x)
        };
        for (center, r) in [(Vec3::new(0.35, 0.33, 0.2), 0.12), (Vec3::new(0.45, -0.31, 0.28), 0.1), (Vec3::new(0.1, 0.0, 0.31), 0.08)] {
            let ((coarse, _), (fine, scale)) = (ball_box_reference(half, center, r, 64), ball_box_reference(half, center, r, 128));
            // Nested recursive sums round by at most (inner terms + outer terms + the few
            // roundings forming each term) eps of the sum of magnitudes (Higham): 128 nodes
            // on each piece, at most 27 outer pieces (2 face and 24 rim kinks) and 7
            // inner ones (2 face and 4 rim kinks).
            let terms = 128.0 * (27.0 + 7.0) + 16.0;
            let reference_error = |k: usize| (fine[k] - coarse[k]).abs() + terms * f64::EPSILON * scale * if k == 0 { 1.0 } else { half.norm() + r };
            let measure = |x: Vec3| {
                let c = LayerContact::with_ball(&block(x), center + x, r).unwrap();
                [c.volume, c.centroid.x - x.x, c.centroid.y - x.y, c.centroid.z - x.z]
            };
            let got = measure(Vec3::ZERO);
            let mut noise = [0.0f64; 4];
            let mut seed = 0x9e37_79b9_7f4a_7c15u64;
            for _ in 0..19 {
                let mut rnd = || {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.2
                };
                let m = measure(Vec3::new(rnd(), rnd(), rnd()));
                for k in 0..4 {
                    noise[k] = noise[k].max((m[k] - got[k]).abs());
                }
            }
            let v = fine[0];
            assert!((got[0] - v).abs() <= reference_error(0) + noise[0], "{center:?}: volume {} vs {v} (uncertainty {:.1e} + {:.1e})", got[0], reference_error(0), noise[0]);
            for k in 1..4 {
                let want = fine[k] / v;
                // The centroid's uncertainty from the moment's and the volume's.
                let allowed = (reference_error(k) + reference_error(0) * want.abs()) / v + noise[k];
                println!("{center:?} centroid {k}: {:.3e} off, uncertainty {allowed:.1e}", (got[k] - want).abs());
                assert!((got[k] - want).abs() <= allowed, "{center:?}: centroid {k}: {} vs {want} (uncertainty {allowed:.1e})", got[k]);
            }
            println!("{center:?}: volume {:.3e} off of {v:.6e}, uncertainty {:.1e} + {:.1e}", (got[0] - v).abs(), reference_error(0), noise[0]);
        }
    }

    #[test]
    fn ball_overlaps_add_over_a_split_polytope() {
        // A box cut in two (in any plane): the halves' overlaps with any ball add up to
        // the whole box's, volume and first moment, whatever edges the ball crosses.
        let rot = Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.2).normalized(), 0.35).to_mat3();
        let whole = Polytope::cuboid(Vec3::new(0.05, 0.0, -0.02), rot, Vec3::new(0.3, 0.2, 0.15), OWNER_A);
        let cut_n = Vec3::new(0.2, 1.0, 0.1).normalized();
        let (left, right) = (whole.clip(cut_n, 0.01, OWNER_A), whole.clip(-cut_n, -0.01, OWNER_A));
        let mut seed = 0x1234_5678u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut tested = 0;
        for _ in 0..400 {
            let center = Vec3::new(rnd(), rnd(), rnd()) * 1.0;
            let r = 0.05 + 0.3 * (rnd() + 0.5);
            let moments = |p: &Polytope| LayerContact::with_ball(p, center, r).map_or((0.0, Vec3::ZERO), |c| (c.volume, c.centroid * c.volume));
            let (vw, mw) = moments(&whole);
            let ((vl, ml), (vr, mr)) = (moments(&left), moments(&right));
            if vw == 0.0 {
                continue;
            }
            tested += 1;
            assert!((vl + vr - vw).abs() < 1e-12 * r * r * r, "volume {vl} + {vr} vs {vw}");
            assert!((ml + mr - mw).norm() < 1e-12 * r * r * r, "first moment");
        }
        assert!(tested > 100, "{tested}");
    }

    #[test]
    fn a_plane_through_an_edge_cuts_an_exact_prism() {
        // A thin slab cut by a 45-degree plane through one of its long edges: a prism of
        // right-triangle section (legs the slab's thickness) along the edge, whichever
        // edge and side.
        let (lo, hi) = (Vec3::new(-0.02, -0.04, 0.0499), Vec3::new(0.05, 0.05, 0.05));
        let slab = Polytope::cuboid((lo + hi) * 0.5, Mat3::IDENTITY, (hi - lo) * 0.5, OWNER_A);
        let t = hi.z - lo.z;
        let cuts = [
            (Vec3::new(1.0, 0.0, -1.0), lo.x - lo.z, hi.y - lo.y),
            (Vec3::new(-1.0, 0.0, 1.0), -lo.x + lo.z, hi.y - lo.y),
            (Vec3::new(0.0, -1.0, -1.0), -hi.y - hi.z + t, hi.x - lo.x),
            (Vec3::new(-1.0, 0.0, -1.0), -hi.x - hi.z + t, hi.y - lo.y),
        ];
        for (n, o, len) in cuts {
            let len_n = n.norm();
            let cut = slab.clip(n / len_n, o / len_n, OWNER_B);
            let v = cut.mass_properties().map_or(0.0, |m| m.volume);
            let full = slab.mass_properties().unwrap().volume;
            let prism = 0.5 * t * t * len;
            let got = v.min(full - v);
            assert!((got - prism).abs() < 1e-9 * prism, "{n:?}: {got:e} vs {prism:e}");
        }
    }

    #[test]
    fn tiny_positive_volumes_are_retained_without_a_cutoff() {
        for scale in [1.0, 1e-3, 1e-6] {
            let half = Vec3::new(1e-12, 1e-10, 1e-8) * scale;
            let p = Polytope::cuboid(Vec3::ZERO, Mat3::IDENTITY, half, OWNER_A);
            let mp = p.mass_properties().expect("a thin solid still has volume");
            let expected = 8.0 * half.x * half.y * half.z;
            assert!((mp.volume - expected).abs() <= expected * 1e-12);
            assert!(mp.centroid.norm() <= half.norm() * 1e-12);
            for i in 0..3 {
                let expected_second = expected * half[i] * half[i] / 3.0;
                assert!((mp.second_moment.m[i][i] - expected_second).abs() <= expected_second * 1e-12);
            }
        }
    }

    #[test]
    fn cuboid_mass_properties_are_exact() {
        let half = Vec3::new(0.5, 1.0, 1.5);
        let r = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalized(), 0.9).to_mat3();
        let p = Polytope::cuboid(Vec3::new(1.0, -2.0, 3.0), r, half, OWNER_A);
        let mp = p.mass_properties().unwrap();
        assert!((mp.volume - 6.0).abs() < 1e-12, "{}", mp.volume);
        assert!((mp.centroid - Vec3::new(1.0, -2.0, 3.0)).norm() < 1e-12);
        // Body-frame second moments V a^2 / 3 along each axis.
        let local = r.transpose() * mp.second_moment * r;
        for (k, h) in [half.x, half.y, half.z].iter().enumerate() {
            assert!((local.m[k][k] - 6.0 * h * h / 3.0).abs() < 1e-10, "{k}: {}", local.m[k][k]);
        }
        for f in p.faces() {
            let (_, c) = polygon_area_centroid(&f.vertices);
            assert!(f.normal.dot(c - Vec3::new(1.0, -2.0, 3.0)) > 0.0, "outward normals");
        }
    }

    #[test]
    fn a_resting_box_overlaps_the_ground_over_its_whole_face() {
        // Box of half 0.1 sunk by 1e-3 into the ground z <= 0.
        let p = cube(Vec3::new(0.3, -0.2, 0.1 - 1e-3), 0.1);
        let c = LayerContact::with_half_space(&p, Vec3::Z, 0.0, 0.0).unwrap().contact.unwrap();
        assert!((c.volume - 0.04 * 1e-3).abs() < 1e-15, "{}", c.volume);
        assert!((c.area - 0.04).abs() < 1e-12);
        assert!((c.normal - Vec3::Z).norm() < 1e-12);
        assert!((c.centroid - Vec3::new(0.3, -0.2, -0.5e-3)).norm() < 1e-12);
        assert!((c.depth - 0.5e-3).abs() < 1e-12);
        // Square of side 0.2: polar moment a^4 / 6, mean distance 0.3826 * a.
        assert!((c.polar_moment - 0.2f64.powi(4) / 6.0).abs() < 1e-9, "{}", c.polar_moment);
        assert!((c.mean_radius - 0.382_597_858_6 * 0.2).abs() < 2e-3 * 0.2, "{}", c.mean_radius / 0.2);
    }

    #[test]
    fn a_tilted_box_overlaps_the_ground_in_a_wedge() {
        // Unit cube tilted by t about y, its lowest edge 1e-2 below the ground: the
        // overlap is a triangular prism.
        let t = 0.1f64;
        let r = Quat::from_axis_angle(Vec3::Y, t).to_mat3();
        let half = 0.5;
        // Lowest corner height relative to the centre: -half (cos t + sin t).
        let low = half * (t.cos() + t.sin());
        let d = 1e-2;
        let p = Polytope::cuboid(Vec3::new(0.0, 0.0, low - d), r, Vec3::splat(half), OWNER_A);
        let c = LayerContact::with_half_space(&p, Vec3::Z, 0.0, 0.0).unwrap().contact.unwrap();
        // Cross-section in x-z: triangle with legs d / sin t and d / cos t, depth 1.
        let expected = 0.5 * (d / t.sin()) * (d / t.cos()) * 1.0;
        assert!((c.volume - expected).abs() < 1e-12, "{} vs {expected}", c.volume);
        assert!((c.normal - Vec3::Z).norm() < 1e-12);
        // Triangle centroid one third of the way up from the edge: depth d / 3.
        assert!((c.depth - d / 3.0).abs() < 1e-10, "{}", c.depth);
    }

    #[test]
    fn overlaps_are_symmetric_and_face_contact_is_exact() {
        let a = cube(Vec3::new(0.0, 0.0, 0.199), 0.1);
        let mut b = cube(Vec3::new(0.05, 0.0, 0.0), 0.1);
        b.set_owner(OWNER_B);
        let ab = LayerContact::between(&a, &b, 0.0).unwrap().contact.unwrap();
        // Face overlap 0.15 x 0.2, thickness 1e-3.
        assert!((ab.volume - 0.15 * 0.2 * 1e-3).abs() < 1e-15, "{}", ab.volume);
        // Overhang tilt: delta / w = 1e-3 / 0.15.
        assert!((ab.normal - Vec3::Z).norm() < 1e-3 / 0.15 * 1.01, "{:?}", ab.normal);
        assert!((ab.area - 0.03).abs() < 1e-6);
        let mut a2 = a.clone();
        a2.set_owner(OWNER_B);
        let mut b2 = b.clone();
        b2.set_owner(OWNER_A);
        let ba = LayerContact::between(&b2, &a2, 0.0).unwrap().contact.unwrap();
        assert!((ba.volume - ab.volume).abs() < 1e-15 && (ba.normal + ab.normal).norm() < 1e-12);
        assert!((ba.centroid - ab.centroid).norm() < 1e-12);
    }

    #[test]
    fn random_intersections_agree_both_ways() {
        let mut seed = 99u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut hits = 0;
        for _ in 0..2000 {
            let mut shape = |owner| {
                let axis = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
                let r = Quat::from_axis_angle(axis.normalized(), axis.norm() * 6.0).to_mat3();
                let c = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
                Polytope::cuboid(c, r, Vec3::new(0.1 + rnd(), 0.1 + rnd(), 0.1 + rnd()) * 0.5, owner)
            };
            let (a, b) = (shape(OWNER_A), shape(OWNER_B));
            let (ab, ba) = (a.intersect(&b), b.intersect(&a));
            match (ab.mass_properties(), ba.mass_properties()) {
                (Some(x), Some(y)) => {
                    hits += 1;
                    let scale = x.volume.max(y.volume);
                    assert!((x.volume - y.volume).abs() < 1e-9 * scale.max(1e-12) + 1e-15, "{} vs {}", x.volume, y.volume);
                    assert!((x.centroid - y.centroid).norm() < 1e-7, "{:?} vs {:?}", x.centroid, y.centroid);
                    // A closed surface: owner normals cancel.
                    let (na, _, _) = ab.owner_surface(OWNER_A);
                    let (nb, _, _) = ab.owner_surface(OWNER_B);
                    assert!((na + nb).norm() < 1e-9 * (1.0 + na.norm()), "{:?} {:?}", na, nb);
                }
                (None, None) => {}
                (x, y) => {
                    // Only slivers below rounding may disagree.
                    let v = x.map_or(0.0, |m| m.volume).max(y.map_or(0.0, |m| m.volume));
                    assert!(v < 1e-12, "one-sided intersection of volume {v}");
                }
            }
        }
        assert!(hits > 300, "{hits}");
    }

    #[test]
    fn an_indentation_removes_that_depth_of_overlap() {
        // Box sunk 1e-3 into the ground, indented 4e-4: the layer acts on the deepest 6e-4.
        let p = cube(Vec3::new(0.0, 0.0, 0.1 - 1e-3), 0.1);
        let o = LayerContact::with_half_space(&p, Vec3::Z, 0.0, 4e-4).unwrap();
        assert!((o.max_depth - 1e-3).abs() < 1e-12);
        let c = o.contact.unwrap();
        assert!((c.volume - 0.04 * 6e-4).abs() < 1e-15, "{}", c.volume);
        assert!((c.depth - 3e-4).abs() < 1e-12 && (c.area - 0.04).abs() < 1e-12);
        // Fully indented: in contact, but no force (at most a round-off sliver).
        let o = LayerContact::with_half_space(&p, Vec3::Z, 0.0, 1e-3).unwrap();
        assert!(o.contact.is_none_or(|c| c.volume < 1e-18) && (o.max_depth - 1e-3).abs() < 1e-12);
    }

    /// A ball's nearest-face cells partition its overlap with the body: their volumes add
    /// up to the overlap's, to the rounding of terms of order `r^3`, in every
    /// orientation of a box pressed shallowly by a ball near an edge or corner.
    #[test]
    fn ball_cells_partition_the_overlap_in_every_orientation() {
        let half = Vec3::splat(0.05);
        let r = 0.08;
        let mut worst = 0.0f64;
        for k in 0..20000u64 {
            let u = |j: u64| crate::rng::uniform(7, &[k, j]);
            let axis = Vec3::new(u(0) - 0.5, u(1) - 0.5, u(2) - 0.5).normalized();
            let rot = Quat::from_axis_angle(axis, u(3) * PI).to_mat3();
            let body = Polytope::cuboid(Vec3::ZERO, rot, half, OWNER_A);
            // Pressed 1e-5 to 5e-3 into a face, the centre within 2 mm of its edge (or
            // of two edges: a corner).
            let local = Vec3::new(0.05 + 0.004 * (u(4) - 0.5), -0.05 - (r - (1e-5 + 5e-3 * u(5))), 0.05 + 0.004 * (u(6) - 0.5));
            let centre = rot * local;
            let Some(whole) = LayerContact::with_ball(&body, centre, r) else { continue };
            let cells: f64 = ball_cells(&body, centre, r).iter().map(|c| c.volume).sum();
            worst = worst.max((cells - whole.volume).abs() / (r * r * r));
        }
        println!("worst mismatch of the cells' volume {worst:.3e} r^3");
        assert!(worst < 1e-12, "{worst}");
    }

    /// The configuration that gave a struck chunk a 45 m/s kick (a free wall hit by a
    /// ball, the scene turned by 0.7 rad, at 2.733 ms): one of its cells is a sliver
    /// whose face toward the ball degenerates to a near-collinear triangle, which
    /// `disc_region` took to hold the ball's whole disc (a 1.04e-4 m^3 cell in a
    /// 2.1e-7 m^3 overlap). Its cells add up to its overlap.
    #[test]
    fn ball_cells_of_a_struck_chunk_add_up_to_its_overlap() {
        let b = f64::from_bits;
        let centre = Vec3::new(b(0xbf75c71c78def992), b(0x3fb2df8f86631d00), b(0xbfa7cd9bfdda0d89));
        let rot = Mat3 { m: [[b(0x3fe8fd4ca5f08d44), b(0xbfdf09e066487eda), b(0x3fd9318dbef9cc29)], [b(0x3fe1a4a77ba6dd99), b(0x3fea9a9d327f9aba), b(0xbfb1cb545c4e2917)], [b(0xbfd2c9ae7208c8db), b(0x3fd15d4b7db5c651), b(0x3fed55062dd05329)]] };
        let half = Vec3::splat(b(0x3fa999999999999a));
        let ball = Vec3::new(b(0x3fb6299034e983c2), b(0xbf983f1071f5db4c), b(0x3f8fc8c793ac5366));
        let r = b(0x3fb47ae147ae147b);
        let body = Polytope::cuboid(centre, rot, half, OWNER_A);
        let whole = LayerContact::with_ball(&body, ball, r).expect("overlap");
        let cells = ball_cells(&body, ball, r);
        let sum: f64 = cells.iter().map(|c| c.volume).sum();
        println!("whole {:.6e}, cells {sum:.6e}", whole.volume);
        assert!((sum - whole.volume).abs() < 1e-12 * r * r * r, "{sum} vs {}", whole.volume);
        assert!(cells.iter().all(|c| c.depth > 0.0), "a cell below its face");
    }

    /// A shallow cap's centroid lies a third of its height below the plane (the
    /// paraboloid limit), to round-off: the depth is not a difference of two values of
    /// order the radius.
    #[test]
    fn shallow_sphere_cap_depth_is_exact() {
        for d in [1e-4, 1e-8, 1e-12, 1e-16] {
            let c = LayerContact::sphere_cap(Vec3::new(0.3, -0.2, 0.7), 0.5, Vec3::Z, d).unwrap();
            let exact = d * (4.0 * 0.5 - d) / (4.0 * (1.5 - d));
            assert!((c.depth - exact).abs() <= 4.0 * f64::EPSILON * exact, "d {d}: {} vs {exact}", c.depth);
            assert!((c.depth / d - 1.0 / 3.0).abs() < d, "d {d}: {}", c.depth / d);
        }
    }

    #[test]
    fn sphere_cap_volume_and_centroid() {
        let (r, d) = (0.5, 0.1);
        let c = LayerContact::sphere_cap(Vec3::new(0.0, 0.0, r - d), r, Vec3::Z, d).unwrap();
        assert!((c.volume - std::f64::consts::PI * d * d * (3.0 * r - d) / 3.0).abs() < 1e-15);
        // Cap centroid height below the plane z = 0: from the classical formula.
        let zc = (r - d) - 3.0 * (2.0 * r - d).powi(2) / (4.0 * (3.0 * r - d));
        assert!((c.centroid.z - zc).abs() < 1e-12 && (c.depth + zc).abs() < 1e-12);
        assert!((c.area - std::f64::consts::PI * (2.0 * r * d - d * d)).abs() < 1e-12);
    }

    /// `cap_polygon` as first written (stable `sort_by`, then deduplication; the ring
    /// starting at the hull vertex met first among the segments).
    fn cap_polygon_reference(segments: &[(Vec3, Vec3)], normal: Vec3) -> Vec<Vec3> {
        let u = normal.any_perpendicular().normalized();
        let v = normal.cross(u);
        let mut pts: Vec<(f64, f64, Vec3)> = segments.iter().flat_map(|&(a, b)| [a, b]).map(|p| (p.dot(u), p.dot(v), p)).collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        pts.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        if pts.len() < 3 {
            return Vec::new();
        }
        let cross = |o: &(f64, f64, Vec3), a: &(f64, f64, Vec3), b: &(f64, f64, Vec3)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
        let mut hull: Vec<(f64, f64, Vec3)> = Vec::new();
        for p in pts.iter() {
            while hull.len() >= 2 && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(*p);
        }
        let lower = hull.len() + 1;
        for p in pts.iter().rev().skip(1) {
            while hull.len() >= lower && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
                hull.pop();
            }
            hull.push(*p);
        }
        hull.pop();
        let order = |p: Vec3| segments.iter().flat_map(|&(a, b)| [a, b]).position(|q| q == p).unwrap_or(usize::MAX);
        let start = (0..hull.len()).min_by_key(|&i| order(hull[i].2)).unwrap_or(0);
        hull.rotate_left(start);
        hull.iter().map(|p| p.2).collect()
    }

    /// The cap of a clip is the reference's, bit for bit: shared crossing points, points
    /// that coincide to round-off, collinear points, signed zeros and NaN.
    #[test]
    fn cap_polygon_matches_its_reference() {
        let mut state = 11u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let bits = |r: &[Vec3]| r.iter().map(|p| p.to_array().map(f64::to_bits)).collect::<Vec<_>>();
        for trial in 0..3000 {
            let normal = Vec3::new(next() - 0.5, next() - 0.5, next() - 0.5).normalized();
            let n = 3 + (next() * 7.0) as usize;
            let mut ring: Vec<Vec3> = (0..n)
                .map(|_| {
                    let c = |x: f64| if trial % 3 == 0 { (x * 4.0).round() / 4.0 } else { x };
                    Vec3::new(c(next() * 2.0 - 1.0), c(next() * 2.0 - 1.0), c(next() * 2.0 - 1.0))
                })
                .collect();
            if trial % 5 == 1 {
                ring[0] = Vec3::new(-0.0, 0.0, -0.0);
                ring[1] = Vec3::new(0.0, -0.0, 0.0);
            }
            if trial % 7 == 2 {
                // Coincident to round-off, not bit for bit.
                ring[1] = ring[0] + Vec3::new(f64::EPSILON, 0.0, 0.0);
            }
            if trial % 11 == 3 {
                ring[2].y = f64::NAN;
            }
            // Segments head to tail (each point an exit and the next entry), some
            // reversed, some repeated.
            let mut segments: Vec<(Vec3, Vec3)> = (0..n).map(|i| (ring[i], ring[(i + 1) % n])).collect();
            if trial % 4 == 0 {
                segments.reverse();
            }
            if trial % 6 == 5 {
                segments.push(segments[0]);
            }
            let (mut pts, mut hull, mut out) = (Vec::new(), Vec::new(), Vec::new());
            cap_polygon(&segments, normal, &mut pts, &mut hull, &mut out);
            assert_eq!(bits(&out), bits(&cap_polygon_reference(&segments, normal)), "trial {trial}");
        }
    }
}

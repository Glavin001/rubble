//! Strict closest-feature queries (`slang/physics/feature_query.slang`, its one Rust
//! twin): a point, a segment or a triangle against a rigid shape (box, hull, capsule,
//! vertical cylinder, sphere, and a single triangle: a trimesh's or a heightfield
//! cell's), each contact with barycentric weights on the element's vertices, its
//! normal, gap and point. Shared by the explicit vehicles' surface contact
//! (`engine/docs/jbeam-semantics-spec.md` §38, `engine_softbody::explicit_surface`) and
//! any rigid caller that wants a weighted closest feature; the rigid engine's SAT
//! manifolds (`convex.slang`, `box_collide.slang`) are separate and unchanged.
//!
//! **Features.** Per (element, shape) the query collects
//! * V: each element vertex outside the shape, its exact distance to the shape;
//! * X: the shape's features against the element (closest points), each with its
//!   point on the element (barycentric weights), its point on the shape and its gap.
//!
//! When a vertex is no farther than the nearest X candidate (within [`TIE`]) the
//! contact is the vertex's (a node's own), and the element takes none; else the X
//! candidates within [`TIE`] of the nearest are one contact at their mean; touching or
//! crossing, the deepest along the separating axis of least overlap.
//!
//! Every operation is single precision in a fixed order (sums left to right, no
//! fusing): the Slang twin (`feature_query.slang`, with `strict.slang`'s helpers) does
//! the same, so the GPU and the CPU agree bit for bit.
//!
//! A hull is given in world coordinates by its vertices, edges and planes; [`hull_world`]
//! places one from its local (scaled) records, as the World's hull arena
//! ([`crate::convex`]) stores them, by the strict pose of [`shape_pose`].

use glam::{Quat, Vec3};

/// Candidates within this of the nearest (m) are one contact (their mean); a vertex
/// within it of the nearest candidate makes the contact the node's.
pub const TIE: f32 = 1e-4;
/// A gap at most this (m) is touching or crossing: the normal is then the separating
/// axis of least overlap ([`overlap_axis`]).
pub const PIERCE: f32 = 1e-5;

/// A rigid shape in world coordinates, as the surface contact meets it.
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceShape {
    /// Centre, unit axes (the box's local x, y, z in world), half extents.
    Box {
        c: Vec3,
        axes: [Vec3; 3],
        half: Vec3,
    },
    /// A vertical (world Y) cylinder.
    Cylinder {
        c: Vec3,
        radius: f32,
        half_height: f32,
    },
    /// Segment a–b swept by `radius`.
    Capsule {
        a: Vec3,
        b: Vec3,
        radius: f32,
    },
    Sphere {
        c: Vec3,
        radius: f32,
    },
    /// A convex hull: vertices, edges (vertex pairs) and outward planes (n · x ≤ d
    /// inside), all world.
    Hull {
        verts: Vec<Vec3>,
        edges: Vec<[u32; 2]>,
        planes: Vec<(Vec3, f32)>,
    },
    /// A single triangle (a trimesh's, or one of a heightfield cell's two), two-sided
    /// and of no thickness: no point is inside it.
    Triangle {
        t: [Vec3; 3],
    },
}

impl SurfaceShape {
    /// A reference point inside the shape (its centre).
    pub fn centre(&self) -> Vec3 {
        match self {
            Self::Box { c, .. } | Self::Cylinder { c, .. } | Self::Sphere { c, .. } => *c,
            Self::Capsule { a, b, .. } => (*a + *b) * 0.5,
            Self::Hull { verts, .. } => {
                let mut s = Vec3::ZERO;
                for &v in verts {
                    s += v;
                }
                s / verts.len().max(1) as f32
            }
            Self::Triangle { t } => ((t[0] + t[1]) + t[2]) * (1.0 / 3.0),
        }
    }
}

/// One surface contact: weights of the triangle's vertices (in its node order), the
/// unit normal from the triangle toward the shape, the gap (m, ≤ the band) and the
/// contact point on the triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceHit {
    pub w: [f32; 3],
    pub n: Vec3,
    pub gap: f32,
    pub q: Vec3,
}

/// `a · b` as ((x + y) + z) (the kernel's `sdot`).
pub fn dot3(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// `a × b` in the kernel's `scross` order.
pub fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - b.y * a.z,
        a.z * b.x - b.z * a.x,
        a.x * b.y - b.x * a.y,
    )
}

fn len3(v: Vec3) -> f32 {
    dot3(v, v).sqrt()
}

fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// The point of triangle a b c nearest p and its barycentric weights (Ericson,
/// *Real-Time Collision Detection* §5.1.5).
pub fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f32; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = p - b;
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }
    let cp = p - c;
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return (a + ac * t, [1.0 - t, 0.0, t]);
    }
    let va = d3 * d6 - d5 * d4;
    let e43 = d4 - d3;
    let e56 = d5 - d6;
    if va <= 0.0 && e43 >= 0.0 && e56 >= 0.0 {
        let t = e43 / (e43 + e56);
        return (b + (c - b) * t, [0.0, 1.0 - t, t]);
    }
    let denom = 1.0 / ((va + vb) + vc);
    let v = vb * denom;
    let u = vc * denom;
    (a + ab * v + ac * u, [(1.0 - v) - u, v, u])
}

/// Closest points of segments p1–q1 and p2–q2 (Ericson §5.1.9): their parameters.
pub fn closest_seg_seg(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (f32, f32) {
    const EPS: f32 = 1e-12;
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let f = dot3(d2, r);
    if a <= EPS && e <= EPS {
        return (0.0, 0.0);
    }
    if a <= EPS {
        return (0.0, clamp01(f / e));
    }
    let c = dot3(d1, r);
    if e <= EPS {
        return (clamp01(-c / a), 0.0);
    }
    let b = dot3(d1, d2);
    let denom = a * e - b * b;
    let mut s = if denom > EPS {
        clamp01((b * f - c * e) / denom)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = clamp01(-c / a);
    } else if t > 1.0 {
        t = 1.0;
        s = clamp01((b - c) / a);
    }
    (s, t)
}

/// Weights of triangle edge k (x_k → x_{k+1}) at parameter t.
fn edge_weights(k: usize, t: f32) -> [f32; 3] {
    let mut w = [0.0; 3];
    w[k] = 1.0 - t;
    w[(k + 1) % 3] = t;
    w
}

/// A candidate pair of closest points: weights and point on the triangle, point on
/// the shape (its core for the round shapes), distance between them.
#[derive(Clone, Copy, Debug)]
struct Pair {
    w: [f32; 3],
    q: Vec3,
    cs: Vec3,
    d: f32,
}

/// The candidate pairs of segment p0–p1 against triangle x, each with its gap from
/// `map` (the pair as the shape takes it, and its gap): each end against the triangle, the segment against each edge, and the
/// segment piercing the triangle at the crossing. All of them, not only the nearest:
/// a segment parallel to the face is nearest along a run, whose two ends these are.
fn seg_triangle(
    p0: Vec3,
    p1: Vec3,
    x: &[Vec3; 3],
    map: &impl Fn(Pair) -> (Pair, f32),
    visit: &mut impl FnMut(Pair, f32),
) {
    let mut emit = |p: Pair| {
        let (p, g) = map(p);
        visit(p, g);
    };
    for e in [p0, p1] {
        let (q, w) = closest_on_triangle(e, x[0], x[1], x[2]);
        emit(Pair {
            w,
            q,
            cs: e,
            d: len3(q - e),
        });
    }
    for k in 0..3 {
        let (ea, eb) = (x[k], x[(k + 1) % 3]);
        let (s, t) = closest_seg_seg(p0, p1, ea, eb);
        let a = p0 + (p1 - p0) * s;
        let y = ea + (eb - ea) * t;
        emit(Pair {
            w: edge_weights(k, t),
            q: y,
            cs: a,
            d: len3(a - y),
        });
    }
    // The segment piercing the triangle.
    let n = cross3(x[1] - x[0], x[2] - x[0]);
    let d0 = dot3(p0 - x[0], n);
    let d1 = dot3(p1 - x[0], n);
    if (d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0) {
        let c = p0 + (p1 - p0) * (d0 / (d0 - d1));
        let (q, w) = closest_on_triangle(c, x[0], x[1], x[2]);
        if dot3(q - c, q - c) < 1e-10 {
            emit(Pair {
                w,
                q,
                cs: c,
                d: 0.0,
            });
        }
    }
}

/// The part of segment p0–p1 inside the planes (n · x ≤ d; strictly inside a run of
/// positive length): its parameter range.
fn clip_planes(
    p0: Vec3,
    p1: Vec3,
    planes: impl Iterator<Item = (Vec3, f32)>,
) -> Option<(f32, f32)> {
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (n, d) in planes {
        let s0 = dot3(n, p0) - d;
        let s1 = dot3(n, p1) - d;
        if s0 >= 0.0 && s1 >= 0.0 {
            return None;
        }
        if s0 < 0.0 && s1 < 0.0 {
            continue;
        }
        let t = s0 / (s0 - s1);
        if s0 >= 0.0 {
            t0 = t0.max(t);
        } else {
            t1 = t1.min(t);
        }
    }
    (t0 < t1).then_some((t0, t1))
}

/// A box's six face planes (outward), in the order −x, +x, −y, +y, −z, +z.
fn box_planes(c: Vec3, axes: &[Vec3; 3], half: Vec3) -> [(Vec3, f32); 6] {
    let h = [half.x, half.y, half.z];
    let mut out = [(Vec3::ZERO, 0.0); 6];
    for i in 0..3 {
        let ca = dot3(axes[i], c);
        out[2 * i] = (-axes[i], h[i] - ca);
        out[2 * i + 1] = (axes[i], ca + h[i]);
    }
    out
}

/// The distance from point x to the shape when x is outside it (exact; a hull's is
/// its largest plane distance, a lower bound), `None` inside.
fn point_distance(s: &SurfaceShape, x: Vec3) -> Option<f32> {
    match s {
        SurfaceShape::Box { c, axes, half } => {
            let d = x - *c;
            let l = Vec3::new(dot3(axes[0], d), dot3(axes[1], d), dot3(axes[2], d));
            let inside = l.x > -half.x
                && l.x < half.x
                && l.y > -half.y
                && l.y < half.y
                && l.z > -half.z
                && l.z < half.z;
            if inside {
                return None;
            }
            let k = Vec3::new(
                l.x.max(-half.x).min(half.x),
                l.y.max(-half.y).min(half.y),
                l.z.max(-half.z).min(half.z),
            );
            Some(len3(l - k))
        }
        SurfaceShape::Sphere { c, radius } => {
            let d = len3(x - *c) - *radius;
            (d > 0.0).then_some(d)
        }
        SurfaceShape::Capsule { a, b, radius } => {
            let ab = *b - *a;
            let l2 = dot3(ab, ab);
            let t = if l2 > 0.0 {
                clamp01(dot3(x - *a, ab) / l2)
            } else {
                0.0
            };
            let d = len3(x - (*a + ab * t)) - *radius;
            (d > 0.0).then_some(d)
        }
        SurfaceShape::Cylinder {
            c,
            radius,
            half_height,
        } => {
            let d = x - *c;
            let dh = (d.x * d.x + d.z * d.z).sqrt() - *radius;
            let dy = d.y.abs() - *half_height;
            if dh < 0.0 && dy < 0.0 {
                return None;
            }
            let (a, b) = (dh.max(0.0), dy.max(0.0));
            Some((a * a + b * b).sqrt())
        }
        SurfaceShape::Hull { planes, .. } => {
            let mut m = f32::NEG_INFINITY;
            for &(n, d) in planes {
                m = m.max(dot3(n, x) - d);
            }
            (m >= 0.0).then_some(m)
        }
        SurfaceShape::Triangle { t } => {
            let (q, _) = closest_on_triangle(x, t[0], t[1], t[2]);
            Some(len3(x - q))
        }
    }
}

/// The shape's point nearest x when x is outside it (a hull's: x moved back along its
/// farthest plane, the lower bound's point), `None` inside.
fn point_nearest(s: &SurfaceShape, x: Vec3) -> Option<Vec3> {
    match s {
        SurfaceShape::Box { c, axes, half } => {
            let d = x - *c;
            let l = Vec3::new(dot3(axes[0], d), dot3(axes[1], d), dot3(axes[2], d));
            let inside = l.x > -half.x
                && l.x < half.x
                && l.y > -half.y
                && l.y < half.y
                && l.z > -half.z
                && l.z < half.z;
            if inside {
                return None;
            }
            let k = Vec3::new(
                l.x.max(-half.x).min(half.x),
                l.y.max(-half.y).min(half.y),
                l.z.max(-half.z).min(half.z),
            );
            Some(((*c + axes[0] * k.x) + axes[1] * k.y) + axes[2] * k.z)
        }
        SurfaceShape::Sphere { c, radius } => {
            let d = x - *c;
            let l = len3(d);
            (l > *radius).then(|| *c + d * (*radius / l))
        }
        SurfaceShape::Capsule { a, b, radius } => {
            let ab = *b - *a;
            let l2 = dot3(ab, ab);
            let t = if l2 > 0.0 {
                clamp01(dot3(x - *a, ab) / l2)
            } else {
                0.0
            };
            let m = *a + ab * t;
            let d = x - m;
            let l = len3(d);
            (l > *radius).then(|| m + d * (*radius / l))
        }
        SurfaceShape::Cylinder {
            c,
            radius,
            half_height,
        } => {
            point_distance(s, x)?;
            Some(cylinder_point(*c, *radius, *half_height, x))
        }
        SurfaceShape::Hull { planes, .. } => {
            let mut m = f32::NEG_INFINITY;
            let mut n = Vec3::ZERO;
            for &(pn, d) in planes {
                let e = dot3(pn, x) - d;
                if e > m {
                    m = e;
                    n = pn;
                }
            }
            (m >= 0.0).then(|| x - n * m)
        }
        SurfaceShape::Triangle { t } => Some(closest_on_triangle(x, t[0], t[1], t[2]).0),
    }
}

/// The band contact of a node at x (point-vs-shape): outside the shape and within
/// `r` of it, the normal toward its nearest point; `None` inside (node contact's),
/// past the band, or touching (gap ≤ [`PIERCE`]: no direction; node contact takes it
/// once inside). Weights (1, 0, 0).
pub fn point_contact(x: Vec3, s: &SurfaceShape, r: f32) -> Option<SurfaceHit> {
    let p = point_nearest(s, x)?;
    let dv = p - x;
    let g = len3(dv);
    if !(g <= r) || !(g > PIERCE) {
        return None;
    }
    Some(SurfaceHit {
        w: [1.0, 0.0, 0.0],
        n: dv / g,
        gap: g,
        q: x,
    })
}

/// The point of the solid vertical cylinder (centre c, radius r, half height hh)
/// nearest x (x itself inside).
fn cylinder_point(c: Vec3, r: f32, hh: f32, x: Vec3) -> Vec3 {
    let d = x - c;
    let l = (d.x * d.x + d.z * d.z).sqrt();
    let (px, pz) = if l > r {
        (d.x * (r / l), d.z * (r / l))
    } else {
        (d.x, d.z)
    };
    let py = d.y.max(-hh).min(hh);
    c + Vec3::new(px, py, pz)
}

/// The shape's projection onto unit axis u: (lo, hi).
fn project(s: &SurfaceShape, u: Vec3) -> (f32, f32) {
    let (cu, e) = match s {
        SurfaceShape::Box { c, axes, half } => (
            dot3(u, *c),
            dot3(u, axes[0]).abs() * half.x
                + dot3(u, axes[1]).abs() * half.y
                + dot3(u, axes[2]).abs() * half.z,
        ),
        SurfaceShape::Sphere { c, radius } => (dot3(u, *c), *radius),
        SurfaceShape::Capsule { a, b, radius } => {
            let m = (*a + *b) * 0.5;
            let h = (*b - *a) * 0.5;
            (dot3(u, m), dot3(u, h).abs() + *radius)
        }
        SurfaceShape::Cylinder {
            c,
            radius,
            half_height,
        } => (
            dot3(u, *c),
            u.y.abs() * *half_height + *radius * (u.x * u.x + u.z * u.z).sqrt(),
        ),
        SurfaceShape::Hull { verts, .. } => {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in verts {
                let p = dot3(u, v);
                lo = lo.min(p);
                hi = hi.max(p);
            }
            return (lo, hi);
        }
        SurfaceShape::Triangle { t } => {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in t {
                let p = dot3(u, v);
                lo = lo.min(p);
                hi = hi.max(p);
            }
            return (lo, hi);
        }
    };
    (cu - e, cu + e)
}

/// The separating axis of least overlap between triangle x (unit normal `n`) and the
/// shape, oriented from the triangle toward the shape: the triangle's normal, the
/// shape's face axes, then the cross products of the triangle's edges with the
/// shape's edge directions (near-parallel pairs left out); a later axis replaces the
/// best only when it overlaps less by more than 10 µm.
pub fn overlap_axis(x: &[Vec3; 3], n: Vec3, s: &SurfaceShape) -> Vec3 {
    let mut best = (f32::INFINITY, n);
    let mut first = true;
    let mut test = |u: Vec3| {
        let t0 = dot3(u, x[0]);
        let t1 = dot3(u, x[1]);
        let t2 = dot3(u, x[2]);
        let tlo = t0.min(t1).min(t2);
        let thi = t0.max(t1).max(t2);
        let (slo, shi) = project(s, u);
        // Normal +u: the triangle moves toward −u by thi − slo; −u: by shi − tlo.
        let plus = thi - slo;
        let minus = shi - tlo;
        let (ov, axis) = if plus <= minus {
            (plus, u)
        } else {
            (minus, -u)
        };
        if first || ov < best.0 - 1e-5 {
            best = (ov, axis);
            first = false;
        }
    };
    test(n);
    let mut dirs: Vec<Vec3> = Vec::new();
    match s {
        SurfaceShape::Box { axes, .. } => {
            for &a in axes {
                test(a);
            }
            dirs.extend_from_slice(axes);
        }
        SurfaceShape::Cylinder { .. } => {
            test(Vec3::Y);
            dirs.push(Vec3::Y);
        }
        SurfaceShape::Capsule { a, b, .. } => dirs.push(*b - *a),
        SurfaceShape::Sphere { .. } => {}
        SurfaceShape::Hull {
            verts,
            edges,
            planes,
        } => {
            for &(p, _) in planes {
                test(p);
            }
            for e in edges {
                dirs.push(verts[e[1] as usize] - verts[e[0] as usize]);
            }
        }
        SurfaceShape::Triangle { t } => {
            if let Some(nt) = tri_normal(t) {
                test(nt);
            }
            for k in 0..3 {
                dirs.push(t[(k + 1) % 3] - t[k]);
            }
        }
    }
    for k in 0..3 {
        let e = x[(k + 1) % 3] - x[k];
        for &d in &dirs {
            let a = cross3(e, d);
            let l2 = dot3(a, a);
            if !(l2 > 1e-12 * (dot3(e, e) * dot3(d, d))) {
                continue;
            }
            test(a / l2.sqrt());
        }
    }
    best.1
}

/// The X candidates of the shape against triangle x: closest-point pairs, each with
/// its gap (distance less the shape's radius), in a fixed order.
fn x_candidates(s: &SurfaceShape, x: &[Vec3; 3], visit: &mut impl FnMut(Pair, f32)) {
    match s {
        SurfaceShape::Sphere { c, radius } => {
            let (q, w) = closest_on_triangle(*c, x[0], x[1], x[2]);
            let d = len3(*c - q);
            visit(Pair { w, q, cs: *c, d }, d - *radius);
        }
        SurfaceShape::Capsule { a, b, radius } => {
            seg_triangle(*a, *b, x, &|p: Pair| (p, p.d - *radius), visit);
        }
        SurfaceShape::Cylinder {
            c,
            radius,
            half_height,
        } => {
            // The axis segment's pairs, each triangle point taken to the solid
            // cylinder: its nearest point and distance (0 inside).
            let h = Vec3::new(0.0, *half_height, 0.0);
            let (cc, hh, rr) = (*c, *half_height, *radius);
            seg_triangle(
                cc - h,
                cc + h,
                x,
                &|p: Pair| {
                    let cs = cylinder_point(cc, rr, hh, p.q);
                    let d = len3(p.q - cs);
                    (Pair { cs, d, ..p }, d)
                },
                visit,
            );
        }
        SurfaceShape::Box { c, axes, half } => {
            // Its twelve edges against the triangle, by axis (x, y, z), each at the
            // corners (−, −), (+, −), (−, +), (+, +) of the other two.
            let h = [half.x, half.y, half.z];
            for i in 0..3 {
                let (u, v) = ((i + 1) % 3, (i + 2) % 3);
                for k in 0..4 {
                    let su = if k & 1 != 0 { h[u] } else { -h[u] };
                    let sv = if k & 2 != 0 { h[v] } else { -h[v] };
                    let m = *c + axes[u] * su + axes[v] * sv;
                    let e = axes[i] * h[i];
                    seg_triangle(m - e, m + e, x, &|p: Pair| (p, p.d), visit);
                }
            }
            let planes = box_planes(*c, axes, *half);
            tri_edges_inside(x, &planes, visit);
        }
        SurfaceShape::Hull {
            verts,
            edges,
            planes,
        } => {
            for e in edges {
                seg_triangle(
                    verts[e[0] as usize],
                    verts[e[1] as usize],
                    x,
                    &|p: Pair| (p, p.d),
                    visit,
                );
            }
            tri_edges_inside(x, planes, visit);
        }
        SurfaceShape::Triangle { t } => {
            // Its three edges against the element, then each element edge piercing it.
            for k in 0..3 {
                seg_triangle(t[k], t[(k + 1) % 3], x, &|p: Pair| (p, p.d), visit);
            }
            for k in 0..3 {
                let (p0, p1) = (x[k], x[(k + 1) % 3]);
                if let Some((q, s)) = pierce(p0, p1, t) {
                    visit(
                        Pair {
                            w: edge_weights(k, s),
                            q,
                            cs: q,
                            d: 0.0,
                        },
                        0.0,
                    );
                }
            }
        }
    }
}

/// The unit normal of triangle t (none when degenerate).
fn tri_normal(t: &[Vec3; 3]) -> Option<Vec3> {
    let n = cross3(t[1] - t[0], t[2] - t[0]);
    let l = len3(n);
    (l > 1e-8).then(|| n / l)
}

/// Segment p0–p1 crossing triangle t strictly (its ends on either side of t's plane)
/// at a point inside t: the point and its parameter on the segment.
fn pierce(p0: Vec3, p1: Vec3, t: &[Vec3; 3]) -> Option<(Vec3, f32)> {
    let n = cross3(t[1] - t[0], t[2] - t[0]);
    let d0 = dot3(p0 - t[0], n);
    let d1 = dot3(p1 - t[0], n);
    if !((d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0)) {
        return None;
    }
    let s = d0 / (d0 - d1);
    let c = p0 + (p1 - p0) * s;
    let (q, _) = closest_on_triangle(c, t[0], t[1], t[2]);
    (dot3(q - c, q - c) < 1e-10).then_some((c, s))
}

/// Each triangle edge's run inside the planes (an edge through the shape): a pair of
/// distance 0 at the run's middle.
fn tri_edges_inside(x: &[Vec3; 3], planes: &[(Vec3, f32)], visit: &mut impl FnMut(Pair, f32)) {
    for k in 0..3 {
        let (p0, p1) = (x[k], x[(k + 1) % 3]);
        if let Some((t0, t1)) = clip_planes(p0, p1, planes.iter().copied()) {
            for t in [t0, t1] {
                let q = p0 + (p1 - p0) * t;
                visit(
                    Pair {
                        w: edge_weights(k, t),
                        q,
                        cs: q,
                        d: 0.0,
                    },
                    0.0,
                );
            }
        }
    }
}

/// Weights on two or more nodes (not a node's own point).
fn shared(w: &[f32; 3]) -> bool {
    (w[0] > 0.0) as u32 + (w[1] > 0.0) as u32 + (w[2] > 0.0) as u32 >= 2
}

/// Touching or crossing (the normal n from the separating axis): the contact at the
/// element's deepest point along n among the candidates within [`TIE`] of the nearest
/// gap `gx` whose weights are shared (a node's own point is its contact), the mean
/// of those within [`TIE`] of the deepest; `None` when there is none. A crossing's
/// candidates are the ends of its runs inside the shape (each the element's deepest
/// on its run), so the contact holds the point that went deepest, not a run's middle
/// the element turns about. Two more passes over the candidates.
fn deepest(
    n: Vec3,
    gx: f32,
    cands: &dyn Fn(&mut dyn FnMut(Pair, f32)),
) -> Option<([f32; 3], Vec3)> {
    let mut top = f32::NEG_INFINITY;
    cands(&mut |p, g| {
        if g <= gx + TIE && shared(&p.w) {
            top = top.max(dot3(n, p.q));
        }
    });
    if top == f32::NEG_INFINITY {
        return None;
    }
    let (mut w, mut q, mut k) = ([0.0f32; 3], Vec3::ZERO, 0u32);
    cands(&mut |p, g| {
        if g <= gx + TIE && shared(&p.w) && dot3(n, p.q) >= top - TIE {
            w = [w[0] + p.w[0], w[1] + p.w[1], w[2] + p.w[2]];
            q += p.q;
            k += 1;
        }
    });
    let kf = k as f32;
    Some(([w[0] / kf, w[1] / kf, w[2] / kf], q / kf))
}

/// The surface contact of triangle x (world positions, node order) with shape s in a
/// band of `r` (m): `None` when they are apart by more than r or when the nearest
/// feature is a vertex (node contact's).
pub fn triangle_contact(x: &[Vec3; 3], s: &SurfaceShape, r: f32) -> Option<SurfaceHit> {
    let nc = cross3(x[1] - x[0], x[2] - x[0]);
    let nl = len3(nc);
    if !(nl > 1e-8) {
        return None;
    }
    let n = nc / nl;
    // Two passes over the candidates (as the kernel, which keeps no list): the
    // nearest gap, then the mean of those within TIE of it.
    let mut gx = f32::INFINITY;
    x_candidates(s, x, &mut |_, g| gx = gx.min(g));
    if !(gx <= r) {
        return None;
    }
    // A vertex outside the shape and no farther than the nearest candidate: the
    // node's contact.
    for &v in x {
        if let Some(d) = point_distance(s, v) {
            if d <= gx + TIE {
                return None;
            }
        }
    }
    let mut w = [0.0f32; 3];
    let mut q = Vec3::ZERO;
    let mut cs = Vec3::ZERO;
    let mut k = 0u32;
    x_candidates(s, x, &mut |p, g| {
        if g <= gx + TIE {
            w = [w[0] + p.w[0], w[1] + p.w[1], w[2] + p.w[2]];
            q += p.q;
            cs += p.cs;
            k += 1;
        }
    });
    let kf = k as f32;
    let w = [w[0] / kf, w[1] / kf, w[2] / kf];
    let (q, cs) = (q / kf, cs / kf);
    if (w[0] > 0.0) as u32 + (w[1] > 0.0) as u32 + (w[2] > 0.0) as u32 <= 1 {
        return None;
    }
    let dv = cs - q;
    let dl = len3(dv);
    if gx > PIERCE && dl > PIERCE {
        return Some(SurfaceHit {
            w,
            n: dv / dl,
            gap: gx,
            q,
        });
    }
    let normal = overlap_axis(x, n, s);
    let (w, q) = deepest(normal, gx, &|mut v| x_candidates(s, x, &mut v)).unwrap_or((w, q));
    Some(SurfaceHit {
        w,
        n: normal,
        gap: gx,
        q,
    })
}

/// The parameter on segment p0 + d·t (d·d = dd > 0) nearest x.
fn seg_param(p0: Vec3, d: Vec3, dd: f32, x: Vec3) -> f32 {
    clamp01(dot3(x - p0, d) / dd)
}

/// The candidate pairs of shape edge e0–e1 against beam p0–p1 (direction d, d·d =
/// dd): the two segments' closest points unless they are parallel, then each edge end
/// against the beam (a run's ends when they are parallel), each through `map`.
#[allow(clippy::too_many_arguments)]
fn edge_beam(
    e0: Vec3,
    e1: Vec3,
    p0: Vec3,
    p1: Vec3,
    d: Vec3,
    dd: f32,
    map: &impl Fn(Pair) -> (Pair, f32),
    visit: &mut impl FnMut(Pair, f32),
) {
    let mut emit = |p: Pair| {
        let (p, g) = map(p);
        visit(p, g);
    };
    let de = e1 - e0;
    let c = cross3(d, de);
    if dot3(c, c) > 1e-12 * (dd * dot3(de, de)) {
        let (s, t) = closest_seg_seg(p0, p1, e0, e1);
        let a = p0 + d * s;
        let y = e0 + de * t;
        emit(Pair {
            w: [1.0 - s, s, 0.0],
            q: a,
            cs: y,
            d: len3(a - y),
        });
    }
    for e in [e0, e1] {
        let t = seg_param(p0, d, dd, e);
        let a = p0 + d * t;
        emit(Pair {
            w: [1.0 - t, t, 0.0],
            q: a,
            cs: e,
            d: len3(a - e),
        });
    }
}

/// The beam's run inside the planes (it crosses the shape): a pair of distance 0 at
/// each of the run's ends.
fn beam_inside(p0: Vec3, p1: Vec3, planes: &[(Vec3, f32)], visit: &mut impl FnMut(Pair, f32)) {
    if let Some((t0, t1)) = clip_planes(p0, p1, planes.iter().copied()) {
        for t in [t0, t1] {
            let q = p0 + (p1 - p0) * t;
            visit(
                Pair {
                    w: [1.0 - t, t, 0.0],
                    q,
                    cs: q,
                    d: 0.0,
                },
                0.0,
            );
        }
    }
}

/// The X candidates of the shape against beam x₀–x₁, in a fixed order (the beam's
/// end points are its nodes': V, not here).
fn seg_candidates(s: &SurfaceShape, x: &[Vec3; 2], visit: &mut impl FnMut(Pair, f32)) {
    let (p0, p1) = (x[0], x[1]);
    let d = p1 - p0;
    let dd = dot3(d, d);
    let id = |p: Pair| (p, p.d);
    match s {
        SurfaceShape::Sphere { c, radius } => {
            let t = seg_param(p0, d, dd, *c);
            let q = p0 + d * t;
            let dist = len3(*c - q);
            visit(
                Pair {
                    w: [1.0 - t, t, 0.0],
                    q,
                    cs: *c,
                    d: dist,
                },
                dist - *radius,
            );
        }
        SurfaceShape::Capsule { a, b, radius } => {
            let r = *radius;
            edge_beam(*a, *b, p0, p1, d, dd, &|p: Pair| (p, p.d - r), visit);
        }
        SurfaceShape::Cylinder {
            c,
            radius,
            half_height,
        } => {
            // The axis's pairs, each beam point taken to the solid cylinder.
            let h = Vec3::new(0.0, *half_height, 0.0);
            let (cc, hh, rr) = (*c, *half_height, *radius);
            edge_beam(
                cc - h,
                cc + h,
                p0,
                p1,
                d,
                dd,
                &|p: Pair| {
                    let cs = cylinder_point(cc, rr, hh, p.q);
                    let dist = len3(p.q - cs);
                    (Pair { cs, d: dist, ..p }, dist)
                },
                visit,
            );
        }
        SurfaceShape::Box { c, axes, half } => {
            // The twelve edges in x_candidates' order, then the beam's run inside.
            let h = [half.x, half.y, half.z];
            for i in 0..3 {
                let (u, v) = ((i + 1) % 3, (i + 2) % 3);
                for k in 0..4 {
                    let su = if k & 1 != 0 { h[u] } else { -h[u] };
                    let sv = if k & 2 != 0 { h[v] } else { -h[v] };
                    let m = *c + axes[u] * su + axes[v] * sv;
                    let e = axes[i] * h[i];
                    edge_beam(m - e, m + e, p0, p1, d, dd, &id, visit);
                }
            }
            beam_inside(p0, p1, &box_planes(*c, axes, *half), visit);
        }
        SurfaceShape::Hull {
            verts,
            edges,
            planes,
        } => {
            for e in edges {
                edge_beam(
                    verts[e[0] as usize],
                    verts[e[1] as usize],
                    p0,
                    p1,
                    d,
                    dd,
                    &id,
                    visit,
                );
            }
            beam_inside(p0, p1, planes, visit);
        }
        SurfaceShape::Triangle { t } => {
            for k in 0..3 {
                edge_beam(t[k], t[(k + 1) % 3], p0, p1, d, dd, &id, visit);
            }
            if let Some((q, s)) = pierce(p0, p1, t) {
                visit(
                    Pair {
                        w: [1.0 - s, s, 0.0],
                        q,
                        cs: q,
                        d: 0.0,
                    },
                    0.0,
                );
            }
        }
    }
}

/// [`overlap_axis`] for beam x₀–x₁: the shape's face axes, then the beam's direction
/// crossed with the shape's edge directions; with none (a sphere), a fixed
/// perpendicular of the beam.
pub fn seg_overlap_axis(x: &[Vec3; 2], s: &SurfaceShape) -> Vec3 {
    let d = x[1] - x[0];
    // (least overlap, its axis, none yet)
    let mut st = (f32::INFINITY, Vec3::ZERO, true);
    let test = |st: &mut (f32, Vec3, bool), u: Vec3| {
        let t0 = dot3(u, x[0]);
        let t1 = dot3(u, x[1]);
        let (tlo, thi) = (t0.min(t1), t0.max(t1));
        let (slo, shi) = project(s, u);
        let plus = thi - slo;
        let minus = shi - tlo;
        let (ov, axis) = if plus <= minus {
            (plus, u)
        } else {
            (minus, -u)
        };
        if st.2 || ov < st.0 - 1e-5 {
            *st = (ov, axis, false);
        }
    };
    let mut dirs: Vec<Vec3> = Vec::new();
    match s {
        SurfaceShape::Box { axes, .. } => {
            for &a in axes {
                test(&mut st, a);
            }
            dirs.extend_from_slice(axes);
        }
        SurfaceShape::Cylinder { .. } => {
            test(&mut st, Vec3::Y);
            dirs.push(Vec3::Y);
        }
        SurfaceShape::Capsule { a, b, .. } => dirs.push(*b - *a),
        SurfaceShape::Sphere { .. } => {}
        SurfaceShape::Hull {
            verts,
            edges,
            planes,
        } => {
            for &(p, _) in planes {
                test(&mut st, p);
            }
            for e in edges {
                dirs.push(verts[e[1] as usize] - verts[e[0] as usize]);
            }
        }
        SurfaceShape::Triangle { t } => {
            if let Some(nt) = tri_normal(t) {
                test(&mut st, nt);
            }
            for k in 0..3 {
                dirs.push(t[(k + 1) % 3] - t[k]);
            }
        }
    }
    for &e in &dirs {
        let a = cross3(d, e);
        let l2 = dot3(a, a);
        if !(l2 > 1e-12 * (dot3(d, d) * dot3(e, e))) {
            continue;
        }
        test(&mut st, a / l2.sqrt());
    }
    if st.2 {
        // The beam crossed with x, or with y when it is nearest along x.
        let ax = if d.x.abs() <= d.y.abs().max(d.z.abs()) {
            Vec3::X
        } else {
            Vec3::Y
        };
        let a = cross3(d, ax);
        test(&mut st, a / len3(a));
    }
    st.1
}

/// The surface contact of beam x₀–x₁ (world positions of its nodes a, b) with shape
/// s in a band of `r` (m), as [`triangle_contact`]: `None` when apart by more than r
/// or when the nearest feature is an end (its node's contact). Weights (w₀, w₁, 0).
/// A beam through a shape may have a second contact ([`deepest_ends`]).
pub fn segment_contact(
    x: &[Vec3; 2],
    s: &SurfaceShape,
    r: f32,
) -> Option<(SurfaceHit, Option<SurfaceHit>)> {
    let d = x[1] - x[0];
    if !(dot3(d, d) > 1e-12) {
        return None;
    }
    let mut gx = f32::INFINITY;
    seg_candidates(s, x, &mut |_, g| gx = gx.min(g));
    if !(gx <= r) {
        return None;
    }
    for &v in x {
        if let Some(dv) = point_distance(s, v) {
            if dv <= gx + TIE {
                return None;
            }
        }
    }
    let mut w = [0.0f32; 2];
    let mut q = Vec3::ZERO;
    let mut cs = Vec3::ZERO;
    let mut k = 0u32;
    seg_candidates(s, x, &mut |p, g| {
        if g <= gx + TIE {
            w = [w[0] + p.w[0], w[1] + p.w[1]];
            q += p.q;
            cs += p.cs;
            k += 1;
        }
    });
    let kf = k as f32;
    let w = [w[0] / kf, w[1] / kf];
    let (q, cs) = (q / kf, cs / kf);
    if !(w[0] > 0.0 && w[1] > 0.0) {
        return None;
    }
    let dv = cs - q;
    let dl = len3(dv);
    let w = [w[0], w[1], 0.0];
    if gx > PIERCE && dl > PIERCE {
        return Some((
            SurfaceHit {
                w,
                n: dv / dl,
                gap: gx,
                q,
            },
            None,
        ));
    }
    let normal = seg_overlap_axis(x, s);
    let hit = |w: [f32; 3], q: Vec3| SurfaceHit {
        w,
        n: normal,
        gap: gx,
        q,
    };
    // Touching or crossing: the deepest candidates along n; when they spread along
    // the beam (a run through the shape, its ends equally deep along an axis across
    // the beam) one contact at each extreme, so the beam is held at both and cannot
    // turn deeper about one.
    let cands = |mut v: &mut dyn FnMut(Pair, f32)| seg_candidates(s, x, &mut v);
    match deepest_ends(normal, gx, &cands) {
        Some((lo, hi)) if hi.0[1] > lo.0[1] => Some((hit(lo.0, lo.1), Some(hit(hi.0, hi.1)))),
        Some((lo, _)) => Some((hit(lo.0, lo.1), None)),
        None => Some((hit(w, q), None)),
    }
}

/// [`deepest`] for a beam: of the deepest candidates along n (within [`TIE`]), those
/// at the least and the most parameter along the beam (the first of equals).
#[allow(clippy::type_complexity)]
fn deepest_ends(
    n: Vec3,
    gx: f32,
    cands: &dyn Fn(&mut dyn FnMut(Pair, f32)),
) -> Option<(([f32; 3], Vec3), ([f32; 3], Vec3))> {
    let mut top = f32::NEG_INFINITY;
    cands(&mut |p, g| {
        if g <= gx + TIE && shared(&p.w) {
            top = top.max(dot3(n, p.q));
        }
    });
    if top == f32::NEG_INFINITY {
        return None;
    }
    let mut lo: Option<([f32; 3], Vec3)> = None;
    let mut hi: Option<([f32; 3], Vec3)> = None;
    cands(&mut |p, g| {
        if g <= gx + TIE && shared(&p.w) && dot3(n, p.q) >= top - TIE {
            if lo.is_none_or(|l| p.w[1] < l.0[1]) {
                lo = Some((p.w, p.q));
            }
            if hi.is_none_or(|h| p.w[1] > h.0[1]) {
                hi = Some((p.w, p.q));
            }
        }
    });
    Some((lo?, hi?))
}

/// A beam's box padded by `r` overlapping `lo..hi`.
pub fn seg_near(x: &[Vec3; 2], r: f32, lo: Vec3, hi: Vec3) -> bool {
    let tlo = x[0].min(x[1]) - Vec3::splat(r);
    let thi = x[0].max(x[1]) + Vec3::splat(r);
    tlo.cmple(hi).all() && thi.cmpge(lo).all()
}

/// A triangle's box padded by `r` overlapping `lo..hi`.
pub fn near(x: &[Vec3; 3], r: f32, lo: Vec3, hi: Vec3) -> bool {
    let tlo = x[0].min(x[1]).min(x[2]) - Vec3::splat(r);
    let thi = x[0].max(x[1]).max(x[2]) + Vec3::splat(r);
    tlo.cmple(hi).all() && thi.cmpge(lo).all()
}

/// The separation cull's allowance for rounding beyond the band (m), besides
/// [`SEPARATION_REL`] of the largest coordinate.
pub const SEPARATION_ABS: f32 = 1e-3;
/// The separation cull's allowance relative to the largest coordinate it reads.
pub const SEPARATION_REL: f32 = 1e-4;

/// The surface pass's separation cull (lossless; the GPU's `fqSeparated`): whether the
/// element (its points `x`: a triangle's three, a segment's two) lies wholly outside one
/// of the shape's outward face planes by more than `r` plus the rounding allowance
/// ([`SEPARATION_ABS`] and [`SEPARATION_REL`] of the largest coordinate read). Every
/// point of the element is then farther than `r` from every point of the shape (the
/// plane separates them), so the narrow phase, whose computed gaps are within 1e-5 of
/// the largest coordinate and 1e-4 m of the true ones, finds no contact:
/// [`triangle_contact`] and [`segment_contact`] return none. A box's six planes, a
/// hull's, a cylinder's two ends; for a sphere, a capsule and a cylinder's side the gap
/// from the element's box to their core ([`apart`]); a triangle shape is never
/// separated here. The shape's bounding box ([`near`]) is a far weaker test for a turned
/// shape: a slab on a slope holds the whole car in its box.
pub fn separated(x: &[Vec3], s: &SurfaceShape, r: f32) -> bool {
    let big = x.iter().fold(0.0f32, |m, p| {
        m.max(p.x.abs()).max(p.y.abs()).max(p.z.abs())
    });
    let outside = |n: Vec3, d: f32| {
        let thr = r + SEPARATION_ABS + SEPARATION_REL * big.max(d.abs());
        x.iter().all(|&p| dot3(n, p) - d > thr)
    };
    match s {
        SurfaceShape::Box { c, axes, half } => (0..3).any(|i| {
            let ca = dot3(axes[i], *c);
            outside(-axes[i], half[i] - ca) || outside(axes[i], ca + half[i])
        }),
        SurfaceShape::Hull { planes, .. } => planes.iter().any(|&(n, d)| outside(n, d)),
        SurfaceShape::Cylinder {
            c,
            radius,
            half_height,
        } => {
            outside(Vec3::NEG_Y, *half_height - c.y)
                || outside(Vec3::Y, c.y + *half_height)
                || apart(x, *c, *c, Vec3::new(1.0, 0.0, 1.0), *radius, r, big)
        }
        SurfaceShape::Sphere { c, radius } => apart(x, *c, *c, Vec3::ONE, *radius, r, big),
        SurfaceShape::Capsule { a, b, radius } => {
            apart(x, a.min(*b), a.max(*b), Vec3::ONE, *radius, r, big)
        }
        SurfaceShape::Triangle { .. } => false,
    }
}

/// [`separated`] for a round shape: the gap between the element's box and the box
/// `lo..hi` holding the shape's core (a sphere's or cylinder's centre, a capsule's
/// segment), over the axes `mask` keeps (a cylinder's horizontal ones), farther than the
/// shape's `radius`, `r` and the allowance. Every point of the element is in its box and
/// every point of the core in `lo..hi`, so their distance is at least that gap.
fn apart(x: &[Vec3], lo: Vec3, hi: Vec3, mask: Vec3, radius: f32, r: f32, big: f32) -> bool {
    let (elo, ehi) = x.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(a, b), &p| (a.min(p), b.max(p)),
    );
    let gap = (elo - hi).max(lo - ehi).max(Vec3::ZERO) * mask;
    let far = big.max(lo.abs().max_element()).max(hi.abs().max_element());
    let thr = radius + r + SEPARATION_ABS + SEPARATION_REL * far;
    dot3(gap, gap) > thr * thr
}

/// A world box holding the shape, 1 cm wide of it (the surface pass's cull).
pub fn bounds(s: &SurfaceShape) -> (Vec3, Vec3) {
    let fold = |pts: &[Vec3]| {
        pts.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(a, b), &v| (a.min(v), b.max(v)),
        )
    };
    let (lo, hi) = match s {
        SurfaceShape::Box { c, axes, half } => {
            let e = axes[0].abs() * half.x + axes[1].abs() * half.y + axes[2].abs() * half.z;
            (*c - e, *c + e)
        }
        SurfaceShape::Sphere { c, radius } => {
            (*c - Vec3::splat(*radius), *c + Vec3::splat(*radius))
        }
        SurfaceShape::Capsule { a, b, radius } => (
            a.min(*b) - Vec3::splat(*radius),
            a.max(*b) + Vec3::splat(*radius),
        ),
        SurfaceShape::Cylinder {
            c,
            radius,
            half_height,
        } => {
            let e = Vec3::new(*radius, *half_height, *radius);
            (*c - e, *c + e)
        }
        SurfaceShape::Hull { verts, .. } => fold(verts),
        SurfaceShape::Triangle { t } => fold(t),
    };
    (lo - Vec3::splat(0.01), hi + Vec3::splat(0.01))
}

// --- Strict poses (the Slang twin's fqRot, fqQmul, fqNormalize4, fqPose) ---

/// `q` applied to `v`: t = 2 (u × v), (v + w t) + u × t (u = q.xyz), each operation
/// rounded in this order.
pub fn rot(q: Quat, v: Vec3) -> Vec3 {
    let u = Vec3::new(q.x, q.y, q.z);
    let t = cross3(u, v) * 2.0;
    (v + t * q.w) + cross3(u, t)
}

/// `a b` (Hamilton product), strict: ((b.xyz a.w + a.xyz b.w) + a.xyz × b.xyz,
/// a.w b.w − a.xyz · b.xyz).
pub fn qmul(a: Quat, b: Quat) -> Quat {
    let (au, bu) = (Vec3::new(a.x, a.y, a.z), Vec3::new(b.x, b.y, b.z));
    let v = (bu * a.w + au * b.w) + cross3(au, bu);
    Quat::from_xyzw(v.x, v.y, v.z, a.w * b.w - dot3(au, bu))
}

/// `q / |q|`, strict (the length as √(((x² + y²) + z²) + w²)).
pub fn normalize4(q: Quat) -> Quat {
    let l = (((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w).sqrt();
    Quat::from_xyzw(q.x / l, q.y / l, q.z / l, q.w / l)
}

/// A collider's world position and orientation from its body's (centre of mass `c`,
/// rotation `q`) and its own pose in the body's frame: (c + q·lp, normalize(q lq)).
pub fn shape_pose(c: Vec3, q: Quat, lp: Vec3, lq: Quat) -> (Vec3, Quat) {
    (c + rot(q, lp), normalize4(qmul(q, lq)))
}

/// The world axes of rotation `r` (its local x, y, z).
pub fn axes(r: Quat) -> [Vec3; 3] {
    [rot(r, Vec3::X), rot(r, Vec3::Y), rot(r, Vec3::Z)]
}

/// Local point `l` placed at (p, axes): ((p + a₀ lₓ) + a₁ l_y) + a₂ l_z.
pub fn to_world(p: Vec3, a: &[Vec3; 3], l: Vec3) -> Vec3 {
    ((p + a[0] * l.x) + a[1] * l.y) + a[2] * l.z
}

/// A local direction `n` turned by `axes`: (a₀ nₓ + a₁ n_y) + a₂ n_z.
pub fn turn(a: &[Vec3; 3], n: Vec3) -> Vec3 {
    (a[0] * n.x + a[1] * n.y) + a[2] * n.z
}

/// A hull at (p, axes) from its local records (vertices and planes with the collider's
/// scale applied, edges as vertex pairs): world vertices, edges, planes (normal turned,
/// offset d + n·p).
pub fn hull_world(
    p: Vec3,
    a: &[Vec3; 3],
    verts: &[Vec3],
    edges: &[[u32; 2]],
    planes: &[(Vec3, f32)],
) -> SurfaceShape {
    SurfaceShape::Hull {
        verts: verts.iter().map(|&v| to_world(p, a, v)).collect(),
        edges: edges.to_vec(),
        planes: planes
            .iter()
            .map(|&(n, d)| {
                let wn = turn(a, n);
                (wn, d + dot3(wn, p))
            })
            .collect(),
    }
}

/// A heightfield cell's two triangles from its corners (i, j), (i+1, j), (i, j+1),
/// (i+1, j+1), split along the (i, j)–(i+1, j+1) diagonal as the World's heightfield
/// (`crate::heightfield`): (00, 10, 11) and (00, 11, 01).
pub fn heightfield_cell(c00: Vec3, c10: Vec3, c01: Vec3, c11: Vec3) -> [SurfaceShape; 2] {
    [
        SurfaceShape::Triangle { t: [c00, c10, c11] },
        SurfaceShape::Triangle { t: [c00, c11, c01] },
    ]
}

// --- Casts (continuous contact for fast shapes: the Slang twin's fqTranslate, fqDistance*,
// fqToi) ---

/// The shape moved by `d` (every point plus d; a hull's plane offsets plus n·d).
pub fn translated(s: &SurfaceShape, d: Vec3) -> SurfaceShape {
    match s {
        SurfaceShape::Box { c, axes, half } => SurfaceShape::Box {
            c: *c + d,
            axes: *axes,
            half: *half,
        },
        SurfaceShape::Cylinder {
            c,
            radius,
            half_height,
        } => SurfaceShape::Cylinder {
            c: *c + d,
            radius: *radius,
            half_height: *half_height,
        },
        SurfaceShape::Capsule { a, b, radius } => SurfaceShape::Capsule {
            a: *a + d,
            b: *b + d,
            radius: *radius,
        },
        SurfaceShape::Sphere { c, radius } => SurfaceShape::Sphere {
            c: *c + d,
            radius: *radius,
        },
        SurfaceShape::Hull {
            verts,
            edges,
            planes,
        } => SurfaceShape::Hull {
            verts: verts.iter().map(|&v| v + d).collect(),
            edges: edges.clone(),
            planes: planes.iter().map(|&(n, o)| (n, o + dot3(n, d))).collect(),
        },
        SurfaceShape::Triangle { t } => SurfaceShape::Triangle {
            t: [t[0] + d, t[1] + d, t[2] + d],
        },
    }
}

/// The element's distance to the shape (0 when a vertex is inside): the nearest of
/// its closest-point candidates and its vertices' distances.
pub fn distance(x: &[Vec3], s: &SurfaceShape) -> f32 {
    let mut g = f32::INFINITY;
    match x.len() {
        3 => {
            let t = [x[0], x[1], x[2]];
            x_candidates(s, &t, &mut |_, gp| g = g.min(gp));
        }
        2 => {
            let b = [x[0], x[1]];
            seg_candidates(s, &b, &mut |_, gp| g = g.min(gp));
        }
        _ => {}
    }
    for &v in x {
        match point_distance(s, v) {
            Some(d) => g = g.min(d),
            None => return 0.0,
        }
    }
    g
}

/// Cast iterations at most ([`toi`]).
pub const CAST_ITERS: u32 = 24;

/// The earliest fraction t ∈ [0, 1] of the shape's relative motion `d` at which the
/// element (a triangle, a segment or a point) comes within `r` of it (conservative
/// advancement: each advance is the distance past the band over |d|, so the band is
/// never jumped); `None` when it does not within the motion. 0 when it already is.
pub fn toi(x: &[Vec3], s: &SurfaceShape, d: Vec3, r: f32) -> Option<f32> {
    let len = len3(d);
    let mut t = 0.0f32;
    for _ in 0..CAST_ITERS {
        let g = distance(x, &translated(s, d * t));
        if g <= r {
            return Some(t);
        }
        if !(len > 0.0) {
            return None;
        }
        t += (g - r) / len;
        if t > 1.0 {
            return None;
        }
    }
    None
}

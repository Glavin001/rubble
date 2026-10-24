//! Exact phase-through detector for the fracture lab tests: in a step, the
//! ball's centre path (its pose before → after the step) meets the hull of a
//! piece that did not move during that step (GJK distance between the segment
//! and the convex hull of the piece's vertices is zero).
#![allow(dead_code)]

use engine_fracture::scene::{collider_vertices, Collider, Scene, SHAPE_BOX, SHAPE_CONVEX};

pub type V = [f64; 3];
pub fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn mul(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn qrot(q: [f64; 4], v: V) -> V {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let c = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    add(
        add(mul(u, 2.0 * dot(u, v)), mul(v, w * w - dot(u, u))),
        mul(c, 2.0 * w),
    )
}
pub fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Closest point to the origin of the simplex `pts` (1–4 points): every face
/// subset, keeping only affine combinations with non-negative weights.
fn simplex_closest(pts: &[V]) -> (V, Vec<usize>) {
    let n = pts.len();
    let mut best: Option<(f64, V, Vec<usize>)> = None;
    for mask in 1u32..(1 << n) {
        let idx: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let p0 = pts[idx[0]];
        let e: Vec<V> = idx[1..].iter().map(|&i| sub(pts[i], p0)).collect();
        let m = e.len();
        let mut a = [[0.0f64; 3]; 3];
        let mut b = [0.0f64; 3];
        for r in 0..m {
            for c in 0..m {
                a[r][c] = dot(e[r], e[c]);
            }
            b[r] = -dot(e[r], p0);
        }
        let mut ok = true;
        for col in 0..m {
            let piv = (col..m)
                .max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
                .unwrap();
            if a[piv][col].abs() < 1e-14 {
                ok = false;
                break;
            }
            a.swap(col, piv);
            b.swap(col, piv);
            for r in 0..m {
                if r != col {
                    let f = a[r][col] / a[col][col];
                    for c in 0..m {
                        a[r][c] -= f * a[col][c];
                    }
                    b[r] -= f * b[col];
                }
            }
        }
        if !ok {
            continue;
        }
        let w: Vec<f64> = (0..m).map(|r| b[r] / a[r][r]).collect();
        let w0 = 1.0 - w.iter().sum::<f64>();
        if w0 < -1e-12 || w.iter().any(|&x| x < -1e-12) {
            continue;
        }
        let mut p = mul(p0, w0);
        for r in 0..m {
            p = add(p, mul(pts[idx[r + 1]], w[r]));
        }
        let d = dot(p, p);
        if best.as_ref().is_none_or(|bb| d < bb.0) {
            best = Some((d, p, idx));
        }
    }
    let (_, p, idx) = best.expect("a vertex is always a candidate");
    (p, idx)
}

/// GJK distance between the segment [s0, s1] and the convex hull of `verts`.
pub fn segment_hull_distance(s0: V, s1: V, verts: &[V]) -> f64 {
    let support = |d: V| -> V {
        let v = *verts
            .iter()
            .max_by(|a, b| dot(**a, d).total_cmp(&dot(**b, d)))
            .unwrap();
        let s = if dot(s0, d) < dot(s1, d) { s0 } else { s1 };
        sub(v, s)
    };
    let mut simplex = vec![support([1.0, 0.0, 0.0])];
    let mut closest = simplex[0];
    for _ in 0..64 {
        let dist2 = dot(closest, closest);
        if dist2 < 1e-18 {
            return 0.0;
        }
        let p = support(mul(closest, -1.0));
        if dist2 - dot(closest, p) <= 1e-12 * dist2.max(1.0) {
            return dist2.sqrt();
        }
        simplex.push(p);
        let (c, keep) = simplex_closest(&simplex);
        simplex = keep.iter().map(|&i| simplex[i]).collect();
        closest = c;
        if simplex.len() == 4 {
            return 0.0;
        }
    }
    dot(closest, closest).sqrt()
}

/// World vertices of a hull or box piece at a body pose.
pub fn piece_vertices(scene: &Scene, c: &Collider, p: V, q: [f64; 4]) -> Vec<V> {
    if c.shape == SHAPE_CONVEX {
        return collider_vertices(&scene.hull_data, c, p, q);
    }
    let h = [c.params[0], c.params[1], c.params[2]];
    let centre = add(p, qrot(q, c.local_pos));
    let rot = qmul(q, c.local_rot);
    (0..8)
        .map(|i| {
            let corner = [
                if i & 1 != 0 { h[0] } else { -h[0] },
                if i & 2 != 0 { h[1] } else { -h[1] },
                if i & 4 != 0 { h[2] } else { -h[2] },
            ];
            add(centre, qrot(rot, corner))
        })
        .collect()
}

/// A piece collider: its body and collider index, and a bound radius about
/// the body position (its farthest vertex).
pub struct Piece {
    pub body: usize,
    pub collider: usize,
    pub bound: f64,
}

/// Every hull or box collider of the scene's structure bodies (not the
/// ground, not the projectile slots).
pub fn pieces(scene: &Scene) -> Vec<Piece> {
    let start = scene.projectile_start as usize;
    scene
        .colliders
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            (c.shape == SHAPE_CONVEX || c.shape == SHAPE_BOX)
                && (c.body as usize) >= 1
                && (c.body as usize) < start
        })
        .map(|(i, c)| {
            let b = &scene.bodies[c.body as usize];
            let bound = piece_vertices(scene, c, b.p, b.q)
                .iter()
                .map(|v| dot(sub(*v, b.p), sub(*v, b.p)).sqrt())
                .fold(0.0, f64::max);
            Piece {
                body: c.body as usize,
                collider: i,
                bound,
            }
        })
        .collect()
}

pub fn pose(snap: &[f32], b: usize) -> (V, [f64; 4]) {
    let o = b * 28;
    let f = |k: usize| snap[o + k] as f64;
    ([f(0), f(1), f(2)], [f(4), f(5), f(6), f(7)])
}

/// The first still piece the ball's centre crossed between `prev` and `now`.
pub fn crossing(
    scene: &Scene,
    pieces: &[Piece],
    prev: &[f32],
    now: &[f32],
    ball: usize,
) -> Option<String> {
    let (b0, _) = pose(prev, ball);
    let (b1, _) = pose(now, ball);
    let seg = sub(b1, b0);
    if dot(seg, seg) < 1e-12 {
        return None;
    }
    for p in pieces {
        let (x0, q0) = pose(prev, p.body);
        let (x1, q1) = pose(now, p.body);
        let moved = dot(sub(x1, x0), sub(x1, x0)).sqrt() > 1e-3
            || (q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]).abs() < 1.0 - 1e-6;
        if moved {
            continue;
        }
        // Bound prefilter: the segment's distance to the piece's body.
        let w = sub(x0, b0);
        let t = (dot(w, seg) / dot(seg, seg)).clamp(0.0, 1.0);
        let c = sub(add(b0, mul(seg, t)), x0);
        if dot(c, c).sqrt() > p.bound + 1e-3 {
            continue;
        }
        let verts = piece_vertices(scene, &scene.colliders[p.collider], x0, q0);
        if segment_hull_distance(b0, b1, &verts) <= 1e-6 {
            let r = |v: V| v.map(|x| (x * 100.0).round() / 100.0);
            return Some(format!(
                "ball {ball} centre {:?} -> {:?} crossed still piece {} (at {:?})",
                r(b0),
                r(b1),
                p.body,
                r(x0)
            ));
        }
    }
    None
}

/// Whether any piece moved between the two snapshots (the ball hit something).
pub fn any_piece_moved(pieces: &[Piece], prev: &[f32], now: &[f32]) -> bool {
    pieces.iter().any(|p| {
        let (x0, _) = pose(prev, p.body);
        let (x1, _) = pose(now, p.body);
        dot(sub(x1, x0), sub(x1, x0)).sqrt() > 1e-3
    })
}

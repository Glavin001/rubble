//! Exact phase-through detector for engine tests (box pieces, any CCD body):
//! in a step, a CCD body's centre path (pose before → after) meets the box of
//! a body that did not move during that step (GJK segment–hull distance zero).
#![allow(dead_code)]

use engine_schema::{BodyState, Collider, ShapeType};

type V = [f64; 3];
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn qrot(q: [f64; 4], v: V) -> V {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    add(add(mul(u, 2.0 * dot(u, v)), mul(v, w * w - dot(u, u))), mul(c, 2.0 * w))
}
fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}
fn f3(v: [f32; 3]) -> V {
    v.map(|x| x as f64)
}
fn f4(v: [f32; 4]) -> [f64; 4] {
    v.map(|x| x as f64)
}

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
            let piv = (col..m).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs())).unwrap();
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
        let v = *verts.iter().max_by(|a, b| dot(**a, d).total_cmp(&dot(**b, d))).unwrap();
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

fn box_vertices(b: &BodyState, c: &Collider) -> Vec<V> {
    let q = f4(b.orientation);
    let centre = add(f3(b.position), qrot(q, f3(c.local_pos)));
    let rot = qmul(q, f4(c.local_rot));
    let h = [c.params[0] as f64, c.params[1] as f64, c.params[2] as f64];
    (0..8)
        .map(|i| {
            let corner = [if i & 1 != 0 { h[0] } else { -h[0] }, if i & 2 != 0 { h[1] } else { -h[1] }, if i & 4 != 0 { h[2] } else { -h[2] }];
            add(centre, qrot(rot, corner))
        })
        .collect()
}

/// Every box collider of a body that is not `except` and did not move between
/// the snapshots, that `ball`'s centre path crossed (first one found).
pub fn crossing(cols: &[Collider], prev: &[BodyState], now: &[BodyState], ball: usize) -> Option<String> {
    let (b0, b1) = (f3(prev[ball].position), f3(now[ball].position));
    let seg = sub(b1, b0);
    if dot(seg, seg) < 1e-12 {
        return None;
    }
    for c in cols {
        let body = c.body as usize;
        if body == ball || c.shape != ShapeType::Box as u32 {
            continue;
        }
        let (p0, p1) = (&prev[body], &now[body]);
        let moved = dot(sub(f3(p1.position), f3(p0.position)), sub(f3(p1.position), f3(p0.position))).sqrt() > 1e-3
            || (0..4).map(|k| p0.orientation[k] as f64 * p1.orientation[k] as f64).sum::<f64>().abs() < 1.0 - 1e-6;
        if moved {
            continue;
        }
        let x0 = f3(p0.position);
        let w = sub(x0, b0);
        let t = (dot(w, seg) / dot(seg, seg)).clamp(0.0, 1.0);
        let d = sub(add(b0, mul(seg, t)), x0);
        let bound = (c.params[0].powi(2) + c.params[1].powi(2) + c.params[2].powi(2)).sqrt() as f64 + f3(c.local_pos).iter().map(|x| x * x).sum::<f64>().sqrt();
        if dot(d, d).sqrt() > bound + 1e-3 {
            continue;
        }
        if segment_hull_distance(b0, b1, &box_vertices(p0, c)) <= 1e-6 {
            let r = |v: V| v.map(|x| (x * 100.0).round() / 100.0);
            return Some(format!("body {ball} centre {:?} -> {:?} crossed still body {body} at {:?}", r(b0), r(b1), r(x0)));
        }
    }
    None
}

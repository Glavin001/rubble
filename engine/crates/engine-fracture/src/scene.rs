//! Scene builder (port of `fracture/geometry.mjs`).
//!
//! Selects one hierarchy level, rotates every fragment into its principal inertia
//! frame (the body ABI stores diagonal inverse inertia), packs baked hulls into the
//! convex hull arena, and emits render / bond-line / projectile vertex streams.
//! Math runs in f64 to match the JS reference bit-for-bit where it matters.

// `!(x > y)` keeps the JS reference's NaN behaviour; index loops mirror its math.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]

use crate::asset::{FractureAsset, Hull};
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

type V3 = [f64; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn len(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn norm(a: V3) -> V3 {
    let l = len(a);
    scale(a, 1.0 / if l == 0.0 { 1.0 } else { l })
}
fn rotate(q: [f64; 4], v: V3) -> V3 {
    let u = [q[0], q[1], q[2]];
    add(v, scale(cross(u, add(cross(u, v), scale(v, q[3]))), 2.0))
}
fn inverse_rotate(q: [f64; 4], v: V3) -> V3 {
    rotate([-q[0], -q[1], -q[2], q[3]], v)
}
fn v3(a: [f32; 3]) -> V3 {
    [a[0] as f64, a[1] as f64, a[2] as f64]
}

/// Body record matching the 112-byte `BodyState` ABI fields the host writes.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub p: [f32; 3],
    pub q: [f32; 4],
    pub v: [f32; 3],
    pub inv_mass: f32,
    pub inv_i: [f32; 3],
    pub flags: u32,
}

impl Body {
    /// 112-byte `BodyState`; initial pose mirrors the current pose (frame-0 warm start).
    pub fn to_bytes(&self) -> [u8; 112] {
        let mut w = [0u32; 28];
        let f = |x: f32| x.to_bits();
        w[0..3].copy_from_slice(&self.p.map(f));
        w[3] = f(self.inv_mass);
        w[4..8].copy_from_slice(&self.q.map(f));
        w[8..11].copy_from_slice(&self.inv_i.map(f));
        w[11] = self.flags;
        w[12..15].copy_from_slice(&self.v.map(f));
        w[20..23].copy_from_slice(&self.p.map(f));
        w[24..28].copy_from_slice(&self.q.map(f));
        let mut out = [0u8; 112];
        for (i, x) in w.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        out
    }
}

/// Collider: shape 0 = sphere (params.x radius), 3 = convex hull
/// (params = [arena base, vertex count, face count, bounding radius]), 4 = plane.
#[derive(Clone, Copy, Debug)]
pub struct Collider {
    pub body: u32,
    pub shape: u32,
    pub density: f32,
    pub params: [f32; 4],
}

impl Collider {
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut w = [0u32; 16];
        w[0] = self.body;
        w[1] = self.shape;
        w[7] = self.density.to_bits();
        w[11] = 1f32.to_bits(); // localRot.w
        for k in 0..4 {
            w[12 + k] = self.params[k].to_bits();
        }
        let mut out = [0u8; 64];
        for (i, x) in w.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        out
    }
}

/// Solver parameters the scene overrides (see `buildScene().params`).
#[derive(Clone, Copy, Debug)]
pub struct SceneParams {
    pub flags: u32,
    pub iterations: u32,
    pub dt: f32,
    pub gravity_y: f32,
    pub contact_offset: f32,
    pub alpha: f32,
    pub sleep_lin: f32,
    pub sleep_ang: f32,
    pub sleep_time: f32,
}

pub struct FractureScene {
    pub level: u8,
    pub bodies: Vec<Body>,
    pub colliders: Vec<Collider>,
    /// Hull arena as padded float3 records (`.w` unused).
    pub hull_data: Vec<[f32; 4]>,
    /// Render vertices: position, normal, color, body id (10 floats each).
    pub vertices: Vec<f32>,
    pub lines: Vec<f32>,
    pub projectile_vertices: Vec<f32>,
    pub projectile_start: u32,
    pub projectile_count: u32,
    pub projectile_radius: f32,
    pub projectile_mass: f32,
    pub fragment_count: usize,
    pub hull_count: usize,
    pub bond_count: usize,
    pub extent: f32,
    pub height: f32,
    pub center: [f32; 3],
    pub grid_cell_size: f32,
    pub grid_levels: u32,
    pub params: SceneParams,
}

/// Symmetric inertia eigenbasis (Jacobi). Returns the principal-frame quaternion and moments.
pub fn principal_axes(inertia: &[f32; 9]) -> Result<([f64; 4], V3)> {
    let mut a = [[0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = inertia[i * 3 + j] as f64;
        }
    }
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let (mut p, mut q) = (0usize, 1usize);
        for (i, j) in [(0, 2), (1, 2)] {
            if a[i][j].abs() > a[p][q].abs() {
                p = i;
                q = j;
            }
        }
        let diag = a[0][0].abs().max(a[1][1].abs()).max(a[2][2].abs()).max(1.0);
        if a[p][q].abs() < 1e-10 * diag {
            break;
        }
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (c, s) = (theta.cos(), theta.sin());
        let (ap, aq, apq) = (a[p][p], a[q][q], a[p][q]);
        for k in 0..3 {
            if k != p && k != q {
                let (kp, kq) = (a[k][p], a[k][q]);
                a[k][p] = c * kp - s * kq;
                a[p][k] = a[k][p];
                a[k][q] = s * kp + c * kq;
                a[q][k] = a[k][q];
            }
        }
        a[p][p] = c * c * ap - 2.0 * s * c * apq + s * s * aq;
        a[q][q] = s * s * ap + 2.0 * s * c * apq + c * c * aq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for row in &mut v {
            let (kp, kq) = (row[p], row[q]);
            row[p] = c * kp - s * kq;
            row[q] = s * kp + c * kq;
        }
    }
    let m: Vec<f64> = v.iter().flatten().copied().collect();
    let trace = m[0] + m[4] + m[8];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (m[7] - m[5]) / s,
            (m[2] - m[6]) / s,
            (m[3] - m[1]) / s,
            s / 4.0,
        ]
    } else {
        let mut i = 0;
        if m[4] > m[0] {
            i = 1;
        }
        if m[8] > m[i * 3 + i] {
            i = 2;
        }
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let s = (1.0 + m[i * 3 + i] - m[j * 3 + j] - m[k * 3 + k]).sqrt() * 2.0;
        let mut q = [0.0, 0.0, 0.0, (m[k * 3 + j] - m[j * 3 + k]) / s];
        q[i] = s / 4.0;
        q[j] = (m[j * 3 + i] + m[i * 3 + j]) / s;
        q[k] = (m[k * 3 + i] + m[i * 3 + k]) / s;
        q
    };
    let moments = [a[0][0], a[1][1], a[2][2]];
    if moments.iter().any(|x| !(*x > 0.0) || !x.is_finite()) {
        bail!("Fragment inertia is not positive definite");
    }
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let l = if l == 0.0 { 1.0 } else { l };
    Ok(([q[0] / l, q[1] / l, q[2] / l, q[3] / l], moments))
}

#[derive(Debug)]
enum HullError {
    DegenerateFace,
    Other(String),
}

/// Padded float3 hull ABI: counts, vertices, outward normals, face ranges, two
/// records per edge, interior/lo/hi, then face loops. Falls back to exhaustive
/// (uncertified-adjacency) SAT when coplanar normalization cannot certify a face.
pub fn append_hull(data: &mut Vec<[f32; 4]>, vertices: &[V3], hull: &Hull) -> Result<[f64; 4]> {
    let length = data.len();
    match pack_hull(data, vertices, hull, true) {
        Ok(p) => Ok(p),
        Err(HullError::DegenerateFace) => {
            data.truncate(length);
            pack_hull(data, vertices, hull, false).map_err(|e| match e {
                HullError::DegenerateFace => anyhow!("Degenerate convex face"),
                HullError::Other(s) => anyhow!(s),
            })
        }
        Err(HullError::Other(s)) => {
            data.truncate(length);
            Err(anyhow!(s))
        }
    }
}

fn pack_hull(
    data: &mut Vec<[f32; 4]>,
    vertices: &[V3],
    topology: &Hull,
    normalize_faces: bool,
) -> std::result::Result<[f64; 4], HullError> {
    let other = |s: &str| HullError::Other(s.to_string());
    if vertices.len() < 4
        || vertices.len() > 64
        || vertices.iter().any(|v| !v.iter().all(|x| x.is_finite()))
    {
        return Err(other("Convex hull needs 4–64 finite vertices"));
    }
    let nvf = vertices.len() as f64;
    let center = scale(vertices.iter().fold([0.0; 3], |a, v| add(a, *v)), 1.0 / nvf);
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for v in vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    // Weld exact duplicates for adjacency only; original coordinates/order are kept.
    let mut canonical: HashMap<[u64; 3], usize> = HashMap::new();
    let mut canonical_order = Vec::new();
    let vertex_ids: Vec<usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let key = v.map(|x| (if x == 0.0 { 0.0 } else { x }).to_bits());
            *canonical.entry(key).or_insert_with(|| {
                canonical_order.push(i);
                i
            })
        })
        .collect();
    let span = len(sub(hi, lo));
    let relative: Vec<V3> = vertices.iter().map(|v| sub(*v, center)).collect();
    let pick = |items: &mut dyn Iterator<Item = V3>| {
        let mut best = items.next().unwrap();
        for b in items {
            if !(dot(best, best) > dot(b, b)) {
                best = b;
            }
        }
        best
    };
    let axis = pick(&mut relative.iter().copied());
    let plane = pick(&mut relative.iter().map(|v| cross(axis, *v)));
    let plane_length = len(plane);
    let flat = relative
        .iter()
        .map(|v| dot(*v, plane).abs() / plane_length)
        .fold(f64::NEG_INFINITY, f64::max);
    if plane_length == 0.0 || flat <= (1e-10f64).max(span * 1e-8) {
        return Err(other(
            "Degenerate zero-volume hull: recook the source collision asset",
        ));
    }
    let base = data.len();
    let tolerance = (1e-9f64).max(span * 1e-6);
    struct Face {
        ids: Vec<usize>,
        n: V3,
        d: f64,
    }
    let mut raw_faces: Vec<Face> = Vec::new();
    let mut planes: Vec<Face> = Vec::new();
    let mut offset = 0usize;
    let mut closed = normalize_faces;
    for &count in &topology.sizes {
        let count = count as usize;
        if !(3..=64).contains(&count) {
            return Err(other("Convex face size outside 3..64"));
        }
        let end = (offset + count).min(topology.indices.len());
        let ids: Vec<usize> = topology.indices[offset..end]
            .iter()
            .map(|&i| vertex_ids[i as usize])
            .collect();
        offset += count;
        if ids.len() != count {
            return Err(other("Invalid convex face indices"));
        }
        let mut sum = [0.0; 3];
        for i in 1..count - 1 {
            sum = add(
                sum,
                cross(
                    sub(vertices[ids[i]], vertices[ids[0]]),
                    sub(vertices[ids[i + 1]], vertices[ids[0]]),
                ),
            );
        }
        if len(sum) < 1e-14 {
            continue;
        }
        let mut n = norm(sum);
        if dot(n, sub(vertices[ids[0]], center)) < 0.0 {
            n = scale(n, -1.0);
        }
        let mut clean: Vec<usize> = Vec::new();
        for &i in &ids {
            if !clean.contains(&i) {
                clean.push(i);
            }
        }
        if dot(sum, n) < 0.0 {
            clean.reverse();
        }
        let d = dot(n, vertices[ids[0]]);
        if !planes
            .iter()
            .any(|p| dot(p.n, n) > 1.0 - 1e-7 && (p.d - d).abs() < tolerance)
        {
            planes.push(Face { ids: vec![], n, d });
        }
        raw_faces.push(Face {
            ids: clean,
            n,
            d: f64::NAN,
        });
    }
    // Merge coplanar patches into planar convex loops (normalized path).
    let mut faces: Vec<(Vec<usize>, V3)> = Vec::new();
    let mut edge_order: Vec<(usize, usize)> = Vec::new();
    let mut edges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let source = if normalize_faces { &planes } else { &raw_faces };
    for entry in source {
        let (n, d) = (entry.n, entry.d);
        let ids = if normalize_faces {
            let u = norm(cross(
                n,
                if n[0].abs() < 0.8 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                },
            ));
            let v = cross(n, u);
            #[derive(Clone, Copy)]
            struct P {
                i: usize,
                x: f64,
                y: f64,
            }
            let mut points: Vec<P> = canonical_order
                .iter()
                .filter(|&&i| (dot(n, vertices[i]) - d).abs() < tolerance)
                .map(|&i| P {
                    i,
                    x: dot(vertices[i], u),
                    y: dot(vertices[i], v),
                })
                .collect();
            points.sort_by(|a, b| {
                let k = if a.x - b.x != 0.0 {
                    a.x - b.x
                } else {
                    a.y - b.y
                };
                k.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let turn = |a: P, b: P, c: P| (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            let mut lower: Vec<P> = Vec::new();
            let mut upper: Vec<P> = Vec::new();
            for &p in &points {
                while lower.len() > 1
                    && turn(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-14
                {
                    lower.pop();
                }
                lower.push(p);
            }
            for &p in points.iter().rev() {
                while upper.len() > 1
                    && turn(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-14
                {
                    upper.pop();
                }
                upper.push(p);
            }
            lower.pop();
            upper.pop();
            lower.into_iter().chain(upper).map(|p| p.i).collect()
        } else {
            entry.ids.clone()
        };
        if ids.len() < 3 {
            return Err(HullError::DegenerateFace);
        }
        if normalize_faces && vertices.iter().any(|p| dot(n, *p) - d > tolerance * 2.0) {
            return Err(other("Non-convex collision hull"));
        }
        let face = faces.len();
        for i in 0..ids.len() {
            let (a, b) = (ids[i], ids[(i + 1) % ids.len()]);
            let key = (a.min(b), a.max(b));
            match edges.get_mut(&key) {
                Some(edge) => {
                    if edge.len() != 3 || edge[0] != b || edge[1] != a {
                        closed = false;
                    }
                    if edge.len() == 3 {
                        edge.push(face);
                    }
                }
                None => {
                    edges.insert(key, vec![a, b, face]);
                    edge_order.push(key);
                }
            }
        }
        faces.push((ids, n));
    }
    if offset != topology.indices.len() || faces.len() > 128 || edges.len() > 192 {
        return Err(other("Invalid convex topology"));
    }
    if edges.values().any(|e| e.len() != 4) {
        closed = false;
    }
    if faces.len() < 4 {
        return Err(other(
            "Degenerate zero-volume hull: recook the source collision asset",
        ));
    }
    let (nv, nf, ne) = (vertices.len(), faces.len(), edges.len());
    let mut push = |v: V3| data.push([v[0] as f32, v[1] as f32, v[2] as f32, 0.0]);
    push([
        nv as f64,
        nf as f64,
        if closed { ne as f64 } else { -(ne as f64) },
    ]);
    vertices.iter().for_each(|v| push(*v));
    faces.iter().for_each(|f| push(f.1));
    let mut start = base + 1 + nv + 2 * nf + 2 * ne + 3;
    for f in &faces {
        push([start as f64, f.0.len() as f64, 0.0]);
        start += f.0.len();
    }
    for key in &edge_order {
        let e = &edges[key];
        push([e[0] as f64, e[1] as f64, e[2] as f64]);
        push([e.get(3).copied().unwrap_or(0) as f64, 0.0, 0.0]);
    }
    push(center);
    push(lo);
    push(hi);
    for f in &faces {
        for &i in &f.0 {
            push(vertices[i]);
        }
    }
    if data.len() >= 1 << 24 {
        return Err(other("Hull arena exceeds exact float offset range"));
    }
    // Radius only classifies grid levels; it is not a collision shape.
    Ok([base as f64, nv as f64, nf as f64, span * 0.5])
}

/// Conservative world AABB published for sleeping colliders (they skip refit).
pub fn collider_bounds(scene: &FractureScene, c: &Collider) -> ([f32; 3], [f32; 3]) {
    let body = &scene.bodies[c.body as usize];
    let p = v3(body.p);
    let q = body.q.map(|x| x as f64);
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    if c.shape == 3 {
        for i in 0..c.params[1] as usize {
            let h = scene.hull_data[c.params[0] as usize + 1 + i];
            let v = add(p, rotate(q, [h[0] as f64, h[1] as f64, h[2] as f64]));
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    } else {
        let r = if c.shape == 4 {
            1e6
        } else {
            c.params[0] as f64
        };
        for k in 0..3 {
            lo[k] = p[k] - r;
            hi[k] = p[k] + r;
        }
    }
    (
        lo.map(|x| (x - 0.0001) as f32),
        hi.map(|x| (x + 0.0001) as f32),
    )
}

/// Hierarchical hash grid from collider bounding radii (f64, as the JS host keeps
/// them): median diameter base cell, enough levels to cover the largest.
pub fn grid_config(radii: &[f64]) -> (f32, u32) {
    let mut d: Vec<f64> = radii.iter().map(|r| 2.0 * r + 0.0004).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = d.get(d.len() / 2).copied().unwrap_or(1.0);
    let base = ((0.001f64).max(median) * (1.0 + 1e-5)) as f32;
    let largest = d.last().copied().unwrap_or(base as f64);
    let levels = 2.max(2 + (largest / base as f64).log2().ceil() as i64) as u32;
    (base, levels)
}

pub fn build_scene(asset: &FractureAsset, level: u8) -> Result<FractureScene> {
    let physics = &asset.physics;
    let selected: Vec<_> = physics
        .fragments
        .iter()
        .filter(|f| f.level == level)
        .collect();
    if selected.is_empty() {
        bail!("No fragments at this level");
    }
    let mut bodies = vec![Body {
        p: [0.0; 3],
        q: [0.0, 0.0, 0.0, 1.0],
        v: [0.0; 3],
        inv_mass: 0.0,
        inv_i: [0.0; 3],
        flags: 0,
    }];
    let mut colliders = vec![Collider {
        body: 0,
        shape: 4,
        density: 0.0,
        params: [0.0, 1.0, 0.0, 0.0],
    }];
    let (mut hull_data, mut vertices, mut lines) = (Vec::new(), Vec::new(), Vec::new());
    let mut by_id: HashMap<u32, u32> = HashMap::new();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for f in &selected {
        for h in &physics.hulls[f.hull_begin as usize..(f.hull_begin + f.hull_count) as usize] {
            for v in &h.vertices {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k] as f64);
                    hi[k] = hi[k].max(v[k] as f64);
                }
            }
        }
    }
    if !lo.iter().all(|x| x.is_finite()) {
        bail!("Selected level has no collision hulls");
    }
    let center = scale(add(lo, hi), 0.5);
    let height = hi[1] - lo[1];
    let extent = sub(hi, lo).into_iter().fold(f64::NEG_INFINITY, f64::max);
    let placement = [-center[0], -lo[1], -center[2]];
    let mut hull_count = 0;
    let mut radii = Vec::new();
    for f in &selected {
        if !(f.mass > 0.0) || f.hull_count == 0 {
            bail!("Fragment {} has no dynamic mass or hull", f.id);
        }
        let (q, moments) = principal_axes(&f.inertia)?;
        let id = bodies.len() as u32;
        let p = add(v3(f.com), placement);
        bodies.push(Body {
            p: p.map(|x| x as f32),
            q: q.map(|x| x as f32),
            v: [0.0; 3],
            inv_mass: 1.0 / f.mass,
            inv_i: moments.map(|x| (1.0 / x) as f32),
            flags: 5,
        });
        by_id.insert(f.id, id);
        for h in f.hull_begin..f.hull_begin + f.hull_count {
            let hull = &physics.hulls[h as usize];
            let verts: Vec<V3> = hull
                .vertices
                .iter()
                .map(|v| inverse_rotate(q, sub(v3(*v), v3(f.com))))
                .collect();
            let params = append_hull(&mut hull_data, &verts, hull)
                .map_err(|e| anyhow!("Fragment {}, hull {h}: {e}", f.id))?;
            radii.push(params[3]);
            colliders.push(Collider {
                body: id,
                shape: 3,
                density: (f.mass as f64 / (f.volume as f64).max(1e-12)) as f32,
                params: params.map(|x| x as f32),
            });
            hull_count += 1;
        }
        let json = &asset.json;
        let mesh_index = if f.node >= 0 {
            json["nodes"][f.node as usize]["mesh"].as_u64()
        } else {
            None
        };
        let mesh = mesh_index
            .map(|m| &json["meshes"][m as usize])
            .filter(|m| m.is_object())
            .ok_or_else(|| anyhow!("Fragment {} has no render mesh", f.id))?;
        for prim in mesh["primitives"].as_array().into_iter().flatten() {
            if prim["mode"].as_u64().unwrap_or(4) != 4 {
                bail!("Demo supports triangle meshes");
            }
            let attr = |name: &str| prim["attributes"][name].as_u64();
            let positions = asset
                .accessor(attr("POSITION").ok_or_else(|| anyhow!("Missing POSITION"))? as usize)?;
            let normals = attr("NORMAL")
                .map(|i| asset.accessor(i as usize))
                .transpose()?;
            let indices: Vec<usize> = match prim["indices"].as_u64() {
                Some(i) => asset
                    .accessor(i as usize)?
                    .into_iter()
                    .map(|v| v[0] as usize)
                    .collect(),
                None => (0..positions.len()).collect(),
            };
            let color: V3 = prim["material"]
                .as_u64()
                .and_then(|m| {
                    json["materials"][m as usize]["pbrMetallicRoughness"]["baseColorFactor"]
                        .as_array()
                })
                .map(|c| [0, 1, 2].map(|k| c.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0)))
                .unwrap_or([0.65, 0.58, 0.47]);
            let pos = |i: usize| [positions[i][0], positions[i][1], positions[i][2]];
            for t in indices.chunks(3) {
                if t.len() != 3 || t.iter().any(|&i| i >= positions.len()) {
                    bail!("Invalid triangle indices");
                }
                let normal = norm(cross(sub(pos(t[1]), pos(t[0])), sub(pos(t[2]), pos(t[0]))));
                for &i in t {
                    let n = normals
                        .as_ref()
                        .and_then(|n| n.get(i))
                        .map(|n| [n[0], n[1], n[2]])
                        .unwrap_or(normal);
                    for x in inverse_rotate(q, pos(i))
                        .into_iter()
                        .chain(inverse_rotate(q, n))
                        .chain(color)
                    {
                        vertices.push(x as f32);
                    }
                    vertices.push(id as f32);
                }
            }
        }
    }
    // Bonds are inspection-only line segments between body origins (or a world anchor).
    let bonds: Vec<_> = physics.bonds.iter().filter(|b| b.level == level).collect();
    for b in &bonds {
        let Some(&a) = by_id.get(&b.a) else { continue };
        let other = if b.b >= 0 {
            by_id.get(&(b.b as u32)).copied()
        } else {
            None
        };
        let color = if b.anchor {
            [0.95, 0.6, 0.2]
        } else {
            [0.25, 0.9, 0.8]
        };
        let endpoints = match other {
            Some(o) => [(a, [0.0; 3]), (o, [0.0; 3])],
            None => [(a, [0.0; 3]), (0, add(v3(b.centroid), placement))],
        };
        for (body, p) in endpoints {
            lines.extend(p.iter().map(|x| *x as f32));
            lines.extend([0.0, 1.0, 0.0]);
            lines.extend(color);
            lines.push(body as f32);
        }
    }
    let projectile_start = bodies.len() as u32;
    let projectile_count = 16;
    let projectile_radius = (0.08f64).max((0.4f64).min(extent * 0.04));
    let projectile_mass =
        selected.iter().map(|f| f.mass as f64).sum::<f64>() / selected.len() as f64 * 1.5;
    let mut projectile_vertices = Vec::new();
    let pi = std::f64::consts::PI;
    for slot in 0..projectile_count {
        let id = bodies.len() as u32;
        // Inactive slots are static and below the world; launch gives them mass.
        bodies.push(Body {
            p: [slot as f32 * 2.0, -1000.0, 0.0],
            q: [0.0, 0.0, 0.0, 1.0],
            v: [0.0; 3],
            inv_mass: 0.0,
            inv_i: [0.0; 3],
            flags: 0,
        });
        radii.push(projectile_radius);
        colliders.push(Collider {
            body: id,
            shape: 0,
            density: 0.0,
            params: [projectile_radius as f32, 0.0, 0.0, 0.0],
        });
        let point = |u: f64, v: f64| [v.sin() * u.cos(), v.cos(), v.sin() * u.sin()];
        for y in 0..12 {
            for x in 0..24 {
                let (u0, u1) = (x as f64 / 24.0 * pi * 2.0, (x + 1) as f64 / 24.0 * pi * 2.0);
                let (v0, v1) = (y as f64 / 12.0 * pi, (y + 1) as f64 / 12.0 * pi);
                let (a, b, c, d) = (point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1));
                for n in [a, b, c, a, c, d] {
                    for x in scale(n, projectile_radius)
                        .into_iter()
                        .chain(n)
                        .chain([1.0, 0.3, 0.08])
                    {
                        projectile_vertices.push(x as f32);
                    }
                    projectile_vertices.push(id as f32);
                }
            }
        }
    }
    let (grid_cell_size, grid_levels) = grid_config(&radii);
    Ok(FractureScene {
        level,
        bodies,
        colliders,
        hull_data,
        vertices,
        lines,
        projectile_vertices,
        projectile_start,
        projectile_count,
        projectile_radius: projectile_radius as f32,
        projectile_mass: projectile_mass as f32,
        fragment_count: selected.len(),
        hull_count,
        bond_count: bonds.len(),
        extent: extent as f32,
        height: height as f32,
        center: [0.0, (height * 0.5 + placement[1] + lo[1]) as f32, 0.0],
        grid_cell_size,
        grid_levels,
        params: SceneParams {
            flags: 1,
            iterations: 8,
            dt: 1.0 / 120.0,
            gravity_y: -9.81,
            contact_offset: 0.002,
            alpha: 0.95,
            sleep_lin: 0.03,
            sleep_ang: 0.05,
            sleep_time: 1.0,
        },
    })
}

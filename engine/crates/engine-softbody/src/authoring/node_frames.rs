//! The body's skin as node frames: a two-level binding that is continuous by
//! construction, follows only its own part, and tears where the structure
//! does (engine/docs/vehicle-authoring.md, "Node-frame skin").
//!
//! **Why** (engine/docs/measurements/skin-shards.md): the earlier binding put
//! each vertex on one triangle of three nodes, chosen per vertex. Neighbouring
//! vertices rode different triangles (none of 7,004 shard triangles in a
//! 70 km/h wall crash had its three vertices on one triangle), 30 % of them
//! extrapolated outside their triangle and 28 % sat over 3 cm off it, so when
//! the coarse lattice folded each vertex followed its own triangle's rotation
//! and the mesh between them broke into shards, flipped slivers and a
//! saw-tooth, while the skeleton under them stretched only 1.2–2×.
//!
//! **Level 1, per node** (the part's smooth field: shape matching). Each node
//! j has a neighbourhood: the nodes of its own part within its kernel radius
//! Rⱼ ([`R_SPACING`] × its fifth-nearest part node's distance, at least
//! [`R_MIN`]: `flexskin`'s pool radii), weighted ωᵢ = (1 − r²/Rⱼ²)³ by rest
//! distance and gated by each node's tear gᵢ = 1 − tᵢ ([`node_tears`]: its
//! part's skin triangles or stiff beams stretched past steel's elongation, so
//! a node pulled out of its panel stops steering it). With p* and p⁰* the
//! weighted centroids now and at rest, and t, b the neighbourhood's two
//! principal rest directions,
//!
//! ```text
//! u = Σ ω g (q − p*) ((p⁰ − p⁰*)·t),  v = Σ ω g (q − p*) ((p⁰ − p⁰*)·b)
//! F = [û, v̂, ŵ]: û = u/|u|, ŵ = (u × v)/|u × v|, v̂ = ŵ × û
//! ```
//!
//! and F⁰ the same at rest with the same gates. The node's frame is the
//! rotation Mⱼ = F F⁰ᵀ and the position Pⱼ = p* + Mⱼ (p⁰ⱼ − p⁰*). Under any
//! rigid motion F = R F⁰, so Mⱼ = R and Pⱼ moves rigidly; at rest Mⱼ = I and
//! Pⱼ = p⁰ⱼ, whatever the gates. The kernels overlap smoothly, so
//! neighbouring nodes' frames differ smoothly even where the lattice folds.
//!
//! **Level 2, per vertex** (linear blend skinning). Each vertex blends the
//! frames of the [`BLEND`] nodes of its own part nearest it at rest, with
//! weights ω = (1 − d²/d²ₖ₊₁)² (dₖ₊₁ the next nearest node's distance: a node
//! enters or leaves the set exactly where its weight is zero, so neighbouring
//! vertices' weights are continuous):
//!
//! ```text
//! x = Σ wₖ (Pₖ + Mₖ (x⁰ − p⁰ₖ)),   n = normalize(Σ wₖ Mₖ n⁰)
//! ```
//!
//! each weight gated by its node's tear (1 − tₖ) and by whether the node is
//! still with the vertex's anchor (its untorn heaviest node: [`pair_gate`],
//! their distance against rest), then renormalised. The vertex's tear (the
//! renderer cuts at ½) is the larger of [`CUT_SHARE_SCALE`] × the share of
//! its weight so gated away, and the nodes' disagreement on where the vertex
//! is (RMS of their predictions about the blend, over its reach at rest)
//! through [`FOLD_LO`]–[`FOLD_HI`]: so a vertex at a tear line, or where the
//! structure crushed or folded within its reach past what one smooth sheet
//! can follow, is cut, and no drawn triangle spans either.
//!
//! At rest every term is x⁰ and nothing is gated, so the rest pose is exact;
//! under a rigid motion every frame moves rigidly, so the skin does. A rigid
//! piece (lamp, mirror, handle, pane) takes its centroid's weights for every
//! vertex: one map for the piece. A piece on a structure group (dash,
//! console, bay hardware: no part) blends the group nodes its triangle
//! binding used. Wheels keep their wheel binding.
//!
//! **Node tears** ([`node_tears`]): the largest stretch, against spawn, of a
//! node's own part's skin triangles' perimeters, its own part's stiff beams
//! and its distances to its neighbourhood, through [`TEAR_LO`]–[`TEAR_HI`].
//! Stretch only: bending and crushing never pull two points of a sheet apart.
//!
//! `showcase-car-render.mjs` evaluates the same on the GPU (`nodeFrames`,
//! `blendVertex`); [`eval`] is the CPU reference.

use super::skin::ShowcaseVisual;
use super::vehicle::Built;
use glam::{Mat3, Vec3};

/// A vertex's tear is this × the share of its weight gated away (torn nodes,
/// nodes no longer with its anchor): the renderer cuts at ½, so at a quarter.
pub const CUT_SHARE_SCALE: f32 = 2.0;

/// The disagreement ramp (RMS prediction spread over reach): a relative
/// rotation of about 2 × this across a vertex's nodes.
pub const FOLD_LO: f32 = 0.5;
pub const FOLD_HI: f32 = 1.0;

/// Nodes each vertex blends.
pub const BLEND: usize = 8;

/// The tear field's ramp: a node is whole below this stretch of its part's
/// skin triangles' perimeter or its stiff beams' length (over spawn), torn
/// above [`TEAR_HI`] (smoothstep between): the renderer's long-standing
/// field. Steel sheet's elongation at break is 20–40 %, so past 35 % the
/// sheet there has parted; by 80 % it certainly has.
pub const TEAR_LO: f32 = 1.35;
pub const TEAR_HI: f32 = 1.8;

/// The node frames: per node its neighbourhood and rest basis.
#[derive(Clone, Debug, Default)]
pub struct NodeFrames {
    /// Per node (CSR, `node_count + 1` starts) its neighbourhood: node and
    /// kernel weight ω (itself included); none: the node has no frame.
    pub start: Vec<u32>,
    pub node: Vec<u32>,
    pub w: Vec<f32>,
    /// Per node its neighbourhood's principal rest directions t, b.
    pub basis: Vec<[f32; 6]>,
    /// Per node its part (`u16::MAX`: none).
    pub part: Vec<u16>,
    /// Per node (CSR) its own part's stiff beams' other nodes and spawn lengths.
    pub tb_start: Vec<u32>,
    pub tb_node: Vec<u32>,
    pub tb_rest: Vec<f32>,
}

/// Kernel radius: this multiple of a node's distance to its
/// [`K_SPACING`]-th nearest node of its part, at least [`R_MIN`] (m).
pub const R_SPACING: f32 = 1.45;
pub const K_SPACING: usize = 5;
pub const R_MIN: f32 = 0.42;

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Per node its tear (0 whole … 1 torn): the largest stretch of its own part's
/// skin triangles' perimeters and of its own part's stiff beams' lengths,
/// against spawn, through the [`TEAR_LO`]–[`TEAR_HI`] ramp. Broken beams count
/// by their nodes' distance: a node pulled out of its panel is torn whether
/// its beams broke or still hold.
pub fn node_tears(v: &ShowcaseVisual, nodes: &[Vec3]) -> Vec<f32> {
    let f = &v.frames;
    let rest = |i: u32| Vec3::from_array(v.node_rest[i as usize]);
    let per = |a: Vec3, b: Vec3, c: Vec3| a.distance(b) + b.distance(c) + c.distance(a);
    (0..nodes.len())
        .map(|i| {
            let part = f.part.get(i).copied().unwrap_or(u16::MAX);
            let mut st = 0.0f32;
            if i + 1 < v.node_tri_start.len() {
                for &ti in &v.node_tri_list
                    [v.node_tri_start[i] as usize..v.node_tri_start[i + 1] as usize]
                {
                    let t = v.normal_tris[ti as usize];
                    if t.iter().any(|&k| f.part.get(k as usize) != Some(&part)) {
                        continue;
                    }
                    let r = per(rest(t[0]), rest(t[1]), rest(t[2]));
                    if r > 0.0 {
                        let q = t.map(|k| nodes[k as usize]);
                        st = st.max(per(q[0], q[1], q[2]) / r);
                    }
                }
            }
            if i + 1 < f.tb_start.len() {
                for k in f.tb_start[i] as usize..f.tb_start[i + 1] as usize {
                    st = st.max(nodes[i].distance(nodes[f.tb_node[k] as usize]) / f.tb_rest[k]);
                }
            }
            // And every node of its neighbourhood, beam or not: a node whose
            // neighbour has moved away from it past the ramp has a tear
            // beside it (a crack between nodes no beam joins).
            if i + 1 < f.start.len() {
                for k in f.start[i] as usize..f.start[i + 1] as usize {
                    let n = f.node[k] as usize;
                    let r = rest(i as u32).distance(rest(n as u32));
                    if n != i && r > 1e-6 {
                        st = st.max(nodes[i].distance(nodes[n]) / r);
                    }
                }
            }
            smoothstep(TEAR_LO, TEAR_HI, st)
        })
        .collect()
}

/// Whether nodes `i` and `j` are still together (1) or have come apart (0):
/// their distance over their rest distance through the tear ramp. Two
/// points of one sheet cannot move apart past its elongation at break
/// without the sheet parting between them; bending and crushing never pull
/// them apart, so a fold or a crush is never read as a tear.
pub fn pair_gate(nodes: &[Vec3], rest: &[Vec3], i: usize, j: usize) -> f32 {
    if i == j {
        return 1.0;
    }
    let r = rest[i].distance(rest[j]);
    if r <= 1e-6 {
        return 1.0;
    }
    1.0 - smoothstep(TEAR_LO, TEAR_HI, nodes[i].distance(nodes[j]) / r)
}

/// The Gram–Schmidt frame of node `j`'s neighbourhood at positions `q`
/// (rest positions `rest`), gated by `g`: (p*, p⁰*, F), or None where it is
/// degenerate.
fn frame_at(f: &NodeFrames, j: usize, q: &[Vec3], rest: &[Vec3], ge: &[f32]) -> Option<(Vec3, Vec3, Mat3)> {
    let (a, b) = (f.start[j] as usize, f.start[j + 1] as usize);
    let (mut w, mut c, mut c0) = (0.0f32, Vec3::ZERO, Vec3::ZERO);
    for k in a..b {
        let n = f.node[k] as usize;
        let o = f.w[k] * ge[k - a];
        w += o;
        c += q[n] * o;
        c0 += rest[n] * o;
    }
    if w <= 1e-6 {
        return None;
    }
    c /= w;
    c0 /= w;
    let bs = f.basis[j];
    let (t, bb) = (Vec3::new(bs[0], bs[1], bs[2]), Vec3::new(bs[3], bs[4], bs[5]));
    let (mut u, mut v) = (Vec3::ZERO, Vec3::ZERO);
    for k in a..b {
        let n = f.node[k] as usize;
        let o = f.w[k] * ge[k - a];
        let d = q[n] - c;
        let d0 = rest[n] - c0;
        u += d * (o * d0.dot(t));
        v += d * (o * d0.dot(bb));
    }
    let x = u.cross(v);
    let (lu, lv, lx) = (u.length(), v.length(), x.length());
    if lu <= 1e-12 || lv <= 1e-12 || lx <= 1e-4 * lu * lv {
        return None;
    }
    let un = u / lu;
    let wn = x / lx;
    Some((c, c0, Mat3::from_cols(un, wn.cross(un), wn)))
}

/// Node `j`'s frame now, (Pⱼ, Mⱼ), its neighbourhood gated by `g` (1 − tear),
/// or None (no neighbourhood, or degenerate).
pub fn node_frame(f: &NodeFrames, j: usize, nodes: &[Vec3], g: &[f32], rest: &[Vec3]) -> Option<(Vec3, Mat3)> {
    if f.start[j] == f.start[j + 1] {
        return None;
    }
    // Each neighbour gated by its own tear and by whether it is still
    // with this node (their distance against rest: [`pair_gate`]).
    let ge: Vec<f32> = (f.start[j] as usize..f.start[j + 1] as usize)
        .map(|k| {
            let n = f.node[k] as usize;
            g[n] * pair_gate(nodes, rest, j, n)
        })
        .collect();
    let (c, _, now) = frame_at(f, j, nodes, rest, &ge)?;
    let (_, c0, then) = frame_at(f, j, rest, rest, &ge)?;
    let m = now * then.transpose();
    Some((c + m * (rest[j] - c0), m))
}

/// Per vertex (or `None`: not on node frames) its position, normal and tear;
/// a vertex whose nodes have all torn is NaN with tear 1.
pub fn eval(v: &ShowcaseVisual, nodes: &[Vec3]) -> Vec<Option<(Vec3, Vec3, f32)>> {
    let f = &v.frames;
    let tears = node_tears(v, nodes);
    let rest: Vec<Vec3> = v.node_rest.iter().map(|q| Vec3::from_array(*q)).collect();
    let gates: Vec<f32> = tears.iter().map(|t| 1.0 - t).collect();
    let frames: Vec<Option<(Vec3, Mat3)>> = (0..nodes.len().min(f.start.len().saturating_sub(1)))
        .map(|j| node_frame(f, j, nodes, &gates, &rest))
        .collect();
    (0..v.positions.len())
        .map(|i| {
            let ids = v.bl_node.get(i)?;
            if ids[0] == u32::MAX {
                return None;
            }
            let ws = v.bl_w[i];
            let x0 = Vec3::from_array(v.positions[i]);
            let n0 = Vec3::from_array(v.normals[i]);
            // The anchor: the node carrying most of the vertex, untorn.
            let mut anchor = usize::MAX;
            let mut best = 0.0f32;
            for k in 0..BLEND {
                let j = ids[k];
                if j == u32::MAX {
                    break;
                }
                let s = ws[k] * (1.0 - tears[j as usize]);
                if s > best && frames[j as usize].is_some() {
                    best = s;
                    anchor = j as usize;
                }
            }
            if anchor == usize::MAX {
                return Some((Vec3::splat(f32::NAN), Vec3::Y, 1.0));
            }
            // Each node gated by its tear and by staying with the anchor; the
            // vertex's tear is the share of its weight so gated away.
            let (mut p, mut n, mut sum, mut all) = (Vec3::ZERO, Vec3::ZERO, 0.0f32, 0.0f32);
            let mut preds = [(Vec3::ZERO, 0.0f32, 0.0f32); BLEND];
            for k in 0..BLEND {
                let j = ids[k];
                if j == u32::MAX {
                    break;
                }
                let j = j as usize;
                all += ws[k];
                let Some((pk, m)) = frames[j] else { continue };
                let g = ws[k] * (1.0 - tears[j]) * pair_gate(nodes, &rest, anchor, j);
                if g <= 0.0 {
                    continue;
                }
                let pred = pk + m * (x0 - rest[j]);
                preds[k] = (pred, g, (x0 - rest[j]).length());
                p += g * pred;
                n += g * (m * n0);
                sum += g;
            }
            if sum <= 1e-6 {
                return Some((Vec3::splat(f32::NAN), Vec3::Y, 1.0));
            }
            // Cut where a quarter of its support has gone: a vertex at a
            // tear line (half on each side) is always cut, so no triangle
            // spans the gap (its neighbours, of continuous weights, agree on
            // their side).
            let tear = (CUT_SHARE_SCALE * (1.0 - sum / all)).clamp(0.0, 1.0);
            // Where its nodes no longer agree where the vertex is (the
            // structure folded or crushed within its reach past what one
            // smooth sheet can follow), the skin there has creased past
            // drawing: its tear rises with the disagreement (RMS of the
            // nodes' predictions about their blend) over its reach (the
            // weighted distance to its nodes at rest).
            let x = p / sum;
            let (mut dd, mut reach) = (0.0f32, 0.0f32);
            for &(q, g, r) in &preds {
                dd += g * (q - x).length_squared();
                reach += g * r;
            }
            let ratio = (dd / sum).sqrt() / (reach / sum).max(1e-3);
            let tear = tear.max(smoothstep(FOLD_LO, FOLD_HI, ratio));
            Some((p / sum, n.normalize_or_zero(), tear.clamp(0.0, 1.0)))
        })
        .collect()
}

/// Eigen-decomposition of a symmetric 3 × 3 (cyclic Jacobi): the values and
/// unit vectors.
fn jacobi3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0f64]];
    for _ in 0..50 {
        if a[0][1].abs() + a[0][2].abs() + a[1][2].abs() < 1e-18 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-30 {
                continue;
            }
            let th = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = if th == 0.0 { 1.0 } else { th.signum() / (th.abs() + (th * th + 1.0).sqrt()) };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in a.iter_mut() {
                let (x, y) = (row[p], row[q]);
                row[p] = c * x - s * y;
                row[q] = s * x + c * y;
            }
            for k in 0..3 {
                let (x, y) = (a[p][k], a[q][k]);
                a[p][k] = c * x - s * y;
                a[q][k] = s * x + c * y;
            }
            for row in v.iter_mut() {
                let (x, y) = (row[p], row[q]);
                row[p] = c * x - s * y;
                row[q] = s * x + c * y;
            }
        }
    }
    let col = |k: usize| [v[0][k], v[1][k], v[2][k]];
    ([a[0][0], a[1][1], a[2][2]], [col(0), col(1), col(2)])
}

/// Bind the visual's body to node frames (every vertex but the wheels'):
/// fills `v.frames`, `v.bl_node` and `v.bl_w`. `built`'s skeleton gives each
/// node's part and the stiff beams.
pub fn bind(v: &mut ShowcaseVisual, built: &Built) -> Result<(), String> {
    let sk = &built.skeleton;
    let nn = v.node_rest.len();
    let rest: Vec<Vec3> = v.node_rest.iter().map(|q| Vec3::from_array(*q)).collect();
    let part: Vec<u16> = (0..nn)
        .map(|i| sk.nodes.get(i).map_or(u16::MAX, |n| n.part))
        .collect();
    // Level 1: per part, each node's neighbourhood and rest basis.
    let mut by_part: std::collections::BTreeMap<u16, Vec<u32>> = Default::default();
    for (i, &p) in part.iter().enumerate() {
        if p != u16::MAX {
            by_part.entry(p).or_default().push(i as u32);
        }
    }
    let mut nb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); nn];
    let mut basis = vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0f32]; nn];
    for ids in by_part.values() {
        for &j in ids {
            let pj = rest[j as usize];
            let mut d: Vec<f32> = ids
                .iter()
                .filter(|&&i| i != j)
                .map(|&i| rest[i as usize].distance(pj))
                .collect();
            d.sort_by(f32::total_cmp);
            let dk = d.get(K_SPACING - 1).or(d.last()).copied().unwrap_or(R_MIN);
            let r = (R_SPACING * dk).max(R_MIN);
            let e: Vec<(u32, f32)> = ids
                .iter()
                .filter_map(|&i| {
                    let q = rest[i as usize].distance_squared(pj) / (r * r);
                    (q < 1.0).then(|| (i, (1.0 - q).powi(3)))
                })
                .collect();
            // The principal rest directions: the weighted covariance's two
            // largest eigenvectors (f64 Jacobi).
            let w: f64 = e.iter().map(|x| x.1 as f64).sum();
            let c0 = e
                .iter()
                .map(|&(i, o)| rest[i as usize].as_dvec3() * o as f64)
                .sum::<glam::DVec3>()
                / w;
            let mut cov = [[0.0f64; 3]; 3];
            for &(i, o) in &e {
                let dd = rest[i as usize].as_dvec3() - c0;
                for (a, row) in cov.iter_mut().enumerate() {
                    for (b, x) in row.iter_mut().enumerate() {
                        *x += o as f64 * dd[a] * dd[b];
                    }
                }
            }
            let (vals, vecs) = jacobi3(cov);
            let mut ord = [0usize, 1, 2];
            ord.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
            let (t, bb) = (vecs[ord[0]], vecs[ord[1]]);
            basis[j as usize] = [t[0], t[1], t[2], bb[0], bb[1], bb[2]].map(|x| x as f32);
            nb[j as usize] = e;
        }
    }
    let csr = |nb: &[Vec<(u32, f32)>]| -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        (
            std::iter::once(0)
                .chain(nb.iter().scan(0u32, |a, e| {
                    *a += e.len() as u32;
                    Some(*a)
                }))
                .collect(),
            nb.iter().flatten().map(|e| e.0).collect(),
            nb.iter().flatten().map(|e| e.1).collect(),
        )
    };
    let (start, node, w) = csr(&nb);
    let mut f = NodeFrames {
        start,
        node,
        w,
        basis,
        part: part.clone(),
        ..Default::default()
    };
    // A node whose neighbourhood has no frame at rest drives nothing.
    let ones = vec![1.0f32; nn];
    for (j, e) in nb.iter_mut().enumerate() {
        if !e.is_empty() && frame_at(&f, j, &rest, &rest, &ones).is_none() {
            e.clear();
        }
    }
    (f.start, f.node, f.w) = csr(&nb);
    // The tear beams: each node's own part's stiff beams (no dashpots), by
    // node pair once.
    let mut tb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); nn];
    let mut seen = std::collections::HashSet::new();
    for b in &sk.beams {
        let (a, c) = (b.a as usize, b.b as usize);
        if b.damper || a >= nn || c >= nn || part[a] != part[c] || part[a] == u16::MAX {
            continue;
        }
        if !seen.insert((a.min(c), a.max(c))) {
            continue;
        }
        let r0 = rest[a].distance(rest[c]);
        if r0 > 1e-4 {
            tb[a].push((c as u32, r0));
            tb[c].push((a as u32, r0));
        }
    }
    f.tb_start = std::iter::once(0)
        .chain(tb.iter().scan(0u32, |a, e| {
            *a += e.len() as u32;
            Some(*a)
        }))
        .collect();
    f.tb_node = tb.iter().flatten().map(|e| e.0).collect();
    f.tb_rest = tb.iter().flatten().map(|e| e.1).collect();
    // Level 2: per vertex its part's BLEND nearest framed nodes (a rigid
    // piece: its centroid's).
    let nv = v.positions.len();
    let mut centroid = vec![Vec3::ZERO; nv];
    for pc in &v.pieces {
        let r = pc.1 as usize..(pc.1 + pc.2) as usize;
        let c = r.clone().map(|i| Vec3::from_array(v.positions[i])).sum::<Vec3>() / pc.2.max(1) as f32;
        for i in r {
            centroid[i] = c;
        }
    }
    let framed: std::collections::BTreeMap<u16, Vec<u32>> = by_part
        .iter()
        .map(|(p, ids)| (*p, ids.iter().copied().filter(|&i| f.start[i as usize + 1] > f.start[i as usize]).collect()))
        .collect();
    // Per piece on a structure group (`part` none): the framed nodes its
    // triangle binding used.
    let mut piece_of = vec![usize::MAX; nv];
    for (k, pc) in v.pieces.iter().enumerate() {
        for i in pc.1 as usize..(pc.1 + pc.2) as usize {
            piece_of[i] = k;
        }
    }
    let mut group_nodes: Vec<Vec<u32>> = vec![Vec::new(); v.pieces.len()];
    for i in 0..nv {
        let k = piece_of[i];
        if k == usize::MAX || v.part.get(i).copied().unwrap_or(u16::MAX) != u16::MAX {
            continue;
        }
        if v.rigid.get(i).copied().unwrap_or(0) >= 2 || v.wheel.get(i).is_some_and(|&w| w != u32::MAX) {
            continue;
        }
        for &j in &v.bind_nodes[i] {
            if (j as usize) < nn && f.start[j as usize + 1] > f.start[j as usize] {
                group_nodes[k].push(j);
            }
        }
    }
    for g in &mut group_nodes {
        g.sort_unstable();
        g.dedup();
    }
    v.bl_node = vec![[u32::MAX; BLEND]; nv];
    v.bl_w = vec![[0.0; BLEND]; nv];
    for i in 0..nv {
        if v.rigid.get(i).copied().unwrap_or(0) >= 2 {
            continue;
        }
        if v.wheel.get(i).is_some_and(|&w| w != u32::MAX) {
            continue;
        }
        let vp = v.part.get(i).copied().unwrap_or(u16::MAX);
        let ids: &[u32] = if vp == u16::MAX {
            // A piece on a structure group (dash, console, bay hardware,
            // seats): the group's nodes it was bound to, framed.
            let k = piece_of[i];
            if group_nodes[k].is_empty() {
                continue;
            }
            &group_nodes[k]
        } else {
            match framed.get(&vp).filter(|x| !x.is_empty()) {
                Some(x) => x,
                None => return Err(format!("node frames: vertex {i} on part {vp} without framed nodes")),
            }
        };
        let x = if v.rigid[i] == 1 { centroid[i] } else { Vec3::from_array(v.positions[i]) };
        let mut d: Vec<(u32, f32)> = ids.iter().map(|&j| (j, rest[j as usize].distance_squared(x))).collect();
        // The BLEND + 1 nearest, in order (a partial selection: parts have
        // hundreds of nodes, the car hundreds of thousands of vertices).
        let cmp = |a: &(u32, f32), b: &(u32, f32)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
        if d.len() > BLEND + 1 {
            d.select_nth_unstable_by(BLEND, cmp);
            d.truncate(BLEND + 1);
        }
        d.sort_by(cmp);
        let k = BLEND.min(d.len());
        let r2 = if d.len() > k { d[k].1 } else { 2.25 * d[k - 1].1.max(1e-6) }.max(1e-12);
        let mut e: Vec<(u32, f32)> = d[..k]
            .iter()
            .map(|&(j, q)| {
                let t = (1.0 - q / r2).max(0.0);
                (j, t * t)
            })
            .filter(|e| e.1 > 0.0)
            .collect();
        let s: f32 = e.iter().map(|e| e.1).sum();
        if s <= 0.0 {
            e = vec![(d[0].0, 1.0)];
        } else {
            for x in &mut e {
                x.1 /= s;
            }
        }
        // On the renderer's unorm16 grid (the blend divides by the weights'
        // sum, so the rest pose stays exact).
        for (k, (j, w)) in e.into_iter().enumerate() {
            v.bl_node[i][k] = j;
            v.bl_w[i][k] = (w * 65535.0).round() / 65535.0;
        }
    }
    v.frames = f;
    Ok(())
}

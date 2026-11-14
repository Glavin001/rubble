//! Binding-quality lint for a vehicle's visual skin: static measures of how
//! each vertex is bound and exact probe deformations of the nodes (no
//! solver), each a number with a threshold (engine/docs/measurements/
//! skin-shards.md, "Binding lint"). Run for every authored or generated car;
//! `tests/skin_lint.rs` holds the thresholds and the mutation tests.
//!
//! A vertex's **drivers** are the nodes its position depends on directly:
//! its node-frame blend nodes, else its binding triangle (and its shut-line
//! fallback's). Its **group** is its piece's own part's nodes (a piece on a
//! structure group: the nodes its triangle binding used).

use super::skin::{deform_beams, ShowcaseVisual};
use super::vehicle::Built;
use glam::{Mat3, Quat, Vec3};
use std::collections::{HashMap, HashSet};

/// The lint's measures (see each field).
#[derive(Clone, Debug, Default)]
pub struct LintReport {
    pub vertices: usize,
    /// A1 group purity: vertices with a driver outside their group.
    pub cross_group: usize,
    /// A3/B3 amplification: per probed node, the largest vertex displacement
    /// over the node's (1 cm along x, y, z): p99 and max over vertices that
    /// move, and the worst (vertex, gain).
    pub gain_p99: f32,
    pub gain_max: f32,
    pub gain_worst: Vec<(usize, f32)>,
    /// A3 weight norm: a vertex's Σ|weight| over its drivers (blend weights;
    /// a triangle binding's |1 − α − β| + |α| + |β|): 1 for a convex blend,
    /// above 1 where it extrapolates (the static spike predictor). Max, and
    /// the count above [`WEIGHT_NORM_LIMIT`].
    pub weight_norm_max: f32,
    pub weight_norm_over: usize,
    /// A4 reach: a vertex's distance to its nearest driver over the median
    /// spacing of its drivers' part (max, and the count past [`REACH_LIMIT`]).
    pub reach_max: f32,
    pub reach_over: usize,
    /// A6 neighbour coherence: mesh edges whose two vertices share no driver
    /// and no beam joins any of their drivers.
    pub incoherent_edges: usize,
    /// A7 group health: parts with under 3 nodes or all on a line.
    pub bad_groups: Vec<String>,
    /// B1 locality: moving one part's nodes rigidly (alone) moves vertices
    /// outside it (count, summed over parts) and its own vertices off the
    /// rigid motion (max error, m).
    pub leaks: usize,
    pub rigid_err: f32,
    /// B4 part separation: with one detachable part moved 1 m away, drawn
    /// triangles with an edge more than 1 cm longer than at rest (summed).
    pub bridges: usize,
}

/// A4: a vertex this many of its part's node spacings from its nearest
/// driver extrapolates its frame (BeamNG: no part of a mesh far from its
/// nodes). **Choice** from the measured cars (skin-shards.md).
pub const REACH_LIMIT: f32 = 3.0;

/// A3: a convex blend has Σ|w| = 1; the unorm16 grid and f32 rounding add
/// at most 10⁻⁴. Above this a vertex extrapolates.
pub const WEIGHT_NORM_LIMIT: f32 = 1.001;

fn drivers(v: &ShowcaseVisual, i: usize) -> Vec<u32> {
    if let Some(b) = v.bl_node.get(i).filter(|b| b[0] != u32::MAX) {
        return b.iter().copied().filter(|&j| j != u32::MAX).collect();
    }
    if v.rigid.get(i).copied().unwrap_or(0) >= 2 {
        return Vec::new();
    }
    let mut d = v.bind_nodes[i].to_vec();
    if let Some(&a) = v.alt.get(i) {
        if a > 0 {
            d.extend(v.alt_nodes[a as usize - 1]);
        }
    }
    d
}

/// Every vertex's drivers' rest-position evaluation (the deform at rest
/// moved by `f` on the nodes).
fn positions(v: &ShowcaseVisual, nodes: &[Vec3]) -> Vec<Vec3> {
    deform_beams(v, nodes, None, None).0
}

/// Run the lint. `built`: the car's skeleton (parts, beams). `probe_every`:
/// B3 probes every this-many-th node (0: none).
pub fn lint(v: &ShowcaseVisual, built: &Built, probe_every: usize) -> LintReport {
    let sk = &built.skeleton;
    let nn = v.node_rest.len();
    let rest: Vec<Vec3> = v.node_rest.iter().map(|q| Vec3::from_array(*q)).collect();
    let part = |j: u32| sk.nodes.get(j as usize).map_or(u16::MAX, |n| n.part);
    let n = v.positions.len();
    let mut r = LintReport {
        vertices: n,
        ..Default::default()
    };
    let drv: Vec<Vec<u32>> = (0..n).map(|i| drivers(v, i)).collect();
    // Groups: per piece on a structure group, the nodes its triangle binding
    // used; otherwise its part.
    let mut piece_of = vec![usize::MAX; n];
    for (k, pc) in v.pieces.iter().enumerate() {
        for i in pc.1 as usize..(pc.1 + pc.2) as usize {
            piece_of[i] = k;
        }
    }
    let mut group_nodes: Vec<HashSet<u32>> = vec![HashSet::new(); v.pieces.len()];
    for i in 0..n {
        if v.part.get(i).copied().unwrap_or(u16::MAX) == u16::MAX && piece_of[i] != usize::MAX {
            group_nodes[piece_of[i]].extend(v.bind_nodes[i]);
        }
    }
    // A1.
    for i in 0..n {
        if drv[i].is_empty() {
            continue;
        }
        let p = v.part.get(i).copied().unwrap_or(u16::MAX);
        let ok = if p != u16::MAX {
            drv[i].iter().all(|&j| part(j) == p)
        } else {
            piece_of[i] != usize::MAX && drv[i].iter().all(|j| group_nodes[piece_of[i]].contains(j))
        };
        r.cross_group += !ok as usize;
    }
    // A7 and part spacing.
    let mut by_part: HashMap<u16, Vec<u32>> = HashMap::new();
    for j in 0..nn.min(sk.nodes.len()) {
        by_part.entry(part(j as u32)).or_default().push(j as u32);
    }
    let mut spacing: HashMap<u16, f32> = HashMap::new();
    for (p, ids) in &by_part {
        let name = sk.parts.get(*p as usize).cloned().unwrap_or_else(|| format!("{p}"));
        let mut nearest: Vec<f32> = ids
            .iter()
            .map(|&a| {
                ids.iter()
                    .filter(|&&b| b != a)
                    .map(|&b| rest[a as usize].distance(rest[b as usize]))
                    .fold(f32::MAX, f32::min)
            })
            .filter(|d| d.is_finite() && *d < f32::MAX)
            .collect();
        nearest.sort_by(f32::total_cmp);
        spacing.insert(*p, nearest.get(nearest.len() / 2).copied().unwrap_or(0.0));
        if ids.len() < 3 {
            r.bad_groups.push(format!("{name}: {} nodes", ids.len()));
            continue;
        }
        // Collinear: the covariance's second eigenvalue against its first.
        let c = ids.iter().map(|&j| rest[j as usize]).sum::<Vec3>() / ids.len() as f32;
        let mut m = Mat3::ZERO;
        for &j in ids {
            let d = rest[j as usize] - c;
            m += Mat3::from_cols(d * d.x, d * d.y, d * d.z);
        }
        let tr = m.x_axis.x + m.y_axis.y + m.z_axis.z;
        // Sum of the 2×2 principal minors = λ₁λ₂ + λ₁λ₃ + λ₂λ₃.
        let minors = m.x_axis.x * m.y_axis.y - m.y_axis.x * m.x_axis.y + m.x_axis.x * m.z_axis.z
            - m.z_axis.x * m.x_axis.z
            + m.y_axis.y * m.z_axis.z
            - m.z_axis.y * m.y_axis.z;
        if minors <= 1e-4 * tr * tr {
            r.bad_groups.push(format!("{name}: its nodes lie on a line"));
        }
    }
    // A3.
    for i in 0..n {
        let norm = if let Some(b) = v.bl_node.get(i).filter(|b| b[0] != u32::MAX) {
            b.iter().zip(&v.bl_w[i]).filter(|(&j, _)| j != u32::MAX).map(|(_, w)| w.abs()).sum::<f32>()
        } else if v.rigid.get(i).copied().unwrap_or(0) == 0 {
            let [a, bb, _] = v.bind_coords[i];
            (1.0 - a - bb).abs() + a.abs() + bb.abs()
        } else {
            1.0
        };
        r.weight_norm_max = r.weight_norm_max.max(norm);
        r.weight_norm_over += (norm > WEIGHT_NORM_LIMIT) as usize;
    }
    // A4.
    for i in 0..n {
        if drv[i].is_empty() {
            continue;
        }
        let x = Vec3::from_array(v.positions[i]);
        let d = drv[i].iter().map(|&j| rest[j as usize].distance(x)).fold(f32::MAX, f32::min);
        let s = spacing.get(&part(drv[i][0])).copied().unwrap_or(0.0).max(1e-3);
        let q = d / s;
        r.reach_max = r.reach_max.max(q);
        r.reach_over += (q > REACH_LIMIT) as usize;
    }
    // A6.
    let mut beams: HashSet<(u32, u32)> = HashSet::new();
    for b in &sk.beams {
        beams.insert((b.a.min(b.b), b.a.max(b.b)));
    }
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for t in v.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            if !seen.insert((a.min(b), a.max(b))) {
                continue;
            }
            let (da, db) = (&drv[a as usize], &drv[b as usize]);
            if da.is_empty() || db.is_empty() || da.iter().any(|j| db.contains(j)) {
                continue;
            }
            let joined = da
                .iter()
                .any(|&x| db.iter().any(|&y| beams.contains(&(x.min(y), x.max(y)))));
            r.incoherent_edges += !joined as usize;
        }
    }
    // B3: node probes (1 cm along x, y, z), every `probe_every`-th node (0:
    // none).
    let base = positions(v, &rest);
    let mut gains: Vec<(usize, f32)> = Vec::new();
    let delta = 0.01f32;
    let probed: Vec<usize> = if probe_every == 0 { Vec::new() } else { (0..nn).step_by(probe_every).collect() };
    for j in probed {
        if part(j as u32) == u16::MAX {
            continue;
        }
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let mut q = rest.clone();
            q[j] += axis * delta;
            let p = positions(v, &q);
            for i in 0..n {
                let g = p[i].distance(base[i]) / delta;
                if g > 1e-3 && g.is_finite() {
                    gains.push((i, g));
                }
            }
        }
    }
    gains.sort_by(|a, b| a.1.total_cmp(&b.1));
    if !gains.is_empty() {
        r.gain_p99 = gains[(gains.len() - 1) * 99 / 100].1;
        r.gain_max = gains.last().unwrap().1;
        r.gain_worst = gains.iter().rev().take(5).copied().collect();
    }
    // B1 and B4: per part, a rigid motion of its nodes alone, then a 1 m
    // translation.
    let q = Quat::from_euler(glam::EulerRot::YXZ, 0.7, -0.3, 0.5);
    let shift = Vec3::new(0.4, 0.2, -0.3);
    for (&p, ids) in &by_part {
        if p == u16::MAX {
            continue;
        }
        let set: HashSet<u32> = ids.iter().copied().collect();
        let c = ids.iter().map(|&j| rest[j as usize]).sum::<Vec3>() / ids.len() as f32;
        let mut moved = rest.clone();
        for &j in ids {
            moved[j as usize] = c + q * (rest[j as usize] - c) + shift;
        }
        let pos = positions(v, &moved);
        for i in 0..n {
            if drv[i].is_empty() || !pos[i].is_finite() {
                continue;
            }
            let mine = drv[i].iter().all(|j| set.contains(j));
            let any = drv[i].iter().any(|j| set.contains(j));
            if mine {
                let want = c + q * (Vec3::from_array(v.positions[i]) - c) + shift;
                r.rigid_err = r.rigid_err.max(pos[i].distance(want));
            } else if pos[i].distance(base[i]) > 1e-5 || any {
                r.leaks += 1;
            }
        }
        // B4.
        let mut away = rest.clone();
        for &j in ids {
            away[j as usize] += Vec3::new(1.0, 0.0, 0.0);
        }
        let pos = positions(v, &away);
        let tears = super::skin::vertex_tears(v, &away, None, &pos);
        for t in v.indices.chunks_exact(3) {
            let ids3 = [t[0] as usize, t[1] as usize, t[2] as usize];
            if ids3.iter().any(|&i| !pos[i].is_finite()) || ids3.iter().all(|&i| tears[i] >= 0.5) {
                continue;
            }
            let long = [(0, 1), (1, 2), (2, 0)].iter().any(|&(a, b)| {
                let l0 = Vec3::from_array(v.positions[ids3[a]]).distance(Vec3::from_array(v.positions[ids3[b]]));
                pos[ids3[a]].distance(pos[ids3[b]]) > l0 + 0.01
            });
            r.bridges += long as usize;
        }
    }
    r
}

/// B3 for one node and vertex: the vertex's displacement over node `j`'s
/// (1 cm, the largest of x, y, z).
pub fn node_gain(v: &ShowcaseVisual, j: usize, i: usize) -> f32 {
    let rest: Vec<Vec3> = v.node_rest.iter().map(|q| Vec3::from_array(*q)).collect();
    let base = positions(v, &rest)[i];
    [Vec3::X, Vec3::Y, Vec3::Z]
        .iter()
        .map(|&a| {
            let mut q = rest.clone();
            q[j] += a * 0.01;
            positions(v, &q)[i].distance(base) / 0.01
        })
        .fold(0.0, f32::max)
}

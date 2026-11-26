//! Continuous collision response (Bridson, Fedkiw and Anderson 2002; Harmon et al. 2008), the
//! physical half of continuous contact: every listed pair's straight paths over the step are
//! tested with additive CCD (Li, Kaufman and Jiang 2021: conservative, it may report an impact
//! early, never late) for the first time its gap closes to its thickness; a pair that does
//! gets an inelastic impulse along its normal at that time, equal and opposite on its two
//! sides by their nodes' weights and inverse masses (momentum kept, approach removed, no
//! energy added), Jacobi with mass splitting, and the paths are tested again. Pairs still
//! colliding after `rounds` form impact zones (union of their nodes), each moved as one rigid
//! body over the step with its linear and angular momentum (Bridson 2002 §7.4): inelastic
//! within the zone, kinetic energy never rises. Nothing is scaled or truncated.

use glam::{DMat3, DVec3, Vec3};

use super::dat::{closest_on_triangle, closest_segments};

/// A pair for the response: its four nodes (vehicle, node), the first `nn` one side.
pub(super) struct CcdPair {
    pub k: [(usize, u32); 4],
    pub nn: usize,
    /// The gap the response keeps (m).
    pub xi: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CcdStats {
    /// Pair impacts found (summed over rounds), rounds run, impact zones formed and their
    /// nodes, and pairs still colliding when the zones' cap was reached (an error: counted).
    pub hits: u64,
    pub rounds: u64,
    pub zones: u64,
    pub zone_nodes: u64,
    pub unresolved: u64,
    /// Kinetic energy change of the response (J; ≤ 0), and the largest change in a step of
    /// the vehicles' total linear (N·s) and angular (N·m·s) momentum (0 but rounding).
    pub ke: f64,
    pub dp: f64,
    pub dl: f64,
}

/// The closest points of a pair at positions `p` (four points): their weights (signed by
/// side: + on the second side) and the gap vector from the first side to the second.
fn closest(nn: usize, p: [Vec3; 4]) -> ([f32; 4], Vec3) {
    if nn == 3 {
        let c = closest_on_triangle(p[3], p[0], p[1], p[2]);
        let w = super::dat::barycentric_pub(c, p[0], p[1], p[2]);
        ([-w[0], -w[1], -w[2], 1.0], p[3] - c)
    } else {
        let (c1, c2) = closest_segments(p[0], p[1], p[2], p[3]);
        let s = super::dat::along_pub(c1, p[0], p[1]);
        let t = super::dat::along_pub(c2, p[2], p[3]);
        ([-(1.0 - s), -s, 1.0 - t, t], c2 - c1)
    }
}

/// Additive CCD: the first time in [0, 1] the gap closes to `xi` along the straight paths
/// `p + τ·dp`, conservatively (None: it stays above). Its iteration count is capped
/// (`MAX_ITERS`, 32): at the cap the time reached is returned (an impact, conservative).
pub(super) fn accd(nn: usize, p: [Vec3; 4], dp: [Vec3; 4], xi: f32) -> Option<f32> {
    const S: f32 = 0.1;
    const MAX_ITERS: u32 = 32;
    let mean = (dp[0] + dp[1] + dp[2] + dp[3]) * 0.25;
    let d: [Vec3; 4] = dp.map(|v| v - mean);
    let mx = |r: std::ops::Range<usize>| r.map(|j| d[j].length()).fold(0.0f32, f32::max);
    let lp = mx(0..nn) + mx(nn..4);
    if lp <= 0.0 {
        return None;
    }
    let gap = |q: [Vec3; 4]| closest(nn, q).1.length();
    let mut x = p;
    let dist = gap(x);
    if dist <= xi {
        return Some(0.0);
    }
    let g = S * (dist * dist - xi * xi) / (dist + xi);
    let mut t = 0.0f32;
    let mut tl = (1.0 - S) * (dist * dist - xi * xi) / ((dist + xi) * lp);
    for it in 0..MAX_ITERS {
        for j in 0..4 {
            x[j] += d[j] * tl;
        }
        let dd = gap(x);
        if it > 0 && (dd * dd - xi * xi) / (dd + xi) < g {
            return Some(t);
        }
        t += tl;
        if t > 1.0 {
            return None;
        }
        tl = 0.9 * (dd * dd - xi * xi) / ((dd + xi) * lp);
    }
    Some(t)
}

/// The response over the step: `x0` the start, `x1` the end (moved by the impulses), `vel`
/// the velocities (changed with them: a position change h·Δv), `im` inverse masses.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve(
    pairs: &[CcdPair],
    x0: &[Vec<Vec3>],
    x1: &mut [Vec<Vec3>],
    vel: &mut [Vec<Vec3>],
    m: &[Vec<f32>],
    im: &[Vec<f32>],
    h: f32,
    rounds: u32,
    zone_rounds: u32,
    st: &mut CcdStats,
) {
    let ke = |vel: &[Vec<Vec3>]| -> f64 {
        vel.iter().zip(m).map(|(v, m)| v.iter().zip(m).map(|(v, &m)| 0.5 * m as f64 * v.length_squared() as f64).sum::<f64>()).sum()
    };
    // Linear and angular momentum (about the origin, at the step's end positions).
    let mom = |x: &[Vec<Vec3>], vel: &[Vec<Vec3>]| -> (DVec3, DVec3) {
        let (mut p, mut l) = (DVec3::ZERO, DVec3::ZERO);
        for v in 0..vel.len() {
            for i in 0..vel[v].len() {
                let q = vel[v][i].as_dvec3() * m[v][i] as f64;
                p += q;
                l += x[v][i].as_dvec3().cross(q);
            }
        }
        (p, l)
    };
    let ke0 = ke(vel);
    let (p0, l0) = mom(x1, vel);
    let pos = |x: &[Vec<Vec3>], k: &[(usize, u32); 4]| k.map(|(v, i)| x[v][i as usize]);
    // The pairs that collide over the current paths, with their time of impact.
    let collide = |x1: &[Vec<Vec3>]| -> Vec<(usize, f32)> {
        pairs
            .iter()
            .enumerate()
            .filter_map(|(q, pr)| {
                let (a, b) = (pos(x0, &pr.k), pos(x1, &pr.k));
                let dp = [0, 1, 2, 3].map(|j| b[j] - a[j]);
                // An impact only while the pair still closes there (one inside its gap and
                // parting is not).
                accd(pr.nn, a, dp, pr.xi).filter(|&t| {
                    let at = [0, 1, 2, 3].map(|j| a[j] + dp[j] * t);
                    let (w, gv) = closest(pr.nn, at);
                    let n = gv.normalize_or_zero();
                    (0..4).map(|j| w[j] * dp[j].dot(n)).sum::<f32>() < 0.0
                }).map(|t| (q, t))
            })
            .collect()
    };
    let finish = |st: &mut CcdStats, x1: &[Vec<Vec3>], vel: &[Vec<Vec3>]| {
        st.ke += ke(vel) - ke0;
        let (p, l) = mom(x1, vel);
        st.dp = st.dp.max((p - p0).length());
        st.dl = st.dl.max((l - l0).length());
    };
    // Mass splitting (Tonge et al. 2012): each node's share of a round's impulses is its
    // inverse mass over its count of active pairs, in every pair's effective mass; each pair's
    // impulse is then applied in full on both sides, equal and opposite (its signed weights
    // sum to zero), so momentum is exact.
    let mut count: Vec<Vec<u32>> = x1.iter().map(|x| vec![0; x.len()]).collect();
    for _ in 0..rounds {
        let hit = collide(x1);
        st.rounds += 1;
        if hit.is_empty() {
            finish(st, x1, vel);
            return;
        }
        st.hits += hit.len() as u64;
        let act: Vec<(usize, Vec3, [f32; 4], f32, f32)> = hit
            .iter()
            .filter_map(|&(q, t)| {
                let pr = &pairs[q];
                let (a, b) = (pos(x0, &pr.k), pos(x1, &pr.k));
                let at = [0, 1, 2, 3].map(|j| a[j] + (b[j] - a[j]) * t);
                let (w, gv) = closest(pr.nn, at);
                let n = gv.normalize_or_zero();
                // Relative normal closing over the path (Δx / h) and of the velocities: the
                // velocities' closing is cancelled by an impulse; the rest (the contact's
                // position shifts) by a position change alone: no speed added.
                let vr: f32 = (0..4).map(|j| w[j] * (b[j] - a[j]).dot(n)).sum::<f32>() / h;
                let vv: f32 = (0..4).map(|j| w[j] * vel[pr.k[j].0][pr.k[j].1 as usize].dot(n)).sum();
                (n != Vec3::ZERO && vr < 0.0).then_some((q, n, w, vr, vv))
            })
            .collect();
        for &(q, _, w, _, _) in &act {
            for j in 0..4 {
                let (v, i) = pairs[q].k[j];
                if w[j] != 0.0 && im[v][i as usize] > 0.0 {
                    count[v][i as usize] += 1;
                }
            }
        }
        let mut dx: Vec<(usize, u32, Vec3, Vec3)> = Vec::new();
        for &(q, n, w, vr, vv) in &act {
            let k = pairs[q].k;
            let wsum: f32 = (0..4).map(|j| w[j] * w[j] * im[k[j].0][k[j].1 as usize] * count[k[j].0][k[j].1 as usize] as f32).sum();
            if wsum <= 0.0 {
                continue;
            }
            let (l, lv) = (-vr / wsum, (-vv / wsum).clamp(0.0, -vr / wsum));
            for j in 0..4 {
                let mi = im[k[j].0][k[j].1 as usize];
                if mi > 0.0 && w[j] != 0.0 {
                    dx.push((k[j].0, k[j].1, n * (w[j] * mi * l * h), n * (w[j] * mi * lv)));
                }
            }
        }
        for (v, i, d, dv) in dx {
            x1[v][i as usize] += d;
            vel[v][i as usize] += dv;
            count[v][i as usize] = 0;
        }
    }
    // Impact zones: the nodes of still-colliding pairs, merged; each moves rigidly. A pair
    // whose nodes are all in one zone moves rigidly and cannot collide: it is not re-tested.
    let mut zone_of: std::collections::HashMap<(usize, u32), usize> = Default::default();
    let mut parent: Vec<usize> = Vec::new();
    fn find(p: &mut [usize], mut a: usize) -> usize {
        while p[a] != a {
            p[a] = p[p[a]];
            a = p[a];
        }
        a
    }
    let across = |q: usize, zone_of: &std::collections::HashMap<(usize, u32), usize>, parent: &mut Vec<usize>| {
        let ids: Vec<Option<usize>> = pairs[q].k.iter().map(|key| zone_of.get(key).map(|&z| find(parent, z))).collect();
        !(ids[0].is_some() && ids.iter().all(|z| *z == ids[0]))
    };
    for _ in 0..zone_rounds {
        let hit: Vec<(usize, f32)> = collide(x1).into_iter().filter(|&(q, _)| across(q, &zone_of, &mut parent)).collect();
        if hit.is_empty() {
            finish(st, x1, vel);
            return;
        }
        for &(q, _) in &hit {
            let ids: Vec<usize> = pairs[q]
                .k
                .iter()
                .map(|key| {
                    *zone_of.entry(*key).or_insert_with(|| {
                        parent.push(parent.len());
                        parent.len() - 1
                    })
                })
                .collect();
            for &id in &ids[1..] {
                let (ra, rb) = (find(&mut parent, ids[0]), find(&mut parent, id));
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }
        let mut zones: std::collections::BTreeMap<usize, Vec<(usize, u32)>> = Default::default();
        let keys: Vec<(usize, u32)> = zone_of.keys().copied().collect();
        for key in keys {
            let z = find(&mut parent, zone_of[&key]);
            zones.entry(z).or_default().push(key);
        }
        st.zones += zones.len() as u64;
        for nodes in zones.values() {
            st.zone_nodes += nodes.len() as u64;
            rigid_zone(nodes, x0, x1, vel, m, im, h);
        }
    }
    st.unresolved += collide(x1).into_iter().filter(|&(q, _)| across(q, &zone_of, &mut parent)).count() as u64;
    finish(st, x1, vel);
}

/// One impact zone moved as a rigid body over the step: its centre at the start, its linear
/// and angular momentum (ω = I⁻¹L) from the nodes' path velocities, each node's end the
/// zone's exact rotation and translation of its start (Bridson 2002 §7.4). An immovable node
/// pins the zone: it then stays where it started.
fn rigid_zone(nodes: &[(usize, u32)], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], m: &[Vec<f32>], im: &[Vec<f32>], h: f32) {
    let at = |x: &[Vec<Vec3>], k: &(usize, u32)| x[k.0][k.1 as usize].as_dvec3();
    if nodes.iter().any(|k| im[k.0][k.1 as usize] == 0.0) {
        for k in nodes {
            vel[k.0][k.1 as usize] = Vec3::ZERO;
            x1[k.0][k.1 as usize] = x0[k.0][k.1 as usize];
        }
        return;
    }
    let hh = h as f64;
    let mass: f64 = nodes.iter().map(|k| m[k.0][k.1 as usize] as f64).sum();
    let c = nodes.iter().map(|k| at(x0, k) * m[k.0][k.1 as usize] as f64).sum::<DVec3>() / mass;
    // The zone's momentum and angular momentum from the nodes' velocities (a projection onto
    // rigid motion: kinetic energy never rises).
    let v = |k: &(usize, u32)| vel[k.0][k.1 as usize].as_dvec3();
    let vc = nodes.iter().map(|k| v(k) * m[k.0][k.1 as usize] as f64).sum::<DVec3>() / mass;
    let (mut l, mut i) = (DVec3::ZERO, DMat3::ZERO);
    for k in nodes {
        let (r, mm) = (at(x0, k) - c, m[k.0][k.1 as usize] as f64);
        l += r.cross(v(k) - vc) * mm;
        i += (DMat3::IDENTITY * r.length_squared() - DMat3::from_cols(r * r.x, r * r.y, r * r.z)) * mm;
    }
    let w = if i.determinant().abs() > 1e-18 { i.inverse() * l } else { DVec3::ZERO };
    let ang = w.length() * hh;
    let rot = if ang > 0.0 { glam::DQuat::from_axis_angle(w / w.length(), ang) } else { glam::DQuat::IDENTITY };
    for k in nodes {
        let r = at(x0, k) - c;
        x1[k.0][k.1 as usize] = (c + vc * hh + rot * r).as_vec3();
        vel[k.0][k.1 as usize] = (vc + w.cross(r)).as_vec3();
    }
}

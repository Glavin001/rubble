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

use glam::{DVec3, Vec3};

use crate::explicit_beams::{to_fixed_shift, SHIFT_FIXED};

/// Velocity changes summed per node in fixed point: units per m/s, 2²² (±476 m/s in an i32),
/// so the GPU's atomic integer sums give the same bits in any order.
pub const VEL_FIXED: f32 = 4_194_304.0;

/// A velocity-change component in [`VEL_FIXED`] units, rounded to nearest with ties to even
/// (as [`to_fixed_shift`]; the kernels' twin spells it out).
pub fn to_fixed_vel(x: f32) -> i32 {
    if !(x * VEL_FIXED).abs().le(&2.0e9) {
        FIXED_OVERFLOWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    (x * VEL_FIXED).clamp(-2.0e9, 2.0e9).round_ties_even() as i32
}

/// Fixed-point sums past their range (a velocity change past ±476 m/s in one term or one node's
/// sum, a position step past the shift's): an error, counted here rather than wrapped silently
/// (the result is then wrong, and momentum is not kept). Tests assert it stays 0.
pub static FIXED_OVERFLOWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`FIXED_OVERFLOWS`] so far.
pub fn fixed_overflows() -> u64 {
    FIXED_OVERFLOWS.load(std::sync::atomic::Ordering::Relaxed)
}

fn add_counted(a: i32, b: i32) -> i32 {
    let (r, o) = a.overflowing_add(b);
    if o {
        FIXED_OVERFLOWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    r
}

/// Per-node fixed-point sums of a round's velocity changes and position steps.
pub(super) struct FixedSums {
    pub dv: Vec<Vec<[i32; 3]>>,
    pub dx: Vec<Vec<[i32; 3]>>,
}

impl FixedSums {
    pub fn new(shape: &[Vec<Vec3>]) -> Self {
        Self { dv: shape.iter().map(|x| vec![[0; 3]; x.len()]).collect(), dx: shape.iter().map(|x| vec![[0; 3]; x.len()]).collect() }
    }
    pub fn add_v(&mut self, v: usize, i: u32, d: Vec3) {
        let e = &mut self.dv[v][i as usize];
        for (k, c) in d.to_array().into_iter().enumerate() {
            e[k] = add_counted(e[k], to_fixed_vel(c));
        }
    }
    pub fn add_x(&mut self, v: usize, i: u32, d: Vec3) {
        let e = &mut self.dx[v][i as usize];
        for (k, c) in d.to_array().into_iter().enumerate() {
            e[k] = add_counted(e[k], to_fixed_shift(c));
        }
    }
    /// Applies the sums (velocities and, if given, positions) and clears them.
    pub fn apply(&mut self, vel: &mut [Vec<Vec3>], mut x: Option<&mut [Vec<Vec3>]>) {
        for v in 0..vel.len() {
            for i in 0..vel[v].len() {
                let a = std::mem::take(&mut self.dv[v][i]);
                if a != [0; 3] {
                    vel[v][i] += Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32) * (1.0 / VEL_FIXED);
                }
                let b = std::mem::take(&mut self.dx[v][i]);
                if let (Some(x), true) = (x.as_deref_mut(), b != [0; 3]) {
                    x[v][i] += Vec3::new(b[0] as f32, b[1] as f32, b[2] as f32) * (1.0 / SHIFT_FIXED);
                }
            }
        }
    }
}

/// A pair for the response: its four nodes (vehicle, node), the first `nn` one side.
#[derive(Clone)]
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
    /// Zones that would turn more than 1 rad in a step (an error: not moved).
    pub zone_spin: u64,
    pub dp: f64,
    pub dl: f64,
    /// Non-rigid zones (`soft` rounds): rounds run, the constraint rows they held (summed over
    /// rounds), and the pairs left to the rigid zones after them.
    pub soft_rounds: u64,
    pub soft_rows: u64,
    pub soft_left: u64,
    /// Velocity rows (`DAT_RESPONSE=vel`): Jacobi iterations run, summed over rounds.
    pub row_iters: u64,
    /// Exact cluster solves (`ccd_exact`): clusters solved, their rows and pivots, clusters too
    /// big to solve, and solves still unsettled after them.
    pub exact: u64,
    pub exact_rows: u64,
    pub exact_pivots: u64,
    pub exact_big: u64,
    pub exact_left: u64,
    /// Velocity contact's work on the integrator's conserved energy (J): by its rows' solve and
    /// by the impact zones after it (`f(v) = m(v·v⁻ + ½|v|²)/2` summed, its change).
    pub work_rows: f64,
    pub work_zones: f64,
    /// Velocity contact's passes solved again with a smaller e by the impact law's guard.
    pub e_scaled: u64,
    /// Rows given Poisson's restitution impulse (velocity contact).
    pub restituted: u64,
    /// Solves taken as resolved with an over-push (`DAT_OVERPUSH=1`).
    pub overpush: u64,
}

/// The impact law's guard's tolerance (J a pass): the rows' work on the conserved energy above
/// this solves the pass again with a smaller e.
pub(super) const E_GUARD_TOL: f64 = 1e-3;

/// The closest points of a pair at positions `p` (four points): their weights (signed by
/// side: + on the second side) and the gap vector from the first side to the second.
pub(super) fn closest(nn: usize, p: [Vec3; 4]) -> ([f32; 4], Vec3) {
    use engine_schema::feature_query as fq;
    if nn == 3 {
        let (c, w) = fq::closest_on_triangle(p[3], p[0], p[1], p[2]);
        ([-w[0], -w[1], -w[2], 1.0], p[3] - c)
    } else {
        let (s, t) = fq::closest_seg_seg(p[0], p[1], p[2], p[3]);
        let (c1, c2) = (p[0] + (p[1] - p[0]) * s, p[2] + (p[3] - p[2]) * t);
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
    soft: u32,
    zone_rounds: u32,
    st: &mut CcdStats,
    vpre: &[Vec<Vec3>],
    e: f32,
    warm: Option<&mut super::ccd_vel::Warm>,
) {
    let vel_only = warm.is_some();
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
                let closing = |t: &f32| {
                    let at = [0, 1, 2, 3].map(|j| a[j] + dp[j] * *t);
                    let (w, gv) = closest(pr.nn, at);
                    let n = gv.normalize_or_zero();
                    (0..4).map(|j| w[j] * dp[j].dot(n)).sum::<f32>() < 0.0
                };
                accd(pr.nn, a, dp, pr.xi).filter(closing).or_else(|| {
                    // Velocity contact: a pair that starts inside its gap and parts there can
                    // still turn and pass through later in the step; it is an impact when it
                    // comes within half its start distance (head 100: a node 0.05 mm from its
                    // own triangle ended 0.1 mm through it, 2026-11-27).
                    let d0 = closest(pr.nn, a).1.length();
                    (vel_only && d0 <= pr.xi && d0 > 0.0).then(|| accd(pr.nn, a, dp, 0.5 * d0).filter(closing)).flatten()
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
    // (DAT_RESPONSE=vel: velocity-only impulse contact, `ccd_vel::velocity_rounds`, in place of
    // these rounds and with no non-rigid zones (they move positions alone).)
    let vf: Vec<Vec<Vec3>> = warm.as_ref().map_or(Vec::new(), |w| w.vf.clone());
    let shadow = |vel: &[Vec<Vec3>]| super::ccd_vel::shadow(vel, vpre, m, &vf);
    let f_in = if vel_only { shadow(vel) } else { 0.0 };
    let mut f_rows = f_in;
    if let Some(warm) = warm {
        let done = super::ccd_vel::guarded_rounds(pairs, x0, x1, vel, im, h, rounds, st, &collide, vpre, e, warm, &|v| shadow(v), f_in);
        f_rows = shadow(vel);
        st.work_rows += f_rows - f_in;
        if done {
            finish(st, x1, vel);
            return;
        }
    } else
    // (DAT_RESPONSE=rows: the rows' response, `ccd_rows`, in place of these rounds.)
    if std::env::var("DAT_RESPONSE").is_ok_and(|v| v == "rows") {
        if super::ccd_rows::rows_rounds(pairs, x0, x1, vel, im, rounds, st, &collide) {
            finish(st, x1, vel);
            return;
        }
    } else {
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
        let mut sums = FixedSums::new(x1);
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
                    sums.add_x(k[j].0, k[j].1, n * (w[j] * mi * l * h));
                    sums.add_v(k[j].0, k[j].1, n * (w[j] * mi * lv));
                }
            }
        }
        sums.apply(vel, Some(x1));
        for c in count.iter_mut() {
            c.iter_mut().for_each(|c| *c = 0);
        }
    }
    }
    // Non-rigid impact zones first (`soft` rounds, [`soft_rounds`]), then the rigid ones.
    if soft > 0 && !vel_only && soft_rounds(pairs, x0, x1, vel, im, soft, st, &collide) {
        finish(st, x1, vel);
        return;
    }
    impact_zones(pairs, x0, x1, vel, m, im, h, zone_rounds, st, &collide);
    if vel_only {
        st.work_zones += shadow(vel) - f_rows;
    }
    finish(st, x1, vel);
}

/// Impact zones (Bridson 2002 §7.4) after the response's rounds: the nodes of still-colliding
/// pairs, merged, each moved rigidly over the step, `zone_rounds` times or until nothing collides
/// across zones; pairs still colliding then are counted (`unresolved`, an error).
#[allow(clippy::too_many_arguments)]
fn impact_zones(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], m: &[Vec<f32>], im: &[Vec<f32>], h: f32, zone_rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>) {
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
        for nodes in zones.values_mut() {
            nodes.sort_unstable();
            st.zone_nodes += nodes.len() as u64;
            if !rigid_zone(nodes, x0, x1, vel, m, im, h) {
                st.zone_spin += 1;
            }
        }
    }
    st.unresolved += collide(x1).into_iter().filter(|&(q, _)| across(q, &zone_of, &mut parent)).count() as u64;
}

/// Non-rigid impact zones (Harmon et al. 2008, inelastic projection), `soft` rounds: every
/// pair still colliding becomes a constraint row, kept for the rounds after (so a pair
/// resolved stays resolved), and each round projects the step's end positions and the
/// velocities onto them, the smallest change by mass ([`soft_project`]). The zones are the
/// rows' connected nodes; they deform, they are not frozen. What is left after these rounds
/// goes to the rigid zones ([`resolve`]). Returns true when nothing collides any more.
#[allow(clippy::too_many_arguments)]
fn soft_rounds(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], soft: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>) -> bool {
    let pos = |x: &[Vec<Vec3>], k: &[(usize, u32); 4]| k.map(|(v, i)| x[v][i as usize]);
    let mut rows: Vec<Row> = Vec::new();
    let mut held: std::collections::HashMap<usize, usize> = Default::default();
    for _ in 0..soft {
        let hit = collide(x1);
        if hit.is_empty() {
            return true;
        }
        st.soft_rounds += 1;
        for &(q, t) in &hit {
            let pr = &pairs[q];
            let (a, b) = (pos(x0, &pr.k), pos(x1, &pr.k));
            let at = [0, 1, 2, 3].map(|j| a[j] + (b[j] - a[j]) * t);
            let (w, gv) = closest(pr.nn, at);
            let n = gv.normalize_or_zero();
            if n == Vec3::ZERO {
                continue;
            }
            // (A pair hit again: its row re-linearised at the new impact.)
            if let Some(&r) = held.get(&q) {
                (rows[r].w, rows[r].n, rows[r].at) = (w, n, at);
            } else {
                held.insert(q, rows.len());
                rows.push(Row { k: pr.k, w, n, at });
            }
        }
        st.soft_rows += rows.len() as u64;
        soft_project(&rows, x1, vel, im);
    }
    st.soft_left += collide(x1).len() as u64;
    false
}

/// A non-rigid zone's constraint: a pair's nodes, its closest points' signed weights and
/// normal at its impact, and its nodes' positions there.
struct Row {
    k: [(usize, u32); 4],
    w: [f32; 4],
    n: Vec3,
    at: [Vec3; 4],
}

/// The inelastic projection of the rows (Harmon et al. 2008): the step's end positions moved
/// the least by mass so that no row's pair is closer along its normal than at its impact
/// (Σⱼ wⱼ n·(x1ⱼ − atⱼ) ≥ 0), and the velocities so that none closes (Σⱼ wⱼ n·vⱼ ≥ 0). Each
/// row's push is equal and opposite (its weights sum to zero) along the gap's direction, so
/// momentum and angular momentum are kept; the velocities' set is a cone through zero, so
/// kinetic energy never rises (every projected Gauss-Seidel step lowers it). Rows in their
/// order, [`SOFT_SWEEPS`] sweeps, fewer once none moves; a row not met then is caught by the
/// next round's detection.
fn soft_project(rows: &[Row], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>]) {
    let mi = |k: (usize, u32)| im[k.0][k.1 as usize];
    let s: Vec<f32> = rows.iter().map(|r| (0..4).map(|j| r.w[j] * r.w[j] * mi(r.k[j])).sum()).collect();
    for part in [true, false] {
        let mut lam = vec![0.0f32; rows.len()];
        for _ in 0..std::env::var("DAT_SOFT_SWEEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(SOFT_SWEEPS) {
            let mut moved = false;
            for (q, r) in rows.iter().enumerate() {
                if s[q] <= 0.0 {
                    continue;
                }
                let c: f32 = (0..4)
                    .map(|j| {
                        let (v, i) = (r.k[j].0, r.k[j].1 as usize);
                        r.w[j] * r.n.dot(if part { x1[v][i] - r.at[j] } else { vel[v][i] })
                    })
                    .sum();
                let d = (-c / s[q]).max(-lam[q]);
                if d == 0.0 {
                    continue;
                }
                lam[q] += d;
                moved = true;
                for j in 0..4 {
                    let (v, i) = (r.k[j].0, r.k[j].1 as usize);
                    let push = r.n * (r.w[j] * mi(r.k[j]) * d);
                    if part {
                        x1[v][i] += push;
                    } else {
                        vel[v][i] += push;
                    }
                }
            }
            if !moved {
                break;
            }
        }
    }
}

/// Projected Gauss-Seidel sweeps of a non-rigid zone's projection.
const SOFT_SWEEPS: u32 = 32;

/// sin and cos of `t` (|t| ≤ 1) by their Taylor series to the t¹¹ and t¹⁰ terms, Horner in a
/// fixed order: below 2⁻²³ relative error there, and the same bits on the CPU and the kernels
/// (the hardware's sin and cos differ between GPUs).
pub(super) fn sin_cos(t: f32) -> (f32, f32) {
    let t2 = t * t;
    let s = t * (1.0 + t2 * (-1.0 / 6.0 + t2 * (1.0 / 120.0 + t2 * (-1.0 / 5040.0 + t2 * (1.0 / 362_880.0 + t2 * (-1.0 / 39_916_800.0))))));
    let c = 1.0 + t2 * (-0.5 + t2 * (1.0 / 24.0 + t2 * (-1.0 / 720.0 + t2 * (1.0 / 40_320.0 + t2 * (-1.0 / 3_628_800.0)))));
    (s, c)
}

/// The inverse of the symmetric 3×3 matrix `m` (rows) by its adjugate, or None when singular.
pub(super) fn inverse3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() <= 1e-12 {
        return None;
    }
    let r = 1.0 / det;
    let c10 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    Some([[c00 * r, c10 * r, c20 * r], [c01 * r, c11 * r, c21 * r], [c02 * r, c12 * r, c22 * r]])
}

/// One impact zone moved as a rigid body over the step (Bridson 2002 §7.4): its centre of mass
/// at the start, its momentum and angular momentum from the nodes' velocities (a projection
/// onto rigid motion: kinetic energy cannot rise), each node's end the exact rotation
/// (Rodrigues) and translation of its start. f32 sums over the nodes in ascending order, as
/// the kernel's. An immovable node pins the zone where it started. Returns false when the zone
/// would turn more than 1 rad in the step (an error, counted; the zone is not moved).
fn rigid_zone(nodes: &[(usize, u32)], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], m: &[Vec<f32>], im: &[Vec<f32>], h: f32) -> bool {
    use engine_schema::feature_query::{cross3, dot3};
    let at = |k: &(usize, u32)| x0[k.0][k.1 as usize];
    let ms = |k: &(usize, u32)| m[k.0][k.1 as usize];
    let vs = |vel: &[Vec<Vec3>], k: &(usize, u32)| vel[k.0][k.1 as usize];
    if nodes.iter().any(|k| im[k.0][k.1 as usize] == 0.0) {
        for k in nodes {
            vel[k.0][k.1 as usize] = Vec3::ZERO;
            x1[k.0][k.1 as usize] = at(k);
        }
        return true;
    }
    let (mut mass, mut cm, mut p) = (0.0f32, Vec3::ZERO, Vec3::ZERO);
    for k in nodes {
        mass += ms(k);
        cm += at(k) * ms(k);
        p += vs(vel, k) * ms(k);
    }
    let (c, vc) = (cm / mass, p / mass);
    let (mut l, mut i) = (Vec3::ZERO, [[0.0f32; 3]; 3]);
    for k in nodes {
        let (r, mm) = (at(k) - c, ms(k));
        l += cross3(r, vs(vel, k) - vc) * mm;
        let rr = dot3(r, r);
        let ra = r.to_array();
        for a in 0..3 {
            for b in 0..3 {
                i[a][b] += mm * (if a == b { rr } else { 0.0 } - ra[a] * ra[b]);
            }
        }
    }
    let w = inverse3(i).map_or(Vec3::ZERO, |n| {
        let row = |a: usize| n[a][0] * l.x + n[a][1] * l.y + n[a][2] * l.z;
        Vec3::new(row(0), row(1), row(2))
    });
    let wl = dot3(w, w).sqrt();
    let ang = wl * h;
    if ang > 1.0 {
        return false;
    }
    let (sn, cs) = sin_cos(ang);
    let axis = if wl > 0.0 { w / wl } else { Vec3::ZERO };
    for k in nodes {
        let r = at(k) - c;
        let rot = r * cs + cross3(axis, r) * sn + axis * (dot3(axis, r) * (1.0 - cs));
        x1[k.0][k.1 as usize] = c + vc * h + rot;
        vel[k.0][k.1 as usize] = vc + cross3(w, r);
    }
    true
}

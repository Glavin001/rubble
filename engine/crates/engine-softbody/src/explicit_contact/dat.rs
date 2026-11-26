//! Continuous contact between vehicles: Planar Divide and Truncate (Chen, Hsu, Ayman and
//! Macklin, SIGGRAPH 2026, arXiv:2604.15513; `docs/continuous-contact-design.md`). A CPU
//! prototype, car against car, on the collision triangles (the node mesh).
//!
//! Each step, after every vehicle's node update and the work guard: every point-triangle
//! and edge-edge pair between two vehicles closer than the query radius `r_q` at the
//! step's start gets a plane square to its closest-point segment, placed so that each
//! side's room is proportional to its own approach this step (Newton's
//! `place_dat_division_plane`). Each node then takes the smallest fraction of its step
//! that keeps it on its side of all its planes, by `gamma` short of each (`t = γ·s0/(s0 −
//! s1)` where the end distance `s1` is below zero), and keeps that fraction of its step
//! and of its velocity. A primitive is the convex hull of its nodes and each node's
//! distance to its fixed plane is linear in time, so no listed pair can meet during the
//! step: the surfaces never cross, from a start where they do not. The fraction is a
//! minimum over pairs, the same in any order (bit exact). It only takes energy away
//! (the kinetic energy of a truncated node scales by t²), counted in [`DatStats`].
//!
//! Pairs farther than `r_q` at the start are not listed; they cannot meet while no node
//! moves more than `0.5·γ·r_q` a step. A node that does is counted (`over_step`), never
//! clamped: the step is then not covered, and the count says so.

use super::ExplicitBeamsWorld;
use crate::explicit_beams::{ExplicitBeams, NODE_IMMOVABLE, STEP};
use glam::Vec3;
use std::collections::HashMap;

/// The continuous mode's setting and counters (on [`ExplicitBeamsWorld::continuous`]).
#[derive(Clone, Debug)]
pub struct Dat {
    /// Query radius (m): pairs closer than this at a step's start get a plane.
    pub r_q: f32,
    /// The fraction of the room to its plane a node may use (Newton's γ_r, 0.85).
    pub gamma: f32,
    /// Pairs on the contact surface (`DAT_SURFACE=paint`: each node pushed out by its
    /// offset as at the step's start; a measurement: the offsets' directions jump between
    /// steps as panels crumple, and the node meshes then cross). Off (default): the
    /// collision triangles, each pair's thickness its nodes' offsets along its normal.
    pub paint: bool,
    /// Apply the truncation (`DAT_ENFORCE=1`, a measurement: it scales a node's step and
    /// speed, so it neither conserves momentum nor is physics). Off (default): the planes
    /// only detect, and `truncated` counts the node steps that would have crossed one.
    pub enforce: bool,
    /// The CCD response's impulse rounds and impact-zone rounds (`DAT_CCD`, `DAT_ZONES`; 0:
    /// no response), and the smallest gap it keeps (m, `DAT_MIN_GAP`; a pair's thickness
    /// when larger).
    pub ccd_rounds: u32,
    pub zone_rounds: u32,
    pub min_gap: f32,
    /// A car against itself too (`DAT_SELF=0`: only car against car).
    pub self_contact: bool,
    /// A car's own clearance (m, `DAT_SELF_GAP`): its own pairs closing to it close no
    /// further (the thickness of car-against-car pairs is their paint).
    pub self_gap: f32,
    /// The split correction's share of a paint overlap removed a step (0: none).
    pub beta: f32,
    /// Rounds of the speculative inelastic contact before the truncation (0: none).
    pub iters: u32,
    /// `DAT_WHO=1` (a measurement): per (vehicle, node), the steps it moved past the cover
    /// and the steps it was frozen.
    pub who: Option<HashMap<(usize, u32), [u32; 2]>>,
    /// Per vehicle, its own pairs closer than `self_gap` at its first step (neighbours across
    /// a seam, as self contact's spawn exclusion, spec §10.1): left out of both passes.
    pub rest_excl: Vec<Option<std::collections::HashSet<[u32; 4]>>>,
    /// The last step's counts and every step's, summed.
    pub last: DatStats,
    pub total: DatStats,
}

/// What the truncation did (the mode's health: how often it binds and what it removes).
#[derive(Clone, Copy, Debug, Default)]
pub struct DatStats {
    pub steps: u64,
    /// Point-triangle and edge-edge pairs given a plane.
    pub pairs_vt: u64,
    pub pairs_ee: u64,
    /// Of those, the pairs whose sides could meet this step (given a plane), and those the
    /// CCD response tested.
    pub pairs_dat: u64,
    pub pairs_ccd: u64,
    /// The CCD response's counts.
    pub ccd: super::ccd::CcdStats,
    /// Nodes whose step was cut (t < 1), and frozen for want of a separating plane (t = 0:
    /// a pair touching or crossed at the start).
    pub truncated: u64,
    pub uncertified: u64,
    /// Of those: the closest points within 2ε (touching), and the gap (m) of every listed
    /// pair at the start, smallest.
    pub touching: u64,
    pub min_gap: f32,
    /// Nodes that moved more than `0.5·γ·r_q` (the step not covered for unlisted pairs).
    pub over_step: u64,
    /// The smallest fraction taken, the kinetic energy (J) and momentum (N·s, summed
    /// magnitudes) the cuts removed.
    pub min_t: f32,
    pub ke_removed: f64,
    pub p_removed: f64,
    /// The kinetic energy change (J) of the speculative contact (≤ 0: it only removes approach).
    pub spec_ke: f64,
}

impl DatStats {
    fn add(&mut self, o: &DatStats) {
        self.steps += o.steps;
        self.pairs_vt += o.pairs_vt;
        self.pairs_ee += o.pairs_ee;
        self.pairs_dat += o.pairs_dat;
        self.pairs_ccd += o.pairs_ccd;
        self.ccd.hits += o.ccd.hits;
        self.ccd.rounds += o.ccd.rounds;
        self.ccd.zones += o.ccd.zones;
        self.ccd.zone_nodes += o.ccd.zone_nodes;
        self.ccd.unresolved += o.ccd.unresolved;
        self.ccd.ke += o.ccd.ke;
        self.truncated += o.truncated;
        self.uncertified += o.uncertified;
        self.touching += o.touching;
        self.min_gap = self.min_gap.min(o.min_gap);
        self.over_step += o.over_step;
        self.min_t = self.min_t.min(o.min_t);
        self.ke_removed += o.ke_removed;
        self.p_removed += o.p_removed;
        self.spec_ke += o.spec_ke;
    }
}

impl Dat {
    /// `CONTACT_MODE=continuous` turns it on (`DAT_RQ`, `DAT_GAMMA` its settings); None
    /// otherwise (discrete contact, as before).
    pub fn from_env() -> Option<Self> {
        let on = std::env::var("CONTACT_MODE").is_ok_and(|v| v == "continuous");
        let f = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let paint = std::env::var("DAT_SURFACE").is_ok_and(|v| v == "paint");
        let self_contact = std::env::var("DAT_SELF").map_or(true, |v| v != "0");
        let enforce = std::env::var("DAT_ENFORCE").is_ok_and(|v| v == "1");
        on.then(|| Self { iters: f("DAT_ITERS", 8.0) as u32, paint, self_contact, enforce, ccd_rounds: f("DAT_CCD", 8.0) as u32, zone_rounds: f("DAT_ZONES", 16.0) as u32, min_gap: f("DAT_MIN_GAP", 0.0005), self_gap: f("DAT_SELF_GAP", 0.002), beta: f("DAT_BETA", 0.0), ..Self::new(f("DAT_RQ", 0.08), f("DAT_GAMMA", 0.85)) })
    }

    pub fn new(r_q: f32, gamma: f32) -> Self {
        Self { r_q, gamma, iters: 8, paint: false, self_contact: true, enforce: false, ccd_rounds: 8, zone_rounds: 16, min_gap: 0.0005, self_gap: 0.002, beta: 0.0, who: std::env::var("DAT_WHO").is_ok_and(|v| v == "1").then(HashMap::new), rest_excl: Vec::new(), last: DatStats::default(), total: DatStats { min_t: 1.0, min_gap: f32::MAX, ..Default::default() } }
    }
}

/// A vehicle's live collision surface: its triangles (not torn, not anti-clip only) and
/// their edges, each once.
pub(crate) struct Surface {
    pub tris: Vec<[u32; 3]>,
    pub edges: Vec<[u32; 2]>,
    pub nodes: Vec<u32>,
}

pub(crate) fn surface(v: &ExplicitBeams) -> Surface {
    let mut tris = Vec::new();
    for (t, tri) in v.triangles.iter().enumerate() {
        let dead = v.tri_dead.get(t).copied().unwrap_or(false) || v.tri_anti_only(t);
        if !dead && !tri.nodes.iter().any(|&i| v.is_retired(i as usize)) {
            tris.push(tri.nodes);
        }
    }
    let mut edges: Vec<[u32; 2]> = tris
        .iter()
        .flat_map(|t| [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]])
        .map(|[a, b]| [a.min(b), a.max(b)])
        .collect();
    edges.sort_unstable();
    edges.dedup();
    let mut nodes: Vec<u32> = tris.iter().flatten().copied().collect();
    nodes.sort_unstable();
    nodes.dedup();
    Surface { tris, edges, nodes }
}

/// A uniform grid of primitives' boxes (cell `h`), each primitive in every cell its box
/// grown by `grow` touches.
pub(crate) struct Grid {
    h: f32,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl Grid {
    pub fn new(h: f32) -> Self {
        Self { h, cells: HashMap::new() }
    }
    fn cell(&self, p: Vec3) -> (i32, i32, i32) {
        let q = (p / self.h).floor();
        (q.x as i32, q.y as i32, q.z as i32)
    }
    pub fn insert(&mut self, id: u32, lo: Vec3, hi: Vec3) {
        let (a, b) = (self.cell(lo), self.cell(hi));
        for x in a.0..=b.0 {
            for y in a.1..=b.1 {
                for z in a.2..=b.2 {
                    self.cells.entry((x, y, z)).or_default().push(id);
                }
            }
        }
    }
    /// Every primitive in the cells a box touches, each once, ascending.
    pub fn query(&self, lo: Vec3, hi: Vec3, out: &mut Vec<u32>) {
        out.clear();
        let (a, b) = (self.cell(lo), self.cell(hi));
        for x in a.0..=b.0 {
            for y in a.1..=b.1 {
                for z in a.2..=b.2 {
                    if let Some(c) = self.cells.get(&(x, y, z)) {
                        out.extend_from_slice(c);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }
}

/// The closest point to `p` on triangle `a b c` (Ericson, Real-Time Collision Detection
/// §5.1.5).
pub(crate) fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let den = 1.0 / (va + vb + vc);
    a + ab * (vb * den) + ac * (vc * den)
}

/// The closest points of segments `p1 q1` and `p2 q2` (Ericson §5.1.9).
pub(crate) fn closest_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    const EPS: f32 = 1e-12;
    let (s, t);
    if a <= EPS && e <= EPS {
        return (p1, p2);
    }
    if a <= EPS {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= EPS {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let den = a * e - b * b;
            let mut s0 = if den > 0.0 { ((b * f - c * e) / den).clamp(0.0, 1.0) } else { 0.0 };
            let mut t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t0 = 0.0;
                s0 = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t0 = 1.0;
                s0 = ((b - c) / a).clamp(0.0, 1.0);
            }
            s = s0;
            t = t0;
        }
    }
    (p1 + d1 * s, p2 + d2 * t)
}

/// A listed pair: its nodes (vehicle, node), the first `nn` on the negative side, with the
/// closest points' weights; the unit normal from the negative side's closest point `cn`
/// to the positive side's at the start, the gap `g` and the paint thickness `d` (the two
/// sides' contact offsets at their closest points: where the drawn surfaces meet).
struct Rec {
    k: [(usize, u32); 4],
    w: [f32; 4],
    nn: usize,
    len: usize,
    n: Vec3,
    cn: Vec3,
    g: f32,
    d: f32,
}

impl Rec {
    fn new(neg: &[(usize, u32)], wn: &[f32], pos: &[(usize, u32)], wp: &[f32], cn: Vec3, cp: Vec3, off: &[Vec<Vec3>]) -> Self {
        let (mut k, mut w) = ([(0, 0); 4], [0.0; 4]);
        for (j, (&x, &y)) in neg.iter().chain(pos).zip(wn.iter().chain(wp)).enumerate() {
            k[j] = x;
            w[j] = y;
        }
        let len = neg.len() + pos.len();
        let g = (cp - cn).length();
        let n = (cp - cn) / g.max(f32::MIN_POSITIVE);
        // Each side's paint toward the other along the normal: its nodes' offsets projected.
        let sign = |j: usize| if j < neg.len() { 1.0 } else { -1.0 };
        let d = (0..len).map(|j| w[j] * (sign(j) * off[k[j].0].get(k[j].1 as usize).map_or(0.0, |o| o.dot(n))).max(0.0)).sum();
        Self { k, w, nn: neg.len(), len, n, cn, g, d }
    }
    fn sign(&self, j: usize) -> f32 {
        if j < self.nn { -1.0 } else { 1.0 }
    }
}

/// The speculative inelastic contact: Jacobi with mass splitting over the pairs, `iters`
/// rounds, on velocities. Each pair's closing speed along its normal is cut to what
/// brings its gap to its paint thickness by the step's end (to zero closing speed when it
/// is already closer): an inelastic impulse by the nodes' inverse masses, so momentum
/// passes to the struck side. It removes approach and never adds separation. Each change
/// of velocity moves the node by h times it (as the node update would have).
fn speculative(recs: &[&Rec], v: &mut [Vec<Vec3>], im: &[Vec<f32>], iters: u32) {
    let h = STEP;
    let mut acc: Vec<Vec<(Vec3, u32)>> = v.iter().map(|x| vec![(Vec3::ZERO, 0); x.len()]).collect();
    for _ in 0..iters {
        let mut any = false;
        for &r in recs {
            let rel: f32 = (0..r.len).map(|j| r.sign(j) * r.w[j] * v[r.k[j].0][r.k[j].1 as usize].dot(r.n)).sum();
            let c = rel + (r.g - r.d).max(0.0) / h;
            if c >= 0.0 {
                continue;
            }
            let wsum: f32 = (0..r.len).map(|j| r.w[j] * r.w[j] * im[r.k[j].0][r.k[j].1 as usize]).sum();
            if wsum <= 0.0 {
                continue;
            }
            let l = -c / wsum;
            for j in 0..r.len {
                let (q, i) = r.k[j];
                let m = im[q][i as usize];
                if m > 0.0 && r.w[j] > 0.0 {
                    let e = &mut acc[q][i as usize];
                    e.0 += r.n * (r.sign(j) * r.w[j] * m * l);
                    e.1 += 1;
                    any = true;
                }
            }
        }
        if !any {
            break;
        }
        for (x, a) in v.iter_mut().zip(acc.iter_mut()) {
            for (p, e) in x.iter_mut().zip(a.iter_mut()) {
                if e.1 > 0 {
                    *p += e.0 / e.1 as f32;
                    *e = (Vec3::ZERO, 0);
                }
            }
        }
    }
}

/// The split correction (Box2D's and Bullet's split impulse): pairs whose paint overlaps
/// at the step's end (gap below their thickness) are moved apart by `beta` of the overlap,
/// positions only, Jacobi with mass splitting, `iters` rounds. No velocity changes, so no
/// kinetic energy is added; the truncation that follows keeps the node mesh uncrossed.
fn separate(recs: &[&Rec], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], im: &[Vec<f32>], iters: u32, beta: f32) {
    let mut acc: Vec<Vec<(Vec3, u32)>> = x1.iter().map(|x| vec![(Vec3::ZERO, 0); x.len()]).collect();
    for _ in 0..iters {
        let mut any = false;
        for &r in recs {
            let dx = |j: usize| x1[r.k[j].0][r.k[j].1 as usize] - x0[r.k[j].0][r.k[j].1 as usize];
            let gap = r.g + (0..r.len).map(|j| r.sign(j) * r.w[j] * dx(j).dot(r.n)).sum::<f32>();
            let c = gap - r.d;
            if c >= 0.0 {
                continue;
            }
            let wsum: f32 = (0..r.len).map(|j| r.w[j] * r.w[j] * im[r.k[j].0][r.k[j].1 as usize]).sum();
            if wsum <= 0.0 {
                continue;
            }
            let l = -c * beta / wsum;
            for j in 0..r.len {
                let (q, i) = r.k[j];
                let m = im[q][i as usize];
                if m > 0.0 && r.w[j] > 0.0 {
                    let e = &mut acc[q][i as usize];
                    e.0 += r.n * (r.sign(j) * r.w[j] * m * l);
                    e.1 += 1;
                    any = true;
                }
            }
        }
        if !any {
            break;
        }
        for (x, a) in x1.iter_mut().zip(acc.iter_mut()) {
            for (p, e) in x.iter_mut().zip(a.iter_mut()) {
                if e.1 > 0 {
                    *p += e.0 / e.1 as f32;
                    *e = (Vec3::ZERO, 0);
                }
            }
        }
    }
}

struct Pass<'a> {
    x0: &'a [Vec<Vec3>],
    x1: &'a [Vec<Vec3>],
    t: &'a mut [Vec<f32>],
    gamma: f32,
    stats: &'a mut DatStats,
}

impl Pass<'_> {
    /// One pair's plane, placed by each side's approach, and each node's fraction.
    fn pair(&mut self, r: &Rec) {
        let (g, n, cn) = (r.g, r.n, r.cn);
        let ks = &r.k[..r.len];
        let scale = cn.abs().max_element().max(1.0) + g;
        let eps = 4.0 * f32::EPSILON * scale;
        self.stats.min_gap = self.stats.min_gap.min(g);
        if g <= 2.0 * eps {
            self.stats.touching += 1;
            self.freeze(r);
            return;
        }
        let dx = |&(v, i): &(usize, u32)| self.x1[v][i as usize] - self.x0[v][i as usize];
        let dn: Vec<f32> = ks.iter().map(|k| dx(k).dot(n)).collect();
        // The plane moves with the pair's common motion along the normal (the mean of the two
        // closest points' normal steps): only relative approach takes room. Each node's
        // distance to the moving plane stays linear in time, so the guarantee holds.
        let side_mean = |js: std::ops::Range<usize>| js.map(|j| r.w[j] * dn[j]).sum::<f32>();
        let mut u = 0.5 * (side_mean(0..r.nn) + side_mean(r.nn..r.len));
        let dpos = (r.nn..r.len).map(|j| u - dn[j]).fold(0.0f32, f32::max);
        let dneg = (0..r.nn).map(|j| dn[j] - u).fold(0.0f32, f32::max);
        let lambda = if dpos + dneg > 0.0 { (dneg / (dpos + dneg)).clamp(0.05, 0.95) } else { 0.5 };
        let p = cn + n * (lambda * g);
        // Certify the plane: every node strictly on its side at the start.
        let s0: Vec<f32> = (0..r.len).map(|j| r.sign(j) * (self.x0[ks[j].0][ks[j].1 as usize] - p).dot(n)).collect();
        if s0.iter().any(|&a| a <= eps) {
            self.freeze(r);
            return;
        }
        // The plane moves no further than every node could follow (by γ of its room), so a
        // node held still stays on its side.
        let lo = (0..r.nn).map(|j| s0[j]).fold(f32::MAX, f32::min);
        let hi = (r.nn..r.len).map(|j| s0[j]).fold(f32::MAX, f32::min);
        u = u.clamp(-self.gamma * lo, self.gamma * hi);
        for j in 0..r.len {
            let k = ks[j];
            let a = s0[j] - r.sign(j) * u;
            let b = r.sign(j) * dn[j];
            if a + b < 0.0 {
                let t = &mut self.t[k.0][k.1 as usize];
                *t = t.min(self.gamma * a / -b);
            }
        }
    }

    /// No plane can be certified (the gap within rounding): each node that moves toward
    /// the other side along the normal stays where it is; one moving away keeps its step
    /// (its distance only grows), so the pair can part.
    fn freeze(&mut self, r: &Rec) {
        self.stats.uncertified += 1;
        for j in 0..r.len {
            let k = r.k[j];
            let dx = self.x1[k.0][k.1 as usize] - self.x0[k.0][k.1 as usize];
            if r.g <= 0.0 || r.sign(j) * dx.dot(r.n) < 0.0 {
                self.t[k.0][k.1 as usize] = 0.0;
            }
        }
    }
}

/// The barycentric weights of point `p` (on the triangle) in `a b c`.
pub(super) fn barycentric_pub(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f32; 3] {
    barycentric(p, a, b, c)
}

pub(super) fn along_pub(c: Vec3, p: Vec3, q: Vec3) -> f32 {
    along(c, p, q)
}

fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f32; 3] {
    let (v0, v1, v2) = (b - a, c - a, p - a);
    let (d00, d01, d11, d20, d21) = (v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1));
    let den = d00 * d11 - d01 * d01;
    if den <= 0.0 {
        return [1.0 / 3.0; 3];
    }
    let v = ((d11 * d20 - d01 * d21) / den).clamp(0.0, 1.0);
    let w = ((d00 * d21 - d01 * d20) / den).clamp(0.0, 1.0 - v);
    [1.0 - v - w, v, w]
}

/// The parameter of point `c` (on the segment) along `p q`.
fn along(c: Vec3, p: Vec3, q: Vec3) -> f32 {
    let d = q - p;
    let l = d.length_squared();
    if l <= 0.0 { 0.0 } else { ((c - p).dot(d) / l).clamp(0.0, 1.0) }
}

fn bounds(ps: impl IntoIterator<Item = Vec3>) -> (Vec3, Vec3) {
    ps.into_iter()
        .fold((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |(l, h), p| (l.min(p), h.max(p)))
}

/// A step's start: every node's World position and its offset to the paint.
pub(super) type DatStart = (Vec<Vec<Vec3>>, Vec<Vec<Vec3>>);

impl ExplicitBeamsWorld {
    /// The continuous mode's start of a step: every node's World position (None: off).
    pub(super) fn dat_start(&self) -> Option<DatStart> {
        let dat = self.continuous.as_ref()?;
        (self.contact && (self.vehicles.len() > 1 || dat.self_contact)).then(|| {
            let x = world_positions(&self.vehicles);
            // On the contact surface: each node's offset to the paint, as at the step's start
            // (held through the step, so the surface's vertices move as their nodes do).
            let o = self.vehicles.iter().zip(&x).map(|(v, x)| {
                let o = super::edge::contact_offset_vectors(v);
                if o.is_empty() { vec![Vec3::ZERO; x.len()] } else { o }
            }).collect();
            (x, o)
        })
    }

    /// After the node update and the work guard: the planes and the truncation.
    pub(super) fn dat_truncate(&mut self, start: Option<DatStart>) {
        let (Some((mut x0, mut ov)), Some(dat)) = (start, self.continuous.as_mut()) else {
            return;
        };
        let off: Vec<Vec<Vec3>> = if dat.paint { vec![Vec::new(); x0.len()] } else { ov.clone() };
        if !dat.paint {
            ov.iter_mut().for_each(|o| o.iter_mut().for_each(|x| *x = Vec3::ZERO));
        }
        let mut x1p = world_positions(&self.vehicles);
        for v in 0..x0.len() {
            for i in 0..x0[v].len() {
                x0[v][i] += ov[v][i];
                x1p[v][i] += ov[v][i];
            }
        }
        let mut x1 = x1p.clone();
        let im: Vec<Vec<f32>> = self
            .vehicles
            .iter()
            .map(|v| (0..v.nodes.len()).map(|i| if v.nodes[i].has(NODE_IMMOVABLE) || v.is_retired(i) { 0.0 } else { 1.0 / v.nodes[i].mass }).collect())
            .collect();
        let mut recs: Vec<Rec> = Vec::new();
        // Pairs are listed within r_q of their paint (the gap less the two sides' offsets).
        let mo: Vec<f32> = off.iter().map(|o| o.iter().map(|v| v.length()).fold(0.0f32, f32::max)).collect();
        let mut t: Vec<Vec<f32>> = x0.iter().map(|x| vec![1.0; x.len()]).collect();
        let mut st = DatStats { steps: 1, min_t: 1.0, min_gap: f32::MAX, ..Default::default() };
        let (r_q, cap) = (dat.r_q, 0.5 * dat.gamma * dat.r_q);
        let surf: Vec<Surface> = self.vehicles.iter().map(surface).collect();
        for (v, s) in surf.iter().enumerate() {
            for &i in s.nodes.iter().filter(|&&i| (x1[v][i as usize] - x0[v][i as usize]).length() > cap) {
                st.over_step += 1;
                if let Some(w) = dat.who.as_mut() {
                    w.entry((v, i)).or_default()[0] += 1;
                }
            }
        }
        let boxes: Vec<(Vec3, Vec3)> =
            surf.iter().enumerate().map(|(v, s)| bounds(s.nodes.iter().map(|&i| x0[v][i as usize]))).collect();
        let mut found = Vec::new();
        let nv = self.vehicles.len();
        let none: Vec<Vec<Vec3>> = vec![Vec::new(); nv];
        if dat.self_contact {
            dat.rest_excl.resize(nv, None);
            for v in 0..nv {
                if dat.rest_excl[v].is_none() {
                    let near = close_pairs(&x0[v], &surf[v], dat.self_gap);
                    dat.rest_excl[v] = Some(near.into_iter().map(|p| p.0).collect());
                }
            }
        }
        let excl = |v: usize, k: [u32; 4]| dat.rest_excl.get(v).and_then(|e| e.as_ref()).is_some_and(|e| e.contains(&k));
        for a in 0..nv {
            for b in 0..nv {
                let ((alo, ahi), (blo, bhi)) = (boxes[a], boxes[b]);
                // A car against itself: its own pairs sharing no node, at thickness 0 (its
                // collision surface never passes through itself).
                let own = a == b;
                let (mo_a, mo_b) = if own { (0.0, 0.0) } else { (mo[a], mo[b]) };
                let off = if own { &none } else { &off };
                let r = Vec3::splat(r_q + mo_a + mo_b);
                if (own && !dat.self_contact) || (alo - r).cmpgt(bhi).any() || (blo - r).cmpgt(ahi).any() {
                    continue;
                }
                // a's nodes against b's triangles (both orders run: every node of both).
                let (sa, sb) = (&surf[a], &surf[b]);
                let mut grid = Grid::new((r_q + mo_a + mo_b).max(0.02));
                for (k, tri) in sb.tris.iter().enumerate() {
                    let (lo, hi) = bounds(tri.map(|i| x0[b][i as usize]));
                    grid.insert(k as u32, lo - r, hi + r);
                }
                for &i in &sa.nodes {
                    let p = x0[a][i as usize];
                    grid.query(p, p, &mut found);
                    for &k in &found {
                        let tri = sb.tris[k as usize];
                        if own && (tri.contains(&i) || excl(a, [i, tri[0], tri[1], tri[2]])) {
                            continue;
                        }
                        let [q0, q1, q2] = tri.map(|j| x0[b][j as usize]);
                        let c = closest_on_triangle(p, q0, q1, q2);
                        if (p - c).length_squared() < (r_q + mo_a + mo_b).powi(2) {
                            let neg = tri.map(|j| (b, j));
                            let mut rec = Rec::new(&neg, &barycentric(c, q0, q1, q2), &[(a, i)], &[1.0], c, p, &off);
                            if own {
                                rec.d = dat.self_gap;
                            }
                            if rec.g < rec.d + r_q {
                                st.pairs_vt += 1;
                                recs.push(rec);
                            }
                        }
                    }
                }
                // Edge pairs once (a < b).
                if a > b {
                    continue;
                }
                let mut grid = Grid::new((r_q + mo_a + mo_b).max(0.02));
                for (k, e) in sb.edges.iter().enumerate() {
                    let (lo, hi) = bounds(e.map(|i| x0[b][i as usize]));
                    grid.insert(k as u32, lo - r, hi + r);
                }
                for (ke, e) in sa.edges.iter().enumerate() {
                    let [p1, q1] = e.map(|i| x0[a][i as usize]);
                    let (lo, hi) = bounds([p1, q1]);
                    grid.query(lo, hi, &mut found);
                    for &k in &found {
                        let f = sb.edges[k as usize];
                        if own && (k as usize <= ke || f.iter().any(|x| e.contains(x)) || excl(a, [e[0], e[1], f[0], f[1]])) {
                            continue;
                        }
                        let [p2, q2] = f.map(|j| x0[b][j as usize]);
                        let (c1, c2) = closest_segments(p1, q1, p2, q2);
                        if (c2 - c1).length_squared() < (r_q + mo_a + mo_b).powi(2) {
                            let (s1, s2) = (along(c1, p1, q1), along(c2, p2, q2));
                            let mut rec = Rec::new(&e.map(|i| (a, i)), &[1.0 - s1, s1], &f.map(|j| (b, j)), &[1.0 - s2, s2], c1, c2, &off);
                            if own {
                                rec.d = dat.self_gap;
                            }
                            if rec.g < rec.d + r_q {
                                st.pairs_ee += 1;
                                recs.push(rec);
                            }
                        }
                    }
                }
            }
        }
        // How far each node can close on anything this step (its step, or its speed times h,
        // whichever is longer: the speculative pass works on speeds). A pair whose two sides
        // together cannot cover its gap cannot meet and is left out of either pass.
        let reach = |x1: &[Vec<Vec3>], vel: &[Vec<Vec3>]| -> Vec<Vec<f32>> {
            (0..x0.len())
                .map(|v| (0..x0[v].len()).map(|i| (x1[v][i] - x0[v][i]).length().max(vel[v][i].length() * STEP)).collect())
                .collect()
        };
        let side = |r: &Rec, q: &[Vec<f32>], neg: bool| {
            let js = if neg { 0..r.nn } else { r.nn..r.len };
            js.map(|j| q[r.k[j].0][r.k[j].1 as usize]).fold(0.0f32, f32::max)
        };
        let v0: Vec<Vec<Vec3>> = self.vehicles.iter().map(|v| v.nodes.iter().map(|n| n.vel).collect()).collect();
        let mut shift: Vec<Vec<Vec3>> = x0.iter().map(|x| vec![Vec3::ZERO; x.len()]).collect();
        if dat.iters > 0 {
            let q = reach(&x1, &v0);
            let near: Vec<&Rec> = recs.iter().filter(|r| r.g - r.d < side(r, &q, true) + side(r, &q, false)).collect();
            let mut v1 = v0.clone();
            speculative(&near, &mut v1, &im, dat.iters);
            for (v, veh) in v1.iter().enumerate() {
                for (i, &u) in veh.iter().enumerate() {
                    x1[v][i] += (u - v0[v][i]) * STEP;
                }
            }
            if dat.beta > 0.0 {
                let over: Vec<&Rec> = near.iter().copied().filter(|r| r.g < r.d + side(r, &q, true) + side(r, &q, false)).collect();
                let before = x1.clone();
                separate(&over, &x0, &mut x1, &im, dat.iters, dat.beta);
                // (positions only: the velocity change below must not see it)
                for v in 0..x1.len() {
                    for i in 0..x1[v].len() {
                        shift[v][i] = x1[v][i] - before[v][i];
                    }
                }
            }
        }
        let zero: Vec<Vec<Vec3>> = x0.iter().map(|x| vec![Vec3::ZERO; x.len()]).collect();
        let mut vfinal: Option<Vec<Vec<Vec3>>> = None;
        if dat.ccd_rounds > 0 {
            // The physical guarantee: CCD with inelastic impulses, then impact zones.
            let q = reach(&x1, &zero);
            let pairs: Vec<super::ccd::CcdPair> = recs
                .iter()
                .filter(|r| r.len == 4 && r.g - r.d < 2.0 * (side(r, &q, true) + side(r, &q, false)))
                .map(|r| super::ccd::CcdPair { k: r.k, nn: r.nn, xi: r.d.max(dat.min_gap) })
                .collect();
            st.pairs_ccd = pairs.len() as u64;
            let m: Vec<Vec<f32>> = self.vehicles.iter().map(|v| v.nodes.iter().map(|n| n.mass).collect()).collect();
            let mut vs: Vec<Vec<Vec3>> = (0..x1.len())
                .map(|v| (0..x1[v].len()).map(|i| v0[v][i] + (x1[v][i] - x1p[v][i] - shift[v][i]) / STEP).collect())
                .collect();
            super::ccd::resolve(&pairs, &x0, &mut x1, &mut vs, &m, &im, STEP, dat.ccd_rounds, dat.zone_rounds, &mut st.ccd);
            vfinal = Some(vs);
        }
        let q = reach(&x1, &zero);
        recs.retain(|r| r.g < side(r, &q, true) + side(r, &q, false));
        st.pairs_dat = recs.len() as u64;
        let mut pass = Pass { x0: &x0, x1: &x1, t: &mut t, gamma: dat.gamma, stats: &mut st };
        for r in &recs {
            pass.pair(r);
        }
        for (v, veh) in self.vehicles.iter_mut().enumerate() {
            let o = veh.origin;
            for (i, n) in veh.nodes.iter_mut().enumerate() {
                let c = x1[v][i] - x1p[v][i] - shift[v][i];
                if x1[v][i] != x1p[v][i] || vfinal.as_ref().is_some_and(|f| f[v][i] != n.vel) {
                    // (with the CCD response its velocities, which it changes apart from the
                    // positions; without, the position change's speed)
                    let v1 = vfinal.as_ref().map_or(n.vel + c / STEP, |f| f[v][i]);
                    st.spec_ke += 0.5 * n.mass as f64 * (v1.length_squared() as f64 - n.vel.length_squared() as f64);
                    n.vel = v1;
                    n.pos = x1[v][i] - ov[v][i] - o;
                }
            }
            for (i, &f) in t[v].iter().enumerate() {
                let n = &mut veh.nodes[i];
                if f >= 1.0 || n.has(NODE_IMMOVABLE) {
                    continue;
                }
                st.truncated += 1;
                st.min_t = st.min_t.min(f);
                if f == 0.0 {
                    if let Some(w) = dat.who.as_mut() {
                        w.entry((v, i as u32)).or_default()[1] += 1;
                    }
                }
                // (What the cut would remove; with `enforce` off, a detector only: nothing moves.)
                st.ke_removed += 0.5 * n.mass as f64 * n.vel.length_squared() as f64 * (1.0 - (f as f64).powi(2));
                st.p_removed += n.mass as f64 * n.vel.length() as f64 * (1.0 - f as f64);
                if dat.enforce {
                    n.pos = x0[v][i] + (x1[v][i] - x0[v][i]) * f - ov[v][i] - o;
                    n.vel *= f;
                }
            }
        }
        dat.last = st;
        dat.total.add(&st);
    }

    /// Vehicle `v`'s own surface pairs sharing no node closer than `eps` (m) now: point
    /// against triangle (4 nodes, the point first) and edge against edge, with the gap. The
    /// continuous mode cannot certify a plane between primitives that touch.
    pub fn close_self_pairs(&self, v: usize, eps: f32) -> Vec<([u32; 4], f32)> {
        let veh = &self.vehicles[v];
        let x: Vec<Vec3> = veh.nodes.iter().map(|n| veh.origin + n.pos).collect();
        close_pairs(&x, &surface(veh), eps)
    }

    /// Each vehicle's edges crossing its own live triangles that share no node with them.
    pub fn self_crossings(&self) -> Vec<usize> {
        let x = world_positions(&self.vehicles);
        let mut found = Vec::new();
        self.vehicles
            .iter()
            .enumerate()
            .map(|(v, veh)| {
                let s = surface(veh);
                let mut grid = Grid::new(0.05);
                for (k, tri) in s.tris.iter().enumerate() {
                    let (lo, hi) = bounds(tri.map(|i| x[v][i as usize]));
                    grid.insert(k as u32, lo, hi);
                }
                let mut n = 0;
                for e in &s.edges {
                    let [p, q] = e.map(|i| x[v][i as usize]);
                    let (lo, hi) = bounds([p, q]);
                    grid.query(lo, hi, &mut found);
                    n += found
                        .iter()
                        .filter(|&&k| {
                            let t = s.tris[k as usize];
                            !t.iter().any(|j| e.contains(j)) && {
                                let [t0, t1, t2] = t.map(|j| x[v][j as usize]);
                                seg_crosses(p, q, t0, t1, t2)
                            }
                        })
                        .count();
                }
                n
            })
            .collect()
    }

    /// Edges of one vehicle crossing live triangles of another, at the current positions
    /// (the invariant's test: zero in continuous mode from a start with none).
    pub fn mesh_crossings(&self) -> usize {
        let x = world_positions(&self.vehicles);
        let surf: Vec<Surface> = self.vehicles.iter().map(surface).collect();
        let mut n = 0;
        let mut found = Vec::new();
        for a in 0..surf.len() {
            for b in 0..surf.len() {
                if a == b {
                    continue;
                }
                let mut grid = Grid::new(0.05);
                for (k, tri) in surf[b].tris.iter().enumerate() {
                    let (lo, hi) = bounds(tri.map(|i| x[b][i as usize]));
                    grid.insert(k as u32, lo, hi);
                }
                for e in &surf[a].edges {
                    let [p, q] = e.map(|i| x[a][i as usize]);
                    let (lo, hi) = bounds([p, q]);
                    grid.query(lo, hi, &mut found);
                    n += found
                        .iter()
                        .filter(|&&k| {
                            let [t0, t1, t2] = surf[b].tris[k as usize].map(|j| x[b][j as usize]);
                            seg_crosses(p, q, t0, t1, t2)
                        })
                        .count();
                }
            }
        }
        n
    }
}

/// A surface's own pairs sharing no node closer than `eps` at positions `x` (keys as the
/// passes list them: the point then its triangle; the two edges).
fn close_pairs(x: &[Vec3], s: &Surface, eps: f32) -> Vec<([u32; 4], f32)> {
    let mut out = Vec::new();
    let mut found = Vec::new();
    let mut grid = Grid::new(0.05);
    for (k, t) in s.tris.iter().enumerate() {
        let (lo, hi) = bounds(t.map(|i| x[i as usize]));
        grid.insert(k as u32, lo - Vec3::splat(eps), hi + Vec3::splat(eps));
    }
    for &i in &s.nodes {
        let p = x[i as usize];
        grid.query(p, p, &mut found);
        for &k in &found {
            let t = s.tris[k as usize];
            if t.contains(&i) {
                continue;
            }
            let g = (p - closest_on_triangle(p, x[t[0] as usize], x[t[1] as usize], x[t[2] as usize])).length();
            if g < eps {
                out.push(([i, t[0], t[1], t[2]], g));
            }
        }
    }
    let mut grid = Grid::new(0.05);
    for (k, e) in s.edges.iter().enumerate() {
        let (lo, hi) = bounds(e.map(|i| x[i as usize]));
        grid.insert(k as u32, lo - Vec3::splat(eps), hi + Vec3::splat(eps));
    }
    for (ke, e) in s.edges.iter().enumerate() {
        let (lo, hi) = bounds(e.map(|i| x[i as usize]));
        grid.query(lo, hi, &mut found);
        for &k in &found {
            let f = s.edges[k as usize];
            if k as usize <= ke || f.iter().any(|y| e.contains(y)) {
                continue;
            }
            let (c1, c2) = closest_segments(x[e[0] as usize], x[e[1] as usize], x[f[0] as usize], x[f[1] as usize]);
            let g = (c2 - c1).length();
            if g < eps {
                out.push(([e[0], e[1], f[0], f[1]], g));
            }
        }
    }
    out
}

fn world_positions(vs: &[ExplicitBeams]) -> Vec<Vec<Vec3>> {
    vs.iter().map(|v| v.nodes.iter().map(|n| v.origin + n.pos).collect()).collect()
}

/// Whether segment `p q` passes through triangle `a b c` (signed volumes in f64).
pub(crate) fn seg_crosses(p: Vec3, q: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let d = |v: Vec3| v.as_dvec3();
    let vol = |w: glam::DVec3, x: glam::DVec3, y: glam::DVec3, z: glam::DVec3| (x - w).cross(y - w).dot(z - w);
    let (p, q, a, b, c) = (d(p), d(q), d(a), d(b), d(c));
    let (sp, sq) = (vol(a, b, c, p), vol(a, b, c, q));
    if sp * sq >= 0.0 {
        return false;
    }
    let (s1, s2, s3) = (vol(p, q, a, b), vol(p, q, b, c), vol(p, q, c, a));
    (s1 > 0.0 && s2 > 0.0 && s3 > 0.0) || (s1 < 0.0 && s2 < 0.0 && s3 < 0.0)
}

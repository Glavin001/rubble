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
#[path = "dat_passes.rs"]
mod passes;
use passes::{separate, speculative};
use crate::explicit_beams::{ExplicitBeams, NODE_IMMOVABLE, STEP};
use engine_schema::feature_query as fq;
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
    /// Rounds of non-rigid impact zones before the rigid ones (0: rigid only; `DAT_SOFT`, a
    /// comparison, `ccd::resolve`).
    pub soft_zones: u32,
    /// The impact law's restitution e (Moreau: v⁺ = v_p + e·(v_p − v⁻) over the response's
    /// velocity change; [`CONTACT_RESTITUTION`]; `DAT_E` measures another, [`restitute`]).
    pub restitution: f32,
    /// Subcycled penalty contact's stiffness scale (0: off; `DAT_PENALTY`): each pair under its
    /// thickness a spring k = scale·m/h_s² (m the pair's lightest node, h_s the sub-step: the
    /// soft-constraint stiffness of explicit crash codes at their own small step); the CCD then
    /// only keeps the collision mesh from crossing (gap `min_gap`), the speculative pass is off.
    pub penalty: f32,
    pub penalty_f: Vec<Vec<Vec3>>,
    /// Penalty contact's sub-steps a step (`DAT_SUBSTEPS`, 32).
    pub substeps: u32,
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
    /// Per vehicle, whether its first step's clearance was checked ([`own_clearance`]: no
    /// own pair is excluded, so a car built closer than `self_gap` fails).
    pub checked: Vec<bool>,
    /// This step's vehicle pairs (a < b) whose boxes the listing found near (the kernel's
    /// listing test): the impact law's guard groups ([`restitute`]).
    pub near: Vec<(usize, usize)>,
    /// With `keys` Some, each step's listed pairs' four (vehicle, node) keys (a diagnostic).
    pub keys: Option<Vec<[(usize, u32); 4]>>,
    /// Each vehicle's positions at its first step.
    pub first: Vec<Vec<Vec3>>,
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
    /// Surface nodes the speculative pass moved past twice their vehicle's listing reach (the
    /// listing's bound broken: an error).
    pub reach_over: u64,
    /// The smallest fraction taken, the kinetic energy (J) and momentum (N·s, summed
    /// magnitudes) the cuts removed.
    pub min_t: f32,
    pub ke_removed: f64,
    pub p_removed: f64,
    /// The kinetic energy change (J) of the speculative contact (≤ 0: it only removes approach).
    pub spec_ke: f64,
    /// The vehicles' kinetic energy at the first step (J): the scale of what the response may
    /// remove (AGENTS.md, "Contact never stands in for metal").
    pub ke_start: f64,
    /// The restitution's guard: the steps it scaled e down to keep kinetic energy from rising.
    pub guarded: u64,
    /// The vehicles' mechanical energy changed by the pass (J, summed; `DAT_ENERGY=1`).
    pub contact_de: f64,
    /// The pass's work on the integrator's conserved energy (J, summed; `DAT_ENERGY=1`, with the
    /// step's start velocities, `DAT_RESPONSE=vel`): Σ m·Δv·(v⁻ + v⁺)/2, Δv the pass's change,
    /// v⁻ the velocity before the step's kick and v⁺ after the pass. Symplectic Euler with an
    /// impulse at the kick: a contact holding a node its beams press does none, an inelastic
    /// impact −½μv². (`contact_de` also counts the beams' energy a held node keeps.)
    pub contact_work: f64,
}

impl DatStats {
    fn add(&mut self, o: &DatStats) {
        if self.steps == 0 {
            self.ke_start = o.ke_start;
        }
        self.guarded += o.guarded;
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
        self.ccd.zone_spin += o.ccd.zone_spin;
        (self.ccd.soft_rounds, self.ccd.soft_rows, self.ccd.soft_left, self.ccd.row_iters) = (self.ccd.soft_rounds + o.ccd.soft_rounds, self.ccd.soft_rows + o.ccd.soft_rows, self.ccd.soft_left + o.ccd.soft_left, self.ccd.row_iters + o.ccd.row_iters);
        self.ccd.ke += o.ccd.ke;
        self.ccd.dp = self.ccd.dp.max(o.ccd.dp);
        self.ccd.dl = self.ccd.dl.max(o.ccd.dl);
        self.truncated += o.truncated;
        self.uncertified += o.uncertified;
        self.touching += o.touching;
        self.min_gap = self.min_gap.min(o.min_gap);
        self.over_step += o.over_step;
        self.reach_over += o.reach_over;
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
        Self { r_q, gamma, iters: std::env::var("DAT_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8), paint: false, self_contact: true, enforce: false, ccd_rounds: 8, zone_rounds: std::env::var("DAT_ZONES").ok().and_then(|v| v.parse().ok()).unwrap_or(16), penalty: std::env::var("DAT_PENALTY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0), penalty_f: Vec::new(), substeps: std::env::var("DAT_SUBSTEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(32), restitution: std::env::var("DAT_E").ok().and_then(|v| v.parse().ok()).unwrap_or(super::restitute::CONTACT_RESTITUTION), soft_zones: std::env::var("DAT_SOFT").ok().and_then(|v| v.parse().ok()).unwrap_or(0), min_gap: 0.0005, self_gap: 0.002, beta: 0.0, who: std::env::var("DAT_WHO").is_ok_and(|v| v == "1").then(HashMap::new), checked: Vec::new(), near: Vec::new(), keys: None, first: Vec::new(), last: DatStats::default(), total: DatStats { min_t: 1.0, min_gap: f32::MAX, ..Default::default() } }
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
        let dead = v.tri_torn(t) || v.tri_anti_only(t);
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

/// The closest point to `p` on triangle `a b c` (Ericson §5.1.5; the kernels' twin,
/// `engine_schema::feature_query`, so the GPU computes the same bits).
pub fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    fq::closest_on_triangle(p, a, b, c).0
}

/// The closest points of segments `p1 q1` and `p2 q2` (Ericson §5.1.9, the kernels' twin).
pub fn closest_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let (s, t) = fq::closest_seg_seg(p1, q1, p2, q2);
    (p1 + (q1 - p1) * s, p2 + (q2 - p2) * t)
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

fn bounds(ps: impl IntoIterator<Item = Vec3>) -> (Vec3, Vec3) {
    ps.into_iter()
        .fold((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |(l, h), p| (l.min(p), h.max(p)))
}

/// A step's start: every node's World position and its offset to the paint.
/// The step's start: world positions, contact offsets, and each node's force and velocity
/// before the node update (subcycled penalty contact's).
pub(super) type DatStart = (Vec<Vec<Vec3>>, Vec<Vec<Vec3>>, Vec<Vec<(Vec3, Vec3)>>);

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
            let pre = if dat.penalty > 0.0 || vel_mode() { self.vehicles.iter().map(|v| v.nodes.iter().map(|n| (n.force, n.vel)).collect()).collect() } else { Vec::new() };
            (x, o, pre)
        })
    }

    /// After the node update and the work guard: the planes and the truncation.
    pub(super) fn dat_truncate(&mut self, start: Option<DatStart>) {
        let (Some((mut x0, mut ov, pre)), Some(dat)) = (start, self.continuous.as_mut()) else {
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
        let mut t: Vec<Vec<f32>> = x0.iter().map(|x| vec![1.0; x.len()]).collect();
        let mut st = DatStats { steps: 1, min_t: 1.0, min_gap: f32::MAX, ..Default::default() };
        if dat.total.steps == 0 {
            st.ke_start = self.vehicles.iter().flat_map(|v| v.nodes.iter()).map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum();
            // (and the rigid bodies the vehicles hold as proxies: a ball fired into a car)
            st.ke_start += self.vehicles.iter().filter_map(|v| v.rigid.as_ref()).flat_map(|r| r.bodies.iter()).filter(|b| b.inv_mass > 0.0).map(|b| {
                let w = b.rot.inverse() * b.ang_vel;
                0.5 * (b.vel.length_squared() / b.inv_mass) as f64 + 0.5 * [w.x, w.y, w.z].iter().zip(b.inv_inertia.to_array()).map(|(w, i)| if i > 0.0 { (w * w / i) as f64 } else { 0.0 }).sum::<f64>()
            }).sum::<f64>();
        }
        let cap = 0.5 * dat.gamma * dat.r_q;
        let surf: Vec<Surface> = self.vehicles.iter().map(surface).collect();
        for (v, s) in surf.iter().enumerate() {
            for &i in s.nodes.iter().filter(|&&i| (x1[v][i as usize] - x0[v][i as usize]).length() > cap) {
                st.over_step += 1;
                if let Some(w) = dat.who.as_mut() {
                    w.entry((v, i)).or_default()[0] += 1;
                }
            }
        }
        // How far each vehicle's surface moves this step at most (its step or its speed times
        // h): pairs are listed within four times the two sides' (and r_q at most).
        // (Relative to the vehicle's first node: the car's own travel brings none of its pairs
        // closer; between cars the two first nodes' relative motion is added.)
        let dx = |v: usize, i: usize| x1[v][i] - x0[v][i];
        let vel = |v: usize, i: usize| self.vehicles[v].nodes[i].vel;
        let reach_v: Vec<f32> = surf
            .iter()
            .enumerate()
            .map(|(v, s)| {
                s.nodes.iter().map(|&i| (dx(v, i as usize) - dx(v, 0)).length().max((vel(v, i as usize) - vel(v, 0)).length() * STEP)).fold(0.0f32, f32::max)
            })
            .collect();
        let refs: Vec<(Vec3, Vec3)> = (0..x0.len()).map(|v| (dx(v, 0), vel(v, 0))).collect();
        let mut own_reach = vec![0.0f32; x0.len()];
        let mut recs = list_pairs(dat, &x0, &surf, &off, &reach_v, &refs, &mut own_reach, &mut st);
        // Subcycled penalty contact (`DAT_PENALTY`, `DAT_SUBSTEPS`): the nodes of the listed pairs
        // move over the step again in sub-steps with the pairs' springs ([`passes::penalty_substeps`]).
        if dat.penalty > 0.0 {
            let mut vel: Vec<Vec<Vec3>> = self.vehicles.iter().map(|v| v.nodes.iter().map(|n| n.vel).collect()).collect();
            let moved = passes::penalty_substeps(&recs, &x0, &mut x1p, &mut vel, &im, dat.penalty, dat.substeps, dat.self_gap, &self.vehicles, &pre);
            for (v, veh) in self.vehicles.iter_mut().enumerate() {
                let o = veh.origin;
                for &i in &moved[v] {
                    let n = &mut veh.nodes[i as usize];
                    n.vel = vel[v][i as usize];
                    n.pos = x1p[v][i as usize] - ov[v][i as usize] - o;
                }
            }
            x1 = x1p.clone();
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
        if dat.iters > 0 && dat.penalty == 0.0 && !vel_mode() {
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
        // The listing's bound: a surface node the speculative pass moved, from its vehicle's
        // first node, past half its own pairs' listing reach may meet a pair not listed (an
        // error, counted).
        for (v, sf) in surf.iter().enumerate() {
            let r = x1[v][0] - x0[v][0];
            st.reach_over += sf.nodes.iter().filter(|&&i| ((x1[v][i as usize] - x0[v][i as usize]) - r).length() > 0.5 * own_reach[v]).count() as u64;
        }
        let mut vfinal: Option<Vec<Vec<Vec3>>> = None;
        if dat.ccd_rounds > 0 {
            // The physical guarantee: CCD with inelastic impulses, then impact zones.
            let q = reach(&x1, &zero);
            let pairs: Vec<super::ccd::CcdPair> = recs
                .iter()
                .filter(|r| r.len == 4 && r.g - r.d < 2.0 * (side(r, &q, true) + side(r, &q, false)))
                .map(|r| super::ccd::CcdPair { k: r.k, nn: r.nn, xi: if dat.penalty > 0.0 { dat.min_gap } else { (xi_share() * r.d).max(dat.min_gap) } })
                .collect();
            st.pairs_ccd = pairs.len() as u64;
            let m: Vec<Vec<f32>> = self.vehicles.iter().map(|v| v.nodes.iter().map(|n| n.mass).collect()).collect();
            let mut vs: Vec<Vec<Vec3>> = (0..x1.len())
                .map(|v| (0..x1[v].len()).map(|i| v0[v][i] + (x1[v][i] - x1p[v][i] - shift[v][i]) / STEP).collect())
                .collect();
            let vpre: Vec<Vec<Vec3>> = if pre.is_empty() { Vec::new() } else { pre.iter().map(|p| p.iter().map(|q| q.1).collect()).collect() };
            super::ccd::resolve(&pairs, &x0, &mut x1, &mut vs, &m, &im, STEP, dat.ccd_rounds, dat.soft_zones, dat.zone_rounds, &mut st.ccd, &vpre, dat.restitution);
            vfinal = Some(vs);
        }
        let q = reach(&x1, &zero);
        recs.retain(|r| r.g < side(r, &q, true) + side(r, &q, false));
        st.pairs_dat = recs.len() as u64;
        let mut pass = Pass { x0: &x0, x1: &x1, t: &mut t, gamma: dat.gamma, stats: &mut st };
        for r in &recs {
            pass.pair(r);
        }
        if let Some(f) = vfinal.as_mut() {
            super::restitute::restitute(&self.vehicles, f, dat.restitution, &dat.near, &mut st);
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

    /// Vehicle `v`'s own crossings now (edge, triangle) with their distance at its first
    /// step (`Dat::first`; NaN without).
    pub fn own_crossings_at(&self, v: usize) -> Vec<([u32; 2], [u32; 3], f32)> {
        let veh = &self.vehicles[v];
        let x: Vec<Vec3> = veh.nodes.iter().map(|n| veh.origin + n.pos).collect();
        let s = surface(veh);
        let first = self.continuous.as_ref().and_then(|d| d.first.get(v)).filter(|f| !f.is_empty());
        let mut out = Vec::new();
        for e in &s.edges {
            for t in &s.tris {
                if t.iter().any(|j| e.contains(j)) {
                    continue;
                }
                let [p, q] = e.map(|i| x[i as usize]);
                let [a, b, c] = t.map(|j| x[j as usize]);
                if !seg_crosses(p, q, a, b, c) {
                    continue;
                }
                let gap = first.map_or(f32::NAN, |f| {
                    let [p, q] = e.map(|i| f[i as usize]);
                    let [a, b, c] = t.map(|j| f[j as usize]);
                    let vt = [p, q].iter().map(|&y| (y - closest_on_triangle(y, a, b, c)).length()).fold(f32::MAX, f32::min);
                    let ee = [(a, b), (b, c), (c, a)].iter().map(|&(u, w)| { let (c1, c2) = closest_segments(p, q, u, w); (c2 - c1).length() }).fold(f32::MAX, f32::min);
                    vt.min(ee)
                });
                out.push((*e, *t, gap));
            }
        }
        out
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
pub fn close_pairs_at(x: &[Vec3], v: &ExplicitBeams, eps: f32) -> Vec<([u32; 4], f32)> {
    close_pairs(x, &surface(v), eps)
}

/// Continuous contact's own clearance at positions `x`: a car's own surfaces are held `gap`
/// apart and no pair is left out, so one built with own pairs (sharing no node) closer than
/// that cannot run; the error lists them (fix the authoring: the lint's `skin_clearance`).
pub fn own_clearance(x: &[Vec3], v: &ExplicitBeams, gap: f32) -> Result<(), String> {
    clearance(close_pairs_at(x, v, gap), gap)
}

fn clearance(mut near: Vec<([u32; 4], f32)>, gap: f32) -> Result<(), String> {
    if near.is_empty() {
        return Ok(());
    }
    near.sort_by(|a, b| a.1.total_cmp(&b.1));
    let list: Vec<String> = near.iter().take(8).map(|(k, g)| format!("nodes {k:?} {:.2} mm", g * 1e3)).collect();
    Err(format!(
        "continuous contact: {} own pairs of the vehicle's collision surface sharing no node are closer than {:.1} mm at spawn ({}); the authoring must keep them apart (lint skin_clearance)",
        near.len(),
        gap * 1e3,
        list.join(", ")
    ))
}

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

/// Every listed pair at the step's start `x0` (point-triangle, edge-edge; between vehicles
/// within r_q of their paint, within one vehicle sharing no node),
/// in a fixed order: vehicle a, vehicle b, then a's nodes or edges ascending, b's primitives
/// ascending.
#[allow(clippy::too_many_arguments)]
fn list_pairs(dat: &mut Dat, x0: &[Vec<Vec3>], surf: &[Surface], off: &[Vec<Vec3>], reach: &[f32], refs: &[(Vec3, Vec3)], own_reach: &mut [f32], st: &mut DatStats) -> Vec<Rec> {
    let r_q = dat.r_q;
    dat.near.clear();
    let mut recs: Vec<Rec> = Vec::new();
    // Pairs are listed within r_q of their paint (the gap less the two sides' offsets).
    let mo: Vec<f32> = off.iter().map(|o| o.iter().map(|v| v.length()).fold(0.0f32, f32::max)).collect();
    let boxes: Vec<(Vec3, Vec3)> =
        surf.iter().enumerate().map(|(v, s)| bounds(s.nodes.iter().map(|&i| x0[v][i as usize]))).collect();
    let mut found = Vec::new();
    let nv = x0.len();
    let none: Vec<Vec<Vec3>> = vec![Vec::new(); nv];
    if dat.self_contact {
        dat.checked.resize(nv, false);
        for v in 0..nv {
            if !dat.checked[v] {
                dat.first.resize(nv, Vec::new());
                dat.first[v] = x0[v].clone();
                if let Err(e) = clearance(close_pairs(&x0[v], &surf[v], dat.self_gap), dat.self_gap) {
                    panic!("vehicle {v}: {e}");
                }
                dat.checked[v] = true;
            }
        }
    }
    for a in 0..nv {
        for b in 0..nv {
            let ((alo, ahi), (blo, bhi)) = (boxes[a], boxes[b]);
            // A car against itself: its own pairs sharing no node, at thickness 0 (its
            // collision surface never passes through itself).
            let own = a == b;
            let apart = |a: usize, b: usize| (refs[a].0 - refs[b].0).length().max((refs[a].1 - refs[b].1).length() * STEP);
            // (A car's own: the largest relative motion of any vehicle near enough to touch it,
            // whose impulses can shove its surface this step.)
            let near = |a: usize, c: usize| {
                let ((alo, ahi), (clo, chi)) = (boxes[a], boxes[c]);
                let r = Vec3::splat(r_q + mo[a] + mo[c]);
                c != a && !((alo - r).cmpgt(chi).any() || (clo - r).cmpgt(ahi).any())
            };
            let apart_ab = if own { (0..nv).filter(|&c| near(a, c)).map(|c| apart(a, c)).fold(0.0f32, f32::max) } else { apart(a, b) };
            let r_q = r_q.min(4.0 * (reach[a] + reach[b] + apart_ab));
            if own {
                own_reach[a] = r_q;
            }
            let (mo_a, mo_b) = if own { (0.0, 0.0) } else { (mo[a], mo[b]) };
            let off: &[Vec<Vec3>] = if own { &none } else { off };
            let r = Vec3::splat(r_q + mo_a + mo_b);
            if (own && !dat.self_contact) || (alo - r).cmpgt(bhi).any() || (blo - r).cmpgt(ahi).any() {
                continue;
            }
            if a < b {
                dat.near.push((a, b));
            }
            // a's nodes against b's triangles (both orders run: every node of both).
            let (sa, sb) = (&surf[a], &surf[b]);
            let mut grid = Grid::new((r_q + mo_a + mo_b).max(0.15));
            for (k, tri) in sb.tris.iter().enumerate() {
                let (lo, hi) = bounds(tri.map(|i| x0[b][i as usize]));
                grid.insert(k as u32, lo - r, hi + r);
            }
            for &i in &sa.nodes {
                let p = x0[a][i as usize];
                grid.query(p, p, &mut found);
                for &k in &found {
                    let tri = sb.tris[k as usize];
                    if own && tri.contains(&i) {
                        continue;
                    }
                    let [q0, q1, q2] = tri.map(|j| x0[b][j as usize]);
                    let (c, cw) = fq::closest_on_triangle(p, q0, q1, q2);
                    if (p - c).length_squared() < (r_q + mo_a + mo_b).powi(2) {
                        let neg = tri.map(|j| (b, j));
                        let mut rec = Rec::new(&neg, &cw, &[(a, i)], &[1.0], c, p, off);
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
            let mut grid = Grid::new((r_q + mo_a + mo_b).max(0.15));
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
                    if own && (k as usize <= ke || f.iter().any(|x| e.contains(x))) {
                        continue;
                    }
                    let [p2, q2] = f.map(|j| x0[b][j as usize]);
                    let (s1, s2) = fq::closest_seg_seg(p1, q1, p2, q2);
                    let (c1, c2) = (p1 + (q1 - p1) * s1, p2 + (q2 - p2) * s2);
                    if (c2 - c1).length_squared() < (r_q + mo_a + mo_b).powi(2) {
                        let mut rec = Rec::new(&e.map(|i| (a, i)), &[1.0 - s1, s1], &f.map(|j| (b, j)), &[1.0 - s2, s2], c1, c2, off);
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
    if let Some(k) = dat.keys.as_mut() {
        *k = recs.iter().map(|r| r.k).collect();
        // (CC_DEBUG_PAIR="v:i,v:j,v:k,v:l" sorted: that pair's record, a diagnostic.)
        if let Ok(want) = std::env::var("CC_DEBUG_PAIR") {
            for r in &recs {
                let mut k: Vec<String> = r.k.iter().map(|(v, i)| format!("{v}:{i}")).collect();
                k.sort();
                if k.join(",") == want {
                    eprintln!("CC_DEBUG_PAIR {want}: g {:e} d {:e} r_q {:e} nn {} len {} offsets {:?} x0 {:?}", r.g, r.d, r_q, r.nn, r.len, r.k.map(|(v, i)| off[v].get(i as usize).map_or(0.0, |o| o.length())), r.k.map(|(v, i)| x0[v][i as usize]));
                }
            }
        }
    }
    recs
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

/// The share of a pair's thickness the CCD keeps (0.5; `DAT_XI`, a measurement).
fn xi_share() -> f32 {
    std::env::var("DAT_XI").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5)
}

/// Velocity-only impulse contact (`DAT_RESPONSE=vel`, `ccd_vel::velocity_rounds`): its rows take
/// the speculative pass's place too, so every contact goes through one law.
fn vel_mode() -> bool {
    std::env::var("DAT_RESPONSE").is_ok_and(|v| v == "vel")
}

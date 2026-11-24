//! Vehicle–vehicle contact for the explicit node–beam solver (CPU reference),
//! written from `engine/docs/explicit-beams-vehicle-contact-spec.md`.
//!
//! [`ExplicitBeamsWorld`] steps several [`ExplicitBeams`] vehicles together. One
//! world step is every vehicle's own step (nodes, beams, self-collision, slide
//! nodes), in vehicle order; then each vehicle's bounding box; then the
//! vehicle-contact pass, one owner at a time in vehicle order: the owner's partner
//! scan (other vehicles whose padded boxes overlap its own; a fast approach resets
//! both vehicles' throttles) and, when its partners offer contact points, its
//! triangle loop. A triangle loop tests each of the owner's collision triangles
//! (throttled by its contact throttle) against the partners' contact points inside
//! its box grown by the owner's collision range, decides the side of each hit by a
//! vote of the point and its beam neighbours, and applies the contact force to the
//! point and its reaction to the triangle's nodes by their weights. The forces are
//! consumed by the next step's node pass.
//!
//! Vehicles whose boxes never overlap are not touched: their results are
//! bit-identical to stepping each alone.

use crate::explicit_beams::{
    contact_response, shift_depth, ContactSurface, ExplicitBeams, ExplicitKind, ForceTrace, Node,
    GRAVITY, NODE_IMMOVABLE, STEP,
};
use glam::Vec3;

mod edge;
pub(crate) use edge::self_edge_hits;

/// Speculative look-ahead for vehicle contact (under measurement, `SPECULATIVE=1`).
/// The push-out rate (m/s) of a pair already behind the surface (under measurement,
/// `PUSH_OUT`; 0: none, as before).
pub(crate) fn push_out_rate() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("PUSH_OUT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0))
}

/// Residual sweeps of vehicle contact per turn (under measurement, `CONTACT_SWEEPS`;
/// 1: the one residual pass, as before).
pub(crate) fn contact_sweeps() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("CONTACT_SWEEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1))
}

/// Vehicle contact's residual sweeps as Gauss–Seidel (under measurement, `CONTACT_GS=1`).
pub(crate) fn gauss_seidel() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CONTACT_GS").is_ok_and(|v| v == "1"))
}

pub(crate) fn speculative() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SPECULATIVE").is_ok_and(|v| v == "1"))
}
pub use edge::EDGE_DEPTH;

/// Each face of a vehicle's box is moved out by this much (m).
pub const BOX_MARGIN: f32 = 0.05;
/// Squared speed difference of two partners' first nodes above which both
/// vehicles' throttles reset ((m/s)², i.e. 4 m/s).
pub const FAST_APPROACH_SQ: f32 = 16.0;
/// The longest contact-throttle wait (steps).
pub const MAX_WAIT: i32 = 12;

/// `min(a, b)` as the spec writes it: b if b < a, else a.
fn min_spec(a: i32, b: i32) -> i32 {
    if b < a {
        b
    } else {
        a
    }
}

/// +1 for x ≥ 0 (−0 included), −1 otherwise (NaN included).
fn sign(x: f32) -> i32 {
    if x >= 0.0 {
        1
    } else {
        -1
    }
}

/// A collision triangle's local frame for vehicle contact (spec §5).
#[derive(Clone, Copy, Debug)]
pub struct ContactFrame {
    pub p0: Vec3,
    /// Unit normal (or the unnormalised cross product when shorter than 1e-8).
    pub n: Vec3,
    /// Rows of the inverse of the matrix with columns (u, w, n), or `None` when its
    /// determinant's magnitude is at most 1e-6 (the zero matrix).
    pub inv: Option<[Vec3; 3]>,
}

/// The swept crossing (anti-clip, spec §36): point x_prev → x against the triangle
/// whose frame was `prev` and is `now` over the step. When the point went from one
/// side of the plane to the other and the crossing (linear in the step) is inside
/// the triangle: the weights (α, β, γ) at the crossing and the side it came from
/// (+1: the normal's side). α and β are interpolated, γ = (1 − α) − β (the weights
/// sum to one, and the GPU keeps α and β only).
pub fn swept_crossing(
    prev: &ContactFrame,
    now: &ContactFrame,
    x_prev: Vec3,
    x: Vec3,
) -> Option<([f32; 3], f32)> {
    let (a0, b0, _, d0) = prev.local(x_prev);
    let (a1, b1, _, d1) = now.local(x);
    crossing_weights(a0, b0, d0, a1, b1, d1)
}

/// The crossing of a point between two frames' local coordinates (α, β, δ before
/// and after): the weights at the plane, if inside, and the side it came from.
pub fn crossing_weights(
    a0: f32,
    b0: f32,
    d0: f32,
    a1: f32,
    b1: f32,
    d1: f32,
) -> Option<([f32; 3], f32)> {
    if !((d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0)) {
        return None;
    }
    let s = d0 / (d0 - d1);
    let a = a0 + (a1 - a0) * s;
    let b = b0 + (b1 - b0) * s;
    let w = [a, b, (1.0 - a) - b];
    if w.iter().all(|&c| c >= 0.0) {
        Some((w, if d0 > 0.0 { 1.0 } else { -1.0 }))
    } else {
        None
    }
}

/// Anti-clip's side memory depth (§36): a remembered node is pushed back to its side
/// while it is at most this far behind the triangle (m). **Choice** (the soft-body
/// solver's anti-clip keeps 30 cm; a node 10 cm through a panel is a failure; BeamNG
/// documents that a triangle with no node within 10 cm goes inactive).
pub const CLIP_DEPTH: f32 = 0.1;

/// Anti-clip's side memory (§36), per node four words: the sides it holds this
/// step (m0, m1: last step's contacts) and those it records this step (p0, p1),
/// each a [`clip_word`] or [`CLIP_EMPTY`]. Memory is the node's contacts of the
/// step before: a contact (a hit, or an anti-clip-only triangle's side check)
/// records its triangle and side; the node phase makes the recorded words the
/// held ones. Two slots, the two smallest words recorded (an order-free rule, as
/// the GPU's lanes record in any order: a node in contact with three triangles at
/// once keeps the two of smallest key).
pub const CLIP_EMPTY: u32 = 0x7FFF_FFFF;
/// The owner of self-collision keys (vehicle contact keys: the owner vehicle).
pub const CLIP_SELF: u32 = 1023;
/// Triangles per vehicle anti-clip keys can name (the last key is CLIP_EMPTY's).
pub const CLIP_MAX_TRIANGLES: u32 = (1 << 20) - 1;

/// The memory word of triangle `t` of owner `owner` (a vehicle below 1023, or
/// [`CLIP_SELF`]) on side `side` (±1): owner << 21 | t << 1 | (side > 0).
pub fn clip_word(owner: u32, t: u32, side: f32) -> u32 {
    owner << 21 | t << 1 | (side > 0.0) as u32
}

/// The held side (±1) of node memory `c` for triangle `t` of `owner`, if any.
pub fn clip_side(c: &[u32; 4], owner: u32, t: u32) -> Option<f32> {
    let key = owner << 20 | t;
    for &w in &c[..2] {
        if w != CLIP_EMPTY && w >> 1 == key {
            return Some(if w & 1 != 0 { 1.0 } else { -1.0 });
        }
    }
    None
}

/// Records `word` for the next step (the two smallest of the step's words).
pub fn clip_record(c: &mut [u32; 4], word: u32) {
    if word < c[2] {
        c[3] = c[2];
        c[2] = word;
    } else if word < c[3] {
        c[3] = word;
    }
}

/// The node phase: the recorded words become the held ones.
pub fn clip_roll(c: &mut [u32; 4]) {
    c[0] = c[2];
    c[1] = c[3];
    c[2] = CLIP_EMPTY;
    c[3] = CLIP_EMPTY;
}

/// The anti-clip test (§36) of a point against a triangle, from the point's local
/// coordinates now (α, β, γ, δ) and, when it holds no side for the triangle, its
/// crossing of the plane over the step: the side it is kept on (held, or the one
/// it came from, or, new and in the band over the triangle, the one it is on),
/// the weights (the crossing's when it crossed) and its distance in front of that
/// side, d = side·δ. A contact when the weights are inside and −CLIP_DEPTH ≤ d ≤ r.
pub struct ClipTest {
    pub side: f32,
    pub w: [f32; 3],
    pub d: f32,
}

/// The depth an anti-clip contact pushes with (§36): r − d in the band (an
/// anti-clip-only triangle: −d behind its plane), at most r. **Choice:** a node held
/// behind the plane is pushed as one at the plane, not by its whole depth: the
/// contact rule's push grows with depth (0.2·p/h, 10 m/s of separation at 2.5 cm),
/// so a node 10 cm through was kicked out at 40 m/s and the BX coupe's side-pole
/// run went non-finite.
pub fn clip_depth(only: bool, d: f32, r: f32) -> f32 {
    let p = if only { -d } else { r - d };
    p.min(r)
}

/// [`ClipTest`]: `held` the remembered side, `crossing` the swept crossing (only
/// asked for when nothing is held); `None` when the point is not in contact.
pub fn clip_test(
    local: (f32, f32, f32, f32),
    held: Option<f32>,
    crossing: impl FnOnce() -> Option<([f32; 3], f32)>,
    r: f32,
    band_side: impl FnOnce() -> f32,
) -> Option<ClipTest> {
    let (alpha, beta, gamma, delta) = local;
    let over = alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0;
    let (side, w) = match held {
        Some(s) => (s, over.then_some([alpha, beta, gamma])),
        None => match crossing() {
            Some((w, s)) => (s, Some(w)),
            None if over && delta.abs() <= r => (band_side(), Some([alpha, beta, gamma])),
            None => return None,
        },
    };
    let d = side * delta;
    match w {
        Some(w) if d <= r && d >= -CLIP_DEPTH => Some(ClipTest { side, w, d }),
        _ => None,
    }
}

impl ContactFrame {
    pub fn new(p0: Vec3, p1: Vec3, p2: Vec3) -> Self {
        let u = p1 - p0;
        let w = p2 - p0;
        let c = u.cross(w);
        let len = (c.x * c.x + c.y * c.y + c.z * c.z).sqrt();
        let n = if len > 1e-8 { c * (1.0 / len) } else { c };
        let det = u.dot(w.cross(n));
        let inv = (det.abs() > 1e-6).then(|| {
            let r = 1.0 / det;
            [w.cross(n) * r, n.cross(u) * r, u.cross(w) * r]
        });
        Self { p0, n, inv }
    }

    /// (α, β, γ, δ) of world point `x`: α weights n₁, β n₂, γ n₀; δ the signed
    /// distance from the plane.
    pub fn local(&self, x: Vec3) -> (f32, f32, f32, f32) {
        let q = x - self.p0;
        let (alpha, beta, delta) = match self.inv {
            Some([ra, rb, rd]) => (ra.dot(q), rb.dot(q), rd.dot(q)),
            None => (0.0, 0.0, 0.0),
        };
        (alpha, beta, (1.0 - alpha) - beta, delta)
    }
}

/// The side vote (spec §5.3): true when the point is behind the triangle (flip the
/// normal and δ). `neighbours` are the world positions of the point's neighbour
/// list entries, in list order.
pub fn vote_behind(
    delta: f32,
    n: Vec3,
    p0: Vec3,
    neighbours: impl ExactSizeIterator<Item = Vec3>,
) -> bool {
    let mut score = 3 * sign(delta);
    if neighbours.len() > 3 {
        for y in neighbours {
            let e = y - p0;
            score += sign(n.x * e.x + n.y * e.y + n.z * e.z);
        }
    }
    score < 0
}

/// The inverse mass of a triangle's point at barycentric weights `w` (spec §10.3)
/// from its nodes' inverse masses `im`: the reaction J is shared out as w_k·J, so
/// the point's velocity changes by J·Σ w_k²·im_k (terms with w_k = 0 left out). The
/// weighted mean Σ w_k·m_k (RoR's rule) overstates the mass wherever a light node
/// carries most of the weight beside a heavy one (0.07 kg at 0.77 next to 10 kg at
/// 0.14: 1.5 kg against 0.12 kg), and the light node is then thrown off at hundreds
/// of m/s. In the order n₁, n₂, n₀ (α, β, γ), summed left to right.
pub fn two_body_inverse_mass(im: [f32; 3], w: [f32; 3]) -> f32 {
    let term = |k: usize| {
        if w[k] > 0.0 {
            (w[k] * w[k]) * im[k]
        } else {
            0.0
        }
    };
    (term(0) + term(1)) + term(2)
}

/// Statistics of the last contact pass (not simulated state).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContactStats {
    /// Owner–partner pairs (ordered) whose boxes overlapped.
    pub partner_pairs: u32,
    /// Triangles tested, candidates tested, hits.
    pub triangles_tested: u32,
    pub candidates: u32,
    pub hits: u32,
    /// Fast-approach resets.
    pub resets: u32,
}

/// Several explicit-beams vehicles stepped together with vehicle–vehicle contact.
#[derive(Clone, Debug)]
pub struct ExplicitBeamsWorld {
    /// In vehicle order (spawn order): the order of the contact pass.
    pub vehicles: Vec<ExplicitBeams>,
    /// Vehicle contact on (default).
    pub contact: bool,
    /// The contact throttle (default on, as the reference): off tests every
    /// triangle every step.
    pub throttle: bool,
    /// Steps run.
    pub steps: u64,
    /// The last contact pass.
    pub last: ContactStats,
    /// Every contact pass so far, summed.
    pub total: ContactStats,
    /// The first step (1-based count of steps run) whose contact pass had a hit.
    pub first_hit_step: Option<u64>,
    /// Scratch: world positions per vehicle and padded boxes.
    world: Vec<Vec<Vec3>>,
    /// Scratch: the current owner's and its partners' positions in the owner's frame
    /// (relative to its origin: the contact's geometry; the boxes stay in the World's).
    frame: Vec<Vec<Vec3>>,
    /// Scratch: every node's force at the contact pass's start (two-body hits).
    pass_force: Vec<Vec<Vec3>>,
    /// The energy ledger (None: off; [`Self::enable_ledger`]).
    pub ledger: Option<EnergyLedger>,
    boxes: Vec<(Vec3, Vec3)>,
    /// Vehicle sleep ([`crate::explicit_sleep`]): `None` (the default) off, every frame
    /// as before.
    pub sleep: Option<crate::explicit_sleep::VehicleSleep>,
    /// Vehicles the contact pass leaves out (sleeping; empty: none).
    skip: Vec<bool>,
    /// Vehicle sleep's proximity wake: the GPU contact plan's frame bounds margin (m,
    /// `engine_exec::explicit_plan::DEFAULT_CONTACT_MARGIN`) and the steps a plan covers
    /// at least (`VV_PLAN_STEPS`).
    pub plan_margin: f32,
    pub plan_steps: u32,
    /// The last contact pass's dissipative two-body hits with their final forces,
    /// for the next step's work guard on the final velocities
    /// ([`Self::guard_final`], spec §37.1).
    staged_hits: Vec<StagedHit>,
    /// §37.1's work guard on the final velocities ([`Self::guard_final`]). Off by
    /// default until the GPU kernels carry it (the CPU reference and the kernels
    /// agree bit for bit); `FINAL_GUARD=1` (or this field) turns it on.
    pub final_guard: bool,
    /// The edge pass's counts so far (a measurement): pairs tested, closest points at an
    /// end, parallel, close but not facing, crossed deeper than `EDGE_DEPTH`, hits, hits
    /// crossed (gap below zero).
    pub edge_stats: [u64; 7],
    /// `EDGE_TRACE=1` (a measurement): each tested edge pair's last outcome, and the
    /// hits first seen crossed by the pair's outcome the step before (at an end,
    /// parallel, not facing, outside the band, deeper, a hit, not tested).
    pub(crate) edge_trace: Option<std::collections::HashMap<(usize, usize, u32, u32, u32, u32), (u64, u8)>>,
    pub edge_first: [u64; 7],
    /// Vehicle contact's band between vehicles (m; None: each owner's own
    /// `collision_range`, as before). Self contact keeps each vehicle's own. Under
    /// measurement (`VV_RANGE`).
    pub vv_range: Option<f32>,
    /// Scratch: each vehicle's live skin edges this contact pass (edge contact on).
    skin: Vec<Option<Vec<edge::SkinEdge>>>,
}

/// A dissipative two-body hit as the contact pass staged it: node `h` of
/// vehicle `b` takes `f`, the nodes `tri` of vehicle `a` take −f·`w`.
#[derive(Clone, Copy, Debug)]
struct StagedHit {
    a: usize,
    b: usize,
    h: u32,
    /// A second contacter node and its weight (edge hits; t = 0: h alone).
    h2: u32,
    t: f32,
    tri: [u32; 3],
    w: [f32; 3],
    n: Vec3,
    f: Vec3,
}

impl ExplicitBeamsWorld {
    /// A world of `vehicles` (spawned with [`ExplicitBeams::finish`], which builds
    /// their contact points, neighbour lists and contact throttles).
    pub fn new(mut vehicles: Vec<ExplicitBeams>) -> Self {
        for v in &mut vehicles {
            if v.contact_throttle.len() != v.triangles.len()
                || v.neighbour_start.len() != v.nodes.len() + 1
            {
                v.finish_contact();
            }
        }
        Self {
            vehicles,
            contact: true,
            throttle: true,
            steps: 0,
            last: ContactStats::default(),
            total: ContactStats::default(),
            first_hit_step: None,
            world: Vec::new(),
            frame: Vec::new(),
            ledger: None,
            pass_force: Vec::new(),
            boxes: Vec::new(),
            sleep: None,
            skip: Vec::new(),
            plan_margin: 1.0,
            plan_steps: 34,
            staged_hits: Vec::new(),
            final_guard: std::env::var("FINAL_GUARD").is_ok_and(|v| v == "1"),
            edge_stats: [0; 7],
            edge_trace: std::env::var("EDGE_TRACE").is_ok_and(|v| v == "1").then(Default::default),
            edge_first: [0; 7],
            vv_range: std::env::var("VV_RANGE").ok().and_then(|v| v.parse().ok()),
            skin: Vec::new(),
        }
    }

    /// One frame: every vehicle's wing check and origin rebase, then `steps` world
    /// steps. With [`Self::sleep`], sleeping vehicles are not stepped
    /// ([`Self::frame_sleeping`]).
    pub fn frame(&mut self, steps: u32) {
        if let Some(cfg) = self.sleep {
            self.frame_sleeping(steps, &cfg);
            return;
        }
        for v in &mut self.vehicles {
            v.check_wings();
            v.rebase_origin();
        }
        for _ in 0..steps {
            self.step();
        }
    }

    /// One 60 fps frame with the remainder carried (vehicle 0's clock; every
    /// vehicle's clock advances). Returns the steps.
    pub fn frame_60(&mut self) -> u32 {
        let mut steps = 0;
        for (i, v) in self.vehicles.iter_mut().enumerate() {
            let s = v.clock.next();
            if i == 0 {
                steps = s;
            }
        }
        self.frame(steps);
        steps
    }

    /// The GPU contact plan's padded frame boxes (`explicit_beams_vv_bounds`) of a frame
    /// of `steps` steps, from the current state: each vehicle's node box grown by twice
    /// the travel at its extreme velocities over the plan's steps, the margin, and
    /// [`BOX_MARGIN`].
    pub fn plan_boxes(&self, steps: u32) -> Vec<(Vec3, Vec3)> {
        let time = 2.0 * steps.max(self.plan_steps) as f32 * STEP;
        let m = Vec3::splat(self.plan_margin);
        self.vehicles
            .iter()
            .map(|v| {
                let (mut lo, mut hi) = (Vec3::splat(3.0e38), Vec3::splat(-3.0e38));
                let (mut vl, mut vh) = (Vec3::ZERO, Vec3::ZERO);
                for n in &v.nodes {
                    let x = v.origin + n.pos;
                    lo = lo.min(x);
                    hi = hi.max(x);
                    vl = vl.min(n.vel);
                    vh = vh.max(n.vel);
                }
                let alo = lo - (-vl * time + m);
                let ahi = hi + (vh * time + m);
                (alo - Vec3::splat(BOX_MARGIN), ahi + Vec3::splat(BOX_MARGIN))
            })
            .collect()
    }

    /// Vehicle sleep's groups: vehicles whose [`Self::plan_boxes`] overlap, connected
    /// (the GPU plan's contact groups), as a label (smallest member) per vehicle. Without
    /// vehicle contact (or one vehicle) every vehicle is its own group.
    pub fn sleep_groups(&self, steps: u32) -> Vec<usize> {
        let nv = self.vehicles.len();
        let mut label: Vec<usize> = (0..nv).collect();
        if !self.contact || nv < 2 {
            return label;
        }
        fn root(l: &mut [usize], mut x: usize) -> usize {
            while l[x] != x {
                l[x] = l[l[x]];
                x = l[x];
            }
            x
        }
        let b = self.plan_boxes(steps);
        for i in 0..nv {
            for j in i + 1..nv {
                let apart = b[i].1.cmplt(b[j].0).any() || b[j].1.cmplt(b[i].0).any();
                if !apart {
                    let (ri, rj) = (root(&mut label, i), root(&mut label, j));
                    let m = ri.min(rj);
                    label[ri] = m;
                    label[rj] = m;
                }
            }
        }
        for i in 0..nv {
            label[i] = root(&mut label, i);
        }
        label
    }

    /// Wake vehicle `v` (a host write: input, velocities, a force).
    pub fn wake(&mut self, v: usize) {
        self.vehicles[v].sleep.wake();
    }

    /// One frame with vehicle sleep ([`crate::explicit_sleep`]): a group with an awake
    /// member wakes its sleeping members; the awake vehicles are stepped (the contact
    /// pass among them), their stillness tracked each step after the node pass; at the
    /// frame's end the rule updates each one, and a group whose every member reached the
    /// dwell falls asleep.
    pub fn frame_sleeping(&mut self, steps: u32, cfg: &crate::explicit_sleep::VehicleSleep) {
        let nv = self.vehicles.len();
        let groups = self.sleep_groups(steps);
        for g in 0..nv {
            let members = (0..nv).filter(|&v| groups[v] == g);
            let any_awake = members.clone().any(|v| !self.vehicles[v].sleep.asleep);
            if any_awake {
                for v in members {
                    if self.vehicles[v].sleep.asleep {
                        self.vehicles[v].sleep.wake();
                    }
                }
            }
        }
        let awake: Vec<bool> = self.vehicles.iter().map(|v| !v.sleep.asleep).collect();
        for v in self.vehicles.iter_mut().filter(|v| !v.sleep.asleep) {
            v.check_wings();
            v.rebase_origin();
            v.sleep.track = Some(cfg.speed);
            v.sleep.moving = v.grab.is_some();
        }
        for _ in 0..steps {
            let v0 = self.staged_velocities();
            let wheels = self.wheels_begin(Some(&awake));
            for v in self.vehicles.iter_mut().filter(|v| !v.sleep.asleep) {
                v.step_nodes();
            }
            self.guard_final(&v0, None);
            for v in self.vehicles.iter_mut().filter(|v| !v.sleep.asleep) {
                v.step_rest();
            }
            self.steps += 1;
            if self.contact && nv > 1 {
                self.skip = self.vehicles.iter().map(|v| v.sleep.asleep).collect();
                self.contact_pass();
                self.skip.clear();
            }
            if wheels {
                self.wheels_end();
            }
        }
        let mut eligible = vec![false; nv];
        for (i, v) in self.vehicles.iter_mut().enumerate() {
            if !awake[i] {
                continue;
            }
            let moving = v.sleep.moving;
            v.sleep.track = None;
            v.sleep.moving = false;
            let pos: Vec<Vec3> = v.nodes.iter().map(|n| n.pos).collect();
            eligible[i] = crate::explicit_sleep::frame_end(&mut v.sleep, &pos, moving, steps, cfg);
        }
        for g in 0..nv {
            let members: Vec<usize> = (0..nv).filter(|&v| groups[v] == g && awake[v]).collect();
            if !members.is_empty() && members.iter().all(|&v| eligible[v]) {
                for v in members {
                    self.vehicles[v].sleep.asleep = true;
                }
            }
        }
    }

    /// One step: every vehicle's own step in vehicle order, then the contact pass.
    pub fn step(&mut self) {
        let v0 = self.staged_velocities();
        let wheels = self.wheels_begin(None);
        let Some(mut ledger) = self.ledger.take() else {
            for v in &mut self.vehicles {
                v.step_nodes();
            }
            self.guard_final(&v0, None);
            for v in &mut self.vehicles {
                v.step_rest();
            }
            self.steps += 1;
            if self.contact && self.vehicles.len() > 1 {
                self.contact_pass();
            }
            if wheels {
                self.wheels_end();
            }
            return;
        };
        let mut before: Vec<Vec<(Vec3, Vec3, Vec3)>> = self
            .vehicles
            .iter()
            .map(|v| {
                v.nodes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (v.origin + n.pos, n.vel, v.force_now(i)))
                    .collect()
            })
            .collect();
        for v in &mut self.vehicles {
            v.step_nodes();
        }
        self.guard_final(&v0, Some((&mut ledger, &mut before)));
        for (k, v) in self.vehicles.iter_mut().enumerate() {
            v.step_rest();
            ledger.account(k, v, &before[k]);
        }
        self.steps += 1;
        ledger.vc_reset(&self.vehicles);
        self.ledger = Some(ledger);
        if self.contact && self.vehicles.len() > 1 {
            self.contact_pass();
        }
        if wheels {
            self.wheels_end();
        }
        if let Some(l) = self.ledger.as_mut() {
            l.refresh(&self.vehicles);
        }
    }

    /// Raycast wheels on other vehicles (`DriveConfig::see_bodies`, with vehicle contact
    /// on): at a step's start, every vehicle's live collision triangles as they stand
    /// (World positions, velocities, inverse masses) for the wheels' rays of every vehicle
    /// stepped (`awake`: those; None every one). False: no vehicle's wheels look (nothing
    /// set, every step as without it).
    fn wheels_begin(&mut self, awake: Option<&[bool]>) -> bool {
        let nv = self.vehicles.len();
        let on = |v: &ExplicitBeams| v.drive.as_ref().is_some_and(|d| d.config.see_bodies && !d.config.wheels.is_empty());
        if !self.contact || nv < 2 || !self.vehicles.iter().any(on) {
            return false;
        }
        let mut sc = crate::explicit_drive::PartnerScene::default();
        for (j, v) in self.vehicles.iter().enumerate() {
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for n in &v.nodes {
                let x = v.origin + n.pos;
                lo = lo.min(x);
                hi = hi.max(x);
            }
            let t0 = sc.tris.len() as u32;
            for (t, tri) in v.triangles.iter().enumerate() {
                let dead = v.tri_dead.get(t).copied().unwrap_or(false) || v.tri_anti_only(t);
                if dead {
                    continue;
                }
                let n = tri.nodes.map(|i| v.nodes[i as usize]);
                sc.tris.push(crate::explicit_drive::PartnerTri {
                    vehicle: j as u32,
                    nodes: tri.nodes,
                    x: n.map(|n| v.origin + n.pos),
                    v: n.map(|n| n.vel),
                    im: n.map(|n| if n.has(NODE_IMMOVABLE) { 0.0 } else { 1.0 / n.mass }),
                });
            }
            sc.vehicles.push((lo, hi, t0, sc.tris.len() as u32));
        }
        let sc = std::sync::Arc::new(sc);
        for (j, v) in self.vehicles.iter_mut().enumerate() {
            if awake.is_some_and(|a| !a[j]) {
                continue;
            }
            if let Some(d) = v.drive.as_mut().filter(|d| d.config.see_bodies) {
                d.partners = Some(sc.clone());
                d.owner = j as u32;
            }
        }
        true
    }

    /// After a step's contact pass: the wheels' reactions on other vehicles' triangles
    /// onto those nodes' contact sums (consumed by the next step's node pass, as vehicle
    /// contact's), and the scene dropped.
    fn wheels_end(&mut self) {
        let mut all = Vec::new();
        for v in &mut self.vehicles {
            if let Some(d) = v.drive.as_mut() {
                d.partners = None;
                all.append(&mut d.reactions);
            }
        }
        for r in all {
            let v = &mut self.vehicles[r.vehicle as usize];
            for k in 0..3 {
                if r.w[k] > 0.0 {
                    crate::explicit_drive::apply_contact(v, r.nodes[k], r.force * r.w[k]);
                }
            }
        }
    }

    /// Turns the energy ledger on from the current state (its start energy).
    pub fn enable_ledger(&mut self) {
        for v in &mut self.vehicles {
            v.trace = Some(Box::default());
        }
        let mut l = EnergyLedger::default();
        l.prev = self
            .vehicles
            .iter()
            .map(|v| ForceSplit::new(v.nodes.len()))
            .collect();
        l.vc = self
            .vehicles
            .iter()
            .map(|v| {
                [
                    vec![Vec3::ZERO; v.nodes.len()],
                    vec![Vec3::ZERO; v.nodes.len()],
                ]
            })
            .collect();
        l.refresh(&self.vehicles);
        l.start = l.mechanical();
        self.ledger = Some(l);
    }

    /// The bounding boxes (spec §4.1) and the vehicle-contact pass (§3.1 3).
    pub fn contact_pass(&mut self) {
        self.staged_hits.clear();
        let nv = self.vehicles.len();
        if self.vehicles.iter().any(|v| v.two_body_contact) {
            self.pass_force.resize(nv, Vec::new());
            for (pf, v) in self.pass_force.iter_mut().zip(&self.vehicles) {
                pf.clear();
                pf.extend(v.forces_now());
            }
        }
        self.world.resize(nv, Vec::new());
        self.boxes.clear();
        for (v, w) in self.vehicles.iter().zip(&mut self.world) {
            w.clear();
            w.extend(v.nodes.iter().map(|n| v.origin + n.pos));
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for &x in w.iter() {
                lo = lo.min(x);
                hi = hi.max(x);
            }
            self.boxes
                .push((lo - Vec3::splat(BOX_MARGIN), hi + Vec3::splat(BOX_MARGIN)));
        }
        self.edge_begin();
        let mut stats = ContactStats::default();
        let mut partners: Vec<usize> = Vec::new();
        for a in 0..nv {
            partners.clear();
            if self.skip.get(a).copied().unwrap_or(false) {
                continue;
            }
            let (alo, ahi) = self.boxes[a];
            for b in 0..nv {
                if b == a || self.skip.get(b).copied().unwrap_or(false) {
                    continue;
                }
                let (blo, bhi) = self.boxes[b];
                let apart = ahi.cmplt(blo).any() || bhi.cmplt(alo).any();
                if apart {
                    continue;
                }
                partners.push(b);
                stats.partner_pairs += 1;
                // Fast approach (§3.4): every throttle of both vehicles resets.
                let va = self.vehicles[a].nodes.first().map_or(Vec3::ZERO, |n| n.vel);
                let vb = self.vehicles[b].nodes.first().map_or(Vec3::ZERO, |n| n.vel);
                let d = va - vb;
                let q = d.x * d.x + d.y * d.y + d.z * d.z;
                if q > FAST_APPROACH_SQ {
                    stats.resets += 1;
                    for x in [a, b] {
                        let v = &mut self.vehicles[x];
                        for t in &mut v.triangles {
                            t.wait = 0;
                        }
                        for c in &mut v.contact_throttle {
                            c[0] = 0;
                        }
                    }
                }
            }
            let offered: usize = partners
                .iter()
                .map(|&b| self.vehicles[b].contact_points.len())
                .sum();
            if offered > 0 {
                self.owner_frame(a, &partners);
                self.triangle_loop(a, &partners, &mut stats);
            }
        }
        if stats.hits > 0 && self.first_hit_step.is_none() {
            self.first_hit_step = Some(self.steps);
        }
        self.total.partner_pairs += stats.partner_pairs;
        self.total.triangles_tested += stats.triangles_tested;
        self.total.candidates += stats.candidates;
        self.total.hits += stats.hits;
        self.total.resets += stats.resets;
        self.last = stats;
    }

    /// The owner `a`'s and its partners' node positions in `a`'s frame: each vehicle's
    /// origin less `a`'s, plus its node positions (the GPU's `vvRel`; exactly the world
    /// positions while every origin is 0).
    fn owner_frame(&mut self, a: usize, partners: &[usize]) {
        let nv = self.vehicles.len();
        self.frame.resize(nv, Vec::new());
        let oa = self.vehicles[a].origin;
        for &x in std::iter::once(&a).chain(partners) {
            let v = &self.vehicles[x];
            let rel = v.origin - oa;
            let f = &mut self.frame[x];
            f.clear();
            f.extend(v.nodes.iter().map(|n| rel + n.pos));
        }
    }

    /// Owner `a`'s triangle loop against `partners` (§3.3, §4.2, §5, §6): the walk
    /// (throttles, boxes, hit tests and side votes, all from positions), then the
    /// hits' forces in the walk's order (triangles in list order, candidates in
    /// order). The walk reads no force, so the two phases are the specification's
    /// single loop.
    fn triangle_loop(&mut self, a: usize, partners: &[usize], stats: &mut ContactStats) {
        let r = self.vv_range.unwrap_or(self.vehicles[a].collision_range);
        let rr = Vec3::splat(r);
        let mut cands: Vec<(usize, u32)> = Vec::new();
        let mut hits: Vec<Hit> = Vec::new();
        // The owner's profile without the throttle (§36): a triangle never waits.
        let cap = if self.vehicles[a].settings.triangle_throttle {
            i32::MAX
        } else {
            0
        };
        for t in 0..self.vehicles[a].triangles.len() {
            let [mut cwait, mut cidle] = self.vehicles[a].contact_throttle[t];
            if self.throttle {
                if cwait > 0 {
                    cidle += 1;
                    cwait -= 1;
                    self.vehicles[a].contact_throttle[t] = [cwait, cidle];
                    continue;
                }
                cwait = min_spec(cidle, MAX_WAIT);
                cidle = 0;
            }
            stats.triangles_tested += 1;
            let [i0, i1, i2] = self.vehicles[a].triangles[t].nodes;
            let wa = &self.frame[a];
            let (p0, p1, p2) = (wa[i0 as usize], wa[i1 as usize], wa[i2 as usize]);
            // A broken triangle, or an anti-clip-only one while its vehicle's
            // anti-clip is off: no candidates.
            let dead = self.vehicles[a].tri_dead.get(t).copied().unwrap_or(false)
                || (self.vehicles[a].tri_anti_only(t) && !self.vehicles[a].anti_clip);
            let (lo, hi) = if dead {
                // A broken JBeam triangle (jbeam-semantics-spec.md §18): no candidates.
                (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY))
            } else {
                (p0.min(p1).min(p2) - rr, p0.max(p1).max(p2) + rr)
            };
            cands.clear();
            // Anti-clip (§36): the box also reaches what crossed it over the step
            // (with anti-clip on the owner or the partner).
            let anti_a = self.vehicles[a].anti_clip && !dead;
            let tri_travel = if !dead {
                let va = &self.vehicles[a];
                [i0, i1, i2]
                    .iter()
                    .map(|&i| va.nodes[i as usize].vel.length())
                    .fold(0.0f32, f32::max)
                    * STEP
            } else {
                0.0
            };
            for &b in partners {
                let wb = &self.frame[b];
                let anti = anti_a || (self.vehicles[b].anti_clip && !dead);
                for &h in &self.vehicles[b].contact_points {
                    let x = wb[h as usize];
                    let pad = if anti {
                        Vec3::splat(
                            tri_travel
                                + self.vehicles[b].nodes[h as usize].vel.length() * STEP
                                + CLIP_DEPTH,
                        )
                    } else {
                        Vec3::ZERO
                    };
                    if x.cmpge(lo - pad).all() && x.cmple(hi + pad).all() {
                        cands.push((b, h));
                    }
                }
            }
            if cands.is_empty() {
                cwait += 1;
            } else {
                let frame = ContactFrame::new(p0, p1, p2);
                for &(b, h) in &cands {
                    stats.candidates += 1;
                    let x = self.frame[b][h as usize];
                    let (alpha, beta, gamma, delta) = frame.local(x);
                    // Anti-clip (§36): the side a point is kept on is the one it
                    // held for this triangle last step (it was in contact), or the
                    // one it came from when it crossed the plane over the step
                    // inside the triangle, or (new, in the band) the side vote's;
                    // it is pushed back to that side from up to CLIP_DEPTH behind.
                    // Each contact records its side for the next step.
                    let anti = self.vehicles[a].anti_clip || self.vehicles[b].anti_clip;
                    if anti {
                        if self.vehicles[b].clip_sides.len() != self.vehicles[b].nodes.len() {
                            let n = self.vehicles[b].nodes.len();
                            self.vehicles[b].clip_sides = vec![[CLIP_EMPTY; 4]; n];
                        }
                        // Outward skin: the partner's node belongs on the normal side,
                        // whatever it held or crossed.
                        let held = if self.vehicles[a].outward_skin
                            && self.vehicles[a].outward_tris.get(t).copied().unwrap_or(true)
                        {
                            Some(1.0)
                        } else {
                            clip_side(&self.vehicles[b].clip_sides[h as usize], a as u32, t as u32)
                        };
                        let va = &self.vehicles[a];
                        let vb = &self.vehicles[b];
                        let wb = &self.frame[b];
                        let crossing = || {
                            let back = |i: u32, p: Vec3| p - va.nodes[i as usize].vel * STEP;
                            let pf = ContactFrame::new(back(i0, p0), back(i1, p1), back(i2, p2));
                            let x_prev = x - vb.nodes[h as usize].vel * STEP;
                            swept_crossing(&pf, &frame, x_prev, x)
                        };
                        let vote = || {
                            let (s, e) = (
                                vb.neighbour_start[h as usize] as usize,
                                vb.neighbour_start[h as usize + 1] as usize,
                            );
                            let behind = vote_behind(
                                delta,
                                frame.n,
                                p0,
                                vb.neighbours[s..e].iter().map(|&j| wb[j as usize]),
                            );
                            if behind {
                                -1.0
                            } else {
                                1.0
                            }
                        };
                        let only = va.tri_anti_only(t);
                        // Speculative look-ahead (`speculative()`, under measurement): the
                        // band reaches as far as the pair closes this step (its relative
                        // speed at the triangle's point along the normal, from the side the
                        // node is on), as the rigid path's pairReach and closingPoints.
                        let closing_of = |side: f32| {
                            let vrel = vb.nodes[h as usize].vel
                                - (va.nodes[i1 as usize].vel * alpha.clamp(0.0, 1.0)
                                    + va.nodes[i2 as usize].vel * beta.clamp(0.0, 1.0)
                                    + va.nodes[i0 as usize].vel * gamma.clamp(0.0, 1.0));
                            (-(vrel.dot(frame.n)) * side).max(0.0)
                        };
                        let spec_on = speculative() && !only;
                        let r_eff = if spec_on {
                            r + closing_of(if delta >= 0.0 { 1.0 } else { -1.0 }) * STEP
                        } else {
                            r
                        };
                        let Some(c) =
                            clip_test((alpha, beta, gamma, delta), held, crossing, r_eff, vote)
                        else {
                            continue;
                        };
                        // A speculative pair (beyond the band): it may close to the band's
                        // edge this step and no further; none if it would not reach it.
                        let spec = if c.d > r { c.d - r } else { 0.0 };
                        if spec > 0.0 && spec >= closing_of(c.side) * STEP {
                            continue;
                        }
                        clip_record(
                            &mut self.vehicles[b].clip_sides[h as usize],
                            clip_word(a as u32, t as u32, c.side),
                        );
                        // An anti-clip-only triangle (BeamNG NONCOLLIDABLE): no force
                        // in front of its plane; behind it, pushed back to the plane.
                        let p = if spec > 0.0 { 0.0 } else { clip_depth(only, c.d, r) };
                        let behind = if only { 0.0 } else { (-c.d).max(0.0) };
                        if only && c.d >= 0.0 {
                            continue;
                        }
                        stats.hits += 1;
                        cwait = 0;
                        hits.push(Hit::node([i0, i1, i2], c.w, b, h, frame.n * c.side, p, spec, behind));
                        continue;
                    }
                    if !(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && delta.abs() <= r) {
                        continue;
                    }
                    stats.hits += 1;
                    cwait = 0;
                    let vb = &self.vehicles[b];
                    let (s, e) = (
                        vb.neighbour_start[h as usize] as usize,
                        vb.neighbour_start[h as usize + 1] as usize,
                    );
                    let wb = &self.frame[b];
                    let behind = vote_behind(
                        delta,
                        frame.n,
                        p0,
                        vb.neighbours[s..e].iter().map(|&j| wb[j as usize]),
                    );
                    let (n, delta) = if behind {
                        (-frame.n, -delta)
                    } else {
                        (frame.n, delta)
                    };
                    hits.push(Hit::node([i0, i1, i2], [alpha, beta, gamma], b, h, n, r - delta, 0.0, (-delta).max(0.0)));
                }
            }
            self.vehicles[a].contact_throttle[t] = [cwait.min(cap), cidle];
        }
        // Edge against edge and node against edge (`edge_contact`), in this owner's turn
        // and its list: one splitting, the same guards (contact design, step 4's fix).
        self.edge_turn(a, partners, &mut hits);
        if hits.is_empty() {
            return;
        }
        // Two-body hits' mass splitting (spec §10.3): per node, the two-body hits of
        // this turn it takes part in (as the contacter, or a vertex of the hit's
        // triangle); and each contacter's force before its first hit of the turn.
        let tb: Vec<bool> = self.vehicles.iter().map(|v| v.two_body_contact).collect();
        let two_body = |b: usize| tb[a] || tb[b];
        let mut split_a: Vec<u32> = Vec::new();
        let mut split_b: Vec<Vec<u32>> = vec![Vec::new(); self.vehicles.len()];
        if hits.iter().any(|x| two_body(x.b)) {
            split_a = vec![0; self.vehicles[a].nodes.len()];
            for x in hits.iter().filter(|x| two_body(x.b)) {
                if x.kind == 0 {
                    for &i in &x.tri {
                        split_a[i as usize] += 1;
                    }
                } else {
                    for (i, _) in x.owner_nodes() {
                        split_a[i as usize] += 1;
                    }
                }
                if split_b[x.b].is_empty() {
                    split_b[x.b] = vec![0; self.vehicles[x.b].nodes.len()];
                }
                split_b[x.b][x.h as usize] += 1;
                if x.t > 0.0 {
                    split_b[x.b][x.h2 as usize] += 1;
                }
            }
        }
        // §37.1: the dissipative two-body hits' sums are a staging the residual
        // pass and the work guard read; each hit's total is added once at the end
        // (explicit_beams' self-hit order).
        let on: Vec<bool> = hits
            .iter()
            .map(|x| {
                two_body(x.b)
                    && (self.vehicles[a].dissipative_contact
                        || self.vehicles[x.b].dissipative_contact)
            })
            .collect();
        let staged = on.iter().any(|&x| x);
        let snapshot: Vec<Option<(Vec<[i32; 3]>, Option<Vec<(glam::DVec3, u32)>>)>> = (0
            ..self.vehicles.len())
            .map(|v| {
                (staged && (v == a || hits.iter().any(|x| x.b == v))).then(|| {
                    (
                        self.vehicles[v].contact_acc.clone(),
                        self.vehicles[v].contact_exact.clone(),
                    )
                })
            })
            .collect();
        let mut first: Vec<Vec3> = Vec::with_capacity(hits.len());
        for (j, x) in hits.iter().enumerate() {
            let (b, h) = (x.b, x.h);
            let [i0, i1, i2] = x.tri;
            let [alpha, beta, gamma] = x.w;
            let node = self.vehicles[b].nodes[h as usize];
            let node2 = self.vehicles[b].nodes[x.h2 as usize];
            let va = &self.vehicles[a];
            let (n0, n1, n2) = (
                va.nodes[i0 as usize],
                va.nodes[i1 as usize],
                va.nodes[i2 as usize],
            );
            let cvel = if x.t > 0.0 { node.vel * (1.0 - x.t) + node2.vel * x.t } else { node.vel };
            let v = cvel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma) + x.n * (x.spec / STEP);
            let dissipative =
                self.vehicles[a].dissipative_contact || self.vehicles[b].dissipative_contact;
            // (§37's position step shares: the sub-bodies' inverse masses; none
            // outside the two-body rule.)
            let (m, fh, ims) = if two_body(b) {
                // Each node of the hit is one of its k sub-bodies (mass m/k); an
                // immovable node is of infinite mass. The triangle point's inverse
                // mass is Σ w²·k/m; the pair's relative normal acceleration from the
                // forces at the pass's start (the contacter's F/m less the point's
                // Σ w·F/m) is cancelled with their reduced mass.
                let im = |n: &Node, k: f32| {
                    if n.has(NODE_IMMOVABLE) {
                        0.0
                    } else {
                        k / n.mass
                    }
                };
                let acc = |f: Vec3, n: &Node| {
                    if n.has(NODE_IMMOVABLE) {
                        Vec3::ZERO
                    } else {
                        f / n.mass
                    }
                };
                let k = |i: u32| split_a[i as usize] as f32;
                let imh = im(&node, split_b[b][h as usize] as f32);
                let imh2 = im(&node2, split_b[b][x.h2 as usize] as f32);
                let imt = two_body_inverse_mass(
                    [im(&n1, k(i1)), im(&n2, k(i2)), im(&n0, k(i0))],
                    [alpha, beta, gamma],
                );
                let inv = if x.t > 0.0 {
                    ((1.0 - x.t) * (1.0 - x.t) * imh + x.t * x.t * imh2) + imt
                } else {
                    imh + imt
                };
                let m = if inv > 0.0 { 1.0 / inv } else { 0.0 };
                let pf = &self.pass_force[a];
                let at = acc(pf[i1 as usize], &n1) * alpha
                    + acc(pf[i2 as usize], &n2) * beta
                    + acc(pf[i0 as usize], &n0) * gamma;
                let ac = if x.t > 0.0 {
                    acc(self.pass_force[b][h as usize], &node) * (1.0 - x.t)
                        + acc(self.pass_force[b][x.h2 as usize], &node2) * x.t
                } else {
                    acc(self.pass_force[b][h as usize], &node)
                };
                (
                    m,
                    (ac - at) * m,
                    // (§37's position step: the sub-bodies' inverse masses k/m.
                    // Self contact shares by the nodes' own masses; here own masses
                    // let the 80 km/h T-bone leak 15–17 of 1,026 striker nodes, k/m
                    // 6–7.)
                    Some([imh, im(&n1, k(i1)), im(&n2, k(i2)), im(&n0, k(i0)), imh2]),
                )
            } else {
                let mt = n1.mass * alpha + n2.mass * beta + n0.mass * gamma;
                (
                    (node.mass * mt) / (node.mass + mt),
                    self.vehicles[b].force_now(h as usize),
                    None,
                )
            };
            let (f, budget) = contact_response(
                fh,
                v,
                m,
                x.n,
                STEP,
                &ContactSurface::BOX
                    .exact_if(self.vehicles[b].exact_friction)
                    .dissipative_if(dissipative),
                x.p,
                if x.t > 0.0 { node.friction.min(node2.friction) } else { node.friction },
            );
            if let (true, Some([imh, im1, im2, im0, imh2])) = ((f != Vec3::ZERO && dissipative) || x.behind > 0.0, ims) {
                // §37: the position step, a position impulse δ·m shared by the
                // sub-bodies' inverse masses (explicit_beams' self-hit order).
                let mut delta = if f != Vec3::ZERO && dissipative {
                    shift_depth(x.p, budget, fh.dot(x.n))
                } else {
                    0.0
                };
                // Push-out (contact design step 4, `push_out_rate()`): a pair already behind
                // the surface moves out at that rate even under load (the §37 step gives a
                // loaded pair none), the energy it stores in the ledger's position steps.
                let rate = push_out_rate();
                if rate > 0.0 && x.behind > 0.0 {
                    delta = delta.max((rate * STEP).min(x.behind));
                }
                if delta > 0.0 {
                    let q = delta * m;
                    if x.t > 0.0 {
                        self.vehicles[b].add_shift(h as usize, x.n * ((1.0 - x.t) * (q * imh)));
                        self.vehicles[b].add_shift(x.h2 as usize, x.n * (x.t * (q * imh2)));
                    } else {
                        self.vehicles[b].add_shift(h as usize, x.n * (q * imh));
                    }
                    let va = &mut self.vehicles[a];
                    for (i, w, imk) in [(i1, alpha, im1), (i2, beta, im2), (i0, gamma, im0)] {
                        if w > 0.0 {
                            va.add_shift(i as usize, x.n * -(w * (q * imk)));
                        }
                    }
                }
            }
            first.push(f);
            if on[j] {
                if x.t > 0.0 {
                    self.vehicles[b].add_contact(h as usize, f * (1.0 - x.t));
                    self.vehicles[b].add_contact(x.h2 as usize, f * x.t);
                } else {
                    self.vehicles[b].add_contact(h as usize, f);
                }
                let va = &mut self.vehicles[a];
                va.add_contact(i1 as usize, -(f * alpha));
                va.add_contact(i2 as usize, -(f * beta));
                va.add_contact(i0 as usize, -(f * gamma));
            } else {
                self.apply_vehicle_hit(a, x, f);
            }
        }
        if !staged {
            return;
        }
        let mut res = if vehicle_residual() {
            self.residual_vehicle_pass(a, &hits, &on, &split_a, &split_b, &first)
        } else {
            vec![None; hits.len()]
        };
        // Iterative contact (under measurement, `CONTACT_SWEEPS`, default 1: as before):
        // more residual sweeps, each seeing every hit's force so far (the staging sums),
        // so the turn's contacts are satisfied together; the work guard follows.
        if vehicle_residual() && !gauss_seidel() {
            for _ in 1..contact_sweeps() {
                let more = self.residual_vehicle_pass(a, &hits, &on, &split_a, &split_b, &first);
                if more.iter().all(|r| r.is_none()) {
                    break;
                }
                for (r, m) in res.iter_mut().zip(more) {
                    if let Some(m) = m {
                        *r = Some(r.unwrap_or(0.0) + m);
                    }
                }
            }
        }
        let with_res: Vec<Vec3> = hits
            .iter()
            .zip(&first)
            .zip(&res)
            .map(|((x, &f), r)| match r {
                Some(r) => f + x.n * *r,
                None => f,
            })
            .collect();
        let guard = self.guard_vehicle_pass(a, &hits, &on, &with_res);
        for (v, snap) in snapshot.into_iter().enumerate() {
            if let Some((acc, exact)) = snap {
                self.vehicles[v].contact_acc = acc;
                self.vehicles[v].contact_exact = exact;
            }
        }
        for (j, x) in hits.iter().enumerate() {
            if !on[j] {
                continue;
            }
            let mut f = with_res[j];
            if let Some(g) = guard[j] {
                f = f + g;
            }
            self.apply_vehicle_hit(a, x, f);
            if f != Vec3::ZERO {
                self.staged_hits.push(StagedHit {
                    a,
                    b: x.b,
                    h: x.h,
                    h2: x.h2,
                    t: x.t,
                    tri: x.tri,
                    w: x.w,
                    n: x.n,
                    f,
                });
            }
        }
    }

    /// Owner `a`'s hit `x` adds force `f` (on its contact point; its share to the
    /// triangle's nodes), with the ledger's and the contact-force log's shares.
    fn apply_vehicle_hit(&mut self, a: usize, x: &Hit, f: Vec3) {
        let (b, h) = (x.b, x.h);
        let [i0, i1, i2] = x.tri;
        let [alpha, beta, gamma] = x.w;
        if let Some(l) = self.ledger.as_mut() {
            let fnorm = x.n * f.dot(x.n);
            let ft = f - fnorm;
            let (wh, wh2) = if x.t > 0.0 { (-(1.0 - x.t), -x.t) } else { (-1.0f32, 0.0) };
            for (v, i, w) in [
                (b, h, wh),
                (a, i1, alpha),
                (a, i2, beta),
                (a, i0, gamma),
            ] {
                l.vc[v][0][i as usize] -= fnorm * w;
                l.vc[v][1][i as usize] -= ft * w;
            }
            if x.t > 0.0 {
                l.vc[b][0][x.h2 as usize] -= fnorm * wh2;
                l.vc[b][1][x.h2 as usize] -= ft * wh2;
            }
        }
        if x.t > 0.0 {
            self.vehicles[b].add_contact(h as usize, f * (1.0 - x.t));
            self.vehicles[b].add_contact(x.h2 as usize, f * x.t);
        } else {
            self.vehicles[b].add_contact(h as usize, f);
        }
        let va = &mut self.vehicles[a];
        va.add_contact(i1 as usize, -(f * alpha));
        va.add_contact(i2 as usize, -(f * beta));
        va.add_contact(i0 as usize, -(f * gamma));
        if let Some(l) = self.vehicles[a].contact_force.as_mut() {
            l.add(i1 as usize, f * alpha);
            l.add(i2 as usize, f * beta);
            l.add(i0 as usize, f * gamma);
        }
        if let Some(l) = self.vehicles[b].contact_force.as_mut() {
            if x.t > 0.0 {
                l.add(h as usize, f * (1.0 - x.t));
                l.add(x.h2 as usize, f * x.t);
            } else {
                l.add(h as usize, f);
            }
        }
    }

    /// Every vehicle's velocities before this step's node updates, when the last
    /// contact pass staged hits ([`Self::guard_final`]); empty otherwise.
    fn staged_velocities(&self) -> Vec<Vec<Vec3>> {
        if self.staged_hits.is_empty() {
            return Vec::new();
        }
        self.vehicles
            .iter()
            .map(|v| v.nodes.iter().map(|n| n.vel).collect())
            .collect()
    }

    /// §37.1's work guard on the final velocities: after every vehicle's node
    /// update and centred damping correction (`ExplicitBeams::step_nodes`), each
    /// hit the last contact pass staged ([`StagedHit`], in its order) is checked
    /// at the pair's realised mean velocity over the step, (v₀ + v′)/2, with the
    /// forces its nodes actually took (the fixed-point sums' rounding of its
    /// shares, `ExplicitBeams::applied_contact`). Where its work is positive its
    /// force is scaled by s ∈ [0, 1) to zero work, s solving
    /// s·q·(u + s·h·q·Σ w²/m) = 0 with u the pair's mean relative velocity less
    /// the hit's own share: its nodes' velocities lose (1 − s)·h·q·w/m and their
    /// positions h times that (the node update's x += h·v), equal and opposite
    /// (momentum kept).
    ///
    /// The guard at the contact pass (`guard_vehicle_pass`) predicts the end
    /// velocity as v + h·F/m; the centred damping's correction after the node
    /// update moves it by h·c/m, and the shared nodes of several hits, the
    /// residual pass and the sums' rounding move it too. Measured: a stuck striker
    /// node on the struck door in the showcase's 80 km/h T-bone did +0.5–1 mJ a
    /// step for 0.1 s with the earlier guard alone (`tests/vehicle_contact_work.rs`).
    ///
    /// With the ledger, the scaled part of each force leaves its vehicle-contact
    /// terms and the forces the step started with (`before`), so the ledger
    /// still closes.
    fn guard_final(
        &mut self,
        v0: &[Vec<Vec3>],
        mut ledger: Option<(&mut EnergyLedger, &mut Vec<Vec<(Vec3, Vec3, Vec3)>>)>,
    ) {
        if self.staged_hits.is_empty() || !self.final_guard {
            return;
        }
        let hits = std::mem::take(&mut self.staged_hits);
        let h = STEP;
        for x in &hits {
            let (va, vb) = (&self.vehicles[x.a], &self.vehicles[x.b]);
            let im = |v: &ExplicitBeams, i: u32| {
                let n = &v.nodes[i as usize];
                if n.has(NODE_IMMOVABLE) || v.is_retired(i as usize) {
                    0.0
                } else {
                    1.0 / n.mass
                }
            };
            // The shares each node took: +f on the contacter, −f·w on the triangle.
            let (qb, qb2) = if x.t > 0.0 {
                (vb.applied_contact(x.f * (1.0 - x.t)), vb.applied_contact(x.f * x.t))
            } else {
                (vb.applied_contact(x.f), Vec3::ZERO)
            };
            let qa = [0, 1, 2].map(|k| va.applied_contact(-(x.f * x.w[k])));
            let mut work = qb.dot(v0[x.b][x.h as usize] + vb.nodes[x.h as usize].vel);
            let mut lose = qb.dot(qb) * im(vb, x.h);
            if x.t > 0.0 {
                work += qb2.dot(v0[x.b][x.h2 as usize] + vb.nodes[x.h2 as usize].vel);
                lose += qb2.dot(qb2) * im(vb, x.h2);
            }
            for k in 0..3 {
                let i = x.tri[k] as usize;
                work += qa[k].dot(v0[x.a][i] + va.nodes[i].vel);
                lose += qa[k].dot(qa[k]) * im(va, x.tri[k]);
            }
            // `work` is 2·W/h at the full force. At s of it the nodes keep s of the
            // hit's velocity change: 2·W(s)/h = s·(work − (1 − s)·h·lose).
            if work <= 0.0 || lose <= 0.0 {
                continue;
            }
            // Zero work at r = 1 − s = work/(h·lose) (all of it past 1). Taken as r
            // itself, not 1 − s: a stuck contact needs r ~ 10⁻⁴, which 1 − s
            // in single precision leaves with a relative error of 10⁻³.
            let r = (work / (h * lose)).min(1.0);
            let dvb = qb * (-r * h * im(vb, x.h));
            let dvb2 = qb2 * (-r * h * im(vb, x.h2));
            self.vehicles[x.b].nudge(x.h as usize, dvb);
            if x.t > 0.0 {
                self.vehicles[x.b].nudge(x.h2 as usize, dvb2);
            }
            for k in 0..3 {
                let i = x.tri[k] as usize;
                let dva = qa[k] * (-r * h * im(&self.vehicles[x.a], x.tri[k]));
                self.vehicles[x.a].nudge(i, dva);
            }
            if let Some((l, before)) = ledger.as_mut() {
                let fnorm = x.n * x.f.dot(x.n);
                let ft = x.f - fnorm;
                let wh = if x.t > 0.0 { -(1.0 - x.t) } else { -1.0f32 };
                for (v, i, wk) in [
                    (x.b, x.h, wh),
                    (x.a, x.tri[0], x.w[0]),
                    (x.a, x.tri[1], x.w[1]),
                    (x.a, x.tri[2], x.w[2]),
                    (x.b, x.h2, if x.t > 0.0 { -x.t } else { 0.0 }),
                ] {
                    if wk == 0.0 && v == x.b && i == x.h2 && x.t == 0.0 {
                        continue;
                    }
                    let i = i as usize;
                    l.vc[v][0][i] += fnorm * (wk * r);
                    l.vc[v][1][i] += ft * (wk * r);
                    before[v][i].2 += x.f * (wk * r);
                }
            }
        }
        // (The buffer back, empty, for the next contact pass.)
        self.staged_hits = hits;
        self.staged_hits.clear();
    }

    /// §37.1's work guard over owner `a`'s staged hits (`on`; forces `first`,
    /// each with its residual), as `ExplicitBeams::friction_guard_pass` takes self
    /// hits: at the pair's realised mean velocity over the step (the forces now,
    /// the turn's staging sums with them, on the nodes' own masses) a hit whose
    /// force would do positive work is scaled to zero work; returns each hit's
    /// −(1 − s)·t. All found from the same sums.
    fn guard_vehicle_pass(&self, a: usize, hits: &[Hit], on: &[bool], first: &[Vec3]) -> Vec<Option<Vec3>> {
        let mut out = vec![None; hits.len()];
        if !crate::explicit_beams::friction_guard() {
            return out;
        }
        let node = |v: usize, i: u32| &self.vehicles[v].nodes[i as usize];
        let acc = |v: usize, i: u32| {
            let n = node(v, i);
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                self.vehicles[v].force_now(i as usize) / n.mass
            }
        };
        let im = |v: usize, i: u32| {
            let n = node(v, i);
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                1.0 / n.mass
            }
        };
        for (j, x) in hits.iter().enumerate() {
            let t = first[j];
            if !on[j] || t == Vec3::ZERO {
                continue;
            }
            let [i0, i1, i2] = x.tri;
            let [alpha, beta, gamma] = x.w;
            let (cv, ca, ci) = if x.t > 0.0 {
                (
                    node(x.b, x.h).vel * (1.0 - x.t) + node(x.b, x.h2).vel * x.t,
                    acc(x.b, x.h) * (1.0 - x.t) + acc(x.b, x.h2) * x.t,
                    (1.0 - x.t) * (1.0 - x.t) * im(x.b, x.h) + x.t * x.t * im(x.b, x.h2),
                )
            } else {
                (node(x.b, x.h).vel, acc(x.b, x.h), im(x.b, x.h))
            };
            let v = cv
                - (node(a, i1).vel * alpha + node(a, i2).vel * beta + node(a, i0).vel * gamma);
            let ac = ca - (acc(a, i1) * alpha + acc(a, i2) * beta + acc(a, i0) * gamma);
            let inv = ci
                + two_body_inverse_mass([im(a, i1), im(a, i2), im(a, i0)], [alpha, beta, gamma]);
            let a_rest = ac - t * inv;
            let aa = t.dot(v + v + a_rest * STEP);
            let bb = STEP * t.dot(t) * inv;
            if aa + bb > 0.0 && bb > 0.0 {
                let s = if aa < 0.0 { (-aa / bb).min(1.0) } else { 0.0 };
                out[j] = Some(t * -(1.0 - s));
            }
        }
        out
    }

    /// §37.1's residual pass over owner `a`'s staged hits (`on`), as
    /// `ExplicitBeams::residual_self_pass` takes self hits: each pair approaching
    /// and still approaching at the forces now (the turn's staging sums with them)
    /// takes a normal force r with the first pass's reduced mass (its counts
    /// `split_a`, `split_b`), into the staging sums; returns each hit's r.
    /// One hit's Gauss–Seidel residual: the normal force (> 0) that stops its pair's
    /// approach over the step at the forces now, on the nodes' own masses; None if it is
    /// not approaching or would not be.
    fn residual_of(&self, a: usize, x: &Hit) -> Option<f32> {
        self.residual_signed(a, x).filter(|&r| r > 0.0)
    }

    /// The normal force change that brings hit `x`'s pair's normal velocity at the end of
    /// the step to zero, at the forces now (signed: negative takes back an excess).
    fn residual_signed(&self, a: usize, x: &Hit) -> Option<f32> {
        let acc = |v: usize, i: u32| {
            let n = &self.vehicles[v].nodes[i as usize];
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                self.vehicles[v].force_now(i as usize) / n.mass
            }
        };
        let im = |n: &Node| if n.has(NODE_IMMOVABLE) { 0.0 } else { 1.0 / n.mass };
        let [i0, i1, i2] = x.tri;
        let [alpha, beta, gamma] = x.w;
        let va = &self.vehicles[a];
        let node = &self.vehicles[x.b].nodes[x.h as usize];
        let node2 = &self.vehicles[x.b].nodes[x.h2 as usize];
        let (n0, n1, n2) = (&va.nodes[i0 as usize], &va.nodes[i1 as usize], &va.nodes[i2 as usize]);
        let cvel = if x.t > 0.0 { node.vel * (1.0 - x.t) + node2.vel * x.t } else { node.vel };
        let v = cvel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma) + x.n * (x.spec / STEP);
        let at = acc(a, i1) * alpha + acc(a, i2) * beta + acc(a, i0) * gamma;
        let cacc = if x.t > 0.0 { acc(x.b, x.h) * (1.0 - x.t) + acc(x.b, x.h2) * x.t } else { acc(x.b, x.h) };
        let an = (cacc - at).dot(x.n);
        let vn = v.dot(x.n);
        let imc = if x.t > 0.0 {
            (1.0 - x.t) * (1.0 - x.t) * im(node) + x.t * x.t * im(node2)
        } else {
            im(node)
        };
        let inv = imc + two_body_inverse_mass([im(n1), im(n2), im(n0)], x.w);
        if inv <= 0.0 {
            return None;
        }
        let m = 1.0 / inv;
        Some(-(an * m) - vn * m / STEP)
    }

    fn residual_vehicle_pass(
        &mut self,
        a: usize,
        hits: &[Hit],
        on: &[bool],
        split_a: &[u32],
        split_b: &[Vec<u32>],
        first: &[Vec3],
    ) -> Vec<Option<f32>> {
        let acc = |v: usize, i: u32| {
            let n = &self.vehicles[v].nodes[i as usize];
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                self.vehicles[v].force_now(i as usize) / n.mass
            }
        };
        let im = |n: &Node, k: u32| {
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                k as f32 / n.mass
            }
        };
        if gauss_seidel() {
            // Gauss–Seidel (`CONTACT_GS=1`, under measurement): each still-approaching hit,
            // in order, takes the force that stops its pair at the forces now, every hit
            // before it included, on the nodes' own masses (no split: the others' pushes
            // are already in the sums), applied at once.
            // Each hit's normal force so far (the first pass's), which a sweep may raise or
            // lower but never take below zero (sequential impulses' accumulated clamp: an
            // excess is taken back, so a pair is not pushed apart to bounce back in).
            let mut out: Vec<Option<f32>> = vec![None; hits.len()];
            for j in (0..contact_sweeps() as usize).flat_map(|_| 0..hits.len()) {
                if !on[j] {
                    continue;
                }
                let x = &hits[j];
                let lam0 = first.get(j).map_or(0.0, |f| f.dot(x.n)).max(0.0) + out[j].unwrap_or(0.0);
                let Some(d) = self.residual_signed(a, x) else { continue };
                let lam = (lam0 + d).max(0.0);
                let r = lam - lam0;
                if r == 0.0 {
                    continue;
                }
                out[j] = Some(out[j].unwrap_or(0.0) + r);
                let f = x.n * r;
                if x.t > 0.0 {
                    self.vehicles[x.b].add_contact(x.h as usize, f * (1.0 - x.t));
                    self.vehicles[x.b].add_contact(x.h2 as usize, f * x.t);
                } else {
                    self.vehicles[x.b].add_contact(x.h as usize, f);
                }
                let va = &mut self.vehicles[a];
                va.add_contact(x.tri[1] as usize, -(f * x.w[0]));
                va.add_contact(x.tri[2] as usize, -(f * x.w[1]));
                va.add_contact(x.tri[0] as usize, -(f * x.w[2]));
            }
            return out;
        }
        let mut out = vec![None; hits.len()];
        for (j, x) in hits.iter().enumerate() {
            if !on[j] {
                continue;
            }
            let [i0, i1, i2] = x.tri;
            let [alpha, beta, gamma] = x.w;
            let va = &self.vehicles[a];
            let node = &self.vehicles[x.b].nodes[x.h as usize];
            let node2 = &self.vehicles[x.b].nodes[x.h2 as usize];
            let (n0, n1, n2) = (
                &va.nodes[i0 as usize],
                &va.nodes[i1 as usize],
                &va.nodes[i2 as usize],
            );
            let cvel = if x.t > 0.0 { node.vel * (1.0 - x.t) + node2.vel * x.t } else { node.vel };
            let v = cvel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma) + x.n * (x.spec / STEP);
            let at = acc(a, i1) * alpha + acc(a, i2) * beta + acc(a, i0) * gamma;
            let cacc = if x.t > 0.0 { acc(x.b, x.h) * (1.0 - x.t) + acc(x.b, x.h2) * x.t } else { acc(x.b, x.h) };
            let an = (cacc - at).dot(x.n);
            let vn = v.dot(x.n);
            if !(vn < 0.0 && an * STEP + vn < 0.0) {
                continue;
            }
            let imc = if x.t > 0.0 {
                (1.0 - x.t) * (1.0 - x.t) * im(node, split_b[x.b][x.h as usize])
                    + x.t * x.t * im(node2, split_b[x.b][x.h2 as usize])
            } else {
                im(node, split_b[x.b][x.h as usize])
            };
            let inv = imc
                + two_body_inverse_mass(
                    [
                        im(n1, split_a[i1 as usize]),
                        im(n2, split_a[i2 as usize]),
                        im(n0, split_a[i0 as usize]),
                    ],
                    x.w,
                );
            if inv <= 0.0 {
                continue;
            }
            let m = 1.0 / inv;
            let r = -(an * m) - vn * m / STEP;
            if r > 0.0 {
                out[j] = Some(r);
            }
        }
        for (j, x) in hits.iter().enumerate() {
            let Some(r) = out[j] else { continue };
            let f = x.n * r;
            let [i0, i1, i2] = x.tri;
            let [alpha, beta, gamma] = x.w;
            if x.t > 0.0 {
                self.vehicles[x.b].add_contact(x.h as usize, f * (1.0 - x.t));
                self.vehicles[x.b].add_contact(x.h2 as usize, f * x.t);
            } else {
                self.vehicles[x.b].add_contact(x.h as usize, f);
            }
            let va = &mut self.vehicles[a];
            va.add_contact(i1 as usize, -(f * alpha));
            va.add_contact(i2 as usize, -(f * beta));
            va.add_contact(i0 as usize, -(f * gamma));
        }
        out
    }

}

/// A vehicle's per-node forces of the last step by source, for the ledger.
#[derive(Clone, Debug, Default)]
struct ForceSplit {
    trace: ForceTrace,
}

impl ForceSplit {
    fn new(n: usize) -> Self {
        let mut trace = ForceTrace::default();
        for p in &mut trace.parts {
            p.resize(n, Vec3::ZERO);
        }
        Self { trace }
    }
}

/// The energy ledger of an [`ExplicitBeamsWorld`] (CPU, opt-in): the state's
/// kinetic, gravitational and elastic energy, and the work every force source
/// has done on the nodes since it was enabled (J). The node pass integrates
/// v' = v + h·F/m, x' = x + h·v'; a force's work is F·(v + v')/2·h per step, so the
/// sources' work sums to the kinetic energy's change exactly (`discretisation`
/// stays 0; the positions' step differs from it by ½m|v' − v|², the symplectic
/// step's term, which the elastic energy's change carries). Sources: gravity; the air
/// (drag, turbulence, wings; gravity taken out); beams (every beam record); self-
/// collision; slide nodes; vehicle contact (normal along each hit's normal, and
/// tangential: friction); and the node pass's own contacts (ground, boxes, static
/// shapes, rigid partners: m·(v' − v)/h less the accumulated force), split into
/// vertical and horizontal. Elastic energy is every conservative record's stored
/// energy ([`elastic_energy`]: the linear beam kinds, bounded beams across their
/// bounds, torsion elements, pressure groups' gas); aerodynamic and trigger records
/// store none. The beams' dissipation, −(W_beams + ΔU), is then damping, yield and
/// fracture only, and a rise of kinetic + gravitational + elastic is created energy.
#[derive(Clone, Debug, Default)]
pub struct EnergyLedger {
    pub kinetic: f64,
    pub gravity_pe: f64,
    pub elastic: f64,
    /// Kinetic + gravitational + elastic when enabled.
    pub start: f64,
    pub elastic_start: f64,
    pub w_gravity: f64,
    pub w_air: f64,
    pub w_beams: f64,
    pub w_self_contact: f64,
    pub w_slides: f64,
    pub w_contact_normal: f64,
    pub w_contact_friction: f64,
    pub w_ground_vertical: f64,
    pub w_ground_horizontal: f64,
    /// The contact position steps' energy (§37, self- and vehicle contact): per
    /// step Σ −F·Δx over the nodes they moved, F the node's air, beam and slide
    /// forces of the step (the first-order change of the potential and elastic
    /// energy they store). Not a work on the velocities: not in kinetic_change.
    pub w_contact_shift: f64,
    /// The ground's position steps' energy (§37), as [`Self::w_contact_shift`]
    /// (−F·Δx over the nodes they lifted).
    pub w_ground_shift: f64,
    pub discretisation: f64,
    prev: Vec<ForceSplit>,
    vc: Vec<[Vec<Vec3>; 2]>,
    started: bool,
}

impl EnergyLedger {
    /// Kinetic + gravitational + elastic now.
    pub fn mechanical(&self) -> f64 {
        self.kinetic + self.gravity_pe + self.elastic
    }

    /// The beams' dissipation so far (damping, yield, fracture, and the
    /// non-linear records' storage): −(W_beams + ΔU).
    pub fn beam_dissipation(&self) -> f64 {
        -(self.w_beams + self.elastic - self.elastic_start)
    }

    /// The work of every source so far, less the discretisation term: the kinetic
    /// energy's change, to rounding.
    pub fn kinetic_change(&self) -> f64 {
        self.w_gravity
            + self.w_air
            + self.w_beams
            + self.w_self_contact
            + self.w_slides
            + self.w_contact_normal
            + self.w_contact_friction
            + self.w_ground_vertical
            + self.w_ground_horizontal
            - self.discretisation
    }

    /// Vehicle `v`'s node `i`: the vehicle-contact force on it this step
    /// (normal and friction; read after [`ExplicitBeamsWorld::step`]).
    pub fn vehicle_contact(&self, v: usize, i: usize) -> Vec3 {
        self.vc
            .get(v)
            .map_or(Vec3::ZERO, |c| c[0].get(i).copied().unwrap_or_default() + c[1].get(i).copied().unwrap_or_default())
    }

    fn account(&mut self, k: usize, v: &ExplicitBeams, before: &[(Vec3, Vec3, Vec3)]) {
        let prev = &self.prev[k].trace.parts;
        let vc = &self.vc[k];
        for (i, (n, &(x0, v0, f_acc))) in v.nodes.iter().zip(before).enumerate() {
            if n.has(NODE_IMMOVABLE) || i >= prev[0].len() {
                continue;
            }
            // The work over the step at its mean velocity, (v + v')/2·h: the
            // kinetic energy's change exactly (the position step v'·h would add
            // ½m|v' − v|², large for the stiff modes of a crash).
            let _ = x0;
            let dx = ((n.vel + v0) * (0.5 * STEP)).as_dvec3();
            let w = |f: Vec3| f.as_dvec3().dot(dx);
            let g = Vec3::new(0.0, n.mass * GRAVITY, 0.0);
            let dv = n.vel - v0;
            let f_tot = dv * (n.mass / STEP);
            let ground = f_tot - f_acc;
            self.w_gravity += w(g);
            self.w_air += w(prev[0][i] - g);
            self.w_beams += w(prev[1][i]);
            self.w_self_contact += w(prev[2][i]);
            self.w_slides += w(prev[3][i]);
            self.w_contact_normal += w(vc[0][i]);
            self.w_contact_friction += w(vc[1][i]);
            self.w_ground_vertical += w(Vec3::new(0.0, ground.y, 0.0));
            self.w_ground_horizontal += w(Vec3::new(ground.x, 0.0, ground.z));
            let held = (prev[0][i] + prev[1][i] + prev[3][i]).as_dvec3();
            if let Some(&d) = v.shift_applied.get(i) {
                self.w_contact_shift -= held.dot(d.as_dvec3());
            }
            if let Some(&y) = v.ground_lift.get(i) {
                self.w_ground_shift -= held.y * f64::from(y);
            }
        }
        if let Some(t) = v.trace.as_ref() {
            self.prev[k].trace = (**t).clone();
        }
    }

    fn vc_reset(&mut self, vehicles: &[ExplicitBeams]) {
        for (vc, v) in self.vc.iter_mut().zip(vehicles) {
            for a in vc.iter_mut() {
                a.clear();
                a.resize(v.nodes.len(), Vec3::ZERO);
            }
        }
    }

    fn refresh(&mut self, vehicles: &[ExplicitBeams]) {
        let (mut ke, mut pe, mut u) = (0.0f64, 0.0f64, 0.0f64);
        for v in vehicles {
            for n in &v.nodes {
                let m = f64::from(n.mass);
                ke += 0.5 * m * f64::from(n.vel.length_squared());
                pe -= m * f64::from(GRAVITY) * f64::from(v.origin.y + n.pos.y);
            }
            u += elastic_energy(v);
        }
        (self.kinetic, self.gravity_pe, self.elastic) = (ke, pe, u);
        if !self.started {
            self.elastic_start = u;
            self.started = true;
        }
    }
}

/// The stored energy (J) of `v`'s conservative records (see [`EnergyLedger`]): ½k·e²
/// of the linear beam kinds (plain, support in compression, ropes in tension,
/// couplers, anisotropic at their base rate, LBEAMs at their arm-following rest);
/// bounded beams' ∫k(x)·x dx across their bound zone and limit spring; torsion
/// elements' ½k·wrap(φ − φ₀)²; and pressure groups' gas, ∫ p_gauge dV from the
/// spawn volume (isothermal, capped as the force is). With every stored term in, a
/// rise of kinetic + gravitational + this is energy the step created.
pub fn elastic_energy(v: &ExplicitBeams) -> f64 {
    let mut u = 0.0f64;
    for (j, b) in v.beams.iter().enumerate() {
        if b.broken {
            continue;
        }
        if b.kind == ExplicitKind::JbeamTorsion {
            let (c, d) = (v.beams[j + 1].a, v.beams[j + 1].b);
            let x = [b.a, b.b, c, d].map(|i| v.nodes[i as usize].pos);
            let dphi = crate::explicit_beams::wrap_angle(crate::explicit_beams::dihedral(x) - b.rest);
            u += 0.5 * f64::from(b.k) * f64::from(dphi) * f64::from(dphi);
            continue;
        }
        if b.kind == ExplicitKind::JbeamPressureGroup {
            u += gas_energy(v, j);
            continue;
        }
        let (pa, pb) = (v.nodes[b.a as usize].pos, v.nodes[b.b as usize].pos);
        let len = pa.distance(pb);
        let e = match b.kind {
            ExplicitKind::Plain
            | ExplicitKind::LimitedSpring
            | ExplicitKind::Shock2
            | ExplicitKind::JbeamPlain
            | ExplicitKind::JbeamCoupler
            | ExplicitKind::JbeamAniso => len - b.rest,
            ExplicitKind::JbeamBounded => {
                u += bounded_energy(b, len - b.rest);
                continue;
            }
            ExplicitKind::Support | ExplicitKind::JbeamSupport => (len - b.rest).min(0.0),
            ExplicitKind::Rope | ExplicitKind::JbeamRope => (len - b.rest).max(0.0),
            ExplicitKind::JbeamLbeam => {
                let c = v.beam_ext[v.ext_of[j] as usize - 1].index(0) as usize;
                let (a, bb) = (pa.distance(v.nodes[c].pos), pb.distance(v.nodes[c].pos));
                len - (a * a + bb * bb - 2.0 * a * bb * b.rest).max(0.0).sqrt()
            }
            _ => 0.0,
        };
        u += 0.5 * f64::from(b.k) * f64::from(e) * f64::from(e);
    }
    u
}

/// A bounded beam's spring energy at extension `e`: ∫₀ᵉ k(x)·x dx with the rate
/// `jbeam_force_filtered` gives it (the base rate inside the bounds, blending to the
/// limit rate across the bound zone past them).
fn bounded_energy(b: &crate::explicit_beams::Beam, e: f32) -> f64 {
    let (long, short) = (b.long_bound * b.rest, b.short_bound * b.rest);
    let rate = |x: f64| -> f64 {
        let o = if x > f64::from(long) {
            x - f64::from(long)
        } else if x < -f64::from(short) {
            -x - f64::from(short)
        } else {
            0.0
        };
        let k = f64::from(b.k);
        if o > 0.0 {
            let zone = f64::from(b.plastic);
            let s = if zone > 0.0 { (o / zone).min(1.0) } else { 1.0 };
            k + (f64::from(b.limit_k) - k) * s
        } else {
            k
        }
    };
    let e = f64::from(e);
    let inside = if e >= 0.0 { f64::from(long) } else { -f64::from(short) };
    if (e >= 0.0 && e <= inside) || (e < 0.0 && e >= inside) {
        return 0.5 * f64::from(b.k) * e * e;
    }
    // Inside the bound at the base rate, then Simpson's rule beyond it.
    let mut u = 0.5 * f64::from(b.k) * inside * inside;
    let n = 64;
    let h = (e - inside) / n as f64;
    let f = |x: f64| rate(x) * x;
    let mut acc = f(inside) + f(e);
    for i in 1..n {
        acc += f(inside + h * i as f64) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    u += acc * h / 3.0;
    u
}

/// A pressure group's gas energy: ∫_V^{V₀} p(V') dV', p the gauge pressure the group
/// applies (`pressure_group_gauge`: isothermal p_abs = p_abs,0·V₀/V, the volume held
/// at 1 % of V₀ or more, the gauge capped).
fn gas_energy(v: &ExplicitBeams, j: usize) -> f64 {
    let ext = &v.beam_ext[v.ext_of[j] as usize - 1];
    let c = v.nodes[v.beams[j].a as usize].pos;
    let first = ext.index(0) as usize;
    let count = ext.index(1) as usize;
    let mut vol = 0.0f64;
    for t in 0..count {
        let r = first + 2 * t;
        let x0 = (v.nodes[v.beams[r].a as usize].pos - c).as_dvec3();
        let x1 = (v.nodes[v.beams[r].b as usize].pos - c).as_dvec3();
        let x2 = (v.nodes[v.beams[r + 1].a as usize].pos - c).as_dvec3();
        vol += x0.dot(x1.cross(x2));
    }
    let vol = vol / 6.0;
    let v0 = f64::from(ext.e1[2]);
    let cst = f64::from(ext.e1[3]) * v0;
    let atm = f64::from(crate::explicit_jbeam_ext::ATMOSPHERE);
    let cap = f64::from(ext.e2[0]);
    let p = |x: f64| (cst / x.max(0.01 * v0) - atm).min(cap);
    // Breakpoints: the volume floor and where the cap starts.
    let (lo, hi, sign) = if vol < v0 { (vol, v0, 1.0) } else { (v0, vol, -1.0) };
    let mut cuts = vec![lo, hi, 0.01 * v0];
    if cap.is_finite() {
        cuts.push(cst / (atm + cap));
    }
    cuts.retain(|&x| x >= lo && x <= hi);
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let mut u = 0.0;
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let m = 0.5 * (a + b);
        u += if m < 0.01 * v0 || p(m) >= cap {
            p(m) * (b - a)
        } else {
            cst * (b / a).ln() - atm * (b - a)
        };
    }
    sign * u
}

/// One hit of an owner's walk: the triangle's nodes (n₀, n₁, n₂) and weights
/// (α, β, γ), the partner and its contact point, the normal after the side vote and
/// the penetration r − δ.
struct Hit {
    tri: [u32; 3],
    w: [f32; 3],
    b: usize,
    h: u32,
    n: Vec3,
    p: f32,
    /// Speculative: the gap past the band this pair may still close this step (m; 0 in
    /// the band). The response targets the normal velocity −spec/h instead of 0.
    spec: f32,
    /// How far behind the surface the contact point is (m; 0 in front).
    behind: f32,
    /// A second contacter node and its weight (edge against edge: the contacter's
    /// point is (1 − t)·h + t·h2); t = 0: node h alone (h2 = h), every path as before.
    h2: u32,
    t: f32,
    /// 0 node against face, 1 edge against edge, 2 node against edge (the owner's
    /// `tri` an edge [i, j, j], weights [s, 0, 1 − s]).
    kind: u8,
}

impl Hit {
    /// A node-against-face hit (one contacter node).
    fn node(tri: [u32; 3], w: [f32; 3], b: usize, h: u32, n: Vec3, p: f32, spec: f32, behind: f32) -> Self {
        Hit { tri, w, b, h, n, p, spec, behind, h2: h, t: 0.0, kind: 0 }
    }
    /// The owner's nodes with their weights, each once and only with a weight (edge
    /// kinds; a face's three as they are).
    fn owner_nodes(&self) -> Vec<(u32, f32)> {
        let [i0, i1, i2] = self.tri;
        let [alpha, beta, gamma] = self.w;
        if self.kind == 0 {
            return vec![(i1, alpha), (i2, beta), (i0, gamma)];
        }
        [(i1, alpha), (i2, beta), (i0, gamma)].into_iter().filter(|x| x.1 > 0.0).collect()
    }
}

/// Vehicle contact's residual pass (§37.1; on, `VEHICLE_RESIDUAL=0` a measurement
/// switch: off). Measured, the 80 km/h T-bone's leak at the five offsets: off
/// 8 / 9 / 5 / 2 / 2 of 1,026 striker nodes, on 3 / 4 / 6 / 0 / 2; the contact-work
/// invariant holds either way.
fn vehicle_residual() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("VEHICLE_RESIDUAL").is_ok_and(|v| v == "0")
            && crate::explicit_beams::residual_on()
    })
}

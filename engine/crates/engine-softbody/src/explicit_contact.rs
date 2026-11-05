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
    contact_force, ContactSurface, ExplicitBeams, ExplicitKind, ForceTrace, Node, GRAVITY,
    NODE_IMMOVABLE, STEP,
};
use glam::Vec3;

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
    /// Scratch: every node's force at the contact pass's start (two-body hits).
    pass_force: Vec<Vec<Vec3>>,
    /// The energy ledger (None: off; [`Self::enable_ledger`]).
    pub ledger: Option<EnergyLedger>,
    boxes: Vec<(Vec3, Vec3)>,
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
            ledger: None,
            pass_force: Vec::new(),
            boxes: Vec::new(),
        }
    }

    /// One frame: every vehicle's wing check and origin rebase, then `steps` world
    /// steps.
    pub fn frame(&mut self, steps: u32) {
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

    /// One step: every vehicle's own step in vehicle order, then the contact pass.
    pub fn step(&mut self) {
        let Some(mut ledger) = self.ledger.take() else {
            for v in &mut self.vehicles {
                v.step();
            }
            self.steps += 1;
            if self.contact && self.vehicles.len() > 1 {
                self.contact_pass();
            }
            return;
        };
        for (k, v) in self.vehicles.iter_mut().enumerate() {
            let before: Vec<(Vec3, Vec3, Vec3)> = v
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (v.origin + n.pos, n.vel, v.force_now(i)))
                .collect();
            v.step();
            ledger.account(k, v, &before);
        }
        self.steps += 1;
        ledger.vc_reset(&self.vehicles);
        self.ledger = Some(ledger);
        if self.contact && self.vehicles.len() > 1 {
            self.contact_pass();
        }
        if let Some(l) = self.ledger.as_mut() {
            l.refresh(&self.vehicles);
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
        let mut stats = ContactStats::default();
        let mut partners: Vec<usize> = Vec::new();
        for a in 0..nv {
            partners.clear();
            let (alo, ahi) = self.boxes[a];
            for b in 0..nv {
                if b == a {
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

    /// Owner `a`'s triangle loop against `partners` (§3.3, §4.2, §5, §6): the walk
    /// (throttles, boxes, hit tests and side votes, all from positions), then the
    /// hits' forces in the walk's order (triangles in list order, candidates in
    /// order). The walk reads no force, so the two phases are the specification's
    /// single loop.
    fn triangle_loop(&mut self, a: usize, partners: &[usize], stats: &mut ContactStats) {
        let r = self.vehicles[a].collision_range;
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
            let wa = &self.world[a];
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
                let wb = &self.world[b];
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
                    let x = self.world[b][h as usize];
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
                        let held =
                            clip_side(&self.vehicles[b].clip_sides[h as usize], a as u32, t as u32);
                        let va = &self.vehicles[a];
                        let vb = &self.vehicles[b];
                        let wb = &self.world[b];
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
                        let Some(c) =
                            clip_test((alpha, beta, gamma, delta), held, crossing, r, vote)
                        else {
                            continue;
                        };
                        clip_record(
                            &mut self.vehicles[b].clip_sides[h as usize],
                            clip_word(a as u32, t as u32, c.side),
                        );
                        // An anti-clip-only triangle (BeamNG NONCOLLIDABLE): no force
                        // in front of its plane; behind it, pushed back to the plane.
                        let p = clip_depth(only, c.d, r);
                        if only && c.d >= 0.0 {
                            continue;
                        }
                        stats.hits += 1;
                        cwait = 0;
                        hits.push(Hit {
                            tri: [i0, i1, i2],
                            w: c.w,
                            b,
                            h,
                            n: frame.n * c.side,
                            p,
                        });
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
                    let wb = &self.world[b];
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
                    hits.push(Hit {
                        tri: [i0, i1, i2],
                        w: [alpha, beta, gamma],
                        b,
                        h,
                        n,
                        p: r - delta,
                    });
                }
            }
            self.vehicles[a].contact_throttle[t] = [cwait.min(cap), cidle];
        }
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
                for &i in &x.tri {
                    split_a[i as usize] += 1;
                }
                if split_b[x.b].is_empty() {
                    split_b[x.b] = vec![0; self.vehicles[x.b].nodes.len()];
                }
                split_b[x.b][x.h as usize] += 1;
            }
        }
        for x in &hits {
            let (b, h) = (x.b, x.h);
            let [i0, i1, i2] = x.tri;
            let [alpha, beta, gamma] = x.w;
            let node = self.vehicles[b].nodes[h as usize];
            let va = &self.vehicles[a];
            let (n0, n1, n2) = (
                va.nodes[i0 as usize],
                va.nodes[i1 as usize],
                va.nodes[i2 as usize],
            );
            let v = node.vel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma);
            let (m, fh) = if two_body(b) {
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
                let imt = two_body_inverse_mass(
                    [im(&n1, k(i1)), im(&n2, k(i2)), im(&n0, k(i0))],
                    [alpha, beta, gamma],
                );
                let inv = imh + imt;
                let m = if inv > 0.0 { 1.0 / inv } else { 0.0 };
                let pf = &self.pass_force[a];
                let at = acc(pf[i1 as usize], &n1) * alpha
                    + acc(pf[i2 as usize], &n2) * beta
                    + acc(pf[i0 as usize], &n0) * gamma;
                (m, (acc(self.pass_force[b][h as usize], &node) - at) * m)
            } else {
                let mt = n1.mass * alpha + n2.mass * beta + n0.mass * gamma;
                (
                    (node.mass * mt) / (node.mass + mt),
                    self.vehicles[b].force_now(h as usize),
                )
            };
            let f = contact_force(
                fh,
                v,
                m,
                x.n,
                STEP,
                &ContactSurface::BOX,
                x.p,
                node.friction,
            );
            if let Some(l) = self.ledger.as_mut() {
                let fnorm = x.n * f.dot(x.n);
                let ft = f - fnorm;
                for (v, i, w) in [
                    (b, h, -1.0f32),
                    (a, i1, alpha),
                    (a, i2, beta),
                    (a, i0, gamma),
                ] {
                    l.vc[v][0][i as usize] -= fnorm * w;
                    l.vc[v][1][i as usize] -= ft * w;
                }
            }
            self.vehicles[b].add_contact(h as usize, f);
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
                l.add(h as usize, f);
            }
        }
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
/// vertical and horizontal. Elastic energy is the state's ½k·e² of the linear
/// beam kinds (plain, support in compression, rope in tension, couplers, bounded
/// and anisotropic at their base rate, LBEAMs at their arm-following rest);
/// pressure, torsion, aerodynamic and trigger records store none here, so their
/// changes count in the beams' dissipation, −(W_beams + ΔU): damping, yield,
/// fracture.
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

/// The elastic energy (J) of `v`'s linear beam kinds (see [`EnergyLedger`]).
pub fn elastic_energy(v: &ExplicitBeams) -> f64 {
    let mut u = 0.0f64;
    for (j, b) in v.beams.iter().enumerate() {
        if b.broken {
            continue;
        }
        let (pa, pb) = (v.nodes[b.a as usize].pos, v.nodes[b.b as usize].pos);
        let len = pa.distance(pb);
        let e = match b.kind {
            ExplicitKind::Plain
            | ExplicitKind::LimitedSpring
            | ExplicitKind::Shock2
            | ExplicitKind::JbeamPlain
            | ExplicitKind::JbeamBounded
            | ExplicitKind::JbeamCoupler
            | ExplicitKind::JbeamAniso => len - b.rest,
            ExplicitKind::Support | ExplicitKind::JbeamSupport => (len - b.rest).min(0.0),
            ExplicitKind::Rope => (len - b.rest).max(0.0),
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
}

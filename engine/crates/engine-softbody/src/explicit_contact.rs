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

use crate::explicit_beams::{contact_force, ContactSurface, ExplicitBeams, STEP};
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

/// The effective mass of a triangle's point at barycentric weights `w` (spec
/// §10.3): the reaction J is shared out as w_k·J, so the point's velocity changes by
/// J·Σ w_k²/m_k, and its mass is 1/Σ w_k²/m_k (terms with w_k = 0 left out). The
/// weighted mean Σ w_k·m_k (RoR's rule) overstates it wherever a light node carries
/// most of the weight beside a heavy one (0.07 kg at 0.77 next to 10 kg at 0.14:
/// 1.5 kg against 0.12 kg), and the light node is then thrown off at hundreds of
/// m/s. `m` and `w` in the order n₁, n₂, n₀ (α, β, γ), summed left to right.
pub fn two_body_triangle_mass(m: [f32; 3], w: [f32; 3]) -> f32 {
    let term = |k: usize| {
        if w[k] > 0.0 {
            (w[k] * w[k]) / m[k]
        } else {
            0.0
        }
    };
    1.0 / ((term(0) + term(1)) + term(2))
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
        for v in &mut self.vehicles {
            v.step();
        }
        self.steps += 1;
        if self.contact && self.vehicles.len() > 1 {
            self.contact_pass();
        }
    }

    /// The bounding boxes (spec §4.1) and the vehicle-contact pass (§3.1 3).
    pub fn contact_pass(&mut self) {
        let nv = self.vehicles.len();
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
            let (lo, hi) = if self.vehicles[a].tri_dead.get(t).copied().unwrap_or(false) {
                // A broken JBeam triangle (jbeam-semantics-spec.md §18): no candidates.
                (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY))
            } else {
                (p0.min(p1).min(p2) - rr, p0.max(p1).max(p2) + rr)
            };
            cands.clear();
            for &b in partners {
                let wb = &self.world[b];
                for &h in &self.vehicles[b].contact_points {
                    let x = wb[h as usize];
                    if x.cmpge(lo).all() && x.cmple(hi).all() {
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
            self.vehicles[a].contact_throttle[t] = [cwait, cidle];
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
        let mut f_start: Vec<Vec<Vec3>> = vec![Vec::new(); self.vehicles.len()];
        if hits.iter().any(|x| two_body(x.b)) {
            split_a = vec![0; self.vehicles[a].nodes.len()];
            for x in hits.iter().filter(|x| two_body(x.b)) {
                for &i in &x.tri {
                    split_a[i as usize] += 1;
                }
                if split_b[x.b].is_empty() {
                    split_b[x.b] = vec![0; self.vehicles[x.b].nodes.len()];
                    f_start[x.b] = self.vehicles[x.b].nodes.iter().map(|n| n.force).collect();
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
                // Each node of the hit is one of its k sub-bodies (mass m/k, force F/k).
                let k = |i: u32| split_a[i as usize] as f32;
                let kh = split_b[b][h as usize] as f32;
                let mh = node.mass / kh;
                let mt = two_body_triangle_mass(
                    [n1.mass / k(i1), n2.mass / k(i2), n0.mass / k(i0)],
                    [alpha, beta, gamma],
                );
                // Only the share m_t/(m_h + m_t) of the contacter's force (before
                // its first hit of the turn, split k ways) is cancelled: the
                // relative normal acceleration's zero.
                let m = (mh * mt) / (mh + mt);
                (m, (f_start[b][h as usize] / kh) * (mt / (mh + mt)))
            } else {
                let mt = n1.mass * alpha + n2.mass * beta + n0.mass * gamma;
                ((node.mass * mt) / (node.mass + mt), node.force)
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
            self.vehicles[b].nodes[h as usize].force += f;
            let va = &mut self.vehicles[a];
            va.nodes[i1 as usize].force -= f * alpha;
            va.nodes[i2 as usize].force -= f * beta;
            va.nodes[i0 as usize].force -= f * gamma;
        }
    }
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

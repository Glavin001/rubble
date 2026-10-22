//! The island's penalty contacts in the decode's bounds (planning's decision on the
//! penetration leak, 2026-10-22):
//!
//! 1. their energy (`k depth^2 / 2` per point, the oracle's own law: `World::island_contacts`)
//!    counts in the island's mechanical energy the event's energy budget sees
//!    (`KE + E_elastic + PE + E_contact`);
//! 2. non-penetration: at the frame's end no contact may be deeper than
//!    `max(depth_start, sqrt(2 E_avail / k))`, the larger of its frame-start depth and the
//!    depth at which its deepest point (stiffness `k`) alone would hold all the energy the
//!    island has for the frame, `E_avail` (its frame-start `KE + E_elastic + E_contact` plus the
//!    event budget's remaining energy). A contact deeper than that is projected out
//!    (`project`).
//!
//! A contact is one pair of bodies of one kind (`ContactProbe::kind`, `a`, `b`), its depth its
//! deepest point's (the points of a box pair are not the same points from one state to the
//! next).
//!
//! Why energy and not kinematics: the depth a contact reaches in one frame of its own
//! approach is not oracle-neutral. The spring-mass law at the frame-start approach speed
//! (`law_depth`) is exceeded by the oracle at 1,240 of 4,367 frame-end contacts of four whole
//! events (chunks pressed by the structure behind them, not by their own approach), and the
//! kinematic `a dt + |g| dt^2 / 2` (`one_frame_depth`) holds on P1's scenes only through its
//! free-fall term: without gravity (the wall-row test scene) pressed fragment pairs exceed
//! it. A pressed contact's energy is a small share of its island's.

use std::collections::{BTreeMap, HashMap};

use stress_ref::math::Vec3;
use stress_ref::world::{ContactProbe, IslandMembers, IslandState, World};

/// Projection passes before the non-penetration bound gives up (the residual is reported).
pub const MAX_PASSES: usize = 32;

/// A contact's identity: kind and the two bodies (the ground's sample point folded away).
pub type ContactKey = (u8, (usize, usize), (usize, usize));

pub fn key_of(p: &ContactProbe) -> ContactKey {
    match p.kind {
        0 => (0, p.a, (0, 0)),
        1 => (1, p.a, (0, 0)),
        _ => (p.kind, p.a, p.b),
    }
}

/// The deepest point of every contact.
pub fn deepest(probes: &[ContactProbe]) -> BTreeMap<ContactKey, ContactProbe> {
    let mut out: BTreeMap<ContactKey, ContactProbe> = BTreeMap::new();
    for p in probes {
        let e = out.entry(key_of(p)).or_insert(*p);
        if p.depth > e.depth {
            *e = *p;
        }
    }
    out
}

/// Diagnostic: the peak compression of a penalty spring `k` closing at `v` (>= 0) on reduced
/// mass `m` pressed by gravity `g_n` (>= 0, along the approach),
/// `d = (m g_n + sqrt((m g_n)^2 + k m v^2)) / k` (not the bound: see the module docs).
pub fn law_depth(v: f64, m: f64, k: f64, g_n: f64) -> f64 {
    let (v, g_n) = (v.max(0.0), g_n.max(0.0));
    let w = m * g_n;
    (w + (w * w + k * m * v * v).sqrt()) / k
}

/// The depth one frame of approach at `approach` (never below 0) and free fall reach:
/// `a dt + |g| dt^2 / 2`.
pub fn one_frame_depth(approach: f64, g: f64, dt: f64) -> f64 {
    approach.max(0.0) * dt + 0.5 * g * dt * dt
}

/// The relative velocity (`a`'s minus the other's) at `p.point` in `w` of the bodies of the
/// contact `p` (the ground at rest).
pub fn rel_velocity(w: &World, p: &ContactProbe) -> Vec3 {
    let imp = |i: usize, x: Vec3| {
        let b = &w.impactors[i];
        b.velocity + b.angular_velocity.cross(x - b.pose.position)
    };
    match p.kind {
        0 => w.solver.point_velocity(p.a.0, p.a.1, p.point),
        1 => imp(p.a.0, p.point),
        2 => w.solver.point_velocity(p.a.0, p.a.1, p.point) - imp(p.b.0, p.point),
        _ => w.solver.point_velocity(p.a.0, p.a.1, p.point) - w.solver.point_velocity(p.b.0, p.b.1, p.point),
    }
}

/// One contact at the frame's end against the non-penetration bound.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Contact {
    pub key: ContactKey,
    pub depth_start: f64,
    pub depth_end: f64,
    /// The frame-start approach speed at the end's deepest point (frame-start velocities).
    pub approach: f64,
    /// `energy_depth(E_avail, k)`.
    pub energy_depth: f64,
    /// `max(depth_start, energy_depth)`.
    pub allowed: f64,
    /// Diagnostic: `one_frame_depth(approach)`.
    pub frame_depth: f64,
    /// The frame-end deepest point's normal (pushing `a`).
    pub normal: Vec3,
    /// Diagnostic: the frame-end approach speed at the same point, the point's stiffness and
    /// reduced mass, and the spring-mass law's depth at the frame-start approach.
    pub approach_end: f64,
    pub k: f64,
    pub m_red: f64,
    pub law_depth: f64,
}

impl Contact {
    pub fn excess(&self) -> f64 {
        self.depth_end - self.allowed
    }
    pub fn over(&self) -> bool {
        self.depth_end > self.allowed * (1.0 + 1e-9) + 1e-12
    }
}

/// The depth at which a point of stiffness `k` holds the energy `e`: `sqrt(2 e / k)`.
pub fn energy_depth(e: f64, k: f64) -> f64 {
    (2.0 * e.max(0.0) / k).sqrt()
}

fn contacts(start: &World, end: &World, d0: &BTreeMap<ContactKey, ContactProbe>, p1: &[ContactProbe], e_avail: f64) -> Vec<Contact> {
    let g = start.solver.config.gravity;
    let dt = start.scene.sim.frame_dt;
    deepest(p1)
        .into_iter()
        .map(|(k, p)| {
            let depth_start = d0.get(&k).map_or(0.0, |q| q.depth);
            let approach = -rel_velocity(start, &p).dot(p.normal);
            let energy_depth = energy_depth(e_avail, p.k);
            Contact {
                key: k,
                depth_start,
                depth_end: p.depth,
                approach,
                energy_depth,
                frame_depth: one_frame_depth(approach, g.norm(), dt),
                allowed: depth_start.max(energy_depth),
                normal: p.normal,
                approach_end: -rel_velocity(end, &p).dot(p.normal),
                k: p.k,
                m_red: p.m_red,
                law_depth: law_depth(approach, p.m_red, p.k, -g.dot(p.normal)),
            }
        })
        .collect()
}

/// The island's frame-start energy available to a contact: `KE + E_elastic + E_contact` at
/// the frame's start plus `remaining` (the event budget's).
pub fn available_energy(start: &World, m: &IslandMembers, remaining: f64) -> f64 {
    let (k, e, _) = start.island_energy(m);
    k + e + start.island_contact_energy(m) + remaining.max(0.0)
}

/// The island's contacts at the frame's end (`end`: the frame-start world with the island's
/// frame-end state merged) against the bound (`e_avail`: `available_energy`), with the
/// frame-start world `start`; and the island's contact energy at the start and at the end.
pub fn check(start: &World, end: &World, m: &IslandMembers, e_avail: f64) -> (Vec<Contact>, f64, f64) {
    let (p0, p1) = (start.island_contacts(m), end.island_contacts(m));
    let (e0, e1) = (p0.iter().map(|p| p.energy).sum(), p1.iter().map(|p| p.energy).sum());
    (contacts(start, end, &deepest(&p0), &p1, e_avail), e0, e1)
}

/// What `project` did.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Projection {
    /// Contacts projected out (each contact once per pass it was over).
    pub clamps: usize,
    pub passes: usize,
    /// Contacts still over after `MAX_PASSES`, and the largest excess left (m).
    pub residual: usize,
    pub residual_excess: f64,
    /// The largest excess projected out (m).
    pub max_excess: f64,
    /// The share of `E_avail` the contacts were held to (1: the bound itself; below 1: the
    /// energy budget tightened it, `decoded_full::island_frame_budget`).
    pub energy_scale: f64,
}

/// A body a contact moves: a cluster (by its index in the world) or an impactor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Body {
    Cluster(usize),
    Impactor(usize),
}

/// The non-penetration bound's corrective action on the island's frame-end state `out`
/// (frame-start world `start`, `e_avail`: `available_energy`): every contact deeper than it may be is projected out along
/// its normal by its excess, split between its two bodies by mass (each moves by the other's
/// share, `m_b / (m_a + m_b)`; the ground and anchored bodies do not move), translating whole
/// bodies (clusters and impactors) and leaving every velocity as it is (momentum kept).
/// Contacts are taken in key order, each against the depth left after the moves before it in
/// the pass (translation only); passes repeat on the recomputed contacts until none is over,
/// at most `MAX_PASSES`. A state none of whose contacts is over is returned as it is.
pub fn project(start: &World, m: &IslandMembers, out: &IslandState, e_avail: f64) -> (IslandState, Projection) {
    let d0 = deepest(&start.island_contacts(m));
    let mut proj = Projection::default();
    let mut x = out.clone();
    for pass in 0..=MAX_PASSES {
        let mut end = World::merge_island(start, &x, m);
        let cs = contacts(start, &end, &d0, &end.island_contacts(m), e_avail);
        let over: Vec<Contact> = cs.into_iter().filter(|c| c.over()).collect();
        if over.is_empty() {
            break;
        }
        if pass == MAX_PASSES {
            proj.residual = over.len();
            proj.residual_excess = over.iter().map(|c| c.excess()).fold(0.0, f64::max);
            break;
        }
        proj.passes = pass + 1;
        let body = |w: &World, kind: u8, side: (usize, usize), first: bool| -> Option<Body> {
            match (kind, first) {
                (0, true) | (2, true) | (3, _) => Some(Body::Cluster(w.solver.chunks[side.0][side.1].cluster)),
                (1, true) | (2, false) => Some(Body::Impactor(side.0)),
                _ => None,
            }
        };
        let mass = |w: &World, b: Option<Body>| -> f64 {
            match b {
                None => f64::INFINITY,
                Some(Body::Cluster(i)) => {
                    let c = &w.solver.clusters[i];
                    if c.anchored { f64::INFINITY } else { c.mass }
                }
                Some(Body::Impactor(i)) => w.impactors[i].mass,
            }
        };
        let mut moved: HashMap<Body, Vec3> = HashMap::new();
        for c in &over {
            let (ba, bb) = (body(&end, c.key.0, c.key.1, true), body(&end, c.key.0, c.key.2, false));
            let get = |b: Option<Body>| b.and_then(|b| moved.get(&b).copied()).unwrap_or(Vec3::ZERO);
            // the depth left: the moves so far along this contact's normal
            let left = c.depth_end - (get(ba) - get(bb)).dot(c.normal);
            let e = left - c.allowed;
            if e <= c.allowed * 1e-9 + 1e-12 {
                continue;
            }
            let (ma, mb) = (mass(&end, ba), mass(&end, bb));
            if ma.is_infinite() && mb.is_infinite() {
                continue;
            }
            let wa = if ma.is_infinite() { 0.0 } else if mb.is_infinite() { 1.0 } else { mb / (ma + mb) };
            if let Some(b) = ba {
                *moved.entry(b).or_insert(Vec3::ZERO) += c.normal * (e * wa);
            }
            if let Some(b) = bb {
                *moved.entry(b).or_insert(Vec3::ZERO) -= c.normal * (e * (1.0 - wa));
            }
            proj.clamps += 1;
            proj.max_excess = proj.max_excess.max(e);
        }
        let mut bodies: Vec<(Body, Vec3)> = moved.into_iter().collect();
        bodies.sort_by_key(|(b, _)| *b);
        for (b, d) in bodies {
            match b {
                Body::Cluster(i) => end.solver.clusters[i].pose.position += d,
                Body::Impactor(i) => end.impactors[i].pose.position += d,
            }
        }
        x = end.extract_island(m);
    }
    (x, proj)
}

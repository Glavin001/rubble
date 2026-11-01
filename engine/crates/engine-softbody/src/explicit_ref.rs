//! Explicit reference oracle: the node–beam model integrated the way Rigs of Rods
//! and BeamNG do it, for calibration and tests only (speed does not matter).
//!
//! * f64 throughout, symplectic (semi-implicit) Euler at a fixed rate, 2,000 Hz by
//!   default, raised automatically when the stiffest node needs it (see
//!   [`ExplicitSim::stable_hz`]).
//! * The same beams as the GPU ([`crate::pack::pack_soft`]) and the same rules
//!   ([`crate::model`]'s generic elastic force, damping, plasticity and breaking),
//!   with yield and break checked every substep, as Rigs of Rods does.
//! * Contacts as BeamNG does them: only nodes collide. Node spheres against static
//!   obstacles (planes, oriented boxes, capsules), and nodes against collision
//!   triangles of other vehicles and, for self-colliding nodes, of their own,
//!   as penalties that start `thickness` in front of the face. A pair cache keeps
//!   the side a node came from (anti-clip), so a node pushed deep is pushed back
//!   to its own side rather than through.
//!
//! This replaces "Rubble at 960 Hz × 30" as the reference: that shares the
//! implicit solver's biases (backward Euler's damping, unconverged sweeps).

use crate::desc::SoftVehicleDesc;
use crate::layout::{BEAM_BROKEN, BEAM_YIELDED};
use crate::model::{plastic_update, readout_force_rate, BeamT};
use crate::pack::{pack_soft, Placement, SpawnOptions};
use glam::{DQuat, DVec3, Quat, Vec3};
use std::collections::HashMap;

/// A static obstacle the node spheres collide with.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// Half-space below `normal · x = offset`.
    Plane { normal: DVec3, offset: f64 },
    /// Oriented box.
    Box {
        center: DVec3,
        half: DVec3,
        rotation: DQuat,
    },
    /// Capsule around segment a–b.
    Capsule { a: DVec3, b: DVec3, radius: f64 },
}

#[derive(Clone, Copy, Debug)]
pub struct Obstacle {
    pub shape: Shape,
    pub friction: f64,
}

impl Shape {
    /// Signed distance from `p` to the surface and the outward normal there.
    pub fn distance(&self, p: DVec3) -> (f64, DVec3) {
        match *self {
            Shape::Plane { normal, offset } => (normal.dot(p) - offset, normal),
            Shape::Box {
                center,
                half,
                rotation,
            } => {
                let q = rotation.inverse() * (p - center);
                let d = q.abs() - half;
                if d.max_element() <= 0.0 {
                    // Inside: the nearest face.
                    let axis = if d.x >= d.y && d.x >= d.z {
                        0
                    } else if d.y >= d.z {
                        1
                    } else {
                        2
                    };
                    let mut n = DVec3::ZERO;
                    n[axis] = q[axis].signum();
                    (d[axis], rotation * n)
                } else {
                    let outside = d.max(DVec3::ZERO);
                    let len = outside.length();
                    let n = DVec3::new(
                        outside.x * q.x.signum(),
                        outside.y * q.y.signum(),
                        outside.z * q.z.signum(),
                    ) / len;
                    (len, rotation * n)
                }
            }
            Shape::Capsule { a, b, radius } => {
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-18)).clamp(0.0, 1.0);
                let d = p - (a + ab * t);
                let len = d.length();
                let n = if len > 1e-12 { d / len } else { DVec3::Y };
                (len - radius, n)
            }
        }
    }
}

/// A collision triangle (counter-clockwise front face) of vehicle `body`.
#[derive(Clone, Copy, Debug)]
pub struct OracleTri {
    pub nodes: [u32; 3],
    pub body: u32,
    pub collide: bool,
    /// Beams along its edges (ab, bc, ca), when any.
    pub edge_beams: [Option<u32>; 3],
}

/// One vehicle's ranges in the oracle's arrays.
#[derive(Clone, Copy, Debug, Default)]
pub struct OracleBody {
    pub first_node: u32,
    pub nodes: u32,
    pub first_beam: u32,
    pub beams: u32,
    pub first_tri: u32,
    pub tris: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ExplicitSettings {
    /// Requested step rate (Hz); raised to [`ExplicitSim::stable_hz`] when lower.
    pub hz: f64,
    pub gravity: DVec3,
    pub damping_scale: f64,
    /// Shortest rest length plasticity may produce (RoR: 0.1 m).
    pub min_length: f64,
    pub plasticity: bool,
    pub breaking: bool,
    /// Node sphere radius against obstacles (the GPU's node collider).
    pub node_radius: f64,
    /// Node–obstacle and node–triangle penalty (N/m) and damping ratio.
    pub contact_stiffness: f64,
    pub contact_damping_ratio: f64,
    /// Node–triangle friction coefficient.
    pub tri_friction: f64,
    /// Node–triangle contact thickness: forces start this far in front of the face
    /// (BeamNG: about 2.5 cm).
    pub tri_thickness: f64,
    /// Nodes collide with their own vehicle's triangles (BeamNG `selfCollision`).
    pub self_collision: bool,
    /// Pair cache range: a (node, triangle) pair is tracked once the node comes
    /// within this distance of the face (BeamNG activates triangles at ~10 cm).
    pub pair_range: f64,
}

impl Default for ExplicitSettings {
    fn default() -> Self {
        Self {
            hz: 2000.0,
            gravity: DVec3::new(0.0, -9.81, 0.0),
            damping_scale: 1.0,
            min_length: 0.1,
            plasticity: true,
            breaking: true,
            node_radius: 0.06,
            contact_stiffness: 1.0e7,
            // Critically damped: inelastic, like the GPU's (restitution 0).
            contact_damping_ratio: 1.0,
            tri_friction: 0.5,
            tri_thickness: 0.025,
            self_collision: true,
            pair_range: 0.1,
        }
    }
}

/// Self-collision exclusion (decision s18, shared with the GPU kernel): a node
/// skips its own body's triangles it is a corner of, and any whose face it sat
/// over within `SELF_REST_GAP` in the undeformed body (its own panel, or the
/// shell under a bolt-on part).
pub const SELF_REST_GAP: f64 = 0.12;

/// Whether rest-pose node `p` is excluded from rest-pose triangle `a b c`.
pub fn self_excluded(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> bool {
    let n = (b - a).cross(c - a);
    let l = n.length();
    let d_plane = if l > 1e-12 {
        (p - a).dot(n / l).abs()
    } else {
        0.0
    };
    let d_rest = (closest_on_triangle(p, a, b, c).0 - p).length();
    d_rest < SELF_REST_GAP && d_rest - d_plane < 0.02
}

/// Closest point on triangle abc to p (Ericson 5.1.5) and its barycentrics.
pub fn closest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> (DVec3, DVec3) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, DVec3::new(1.0, 0.0, 0.0));
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, DVec3::new(0.0, 1.0, 0.0));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, DVec3::new(1.0 - v, v, 0.0));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, DVec3::new(0.0, 0.0, 1.0));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, DVec3::new(1.0 - w, 0.0, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, DVec3::new(0.0, 1.0 - w, w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, DVec3::new(1.0 - v - w, v, w))
}

/// Uniform hash grid of triangle bounds, rebuilt every substep.
#[derive(Default)]
struct TriGrid {
    cell: f64,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl TriGrid {
    fn key(&self, p: DVec3) -> (i32, i32, i32) {
        let q = (p / self.cell).floor();
        (q.x as i32, q.y as i32, q.z as i32)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SubstepReport {
    pub yielded: u32,
    pub broke: u32,
}

pub struct ExplicitSim {
    pub x: Vec<DVec3>,
    pub v: Vec<DVec3>,
    pub mass: Vec<f64>,
    pub fixed: Vec<bool>,
    pub collide: Vec<bool>,
    /// Collides with other vehicles' (and, with self-collision, its own) triangles.
    pub contacter: Vec<bool>,
    /// Vehicle of each node.
    pub node_body: Vec<u32>,
    /// Rest position (vehicle frame at spawn) for the self-collision exclusion.
    pub rest: Vec<DVec3>,
    pub beams: Vec<BeamT<f64>>,
    pub tris: Vec<OracleTri>,
    pub bodies: Vec<OracleBody>,
    pub obstacles: Vec<Obstacle>,
    pub settings: ExplicitSettings,
    /// (node, triangle) → the side (±1) the node came from.
    pub pairs: HashMap<(u32, u32), f64>,
    /// Self-collision exclusions (node, triangle), from the rest pose.
    excluded: std::collections::HashSet<(u32, u32)>,
    pub time: f64,
    pub substeps: u64,
    /// Totals over the run.
    pub yielded_events: u64,
    pub broken_events: u64,
    grid: TriGrid,
    hz: f64,
}

impl Default for ExplicitSim {
    fn default() -> Self {
        Self::new(ExplicitSettings::default())
    }
}

impl ExplicitSim {
    pub fn new(settings: ExplicitSettings) -> Self {
        Self {
            x: Vec::new(),
            v: Vec::new(),
            mass: Vec::new(),
            fixed: Vec::new(),
            collide: Vec::new(),
            contacter: Vec::new(),
            node_body: Vec::new(),
            rest: Vec::new(),
            beams: Vec::new(),
            tris: Vec::new(),
            bodies: Vec::new(),
            obstacles: Vec::new(),
            settings,
            pairs: HashMap::new(),
            excluded: Default::default(),
            time: 0.0,
            substeps: 0,
            yielded_events: 0,
            broken_events: 0,
            grid: TriGrid {
                cell: 0.3,
                cells: HashMap::new(),
            },
            hz: settings.hz,
        }
    }

    /// Adds a vehicle without a rigid core (every node free, its own mass), as
    /// the GPU spawns it with `SpawnOptions { rigid_core: false, .. }`.
    pub fn add_vehicle(
        &mut self,
        desc: &SoftVehicleDesc,
        position: Vec3,
        rotation: Quat,
        velocity: Vec3,
    ) -> u32 {
        let opts = SpawnOptions {
            rigid_core: false,
            velocity: velocity.to_array(),
            ..Default::default()
        };
        let p = pack_soft(desc, Placement { position, rotation }, &opts, 0, -1);
        let body = self.bodies.len() as u32;
        let first = self.x.len() as u32;
        let rec = OracleBody {
            first_node: first,
            nodes: desc.nodes.len() as u32,
            first_beam: self.beams.len() as u32,
            beams: p.beams.len() as u32,
            first_tri: self.tris.len() as u32,
            tris: desc.triangles.len() as u32,
        };
        for (i, nd) in desc.nodes.iter().enumerate() {
            self.x.push(p.bodies[i].position.map(f64::from).into());
            self.v.push(if nd.fixed {
                DVec3::ZERO
            } else {
                velocity.as_dvec3()
            });
            self.mass.push(f64::from(p.masses[i]));
            self.fixed.push(nd.fixed);
            self.collide.push(nd.collision);
            self.contacter.push(nd.contacter);
            self.node_body.push(body);
            self.rest.push(Vec3::from_array(nd.position).as_dvec3());
        }
        for b in &p.beams {
            let mut e = b.cast::<f64>();
            e.a += first;
            e.b += first;
            // The oracle runs the material stiffness: no AVBD penalty ramp.
            e.penalty = e.spring;
            self.beams.push(e);
        }
        let mut by_edge: HashMap<(u32, u32), u32> = HashMap::new();
        for (k, b) in self.beams[rec.first_beam as usize..].iter().enumerate() {
            by_edge
                .entry((b.a.min(b.b), b.a.max(b.b)))
                .or_insert(rec.first_beam + k as u32);
        }
        for t in &desc.triangles {
            let nodes = t.nodes.map(|k| k + first);
            let edge = |p: u32, q: u32| by_edge.get(&(p.min(q), p.max(q))).copied();
            self.tris.push(OracleTri {
                nodes,
                body,
                collide: t.collision,
                edge_beams: [
                    edge(nodes[0], nodes[1]),
                    edge(nodes[1], nodes[2]),
                    edge(nodes[2], nodes[0]),
                ],
            });
        }
        // Self-collision exclusions from the rest pose.
        for ti in rec.first_tri..rec.first_tri + rec.tris {
            let t = self.tris[ti as usize];
            let [a, b, c] = t.nodes.map(|k| self.rest[k as usize]);
            for n in first..first + rec.nodes {
                if t.nodes.contains(&n) || self_excluded(self.rest[n as usize], a, b, c) {
                    self.excluded.insert((n, ti));
                }
            }
        }
        self.bodies.push(rec);
        self.hz = self.settings.hz.max(self.stable_hz());
        body
    }

    /// The step rate in use (Hz).
    pub fn hz(&self) -> f64 {
        self.hz
    }

    /// Smallest rate at which symplectic Euler stays stable on the stiffest node:
    /// h ≤ 1.6/ω with ω² = (Σ k + contact k)/m (2/ω is the bound; 20% margin for
    /// the coupling between neighbours and the dashpots).
    pub fn stable_hz(&self) -> f64 {
        let mut ksum = vec![self.settings.contact_stiffness; self.x.len()];
        let mut csum: Vec<f64> = self.mass.iter().map(|&m| self.contact_damping(m)).collect();
        for b in &self.beams {
            let k = b.spring.max(if b.kind == crate::model::KIND_BOUNDED {
                b.deform_c
            } else {
                0.0
            });
            for n in [b.a, b.b] {
                ksum[n as usize] += k;
                csum[n as usize] += b.damp * self.settings.damping_scale;
            }
        }
        let mut hz: f64 = 0.0;
        for i in 0..self.x.len() {
            if self.fixed[i] {
                continue;
            }
            let w = (ksum[i] / self.mass[i]).sqrt();
            // Damped bound: h·(ζ + √(ζ² + 1))·ω ≤ 2, with c/m = 2ζω.
            let zeta = csum[i] / self.mass[i] / (2.0 * w);
            hz = hz.max(w * (zeta + (zeta * zeta + 1.0).sqrt()) / 1.6);
        }
        hz
    }

    pub fn add_obstacle(&mut self, shape: Shape, friction: f64) {
        self.obstacles.push(Obstacle { shape, friction });
    }

    /// Advances `seconds` (whole substeps).
    pub fn run(&mut self, seconds: f64) -> SubstepReport {
        let n = (seconds * self.hz).round() as u64;
        let mut rep = SubstepReport::default();
        for _ in 0..n {
            let r = self.substep();
            rep.yielded += r.yielded;
            rep.broke += r.broke;
        }
        rep
    }

    pub fn substep(&mut self) -> SubstepReport {
        let h = 1.0 / self.hz;
        let n = self.x.len();
        let mut f: Vec<DVec3> = (0..n)
            .map(|i| self.settings.gravity * self.mass[i])
            .collect();
        let mut rep = SubstepReport::default();
        let mut groups: Vec<u32> = Vec::new();
        let st = self.settings;
        for b in &mut self.beams {
            if b.broken() {
                continue;
            }
            let (ia, ib) = (b.a as usize, b.b as usize);
            let d = self.x[ia] - self.x[ib];
            let len = d.length();
            if len < 1e-9 {
                continue;
            }
            let dir = d / len;
            let rate = (self.v[ia] - self.v[ib]).dot(dir);
            let force = readout_force_rate(b, len, rate, st.damping_scale);
            let o = plastic_update(b, len, force, st.min_length, st.plasticity, st.breaking);
            rep.yielded += u32::from(o.yielded);
            if o.broke {
                rep.broke += 1;
                if b.group != 0 {
                    groups.push(b.group);
                }
                continue;
            }
            // + = compression: pushes a away from b. A yielded beam carries its
            // threshold (plastic_update leaves the capped force in `force`).
            let fa = dir * b.force;
            f[ia] += fa;
            f[ib] -= fa;
        }
        if !groups.is_empty() {
            for b in &mut self.beams {
                if b.group != 0 && !b.broken() && groups.contains(&b.group) {
                    b.flags |= BEAM_BROKEN;
                    rep.broke += 1;
                }
            }
        }
        self.obstacle_forces(&mut f);
        self.triangle_forces(&mut f);
        for i in 0..n {
            if self.fixed[i] {
                self.v[i] = DVec3::ZERO;
                continue;
            }
            self.v[i] += f[i] * (h / self.mass[i]);
            self.x[i] += self.v[i] * h;
        }
        self.time += h;
        self.substeps += 1;
        self.yielded_events += u64::from(rep.yielded);
        self.broken_events += u64::from(rep.broke);
        rep
    }

    fn contact_damping(&self, m: f64) -> f64 {
        2.0 * self.settings.contact_damping_ratio * (self.settings.contact_stiffness * m).sqrt()
    }

    /// Node spheres against static obstacles: penalty with damping, and Coulomb
    /// friction regularised so it stops sliding within a substep without
    /// reversing it.
    fn obstacle_forces(&self, f: &mut [DVec3]) {
        let h = 1.0 / self.hz;
        let k = self.settings.contact_stiffness;
        for i in 0..self.x.len() {
            if !self.collide[i] || self.fixed[i] {
                continue;
            }
            for o in &self.obstacles {
                let (d, nrm) = o.shape.distance(self.x[i]);
                let pen = self.settings.node_radius - d;
                if pen <= 0.0 {
                    continue;
                }
                let m = self.mass[i];
                let vn = self.v[i].dot(nrm);
                let fn_ = (k * pen - self.contact_damping(m) * vn).max(0.0);
                let vt = self.v[i] - nrm * vn;
                let ft = friction(vt, o.friction * fn_, m / h);
                f[i] += nrm * fn_ + ft;
            }
        }
    }

    fn rebuild_grid(&mut self) {
        let pad = self.settings.pair_range + self.settings.tri_thickness;
        self.grid.cells.clear();
        for (ti, t) in self.tris.iter().enumerate() {
            if !t.collide || self.tri_broken(ti) {
                continue;
            }
            let [a, b, c] = t.nodes.map(|k| self.x[k as usize]);
            let lo = a.min(b).min(c) - DVec3::splat(pad);
            let hi = a.max(b).max(c) + DVec3::splat(pad);
            let (k0, k1) = (self.grid.key(lo), self.grid.key(hi));
            for gx in k0.0..=k1.0 {
                for gy in k0.1..=k1.1 {
                    for gz in k0.2..=k1.2 {
                        self.grid
                            .cells
                            .entry((gx, gy, gz))
                            .or_default()
                            .push(ti as u32);
                    }
                }
            }
        }
    }

    /// A triangle loses collision once a beam along one of its edges has broken
    /// (BeamNG's rule), or stretched past 2.5× (the GPU's tear test) when no beam
    /// runs along an edge.
    pub fn tri_broken(&self, ti: usize) -> bool {
        let t = self.tris[ti];
        if t.edge_beams
            .iter()
            .flatten()
            .any(|&e| self.beams[e as usize].broken())
        {
            return true;
        }
        let rest = t.nodes.map(|k| self.rest[k as usize]);
        let now = t.nodes.map(|k| self.x[k as usize]);
        (0..3).any(|k| {
            let r = (rest[k] - rest[(k + 1) % 3]).length();
            (now[k] - now[(k + 1) % 3]).length() > 2.5 * r
        })
    }

    /// Nodes against collision triangles: other vehicles', and (self-collision)
    /// their own outside the rest-pose exclusion. A pair enters the cache with the
    /// side the node is on when it first comes within `pair_range` of the face
    /// inside the triangle's prism; from then on the penalty pushes it toward that
    /// side, however deep it is pressed (anti-clip).
    fn triangle_forces(&mut self, f: &mut [DVec3]) {
        if self.tris.is_empty() {
            return;
        }
        self.rebuild_grid();
        let h = 1.0 / self.hz;
        let st = self.settings;
        let mut live: HashMap<(u32, u32), f64> = HashMap::with_capacity(self.pairs.len());
        for i in 0..self.x.len() {
            if !self.contacter[i] || self.fixed[i] {
                continue;
            }
            let p = self.x[i];
            let Some(cands) = self.grid.cells.get(&self.grid.key(p)) else {
                continue;
            };
            let me = self.node_body[i];
            for &ti in cands {
                let t = self.tris[ti as usize];
                if t.body == me && (!st.self_collision || self.excluded.contains(&(i as u32, ti))) {
                    continue;
                }
                let [a, b, c] = t.nodes.map(|k| self.x[k as usize]);
                let nr = (b - a).cross(c - a);
                let area2 = nr.length();
                if area2 < 1e-12 {
                    continue;
                }
                let nrm = nr / area2;
                let d = (p - a).dot(nrm);
                let key = (i as u32, ti);
                let (q, w) = closest_on_triangle(p - nrm * d, a, b, c);
                let inside = (q - (p - nrm * d)).length_squared() < 1e-10;
                let side = match self.pairs.get(&key) {
                    Some(&s) => s,
                    None if inside && d.abs() < st.pair_range + st.tri_thickness => {
                        if d >= 0.0 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    None => continue,
                };
                // Keep the pair while the node is near the face's prism.
                if d.abs() > st.pair_range + st.tri_thickness + 0.05
                    || (q - (p - nrm * d)).length() > 0.05
                {
                    continue;
                }
                live.insert(key, side);
                if !inside {
                    continue;
                }
                let pen = st.tri_thickness - side * d;
                if pen <= 0.0 {
                    continue;
                }
                let n = nrm * side;
                let tn = t.nodes.map(|k| k as usize);
                let vt_tri = self.v[tn[0]] * w.x + self.v[tn[1]] * w.y + self.v[tn[2]] * w.z;
                let m_tri =
                    self.mass[tn[0]] * w.x + self.mass[tn[1]] * w.y + self.mass[tn[2]] * w.z;
                let m = self.mass[i] * m_tri / (self.mass[i] + m_tri).max(1e-9);
                let rel = self.v[i] - vt_tri;
                let vn = rel.dot(n);
                let fn_ = (st.contact_stiffness * pen - self.contact_damping(m) * vn).max(0.0);
                let ft = friction(rel - n * vn, st.tri_friction * fn_, m / h);
                let fi = n * fn_ + ft;
                f[i] += fi;
                for k in 0..3 {
                    f[tn[k]] -= fi * w[k];
                }
            }
        }
        self.pairs = live;
    }

    /// Positions as f32 (for the shared metrics).
    pub fn positions(&self) -> Vec<Vec3> {
        self.x.iter().map(|p| p.as_vec3()).collect()
    }

    pub fn body_positions(&self, body: u32) -> Vec<Vec3> {
        let r = self.bodies[body as usize];
        self.x[r.first_node as usize..(r.first_node + r.nodes) as usize]
            .iter()
            .map(|p| p.as_vec3())
            .collect()
    }

    pub fn body_velocities(&self, body: u32) -> Vec<Vec3> {
        let r = self.bodies[body as usize];
        self.v[r.first_node as usize..(r.first_node + r.nodes) as usize]
            .iter()
            .map(|p| p.as_vec3())
            .collect()
    }

    /// Broken beam indices of `body` (relative to its first beam).
    pub fn broken(&self, body: u32) -> Vec<u32> {
        let r = self.bodies[body as usize];
        (0..r.beams)
            .filter(|&k| self.beams[(r.first_beam + k) as usize].broken())
            .collect()
    }

    pub fn yielded_beams(&self, body: u32) -> usize {
        let r = self.bodies[body as usize];
        self.beams[r.first_beam as usize..(r.first_beam + r.beams) as usize]
            .iter()
            .filter(|b| b.flags & BEAM_YIELDED != 0)
            .count()
    }

    pub fn kinetic_energy(&self, body: u32) -> f64 {
        let r = self.bodies[body as usize];
        (r.first_node..r.first_node + r.nodes)
            .map(|i| 0.5 * self.mass[i as usize] * self.v[i as usize].length_squared())
            .sum()
    }

    pub fn total_mass(&self, body: u32) -> f64 {
        let r = self.bodies[body as usize];
        self.mass[r.first_node as usize..(r.first_node + r.nodes) as usize]
            .iter()
            .sum()
    }

    /// Plastic rest lengths (for the oracle-vs-GPU per-beam checks).
    pub fn rest_lengths(&self, body: u32) -> Vec<f64> {
        let r = self.bodies[body as usize];
        self.beams[r.first_beam as usize..(r.first_beam + r.beams) as usize]
            .iter()
            .map(|b| b.rest)
            .collect()
    }
}

/// Coulomb friction against relative tangential velocity `vt`, capped at `cap`
/// and at the force that stops `m_eff` within the substep (`m_over_h` = m/h).
fn friction(vt: DVec3, cap: f64, m_over_h: f64) -> DVec3 {
    let s = vt.length();
    if s < 1e-12 || cap <= 0.0 {
        return DVec3::ZERO;
    }
    -vt / s * cap.min(m_over_h * s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::KIND_NORMAL;
    use crate::procedural::box_truck;
    use crate::truck::load_truck;

    fn ground() -> Shape {
        Shape::Plane {
            normal: DVec3::Y,
            offset: 0.0,
        }
    }

    #[test]
    fn undamped_oscillator_conserves_energy() {
        // Two nodes, one fixed, an undamped elastic beam: symplectic Euler keeps
        // the energy within a bounded band (no drift over many periods).
        let mut sim = ExplicitSim::new(ExplicitSettings {
            gravity: DVec3::ZERO,
            ..Default::default()
        });
        sim.x = vec![DVec3::ZERO, DVec3::new(1.1, 0.0, 0.0)];
        sim.v = vec![DVec3::ZERO; 2];
        sim.mass = vec![1.0, 2.0];
        sim.fixed = vec![true, false];
        sim.collide = vec![false; 2];
        sim.contacter = vec![false; 2];
        sim.node_body = vec![0; 2];
        sim.rest = sim.x.clone();
        sim.beams.push(BeamT {
            a: 1,
            b: 0,
            rest: 1.0,
            rest0: 1.0,
            spring: 1.0e4,
            damp: 0.0,
            deform_c: f64::INFINITY,
            strength: f64::INFINITY,
            coef: 0.0,
            short_bound: 0.0,
            long_bound: 0.0,
            kind: KIND_NORMAL,
            flags: 0,
            body: 0,
            group: 0,
            penalty: 1.0e4,
            lambda: 0.0,
            deform_t: f64::INFINITY,
            force: 0.0,
        });
        sim.bodies.push(OracleBody {
            nodes: 2,
            beams: 1,
            ..Default::default()
        });
        let energy = |s: &ExplicitSim| {
            let l = (s.x[1] - s.x[0]).length();
            0.5 * 2.0 * s.v[1].length_squared() + 0.5 * 1.0e4 * (l - 1.0).powi(2)
        };
        let e0 = energy(&sim);
        let mut lo = e0;
        let mut hi = e0;
        let mut crossings = 0;
        let mut last = sim.x[1].x - 1.0;
        let (mut early, mut late) = (0.0, 0.0);
        for k in 0..20_000 {
            sim.substep();
            let e = energy(&sim);
            lo = lo.min(e);
            hi = hi.max(e);
            if k < 2000 {
                early += e / 2000.0;
            } else if k >= 18_000 {
                late += e / 2000.0;
            }
            let now = sim.x[1].x - 1.0;
            if now.signum() != last.signum() {
                crossings += 1;
            }
            last = now;
        }
        // The band is about ωh (3.5% here); what matters is no drift.
        assert!((hi - lo) / e0 < 0.05, "energy band {lo}..{hi} of {e0}");
        assert!((late - early).abs() / e0 < 1e-3, "drift {early} → {late}");
        // ω = √(k/m) = 70.7 rad/s → 11.25 Hz → 225 crossings in 10 s.
        assert!((crossings as i32 - 225).abs() <= 2, "crossings {crossings}");
    }

    #[test]
    fn rate_rises_to_keep_stiff_content_stable() {
        let d = load_truck(&box_truck(
            4,
            2,
            2,
            0.4,
            10.0,
            3.0e7,
            f32::INFINITY,
            f32::INFINITY,
        ));
        let mut sim = ExplicitSim::default();
        sim.add_vehicle(&d, Vec3::new(0.0, 0.1, 0.0), Quat::IDENTITY, Vec3::ZERO);
        sim.add_obstacle(ground(), 0.8);
        assert!(
            sim.hz() > 2000.0,
            "stiff light lattice needs > 2 kHz ({})",
            sim.hz()
        );
        sim.run(0.3);
        assert!(sim.x.iter().all(|p| p.is_finite()));
        assert!(sim.kinetic_energy(0) < 5.0, "{}", sim.kinetic_energy(0));
    }

    #[test]
    fn closest_point_regions() {
        let (a, b, c) = (DVec3::ZERO, DVec3::X, DVec3::Y);
        let (q, w) = closest_on_triangle(DVec3::new(0.25, 0.25, 1.0), a, b, c);
        assert!((q - DVec3::new(0.25, 0.25, 0.0)).length() < 1e-12);
        assert!((w - DVec3::new(0.5, 0.25, 0.25)).length() < 1e-12);
        let (q, _) = closest_on_triangle(DVec3::new(2.0, -1.0, 0.0), a, b, c);
        assert!((q - b).length() < 1e-12);
    }

    #[test]
    fn box_distance_inside_and_out() {
        let s = Shape::Box {
            center: DVec3::ZERO,
            half: DVec3::new(1.0, 2.0, 3.0),
            rotation: DQuat::IDENTITY,
        };
        let (d, n) = s.distance(DVec3::new(0.9, 0.0, 0.0));
        assert!((d + 0.1).abs() < 1e-12 && n == DVec3::X);
        let (d, n) = s.distance(DVec3::new(0.0, 0.0, -3.5));
        assert!((d - 0.5).abs() < 1e-12 && n == DVec3::NEG_Z);
    }
}

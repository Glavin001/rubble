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

use crate::desc::{ElementKind, SoftVehicleDesc};
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
    /// Collision force and anti-clip; false: anti-clip only (BeamNG `NONCOLLIDABLE`).
    pub collide: bool,
    /// Extra thickness (BeamNG `externalCollisionBias`).
    pub bias: f64,
    /// Beams along its edges (ab, bc, ca), when any.
    pub edge_beams: [Option<u32>; 3],
    /// A beam of the triangle's break group: once it breaks, so does the triangle.
    pub group_beam: Option<u32>,
    /// Its nodes' shared self-collision group (0 = none or mixed).
    pub group: u32,
}

/// An LBEAM or torsion bar in the oracle (global node indices).
#[derive(Clone, Copy, Debug)]
pub struct OracleElement {
    pub kind: ElementKind,
    pub nodes: [u32; 4],
    pub spring: f64,
    pub damp: f64,
    pub deform: f64,
    pub strength: f64,
    /// Rest angle (moves with plasticity).
    pub rest: f64,
    pub broken: bool,
    pub yielded: bool,
}

impl OracleElement {
    pub fn count(&self) -> usize {
        match self.kind {
            ElementKind::Lbeam => 3,
            ElementKind::Torsion => 4,
        }
    }

    pub fn angle(&self, x: &[DVec3]) -> f64 {
        let p = |i: usize| x[self.nodes[i] as usize];
        match self.kind {
            ElementKind::Lbeam => crate::elements::angle(p(0), p(1), p(2)),
            ElementKind::Torsion => crate::elements::dihedral([p(0), p(1), p(2), p(3)]),
        }
    }

    /// ∂angle/∂x per node.
    pub fn jacobian(&self, x: &[DVec3]) -> [DVec3; 4] {
        let p = |i: usize| x[self.nodes[i] as usize];
        match self.kind {
            ElementKind::Lbeam => {
                // k = 1 and a rest one radian off gives ∂θ/∂x itself.
                let th = crate::elements::angle(p(0), p(1), p(2));
                let g = crate::elements::lbeam_gradient(p(0), p(1), p(2), th - 1.0, 1.0);
                [g[0], g[1], g[2], DVec3::ZERO]
            }
            ElementKind::Torsion => crate::elements::dihedral_jacobian([p(0), p(1), p(2), p(3)]),
        }
    }
}

/// A slidenode on its rail in the oracle (global node indices).
#[derive(Clone, Debug)]
pub struct OracleSlide {
    pub node: u32,
    pub rail: Vec<u32>,
    pub looped: bool,
    pub capped: bool,
    pub attached: bool,
    pub tolerance: f64,
    pub spring: f64,
    pub damp: f64,
    pub strength: f64,
    pub cap_strength: f64,
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

/// Whether `node` of `desc` skips its own vehicle's triangle `tri` (shared by the
/// oracle, the metrics and, mirrored, `soft_collide_nodes`):
/// * the node does not self-collide (BeamNG `selfCollision`), or is a corner;
/// * with node groups (BeamNG): the triangle's three nodes share its group;
/// * without (Rigs of Rods, procedural): the rest-distance rule [`self_excluded`].
pub fn self_skip(desc: &SoftVehicleDesc, node: u32, tri: [u32; 3]) -> bool {
    let n = &desc.nodes[node as usize];
    if !n.self_collision || tri.contains(&node) {
        return true;
    }
    if !desc.node_groups.is_empty() {
        let g = n.self_group;
        return g != 0 && tri.iter().all(|&k| desc.nodes[k as usize].self_group == g);
    }
    let p = |k: u32| Vec3::from_array(desc.nodes[k as usize].position).as_dvec3();
    self_excluded(p(node), p(tri[0]), p(tri[1]), p(tri[2]))
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
    /// Substeps left before the next rebuild (the padding covers the travel).
    fresh: u32,
}

/// Substeps between grid rebuilds; the padding grows by the travel over them.
const GRID_REUSE: u32 = 8;

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
    /// LBEAMs and torsion bars (global node indices) with their plastic state.
    pub elements: Vec<OracleElement>,
    /// Slidenodes on rails (global node indices).
    pub slides: Vec<OracleSlide>,
    pub settings: ExplicitSettings,
    /// (node, triangle) → the side (±1) the node came from.
    pub pairs: HashMap<(u32, u32), f64>,
    /// Self-collision exclusions (node, triangle) by the rest-distance rule, for
    /// vehicles without node groups.
    excluded: std::collections::HashSet<(u32, u32)>,
    /// Per node: self-collides, and its group when its vehicle has groups
    /// (`Some(0)` = grouped vehicle, no group; `None` = rest-distance rule).
    pub self_collision: Vec<bool>,
    pub node_group: Vec<Option<u32>>,
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
            elements: Vec::new(),
            slides: Vec::new(),
            settings,
            pairs: HashMap::new(),
            excluded: Default::default(),
            self_collision: Vec::new(),
            node_group: Vec::new(),
            time: 0.0,
            substeps: 0,
            yielded_events: 0,
            broken_events: 0,
            grid: TriGrid {
                cell: 0.3,
                cells: HashMap::new(),
                fresh: 0,
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
            self.self_collision.push(nd.self_collision);
            self.node_group
                .push((!desc.node_groups.is_empty()).then_some(nd.self_group));
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
            let group_beam = (t.break_group != 0)
                .then(|| {
                    self.beams[rec.first_beam as usize..]
                        .iter()
                        .position(|b| b.group == t.break_group)
                        .map(|k| rec.first_beam + k as u32)
                })
                .flatten();
            let g = desc.nodes[t.nodes[0] as usize].self_group;
            let shared = t
                .nodes
                .iter()
                .all(|&k| desc.nodes[k as usize].self_group == g);
            self.tris.push(OracleTri {
                nodes,
                body,
                group: if shared { g } else { 0 },
                collide: t.collision,
                bias: f64::from(t.bias),
                group_beam,
                edge_beams: [
                    edge(nodes[0], nodes[1]),
                    edge(nodes[1], nodes[2]),
                    edge(nodes[2], nodes[0]),
                ],
            });
        }
        // Rest-distance exclusions (vehicles without node groups; grouped ones
        // apply the group rule as they go, the same as `self_skip`).
        if desc.node_groups.is_empty() {
            for (k, t) in desc.triangles.iter().enumerate() {
                for n in 0..rec.nodes {
                    if !t.nodes.contains(&n) && self_skip(desc, n, t.nodes) {
                        self.excluded.insert((first + n, rec.first_tri + k as u32));
                    }
                }
            }
        }
        for e in &desc.elements {
            let nodes = e.nodes.map(|k| if k == u32::MAX { k } else { k + first });
            let mut el = OracleElement {
                kind: e.kind,
                nodes,
                spring: f64::from(e.spring),
                damp: f64::from(e.damp),
                deform: f64::from(e.deform),
                strength: f64::from(e.strength),
                rest: 0.0,
                broken: false,
                yielded: false,
            };
            el.rest = el.angle(&self.x);
            self.elements.push(el);
        }
        for sn in &desc.slidenodes {
            let rail = &desc.rails[sn.rail as usize];
            self.slides.push(OracleSlide {
                node: sn.node + first,
                rail: rail.nodes.iter().map(|&k| k + first).collect(),
                looped: rail.looped,
                capped: rail.capped,
                attached: sn.attached,
                tolerance: f64::from(sn.tolerance),
                spring: f64::from(sn.spring),
                damp: f64::from(sn.damp),
                strength: f64::from(sn.strength),
                cap_strength: f64::from(sn.cap_strength),
            });
        }
        self.bodies.push(rec);
        self.grid.fresh = 0;
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
        // Elements: an angular spring k over an arm L is k/L² across it.
        for e in &self.elements {
            let n = e.count();
            for i in 0..n {
                let arm = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (self.x[e.nodes[i] as usize] - self.x[e.nodes[j] as usize]).length())
                    .fold(f64::MAX, f64::min)
                    .max(1e-3);
                ksum[e.nodes[i] as usize] += 4.0 * e.spring / (arm * arm);
                csum[e.nodes[i] as usize] +=
                    4.0 * e.damp * self.settings.damping_scale / (arm * arm);
            }
        }
        for sl in &self.slides {
            for k in std::iter::once(sl.node).chain(sl.rail.iter().copied()) {
                ksum[k as usize] += sl.spring;
                csum[k as usize] += sl.damp;
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
                if b.group != 0 && !b.group_passive {
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
        self.element_forces(&mut f);
        self.slide_forces(&mut f);
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

    /// LBEAMs and torsion bars: −k(angle − rest)∂angle/∂x and an angular dashpot,
    /// with rest-angle plasticity and breaking every substep (as the beams).
    fn element_forces(&mut self, f: &mut [DVec3]) {
        let st = self.settings;
        for e in &mut self.elements {
            if e.broken {
                continue;
            }
            let ang = e.angle(&self.x);
            let err = match e.kind {
                ElementKind::Lbeam => ang - e.rest,
                ElementKind::Torsion => crate::elements::wrap(ang - e.rest),
            };
            if st.plasticity {
                let (y, b) = crate::elements::angular_plastic(
                    &mut e.rest,
                    &mut e.deform,
                    e.strength,
                    e.spring,
                    err,
                );
                e.yielded |= y;
                if b && st.breaking {
                    e.broken = true;
                    continue;
                }
            }
            let err = match e.kind {
                ElementKind::Lbeam => ang - e.rest,
                ElementKind::Torsion => crate::elements::wrap(ang - e.rest),
            };
            let jac = e.jacobian(&self.x);
            let n = e.count();
            let rate: f64 = (0..n)
                .map(|i| jac[i].dot(self.v[e.nodes[i] as usize]))
                .sum();
            let torque = e.spring * err + e.damp * st.damping_scale * rate;
            for i in 0..n {
                f[e.nodes[i] as usize] -= jac[i] * torque;
            }
        }
    }

    /// Slidenodes: a spring to the rail across it (outside the tolerance), the
    /// reaction on the rail's segment, end caps; it comes off past its strength,
    /// or off an uncapped end.
    fn slide_forces(&mut self, f: &mut [DVec3]) {
        for sl in &mut self.slides {
            if !sl.attached {
                continue;
            }
            let rail: Vec<DVec3> = sl.rail.iter().map(|&k| self.x[k as usize]).collect();
            let p = self.x[sl.node as usize];
            let (q, t, seg, u) = crate::elements::rail_closest(p, &rail, sl.looped);
            let last = rail.len() - 2;
            let off_end = !sl.looped
                && ((seg == 0 && u <= 0.0 && (p - q).dot(t) < -1e-9)
                    || (seg == last && u >= 1.0 && (p - q).dot(t) > 1e-9));
            if off_end && !sl.capped {
                sl.attached = false;
                continue;
            }
            let d = p - q;
            let along = if off_end { DVec3::ZERO } else { t * d.dot(t) };
            let perp = d - along;
            let len = perp.length();
            if len <= sl.tolerance || len < 1e-12 {
                continue;
            }
            let dir = perp / len;
            let (ia, ib) = (
                sl.rail[seg] as usize,
                sl.rail[(seg + 1) % sl.rail.len()] as usize,
            );
            let vq = self.v[ia] * (1.0 - u) + self.v[ib] * u;
            let rel = (self.v[sl.node as usize] - vq).dot(dir);
            let force = sl.spring * (len - sl.tolerance) + sl.damp * rel;
            let limit = if off_end {
                sl.cap_strength
            } else {
                sl.strength
            };
            if force.abs() > limit {
                sl.attached = false;
                continue;
            }
            let fv = dir * force;
            f[sl.node as usize] -= fv;
            f[ia] += fv * (1.0 - u);
            f[ib] += fv * u;
        }
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

    /// Rebuilds the triangle grid every `GRID_REUSE` substeps. Triangles are
    /// padded by the pair range plus twice the fastest node's travel until the
    /// next rebuild (a node and a triangle each move at most that far), so every
    /// substep in between finds the same candidates as a fresh grid.
    fn rebuild_grid(&mut self) {
        if self.grid.fresh > 0 {
            self.grid.fresh -= 1;
            return;
        }
        self.grid.fresh = GRID_REUSE - 1;
        let vmax = self.v.iter().map(|v| v.length()).fold(0.0f64, f64::max);
        // Acceleration over the window is bounded loosely by doubling the travel.
        let travel = 2.0 * vmax * f64::from(GRID_REUSE) / self.hz + 0.01;
        let pad = self.settings.pair_range + self.settings.tri_thickness + 2.0 * travel;
        self.grid.cells.clear();
        for (ti, t) in self.tris.iter().enumerate() {
            if self.tri_broken(ti) {
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
            .chain(std::iter::once(&t.group_beam))
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
        // This substep's planes (unit normal; none when torn or degenerate).
        let planes: Vec<Option<DVec3>> = (0..self.tris.len())
            .map(|ti| {
                if self.tri_broken(ti) {
                    return None;
                }
                let [a, b, c] = self.tris[ti].nodes.map(|k| self.x[k as usize]);
                let nr = (b - a).cross(c - a);
                let l = nr.length();
                (l > 1e-12).then(|| nr / l)
            })
            .collect();
        let band = st.pair_range + st.tri_thickness + 0.05;
        let mut live: HashMap<(u32, u32), f64> = HashMap::with_capacity(self.pairs.len());
        for i in 0..self.x.len() {
            if !(self.contacter[i] || self.self_collision[i]) || self.fixed[i] {
                continue;
            }
            let p = self.x[i];
            let Some(cands) = self.grid.cells.get(&self.grid.key(p)) else {
                continue;
            };
            let me = self.node_body[i];
            for &ti in cands {
                let t = self.tris[ti as usize];
                if t.body == me {
                    if !st.self_collision
                        || !self.self_collision[i]
                        || t.nodes.contains(&(i as u32))
                    {
                        continue;
                    }
                    let skip = match self.node_group[i] {
                        Some(g) => g != 0 && t.group == g,
                        None => self.excluded.contains(&(i as u32, ti)),
                    };
                    if skip {
                        continue;
                    }
                } else if !self.contacter[i] {
                    continue;
                }
                let Some(nrm) = planes[ti as usize] else {
                    continue;
                };
                let a = self.x[t.nodes[0] as usize];
                let d = (p - a).dot(nrm);
                if d.abs() > band {
                    continue;
                }
                let [b, c] = [t.nodes[1], t.nodes[2]].map(|k| self.x[k as usize]);
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
                if (q - (p - nrm * d)).length() > 0.05 {
                    continue;
                }
                live.insert(key, side);
                if !inside {
                    continue;
                }
                // Anti-clip only (non-collidable): no force in front of the face.
                let thickness = if t.collide {
                    st.tri_thickness + t.bias
                } else {
                    0.0
                };
                let pen = thickness - side * d;
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
            spring_exp: 1.0e4,
            damp_exp: 0.0,
            deform_limit: f64::INFINITY,
            group_passive: false,
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

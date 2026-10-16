//! Engine-facing stress-solver contract, so implementations can be swapped.
//!
//! A rigid-body engine (PhysX) owns cluster motion and contacts. Once per frame it
//! hands the stress solver the clusters' rigid state and the contact impulses their
//! chunks received; the solver returns the fractures to apply (child bodies with
//! their chunk sets and inherited velocities) and per-bond state for rendering, audio
//! and VFX. Applying fractures to the engine is shared code on the engine side.
//!
//! [`StressSolverApi`] is the contract; [`EngineCoupledSolver`] implements it with the
//! reference solver. Contact impulses go through [`ContactLoadFilter`], which splits
//! each into the impact that stopped the bodies' approach (bounded by momentum) and a
//! force sustained over the frame; an impact is never `impulse / dt`. It is either a
//! pulse spread over a physical impact duration (Hertz, or crush-limited for soft impactors), or,
//! since inertia is in the stress solve, an instantaneous velocity change of the struck
//! chunk ([`ImpactModel::VelocityCondition`]): momentum-exact and free of any assumed
//! duration — the bond network and the chunks' inertia shape the stress pulse.

use std::collections::HashMap;

use serde::Serialize;

use crate::joint::FailureMode;
use crate::math::{Pose, Vec3};
use crate::scene::{CrushDesc, Scene, SolveMode};
use crate::solver::{ChunkLoads, ReferenceSolver, SolverEvent};

/// Stable id of a cluster (rigid body) across frames.
pub type ClusterId = u64;

/// One contact impulse delivered to a chunk during the last engine step.
#[derive(Clone, Debug)]
pub struct ContactImpulse {
    pub structure: usize,
    pub chunk: usize,
    /// World contact point.
    pub point: Vec3,
    /// Impulse applied to the chunk's body (N s, world).
    pub impulse: Vec3,
    /// Mass of the other body (infinite for static geometry).
    pub other_mass: f64,
    /// Closing speed along the contact normal before the contact (m/s).
    pub approach_speed: f64,
    /// Young's modulus and size (radius) of the other body, for the Hertz duration.
    pub other_modulus: f64,
    pub other_radius: f64,
    /// Crush model of a soft impactor (vehicle).
    pub crush: Option<CrushDesc>,
    /// The other body, by any id the engine likes (`None`: static geometry). A frame's
    /// contact points between one chunk and one body are one manifold and are filtered
    /// as one contact.
    pub other: Option<u64>,
}

/// Rigid state of a cluster as integrated by the engine.
#[derive(Clone, Copy, Debug)]
pub struct ClusterMotion {
    pub id: ClusterId,
    pub pose: Pose,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct FrameInput {
    pub dt: f64,
    pub motion: Vec<ClusterMotion>,
    pub contacts: Vec<ContactImpulse>,
}

/// A new rigid body created by fracture.
#[derive(Clone, Debug, Serialize)]
pub struct ChildBody {
    pub id: ClusterId,
    pub structure: usize,
    pub chunks: Vec<usize>,
    pub anchored: bool,
    pub pose: [f64; 7],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub mass: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Fracture {
    pub parent: ClusterId,
    pub children: Vec<ChildBody>,
}

#[derive(Clone, Debug, Default)]
pub struct FrameOutput {
    pub fractures: Vec<Fracture>,
    pub events: Vec<SolverEvent>,
}

/// Per-bond state exposed to rendering (crack decals) and other systems.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct BondReport {
    pub chunks: (usize, usize),
    pub damage: f64,
    pub crush: f64,
    /// Strength lost to static fatigue (0..1).
    pub fatigue_loss: f64,
    /// Largest stress-over-strength ratio at the last evaluation.
    pub utilization: f64,
    pub mode: Option<FailureMode>,
    pub broken: bool,
    pub tension: f64,
    pub shear: f64,
    pub compression: f64,
}

/// The contract every stress-solver implementation (reference CPU, CUDA) provides.
pub trait StressSolverApi {
    /// Advance one engine frame.
    fn step(&mut self, input: &FrameInput) -> FrameOutput;
    /// Every live cluster with its chunks and rigid state.
    fn clusters(&self) -> Vec<ChildBody>;
    /// State of every bond of a structure (live and broken).
    fn bond_reports(&self, structure: usize) -> Vec<BondReport>;
}

/// How the filter turns an impact impulse into stress-solver input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImpactModel {
    /// A force pulse over the Hertz contact duration (crush plateau first for a soft
    /// impactor with a crush law).
    #[default]
    Pulse,
    /// The impulse changes the struck chunk's (hidden) velocity at the start of the
    /// frame; the engine already gave the whole body its rigid share. Crush laws are
    /// not applied (the engine's contact decides the impulse).
    VelocityCondition,
}

/// Turns per-frame contact impulses into a substep force history.
///
/// An engine reports, per touching pair, the impulse it applied over its step. Part of
/// it stopped the bodies' approach (an impact), the rest held them against steady
/// forces (gravity, pushing). An impact at closing speed `v` between bodies of reduced
/// mass `m` delivers at most `2 m v` (a perfectly elastic one), so the impact part is
/// `min(J, 2 m v)`, delivered over its Hertz contact duration, and the remainder is a
/// force sustained over the frame. `v` counts only what exceeds the engine's resting
/// cycle, `|g| dt` (see `ingest`).
#[derive(Clone, Debug, Default)]
pub struct ContactLoadFilter {
    pub impact_model: ImpactModel,
    /// This frame's sustained contact force and its point, per (structure, chunk,
    /// other body).
    sustained: std::collections::BTreeMap<(usize, usize, Option<u64>), (Vec3, Vec3)>,
    pulses: Vec<Pulse>,
    /// Impact impulses waiting to be applied as velocity conditions.
    kicks: Vec<(usize, usize, Vec3, Vec3)>,
}

#[derive(Clone, Debug)]
struct Pulse {
    structure: usize,
    chunk: usize,
    point: Vec3,
    direction: Vec3,
    start: f64,
    /// Plateau force and duration (crush-limited part), then a half-sine of the rest.
    plateau: f64,
    plateau_time: f64,
    peak: f64,
    duration: f64,
}

impl Pulse {
    fn magnitude(&self, t: f64) -> f64 {
        let tau = t - self.start;
        if tau < 0.0 {
            return 0.0;
        }
        if tau < self.plateau_time {
            return self.plateau;
        }
        let s = tau - self.plateau_time;
        if s > self.duration {
            0.0
        } else {
            self.peak * (std::f64::consts::PI * s / self.duration).sin()
        }
    }
    fn end(&self) -> f64 {
        self.start + self.plateau_time + self.duration
    }
}

/// One contact per (structure, chunk, other body): the summed impulse, with the point
/// and closing speed of its centre of pressure (impulse-weighted means; a rocking body
/// approaches at one corner and recedes at another). Engines report a
/// manifold (several points for one touching pair); the impact bound is per pair, not
/// per point.
fn manifolds(contacts: &[ContactImpulse]) -> Vec<ContactImpulse> {
    let mut index: HashMap<(usize, usize, Option<u64>), usize> = HashMap::new();
    let mut out: Vec<ContactImpulse> = Vec::new();
    let mut weight: Vec<f64> = Vec::new();
    for c in contacts {
        let w = c.impulse.norm();
        match index.get(&(c.structure, c.chunk, c.other)) {
            Some(&i) => {
                let m = &mut out[i];
                let total = weight[i] + w;
                if total > 0.0 {
                    m.point = (m.point * weight[i] + c.point * w) * (1.0 / total);
                    m.approach_speed = (m.approach_speed * weight[i] + c.approach_speed * w) / total;
                }
                m.impulse += c.impulse;
                weight[i] = total;
            }
            None => {
                index.insert((c.structure, c.chunk, c.other), out.len());
                out.push(c.clone());
                weight.push(w);
            }
        }
    }
    out
}

/// Hertz contact duration of a sphere (reduced mass `m`, radius `r`) on a half-space,
/// effective modulus `e`, approach speed `v`.
pub fn hertz_duration(m: f64, r: f64, e: f64, v: f64) -> f64 {
    let v = v.abs().max(1e-3);
    2.87 * (m * m / (r * e * e * v)).powf(0.2)
}

impl ContactLoadFilter {
    pub fn new() -> ContactLoadFilter {
        ContactLoadFilter::default()
    }

    /// Ingest one frame of impulses (the frame starts at `t0` and lasts `dt`).
    pub fn ingest(&mut self, solver: &ReferenceSolver, t0: f64, dt: f64, contacts: &[ContactImpulse]) {
        self.sustained.clear();
        // The engine resolves contact once per step: a body held against gravity is
        // re-accelerated towards its support by up to |g| dt every step and stopped
        // again. Closing speeds within that band are how a discrete engine holds a
        // resting contact; only the closing speed above it is a resolvable impact.
        let resolution = solver.config.gravity.norm() * dt;
        for c in &manifolds(contacts) {
            let j = c.impulse.norm();
            if j == 0.0 {
                continue;
            }
            let ch = &solver.structures[c.structure].chunks[c.chunk];
            // An impact loads the whole struck body (an anchored one is immovable).
            let cluster = &solver.clusters[solver.chunks[c.structure][c.chunk].cluster];
            let m_body = if cluster.anchored { f64::INFINITY } else { cluster.mass };
            let m = match (m_body.is_finite(), c.other_mass.is_finite()) {
                (true, true) => m_body * c.other_mass / (m_body + c.other_mass),
                (true, false) => m_body,
                (false, true) => c.other_mass,
                (false, false) => ch.mass,
            };
            let j_impact = j.min(2.0 * m * (c.approach_speed - resolution).max(0.0));
            let dir = c.impulse / j;
            if j_impact < j {
                let f = dir * ((j - j_impact) / dt);
                self.sustained.insert((c.structure, c.chunk, c.other), (f, c.point));
            }
            if j_impact <= 0.0 {
                continue;
            }
            if self.impact_model == ImpactModel::VelocityCondition {
                self.kicks.push((c.structure, c.chunk, dir * j_impact, c.point));
                continue;
            }
            let e_star = 1.0 / ((1.0 - ch.poisson_ratio.powi(2)) / ch.youngs_modulus + 1.0 / c.other_modulus);
            let r = c.other_radius.min(ch.volume().cbrt());
            let duration = hertz_duration(m, r, e_star, c.approach_speed);
            let (mut plateau, mut plateau_time, mut rest_j) = (0.0, 0.0, j_impact);
            if let Some(cr) = c.crush {
                // Crushing absorbs up to its energy at the plateau force; the
                // remaining impulse arrives as a stiff (bottomed-out) pulse.
                let ke = 0.5 * m * c.approach_speed * c.approach_speed;
                let e_crush = ke.min(cr.energy);
                let v1 = (c.approach_speed * c.approach_speed - 2.0 * e_crush / m).max(0.0).sqrt();
                let jc = (m * (c.approach_speed - v1)).min(j_impact);
                plateau = cr.max_force;
                plateau_time = jc / cr.max_force;
                rest_j = j_impact - jc;
            }
            let peak = std::f64::consts::PI * rest_j / (2.0 * duration);
            self.pulses.push(Pulse {
                structure: c.structure,
                chunk: c.chunk,
                point: c.point,
                direction: dir,
                start: t0,
                plateau,
                plateau_time,
                peak,
                duration,
            });
        }
        self.pulses.retain(|p| p.end() > t0);
    }

    /// Force on every chunk at time `t`.
    pub fn loads_at(&self, solver: &ReferenceSolver, t: f64, loads: &mut ChunkLoads) {
        for (&(s, c, _), &(f, p)) in &self.sustained {
            if solver.chunks[s][c].active {
                loads.add_at(s, c, f, p, solver.chunk_position(s, c));
            }
        }
        for p in &self.pulses {
            if solver.chunks[p.structure][p.chunk].active {
                let f = p.direction * p.magnitude(t);
                loads.add_at(p.structure, p.chunk, f, p.point, solver.chunk_position(p.structure, p.chunk));
            }
        }
    }

    /// Apply the pending velocity-condition impacts to the solver.
    pub fn apply_kicks(&mut self, solver: &mut ReferenceSolver) {
        for (s, c, impulse, point) in self.kicks.drain(..) {
            if solver.chunks[s][c].active {
                solver.apply_chunk_impulse(s, c, impulse, point);
            }
        }
    }

    /// Total impulse of all pulses (for momentum checks).
    pub fn pulse_impulse(&self) -> f64 {
        self.pulses.iter().map(|p| p.plateau * p.plateau_time + 2.0 * p.peak * p.duration / std::f64::consts::PI).sum()
    }
}

/// The reference solver driven by an external rigid-body engine.
#[derive(Clone)]
pub struct EngineCoupledSolver {
    pub solver: ReferenceSolver,
    pub filter: ContactLoadFilter,
    loads: ChunkLoads,
}

impl EngineCoupledSolver {
    pub fn new(scene: &Scene) -> EngineCoupledSolver {
        let mut solver = ReferenceSolver::new(scene);
        solver.config.integrate_rigid = false;
        let loads = ChunkLoads::new(&solver);
        let mut s = EngineCoupledSolver { solver, filter: ContactLoadFilter::new(), loads };
        s.loads.clear();
        s.solver.gravity_prestress();
        s.solver.mark_implicit_step_start();
        s
    }

    /// Accept a fracture verdict at the current (start-of-tick) state: break the given
    /// bonds, split, and return the resulting fractures for the engine to install
    /// before it re-solves the tick (a correction pass).
    pub fn accept_verdict(&mut self, bonds: &[(usize, usize)]) -> FrameOutput {
        let n_events = self.solver.events.len();
        self.solver.break_bonds(bonds);
        self.output_since(n_events)
    }

    /// Fractures and events recorded since event index `n_events`.
    fn output_since(&self, n_events: usize) -> FrameOutput {
        let events: Vec<SolverEvent> = self.solver.events[n_events..].to_vec();
        // A child can split again within the same frame: the engine only knows the
        // bodies that existed before it, so each of those is reported with the pieces it
        // ended as (the leaves of this frame's splits), and the intermediate clusters
        // are not reported at all.
        let splits: std::collections::HashMap<ClusterId, Vec<ClusterId>> = events
            .iter()
            .filter_map(|e| match e {
                SolverEvent::Split { parent, children, .. } => Some((*parent, children.clone())),
                _ => None,
            })
            .collect();
        let born: std::collections::HashSet<ClusterId> = splits.values().flatten().copied().collect();
        fn leaves(id: ClusterId, splits: &std::collections::HashMap<ClusterId, Vec<ClusterId>>, out: &mut Vec<ClusterId>) {
            match splits.get(&id) {
                Some(children) => children.iter().for_each(|&c| leaves(c, splits, out)),
                None => out.push(id),
            }
        }
        let mut fractures = Vec::new();
        for e in &events {
            if let SolverEvent::Split { parent, .. } = e {
                if born.contains(parent) {
                    continue;
                }
                let mut ids = Vec::new();
                leaves(*parent, &splits, &mut ids);
                let kids = ids
                    .iter()
                    .filter_map(|id| self.solver.clusters.iter().position(|c| c.id == *id))
                    .map(|ci| self.child_body(ci))
                    .collect();
                fractures.push(Fracture { parent: *parent, children: kids });
            }
        }
        FrameOutput { fractures, events }
    }

    fn child_body(&self, ci: usize) -> ChildBody {
        let c = &self.solver.clusters[ci];
        let p = c.pose;
        ChildBody {
            id: c.id,
            structure: c.structure,
            chunks: c.chunks.clone(),
            anchored: c.anchored,
            pose: [p.position.x, p.position.y, p.position.z, p.rotation.w, p.rotation.x, p.rotation.y, p.rotation.z],
            velocity: c.velocity.to_array(),
            angular_velocity: c.angular_velocity.to_array(),
            mass: c.mass,
        }
    }
}

impl StressSolverApi for EngineCoupledSolver {
    fn step(&mut self, input: &FrameInput) -> FrameOutput {
        for m in &input.motion {
            if let Some(c) = self.solver.clusters.iter_mut().find(|c| c.id == m.id) {
                if !c.anchored {
                    c.pose = m.pose;
                    c.velocity = m.velocity;
                    c.angular_velocity = m.angular_velocity;
                }
            }
        }
        let t0 = self.solver.time;
        self.filter.ingest(&self.solver, t0, input.dt, &input.contacts);
        self.filter.apply_kicks(&mut self.solver);
        let n_events = self.solver.events.len();
        if self.solver.config.mode == SolveMode::Implicit {
            // One implicit step per frame with the frame's average filtered load.
            let samples = 16;
            let mut average = ChunkLoads::new(&self.solver);
            for k in 0..samples {
                self.loads.resize(&self.solver);
                self.filter.loads_at(&self.solver, t0 + (k as f64 + 0.5) * input.dt / samples as f64, &mut self.loads);
                average.add_scaled(&self.loads, 1.0 / samples as f64);
            }
            for ci in 0..self.solver.clusters.len() {
                self.solver.advance_rigid(ci, input.dt, &average);
            }
            self.solver.time += input.dt;
            self.solver.implicit_step_all(&average, input.dt);
        } else {
            let dt_sub = self.solver.stable_dt().min(input.dt);
            let n = (input.dt / dt_sub).ceil().max(1.0) as usize;
            let h = input.dt / n as f64;
            for _ in 0..n {
                self.loads.resize(&self.solver);
                self.loads.clear();
                self.filter.loads_at(&self.solver, self.solver.time, &mut self.loads);
                self.solver.substep(h, &self.loads);
            }
        }
        self.output_since(n_events)
    }

    fn clusters(&self) -> Vec<ChildBody> {
        (0..self.solver.clusters.len()).map(|ci| self.child_body(ci)).collect()
    }

    fn bond_reports(&self, structure: usize) -> Vec<BondReport> {
        self.solver.bonds[structure]
            .iter()
            .map(|b| BondReport {
                chunks: (b.geometry.a, b.geometry.b),
                damage: b.joint.damage,
                crush: b.joint.crush,
                fatigue_loss: 1.0 - crate::joint::fatigue_factor(&b.strength, &b.joint),
                utilization: b.joint.utilization,
                mode: b.joint.mode,
                broken: !b.connected(),
                tension: b.measures.tension,
                shear: b.measures.shear,
                compression: b.measures.compression,
            })
            .collect()
    }
}

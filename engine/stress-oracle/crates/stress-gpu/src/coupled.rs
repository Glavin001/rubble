//! The GPU stress solver driven by an external rigid-body engine (PhysX): stress-ref's
//! `EngineCoupledSolver` (api.rs) with its explicit substeps on the GPU.
//!
//! Each engine frame the engine hands over the clusters' rigid state and the contact
//! impulses their chunks received. The contact load filter (api.rs `ContactLoadFilter`,
//! ported here because its state is private) splits each manifold's impulse into the
//! impact that stopped the approach, delivered as a pulse over its Hertz duration (crush
//! plateau first for a soft impactor), and a force sustained over the frame. Both become
//! GPU load terms (`LOAD_POINT` with a constant or `Pulse` time function), so the loads
//! are evaluated on the GPU at every substep exactly as `loads_at` does on the host.
//! Clusters do not integrate their own rigid motion (`integrate_rigid` off): the engine
//! owns it, and the stress solve runs the hidden deformation, damage and fracture.

use std::collections::{BTreeMap, HashMap};

use stress_ref::api::{hertz_duration, BondReport, ChildBody, ContactImpulse, FrameInput, FrameOutput, Fracture, StressSolverApi};
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, SolveMode};
use stress_ref::solver::{Activity, ReferenceSolver, SolverEvent};

use crate::gpu::Gpu;
use crate::loads::{Function, LoadTerm, LOAD_POINT};
use crate::solver::GpuSolver;

/// An impact pulse (api.rs `Pulse`).
#[derive(Clone, Debug)]
struct Pulse {
    structure: usize,
    chunk: usize,
    point: Vec3,
    direction: Vec3,
    start: f64,
    plateau: f64,
    plateau_time: f64,
    peak: f64,
    duration: f64,
}

impl Pulse {
    fn end(&self) -> f64 {
        self.start + self.plateau_time + self.duration
    }
}

/// api.rs `manifolds`: one contact per (structure, chunk, other body), with the summed
/// impulse and the impulse-weighted point and approach speed.
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

/// api.rs `ContactLoadFilter` with the default `ImpactModel::Pulse`.
#[derive(Clone, Debug, Default)]
pub struct LoadFilter {
    sustained: BTreeMap<(usize, usize, Option<u64>), (Vec3, Vec3)>,
    pulses: Vec<Pulse>,
}

impl LoadFilter {
    /// api.rs `ingest`: one frame of impulses (the frame starts at `t0`, lasts `dt`).
    pub fn ingest(&mut self, solver: &ReferenceSolver, t0: f64, dt: f64, contacts: &[ContactImpulse]) {
        self.sustained.clear();
        let resolution = solver.config.gravity.norm() * dt;
        for c in &manifolds(contacts) {
            let j = c.impulse.norm();
            if j == 0.0 {
                continue;
            }
            let ch = &solver.structures[c.structure].chunks[c.chunk];
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
            let e_star = 1.0 / ((1.0 - ch.poisson_ratio.powi(2)) / ch.youngs_modulus + 1.0 / c.other_modulus);
            let r = c.other_radius.min(ch.volume().cbrt());
            let duration = hertz_duration(m, r, e_star, c.approach_speed);
            let (mut plateau, mut plateau_time, mut rest_j) = (0.0, 0.0, j_impact);
            if let Some(cr) = c.crush {
                let ke = 0.5 * m * c.approach_speed * c.approach_speed;
                let e_crush = ke.min(cr.energy);
                let v1 = (c.approach_speed * c.approach_speed - 2.0 * e_crush / m).max(0.0).sqrt();
                let jc = (m * (c.approach_speed - v1)).min(j_impact);
                plateau = cr.max_force;
                plateau_time = jc / cr.max_force;
                rest_j = j_impact - jc;
            }
            let peak = std::f64::consts::PI * rest_j / (2.0 * duration);
            self.pulses.push(Pulse { structure: c.structure, chunk: c.chunk, point: c.point, direction: dir, start: t0, plateau, plateau_time, peak, duration });
        }
        self.pulses.retain(|p| p.end() > t0);
    }

    /// The filter's loads as GPU load terms (api.rs `loads_at`): a force at a point fixed
    /// in the struck cluster's frame, constant (sustained) or a pulse. Order: sustained,
    /// then pulses, as `loads_at` adds them.
    pub fn terms(&self, solver: &ReferenceSolver) -> Vec<LoadTerm> {
        let point_term = |s: usize, c: usize, dir: Vec3, point: Vec3, function: Function| -> Option<LoadTerm> {
            if !solver.chunks[s][c].active {
                return None;
            }
            let cl = &solver.clusters[solver.chunks[s][c].cluster];
            let arm = cl.pose.inverse_transform_point(point) - solver.structures[s].chunks[c].center;
            Some(LoadTerm { structure: s, chunk: c, kind: LOAD_POINT, dir, area: 0.0, arm, function })
        };
        let mut out = Vec::new();
        for (&(s, c, _), &(f, p)) in &self.sustained {
            let mag = f.norm();
            if mag > 0.0 {
                out.extend(point_term(s, c, f * (1.0 / mag), p, Function::Constant(mag)));
            }
        }
        for p in &self.pulses {
            let function = Function::Pulse { start: p.start, plateau: p.plateau, plateau_time: p.plateau_time, peak: p.peak, duration: p.duration };
            out.extend(point_term(p.structure, p.chunk, p.direction, p.point, function));
        }
        out
    }
}

/// Integral of an impact pulse's magnitude (`Function::Pulse`) over `[a, b]`.
fn pulse_integral(start: f64, plateau: f64, plateau_time: f64, peak: f64, duration: f64, a: f64, b: f64) -> f64 {
    let overlap = |lo: f64, hi: f64| (hi.min(b) - lo.max(a)).max(0.0);
    let mut sum = plateau * overlap(start, start + plateau_time);
    let (s0, s1) = (start + plateau_time, start + plateau_time + duration);
    let (lo, hi) = (a.max(s0), b.min(s1));
    if hi > lo && duration > 0.0 {
        let w = std::f64::consts::PI / duration;
        sum += peak / w * ((w * (lo - s0)).cos() - (w * (hi - s0)).cos());
    }
    sum
}

impl LoadFilter {
    /// The filter's loads averaged exactly over the frame `[t0, t0 + dt]`: each pulse as a
    /// constant force of its mean over the frame (its direction and point are fixed), so
    /// a quasi-static frame solves under the frame's average load (impulse / dt for a
    /// pulse inside the frame, the force Blast applies) from one load evaluation instead
    /// of the average of many substep samples.
    pub fn frame_average_terms(&self, solver: &ReferenceSolver, t0: f64, dt: f64) -> Vec<LoadTerm> {
        self.terms(solver)
            .into_iter()
            .filter_map(|mut t| {
                if let Function::Pulse { start, plateau, plateau_time, peak, duration } = t.function {
                    let mean = pulse_integral(start, plateau, plateau_time, peak, duration, t0, t0 + dt) / dt;
                    if mean == 0.0 {
                        return None;
                    }
                    t.function = Function::Constant(mean);
                }
                Some(t)
            })
            .collect()
    }
}

/// Host seconds per stage of the coupled frames so far.
#[derive(Clone, Debug, Default)]
pub struct CoupledProfile {
    pub frames: u64,
    pub sync: f64,
    pub filter: f64,
    pub refresh: f64,
    pub step: f64,
    pub output: f64,
}

/// The GPU stress solver under an engine (api.rs `EngineCoupledSolver`).
pub struct GpuCoupledSolver {
    pub solver: GpuSolver,
    pub filter: LoadFilter,
    pub profile: CoupledProfile,
}

impl GpuCoupledSolver {
    pub fn new(gpu: &Gpu, scene: &Scene) -> Result<GpuCoupledSolver, String> {
        // api.rs `EngineCoupledSolver::new`: no rigid integration, gravity prestress.
        let mut mirror = ReferenceSolver::new(scene);
        mirror.config.integrate_rigid = false;
        let mut solver = GpuSolver::new(gpu, mirror, None, Vec::new())?;
        solver.own_contacts = false;
        let clusters: Vec<usize> = (0..solver.mirror.clusters.len()).filter(|&ci| solver.mirror.clusters[ci].anchored && !solver.mirror.clusters[ci].bonds.is_empty()).collect();
        if !clusters.is_empty() {
            let loads = stress_ref::solver::ChunkLoads::new(&solver.mirror);
            for (&ci, r) in clusters.iter().zip(solver.equilibrate(gpu, &clusters, &loads)) {
                if !r.converged {
                    return Err(format!("prestress did not converge (cluster {ci}, residual {:.3e})", r.residual));
                }
                // world.rs `prestress`: outside explicit mode the prestressed clusters
                // start settled (no external load beside gravity: norm 0).
                if solver.mirror.config.mode != SolveMode::Explicit {
                    solver.mirror.clusters[ci].activity = Activity::Settled;
                    solver.mirror.clusters[ci].settled_load_norm = 0.0;
                }
            }
            solver.rebuild(gpu, 0.0)?;
        }
        Ok(GpuCoupledSolver { solver, filter: LoadFilter::default(), profile: CoupledProfile::default() })
    }

    /// One engine frame on the GPU; the fractures and events it produced.
    pub fn step_gpu(&mut self, gpu: &Gpu, input: &FrameInput) -> Result<FrameOutput, String> {
        // The mirror's motion and history must be current before it is changed.
        let t = std::time::Instant::now();
        self.solver.sync(gpu);
        self.profile.sync += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let m = &mut self.solver.mirror;
        for motion in &input.motion {
            if let Some(c) = m.clusters.iter_mut().find(|c| c.id == motion.id) {
                if !c.anchored {
                    c.pose = motion.pose;
                    c.velocity = motion.velocity;
                    c.angular_velocity = motion.angular_velocity;
                }
            }
        }
        let t0 = m.time;
        self.filter.ingest(m, t0, input.dt, &input.contacts);
        self.solver.extra_terms = self.filter.terms(&self.solver.mirror);
        self.profile.filter += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        self.solver.refresh(gpu)?;
        self.profile.refresh += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let n_events = self.solver.mirror.events.len();
        // Adaptive mode: a settled cluster wakes on the GPU when its loads change, in the
        // middle of the frame; the substep must already resolve it. Those that can wake
        // this frame (engine contacts reach them, or the engine moves them) count as
        // active for the step.
        let adaptive = self.solver.mirror.config.mode == SolveMode::Adaptive;
        let mut may_wake = Vec::new();
        if adaptive {
            let m = &mut self.solver.mirror;
            for t in &self.solver.extra_terms {
                may_wake.push(m.chunks[t.structure][t.chunk].cluster);
            }
            may_wake.extend((0..m.clusters.len()).filter(|&ci| !m.clusters[ci].anchored));
            may_wake.sort_unstable();
            may_wake.dedup();
            may_wake.retain(|&ci| m.clusters[ci].activity == Activity::Settled);
            for &ci in &may_wake {
                m.clusters[ci].activity = Activity::Active;
            }
        }
        let dt_sub = self.solver.mirror.stable_dt().min(input.dt);
        if std::env::var("STRESS_GPU_COUPLED_TRACE").is_ok() {
            let m = &self.solver.mirror;
            let active: Vec<String> = (0..m.clusters.len())
                .filter(|&ci| m.clusters[ci].activity == Activity::Active && !may_wake.contains(&ci))
                .map(|ci| format!("{}({}{})", ci, m.clusters[ci].chunks.len(), if m.clusters[ci].anchored { "a" } else { "" }))
                .collect();
            let woken: Vec<String> = may_wake.iter().map(|&ci| format!("{}({}{})", ci, m.clusters[ci].chunks.len(), if m.clusters[ci].anchored { "a" } else { "" })).collect();
            eprintln!("t {:.3}: {} clusters, active {:?}, may wake {:?}, {} terms, substep {:.2e}", m.time, m.clusters.len(), active, woken, self.solver.extra_terms.len(), dt_sub);
        }
        for &ci in &may_wake {
            self.solver.mirror.clusters[ci].activity = Activity::Settled;
        }
        if !may_wake.is_empty() {
            self.solver.coverage |= crate::coverage::COUPLED_MAY_WAKE;
        }
        let n = (input.dt / dt_sub).ceil().max(1.0) as usize;
        let h = input.dt / n as f64;
        self.solver.step(gpu, h, n)?;
        if adaptive && self.solver.adaptive_tail(gpu, input.dt)? {
            self.solver.rebuild(gpu, 0.0)?;
        }
        self.profile.step += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        self.solver.sync(gpu);
        let out = self.output_since(n_events);
        self.profile.output += t.elapsed().as_secs_f64();
        self.profile.frames += 1;
        Ok(out)
    }

    fn output_since(&self, n_events: usize) -> FrameOutput {
        let m = &self.solver.mirror;
        let events: Vec<SolverEvent> = m.events[n_events..].to_vec();
        let mut fractures = Vec::new();
        for e in &events {
            if let SolverEvent::Split { parent, children, .. } = e {
                let kids = children.iter().filter_map(|id| m.clusters.iter().position(|c| c.id == *id)).map(|ci| child_body(m, ci)).collect();
                fractures.push(Fracture { parent: *parent, children: kids });
            }
        }
        FrameOutput { fractures, events }
    }
}

fn child_body(m: &ReferenceSolver, ci: usize) -> ChildBody {
    let c = &m.clusters[ci];
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

/// The GPU coupled solver as a `StressSolverApi`, owning its device.
pub struct GpuStressSolverApi {
    pub gpu: Gpu,
    pub coupled: GpuCoupledSolver,
}

impl GpuStressSolverApi {
    pub fn new(scene: &Scene) -> Result<GpuStressSolverApi, String> {
        Self::with_gpu(Gpu::new()?, scene)
    }

    /// On a device the caller already has (one device for many runs: the test tiers).
    pub fn with_gpu(gpu: Gpu, scene: &Scene) -> Result<GpuStressSolverApi, String> {
        let coupled = GpuCoupledSolver::new(&gpu, scene)?;
        Ok(GpuStressSolverApi { gpu, coupled })
    }
}

impl StressSolverApi for GpuStressSolverApi {
    fn step(&mut self, input: &FrameInput) -> FrameOutput {
        self.coupled.step_gpu(&self.gpu, input).expect("GPU stress frame")
    }

    fn clusters(&self) -> Vec<ChildBody> {
        let m = &self.coupled.solver.mirror;
        (0..m.clusters.len()).map(|ci| child_body(m, ci)).collect()
    }

    fn bond_reports(&self, structure: usize) -> Vec<BondReport> {
        let m = &self.coupled.solver.mirror;
        m.bonds[structure]
            .iter()
            .map(|b| BondReport {
                chunks: (b.geometry.a, b.geometry.b),
                damage: b.joint.damage,
                crush: b.joint.crush,
                fatigue_loss: 1.0 - stress_ref::joint::fatigue_factor(&b.strength, &b.joint),
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

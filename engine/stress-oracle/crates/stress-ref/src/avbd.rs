//! AVBD as a stress solver: the chunk network integrated by Augmented Vertex Block
//! Descent (Giles, Diaz, Yuksel, SIGGRAPH 2025) instead of explicit substeps. A
//! research prototype for `findings/avbd-stress.md`; nothing else uses it.
//!
//! **The model is the oracle's**, at chunk level, the way an AVBD engine holds it:
//! - every active chunk is a 6-DOF rigid body in world coordinates (mass and inertia
//!   from the structure), every impactor another;
//! - every bond is the oracle's damageable joint (`joint.rs`) with its true stiffness
//!   (`BondStiffness`, no solver penalty), its stiffness-proportional dashpot and its
//!   co-rotational kinematics (`bond.rs`; to first order the oracle's small-strain map);
//! - contact is the established sample-point penalty law (`world.rs`): the same sample
//!   points, stiffness shares, dashpot (capped at the oracle's own substep, so the law is
//!   one law whatever the integrator) and regularised Coulomb friction; chunk pairs
//!   keep the pre-existing-overlap memory;
//! - clusters are the connected components of connected bonds: a bond whose chunks end
//!   up in different components stops acting, contact between components starts.
//!
//! The world is initialised from an oracle [`World`] (its gravity prestress and state),
//! so both start from the same equilibrium.
//! It replicates the `established` methods (the rubble references': no Rayleigh mass
//! damping, sample-point contact); run the oracle with them to compare.
//!
//! **Three integrators share that force model** (`Integrator`):
//! - `Explicit`: the oracle's symplectic substep at the oracle's own (certified) substep:
//!   the truth, replicated, to check this chunk-level world against the oracle's;
//! - `BackwardEuler`: AVBD as published: per step `h`, minimise
//!   `1/(2 h^2) |x - x~|_M^2 + E(x)` by block Gauss-Seidel over a graph colouring, one
//!   6x6 Newton step per body per visit, warm-started from the previous acceleration;
//! - `Newmark`: the same block descent on Newmark's average-acceleration step
//!   (`beta = 1/4`; the oracle's `implicit.rs`), which is also a minimisation
//!   (`2/h^2 |x - x~|_M^2 + E(x)`) and has no numerical damping.
//!
//! **Damage inside the iteration** (`DamageMode`): `Implicit` evaluates each bond's law at
//! the trial deformation *from its history at the step's start* on every visit (the
//! damage the step's end state implies), with the law's frozen-history tangent (the
//! secant in the damage, positive semi-definite) as the block Hessian; nothing is
//! written back. `Frozen` holds damage during the iteration (the oracle's implicit
//! mode). Either way the history is committed once, at the accepted state.
//!
//! **Contact inside the solve** (`ContactMode`): contacts are found at the step's start
//! within a speculative margin (the bodies' travel in the step), linearised about it
//! (fixed lever arms and normals, AVBD's practice), and either keep the penalty law
//! (`Penalty`: the oracle's stiffness and dashpot) or become hard augmented-Lagrangian
//! constraints (`Hard`: AVBD's ramped penalty and multiplier, `alpha` stabilisation).

use std::collections::HashMap;
use std::time::Instant;

use crate::bond::{ChunkTurn, CoRotation, Mat6};
use crate::contact::OBox;
use crate::joint::JointResponse;
use crate::material::Material;
use crate::math::{Mat3, Pose, Quat, Vec3};
use crate::scene::{EventDesc, ImpactorShape, LoadDesc, ProbeKind, Scene, Support};
use crate::solver::RtBond;
use crate::world::{
    contact_stiffness, damping_ratio, penalty_force, World, FACE_PAIR_POINTS,
    FRICTION_REGULARIZATION, MAX_SET_VISCOSITY,
};

/// Time integration of the chunk network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    Explicit,
    BackwardEuler,
    Newmark,
}

/// How a bond's damage evolves inside one implicit step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DamageMode {
    /// The law at the trial deformation from the step-start history, every visit.
    Implicit,
    /// History (damage) frozen during the iteration.
    Frozen,
}

/// Contacts inside the implicit solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactMode {
    /// The oracle's penalty law (its stiffness and dashpot).
    Penalty,
    /// Hard non-penetration, AVBD's augmented Lagrangian.
    Hard,
}

#[derive(Clone, Debug)]
pub struct AvbdConfig {
    pub integrator: Integrator,
    /// The implicit step (ignored by `Explicit`, which takes the oracle's substep).
    pub step: f64,
    /// Block Gauss-Seidel sweeps per step (the most, with `tolerance`).
    pub iterations: usize,
    /// Stop a step once the sweep residual falls below this (relative); 0 = never.
    pub tolerance: f64,
    pub damage: DamageMode,
    pub contact: ContactMode,
    /// AVBD's parameters for hard contacts: stabilisation `alpha`, penalty growth `beta`
    /// (per unit violation, relative to the contact's law stiffness per metre), warm-start
    /// decay `gamma`.
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    /// Hard contacts: the penalty's cap, as a multiple of the law's stiffness.
    pub hard_cap: f64,
    /// Backward Euler's warm start: the previous step's acceleration (`true`, AVBD's
    /// default in spirit) or the inertial prediction under gravity and loads (`false`).
    /// Newmark always predicts with its own `a_n`.
    pub warm_previous: bool,
    /// Chebyshev semi-iterative acceleration of the sweeps (VBD's), with this spectral
    /// radius estimate; 0 = plain block Gauss-Seidel.
    pub chebyshev: f64,
    /// With a tolerance: a step whose best sweep residual has not fallen by 10 % over this
    /// many sweeps stops there (stalled: a non-smooth law cycling); 0 = never.
    pub stall: usize,
    /// Speculative contact margin as a multiple of the bodies' travel in a step.
    pub margin: f64,
    /// Find each step's contacts at the pose its start velocities predict for its end.
    pub predict_contacts: bool,
    /// Keep every step's residual history (sweep by sweep).
    pub trace_residuals: bool,
    /// Stop the run at this time (default the scene's duration).
    pub end_time: Option<f64>,
}

impl Default for AvbdConfig {
    fn default() -> Self {
        AvbdConfig {
            integrator: Integrator::Explicit,
            step: 1e-4,
            iterations: 20,
            tolerance: 0.0,
            damage: DamageMode::Implicit,
            contact: ContactMode::Penalty,
            alpha: 0.95,
            beta: 10.0,
            gamma: 0.99,
            hard_cap: 1e6,
            warm_previous: true,
            margin: 0.25,
            chebyshev: 0.0,
            stall: 40,
            predict_contacts: true,
            trace_residuals: false,
            end_time: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Chunk(usize, usize),
    Impactor(usize),
}

#[derive(Clone, Debug)]
struct Body {
    kind: Kind,
    mass: f64,
    /// Body-frame inertia.
    inertia: Mat3,
    fixed_lin: bool,
    fixed_ang: bool,
    x: Vec3,
    q: Quat,
    v: Vec3,
    w: Vec3,
    // Shape: box (or hull) orientation in the body frame, half extents.
    shape_rot: Mat3,
    half: Vec3,
    hull: Option<std::sync::Arc<crate::hull::ConvexHull>>,
    parts: Option<std::sync::Arc<Vec<crate::contact::Part>>>,
    sphere: Option<f64>,
    material: Material,
    // Step state.
    x0: Vec3,
    q0: Quat,
    v0: Vec3,
    w0: Vec3,
    a0: Vec3,
    alpha0: Vec3,
    dx: Vec3,
    dth: Vec3,
    /// External load this step (world, gravity excluded).
    ext_f: Vec3,
    ext_t: Vec3,
    /// Forces on the body at the last evaluation (for reactions), gravity included.
    force: Vec3,
    torque: Vec3,
}

impl Body {
    fn free(&self) -> bool {
        !(self.fixed_lin && self.fixed_ang)
    }
    fn world_inertia(&self, q: Quat) -> Mat3 {
        let r = q.to_mat3();
        r * self.inertia * r.transpose()
    }
    fn obox(&self) -> OBox {
        OBox {
            center: self.x,
            rotation: self.q.to_mat3() * self.shape_rot,
            half: self.half,
            hull: self.hull.clone(),
            parts: self.parts.clone(),
        }
    }
    fn radius(&self) -> f64 {
        self.half.norm()
    }
    fn point_velocity(&self, p: Vec3) -> Vec3 {
        self.v + self.w.cross(p - self.x)
    }
}

#[derive(Clone, Debug)]
struct Bond {
    s: usize,
    a: usize,
    b: usize,
    center_a: Vec3,
    center_b: Vec3,
    rt: RtBond,
    active: bool,
}

/// One contact point linearised at the step's start: body `a` is pushed along `n`, `b`
/// (if any) along `-n`; `depth0` is the (effective) overlap at the start.
#[derive(Clone, Debug)]
struct LinContact {
    a: usize,
    b: Option<usize>,
    ra: Vec3,
    rb: Vec3,
    n: Vec3,
    depth0: f64,
    k: f64,
    c: f64,
    c_max: f64,
    mu: f64,
    /// Hard contacts: multiplier and penalty.
    lambda: f64,
    kpen: f64,
    key: (usize, usize, usize),
}

/// Per-step and per-run counters.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct AvbdStats {
    pub steps: u64,
    pub sweeps: u64,
    pub bond_evaluations: u64,
    pub contact_evaluations: u64,
    pub block_solves: u64,
    pub max_colours: usize,
    pub colour_sum: u64,
    /// Sweep residual (relative) at each step's last sweep: worst and mean.
    pub worst_residual: f64,
    pub residual_sum: f64,
    /// Steps whose last sweep stayed above the tolerance.
    pub unconverged_steps: u64,
    /// Of those, steps stopped by the stall test, and the residual they stalled at (worst).
    pub stalled_steps: u64,
    pub worst_stall: f64,
    /// Sweeps of every step.
    pub sweep_counts: Vec<u32>,
    pub wall_seconds: f64,
    /// Per step: sweep residuals (with `trace_residuals`).
    #[serde(skip)]
    pub traces: Vec<Vec<f64>>,
}

/// Outcome of a run, comparable with an oracle observation.
#[derive(Clone, Debug, serde::Serialize)]
pub struct AvbdReport {
    pub time: f64,
    pub broken_bonds: usize,
    pub fragments: usize,
    pub detached_chunks: usize,
    /// Detached chunks moving faster than half the fastest impactor's initial speed.
    pub fast_detached: usize,
    pub first_crack_time: Option<f64>,
    pub max_utilization: f64,
    pub impactor_velocity: Vec<[f64; 3]>,
    pub bond_dissipation: f64,
    pub damping_dissipation: f64,
    pub stats: AvbdStats,
    pub probes: HashMap<String, Vec<[f64; 2]>>,
}

type M6 = [[f64; 6]; 6];

fn m6_add_jtkj(h: &mut M6, j: &M6, k: &Mat6) {
    // h += J^T K J
    let mut kj = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            let mut s = 0.0;
            for m in 0..6 {
                s += k.0[r][m] * j[m][c];
            }
            kj[r][c] = s;
        }
    }
    for r in 0..6 {
        for c in 0..6 {
            let mut s = 0.0;
            for m in 0..6 {
                s += j[m][r] * kj[m][c];
            }
            h[r][c] += s;
        }
    }
}

/// Solve `H x = b` for the free DOFs (`free[i]`), Cholesky with a diagonal shift if `H` is
/// not positive definite.
fn solve6(h: &M6, b: &[f64; 6], free: &[bool; 6]) -> [f64; 6] {
    let idx: Vec<usize> = (0..6).filter(|&i| free[i]).collect();
    let n = idx.len();
    let mut x = [0.0; 6];
    if n == 0 {
        return x;
    }
    let scale = idx
        .iter()
        .map(|&i| h[i][i].abs())
        .fold(0.0, f64::max)
        .max(1e-300);
    let mut shift = 0.0;
    loop {
        let mut l = [[0.0; 6]; 6];
        let mut ok = true;
        'outer: for i in 0..n {
            for j in 0..=i {
                let mut s = h[idx[i]][idx[j]] + if i == j { shift } else { 0.0 };
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    if s <= 0.0 {
                        ok = false;
                        break 'outer;
                    }
                    l[i][i] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        if ok {
            let mut y = [0.0; 6];
            for i in 0..n {
                let mut s = b[idx[i]];
                for k in 0..i {
                    s -= l[i][k] * y[k];
                }
                y[i] = s / l[i][i];
            }
            let mut z = [0.0; 6];
            for i in (0..n).rev() {
                let mut s = y[i];
                for k in i + 1..n {
                    s -= l[k][i] * z[k];
                }
                z[i] = s / l[i][i];
            }
            for i in 0..n {
                x[idx[i]] = z[i];
            }
            return x;
        }
        shift = if shift == 0.0 {
            1e-12 * scale
        } else {
            shift * 10.0
        };
    }
}

/// Jacobian rows of a bond's local deformation with respect to one body's
/// `(dx, dtheta)` (world), to first order: `sign = -1` for chunk `a`, `+1` for `b`.
fn bond_jacobian(e: &[Vec3; 3], r: Vec3, sign: f64) -> M6 {
    let mut j = [[0.0; 6]; 6];
    for (k, ek) in e.iter().enumerate() {
        let er = ek.cross(r);
        for d in 0..3 {
            j[k][d] = sign * ek[d];
            j[k][3 + d] = -sign * er[d];
            j[3 + k][3 + d] = sign * ek[d];
        }
    }
    j
}

/// Point-contact Jacobian `dp = dx + dtheta x r` as a 3x6 (rows padded to 6x6) and the
/// contribution `J^T K J` of a 3x3 point stiffness `k`.
fn add_point_hessian(h: &mut M6, r: Vec3, k: &Mat3) {
    // J = [I, -skew(r)].
    let s = Mat3::skew(r);
    let ks = *k * s; // K skew(r)
    let sk = s.transpose() * *k; // skew(r)^T K
    let sks = s.transpose() * ks;
    for a in 0..3 {
        for b in 0..3 {
            h[a][b] += k.m[a][b];
            h[a][3 + b] -= ks.m[a][b];
            h[3 + a][b] -= sk.m[a][b];
            h[3 + a][3 + b] += sks.m[a][b];
        }
    }
}

struct Detected {
    a: usize,
    b: Option<usize>,
    point: Vec3,
    n: Vec3,
    depth: f64,
    k: f64,
    m_red: f64,
    mu: f64,
    points: usize,
    key: (usize, usize, usize),
}

pub struct AvbdWorld {
    pub scene: Scene,
    pub config: AvbdConfig,
    poses: Vec<Pose>,
    bodies: Vec<Body>,
    bonds: Vec<Bond>,
    chunk_body: Vec<Vec<Option<usize>>>,
    impactor_bodies: Vec<usize>,
    body_bonds: Vec<Vec<usize>>,
    comp: Vec<usize>,
    comp_anchored: Vec<bool>,
    comp_count: usize,
    offsets: HashMap<(usize, usize), Vec<(f64, Vec3)>>,
    hard: HashMap<(usize, usize, usize), (f64, f64)>,
    /// Replacement loads: body, world force and moment at removal, start, duration.
    replacements: Vec<(usize, Vec3, Vec3, f64, f64)>,
    events_done: Vec<bool>,
    pub time: f64,
    /// The oracle's substep (the explicit truth's step and the contact law's dashpot cap).
    pub law_dt: f64,
    gravity: Vec3,
    initial_x: Vec<Vec3>,
    impactor_speed0: f64,
    first_crack: Option<f64>,
    max_utilization: f64,
    bond_dissipation: f64,
    damping_dissipation: f64,
    pub stats: AvbdStats,
    probes: HashMap<String, Vec<[f64; 2]>>,
    next_sample: f64,
    /// Contacts of the current implicit step.
    contacts: Vec<LinContact>,
    body_contacts: Vec<Vec<usize>>,
}

impl AvbdWorld {
    /// The oracle world's current state (after its prestress) as a chunk-level world.
    pub fn from_world(w: &World, config: AvbdConfig) -> AvbdWorld {
        let scene = w.scene.clone();
        let sv = &w.solver;
        let mut poses = Vec::new();
        let mut bodies = Vec::new();
        let mut chunk_body = Vec::new();
        for (s, st) in sv.structures.iter().enumerate() {
            let first = (0..st.chunks.len())
                .find(|&c| sv.chunks[s][c].active)
                .expect("an active chunk");
            let pose = sv.clusters[sv.chunks[s][first].cluster].pose;
            poses.push(pose);
            let mut map = vec![None; st.chunks.len()];
            for (c, ch) in st.chunks.iter().enumerate() {
                let cs = &sv.chunks[s][c];
                if !cs.active {
                    continue;
                }
                let cl = &sv.clusters[cs.cluster];
                assert!(
                    (cl.pose.position - pose.position).norm() < 1e-9,
                    "AVBD world: clusters of one structure start in one frame"
                );
                assert!(cs.inertia_scale == 1.0, "AVBD world: no mass scaling");
                let (v, wv) = sv.chunk_velocity(s, c);
                let q = (cl.pose.rotation * Quat::from_rotation_vector(cs.th)).normalized();
                map[c] = Some(bodies.len());
                bodies.push(Body {
                    kind: Kind::Chunk(s, c),
                    mass: ch.mass,
                    inertia: ch.inertia,
                    fixed_lin: ch.support != Support::None,
                    fixed_ang: ch.support == Support::Fixed,
                    x: sv.chunk_position(s, c),
                    q,
                    v,
                    w: wv,
                    shape_rot: ch.rotation,
                    half: ch.half_extents,
                    hull: ch.hull.clone(),
                    parts: ch.parts.clone(),
                    sphere: None,
                    material: scene.material(&ch.material).clone(),
                    x0: Vec3::ZERO,
                    q0: Quat::IDENTITY,
                    v0: Vec3::ZERO,
                    w0: Vec3::ZERO,
                    a0: Vec3::ZERO,
                    alpha0: Vec3::ZERO,
                    dx: Vec3::ZERO,
                    dth: Vec3::ZERO,
                    ext_f: Vec3::ZERO,
                    ext_t: Vec3::ZERO,
                    force: Vec3::ZERO,
                    torque: Vec3::ZERO,
                });
            }
            chunk_body.push(map);
        }
        let mut impactor_bodies = Vec::new();
        let mut impactor_speed0: f64 = 0.0;
        for (i, imp) in w.impactors.iter().enumerate() {
            assert!(
                imp.crush.is_none(),
                "AVBD world: impactor crush laws are not modelled"
            );
            let (half, sphere) = match imp.shape {
                ImpactorShape::Box { half_extents } => (Vec3::from_array(half_extents), None),
                ImpactorShape::Sphere { radius } => (Vec3::splat(radius), Some(radius)),
            };
            impactor_speed0 = impactor_speed0.max(imp.velocity.norm());
            impactor_bodies.push(bodies.len());
            bodies.push(Body {
                kind: Kind::Impactor(i),
                mass: imp.mass,
                inertia: imp.inertia,
                fixed_lin: false,
                fixed_ang: false,
                x: imp.pose.position,
                q: imp.pose.rotation,
                v: imp.velocity,
                w: imp.angular_velocity,
                shape_rot: Mat3::IDENTITY,
                half,
                hull: None,
                parts: None,
                sphere,
                material: imp.material.clone(),
                x0: Vec3::ZERO,
                q0: Quat::IDENTITY,
                v0: Vec3::ZERO,
                w0: Vec3::ZERO,
                a0: Vec3::ZERO,
                alpha0: Vec3::ZERO,
                dx: Vec3::ZERO,
                dth: Vec3::ZERO,
                ext_f: Vec3::ZERO,
                ext_t: Vec3::ZERO,
                force: Vec3::ZERO,
                torque: Vec3::ZERO,
            });
        }
        let mut bonds = Vec::new();
        for cl in &sv.clusters {
            let s = cl.structure;
            for &bi in &cl.bonds {
                let rt = sv.bonds[s][bi].clone();
                let (ca, cb) = (rt.geometry.a, rt.geometry.b);
                bonds.push(Bond {
                    s,
                    a: chunk_body[s][ca].expect("active"),
                    b: chunk_body[s][cb].expect("active"),
                    center_a: sv.structures[s].chunks[ca].center,
                    center_b: sv.structures[s].chunks[cb].center,
                    rt,
                    active: true,
                });
            }
        }
        let mut body_bonds = vec![Vec::new(); bodies.len()];
        for (k, b) in bonds.iter().enumerate() {
            body_bonds[b.a].push(k);
            body_bonds[b.b].push(k);
        }
        // Displacements are measured from the undeformed layout, as the oracle's.
        let initial_x = bodies
            .iter()
            .map(|b| match b.kind {
                Kind::Chunk(s, c) => poses[s].transform_point(sv.structures[s].chunks[c].center),
                Kind::Impactor(_) => b.x,
            })
            .collect();
        let n = bodies.len();
        let mut world = AvbdWorld {
            gravity: Vec3::from_array(scene.gravity),
            events_done: vec![false; scene.events.len()],
            scene,
            config,
            poses,
            bodies,
            bonds,
            chunk_body,
            impactor_bodies,
            body_bonds,
            comp: vec![0; n],
            comp_anchored: Vec::new(),
            comp_count: 0,
            offsets: HashMap::new(),
            hard: HashMap::new(),
            replacements: Vec::new(),
            time: sv.time,
            law_dt: w.substep_dt(),
            initial_x,
            impactor_speed0,
            first_crack: None,
            max_utilization: sv.max_utilization,
            bond_dissipation: 0.0,
            damping_dissipation: 0.0,
            stats: AvbdStats::default(),
            probes: HashMap::new(),
            next_sample: 0.0,
            contacts: Vec::new(),
            body_contacts: vec![Vec::new(); n],
        };
        world.rebuild_components();
        world
    }

    // ------------------------------------------------------------------ topology

    fn rebuild_components(&mut self) {
        let n = self.bodies.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for b in &self.bonds {
            if b.active && b.rt.connected() {
                let (ra, rb) = (find(&mut parent, b.a), find(&mut parent, b.b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
        let mut id = HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            let next = id.len();
            self.comp[i] = *id.entry(r).or_insert(next);
        }
        self.comp_count = id.len();
        self.comp_anchored = vec![false; self.comp_count];
        for (i, b) in self.bodies.iter().enumerate() {
            if b.fixed_lin {
                self.comp_anchored[self.comp[i]] = true;
            }
        }
        for b in self.bonds.iter_mut() {
            if b.active && self.comp[b.a] != self.comp[b.b] {
                b.active = false;
            }
        }
    }

    fn chunk_count_fragments(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for (i, b) in self.bodies.iter().enumerate() {
            if matches!(b.kind, Kind::Chunk(..)) && b.x.x < 1e8 {
                seen.insert(self.comp[i]);
            }
        }
        seen.len()
    }

    // ------------------------------------------------------------------ loads

    fn process_events(&mut self) {
        for ei in 0..self.scene.events.len() {
            if self.events_done[ei] {
                continue;
            }
            match self.scene.events[ei].clone() {
                EventDesc::RemoveChunks {
                    time,
                    duration,
                    body,
                    chunks,
                } => {
                    if self.time + 1e-12 < time {
                        continue;
                    }
                    let s = self
                        .scene
                        .bodies
                        .iter()
                        .position(|b| b.name == body)
                        .expect("body");
                    let sel = chunks.select(&self.scene.bodies[s]);
                    let removed: Vec<usize> =
                        sel.iter().filter_map(|&c| self.chunk_body[s][c]).collect();
                    for k in 0..self.bonds.len() {
                        let (a, b) = (self.bonds[k].a, self.bonds[k].b);
                        if !self.bonds[k].active {
                            continue;
                        }
                        let (ra, rb) = (removed.contains(&a), removed.contains(&b));
                        if !(ra || rb) {
                            continue;
                        }
                        if ra != rb {
                            // The bond's force on the kept chunk (current state, damping included).
                            let ev = self.bond_eval(k, 0.0, false, None);
                            let (f, m) = if ra {
                                (ev.loads[2], ev.loads[3])
                            } else {
                                (ev.loads[0], ev.loads[1])
                            };
                            let kept = if ra { b } else { a };
                            self.replacements.push((kept, f, m, self.time, duration));
                        }
                        self.bonds[k].active = false;
                    }
                    for &bi in &removed {
                        // Out of the simulation: fixed, far away, no contacts.
                        let body = &mut self.bodies[bi];
                        body.fixed_lin = true;
                        body.fixed_ang = true;
                        body.x = Vec3::splat(1e9) + Vec3::new(bi as f64 * 10.0, 0.0, 0.0);
                        body.v = Vec3::ZERO;
                        body.w = Vec3::ZERO;
                    }
                    for &(s2, c2) in &sel.iter().map(|&c| (s, c)).collect::<Vec<_>>() {
                        self.chunk_body[s2][c2] = None;
                    }
                    self.rebuild_components();
                }
                EventDesc::RemoveSupports { .. } => {
                    panic!("AVBD world: support removal is not modelled")
                }
            }
            self.events_done[ei] = true;
        }
    }

    /// External loads at time `t` (world), gravity excluded, into `ext_f`/`ext_t`.
    fn external_loads(&mut self, t: f64) {
        for b in self.bodies.iter_mut() {
            b.ext_f = Vec3::ZERO;
            b.ext_t = Vec3::ZERO;
        }
        for load in self.scene.loads.clone() {
            match load {
                LoadDesc::PointForce {
                    body,
                    chunk,
                    direction,
                    magnitude,
                    point,
                } => {
                    let s = self
                        .scene
                        .bodies
                        .iter()
                        .position(|b| b.name == body)
                        .expect("body");
                    let Some(bi) = self.chunk_body[s][chunk] else {
                        continue;
                    };
                    let f = Vec3::from_array(direction).normalized() * magnitude.eval(t);
                    let b = &mut self.bodies[bi];
                    b.ext_f += f;
                    if let Some(p) = point {
                        let center = self.scene.bodies[s].chunks[chunk].center;
                        let arm = b.q.rotate(Vec3::from_array(p) - Vec3::from_array(center));
                        b.ext_t += arm.cross(f);
                    }
                }
                _ => panic!("AVBD world: only point forces are modelled"),
            }
        }
        for &(bi, f, m, start, duration) in &self.replacements {
            let k = if duration <= 0.0 {
                0.0
            } else {
                (1.0 - (t - start) / duration).clamp(0.0, 1.0)
            };
            self.bodies[bi].ext_f += f * k;
            self.bodies[bi].ext_t += m * k;
        }
    }

    // ------------------------------------------------------------------ bonds

    fn corotation(&self, k: usize) -> CoRotation {
        let b = &self.bonds[k];
        let p = self.poses[b.s];
        let (ba, bb) = (&self.bodies[b.a], &self.bodies[b.b]);
        let inv = p.rotation.conjugate();
        let ua = p.inverse_transform_point(ba.x) - b.center_a;
        let ub = p.inverse_transform_point(bb.x) - b.center_b;
        let ta = ChunkTurn::new((inv * ba.q).to_rotation_vector());
        let tb = ChunkTurn::new((inv * bb.q).to_rotation_vector());
        b.rt.geometry.corotation_with(ua, &ta, ub, &tb)
    }

    /// Bond `k` at the bodies' current poses and velocities: the law evaluated from the
    /// committed history (over `dt`, `fracture`), its dashpot, chunk loads (world) and,
    /// with `hessian = Some(c_v)` (`d velocity / d displacement`), the Jacobians and the
    /// frozen-history tangent with the dashpot's share.
    fn bond_eval(&self, k: usize, dt: f64, fracture: bool, hessian: Option<f64>) -> BondEval {
        let b = &self.bonds[k];
        let p = self.poses[b.s];
        let inv = p.rotation.conjugate();
        let cr = self.corotation(k);
        let d = cr.deformation();
        let (ba, bb) = (&self.bodies[b.a], &self.bodies[b.b]);
        let rate = cr.variation(
            inv.rotate(ba.v),
            inv.rotate(ba.w),
            inv.rotate(bb.v),
            inv.rotate(bb.w),
        );
        let model = b.rt.model();
        let resp = model.evaluate(&b.rt.joint, &d, dt, fracture);
        let factors = model.secant_factors(&resp.state, &d);
        let c = b.rt.damping.mul_elem(&factors);
        let q_damp = rate.mul_elem(&c);
        let q = resp.force.add(&q_damp);
        let (fa, ma, fb, mb) = cr.loads(&q);
        let loads = [
            p.rotation.rotate(fa),
            p.rotation.rotate(ma),
            p.rotation.rotate(fb),
            p.rotation.rotate(mb),
        ];
        let damp_power = q_damp.dot(&rate);
        let jac = hessian.map(|c_v| {
            let g = &b.rt.geometry;
            let ra_rot = ba.q.to_mat3();
            let rb_rot = bb.q.to_mat3();
            // Local axes in the world ~ chunk a's turned frame (the mean frame to first order).
            let e = [ra_rot * g.t1, ra_rot * g.t2, ra_rot * g.normal];
            let ja = bond_jacobian(&e, ra_rot * g.ra, -1.0);
            let jb = bond_jacobian(&e, rb_rot * g.rb, 1.0);
            let kt = model.tangent(&resp.state, &d).add_diag(&c.scale(c_v));
            (ja, jb, kt)
        });
        BondEval {
            resp,
            loads,
            damp_power,
            jac,
        }
    }

    // ------------------------------------------------------------------ contacts

    /// Contacts at the current state within `margin(a, b)` of touching (0: the oracle's
    /// detection exactly), as the established sample-point penalty law sets them up.
    fn detect(&self, margin: &dyn Fn(usize, Option<usize>) -> f64) -> Vec<Detected> {
        let scale = self.scene.sim.stiffness_scale;
        let mut out = Vec::new();
        let n = self.bodies.len();
        let boxes: Vec<OBox> = self.bodies.iter().map(|b| b.obox()).collect();
        let inflate = |b: &OBox, m: f64| -> OBox {
            if m > 0.0 && b.hull.is_none() && b.parts.is_none() {
                OBox {
                    half: b.half + Vec3::splat(m),
                    ..b.clone()
                }
            } else {
                b.clone()
            }
        };
        let restitution = self.scene.sim.contact_restitution;
        let _ = restitution;
        // Impactors against chunks.
        for &ib in &self.impactor_bodies {
            let imp = &self.bodies[ib];
            let ibox = &boxes[ib];
            let reach = ibox.bounding_radius();
            let mut found: Vec<Detected> = Vec::new();
            for j in 0..n {
                let Kind::Chunk(..) = self.bodies[j].kind else {
                    continue;
                };
                let ch = &self.bodies[j];
                let m = margin(j, Some(ib));
                if (ch.x - imp.x).norm() > reach + ch.radius() + m {
                    continue;
                }
                let b = &boxes[j];
                let kc = contact_stiffness(
                    &imp.material,
                    ibox,
                    &ch.material,
                    b,
                    b.center - ibox.center,
                    scale,
                );
                let m_red = ch.mass * imp.mass / (ch.mass + imp.mass);
                let mu = self
                    .scene
                    .sim
                    .contact_friction
                    .unwrap_or(imp.material.friction.min(ch.material.friction));
                if let Some(radius) = imp.sphere {
                    if let Some(cp) = b.sphere_contact(imp.x, radius + m) {
                        found.push(Detected {
                            a: j,
                            b: Some(ib),
                            point: cp.point,
                            n: -cp.normal,
                            depth: cp.depth - m,
                            k: kc,
                            m_red,
                            mu,
                            points: 1,
                            key: (ib, j, 0),
                        });
                    }
                    continue;
                }
                let shrunk = inflate(ibox, m);
                let bb = inflate(b, m);
                let mut pts: Vec<(usize, Vec3, Vec3, f64)> = b
                    .points_inside(&shrunk)
                    .into_iter()
                    .map(|cp| (cp.index, cp.point, cp.normal, cp.depth - m))
                    .collect();
                let na = b.sample_count();
                pts.extend(
                    ibox.points_inside(&bb)
                        .into_iter()
                        .map(|cp| (na + cp.index, cp.point, -cp.normal, cp.depth - m)),
                );
                let np = pts.len();
                for (idx, p, nrm, depth) in pts {
                    found.push(Detected {
                        a: j,
                        b: Some(ib),
                        point: p,
                        n: nrm,
                        depth,
                        k: kc,
                        m_red,
                        mu,
                        points: np,
                        key: (ib, j, idx),
                    });
                }
            }
            out.extend(found);
        }
        // Ground.
        if let Some(g) = &self.scene.ground {
            let gm = self.scene.material(&g.material).clone();
            let mu = self.scene.sim.contact_friction.unwrap_or(g.friction);
            for j in 0..n {
                let body = &self.bodies[j];
                if !body.free() || body.fixed_lin {
                    continue;
                }
                if let Kind::Chunk(..) = body.kind {
                    if self.comp_anchored[self.comp[j]] {
                        continue;
                    }
                }
                let m = margin(j, None);
                if body.x.z - body.radius() - m > g.height {
                    continue;
                }
                let b = &boxes[j];
                let kc = contact_stiffness(&gm, b, &body.material, b, Vec3::Z, scale);
                let pts: Vec<(usize, Vec3)> = match (body.kind, body.sphere) {
                    (Kind::Impactor(_), Some(r)) => vec![(0, body.x - Vec3::Z * r)],
                    _ => b.sample_points().into_iter().enumerate().collect(),
                };
                let below: Vec<(usize, Vec3)> = pts
                    .into_iter()
                    .filter(|(_, p)| p.z < g.height + m)
                    .collect();
                let np = below.iter().filter(|(_, p)| p.z < g.height).count();
                let share = match body.kind {
                    Kind::Impactor(_) => kc / body.sample_count_for_ground() as f64,
                    Kind::Chunk(..) => kc / np.max(5) as f64,
                };
                for (idx, p) in below {
                    found_ground(
                        &mut out,
                        j,
                        p,
                        g.height - p.z,
                        share,
                        body.mass,
                        mu,
                        np,
                        idx,
                    );
                }
            }
        }
        // Chunks of different components (not both anchored).
        let chunk_ids: Vec<usize> = (0..n)
            .filter(|&j| matches!(self.bodies[j].kind, Kind::Chunk(..)) && self.bodies[j].x.x < 1e8)
            .collect();
        if self.comp_count > 1 {
            let cell = chunk_ids
                .iter()
                .map(|&j| self.bodies[j].radius())
                .fold(0.0, f64::max)
                * 2.0
                + 1e-6;
            let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
            let key = |p: Vec3| {
                (
                    (p.x / cell).floor() as i64,
                    (p.y / cell).floor() as i64,
                    (p.z / cell).floor() as i64,
                )
            };
            for &j in &chunk_ids {
                grid.entry(key(self.bodies[j].x)).or_default().push(j);
            }
            for &a in &chunk_ids {
                let (kx, ky, kz) = key(self.bodies[a].x);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let Some(list) = grid.get(&(kx + dx, ky + dy, kz + dz)) else {
                                continue;
                            };
                            for &b in list {
                                if b <= a || self.comp[a] == self.comp[b] {
                                    continue;
                                }
                                if self.comp_anchored[self.comp[a]]
                                    && self.comp_anchored[self.comp[b]]
                                {
                                    continue;
                                }
                                let m = margin(a, Some(b));
                                let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
                                if (ba.x - bb.x).norm() > ba.radius() + bb.radius() + m {
                                    continue;
                                }
                                let (oa, ob) = (inflate(&boxes[a], m), inflate(&boxes[b], m));
                                if !oa.may_overlap(&ob) {
                                    continue;
                                }
                                let na = boxes[a].sample_count();
                                let mut pts: Vec<(usize, Vec3, Vec3, f64)> = boxes[a]
                                    .points_inside(&ob)
                                    .into_iter()
                                    .map(|p| (p.index, p.point, p.normal, p.depth - m))
                                    .collect();
                                pts.extend(
                                    boxes[b]
                                        .points_inside(&oa)
                                        .into_iter()
                                        .map(|p| (na + p.index, p.point, -p.normal, p.depth - m)),
                                );
                                if pts.is_empty() {
                                    continue;
                                }
                                let k_pair = contact_stiffness(
                                    &ba.material,
                                    &boxes[a],
                                    &bb.material,
                                    &boxes[b],
                                    boxes[b].center - boxes[a].center,
                                    scale,
                                );
                                let m_red = ba.mass * bb.mass / (ba.mass + bb.mass);
                                let mu = self
                                    .scene
                                    .sim
                                    .contact_friction
                                    .unwrap_or(ba.material.friction.min(bb.material.friction));
                                let np = pts.len();
                                for (idx, p, nrm, depth) in pts {
                                    out.push(Detected {
                                        a,
                                        b: Some(b),
                                        point: p,
                                        n: nrm,
                                        depth,
                                        k: k_pair,
                                        m_red,
                                        mu,
                                        points: np,
                                        key: (a, b, idx),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Pre-existing-overlap memory of chunk pairs (as the oracle's `pair_contact`): a point
    /// born deeper than `reach_dt` of approach keeps that overlap as an offset that only
    /// ratchets down. Returns the effective depths, and rewrites the memory.
    fn pair_offsets(&mut self, found: &[Detected], reach_dt: f64) -> Vec<f64> {
        let mut eff = Vec::with_capacity(found.len());
        let mut next: HashMap<(usize, usize), Vec<(f64, Vec3)>> = HashMap::new();
        for d in found {
            let Some(b) = d.b else {
                eff.push(d.depth);
                continue;
            };
            if !matches!(self.bodies[b].kind, Kind::Chunk(..)) {
                eff.push(d.depth);
                continue;
            }
            let pair = (d.a, b);
            if d.depth <= 0.0 {
                eff.push(d.depth);
                continue;
            }
            let (ba, bb) = (&self.bodies[d.a], &self.bodies[b]);
            let entry = next
                .entry(pair)
                .or_insert_with(|| self.offsets.get(&pair).cloned().unwrap_or_default());
            let i = d.key.2;
            if entry.len() <= i {
                entry.resize(i + 1, (f64::NAN, Vec3::ZERO));
            }
            if entry[i].0.is_nan() || entry[i].1.dot(d.n) < 0.99 {
                let rel = ba.point_velocity(d.point) - bb.point_velocity(d.point);
                let reach = 2.0 * rel.dot(d.n).abs() * reach_dt + 1e-9;
                entry[i] = (if d.depth > reach { d.depth } else { 0.0 }, d.n);
            }
            entry[i].0 = entry[i].0.min(d.depth);
            eff.push(d.depth - entry[i].0);
        }
        // Points not inside this time forget their offsets.
        let mut inside: HashMap<(usize, usize), std::collections::HashSet<usize>> = HashMap::new();
        for d in found {
            if let Some(b) = d.b {
                if d.depth > 0.0 {
                    inside.entry((d.a, b)).or_default().insert(d.key.2);
                }
            }
        }
        for (pair, list) in next.iter_mut() {
            let set = inside.get(pair);
            for (i, e) in list.iter_mut().enumerate() {
                if !set.is_some_and(|s| s.contains(&i)) {
                    e.0 = f64::NAN;
                }
            }
        }
        self.offsets = next;
        eff
    }

    // ------------------------------------------------------------------ explicit

    /// One explicit substep of `dt` (the oracle's integrator and order: events, loads,
    /// contacts and bonds at the substep's start, symplectic update, splits).
    pub fn step_explicit(&mut self, dt: f64) {
        self.process_events();
        self.external_loads(self.time);
        let g = self.gravity;
        for b in self.bodies.iter_mut() {
            b.force = b.ext_f + g * b.mass;
            b.torque = b.ext_t;
        }
        // Contacts.
        let found = self.detect(&|_, _| 0.0);
        let eff = self.pair_offsets(&found, dt);
        let restitution = self.scene.sim.contact_restitution;
        // Pair stiffness shares count the engaged (effective) points.
        let mut engaged: HashMap<(usize, usize), usize> = HashMap::new();
        for (d, &e) in found.iter().zip(&eff) {
            if let Some(b) = d.b {
                if matches!(self.bodies[b].kind, Kind::Chunk(..)) && e > 0.0 {
                    *engaged.entry((d.a, b)).or_default() += 1;
                }
            }
        }
        for (d, &depth) in found.iter().zip(&eff) {
            if depth <= 0.0 {
                continue;
            }
            let pair_chunks =
                d.b.is_some_and(|b| matches!(self.bodies[b].kind, Kind::Chunk(..)));
            let (k, points) = if pair_chunks {
                let e = engaged[&(d.a, d.b.unwrap())];
                (d.k / e.max(FACE_PAIR_POINTS) as f64, e)
            } else if d.b.is_some() && self.bodies[d.b.unwrap()].sphere.is_none() {
                (d.k / d.points.max(FACE_PAIR_POINTS) as f64, d.points)
            } else {
                (d.k, d.points)
            };
            let va = self.bodies[d.a].point_velocity(d.point);
            let vb =
                d.b.map_or(Vec3::ZERO, |b| self.bodies[b].point_velocity(d.point));
            let (f, _, _) = penalty_force(
                k,
                d.m_red,
                restitution,
                d.mu,
                depth,
                d.n,
                va - vb,
                dt,
                points,
            );
            self.stats.contact_evaluations += 1;
            let xa = self.bodies[d.a].x;
            self.bodies[d.a].force += f;
            self.bodies[d.a].torque += (d.point - xa).cross(f);
            if let Some(b) = d.b {
                let xb = self.bodies[b].x;
                self.bodies[b].force -= f;
                self.bodies[b].torque -= (d.point - xb).cross(f);
            }
        }
        // Bonds.
        let mut split = false;
        for k in 0..self.bonds.len() {
            if !self.bonds[k].active {
                continue;
            }
            let ev = self.bond_eval(k, dt, self.scene.sim.fracture, None);
            self.stats.bond_evaluations += 1;
            split |= self.commit_bond(k, &ev, dt);
            let (a, b) = (self.bonds[k].a, self.bonds[k].b);
            self.bodies[a].force += ev.loads[0];
            self.bodies[a].torque += ev.loads[1];
            self.bodies[b].force += ev.loads[2];
            self.bodies[b].torque += ev.loads[3];
        }
        // Symplectic update.
        for b in self.bodies.iter_mut() {
            if !b.fixed_lin {
                b.v += b.force * (dt / b.mass);
                b.x += b.v * dt;
            }
            if !b.fixed_ang {
                let iw = b.world_inertia(b.q);
                let l = iw * b.w + b.torque * dt;
                let w_mid = iw.inverse().expect("inertia") * l;
                b.q = b.q.integrate(w_mid, dt);
                b.w = b.world_inertia(b.q).inverse().expect("inertia") * l;
            }
        }
        if split {
            self.rebuild_components();
        }
        self.time += dt;
        self.stats.steps += 1;
    }

    /// Commit bond `k`'s response (history, events, ledger). True if it disconnected.
    fn commit_bond(&mut self, k: usize, ev: &BondEval, dt: f64) -> bool {
        let t = self.time + dt;
        let b = &mut self.bonds[k];
        let prev = std::mem::replace(&mut b.rt.joint, ev.resp.state.clone());
        if prev.damage == 0.0
            && prev.crush == 0.0
            && ev.resp.state.is_damaged()
            && self.first_crack.is_none()
        {
            self.first_crack = Some(t);
        }
        self.max_utilization = self.max_utilization.max(ev.resp.state.utilization);
        self.bond_dissipation += ev.resp.dissipated;
        self.damping_dissipation += ev.damp_power * dt;
        ev.resp.disconnected
    }

    // ------------------------------------------------------------------ AVBD

    fn coefficients(&self, h: f64) -> (f64, f64) {
        // (inertia weight c_m, d velocity / d displacement c_v)
        match self.config.integrator {
            Integrator::BackwardEuler => (1.0 / (h * h), 1.0 / h),
            Integrator::Newmark => (4.0 / (h * h), 2.0 / h),
            Integrator::Explicit => unreachable!(),
        }
    }

    /// Set a body's trial pose and velocity from its increment.
    fn set_trial(&mut self, i: usize, h: f64) {
        let nm = self.config.integrator == Integrator::Newmark;
        let b = &mut self.bodies[i];
        b.x = b.x0 + b.dx;
        b.q = (Quat::from_rotation_vector(b.dth) * b.q0).normalized();
        if nm {
            b.v = b.dx * (2.0 / h) - b.v0;
            b.w = b.dth * (2.0 / h) - b.w0;
        } else {
            b.v = b.dx / h;
            b.w = b.dth / h;
        }
    }

    /// Linearised contacts for the step from `detect` with the speculative margin.
    fn build_contacts(&mut self, h: f64) {
        let speed = |b: &Body| b.v.norm() + b.w.norm() * b.radius();
        let speeds: Vec<f64> = self
            .bodies
            .iter()
            .map(|b| if b.free() { speed(b) } else { 0.0 })
            .collect();
        let factor = self.config.margin;
        let margin = |a: usize, b: Option<usize>| {
            if factor > 0.0 {
                factor * h * (speeds[a] + b.map_or(0.0, |b| speeds[b])) + 1e-7
            } else {
                0.0
            }
        };
        // Contacts are found at the pose the start velocities predict for the step's end
        // (where backward Euler and Newmark evaluate the forces), so each point takes the
        // face nearest its overlap there, as the law does; the margin only catches points
        // the prediction just misses.
        let saved: Vec<(Vec3, Quat)> = self.bodies.iter().map(|b| (b.x, b.q)).collect();
        if self.config.predict_contacts {
            for b in self.bodies.iter_mut() {
                if !b.fixed_lin {
                    b.x += b.v * h;
                }
                if !b.fixed_ang {
                    b.q = b.q.integrate(b.w, h);
                }
            }
        }
        let mut found = self.detect(&margin);
        for (b, (x, q)) in self.bodies.iter_mut().zip(saved) {
            b.x = x;
            b.q = q;
        }
        if self.config.predict_contacts {
            // Back to the start: depth at the start along the predicted normal.
            for d in found.iter_mut() {
                let va = self.bodies[d.a].v;
                let vb = d.b.map_or(Vec3::ZERO, |b| {
                    if self.bodies[b].fixed_lin {
                        Vec3::ZERO
                    } else {
                        self.bodies[b].v
                    }
                });
                let wa = self.bodies[d.a].w;
                let wb = d.b.map_or(Vec3::ZERO, |b| self.bodies[b].w);
                let pa = d.point - va * h;
                let pb = d.point - vb * h;
                let rel = (va + wa.cross(pa - self.bodies[d.a].x))
                    - d.b
                        .map_or(Vec3::ZERO, |b| vb + wb.cross(pb - self.bodies[b].x));
                d.depth += rel.dot(d.n) * h;
                // The point on `a` at the start (lever arms are taken from it below).
                d.point = pa;
            }
        }
        let eff = self.pair_offsets(&found, h);
        let restitution = self.scene.sim.contact_restitution;
        let zeta = damping_ratio(restitution);
        // Points engaged by the step's end, predicted from the start velocities.
        let mut engaged: HashMap<(usize, usize), usize> = HashMap::new();
        let mut predicted = Vec::with_capacity(found.len());
        for (d, &e) in found.iter().zip(&eff) {
            let va = self.bodies[d.a].point_velocity(d.point);
            let vb =
                d.b.map_or(Vec3::ZERO, |b| self.bodies[b].point_velocity(d.point));
            let g_end = e - (va - vb).dot(d.n) * h;
            let on = e > 0.0 || g_end > 0.0;
            predicted.push(on);
            if on {
                *engaged.entry((d.a, d.b.unwrap_or(usize::MAX))).or_default() += 1;
            }
        }
        let mut contacts = Vec::with_capacity(found.len());
        for ((d, &depth), &on) in found.iter().zip(&eff).zip(&predicted) {
            let pair_key = (d.a, d.b.unwrap_or(usize::MAX));
            let e = engaged
                .get(&pair_key)
                .copied()
                .unwrap_or(0)
                .max(usize::from(on));
            let chunk_pair =
                d.b.is_some_and(|b| matches!(self.bodies[b].kind, Kind::Chunk(..)));
            let sphere = d.b.is_some_and(|b| self.bodies[b].sphere.is_some());
            let (k, points) = if d.b.is_none() {
                (d.k, d.points)
            } else if chunk_pair || !sphere {
                (d.k / e.max(FACE_PAIR_POINTS) as f64, e)
            } else {
                (d.k, 1)
            };
            let c_max =
                MAX_SET_VISCOSITY / points.max(FACE_PAIR_POINTS) as f64 * d.m_red / self.law_dt;
            let c = (2.0 * zeta * (k * d.m_red).sqrt()).min(c_max);
            let (lambda, kpen) = match self.config.contact {
                ContactMode::Penalty => (0.0, k),
                ContactMode::Hard => {
                    let (l, kp) = self
                        .hard
                        .get(&d.key)
                        .copied()
                        .unwrap_or((0.0, d.m_red / (h * h)));
                    (
                        l * self.config.alpha * self.config.gamma,
                        (kp * self.config.gamma).max(d.m_red / (h * h)),
                    )
                }
            };
            let ra = d.point - self.bodies[d.a].x;
            let rb = d.b.map_or(Vec3::ZERO, |b| d.point - self.bodies[b].x);
            contacts.push(LinContact {
                a: d.a,
                b: d.b,
                ra,
                rb,
                n: d.n,
                depth0: depth,
                k,
                c,
                c_max,
                mu: d.mu,
                lambda,
                kpen,
                key: d.key,
            });
        }
        self.body_contacts.iter_mut().for_each(|l| l.clear());
        for (ci, c) in contacts.iter().enumerate() {
            self.body_contacts[c.a].push(ci);
            if let Some(b) = c.b {
                self.body_contacts[b].push(ci);
            }
        }
        self.contacts = contacts;
    }

    /// Contact `ci` at the trial state: normal force, the force on `a` and the 3x3
    /// point stiffness (relative displacement of `a`'s point).
    fn contact_eval(&self, ci: usize, c_v: f64) -> (f64, Vec3, Mat3, f64) {
        let c = &self.contacts[ci];
        let ba = &self.bodies[c.a];
        let dpa = ba.dx + ba.dth.cross(c.ra);
        let va = ba.v + ba.w.cross(c.ra);
        let (dpb, vb) = match c.b {
            Some(b) => {
                let bb = &self.bodies[b];
                (bb.dx + bb.dth.cross(c.rb), bb.v + bb.w.cross(c.rb))
            }
            None => (Vec3::ZERO, Vec3::ZERO),
        };
        let gap = c.depth0 - c.n.dot(dpa - dpb);
        let rel = va - vb;
        let vn = rel.dot(c.n);
        let hard = self.config.contact == ContactMode::Hard;
        let (fn_mag, kn) = if hard {
            let g_stab = gap - self.config.alpha * c.depth0.max(0.0);
            let f = c.kpen * g_stab + c.lambda;
            if f > 0.0 {
                (f, c.kpen)
            } else {
                (0.0, 0.0)
            }
        } else {
            // The law acts only on overlap (a speculative point is no contact yet).
            let f = if gap > 0.0 { c.k * gap - c.c * vn } else { 0.0 };
            if f > 0.0 {
                (f, c.k + c.c * c_v)
            } else {
                (0.0, 0.0)
            }
        };
        let vt = rel - c.n * vn;
        let vt_mag = vt.norm();
        let cfr = c.c_max.min(c.mu * fn_mag / FRICTION_REGULARIZATION);
        let ft_mag = (c.mu * fn_mag).min(cfr * vt_mag);
        let ft = if vt_mag > 0.0 {
            -vt * (ft_mag / vt_mag)
        } else {
            Vec3::ZERO
        };
        let nn = Mat3::outer(c.n, c.n);
        let mut k3 = nn * kn;
        if fn_mag > 0.0 && cfr * vt_mag < c.mu * fn_mag {
            k3 += (Mat3::IDENTITY - nn) * (cfr * c_v);
        }
        (fn_mag, c.n * fn_mag + ft, k3, gap)
    }

    /// One block Newton step of body `i` at the current trial state: its increment, the
    /// squared residual before it, the squared force scale and the bonds and contacts
    /// it evaluated.
    fn block_solve(&self, i: usize, h: f64) -> ([f64; 6], f64, f64, u64, u64) {
        let (c_m, c_v) = self.coefficients(h);
        let dt_law = h;
        let fracture = self.config.damage == DamageMode::Implicit && self.scene.sim.fracture;
        let nm = self.config.integrator == Integrator::Newmark;
        let b = &self.bodies[i];
        let iw = b.world_inertia(b.q0);
        let target_x = b.v0 * h
            + if nm {
                b.a0 * (0.25 * h * h)
            } else {
                Vec3::ZERO
            };
        let target_th = b.w0 * h
            + if nm {
                b.alpha0 * (0.25 * h * h)
            } else {
                Vec3::ZERO
            };
        let inert_f = (b.dx - target_x) * (b.mass * c_m);
        let inert_t = iw * (b.dth - target_th) * c_m;
        let mut f = b.ext_f + self.gravity * b.mass;
        let mut t = b.ext_t;
        let mut hm: M6 = [[0.0; 6]; 6];
        for d in 0..3 {
            hm[d][d] = b.mass * c_m;
            for e in 0..3 {
                hm[3 + d][3 + e] = iw.m[d][e] * c_m;
            }
        }
        let mut scale2 = 0.0;
        for &k in &self.body_bonds[i] {
            if !self.bonds[k].active {
                continue;
            }
            let ev = self.bond_eval(k, dt_law, fracture, Some(c_v));
            let (ja, jb, kt) = ev.jac.as_ref().expect("hessian");
            if self.bonds[k].a == i {
                f += ev.loads[0];
                t += ev.loads[1];
                m6_add_jtkj(&mut hm, ja, kt);
                scale2 += ev.loads[0].norm2();
            } else {
                f += ev.loads[2];
                t += ev.loads[3];
                m6_add_jtkj(&mut hm, jb, kt);
                scale2 += ev.loads[2].norm2();
            }
        }
        let n_bonds = self.body_bonds[i].len() as u64;
        for &ci in &self.body_contacts[i] {
            let (_, fa, k3, _) = self.contact_eval(ci, c_v);
            let c = &self.contacts[ci];
            let (r, sign) = if c.a == i { (c.ra, 1.0) } else { (c.rb, -1.0) };
            let fc = fa * sign;
            f += fc;
            t += r.cross(fc);
            scale2 += fc.norm2();
            add_point_hessian(&mut hm, r, &k3);
        }
        let n_contacts = self.body_contacts[i].len() as u64;
        let b = &self.bodies[i];
        let gvec = [
            inert_f.x - f.x,
            inert_f.y - f.y,
            inert_f.z - f.z,
            inert_t.x - t.x,
            inert_t.y - t.y,
            inert_t.z - t.z,
        ];
        let free = [
            !b.fixed_lin,
            !b.fixed_lin,
            !b.fixed_lin,
            !b.fixed_ang,
            !b.fixed_ang,
            !b.fixed_ang,
        ];
        let r_char = b.radius().max(1e-6);
        let mut res2 = 0.0;
        for d in 0..6 {
            if free[d] {
                let s = if d < 3 { 1.0 } else { 1.0 / r_char };
                res2 += (gvec[d] * s).powi(2);
            }
        }
        scale2 += (self.gravity * b.mass + b.ext_f).norm2();
        let rhs = gvec.map(|x| -x);
        (solve6(&hm, &rhs, &free), res2, scale2, n_bonds, n_contacts)
    }

    fn colour(&self) -> Vec<Vec<usize>> {
        let n = self.bodies.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in &self.bonds {
            if b.active {
                adj[b.a].push(b.b);
                adj[b.b].push(b.a);
            }
        }
        for c in &self.contacts {
            if let Some(b) = c.b {
                adj[c.a].push(b);
                adj[b].push(c.a);
            }
        }
        let mut colour = vec![usize::MAX; n];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for i in 0..n {
            if !self.bodies[i].free() || self.bodies[i].x.x > 1e8 {
                continue;
            }
            let used: std::collections::HashSet<usize> =
                adj[i].iter().map(|&j| colour[j]).collect();
            let c = (0..).find(|c| !used.contains(c)).unwrap();
            colour[i] = c;
            if groups.len() <= c {
                groups.push(Vec::new());
            }
            groups[c].push(i);
        }
        groups
    }

    /// One implicit step of `h` by block descent.
    pub fn step_implicit(&mut self, h: f64) {
        self.process_events();
        self.external_loads(self.time + h);
        let nm = self.config.integrator == Integrator::Newmark;
        let warm_previous = self.config.warm_previous;
        let g = self.gravity;
        // Step start and warm start.
        for b in self.bodies.iter_mut() {
            let (acc, alpha) = if nm || warm_previous {
                (b.a0, b.alpha0)
            } else {
                (g + b.ext_f / b.mass, Vec3::ZERO)
            };
            b.x0 = b.x;
            b.q0 = b.q;
            b.v0 = b.v;
            b.w0 = b.w;
            if b.fixed_lin {
                b.dx = Vec3::ZERO;
            } else {
                b.dx = b.v0 * h + acc * (if nm { 0.5 } else { 1.0 } * h * h);
            }
            if b.fixed_ang {
                b.dth = Vec3::ZERO;
            } else {
                b.dth = b.w0 * h + alpha * (if nm { 0.5 } else { 1.0 } * h * h);
            }
        }
        // Contacts at the start pose (velocities at the start set the margins).
        self.build_contacts(h);
        for i in 0..self.bodies.len() {
            self.set_trial(i, h);
        }
        let groups = self.colour();
        self.stats.max_colours = self.stats.max_colours.max(groups.len());
        self.stats.colour_sum += groups.len() as u64;
        let mut trace = Vec::new();
        let mut last = 0.0;
        let rho = self.config.chebyshev;
        let mut omega = 1.0;
        let mut older: Vec<(Vec3, Vec3)> = self.bodies.iter().map(|b| (b.dx, b.dth)).collect();
        let mut best = (f64::INFINITY, 0usize);
        let mut stalled = false;
        let sweeps_before = self.stats.sweeps;
        for sweep in 0..self.config.iterations {
            let before: Vec<(Vec3, Vec3)> = if rho > 0.0 {
                self.bodies.iter().map(|b| (b.dx, b.dth)).collect()
            } else {
                Vec::new()
            };
            let (mut res2, mut scale2) = (0.0, 0.0);
            for g in &groups {
                // Bodies of one colour share no bond or contact: their block steps are
                // independent (computed in parallel, applied in order).
                let out = crate::par::map_range(0..g.len(), 8, |k| self.block_solve(g[k], h));
                for (&i, (delta, r, s, nb, nc)) in g.iter().zip(out) {
                    let b = &mut self.bodies[i];
                    b.dx += Vec3::new(delta[0], delta[1], delta[2]);
                    b.dth += Vec3::new(delta[3], delta[4], delta[5]);
                    self.set_trial(i, h);
                    self.stats.block_solves += 1;
                    self.stats.bond_evaluations += nb;
                    self.stats.contact_evaluations += nc;
                    res2 += r;
                    scale2 += s;
                }
            }
            if rho > 0.0 {
                // omega_1 = 1, omega_2 = 2 / (2 - rho^2), omega_k = 4 / (4 - rho^2 omega_{k-1}).
                omega = match sweep {
                    0 => 1.0,
                    1 => 2.0 / (2.0 - rho * rho),
                    _ => 4.0 / (4.0 - rho * rho * omega),
                };
                if sweep > 0 {
                    for i in 0..self.bodies.len() {
                        if !self.bodies[i].free() {
                            continue;
                        }
                        let (ox, ot) = older[i];
                        let b = &mut self.bodies[i];
                        b.dx = ox + (b.dx - ox) * omega;
                        b.dth = ot + (b.dth - ot) * omega;
                        self.set_trial(i, h);
                    }
                }
                older = before;
            }
            if self.config.contact == ContactMode::Hard {
                self.dual_update(h);
            }
            self.stats.sweeps += 1;
            last = (res2 / scale2.max(1e-300)).sqrt();
            if self.config.trace_residuals {
                trace.push(last);
            }
            if self.config.tolerance > 0.0 && last <= self.config.tolerance {
                break;
            }
            if self.config.tolerance > 0.0 && self.config.stall > 0 {
                if last < 0.9 * best.0 {
                    best = (last, sweep);
                } else if sweep - best.1 >= self.config.stall {
                    stalled = true;
                    break;
                }
            }
        }
        self.stats
            .sweep_counts
            .push((self.stats.sweeps - sweeps_before) as u32);
        if stalled {
            self.stats.stalled_steps += 1;
            self.stats.worst_stall = self.stats.worst_stall.max(last);
        }
        if self.config.tolerance > 0.0 && last > self.config.tolerance {
            self.stats.unconverged_steps += 1;
        }
        self.stats.worst_residual = self.stats.worst_residual.max(last);
        self.stats.residual_sum += last;
        if self.config.trace_residuals {
            self.stats.traces.push(trace);
        }
        // Accept: accelerations for the next step, history committed once.
        for b in self.bodies.iter_mut() {
            if nm {
                let a = (b.dx - b.v0 * h) * (4.0 / (h * h)) - b.a0;
                let al = (b.dth - b.w0 * h) * (4.0 / (h * h)) - b.alpha0;
                b.a0 = if b.fixed_lin { Vec3::ZERO } else { a };
                b.alpha0 = if b.fixed_ang { Vec3::ZERO } else { al };
            } else {
                b.a0 = if b.fixed_lin {
                    Vec3::ZERO
                } else {
                    (b.v - b.v0) / h
                };
                b.alpha0 = if b.fixed_ang {
                    Vec3::ZERO
                } else {
                    (b.w - b.w0) / h
                };
            }
        }
        if self.config.contact == ContactMode::Hard {
            for c in &self.contacts {
                self.hard.insert(c.key, (c.lambda, c.kpen));
            }
        }
        let g = self.gravity;
        for b in self.bodies.iter_mut() {
            b.force = b.ext_f + g * b.mass;
            b.torque = b.ext_t;
        }
        let mut split = false;
        for k in 0..self.bonds.len() {
            if !self.bonds[k].active {
                continue;
            }
            let ev = self.bond_eval(k, h, self.scene.sim.fracture, None);
            split |= self.commit_bond(k, &ev, h);
            let (a, b) = (self.bonds[k].a, self.bonds[k].b);
            self.bodies[a].force += ev.loads[0];
            self.bodies[a].torque += ev.loads[1];
            self.bodies[b].force += ev.loads[2];
            self.bodies[b].torque += ev.loads[3];
        }
        let (_, c_v) = self.coefficients(h);
        for ci in 0..self.contacts.len() {
            let (_, fa, _, _) = self.contact_eval(ci, c_v);
            let c = &self.contacts[ci];
            let (a, b, ra, rb) = (c.a, c.b, c.ra, c.rb);
            self.bodies[a].force += fa;
            self.bodies[a].torque += ra.cross(fa);
            if let Some(b) = b {
                self.bodies[b].force -= fa;
                self.bodies[b].torque -= rb.cross(fa);
            }
        }
        if std::env::var("AVBD_TRACE").is_ok() {
            let active = (0..self.contacts.len())
                .filter(|&ci| self.contact_eval(ci, c_v).0 > 0.0)
                .count();
            let ram = self
                .impactor_bodies
                .first()
                .map(|&i| self.bodies[i].force)
                .unwrap_or(Vec3::ZERO);
            let broken = self
                .bonds
                .iter()
                .filter(|b| !b.rt.joint.connected(b.rt.rebar.is_some()))
                .count();
            eprintln!(
                "t {:.4e} contacts {} active {} ram force {:.3e} {:.3e} {:.3e} broken {} comps {}",
                self.time + h,
                self.contacts.len(),
                active,
                ram.x,
                ram.y,
                ram.z,
                broken,
                self.comp_count
            );
        }
        if split {
            self.rebuild_components();
        }
        self.time += h;
        self.stats.steps += 1;
    }

    fn dual_update(&mut self, h: f64) {
        let (_, c_v) = self.coefficients(h);
        for ci in 0..self.contacts.len() {
            let (fn_mag, _, _, gap) = self.contact_eval(ci, c_v);
            let alpha = self.config.alpha;
            let beta = self.config.beta;
            let cap = self.config.hard_cap;
            let c = &mut self.contacts[ci];
            let g_stab = gap - alpha * c.depth0.max(0.0);
            c.lambda = fn_mag.max(0.0);
            if fn_mag > 0.0 {
                // Penalty growth per unit violation, scaled by the law's stiffness per metre.
                c.kpen = (c.kpen + beta * c.k * g_stab.abs() / 1e-3).min(cap * c.k);
            }
        }
    }

    // ------------------------------------------------------------------ run

    fn record_probes(&mut self) {
        if self.time + 1e-12 < self.next_sample {
            return;
        }
        let interval = self
            .scene
            .sim
            .sample_interval
            .unwrap_or(self.scene.sim.frame_dt);
        self.next_sample += interval;
        for p in self.scene.probes.clone() {
            let value = match &p.kind {
                ProbeKind::ImpactorVelocity { impactor, axis } => {
                    let i = self
                        .scene
                        .impactors
                        .iter()
                        .position(|d| &d.name == impactor)
                        .expect("impactor");
                    self.bodies[self.impactor_bodies[i]]
                        .v
                        .dot(Vec3::from_array(*axis))
                }
                ProbeKind::ChunkDisplacement { body, chunk, axis } => {
                    let s = self
                        .scene
                        .bodies
                        .iter()
                        .position(|b| &b.name == body)
                        .expect("body");
                    match self.chunk_body[s][*chunk] {
                        Some(bi) => {
                            (self.bodies[bi].x - self.initial_x[bi]).dot(Vec3::from_array(*axis))
                        }
                        None => continue,
                    }
                }
                ProbeKind::ChunkVelocity { body, chunk, axis } => {
                    let s = self
                        .scene
                        .bodies
                        .iter()
                        .position(|b| &b.name == body)
                        .expect("body");
                    match self.chunk_body[s][*chunk] {
                        Some(bi) => self.bodies[bi].v.dot(Vec3::from_array(*axis)),
                        None => continue,
                    }
                }
                ProbeKind::Reaction { body, chunks, axis } => {
                    let s = self
                        .scene
                        .bodies
                        .iter()
                        .position(|b| &b.name == body)
                        .expect("body");
                    let sel = chunks.select(&self.scene.bodies[s]);
                    let mut sum = 0.0;
                    for c in sel {
                        if let Some(bi) = self.chunk_body[s][c] {
                            let b = &self.bodies[bi];
                            if b.fixed_lin {
                                sum -= b.force.dot(Vec3::from_array(*axis));
                            }
                        }
                    }
                    sum
                }
                _ => continue,
            };
            self.probes
                .entry(p.name.clone())
                .or_default()
                .push([self.time, value]);
        }
    }

    /// Accelerations at the start state (Newmark's `a_n`, the warm start): bonds,
    /// external loads and gravity.
    fn init_accelerations(&mut self) {
        self.external_loads(self.time);
        let g = self.gravity;
        let mut f: Vec<(Vec3, Vec3)> = self
            .bodies
            .iter()
            .map(|b| (b.ext_f + g * b.mass, b.ext_t))
            .collect();
        for k in 0..self.bonds.len() {
            if self.bonds[k].active {
                let ev = self.bond_eval(k, 0.0, false, None);
                let (a, b) = (self.bonds[k].a, self.bonds[k].b);
                f[a].0 += ev.loads[0];
                f[a].1 += ev.loads[1];
                f[b].0 += ev.loads[2];
                f[b].1 += ev.loads[3];
            }
        }
        for (b, (fi, ti)) in self.bodies.iter_mut().zip(f) {
            b.a0 = if b.fixed_lin { Vec3::ZERO } else { fi / b.mass };
            b.alpha0 = if b.fixed_ang {
                Vec3::ZERO
            } else {
                b.world_inertia(b.q).inverse().expect("inertia") * ti
            };
        }
    }

    /// Run to the scene's end (or `config.end_time`).
    pub fn run(&mut self) -> AvbdReport {
        let end = self.config.end_time.unwrap_or(self.scene.sim.duration);
        let start = Instant::now();
        self.next_sample = self.time;
        // Reactions before the first step: the prestressed bond forces.
        if self
            .scene
            .probes
            .iter()
            .any(|p| matches!(p.kind, ProbeKind::Reaction { .. }))
        {
            let g = self.gravity;
            for b in self.bodies.iter_mut() {
                b.force = g * b.mass;
            }
            for k in 0..self.bonds.len() {
                if self.bonds[k].active {
                    let ev = self.bond_eval(k, 0.0, false, None);
                    let (a, b) = (self.bonds[k].a, self.bonds[k].b);
                    self.bodies[a].force += ev.loads[0];
                    self.bodies[b].force += ev.loads[2];
                }
            }
        }
        if self.config.integrator != Integrator::Explicit {
            self.init_accelerations();
        }
        self.record_probes();
        while self.time < end - 1e-12 {
            match self.config.integrator {
                Integrator::Explicit => {
                    let dt = self.law_dt;
                    self.step_explicit(dt);
                }
                _ => {
                    let h = self.config.step.min(end - self.time);
                    self.step_implicit(h);
                }
            }
            self.record_probes();
            if self
                .bodies
                .iter()
                .any(|b| !b.x.is_finite() || !b.v.is_finite())
            {
                eprintln!("AVBD world: state diverged at t = {:.6e}", self.time);
                break;
            }
        }
        self.stats.wall_seconds = start.elapsed().as_secs_f64();
        self.report()
    }

    pub fn report(&self) -> AvbdReport {
        let broken = self
            .bonds
            .iter()
            .filter(|b| !b.rt.joint.connected(b.rt.rebar.is_some()))
            .count();
        // Main piece: anchored components, else the heaviest per structure.
        let mut comp_mass: HashMap<usize, f64> = HashMap::new();
        for (i, b) in self.bodies.iter().enumerate() {
            if matches!(b.kind, Kind::Chunk(..)) && b.x.x < 1e8 {
                *comp_mass.entry(self.comp[i]).or_default() += b.mass;
            }
        }
        let has_supports = self.comp_anchored.iter().any(|&a| a);
        let heaviest = comp_mass
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&c, _)| c);
        let mut detached = 0;
        let mut fast = 0;
        for (i, b) in self.bodies.iter().enumerate() {
            if !matches!(b.kind, Kind::Chunk(..)) || b.x.x > 1e8 {
                continue;
            }
            let main = if has_supports {
                self.comp_anchored[self.comp[i]]
            } else {
                Some(self.comp[i]) == heaviest
            };
            if !main {
                detached += 1;
                if self.impactor_speed0 > 0.0 && b.v.norm() > 0.5 * self.impactor_speed0 {
                    fast += 1;
                }
            }
        }
        AvbdReport {
            time: self.time,
            broken_bonds: broken,
            fragments: self.chunk_count_fragments(),
            detached_chunks: detached,
            fast_detached: fast,
            first_crack_time: self.first_crack,
            max_utilization: self.max_utilization,
            impactor_velocity: self
                .impactor_bodies
                .iter()
                .map(|&i| self.bodies[i].v.to_array())
                .collect(),
            bond_dissipation: self.bond_dissipation,
            damping_dissipation: self.damping_dissipation,
            stats: self.stats.clone(),
            probes: self.probes.clone(),
        }
    }

    /// Seconds per explicit substep's bond pass (every bond's law, dashpot and loads, plus
    /// the symplectic update) and per block Gauss-Seidel sweep (every free body's block
    /// solve with its bonds' tangents), at the current state, over `n` repetitions on
    /// the calling thread. The state is restored afterwards. Contacts are not included
    /// (none at rest).
    pub fn bench_kernels(&mut self, n: usize, h: f64) -> (f64, f64, usize) {
        let saved = self.bodies.clone();
        let fracture = self.scene.sim.fracture;
        let t0 = Instant::now();
        let mut sink = 0.0;
        for _ in 0..n {
            for b in self.bodies.iter_mut() {
                b.force = Vec3::ZERO;
                b.torque = Vec3::ZERO;
            }
            for k in 0..self.bonds.len() {
                if !self.bonds[k].active {
                    continue;
                }
                let ev = self.bond_eval(k, self.law_dt, fracture, None);
                let (a, b) = (self.bonds[k].a, self.bonds[k].b);
                self.bodies[a].force += ev.loads[0];
                self.bodies[a].torque += ev.loads[1];
                self.bodies[b].force += ev.loads[2];
                self.bodies[b].torque += ev.loads[3];
                sink += ev.resp.stored;
            }
            let dt = self.law_dt;
            for b in self.bodies.iter_mut() {
                if b.free() {
                    b.v += b.force * (1e-30 * dt / b.mass);
                    b.x += b.v * (1e-30 * dt);
                }
            }
        }
        let explicit = t0.elapsed().as_secs_f64() / n as f64;
        self.bodies = saved.clone();
        for b in self.bodies.iter_mut() {
            b.x0 = b.x;
            b.q0 = b.q;
            b.v0 = b.v;
            b.w0 = b.w;
            b.dx = Vec3::ZERO;
            b.dth = Vec3::ZERO;
        }
        self.contacts.clear();
        self.body_contacts.iter_mut().for_each(|l| l.clear());
        let groups = self.colour();
        let t1 = Instant::now();
        for _ in 0..n {
            for g in &groups {
                for &i in g {
                    let (delta, r, _, _, _) = self.block_solve(i, h);
                    sink += r;
                    let b = &mut self.bodies[i];
                    b.dx += Vec3::new(delta[0], delta[1], delta[2]) * 1e-30;
                }
            }
        }
        let sweep = t1.elapsed().as_secs_f64() / n as f64;
        self.bodies = saved;
        assert!(sink.is_finite());
        (explicit, sweep, groups.len())
    }

    /// Number of free bodies and active bonds (for cost normalisation).
    pub fn size(&self) -> (usize, usize) {
        (
            self.bodies.iter().filter(|b| b.free()).count(),
            self.bonds.iter().filter(|b| b.active).count(),
        )
    }
}

struct BondEval {
    resp: JointResponse,
    loads: [Vec3; 4],
    damp_power: f64,
    jac: Option<(M6, M6, Mat6)>,
}

#[allow(clippy::too_many_arguments)]
fn found_ground(
    out: &mut Vec<Detected>,
    j: usize,
    p: Vec3,
    depth: f64,
    k: f64,
    m: f64,
    mu: f64,
    np: usize,
    idx: usize,
) {
    out.push(Detected {
        a: j,
        b: None,
        point: p,
        n: Vec3::Z,
        depth,
        k,
        m_red: m,
        mu,
        points: np,
        key: (usize::MAX, j, idx),
    });
}

impl Body {
    fn sample_count_for_ground(&self) -> usize {
        if self.sphere.is_some() {
            1
        } else {
            5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::support_loss;

    fn peak_reaction(report: &AvbdReport) -> f64 {
        report.probes["reaction"].iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max)
    }

    /// b4 sudden support loss: the chunk-level explicit replica reproduces the oracle's
    /// peak reaction, and converged Newmark block descent at 5 us its amplification.
    #[test]
    fn b4_replica_matches_the_oracle_and_newmark_descent_converges_to_it() {
        // The established methods (what the world replicates: no Rayleigh mass damping).
        let mut scene = support_loss(0.0, "b4_support_loss_sudden");
        scene.sim.methods = crate::scene::Methods::established();
        let scene = scene.with_derived();
        let mut oracle = World::new(&scene);
        let obs = oracle.run();
        let oracle_peak = obs.probes["reaction"].hi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let world = World::new(&scene);
        let explicit = AvbdWorld::from_world(&world, AvbdConfig { integrator: Integrator::Explicit, end_time: Some(0.05), ..AvbdConfig::default() }).run();
        assert!((peak_reaction(&explicit) - oracle_peak).abs() < 1e-6 * oracle_peak, "{} vs {oracle_peak}", peak_reaction(&explicit));
        let newmark = AvbdWorld::from_world(
            &world,
            AvbdConfig { integrator: Integrator::Newmark, step: 5e-6, iterations: 50, tolerance: 1e-10, end_time: Some(0.05), ..AvbdConfig::default() },
        )
        .run();
        // Amplification from the reactions before (half the stack's weight plus the
        // lower chunk's own) and after (all of it).
        let (before, after) = (29.43, 39.24);
        let amp = |peak: f64| (peak - before) / (after - before);
        assert!((amp(peak_reaction(&newmark)) - amp(oracle_peak)).abs() < 0.01, "{} vs {}", amp(peak_reaction(&newmark)), amp(oracle_peak));
    }
}

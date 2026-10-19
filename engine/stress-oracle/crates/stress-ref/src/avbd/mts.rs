//! Multi-time-step explicit-implicit domain decomposition of a struck structure, after
//! Gravouil and Combescure (IJNME 2001, CMAME 2002). A research prototype for
//! `findings/multi-time-step.md`; nothing else uses it (feature `mts`).
//!
//! **Subdomains.** The chunk-level world of [`AvbdWorld`] (the oracle's laws: damageable
//! joints with true stiffness, sample-point penalty contact) is split by *bonds*:
//! - the **hot zone** (`core` chunks and a one-bond ring around them, the impactors and
//!   every contact) runs the oracle's explicit symplectic substep at its own step `dt`
//!   (the certified true step with `methods.true_step`);
//! - the **exterior** (every other bond) runs Newmark's average acceleration at the
//!   coarse step `h = m dt`, nonlinear: each bond's law is evaluated at the trial
//!   deformation from its step-start history (damage inside the iteration, as
//!   `avbd-stress.md` found necessary), solved by modified Newton with a profile Cholesky
//!   of `K_t + 2/h C + 4/h^2 M`, so the exterior can crack (`Exterior::Newmark`); or a
//!   linear Craig-Bampton reduction about its state at each repartition
//!   (`Exterior::CraigBampton`), which cannot;
//! - a ring chunk carries bonds of both zones: it is duplicated, its mass and inertia
//!   split in proportion to the axial stiffness of its bonds in each zone (each copy keeps
//!   the chunk's own `k/m`, so neither zone's stability changes), and the two copies are
//!   held to one velocity by interface forces `lambda` (6 per ring chunk).
//!
//! **Coupling (GC with a multiplier predictor).** Per coarse step from `t_n`: the hot
//! zone's force pass at `t_n`; the exterior's *free* problem with the multiplier at `t_n`
//! held as its interface load; for each substep `k` the condensed interface problem
//! `(dt/2 D + S) lambda_k = (1 - k/m) V_B(t_n) + (k/m) V_B^free + S lambda_0 - v_A^free`
//! for continuity of the hot copies' integer-time velocity (the central-difference form
//! of the symplectic substep) with the exterior's interpolated velocity; a look-ahead
//! force pass at `t_{n+1}` (not committed, identical to the next substep's) for the end
//! multiplier; the exterior's *link* problem with its change. `D` is the hot copies'
//! inverse mass, `S = 2/h C K_eff^-1 C^T` the exterior's condensed end-of-step velocity
//! response, the same at every substep. Weighting `S` by `k/m`, or leaving the predictor
//! out (the free problem without the interface force), is unstable or drains the
//! vibration through the interface (`findings/multi-time-step.md`). A velocity bias
//! `(x_hot - x_ext) / h` closing the copies' drift is optional and off by default.
//!
//! **Escalation** into the core (never back) at each coarse step's start, by
//! - contact closure: anything within `contact_margin` coarse steps of travel of touching
//!   anything (impactor, another component, the ground);
//! - detachment from the main piece, or a damaged bond;
//! - utilisation and rate: a bond whose utilisation, extrapolated `horizon` coarse steps
//!   at its current rate, plus (with `influence`) the bound of what the next coarse step's
//!   interface-force change can add to it, reaches `u_hot`. The test is on every exterior
//!   bond's own solved state, so remote failure by leverage (a base bond loaded by a
//!   moment arm) is caught by its own utilisation, not by its distance from the impact.
//! **Verification** after each coarse step: an exterior bond at or above `u_hot`, damaged,
//! or a contact touching the exterior is a *miss*; `Verify::Rollback` restores the step's
//! start, escalates the missed chunks and redoes it, so no accepted step has a crack or a
//! contact outside the hot zone.

use std::collections::HashMap;
use web_time::Instant;

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd;
use faer::linalg::solvers::DenseSolveCore;
use faer::Mat;

use super::skyline::{rcm, Skyline};
use super::*;
use crate::bond::Local6;

/// The exterior's model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Exterior {
    /// Nonlinear Newmark (the bonds' full laws, co-rotational kinematics).
    Newmark,
    /// Linear Craig-Bampton reduction about the step's start (cannot crack).
    CraigBampton,
}

/// What a verification miss does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Verify {
    /// Count it (the next step's escalation sees it).
    Count,
    /// Redo the coarse step with the missed chunks in the core.
    Rollback,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct MtsConfig {
    /// Coarse step over fine step, `m`.
    pub ratio: usize,
    pub exterior: Exterior,
    /// Craig-Bampton: fixed-interface modes kept.
    pub cb_modes: usize,
    /// Escalate a bond's chunks at this (predicted) utilisation.
    pub u_hot: f64,
    /// Rate extrapolation, in coarse steps.
    pub horizon: f64,
    /// Add the interface-load influence bound to the prediction.
    pub influence: bool,
    /// The bound on the next coarse step's interface-force change: this times the last
    /// change, per component.
    pub influence_growth: f64,
    pub verify: Verify,
    pub max_rollbacks: usize,
    /// Velocity bias `(x_hot - x_ext) / h` closing the copies' position drift (off by
    /// default: it destabilises the coupling on a spring chain).
    pub drift_correction: bool,
    pub newton_tol: f64,
    pub max_newton: usize,
    /// Contact closure margin, in coarse steps of travel.
    pub contact_margin: f64,
    /// Everything explicit (no exterior): must reproduce the explicit replica exactly.
    pub all_hot: bool,
    /// Chunks (of the first structure) in the core from the start.
    pub seed: Vec<usize>,
    /// Escalate (false: the partition stays the seed's, for coupling tests).
    pub escalation: bool,
    /// Record the energy ledger at every coarse step's end.
    pub energy: bool,
    /// The fine step as a fraction of the world's (tests only; the contact law's
    /// dashpot cap keeps the world's step).
    pub fine_scale: f64,
    pub end_time: Option<f64>,
}

impl Default for MtsConfig {
    fn default() -> Self {
        MtsConfig {
            ratio: 100,
            exterior: Exterior::Newmark,
            cb_modes: 40,
            u_hot: 0.5,
            horizon: 2.0,
            influence: true,
            influence_growth: 2.0,
            verify: Verify::Rollback,
            max_rollbacks: 3,
            drift_correction: false,
            newton_tol: 1e-8,
            max_newton: 60,
            contact_margin: 2.0,
            all_hot: false,
            seed: Vec::new(),
            escalation: true,
            energy: false,
            fine_scale: 1.0,
            end_time: None,
        }
    }
}

/// One coarse step's record.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CoarseRecord {
    pub t: f64,
    pub core: usize,
    pub hot: usize,
    pub ring: usize,
    pub hot_bonds: usize,
    /// Active bonds (what the full explicit solve evaluates per substep) and substeps.
    pub active_bonds: usize,
    pub substeps: usize,
    pub ext_bonds: usize,
    pub ext_dofs: usize,
    pub newton: usize,
    pub factorizations: usize,
    pub residual: f64,
    pub misses: usize,
    pub rollbacks: usize,
    /// Largest position mismatch of the ring's copies at the step's end (m) before the
    /// next step's bias, and the largest velocity mismatch (m/s).
    pub drift: f64,
    pub velocity_gap: f64,
    /// Largest exterior bond utilisation at the step's end, and the exterior bonds
    /// damaged (Craig-Bampton: over capacity) in this step.
    pub ext_u_max: f64,
    pub ext_cracks: usize,
    /// Energy at the step's end (with `energy`): kinetic, stored in the bonds, gravity's
    /// potential, and dissipated so far (bond laws and dashpots).
    pub kinetic: f64,
    pub stored: f64,
    pub potential: f64,
    pub dissipated: f64,
    /// Total linear momentum (all bodies, both copies).
    pub momentum: [f64; 3],
    /// Seconds since the run's start at the step's end.
    pub wall: f64,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct MtsStats {
    pub coarse_steps: u64,
    pub fine_steps: u64,
    /// Hot-zone work: bond and contact-point evaluations, body updates.
    pub hot_bond_evals: u64,
    pub hot_contact_evals: u64,
    pub hot_body_updates: u64,
    /// The full explicit solve's bond evaluations over the same fine steps (every active
    /// bond, every fine step) and body updates.
    pub full_bond_evals: u64,
    pub full_body_updates: u64,
    /// The look-ahead force passes (one per coarse step, for the end multiplier).
    pub lookahead_bond_evals: u64,
    /// Exterior work.
    pub newton_iterations: u64,
    pub ext_bond_evals: u64,
    pub assemblies: u64,
    pub factorizations: u64,
    pub factor_flops: f64,
    pub solves: u64,
    pub solve_flops: f64,
    pub interface_dofs_max: usize,
    pub eig_flops: f64,
    pub unconverged_steps: u64,
    pub worst_newton_residual: f64,
    pub regularised_factorizations: u64,
    /// Craig-Bampton: reductions, the largest reduced size, the last basis' lowest and
    /// highest kept fixed-interface frequency (rad/s).
    pub cb_reductions: u64,
    pub cb_reduced_max: usize,
    pub cb_mode_omega: [f64; 2],
    /// Verification.
    pub misses_utilization: u64,
    pub misses_contact: u64,
    pub misses_damage: u64,
    pub rollbacks: u64,
    pub repartitions: u64,
    /// Bonds damaged while in the exterior (accepted).
    pub exterior_cracks: u64,
    /// Seconds: hot fine steps, exterior Newton, factorisations with the interface
    /// operator, escalation and verification, total.
    pub t_hot: f64,
    pub t_newton: f64,
    pub t_factor: f64,
    pub t_escalate: f64,
    pub t_total: f64,
    pub records: Vec<CoarseRecord>,
}

/// A linear Craig-Bampton model of the exterior: `u = T q`, `q` the interface DOFs
/// (first `nb`, in the interface order) and fixed-interface modal amplitudes, about the
/// state at the reduction (`reference`), with the reduced Newmark operator's inverse.
#[derive(Clone)]
struct Cb {
    nb: usize,
    t: Vec<Vec<f64>>,
    mr: Mat<f64>,
    cr: Mat<f64>,
    keff_inv: Mat<f64>,
    /// Reduced forces at the reference (gravity, loads, bonds), and the external loads
    /// then (solver rows; their change enters each step).
    f0: Vec<f64>,
    fe_ref: Vec<f64>,
    q: Vec<f64>,
    qd: Vec<f64>,
    qdd: Vec<f64>,
    reference: Vec<(Vec3, Quat)>,
    omega_min: f64,
    omega_max: f64,
}

/// The exterior's solver for one partition.
#[derive(Clone)]
struct ExtSolver {
    /// Bodies with exterior DOFs, in solver order, and each body's position (or MAX).
    bodies: Vec<usize>,
    pos: Vec<usize>,
    /// Exterior bonds.
    bonds: Vec<usize>,
    mat: Skyline,
    factored: bool,
    /// Interface DOFs: (ring index, component 0..6) and their solver rows.
    if_dofs: Vec<(usize, usize)>,
    if_row: Vec<usize>,
    /// Per ring chunk: the index of its first linear / angular interface DOF.
    if_lin: Vec<Option<usize>>,
    if_ang: Vec<Option<usize>>,
    /// `K_eff^-1 e_i` for every interface DOF (solver rows).
    z: Vec<Vec<f64>>,
    /// `S = 2/h C K_eff^-1 C^T` (row-major, n_if^2).
    s: Vec<f64>,
    /// `D = L L^T`, per ring chunk: `sqrt(m)` and the angular block's Cholesky factor.
    l_lin: Vec<f64>,
    l_ang: Vec<Mat3>,
    /// Eigenvectors (column-major, n_if^2) and eigenvalues of `L^-1 S L^-T`.
    q: Vec<f64>,
    lam: Vec<f64>,
    /// The Craig-Bampton model (`Exterior::CraigBampton`).
    cb: Option<Cb>,
    /// The coarse step its operators were built for.
    h: f64,
}

impl ExtSolver {
    fn n_if(&self) -> usize {
        self.if_dofs.len()
    }

    /// `x <- L^-1 x` (`transpose = false`) or `L^-T x` on an interface vector.
    fn apply_l_inv(&self, x: &mut [f64], transpose: bool) {
        for k in 0..self.if_lin.len() {
            if let Some(i0) = self.if_lin[k] {
                for c in 0..3 {
                    x[i0 + c] /= self.l_lin[k];
                }
            }
            if let Some(i0) = self.if_ang[k] {
                let l = &self.l_ang[k].m;
                let v = [x[i0], x[i0 + 1], x[i0 + 2]];
                let mut y = [0.0; 3];
                if !transpose {
                    for r in 0..3 {
                        let mut s = v[r];
                        for c in 0..r {
                            s -= l[r][c] * y[c];
                        }
                        y[r] = s / l[r][r];
                    }
                } else {
                    for r in (0..3).rev() {
                        let mut s = v[r];
                        for c in r + 1..3 {
                            s -= l[c][r] * y[c];
                        }
                        y[r] = s / l[r][r];
                    }
                }
                x[i0..i0 + 3].copy_from_slice(&y);
            }
        }
    }

    fn s_mul(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n_if();
        (0..n)
            .map(|i| self.s[i * n..(i + 1) * n].iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Solve `(dt D + c S) x = b`.
    fn solve_interface(&self, dt: f64, c: f64, b: &[f64]) -> Vec<f64> {
        let n = self.n_if();
        let mut y = b.to_vec();
        self.apply_l_inv(&mut y, false);
        let mut t = vec![0.0; n];
        for e in 0..n {
            let col = &self.q[e * n..(e + 1) * n];
            let mut s = 0.0;
            for i in 0..n {
                s += col[i] * y[i];
            }
            t[e] = s / (dt + c * self.lam[e]);
        }
        let mut x = vec![0.0; n];
        for e in 0..n {
            let col = &self.q[e * n..(e + 1) * n];
            for i in 0..n {
                x[i] += col[i] * t[e];
            }
        }
        self.apply_l_inv(&mut x, true);
        x
    }
}

/// Eigenvalues and eigenvectors (columns) of a symmetric matrix (faer's tridiagonal QR).
fn sym_eig(a: &Mat<f64>) -> (Vec<f64>, Mat<f64>) {
    let n = a.nrows();
    let par = faer::Par::Seq;
    let mut sdiag = Diag::<f64>::zeros(n);
    let mut qm = Mat::<f64>::zeros(n, n);
    let mut params = <evd::SelfAdjointEvdParams as faer::Auto<f64>>::auto();
    params.recursion_threshold = usize::MAX;
    let spec: faer::Spec<evd::SelfAdjointEvdParams, f64> = params.into();
    let req = evd::self_adjoint_evd_scratch::<f64>(n, evd::ComputeEigenvectors::Yes, par, spec);
    let mut buf = MemBuffer::new(req);
    evd::self_adjoint_evd(a.as_ref(), sdiag.as_mut(), Some(qm.as_mut()), par, MemStack::new(&mut buf), spec)
        .expect("symmetric eigensolver");
    ((0..n).map(|i| sdiag[i]).collect(), qm)
}

/// Lower Cholesky factor of a 3x3 SPD matrix.
fn chol3(a: &Mat3) -> Mat3 {
    let m = &a.m;
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut s = m[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][i] = s.max(1e-300).sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Mat3 { m: l }
}

/// `a^T k b` for 6x6 blocks.
fn jtkj(a: &M6, k: &Mat6, b: &M6) -> M6 {
    let mut kb = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            let mut s = 0.0;
            for m in 0..6 {
                s += k.0[r][m] * b[m][c];
            }
            kb[r][c] = s;
        }
    }
    let mut out = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            let mut s = 0.0;
            for m in 0..6 {
                s += a[m][r] * kb[m][c];
            }
            out[r][c] = s;
        }
    }
    out
}


/// The multi-time-step world (see the module documentation).
#[derive(Clone)]
pub struct MtsWorld {
    pub w: AvbdWorld,
    pub config: MtsConfig,
    mass0: Vec<f64>,
    inertia0: Vec<Mat3>,
    core: Vec<bool>,
    hot: Vec<bool>,
    /// Hot bodies and impactors: what the fine steps integrate.
    active: Vec<bool>,
    bond_hot: Vec<bool>,
    hot_bonds: Vec<usize>,
    ring: Vec<usize>,
    ring_of: Vec<usize>,
    /// Hot mass share per body (1 in the core, the split share on the ring, 0 outside).
    share: Vec<f64>,
    /// The ring's exterior copies.
    ext_copy: Vec<Body>,
    /// Interface force on each ring chunk's hot copy at the last coarse step's end, and
    /// the one before (the influence bound's change).
    lam_prev: Vec<[f64; 6]>,
    lam_prev2: Vec<[f64; 6]>,
    solver: Option<ExtSolver>,
    /// The end-of-step multiplier (the next coarse step's first), in the solver's
    /// interface order; `None` after a repartition.
    lam_next: Option<Vec<f64>>,
    /// The ring's hot copies' last look-ahead force and torque with the multiplier.
    last_force: Vec<(Vec3, Vec3)>,
    /// Bodies whose velocity is at the integer time (after a repartition).
    rebase: Vec<bool>,
    prev_u: Vec<f64>,
    influence: Vec<f64>,
    forced: Vec<bool>,
    pub stats: MtsStats,
    diverged: bool,
    started: Option<Instant>,
}

/// Outcome of a run: the AVBD world's report plus the detached set and the cost.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MtsReport {
    pub outcome: AvbdReport,
    /// Detached chunks (structure, chunk) and their mean speed.
    pub detached: Vec<(usize, usize)>,
    pub debris_speed: f64,
    pub diverged: bool,
    pub stats: MtsStats,
}

impl MtsWorld {
    pub fn new(world: &World, config: MtsConfig) -> MtsWorld {
        let aw = AvbdWorld::from_world(
            world,
            AvbdConfig {
                integrator: Integrator::Explicit,
                end_time: config.end_time,
                ..AvbdConfig::default()
            },
        );
        let n = aw.bodies.len();
        let nb = aw.bonds.len();
        let mut m = MtsWorld {
            mass0: aw.bodies.iter().map(|b| b.mass).collect(),
            inertia0: aw.bodies.iter().map(|b| b.inertia).collect(),
            prev_u: aw.bonds.iter().map(|b| b.rt.joint.utilization).collect(),
            w: aw,
            config,
            core: vec![false; n],
            hot: vec![false; n],
            active: vec![false; n],
            bond_hot: vec![false; nb],
            hot_bonds: Vec::new(),
            ring: Vec::new(),
            ring_of: vec![usize::MAX; n],
            share: vec![0.0; n],
            ext_copy: Vec::new(),
            lam_prev: Vec::new(),
            lam_prev2: Vec::new(),
            solver: None,
            lam_next: None,
            last_force: vec![(Vec3::ZERO, Vec3::ZERO); n],
            rebase: vec![false; n],
            influence: vec![0.0; nb],
            forced: vec![false; n],
            stats: MtsStats::default(),
            diverged: false,
            started: None,
        };
        // Newmark's start acceleration (the prestressed equilibrium's, about zero).
        m.w.init_accelerations();
        m.repartition();
        m
    }

    fn free_chunk(&self, i: usize) -> bool {
        let b = &self.w.bodies[i];
        matches!(b.kind, Kind::Chunk(..)) && b.free() && b.x.x < 1e8
    }

    fn is_impactor(&self, i: usize) -> bool {
        matches!(self.w.bodies[i].kind, Kind::Impactor(_))
    }

    /// Bodies of the main piece (anchored components, else the heaviest).
    fn main_mask(&self) -> Vec<bool> {
        let w = &self.w;
        let mut comp_mass: HashMap<usize, f64> = HashMap::new();
        for (i, b) in w.bodies.iter().enumerate() {
            if matches!(b.kind, Kind::Chunk(..)) && b.x.x < 1e8 {
                *comp_mass.entry(w.comp[i]).or_default() += self.mass0[i];
            }
        }
        let has_supports = w.comp_anchored.iter().any(|&a| a);
        let heaviest = comp_mass
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&c, _)| c);
        (0..w.bodies.len())
            .map(|i| {
                if has_supports {
                    w.comp_anchored[w.comp[i]]
                } else {
                    Some(w.comp[i]) == heaviest
                }
            })
            .collect()
    }

    // ------------------------------------------------------------------ partition

    /// Swap the ring's exterior copies into the world (and back).
    fn swap_ring(&mut self) {
        for (k, &i) in self.ring.iter().enumerate() {
            std::mem::swap(&mut self.w.bodies[i], &mut self.ext_copy[k]);
        }
    }

    /// Merge the ring's copies, then rebuild the hot set, the ring and its copies from
    /// `core`.
    fn repartition(&mut self) {
        self.stats.repartitions += 1;
        let n = self.w.bodies.len();
        let dt = self.w.law_dt * self.config.fine_scale;
        let old_hot = self.hot.clone();
        // Merge at the integer time (the hot copy's half-step velocity plus half its last
        // look-ahead kick): momentum-weighted velocities, mass-weighted positions, the
        // exterior's acceleration memory kept.
        for (k, &i) in self.ring.clone().iter().enumerate() {
            let e = self.ext_copy[k].clone();
            let m0 = self.mass0[i];
            let (f, t) = self.last_force[i];
            self.rebase[i] = true;
            let b = &mut self.w.bodies[i];
            let (mb, me) = (b.mass, e.mass);
            let l = b.world_inertia(b.q) * b.w + t * (0.5 * dt) + e.world_inertia(e.q) * e.w;
            b.v = ((b.v + f * (0.5 * dt / mb)) * mb + e.v * me) / m0;
            b.x = (b.x * mb + e.x * me) / m0;
            b.mass = m0;
            b.inertia = self.inertia0[i];
            b.w = b.world_inertia(b.q).inverse().expect("inertia") * l;
            b.a0 = e.a0;
            b.alpha0 = e.alpha0;
        }
        // Hot: core, impactors, and the ring (free chunks bonded to the core).
        let mut hot = self.core.clone();
        for b in &self.w.bonds {
            if !b.active {
                continue;
            }
            if self.core[b.a] && self.free_chunk(b.b) {
                hot[b.b] = true;
            }
            if self.core[b.b] && self.free_chunk(b.a) {
                hot[b.a] = true;
            }
        }
        for i in 0..n {
            if !self.free_chunk(i) {
                hot[i] = false;
            }
        }
        self.hot = hot;
        self.active = (0..n)
            .map(|i| self.hot[i] || (self.is_impactor(i) && self.w.bodies[i].free()))
            .collect();
        // Bonds: hot if every free end is hot.
        let nb = self.w.bonds.len();
        self.bond_hot = vec![false; nb];
        self.hot_bonds.clear();
        for k in 0..nb {
            let b = &self.w.bonds[k];
            if !b.active {
                continue;
            }
            let fa = self.w.bodies[b.a].free();
            let fb = self.w.bodies[b.b].free();
            if !(fa || fb) {
                continue;
            }
            let h = (!fa || self.hot[b.a]) && (!fb || self.hot[b.b]);
            self.bond_hot[k] = h;
            if h {
                self.hot_bonds.push(k);
            }
        }
        // Ring: hot chunks with an exterior bond; split by axial stiffness share.
        let mut ring = Vec::new();
        let mut share = vec![0.0; n];
        for i in 0..n {
            if self.active[i] {
                share[i] = 1.0;
            }
            if !self.hot[i] {
                continue;
            }
            let (mut k_hot, mut k_all, mut has_ext) = (0.0, 0.0, false);
            for &k in &self.w.body_bonds[i] {
                let b = &self.w.bonds[k];
                if !b.active {
                    continue;
                }
                k_all += b.rt.stiffness.kn;
                if self.bond_hot[k] {
                    k_hot += b.rt.stiffness.kn;
                } else {
                    has_ext = true;
                }
            }
            if has_ext {
                ring.push(i);
                share[i] = k_hot / k_all;
            }
        }
        self.ring_of = vec![usize::MAX; n];
        self.ext_copy.clear();
        for (k, &i) in ring.iter().enumerate() {
            self.ring_of[i] = k;
            let phi = share[i];
            let b = &mut self.w.bodies[i];
            let mut e = b.clone();
            e.mass = self.mass0[i] * (1.0 - phi);
            e.inertia = self.inertia0[i] * (1.0 - phi);
            b.mass = self.mass0[i] * phi;
            b.inertia = self.inertia0[i] * phi;
            self.ext_copy.push(e);
        }
        self.ring = ring;
        self.share = share;
        // Bodies new to the hot zone carry the exterior's integer-time velocity (at the
        // start, the initial velocities are the truth's first half-step velocities).
        let started = self.stats.coarse_steps > 0;
        for i in 0..n {
            if started && self.hot[i] && !old_hot[i] {
                self.rebase[i] = true;
            }
        }
        self.lam_next = None;
        self.lam_prev = vec![[0.0; 6]; self.ring.len()];
        self.lam_prev2 = self.lam_prev.clone();
        self.solver = None;
        self.w.detect_mask = Some(self.active.clone());
    }

    /// Escalate chunks into the core for the coarse step about to start.
    fn escalate(&mut self, h: f64) {
        let t0 = Instant::now();
        let n = self.w.bodies.len();
        let mut core = self.core.clone();
        for i in 0..n {
            if !self.free_chunk(i) {
                continue;
            }
            let seeded = matches!(self.w.bodies[i].kind, Kind::Chunk(0, c) if self.config.seed.contains(&c));
            if self.config.all_hot || self.forced[i] || seeded {
                core[i] = true;
            }
        }
        if !self.config.escalation {
            if core != self.core {
                self.core = core;
                self.repartition();
            }
            self.stats.t_escalate += t0.elapsed().as_secs_f64();
            return;
        }
        let main = self.main_mask();
        for i in 0..n {
            if self.free_chunk(i) && !main[i] {
                core[i] = true;
            }
        }
        let u_hot = self.config.u_hot;
        for (k, b) in self.w.bonds.iter().enumerate() {
            if !b.active {
                continue;
            }
            let u = b.rt.joint.utilization;
            let rate = ((u - self.prev_u[k]) / h).max(0.0);
            let pred = u + self.config.horizon * h * rate + self.influence[k];
            if b.rt.joint.is_damaged() || pred >= u_hot {
                for i in [b.a, b.b] {
                    if self.free_chunk(i) {
                        core[i] = true;
                    }
                }
            }
        }
        // Contact closure.
        self.w.detect_mask = None;
        let speeds: Vec<f64> = self
            .w
            .bodies
            .iter()
            .map(|b| if b.free() { b.v.norm() + b.w.norm() * b.radius() } else { 0.0 })
            .collect();
        let factor = self.config.contact_margin;
        let margin = |a: usize, b: Option<usize>| {
            factor * h * (speeds[a] + b.map_or(0.0, |b| speeds[b])) + 1e-7
        };
        let found = self.w.detect(&margin);
        for d in &found {
            for i in [Some(d.a), d.b].into_iter().flatten() {
                if self.free_chunk(i) {
                    core[i] = true;
                }
            }
        }
        self.w.detect_mask = Some(self.active.clone());
        if core != self.core {
            self.core = core;
            self.repartition();
        }
        self.stats.t_escalate += t0.elapsed().as_secs_f64();
    }

    // ------------------------------------------------------------------ exterior

    fn build_solver(&mut self) {
        let n = self.w.bodies.len();
        let bodies0: Vec<usize> = (0..n)
            .filter(|&i| self.free_chunk(i) && (!self.hot[i] || self.ring_of[i] != usize::MAX))
            .collect();
        let mut local = vec![usize::MAX; n];
        for (p, &i) in bodies0.iter().enumerate() {
            local[i] = p;
        }
        let bonds: Vec<usize> = (0..self.w.bonds.len())
            .filter(|&k| {
                let b = &self.w.bonds[k];
                b.active
                    && !self.bond_hot[k]
                    && (local[b.a] != usize::MAX || local[b.b] != usize::MAX)
            })
            .collect();
        let mut adj = vec![Vec::new(); bodies0.len()];
        for &k in &bonds {
            let b = &self.w.bonds[k];
            let (pa, pb) = (local[b.a], local[b.b]);
            if pa != usize::MAX && pb != usize::MAX {
                adj[pa].push(pb);
                adj[pb].push(pa);
            }
        }
        let order = rcm(&adj);
        let bodies: Vec<usize> = order.iter().map(|&p| bodies0[p]).collect();
        let mut pos = vec![usize::MAX; n];
        for (p, &i) in bodies.iter().enumerate() {
            pos[i] = p;
        }
        let mut pattern = Vec::new();
        for &i in &bodies {
            let p = pos[i];
            for r in 0..6 {
                for c in 0..=r {
                    pattern.push((6 * p + r, 6 * p + c));
                }
            }
        }
        for &k in &bonds {
            let b = &self.w.bonds[k];
            let (pa, pb) = (pos[b.a], pos[b.b]);
            if pa != usize::MAX && pb != usize::MAX {
                let (hi, lo) = if pa > pb { (pa, pb) } else { (pb, pa) };
                pattern.push((6 * hi, 6 * lo));
            }
        }
        // A block's corner sets its profile: every row of the higher body reaches the
        // lower body's first column.
        let mut first_body = (0..bodies.len()).collect::<Vec<_>>();
        for &(r, c) in &pattern {
            let (bi, bj) = (r / 6, c / 6);
            first_body[bi] = first_body[bi].min(bj);
        }
        let nd = bodies.len();
        let full_pattern = (0..nd).flat_map(|p| {
            let f = first_body[p];
            (0..6).map(move |r| (6 * p + r, 6 * f))
        });
        let mat = Skyline::with_pattern(6 * nd, full_pattern);
        let mut if_dofs = Vec::new();
        let mut if_row = Vec::new();
        let mut if_lin = vec![None; self.ring.len()];
        let mut if_ang = vec![None; self.ring.len()];
        for (k, &i) in self.ring.iter().enumerate() {
            let b = &self.w.bodies[i];
            if !b.fixed_lin {
                if_lin[k] = Some(if_dofs.len());
                for c in 0..3 {
                    if_dofs.push((k, c));
                    if_row.push(6 * pos[i] + c);
                }
            }
            if !b.fixed_ang {
                if_ang[k] = Some(if_dofs.len());
                for c in 3..6 {
                    if_dofs.push((k, c));
                    if_row.push(6 * pos[i] + c);
                }
            }
        }
        self.stats.interface_dofs_max = self.stats.interface_dofs_max.max(if_dofs.len());
        self.solver = Some(ExtSolver {
            bodies,
            pos,
            bonds,
            mat,
            factored: false,
            if_dofs,
            if_row,
            if_lin,
            if_ang,
            z: Vec::new(),
            s: Vec::new(),
            l_lin: Vec::new(),
            l_ang: Vec::new(),
            q: Vec::new(),
            lam: Vec::new(),
            cb: None,
            h: 0.0,
        });
    }

    fn fixed_comp(b: &Body, c: usize) -> bool {
        if c < 3 {
            b.fixed_lin
        } else {
            b.fixed_ang
        }
    }

    /// Newmark trial state of an exterior body from its increment.
    fn set_trial_nm(b: &mut Body, h: f64) {
        b.x = b.x0 + b.dx;
        b.q = (Quat::from_rotation_vector(b.dth) * b.q0).normalized();
        b.v = b.dx * (2.0 / h) - b.v0;
        b.w = b.dth * (2.0 / h) - b.w0;
    }

    /// Residual `M a - f` of the exterior (solver rows) and its force scale. The ring's
    /// exterior copies must be swapped in.
    fn ext_residual(&mut self, h: f64, iface_load: &[[f64; 6]]) -> (Vec<f64>, f64, f64) {
        let fracture = self.w.scene.sim.fracture;
        let sv = self.solver.as_ref().expect("solver");
        let nd = sv.bodies.len();
        let mut r = vec![0.0; 6 * nd];
        let mut scale2 = 0.0;
        let c_m = 4.0 / (h * h);
        let g = self.w.gravity;
        for (p, &i) in sv.bodies.iter().enumerate() {
            let b = &self.w.bodies[i];
            let es = 1.0 - self.share[i];
            let iw = b.world_inertia(b.q0);
            let a = (b.dx - b.v0 * h) * c_m - b.a0;
            let al = (b.dth - b.w0 * h) * c_m - b.alpha0;
            let mut f = b.ext_f * es + g * b.mass;
            let mut t = b.ext_t * es;
            let k = self.ring_of[i];
            if k != usize::MAX {
                let l = iface_load[k];
                f -= Vec3::new(l[0], l[1], l[2]);
                t -= Vec3::new(l[3], l[4], l[5]);
            }
            scale2 += f.norm2();
            let fi = a * b.mass - f;
            let ti = iw * al - t;
            for c in 0..3 {
                r[6 * p + c] = fi[c];
                r[6 * p + 3 + c] = ti[c];
            }
        }
        let bonds = sv.bonds.clone();
        let evs: Vec<BondEval> =
            crate::par::map_range(0..bonds.len(), 64, |q| self.w.bond_eval(bonds[q], h, fracture, None));
        let sv = self.solver.as_ref().expect("solver");
        for (q, ev) in evs.iter().enumerate() {
            let b = &self.w.bonds[bonds[q]];
            for (body, fl, ml) in [(b.a, ev.loads[0], ev.loads[1]), (b.b, ev.loads[2], ev.loads[3])] {
                let p = sv.pos[body];
                if p == usize::MAX {
                    continue;
                }
                scale2 += fl.norm2();
                for c in 0..3 {
                    r[6 * p + c] -= fl[c];
                    r[6 * p + 3 + c] -= ml[c];
                }
            }
        }
        self.stats.ext_bond_evals += bonds.len() as u64;
        // Fixed components; the norm weighs moments by 1/radius (as the block solves).
        let mut norm2 = 0.0;
        for (p, &i) in sv.bodies.iter().enumerate() {
            let b = &self.w.bodies[i];
            let rc = b.radius().max(1e-6);
            for c in 0..6 {
                if Self::fixed_comp(b, c) {
                    r[6 * p + c] = 0.0;
                } else {
                    let s = if c < 3 { 1.0 } else { 1.0 / rc };
                    norm2 += (r[6 * p + c] * s).powi(2);
                }
            }
        }
        (r, norm2.sqrt(), scale2.sqrt())
    }

    /// Assemble `K_t + c_v C + c_m M` (the bonds' tangent with the dashpots' share and
    /// the mass) at the current trial state, on the solver's profile. Fixed components
    /// get a unit diagonal; rows and columns flagged in `skip` are left out except their
    /// unit diagonal (Craig-Bampton's interior operator).
    fn assemble(&mut self, h: f64, c_m: f64, c_v: f64, skip: Option<&[bool]>) -> Skyline {
        let fracture = self.w.scene.sim.fracture;
        let bonds = self.solver.as_ref().unwrap().bonds.clone();
        let evs: Vec<BondEval> =
            crate::par::map_range(0..bonds.len(), 64, |q| self.w.bond_eval(bonds[q], h, fracture, Some(c_v)));
        self.stats.ext_bond_evals += bonds.len() as u64;
        self.stats.assemblies += 1;
        let sv = self.solver.as_ref().unwrap();
        let mut mat = sv.mat.clone();
        mat.clear();
        let skipped = |r: usize| skip.is_some_and(|s| s[r]);
        for (p, &i) in sv.bodies.iter().enumerate() {
            let b = &self.w.bodies[i];
            let iw = b.world_inertia(b.q0);
            for r in 0..6 {
                if Self::fixed_comp(b, r) || skipped(6 * p + r) {
                    mat.add(6 * p + r, 6 * p + r, 1.0);
                    continue;
                }
                if c_m == 0.0 {
                    continue;
                }
                if r < 3 {
                    mat.add(6 * p + r, 6 * p + r, b.mass * c_m);
                } else {
                    for c in 3..=r {
                        if !Self::fixed_comp(b, c) && !skipped(6 * p + c) {
                            mat.add(6 * p + r, 6 * p + c, iw.m[r - 3][c - 3] * c_m);
                        }
                    }
                }
            }
        }
        for (q, ev) in evs.iter().enumerate() {
            let b = &self.w.bonds[bonds[q]];
            let (ja, jb, kt) = ev.jac.as_ref().expect("jacobians");
            let (pa, pb) = (sv.pos[b.a], sv.pos[b.b]);
            let (ba, bb) = (&self.w.bodies[b.a], &self.w.bodies[b.b]);
            let blocks = [(pa, pa, ja, ja, ba, ba), (pb, pb, jb, jb, bb, bb), (pb, pa, jb, ja, bb, ba)];
            for (pi, pj, ji, jj, bi, bj) in blocks {
                if pi == usize::MAX || pj == usize::MAX {
                    continue;
                }
                let blk = jtkj(ji, kt, jj);
                for r in 0..6 {
                    if Self::fixed_comp(bi, r) || skipped(6 * pi + r) {
                        continue;
                    }
                    for c in 0..6 {
                        if Self::fixed_comp(bj, c) || skipped(6 * pj + c) {
                            continue;
                        }
                        let (gr, gc) = (6 * pi + r, 6 * pj + c);
                        if pi == pj {
                            if gc <= gr {
                                // Symmetric part of the diagonal block (stored once).
                                mat.add(gr, gc, 0.5 * (blk[r][c] + blk[c][r]));
                            }
                        } else {
                            mat.add(gr, gc, blk[r][c]);
                        }
                    }
                }
            }
        }
        mat
    }

    /// Factor an assembled operator; an indefinite one (softening bonds, a cracked patch)
    /// is retried with its diagonal raised (which only slows a modified Newton
    /// iteration); past that the run is marked diverged.
    fn factor_matrix(&mut self, mut mat: Skyline) -> Skyline {
        let assembled = mat.clone();
        let diag = assembled.diagonal();
        let mut ok = mat.factor();
        let mut shift = 0.01;
        while !ok && shift < 1e3 {
            self.stats.regularised_factorizations += 1;
            mat = assembled.clone();
            for (r, d) in diag.iter().enumerate() {
                mat.add(r, r, shift * d.abs());
            }
            ok = mat.factor();
            shift *= 10.0;
        }
        if !ok {
            self.diverged = true;
            eprintln!("MTS world: exterior operator indefinite at t = {:.6e}", self.w.time);
            // Leave a usable (identity-scaled) operator; the run stops at this step.
            mat = assembled;
            mat.clear();
            for (r, d) in diag.iter().enumerate() {
                mat.add(r, r, d.abs().max(1.0));
            }
            assert!(mat.factor());
        }
        self.stats.factorizations += 1;
        self.stats.factor_flops += mat.factor_flops();
        mat
    }

    /// Newmark's operator `K_t + 2/h C + 4/h^2 M` at the trial state, factored, and the
    /// interface operator from it.
    fn factor(&mut self, h: f64, hot_mass: &[(f64, Mat3)]) {
        let t0 = Instant::now();
        let mat = self.assemble(h, 4.0 / (h * h), 2.0 / h, None);
        let mat = self.factor_matrix(mat);
        let sv = self.solver.as_mut().unwrap();
        sv.mat = mat;
        sv.factored = true;
        let n_if = sv.n_if();
        let nr = sv.mat.n;
        sv.z = (0..n_if)
            .map(|d| {
                let mut e = vec![0.0; nr];
                e[sv.if_row[d]] = 1.0;
                e
            })
            .collect();
        {
            let mat = &sv.mat;
            sv.z.iter_mut().for_each(|e| mat.solve(e));
        }
        self.stats.solves += n_if as u64;
        self.stats.solve_flops += n_if as f64 * sv.mat.solve_flops();
        let mut s = vec![0.0; n_if * n_if];
        for a in 0..n_if {
            for b in 0..n_if {
                s[a * n_if + b] = (2.0 / h) * 0.5 * (sv.z[b][sv.if_row[a]] + sv.z[a][sv.if_row[b]]);
            }
        }
        self.set_interface_operator(s, hot_mass);
        self.stats.t_factor += t0.elapsed().as_secs_f64();
    }

    /// Install `S` and the eigendecomposition of `L^-1 S L^-T`, `D = L L^T` the hot
    /// copies' inverse masses.
    fn set_interface_operator(&mut self, s: Vec<f64>, hot_mass: &[(f64, Mat3)]) {
        let sv = self.solver.as_mut().unwrap();
        let n_if = sv.n_if();
        sv.s = s;
        sv.l_lin = hot_mass.iter().map(|(m, _)| (1.0 / m).sqrt()).collect();
        sv.l_ang = hot_mass.iter().map(|(_, iw)| chol3(&iw.inverse().expect("inertia"))).collect();
        let mut a = sv.s.clone();
        for c in 0..n_if {
            let mut col: Vec<f64> = (0..n_if).map(|r| a[r * n_if + c]).collect();
            sv.apply_l_inv(&mut col, false);
            for r in 0..n_if {
                a[r * n_if + c] = col[r];
            }
        }
        for r in 0..n_if {
            let mut row: Vec<f64> = a[r * n_if..(r + 1) * n_if].to_vec();
            sv.apply_l_inv(&mut row, false);
            a[r * n_if..(r + 1) * n_if].copy_from_slice(&row);
        }
        if n_if > 0 {
            let am = Mat::<f64>::from_fn(n_if, n_if, |r, c| 0.5 * (a[r * n_if + c] + a[c * n_if + r]));
            let (lam, q) = sym_eig(&am);
            sv.lam = lam.iter().map(|l| l.max(0.0)).collect();
            sv.q = (0..n_if * n_if).map(|k| q[(k % n_if, k / n_if)]).collect();
            self.stats.eig_flops += 9.0 * (n_if as f64).powi(3);
        } else {
            sv.lam.clear();
            sv.q.clear();
        }
    }

    /// The exterior's free Newmark step (or its link correction, with `link`): modified
    /// Newton from the current increments. Ring copies swapped in.
    fn ext_solve(&mut self, h: f64, iface_load: &[[f64; 6]], hot_mass: &[(f64, Mat3)]) -> (usize, f64) {
        let t0 = Instant::now();
        let tol = self.config.newton_tol;
        let mut last = f64::INFINITY;
        let mut iters = 0;
        let mut rel = 0.0;
        let mut since_factor = 0;
        for it in 0..self.config.max_newton {
            let (r, norm, scale) = self.ext_residual(h, iface_load);
            rel = norm / scale.max(1e-300);
            if rel <= tol {
                break;
            }
            let need = !self.solver.as_ref().unwrap().factored
                || (since_factor >= 2 && norm > 0.5 * last)
                || since_factor >= 12;
            if need {
                let tf = Instant::now();
                self.factor(h, hot_mass);
                self.stats.t_newton -= tf.elapsed().as_secs_f64();
                since_factor = 0;
            }
            last = norm;
            let sv = self.solver.as_ref().unwrap();
            let mut d: Vec<f64> = r.iter().map(|x| -x).collect();
            sv.mat.solve(&mut d);
            self.stats.solves += 1;
            self.stats.solve_flops += sv.mat.solve_flops();
            let bodies = sv.bodies.clone();
            for (p, &i) in bodies.iter().enumerate() {
                let b = &mut self.w.bodies[i];
                b.dx += Vec3::new(d[6 * p], d[6 * p + 1], d[6 * p + 2]);
                b.dth += Vec3::new(d[6 * p + 3], d[6 * p + 4], d[6 * p + 5]);
                Self::set_trial_nm(b, h);
            }
            iters = it + 1;
            since_factor += 1;
        }
        if rel > tol {
            self.stats.unconverged_steps += 1;
        }
        self.stats.worst_newton_residual = self.stats.worst_newton_residual.max(rel);
        self.stats.newton_iterations += iters as u64;
        self.stats.t_newton += t0.elapsed().as_secs_f64();
        (iters, rel)
    }

    // ------------------------------------------------------------------ Craig-Bampton

    /// The exterior's physical forces without inertia (gravity, external loads and its
    /// bonds' forces at the current state), solver rows; ring copies swapped in.
    fn ext_forces(&mut self, h: f64) -> Vec<f64> {
        let zero = vec![[0.0; 6]; self.ring.len()];
        let sv = self.solver.as_ref().unwrap();
        // The residual with zero inertia: set the increments so `M a = 0`.
        let bodies = sv.bodies.clone();
        let saved: Vec<(Vec3, Vec3, Vec3, Vec3)> =
            bodies.iter().map(|&i| { let b = &self.w.bodies[i]; (b.dx, b.dth, b.a0, b.alpha0) }).collect();
        for &i in &bodies {
            let b = &mut self.w.bodies[i];
            b.a0 = Vec3::ZERO;
            b.alpha0 = Vec3::ZERO;
            b.dx = b.v0 * h;
            b.dth = b.w0 * h;
        }
        let (r, _, _) = self.ext_residual(h, &zero);
        for (&i, s) in bodies.iter().zip(saved) {
            let b = &mut self.w.bodies[i];
            (b.dx, b.dth, b.a0, b.alpha0) = s;
        }
        r.iter().map(|x| -x).collect()
    }

    /// External loads on the exterior at `t` (solver rows, each copy's share); ring
    /// copies swapped in.
    fn ext_loads_at(&mut self, t: f64) -> Vec<f64> {
        self.w.external_loads(t);
        let sv = self.solver.as_ref().unwrap();
        let mut f = vec![0.0; sv.mat.n];
        for (p, &i) in sv.bodies.iter().enumerate() {
            let b = &self.w.bodies[i];
            let es = 1.0 - self.share[i];
            for c in 0..3 {
                f[6 * p + c] = b.ext_f[c] * es;
                f[6 * p + 3 + c] = b.ext_t[c] * es;
            }
        }
        f
    }

    /// `M x` of the exterior (solver rows), inertia at the step's start.
    fn ext_mass_mul(&self, x: &[f64]) -> Vec<f64> {
        let sv = self.solver.as_ref().unwrap();
        let mut y = vec![0.0; x.len()];
        for (p, &i) in sv.bodies.iter().enumerate() {
            let b = &self.w.bodies[i];
            let iw = b.world_inertia(b.q0);
            for r in 0..3 {
                if !b.fixed_lin {
                    y[6 * p + r] = b.mass * x[6 * p + r];
                }
                if !b.fixed_ang {
                    y[6 * p + 3 + r] = (0..3).map(|c| iw.m[r][c] * x[6 * p + 3 + c]).sum();
                }
            }
        }
        y
    }

    /// Reduce the exterior (ring copies swapped in, at the step's start) to Craig-Bampton
    /// coordinates: the interface DOFs and the lowest `cb_modes` fixed-interface modes,
    /// linear about the current state, and install its interface operator.
    fn build_cb(&mut self, h: f64, hot_mass: &[(f64, Mat3)], lam0: &[f64]) {
        let t0 = Instant::now();
        let (nrow, nb, if_row, bodies) = {
            let sv = self.solver.as_ref().unwrap();
            (sv.mat.n, sv.n_if(), sv.if_row.clone(), sv.bodies.clone())
        };
        for &i in &bodies {
            let b = &mut self.w.bodies[i];
            b.x0 = b.x;
            b.q0 = b.q;
            b.v0 = b.v;
            b.w0 = b.w;
        }
        let mut is_b = vec![false; nrow];
        for &r in &if_row {
            is_b[r] = true;
        }
        let k = self.assemble(h, 0.0, 0.0, None);
        let kc = self.assemble(h, 0.0, 1.0, None);
        // Interior operator, shifted by (2 pi 0.1 Hz)^2 M so floating parts stay definite.
        let sigma = (2.0 * std::f64::consts::PI * 0.1f64).powi(2);
        let kii = self.assemble(h, sigma, 0.0, Some(&is_b));
        let kii = self.factor_matrix(kii);
        let mut solves = 0u64;
        // Constraint modes: unit interface motion, interior statically condensed.
        let mut t: Vec<Vec<f64>> = Vec::with_capacity(nb + self.config.cb_modes);
        for &rb in &if_row {
            let mut e = vec![0.0; nrow];
            e[rb] = 1.0;
            let mut x: Vec<f64> = k.mul(&e).iter().map(|v| -v).collect();
            for (r, xb) in x.iter_mut().enumerate() {
                if is_b[r] {
                    *xb = 0.0;
                }
            }
            kii.solve(&mut x);
            solves += 1;
            for (r, xb) in x.iter_mut().enumerate() {
                if is_b[r] {
                    *xb = 0.0;
                }
            }
            x[rb] = 1.0;
            t.push(x);
        }
        // Fixed-interface modes: shift-inverted subspace iteration with Rayleigh-Ritz.
        let nk = self.config.cb_modes;
        let p = nk + 8;
        let mut x: Vec<Vec<f64>> = (0..p)
            .map(|j| (0..nrow).map(|r| if is_b[r] { 0.0 } else { ((r * 7919 + j * 104729) % 1000) as f64 / 1000.0 - 0.5 }).collect())
            .collect();
        let mut prev = vec![0.0; nk];
        let mut omega = Vec::new();
        for _ in 0..60 {
            let y: Vec<Vec<f64>> = x
                .iter()
                .map(|xc| {
                    let mut m = self.ext_mass_mul(xc);
                    for (r, v) in m.iter_mut().enumerate() {
                        if is_b[r] {
                            *v = 0.0;
                        }
                    }
                    kii.solve(&mut m);
                    m
                })
                .collect();
            solves += p as u64;
            // M-orthonormalise (modified Gram-Schmidt, twice); a collapsed vector is
            // replaced by a fresh pseudo-random one.
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            let mut y = y;
            for j in 0..p {
                for _ in 0..2 {
                    for i in 0..j {
                        let mi = self.ext_mass_mul(&y[i]);
                        let c = dot(&y[j], &mi);
                        let yi = y[i].clone();
                        y[j].iter_mut().zip(&yi).for_each(|(a, b)| *a -= c * b);
                    }
                }
                let n2 = dot(&y[j], &self.ext_mass_mul(&y[j]));
                if !(n2 > 1e-24) {
                    y[j] = (0..nrow).map(|r| if is_b[r] { 0.0 } else { ((r * 31 + j * 977) % 101) as f64 / 101.0 - 0.5 }).collect();
                    let n2 = dot(&y[j], &self.ext_mass_mul(&y[j]));
                    y[j].iter_mut().for_each(|a| *a /= n2.sqrt());
                } else {
                    y[j].iter_mut().for_each(|a| *a /= n2.sqrt());
                }
            }
            let ky: Vec<Vec<f64>> = y.iter().map(|c| k.mul(c)).collect();
            let a = Mat::<f64>::from_fn(p, p, |i, j| 0.5 * (dot(&y[i], &ky[j]) + dot(&y[j], &ky[i])));
            let (lam, w) = sym_eig(&a);
            let mut order: Vec<usize> = (0..p).collect();
            order.sort_by(|&i, &j| lam[i].total_cmp(&lam[j]));
            x = order
                .iter()
                .map(|&c| (0..nrow).map(|r| (0..p).map(|i| y[i][r] * w[(i, c)]).sum()).collect())
                .collect();
            omega = order.iter().map(|&c| lam[c]).collect::<Vec<f64>>();
            let change = (0..nk).map(|i| ((omega[i] - prev[i]) / omega[i].abs().max(1e-300)).abs()).fold(0.0, f64::max);
            prev = omega[..nk].to_vec();
            if change < 1e-8 {
                break;
            }
        }
        t.extend(x.into_iter().take(nk));
        let nr = t.len();
        // Reduced matrices.
        let kt: Vec<Vec<f64>> = t.iter().map(|c| k.mul(c)).collect();
        let kct: Vec<Vec<f64>> = t.iter().map(|c| kc.mul(c)).collect();
        let mt: Vec<Vec<f64>> = t.iter().map(|c| self.ext_mass_mul(c)).collect();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let kr = Mat::<f64>::from_fn(nr, nr, |i, j| 0.5 * (dot(&t[i], &kt[j]) + dot(&t[j], &kt[i])));
        let cr = Mat::<f64>::from_fn(nr, nr, |i, j| {
            0.5 * (dot(&t[i], &kct[j]) + dot(&t[j], &kct[i])) - kr[(i, j)]
        });
        let mr = Mat::<f64>::from_fn(nr, nr, |i, j| 0.5 * (dot(&t[i], &mt[j]) + dot(&t[j], &mt[i])));
        let now = self.w.time;
        let fe_ref = self.ext_loads_at(now);
        let f = self.ext_forces(h);
        let f0: Vec<f64> = t.iter().map(|c| dot(c, &f)).collect();
        // Reduced initial state: q = 0, M_r qd = T^T M v, M_r qdd = F - C_r qd.
        let mut v = vec![0.0; nrow];
        for (pp, &i) in bodies.iter().enumerate() {
            let b = &self.w.bodies[i];
            for c in 0..3 {
                v[6 * pp + c] = b.v[c];
                v[6 * pp + 3 + c] = b.w[c];
            }
        }
        let mv = self.ext_mass_mul(&v);
        let mr_inv = mr.partial_piv_lu().inverse();
        let tmv = Mat::<f64>::from_fn(nr, 1, |i, _| dot(&t[i], &mv));
        let qd = &mr_inv * &tmv;
        let crqd = &cr * &qd;
        let fq = Mat::<f64>::from_fn(nr, 1, |i, _| f0[i] - if i < nb { lam0[i] } else { 0.0 } - crqd[(i, 0)]);
        let qdd = &mr_inv * &fq;
        let keff = Mat::<f64>::from_fn(nr, nr, |i, j| kr[(i, j)] + 2.0 / h * cr[(i, j)] + 4.0 / (h * h) * mr[(i, j)]);
        let keff_inv = keff.partial_piv_lu().inverse();
        let s: Vec<f64> = (0..nb * nb).map(|e| (2.0 / h) * 0.5 * (keff_inv[(e / nb, e % nb)] + keff_inv[(e % nb, e / nb)])).collect();
        self.stats.solves += solves;
        self.stats.solve_flops += solves as f64 * kii.solve_flops();
        self.stats.cb_reductions += 1;
        self.stats.cb_reduced_max = self.stats.cb_reduced_max.max(nr);
        let reference: Vec<(Vec3, Quat)> = bodies.iter().map(|&i| (self.w.bodies[i].x, self.w.bodies[i].q)).collect();
        let cb = Cb {
            nb,
            t,
            mr,
            cr,
            keff_inv,
            f0,
            fe_ref,
            q: vec![0.0; nr],
            qd: (0..nr).map(|i| qd[(i, 0)]).collect(),
            qdd: (0..nr).map(|i| qdd[(i, 0)]).collect(),
            reference,
            omega_min: omega.first().copied().unwrap_or(0.0).max(0.0).sqrt(),
            omega_max: omega.get(nk.saturating_sub(1)).copied().unwrap_or(0.0).max(0.0).sqrt(),
        };
        self.stats.cb_mode_omega = [cb.omega_min, cb.omega_max];
        self.solver.as_mut().unwrap().cb = Some(cb);
        self.set_interface_operator(s, hot_mass);
        self.stats.t_factor += t0.elapsed().as_secs_f64();
    }

    /// Put the Craig-Bampton state into the exterior's bodies (ring copies swapped in).
    fn cb_reconstruct(&mut self) {
        let sv = self.solver.as_ref().unwrap();
        let cb = sv.cb.as_ref().unwrap();
        let nr = cb.t.len();
        let nrow = sv.mat.n;
        let mut u = vec![0.0; nrow];
        let mut ud = vec![0.0; nrow];
        let mut udd = vec![0.0; nrow];
        for c in 0..nr {
            let col = &cb.t[c];
            let (a, b, d) = (cb.q[c], cb.qd[c], cb.qdd[c]);
            for r in 0..nrow {
                u[r] += col[r] * a;
                ud[r] += col[r] * b;
                udd[r] += col[r] * d;
            }
        }
        for (p, &i) in sv.bodies.iter().enumerate() {
            let (xr, qr) = cb.reference[p];
            let b = &mut self.w.bodies[i];
            let g = |v: &[f64], o: usize| Vec3::new(v[6 * p + o], v[6 * p + o + 1], v[6 * p + o + 2]);
            if !b.fixed_lin {
                b.x = xr + g(&u, 0);
                b.v = g(&ud, 0);
                b.a0 = g(&udd, 0);
            }
            if !b.fixed_ang {
                b.q = (Quat::from_rotation_vector(g(&u, 3)) * qr).normalized();
                b.w = g(&ud, 3);
                b.alpha0 = g(&udd, 3);
            }
        }
        self.stats.solve_flops += 3.0 * (nr * nrow) as f64;
    }

    /// One Craig-Bampton Newmark step (the free problem, interface force held at
    /// `lam0`), or with `link` its correction by `link` (the multiplier's change).
    fn cb_step(&mut self, h: f64, lam0: &[f64], link: Option<&[f64]>) {
        let df = if link.is_none() {
            let t = self.w.time + h;
            let fe = self.ext_loads_at(t);
            let cb = self.solver.as_ref().unwrap().cb.as_ref().unwrap();
            cb.t.iter().map(|c| c.iter().zip(fe.iter().zip(&cb.fe_ref)).map(|(a, (x, y))| a * (x - y)).sum::<f64>()).collect::<Vec<f64>>()
        } else {
            Vec::new()
        };
        let sv = self.solver.as_mut().unwrap();
        let cb = sv.cb.as_mut().unwrap();
        let nr = cb.t.len();
        let nb = cb.nb;
        let dot_row = |m: &Mat<f64>, i: usize, x: &[f64]| (0..nr).map(|j| m[(i, j)] * x[j]).sum::<f64>();
        match link {
            None => {
                let (q, qd, qdd) = (cb.q.clone(), cb.qd.clone(), cb.qdd.clone());
                let a: Vec<f64> = (0..nr).map(|i| 4.0 / (h * h) * (q[i] + h * qd[i]) + qdd[i]).collect();
                let c: Vec<f64> = (0..nr).map(|i| 2.0 / h * q[i] + qd[i]).collect();
                let rhs: Vec<f64> = (0..nr)
                    .map(|i| cb.f0[i] + df[i] - if i < nb { lam0[i] } else { 0.0 } + dot_row(&cb.mr, i, &a) + dot_row(&cb.cr, i, &c))
                    .collect();
                let q1: Vec<f64> = (0..nr).map(|i| dot_row(&cb.keff_inv, i, &rhs)).collect();
                cb.qd = (0..nr).map(|i| 2.0 / h * (q1[i] - q[i]) - qd[i]).collect();
                cb.qdd = (0..nr).map(|i| 4.0 / (h * h) * (q1[i] - q[i] - h * qd[i]) - qdd[i]).collect();
                cb.q = q1;
            }
            Some(dl) => {
                let rhs: Vec<f64> = (0..nr).map(|i| if i < nb { -dl[i] } else { 0.0 }).collect();
                let dq: Vec<f64> = (0..nr).map(|i| dot_row(&cb.keff_inv, i, &rhs)).collect();
                for i in 0..nr {
                    cb.q[i] += dq[i];
                    cb.qd[i] += 2.0 / h * dq[i];
                    cb.qdd[i] += 4.0 / (h * h) * dq[i];
                }
            }
        }
        self.stats.solve_flops += 3.0 * (nr * nr) as f64;
        self.cb_reconstruct();
    }

    // ------------------------------------------------------------------ hot zone

    /// The hot zone's forces at the current state (`step_explicit`'s law and order) into
    /// the active bodies' `force`/`torque`. With `commit` the substep's history advances
    /// (bond laws, contact memory, ledgers, events); without, nothing is written: the
    /// look-ahead at a coarse step's end, identical to the next substep's own pass.
    /// Returns whether a bond disconnected.
    fn hot_forces(&mut self, dt: f64, commit: bool) -> bool {
        let w = &mut self.w;
        if commit {
            w.process_events();
        }
        w.external_loads(w.time);
        let g = w.gravity;
        let n = w.bodies.len();
        for i in 0..n {
            if self.active[i] {
                let s = self.share[i];
                let b = &mut w.bodies[i];
                b.force = b.ext_f * s + g * b.mass;
                b.torque = b.ext_t * s;
            }
        }
        let saved_offsets = if commit { None } else { Some(w.offsets.clone()) };
        let found = w.detect(&|_, _| 0.0);
        let eff = w.pair_offsets(&found, dt);
        let restitution = w.scene.sim.contact_restitution;
        let mut engaged: HashMap<(usize, usize), usize> = HashMap::new();
        for (d, &e) in found.iter().zip(&eff) {
            if let Some(b) = d.b {
                if matches!(w.bodies[b].kind, Kind::Chunk(..)) && e > 0.0 {
                    *engaged.entry((d.a, b)).or_default() += 1;
                }
            }
        }
        for (d, &depth) in found.iter().zip(&eff) {
            if depth <= 0.0 {
                continue;
            }
            let pair_chunks = d.b.is_some_and(|b| matches!(w.bodies[b].kind, Kind::Chunk(..)));
            let (k, points) = if pair_chunks {
                let e = engaged[&(d.a, d.b.unwrap())];
                (d.k / e.max(FACE_PAIR_POINTS) as f64, e)
            } else if d.b.is_some() && w.bodies[d.b.unwrap()].sphere.is_none() {
                (d.k / d.points.max(FACE_PAIR_POINTS) as f64, d.points)
            } else {
                (d.k, d.points)
            };
            let va = w.bodies[d.a].point_velocity(d.point);
            let vb = d.b.map_or(Vec3::ZERO, |b| w.bodies[b].point_velocity(d.point));
            let (f, _, _) = penalty_force(k, d.m_red, restitution, d.mu, depth, d.n, va - vb, dt, points);
            if commit {
                w.stats.contact_evaluations += 1;
                self.stats.hot_contact_evals += 1;
            }
            let xa = w.bodies[d.a].x;
            w.bodies[d.a].force += f;
            w.bodies[d.a].torque += (d.point - xa).cross(f);
            if let Some(b) = d.b {
                let xb = w.bodies[b].x;
                w.bodies[b].force -= f;
                w.bodies[b].torque -= (d.point - xb).cross(f);
            }
        }
        if let Some(o) = saved_offsets {
            w.offsets = o;
        }
        let mut split = false;
        let fracture = w.scene.sim.fracture;
        let mut evals = 0;
        for &k in &self.hot_bonds {
            if !w.bonds[k].active {
                continue;
            }
            let ev = w.bond_eval(k, dt, fracture, None);
            evals += 1;
            if commit {
                w.stats.bond_evaluations += 1;
                split |= w.commit_bond(k, &ev, dt);
            }
            let (a, b) = (w.bonds[k].a, w.bonds[k].b);
            w.bodies[a].force += ev.loads[0];
            w.bodies[a].torque += ev.loads[1];
            w.bodies[b].force += ev.loads[2];
            w.bodies[b].torque += ev.loads[3];
        }
        if commit {
            self.stats.hot_bond_evals += evals;
            self.stats.full_bond_evals += w.bonds.iter().filter(|b| b.active).count() as u64;
        } else {
            self.stats.lookahead_bond_evals += evals;
        }
        split
    }

    /// The ring's hot copies' velocities at the current (integer) time without a new
    /// interface force: `v_{n-1/2} + dt/2 F_n / m` (and the angular analogue).
    fn ring_velocity(&self, sv: &ExtSolver, dt: f64) -> Vec<f64> {
        let mut va = vec![0.0; sv.n_if()];
        for (k, &i) in self.ring.iter().enumerate() {
            let b = &self.w.bodies[i];
            if let Some(i0) = sv.if_lin[k] {
                let v = b.v + b.force * (0.5 * dt / b.mass);
                for c in 0..3 {
                    va[i0 + c] = v[c];
                }
            }
            if let Some(i0) = sv.if_ang[k] {
                let l = b.world_inertia(b.q) * b.w + b.torque * (0.5 * dt);
                let om = b.world_inertia(b.q).inverse().expect("inertia") * l;
                for c in 0..3 {
                    va[i0 + c] = om[c];
                }
            }
        }
        va
    }

    /// Add the interface forces to the ring's hot copies, then the substep's kick and
    /// drift (`step_explicit`'s update; a body whose velocity is at the integer time
    /// after a repartition is first moved back half a kick).
    fn kick_drift(&mut self, dt: f64, sv: Option<&ExtSolver>, lambda: &[f64]) {
        if let Some(sv) = sv {
            for (k, &i) in self.ring.iter().enumerate() {
                let b = &mut self.w.bodies[i];
                if let Some(i0) = sv.if_lin[k] {
                    b.force += Vec3::new(lambda[i0], lambda[i0 + 1], lambda[i0 + 2]);
                }
                if let Some(i0) = sv.if_ang[k] {
                    b.torque += Vec3::new(lambda[i0], lambda[i0 + 1], lambda[i0 + 2]);
                }
            }
        }
        let n = self.w.bodies.len();
        let mut updates = 0;
        for i in 0..n {
            if !self.active[i] {
                continue;
            }
            let rebase = std::mem::replace(&mut self.rebase[i], false);
            let b = &mut self.w.bodies[i];
            if rebase {
                if !b.fixed_lin {
                    b.v -= b.force * (0.5 * dt / b.mass);
                }
                if !b.fixed_ang {
                    let l = b.world_inertia(b.q) * b.w - b.torque * (0.5 * dt);
                    b.w = b.world_inertia(b.q).inverse().expect("inertia") * l;
                }
            }
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
            updates += 1;
        }
        self.stats.hot_body_updates += updates;
        self.stats.full_body_updates += self.w.bodies.iter().filter(|b| b.free() && b.x.x < 1e8).count() as u64;
        self.w.time += dt;
        self.w.stats.steps += 1;
        self.stats.fine_steps += 1;
    }

    /// The interface force at a repartition (no multiplier carried over): the one that
    /// gives each ring chunk's hot copy its exterior copy's acceleration.
    fn matching_lambda(&self, sv: &ExtSolver) -> Vec<f64> {
        let mut lam = vec![0.0; sv.n_if()];
        for (k, &i) in self.ring.iter().enumerate() {
            let (b, e) = (&self.w.bodies[i], &self.ext_copy[k]);
            if let Some(i0) = sv.if_lin[k] {
                let l = e.a0 * b.mass - b.force;
                for c in 0..3 {
                    lam[i0 + c] = l[c];
                }
            }
            if let Some(i0) = sv.if_ang[k] {
                let l = b.world_inertia(b.q) * e.alpha0 - b.torque;
                for c in 0..3 {
                    lam[i0 + c] = l[c];
                }
            }
        }
        lam
    }

    // ------------------------------------------------------------------ coarse step

    /// One coarse step of `m` substeps (Gravouil-Combescure). Returns the bodies
    /// verification missed.
    fn advance(&mut self, m: usize) -> Vec<usize> {
        let dt = self.w.law_dt * self.config.fine_scale;
        let h = dt * m as f64;
        let t_n = self.w.time;
        let mut record = CoarseRecord {
            t: t_n,
            core: self.core.iter().filter(|&&c| c).count(),
            hot: self.hot.iter().filter(|&&c| c).count(),
            ring: self.ring.len(),
            hot_bonds: self.hot_bonds.iter().filter(|&&k| self.w.bonds[k].active).count(),
            active_bonds: self.w.bonds.iter().filter(|b| b.active).count(),
            substeps: m,
            ..CoarseRecord::default()
        };
        let have_ext = (0..self.w.bodies.len()).any(|i| self.free_chunk(i) && !self.hot[i]);
        let hot_mass: Vec<(f64, Mat3)> = self
            .ring
            .iter()
            .map(|&i| {
                let b = &self.w.bodies[i];
                (b.mass, b.world_inertia(b.q))
            })
            .collect();
        // Substep 0's forces at t_n.
        let t0 = Instant::now();
        let mut split = self.hot_forces(dt, true);
        self.stats.t_hot += t0.elapsed().as_secs_f64();
        // The multiplier at t_n: carried from the last step's end, or (after a
        // repartition) the one matching the copies' accelerations.
        if self.solver.as_ref().is_some_and(|sv| sv.h != h) {
            // A shorter last step: the operators are the coarse step's.
            self.solver = None;
        }
        if have_ext && self.solver.is_none() {
            self.build_solver();
            self.solver.as_mut().unwrap().h = h;
        }
        let lam0: Vec<f64> = match &self.solver {
            Some(sv) if have_ext && sv.n_if() > 0 => match self.lam_next.take() {
                Some(l) if l.len() == sv.n_if() => l,
                _ => self.matching_lambda(sv),
            },
            _ => Vec::new(),
        };
        // Exterior: the free problem over the coarse step, the interface force held at
        // `lambda_0` (the link then carries its change).
        let mut ctx = None;
        let mut ext_damage0 = Vec::new();
        if have_ext {
            self.swap_ring();
            // The ring's exterior copies at t_n.
            let ring_start: Vec<(Vec3, Quat, Vec3, Vec3)> = self
                .ring
                .iter()
                .map(|&i| {
                    let b = &self.w.bodies[i];
                    (b.x, b.q, b.v, b.w)
                })
                .collect();
            if self.config.exterior == Exterior::CraigBampton {
                if self.solver.as_ref().unwrap().cb.is_none() {
                    self.build_cb(h, &hot_mass, &lam0);
                }
                self.cb_step(h, &lam0, None);
            } else {
            let bodies = self.solver.as_ref().unwrap().bodies.clone();
            for &i in &bodies {
                let b = &mut self.w.bodies[i];
                b.x0 = b.x;
                b.q0 = b.q;
                b.v0 = b.v;
                b.w0 = b.w;
                b.dx = b.v0 * h + b.a0 * (0.5 * h * h);
                b.dth = b.w0 * h + b.alpha0 * (0.5 * h * h);
                Self::set_trial_nm(b, h);
            }
            self.w.external_loads(t_n + h);
            ext_damage0 = self
                .solver
                .as_ref()
                .unwrap()
                .bonds
                .iter()
                .map(|&k| self.w.bonds[k].rt.joint.damage + self.w.bonds[k].rt.joint.crush)
                .collect();
            let mut load = vec![[0.0; 6]; self.ring.len()];
            {
                let sv = self.solver.as_ref().unwrap();
                for (d, &(k, c)) in sv.if_dofs.iter().enumerate() {
                    load[k][c] = lam0[d];
                }
            }
            let (it, rel) = self.ext_solve(h, &load, &hot_mass);
            if !self.solver.as_ref().unwrap().factored {
                // Converged on the predictor: the link still needs the operator.
                self.factor(h, &hot_mass);
            }
            record.newton = it;
            record.residual = rel;
            }
            let sv = self.solver.as_ref().unwrap();
            let n_if = sv.n_if();
            let (if_lin, if_ang) = (sv.if_lin.clone(), sv.if_ang.clone());
            let mut vb0 = vec![0.0; n_if];
            let mut vbf = vec![0.0; n_if];
            let mut xb0 = Vec::new();
            for (k, &i) in self.ring.iter().enumerate() {
                let b = &self.w.bodies[i];
                let (x0, q0, v0, w0) = ring_start[k];
                xb0.push((x0, q0));
                if let Some(i0) = if_lin[k] {
                    for c in 0..3 {
                        vb0[i0 + c] = v0[c];
                        vbf[i0 + c] = b.v[c];
                    }
                }
                if let Some(i0) = if_ang[k] {
                    for c in 0..3 {
                        vb0[i0 + c] = w0[c];
                        vbf[i0 + c] = b.w[c];
                    }
                }
            }
            // GC's constant operator: the hot side follows the exterior's interpolated free
            // velocity minus its full link response, `- S (lambda - lambda_0)`; the
            // `S lambda_0` part enters unweighted.
            let sl0 = if n_if > 0 { sv.s_mul(&lam0) } else { Vec::new() };
            self.swap_ring();
            // Optional bias closing the copies' position drift over the coarse step.
            let mut bias = vec![0.0; n_if];
            if self.config.drift_correction {
                for (k, &i) in self.ring.iter().enumerate() {
                    let b = &self.w.bodies[i];
                    let (xe, qe) = xb0[k];
                    let ex = (b.x - xe) / h;
                    let er = (b.q * qe.conjugate()).to_rotation_vector() / h;
                    if let Some(i0) = if_lin[k] {
                        for c in 0..3 {
                            bias[i0 + c] = ex[c];
                        }
                    }
                    if let Some(i0) = if_ang[k] {
                        for c in 0..3 {
                            bias[i0 + c] = er[c];
                        }
                    }
                }
            }
            for d in 0..n_if {
                bias[d] -= sl0[d];
            }
            ctx = Some((vb0, vbf, bias));
        }
        // Substeps.
        let t0 = Instant::now();
        let solver = self.solver.take();
        let iface = match (&ctx, &solver) {
            (Some(c), Some(sv)) if sv.n_if() > 0 => Some((c, sv)),
            _ => None,
        };
        for k in 0..m {
            if k > 0 {
                split |= self.hot_forces(dt, true);
            }
            let lam = match iface {
                Some(_) if k == 0 => lam0.clone(),
                Some(((vb0, vbf, bias), sv)) => {
                    let wk = k as f64 / m as f64;
                    let va = self.ring_velocity(sv, dt);
                    let b: Vec<f64> = (0..sv.n_if())
                        .map(|d| (1.0 - wk) * vb0[d] + wk * vbf[d] - bias[d] - va[d])
                        .collect();
                    // GC's condensed operator is the same at every substep (the exterior's
                    // full end-of-step compliance); weighting it by k/m is unstable.
                    sv.solve_interface(0.5 * dt, 1.0, &b)
                }
                None => Vec::new(),
            };
            if split {
                self.w.rebuild_components();
                split = false;
            }
            self.kick_drift(dt, iface.map(|(_, sv)| sv), &lam);
            self.w.record_probes();
            if !self.w.bodies.iter().all(|b| b.x.is_finite() && b.v.is_finite()) {
                self.diverged = true;
                break;
            }
        }
        // Look-ahead: the next substep's forces at t_{n+1}, and the end multiplier.
        self.hot_forces(dt, false);
        let mut lam_end = Vec::new();
        if let Some(((_, vbf, bias), sv)) = iface {
            let va = self.ring_velocity(sv, dt);
            let b: Vec<f64> = (0..sv.n_if()).map(|d| vbf[d] - bias[d] - va[d]).collect();
            lam_end = sv.solve_interface(0.5 * dt, 1.0, &b);
        }
        self.solver = solver;
        self.stats.t_hot += t0.elapsed().as_secs_f64();
        let mut misses = Vec::new();
        // Exterior: link correction with the end multiplier, commit.
        if ctx.is_some() {
            self.swap_ring();
            let n_if = self.solver.as_ref().unwrap().n_if();
            let mut lam_new = vec![[0.0; 6]; self.ring.len()];
            {
                let sv = self.solver.as_ref().unwrap();
                for (d, &(k, c)) in sv.if_dofs.iter().enumerate() {
                    lam_new[k][c] = lam_end[d];
                }
            }
            let cb = self.config.exterior == Exterior::CraigBampton;
            if cb {
                let dl: Vec<f64> = (0..n_if).map(|d| lam_end[d] - lam0[d]).collect();
                self.cb_step(h, &lam0, Some(&dl));
            }
            let sv = self.solver.take().unwrap();
            if n_if > 0 && !cb {
                let mut rhs = vec![0.0; sv.mat.n];
                for d in 0..n_if {
                    rhs[sv.if_row[d]] = -(lam_end[d] - lam0[d]);
                }
                sv.mat.solve(&mut rhs);
                self.stats.solves += 1;
                self.stats.solve_flops += sv.mat.solve_flops();
                for (p, &i) in sv.bodies.iter().enumerate() {
                    let b = &mut self.w.bodies[i];
                    b.dx += Vec3::new(rhs[6 * p], rhs[6 * p + 1], rhs[6 * p + 2]);
                    b.dth += Vec3::new(rhs[6 * p + 3], rhs[6 * p + 4], rhs[6 * p + 5]);
                    Self::set_trial_nm(b, h);
                }
            }
            // Commit the exterior's bonds at the accepted state (Craig-Bampton, linear:
            // only their utilisation, for the triggers), then Newmark's accelerations.
            let t_end = self.w.time;
            self.w.time = t_n;
            let fracture = self.w.scene.sim.fracture;
            let mut ext_split = false;
            let mut u_max: f64 = 0.0;
            for (q, &k) in sv.bonds.iter().enumerate() {
                if !self.w.bonds[k].active {
                    continue;
                }
                if cb {
                    let u = self.w.bond_eval(k, h, false, None).resp.state.utilization;
                    self.w.bonds[k].rt.joint.utilization = u;
                    u_max = u_max.max(u);
                    if u >= 1.0 {
                        // Over capacity: the linear exterior would have cracked here.
                        self.stats.exterior_cracks += 1;
                        record.ext_cracks += 1;
                        if self.config.verify == Verify::Rollback {
                            self.stats.misses_damage += 1;
                            misses.extend([self.w.bonds[k].a, self.w.bonds[k].b]);
                        }
                    }
                    if u >= self.config.u_hot {
                        self.stats.misses_utilization += 1;
                        misses.extend([self.w.bonds[k].a, self.w.bonds[k].b]);
                    }
                    continue;
                }
                let ev = self.w.bond_eval(k, h, fracture, None);
                ext_split |= self.w.commit_bond(k, &ev, h);
                let j = &self.w.bonds[k].rt.joint;
                u_max = u_max.max(j.utilization);
                if j.damage + j.crush > ext_damage0[q] {
                    self.stats.exterior_cracks += 1;
                    record.ext_cracks += 1;
                    if self.config.verify == Verify::Rollback {
                        self.stats.misses_damage += 1;
                        misses.extend([self.w.bonds[k].a, self.w.bonds[k].b]);
                    }
                }
                if j.utilization >= self.config.u_hot {
                    self.stats.misses_utilization += 1;
                    misses.extend([self.w.bonds[k].a, self.w.bonds[k].b]);
                }
            }
            self.stats.ext_bond_evals += sv.bonds.len() as u64;
            record.ext_u_max = u_max;
            record.ext_bonds = sv.bonds.len();
            record.ext_dofs = sv.bodies.len();
            self.w.time = t_end;
            if !cb {
                let c_m = 4.0 / (h * h);
                for &i in &sv.bodies {
                    let b = &mut self.w.bodies[i];
                    b.a0 = if b.fixed_lin { Vec3::ZERO } else { (b.dx - b.v0 * h) * c_m - b.a0 };
                    b.alpha0 = if b.fixed_ang { Vec3::ZERO } else { (b.dth - b.w0 * h) * c_m - b.alpha0 };
                }
            }
            self.solver = Some(sv);
            self.swap_ring();
            // Drift and velocity gap at the step's end (integer-time hot velocities).
            let sv = self.solver.take().unwrap();
            if sv.n_if() > 0 {
                let va = self.ring_velocity(&sv, dt);
                for (k, &i) in self.ring.iter().enumerate() {
                    let (b, e) = (&self.w.bodies[i], &self.ext_copy[k]);
                    record.drift = record.drift.max((b.x - e.x).norm());
                    if let Some(i0) = sv.if_lin[k] {
                        let l = Vec3::new(lam_end[i0], lam_end[i0 + 1], lam_end[i0 + 2]);
                        let v = Vec3::new(va[i0], va[i0 + 1], va[i0 + 2]) + l * (0.5 * dt / b.mass);
                        record.velocity_gap = record.velocity_gap.max((v - e.v).norm());
                    }
                }
            }
            // The hot copies' forces with the end multiplier (for merges at a repartition).
            for (k, &i) in self.ring.iter().enumerate() {
                let b = &self.w.bodies[i];
                let (mut f, mut t) = (b.force, b.torque);
                if let Some(i0) = sv.if_lin[k] {
                    f += Vec3::new(lam_end[i0], lam_end[i0 + 1], lam_end[i0 + 2]);
                }
                if let Some(i0) = sv.if_ang[k] {
                    t += Vec3::new(lam_end[i0], lam_end[i0 + 1], lam_end[i0 + 2]);
                }
                self.last_force[i] = (f, t);
            }
            self.solver = Some(sv);
            self.lam_prev2 = std::mem::replace(&mut self.lam_prev, lam_new);
            self.lam_next = Some(lam_end);
            if ext_split {
                self.w.rebuild_components();
            }
            if self.config.influence {
                self.influence_bounds();
            }
        }
        // Verification: contacts touching the exterior.
        let tv = Instant::now();
        self.w.detect_mask = None;
        let found = self.w.detect(&|_, _| 0.0);
        self.w.detect_mask = Some(self.active.clone());
        for d in &found {
            if d.depth <= 0.0 {
                continue;
            }
            for i in [Some(d.a), d.b].into_iter().flatten() {
                if self.free_chunk(i) && !self.hot[i] {
                    self.stats.misses_contact += 1;
                    misses.push(i);
                }
            }
        }
        self.stats.t_escalate += tv.elapsed().as_secs_f64();
        misses.retain(|&i| self.free_chunk(i));
        misses.sort_unstable();
        misses.dedup();
        record.misses = misses.len();
        record.factorizations = self.stats.factorizations as usize;
        if self.config.energy {
            let (k, st, p, mom) = self.energy();
            record.kinetic = k;
            record.stored = st;
            record.potential = p;
            record.momentum = mom.to_array();
            record.dissipated = self.w.bond_dissipation + self.w.damping_dissipation;
        }
        record.wall = self.started.map_or(0.0, |s| s.elapsed().as_secs_f64());
        self.stats.records.push(record);
        misses
    }

    /// Utilisation that the next coarse step's interface-force change can add to each
    /// exterior bond: `sum_c |du/dd_c| sum_i |(J K_eff^-1 C^T)_{c i}| dlambda_i` with
    /// `dlambda` bounded by `influence_growth` times the last change. The exterior
    /// response is Newmark's to a load step at the step's end.
    fn influence_bounds(&mut self) {
        let Some(sv) = self.solver.as_ref() else {
            return;
        };
        let n_if = sv.n_if();
        self.influence.iter_mut().for_each(|x| *x = 0.0);
        if n_if == 0 || sv.z.is_empty() {
            return;
        }
        let growth = self.config.influence_growth;
        let dl: Vec<f64> = sv
            .if_dofs
            .iter()
            .map(|&(k, c)| growth * (self.lam_prev[k][c] - self.lam_prev2[k][c]).abs())
            .collect();
        let gain = 1.0;
        self.swap_ring();
        let sv = self.solver.as_ref().unwrap();
        let fracture = self.w.scene.sim.fracture;
        let out: Vec<(usize, f64)> = crate::par::map_range(0..sv.bonds.len(), 32, |q| {
            let k = sv.bonds[q];
            let ev = self.w.bond_eval(k, 0.0, false, Some(0.0));
            let (ja, jb, _) = ev.jac.as_ref().unwrap();
            let b = &self.w.bonds[k];
            // Utilisation gradient by central differences in the local deformation.
            let cr = self.w.corotation(k);
            let d0 = cr.deformation();
            let model = b.rt.model();
            let mut grad = [0.0; 6];
            for c in 0..6 {
                let mut a = d0.to_array();
                let eps = 1e-3 * a.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-9);
                a[c] += eps;
                let up = model.evaluate(&b.rt.joint, &Local6::from_array(a), 0.0, false).state.utilization;
                a[c] -= 2.0 * eps;
                let dn = model.evaluate(&b.rt.joint, &Local6::from_array(a), 0.0, false).state.utilization;
                grad[c] = (up - dn) / (2.0 * eps);
            }
            let _ = fracture;
            // |J Z| dlambda per local component.
            let mut t = [0.0; 6];
            for d in 0..n_if {
                if dl[d] == 0.0 {
                    continue;
                }
                let z = &sv.z[d];
                let mut dd = [0.0; 6];
                for (body, j) in [(b.a, ja), (b.b, jb)] {
                    let p = sv.pos[body];
                    if p == usize::MAX {
                        continue;
                    }
                    for r in 0..6 {
                        let mut s = 0.0;
                        for c in 0..6 {
                            s += j[r][c] * z[6 * p + c];
                        }
                        dd[r] += s;
                    }
                }
                for r in 0..6 {
                    t[r] += dd[r].abs() * dl[d];
                }
            }
            let bound: f64 = (0..6).map(|c| grad[c].abs() * t[c]).sum::<f64>() * gain;
            (k, bound)
        });
        self.swap_ring();
        for (k, b) in out {
            self.influence[k] = b;
        }
    }

    /// Kinetic energy, energy stored in the bonds (each zone's bonds with its copies),
    /// gravity's potential and the linear momentum of every body and copy.
    fn energy(&mut self) -> (f64, f64, f64, Vec3) {
        let g = self.w.gravity;
        let body_terms = |b: &Body| {
            let ke = 0.5 * b.mass * b.v.norm2() + 0.5 * b.w.dot(b.world_inertia(b.q) * b.w);
            (ke, -b.mass * g.dot(b.x), b.v * b.mass)
        };
        let (mut ke, mut pe, mut mom) = (0.0, 0.0, Vec3::ZERO);
        for b in self.w.bodies.iter().chain(self.ext_copy.iter()) {
            if b.free() && b.x.x < 1e8 {
                let (k, p, m) = body_terms(b);
                ke += k;
                pe += p;
                mom += m;
            }
        }
        let mut stored = 0.0;
        for k in 0..self.w.bonds.len() {
            if self.w.bonds[k].active && self.bond_hot[k] {
                stored += self.w.bond_eval(k, 0.0, false, None).resp.stored;
            }
        }
        self.swap_ring();
        for k in 0..self.w.bonds.len() {
            if self.w.bonds[k].active && !self.bond_hot[k] {
                stored += self.w.bond_eval(k, 0.0, false, None).resp.stored;
            }
        }
        self.swap_ring();
        (ke, stored, pe, mom)
    }

    /// Run to the end.
    pub fn run(&mut self) -> MtsReport {
        let start = Instant::now();
        self.started = Some(start);
        let end = self.config.end_time.unwrap_or(self.w.scene.sim.duration);
        self.w.next_sample = self.w.time;
        self.w.record_probes();
        let m_full = self.config.ratio.max(1);
        while self.w.time < end - 1e-12 && !self.diverged {
            let dt = self.w.law_dt * self.config.fine_scale;
            let remaining = ((end - self.w.time) / dt - 1e-9).ceil().max(1.0) as usize;
            let m = m_full.min(remaining);
            let h = dt * m as f64;
            let snapshot = (self.config.verify == Verify::Rollback).then(|| self.clone());
            let mut attempt = 0;
            loop {
                // Utilisation rates are per coarse step: the last accepted step's.
                self.escalate(h);
                let misses = self.advance(m);
                if misses.is_empty() || self.config.verify == Verify::Count || attempt >= self.config.max_rollbacks || self.diverged {
                    break;
                }
                let stats = std::mem::take(&mut self.stats);
                *self = snapshot.clone().unwrap();
                self.stats = stats;
                self.stats.rollbacks += 1;
                if let Some(r) = self.stats.records.last_mut() {
                    r.rollbacks += 1;
                }
                for i in misses {
                    self.forced[i] = true;
                }
                attempt += 1;
            }
            self.prev_u = self.w.bonds.iter().map(|b| b.rt.joint.utilization).collect();
            self.stats.coarse_steps += 1;
        }
        if self.diverged {
            eprintln!("MTS world: state diverged at t = {:.6e}", self.w.time);
        }
        self.stats.t_total = start.elapsed().as_secs_f64();
        self.report()
    }

    pub fn report(&self) -> MtsReport {
        let mut outcome = self.w.report();
        outcome.stats.wall_seconds = self.stats.t_total;
        let main = self.main_mask();
        let mut detached = Vec::new();
        let mut speed = 0.0;
        for (i, b) in self.w.bodies.iter().enumerate() {
            if let Kind::Chunk(s, c) = b.kind {
                if b.x.x < 1e8 && !main[i] {
                    detached.push((s, c));
                    speed += b.v.norm();
                }
            }
        }
        let debris_speed = if detached.is_empty() { 0.0 } else { speed / detached.len() as f64 };
        MtsReport { outcome, detached, debris_speed, diverged: self.diverged, stats: self.stats.clone() }
    }
}

/// Detached chunks and their mean speed of a plain AVBD world (the explicit replica).
pub fn detached_set(w: &AvbdWorld) -> (Vec<(usize, usize)>, f64) {
    let mut comp_mass: HashMap<usize, f64> = HashMap::new();
    for (i, b) in w.bodies.iter().enumerate() {
        if matches!(b.kind, Kind::Chunk(..)) && b.x.x < 1e8 {
            *comp_mass.entry(w.comp[i]).or_default() += b.mass;
        }
    }
    let has_supports = w.comp_anchored.iter().any(|&a| a);
    let heaviest = comp_mass.iter().max_by(|a, b| a.1.total_cmp(b.1)).map(|(&c, _)| c);
    let mut out = Vec::new();
    let mut speed = 0.0;
    for (i, b) in w.bodies.iter().enumerate() {
        if let Kind::Chunk(s, c) = b.kind {
            let main = if has_supports { w.comp_anchored[w.comp[i]] } else { Some(w.comp[i]) == heaviest };
            if b.x.x < 1e8 && !main {
                out.push((s, c));
                speed += b.v.norm();
            }
        }
    }
    let n = out.len().max(1) as f64;
    (out, speed / n)
}

/// The explicit replica's per-bond utilisation history: for every bond its peak
/// utilisation and the first time it reached each of `levels` (or none), with the bond's
/// distance from the first impactor's start position. A diagnostic for how large a hot
/// zone the truth needs.
pub fn utilisation_map(world: &World, end: f64, levels: &[f64]) -> Vec<(f64, Vec<Option<f64>>, f64)> {
    let mut w = AvbdWorld::from_world(world, AvbdConfig { integrator: Integrator::Explicit, ..AvbdConfig::default() });
    let origin = w.impactor_bodies.first().map(|&i| w.bodies[i].x).unwrap_or(Vec3::ZERO);
    let nb = w.bonds.len();
    let mut peak = vec![0.0f64; nb];
    let mut first = vec![vec![None; levels.len()]; nb];
    let dist: Vec<f64> = w
        .bonds
        .iter()
        .map(|b| ((w.bodies[b.a].x + w.bodies[b.b].x) * 0.5 - origin).norm())
        .collect();
    while w.time < end - 1e-12 {
        let dt = w.law_dt;
        w.step_explicit(dt);
        for k in 0..nb {
            let u = w.bonds[k].rt.joint.utilization;
            peak[k] = peak[k].max(u);
            for (l, &lv) in levels.iter().enumerate() {
                if first[k][l].is_none() && u >= lv {
                    first[k][l] = Some(w.time);
                }
            }
        }
    }
    (0..nb).map(|k| (peak[k], first[k].clone(), dist[k])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::cantilever;

    fn b2(end: f64) -> World {
        let mut scene = cantilever(20, "b2_cantilever");
        scene.sim.methods = crate::scene::Methods::established();
        scene.sim.duration = end;
        World::new(&scene.with_derived())
    }

    fn tip(probes: &HashMap<String, Vec<[f64; 2]>>) -> Vec<[f64; 2]> {
        probes["tip"].clone()
    }

    /// With everything in the hot zone the coarse-step loop is the explicit replica, bit
    /// for bit.
    #[test]
    fn all_hot_is_the_explicit_replica() {
        let world = b2(0.08);
        let explicit = AvbdWorld::from_world(&world, AvbdConfig { integrator: Integrator::Explicit, ..AvbdConfig::default() }).run();
        let mts = MtsWorld::new(&world, MtsConfig { all_hot: true, ratio: 7, ..MtsConfig::default() }).run();
        assert_eq!(tip(&explicit.probes), tip(&mts.outcome.probes));
    }

    /// Gravouil-Combescure with the multiplier predictor on the cantilever, root explicit,
    /// tip in Newmark at 30 explicit steps: the tip follows the explicit truth through the
    /// release and the free vibration.
    #[test]
    fn gc_coupling_follows_the_cantilever() {
        let world = b2(0.6);
        let explicit = AvbdWorld::from_world(&world, AvbdConfig { integrator: Integrator::Explicit, ..AvbdConfig::default() }).run();
        let config = MtsConfig { ratio: 30, seed: (1..=6).collect(), escalation: false, verify: Verify::Count, ..MtsConfig::default() };
        let mts = MtsWorld::new(&world, config).run();
        let (a, b) = (tip(&explicit.probes), tip(&mts.outcome.probes));
        let amp = a.iter().map(|p| p[1].abs()).fold(0.0, f64::max);
        let n = a.len().min(b.len());
        let worst = (0..n).map(|i| (a[i][1] - b[i][1]).abs()).fold(0.0, f64::max);
        assert!(worst < 0.02 * amp, "tip error {worst:.3e} of amplitude {amp:.3e}");
        assert!(!mts.diverged);
    }
}

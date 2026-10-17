//! # Modal pre-window for impacts (prototype, `modal` feature)
//!
//! While a struck cluster is intact and every bond is below its damage onset, its hidden
//! dynamics are the linear system `M x'' + C x' + K x = f(t)` (small-strain bonds, the
//! bonds' dashpots, Rayleigh mass damping when it is on). This module precomputes the
//! system's exact modes once per asset ([`ModalBasis`]) and runs the start of an impact in
//! them ([`run`]):
//!
//! * **Modes.** `K` and `C` are assembled from the cluster's live bonds exactly as the
//!   explicit solver applies them (the bond kinematics `B`, `B^T diag(k) B` and
//!   `B^T diag(c) B`), `M` from the chunks' masses and inertias, supports removed. With
//!   `M = L L^T` (chunk by chunk), the undamped modes of `L^-1 K L^-T` give `Omega` and
//!   `Phi = L^-T Q`; a free cluster's six rigid modes are dropped (the floating frame
//!   carries rigid motion; the bonds' dashpots and the self-equilibrated mass damping do
//!   no work on it). The bond dashpots are not proportional to `K` (each bond's ratio
//!   `beta = 2 zeta / omega_b` depends on its reduced mass), so `C~ = Phi^T C Phi` is not
//!   diagonal: the damped system is diagonalised in state space, `s = [Omega eta; eta']`,
//!   `s' = A s + [0; Phi^T f]`, `A = [[0, Omega], [-Omega, -C~]]`, by a real
//!   eigendecomposition `A = V D V^-1` (complex-conjugate pairs) and the explicit inverse
//!   `V^-1` (exact biorthogonality, also for repeated frequencies). Each complex modal
//!   coordinate obeys `c' = lambda c + b^T f`, and the hidden displacement and velocity
//!   are `x = Re(sum psi c)`, `x' = Re(sum psi lambda c)`.
//! * **Stepping.** The contact (the world's own penalty or layer contact, `apply_contacts`)
//!   is evaluated at the world's substep from the hidden state of the chunks it can touch
//!   (modal sums over their rows only), held over the step, and the modes advance in
//!   closed form under that piecewise-constant load: `c <- e^(lambda h) c +
//!   (e^(lambda h) - 1)/lambda g`. No stability limit applies to the structure; the step
//!   is the contact's resolution. The impactor and a free cluster's rigid motion are
//!   integrated by the world's own update.
//! * **Onset.** Every bond's deformation is linear in the modal coordinates. Bonds are
//!   checked in blocks of steps (one GEMM gives every DOF at every step of the block) by
//!   the joint law itself (`JointModel::evaluate`, committing its history exactly as the
//!   explicit substep does: strain-rate filter, fatigue, the dormant contact patch). The
//!   first step at which any bond reaches damage onset (or the configured failure index)
//!   ends the window: the state rolls back to the start of that step and is handed over
//!   to the explicit solver, which repeats that step's evaluation and continues.
//! * **No damage.** Once the contact has stopped, the free response of each mode can only
//!   decay (`Re lambda <= 0`), so `|q_b(t) - q_b,inf| <= sum |D_bj| |c_j|` bounds every
//!   bond force for all later time. If that bound keeps every bond's failure index (with
//!   conservative strength: no rate increase, fatigue at its worst over the remaining
//!   time) below a threshold, and the impactor's ballistic path cannot come back into
//!   contact (a swept separating-axis test with the same displacement bound), no bond can
//!   reach onset: no explicit window is needed.
//!
//! Exactness: for the linear system with piecewise-constant loads the modal update is
//! exact up to floating point (`tests/modal_prewindow.rs` compares it against RK4 at a
//! tiny step). It differs from the explicit solver only by the explicit solver's own
//! time-discretisation error.
//!
//! Scope (prototype): explicit solve mode; one bonded cluster in the world (anchored or
//! free); intact bonds without rebar; no scripted events or engine coupling. Free
//! clusters are linearised about their rigid motion (centrifugal, Coriolis and gyroscopic
//! terms of the floating frame are second order and left out); `corotational_bonds` is
//! linearised in the dynamics (the onset check uses the co-rotational kinematics). The
//! certificate is offered for anchored clusters only.

use std::time::Instant;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd;
use faer::linalg::solvers::DenseSolveCore;
use faer::diag::Diag;
use faer::Mat;
use rayon::prelude::*;

use super::{ContactLedger, Impactor, World};
use crate::bond::{ChunkTurn, Local6};
use crate::contact::OBox;
use crate::joint::{failure_indices, fatigue_factor, stress_measures, JointResponse, JointState};
use crate::math::{Pose, Quat, Vec3};
use crate::scene::{SolveMode, Support};

pub type ModalResult<T> = Result<T, String>;

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

// ------------------------------------------------------------------ assembly

/// A chunk's block of the mass factor `M = L L^T` over its free components.
#[derive(Clone, Debug)]
struct Block {
    chunk: usize,
    /// First free DOF and the components it holds (`first..first + comps.len()`).
    first: usize,
    comps: Vec<usize>,
    /// `L^-1` restricted to `comps` (row-major, `comps.len()` square).
    linv: Vec<f64>,
}

/// The linear hidden dynamics of one cluster: free DOFs and dense `K`, `C`, `M`.
pub struct Assembly {
    pub structure: usize,
    pub chunks: Vec<usize>,
    pub anchored: bool,
    /// Free DOF per structure chunk and component (None: held, or not in the cluster).
    pub dof: Vec<[Option<usize>; 6]>,
    pub n: usize,
    pub k: Mat<f64>,
    pub c: Mat<f64>,
    pub m: Mat<f64>,
    blocks: Vec<Block>,
}

/// The 6 x 12 bond kinematics `d = B [u_a; th_a; u_b; th_b]` (local components), as the
/// explicit solver's `BondGeometry::kinematics`.
fn bond_rows(g: &crate::bond::BondGeometry) -> [[f64; 12]; 6] {
    let axes = [g.t1, g.t2, g.normal];
    std::array::from_fn(|comp| {
        let t = axes[comp % 3];
        let r: [Vec3; 4] = if comp < 3 { [-t, -(g.ra.cross(t)), t, g.rb.cross(t)] } else { [Vec3::ZERO, -t, Vec3::ZERO, t] };
        std::array::from_fn(|x| r[x / 3][x % 3])
    })
}

fn local6(l: &Local6) -> [f64; 6] {
    [l.lin.x, l.lin.y, l.lin.z, l.ang.x, l.ang.y, l.ang.z]
}

fn comp(u: Vec3, th: Vec3, d: usize) -> f64 {
    if d < 3 {
        u[d]
    } else {
        th[d - 3]
    }
}

/// Lower Cholesky factor of a small symmetric positive matrix (row-major `n x n`) and its inverse.
fn chol_and_inverse(a: &[f64], n: usize) -> ModalResult<(Vec<f64>, Vec<f64>)> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|p| l[i * n + p] * l[j * n + p]).sum();
            if i == j {
                let v = a[i * n + i] - sum;
                if !(v > 0.0) {
                    return Err("chunk mass matrix is not positive definite".into());
                }
                l[i * n + i] = v.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let sum: f64 = (j..i).map(|p| l[i * n + p] * inv[p * n + j]).sum();
            inv[i * n + j] = -sum / l[i * n + i];
        }
    }
    Ok((l, inv))
}

impl Assembly {
    /// Assemble cluster `ci` of `world`'s solver. Refused unless every bond of the
    /// cluster is intact (no damage, no crushing) and carries no rebar.
    pub fn new(world: &World, ci: usize) -> ModalResult<Assembly> {
        let sv = &world.solver;
        let cl = &sv.clusters[ci];
        let s = cl.structure;
        let st = &sv.structures[s];
        let mut dof = vec![[None; 6]; st.chunks.len()];
        let mut blocks = Vec::new();
        let mut n = 0;
        for &c in &cl.chunks {
            let ch = &st.chunks[c];
            let comps: Vec<usize> = match ch.support {
                Support::Fixed => vec![],
                Support::Pinned => vec![3, 4, 5],
                Support::None => (0..6).collect(),
            };
            if comps.is_empty() {
                continue;
            }
            let mu = sv.chunks[s][c].inertia_scale;
            let mut m6 = [[0.0f64; 6]; 6];
            for d in 0..3 {
                m6[d][d] = ch.mass * mu;
            }
            for p in 0..3 {
                for q in 0..3 {
                    m6[3 + p][3 + q] = ch.inertia.m[p][q] * mu;
                }
            }
            let k = comps.len();
            let a: Vec<f64> = (0..k * k).map(|x| m6[comps[x / k]][comps[x % k]]).collect();
            let (_, linv) = chol_and_inverse(&a, k)?;
            for (x, &d) in comps.iter().enumerate() {
                dof[c][d] = Some(n + x);
            }
            blocks.push(Block { chunk: c, first: n, comps, linv });
            n += k;
        }
        if n == 0 {
            return Err("cluster has no free degrees of freedom".into());
        }
        let mut k = Mat::<f64>::zeros(n, n);
        let mut cm = Mat::<f64>::zeros(n, n);
        let mut m = Mat::<f64>::zeros(n, n);
        for &bi in &cl.bonds {
            let b = &sv.bonds[s][bi];
            if !b.alive {
                continue;
            }
            if b.joint.damage != 0.0 || b.joint.crush != 0.0 {
                return Err(format!("bond {bi} is damaged: the cluster is not in its linear range"));
            }
            if b.rebar.is_some() && !b.joint.rebar_broken {
                return Err(format!("bond {bi} carries rebar (not supported by the prototype)"));
            }
            let g = &b.geometry;
            let rows = bond_rows(g);
            let k6 = local6(&b.stiffness.as_local());
            let c6 = local6(&b.damping);
            let dofs: [Option<usize>; 12] = std::array::from_fn(|x| if x < 6 { dof[g.a][x] } else { dof[g.b][x - 6] });
            for x in 0..12 {
                let Some(i) = dofs[x] else { continue };
                for y in 0..12 {
                    let Some(j) = dofs[y] else { continue };
                    let (mut kv, mut cv) = (0.0, 0.0);
                    for q in 0..6 {
                        let r = rows[q][x] * rows[q][y];
                        kv += k6[q] * r;
                        cv += c6[q] * r;
                    }
                    k[(i, j)] += kv;
                    cm[(i, j)] += cv;
                }
            }
        }
        // `rayleigh_damping`: the mass-proportional dashpot on the hidden motion of
        // unsupported chunks (anchored clusters; free ones only with co-rotational bonds).
        let alpha = if cl.anchored || sv.config.methods.corotational_bonds { sv.mass_damping.get(s).copied().unwrap_or(0.0) } else { 0.0 };
        for blk in &blocks {
            let kk = blk.comps.len();
            let ch = &st.chunks[blk.chunk];
            let mu = sv.chunks[s][blk.chunk].inertia_scale;
            for x in 0..kk {
                for y in 0..kk {
                    let (p, q) = (blk.comps[x], blk.comps[y]);
                    let v = if p < 3 || q < 3 {
                        if p == q {
                            ch.mass * mu
                        } else {
                            0.0
                        }
                    } else {
                        ch.inertia.m[p - 3][q - 3] * mu
                    };
                    m[(blk.first + x, blk.first + y)] = v;
                    if alpha > 0.0 && ch.support == Support::None {
                        cm[(blk.first + x, blk.first + y)] += alpha * v;
                    }
                }
            }
        }
        Ok(Assembly { structure: s, chunks: cl.chunks.clone(), anchored: cl.anchored, dof, n, k, c: cm, m, blocks })
    }

    /// `L^-1 A L^-T` for a symmetric `A` (block by block).
    fn congruent(&self, a: &Mat<f64>) -> Mat<f64> {
        let mut out = a.clone();
        self.left(&mut out, false);
        let mut t = out.transpose().to_owned();
        self.left(&mut t, false);
        t
    }

    /// `A <- L^-1 A` (or `L^-T A` with `transpose`), rows block by block.
    fn left(&self, a: &mut Mat<f64>, transpose: bool) {
        let cols = a.ncols();
        for blk in &self.blocks {
            let k = blk.comps.len();
            for col in 0..cols {
                let v: Vec<f64> = (0..k).map(|x| a[(blk.first + x, col)]).collect();
                for x in 0..k {
                    let mut s = 0.0;
                    for y in 0..k {
                        let coef = if transpose { blk.linv[y * k + x] } else { blk.linv[x * k + y] };
                        s += coef * v[y];
                    }
                    a[(blk.first + x, col)] = s;
                }
            }
        }
    }
}

// ------------------------------------------------------------------ basis

/// Wall-clock seconds of the basis precompute, by stage.
#[derive(Clone, Debug, Default)]
pub struct BasisTimings {
    pub assemble: f64,
    pub undamped: f64,
    pub project: f64,
    pub state_space: f64,
    pub inverse: f64,
    pub outputs: f64,
    pub total: f64,
}

/// The exact complex modes of one intact cluster (per asset: independent of the impact).
pub struct ModalBasis {
    pub structure: usize,
    pub chunks: Vec<usize>,
    pub anchored: bool,
    pub dof: Vec<[Option<usize>; 6]>,
    /// Free DOFs, rigid modes dropped, flexible modes.
    pub n: usize,
    pub rigid_dropped: usize,
    pub nf: usize,
    /// One eigenvalue per complex-conjugate pair (or real mode): `(Re, Im)`.
    pub lambda: Vec<(f64, f64)>,
    /// `psi` (displacement per modal coordinate, conjugate pair's factor 2 folded in) and
    /// the input map `b` (modal force per DOF force), `n x lambda.len()`.
    pub psi_re: Mat<f64>,
    pub psi_im: Mat<f64>,
    pub bin_re: Mat<f64>,
    pub bin_im: Mat<f64>,
    /// Undamped circular frequencies (rad/s), lowest and highest flexible.
    pub omega_range: (f64, f64),
    /// Modal damping ratios `-Re lambda / |lambda|`, lowest and highest.
    pub zeta_range: (f64, f64),
    /// Largest off-diagonal of `C~` over the geometric mean of its diagonals: 0 for
    /// classical (proportional) damping.
    pub nonclassical: f64,
    /// Relative residuals of the undamped eigenpairs, the state-space eigenpairs and the
    /// inverse (each for a pseudo-random vector).
    pub residuals: [f64; 3],
    pub timings: BasisTimings,
}

impl ModalBasis {
    /// Precompute the modes of cluster `ci` of `world` (see the module documentation).
    pub fn new(world: &World, ci: usize) -> ModalResult<ModalBasis> {
        let t_all = Instant::now();
        let mut timings = BasisTimings::default();
        let t = Instant::now();
        let asm = Assembly::new(world, ci)?;
        let khat = asm.congruent(&asm.k);
        let chat = asm.congruent(&asm.c);
        timings.assemble = secs(t);

        let t = Instant::now();
        let n = asm.n;
        // The tridiagonal QR algorithm, not divide and conquer: faer's divide and conquer
        // (n >= 128) returned eigenvectors with `Q^T K Q` off-diagonals of 1e-4 relative on
        // a free wall (a six-fold zero eigenvalue and symmetric, repeated frequencies).
        let par = faer::get_global_parallelism();
        let mut sdiag = Diag::<f64>::zeros(n);
        let mut qall = Mat::<f64>::zeros(n, n);
        {
            let mut params = <evd::SelfAdjointEvdParams as faer::Auto<f64>>::auto();
            params.recursion_threshold = usize::MAX;
            let spec: faer::Spec<evd::SelfAdjointEvdParams, f64> = params.into();
            let req = evd::self_adjoint_evd_scratch::<f64>(n, evd::ComputeEigenvectors::Yes, par, spec);
            let mut buf = MemBuffer::new(req);
            evd::self_adjoint_evd(khat.as_ref(), sdiag.as_mut(), Some(qall.as_mut()), par, MemStack::new(&mut buf), spec)
                .map_err(|e| format!("symmetric eigensolver: {e:?}"))?;
        }
        let sval: Vec<f64> = (0..n).map(|i| sdiag[i]).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| sval[a].total_cmp(&sval[b]));
        let top = sval[order[n - 1]];
        let n_rigid = if asm.anchored { 0 } else { 6 };
        if n <= n_rigid {
            return Err("cluster too small for flexible modes".into());
        }
        for (r, &i) in order.iter().enumerate() {
            let w2 = sval[i];
            if r < n_rigid {
                if w2.abs() > 1e-8 * top {
                    return Err(format!("free cluster: rigid mode {r} has omega^2 {w2:.3e} (top {top:.3e})"));
                }
            } else if !(w2 > 1e-12 * top) {
                return Err(format!("mechanism: flexible mode {r} has omega^2 {w2:.3e} (top {top:.3e})"));
            }
        }
        let flex: Vec<usize> = order[n_rigid..].to_vec();
        let nf = flex.len();
        let omega: Vec<f64> = flex.iter().map(|&i| sval[i].sqrt()).collect();
        let qf = Mat::<f64>::from_fn(n, nf, |r, j| qall[(r, flex[j])]);
        // Residual `|K^ Q y - Q S y| / |K^ Q y|` for a pseudo-random `y`.
        let residual_undamped = {
            let yv = pseudo_random(n, 11);
            let ym = Mat::<f64>::from_fn(n, 1, |i, _| yv[i]);
            let sy = Mat::<f64>::from_fn(n, 1, |i, _| sval[i] * yv[i]);
            let lhs = &khat * (&qall * &ym);
            let rhs = &qall * &sy;
            max_abs_diff(&lhs, &rhs) / max_abs(&lhs)
        };
        drop(qall);
        timings.undamped = secs(t);

        let t = Instant::now();
        let cq = &chat * &qf;
        let ct = qf.transpose() * &cq;
        drop(cq);
        let mut nonclassical = 0.0f64;
        for i in 0..nf {
            for j in 0..nf {
                if i != j {
                    let d = (ct[(i, i)].abs() * ct[(j, j)].abs()).sqrt();
                    if d > 0.0 {
                        nonclassical = nonclassical.max(ct[(i, j)].abs() / d);
                    } else if ct[(i, j)] != 0.0 {
                        nonclassical = f64::INFINITY;
                    }
                }
            }
        }
        // Phi_f = L^-T Q_f.
        let mut phi = qf;
        asm.left(&mut phi, true);
        timings.project = secs(t);

        let modes = if nonclassical <= CLASSICAL {
            let t = Instant::now();
            let m = classical_modes(&omega, &ct, &phi);
            timings.outputs = secs(t);
            m
        } else {
            general_modes(&omega, ct, &phi, &mut timings)?
        };
        let Modes { lambda, psi_re, psi_im, bin_re, bin_im, residual_state, residual_inverse } = modes;

        let mut zeta = (f64::INFINITY, 0.0f64);
        for &(re, im) in &lambda {
            let z = -re / re.hypot(im);
            zeta = (zeta.0.min(z), zeta.1.max(z));
        }
        if lambda.iter().any(|&(re, _)| re > 1e-9 * top.sqrt()) {
            return Err("unstable mode (Re lambda > 0)".into());
        }
        timings.total = secs(t_all);
        Ok(ModalBasis {
            structure: asm.structure,
            chunks: asm.chunks.clone(),
            anchored: asm.anchored,
            dof: asm.dof.clone(),
            n,
            rigid_dropped: n_rigid,
            nf,
            lambda,
            psi_re,
            psi_im,
            bin_re,
            bin_im,
            omega_range: (omega[0], omega[nf - 1]),
            zeta_range: zeta,
            nonclassical,
            residuals: [residual_undamped, residual_state, residual_inverse],
            timings,
        })
    }

    /// Number of modal coordinates (one per conjugate pair).
    pub fn modes(&self) -> usize {
        self.lambda.len()
    }

    /// Bytes held by the four `n x modes` matrices.
    pub fn bytes(&self) -> usize {
        4 * self.n * self.modes() * 8
    }

    /// Whether this basis was built from the same cluster (structure and chunks) as
    /// `world`'s cluster `ci`.
    pub fn matches(&self, world: &World, ci: usize) -> bool {
        let cl = &world.solver.clusters[ci];
        cl.structure == self.structure && cl.chunks == self.chunks && cl.anchored == self.anchored
    }

    /// Advance modal coordinates `c` (complex, `(re, im)` arrays) by time `h` under the
    /// constant modal input `g` (`b^T f`), in closed form.
    pub fn advance(&self, c: (&mut [f64], &mut [f64]), g: (&[f64], &[f64]), h: f64) {
        for m in 0..self.modes() {
            let (e, p) = step_coefficients(self.lambda[m], h);
            let (cr, ci) = (c.0[m], c.1[m]);
            let (gr, gi) = (g.0[m], g.1[m]);
            c.0[m] = e.0 * cr - e.1 * ci + p.0 * gr - p.1 * gi;
            c.1[m] = e.0 * ci + e.1 * cr + p.0 * gi + p.1 * gr;
        }
    }

    /// Hidden displacement and velocity (`n` each) of modal coordinates `c`.
    pub fn state(&self, c: (&[f64], &[f64])) -> (Vec<f64>, Vec<f64>) {
        let np = self.modes();
        let cre = Mat::<f64>::from_fn(np, 2, |j, k| if k == 0 { c.0[j] } else { self.lambda[j].0 * c.0[j] - self.lambda[j].1 * c.1[j] });
        let cim = Mat::<f64>::from_fn(np, 2, |j, k| if k == 0 { c.1[j] } else { self.lambda[j].0 * c.1[j] + self.lambda[j].1 * c.0[j] });
        let out = &self.psi_re * &cre - &self.psi_im * &cim;
        ((0..self.n).map(|i| out[(i, 0)]).collect(), (0..self.n).map(|i| out[(i, 1)]).collect())
    }

    /// Modal input `b^T f` of DOF forces `f` (`n`).
    pub fn input(&self, f: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let fm = Mat::<f64>::from_fn(self.n, 1, |i, _| f[i]);
        let gr = self.bin_re.transpose() * &fm;
        let gi = self.bin_im.transpose() * &fm;
        ((0..self.modes()).map(|j| gr[(j, 0)]).collect(), (0..self.modes()).map(|j| gi[(j, 0)]).collect())
    }
}

/// `C~` off-diagonals below this (relative) count as classical damping.
const CLASSICAL: f64 = 1e-10;

struct Modes {
    lambda: Vec<(f64, f64)>,
    psi_re: Mat<f64>,
    psi_im: Mat<f64>,
    bin_re: Mat<f64>,
    bin_im: Mat<f64>,
    residual_state: f64,
    residual_inverse: f64,
}

/// Classical damping (`C~` diagonal): each undamped mode is its own 2 x 2 block
/// `[[0, w], [-w, -c]]`, with eigenvalues `lambda^2 + c lambda + w^2 = 0`, right
/// eigenvector `(w, lambda)` and left `(-w / lambda, 1) lambda / (lambda^2 - w^2)`, in
/// closed form (exactly conditioned, also for repeated frequencies).
fn classical_modes(omega: &[f64], ct: &Mat<f64>, phi: &Mat<f64>) -> Modes {
    let nf = omega.len();
    // Per kept mode: (undamped mode, lambda, factor, input scalar lambda / (lambda^2 - w^2)).
    let mut kept: Vec<(usize, (f64, f64), f64, (f64, f64))> = Vec::new();
    let input = |l: (f64, f64), w: f64| -> (f64, f64) {
        // lambda / (lambda^2 - w^2), complex.
        let (dr, di) = (l.0 * l.0 - l.1 * l.1 - w * w, 2.0 * l.0 * l.1);
        let den = dr * dr + di * di;
        ((l.0 * dr + l.1 * di) / den, (l.1 * dr - l.0 * di) / den)
    };
    for i in 0..nf {
        let (w, c) = (omega[i], ct[(i, i)]);
        let disc = c * c - 4.0 * w * w;
        if disc < 0.0 {
            let l = (-0.5 * c, 0.5 * (-disc).sqrt());
            kept.push((i, l, 2.0, input(l, w)));
        } else {
            for sgn in [1.0, -1.0] {
                let l = (0.5 * (-c + sgn * disc.sqrt()), 0.0);
                kept.push((i, l, 1.0, input(l, w)));
            }
        }
    }
    let np = kept.len();
    let n = phi.nrows();
    let psi_re = Mat::<f64>::from_fn(n, np, |r, m| kept[m].2 * phi[(r, kept[m].0)]);
    let psi_im = Mat::<f64>::zeros(n, np);
    let bin_re = Mat::<f64>::from_fn(n, np, |r, m| kept[m].3 .0 * phi[(r, kept[m].0)]);
    let bin_im = Mat::<f64>::from_fn(n, np, |r, m| kept[m].3 .1 * phi[(r, kept[m].0)]);
    Modes { lambda: kept.iter().map(|k| k.1).collect(), psi_re, psi_im, bin_re, bin_im, residual_state: 0.0, residual_inverse: 0.0 }
}

/// Non-classical damping: the real eigendecomposition of the state-space matrix and its
/// explicit inverse (see the module documentation).
fn general_modes(omega: &[f64], ct: Mat<f64>, phi: &Mat<f64>, timings: &mut BasisTimings) -> ModalResult<Modes> {
    let nf = omega.len();
    let t = Instant::now();
    let n2 = 2 * nf;
    let par = faer::get_global_parallelism();
    let mut a = Mat::<f64>::zeros(n2, n2);
    for i in 0..nf {
        a[(i, nf + i)] = omega[i];
        a[(nf + i, i)] = -omega[i];
        for j in 0..nf {
            a[(nf + i, nf + j)] = -ct[(i, j)];
        }
    }
    drop(ct);
    let mut ur = Mat::<f64>::zeros(n2, n2);
    let mut sre = Diag::<f64>::zeros(n2);
    let mut sim = Diag::<f64>::zeros(n2);
    {
        let req = evd::evd_scratch::<f64>(n2, evd::ComputeEigenvectors::No, evd::ComputeEigenvectors::Yes, par, Default::default());
        let mut buf = MemBuffer::new(req);
        evd::evd_real(a.as_ref(), sre.as_mut(), sim.as_mut(), None, Some(ur.as_mut()), par, MemStack::new(&mut buf), Default::default())
            .map_err(|e| format!("state-space eigensolver: {e:?}"))?;
    }
    // Residual `|A U y - U D y| / |A U y|` (real block form of `D`).
    let residual_state = {
        let yv = pseudo_random(n2, 12);
        let ym = Mat::<f64>::from_fn(n2, 1, |i, _| yv[i]);
        let mut dy = vec![0.0; n2];
        let mut j = 0;
        while j < n2 {
            if sim[j] == 0.0 {
                dy[j] = sre[j] * yv[j];
                j += 1;
            } else {
                dy[j] = sre[j] * yv[j] + sim[j] * yv[j + 1];
                dy[j + 1] = -sim[j] * yv[j] + sre[j] * yv[j + 1];
                j += 2;
            }
        }
        let dym = Mat::<f64>::from_fn(n2, 1, |i, _| dy[i]);
        let lhs = &a * (&ur * &ym);
        let rhs = &ur * &dym;
        max_abs_diff(&lhs, &rhs) / max_abs(&lhs)
    };
    drop(a);
    timings.state_space = secs(t);

    let t = Instant::now();
    let y = ur.partial_piv_lu().inverse();
    let residual_inverse = {
        let yv = pseudo_random(n2, 13);
        let ym = Mat::<f64>::from_fn(n2, 1, |i, _| yv[i]);
        let back = &y * (&ur * &ym);
        max_abs_diff(&back, &ym) / max_abs(&ym)
    };
    timings.inverse = secs(t);

    let t = Instant::now();
    // Kept modes: one per conjugate pair (factor 2), real modes (factor 1).
    let mut kept: Vec<(usize, bool)> = Vec::new();
    let mut j = 0;
    while j < n2 {
        if sim[j] == 0.0 {
            kept.push((j, false));
            j += 1;
        } else {
            kept.push((j, true));
            j += 2;
        }
    }
    let np = kept.len();
    let lambda: Vec<(f64, f64)> = kept.iter().map(|&(j, _)| (sre[j], sim[j])).collect();
    // W = Omega^-1 V_top (nf x np) and the left rows' bottom block (np x nf).
    let wre = Mat::<f64>::from_fn(nf, np, |r, m| {
        let (j, pair) = kept[m];
        let f = if pair { 2.0 } else { 1.0 };
        f * ur[(r, j)] / omega[r]
    });
    let wim = Mat::<f64>::from_fn(nf, np, |r, m| {
        let (j, pair) = kept[m];
        if pair {
            2.0 * ur[(r, j + 1)] / omega[r]
        } else {
            0.0
        }
    });
    let lre = Mat::<f64>::from_fn(nf, np, |r, m| {
        let (j, pair) = kept[m];
        if pair {
            0.5 * y[(j, nf + r)]
        } else {
            y[(j, nf + r)]
        }
    });
    let lim = Mat::<f64>::from_fn(nf, np, |r, m| {
        let (j, pair) = kept[m];
        if pair {
            -0.5 * y[(j + 1, nf + r)]
        } else {
            0.0
        }
    });
    drop(ur);
    drop(y);
    let psi_re = phi * &wre;
    let psi_im = phi * &wim;
    let bin_re = phi * &lre;
    let bin_im = phi * &lim;
    timings.outputs = secs(t);
    Ok(Modes { lambda, psi_re, psi_im, bin_re, bin_im, residual_state, residual_inverse })
}

fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs(a: &Mat<f64>) -> f64 {
    let mut m = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = m.max(a[(i, j)].abs());
        }
    }
    m
}

fn max_abs_diff(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut m = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = m.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    m
}

/// `(e^(lambda h), (e^(lambda h) - 1) / lambda)` as complex pairs.
fn step_coefficients(lambda: (f64, f64), h: f64) -> ((f64, f64), (f64, f64)) {
    let (a, b) = lambda;
    let ea = (a * h).exp();
    let (sb, cb) = (b * h).sin_cos();
    let e = (ea * cb, ea * sb);
    // e - 1 without cancellation: expm1(ah) cos(bh) - 2 sin^2(bh / 2) + i e^(ah) sin(bh).
    let s2 = (0.5 * b * h).sin();
    let em1 = ((a * h).exp_m1() * cb - 2.0 * s2 * s2, ea * sb);
    let den = a * a + b * b;
    let p = if den > 0.0 { ((em1.0 * a + em1.1 * b) / den, (em1.1 * a - em1.0 * b) / den) } else { (h, 0.0) };
    (e, p)
}

// ------------------------------------------------------------------ the window

/// Settings of the pre-window.
#[derive(Clone, Debug)]
pub struct ModalConfig {
    /// Parallel modal sums, GEMMs and bond checks (faer's global setting is left alone).
    pub parallel: bool,
    /// Steps per batched bond check (one GEMM per block; a block is rolled back to the
    /// step where a bond reaches onset).
    pub block: usize,
    /// Hand over when an intact bond's failure index (with its rate factor, as the law
    /// computes it) reaches this, or when the law reports damage (1: at onset).
    pub handover_utilization: f64,
    /// Try the no-damage certificate once the contact has stopped (anchored clusters).
    pub certify: bool,
    /// The certificate holds when every bond's failure-index bound stays below this.
    pub certificate_threshold: f64,
    /// Contact-free steps before the first certificate attempt.
    pub quiet_steps: usize,
    /// Simulated time between certificate attempts (s).
    pub certificate_interval: f64,
    /// Distance (m) within which a chunk counts as a contact candidate (its hidden state
    /// is reconstructed each step); far above any hidden displacement before onset.
    pub contact_margin: f64,
    /// How the contact force acts over a step. `true`: as an impulse at the step's start,
    /// then the exact free response (kick, then exact drift: the explicit solver's own
    /// symplectic-Euler staggering, which the impactor's update also uses). `false`: held
    /// constant over the step (zero-order hold), exactly integrated; this lags the force
    /// by half a step against the explicit solver: on the free-wall ram the hidden
    /// displacement at the handover differed from the explicit solver's by 20% (impulse:
    /// 0.3%), converging only to first order in the step.
    pub contact_impulse: bool,
}

impl Default for ModalConfig {
    fn default() -> Self {
        ModalConfig {
            parallel: true,
            block: 32,
            handover_utilization: 1.0,
            certify: true,
            certificate_threshold: 0.95,
            quiet_steps: 200,
            certificate_interval: 2e-3,
            contact_margin: 2e-3,
            contact_impulse: true,
        }
    }
}

/// How the window ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Bond `bond` reached onset in the state at `time` (the start of that step): the
    /// world was handed over to the explicit solver there. `crossing` is the time its
    /// failure index (rate factor frozen over the step) crossed the threshold in the
    /// closed-form trajectory.
    Handover { time: f64, bond: usize, crossing: f64 },
    /// No bond can reach onset after `time` (certificate; worst failure-index bound).
    Certified { time: f64, bound: f64 },
    /// The scene ended at `time` without any bond reaching onset.
    NoOnset { time: f64 },
}

impl Verdict {
    pub fn damages(&self) -> bool {
        matches!(self, Verdict::Handover { .. })
    }
    pub fn time(&self) -> f64 {
        match *self {
            Verdict::Handover { time, .. } | Verdict::Certified { time, .. } | Verdict::NoOnset { time } => time,
        }
    }
}

/// Wall-clock seconds spent in the window, by stage.
#[derive(Clone, Debug, Default)]
pub struct WindowTimings {
    pub setup: f64,
    /// Candidates, the world's contact evaluation and its loads.
    pub contact: f64,
    /// Modal sums at the contact rows, the modal input and the closed-form update.
    pub modal: f64,
    /// Impactors and rigid motion, snapshots for rollback.
    pub bodies: f64,
    /// Bond checks: GEMM to every DOF, then the joint law.
    pub checks_gemm: f64,
    pub checks_law: f64,
    pub certificate: f64,
    pub handover: f64,
    pub total: f64,
}

#[derive(Clone, Debug)]
pub struct WindowReport {
    pub verdict: Verdict,
    /// Contact-resolution steps taken (the explicit substeps the window replaced).
    pub steps: u64,
    /// Steps with a contact force on the cluster.
    pub contact_steps: u64,
    pub first_contact: Option<f64>,
    /// Largest failure index the law reported during the window.
    pub max_utilization: f64,
    pub max_candidates: usize,
    /// Largest hidden displacement seen at a contact row (m): must stay far below
    /// `contact_margin`.
    pub max_displacement: f64,
    pub certificate_attempts: u32,
    /// The world's state where the window ended (for comparison with the explicit solver
    /// at the same time).
    pub end_state: Option<WorldState>,
    pub last_certificate_bound: f64,
    pub timings: WindowTimings,
}

/// What is needed to restart a step: the modal state and the world's moving parts.
struct Entry {
    c_re: Vec<f64>,
    c_im: Vec<f64>,
    g_re: Vec<f64>,
    g_im: Vec<f64>,
    time: f64,
    dt: f64,
    frame: u64,
    sub: usize,
    nsub: usize,
    impactors: Vec<Impactor>,
    rigid: (Pose, Vec3, Vec3),
    sticks: std::collections::HashMap<super::ContactKey, super::Stick>,
    pair_offsets: std::collections::HashMap<super::ChunkPair, Vec<(f64, Vec3)>>,
    contact: ContactLedger,
    contact_force: bool,
    /// Largest hidden displacement at a contact row in the state this step started from.
    displacement: f64,
}

struct Window<'a> {
    basis: &'a ModalBasis,
    cfg: &'a ModalConfig,
    ci: usize,
    s: usize,
    np: usize,
    /// Hidden state at the window's start (the reference of the modal coordinates).
    x0: Vec<f64>,
    c_re: Vec<f64>,
    c_im: Vec<f64>,
    /// Modal input of the constant loads (gravity and the start's residual).
    gres_re: Vec<f64>,
    gres_im: Vec<f64>,
    h: f64,
    e: Vec<(f64, f64)>,
    p: Vec<(f64, f64)>,
    /// Contact candidates (structure chunk ids) and their DOF rows.
    cand: Vec<usize>,
    rows: Vec<usize>,
    psi_rows: (Vec<f64>, Vec<f64>),
    bin_rows: (Vec<f64>, Vec<f64>),
    report: WindowReport,
    step_displacement: f64,
}

impl WindowReport {
    /// Book the steps of `entries` that stand (their states were checked).
    fn book(&mut self, entries: &[Entry]) {
        for e in entries {
            self.max_displacement = self.max_displacement.max(e.displacement);
            if e.contact_force {
                self.contact_steps += 1;
                self.first_contact.get_or_insert(e.time);
            }
        }
    }
}

fn par_rows<F: Fn(usize) -> f64 + Sync + Send>(parallel: bool, n: usize, f: F) -> Vec<f64> {
    if parallel && n >= 8 {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

impl World {
    /// Finish the current frame explicitly from substep `from` of `n` (explicit mode).
    fn finish_frame_explicit(&mut self, from: usize, n: usize, dt: f64) {
        self.frame_loads.resize(&self.solver);
        let splits_before = self.solver.events.len();
        for _ in from..n {
            self.substep(dt);
            let mut fl = std::mem::take(&mut self.frame_loads);
            fl.add_scaled(&self.loads, 1.0 / n as f64);
            self.frame_loads = fl;
        }
        self.frame += 1;
        if self.solver.events.len() != splits_before {
            self.topology_version += 1;
        }
        self.frame_loads.clear();
    }
}

impl<'a> Window<'a> {
    fn set_step(&mut self, h: f64) {
        if h != self.h {
            self.h = h;
            let lam = &self.basis.lambda;
            let coeffs: Vec<_> = (0..self.np).map(|m| step_coefficients(lam[m], h)).collect();
            self.e = coeffs.iter().map(|x| x.0).collect();
            self.p = coeffs.iter().map(|x| x.1).collect();
        }
    }

    /// Chunks the impactors (or the ground) can touch within a step, with the margin.
    fn update_candidates(&mut self, world: &World) {
        let sv = &world.solver;
        let st = &sv.structures[self.s];
        let cl = &sv.clusters[self.ci];
        let margin = self.cfg.contact_margin;
        let mut cand = Vec::new();
        for imp in &world.impactors {
            if imp.driven {
                continue;
            }
            let mut ib = imp.obox();
            let travel = imp.velocity.norm() * self.h + margin;
            ib.half = ib.half + Vec3::splat(travel);
            let reach = ib.bounding_radius();
            for &c in &cl.chunks {
                let ch = &st.chunks[c];
                let x = sv.chunk_position(self.s, c);
                if (x - ib.center).norm() > reach + ch.half_extents.norm() + margin {
                    continue;
                }
                let mut b = world.chunk_box(self.s, c);
                b.half = b.half + Vec3::splat(margin);
                if b.may_overlap(&ib) {
                    cand.push(c);
                }
            }
        }
        if let Some(g) = &world.scene.ground {
            if !cl.anchored {
                for &c in &cl.chunks {
                    let x = sv.chunk_position(self.s, c);
                    if x.z - st.chunks[c].half_extents.norm() <= g.height + margin {
                        cand.push(c);
                    }
                }
            }
        }
        cand.sort_unstable();
        cand.dedup();
        cand.retain(|&c| self.basis.dof[c].iter().any(|d| d.is_some()));
        if cand != self.cand {
            self.cand = cand;
            self.rows = self.cand.iter().flat_map(|&c| self.basis.dof[c].iter().flatten().copied()).collect();
            let np = self.np;
            let gather = |m: &Mat<f64>| -> Vec<f64> {
                let mut out = vec![0.0; self.rows.len() * np];
                for (r, &i) in self.rows.iter().enumerate() {
                    for j in 0..np {
                        out[r * np + j] = m[(i, j)];
                    }
                }
                out
            };
            self.psi_rows = (gather(&self.basis.psi_re), gather(&self.basis.psi_im));
            self.bin_rows = (gather(&self.basis.bin_re), gather(&self.basis.bin_im));
            self.report.max_candidates = self.report.max_candidates.max(self.cand.len());
        }
    }

    /// Write the candidates' hidden state (from the modal coordinates) into the solver.
    fn write_candidates(&mut self, world: &mut World) {
        let np = self.np;
        let lam = &self.basis.lambda;
        let lc: Vec<(f64, f64)> = (0..np).map(|j| (lam[j].0 * self.c_re[j] - lam[j].1 * self.c_im[j], lam[j].0 * self.c_im[j] + lam[j].1 * self.c_re[j])).collect();
        let (pr, pi) = (&self.psi_rows.0, &self.psi_rows.1);
        let (cr, cim) = (&self.c_re, &self.c_im);
        let nr = self.rows.len();
        let both = par_rows(self.cfg.parallel, 2 * nr, |q| {
            let r = q / 2;
            let (a, b) = (&pr[r * np..(r + 1) * np], &pi[r * np..(r + 1) * np]);
            let mut s = 0.0;
            if q % 2 == 0 {
                for j in 0..np {
                    s += a[j] * cr[j] - b[j] * cim[j];
                }
            } else {
                for j in 0..np {
                    s += a[j] * lc[j].0 - b[j] * lc[j].1;
                }
            }
            s
        });
        let mut r = 0;
        for &c in &self.cand {
            let st = &mut world.solver.chunks[self.s][c];
            let (mut u, mut th, mut v, mut w) = (st.u, st.th, st.v, st.w);
            for d in 0..6 {
                if let Some(i) = self.basis.dof[c][d] {
                    let (x, xv) = (self.x0[i] + both[2 * r], both[2 * r + 1]);
                    self.step_displacement = self.step_displacement.max(both[2 * r].abs());
                    if d < 3 {
                        u[d] = x;
                        v[d] = xv;
                    } else {
                        th[d - 3] = x;
                        w[d - 3] = xv;
                    }
                    r += 1;
                }
            }
            st.u = u;
            st.th = th;
            st.v = v;
            st.w = w;
        }
    }

    /// Hidden displacement (`n`) and velocity of every free DOF, written into the solver.
    /// With `previous` (the modal state one step `h` earlier), the velocity written is the
    /// step's mean `(x(t) - x(t - h)) / h`: the explicit solver's staggered velocity
    /// `v_(n-1/2)` (symplectic Euler stores it at the start of step `n`), so that the
    /// explicit solver continues from the state its own scheme would hold.
    fn write_all(&self, world: &mut World, previous: Option<(&[f64], &[f64], f64)>) {
        let (x, mut v) = self.basis.state((&self.c_re, &self.c_im));
        if let Some((pr, pi, h)) = previous {
            let (xp, _) = self.basis.state((pr, pi));
            for i in 0..v.len() {
                v[i] = (x[i] - xp[i]) / h;
            }
        }
        for &c in &self.basis.chunks {
            let st = &mut world.solver.chunks[self.s][c];
            for d in 0..6 {
                if let Some(i) = self.basis.dof[c][d] {
                    if d < 3 {
                        st.u[d] = self.x0[i] + x[i];
                        st.v[d] = v[i];
                    } else {
                        st.th[d - 3] = self.x0[i] + x[i];
                        st.w[d - 3] = v[i];
                    }
                }
            }
        }
    }

    /// One contact-resolution step of length `h` (see the module documentation).
    /// Returns the modal input it used and whether a contact force acted.
    fn step(&mut self, world: &mut World) -> (Vec<f64>, Vec<f64>, bool) {
        let h = self.h;
        let np = self.np;
        let t = Instant::now();
        self.update_candidates(world);
        self.report.timings.contact += secs(t);
        let t = Instant::now();
        self.write_candidates(world);
        self.report.timings.modal += secs(t);

        let t = Instant::now();
        world.loads.resize(&world.solver);
        world.loads.clear();
        world.apply_scripted_loads(world.solver.time);
        let imp_loads = world.apply_contacts(h);
        // DOF forces of every loaded chunk of the cluster (body frame).
        let pose = world.solver.clusters[self.ci].pose;
        let mut loaded: Vec<(usize, f64)> = Vec::new();
        for &c in &self.basis.chunks {
            let (f, m) = (world.loads.force[self.s][c], world.loads.torque[self.s][c]);
            if f == Vec3::ZERO && m == Vec3::ZERO {
                continue;
            }
            let (fb, mb) = (pose.inverse_transform_vector(f), pose.inverse_transform_vector(m));
            for d in 0..6 {
                if let Some(i) = self.basis.dof[c][d] {
                    loaded.push((i, comp(fb, mb, d)));
                }
            }
        }
        let contact = !loaded.is_empty();
        self.report.timings.contact += secs(t);

        let t = Instant::now();
        // The contact's modal input (the constant loads' `gres` is added below).
        let mut g_re = vec![0.0; np];
        let mut g_im = vec![0.0; np];
        for &(i, f) in &loaded {
            if let Some(r) = self.rows.iter().position(|&x| x == i) {
                let (a, b) = (&self.bin_rows.0[r * np..(r + 1) * np], &self.bin_rows.1[r * np..(r + 1) * np]);
                for j in 0..np {
                    g_re[j] += a[j] * f;
                    g_im[j] += b[j] * f;
                }
            } else {
                for j in 0..np {
                    g_re[j] += self.basis.bin_re[(i, j)] * f;
                    g_im[j] += self.basis.bin_im[(i, j)] * f;
                }
            }
        }
        let impulse = self.cfg.contact_impulse;
        for j in 0..np {
            let (e, p) = (self.e[j], self.p[j]);
            let (mut cr, mut ci) = (self.c_re[j], self.c_im[j]);
            let (mut gr, mut gi) = (self.gres_re[j], self.gres_im[j]);
            if impulse {
                cr += h * g_re[j];
                ci += h * g_im[j];
            } else {
                gr += g_re[j];
                gi += g_im[j];
            }
            self.c_re[j] = e.0 * cr - e.1 * ci + p.0 * gr - p.1 * gi;
            self.c_im[j] = e.0 * ci + e.1 * cr + p.0 * gi + p.1 * gr;
        }
        self.report.timings.modal += secs(t);

        let t = Instant::now();
        if !world.solver.clusters[self.ci].anchored {
            let loads = std::mem::take(&mut world.loads);
            world.solver.advance_rigid(self.ci, h, &loads);
            world.loads = loads;
        }
        let gravity = world.solver.config.gravity;
        for (imp, (f, tq)) in world.impactors.iter_mut().zip(imp_loads) {
            if imp.driven {
                continue;
            }
            imp.velocity += (f / imp.mass + gravity) * h;
            let r = imp.pose.rotation.to_mat3();
            let l = r * imp.inertia * r.transpose() * imp.angular_velocity + tq * h;
            let w_mid = (r * imp.inertia * r.transpose()).inverse().unwrap() * l;
            imp.pose.position += imp.velocity * h;
            imp.pose.rotation = imp.pose.rotation.integrate(w_mid, h);
            let r = imp.pose.rotation.to_mat3();
            imp.angular_velocity = (r * imp.inertia * r.transpose()).inverse().unwrap() * l;
        }
        world.solver.time += h;
        self.report.timings.bodies += secs(t);
        (g_re, g_im, contact)
    }

    fn snapshot(&self, world: &World, frame: u64, sub: usize, nsub: usize) -> Entry {
        let cl = &world.solver.clusters[self.ci];
        Entry {
            c_re: self.c_re.clone(),
            c_im: self.c_im.clone(),
            g_re: Vec::new(),
            g_im: Vec::new(),
            time: world.solver.time,
            dt: self.h,
            frame,
            sub,
            nsub,
            impactors: world.impactors.clone(),
            rigid: (cl.pose, cl.velocity, cl.angular_velocity),
            sticks: world.sticks.clone(),
            pair_offsets: world.pair_offsets.clone(),
            contact: world.contact,
            contact_force: false,
            displacement: 0.0,
        }
    }

    fn restore(&mut self, world: &mut World, e: &Entry) {
        self.c_re = e.c_re.clone();
        self.c_im = e.c_im.clone();
        world.solver.time = e.time;
        world.frame = e.frame;
        world.impactors = e.impactors.clone();
        let cl = &mut world.solver.clusters[self.ci];
        (cl.pose, cl.velocity, cl.angular_velocity) = e.rigid;
        world.sticks = e.sticks.clone();
        world.pair_offsets = e.pair_offsets.clone();
        world.contact = e.contact;
        self.set_step(e.dt);
    }

    /// The hidden value of every component of chunk `c` at block step `k` (`u` the
    /// `n x T` block of displacements from the reference).
    fn hidden(&self, sv: &crate::solver::ReferenceSolver, u: &Mat<f64>, c: usize, k: usize) -> (Vec3, Vec3) {
        let st = &sv.chunks[self.s][c];
        let (mut uu, mut th) = (st.u, st.th);
        for d in 0..6 {
            if let Some(i) = self.basis.dof[c][d] {
                let x = self.x0[i] + u[(i, k)];
                if d < 3 {
                    uu[d] = x;
                } else {
                    th[d - 3] = x;
                }
            }
        }
        (uu, th)
    }

    /// Check every bond at every step of the block with the joint law. Commits the
    /// bonds' history up to (not including) the first step at which a bond reaches the
    /// handover criterion, and returns that step and bond.
    fn check_block(&mut self, world: &mut World, blk: &[Entry]) -> Option<(usize, usize)> {
        let np = self.np;
        let tn = blk.len();
        let t = Instant::now();
        let cre = Mat::<f64>::from_fn(np, tn, |j, k| blk[k].c_re[j]);
        let cim = Mat::<f64>::from_fn(np, tn, |j, k| blk[k].c_im[j]);
        let u = &self.basis.psi_re * &cre - &self.basis.psi_im * &cim;
        self.report.timings.checks_gemm += secs(t);

        let t = Instant::now();
        let bonds = world.solver.clusters[self.ci].bonds.clone();
        let fracture = world.solver.config.fracture;
        let corot = world.solver.config.methods.corotational_bonds;
        let threshold = self.cfg.handover_utilization;
        let this = &*self;
        let wref = &world.solver;
        // The law over `upto` steps from the bond's committed state: first onset step (or
        // `upto`), the state before it and the last committed response.
        // Also the largest failure index committed (every step, as the explicit solver's
        // `max_utilization`).
        let run = |bi: usize, upto: usize| -> (usize, JointState, Option<JointResponse>, f64) {
            let b = &wref.bonds[this.s][bi];
            let g = &b.geometry;
            let model = b.model();
            let mut st = b.joint.clone();
            let mut last = None;
            let mut peak = 0.0f64;
            for k in 0..upto {
                let (ua, tha) = this.hidden(wref, &u, g.a, k);
                let (ub, thb) = this.hidden(wref, &u, g.b, k);
                let d = if corot { g.corotation_with(ua, &ChunkTurn::new(tha), ub, &ChunkTurn::new(thb)).deformation() } else { g.kinematics(ua, tha, ub, thb) };
                let r = model.evaluate(&st, &d, blk[k].dt, fracture);
                let onset = r.state.damage > 0.0 || r.state.crush > 0.0 || r.disconnected || r.state.utilization >= threshold;
                if onset {
                    return (k, st, last, peak);
                }
                peak = peak.max(r.state.utilization);
                st = r.state.clone();
                last = Some(r);
            }
            (upto, st, last, peak)
        };
        let first: Vec<(usize, JointState, Option<JointResponse>, f64)> =
            if self.cfg.parallel { bonds.par_iter().map(|&bi| run(bi, tn)).collect() } else { bonds.iter().map(|&bi| run(bi, tn)).collect() };
        let kstar = first.iter().map(|r| r.0).min().unwrap_or(tn);
        let results = if kstar < tn {
            if self.cfg.parallel {
                bonds.par_iter().map(|&bi| run(bi, kstar)).collect()
            } else {
                bonds.iter().map(|&bi| run(bi, kstar)).collect()
            }
        } else {
            first.clone()
        };
        for (k, &bi) in bonds.iter().enumerate() {
            let (_, st, last, peak) = &results[k];
            self.report.max_utilization = self.report.max_utilization.max(*peak);
            world.solver.max_utilization = world.solver.max_utilization.max(*peak);
            let bm = &mut world.solver.bonds[self.s][bi];
            bm.joint = st.clone();
            if let Some(r) = last {
                bm.force = r.force;
                bm.elastic = r.force;
                bm.measures = r.measures;
                bm.stored = r.stored;
            }
        }
        self.report.timings.checks_law += secs(t);
        if kstar < tn {
            let bond = bonds.iter().zip(&first).find(|(_, r)| r.0 == kstar).map(|(&b, _)| b).unwrap();
            Some((kstar, bond))
        } else {
            None
        }
    }

    /// The time in `[t0, t0 + h]` at which bond `bi`'s failure index (rate factor and
    /// fatigue frozen at the bond's committed state) first reaches the threshold, from
    /// the closed-form trajectory of the step that starts in `e`.
    fn crossing(&self, world: &World, e: &Entry, bi: usize) -> f64 {
        let b = &world.solver.bonds[self.s][bi];
        let g = &b.geometry;
        let model = b.model();
        let st = &b.joint;
        let s = model.strength;
        let dif = s.dif.map(|p| p.factor(st.strain_rate)).unwrap_or(1.0);
        let mult = model.weibull * dif * fatigue_factor(s, st);
        let k = b.stiffness.as_local();
        let lam = &self.basis.lambda;
        let index_at = |tau: f64| -> f64 {
            let mut x = vec![0.0; 12];
            let dofs: [Option<usize>; 12] = std::array::from_fn(|x| if x < 6 { self.basis.dof[g.a][x] } else { self.basis.dof[g.b][x - 6] });
            for (slot, d) in dofs.iter().enumerate() {
                if let Some(i) = *d {
                    let mut v = 0.0;
                    for j in 0..self.np {
                        let (ee, pp) = step_coefficients(lam[j], tau);
                        let (mut c0r, mut c0i) = (e.c_re[j], e.c_im[j]);
                        let (mut gr, mut gi) = (self.gres_re[j], self.gres_im[j]);
                        if self.cfg.contact_impulse {
                            c0r += e.dt * e.g_re[j];
                            c0i += e.dt * e.g_im[j];
                        } else {
                            gr += e.g_re[j];
                            gi += e.g_im[j];
                        }
                        let cr = ee.0 * c0r - ee.1 * c0i + pp.0 * gr - pp.1 * gi;
                        let ci = ee.0 * c0i + ee.1 * c0r + pp.0 * gi + pp.1 * gr;
                        v += self.basis.psi_re[(i, j)] * cr - self.basis.psi_im[(i, j)] * ci;
                    }
                    x[slot] = self.x0[i] + v;
                }
            }
            let chunk = |c: usize, off: usize| -> (Vec3, Vec3) {
                let stc = &world.solver.chunks[self.s][c];
                let (mut u, mut th) = (stc.u, stc.th);
                for d in 0..6 {
                    if dofs[off + d].is_some() {
                        if d < 3 {
                            u[d] = x[off + d];
                        } else {
                            th[d - 3] = x[off + d];
                        }
                    }
                }
                (u, th)
            };
            let (ua, tha) = chunk(g.a, 0);
            let (ub, thb) = chunk(g.b, 6);
            let d = g.kinematics(ua, tha, ub, thb);
            failure_indices(s, &stress_measures(g, &d.mul_elem(&k)), mult).max()
        };
        let target = self.cfg.handover_utilization.min(1.0);
        let (mut lo, mut hi) = (0.0, e.dt);
        if index_at(hi) < target {
            return e.time + e.dt;
        }
        if index_at(lo) >= target {
            return e.time;
        }
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if index_at(mid) >= target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        e.time + hi
    }

    /// The no-damage certificate (see the module documentation): `(holds, worst bound)`.
    fn certificate(&mut self, world: &World, t_end: f64) -> (bool, f64) {
        let np = self.np;
        let n = self.basis.n;
        let lam = &self.basis.lambda;
        // Steady state under the constant loads, and the amplitudes about it.
        let mut ss = (vec![0.0; np], vec![0.0; np]);
        let mut amp = vec![0.0; np];
        for j in 0..np {
            let (a, b) = lam[j];
            let den = a * a + b * b;
            // -g / lambda
            let (gr, gi) = (self.gres_re[j], self.gres_im[j]);
            let q = (-(gr * a + gi * b) / den, -(gi * a - gr * b) / den);
            ss.0[j] = q.0;
            ss.1[j] = q.1;
            amp[j] = (self.c_re[j] - q.0).hypot(self.c_im[j] - q.1);
        }
        let (xss, _) = self.basis.state((&ss.0, &ss.1));
        let (pr, pi) = (&self.basis.psi_re, &self.basis.psi_im);
        // Per-DOF displacement amplitude.
        let dof_amp = par_rows(self.cfg.parallel, n, |i| (0..np).map(|j| pr[(i, j)].hypot(pi[(i, j)]) * amp[j]).sum());
        let sv = &world.solver;
        let s = self.s;
        let remaining = (t_end - sv.time).max(0.0);
        let bonds = sv.clusters[self.ci].bonds.clone();
        let this = &*self;
        let bound = |bi: usize| -> f64 {
            let b = &sv.bonds[s][bi];
            let g = &b.geometry;
            let rows = bond_rows(g);
            let dofs: [Option<usize>; 12] = std::array::from_fn(|x| if x < 6 { this.basis.dof[g.a][x] } else { this.basis.dof[g.b][x - 6] });
            // Centre: the deformation at the reference plus the steady state.
            let base = |c: usize, off: usize| -> (Vec3, Vec3) {
                let stc = &sv.chunks[s][c];
                let (mut u, mut th) = (stc.u, stc.th);
                for d in 0..6 {
                    if let Some(i) = dofs[off + d] {
                        let x = this.x0[i] + xss[i];
                        if d < 3 {
                            u[d] = x;
                        } else {
                            th[d - 3] = x;
                        }
                    }
                }
                (u, th)
            };
            let (ua, tha) = base(g.a, 0);
            let (ub, thb) = base(g.b, 6);
            let d0 = local6(&g.kinematics(ua, tha, ub, thb));
            let mut half = [0.0f64; 6];
            for j in 0..np {
                let mut dr = [0.0f64; 6];
                let mut di = [0.0f64; 6];
                for x in 0..12 {
                    if let Some(i) = dofs[x] {
                        let (a, bb) = (pr[(i, j)], pi[(i, j)]);
                        for q in 0..6 {
                            dr[q] += rows[q][x] * a;
                            di[q] += rows[q][x] * bb;
                        }
                    }
                }
                for q in 0..6 {
                    half[q] += dr[q].hypot(di[q]) * amp[j];
                }
            }
            let k6 = local6(&b.stiffness.as_local());
            let q0: [f64; 6] = std::array::from_fn(|q| k6[q] * d0[q]);
            let dq: [f64; 6] = std::array::from_fn(|q| k6[q] * half[q]);
            let absmax = |q: usize| q0[q].abs() + dq[q];
            let (n_hi, n_lo) = (q0[2] + dq[2], q0[2] - dq[2]);
            let bending = absmax(3) / g.s_t1 + absmax(4) / g.s_t2;
            let tension = n_hi / g.area + bending;
            let shear = absmax(0).hypot(absmax(1)) / g.area + absmax(5) / g.torsion_modulus;
            let compression = -n_lo / g.area + bending;
            let sigma_c = (-n_hi / g.area).max(0.0);
            let cforce = (-n_lo).max(0.0);
            let sth = &b.strength;
            // Fatigue at its worst over the remaining time (rate factor >= 1 left out).
            let mut fat = fatigue_factor(sth, &b.joint);
            if let Some(f) = sth.static_fatigue {
                let ratio = (tension / (sth.tensile * b.weibull)).max(shear / (sth.cohesion * b.weibull).min(sth.shear_cap * b.weibull)).max(compression / (sth.compressive * b.weibull));
                let omega = (b.joint.fatigue + f.life_rate(ratio.max(0.0)) * remaining).min(1.0);
                fat = f.strength_factor(omega);
            }
            let m = b.weibull * fat;
            let ft = sth.tensile * m;
            let fc = sth.compressive * m;
            let tau = (sth.cohesion * m + sth.friction * sigma_c).min(sth.shear_cap * m);
            let mut idx = (tension / ft).max(0.0).max(compression / fc).max(if tau > 0.0 { shear / tau } else { f64::INFINITY });
            if let Some(p) = sth.buckling_load {
                idx = idx.max(cforce / p);
            }
            idx
        };
        let worst = if self.cfg.parallel { bonds.par_iter().map(|&bi| bound(bi)).reduce(|| 0.0, f64::max) } else { bonds.iter().map(|&bi| bound(bi)).fold(0.0, f64::max) };
        if worst > self.cfg.certificate_threshold {
            return (false, worst);
        }
        // The impactors' ballistic paths must stay clear of every chunk (swept test).
        let st = &sv.structures[s];
        let pose = sv.clusters[self.ci].pose;
        let margin = self.cfg.contact_margin;
        let chunk_boxes: Vec<(OBox, f64)> = this
            .basis
            .chunks
            .iter()
            .map(|&c| {
                let ch = &st.chunks[c];
                let stc = &sv.chunks[s][c];
                let mut off = [0.0f64; 6];
                let mut a6 = [0.0f64; 6];
                for d in 0..6 {
                    let v = comp(stc.u, stc.th, d);
                    match this.basis.dof[c][d] {
                        Some(i) => {
                            off[d] = this.x0[i] + xss[i];
                            a6[d] = dof_amp[i];
                        }
                        None => off[d] = v,
                    }
                }
                let du = Vec3::new(off[0].abs() + a6[0], off[1].abs() + a6[1], off[2].abs() + a6[2]).norm();
                let dth = Vec3::new(off[3].abs() + a6[3], off[4].abs() + a6[4], off[5].abs() + a6[5]).norm();
                let grow = du + dth * ch.half_extents.norm() + margin;
                let b = OBox { center: pose.transform_point(ch.center), rotation: pose.rotation.to_mat3() * ch.rotation, half: ch.half_extents + Vec3::splat(grow), hull: None, parts: None };
                let r = b.bounding_radius();
                (b, r)
            })
            .collect();
        let gravity = sv.config.gravity;
        for imp in &world.impactors {
            if imp.driven {
                continue;
            }
            let ib0 = imp.obox();
            let radius = ib0.bounding_radius();
            let vmax = imp.velocity.norm() + gravity.norm() * remaining + imp.angular_velocity.norm() * radius;
            let dtau = if vmax > 0.0 { (0.5 * margin / vmax).min(remaining.max(1e-12)) } else { remaining.max(1e-12) };
            let samples = ((remaining / dtau).ceil() as usize).min(5_000_000);
            for k in 0..=samples {
                let tau = (k as f64 * dtau).min(remaining);
                let pos = imp.pose.position + imp.velocity * tau + gravity * (0.5 * tau * tau);
                let rot = Quat::from_rotation_vector(imp.angular_velocity * tau) * imp.pose.rotation;
                let ib = OBox { center: pos, rotation: rot.to_mat3(), half: ib0.half + Vec3::splat(margin), hull: None, parts: None };
                let reach = ib.bounding_radius();
                for (b, r) in &chunk_boxes {
                    if (b.center - ib.center).norm() <= reach + r && b.may_overlap(&ib) {
                        return (false, worst);
                    }
                }
            }
        }
        (true, worst)
    }
}

/// Run the modal pre-window on `world` from its current state (a frame boundary) to the
/// first damage onset, a no-damage certificate, or the scene's end. On onset the world
/// is handed over to the explicit solver at the start of that step and the frame is
/// finished explicitly: the caller continues with `step_frame`.
pub fn run(world: &mut World, basis: &ModalBasis, cfg: &ModalConfig) -> ModalResult<WindowReport> {
    let t_total = Instant::now();
    let t_setup = Instant::now();
    if world.scene.sim.solve_mode != SolveMode::Explicit {
        return Err("the modal pre-window runs in explicit mode only".into());
    }
    if world.coupling.is_some() || !world.scene.events.is_empty() || world.scene.sim.refine_utilization.is_some() {
        return Err("engine coupling, scripted events and refinement are not supported".into());
    }
    let bonded: Vec<usize> = (0..world.solver.clusters.len()).filter(|&c| !world.solver.clusters[c].bonds.is_empty()).collect();
    if world.solver.clusters.len() != 1 || bonded.len() != 1 {
        return Err("the prototype takes one cluster in the world".into());
    }
    let ci = bonded[0];
    if !basis.matches(world, ci) {
        return Err("the basis was built for another cluster".into());
    }
    // Refuse a cluster that is no longer intact.
    Assembly::check_intact(world, ci)?;
    let s = basis.structure;
    let n = basis.n;
    let np = basis.modes();
    let sv = &world.solver;
    let mut x0 = vec![0.0; n];
    let mut mv0 = vec![0.0; n];
    for &c in &basis.chunks {
        let stc = &sv.chunks[s][c];
        let ch = &sv.structures[s].chunks[c];
        let mu = stc.inertia_scale;
        let p = ch.mass * mu * stc.v;
        let l = ch.inertia * stc.w * mu;
        for d in 0..6 {
            if let Some(i) = basis.dof[c][d] {
                x0[i] = comp(stc.u, stc.th, d);
                mv0[i] = comp(p, l, d);
            }
        }
    }
    // The constant loads: gravity (body frame) and the bonds' elastic forces at the start.
    let mut fres = vec![0.0; n];
    let cl = &sv.clusters[ci];
    let pose = cl.pose;
    let gravity_body = pose.inverse_transform_vector(sv.config.gravity);
    for &c in &basis.chunks {
        let ch = &sv.structures[s].chunks[c];
        let f = gravity_body * ch.mass;
        for d in 0..3 {
            if let Some(i) = basis.dof[c][d] {
                fres[i] += f[d];
            }
        }
    }
    for &bi in &cl.bonds {
        let b = &sv.bonds[s][bi];
        let g = &b.geometry;
        let (sa, sb) = (&sv.chunks[s][g.a], &sv.chunks[s][g.b]);
        let q = g.kinematics(sa.u, sa.th, sb.u, sb.th).mul_elem(&b.stiffness.as_local());
        let (fa, ma, fb, mb) = g.chunk_loads(&q);
        for d in 0..6 {
            if let Some(i) = basis.dof[g.a][d] {
                fres[i] += comp(fa, ma, d);
            }
            if let Some(i) = basis.dof[g.b][d] {
                fres[i] += comp(fb, mb, d);
            }
        }
    }
    let (gres_re, gres_im) = basis.input(&fres);
    let (c_re, c_im) = basis.input(&mv0);
    let mut win = Window {
        basis,
        cfg,
        ci,
        s,
        np,
        x0,
        c_re,
        c_im,
        gres_re,
        gres_im,
        h: f64::NAN,
        e: Vec::new(),
        p: Vec::new(),
        cand: vec![usize::MAX],
        rows: Vec::new(),
        psi_rows: (Vec::new(), Vec::new()),
        bin_rows: (Vec::new(), Vec::new()),
        report: WindowReport {
            verdict: Verdict::NoOnset { time: 0.0 },
            steps: 0,
            contact_steps: 0,
            first_contact: None,
            max_utilization: 0.0,
            max_candidates: 0,
            max_displacement: 0.0,
            certificate_attempts: 0,
            last_certificate_bound: f64::NAN,
            end_state: None,
            timings: WindowTimings::default(),
        },
        step_displacement: 0.0,
    };
    win.report.timings.setup = secs(t_setup);

    let frame_dt = world.scene.sim.frame_dt;
    let frames = (world.scene.sim.duration / frame_dt).round() as u64;
    let t_end = frames as f64 * frame_dt;
    let can_certify = cfg.certify && basis.anchored && world.scene.ground.is_none();
    let mut blk: Vec<Entry> = Vec::with_capacity(cfg.block);
    let mut prev: Option<Entry> = None;
    let mut quiet = 0usize;
    let mut next_attempt = 0.0f64;
    while world.frame < frames {
        let dt = world.substep_dt();
        let nsub = (frame_dt / dt).round().max(1.0) as usize;
        win.set_step(dt);
        world.frame_loads.resize(&world.solver);
        let frame = world.frame;
        for sub in 0..nsub {
            let t = Instant::now();
            let mut e = win.snapshot(world, frame, sub, nsub);
            win.report.timings.bodies += secs(t);
            win.step_displacement = 0.0;
            let (g_re, g_im, contact) = win.step(world);
            e.g_re = g_re;
            e.g_im = g_im;
            e.contact_force = contact;
            e.displacement = win.step_displacement;
            blk.push(e);
            win.report.steps += 1;
            if contact {
                quiet = 0;
            } else {
                quiet += 1;
            }
            let last_of_run = sub + 1 == nsub && frame + 1 == frames;
            if blk.len() == cfg.block || last_of_run {
                if let Some((k, bond)) = win.check_block(world, &blk) {
                    // Roll back to the start of step k and hand over.
                    let t = Instant::now();
                    let crossing = {
                        let before = if k > 0 { Some(&blk[k - 1]) } else { prev.as_ref() };
                        match before {
                            Some(e) => win.crossing(world, e, bond),
                            None => blk[k].time,
                        }
                    };
                    let previous = (if k > 0 { Some(&blk[k - 1]) } else { prev.as_ref() }).map(|p| (p.c_re.clone(), p.c_im.clone(), p.dt));
                    win.report.book(&blk[..k]);
                    win.report.max_displacement = win.report.max_displacement.max(blk[k].displacement);
                    let e = blk.swap_remove(k);
                    win.restore(world, &e);
                    win.write_all(world, previous.as_ref().map(|p| (&p.0[..], &p.1[..], p.2)));
                    win.report.end_state = Some(WorldState::of(world, win.s));
                    win.report.steps -= (blk.len() + 1 - k) as u64;
                    win.report.verdict = Verdict::Handover { time: e.time, bond, crossing };
                    win.report.timings.handover += secs(t);
                    // The window's own time ends here; the rest of the frame is explicit.
                    win.report.timings.total = secs(t_total);
                    world.finish_frame_explicit(e.sub, e.nsub, e.dt);
                    return Ok(win.report);
                }
                win.report.book(&blk);
                prev = blk.pop();
                blk.clear();
                // Certificate once the contact has stopped.
                if can_certify && quiet >= cfg.quiet_steps && world.solver.time >= next_attempt {
                    let t = Instant::now();
                    let (holds, bound) = win.certificate(world, t_end);
                    win.report.certificate_attempts += 1;
                    win.report.last_certificate_bound = bound;
                    next_attempt = world.solver.time + cfg.certificate_interval;
                    win.report.timings.certificate += secs(t);
                    if holds {
                        win.write_all(world, None);
                        win.report.end_state = Some(WorldState::of(world, win.s));
                        win.report.verdict = Verdict::Certified { time: world.solver.time, bound };
                        win.report.timings.total = secs(t_total);
                        return Ok(win.report);
                    }
                }
            }
        }
        world.frame += 1;
        world.frame_loads.clear();
    }
    win.write_all(world, None);
    win.report.end_state = Some(WorldState::of(world, win.s));
    win.report.verdict = Verdict::NoOnset { time: world.solver.time };
    win.report.timings.total = secs(t_total);
    Ok(win.report)
}

impl Assembly {
    fn check_intact(world: &World, ci: usize) -> ModalResult<()> {
        let sv = &world.solver;
        let cl = &sv.clusters[ci];
        for &bi in &cl.bonds {
            let b = &sv.bonds[cl.structure][bi];
            if b.alive && (b.joint.damage != 0.0 || b.joint.crush != 0.0) {
                return Err(format!("bond {bi} is damaged"));
            }
        }
        Ok(())
    }
}

/// Step `world` explicitly, substep by substep exactly as `step_frame` does, until its
/// time reaches `until` (at a substep boundary); then finish that frame. Returns the
/// substeps and wall-clock seconds up to `until` (the span a pre-window would replace),
/// so a modal window and the explicit solver can be timed over the same span.
pub fn explicit_until(world: &mut World, until: f64) -> (u64, f64, WorldState) {
    assert_eq!(world.scene.sim.solve_mode, SolveMode::Explicit);
    let t0 = Instant::now();
    let start = world.solver.substeps;
    let frame_dt = world.scene.sim.frame_dt;
    let frames = (world.scene.sim.duration / frame_dt).round() as u64;
    while world.frame < frames {
        let dt = world.substep_dt();
        let n = (frame_dt / dt).round().max(1.0) as usize;
        world.frame_loads.resize(&world.solver);
        let splits_before = world.solver.events.len();
        for i in 0..n {
            if world.solver.time >= until - 0.5 * dt {
                let span = (world.solver.substeps - start, secs(t0), WorldState::of(world, 0));
                world.finish_frame_explicit(i, n, dt);
                return span;
            }
            world.substep(dt);
            let mut fl = std::mem::take(&mut world.frame_loads);
            fl.add_scaled(&world.loads, 1.0 / n as f64);
            world.frame_loads = fl;
        }
        world.frame += 1;
        if world.solver.events.len() != splits_before {
            world.topology_version += 1;
        }
        world.frame_loads.clear();
    }
    (world.solver.substeps - start, secs(t0), WorldState::of(world, 0))
}

/// Hidden chunk states of one structure (body frame) and the impactors' motion.
#[derive(Clone, Debug)]
pub struct WorldState {
    pub time: f64,
    /// `(u, th, v, w)` per chunk.
    pub chunks: Vec<[Vec3; 4]>,
    pub impactors: Vec<(Vec3, Vec3)>,
}

impl WorldState {
    pub fn of(world: &World, s: usize) -> WorldState {
        WorldState {
            time: world.solver.time,
            chunks: world.solver.chunks[s].iter().map(|c| [c.u, c.th, c.v, c.w]).collect(),
            impactors: world.impactors.iter().map(|i| (i.pose.position, i.velocity)).collect(),
        }
    }

    /// Largest difference over the largest magnitude of the hidden displacements and of
    /// the hidden velocities (translations; rotations scaled by `length`), and the
    /// largest impactor velocity difference.
    pub fn compare(&self, other: &WorldState, length: f64) -> (f64, f64, f64) {
        let (mut dx, mut x, mut dv, mut v) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (a, b) in self.chunks.iter().zip(&other.chunks) {
            for k in 0..3 {
                dx = dx.max((a[0][k] - b[0][k]).abs()).max(length * (a[1][k] - b[1][k]).abs());
                x = x.max(a[0][k].abs()).max(length * a[1][k].abs());
                dv = dv.max((a[2][k] - b[2][k]).abs()).max(length * (a[3][k] - b[3][k]).abs());
                v = v.max(a[2][k].abs()).max(length * a[3][k].abs());
            }
        }
        let di = self.impactors.iter().zip(&other.impactors).map(|(a, b)| (a.1 - b.1).norm()).fold(0.0, f64::max);
        (dx / x.max(1e-300), dv / v.max(1e-300), di)
    }
}

/// Dense `(M, C, K)` of cluster `ci` (physical coordinates) and its DOF map, for tests.
pub fn assemble(world: &World, ci: usize) -> ModalResult<Assembly> {
    Assembly::new(world, ci)
}

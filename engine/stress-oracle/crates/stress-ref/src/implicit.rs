//! Implicit dynamics: one Newmark average-acceleration step per cluster and frame
//! (`SolveMode::Implicit`).
//!
//! The explicit central-difference solve (`SolveMode::Explicit`) is the ground truth
//! but needs substeps below the bond network's highest period. The implicit step keeps
//! inertia, damping and true stiffness at the frame step instead: with Newmark's
//! `beta = 1/4`, `gamma = 1/2` (trapezoidal rule, unconditionally stable and free of
//! numerical damping for linear systems) the hidden displacements `u` solve
//!
//! `M a(u) + C v(u) + f_int(u) = f_ext`,
//! `a = (u - u_n - dt v_n) / (beta dt^2) - (1/(2 beta) - 1) a_n`,
//! `v = v_n + dt ((1 - gamma) a_n + gamma a)`,
//!
//! by modified Newton on the true residual. Each correction solves
//! `(K + gamma/(beta dt) C + M/(beta dt^2)) du = r` with the same block-Jacobi
//! preconditioned conjugate gradient as the static solve (`statics.rs`); the mass term
//! makes the operator positive definite for free clusters too, so no inertia relief
//! projection is needed. It is the static solve plus one diagonal term.
//!
//! **Coupling with substep contacts.** The world resolves contacts at its (contact
//! limited) substep, between implicit steps. A struck chunk must recede during that
//! time or the penalty contact keeps building up, so between steps every hidden DOF
//! follows a *predictor* driven by the external loads alone (`implicit_predict`):
//! contact impulses become velocity kicks on the struck chunks, the momentum-exact
//! "impact as an initial velocity condition". The implicit step then restarts from the
//! state at the end of the previous step with the average loads of the interval, i.e.
//! the same impulse, and replaces the predictor by the solution with internal forces.
//!
//! Damage is held during the iterations and committed at the converged state (the
//! same as the quasi-static solve), so a cascade of failures spreads over steps, as it
//! physically takes time. What the large step cannot represent are wave fronts sharper
//! than `c dt` (spalling); [`SolveMode::Explicit`] remains the reference for those, and
//! the comparison between the two modes is what this mode is for.
//!
//! [`SolveMode::Explicit`]: crate::scene::SolveMode::Explicit

use crate::bond::{Local6, Mat6};
use crate::math::{Mat3, Vec3};
use crate::solver::{ChunkLoads, ReferenceSolver};
use crate::statics::{dot6, join, pcg_solve, split, Block6, Layout, V6};

/// Newmark parameters of the average-acceleration (trapezoidal) rule.
pub const NEWMARK_BETA: f64 = 0.25;
pub const NEWMARK_GAMMA: f64 = 0.5;

/// Outcome of one implicit step of a cluster.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImplicitReport {
    pub newton_iterations: usize,
    pub cg_iterations: usize,
    /// Final residual relative to the step's load scale.
    pub residual: f64,
    pub converged: bool,
}

/// Halvings of an implicit step whose Newton iteration does not converge (down to
/// 1/64 of it); a step still unconverged there is kept as the best state.
const STEP_HALVINGS: usize = 6;

/// Relative residual at which a step's Newton iteration stops.
pub const IMPLICIT_TOLERANCE: f64 = 1e-9;
const MAX_NEWTON: usize = 60;

fn mass_block(mass: f64, inertia: &Mat3, scale: f64) -> Block6 {
    let mut b = Block6::zero();
    for i in 0..3 {
        b.0[i][i] = mass * scale;
        for j in 0..3 {
            b.0[3 + i][3 + j] = inertia.m[i][j] * scale;
        }
    }
    b
}

impl ReferenceSolver {
    /// Record every active chunk's hidden state as the start of the next implicit step.
    pub fn mark_implicit_step_start(&mut self) {
        for chunks in &mut self.chunks {
            for st in chunks.iter_mut() {
                st.step_start = [st.u, st.th, st.v, st.w];
            }
        }
    }

    /// Predictor between implicit steps: hidden DOFs move under the external loads
    /// only (in the cluster frame, rigid inertial loads subtracted).
    pub(crate) fn implicit_predict(&mut self, ci: usize, dt: f64, loads: &ChunkLoads, t: f64) {
        if self.clusters[ci].bonds.is_empty() {
            return;
        }
        let (a, alpha) = self.rigid_acceleration(ci, loads);
        let s = self.clusters[ci].structure;
        for c in self.clusters[ci].chunks.clone() {
            let (f, m) = self.frame_loads(ci, c, loads, a, alpha, t);
            let ch = &self.structures[s].chunks[c];
            if ch.support != crate::scene::Support::None {
                continue;
            }
            let st = &mut self.chunks[s][c];
            st.v += f * (dt / (ch.mass * st.inertia_scale));
            st.w += ch.inv_inertia * m * (dt / st.inertia_scale);
            st.u += st.v * dt;
            st.th += st.w * dt;
        }
    }

    /// Advance every cluster's deformation by one implicit step of `dt` under the
    /// (time-averaged) loads, commit damage and split what disconnected.
    pub fn implicit_step_all(&mut self, loads: &ChunkLoads, dt: f64) -> ImplicitReport {
        let mut total = ImplicitReport { converged: true, ..Default::default() };
        let mut to_split = Vec::new();
        for ci in 0..self.clusters.len() {
            // A cluster in an explicit window was integrated by the substeps.
            if self.clusters[ci].bonds.is_empty() || self.clusters[ci].explicit_window {
                continue;
            }
            let r = self.newmark_step_subdivided(ci, loads, dt, STEP_HALVINGS);
            // Fold the hidden field's rigid motion into the frame before any split, so
            // children inherit each momentum once.
            if !self.clusters[ci].anchored {
                self.remove_rigid_drift(ci);
            }
            total.newton_iterations += r.newton_iterations;
            total.cg_iterations += r.cg_iterations;
            total.residual = total.residual.max(r.residual);
            total.converged &= r.converged;
            let (_, disconnected) = self.commit_damage(ci, dt);
            if disconnected {
                to_split.push(ci);
            }
        }
        for &ci in to_split.iter().rev() {
            self.split_cluster(ci);
        }
        self.mark_implicit_step_start();
        total
    }

    /// Damping forces on the cluster's chunks at the current hidden velocities, and
    /// the per-bond dashpot coefficients (secant-scaled) for the tangent.
    fn damping_forces(&self, ci: usize, lay: &Layout) -> (Vec<V6>, Vec<Local6>, f64) {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let mut out = vec![[0.0; 6]; lay.chunks.len()];
        let mut coeffs = Vec::with_capacity(cl.bonds.len());
        let mut power = 0.0;
        for &bi in &cl.bonds {
            let b = &self.bonds[s][bi];
            let g = &b.geometry;
            let (sa, sb) = (&self.chunks[s][g.a], &self.chunks[s][g.b]);
            let d = g.kinematics(sa.u, sa.th, sb.u, sb.th);
            let rate = g.kinematics(sa.v, sa.w, sb.v, sb.w);
            let c = b.damping.mul_elem(&b.model().secant_factors(&b.joint, &d));
            let q = rate.mul_elem(&c);
            power += q.dot(&rate);
            let (fa, ma, fb, mb) = g.chunk_loads(&q);
            let (ia, ib) = (lay.local[g.a], lay.local[g.b]);
            for k in 0..3 {
                out[ia][k] += fa[k];
                out[ia][3 + k] += ma[k];
                out[ib][k] += fb[k];
                out[ib][3 + k] += mb[k];
            }
            coeffs.push(c);
        }
        // `rayleigh_damping`: the mass-proportional dashpot on the hidden motion.
        if (cl.anchored || self.config.methods.corotational_bonds) && self.mass_damping.get(s).copied().unwrap_or(0.0) > 0.0 {
            for (&c, (fd, md)) in cl.chunks.iter().zip(self.mass_damping_loads(ci)) {
                let st = &self.chunks[s][c];
                power -= fd.dot(st.v) + md.dot(st.w);
                let i = lay.local[c];
                for k in 0..3 {
                    out[i][k] += fd[k];
                    out[i][3 + k] += md[k];
                }
            }
        }
        (out, coeffs, power)
    }

    /// `newmark_step`, rejecting a step whose Newton iteration does not converge: the
    /// cluster goes back to the step's start and takes two half steps under the same
    /// (time-averaged) loads, each subdivided the same way, at most `halvings` deep.
    /// Committing an unconverged step commits a state that is no solution of the
    /// step's equations of motion (a 0.3 m brick wall struck at 15 m/s accepted relative
    /// residuals near 1 and reached chunks at 1e18 m/s); a step that converges is
    /// unchanged.
    fn newmark_step_subdivided(&mut self, ci: usize, loads: &ChunkLoads, dt: f64, halvings: usize) -> ImplicitReport {
        let s = self.clusters[ci].structure;
        let chunks = self.clusters[ci].chunks.clone();
        let saved: Vec<([Vec3; 4], Vec3, Vec3)> = chunks.iter().map(|&c| (self.chunks[s][c].step_start, self.chunks[s][c].a, self.chunks[s][c].alpha)).collect();
        let damping = self.energy.damping_dissipation;
        let r = self.newmark_step(ci, loads, dt);
        if r.converged || halvings == 0 {
            return r;
        }
        for (&c, (start, a, alpha)) in chunks.iter().zip(&saved) {
            let st = &mut self.chunks[s][c];
            [st.u, st.th, st.v, st.w] = *start;
            (st.a, st.alpha) = (*a, *alpha);
        }
        self.energy.damping_dissipation = damping;
        let first = self.newmark_step_subdivided(ci, loads, 0.5 * dt, halvings - 1);
        for &c in &chunks {
            let st = &mut self.chunks[s][c];
            st.step_start = [st.u, st.th, st.v, st.w];
        }
        let second = self.newmark_step_subdivided(ci, loads, 0.5 * dt, halvings - 1);
        ImplicitReport {
            newton_iterations: r.newton_iterations + first.newton_iterations + second.newton_iterations,
            cg_iterations: r.cg_iterations + first.cg_iterations + second.cg_iterations,
            residual: first.residual.max(second.residual),
            converged: first.converged && second.converged,
        }
    }

    /// One Newmark step of cluster `ci` (damage frozen; see the module docs).
    pub fn newmark_step(&mut self, ci: usize, loads: &ChunkLoads, dt: f64) -> ImplicitReport {
        let (beta, gamma) = (NEWMARK_BETA, NEWMARK_GAMMA);
        let (c_m, c_v) = (1.0 / (beta * dt * dt), gamma / (beta * dt));
        let lay = self.layout(ci);
        let s = self.clusters[ci].structure;
        let f_ext = self.static_loads(ci, &lay, loads);
        let mass: Vec<Block6> = lay
            .chunks
            .iter()
            .map(|&c| {
                let ch = &self.structures[s].chunks[c];
                mass_block(ch.mass, &ch.inertia, 1.0)
            })
            .collect();
        let state = |solver: &ReferenceSolver, c: usize| {
            let st = &solver.chunks[s][c];
            let [u, th, v, w] = st.step_start;
            (join(u, th), join(v, w), join(st.a, st.alpha))
        };
        let start: Vec<(V6, V6, V6)> = lay.chunks.iter().map(|&c| state(self, c)).collect();
        let held = |i: usize, d: usize| lay.fixed[i][d];

        // Accelerations and velocities implied by a trial displacement field.
        let kinematics = |x: &[V6]| -> (Vec<V6>, Vec<V6>) {
            let mut acc = Vec::with_capacity(x.len());
            let mut vel = Vec::with_capacity(x.len());
            for (i, (un, vn, an)) in start.iter().enumerate() {
                let a: V6 = std::array::from_fn(|d| {
                    if held(i, d) {
                        0.0
                    } else {
                        c_m * (x[i][d] - un[d] - dt * vn[d]) - (0.5 / beta - 1.0) * an[d]
                    }
                });
                let v: V6 = std::array::from_fn(|d| if held(i, d) { 0.0 } else { vn[d] + dt * ((1.0 - gamma) * an[d] + gamma * a[d]) });
                acc.push(a);
                vel.push(v);
            }
            (acc, vel)
        };

        // Residual of a trial displacement field (sets the trial state on the chunks).
        let residual = |solver: &mut ReferenceSolver, x: &[V6]| -> (Vec<V6>, Vec<Local6>) {
            let (acc, vel) = kinematics(x);
            for (i, &c) in lay.chunks.iter().enumerate() {
                let st = &mut solver.chunks[s][c];
                (st.u, st.th) = split(&x[i]);
                (st.v, st.w) = split(&vel[i]);
            }
            let fb = solver.bond_forces(ci, &lay);
            let (fd, damping, _) = solver.damping_forces(ci, &lay);
            let r = (0..x.len())
                .map(|i| {
                    let ma = mass[i].mul(&acc[i]);
                    std::array::from_fn(|d| if held(i, d) { 0.0 } else { f_ext[i][d] + fb[i][d] + fd[i][d] - ma[d] })
                })
                .collect();
            (r, damping)
        };

        let shifted: Vec<Block6> = lay
            .chunks
            .iter()
            .map(|&c| {
                let ch = &self.structures[s].chunks[c];
                // Mass and, with `rayleigh_damping`, the mass-proportional dashpot.
                mass_block(ch.mass, &ch.inertia, c_m + if self.clusters[ci].anchored || self.config.methods.corotational_bonds { c_v * self.mass_damping.get(s).copied().unwrap_or(0.0) } else { 0.0 })
            })
            .collect();
        let mut x: Vec<V6> = start.iter().map(|(u, _, _)| *u).collect();
        let mut report = ImplicitReport::default();
        let (mut r, mut damping) = residual(self, &x);
        let mut norm = dot6(&r, &r).sqrt();
        let scale = dot6(&f_ext, &f_ext).sqrt().max(norm).max(1e-300);
        for it in 0..MAX_NEWTON {
            report.residual = norm / scale;
            report.newton_iterations = it;
            if report.residual <= IMPLICIT_TOLERANCE {
                report.converged = true;
                break;
            }
            // Correction with the joints' tangent, falling back to their secant (an upper
            // bound, always a descent direction) when the tangent step does not reduce
            // the residual: the contact patch and friction make the law nonsmooth.
            let mut improved = false;
            for tangent in [true, false] {
                let k: Vec<Mat6> =
                    self.newton_stiffness(ci, tangent).iter().zip(&damping).map(|(k, c)| k.add_diag(&c.scale(c_v))).collect();
                let pre = self.block_jacobi(ci, &lay, &k, Some(&shifted));
                let apply = |p: &[V6]| -> Vec<V6> {
                    let mut y = self.apply_k(ci, &lay, &k, p);
                    for (i, yi) in y.iter_mut().enumerate() {
                        let mp = shifted[i].mul(&p[i]);
                        for d in 0..6 {
                            if !held(i, d) {
                                yi[d] += mp[d];
                            }
                        }
                    }
                    y
                };
                let precondition = |r: &[V6]| -> Vec<V6> { r.iter().zip(&pre).map(|(ri, b)| b.mul(ri)).collect() };
                let (dx, cg) = pcg_solve(apply, precondition, &r, IMPLICIT_TOLERANCE * 0.1);
                report.cg_iterations += cg;
                // Backtracking line search on the residual norm.
                let mut step = 1.0;
                while step >= 1.0 / 64.0 {
                    let trial: Vec<V6> =
                        x.iter().zip(&dx).map(|(xi, di)| std::array::from_fn(|d| xi[d] + step * di[d])).collect();
                    let (rt, trial_damping) = residual(self, &trial);
                    let nt = dot6(&rt, &rt).sqrt();
                    if nt < norm {
                        (x, r, damping, norm) = (trial, rt, trial_damping, nt);
                        improved = true;
                        break;
                    }
                    step *= 0.5;
                }
                if improved {
                    break;
                }
            }
            if !improved {
                // Stalled at the nonsmooth law's resolution; keep the best state.
                residual(self, &x);
                break;
            }
        }

        // Commit the step's kinematics, reactions and dashpot dissipation.
        let (acc, vel) = kinematics(&x);
        for (i, &c) in lay.chunks.iter().enumerate() {
            let st = &mut self.chunks[s][c];
            (st.u, st.th) = split(&x[i]);
            (st.v, st.w) = split(&vel[i]);
            (st.a, st.alpha) = split(&acc[i]);
        }
        let fb = self.bond_forces(ci, &lay);
        let (fd, _, power) = self.damping_forces(ci, &lay);
        self.energy.damping_dissipation += power * dt;
        for (i, &c) in lay.chunks.iter().enumerate() {
            if lay.fixed[i].iter().any(|&h| h) {
                let (fe, me) = split(&f_ext[i]);
                let (fi, mi) = split(&fb[i]);
                let (fdi, mdi) = split(&fd[i]);
                let moment = if lay.fixed[i][3] { -(me + mi + mdi) } else { Vec3::ZERO };
                self.chunks[s][c].reaction = (-(fe + fi + fdi), moment);
            }
        }
        report
    }
}

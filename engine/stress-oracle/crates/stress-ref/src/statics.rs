//! Quasi-static equilibrium of a cluster: `K(D) u = f`.
//!
//! Used for gravity prestress (structures settle to equilibrium before any event), for
//! settled clusters in `SolveMode::Adaptive`, and for `SolveMode::QuasiStatic` with its
//! same-step cascade (re-solve and recheck after breaks until stable).
//!
//! The nonlinear joint response (contacts, damage) is handled by a modified-Newton
//! iteration on the true residual; each correction solves the secant system with a
//! matrix-free conjugate gradient, block-Jacobi (6x6 per chunk) preconditioned. Free
//! clusters use inertia relief: the loads are made self-equilibrated by the cluster's
//! rigid acceleration, and the rigid modes are projected out (mass-orthogonally).

use crate::bond::Mat6;
use crate::math::Vec3;
use crate::scene::Support;
use crate::solver::{Activity, ChunkLoads, ChunkState, ReferenceSolver};

pub(crate) type V6 = [f64; 6];

/// `sum x_i y_i` in four interleaved partial sums (the direct solve's inner products: no
/// order to keep there, and the CPU overlaps the four chains).
fn dot4(x: &[f64], y: &[f64]) -> f64 {
    let (xc, yc) = (x.chunks_exact(4), y.chunks_exact(4));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    let mut acc = [0.0f64; 4];
    for (a, b) in xc.zip(yc) {
        for l in 0..4 {
            acc[l] += a[l] * b[l];
        }
    }
    let tail: f64 = xr.iter().zip(yr).map(|(a, b)| a * b).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// Newton steps of a static solve taken by the PCG before a direct solve may take over.
const DIRECT_AFTER: usize = 10;
/// Widest band (degrees of freedom) and largest band storage (64 MB) the direct solve
/// takes on: cost limits only (a factor of n b^2 / 2 flops against PCG steps of
/// thousands of iterations there); the accepted state is decided by the residual test.
const DIRECT_MAX_BAND: usize = 1200;
const DIRECT_MAX_ENTRIES: usize = 8_000_000;
/// Secant steps of a static solve before the `tangent_statics` re-solve takes over: the
/// PCG's DIRECT_AFTER, then ten direct steps.
const SECANT_STEPS: usize = DIRECT_AFTER + 10;
/// Step limit of the `tangent_statics` re-solve, in multiples of `max_newton`: near a
/// cracked joint's contact kink the force law is not differentiable and the globalised
/// Newton converges linearly (b5 quasi_static fragments: about x0.7 per step), so it
/// gets room the secant's limit does not give; where no descent remains it stops at once.
const RESOLVE_STEPS_FACTOR: usize = 3;
/// Largest cluster (chunks) whose re-solve takes the full tangent by a dense LU.
const DENSE_MAX_CHUNKS: usize = 128;
/// Rows of a Cholesky column per parallel task (each an inner product of up to a band).
const CHOLESKY_ROWS_PER_TASK: usize = 64;
/// Halvings of a globalised (`tangent_statics`) Newton step before its direction is given
/// up: steps down to 2^-30 of the full correction.
const LINE_SEARCH_HALVINGS: usize = 31;
/// Armijo fraction: a scaled step `alpha dx` is kept when it lowers the residual norm by at
/// least `ARMIJO alpha` of itself.
const ARMIJO: f64 = 1e-4;

pub(crate) fn dot6(a: &[V6], b: &[V6]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (0..6).map(|i| x[i] * y[i]).sum::<f64>()).sum()
}

pub(crate) fn split(v: &V6) -> (Vec3, Vec3) {
    (Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5]))
}

pub(crate) fn join(a: Vec3, b: Vec3) -> V6 {
    [a.x, a.y, a.z, b.x, b.y, b.z]
}

/// Dense symmetric 6x6 block with an in-place Cholesky inverse.
#[derive(Clone, Copy)]
pub(crate) struct Block6(pub(crate) [[f64; 6]; 6]);

impl Block6 {
    pub(crate) fn zero() -> Block6 {
        Block6([[0.0; 6]; 6])
    }
    /// `self += k u v^T`.
    fn add_product(&mut self, k: f64, u: &V6, v: &V6) {
        for i in 0..6 {
            for j in 0..6 {
                self.0[i][j] += k * u[i] * v[j];
            }
        }
    }
    /// Inverse of an SPD matrix (falls back to the diagonal if not SPD).
    fn inverse(&self) -> Block6 {
        let a = self.0;
        let mut l = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let mut sum = a[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }
                if i == j {
                    if sum <= 0.0 {
                        let mut d = Block6::zero();
                        for k in 0..6 {
                            d.0[k][k] = if a[k][k] > 0.0 { 1.0 / a[k][k] } else { 0.0 };
                        }
                        return d;
                    }
                    l[i][i] = sum.sqrt();
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
        }
        // inv = L^-T L^-1, column by column.
        let mut inv = Block6::zero();
        for col in 0..6 {
            let mut y = [0.0f64; 6];
            for i in 0..6 {
                let mut s = if i == col { 1.0 } else { 0.0 };
                for k in 0..i {
                    s -= l[i][k] * y[k];
                }
                y[i] = s / l[i][i];
            }
            let mut x = [0.0f64; 6];
            for i in (0..6).rev() {
                let mut s = y[i];
                for k in i + 1..6 {
                    s -= l[k][i] * x[k];
                }
                x[i] = s / l[i][i];
            }
            for i in 0..6 {
                inv.0[i][col] = x[i];
            }
        }
        inv
    }
    pub(crate) fn mul(&self, v: &V6) -> V6 {
        let mut o = [0.0; 6];
        for (i, oi) in o.iter_mut().enumerate() {
            *oi = (0..6).map(|j| self.0[i][j] * v[j]).sum();
        }
        o
    }
}

/// Preconditioned conjugate gradient for an SPD operator: solves `A x = r0` to a
/// relative residual `tol`; returns (x, iterations). Stops early if the operator
/// shows a non-positive curvature (the caller's Newton loop then re-linearises).
pub(crate) fn pcg_solve(
    apply: impl Fn(&[V6]) -> Vec<V6>,
    precondition: impl Fn(&[V6]) -> Vec<V6>,
    r0: &[V6],
    tol: f64,
) -> (Vec<V6>, usize) {
    let n = r0.len();
    let mut x = vec![[0.0; 6]; n];
    let mut r = r0.to_vec();
    let norm0 = dot6(&r, &r).sqrt();
    if norm0 == 0.0 {
        return (x, 0);
    }
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rz = dot6(&r, &z);
    let max_iter = 20 * n * 6 + 200;
    for it in 0..max_iter {
        let ap = apply(&p);
        let pap = dot6(&p, &ap);
        if pap <= 0.0 {
            return (x, it);
        }
        let alpha = rz / pap;
        for i in 0..n {
            for d in 0..6 {
                x[i][d] += alpha * p[i][d];
                r[i][d] -= alpha * ap[i][d];
            }
        }
        if dot6(&r, &r).sqrt() <= tol * norm0 {
            return (x, it + 1);
        }
        z = precondition(&r);
        let rz_new = dot6(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            for d in 0..6 {
                p[i][d] = z[i][d] + beta * p[i][d];
            }
        }
    }
    (x, max_iter)
}

/// Outcome of a quasi-static solve.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticReport {
    pub newton_iterations: usize,
    pub cg_iterations: usize,
    pub residual: f64,
    pub converged: bool,
    pub cascade_passes: usize,
    pub bonds_broken: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct StaticOptions {
    /// Relative residual tolerance (against the external load norm).
    pub tolerance: f64,
    pub max_newton: usize,
    /// Update damage after equilibrium and re-solve until no bond changes.
    pub cascade: bool,
    pub max_cascade: usize,
}

impl Default for StaticOptions {
    fn default() -> Self {
        StaticOptions { tolerance: 1e-10, max_newton: 60, cascade: false, max_cascade: 200 }
    }
}

impl StaticOptions {
    /// Secant steps `equilibrate` takes before the `tangent_statics` re-solve
    /// (SECANT_STEPS within `max_newton`); without it, all of `max_newton`.
    pub fn secant_steps(&self, tangent_statics: bool) -> usize {
        if tangent_statics {
            self.max_newton.min(SECANT_STEPS)
        } else {
            self.max_newton
        }
    }
}

pub(crate) struct Layout {
    pub(crate) chunks: Vec<usize>,
    /// Local index of each of the structure's chunks (`usize::MAX` outside the cluster).
    pub(crate) local: Vec<usize>,
    /// Per local chunk: which of its 6 DOFs are held.
    pub(crate) fixed: Vec<[bool; 6]>,
}

impl ReferenceSolver {
    pub(crate) fn layout(&self, ci: usize) -> Layout {
        let cl = &self.clusters[ci];
        let st = &self.structures[cl.structure];
        let chunks = cl.chunks.clone();
        let mut local = vec![usize::MAX; st.chunks.len()];
        for (i, &c) in chunks.iter().enumerate() {
            local[c] = i;
        }
        let fixed = chunks
            .iter()
            .map(|&c| match st.chunks[c].support {
                Support::Fixed => [true; 6],
                Support::Pinned => [true, true, true, false, false, false],
                Support::None => [false; 6],
            })
            .collect();
        Layout { chunks, local, fixed }
    }

    /// Bond forces exerted on the cluster's chunks at the current displacements,
    /// without committing any history. Returns per local chunk `[force; moment]`.
    pub(crate) fn bond_forces(&self, ci: usize, lay: &Layout) -> Vec<V6> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let mut out = vec![[0.0; 6]; lay.chunks.len()];
        for &bi in &cl.bonds {
            let b = &self.bonds[s][bi];
            let (ga, gb) = (b.geometry.a, b.geometry.b);
            let (sa, sb) = (&self.chunks[s][ga], &self.chunks[s][gb]);
            let d = b.geometry.kinematics(sa.u, sa.th, sb.u, sb.th);
            let resp = b.model().evaluate(&b.joint, &d, 0.0, false);
            let (fa, ma, fb, mb) = b.geometry.chunk_loads(&resp.force);
            let (ia, ib) = (lay.local[ga], lay.local[gb]);
            for k in 0..3 {
                out[ia][k] += fa[k];
                out[ia][3 + k] += ma[k];
                out[ib][k] += fb[k];
                out[ib][3 + k] += mb[k];
            }
        }
        out
    }

    /// Stiffness per bond at the current state for a Newton correction: the joint's
    /// tangent (`tangent = true`) or its secant, an upper bound of every tangent of the
    /// nonsmooth contact patch that makes a safe (if slower) direction.
    pub(crate) fn newton_stiffness(&self, ci: usize, tangent: bool) -> Vec<Mat6> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        cl.bonds
            .iter()
            .map(|&bi| {
                let b = &self.bonds[s][bi];
                let (sa, sb) = (&self.chunks[s][b.geometry.a], &self.chunks[s][b.geometry.b]);
                let d = b.geometry.kinematics(sa.u, sa.th, sb.u, sb.th);
                if tangent {
                    b.model().tangent(&b.joint, &d)
                } else {
                    b.model().secant(&b.joint, &d)
                }
            })
            .collect()
    }

    /// `K x` (the resisting force of displacement field `x`) with bond stiffnesses `k`.
    pub(crate) fn apply_k(&self, ci: usize, lay: &Layout, k: &[Mat6], x: &[V6]) -> Vec<V6> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let mut y = vec![[0.0; 6]; x.len()];
        for (j, &bi) in cl.bonds.iter().enumerate() {
            let g = &self.bonds[s][bi].geometry;
            let (ia, ib) = (lay.local[g.a], lay.local[g.b]);
            let (ua, ta) = split(&x[ia]);
            let (ub, tb) = split(&x[ib]);
            let q = k[j].mul(&g.kinematics(ua, ta, ub, tb));
            let (fa, ma, fb, mb) = g.chunk_loads(&q);
            for c in 0..3 {
                y[ia][c] -= fa[c];
                y[ia][3 + c] -= ma[c];
                y[ib][c] -= fb[c];
                y[ib][3 + c] -= mb[c];
            }
        }
        for (i, f) in lay.fixed.iter().enumerate() {
            for d in 0..6 {
                if f[d] {
                    y[i][d] = x[i][d];
                }
            }
        }
        y
    }

    /// Inverse 6x6 diagonal blocks of `K` (secant `k`) plus optional per-chunk blocks
    /// (the mass term of a dynamic step).
    pub(crate) fn block_jacobi(&self, ci: usize, lay: &Layout, k: &[Mat6], extra: Option<&[Block6]>) -> Vec<Block6> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let mut blocks = vec![Block6::zero(); lay.chunks.len()];
        for (j, &bi) in cl.bonds.iter().enumerate() {
            let g = &self.bonds[s][bi].geometry;
            let (ia, ib) = (lay.local[g.a], lay.local[g.b]);
            let axes = [g.t1, g.t2, g.normal];
            // Kinematic rows of the six local components on each chunk's 6 DOFs.
            let rows = |comp: usize| -> (V6, V6) {
                let t = axes[comp % 3];
                if comp < 3 {
                    (join(-t, -(g.ra.cross(t))), join(t, g.rb.cross(t)))
                } else {
                    (join(Vec3::ZERO, -t), join(Vec3::ZERO, t))
                }
            };
            let r: Vec<(V6, V6)> = (0..6).map(rows).collect();
            for p in 0..6 {
                for q in 0..6 {
                    let kpq = k[j].0[p][q];
                    if kpq != 0.0 {
                        blocks[ia].add_product(kpq, &r[p].0, &r[q].0);
                        blocks[ib].add_product(kpq, &r[p].1, &r[q].1);
                    }
                }
            }
        }
        if let Some(extra) = extra {
            for (b, e) in blocks.iter_mut().zip(extra) {
                for i in 0..6 {
                    for j in 0..6 {
                        b.0[i][j] += e.0[i][j];
                    }
                }
            }
        }
        for (i, f) in lay.fixed.iter().enumerate() {
            for d in 0..6 {
                if f[d] {
                    for e in 0..6 {
                        blocks[i].0[d][e] = 0.0;
                        blocks[i].0[e][d] = 0.0;
                    }
                    blocks[i].0[d][d] = 1.0;
                }
            }
            // Isolated rotational DOFs (e.g. a pinned chunk with no bending bonds).
            for d in 0..6 {
                if blocks[i].0[d][d] == 0.0 {
                    blocks[i].0[d][d] = 1.0;
                }
            }
        }
        blocks.iter().map(|b| b.inverse()).collect()
    }

    /// Mass-orthogonal projection of a displacement field off the rigid modes of a free cluster.
    fn project_displacement(&self, ci: usize, lay: &Layout, x: &mut [V6]) {
        let cl = &self.clusters[ci];
        let st = &self.structures[cl.structure];
        let mut p = Vec3::ZERO;
        let mut l = Vec3::ZERO;
        for (i, &c) in lay.chunks.iter().enumerate() {
            let ch = &st.chunks[c];
            let (u, th) = split(&x[i]);
            let r = ch.center - cl.com;
            p += u * ch.mass;
            l += r.cross(u) * ch.mass + ch.inertia * th;
        }
        let t = p / cl.mass;
        let th0 = cl.inertia.inverse().expect("invertible") * l;
        for (i, &c) in lay.chunks.iter().enumerate() {
            let r = st.chunks[c].center - cl.com;
            let (u, th) = split(&x[i]);
            x[i] = join(u - t - th0.cross(r), th - th0);
        }
    }

    /// Remove the net force and moment from a load field (inertia relief).
    fn project_load(&self, ci: usize, lay: &Layout, f: &mut [V6]) {
        let cl = &self.clusters[ci];
        let st = &self.structures[cl.structure];
        let mut net_f = Vec3::ZERO;
        let mut net_m = Vec3::ZERO;
        for (i, &c) in lay.chunks.iter().enumerate() {
            let (fi, mi) = split(&f[i]);
            net_f += fi;
            net_m += (st.chunks[c].center - cl.com).cross(fi) + mi;
        }
        let a = net_f / cl.mass;
        let alpha = cl.inertia.inverse().expect("invertible") * net_m;
        for (i, &c) in lay.chunks.iter().enumerate() {
            let ch = &st.chunks[c];
            let r = ch.center - cl.com;
            let (fi, mi) = split(&f[i]);
            f[i] = join(fi - (a + alpha.cross(r)) * ch.mass, mi - ch.inertia * alpha);
        }
    }

    /// Preconditioned CG for `K x = r` (secant `k`); returns (x, iterations).
    fn pcg(&self, ci: usize, lay: &Layout, k: &[Mat6], r0: &[V6], tol: f64) -> (Vec<V6>, usize) {
        let free = !self.clusters[ci].anchored;
        let pre = self.block_jacobi(ci, lay, k, None);
        let precondition = |r: &[V6]| -> Vec<V6> {
            let mut z: Vec<V6> = r.iter().zip(&pre).map(|(ri, b)| b.mul(ri)).collect();
            if free {
                self.project_displacement(ci, lay, &mut z);
            }
            z
        };
        pcg_solve(|x| self.apply_k(ci, lay, k, x), precondition, r0, tol)
    }

    /// `K x = r` for an anchored cluster's tangent stiffness `K` (bond stiffnesses `k`,
    /// held degrees of freedom as identity rows, as `apply_k`) by Cholesky of its band:
    /// chunks in layout order, or where that band is too wide in reverse Cuthill-McKee
    /// order (the same system, renumbered). None when the band is still too wide to pay
    /// or `K` is not positive definite (the caller then uses the PCG).
    fn banded_tangent_solve(&self, ci: usize, lay: &Layout, k: &[Mat6], r: &[V6]) -> Option<Vec<V6>> {
        let n = lay.chunks.len() * 6;
        let fits = |band: usize| band <= DIRECT_MAX_BAND && n * band <= DIRECT_MAX_ENTRIES;
        let mut at: Vec<usize> = (0..lay.chunks.len()).collect();
        if !fits(self.band_of(ci, lay, &at)?) {
            at = self.reverse_cuthill_mckee(ci, lay);
            if !fits(self.band_of(ci, lay, &at)?) {
                return None;
            }
        }
        self.banded_solve(ci, lay, k, r, &at)
    }

    /// Band (degrees of freedom, diagonal included) of a cluster's stiffness with local
    /// chunk i at band position `at[i]`; None without bonds.
    fn band_of(&self, ci: usize, lay: &Layout, at: &[usize]) -> Option<usize> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let width = cl.bonds.iter().map(|&bi| {
            let g = &self.bonds[s][bi].geometry;
            at[lay.local[g.a]].abs_diff(at[lay.local[g.b]])
        }).max()?;
        Some(6 * (width + 1))
    }

    /// `banded_tangent_solve` with local chunk i at band position `at[i]`.
    fn banded_solve(&self, ci: usize, lay: &Layout, k: &[Mat6], r: &[V6], at: &[usize]) -> Option<Vec<V6>> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let n = lay.chunks.len() * 6;
        let band = self.band_of(ci, lay, at)?;
        // Row i, column j (i - band < j <= i) at a[i * band + (j + band - 1 - i)].
        let pos = |i: usize, j: usize| i * band + (j + band - 1 - i);
        let mut a = vec![0.0f64; n * band];
        for (jb, &bi) in cl.bonds.iter().enumerate() {
            let g = &self.bonds[s][bi].geometry;
            let (ia, ib) = (lay.local[g.a], lay.local[g.b]);
            let axes = [g.t1, g.t2, g.normal];
            // The six kinematic rows (as `block_jacobi`) on the pair's 12 DOFs.
            let rows: [[f64; 12]; 6] = std::array::from_fn(|comp| {
                let t = axes[comp % 3];
                let (ra, rb) = if comp < 3 { (join(-t, -(g.ra.cross(t))), join(t, g.rb.cross(t))) } else { (join(Vec3::ZERO, -t), join(Vec3::ZERO, t)) };
                std::array::from_fn(|x| if x < 6 { ra[x] } else { rb[x - 6] })
            });
            let mut kb = [[0.0f64; 12]; 12];
            for p in 0..6 {
                for q in 0..6 {
                    let kpq = k[jb].0[p][q];
                    if kpq == 0.0 {
                        continue;
                    }
                    for x in 0..12 {
                        for y in 0..12 {
                            kb[x][y] += kpq * rows[p][x] * rows[q][y];
                        }
                    }
                }
            }
            let base = [6 * at[ia], 6 * at[ib]];
            for x in 0..12 {
                for y in 0..12 {
                    let (gi, gj) = (base[x / 6] + x % 6, base[y / 6] + y % 6);
                    if gi >= gj {
                        a[pos(gi, gj)] += kb[x][y];
                    }
                }
            }
        }
        for (i, f) in lay.fixed.iter().enumerate() {
            for d in 0..6 {
                if f[d] {
                    let gi = 6 * at[i] + d;
                    for j in (gi + 1).saturating_sub(band)..=gi {
                        a[pos(gi, j)] = 0.0;
                    }
                    for j in gi..(gi + band).min(n) {
                        a[pos(j, gi)] = 0.0;
                    }
                    a[pos(gi, gi)] = 1.0;
                }
            }
        }
        // Cholesky L L^T in place, column by column: each entry is the same inner product
        // of earlier entries as row by row, so the factor does not depend on the order.
        // A column's entries below the diagonal are independent (rows in parallel).
        for j in 0..n {
            let lo_j = (j + 1).saturating_sub(band);
            let v = a[pos(j, j)] - dot4(&a[pos(j, lo_j)..pos(j, j)], &a[pos(j, lo_j)..pos(j, j)]);
            if !(v > 0.0) {
                return None;
            }
            let ljj = v.sqrt();
            a[pos(j, j)] = ljj;
            let end = (j + band).min(n);
            let col = crate::par::map_range(j + 1..end, CHOLESKY_ROWS_PER_TASK, |i| {
                let lo = (i + 1).saturating_sub(band);
                let (ri, rj) = (pos(i, lo), pos(j, lo));
                (a[pos(i, j)] - dot4(&a[ri..ri + (j - lo)], &a[rj..rj + (j - lo)])) / ljj
            });
            for (i, v) in (j + 1..end).zip(col) {
                a[pos(i, j)] = v;
            }
        }
        let mut y = vec![0.0f64; n];
        for (i, v) in r.iter().enumerate() {
            y[6 * at[i]..6 * at[i] + 6].copy_from_slice(v);
        }
        for i in 0..n {
            let lo = (i + 1).saturating_sub(band);
            let dot = dot4(&a[pos(i, lo)..pos(i, i)], &y[lo..i]);
            y[i] = (y[i] - dot) / a[pos(i, i)];
        }
        for i in (0..n).rev() {
            let mut v = y[i];
            for j in i + 1..(i + band).min(n) {
                v -= a[pos(j, i)] * y[j];
            }
            y[i] = v / a[pos(i, i)];
        }
        Some(at.iter().map(|&b| std::array::from_fn(|d| y[6 * b + d])).collect())
    }

    /// Reverse Cuthill-McKee numbering of a cluster's chunks over its bond graph: band
    /// position per local chunk. Each connected part starts from a pseudo-peripheral chunk
    /// (George-Liu: restart from the farthest, least-connected chunk of the last level
    /// while the depth grows); neighbours by ascending degree, then index.
    fn reverse_cuthill_mckee(&self, ci: usize, lay: &Layout) -> Vec<usize> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let m = lay.chunks.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
        for &bi in &cl.bonds {
            let g = &self.bonds[s][bi].geometry;
            let (a, b) = (lay.local[g.a], lay.local[g.b]);
            if a != b {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
        for v in adj.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }
        let deg: Vec<usize> = adj.iter().map(Vec::len).collect();
        for v in adj.iter_mut() {
            v.sort_by_key(|&j| (deg[j], j));
        }
        // Breadth-first levels from `root` over unplaced chunks: (order, start of the
        // last level, number of levels).
        let bfs = |root: usize, placed: &[bool]| -> (Vec<usize>, usize, usize) {
            let mut seen = placed.to_vec();
            seen[root] = true;
            let (mut order, mut level, mut depth) = (vec![root], 0, 0);
            let mut head = 0;
            while head < order.len() {
                let end = order.len();
                level = head;
                depth += 1;
                for q in head..end {
                    for &j in &adj[order[q]] {
                        if !seen[j] {
                            seen[j] = true;
                            order.push(j);
                        }
                    }
                }
                head = end;
            }
            (order, level, depth)
        };
        let mut placed = vec![false; m];
        let mut order = Vec::with_capacity(m);
        for start in 0..m {
            if placed[start] {
                continue;
            }
            // `start` is the first unplaced chunk; its part's least-connected chunk roots it.
            let (first, _, _) = bfs(start, &placed);
            let root = first.iter().copied().min_by_key(|&i| (deg[i], i)).unwrap_or(start);
            let (mut part, mut last, mut depth) = bfs(root, &placed);
            loop {
                let far = part[last..].iter().copied().min_by_key(|&i| (deg[i], i)).unwrap();
                let (p2, l2, d2) = bfs(far, &placed);
                if d2 <= depth {
                    break;
                }
                (part, last, depth) = (p2, l2, d2);
            }
            for &i in &part {
                placed[i] = true;
            }
            order.extend(part);
        }
        let mut at = vec![0; m];
        for (p, &i) in order.iter().rev().enumerate() {
            at[i] = p;
        }
        at
    }

    /// External chunk loads of a cluster in its body frame (gravity, given loads,
    /// replacement loads, minus rigid inertial loads), per local chunk.
    pub(crate) fn static_loads(&self, ci: usize, lay: &Layout, loads: &ChunkLoads) -> Vec<V6> {
        let (a, alpha) = self.rigid_acceleration(ci, loads);
        lay.chunks
            .iter()
            .map(|&c| {
                let (f, m) = self.frame_loads(ci, c, loads, a, alpha, self.time);
                join(f, m)
            })
            .collect()
    }

    /// Solve one cluster to equilibrium under `loads` (no damage update).
    pub fn equilibrate(&mut self, ci: usize, loads: &ChunkLoads, opts: &StaticOptions) -> StaticReport {
        let lay = self.layout(ci);
        let s = self.clusters[ci].structure;
        let free = !self.clusters[ci].anchored;
        let mut f_ext = self.static_loads(ci, &lay, loads);
        if free {
            self.project_load(ci, &lay, &mut f_ext);
        }
        let scale = dot6(&f_ext, &f_ext).sqrt().max(1e-300);
        let start: Vec<(Vec3, Vec3)> = lay.chunks.iter().map(|&c| (self.chunks[s][c].u, self.chunks[s][c].th)).collect();
        // `tangent_statics`: the secant iteration has SECANT_STEPS before the re-solve.
        let cap = opts.secant_steps(self.config.methods.tangent_statics);
        let mut report = self.newton(ci, &lay, &f_ext, free, scale, opts, false, 0..cap);
        if !report.converged && self.config.methods.tangent_statics {
            // `sim.methods.tangent_statics`: the secant iteration has not converged;
            // solve again from the same start with tangent steps. A solve the secant
            // converges within SECANT_STEPS is unchanged.
            let secant: Vec<(Vec3, Vec3)> = lay.chunks.iter().map(|&c| (self.chunks[s][c].u, self.chunks[s][c].th)).collect();
            for (i, &c) in lay.chunks.iter().enumerate() {
                (self.chunks[s][c].u, self.chunks[s][c].th) = start[i];
            }
            // Its residual is measured against the larger of the load and the starting
            // out-of-balance force, as the implicit solve's: a cluster whose deforming
            // load vanishes (a free fragment under rigid loads alone, its load round-off
            // after the rigid modes are projected out) relaxes its bonds, and that
            // equilibrium is reached when their forces have fallen by the tolerance.
            let r0 = self.static_residual(ci, &lay, &f_ext, free);
            let report_scale = scale;
            let scale = scale.max(dot6(&r0, &r0).sqrt());
            let second = self.newton(ci, &lay, &f_ext, free, scale, opts, true, 0..RESOLVE_STEPS_FACTOR * opts.max_newton);
            let first = report;
            report = StaticReport {
                newton_iterations: first.newton_iterations + second.newton_iterations,
                cg_iterations: first.cg_iterations + second.cg_iterations,
                ..second
            };
            // Where the re-solve fails too, the secant iteration resumes where it
            // stopped, its remaining steps exactly as without the re-solve: every solve
            // it converges by the step limit still converges. (A non-finite residual
            // never comes back.)
            if !second.converged && first.residual.is_finite() && cap < opts.max_newton {
                for (i, &c) in lay.chunks.iter().enumerate() {
                    (self.chunks[s][c].u, self.chunks[s][c].th) = secant[i];
                }
                let third = self.newton(ci, &lay, &f_ext, free, report_scale, opts, false, cap..opts.max_newton);
                report = StaticReport {
                    newton_iterations: report.newton_iterations + third.newton_iterations - cap,
                    cg_iterations: report.cg_iterations + third.cg_iterations,
                    ..third
                };
            }
        }
        // Equilibrium: no deformation velocity; refresh the stored bond forces.
        for &c in &lay.chunks {
            let st = &mut self.chunks[s][c];
            st.v = Vec3::ZERO;
            st.w = Vec3::ZERO;
        }
        let fb = self.bond_forces(ci, &lay);
        for (i, &c) in lay.chunks.iter().enumerate() {
            let (fe, me) = split(&f_ext[i]);
            let (fi, mi) = split(&fb[i]);
            if lay.fixed[i].iter().any(|&x| x) {
                let rm = if lay.fixed[i][3] { -(me + mi) } else { Vec3::ZERO };
                self.chunks[s][c].reaction = (-(fe + fi), rm);
            }
        }
        self.refresh_bond_forces(ci);
        crate::profile::count(crate::profile::Counter::StaticSolves, 1);
        crate::profile::count(crate::profile::Counter::StaticConverged, report.converged as u64);
        crate::profile::count(crate::profile::Counter::NewtonIterations, report.newton_iterations as u64);
        crate::profile::count(crate::profile::Counter::CgIterations, report.cg_iterations as u64);
        report
    }

    /// The Newton iteration of `equilibrate` from the cluster's current displacements:
    /// secant steps, or with `tangent` globalised tangent (else secant) steps.
    #[allow(clippy::too_many_arguments)]
    fn newton(&mut self, ci: usize, lay: &Layout, f_ext: &[V6], free: bool, scale: f64, opts: &StaticOptions, tangent: bool, steps: std::ops::Range<usize>) -> StaticReport {
        let s = self.clusters[ci].structure;
        let mut report = StaticReport::default();
        for it in steps {
            let r = self.static_residual(ci, lay, f_ext, free);
            let res = dot6(&r, &r).sqrt() / scale;
            report.residual = res;
            report.newton_iterations = it;
            if res <= opts.tolerance {
                report.converged = true;
                break;
            }
            // The re-solve also accepts an iterate whose out-of-balance force is within
            // the rounding error of computing it: an equilibrium to working precision,
            // which a residual relative to a load of round-off (a free fragment at rest)
            // can never certify.
            if tangent && res.is_finite() {
                let floor = self.residual_floor(ci, lay, f_ext, free);
                if r.iter().zip(&floor).all(|(ri, fi)| (0..6).all(|d| ri[d].abs() <= fi[d])) {
                    report.converged = true;
                    crate::profile::count(crate::profile::Counter::StaticAtRoundoff, 1);
                    break;
                }
            }
            // A non-finite residual never comes back (every later iterate is NaN or
            // infinite). Under `tangent_statics` the secant's unconverged state is
            // discarded for the re-solve from the start, so it ends here.
            if !res.is_finite() && !tangent && self.config.methods.tangent_statics {
                break;
            }
            if !tangent {
                let k = self.newton_stiffness(ci, false);
                let dx = self.newton_step(ci, lay, &k, &r, it, free, opts, &mut report);
                for (i, &c) in lay.chunks.iter().enumerate() {
                    let (du, dth) = split(&dx[i]);
                    let st = &mut self.chunks[s][c];
                    st.u += du;
                    st.th += dth;
                }
                continue;
            }
            // With `tangent`, a globalised Newton step: the joints' tangent, else the
            // secant (an upper bound of the nonsmooth patch's tangents), each scaled back
            // by halving until the residual falls by the Armijo fraction. Where neither
            // direction lowers it the iterate is a stationary point of the residual norm
            // that is no equilibrium, and the solve stops unconverged.
            let saved: Vec<(Vec3, Vec3)> = lay.chunks.iter().map(|&c| (self.chunks[s][c].u, self.chunks[s][c].th)).collect();
            let mut accepted = false;
            'directions: for use_tangent in [true, false] {
                // A small cluster's tangent step is the consistent one: the full
                // (non-symmetric) tangent by a dense LU.
                let dense = if use_tangent { self.dense_tangent_solve(ci, lay, free, &r) } else { None };
                let dx = match dense {
                    Some(dx) => dx,
                    None => {
                        let k = self.newton_stiffness(ci, use_tangent);
                        self.newton_step(ci, lay, &k, &r, it, free, opts, &mut report)
                    }
                };
                let mut alpha = 1.0;
                for _ in 0..LINE_SEARCH_HALVINGS {
                    for (i, &c) in lay.chunks.iter().enumerate() {
                        let (du, dth) = split(&dx[i]);
                        let st = &mut self.chunks[s][c];
                        st.u = saved[i].0 + du * alpha;
                        st.th = saved[i].1 + dth * alpha;
                    }
                    let tried = self.static_residual(ci, lay, f_ext, free);
                    if dot6(&tried, &tried).sqrt() / scale <= (1.0 - ARMIJO * alpha) * res {
                        accepted = true;
                        break 'directions;
                    }
                    alpha *= 0.5;
                }
            }
            if !accepted {
                for (i, &c) in lay.chunks.iter().enumerate() {
                    (self.chunks[s][c].u, self.chunks[s][c].th) = saved[i];
                }
                break;
            }
        }
        report
    }

    /// One Newton correction `K dx = r` with bond stiffnesses `k`: by the PCG, or once
    /// the solve has taken DIRECT_AFTER steps by a direct (banded Cholesky) solve where
    /// it fits.
    #[allow(clippy::too_many_arguments)]
    fn newton_step(&self, ci: usize, lay: &Layout, k: &[Mat6], r: &[V6], it: usize, free: bool, opts: &StaticOptions, report: &mut StaticReport) -> Vec<V6> {
        // A solve still unconverged after DIRECT_AFTER Newton steps takes its remaining
        // steps by a direct (banded Cholesky) solve of the same system where it fits:
        // about twice as fast as the PCG's 500-900 iterations there, and exact to
        // round-off rather than to the PCG's tolerance. Earlier steps are the PCG's, so
        // every solve that converges within them is unchanged.
        let direct = if it >= DIRECT_AFTER && !free { self.banded_tangent_solve(ci, lay, k, r) } else { None };
        let (dx, cg) = match direct {
            Some(dx) => {
                crate::profile::count(crate::profile::Counter::DirectSolves, 1);
                (dx, 0)
            }
            None => self.pcg(ci, lay, k, r, (opts.tolerance * 0.1).max(1e-14)),
        };
        report.cg_iterations += cg;
        dx
    }

    /// Bound of the rounding error of `static_residual` at the current displacements, per
    /// local chunk and component (`gamma_n = n u / (1 - n u)`, `u` the unit roundoff):
    /// - a bond's relative displacement `(u_b + th_b x r_b) - (u_a + th_a x r_a)` in local
    ///   axes errs by at most `gamma_7 |D|`, `D = |u_a| + |u_b| + |th_a| x |r_a| +
    ///   |th_b| x |r_b|` (the cancellation of large rigid motions), its rotation by
    ///   `gamma_4 ||th_a| + |th_b||`;
    /// - its force by the stiffness times that, plus `gamma_20 |q|` for the joint law;
    /// - the chunk loads by `gamma_3` of their terms plus the propagated errors (lever
    ///   arms included), and the sum over a chunk's bonds and its load by
    ///   `gamma_(n+1)` of the absolute sum;
    /// - a free cluster's projection of the rigid modes by `gamma_(N+10)` of the absolute
    ///   sum over the cluster.
    fn residual_floor(&self, ci: usize, lay: &Layout, f_ext: &[V6], free: bool) -> Vec<V6> {
        let gamma = |n: f64| {
            let u = f64::EPSILON / 2.0;
            n * u / (1.0 - n * u)
        };
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let m = lay.chunks.len();
        let mut prop = vec![[0.0f64; 6]; m];
        let mut abs_sum: Vec<V6> = f_ext.iter().map(|f| f.map(f64::abs)).collect();
        let mut count = vec![1.0f64; m];
        let abs_cross = |a: Vec3, b: Vec3| Vec3::new(a.y.abs() * b.z.abs() + a.z.abs() * b.y.abs(), a.z.abs() * b.x.abs() + a.x.abs() * b.z.abs(), a.x.abs() * b.y.abs() + a.y.abs() * b.x.abs());
        let r3 = 3f64.sqrt();
        for &bi in &cl.bonds {
            let b = &self.bonds[s][bi];
            let g = &b.geometry;
            let (sa, sb) = (&self.chunks[s][g.a], &self.chunks[s][g.b]);
            let d = g.kinematics(sa.u, sa.th, sb.u, sb.th);
            let q = b.model().evaluate(&b.joint, &d, 0.0, false).force;
            let (fa, ma, fb, mb) = g.chunk_loads(&q);
            let dabs = sa.u.abs() + sb.u.abs() + abs_cross(sa.th, g.ra) + abs_cross(sb.th, g.rb);
            let err_lin = gamma(7.0) * dabs.norm();
            let err_ang = gamma(4.0) * (sa.th.abs() + sb.th.abs()).norm();
            let mut k = b.stiffness.as_local();
            if let Some(rb) = &b.rebar {
                k.lin += Vec3::new(rb.k_dowel, rb.k_dowel, rb.k_axial);
            }
            let eq_lin = k.lin.norm() * err_lin + gamma(20.0) * q.lin.norm();
            let eq_ang = k.ang.norm() * err_ang + gamma(20.0) * q.ang.norm();
            let ef = gamma(3.0) * r3 * q.lin.norm() + r3 * eq_lin;
            let em = gamma(3.0) * r3 * q.ang.norm() + r3 * eq_ang;
            let (ia, ib) = (lay.local[g.a], lay.local[g.b]);
            for (i, f, mo, r) in [(ia, fa, ma, g.ra), (ib, fb, mb, g.rb)] {
                let em_i = em + gamma(3.0) * (q.ang.norm() + r.norm() * f.norm()) + r.norm() * ef;
                for t in 0..3 {
                    prop[i][t] += ef;
                    prop[i][3 + t] += em_i;
                    abs_sum[i][t] += f.to_array()[t].abs();
                    abs_sum[i][3 + t] += mo.to_array()[t].abs();
                }
                count[i] += 1.0;
            }
        }
        let mut floor: Vec<V6> = (0..m).map(|i| std::array::from_fn(|d| prop[i][d] + gamma(count[i] + 1.0) * abs_sum[i][d])).collect();
        if free {
            let total: f64 = abs_sum.iter().flat_map(|v| v.iter()).sum();
            let extra = gamma(m as f64 + 10.0) * total;
            for f in floor.iter_mut() {
                for v in f.iter_mut() {
                    *v += extra;
                }
            }
        }
        for (i, f) in lay.fixed.iter().enumerate() {
            for d in 0..6 {
                if f[d] {
                    floor[i][d] = 0.0;
                }
            }
        }
        floor
    }

    /// `K dx = r` for a cluster of at most DENSE_MAX_CHUNKS chunks with every joint's full
    /// tangent (`JointModel::tangent_full`, the sliding friction's dependence on the
    /// normal force included, so not symmetric), by LU with partial pivoting. Held DOFs
    /// are identity rows and columns (as the banded solve). A free cluster's rigid modes
    /// are fixed by the constraints `project_displacement` imposes (no net momentum-like
    /// sums), `[K G^T; G 0]`: `G dx = 0`, and `G^T` spans the rigid load fields
    /// `project_load` removes. None for a larger cluster or a singular system.
    fn dense_tangent_solve(&self, ci: usize, lay: &Layout, free: bool, r: &[V6]) -> Option<Vec<V6>> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let m = lay.chunks.len();
        if m > DENSE_MAX_CHUNKS {
            return None;
        }
        let nk = 6 * m;
        let n = if free { nk + 6 } else { nk };
        let mut a = vec![0.0f64; n * n];
        for &bi in &cl.bonds {
            let b = &self.bonds[s][bi];
            let g = &b.geometry;
            let (sa, sb) = (&self.chunks[s][g.a], &self.chunks[s][g.b]);
            let d = g.kinematics(sa.u, sa.th, sb.u, sb.th);
            let t = b.model().tangent_full(&b.joint, &d);
            let axes = [g.t1, g.t2, g.normal];
            let rows: [[f64; 12]; 6] = std::array::from_fn(|comp| {
                let ax = axes[comp % 3];
                let (ra, rb) = if comp < 3 { (join(-ax, -(g.ra.cross(ax))), join(ax, g.rb.cross(ax))) } else { (join(Vec3::ZERO, -ax), join(Vec3::ZERO, ax)) };
                std::array::from_fn(|x| if x < 6 { ra[x] } else { rb[x - 6] })
            });
            // B^T T B on the pair's 12 DOFs.
            let tb: [[f64; 12]; 6] = std::array::from_fn(|p| std::array::from_fn(|y| (0..6).map(|q| t.0[p][q] * rows[q][y]).sum()));
            let base = [6 * lay.local[g.a], 6 * lay.local[g.b]];
            for x in 0..12 {
                for y in 0..12 {
                    let v: f64 = (0..6).map(|p| rows[p][x] * tb[p][y]).sum();
                    a[(base[x / 6] + x % 6) * n + base[y / 6] + y % 6] += v;
                }
            }
        }
        for (i, f) in lay.fixed.iter().enumerate() {
            for d in 0..6 {
                if f[d] {
                    let gi = 6 * i + d;
                    for j in 0..n {
                        a[gi * n + j] = 0.0;
                        a[j * n + gi] = 0.0;
                    }
                    a[gi * n + gi] = 1.0;
                }
            }
        }
        if free {
            // G: the net linear and angular momentum-like sums of a displacement field.
            let st = &self.structures[s];
            for (i, &c) in lay.chunks.iter().enumerate() {
                let ch = &st.chunks[c];
                let rr = ch.center - cl.com;
                for k in 0..3 {
                    let e = Vec3::from_array(std::array::from_fn(|q| if q == k { 1.0 } else { 0.0 }));
                    // Linear: sum m u_k.
                    a[(nk + k) * n + 6 * i + k] = ch.mass;
                    a[(6 * i + k) * n + nk + k] = ch.mass;
                    // Angular: sum m (r x u)_k + (I th)_k.
                    for q in 0..3 {
                        let eq = Vec3::from_array(std::array::from_fn(|w| if w == q { 1.0 } else { 0.0 }));
                        let lin = ch.mass * rr.cross(eq).dot(e);
                        let ang = e.dot(ch.inertia * eq);
                        a[(nk + 3 + k) * n + 6 * i + q] = lin;
                        a[(nk + 3 + k) * n + 6 * i + 3 + q] = ang;
                        a[(6 * i + q) * n + nk + 3 + k] = lin;
                        a[(6 * i + 3 + q) * n + nk + 3 + k] = ang;
                    }
                }
            }
        }
        let mut rhs = vec![0.0f64; n];
        for (i, v) in r.iter().enumerate() {
            rhs[6 * i..6 * i + 6].copy_from_slice(v);
        }
        // LU with partial pivoting.
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut perm: Vec<usize> = (0..n).collect();
        for col in 0..n {
            let piv = (col..n).max_by(|&x, &y| a[perm[x] * n + col].abs().total_cmp(&a[perm[y] * n + col].abs()))?;
            if !(a[perm[piv] * n + col].abs() > 1e-14 * scale) {
                return None;
            }
            perm.swap(col, piv);
            let pr = perm[col];
            for row in col + 1..n {
                let rr = perm[row];
                let f = a[rr * n + col] / a[pr * n + col];
                if f != 0.0 {
                    a[rr * n + col] = f;
                    for j in col + 1..n {
                        a[rr * n + j] -= f * a[pr * n + j];
                    }
                }
            }
        }
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let pi = perm[i];
            y[i] = rhs[pi] - (0..i).map(|j| a[pi * n + j] * y[j]).sum::<f64>();
        }
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let pi = perm[i];
            x[i] = (y[i] - (i + 1..n).map(|j| a[pi * n + j] * x[j]).sum::<f64>()) / a[pi * n + i];
        }
        if !x.iter().all(|v| v.is_finite()) {
            return None;
        }
        Some((0..m).map(|i| std::array::from_fn(|d| x[6 * i + d])).collect())
    }

    /// Fundamental period (s) of a cluster at full stiffness, by inverse iteration
    /// `y = K^-1 M x` with the Rayleigh quotient, to a relative change of 1e-10.
    /// - Anchored: the banded direct solve of `banded_tangent_solve`, held DOFs fixed.
    /// - Free: deflated against the six rigid modes. `M x` is made self-equilibrated
    ///   (inertia relief), `K y = M x` is solved by the PCG on the complement of the
    ///   rigid modes, and the rigid part of `y` is removed. The period is that of the
    ///   first flexible mode.
    ///
    /// None for a cluster without bonds, or an anchored one whose band the direct solve
    /// does not take.
    pub fn fundamental_period(&self, ci: usize) -> Option<f64> {
        let cl = &self.clusters[ci];
        if cl.bonds.is_empty() {
            return None;
        }
        let free = !cl.anchored;
        let lay = self.layout(ci);
        let st = &self.structures[cl.structure];
        let k: Vec<Mat6> = cl.bonds.iter().map(|&bi| Mat6::diag(&{
            let b = &self.bonds[cl.structure][bi];
            let mut k = b.stiffness.as_local();
            if let Some(rb) = &b.rebar {
                k.lin += Vec3::new(rb.k_dowel, rb.k_dowel, rb.k_axial);
            }
            k
        })).collect();
        let mass = |i: usize, x: &V6| -> V6 {
            let c = lay.chunks[i];
            let ch = &st.chunks[c];
            let mu = self.chunks[cl.structure][c].inertia_scale;
            let (u, th) = split(x);
            let mut out = join(u * (ch.mass * mu), (ch.inertia * th) * mu);
            for d in 0..6 {
                if lay.fixed[i][d] {
                    out[d] = 0.0;
                }
            }
            out
        };
        // Start vector in consistent units (translations in chunk sizes, rotations in
        // radians), so the iteration, and the period, transform exactly with the units.
        let mut x: Vec<V6> = (0..lay.chunks.len())
            .map(|i| {
                let size = st.chunks[lay.chunks[i]].half_extents.norm();
                std::array::from_fn(|d| if lay.fixed[i][d] { 0.0 } else { (if d < 3 { size } else { 1.0 }) * (1.0 + 0.1 * ((i * 6 + d) as f64 * 0.618_033_988_75).fract()) })
            })
            .collect();
        if free {
            self.project_displacement(ci, &lay, &mut x);
        }
        let mut lambda_prev = 0.0;
        for _ in 0..500 {
            let mx: Vec<V6> = x.iter().enumerate().map(|(i, v)| mass(i, v)).collect();
            let y = if free {
                let mut rhs = mx.clone();
                self.project_load(ci, &lay, &mut rhs);
                let (mut y, _) = self.pcg(ci, &lay, &k, &rhs, 1e-13);
                self.project_displacement(ci, &lay, &mut y);
                y
            } else {
                self.banded_tangent_solve(ci, &lay, &k, &mx)?
            };
            // Rayleigh quotient x^T M x / x^T M y (y = K^-1 M x).
            let (xmx, xmy) = x.iter().zip(&mx).zip(&y).fold((0.0, 0.0), |(a, b), ((xi, mxi), yi)| (a + dot6(std::slice::from_ref(xi), std::slice::from_ref(mxi)), b + dot6(std::slice::from_ref(mxi), std::slice::from_ref(yi))));
            if !(xmy > 0.0) {
                return None;
            }
            let lambda = xmx / xmy;
            let norm = y.iter().enumerate().map(|(i, v)| dot6(std::slice::from_ref(v), std::slice::from_ref(&mass(i, v)))).sum::<f64>().sqrt();
            x = y.iter().map(|v| v.map(|c| c / norm)).collect();
            if (lambda - lambda_prev).abs() <= 1e-10 * lambda {
                return Some(2.0 * std::f64::consts::PI / lambda.sqrt());
            }
            lambda_prev = lambda;
        }
        Some(2.0 * std::f64::consts::PI / lambda_prev.sqrt())
    }

    /// Out-of-balance loads of a cluster at its current displacements (external plus bond
    /// forces, held DOFs zeroed, rigid modes projected out of a free cluster).
    fn static_residual(&self, ci: usize, lay: &Layout, f_ext: &[V6], free: bool) -> Vec<V6> {
        let fb = self.bond_forces(ci, lay);
        let mut r: Vec<V6> = f_ext.iter().zip(&fb).map(|(e, b)| std::array::from_fn(|d| e[d] + b[d])).collect();
        for (i, f) in lay.fixed.iter().enumerate() {
            for d in 0..6 {
                if f[d] {
                    r[i][d] = 0.0;
                }
            }
        }
        if free {
            self.project_load(ci, lay, &mut r);
        }
        r
    }

    /// `equilibrate`, keeping the cluster's previous state when no equilibrium is found:
    /// a mechanism (e.g. a joint yielding through) has no static solution, and damage
    /// evaluated at a non-equilibrium iterate would be meaningless. Its motion needs the
    /// dynamic solve.
    pub fn equilibrate_or_keep(&mut self, ci: usize, loads: &ChunkLoads, opts: &StaticOptions) -> StaticReport {
        let s = self.clusters[ci].structure;
        let saved: Vec<(usize, ChunkState)> = self.clusters[ci].chunks.iter().map(|&c| (c, self.chunks[s][c])).collect();
        let r = self.equilibrate(ci, loads, opts);
        if !r.converged {
            for (c, st) in saved {
                self.chunks[s][c] = st;
            }
            self.refresh_bond_forces(ci);
        }
        r
    }

    /// Re-evaluate and store each bond's force/energy without advancing history.
    pub fn refresh_bond_forces(&mut self, ci: usize) {
        let s = self.clusters[ci].structure;
        for &bi in &self.clusters[ci].bonds.clone() {
            let b = &self.bonds[s][bi];
            let (sa, sb) = (&self.chunks[s][b.geometry.a], &self.chunks[s][b.geometry.b]);
            let d = b.geometry.kinematics(sa.u, sa.th, sb.u, sb.th);
            let resp = b.model().evaluate(&b.joint, &d, 0.0, false);
            let bm = &mut self.bonds[s][bi];
            bm.force = resp.force;
            bm.elastic = resp.force;
            bm.measures = resp.measures;
            bm.stored = resp.stored;
        }
    }

    /// Commit damage at the current (equilibrium) displacements; returns
    /// (damage changed, any bond disconnected). `dt` advances static fatigue.
    pub fn commit_damage(&mut self, ci: usize, dt: f64) -> (bool, bool) {
        let s = self.clusters[ci].structure;
        let mut changed = false;
        let mut disconnected = false;
        for &bi in &self.clusters[ci].bonds.clone() {
            let b = &self.bonds[s][bi];
            let (sa, sb) = (&self.chunks[s][b.geometry.a], &self.chunks[s][b.geometry.b]);
            let d = b.geometry.kinematics(sa.u, sa.th, sb.u, sb.th);
            let resp = b.model().evaluate(&b.joint, &d, dt, self.config.fracture);
            self.max_utilization = self.max_utilization.max(resp.state.utilization);
            let previous = b.joint.clone();
            if resp.state.damage > previous.damage + 1e-9 || resp.state.crush > previous.crush + 1e-9 {
                changed = true;
            }
            self.energy.bond_dissipation += resp.dissipated;
            self.energy.softening_overshoot += resp.overshoot;
            self.energy.softening_overshoot_negative += resp.overshoot.min(0.0);
            let bm = &mut self.bonds[s][bi];
            bm.joint = resp.state;
            bm.observe_utilization();
            bm.force = resp.force;
            bm.elastic = resp.force;
            bm.measures = resp.measures;
            bm.stored = resp.stored;
            disconnected |= self.record_bond_events(ci, bi, &previous, resp.disconnected, self.time);
        }
        (changed, disconnected)
    }

    /// Quasi-static solve of every cluster with same-step cascade: equilibrate, update
    /// damage, split, and repeat until no bond changes.
    pub fn solve_static_all(&mut self, loads: &ChunkLoads, opts: &StaticOptions, dt: f64) -> StaticReport {
        let mut total = StaticReport { converged: true, ..Default::default() };
        let broken_before = self.broken_bond_count();
        for pass in 0..opts.max_cascade.max(1) {
            let mut any = false;
            let n = self.clusters.len();
            let mut to_split = Vec::new();
            for ci in 0..n {
                if self.clusters[ci].bonds.is_empty() {
                    continue;
                }
                let r = self.equilibrate_or_keep(ci, loads, opts);
                total.newton_iterations += r.newton_iterations;
                total.cg_iterations += r.cg_iterations;
                total.residual = total.residual.max(r.residual);
                total.converged &= r.converged;
                self.clusters[ci].activity = Activity::Settled;
                if opts.cascade && r.converged {
                    // Static fatigue advances once per call, not per cascade pass.
                    let (changed, disc) = self.commit_damage(ci, if pass == 0 { dt } else { 0.0 });
                    any |= changed || disc;
                    if disc {
                        to_split.push(ci);
                    }
                }
            }
            for &ci in to_split.iter().rev() {
                self.split_cluster(ci);
            }
            total.cascade_passes = pass + 1;
            if !opts.cascade || !any {
                break;
            }
        }
        total.bonds_broken = self.broken_bond_count() - broken_before;
        total
    }

    /// Settle every anchored cluster under gravity (and nothing else) before t = 0.
    pub fn gravity_prestress(&mut self) -> StaticReport {
        let loads = ChunkLoads::new(self);
        let opts = StaticOptions::default();
        let mut total = StaticReport { converged: true, ..Default::default() };
        for ci in 0..self.clusters.len() {
            if !self.clusters[ci].anchored || self.clusters[ci].bonds.is_empty() {
                continue;
            }
            let r = self.equilibrate(ci, &loads, &opts);
            total.newton_iterations += r.newton_iterations;
            total.cg_iterations += r.cg_iterations;
            total.residual = total.residual.max(r.residual);
            total.converged &= r.converged;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The direct solve in reverse Cuthill-McKee order solves the same system as in layout
    /// order: the numbering is a permutation, its band is no wider, and both solutions meet
    /// banded Cholesky's backward-error bound. The normwise error `|r - K x| / (|K| |x|)`
    /// (infinity norms, `K x` summed densely from `K`'s columns) stays below
    /// `n gamma(4b + 2)`, `gamma(m) = m u / (1 - m u)`: Higham, Accuracy and Stability of
    /// Numerical Algorithms, Thm 10.4 (`gamma(3b + 1)` for the factor and both
    /// substitutions, `| |R^T| |R| | <= n |K|`) plus `gamma(b)` for the residual's sums.
    /// The cantilever's fundamental period by inverse iteration is Euler-Bernoulli's first
    /// bending mode, `f1 = 1.875^2 / (2 pi) sqrt(EI / (rho A L^4))`, within the 1 % the
    /// chunk model holds it to (its `first_frequency` metric, also OpenSees').
    #[test]
    fn fundamental_period_is_the_first_bending_mode() {
        let scene = crate::builders::cantilever(20, "cantilever");
        let sv = ReferenceSolver::new(&scene);
        let period = sv.fundamental_period(0).expect("anchored cluster");
        let f1 = scene.metrics.iter().find(|m| m.name == "first_frequency").and_then(|m| m.expected.as_ref()).and_then(|v| v.as_f64()).expect("expected frequency");
        let f = 1.0 / period;
        println!("fundamental frequency {f:.6} Hz, Euler-Bernoulli {f1:.6} Hz");
        assert!((f - f1).abs() <= 0.01 * f1, "{f} Hz against {f1} Hz");
    }

    /// A free-free beam (the cantilever's 21 chunks, unsupported, 2.1 m): the first
    /// flexible mode is Euler-Bernoulli's `f1 = 4.730^2 / (2 pi L^2) sqrt(EI / (rho A))`,
    /// within the 1 % the chunk model holds the cantilever to.
    #[test]
    fn a_free_cluster_has_the_free_free_bending_period() {
        let mut scene = crate::builders::cantilever(20, "free beam");
        for c in &mut scene.bodies[0].chunks {
            c.support = crate::scene::Support::None;
        }
        let sv = ReferenceSolver::new(&scene);
        assert!(!sv.clusters[0].anchored);
        let f = 1.0 / sv.fundamental_period(0).expect("free cluster");
        let chunks = &scene.bodies[0].chunks;
        let h = 2.0 * chunks[0].half_extents[0];
        let xs = chunks.iter().map(|c| c.center[0]);
        let length = xs.clone().fold(f64::NEG_INFINITY, f64::max) - xs.fold(f64::INFINITY, f64::min) + h;
        let m = scene.material("beam");
        // Square section of side h: EI / (rho A) = E h^2 / (12 rho).
        let expected = 4.730f64.powi(2) / (2.0 * std::f64::consts::PI * length * length) * (m.youngs_modulus * h * h / (12.0 * m.density)).sqrt();
        println!("free-free {f:.6} Hz, Euler-Bernoulli {expected:.6} Hz over {length} m");
        assert!((f - expected).abs() <= 0.01 * expected, "{f} Hz against {expected} Hz");
    }

    #[test]
    fn reordered_direct_solve_solves_the_same_system() {
        let scene = crate::builders::pressure_panel(300e3, "panel");
        let world = crate::world::World::new(&scene);
        let sv = &world.solver;
        let ci = (0..sv.clusters.len()).find(|&c| sv.clusters[c].anchored && !sv.clusters[c].bonds.is_empty()).unwrap();
        let lay = sv.layout(ci);
        let m = lay.chunks.len();
        let n = 6 * m;
        let k = sv.newton_stiffness(ci, false);
        let at = sv.reverse_cuthill_mckee(ci, &lay);
        let mut sorted = at.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..m).collect::<Vec<_>>(), "not a permutation");
        let natural: Vec<usize> = (0..m).collect();
        let (b_nat, b_rcm) = (sv.band_of(ci, &lay, &natural).unwrap(), sv.band_of(ci, &lay, &at).unwrap());
        assert!(b_rcm <= b_nat, "RCM band {b_rcm} wider than the layout's {b_nat}");
        // K's columns (unit displacements through `apply_k`).
        let cols: Vec<Vec<V6>> = (0..n)
            .map(|j| {
                let mut e = vec![[0.0; 6]; m];
                e[j / 6][j % 6] = 1.0;
                sv.apply_k(ci, &lay, &k, &e)
            })
            .collect();
        let k_norm = (0..n).map(|i| cols.iter().map(|c| c[i / 6][i % 6].abs()).sum::<f64>()).fold(0.0, f64::max);
        // Zero on held DOFs, as every Newton residual (there `apply_k` keeps K's columns
        // and the direct solve drops them; they agree only on such loads).
        let r: Vec<V6> = (0..m).map(|i| std::array::from_fn(|d| if lay.fixed[i][d] { 0.0 } else { ((7 * i + d) as f64 * 0.37).sin() * 1e3 })).collect();
        let u = f64::EPSILON / 2.0;
        for (name, order, band) in [("layout", &natural, b_nat), ("rcm", &at, b_rcm)] {
            let x = sv.banded_solve(ci, &lay, &k, &r, order).expect("secant stiffness is positive definite");
            let x_norm = x.iter().flat_map(|v| v.iter()).fold(0.0f64, |a, v| a.max(v.abs()));
            let mut res = 0.0f64;
            for i in 0..n {
                let kx: f64 = (0..n).map(|j| cols[j][i / 6][i % 6] * x[j / 6][j % 6]).sum();
                res = res.max((r[i / 6][i % 6] - kx).abs());
            }
            let eta = res / (k_norm * x_norm);
            let mb = (4 * band + 2) as f64 * u;
            let bound = n as f64 * mb / (1.0 - mb);
            println!("{name}: band {band}, backward error {eta:.3e} (bound {bound:.3e})");
            assert!(eta <= bound, "{name}: backward error {eta:.3e} above {bound:.3e}");
        }
    }
}


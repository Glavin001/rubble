//! Offline model reduction of stress-ref's intact bond networks: static condensation of
//! a chosen set of chunks (Guyan), exact static recovery of the eliminated chunks, and
//! the highest natural frequency of the reduced dynamics (which sets the explicit step).
//!
//! The network of an anchored cluster is assembled from the oracle's own data: each
//! bond's six kinematic rows (`bond.rs` `kinematics`: `d = (u_b + th_b x r_b) - (u_a +
//! th_a x r_a)` in the patch frame `(t1, t2, n)`, rotation difference for the angular
//! rows) and its intact stiffness (`BondStiffness`, rebar in parallel), `K = sum B^T k
//! B`; held degrees of freedom (supports) are left out. Mass: each chunk's mass and
//! inertia (deformation inertia scale included).
//!
//! Eliminating chunk set `I` (the rest `R` retained), with `K_II` factored per connected
//! component of `I` (components share no bond, so `K_II` is block diagonal):
//!   `T = -K_II^-1 K_IR`,  `K~ = K_RR + K_RI T`,  `M~ = M_RR + T^T M_II T`,
//!   `f~ = f_R + T^T f_I`,  `u_I = T u_R + K_II^-1 f_I`.
//! Statics: exact (the reduced solve plus recovery is the full solve). Dynamics (Guyan):
//! the eliminated chunks follow the retained ones quasi-statically, with their inertia
//! carried by `M~`; the reduced spectrum interlaces the full one from below (a Ritz
//! projection), so its highest frequency, and so the explicit step's, can only improve.

use std::collections::BTreeMap;

use stress_ref::math::Vec3;
use stress_ref::scene::Support;
use stress_ref::solver::ReferenceSolver;

// ------------------------------------------------------------------ sparse matrices

/// A symmetric matrix by rows (both triangles stored), columns sorted.
#[derive(Clone, Debug)]
pub struct Sparse {
    pub n: usize,
    pub rows: Vec<Vec<(usize, f64)>>,
}

impl Sparse {
    /// From accumulated entries (duplicates summed).
    pub fn from_entries(n: usize, entries: BTreeMap<(usize, usize), f64>) -> Sparse {
        let mut rows = vec![Vec::new(); n];
        for ((i, j), v) in entries {
            if v != 0.0 {
                rows[i].push((j, v));
            }
        }
        Sparse { n, rows }
    }

    pub fn mul(&self, x: &[f64]) -> Vec<f64> {
        self.rows.iter().map(|r| r.iter().map(|&(j, v)| v * x[j]).sum()).collect()
    }

    pub fn nnz(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    pub fn diag(&self) -> Vec<f64> {
        self.rows.iter().enumerate().map(|(i, r)| r.iter().find(|e| e.0 == i).map_or(0.0, |e| e.1)).collect()
    }

    /// Entry (i, j) (zero when absent).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.rows[i].binary_search_by_key(&j, |e| e.0).map_or(0.0, |k| self.rows[i][k].1)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Jacobi-preconditioned conjugate gradients on an SPD `a`, to relative residual `tol`.
/// Returns the solution and the iterations taken.
pub fn pcg(a: &Sparse, b: &[f64], tol: f64, max_iter: usize) -> (Vec<f64>, usize) {
    let d: Vec<f64> = a.diag().iter().map(|&v| if v > 0.0 { 1.0 / v } else { 1.0 }).collect();
    let mut x = vec![0.0; a.n];
    let mut r = b.to_vec();
    let bn = dot(b, b).sqrt();
    if bn == 0.0 {
        return (x, 0);
    }
    let mut z: Vec<f64> = r.iter().zip(&d).map(|(r, d)| r * d).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for it in 0..max_iter {
        let ap = a.mul(&p);
        let alpha = rz / dot(&p, &ap);
        for i in 0..a.n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() <= tol * bn {
            return (x, it + 1);
        }
        z = r.iter().zip(&d).map(|(r, d)| r * d).collect();
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..a.n {
            p[i] = z[i] + beta * p[i];
        }
    }
    (x, max_iter)
}

// ------------------------------------------------------------------ dense Cholesky

/// In-place lower Cholesky of a row-major `n x n` SPD matrix; false if not SPD.
pub fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut v = a[j * n + j];
        for k in 0..j {
            v -= a[j * n + k] * a[j * n + k];
        }
        if !(v > 0.0) {
            return false;
        }
        let l = v.sqrt();
        a[j * n + j] = l;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / l;
        }
    }
    true
}

/// Solve `L L^T x = b` in place with the factor from `cholesky`.
pub fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

// ------------------------------------------------------------------ the network

/// An anchored cluster's intact bond network: free degrees of freedom, `K` and `M`.
pub struct Network {
    /// Chunk indices (structure-level) in local order.
    pub chunks: Vec<usize>,
    pub centers: Vec<Vec3>,
    /// Free degree of freedom index per local chunk and component (None: held).
    pub dof: Vec<[Option<usize>; 6]>,
    pub n: usize,
    /// Bonds as local chunk pairs.
    pub bonds: Vec<(usize, usize)>,
    pub k: Sparse,
    pub m: Sparse,
    /// Gravity load on the free degrees of freedom.
    pub gravity: Vec<f64>,
}

impl Network {
    /// The network of cluster `ci` (its live bonds at intact stiffness).
    pub fn of_cluster(m: &ReferenceSolver, ci: usize, gravity: Vec3) -> Network {
        let cl = &m.clusters[ci];
        let s = cl.structure;
        let st = &m.structures[s];
        let chunks = cl.chunks.clone();
        let mut local = vec![usize::MAX; st.chunks.len()];
        for (i, &c) in chunks.iter().enumerate() {
            local[c] = i;
        }
        let mut n = 0;
        let dof: Vec<[Option<usize>; 6]> = chunks
            .iter()
            .map(|&c| {
                let held = |d: usize| match st.chunks[c].support {
                    Support::Fixed => true,
                    Support::Pinned => d < 3,
                    Support::None => false,
                };
                std::array::from_fn(|d| {
                    (!held(d)).then(|| {
                        n += 1;
                        n - 1
                    })
                })
            })
            .collect();
        let mut ke: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut bonds = Vec::new();
        for &bi in &cl.bonds {
            let b = &m.bonds[s][bi];
            if !b.alive {
                continue;
            }
            let g = &b.geometry;
            let (la, lb) = (local[g.a], local[g.b]);
            bonds.push((la, lb));
            let mut k6 = b.stiffness.as_local();
            if let Some(r) = &b.rebar {
                k6.lin.z += r.k_axial;
                k6.lin.x += r.k_dowel;
                k6.lin.y += r.k_dowel;
            }
            let kd = [k6.lin.x, k6.lin.y, k6.lin.z, k6.ang.x, k6.ang.y, k6.ang.z];
            let axes = [g.t1, g.t2, g.normal];
            let rows: [[f64; 12]; 6] = std::array::from_fn(|comp| {
                let t = axes[comp % 3];
                let r: [Vec3; 4] = if comp < 3 { [-t, -(g.ra.cross(t)), t, g.rb.cross(t)] } else { [Vec3::ZERO, -t, Vec3::ZERO, t] };
                std::array::from_fn(|x| r[x / 3].to_array()[x % 3])
            });
            let dofs: [Option<usize>; 12] = std::array::from_fn(|x| if x < 6 { dof[la][x] } else { dof[lb][x - 6] });
            for x in 0..12 {
                let Some(i) = dofs[x] else { continue };
                for y in 0..12 {
                    let Some(j) = dofs[y] else { continue };
                    let v: f64 = (0..6).map(|c| kd[c] * rows[c][x] * rows[c][y]).sum();
                    *ke.entry((i, j)).or_default() += v;
                }
            }
        }
        let mut me: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut grav = vec![0.0; n];
        for (i, &c) in chunks.iter().enumerate() {
            let ch = &st.chunks[c];
            let mu = m.chunks[s][c].inertia_scale;
            let g = gravity.to_array();
            for d in 0..3 {
                if let Some(a) = dof[i][d] {
                    me.insert((a, a), ch.mass * mu);
                    grav[a] = ch.mass * g[d];
                }
            }
            for p in 0..3 {
                for q in 0..3 {
                    if let (Some(a), Some(b)) = (dof[i][3 + p], dof[i][3 + q]) {
                        me.insert((a, b), ch.inertia.m[p][q] * mu);
                    }
                }
            }
        }
        let centers = chunks.iter().map(|&c| st.chunks[c].center).collect();
        Network { chunks, centers, dof, n, bonds, k: Sparse::from_entries(n, ke), m: Sparse::from_entries(n, me), gravity: grav }
    }

    /// Free degrees of freedom of local chunk `c`.
    pub fn dofs_of(&self, c: usize) -> impl Iterator<Item = usize> + '_ {
        self.dof[c].iter().flatten().copied()
    }
}

// ------------------------------------------------------------------ partitions

/// Recursive coordinate bisection of the chunk centres into parts of at most `target`
/// chunks (split at the median of the widest extent). Part index per local chunk.
pub fn rcb(centers: &[Vec3], target: usize) -> Vec<usize> {
    fn split(idx: &mut [usize], centers: &[Vec3], target: usize, next: &mut usize, out: &mut [usize]) {
        if idx.len() <= target.max(1) {
            for &i in idx.iter() {
                out[i] = *next;
            }
            *next += 1;
            return;
        }
        let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
        for &i in idx.iter() {
            lo = lo.component_min(centers[i]);
            hi = hi.component_max(centers[i]);
        }
        let e = hi - lo;
        let axis = if e.x >= e.y && e.x >= e.z { 0 } else if e.y >= e.z { 1 } else { 2 };
        idx.sort_by(|&a, &b| centers[a].to_array()[axis].total_cmp(&centers[b].to_array()[axis]).then(a.cmp(&b)));
        let mid = idx.len() / 2;
        let (l, r) = idx.split_at_mut(mid);
        split(l, centers, target, next, out);
        split(r, centers, target, next, out);
    }
    let mut out = vec![0; centers.len()];
    let mut idx: Vec<usize> = (0..centers.len()).collect();
    let mut next = 0;
    split(&mut idx, centers, target, &mut next, &mut out);
    out
}

/// Chunks with no bond leaving their part: the eliminable interior of a partition.
pub fn interior_of(net: &Network, part: &[usize]) -> Vec<bool> {
    let mut interior = vec![true; net.chunks.len()];
    for &(a, b) in &net.bonds {
        if part[a] != part[b] {
            interior[a] = false;
            interior[b] = false;
        }
    }
    interior
}

// ------------------------------------------------------------------ condensation

/// One connected component of the eliminated chunks.
pub struct Component {
    /// Its free degrees of freedom (full numbering).
    pub inner: Vec<usize>,
    /// Retained degrees of freedom it is bonded to (full numbering).
    pub outer: Vec<usize>,
    /// Cholesky factor of `K_II` (row-major, `inner.len()` square).
    pub factor: Vec<f64>,
    /// `T = -K_II^-1 K_IR` (row-major, `inner.len() x outer.len()`).
    pub t: Vec<f64>,
}

/// The network with a chunk set eliminated.
pub struct Reduced {
    /// Reduced index per full degree of freedom (None: eliminated).
    pub keep: Vec<Option<usize>>,
    pub nr: usize,
    pub k: Sparse,
    pub m: Sparse,
    pub components: Vec<Component>,
}

impl Reduced {
    /// Eliminate the chunks flagged in `eliminate` (local chunk indices).
    pub fn new(net: &Network, eliminate: &[bool]) -> Reduced {
        // Components of the eliminated chunks over their bonds (union-find).
        let nc = net.chunks.len();
        let mut parent: Vec<usize> = (0..nc).collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for &(a, b) in &net.bonds {
            if eliminate[a] && eliminate[b] {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for c in (0..nc).filter(|&c| eliminate[c]) {
            groups.entry(root(&mut parent, c)).or_default().push(c);
        }
        let mut keep = vec![None; net.n];
        let mut nr = 0;
        for c in (0..nc).filter(|&c| !eliminate[c]) {
            for d in net.dofs_of(c) {
                keep[d] = Some(nr);
                nr += 1;
            }
        }
        // K_RR and M_RR.
        let mut ke: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut me: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for (src, dst) in [(&net.k, &mut ke), (&net.m, &mut me)] {
            for (i, row) in src.rows.iter().enumerate() {
                let Some(ri) = keep[i] else { continue };
                for &(j, v) in row {
                    if let Some(rj) = keep[j] {
                        *dst.entry((ri, rj)).or_default() += v;
                    }
                }
            }
        }
        let mut components = Vec::new();
        for (_, chunks) in groups {
            let inner: Vec<usize> = chunks.iter().flat_map(|&c| net.dofs_of(c)).collect();
            if inner.is_empty() {
                continue;
            }
            let ni = inner.len();
            let pos: BTreeMap<usize, usize> = inner.iter().enumerate().map(|(k, &d)| (d, k)).collect();
            let mut outer: Vec<usize> = inner.iter().flat_map(|&d| net.k.rows[d].iter().map(|e| e.0)).filter(|j| keep[*j].is_some()).collect();
            outer.sort_unstable();
            outer.dedup();
            let no = outer.len();
            let mut kii = vec![0.0; ni * ni];
            let mut kio = vec![0.0; ni * no];
            for (a, &d) in inner.iter().enumerate() {
                for &(j, v) in &net.k.rows[d] {
                    if let Some(&b) = pos.get(&j) {
                        kii[a * ni + b] = v;
                    } else if let Ok(b) = outer.binary_search(&j) {
                        kio[a * no + b] = v;
                    }
                }
            }
            assert!(cholesky(&mut kii, ni), "eliminated block not positive definite");
            // T = -K_II^-1 K_IO, column by column.
            let mut t = vec![0.0; ni * no];
            let mut col = vec![0.0; ni];
            for b in 0..no {
                for a in 0..ni {
                    col[a] = -kio[a * no + b];
                }
                cholesky_solve(&kii, ni, &mut col);
                for a in 0..ni {
                    t[a * no + b] = col[a];
                }
            }
            // K~ += K_OI T; M~ += T^T (M_II T) (M_II has a chunk's 6 x 6 block per row).
            let mut mt = vec![0.0; ni * no];
            for (a, &d) in inner.iter().enumerate() {
                for &(j, v) in &net.m.rows[d] {
                    if let Some(&b) = pos.get(&j) {
                        for y in 0..no {
                            mt[a * no + y] += v * t[b * no + y];
                        }
                    }
                }
            }
            for x in 0..no {
                for y in 0..no {
                    let (mut kv, mut mv) = (0.0, 0.0);
                    for a in 0..ni {
                        kv += kio[a * no + x] * t[a * no + y];
                        mv += t[a * no + x] * mt[a * no + y];
                    }
                    let (rx, ry) = (keep[outer[x]].unwrap(), keep[outer[y]].unwrap());
                    *ke.entry((rx, ry)).or_default() += kv;
                    *me.entry((rx, ry)).or_default() += mv;
                }
            }
            components.push(Component { inner, outer, factor: kii, t });
        }
        Reduced { keep, nr, k: Sparse::from_entries(nr, ke), m: Sparse::from_entries(nr, me), components }
    }

    /// The reduced load `f~ = f_R + T^T f_I`.
    pub fn load(&self, f: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.nr];
        for (d, &k) in self.keep.iter().enumerate() {
            if let Some(r) = k {
                out[r] = f[d];
            }
        }
        for c in &self.components {
            let no = c.outer.len();
            for (a, &d) in c.inner.iter().enumerate() {
                for b in 0..no {
                    out[self.keep[c.outer[b]].unwrap()] += c.t[a * no + b] * f[d];
                }
            }
        }
        out
    }

    /// Full displacements from the reduced ones: `u_I = T u_R + K_II^-1 f_I`.
    pub fn recover(&self, ur: &[f64], f: &[f64]) -> Vec<f64> {
        let mut u = vec![0.0; self.keep.len()];
        for (d, &k) in self.keep.iter().enumerate() {
            if let Some(r) = k {
                u[d] = ur[r];
            }
        }
        for c in &self.components {
            let (ni, no) = (c.inner.len(), c.outer.len());
            let mut ui0: Vec<f64> = c.inner.iter().map(|&d| f[d]).collect();
            cholesky_solve(&c.factor, ni, &mut ui0);
            for a in 0..ni {
                let tu: f64 = (0..no).map(|b| c.t[a * no + b] * ur[self.keep[c.outer[b]].unwrap()]).sum();
                u[c.inner[a]] = tu + ui0[a];
            }
        }
        u
    }

    /// Stored floats of the reduction (`K~`, `M~`, every `T` and `K_II` factor).
    pub fn floats(&self) -> usize {
        self.k.nnz() + self.m.nnz() + self.components.iter().map(|c| c.t.len() + c.inner.len() * (c.inner.len() + 1) / 2).sum::<usize>()
    }
}

// ------------------------------------------------------------------ spectrum

/// Highest eigenvalue of `M^-1 K` (power iteration, Rayleigh quotient; `M` solved by
/// PCG when it is not diagonal) and its mode shape. Converges from below.
pub fn top_eigen(k: &Sparse, m: &Sparse, iters: usize) -> (f64, Vec<f64>) {
    let n = k.n;
    if n == 0 {
        return (0.0, Vec::new());
    }
    let diagonal = m.rows.iter().enumerate().all(|(i, r)| r.iter().all(|e| e.0 == i));
    let block_diagonal_or_cg = |y: &[f64]| -> Vec<f64> {
        if diagonal {
            y.iter().zip(m.diag()).map(|(y, d)| y / d).collect()
        } else {
            pcg(m, y, 1e-12, 10_000).0
        }
    };
    let mut x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.618_033_988_75 + 0.1).fract() - 0.5).collect();
    let mut lambda = 0.0;
    for _ in 0..iters {
        let kx = k.mul(&x);
        let mx = m.mul(&x);
        lambda = dot(&x, &kx) / dot(&x, &mx);
        let mut next = block_diagonal_or_cg(&kx);
        let norm = dot(&next, &m.mul(&next)).sqrt();
        if !(norm > 0.0) {
            break;
        }
        next.iter_mut().for_each(|v| *v /= norm);
        x = next;
    }
    (lambda, x)
}

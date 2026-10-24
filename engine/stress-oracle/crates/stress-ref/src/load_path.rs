//! The load path (`capability::Profile::LoadPath`; stress-gpu fracture.slang `fr_lp_*`) in
//! f64, operation for operation where the GPU's f32 allows: the host's twin for the tests,
//! and the warm start the fracture world uploads at load.
//!
//! Each chunk has a small rigid displacement `x = (u, theta)` (structure frame); each bond
//! is the linear spring of [`BondGeometry::kinematics`] with its [`BondStiffness`], so the
//! bond graph's quasi-static equilibrium is `K x = f`: the stiffness solve of
//! `statics.rs`, linear, without damage. The loads `f` are gravity and the engine's
//! contact forces and torques, with inertia relief on a free body (its rigid acceleration's
//! d'Alembert loads, so a free body's loads balance and only its deformation loads its
//! bonds). A body's chunks hold still where it is supported; a body without a support holds
//! its lowest chunk (its loads balance, so the pin carries nothing but round-off).
//!
//! The solve is block Jacobi accelerated by Chebyshev's recurrence at its limit (Saad,
//! Iterative Methods for Sparse Linear Systems, 2nd ed., Alg. 12.1, with the block
//! diagonal as the preconditioner; its coefficients constant, the step carried over): a
//! fixed number of sweeps a frame from the last frame's displacements and step, as Blast's
//! stress solver iterates a few times a frame and carries its state over. Block Jacobi's
//! preconditioned spectrum lies in (0, 2]: every bond's term `B^T k B` is at most twice
//! its two diagonal blocks (`(p - q)^T k (p - q) <= 2 (p^T k p + q^T k q)`). Restarting
//! the recurrence each frame (or conjugate gradients restarted each frame) loses the slow
//! modes' progress: on a slender cantilever it converges 8 to 18 times slower (the test
//! below).

use crate::bond::{BondGeometry, BondStiffness, Local6};
use crate::math::{Mat3, Vec3};

/// A chunk's generalised displacement or load: (translation or force; rotation or moment).
pub type Dof = [f64; 6];

/// The interval of the preconditioned spectrum the recurrence is tuned to: its upper end
/// 2 as above; its lower end [`LOWER`] (modes below it still converge, more slowly).
pub const LOWER: f64 = 1e-6;
pub const UPPER: f64 = 2.0;

/// The coefficients `(c1, c2)` of Chebyshev's recurrence at its limit (the optimal
/// stationary second-order iteration for `[lower, UPPER]`; Saad, Alg. 12.1 as its `rho`
/// converges): each sweep `d = c1 d + c2 D^-1 r`, `x += d`, with `d` carried from sweep to
/// sweep and frame to frame. Every mode in `[lower, UPPER]` shrinks by
/// `sqrt(c1) = sigma - sqrt(sigma^2 - 1)` a sweep; one below converges monotonically.
pub fn momentum_coefficients(lower: f64) -> (f64, f64) {
    let theta = 0.5 * (UPPER + lower);
    let delta = 0.5 * (UPPER - lower);
    let sigma = theta / delta;
    let rho = 1.0 / (sigma + (sigma * sigma - 1.0).sqrt());
    (rho * rho, 2.0 * rho / delta)
}

#[derive(Clone, Debug)]
pub struct Chunk {
    /// Centre of mass (structure frame).
    pub centre: Vec3,
    pub mass: f64,
    /// Inertia about the centre (structure frame).
    pub inertia: Mat3,
    /// The engine body it belongs to.
    pub body: usize,
    pub support: bool,
}

#[derive(Clone, Debug)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
    pub centroid: Vec3,
    pub normal: Vec3,
    pub t1: Vec3,
    pub t2: Vec3,
    /// Local stiffness (shear t1, shear t2, axial n; bending t1, bending t2, torsion n).
    pub k: Local6,
}

impl Bond {
    /// The bond of `g` between global chunks `a` and `b`.
    pub fn new(a: usize, b: usize, g: &BondGeometry, k: &BondStiffness) -> Bond {
        Bond { a, b, centroid: g.centroid, normal: g.normal, t1: g.t1, t2: g.t2, k: k.as_local() }
    }

    fn to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.t1), v.dot(self.t2), v.dot(self.normal))
    }

    fn to_body(&self, l: Vec3) -> Vec3 {
        self.t1 * l.x + self.t2 * l.y + self.normal * l.z
    }
}

fn lin(x: &Dof) -> Vec3 {
    Vec3::new(x[0], x[1], x[2])
}

fn ang(x: &Dof) -> Vec3 {
    Vec3::new(x[3], x[4], x[5])
}

fn dof(l: Vec3, a: Vec3) -> Dof {
    [l.x, l.y, l.z, a.x, a.y, a.z]
}

/// The bond graph: chunks, bonds, and each chunk's bonds (ascending: the GPU's order).
#[derive(Clone, Debug)]
pub struct Network {
    pub chunks: Vec<Chunk>,
    pub bonds: Vec<Bond>,
    pub start: Vec<usize>,
    pub list: Vec<usize>,
}

impl Network {
    pub fn new(chunks: Vec<Chunk>, bonds: Vec<Bond>) -> Network {
        let n = chunks.len();
        let mut count = vec![0usize; n];
        for b in &bonds {
            count[b.a] += 1;
            count[b.b] += 1;
        }
        let mut start = vec![0usize; n + 1];
        for k in 0..n {
            start[k + 1] = start[k] + count[k];
        }
        let mut fill = start.clone();
        let mut list = vec![0usize; start[n]];
        for (i, b) in bonds.iter().enumerate() {
            for k in [b.a, b.b] {
                list[fill[k]] = i;
                fill[k] += 1;
            }
        }
        Network { chunks, bonds, start, list }
    }

    /// The bond's patch centroid relative to chunk `k`'s centre.
    pub fn arm(&self, i: usize, k: usize) -> Vec3 {
        self.bonds[i].centroid - self.chunks[k].centre
    }

    /// Bond `i`'s generalised deformation (local), as `BondGeometry::kinematics`.
    pub fn deformation(&self, i: usize, x: &[Dof]) -> Local6 {
        let b = &self.bonds[i];
        let (xa, xb) = (&x[b.a], &x[b.b]);
        let delta = (lin(xb) + ang(xb).cross(self.arm(i, b.b))) - (lin(xa) + ang(xa).cross(self.arm(i, b.a)));
        let phi = ang(xb) - ang(xa);
        Local6 { lin: b.to_local(delta), ang: b.to_local(phi) }
    }

    /// Bond `i`'s generalised force (local, tension positive).
    pub fn force(&self, i: usize, x: &[Dof]) -> Local6 {
        self.deformation(i, x).mul_elem(&self.bonds[i].k)
    }

    /// The load bond `i`'s force `q` puts on chunk `k` (one of its two), about `k`'s
    /// centre, as `BondGeometry::chunk_loads`.
    pub fn chunk_load(&self, i: usize, k: usize, q: &Local6) -> Dof {
        let b = &self.bonds[i];
        let f = b.to_body(q.lin);
        let m = b.to_body(q.ang);
        let r = self.arm(i, k);
        if k == b.a {
            dof(f, m + r.cross(f))
        } else {
            dof(-f, -m + r.cross(-f))
        }
    }

    /// `f - K x` (the bonds' loads added to the applied ones), every chunk, over the intact
    /// bonds.
    pub fn residual(&self, f: &[Dof], x: &[Dof], alive: &[bool]) -> Vec<Dof> {
        (0..self.chunks.len())
            .map(|k| {
                let mut r = f[k];
                for &i in &self.list[self.start[k]..self.start[k + 1]] {
                    if !alive[i] {
                        continue;
                    }
                    let l = self.chunk_load(i, k, &self.force(i, x));
                    for j in 0..6 {
                        r[j] += l[j];
                    }
                }
                r
            })
            .collect()
    }

    /// Chunk `k`'s 6 x 6 diagonal block of `K` over the intact bonds: `J^T diag(k) J` per
    /// bond, `J = [[R, -R [r]x], [0, R]]` (`R`'s rows t1, t2, n; `r` the bond's arm).
    pub fn block(&self, k: usize, alive: &[bool]) -> [[f64; 6]; 6] {
        let mut d = [[0.0; 6]; 6];
        for &i in &self.list[self.start[k]..self.start[k + 1]] {
            if !alive[i] {
                continue;
            }
            let b = &self.bonds[i];
            let rows = [b.t1, b.t2, b.normal];
            let r = self.arm(i, k);
            let kl = [b.k.lin.x, b.k.lin.y, b.k.lin.z, b.k.ang.x, b.k.ang.y, b.k.ang.z];
            let mut j = [[0.0; 6]; 6];
            for (a, row) in rows.iter().enumerate() {
                // Translation: R; rotation: -R [r]x = rows . (theta x r) = (r x row) . theta.
                let c = r.cross(*row);
                j[a] = [row.x, row.y, row.z, c.x, c.y, c.z];
                j[3 + a] = [0.0, 0.0, 0.0, row.x, row.y, row.z];
            }
            for p in 0..6 {
                for q in 0..6 {
                    let mut s = 0.0;
                    for a in 0..6 {
                        s += j[a][p] * kl[a] * j[a][q];
                    }
                    d[p][q] += s;
                }
            }
        }
        d
    }

    /// Which chunks hold still: supports, and each unsupported body's lowest chunk.
    pub fn fixed(&self) -> Vec<bool> {
        let bodies = self.chunks.iter().map(|c| c.body + 1).max().unwrap_or(0);
        let mut supported = vec![false; bodies];
        let mut lowest = vec![usize::MAX; bodies];
        for (k, c) in self.chunks.iter().enumerate() {
            supported[c.body] |= c.support;
            lowest[c.body] = lowest[c.body].min(k);
        }
        self.chunks.iter().enumerate().map(|(k, c)| c.support || (!supported[c.body] && lowest[c.body] == k)).collect()
    }

    /// Inertia relief: every unsupported body's loads less the d'Alembert loads of the
    /// rigid acceleration they give it (`a = F / M`, `alpha = I^-1 tau` about its centre of
    /// mass), so they balance.
    pub fn relieve(&self, f: &[Dof]) -> Vec<Dof> {
        let bodies = self.chunks.iter().map(|c| c.body + 1).max().unwrap_or(0);
        let mut supported = vec![false; bodies];
        let mut mass = vec![0.0; bodies];
        let mut first = vec![Vec3::ZERO; bodies];
        for c in &self.chunks {
            supported[c.body] |= c.support;
            mass[c.body] += c.mass;
            first[c.body] += c.centre * c.mass;
        }
        let com: Vec<Vec3> = (0..bodies).map(|b| if mass[b] > 0.0 { first[b] * (1.0 / mass[b]) } else { Vec3::ZERO }).collect();
        let mut inertia = vec![Mat3::ZERO; bodies];
        let mut force = vec![Vec3::ZERO; bodies];
        let mut torque = vec![Vec3::ZERO; bodies];
        for (k, c) in self.chunks.iter().enumerate() {
            let r = c.centre - com[c.body];
            inertia[c.body] += c.inertia + (Mat3::diag(Vec3::splat(r.norm2())) - Mat3::outer(r, r)) * c.mass;
            force[c.body] += lin(&f[k]);
            torque[c.body] += ang(&f[k]) + r.cross(lin(&f[k]));
        }
        self.chunks
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let b = c.body;
                if supported[b] || !(mass[b] > 0.0) {
                    return f[k];
                }
                let a = force[b] * (1.0 / mass[b]);
                let alpha = inertia[b].inverse().map_or(Vec3::ZERO, |i| i * torque[b]);
                let r = c.centre - com[b];
                dof(lin(&f[k]) - (a + alpha.cross(r)) * c.mass, ang(&f[k]) - c.inertia * alpha)
            })
            .collect()
    }

    /// The block diagonal's factors (`None` for a fixed chunk, or one whose block is not
    /// positive definite: it holds still, and the second value says so).
    pub fn factors(&self, fixed: &[bool], alive: &[bool]) -> (Vec<Option<[f64; 21]>>, bool) {
        let mut ok = true;
        let f = (0..self.chunks.len())
            .map(|k| {
                if fixed[k] {
                    return None;
                }
                let l = cholesky6(&self.block(k, alive));
                ok &= l.is_some();
                l
            })
            .collect();
        (f, ok)
    }

    /// `sweeps` sweeps of the recurrence ([`momentum_coefficients`]) on `K x = f`, from `x`
    /// and the carried step `d` (fixed chunks keep theirs, their step zero). Returns
    /// whether every free chunk's block factored.
    pub fn sweep(&self, f: &[Dof], fixed: &[bool], alive: &[bool], x: &mut [Dof], d: &mut [Dof], sweeps: u32, lower: f64) -> bool {
        let (factors, ok) = self.factors(fixed, alive);
        let (c1, c2) = momentum_coefficients(lower);
        for _ in 0..sweeps {
            let r = self.residual(f, x, alive);
            for k in 0..self.chunks.len() {
                let Some(l) = &factors[k] else {
                    d[k] = [0.0; 6];
                    continue;
                };
                let z = solve6(l, &r[k]);
                for j in 0..6 {
                    d[k][j] = c1 * d[k][j] + c2 * z[j];
                    x[k][j] += d[k][j];
                }
            }
        }
        ok
    }

    /// `K x = f` solved by conjugate gradients preconditioned with the block diagonal, from
    /// `x` (fixed chunks keep theirs), to `|r| <= tol |f|` or `max_iterations`. Returns the
    /// iterations and the final relative residual.
    pub fn solve(&self, f: &[Dof], fixed: &[bool], alive: &[bool], x: &mut [Dof], tol: f64, max_iterations: usize) -> (usize, f64) {
        let n = self.chunks.len();
        let factors: Vec<Option<[f64; 21]>> = (0..n).map(|k| if fixed[k] { None } else { cholesky6(&self.block(k, alive)) }).collect();
        let mask = |v: &mut Vec<Dof>| {
            for k in 0..n {
                if factors[k].is_none() {
                    v[k] = [0.0; 6];
                }
            }
        };
        let dotv = |a: &[Dof], b: &[Dof]| a.iter().zip(b).map(|(p, q)| (0..6).map(|j| p[j] * q[j]).sum::<f64>()).sum::<f64>();
        let precondition = |r: &[Dof]| -> Vec<Dof> { (0..n).map(|k| factors[k].as_ref().map_or([0.0; 6], |l| solve6(l, &r[k]))).collect() };
        let mut fm: Vec<Dof> = f.to_vec();
        mask(&mut fm);
        let norm_f = dotv(&fm, &fm).sqrt().max(1e-300);
        let mut r = self.residual(f, x, alive);
        mask(&mut r);
        let mut z = precondition(&r);
        let mut p = z.clone();
        let mut rz = dotv(&r, &z);
        let zero = vec![[0.0; 6]; n];
        for it in 0..max_iterations {
            let rel = dotv(&r, &r).sqrt() / norm_f;
            if rel <= tol {
                return (it, rel);
            }
            // K p = -(residual of p with no load).
            let mut kp: Vec<Dof> = self.residual(&zero, &p, alive).iter().map(|v| v.map(|c| -c)).collect();
            mask(&mut kp);
            let pkp = dotv(&p, &kp);
            if !(pkp > 0.0) {
                return (it, rel);
            }
            let alpha = rz / pkp;
            for k in 0..n {
                for j in 0..6 {
                    x[k][j] += alpha * p[k][j];
                    r[k][j] -= alpha * kp[k][j];
                }
            }
            z = precondition(&r);
            let rz_new = dotv(&r, &z);
            let beta = rz_new / rz;
            rz = rz_new;
            for k in 0..n {
                for j in 0..6 {
                    p[k][j] = z[k][j] + beta * p[k][j];
                }
            }
        }
        (max_iterations, dotv(&r, &r).sqrt() / norm_f)
    }
}

/// The lower Cholesky factor of a symmetric 6 x 6 matrix, packed by rows (`L[i][j]`,
/// `j <= i`, at `i (i + 1) / 2 + j`); `None` unless positive definite.
pub fn cholesky6(a: &[[f64; 6]; 6]) -> Option<[f64; 21]> {
    let mut l = [0.0; 21];
    let at = |i: usize, j: usize| i * (i + 1) / 2 + j;
    for i in 0..6 {
        for j in 0..=i {
            let mut s = a[i][j];
            for p in 0..j {
                s -= l[at(i, p)] * l[at(j, p)];
            }
            if i == j {
                if !(s > 0.0) {
                    return None;
                }
                l[at(i, i)] = s.sqrt();
            } else {
                l[at(i, j)] = s / l[at(j, j)];
            }
        }
    }
    Some(l)
}

/// `(L L^T)^-1 b` for a factor of [`cholesky6`].
pub fn solve6(l: &[f64; 21], b: &Dof) -> Dof {
    let at = |i: usize, j: usize| i * (i + 1) / 2 + j;
    let mut y = [0.0; 6];
    for i in 0..6 {
        let mut s = b[i];
        for p in 0..i {
            s -= l[at(i, p)] * y[p];
        }
        y[i] = s / l[at(i, i)];
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut s = y[i];
        for p in i + 1..6 {
            s -= l[at(p, i)] * x[p];
        }
        x[i] = s / l[at(i, i)];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ReferenceSolver;

    /// The load path's network of a solver's structure 0 (one body; supports as given).
    pub fn network_of(sv: &ReferenceSolver) -> Network {
        let st = &sv.structures[0];
        let chunks = st.chunks.iter().map(|c| Chunk { centre: c.center, mass: c.mass, inertia: c.inertia, body: 0, support: c.support != crate::scene::Support::None }).collect();
        let bonds = st.bonds.iter().map(|b| Bond::new(b.geometry.a, b.geometry.b, &b.geometry, &b.stiffness)).collect();
        Network::new(chunks, bonds)
    }

    fn gravity_loads(net: &Network, g: Vec3) -> Vec<Dof> {
        net.chunks.iter().map(|c| dof(g * c.mass, Vec3::ZERO)).collect()
    }

    /// The cantilever (20 chunks of 0.1 m, the first a support) under its own weight: the
    /// root bond carries the beam's whole weight as shear and its moment `w L^2 / 2` as
    /// bending (statics of a uniform cantilever, exact for the chunk model: each chunk's
    /// weight acts at its centre), solved directly and by enough warm Chebyshev frames.
    #[test]
    fn a_cantilever_root_carries_its_weight_and_moment() {
        let scene = crate::builders::cantilever(20, "cantilever");
        let sv = ReferenceSolver::new(&scene);
        let net = network_of(&sv);
        let g = Vec3::new(0.0, 0.0, -9.81);
        let f = gravity_loads(&net, g);
        let fixed = net.fixed();
        let alive = vec![true; net.bonds.len()];
        let mut x = vec![[0.0; 6]; net.chunks.len()];
        let (its, rel) = net.solve(&f, &fixed, &alive, &mut x, 1e-12, 10_000);
        assert!(rel <= 1e-12, "CG: {its} iterations, residual {rel:e}");
        // The root bond: the one at the support.
        let root = (0..net.bonds.len()).find(|&i| fixed[net.bonds[i].a] || fixed[net.bonds[i].b]).expect("a support bond");
        let free_mass: f64 = net.chunks.iter().zip(&fixed).filter(|(_, f)| !**f).map(|(c, _)| c.mass).sum();
        let support = if fixed[net.bonds[root].a] { net.bonds[root].a } else { net.bonds[root].b };
        // Moment of the free chunks' weights about the root bond's centroid.
        let c = net.bonds[root].centroid;
        let moment: Vec3 = net.chunks.iter().zip(&fixed).filter(|(_, f)| !**f).map(|(ch, _)| (ch.centre - c).cross(g * ch.mass)).fold(Vec3::ZERO, |a, b| a + b);
        let q = net.force(root, &x);
        let b = &net.bonds[root];
        let l = net.chunk_load(root, support, &q);
        // The bond's load on the support equals the free part's weight and its moment about
        // the centroid (`chunk_load` is about the support's centre: shift it).
        let force = lin(&l);
        let about_centroid = ang(&l) - (b.centroid - net.chunks[support].centre).cross(force);
        let w = g * free_mass;
        println!("root force {force:?} (weight {w:?}), moment {about_centroid:?} ({moment:?})");
        assert!((force - w).norm() <= 1e-9 * w.norm(), "root force {force:?} against the weight {w:?}");
        assert!((about_centroid - moment).norm() <= 1e-9 * moment.norm(), "root moment {about_centroid:?} against {moment:?}");
        // Frames of 16 sweeps reach the direct solution's bond forces (to 1e-3): from rest,
        // and from the settled beam after a sideways load at its tip. Measured (2026-10-24,
        // this beam): 189 and 237 frames; restarted PCG of 16 iterations a frame takes 3443
        // and 3219, a lower end of 1e-4 796 and 791.
        let close = |y: &[Dof], q: &Local6| {
            let p = net.force(root, y);
            (p.lin - q.lin).norm() / q.lin.norm() + (p.ang - q.ang).norm() / q.ang.norm()
        };
        let frames_to = |f: &[Dof], start: &[Dof], q: &Local6| {
            let mut y = start.to_vec();
            let mut d = vec![[0.0; 6]; net.chunks.len()];
            for frame in 1..=400 {
                assert!(net.sweep(f, &fixed, &alive, &mut y, &mut d, 16, LOWER), "every free block factors");
                if close(&y, q) <= 1e-3 {
                    return frame;
                }
            }
            panic!("not within 1e-3 after 400 frames: {:e}", close(&y, q));
        };
        let rest = frames_to(&f, &vec![[0.0; 6]; net.chunks.len()], &q);
        let mut f2 = f.clone();
        f2[net.chunks.len() - 1][1] += 200.0;
        let mut x2 = x.clone();
        net.solve(&f2, &fixed, &alive, &mut x2, 1e-13, 10_000);
        let tip = frames_to(&f2, &x, &net.force(root, &x2));
        println!("frames of 16 sweeps to 1e-3: {rest} from rest, {tip} after a tip load");
    }

    /// A free beam (the cantilever unsupported) under gravity alone falls freely: inertia
    /// relief takes every load away, so no bond carries anything.
    #[test]
    fn a_free_body_in_free_fall_carries_nothing() {
        let mut scene = crate::builders::cantilever(20, "free beam");
        for c in &mut scene.bodies[0].chunks {
            c.support = crate::scene::Support::None;
        }
        let sv = ReferenceSolver::new(&scene);
        let net = network_of(&sv);
        let f = net.relieve(&gravity_loads(&net, Vec3::new(0.0, 0.0, -9.81)));
        let worst = f.iter().map(|d| lin(d).norm() + ang(d).norm()).fold(0.0, f64::max);
        let weight = net.chunks[0].mass * 9.81;
        assert!(worst <= 1e-9 * weight, "relieved loads {worst:e} N");
    }

    /// A free beam pushed at one end: relief leaves loads that balance (no net force or
    /// moment), and the pinned chunk then carries nothing.
    #[test]
    fn relieved_loads_balance_and_the_pin_carries_nothing() {
        let mut scene = crate::builders::cantilever(20, "free beam");
        for c in &mut scene.bodies[0].chunks {
            c.support = crate::scene::Support::None;
        }
        let sv = ReferenceSolver::new(&scene);
        let net = network_of(&sv);
        let mut f = gravity_loads(&net, Vec3::new(0.0, 0.0, -9.81));
        let last = net.chunks.len() - 1;
        f[last][1] += 500.0; // a sideways push at the free end
        let f = net.relieve(&f);
        let (sum, moment) = f.iter().zip(&net.chunks).fold((Vec3::ZERO, Vec3::ZERO), |(s, m), (d, c)| (s + lin(d), m + ang(d) + c.centre.cross(lin(d))));
        assert!(sum.norm() <= 1e-9 && moment.norm() <= 1e-9, "relieved loads: net force {sum:?}, moment {moment:?}");
        let fixed = net.fixed();
        let alive = vec![true; net.bonds.len()];
        let mut x = vec![[0.0; 6]; net.chunks.len()];
        net.solve(&f, &fixed, &alive, &mut x, 1e-13, 10_000);
        let r = net.residual(&f, &x, &alive);
        let pin = fixed.iter().position(|&p| p).expect("a pin");
        let reaction = lin(&r[pin]).norm() + ang(&r[pin]).norm();
        assert!(reaction <= 1e-6 * 500.0, "the pin carries {reaction:e}");
    }

    #[test]
    fn cholesky6_solves_a_positive_definite_system() {
        let mut a = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                a[i][j] = 1.0 / (1.0 + i as f64 + j as f64);
            }
            a[i][i] += 2.0;
        }
        let l = cholesky6(&a).expect("positive definite");
        let b = [1.0, -2.0, 3.0, 0.5, -0.25, 4.0];
        let x = solve6(&l, &b);
        for i in 0..6 {
            let s: f64 = (0..6).map(|j| a[i][j] * x[j]).sum();
            assert!((s - b[i]).abs() < 1e-12);
        }
        assert!(cholesky6(&[[0.0; 6]; 6]).is_none());
    }
}

//! JGS2 gate experiment (phase 7, `engine/docs/jgs2-notes.md`): does a
//! JGS2-style local solve converge a node–beam car's implicit step far faster
//! than the block Gauss–Seidel (VBD) solve Rubble runs today?
//!
//! One backward-Euler step of the elastic beam energy (no damping, no
//! plasticity, no contacts; the lowest nodes are held fixed as the ground) is
//! solved exactly by projected Newton with dense f64 solves, giving x*. Then,
//! from the same initial guess, each method's error ‖x − x*‖ / ‖x₀ − x*‖ is
//! printed per iteration:
//!
//! * `vbd`: coloured Gauss–Seidel, per node δx = −H_ii⁻¹ g_i (Rubble's solve).
//! * `jgs2-rest`: Jacobi, per node δx = −(φᵀHφ)⁻¹ φᵀg with φ the co-rotated
//!   rest-shape perturbation basis (JGS2 Eq. 14), the complementary terms summed
//!   exactly (no cubature).
//! * `jgs2-exact`: the same with φ from the current Hessian (Eq. 9); by the
//!   paper's derivation one iteration equals one Newton step. A check.
//!
//! ```bash
//! cargo run -p engine-softbody --release --example jgs2_probe -- car=sedan scene=gravity
//! cargo run -p engine-softbody --release --example jgs2_probe -- car=hires scene=impact
//! ```
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::hires::{hires_car_desc, HiresParams};
use engine_softbody::model::{Beam, KIND_NORMAL};
use engine_softbody::pack::{pack_soft, Placement, SpawnOptions};
use engine_softbody::procedural::{car_desc, CarParams};
use glam::{DMat3, DVec3, Quat, Vec3};

fn arg(name: &str) -> Option<String> {
    std::env::args()
        .skip(1)
        .find_map(|a| a.strip_prefix(&format!("{name}=")).map(str::to_string))
}

/// Dense symmetric positive definite matrix, Cholesky in place.
struct Dense {
    n: usize,
    a: Vec<f64>,
}

impl Dense {
    fn zeros(n: usize) -> Self {
        Self {
            n,
            a: vec![0.0; n * n],
        }
    }
    fn add_block(&mut self, i: usize, j: usize, m: DMat3) {
        for r in 0..3 {
            for c in 0..3 {
                self.a[(3 * i + r) * self.n + 3 * j + c] += m.col(c)[r];
            }
        }
    }
    /// Lower Cholesky factor in place; false if not positive definite.
    fn cholesky(&mut self) -> bool {
        let n = self.n;
        for j in 0..n {
            let mut d = self.a[j * n + j];
            for k in 0..j {
                d -= self.a[j * n + k] * self.a[j * n + k];
            }
            if d <= 0.0 {
                return false;
            }
            let d = d.sqrt();
            self.a[j * n + j] = d;
            for i in j + 1..n {
                let mut s = self.a[i * n + j];
                for k in 0..j {
                    s -= self.a[i * n + k] * self.a[j * n + k];
                }
                self.a[i * n + j] = s / d;
            }
        }
        true
    }
    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = b.to_vec();
        for i in 0..n {
            let mut s = y[i];
            for k in 0..i {
                s -= self.a[i * n + k] * y[k];
            }
            y[i] = s / self.a[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in i + 1..n {
                s -= self.a[k * n + i] * y[k];
            }
            y[i] = s / self.a[i * n + i];
        }
        y
    }
}

/// One implicit step of the beam energy over free nodes.
struct Step {
    /// Free node → solve index (None = fixed).
    idx: Vec<Option<usize>>,
    nfree: usize,
    mass: Vec<f64>,
    beams: Vec<(usize, usize, f64, f64)>, // a, b, k, rest
    y: Vec<DVec3>,
    rest: Vec<DVec3>,
    h: f64,
    adj: Vec<Vec<(usize, usize)>>, // node → (beam, other)
    colors: Vec<u32>,
}

impl Step {
    fn energy(&self, x: &[DVec3]) -> f64 {
        let mut e = 0.0;
        for i in 0..x.len() {
            if self.idx[i].is_some() {
                e += 0.5 * self.mass[i] / (self.h * self.h) * (x[i] - self.y[i]).length_squared();
            }
        }
        for &(a, b, k, l) in &self.beams {
            e += 0.5 * k * ((x[a] - x[b]).length() - l).powi(2);
        }
        e
    }

    /// Gradient (free nodes, indexed by node).
    fn gradient(&self, x: &[DVec3]) -> Vec<DVec3> {
        let mut g: Vec<DVec3> = (0..x.len())
            .map(|i| (x[i] - self.y[i]) * (self.mass[i] / (self.h * self.h)))
            .collect();
        for &(a, b, k, l) in &self.beams {
            let d = x[a] - x[b];
            let len = d.length().max(1e-12);
            let f = d / len * (k * (len - l));
            g[a] += f;
            g[b] -= f;
        }
        for i in 0..x.len() {
            if self.idx[i].is_none() {
                g[i] = DVec3::ZERO;
            }
        }
        g
    }

    /// Projected (PSD) beam Hessian block at x: k n nᵀ + max(0, k(ℓ − L)/ℓ)(I − n nᵀ).
    fn beam_block(&self, x: &[DVec3], e: usize) -> DMat3 {
        let (a, b, k, l) = self.beams[e];
        let d = x[a] - x[b];
        let len = d.length().max(1e-12);
        let n = d / len;
        let nn = DMat3::from_cols(n * n.x, n * n.y, n * n.z);
        nn * k + (DMat3::IDENTITY - nn) * (k * (len - l) / len).max(0.0)
    }

    fn hessian(&self, x: &[DVec3]) -> Dense {
        let mut h = Dense::zeros(3 * self.nfree);
        for i in 0..x.len() {
            if let Some(p) = self.idx[i] {
                h.add_block(p, p, DMat3::IDENTITY * (self.mass[i] / (self.h * self.h)));
            }
        }
        for e in 0..self.beams.len() {
            let (a, b, _, _) = self.beams[e];
            let m = self.beam_block(x, e);
            let (pa, pb) = (self.idx[a], self.idx[b]);
            if let Some(pa) = pa {
                h.add_block(pa, pa, m);
            }
            if let Some(pb) = pb {
                h.add_block(pb, pb, m);
            }
            if let (Some(pa), Some(pb)) = (pa, pb) {
                h.add_block(pa, pb, -m);
                h.add_block(pb, pa, -m);
            }
        }
        h
    }

    fn flat(&self, v: &[DVec3]) -> Vec<f64> {
        let mut f = vec![0.0; 3 * self.nfree];
        for i in 0..v.len() {
            if let Some(p) = self.idx[i] {
                f[3 * p..3 * p + 3].copy_from_slice(&v[i].to_array());
            }
        }
        f
    }

    /// Projected Newton with a backtracking line search, until the step stops
    /// moving anything (below 1e-13 m) or the gradient falls 1e-12 of its start.
    fn newton(&self, x0: &[DVec3]) -> (Vec<DVec3>, usize) {
        let mut x = x0.to_vec();
        let mut g0 = 0.0;
        for it in 0..60 {
            let g = self.gradient(&x);
            let gn = g.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
            if it == 0 {
                g0 = gn;
            }
            if gn <= 1e-12 * g0 {
                return (x, it);
            }
            let mut h = self.hessian(&x);
            assert!(h.cholesky(), "Hessian not positive definite");
            let dx = h.solve(&self.flat(&g));
            let e0 = self.energy(&x);
            let mut t = 1.0;
            loop {
                let mut xn = x.clone();
                for i in 0..x.len() {
                    if let Some(p) = self.idx[i] {
                        xn[i] -= DVec3::new(dx[3 * p], dx[3 * p + 1], dx[3 * p + 2]) * t;
                    }
                }
                if self.energy(&xn) <= e0 || t < 1e-6 {
                    let moved = dx.iter().map(|d| d * d).sum::<f64>().sqrt() * t;
                    x = xn;
                    if moved < 1e-13 {
                        return (x, it + 1);
                    }
                    break;
                }
                t *= 0.5;
            }
        }
        (x, 60)
    }

    /// Rubble's solve: coloured Gauss–Seidel, per node δx = −H_ii⁻¹ g_i.
    fn vbd_iteration(&self, x: &mut [DVec3]) {
        let ncol = self.colors.iter().copied().max().unwrap_or(0) + 1;
        for c in 0..ncol {
            let mut upd = Vec::new();
            for i in 0..x.len() {
                if self.idx[i].is_none() || self.colors[i] != c {
                    continue;
                }
                let mut g = (x[i] - self.y[i]) * (self.mass[i] / (self.h * self.h));
                let mut hm = DMat3::IDENTITY * (self.mass[i] / (self.h * self.h));
                for &(e, o) in &self.adj[i] {
                    let (_, _, k, l) = self.beams[e];
                    let d = x[i] - x[o];
                    let len = d.length().max(1e-12);
                    g += d / len * (k * (len - l));
                    hm += self.beam_block(x, e);
                }
                upd.push((i, -(hm.inverse() * g)));
            }
            for (i, d) in upd {
                x[i] += d;
            }
        }
    }

    /// JGS2 Jacobi iteration with basis φ_i (3N×3 columns per free node, the
    /// node's own block I): δx_i = −(φᵀHφ)⁻¹ φᵀ g at the iterate, all nodes at once.
    fn jgs2_iteration(&self, x: &mut [DVec3], basis: &dyn Fn(usize, &[DVec3]) -> Vec<DVec3>) {
        let upd = self.jgs2_update(x, basis);
        for (i, d) in upd {
            x[i] += d;
        }
    }

    /// The same Jacobi update scaled back (halving, Armijo on the true energy)
    /// until the energy falls: a global line search over the whole sweep.
    fn jgs2_iteration_ls(&self, x: &mut [DVec3], basis: &dyn Fn(usize, &[DVec3]) -> Vec<DVec3>) {
        let upd = self.jgs2_update(x, basis);
        let e0 = self.energy(x);
        let g = self.gradient(x);
        let slope: f64 = upd.iter().map(|&(i, d)| g[i].dot(d)).sum();
        let mut t = 1.0;
        for _ in 0..20 {
            let mut xn = x.to_vec();
            for &(i, d) in &upd {
                xn[i] += d * t;
            }
            if self.energy(&xn) <= e0 + 1e-4 * t * slope.min(0.0) {
                x.copy_from_slice(&xn);
                return;
            }
            t *= 0.5;
        }
    }

    fn jgs2_update(
        &self,
        x: &[DVec3],
        basis: &dyn Fn(usize, &[DVec3]) -> Vec<DVec3>,
    ) -> Vec<(usize, DVec3)> {
        let g = self.gradient(x);
        let mut upd = Vec::new();
        for i in 0..x.len() {
            if self.idx[i].is_none() {
                continue;
            }
            // φ columns as per-node 3×3 blocks: phi[v] maps δx_i to node v's motion.
            let phi = basis(i, x);
            let blocks: Vec<DMat3> = (0..x.len())
                .map(|v| DMat3::from_cols(phi[3 * v], phi[3 * v + 1], phi[3 * v + 2]))
                .collect();
            // φᵀg and φᵀHφ, summed over the sparse Hessian (inertia + beams).
            let mut rg = DVec3::ZERO;
            let mut rh = DMat3::ZERO;
            for v in 0..x.len() {
                if self.idx[v].is_none() {
                    continue;
                }
                rg += blocks[v].transpose() * g[v];
                rh += blocks[v].transpose()
                    * (DMat3::IDENTITY * (self.mass[v] / (self.h * self.h)))
                    * blocks[v];
            }
            for e in 0..self.beams.len() {
                let (a, b, _, _) = self.beams[e];
                // The beam's 6×6 Hessian [K −K; −K K] projected: (Pa − Pb)ᵀ K (Pa − Pb).
                let pa = if self.idx[a].is_some() {
                    blocks[a]
                } else {
                    DMat3::ZERO
                };
                let pb = if self.idx[b].is_some() {
                    blocks[b]
                } else {
                    DMat3::ZERO
                };
                let q = pa - pb;
                if q.abs_diff_eq(DMat3::ZERO, 0.0) {
                    continue;
                }
                rh += q.transpose() * self.beam_block(x, e) * q;
            }
            upd.push((i, -(rh.inverse() * rg)));
        }
        upd
    }
}

/// Per-node rotation from its beam neighbours (polar part of the local
/// deformation gradient, least squares over its edges).
fn node_rotations(step: &Step, x: &[DVec3]) -> Vec<DMat3> {
    (0..x.len())
        .map(|i| {
            let mut a = DMat3::ZERO;
            let mut b = DMat3::ZERO;
            for &(_, o) in &step.adj[i] {
                let (dx, dX) = (x[o] - x[i], step.rest[o] - step.rest[i]);
                a += DMat3::from_cols(dx * dX.x, dx * dX.y, dx * dX.z);
                b += DMat3::from_cols(dX * dX.x, dX * dX.y, dX * dX.z);
            }
            if b.determinant().abs() < 1e-12 {
                return DMat3::IDENTITY;
            }
            let mut r = a * b.inverse();
            for _ in 0..30 {
                if r.determinant().abs() < 1e-12 {
                    return DMat3::IDENTITY;
                }
                r = (r + r.inverse().transpose()) * 0.5;
            }
            if r.determinant() < 0.0 {
                DMat3::IDENTITY
            } else {
                r
            }
        })
        .collect()
}

/// φ_i for every free node from a factored Hessian: Ū_i = H⁻¹ S_iᵀ (S_i H⁻¹ S_iᵀ)⁻¹
/// (JGS2 Eqs. 19–23), as per-node 3×3 blocks (columns flattened per node).
fn full_basis(step: &Step, h: &Dense, i: usize) -> Vec<DVec3> {
    let p = step.idx[i].unwrap();
    let n = 3 * step.nfree;
    let mut cols = Vec::new();
    for c in 0..3 {
        let mut e = vec![0.0; n];
        e[3 * p + c] = 1.0;
        cols.push(h.solve(&e));
    }
    // Compliance block C = S H⁻¹ Sᵀ and its inverse.
    let cmat = DMat3::from_cols(
        DVec3::new(cols[0][3 * p], cols[0][3 * p + 1], cols[0][3 * p + 2]),
        DVec3::new(cols[1][3 * p], cols[1][3 * p + 1], cols[1][3 * p + 2]),
        DVec3::new(cols[2][3 * p], cols[2][3 * p + 1], cols[2][3 * p + 2]),
    );
    let cinv = cmat.inverse();
    let mut out = vec![DVec3::ZERO; 3 * step.idx.len()];
    for v in 0..step.idx.len() {
        let Some(q) = step.idx[v] else { continue };
        // Block v: rows of H⁻¹Sᵀ at node v, times C⁻¹.
        let hv = DMat3::from_cols(
            DVec3::new(cols[0][3 * q], cols[0][3 * q + 1], cols[0][3 * q + 2]),
            DVec3::new(cols[1][3 * q], cols[1][3 * q + 1], cols[1][3 * q + 2]),
            DVec3::new(cols[2][3 * q], cols[2][3 * q + 1], cols[2][3 * q + 2]),
        );
        let blk = hv * cinv;
        out[3 * v] = blk.col(0);
        out[3 * v + 1] = blk.col(1);
        out[3 * v + 2] = blk.col(2);
    }
    out
}

fn main() {
    let car = arg("car").unwrap_or_else(|| "sedan".into());
    let scene = arg("scene").unwrap_or_else(|| "gravity".into());
    let iters: usize = arg("iters").and_then(|v| v.parse().ok()).unwrap_or(40);
    let hz: f64 = arg("hz").and_then(|v| v.parse().ok()).unwrap_or(240.0);
    let desc: SoftVehicleDesc = match car.as_str() {
        "hires" => hires_car_desc(&HiresParams::default()).0,
        "default" => car_desc(&CarParams::default()).0,
        _ => car_desc(&CarParams::sedan()).0,
    };
    let p = pack_soft(
        &desc,
        Placement {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        },
        &SpawnOptions {
            rigid_core: false,
            ..Default::default()
        },
        0,
        -1,
    );
    let n = desc.nodes.len();
    let x0: Vec<DVec3> = p
        .bodies
        .iter()
        .map(|b| Vec3::from_array(b.position).as_dvec3())
        .collect();
    // The ground: nodes within 3 cm of the lowest are held fixed.
    let ymin = x0.iter().map(|v| v.y).fold(f64::MAX, f64::min);
    let mut idx = vec![None; n];
    let mut nfree = 0;
    for i in 0..n {
        if x0[i].y > ymin + 0.03 && !desc.nodes[i].fixed {
            idx[i] = Some(nfree);
            nfree += 1;
        }
    }
    let beams: Vec<(usize, usize, f64, f64)> = p
        .beams
        .iter()
        .filter(|b: &&Beam| b.kind == KIND_NORMAL || b.kind == 5)
        .map(|b| {
            (
                b.a as usize,
                b.b as usize,
                f64::from(b.spring),
                f64::from(b.rest),
            )
        })
        .collect();
    let mut adj = vec![Vec::new(); n];
    for (e, &(a, b, _, _)) in beams.iter().enumerate() {
        adj[a].push((e, b));
        adj[b].push((e, a));
    }
    let h = 1.0 / hz;
    let g = DVec3::new(0.0, -9.81, 0.0);
    // Velocity: at rest (gravity), or the front third moving back at 15 m/s
    // (an impact's first step: the nose stopped by a wall, the rest still coming).
    let zmax = x0.iter().map(|v| v.z).fold(f64::MIN, f64::max);
    let v: Vec<DVec3> = x0
        .iter()
        .map(|p| match scene.as_str() {
            "impact" if p.z < zmax - 1.2 => DVec3::new(0.0, 0.0, 15.0),
            _ => DVec3::ZERO,
        })
        .collect();
    let y: Vec<DVec3> = (0..n).map(|i| x0[i] + v[i] * h + g * h * h).collect();
    let edges: Vec<(u32, u32)> = beams
        .iter()
        .map(|&(a, b, _, _)| (a as u32, b as u32))
        .collect();
    let (colors, _) = engine_softbody::color::color_graph(n, &edges);
    let mut step = Step {
        idx,
        nfree,
        mass: p.masses.iter().map(|&m| f64::from(m)).collect(),
        beams,
        y: y.clone(),
        rest: x0.clone(),
        h,
        adj,
        colors,
    };
    // Initial guess: the inertial target (fixed nodes stay).
    let guess: Vec<DVec3> = (0..n)
        .map(|i| if step.idx[i].is_some() { y[i] } else { x0[i] })
        .collect();
    // Rest-shape basis (factor H̄ = inertia + rest stiffness once).
    let t = std::time::Instant::now();
    let mut hbar = step.hessian(&x0);
    assert!(hbar.cholesky());
    let rest_basis: Vec<Option<Vec<DVec3>>> = (0..n)
        .map(|i| step.idx[i].map(|_| full_basis(&step, &hbar, i)))
        .collect();
    eprintln!("rest basis: {:.1} s", t.elapsed().as_secs_f32());
    // broken=f: a share f of the beams in the front 1.2 m break after the basis
    // was built (every k-th by index), so the basis is stale as in a crash.
    if let Some(f) = arg("broken").and_then(|v| v.parse::<f64>().ok()) {
        let every = (1.0 / f.max(1e-6)).round().max(1.0) as usize;
        let mut k = 0usize;
        let before = step.beams.len();
        let front =
            |&(a, b, _, _): &(usize, usize, f64, f64)| 0.5 * (x0[a].z + x0[b].z) > zmax - 1.2;
        step.beams.retain(|e| {
            if !front(e) {
                return true;
            }
            k += 1;
            k % every != 0
        });
        let mut adj = vec![Vec::new(); n];
        for (e, &(a, b, _, _)) in step.beams.iter().enumerate() {
            adj[a].push((e, b));
            adj[b].push((e, a));
        }
        step.adj = adj;
        eprintln!(
            "broke {} of {before} beams after the basis",
            before - step.beams.len()
        );
    }
    let t = std::time::Instant::now();
    let (xs, newton_its) = step.newton(&guess);
    eprintln!(
        "{car} {scene}: {n} nodes, {nfree} free, {} beams, h = 1/{hz}; Newton converged in {newton_its} iterations ({:.1} s)",
        step.beams.len(),
        t.elapsed().as_secs_f32()
    );
    let err0 = (0..n)
        .map(|i| (guess[i] - xs[i]).length_squared())
        .sum::<f64>()
        .sqrt();
    let err = |x: &[DVec3]| {
        (0..n)
            .map(|i| (x[i] - xs[i]).length_squared())
            .sum::<f64>()
            .sqrt()
            / err0
    };
    // The exact-basis check refactors the Hessian and solves 3N times per node per
    // iteration: skip it on big cars (exact=0).
    let methods: Vec<&str> = if arg("exact").as_deref() == Some("0") {
        vec!["vbd", "jgs2-rest", "jgs2-rest-ls"]
    } else {
        vec!["vbd", "jgs2-rest", "jgs2-rest-ls", "jgs2-exact"]
    };
    let mut curves: Vec<Vec<f64>> = Vec::new();
    for &m in &methods {
        let mut x = guess.clone();
        let mut curve = Vec::new();
        for _ in 0..iters {
            match m {
                "vbd" => step.vbd_iteration(&mut x),
                "jgs2-rest" | "jgs2-rest-ls" => {
                    let rot = node_rotations(&step, &x);
                    let basis = |i: usize, _x: &[DVec3]| {
                        // Co-rotated: R_v Ū_{v,i} R_iᵀ.
                        let ub = rest_basis[i].as_ref().unwrap();
                        let ri = rot[i].transpose();
                        let mut out = vec![DVec3::ZERO; 3 * n];
                        for v in 0..n {
                            let blk = rot[v]
                                * DMat3::from_cols(ub[3 * v], ub[3 * v + 1], ub[3 * v + 2])
                                * ri;
                            out[3 * v] = blk.col(0);
                            out[3 * v + 1] = blk.col(1);
                            out[3 * v + 2] = blk.col(2);
                        }
                        out
                    };
                    if m == "jgs2-rest" {
                        step.jgs2_iteration(&mut x, &basis);
                    } else {
                        step.jgs2_iteration_ls(&mut x, &basis);
                    }
                }
                _ => {
                    let mut hk = step.hessian(&x);
                    assert!(hk.cholesky());
                    step.jgs2_iteration(&mut x, &|i, _x| full_basis(&step, &hk, i));
                }
            }
            curve.push(err(&x));
            if m == "jgs2-exact" && curve.len() >= 6 {
                break;
            }
        }
        curves.push(curve);
    }
    println!("| iteration | {} |", methods.join(" | "));
    println!("|---|{}", "---|".repeat(methods.len()));
    for k in 0..iters {
        let cell = |c: &Vec<f64>| c.get(k).map_or("".to_string(), |e| format!("{e:.2e}"));
        let row: Vec<String> = curves.iter().map(cell).collect();
        if k < 10 || (k + 1) % 5 == 0 {
            println!("| {} | {} |", k + 1, row.join(" | "));
        }
    }
}

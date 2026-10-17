//! The true stable explicit step of a cluster (the colleague's speed-of-light analysis,
//! section 4): the largest eigenvalue of M^-1 K by power iteration (supports treated as
//! free, which only raises it; intact stiffness with rebar; deformation inertia), the
//! damping ratio of that mode (strain-energy-weighted stiffness-proportional dashpots),
//! and the damped central-difference limit `s (2 / w) (sqrt(1 + z^2) - z)`.
//!
//! Damage only removes stiffness, so the bound of an intact cluster holds for it and for
//! every fragment it breaks into (refinement excepted).

use stress_ref::bond::Local6;
use stress_ref::math::Vec3;
use stress_ref::solver::ReferenceSolver;

/// Highest natural frequency (rad/s) and the damping ratio of that mode.
#[derive(Clone, Copy, Debug)]
pub struct TopMode {
    pub omega: f64,
    pub zeta: f64,
}

fn stiffness6(b: &stress_ref::solver::RtBond) -> Local6 {
    let mut k = b.stiffness.as_local();
    if let Some(r) = &b.rebar {
        k.lin.z += r.k_axial;
        k.lin.x += r.k_dowel;
        k.lin.y += r.k_dowel;
    }
    k
}

/// Power iteration on M^-1 K of cluster `ci` (`iters` steps; the estimate converges
/// from below, so callers add a margin).
pub fn top_mode(m: &ReferenceSolver, ci: usize, iters: usize) -> Option<TopMode> {
    top_mode_detail(m, ci, iters, 1.0).map(|(mode, _)| mode)
}

/// `top_mode` with every chunk's rotational inertia multiplied by `rot_scale` (1 is
/// `top_mode` exactly: the factor multiplies by one), also returning the mode shape
/// (per chunk: translation, rotation; mass-normalised; zero outside the cluster).
pub fn top_mode_detail(m: &ReferenceSolver, ci: usize, iters: usize, rot_scale: f64) -> Option<(TopMode, Vec<(Vec3, Vec3)>)> {
    let cl = &m.clusters[ci];
    if cl.bonds.is_empty() {
        return None;
    }
    let s = cl.structure;
    let st = &m.structures[s];
    let n = st.chunks.len();
    let mut x: Vec<(Vec3, Vec3)> = vec![(Vec3::ZERO, Vec3::ZERO); n];
    for &c in &cl.chunks {
        let h = (c as f64 * 0.618_033_988_75 + 0.1).fract();
        x[c] = (
            Vec3::new(h - 0.5, (h * 7.0).fract() - 0.5, (h * 13.0).fract() - 0.5),
            Vec3::new((h * 3.0).fract() - 0.5, (h * 5.0).fract() - 0.5, (h * 11.0).fract() - 0.5),
        );
    }
    let mut lambda = 0.0;
    let apply = |x: &[(Vec3, Vec3)]| -> Vec<(Vec3, Vec3)> {
        let mut y = vec![(Vec3::ZERO, Vec3::ZERO); n];
        for &bi in &cl.bonds {
            let b = &m.bonds[s][bi];
            let g = &b.geometry;
            let d = g.kinematics(x[g.a].0, x[g.a].1, x[g.b].0, x[g.b].1);
            let (fa, ma, fb, mb) = g.chunk_loads(&d.mul_elem(&stiffness6(b)));
            y[g.a].0 -= fa;
            y[g.a].1 -= ma;
            y[g.b].0 -= fb;
            y[g.b].1 -= mb;
        }
        y
    };
    // Mass-normalised form M^-1/2 K M^-1/2 (symmetric): iterate on z = M^1/2 x.
    let sqrt_m = |c: usize| (st.chunks[c].mass * m.chunks[s][c].inertia_scale).sqrt();
    let i_scale = |c: usize| m.chunks[s][c].inertia_scale;
    for _ in 0..iters {
        let y = apply(&x);
        let (mut num, mut den) = (0.0, 0.0);
        let mut next = vec![(Vec3::ZERO, Vec3::ZERO); n];
        for &c in &cl.chunks {
            let ch = &st.chunks[c];
            // M^-1 K x.
            let v = y[c].0 * (1.0 / (ch.mass * i_scale(c)));
            let w = (ch.inv_inertia * y[c].1) * (1.0 / (i_scale(c) * rot_scale));
            next[c] = (v, w);
            // Rayleigh quotient x^T K x / x^T M x.
            num += x[c].0.dot(y[c].0) + x[c].1.dot(y[c].1);
            den += x[c].0.norm2() * ch.mass * i_scale(c) + x[c].1.dot(ch.inertia * x[c].1) * (i_scale(c) * rot_scale);
        }
        lambda = if den > 0.0 { num / den } else { 0.0 };
        // Normalise in the mass norm.
        let norm: f64 = cl
            .chunks
            .iter()
            .map(|&c| {
                let ch = &st.chunks[c];
                next[c].0.norm2() * ch.mass * i_scale(c) + next[c].1.dot(ch.inertia * next[c].1) * (i_scale(c) * rot_scale)
            })
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            break;
        }
        for &c in &cl.chunks {
            x[c] = (next[c].0 * (1.0 / norm), next[c].1 * (1.0 / norm));
        }
        let _ = sqrt_m;
    }
    let omega = lambda.max(0.0).sqrt();
    // Damping ratio of the top mode: (w / 2) sum beta_b e_b / sum e_b, beta = c / k.
    let (mut num, mut den) = (0.0, 0.0);
    for &bi in &cl.bonds {
        let b = &m.bonds[s][bi];
        let g = &b.geometry;
        let d = g.kinematics(x[g.a].0, x[g.a].1, x[g.b].0, x[g.b].1);
        let k = stiffness6(b);
        let e = d.dot(&d.mul_elem(&k));
        let c = d.dot(&d.mul_elem(&b.damping));
        num += c;
        den += e;
    }
    let zeta = if den > 0.0 { 0.5 * omega * num / den } else { 0.0 };
    Some((TopMode { omega, zeta }, x))
}

/// The damped central-difference step for a mode, with safety `s`.
pub fn stable_step(mode: TopMode, s: f64) -> f64 {
    let (w, z) = (mode.omega, mode.zeta);
    if w <= 0.0 {
        return f64::INFINITY;
    }
    s * 2.0 / w * ((1.0 + z * z).sqrt() - z)
}

/// The true step over all clusters with bonds (power iteration estimates converge from
/// below: the frequency is raised by 3%).
pub fn true_stable_dt(m: &ReferenceSolver, iters: usize, safety: f64) -> f64 {
    let mut dt = f64::INFINITY;
    for ci in 0..m.clusters.len() {
        if let Some(mut mode) = top_mode(m, ci, iters) {
            mode.omega *= 1.03;
            dt = dt.min(stable_step(mode, safety));
        }
    }
    dt
}

/// The scene with the true stable step: its Courant safety scaled so the reference's
/// stress step equals the power-iteration bound at `safety` (contacts scale with it).
/// Both solvers then run the same substep, so the GPU and the reference stay comparable
/// like for like; the step change itself is measured by `stress-gpu-stepcheck`.
/// Returns the scene and the step ratio (true over the reference's own).
pub fn with_true_step(scene: &stress_ref::scene::Scene, safety: f64) -> (stress_ref::scene::Scene, f64) {
    let m = ReferenceSolver::new(scene);
    let reference = m.stable_dt();
    let truth = true_stable_dt(&m, 1000, safety);
    let mut s = scene.clone();
    if !(truth.is_finite() && reference.is_finite() && reference > 0.0) {
        return (s, 1.0);
    }
    let ratio = (truth / reference).max(1.0);
    s.sim.courant_safety *= ratio;
    (s, ratio)
}

/// Certified upper bound of the spectral radius of `M^-1 (K + (2/h) C)` over structure
/// `s`'s live bonds (`h` infinite: `M^-1 K`), with each bond at full stiffness (rebar
/// included) and its dashpots at their full rate (`secant_factors` only lower them):
/// the oracle's `structure_frequency_bound` (Collatz-Wielandt on `|L^-1 X L^-T|`, chunk
/// blocks, supports free) with the dashpots on the same kinematic rows as the springs.
pub fn certified_radius(m: &ReferenceSolver, s: usize, h: f64) -> Option<f64> {
    const ITERATIONS: usize = 2000;
    const CONVERGED: f64 = 1e-4;
    type Block = [[f64; 6]; 6];
    let st = &m.structures[s];
    let n = st.chunks.len();
    let bonds: Vec<usize> = (0..m.bonds[s].len()).filter(|&b| m.bonds[s][b].alive).collect();
    if bonds.is_empty() {
        return None;
    }
    let linv: Vec<Block> = (0..n)
        .map(|c| {
            let ch = &st.chunks[c];
            let mu = m.chunks[s][c].inertia_scale;
            let mut out = [[0.0f64; 6]; 6];
            for t in 0..3 {
                out[t][t] = 1.0 / (ch.mass * mu).sqrt();
            }
            let a: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| ch.inertia.m[i][j] * mu));
            let mut l = [[0.0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..=i {
                    let sum: f64 = (0..j).map(|p| l[i][p] * l[j][p]).sum();
                    l[i][j] = if i == j { (a[i][i] - sum).max(0.0).sqrt() } else { (a[i][j] - sum) / l[j][j] };
                }
            }
            for i in 0..3 {
                out[3 + i][3 + i] = 1.0 / l[i][i];
                for j in 0..i {
                    let sum: f64 = (j..i).map(|p| l[i][p] * out[3 + p][3 + j]).sum();
                    out[3 + i][3 + j] = -sum / l[i][i];
                }
            }
            out
        })
        .collect();
    let congruent = |la: &Block, x: &Block, lb: &Block| -> Block {
        let t: Block = std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|p| la[i][p] * x[p][j]).sum()));
        std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|p| t[i][p] * lb[j][p]).sum()))
    };
    let mut diag = vec![[[0.0f64; 6]; 6]; n];
    let mut off: Vec<((usize, usize), Block)> = Vec::with_capacity(bonds.len());
    for &bi in &bonds {
        let b = &m.bonds[s][bi];
        let g = &b.geometry;
        let (k, c) = (stiffness6(b), b.damping);
        let rate = if h.is_finite() { 2.0 / h } else { 0.0 };
        let ks = [k.lin.x + rate * c.lin.x, k.lin.y + rate * c.lin.y, k.lin.z + rate * c.lin.z, k.ang.x + rate * c.ang.x, k.ang.y + rate * c.ang.y, k.ang.z + rate * c.ang.z];
        let axes = [g.t1, g.t2, g.normal];
        let rows: [[f64; 12]; 6] = std::array::from_fn(|comp| {
            let t = axes[comp % 3];
            let r: [Vec3; 4] = if comp < 3 { [-t, -(g.ra.cross(t)), t, g.rb.cross(t)] } else { [Vec3::ZERO, -t, Vec3::ZERO, t] };
            std::array::from_fn(|x| r[x / 3].to_array()[x % 3])
        });
        let kb: [[f64; 12]; 12] = std::array::from_fn(|x| std::array::from_fn(|y| (0..6).map(|c| ks[c] * rows[c][x] * rows[c][y]).sum()));
        let blk = |r0: usize, c0: usize| -> Block { std::array::from_fn(|i| std::array::from_fn(|j| kb[r0 + i][c0 + j])) };
        let (a, bb) = (g.a, g.b);
        let aa = congruent(&linv[a], &blk(0, 0), &linv[a]);
        let bbk = congruent(&linv[bb], &blk(6, 6), &linv[bb]);
        for i in 0..6 {
            for j in 0..6 {
                diag[a][i][j] += aa[i][j];
                diag[bb][i][j] += bbk[i][j];
            }
        }
        let (lo, hi, x) = if a < bb { (a, bb, blk(0, 6)) } else { (bb, a, blk(6, 0)) };
        off.push(((lo, hi), congruent(&linv[lo], &x, &linv[hi])));
    }
    off.sort_by_key(|e| e.0);
    let mut merged: Vec<((usize, usize), Block)> = Vec::with_capacity(off.len());
    for (key, x) in off {
        match merged.last_mut() {
            Some((k, acc)) if *k == key => {
                for i in 0..6 {
                    for j in 0..6 {
                        acc[i][j] += x[i][j];
                    }
                }
            }
            _ => merged.push((key, x)),
        }
    }
    let abs = |x: &Block| -> Block { x.map(|r| r.map(f64::abs)) };
    let diag: Vec<Block> = diag.iter().map(abs).collect();
    let merged: Vec<((usize, usize), Block)> = merged.iter().map(|(k, x)| (*k, abs(x))).collect();
    let apply = |v: &[[f64; 6]]| -> Vec<[f64; 6]> {
        let mut y = vec![[0.0f64; 6]; n];
        for c in 0..n {
            for i in 0..6 {
                y[c][i] = (0..6).map(|j| diag[c][i][j] * v[c][j]).sum();
            }
        }
        for &((a, b), x) in &merged {
            for i in 0..6 {
                for j in 0..6 {
                    y[a][i] += x[i][j] * v[b][j];
                    y[b][j] += x[i][j] * v[a][i];
                }
            }
        }
        y
    };
    let mut v = vec![[1.0f64; 6]; n];
    let mut bound = f64::INFINITY;
    for _ in 0..ITERATIONS {
        let y = apply(&v);
        let (mut hi, mut lo, mut top) = (0.0f64, f64::INFINITY, 0.0f64);
        for c in 0..n {
            for d in 0..6 {
                if y[c][d] > 0.0 {
                    let ratio = if v[c][d] > 0.0 { y[c][d] / v[c][d] } else { f64::INFINITY };
                    hi = hi.max(ratio);
                    lo = lo.min(ratio);
                }
                top = top.max(y[c][d]);
            }
        }
        bound = bound.min(hi);
        if !(top > 0.0) || hi - lo <= CONVERGED * hi {
            break;
        }
        v = y.iter().map(|r| r.map(|x| x / top)).collect();
    }
    (bound > 0.0 && bound.is_finite()).then_some(bound)
}

/// `certified_damped_step` of every structure, the structures spread over the cores
/// (each is independent; results in structure order, so identical to a serial loop).
pub fn certified_damped_steps(m: &ReferenceSolver) -> Vec<f64> {
    let n = m.structures.len();
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(n.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out = vec![0.0; n];
    let parts: Vec<Vec<(usize, f64)>> = std::thread::scope(|sc| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut got = Vec::new();
                    loop {
                        let s = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if s >= n {
                            break got;
                        }
                        got.push((s, certified_damped_step(m, s)));
                    }
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().expect("certified step worker")).collect()
    });
    for (s, h) in parts.into_iter().flatten() {
        out[s] = h;
    }
    out
}

/// The certified stable central-difference substep of structure `s` (before the safety
/// factor) from the dashpots each bond actually has. Leapfrog with dashpots on the lagged
/// velocity is stable while `(h^2 / 4) K + (h / 2) C < M`, i.e. while
/// `h <= 2 / sqrt(rho(M^-1 (K + (2/h) C)))` (one mode: `(2/w)(sqrt(1 + z^2) - z)`).
/// The oracle's `structure_true_step` bounds `C <= beta K` with `beta` the largest
/// dashpot-to-stiffness ratio of any bond, which pairs the most damped bond with the
/// stiffest mode. Here `h <- 2 / sqrt(rho(h))` from the oracle's certified step: each
/// iterate is checked against its own certified radius, so every returned step is
/// certified; the iteration rises monotonically to the fixed point.
pub fn certified_damped_step(m: &ReferenceSolver, s: usize) -> f64 {
    let start = m.structure_true_step(s);
    if !start.is_finite() {
        return start;
    }
    let mut h = start;
    let Some(mut rho) = certified_radius(m, s, h) else { return h };
    for _ in 0..32 {
        let next = 2.0 / rho.sqrt();
        if next <= h * (1.0 + 1e-3) {
            break;
        }
        // `next` is certified only if it holds against its own (smaller) radius.
        match certified_radius(m, s, next) {
            Some(r) if next * next * r <= 4.0 => (h, rho) = (next, r),
            _ => break,
        }
    }
    h
}

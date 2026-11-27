//! Barrier contact for one step (`DAT_BARRIER=1`, a measurement, off): IPC's incremental potential
//! (Li et al. 2020) on the step's contact set, in place of velocity contact's rows, rounds and
//! zones. Over the nodes of the CCD's pairs it minimises
//!
//!   E(x) = Σᵢ ½·mᵢ·|xᵢ − x̂ᵢ|²/h² + κ·Σₚ b(dₚ),   b(d) = −(d − d̂)²·ln(d/d̂) for 0 < d < d̂,
//!
//! where x̂ is the explicit step's end (its prediction), dₚ the pair's distance past its gap, and
//! d̂ the barrier's reach. The method is preconditioned gradient steps (the diagonal of E's
//! Hessian, the barrier's taken at fixed closest-point weights). Each step is limited by CCD so
//! no pair reaches its gap along it, and backtracked until E falls. Starting from the step's start
//! (crossing-free), every iterate is crossing-free, and E only falls, so it ends. Velocities follow
//! the positions, v = (x − x₀)/h; nodes outside the contact set keep their explicit step.

use super::super::ccd::{accd, closest, CcdPair, CcdStats};
use glam::Vec3;
use std::collections::BTreeMap;

/// The barrier's reach past each pair's gap (m; `DAT_DHAT`).
const DHAT: f32 = 1e-3;
/// Gauss–Newton steps at most (`DAT_BARRIER_ITERS`).
const ITERS: u32 = 50;

fn env(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn b(d: f64, dh: f64) -> f64 {
    if d >= dh { 0.0 } else { -(d - dh) * (d - dh) * (d / dh).ln() }
}
fn db(d: f64, dh: f64) -> f64 {
    if d >= dh { 0.0 } else { -2.0 * (d - dh) * (d / dh).ln() - (d - dh) * (d - dh) / d }
}
fn ddb(d: f64, dh: f64) -> f64 {
    if d >= dh { 0.0 } else { -2.0 * (d / dh).ln() - 4.0 * (d - dh) / d + (d - dh) * (d - dh) / (d * d) }
}

/// The step's barrier contact: `x1`, `vel` moved for the contact set's nodes. Returns the number
/// of gradient steps taken; `st.barrier_unconverged` counts steps that reached [`ITERS`].
pub(in super::super) fn barrier(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], h: f32, st: &mut CcdStats) {
    let (dh, iters) = (env("DAT_DHAT", DHAT) as f64, env("DAT_BARRIER_ITERS", ITERS as f32) as u32);
    // The contact set: every movable node of a pair, in a fixed order.
    let mut idx: BTreeMap<(usize, u32), usize> = BTreeMap::new();
    for p in pairs {
        for k in p.k {
            if im[k.0][k.1 as usize] > 0.0 {
                let n = idx.len();
                idx.entry(k).or_insert(n);
            }
        }
    }
    if idx.is_empty() {
        return;
    }
    let keys: Vec<(usize, u32)> = idx.keys().copied().collect();
    for (i, k) in keys.iter().enumerate() {
        idx.insert(*k, i);
    }
    let m: Vec<f64> = keys.iter().map(|k| 1.0 / im[k.0][k.1 as usize] as f64).collect();
    let xh: Vec<Vec3> = keys.iter().map(|k| x1[k.0][k.1 as usize]).collect();
    let mut x: Vec<Vec3> = keys.iter().map(|k| x0[k.0][k.1 as usize]).collect();
    let kappa = env("DAT_KAPPA", 1.0) as f64 * (m.iter().sum::<f64>() / m.len() as f64) / (h as f64 * h as f64);
    let h2 = (h as f64) * (h as f64);
    // Each pair's gap, held under half its start distance (a pair that began inside its gap).
    let at = |x: &[Vec3], p: &CcdPair| p.k.map(|k| idx.get(&k).map_or(x0[k.0][k.1 as usize], |&i| x[i]));
    let xi: Vec<f32> = pairs.iter().map(|p| p.xi.min(0.5 * closest(p.nn, at(&x, p)).1.length())).collect();
    let energy = |x: &[Vec3]| -> f64 {
        let mut e: f64 = (0..x.len()).map(|i| 0.5 * m[i] * (x[i] - xh[i]).length_squared() as f64 / h2).sum();
        for (p, &g) in pairs.iter().zip(&xi) {
            let d = closest(p.nn, at(x, p)).1.length() as f64 - g as f64;
            if d <= 0.0 {
                return f64::INFINITY;
            }
            e += kappa * b(d, dh);
        }
        e
    };
    let mut e = energy(&x);
    let mut used = 0u32;
    for _ in 0..iters {
        used += 1;
        let mut g: Vec<glam::DVec3> = (0..x.len()).map(|i| (x[i] - xh[i]).as_dvec3() * (m[i] / h2)).collect();
        let mut diag: Vec<f64> = m.iter().map(|&m| m / h2).collect();
        for (p, &gap) in pairs.iter().zip(&xi) {
            let (w, gv) = closest(p.nn, at(&x, p));
            let dist = gv.length() as f64;
            let d = dist - gap as f64;
            if d >= dh || dist <= 0.0 {
                continue;
            }
            let n = (gv.as_dvec3()) / dist;
            let (d1, d2) = (kappa * db(d, dh), (kappa * ddb(d, dh)).max(0.0));
            for j in 0..4 {
                if let Some(&i) = idx.get(&p.k[j]) {
                    g[i] += n * (d1 * w[j] as f64);
                    diag[i] += d2 * (w[j] as f64) * (w[j] as f64);
                }
            }
        }
        // Gauss–Newton: H·dx = −g, H = M/h² + Σₚ κ·max(b″, 0)·JₚᵀJₚ (Jₚ = the pair's weights
        // along its normal), by conjugate gradients, matrix-free, preconditioned by H's diagonal.
        let act: Vec<(usize, [f32; 4], glam::DVec3, f64)> = pairs.iter().zip(&xi).enumerate().filter_map(|(q, (p, &gap))| {
            let (w, gv) = closest(p.nn, at(&x, p));
            let dist = gv.length() as f64;
            let d = dist - gap as f64;
            (d < dh && dist > 0.0).then(|| (q, w, gv.as_dvec3() / dist, (kappa * ddb(d, dh)).max(0.0)))
        }).collect();
        let hv = |v: &[glam::DVec3]| -> Vec<glam::DVec3> {
            let mut o: Vec<glam::DVec3> = (0..v.len()).map(|i| v[i] * (m[i] / h2)).collect();
            for &(q, w, n, k) in &act {
                let kk = pairs[q].k;
                let s: f64 = (0..4).filter_map(|j| idx.get(&kk[j]).map(|&i| w[j] as f64 * n.dot(v[i]))).sum();
                for j in 0..4 {
                    if let Some(&i) = idx.get(&kk[j]) {
                        o[i] += n * (k * w[j] as f64 * s);
                    }
                }
            }
            o
        };
        let dot = |a: &[glam::DVec3], b: &[glam::DVec3]| -> f64 { a.iter().zip(b).map(|(a, b)| a.dot(*b)).sum() };
        let mut dx = vec![glam::DVec3::ZERO; x.len()];
        let mut r: Vec<glam::DVec3> = g.iter().map(|g| -*g).collect();
        let mut z: Vec<glam::DVec3> = r.iter().zip(&diag).map(|(r, &d)| *r / d).collect();
        let mut pp = z.clone();
        let mut rz = dot(&r, &z);
        let r0 = rz.abs().max(1e-300);
        for _ in 0..env("DAT_BARRIER_CG", 30.0) as u32 {
            let hp = hv(&pp);
            let a = rz / dot(&pp, &hp).max(1e-300);
            for i in 0..dx.len() {
                dx[i] += pp[i] * a;
                r[i] -= hp[i] * a;
            }
            z = r.iter().zip(&diag).map(|(r, &d)| *r / d).collect();
            let rz1 = dot(&r, &z);
            if rz1.abs() < 1e-12 * r0 {
                break;
            }
            let beta = rz1 / rz;
            rz = rz1;
            for i in 0..pp.len() {
                pp[i] = z[i] + pp[i] * beta;
            }
        }
        let dir: Vec<Vec3> = dx.iter().map(|d| d.as_vec3()).collect();
        let big = dir.iter().fold(0.0f32, |a, d| a.max(d.length()));
        if big < env("DAT_BARRIER_TOL", 1e-8) {
            break;
        }
        // The longest step along `dir` before any pair reaches its gap, with room.
        let mut a = 1.0f32;
        for (p, &gap) in pairs.iter().zip(&xi) {
            let dp = p.k.map(|k| idx.get(&k).map_or(Vec3::ZERO, |&i| dir[i]));
            if let Some(t) = accd(p.nn, at(&x, p), dp, gap) {
                a = a.min(0.8 * t);
            }
        }
        let slope: f64 = g.iter().zip(&dir).map(|(g, d)| g.dot(d.as_dvec3())).sum();
        let mut moved = false;
        for _ in 0..30 {
            let xn: Vec<Vec3> = x.iter().zip(&dir).map(|(x, d)| *x + *d * a).collect();
            let en = energy(&xn);
            if en <= e + 1e-4 * a as f64 * slope {
                x = xn;
                e = en;
                moved = true;
                break;
            }
            a *= 0.5;
        }
        if !moved {
            break;
        }
    }
    if used == iters {
        st.barrier_unconverged += 1;
    }
    st.barrier_iters += used as u64;
    for (i, k) in keys.iter().enumerate() {
        x1[k.0][k.1 as usize] = x[i];
        vel[k.0][k.1 as usize] = (x[i] - x0[k.0][k.1 as usize]) / h;
    }
}

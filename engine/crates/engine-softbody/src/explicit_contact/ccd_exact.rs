//! The exact solve of a cluster of velocity-contact rows (`ccd_vel`) that APGD left unsettled.
//!
//! A cluster is the set of rows joined through the nodes they share. When the iterations end
//! with a row of a cluster unsettled, the whole cluster is solved exactly. That is the LCP
//! λ ≥ 0 ⟂ Aλ + c ≥ 0, with A = J M⁻¹ Jᵀ dense in f64 and the other rows' impulses held. Every
//! row that APGD failed on in a head-on crush (2026-11-27, head 100) already kept its gap. What
//! failed was a small over-push past the bound, about 1e-3 J a row, on rows that are nearly
//! parallel and share nodes. A is near singular there, and the first-order iterations crawl.
//! Before this, such a step fell to rigid impact zones, which took about 114 kJ of head 100's
//! 1,117 kJ.
//!
//! The method is Lawson–Hanson's active set (the NNLS algorithm, here on a convex QP): the free
//! set's equations solved by Cholesky; a step back to the boundary when a free row's impulse
//! would turn negative; the most violated bound row freed next. It starts from APGD's free set.
//! A near-dependent row whose pivot vanishes is not freed (its constraint is implied by the
//! free rows). The answer is exact to f64 rounding; nothing is regularised.

use super::{Ctx, VRow};
use glam::Vec3;
use std::collections::HashMap;

/// The largest cluster solved exactly (rows; a dense p×p matrix and its Cholesky factor). A
/// larger one is left to the zones and counted (`CcdStats::exact_big`).
pub(super) const EXACT_MAX_ROWS: usize = 2048;

/// A free row whose Cholesky pivot is below this share of its diagonal is dependent on the free
/// rows before it, and is held bound. (At 1e-12, nearly parallel rows stayed free with impulses of
/// ±10 N·s against one another where 0.4 settles; their rounding and lever arms then moved the
/// cars, 2026-11-27.)
pub(super) const EXACT_PIVOT: f64 = 1e-6;

impl Ctx<'_> {
    /// Each row's cluster id: rows joined through a shared moving node.
    pub(super) fn clusters(&self, rows: &[VRow]) -> Vec<usize> {
        let mut parent: Vec<usize> = (0..rows.len()).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let mut at: HashMap<(usize, u32), usize> = HashMap::new();
        for (i, r) in rows.iter().enumerate() {
            for j in 0..4 {
                if self.mi(r.k[j]) > 0.0 && r.w[j] != 0.0 {
                    if let Some(&o) = at.get(&r.k[j]) {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, o));
                        parent[a] = b;
                    } else {
                        at.insert(r.k[j], i);
                    }
                }
            }
        }
        (0..rows.len()).map(|i| find(&mut parent, i)).collect()
    }

    /// Solves every cluster that holds an unsettled row exactly, in place in `l`. Returns false
    /// when a cluster was past [`EXACT_MAX_ROWS`] (left as it was).
    pub(super) fn exact(&self, rows: &[VRow], l: &mut [f32], vel: &[Vec<Vec3>], st: &mut super::CcdStats) -> bool {
        let comp = self.clusters(rows);
        let prec = self.precision(rows, l, vel);
        let bad: Vec<usize> = (0..rows.len()).filter(|&i| !self.row_settled(&rows[i], l[i], vel, &prec)).map(|i| comp[i]).collect();
        let mut done: Vec<usize> = Vec::new();
        let mut all = true;
        for c in bad {
            if done.contains(&c) {
                continue;
            }
            done.push(c);
            let idx: Vec<usize> = (0..rows.len()).filter(|&i| comp[i] == c).collect();
            if idx.len() > EXACT_MAX_ROWS {
                st.exact_big += 1;
                all = false;
                continue;
            }
            st.exact += 1;
            st.exact_rows += idx.len() as u64;
            st.exact_pivots += self.exact_cluster(rows, &idx, l, vel) as u64;
        }
        all
    }

    /// One cluster's LCP by Lawson–Hanson; returns the pivots it took.
    fn exact_cluster(&self, rows: &[VRow], idx: &[usize], l: &mut [f32], vel: &[Vec<Vec3>]) -> usize {
        let p = idx.len();
        // A = J M⁻¹ Jᵀ over the cluster, from each node's rows.
        let mut at: std::collections::BTreeMap<(usize, u32), Vec<(usize, glam::DVec3)>> = Default::default();
        for (a, &i) in idx.iter().enumerate() {
            let r = &rows[i];
            for j in 0..4 {
                let m = self.mi(r.k[j]);
                if m > 0.0 && r.w[j] != 0.0 {
                    at.entry(r.k[j]).or_default().push((a, (r.n * r.w[j]).as_dvec3() * (m as f64).sqrt()));
                }
            }
        }
        let mut am = vec![0.0f64; p * p];
        for list in at.values() {
            for &(a, u) in list {
                for &(b, v) in list {
                    am[a * p + b] += u.dot(v);
                }
            }
        }
        // c = g − Aλ: the gaps with the cluster's own impulses taken out (vel holds them).
        let mut lam: Vec<f64> = idx.iter().map(|&i| l[i] as f64).collect();
        let g: Vec<f64> = idx.iter().map(|&i| {
            let r = &rows[i];
            (0..4).map(|j| r.w[j] as f64 * r.n.as_dvec3().dot(vel[r.k[j].0][r.k[j].1 as usize].as_dvec3())).sum::<f64>() + r.b as f64
        }).collect();
        let c: Vec<f64> = (0..p).map(|a| g[a] - (0..p).map(|b| am[a * p + b] * lam[b]).sum::<f64>()).collect();
        let w = |lam: &[f64]| -> Vec<f64> { (0..p).map(|a| c[a] + (0..p).map(|b| am[a * p + b] * lam[b]).sum::<f64>()).collect() };
        // A row's violation worth freeing: closing past its bound by more than f64 rounding of
        // the speeds involved.
        let scale = g.iter().fold(1.0f64, |s, x| s.max(x.abs())) + c.iter().fold(0.0f64, |s, x| s.max(x.abs()));
        let tol = 1e-12 * scale;
        let mut free: Vec<bool> = lam.iter().map(|&x| x > 0.0).collect();
        let mut stuck: Vec<bool> = vec![false; p];
        let mut pivots = 0usize;
        for _ in 0..4 * p + 8 {
            // Inner loop: the free set's equations, stepping back while an impulse turns negative.
            for _ in 0..p + 1 {
                let f: Vec<usize> = (0..p).filter(|&a| free[a]).collect();
                let z = match chol_solve(&am, p, &f, &c) {
                    Ok(z) => z,
                    Err(k) => {
                        // Row f[k] is dependent on the free rows before it: hold it bound.
                        free[f[k]] = false;
                        stuck[f[k]] = true;
                        lam[f[k]] = 0.0;
                        continue;
                    }
                };
                if z.iter().all(|&x| x > 0.0) {
                    for (k, &a) in f.iter().enumerate() {
                        lam[a] = z[k];
                    }
                    break;
                }
                let mut alpha = 1.0f64;
                for (k, &a) in f.iter().enumerate() {
                    if z[k] <= 0.0 {
                        alpha = alpha.min(lam[a] / (lam[a] - z[k]));
                    }
                }
                let top = lam.iter().fold(0.0f64, |m, &x| m.max(x));
                for (k, &a) in f.iter().enumerate() {
                    lam[a] += alpha * (z[k] - lam[a]);
                    if lam[a] <= 1e-14 * top {
                        lam[a] = 0.0;
                        free[a] = false;
                    }
                }
            }
            let wv = w(&lam);
            let next = (0..p).filter(|&a| !free[a] && !stuck[a] && wv[a] < -tol).min_by(|&a, &b| wv[a].total_cmp(&wv[b]));
            let Some(j) = next else { break };
            free[j] = true;
            pivots += 1;
        }
        for (k, &i) in idx.iter().enumerate() {
            l[i] = lam[k].max(0.0) as f32;
        }
        pivots
    }
}

/// Solves A_FF z = −c_F by Cholesky (f64). Err(k) when the k-th free row's pivot is not
/// positive past rounding (that row is a combination of the ones before it).
fn chol_solve(am: &[f64], p: usize, f: &[usize], c: &[f64]) -> Result<Vec<f64>, usize> {
    let n = f.len();
    let mut lo = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = am[f[i] * p + f[j]];
            for k in 0..j {
                s -= lo[i * n + k] * lo[j * n + k];
            }
            if i == j {
                if s <= EXACT_PIVOT * am[f[i] * p + f[i]].max(1e-300) {
                    return Err(i);
                }
                lo[i * n + i] = s.sqrt();
            } else {
                lo[i * n + j] = s / lo[j * n + j];
            }
        }
    }
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = -c[f[i]];
        for k in 0..i {
            s -= lo[i * n + k] * y[k];
        }
        y[i] = s / lo[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= lo[k * n + i] * y[k];
        }
        y[i] = s / lo[i * n + i];
    }
    Ok(y)
}

//! Continuous contact's response as rows with accumulated impulses
//! (`docs/continuous-response-parallel-design.md`, design C; CPU reference, `DAT_RESPONSE=rows`):
//! every pair that collides over the step's paths becomes a row at its impact (normal, signed
//! weights, the nodes' start positions) with two accumulated impulses, λx on the step's end
//! positions (the pair's path over the step not closing along its normal) and λv on the
//! velocities (no row closing). A round's rows are solved by projected Jacobi with mass splitting (Tonge 2012):
//! each iteration every row moves its λ to where its own constraint would hold with its nodes
//! shared among their rows, λ ≥ 0, and the changes are summed per node in fixed point (order
//! free, bit for bit on the GPU). Iterations run until no row changes or [`ROW_ITERS`]; then the
//! paths are tested again and new impacts add rows. The velocity rows converge to the
//! mass-weighted projection onto "nothing closing" (Moreau's inelastic solution), so the impact
//! law's e = 1 gives the energy back exactly. What still collides goes to the zones.

use super::ccd::{closest, CcdPair, CcdStats, FixedSums};
use glam::Vec3;

/// The most Jacobi iterations a round (`DAT_ROW_ITERS`; a fixed point ends it sooner).
pub const ROW_ITERS: u32 = 32;

struct Row {
    k: [(usize, u32); 4],
    w: [f32; 4],
    n: Vec3,
    at: [Vec3; 4],
    lx: f32,
    lv: f32,
}

/// `rounds` rounds of detection and the rows' solve; true when nothing collides any more.
#[allow(clippy::too_many_arguments)]
pub(super) fn rows_rounds(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>) -> bool {
    let iters = std::env::var("DAT_ROW_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(ROW_ITERS);
    let pos = |x: &[Vec<Vec3>], k: &[(usize, u32); 4]| k.map(|(v, i)| x[v][i as usize]);
    let mi = |k: (usize, u32)| im[k.0][k.1 as usize];
    let mut rows: Vec<Row> = Vec::new();
    let mut count: Vec<Vec<u32>> = x1.iter().map(|x| vec![0; x.len()]).collect();
    // (The state before the solve: every iteration of every round rebuilds from it with all
    // the rows' impulses.)
    let (xb, vb) = (x1.to_vec(), vel.to_vec());
    for _ in 0..rounds {
        let hit = collide(x1);
        st.rounds += 1;
        if hit.is_empty() {
            return true;
        }
        st.hits += hit.len() as u64;
        // Each impact a row of its own, linearised there (an earlier row of the pair keeps its
        // own impulses along its own normal).
        for &(q, t) in &hit {
            let pr = &pairs[q];
            let (a, b) = (pos(x0, &pr.k), pos(x1, &pr.k));
            let at = [0, 1, 2, 3].map(|j| a[j] + (b[j] - a[j]) * t);
            let (w, gv) = closest(pr.nn, at);
            let n = gv.normalize_or_zero();
            // (Positions: the pair's path over the step not closing along the normal at its
            // impact, Σ wⱼ n·(x1ⱼ − x0ⱼ) ≥ 0, as the rounds' position step; `at` the start.)
            if n != Vec3::ZERO {
                rows.push(Row { k: pr.k, w, n, at: a, lx: 0.0, lv: 0.0 });
            }
        }
        for c in count.iter_mut() {
            c.iter_mut().for_each(|c| *c = 0);
        }
        for r in &rows {
            for j in 0..4 {
                if r.w[j] != 0.0 && mi(r.k[j]) > 0.0 {
                    count[r.k[j].0][r.k[j].1 as usize] += 1;
                }
            }
        }
        let d: Vec<f32> = rows.iter().map(|r| (0..4).map(|j| r.w[j] * r.w[j] * mi(r.k[j]) * count[r.k[j].0][r.k[j].1 as usize] as f32).sum()).collect();
        // Each iteration rebuilds every node's change from all the rows' whole impulses, from
        // the state before the solve, so fixed-point rounding never accumulates over iterations
        // (each row's push is rounded once a node: momentum to that one rounding).
        let mut sums = FixedSums::new(x1);
        for _ in 0..iters {
            let mut changed = false;
            let mut next: Vec<(f32, f32)> = Vec::with_capacity(rows.len());
            for (r, &dr) in rows.iter().zip(&d) {
                if dr <= 0.0 {
                    next.push((r.lx, r.lv));
                    continue;
                }
                let (mut cx, mut cv) = (0.0f32, 0.0f32);
                for j in 0..4 {
                    let (v, i) = (r.k[j].0, r.k[j].1 as usize);
                    cx += r.w[j] * r.n.dot(x1[v][i] - r.at[j]);
                    cv += r.w[j] * r.n.dot(vel[v][i]);
                }
                let l = ((r.lx - cx / dr).max(0.0), (r.lv - cv / dr).max(0.0));
                changed |= l != (r.lx, r.lv);
                next.push(l);
            }
            for (r, l) in rows.iter_mut().zip(next) {
                (r.lx, r.lv) = l;
            }
            for (r, &dr) in rows.iter().zip(&d) {
                if dr <= 0.0 || (r.lx == 0.0 && r.lv == 0.0) {
                    continue;
                }
                for j in 0..4 {
                    let m = mi(r.k[j]);
                    if m > 0.0 && r.w[j] != 0.0 {
                        sums.add_x(r.k[j].0, r.k[j].1, r.n * (r.w[j] * m * r.lx));
                        sums.add_v(r.k[j].0, r.k[j].1, r.n * (r.w[j] * m * r.lv));
                    }
                }
            }
            x1.clone_from_slice(&xb);
            vel.clone_from_slice(&vb);
            sums.apply(vel, Some(x1));
            if !changed {
                break;
            }
        }
    }
    false
}

/// Velocity-only impulse contact (Moreau–Jean time-stepping; `DAT_RESPONSE=vel`; the owner,
/// 2026-11-26: created energy is a bug). Every pair the CCD finds colliding over the step's
/// paths becomes a row on the velocities at its impact normal: Σ wⱼ n·vⱼ ≥ −b with
/// b = max(s₀ − ξ, 0)/h, s₀ the pair's start separation along n, ξ its gap: its closing may
/// bring it to its gap by the step's end and no further. Rows are solved by projected Jacobi with
/// mass splitting and accumulated impulses to a fixed point, each iteration rebuilt from the
/// velocities before the solve (fixed point: momentum to one rounding); the positions then follow
/// the velocity change (x1 += h·Δv), nothing moves a position alone. With every b ≥ 0, v = 0 is
/// feasible and every iteration lowers the kinetic energy: none is created. Repeated `rounds`
/// times over the new paths; true when nothing collides any more.
#[allow(clippy::too_many_arguments)]
pub(super) fn velocity_rounds(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], h: f32, rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>) -> bool {
    struct VRow {
        k: [(usize, u32); 4],
        w: [f32; 4],
        n: Vec3,
        b: f32,
        l: f32,
    }
    let iters = std::env::var("DAT_ROW_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(ROW_ITERS);
    let pos = |x: &[Vec<Vec3>], k: &[(usize, u32); 4]| k.map(|(v, i)| x[v][i as usize]);
    let mi = |k: (usize, u32)| im[k.0][k.1 as usize];
    let (vb, xb) = (vel.to_vec(), x1.to_vec());
    let mut rows: Vec<VRow> = Vec::new();
    let mut count: Vec<Vec<u32>> = x1.iter().map(|x| vec![0; x.len()]).collect();
    for _ in 0..rounds {
        let hit = collide(x1);
        st.rounds += 1;
        if hit.is_empty() {
            return true;
        }
        st.hits += hit.len() as u64;
        for &(q, t) in &hit {
            let pr = &pairs[q];
            let (a, b) = (pos(x0, &pr.k), pos(x1, &pr.k));
            let at = [0, 1, 2, 3].map(|j| a[j] + (b[j] - a[j]) * t);
            let (w, gv) = closest(pr.nn, at);
            let n = gv.normalize_or_zero();
            if n == Vec3::ZERO {
                continue;
            }
            let s0: f32 = (0..4).map(|j| w[j] * n.dot(a[j])).sum();
            rows.push(VRow { k: pr.k, w, n, b: (s0 - pr.xi).max(0.0) / h, l: 0.0 });
        }
        for c in count.iter_mut() {
            c.iter_mut().for_each(|c| *c = 0);
        }
        for r in &rows {
            for j in 0..4 {
                if r.w[j] != 0.0 && mi(r.k[j]) > 0.0 {
                    count[r.k[j].0][r.k[j].1 as usize] += 1;
                }
            }
        }
        let d: Vec<f32> = rows.iter().map(|r| (0..4).map(|j| r.w[j] * r.w[j] * mi(r.k[j]) * count[r.k[j].0][r.k[j].1 as usize] as f32).sum()).collect();
        let mut sums = FixedSums::new(x1);
        for _ in 0..iters {
            let mut changed = false;
            let next: Vec<f32> = rows
                .iter()
                .zip(&d)
                .map(|(r, &dr)| {
                    if dr <= 0.0 {
                        return r.l;
                    }
                    let cv: f32 = (0..4).map(|j| r.w[j] * r.n.dot(vel[r.k[j].0][r.k[j].1 as usize])).sum();
                    (r.l - (cv + r.b) / dr).max(0.0)
                })
                .collect();
            for (r, l) in rows.iter_mut().zip(next) {
                changed |= l != r.l;
                r.l = l;
            }
            for (r, &dr) in rows.iter().zip(&d) {
                if dr <= 0.0 || r.l == 0.0 {
                    continue;
                }
                for j in 0..4 {
                    let m = mi(r.k[j]);
                    if m > 0.0 && r.w[j] != 0.0 {
                        sums.add_v(r.k[j].0, r.k[j].1, r.n * (r.w[j] * m * r.l));
                    }
                }
            }
            vel.clone_from_slice(&vb);
            sums.apply(vel, None);
            if !changed {
                break;
            }
        }
        // Positions follow the velocity change.
        for v in 0..x1.len() {
            for i in 0..x1[v].len() {
                x1[v][i] = xb[v][i] + (vel[v][i] - vb[v][i]) * h;
            }
        }
    }
    false
}

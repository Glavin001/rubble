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
        let mut sums = FixedSums::new(x1);
        for _ in 0..iters {
            let mut changed = false;
            for (r, &dr) in rows.iter_mut().zip(&d) {
                if dr <= 0.0 {
                    continue;
                }
                let (mut cx, mut cv) = (0.0f32, 0.0f32);
                for j in 0..4 {
                    let (v, i) = (r.k[j].0, r.k[j].1 as usize);
                    cx += r.w[j] * r.n.dot(x1[v][i] - r.at[j]);
                    cv += r.w[j] * r.n.dot(vel[v][i]);
                }
                let (lx, lv) = ((r.lx - cx / dr).max(0.0), (r.lv - cv / dr).max(0.0));
                let (dx, dv) = (lx - r.lx, lv - r.lv);
                if dx == 0.0 && dv == 0.0 {
                    continue;
                }
                changed = true;
                (r.lx, r.lv) = (lx, lv);
                for j in 0..4 {
                    let m = mi(r.k[j]);
                    if m > 0.0 && r.w[j] != 0.0 {
                        sums.add_x(r.k[j].0, r.k[j].1, r.n * (r.w[j] * m * dx));
                        sums.add_v(r.k[j].0, r.k[j].1, r.n * (r.w[j] * m * dv));
                    }
                }
            }
            sums.apply(vel, Some(x1));
            if !changed {
                break;
            }
        }
    }
    false
}

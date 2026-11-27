//! Velocity-only impulse contact (`DAT_RESPONSE=vel`; the owner, 2026-11-26: created energy is a
//! bug; Fable's recommendation 2026-11-27, `docs/continuous-contact-design.md`).
//!
//! Every pair the CCD finds colliding over the step's paths becomes a row on the velocities at
//! its impact normal, Σ wⱼ n·vⱼ ≥ −b. Positions follow the velocity change (x1 += h·Δv); nothing
//! moves a position alone. One step:
//!
//! A row's bound is the larger of two: closing at most to its gap by the step's end,
//! b = max(s₀ − ξ, 0)/h, and Newton's impact law on the closing it had before this step's kick:
//! J v⁺ ≥ −e·min(J v⁻, 0), v⁻ the velocity before the kick. In symplectic Euler an impulse at the
//! kick does work Σ m·Δv·(v⁻ + v⁺)/2 on the integrator's conserved energy. Per row that is
//! ½·(u⁺ − u_f)·(1 − e)·u⁻ in the row's mass, so it is 0 at e = 1 and ≤ 0 below. A contact
//! its beams press (u⁻ = 0) gets no bounce. Reflecting the kicked velocity instead (Cirak–West's
//! momentum reflection) bounces a pressed node at hF/m every step and creates m·u² each time:
//! +95 J on the elastic bars at 5 m/s, measured 2026-11-27.
//!
//! The solve is accelerated projected gradient descent (APGD, Mazhar et al. 2015), with Nesterov
//! momentum and a fixed restart every [`APGD_RESTART`] iterations (no global reduction, so it is
//! deterministic). Its preconditioner is the mass-splitting diagonal (Tonge 2012:
//! D = Σ cⱼ wⱼ²/mⱼ, cⱼ the node's row count, D ⪰ A). Every evaluation rebuilds the velocities
//! from v_f and the whole impulses, summed per node in fixed point, so momentum is kept to one
//! rounding and nothing accumulates.

use super::ccd::{closest, CcdPair, CcdStats, FixedSums};
use super::ccd_rows::{ROW_GAP_SHARE, ROW_ITERS};
use glam::Vec3;

/// A pushing row may part faster than its bound by at most this (m/s; `DAT_PUSH_TOL`).
pub(super) const PUSH_TOL: f32 = 1.0e-4;

fn push_tol() -> f32 {
    std::env::var("DAT_PUSH_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(PUSH_TOL)
}

/// APGD's momentum restarts every this many iterations.
pub(super) const APGD_RESTART: u32 = 16;

struct VRow {
    q: usize,
    k: [(usize, u32); 4],
    w: [f32; 4],
    n: Vec3,
    b: f32,
    l: f32,
}

struct Ctx<'a> {
    pairs: &'a [CcdPair],
    x0: &'a [Vec<Vec3>],
    im: &'a [Vec<f32>],
    h: f32,
    iters: u32,
}

impl Ctx<'_> {
    fn mi(&self, k: (usize, u32)) -> f32 {
        self.im[k.0][k.1 as usize]
    }

    /// vel = base + M⁻¹Jᵀ·(s·λ), the impulses summed per node in fixed point.
    fn velocities(&self, rows: &[VRow], l: &[f32], s: f32, base: &[Vec<Vec3>], vel: &mut [Vec<Vec3>]) {
        let mut sums = FixedSums::new(base);
        for (r, &l) in rows.iter().zip(l) {
            if l == 0.0 {
                continue;
            }
            for j in 0..4 {
                let m = self.mi(r.k[j]);
                if m > 0.0 && r.w[j] != 0.0 {
                    sums.add_v(r.k[j].0, r.k[j].1, r.n * (r.w[j] * m * l * s));
                }
            }
        }
        vel.clone_from_slice(base);
        sums.apply(vel, None);
    }

    /// A row's closing speed J v (separating positive).
    fn normal(r: &VRow, vel: &[Vec<Vec3>]) -> f32 {
        (0..4).map(|j| r.w[j] * r.n.dot(vel[r.k[j].0][r.k[j].1 as usize])).sum()
    }

    /// No row closes past its bound by more than [`ROW_GAP_SHARE`] of its gap over the step, and
    /// no row pushing (λ > 0) parts faster than its bound by more than [`PUSH_TOL`]: a push past
    /// the bound does work λ·excess/2 on the conserved energy.
    fn settled(&self, rows: &[VRow], l: &[f32], vel: &[Vec<Vec3>]) -> bool {
        let tol = push_tol();
        rows.iter().zip(l).all(|(r, &l)| {
            let g = Self::normal(r, vel) + r.b;
            -g * self.h <= ROW_GAP_SHARE * self.pairs[r.q].xi && (l == 0.0 || g <= tol)
        })
    }

    /// APGD on the rows from their current λ: λ ≥ 0 ⟂ J v + b ≥ 0, v = base + M⁻¹Jᵀλ. Leaves
    /// `vel` at the final λ; true when settled.
    fn solve(&self, rows: &mut [VRow], base: &[Vec<Vec3>], vel: &mut [Vec<Vec3>], st: &mut CcdStats) -> bool {
        let shape: Vec<Vec<u32>> = base.iter().map(|x| vec![0; x.len()]).collect();
        let mut count = shape;
        for r in rows.iter() {
            for j in 0..4 {
                if r.w[j] != 0.0 && self.mi(r.k[j]) > 0.0 {
                    count[r.k[j].0][r.k[j].1 as usize] += 1;
                }
            }
        }
        let d: Vec<f32> = rows.iter().map(|r| (0..4).map(|j| r.w[j] * r.w[j] * self.mi(r.k[j]) * count[r.k[j].0][r.k[j].1 as usize] as f32).sum()).collect();
        let mut l: Vec<f32> = rows.iter().map(|r| r.l).collect();
        let mut y = l.clone();
        let mut theta = 1.0f32;
        let mut vy = base.to_vec();
        self.velocities(rows, &l, 1.0, base, vel);
        let mut ok = self.settled(rows, &l, vel);
        let mut used = 0u32;
        while !ok && used < self.iters {
            used += 1;
            self.velocities(rows, &y, 1.0, base, &mut vy);
            let next: Vec<f32> = rows.iter().zip(&y).zip(&d).map(|((r, &y), &d)| if d > 0.0 { (y - (Self::normal(r, &vy) + r.b) / d).max(0.0) } else { 0.0 }).collect();
            if used % APGD_RESTART == 0 {
                theta = 1.0;
                y.clone_from(&next);
            } else {
                let t1 = 0.5 * (1.0 + (1.0 + 4.0 * theta * theta).sqrt());
                let beta = (theta - 1.0) / t1;
                theta = t1;
                for ((y, &n), &o) in y.iter_mut().zip(&next).zip(&l) {
                    *y = n + beta * (n - o);
                }
            }
            l = next;
            self.velocities(rows, &l, 1.0, base, vel);
            ok = self.settled(rows, &l, vel);
        }
        st.row_iters += used as u64;
        for (r, l) in rows.iter_mut().zip(l) {
            r.l = l;
        }
        ok
    }

    /// The rounds: rows from the pairs colliding over the paths, solved, positions following, until
    /// nothing new collides. True when resolved.
    #[allow(clippy::too_many_arguments)]
    fn rounds(&self, x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], e: f32) -> bool {
        let pos = |x: &[Vec<Vec3>], k: &[(usize, u32); 4]| k.map(|(v, i)| x[v][i as usize]);
        let (vb, xb) = (vel.to_vec(), x1.to_vec());
        let mut rows: Vec<VRow> = Vec::new();
        let mut listed = std::collections::HashSet::new();
        let mut settled = false;
        for _ in 0..rounds {
            let hit = collide(x1);
            st.rounds += 1;
            if hit.is_empty() {
                return true;
            }
            st.hits += hit.len() as u64;
            let mut fresh = 0usize;
            for &(q, t) in &hit {
                // A pair already a row keeps it (its λ warm-starts the solve).
                if !listed.insert(q) {
                    continue;
                }
                fresh += 1;
                let pr = &self.pairs[q];
                let (a, b) = (pos(self.x0, &pr.k), pos(x1, &pr.k));
                let at = [0, 1, 2, 3].map(|j| a[j] + (b[j] - a[j]) * t);
                let (w, gv) = closest(pr.nn, at);
                let n = gv.normalize_or_zero();
                if n == Vec3::ZERO {
                    continue;
                }
                let s0: f32 = (0..4).map(|j| w[j] * n.dot(a[j])).sum();
                let mut r = VRow { q, k: pr.k, w, n, b: (s0 - pr.xi).max(0.0) / self.h, l: 0.0 };
                if e > 0.0 && !vpre.is_empty() {
                    r.b = r.b.min(e * Self::normal(&r, vpre).min(0.0));
                }
                rows.push(r);
            }
            // Every pair still meeting its gap is a row the last solve settled: resolved.
            if fresh == 0 && settled {
                return true;
            }
            settled = self.solve(&mut rows, &vb, vel, st);
            for v in 0..x1.len() {
                for i in 0..x1[v].len() {
                    x1[v][i] = xb[v][i] + (vel[v][i] - vb[v][i]) * self.h;
                }
            }
        }
        false
    }
}

/// The step's velocity contact (module doc): `x0` the start, `x1` the end and `vel` the kicked
/// velocities (both moved by the impulses), `vpre` the velocities before the kick, `e` the impact
/// law's restitution. True when nothing collides any more.
#[allow(clippy::too_many_arguments)]
pub(super) fn velocity_rounds(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], h: f32, rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], e: f32) -> bool {
    let iters = std::env::var("DAT_ROW_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(ROW_ITERS);
    Ctx { pairs, x0, im, h, iters }.rounds(x1, vel, rounds, st, collide, vpre, e)
}

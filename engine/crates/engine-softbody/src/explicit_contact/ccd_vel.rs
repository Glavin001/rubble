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

use super::ccd::{closest, CcdPair, CcdStats, VEL_FIXED};
use super::ccd_rows::{ROW_GAP_SHARE, ROW_ITERS};
use glam::Vec3;
use std::collections::HashMap;

/// Warm starts (`Dat::warm`): each pair's impulse at the end of the last step, keyed by its four
/// nodes, as the first iterate of the next step's solve (Box2D's and PhysX's accumulated
/// impulses): a contact a crush presses for many steps starts near its answer.
#[derive(Default, Clone, Debug)]
pub struct Warm {
    pub prev: HashMap<[(usize, u32); 4], f32>,
    pub next: HashMap<[(usize, u32); 4], f32>,
}

/// A pushing row's work past its bound, λ·excess (J, about twice the energy it may add; `DAT_PUSH_TOL`),
/// at most this a step: 10 µJ, a car's 2,000 steps a second over its hundred-odd rows at most
/// about 2 J a second of 200 kJ.
pub(super) const PUSH_TOL: f32 = 1.0e-5;

fn push_tol() -> f32 {
    std::env::var("DAT_PUSH_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(PUSH_TOL)
}

/// A pair already a row is linearised again only when it ends closer than this share of what it
/// may close to (`DAT_RELIN`).
fn relin_share() -> f32 {
    std::env::var("DAT_RELIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0 - ROW_GAP_SHARE)
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
    /// The pair's separation along `n` at the step's start.
    s0: f32,
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

    /// vel = base + M⁻¹Jᵀ·(s·λ), the impulses summed per node in fixed point (the kernels'
    /// scale, `VEL_FIXED`) as 64-bit integers: order free and exact, with no range to overflow.
    /// (In 32 bits, ±476 m/s, the accelerated iterates of a 130 km/h head-on's impact rows,
    /// many on one node, wrapped 902 times: 2026-11-27.) A term past 2⁶² units is counted.
    fn velocities(&self, rows: &[VRow], l: &[f32], s: f32, base: &[Vec<Vec3>], vel: &mut [Vec<Vec3>]) {
        let mut sums: Vec<Vec<[i64; 3]>> = base.iter().map(|x| vec![[0i64; 3]; x.len()]).collect();
        for (r, &l) in rows.iter().zip(l) {
            if l == 0.0 {
                continue;
            }
            for j in 0..4 {
                let m = self.mi(r.k[j]);
                if m > 0.0 && r.w[j] != 0.0 {
                    let d = r.n * (r.w[j] * m * l * s);
                    let e = &mut sums[r.k[j].0][r.k[j].1 as usize];
                    for (k, c) in d.to_array().into_iter().enumerate() {
                        let f = (c as f64 * VEL_FIXED as f64).round_ties_even();
                        if !(f.abs() < 4.6e18) {
                            super::ccd::FIXED_OVERFLOWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        e[k] = e[k].wrapping_add(f.clamp(-4.6e18, 4.6e18) as i64);
                    }
                }
            }
        }
        vel.clone_from_slice(base);
        for (v, row) in sums.iter().enumerate() {
            for (i, a) in row.iter().enumerate() {
                if *a != [0; 3] {
                    vel[v][i] += Vec3::new((a[0] as f64 / VEL_FIXED as f64) as f32, (a[1] as f64 / VEL_FIXED as f64) as f32, (a[2] as f64 / VEL_FIXED as f64) as f32);
                }
            }
        }
    }

    /// A row's closing speed J v (separating positive).
    fn normal(r: &VRow, vel: &[Vec<Vec3>]) -> f32 {
        (0..4).map(|j| r.w[j] * r.n.dot(vel[r.k[j].0][r.k[j].1 as usize])).sum()
    }

    /// No row closes past its bound by more than [`ROW_GAP_SHARE`] of its gap over the step, and
    /// no row pushing (λ > 0) does more than [`PUSH_TOL`] of work past its bound: a push past the
    /// bound does work λ·excess/2 on the conserved energy.
    fn settled(&self, rows: &[VRow], l: &[f32], vel: &[Vec<Vec3>]) -> bool {
        let tol = push_tol();
        rows.iter().zip(l).all(|(r, &l)| {
            let g = Self::normal(r, vel) + r.b;
            -g * self.h <= ROW_GAP_SHARE * self.pairs[r.q].xi && l * g <= tol
        })
    }

    /// APGD on the rows from their current λ: λ ≥ 0 ⟂ J v + b ≥ 0, v = base + M⁻¹Jᵀλ. Leaves
    /// `vel` at the final λ; true when settled.
    fn solve(&self, rows: &mut [VRow], base: &[Vec<Vec3>], vel: &mut [Vec<Vec3>], st: &mut CcdStats) -> bool {
        // The preconditioner D ⪰ A (Gershgorin): row i's D is Σⱼ |wᵢⱼ|·sⱼ/mⱼ, sⱼ the sum of the
        // |w| of every row at node j. (Mass splitting's Σⱼ cⱼ·wᵢⱼ²/mⱼ, cⱼ the rows at j, falls
        // under A where barycentric weights are below 1, and the accelerated steps then
        // diverged on a head-on crush's 744 rows: 2026-11-27.)
        let mut sum: Vec<Vec<f32>> = base.iter().map(|x| vec![0.0; x.len()]).collect();
        for r in rows.iter() {
            for j in 0..4 {
                if self.mi(r.k[j]) > 0.0 {
                    sum[r.k[j].0][r.k[j].1 as usize] += r.w[j].abs();
                }
            }
        }
        let d: Vec<f32> = rows.iter().map(|r| (0..4).map(|j| r.w[j].abs() * self.mi(r.k[j]) * sum[r.k[j].0][r.k[j].1 as usize]).sum()).collect();
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
    fn rounds(&self, x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], e: f32, warm: &mut Warm) -> bool {
        self.rounds_inner(x1, vel, rounds, st, collide, vpre, e, warm)
    }

    /// The rows' impulses into the next step's warm starts: pressed contacts only (an impact's
    /// impulse, b < 0, is no guess for the next step: seeded there, a node shared by many such
    /// rows started past 476 m/s and its fixed-point sums overflowed, head-on 100).
    fn keep(rows: &[VRow], warm: &mut Warm, done: bool) -> bool {
        for r in rows.iter().filter(|r| r.l > 0.0 && r.b >= 0.0) {
            warm.next.insert(r.k, r.l);
        }
        done
    }

    #[allow(clippy::too_many_arguments)]
    fn rounds_inner(&self, x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], e: f32, warm: &mut Warm) -> bool {
        let pos = |x: &[Vec<Vec3>], k: &[(usize, u32); 4]| k.map(|(v, i)| x[v][i as usize]);
        let (vb, xb) = (vel.to_vec(), x1.to_vec());
        let mut rows: Vec<VRow> = Vec::new();
        let mut listed: std::collections::HashMap<usize, usize> = Default::default();
        let mut settled = false;
        for _ in 0..rounds {
            let hit = collide(x1);
            st.rounds += 1;
            if hit.is_empty() {
                return Self::keep(&rows, warm, true);
            }
            st.hits += hit.len() as u64;
            let mut fresh = 0usize;
            let mut relin = 0usize;
            for &(q, t) in &hit {
                let pr = &self.pairs[q];
                let (a, b) = (pos(self.x0, &pr.k), pos(x1, &pr.k));
                // A pair already a row that ends at least (1 − ROW_GAP_SHARE) of what it may
                // close to (its gap, or its start distance when it began inside the gap) keeps
                // its row; one closer is linearised again at its impact now (its normal turns
                // with its features), keeping its λ.
                // The end separation is signed, along the row's own normal: a pair that passed
                // through within the step is as far apart unsigned as one that stopped short.
                let old = listed.get(&q).copied();
                if let Some(i) = old {
                    let r = &rows[i];
                    let s_end: f32 = (0..4).map(|j| r.w[j] * r.n.dot(b[j])).sum();
                    if s_end > 0.0 && s_end >= relin_share() * r.s0.min(pr.xi) {
                        continue;
                    }
                }
                let at = [0, 1, 2, 3].map(|j| a[j] + (b[j] - a[j]) * t);
                let (w, gv) = closest(pr.nn, at);
                let n = gv.normalize_or_zero();
                if n == Vec3::ZERO {
                    continue;
                }
                fresh += 1;
                let s0: f32 = (0..4).map(|j| w[j] * n.dot(a[j])).sum();
                // The bound on the end separation along n: close to the gap ξ and no further.
                // (Measured from the step's start, as the CCD's paths are: the same bound taken
                // from the rounds' start positions cut side 60's zones from 2,750 to 30 but let
                // head 100 cross 7 times, 2026-11-27.)
                let mut r = VRow { q, k: pr.k, w, n, b: (s0 - pr.xi).max(0.0) / self.h, l: warm.prev.get(&pr.k).copied().unwrap_or(0.0), s0 };
                // Newton's law on a true impact: closing before the kick (u⁻ < 0) leaves parting at
                // e·|u⁻| at least.
                if e > 0.0 && !vpre.is_empty() {
                    let u = Self::normal(&r, vpre);
                    if u < 0.0 {
                        r.b = r.b.min(e * u);
                    }
                }
                match old {
                    Some(i) => {
                        relin += 1;
                        r.l = rows[i].l;
                        rows[i] = r;
                    }
                    None => {
                        listed.insert(q, rows.len());
                        rows.push(r);
                    }
                }
            }
            // Every pair still meeting its gap is a row the last solve settled: resolved.
            if fresh == 0 && settled {
                return Self::keep(&rows, warm, true);
            }
            if std::env::var("DAT_VDBG").is_ok() {
                eprintln!("vdbg round {} hits {} fresh {} relin {} settled {} rows {}", st.rounds, hit.len(), fresh, relin, settled, rows.len());
            }
            settled = self.solve(&mut rows, &vb, vel, st);
            for v in 0..x1.len() {
                for i in 0..x1[v].len() {
                    x1[v][i] = xb[v][i] + (vel[v][i] - vb[v][i]) * self.h;
                }
            }
        }
        Self::keep(&rows, warm, false)
    }
}

/// The step's velocity contact (module doc): `x0` the start, `x1` the end and `vel` the kicked
/// velocities (both moved by the impulses), `vpre` the velocities before the kick, `e` the impact
/// law's restitution. True when nothing collides any more.
#[allow(clippy::too_many_arguments)]
pub(super) fn velocity_rounds(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], h: f32, rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], e: f32, warm: &mut Warm) -> bool {
    let iters = std::env::var("DAT_ROW_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(ROW_ITERS);
    Ctx { pairs, x0, im, h, iters }.rounds(x1, vel, rounds, st, collide, vpre, e, warm)
}

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

#[path = "ccd_exact.rs"]
mod exact;

/// Warm starts (`Dat::warm`): each pair's impulse at the end of the last step, keyed by its four
/// nodes, as the first iterate of the next step's solve (Box2D's and PhysX's accumulated
/// impulses): a contact a crush presses for many steps starts near its answer.
#[derive(Default, Clone, Debug)]
pub struct Warm {
    pub prev: HashMap<[(usize, u32); 4], f32>,
    pub next: HashMap<[(usize, u32); 4], f32>,
    /// The step's kicked velocities before any response (a measurement's reference: the
    /// phases' work, `CcdStats::work_rows`; empty when not measured).
    pub vf: Vec<Vec<Vec3>>,
    /// Every pair that was a row in the last step and in this one (keyed by its four nodes): a
    /// pair newly closing is one that was not ([`VRow::imp`]).
    pub touched_prev: std::collections::HashSet<[(usize, u32); 4]>,
    pub touched_next: std::collections::HashSet<[(usize, u32); 4]>,
}

/// A pushing row's work past its bound, λ·excess (J, about twice the energy it may add; `DAT_PUSH_TOL`),
/// at most this a step: 10 µJ, a car's 2,000 steps a second over its hundred-odd rows at most
/// about 2 J a second of 200 kJ.
pub(super) const PUSH_TOL: f32 = 1.0e-5;

fn push_tol() -> f32 {
    std::env::var("DAT_PUSH_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(PUSH_TOL)
}

/// A pair already a row is linearised again only when it ends closer than this share of what it
/// may close to (`DAT_RELIN`): twice the settle test's slack ([`ROW_GAP_SHARE`]), so a row the
/// solve settled is not linearised again by its own rounding.
fn relin_share() -> f32 {
    std::env::var("DAT_RELIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0 - 2.0 * ROW_GAP_SHARE)
}

/// Or a pushing row parts past its bound by at most this share of the bound's speed (`DAT_PUSH_SHARE`):
/// the work bound alone asks a 55 m/s impact's row for 1e-6 m/s, below an f32's resolution there.
fn push_share() -> f32 {
    std::env::var("DAT_PUSH_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0e-5)
}

/// The exact cluster solve (`ccd_exact`), a measurement behind `DAT_EXACT=1`, off: it replaces
/// most rigid zones (head 100: 5,187 → 435 in 0.3 s) but lets panels deform where the zones held
/// them, and the paint, which does not follow the collision skin there yet, shows it: ball 100
/// 42 → 133 mm, side 90 → 86 mm, side 60 0 → 37 mm (the door top's paint, bound across a
/// half-metre triangle), 2026-11-27. On once the generator's paint meets the skin there.
fn lever() -> bool {
    std::env::var("DAT_LEVER").is_ok_and(|v| v == "1")
}

fn exact_on() -> bool {
    std::env::var("DAT_EXACT").is_ok_and(|v| v == "1")
}

fn pgs() -> bool {
    std::env::var("DAT_PGS").is_ok_and(|v| v == "1")
}

/// Units in the last place a row's closing may sit past its bound by rounding ([`Ctx::precision`]).
pub(super) const ROUND_ULPS: f32 = 4.0;

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
    /// A true impact: the pair closes before the step's kick (u⁻ < 0) and was not a row in the
    /// last step (newly closing). Poisson's restitution applies to it alone: a pair a crush keeps
    /// closing step after step is pressed, and its restitution was the pressed contact's bounce
    /// (the guard lowered e in 2,700 of head 100's passes, 2026-11-27).
    imp: bool,
}

struct Ctx<'a> {
    pairs: &'a [CcdPair],
    x0: &'a [Vec<Vec3>],
    im: &'a [Vec<f32>],
    h: f32,
    iters: u32,
    /// Newton's law on newly closing rows (`DAT_RESTITUTE=newton`, a measurement): J v⁺ ≥ −e·u⁻
    /// on a row that is a true impact ([`VRow::imp`]), in place of Poisson's form; 0: off.
    newton_e: f32,
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
        let prec = self.precision(rows, l, vel);
        rows.iter().zip(l).all(|(r, &l)| self.row_settled(r, l, vel, &prec))
    }

    fn row_settled(&self, r: &VRow, l: f32, vel: &[Vec<Vec3>], prec: &[Vec<f32>]) -> bool {
        let g = Self::normal(r, vel) + r.b;
        // Closing past the bound by at most ROW_GAP_SHARE of what it may close to: its gap, or its
        // start distance when it began inside the gap (as the re-linearisation's test: with the
        // gap alone here, a pair inside its gap settled where the rounds linearised it again
        // every round, and the rounds ran out into rigid zones, 2026-11-27).
        let rounding: f32 = (0..4).map(|j| r.w[j].abs() * prec[r.k[j].0][r.k[j].1 as usize]).sum();
        -g * self.h <= ROW_GAP_SHARE * self.pairs[r.q].xi.min(r.s0.max(0.0)) && (l * g <= push_tol() || g <= push_share() * r.b.abs() || g <= rounding)
    }

    /// Each node's velocity as far as its storage and sums can tell it (m/s): [`ROUND_ULPS`] units
    /// in the last place of its largest component and the fixed point's unit for each row added.
    /// (Not the impulses' own rounding, f32's ε of Σ|λ·w/m|: it grows with λ, so a solve of huge
    /// opposing impulses passed by its own size, and their lever arms turned head 100's cars by
    /// 16 N·m·s in a step, 2026-11-27.) A row pushing past its bound by
    /// less than its nodes' share of this is rounding: no solve in f32 settles it. (Head 100's
    /// exact cluster solves left λ ≈ 9 N·s at g = 1.7e-6 m/s, one unit at 28 m/s: 1.6e-5 J
    /// against [`PUSH_TOL`]; and near-parallel rows' large opposing impulses, 3–14e-6 m/s,
    /// 2026-11-27.)
    fn precision(&self, rows: &[VRow], l: &[f32], vel: &[Vec<Vec3>]) -> Vec<Vec<f32>> {
        let mut sum: Vec<Vec<(f32, u32)>> = vel.iter().map(|v| vec![(0.0, 0); v.len()]).collect();
        for (r, &l) in rows.iter().zip(l) {
            for j in 0..4 {
                let m = self.mi(r.k[j]);
                if m > 0.0 && r.w[j] != 0.0 {
                    let e = &mut sum[r.k[j].0][r.k[j].1 as usize];
                    e.0 += (r.w[j] * m * l).abs();
                    e.1 += 1;
                }
            }
        }
        vel.iter().zip(sum).map(|(v, s)| v.iter().zip(s).map(|(v, (a, n))| {
            let x = v.abs().max_element().max(f32::MIN_POSITIVE);
            let _ = a;
            ROUND_ULPS * ((f32::from_bits(x.to_bits() + 1) - x) + n as f32 / VEL_FIXED)
        }).collect()).collect()
    }

    /// APGD on the rows from their current λ: λ ≥ 0 ⟂ J v + b ≥ 0, v = base + M⁻¹Jᵀλ. Leaves
    /// `vel` at the final λ; true when settled.
    fn solve(&self, rows: &mut [VRow], base: &[Vec<Vec3>], vel: &mut [Vec<Vec3>], st: &mut CcdStats) -> bool {
        if pgs() {
            return self.solve_pgs(rows, base, vel, st);
        }
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
        if !ok && std::env::var("DAT_CLUSTERS").is_ok() {
            self.clusters_dbg("apgd", rows, &l, vel);
        }
        // The clusters APGD left unsettled, solved exactly (`ccd_exact`; `DAT_EXACT=1`, off).
        if !ok && exact_on() {
            // Kept only when it settles: a cluster too big, or a dependent row held bound, can
            // leave a row closing, and then APGD's answer stays for the zones.
            let mut lx = l.clone();
            self.exact(rows, &mut lx, vel, st);
            let mut vx = base.to_vec();
            self.velocities(rows, &lx, 1.0, base, &mut vx);
            ok = self.settled(rows, &lx, &vx);
            if std::env::var("DAT_CLUSTERS").is_ok() {
                let dv = |a: &[Vec<Vec3>]| a.iter().zip(base).flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (*a - *b).length())).fold(0f32, f32::max);
                let lmax = |l: &[f32]| l.iter().fold(0f32, |a, &b| a.max(b));
                eprintln!("exactdv ok {ok} | apgd max dv {:.3} m/s, max l {:.3} | exact max dv {:.3} m/s, max l {:.3}", dv(vel), lmax(&l), dv(&vx), lmax(&lx));
            }
            if ok {
                l = lx;
                vel.clone_from_slice(&vx);
            } else {
                st.exact_left += 1;
                if std::env::var("DAT_CLUSTERS").is_ok() {
                    self.clusters_dbg("exact", rows, &lx, &vx);
                }
            }
        }
        // A solve whose every row keeps its gap but some row pushes past its bound (over-push:
        // a slightly faster parting) is taken as resolved (`DAT_OVERPUSH=1`, a measurement): the
        // pass's work on the conserved energy is checked by the energy tests.
        if !ok && std::env::var("DAT_OVERPUSH").is_ok_and(|v| v == "1") && rows.iter().all(|r| self.keeps_gap(r, vel)) {
            ok = true;
            st.overpush += 1;
        }
        for (r, l) in rows.iter_mut().zip(l) {
            r.l = l;
        }
        ok
    }

    /// The row closes past its bound by at most [`ROW_GAP_SHARE`] of what it may close to.
    fn keeps_gap(&self, r: &VRow, vel: &[Vec<Vec3>]) -> bool {
        -(Self::normal(r, vel) + r.b) * self.h <= ROW_GAP_SHARE * self.pairs[r.q].xi.min(r.s0.max(0.0))
    }

    /// A measurement (`DAT_CLUSTERS=1`): the rows left unsettled and the sizes of the clusters
    /// that hold them.
    fn clusters_dbg(&self, tag: &str, rows: &[VRow], l: &[f32], vel: &[Vec<Vec3>]) {
        let comp = self.clusters(rows);
        let prec = self.precision(rows, l, vel);
        let mut size: HashMap<usize, (usize, usize)> = Default::default();
        for i in 0..rows.len() {
            let e = size.entry(comp[i]).or_default();
            e.0 += 1;
            if !self.row_settled(&rows[i], l[i], vel, &prec) {
                e.1 += 1;
            }
        }
        let mut v: Vec<(usize, usize)> = size.values().copied().filter(|x| x.1 > 0).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        let worst = rows.iter().zip(l).filter(|(r, &l)| !self.row_settled(r, l, vel, &prec)).map(|(r, &l)| { let g = Self::normal(r, vel) + r.b; (-g * self.h / self.pairs[r.q].xi.min(r.s0.max(1e-9)), l * g, g, r.b) }).fold((0f32, 0f32, 0f32, 0f32), |a, b| if b.1 > a.1 { (a.0.max(b.0), b.1, b.2, b.3) } else { (a.0.max(b.0), a.1, a.2, a.3) });
        eprintln!("clusters {tag} rows {} | clusters with unsettled (rows, unsettled): {:?} | worst closing/xi {:.3}, push work {:.2e} (g {:.2e}, b {:.2e})", rows.len(), &v[..v.len().min(8)], worst.0, worst.1, worst.2, worst.3);
    }

    /// The rounds: rows from the pairs colliding over the paths, solved, positions following, until
    /// nothing new collides. True when resolved.
    #[allow(clippy::too_many_arguments)]
    fn rounds(&self, x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], e: f32, warm: &mut Warm) -> bool {
        // Poisson's restitution (Stronge; frictionless multi-contact: no energy gained for
        // e ≤ 1): the impact solved inelastic first, its compression impulses λ_c; then e·λ_c
        // added on the true impacts' rows alone, positions following; then the rounds again,
        // inelastic, for what that extra parting meets. The kinetic energy's change is then
        // (1 + e)(e − 1)/2·|Δv_c|²_M: 0 at e = 1. (Newton's law in the rows themselves,
        // J v⁺ ≥ −e·u⁻ on every row at once, let coupled rows do +591 kJ of work over head
        // 100 at e = 1, 2026-11-27.)
        let mut rows = Vec::new();
        let done = self.rounds_inner(x1, vel, rounds, st, collide, vpre, warm, &mut rows);
        if !done || e <= 0.0 || self.newton_e > 0.0 {
            return done;
        }
        let imp: Vec<VRow> = rows.into_iter().filter(|r| r.imp && r.l > 0.0).collect();
        if imp.is_empty() {
            return done;
        }
        let l: Vec<f32> = imp.iter().map(|r| r.l).collect();
        let base = vel.to_vec();
        self.velocities(&imp, &l, e, &base, vel);
        for v in 0..x1.len() {
            for i in 0..x1[v].len() {
                x1[v][i] += (vel[v][i] - base[v][i]) * self.h;
            }
        }
        st.restituted += imp.len() as u64;
        let mut more = Vec::new();
        self.rounds_inner(x1, vel, rounds, st, collide, &[], warm, &mut more)
    }

    /// Projected Gauss–Seidel (`DAT_PGS=1`, a CPU measurement): each row in order moves its λ to
    /// where its own constraint holds against the velocities the rows before it left (the exact
    /// diagonal, Σ wⱼ²/mⱼ), the changes summed per node in 64 bits; settled as [`Self::solve`].
    /// Sequential in a fixed order: deterministic on the CPU (a GPU needs colours).
    fn solve_pgs(&self, rows: &mut [VRow], base: &[Vec<Vec3>], vel: &mut [Vec<Vec3>], st: &mut CcdStats) -> bool {
        let d: Vec<f32> = rows.iter().map(|r| (0..4).map(|j| r.w[j] * r.w[j] * self.mi(r.k[j])).sum()).collect();
        let mut l: Vec<f32> = rows.iter().map(|r| r.l).collect();
        // Velocities as f64 while sweeping (base + M⁻¹Jᵀλ), rebuilt exactly at the end.
        let mut v: Vec<Vec<glam::DVec3>> = base.iter().map(|x| x.iter().map(|u| u.as_dvec3()).collect()).collect();
        for (r, &lr) in rows.iter().zip(&l) {
            for j in 0..4 {
                let m = self.mi(r.k[j]);
                if m > 0.0 && lr != 0.0 {
                    v[r.k[j].0][r.k[j].1 as usize] += (r.n * (r.w[j] * m * lr)).as_dvec3();
                }
            }
        }
        self.velocities(rows, &l, 1.0, base, vel);
        let mut ok = self.settled(rows, &l, vel);
        let mut used = 0u32;
        while !ok && used < self.iters {
            used += 1;
            for (i, r) in rows.iter().enumerate() {
                if d[i] <= 0.0 {
                    continue;
                }
                let cv: f64 = (0..4).map(|j| r.w[j] as f64 * r.n.as_dvec3().dot(v[r.k[j].0][r.k[j].1 as usize])).sum();
                let nl = (l[i] as f64 - (cv + r.b as f64) / d[i] as f64).max(0.0) as f32;
                let dl = nl - l[i];
                if dl != 0.0 {
                    for j in 0..4 {
                        let m = self.mi(r.k[j]);
                        if m > 0.0 {
                            v[r.k[j].0][r.k[j].1 as usize] += (r.n * (r.w[j] * m * dl)).as_dvec3();
                        }
                    }
                    l[i] = nl;
                }
            }
            self.velocities(rows, &l, 1.0, base, vel);
            ok = self.settled(rows, &l, vel);
        }
        st.row_iters += used as u64;
        for (r, l) in rows.iter_mut().zip(l) {
            r.l = l;
        }
        ok
    }

    /// The rows' impulses into the next step's warm starts: pressed contacts only (an impact's
    /// impulse, b < 0, is no guess for the next step: seeded there, a node shared by many such
    /// rows started past 476 m/s and its fixed-point sums overflowed, head-on 100).
    fn keep(rows: &[VRow], warm: &mut Warm, done: bool) -> bool {
        // (`DAT_NOWARM=1`: none kept, a measurement.)
        let none = std::env::var("DAT_NOWARM").is_ok();
        for r in rows.iter().filter(|r| !none && r.l > 0.0 && r.b >= 0.0) {
            warm.next.insert(r.k, r.l);
        }
        done
    }

    #[allow(clippy::too_many_arguments)]
    fn rounds_inner(&self, x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], warm: &mut Warm, out: &mut Vec<VRow>) -> bool {
        let pos = |x: &[Vec<Vec3>], k: &[(usize, u32); 4]| k.map(|(v, i)| x[v][i as usize]);
        let (vb, xb) = (vel.to_vec(), x1.to_vec());
        let mut rows: Vec<VRow> = Vec::new();
        let mut listed: std::collections::HashMap<usize, usize> = Default::default();
        let mut settled = false;
        let keep_rows = std::env::var("DAT_KEEPROWS").is_ok_and(|v| v == "1");
        // Speculative rows (`DAT_SPEC=1`, a measurement; Catto's speculative contact): every pair
        // whose start distance is within its gap and the two sides' paths this step is a row from
        // the first round, linearised at the start, bounded by its own gap (b ≥ 0: inactive until
        // the solve needs it), so a round's response does not cascade into pairs not yet rows.
        if std::env::var("DAT_SPEC").is_ok_and(|v| v == "1") {
            for (q, pr) in self.pairs.iter().enumerate() {
                let (a, b) = (pos(self.x0, &pr.k), pos(x1, &pr.k));
                let path = |r: std::ops::Range<usize>| r.map(|j| (b[j] - a[j]).length()).fold(0.0f32, f32::max);
                let (w, gv) = closest(pr.nn, a);
                let d0 = gv.length();
                if d0 <= 0.0 || d0 - pr.xi > path(0..pr.nn) + path(pr.nn..4) {
                    continue;
                }
                let n = gv / d0;
                let s0: f32 = (0..4).map(|j| w[j] * n.dot(a[j])).sum();
                let mut r = VRow { q, k: pr.k, w, n, b: (s0 - pr.xi).max(0.0) / self.h, l: warm.prev.get(&pr.k).copied().unwrap_or(0.0), s0, imp: false };
                r.imp = !vpre.is_empty() && Self::normal(&r, vpre) < 0.0 && !warm.touched_prev.contains(&pr.k);
                warm.touched_next.insert(pr.k);
                listed.insert(q, rows.len());
                rows.push(r);
            }
            st.spec_rows += rows.len() as u64;
        }
        for _ in 0..rounds {
            let hit = collide(x1);
            st.rounds += 1;
            if hit.is_empty() {
                { let done = Self::keep(&rows, warm, true); *out = rows; return done; }
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
                    // And the pair's true closest gap at the end, along the row's normal: the
                    // row's own weights are those of its impact, and a feature that turned since
                    // can let the node through while that stays positive (head 100: a node
                    // 0.5 mm through its own triangle with the rounds "resolved", 2026-11-27).
                    let s_true = closest(pr.nn, b).1.dot(r.n);
                    if s_end.min(s_true) > 0.0 && s_end.min(s_true) >= relin_share() * r.s0.min(pr.xi) {
                        continue;
                    }
                }
                // A fresh row at its impact; a row linearised again at the step's end, where it
                // is tested (at its impact again, a turned feature's row settled each round with
                // the true end gap still under the test: the rounds ran out, 2026-11-27).
                // (Only while the end is on the row's side: a pair through at the end is
                // linearised at its impact, or its normal would point through.)
                let t = match old {
                    Some(i) if closest(pr.nn, b).1.dot(rows[i].n) > 0.0 => 1.0,
                    _ => t,
                };
                let at = [0, 1, 2, 3].map(|j| a[j] + (b[j] - a[j]) * t);
                let (w, gv) = closest(pr.nn, at);
                let mut n = gv.normalize_or_zero();
                if n == Vec3::ZERO {
                    continue;
                }
                // Torque-free rows (`DAT_LEVER=1`, a measurement): the step applies an impulse
                // from the start positions (x1 = x0 + h·v), so a row's torque is its start gap
                // g₀ = Σ wⱼ·x0ⱼ crossed with λn; along g₀ it is none (frictionless contact's line
                // of action through both points), taken where g₀ is within 60° of the impact's n.
                if lever() {
                    let g0: Vec3 = (0..4).map(|j| a[j] * w[j]).sum();
                    let d = g0.normalize_or_zero();
                    if d.dot(n) > 0.5 {
                        n = d;
                    }
                }
                fresh += 1;
                let s0: f32 = (0..4).map(|j| w[j] * n.dot(a[j])).sum();
                // The bound on the end separation along n: close to the gap ξ and no further.
                // (Measured from the step's start, as the CCD's paths are: the same bound taken
                // from the rounds' start positions cut side 60's zones from 2,750 to 30 but let
                // head 100 cross 7 times, 2026-11-27.)
                let mut r = VRow { q, k: pr.k, w, n, b: (s0 - pr.xi).max(0.0) / self.h, l: warm.prev.get(&pr.k).copied().unwrap_or(0.0), s0, imp: false };
                r.imp = !vpre.is_empty() && Self::normal(&r, vpre) < 0.0 && !warm.touched_prev.contains(&pr.k);
                if self.newton_e > 0.0 && r.imp {
                    r.b = r.b.min(self.newton_e * Self::normal(&r, vpre));
                }
                warm.touched_next.insert(pr.k);
                match old {
                    // Monotone rounds (`DAT_KEEPROWS=1`, a measurement): the pair's earlier row is
                    // kept beside its new one, so the constraint set only grows and a pair cannot
                    // cycle between its forms.
                    Some(_) if keep_rows => {
                        relin += 1;
                        listed.insert(q, rows.len());
                        rows.push(r);
                    }
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
                { let done = Self::keep(&rows, warm, true); *out = rows; return done; }
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
        let done = Self::keep(&rows, warm, false);
        *out = rows;
        done
    }
}

/// The step's velocity contact (module doc): `x0` the start, `x1` the end and `vel` the kicked
/// velocities (both moved by the impulses), `vpre` the velocities before the kick, `e` the impact
/// law's restitution. True when nothing collides any more.
#[allow(clippy::too_many_arguments)]
pub(super) fn velocity_rounds(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], h: f32, rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], e: f32, warm: &mut Warm) -> bool {
    let iters = std::env::var("DAT_ROW_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(ROW_ITERS);
    let newton_e = if std::env::var("DAT_RESTITUTE").is_ok_and(|v| v == "newton") { e } else { 0.0 };
    Ctx { pairs, x0, im, h, iters, newton_e }.rounds(x1, vel, rounds, st, collide, vpre, e, warm)
}

/// The contact's work m·(v − v_f)·(v⁻ + v)/2 is g(v) − g(v_f), g(v) = ½m(v·(v⁻ − v_f) + |v|²):
/// a phase's work is g's change over it. 0 when not measured (`vf` empty).
pub(super) fn shadow(vel: &[Vec<Vec3>], vpre: &[Vec<Vec3>], m: &[Vec<f32>], vf: &[Vec<Vec3>]) -> f64 {
    if vpre.is_empty() || vf.len() != vel.len() {
        return 0.0;
    }
    vel.iter().zip(vpre).zip(m).zip(vf).map(|(((v, p), m), f)| v.iter().zip(p).zip(m).zip(f).map(|(((v, p), &m), f)| 0.5 * m as f64 * (v.dot(*p - *f) as f64 + v.length_squared() as f64)).sum::<f64>()).sum()
}

/// [`velocity_rounds`] under the impact law's guard (the owner's rule: e never above the energy
/// before the pass): a pass whose rows did positive work (`g`'s change past
/// [`super::ccd::E_GUARD_TOL`]) is solved again from its start with e halved, then with e = 0.
/// (Coupled rows' Newton laws did +591 kJ over a head 100 run at e = 1, 2026-11-27.)
#[allow(clippy::too_many_arguments)]
pub(super) fn guarded_rounds(pairs: &[CcdPair], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], h: f32, rounds: u32, st: &mut CcdStats, collide: &dyn Fn(&[Vec<Vec3>]) -> Vec<(usize, f32)>, vpre: &[Vec<Vec3>], e: f32, warm: &mut Warm, g: &dyn Fn(&[Vec<Vec3>]) -> f64, g_in: f64) -> bool {
    let (x1s, vels) = (x1.to_vec(), vel.to_vec());
    let mut e_now = e;
    loop {
        let done = velocity_rounds(pairs, x0, x1, vel, im, h, rounds, st, collide, vpre, e_now, warm);
        if e_now == 0.0 || warm.vf.is_empty() || g(vel) - g_in <= super::ccd::E_GUARD_TOL {
            return done;
        }
        st.e_scaled += 1;
        e_now = if e_now > 0.5 * e { 0.5 * e } else { 0.0 };
        x1.clone_from_slice(&x1s);
        vel.clone_from_slice(&vels);
    }
}

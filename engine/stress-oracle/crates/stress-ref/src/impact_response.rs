//! A strike's structural response: what the whole member does when a shot's momentum goes
//! into it, beside the local outcome at the point struck (`impact_local`: crater, spall,
//! plug). `capability::ImpactResponse`; fracture.slang `fr_gi_solve`.
//!
//! The strike is a force pulse: a half sine of duration `tau` carrying the impulse `J` the
//! structure takes, so its peak is `F0 = pi J / (2 tau)`. A structure answers a pulse by its
//! own period. The load path solves the peak as a static load, which gives the deflected
//! shape `x`; Rayleigh's quotient of that shape estimates the natural frequency it
//! vibrates at, `omega^2 = (f . x) / (x . M x)`. The single-degree-of-freedom response to
//! the pulse (Biggs, "Introduction to Structural Dynamics", 1964, ch. 2; Chopra, "Dynamics
//! of Structures", 4.8) is then a factor on the static solution, the dynamic load factor:
//!   a pulse much shorter than the period (most strikes): the response is the impulse's,
//!     `F_eq = J omega`, whatever the peak (`dlf -> 2 r`, `r = omega tau / pi`);
//!   a pulse much longer than the period: the static load, `dlf -> 1`;
//!   between: up to 1.77 near `tau = 0.8 T`.
//! `PulseModel::Static` skips the factor (the peak as a static load: what a quasi-static
//! solver does with a contact force).

use std::f64::consts::PI;

/// Past this ratio the forced maxima are bounded instead of searched (`dlf_half_sine`).
pub const DLF_LONG: f64 = 33.0;

/// The dynamic load factor of an undamped single-degree-of-freedom system under a half-sine
/// pulse: its largest displacement over the static one under the peak. `r = omega tau / pi`
/// (the system's frequency over the pulse's; `2 tau / T`).
///
/// During the pulse `u / u_st = (sin(w_p t) - sin(w t) / r) / (1 - 1 / r^2)`, whose
/// maxima are at `t = 2 pi l / (w + w_p)` (within the pulse for `l <= (r + 1) / 2`: none
/// when `r < 1`); after it the system vibrates freely at amplitude `2 r cos(pi r / 2) /
/// |1 - r^2|`. The factor is the largest of these. Past `DLF_LONG` the bound `r / (r - 1)`
/// (within 3% of the factor there, above it).
pub fn dlf_half_sine(r: f64) -> f64 {
    if !(r > 0.0) {
        return 0.0;
    }
    if r > DLF_LONG {
        return r / (r - 1.0);
    }
    if (r - 1.0).abs() < 1e-4 {
        return 0.5 * PI;
    }
    let mut best = (2.0 * r * (0.5 * PI * r).cos() / (1.0 - r * r)).abs();
    let peaks = (0.5 * (r + 1.0)).floor() as u32;
    for l in 1..=peaks {
        // In units of the pulse: w_p t = 2 pi l / (r + 1), w t = r w_p t.
        let a = 2.0 * PI * l as f64 / (r + 1.0);
        best = best.max(((a.sin() - (r * a).sin() / r) / (1.0 - 1.0 / (r * r))).abs());
    }
    best
}

/// Rayleigh's estimate of the frequency (rad/s) a structure vibrates at in the shape `x` a
/// static load `f` gives it: `work = f . x` (N m), `inertia = x . M x` (kg m^2). 0 when the
/// shape is none (nothing moved: a rigid answer, the static load).
pub fn rayleigh_omega(work: f64, inertia: f64) -> f64 {
    if work > 0.0 && inertia > 0.0 {
        (work / inertia).sqrt()
    } else {
        0.0
    }
}

/// The factor on the static solution of the pulse's peak: `dlf_half_sine` at the
/// structure's frequency `omega`; 1 when it has none (rigid: the load is static to it).
pub fn dynamic_factor(omega: f64, tau: f64) -> f64 {
    if omega > 0.0 && tau > 0.0 {
        dlf_half_sine(omega * tau / PI)
    } else {
        1.0
    }
}

/// The impulse a strike of effective mass `m` closing at `v` leaves in the structure: all of
/// it when the striker stops; when it perforates (ballistic limit `v_bl`), what it and its
/// plug do not carry on with, `m (v - sqrt(v^2 - v_bl^2))` (`impact_local`'s residual
/// speed: the striker and plug leave together with the momentum `m sqrt(v^2 - v_bl^2)`).
pub fn transferred_impulse(m: f64, v: f64, v_bl: f64, perforates: bool) -> f64 {
    if perforates {
        m * (v - (v * v - v_bl * v_bl).max(0.0).sqrt())
    } else {
        m * v
    }
}

/// The pulse's duration: the Hertz contact's `t_c` for an elastic strike; when the target
/// is damaged, the time the striker takes over the depth it stops in or the thickness it
/// goes through, at its mean speed, if longer.
pub fn pulse_duration(t_c: f64, v: f64, v_res: f64, travel: f64) -> f64 {
    t_c.max(2.0 * travel / (v + v_res).max(1e-9))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The largest displacement over the static one, by integrating `u'' + w^2 u = w^2
    /// sin(w_p t)` through the pulse and a period after it.
    fn integrated(r: f64) -> f64 {
        let wp = PI;
        let w = r * wp;
        let end = 1.0 + 2.0 * PI / w;
        let n = (end * 400_000.0 * r.max(1.0)) as usize;
        let dt = end / n as f64;
        let (mut u, mut v, mut best) = (0.0f64, 0.0f64, 0.0f64);
        let acc = |t: f64, u: f64| w * w * ((if t < 1.0 { (wp * t).sin() } else { 0.0 }) - u);
        let mut a = acc(0.0, u);
        for i in 0..n {
            // Velocity Verlet.
            u += v * dt + 0.5 * a * dt * dt;
            let a1 = acc((i + 1) as f64 * dt, u);
            v += 0.5 * (a + a1) * dt;
            a = a1;
            best = best.max(u.abs());
        }
        best
    }

    #[test]
    fn the_factor_matches_the_integrated_response() {
        for r in [0.05, 0.2, 0.5, 0.9, 1.0, 1.1, 1.6, 2.0, 3.0, 4.5, 7.0, 12.0, 20.0, 30.0] {
            let (f, g) = (dlf_half_sine(r), integrated(r));
            assert!((f - g).abs() < 2e-3 * g.max(1.0), "r {r}: closed form {f}, integrated {g}");
        }
        // The long-pulse bound: above the response, within 3%.
        for r in [34.0, 60.0, 200.0] {
            let (f, g) = (dlf_half_sine(r), integrated(r));
            assert!(f >= g - 1e-3 && f < 1.03 * g, "r {r}: bound {f}, integrated {g}");
        }
    }

    #[test]
    fn a_short_pulse_is_its_impulse_and_a_long_one_its_peak() {
        // F_eq = J omega: with F0 = pi J / (2 tau), dlf = 2 r.
        let r = 1e-3;
        assert!((dlf_half_sine(r) / (2.0 * r) - 1.0).abs() < 1e-5);
        assert!((dlf_half_sine(500.0) - 1.0).abs() < 3e-3);
        // Never past the half sine's largest factor.
        let most = (1..4000).map(|i| dlf_half_sine(i as f64 * 0.01)).fold(0.0, f64::max);
        assert!((1.76..1.78).contains(&most), "{most}");
    }

    #[test]
    fn the_same_momentum_loads_a_stiff_member_more() {
        // One impulse, two members: the equivalent static force is J omega.
        let (j, tau) = (1000.0, 1e-3);
        let f0 = PI * j / (2.0 * tau);
        let soft = f0 * dynamic_factor(2.0 * PI * 5.0, tau);
        let stiff = f0 * dynamic_factor(2.0 * PI * 50.0, tau);
        assert!((soft / (j * 2.0 * PI * 5.0) - 1.0).abs() < 1e-3);
        assert!((stiff / soft - 10.0).abs() < 0.2);
    }

    #[test]
    fn a_perforating_shot_leaves_less_the_faster_it_goes() {
        let (m, v_bl) = (10.0, 100.0);
        assert_eq!(transferred_impulse(m, 50.0, v_bl, false), 500.0);
        let at = transferred_impulse(m, 100.0, v_bl, true);
        assert!((at - 1000.0).abs() < 1e-9);
        let fast = transferred_impulse(m, 400.0, v_bl, true);
        assert!(fast < at && fast > 0.0);
        assert!((rayleigh_omega(4.0, 1.0) - 2.0).abs() < 1e-12 && rayleigh_omega(0.0, 1.0) == 0.0);
    }
}

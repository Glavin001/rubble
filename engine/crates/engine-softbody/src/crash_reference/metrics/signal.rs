//! Signal processing for crash time histories, as crash tests process theirs: the SAE J211-1
//! channel-frequency-class filter and trapezoidal integration.
//!
//! **CFC filter** (SAE J211-1 Appendix C): a second-order Butterworth low-pass run forward
//! and then backward over the record (four poles, no phase shift), its digital design
//! frequency prewarped as ωd = 2π · CFC · 2.0775, so that the two passes together are
//! −3 dB at 1.667 × CFC (100 Hz for CFC 60):
//!
//! ```text
//! ωa = tan(ωd T / 2)
//! a0 = ωa² / (1 + √2 ωa + ωa²),  a1 = 2 a0,  a2 = a0
//! b1 = −2 (ωa² − 1) / (1 + √2 ωa + ωa²),  b2 = (−1 + √2 ωa − ωa²) / (1 + √2 ωa + ωa²)
//! y[i] = a0 x[i] + a1 x[i−1] + a2 x[i−2] + b1 y[i−1] + b2 y[i−2]
//! ```
//!
//! Each pass starts at rest at its first sample (x and y before the record equal to that
//! sample), so a constant record passes unchanged. The design is valid while ωd T / 2 < π/2,
//! i.e. the sample rate exceeds 2 × 2.0775 × CFC (249.3 Hz for CFC 60); [`cfc`] refuses
//! slower records rather than return a degenerate filter.

/// SAE J211-1's prewarping factor: the design frequency is this × CFC.
pub const J211_DESIGN: f64 = 2.0775;

/// `x` sampled every `dt` seconds through the J211 CFC `class` filter (two passes of a
/// two-pole Butterworth, no phase shift). Err when the sample rate is too low for the class.
pub fn cfc(x: &[f64], dt: f64, class: f64) -> Result<Vec<f64>, String> {
    let wd = 2.0 * std::f64::consts::PI * class * J211_DESIGN;
    let half = wd * dt / 2.0;
    if !(half > 0.0 && half < 0.49 * std::f64::consts::PI) {
        return Err(format!("CFC {class} needs a sample rate above {:.1} Hz (this record: {:.1} Hz)", 2.0 * class * J211_DESIGN / 0.98, 1.0 / dt));
    }
    let wa = half.tan();
    let den = 1.0 + std::f64::consts::SQRT_2 * wa + wa * wa;
    let a0 = wa * wa / den;
    let (a1, a2) = (2.0 * a0, a0);
    let b1 = -2.0 * (wa * wa - 1.0) / den;
    let b2 = (-1.0 + std::f64::consts::SQRT_2 * wa - wa * wa) / den;
    let pass = |s: &[f64]| -> Vec<f64> {
        let Some(&s0) = s.first() else { return Vec::new() };
        let (mut x1, mut x2, mut y1, mut y2) = (s0, s0, s0, s0);
        s.iter()
            .map(|&x| {
                let y = a0 * x + a1 * x1 + a2 * x2 + b1 * y1 + b2 * y2;
                (x2, x1, y2, y1) = (x1, x, y1, y);
                y
            })
            .collect()
    };
    let f = pass(x);
    let mut r: Vec<f64> = f.into_iter().rev().collect();
    r = pass(&r);
    r.reverse();
    Ok(r)
}

/// Trapezoidal running integral of `y` sampled every `dt`, starting at `y0`.
pub fn integrate(y: &[f64], dt: f64, y0: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(y.len());
    let mut acc = y0;
    for (i, v) in y.iter().enumerate() {
        if i > 0 {
            acc += 0.5 * dt * (v + y[i - 1]);
        }
        out.push(acc);
    }
    out
}

/// The derivative of `y` sampled every `dt`: central differences inside, one-sided at the
/// ends (the mean slope over the two samples' span: exact for a quadratic).
pub fn differentiate(y: &[f64], dt: f64) -> Vec<f64> {
    let n = y.len();
    (0..n)
        .map(|i| match (i.checked_sub(1), (i + 1 < n).then_some(i + 1)) {
            (Some(a), Some(b)) => (y[b] - y[a]) / (2.0 * dt),
            (None, Some(b)) => (y[b] - y[i]) / dt,
            (Some(a), None) => (y[i] - y[a]) / dt,
            (None, None) => 0.0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A half-sine deceleration pulse a(t) = −A sin(π t / T) on [0, T]: Δv = −2AT/π, and
    /// from v0 the displacement at T is v0 T − AT²/π.
    fn half_sine(a: f64, t: f64, dt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * dt).map(|x| if x <= t { -a * (PI * x / t).sin() } else { 0.0 }).collect()
    }

    #[test]
    fn crash_signal_integration_of_a_half_sine() {
        let (a, t, v0) = (400.0, 0.08, 15.6);
        for dt in [1e-4, 5e-4, 4e-3] {
            let n = (0.2 / dt) as usize + 1;
            let acc = half_sine(a, t, dt, n);
            let v = integrate(&acc, dt, v0);
            let s = integrate(&v, dt, 0.0);
            let k = (t / dt).round() as usize;
            let (dv, d) = (v[k] - v0, s[k]);
            let (dv_want, d_want) = (-2.0 * a * t / PI, v0 * t - a * t * t / PI);
            println!("dt {dt}: dv {dv:.5} (want {dv_want:.5}), d {d:.5} (want {d_want:.5})");
            // Trapezoid error on a sine: (π dt / T)² / 12 relative (Δv); the displacement
            // integrates that error once more (at most twice it, with the kink at T).
            let rel = (PI * dt / t).powi(2) / 12.0 * 3.0 + 1e-12;
            assert!((dv - dv_want).abs() <= rel * dv_want.abs(), "dv {dv} vs {dv_want}");
            assert!((d - d_want).abs() <= rel * d_want.abs() + 1e-9, "d {d} vs {d_want}");
        }
    }

    #[test]
    fn crash_signal_cfc60_passes_a_crash_pulse_and_keeps_its_area() {
        // A 80 ms half-sine of 40 g at 10 kHz after 20 ms at rest: CFC 60 passes it nearly
        // unchanged (its spectrum is below 30 Hz) and, having unit gain at DC, keeps its area
        // (Δv) to 1e-6; the peak stays in place (no phase shift).
        let (a, t, dt) = (40.0 * 9.80665, 0.08, 1e-4);
        let n = (0.3 / dt) as usize;
        let mut x = vec![0.0; 200];
        x.extend(half_sine(a, t, dt, n));
        let y = cfc(&x, dt, 60.0).unwrap();
        let peak = |s: &[f64]| s.iter().enumerate().fold((0, 0.0f64), |b, (i, v)| if v.abs() > b.1.abs() { (i, *v) } else { b });
        let ((ix, px), (iy, py)) = (peak(&x), peak(&y));
        let (ax, ay) = (x.iter().sum::<f64>() * dt, y.iter().sum::<f64>() * dt);
        println!("peak {px:.3} at {ix}, filtered {py:.3} at {iy}; area {ax:.6} vs {ay:.6}");
        assert!((py / px - 1.0).abs() < 0.01, "peak {py} vs {px}");
        assert!(iy.abs_diff(ix) <= 1, "no phase shift: {iy} vs {ix}");
        assert!((ay - ax).abs() < 1e-6 * ax.abs(), "area {ay} vs {ax}");
    }

    #[test]
    fn crash_signal_cfc60_corner_and_stop_band() {
        // Steady sines at 10 kHz: the two passes give |H|² of the two-pole design, −3 dB at
        // 100 Hz (1.667 × CFC), unity at DC and well below at 400 Hz.
        let dt = 1e-4;
        let gain = |f: f64| {
            let n = 40_000;
            let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * f * i as f64 * dt).sin()).collect();
            let y = cfc(&x, dt, 60.0).unwrap();
            // Amplitude over the middle half (away from the record's ends).
            let mid = &y[n / 4..3 * n / 4];
            let rms = (mid.iter().map(|v| v * v).sum::<f64>() / mid.len() as f64).sqrt();
            rms * 2f64.sqrt()
        };
        let (g10, g100, g400) = (gain(10.0), gain(100.0), gain(400.0));
        println!("|H| at 10 Hz {g10:.4}, 100 Hz {g100:.4}, 400 Hz {g400:.5}");
        assert!((g10 - 1.0).abs() < 0.005);
        assert!((g100 - 0.5f64.sqrt()).abs() < 0.02, "{g100}");
        assert!(g400 < 0.05, "{g400}");
        // A constant passes unchanged; a record too slow for CFC 60 is refused.
        let c = cfc(&[3.0; 50], dt, 60.0).unwrap();
        assert!(c.iter().all(|v| (v - 3.0).abs() < 1e-12));
        assert!(cfc(&[0.0; 50], 0.004, 60.0).is_err());
    }

    #[test]
    fn crash_signal_differentiate_is_exact_on_quadratics() {
        let dt = 0.004;
        let y: Vec<f64> = (0..20).map(|i| 3.0 + 2.0 * (i as f64 * dt) - 50.0 * (i as f64 * dt).powi(2)).collect();
        let d = differentiate(&y, dt);
        for (i, v) in d.iter().enumerate().skip(1).take(18) {
            let want = 2.0 - 100.0 * i as f64 * dt;
            assert!((v - want).abs() < 1e-9, "{v} vs {want}");
        }
    }
}

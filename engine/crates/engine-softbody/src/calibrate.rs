//! Calibration and validation metrics (design: "Calibration and validation").
//!
//! The same metrics compare a Rubble run against a reference run of the same
//! vehicle and test: Rigs of Rods recordings when available, or (the convergence
//! check) Rubble itself at a fine step and many iterations, which is what the
//! explicit 2,000 Hz engines approximate.

use glam::{Mat3, Vec3};

/// Best-fit rigid transform (rotation, translation) taking `from` onto `to`
/// (Kabsch with a polar decomposition), so crush is measured without the
/// vehicle's overall motion.
pub fn rigid_fit(from: &[Vec3], to: &[Vec3]) -> (Mat3, Vec3) {
    let n = from.len().max(1) as f32;
    let ca = from.iter().copied().sum::<Vec3>() / n;
    let cb = to.iter().copied().sum::<Vec3>() / n;
    // Cross-covariance M = Σ (b − cb)(a − ca)ᵀ; the rotation is its polar factor.
    let mut m = Mat3::ZERO;
    for (a, b) in from.iter().zip(to) {
        let (p, q) = (*a - ca, *b - cb);
        m += Mat3::from_cols(q * p.x, q * p.y, q * p.z);
    }
    let mut r = m;
    if r.determinant().abs() < 1e-12 {
        return (Mat3::IDENTITY, cb - ca);
    }
    for _ in 0..30 {
        let next = 0.5 * (r + r.inverse().transpose());
        if (next - r).abs().to_cols_array().iter().sum::<f32>() < 1e-7 {
            r = next;
            break;
        }
        r = next;
    }
    if r.determinant() < 0.0 {
        r = -r;
    }
    (r, cb - r * ca)
}

/// Per-node residual displacement after removing the best-fit rigid motion.
pub fn residual_displacement(rest: &[Vec3], now: &[Vec3]) -> Vec<f32> {
    let (r, t) = rigid_fit(rest, now);
    rest.iter()
        .zip(now)
        .map(|(a, b)| (*b - (r * *a + t)).length())
        .collect()
}

/// (RMS, max) of a set of displacements.
pub fn rms_max(d: &[f32]) -> (f32, f32) {
    let n = d.len().max(1) as f32;
    (
        (d.iter().map(|x| x * x).sum::<f32>() / n).sqrt(),
        d.iter().copied().fold(0.0, f32::max),
    )
}

/// Dominant frequency (Hz) of a signal sampled every `dt` seconds (DFT peak,
/// mean removed, below `max_hz`).
pub fn dominant_frequency(signal: &[f32], dt: f32, max_hz: f32) -> f32 {
    let n = signal.len();
    if n < 4 {
        return 0.0;
    }
    let mean = signal.iter().sum::<f32>() / n as f32;
    let duration = n as f32 * dt;
    let mut best = (0.0f32, 0.0f32);
    let kmax = ((max_hz * duration) as usize).min(n / 2);
    for k in 1..=kmax {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &x) in signal.iter().enumerate() {
            let ph = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            re += f64::from(x - mean) * ph.cos();
            im += f64::from(x - mean) * ph.sin();
        }
        let p = (re * re + im * im) as f32;
        if p > best.1 {
            best = (k as f32 / duration, p);
        }
    }
    best.0
}

/// Jaccard index of two broken-beam sets (1 = identical; two empty sets = 1).
pub fn jaccard(a: &[u32], b: &[u32]) -> f32 {
    let sa: std::collections::BTreeSet<_> = a.iter().collect();
    let sb: std::collections::BTreeSet<_> = b.iter().collect();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 1.0;
    }
    sa.intersection(&sb).count() as f32 / union as f32
}

/// Relative error |x − reference| / |reference|.
pub fn rel_error(x: f32, reference: f32) -> f32 {
    (x - reference).abs() / reference.abs().max(1e-9)
}

/// Golden-section search of `f` on [lo, hi] (calibration of one global setting).
pub fn minimize_1d(mut f: impl FnMut(f32) -> f32, mut lo: f32, mut hi: f32, iters: u32) -> f32 {
    let g = 0.618_034f32;
    let (mut x1, mut x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
    let (mut f1, mut f2) = (f(x1), f(x2));
    for _ in 0..iters {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - g * (hi - lo);
            f1 = f(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + g * (hi - lo);
            f2 = f(x2);
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn rigid_fit_recovers_a_transform() {
        let pts: Vec<Vec3> = (0..20)
            .map(|i| {
                Vec3::new(
                    (i as f32 * 0.7).sin(),
                    (i as f32 * 1.3).cos(),
                    i as f32 * 0.1,
                )
            })
            .collect();
        let q = Quat::from_rotation_y(0.9) * Quat::from_rotation_z(0.2);
        let t = Vec3::new(1.0, 2.0, -3.0);
        let moved: Vec<Vec3> = pts.iter().map(|p| q * *p + t).collect();
        let d = residual_displacement(&pts, &moved);
        assert!(d.iter().all(|x| *x < 1e-4), "{d:?}");
        let mut dented = moved.clone();
        dented[3] += Vec3::new(0.0, 0.05, 0.0);
        let (_, mx) = rms_max(&residual_displacement(&pts, &dented));
        assert!(mx > 0.04 && mx < 0.06);
    }

    #[test]
    fn frequency_of_a_sine() {
        let dt = 1.0 / 240.0;
        let s: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 7.0 * i as f32 * dt).sin())
            .collect();
        assert!((dominant_frequency(&s, dt, 50.0) - 7.0).abs() < 0.3);
    }

    #[test]
    fn jaccard_and_search() {
        assert_eq!(jaccard(&[1, 2, 3], &[2, 3, 4]), 0.5);
        assert_eq!(jaccard(&[], &[]), 1.0);
        let x = minimize_1d(|x| (x - 0.37).powi(2), 0.0, 1.0, 40);
        assert!((x - 0.37).abs() < 1e-3);
    }
}

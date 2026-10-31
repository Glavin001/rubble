//! Bekker–Wong soil for vehicle ruts (deformable terrain T5;
//! `engine/docs/deformable-terrain.md`, "T5 ruts").
//!
//! A heightfield may carry a [`Soil`]. Each step, every wheel standing on it
//! sinks to the quasi-static rigid-wheel sinkage of its load, the samples under
//! its contact patch are lowered plastically to that depth below the soil's
//! uncompacted surface (ruts only deepen), and the wheel pays the compaction
//! resistance of the soil it newly compacts. `slang/physics/soil.slang` is the
//! GPU side; the functions here mirror it for hosts and tests.
//!
//! Relations (Bekker 1956, 1969; Wong 2008, *Theory of Ground Vehicles*, ch. 2):
//! - pressure–sinkage: p = (k_c/b + k_φ)·zⁿ, b the smaller side of the
//!   loaded area (the tyre width);
//! - rigid wheel of diameter D under load W, the pressure acting radially on
//!   the arc ahead of the wheel's bottom (no rebound behind it) and the arc's
//!   chord approximated by x² = D·(z₀ − z), as Bekker does:
//!   W = b·k·√D·c_n·z₀^(n+½), so z₀ = [W / (b·k·√D·c_n)]^(2/(2n+1)), with
//!   k = k_c/b + k_φ and c_n = ∫₀¹(1 − s²)ⁿ ds = B(n+1, ½)/2. Bekker's
//!   closed form z₀ = [3W / ((3 − n)·b·k·√D)]^(2/(2n+1)) replaces c_n by its
//!   approximation (3 − n)/3: equal at n = 1, 5 % more load at n = 0.7. The
//!   integral is used here; it converges to the exact arc as z₀/D → 0;
//! - compaction (motion) resistance, the work of compacting a track of width b
//!   to depth z₀ per unit length: R_c = b·k·z₀^(n+1)/(n+1). Soil already
//!   compacted to z₁ by an earlier pass costs only b·k·(z₀^(n+1) − z₁^(n+1))/(n+1).

/// Bekker pressure–sinkage parameters, SI: `k_c` in N/m^(n+1), `k_phi` in
/// N/m^(n+2), `n` dimensionless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Soil {
    pub k_c: f32,
    pub k_phi: f32,
    pub n: f32,
    /// Deepest sinkage (m) the soil may reach. It bounds the field's height
    /// range (the collider's bounds hold every rut without a readback); a wheel
    /// whose load would sink it deeper fails the step with
    /// [`crate::OVERFLOW_SOIL_SINKAGE`].
    pub max_sinkage: f32,
}

/// Default [`Soil::max_sinkage`] of the presets (m).
pub const DEFAULT_MAX_SINKAGE: f32 = 0.5;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SoilError {
    #[error("soil k_c and k_phi must be finite and non-negative, and not both zero (got {k_c}, {k_phi})")]
    Moduli { k_c: f32, k_phi: f32 },
    #[error("soil exponent n must be in (0, 3) for the rigid-wheel relation (got {0})")]
    Exponent(f32),
    #[error("soil max_sinkage must be finite and positive (got {0})")]
    MaxSinkage(f32),
}

impl Soil {
    pub fn new(k_c: f32, k_phi: f32, n: f32, max_sinkage: f32) -> Result<Self, SoilError> {
        let soil = Self {
            k_c,
            k_phi,
            n,
            max_sinkage,
        };
        soil.validate()?;
        Ok(soil)
    }

    pub fn validate(&self) -> Result<(), SoilError> {
        let ok = |k: f32| k.is_finite() && k >= 0.0;
        if !ok(self.k_c) || !ok(self.k_phi) || self.k_c + self.k_phi <= 0.0 {
            return Err(SoilError::Moduli {
                k_c: self.k_c,
                k_phi: self.k_phi,
            });
        }
        if !(self.n.is_finite() && self.n > 0.0 && self.n < 3.0) {
            return Err(SoilError::Exponent(self.n));
        }
        if !(self.max_sinkage.is_finite() && self.max_sinkage > 0.0) {
            return Err(SoilError::MaxSinkage(self.max_sinkage));
        }
        Ok(())
    }

    /// Dry sand (Land Locomotion Laboratory), as tabulated by Wong (2008),
    /// Table 2.3: n = 1.1, k_c = 0.99 kN/m^(n+1), k_φ = 1528.43 kN/m^(n+2).
    pub fn dry_sand() -> Self {
        Self {
            k_c: 0.99e3,
            k_phi: 1528.43e3,
            n: 1.1,
            max_sinkage: DEFAULT_MAX_SINKAGE,
        }
    }

    /// Sandy loam, 15 % moisture (Land Locomotion Laboratory), Wong (2008),
    /// Table 2.3: n = 0.7, k_c = 5.27 kN/m^(n+1), k_φ = 1515.04 kN/m^(n+2).
    pub fn sandy_loam() -> Self {
        Self {
            k_c: 5.27e3,
            k_phi: 1515.04e3,
            n: 0.7,
            max_sinkage: DEFAULT_MAX_SINKAGE,
        }
    }

    pub fn with_max_sinkage(self, max_sinkage: f32) -> Self {
        Self {
            max_sinkage,
            ..self
        }
    }

    /// k = k_c/b + k_φ for a loaded area of smaller side `width` (m).
    pub fn modulus(&self, width: f32) -> f32 {
        sinkage_modulus(self.k_c, self.k_phi, width)
    }

    /// Plate sinkage under pressure `load / (width · length)`: z = (p/k)^(1/n).
    pub fn plate_sinkage(&self, load: f32, width: f32, length: f32) -> f32 {
        (load.max(0.0) / (width * length * self.modulus(width))).powf(1.0 / self.n)
    }

    /// Rigid-wheel sinkage (m) of a wheel of `radius` and `width` under
    /// `load` (N) (module docs).
    pub fn wheel_sinkage(&self, load: f32, width: f32, radius: f32) -> f32 {
        wheel_sinkage(
            load,
            width,
            2.0 * radius,
            self.modulus(width),
            self.n,
            self.shape_factor(),
        )
    }

    /// c_n = ∫₀¹(1 − s²)ⁿ ds (module docs), the rigid wheel's load factor.
    /// The heightfield packs it with the soil, so the GPU needs no Γ.
    pub fn shape_factor(&self) -> f32 {
        shape_factor(self.n)
    }

    /// [`series_sinkage`] for a wheel of `radius` and `width`.
    pub fn series_sinkage(
        &self,
        load: f32,
        z_now: f32,
        stiffness: f32,
        width: f32,
        radius: f32,
    ) -> f32 {
        series_sinkage(
            load,
            z_now,
            stiffness,
            width,
            2.0 * radius,
            self.modulus(width),
            self.n,
            self.shape_factor(),
        )
    }

    /// Compaction resistance (N) of a wheel of `width` sinking to `z` in soil
    /// already compacted to `z_before` (0 for a first pass).
    pub fn compaction_resistance(&self, width: f32, z: f32, z_before: f32) -> f32 {
        compaction_resistance(width, self.modulus(width), self.n, z, z_before)
    }
}

/// k_c/b + k_φ. Mirrors `soilModulus` in `soil.slang`.
pub fn sinkage_modulus(k_c: f32, k_phi: f32, width: f32) -> f32 {
    k_c / width + k_phi
}

/// c_n = ∫₀¹(1 − s²)ⁿ ds = ∫₀^(π/2) cos^(2n+1) φ dφ (s = sin φ, which removes
/// the endpoint singularity of n < 1), by Simpson's rule in f64 over 4096
/// panels: below f32 resolution for every n in (0, 3).
pub fn shape_factor(n: f32) -> f32 {
    let p = 2.0 * n as f64 + 1.0;
    let m = 4096;
    let h = std::f64::consts::FRAC_PI_2 / m as f64;
    let f = |phi: f64| phi.cos().max(0.0).powf(p);
    let mut sum = f(0.0) + f(std::f64::consts::FRAC_PI_2);
    for k in 1..m {
        sum += f(k as f64 * h) * if k % 2 == 1 { 4.0 } else { 2.0 };
    }
    (sum * h / 3.0) as f32
}

/// The rigid wheel's sinkage (module docs): `c` is [`shape_factor`] of `n`.
/// Mirrors `soilWheelSinkage`.
pub fn wheel_sinkage(load: f32, width: f32, diameter: f32, k: f32, n: f32, c: f32) -> f32 {
    let w = load.max(0.0);
    let base = w / (width * k * diameter.sqrt() * c);
    base.powf(2.0 / (2.0 * n + 1.0))
}

/// The load the soil carries under a rigid wheel sunk `z`: b·k·√D·c·z^(n+½)
/// (the inverse of [`wheel_sinkage`]). Mirrors `soilWheelLoad`.
pub fn wheel_load(z: f32, width: f32, diameter: f32, k: f32, n: f32, c: f32) -> f32 {
    if z <= 0.0 {
        return 0.0;
    }
    width * k * diameter.sqrt() * c * z.powf(n + 0.5)
}

/// The sinkage one step reaches with the soil in series with the suspension:
/// the wheel's bottom is a massless node between them, so the soil yields
/// until it carries what the suspension pushes, and that push falls by
/// `stiffness` (spring plus damper over the step, K + D/dt) per metre the
/// ground gives way. From depth `z_now` under `load`: the root of
/// soil(z) = load − S·(z − z_now), which lies between the static sinkage of
/// `load` and `z_now`. At rest (load = soil(z_now)) it is `z_now`; with S = 0
/// it is the static sinkage. Explicitly setting the static sinkage of the
/// last load instead feeds the damper's reaction to each drop back into the
/// next load (gain (D/dt)·dz/dW, about 2 for a sedan at 60 Hz), which
/// diverges. Mirrors `soilSeriesSinkage` (the same 32 bisection halvings).
#[allow(clippy::too_many_arguments)]
pub fn series_sinkage(
    load: f32,
    z_now: f32,
    stiffness: f32,
    width: f32,
    diameter: f32,
    k: f32,
    n: f32,
    c: f32,
) -> f32 {
    let zs = wheel_sinkage(load, width, diameter, k, n, c);
    let (mut lo, mut hi) = (zs.min(z_now), zs.max(z_now));
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        let g = wheel_load(mid, width, diameter, k, n, c) - load + stiffness * (mid - z_now);
        if g > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// b·k·(z^(n+1) − z₀^(n+1))/(n+1), never negative. Mirrors `soilCompaction`.
pub fn compaction_resistance(width: f32, k: f32, n: f32, z: f32, z_before: f32) -> f32 {
    let z0 = z_before.clamp(0.0, z.max(0.0));
    width * k * (z.max(0.0).powf(n + 1.0) - z0.powf(n + 1.0)) / (n + 1.0)
}

/// Contact length ahead of (and behind) the bottom of a wheel of `radius`
/// sunk `z`: the half chord √(2Rz − z²).
pub fn contact_half_length(radius: f32, z: f32) -> f32 {
    (2.0 * radius * z - z * z).max(0.0).sqrt()
}

/// Depth of the imprint a wheel leaves at a point, as the wheel's bottom
/// sweeps a segment: the point's lateral and longitudinal offsets (`lat`,
/// `long`) from the segment's start, the segment (`d_lat`, `d_long`) in the
/// same axes, the imprint's half width `half_width`, the sinkage `z` and the
/// wheel's `radius`. The imprint is the wheel's lower circle (depth
/// z − (R − √(R² − u²)) at u along its track) across its width. Mirrors
/// `soilImprintDepth`.
pub fn imprint_depth(
    lat: f32,
    long: f32,
    d_lat: f32,
    d_long: f32,
    half_width: f32,
    z: f32,
    radius: f32,
) -> f32 {
    let (mut s0, mut s1) = (0.0f32, 1.0f32);
    if d_lat.abs() > 1e-9 {
        let a = (lat - half_width) / d_lat;
        let b = (lat + half_width) / d_lat;
        s0 = s0.max(a.min(b));
        s1 = s1.min(a.max(b));
    } else if lat.abs() > half_width {
        return 0.0;
    }
    if s0 > s1 {
        return 0.0;
    }
    let s = if d_long.abs() > 1e-9 {
        (long / d_long).clamp(s0, s1)
    } else {
        s0
    };
    let u = (long - s * d_long).abs();
    if u >= radius {
        return 0.0;
    }
    let sag = radius - (radius * radius - u * u).sqrt();
    (z - sag).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wong's rigid-wheel equations integrated numerically: with the pressure
    /// p(θ) = k·(R(cos θ − cos θ₀))ⁿ acting radially on the arc 0..θ₀ (no
    /// rebound behind the wheel's bottom), the vertical load is
    /// W = b∫p R cos θ dθ and the horizontal resistance R_c = b∫p R sin θ dθ.
    fn integrate(soil: &Soil, width: f32, radius: f32, z0: f64) -> (f64, f64) {
        let (r, n, k) = (radius as f64, soil.n as f64, soil.modulus(width) as f64);
        let theta0 = (1.0 - z0 / r).acos();
        let steps = 20_000;
        let (mut w, mut rc) = (0.0, 0.0);
        for s in 0..steps {
            // Midpoint rule.
            let th = theta0 * (s as f64 + 0.5) / steps as f64;
            let z = r * (th.cos() - theta0.cos());
            let p = k * z.max(0.0).powf(n);
            let dth = theta0 / steps as f64;
            w += p * r * th.cos() * dth;
            rc += p * r * th.sin() * dth;
        }
        (width as f64 * w, width as f64 * rc)
    }

    #[test]
    fn compaction_resistance_is_the_horizontal_soil_reaction() {
        // R_c = b∫p dz is exact for radial pressure: dz = R sin θ dθ.
        for soil in [Soil::dry_sand(), Soil::sandy_loam()] {
            for z in [0.01f32, 0.03, 0.08] {
                let (_, rc) = integrate(&soil, 0.22, 0.34, z as f64);
                let want = soil.compaction_resistance(0.22, z, 0.0) as f64;
                assert!((rc - want).abs() < 1e-3 * want, "z {z}: {rc} vs {want}");
            }
        }
    }

    #[test]
    fn rigid_wheel_sinkage_carries_its_load() {
        // The sinkage carries the load the exact arc integral gives there, to
        // the chord approximation's error, which grows with z/D: within
        // 0.35·z/D (0.3·z/D for both soils: 1.7 % at z/D = 0.057, 5.2 % at 0.175).
        for soil in [Soil::dry_sand(), Soil::sandy_loam()] {
            for load in [1000.0f32, 3500.0, 6000.0] {
                let z = soil.wheel_sinkage(load, 0.22, 0.34);
                let (w, _) = integrate(&soil, 0.22, 0.34, z as f64);
                let err = (w - load as f64).abs() / load as f64;
                let rel = z as f64 / 0.68;
                eprintln!(
                    "n {} load {load}: z {z:.4} m (z/D {rel:.3}), arc load {w:.0} N, {:.2} %",
                    soil.n,
                    100.0 * err
                );
                assert!(
                    err < 0.35 * rel,
                    "n {} load {load}: arc carries {w} N",
                    soil.n
                );
            }
        }
    }

    #[test]
    fn shape_factor_is_half_the_beta_function() {
        // c_n = B(n+1, 1/2)/2: 2/3 at n = 1 (Bekker's (3 - n)/3), π/4 at n = 1/2,
        // 8/15 at n = 2.
        for (n, want) in [
            (1.0f32, 2.0 / 3.0),
            (0.5, std::f32::consts::FRAC_PI_4),
            (2.0, 8.0 / 15.0),
        ] {
            assert!(
                (shape_factor(n) - want).abs() < 1e-6,
                "n {n}: {}",
                shape_factor(n)
            );
        }
    }

    #[test]
    fn series_sinkage_balances_the_soil_against_the_suspension() {
        let soil = Soil::sandy_loam();
        let (b, r, w) = (0.22, 0.34, 3200.0);
        let zs = soil.wheel_sinkage(w, b, r);
        // No suspension in series: the static sinkage. At rest: no change.
        assert!((soil.series_sinkage(w, 0.0, 0.0, b, r) - zs).abs() < 1e-6);
        assert!((soil.series_sinkage(w, zs, 3e5, b, r) - zs).abs() < 1e-6);
        // From fresh soil under a stiff suspension: part way, at the balance.
        let s = 2.5e5;
        let z = soil.series_sinkage(w, 0.0, s, b, r);
        let carried = wheel_load(z, b, 2.0 * r, soil.modulus(b), soil.n, soil.shape_factor());
        assert!(z > 0.0 && z < zs, "{z} vs {zs}");
        assert!(
            (carried - (w - s * z)).abs() < 1e-3 * w,
            "{carried} vs {}",
            w - s * z
        );
        // Deeper than the load holds: the node rises toward it, as far as
        // the balance allows.
        let z = soil.series_sinkage(w, 2.0 * zs, s, b, r);
        assert!(z > zs && z < 2.0 * zs);
    }

    #[test]
    fn plate_pressure_follows_bekker() {
        let soil = Soil::sandy_loam();
        let z = soil.plate_sinkage(2000.0, 0.2, 0.5);
        let p = 2000.0 / (0.2 * 0.5);
        assert!((soil.modulus(0.2) * z.powf(soil.n) - p).abs() < 1e-3 * p);
    }

    #[test]
    fn a_compacted_track_costs_only_the_new_compaction() {
        let soil = Soil::dry_sand();
        let z = 0.05;
        assert_eq!(soil.compaction_resistance(0.22, z, z), 0.0);
        assert_eq!(soil.compaction_resistance(0.22, z, 0.08), 0.0);
        let first = soil.compaction_resistance(0.22, z, 0.0);
        let later = soil.compaction_resistance(0.22, z, 0.03);
        assert!(later > 0.0 && later < first);
    }

    #[test]
    fn imprint_is_the_wheel_circle_swept_along_its_track() {
        let (r, z, hw) = (0.34, 0.03, 0.12);
        // At rest: the circle's depth along the track, nothing past the width.
        assert!((imprint_depth(0.0, 0.0, 0.0, 0.0, hw, z, r) - z).abs() < 1e-7);
        let u: f32 = 0.1;
        let want = z - (r - (r * r - u * u).sqrt());
        assert!((imprint_depth(0.05, -u, 0.0, 0.0, hw, z, r) - want).abs() < 1e-7);
        assert_eq!(imprint_depth(0.13, 0.0, 0.0, 0.0, hw, z, r), 0.0);
        let l = contact_half_length(r, z);
        assert!(imprint_depth(0.0, l + 1e-3, 0.0, 0.0, hw, z, r) == 0.0);
        // Rolling 0.2 m forward: full depth along the whole segment.
        for long in [0.0f32, 0.07, 0.2] {
            assert!((imprint_depth(0.0, long, 0.0, 0.2, hw, z, r) - z).abs() < 1e-7);
        }
        // Sliding sideways 0.3 m: the strip it crosses, at full depth.
        assert!((imprint_depth(0.25, 0.0, 0.3, 0.0, hw, z, r) - z).abs() < 1e-7);
        assert_eq!(imprint_depth(0.45, 0.0, 0.3, 0.0, hw, z, r), 0.0);
        // Diagonal: the nearest point of the swept rectangle.
        let d = imprint_depth(0.1, 0.25, 0.1, 0.1, hw, z, r);
        let u = 0.15f32;
        assert!(
            (d - (z - (r - (r * r - u * u).sqrt())).max(0.0)).abs() < 1e-6,
            "{d}"
        );
    }

    #[test]
    fn validation_names_the_bad_parameter() {
        assert!(matches!(
            Soil::new(-1.0, 1.0, 1.0, 0.5),
            Err(SoilError::Moduli { .. })
        ));
        assert!(matches!(
            Soil::new(0.0, 0.0, 1.0, 0.5),
            Err(SoilError::Moduli { .. })
        ));
        assert_eq!(Soil::new(1.0, 1.0, 3.0, 0.5), Err(SoilError::Exponent(3.0)));
        assert_eq!(
            Soil::new(1.0, 1.0, 1.0, 0.0),
            Err(SoilError::MaxSinkage(0.0))
        );
        assert!(Soil::sandy_loam().validate().is_ok());
    }
}

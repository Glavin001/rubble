//! The initial excavation flow of a km/s point source: Maxwell's Z-model with
//! its flow centre below the surface (`engine/docs/deformable-terrain.md`,
//! "Event sources"), its magnitude calibrated by the point-source ejection
//! velocity of Housen & Holsapple (2011). The MPM starts from this field and
//! computes everything after it.
//!
//! # The field (Maxwell 1977)
//!
//! Maxwell (1977, "Simple Z model of cratering, ejection, and the overturned
//! flap", in Impact and Explosion Cratering), as restated by Kurosawa &
//! Takada (2019, Icarus 317, Eqs. 7–8; arXiv:1806.07665): in spherical
//! coordinates about the flow centre, θ from the downward vertical,
//!
//! ```text
//! u_r = α·r^(−Z),   u_θ = (Z − 2)·u_r·sin θ / (1 + cos θ)
//! streamlines:  r = R·(1 − cos θ)^(1/(Z−2))
//! ```
//!
//! which is incompressible (∇·u = 0) for any α and Z; R labels a streamline by
//! where it meets the horizontal plane through the centre. Z > 2 is needed for
//! streamlines to reach the surface. With the centre at the surface, every
//! streamline leaves it at atan(Z − 2) above horizontal (45° for Z = 3), and
//! the material inside the streamline leaving at R, the excavated volume, is
//! (2π/3)·(Z − 2)/(Z + 1)·R³ (integrating the streamline above).
//!
//! Croft (1980, "Cratering flow fields: implications for the excavation and
//! transient expansion stages of crater formation", Proc. LPSC 11) moved the
//! flow centre to a depth below the surface (the "EDZ" or Z-EDOZ model). Here
//! it sits at the impact's equivalent depth of burst (1.5 impactor radii,
//! Holsapple 1980) or the charge's centre; the depth is a parameter.
//!
//! The field is singular at the centre and, for a buried centre, along the
//! vertical above it (1 + cos θ → 0). Both lie inside a *core* of radius
//! r₀ > depth around the centre, where the shock stage would be; the core moves
//! as a uniform radial expansion that matches u_r at r₀:
//! u = α·r₀^(−Z)·(r/r₀)·r̂. The core is a modelling choice, not a published
//! one; its radius must keep the field's energy below the source's.
//!
//! Default Z is 2.7 (the design's value, after Maxwell's typical range;
//! Melosh 1989). Housen, Schmidt & Holsapple (1983, JGR 88) show a steady
//! Z-flow reproduces point-source ejecta scaling only for Z = 1/μ (2.44 for
//! sand); [`point_source_z`] gives it.
//!
//! # Magnitude (calibration)
//!
//! Housen & Holsapple (2011, "Ejecta from impact craters", Icarus 211, 856–875,
//! Eq. 14) give the speed at which ground at distance x from the impact point
//! is launched, from the coupling parameter alone:
//!
//! ```text
//! v(x)/U = C1·[(x/a)·(ρ/δ)^ν]^(−1/μ)·(1 − x/(n2·R))^p,   n1·a ≤ x ≤ n2·R
//! ```
//!
//! with, for sand, μ = 0.41, C1 = 0.55, n1 = 1.2, n2 = 1.3, p = 0.3 and
//! k = 0.3 (ejected volume k·x³ from inside x) (their sand set C4, as tabulated
//! by Hirata & Ikeya 2022, arXiv:2205.06607, Table B1), and ν = 0.4.
//!
//! The field's α is the one whose surface speed |u(x, 0)| best matches the
//! point-source power law C1·U·[(x/a)(ρ/δ)^ν]^(−1/μ) in least squares of
//! ln v over ln x, from the core's edge on the surface, √(r₀² − h²) (which must
//! be at least n1·a), out to the transient radius R:
//!
//! ```text
//! ln α = mean over ln x of [ln v_HH(x) − ln |u(x, 0)|_(α=1)]
//! ```
//!
//! For Z = 1/μ and the centre at the surface the two laws have the same form
//! and agree at every x. The rim factor (1 − x/(n2 R))^p is left out: it is the
//! flow stopping against gravity and strength, which the MPM computes. A steady
//! (constant-α) field is Housen et al.'s (1983) assumption under which the
//! field's surface speed is the launch speed.

//! # Extent
//!
//! A steady Z-flow fills the half-space with u ∝ r^(−Z). Applied at once to a
//! whole region, it moves ground far beyond the crater, and its downward
//! momentum grows with the region (2.4× the impactor's over a box 1.6 R wide
//! for the 5 mm / 2 km/s sand test; the volume came out 3.6× π-scaling's).
//! Bounded at the transient radius R about the impact point, the field keeps
//! almost all its energy (266 of 268 J there) and carries 1.26× the
//! impactor's momentum downward, a momentum-transfer β in the range measured
//! for impacts into granular targets: the target takes the impactor's
//! momentum plus the ejecta's recoil. Ground beyond R starts at rest and the
//! continuum moves it.

use crate::config::TerrainError;
use glam::{DVec3, Vec3};
use std::f64::consts::PI;

/// Default Maxwell exponent.
pub const DEFAULT_Z: f64 = 2.7;
/// Samples of ln x in the calibration's least squares.
const CALIBRATION_SAMPLES: usize = 512;
/// Samples of the energy quadrature.
const ENERGY_SAMPLES: usize = 2048;

/// The Z of a steady Z-flow consistent with point-source ejecta scaling,
/// Z = 1/μ (Housen et al. 1983).
pub fn point_source_z(mu: f64) -> f64 {
    1.0 / mu
}

/// Maxwell's excavated volume (centre at the surface): the material inside
/// the streamline that leaves the surface at `exit_radius` (m³).
pub fn excavated_volume(z: f64, exit_radius: f64) -> f64 {
    2.0 * PI / 3.0 * (z - 2.0) / (z + 1.0) * exit_radius.powi(3)
}

fn out_of_range(quantity: &'static str, value: f64, min: f64, max: f64) -> TerrainError {
    TerrainError::SourceOutOfRange {
        quantity,
        value,
        min,
        max,
    }
}

/// Housen & Holsapple's (2011) ejection-velocity constants for one target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EjectaScaling {
    pub mu: f64,
    pub nu: f64,
    pub c1: f64,
    /// The point-source law holds from n1·a ...
    pub n1: f64,
    /// ... out to n2·R.
    pub n2: f64,
    /// Exponent of the rim factor.
    pub p: f64,
    /// Ejected volume from inside x is k·x³.
    pub k: f64,
}

impl EjectaScaling {
    /// Sand (Housen & Holsapple 2011, set C4).
    pub const DRY_SAND: Self = Self {
        mu: 0.41,
        nu: 0.4,
        c1: 0.55,
        n1: 1.2,
        n2: 1.3,
        p: 0.3,
        k: 0.3,
    };

    pub fn validate(&self) -> Result<(), TerrainError> {
        if !(self.mu > 0.0 && self.mu <= 1.0) {
            return Err(out_of_range("ejecta mu", self.mu, 0.0, 1.0));
        }
        if !(0.0..=1.0).contains(&self.nu) {
            return Err(out_of_range("ejecta nu", self.nu, 0.0, 1.0));
        }
        for (q, v) in [("c1", self.c1), ("n1", self.n1), ("n2", self.n2)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(out_of_range(q, v, f64::MIN_POSITIVE, f64::MAX));
            }
        }
        Ok(())
    }

    /// The point-source launch speed at distance `x` (m) from the impact of an
    /// impactor of radius `a`, normal speed `u` and density `delta` into a
    /// target of density `rho`, without the rim factor.
    pub fn launch_speed(&self, a: f64, u: f64, delta: f64, rho: f64, x: f64) -> f64 {
        self.c1 * u * ((x / a) * (rho / delta).powf(self.nu)).powf(-1.0 / self.mu)
    }
}

/// A Z-model excavation field. Positions are relative to the source's point on
/// the surface, +y up; the target is y ≤ 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZModel {
    /// Maxwell's exponent (> 2).
    pub z: f64,
    /// Depth of the flow centre below the surface (m).
    pub flow_center_depth: f64,
    /// Radius of the core around the centre (m), greater than its depth.
    pub core_radius: f64,
    /// Flow strength α (m^(Z+1)/s).
    pub alpha: f64,
    /// The field moves the target within this distance of the surface point
    /// (m; infinite: everywhere). A source bounds it at the transient crater
    /// radius: the excavation flow is the crater's; ground beyond it starts at
    /// rest and is moved by the continuum (module docs, "Extent").
    pub extent: f64,
}

impl ZModel {
    pub fn new(
        z: f64,
        flow_center_depth: f64,
        core_radius: f64,
        alpha: f64,
    ) -> Result<Self, TerrainError> {
        if !(z.is_finite() && z > 2.0) {
            return Err(out_of_range("z", z, 2.0, f64::MAX));
        }
        if !(flow_center_depth.is_finite() && flow_center_depth >= 0.0) {
            return Err(out_of_range(
                "flow_center_depth",
                flow_center_depth,
                0.0,
                f64::MAX,
            ));
        }
        if !(core_radius.is_finite() && core_radius > flow_center_depth && core_radius > 0.0) {
            return Err(out_of_range(
                "core_radius",
                core_radius,
                flow_center_depth,
                f64::MAX,
            ));
        }
        if !(alpha.is_finite() && alpha >= 0.0) {
            return Err(out_of_range("alpha", alpha, 0.0, f64::MAX));
        }
        Ok(Self {
            z,
            flow_center_depth,
            core_radius,
            alpha,
            extent: f64::INFINITY,
        })
    }

    /// This field, moving the target only within `extent` of the surface point
    /// (at least the core's reach to the surface).
    pub fn bounded(self, extent: f64) -> Result<Self, TerrainError> {
        if !(extent.is_finite() && extent > self.core_radius) {
            return Err(out_of_range("extent", extent, self.core_radius, f64::MAX));
        }
        Ok(Self { extent, ..self })
    }

    /// The field whose surface speed matches Housen & Holsapple's point-source
    /// launch speed (module docs) between the core's edge and the transient
    /// `crater_radius`.
    #[allow(clippy::too_many_arguments)]
    pub fn calibrated(
        z: f64,
        flow_center_depth: f64,
        core_radius: f64,
        ejecta: &EjectaScaling,
        a: f64,
        u: f64,
        delta: f64,
        rho: f64,
        crater_radius: f64,
    ) -> Result<Self, TerrainError> {
        ejecta.validate()?;
        for (q, v) in [
            ("impactor radius", a),
            ("impact speed", u),
            ("impactor density", delta),
            ("target density", rho),
            ("crater radius", crater_radius),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(out_of_range(q, v, f64::MIN_POSITIVE, f64::MAX));
            }
        }
        let unit = Self::new(z, flow_center_depth, core_radius, 1.0)?;
        let x_lo = (core_radius * core_radius - flow_center_depth * flow_center_depth).sqrt();
        if !(x_lo >= ejecta.n1 * a && x_lo < crater_radius) {
            return Err(out_of_range(
                "core surface radius / impactor radius",
                x_lo / a,
                ejecta.n1,
                crater_radius / a,
            ));
        }
        let (l0, l1) = (x_lo.ln(), crater_radius.ln());
        let mut sum = 0.0;
        for k in 0..CALIBRATION_SAMPLES {
            let x = (l0 + (k as f64 + 0.5) / CALIBRATION_SAMPLES as f64 * (l1 - l0)).exp();
            sum += ejecta.launch_speed(a, u, delta, rho, x).ln() - unit.surface_speed(x).ln();
        }
        Self::new(
            z,
            flow_center_depth,
            core_radius,
            (sum / CALIBRATION_SAMPLES as f64).exp(),
        )
    }

    fn center(&self) -> DVec3 {
        DVec3::new(0.0, -self.flow_center_depth, 0.0)
    }

    /// Velocity (m/s) at `p` (m), a point of the target (y ≤ 0).
    pub fn velocity(&self, p: DVec3) -> DVec3 {
        if p.length() > self.extent {
            return DVec3::ZERO;
        }
        let d = p - self.center();
        let r = d.length();
        let r0 = self.core_radius;
        if r < r0 {
            return d * (self.alpha * r0.powf(-self.z) / r0);
        }
        let cos = -d.y / r;
        let f = (self.z - 2.0) / (1.0 + cos);
        let horizontal = d.x * d.x + d.z * d.z;
        let tangential = DVec3::new(cos * d.x / r, horizontal / (r * r), cos * d.z / r) * f;
        (d / r + tangential) * (self.alpha * r.powf(-self.z))
    }

    /// [`Self::velocity`] for the host's f32 particles.
    pub fn velocity_f32(&self, p: Vec3) -> Vec3 {
        self.velocity(p.as_dvec3()).as_vec3()
    }

    /// Speed at the surface at horizontal distance `x` (m).
    pub fn surface_speed(&self, x: f64) -> f64 {
        self.velocity(DVec3::new(x, 0.0, 0.0)).length()
    }

    /// Maxwell's streamline label r·(1 − cos θ)^(−1/(Z−2)) about the flow
    /// centre: constant along the flow (for a centre at the surface, where the
    /// streamline leaves the surface).
    pub fn stream_label(&self, p: DVec3) -> f64 {
        let d = p - self.center();
        let r = d.length();
        let cos = -d.y / r;
        r * (1.0 - cos).powf(-1.0 / (self.z - 2.0))
    }

    /// Kinetic energy of the field over the target (J) for bulk density `rho`,
    /// unbounded (an upper bound for a bounded field).
    pub fn kinetic_energy(&self, rho: f64) -> f64 {
        let (z, h, r0, a) = (self.z, self.flow_center_depth, self.core_radius, self.alpha);
        // Outside the core: ½ρα²∫ r^(2−2Z) A(r) dr, with A the angular integral
        // of 1 + (Z−2)²·tan²(θ/2) over the target's directions (cos θ ≥ −h/r)
        // and t = (r₀/r)^(2Z−3) mapping [r₀, ∞) onto (0, 1].
        let e = 2.0 * z - 3.0;
        let ln2 = std::f64::consts::LN_2;
        let mut sum = 0.0;
        for k in 0..ENERGY_SAMPLES {
            let t = (k as f64 + 0.5) / ENERGY_SAMPLES as f64;
            let r = r0 * t.powf(-1.0 / e);
            let c0 = -h / r;
            sum += 2.0
                * PI
                * ((1.0 - c0) + (z - 2.0).powi(2) * (2.0 * ln2 - 1.0 - 2.0 * (1.0 + c0).ln() + c0));
        }
        let outside = 0.5 * rho * a * a * r0.powf(3.0 - 2.0 * z) / e * sum / ENERGY_SAMPLES as f64;
        // The core: |u|² = α²r₀^(−2Z−2)·r² over the solid angle below the surface
        // (4π for r ≤ h, 2π(1 + h/r) beyond).
        let core = 0.5
            * rho
            * a
            * a
            * r0.powf(-2.0 * z - 2.0)
            * (4.0 * PI * h.powi(5) / 5.0
                + 2.0 * PI * ((r0.powi(5) - h.powi(5)) / 5.0 + h * (r0.powi(4) - h.powi(4)) / 4.0));
        outside + core
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{CraterTarget, ExplosionSource, ImpactSource};

    const G: f64 = 9.81;

    /// Deterministic points of the target outside the core.
    fn target_points(m: &ZModel, n: usize) -> Vec<DVec3> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut out = Vec::new();
        while out.len() < n {
            let r = m.core_radius * (1.05 + 20.0 * next());
            let p = DVec3::new(
                (2.0 * next() - 1.0) * r,
                -next() * r,
                (2.0 * next() - 1.0) * r,
            );
            if (p - m.center()).length() > 1.05 * m.core_radius && p.y < -1e-3 * r {
                out.push(p);
            }
        }
        out
    }

    fn divergence(m: &ZModel, p: DVec3) -> f64 {
        let h = 1e-5 * (p - m.center()).length();
        let mut div = 0.0;
        for k in 0..3 {
            let mut e = DVec3::ZERO;
            e[k] = h;
            div += (m.velocity(p + e)[k] - m.velocity(p - e)[k]) / (2.0 * h);
        }
        div
    }

    #[test]
    fn the_field_is_divergence_free_outside_the_core() {
        for (z, depth) in [
            (2.7, 0.0),
            (2.7, 0.15),
            (3.0, 0.3),
            (point_source_z(0.41), 0.1),
        ] {
            let m = ZModel::new(z, depth, 2.0 * depth + 0.05, 1.0).unwrap();
            for p in target_points(&m, 200) {
                let r = (p - m.center()).length();
                let scale = m.velocity(p).length() / r;
                let rel = divergence(&m, p).abs() / scale;
                assert!(rel < 1e-6, "z {z} depth {depth} at {p}: {rel}");
            }
        }
    }

    #[test]
    fn speed_falls_as_r_to_the_minus_z_along_a_ray() {
        let m = ZModel::new(2.7, 0.2, 0.5, 3.0).unwrap();
        for p in target_points(&m, 50) {
            let d = p - m.center();
            // Twice as far along the same ray from the centre (the same θ).
            let far = m.center() + 2.0 * d;
            let ratio = m.velocity(far).length() / m.velocity(p).length();
            assert!((ratio - 2f64.powf(-2.7)).abs() < 1e-12 * ratio.max(1.0));
            // And the direction is unchanged.
            let (a, b) = (m.velocity(p).normalize(), m.velocity(far).normalize());
            assert!(a.dot(b) > 1.0 - 1e-12);
        }
    }

    #[test]
    fn streamlines_leave_the_surface_at_atan_z_minus_2() {
        for z in [2.44, 2.7, 3.0, 3.5] {
            let m = ZModel::new(z, 0.0, 0.1, 1.0).unwrap();
            for x in [0.2, 0.7, 3.0, 11.0] {
                for phi in [0.0f64, 1.0, 2.5] {
                    let p = DVec3::new(x * phi.cos(), 0.0, x * phi.sin());
                    let v = m.velocity(p);
                    let horizontal = (v.x * v.x + v.z * v.z).sqrt();
                    let angle = v.y.atan2(horizontal);
                    assert!((angle - (z - 2.0).atan()).abs() < 1e-12, "z {z} x {x}");
                    // Radially outward.
                    assert!((v.x * p.x + v.z * p.z) > 0.0);
                }
            }
        }
        // A buried centre still launches every surface point upward and out.
        let m = ZModel::new(2.7, 0.3, 0.6, 1.0).unwrap();
        for x in [0.55, 0.8, 2.0, 9.0] {
            let v = m.velocity(DVec3::new(x, 0.0, 0.0));
            assert!(v.y > 0.0 && v.x > 0.0);
        }
    }

    #[test]
    fn particles_follow_maxwells_streamlines() {
        for depth in [0.0, 0.25] {
            let m = ZModel::new(2.7, depth, 2.0 * depth + 0.1, 1.0).unwrap();
            let mut p = DVec3::new(0.6, -0.5, 0.2);
            let label = m.stream_label(p);
            let dt = 2e-3;
            let mut steps = 0;
            // RK4 along the flow until the particle reaches the surface.
            while p.y < 0.0 && steps < 2_000_000 {
                let v = |q: DVec3| {
                    let u = m.velocity(q);
                    u / u.length()
                };
                let k1 = v(p);
                let k2 = v(p + 0.5 * dt * k1);
                let k3 = v(p + 0.5 * dt * k2);
                let k4 = v(p + dt * k3);
                p += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                steps += 1;
                let rel = (m.stream_label(p) / label - 1.0).abs();
                assert!(rel < 1e-8, "depth {depth}: label drifted {rel}");
            }
            assert!(p.y >= 0.0, "never reached the surface");
            if depth == 0.0 {
                // Centre at the surface: the streamline exits at R = label.
                let x = (p.x * p.x + p.z * p.z).sqrt();
                assert!((x / label - 1.0).abs() < 2e-3, "{x} vs {label}");
            }
        }
    }

    #[test]
    fn excavated_volume_is_maxwells() {
        // Count cells below the surface inside the streamline leaving at R.
        let (z, big_r) = (2.7, 1.0);
        let m = ZModel::new(z, 0.0, 0.01, 1.0).unwrap();
        let n = 200;
        let h = 2.0 * big_r / n as f64;
        let mut count = 0u64;
        for i in 0..n {
            for j in 0..n / 2 {
                for k in 0..n {
                    let p = DVec3::new(
                        -big_r + (i as f64 + 0.5) * h,
                        -(j as f64 + 0.5) * h,
                        -big_r + (k as f64 + 0.5) * h,
                    );
                    if m.stream_label(p) <= big_r {
                        count += 1;
                    }
                }
            }
        }
        let counted = count as f64 * h * h * h;
        let exact = excavated_volume(z, big_r);
        assert!((counted / exact - 1.0).abs() < 0.01, "{counted} vs {exact}");
    }

    #[test]
    fn point_source_z_excavates_housen_holsapples_volume() {
        // Housen & Holsapple (2011): sand ejects k·x³ (k = 0.3) from inside x.
        // Maxwell's Z-flow with Z = 1/μ ejects (2π/3)(Z−2)/(Z+1)·x³ = 0.27·x³.
        let e = EjectaScaling::DRY_SAND;
        let k_maxwell = excavated_volume(point_source_z(e.mu), 1.0);
        assert!(
            (k_maxwell / e.k - 1.0).abs() < 0.15,
            "{k_maxwell} vs {}",
            e.k
        );
    }

    #[test]
    fn calibration_matches_the_point_source_launch_speed() {
        let e = EjectaScaling::DRY_SAND;
        let (a, u, delta, rho, big_r) = (0.01, 3000.0, 2700.0, 1700.0, 0.4);
        // Z = 1/μ, centre at the surface: equal at every x.
        let m = ZModel::calibrated(point_source_z(e.mu), 0.0, 0.02, &e, a, u, delta, rho, big_r)
            .unwrap();
        for x in [0.02, 0.05, 0.13, 0.39] {
            let hh = e.launch_speed(a, u, delta, rho, x);
            assert!((m.surface_speed(x) / hh - 1.0).abs() < 1e-9, "x {x}");
        }
        // Any Z and depth: the log residual averages to zero over the range.
        let m = ZModel::calibrated(2.7, 0.015, 0.03, &e, a, u, delta, rho, big_r).unwrap();
        let x_lo = (0.03f64.powi(2) - 0.015f64.powi(2)).sqrt();
        let n = 4096;
        let mut mean = 0.0;
        for k in 0..n {
            let x = (x_lo.ln() + (k as f64 + 0.5) / n as f64 * (big_r / x_lo).ln()).exp();
            mean += (m.surface_speed(x) / e.launch_speed(a, u, delta, rho, x)).ln();
        }
        assert!((mean / n as f64).abs() < 1e-4);
        // A core inside the point-source range's start (n1·a) is rejected.
        assert!(ZModel::calibrated(2.7, 0.0, 0.011, &e, a, u, delta, rho, big_r).is_err());
        assert!(ZModel::calibrated(2.7, 0.0, 0.5, &e, a, u, delta, rho, big_r).is_err());
    }

    #[test]
    fn kinetic_energy_matches_direct_quadrature() {
        let rho = 1600.0;
        for (z, depth, r0) in [(2.7, 0.0, 0.1), (2.44, 0.2, 0.5), (3.2, 0.1, 0.15)] {
            let m = ZModel::new(z, depth, r0, 2.0).unwrap();
            // ½ρ|u|² summed over (r, cos θ) shells of the target, from the
            // velocity function itself, out to 400 core radii plus the
            // analytic far tail.
            let (nr, nc) = (4000, 400);
            let r_max = 400.0 * r0;
            let mut sum = 0.0;
            let (l0, l1) = ((1e-6 * r0).ln(), r_max.ln());
            for i in 0..nr {
                let r = (l0 + (i as f64 + 0.5) / nr as f64 * (l1 - l0)).exp();
                let dr = r * (l1 - l0) / nr as f64;
                let c_min = if r <= depth { -1.0 } else { -depth / r };
                for j in 0..nc {
                    let c = c_min + (j as f64 + 0.5) / nc as f64 * (1.0 - c_min);
                    let s = (1.0 - c * c).sqrt();
                    let p = DVec3::new(r * s, -depth - r * c, 0.0);
                    let v2 = m.velocity(p).length_squared();
                    sum += 0.5 * rho * v2 * 2.0 * PI * r * r * dr * (1.0 - c_min) / nc as f64;
                }
            }
            let tail_a =
                2.0 * PI * (1.0 + (z - 2.0).powi(2) * (2.0 * std::f64::consts::LN_2 - 1.0));
            sum += 0.5 * rho * 4.0 * tail_a * r_max.powf(3.0 - 2.0 * z) / (2.0 * z - 3.0);
            let ke = m.kinetic_energy(rho);
            assert!((sum / ke - 1.0).abs() < 2e-3, "z {z}: {sum} vs {ke}");
        }
    }

    #[test]
    fn source_fields_stay_within_the_source_energy() {
        for a in [0.002, 0.02, 0.2, 2.0] {
            let s = ImpactSource {
                impactor_radius: a,
                impactor_density: 2700.0,
                velocity: DVec3::new(0.0, -4000.0, 0.0),
                target: CraterTarget::DRY_SAND,
            };
            let m = s.field(G, DEFAULT_Z).unwrap();
            let ke = m.kinetic_energy(CraterTarget::DRY_SAND.crater.density);
            assert!(ke < s.kinetic_energy(), "a {a}: {ke}");
            assert_eq!(m.flow_center_depth, 1.5 * a);
        }
        for kg in [0.1, 1.0, 100.0] {
            let e = ExplosionSource::tnt(kg, 0.0, CraterTarget::DRY_SAND);
            let e = ExplosionSource {
                depth_of_burial: 1.5 * e.charge_radius(),
                ..e
            };
            let m = e.field(G, DEFAULT_Z).unwrap();
            assert!(m.kinetic_energy(CraterTarget::DRY_SAND.crater.density) < e.energy);
        }
        // A field stronger than the source fails by name.
        let mut target = CraterTarget::DRY_SAND;
        target.ejecta.c1 = 20.0;
        let s = ImpactSource {
            impactor_radius: 0.01,
            impactor_density: 2700.0,
            velocity: DVec3::new(0.0, -4000.0, 0.0),
            target,
        };
        assert!(matches!(
            s.field(G, DEFAULT_Z),
            Err(TerrainError::FieldEnergy { .. })
        ));
    }

    #[test]
    fn invalid_fields_fail() {
        assert!(ZModel::new(2.0, 0.0, 1.0, 1.0).is_err());
        assert!(
            ZModel::new(2.7, 0.5, 0.5, 1.0).is_err(),
            "core must exceed depth"
        );
        assert!(ZModel::new(2.7, -0.1, 1.0, 1.0).is_err());
        assert!(ZModel::new(2.7, 0.0, 1.0, f64::NAN).is_err());
    }
}

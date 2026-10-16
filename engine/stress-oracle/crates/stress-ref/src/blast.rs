//! Empirical blast loading (no gas dynamics): a pressure-time pulse on exposed faces.
//!
//! * Free-air incident overpressure, positive duration and impulse from the
//!   Kinney-Graham fits (Kinney & Graham, *Explosive Shocks in Air*, 1985), scaled by
//!   Hopkinson-Cranz distance `Z = R / W^(1/3)` (TNT equivalent mass `W`).
//! * Friedlander waveform `p(t) = P (1 - s) exp(-b s)`, `s = (t - t_a)/t_d`, with the
//!   decay `b` fitted so the waveform carries the Kinney-Graham impulse.
//! * Normal reflection `P_r = 2 P_so (7 P_0 + 4 P_so)/(7 P_0 + P_so)`; at incidence angle
//!   `theta`: `P = P_r cos^2 theta + P_so (1 + cos^2 theta - 2 cos theta)`. Faces turned
//!   away from the charge get no reflected load.
//! * Shadowing: a face whose line of sight to the charge crosses another chunk receives
//!   only the diffracted fraction `SHADOW_FACTOR` of the incident pressure.
//! * Clearing / venting (UFC 3-340-02): the reflected pressure on a face relaxes to the
//!   stagnation pressure `P_so + q` over `t_c = 3 S / U`, where `S` is the distance from
//!   the face to the nearest free edge of the reflecting surface. A breach creates new
//!   edges, so the surface around a hole clears (vents) faster.

pub const P_ATM: f64 = 101_325.0;
pub const SOUND_SPEED: f64 = 340.0;
/// Fraction of the incident overpressure reaching a shadowed face (diffraction).
pub const SHADOW_FACTOR: f64 = 0.2;

/// Incident (side-on) peak overpressure (Pa) at scaled distance `z` (m/kg^(1/3)).
pub fn incident_overpressure(z: f64) -> f64 {
    let z = z.max(0.05);
    let num = 808.0 * (1.0 + (z / 4.5).powi(2));
    let den = ((1.0 + (z / 0.048).powi(2)) * (1.0 + (z / 0.32).powi(2)) * (1.0 + (z / 1.35).powi(2))).sqrt();
    P_ATM * num / den
}

/// Positive-phase duration (s).
pub fn positive_duration(z: f64, w: f64) -> f64 {
    let z = z.max(0.05);
    let num = 980.0 * (1.0 + (z / 0.54).powi(10));
    let den = (1.0 + (z / 0.02).powi(3)) * (1.0 + (z / 0.74).powi(6)) * (1.0 + (z / 6.9).powi(2)).sqrt();
    num / den * 1e-3 * w.cbrt()
}

/// Incident positive-phase impulse per unit area (Pa s).
pub fn incident_impulse(z: f64, w: f64) -> f64 {
    let z = z.max(0.05);
    let bar_ms = 0.067 * (1.0 + (z / 0.23).powi(4)).sqrt() / (z * z * (1.0 + (z / 1.55).powi(3)).cbrt());
    bar_ms * 1e5 * 1e-3 * w.cbrt()
}

/// Shock front speed for an overpressure.
pub fn shock_speed(p_so: f64) -> f64 {
    SOUND_SPEED * (1.0 + 6.0 * p_so / (7.0 * P_ATM)).sqrt()
}

/// Arrival time at distance `r` (s): `integral dr / U(r)`.
pub fn arrival_time(r: f64, w: f64) -> f64 {
    let n = 200;
    let r0 = 0.05 * w.cbrt();
    if r <= r0 {
        return 0.0;
    }
    let inv_u = |x: f64| 1.0 / shock_speed(incident_overpressure(x / w.cbrt()));
    // Simpson in log r.
    let (la, lb) = (r0.ln(), r.ln());
    let h = (lb - la) / n as f64;
    let f = |l: f64| {
        let x = l.exp();
        inv_u(x) * x
    };
    let mut s = f(la) + f(lb);
    for i in 1..n {
        s += f(la + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    s * h / 3.0
}

/// Reflected peak overpressure at normal incidence.
pub fn reflected_overpressure(p_so: f64) -> f64 {
    2.0 * p_so * (7.0 * P_ATM + 4.0 * p_so) / (7.0 * P_ATM + p_so)
}

/// Peak overpressure on a face at incidence `cos_theta` (1 = facing the charge).
pub fn oblique_overpressure(p_so: f64, cos_theta: f64) -> f64 {
    let c = cos_theta.clamp(0.0, 1.0);
    reflected_overpressure(p_so) * c * c + p_so * (1.0 + c * c - 2.0 * c)
}

/// Peak dynamic pressure.
pub fn dynamic_pressure(p_so: f64) -> f64 {
    2.5 * p_so * p_so / (7.0 * P_ATM + p_so)
}

/// Impulse fraction of a unit Friedlander pulse: `integral_0^1 (1-s) e^(-b s) ds`.
pub fn friedlander_impulse_fraction(b: f64) -> f64 {
    if b.abs() < 1e-8 {
        return 0.5;
    }
    1.0 / b - (1.0 - (-b).exp()) / (b * b)
}

/// Decay coefficient `b` giving the pulse `impulse = peak * duration * fraction(b)`.
pub fn fit_decay(peak: f64, duration: f64, impulse: f64) -> f64 {
    let target = (impulse / (peak * duration)).clamp(1e-4, 0.5);
    let (mut lo, mut hi) = (0.0f64, 1e4f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let (was_lo, was_hi) = (lo, hi);
        if friedlander_impulse_fraction(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
        // A step that changes nothing is a fixed point (each step depends only on
        // `(lo, hi)`): the remaining steps would change nothing either.
        if lo.to_bits() == was_lo.to_bits() && hi.to_bits() == was_hi.to_bits() {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Unit Friedlander shape at normalised time `s`.
pub fn friedlander(s: f64, b: f64) -> f64 {
    if !(0.0..=1.0).contains(&s) {
        0.0
    } else {
        (1.0 - s) * (-b * s).exp()
    }
}

/// Blast parameters seen by one face.
#[derive(Clone, Copy, Debug)]
pub struct FaceBlast {
    pub arrival: f64,
    pub duration: f64,
    pub decay: f64,
    pub incident: f64,
    pub peak: f64,
    pub stagnation: f64,
    pub clearing_time: f64,
}

impl FaceBlast {
    /// `distance` to the charge, `cos_theta` between the outward face normal and the
    /// direction to the charge, `shadowed` line of sight, `clearing_distance` S.
    pub fn new(tnt: f64, detonation: f64, distance: f64, cos_theta: f64, shadowed: bool, clearing_distance: f64) -> FaceBlast {
        let z = distance / tnt.cbrt();
        let p_so = incident_overpressure(z);
        let duration = positive_duration(z, tnt);
        let decay = fit_decay(p_so, duration, incident_impulse(z, tnt));
        let (peak, stagnation) = if shadowed || cos_theta <= 0.0 {
            let p = if shadowed && cos_theta > 0.0 { SHADOW_FACTOR * p_so } else { 0.0 };
            (p, p)
        } else {
            let peak = oblique_overpressure(p_so, cos_theta);
            (peak, (p_so + dynamic_pressure(p_so) * cos_theta).min(peak))
        };
        FaceBlast {
            arrival: detonation + arrival_time(distance, tnt),
            duration,
            decay,
            incident: p_so,
            peak,
            stagnation,
            clearing_time: 3.0 * clearing_distance / shock_speed(p_so),
        }
    }

    pub fn pressure(&self, t: f64) -> f64 {
        let tau = t - self.arrival;
        if tau < 0.0 {
            return 0.0;
        }
        let shape = friedlander(tau / self.duration, self.decay);
        let relax = if self.clearing_time > 0.0 { (1.0 - tau / self.clearing_time).max(0.0) } else { 0.0 };
        (self.stagnation + (self.peak - self.stagnation) * relax) * shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinney_graham_reference_values() {
        // Z = 1 m/kg^(1/3): ~1 MPa side-on (Kinney & Graham table: 9.96 atm).
        let p = incident_overpressure(1.0);
        assert!((p / P_ATM - 9.96).abs() < 0.05, "{}", p / P_ATM);
        // Far field tends to acoustic: overpressure falls ~1/Z.
        let r = incident_overpressure(20.0) / incident_overpressure(40.0);
        assert!(r > 1.8 && r < 2.4);
        // Normal reflection factor is 2 for weak shocks and grows towards 8.
        assert!((reflected_overpressure(1.0) / 1.0 - 2.0).abs() < 1e-4);
        assert!(reflected_overpressure(50.0 * P_ATM) / (50.0 * P_ATM) > 6.0);
    }

    #[test]
    fn friedlander_fit_reproduces_impulse() {
        let (p, td, i) = (1e6, 1e-3, 220.0);
        let b = fit_decay(p, td, i);
        assert!((p * td * friedlander_impulse_fraction(b) - i).abs() < 1e-6 * i);
    }

    #[test]
    fn facing_faces_load_more_than_oblique_and_shadowed() {
        let front = FaceBlast::new(10.0, 0.0, 5.0, 1.0, false, 1.0);
        let oblique = FaceBlast::new(10.0, 0.0, 5.0, 0.5, false, 1.0);
        let shadow = FaceBlast::new(10.0, 0.0, 5.0, 1.0, true, 1.0);
        let back = FaceBlast::new(10.0, 0.0, 5.0, -1.0, false, 1.0);
        assert!(front.peak > oblique.peak && oblique.peak > shadow.peak && back.peak == 0.0);
        // A wider surface (larger S) clears more slowly, so it keeps more pressure.
        let small = FaceBlast::new(10.0, 0.0, 5.0, 1.0, false, 0.1);
        let t = front.arrival + 0.3 * front.duration;
        assert!(front.pressure(t) > small.pressure(t));
    }
}

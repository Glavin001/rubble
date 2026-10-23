//! Beam element model: energy derivatives for the per-particle block solve, the
//! force readout and the post-step plasticity and breaking rules. The Slang
//! kernels (`soft_step.slang`) implement the same functions; the CPU reference
//! solver uses these directly.
//!
//! With d = x_a − x_b, ℓ = |d|, d̂ = d/ℓ and an energy φ(ℓ), the block of particle a is
//!   ∇_a E = φ'(ℓ) d̂,   H_aa = φ''(ℓ) d̂d̂ᵀ + max(0, φ'(ℓ)/ℓ)(I − d̂d̂ᵀ),
//! (the clamp keeps H positive semi-definite in compression) and particle b gets
//! the negated gradient and the same Hessian.

use crate::layout::*;
use glam::{Mat3, Vec3};

/// Beam record in host form (the 16 GPU words, typed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beam {
    pub a: u32,
    pub b: u32,
    pub rest: f32,
    pub rest0: f32,
    pub spring: f32,
    pub damp: f32,
    /// Compression yield threshold; for bounded beams the limit spring.
    pub deform_c: f32,
    pub strength: f32,
    /// Plastic coefficient, or hydro factor for hydros.
    pub coef: f32,
    /// Bounded beams: bounds as fractions of L (short = compression). Normal beams:
    /// plastic rest-length limits as ratios of L₀ (0 = none). Supports: long =
    /// break factor.
    pub short_bound: f32,
    pub long_bound: f32,
    pub kind: u32,
    pub flags: u32,
    pub body: u32,
    pub group: u32,
    pub penalty: f32,
    pub lambda: f32,
    /// Tension yield threshold (positive).
    pub deform_t: f32,
    /// Signed axial force (+ = compression) at the end of the last step.
    pub force: f32,
}

pub const KIND_NORMAL: u32 = 0;
pub const KIND_SUPPORT: u32 = 1;
pub const KIND_ROPE: u32 = 2;
pub const KIND_BOUNDED: u32 = 3;
pub const KIND_HYDRO: u32 = 4;

impl Beam {
    pub fn to_words(&self) -> [u32; 16] {
        let mut w = [0u32; 16];
        w[BW_A] = self.a;
        w[BW_B] = self.b;
        w[BW_REST] = self.rest.to_bits();
        w[BW_REST0] = self.rest0.to_bits();
        w[BW_SPRING] = self.spring.to_bits();
        w[BW_DAMP] = self.damp.to_bits();
        w[BW_DEFORM_C] = self.deform_c.to_bits();
        w[BW_STRENGTH] = self.strength.to_bits();
        w[BW_COEF] = self.coef.to_bits();
        w[BW_SHORT] = self.short_bound.to_bits();
        w[BW_LONG] = self.long_bound.to_bits();
        w[BW_META] = (self.kind & 0xF)
            | (self.flags & 0xF0)
            | ((self.body & 0xFF) << 8)
            | ((self.group & 0xFFFF) << 16);
        w[BW_PENALTY] = self.penalty.to_bits();
        w[BW_LAMBDA] = self.lambda.to_bits();
        w[BW_DEFORM_T] = self.deform_t.to_bits();
        w[BW_FORCE] = self.force.to_bits();
        w
    }

    pub fn from_words(w: &[u32]) -> Self {
        let f = |i: usize| f32::from_bits(w[i]);
        let meta = w[BW_META];
        Self {
            a: w[BW_A],
            b: w[BW_B],
            rest: f(BW_REST),
            rest0: f(BW_REST0),
            spring: f(BW_SPRING),
            damp: f(BW_DAMP),
            deform_c: f(BW_DEFORM_C),
            strength: f(BW_STRENGTH),
            coef: f(BW_COEF),
            short_bound: f(BW_SHORT),
            long_bound: f(BW_LONG),
            kind: meta & 0xF,
            flags: meta & 0xF0,
            body: (meta >> 8) & 0xFF,
            group: meta >> 16,
            penalty: f(BW_PENALTY),
            lambda: f(BW_LAMBDA),
            deform_t: f(BW_DEFORM_T),
            force: f(BW_FORCE),
        }
    }

    pub fn broken(&self) -> bool {
        self.flags & BEAM_BROKEN != 0
    }

    pub fn hard(&self) -> bool {
        self.flags & BEAM_HARD != 0
    }
}

/// φ'(ℓ) and φ''(ℓ) of the elastic part. `k` is the stiffness in use (the AVBD
/// penalty while it ramps, the material k* for the readout).
pub fn elastic_terms(beam: &Beam, len: f32, k: f32) -> (f32, f32) {
    let l = beam.rest;
    let delta = len - l;
    match beam.kind {
        KIND_SUPPORT if delta > 0.0 => (0.0, 0.0),
        KIND_ROPE if delta < 0.0 => (0.0, 0.0),
        KIND_BOUNDED => {
            let mut d1 = k * delta;
            let mut d2 = k;
            let lo = l * (1.0 - beam.short_bound);
            let hi = l * (1.0 + beam.long_bound);
            let kl = beam.deform_c;
            if len < lo {
                d1 += kl * (len - lo);
                d2 += kl;
            } else if len > hi {
                d1 += kl * (len - hi);
                d2 += kl;
            }
            (d1, d2)
        }
        _ => {
            if beam.hard() {
                (k * delta + beam.lambda, k)
            } else {
                (k * delta, k)
            }
        }
    }
}

/// Damping coefficient in use for this state (supports and ropes slack at 10%).
pub fn damping_coef(beam: &Beam, len: f32, scale: f32) -> f32 {
    let delta = len - beam.rest;
    let slack =
        (beam.kind == KIND_SUPPORT && delta > 0.0) || (beam.kind == KIND_ROPE && delta < 0.0);
    beam.damp * scale * if slack { 0.1 } else { 1.0 }
}

/// Gradient and Hessian block of `beam` for the particle at `xa` (the other end at
/// `xb`), with step-start positions `xa0`, `xb0` for the axial dashpot.
#[allow(clippy::too_many_arguments)]
pub fn beam_block(
    beam: &Beam,
    xa: Vec3,
    xb: Vec3,
    xa0: Vec3,
    xb0: Vec3,
    h: f32,
    damp_scale: f32,
) -> (Vec3, Mat3) {
    let d = xa - xb;
    let len = d.length();
    if len < 1.0e-6 {
        return (Vec3::ZERO, Mat3::ZERO);
    }
    let n = d / len;
    let (mut d1, mut d2) = elastic_terms(beam, len, beam.penalty);
    let c = damping_coef(beam, len, damp_scale);
    let len0 = (xa0 - xb0).length();
    d1 += c / h * (len - len0);
    d2 += c / h;
    let nn = Mat3::from_cols(n * n.x, n * n.y, n * n.z);
    let lateral = (Mat3::IDENTITY - nn) * (elastic_terms(beam, len, beam.penalty).0 / len).max(0.0);
    (n * d1, nn * d2 + lateral)
}

/// Signed axial force (+ = compression, RoR's `slen`) with the material stiffness:
/// F = −(k*(ℓ − L) + c(ℓ − ℓ_prev)/h). Hard beams report their multiplier.
pub fn readout_force(beam: &Beam, len: f32, len0: f32, h: f32, damp_scale: f32) -> f32 {
    let (spring, _) = if beam.hard() {
        (beam.penalty * (len - beam.rest) + beam.lambda, 0.0)
    } else {
        elastic_terms(beam, len, beam.spring)
    };
    let c = damping_coef(beam, len, damp_scale);
    -(spring + c * (len - len0) / h)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlasticOutcome {
    pub yielded: bool,
    pub broke: bool,
}

/// Post-step plasticity and breaking (RoR rules) at the converged length.
/// `force` is [`readout_force`]. Only normal beams deform; hydros, shocks and
/// supports never do.
pub fn plastic_update(
    beam: &mut Beam,
    len: f32,
    force: f32,
    min_length: f32,
    plasticity: bool,
    breaking: bool,
) -> PlasticOutcome {
    let mut out = PlasticOutcome::default();
    beam.force = force;
    if beam.broken() {
        return out;
    }
    let delta = len - beam.rest;
    if plasticity && beam.kind == KIND_NORMAL && beam.spring > 0.0 {
        let k = beam.spring;
        if force > beam.deform_c && delta < 0.0 {
            // Compression past the threshold: the rest length moves so the beam
            // sits on the yield surface; the threshold hardens as it shortens.
            let old = beam.rest;
            let deform = delta + beam.deform_c * (1.0 - beam.coef) / k;
            beam.rest = (beam.rest + deform).max(min_length);
            if beam.short_bound > 0.0 {
                beam.rest = beam.rest.max(beam.rest0 * beam.short_bound);
            }
            beam.deform_c *= old / beam.rest;
            beam.flags |= BEAM_YIELDED;
            out.yielded = true;
        } else if -force > beam.deform_t && delta > 0.0 {
            // Tension: symmetric, and the beam weakens by the plastic stretch.
            let old = beam.rest;
            let deform = delta - beam.deform_t * (1.0 - beam.coef) / k;
            beam.rest += deform;
            if beam.long_bound > 0.0 {
                beam.rest = beam.rest.min(beam.rest0 * beam.long_bound);
            }
            beam.deform_t *= beam.rest / old;
            beam.strength -= deform * k;
            beam.flags |= BEAM_YIELDED;
            out.yielded = true;
        }
    }
    if breaking {
        let broke = match beam.kind {
            // A support breaks once stretched past L·factor.
            KIND_SUPPORT => delta > beam.rest * beam.long_bound && beam.long_bound > 0.0,
            _ => force.abs() > beam.strength,
        };
        if broke {
            beam.flags |= BEAM_BROKEN;
            out.broke = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn beam(len: f32) -> Beam {
        Beam {
            a: 0,
            b: 1,
            rest: len,
            rest0: len,
            spring: 1.0e6,
            damp: 0.0,
            deform_c: 1.0e4,
            strength: 1.0e5,
            coef: 0.0,
            short_bound: 0.0,
            long_bound: 0.0,
            kind: KIND_NORMAL,
            flags: 0,
            body: 0,
            group: 0,
            penalty: 1.0e6,
            lambda: 0.0,
            deform_t: 1.0e4,
            force: 0.0,
        }
    }

    #[test]
    fn words_round_trip() {
        let mut b = beam(1.0);
        b.kind = KIND_BOUNDED;
        b.flags = BEAM_BROKEN | BEAM_INVISIBLE;
        b.group = 513;
        b.body = 7;
        assert_eq!(Beam::from_words(&b.to_words()), b);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let b = beam(1.0);
        let xb = Vec3::new(0.1, -0.2, 0.05);
        let xa = Vec3::new(1.2, 0.9, 0.3);
        let e = |x: Vec3| 0.5 * b.penalty * ((x - xb).length() - b.rest).powi(2);
        let (g, h) = beam_block(&b, xa, xb, xa, xb, 1.0 / 60.0, 1.0);
        let eps = 1e-3;
        for i in 0..3 {
            let mut dp = Vec3::ZERO;
            dp[i] = eps;
            let fd = (e(xa + dp) - e(xa - dp)) / (2.0 * eps);
            assert!(
                (fd - g[i]).abs() < 1e-2 * g.length().max(1.0),
                "axis {i}: {fd} vs {}",
                g[i]
            );
            // Hessian column by central difference of the gradient (stretched beam: no clamp).
            let (gp, _) = beam_block(&b, xa + dp, xb, xa + dp, xb, 1.0 / 60.0, 1.0);
            let (gm, _) = beam_block(&b, xa - dp, xb, xa - dp, xb, 1.0 / 60.0, 1.0);
            let col = (gp - gm) / (2.0 * eps);
            assert!((col - h.col(i)).length() < 1e-2 * h.col(i).length().max(1.0));
        }
    }

    #[test]
    fn compressed_hessian_stays_psd() {
        let b = beam(1.0);
        let (_, h) = beam_block(
            &b,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::ZERO,
            0.01,
            1.0,
        );
        for v in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE.normalize()] {
            assert!(v.dot(h * v) >= -1e-6);
        }
    }

    #[test]
    fn compression_yield_lands_on_threshold_and_hardens() {
        let mut b = beam(1.0);
        b.strength = 1.0e6;
        let len = 0.9; // 0.1 m compression ⇒ 1e5 N > 1e4 N threshold
        let f = readout_force(&b, len, len, 0.01, 1.0);
        assert!((f - 1.0e5).abs() < 1.0);
        let o = plastic_update(&mut b, len, f, 0.1, true, true);
        assert!(o.yielded && !o.broke);
        // Coefficient 0: the elastic force after yielding equals the old threshold.
        assert!(
            (b.spring * (b.rest - len) - 1.0e4).abs() < 1.0,
            "rest {}",
            b.rest
        );
        assert!(b.deform_c > 1.0e4, "hardens as it shortens");
    }

    #[test]
    fn tension_yield_weakens_and_breaks() {
        let mut b = beam(1.0);
        let len = 1.08;
        let f = readout_force(&b, len, len, 0.01, 1.0);
        assert!(f < 0.0);
        plastic_update(&mut b, len, f, 0.1, true, true);
        assert!(b.rest > 1.0);
        assert!(b.strength < 1.0e5);
        let mut c = beam(1.0);
        c.deform_c = f32::INFINITY;
        c.deform_t = f32::INFINITY;
        let len = 1.2;
        let f = readout_force(&c, len, len, 0.01, 1.0);
        let o = plastic_update(&mut c, len, f, 0.1, true, true);
        assert!(o.broke && c.broken());
    }

    #[test]
    fn support_and_rope_are_one_sided() {
        let mut s = beam(1.0);
        s.kind = KIND_SUPPORT;
        assert_eq!(elastic_terms(&s, 1.2, 1e6).0, 0.0);
        assert!(elastic_terms(&s, 0.8, 1e6).0 < 0.0);
        let mut r = beam(1.0);
        r.kind = KIND_ROPE;
        assert_eq!(elastic_terms(&r, 0.8, 1e6).0, 0.0);
        assert!(elastic_terms(&r, 1.2, 1e6).0 > 0.0);
    }

    #[test]
    fn bounded_beam_hits_its_limit_spring() {
        let mut b = beam(1.0);
        b.kind = KIND_BOUNDED;
        b.spring = 1e4;
        b.penalty = 1e4;
        b.deform_c = 1e7;
        b.short_bound = 0.2;
        b.long_bound = 0.2;
        let inside = elastic_terms(&b, 0.9, b.penalty).0;
        let outside = elastic_terms(&b, 0.7, b.penalty).0;
        assert!((inside + 1e3).abs() < 1.0);
        assert!(outside < -1e5);
    }
}

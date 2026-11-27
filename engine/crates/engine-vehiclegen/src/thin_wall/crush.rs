//! Folding and collapse of thin-walled steel members: the established closed forms, each
//! with its source, the range it was established on, and how sure we are of its numbers.
//!
//! **Confidence.** These are written from the literature as we know it, without the papers
//! at hand. Each function says "verified" only where the coefficient is checked by an
//! independent identity (below), and "to be verified against the source" otherwise; the
//! report and `engine/docs/section-matching.md` list which is which.
//!
//! - The Wierzbicki–Abramowicz superfolding element (1983) gives a square tube's mean crush
//!   force P_m / M₀ = 38.12 (C/t)^(1/3) with M₀ = σ₀t²/4 (P_m = 9.53 σ₀ t^(5/3) C^(1/3); the
//!   coefficient is often quoted as 9.56).
//! - Abramowicz (1983) showed that a fold does not flatten completely: only an effective
//!   crush distance δ_e = 0.73 of the fold length 2H is crushed. Dividing by it,
//!   38.12 / 0.73 = 52.2: Abramowicz and Jones's P_m / M₀ = 52.22 (C/t)^(1/3) (1984, 1986),
//!   that is P_m = 13.06 σ₀ t^(5/3) C^(1/3). The identity 9.53 / 0.73 = 13.06 checks the two
//!   coefficients against each other, so 13.06 and 0.73 are the pair we are surest of.

/// The effective crush distance of a fold, δ_e / 2H (Abramowicz 1983, square tubes): a
/// folded wall stops shortening at 73 % of the folds' length. Used also as the usable
/// stroke of a progressively folding member before its folds stack solid (a tube's
/// "stroke efficiency"; crash literature quotes 0.70–0.75 for steel tubes, e.g. Jones,
/// *Structural Impact*, ch. 7). To be verified against the source for the stroke use.
pub const EFFECTIVE_CRUSH_FRACTION: f64 = 0.73;

/// Cowper–Symonds constants for mild steel, D = 40.4 s⁻¹, q = 5 (Cowper and Symonds 1957,
/// fitted to Manjoine's 1944 tests). Confident. They overstate the rate effect of
/// high-strength steels.
pub const CS_MILD_STEEL: (f64, f64) = (40.4, 5.0);
/// The rate scaling in the CCSA 2012 Camry model's steel cards, C = 8,000 s⁻¹, p = 8
/// (`engine/docs/fe-import-spike.md`, read from the deck): the automotive grades of a
/// validated vehicle model.
pub const CS_CAMRY_DECK: (f64, f64) = (8000.0, 8.0);

/// Abramowicz–Jones mean crush force of a square tube of side `c` (mean side for a
/// rectangle: (b + h)/2), wall `t`, flow stress `sigma0` (N):
/// P_m = 13.06 σ₀ t² (C/t)^(1/3) = 13.06 σ₀ t^(5/3) C^(1/3).
///
/// Abramowicz W., Jones N., "Dynamic axial crushing of square tubes", Int. J. Impact Eng.
/// 2(2), 1984, and "Dynamic progressive buckling of circular and square tubes", Int. J.
/// Impact Eng. 4(4), 1986 (P_m / M₀ = 52.22 (C/t)^(1/3), M₀ = σ₀t²/4). Validity: progressive
/// (not global) folding of a straight steel tube under axial load, square tubes with C/t of
/// roughly 10–50 in their tests (the range is to be verified against the source); a
/// rectangle as a square of its mean side is common practice for aspect ratios near 1–2,
/// not part of the derivation. Coefficient: verified by the identity in the module
/// comment. Quasi-static; see [`dynamic_factor`] for rate.
pub fn abramowicz_jones(sigma0: f64, t: f64, c: f64) -> f64 {
    13.06 * sigma0 * t.powf(5.0 / 3.0) * c.powf(1.0 / 3.0)
}

/// The superfolding element's mean crush force of a square tube of side `b`, wall `t`
/// (Wierzbicki T., Abramowicz W., "On the crushing mechanics of thin-walled structures",
/// J. Appl. Mech. 50, 1983): P_m = 9.56 σ₀ b^(1/3) t^(5/3), the force averaged over the
/// full fold length (folds assumed to flatten completely). The coefficient as commonly
/// quoted; 38.12 / 4 = 9.53 from the paper's P_m / M₀ = 38.12 (C/t)^(1/3), 0.3 % apart: to
/// be verified against the source. It is [`abramowicz_jones`] without the effective crush
/// distance (× 0.73), and under-predicts tests for that reason.
pub fn wierzbicki_abramowicz(sigma0: f64, t: f64, b: f64) -> f64 {
    9.56 * sigma0 * b.powf(1.0 / 3.0) * t.powf(5.0 / 3.0)
}

/// A rectangular tube `b` × `h` (mean sides) folding as a square tube of its mean side
/// (N): [`abramowicz_jones`] with C = (b + h) / 2.
pub fn rect_tube_mean_crush(sigma0: f64, t: f64, b: f64, h: f64) -> f64 {
    abramowicz_jones(sigma0, t, 0.5 * (b + h))
}

/// The mean strain rate in the folds of a square tube crushed at `v` (m/s), side `c`:
/// ε̇ ≈ 0.33 v / C (Abramowicz and Jones 1984, from the fold's kinematics). The form is
/// theirs as we recall it; the coefficient is to be verified against the source. `v` is
/// the crush speed: at the start of a crash the impact speed, falling to zero at t_m (a
/// mean over the crush of a uniformly decelerated car is about v₀/2).
pub fn fold_strain_rate(v: f64, c: f64) -> f64 {
    0.33 * v / c
}

/// The Cowper–Symonds dynamic flow stress ratio σ_d / σ₀ = 1 + (ε̇ / D)^(1/q).
/// (Cowper G.R., Symonds P.S., Brown Univ. Tech. Rep. 28, 1957.) Confident in the form.
pub fn cowper_symonds(rate: f64, (d, q): (f64, f64)) -> f64 {
    1.0 + (rate / d).powf(1.0 / q)
}

/// The mean crush force at crush speed `v` over the quasi-static one: Cowper–Symonds at
/// [`fold_strain_rate`] (Abramowicz–Jones 1984 apply the factor to the whole P_m).
pub fn dynamic_factor(v: f64, c: f64, cs: (f64, f64)) -> f64 {
    cowper_symonds(fold_strain_rate(v, c), cs)
}

/// The effective-width ratio b_e / b of a plate strip of width `b`, gauge `t`, simply
/// supported on both long edges, compressed to its yield (Winter's formula, as in AISI S100
/// and EN 1993-1-5 §4.4): σ_cr = 4 π² E / (12 (1 − ν²)) (t/b)², λ = √(σ_y / σ_cr),
/// ρ = (1 − 0.22/λ)/λ for λ > 0.673, else 1. Confident.
pub fn winter_effective_width(b: f64, t: f64, e: f64, sigma_y: f64) -> f64 {
    let nu: f64 = 0.3;
    let s_cr = 4.0 * std::f64::consts::PI.powi(2) * e / (12.0 * (1.0 - nu * nu)) * (t / b).powi(2);
    let lam = (sigma_y / s_cr).sqrt();
    if lam <= 0.673 {
        1.0
    } else {
        ((1.0 - 0.22 / lam) / lam).min(1.0)
    }
}

/// A rectangular box's bending collapse moment about its horizontal axis, `b` wide by `h`
/// high (outer), wall `t`, compression on its top flange (N·m): the fully plastic moment of
/// the section with that flange reduced to its effective width ([`winter_effective_width`],
/// half at each web). This follows the idea of Kecman's model (Kecman D., "Bending collapse
/// of rectangular and square section tubes", Int. J. Mech. Sci. 25, 1983: the compression
/// flange's effective width sets the maximum moment) but is **not** his formula, which we
/// do not have at hand: an estimate, to be checked against the source. It is an upper
/// estimate (the whole remaining section taken fully plastic). Returns (moment, the
/// effective-width ratio).
pub fn box_bending_collapse(b: f64, h: f64, t: f64, m: &super::Material) -> Result<(f64, f64), String> {
    let (x, y) = (0.5 * (b - t), 0.5 * (h - t));
    let rho = winter_effective_width(b - 2.0 * t, t, m.e, m.yield_pa);
    if rho >= 1.0 {
        let p = super::ThinSection::rect_tube(b, h, t).properties()?;
        return Ok((m.yield_pa * p.zx, 1.0));
    }
    let stub = 0.5 * rho * (b - 2.0 * t);
    let pts = vec![[-x + stub, y], [-x, y], [-x, -y], [x, -y], [x, y], [x - stub, y]];
    let s = super::ThinSection { walls: vec![super::Wall { points: pts, t, closed: false }] };
    Ok((m.yield_pa * s.properties()?.zx, rho))
}

/// The axial peak (first-buckle) force of a rectangular tube `b` × `h` (outer), wall `t`: its
/// squash load with each wall at its effective width, σ_y·A_eff, A_eff = t·Σ ρ·w + 4t² (the walls'
/// flats w = b − 2t, h − 2t at [`winter_effective_width`], the four corners fully effective). The
/// effective-width method is the established ultimate load of a thin-walled stub column (AISI S100,
/// EN 1993-1-5 §4.4): confident as that. Its use as the initial peak of a crushing tube (the force
/// at the first buckle, before the first fold forms) is an engineering estimate; tests of steel
/// tubes show peaks between this and the full squash load. Returns (force N, ρ_b, ρ_h).
pub fn box_axial_peak(b: f64, h: f64, t: f64, m: &super::Material) -> (f64, f64, f64) {
    let (wb, wh) = (b - 2.0 * t, h - 2.0 * t);
    let rb = winter_effective_width(wb, t, m.e, m.yield_pa);
    let rh = winter_effective_width(wh, t, m.e, m.yield_pa);
    (m.yield_pa * (t * 2.0 * (rb * wb + rh * wh) + 4.0 * t * t), rb, rh)
}

/// The half-length H of a square tube's fold (side `c`, wall `t`): H = 0.99 t^(1/3) c^(2/3)
/// (Wierzbicki and Abramowicz 1983, the superfolding element; the coefficient as we recall it,
/// to be verified against the source). The force falls from its peak to the folding plateau
/// while the first lobe forms, over about this much shortening (an estimate).
pub fn fold_half_length(t: f64, c: f64) -> f64 {
    0.99 * t.powf(1.0 / 3.0) * c.powf(2.0 / 3.0)
}

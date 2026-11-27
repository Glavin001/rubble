//! The front crash structure's rules ([`crate::physics::FrontRules`]): the parts of each rule
//! that are statements about the definitions (which nodes a mount lands on, a member's force
//! at crash rate, a panel's buckling force). The rigger (`engine-softbody-authoring`
//! `authoring::gen2::front_rules`) applies them to the lattice. Derivations and measurements:
//! `engine/docs/generated-vehicles-lessons.md` §25; the evidence: `engine/docs/sedan-front-audit.md`.

use crate::mechanical::MechPart;
use crate::physics::CRASH_RATE_SPEED;
use crate::structure::{Member, Section};
use crate::thin_wall::crush;

/// [`crate::physics::FrontRules::subframe_mounts_off_crash_box`]: the front subframe's mount
/// groups no longer list the crash boxes' rear nodes (`sh_a2_d*`, `sh_a2_u*`). A real front
/// subframe bolts to the rails at its own front corners, under the rail station directly above
/// them (`sh_a4_*`, which the mounts already list); the beams from its front corners forward
/// and up to the crash boxes' rear crossed the rail's whole first bay at 157 kN each (277 kN
/// of the bay's ~580 kN, audit §1), so the rail could not fold. The mount keeps its beam count
/// (picked among the remaining nodes). A design without a `subframe_front` is unchanged.
pub fn subframe_mounts_off_crash_box(parts: &mut [MechPart]) {
    for p in parts.iter_mut().filter(|p| p.name == "subframe_front") {
        for m in &mut p.mounts {
            m.to.retain(|n| !(n.starts_with("sh_a2_d") || n.starts_with("sh_a2_u")));
        }
    }
}

/// The mean crush force at crash rate over the quasi-static one for a boxed member
/// ([`crush::dynamic_factor`] at [`CRASH_RATE_SPEED`], C its mean side, the Camry deck's
/// Cowper–Symonds constants [`crush::CS_CAMRY_DECK`]); 1 for a panel. The fold strain-rate
/// coefficient inside it (0.33 V/C) is to be verified against its source
/// (`engine/docs/section-matching.md` §1).
pub fn crash_rate_factor(m: &Member) -> f64 {
    match m.section {
        Section::Box { w, h } | Section::Hat { w, h } => crush::dynamic_factor(CRASH_RATE_SPEED, 0.5 * (w + h), crush::CS_CAMRY_DECK),
        Section::Panel => 1.0,
    }
}

/// A closure panel's sheet, for its buckling force: (gauge m, yield Pa, Young's modulus Pa,
/// skins). Engineering estimates, not definitions (the closures have none yet): drawing-quality
/// mild steel (σ_y 180 MPa, the generator's [`crate::structure::Grade::Mild`]) at 0.7 mm, the
/// usual 0.65–0.75 mm of a steel hood or fender skin. The hood counts its outer skin and its
/// inner panel (two sheets of 0.7 mm over the hood's ~1.6 m² weigh 17.6 kg against the
/// design's 17.9 kg hood); a fender is one skin.
pub const HOOD_SHEET: (f64, f64, f64, f64) = (0.0007, 180e6, 210e9, 2.0);
pub const FENDER_SHEET: (f64, f64, f64, f64) = (0.0007, 180e6, 210e9, 1.0);

/// The axial force at which a panel's sheet buckles and carries no more: per strip of sheet
/// between two of its lengthwise node lines (each line a stiffener: a crease, a flange, an
/// inner panel's rib), σ_y · t · b_e with b_e Winter's effective width of the strip
/// ([`crush::winter_effective_width`], simply supported on both long edges: AISI S100 and
/// EN 1993-1-5 §4.4), summed over the strips and the sheet's skins. `strips` are the strips'
/// widths (m). A thin wide strip keeps about b_e ≈ 1.9 t √(E/σ_y) whatever its width (45 mm at
/// 0.7 mm mild steel): a 0.35 m strip carries 5.6 kN, so a hood of four strips and two skins
/// about 45 kN. Confidence: the formula is established; the sheet's gauge, grade and the
/// strips' support (a lattice line taken as a stiffener) are estimates.
pub fn panel_buckling_force(strips: &[f64], (t, sigma_y, e, skins): (f64, f64, f64, f64)) -> f64 {
    skins * strips.iter().map(|&b| sigma_y * t * b * crush::winter_effective_width(b, t, e, sigma_y)).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_thin_strip_keeps_its_effective_width() {
        // b_e → 1.9 t √(E/σ_y) for λ ≫ 1 (Winter's ρ ≈ 1/λ − 0.22/λ²)
        let (t, s, e) = (0.0007, 180e6, 210e9);
        let f = panel_buckling_force(&[0.35], (t, s, e, 1.0));
        let be = f / (s * t);
        assert!((0.040..0.048).contains(&be), "effective width {be}");
        // two strips of half the width carry more (each keeps its own effective width)
        assert!(panel_buckling_force(&[0.175, 0.175], (t, s, e, 1.0)) > f);
        // a narrow strip is fully effective
        let n = panel_buckling_force(&[0.02], (t, s, e, 1.0));
        assert!((n - s * t * 0.02).abs() < 1e-9 * n);
    }
}

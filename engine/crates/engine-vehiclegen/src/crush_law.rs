//! A crush member's values for Rubble's crush law (`engine/docs/beam-yield-law.md`,
//! `engine_softbody::explicit_crush`), each from its section through the calculator
//! ([`crate::thin_wall`]) and the cited formulas, never fitted to a test.

use crate::structure::{Member, Section};
use crate::thin_wall::{crush, Material};

/// A member's crush curve and strengths (N, m, N·m), quasi-static: the caller scales the forces by
/// its character's multiple and the crash-rate factor, as it scales the plateau today.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemberCrush {
    /// The first-buckle peak: the effective-width squash load ([`crush::box_axial_peak`], an estimate
    /// as a crush peak).
    pub peak: f64,
    /// The folding plateau: the Abramowicz–Jones mean crush force ([`Member::crush_n`]).
    pub plateau: f64,
    /// The shortening over which the force falls from the peak to the plateau: the first fold's
    /// half-length H ([`crush::fold_half_length`] at the mean side; an estimate).
    pub drop: f64,
    /// The tensile yield: the full section's squash load σ_y·A (a wall in tension does not buckle).
    pub tension: f64,
    /// Collapse moments about the strong and weak axes ([`crush::box_bending_collapse`]: σ_y·Z with
    /// the compression flange at its effective width; an upper estimate in Kecman's spirit).
    pub moment: [f64; 2],
    /// Plastic shear of one wall in its own plane, a side wall (depth h) and a top or bottom wall
    /// (width w): τ_y·depth·t, τ_y = σ_y/√3 (von Mises; EN 1993-1-1 §6.2.6's V_pl). Thin webs past
    /// h/t ≈ 60ε buckle in shear first (EN 1993-1-5 §5), so an upper estimate for these walls.
    pub shear: [f64; 2],
}

/// The member's values; None for a panel (no closed section).
pub fn member_crush(m: &Member) -> Option<MemberCrush> {
    let (w, h) = match m.section {
        Section::Box { w, h } | Section::Hat { w, h } => (w, h),
        Section::Panel => return None,
    };
    let mat = Material { e: 210e9, yield_pa: m.grade.yield_pa(), ultimate_pa: m.grade.ultimate_pa(), flow_pa: Some(m.grade.flow_pa()), density: 7850.0 };
    let t = m.gauge;
    let (peak, _, _) = crush::box_axial_peak(w, h, t, &mat);
    let (deep, wide) = (w.max(h), w.min(h));
    let strong = crush::box_bending_collapse(wide, deep, t, &mat).ok()?.0;
    let weak = crush::box_bending_collapse(deep, wide, t, &mat).ok()?.0;
    Some(MemberCrush { peak, plateau: m.crush_n(), drop: crush::fold_half_length(t, 0.5 * (w + h)), tension: m.squash_n(), moment: [strong, weak], shear: [h, w].map(|d| m.grade.yield_pa() / 3f64.sqrt() * d * t) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::Grade;

    #[test]
    fn a_box_peaks_above_its_plateau_and_below_its_squash() {
        // the sedan's crash box: 70 × 110 mm, 1.5 mm (the Camry's crash zone is 92 × 163, 1.7)
        let m = Member { name: "cb".into(), kind: "crash_box", part: "body_shell", section: Section::Box { w: 0.07, h: 0.11 }, gauge: 0.0015, grade: Grade::Hsla, length: 0.18, sheet_area: 0.18 * 0.354, mesh: Default::default(), on_shell: true };
        let c = member_crush(&m).unwrap();
        assert!(c.plateau < c.peak && c.peak <= c.tension * 1.0001, "{c:?}");
        assert!(c.moment[0] > c.moment[1] && c.moment[1] > 0.0);
        assert!(c.drop > 0.01 && c.drop < 0.04, "fold half-length {}", c.drop);
    }
}

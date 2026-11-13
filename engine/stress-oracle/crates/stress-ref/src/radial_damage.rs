//! Radial damage (`capability::StrikeModel::Radial`): the rule many games use for an
//! impact, kept here as a comparison against the strike models that read the shot. A hit
//! breaks every bond within a radius of the point struck. The radius belongs to the weapon:
//! the shot's speed, mass and size do not enter, and neither does the bond's strength.
//!
//! The reference of stress-gpu fracture.slang `fr_strike_radial`.

use crate::math::Vec3;

/// Whether a bond whose chunks' centres are `a` and `b` breaks under a hit at `contact`
/// with damage radius `radius` (m): its midpoint is within the radius.
pub fn breaks(contact: Vec3, a: Vec3, b: Vec3, radius: f64) -> bool {
    ((a + b) * 0.5 - contact).norm() <= radius
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inside the radius breaks, outside does not, whatever else is true of the shot.
    #[test]
    fn a_bond_breaks_exactly_within_the_radius() {
        let hit = Vec3::new(1.0, 2.0, 3.0);
        let at = |d: f64| (hit + Vec3::new(d - 0.05, 0.0, 0.0), hit + Vec3::new(d + 0.05, 0.0, 0.0));
        for (d, want) in [(0.0, true), (0.49, true), (0.5, true), (0.51, false), (3.0, false)] {
            let (a, b) = at(d);
            assert_eq!(breaks(hit, a, b, 0.5), want, "a bond {d} m from the hit");
        }
    }
}

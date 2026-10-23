//! Damage readback for integrations: how much a soft body has bent, broken and
//! absorbed, from its beam states (and optionally its node positions).

use crate::layout::BEAM_YIELDED;
use crate::model::{Beam, KIND_NORMAL};
use glam::Vec3;

/// A soft body's accumulated damage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoftDamage {
    /// Beams that have yielded (permanent deformation) at least once.
    pub bent: u32,
    pub broken: u32,
    pub beams: u32,
    /// Plastic work absorbed (J), Σ threshold · |L − L₀| over yielded normal
    /// beams: the energy the deformation took out of the crash.
    pub plastic_work: f32,
    /// Largest permanent shortening or stretch of a beam, |L/L₀ − 1|.
    pub max_beam_strain: f32,
    /// Per node: distance from its rest position after removing the best-fit
    /// rigid motion of the nodes still attached (m); detached nodes report how
    /// far they have gone. Empty unless positions were given.
    pub node_deformation: Vec<f32>,
}

/// Summarise `beams` (a readback, e.g. `WgpuSim::read_soft_beams`). With `rest`
/// and `now` node positions (and which nodes are detached, if known), also the
/// per-node deformation.
pub fn damage_summary(
    beams: &[Beam],
    positions: Option<(&[Vec3], &[Vec3])>,
    detached: Option<&[bool]>,
) -> SoftDamage {
    let mut d = SoftDamage {
        beams: beams.len() as u32,
        ..Default::default()
    };
    for b in beams {
        if b.broken() {
            d.broken += 1;
        }
        if b.flags & BEAM_YIELDED != 0 {
            d.bent += 1;
        }
        if b.kind == KIND_NORMAL && b.rest0 > 0.0 {
            let strain = (b.rest / b.rest0 - 1.0).abs();
            d.max_beam_strain = d.max_beam_strain.max(strain);
            if b.flags & BEAM_YIELDED != 0 {
                // Yield happened at the threshold on the side it moved.
                let threshold = if b.rest < b.rest0 {
                    b.deform_c
                } else {
                    b.deform_t
                };
                if threshold.is_finite() {
                    d.plastic_work += threshold * (b.rest - b.rest0).abs();
                }
            }
        }
    }
    if let Some((rest, now)) = positions {
        let keep = |i: usize| detached.is_none_or(|m| !m.get(i).copied().unwrap_or(false));
        let (fr, fn_): (Vec<Vec3>, Vec<Vec3>) = (0..rest.len().min(now.len()))
            .filter(|&i| keep(i))
            .map(|i| (rest[i], now[i]))
            .unzip();
        let (r, t) = crate::calibrate::rigid_fit(&fr, &fn_);
        d.node_deformation = rest
            .iter()
            .zip(now)
            .map(|(a, b)| (*b - (r * *a + t)).length())
            .collect();
    }
    d
}

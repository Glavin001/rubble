//! Soft-body arena (`gSoftData`, binding 32, `float4` slots) shared with
//! `slang/kernels/soft_step.slang`. Every constant here has a twin there.
//!
//! ```text
//! slot 0..7              header (counts, bases, flags, solver settings)
//! bodyRecBase + 2·s      soft body s: [firstBody, particles, firstBeam, beams], [steer, …]
//! beamBase + 4·e         beam e: 16 words (BEAM_* below)
//! groupBase + g          break group g: x ≠ 0 once triggered this step
//! colorBase + c          colour c: [listStart, listCount, 0, 0]
//! listBase + i           list entry: [body, adjStart, adjCount, softBody]
//! adjBase + i/4          adjacency: beam ids, 4 per slot
//! chebBase + 3·i         list entry i: Chebyshev x⁽ⁿ⁻²⁾, x⁽ⁿ⁻¹⁾, last step's velocity
//! ```

use serde::{Deserialize, Serialize};

pub const HEADER_SLOTS: u32 = 8;
pub const BODY_STRIDE: u32 = 2;
pub const BEAM_STRIDE: u32 = 4;

/// Beam record words (u32/f32 view of its 4 slots).
pub const BW_A: usize = 0;
pub const BW_B: usize = 1;
pub const BW_REST: usize = 2;
pub const BW_REST0: usize = 3;
pub const BW_SPRING: usize = 4;
pub const BW_DAMP: usize = 5;
/// Compression yield threshold (RoR maxposstress), N.
pub const BW_DEFORM_C: usize = 6;
pub const BW_STRENGTH: usize = 7;
/// Plastic coefficient (normal beams) or hydro factor (hydros).
pub const BW_COEF: usize = 8;
pub const BW_SHORT: usize = 9;
pub const BW_LONG: usize = 10;
/// type (bits 0..3) | flags (4..7) | soft body (8..15) | break group (16..31).
pub const BW_META: usize = 11;
/// AVBD penalty (ramps toward BW_SPRING).
pub const BW_PENALTY: usize = 12;
/// AVBD multiplier (hard beams).
pub const BW_LAMBDA: usize = 13;
/// Tension yield threshold (RoR |maxnegstress|), N.
pub const BW_DEFORM_T: usize = 14;
/// Signed axial force at the end of the last step (+ = compression), for readout.
pub const BW_FORCE: usize = 15;

pub const BEAM_BROKEN: u32 = 1 << 4;
pub const BEAM_HARD: u32 = 1 << 5;
pub const BEAM_INVISIBLE: u32 = 1 << 6;
/// Plastic deformation happened at least once (render tint).
pub const BEAM_YIELDED: u32 = 1 << 7;

/// Header flags (slot 2 .w).
pub const SOFT_PLASTICITY: u32 = 1;
pub const SOFT_BREAKING: u32 = 2;
pub const SOFT_SLEEP_SYNC: u32 = 4;

/// Body flag marking a particle (3-DOF node) in `gBodies`.
pub const BODY_PARTICLE: u32 = 64;

/// Event type written to the shared event ring when a beam breaks:
/// [EVENT_BEAM_BREAK, beam, |force| bits, soft body].
pub const EVENT_BEAM_BREAK: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftCapacity {
    pub max_bodies: u32,
    pub max_beams: u32,
    /// Particles across all soft bodies (colour-list entries).
    pub max_particles: u32,
    pub max_colors: u32,
    pub max_groups: u32,
}

impl Default for SoftCapacity {
    fn default() -> Self {
        Self { max_bodies: 64, max_beams: 65536, max_particles: 16384, max_colors: 64, max_groups: 4096 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftLayout {
    pub body_base: u32,
    pub beam_base: u32,
    pub group_base: u32,
    pub color_base: u32,
    pub list_base: u32,
    pub adj_base: u32,
    pub cheb_base: u32,
    pub total_slots: u32,
}

impl SoftCapacity {
    pub fn layout(&self) -> SoftLayout {
        let body_base = HEADER_SLOTS;
        let beam_base = body_base + BODY_STRIDE * self.max_bodies;
        let group_base = beam_base + BEAM_STRIDE * self.max_beams;
        let color_base = group_base + self.max_groups;
        let list_base = color_base + self.max_colors;
        let adj_base = list_base + self.max_particles;
        // Every beam appears in the adjacency of both of its particles.
        let adj_slots = (2 * self.max_beams).div_ceil(4);
        let cheb_base = adj_base + adj_slots;
        SoftLayout {
            body_base,
            beam_base,
            group_base,
            color_base,
            list_base,
            adj_base,
            cheb_base,
            total_slots: cheb_base + 3 * self.max_particles,
        }
    }

    pub fn bytes(&self) -> u64 {
        u64::from(self.layout().total_slots) * 16
    }
}

/// Global soft-solver settings (header slots 3 and 4).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftSettings {
    /// Scale on imported damping (backward Euler already dissipates).
    pub damping_scale: f32,
    /// AVBD finite-stiffness ramp k ← min(k*, k + β|C|) (N/m²).
    pub beta: f32,
    /// Penalty at spawn as a fraction of k*.
    pub ramp_start: f32,
    /// Shortest rest length plasticity may produce (RoR floors at 0.1 m).
    pub min_length: f32,
    /// Beams with k*/(m/h²) above this become hard AL constraints (0 = never).
    pub hard_ratio: f32,
    pub plasticity: bool,
    pub breaking: bool,
    /// Sleep and wake each soft body as a whole.
    pub sleep_sync: bool,
    /// Scale on every deform and break threshold (calibration knob for the
    /// lower peak forces of a coarse step).
    pub threshold_scale: f32,
    /// Chebyshev acceleration spectral radius ρ for the particle sweep (0 = off).
    pub chebyshev_rho: f32,
}

impl Default for SoftSettings {
    fn default() -> Self {
        Self {
            damping_scale: 1.0,
            beta: 1.0e9,
            ramp_start: 1.0,
            min_length: 0.1,
            hard_ratio: 0.0,
            plasticity: true,
            breaking: true,
            sleep_sync: true,
            threshold_scale: 1.0,
            chebyshev_rho: 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_regions_do_not_overlap() {
        let c = SoftCapacity::default();
        let l = c.layout();
        assert!(l.body_base >= HEADER_SLOTS);
        assert_eq!(l.beam_base - l.body_base, 2 * c.max_bodies);
        assert_eq!(l.group_base - l.beam_base, 4 * c.max_beams);
        assert_eq!(l.adj_base + (2 * c.max_beams).div_ceil(4), l.cheb_base);
        assert_eq!(l.cheb_base + 3 * c.max_particles, l.total_slots);
        // The default pool stays well under WebGPU's 128 MiB binding limit.
        assert!(c.bytes() < 128 << 20);
    }
}

/// Solver parameters for scenes with soft bodies at `hz` steps per second.
///
/// The only change from the rigid defaults besides the step is the contact
/// penalty ramp `beta_lin`: a light node (a few kg) needs its contact stiffness
/// to reach its own m/h² within an iteration or two, or it sinks and is pushed
/// back out like a spring. At the rigid default (1e4 N/m²) a dropped lattice
/// bounces back to most of its height; at 1e7 it lands and settles.
pub fn soft_scene_params(hz: f32, iterations: u32) -> engine_schema::AvbdParams {
    engine_schema::AvbdParams {
        dt: 1.0 / hz,
        iterations,
        gravity_y: -9.81,
        beta_lin: 1.0e7,
        ..Default::default()
    }
}

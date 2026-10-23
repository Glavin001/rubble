//! Soft-body arena (`gSoftData`, binding 32, `float4` slots) shared with
//! `slang/kernels/soft_step.slang`. Every constant here has a twin there.
//!
//! ```text
//! slot 0..7              header (counts, bases, flags, solver settings; see SoftHost::write_header)
//! bodyBase + 3·s         soft body s: [firstBody, particles, firstBeam, beams],
//!                        [steer, group, core body, ordinal base], [firstTri, tris, δ, 0]
//! beamBase + 4·e         beam e: 16 words (BW_* below)
//! groupBase + g          break group g: x ≠ 0 once triggered this step
//! colorBase + c          colour c: [listStart, listCount, 0, 0]
//! listBase + i           list entry: [body, adjStart, beams | tris << 16, softBody | flags]
//! adjBase + i/4          adjacency: beam ids, then triangle ids, 4 per slot
//! chebBase + 3·i         list entry i: Chebyshev x⁽ⁿ⁻²⁾, x⁽ⁿ⁻¹⁾, last step's velocity
//! triBase + t            triangle t: [a, b, c, softBody | collide << 8]
//! triContactBase + 15·t  triangle t's contacts this step: obstacle, then four nodes
//! nodeSlotBase + 3·o     particle ordinal o: the triangle it touches this step
//! flagBase + o           particle ordinal o: x ≠ 0 when in a soft contact this step
//! obstacleBase + k       thin obstacle k (static sphere / capsule): [collider, 0, 0, 0]
//! aabbBase + 2·s         soft body s bounds this step: min, max
//! ```

use serde::{Deserialize, Serialize};

pub const HEADER_SLOTS: u32 = 10;
/// Slots per soft body in the pair list (a count and up to 31 bodies).
pub const SOFT_PAIR_STRIDE: u32 = 8;
/// Soft body record: [firstBody, particles, firstBeam, beams], [steer, group,
/// core, ordinal base], [firstTri, tris, δ, 0], then rigid mode: [mode, quiet
/// steps, flags | core joint count << 8, first core joint], the core's soft-mode
/// [1/m, 1/I xyz] and rigid-mode [1/m, 1/I xyz] (f32).
pub const BODY_STRIDE: u32 = 6;
pub const BEAM_STRIDE: u32 = 4;
/// Node contacts a triangle records (its side of node–triangle contacts).
pub const TRI_NODE_CONTACTS: u32 = 4;
pub const TRI_CONTACT_STRIDE: u32 = 3 + 3 * TRI_NODE_CONTACTS;
/// Node contact slot stride (normal + triangle, barycentrics + δ, initial gap).
pub const NODE_SLOT_STRIDE: u32 = 3;

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
/// Quiet soft bodies ride their core rigidly (`SoftSettings::rigid_mode`).
pub const SOFT_RIGID_MODE: u32 = 16;
/// Particles integrate with BDF2 (`SoftSettings::bdf2`).
pub const SOFT_BDF2: u32 = 32;
pub const SOFT_BREAKING: u32 = 2;
pub const SOFT_SLEEP_SYNC: u32 = 4;
/// Node–triangle and triangle–obstacle contacts.
pub const SOFT_CONTACTS: u32 = 8;

/// List entry flags (word w, above the soft body index).
pub const ENTRY_CONTACTER: u32 = 1 << 16;

/// Body flag marking a particle (3-DOF node) in `gBodies`.
pub const BODY_PARTICLE: u32 = 64;

/// "No index" marker in `gSoftData` slots (FLT_MAX bits: an all-ones NaN pattern
/// cannot be a WGSL constant).
pub const SOFT_NONE: u32 = 0x7F7F_FFFF;

/// Event type written to the shared event ring when a beam breaks:
/// [EVENT_BEAM_BREAK, beam, |force| bits, soft body].
pub const EVENT_BEAM_BREAK: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftCapacity {
    pub max_bodies: u32,
    pub max_beams: u32,
    /// Particles across all soft bodies.
    pub max_particles: u32,
    pub max_colors: u32,
    pub max_groups: u32,
    pub max_triangles: u32,
    /// Thin static obstacles (spheres, capsules) tested against soft triangles.
    pub max_obstacles: u32,
}

impl Default for SoftCapacity {
    fn default() -> Self {
        Self {
            max_bodies: 64,
            max_beams: 65536,
            max_particles: 16384,
            max_colors: 64,
            max_groups: 4096,
            max_triangles: 32768,
            max_obstacles: 64,
        }
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
    pub tri_base: u32,
    /// Per particle (by ordinal), 2 slots: its position in its core's frame
    /// (captured each step while the body is soft, followed while it is rigid),
    /// and its rest position (desc frame: the self-collision filter).
    pub rigid_base: u32,
    /// Rest edge lengths per triangle (ab, bc, ca as f32): a triangle stretched
    /// past `TRI_TEAR` × rest is torn and stops colliding.
    pub tri_rest_base: u32,
    pub tri_contact_base: u32,
    pub node_slot_base: u32,
    pub flag_base: u32,
    pub obstacle_base: u32,
    pub aabb_base: u32,
    /// Per soft body, its share of every colour list: `(start, count)` pairs, two
    /// per slot, `body_color_stride` slots per body (one workgroup per body).
    pub body_color_base: u32,
    pub body_color_stride: u32,
    /// Per triangle, 2 slots written each step before node contacts: (unit normal,
    /// area; 0 = torn or degenerate) and (first vertex, cull radius).
    pub tri_plane_base: u32,
    /// Per soft body, `SOFT_PAIR_STRIDE` slots written each step by `soft_pairs`:
    /// a count, then the soft bodies (itself included, ascending) whose bounds
    /// come within reach of its own (count `u32::MAX`: too many, walk them all).
    pub body_pair_base: u32,
    /// Per particle (ordinal): its position one step back (w = 1 when valid),
    /// for BDF2.
    pub bdf_base: u32,
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
        // Every beam appears in the adjacency of both particles, every triangle in three.
        let adj_slots = (2 * self.max_beams + 3 * self.max_triangles).div_ceil(4);
        let cheb_base = adj_base + adj_slots;
        let tri_base = cheb_base + 3 * self.max_particles;
        let tri_rest_base = tri_base + self.max_triangles;
        let tri_contact_base = tri_rest_base + self.max_triangles;
        let node_slot_base = tri_contact_base + TRI_CONTACT_STRIDE * self.max_triangles;
        let flag_base = node_slot_base + NODE_SLOT_STRIDE * self.max_particles;
        let obstacle_base = flag_base + self.max_particles;
        let aabb_base = obstacle_base + self.max_obstacles;
        SoftLayout {
            body_base,
            beam_base,
            group_base,
            color_base,
            list_base,
            adj_base,
            cheb_base,
            tri_base,
            tri_rest_base,
            tri_contact_base,
            node_slot_base,
            flag_base,
            obstacle_base,
            aabb_base,
            rigid_base: aabb_base + 2 * self.max_bodies,
            body_color_base: aabb_base + 2 * self.max_bodies + 2 * self.max_particles,
            body_color_stride: self.max_colors.div_ceil(2),
            tri_plane_base: aabb_base
                + 2 * self.max_bodies
                + 2 * self.max_particles
                + self.max_bodies * self.max_colors.div_ceil(2),
            body_pair_base: aabb_base
                + 2 * self.max_bodies
                + 2 * self.max_particles
                + self.max_bodies * self.max_colors.div_ceil(2)
                + 2 * self.max_triangles,
            bdf_base: aabb_base
                + 2 * self.max_bodies
                + 2 * self.max_particles
                + self.max_bodies * self.max_colors.div_ceil(2)
                + 2 * self.max_triangles
                + SOFT_PAIR_STRIDE * self.max_bodies,
            total_slots: aabb_base
                + 2 * self.max_bodies
                + 2 * self.max_particles
                + self.max_bodies * self.max_colors.div_ceil(2)
                + 2 * self.max_triangles
                + SOFT_PAIR_STRIDE * self.max_bodies
                + self.max_particles,
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
    pub plasticity: bool,
    pub breaking: bool,
    /// Sleep and wake each soft body as a whole.
    pub sleep_sync: bool,
    /// Scale on every deform and break threshold (calibration knob for the
    /// lower peak forces of a coarse step).
    pub threshold_scale: f32,
    /// Chebyshev acceleration spectral radius ρ for the particle sweep (0 = off).
    pub chebyshev_rho: f32,
    /// Node–triangle and triangle–obstacle contacts.
    pub contacts: bool,
    /// Penalty stiffness of soft contacts (N/m) and their friction coefficient.
    pub contact_stiffness: f32,
    pub contact_friction: f32,
    /// Rigid mode (design: "rigid mode with promotion"): a soft body with a rigid
    /// core that has had no contact and no plastic yield for `rigid_quiet` s
    /// rides its core rigidly (its particles follow the core's pose, the core
    /// carries the whole mass and inertia) and skips the particle solve. It turns
    /// soft again, within the same step, on any contact of its particles or when
    /// its bounds near another soft body or an obstacle.
    pub rigid_mode: bool,
    pub rigid_quiet: f32,
    /// Particle sweeps per solver iteration (1 = one per iteration). The soft
    /// bodies couple to the rigid passes only through their core joints, so
    /// extra sweeps converge them at about half the cost of extra iterations.
    pub sweeps: u32,
    /// Second-order BDF2 for the particles instead of backward Euler (BDF1):
    /// far less numerical damping of the structure's vibration at a given step
    /// (experimental; validate with `soft_calibrate`). Backward Euler wherever a
    /// particle has no previous step (spawn, wake).
    pub bdf2: bool,
}

impl Default for SoftSettings {
    fn default() -> Self {
        Self {
            damping_scale: 1.0,
            beta: 1.0e9,
            ramp_start: 1.0,
            min_length: 0.1,
            plasticity: true,
            breaking: true,
            sleep_sync: true,
            threshold_scale: 1.0,
            chebyshev_rho: 0.9,
            contacts: true,
            contact_stiffness: 1.0e7,
            contact_friction: 0.5,
            sweeps: 1,
            rigid_mode: false,
            rigid_quiet: 0.5,
            bdf2: false,
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
        assert_eq!(l.beam_base - l.body_base, BODY_STRIDE * c.max_bodies);
        assert_eq!(l.group_base - l.beam_base, 4 * c.max_beams);
        assert_eq!(
            l.adj_base + (2 * c.max_beams + 3 * c.max_triangles).div_ceil(4),
            l.cheb_base
        );
        assert_eq!(l.rigid_base + 2 * c.max_particles, l.body_color_base);
        assert_eq!(
            l.body_color_base + c.max_bodies * l.body_color_stride,
            l.tri_plane_base
        );
        assert_eq!(l.tri_plane_base + 2 * c.max_triangles, l.body_pair_base);
        assert_eq!(
            l.body_pair_base + SOFT_PAIR_STRIDE * c.max_bodies,
            l.bdf_base
        );
        assert_eq!(l.bdf_base + c.max_particles, l.total_slots);
        // The default pool stays well under WebGPU's 128 MiB binding limit.
        assert!(c.bytes() < 128 << 20);
    }
}

impl SoftSettings {
    /// Settings with the deform/break threshold scale fitted for a step rate by
    /// `soft_calibrate` (60 km/h wall crush of the procedural car against a
    /// 960 Hz × 30-iteration reference). With yield applied inside the solve the
    /// crush no longer depends on the step at 240 Hz and finer (fitted 1.08 at
    /// 240 Hz: kept at 1). 120 Hz × 10 is under-converged, crushes deeper and
    /// needs thresholds × 1.44. Interpolated in log(h).
    pub fn calibrated_for(hz: f32) -> Self {
        const FIT: [(f32, f32); 2] = [(120.0, 1.44), (240.0, 1.0)];
        let scale = if hz <= FIT[0].0 {
            FIT[0].1
        } else if hz >= FIT[1].0 {
            FIT[1].1
        } else {
            let t = (hz.ln() - FIT[0].0.ln()) / (FIT[1].0.ln() - FIT[0].0.ln());
            FIT[0].1 + t * (FIT[1].1 - FIT[0].1)
        };
        Self {
            threshold_scale: scale,
            ..Default::default()
        }
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

//! AVBD numerical mode `avbd_giles` — official 3D demo defaults.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// `AvbdParams::flags` bit: three-avbd reuseContacts (SAT-skip for still pairs).
pub const FLAG_REUSE_CONTACTS: u32 = 1;

/// `AvbdParams::flags` bit: contacts target zero gap. `contact_offset` only
/// widens detection (speculative contacts); without it resting bodies sit about
/// `contact_offset` deep (three-avbd).
pub const FLAG_CONTACT_GAP: u32 = 32;

/// `AvbdParams::flags` bit: the primal inertial block uses the world inertia
/// `R·diag(I)·Rᵀ`. Without it (three-avbd) principal moments act on world axes.
pub const FLAG_WORLD_INERTIA: u32 = 64;

/// Giles/Diaz/Yuksel AVBD parameters (avbd-demo3d `Solver::defaultParams`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct AvbdParams {
    pub dt: f32,
    pub gravity_y: f32,
    pub iterations: u32,
    /// Bit 0: reuseContacts (three-avbd SAT-skip for still pairs). Off by default —
    /// on Metal it slows Brick Ring smash; enable for Box Pile settle.
    pub flags: u32,
    /// Contact / constraint stabilization (paper α). Official demo: 0.99.
    pub alpha: f32,
    /// Warm-start decay. Official demo: 0.999.
    pub gamma: f32,
    pub beta_lin: f32,
    pub beta_ang: f32,
    pub k_start: f32,
    pub k_max_n: f32,
    pub k_max_t: f32,
    pub contact_offset: f32,
    pub sleep_lin: f32,
    pub sleep_ang: f32,
    pub sleep_time: f32,
    pub residual_exit: f32,
    pub trust_lin: f32,
    pub trust_ang: f32,
    pub pivot_floor: f32,
    /// Broadphase grid cell size (three-avbd ≈ 2·maxSmallRadius). Host refreshes each step.
    pub grid_cell_size: f32,
}

impl Default for AvbdParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            gravity_y: -10.0,
            iterations: 10,
            flags: 0,
            alpha: 0.99,
            gamma: 0.999,
            beta_lin: 10_000.0,
            beta_ang: 100.0,
            k_start: 1_000.0,
            k_max_n: 1.0e10,
            k_max_t: 1.0e6,
            contact_offset: 0.01,
            // Slightly looser than demo3d's tiny thresholds: Jacobi residual jitter on
            // tall stacks never drops below 0.05, so bodies never slept and KE pumped.
            sleep_lin: 0.15,
            sleep_ang: 0.15,
            // Island sleep gate protects stacks; 0.5s is enough once candidates agree.
            sleep_time: 0.5,
            residual_exit: 1.0e-4,
            trust_lin: 0.35,
            trust_ang: 0.5,
            pivot_floor: 1.0e-6,
            grid_cell_size: 2.0,
        }
    }
}

/// Capacity and high-water configuration. Overflow fails the step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldCapacity {
    pub max_bodies: u32,
    pub max_colliders: u32,
    pub max_contacts: u32,
    pub max_joints: u32,
    pub max_candidates: u32,
    pub max_colors: u32,
    pub max_commands: u32,
    pub max_hull_vertices: u32,
}

impl Default for WorldCapacity {
    fn default() -> Self {
        Self {
            max_bodies: 131_072,
            max_colliders: 262_144,
            max_contacts: 1_048_576,
            max_joints: 262_144,
            max_candidates: 2_097_152,
            max_colors: 64,
            max_commands: 65_536,
            max_hull_vertices: 1_048_576,
        }
    }
}

impl WorldCapacity {
    pub fn small() -> Self {
        Self {
            max_bodies: 4096,
            max_colliders: 8192,
            max_contacts: 32_768,
            max_joints: 8192,
            max_candidates: 65_536,
            max_colors: 32,
            max_commands: 4096,
            max_hull_vertices: 65_536,
        }
    }

    pub fn scale() -> Self {
        Self {
            max_bodies: 524_288,
            max_colliders: 1_048_576,
            max_contacts: 4_194_304,
            max_joints: 524_288,
            max_candidates: 8_388_608,
            max_colors: 64,
            max_commands: 131_072,
            max_hull_vertices: 2_097_152,
        }
    }
}

//! GPU records of the MPM kernels (`slang/kernels/terrain_mpm.slang`), byte for
//! byte.

use bytemuck::{Pod, Zeroable};

/// Bindings of the terrain buffers, after the rigid engine's and the render-pose
/// buffers.
pub const MPM_PARAMS_BINDING: usize = 37;
pub const MPM_COUNTERS_BINDING: usize = 38;
pub const MPM_PARTICLES_BINDING: usize = 39;
pub const MPM_PAYLOAD_BINDING: usize = 40;
pub const MPM_HASH_BINDING: usize = 41;
pub const MPM_BLOCKS_BINDING: usize = 42;
pub const MPM_CELLS_BINDING: usize = 43;
pub const MPM_SORTED_BINDING: usize = 44;
pub const MPM_GRID_BINDING: usize = 45;
/// Indirect workgroup counts of the per-block kernels.
pub const MPM_ARGS_BINDING: usize = 46;
/// One past the last terrain binding.
pub const MPM_BINDING_END: usize = 47;

/// `MpmParams.dims.w` without a bed heightfield.
pub const MPM_NO_BED: u32 = u32::MAX;
/// `MpmParams.counts.w`: no stress (transfers only).
pub const MPM_MATERIAL_NONE: u32 = 0;
/// `MpmParams.counts.w`: Drucker–Prager sand.
pub const MPM_MATERIAL_SAND: u32 = 1;

/// Counter flags (`gMpmCounters[1]`).
pub const MPM_OVERFLOW_BLOCKS: u32 = 1;
pub const MPM_OVERFLOW_HASH: u32 = 2;
pub const MPM_OUTSIDE: u32 = 4;
pub const MPM_CFL: u32 = 8;
pub const MPM_INVALID: u32 = 16;

/// Block coordinates are packed in 10 bits per axis.
pub const MPM_MAX_BLOCKS_PER_AXIS: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MpmParams {
    /// Region origin, cell size.
    pub origin: [f32; 4],
    /// dt, gravity along y, node mass threshold, bed friction μ.
    pub step: [f32; 4],
    /// Particles, hash slots (a power of two), max blocks, material kind.
    pub counts: [u32; 4],
    /// Region size in cells, bed heightfield collider (or [`MPM_NO_BED`]).
    pub dims: [u32; 4],
    /// x: largest particle speed the substep's dt allows (CFL); y, z, w unused.
    pub limits: [f32; 4],
    /// Sand: λ, μ, α (friction), cohesion (log strain).
    pub sand: [f32; 4],
}

/// One particle (128 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MpmParticle {
    /// Position, mass.
    pub xm: [f32; 4],
    /// Velocity, log Jp (plastic volume).
    pub vj: [f32; 4],
    /// Rows of C, the APIC affine velocity.
    pub c: [[f32; 4]; 3],
    /// Rows of the deformation gradient F; `f[0][3]` is the rest volume.
    pub f: [[f32; 4]; 3],
}

impl MpmParticle {
    /// At rest, undeformed: `volume` m³ of material of `density` kg/m³ at `x`.
    pub fn at_rest(x: [f32; 3], volume: f32, density: f32) -> Self {
        let mut f = [[0.0; 4]; 3];
        for (k, row) in f.iter_mut().enumerate() {
            row[k] = 1.0;
        }
        f[0][3] = volume;
        Self {
            xm: [x[0], x[1], x[2], volume * density],
            vj: [0.0; 4],
            c: [[0.0; 4]; 3],
            f,
        }
    }
}

/// Words of the per-substep scratch buffers for `max_blocks` blocks.
pub fn hash_slots(max_blocks: u32) -> u32 {
    (2 * max_blocks).max(64).next_power_of_two()
}

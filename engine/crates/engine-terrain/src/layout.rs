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
/// Per heightfield sample under the region: quanta, particles, first particle, baked quanta.
pub const MPM_COLUMNS_BINDING: usize = 47;
/// Coupled colliders and the region's block cells listing them.
pub const MPM_COUPLE_BINDING: usize = 48;
/// Per coupling slot: the frame's reaction on the body and its proxy velocity;
/// after the slots, the absorbing boundary's nodes (displacement, the column's
/// surface height above them).
pub const MPM_COUPLED_BINDING: usize = 49;
/// Ejecta parcels: (position, quanta), (velocity, mass; 0 once landed).
pub const MPM_PARCELS_BINDING: usize = 50;
/// Per heightfield sample: quanta deposited this frame.
pub const MPM_DEPOSIT_BINDING: usize = 51;
/// One past the last terrain binding.
/// Terrain look-ahead (`terrain_watch.slang`): parameters, then the GPU's pick.
pub const MPM_WATCH_BINDING: usize = 52;
pub const MPM_BINDING_END: usize = 53;
/// Colliders one block cell of the region lists (`MPM_CELL_COLLIDERS`): 256, so
/// a vehicle of many collider hulls (the dune buggy has 187) can drive into a
/// region whose block cells are a few metres across. The coupling buffer holds
/// 1 + 256 words per block (8.4 MB at 8192 blocks); a node only visits the
/// colliders its cell lists.
pub const MPM_CELL_COLLIDERS: u32 = 256;

/// Counter words (`gMpmCounters`).
pub const MPM_C_BLOCKS: usize = 0;
pub const MPM_C_FLAGS: usize = 1;
pub const MPM_C_MAX_SPEED: usize = 2;
pub const MPM_C_PARTICLES: usize = 3;
pub const MPM_C_PARCELS: usize = 5;
pub const MPM_C_DEPOSITED: usize = 6;
pub const MPM_C_ESCAPED: usize = 7;
pub const MPM_C_AIRBORNE: usize = 9;
pub const MPM_C_BODY_SPEED: usize = 10;
pub const MPM_COUNTER_WORDS: usize = 12;

/// `MpmParams.dims.w` without a bed heightfield.
pub const MPM_NO_BED: u32 = u32::MAX;
/// `MpmParams.counts.w`: no stress (transfers only).
pub const MPM_MATERIAL_NONE: u32 = 0;
/// `MpmParams.counts.w`: Drucker–Prager sand.
pub const MPM_MATERIAL_SAND: u32 = 1;
/// `MpmParams.counts.w`: Mohr–Coulomb sand.
pub const MPM_MATERIAL_SAND_MC: u32 = 2;
/// `MpmParams.counts.w`: cohesive Mohr–Coulomb soil (cohesion in stress space).
pub const MPM_MATERIAL_SOIL_MC: u32 = 3;
/// `MpmParams.counts.w`: damage-softening rock.
pub const MPM_MATERIAL_ROCK: u32 = 4;

/// Counter flags (`gMpmCounters[1]`).
pub const MPM_OVERFLOW_BLOCKS: u32 = 1;
pub const MPM_OVERFLOW_HASH: u32 = 2;
pub const MPM_OUTSIDE: u32 = 4;
pub const MPM_CFL: u32 = 8;
pub const MPM_INVALID: u32 = 16;
pub const MPM_OVERFLOW_PARTICLES: u32 = 32;
pub const MPM_OVERFLOW_COUPLED: u32 = 64;
pub const MPM_OVERFLOW_CELL: u32 = 128;
pub const MPM_OVERFLOW_IMPULSES: u32 = 256;
pub const MPM_OVERFLOW_PARCELS: u32 = 512;
pub const MPM_OVERFLOW_FIELD: u32 = 1024;

/// `MpmParams::boundary[0]` bits (u32): the bed and walls absorb
/// ([`crate::RegionBoundary::Absorbing`]) ...
pub const MPM_BOUNDARY_ABSORBING: u32 = 1;
/// ... the first step after activation gives each particle the geostatic F
/// of its depth below its home column's surface (`mpm_geostatic`).
pub const MPM_BOUNDARY_GEOSTATIC: u32 = 2;

/// `MpmParticle::f[2][3]` bit: the particle left the region as a parcel.
pub const MPM_PARTICLE_DEAD: u32 = 1;
/// `MpmParticle::f[2][3]` bit: the particle was made by activation, and
/// `c[1][3]`, `c[2][3]` hold the heightfield-local x and z it was emitted at.
/// The bake splats it from its home sample moved by its displacement since.
pub const MPM_PARTICLE_HOME: u32 = 2;

/// One ejecta parcel: position, quanta (u32 bits), velocity, mass (0 once landed).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MpmParcel {
    pub xq: [f32; 4],
    pub vm: [f32; 4],
}

/// Block coordinates are packed in 10 bits per axis.
pub const MPM_MAX_BLOCKS_PER_AXIS: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MpmParams {
    /// Region origin, cell size.
    pub origin: [f32; 4],
    /// dt, gravity along y, node mass threshold, bed friction μ.
    pub step: [f32; 4],
    /// Max particles, hash slots (a power of two), max blocks, material kind.
    pub counts: [u32; 4],
    /// Region size in cells, bed heightfield collider (or [`MPM_NO_BED`]).
    pub dims: [u32; 4],
    /// x: largest particle speed the substep's dt allows (CFL); y: the bake's
    /// home fade ([`crate::TerrainConfig::bake_home_fade`], lattice spacings);
    /// z, w unused.
    pub limits: [f32; 4],
    /// The material's words: λ, μ, then for sand and soil friction
    /// (Drucker–Prager α, or sin φ for Mohr–Coulomb) and cohesion (the apex
    /// stretch c·cot φ/(3λ + 2μ) for Drucker–Prager, 0 for dry Mohr–Coulomb
    /// sand, c·cos φ (Pa) for cohesive soil); for rock the tensile strength f_t
    /// (Pa) and `RockMaterial::band_stress`, E·G_f/(2f_t) (Pa·m).
    pub sand: [f32; 4],
    /// Heightfield samples under the region: i0, j0, ni, nj.
    pub window: [u32; 4],
    /// Bed height (heightfield-local y), volume quantum (m³), particle lattice
    /// spacing (m), density (kg/m³).
    pub column: [f32; 4],
    /// Frame duration (s; pads collider bounds by their motion), max coupled
    /// colliders, max parcels (0: leaving the region is an error), heightfield
    /// samples parcels may deposit on (the last three u32 bits).
    pub couple: [f32; 4],
    /// Z-model source: surface point (world), Z; flow-centre depth, core
    /// radius, strength α, extent (f32::MAX: unbounded).
    pub source: [[f32; 4]; 2],
    /// The region's boundary (u32 bits): [`MPM_BOUNDARY_ABSORBING`] |
    /// [`MPM_BOUNDARY_GEOSTATIC`], the first absorbing-node record in
    /// [`MPM_COUPLED_BINDING`], the records the region uses, unused.
    pub boundary: [u32; 4],
}

/// One particle (128 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MpmParticle {
    /// Position, mass.
    pub xm: [f32; 4],
    /// Velocity, log Jp (plastic volume).
    pub vj: [f32; 4],
    /// Rows of C, the APIC affine velocity; `c[0][3]` is the rock's damage
    /// (0 intact, 1 broken; 0 for other materials); `c[1][3]`, `c[2][3]`
    /// where activation emitted it (with [`MPM_PARTICLE_HOME`]).
    pub c: [[f32; 4]; 3],
    /// Rows of the deformation gradient F; `f[0][3]` is the rest volume,
    /// `f[1][3]` the volume quanta (u32 bits; 0 for particles loaded by the host),
    /// `f[2][3]` flags (u32 bits; [`MPM_PARTICLE_DEAD`]: left as a parcel;
    /// [`MPM_PARTICLE_HOME`]: made by activation).
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

    /// Damage of a rock particle (`c[0][3]`).
    pub fn damage(&self) -> f32 {
        self.c[0][3]
    }
}

/// Words of the per-substep scratch buffers for `max_blocks` blocks.
pub fn hash_slots(max_blocks: u32) -> u32 {
    (2 * max_blocks).max(64).next_power_of_two()
}

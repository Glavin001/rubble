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
/// Nonzero: the next substep's `mpm_payload` sets the geostatic F first.
pub const MPM_C_GEOSTATIC: usize = 11;
/// Parcels thrown before this substep's G2P: the ones `mpm_parcels` flies.
pub const MPM_C_FLYING: usize = 12;
/// The first 16 words are the frame's counters; the areas' follow.
/// The material's surface under a source point (order-preserving f32 bits),
/// found on the GPU when a source lands on ground already moving.
pub const MPM_C_SURFACE: usize = 13;
/// Wheel loads `mpm_wheel_loads` listed this frame (vehicle wheels standing
/// on an open area's sand, [`TerrainConfig::max_wheel_loads`]).
///
/// [`TerrainConfig::max_wheel_loads`]: crate::TerrainConfig::max_wheel_loads
pub const MPM_C_WHEELS: usize = 14;
/// Per area slot k: its fastest particle over the frame (f32 bits) at
/// `MPM_C_AREA_SPEED + k`, and its fastest coupled body at
/// `MPM_C_AREA_BODY + k`.
pub const MPM_C_AREA_SPEED: usize = 16;
pub const MPM_C_AREA_BODY: usize = MPM_C_AREA_SPEED + MPM_MAX_AREAS as usize;
/// Per area slot: the pool's particle count when it opened (the bake's
/// dithers count slots from it, so an area bakes as it would alone).
pub const MPM_C_AREA_FIRST: usize = MPM_C_AREA_BODY + MPM_MAX_AREAS as usize;
/// Area clocks ([`TerrainConfig::area_clocks`]): bit k set while area slot k
/// sits out the current substep pass (none without area clocks).
///
/// [`TerrainConfig::area_clocks`]: crate::TerrainConfig::area_clocks
pub const MPM_C_SKIP: usize = MPM_C_AREA_FIRST + MPM_MAX_AREAS as usize;
/// Area clocks: the frame's next pass (its record in the pass table).
pub const MPM_C_PASS: usize = MPM_C_SKIP + 1;
/// Block sleep ([`TerrainConfig::sleep_speed`]): the speed (f32 bits) under
/// which material counts as at rest; 0 with sleep off.
///
/// [`TerrainConfig::sleep_speed`]: crate::TerrainConfig::sleep_speed
pub const MPM_C_SLEEP: usize = MPM_C_PASS + 1;
/// Block sleep: the substeps at rest after which a particle sleeps.
pub const MPM_C_SLEEP_AFTER: usize = MPM_C_SLEEP + 1;
/// Block sleep: this substep's blocks below this index are its awake blocks
/// (where awake material and the wake sources reach), below the next word's
/// those and their neighbours; the rest are the second ring.
pub const MPM_C_SLEEP_RING0: usize = MPM_C_SLEEP_AFTER + 1;
pub const MPM_C_SLEEP_RING1: usize = MPM_C_SLEEP_RING0 + 1;
/// Per area slot k: the parcels in flight after the last substep whose
/// ballistic path can still reach its box (`MPM_C_AREA_AIR + k`), and its
/// fastest body touching its material this frame (`MPM_C_AREA_TOUCH + k`,
/// f32 bits): the local settle rule's words ([`crate::SettleRule::Local`]).
pub const MPM_C_AREA_AIR: usize = MPM_C_SLEEP_RING1 + 1;
pub const MPM_C_AREA_TOUCH: usize = MPM_C_AREA_AIR + MPM_MAX_AREAS as usize;
/// Pool reuse: one past the highest slot still held (`mpm_pool_top`).
pub const MPM_C_POOL_TOP: usize = MPM_C_AREA_TOUCH + MPM_MAX_AREAS as usize;
/// Pool reuse: the free slots found, then those an activation takes.
pub const MPM_C_HOLES: usize = MPM_C_POOL_TOP + 1;
/// Pool reuse: nonzero while a pass reuses slots (an activation past the
/// pool's end: its particle count).
pub const MPM_C_REUSE: usize = MPM_C_HOLES + 1;
/// Live particles: the pool's held slots are these plus the parcels in
/// flight (each comes back to the slot it left). With slots reused,
/// [`MPM_C_PARTICLES`] is the pool's extent, not its use.
pub const MPM_C_LIVE: usize = MPM_C_REUSE + 1;
pub const MPM_COUNTER_WORDS: usize = MPM_C_LIVE + 1;

/// Area slots (`TerrainConfig::max_regions` is at most this): records
/// `1..=MPM_MAX_AREAS` of [`MPM_PARAMS_BINDING`] after the domain's.
pub const MPM_MAX_AREAS: u32 = 16;
/// `MpmParams::area[3]` of the domain record: this bake empties the pool
/// (the last open area).
pub const MPM_DOMAIN_CLOSE: u32 = 1;
/// ... the source finds the material's surface under its point
/// (`mpm_source_surface`) rather than taking the point's height.
pub const MPM_DOMAIN_SURFACE: u32 = 2;
/// `MpmParams::area[3]` of the domain record, from this bit on (low x, high
/// x, low z, high z): the selected old area's side stays a wall of the area
/// growing over it (`mpm_grow_shift` keeps its give).
pub const MPM_DOMAIN_KEEP: u32 = 4;
/// Block keys pack x, y, z in 11, 10 and 11 bits, biased by these: the open
/// areas reach this many blocks either way from the domain's origin (the
/// first area's).
pub const MPM_KEY_BIAS: [i32; 3] = [1024, 512, 1024];

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
/// More wheels stood on open sand than `max_wheel_loads` (the rest pressed nothing).
pub const MPM_OVERFLOW_WHEELS: u32 = 2048;
/// float4 records per wheel load in [`MPM_COUPLED_BINDING`], after the
/// coupling slots and the absorbing face points: (the contact's offset in its
/// base cell, area slot bits), (base node, domain lattice, int bits), (force
/// on the sand N, the material's mass at the contact this substep).
pub const MPM_WHEEL_RECORDS: u32 = 3;

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
/// `MpmParticle::f[2][3]` bit: activated since the last step; the next
/// geostatic start (`mpm_geostatic`) gives it its F and clears the bit.
pub const MPM_PARTICLE_FRESH: u32 = 4;
/// `MpmParticle::f[2][3]` bit: block sleep froze it at rest
/// ([`crate::TerrainConfig::sleep_speed`]); bits 8-15 count its substeps at rest.
pub const MPM_PARTICLE_ASLEEP: u32 = 8;

/// One ejecta parcel: position, quanta (u32 bits), velocity, mass (0 once landed).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MpmParcel {
    pub xq: [f32; 4],
    pub vm: [f32; 4],
}

/// Block coordinates are packed in 11 (x, z) and 10 (y) bits per axis.
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
    /// [`MPM_COUPLED_BINDING`], the records the region uses, its first
    /// record in [`MPM_COLUMNS_BINDING`].
    pub boundary: [u32; 4],
    /// The domain record (0): area slots in use, the selected area (of an
    /// activation, a source or a bake), the areas a growing one takes over
    /// (bits), [`MPM_DOMAIN_CLOSE`] | [`MPM_DOMAIN_SURFACE`]. An area record
    /// (1 + slot): its origin's node offset from the domain's (i32 bits),
    /// its first block cell in [`MPM_COUPLE_BINDING`].
    pub area: [u32; 4],
}

/// Bytes of [`MPM_PARAMS_BINDING`]: the domain record and every area slot's.
/// With [`TerrainConfig::area_clocks`] the pass table follows them: one
/// record per substep pass of the frame (`step`: dt, the CFL speed bound, the
/// areas it steps as bits, nonzero to set the geostatic F first), at most
/// [`area_clock_passes`] of them.
///
/// [`TerrainConfig::area_clocks`]: crate::TerrainConfig::area_clocks
pub const MPM_PARAMS_BYTES: u64 =
    (1 + MPM_MAX_AREAS as u64) * std::mem::size_of::<MpmParams>() as u64;

/// The area clocks' passes a frame may hold: every open area at the frame's
/// substep limit.
pub fn area_clock_passes(config: &crate::TerrainConfig) -> u32 {
    if config.area_clocks {
        config.max_regions * config.max_substeps
    } else {
        0
    }
}

/// [`MPM_PARAMS_BINDING`]'s bytes under `config`: the records, then the pass table.
pub fn params_bytes(config: &crate::TerrainConfig) -> u64 {
    MPM_PARAMS_BYTES
        + u64::from(area_clock_passes(config)) * std::mem::size_of::<MpmParams>() as u64
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

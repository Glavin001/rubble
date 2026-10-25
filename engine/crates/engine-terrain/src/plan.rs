//! One terrain command sequence for every host: the native wgpu host
//! (`engine-webgpu`'s `WgpuSim`) and the browser (through wasm) replay the
//! same [`TerrainOps`] for activation, the Z-model source, uncoupled and
//! coupled frames of substeps, the parcels' deposits and the bake, so the two
//! cannot drift apart (as `engine_exec::rigid_plan` is for the rigid step).
//!
//! A [`TerrainPlan`] holds the host state of the terrain pool (limits, the
//! `MpmParams` record, the material's wave speed) and turns each call into
//! host writes plus the dispatches of one compute pass. Kernel names are the
//! entry points of `slang/kernels/terrain_mpm.slang` (bindings 37-51); every
//! dispatch is `threads` invocations at 64 per workgroup, and an indirect one
//! reads its workgroup counts from [`MPM_ARGS_BINDING`] at a byte offset.
//!
//! [`RegionClock`] is the open region's frame schedule: each frame's substeps
//! from a speed bound the GPU checks, and when the region has settled.

use crate::config::{TerrainConfig, TerrainError};
use crate::layout::*;
use crate::material::{RockMaterial, TerrainMaterial, YieldSurface};
use crate::region::{RegionPlan, BLOCK_CELLS};
use crate::step::substeps_for;
use crate::zmodel::ZModel;

/// Every kernel a [`TerrainPlan`] dispatches.
pub const KERNELS: &[&str] = &[
    "mpm_clear",
    "mpm_touch",
    "mpm_write_args",
    "mpm_count",
    "mpm_scan_cells",
    "mpm_scatter",
    "mpm_cell_sort",
    "mpm_payload",
    "mpm_p2g",
    "mpm_grid",
    "mpm_g2p",
    "mpm_activate_count",
    "mpm_activate_scan",
    "mpm_activate_emit",
    "mpm_activate_lower",
    "mpm_bake_bin",
    "mpm_bake_raise",
    "mpm_bake_header",
    "mpm_couple_clear",
    "mpm_couple_bin",
    "mpm_couple_gather",
    "mpm_couple_apply",
    "mpm_couple_emit",
    "mpm_couple_finish",
    "mpm_parcels",
    "mpm_deposit_raise",
    "mpm_deposit_header",
    "mpm_source",
];

/// Byte offsets of the indirect workgroup counts in [`MPM_ARGS_BINDING`].
pub const ARGS_BLOCKS: u64 = 0;
pub const ARGS_PARTICLES: u64 = 12;
pub const ARGS_PARCELS: u64 = 24;

/// Quanta per lattice particle: a column's volume is rounded to s³/64 (s the
/// particle spacing), so a sample's height moves by whole quanta.
const QUANTA_PER_LATTICE_CELL: f32 = 64.0;

/// What a run of substeps reported (the counters after it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainStepReport {
    /// Blocks the last substep used.
    pub blocks: u32,
    /// Largest particle speed over the frame's substeps (m/s).
    pub max_speed: f32,
    /// Particle slots in the region (live, or left as parcels).
    pub particles: u32,
    /// Parcels thrown so far (in flight or landed).
    pub parcels: u32,
    /// Quanta parcels have deposited on the heightfield.
    pub deposited: u32,
    /// Quanta that left the heightfield's footprint.
    pub escaped: u32,
    /// Parcels still in flight after the last substep.
    pub airborne: u32,
    /// Fastest surface speed of a dynamic body coupled to the region this
    /// frame (m/s; 0 without coupling).
    pub body_speed: f32,
}

/// A host write (`queue.write_buffer`) of `words` at byte `offset` of `binding`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainWrite {
    pub binding: usize,
    pub offset: u64,
    pub words: Vec<u32>,
}

/// One dispatch of the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainCmd {
    /// `threads` invocations, 64 per workgroup.
    Dispatch { kernel: &'static str, threads: u64 },
    /// Workgroup counts from [`MPM_ARGS_BINDING`] at byte `offset`.
    Indirect { kernel: &'static str, offset: u64 },
}

/// A call's work: the host writes first (ahead of the submission that carries
/// the commands), then the commands in order, in one compute pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainOps {
    pub writes: Vec<TerrainWrite>,
    pub cmds: Vec<TerrainCmd>,
}

impl TerrainOps {
    fn write_words(&mut self, binding: usize, offset: u64, words: &[u32]) {
        self.writes.push(TerrainWrite {
            binding,
            offset,
            words: words.to_vec(),
        });
    }

    fn counter(&mut self, word: usize, value: u32) {
        self.write_words(MPM_COUNTERS_BINDING, 4 * word as u64, &[value]);
    }

    fn params(&mut self, params: &MpmParams) {
        self.write_words(
            MPM_PARAMS_BINDING,
            0,
            bytemuck::cast_slice(std::slice::from_ref(params)),
        );
    }

    fn disp(&mut self, kernel: &'static str, threads: u64) {
        self.cmds.push(TerrainCmd::Dispatch { kernel, threads });
    }

    fn indirect(&mut self, kernel: &'static str, offset: u64) {
        self.cmds.push(TerrainCmd::Indirect { kernel, offset });
    }

    /// One workgroup (64 threads) per block in use, counted on the GPU.
    fn blocks(&mut self, kernel: &'static str) {
        self.indirect(kernel, ARGS_BLOCKS);
    }

    /// One thread per particle in the region, counted on the GPU.
    fn particles(&mut self, kernel: &'static str) {
        self.indirect(kernel, ARGS_PARTICLES);
    }
}

/// Size a buffer of `bytes` is allocated with (never empty).
pub fn alloc_bytes(bytes: u64) -> u64 {
    bytes.max(256)
}

/// The size of the heightfield a region opens over: samples and cell size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldGrid {
    pub nx: u32,
    pub nz: u32,
    pub dx: f32,
    pub dz: f32,
}

/// Host state of the terrain pool, and the commands of each call.
#[derive(Clone, Debug)]
pub struct TerrainPlan {
    config: TerrainConfig,
    params: MpmParams,
    /// Elastic (P-) wave speed of the loaded material (0 without stress).
    wave_speed: f32,
    /// The loaded rock, whose crack band each step checks against the cell size.
    rock: Option<RockMaterial>,
    /// Heightfield records the activation window touches (bake rewrites them).
    window_records: u32,
}

impl TerrainPlan {
    /// The pool for `config` (no particles, no material).
    pub fn new(config: &TerrainConfig) -> anyhow::Result<Self> {
        config.validate()?;
        // Per-block and per-particle kernels run in a single dispatch row.
        anyhow::ensure!(
            config.max_blocks <= 65535 && config.max_particles <= 65535 * 64,
            "max_blocks {} or max_particles {} exceeds one indirect dispatch row (65535 workgroups)",
            config.max_blocks,
            config.max_particles
        );
        Ok(Self {
            config: *config,
            params: MpmParams {
                counts: [
                    config.max_particles,
                    hash_slots(config.max_blocks),
                    config.max_blocks,
                    MPM_MATERIAL_NONE,
                ],
                dims: [0, 0, 0, MPM_NO_BED],
                couple: [
                    0.0,
                    f32::from_bits(config.max_coupled),
                    f32::from_bits(config.max_parcels),
                    f32::from_bits(config.max_field_samples),
                ],
                ..MpmParams::default()
            },
            wave_speed: 0.0,
            rock: None,
            window_records: 0,
        })
    }

    /// The bytes each binding of the terrain pool needs (before
    /// [`alloc_bytes`]): bindings 37-51. The look-ahead's binding
    /// ([`MPM_WATCH_BINDING`]) is not the pool's: hosts keep it at 256 bytes.
    pub fn buffer_sizes(&self) -> [(usize, u64); MPM_WATCH_BINDING - MPM_PARAMS_BINDING] {
        let config = &self.config;
        let blocks = u64::from(config.max_blocks);
        let particles = u64::from(config.max_particles);
        let slots = u64::from(hash_slots(config.max_blocks));
        let nodes = blocks * u64::from(BLOCK_CELLS.pow(3));
        [
            (MPM_PARAMS_BINDING, std::mem::size_of::<MpmParams>() as u64),
            (MPM_COUNTERS_BINDING, 4 * MPM_COUNTER_WORDS as u64),
            (
                MPM_PARTICLES_BINDING,
                particles * std::mem::size_of::<MpmParticle>() as u64,
            ),
            (MPM_PAYLOAD_BINDING, particles * 80),
            (MPM_HASH_BINDING, slots * 8),
            (MPM_BLOCKS_BINDING, blocks * 16),
            (MPM_CELLS_BINDING, nodes * 16),
            (MPM_SORTED_BINDING, particles * 4),
            (MPM_GRID_BINDING, nodes * 16),
            (MPM_ARGS_BINDING, 36),
            (MPM_COLUMNS_BINDING, u64::from(config.max_columns) * 16),
            (
                MPM_COUPLE_BINDING,
                (4 + blocks * u64::from(1 + MPM_CELL_COLLIDERS)) * 4,
            ),
            (MPM_COUPLED_BINDING, u64::from(config.max_coupled) * 64),
            (MPM_PARCELS_BINDING, u64::from(config.max_parcels) * 32),
            (MPM_DEPOSIT_BINDING, u64::from(config.max_field_samples) * 4),
        ]
    }

    pub fn config(&self) -> &TerrainConfig {
        &self.config
    }

    pub fn params(&self) -> &MpmParams {
        &self.params
    }

    /// Elastic (P-) wave speed of the loaded material (m/s; 0 without stress).
    pub fn wave_speed(&self) -> f32 {
        self.wave_speed
    }

    /// The region's bed heightfield collider, if it has one.
    pub fn bed(&self) -> Option<u32> {
        let bed = self.params.dims[3];
        (bed != MPM_NO_BED).then_some(bed)
    }

    /// The region's box and its bed (the heightfield collider under it, or none).
    fn set_region(
        &mut self,
        plan: &RegionPlan,
        bed: Option<u32>,
        bed_friction: f32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            plan.blocks.max_element() as u32 <= MPM_MAX_BLOCKS_PER_AXIS,
            "region spans {:?} blocks; block keys hold {MPM_MAX_BLOCKS_PER_AXIS} per axis",
            plan.blocks
        );
        let cells = plan.blocks * BLOCK_CELLS as i32;
        self.params.origin = [plan.origin.x, plan.origin.y, plan.origin.z, plan.dx];
        self.params.dims = [
            cells.x as u32,
            cells.y as u32,
            cells.z as u32,
            bed.unwrap_or(MPM_NO_BED),
        ];
        self.params.step[3] = bed_friction;
        Ok(())
    }

    /// A new region throws its ejecta from slot 0 and counts afresh.
    fn reset_ejecta(ops: &mut TerrainOps) {
        for w in [MPM_C_PARCELS, MPM_C_DEPOSITED, MPM_C_ESCAPED] {
            ops.counter(w, 0);
        }
    }

    /// Make `plan` the active region holding `particles` host-loaded
    /// particles (the host writes them to [`MPM_PARTICLES_BINDING`] itself).
    pub fn load(
        &mut self,
        plan: &RegionPlan,
        particles: usize,
        bed: Option<u32>,
        bed_friction: f32,
    ) -> anyhow::Result<TerrainOps> {
        let n = u32::try_from(particles).unwrap_or(u32::MAX);
        TerrainConfig::check(
            "max_particles",
            u64::from(n),
            u64::from(self.config.max_particles),
        )?;
        self.set_region(plan, bed, bed_friction)?;
        // No activation window: the bed is the field's surface everywhere.
        self.params.window = [0; 4];
        self.window_records = 0;
        let mut ops = TerrainOps::default();
        ops.counter(MPM_C_PARTICLES, n);
        Self::reset_ejecta(&mut ops);
        Ok(ops)
    }

    /// Open `plan` as the active region over heightfield `collider` (`field`'s
    /// grid, its frame at `field_origin` with no rotation, y up): every sample
    /// whose cell lies inside the region (a cell in from its box) is lowered to
    /// the region's bed, and the material above becomes particles of `density`
    /// holding exactly the volume removed (in whole quanta). An activation
    /// past `max_particles` changes nothing and fails the next step's check.
    pub fn activate(
        &mut self,
        plan: &RegionPlan,
        collider: u32,
        field: FieldGrid,
        field_origin: glam::Vec3,
        density: f32,
        bed_friction: f32,
    ) -> anyhow::Result<TerrainOps> {
        let cfg = self.config;
        anyhow::ensure!(
            density > 0.0 && density.is_finite(),
            "density must be positive"
        );
        self.set_region(plan, Some(collider), bed_friction)?;
        let FieldGrid {
            nx,
            nz,
            dx: sx,
            dz: sz,
        } = field;
        let x0 = field_origin.x - 0.5 * (nx - 1) as f32 * sx;
        let z0 = field_origin.z - 0.5 * (nz - 1) as f32 * sz;
        let lo = plan.origin + glam::Vec3::splat(plan.dx);
        let hi = plan.end() - glam::Vec3::splat(plan.dx);
        // Interior samples (full-area cells) whose cell lies within [lo, hi].
        let range = |o: f32, d: f32, n: u32, a: f32, b: f32| {
            let first = ((a - o) / d + 0.5).ceil().max(1.0) as u32;
            let last = ((b - o) / d - 0.5).floor().min((n - 2) as f32) as i64;
            (first, (last - i64::from(first) + 1).max(0) as u32)
        };
        let (i0, ni) = range(x0, sx, nx, lo.x, hi.x);
        let (j0, nj) = range(z0, sz, nz, lo.z, hi.z);
        anyhow::ensure!(ni > 0 && nj > 0, "the region covers no heightfield samples");
        TerrainConfig::check(
            "max_field_samples",
            u64::from(nx) * u64::from(nz),
            u64::from(cfg.max_field_samples),
        )?;
        TerrainConfig::check(
            "max_columns",
            u64::from(ni) * u64::from(nj),
            u64::from(cfg.max_columns),
        )?;
        let spacing = plan.dx / cfg.particles_per_cell_axis as f32;
        self.params.window = [i0, j0, ni, nj];
        self.params.column = [
            lo.y + 0.5 * plan.dx - field_origin.y,
            spacing.powi(3) / QUANTA_PER_LATTICE_CELL,
            spacing,
            density,
        ];
        let first = (j0 * nx + i0) / 4;
        let last = ((j0 + nj - 1) * nx + i0 + ni - 1) / 4;
        self.window_records = last - first + 1;
        let samples = u64::from(ni) * u64::from(nj);
        let records = u64::from(self.window_records);
        let mut ops = TerrainOps::default();
        ops.params(&self.params);
        ops.counter(MPM_C_PARTICLES, 0);
        Self::reset_ejecta(&mut ops);
        ops.disp("mpm_activate_count", samples);
        ops.disp("mpm_activate_scan", 64);
        ops.disp("mpm_activate_emit", samples);
        ops.disp("mpm_activate_lower", records);
        Ok(ops)
    }

    /// Close the region: every particle's quanta are splatted bilinearly, in
    /// whole quanta, onto the four heightfield samples around its home sample
    /// moved by its displacement (window samples directly, others through the
    /// deposits), and the region holds no particles.
    pub fn bake(&self) -> anyhow::Result<TerrainOps> {
        anyhow::ensure!(
            self.window_records > 0,
            "terrain_bake needs a region opened by terrain_activate"
        );
        let records = u64::from(self.window_records);
        let mut ops = TerrainOps::default();
        ops.disp("mpm_write_args", 1);
        ops.particles("mpm_bake_bin");
        ops.disp("mpm_bake_raise", records);
        ops.disp("mpm_bake_header", 1);
        self.deposits(&mut ops);
        Ok(ops)
    }

    /// The loaded material's constitutive model (see `WgpuSim::terrain_set_material`).
    pub fn set_material(&mut self, material: impl Into<TerrainMaterial>) -> anyhow::Result<()> {
        let material = material.into();
        material.validate()?;
        self.rock = None;
        self.wave_speed = material.p_wave_speed();
        let (kind, words) = match material {
            TerrainMaterial::Inert => (MPM_MATERIAL_NONE, [0.0; 4]),
            TerrainMaterial::Sand(m) => {
                let (kind, friction, cohesion) = match m.yield_surface {
                    // Dry sand keeps its kernel (and its results) unchanged.
                    YieldSurface::MohrCoulomb if m.cohesion == 0.0 => {
                        (MPM_MATERIAL_SAND_MC, m.friction_angle.sin(), 0.0)
                    }
                    YieldSurface::MohrCoulomb => (
                        MPM_MATERIAL_SOIL_MC,
                        m.friction_angle.sin(),
                        m.cohesion_cos_phi(),
                    ),
                    YieldSurface::DruckerPrager => (MPM_MATERIAL_SAND, m.alpha(), m.apex_strain()),
                };
                (kind, [m.lambda(), m.mu(), friction, cohesion])
            }
            TerrainMaterial::Rock(m) => {
                self.rock = Some(m);
                (
                    MPM_MATERIAL_ROCK,
                    [m.lambda(), m.mu(), m.tensile_strength, m.band_stress()],
                )
            }
        };
        self.params.counts[3] = kind;
        self.params.sand = words;
        Ok(())
    }

    /// The substeps' parameters (dt, gravity, the CFL speed bound), as a write.
    fn step_params(
        &mut self,
        ops: &mut TerrainOps,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            substeps > 0 && dt > 0.0,
            "terrain substeps need a positive count and dt"
        );
        TerrainConfig::check(
            "max_substeps",
            u64::from(substeps),
            u64::from(self.config.max_substeps),
        )?;
        if let Some(rock) = self.rock {
            rock.softening_stress(self.params.origin[3])?;
        }
        self.params.step[0] = dt;
        self.params.step[1] = gravity;
        // Nodes lighter than this carry no velocity (f32 momentum over mass).
        self.params.step[2] = 1.0e-12;
        // CFL: courant·dx per substep, less what the elastic wave takes.
        let courant_speed = self.config.courant * self.params.origin[3] / dt;
        self.params.limits[0] = courant_speed - self.wave_speed;
        anyhow::ensure!(
            self.params.limits[0] > 0.0,
            "terrain dt {dt} s is above the CFL bound for the material's wave speed ({} m/s): use \
             engine_terrain::substeps_for",
            self.wave_speed
        );
        ops.params(&self.params);
        Ok(())
    }

    /// `substeps` substeps of `dt` under `gravity` (m/s² along y) with no
    /// rigid coupling, then the parcels' deposits.
    pub fn step(&mut self, substeps: u32, dt: f32, gravity: f32) -> anyhow::Result<TerrainOps> {
        let mut ops = TerrainOps::default();
        self.step_params(&mut ops, substeps, dt, gravity)?;
        // No colliders listed: the grid holds no rigid contacts.
        ops.disp("mpm_couple_clear", u64::from(self.config.max_blocks));
        for _ in 0..substeps {
            self.substep(&mut ops, None);
        }
        self.deposits(&mut ops);
        Ok(ops)
    }

    /// A coupled frame (explicit, staggered) ahead of the rigid step: the
    /// rigid colliders (`colliders` in the world) overlapping the region are
    /// listed, each substep holds the material to them, and their reactions
    /// become impulse records the rigid step applies at its start (so the
    /// step must run with its impulse pass on).
    pub fn coupled(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
        colliders: u32,
    ) -> anyhow::Result<TerrainOps> {
        self.params.couple[0] = substeps as f32 * dt;
        let mut ops = TerrainOps::default();
        self.step_params(&mut ops, substeps, dt, gravity)?;
        let slots = u64::from(self.config.max_coupled);
        ops.disp("mpm_couple_clear", u64::from(self.config.max_blocks));
        ops.disp("mpm_couple_bin", u64::from(colliders));
        for _ in 0..substeps {
            self.substep(&mut ops, Some(slots));
        }
        self.deposits(&mut ops);
        ops.disp("mpm_couple_emit", slots);
        ops.disp("mpm_couple_finish", 1);
        Ok(ops)
    }

    /// One MPM substep's dispatches. With `coupled` (the coupling slots), the
    /// rigid contacts' exchange runs between the grid update and G2P.
    fn substep(&self, ops: &mut TerrainOps, coupled: Option<u64>) {
        // Hash words to empty.
        let clear = 2 * u64::from(self.params.counts[1]);
        ops.disp("mpm_clear", clear);
        ops.disp("mpm_write_args", 1);
        ops.particles("mpm_touch");
        ops.disp("mpm_write_args", 1);
        ops.particles("mpm_count");
        ops.blocks("mpm_scan_cells");
        ops.particles("mpm_scatter");
        ops.blocks("mpm_cell_sort");
        ops.particles("mpm_payload");
        ops.blocks("mpm_p2g");
        ops.blocks("mpm_grid");
        if let Some(slots) = coupled {
            ops.disp("mpm_couple_gather", slots * 64);
            ops.blocks("mpm_couple_apply");
        }
        ops.particles("mpm_g2p");
        ops.indirect("mpm_parcels", ARGS_PARCELS);
    }

    /// Once per frame: the parcels' deposits onto the heightfield.
    fn deposits(&self, ops: &mut TerrainOps) {
        let records = u64::from(self.config.max_field_samples).div_ceil(4);
        ops.disp("mpm_deposit_raise", records);
        ops.disp("mpm_deposit_header", 1);
    }

    /// Start a km/s impact or explosion at `point` (on the surface): every
    /// live particle of the region takes `field`'s excavation velocity.
    pub fn apply_source(
        &mut self,
        field: &ZModel,
        point: glam::Vec3,
    ) -> anyhow::Result<TerrainOps> {
        anyhow::ensure!(point.is_finite(), "source point must be finite");
        self.params.source = [
            [point.x, point.y, point.z, field.z as f32],
            [
                field.flow_center_depth as f32,
                field.core_radius as f32,
                field.alpha as f32,
                0.0,
            ],
        ];
        let mut ops = TerrainOps::default();
        ops.params(&self.params);
        ops.disp("mpm_write_args", 1);
        ops.particles("mpm_source");
        Ok(ops)
    }

    /// After a host has read the counters: their flags start clear again.
    pub fn flags_reset(&self) -> TerrainWrite {
        TerrainWrite {
            binding: MPM_COUNTERS_BINDING,
            offset: 4 * MPM_C_FLAGS as u64,
            words: vec![0],
        }
    }

    /// The flags word as a `clear_buffer` range (binding, byte offset, bytes),
    /// for pipelined hosts that clear it after copying the counters.
    pub fn flags_clear(&self) -> (usize, u64, u64) {
        (MPM_COUNTERS_BINDING, 4 * MPM_C_FLAGS as u64, 4)
    }

    /// The report a copy of the counters (`MPM_COUNTER_WORDS` words) carries,
    /// or the error a flag names.
    pub fn check(&self, c: &[u32]) -> anyhow::Result<TerrainStepReport> {
        let (cfg, params) = (self.config, self.params);
        anyhow::ensure!(
            c.len() >= MPM_COUNTER_WORDS,
            "terrain counters are {MPM_COUNTER_WORDS} words"
        );
        let flags = c[MPM_C_FLAGS];
        if flags & MPM_OVERFLOW_PARTICLES != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_particles",
                requested: u64::from(cfg.max_particles) + 1,
                allowed: u64::from(cfg.max_particles),
            }
            .into());
        }
        if flags & MPM_OVERFLOW_BLOCKS != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_blocks",
                requested: u64::from(c[MPM_C_BLOCKS]),
                allowed: u64::from(cfg.max_blocks),
            }
            .into());
        }
        if flags & MPM_OVERFLOW_PARCELS != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_parcels",
                requested: u64::from(c[MPM_C_PARCELS]),
                allowed: u64::from(cfg.max_parcels),
            }
            .into());
        }
        anyhow::ensure!(
            flags & MPM_OVERFLOW_FIELD == 0,
            "the bed heightfield has more samples than max_field_samples ({})",
            cfg.max_field_samples
        );
        if flags & MPM_OVERFLOW_COUPLED != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_coupled",
                requested: u64::from(cfg.max_coupled) + 1,
                allowed: u64::from(cfg.max_coupled),
            }
            .into());
        }
        anyhow::ensure!(
            flags & MPM_OVERFLOW_CELL == 0,
            "more than {MPM_CELL_COLLIDERS} rigid colliders overlap one terrain block (4 cells across)"
        );
        anyhow::ensure!(
            flags & MPM_OVERFLOW_IMPULSES == 0,
            "the rigid step's impulse records are full: terrain reactions were not all applied"
        );
        anyhow::ensure!(
            flags & MPM_OVERFLOW_HASH == 0,
            "terrain block hash full ({} slots)",
            params.counts[1]
        );
        anyhow::ensure!(
            flags & MPM_OUTSIDE == 0,
            "terrain material left the active region (one cell inside its box): enlarge the region's margin"
        );
        anyhow::ensure!(
            flags & MPM_CFL == 0,
            "terrain material moved faster ({} m/s) than the substep's CFL bound ({} m/s): take \
             more substeps (engine_terrain::substeps_for with a higher max_speed)",
            f32::from_bits(c[MPM_C_MAX_SPEED]),
            params.limits[0]
        );
        anyhow::ensure!(
            flags & MPM_INVALID == 0,
            "a terrain particle's deformation or plastic state is not finite"
        );
        Ok(TerrainStepReport {
            blocks: c[MPM_C_BLOCKS],
            max_speed: f32::from_bits(c[MPM_C_MAX_SPEED]),
            particles: c[MPM_C_PARTICLES],
            parcels: c[MPM_C_PARCELS].min(cfg.max_parcels),
            deposited: c[MPM_C_DEPOSITED],
            escaped: c[MPM_C_ESCAPED],
            airborne: c[MPM_C_AIRBORNE],
            body_speed: f32::from_bits(c[MPM_C_BODY_SPEED]),
        })
    }
}

/// The open region's frame schedule (`engine-world`'s terrain host): each
/// frame's substeps cover the frame at the CFL bound for a speed bound the
/// GPU checks, from the last reported speed; the region is settled once its
/// fastest particle has stayed under a rest speed, with nothing in flight,
/// for a number of frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionClock {
    /// Speed bound basis: the source's peak, then each frame's report.
    speed: f32,
    /// The step whose report set `speed`.
    speed_step: u64,
    /// Consecutive settled frames.
    quiet: u32,
    /// No settling before this step (a strike still on its way in).
    hold_until: u64,
}

impl RegionClock {
    /// A region opened at `step` whose material starts at up to `speed` m/s.
    pub fn new(speed: f32, step: u64) -> Self {
        Self {
            speed,
            speed_step: step,
            quiet: 0,
            hold_until: 0,
        }
    }

    /// Keep the region open (not settled) until `step`.
    pub fn hold_until(&mut self, step: u64) {
        self.hold_until = step;
    }

    /// `(substeps, dt)` for the frame at `step`: the bound is 1.5 × the last
    /// reported speed, plus what gravity can add over the readback lag, plus
    /// `headroom` (bodies entering faster than the sand).
    #[allow(clippy::too_many_arguments)]
    pub fn substeps(
        &self,
        config: &TerrainConfig,
        step: u64,
        frame_dt: f32,
        gravity: f32,
        dx: f32,
        wave_speed: f32,
        headroom: f32,
    ) -> Result<(u32, f32), TerrainError> {
        let lag = (step - self.speed_step + 1) as f32;
        let bound = 1.5 * self.speed + gravity.abs() * frame_dt * lag + headroom;
        substeps_for(config, frame_dt, dx, wave_speed, bound)
    }

    /// The report of the frame at `step` (reports arrive in order; one from
    /// before the region opened is ignored). A frame is quiet when its
    /// material and every body coupled to it move slower than `settle_speed`,
    /// nothing is in flight, and no hold is on.
    pub fn report(&mut self, step: u64, report: &TerrainStepReport, settle_speed: f32) {
        if step < self.speed_step {
            return;
        }
        self.speed = report.max_speed;
        self.speed_step = step;
        if report.max_speed < settle_speed
            && report.body_speed < settle_speed
            && report.airborne == 0
            && step >= self.hold_until
        {
            self.quiet += 1;
        } else {
            self.quiet = 0;
        }
    }

    /// Settled for `frames` frames: bake it.
    pub fn settled(&self, frames: u32) -> bool {
        self.quiet >= frames
    }

    /// The speed the next frame's bound starts from (m/s).
    pub fn speed(&self) -> f32 {
        self.speed
    }
}

/// The largest speed a Z-model excavation field starts: at its core's edge.
pub fn source_peak_speed(model: &ZModel) -> f32 {
    let core = model.core_radius as f32;
    let depth = model.flow_center_depth as f32;
    let deep = model
        .velocity_f32(glam::Vec3::new(0.0, -depth, core))
        .length();
    deep.max(model.surface_speed(model.core_radius) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> FieldGrid {
        FieldGrid {
            nx: 33,
            nz: 33,
            dx: 0.125,
            dz: 0.125,
        }
    }

    #[test]
    fn a_coupled_frame_lists_colliders_once_and_gathers_every_substep() {
        let cfg = TerrainConfig::default();
        let mut p = TerrainPlan::new(&cfg).unwrap();
        p.set_material(Some(&crate::SandMaterial::default()))
            .unwrap();
        let region = RegionPlan::spanning(
            &cfg,
            glam::Vec3::splat(-1.0),
            glam::Vec3::splat(1.0),
            0.1,
            0,
            0,
        )
        .unwrap();
        p.activate(&region, 0, grid(), glam::Vec3::ZERO, 1600.0, 0.6)
            .unwrap();
        let ops = p.coupled(3, 1.0e-4, -9.81, 5).unwrap();
        let count = |k: &str| {
            ops.cmds
                .iter()
                .filter(|c| matches!(c, TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. } if *kernel == k))
                .count()
        };
        assert_eq!(count("mpm_couple_bin"), 1);
        assert_eq!(count("mpm_couple_gather"), 3);
        assert_eq!(count("mpm_couple_apply"), 3);
        assert_eq!(count("mpm_p2g"), 3);
        assert_eq!(count("mpm_couple_finish"), 1);
        // The params ride ahead of the pass.
        assert_eq!(ops.writes.len(), 1);
        assert_eq!(ops.writes[0].binding, MPM_PARAMS_BINDING);
        // Every kernel the plan names is in KERNELS.
        for c in &ops.cmds {
            let (TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. }) = c;
            assert!(KERNELS.contains(kernel), "{kernel}");
        }
    }

    #[test]
    fn a_step_past_the_cfl_bound_fails_before_any_work() {
        let cfg = TerrainConfig::default();
        let mut p = TerrainPlan::new(&cfg).unwrap();
        p.set_material(Some(&crate::SandMaterial::default()))
            .unwrap();
        let region = RegionPlan::spanning(
            &cfg,
            glam::Vec3::splat(-1.0),
            glam::Vec3::splat(1.0),
            0.1,
            0,
            0,
        )
        .unwrap();
        p.load(&region, 10, None, 0.5).unwrap();
        assert!(p.step(1, 1.0 / 60.0, -9.81).is_err());
        assert!(
            p.bake().is_err(),
            "a loaded region has no window to bake into"
        );
    }

    #[test]
    fn the_clock_settles_after_quiet_frames_and_resets_on_motion() {
        let mut c = RegionClock::new(100.0, 10);
        let r = |s: f32, air: u32| TerrainStepReport {
            blocks: 1,
            max_speed: s,
            particles: 1,
            parcels: 0,
            deposited: 0,
            escaped: 0,
            airborne: air,
            body_speed: 0.0,
        };
        c.report(9, &r(0.0, 0), 0.01);
        assert_eq!(
            c.speed(),
            100.0,
            "a report from before the region is ignored"
        );
        c.report(10, &r(0.001, 0), 0.01);
        c.report(11, &r(0.001, 1), 0.01);
        assert!(!c.settled(1), "a parcel in flight keeps it open");
        c.report(12, &r(0.001, 0), 0.01);
        c.report(13, &r(0.001, 0), 0.01);
        assert!(c.settled(2));
        let cfg = TerrainConfig::default();
        // The bound grows by g·dt per frame of readback lag.
        let a = c
            .substeps(&cfg, 14, 1.0 / 60.0, -9.81, 1.0, 200.0, 0.0)
            .unwrap();
        let b = c
            .substeps(&cfg, 400, 1.0 / 60.0, -9.81, 1.0, 200.0, 0.0)
            .unwrap();
        assert!(b.0 >= a.0);
    }
}

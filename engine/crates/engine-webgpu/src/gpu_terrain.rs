//! Deformable-terrain MPM on the native wgpu host (`slang/kernels/terrain_mpm.slang`,
//! bindings 37-47): one active region's particles, stepped on the GPU in
//! substeps. Limits come from a [`TerrainConfig`]; a substep that would need more
//! blocks than it allows, or a particle leaving the region, fails the call.
//!
//! Material enters the region either from the host ([`WgpuSim::terrain_load`],
//! for tests) or from the heightfield under it ([`WgpuSim::terrain_activate`]),
//! and returns to the heightfield with [`WgpuSim::terrain_bake`]. The particle
//! count lives on the GPU, so activation needs no readback. Errors the GPU
//! records outside a step (an activation that would pass `max_particles`) are
//! reported by the next [`WgpuSim::terrain_step`].

use super::WgpuSim;
use engine_schema::heightfield::HeightField;
use engine_terrain::layout::*;
use engine_terrain::{RegionPlan, SandMaterial, TerrainConfig, TerrainError, BLOCK_CELLS};

/// Host-side state of the terrain pool.
pub(crate) struct TerrainState {
    config: TerrainConfig,
    params: MpmParams,
    /// Elastic (P-) wave speed of the loaded material (0 without stress).
    wave_speed: f32,
    /// Heightfield records the activation window touches (bake rewrites them).
    window_records: u32,
}

/// What a run of substeps reported (one readback at the end).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainStepReport {
    /// Blocks the last substep used.
    pub blocks: u32,
    /// Largest particle speed after the last substep (m/s).
    pub max_speed: f32,
    /// Particles in the region.
    pub particles: u32,
}

/// Quanta per lattice particle: a column's volume is rounded to s³/64 (s the
/// particle spacing), so a sample's height moves by whole quanta.
const QUANTA_PER_LATTICE_CELL: f32 = 64.0;

pub(crate) fn terrain_buffer(device: &wgpu::Device, binding: usize, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("b{binding} terrain")),
        size: bytes.max(256),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | if binding == MPM_ARGS_BINDING {
                wgpu::BufferUsages::INDIRECT
            } else {
                wgpu::BufferUsages::empty()
            },
        mapped_at_creation: false,
    })
}

impl WgpuSim {
    /// Size the terrain buffers for `config` (and drop any particles).
    pub fn terrain_enable(&mut self, config: &TerrainConfig) -> anyhow::Result<()> {
        config.validate()?;
        // Per-block and per-particle kernels run in a single dispatch row.
        anyhow::ensure!(
            config.max_blocks <= 65535 && config.max_particles <= 65535 * 64,
            "max_blocks {} or max_particles {} exceeds one indirect dispatch row (65535 workgroups)",
            config.max_blocks,
            config.max_particles
        );
        let blocks = u64::from(config.max_blocks);
        let particles = u64::from(config.max_particles);
        let slots = u64::from(hash_slots(config.max_blocks));
        let nodes = blocks * u64::from(BLOCK_CELLS.pow(3));
        let sizes = [
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
            (MPM_ARGS_BINDING, 24),
            (MPM_COLUMNS_BINDING, u64::from(config.max_columns) * 16),
        ];
        let limits = self.device.limits();
        for &(binding, bytes) in &sizes {
            crate::features::check_limits(&limits, binding, bytes)?;
        }
        for &(binding, bytes) in &sizes {
            self.bufs[binding] = terrain_buffer(&self.device, binding, bytes);
            self.rebind_buffer(binding as u32);
        }
        self.terrain = Some(TerrainState {
            config: *config,
            params: MpmParams {
                counts: [
                    config.max_particles,
                    hash_slots(config.max_blocks),
                    config.max_blocks,
                    MPM_MATERIAL_NONE,
                ],
                dims: [0, 0, 0, MPM_NO_BED],
                ..MpmParams::default()
            },
            wave_speed: 0.0,
            window_records: 0,
        });
        Ok(())
    }

    fn terrain_state(&mut self) -> anyhow::Result<&mut TerrainState> {
        self.terrain
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call terrain_enable)"))
    }

    /// The region's box and its bed (the heightfield collider under it, or none).
    fn terrain_set_region(
        state: &mut TerrainState,
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
        state.params.origin = [plan.origin.x, plan.origin.y, plan.origin.z, plan.dx];
        state.params.dims = [
            cells.x as u32,
            cells.y as u32,
            cells.z as u32,
            bed.unwrap_or(MPM_NO_BED),
        ];
        state.params.step[3] = bed_friction;
        Ok(())
    }

    fn terrain_write_counter(&self, word: usize, value: u32) {
        self.queue.write_buffer(
            &self.bufs[MPM_COUNTERS_BINDING],
            4 * word as u64,
            bytemuck::bytes_of(&value),
        );
    }

    /// Make `plan` the active region and `particles` its material. `bed` is the
    /// heightfield collider under it, `bed_friction` the Coulomb μ there.
    pub fn terrain_load(
        &mut self,
        plan: &RegionPlan,
        particles: &[MpmParticle],
        bed: Option<u32>,
        bed_friction: f32,
    ) -> anyhow::Result<()> {
        let state = self.terrain_state()?;
        let cfg = state.config;
        let n = u32::try_from(particles.len()).unwrap_or(u32::MAX);
        TerrainConfig::check("max_particles", u64::from(n), u64::from(cfg.max_particles))?;
        Self::terrain_set_region(state, plan, bed, bed_friction)?;
        state.window_records = 0;
        if !particles.is_empty() {
            self.queue.write_buffer(
                &self.bufs[MPM_PARTICLES_BINDING],
                0,
                bytemuck::cast_slice(particles),
            );
        }
        self.terrain_write_counter(MPM_C_PARTICLES, n);
        Ok(())
    }

    /// Open `plan` as the active region over heightfield `collider` (the field
    /// `field`, its frame at `field_origin` with no rotation, y up): every sample
    /// whose cell lies inside the region (a cell in from its box) is lowered to
    /// the region's bed, and the material above becomes particles of `density`
    /// holding exactly the volume removed (in whole quanta). Runs on the GPU;
    /// an activation past `max_particles` changes nothing and fails the next
    /// [`Self::terrain_step`].
    pub fn terrain_activate(
        &mut self,
        plan: &RegionPlan,
        collider: u32,
        field: &HeightField,
        field_origin: glam::Vec3,
        density: f32,
        bed_friction: f32,
    ) -> anyhow::Result<()> {
        let state = self.terrain_state()?;
        let cfg = state.config;
        anyhow::ensure!(
            density > 0.0 && density.is_finite(),
            "density must be positive"
        );
        Self::terrain_set_region(state, plan, Some(collider), bed_friction)?;
        let (nx, nz) = field.samples();
        let (sx, sz) = field.cell_size();
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
            "max_columns",
            u64::from(ni) * u64::from(nj),
            u64::from(cfg.max_columns),
        )?;
        let spacing = plan.dx / cfg.particles_per_cell_axis as f32;
        state.params.window = [i0, j0, ni, nj];
        state.params.column = [
            lo.y + 0.5 * plan.dx - field_origin.y,
            spacing.powi(3) / QUANTA_PER_LATTICE_CELL,
            spacing,
            density,
        ];
        let first = (j0 * nx + i0) / 4;
        let last = ((j0 + nj - 1) * nx + i0 + ni - 1) / 4;
        state.window_records = last - first + 1;
        let params = state.params;
        let samples = u64::from(ni) * u64::from(nj);
        let records = u64::from(state.window_records);
        self.queue.write_buffer(
            &self.bufs[MPM_PARAMS_BINDING],
            0,
            bytemuck::bytes_of(&params),
        );
        self.terrain_write_counter(MPM_C_PARTICLES, 0);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain activate"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain activate"),
                timestamp_writes: None,
            });
            self.disp(&mut pass, "mpm_activate_count", samples, None);
            self.disp(&mut pass, "mpm_activate_scan", 64, None);
            self.disp(&mut pass, "mpm_activate_emit", samples, None);
            self.disp(&mut pass, "mpm_activate_lower", records, None);
        }
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }

    /// Close the region: every particle's quanta go back onto the heightfield
    /// sample it stands over (within the activation window), and the region
    /// holds no particles. Runs on the GPU.
    pub fn terrain_bake(&mut self) -> anyhow::Result<()> {
        let state = self.terrain_state()?;
        anyhow::ensure!(
            state.window_records > 0,
            "terrain_bake needs a region opened by terrain_activate"
        );
        let records = u64::from(state.window_records);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain bake"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain bake"),
                timestamp_writes: None,
            });
            self.disp(&mut pass, "mpm_write_args", 1, None);
            self.disp_terrain_particles(&mut pass, "mpm_bake_bin");
            self.disp(&mut pass, "mpm_bake_raise", records, None);
            self.disp(&mut pass, "mpm_bake_header", 1, None);
        }
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }

    /// The loaded material's constitutive model: `None` carries no stress
    /// (transfers only), `Some` is Drucker–Prager sand. Particles' masses are the
    /// caller's (density × rest volume).
    pub fn terrain_set_material(&mut self, material: Option<&SandMaterial>) -> anyhow::Result<()> {
        let state = self.terrain_state()?;
        match material {
            None => {
                state.params.counts[3] = MPM_MATERIAL_NONE;
                state.params.sand = [0.0; 4];
                state.wave_speed = 0.0;
            }
            Some(m) => {
                m.validate()?;
                state.params.counts[3] = MPM_MATERIAL_SAND;
                state.params.sand = [m.lambda(), m.mu(), m.alpha(), m.cohesion];
                state.wave_speed = m.p_wave_speed();
            }
        }
        Ok(())
    }

    /// Run `substeps` MPM substeps of `dt` under `gravity` (m/s² along y), in one
    /// submission. Fails if any substep (or the activation before it) overflowed
    /// a limit or a particle left the region (its state is then the GPU's as of
    /// the failure).
    pub fn terrain_step(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<TerrainStepReport> {
        let state = self.terrain_state()?;
        state.params.step[0] = dt;
        state.params.step[1] = gravity;
        // Nodes lighter than this carry no velocity (f32 momentum over mass).
        state.params.step[2] = 1.0e-12;
        // CFL: courant·dx per substep, less what the elastic wave takes.
        let courant_speed = state.config.courant * state.params.origin[3] / dt;
        state.params.limits[0] = courant_speed - state.wave_speed;
        anyhow::ensure!(
            state.params.limits[0] > 0.0,
            "terrain dt {dt} s is above the CFL bound for the material's wave speed ({} m/s): use \
             engine_terrain::substeps_for",
            state.wave_speed
        );
        let params = state.params;
        let cfg = state.config;
        self.queue.write_buffer(
            &self.bufs[MPM_PARAMS_BINDING],
            0,
            bytemuck::bytes_of(&params),
        );
        let clear = 2 * u64::from(params.counts[1]);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain"),
                timestamp_writes: None,
            });
            for _ in 0..substeps {
                self.disp(&mut pass, "mpm_clear", clear, None);
                self.disp(&mut pass, "mpm_write_args", 1, None);
                self.disp_terrain_particles(&mut pass, "mpm_touch");
                self.disp(&mut pass, "mpm_write_args", 1, None);
                self.disp_terrain_particles(&mut pass, "mpm_count");
                self.disp_terrain_blocks(&mut pass, "mpm_scan_cells");
                self.disp(&mut pass, "mpm_scan_blocks", 64, None);
                self.disp_terrain_particles(&mut pass, "mpm_scatter");
                self.disp_terrain_blocks(&mut pass, "mpm_cell_sort");
                self.disp_terrain_particles(&mut pass, "mpm_payload");
                self.disp_terrain_blocks(&mut pass, "mpm_p2g");
                self.disp_terrain_blocks(&mut pass, "mpm_grid");
                self.disp_terrain_particles(&mut pass, "mpm_g2p");
            }
        }
        self.queue.submit(Some(enc.finish()));
        let c: Vec<u32> = self.read_binding(MPM_COUNTERS_BINDING, 0, MPM_COUNTER_WORDS);
        // Flags are reported once.
        self.terrain_write_counter(MPM_C_FLAGS, 0);
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
        })
    }

    fn disp_terrain_indirect<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        name: &str,
        offset: u64,
    ) {
        let p = &self.pipes[name];
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &p.bind, &[]);
        pass.dispatch_workgroups_indirect(&self.bufs[MPM_ARGS_BINDING], offset);
    }

    /// One workgroup (64 threads) per block in use, counted on the GPU.
    fn disp_terrain_blocks<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>, name: &str) {
        self.disp_terrain_indirect(pass, name, 0);
    }

    /// One thread per particle in the region, counted on the GPU.
    fn disp_terrain_particles<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>, name: &str) {
        self.disp_terrain_indirect(pass, name, 12);
    }

    /// The active region's particles (blocking readback).
    pub fn terrain_particles(&self) -> Vec<MpmParticle> {
        if self.terrain.is_none() {
            return Vec::new();
        }
        let n = self.read_binding::<u32>(MPM_COUNTERS_BINDING, 4 * MPM_C_PARTICLES as u64, 1)[0];
        if n == 0 {
            return Vec::new();
        }
        self.read_binding(MPM_PARTICLES_BINDING, 0, n as usize)
    }
}

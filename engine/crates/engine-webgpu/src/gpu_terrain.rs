//! Deformable-terrain MPM on the native wgpu host (`slang/kernels/terrain_mpm.slang`,
//! bindings 37-45): one active region's particles, stepped on the GPU in
//! substeps. Limits come from a [`TerrainConfig`]; a substep that would need more
//! blocks than it allows, or a particle leaving the region, fails the call.

use super::WgpuSim;
use engine_terrain::layout::*;
use engine_terrain::{RegionPlan, SandMaterial, TerrainConfig, TerrainError, BLOCK_CELLS};

/// Host-side state of the terrain pool.
pub(crate) struct TerrainState {
    config: TerrainConfig,
    params: MpmParams,
    /// Elastic (P-) wave speed of the loaded material (0 without stress).
    wave_speed: f32,
}

/// What a run of substeps reported (one readback at the end).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainStepReport {
    /// Blocks the last substep used.
    pub blocks: u32,
    /// Largest particle speed after the last substep (m/s).
    pub max_speed: f32,
}

pub(crate) fn terrain_buffer(device: &wgpu::Device, binding: usize, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("b{binding} terrain")),
        size: bytes.max(256),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl WgpuSim {
    /// Size the terrain buffers for `config` (and drop any particles).
    pub fn terrain_enable(&mut self, config: &TerrainConfig) -> anyhow::Result<()> {
        config.validate()?;
        let blocks = u64::from(config.max_blocks);
        let particles = u64::from(config.max_particles);
        let slots = u64::from(hash_slots(config.max_blocks));
        let nodes = blocks * u64::from(BLOCK_CELLS.pow(3));
        let sizes = [
            (MPM_PARAMS_BINDING, std::mem::size_of::<MpmParams>() as u64),
            (MPM_COUNTERS_BINDING, 16),
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
                    0,
                    hash_slots(config.max_blocks),
                    config.max_blocks,
                    MPM_MATERIAL_NONE,
                ],
                dims: [0, 0, 0, MPM_NO_BED],
                ..MpmParams::default()
            },
            wave_speed: 0.0,
        });
        Ok(())
    }

    fn terrain_state(&mut self) -> anyhow::Result<&mut TerrainState> {
        self.terrain
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call terrain_enable)"))
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
        if n > cfg.max_particles {
            return Err(TerrainError::LimitExceeded {
                limit: "max_particles",
                requested: u64::from(n),
                allowed: u64::from(cfg.max_particles),
            }
            .into());
        }
        let cells = plan.blocks * BLOCK_CELLS as i32;
        anyhow::ensure!(
            plan.blocks.max_element() as u32 <= MPM_MAX_BLOCKS_PER_AXIS,
            "region spans {:?} blocks; block keys hold {MPM_MAX_BLOCKS_PER_AXIS} per axis",
            plan.blocks
        );
        state.params.origin = [plan.origin.x, plan.origin.y, plan.origin.z, plan.dx];
        state.params.counts[0] = n;
        state.params.dims = [
            cells.x as u32,
            cells.y as u32,
            cells.z as u32,
            bed.unwrap_or(MPM_NO_BED),
        ];
        state.params.step[3] = bed_friction;
        if !particles.is_empty() {
            self.queue.write_buffer(
                &self.bufs[MPM_PARTICLES_BINDING],
                0,
                bytemuck::cast_slice(particles),
            );
        }
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
    /// submission. Fails if any substep overflowed a limit or a particle left the
    /// region (its state is then the GPU's as of the failure).
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
        self.queue.write_buffer(
            &self.bufs[MPM_COUNTERS_BINDING],
            0,
            bytemuck::cast_slice(&[0u32; 4]),
        );
        let particles = u64::from(params.counts[0]);
        let nodes = u64::from(cfg.max_blocks) * 64;
        let clear = (2 * u64::from(params.counts[1])).max(4 * nodes);
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
                self.disp(&mut pass, "mpm_touch", particles, None);
                self.disp(&mut pass, "mpm_count", particles, None);
                self.disp(&mut pass, "mpm_scan_cells", nodes, None);
                self.disp(&mut pass, "mpm_scan_blocks", 64, None);
                self.disp(&mut pass, "mpm_scatter", particles, None);
                self.disp(&mut pass, "mpm_cell_sort", nodes, None);
                self.disp(&mut pass, "mpm_payload", particles, None);
                self.disp(&mut pass, "mpm_p2g", nodes, None);
                self.disp(&mut pass, "mpm_grid", nodes, None);
                self.disp(&mut pass, "mpm_g2p", particles, None);
            }
        }
        self.queue.submit(Some(enc.finish()));
        let c: Vec<u32> = self.read_binding(MPM_COUNTERS_BINDING, 0, 4);
        let flags = c[1];
        if flags & MPM_OVERFLOW_BLOCKS != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_blocks",
                requested: u64::from(c[0]),
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
            f32::from_bits(c[2]),
            params.limits[0]
        );
        anyhow::ensure!(
            flags & MPM_INVALID == 0,
            "a terrain particle's deformation or plastic state is not finite"
        );
        Ok(TerrainStepReport {
            blocks: c[0],
            max_speed: f32::from_bits(c[2]),
        })
    }

    /// The active region's particles (blocking readback).
    pub fn terrain_particles(&self) -> Vec<MpmParticle> {
        let n = self.terrain.as_ref().map_or(0, |t| t.params.counts[0]);
        if n == 0 {
            return Vec::new();
        }
        self.read_binding(MPM_PARTICLES_BINDING, 0, n as usize)
    }
}

//! Deformable-terrain MPM on the native wgpu host (`slang/kernels/terrain_mpm.slang`,
//! bindings 37-51): one active region's particles, stepped on the GPU in
//! substeps. Limits come from a [`TerrainConfig`]; a substep that would need more
//! blocks than it allows, or a particle leaving the region, fails the call.
//!
//! The commands come from [`engine_terrain::plan::TerrainPlan`], the sequence
//! the browser host replays too; this file runs them on wgpu.
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
use engine_terrain::plan::{alloc_bytes, FieldGrid, TerrainCmd, TerrainOps, TerrainPlan};
use engine_terrain::{RegionPlan, TerrainConfig, TerrainMaterial};

pub use engine_terrain::TerrainStepReport;

/// Host-side state of the terrain pool: the shared plan.
pub(crate) struct TerrainState {
    plan: TerrainPlan,
}

pub(crate) fn terrain_buffer(device: &wgpu::Device, binding: usize, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("b{binding} terrain")),
        size: alloc_bytes(bytes),
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
        let plan = TerrainPlan::new(config)?;
        let sizes = plan.buffer_sizes();
        let limits = self.device.limits();
        for &(binding, bytes) in &sizes {
            crate::features::check_limits(&limits, binding, bytes)?;
        }
        for &(binding, bytes) in &sizes {
            self.bufs[binding] = terrain_buffer(&self.device, binding, bytes);
            self.rebind_buffer(binding as u32);
        }
        self.terrain = Some(TerrainState { plan });
        Ok(())
    }

    fn terrain_plan(&mut self) -> anyhow::Result<&mut TerrainPlan> {
        self.terrain
            .as_mut()
            .map(|t| &mut t.plan)
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call terrain_enable)"))
    }

    /// The plan's host writes, queued ahead of the next submission.
    fn terrain_writes(&self, ops: &TerrainOps) {
        for w in &ops.writes {
            self.queue.write_buffer(
                &self.bufs[w.binding],
                w.offset,
                bytemuck::cast_slice(&w.words),
            );
        }
    }

    /// The plan's commands as one compute pass of `enc`.
    fn terrain_pass(&self, enc: &mut wgpu::CommandEncoder, ops: &TerrainOps, label: &str) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        for c in &ops.cmds {
            match *c {
                TerrainCmd::Dispatch { kernel, threads } => {
                    self.disp(&mut pass, kernel, threads, None)
                }
                TerrainCmd::Indirect { kernel, offset } => {
                    self.disp_terrain_indirect(&mut pass, kernel, offset)
                }
            }
        }
    }

    /// The plan's writes and commands in a submission of their own.
    fn terrain_submit(&self, ops: &TerrainOps, label: &str) {
        self.terrain_writes(ops);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        self.terrain_pass(&mut enc, ops, label);
        self.queue.submit(Some(enc.finish()));
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
        let ops = self
            .terrain_plan()?
            .load(plan, particles.len(), bed, bed_friction)?;
        if !particles.is_empty() {
            self.queue.write_buffer(
                &self.bufs[MPM_PARTICLES_BINDING],
                0,
                bytemuck::cast_slice(particles),
            );
        }
        self.terrain_writes(&ops);
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
        let (nx, nz) = field.samples();
        let (dx, dz) = field.cell_size();
        let ops = self.terrain_plan()?.activate(
            plan,
            collider,
            FieldGrid { nx, nz, dx, dz },
            field_origin,
            density,
            bed_friction,
        )?;
        self.terrain_submit(&ops, "terrain activate");
        let body = self.collider(collider).body;
        self.mark_surface_moved(body);
        Ok(())
    }

    /// Close the region: every particle's quanta go back onto the heightfield
    /// sample it stands over (within the activation window), and the region
    /// holds no particles. Runs on the GPU.
    pub fn terrain_bake(&mut self) -> anyhow::Result<()> {
        let plan = self.terrain_plan()?;
        let ops = plan.bake()?;
        let bed = plan.bed();
        self.terrain_submit(&ops, "terrain bake");
        if let Some(bed) = bed {
            let body = self.collider(bed).body;
            self.mark_surface_moved(body);
        }
        Ok(())
    }

    /// The loaded material's constitutive model: [`TerrainMaterial::Inert`]
    /// (or `None`) carries no stress (transfers only); sand or soil (`Some(&sand)`
    /// or `&sand`) is elastoplastic with its yield surface (dry Mohr–Coulomb sand,
    /// cohesive soil or the Drucker–Prager cone); rock (`&rock`) softens with
    /// damage, and each step fails ([`engine_terrain::TerrainError::CrackBand`])
    /// while the region's cell is too large for its crack band. Particles'
    /// masses are the caller's (density × rest volume).
    pub fn terrain_set_material(
        &mut self,
        material: impl Into<TerrainMaterial>,
    ) -> anyhow::Result<()> {
        self.terrain_plan()?.set_material(material)
    }

    /// Run `substeps` MPM substeps of `dt` under `gravity` (m/s² along y), in one
    /// submission, with no rigid coupling. Fails if any substep (or the
    /// activation before it) overflowed a limit or a particle left the region
    /// (its state is then the GPU's as of the failure).
    pub fn terrain_step(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<TerrainStepReport> {
        let ops = self.terrain_plan()?.step(substeps, dt, gravity)?;
        self.terrain_submit(&ops, "terrain");
        self.terrain_report()
    }

    /// The counters after the last submission (blocking), as a report or the
    /// error any flag names; flags are reported once.
    fn terrain_report(&mut self) -> anyhow::Result<TerrainStepReport> {
        let c: Vec<u32> = self.read_binding(MPM_COUNTERS_BINDING, 0, MPM_COUNTER_WORDS);
        let reset = self.terrain_plan()?.flags_reset();
        self.queue.write_buffer(
            &self.bufs[reset.binding],
            reset.offset,
            bytemuck::cast_slice(&reset.words),
        );
        self.terrain_check(&c)
    }

    /// The report a copy of the terrain counters (`MPM_COUNTER_WORDS` words of
    /// [`Self::terrain_counters_buffer`]) carries, or the error a flag names.
    pub fn terrain_check(&self, c: &[u32]) -> anyhow::Result<TerrainStepReport> {
        self.terrain
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call terrain_enable)"))?
            .plan
            .check(c)
    }

    /// The terrain counters (`MPM_COUNTER_WORDS` u32 words), for hosts that copy
    /// them into their own readback ([`Self::terrain_check`] reads the copy).
    pub fn terrain_counters_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[MPM_COUNTERS_BINDING]
    }

    /// The hull arena (binding 18): a heightfield's samples are the f32 words
    /// from its collider's base + 2, row-major, for hosts that copy them into
    /// their readback.
    pub fn hull_arena_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[18]
    }

    /// Encode a coupled terrain frame into `enc`, ahead of the rigid step the
    /// host encodes next ([`Self::step_encode`]), for pipelined hosts: no
    /// submission, no readback. The step applies the reactions; the host copies
    /// [`Self::terrain_counters_buffer`] into its readback after it and then
    /// encodes [`Self::terrain_encode_flags_clear`].
    pub fn terrain_encode_coupled(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        substeps: u32,
        dt: f32,
    ) -> anyhow::Result<()> {
        let gravity = self.params.gravity_y;
        let colliders = self.collider_count;
        let ops = self
            .terrain_plan()?
            .coupled(substeps, dt, gravity, colliders)?;
        self.terrain_writes(&ops);
        self.terrain_pass(enc, &ops, "terrain coupled");
        self.terrain_impulses_pending = true;
        self.wake_pending = true;
        Ok(())
    }

    /// After the host's copy of the counters: their flags start the next frame clear.
    pub fn terrain_encode_flags_clear(&self, enc: &mut wgpu::CommandEncoder) {
        let (binding, offset, size) = self
            .terrain
            .as_ref()
            .map_or((MPM_COUNTERS_BINDING, 4 * MPM_C_FLAGS as u64, 4), |t| {
                t.plan.flags_clear()
            });
        enc.clear_buffer(&self.bufs[binding], offset, Some(size));
    }

    /// One frame of terrain and rigid bodies, coupled (explicit, staggered): the
    /// MPM substeps hold the material to the rigid colliders in the region as
    /// they stood at the frame's start, and each dynamic body's reaction becomes
    /// an impulse the rigid step then applies at its start. `substeps` of `dt`
    /// should cover the rigid step's dt; gravity is the rigid world's. Forward
    /// only: nothing is re-stepped.
    pub fn step_with_terrain(
        &mut self,
        substeps: u32,
        dt: f32,
    ) -> anyhow::Result<(engine_schema::StepCounters, TerrainStepReport)> {
        let gravity = self.params.gravity_y;
        let colliders = self.collider_count;
        // The terrain must be enabled before the host's records go up.
        self.terrain_plan()?;
        // The host's records first: the coupling appends after them.
        self.upload_impulses();
        let ops = self
            .terrain_plan()?
            .coupled(substeps, dt, gravity, colliders)?;
        self.terrain_submit(&ops, "terrain coupled");
        self.terrain_impulses_pending = true;
        self.wake_pending = true;
        let counters = self
            .step()
            .map_err(|e| anyhow::anyhow!("rigid step: {e:?}"))?;
        let report = self.terrain_report()?;
        Ok((counters, report))
    }

    fn disp_terrain_indirect<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        name: &str,
        offset: u64,
    ) {
        if !self.wg_allows(name) {
            return;
        }
        let p = &self.pipes[name];
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &p.bind, &[]);
        pass.dispatch_workgroups_indirect(&self.bufs[MPM_ARGS_BINDING], offset);
    }

    /// Start a km/s impact or explosion at `point` (on the surface): every live
    /// particle of the region takes `field`'s excavation velocity (added to its
    /// own). The source's shock stage is replaced by this outcome; the MPM
    /// computes excavation, ejection, rim uplift and collapse. Runs on the GPU.
    pub fn terrain_apply_source(
        &mut self,
        field: &engine_terrain::ZModel,
        point: glam::Vec3,
    ) -> anyhow::Result<()> {
        let ops = self.terrain_plan()?.apply_source(field, point)?;
        self.terrain_submit(&ops, "terrain source");
        Ok(())
    }

    /// The region's parcels, in flight or landed (blocking readback).
    pub fn terrain_parcels(&self) -> Vec<MpmParcel> {
        let Some(t) = self.terrain.as_ref() else {
            return Vec::new();
        };
        let n = self.read_binding::<u32>(MPM_COUNTERS_BINDING, 4 * MPM_C_PARCELS as u64, 1)[0]
            .min(t.plan.config().max_parcels);
        if n == 0 {
            return Vec::new();
        }
        self.read_binding(MPM_PARCELS_BINDING, 0, n as usize)
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

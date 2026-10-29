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
use engine_terrain::plan::{
    alloc_bytes, AreaInfo, FieldGrid, Opened, TerrainCmd, TerrainOps, TerrainPlan,
};
use engine_terrain::{RegionPlan, TerrainConfig, TerrainMaterial};

pub use engine_terrain::TerrainStepReport;

/// Host-side state of the terrain pool: the shared plan.
pub(crate) struct TerrainState {
    plan: TerrainPlan,
    /// The last submission's GPU ms (`RUBBLE_TERRAIN_GPU_MS=1`).
    gpu_ms: std::cell::Cell<Option<f64>>,
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
        self.terrain = Some(TerrainState {
            plan,
            gpu_ms: std::cell::Cell::new(None),
        });
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
        // Diagnostic (`RUBBLE_TERRAIN_TIMING=1`, on a device with timestamp
        // queries): each kernel's GPU time, printed per submission.
        if std::env::var("RUBBLE_TERRAIN_TIMING").is_ok_and(|v| v == "1")
            && self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            if let Ok(times) = self.terrain_timed_pass(ops) {
                let total: f64 = times.iter().map(|(_, ms)| ms).sum();
                let parts: Vec<String> =
                    times.iter().map(|(n, ms)| format!("{n} {ms:.3}")).collect();
                eprintln!("{label} GPU {total:.3} ms: {}", parts.join(", "));
            }
            return;
        }
        if std::env::var("RUBBLE_TERRAIN_GPU_MS").is_ok_and(|v| v == "1")
            && self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            // The frame's GPU time: timestamps written on the encoder around
            // its one pass where the device can (a pass's own ends, around
            // many dispatches, read 0 on Metal); else the kernels' times
            // summed, each timed in a pass of its own.
            let ms = if self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
            {
                self.terrain_span_pass(ops)
            } else {
                self.terrain_timed_pass(ops)
                    .map_or(0.0, |t| t.iter().map(|(_, ms)| ms).sum())
            };
            if let Some(t) = self.terrain.as_ref() {
                t.gpu_ms.set(Some(ms));
            }
            return;
        }
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
        self.terrain_open_area(plan, collider, field, field_origin, density, bed_friction)?;
        Ok(())
    }

    /// Open `plan` over heightfield `collider` as [`Self::terrain_activate`],
    /// beside the open areas: far from them it is an area of its own; near
    /// them (overlapping, on a rim, partly outside) they grow into one box
    /// with it; inside one, nothing is activated
    /// ([`engine_terrain::plan::TerrainPlan::open`]). Returns which area
    /// covers it and which it took over.
    pub fn terrain_open_area(
        &mut self,
        plan: &RegionPlan,
        collider: u32,
        field: &HeightField,
        field_origin: glam::Vec3,
        density: f32,
        bed_friction: f32,
    ) -> anyhow::Result<Opened> {
        let (nx, nz) = field.samples();
        let (dx, dz) = field.cell_size();
        let mut opened = self.terrain_plan()?.open(
            plan,
            collider,
            FieldGrid { nx, nz, dx, dz },
            field_origin,
            density,
            bed_friction,
        )?;
        let ops = std::mem::take(&mut opened.ops);
        if opened.activated {
            // A grown area's old walls first (each its own pass), then the
            // activation.
            for pre in std::mem::take(&mut opened.pre) {
                self.terrain_submit(&pre, "terrain grow");
            }
            self.terrain_submit(&ops, "terrain activate");
            let body = self.collider(collider).body;
            self.mark_surface_moved(body);
        }
        Ok(opened)
    }

    /// Close every open area: each particle's quanta are split, in whole
    /// quanta, between the tent at its home sample moved by its displacement
    /// (unmoved material returns exactly where it came from) and P2G's
    /// B-spline at its position, and the pool holds no particles. Runs on the
    /// GPU.
    pub fn terrain_bake(&mut self) -> anyhow::Result<()> {
        let plan = self.terrain_plan()?;
        let bed = plan.bed();
        let all = plan.bake()?;
        for ops in &all {
            self.terrain_submit(ops, "terrain bake");
        }
        if let Some(bed) = bed {
            let body = self.collider(bed).body;
            self.mark_surface_moved(body);
        }
        Ok(())
    }

    /// Close area `slot` alone (see [`Self::terrain_bake`]); the others go on.
    pub fn terrain_bake_area(&mut self, slot: u32) -> anyhow::Result<()> {
        let plan = self.terrain_plan()?;
        let bed = plan.bed();
        let ops = plan.bake_area(slot)?;
        self.terrain_submit(&ops, "terrain bake");
        if let Some(bed) = bed {
            let body = self.collider(bed).body;
            self.mark_surface_moved(body);
        }
        Ok(())
    }

    /// The open areas.
    pub fn terrain_areas(&self) -> Vec<AreaInfo> {
        self.terrain
            .as_ref()
            .map_or_else(Vec::new, |t| t.plan.areas())
    }

    /// The terrain's plan (cells, areas), if terrain is enabled.
    pub fn terrain_plan_ref(&self) -> Option<&TerrainPlan> {
        self.terrain.as_ref().map(|t| &t.plan)
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

    /// Diagnostic: [`Self::terrain_step`] with every dispatch in a timestamped
    /// pass of its own, and each kernel's GPU time summed over the frame (ms,
    /// in first-dispatch order). Needs a device with timestamp queries (the
    /// default device asks for them under `RUBBLE_GPU_TIMING=1`); never on a
    /// host's hot path (it waits for the GPU).
    pub fn terrain_step_timed(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<(TerrainStepReport, Vec<(&'static str, f64)>)> {
        anyhow::ensure!(
            self.device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY),
            "terrain_step_timed needs timestamp queries (RUBBLE_GPU_TIMING=1)"
        );
        let ops = self.terrain_plan()?.step(substeps, dt, gravity)?;
        self.terrain_writes(&ops);
        let times = self.terrain_timed_pass(&ops)?;
        Ok((self.terrain_report()?, times))
    }

    /// `ops` as one pass between two encoder timestamps, submitted and waited
    /// for: its GPU ms.
    fn terrain_span_pass(&self, ops: &TerrainOps) -> f64 {
        let queries = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("terrain span"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain span"),
            });
        enc.write_timestamp(&queries, 0);
        self.terrain_pass(&mut enc, ops, "terrain span");
        enc.write_timestamp(&queries, 1);
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain span"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain span read"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.resolve_query_set(&queries, 0..2, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, 16);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let t: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        if std::env::var("RUBBLE_TS_DEBUG").is_ok() {
            eprintln!("span ts {t:?}");
        }
        t[1].saturating_sub(t[0]) as f64 * f64::from(self.queue.get_timestamp_period()) * 1e-6
    }

    /// Diagnostic: the GPU time (ms) of the last terrain submission when
    /// `RUBBLE_TERRAIN_GPU_MS=1` (on a device with timestamp queries; each
    /// submission then waits for the GPU): its kernels' times summed, each
    /// dispatch timed in a pass of its own. Else `None`.
    pub fn terrain_last_gpu_ms(&self) -> Option<f64> {
        self.terrain.as_ref().and_then(|t| t.gpu_ms.get())
    }

    /// `ops`' commands, each in a timestamped pass, submitted and waited for:
    /// each kernel's summed GPU time (ms).
    fn terrain_timed_pass(&self, ops: &TerrainOps) -> anyhow::Result<Vec<(&'static str, f64)>> {
        let mut times: Vec<(&'static str, f64)> = Vec::new();
        // A query set holds at most 4096 timestamps: chunks of 2048 commands.
        for chunk in ops.cmds.chunks(2048) {
            let count = 2 * chunk.len() as u32;
            let queries = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("terrain kernels"),
                ty: wgpu::QueryType::Timestamp,
                count,
            });
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain timed"),
                });
            for (k, c) in chunk.iter().enumerate() {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain timed"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &queries,
                        beginning_of_pass_write_index: Some(2 * k as u32),
                        end_of_pass_write_index: Some(2 * k as u32 + 1),
                    }),
                });
                match *c {
                    TerrainCmd::Dispatch { kernel, threads } => {
                        self.disp(&mut pass, kernel, threads, None)
                    }
                    TerrainCmd::Indirect { kernel, offset } => {
                        self.disp_terrain_indirect(&mut pass, kernel, offset)
                    }
                }
            }
            let bytes = 8 * u64::from(count);
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("terrain kernels"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("terrain kernels read"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.resolve_query_set(&queries, 0..count, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
            self.queue.submit([enc.finish()]);
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            let t: Vec<u64> =
                bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
            let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
            for (k, c) in chunk.iter().enumerate() {
                let (TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. }) =
                    *c;
                let ms = t[2 * k + 1].saturating_sub(t[2 * k]) as f64 * period;
                match times.iter_mut().find(|(n, _)| *n == kernel) {
                    Some((_, sum)) => *sum += ms,
                    None => times.push((kernel, ms)),
                }
            }
        }
        Ok(times)
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

    /// [`Self::terrain_apply_source`] in area `slot`: its particles within the
    /// field's extent add the field's velocity to their own (sources
    /// superpose). On material that has moved, the source sits on its surface
    /// under `point`, found on the GPU.
    pub fn terrain_apply_source_to(
        &mut self,
        slot: u32,
        field: &engine_terrain::ZModel,
        point: glam::Vec3,
    ) -> anyhow::Result<()> {
        let ops = self.terrain_plan()?.apply_source_to(slot, field, point)?;
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

    /// Overwrite the active region's particles from `first` on with
    /// `particles` (their count is unchanged), ahead of the next submission:
    /// tests set an initial state on activated material this way.
    pub fn terrain_write_particles(
        &self,
        first: usize,
        particles: &[MpmParticle],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.terrain.is_some(),
            "terrain is not enabled (call terrain_enable)"
        );
        let n = self.read_binding::<u32>(MPM_COUNTERS_BINDING, 4 * MPM_C_PARTICLES as u64, 1)[0];
        anyhow::ensure!(
            first + particles.len() <= n as usize,
            "particles {first}..{} are past the region's {n}",
            first + particles.len()
        );
        self.queue.write_buffer(
            &self.bufs[MPM_PARTICLES_BINDING],
            (first * std::mem::size_of::<MpmParticle>()) as u64,
            bytemuck::cast_slice(particles),
        );
        Ok(())
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

    /// Diagnostic: overwrite the active region's particles with `particles`
    /// (as [`Self::terrain_particles`] read them, the same count), keeping the
    /// region and its activation window. For tests that move material by hand
    /// before a bake; never on a host's hot path.
    pub fn terrain_overwrite_particles(&mut self, particles: &[MpmParticle]) -> anyhow::Result<()> {
        anyhow::ensure!(self.terrain.is_some(), "terrain_enable first");
        let n = self.read_binding::<u32>(MPM_COUNTERS_BINDING, 4 * MPM_C_PARTICLES as u64, 1)[0];
        anyhow::ensure!(
            particles.len() == n as usize,
            "terrain_overwrite_particles: {} particles given, the region holds {n}",
            particles.len()
        );
        if !particles.is_empty() {
            self.queue.write_buffer(
                &self.bufs[MPM_PARTICLES_BINDING],
                0,
                bytemuck::cast_slice(particles),
            );
        }
        Ok(())
    }
}

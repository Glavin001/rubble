//! Deformable-terrain MPM on native Metal: the shared
//! [`engine_terrain::plan::TerrainPlan`] replayed on [`MetalSim`]'s buffers.
//!
//! The kernels are `slang/kernels/terrain_mpm.slang` (and the look-ahead's
//! `terrain_watch.slang`) compiled Slang → MSL in the browser's `ENGINE_WGSL`
//! layout (build.rs `terrain_kernels`), the same source the wgpu host runs.
//! Bindings 37-51 are the terrain pool, sized from the plan; 52 the
//! look-ahead's. Each call's host writes reach the GPU through blit copies at
//! the head of its own command buffer (the queue-ordered writes of wgpu's
//! `write_buffer`), then its commands run in one serial compute encoder.
//! Indirect dispatches read their threadgroup counts from binding 46 on the
//! GPU, as on wgpu: no host readback inside a frame.
//!
//! [`TerrainCaps`] holds the Metal-only fast paths (none changes results
//! unless it says so).

use crate::sim::{gpu_ms, groups, shared, write, EncodeState, MetalSim, RigidKernel};
use engine_schema::heightfield::HeightField;
use engine_terrain::layout::*;
use engine_terrain::plan::{
    alloc_bytes, FieldGrid, TerrainCmd, TerrainOps, TerrainPlan, TerrainWrite,
};
use engine_terrain::{RegionPlan, TerrainConfig, TerrainMaterial};
use metal::*;

pub use engine_terrain::TerrainStepReport;

mod table {
    use super::RigidKernel;
    include!(concat!(env!("OUT_DIR"), "/terrain_kernels.rs"));
}
pub use table::TERRAIN_KERNELS;

/// Native fast paths for the terrain plan (`slang/kernels/terrain_mpm_fast.slang`),
/// each behind its own flag. [`TerrainCaps::default`] is the portable path:
/// the plan's own kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainCaps {
    /// P2G in two passes (`mpm_p2g_cells`, `mpm_p2g_nodes`): each particle is
    /// read once, by the SIMD group of its cell, into per-(cell, node)
    /// partials that each node then sums. Deterministic (fixed orders, no
    /// atomics); it groups the same terms by cell, so it differs from the
    /// portable kernel in rounding only. Binding 53 holds the partials
    /// (`max_blocks` · 64 · 27 · 16 bytes).
    pub p2g_cells: bool,
    /// The binning reads each particle once (`mpm_touch_keys`,
    /// `mpm_count_keys`, `mpm_scatter_keys`): touch writes its base node,
    /// count turns it into its cell, scatter uses the cell, instead of each
    /// reading the 128-byte particle and looking the cell up again. Same cells
    /// and counts, same sorted order: bit for bit the portable binning.
    /// Binding 54 holds the keys (`max_particles` · 8 bytes).
    pub sort_keys: bool,
}

/// Bytes of the partials buffer (binding 53) for `max_blocks`.
pub fn p2g_partial_bytes(max_blocks: u32) -> u64 {
    u64::from(max_blocks) * 64 * 27 * 16
}

impl TerrainCaps {
    /// Every fast path on.
    pub fn all() -> Self {
        Self {
            p2g_cells: true,
            sort_keys: true,
        }
    }

    /// Flag names separated by `,` or `+` (`all` for every one).
    pub fn parse(flags: &str) -> anyhow::Result<Self> {
        let mut caps = Self::default();
        for f in flags.split([',', '+']).map(str::trim) {
            match f {
                "all" => caps = Self::all(),
                "p2g_cells" => caps.p2g_cells = true,
                "sort_keys" => caps.sort_keys = true,
                "" => {}
                other => anyhow::bail!("unknown terrain fast path {other}"),
            }
        }
        Ok(caps)
    }

    /// From `ENGINE_TERRAIN_CAPS` (as [`Self::parse`]; none when unset).
    pub fn from_env() -> Self {
        std::env::var("ENGINE_TERRAIN_CAPS")
            .map(|v| Self::parse(&v).unwrap_or_else(|e| panic!("ENGINE_TERRAIN_CAPS: {e}")))
            .unwrap_or_default()
    }
}

/// GPU time of one terrain call, whole and by kernel.
#[derive(Clone, Debug, Default)]
pub struct TerrainTiming {
    /// The command buffer's GPU time (ms): blits and every dispatch.
    pub gpu_ms: f64,
    /// Per kernel (ms, summed over its dispatches), in first-dispatch order;
    /// filled by the profiled calls only (one encoder per dispatch, so the sum
    /// runs a little above `gpu_ms` of an unprofiled call).
    pub kernels: Vec<(&'static str, f64, u32)>,
}

/// Host state of the terrain pool on Metal.
pub(crate) struct MetalTerrain {
    plan: TerrainPlan,
    pipes: Vec<ComputePipelineState>,
    /// Writes queued for the next terrain command buffer (flag resets).
    pending: Vec<TerrainWrite>,
    caps: TerrainCaps,
    /// The last terrain command buffer (its GPU time, for hosts that time calls).
    last: Option<CommandBuffer>,
    /// Timestamps of the profiled calls (one encoder per dispatch).
    samples: Option<CounterSampleBuffer>,
    /// Profile the next call (`terrain_profile_next`), and what it measured.
    profile_next: bool,
    profiled: Option<TerrainTiming>,
}

/// Binding of the `p2g_cells` partials (after the look-ahead's 52).
pub(crate) const P2G_PARTIALS_BINDING: usize = MPM_WATCH_BINDING + 1;
/// Binding of the `sort_keys` per-particle keys.
pub(crate) const SORT_KEYS_BINDING: usize = MPM_WATCH_BINDING + 2;

/// The plan's commands as this host runs them: a fast path replaces the
/// commands of the kernels it stands for.
fn lower(ops: &TerrainOps, caps: TerrainCaps) -> Vec<TerrainCmd> {
    let mut out = Vec::with_capacity(ops.cmds.len() + 16);
    for &c in &ops.cmds {
        match c {
            TerrainCmd::Indirect {
                kernel: "mpm_p2g",
                offset,
            } if caps.p2g_cells => {
                out.push(TerrainCmd::Indirect {
                    kernel: "mpm_p2g_cells",
                    offset,
                });
                out.push(TerrainCmd::Indirect {
                    kernel: "mpm_p2g_nodes",
                    offset,
                });
            }
            TerrainCmd::Indirect { kernel, offset }
                if caps.sort_keys
                    && matches!(kernel, "mpm_touch" | "mpm_count" | "mpm_scatter") =>
            {
                out.push(TerrainCmd::Indirect {
                    kernel: match kernel {
                        "mpm_touch" => "mpm_touch_keys",
                        "mpm_count" => "mpm_count_keys",
                        _ => "mpm_scatter_keys",
                    },
                    offset,
                })
            }
            c => out.push(c),
        }
    }
    out
}

fn kernel_index(name: &str) -> anyhow::Result<usize> {
    TERRAIN_KERNELS
        .iter()
        .position(|k| k.entry == name)
        .ok_or_else(|| anyhow::anyhow!("MetalSim: no terrain kernel {name}"))
}

fn threads(k: &RigidKernel) -> MTLSize {
    MTLSize::new(
        u64::from(k.threads[0]),
        u64::from(k.threads[1]),
        u64::from(k.threads[2]),
    )
}

impl MetalSim {
    /// Size the terrain buffers for `config` (and drop any particles), and
    /// build the terrain kernels.
    pub fn terrain_enable(&mut self, config: &TerrainConfig) -> anyhow::Result<()> {
        self.terrain_enable_with(config, self.terrain_caps)
    }

    /// The fast paths the next [`Self::terrain_enable`] builds.
    pub fn set_terrain_caps(&mut self, caps: TerrainCaps) {
        self.terrain_caps = caps;
    }

    /// The fast paths of the enabled terrain pool (or the next one's).
    pub fn terrain_caps(&self) -> TerrainCaps {
        self.terrain.as_ref().map_or(self.terrain_caps, |t| t.caps)
    }

    /// [`Self::terrain_enable`] with explicit fast paths.
    pub fn terrain_enable_with(
        &mut self,
        config: &TerrainConfig,
        caps: TerrainCaps,
    ) -> anyhow::Result<()> {
        let plan = TerrainPlan::new(config)?;
        self.wait_all();
        let max = self.device.max_buffer_length();
        for (binding, bytes) in plan.buffer_sizes() {
            let bytes = alloc_bytes(bytes);
            anyhow::ensure!(
                bytes <= max,
                "terrain binding {binding} needs {bytes} bytes; the device allows {max}"
            );
            let b = shared(&self.device, bytes);
            b.set_label(&format!("b{binding} terrain"));
            self.bufs[binding] = b;
        }
        if caps.p2g_cells {
            let bytes = p2g_partial_bytes(config.max_blocks);
            anyhow::ensure!(
                bytes <= max,
                "the p2g_cells partials need {bytes} bytes; the device allows {max}"
            );
            let b = shared(&self.device, bytes);
            b.set_label("b53 terrain partials");
            self.bufs[P2G_PARTIALS_BINDING] = b;
        }
        if caps.sort_keys {
            let b = shared(&self.device, u64::from(config.max_particles) * 8);
            b.set_label("b54 terrain keys");
            self.bufs[SORT_KEYS_BINDING] = b;
        }
        // The GPU's first touch of a page costs far more than a later one
        // (measured: the strike frame ran ~10 ms longer on fresh buffers). Pay
        // it here, at enable, with a fill of every terrain buffer.
        {
            let cb = self.queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            for (binding, _) in plan.buffer_sizes() {
                let b = &self.bufs[binding];
                blit.fill_buffer(b, NSRange::new(0, b.length()), 0);
            }
            for binding in [P2G_PARTIALS_BINDING, SORT_KEYS_BINDING] {
                let b = &self.bufs[binding];
                blit.fill_buffer(b, NSRange::new(0, b.length()), 0);
            }
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let pipes = match self.terrain.take() {
            Some(t) => t.pipes,
            None => self.terrain_pipelines()?,
        };
        self.terrain = Some(MetalTerrain {
            plan,
            pipes,
            pending: Vec::new(),
            caps,
            last: None,
            samples: None,
            profile_next: false,
            profiled: None,
        });
        Ok(())
    }

    fn terrain_pipelines(&self) -> anyhow::Result<Vec<ComputePipelineState>> {
        let (fast, strict) = &self.fast_math;
        let options = |fast_math: bool| {
            let o = CompileOptions::new();
            o.set_fast_math_enabled(fast_math);
            o
        };
        let (fast_opts, strict_opts) = (options(*fast), options(false));
        TERRAIN_KERNELS
            .iter()
            .map(|k| {
                let o = if strict.iter().any(|s| k.entry.starts_with(s.as_str())) {
                    &strict_opts
                } else {
                    &fast_opts
                };
                let lib = self
                    .device
                    .new_library_with_source(k.msl, o)
                    .map_err(|e| anyhow::anyhow!("compile {}: {e}", k.entry))?;
                let func = lib
                    .get_function(k.entry, None)
                    .map_err(|e| anyhow::anyhow!("function {}: {e}", k.entry))?;
                self.device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| anyhow::anyhow!("pipeline {}: {e}", k.entry))
            })
            .collect()
    }

    fn terrain_state(&mut self) -> anyhow::Result<&mut MetalTerrain> {
        self.terrain
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call terrain_enable)"))
    }

    fn terrain_plan(&mut self) -> anyhow::Result<&mut TerrainPlan> {
        Ok(&mut self.terrain_state()?.plan)
    }

    /// The shared plan (limits, parameters).
    pub fn terrain_plan_ref(&self) -> Option<&TerrainPlan> {
        self.terrain.as_ref().map(|t| &t.plan)
    }

    /// Bind kernel `idx`'s pipeline and buffers on `e`, skipping what it holds.
    fn terrain_bind(&self, e: &ComputeCommandEncoderRef, st: &mut EncodeState, idx: usize) {
        let t = self.terrain.as_ref().expect("terrain enabled");
        let k = &TERRAIN_KERNELS[idx];
        if st.pipeline != Some(idx) {
            e.set_compute_pipeline_state(&t.pipes[idx]);
            st.pipeline = Some(idx);
        }
        for &(index, slot) in k.binds {
            let at = index as usize;
            let want = (slot, 0);
            if st.bound[at] == Some(want) {
                continue;
            }
            st.bound[at] = Some(want);
            e.set_buffer(u64::from(index), Some(&self.bufs[slot as usize]), 0);
        }
    }

    /// Encode one plan command on `e`.
    fn terrain_dispatch(
        &self,
        e: &ComputeCommandEncoderRef,
        st: &mut EncodeState,
        cmd: &TerrainCmd,
    ) -> anyhow::Result<()> {
        match *cmd {
            TerrainCmd::Dispatch { kernel, threads: n } => {
                let idx = kernel_index(kernel)?;
                self.terrain_bind(e, st, idx);
                let (x, y) = groups(n);
                e.dispatch_thread_groups(MTLSize::new(x, y, 1), threads(&TERRAIN_KERNELS[idx]));
            }
            TerrainCmd::Indirect { kernel, offset } => {
                let idx = kernel_index(kernel)?;
                self.terrain_bind(e, st, idx);
                e.dispatch_thread_groups_indirect(
                    &self.bufs[MPM_ARGS_BINDING],
                    offset,
                    threads(&TERRAIN_KERNELS[idx]),
                );
            }
        }
        Ok(())
    }

    /// Blit the pending writes and `ops`'s writes into place at the head of `cb`.
    fn terrain_blit_writes(&mut self, cb: &CommandBufferRef, ops: &TerrainOps) {
        let pending = std::mem::take(&mut self.terrain.as_mut().expect("terrain").pending);
        let writes: Vec<&TerrainWrite> = pending.iter().chain(&ops.writes).collect();
        if writes.is_empty() {
            return;
        }
        let bytes: usize = writes.iter().map(|w| 4 * w.words.len()).sum();
        let staging = shared(&self.device, bytes as u64);
        let mut at = 0u64;
        let blit = cb.new_blit_command_encoder();
        for w in writes {
            let n = 4 * w.words.len() as u64;
            write(&staging, at, bytemuck::cast_slice(&w.words));
            blit.copy_from_buffer(&staging, at, &self.bufs[w.binding], w.offset, n);
            at += n;
        }
        blit.end_encoding();
    }

    /// The plan's writes and commands in one command buffer, committed and
    /// tracked with the rigid steps (no wait).
    fn terrain_submit(&mut self, ops: &TerrainOps, label: &str) -> anyhow::Result<CommandBuffer> {
        if std::mem::take(&mut self.terrain.as_mut().expect("terrain").profile_next) {
            self.wait_all();
            let (cb, timing) = self.terrain_submit_profiled(ops, label)?;
            let t = self.terrain.as_mut().expect("terrain");
            t.profiled = Some(timing);
            t.last = Some(cb.clone());
            return Ok(cb);
        }
        let cb = self.queue.new_command_buffer().to_owned();
        cb.set_label(label);
        self.terrain_blit_writes(&cb, ops);
        let desc = ComputePassDescriptor::new();
        desc.set_dispatch_type(MTLDispatchType::Serial);
        let e = cb.compute_command_encoder_with_descriptor(desc);
        let mut st = EncodeState::new();
        let caps = self.terrain.as_ref().expect("terrain").caps;
        for c in &lower(ops, caps) {
            self.terrain_dispatch(e, &mut st, c)?;
        }
        e.end_encoding();
        cb.commit();
        self.in_flight.push_back(cb.clone());
        self.terrain.as_mut().expect("terrain").last = Some(cb.clone());
        Ok(cb)
    }

    /// Like [`Self::terrain_submit`] with one encoder per dispatch, each
    /// timestamped at its boundaries; waits and returns the per-kernel times.
    fn terrain_submit_profiled(
        &mut self,
        ops: &TerrainOps,
        label: &str,
    ) -> anyhow::Result<(CommandBuffer, TerrainTiming)> {
        let caps = self.terrain.as_ref().expect("terrain").caps;
        let cmds = lower(ops, caps);
        let n = cmds.len() as u64;
        // One sample buffer, kept: the device allows only a few alive.
        let have = self
            .terrain
            .as_ref()
            .and_then(|t| t.samples.as_ref())
            .map_or(0, |b| b.sample_count());
        if have < 2 * n {
            let set = self
                .device
                .counter_sets()
                .into_iter()
                .find(|s| s.name() == "timestamp")
                .ok_or_else(|| anyhow::anyhow!("no timestamp counter set on this device"))?;
            anyhow::ensure!(
                self.device
                    .supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary),
                "the device does not sample counters at stage boundaries"
            );
            let desc = CounterSampleBufferDescriptor::new();
            desc.set_counter_set(&set);
            desc.set_sample_count((2 * n).next_power_of_two().max(1024));
            desc.set_storage_mode(MTLStorageMode::Shared);
            let b = self
                .device
                .new_counter_sample_buffer_with_descriptor(&desc)
                .map_err(|e| anyhow::anyhow!("counter sample buffer: {e}"))?;
            self.terrain.as_mut().expect("terrain").samples = Some(b);
        }
        let samples = self
            .terrain
            .as_ref()
            .and_then(|t| t.samples.clone())
            .expect("sample buffer");
        let cb = self.queue.new_command_buffer().to_owned();
        cb.set_label(label);
        self.terrain_blit_writes(&cb, ops);
        for (i, c) in cmds.iter().enumerate() {
            let pd = ComputePassDescriptor::new();
            pd.set_dispatch_type(MTLDispatchType::Serial);
            let att = pd.sample_buffer_attachments().object_at(0).unwrap();
            att.set_sample_buffer(&samples);
            att.set_start_of_encoder_sample_index(2 * i as u64);
            att.set_end_of_encoder_sample_index(2 * i as u64 + 1);
            let e = cb.compute_command_encoder_with_descriptor(pd);
            let mut st = EncodeState::new();
            self.terrain_dispatch(e, &mut st, c)?;
            e.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
        let total = gpu_ms(&cb);
        let ts = samples.resolve_counter_range(NSRange::new(0, 2 * n));
        // Timestamps are in GPU ticks: scale the encoders' span to the
        // command buffer's GPU time (the blits before them are tiny).
        // A sample the GPU could not take reads 0 or MTLCounterErrorValue (!0).
        let valid = |t: u64| t != 0 && t != u64::MAX;
        let lo = ts.iter().copied().filter(|&t| valid(t)).min().unwrap_or(0);
        let hi = ts.iter().copied().filter(|&t| valid(t)).max().unwrap_or(0);
        let span = hi.saturating_sub(lo) as f64;
        let scale = if span > 0.0 { total / span } else { 0.0 };
        let mut kernels: Vec<(&'static str, f64, u32)> = Vec::new();
        for (i, c) in cmds.iter().enumerate() {
            let name = match *c {
                TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. } => kernel,
            };
            let (t0, t1) = (ts[2 * i], ts[2 * i + 1]);
            let ms = if valid(t0) && valid(t1) && t1 >= t0 {
                (t1 - t0) as f64 * scale
            } else {
                0.0
            };
            match kernels.iter_mut().find(|(k, _, _)| *k == name) {
                Some(slot) => {
                    slot.1 += ms;
                    slot.2 += 1;
                }
                None => kernels.push((name, ms, 1)),
            }
        }
        Ok((
            cb,
            TerrainTiming {
                gpu_ms: total,
                kernels,
            },
        ))
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
        self.wait_all();
        if !particles.is_empty() {
            write(
                &self.bufs[MPM_PARTICLES_BINDING],
                0,
                bytemuck::cast_slice(particles),
            );
        }
        self.terrain_state()?.pending.extend(ops.writes);
        Ok(())
    }

    /// Open `plan` as the active region over heightfield `collider` (see
    /// `WgpuSim::terrain_activate`). Runs on the GPU.
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
        self.terrain_submit(&ops, "terrain activate")?;
        let body = self.collider(collider).body;
        self.mark_surface_moved(body);
        Ok(())
    }

    /// Close the region (see `WgpuSim::terrain_bake`). Runs on the GPU.
    pub fn terrain_bake(&mut self) -> anyhow::Result<()> {
        let plan = self.terrain_plan()?;
        let ops = plan.bake()?;
        let bed = plan.bed();
        self.terrain_submit(&ops, "terrain bake")?;
        if let Some(bed) = bed {
            let body = self.collider(bed).body;
            self.mark_surface_moved(body);
        }
        Ok(())
    }

    /// The loaded material's constitutive model (see `WgpuSim::terrain_set_material`).
    pub fn terrain_set_material(
        &mut self,
        material: impl Into<TerrainMaterial>,
    ) -> anyhow::Result<()> {
        self.terrain_plan()?.set_material(material)
    }

    /// Start a km/s impact or explosion at `point` (see
    /// `WgpuSim::terrain_apply_source`). Runs on the GPU.
    pub fn terrain_apply_source(
        &mut self,
        field: &engine_terrain::ZModel,
        point: glam::Vec3,
    ) -> anyhow::Result<()> {
        let ops = self.terrain_plan()?.apply_source(field, point)?;
        self.terrain_submit(&ops, "terrain source")?;
        Ok(())
    }

    /// Run `substeps` MPM substeps of `dt` under `gravity` in one command
    /// buffer, with no rigid coupling, and wait for the counters.
    pub fn terrain_step(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<TerrainStepReport> {
        Ok(self.terrain_step_timed(substeps, dt, gravity)?.0)
    }

    /// [`Self::terrain_step`] and the frame's GPU time (ms).
    pub fn terrain_step_timed(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<(TerrainStepReport, f64)> {
        let ops = self.terrain_plan()?.step(substeps, dt, gravity)?;
        let cb = self.terrain_submit(&ops, "terrain")?;
        cb.wait_until_completed();
        let ms = gpu_ms(&cb);
        Ok((self.terrain_report()?, ms))
    }

    /// [`Self::terrain_step`] with one encoder per dispatch, each timed.
    pub fn terrain_step_profiled(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<(TerrainStepReport, TerrainTiming)> {
        let ops = self.terrain_plan()?.step(substeps, dt, gravity)?;
        self.wait_all();
        let (_, timing) = self.terrain_submit_profiled(&ops, "terrain profiled")?;
        Ok((self.terrain_report()?, timing))
    }

    /// Run the next terrain call (activation, source, step, bake) with one
    /// timestamped encoder per dispatch; [`Self::terrain_profiled`] then
    /// returns what it measured.
    pub fn terrain_profile_next(&mut self) {
        if let Some(t) = self.terrain.as_mut() {
            t.profile_next = true;
        }
    }

    /// The per-kernel timing of the call [`Self::terrain_profile_next`] marked.
    pub fn terrain_profiled(&mut self) -> Option<TerrainTiming> {
        self.terrain.as_mut().and_then(|t| t.profiled.take())
    }

    /// GPU time (ms) of the last terrain call's command buffer (waits for it).
    pub fn terrain_last_gpu_ms(&mut self) -> f64 {
        let Some(cb) = self.terrain.as_mut().and_then(|t| t.last.take()) else {
            return 0.0;
        };
        cb.wait_until_completed();
        gpu_ms(&cb)
    }

    /// The commands [`Self::terrain_step`] would run for a frame, without
    /// running them (tests and the benchmark's census).
    pub fn terrain_step_ops(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<TerrainOps> {
        // The plan's state advances as if stepped (a geostatic start is taken).
        self.terrain_plan()?.step(substeps, dt, gravity)
    }

    /// One frame of terrain and rigid bodies, coupled (explicit, staggered;
    /// see `WgpuSim::step_with_terrain`).
    pub fn step_with_terrain(
        &mut self,
        substeps: u32,
        dt: f32,
    ) -> anyhow::Result<(engine_schema::StepCounters, TerrainStepReport)> {
        let gravity = self.params().gravity_y;
        let colliders = self.host_cols.len() as u32;
        let ops = self
            .terrain_plan()?
            .coupled(substeps, dt, gravity, colliders)?;
        self.terrain_submit(&ops, "terrain coupled")?;
        self.terrain_impulses_pending = true;
        let counters = self
            .step()
            .map_err(|e| anyhow::anyhow!("rigid step: {e:?}"))?;
        let report = self.terrain_report()?;
        Ok((counters, report))
    }

    /// The counters after the last submission (waits), as a report or the
    /// error any flag names; flags are reported once.
    fn terrain_report(&mut self) -> anyhow::Result<TerrainStepReport> {
        let c = self.terrain_counters();
        let t = self.terrain_state()?;
        let reset = t.plan.flags_reset();
        t.pending.push(reset);
        t.plan.check(&c)
    }

    /// The terrain counters (`MPM_COUNTER_WORDS` words; waits).
    pub fn terrain_counters(&mut self) -> Vec<u32> {
        self.read_words(MPM_COUNTERS_BINDING, 0, MPM_COUNTER_WORDS)
    }

    /// The region's parcels, in flight or landed (waits).
    pub fn terrain_parcels(&mut self) -> Vec<MpmParcel> {
        let Some(max) = self.terrain.as_ref().map(|t| t.plan.config().max_parcels) else {
            return Vec::new();
        };
        let n = self.read_words(MPM_COUNTERS_BINDING, MPM_C_PARCELS as u64, 1)[0].min(max);
        self.read_pod(MPM_PARCELS_BINDING, n as usize)
    }

    /// The active region's particles (waits).
    pub fn terrain_particles(&mut self) -> Vec<MpmParticle> {
        if self.terrain.is_none() {
            return Vec::new();
        }
        let n = self.read_words(MPM_COUNTERS_BINDING, MPM_C_PARTICLES as u64, 1)[0];
        self.read_pod(MPM_PARTICLES_BINDING, n as usize)
    }

    /// Overwrite the active region's particles from `first` on (their count
    /// is unchanged), ahead of the next submission.
    pub fn terrain_write_particles(
        &mut self,
        first: usize,
        particles: &[MpmParticle],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.terrain.is_some(),
            "terrain is not enabled (call terrain_enable)"
        );
        let n = self.read_words(MPM_COUNTERS_BINDING, MPM_C_PARTICLES as u64, 1)[0];
        anyhow::ensure!(
            first + particles.len() <= n as usize,
            "particles {first}..{} are past the region's {n}",
            first + particles.len()
        );
        write(
            &self.bufs[MPM_PARTICLES_BINDING],
            (first * std::mem::size_of::<MpmParticle>()) as u64,
            bytemuck::cast_slice(particles),
        );
        Ok(())
    }

    /// The heights of heightfield `collider` as the GPU holds them (waits).
    pub fn height_field_heights(&mut self, collider: u32, field: &HeightField) -> Vec<f32> {
        let base = engine_schema::convex::convex_params_base(self.collider(collider).params);
        let (nx, nz) = field.samples();
        self.read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize)
            .into_iter()
            .map(f32::from_bits)
            .collect()
    }

    fn read_pod<T: bytemuck::Pod>(&mut self, slot: usize, n: usize) -> Vec<T> {
        self.wait_all();
        if n == 0 {
            return Vec::new();
        }
        let buf = &self.bufs[slot];
        assert!(n * std::mem::size_of::<T>() <= buf.length() as usize);
        unsafe { std::slice::from_raw_parts(buf.contents() as *const T, n).to_vec() }
    }
}

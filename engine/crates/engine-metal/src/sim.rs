//! `MetalSim`: native Metal replay of the shared rigid step graph.
//!
//! Same buffers, host words and command list as the WebGPU host
//! (`engine_exec::rigid_plan`), with kernels compiled Slang → MSL in the
//! browser's `ENGINE_WGSL` layout (build.rs `rigid_kernels`), so a measured win
//! lands on both hosts at once. Steps are pipelined: each step's host inputs go
//! to a ring slot and reach the GPU through blit copies inside its own command
//! buffer, and last step's counts are forwarded GPU-side (no readback).

use engine_exec::rigid_plan::{
    self, Cmd, StepInputs, StepModes, BIND_COUNT, COUNTER_UPLOAD, PASS_ALIGN,
};
use engine_exec::ExecError;
use engine_schema::*;
use metal::*;
use std::collections::VecDeque;

mod table {
    pub struct RigidKernel {
        pub entry: &'static str,
        pub msl: &'static str,
        pub threads: [u32; 3],
        /// (Metal `[[buffer(n)]]`, shared `BIND(slot)`)
        pub binds: &'static [(u32, u32)],
    }
    include!(concat!(env!("OUT_DIR"), "/rigid_kernels.rs"));
}
use table::{RigidKernel, RIGID_KERNELS};

/// Host-upload ring slots (steps in flight before the host waits).
const RING: usize = 4;
/// Per-slot layout: scratch words (256 B), host uniform (256 B), counters
/// (128 B), colour counts/degrees (512 B).
const SLOT_BYTES: u64 = 256 + 256 + 128 + 512;
const UP_SCRATCH: u64 = 0;
const UP_HOST: u64 = 256;
const UP_COUNTERS: u64 = 512;
const UP_COLORS: u64 = 640;

fn shared(device: &Device, size: u64) -> Buffer {
    let buf = device.new_buffer(size.max(4), MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, size.max(4) as usize) };
    buf
}

fn write(buf: &Buffer, offset: u64, bytes: &[u8]) {
    assert!(offset as usize + bytes.len() <= buf.length() as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (buf.contents() as *mut u8).add(offset as usize),
            bytes.len(),
        )
    };
}

/// What a compute encoder already has bound (pipeline, (slot, offset) per index).
struct EncodeState {
    pipeline: Option<usize>,
    bound: [Option<(u32, u64)>; 32],
}

impl EncodeState {
    fn new() -> Self {
        Self { pipeline: None, bound: [None; 32] }
    }
}

/// GPU execution time of a completed command buffer (`GPUEndTime - GPUStartTime`).
#[allow(unexpected_cfgs)] // objc 0.2 msg_send! checks a cargo-clippy feature
fn gpu_ms(cb: &CommandBufferRef) -> f64 {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, sel, sel_impl};
    let obj = cb.as_ptr() as *mut objc::runtime::Object;
    // SAFETY: MTLCommandBuffer responds to both selectors once completed.
    let (start, end): (f64, f64) = unsafe { (msg_send![obj, GPUStartTime], msg_send![obj, GPUEndTime]) };
    (end - start) * 1000.0
}

/// Workgroup grid for `threads` invocations at 64 per group, rows of 32768
/// (must agree with Slang's dispatchIndex).
fn groups(threads: u64) -> (u64, u64) {
    let g = threads.max(1).div_ceil(64);
    (g.min(32768), g.div_ceil(32768))
}

pub struct MetalSim {
    queue: CommandQueue,
    chip: String,
    pipes: Vec<ComputePipelineState>,
    bufs: Vec<Buffer>,
    upload: Buffer,
    capacity: WorldCapacity,
    params: AvbdParams,
    modes: StepModes,
    body_count: u32,
    collider_count: u32,
    host_bodies: Vec<BodyState>,
    /// CCD / kinematic bodies uploaded (the motion passes of the step graph).
    motion: rigid_plan::MotionInputs,
    host_cols: Vec<Collider>,
    grid_levels: u32,
    sleep_island_rounds: u32,
    last_counters: StepCounters,
    steps_done: u32,
    contacts_parity: u32,
    in_flight: VecDeque<CommandBuffer>,
    ring: usize,
    /// Per-phase command buffers of a profiled step (`step_profiled`).
    profiling: bool,
    profile_cbs: Vec<(&'static str, CommandBuffer)>,
    kernel_cache: std::cell::RefCell<std::collections::HashMap<&'static str, usize>>,
    /// Steps the counters in `last_counters` describe.
    counted_steps: u32,
}

impl MetalSim {
    /// Persistent-contact world with every default the WebGPU host ships.
    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
        let queue = device.new_command_queue();
        let chip = device.name().to_string();
        let sizes = rigid_plan::buffer_sizes(&capacity, true, true);
        let bufs: Vec<Buffer> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let b = shared(&device, size);
                b.set_label(&format!("rigid {i}"));
                b
            })
            .collect();
        assert_eq!(bufs.len(), BIND_COUNT);
        // Metal's default math mode, as wgpu's Metal backend compiles the same
        // kernels (MTLCompileOptions defaults): fast math on. ENGINE_FAST_MATH=off
        // compiles strict IEEE; ENGINE_STRICT_KERNELS=a,b keeps those strict.
        let fast = std::env::var("ENGINE_FAST_MATH").as_deref() != Ok("off");
        // Collision geometry stays strict IEEE by default: with fast math in the
        // narrowphase a wall-smash brick tunnelled through the ground slab.
        let strict: Vec<String> = std::env::var("ENGINE_STRICT_KERNELS")
            .unwrap_or_else(|_| "narrowphase,convex_".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let options = |fast_math: bool| {
            let o = CompileOptions::new();
            o.set_fast_math_enabled(fast_math);
            o
        };
        let (fast_opts, strict_opts) = (options(fast), options(false));
        let pipes = RIGID_KERNELS
            .iter()
            .map(|k| {
                let compile = if strict.iter().any(|s| k.entry.starts_with(s.as_str())) {
                    &strict_opts
                } else {
                    &fast_opts
                };
                let lib = device
                    .new_library_with_source(k.msl, compile)
                    .map_err(|e| anyhow::anyhow!("compile {}: {e}", k.entry))?;
                let func = lib
                    .get_function(k.entry, None)
                    .map_err(|e| anyhow::anyhow!("function {}: {e}", k.entry))?;
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| anyhow::anyhow!("pipeline {}: {e}", k.entry))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        write(&bufs[22], 0, &rigid_plan::pass_table());
        let mats = [Material {
            friction: 0.5,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        }; 64];
        write(&bufs[3], 0, bytemuck::cast_slice(&mats));
        // Every arena starts with the unit cube that box-versus-hull pairs reuse.
        let arena = engine_schema::convex::HullArena::new();
        write(&bufs[18], 0, bytemuck::cast_slice(arena.records()));
        let modes = StepModes {
            compact_contacts: true,
            persistent_contacts: true,
            manifold_insert: true,
            copy_reuse: true,
            split_np: true,
            listed_persist: true,
            carry_rest: true,
            local_repair: true,
            grid_split: true,
            sparse_start: 32,
            color_headroom: Some(4),
            color_round_headroom: Some(2),
            tail_ok: capacity.max_bodies <= (1 << 16) * 8,
            ..StepModes::default()
        };
        Ok(Self {
            queue,
            chip,
            pipes,
            bufs,
            upload: shared(&device, SLOT_BYTES * RING as u64),
            capacity,
            params: AvbdParams::default(),
            modes,
            body_count: 0,
            collider_count: 0,
            host_bodies: Vec::new(),
            motion: rigid_plan::MotionInputs::default(),
            host_cols: Vec::new(),
            grid_levels: 0,
            sleep_island_rounds: 8,
            last_counters: StepCounters::default(),
            steps_done: 0,
            contacts_parity: 0,
            in_flight: VecDeque::new(),
            ring: 0,
            profiling: false,
            profile_cbs: Vec::new(),
            kernel_cache: Default::default(),
            counted_steps: 0,
        })
    }

    pub fn chip(&self) -> &str {
        &self.chip
    }

    pub fn set_params(&mut self, params: AvbdParams) {
        self.params = params;
    }

    pub fn set_materials(&mut self, mats: &[Material]) {
        assert!(mats.len() <= 64, "the material table holds 64 entries");
        self.wait_all();
        write(&self.bufs[3], 0, bytemuck::cast_slice(mats));
    }

    /// Append bodies (current and previous halves of binding 0).
    pub fn add_bodies(&mut self, bodies: &[BodyState]) -> std::ops::Range<u32> {
        self.wait_all();
        let start = self.body_count;
        let count = bodies.len() as u32;
        assert!(
            count <= self.capacity.max_bodies - start,
            "body upload exceeds capacity"
        );
        let stride = std::mem::size_of::<BodyState>() as u64;
        let bytes = bytemuck::cast_slice(bodies);
        write(&self.bufs[0], u64::from(start) * stride, bytes);
        write(
            &self.bufs[0],
            u64::from(self.capacity.max_bodies + start) * stride,
            bytes,
        );
        self.host_bodies.extend_from_slice(bodies);
        self.motion.ccd |= bodies.iter().any(|b| b.flags & engine_schema::BODY_CCD != 0);
        self.motion.kinematic |= bodies.iter().any(|b| b.flags & 3 == 2);
        self.body_count += count;
        start..self.body_count
    }

    pub fn add_colliders(&mut self, colliders: &[Collider]) -> std::ops::Range<u32> {
        self.wait_all();
        let start = self.collider_count;
        let count = colliders.len() as u32;
        assert!(
            count <= self.capacity.max_colliders - start,
            "collider upload exceeds capacity"
        );
        assert!(
            colliders
                .iter()
                .all(|c| c.shape != ShapeType::ConvexHull as u32),
            "MetalSim: convex hulls are not wired yet"
        );
        write(
            &self.bufs[2],
            u64::from(start) * std::mem::size_of::<Collider>() as u64,
            bytemuck::cast_slice(colliders),
        );
        self.host_cols.extend_from_slice(colliders);
        self.collider_count += count;
        start..self.collider_count
    }

    fn reuse_on(&self) -> bool {
        (self.params.flags & FLAG_REUSE_CONTACTS) != 0
    }

    fn step_modes(&self) -> StepModes {
        let reuse = self.reuse_on();
        StepModes {
            manifold_insert: self.modes.manifold_insert && reuse,
            ..self.modes
        }
    }

    fn record_hash(&self) -> bool {
        self.reuse_on()
    }

    fn step_inputs(&self, m: &StepModes) -> StepInputs {
        let prev_n = self
            .last_counters
            .contact_count
            .min(self.capacity.max_contacts)
            .max(256);
        StepInputs {
            body_count: self.body_count,
            collider_count: self.collider_count,
            joint_count: 0,
            joint_hash_slots: engine_schema::joint_hash_size(0),
            has_convex: false,
            reuse_contacts: self.reuse_on(),
            max_colors: self.capacity.max_colors,
            iterations: self.params.iterations,
            sleep_time: self.params.sleep_time,
            sleep_island_rounds: self.sleep_island_rounds,
            grid_hash_words: rigid_plan::grid_hash_words(
                &self.capacity,
                self.collider_count,
                m.grid_split,
                false,
            ),
            warm_hash_words: u64::from(rigid_plan::warm_hash_slots(
                &self.capacity,
                self.body_count,
                prev_n,
                self.record_hash(),
            )) * if self.record_hash() { 4 } else { 1 },
            steps_done: self.steps_done,
            last: self.last_counters,
            vehicle_count: 0,
            wheel_count: 0,
            motion: rigid_plan::MotionInputs {
                look_ahead: self.params.flags & engine_schema::FLAG_SPEC_VELOCITY != 0,
                ..self.motion
            },
            soft: Default::default(),
            command_count: 0,
        }
    }

    /// Encode and commit one step without waiting.
    pub fn step_submit(&mut self) -> Result<(), ExecError> {
        // Skip only when the counters describe the latest step: pipelined steps
        // leave them stale until `drain`.
        if self.steps_done > 0
            && self.counted_steps == self.steps_done
            && self.last_counters.awake_count == 0
            && self.body_count > 0
        {
            self.counted_steps += 1;
            self.steps_done += 1;
            return Ok(());
        }
        let (cell, levels) = rigid_plan::grid_cell(&self.host_cols, &self.host_bodies, false);
        self.params.grid_cell_size = cell;
        self.grid_levels = levels;
        // A ring slot is reused only after the step that last used it finished.
        while self.in_flight.len() >= RING {
            let cb = self.in_flight.pop_front().unwrap();
            cb.wait_until_completed();
        }
        let m = self.step_modes();
        let i = self.step_inputs(&m);
        let s = rigid_plan::host_words(
            &self.capacity,
            &m,
            &i,
            &rigid_plan::HostState {
                contacts_parity: self.contacts_parity,
                grid_levels: self.grid_levels,
                has_filters: false,
                record_hash: self.record_hash(),
                np_defer_limit: 0,
                persist_list_limit: 0,
            },
        );
        let base = self.ring as u64 * SLOT_BYTES;
        self.ring = (self.ring + 1) % RING;
        write(&self.upload, base + UP_SCRATCH, bytemuck::cast_slice(&s));
        let host = rigid_plan::host_uniforms(&self.params, &s);
        write(&self.upload, base + UP_HOST, bytemuck::bytes_of(&host));
        write(
            &self.upload,
            base + UP_COUNTERS,
            bytemuck::bytes_of(&rigid_plan::step_counters(&i)),
        );
        write(&self.upload, base + UP_COLORS, &[0u8; 512]);
        let cmds = rigid_plan::encode_step(&m, &i, false, true);

        let mut cb = self.queue.new_command_buffer().to_owned();
        let blit = cb.new_blit_command_encoder();
        // Host words and uniform first: the plan's copies then forward last
        // step's GPU counts into the uniform and reset the counters.
        blit.copy_from_buffer(&self.upload, base + UP_SCRATCH, &self.bufs[21], 0, 256);
        blit.copy_from_buffer(
            &self.upload,
            base + UP_HOST,
            &self.bufs[6],
            0,
            std::mem::size_of::<rigid_plan::HostUniforms>() as u64,
        );
        blit.copy_from_buffer(&self.upload, base + UP_COLORS, &self.bufs[17], 0, 512);
        for c in &cmds {
            if let Cmd::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } = *c
            {
                let (from, offset) = if src == COUNTER_UPLOAD {
                    (&self.upload, base + UP_COUNTERS + u64::from(src_offset))
                } else {
                    (&self.bufs[src as usize], u64::from(src_offset))
                };
                blit.copy_from_buffer(
                    from,
                    offset,
                    &self.bufs[dst as usize],
                    u64::from(dst_offset),
                    u64::from(size),
                );
            }
        }
        blit.end_encoding();
        let begin = |cb: &CommandBufferRef| {
            let desc = ComputePassDescriptor::new();
            desc.set_dispatch_type(MTLDispatchType::Serial);
            cb.compute_command_encoder_with_descriptor(desc).to_owned()
        };
        let mut e = begin(&cb);
        let mut st = EncodeState::new();
        for c in &cmds {
            match *c {
                Cmd::Copy { .. } => {}
                // Profiling: one command buffer per phase, timed by the GPU.
                Cmd::Phase { name } => {
                    if self.profiling {
                        e.end_encoding();
                        cb.commit();
                        self.profile_cbs.push((name, cb));
                        cb = self.queue.new_command_buffer().to_owned();
                        e = begin(&cb);
                        st = EncodeState::new();
                    }
                }
                Cmd::Dispatch {
                    kernel,
                    threads,
                    pass,
                } => {
                    let k = self.bind(&e, &mut st, kernel, pass, false)?;
                    let (x, y) = groups(threads);
                    e.dispatch_thread_groups(MTLSize::new(x, y, 1), Self::threads(k));
                }
                Cmd::Indirect { kernel, slot, pass } => {
                    let k = self.bind(&e, &mut st, kernel, pass, true)?;
                    e.dispatch_thread_groups_indirect(
                        &self.bufs[23],
                        u64::from(slot) * 12,
                        Self::threads(k),
                    );
                }
            }
        }
        e.end_encoding();
        cb.commit();
        self.in_flight.push_back(cb);
        self.contacts_parity ^= 1;
        self.steps_done = self.steps_done.saturating_add(1);
        Ok(())
    }

    fn threads(k: &RigidKernel) -> MTLSize {
        MTLSize::new(
            u64::from(k.threads[0]),
            u64::from(k.threads[1]),
            u64::from(k.threads[2]),
        )
    }

    /// Kernel table index of a logical plan name (resolved once per name).
    fn kernel_index(&self, logical: &'static str) -> Result<usize, ExecError> {
        if let Some(&idx) = self.kernel_cache.borrow().get(logical) {
            return Ok(idx);
        }
        let entry = rigid_plan::resolve_entry(logical, true);
        let idx = RIGID_KERNELS
            .iter()
            .position(|k| k.entry == entry)
            .ok_or_else(|| ExecError::Backend(format!("MetalSim: no kernel {entry}")))?;
        self.kernel_cache.borrow_mut().insert(logical, idx);
        Ok(idx)
    }

    /// Bind a kernel's pipeline and buffers, skipping whatever the encoder
    /// already holds (each objc call costs ~1 µs; a step makes hundreds).
    fn bind(
        &self,
        e: &ComputeCommandEncoderRef,
        st: &mut EncodeState,
        logical: &'static str,
        pass: u32,
        indirect: bool,
    ) -> Result<&'static RigidKernel, ExecError> {
        let idx = self.kernel_index(logical)?;
        let k = &RIGID_KERNELS[idx];
        if st.pipeline != Some(idx) {
            e.set_compute_pipeline_state(&self.pipes[idx]);
            st.pipeline = Some(idx);
        }
        for &(index, slot) in k.binds {
            let want = if slot == 22 {
                (slot, u64::from(pass) * PASS_ALIGN)
            } else if slot == 23 && indirect {
                (u32::MAX, 0)
            } else {
                (slot, 0)
            };
            let at = index as usize;
            if st.bound[at] == Some(want) {
                continue;
            }
            st.bound[at] = Some(want);
            if want.0 == u32::MAX {
                // Never bind the indirect buffer as a resource of the dispatch it sizes.
                e.set_buffer(u64::from(index), None, 0);
            } else {
                e.set_buffer(u64::from(index), Some(&self.bufs[slot as usize]), want.1);
            }
        }
        Ok(k)
    }

    fn wait_all(&mut self) {
        while let Some(cb) = self.in_flight.pop_front() {
            cb.wait_until_completed();
        }
    }

    /// Wait for every submitted step and read the counters in place.
    pub fn drain(&mut self) -> Result<StepCounters, ExecError> {
        self.wait_all();
        let c: StepCounters = unsafe { *(self.bufs[7].contents() as *const StepCounters) };
        self.last_counters = c;
        self.counted_steps = self.steps_done;
        if c.overflow_flags != 0 {
            return Err(ExecError::Overflow(c.overflow_flags));
        }
        Ok(c)
    }

    /// One step, waiting for it (tests, probes).
    pub fn step(&mut self) -> Result<StepCounters, ExecError> {
        self.step_submit()?;
        self.drain()
    }

    /// One step with GPU time per phase (ms, summed over the iterations' primal
    /// and dual segments). Splitting the step adds a little submission cost.
    pub fn step_profiled(&mut self) -> Result<Vec<(&'static str, f64)>, ExecError> {
        self.wait_all();
        self.profiling = true;
        self.profile_cbs.clear();
        let r = self.step_submit();
        self.profiling = false;
        r?;
        self.drain()?;
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        for (name, cb) in self.profile_cbs.drain(..) {
            cb.wait_until_completed();
            let ms = gpu_ms(&cb);
            match out.iter_mut().find(|(n, _)| *n == name) {
                Some(slot) => slot.1 += ms,
                None => out.push((name, ms)),
            }
        }
        Ok(out)
    }

    pub fn download_bodies(&mut self) -> Vec<BodyState> {
        self.wait_all();
        let n = self.body_count as usize;
        unsafe {
            std::slice::from_raw_parts(self.bufs[0].contents() as *const BodyState, n).to_vec()
        }
    }
}

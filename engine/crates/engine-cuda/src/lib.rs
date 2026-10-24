//! `CudaSim`: CUDA replay of the shared rigid step graph.
//!
//! Same buffers, host words and command list as the WebGPU and Metal hosts
//! (`engine_exec::rigid_plan`), with kernels compiled Slang → CUDA → PTX in the
//! browser's `ENGINE_WGSL` layout (build.rs), so a measured win lands on every
//! host at once. Each kernel ships as a fatbin (SASS for T4 and Ada, PTX for
//! the rest), so supported GPUs load it without JIT. Slang gathers each kernel's globals into one `__constant__
//! SLANG_globalParams` block; the host fills it once per module with device
//! pointers and rewrites only the pass-uniform pointer between dispatches.
//!
//! CUDA has no indirect dispatch: the host reads the 12-byte work/colour slots
//! of binding 23 back after a kernel that may have written them, then launches
//! the counted grid. Every other step input is forwarded on the GPU, as on Metal.
//!
//! Needs only the NVIDIA driver at run time (`libcuda` is loaded dynamically).
//! Build hosts without nvcc embed no device code; [`CudaSim::available`] is then false.

use cudarc::driver::{result, sys, DriverError};
use engine_exec::rigid_plan::{
    self, Cmd, StepInputs, StepModes, BIND_COUNT, COUNTER_UPLOAD, PASS_ALIGN,
};
use engine_exec::{ExecError, PhaseTimingsMs, PhysicsBackend};
use engine_schema::*;
use std::ffi::{c_void, CString};

pub mod table {
    pub struct CudaKernel {
        pub entry: &'static str,
        /// Bytes of Slang-emitted CUDA C++ (non-zero proves the source built).
        pub cu_bytes: u64,
        /// Fatbin: SASS for [`SASS_ARCHS`] plus PTX for [`PTX_ARCH`]; empty
        /// when the build host had no nvcc.
        pub image: &'static [u8],
        pub threads: [u32; 3],
        pub params_bytes: u32,
        /// The kernel body indexes `gIndirectArgs` (binding 23).
        pub writes_indirect: bool,
        /// (byte offset in `SLANG_globalParams`, field bytes 8|16, shared `BIND(slot)`)
        pub binds: &'static [(u32, u32, u32)],
    }
    include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));
}
pub use table::{CudaKernel, CUDA_KERNELS, PTX_ARCH, SASS_ARCHS};

/// Binding 23: colour slots, work slots and convex slots, three words each.
const INDIRECT_BINDING: u32 = 23;
const PASS_BINDING: u32 = 22;

fn backend(e: DriverError) -> ExecError {
    ExecError::Backend(format!("CUDA: {e:?}"))
}

/// Workgroup grid for `threads` invocations at 64 per group, rows of 32768
/// (must agree with Slang's dispatchIndex).
fn groups(threads: u64) -> (u32, u32) {
    let g = threads.max(1).div_ceil(64);
    (g.min(32768) as u32, g.div_ceil(32768) as u32)
}

struct Kernel {
    module: sys::CUmodule,
    function: sys::CUfunction,
    /// `SLANG_globalParams` in this module's constant bank.
    globals: sys::CUdeviceptr,
    /// Byte offset of the pass-uniform pointer, if the kernel reads it.
    pass_offset: Option<u32>,
    /// The kernel may rewrite indirect slots (binding 23).
    writes_indirect: bool,
    /// Pass slot the constant bank currently points at.
    bound_pass: Option<u32>,
}

pub struct CudaSim {
    ctx: sys::CUcontext,
    stream: sys::CUstream,
    device_name: String,
    kernels: Vec<Kernel>,
    bufs: Vec<sys::CUdeviceptr>,
    sizes: [u64; BIND_COUNT],
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
    last_timings: PhaseTimingsMs,
    steps_done: u32,
    counted_steps: u32,
    contacts_parity: u32,
    /// Host copy of binding 23; stale once a kernel that binds it is queued.
    indirect: Vec<u32>,
    indirect_dirty: bool,
    /// Host readbacks of binding 23 (each one waits for the stream).
    pub indirect_syncs: u64,
    /// `ENGINE_CUDA_SYNC=1`: wait after every launch so a fault names its kernel.
    sync_each: bool,
    /// Last kernel queued (error context: faults surface at the next sync).
    last_kernel: &'static str,
    /// Events at each `Cmd::Phase` of a profiled step.
    profiling: bool,
    phase_events: Vec<(&'static str, sys::CUevent)>,
}

impl CudaSim {
    /// The build embedded device code (nvcc was present), so a driver can run it.
    pub fn has_ptx() -> bool {
        !PTX_ARCH.is_empty() && CUDA_KERNELS.iter().all(|k| !k.image.is_empty())
    }

    /// Device code is embedded and an NVIDIA driver with at least one device loads.
    pub fn available() -> bool {
        Self::has_ptx()
            && std::panic::catch_unwind(|| {
                result::init().is_ok() && result::device::get_count().is_ok_and(|n| n > 0)
            })
            .unwrap_or(false)
    }

    /// Persistent-contact world with every default the WebGPU host ships.
    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        anyhow::ensure!(
            Self::has_ptx(),
            "engine-cuda was built without nvcc: no PTX embedded (see engine/TOOLCHAIN.md)"
        );
        anyhow::ensure!(
            capacity.max_candidates < 0x3fff_ffff,
            "persistent owner index exceeds 30 bits"
        );
        let map = |e: DriverError| anyhow::anyhow!("CUDA: {e:?}");
        // `init` panics when libcuda cannot be loaded; report it as an error.
        std::panic::catch_unwind(result::init)
            .map_err(|_| anyhow::anyhow!("CUDA: libcuda could not be loaded (no NVIDIA driver)"))?
            .map_err(map)?;
        let dev = result::device::get(0).map_err(map)?;
        let device_name = result::device::get_name(dev).map_err(map)?;
        let ctx = unsafe { result::primary_ctx::retain(dev) }.map_err(map)?;
        unsafe { result::ctx::set_current(ctx) }.map_err(map)?;
        let stream =
            result::stream::create(result::stream::StreamKind::NonBlocking).map_err(map)?;

        let sizes = rigid_plan::buffer_sizes(&capacity, true, true);
        let mut bufs = Vec::with_capacity(BIND_COUNT);
        for &size in &sizes {
            let bytes = size.max(16) as usize;
            let ptr = unsafe { result::malloc_sync(bytes) }.map_err(map)?;
            unsafe { result::memset_d8_sync(ptr, 0, bytes) }.map_err(map)?;
            bufs.push(ptr);
        }
        unsafe {
            result::memcpy_htod_sync(bufs[PASS_BINDING as usize], &rigid_plan::pass_table())
                .map_err(map)?;
            let mats = [Material {
                friction: 0.5,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            }; 64];
            result::memcpy_htod_sync(bufs[3], bytemuck::cast_slice::<_, u8>(&mats)).map_err(map)?;
            // Every arena starts with the unit cube that box-versus-hull pairs reuse.
            let arena = engine_schema::convex::HullArena::new();
            result::memcpy_htod_sync(bufs[18], bytemuck::cast_slice::<_, u8>(arena.records()))
                .map_err(map)?;
        }

        let mut kernels = Vec::with_capacity(CUDA_KERNELS.len());
        for k in CUDA_KERNELS {
            let module = unsafe { result::module::load_data(k.image.as_ptr() as *const c_void) }
                .map_err(|e| anyhow::anyhow!("CUDA: load module {}: {e:?}", k.entry))?;
            let function = unsafe { result::module::get_function(module, CString::new(k.entry)?) }
                .map_err(|e| anyhow::anyhow!("CUDA: function {}: {e:?}", k.entry))?;
            let mut globals = 0;
            let mut pass_offset = None;
            if k.params_bytes > 0 {
                let (ptr, bytes) = unsafe {
                    result::module::get_global(module, CString::new("SLANG_globalParams")?)
                }
                .map_err(|e| anyhow::anyhow!("CUDA: {} globals: {e:?}", k.entry))?;
                anyhow::ensure!(
                    bytes >= k.params_bytes as usize,
                    "{}: SLANG_globalParams is {bytes} bytes, reflection says {}",
                    k.entry,
                    k.params_bytes
                );
                // Fill the bank once: (pointer, element count) or a bare pointer per global.
                let mut bank = vec![0u8; bytes];
                for &(offset, size, slot) in k.binds {
                    let o = offset as usize;
                    bank[o..o + 8].copy_from_slice(&bufs[slot as usize].to_le_bytes());
                    if size == 16 {
                        // Kernels never query dimensions; the word count bounds any debug check.
                        bank[o + 8..o + 16]
                            .copy_from_slice(&(sizes[slot as usize] / 4).to_le_bytes());
                    }
                    if slot == PASS_BINDING {
                        pass_offset = Some(offset);
                    }
                }
                unsafe { result::memcpy_htod_sync(ptr, &bank) }.map_err(map)?;
                globals = ptr;
            }
            kernels.push(Kernel {
                module,
                function,
                globals,
                pass_offset,
                writes_indirect: k.writes_indirect,
                bound_pass: pass_offset.map(|_| 0),
            });
        }

        // Shipped defaults; the WebGPU host's `ENGINE_*` control values restore
        // each older path (ENGINE_AUDIT P0), so a CUDA difference can be bisected.
        let control = |name: &str, values: &[&str]| {
            std::env::var(name).is_ok_and(|v| values.contains(&v.as_str()))
        };
        let reuse = !control("ENGINE_REUSE", &["pass"]);
        let modes = StepModes {
            compact_contacts: true,
            persistent_contacts: true,
            manifold_insert: !control("ENGINE_HASH_INSERT", &["contact"]),
            copy_reuse: reuse,
            split_np: reuse && !control("ENGINE_NP", &["fused"]),
            listed_persist: reuse && !control("ENGINE_PERSIST", &["scan"]),
            carry_rest: !control("ENGINE_REST", &["drop"]),
            local_repair: !control("ENGINE_REPAIR", &["rounds"]),
            grid_split: !control("ENGINE_GRID", &["scaled", "fixed"]),
            sparse_start: 32,
            color_headroom: (!control("ENGINE_COLOR_HEADROOM", &["full"])).then_some(4),
            color_round_headroom: (!control("ENGINE_COLOR_ROUND_HEADROOM", &["full"])).then_some(2),
            tail_ok: capacity.max_bodies <= (1 << 16) * 8,
            pose_output: false,
            ..StepModes::default()
        };
        let indirect = vec![0u32; (sizes[INDIRECT_BINDING as usize] / 4) as usize];
        Ok(Self {
            ctx,
            stream,
            device_name,
            kernels,
            bufs,
            sizes,
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
            last_timings: PhaseTimingsMs::default(),
            steps_done: 0,
            counted_steps: 0,
            contacts_parity: 0,
            indirect,
            indirect_dirty: true,
            indirect_syncs: 0,
            sync_each: std::env::var("ENGINE_CUDA_SYNC").as_deref() == Ok("1"),
            last_kernel: "",
            profiling: false,
            phase_events: Vec::new(),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Driver calls act on the calling thread's current context.
    fn bind_ctx(&self) -> Result<(), ExecError> {
        unsafe { result::ctx::set_current(self.ctx) }.map_err(backend)
    }

    fn upload(&self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ExecError> {
        assert!(
            offset + bytes.len() as u64 <= self.sizes[slot].max(16),
            "upload past binding {slot}"
        );
        // Pageable sources are staged before the call returns, so `bytes` may drop.
        unsafe { result::memcpy_htod_async(self.bufs[slot] + offset, bytes, self.stream) }
            .map_err(backend)
    }

    fn download<T: bytemuck::Pod>(&self, slot: usize, count: usize) -> Result<Vec<T>, ExecError> {
        self.bind_ctx()?;
        let mut out = vec![T::zeroed(); count];
        unsafe {
            result::memcpy_dtoh_async(
                bytemuck::cast_slice_mut::<T, u8>(&mut out),
                self.bufs[slot],
                self.stream,
            )
            .map_err(backend)?;
            result::stream::synchronize(self.stream).map_err(backend)?;
        }
        Ok(out)
    }

    pub fn set_params(&mut self, params: AvbdParams) {
        self.params = params;
    }

    pub fn set_materials(&mut self, mats: &[Material]) -> Result<(), ExecError> {
        assert!(mats.len() <= 64, "the material table holds 64 entries");
        self.bind_ctx()?;
        self.upload(3, 0, bytemuck::cast_slice(mats))
    }

    pub fn set_sleep_island_rounds(&mut self, rounds: u32) {
        self.sleep_island_rounds = rounds.min(16);
    }

    /// Append bodies (current and previous halves of binding 0).
    pub fn add_bodies(&mut self, bodies: &[BodyState]) -> std::ops::Range<u32> {
        self.bind_ctx().expect("CUDA context");
        let start = self.body_count;
        let count = bodies.len() as u32;
        assert!(
            count <= self.capacity.max_bodies - start,
            "body upload exceeds capacity"
        );
        let stride = std::mem::size_of::<BodyState>() as u64;
        let bytes = bytemuck::cast_slice(bodies);
        self.upload(0, u64::from(start) * stride, bytes)
            .expect("upload bodies");
        self.upload(
            0,
            u64::from(self.capacity.max_bodies + start) * stride,
            bytes,
        )
        .expect("upload previous bodies");
        self.host_bodies.extend_from_slice(bodies);
        self.motion.ccd |= bodies.iter().any(|b| b.flags & engine_schema::BODY_CCD != 0);
        self.motion.kinematic |= bodies.iter().any(|b| b.flags & 3 == 2);
        self.body_count += count;
        start..self.body_count
    }

    pub fn add_colliders(&mut self, colliders: &[Collider]) -> std::ops::Range<u32> {
        self.bind_ctx().expect("CUDA context");
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
            "CudaSim: convex hulls are not wired yet"
        );
        self.upload(
            2,
            u64::from(start) * std::mem::size_of::<Collider>() as u64,
            bytemuck::cast_slice(colliders),
        )
        .expect("upload colliders");
        self.host_cols.extend_from_slice(colliders);
        self.collider_count += count;
        start..self.collider_count
    }

    fn reuse_on(&self) -> bool {
        (self.params.flags & FLAG_REUSE_CONTACTS) != 0
    }

    fn step_modes(&self) -> StepModes {
        StepModes {
            manifold_insert: self.modes.manifold_insert && self.reuse_on(),
            ..self.modes
        }
    }

    fn step_inputs(&self, m: &StepModes) -> StepInputs {
        let prev_n = self
            .last_counters
            .contact_count
            .min(self.capacity.max_contacts)
            .max(256);
        let record = self.reuse_on();
        StepInputs {
            body_count: self.body_count,
            collider_count: self.collider_count,
            joint_count: 0,
            joint_hash_slots: engine_schema::joint_hash_size(0),
            has_convex: false,
            reuse_contacts: record,
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
                record,
            )) * if record { 4 } else { 1 },
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
            explosion_count: 0,
        }
    }

    fn kernel_index(logical: &'static str) -> Result<usize, ExecError> {
        let entry = rigid_plan::resolve_entry(logical, true);
        CUDA_KERNELS
            .iter()
            .position(|k| k.entry == entry)
            .ok_or_else(|| ExecError::Backend(format!("CudaSim: no kernel {entry}")))
    }

    /// Queue one kernel over a `(x, y)` grid of workgroups.
    fn launch(
        &mut self,
        logical: &'static str,
        grid: (u32, u32, u32),
        pass: u32,
    ) -> Result<(), ExecError> {
        let idx = Self::kernel_index(logical)?;
        let threads = CUDA_KERNELS[idx].threads;
        self.last_kernel = logical;
        let stream = self.stream;
        let pass_ptr = self.bufs[PASS_BINDING as usize] + u64::from(pass) * PASS_ALIGN;
        let k = &mut self.kernels[idx];
        if let Some(offset) = k.pass_offset {
            if k.bound_pass != Some(pass) {
                // Stream-ordered, so earlier launches keep the slot they were queued with.
                unsafe {
                    result::memcpy_htod_async(
                        k.globals + u64::from(offset),
                        &pass_ptr.to_le_bytes(),
                        stream,
                    )
                }
                .map_err(backend)?;
                k.bound_pass = Some(pass);
            }
        }
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        unsafe {
            result::launch_kernel(
                k.function,
                grid,
                (threads[0], threads[1], threads[2]),
                0,
                stream,
                &mut [],
            )
        }
        .map_err(|e| ExecError::Backend(format!("CUDA: launch {logical}: {e:?}")))?;
        if k.writes_indirect {
            self.indirect_dirty = true;
        }
        if self.sync_each {
            unsafe { result::stream::synchronize(stream) }.map_err(|e| {
                ExecError::Backend(format!(
                    "CUDA: {logical} (pass {pass}, grid {grid:?}) at step {}: {e:?}",
                    self.steps_done
                ))
            })?;
        }
        Ok(())
    }

    /// Work-group counts of indirect slot `slot`, read back only when a queued
    /// kernel may have changed binding 23 since the last read.
    fn indirect_groups(&mut self, slot: u32) -> Result<(u32, u32, u32), ExecError> {
        if self.indirect_dirty {
            let mut words = std::mem::take(&mut self.indirect);
            unsafe {
                result::memcpy_dtoh_async(
                    bytemuck::cast_slice_mut::<u32, u8>(&mut words),
                    self.bufs[INDIRECT_BINDING as usize],
                    self.stream,
                )
                .map_err(backend)?;
                result::stream::synchronize(self.stream).map_err(backend)?;
            }
            self.indirect = words;
            self.indirect_dirty = false;
            self.indirect_syncs += 1;
        }
        let o = slot as usize * 3;
        Ok((self.indirect[o], self.indirect[o + 1], self.indirect[o + 2]))
    }

    /// Encode and queue one step without waiting (except for indirect counts).
    pub fn step_submit(&mut self) -> Result<(), ExecError> {
        self.bind_ctx()?;
        // Skip only when the counters describe the latest step.
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
                record_hash: self.reuse_on(),
                np_defer_limit: 0,
                persist_list_limit: 0,
            },
        );
        // Host words and uniform first: the plan's copies then forward last
        // step's GPU counts into the uniform and reset the counters.
        self.upload(21, 0, bytemuck::cast_slice(&s))?;
        let host = rigid_plan::host_uniforms(&self.params, &s);
        self.upload(6, 0, bytemuck::bytes_of(&host))?;
        self.upload(17, 0, &[0u8; 512])?;
        let counters = rigid_plan::step_counters(&i);
        let cmds = rigid_plan::encode_step(&m, &i, false, true);
        for c in &cmds {
            if let Cmd::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } = *c
            {
                if src == COUNTER_UPLOAD {
                    let bytes = bytemuck::bytes_of(&counters);
                    let range = src_offset as usize..(src_offset + size) as usize;
                    self.upload(dst as usize, u64::from(dst_offset), &bytes[range])?;
                } else {
                    unsafe {
                        result::memcpy_dtod_async(
                            self.bufs[dst as usize] + u64::from(dst_offset),
                            self.bufs[src as usize] + u64::from(src_offset),
                            size as usize,
                            self.stream,
                        )
                    }
                    .map_err(backend)?;
                }
                if dst == INDIRECT_BINDING {
                    self.indirect_dirty = true;
                }
            }
        }
        for c in &cmds {
            match *c {
                Cmd::Copy { .. } => {}
                Cmd::Phase { name } => {
                    if self.profiling {
                        let ev = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
                            .map_err(backend)?;
                        unsafe { result::event::record(ev, self.stream) }.map_err(backend)?;
                        self.phase_events.push((name, ev));
                    }
                }
                Cmd::Dispatch {
                    kernel,
                    threads,
                    pass,
                } => {
                    let (x, y) = groups(threads);
                    self.launch(kernel, (x, y, 1), pass)?;
                }
                Cmd::Indirect { kernel, slot, pass } => {
                    let g = self.indirect_groups(slot)?;
                    self.launch(kernel, g, pass)?;
                }
            }
            // Diagnostics: `ENGINE_CUDA_STOP_AFTER=kernel@step` ends that step
            // right after the kernel so the counters show its output.
            if let Ok(spec) = std::env::var("ENGINE_CUDA_STOP_AFTER") {
                if let Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } = *c {
                    if spec == format!("{kernel}@{}", self.steps_done) {
                        break;
                    }
                }
            }
        }
        self.contacts_parity ^= 1;
        self.steps_done = self.steps_done.saturating_add(1);
        Ok(())
    }

    /// Wait for every queued step and read the counters.
    pub fn drain(&mut self) -> Result<StepCounters, ExecError> {
        let c: StepCounters = self.download::<StepCounters>(7, 1)?[0];
        self.last_counters = c;
        self.counted_steps = self.steps_done;
        if c.overflow_flags != 0 {
            return Err(ExecError::Overflow(c.overflow_flags));
        }
        Ok(c)
    }

    /// One step, waiting for it (tests, probes).
    pub fn step(&mut self) -> Result<StepCounters, ExecError> {
        let r = self.step_submit().and_then(|()| self.drain());
        r.map_err(|e| match e {
            ExecError::Backend(msg) => ExecError::Backend(format!(
                "{msg} (step {}, last kernel queued: {})",
                self.steps_done, self.last_kernel
            )),
            other => other,
        })
    }

    /// One step with GPU time per phase (ms, summed over the iterations'
    /// segments), from events recorded at each `Cmd::Phase`.
    pub fn step_profiled(&mut self) -> Result<Vec<(&'static str, f64)>, ExecError> {
        self.bind_ctx()?;
        let start = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(backend)?;
        unsafe { result::event::record(start, self.stream) }.map_err(backend)?;
        self.profiling = true;
        let r = self.step_submit();
        self.profiling = false;
        r?;
        self.drain()?;
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        let mut prev = start;
        for (name, ev) in std::mem::take(&mut self.phase_events) {
            let ms = f64::from(unsafe { result::event::elapsed(prev, ev) }.map_err(backend)?);
            match out.iter_mut().find(|(n, _)| *n == name) {
                Some(slot) => slot.1 += ms,
                None => out.push((name, ms)),
            }
            unsafe { result::event::destroy(prev) }.map_err(backend)?;
            prev = ev;
        }
        unsafe { result::event::destroy(prev) }.map_err(backend)?;
        self.last_timings.total_gpu = out.iter().map(|(_, ms)| ms).sum();
        Ok(out)
    }

    /// The first `words` 32-bit words of binding `slot` (diagnostics).
    pub fn download_words(&mut self, slot: usize, words: usize) -> Vec<u32> {
        let words = words.min((self.sizes[slot] / 4) as usize);
        self.download(slot, words).expect("download binding")
    }

    pub fn download_bodies(&mut self) -> Vec<BodyState> {
        self.download(0, self.body_count as usize)
            .expect("download bodies")
    }
}

impl Drop for CudaSim {
    fn drop(&mut self) {
        unsafe {
            if result::ctx::set_current(self.ctx).is_err() {
                return;
            }
            let _ = result::stream::synchronize(self.stream);
            for k in &self.kernels {
                let _ = result::module::unload(k.module);
            }
            for &b in &self.bufs {
                let _ = result::free_sync(b);
            }
            let _ = result::stream::destroy(self.stream);
        }
    }
}

impl PhysicsBackend for CudaSim {
    fn name(&self) -> &'static str {
        "cuda"
    }
    fn capacity(&self) -> &WorldCapacity {
        &self.capacity
    }
    fn params(&self) -> &AvbdParams {
        &self.params
    }
    fn set_params(&mut self, params: AvbdParams) {
        self.params = params;
    }
    fn step(&mut self) -> Result<StepCounters, ExecError> {
        CudaSim::step(self)
    }
    fn last_timings(&self) -> PhaseTimingsMs {
        self.last_timings.clone()
    }
    fn last_counters(&self) -> StepCounters {
        self.last_counters
    }
}

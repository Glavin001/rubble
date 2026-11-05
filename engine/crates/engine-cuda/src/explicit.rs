//! `CudaExplicitBeams`: CUDA host of the explicit node–beam solver.
//!
//! The same Slang kernels (`slang/kernels/explicit_beams.slang`, Slang → CUDA →
//! fatbin by build.rs `explicit_kernels` from `engine_exec::explicit_plan::MODULES`),
//! the same buffer contents ([`ExplicitUpload`]) and the same per-frame command
//! list ([`FramePlanner`]) as the WebGPU and Metal hosts: this file only allocates
//! device memory, fills each module's `SLANG_globalParams` bank and replays the
//! commands on one stream.
//!
//! CUDA has no indirect dispatch: after a frame whose contact plan passes ran, the
//! host reads the 32-byte indirect arguments back (one wait per plan, every ~34
//! steps) and launches the counted grids. The wide passes' per-step parameter
//! copies are a pointer rewrite in the bank, stream-ordered between launches.
//! The WGSL loop cap is a WGSL post-process: CUDA kernels run uncapped.

use crate::backend;
use cudarc::driver::{result, sys};
use engine_exec::explicit_plan::{
    self as plan, coupling, Cmd, CouplingCapacity, CouplingLayout, ExplicitLayout, ExplicitUpload, FramePlanner,
    GpuBeam, Kernel, BIND_ARGS, BIND_COUNT, BIND_PARAMS, PARAM_STRIDE,
};
use engine_softbody::explicit_rigid::ProxySet;
use engine_exec::ExecError;
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::Vec3;
use std::collections::HashMap;
use std::ffi::{c_void, CString};

pub mod table {
    pub struct ExplicitCudaKernel {
        /// The plan module (`engine_exec::explicit_plan::MODULES`).
        pub name: &'static str,
        pub entry: &'static str,
        /// Bytes of Slang-emitted CUDA C++ (non-zero proves the source built).
        pub cu_bytes: u64,
        /// Fatbin (empty when the build host had no nvcc).
        pub image: &'static [u8],
        pub threads: [u32; 3],
        pub params_bytes: u32,
        /// (byte offset in `SLANG_globalParams`, field bytes 8|16, plan binding)
        pub binds: &'static [(u32, u32, u32)],
    }
    include!(concat!(env!("OUT_DIR"), "/cuda_explicit_kernels.rs"));
}
pub use table::{ExplicitCudaKernel, CUDA_EXPLICIT_KERNELS};

struct Kernel_ {
    module: sys::CUmodule,
    function: sys::CUfunction,
    globals: sys::CUdeviceptr,
    /// Byte offset of the parameters pointer in the bank, and the copy it points at.
    params: Option<(u32, u32)>,
    threads: u32,
}

pub struct CudaExplicitBeams {
    ctx: sys::CUcontext,
    stream: sys::CUstream,
    device_name: String,
    layout: ExplicitLayout,
    planner: FramePlanner,
    bufs: Vec<sys::CUdeviceptr>,
    sizes: Vec<u64>,
    kernels: HashMap<Kernel, Kernel_>,
    /// Host copy of the indirect arguments; stale once a plan pass is queued.
    args: [u32; 8],
    args_dirty: bool,
    /// Host reads of the indirect arguments (each one waits for the stream).
    pub args_syncs: u64,
    /// A frame queued since the last wait copied the status.
    status_due: bool,
    /// Events around timed frames ([`Self::enable_frame_timing`]).
    timing_left: u32,
    events: Vec<(sys::CUevent, sys::CUevent)>,
}

impl CudaExplicitBeams {
    /// The build embedded device code for every explicit module (nvcc was present).
    pub fn has_ptx() -> bool {
        !crate::PTX_ARCH.is_empty() && CUDA_EXPLICIT_KERNELS.iter().all(|k| !k.image.is_empty())
    }

    /// Device code is embedded and an NVIDIA driver with a device loads.
    pub fn available() -> bool {
        Self::has_ptx() && crate::CudaSim::available()
    }

    /// Upload `vehicles` (their current state) on device 0.
    pub fn new(vehicles: &[ExplicitBeams]) -> anyhow::Result<Self> {
        Self::with_coupling(vehicles, None)
    }

    /// [`Self::new`] with room for rigid-body proxies per vehicle (the coupling,
    /// `explicit_beams_rigid.slang`); off until [`Self::set_coupling_on`].
    pub fn with_coupling(vehicles: &[ExplicitBeams], cap: Option<CouplingCapacity>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            Self::has_ptx(),
            "engine-cuda was built without nvcc: no explicit-beams PTX embedded (see engine/TOOLCHAIN.md)"
        );
        let map = |e: cudarc::driver::DriverError| anyhow::anyhow!("CUDA: {e:?}");
        std::panic::catch_unwind(result::init)
            .map_err(|_| anyhow::anyhow!("CUDA: libcuda could not be loaded (no NVIDIA driver)"))?
            .map_err(map)?;
        let dev = result::device::get(0).map_err(map)?;
        let device_name = result::device::get_name(dev).map_err(map)?;
        let ctx = unsafe { result::primary_ctx::retain(dev) }.map_err(map)?;
        unsafe { result::ctx::set_current(ctx) }.map_err(map)?;
        let stream = result::stream::create(result::stream::StreamKind::NonBlocking).map_err(map)?;

        let upload = ExplicitUpload::new(vehicles, cap);
        let layout = upload.layout.clone();
        let mut bufs = Vec::with_capacity(BIND_COUNT);
        let mut sizes = Vec::with_capacity(BIND_COUNT);
        for b in 0..BIND_COUNT as u32 {
            let bytes = upload.bytes(b);
            let ptr = unsafe { result::malloc_sync(bytes.len()) }.map_err(map)?;
            unsafe { result::memcpy_htod_sync(ptr, &bytes) }.map_err(map)?;
            bufs.push(ptr);
            sizes.push(bytes.len() as u64);
        }

        let mut wanted = vec![Kernel::Frame, Kernel::Solo, Kernel::Contact];
        wanted.extend((0..plan::VV_PASSES.len()).map(Kernel::Vv));
        if layout.wide_ok {
            wanted.extend((0..plan::WIDE_PASSES.len()).map(Kernel::Wide));
        }
        let mut kernels = HashMap::new();
        for k in wanted {
            let name = k.module(layout.aero, layout.lanes);
            let t = CUDA_EXPLICIT_KERNELS
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| anyhow::anyhow!("explicit kernel {name} not built"))?;
            let module = unsafe { result::module::load_data(t.image.as_ptr() as *const c_void) }
                .map_err(|e| anyhow::anyhow!("CUDA: load module {name}: {e:?}"))?;
            let function = unsafe { result::module::get_function(module, CString::new(t.entry)?) }
                .map_err(|e| anyhow::anyhow!("CUDA: function {name}: {e:?}"))?;
            let (globals, bytes) =
                unsafe { result::module::get_global(module, CString::new("SLANG_globalParams")?) }
                    .map_err(|e| anyhow::anyhow!("CUDA: {name} globals: {e:?}"))?;
            anyhow::ensure!(
                bytes >= t.params_bytes as usize,
                "{name}: SLANG_globalParams is {bytes} bytes, reflection says {}",
                t.params_bytes
            );
            // Fill the bank once: (pointer, element count) or a bare pointer per global.
            let mut bank = vec![0u8; bytes];
            let mut params = None;
            for &(offset, size, slot) in t.binds {
                let o = offset as usize;
                bank[o..o + 8].copy_from_slice(&bufs[slot as usize].to_le_bytes());
                if size == 16 {
                    bank[o + 8..o + 16].copy_from_slice(&(sizes[slot as usize] / 4).to_le_bytes());
                }
                if slot == BIND_PARAMS {
                    params = Some((offset, 0));
                }
            }
            unsafe { result::memcpy_htod_sync(globals, &bank) }.map_err(map)?;
            kernels.insert(
                k,
                Kernel_ {
                    module,
                    function,
                    globals,
                    params,
                    threads: t.threads[0],
                },
            );
        }
        let planner = FramePlanner::new(&layout);
        Ok(Self {
            ctx,
            stream,
            device_name,
            layout,
            planner,
            bufs,
            sizes,
            kernels,
            args: [0; 8],
            args_dirty: false,
            args_syncs: 0,
            status_due: false,
            timing_left: 0,
            events: Vec::new(),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn layout(&self) -> &ExplicitLayout {
        &self.layout
    }

    fn bind_ctx(&self) -> Result<(), ExecError> {
        unsafe { result::ctx::set_current(self.ctx) }.map_err(backend)
    }

    pub fn set_vehicle_contact(&mut self, on: bool) {
        self.planner.set_vehicle_contact(on);
    }

    pub fn contact_active(&self) -> bool {
        self.planner.contact_active()
    }

    pub fn set_wide(&mut self, on: bool) -> bool {
        self.planner.set_wide(on)
    }

    /// Record the GPU time of each of the next `frames` frames ([`Self::frame_times_ms`]).
    pub fn enable_frame_timing(&mut self, frames: u32) {
        self.timing_left = frames;
    }

    /// GPU milliseconds of each timed frame (waits); timing restarts empty.
    pub fn frame_times_ms(&mut self) -> Result<Vec<f64>, ExecError> {
        self.wait()?;
        let mut out = Vec::new();
        for (a, b) in std::mem::take(&mut self.events) {
            out.push(f64::from(unsafe { result::event::elapsed(a, b) }.map_err(backend)?));
            unsafe {
                let _ = result::event::destroy(a);
                let _ = result::event::destroy(b);
            }
        }
        self.timing_left = 0;
        Ok(out)
    }

    /// Queue one frame of `steps` steps (the plan's commands). Does not wait, except
    /// to read the indirect arguments after a contact plan.
    pub fn frame(&mut self, steps: u32) -> Result<(), ExecError> {
        if self.layout.spans.is_empty() || steps == 0 {
            return Ok(());
        }
        self.bind_ctx()?;
        let bytes = self.planner.write_params(steps);
        // Pageable sources are staged before the call returns; stream order keeps
        // the last frame's kernels reading the old parameters.
        unsafe { result::memcpy_htod_async(self.bufs[BIND_PARAMS as usize], &bytes, self.stream) }
            .map_err(backend)?;
        let timed = self.timing_left > 0;
        let mut stamp = None;
        if timed {
            self.timing_left -= 1;
            let a = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(backend)?;
            unsafe { result::event::record(a, self.stream) }.map_err(backend)?;
            stamp = Some(a);
        }
        for cmd in self.planner.commands(steps, true) {
            match cmd {
                Cmd::Dispatch { kernel, groups, param_copy } => {
                    self.launch(kernel, (groups, 1, 1), param_copy)?;
                    if matches!(kernel, Kernel::Vv(_)) {
                        self.args_dirty = true;
                    }
                }
                Cmd::Indirect { kernel, offset } => {
                    if self.args_dirty {
                        let mut raw = [0u32; 8];
                        unsafe {
                            result::memcpy_dtoh_async(
                                bytemuck::cast_slice_mut::<u32, u8>(&mut raw),
                                self.bufs[BIND_ARGS as usize],
                                self.stream,
                            )
                            .map_err(backend)?;
                            result::stream::synchronize(self.stream).map_err(backend)?;
                        }
                        self.args = raw;
                        self.args_dirty = false;
                        self.args_syncs += 1;
                    }
                    let w = (offset / 4) as usize;
                    let grid = (self.args[w], self.args[w + 1], self.args[w + 2]);
                    self.launch(kernel, grid, 0)?;
                }
                Cmd::Copy { src, src_offset, dst, dst_offset, size } => unsafe {
                    result::memcpy_dtod_async(
                        self.bufs[dst as usize] + dst_offset,
                        self.bufs[src as usize] + src_offset,
                        size as usize,
                        self.stream,
                    )
                    .map_err(backend)?;
                },
            }
        }
        if let Some(a) = stamp {
            let b = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(backend)?;
            unsafe { result::event::record(b, self.stream) }.map_err(backend)?;
            self.events.push((a, b));
        }
        if self.planner.status_due() {
            self.status_due = true;
        }
        Ok(())
    }

    fn launch(&mut self, kernel: Kernel, grid: (u32, u32, u32), param_copy: u32) -> Result<(), ExecError> {
        let stream = self.stream;
        let params = self.bufs[BIND_PARAMS as usize];
        let k = self
            .kernels
            .get_mut(&kernel)
            .ok_or_else(|| ExecError::Backend(format!("CUDA: explicit kernel {kernel:?} not loaded")))?;
        if let Some((offset, bound)) = &mut k.params {
            if *bound != param_copy {
                let ptr = params + u64::from(param_copy) * PARAM_STRIDE;
                // Stream-ordered: earlier launches keep the copy they were queued with.
                unsafe { result::memcpy_htod_async(k.globals + u64::from(*offset), &ptr.to_le_bytes(), stream) }
                    .map_err(backend)?;
                *bound = param_copy;
            }
        }
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        unsafe { result::launch_kernel(k.function, grid, (k.threads, 1, 1), 0, stream, &mut []) }
            .map_err(|e| ExecError::Backend(format!("CUDA: launch {kernel:?}: {e:?}")))
    }

    fn upload(&self, b: u32, offset: u64, bytes: &[u8]) -> Result<(), ExecError> {
        assert!(offset + bytes.len() as u64 <= self.sizes[b as usize], "upload past binding {b}");
        self.bind_ctx()?;
        unsafe { result::memcpy_htod_async(self.bufs[b as usize] + offset, bytes, self.stream) }.map_err(backend)
    }

    pub fn coupling(&self) -> Option<CouplingLayout> {
        self.layout.coupling
    }

    /// Turn the rigid coupling on or off from the next frame (frames then run per
    /// vehicle: the wide passes do not run it).
    pub fn set_coupling_on(&mut self, on: bool) -> Result<(), ExecError> {
        assert!(self.layout.coupling.is_some(), "explicit beams: built without coupling room");
        if on {
            self.set_wide(false);
        }
        self.upload(plan::BIND_MISC, self.layout.coupling_on_offset(), bytemuck::bytes_of(&(on as u32)))
    }

    /// Write vehicle `v`'s proxies (its accumulators cleared) for the next frames.
    pub fn write_proxies(&mut self, v: usize, set: &ProxySet) -> Result<(), ExecError> {
        let layout = self.layout.coupling.ok_or_else(|| ExecError::Backend("explicit coupling: no room".into()))?;
        let recs = coupling::proxy_records(&layout, set).map_err(ExecError::Backend)?;
        self.upload(plan::BIND_MISC, u64::from(layout.vehicle(v as u32)) * 16, bytemuck::cast_slice(&recs))?;
        let zeros = vec![0i32; layout.acc_per_vehicle() as usize];
        self.upload(plan::BIND_ACC, u64::from(layout.acc(v as u32)) * 4, bytemuck::cast_slice(&zeros))
    }

    /// Read vehicle `v`'s proxies back into `set` (waits).
    pub fn read_proxies(&mut self, v: usize, set: &mut ProxySet) -> Result<(), ExecError> {
        let layout = self.layout.coupling.expect("coupling room");
        self.wait()?;
        let misc: Vec<u32> = self.download(plan::BIND_MISC)?;
        let at = layout.vehicle(v as u32) as usize * 4;
        coupling::decode_proxies(&layout, &misc[at..at + layout.recs() as usize * 4], set);
        Ok(())
    }

    /// Block until queued frames have run; `Err` on a contact error a frame reported.
    pub fn wait(&mut self) -> Result<(), ExecError> {
        self.bind_ctx()?;
        unsafe { result::stream::synchronize(self.stream) }.map_err(backend)?;
        if std::mem::take(&mut self.status_due) {
            let acc = self.download::<i32>(plan::BIND_ACC)?;
            self.layout.contact_status(&acc).map_err(ExecError::Backend)?;
        }
        Ok(())
    }

    fn download<T: bytemuck::Pod>(&self, b: u32) -> Result<Vec<T>, ExecError> {
        self.bind_ctx()?;
        let n = self.sizes[b as usize] as usize / std::mem::size_of::<T>();
        let mut out = vec![T::zeroed(); n];
        unsafe {
            result::memcpy_dtoh_async(bytemuck::cast_slice_mut::<T, u8>(&mut out), self.bufs[b as usize], self.stream)
                .map_err(backend)?;
            result::stream::synchronize(self.stream).map_err(backend)?;
        }
        Ok(out)
    }

    /// Per vehicle, world node positions (waits).
    pub fn positions(&mut self) -> Result<Vec<Vec<Vec3>>, ExecError> {
        self.wait()?;
        Ok(self
            .layout
            .positions(&self.download(plan::BIND_NODES)?, &self.download(plan::BIND_MISC)?))
    }

    /// Per vehicle, node velocities (waits).
    pub fn velocities(&mut self) -> Result<Vec<Vec<Vec3>>, ExecError> {
        self.wait()?;
        Ok(self.layout.velocities(&self.download(plan::BIND_NODES)?))
    }

    /// Per vehicle, every beam's broken flag and rest length (waits).
    pub fn beams(&mut self) -> Result<Vec<Vec<GpuBeam>>, ExecError> {
        self.wait()?;
        Ok(self.layout.beams(&self.download(plan::BIND_BEAMS)?))
    }
}

impl Drop for CudaExplicitBeams {
    fn drop(&mut self) {
        unsafe {
            if result::ctx::set_current(self.ctx).is_err() {
                return;
            }
            let _ = result::stream::synchronize(self.stream);
            for (a, b) in self.events.drain(..) {
                let _ = result::event::destroy(a);
                let _ = result::event::destroy(b);
            }
            for k in self.kernels.values() {
                let _ = result::module::unload(k.module);
            }
            for &b in &self.bufs {
                let _ = result::free_sync(b);
            }
            let _ = result::stream::destroy(self.stream);
        }
    }
}

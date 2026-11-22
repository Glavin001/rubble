//! `CudaExplicitBeams`: CUDA host of the explicit node–beam solver.
//!
//! The same Slang kernels (`slang/kernels/explicit_beams.slang`, Slang → CUDA →
//! fatbin by build.rs `explicit_kernels` from `engine_exec::explicit_plan::MODULES` and
//! `split_modules`), the same buffer contents ([`ExplicitUpload`]) and the same frame
//! schedule (`engine_exec::explicit_plan::schedule::frame`: the planner's commands and
//! the split paths' launches) as the WebGPU host: this file only allocates device
//! memory, fills each module's `SLANG_globalParams` bank and replays the launches on
//! one stream.
//!
//! CUDA has no indirect dispatch. Each module has a second entry, `{entry}_gs`
//! (engine-slang-build `grid_stride_entry`), whose blocks walk the workgroups the GPU wrote
//! in the argument buffer: an indirect launch is that entry over a fixed grid (the blocks
//! one launch keeps resident) with a pointer to its counts, queued without waiting. With
//! [`CudaExplicitBeams::set_device_indirect`] off (measurement), the host reads the
//! argument buffer back instead (a wait for the stream) when a launch since may have
//! written it, and launches the counted grid. The wide passes' per-step parameter
//! copies are a pointer rewrite in the bank, stream-ordered between launches.
//! Loop caps: the kernel caps its loops in its Slang source (`EB_LOOP_CAP`), so the
//! CUDA modules have them too; [`CudaExplicitBeams::wait`] fails on the loop-cap
//! flag as on a contact error.

use crate::backend;
use cudarc::driver::{result, sys};
use engine_exec::explicit_plan::schedule::{self, ArgBuf, Grid, Op, FLEET_ARGS_BYTES, SELF_ARGS_BYTES};
use engine_exec::explicit_plan::{
    self as plan, coupling, Cmd, CouplingCapacity, CouplingLayout, ExplicitLayout, ExplicitUpload, FramePlanner,
    GpuBeam, Kernel, PlannerState, BIND_ARGS, BIND_COUNT, BIND_PARAMS, PARAM_STRIDE,
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
pub use table::{ExplicitCudaKernel, CUDA_EXPLICIT_KERNELS, CUDA_LOOP_CAP_TEST};

struct Kernel_ {
    module: sys::CUmodule,
    function: sys::CUfunction,
    /// The indirect entry (`{entry}_gs`) and the blocks a launch of it runs: as many as
    /// the device keeps resident at once (none: a module without it, e.g. another build's
    /// from `RUBBLE_CUDA_EXPLICIT_DIR`; the host reads the counts back).
    indirect: Option<(sys::CUfunction, u32)>,
    globals: sys::CUdeviceptr,
    /// Byte offset of the parameters pointer in the bank, and the copy it points at.
    params: Option<(u32, u32)>,
    threads: u32,
    /// The bank's fields (byte offset, bytes 8|16, binding): bindings 10 and 11 are
    /// rebound per launch on the split paths ([`schedule::SplitLaunch`]).
    binds: &'static [(u32, u32, u32)],
    /// The buffers bindings 10 and 11 hold now.
    bound: [sys::CUdeviceptr; 2],
}

/// Where a launch's kernel comes from: the plan's ([`Kernel`]) or a split path's module.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    Plan(Kernel),
    Split(String),
}

/// Load module `t` and fill its bank from `bufs` (binding 11 none until a launch binds
/// it). `RUBBLE_CUDA_EXPLICIT_DIR=dir` (measurement only) loads `dir/NAME.fatbin` in place
/// of this build's: another build's device code of the same module, same bindings.
fn load_kernel(t: &'static ExplicitCudaKernel, bufs: &[sys::CUdeviceptr], sizes: &[u64], sms: u32) -> anyhow::Result<Kernel_> {
    let name = t.name;
    anyhow::ensure!(
        !t.image.is_empty(),
        "explicit module {name} has no device code in this build: add it to \
         ENGINE_CUDA_EXPLICIT_MODULES (or unset that variable to build every module)"
    );
    let other = std::env::var("RUBBLE_CUDA_EXPLICIT_DIR")
        .ok()
        .filter(|d| !d.is_empty())
        .and_then(|d| {
            let path = std::path::Path::new(&d).join(format!("{name}.fatbin"));
            let read = std::fs::read(&path).ok();
            eprintln!("RUBBLE_CUDA_EXPLICIT_DIR: {name}: {}", if read.is_some() { "from the directory" } else { "this build's (not in the directory)" });
            read
        });
    let image: &[u8] = other.as_deref().unwrap_or(t.image);
    let module = unsafe { result::module::load_data(image.as_ptr() as *const c_void) }
        .map_err(|e| anyhow::anyhow!("CUDA: load module {name}: {e:?}"))?;
    let function = unsafe { result::module::get_function(module, CString::new(t.entry)?) }
        .map_err(|e| anyhow::anyhow!("CUDA: function {name}: {e:?}"))?;
    let indirect = match unsafe {
        result::module::get_function(module, CString::new(format!("{}{}", t.entry, plan_gs_suffix()))?)
    } {
        Ok(f) => {
            let per_sm = unsafe { result::occupancy::max_active_block_per_multiprocessor(f, t.threads[0] as i32, 0) }
                .map_err(|e| anyhow::anyhow!("CUDA: occupancy {name}: {e:?}"))?;
            Some((f, (per_sm.max(1) as u32) * sms.max(1)))
        }
        Err(_) => None,
    };
    let (globals, bytes) = unsafe { result::module::get_global(module, CString::new("SLANG_globalParams")?) }
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
        let (ptr, len) = match bufs.get(slot as usize) {
            Some(&b) => (b, sizes[slot as usize]),
            None => (0, 0),
        };
        bank[o..o + 8].copy_from_slice(&ptr.to_le_bytes());
        if size == 16 {
            bank[o + 8..o + 16].copy_from_slice(&(len / 4).to_le_bytes());
        }
        if slot == BIND_PARAMS {
            params = Some((offset, 0));
        }
    }
    unsafe { result::memcpy_htod_sync(globals, &bank) }.map_err(|e| anyhow::anyhow!("CUDA: {e:?}"))?;
    let at = |slot: usize| bufs.get(slot).copied().unwrap_or(0);
    Ok(Kernel_ { module, function, indirect, globals, params, threads: t.threads[0], binds: t.binds, bound: [at(10), at(11)] })
}

/// The indirect entry's suffix (build.rs: `grid_stride_entry`).
fn plan_gs_suffix() -> &'static str {
    "_gs"
}

/// A launch's grid: workgroups the host knows, or the counts the GPU wrote at a device
/// address (the indirect entry).
#[derive(Clone, Copy)]
enum Launch {
    Groups((u32, u32, u32)),
    Indirect(sys::CUdeviceptr),
}

pub struct CudaExplicitBeams {
    ctx: sys::CUcontext,
    stream: sys::CUstream,
    /// The stream is this host's own (not another host's: [`Self::use_stream`]).
    owns_stream: bool,
    device_name: String,
    layout: ExplicitLayout,
    planner: FramePlanner,
    bufs: Vec<sys::CUdeviceptr>,
    sizes: Vec<u64>,
    kernels: HashMap<Key, Kernel_>,
    /// The split paths' argument buffers (the plan's is binding 10): (pointer, bytes).
    arg_bufs: HashMap<ArgBuf, (sys::CUdeviceptr, u64)>,
    /// Host copies of the argument buffers indirect launches read; one is read again
    /// (a wait) when a launch was queued since.
    args: HashMap<ArgBuf, Vec<u32>>,
    args_dirty: bool,
    /// The frame's paths ([`schedule::Switches`]): the split group path, its tests across
    /// workgroups, self-collision split, the wide list and its fused aux pass, the
    /// candidates' grid.
    group_split: bool,
    group_split_tests: bool,
    self_split: bool,
    wide_list: bool,
    wide_fused: bool,
    vv_grid: bool,
    /// Raycast wheels see bodies (the drive header's bit 2 in the uploaded state:
    /// `schedule::Switches::wheels_bodies`).
    wheels_bodies: bool,
    /// Indirect launches read their counts on the GPU (default); off, the host reads
    /// them back (a wait).
    device_indirect: bool,
    /// The device's multiprocessors (an indirect launch's blocks).
    sms: u32,
    /// Host reads of the indirect arguments (each one waits for the stream).
    pub args_syncs: u64,
    /// Kernel launches queued, and writes of a module's bank between them (a parameter
    /// copy's or an argument buffer's pointer): the host's per-frame work (profiles).
    pub launches: u64,
    pub bank_writes: u64,
    /// Frames that ran a split path ([`schedule::FrameKind::split`]): tests see the
    /// switches took effect.
    pub split_frames: u64,
    /// A frame queued since the last wait copied the status.
    status_due: bool,
    /// Events around timed frames ([`Self::enable_frame_timing`]).
    timing_left: u32,
    events: Vec<(sys::CUevent, sys::CUevent)>,
    /// Per-launch events while [`Self::set_phase_timing`] is on: (kernel, start, end).
    phase_events: Vec<(Key, sys::CUevent, sys::CUevent)>,
    phase_timing: bool,
}

/// Every binding's bytes and the planner's state ([`CudaExplicitBeams::snapshot`]).
pub struct CudaSnapshot {
    bufs: Vec<Vec<u8>>,
    planner: PlannerState,
    words: Vec<u32>,
}

impl CudaExplicitBeams {
    /// The build embedded device code for every explicit module (nvcc was present).
    pub fn has_ptx() -> bool {
        // (A build with ENGINE_CUDA_EXPLICIT_MODULES embeds only some modules; `new`
        // names a needed one that is missing.)
        !crate::PTX_ARCH.is_empty() && CUDA_EXPLICIT_KERNELS.iter().any(|k| !k.image.is_empty())
    }

    /// Device code is embedded and an NVIDIA driver with a device loads.
    pub fn available() -> bool {
        Self::has_ptx() && crate::CudaSim::available()
    }

    /// The kernels [`Self::new`] loads for a fleet of this layout.
    fn kernels_for_layout(layout: &ExplicitLayout) -> Vec<Kernel> {
        let mut wanted = vec![Kernel::Frame, Kernel::Solo, Kernel::Contact, Kernel::ContactBig];
        wanted.extend((0..plan::VV_PASSES.len()).map(Kernel::Vv));
        if layout.wide_ok {
            wanted.extend((0..plan::WIDE_PASSES.len()).map(Kernel::Wide));
        }
        wanted
    }

    /// The plan modules (`explicit_plan::MODULES` names) [`Self::new`] loads for a
    /// fleet of this layout: the device code a build must embed to run it.
    pub fn modules_for_layout(layout: &ExplicitLayout) -> Vec<&'static str> {
        Self::kernels_for_layout(layout)
            .iter()
            .map(|k| k.module_for(layout.aero, layout.lanes, layout.surface))
            .collect()
    }

    /// Every module a replay of a fleet of this layout may launch, whatever its switches:
    /// [`Self::modules_for_layout`], vehicle sleep's list, and the split paths' modules at
    /// the fleet's frame variant (the device code a matrix build embeds).
    pub fn modules_for_replay(layout: &ExplicitLayout) -> Vec<String> {
        let mut out: Vec<String> = Self::keys_for_replay(layout).iter().map(|k| Self::key_module(layout, k)).collect();
        out.sort();
        out.dedup();
        out
    }

    /// The kernels behind [`Self::modules_for_replay`].
    fn keys_for_replay(layout: &ExplicitLayout) -> Vec<Key> {
        let (aero, lanes) = (layout.aero, layout.lanes);
        let mut out: Vec<Key> = Self::kernels_for_layout(layout).into_iter().map(Key::Plan).collect();
        if layout.sleep.is_some() {
            out.push(Key::Plan(Kernel::Sleep));
        }
        for which in ["member", "group", "tests", "fleetfront", "fleetplan", "fleetpost", "fleetlevels", "groupbegin", "grouplevels", "grouplv"] {
            out.push(Key::Split(plan::split_module_name(which, aero, lanes)));
        }
        for vv in [false, true] {
            out.extend(plan::self_split_module_names(aero, lanes, vv).into_iter().map(|(m, _, _)| Key::Split(m)));
        }
        if !aero {
            out.push(Key::Split(plan::wide_list_module_name(lanes).to_string()));
        }
        // (Surface contact on the frame modules, split: the segments and the pass's
        // pipelines, which the saving host's planner may ask for.)
        if layout.surface {
            out.push(Key::Plan(Kernel::FrameSeg));
            out.extend((0..plan::SURF_PASSES.len()).map(|i| Key::Plan(Kernel::Surf(i))));
            // (And on the fast schedules, FramePlanner::set_surface_fast: the pass's
            // pipelines there, the solo vehicles' rest, and the wide passes where they
            // could run the fleet.)
            out.extend(plan::SURF_FAST_PASSES.iter().map(|(m, _)| Key::Split(m.to_string())));
            out.push(Key::Split(plan::solo_rest_module_name(aero, lanes, true)));
            if layout.wide_capable() {
                out.extend((0..plan::WIDE_PASSES.len()).map(|i| Key::Plan(Kernel::Wide(i))));
            }
        }
        out
    }

    /// The module a kernel of a fleet of `layout` comes from.
    fn key_module(layout: &ExplicitLayout, key: &Key) -> String {
        match key {
            Key::Plan(k) => k.module_for(layout.aero, layout.lanes, layout.surface).to_string(),
            Key::Split(name) => name.clone(),
        }
    }

    /// Load now every module a replay of this fleet may launch that this build embeds
    /// (measurement: a module is otherwise loaded at its first launch, host work inside
    /// a timed frame, and the stream's events around the frame count it).
    pub fn preload(&mut self) -> Result<(), ExecError> {
        self.bind_ctx()?;
        for key in Self::keys_for_replay(&self.layout) {
            let name = Self::key_module(&self.layout, &key);
            if CUDA_EXPLICIT_KERNELS.iter().any(|t| t.name == name.as_str() && !t.image.is_empty()) {
                self.load(&key)?;
            }
        }
        Ok(())
    }

    /// [`Self::modules_for_layout`] of `vehicles` (host only, no device): e.g. the
    /// modules a test's fleets need (`ci_explicit_modules.txt`).
    pub fn modules_for(vehicles: &[ExplicitBeams], cap: Option<CouplingCapacity>) -> Vec<&'static str> {
        Self::modules_for_layout(&ExplicitUpload::new(vehicles, cap).layout)
    }

    /// Upload `vehicles` (their current state) on device 0.
    pub fn new(vehicles: &[ExplicitBeams]) -> anyhow::Result<Self> {
        Self::with_coupling(vehicles, None)
    }

    /// [`Self::new`] with room for rigid-body proxies per vehicle (the coupling,
    /// `explicit_beams_rigid.slang`); off until [`Self::set_coupling_on`].
    pub fn with_coupling(vehicles: &[ExplicitBeams], cap: Option<CouplingCapacity>) -> anyhow::Result<Self> {
        Self::with_settings(vehicles, cap, None)
    }

    /// [`Self::with_coupling`] with room for vehicle sleep when `sleep` is set (on from
    /// the first frame; `engine_softbody::explicit_sleep`), as the WebGPU host's
    /// `with_sleep`.
    pub fn with_settings(
        vehicles: &[ExplicitBeams],
        cap: Option<CouplingCapacity>,
        sleep: Option<engine_softbody::explicit_sleep::VehicleSleep>,
    ) -> anyhow::Result<Self> {
        let upload = ExplicitUpload::with_sleep(vehicles, cap, sleep);
        let bindings = (0..BIND_COUNT as u32).map(|b| upload.bytes(b)).collect();
        Self::from_parts(upload.layout, bindings)
    }

    /// A portable fixture's state (`engine_exec::explicit_plan::portable`, saved by another
    /// host: every binding, the layout, the planner's settings and that host's frame
    /// paths): the same frames from it, with no vehicles or World here.
    pub fn from_portable(p: &plan::portable::Portable) -> anyhow::Result<Self> {
        let mut bindings = p.bindings.clone();
        // A fixture of a fleet with surface contact saved by a host that had no wide
        // passes for it (before FramePlanner::set_surface_fast) holds no vehicle table.
        // The table is the vehicles' static records (the upload's `table_recs`; a host
        // that writes a vehicle's record writes both), so it is made from them here, as
        // the upload makes it.
        let table = plan::BIND_TABLE as usize;
        let layout = &p.header.layout;
        if layout.surface && layout.wide_capable() && (bindings[table].len() as u64) < plan::TABLE_BYTES {
            let n = 16 * 4 * layout.spans.len();
            anyhow::ensure!(bindings[plan::BIND_STATIC as usize].len() >= n, "the fixture's static records are short");
            let mut t = bindings[plan::BIND_STATIC as usize][..n].to_vec();
            t.resize(plan::TABLE_BYTES as usize, 0);
            bindings[table] = t;
        }
        let mut g = Self::from_parts(p.header.layout.clone(), bindings)?;
        g.planner.apply(&p.header.planner);
        let s = p.header.switches;
        g.group_split = s.group_split;
        g.group_split_tests = s.group_split_tests;
        g.self_split = s.self_split;
        g.wide_list = s.wide_list;
        g.wide_fused = s.wide_fused;
        // (The grid's switch is a word in the bindings, already as that host had it.)
        g.vv_grid = s.vv_grid;
        g.wheels_bodies |= s.wheels_bodies;
        Ok(g)
    }

    /// Device buffers of `bindings` for a fleet of `layout`, its modules loaded.
    fn from_parts(layout: ExplicitLayout, bindings: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        anyhow::ensure!(bindings.len() == BIND_COUNT, "{} bindings, not {BIND_COUNT}", bindings.len());
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
        let sms = unsafe { result::device::get_attribute(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) }
            .map_err(map)? as u32;

        // Raycast wheels that see bodies: the drive header's .x bit 2 (explicit_plan::drive).
        let wheels_bodies = layout.drive.is_some_and(|d| {
            let at = d.header as usize * 16;
            bindings[engine_exec::explicit_plan::BIND_MISC as usize]
                .get(at..at + 4)
                .is_some_and(|w| u32::from_le_bytes(w.try_into().unwrap()) & 2 != 0)
        });
        let mut bufs = Vec::with_capacity(BIND_COUNT);
        let mut sizes = Vec::with_capacity(BIND_COUNT);
        for bytes in bindings {
            let ptr = unsafe { result::malloc_sync(bytes.len()) }.map_err(map)?;
            unsafe { result::memcpy_htod_sync(ptr, &bytes) }.map_err(map)?;
            bufs.push(ptr);
            sizes.push(bytes.len() as u64);
        }

        let wanted = Self::kernels_for_layout(&layout);
        // Every module this fleet needs that the build left out (ENGINE_CUDA_EXPLICIT_MODULES),
        // named at once.
        let missing: Vec<&str> = Self::modules_for_layout(&layout)
            .into_iter()
            .filter(|name| CUDA_EXPLICIT_KERNELS.iter().any(|t| t.name == *name && t.image.is_empty()))
            .collect();
        anyhow::ensure!(
            missing.is_empty(),
            "explicit modules without device code in this build: {} (add them to \
             ENGINE_CUDA_EXPLICIT_MODULES, or unset it to build every module)",
            missing.join(",")
        );
        let mut kernels = HashMap::new();
        for k in wanted {
            let name = k.module_for(layout.aero, layout.lanes, layout.surface);
            let t = CUDA_EXPLICIT_KERNELS
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| anyhow::anyhow!("explicit kernel {name} not built"))?;
            anyhow::ensure!(
                !t.image.is_empty(),
                "explicit module {name} has no device code in this build: add it to \
                 ENGINE_CUDA_EXPLICIT_MODULES (or unset that variable to build every module)"
            );
            kernels.insert(Key::Plan(k), load_kernel(t, &bufs, &sizes, sms)?);
        }
        // The split paths' argument buffers, zeroed (as WebGPU's are).
        let mut arg_bufs = HashMap::new();
        for (buf, bytes) in [
            (ArgBuf::SelfSplit, SELF_ARGS_BYTES),
            (ArgBuf::Fleet1, FLEET_ARGS_BYTES),
            (ArgBuf::Fleet2, FLEET_ARGS_BYTES),
            (ArgBuf::Spare1, FLEET_ARGS_BYTES),
            (ArgBuf::Spare2, FLEET_ARGS_BYTES),
            (ArgBuf::Surf, schedule::SURF_ARGS_BYTES),
        ] {
            let ptr = unsafe { result::malloc_sync(bytes as usize) }.map_err(map)?;
            unsafe { result::memset_d8_sync(ptr, 0, bytes as usize) }.map_err(map)?;
            arg_bufs.insert(buf, (ptr, bytes));
        }
        let mut planner = FramePlanner::new(&layout);
        planner.set_surface_fast(schedule::surface_fast_default());
        planner.set_surface_fast_contact(schedule::surface_fast_contact_default(true));
        Ok(Self {
            ctx,
            stream,
            owns_stream: true,
            device_name,
            layout,
            planner,
            bufs,
            sizes,
            kernels,
            arg_bufs,
            args: HashMap::new(),
            args_dirty: true,
            group_split: false,
            group_split_tests: true,
            self_split: false,
            wide_list: false,
            wide_fused: false,
            vv_grid: true,
            wheels_bodies,
            device_indirect: std::env::var("RUBBLE_CUDA_HOST_INDIRECT").as_deref() != Ok("1"),
            sms,
            args_syncs: 0,
            launches: 0,
            bank_writes: 0,
            split_frames: 0,
            status_due: false,
            timing_left: 0,
            events: Vec::new(),
            phase_events: Vec::new(),
            phase_timing: false,
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

    pub(crate) fn context(&self) -> sys::CUcontext {
        self.ctx
    }

    /// Queue everything on `stream` from now on (another host's in this context: the
    /// World's rigid step and its coupling share one, so their launches and this
    /// host's run in the order they were queued with no wait between). The stream
    /// must outlive this host.
    pub(crate) fn use_stream(&mut self, stream: sys::CUstream) -> Result<(), ExecError> {
        self.bind_ctx()?;
        unsafe { result::stream::synchronize(self.stream) }.map_err(backend)?;
        if self.owns_stream {
            unsafe { result::stream::destroy(self.stream) }.map_err(backend)?;
        }
        self.stream = stream;
        self.owns_stream = false;
        Ok(())
    }

    /// Binding `b`'s device buffer and its bytes (the coupling's passes bind the nodes
    /// and gMisc beside the rigid buffers).
    pub(crate) fn buffer(&self, b: u32) -> (sys::CUdeviceptr, u64) {
        (self.bufs[b as usize], self.sizes[b as usize])
    }

    /// Binding `b` as words (waits; tests and tools).
    pub fn download_words(&mut self, b: u32) -> Result<Vec<u32>, ExecError> {
        self.wait()?;
        self.download::<u32>(b)
    }

    /// Vehicle sleep is on (only with room: [`Self::with_settings`]).
    pub fn sleep_on(&self) -> bool {
        self.planner.sleep_on()
    }

    /// gMisc record of the retired nodes' bits (0: none retired).
    pub fn retire_base(&self) -> u32 {
        self.planner.words[plan::W_RETIRE_BASE]
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
        let ready = |_: &str| true;
        // Wheels that see bodies leave out the schedules that run rays ahead.
        if self.wheels_bodies {
            let f = self.planner.perf_flags();
            self.planner.set_perf_flags(f & !schedule::WHEELS_AHEAD_FLAGS);
        }
        let sw = schedule::Switches {
            group_split: self.group_split,
            group_split_tests: self.group_split_tests,
            self_split: self.self_split,
            wide_list: self.wide_list,
            wide_fused: self.wide_fused,
            wheels_bodies: self.wheels_bodies,
            vv_grid: self.vv_grid,
            group_apart: schedule::group_apart_default(),
            current: true,
            ablate: false,
            debug: false,
            wide_allowed: true,
            aero: self.layout.aero,
            lanes: self.layout.lanes,
            vehicles: self.layout.spans.len(),
            // (No level reports read on this host yet: the levels run in their groups'
            // workgroups, the same results.)
            level_want: 0,
            self_split_skip: 0,
            ready: &ready,
        };
        let frame = schedule::frame(&mut self.planner, &sw, steps);
        self.split_frames += u64::from(frame.kind.split());
        // Pageable sources are staged before the call returns; stream order keeps
        // the last frame's kernels reading the old parameters.
        unsafe { result::memcpy_htod_async(self.bufs[BIND_PARAMS as usize], &frame.params, self.stream) }
            .map_err(backend)?;
        let timed = self.timing_left > 0;
        let mut stamp = None;
        if timed {
            self.timing_left -= 1;
            let a = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(backend)?;
            unsafe { result::event::record(a, self.stream) }.map_err(backend)?;
            stamp = Some(a);
        }
        let (aero, lanes) = (self.layout.aero, self.layout.lanes);
        for l in &frame.launches {
            match l.op {
                Op::Cmd(Cmd::Dispatch { kernel, groups, param_copy }) => {
                    self.launch(Key::Plan(kernel), Launch::Groups((groups, 1, 1)), param_copy, [None, None])?;
                    // The contact plan's passes and vehicle sleep's awake list write the
                    // plan's arguments (the frames that follow read them).
                    if matches!(kernel, Kernel::Vv(_) | Kernel::Sleep) {
                        self.args_dirty = true;
                    }
                }
                Op::Cmd(Cmd::Indirect { kernel, offset }) => {
                    let grid = self.indirect(Key::Plan(kernel), ArgBuf::Plan, offset)?;
                    self.launch(Key::Plan(kernel), grid, 0, [None, None])?;
                }
                Op::Cmd(Cmd::Copy { src, src_offset, dst, dst_offset, size }) => unsafe {
                    result::memcpy_dtod_async(
                        self.bufs[dst as usize] + dst_offset,
                        self.bufs[src as usize] + src_offset,
                        size as usize,
                        self.stream,
                    )
                    .map_err(backend)?;
                },
                ref op => {
                    let sl = schedule::split_launch(op, aero, lanes, &frame.kind).expect("a split path launch");
                    let key = Key::Split(sl.module);
                    let grid = match sl.grid {
                        Grid::Groups(n) => Launch::Groups((n, 1, 1)),
                        Grid::Indirect(buf, offset) => self.indirect(key.clone(), buf, offset)?,
                    };
                    self.launch(key, grid, sl.copy, [sl.binding10, sl.binding11])?;
                    // (A split path's kernels write the arguments of the dispatches after
                    // them: the groups', the fleet's, self split's.)
                    self.args_dirty = true;
                }
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

    /// An argument buffer's pointer and bytes.
    fn arg_buf(&self, buf: ArgBuf) -> (sys::CUdeviceptr, u64) {
        match buf {
            ArgBuf::Plan => (self.bufs[BIND_ARGS as usize], self.sizes[BIND_ARGS as usize]),
            b => self.arg_bufs[&b],
        }
    }

    /// The grid of `key`'s indirect launch from byte `offset` of argument buffer `buf`:
    /// that address, for the indirect entry to read on the GPU, or (device indirect off,
    /// or a module without the entry) the workgroups read back, a wait for the stream
    /// when any launch was queued since the buffer was last read.
    fn indirect(&mut self, key: Key, buf: ArgBuf, offset: u64) -> Result<Launch, ExecError> {
        if self.device_indirect {
            self.load(&key)?;
            if self.kernels[&key].indirect.is_some() {
                return Ok(Launch::Indirect(self.arg_buf(buf).0 + offset));
            }
        }
        if self.args_dirty {
            self.args.clear();
            self.args_dirty = false;
        }
        if !self.args.contains_key(&buf) {
            let (ptr, bytes) = self.arg_buf(buf);
            let mut raw = vec![0u32; (bytes / 4) as usize];
            unsafe {
                result::memcpy_dtoh_async(bytemuck::cast_slice_mut::<u32, u8>(&mut raw), ptr, self.stream).map_err(backend)?;
                result::stream::synchronize(self.stream).map_err(backend)?;
            }
            self.args_syncs += 1;
            self.args.insert(buf, raw);
        }
        let a = &self.args[&buf];
        let w = (offset / 4) as usize;
        Ok(Launch::Groups((a[w], a[w + 1], a[w + 2])))
    }

    /// Load `key`'s module if it is not yet.
    fn load(&mut self, key: &Key) -> Result<(), ExecError> {
        if !self.kernels.contains_key(key) {
            // A module the fleet's first frames did not need (a split path's, vehicle
            // sleep's list, the surface segments), loaded at its first launch.
            let name = match &key {
                Key::Plan(k) => k.module_for(self.layout.aero, self.layout.lanes, self.layout.surface).to_string(),
                Key::Split(name) => name.clone(),
            };
            let t = CUDA_EXPLICIT_KERNELS
                .iter()
                .find(|t| t.name == name.as_str())
                .ok_or_else(|| ExecError::Backend(format!("CUDA: explicit module {name} not built")))?;
            let k = load_kernel(t, &self.bufs, &self.sizes, self.sms).map_err(|e| ExecError::Backend(e.to_string()))?;
            self.kernels.insert(key.clone(), k);
        }
        Ok(())
    }

    fn launch(&mut self, key: Key, grid: Launch, param_copy: u32, args: [Option<ArgBuf>; 2]) -> Result<(), ExecError> {
        let stream = self.stream;
        let params = self.bufs[BIND_PARAMS as usize];
        self.load(&key)?;
        let bound: Vec<Option<(sys::CUdeviceptr, u64)>> = args.iter().map(|a| a.map(|b| self.arg_buf(b))).collect();
        let k = self
            .kernels
            .get_mut(&key)
            .ok_or_else(|| ExecError::Backend(format!("CUDA: explicit kernel {key:?} not loaded")))?;
        if let Some((offset, bound)) = &mut k.params {
            if *bound != param_copy {
                let ptr = params + u64::from(param_copy) * PARAM_STRIDE;
                // Stream-ordered: earlier launches keep the copy they were queued with.
                unsafe { result::memcpy_htod_async(k.globals + u64::from(*offset), &ptr.to_le_bytes(), stream) }
                    .map_err(backend)?;
                *bound = param_copy;
                self.bank_writes += 1;
            }
        }
        // Bindings 10 and 11 of a split path's launch: its argument buffers.
        for (i, b) in bound.iter().enumerate() {
            let Some((ptr, bytes)) = *b else { continue };
            if k.bound[i] == ptr {
                continue;
            }
            for &(offset, size, slot) in k.binds {
                if slot as usize == 10 + i {
                    let mut field = ptr.to_le_bytes().to_vec();
                    if size == 16 {
                        field.extend_from_slice(&(bytes / 4).to_le_bytes());
                    }
                    unsafe { result::memcpy_htod_async(k.globals + u64::from(offset), &field, stream) }.map_err(backend)?;
                    self.bank_writes += 1;
                }
            }
            k.bound[i] = ptr;
        }
        if let Launch::Groups(g) = grid {
            if g.0 == 0 || g.1 == 0 || g.2 == 0 {
                return Ok(());
            }
        }
        let (function, threads, indirect) = (k.function, k.threads, k.indirect);
        let stamps = if self.phase_timing {
            let a = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(backend)?;
            unsafe { result::event::record(a, stream) }.map_err(backend)?;
            Some(a)
        } else {
            None
        };
        match grid {
            Launch::Groups(g) => unsafe { result::launch_kernel(function, g, (threads, 1, 1), 0, stream, &mut []) },
            Launch::Indirect(mut counts) => {
                let (f, blocks) = indirect.expect("an indirect launch of a module with the indirect entry");
                let mut params = [&mut counts as *mut sys::CUdeviceptr as *mut c_void];
                unsafe { result::launch_kernel(f, (blocks, 1, 1), (threads, 1, 1), 0, stream, &mut params) }
            }
        }
        .map_err(|e| ExecError::Backend(format!("CUDA: launch {key:?}: {e:?}")))?;
        self.launches += 1;
        if let Some(a) = stamps {
            let b = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(backend)?;
            unsafe { result::event::record(b, stream) }.map_err(backend)?;
            self.phase_events.push((key, a, b));
        }
        Ok(())
    }

    /// The split group path (contact-groups-design.md), its small groups' tests across
    /// workgroups (on by default), self-collision split (selfcol-design.md), the wide
    /// passes' reused pair list and its fused aux pass: the paths engine_exec's frame
    /// schedule runs, the same results bit for bit as without them. Off by default here;
    /// their modules load at first use (a build without them fails that frame).
    pub fn set_group_split(&mut self, on: bool) {
        self.group_split = on;
    }
    pub fn set_group_split_tests(&mut self, on: bool) {
        self.group_split_tests = on;
    }
    pub fn set_self_split(&mut self, on: bool) {
        self.self_split = on;
    }
    pub fn set_wide_list(&mut self, on: bool) {
        self.wide_list = on;
    }
    pub fn set_wide_fused(&mut self, on: bool) {
        self.wide_fused = on;
    }
    /// A fleet with surface contact on the fast schedules, the surface pass as its own
    /// pipelines (`FramePlanner::set_surface_fast`; on by default,
    /// `schedule::surface_fast_default`): the same results bit for bit.
    pub fn set_surface_fast(&mut self, on: bool) {
        self.planner.set_surface_fast(on);
    }

    /// Indirect launches read their workgroup counts on the GPU (on by default; the same
    /// workgroups, bit for bit) or, off, the host reads them back first (a wait for the
    /// stream per argument buffer a launch since wrote; measurement, and
    /// `RUBBLE_CUDA_HOST_INDIRECT=1`).
    pub fn set_device_indirect(&mut self, on: bool) {
        self.device_indirect = on;
        self.args_dirty = true;
    }

    /// Time every launch (measurement: an event pair around each, which serialises
    /// nothing on one stream but adds a little per launch) until turned off;
    /// [`Self::phase_times_ms`] sums them by kernel.
    pub fn set_phase_timing(&mut self, on: bool) {
        self.phase_timing = on;
    }

    /// GPU milliseconds and launches per kernel since the last call (waits).
    pub fn phase_times_ms(&mut self) -> Result<Vec<(String, f64, u32)>, ExecError> {
        self.wait()?;
        let mut by: Vec<(String, f64, u32)> = Vec::new();
        for (k, a, b) in std::mem::take(&mut self.phase_events) {
            let ms = f64::from(unsafe { result::event::elapsed(a, b) }.map_err(backend)?);
            unsafe {
                let _ = result::event::destroy(a);
                let _ = result::event::destroy(b);
            }
            let name = match &k {
                Key::Plan(k) => format!("{k:?}"),
                Key::Split(m) => m.clone(),
            };
            match by.iter_mut().find(|x| x.0 == name) {
                Some(x) => {
                    x.1 += ms;
                    x.2 += 1;
                }
                None => by.push((name, ms, 1)),
            }
        }
        by.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(by)
    }

    /// Tests and A/Bs: a copy of every binding and of the planner's state, to run the
    /// same frames again from it ([`Self::restore`]; waits).
    pub fn snapshot(&mut self) -> Result<CudaSnapshot, ExecError> {
        self.wait()?;
        let bufs = (0..BIND_COUNT as u32).map(|b| self.download::<u8>(b)).collect::<Result<_, _>>()?;
        Ok(CudaSnapshot { bufs, planner: self.planner.state(), words: self.planner.words.to_vec() })
    }

    /// Put back a [`Self::snapshot`] (the bindings, the planner's state and words).
    pub fn restore(&mut self, s: &CudaSnapshot) -> Result<(), ExecError> {
        for (b, bytes) in s.bufs.iter().enumerate() {
            self.upload(b as u32, 0, bytes)?;
        }
        self.planner.restore(s.planner);
        self.planner.words.copy_from_slice(&s.words);
        self.args_dirty = true;
        self.status_due = false;
        self.wait()
    }

    /// The solver's state as words for bit-for-bit comparisons
    /// ([`engine_exec::explicit_plan::state_words`], as the WebGPU host's; waits).
    pub fn state_words(&mut self) -> Result<Vec<(&'static str, Vec<u32>)>, ExecError> {
        self.wait()?;
        let d = |b: u32| self.download::<u32>(b);
        Ok(plan::state_words(
            &self.layout,
            &self.planner.words,
            d(plan::BIND_NODES)?,
            &d(plan::BIND_BEAMS)?,
            &d(plan::BIND_ACC)?,
            d(plan::BIND_GROUPS)?,
            d(plan::BIND_MISC)?,
            &d(plan::BIND_NEXT)?,
        ))
    }

    /// The self-collision pair list's reuse margin (metres; 0 lists every step), as
    /// the WebGPU host's `set_pair_reuse_margin` (a parameter word).
    pub fn set_pair_reuse_margin(&mut self, metres: f32) {
        self.planner.words[plan::W_PAIR_REUSE] = if metres > 0.0 { metres.to_bits() } else { 0 };
    }

    /// Vehicle contact's candidate grid on or off (measurement; a status word in gAcc,
    /// as the WebGPU host's `set_vv_grid`).
    pub fn set_vv_grid(&mut self, on: bool) -> Result<(), ExecError> {
        self.vv_grid = on;
        let w = (self.layout.status.base / 4) as usize + 9;
        self.upload(plan::BIND_ACC, (w * 4) as u64, bytemuck::bytes_of(&i32::from(!on)))
    }

    /// The coupling header: on, plus measurement bits (`COUPLING_LIST_OFF`,
    /// `COUPLING_FOLD_OFF`, `COUPLING_HEADER_BITS`), as the WebGPU host writes it.
    pub fn set_coupling_header(&mut self, on: bool, bits: u32) -> Result<(), ExecError> {
        assert!(self.layout.coupling.is_some(), "explicit beams: built without coupling room");
        if on {
            self.set_wide(false);
        }
        let word = u32::from(on) | if on { bits } else { 0 };
        self.upload(plan::BIND_MISC, self.layout.coupling_on_offset(), bytemuck::bytes_of(&word))
    }

    /// The planner, for switches that are its own (contact margin, ablation mask, ...).
    pub fn planner_mut(&mut self) -> &mut FramePlanner {
        &mut self.planner
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

    /// Set vehicle `v`'s driver input for its raycast wheels (a fleet without raycast
    /// wheels ignores it).
    pub fn set_drive_input(&mut self, v: usize, input: engine_softbody::explicit_drive::DriveInput) -> Result<(), ExecError> {
        let Some(d) = self.layout.drive else { return Ok(()) };
        self.upload(plan::BIND_MISC, d.input_offset(v), bytemuck::cast_slice(&plan::drive::input_words(input)))?;
        self.planner.request_replan();
        Ok(())
    }

    /// [`Self::set_drive_input`] from the input's words (`explicit_plan::drive::input_words`:
    /// a portable World's recorded inputs).
    pub fn set_drive_input_words(&mut self, v: usize, words: [u32; 4]) -> Result<(), ExecError> {
        let Some(d) = self.layout.drive else { return Ok(()) };
        self.upload(plan::BIND_MISC, d.input_offset(v), bytemuck::cast_slice(&words))?;
        self.planner.request_replan();
        Ok(())
    }

    /// Read vehicle `v`'s raycast wheels' state back into `drive` (waits).
    pub fn read_drive(&mut self, v: usize, drive: &mut engine_softbody::explicit_drive::Drive) -> Result<(), ExecError> {
        let Some(l) = self.layout.drive else { return Ok(()) };
        self.wait()?;
        let misc: Vec<u32> = self.download(plan::BIND_MISC)?;
        let at = l.vehicle(v) as usize * 4;
        plan::drive::decode(&misc[at..at + l.stride as usize * 4], drive);
        Ok(())
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

    /// Block until queued frames have run; `Err` on a contact error or a loop held
    /// to its cap a frame reported.
    pub fn wait(&mut self) -> Result<(), ExecError> {
        self.bind_ctx()?;
        unsafe { result::stream::synchronize(self.stream) }.map_err(backend)?;
        if std::mem::take(&mut self.status_due) {
            let acc = self.download::<i32>(plan::BIND_ACC)?;
            self.layout.kernel_status(&acc).map_err(ExecError::Backend)?;
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
            for (_, a, b) in self.phase_events.drain(..) {
                let _ = result::event::destroy(a);
                let _ = result::event::destroy(b);
            }
            for k in self.kernels.values() {
                let _ = result::module::unload(k.module);
            }
            for &b in &self.bufs {
                let _ = result::free_sync(b);
            }
            for &(b, _) in self.arg_bufs.values() {
                let _ = result::free_sync(b);
            }
            if self.owns_stream {
                let _ = result::stream::destroy(self.stream);
            }
        }
    }
}

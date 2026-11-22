//! `CudaWorld`: the World step on CUDA: the rigid step ([`CudaSim`]), the explicit
//! soft-body frame ([`CudaExplicitBeams`]) and the coupling between them
//! (`slang/kernels/vehicle_coupling.slang`), in one context on one stream.
//!
//! The coupling's order and parameters are the shared plan's
//! (`engine_exec::explicit_plan::coupling::world`, which the wgpu host in engine-world
//! follows too); this file only allocates the coupling's buffers, fills each kernel's
//! `SLANG_globalParams` bank with pointers into the rigid and the explicit buffers, and
//! replays the commands. A step queues, without a host wait:
//!
//! 1. the window's parameters (the wgpu host's queue write lands before its submission's
//!    passes, so the emit already reads them);
//! 2. the emit (`cpl_emit` over the proxied list's count on the GPU, `{entry}_gs`), and
//!    the rigid step is told of the impulse records (`rigid_host::GpuImpulses`);
//! 3. the rigid step;
//! 4. the plan (bounds, candidates when the rigid step kept its resting table, plan,
//!    share), the explicit frame, the loads and the asleep bits;
//! 5. a read of the step's counters and status words into pinned memory behind an
//!    event, adopted by a later step (one step late when the caller waits each step,
//!    as the wgpu readback ring's): the rigid counters, the explicit frame's loop-cap
//!    flag, the coupling's status word, the asleep bits.

use crate::{backend, CudaExplicitBeams, CudaSim};
use cudarc::driver::{result, sys};
use engine_exec::explicit_plan::coupling::world as cw;
use engine_exec::explicit_plan::{self as plan, CouplingCapacity, CouplingLayout};
use engine_exec::ExecError;
use engine_schema::StepCounters;
use engine_softbody::explicit_beams::ExplicitBeams;
use std::collections::VecDeque;
use std::ffi::{c_void, CString};

pub mod table {
    pub struct CouplingCudaKernel {
        pub entry: &'static str,
        /// Bytes of Slang-emitted CUDA C++ (non-zero proves the source built).
        pub cu_bytes: u64,
        /// Fatbin (empty when the build host had no nvcc).
        pub image: &'static [u8],
        pub threads: [u32; 3],
        pub params_bytes: u32,
        /// (byte offset in `SLANG_globalParams`, field bytes 8|16, the module's binding)
        pub binds: &'static [(u32, u32, u32)],
    }
    include!(concat!(env!("OUT_DIR"), "/cuda_coupling_kernels.rs"));
}
pub use table::{CouplingCudaKernel, CUDA_COUPLING_KERNELS};

/// How the World couples its explicit vehicles (engine-world's `ExplicitVehiclesDef`,
/// the settings a CUDA host has).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingDef {
    /// Proxy bodies, shapes and hull planes per vehicle per window. More fails the step.
    pub capacity: CouplingCapacity,
    /// The vehicles' collision filter.
    pub categories: u16,
    pub mask: u16,
    /// Extra reach of the proxy plan (m).
    pub margin: f32,
    pub vehicle_contact: bool,
    /// Vehicle sleep (`None`: every vehicle steps every step).
    pub sleep: Option<engine_softbody::explicit_sleep::VehicleSleep>,
}

impl Default for CouplingDef {
    fn default() -> Self {
        Self {
            capacity: CouplingCapacity { bodies: 16, shapes: 64, planes: 1024 },
            categories: 0xFFFF,
            mask: 0xFFFF,
            margin: 0.1,
            vehicle_contact: true,
            sleep: Some(engine_softbody::explicit_sleep::VehicleSleep::default()),
        }
    }
}

struct Kernel {
    module: sys::CUmodule,
    function: sys::CUfunction,
    /// The indirect entry (`{entry}_gs`) and the blocks a launch of it runs.
    indirect: (sys::CUfunction, u32),
    threads: u32,
}

/// A slot of the status ring: pinned host memory a step's words are copied to, and the
/// event behind the copies.
struct Slot {
    host: *mut u8,
    event: sys::CUevent,
    /// The rigid host's `steps_done` after the step it was read with.
    steps_done: u32,
    /// That step's rigid step ran (its counters are new).
    ran: bool,
}

const RING: usize = 3;
/// GPU time of one step by part (ms), from events on the stream.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepTimes {
    /// The whole step.
    pub total: f64,
    /// The coupling's emit, the rigid step, the coupling's plan (bounds, candidates,
    /// plan, share), the explicit frame, the coupling's loads and asleep bits.
    pub emit: f64,
    pub rigid: f64,
    pub plan: f64,
    pub frame: f64,
    pub loads: f64,
}

pub struct CudaWorld {
    // Before `sim`: dropped first, while the stream both use (the rigid host's) lives.
    explicit: CudaExplicitBeams,
    sim: CudaSim,
    stream: sys::CUstream,
    kernels: Vec<Kernel>,
    params: sys::CUdeviceptr,
    work: sys::CUdeviceptr,
    list_args: sys::CUdeviceptr,
    /// Stand-ins: the threshold ingest's three buffers, and `pose_pack`'s header and
    /// moved list while the rigid host keeps none.
    spare: Vec<sys::CUdeviceptr>,
    layout: CouplingLayout,
    work_layout: cw::WorkLayout,
    categories: u16,
    mask: u16,
    /// The proxy plan's reach (the def's margin, or the wheels' rays').
    margin: f32,
    hull_geom: bool,
    /// The piece windows' label and box records ((0, 0): one window per vehicle).
    piece_records: (u32, u32),
    clock: cw::Clock,
    /// The asleep bits the last adopted read reported, and whether a vehicle was awake.
    sleep_mask: Vec<u32>,
    any_awake: bool,
    slots: Vec<Slot>,
    /// Slots whose read is queued and not yet adopted, oldest first.
    in_flight: VecDeque<usize>,
    /// The first error an adopted read reported (returned by the next step or wait).
    failed: Option<String>,
    /// Steps still to time, and the events of the timed ones.
    timing_left: u32,
    events: Vec<[sys::CUevent; 6]>,
    /// The coupling's launches queued (the explicit frame's and the rigid step's are
    /// their hosts').
    pub launches: u64,
    /// Host waits a step took because every ring slot was in flight.
    pub ring_waits: u64,
}

fn status_words(mask_words: usize) -> usize {
    std::mem::size_of::<StepCounters>().div_ceil(4) + 1 + 2 + mask_words
}

impl CudaWorld {
    /// The build embedded device code for the coupling (nvcc was present).
    pub fn has_ptx() -> bool {
        CudaSim::has_ptx() && CudaExplicitBeams::has_ptx() && CUDA_COUPLING_KERNELS.iter().all(|k| !k.image.is_empty())
    }

    /// Device code is embedded and an NVIDIA driver with a device loads.
    pub fn available() -> bool {
        Self::has_ptx() && CudaSim::available()
    }

    /// `vehicles` (their current state) in `sim`'s world: they collide two way with its
    /// bodies from the next step (engine-world's `World::add_explicit_vehicles`).
    pub fn new(sim: CudaSim, vehicles: &[ExplicitBeams], def: &CouplingDef) -> anyhow::Result<Self> {
        anyhow::ensure!(!vehicles.is_empty(), "explicit vehicles: none given");
        let c = def.capacity;
        anyhow::ensure!(
            c.bodies <= c.shapes && c.shapes <= 1024 && c.planes < 1 << 16,
            "explicit vehicles: at most 1024 shapes (and no more bodies) and 65535 hull planes per vehicle, not {c:?}"
        );
        let mut explicit = CudaExplicitBeams::with_settings(vehicles, Some(def.capacity), def.sleep)?;
        explicit.set_vehicle_contact(def.vehicle_contact);
        explicit.set_coupling_on(true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let layout = explicit.coupling().expect("coupling room");
        let work_layout = cw::WorkLayout::new(vehicles.len() as u32, sim.capacity.max_bodies, def.capacity.shapes);
        let mut work = vec![0u8; work_layout.bytes() as usize];
        let rows: Vec<[u32; 4]> = vehicles.iter().enumerate().map(|(i, v)| cw::vehicle_row(explicit.layout().spans[i].node_off, v)).collect();
        let at = work_layout.table as usize;
        work[at..at + rows.len() * 16].copy_from_slice(bytemuck::cast_slice(&rows));
        let state = CouplingState {
            layout,
            work_layout,
            categories: def.categories,
            mask: def.mask,
            margin: def.margin,
            hull_geom: vehicles.iter().any(|v| v.surface_contact),
            piece_records: (0, 0),
            clock: cw::Clock::default(),
            sleep_mask: vec![0; work_layout.mask_words as usize],
            any_awake: true,
            work,
            list_args: cw::LIST_ARGS_INIT,
        };
        Self::from_parts(sim, explicit, state)
    }

    /// A portable World (`engine_exec::world_portable`, saved by the wgpu World at one
    /// moment): its rigid scene ([`CudaSim::from_portable`]: the contacts start cold),
    /// its explicit vehicles ([`CudaExplicitBeams::from_portable`]) and the coupling's
    /// carried state (the clock, the pending emit, the work buffer with the vehicle
    /// table, the proxied list and the broadphase proxies).
    pub fn from_portable(w: &engine_exec::world_portable::PortableWorld) -> anyhow::Result<Self> {
        let sim = CudaSim::from_portable(w)?;
        let explicit = CudaExplicitBeams::from_portable(&w.explicit().map_err(|e| anyhow::anyhow!("{e}"))?)?;
        let c = &w.header.coupling;
        anyhow::ensure!(
            explicit.coupling() == Some(c.layout),
            "the coupling's layout is not the explicit state's: {:?} against {:?}",
            c.layout,
            explicit.coupling()
        );
        anyhow::ensure!(
            c.work_layout.max_bodies == w.header.rigid.capacity.max_bodies && c.work_layout.vehicles == c.layout.vehicles,
            "the coupling's work layout is not the World's: {:?}",
            c.work_layout
        );
        let state = CouplingState {
            layout: c.layout,
            work_layout: c.work_layout,
            categories: c.categories,
            mask: c.mask,
            margin: c.margin,
            hull_geom: c.hull_geom,
            piece_records: c.piece_records,
            clock: c.clock,
            sleep_mask: c.sleep_mask.clone(),
            any_awake: c.any_awake,
            work: w.blob("work").to_vec(),
            list_args: c.list_args,
        };
        Self::from_parts(sim, explicit, state)
    }

    /// A World from its hosts and the coupling's own state (a portable World fixture's,
    /// or [`Self::new`]'s): `explicit` moves onto `sim`'s stream.
    pub fn from_parts(sim: CudaSim, mut explicit: CudaExplicitBeams, state: CouplingState) -> anyhow::Result<Self> {
        anyhow::ensure!(
            CUDA_COUPLING_KERNELS.iter().all(|k| !k.image.is_empty()),
            "engine-cuda was built without nvcc: no device code for vehicle_coupling.slang (see engine/TOOLCHAIN.md)"
        );
        let map = |e: cudarc::driver::DriverError| anyhow::anyhow!("CUDA: {e:?}");
        anyhow::ensure!(
            state.work.len() as u64 == state.work_layout.bytes(),
            "the coupling's work buffer is {} bytes, its layout says {}",
            state.work.len(),
            state.work_layout.bytes()
        );
        // One context (both hosts retain the device's primary context) and one stream.
        anyhow::ensure!(sim.ctx == explicit.context(), "CudaSim and CudaExplicitBeams are in two CUDA contexts");
        unsafe { result::ctx::set_current(sim.ctx) }.map_err(map)?;
        let stream = sim.stream;
        explicit.use_stream(stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        let dev = result::device::get(0).map_err(map)?;
        let sms = unsafe { result::device::get_attribute(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) }.map_err(map)? as u32;

        let alloc = |bytes: &[u8]| -> anyhow::Result<sys::CUdeviceptr> {
            let ptr = unsafe { result::malloc_sync(bytes.len().max(16)) }.map_err(map)?;
            unsafe { result::memset_d8_sync(ptr, 0, bytes.len().max(16)) }.map_err(map)?;
            unsafe { result::memcpy_htod_sync(ptr, bytes) }.map_err(map)?;
            Ok(ptr)
        };
        let params = alloc(&[0u8; std::mem::size_of::<cw::Params>()])?;
        let work = alloc(&state.work)?;
        let list_args = alloc(bytemuck::cast_slice(&state.list_args))?;
        let dummy = vec![0u8; cw::INGEST_DUMMY_BYTES as usize];
        let spare: Vec<sys::CUdeviceptr> = (0..5).map(|_| alloc(&dummy)).collect::<anyhow::Result<_>>()?;

        // Each binding's device buffer and bytes.
        let buffer = |binding: u32| -> (sys::CUdeviceptr, u64) {
            use cw::Source as S;
            let dummy = |k: usize| (spare[k], cw::INGEST_DUMMY_BYTES);
            match cw::source(binding) {
                S::Params => (params, std::mem::size_of::<cw::Params>() as u64),
                S::RigidBodies => sim.buffer(0),
                S::RigidColliders => sim.buffer(2),
                S::RigidHulls => sim.buffer(18),
                S::RigidScratch => sim.buffer(21),
                S::RestIds => sim.buffer(14),
                S::RestAabbs => sim.buffer(9),
                S::MovedHeader => dummy(3),
                S::Moved => dummy(4),
                S::ExplicitNodes => explicit.buffer(plan::BIND_NODES),
                S::ExplicitMisc => explicit.buffer(plan::BIND_MISC),
                S::Work => (work, state.work.len() as u64),
                S::IngestRecords => dummy(0),
                S::IngestRows => dummy(1),
                S::IngestInfo => dummy(2),
            }
        };
        let mut kernels = Vec::with_capacity(cw::Entry::ALL.len());
        for e in cw::Entry::ALL {
            let t = CUDA_COUPLING_KERNELS
                .iter()
                .find(|t| t.entry == e.name())
                .ok_or_else(|| anyhow::anyhow!("coupling kernel {} not built", e.name()))?;
            let module = unsafe { result::module::load_data(t.image.as_ptr() as *const c_void) }
                .map_err(|e2| anyhow::anyhow!("CUDA: load module {}: {e2:?}", t.entry))?;
            let function = unsafe { result::module::get_function(module, CString::new(t.entry)?) }
                .map_err(|e2| anyhow::anyhow!("CUDA: function {}: {e2:?}", t.entry))?;
            let gs = unsafe { result::module::get_function(module, CString::new(format!("{}_gs", t.entry))?) }
                .map_err(|e2| anyhow::anyhow!("CUDA: function {}_gs: {e2:?}", t.entry))?;
            let threads = t.threads[0] * t.threads[1] * t.threads[2];
            let per_sm = unsafe { result::occupancy::max_active_block_per_multiprocessor(gs, threads as i32, 0) }
                .map_err(|e2| anyhow::anyhow!("CUDA: occupancy {}: {e2:?}", t.entry))?;
            let (globals, bytes) = unsafe { result::module::get_global(module, CString::new("SLANG_globalParams")?) }
                .map_err(|e2| anyhow::anyhow!("CUDA: {} globals: {e2:?}", t.entry))?;
            anyhow::ensure!(
                bytes >= t.params_bytes as usize,
                "{}: SLANG_globalParams is {bytes} bytes, reflection says {}",
                t.entry,
                t.params_bytes
            );
            // The bank, once: Slang keeps every global of the module in each entry's
            // bank, so one set of pointers serves cpl_query too (no binding limit here).
            let mut bank = vec![0u8; bytes];
            for &(offset, size, binding) in t.binds {
                let (ptr, len) = buffer(binding);
                let o = offset as usize;
                bank[o..o + 8].copy_from_slice(&ptr.to_le_bytes());
                if size == 16 {
                    bank[o + 8..o + 16].copy_from_slice(&(len / 4).to_le_bytes());
                }
            }
            unsafe { result::memcpy_htod_sync(globals, &bank) }.map_err(map)?;
            kernels.push(Kernel { module, function, indirect: (gs, per_sm.max(1) as u32 * sms.max(1)), threads });
        }
        let words = status_words(state.sleep_mask.len());
        let mut slots = Vec::with_capacity(RING);
        for _ in 0..RING {
            let host = unsafe { result::malloc_host(words * 4, 0) }.map_err(map)? as *mut u8;
            let event = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(map)?;
            slots.push(Slot { host, event, steps_done: 0, ran: false });
        }
        Ok(Self {
            explicit,
            sim,
            stream,
            kernels,
            params,
            work,
            list_args,
            spare,
            layout: state.layout,
            work_layout: state.work_layout,
            categories: state.categories,
            mask: state.mask,
            margin: state.margin,
            hull_geom: state.hull_geom,
            piece_records: state.piece_records,
            clock: state.clock,
            sleep_mask: state.sleep_mask,
            any_awake: state.any_awake,
            slots,
            in_flight: VecDeque::new(),
            failed: None,
            timing_left: 0,
            events: Vec::new(),
            launches: 0,
            ring_waits: 0,
        })
    }

    pub fn sim(&mut self) -> &mut CudaSim {
        &mut self.sim
    }

    pub fn explicit(&mut self) -> &mut CudaExplicitBeams {
        &mut self.explicit
    }

    pub fn clock(&self) -> cw::Clock {
        self.clock
    }

    /// Put the clock back (a replay's rounds from one vehicle state: the explicit
    /// state's snapshot is the explicit host's, [`CudaExplicitBeams::restore`]).
    pub fn set_clock(&mut self, clock: cw::Clock) {
        self.clock = clock;
    }

    pub fn device_name(&self) -> &str {
        self.sim.device_name()
    }

    pub fn coupling_layout(&self) -> CouplingLayout {
        self.layout
    }

    pub fn work_layout(&self) -> cw::WorkLayout {
        self.work_layout
    }

    /// Whether this step must run for the vehicles: vehicle sleep is off, or some
    /// vehicle was awake when last seen.
    fn needs_step(&self) -> bool {
        !self.explicit.sleep_on() || self.any_awake
    }

    fn launch(&mut self, entry: cw::Entry, groups: Option<u32>) -> Result<(), ExecError> {
        let k = &self.kernels[entry.index()];
        let r = match groups {
            Some(0) => return Ok(()),
            Some(n) => unsafe { result::launch_kernel(k.function, (n, 1, 1), (k.threads, 1, 1), 0, self.stream, &mut []) },
            None => {
                // The proxied list's count, read on the GPU (`{entry}_gs`): no host wait.
                let (f, blocks) = k.indirect;
                let mut counts = self.list_args;
                let mut params = [&mut counts as *mut sys::CUdeviceptr as *mut c_void];
                unsafe { result::launch_kernel(f, (blocks, 1, 1), (k.threads, 1, 1), 0, self.stream, &mut params) }
            }
        };
        r.map_err(|e| ExecError::Backend(format!("CUDA: launch {}: {e:?}", entry.name())))?;
        self.launches += 1;
        Ok(())
    }

    fn buf(&self, b: cw::Buf) -> sys::CUdeviceptr {
        match b {
            cw::Buf::Work => self.work,
            cw::Buf::ListArgs => self.list_args,
            cw::Buf::Status => unreachable!("the CUDA host reads the status with every step"),
        }
    }

    fn stamp(&mut self, events: &mut Option<Vec<sys::CUevent>>) -> Result<(), ExecError> {
        if let Some(list) = events {
            let e = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(backend)?;
            unsafe { result::event::record(e, self.stream) }.map_err(backend)?;
            list.push(e);
        }
        Ok(())
    }

    /// Replay the shared plan's commands; `steps`: the explicit frame's at [`cw::Cmd::Frame`].
    fn replay(&mut self, cmds: &[cw::Cmd], steps: u32, events: &mut Option<Vec<sys::CUevent>>) -> Result<(), ExecError> {
        for c in cmds {
            match *c {
                cw::Cmd::Clear { offset, size } => {
                    unsafe { result::memset_d8_async(self.work + offset, 0, size as usize, self.stream) }.map_err(backend)?
                }
                cw::Cmd::Copy { src, src_offset, dst, dst_offset, size } => unsafe {
                    result::memcpy_dtod_async(self.buf(dst) + dst_offset, self.buf(src) + src_offset, size as usize, self.stream)
                }
                .map_err(backend)?,
                cw::Cmd::Pass { .. } | cw::Cmd::Mark(_) => {}
                cw::Cmd::Dispatch { entry, groups } => self.launch(entry, Some(groups))?,
                cw::Cmd::Indirect { entry } => self.launch(entry, None)?,
                cw::Cmd::Frame => {
                    self.stamp(events)?;
                    // The rigid bodies push the vehicles: vehicle contact's frame
                    // bounds are planned from this window's speeds.
                    self.explicit.planner_mut().request_replan();
                    self.explicit.frame(steps)?;
                    self.stamp(events)?;
                }
            }
        }
        Ok(())
    }

    /// The window's parameters, queued on the stream.
    fn write_params(&mut self, steps: u32) -> Result<(), ExecError> {
        let sleep_base = match (self.explicit.layout().sleep, self.explicit.sleep_on()) {
            (Some(s), true) => s.base,
            _ => 0,
        };
        let p = cw::params(&cw::ParamsInput {
            layout: &self.layout,
            work: &self.work_layout,
            colliders: self.sim.collider_count,
            steps,
            margin: self.margin,
            windows: self.clock.windows,
            categories: self.categories,
            mask: self.mask,
            ingest_rows: 0,
            serial_plan: false,
            sleep_base,
            hull_geom: self.hull_geom,
            // The rigid host keeps no resting table for walkers yet (its awake-indexed
            // broadphase and `pose_pack` are not on CUDA): every collider is scanned,
            // the same proxies (engine-world `set_explicit_plan_scan`).
            rest: None,
            retire_base: self.explicit.retire_base(),
            piece_records: self.piece_records,
            loop_budget: plan::loop_budget_override(),
        });
        unsafe { result::memcpy_htod_async(self.params, bytemuck::bytes_of(&p), self.stream) }.map_err(backend)
    }

    /// Queue one World step of `dt` seconds. Does not wait (unless every slot of the
    /// status ring is in flight: then for the oldest). An error a read of an earlier
    /// step found (a rigid overflow, a loop cap, a coupling room exceeded) is returned
    /// here, one step late.
    pub fn step(&mut self, dt: f32) -> Result<(), ExecError> {
        unsafe { result::ctx::set_current(self.sim.ctx) }.map_err(backend)?;
        self.adopt(false)?;
        let mut events = (self.timing_left > 0).then(Vec::new);
        self.timing_left = self.timing_left.saturating_sub(1);
        self.stamp(&mut events)?;
        // Step 1: the last window's impulses, applied by the rigid step.
        let emit = self.clock.take_emit();
        match emit {
            Some(true) => self.sim.mark_gpu_impulses(),
            Some(false) => self.sim.schedule_gpu_impulses(),
            None => {}
        }
        if self.needs_step() {
            self.sim.request_step();
        }
        let ran = self.sim.next_step_runs();
        let steps = if ran { self.clock.window_steps(dt) } else { 0 };
        if ran {
            self.write_params(steps)?;
        }
        if emit.is_some() {
            self.replay(&cw::emit(), 0, &mut None)?;
        }
        self.stamp(&mut events)?;
        self.sim.step_submit()?;
        debug_assert_eq!(ran, self.sim.last_step_ran());
        self.stamp(&mut events)?;
        if ran {
            // Step 3: the next window's plan, frame and chunk loads.
            let cmds = cw::frame(&self.work_layout, cw::FrameOpts { query: false, sleep: self.explicit.sleep_on() });
            self.replay(&cmds, steps, &mut events)?;
            let _ = self.clock.window_done(steps, self.needs_step());
        } else {
            self.stamp(&mut events)?;
            self.stamp(&mut events)?;
        }
        self.read_status(ran)?;
        self.stamp(&mut events)?;
        if let Some(e) = events {
            self.events.push(e.try_into().expect("six events a timed step"));
        }
        Ok(())
    }

    /// Queue this step's status words into a free ring slot.
    fn read_status(&mut self, ran: bool) -> Result<(), ExecError> {
        if self.in_flight.len() == RING {
            // Every slot in flight: wait for the oldest (the wgpu ring's rule).
            let oldest = *self.in_flight.front().expect("in flight");
            unsafe { result::event::synchronize(self.slots[oldest].event) }.map_err(backend)?;
            self.ring_waits += 1;
            self.adopt(false)?;
        }
        let k = (0..RING).find(|k| !self.in_flight.contains(k)).expect("a free slot");
        let counters = std::mem::size_of::<StepCounters>();
        let c4 = counters.div_ceil(4) * 4;
        let masks = self.sleep_mask.len() * 4;
        let host = self.slots[k].host;
        let (acc, _) = self.explicit.buffer(plan::BIND_ACC);
        let cap = u64::from(self.explicit.layout().status.cap);
        unsafe {
            let dst = |at: usize, len: usize| std::slice::from_raw_parts_mut(host.add(at), len);
            result::memcpy_dtoh_async(dst(0, counters), self.sim.buffer(7).0, self.stream).map_err(backend)?;
            result::memcpy_dtoh_async(dst(c4, 4), acc + cap, self.stream).map_err(backend)?;
            result::memcpy_dtoh_async(dst(c4 + 4, 8), self.work, self.stream).map_err(backend)?;
            if masks > 0 {
                result::memcpy_dtoh_async(dst(c4 + 12, masks), self.work + 4 * self.work_layout.sleep_mask_word, self.stream).map_err(backend)?;
            }
            result::event::record(self.slots[k].event, self.stream).map_err(backend)?;
        }
        self.slots[k].steps_done = self.sim.steps_done;
        self.slots[k].ran = ran;
        self.in_flight.push_back(k);
        Ok(())
    }

    /// Adopt, oldest first, the reads that have arrived (`all`: every one; the caller
    /// has waited for the stream).
    fn adopt(&mut self, all: bool) -> Result<(), ExecError> {
        while let Some(&k) = self.in_flight.front() {
            if !all && unsafe { result::event::query(self.slots[k].event) }.is_err() {
                break;
            }
            self.in_flight.pop_front();
            let counters = std::mem::size_of::<StepCounters>();
            let c4 = counters.div_ceil(4) * 4;
            let words = status_words(self.sleep_mask.len());
            let bytes = unsafe { std::slice::from_raw_parts(self.slots[k].host, words * 4) };
            let c: StepCounters = bytemuck::pod_read_unaligned(&bytes[..counters]);
            let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            if self.slots[k].ran {
                self.sim.adopt_counters(c, self.slots[k].steps_done);
                if c.overflow_flags != 0 && self.failed.is_none() {
                    self.failed = Some(format!("rigid step {}: GPU overflow flags=0x{:x}", self.slots[k].steps_done, c.overflow_flags));
                }
            }
            let step = self.slots[k].steps_done;
            if let Err(e) = plan::explicit_loop_cap_hit(word(c4) as i32) {
                self.failed.get_or_insert(format!("World step {step}: explicit vehicles: {e}"));
            }
            if let Err(e) = cw::status_error(word(c4 + 8), self.layout.cap) {
                self.failed.get_or_insert(format!("World step {step}: {e}"));
            }
            if self.explicit.sleep_on() && self.slots[k].ran {
                let nv = self.work_layout.vehicles as usize;
                for (i, m) in self.sleep_mask.iter_mut().enumerate() {
                    *m = word(c4 + 12 + 4 * i);
                }
                let full = |k: usize| if 32 * (k + 1) <= nv { u32::MAX } else { (1u32 << (nv % 32)) - 1 };
                self.any_awake = self.sleep_mask.iter().enumerate().any(|(k, &w)| w != full(k));
            }
        }
        match self.failed.take() {
            Some(e) => Err(ExecError::Backend(e)),
            None => Ok(()),
        }
    }

    /// Block until every queued step has run and adopt its status: `Err` on a rigid
    /// overflow, a loop held to its cap, a coupling room exceeded or a contact error.
    pub fn wait(&mut self) -> Result<(), ExecError> {
        unsafe { result::ctx::set_current(self.sim.ctx) }.map_err(backend)?;
        unsafe { result::stream::synchronize(self.stream) }.map_err(backend)?;
        self.adopt(true)?;
        self.explicit.wait()
    }

    /// Record the GPU time of each of the next `steps` steps ([`Self::step_times_ms`]).
    pub fn enable_step_timing(&mut self, steps: u32) {
        self.timing_left = steps;
    }

    /// GPU milliseconds of each timed step, by part (waits); timing restarts empty.
    pub fn step_times_ms(&mut self) -> Result<Vec<StepTimes>, ExecError> {
        self.wait()?;
        let mut out = Vec::new();
        for e in std::mem::take(&mut self.events) {
            let ms = |a: usize, b: usize| unsafe { result::event::elapsed(e[a], e[b]) }.map(f64::from).map_err(backend);
            out.push(StepTimes { total: ms(0, 5)?, emit: ms(0, 1)?, rigid: ms(1, 2)?, plan: ms(2, 3)?, frame: ms(3, 4)?, loads: ms(4, 5)? });
            for ev in e {
                unsafe {
                    let _ = result::event::destroy(ev);
                }
            }
        }
        self.timing_left = 0;
        Ok(out)
    }

    /// The asleep bits the last adopted read reported (a word per 32 vehicles).
    pub fn sleep_mask(&self) -> &[u32] {
        &self.sleep_mask
    }

    fn read_work(&mut self, offset: u64, words: usize) -> Result<Vec<u32>, ExecError> {
        let mut out = vec![0u32; words];
        unsafe {
            result::memcpy_dtoh_async(bytemuck::cast_slice_mut::<u32, u8>(&mut out), self.work + offset, self.stream).map_err(backend)?;
            result::stream::synchronize(self.stream).map_err(backend)?;
        }
        Ok(out)
    }

    /// The coupling's status word now (waits; tests).
    pub fn coupling_status(&mut self) -> Result<u32, ExecError> {
        self.wait()?;
        Ok(self.read_work(0, 2)?[1])
    }

    /// The coupling's work buffer and indirect arguments (waits): with the clock, the
    /// coupling's carried state.
    pub fn coupling_words(&mut self) -> Result<(Vec<u32>, [u32; 4]), ExecError> {
        self.wait()?;
        let work = self.read_work(0, (self.work_layout.bytes() / 4) as usize)?;
        let mut args = [0u32; 4];
        unsafe {
            result::memcpy_dtoh_async(bytemuck::cast_slice_mut::<u32, u8>(&mut args), self.list_args, self.stream).map_err(backend)?;
            result::stream::synchronize(self.stream).map_err(backend)?;
        }
        Ok((work, args))
    }

    /// Per vehicle, its last window's proxy shapes as (collider, impulse N s, angular
    /// impulse about the shape's reference point, that point): what the next step's
    /// emit puts on the bodies (engine-world's `explicit_window_impulses`; waits).
    pub fn window_impulses(&mut self) -> Result<Vec<Vec<(u32, [f32; 3], [f32; 3], [f32; 3])>>, ExecError> {
        self.wait()?;
        let l = self.layout;
        let misc = self.explicit.download_words(plan::BIND_MISC)?;
        let rec = |r: u32| -> [u32; 4] {
            let i = (r * 4) as usize;
            [misc[i], misc[i + 1], misc[i + 2], misc[i + 3]]
        };
        let f3 = |r: [u32; 4]| [f32::from_bits(r[0]), f32::from_bits(r[1]), f32::from_bits(r[2])];
        Ok((0..l.vehicles)
            .map(|v| {
                let n = rec(l.vehicle(v))[1];
                (0..n)
                    .map(|k| {
                        let at = l.shape(v, k);
                        (rec(at)[2], f3(rec(at + 4)), f3(rec(at + 5)), f3(rec(at + 6)))
                    })
                    .collect()
            })
            .collect())
    }
}

/// The coupling's own state: what [`CudaWorld::from_parts`] starts from.
#[derive(Clone, Debug)]
pub struct CouplingState {
    pub layout: CouplingLayout,
    pub work_layout: cw::WorkLayout,
    pub categories: u16,
    pub mask: u16,
    /// The proxy plan's reach (m), the wheels' rays' included.
    pub margin: f32,
    pub hull_geom: bool,
    pub piece_records: (u32, u32),
    pub clock: cw::Clock,
    pub sleep_mask: Vec<u32>,
    pub any_awake: bool,
    /// The work buffer's bytes ([`cw::WorkLayout`]: the vehicle table, the bounds, the
    /// proxied list, the broadphase proxies) and the proxied list's indirect arguments.
    pub work: Vec<u8>,
    pub list_args: [u32; 4],
}

impl Drop for CudaWorld {
    fn drop(&mut self) {
        unsafe {
            if result::ctx::set_current(self.sim.ctx).is_err() {
                return;
            }
            let _ = result::stream::synchronize(self.stream);
            for e in self.events.drain(..).flatten() {
                let _ = result::event::destroy(e);
            }
            for s in &self.slots {
                let _ = result::event::destroy(s.event);
                let _ = result::free_host(s.host as *mut c_void);
            }
            for k in &self.kernels {
                let _ = result::module::unload(k.module);
            }
            for &b in [self.params, self.work, self.list_args].iter().chain(&self.spare) {
                let _ = result::free_sync(b);
            }
        }
    }
}

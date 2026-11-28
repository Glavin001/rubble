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
//! CUDA has no indirect dispatch. Each kernel has a second entry, `{entry}_gs`
//! (engine-slang-build `grid_stride_entry`), whose blocks walk the workgroups the GPU
//! wrote in its 12-byte slot of binding 23: an indirect command is that entry over a
//! fixed grid (the blocks the device keeps resident) with a pointer to the slot, queued
//! without waiting. No indirect kernel writes binding 23, so every block reads the
//! counts the slot held at launch, as an indirect dispatch would. With
//! [`CudaSim::set_device_indirect`] off (`RUBBLE_CUDA_HOST_INDIRECT=1`, measurement), the
//! host reads binding 23 back after a kernel that may have written it (a wait for the
//! stream) and launches the counted grid. Every other step input is forwarded on the
//! GPU, as on Metal.
//!
//! Needs only the NVIDIA driver at run time (`libcuda` is loaded dynamically).
//! Build hosts without nvcc embed no device code; [`CudaSim::available`] is then false.

use cudarc::driver::{result, sys, DriverError};
use engine_exec::rigid_host;
use engine_exec::rigid_plan::{
    self, Cmd, StepInputs, StepModes, BIND_COUNT, COUNTER_UPLOAD, PASS_ALIGN,
};
use engine_exec::{ExecError, PhaseTimingsMs, PhysicsBackend};
use engine_schema::*;
use std::ffi::{c_void, CString};

pub mod table;
pub use table::{CudaKernel, CUDA_KERNELS, CUDA_TERRAIN_KERNELS, PTX_ARCH, SASS_ARCHS};

pub mod explicit;
pub use explicit::{CudaExplicitBeams, CudaSnapshot, CUDA_EXPLICIT_KERNELS, CUDA_LOOP_CAP_TEST};
pub mod world;
#[cfg(feature = "stress")]
pub mod fracture;
pub use world::{CouplingDef, CouplingState, CudaWorld, StepTimes, CUDA_COUPLING_KERNELS};

/// Binding 23: colour slots, work slots and convex slots, three words each.
const INDIRECT_BINDING: u32 = 23;
const PASS_BINDING: u32 = 22;

fn backend(e: DriverError) -> ExecError {
    ExecError::Backend(format!("CUDA: {e:?}"))
}

/// `bytes` as records (any alignment).
fn pods<T: bytemuck::Pod>(bytes: &[u8]) -> Vec<T> {
    bytes.chunks_exact(std::mem::size_of::<T>()).map(bytemuck::pod_read_unaligned).collect()
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
    /// The indirect entry (`{entry}_gs`) and the blocks a launch of it runs.
    indirect: Option<(sys::CUfunction, u32)>,
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
    /// What the colliders turn on (`rigid_host::ColliderFlags`).
    col_flags: rigid_host::ColliderFlags,
    /// Each body's reach, for joints' default torque arm.
    body_reach: Vec<f32>,
    joint_count: u32,
    free_joints: Vec<u32>,
    /// A height field in the hull arena has a soil (`rigid_host::records_have_soil`).
    has_soil: bool,
    grid_levels: u32,
    sleep_island_rounds: u32,
    last_counters: StepCounters,
    last_timings: PhaseTimingsMs,
    steps_done: u32,
    counted_steps: u32,
    contacts_parity: u32,
    /// A host's GPU pass appended impulse records for the next step (the explicit
    /// vehicles' emit: [`Self::mark_gpu_impulses`]).
    gpu_impulses: rigid_host::GpuImpulses,
    /// The next step runs although the last counters show nothing awake.
    wake_pending: bool,
    /// The last [`Self::step_submit`] queued a step (false: skipped, every body asleep).
    last_step_ran: bool,
    /// Host copy of binding 23; stale once a kernel that binds it is queued.
    indirect: Vec<u32>,
    indirect_dirty: bool,
    /// Host readbacks of binding 23 (each one waits for the stream).
    pub indirect_syncs: u64,
    /// Indirect commands read their counts on the GPU (`{entry}_gs`).
    device_indirect: bool,
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

        let sms = unsafe {
            result::device::get_attribute(
                dev,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
        }
        .map_err(map)? as u32;
        let mut kernels = Vec::with_capacity(CUDA_KERNELS.len());
        for k in CUDA_KERNELS {
            let module = unsafe { result::module::load_data(k.image.as_ptr() as *const c_void) }
                .map_err(|e| anyhow::anyhow!("CUDA: load module {}: {e:?}", k.entry))?;
            let function = unsafe { result::module::get_function(module, CString::new(k.entry)?) }
                .map_err(|e| anyhow::anyhow!("CUDA: function {}: {e:?}", k.entry))?;
            let indirect = match unsafe {
                result::module::get_function(module, CString::new(format!("{}_gs", k.entry))?)
            } {
                Ok(f) => {
                    let per_sm = unsafe {
                        result::occupancy::max_active_block_per_multiprocessor(
                            f,
                            (k.threads[0] * k.threads[1] * k.threads[2]) as i32,
                            0,
                        )
                    }
                    .map_err(|e| anyhow::anyhow!("CUDA: occupancy {}: {e:?}", k.entry))?;
                    Some((f, per_sm.max(1) as u32 * sms))
                }
                Err(_) => None,
            };
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
                indirect,
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
            col_flags: Default::default(),
            body_reach: Vec::new(),
            joint_count: 0,
            free_joints: Vec::new(),
            has_soil: false,
            grid_levels: 0,
            sleep_island_rounds: 8,
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            steps_done: 0,
            counted_steps: 0,
            contacts_parity: 0,
            gpu_impulses: Default::default(),
            wake_pending: false,
            last_step_ran: false,
            indirect,
            indirect_dirty: true,
            indirect_syncs: 0,
            device_indirect: std::env::var("RUBBLE_CUDA_HOST_INDIRECT").as_deref() != Ok("1"),
            sync_each: std::env::var("ENGINE_CUDA_SYNC").as_deref() == Ok("1"),
            last_kernel: "",
            profiling: false,
            phase_events: Vec::new(),
        })
    }

    /// The rigid scene of a portable World (`engine_exec::world_portable`): its bodies
    /// (current and previous rows), colliders, hull arena, joints, materials and
    /// parameters in a world of its capacity, with this host's own step modes. The
    /// contacts start cold (the file holds the scene, not the solver's caches).
    pub fn from_portable(w: &engine_exec::world_portable::PortableWorld) -> anyhow::Result<Self> {
        let r = &w.header.rigid;
        let mut sim = Self::new(r.capacity.clone())?;
        let e = |e: ExecError| anyhow::anyhow!("{e}");
        sim.set_params(r.params);
        sim.set_sleep_island_rounds(r.sleep_island_rounds);
        let mats: Vec<Material> = pods(w.blob("materials"));
        anyhow::ensure!(mats.len() <= 64, "{} materials, the table holds 64", mats.len());
        sim.set_materials(&mats).map_err(e)?;
        let hulls = w.blob("hulls");
        anyhow::ensure!(
            hulls.len() as u64 <= sim.sizes[18],
            "the hull arena's {} records exceed the capacity's {}",
            hulls.len() / 16,
            sim.sizes[18] / 16
        );
        let records: Vec<[u32; 4]> = pods(hulls);
        sim.has_soil |= rigid_host::records_have_soil(&records);
        sim.upload(18, 0, hulls).map_err(e)?;
        let bodies: Vec<BodyState> = pods(w.blob("bodies"));
        anyhow::ensure!(bodies.len() as u32 <= r.capacity.max_bodies, "more bodies than the capacity");
        sim.add_bodies(&bodies);
        let stride = std::mem::size_of::<BodyState>() as u64;
        sim.upload(0, u64::from(r.capacity.max_bodies) * stride, w.blob("bodies_previous")).map_err(e)?;
        let colliders: Vec<Collider> = pods(w.blob("colliders"));
        anyhow::ensure!(colliders.len() as u32 <= r.capacity.max_colliders, "more colliders than the capacity");
        sim.add_colliders(&colliders);
        anyhow::ensure!(r.joint_count <= r.capacity.max_joints, "more joints than the capacity");
        if r.joint_count > 0 {
            sim.upload(5, rigid_host::joint_offset(0), w.blob("joints")).map_err(e)?;
        }
        sim.joint_count = r.joint_count;
        sim.free_joints = r.free_joints.clone();
        Ok(sim)
    }

    /// Indirect commands read their workgroup counts on the GPU (the default), or
    /// the host reads binding 23 back first (`false`: the old path, for A/Bs).
    pub fn set_device_indirect(&mut self, on: bool) {
        self.device_indirect = on;
    }

    pub fn device_indirect(&self) -> bool {
        self.device_indirect
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Driver calls act on the calling thread's current context.
    fn bind_ctx(&self) -> Result<(), ExecError> {
        unsafe { result::ctx::set_current(self.ctx) }.map_err(backend)
    }

    /// Binding `slot`'s device buffer and its bytes (a host pass that binds the rigid
    /// buffers beside its own: the explicit vehicles' coupling, [`crate::CudaWorld`]).
    pub(crate) fn buffer(&self, slot: usize) -> (sys::CUdeviceptr, u64) {
        (self.bufs[slot], self.sizes[slot].max(16))
    }

    pub fn body_count(&self) -> u32 {
        self.body_count
    }

    pub fn collider_count(&self) -> u32 {
        self.collider_count
    }

    /// A host's GPU pass appended impulse records for the next step after the host's
    /// (`IMPULSE_COUNT` atomics, then `IMPULSE_FIRST` set to the count): the step
    /// schedules `apply_impulses` and runs even if nothing is awake (as
    /// `WgpuSim::mark_gpu_impulses`).
    pub fn mark_gpu_impulses(&mut self) {
        self.gpu_impulses.mark();
        self.wake_pending = true;
    }

    /// [`Self::mark_gpu_impulses`] without the wake: the next step that runs applies the
    /// records (as `WgpuSim::schedule_gpu_impulses`).
    pub fn schedule_gpu_impulses(&mut self) {
        self.gpu_impulses.mark();
    }

    /// The next step runs even if the last counters show nothing awake.
    pub fn request_step(&mut self) {
        self.wake_pending = true;
    }

    /// Whether the next [`Self::step_submit`] queues a step: not when the counters
    /// describe the latest step, show no body awake and nothing asked for one.
    pub fn next_step_runs(&self) -> bool {
        !(self.steps_done > 0
            && self.counted_steps == self.steps_done
            && self.last_counters.awake_count == 0
            && self.body_count > 0
            && !self.wake_pending)
    }

    /// Whether the last [`Self::step_submit`] queued a step.
    pub fn last_step_ran(&self) -> bool {
        self.last_step_ran
    }

    /// Counters read without [`Self::drain`]'s wait (a host's own late read of binding
    /// 7): those of the step after which `steps_done` steps were done.
    pub(crate) fn adopt_counters(&mut self, c: StepCounters, steps_done: u32) {
        self.last_counters = c;
        self.counted_steps = steps_done;
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
        let (ccd, kinematic) = rigid_host::body_motion_flags(bodies);
        self.motion.ccd |= ccd;
        self.motion.kinematic |= kinematic;
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
        self.upload(
            2,
            u64::from(start) * std::mem::size_of::<Collider>() as u64,
            bytemuck::cast_slice(colliders),
        )
        .expect("upload colliders");
        self.host_cols.extend_from_slice(colliders);
        self.col_flags.add(colliders);
        rigid_host::note_collider_reach(&mut self.body_reach, colliders);
        self.collider_count += count;
        start..self.collider_count
    }

    /// The convex hull arena (binding 18, `HullArena::records`), whole. Its size is the
    /// capacity's `max_hull_vertices` records (the kernels' constant banks hold the
    /// buffer's address): a larger arena is an error, never a truncation.
    pub fn set_hulls(&mut self, arena: &engine_schema::convex::HullArena) -> Result<(), ExecError> {
        let bytes: &[u8] = bytemuck::cast_slice(arena.records());
        if bytes.len() as u64 > self.sizes[18] {
            return Err(ExecError::Backend(format!(
                "CudaSim: hull arena of {} records exceeds the capacity's {} (WorldCapacity::max_hull_vertices)",
                arena.records().len(),
                self.sizes[18] / 16
            )));
        }
        self.bind_ctx()?;
        self.has_soil |= rigid_host::records_have_soil(arena.records());
        self.upload(18, 0, bytes)
    }

    /// A dynamic body of convex hulls (`rigid_host::convex_body`, as WgpuSim), its hulls
    /// inserted into `arena`, which is then uploaded whole.
    pub fn add_convex_body(
        &mut self,
        arena: &mut engine_schema::convex::HullArena,
        hulls: &[engine_schema::convex::ConvexHull],
        pos: glam::Vec3,
        rot: glam::Quat,
        density: f32,
    ) -> anyhow::Result<u32> {
        let (state, mut colliders) = rigid_host::convex_body(arena, hulls, pos, rot, density, 0);
        let body = self.add_bodies(&[state]).start;
        for c in &mut colliders {
            c.body = body;
        }
        self.add_colliders(&colliders);
        self.set_hulls(arena)?;
        Ok(body)
    }

    /// A height field on a static support body at `pos`, `rot` (inserted into `arena`,
    /// which is then uploaded whole): `(body, collider)`, as WgpuSim and MetalSim.
    pub fn add_height_field(
        &mut self,
        arena: &mut engine_schema::convex::HullArena,
        field: &engine_schema::heightfield::HeightField,
        pos: glam::Vec3,
        rot: glam::Quat,
        material: u32,
    ) -> anyhow::Result<(u32, u32)> {
        let base = arena.insert_height_field(field).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let body = self.add_bodies(&[rigid_host::height_field_support(pos, rot)]).start;
        let collider = self
            .add_colliders(&[rigid_host::height_field_collider(body, base, field, material)])
            .start;
        self.set_hulls(arena)?;
        Ok((body, collider))
    }

    /// Joint records (binding 5), in released slots first, else after the last
    /// (`rigid_host::alloc_joint_slot`, as WgpuSim).
    pub fn add_joints(&mut self, joints: &[Joint]) -> Result<Vec<u32>, ExecError> {
        self.bind_ctx()?;
        let mut ids = Vec::with_capacity(joints.len());
        for j in joints {
            let id = rigid_host::alloc_joint_slot(&mut self.joint_count, &mut self.free_joints, self.capacity.max_joints);
            self.upload(5, rigid_host::joint_offset(id), bytemuck::bytes_of(j))?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// A joint from its description, its default torque arm from the bodies' reach.
    pub fn add_joint_desc(&mut self, desc: &engine_schema::JointDesc) -> Result<u32, ExecError> {
        let j = rigid_host::joint_from_desc(desc, &self.host_bodies, &self.body_reach);
        Ok(self.add_joints(&[j])?[0])
    }

    fn reuse_on(&self) -> bool {
        (self.params.flags & FLAG_REUSE_CONTACTS) != 0
    }

    fn step_modes(&self) -> StepModes {
        StepModes {
            manifold_insert: self.modes.manifold_insert && self.reuse_on(),
            // The persist list is written by narrowphase only; convex kernels emit
            // contacts after it, so hull scenes take the full persistence scan (WgpuSim).
            listed_persist: self.modes.listed_persist && !self.col_flags.convex && !self.col_flags.height_field,
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
            joint_count: self.joint_count,
            joint_hash_slots: engine_schema::joint_hash_size(self.joint_count),
            has_convex: self.col_flags.convex,
            has_height_field: self.col_flags.height_field,
            soil: self.has_soil,
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
                impulses: self.gpu_impulses.this_step(0),
                ..self.motion
            },
            soft: Default::default(),
            command_count: 0,
            guidance: false,
            guide_slots: 0,
            explosion_count: 0,
            post_stabilize: self.params.flags & engine_schema::FLAG_POST_STAB != 0,
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
        self.launch_with(logical, grid, pass, None)
    }

    /// [`Self::launch`], or with `counts` (a device address of three words) the
    /// kernel's indirect entry over its resident grid, reading the counts there.
    fn launch_with(
        &mut self,
        logical: &'static str,
        grid: (u32, u32, u32),
        pass: u32,
        counts: Option<sys::CUdeviceptr>,
    ) -> Result<(), ExecError> {
        let idx = Self::kernel_index(logical)?;
        let threads = CUDA_KERNELS[idx].threads;
        self.last_kernel = logical;
        let stream = self.stream;
        let pass_ptr = self.bufs[PASS_BINDING as usize] + u64::from(pass) * PASS_ALIGN;
        let body_groups = self.body_count.div_ceil(64);
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
        if let Some(mut ptr) = counts {
            let (f, resident) = k.indirect.ok_or_else(|| {
                ExecError::Backend(format!("CUDA: {logical} has no indirect entry"))
            })?;
            // The blocks the device keeps resident, or one per 64 bodies when that is
            // more (most indirect passes run a workgroup per 64 awake bodies): a block
            // then rarely walks a second workgroup, and the hardware keeps balancing
            // blocks across SMs. Measured on a T4 (2026-11-21): resident blocks alone
            // were 12 % slower at 100k bodies (the static split's tail), and 4x or 16x
            // the resident blocks at every size 4 % / 46 % slower at 10k (idle blocks).
            // Any grid gives the same result: each workgroup runs once, whichever block.
            let blocks = resident.max(body_groups);
            let mut params = [&mut ptr as *mut sys::CUdeviceptr as *mut c_void];
            unsafe {
                result::launch_kernel(
                    f,
                    (blocks, 1, 1),
                    (threads[0], threads[1], threads[2]),
                    0,
                    stream,
                    &mut params,
                )
            }
            .map_err(|e| ExecError::Backend(format!("CUDA: launch {logical}_gs: {e:?}")))?;
        } else {
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
        }
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
        if !self.next_step_runs() {
            self.counted_steps += 1;
            self.steps_done += 1;
            self.last_step_ran = false;
            return Ok(());
        }
        self.last_step_ran = true;
        let (cell, levels, _) = rigid_host::choose_grid(
            &self.host_cols,
            &self.host_bodies,
            None,
            self.col_flags.convex,
            0.0,
        );
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
                has_filters: self.col_flags.filters,
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
                    if self.device_indirect {
                        let at = self.bufs[INDIRECT_BINDING as usize] + u64::from(slot) * 12;
                        self.launch_with(kernel, (0, 0, 0), pass, Some(at))?;
                    } else {
                        let g = self.indirect_groups(slot)?;
                        self.launch(kernel, g, pass)?;
                    }
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
        self.gpu_impulses.applied();
        self.wake_pending = false;
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

    /// Overwrite bodies `first..` (pose, velocity, mass, flags), current and previous
    /// rows alike, as [`Self::add_bodies`] writes them; the next step runs even if the
    /// last one left no body awake.
    pub fn write_bodies(&mut self, first: u32, bodies: &[BodyState]) {
        self.bind_ctx().expect("CUDA context");
        let count = bodies.len() as u32;
        assert!(
            first <= self.body_count && count <= self.body_count - first,
            "write_bodies {first}..{} outside the {} bodies",
            first + count,
            self.body_count
        );
        let stride = std::mem::size_of::<BodyState>() as u64;
        let bytes = bytemuck::cast_slice(bodies);
        self.upload(0, u64::from(first) * stride, bytes).expect("write bodies");
        self.upload(0, u64::from(self.capacity.max_bodies + first) * stride, bytes)
            .expect("write previous bodies");
        self.host_bodies[first as usize..(first + count) as usize].copy_from_slice(bodies);
        let (ccd, kinematic) = rigid_host::body_motion_flags(bodies);
        self.motion.ccd |= ccd;
        self.motion.kinematic |= kinematic;
        // Not the skip of a step whose counters say nothing is awake.
        self.counted_steps = self.counted_steps.wrapping_sub(1);
    }

    /// The colliders added so far.
    pub fn colliders(&self) -> &[Collider] {
        &self.host_cols
    }

    /// The parallel solve's precondition on the last step
    /// ([`rigid_plan::validate_contact_coloring`], the WebGPU host's check).
    pub fn validate_contact_coloring(&mut self) -> Result<(), String> {
        let bodies = self.download_bodies();
        let colors = self.download_words(12, self.body_count as usize);
        let manifolds: Vec<Manifold> = {
            let n = self.last_counters.manifold_count as usize;
            let words = self.download_words(25, n * std::mem::size_of::<Manifold>() / 4);
            bytemuck::cast_slice(&words).to_vec()
        };
        rigid_plan::validate_contact_coloring(
            &bodies,
            &colors,
            &manifolds,
            &self.host_cols,
            self.capacity.max_colors,
        )
    }

    /// Words in binding `slot` (for [`Self::download_words`] of the whole buffer).
    pub fn binding_words(&self, slot: usize) -> usize {
        (self.sizes[slot] / 4) as usize
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

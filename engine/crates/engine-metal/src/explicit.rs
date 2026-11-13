//! `MetalExplicitBeams`: native Metal host of the explicit node–beam solver.
//!
//! The same Slang kernels (`slang/kernels/explicit_beams.slang`, compiled to MSL by
//! build.rs `explicit_kernels` from `engine_exec::explicit_plan::MODULES`), the same
//! buffer contents ([`ExplicitUpload`]) and the same per-frame command list
//! ([`FramePlanner`]) as the WebGPU host (`engine_webgpu::ExplicitBeamsGpu`): this
//! file only creates Metal objects and replays the commands. Buffers are shared
//! (unified memory), so readback is a copy out of the buffer once the GPU is idle.
//!
//! Math mode: as wgpu's Metal backend compiles the WGSL (MTLCompileOptions
//! defaults, fast math on); `ENGINE_FAST_MATH=off` compiles strict IEEE.
//! Loop caps: the kernel caps its loops in its Slang source (`EB_LOOP_CAP`), so the
//! MSL modules have them too; a loop held to its cap raises the status's loop-cap
//! flag, and frames fail (panic) on it as on a contact error ([`Self::kernel_status`]).

use crate::sim::{gpu_ms, shared, write};
use engine_exec::explicit_plan::{
    self as plan, coupling, Cmd, CouplingCapacity, CouplingLayout, ExplicitLayout, ExplicitUpload, FramePlanner,
    GpuBeam, Kernel, BIND_ARGS, BIND_COUNT, BIND_PARAMS, PARAM_STRIDE,
};
use engine_softbody::explicit_rigid::ProxySet;
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::Vec3;
use metal::*;
use std::collections::{HashMap, VecDeque};

mod table {
    pub struct ExplicitKernel {
        /// The plan module (`engine_exec::explicit_plan::MODULES`).
        pub name: &'static str,
        pub entry: &'static str,
        pub msl: &'static str,
        pub threads: [u32; 3],
        /// (Metal `[[buffer(n)]]`, plan binding)
        pub binds: &'static [(u32, u32)],
    }
    include!(concat!(env!("OUT_DIR"), "/explicit_kernels.rs"));
}
pub use table::{ExplicitKernel, EXPLICIT_KERNELS, LOOP_CAP_TEST};

/// Parameter buffers in flight (frames queued before the host waits).
const PARAM_RING: usize = 3;

struct Pipe {
    pso: ComputePipelineState,
    threads: u64,
    binds: &'static [(u32, u32)],
}

struct InFlight {
    cb: CommandBuffer,
    /// This frame copied the status: check it once the frame completed.
    status: bool,
}

pub struct MetalExplicitBeams {
    device: Device,
    queue: CommandQueue,
    layout: ExplicitLayout,
    planner: FramePlanner,
    /// Bindings 0..BIND_COUNT (binding 0 is the upload's copy; frames bind a ring slot).
    bufs: Vec<Buffer>,
    params: Vec<Buffer>,
    ring: usize,
    in_flight: VecDeque<InFlight>,
    pipes: HashMap<Kernel, Pipe>,
    /// GPU milliseconds of timed frames ([`Self::enable_frame_timing`]).
    timing: Option<(u32, Vec<f64>)>,
    /// Timed frames still to record.
    timed_left: u32,
    timed: VecDeque<bool>,
}

impl MetalExplicitBeams {
    /// Upload `vehicles` (their current state) on the system default Metal device.
    pub fn new(vehicles: &[ExplicitBeams]) -> anyhow::Result<Self> {
        Self::with_coupling(vehicles, None)
    }

    /// [`Self::new`] with room for rigid-body proxies per vehicle (the coupling,
    /// `explicit_beams_rigid.slang`); off until [`Self::set_coupling_on`].
    pub fn with_coupling(vehicles: &[ExplicitBeams], cap: Option<CouplingCapacity>) -> anyhow::Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
        let queue = device.new_command_queue();
        let upload = ExplicitUpload::new(vehicles, cap);
        let layout = upload.layout.clone();
        let bufs: Vec<Buffer> = (0..BIND_COUNT as u32)
            .map(|b| {
                let bytes = upload.bytes(b);
                let buf = shared(&device, bytes.len() as u64);
                write(&buf, 0, &bytes);
                buf.set_label(&format!("explicit {b}"));
                buf
            })
            .collect();
        let params = (0..PARAM_RING)
            .map(|_| {
                let p = shared(&device, plan::params_bytes());
                write(&p, 0, &upload.bytes(BIND_PARAMS));
                p
            })
            .collect();
        let fast = std::env::var("ENGINE_FAST_MATH").as_deref() != Ok("off");
        let options = CompileOptions::new();
        options.set_fast_math_enabled(fast);
        let mut kernels = vec![Kernel::Frame, Kernel::Solo, Kernel::Contact, Kernel::ContactBig];
        kernels.extend((0..plan::VV_PASSES.len()).map(Kernel::Vv));
        if layout.wide_ok {
            kernels.extend((0..plan::WIDE_PASSES.len()).map(Kernel::Wide));
        }
        let mut pipes = HashMap::new();
        for k in kernels {
            let name = k.module_for(layout.aero, layout.lanes, layout.surface);
            let t = EXPLICIT_KERNELS
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| anyhow::anyhow!("explicit kernel {name} not built"))?;
            let lib = device
                .new_library_with_source(t.msl, &options)
                .map_err(|e| anyhow::anyhow!("compile {name}: {e}"))?;
            let func = lib
                .get_function(t.entry, None)
                .map_err(|e| anyhow::anyhow!("function {name}: {e}"))?;
            let pso = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow::anyhow!("pipeline {name}: {e}"))?;
            let threads = u64::from(t.threads[0]);
            anyhow::ensure!(
                pso.max_total_threads_per_threadgroup() >= threads,
                "{name}: {threads} lanes per workgroup, the device allows {}",
                pso.max_total_threads_per_threadgroup()
            );
            pipes.insert(k, Pipe { pso, threads, binds: t.binds });
        }
        let planner = FramePlanner::new(&layout);
        Ok(Self {
            device,
            queue,
            layout,
            planner,
            bufs,
            params,
            ring: 0,
            in_flight: VecDeque::new(),
            pipes,
            timing: None,
            timed_left: 0,
            timed: VecDeque::new(),
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// The shared plan's layout of this fleet.
    pub fn layout(&self) -> &ExplicitLayout {
        &self.layout
    }

    pub fn vehicle_count(&self) -> usize {
        self.layout.spans.len()
    }

    /// Vehicle–vehicle contact (on by default; runs with more than one vehicle).
    pub fn set_vehicle_contact(&mut self, on: bool) {
        self.planner.set_vehicle_contact(on);
    }

    pub fn contact_active(&self) -> bool {
        self.planner.contact_active()
    }

    /// The frame bounds margin (m), as `ExplicitBeamsGpu::set_contact_margin`.
    pub fn set_contact_margin(&mut self, metres: f32) {
        self.planner.set_contact_margin(metres);
    }

    /// Wide passes (one thread per element over every vehicle) or one workgroup
    /// per vehicle; returns whether wide is on.
    pub fn set_wide(&mut self, on: bool) -> bool {
        self.planner.set_wide(on)
    }

    pub fn is_wide(&self) -> bool {
        self.planner.is_wide()
    }

    /// Record the GPU time of each of the next `frames` frames ([`Self::frame_times_ms`]).
    pub fn enable_frame_timing(&mut self, frames: u32) {
        self.timing = Some((frames, Vec::new()));
        self.timed_left = frames;
    }

    /// GPU milliseconds of each timed frame (waits for them); timing restarts empty.
    pub fn frame_times_ms(&mut self) -> Vec<f64> {
        self.wait();
        self.timed_left = 0;
        self.timing.take().map(|(_, t)| t).unwrap_or_default()
    }

    /// Submit one frame of `steps` steps (the plan's commands). Does not wait.
    pub fn frame(&mut self, steps: u32) {
        if self.layout.spans.is_empty() || steps == 0 {
            return;
        }
        self.retire(false);
        while self.in_flight.len() >= PARAM_RING {
            self.retire_oldest();
        }
        let slot = self.ring;
        self.ring = (self.ring + 1) % PARAM_RING;
        let bytes = self.planner.write_params(steps);
        write(&self.params[slot], 0, &bytes);
        let cmds = self.planner.commands(steps, true);
        let cb = self.queue.new_command_buffer().to_owned();
        let mut enc = Some(cb.new_compute_command_encoder().to_owned());
        for cmd in &cmds {
            match *cmd {
                Cmd::Dispatch { kernel, groups, param_copy } => {
                    let e = enc.get_or_insert_with(|| cb.new_compute_command_encoder().to_owned());
                    let p = self.bind(e, kernel, slot, param_copy);
                    e.dispatch_thread_groups(MTLSize::new(u64::from(groups), 1, 1), MTLSize::new(p, 1, 1));
                }
                Cmd::Indirect { kernel, offset } => {
                    let e = enc.get_or_insert_with(|| cb.new_compute_command_encoder().to_owned());
                    let p = self.bind(e, kernel, slot, 0);
                    e.dispatch_thread_groups_indirect(&self.bufs[BIND_ARGS as usize], offset, MTLSize::new(p, 1, 1));
                }
                Cmd::Copy { src, src_offset, dst, dst_offset, size } => {
                    if let Some(e) = enc.take() {
                        e.end_encoding();
                    }
                    let blit = cb.new_blit_command_encoder();
                    blit.copy_from_buffer(
                        &self.bufs[src as usize],
                        src_offset,
                        &self.bufs[dst as usize],
                        dst_offset,
                        size,
                    );
                    blit.end_encoding();
                }
            }
        }
        if let Some(e) = enc.take() {
            e.end_encoding();
        }
        let status = self.planner.status_due();
        cb.commit();
        let timed = self.timed_left > 0;
        if timed {
            self.timed_left -= 1;
        }
        self.timed.push_back(timed);
        self.in_flight.push_back(InFlight { cb, status });
    }

    /// Set the pipeline and buffers of `kernel`; its lanes per workgroup.
    fn bind(&self, e: &ComputeCommandEncoderRef, kernel: Kernel, slot: usize, param_copy: u32) -> u64 {
        let p = &self.pipes[&kernel];
        e.set_compute_pipeline_state(&p.pso);
        for &(index, b) in p.binds {
            if b == BIND_PARAMS {
                e.set_buffer(u64::from(index), Some(&self.params[slot]), u64::from(param_copy) * PARAM_STRIDE);
            } else {
                e.set_buffer(u64::from(index), Some(&self.bufs[b as usize]), 0);
            }
        }
        p.threads
    }

    /// Finish the oldest frame in flight (waiting for it).
    fn retire_oldest(&mut self) {
        let Some(f) = self.in_flight.pop_front() else { return };
        f.cb.wait_until_completed();
        self.finished(f);
    }

    /// Finish the frames that completed (all of them, waiting, with `wait`).
    fn retire(&mut self, wait: bool) {
        while let Some(f) = self.in_flight.front() {
            let done = matches!(f.cb.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error);
            if !done && !wait {
                break;
            }
            self.retire_oldest();
        }
    }

    fn finished(&mut self, f: InFlight) {
        let timed = self.timed.pop_front().unwrap_or(false);
        if f.cb.status() == MTLCommandBufferStatus::Error {
            panic!("explicit beams (Metal): a frame's command buffer failed");
        }
        if timed {
            if let Some((_, t)) = &mut self.timing {
                t.push(gpu_ms(&f.cb));
            }
        }
        if f.status {
            if let Err(e) = self.layout.kernel_status(&self.acc()) {
                panic!("explicit beams (Metal): {e}");
            }
        }
    }

    /// Block until submitted frames have run. Fails (panics) when one reported a
    /// vehicle-contact error or a loop held to its cap.
    pub fn wait(&mut self) {
        self.retire(true);
    }

    /// The contact status (blocking), after the loop-cap flag.
    pub fn contact_status(&mut self) -> Result<(), String> {
        self.kernel_status()
    }

    /// The kernel status (blocking, without the frames' own check): `Err` when a
    /// kernel loop was held to its cap (`engine_exec::explicit_plan::EXPLICIT_LOOP_CAP`:
    /// a runaway loop, stopped; that frame's results are invalid) or on a contact
    /// error.
    pub fn kernel_status(&mut self) -> Result<(), String> {
        // The frames in flight: retired without their own check (read here).
        while let Some(mut f) = self.in_flight.pop_front() {
            f.cb.wait_until_completed();
            f.status = false;
            self.finished(f);
        }
        self.layout.kernel_status(&self.acc())
    }

    /// The rigid coupling's layout, when built [`Self::with_coupling`].
    pub fn coupling(&self) -> Option<CouplingLayout> {
        self.layout.coupling
    }

    /// Turn the rigid coupling on or off from the next frame (frames then run per
    /// vehicle: the wide passes do not run it). Waits for the frames in flight.
    pub fn set_coupling_on(&mut self, on: bool) {
        assert!(self.layout.coupling.is_some(), "explicit beams: built without coupling room");
        if on {
            self.set_wide(false);
        }
        self.wait();
        write(&self.bufs[plan::BIND_MISC as usize], self.layout.coupling_on_offset(), bytemuck::bytes_of(&(on as u32)));
    }

    /// Write vehicle `v`'s proxies (its accumulators cleared) for the next frames.
    pub fn write_proxies(&mut self, v: usize, set: &ProxySet) -> Result<(), String> {
        let layout = self.layout.coupling.ok_or("explicit coupling: built without coupling room")?;
        let recs = coupling::proxy_records(&layout, set)?;
        self.wait();
        write(&self.bufs[plan::BIND_MISC as usize], u64::from(layout.vehicle(v as u32)) * 16, bytemuck::cast_slice(&recs));
        let zeros = vec![0i32; layout.acc_per_vehicle() as usize];
        write(&self.bufs[plan::BIND_ACC as usize], u64::from(layout.acc(v as u32)) * 4, bytemuck::cast_slice(&zeros));
        Ok(())
    }

    /// Read vehicle `v`'s proxies back into `set` (waits).
    pub fn read_proxies(&mut self, v: usize, set: &mut ProxySet) {
        let layout = self.layout.coupling.expect("coupling room");
        self.wait();
        let misc: Vec<u32> = self.contents(plan::BIND_MISC);
        let at = layout.vehicle(v as u32) as usize * 4;
        coupling::decode_proxies(&layout, &misc[at..at + layout.recs() as usize * 4], set);
    }

    /// Set vehicle `v`'s driver input for its raycast wheels (waits for the frames in
    /// flight; a fleet without raycast wheels ignores it).
    pub fn set_drive_input(&mut self, v: usize, input: engine_softbody::explicit_drive::DriveInput) {
        let Some(d) = self.layout.drive else { return };
        self.wait();
        write(&self.bufs[plan::BIND_MISC as usize], d.input_offset(v), bytemuck::cast_slice(&plan::drive::input_words(input)));
        self.planner.request_replan();
    }

    /// Read vehicle `v`'s raycast wheels' state back into `drive` (waits).
    pub fn read_drive(&mut self, v: usize, drive: &mut engine_softbody::explicit_drive::Drive) {
        let Some(l) = self.layout.drive else { return };
        self.wait();
        let misc: Vec<u32> = self.contents(plan::BIND_MISC);
        let at = l.vehicle(v) as usize * 4;
        plan::drive::decode(&misc[at..at + l.stride as usize * 4], drive);
    }

    /// Set one node's velocity (waits for the frames in flight).
    pub fn set_velocity(&mut self, vehicle: usize, node: u32, v: Vec3) {
        self.wait();
        let offset = self.layout.velocity_offset(vehicle, node);
        write(&self.bufs[plan::BIND_NODES as usize], offset, bytemuck::cast_slice(&v.to_array()));
        self.planner.request_replan();
    }

    fn contents<T: bytemuck::Pod>(&self, b: u32) -> Vec<T> {
        let buf = &self.bufs[b as usize];
        let bytes = unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, buf.length() as usize) };
        bytemuck::pod_collect_to_vec(bytes)
    }

    fn acc(&self) -> Vec<i32> {
        self.contents(plan::BIND_ACC)
    }

    /// Per vehicle, world node positions (waits).
    pub fn positions(&mut self) -> Vec<Vec<Vec3>> {
        self.wait();
        self.layout
            .positions(&self.contents(plan::BIND_NODES), &self.contents(plan::BIND_MISC))
    }

    /// Per vehicle, node velocities (waits).
    pub fn velocities(&mut self) -> Vec<Vec<Vec3>> {
        self.wait();
        self.layout.velocities(&self.contents(plan::BIND_NODES))
    }

    /// Per vehicle, every beam's broken flag and rest length (waits).
    pub fn beams(&mut self) -> Vec<Vec<GpuBeam>> {
        self.wait();
        self.layout.beams(&self.contents(plan::BIND_BEAMS))
    }

    /// Per vehicle its triggered JBeam deform groups (waits).
    pub fn deform_triggered(&mut self) -> Vec<Vec<u32>> {
        self.wait();
        self.layout.deform_triggered(&self.contents(plan::BIND_GROUPS))
    }

    /// The contact groups of the current plan (waits; 0 when contact is not active).
    pub fn contact_groups(&mut self) -> u32 {
        if !self.contact_active() {
            return 0;
        }
        self.wait();
        self.contents::<u32>(BIND_ARGS)[4]
    }
}

impl Drop for MetalExplicitBeams {
    fn drop(&mut self) {
        for f in &self.in_flight {
            f.cb.wait_until_completed();
        }
    }
}

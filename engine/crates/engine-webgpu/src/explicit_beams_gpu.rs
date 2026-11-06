//! Explicit beams on the GPU (`slang/kernels/explicit_beams.slang`): every vehicle
//! in one dispatch per frame, one 256-lane workgroup each, running the origin rebase
//! and all of the frame's fixed steps. The rules are
//! [`engine_softbody::explicit_beams`]'s (the CPU reference); the GPU differs only in
//! summation order, in the self-collision forces (accumulated in 1/32 N fixed point)
//! and in the turbulence noise (a per-node hash instead of the CPU sequence).
//!
//! Build it from spawned [`ExplicitBeams`] (their current state is uploaded), call
//! [`ExplicitBeamsGpu::frame`] once per frame with that frame's steps
//! ([`engine_softbody::explicit_beams::FrameClock`]), and read state back with
//! [`ExplicitBeamsGpu::positions`], [`ExplicitBeamsGpu::velocities`] and
//! [`ExplicitBeamsGpu::beams`].
//!
//! This is the WebGPU host of the shared plan (`engine_exec::explicit_plan`): the
//! buffers' contents, the module choice and each frame's commands come from there,
//! as on the Metal and CUDA hosts; this file only creates the wgpu objects and
//! replays the commands.

use crate::{
    EXPLICIT_BEAMS_AERO_WGSL, EXPLICIT_BEAMS_AERO_WIDE_WGSL, EXPLICIT_BEAMS_CONTACT_AERO_WGSL,
    EXPLICIT_BEAMS_CONTACT_AERO_WIDE_WGSL, EXPLICIT_BEAMS_CONTACT_WGSL,
    EXPLICIT_BEAMS_CONTACT_WIDE_WGSL, EXPLICIT_BEAMS_SOLO_AERO_WGSL, EXPLICIT_BEAMS_SOLO_AERO_WIDE_WGSL,
    EXPLICIT_BEAMS_SOLO_WGSL, EXPLICIT_BEAMS_SOLO_WIDE_WGSL, EXPLICIT_BEAMS_VV_PASSES, EXPLICIT_BEAMS_WGSL,
    EXPLICIT_BEAMS_WIDE_PASSES, EXPLICIT_BEAMS_WIDE_WGSL,
};
use engine_exec::explicit_plan::{
    self as plan, Cmd, ExplicitLayout, ExplicitUpload, FramePlanner, Kernel, PlannerState, BIND_TABLE,
    PARAM_STRIDE, VV_RECS,
};
pub use engine_exec::explicit_plan::{CouplingCapacity, CouplingLayout, GpuBeam};
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::Vec3;
use wgpu::util::DeviceExt;

pub struct ExplicitBeamsGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    nodes: wgpu::Buffer,
    beams: wgpu::Buffer,
    misc: wgpu::Buffer,
    /// gNext (grid ranks, wide-pass scratch, trigger inputs).
    next: wgpu::Buffer,
    /// Where every record lives (the shared plan's).
    layout: ExplicitLayout,
    /// Each frame's parameters and commands (the shared plan's).
    planner: FramePlanner,
    /// The wide passes (one thread per element over every vehicle), when possible.
    wide: Option<Wide>,
    timing: Option<FrameTiming>,
    /// Diagnostic marks after each of the frame's dispatches (`RUBBLE_GPU_TIMING_DISPATCH`,
    /// a host's [`crate::gpu_marks::GpuMarks`]).
    marks: Option<std::sync::Arc<crate::gpu_marks::GpuMarks>>,
    /// The coupling header's state ([`Self::set_coupling_on`], [`Self::set_coupling_cull`]).
    coupling_on: bool,
    coupling_cull_off: bool,
    /// Vehicle–vehicle contact ([`Self::set_vehicle_contact`]).
    contact: Contact,
    /// Contact errors for the host ([`Self::take_contact_errors`]), and those a
    /// `wait` saw, not yet handled.
    contact_errors: std::sync::Mutex<Vec<String>>,
    contact_errors_seen: std::sync::Mutex<Vec<String>>,
    /// The bound buffers (bindings 0..8) and the module choice (aero, 512 lanes),
    /// for the test-only kernel variants ([`Self::set_kernel_variant`]).
    bufs: [wgpu::Buffer; 9],
    /// (aero, a vehicle over WIDE_BEAMS, lanes of the current frame modules).
    module_kind: (bool, bool, u32),
    variant: KernelVariant,
    alts: Vec<(KernelVariant, AltPipes)>,
    probe: Option<wgpu::Buffer>,
    /// The `EB_DET_HITS` variants' hit lists (binding 12).
    det: Option<wgpu::Buffer>,
    acc: wgpu::Buffer,
}

/// Which kernel runs the frames: the shipped one, or (tests only) the frozen
/// baseline (`explicit_beams_baseline.slang`, read from the build's WGSL directory)
/// with or without its race probe. See [`ExplicitBeamsGpu::set_kernel_variant`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelVariant {
    Current,
    Baseline,
    BaselineProbe,
    /// Self-collision and vehicle-contact hits on one lane in a fixed order
    /// (`EB_DET_HITS`): no race, so every frame compares.
    BaselineDet,
    CurrentDet,
    /// Measurement only: the current kernel with phases skipped
    /// ([`ExplicitBeamsGpu::set_ablation`]); results are meaningless.
    Ablate,
    /// The frozen reference (`explicit_beams_ref.slang`, the kernel before the
    /// latency work): the current kernel must leave its state bit for bit.
    Reference,
}

/// A variant's pipelines and bind groups.
struct AltPipes {
    frame: wgpu::ComputePipeline,
    solo: wgpu::ComputePipeline,
    groups: wgpu::ComputePipeline,
    plan: [wgpu::ComputePipeline; 3],
    group: wgpu::BindGroup,
    plan_group: wgpu::BindGroup,
}

/// A copy of every buffer a frame writes, and the host's contact plan state
/// ([`ExplicitBeamsGpu::snapshot`]).
#[doc(hidden)]
pub struct GpuSnapshot {
    copies: Vec<wgpu::Buffer>,
    state: PlannerState,
}

/// Vehicle–vehicle contact's pipelines, buffers and status reads.
struct Contact {
    /// Its pipelines: built only for a fleet of more than one vehicle (a single
    /// vehicle never runs the contact passes: [`FramePlanner`]'s `contact_active`),
    /// which spares a one-vehicle scene the contact frame's compile (the largest
    /// kernel, about a minute cold in Chrome on Metal).
    pipes: Option<ContactPipes>,
    plan_group: wgpu::BindGroup,
    contact_group: wgpu::BindGroup,
    /// Indirect arguments: the solo vehicles (word 0) and the groups (word 4).
    args: wgpu::Buffer,
    /// The status counters (gAcc at byte `status_at`), copied each frame into a
    /// staging buffer and checked once mapped, a frame or two later (no waiting).
    acc: wgpu::Buffer,
    status_at: u64,
    /// The loop-cap flag (gAcc's last word, byte offset): see
    /// `engine_exec::explicit_plan::EXPLICIT_LOOP_CAP`.
    cap_at: u64,
    status: std::sync::Mutex<StatusRing>,
}

/// Vehicle–vehicle contact's pipelines.
struct ContactPipes {
    /// The plan passes (bounds, pairs, group) and the contact groups' entry.
    plan: [wgpu::ComputePipeline; 3],
    groups: wgpu::ComputePipeline,
    /// The solo vehicles' frame (`EB_VV_SOLO`).
    solo: wgpu::ComputePipeline,
}

/// The vehicle-contact passes' bind group layout entries: the frame's bindings,
/// and for the plan passes the indirect arguments ([`plan::BIND_ARGS`]).
fn contact_entries(args: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    let ro = wgpu::BufferBindingType::Storage { read_only: true };
    let mut entries = vec![
        entry(0, wgpu::BufferBindingType::Uniform),
        entry(1, ro),
        entry(2, ro),
    ];
    entries.extend((3..9).map(|b| entry(b, rw)));
    if args {
        entries.push(entry(plan::BIND_ARGS, rw));
    }
    entries
}

/// The wide passes' bind group layout entries (binding 0: one parameter copy per
/// step, picked by the dynamic offset).
fn wide_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: binding == 0,
            min_binding_size: None,
        },
        count: None,
    };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    let ro = wgpu::BufferBindingType::Storage { read_only: true };
    vec![
        entry(0, wgpu::BufferBindingType::Uniform),
        entry(1, ro),
        entry(2, ro),
        entry(3, rw),
        entry(4, rw),
        entry(5, rw),
        entry(6, rw),
        entry(7, rw),
        entry(8, rw),
        entry(BIND_TABLE, wgpu::BufferBindingType::Uniform),
    ]
}

/// A status read: its staging buffer and its map state (0 copied, not yet mapped;
/// 1 map requested; 2 mapped; 3 map failed).
struct StatusRead {
    buffer: wgpu::Buffer,
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

/// Status reads in flight and free staging buffers.
#[derive(Default)]
struct StatusRing {
    flight: Vec<StatusRead>,
    free: Vec<wgpu::Buffer>,
    made: usize,
}

/// Staging buffers for status reads at most.
const STATUS_READS: usize = 4;

/// The error a status read reports: the loop-cap flag (`w[4]`), then the contact
/// counters (outside, overflow, unconverged, too big).
fn status_error(w: &[i32]) -> Result<(), String> {
    crate::explicit_loop_cap_hit(w[4]).map_err(|e| format!("explicit beams kernel: {e}"))?;
    plan::contact_error(&w[..4]).map_err(|e| format!("explicit beams vehicle contact: {e}"))
}

/// The parameter buffer as the non-wide passes bind it: its first copy only (the
/// whole buffer, PARAM_STRIDE × PARAM_COPIES, is past WebGPU's 64 KiB uniform
/// binding limit; those passes read copy 0, the wide ones a dynamic offset).
fn params_window(b: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: b,
        offset: 0,
        size: std::num::NonZeroU64::new(PARAM_STRIDE),
    })
}

/// The WGSL of a plan module ([`engine_exec::explicit_plan::MODULES`]).
fn module_wgsl(name: &str) -> &'static str {
    match name {
        "explicit_beams" => EXPLICIT_BEAMS_WGSL,
        "explicit_beams_wide" => EXPLICIT_BEAMS_WIDE_WGSL,
        "explicit_beams_aero" => EXPLICIT_BEAMS_AERO_WGSL,
        "explicit_beams_aero_wide" => EXPLICIT_BEAMS_AERO_WIDE_WGSL,
        "explicit_beams_solo" => EXPLICIT_BEAMS_SOLO_WGSL,
        "explicit_beams_solo_wide" => EXPLICIT_BEAMS_SOLO_WIDE_WGSL,
        "explicit_beams_solo_aero" => EXPLICIT_BEAMS_SOLO_AERO_WGSL,
        "explicit_beams_solo_aero_wide" => EXPLICIT_BEAMS_SOLO_AERO_WIDE_WGSL,
        "explicit_beams_contact" => EXPLICIT_BEAMS_CONTACT_WGSL,
        "explicit_beams_contact_wide" => EXPLICIT_BEAMS_CONTACT_WIDE_WGSL,
        "explicit_beams_contact_aero" => EXPLICIT_BEAMS_CONTACT_AERO_WGSL,
        "explicit_beams_contact_aero_wide" => EXPLICIT_BEAMS_CONTACT_AERO_WIDE_WGSL,
        _ => EXPLICIT_BEAMS_VV_PASSES
            .iter()
            .chain(EXPLICIT_BEAMS_WIDE_PASSES.iter())
            .find(|(n, _)| *n == name)
            .map(|(_, w)| *w)
            .unwrap_or_else(|| panic!("explicit beams: no WGSL module {name}")),
    }
}

/// The most a frame-bounds error widens the contact margin to (m).
pub const MAX_AUTO_CONTACT_MARGIN: f32 = 16.0;

impl Contact {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        buffers: &[&wgpu::Buffer; 9],
        (contact_wgsl, solo_wgsl): (&str, &str),
        acc: &wgpu::Buffer,
        (status_at, cap_at): (u64, u64),
        args_init: &[u8],
        build_pipes: bool,
    ) -> Self {
        let args_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("explicit-beams-vv-args"),
            contents: args_init,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        // The plan passes bind the indirect arguments (binding 10); the contact
        // entry, dispatched from them, must not.
        let layout_with = |args: bool| {
            let entries = contact_entries(args);
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("explicit-beams-vv"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("explicit-beams-vv"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let mut bind: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 { params_window(b) } else { b.as_entire_binding() },
                })
                .collect();
            if args {
                bind.push(wgpu::BindGroupEntry {
                    binding: plan::BIND_ARGS,
                    resource: args_buffer.as_entire_binding(),
                });
            }
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("explicit-beams-vv"),
                layout: &layout,
                entries: &bind,
            });
            (pipeline_layout, group)
        };
        let make = |entry: &str, wgsl: &str, layout: &wgpu::PipelineLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(crate::uniformize_workgroup_lanes(wgsl).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (plan_layout, plan_group) = layout_with(true);
        let (contact_layout, contact_group) = layout_with(false);
        let pipes = build_pipes.then(|| ContactPipes {
            plan: plan::VV_PASSES.map(|entry| make(entry, module_wgsl(entry), &plan_layout)),
            groups: make(plan::FRAME_ENTRY, contact_wgsl, &contact_layout),
            solo: make(plan::FRAME_ENTRY, solo_wgsl, &contact_layout),
        });
        Self {
            pipes,
            plan_group,
            contact_group,
            args: args_buffer,
            acc: acc.clone(),
            status_at,
            cap_at,
            status: Default::default(),
        }
    }

    /// Copy the status counters into a staging buffer (none free: skip; the
    /// counters keep an error until it is read).
    fn read_status(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder) {
        let mut ring = self.status.lock().unwrap();
        let buffer = match ring.free.pop() {
            Some(b) => b,
            None if ring.made < STATUS_READS => {
                ring.made += 1;
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("explicit-beams-vv-status"),
                    size: 20,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }
            None => return,
        };
        enc.copy_buffer_to_buffer(&self.acc, self.status_at + 16, &buffer, 0, 16);
        enc.copy_buffer_to_buffer(&self.acc, self.cap_at, &buffer, 16, 4);
        ring.flight.push(StatusRead {
            buffer,
            state: Default::default(),
        });
    }

    /// Request the maps of the reads just submitted.
    fn map_status(&self) {
        let ring = self.status.lock().unwrap();
        for r in ring.flight.iter().filter(|r| r.state.load(std::sync::atomic::Ordering::Acquire) == 0) {
            r.state.store(1, std::sync::atomic::Ordering::Release);
            let state = r.state.clone();
            r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                state.store(if res.is_ok() { 2 } else { 3 }, std::sync::atomic::Ordering::Release);
            });
        }
    }

    /// Process finished maps (polling the device without waiting).
    fn poll_status(&self, device: &wgpu::Device) -> Option<String> {
        if self.status.lock().unwrap().flight.is_empty() {
            return None;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        self.ready_error()
    }

    /// The first error among the reads whose maps finished (recycling them).
    fn ready_error(&self) -> Option<String> {
        let mut ring = self.status.lock().unwrap();
        let mut error = None;
        let mut keep = Vec::new();
        for r in std::mem::take(&mut ring.flight) {
            match r.state.load(std::sync::atomic::Ordering::Acquire) {
                2 => {
                    {
                        let view = r.buffer.slice(..).get_mapped_range().expect("mapped");
                        let w: &[i32] = bytemuck::cast_slice(&view);
                        if let Err(e) = status_error(w) {
                            error.get_or_insert(e);
                        }
                    }
                    r.buffer.unmap();
                    ring.free.push(r.buffer);
                }
                3 => ring.free.push(r.buffer),
                _ => keep.push(r),
            }
        }
        ring.flight = keep;
        error
    }
}

/// Timestamps around each frame's pass ([`ExplicitBeamsGpu::enable_frame_timing`]).
struct FrameTiming {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    capacity: u32,
    used: u32,
}

/// A plan command's kernel, for diagnostic marks.
fn kernel_name(cmd: &Cmd) -> &'static str {
    let k = match *cmd {
        Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => kernel,
        Cmd::Copy { .. } => return "copy",
    };
    match k {
        Kernel::Frame => "explicit frame (solo)",
        Kernel::Solo => "explicit solo vehicles",
        Kernel::Contact => "explicit contact groups",
        Kernel::Vv(0) => "explicit vv bounds",
        Kernel::Vv(1) => "explicit vv pairs",
        Kernel::Vv(_) => "explicit vv group plan",
        Kernel::Wide(_) => "explicit wide",
    }
}

/// The wide passes' pipelines and bind group ([`ExplicitBeamsGpu::set_wide`]).
struct Wide {
    passes: Vec<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    /// The kernel's vehicle table (constant memory): static records, then per
    /// vehicle (origin, steps) copied from the misc buffer each frame.
    table: wgpu::Buffer,
}

impl ExplicitBeamsGpu {
    /// Upload `vehicles` (their current state: positions, velocities, forces, beam
    /// state, throttles, origin).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, vehicles: &[ExplicitBeams]) -> Self {
        Self::with_coupling(device, queue, vehicles, None)
    }

    /// Every compute pipeline [`Self::with_coupling`] creates for `vehicles`, as
    /// plain descriptors: a browser host compiles them ahead with
    /// `createComputePipelineAsync` (see [`crate::ComputeSpec`]), largest first.
    pub fn compute_specs(
        vehicles: &[ExplicitBeams],
        coupling: Option<CouplingCapacity>,
    ) -> Vec<crate::ComputeSpec> {
        let layout = ExplicitUpload::new(vehicles, coupling).layout;
        let (aero, lanes) = (layout.aero, layout.lanes);
        let u = crate::uniformize_workgroup_lanes;
        let mut specs = vec![crate::ComputeSpec {
            label: "explicit_beams_frame",
            wgsl: u(module_wgsl(Kernel::Frame.module(aero, lanes))),
            entry: plan::FRAME_ENTRY,
            layout: None,
        }];
        if layout.wide_ok {
            specs.extend(plan::WIDE_PASSES.iter().map(|&entry| crate::ComputeSpec {
                label: entry,
                wgsl: u(module_wgsl(entry)),
                entry,
                layout: Some(wide_entries()),
            }));
        }
        if vehicles.len() > 1 {
            specs.extend(plan::VV_PASSES.iter().map(|&entry| crate::ComputeSpec {
                label: entry,
                wgsl: u(module_wgsl(entry)),
                entry,
                layout: Some(contact_entries(true)),
            }));
            for kernel in [Kernel::Contact, Kernel::Solo] {
                specs.push(crate::ComputeSpec {
                    label: plan::FRAME_ENTRY,
                    wgsl: u(module_wgsl(kernel.module(aero, lanes))),
                    entry: plan::FRAME_ENTRY,
                    layout: Some(contact_entries(false)),
                });
            }
        }
        specs.sort_by(|a, b| b.wgsl.len().cmp(&a.wgsl.len()));
        specs
    }

    /// [`Self::new`] with room for rigid-body proxies per vehicle (the World coupling,
    /// `slang/kernels/explicit_beams_rigid.slang`, [`CouplingLayout`]); off until the
    /// host writes the coupling header's `on` word.
    pub fn with_coupling(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vehicles: &[ExplicitBeams],
        coupling: Option<CouplingCapacity>,
    ) -> Self {
        let upload = ExplicitUpload::new(vehicles, coupling);
        let layout = upload.layout.clone();
        let (aero, big, lanes) = (layout.aero, layout.big, layout.lanes);
        // Large vehicles get twice the lanes (more loads in flight per step), a
        // small fleet 512 (frame_lanes).
        let frame_module = Kernel::Frame.module(aero, lanes);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit_beams"),
            source: wgpu::ShaderSource::Wgsl(crate::uniformize_workgroup_lanes(module_wgsl(frame_module)).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("explicit_beams_frame"),
            layout: None,
            module: &module,
            entry_point: Some(plan::FRAME_ENTRY),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = wgpu::BufferUsages::STORAGE;
        let state = storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let init = |label: &str, b: u32, usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &upload.bytes(b),
                usage,
            })
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-params"),
            size: plan::params_bytes(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&layout.words));
        let statics = init("explicit-beams-static", plan::BIND_STATIC, storage);
        let lists = init("explicit-beams-lists", plan::BIND_LISTS, storage);
        let nodes = init("explicit-beams-nodes", plan::BIND_NODES, state);
        let beams = init("explicit-beams-beams", plan::BIND_BEAMS, state);
        let acc = init("explicit-beams-acc", plan::BIND_ACC, state);
        let groups = init("explicit-beams-groups", plan::BIND_GROUPS, state);
        let misc = init("explicit-beams-misc", plan::BIND_MISC, state);
        let next = init("explicit-beams-grid-next", plan::BIND_NEXT, state);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                (0, &params),
                (1, &statics),
                (2, &lists),
                (3, &nodes),
                (4, &beams),
                (5, &acc),
                (6, &groups),
                (7, &misc),
                (8, &next),
            ]
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: if binding == 0 { params_window(b) } else { b.as_entire_binding() },
            }),
        });
        let buffers = [
            &params, &statics, &lists, &nodes, &beams, &acc, &groups, &misc, &next,
        ];
        let wide = layout
            .wide_ok
            .then(|| Wide::new(device, queue, &buffers, upload.table_records()));
        let contact = Contact::new(
            device,
            &buffers,
            (
                module_wgsl(Kernel::Contact.module(aero, lanes)),
                module_wgsl(Kernel::Solo.module(aero, lanes)),
            ),
            &acc,
            (layout.status.base, layout.status.cap),
            &upload.bytes(plan::BIND_ARGS),
            vehicles.len() > 1,
        );
        let bufs = [
            params.clone(),
            statics.clone(),
            lists.clone(),
            nodes.clone(),
            beams.clone(),
            acc.clone(),
            groups.clone(),
            misc.clone(),
            next.clone(),
        ];
        let planner = FramePlanner::new(&layout);
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            group,
            params,
            nodes,
            beams,
            misc,
            next,
            layout,
            planner,
            timing: None,
            marks: None,
            coupling_on: false,
            coupling_cull_off: false,
            wide,
            contact,
            contact_errors: Default::default(),
            contact_errors_seen: Default::default(),
            bufs,
            module_kind: (aero, big, lanes),
            variant: KernelVariant::Current,
            alts: Vec::new(),
            probe: None,
            det: None,
            acc,
        }
    }

    /// The shared plan's layout of this fleet (where every record lives).
    pub fn layout(&self) -> &ExplicitLayout {
        &self.layout
    }

    /// Vehicle–vehicle contact between every vehicle of the fleet
    /// (`engine/docs/explicit-beams-vehicle-contact-spec.md`): on by default; it
    /// runs when the fleet has more than one vehicle. Off, every vehicle steps
    /// alone (the one-dispatch frame).
    pub fn set_vehicle_contact(&mut self, on: bool) {
        self.planner.set_vehicle_contact(on);
    }

    /// Whether vehicle contact is on ([`Self::set_vehicle_contact`]).
    pub fn vehicle_contact(&self) -> bool {
        self.planner.vehicle_contact()
    }

    /// Whether frames run the vehicle contact passes (on, more than one vehicle).
    pub fn contact_active(&self) -> bool {
        self.planner.contact_active()
    }

    /// The frame bounds margin (m, default [`plan::DEFAULT_CONTACT_MARGIN`]). A contact
    /// plan covers the next frames of at least 34 steps (every frame at 60 fps,
    /// every other at 120): a vehicle whose box, grown by twice the travel of its
    /// fastest node over those steps plus this margin, overlaps no other's is
    /// stepped alone; one that does is stepped with the vehicles it may meet, step
    /// by step. A node leaving those bounds (accelerated past them) could have met
    /// a vehicle unseen: the host then fails (panics) at a later frame or `wait`
    /// ([`Self::contact_status`] at once). A larger margin groups more vehicles
    /// (slower), never changes results.
    pub fn set_contact_margin(&mut self, metres: f32) {
        self.planner.set_contact_margin(metres);
    }

    /// The contact status (blocking): `Err` when a vehicle left its frame bounds,
    /// the pair list overflowed or the grouping did not converge.
    pub fn contact_status(&self) -> Result<(), String> {
        let raw = self.read(&self.contact.acc);
        let w: &[i32] = bytemuck::cast_slice(&raw);
        crate::explicit_loop_cap_hit(self.layout.loop_cap_flag(w))?;
        self.layout.contact_status(w)
    }

    /// The kernel status (blocking): `Err` when a kernel loop reached its cap
    /// ([`crate::EXPLICIT_LOOP_CAP`]: a runaway loop, stopped; that frame's results
    /// are invalid). Frames also check it on their own every few frames
    /// ([`Self::frame`] and [`Self::wait`] fail on it).
    pub fn kernel_status(&self) -> Result<(), String> {
        let raw = self.read(&self.contact.acc);
        let w: &[i32] = bytemuck::cast_slice(&raw);
        crate::explicit_loop_cap_hit(self.layout.loop_cap_flag(w))
    }

    /// The contact groups of the current plan (blocking read; 0 when contact is
    /// not active).
    pub fn contact_groups(&self) -> u32 {
        if !self.contact_active() {
            return 0;
        }
        let raw = self.read(&self.contact.args);
        bytemuck::cast_slice::<u8, u32>(&raw)[4]
    }

    /// Diagnostic (blocking): the member count of each contact group of the current
    /// plan.
    #[doc(hidden)]
    pub fn contact_group_sizes(&self) -> Vec<u32> {
        let n = self.contact_groups() as usize;
        let misc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.misc)).to_vec();
        let at = self.planner.words[46] as usize;
        (0..n).map(|g| misc[(at + g) * 4 + 1]).collect()
    }

    /// Per vehicle with the contact-force log (jbeam-semantics-spec.md §32;
    /// blocking read): per node its last complete step's contact force and the
    /// largest since [`Self::clear_contact_force_peaks`] (N), and per break group
    /// its contact-break state (0 idle, 1 set off, 2 done); `None` for the others.
    #[allow(clippy::type_complexity)]
    pub fn contact_forces(&self) -> Vec<Option<(Vec<f32>, Vec<f32>, Vec<u8>)>> {
        let raw = self.read(&self.contact.acc);
        self.layout.contact_forces(bytemuck::cast_slice(&raw))
    }

    /// Restarts every node's contact-force peak (queued, before the next frame).
    pub fn clear_contact_force_peaks(&self) {
        let Some((at, n)) = self.layout.contact_force_peaks() else {
            return;
        };
        let acc = self.read(&self.contact.acc);
        let mut w: Vec<i32> = bytemuck::cast_slice(&acc).to_vec();
        for i in 0..n as usize {
            w[at as usize + 3 * i] = 0;
        }
        let lo = at as usize;
        let hi = lo + 3 * n as usize;
        self.queue.write_buffer(&self.contact.acc, lo as u64 * 4, bytemuck::cast_slice(&w[lo..hi]));
    }

    /// Per vehicle, its triangles' contact throttles (cwait, cidle).
    pub fn contact_throttles(&self) -> Vec<Vec<[i32; 2]>> {
        let raw = self.read(&self.misc);
        self.layout.contact_throttles(bytemuck::cast_slice(&raw))
    }

    /// Per vehicle, its triangles' self-collision throttles (wait, idle).
    pub fn self_collision_throttles(&self) -> Vec<Vec<[i32; 2]>> {
        let raw = self.read(&self.misc);
        self.layout.self_collision_throttles(bytemuck::cast_slice(&raw))
    }

    /// Run frames as the wide passes (one thread per element over every vehicle,
    /// a dispatch per pass) or as one workgroup per vehicle. Wide is the default
    /// when a vehicle has more than [`plan::WIDE_BEAMS`] beams. Returns whether wide is on.
    /// The wide passes do not run vehicle contact: while it is active
    /// ([`Self::contact_active`]) frames run one workgroup per vehicle.
    pub fn is_wide(&self) -> bool {
        self.planner.is_wide()
    }

    /// Run frames as the wide passes or one workgroup per vehicle (see
    /// [`Self::is_wide`]); returns whether wide is on.
    pub fn set_wide(&mut self, on: bool) -> bool {
        self.planner.set_wide(on)
    }

    /// One wide frame with each dispatch in its own timed pass (a diagnostic: the
    /// passes add their own cost): GPU milliseconds per pass kind, summed over the
    /// frame, in [`EXPLICIT_BEAMS_WIDE_PASSES`] order. Empty unless wide is on and
    /// the device has timestamp queries.
    pub fn profile_wide_frame(&mut self, steps: u32) -> Vec<(&'static str, f64)> {
        if !self.is_wide()
            || !self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Vec::new();
        }
        self.write_params(steps);
        let cmds = self.planner.commands(steps, true);
        let count = 2 * cmds.len() as u32;
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("explicit-beams-wide-profile"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-wide-profile"),
            size: u64::from(count) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut kinds = Vec::new();
        for cmd in &cmds {
            if let Cmd::Dispatch { kernel: Kernel::Wide(kind), .. } = cmd {
                let k = 2 * kinds.len() as u32;
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("explicit_beams_wide_profile"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &set,
                        beginning_of_pass_write_index: Some(k),
                        end_of_pass_write_index: Some(k + 1),
                    }),
                });
                self.dispatch(&mut pass, cmd, None);
                kinds.push(*kind);
            } else {
                self.copy(&mut enc, cmd);
            }
        }
        let used = 2 * kinds.len() as u32;
        enc.resolve_query_set(&set, 0..used, &resolve, 0);
        self.queue.submit([enc.finish()]);
        let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
        let raw = self.read(&resolve);
        let ticks: &[u64] = bytemuck::cast_slice(&raw);
        let mut out: Vec<(&'static str, f64)> = EXPLICIT_BEAMS_WIDE_PASSES
            .iter()
            .map(|(n, _)| (*n, 0.0))
            .collect();
        for (i, &kind) in kinds.iter().enumerate() {
            out[kind].1 += ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period;
        }
        out
    }

    /// One frame with each dispatch in its own timed pass (a diagnostic: the passes
    /// add their own cost): GPU milliseconds per dispatch, labelled. Contact active:
    /// the plan passes (on planning frames), the solo frame, the contact groups;
    /// otherwise the one frame dispatch. Empty without timestamp queries.
    pub fn profile_frame(&mut self, steps: u32) -> Vec<(&'static str, f64)> {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            || self.layout.spans.is_empty()
        {
            return Vec::new();
        }
        self.check_contact_status();
        self.write_params(steps);
        let cmds = self.planner.commands(steps, false);
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("explicit-beams-profile"),
            ty: wgpu::QueryType::Timestamp,
            count: 12,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-profile"),
            size: 12 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut labels: Vec<&'static str> = Vec::new();
        for cmd in &cmds {
            let kernel = match cmd {
                Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => *kernel,
                Cmd::Copy { .. } => {
                    self.copy(&mut enc, cmd);
                    continue;
                }
            };
            let k = 2 * labels.len() as u32;
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(kernel.label()),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &set,
                    beginning_of_pass_write_index: Some(k),
                    end_of_pass_write_index: Some(k + 1),
                }),
            });
            self.dispatch(&mut pass, cmd, None);
            labels.push(kernel.label());
        }
        let n = 2 * labels.len() as u32;
        enc.resolve_query_set(&set, 0..n, &resolve, 0);
        self.queue.submit([enc.finish()]);
        let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
        let raw = self.read(&resolve);
        let ticks: &[u64] = bytemuck::cast_slice(&raw);
        labels
            .iter()
            .enumerate()
            .map(|(i, l)| (*l, ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period))
            .collect()
    }

    /// Record the GPU time of each of the next `frames` frames (read with
    /// [`Self::frame_times_ms`]). False when the device lacks timestamp queries.
    pub fn enable_frame_timing(&mut self, frames: u32) -> bool {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return false;
        }
        let count = 2 * frames.max(1);
        self.timing = Some(FrameTiming {
            set: self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("explicit-beams-frame-timing"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            resolve: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("explicit-beams-frame-timing"),
                size: u64::from(count) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            capacity: frames.max(1),
            used: 0,
        });
        true
    }

    /// GPU milliseconds of each timed frame since [`Self::enable_frame_timing`]
    /// (waits for them); timing restarts empty.
    pub fn frame_times_ms(&mut self) -> Vec<f64> {
        let Some(t) = &mut self.timing else {
            return Vec::new();
        };
        let used = std::mem::take(&mut t.used);
        if used == 0 {
            return Vec::new();
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.resolve_query_set(&t.set, 0..2 * used, &t.resolve, 0);
        self.queue.submit([enc.finish()]);
        let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
        let raw = self.read(&self.timing.as_ref().unwrap().resolve);
        let ticks: &[u64] = bytemuck::cast_slice(&raw[..used as usize * 16]);
        ticks
            .chunks(2)
            .map(|p| p[1].saturating_sub(p[0]) as f64 * period)
            .collect()
    }

    /// Resolve the timed frames since [`Self::enable_frame_timing`] into a new
    /// mappable buffer and restart timing empty, without waiting: the buffer
    /// holds a (begin, end) pair of ticks per frame, the count is the frames.
    /// Map it once the queue reaches it (browsers: `mapAsync`) and scale the
    /// ticks by [`Self::timestamp_period_ms`]. `None` with no timing or no frames.
    pub fn take_frame_times(&mut self) -> Option<(wgpu::Buffer, u32)> {
        let t = self.timing.as_mut()?;
        let used = std::mem::take(&mut t.used);
        if used == 0 {
            return None;
        }
        let size = u64::from(used) * 16;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-frame-times"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.resolve_query_set(&t.set, 0..2 * used, &t.resolve, 0);
        enc.copy_buffer_to_buffer(&t.resolve, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        Some((staging, used))
    }

    /// Milliseconds per timestamp tick.
    pub fn timestamp_period_ms(&self) -> f64 {
        f64::from(self.queue.get_timestamp_period()) * 1e-6
    }

    /// The node state buffer, 3 `vec4<f32>` per node in vehicle order: position
    /// relative to the vehicle's origin and mass, velocity and friction, force.
    /// Vehicle `v`'s nodes start at [`Self::node_offset`]`(v)`.
    pub fn nodes_buffer(&self) -> &wgpu::Buffer {
        &self.nodes
    }

    /// The beam state buffer: 2 `vec4<f32>` per beam in vehicle order, (rest
    /// length, compression and tension yield, gate) and (strength, last force,
    /// broken as 0 or 1, 0), then the kernel's scratch. Vehicle `v`'s beams start
    /// at [`Self::beam_offset`]`(v)`.
    pub fn beams_buffer(&self) -> &wgpu::Buffer {
        &self.beams
    }

    /// The per-vehicle buffer: vehicle `v`'s record (`vec4<u32>`) at index `v` holds
    /// its origin's bits (xyz) and the steps it has run (w); triangle and wide
    /// pass scratch follows.
    pub fn misc_buffer(&self) -> &wgpu::Buffer {
        &self.misc
    }

    /// The rigid coupling's layout, when built [`Self::with_coupling`].
    pub fn coupling(&self) -> Option<CouplingLayout> {
        self.layout.coupling
    }

    /// The accumulators `GPUBuffer` (gAcc; the coupling's sums, [`CouplingLayout`]).
    pub fn acc_buffer(&self) -> &wgpu::Buffer {
        &self.acc
    }

    /// Turn the rigid coupling on or off from the next frame (its header's `on` word).
    /// Panics without [`Self::with_coupling`]. On, the frames run on the per-vehicle
    /// workgroups (the wide passes do not run the coupling; the same steps, as
    /// [`Self::set_wide`] off).
    pub fn set_coupling_on(&mut self, on: bool) {
        assert!(self.layout.coupling.is_some(), "explicit beams: built without coupling room");
        if on {
            self.set_wide(false);
        }
        self.coupling_on = on;
        self.write_coupling_header();
    }

    /// Tests and measurement: the coupling's per-step shape cull (on by default; lossless,
    /// `explicit_beams_rigid.slang` `cplMay`) off, every node testing every proxy shape.
    #[doc(hidden)]
    pub fn set_coupling_cull(&mut self, on: bool) {
        self.coupling_cull_off = !on;
        self.write_coupling_header();
    }

    /// Measurement only: the raycast wheels' pass on or off (off, the frames skip it and
    /// results are meaningless: time only). No-op without wheels.
    #[doc(hidden)]
    pub fn set_drive_pass_for_timing(&mut self, on: bool) {
        if let Some(d) = self.layout.drive {
            self.queue.write_buffer(&self.misc, u64::from(d.header) * 16, bytemuck::bytes_of(&u32::from(on)));
        }
    }

    fn write_coupling_header(&self) {
        let word = u32::from(self.coupling_on) | if self.coupling_on && self.coupling_cull_off { 2 } else { 0 };
        self.queue
            .write_buffer(&self.misc, self.layout.coupling_on_offset(), bytemuck::bytes_of(&word));
    }

    /// The first node of vehicle `v` in [`Self::nodes_buffer`] (in nodes).
    pub fn node_offset(&self, v: usize) -> u32 {
        self.layout.spans[v].node_off
    }

    /// The first beam of vehicle `v` in [`Self::beams_buffer`] (in beams).
    pub fn beam_offset(&self, v: usize) -> u32 {
        self.layout.spans[v].beam_off
    }

    /// Spawn rest lengths of every beam, in buffer order.
    pub fn rest0(&self) -> &[f32] {
        &self.layout.rest0
    }

    /// Vehicles simulated.
    pub fn vehicle_count(&self) -> usize {
        self.layout.spans.len()
    }

    /// Submit one frame: the origin rebase, then `steps` fixed steps, for every
    /// vehicle. Does not wait ([`Self::wait`]).
    pub fn frame(&mut self, steps: u32) {
        self.submit(steps);
    }

    fn submit(&mut self, steps: u32) {
        if self.layout.spans.is_empty() {
            return;
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let read = self.encode(&mut enc, steps);
        self.queue.submit([enc.finish()]);
        if read {
            self.contact.map_status();
        }
    }

    /// [`Self::frame`] encoded into `enc` (a host's own submission, such as the World's
    /// step). Submit it, then call [`Self::frame_submitted`] with the result.
    pub fn frame_encode(&mut self, enc: &mut wgpu::CommandEncoder, steps: u32) -> bool {
        if self.layout.spans.is_empty() || steps == 0 {
            return false;
        }
        self.encode(enc, steps)
    }

    /// Diagnostic: a mark after each of the frame's dispatches, named by kernel (`None`:
    /// stop).
    pub fn set_marks(&mut self, marks: Option<std::sync::Arc<crate::gpu_marks::GpuMarks>>) {
        self.marks = marks;
    }

    /// After the submission holding a [`Self::frame_encode`]: its status read mapped.
    pub fn frame_submitted(&self, read: bool) {
        if read {
            self.contact.map_status();
        }
    }

    /// The frame's commands (the shared plan's), replayed: a compute pass per run of
    /// dispatches, a copy between them; whether they copy the status for a read.
    fn encode(&mut self, enc: &mut wgpu::CommandEncoder, steps: u32) -> bool {
        self.check_contact_status();
        self.write_params(steps);
        // Timestamps at the start of the frame's first pass and the end of its last.
        let slot = self.timing.as_mut().and_then(|t| {
            (t.used < t.capacity).then(|| {
                t.used += 1;
                2 * t.used - 2
            })
        });
        let alt = self.alt_pipes();
        let cmds = self.planner.commands(steps, alt.is_none());
        let set = self.timing.as_ref().map(|t| &t.set);
        let stamps = |begin: bool, end: bool| {
            set.zip(slot)
                .map(|(q, k)| wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: begin.then_some(k),
                    end_of_pass_write_index: end.then_some(k + 1),
                })
        };
        let runs: Vec<&[Cmd]> = cmds.split(|c| matches!(c, Cmd::Copy { .. })).collect();
        let copies: Vec<&Cmd> = cmds.iter().filter(|c| matches!(c, Cmd::Copy { .. })).collect();
        for (i, run) in runs.iter().enumerate() {
            if i > 0 {
                self.copy(enc, copies[i - 1]);
            }
            if let Some(m) = self.marks.clone() {
                for cmd in *run {
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("explicit_beams"),
                            timestamp_writes: None,
                        });
                        self.dispatch(&mut pass, cmd, alt);
                    }
                    m.mark(enc, kernel_name(cmd));
                }
                continue;
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams"),
                timestamp_writes: stamps(i == 0, i + 1 == runs.len()),
            });
            for cmd in *run {
                self.dispatch(&mut pass, cmd, alt);
            }
        }
        // The status every STATUS_EVERY frames (the counters and the loop-cap
        // flag keep an error), with or without contact.
        let read = self.planner.status_due();
        if read {
            self.contact.read_status(&self.device, enc);
        }
        read
    }

    /// One plan dispatch: its pipeline (the variant's, when one is set) and bind
    /// group.
    fn dispatch(&self, pass: &mut wgpu::ComputePass, cmd: &Cmd, alt: Option<&AltPipes>) {
        let c = &self.contact;
        let (kernel, groups) = match *cmd {
            Cmd::Dispatch { kernel, groups, .. } => (kernel, Some(groups)),
            Cmd::Indirect { kernel, .. } => (kernel, None),
            Cmd::Copy { .. } => unreachable!("copies end the pass"),
        };
        match kernel {
            Kernel::Frame => {
                let (p, g) = alt.map_or((&self.pipeline, &self.group), |a| (&a.frame, &a.group));
                pass.set_pipeline(p);
                pass.set_bind_group(0, g, &[]);
            }
            Kernel::Solo | Kernel::Contact => {
                let (solo, groups, g) = alt.map_or_else(
                    || {
                        let p = c.pipes.as_ref().expect("explicit beams: contact passes for one vehicle");
                        (&p.solo, &p.groups, &c.contact_group)
                    },
                    |a| (&a.solo, &a.groups, &a.group),
                );
                pass.set_pipeline(if kernel == Kernel::Solo { solo } else { groups });
                pass.set_bind_group(0, g, &[]);
            }
            Kernel::Vv(i) => {
                let (p, g) = alt.map_or_else(
                    || {
                        let p = c.pipes.as_ref().expect("explicit beams: contact passes for one vehicle");
                        (&p.plan[i], &c.plan_group)
                    },
                    |a| (&a.plan[i], &a.plan_group),
                );
                pass.set_bind_group(0, g, &[]);
                pass.set_pipeline(p);
            }
            Kernel::Wide(i) => {
                let w = self.wide.as_ref().expect("explicit beams: wide passes not built");
                let Cmd::Dispatch { param_copy, .. } = *cmd else { unreachable!() };
                pass.set_pipeline(&w.passes[i]);
                pass.set_bind_group(0, &w.group, &[param_copy * PARAM_STRIDE as u32]);
            }
        }
        match (*cmd, groups) {
            (Cmd::Indirect { offset, .. }, _) => pass.dispatch_workgroups_indirect(&c.args, offset),
            (_, Some(x)) => pass.dispatch_workgroups(x, 1, 1),
            _ => unreachable!(),
        }
    }

    /// One plan copy.
    fn copy(&self, enc: &mut wgpu::CommandEncoder, cmd: &Cmd) {
        let Cmd::Copy { src, src_offset, dst, dst_offset, size } = *cmd else {
            unreachable!()
        };
        let buffer = |b: u32| match b {
            BIND_TABLE => &self.wide.as_ref().expect("explicit beams: no wide table").table,
            b => &self.bufs[b as usize],
        };
        enc.copy_buffer_to_buffer(buffer(src), src_offset, buffer(dst), dst_offset, size);
    }

    fn alt_pipes(&self) -> Option<&AltPipes> {
        if self.variant == KernelVariant::Current {
            return None;
        }
        self.alts.iter().find(|(v, _)| *v == self.variant).map(|(_, p)| p)
    }

    /// Tests only: run frames on the shipped kernel or on the frozen baseline
    /// (`explicit_beams_baseline.slang`, its WGSL read from the build's output
    /// directory, so native only), with or without its race probe
    /// ([`Self::take_race_flag`]). The baseline never runs the wide passes.
    #[doc(hidden)]
    pub fn set_kernel_variant(&mut self, variant: KernelVariant) {
        self.variant = variant;
        if variant == KernelVariant::Current || self.alts.iter().any(|(v, _)| *v == variant) {
            return;
        }
        let probe = variant == KernelVariant::BaselineProbe;
        if probe && self.probe.is_none() {
            let words = 1 + 2 * self.layout.total[0] as u64;
            self.probe = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("explicit-beams-race-probe"),
                size: (words * 4).max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let det = matches!(variant, KernelVariant::BaselineDet | KernelVariant::CurrentDet);
        if det && self.det.is_none() {
            // The kernel's DET_CAP hits per vehicle.
            let words = self.layout.spans.len() as u64 * (1 + 2 * 8192) + 1;
            self.det = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("explicit-beams-det-hits"),
                size: words * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }
        let read = |name: &str| {
            let path = format!("{}/{name}.wgsl", crate::EXPLICIT_BEAMS_WGSL_DIR);
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
        };
        let (aero, big, lanes) = self.module_kind;
        let (prefix, suffix) = match variant {
            KernelVariant::Baseline => ("explicit_baseline_", ""),
            KernelVariant::BaselineProbe => ("explicit_baseline_", "_probe"),
            KernelVariant::BaselineDet => ("explicit_baseline_", "_det"),
            // RUBBLE_DET_AA=1 (harness control): the baseline in place of the
            // current kernel, to see what the comparison flags with equal kernels.
            KernelVariant::CurrentDet if std::env::var("RUBBLE_DET_AA").is_ok() => ("explicit_baseline_", "_det"),
            KernelVariant::CurrentDet => ("explicit_det_", ""),
            KernelVariant::Ablate => ("explicit_ablate_", ""),
            KernelVariant::Reference => ("explicit_ref_", ""),
            KernelVariant::Current => unreachable!(),
        };
        // The baseline picks its lanes as the old host did; the current kernel's
        // variants the current host's.
        let lane_tag = if prefix == "explicit_baseline_" {
            if big { "_wide" } else { "" }
        } else {
            if lanes == 512 { "_wide" } else { "" }
        };
        let tag = format!("{}{}{}", if aero { "_aero" } else { "" }, lane_tag, suffix);
        let device = &self.device;
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let layout_with = |extra: &[(u32, &wgpu::Buffer)]| {
            let mut entries = vec![
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, ro),
                entry(2, ro),
            ];
            entries.extend((3..9).map(|b| entry(b, rw)));
            entries.extend(extra.iter().map(|(b, _)| entry(*b, rw)));
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("explicit-beams-variant"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("explicit-beams-variant"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let mut bind: Vec<wgpu::BindGroupEntry> = self
                .bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 { params_window(b) } else { b.as_entire_binding() },
                })
                .collect();
            bind.extend(extra.iter().map(|(b, buf)| wgpu::BindGroupEntry {
                binding: *b,
                resource: buf.as_entire_binding(),
            }));
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("explicit-beams-variant"),
                layout: &layout,
                entries: &bind,
            });
            (pipeline_layout, group)
        };
        let make = |entry: &str, wgsl: &str, layout: &wgpu::PipelineLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(crate::uniformize_workgroup_lanes(wgsl).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let mut extra: Vec<(u32, &wgpu::Buffer)> = self.probe.iter().filter(|_| probe).map(|b| (11, b)).collect();
        extra.extend(self.det.iter().filter(|_| det).map(|b| (12, b)));
        let (layout, group) = layout_with(&extra);
        let (plan_layout, plan_group) = layout_with(&[(10, &self.contact.args)]);
        let plan = ["bounds", "pairs", "group"].map(|p| {
            let e = format!("explicit_beams_vv_{p}");
            if matches!(variant, KernelVariant::CurrentDet | KernelVariant::Ablate) && prefix != "explicit_baseline_" {
                let wgsl = EXPLICIT_BEAMS_VV_PASSES.iter().find(|(n, _)| *n == e).unwrap().1;
                make(&e, wgsl, &plan_layout)
            } else if variant == KernelVariant::Reference {
                make(&e, &read(&format!("explicit_ref_{e}")), &plan_layout)
            } else {
                make(&e, &read(&format!("explicit_baseline_{e}")), &plan_layout)
            }
        });
        let pipes = AltPipes {
            frame: make("explicit_beams_frame", &read(&format!("{prefix}frame{tag}")), &layout),
            solo: make("explicit_beams_frame", &read(&format!("{prefix}solo{tag}")), &layout),
            groups: make("explicit_beams_frame", &read(&format!("{prefix}contact{tag}")), &layout),
            plan,
            group,
            plan_group,
        };
        self.alts.push((variant, pipes));
    }

    /// Measurement only: run the plain frame with the phases in `mask` skipped (1 the
    /// grid scan, 2 every node, 4 node indices past the lane count, 8 the wing forces'
    /// application, 16 the whole aero phase and its barrier, 32 the beams, 64 one
    /// barrier per sync instead of two, 128 no barriers, 256 the triangles); 0 runs
    /// the shipped kernel. Results are meaningless with a mask: time only.
    #[doc(hidden)]
    pub fn set_ablation(&mut self, mask: u32) {
        self.planner.set_ablation(mask);
        self.set_kernel_variant(if mask == 0 { KernelVariant::Current } else { KernelVariant::Ablate });
    }

    /// Tests only: whether an `EB_DET_HITS` frame had more self-collision hits in a
    /// step than its list holds (the comparison is then void); clears it.
    #[doc(hidden)]
    pub fn take_det_overflow(&mut self) -> bool {
        let Some(d) = &self.det else { return false };
        let raw = self.read(d);
        let at = raw.len() - 4;
        let flag = u32::from_le_bytes(raw[at..].try_into().unwrap());
        flag != 0
    }

    /// Tests only: whether a frame run with [`KernelVariant::BaselineProbe`] since
    /// the last call met one of the kernel's known races (bit 0: a self-collision or
    /// vehicle-contact read raced with another hit's add; bit 2: the break
    /// exception's live-beam count raced with a break), or the probe gave up (bit
    /// 1); clears it.
    #[doc(hidden)]
    pub fn take_race_flag(&mut self) -> u32 {
        let Some(p) = &self.probe else { return 0 };
        let raw = self.read(p);
        let flag = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        self.queue.write_buffer(p, 0, &[0u8; 4]);
        flag
    }

    /// Tests only: a copy of every buffer a frame writes and of the host's contact
    /// plan state, to run the same frame again ([`Self::restore`]).
    #[doc(hidden)]
    pub fn snapshot(&self) -> GpuSnapshot {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let copies = self
            .snapshot_sources()
            .iter()
            .map(|b| {
                let c = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("explicit-beams-snapshot"),
                    size: b.size(),
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                enc.copy_buffer_to_buffer(b, 0, &c, 0, b.size());
                c
            })
            .collect();
        self.queue.submit([enc.finish()]);
        GpuSnapshot {
            copies,
            state: self.planner.state(),
        }
    }

    /// Tests only: put back a [`Self::snapshot`].
    #[doc(hidden)]
    pub fn restore(&mut self, snap: &GpuSnapshot) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for (b, c) in self.snapshot_sources().iter().zip(&snap.copies) {
            enc.copy_buffer_to_buffer(c, 0, b, 0, b.size());
        }
        self.queue.submit([enc.finish()]);
        self.planner.restore(snap.state);
    }

    fn snapshot_sources(&self) -> Vec<wgpu::Buffer> {
        let mut v: Vec<wgpu::Buffer> = self.bufs[3..9].to_vec();
        v.push(self.contact.args.clone());
        v
    }

    /// Tests only: the state a frame leaves that does not depend on scheduling,
    /// as named runs of words: every node record; the beam records, incidence
    /// forces, aero, slide and trigger state; the self-collision accumulators; the
    /// group slots; the per-vehicle and per-triangle records (the contact groups'
    /// numbering, which follows atomic order, masked). Scratch whose order follows
    /// the lanes (grid runs, tested lists, candidate lists) is left out.
    #[doc(hidden)]
    pub fn state_words(&self) -> Vec<(&'static str, Vec<u32>)> {
        let w = &self.planner.words;
        let words = |b: &wgpu::Buffer| -> Vec<u32> { bytemuck::cast_slice(&self.read(b)).to_vec() };
        let nodes = words(&self.bufs[3]);
        let beams = words(&self.bufs[4]);
        let acc = words(&self.bufs[5]);
        let groups = words(&self.bufs[6]);
        let mut misc = words(&self.bufs[7]);
        let next = words(&self.bufs[8]);
        let nv = self.layout.spans.len();
        let (grid_base, aero_state) = (w[9] as usize, w[27] as usize);
        // The aero, slide state and slide-force slots (the contributions after them
        // are scratch).
        let slide_end = (w[35] + w[38]) as usize;
        let (vv_veh, vv_groups) = (w[43] as usize, w[46] as usize);
        for v in 0..nv {
            misc[(vv_veh + v * VV_RECS + 4) * 4 + 2] = 0;
        }
        // The wide passes' per-frame rebase shifts (scratch).
        let wide_misc = w[16] as usize;
        misc[wide_misc * 4..(wide_misc + 1 + nv) * 4].fill(0);
        let mut beam_words = beams[..grid_base * 4].to_vec();
        // A beam's "broke in the last beam phase" mark (.w of its second record) is
        // read only within that step (the next beam phase clears it first).
        for j in 0..self.layout.total[1] as usize {
            beam_words[(2 * j + 1) * 4 + 3] = 0;
        }
        beam_words.extend_from_slice(&beams[aero_state * 4..slide_end * 4]);
        vec![
            ("nodes", nodes),
            ("beams", beam_words),
            ("acc", acc[..3 * self.layout.total[0] as usize].to_vec()),
            // Anti-clip's side memory (order free; jbeam-semantics-spec.md §36).
            ("clip", {
                let at = w[78] as usize;
                if at == 0 {
                    Vec::new()
                } else {
                    acc[at..at + 4 * self.layout.total[0] as usize].to_vec()
                }
            }),
            ("groups", groups),
            ("misc", misc[..vv_groups * 4].to_vec()),
            // The trigger inputs: from keyBase to the contact lists (vvMembers −
            // vehicles).
            ("keys", next[w[31] as usize..w[47] as usize - nv].to_vec()),
        ]
    }

    /// Take in a contact error a finished frame reported ([`Self::contact_status`]),
    /// without waiting for frames in flight; the sim keeps running
    /// ([`Self::take_contact_errors`]).
    fn check_contact_status(&mut self) {
        if let Some(e) = self.contact.poll_status(&self.device) {
            self.contact_error(e);
        }
        let pending = std::mem::take(&mut *self.contact_errors_seen.lock().unwrap());
        for e in pending {
            self.contact_error(e);
        }
    }

    /// A contact error: recorded for the host, the status counters cleared, and for a
    /// vehicle that left its frame bounds the margin doubled (at most
    /// [`MAX_AUTO_CONTACT_MARGIN`] m) and a new plan made, so later frames cover it.
    /// The frame that reported it may have missed a contact; the error says so.
    fn contact_error(&mut self, e: String) {
        self.queue
            .write_buffer(&self.contact.acc, self.contact.status_at + 16, &[0u8; 16]);
        if e.contains("left its frame bounds") {
            let m = (2.0 * self.planner.contact_margin()).clamp(0.5, MAX_AUTO_CONTACT_MARGIN);
            self.planner.set_contact_margin(m);
            self.planner.request_replan();
        }
        let mut errors = self.contact_errors.lock().unwrap();
        if errors.last() != Some(&e) {
            errors.push(e);
        }
    }

    /// The contact errors since the last call (empty: none). A running sim never
    /// panics on one; the host surfaces them.
    pub fn take_contact_errors(&mut self) -> Vec<String> {
        self.check_contact_status();
        std::mem::take(&mut *self.contact_errors.lock().unwrap())
    }

    /// The parameters for a frame of `steps` steps (the plan's copies, one per step).
    fn write_params(&mut self, steps: u32) {
        let bytes = self.planner.write_params(steps);
        self.queue.write_buffer(&self.params, 0, &bytes);
    }

    /// Submit `dispatches` dispatches of `steps` steps each in one compute pass (the
    /// same simulation as one frame of their product): measures what a dispatch per
    /// step costs. Does not wait.
    pub fn frame_split(&mut self, steps: u32, dispatches: u32) {
        if self.layout.spans.is_empty() {
            return;
        }
        let bytes = self.planner.split_params(steps);
        self.queue.write_buffer(&self.params, 0, &bytes);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams_split"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            for _ in 0..dispatches {
                pass.dispatch_workgroups(self.layout.spans.len() as u32, 1, 1);
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// Block until submitted frames have run. Fails (panics) when one reported a
    /// vehicle-contact error ([`Self::contact_status`]).
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(e) = self.contact.ready_error() {
            // Handled at the next frame ([`Self::take_contact_errors`]).
            self.contact_errors_seen.lock().unwrap().push(e);
        }
    }

    /// Grab a node (Rubble only, [`engine_softbody::explicit_beams::Grab`]): from
    /// the next frame on, every step adds `k·(target − x) − c·v` to node
    /// `grab.node` of `vehicle` after its node phase, as the CPU reference's node
    /// pass does; `None` releases it. One grab for the whole fleet; set it again
    /// each frame to move the target.
    pub fn set_grab(&mut self, grab: Option<(usize, engine_softbody::explicit_beams::Grab)>) {
        let rec = self.layout.grab_records(grab);
        self.queue
            .write_buffer(&self.misc, self.layout.grab_rec() * 16, bytemuck::cast_slice(&rec));
        // The contact plan's frame bounds assumed the forces without it.
        self.planner.request_replan();
    }

    /// Set vehicle `vehicle`'s driver input for its raycast wheels
    /// (`engine_softbody::explicit_drive`) from the next frame on; a vehicle without
    /// them (or a fleet without any) ignores it.
    pub fn set_drive_input(&mut self, vehicle: usize, input: engine_softbody::explicit_drive::DriveInput) {
        let Some(d) = self.layout.drive else {
            return;
        };
        if vehicle >= self.layout.vehicles() {
            return;
        }
        let words = plan::drive::input_words(input);
        self.queue
            .write_buffer(&self.misc, d.input_offset(vehicle), bytemuck::cast_slice(&words));
        // Drive forces change the speeds the contact plan's frame bounds assumed.
        self.planner.request_replan();
    }

    /// Per vehicle its raycast wheels' state read back into a copy of `drive`
    /// (`None` where the vehicle has none).
    pub fn drive_states(
        &self,
        vehicles: &[Option<engine_softbody::explicit_drive::Drive>],
    ) -> Vec<Option<engine_softbody::explicit_drive::Drive>> {
        let Some(l) = self.layout.drive else {
            return vec![None; vehicles.len()];
        };
        let misc = self.read_u4(&self.misc);
        vehicles
            .iter()
            .enumerate()
            .map(|(v, d)| {
                let mut d = d.clone()?;
                let at = l.vehicle(v) as usize;
                let words: Vec<u32> = misc[at..at + l.stride as usize].iter().flatten().copied().collect();
                plan::drive::decode(&words, &mut d);
                Some(d)
            })
            .collect()
    }

    /// The next frame re-plans vehicle contact (a host's external forces, such as the
    /// World's rigid coupling, change the speeds the plan's frame bounds assumed).
    pub fn request_contact_replan(&self) {
        self.planner.request_replan();
    }

    /// Set one node's velocity (vehicle index, node index within it).
    pub fn set_velocity(&self, vehicle: usize, node: u32, v: Vec3) {
        let offset = self.layout.velocity_offset(vehicle, node);
        self.queue
            .write_buffer(&self.nodes, offset, bytemuck::cast_slice(&v.to_array()));
        // The contact plan's frame bounds assumed the old speeds.
        self.planner.request_replan();
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let out = slice.get_mapped_range().expect("mapped").to_vec();
        staging.unmap();
        out
    }

    fn read_f4(&self, buffer: &wgpu::Buffer) -> Vec<[f32; 4]> {
        bytemuck::cast_slice(&self.read(buffer)).to_vec()
    }

    fn read_u4(&self, buffer: &wgpu::Buffer) -> Vec<[u32; 4]> {
        bytemuck::cast_slice(&self.read(buffer)).to_vec()
    }

    /// Per vehicle, the origins nodes are relative to.
    pub fn origins(&self) -> Vec<Vec3> {
        self.layout.origins(&self.read_u4(&self.misc))
    }

    /// Per vehicle, world node positions.
    pub fn positions(&self) -> Vec<Vec<Vec3>> {
        self.layout
            .positions(&self.read_f4(&self.nodes), &self.read_u4(&self.misc))
    }

    /// Per vehicle, node velocities.
    pub fn velocities(&self) -> Vec<Vec<Vec3>> {
        self.layout.velocities(&self.read_f4(&self.nodes))
    }

    /// Per vehicle, its wings' broken flags (file order).
    pub fn wings_broken(&self) -> Vec<Vec<bool>> {
        self.layout.wings_broken(&self.read_f4(&self.beams))
    }

    /// Per vehicle, the fuselage drag force on each node (the last step's D).
    pub fn fuse_forces(&self) -> Vec<Vec3> {
        self.layout.fuse_forces(&self.read_f4(&self.beams))
    }

    /// Per vehicle, each slide node's current segment (an index into its
    /// `segments`; `None`: no rail) and broken-away flag.
    pub fn slides(&self) -> Vec<Vec<(Option<u32>, bool)>> {
        self.layout.slides(&self.read_f4(&self.beams))
    }

    /// Per vehicle, the trigger inputs the next step's command pass reads (key
    /// 0…84, 1.0 held), as [`ExplicitBeams::key_input`]; zeros for a vehicle
    /// without triggers. `steps` is the steps every vehicle has run.
    pub fn key_inputs(&self, steps: u64) -> Vec<[f32; plan::KEYS]> {
        let raw = self.read(&self.next);
        self.layout.key_inputs(bytemuck::cast_slice(&raw), steps)
    }

    /// Per vehicle its triggered JBeam deform groups (0 or 1, in
    /// `ExplicitBeams::deform_names` order; jbeam-semantics-spec.md §19).
    pub fn deform_triggered(&self) -> Vec<Vec<u32>> {
        let raw = self.read(&self.bufs[6]);
        self.layout.deform_triggered(bytemuck::cast_slice(&raw))
    }

    /// Per vehicle, each node's anti-clip side memory (held, recorded; as
    /// [`ExplicitBeams::clip_sides`], jbeam-semantics-spec.md §36); empty for an
    /// upload without anti-clip.
    pub fn clip_sides(&self) -> Vec<Vec<[u32; 4]>> {
        let at = self.planner.words[78] as usize;
        let nv = self.layout.spans.len();
        if at == 0 {
            return vec![Vec::new(); nv];
        }
        let raw = self.read(&self.acc);
        let acc: &[u32] = bytemuck::cast_slice(&raw);
        (0..nv)
            .map(|v| {
                let sp = &self.layout.spans[v];
                (0..sp.nodes as usize)
                    .map(|i| {
                        let k = at + 4 * (sp.node_off as usize + i);
                        [acc[k], acc[k + 1], acc[k + 2], acc[k + 3]]
                    })
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, every beam's broken flag and rest length.
    pub fn beams(&self) -> Vec<Vec<GpuBeam>> {
        self.layout.beams(&self.read_f4(&self.beams))
    }
}

impl Wide {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &[&wgpu::Buffer; 9],
        vehicle_recs: &[[u32; 4]],
    ) -> Self {
        let table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-vehicle-table"),
            size: plan::TABLE_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&table, 0, bytemuck::cast_slice(vehicle_recs));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-wide"),
            entries: &wide_entries(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-wide"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let passes = plan::WIDE_PASSES
            .iter()
            .map(|&entry| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(crate::uniformize_workgroup_lanes(module_wgsl(entry)).into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams-wide"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: b,
                            offset: 0,
                            size: std::num::NonZeroU64::new(PARAM_STRIDE),
                        })
                    } else {
                        b.as_entire_binding()
                    },
                })
                .chain(std::iter::once(wgpu::BindGroupEntry {
                    binding: BIND_TABLE,
                    resource: table.as_entire_binding(),
                }))
                .collect::<Vec<_>>(),
        });
        Self { passes, group, table }
    }
}

/// Every shipped explicit-beams module as the host creates it (name, entry point,
/// WGSL after the uniform-load rewrite): the browser check (`explicit-wgsl-check.mjs`
/// via the `explicit_wgsl_dump` example) compiles each in Chrome.
#[doc(hidden)]
pub fn explicit_beams_shipped_wgsl() -> Vec<(String, &'static str, String)> {
    let u = crate::uniformize_workgroup_lanes;
    let mut out: Vec<(String, &'static str, String)> = vec![
        ("explicit_beams".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_WGSL)),
        ("explicit_beams_wide".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_WIDE_WGSL)),
        ("explicit_beams_aero".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_AERO_WGSL)),
        ("explicit_beams_aero_wide".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_AERO_WIDE_WGSL)),
        ("explicit_beams_solo".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_SOLO_WGSL)),
        ("explicit_beams_solo_wide".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_SOLO_WIDE_WGSL)),
        ("explicit_beams_solo_aero".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_SOLO_AERO_WGSL)),
        ("explicit_beams_solo_aero_wide".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_SOLO_AERO_WIDE_WGSL)),
        ("explicit_beams_contact".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_CONTACT_WGSL)),
        ("explicit_beams_contact_wide".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_CONTACT_WIDE_WGSL)),
        ("explicit_beams_contact_aero".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_CONTACT_AERO_WGSL)),
        ("explicit_beams_contact_aero_wide".into(), "explicit_beams_frame", u(EXPLICIT_BEAMS_CONTACT_AERO_WIDE_WGSL)),
    ];
    for (e, w) in EXPLICIT_BEAMS_VV_PASSES.iter().chain(EXPLICIT_BEAMS_WIDE_PASSES.iter()) {
        out.push((e.to_string(), e, u(w)));
    }
    out
}

/// What a renderer needs besides a trace of `copies` vehicles of one `.truck` file,
/// their nodes concatenated in order (as a trace of [`ExplicitBeamsGpu::positions`]
/// writes them): `{"vehicles", "nodes", "triangles", "tyres", "boxes"}` with every
/// `cab` triangle (collision or not) and each tyre's node rings, offset per copy, and
/// the static boxes given (centre, half extents). `extra` is appended inside the
/// object (`,"key":…`).
pub fn render_scene_json(
    text: &str,
    nodes_per_vehicle: usize,
    copies: usize,
    boxes: &[(Vec3, Vec3)],
    extra: &str,
) -> String {
    let desc = engine_softbody::truck::load_truck(text);
    let file = engine_softbody::truck::parse_truck(text);
    let mut tris = Vec::new();
    let mut tyres = Vec::new();
    for k in 0..copies {
        let off = k * nodes_per_vehicle;
        tris.extend(desc.triangles.iter().map(|t| {
            format!(
                "[{},{},{}]",
                off + t.nodes[0] as usize,
                off + t.nodes[1] as usize,
                off + t.nodes[2] as usize
            )
        }));
        // Tyre nodes follow the truck's nodes, two rings of `rays` per wheel.
        let mut base = desc.nodes.len();
        for w in &file.wheels {
            tyres.push(format!(
                "{{\"axle\":[{},{}],\"first\":{},\"rays\":{}}}",
                off + w.node1 as usize,
                off + w.node2 as usize,
                off + base,
                w.rays
            ));
            base += 2 * w.rays as usize;
        }
    }
    let boxes: Vec<String> = boxes
        .iter()
        .map(|(c, h)| format!("[{},{},{},{},{},{}]", c.x, c.y, c.z, h.x, h.y, h.z))
        .collect();
    format!(
        "{{\"vehicles\":{copies},\"nodes\":{},\"triangles\":[{}],\"tyres\":[{}],\"boxes\":[{}]{extra}}}",
        nodes_per_vehicle * copies,
        tris.join(","),
        tyres.join(","),
        boxes.join(",")
    )
}

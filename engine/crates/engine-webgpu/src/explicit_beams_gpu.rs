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

use crate::{
    EXPLICIT_BEAMS_AERO_WGSL, EXPLICIT_BEAMS_AERO_WIDE_WGSL, EXPLICIT_BEAMS_CONTACT_AERO_WGSL,
    EXPLICIT_BEAMS_CONTACT_AERO_WIDE_WGSL, EXPLICIT_BEAMS_CONTACT_WGSL,
    EXPLICIT_BEAMS_CONTACT_WIDE_WGSL, EXPLICIT_BEAMS_SOLO_AERO_WGSL, EXPLICIT_BEAMS_SOLO_AERO_WIDE_WGSL,
    EXPLICIT_BEAMS_SOLO_WGSL, EXPLICIT_BEAMS_SOLO_WIDE_WGSL, EXPLICIT_BEAMS_VV_PASSES, EXPLICIT_BEAMS_WGSL,
    EXPLICIT_BEAMS_WIDE_PASSES, EXPLICIT_BEAMS_WIDE_WGSL,
};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitKind, StaticShape};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// A beam's state as read back: broken, and its rest length (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBeam {
    pub broken: bool,
    pub rest: f32,
    /// Spawn rest length (m).
    pub rest0: f32,
}

impl GpuBeam {
    /// Yielded, as [`engine_softbody::explicit_beams::Beam::bent`].
    pub fn bent(&self) -> bool {
        !self.broken && (self.rest - self.rest0).abs() > 1e-3 * self.rest0.abs()
    }
}

#[derive(Clone, Copy, Debug)]
struct Span {
    node_off: u32,
    nodes: u32,
    beam_off: u32,
    beams: u32,
    /// First slide node and rail segment (global) and the slide node count.
    slide_off: u32,
    slides: u32,
    seg_off: u32,
    tris: u32,
}

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
    spans: Vec<Span>,
    rest0: Vec<f32>,
    words: [u32; PARAM_WORDS],
    /// gBeams record of the aero state and per vehicle its first wing and count.
    aero_state: u32,
    /// gBeams record of the slide node states; gNext word of the trigger inputs.
    slide_state: u32,
    key_base: u32,
    wing_spans: Vec<(u32, u32)>,
    /// The wide passes (one thread per element over every vehicle), when used.
    wide: Option<Wide>,
    total: [u32; 5],
    timing: Option<FrameTiming>,
    /// Vehicle–vehicle contact ([`Self::set_vehicle_contact`]).
    contact: Contact,
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
    /// The rigid coupling's regions ([`Self::with_coupling`]).
    coupling: Option<CouplingLayout>,
    acc: wgpu::Buffer,
    couple_rec: u32,
}

/// Room for rigid-body proxies per vehicle ([`ExplicitBeamsGpu::with_coupling`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CouplingCapacity {
    /// Proxy bodies, shapes (at most 255) and hull planes per vehicle.
    pub bodies: u32,
    pub shapes: u32,
    pub planes: u32,
}

/// Where the coupling's records live (`explicit_beams_rigid.slang`): gMisc records from
/// `misc_base`, `recs` per vehicle (a header, then 5 per body, 7 per shape, 1 per plane);
/// gAcc words from `acc_base`, `bodies + 6 shapes` per vehicle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CouplingLayout {
    pub cap: CouplingCapacity,
    pub vehicles: u32,
    pub misc_base: u32,
    pub acc_base: u32,
}

impl CouplingLayout {
    pub const BODY_RECS: u32 = 5;
    pub const SHAPE_RECS: u32 = 7;

    fn new(cap: CouplingCapacity, vehicles: u32, misc_base: u32, acc_base: u32) -> Self {
        assert!(cap.shapes <= 255 && cap.planes < 1 << 24, "explicit coupling: at most 255 shapes per vehicle");
        Self { cap, vehicles, misc_base, acc_base }
    }
    /// Records per vehicle.
    pub fn recs(&self) -> u32 {
        1 + self.cap.bodies * Self::BODY_RECS + self.cap.shapes * Self::SHAPE_RECS + self.cap.planes
    }
    pub fn misc_words(&self) -> u32 {
        self.recs() * self.vehicles
    }
    pub fn acc_per_vehicle(&self) -> u32 {
        self.cap.bodies + 6 * self.cap.shapes
    }
    pub fn acc_words(&self) -> u32 {
        self.acc_per_vehicle() * self.vehicles
    }
    /// gMisc record of vehicle v's header, body j, shape k, plane i.
    pub fn vehicle(&self, v: u32) -> u32 {
        self.misc_base + v * self.recs()
    }
    pub fn body(&self, v: u32, j: u32) -> u32 {
        self.vehicle(v) + 1 + j * Self::BODY_RECS
    }
    pub fn shape(&self, v: u32, k: u32) -> u32 {
        self.vehicle(v) + 1 + self.cap.bodies * Self::BODY_RECS + k * Self::SHAPE_RECS
    }
    pub fn plane(&self, v: u32, i: u32) -> u32 {
        self.vehicle(v) + 1 + self.cap.bodies * Self::BODY_RECS + self.cap.shapes * Self::SHAPE_RECS + i
    }
    pub fn acc(&self, v: u32) -> u32 {
        self.acc_base + v * self.acc_per_vehicle()
    }
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
    left: u32,
    fresh: bool,
    frames: u64,
    replan: bool,
}

/// Vehicle–vehicle contact's pipelines, buffers and state.
struct Contact {
    /// On (the default); it runs only with more than one vehicle.
    on: bool,
    /// The plan passes (bounds, pairs, group) and the contact groups' entry.
    plan: [wgpu::ComputePipeline; 3],
    groups: wgpu::ComputePipeline,
    /// The solo vehicles' frame (`EB_VV_SOLO`).
    solo: wgpu::ComputePipeline,
    plan_group: wgpu::BindGroup,
    contact_group: wgpu::BindGroup,
    /// Indirect arguments: the solo vehicles (word 0) and the groups (word 4).
    args: wgpu::Buffer,
    /// The status counters (gAcc at byte `status_at`), copied each frame into a
    /// staging buffer and checked once mapped, a frame or two later (no waiting).
    acc: wgpu::Buffer,
    status_at: u64,
    /// The loop-cap flag (gAcc's last word, byte offset): see `cap_explicit_loops`.
    cap_at: u64,
    status: std::sync::Mutex<StatusRing>,
    /// Frames submitted (every STATUS_EVERY-th reads the status).
    frames: u64,
    /// Steps the current plan still covers; a new plan is due when a frame needs
    /// more (or `replan` is set: velocities written by the host).
    left: u32,
    replan: std::sync::atomic::AtomicBool,
    /// This frame runs the plan passes.
    planning: bool,
    /// No plan has run since contact (re)started.
    fresh: bool,
    /// The contact throttles' first record in gMisc.
    tri_base: u32,
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
/// Fleets up to this size find their overlapping pairs in the grouping pass (one
/// dispatch fewer per frame).
const FUSED_PAIRS_MAX: usize = 2048;
/// Frames between status reads.
const STATUS_EVERY: u64 = 16;

/// The error a status read reports: the loop-cap flag (`w[4]`), then the contact
/// counters (outside, overflow, unconverged, too big).
fn status_error(w: &[i32]) -> Result<(), String> {
    crate::explicit_loop_cap_hit(w[4]).map_err(|e| format!("explicit beams kernel: {e}"))?;
    contact_error(&w[..4]).map_err(|e| format!("explicit beams vehicle contact: {e}"))
}

/// The contact error the status counters (outside, overflow, unconverged, too big)
/// report.
fn contact_error(w: &[i32]) -> Result<(), String> {
    if w[0] > 0 {
        return Err(format!(
            "vehicle {} left its frame bounds (a node moved past the travel the frame's \
             contact plan allowed, so a contact may have been missed); raise \
             ExplicitBeamsGpu::set_contact_margin",
            w[0] - 1
        ));
    }
    if w[1] != 0 {
        return Err("the overlapping-pair list overflowed".into());
    }
    if w[2] != 0 {
        return Err("the contact grouping did not converge".into());
    }
    if w[3] != 0 {
        return Err(
            "a contact group had more than 64 vehicles (they ran without contact that frame)"
                .into(),
        );
    }
    Ok(())
}

impl Contact {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        buffers: &[&wgpu::Buffer; 9],
        (contact_wgsl, solo_wgsl): (&str, &str),
        acc: &wgpu::Buffer,
        (status_at, cap_at): (u64, u64),
        tri_base: u32,
    ) -> Self {
        let args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-vv-args"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        // The plan passes bind the indirect arguments (binding 10); the contact
        // entry, dispatched from them, must not.
        let layout_with = |args: bool| {
            let mut entries = vec![
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, ro),
                entry(2, ro),
            ];
            entries.extend((3..9).map(|b| entry(b, rw)));
            if args {
                entries.push(entry(10, rw));
            }
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
                    resource: b.as_entire_binding(),
                })
                .collect();
            if args {
                bind.push(wgpu::BindGroupEntry {
                    binding: 10,
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
        let plan = EXPLICIT_BEAMS_VV_PASSES.map(|(entry, wgsl)| make(entry, wgsl, &plan_layout));
        let groups = make("explicit_beams_frame", contact_wgsl, &contact_layout);
        let solo = make("explicit_beams_frame", solo_wgsl, &contact_layout);
        Self {
            on: true,
            plan,
            groups,
            solo,
            plan_group,
            contact_group,
            args: args_buffer,
            acc: acc.clone(),
            status_at,
            cap_at,
            status: Default::default(),
            frames: 0,
            left: 0,
            replan: Default::default(),
            planning: false,
            fresh: true,
            tri_base,

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

/// The wide passes' pipelines and bind group ([`ExplicitBeamsGpu::set_wide`]).
struct Wide {
    passes: Vec<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    enabled: bool,
    /// The kernel's vehicle table (constant memory): static records, then per
    /// vehicle (origin, steps) copied from the misc buffer each frame.
    table: wgpu::Buffer,
}

/// Wide pass entry points, in [`EXPLICIT_BEAMS_WIDE_PASSES`] order.
const WIDE_BEGIN: usize = 0;
const WIDE_REBASE: usize = 1;
const WIDE_NODES: usize = 2;
const WIDE_BEAMS_PASS: usize = 3;
const WIDE_TRIS: usize = 4;
const WIDE_END: usize = 5;
const WIDE_CLOSE: usize = 6;
const WIDE_HITS: usize = 7;

/// Parameter copies for the wide passes: one per step of a frame (its index in the
/// step word) plus the frame's end, at the uniform offset alignment.
const PARAM_STRIDE: u64 = 256;
const PARAM_COPIES: u32 = 129;
/// Words of the kernel's Params.
const PARAM_WORDS: usize = 64;
/// Params word: the phases a measurement build skips (`EB_ABLATE`).
const W_ABLATE: usize = 63;
/// Params word: the steps the contact plan's frame bounds cover.
const W_VV_PLAN_STEPS: usize = 60;
/// Steps a contact plan covers at least (frames shorter than this share a plan):
/// fewer plan passes, frame bounds a little wider.
const VV_PLAN_STEPS: u32 = 34;
/// Params words of vehicle contact: on, the frame bounds margin.
const W_VV_ON: usize = 39;
const W_VV_MARGIN: usize = 57;
/// The kernel's VV_RECS, VV_SLOTS and the counters' words (VV_CTL_WORDS).
const VV_RECS: usize = 6;
const VV_SLOTS: usize = 8;
const VV_CTL_WORDS: u32 = 8;
/// Frame bounds margin past the travel of a vehicle's fastest node (m): see
/// [`ExplicitBeamsGpu::set_contact_margin`].
pub const DEFAULT_CONTACT_MARGIN: f32 = 1.0;
/// The kernel's KEYS (trigger inputs per vehicle and step parity), NONE and node
/// info flags shift of the slide slots.
const KEYS: usize = engine_softbody::explicit_triggers::KEYS;
const NONE: u32 = u32::MAX;
const NODE_SLOT_SHIFT: u32 = 8;
/// Per vehicle the kernel's self-collision hit list (HIT_CAP records of HIT_WORDS
/// words) and pending-beam list (PEND_CAP) in gNext.
const SELF_HIT_WORDS: usize = 256 * 24 + 64;
/// The kernel's WIDE_MAX_VEHICLES: more vehicles run one workgroup each.
const WIDE_MAX_VEHICLES: usize = 256;

/// Timestamps around each frame's pass ([`ExplicitBeamsGpu::enable_frame_timing`]).
struct FrameTiming {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    capacity: u32,
    used: u32,
}

impl ExplicitBeamsGpu {
    /// Upload `vehicles` (their current state: positions, velocities, forces, beam
    /// state, throttles, origin).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, vehicles: &[ExplicitBeams]) -> Self {
        Self::with_coupling(device, queue, vehicles, None)
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
        let nv = vehicles.len();
        // Beam records carry the vehicle index in 16 bits.
        assert!(nv <= 1 << 16, "explicit beams GPU: at most 65536 vehicles, got {nv}");
        let total_nodes: usize = vehicles.iter().map(|v| v.nodes.len()).sum();
        let total_beams: usize = vehicles.iter().map(|v| v.beams.len()).sum();
        let total_tris: usize = vehicles.iter().map(|v| v.triangles.len()).sum();
        let node_info_base = nv * 4;
        let beam_base = node_info_base + total_nodes;
        let tri_base = beam_base + 3 * total_beams;
        let box_base = tri_base + total_tris;

        let mut vehicle_recs: Vec<[u32; 4]> = Vec::with_capacity(nv * 4);
        let mut node_info: Vec<[u32; 4]> = Vec::with_capacity(total_nodes);
        let mut beam_recs: Vec<[u32; 4]> = Vec::with_capacity(total_beams * 3);
        let mut tri_recs: Vec<[u32; 4]> = Vec::with_capacity(total_tris);
        let mut box_recs: Vec<[u32; 4]> = Vec::new();
        let mut shape_recs: Vec<[u32; 4]> = Vec::new();
        let mut shape_patches: Vec<usize> = Vec::new();
        let mut lists: Vec<u32> = Vec::new();
        let mut nodes: Vec<[f32; 4]> = Vec::with_capacity(total_nodes * 3);
        let mut beams: Vec<[f32; 4]> = Vec::with_capacity(total_beams * 2);
        let mut groups: Vec<u32> = Vec::new();
        let mut misc: Vec<[u32; 4]> = Vec::with_capacity(nv + total_tris);
        let mut tri_misc: Vec<[u32; 4]> = Vec::with_capacity(total_tris);
        let mut spans = Vec::with_capacity(nv);
        let mut rest0 = Vec::with_capacity(total_beams);
        // JBeam damping filters (global) and the beam records to point at them.
        let mut filters: Vec<engine_softbody::explicit_beams::DampFilter> = Vec::new();
        let mut filter_patches: Vec<(usize, usize)> = Vec::new();
        let (mut node_off, mut beam_off, mut tri_off, mut box_off) = (0u32, 0u32, 0u32, 0u32);
        // shocks2 rates (EB_AERO modules), 2 records per shock, global indices.
        let mut shock_recs: Vec<[u32; 4]> = Vec::new();
        let bits = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), 0];
        // Per beam, the lists positions of its end a's and end b's incidence entries.
        let mut slots: Vec<[u32; 2]> = vec![[u32::MAX; 2]; total_beams];
        let mut cand_table: Vec<u32> = Vec::new();
        // Slide-force slots over every vehicle (1-based in the node info).
        let mut slot_count = 0u32;
        assert!(
            total_nodes < (1 << (32 - NODE_SLOT_SHIFT)),
            "explicit beams GPU: too many nodes for the slide slots"
        );
        let mut vv_static: Vec<[u32; 4]> = Vec::with_capacity(nv);
        // Per lists position (incidence entries) the entry's far end.
        let mut nbr: Vec<u32> = Vec::new();
        let mut vv_tri: Vec<[u32; 4]> = Vec::with_capacity(total_tris);
        // Per vehicle its slide slots (node → 1-based global slot) and slot range.
        let mut slot_maps: Vec<Vec<u32>> = Vec::with_capacity(nv);
        let mut slot_ranges: Vec<(u32, u32)> = Vec::with_capacity(nv);
        for (vi, v) in vehicles.iter().enumerate() {
            // The kernel packs a (triangle, candidate) pair into 16 bits each.
            assert!(
                v.triangles.len() < 1 << 16 && v.candidates.len() < 1 << 16,
                "explicit beams GPU: at most 65535 collision triangles and contacters per vehicle"
            );
            let group_count = v
                .beams
                .iter()
                .map(|b| b.group.unsigned_abs() + 1)
                .max()
                .filter(|&g| g > 1)
                .unwrap_or(0);
            let group_off = groups.len() as u32;
            groups.extend(std::iter::repeat_n(u32::MAX, group_count as usize));
            // Slide targets (slide nodes and their rails' beam ends): a slot each,
            // in the node info flags above NODE_SLOT_SHIFT (1-based).
            let mut slot_of = vec![0u32; v.nodes.len()];
            let targets = v
                .slides
                .iter()
                .map(|sl| sl.node)
                .chain(v.segments.iter().flat_map(|seg| {
                    let b = &v.beams[seg.beam as usize];
                    [b.a, b.b]
                }));
            let slot_first = slot_count;
            for t in targets {
                if slot_of[t as usize] == 0 {
                    slot_count += 1;
                    slot_of[t as usize] = slot_count;
                }
            }
            slot_ranges.push((slot_first, slot_count - slot_first));
            // Incidence lists, then the candidates.
            for (i, inc) in v.incidence.iter().enumerate() {
                let n = &v.nodes[i];
                let flags = n.flags | slot_of[i] << NODE_SLOT_SHIFT;
                node_info.push([flags, lists.len() as u32, inc.len() as u32, vi as u32]);
                for &j in inc {
                    let b = &v.beams[j as usize];
                    let is_b = (b.b == i as u32 && b.a != i as u32) as u32;
                    let slot = &mut slots[(beam_off + j) as usize];
                    // A beam from a node to itself has both entries in one list.
                    let end = if is_b == 1 || slot[0] != u32::MAX {
                        1
                    } else {
                        0
                    };
                    slot[end] = lists.len() as u32;
                    lists.push(((beam_off + j) << 1) | is_b);
                    // Vehicle contact's side vote: the entry's far end.
                    nbr.push(node_off + if is_b == 1 { b.a } else { b.b });
                }
            }
            let cand_off = lists.len() as u32;
            cand_table.extend(cand_off..cand_off + v.candidates.len() as u32);
            lists.extend(v.candidates.iter().map(|&c| node_off + c));
            nbr.extend(std::iter::repeat_n(0, v.candidates.len()));
            // Vehicle contact: the contact points (global nodes).
            vv_static.push([lists.len() as u32, v.contact_points.len() as u32, 0, 0]);
            lists.extend(v.contact_points.iter().map(|&c| node_off + c));
            nbr.extend(std::iter::repeat_n(0, v.contact_points.len()));
            vv_tri.extend((0..v.triangles.len()).map(|t| {
                let c = v.contact_throttle.get(t).copied().unwrap_or([0, 0]);
                [c[0] as u32, c[1] as u32, 0, 0]
            }));
            let flags = v.settings.fast_inv_sqrt as u32
                | (v.settings.drag as u32) << 1
                | (v.settings.turbulence as u32) << 2
                | (v.fusedrag.is_some() as u32) << 3
                | (!v.triggers.is_empty() as u32) << 4
                | (!v.slides.is_empty() as u32) << 5
                | (!v.shapes.is_empty() as u32) << 6;
            vehicle_recs.push([
                node_off,
                v.nodes.len() as u32,
                beam_off,
                v.beams.len() as u32,
            ]);
            vehicle_recs.push([
                tri_off,
                v.triangles.len() as u32,
                box_off,
                (v.boxes.len() + v.shapes.len()) as u32,
            ]);
            vehicle_recs.push([cand_off, v.candidates.len() as u32, group_off, group_count]);
            vehicle_recs.push([
                flags,
                v.ground.to_bits(),
                v.collision_range.to_bits(),
                grid_cell(v).to_bits(),
            ]);
            for (i, n) in v.nodes.iter().enumerate() {
                nodes.push([n.pos.x, n.pos.y, n.pos.z, n.mass]);
                nodes.push([n.vel.x, n.vel.y, n.vel.z, n.friction]);
                // .w: the node's JBeam self-collision group (0 none).
                let group = v.self_groups.get(i).copied().unwrap_or(0);
                nodes.push([n.force.x, n.force.y, n.force.z, group as f32]);
            }
            let shock_off = (shock_recs.len() / 2) as u32;
            for sh in &v.shocks2 {
                shock_recs.push([
                    sh.k_out.to_bits(),
                    sh.c_out.to_bits(),
                    sh.pk_in.to_bits(),
                    sh.pc_in.to_bits(),
                ]);
                shock_recs.push([
                    sh.pk_out.to_bits(),
                    sh.pc_out.to_bits(),
                    (sh.soft as u32 as f32).to_bits(),
                    0,
                ]);
            }
            for b in &v.beams {
                beam_recs.push([
                    node_off + b.a,
                    node_off + b.b,
                    b.kind.code() | b.ty.code() << 8 | (vi as u32) << 16,
                    b.group as u32,
                ]);
                beam_recs.push([
                    b.k.to_bits(),
                    b.c.to_bits(),
                    b.limit_k.to_bits(),
                    b.limit_c.to_bits(),
                ]);
                // A shocks2 beam has no plastic coefficient: its shock's index; a
                // trigger its keys and enabled flag.
                let third = if b.kind == ExplicitKind::Shock2 {
                    shock_off + b.shock
                } else if b.kind == ExplicitKind::Trigger {
                    let t = &v.triggers[b.shock as usize];
                    t.short_key.min(KEYS as u32 - 1)
                        | t.long_key.min(KEYS as u32 - 1) << 8
                        | (t.enabled as u32) << 16
                } else {
                    b.plastic.to_bits()
                };
                // JBeam-mode kinds: the damping filter's gBeams record + 1 (0 none),
                // patched once the filter region's place is known.
                let fourth = if b.kind == ExplicitKind::JbeamTorsion {
                    // The watched break groups (the kernel's jtorsionUpdate).
                    let w = v.torsion_watch[b.shock as usize - 1];
                    beam_recs.push([w[0], w[1], w[2], 0]);
                    beams.push([b.rest, b.tc, b.tt, b.gate]);
                    beams.push([b.strength, b.force, b.broken as u32 as f32, 0.0]);
                    rest0.push(b.rest0);
                    continue;
                } else if b.kind.is_jbeam() {
                    if b.shock > 0 {
                        filter_patches.push((beam_recs.len(), filters.len()));
                        filters.push(v.damp_filters[b.shock as usize - 1]);
                    }
                    0
                } else {
                    b.rest0.to_bits()
                };
                beam_recs.push([
                    b.short_bound.to_bits(),
                    b.long_bound.to_bits(),
                    third,
                    fourth,
                ]);
                beams.push([b.rest, b.tc, b.tt, b.gate]);
                beams.push([b.strength, b.force, b.broken as u32 as f32, 0.0]);
                rest0.push(b.rest0);
            }
            for t in &v.triangles {
                tri_recs.push([
                    node_off + t.nodes[0],
                    node_off + t.nodes[1],
                    node_off + t.nodes[2],
                    vi as u32,
                ]);
                tri_misc.push([t.wait as u32, t.idle as u32, 0, 0]);
            }
            // JBeam self-collision groups: per triangle in gMisc .z.
            let first_tri = tri_misc.len() - v.triangles.len();
            for t in 0..v.triangles.len() {
                tri_misc[first_tri + t][2] = v.triangle_self_group(t);
            }
            for bx in &v.boxes {
                box_recs.push(bits(bx.lo));
                box_recs.push(bits(bx.hi));
            }
            // Static shapes (the kernel's nodeUpdateShapes): a slot each after the
            // boxes, (lo, kind) and (hi, parameter record), the records placed after
            // every slot (patched below). An identity-rotation box is a box slot.
            for sh in &v.shapes {
                if let Some(bx) = sh.as_aabb() {
                    box_recs.push(bits(bx.lo));
                    box_recs.push(bits(bx.hi));
                    continue;
                }
                let bd = sh.bounds();
                let mut lo = bits(bd.lo);
                lo[3] = sh.kind();
                let mut hi = bits(bd.hi);
                hi[3] = shape_recs.len() as u32;
                shape_patches.push(box_recs.len() + 1);
                box_recs.push(lo);
                box_recs.push(hi);
                shape_recs.extend(shape_params(sh));
            }
            let mut m = bits(v.origin);
            m[3] = v.steps as u32;
            misc.push(m);
            slot_maps.push(slot_of);
            spans.push(Span {
                node_off,
                nodes: v.nodes.len() as u32,
                beam_off,
                beams: v.beams.len() as u32,
                slide_off: 0,
                slides: v.slides.len() as u32,
                seg_off: 0,
                tris: v.triangles.len() as u32,
            });
            node_off += v.nodes.len() as u32;
            beam_off += v.beams.len() as u32;
            tri_off += v.triangles.len() as u32;
            box_off += v.boxes.len() as u32;
        }
        // Shape parameter records after every box slot (offsets from the box base).
        let shape_at = box_recs.len() as u32;
        for at in shape_patches {
            box_recs[at][3] += shape_at;
        }
        box_recs.extend(shape_recs);
        misc.extend(tri_misc);
        // Wings, the fuselage drag, self-centering commands and shocks2 (the EB_AERO
        // modules).
        let aero = vehicles.iter().any(|v| {
            !v.wings.is_empty()
                || v.fusedrag.is_some()
                || !v.centering.is_empty()
                || !v.shocks2.is_empty()
                || !v.triggers.is_empty()
                || !v.commands.is_empty()
                || !v.slides.is_empty()
        });
        // Command records: with triggers every command line (keyed), else the
        // self-centering ones.
        let cmd_count = |v: &ExplicitBeams| {
            if v.commands.is_empty() {
                v.centering.len()
            } else {
                v.commands.len()
            }
        };
        let mut aero_recs: Vec<[u32; 4]> = Vec::new();
        let mut wing_recs: Vec<[u32; 4]> = Vec::new();
        let mut cmd_recs: Vec<[u32; 4]> = Vec::new();
        let mut table_recs_aero: Vec<[u32; 4]> = Vec::new();
        let mut wing_state: Vec<[f32; 4]> = Vec::new();
        let mut cmd_state: Vec<[f32; 4]> = Vec::new();
        let mut fuse_state: Vec<[f32; 4]> = Vec::new();
        let mut wing_spans = Vec::with_capacity(nv);
        let base_static = (box_base + box_recs.len()) as u32;
        let aero_base = base_static;
        let wing_base = aero_base + 3 * nv as u32;
        let total_wings: usize = vehicles.iter().map(|v| v.wings.len()).sum();
        let total_cmds: usize = vehicles.iter().map(cmd_count).sum();
        let cmd_base = wing_base + 4 * total_wings as u32;
        let table_base = cmd_base + 2 * total_cmds as u32;
        // Airfoil tables shared by name across vehicles.
        let mut tables: Vec<(String, u32)> = Vec::new();
        let (mut node_off, mut beam_off) = (0u32, 0u32);
        let mut slide_recs: Vec<[u32; 4]> = Vec::new();
        let mut seg_recs: Vec<[u32; 4]> = Vec::new();
        let mut slide_state: Vec<[f32; 4]> = Vec::new();
        for (vi, v) in vehicles.iter().enumerate() {
            let table_of = |t: u32, tables: &mut Vec<(String, u32)>, recs: &mut Vec<[u32; 4]>| {
                let a = &v.airfoils[t as usize];
                if let Some((_, at)) = tables.iter().find(|(n, _)| *n == a.name) {
                    return *at;
                }
                let at = table_base + recs.len() as u32;
                for i in 0..engine_softbody::explicit_aero::AIRFOIL_ENTRIES {
                    recs.push([a.cl[i].to_bits(), a.cd[i].to_bits(), a.cm[i].to_bits(), 0]);
                }
                tables.push((a.name.clone(), at));
                at
            };
            aero_recs.push([
                (wing_recs.len() / 4) as u32,
                v.wings.len() as u32,
                (cmd_recs.len() / 2) as u32,
                cmd_count(v) as u32,
            ]);
            wing_spans.push(((wing_recs.len() / 4) as u32, v.wings.len() as u32));
            let fd = v.fusedrag.as_ref();
            aero_recs.push([
                node_off + fd.map_or(0, |f| f.front),
                fd.map_or(0.0f32, |f| f.width).to_bits(),
                v.nodes.len() as u32,
                0,
            ]);
            let slide_off = (slide_recs.len() / 2) as u32;
            let seg_off = seg_recs.len() as u32;
            aero_recs.push([slide_off, v.slides.len() as u32, 0, 0]);
            spans[vi].slide_off = slide_off;
            spans[vi].seg_off = seg_off;
            let seg_global = |s: Option<u32>| s.map_or(NONE, |s| seg_off + s);
            for sl in &v.slides {
                slide_recs.push([node_off + sl.node, seg_global(sl.segment), 0, 0]);
                slide_recs.push([
                    sl.spring.to_bits(),
                    sl.break_force.to_bits(),
                    sl.tolerance.to_bits(),
                    0,
                ]);
                slide_state.push([
                    sl.segment.map_or(-1.0, |s| (seg_off + s) as f32),
                    sl.broken_away as u32 as f32,
                    0.0,
                    0.0,
                ]);
            }
            for seg in &v.segments {
                seg_recs.push([
                    beam_off + seg.beam,
                    seg_global(seg.prev),
                    seg_global(seg.next),
                    0,
                ]);
            }
            fuse_state.push([v.fuse_force.x, v.fuse_force.y, v.fuse_force.z, 0.0]);
            for w in &v.wings {
                let g = |s: usize| node_off + w.nodes[s];
                wing_recs.push([g(0), g(1), g(2), g(3)]);
                wing_recs.push([g(4), g(5), g(6), g(7)]);
                let table = table_of(w.airfoil, &mut tables, &mut table_recs_aero);
                let (side, span, area) = match w.induced {
                    None => (0, 0.0f32, 0.0f32),
                    Some(i) => (
                        if i.side == engine_softbody::explicit_aero::WingSide::Left { 1 } else { 2 },
                        i.span,
                        i.area,
                    ),
                };
                wing_recs.push([w.efficacy.to_bits(), w.break_ref.to_bits(), table, side]);
                wing_recs.push([span.to_bits(), area.to_bits(), 0, 0]);
                wing_state.push([w.broken as u32 as f32, 0.0, 0.0, 0.0]);
                wing_state.extend([[0.0f32; 4]; 3]);
            }
            if v.commands.is_empty() {
                for c in &v.centering {
                    cmd_recs.push([
                        beam_off + c.beam,
                        c.contract_first as u32,
                        c.centre.to_bits(),
                        0,
                    ]);
                    cmd_recs.push([
                        c.shorten_rate.to_bits(),
                        c.lengthen_rate.to_bits(),
                        c.max_contraction.to_bits(),
                        c.max_extension.to_bits(),
                    ]);
                    cmd_state.push([c.mode as f32, 0.0, 0.0, 0.0]);
                }
            }
            for c in &v.commands {
                let keys = c.contract_key.min(KEYS as u32 - 1)
                    | c.extend_key.min(KEYS as u32 - 1) << 8
                    | (c.centering as u32) << 16
                    | 1 << 17;
                cmd_recs.push([
                    beam_off + c.beam,
                    c.contract_first as u32,
                    c.centre.to_bits(),
                    keys,
                ]);
                cmd_recs.push([
                    c.shorten_rate.to_bits(),
                    c.lengthen_rate.to_bits(),
                    c.max_contraction.to_bits(),
                    c.max_extension.to_bits(),
                ]);
                cmd_state.push([c.mode as f32, 0.0, 0.0, 0.0]);
            }
            node_off += v.nodes.len() as u32;
            beam_off += v.beams.len() as u32;
        }
        debug_assert!(slots.iter().flatten().all(|&s| s != u32::MAX));
        // Slide nodes in parallel (the kernel's slideNode): per slot every
        // contribution it can get in one step (its own slide nodes, and the ends of
        // every segment of the rails its slide nodes can reach), as list capacity.
        let mut slot_caps = vec![0u32; slot_count as usize];
        for (vi, v) in vehicles.iter().enumerate() {
            let slot_of = &slot_maps[vi];
            for sl in &v.slides {
                let mut seen = std::collections::BTreeSet::new();
                if slot_of[sl.node as usize] != 0 {
                    seen.insert((slot_of[sl.node as usize], 0));
                }
                let mut stack: Vec<u32> = sl.segment.into_iter().collect();
                let mut visited = std::collections::BTreeSet::new();
                while let Some(si) = stack.pop() {
                    if !visited.insert(si) {
                        continue;
                    }
                    let seg = &v.segments[si as usize];
                    let b = &v.beams[seg.beam as usize];
                    for (end, which) in [(b.a, 1), (b.b, 2)] {
                        if slot_of[end as usize] != 0 {
                            seen.insert((slot_of[end as usize], which));
                        }
                    }
                    stack.extend(seg.prev);
                    stack.extend(seg.next);
                }
                for (slot, _) in seen {
                    slot_caps[slot as usize - 1] += 1;
                }
            }
        }
        let mut slot_recs: Vec<[u32; 4]> = Vec::with_capacity(slot_caps.len());
        let mut slot_list_len = 0u32;
        for &c in &slot_caps {
            slot_recs.push([slot_list_len, c, 0, 0]);
            slot_list_len += c;
        }
        // Wing forces per node (the kernel's wingGather): per wing node its entries
        // (global wing << 2 | 0 induced, 1 front, 2 back) in wingApply's order.
        let mut wing_nodes: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut wing_node_ranges: Vec<(u32, u32)> = Vec::with_capacity(nv);
        {
            let (mut node_off, mut wing_off) = (0u32, 0u32);
            for v in vehicles {
                let first = wing_nodes.len();
                let mut at: std::collections::HashMap<u32, usize> = Default::default();
                for (wi, w) in v.wings.iter().enumerate() {
                    let gw = (wing_off + wi as u32) << 2;
                    let n = |slot: usize| node_off + w.nodes[slot];
                    let mut order: Vec<(u32, u32)> = Vec::new();
                    match w.induced.map(|i| i.side) {
                        Some(engine_softbody::explicit_aero::WingSide::Left) => order.extend([(n(6), 0), (n(4), 0)]),
                        Some(_) => order.extend([(n(7), 0), (n(5), 0)]),
                        None => {}
                    }
                    order.extend([(n(0), 1), (n(2), 1), (n(1), 1), (n(3), 1)]);
                    order.extend([(n(4), 2), (n(6), 2), (n(5), 2), (n(7), 2)]);
                    for (node, kind) in order {
                        let i = *at.entry(node).or_insert_with(|| {
                            wing_nodes.push((node, Vec::new()));
                            wing_nodes.len() - 1
                        });
                        wing_nodes[i].1.push(gw | kind);
                    }
                }
                wing_node_ranges.push(((first) as u32, (wing_nodes.len() - first) as u32));
                node_off += v.nodes.len() as u32;
                wing_off += v.wings.len() as u32;
            }
        }
        let veh_ext: Vec<[u32; 4]> = slot_ranges
            .iter()
            .zip(&wing_node_ranges)
            .map(|(&(a, n), &(wo, wn))| [a, n, wo, wn])
            .collect();
        let inc_base = beams.len() as u32;
        beams.extend(std::iter::repeat_n([0.0f32; 4], lists.len()));
        // The self-collision grid's sorted runs, at each vehicle's candidate offset.
        let grid_base = beams.len() as u32;
        beams.extend(std::iter::repeat_n([0.0f32; 4], lists.len()));
        let grid = vehicles.iter().any(|v| grid_cell(v) > 0.0) as u32;
        debug_assert_eq!(nbr.len(), lists.len());
        let slot_base = lists.len() as u32;
        lists.extend(slots.iter().flatten());
        let vv_nbr = lists.len() as u32;
        lists.extend(nbr);
        let cand_table_base = lists.len() as u32;
        let total_cands = cand_table.len() as u32;
        lists.extend(cand_table);
        // The wide passes' step counter and per-vehicle rebase shift.
        let wide_misc = misc.len() as u32;
        misc.extend(std::iter::repeat_n([0u32; 4], 1 + nv));
        // The node grab ([`ExplicitBeamsGpu::set_grab`]): 2 records right after them
        // (the kernel's grabRec()), off until set.
        misc.extend(std::iter::repeat_n([0u32; 4], 2));
        // The rigid coupling's header (the kernel's coupleRec()): (on, region, B, S |
        // planes << 8), (gAcc region, 0, 0, 0); the regions are placed below.
        let couple_rec = misc.len();
        misc.extend(std::iter::repeat_n([0u32; 4], 2));
        // Vehicle contact: per triangle its contact throttle; per vehicle its records;
        // per group its record.
        let vv_tri_base = misc.len() as u32;
        misc.extend(vv_tri);
        let vv_veh = misc.len() as u32;
        misc.extend(std::iter::repeat_n([0u32; 4], VV_RECS * nv));
        let vv_groups = misc.len() as u32;
        misc.extend(std::iter::repeat_n([0u32; 4], nv.max(1)));
        // gAcc: the accumulators, then the wide passes' grid counts and tested
        // counts (atomic).
        let grid_count_base = (total_nodes * 3) as u32;
        let tested_base = grid_count_base + (nv * WIDE_GRID_BUCKETS) as u32;
        let overflow_base = tested_base + 2 * nv as u32;
        let acc_len = overflow_base as usize + nv;
        // Vehicle contact: per vehicle 2 parities of box keys, the counters, labels
        // and group sums, the vehicle grid's bucket counts.
        let vv_acc = acc_len as u32;
        let vv_label = vv_acc + 12 * nv as u32 + VV_CTL_WORDS;
        let vv_grid_count = vv_label + 4 * nv as u32;
        let vv_buckets = (8 * nv).max(1024).next_power_of_two();
        // Then the slide slots' contribution counts.
        let slot_cnt_base = (vv_grid_count as usize + vv_buckets) as u32;
        let mut acc_init = vec![0i32; slot_cnt_base as usize + slot_count as usize];
        for v in 0..nv {
            for k in 0..12 {
                acc_init[vv_acc as usize + 12 * v + k] = if k % 6 < 3 { i32::MAX } else { i32::MIN };
            }
        }
        // gBeams: the wide grid's bucket slots, for vehicles on the wide passes.
        let slot_base16 = beams.len() as u32;
        if vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS) {
            beams.extend(std::iter::repeat_n(
                [0.0f32; 4],
                nv * WIDE_GRID_BUCKETS * WIDE_SLOTS,
            ));
        }
        // gBeams: the aero state (fuselage drag, wings, commands).
        let aero_state = beams.len() as u32;
        beams.extend(fuse_state);
        beams.extend(wing_state);
        beams.extend(cmd_state);
        // The slide node states, then the slide-force slots.
        let slide_state_base = beams.len() as u32;
        let slide_total = slide_state.len() as u32;
        beams.extend(slide_state);
        let slide_acc = beams.len() as u32;
        beams.extend(std::iter::repeat_n([0.0f32; 4], slot_count as usize));
        // The slide nodes' contributions (3 each; the kernel's slideContribBase).
        beams.extend(std::iter::repeat_n([0.0f32; 4], 3 * slide_total as usize));
        // Vehicle contact: the groups' contact points sorted by cell.
        let total_cp: usize = vehicles.iter().map(|v| v.contact_points.len()).sum();
        // (Four times: a small group's owners each have their own run, the
        // kernel's vvPassSmall.)
        let vv_run = beams.len() as u32;
        beams.extend(std::iter::repeat_n([0.0f32; 4], 4 * total_cp));
        // JBeam damping filters: (last rate, last output, a, b) each.
        let filter_base = beams.len();
        beams.extend(filters.iter().map(|f| [f.u, f.y, f.a, f.b]));
        for &(rec, i) in &filter_patches {
            beam_recs[rec][3] = (filter_base + i + 1) as u32;
        }
        // gNext: per list position the grid rank, then the wide passes' grid
        // offsets and tested lists (plain words: only the counters are atomic).
        let grid_off_base = lists.len() as u32;
        let tested_list_base = grid_off_base + (nv * (WIDE_GRID_BUCKETS + 1)) as u32;
        // Then per vehicle and step parity the trigger inputs: those the next
        // step's command pass reads (the last step's) at parity (steps & 1) ^ 1.
        let key_base = tested_list_base + total_tris as u32;
        let any_keys = vehicles.iter().any(|v| !v.triggers.is_empty());
        let next_len = key_base as usize + if any_keys { nv * 2 * KEYS } else { 0 };
        // Vehicle contact: the solo list, members (5 words per vehicle: lists,
        // sorted, run prefixes, partner and reset flags) and the tested triangles,
        // the tested triangles' marks (two words per triangle: the kernel's
        // vvMarkAt), the pairs, the vehicle grid's slots, the big list, the run
        // ranks.
        let vv_solo = next_len as u32;
        let vv_members = vv_solo + nv as u32;
        let vv_pairs = vv_members + 5 * nv as u32 + 3 * total_tris as u32;
        let vv_pair_cap = (16 * nv).max(1024) as u32;
        let vv_grid = vv_pairs + 2 * vv_pair_cap;
        let vv_big = vv_grid + (vv_buckets * VV_SLOTS) as u32;
        let vv_rank = vv_big + nv as u32;
        // Then the hit lists: per group two hits (two words each) per contact point.
        let vv_hits = vv_rank + total_cp as u32;
        // Then the slide slots' contribution lists.
        // (Hits: per group 8 × its points hits of two words: the kernel's vvHitAt,
        // and vvPassSmall's lists, twice the group's points for each of up to four
        // owners.)
        let slot_list_base = (vv_hits as usize + 16 * total_cp) as u32;
        // Then per vehicle its self-collision hit list and pending beams (the
        // kernel's hitList, pendList), and in gAcc two counters (hitAcc).
        let hit_base = slot_list_base as usize + slot_list_len as usize;
        let next_len = hit_base + nv * SELF_HIT_WORDS;
        let hit_acc = acc_init.len();
        acc_init.extend(std::iter::repeat_n(0, 2 * nv));
        // Last: the loop-cap flag (the WGSL post-process's `cap_explicit_loops`
        // writes gAcc's last word).
        let coupling = coupling.map(|cap| {
            let layout = CouplingLayout::new(cap, nv as u32, misc.len() as u32, acc_init.len() as u32);
            misc.extend(std::iter::repeat_n([0u32; 4], layout.misc_words() as usize));
            acc_init.extend(std::iter::repeat_n(0, layout.acc_words() as usize));
            misc[couple_rec] = [0, layout.misc_base, cap.bodies, cap.shapes | cap.planes << 8];
            misc[couple_rec + 1] = [layout.acc_base, 0, 0, 0];
            layout
        });
        let cap_at = acc_init.len();
        acc_init.push(0);
        for (k, rec) in vv_static.iter_mut().enumerate() {
            rec[2] = (hit_base + k * SELF_HIT_WORDS) as u32;
            rec[3] = (hit_acc + 2 * k) as u32;
        }
        let mut next_init = vec![0u32; next_len.max(1)];
        if any_keys {
            for (vi, v) in vehicles.iter().enumerate() {
                let parity = (v.steps as usize & 1) ^ 1;
                let at = key_base as usize + (vi * 2 + parity) * KEYS;
                for (k, &x) in v.key_input.iter().enumerate() {
                    next_init[at + k] = (x != 0.0) as u32;
                }
            }
        }
        let table_recs = vehicle_recs.clone();
        let mut statics = vehicle_recs;
        statics.extend(node_info);
        statics.extend(beam_recs);
        statics.extend(tri_recs);
        statics.extend(box_recs);
        debug_assert_eq!(statics.len() as u32, aero_base);
        statics.extend(aero_recs);
        statics.extend(wing_recs);
        statics.extend(cmd_recs);
        statics.extend(table_recs_aero);
        let shock2_base = statics.len() as u32;
        let shock2_count = (shock_recs.len() / 2) as u32;
        statics.extend(shock_recs);
        let slide_base = statics.len() as u32;
        statics.extend(slide_recs);
        let seg_base = statics.len() as u32;
        let seg_total = seg_recs.len() as u32;
        statics.extend(seg_recs);
        let vv_static_base = statics.len() as u32;
        statics.extend(vv_static);
        // Per vehicle its slide slots, then per slot its list (the kernel's
        // slotRecBase is veh_ext_base + vehicles).
        let veh_ext_base = statics.len() as u32;
        statics.extend(veh_ext);
        statics.extend(slot_recs);
        // Then the wing node records (the kernel's wingNodeBase), their entries at
        // the end of the lists.
        for (node, entries) in &wing_nodes {
            statics.push([*node, lists.len() as u32, entries.len() as u32, 0]);
            lists.extend(entries);
        }

        let big = vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS);
        let lanes = frame_lanes(vehicles);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit_beams"),
            // Large vehicles get twice the lanes (more loads in flight per step), a
            // small fleet 1024 (frame_lanes).
            source: wgpu::ShaderSource::Wgsl(
                crate::uniformize_workgroup_lanes(match (aero, lanes) {
                    (false, 512) => EXPLICIT_BEAMS_WIDE_WGSL,
                    (false, _) => EXPLICIT_BEAMS_WGSL,
                    (true, 512) => EXPLICIT_BEAMS_AERO_WIDE_WGSL,
                    (true, _) => EXPLICIT_BEAMS_AERO_WGSL,
                })
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("explicit_beams_frame"),
            layout: None,
            module: &module,
            entry_point: Some("explicit_beams_frame"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pad = |b: &[u8]| {
            if b.is_empty() {
                vec![0u8; 16]
            } else {
                b.to_vec()
            }
        };
        // Two parities of group slots.
        let group_total = groups.len() as u32;
        let groups: Vec<u32> = groups.iter().chain(groups.iter()).copied().collect();
        let init = |label: &str, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &pad(contents),
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let state = storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let words = [
            0,
            nv as u32,
            node_info_base as u32,
            beam_base as u32,
            tri_base as u32,
            box_base as u32,
            group_total,
            inc_base,
            slot_base,
            grid_base,
            grid,
            0,
            total_nodes as u32,
            total_beams as u32,
            total_tris as u32,
            total_cands,
            wide_misc,
            grid_count_base,
            grid_off_base,
            tested_base,
            tested_list_base,
            cand_table_base,
            slot_base16,
            overflow_base,
            aero_base,
            wing_base,
            cmd_base,
            aero_state,
            total_wings as u32,
            shock2_base,
            shock2_count,
            key_base,
            slide_base,
            seg_base,
            slide_state_base,
            slide_acc,
            slide_total,
            seg_total,
            slot_count,
            // Vehicle contact (vvOn is set per frame).
            0,
            veh_ext_base,
            vv_static_base,
            vv_tri_base,
            vv_veh,
            vv_acc,
            vv_label,
            vv_groups,
            vv_members,
            vv_pairs,
            vv_pair_cap,
            vv_grid,
            vv_grid_count,
            vv_buckets as u32 - 1,
            vv_cell(vehicles).to_bits(),
            vv_run,
            slot_cnt_base,
            slot_list_base,
            DEFAULT_CONTACT_MARGIN.to_bits(),
            vv_big,
            (nv <= FUSED_PAIRS_MAX) as u32,
            0,
            vv_nbr,
            vv_hits,
            0,
        ];
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-params"),
            size: PARAM_STRIDE * u64::from(PARAM_COPIES),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&words));
        let statics = init(
            "explicit-beams-static",
            bytemuck::cast_slice(&statics),
            storage,
        );
        let lists = init(
            "explicit-beams-lists",
            bytemuck::cast_slice(&lists),
            storage,
        );
        let nodes = init("explicit-beams-nodes", bytemuck::cast_slice(&nodes), state);
        let beams = init("explicit-beams-beams", bytemuck::cast_slice(&beams), state);
        let acc = init(
            "explicit-beams-acc",
            bytemuck::cast_slice(&acc_init),
            state,
        );
        let groups = init(
            "explicit-beams-groups",
            bytemuck::cast_slice(&groups),
            state,
        );
        let misc = init("explicit-beams-misc", bytemuck::cast_slice(&misc), state);
        let next = init(
            "explicit-beams-grid-next",
            bytemuck::cast_slice(&next_init),
            state,
        );
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
                resource: b.as_entire_binding(),
            }),
        });
        let buffers = [
            &params, &statics, &lists, &nodes, &beams, &acc, &groups, &misc, &next,
        ];
        // The wide passes do not run wings, the fuselage drag, commands or shocks2.
        let wide = (nv <= WIDE_MAX_VEHICLES && !aero)
            .then(|| Wide::new(device, queue, &buffers, &table_recs));
        let total = [
            total_nodes as u32,
            total_beams as u32,
            total_tris as u32,
            total_cands,
            group_total,
        ];
        // Every vehicle must have run the same steps for the wide passes (one
        // step parity for the group slots).
        let same_steps = vehicles.windows(2).all(|w| w[0].steps == w[1].steps);
        let wide = wide.map(|mut w| {
            w.enabled = same_steps && vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS);
            w
        });
        let contact = Contact::new(
            device,
            &buffers,
            match (aero, lanes) {
                (false, 512) => (EXPLICIT_BEAMS_CONTACT_WIDE_WGSL, EXPLICIT_BEAMS_SOLO_WIDE_WGSL),
                (false, _) => (EXPLICIT_BEAMS_CONTACT_WGSL, EXPLICIT_BEAMS_SOLO_WGSL),
                (true, 512) => (EXPLICIT_BEAMS_CONTACT_AERO_WIDE_WGSL, EXPLICIT_BEAMS_SOLO_AERO_WIDE_WGSL),
                (true, _) => (
                    EXPLICIT_BEAMS_CONTACT_AERO_WGSL,
                    EXPLICIT_BEAMS_SOLO_AERO_WGSL,
                ),
            },
            &acc,
            (u64::from(vv_acc + 12 * nv as u32) * 4, cap_at as u64 * 4),
            vv_tri_base,
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
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            group,
            params,
            nodes,
            beams,
            misc,
            spans,
            rest0,
            words,
            aero_state,
            slide_state: slide_state_base,
            key_base,
            next,
            wing_spans,
            timing: None,
            wide,
            total,
            contact,
            bufs,
            module_kind: (aero, big, lanes),
            variant: KernelVariant::Current,
            alts: Vec::new(),
            probe: None,
            det: None,
            coupling,
            acc,
            couple_rec: couple_rec as u32,
        }
    }

    /// Vehicle–vehicle contact between every vehicle of the fleet
    /// (`engine/docs/explicit-beams-vehicle-contact-spec.md`): on by default; it
    /// runs when the fleet has more than one vehicle. Off, every vehicle steps
    /// alone (the one-dispatch frame).
    pub fn set_vehicle_contact(&mut self, on: bool) {
        if on && !self.contact.on {
            self.contact.fresh = true;
        }
        self.contact.on = on;
        self.contact.left = 0;
    }

    /// Whether vehicle contact is on ([`Self::set_vehicle_contact`]).
    pub fn vehicle_contact(&self) -> bool {
        self.contact.on
    }

    /// Whether frames run the vehicle contact passes (on, more than one vehicle).
    pub fn contact_active(&self) -> bool {
        self.contact.on && self.spans.len() > 1
    }

    /// The frame bounds margin (m, default [`DEFAULT_CONTACT_MARGIN`]). A contact
    /// plan covers the next frames of at least 34 steps (every frame at 60 fps,
    /// every other at 120): a vehicle whose box, grown by twice the travel of its
    /// fastest node over those steps plus this margin, overlaps no other's is
    /// stepped alone; one that does is stepped with the vehicles it may meet, step
    /// by step. A node leaving those bounds (accelerated past them) could have met
    /// a vehicle unseen: the host then fails (panics) at a later frame or `wait`
    /// ([`Self::contact_status`] at once). A larger margin groups more vehicles
    /// (slower), never changes results.
    pub fn set_contact_margin(&mut self, metres: f32) {
        assert!(metres >= 0.0 && metres.is_finite(), "contact margin {metres}");
        self.words[W_VV_MARGIN] = metres.to_bits();
        self.contact.left = 0;
    }

    /// The contact status (blocking): `Err` when a vehicle left its frame bounds,
    /// the pair list overflowed or the grouping did not converge.
    pub fn contact_status(&self) -> Result<(), String> {
        let raw = self.read(&self.contact.acc);
        let w: &[i32] = bytemuck::cast_slice(&raw);
        let at = (self.contact.status_at / 4) as usize;
        let cap = (self.contact.cap_at / 4) as usize;
        crate::explicit_loop_cap_hit(w[cap])?;
        contact_error(&w[at + 4..at + 8])
    }

    /// The kernel status (blocking): `Err` when a kernel loop reached its cap
    /// ([`crate::EXPLICIT_LOOP_CAP`]: a runaway loop, stopped; that frame's results
    /// are invalid). Frames also check it on their own every few frames
    /// ([`Self::frame`] and [`Self::wait`] fail on it).
    pub fn kernel_status(&self) -> Result<(), String> {
        let raw = self.read(&self.contact.acc);
        let w: &[i32] = bytemuck::cast_slice(&raw);
        crate::explicit_loop_cap_hit(w[(self.contact.cap_at / 4) as usize])
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

    /// Per vehicle, its triangles' contact throttles (cwait, cidle).
    pub fn contact_throttles(&self) -> Vec<Vec<[i32; 2]>> {
        let raw = self.read(&self.misc);
        let m: &[[u32; 4]] = bytemuck::cast_slice(&raw);
        let mut out = Vec::with_capacity(self.spans.len());
        let mut t0 = self.contact.tri_base as usize;
        for &n in &self.tri_counts() {
            out.push(m[t0..t0 + n].iter().map(|r| [r[0] as i32, r[1] as i32]).collect());
            t0 += n;
        }
        out
    }

    /// Per vehicle, its triangles' self-collision throttles (wait, idle).
    pub fn self_collision_throttles(&self) -> Vec<Vec<[i32; 2]>> {
        let raw = self.read(&self.misc);
        let m: &[[u32; 4]] = bytemuck::cast_slice(&raw);
        let mut out = Vec::with_capacity(self.spans.len());
        let mut t0 = self.spans.len();
        for &n in &self.tri_counts() {
            out.push(m[t0..t0 + n].iter().map(|r| [r[0] as i32, r[1] as i32]).collect());
            t0 += n;
        }
        out
    }

    fn tri_counts(&self) -> Vec<usize> {
        self.spans.iter().map(|s| s.tris as usize).collect()
    }

    /// Run frames as the wide passes (one thread per element over every vehicle,
    /// a dispatch per pass) or as one workgroup per vehicle. Wide is the default
    /// when a vehicle has more than [`WIDE_BEAMS`] beams. Returns whether wide is on.
    /// The wide passes do not run vehicle contact: while it is active
    /// ([`Self::contact_active`]) frames run one workgroup per vehicle.
    pub fn is_wide(&self) -> bool {
        self.wide.as_ref().is_some_and(|w| w.enabled) && !self.contact_active()
    }

    /// Run frames as the wide passes or one workgroup per vehicle (see
    /// [`Self::is_wide`]); returns whether wide is on.
    pub fn set_wide(&mut self, on: bool) -> bool {
        if let Some(w) = &mut self.wide {
            w.enabled = on;
        }
        self.is_wide()
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
        let plan = self.wide_plan(steps);
        let w = self.wide.as_ref().unwrap();
        let count = 2 * plan.len() as u32;
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
        for (i, &(step, kind, x)) in plan.iter().enumerate() {
            if i == 2 {
                let nv = self.spans.len() as u64;
                enc.copy_buffer_to_buffer(
                    &self.misc,
                    0,
                    &w.table,
                    (WIDE_MAX_VEHICLES * 4 * 16) as u64,
                    nv * 16,
                );
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams_wide_profile"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &set,
                    beginning_of_pass_write_index: Some(2 * i as u32),
                    end_of_pass_write_index: Some(2 * i as u32 + 1),
                }),
            });
            pass.set_pipeline(&w.passes[kind]);
            pass.set_bind_group(0, &w.group, &[step * PARAM_STRIDE as u32]);
            pass.dispatch_workgroups(x, 1, 1);
        }
        enc.resolve_query_set(&set, 0..count, &resolve, 0);
        self.queue.submit([enc.finish()]);
        let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
        let raw = self.read(&resolve);
        let ticks: &[u64] = bytemuck::cast_slice(&raw);
        let mut out: Vec<(&'static str, f64)> = EXPLICIT_BEAMS_WIDE_PASSES
            .iter()
            .map(|(n, _)| (*n, 0.0))
            .collect();
        for (i, &(_, kind, _)) in plan.iter().enumerate() {
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
            || self.spans.is_empty()
        {
            return Vec::new();
        }
        self.check_contact_status();
        self.write_params(steps);
        let nv = self.spans.len() as u32;
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
        {
            let mut run = |label: &'static str,
                           f: &mut dyn FnMut(&mut wgpu::ComputePass)| {
                let k = 2 * labels.len() as u32;
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &set,
                        beginning_of_pass_write_index: Some(k),
                        end_of_pass_write_index: Some(k + 1),
                    }),
                });
                f(&mut pass);
                labels.push(label);
            };
            if self.contact_active() {
                let c = &self.contact;
                if c.planning {
                    run("vv_bounds", &mut |p| {
                        p.set_bind_group(0, &c.plan_group, &[]);
                        p.set_pipeline(&c.plan[0]);
                        p.dispatch_workgroups(nv, 1, 1);
                    });
                    if nv as usize > FUSED_PAIRS_MAX {
                        run("vv_pairs", &mut |p| {
                            p.set_bind_group(0, &c.plan_group, &[]);
                            p.set_pipeline(&c.plan[1]);
                            p.dispatch_workgroups(nv.div_ceil(256), 1, 1);
                        });
                    }
                    run("vv_group", &mut |p| {
                        p.set_bind_group(0, &c.plan_group, &[]);
                        p.set_pipeline(&c.plan[2]);
                        p.dispatch_workgroups(1, 1, 1);
                    });
                }
                run("solo frame", &mut |p| {
                    p.set_pipeline(&c.solo);
                    p.set_bind_group(0, &c.contact_group, &[]);
                    p.dispatch_workgroups_indirect(&c.args, 0);
                });
                run("frame with groups", &mut |p| {
                    p.set_pipeline(&c.groups);
                    p.set_bind_group(0, &c.contact_group, &[]);
                    p.dispatch_workgroups_indirect(&c.args, 16);
                });
            } else {
                let (pipeline, group) = (&self.pipeline, &self.group);
                run("frame", &mut |p| {
                    p.set_pipeline(pipeline);
                    p.set_bind_group(0, group, &[]);
                    p.dispatch_workgroups(nv, 1, 1);
                });
            }
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

    /// The first node of vehicle `v` in [`Self::nodes_buffer`] (in nodes).
    /// The rigid coupling's layout, when built [`Self::with_coupling`].
    pub fn coupling(&self) -> Option<CouplingLayout> {
        self.coupling
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
        assert!(self.coupling.is_some(), "explicit beams: built without coupling room");
        if on {
            self.set_wide(false);
        }
        self.queue
            .write_buffer(&self.misc, u64::from(self.couple_rec) * 16, bytemuck::bytes_of(&(on as u32)));
    }

    pub fn node_offset(&self, v: usize) -> u32 {
        self.spans[v].node_off
    }

    /// The first beam of vehicle `v` in [`Self::beams_buffer`] (in beams).
    pub fn beam_offset(&self, v: usize) -> u32 {
        self.spans[v].beam_off
    }

    /// Spawn rest lengths of every beam, in buffer order.
    pub fn rest0(&self) -> &[f32] {
        &self.rest0
    }

    /// Vehicles simulated.
    pub fn vehicle_count(&self) -> usize {
        self.spans.len()
    }

    /// Submit one frame: the origin rebase, then `steps` fixed steps, for every
    /// vehicle. Does not wait ([`Self::wait`]).
    pub fn frame(&mut self, steps: u32) {
        self.submit(steps);
    }

    fn submit(&mut self, steps: u32) {
        if self.spans.is_empty() {
            return;
        }
        self.check_contact_status();
        self.write_params(steps);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Timestamps at the start of the frame's first pass and the end of its last.
        let slot = self.timing.as_mut().and_then(|t| {
            (t.used < t.capacity).then(|| {
                t.used += 1;
                2 * t.used - 2
            })
        });
        let set = self.timing.as_ref().map(|t| &t.set);
        let stamps = |begin: bool, end: bool| {
            set.zip(slot)
                .map(|(q, k)| wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: begin.then_some(k),
                    end_of_pass_write_index: end.then_some(k + 1),
                })
        };
        let alt = self.alt_pipes();
        if self.contact_active() {
            let c = &self.contact;
            let (plan, plan_group, solo, groups, contact_group) = match alt {
                Some(a) => (&a.plan, &a.plan_group, &a.solo, &a.groups, &a.group),
                None => (&c.plan, &c.plan_group, &c.solo, &c.groups, &c.contact_group),
            };
            let nv = self.spans.len() as u32;
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams_contact"),
                timestamp_writes: stamps(true, true),
            });
            if c.planning {
                pass.set_bind_group(0, plan_group, &[]);
                pass.set_pipeline(&plan[0]);
                pass.dispatch_workgroups(nv, 1, 1);
                if nv as usize > FUSED_PAIRS_MAX {
                    pass.set_pipeline(&plan[1]);
                    pass.dispatch_workgroups(nv.div_ceil(256), 1, 1);
                }
                pass.set_pipeline(&plan[2]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            // The vehicles (grouped ones return at once) and, past them, the
            // contact groups, side by side.
            // The plan wrote one of the two non-empty: with no group the plain solo
            // frame, else the frame with the groups past the vehicles.
            pass.set_pipeline(solo);
            pass.set_bind_group(0, contact_group, &[]);
            pass.dispatch_workgroups_indirect(&c.args, 0);
            pass.set_pipeline(groups);
            pass.dispatch_workgroups_indirect(&c.args, 16);
        } else if let Some(w) = self.wide.as_ref().filter(|w| w.enabled && alt.is_none()) {
            let plan = self.wide_plan(steps);
            // Begin and rebase, then the vehicles' origins and steps into the
            // table, then the steps.
            let (head, rest) = plan.split_at(2);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("explicit_beams_wide_begin"),
                    timestamp_writes: stamps(true, false),
                });
                for &(k, kind, x) in head {
                    pass.set_pipeline(&w.passes[kind]);
                    pass.set_bind_group(0, &w.group, &[k * PARAM_STRIDE as u32]);
                    pass.dispatch_workgroups(x, 1, 1);
                }
            }
            let nv = self.spans.len() as u64;
            enc.copy_buffer_to_buffer(
                &self.misc,
                0,
                &w.table,
                (WIDE_MAX_VEHICLES * 4 * 16) as u64,
                nv * 16,
            );
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams_wide"),
                timestamp_writes: stamps(false, true),
            });
            for &(k, kind, x) in rest {
                pass.set_pipeline(&w.passes[kind]);
                pass.set_bind_group(0, &w.group, &[k * PARAM_STRIDE as u32]);
                pass.dispatch_workgroups(x, 1, 1);
            }
        } else {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams"),
                timestamp_writes: stamps(true, true),
            });
            let (pipeline, group) = alt.map_or((&self.pipeline, &self.group), |a| (&a.frame, &a.group));
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(self.spans.len() as u32, 1, 1);
        }
        // The status every STATUS_EVERY frames (the counters and the loop-cap
        // flag keep an error), with or without contact.
        let read = {
            self.contact.frames += 1;
            self.contact.frames % STATUS_EVERY == 1
        };
        if read {
            self.contact.read_status(&self.device, &mut enc);
        }
        self.queue.submit([enc.finish()]);
        if read {
            self.contact.map_status();
        }

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
            let words = 1 + 2 * self.total[0] as u64;
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
            let words = self.spans.len() as u64 * (1 + 2 * 8192) + 1;
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
                    resource: b.as_entire_binding(),
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
        self.words[W_ABLATE] = mask;
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
            left: self.contact.left,
            fresh: self.contact.fresh,
            frames: self.contact.frames,
            replan: self.contact.replan.load(std::sync::atomic::Ordering::Relaxed),
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
        self.contact.left = snap.left;
        self.contact.fresh = snap.fresh;
        self.contact.frames = snap.frames;
        self.contact
            .replan
            .store(snap.replan, std::sync::atomic::Ordering::Relaxed);
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
        let w = &self.words;
        let words = |b: &wgpu::Buffer| -> Vec<u32> { bytemuck::cast_slice(&self.read(b)).to_vec() };
        let nodes = words(&self.bufs[3]);
        let beams = words(&self.bufs[4]);
        let acc = words(&self.bufs[5]);
        let groups = words(&self.bufs[6]);
        let mut misc = words(&self.bufs[7]);
        let next = words(&self.bufs[8]);
        let nv = self.spans.len();
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
        for j in 0..self.total[1] as usize {
            beam_words[(2 * j + 1) * 4 + 3] = 0;
        }
        beam_words.extend_from_slice(&beams[aero_state * 4..slide_end * 4]);
        vec![
            ("nodes", nodes),
            ("beams", beam_words),
            ("acc", acc[..3 * self.total[0] as usize].to_vec()),
            ("groups", groups),
            ("misc", misc[..vv_groups * 4].to_vec()),
            // The trigger inputs: from keyBase to the contact lists (vvMembers −
            // vehicles).
            ("keys", next[w[31] as usize..w[47] as usize - nv].to_vec()),
        ]
    }

    /// Fail on a contact error a finished frame reported ([`Self::contact_status`]),
    /// without waiting for frames in flight.
    fn check_contact_status(&mut self) {
        if let Some(e) = self.contact.poll_status(&self.device) {
            panic!("{e}");
        }
    }

    /// The parameters for a frame of `steps` steps: copy k (at k × PARAM_STRIDE)
    /// carries step k of the frame (k = steps: the frame's end).
    fn write_params(&mut self, steps: u32) {
        assert!(
            steps < PARAM_COPIES,
            "at most {} steps per frame",
            PARAM_COPIES - 1
        );
        self.words[0] = steps;
        self.words[W_VV_ON] = self.contact_active() as u32;
        self.contact.planning = false;
        if self.contact_active() {
            let replan = self
                .contact
                .replan
                .swap(false, std::sync::atomic::Ordering::Relaxed);
            if replan || self.contact.left < steps {
                let budget = steps.max(VV_PLAN_STEPS);
                self.words[W_VV_PLAN_STEPS] = budget;
                self.contact.left = budget;
                self.contact.planning = true;
                // The first plan since contact (re)started has no earlier bounds to
                // check the positions against.
                if self.contact.fresh {
                    self.words[W_VV_ON] |= 4;
                    self.contact.fresh = false;
                }
            }
            self.contact.left = self.contact.left.saturating_sub(steps);
        }
        let mut bytes = vec![0u8; (PARAM_STRIDE * u64::from(steps + 1)) as usize];
        for k in 0..=steps {
            let mut w = self.words;
            w[11] = k;
            let at = (PARAM_STRIDE * u64::from(k)) as usize;
            bytes[at..at + 4 * PARAM_WORDS].copy_from_slice(bytemuck::cast_slice(&w));
        }
        self.queue.write_buffer(&self.params, 0, &bytes);
    }

    /// The wide passes of a frame: (parameter copy, pass, workgroups).
    fn wide_plan(&self, steps: u32) -> Vec<(u32, usize, u32)> {
        let groups = |n: u32| n.div_ceil(256).max(1);
        let nv = self.spans.len() as u32;
        let [nodes, beams, tris, cands, slots] = self.total;
        let buckets = nv * WIDE_GRID_BUCKETS as u32;
        let mut plan = vec![(0, WIDE_BEGIN, groups(nv)), (0, WIDE_REBASE, groups(nodes))];
        for k in 0..steps {
            plan.push((k, WIDE_NODES, groups(nodes + slots + buckets)));
            plan.push((k, WIDE_BEAMS_PASS, groups(beams + cands + tris)));
            plan.push((k, WIDE_TRIS, groups(tris * WIDE_SUBS)));
            // One workgroup per vehicle: deferred breaks and self-collision hits.
            plan.push((k, WIDE_HITS, nv));
        }
        plan.push((steps, WIDE_END, groups(tris + beams)));
        plan.push((steps, WIDE_CLOSE, groups(nv)));
        plan
    }

    /// Submit `dispatches` dispatches of `steps` steps each in one compute pass (the
    /// same simulation as one frame of their product): measures what a dispatch per
    /// step costs. Does not wait.
    pub fn frame_split(&mut self, steps: u32, dispatches: u32) {
        if self.spans.is_empty() {
            return;
        }
        self.words[0] = steps;
        self.words[W_VV_ON] = 0;
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&self.words));
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
                pass.dispatch_workgroups(self.spans.len() as u32, 1, 1);
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// Block until submitted frames have run. Fails (panics) when one reported a
    /// vehicle-contact error ([`Self::contact_status`]).
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(e) = self.contact.ready_error() {
            panic!("{e}");
        }
    }

    /// Grab a node (Rubble only, [`engine_softbody::explicit_beams::Grab`]): from
    /// the next frame on, every step adds `k·(target − x) − c·v` to node
    /// `grab.node` of `vehicle` after its node phase, as the CPU reference's node
    /// pass does; `None` releases it. One grab for the whole fleet; set it again
    /// each frame to move the target.
    pub fn set_grab(&mut self, grab: Option<(usize, engine_softbody::explicit_beams::Grab)>) {
        let rec = match grab {
            Some((vehicle, g)) if vehicle < self.spans.len() && g.node < self.spans[vehicle].nodes => [
                1,
                self.spans[vehicle].node_off + g.node,
                g.k.to_bits(),
                g.c.to_bits(),
                g.target.x.to_bits(),
                g.target.y.to_bits(),
                g.target.z.to_bits(),
                0,
            ],
            _ => [0u32; 8],
        };
        self.queue
            .write_buffer(&self.misc, self.grab_rec() * 16, bytemuck::cast_slice(&rec));
        // The contact plan's frame bounds assumed the forces without it.
        self.contact
            .replan
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// gMisc record of the grab (the kernel's grabRec()).
    fn grab_rec(&self) -> u64 {
        u64::from(self.words[16]) + 1 + self.spans.len() as u64
    }

    /// Set one node's velocity (vehicle index, node index within it).
    pub fn set_velocity(&self, vehicle: usize, node: u32, v: Vec3) {
        let s = self.spans[vehicle];
        let offset = ((s.node_off + node) as u64 * 3 + 1) * 16;
        self.queue
            .write_buffer(&self.nodes, offset, bytemuck::cast_slice(&v.to_array()));
        // The contact plan's frame bounds assumed the old speeds.
        self.contact
            .replan
            .store(true, std::sync::atomic::Ordering::Relaxed);
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

    fn node_field(&self, field: usize) -> Vec<Vec<Vec3>> {
        let raw = self.read(&self.nodes);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        self.spans
            .iter()
            .map(|s| {
                (0..s.nodes)
                    .map(|i| {
                        let r = v[(s.node_off + i) as usize * 3 + field];
                        Vec3::new(r[0], r[1], r[2])
                    })
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, the origins nodes are relative to.
    pub fn origins(&self) -> Vec<Vec3> {
        let raw = self.read(&self.misc);
        let m: &[[u32; 4]] = bytemuck::cast_slice(&raw);
        (0..self.spans.len())
            .map(|i| {
                Vec3::new(
                    f32::from_bits(m[i][0]),
                    f32::from_bits(m[i][1]),
                    f32::from_bits(m[i][2]),
                )
            })
            .collect()
    }

    /// Per vehicle, world node positions.
    pub fn positions(&self) -> Vec<Vec<Vec3>> {
        let origins = self.origins();
        let mut rel = self.node_field(0);
        for (o, nodes) in origins.iter().zip(&mut rel) {
            for p in nodes.iter_mut() {
                *p += *o;
            }
        }
        rel
    }

    /// Per vehicle, node velocities.
    pub fn velocities(&self) -> Vec<Vec<Vec3>> {
        self.node_field(1)
    }

    /// Per vehicle, its wings' broken flags (file order).
    pub fn wings_broken(&self) -> Vec<Vec<bool>> {
        let raw = self.read(&self.beams);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        let nv = self.spans.len() as u32;
        self.wing_spans
            .iter()
            .map(|&(first, count)| {
                (first..first + count)
                    .map(|gw| v[(self.aero_state + nv + 4 * gw) as usize][0] != 0.0)
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, the fuselage drag force on each node (the last step's D).
    pub fn fuse_forces(&self) -> Vec<Vec3> {
        let raw = self.read(&self.beams);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        (0..self.spans.len())
            .map(|i| {
                let r = v[self.aero_state as usize + i];
                Vec3::new(r[0], r[1], r[2])
            })
            .collect()
    }

    /// Per vehicle, each slide node's current segment (an index into its
    /// `segments`; `None`: no rail) and broken-away flag.
    pub fn slides(&self) -> Vec<Vec<(Option<u32>, bool)>> {
        let raw = self.read(&self.beams);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        self.spans
            .iter()
            .map(|s| {
                (0..s.slides)
                    .map(|k| {
                        let r = v[(self.slide_state + s.slide_off + k) as usize];
                        let seg = (r[0] >= 0.0).then(|| r[0] as u32 - s.seg_off);
                        (seg, r[1] != 0.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, the trigger inputs the next step's command pass reads (key
    /// 0…84, 1.0 held), as [`ExplicitBeams::key_input`]; zeros for a vehicle
    /// without triggers. `steps` is the steps every vehicle has run.
    pub fn key_inputs(&self, steps: u64) -> Vec<[f32; KEYS]> {
        let raw = self.read(&self.next);
        let w: &[u32] = bytemuck::cast_slice(&raw);
        let parity = (steps as usize & 1) ^ 1;
        (0..self.spans.len())
            .map(|vi| {
                let mut out = [0.0f32; KEYS];
                let at = self.key_base as usize + (vi * 2 + parity) * KEYS;
                if at + KEYS <= w.len() {
                    for (k, o) in out.iter_mut().enumerate() {
                        *o = (w[at + k] != 0) as u32 as f32;
                    }
                }
                out
            })
            .collect()
    }

    /// Per vehicle, every beam's broken flag and rest length.
    pub fn beams(&self) -> Vec<Vec<GpuBeam>> {
        let raw = self.read(&self.beams);
        let v: &[[f32; 4]] = bytemuck::cast_slice(&raw);
        self.spans
            .iter()
            .map(|s| {
                (0..s.beams)
                    .map(|j| {
                        let g = (s.beam_off + j) as usize;
                        GpuBeam {
                            broken: v[g * 2 + 1][2] != 0.0,
                            rest: v[g * 2][0],
                            rest0: self.rest0[g],
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Beams above which a vehicle runs on the 512-lane module, and wide passes are the
/// default.
const WIDE_BEAMS: usize = 2048;
/// The kernel's GRID_BUCKETS and GRID_SUBS.
const WIDE_GRID_BUCKETS: usize = 1024;
const WIDE_SUBS: u32 = 8;
/// The kernel's WIDE_SLOTS (grid bucket slots of the wide passes).
const WIDE_SLOTS: usize = 16;

impl Wide {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &[&wgpu::Buffer; 9],
        vehicle_recs: &[[u32; 4]],
    ) -> Self {
        let table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-vehicle-table"),
            size: (WIDE_MAX_VEHICLES * 5 * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let n = vehicle_recs.len().min(WIDE_MAX_VEHICLES * 4);
        queue.write_buffer(&table, 0, bytemuck::cast_slice(&vehicle_recs[..n]));
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                // The parameters: one copy per step, picked by the offset.
                has_dynamic_offset: binding == 0,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-wide"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, ro),
                entry(2, ro),
                entry(3, rw),
                entry(4, rw),
                entry(5, rw),
                entry(6, rw),
                entry(7, rw),
                entry(8, rw),
                entry(9, wgpu::BufferBindingType::Uniform),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-wide"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let passes = EXPLICIT_BEAMS_WIDE_PASSES
            .iter()
            .map(|(entry, wgsl)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(crate::uniformize_workgroup_lanes(wgsl).into()),
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
                    binding: 9,
                    resource: table.as_entire_binding(),
                }))
                .collect::<Vec<_>>(),
        });
        Self {
            passes,
            group,
            enabled: false,
            table,
        }
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

/// Fleets of at most this many vehicles run 512 lanes per vehicle (about one
/// workgroup per GPU core: a vehicle's step latency sets the frame, and more lanes
/// shorten it); larger fleets keep 256 (512 for large vehicles) for throughput.
/// Measured on an M3 Max, one vehicle, GPU µs per step (drive, 256 / 512 / 1024
/// lanes): Viper 23.0 / 20.3 / 24.6, E36 39.4 / 31.0 / 38.8, Mustang 44.6 / 40.4 /
/// 46.0 (1024 lanes is slower: wider barriers, a longer serial grid scan).
const SMALL_FLEET: usize = 32;

/// Lanes per vehicle workgroup for this fleet. `RUBBLE_EXPLICIT_LANES` (256 or
/// 512; measurement) overrides the choice.
fn frame_lanes(vehicles: &[ExplicitBeams]) -> u32 {
    let big = vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS);
    let default = if vehicles.len() <= SMALL_FLEET || big { 512 } else { 256 };
    match std::env::var("RUBBLE_EXPLICIT_LANES").ok().and_then(|v| v.parse().ok()) {
        Some(n @ (256 | 512)) => n,
        _ => default,
    }
}

/// The self-collision grid's cell size for a vehicle (0: all pairs): the
/// solver's own ([`ExplicitBeams::grid_cell`]), so CPU and GPU share it.
/// A static shape's parameter records (the kernel's `shapeNormal`): an oriented box
/// (centre, half.x), (axis x, half.y), (axis y, half.z), (axis z, 0); a cylinder
/// (centre, radius), (half height, 0, 0, 0); a capsule (a, radius), (b, 0); a
/// sphere (centre, radius), (0).
fn shape_params(sh: &StaticShape) -> Vec<[u32; 4]> {
    let rec = |v: Vec3, w: f32| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), w.to_bits()];
    match *sh {
        StaticShape::OrientedBox {
            centre, half, axes, ..
        } => vec![
            rec(centre, half.x),
            rec(axes[0], half.y),
            rec(axes[1], half.z),
            rec(axes[2], 0.0),
        ],
        StaticShape::Cylinder {
            centre,
            radius,
            half_height,
        } => vec![rec(centre, radius), rec(Vec3::new(half_height, 0.0, 0.0), 0.0)],
        StaticShape::Capsule { a, b, radius } => vec![rec(a, radius), rec(b, 0.0)],
        StaticShape::Sphere { centre, radius } => vec![rec(centre, radius), [0; 4]],
    }
}

fn grid_cell(v: &ExplicitBeams) -> f32 {
    v.grid_cell
}

/// The vehicle grid's cell size: the largest vehicle extent (at upload), plus room
/// for the frame bounds and the box margin.
fn vv_cell(vehicles: &[ExplicitBeams]) -> f32 {
    let extent = vehicles
        .iter()
        .map(|v| {
            let (lo, hi) = v.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)));
            (hi - lo).max_element()
        })
        .filter(|e| e.is_finite())
        .fold(0.0f32, f32::max);
    extent + 2.0 * (DEFAULT_CONTACT_MARGIN + 0.05) + 1.0
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

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
pub use engine_exec::explicit_plan::{
    CouplingCapacity, CouplingLayout, GpuBeam, COUPLING_FOLD_OFF, COUPLING_LIST_OFF, COUPLING_LIST_SKIN_CEILING,
    COUPLING_LIST_SKIN_NODES_ONLY,
};
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::Vec3;
use wgpu::util::DeviceExt;

pub struct ExplicitBeamsGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The plain frame's pipeline and bind group ([`Kernel::Frame`]), created at first
    /// use: a fleet of more than one vehicle runs the contact kernels while vehicle
    /// contact is on, so a browser host compiles this one after the first frame
    /// ([`crate::ComputeSpec::deferred`]).
    frame: std::sync::OnceLock<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    frame_module: &'static str,
    /// Surface contact split from the frame ([`Kernel::FrameSeg`], [`Kernel::Surf`]),
    /// created at first use; its frame module.
    split: std::sync::OnceLock<Split>,
    split_module: &'static str,
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
    coupling_cache_off: bool,
    /// Diagnostic: extra bits OR-ed into the coupling header word ([`Self::debug_coupling_bits`]).
    debug_coupling_bits: u32,
    /// The coupling's per-node shape ranges off ([`Self::set_coupling_node_ranges`]).
    coupling_ranges_off: bool,
    /// The coupling's candidate list off ([`Self::set_coupling_candidate_list`]).
    coupling_list_off: bool,
    /// The coupling's folded step off ([`Self::set_coupling_fold`]).
    coupling_fold_off: bool,
    /// Diagnostic: the large contact groups' pipeline from an edited module
    /// ([`Self::debug_set_contact_big_wgsl`]).
    debug_big: Option<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    /// Diagnostic: the solo and the contact groups' pipelines from edited modules.
    debug_solo: Option<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    debug_groups: Option<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    /// Per vehicle its own frame-bounds margin (m, 0: the fleet's), widened when it
    /// left its frame bounds ([`Self::vehicle_contact_margin`]).
    vehicle_margins: Vec<f32>,
    surface_pairs_off: bool,
    surface_culls_off: bool,
    /// The retired nodes' bits as last written ([`Self::set_retired`]; empty: none).
    retired_bits: Vec<u32>,
    /// The nodes' piece labels packed for the coupling (4 bits each), the vehicles with
    /// labels in use, and whether the coupling takes a window per piece.
    label_words: Vec<u32>,
    labelled: Vec<bool>,
    piece_windows: bool,
    /// Vehicle–vehicle contact ([`Self::set_vehicle_contact`]).
    contact: Contact,
    /// The split group path ([`Self::set_group_split`]): on, its pipelines (created at
    /// first use) and whether they zero workgroup memory (natively; browsers always do).
    group_split: bool,
    group_split_tests: bool,
    /// Vehicle contact's candidates in a grid ([`Self::set_vv_grid`]; the owners' state by
    /// vehicle needs it).
    vv_grid: bool,
    group_split_pipes: std::sync::OnceLock<GroupSplit>,
    group_split_zero: bool,
    /// Self-collision's listed pairs across workgroups ([`Self::set_self_split`]): on,
    /// its pipelines (created at first use), and this frame's steps when it runs it (0
    /// not).
    self_split: bool,
    self_split_pipes: std::sync::OnceLock<SelfSplit>,
    self_split_frame: std::cell::Cell<u32>,
    /// The wide passes with the reused pair list ([`Self::set_wide_list`]), and the steps
    /// of the frame being encoded when it runs so (0: not).
    wide_list: bool,
    /// With the list: the aux pass's work on the list and hits kernels' lanes
    /// ([`Self::set_wide_fused`]).
    wide_fused: bool,
    wide_list_frame: std::cell::Cell<u32>,
    /// Measurement ([`Self::profile_wide_frame`]): which of a wide beams command's
    /// dispatches `dispatch` issues (0 all; 1 its own; 2 the list kernel, 3 the walk,
    /// 4 the pairs).
    wide_part: std::cell::Cell<u32>,
    /// The last frame ran self-collision split with vehicle contact (on the split group
    /// path's dispatches: [`ExplicitBeamsGpu::dispatch_group_split`]).
    self_split_vv: std::cell::Cell<bool>,
    /// Measurement only (time; results meaningless): self split dispatches not encoded
    /// (bit 0 the walks, bit 1 the pairs; [`ExplicitBeamsGpu::set_self_split_skip`]).
    self_split_skip: u32,
    /// Measurement: under per-dispatch timing, each dispatch of a self split or split
    /// group frame its own timed pass (off: the frame's split dispatches one segment).
    split_dispatch_timing: bool,
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
    /// Vehicle sleep's host side: the last drive input per vehicle (a change wakes it),
    /// the grabbed vehicle (kept awake), the vehicles to wake before the next frame.
    sleep_inputs: Vec<Option<engine_softbody::explicit_drive::DriveInput>>,
    sleep_grab: Option<usize>,
    sleep_wake: std::sync::Mutex<Vec<usize>>,
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

/// The state buffers copied for a read that resolves later ([`ExplicitBeamsGpu::stage_state`]):
/// map [`Self::buffer`] (a browser host: `map_async`, awaited), then [`Self::ab_words`].
/// A bit-for-bit A/B of two paths from one state in a host that cannot block.
#[doc(hidden)]
pub struct StagedState {
    staging: wgpu::Buffer,
    /// Byte sizes of nodes, beams, acc, groups, misc, next, back to back.
    sizes: [u64; 6],
    layout: ExplicitLayout,
    words: Vec<u32>,
    vv: std::ops::Range<usize>,
}

impl StagedState {
    /// The MAP_READ copy (its copy already submitted).
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.staging
    }

    /// Once [`Self::buffer`] is mapped: [`ExplicitBeamsGpu::state_words`] with the
    /// vehicle-contact plan's records masked ([`ExplicitBeamsGpu::vv_record_words`],
    /// planning state: as `vehicle_lab --ab-switch`'s verdict), then
    /// [`ExplicitBeamsGpu::coupling_words`]. Unmaps the buffer.
    pub fn ab_words(&self) -> Vec<(&'static str, Vec<u32>)> {
        let all: Vec<u32> = {
            let data = self.staging.slice(..).get_mapped_range().expect("staged state: map it first");
            bytemuck::pod_collect_to_vec(&data[..])
        };
        self.staging.unmap();
        let mut parts = Vec::with_capacity(6);
        let mut at = 0usize;
        for s in self.sizes {
            let n = (s / 4) as usize;
            parts.push(all[at..at + n].to_vec());
            at += n;
        }
        let coupling = coupling_words_of(&self.layout, &parts[4], &parts[2]);
        let [nodes, beams, acc, groups, misc, next]: [Vec<u32>; 6] = parts.try_into().unwrap();
        let mut out = plan::state_words(&self.layout, &self.words, nodes, &beams, &acc, groups, misc, &next);
        for (name, v) in &mut out {
            if *name == "misc" {
                v[self.vv.clone()].fill(0);
            }
        }
        out.extend(coupling);
        out
    }
}

/// Two [`StagedState::ab_words`] compared: one line per run of consecutive differing words
/// (its name, the run, the first pair), at most `cap` lines; empty when bit for bit.
#[doc(hidden)]
pub fn state_differences(a: &[(&'static str, Vec<u32>)], b: &[(&'static str, Vec<u32>)], cap: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut more = 0usize;
    for ((name, x), (name_b, y)) in a.iter().zip(b) {
        if name != name_b || x.len() != y.len() {
            out.push(format!("{name}: {} words / {name_b}: {} words", x.len(), y.len()));
            continue;
        }
        let total = x.iter().zip(y).filter(|(p, q)| p != q).count();
        let mut i = 0;
        while i < x.len() {
            if x[i] == y[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < x.len() && x[i] != y[i] {
                i += 1;
            }
            if out.len() < cap {
                out.push(format!(
                    "{name} [{start}..{i}] of {} ({total} differ in all): {:#x} / {:#x} ({} / {})",
                    x.len(),
                    x[start],
                    y[start],
                    f32::from_bits(x[start]),
                    f32::from_bits(y[start])
                ));
            } else {
                more += 1;
            }
        }
    }
    if a.len() != b.len() {
        out.push(format!("{} runs of words / {}", a.len(), b.len()));
    }
    if more > 0 {
        out.push(format!("… {more} more runs"));
    }
    out
}

/// A [`GpuSnapshot`] copied for a read that resolves later ([`ExplicitBeamsGpu::stage_snapshot`]):
/// map [`Self::buffer`], then [`Self::bytes`] (as [`ExplicitBeamsGpu::snapshot_to_bytes`]).
#[doc(hidden)]
pub struct StagedSnapshot {
    staging: wgpu::Buffer,
    sizes: Vec<u64>,
    state: [u32; 5],
}

impl StagedSnapshot {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.staging
    }

    /// Once [`Self::buffer`] is mapped: the snapshot's bytes, the format of
    /// [`ExplicitBeamsGpu::snapshot_to_bytes`]. Unmaps the buffer.
    pub fn bytes(&self) -> Vec<u8> {
        let all: Vec<u32> = {
            let data = self.staging.slice(..).get_mapped_range().expect("staged snapshot: map it first");
            bytemuck::pod_collect_to_vec(&data[..])
        };
        self.staging.unmap();
        let mut out: Vec<u32> = self.state.to_vec();
        out.push(self.sizes.len() as u32);
        let mut at = 0usize;
        for s in &self.sizes {
            let n = (s / 4) as usize;
            out.push(n as u32);
            out.extend_from_slice(&all[at..at + n]);
            at += n;
        }
        bytemuck::cast_slice(&out).to_vec()
    }
}

/// [`ExplicitBeamsGpu::coupling_words`] from the misc and acc words.
fn coupling_words_of(layout: &ExplicitLayout, misc: &[u32], acc: &[u32]) -> Vec<(&'static str, Vec<u32>)> {
    let Some(c) = layout.coupling else {
        return Vec::new();
    };
    let m0 = c.misc_base as usize * 4;
    let a0 = c.acc_base as usize;
    vec![
        ("coupling records", misc[m0..m0 + c.misc_words() as usize * 4].to_vec()),
        ("coupling sums", acc[a0..a0 + c.acc_words() as usize].to_vec()),
    ]
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
    /// Vehicle sleep's awake list ([`plan::SLEEP_LIST`]).
    sleep: wgpu::ComputePipeline,
    groups: wgpu::ComputePipeline,
    /// The solo vehicles' frame (`EB_VV_SOLO`).
    solo: wgpu::ComputePipeline,
    /// The groups of more than four members ([`Kernel::ContactBig`]); `None` for a
    /// fleet of at most [`plan::VV_PART`] vehicles, which cannot form one (the
    /// plan then never gives that dispatch a workgroup): its module is not compiled.
    big: Option<wgpu::ComputePipeline>,
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

/// The split frame's bind group layout entries ([`Kernel::FrameSeg`], [`Kernel::Surf`]):
/// the wide passes' without their vehicle table (binding 0: the segment's parameter
/// copy, by the dynamic offset).
fn split_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut e = wide_entries();
    e.retain(|x| x.binding != BIND_TABLE);
    e
}

/// Whether hosts run surface contact split from the frame (on unless
/// `RUBBLE_SURFACE_SPLIT=0`, a measurement: the frame with the pass in it).
pub fn surface_split_default() -> bool {
    std::env::var("RUBBLE_SURFACE_SPLIT").as_deref() != Ok("0")
}

/// Whether the coupling's per-node shape ranges start off
/// ([`ExplicitBeamsGpu::set_coupling_node_ranges`]): in browsers' WebGPU (wasm32), where
/// Chrome's compile of the range lookup for Metal (Tint's bounds-checked MSL) runs the
/// coupled frame hundreds of times slower (the crash town's car against a crate:
/// 755 ms a 1/60 s frame with them, 0.87 ms without; `engine/docs/explicit-vehicles-in-world.md`).
/// `RUBBLE_COUPLING_RANGES=0|1` overrides it natively.

pub fn coupling_node_ranges_off_default() -> bool {
    match std::env::var("RUBBLE_COUPLING_RANGES").as_deref() {
        Ok("0") => true,
        Ok("1") => false,
        _ => cfg!(target_arch = "wasm32"),
    }
}

/// The split frame's pipelines and bind group ([`FramePlanner::set_surface_split`]).
struct Split {
    frame: wgpu::ComputePipeline,
    surf: Vec<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
}

impl Split {
    fn new(device: &wgpu::Device, bufs: &[wgpu::Buffer; 9], frame_module: &str) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-split"),
            entries: &split_entries(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-split"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = |module_name: &str, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(module_name),
                source: wgpu::ShaderSource::Wgsl(crate::uniformize_workgroup_lanes(module_wgsl(module_name)).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(module_name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let frame = pipe(frame_module, plan::FRAME_ENTRY);
        let surf = plan::SURF_PASSES.iter().map(|&(m, e)| pipe(m, e)).collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams-split"),
            layout: &layout,
            entries: &std::array::from_fn::<_, 9, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: if i == 0 { params_window(&bufs[0]) } else { bufs[i].as_entire_binding() },
            }),
        });
        Self { frame, surf, group }
    }
}

/// Whether contact groups run split across workgroups ([`ExplicitBeamsGpu::set_group_split`]):
/// `RUBBLE_GROUP_SPLIT=1|0` (default off until its suites and fleet runs are clean:
/// contact-groups-design.md §6).
pub fn group_split_default() -> bool {
    std::env::var("RUBBLE_GROUP_SPLIT").as_deref() == Ok("1")
}

/// Byte offsets in the indirect arguments of the split group path's dispatches (the
/// kernel's vvArgs): the groups, and the vehicles when there is a group.
const SPLIT_GROUP_ARGS: u64 = 64;
const SPLIT_MEMBER_ARGS: u64 = 80;
const SPLIT_TESTS_ARGS: u64 = 96;
/// Self-collision split with vehicle contact: every vehicle's step in every frame
/// (the kernel's vvArgs words 28..30).
const SPLIT_ALL_ARGS: u64 = 112;
/// The kernel's step-word phases (`explicit_beams_split.slang`): the triangle tests in
/// their own dispatches, the levels (and the tests' pairs) after them.
const SPLIT_TESTS: u32 = 0x2000_0000;
const SPLIT_LEVELS: u32 = 0x4000_0000;
/// The split group path's entry points (`explicit_beams_split.slang`).
pub const SPLIT_MEMBER_ENTRY: &str = "explicit_beams_split_member";
pub const SPLIT_GROUP_ENTRY: &str = "explicit_beams_split_group";
pub const SPLIT_TESTS_ENTRY: &str = "explicit_beams_split_tests";

/// Whether the split path runs the small groups' triangle tests across workgroups
/// ([`ExplicitBeamsGpu::set_group_split_tests`]): `RUBBLE_GROUP_SPLIT_TESTS=1|0` (default on).
pub fn group_split_tests_default() -> bool {
    std::env::var("RUBBLE_GROUP_SPLIT_TESTS").as_deref() != Ok("0")
}
/// The kernel's SPLIT_BEGIN: the parameter copy past the frame's end marks the groups'
/// frame start.
const SPLIT_BEGIN: u32 = u32::MAX;

/// The split group path's module `which` (member, group, tests) for a fleet's variant.
/// The self-collision split's modules a frame needs (selfcol-design.md), at the frame's
/// variant: without vehicle contact the step, walk and pairs; with it the vehicles' step,
/// the walk and pairs that skip members, and the split group path's group and tests.
fn self_split_modules(aero: bool, lanes: u32, vv: bool) -> Vec<(String, &'static str, bool)> {
    let tag = match (aero, lanes) {
        (false, 512) => "_wide",
        (false, _) => "",
        (true, 512) => "_aero_wide",
        (true, _) => "_aero",
    };
    // (name, entry, binds the arguments buffer at binding 10)
    if vv {
        vec![
            (format!("explicit_beams_selfsplit_member_vv{tag}"), SPLIT_MEMBER_ENTRY, true),
            (format!("explicit_beams_selfsplit_walk_vv{tag}"), "explicit_beams_self_walk", false),
            (format!("explicit_beams_selfsplit_pairs_vv{tag}"), "explicit_beams_self_pairs", false),
            (split_module("group", aero, lanes).to_string(), SPLIT_GROUP_ENTRY, false),
            (split_module("tests", aero, lanes).to_string(), SPLIT_TESTS_ENTRY, false),
        ]
    } else {
        vec![
            (format!("explicit_beams_selfsplit_step{tag}"), "explicit_beams_self_step", true),
            (format!("explicit_beams_selfsplit_walk{tag}"), "explicit_beams_self_walk", false),
            (format!("explicit_beams_selfsplit_pairs{tag}"), "explicit_beams_self_pairs", false),
        ]
    }
}

/// The wide passes' list module for a fleet at `lanes` (wide fleets have no aero).
fn wide_list_module(lanes: u32) -> &'static str {
    if lanes == 512 { "explicit_beams_selfsplit_list_wide" } else { "explicit_beams_selfsplit_list" }
}

/// The self split's step layout: the split group path's entries and the indirect
/// arguments at binding 10 (storage).
fn self_split_step_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut e = split_entries();
    e.push(wgpu::BindGroupLayoutEntry {
        binding: 10,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    e
}

/// [`split_entries`] with the fleet-wide front's two indirect-argument buffers (bindings
/// 10 and 11, storage): the groups' and the fleet plan's modules.
fn fleet_args_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut e = self_split_step_entries();
    let mut second = *e.last().expect("binding 10");
    second.binding = 11;
    e.push(second);
    e
}

fn split_module(which: &str, aero: bool, lanes: u32) -> &'static str {
    let tag = match (aero, lanes) {
        (false, 512) => "_wide",
        (false, _) => "",
        (true, 512) => "_aero_wide",
        (true, _) => "_aero",
    };
    let name = format!("explicit_beams_split_{which}{tag}");
    crate::EXPLICIT_BEAMS_SPLIT_GROUPS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(n, _)| *n)
        .unwrap_or_else(|| panic!("explicit beams: no split module {name}"))
}

/// The split group path's pipelines and bind group (contact-groups-design.md): the
/// members' step (a workgroup per vehicle) and the groups' pass, binding 0 one
/// parameter copy (the step) by the dynamic offset.
struct GroupSplit {
    /// The members' step: created at first use (self-collision split with contact runs
    /// its own vehicles' module and never needs it; a browser compiles it for minutes).
    member: std::sync::OnceLock<wgpu::ComputePipeline>,
    device: wgpu::Device,
    pipeline_layout: wgpu::PipelineLayout,
    kind: (bool, u32, bool),
    group: wgpu::ComputePipeline,
    tests: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    /// The contact pass's front as fleet-wide dispatches (explicit_beams_fleet.slang,
    /// [`plan::PERF_FLEET_FRONT`]): the thread kernels' module and the one that writes
    /// the next dispatches' counts, created at first use. Two buffers of indirect
    /// arguments (the kernel's gArgs at binding 10, gArgs2 at 11): the first plan, run
    /// from the contact plan's arguments, writes the first (`bind_first`); the later
    /// plans, run from the first, write the second (`bind_second`: a dispatch never binds
    /// the buffer it runs from; a spare at the other binding); the groups' module zeroes
    /// both (`bind_args`).
    fleet: std::sync::OnceLock<[wgpu::ComputePipeline; 2]>,
    args_layout: wgpu::PipelineLayout,
    bind_args: wgpu::BindGroup,
    bind_first: wgpu::BindGroup,
    bind_second: wgpu::BindGroup,
    fleet_args: [wgpu::Buffer; 2],
}

/// Bytes of each of the fleet-wide front's indirect arguments, and their slots (the
/// kernel's FL_ARGS_*: in the first the triangles and points, the grids, the sorted
/// copy, one workgroup; in the second the walk and the pairs).
const FLEET_ARGS_BYTES: u64 = 64;
const FLEET_ELEMENT_ARGS: u64 = 0;
const FLEET_GRID_ARGS: u64 = 16;
const FLEET_SORT_ARGS: u64 = 32;
const FLEET_ONE_ARGS: u64 = 48;
const FLEET_WALK_ARGS: u64 = 0;
const FLEET_PAIRS_ARGS: u64 = 16;
/// Byte offsets in the plan's indirect arguments of the fleet-wide front's first
/// dispatches (the kernel's vvArgs words 32, 36): a thread a pair, one workgroup.
const FLEET_SCAN_ARGS: u64 = 128;
const FLEET_FIRST_ARGS: u64 = 144;
/// … and every group when one is past `VV_PART` members (words 44..46).
const FLEET_LARGER_ARGS: u64 = 176;
const FLEET_FRONT_ENTRY: &str = "explicit_beams_fleet_front";
const FLEET_PLAN_ENTRY: &str = "explicit_beams_fleet_plan";

impl GroupSplit {
    fn new(device: &wgpu::Device, bufs: &[wgpu::Buffer; 9], aero: bool, lanes: u32, zero: bool) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-group-split"),
            entries: &split_entries(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-group-split"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let kind = (aero, lanes, zero);
        // With the fleet's arguments at binding 10 (the groups' module zeroes them after
        // a pass, the plan module writes them).
        let layout_args = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-group-split-args"),
            entries: &fleet_args_entries(),
        });
        let args_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-group-split-args"),
            bind_group_layouts: &[Some(&layout_args)],
            immediate_size: 0,
        });
        let args_buffer = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("explicit-beams-fleet-args"),
                size: FLEET_ARGS_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            })
        };
        let fleet_args = [args_buffer(), args_buffer()];
        let spare = [args_buffer(), args_buffer()];
        let group = split_pipe(device, &args_layout, kind, "group", SPLIT_GROUP_ENTRY);
        let tests = split_pipe(device, &pipeline_layout, kind, "tests", SPLIT_TESTS_ENTRY);
        fn entries<'a>(bufs: &'a [wgpu::Buffer; 9], args: Option<[&'a wgpu::Buffer; 2]>) -> Vec<wgpu::BindGroupEntry<'a>> {
            let mut e: Vec<wgpu::BindGroupEntry> = (0..9)
                .map(|i| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 { params_window(&bufs[0]) } else { bufs[i].as_entire_binding() },
                })
                .collect();
            if let Some([a, b]) = args {
                e.push(wgpu::BindGroupEntry { binding: 10, resource: a.as_entire_binding() });
                e.push(wgpu::BindGroupEntry { binding: 11, resource: b.as_entire_binding() });
            }
            e
        }
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams-group-split"),
            layout: &layout,
            entries: &entries(bufs, None),
        });
        let with = |a: &wgpu::Buffer, b: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("explicit-beams-group-split-args"),
                layout: &layout_args,
                entries: &entries(bufs, Some([a, b])),
            })
        };
        let bind_args = with(&fleet_args[0], &fleet_args[1]);
        let bind_first = with(&fleet_args[0], &spare[0]);
        let bind_second = with(&spare[1], &fleet_args[1]);
        Self {
            member: std::sync::OnceLock::new(),
            device: device.clone(),
            pipeline_layout,
            kind,
            group,
            tests,
            bind,
            fleet: std::sync::OnceLock::new(),
            args_layout,
            bind_args,
            bind_first,
            bind_second,
            fleet_args,
        }
    }

    /// [the fleet-wide front's thread kernels, its plan kernel].
    fn fleet(&self) -> &[wgpu::ComputePipeline; 2] {
        self.fleet.get_or_init(|| {
            [
                split_pipe(&self.device, &self.pipeline_layout, self.kind, "fleetfront", FLEET_FRONT_ENTRY),
                split_pipe(&self.device, &self.args_layout, self.kind, "fleetplan", FLEET_PLAN_ENTRY),
            ]
        })
    }

    fn member(&self) -> &wgpu::ComputePipeline {
        self.member
            .get_or_init(|| split_pipe(&self.device, &self.pipeline_layout, self.kind, "member", SPLIT_MEMBER_ENTRY))
    }
}

/// One of the split group path's pipelines (module `which` at the frame's variant).
fn split_pipe(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    (aero, lanes, zero): (bool, u32, bool),
    which: &str,
    entry: &str,
) -> wgpu::ComputePipeline {
    let name = split_module(which, aero, lanes);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(crate::uniformize_workgroup_lanes(module_wgsl(name)).into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        // Natively, no zeroing of workgroup memory at each workgroup's start
        // (naga's MSL zeroes it from one lane: about 0.16 ms a dispatch): every
        // workgroup value these entries read was written in the same dispatch
        // (contact-groups-design.md §4; checked bit for bit against the zeroing
        // build, RUBBLE_GROUP_SPLIT_ZERO=1).
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: zero,
            ..Default::default()
        },
        cache: None,
    })
}

/// Self-collision's split pipelines (`EB_SELF_SPLIT`; selfcol-design.md): the solo
/// frame's step cut at its tests and the pairs dispatch, bound as the split group path's.
struct SelfSplit {
    /// Without vehicle contact: [the step, the walk, the pairs], compiled on first use.
    solo: std::sync::OnceLock<[wgpu::ComputePipeline; 3]>,
    list: std::sync::OnceLock<wgpu::ComputePipeline>,
    /// The step's bind group (the arguments buffer at binding 10, storage) and the pairs'
    /// (without it: the pairs dispatch takes its size from it).
    step_bind: wgpu::BindGroup,
    pairs_bind: wgpu::BindGroup,
    /// The pairs' then the walk's indirect arguments, three slots of three words each.
    args: wgpu::Buffer,
    /// Measurement only: the step module's `EB_ABLATE` build, compiled on first use
    /// ([`ExplicitBeamsGpu::set_ablation`] with the split on).
    ablate: std::sync::OnceLock<[wgpu::ComputePipeline; 3]>,
    /// With vehicle contact: the vehicles' dispatch (the split group path's members'),
    /// the walk and the pairs that skip a group's members, compiled on first use.
    vv: std::sync::OnceLock<[wgpu::ComputePipeline; 3]>,
    device: wgpu::Device,
    step_layout: wgpu::PipelineLayout,
    pairs_layout: wgpu::PipelineLayout,
    tag: &'static str,
    zero: bool,
}

impl SelfSplit {
    fn new(device: &wgpu::Device, bufs: &[wgpu::Buffer; 9], aero: bool, lanes: u32, zero: bool) -> Self {
        let pairs_entries = split_entries();
        let step_entries = self_split_step_entries();
        let layouts = [&step_entries, &pairs_entries].map(|entries| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("explicit-beams-self-split"),
                entries,
            });
            let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("explicit-beams-self-split"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            (layout, pipeline)
        });
        let tag = match (aero, lanes) {
            (false, 512) => "_wide",
            (false, _) => "",
            (true, 512) => "_aero_wide",
            (true, _) => "_aero",
        };
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-self-split-args"),
            size: SELF_ARGS_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let bind = |layout: &wgpu::BindGroupLayout, with_args: bool| {
            let mut entries: Vec<wgpu::BindGroupEntry> = (0..9)
                .map(|i| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 { params_window(&bufs[0]) } else { bufs[i].as_entire_binding() },
                })
                .collect();
            if with_args {
                entries.push(wgpu::BindGroupEntry { binding: 10, resource: args.as_entire_binding() });
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("explicit-beams-self-split"),
                layout,
                entries: &entries,
            })
        };
        let step_bind = bind(&layouts[0].0, true);
        let pairs_bind = bind(&layouts[1].0, false);
        let [(_, step_layout), (_, pairs_layout)] = layouts;
        Self {
            solo: std::sync::OnceLock::new(),
            list: std::sync::OnceLock::new(),
            step_bind,
            pairs_bind,
            args,
            ablate: std::sync::OnceLock::new(),
            vv: std::sync::OnceLock::new(),
            device: device.clone(),
            step_layout,
            pairs_layout,
            tag,
            zero,
        }
    }

    fn pipeline(&self, name: &str, entry: &str, layout: &wgpu::PipelineLayout) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(crate::uniformize_workgroup_lanes(module_wgsl(name)).into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: self.zero,
                ..Default::default()
            },
            cache: None,
        })
    }

    /// Without vehicle contact: [the step, the walk, the pairs].
    fn solo_pipes(&self) -> &[wgpu::ComputePipeline; 3] {
        self.solo.get_or_init(|| {
            let t = self.tag;
            [
                self.pipeline(&format!("explicit_beams_selfsplit_step{t}"), "explicit_beams_self_step", &self.step_layout),
                self.pipeline(&format!("explicit_beams_selfsplit_walk{t}"), "explicit_beams_self_walk", &self.pairs_layout),
                self.pipeline(&format!("explicit_beams_selfsplit_pairs{t}"), "explicit_beams_self_pairs", &self.pairs_layout),
            ]
        })
    }

    /// The wide passes' reused pair list: the list's part of a step (explicit_beams_wide_list).
    fn list_pipe(&self) -> &wgpu::ComputePipeline {
        self.list.get_or_init(|| self.pipeline(&format!("explicit_beams_selfsplit_list{}", self.tag), "explicit_beams_wide_list", &self.step_layout))
    }

    /// With vehicle contact: [the vehicles' step, the walk, the pairs].
    fn vv_pipes(&self) -> &[wgpu::ComputePipeline; 3] {
        self.vv.get_or_init(|| {
            let t = self.tag;
            [
                self.pipeline(&format!("explicit_beams_selfsplit_member_vv{t}"), SPLIT_MEMBER_ENTRY, &self.step_layout),
                self.pipeline(&format!("explicit_beams_selfsplit_walk_vv{t}"), "explicit_beams_self_walk", &self.pairs_layout),
                self.pipeline(&format!("explicit_beams_selfsplit_pairs_vv{t}"), "explicit_beams_self_pairs", &self.pairs_layout),
            ]
        })
    }

    /// The step module's ablation build (measurement only; `_wide` and 256-lane builds).
    /// The modules' ablation builds [step, walk, pairs] (measurement only; `_wide` and
    /// 256-lane builds).
    fn ablate_pipes(&self) -> &[wgpu::ComputePipeline; 3] {
        self.ablate.get_or_init(|| {
            let t = self.tag;
            [
                self.pipeline(&format!("explicit_beams_selfsplit_step_ablate{t}"), "explicit_beams_self_step", &self.step_layout),
                self.pipeline(&format!("explicit_beams_selfsplit_walk_ablate{t}"), "explicit_beams_self_walk", &self.pairs_layout),
                self.pipeline(&format!("explicit_beams_selfsplit_pairs_ablate{t}"), "explicit_beams_self_pairs", &self.pairs_layout),
            ]
        })
    }
}

/// Whether solo frames run self-collision's listed pairs across workgroups
/// ([`ExplicitBeamsGpu::set_self_split`]): off unless `RUBBLE_SELF_SPLIT=1` (work in
/// progress: not yet verified bit for bit on a GPU).
pub fn self_split_default() -> bool {
    std::env::var("RUBBLE_SELF_SPLIT").as_deref() == Ok("1")
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

/// Measurement only: the explicit modules' WGSL read from another build's generated
/// directory (`NAME.wgsl`, e.g. a copy of an older build's `OUT_DIR/wgsl`) instead of
/// this build's, for the pipelines created after the call (`None`: this build's). With
/// it, one process runs an older kernel against the current one from one snapshot
/// (`vehicle_lab --ab-switch kernel`): bit for bit and time. Process-wide.
#[doc(hidden)]
pub fn set_explicit_module_override(dir: Option<std::path::PathBuf>) {
    *MODULE_OVERRIDE.lock().unwrap() = dir;
}

static MODULE_OVERRIDE: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

fn module_override(name: &str) -> Option<&'static str> {
    type Cache = std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, &'static str>>;
    static READ: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    let dir = MODULE_OVERRIDE.lock().unwrap().clone()?;
    let path = dir.join(format!("{name}.wgsl"));
    let mut cache = READ.get_or_init(Default::default).lock().unwrap();
    let text = cache.entry(path.clone()).or_insert_with(|| {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Box::leak(text.into_boxed_str())
    });
    Some(*text)
}

/// The WGSL of a plan module ([`engine_exec::explicit_plan::MODULES`]).
fn module_wgsl(name: &str) -> &'static str {
    if let Some(text) = module_override(name) {
        return text;
    }
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
        "explicit_beams_sleep_list" => crate::EXPLICIT_BEAMS_SLEEP_LIST_WGSL,
        _ => EXPLICIT_BEAMS_VV_PASSES
            .iter()
            .chain(EXPLICIT_BEAMS_WIDE_PASSES.iter())
            .chain(crate::EXPLICIT_BEAMS_NOSURF.iter())
            .chain(crate::EXPLICIT_BEAMS_NOCPL.iter())
            .chain(crate::EXPLICIT_BEAMS_CONTACT_BIG.iter())
            .chain(crate::EXPLICIT_BEAMS_SURF_SPLIT.iter())
            .chain(crate::EXPLICIT_BEAMS_SPLIT_GROUPS.iter())
            .chain(crate::EXPLICIT_BEAMS_SELF_SPLIT.iter())
            .chain(crate::EXPLICIT_BEAMS_WIDE_LIST.iter())
            .chain(SELF_SPLIT_ABLATE.iter())
            .find(|(n, _)| *n == name)
            .map(|(_, w)| *w)
            .unwrap_or_else(|| panic!("explicit beams: no WGSL module {name} (a measurement module needs the `test-kernels` feature)")),
    }
}

/// The self split's ablation modules (measurement; built with the `test-kernels` feature).
#[cfg(feature = "test-kernels")]
const SELF_SPLIT_ABLATE: &[(&str, &str)] = &crate::EXPLICIT_BEAMS_SELF_SPLIT_ABLATE;
#[cfg(not(feature = "test-kernels"))]
const SELF_SPLIT_ABLATE: &[(&str, &str)] = &[];

/// A frame kernel's module for an upload: [`Kernel::module_for`], and with surface
/// contact but without the World's coupling (`coupled` false) its `_nocpl` twin
/// ([`crate::EXPLICIT_BEAMS_NOCPL`]): the surface pass without the proxy paths no such
/// upload can reach. The same results; a much smaller kernel for a browser to compile.
fn frame_module(kernel: Kernel, aero: bool, lanes: u32, surface: bool, coupled: bool) -> &'static str {
    let name = kernel.module_for(aero, lanes, surface);
    if surface && !coupled {
        if let Some((twin, _)) = crate::EXPLICIT_BEAMS_NOCPL
            .iter()
            .find(|(n, _)| n.strip_suffix("_nocpl") == Some(name))
        {
            return twin;
        }
    }
    name
}

/// Whether an upload's surface contact runs split from the frame
/// ([`surface_split_default`]): only with the World's coupling. Without it the frame
/// with the pass in it (the `_nocpl` module) is small enough, and one pipeline.
fn surface_split_for(coupled: bool) -> bool {
    coupled && surface_split_default()
}

/// The vehicle a "left its frame bounds" error names.
fn left_vehicle(e: &str) -> Option<u32> {
    let head = e.split(" left its frame bounds").next()?;
    head.rsplit(' ').next()?.parse().ok()
}

/// The most a frame-bounds error widens the contact margin to (m).
pub const MAX_AUTO_CONTACT_MARGIN: f32 = 16.0;

impl Contact {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &wgpu::Device,
        buffers: &[&wgpu::Buffer; 9],
        (contact_wgsl, solo_wgsl, big_wgsl): (&str, &str, &str),
        acc: &wgpu::Buffer,
        (status_at, cap_at): (u64, u64),
        args_init: &[u8],
        build_pipes: bool,
        big_groups: bool,
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
            sleep: make(plan::SLEEP_LIST, module_wgsl(plan::SLEEP_LIST), &plan_layout),
            groups: make(plan::FRAME_ENTRY, contact_wgsl, &contact_layout),
            solo: make(plan::FRAME_ENTRY, solo_wgsl, &contact_layout),
            big: big_groups.then(|| make(plan::CONTACT_ENTRY, big_wgsl, &contact_layout)),
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
        Kernel::ContactBig => "explicit large contact groups",
        Kernel::Vv(0) => "explicit vv bounds",
        Kernel::Vv(1) => "explicit vv pairs",
        Kernel::Vv(_) => "explicit vv group plan",
        Kernel::Wide(_) => "explicit wide",
        Kernel::Sleep => "explicit sleep list",
        Kernel::FrameSeg => "explicit frame segment",
        Kernel::Surf(i) => plan::SURF_PASSES[i].0,
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
        let (aero, lanes, surface) = (layout.aero, layout.lanes, layout.surface);
        let u = crate::uniformize_workgroup_lanes;
        // More than one vehicle: the contact kernels run while contact is on; the
        // plain frame only once it is turned off (created at first use).
        let deferred = vehicles.len() > 1;
        let coupled = coupling.is_some();
        let mut specs = if surface && surface_split_for(coupled) {
            // Surface contact split from the frame: its segments' module and the
            // surface pass's pipelines (Split::new's descriptors).
            let seg = Kernel::FrameSeg.module(aero, lanes);
            std::iter::once((seg, plan::FRAME_ENTRY))
                .chain(plan::SURF_PASSES.iter().copied())
                .map(|(module, entry)| crate::ComputeSpec {
                    label: module,
                    module,
                    wgsl: u(module_wgsl(module)),
                    entry,
                    layout: Some(split_entries()),
                    deferred,
                })
                .collect()
        } else {
            vec![crate::ComputeSpec {
                label: "explicit_beams_frame",
                module: frame_module(Kernel::Frame, aero, lanes, surface, coupled),
                wgsl: u(module_wgsl(frame_module(Kernel::Frame, aero, lanes, surface, coupled))),
                entry: plan::FRAME_ENTRY,
                layout: None,
                deferred,
            }]
        };
        if layout.wide_ok {
            specs.extend(plan::WIDE_PASSES.iter().map(|&entry| crate::ComputeSpec {
                label: entry,
                module: entry,
                wgsl: u(module_wgsl(entry)),
                entry,
                layout: Some(wide_entries()),
                deferred: false,
            }));
        }
        if vehicles.len() > 1 {
            specs.extend(plan::VV_PASSES.iter().chain([&plan::SLEEP_LIST]).map(|&entry| crate::ComputeSpec {
                label: entry,
                module: entry,
                wgsl: u(module_wgsl(entry)),
                entry,
                layout: Some(contact_entries(true)),
                deferred: false,
            }));
            // (The larger groups' module only for a fleet that can form one.)
            let big = (vehicles.len() > plan::VV_SIDE).then_some(Kernel::ContactBig);
            for kernel in [Kernel::Contact, Kernel::Solo].into_iter().chain(big) {
                specs.push(crate::ComputeSpec {
                    label: kernel.entry(),
                    module: frame_module(kernel, aero, lanes, surface, coupled),
                    wgsl: u(module_wgsl(frame_module(kernel, aero, lanes, surface, coupled))),
                    entry: kernel.entry(),
                    layout: Some(contact_entries(false)),
                    deferred: false,
                });
            }
        }
        if vehicles.len() > 1 {
            // The split group path's three pipelines (GroupSplit::new's descriptors),
            // after the first frame's: they run only once a contact group forms.
            for (which, entry) in [
                ("member", SPLIT_MEMBER_ENTRY),
                ("group", SPLIT_GROUP_ENTRY),
                ("tests", SPLIT_TESTS_ENTRY),
                ("fleetfront", FLEET_FRONT_ENTRY),
                ("fleetplan", FLEET_PLAN_ENTRY),
            ] {
                let module = split_module(which, aero, lanes);
                // (The groups' and the fleet plan's modules bind the fleet's arguments.)
                let args = matches!(which, "group" | "fleetplan");
                specs.push(crate::ComputeSpec {
                    label: entry,
                    module,
                    wgsl: u(module_wgsl(module)),
                    entry,
                    layout: Some(if args { fleet_args_entries() } else { split_entries() }),
                    deferred: true,
                });
            }
        }
        // Self-collision split's pipelines (selfcol-design.md), deferred: a host that
        // compiles them ahead gates the split on them (crate::pipeline_ready), so the
        // first frames run the tests in the workgroup until they are compiled. Not with
        // surface contact (the split runs only without it).
        if !surface {
            for (name, entry, args) in self_split_modules(aero, lanes, vehicles.len() > 1) {
                if specs.iter().any(|s| s.module == name) {
                    continue;
                }
                let module: &'static str = crate::EXPLICIT_BEAMS_SELF_SPLIT
                    .iter()
                    .chain(crate::EXPLICIT_BEAMS_SPLIT_GROUPS.iter())
                    .find(|(n, _)| *n == name)
                    .map(|(n, _)| *n)
                    .unwrap_or_else(|| panic!("explicit beams: no module {name}"));
                specs.push(crate::ComputeSpec {
                    label: entry,
                    module,
                    wgsl: u(module_wgsl(module)),
                    entry,
                    layout: Some(if args { self_split_step_entries() } else { split_entries() }),
                    deferred: true,
                });
            }
        }
        // The wide passes' reused pair list (explicit_beams_wide_list), deferred as the
        // self split's: wide frames walk their own grid until it is compiled.
        if !surface && !aero && layout.wide_ok {
            let name = wide_list_module(lanes);
            specs.push(crate::ComputeSpec {
                label: "explicit_beams_wide_list",
                module: name,
                wgsl: u(module_wgsl(name)),
                entry: "explicit_beams_wide_list",
                layout: Some(self_split_step_entries()),
                deferred: true,
            });
        }
        specs.sort_by(|a, b| b.wgsl.len().cmp(&a.wgsl.len()));
        // The split group path's member module is the largest by far (209 s cold in
        // Chrome on Metal) and runs only with self-collision split off: where the self
        // split's modules are compiled too (its own members' step takes its place), it
        // goes last, behind every module the default path waits for.
        if !surface {
            let member = split_module("member", aero, lanes);
            if let Some(i) = specs.iter().position(|s| s.module == member) {
                let spec = specs.remove(i);
                specs.push(spec);
            }
        }
        specs
    }

    /// [`Self::compute_specs`] for a host that runs self-collision split
    /// ([`Self::set_self_split`]) from its first frame: where the fleet's touching cars
    /// then take the self split's contact path (more than one vehicle, no surface
    /// contact: its member, walk, pairs, group and tests modules are in `specs`), those
    /// are compiled before the first frame and the groups' own workgroup's kernels
    /// (contact, larger groups, solo: that path's fallback, taken only with the split
    /// off) after it. The same modules, another order: a cold start is no longer, and
    /// touching cars run across the GPU from the first frame instead of after the
    /// deferred compiles (two minutes on Metal in Chrome).
    pub fn prefer_self_split(specs: &mut [crate::ComputeSpec]) {
        let ahead = [
            "explicit_beams_selfsplit_member_vv",
            "explicit_beams_selfsplit_walk_vv",
            "explicit_beams_selfsplit_pairs_vv",
            "explicit_beams_split_group",
            "explicit_beams_split_tests",
            "explicit_beams_split_fleetfront",
            "explicit_beams_split_fleetplan",
        ];
        if !specs.iter().any(|s| s.module.starts_with(ahead[0])) {
            return;
        }
        let fallback = [Kernel::Contact.entry(), Kernel::Solo.entry(), Kernel::ContactBig.entry()];
        for s in specs.iter_mut() {
            if ahead.iter().any(|m| s.module.starts_with(m)) {
                s.deferred = false;
            } else if fallback.contains(&s.label) && !s.deferred {
                s.deferred = true;
            }
        }
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
        Self::with_sleep(device, queue, vehicles, coupling, None)
    }

    /// [`Self::with_coupling`] with vehicle sleep (`engine_softbody::explicit_sleep`, on
    /// from the first frame) when `sleep` is set. `None`: no room; frames as without it,
    /// bit for bit.
    pub fn with_sleep(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vehicles: &[ExplicitBeams],
        coupling: Option<CouplingCapacity>,
        sleep: Option<engine_softbody::explicit_sleep::VehicleSleep>,
    ) -> Self {
        let upload = ExplicitUpload::with_sleep(vehicles, coupling, sleep);
        let layout = upload.layout.clone();
        let (aero, big, lanes) = (layout.aero, layout.big, layout.lanes);
        // Large vehicles get twice the lanes (more loads in flight per step), a
        // small fleet 512 (frame_lanes).
        let coupled = coupling.is_some();
        let frame_module = frame_module(Kernel::Frame, aero, lanes, layout.surface, coupled);
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
        // (COPY_DST: the parking brake input lives in its records.)
        let statics = init(
            "explicit-beams-static",
            plan::BIND_STATIC,
            storage | wgpu::BufferUsages::COPY_DST,
        );
        let lists = init("explicit-beams-lists", plan::BIND_LISTS, storage);
        let nodes = init("explicit-beams-nodes", plan::BIND_NODES, state);
        let beams = init("explicit-beams-beams", plan::BIND_BEAMS, state);
        let acc = init("explicit-beams-acc", plan::BIND_ACC, state);
        let groups = init("explicit-beams-groups", plan::BIND_GROUPS, state);
        let misc = init("explicit-beams-misc", plan::BIND_MISC, state);
        let next = init("explicit-beams-grid-next", plan::BIND_NEXT, state);
        let buffers = [
            &params, &statics, &lists, &nodes, &beams, &acc, &groups, &misc, &next,
        ];
        let vehicles_n = layout.spans.len();
        let wide = layout
            .wide_ok
            .then(|| Wide::new(device, queue, &buffers, upload.table_records()));
        let contact = Contact::new(
            device,
            &buffers,
            (
                module_wgsl(self::frame_module(Kernel::Contact, aero, lanes, layout.surface, coupled)),
                module_wgsl(self::frame_module(Kernel::Solo, aero, lanes, layout.surface, coupled)),
                module_wgsl(self::frame_module(Kernel::ContactBig, aero, lanes, layout.surface, coupled)),
            ),
            &acc,
            (layout.status.base, layout.status.cap),
            &upload.bytes(plan::BIND_ARGS),
            vehicles.len() > 1,
            vehicles.len() > plan::VV_SIDE,
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
        let mut planner = FramePlanner::new(&layout);
        planner.set_surface_split(surface_split_for(coupled));
        if std::env::var("RUBBLE_SLEEP_LIST").as_deref() == Ok("0") {
            planner.set_sleep_listing(false);
        }
        let one = vehicles.len() <= 1;
        let gpu = Self {
            device: device.clone(),
            queue: queue.clone(),
            frame: std::sync::OnceLock::new(),
            frame_module,
            split: std::sync::OnceLock::new(),
            split_module: Kernel::FrameSeg.module(aero, lanes),
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
            coupling_cache_off: false,
            debug_coupling_bits: 0,
            coupling_ranges_off: coupling_node_ranges_off_default(),
            coupling_list_off: std::env::var("RUBBLE_COUPLING_LIST").as_deref() == Ok("0"),
            coupling_fold_off: std::env::var("RUBBLE_COUPLING_FOLD").as_deref() == Ok("0"),
            debug_big: None,
            debug_solo: None,
            debug_groups: None,
            vehicle_margins: vec![0.0; vehicles_n],
            surface_pairs_off: std::env::var("RUBBLE_SURFACE_PAIRS").as_deref() == Ok("0"),
            surface_culls_off: std::env::var("RUBBLE_SURFACE_CULLS").as_deref() == Ok("0"),
            retired_bits: Vec::new(),
            label_words: Vec::new(),
            labelled: vec![false; vehicles_n],
            piece_windows: std::env::var("RUBBLE_PIECE_WINDOWS").as_deref() != Ok("0"),
            wide,
            contact,
            group_split: group_split_default(),
            group_split_tests: group_split_tests_default(),
            vv_grid: true,
            group_split_pipes: std::sync::OnceLock::new(),
            group_split_zero: std::env::var("RUBBLE_GROUP_SPLIT_ZERO").as_deref() == Ok("1"),
            self_split: self_split_default(),
            self_split_pipes: std::sync::OnceLock::new(),
            self_split_frame: std::cell::Cell::new(0),
            wide_list: std::env::var("RUBBLE_WIDE_LIST").as_deref() == Ok("1"),
            wide_fused: std::env::var("RUBBLE_WIDE_FUSED").as_deref() == Ok("1"),
            wide_list_frame: std::cell::Cell::new(0),
            wide_part: std::cell::Cell::new(0),
            self_split_vv: std::cell::Cell::new(false),
            self_split_skip: 0,
            split_dispatch_timing: std::env::var("RUBBLE_SPLIT_DISPATCH_TIMING").as_deref() == Ok("1"),
            contact_errors: Default::default(),
            contact_errors_seen: Default::default(),
            bufs,
            module_kind: (aero, big, lanes),
            variant: KernelVariant::Current,
            alts: Vec::new(),
            probe: None,
            det: None,
            acc,
            sleep_inputs: vehicles.iter().map(|v| v.drive.as_ref().map(|d| d.input)).collect(),
            sleep_grab: None,
            sleep_wake: Default::default(),
        };
        // One vehicle runs the plain frame (or, with surface contact, the split one)
        // from the first frame on.
        if one {
            if gpu.planner.split_frames(true) {
                gpu.split_pipes();
            } else {
                gpu.frame_pipe();
            }
        }
        gpu
    }

    /// Put the fleet back to `vehicles`' state in place: every state buffer rewritten as a
    /// new upload of them would hold it, the contact plan as at the first frame. No buffer,
    /// pipeline or bind group is made (a restart costs a few buffer writes). `vehicles`
    /// must be the fleet this was built for (the same vehicles in the same order, the
    /// same statics count, coupling room and sleep setting): only their state may differ
    /// (node positions and velocities, beam state, the origin, the ground's height, the
    /// static shapes' places, the drive state). A fleet of another layout is an error and
    /// changes nothing. Host settings made since the build (coupling on, vehicle contact,
    /// margins, variants) stay.
    pub fn reset_state(
        &mut self,
        vehicles: &[ExplicitBeams],
        coupling: Option<CouplingCapacity>,
        sleep: Option<engine_softbody::explicit_sleep::VehicleSleep>,
    ) -> Result<(), String> {
        let upload = ExplicitUpload::with_sleep(vehicles, coupling, sleep);
        let new = &upload.layout;
        let old = &self.layout;
        let key = |l: &ExplicitLayout| -> Vec<[u32; 8]> { l.spans.iter().map(|s| [s.node_off, s.nodes, s.beam_off, s.beams, s.slide_off, s.slides, s.seg_off, s.tris]).collect() };
        let same = key(new) == key(old)
            && new.rest0.len() == old.rest0.len()
            && new.total == old.total
            && (new.aero, new.big, new.lanes, new.surface, new.wide_ok) == (old.aero, old.big, old.lanes, old.surface, old.wide_ok);
        if !same {
            return Err("explicit beams: reset_state needs the fleet this was built for (another layout)".into());
        }
        // (binding, buffer index): statics, nodes, beams, acc, groups, misc, next. The
        // lists (topology) are not state.
        let writes = [
            (plan::BIND_STATIC, 1usize),
            (plan::BIND_NODES, 3),
            (plan::BIND_BEAMS, 4),
            (plan::BIND_ACC, 5),
            (plan::BIND_GROUPS, 6),
            (plan::BIND_MISC, 7),
            (plan::BIND_NEXT, 8),
        ];
        let data: Vec<Vec<u8>> = writes.iter().map(|(b, _)| upload.bytes(*b)).collect();
        let args = upload.bytes(plan::BIND_ARGS);
        if writes.iter().zip(&data).any(|((_, i), d)| d.len() as u64 != self.bufs[*i].size()) || args.len() as u64 != self.contact.args.size() {
            return Err("explicit beams: reset_state needs the fleet this was built for (another buffer size)".into());
        }
        for ((_, i), d) in writes.iter().zip(&data) {
            self.queue.write_buffer(&self.bufs[*i], 0, d);
        }
        self.queue.write_buffer(&self.contact.args, 0, &args);
        if let Some(w) = &self.wide {
            self.queue.write_buffer(&w.table, 0, bytemuck::cast_slice(upload.table_records()));
        }
        // The parameter words the new state changes (the ground, the statics' ranges),
        // over the planner's (which hold the host's settings in other words).
        for k in 0..new.words.len() {
            if new.words[k] != old.words[k] {
                self.planner.words[k] = new.words[k];
            }
        }
        self.planner.restore(FramePlanner::new(new).state());
        self.layout = upload.layout.clone();
        // Nothing retired, no labels (the misc and next records were rewritten).
        self.retired_bits.clear();
        self.label_words.clear();
        self.labelled.iter_mut().for_each(|l| *l = false);
        self.planner.words[plan::W_RETIRE_BASE] = 0;
        self.sleep_inputs = vehicles.iter().map(|v| v.drive.as_ref().map(|d| d.input)).collect();
        self.sleep_grab = None;
        self.sleep_wake.lock().unwrap().clear();
        self.contact_errors.lock().unwrap().clear();
        self.contact_errors_seen.lock().unwrap().clear();
        // The host settings that live in the rewritten misc records: the coupling's
        // header and the vehicles' contact margins.
        if self.layout.coupling.is_some() {
            self.write_coupling_header();
        }
        for v in 0..self.vehicle_margins.len() {
            let m = self.vehicle_margins[v];
            if m != 0.0 {
                self.set_vehicle_contact_margin(v, m);
            }
        }
        self.planner.request_replan();
        Ok(())
    }

    /// The split frame's pipelines (see `split`).
    fn split_pipes(&self) -> &Split {
        self.split.get_or_init(|| Split::new(&self.device, &self.bufs, self.split_module))
    }

    /// The plain frame's pipeline and bind group, created at first use (see `frame`).
    fn frame_pipe(&self) -> &(wgpu::ComputePipeline, wgpu::BindGroup) {
        self.frame.get_or_init(|| {
            let device = &self.device;
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("explicit_beams"),
                source: wgpu::ShaderSource::Wgsl(
                    crate::uniformize_workgroup_lanes(module_wgsl(self.frame_module)).into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("explicit_beams_frame"),
                layout: None,
                module: &module,
                entry_point: Some(plan::FRAME_ENTRY),
                compilation_options: Default::default(),
                cache: None,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("explicit-beams"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &std::array::from_fn::<_, 9, _>(|i| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 0 { params_window(&self.bufs[0]) } else { self.bufs[i].as_entire_binding() },
                }),
            });
            (pipeline, group)
        })
    }

    /// Diagnostic: the per-vehicle frame module's WGSL as the frame compiles it.
    #[doc(hidden)]
    pub fn debug_frame_wgsl(&self) -> String {
        crate::uniformize_workgroup_lanes(module_wgsl(self.frame_module))
    }

    /// Diagnostic (ablations of an edited frame module, [`Self::debug_set_frame_wgsl`]):
    /// bits OR-ed into the coupling header word from the next frame (the shipped kernel
    /// reads bits 1 to 32 only).
    #[doc(hidden)]
    pub fn debug_coupling_bits(&mut self, bits: u32) {
        self.debug_coupling_bits = bits;
        self.write_coupling_header();
    }

    /// Diagnostic: the large contact groups' module's WGSL as its pipeline compiles it.
    #[doc(hidden)]
    pub fn debug_contact_big_wgsl(&self) -> String {
        self.debug_contact_wgsl("big")
    }

    /// Diagnostic: a contact module's WGSL (`which`: "solo", "groups" or "big").
    #[doc(hidden)]
    pub fn debug_contact_wgsl(&self, which: &str) -> String {
        let (aero, _, lanes) = self.module_kind;
        let k = match which {
            "solo" => Kernel::Solo,
            "groups" => Kernel::Contact,
            _ => Kernel::ContactBig,
        };
        crate::uniformize_workgroup_lanes(module_wgsl(frame_module(k, aero, lanes, self.layout.surface, self.layout.coupling.is_some())))
    }

    /// Diagnostic: run `src` as a contact module ("solo", "groups" or "big") from now on.
    #[doc(hidden)]
    pub fn debug_set_contact_wgsl(&mut self, which: &str, src: &str) {
        let entry = if which == "big" { plan::CONTACT_ENTRY } else { plan::FRAME_ENTRY };
        let made = self.debug_contact_pipe(src, entry);
        match which {
            "solo" => self.debug_solo = Some(made),
            "groups" => self.debug_groups = Some(made),
            _ => self.debug_big = Some(made),
        }
    }

    /// Diagnostic: run `src` as the large contact groups' module from now on.
    #[doc(hidden)]
    pub fn debug_set_contact_big_wgsl(&mut self, src: &str) {
        self.debug_set_contact_wgsl("big", src);
    }

    fn debug_contact_pipe(&self, src: &str, entry: &str) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let device = &self.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit_beams_contact (debug)"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        // The contact entry's own layout (as Contact::new builds it).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("explicit-beams-vv (debug)"),
            entries: &contact_entries(false),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("explicit-beams-vv (debug)"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("explicit_beams_contact (debug)"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams-vv (debug)"),
            layout: &bgl,
            entries: &std::array::from_fn::<_, 9, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: if i == 0 { params_window(&self.bufs[0]) } else { self.bufs[i].as_entire_binding() },
            }),
        });
        (pipeline, group)
    }

    /// Diagnostic: the byte size of each state binding's buffer (bindings 0..=8).
    #[doc(hidden)]
    pub fn debug_binding_sizes(&self) -> Vec<u64> {
        self.bufs.iter().map(|b| b.size()).collect()
    }

    /// Diagnostic (browser ablations without a rebuild): frames run `src` as the
    /// per-vehicle frame module from now on (its entry point and bindings as the
    /// shipped one's). An invalid module is the device's error.
    #[doc(hidden)]
    pub fn debug_set_frame_wgsl(&mut self, src: &str) {
        let device = &self.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("explicit_beams (debug)"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("explicit_beams_frame (debug)"),
            layout: None,
            module: &module,
            entry_point: Some(plan::FRAME_ENTRY),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("explicit-beams (debug)"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &std::array::from_fn::<_, 9, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: if i == 0 { params_window(&self.bufs[0]) } else { self.bufs[i].as_entire_binding() },
            }),
        });
        self.frame = std::sync::OnceLock::new();
        let _ = self.frame.set((pipeline, group));
    }

    /// Vehicle sleep's layout (`None`: built without room, [`Self::with_sleep`]).
    pub fn sleep_layout(&self) -> Option<plan::SleepLayout> {
        self.layout.sleep
    }

    /// Vehicle sleep on or off (only with room; on from the start). Turning it on wakes
    /// every vehicle (still counts from 0). Returns whether it is on.
    pub fn set_sleep(&mut self, on: bool) -> bool {
        let Some(s) = self.layout.sleep else {
            return false;
        };
        if on && !self.planner.sleep_on() {
            let zeros = vec![[0u32; 4]; self.layout.vehicles()];
            self.queue
                .write_buffer(&self.misc, s.record_offset(0), bytemuck::cast_slice(&zeros));
        }
        self.planner.set_sleep(on)
    }

    /// Whether vehicle sleep runs.
    pub fn sleep_on(&self) -> bool {
        self.planner.sleep_on()
    }

    /// Measurement only: dispatch through the awake list (the default) or every vehicle
    /// with a sleeping one's workgroup returning at once (`RUBBLE_SLEEP_LIST=0` at
    /// creation). The same results.
    pub fn set_sleep_listing(&mut self, on: bool) {
        self.planner.set_sleep_listing(on);
    }

    /// Wake vehicle `v` before the next frame (a scripted force or any host write the
    /// GPU does not see as motion). Without sleep, nothing.
    pub fn wake(&self, v: usize) {
        if self.layout.sleep.is_some() && v < self.layout.vehicles() {
            self.sleep_wake.lock().unwrap().push(v);
        }
    }

    /// The host's wakes, written before a frame: those queued, the grabbed vehicle and
    /// every vehicle with throttle (an engine driving wheels is not still).
    fn write_wakes(&self) {
        let Some(s) = self.layout.sleep else {
            return;
        };
        if !self.planner.sleep_on() {
            return;
        }
        let mut list = std::mem::take(&mut *self.sleep_wake.lock().unwrap());
        list.extend(self.sleep_grab);
        list.extend(
            self.sleep_inputs
                .iter()
                .enumerate()
                .filter(|(_, i)| i.is_some_and(|i| i.forward != 0.0))
                .map(|(v, _)| v),
        );
        list.sort_unstable();
        list.dedup();
        // One write a run of consecutive vehicles (their records are consecutive): a fleet
        // with every car under throttle costs one write, not one a car.
        let mut i = 0;
        while i < list.len() {
            let mut j = i + 1;
            while j < list.len() && list[j] == list[j - 1] + 1 {
                j += 1;
            }
            self.queue
                .write_buffer(&self.misc, s.record_offset(list[i]), &vec![0u8; 16 * (j - i)]);
            i = j;
        }
    }

    /// Host writes that will wake a vehicle before the next frame (queued wakes, a grab,
    /// throttle held).
    pub fn wakes_pending(&self) -> bool {
        self.planner.sleep_on()
            && (!self.sleep_wake.lock().unwrap().is_empty()
                || self.sleep_grab.is_some()
                || self.sleep_inputs.iter().any(|i| i.is_some_and(|i| i.forward != 0.0)))
    }

    /// Blocking read: per vehicle its sleep record (still steps, asleep); empty without
    /// room.
    pub fn sleep_states(&self) -> Vec<(u32, bool)> {
        let Some(s) = self.layout.sleep else {
            return Vec::new();
        };
        let misc = self.read_u4(&self.misc);
        (0..self.layout.vehicles())
            .map(|v| {
                let r = misc[s.base as usize + v];
                (r[0], r[1] & engine_softbody::explicit_sleep::SLEEP_ASLEEP != 0)
            })
            .collect()
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

    /// The fleet's frame bounds margin (m): [`Self::set_contact_margin`]'s.
    pub fn contact_margin(&self) -> f32 {
        self.planner.contact_margin()
    }

    /// Vehicle `v`'s own frame bounds margin (m; the fleet's applies when wider): the
    /// host's after `v` left its frame bounds, or a host setting.
    pub fn set_vehicle_contact_margin(&mut self, v: usize, metres: f32) {
        assert!(metres >= 0.0 && metres.is_finite(), "contact margin {metres}");
        self.vehicle_margins[v] = metres;
        let rec = self.planner.vv_record(v as u32, plan::VV_MEMBER_REC);
        self.queue.write_buffer(&self.misc, u64::from(rec) * 16 + 4, bytemuck::bytes_of(&metres));
        self.planner.request_replan();
    }

    /// Vehicle `v`'s frame bounds margin (m): the fleet's, or its own when it left its
    /// frame bounds (each time doubled, at most [`MAX_AUTO_CONTACT_MARGIN`]).
    pub fn vehicle_contact_margin(&self, v: usize) -> f32 {
        self.vehicle_margins[v].max(self.planner.contact_margin())
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
    /// The loop-cap flag's word (gAcc, byte offset): a host's own readback reads it
    /// with its step ([`crate::explicit_loop_cap_hit`] of the word), without waiting.
    pub fn loop_cap_word(&self) -> (&wgpu::Buffer, u64) {
        (&self.contact.acc, self.contact.cap_at)
    }

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
        if !self.contact_active() {
            return Vec::new();
        }
        // The plan's group count (indirect-arguments word 3; word 4 is the frame's
        // workgroups, vehicles + groups).
        let n = bytemuck::cast_slice::<u8, u32>(&self.read(&self.contact.args))[3] as usize;
        let misc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.misc)).to_vec();
        let at = self.planner.words[46] as usize;
        (0..n).map(|g| misc[(at + g) * 4 + 1]).collect()
    }

    /// Measurement (a blocking read): per vehicle the self-collision hits of the last
    /// step run (the kernel's hit counter, cleared by a step's node phase).
    #[doc(hidden)]
    pub fn self_hit_counts(&self) -> Vec<u32> {
        let acc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[5])).to_vec();
        (0..self.layout.spans.len()).map(|k| acc[self.layout.hit_acc as usize + 2 * k]).collect()
    }

    /// Measurement (blocking reads): per contact group of the last frame on the split
    /// group path, its members, its points and, per member in vehicle order, the last
    /// pass's (listed pairs, hits, listed triangles, candidates) from the group's
    /// scratch (explicit_beams_split.slang splitVvExport).
    #[doc(hidden)]
    pub fn contact_group_counts(&self) -> Vec<(u32, u32, Vec<[u32; 4]>)> {
        if !self.contact_active() {
            return Vec::new();
        }
        let n = bytemuck::cast_slice::<u8, u32>(&self.read(&self.contact.args))[3] as usize;
        let misc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.misc)).to_vec();
        let next: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[8])).to_vec();
        let acc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[5])).to_vec();
        let w = &self.planner.words;
        let (nv, grid, groups, members) = (w[1] as usize, w[plan::W_VV_SCRATCH] as usize, w[46] as usize, w[47] as usize);
        // With the owners' state by vehicle: each member's last pass's counts (the
        // kernel's flOwnerEnd, flVehAt 1..4).
        let fleet = (self.fleet_flags() & plan::PERF_FLEET_STATE != 0).then(|| {
            let pair_cap = w[49] as usize;
            w[plan::W_VV_FLEET_NEXT] as usize + (nv + 1) + 6 * pair_cap + 5 * (nv + 1)
        });
        (0..n)
            .map(|g| {
                let rec = &misc[(groups + g) * 4..(groups + g) * 4 + 4];
                let (off, k, points) = (rec[0] as usize, rec[1] as usize, rec[3]);
                if k == 0 || k > plan::VV_PART {
                    return (k as u32, points, Vec::new());
                }
                if let Some(at) = fleet {
                    let per = (0..k)
                        .map(|a| {
                            let v = next[members + nv + off + a] as usize;
                            std::array::from_fn(|r| next[at + (1 + r) * nv + v])
                        })
                        .collect();
                    return (k as u32, points, per);
                }
                let v0 = next[members + nv + off] as usize;
                let c = grid + v0 * 1024;
                let per = (0..k).map(|a| [acc[c + a], acc[c + plan::VV_PART + a], acc[c + 3 * plan::VV_PART + a], acc[c + 4 * plan::VV_PART + a]]).collect();
                (k as u32, points, per)
            })
            .collect()
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
        let cmds = self.write_params(steps, true);
        // (With the reused pair list a step's beams command also dispatches the list
        // kernel, the walk and the pairs: each in a timed pass of its own.)
        let count = 2 * (cmds.len() as u32 + 3 * steps);
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
        let tails = self.wide_list_frame.get() > 0;
        for cmd in &cmds {
            if let Cmd::Dispatch { kernel: Kernel::Wide(kind), .. } = cmd {
                // Part 1: the command's own dispatch; 2..4: the list kernel, the walk,
                // the pairs after a step's beams.
                let parts = if tails && *kind == plan::WIDE_BEAMS_PASS { 4 } else { 1 };
                for part in 1..=parts {
                    let k = 2 * kinds.len() as u32;
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("explicit_beams_wide_profile"),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &set,
                            beginning_of_pass_write_index: Some(k),
                            end_of_pass_write_index: Some(k + 1),
                        }),
                    });
                    self.wide_part.set(if tails { part } else { 0 });
                    self.dispatch(&mut pass, cmd, None);
                    self.wide_part.set(0);
                    kinds.push(if part == 1 { *kind } else { EXPLICIT_BEAMS_WIDE_PASSES.len() + part as usize - 2 });
                }
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
            .chain([("list", 0.0), ("walk", 0.0), ("pairs", 0.0)])
            .collect();
        for (i, &kind) in kinds.iter().enumerate() {
            // (An empty indirect dispatch has no work to stamp on Metal: its stamps are
            // stale or its neighbours'; an interval past 100 ms is one of those.)
            let ms = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period;
            if ms < 100.0 {
                out[kind].1 += ms;
            }
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
        let cmds = self.write_params(steps, false);
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

    /// Tests and measurement: the surface contact's pair path for the proxies
    /// (`explicit_beams_surface.slang` surfPairFind; lossless) on or off (the coupling
    /// header's bit 16; `RUBBLE_SURFACE_PAIRS=0` starts it off). Off, the per-element
    /// loops meet every proxy.
    #[doc(hidden)]
    pub fn set_surface_pairs(&mut self, on: bool) {
        self.surface_pairs_off = !on;
        self.write_coupling_header();
    }

    /// Tests and measurement: the shared feature queries' edge culls
    /// (`feature_query.slang` fqFar; lossless) on or off (the coupling header's bit 32;
    /// `RUBBLE_SURFACE_CULLS=0` starts them off).
    #[doc(hidden)]
    pub fn set_surface_culls(&mut self, on: bool) {
        self.surface_culls_off = !on;
        self.write_coupling_header();
    }

    /// The self-collision grid walk's pair list built once and reused over steps while
    /// it provably holds every pair the walk would test (`explicit_beams.slang`
    /// pairReuse*; lossless: the same hits, bit for bit) on, or listed every step (off).
    /// On by default ([`plan::PAIR_REUSE_MARGIN`]; `RUBBLE_PAIR_REUSE=0` starts it
    /// off). No-op for a fleet without the pair list.
    pub fn set_pair_reuse(&mut self, on: bool) {
        let m = plan::pair_reuse_default();
        let m = if m == 0 { plan::PAIR_REUSE_MARGIN.to_bits() } else { m };
        self.set_pair_reuse_margin(if on { f32::from_bits(m) } else { 0.0 });
    }

    /// The beam phase on its records sorted by kind ([`plan::W_BEAM_ORDER`]; on, the
    /// default) or the beams in file order (off: beamUpdate for each, as before). The
    /// same results bit for bit (`tests/explicit_beam_order_gpu.rs`). Applies from the
    /// next frame.
    pub fn set_beam_order(&mut self, on: bool) {
        self.planner.words[plan::W_BEAM_ORDER] = if on { self.layout.words[plan::W_BEAM_ORDER] } else { 0 };
    }

    /// The kernels' lossless schedule switches ([`plan::W_PERF_FLAGS`], `plan::PERF_*`
    /// bits; `RUBBLE_PERF_FLAGS` sets the start value): each the same results bit for
    /// bit on or off. Applies from the next frame.
    pub fn set_perf_flags(&mut self, flags: u32) {
        self.planner.set_perf_flags(flags);
    }

    pub fn perf_flags(&self) -> u32 {
        self.planner.perf_flags()
    }

    /// The reused pair list's piece labels for vehicle `v` (`explicit_beams.slang`
    /// pieces*; one per node of the vehicle, [`engine_softbody::explicit_debris::piece_labels`]):
    /// 0 the main body, 1..=6 a piece with collision triangles of its own,
    /// [`plan::PIECE_LOOSE`] any other node off the main body. A hint only: any labels give
    /// the same results bit for bit (the list's validity is proved on the GPU for the
    /// labels given); good labels let a piece torn off the vehicle move without
    /// invalidating the main body's list, and the coupling takes a window per piece
    /// ([`Self::set_piece_windows`]). All zero (the default): one group, as before.
    /// Applies from the next frame.
    pub fn set_piece_labels(&mut self, v: usize, labels: &[u8]) {
        let base = self.planner.words[plan::W_PIECE_BASE] as usize;
        let Some(span) = self.layout.spans.get(v).copied() else { return };
        let n = (span.nodes as usize).min(labels.len());
        let words: Vec<u32> = labels[..n].iter().map(|&l| u32::from(l)).collect();
        let active = u32::from(words.iter().any(|&l| l != 0));
        // The pair list's (a vehicle fleet with it).
        if base != 0 {
            self.queue.write_buffer(&self.bufs[8], ((base + span.node_off as usize) * 4) as u64, bytemuck::cast_slice(&words));
            let at = base + self.layout.total[0] as usize + v;
            self.queue.write_buffer(&self.bufs[8], (at * 4) as u64, bytemuck::bytes_of(&active));
        }
        // The coupling's copy (vehicle_coupling.slang pieceLabelOf, pieceAccept).
        let total = self.layout.total[0] as usize;
        if self.label_words.is_empty() {
            self.label_words = vec![0u32; total.div_ceil(32).max(1) * 4];
        }
        let first = span.node_off as usize;
        for i in first..first + span.nodes as usize {
            let l = labels.get(i - first).copied().unwrap_or(0).min(7) as u32;
            let (w, sh) = (i / 8, (i % 8) * 4);
            self.label_words[w] = (self.label_words[w] & !(0xF << sh)) | (l << sh);
        }
        if span.nodes > 0 {
            let (r0, r1) = (first / 32, (first + span.nodes as usize - 1) / 32);
            let rec = self.layout.label_rec as usize;
            self.queue.write_buffer(&self.misc, ((rec + r0) * 16) as u64, bytemuck::cast_slice(&self.label_words[r0 * 4..(r1 + 1) * 4]));
        }
        self.labelled[v] = active != 0;
        let flag = [active, 0, 0, 0];
        let rec = self.layout.piece_box_rec as usize + v * plan::PIECE_BOX_RECS + 12;
        self.queue.write_buffer(&self.misc, (rec * 16) as u64, bytemuck::cast_slice(&flag));
    }

    /// Vehicle contact's candidates in a grid in the small groups' triangle tests
    /// (`explicit_beams.slang` vvGridBuild; lossless: the same candidates pass the box
    /// test) on (the default) or off (every candidate against every listed triangle).
    #[doc(hidden)]
    pub fn set_vv_grid(&mut self, on: bool) {
        self.vv_grid = on;
        let w = (self.layout.status.base / 4) as usize + 9;
        let off: i32 = if on { 0 } else { 1 };
        self.queue.write_buffer(&self.bufs[5], (w * 4) as u64, bytemuck::bytes_of(&off));
    }

    /// Measurement: candidates the small groups' triangle tests visited since the last
    /// call (which clears the count).
    #[doc(hidden)]
    pub fn take_vv_visits(&mut self) -> u64 {
        let w = (self.layout.status.base / 4) as usize + 8;
        let acc: Vec<i32> = bytemuck::cast_slice(&self.read(&self.bufs[5])).to_vec();
        let n = acc.get(w).copied().unwrap_or(0) as u32 as u64;
        self.queue.write_buffer(&self.bufs[5], (w * 4) as u64, bytemuck::bytes_of(&0i32));
        n
    }

    /// The coupling's window per piece (vehicle_coupling.slang pieceAccept; on by default,
    /// `RUBBLE_PIECE_WINDOWS=0` off): with piece labels a collider is a proxy only when it
    /// can reach some piece's box, not just the whole vehicle's. Off: one window.
    pub fn set_piece_windows(&mut self, on: bool) {
        self.piece_windows = on;
    }

    /// The coupling's piece label and box records (0, 0: one window per vehicle).
    pub fn piece_window_records(&self) -> (u32, u32) {
        if self.piece_windows && self.labelled.iter().any(|&l| l) {
            (self.layout.label_rec, self.layout.piece_box_rec)
        } else {
            (0, 0)
        }
    }

    /// Retire nodes of vehicle `v` (`retired`: one flag per node; `explicit_beams.slang`
    /// nodeRetired, the CPU reference's [`engine_softbody::explicit_beams::ExplicitBeams::retired`]):
    /// from the next frame a retired node is no longer simulated (left at rest where it
    /// is, no contact of any kind, out of the coupling's bounds and the self-collision
    /// list's). The host decides ([`engine_softbody::explicit_debris::DebrisTracker`]);
    /// a flag cleared brings a node back as it was left. Fails while vehicle contact is
    /// active (its contact groups do not read the flags). Nothing retired yet: every
    /// result as before (the kernel's retirement code is off until a node is retired).
    pub fn set_retired(&mut self, v: usize, retired: &[bool]) -> Result<(), String> {
        let Some(span) = self.layout.spans.get(v).copied() else {
            return Err(format!("set_retired: no vehicle {v}"));
        };
        if self.contact_active() && retired.iter().any(|&r| r) {
            return Err("retired debris needs vehicle contact off (its contact groups do not skip retired nodes)".into());
        }
        let total = self.layout.total[0] as usize;
        if self.retired_bits.is_empty() {
            self.retired_bits = vec![0u32; total.div_ceil(128).max(1) * 4];
        }
        let n = (span.nodes as usize).min(retired.len());
        let (first, last) = (span.node_off as usize, span.node_off as usize + span.nodes as usize);
        for i in first..last {
            let on = i - first < n && retired[i - first];
            let (w, b) = (i / 32, i % 32);
            if on {
                self.retired_bits[w] |= 1 << b;
            } else {
                self.retired_bits[w] &= !(1 << b);
            }
        }
        if span.nodes > 0 {
            let (r0, r1) = (first / 128, (last - 1) / 128);
            let rec = self.layout.retire_rec as usize;
            self.queue.write_buffer(
                &self.misc,
                ((rec + r0) * 16) as u64,
                bytemuck::cast_slice(&self.retired_bits[r0 * 4..(r1 + 1) * 4]),
            );
        }
        if self.retired_bits.iter().any(|&w| w != 0) {
            self.planner.words[plan::W_RETIRE_BASE] = self.layout.retire_rec;
        }
        Ok(())
    }

    /// The retired nodes' gMisc record for the coupling's passes (0: none retired).
    pub fn retire_base(&self) -> u32 {
        self.planner.words[plan::W_RETIRE_BASE]
    }

    /// Measurement: the reused pair list's margin in metres (0: off).
    #[doc(hidden)]
    pub fn set_pair_reuse_margin(&mut self, metres: f32) {
        self.planner.words[plan::W_PAIR_REUSE] = if metres > 0.0 { metres.to_bits() } else { 0 };
    }

    /// Measurement: the reused pair list's backoff ([`plan::W_PAIR_STREAK`]): a list
    /// reused on fewer steps than `steps` sends the next grid steps to the walk.
    #[doc(hidden)]
    pub fn set_pair_reuse_streak(&mut self, steps: u32) {
        self.planner.words[plan::W_PAIR_STREAK] = steps;
    }

    /// Measurement: vehicle contact's kept candidates ([`plan::PERF_FLEET_KEEP`], the
    /// kernel's fkCtl) since the last call (which clears them): (builds, contact passes
    /// that reused the kept pairs, pairs kept summed over the builds, passes with partners
    /// on the front's own path, the most pairs one owner kept at a build, builds because
    /// a calm node had moved past its allowance, builds because a pair was not covered or
    /// no list was kept, the largest bound at a build that ended a list for motion (a
    /// node's, or two vehicles') and the current list's allowance for two vehicles'
    /// relative motion, both float bits, builds because two vehicles had moved past it;
    /// the other builds are a wild node's). Zeros without the fleet state.
    #[doc(hidden)]
    pub fn take_fleet_keep_stats(&mut self) -> [u64; 10] {
        let w = &self.planner.words;
        let at = w[plan::W_VV_FLEET_NEXT] as usize;
        if at == 0 {
            return [0; 10];
        }
        let nv = self.layout.spans.len();
        let end = at + plan::fleet_next_words(nv, w[49] as usize, self.layout.total[2] as usize);
        let ctl = end - plan::FLEET_CTL_WORDS;
        let next: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[8])).to_vec();
        // (Word 14 is the passes' clock, not a count of this window.)
        let out = std::array::from_fn(|k| u64::from(next[ctl + if k == 9 { 15 } else { 5 + k }]));
        self.queue.write_buffer(&self.bufs[8], ((ctl + 5) * 4) as u64, &[0u8; 32]);
        self.queue.write_buffer(&self.bufs[8], ((ctl + 15) * 4) as u64, &[0u8; 4]);
        out
    }

    /// Measurement: per vehicle, the kept candidates' reference motion of the next pass
    /// and its tracked nodes' mean motion of the last (the kernel's fkMean; metres since
    /// the list's build), then its origin's change since the build.
    #[doc(hidden)]
    pub fn fleet_keep_motion(&self) -> Vec<[f32; 9]> {
        let w = &self.planner.words;
        let at = w[plan::W_VV_FLEET_NEXT] as usize;
        if at == 0 {
            return Vec::new();
        }
        let nv = self.layout.spans.len();
        let end = at + plan::fleet_next_words(nv, w[49] as usize, self.layout.total[2] as usize);
        let mean = end - plan::FLEET_CTL_WORDS - 6 * nv;
        let origin = mean - 3 * nv;
        let next: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[8])).to_vec();
        let misc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.misc)).to_vec();
        (0..nv)
            .map(|v| {
                std::array::from_fn(|k| {
                    if k < 6 {
                        f32::from_bits(next[mean + 6 * v + k])
                    } else {
                        f32::from_bits(misc[v * 4 + k - 6]) - f32::from_bits(next[origin + 3 * v + k - 6])
                    }
                })
            })
            .collect()
    }

    /// Measurement: the reused pair list's statistics summed over the vehicles since the
    /// last call (which clears them): (lists built, steps that reused one, pairs listed
    /// at the builds, the most listed at one build). Zeros without the pair list.
    #[doc(hidden)]
    pub fn take_pair_reuse_stats(&mut self) -> [u64; 4] {
        let at = self.planner.words[91] as usize;
        if at == 0 {
            return [0; 4];
        }
        let nv = self.layout.spans.len();
        let rec = at + self.layout.total[0] as usize;
        let beams: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[4])).to_vec();
        let mut out = [0u64; 4];
        for v in 0..nv {
            let s = &beams[(rec + v) * 4..(rec + v) * 4 + 4];
            out[0] += u64::from(s[0]);
            out[1] += u64::from(s[1]);
            out[2] += u64::from(s[2]);
            out[3] = out[3].max(u64::from(s[3]));
        }
        self.queue.write_buffer(&self.bufs[4], (rec * 16) as u64, &vec![0u8; nv * 16]);
        out
    }

    /// Measurement: the reused pair list's volatile triangles tested (summed over steps
    /// and vehicles) since the last call (which clears them). 0 without the pair list.
    #[doc(hidden)]
    pub fn take_piece_stats(&mut self) -> u64 {
        let at = self.planner.words[91] as usize;
        if at == 0 {
            return 0;
        }
        let nv = self.layout.spans.len();
        let rec = at + self.layout.total[0] as usize + nv;
        let beams: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[4])).to_vec();
        let out = (0..nv).map(|v| u64::from(beams[(rec + v) * 4])).sum();
        self.queue.write_buffer(&self.bufs[4], (rec * 16) as u64, &vec![0u8; nv * 16]);
        out
    }

    /// Measurement (an `EB_ABLATE` build with mask bit 2^29): why grid steps did not reuse
    /// the pair list, summed over vehicles since the last call (which clears them): (no
    /// list, the main body's spread over the margin, its spread with the boxes' growth,
    /// the pieces' bounds, backoff walks, non-finite, reused, the main spread summed in
    /// 0.1 mm).
    #[doc(hidden)]
    pub fn take_reuse_cause(&mut self) -> [u64; 8] {
        let at = self.planner.words[91] as usize;
        if at == 0 {
            return [0; 8];
        }
        let nv = self.layout.spans.len();
        let rec = at + self.layout.total[0] as usize + 2 * nv;
        let beams: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[4])).to_vec();
        let mut out = [0u64; 8];
        for v in 0..nv {
            for k in 0..8 {
                out[k] += u64::from(beams[(rec + 2 * v) * 4 + k]);
            }
        }
        self.queue.write_buffer(&self.bufs[4], (rec * 16) as u64, &vec![0u8; nv * 32]);
        out
    }

    /// The coupling's per-node shape ranges (`explicit_beams_rigid.slang` `cplGsAdd`, the
    /// header's bit 64): each node's contacts start at the first proxy shape its count
    /// pass found it inside and stop after the last (on), or scan every shape (off; the
    /// same shapes pass the cull and the face test, in the same order: the same results).
    /// Off by default in browsers' WebGPU ([`coupling_node_ranges_off_default`]), on
    /// elsewhere.
    pub fn set_coupling_node_ranges(&mut self, on: bool) {
        self.coupling_ranges_off = !on;
        self.write_coupling_header();
    }

    /// The coupling's candidate list (`explicit_beams_rigid.slang` `cplList*`, lossless:
    /// the same contacts, counts and sums, bit for bit) on (the default), or every node
    /// testing every proxy shape every step (off; `RUBBLE_COUPLING_LIST=0` starts it off).
    /// Each node lists the shapes within its skin at a frame's first step and keeps the
    /// list while its own travel and the shapes' travel since stay within the skin
    /// ([`engine_softbody::explicit_rigid::candidate_list`]); otherwise it rebuilds it.
    pub fn set_coupling_candidate_list(&mut self, on: bool) {
        self.coupling_list_off = !on;
        self.write_coupling_header();
    }

    /// The coupling's folded step (`explicit_beams_rigid.slang` `cplStepFolded`, lossless:
    /// the bodies' integration, the spheres and the count in two phases, 2 barriers a step
    /// instead of 4) on (the default; it needs the candidate list), or the four phases (off;
    /// `RUBBLE_COUPLING_FOLD=0` starts it off).
    pub fn set_coupling_fold(&mut self, on: bool) {
        self.coupling_fold_off = !on;
        self.write_coupling_header();
    }

    /// Tests: per vehicle, the nodes whose candidate list (as the last step left it) holds
    /// at least one shape, and the nodes that keep no list (more than 16 shapes in reach).
    /// Empty without the coupling.
    #[doc(hidden)]
    pub fn coupling_list_stats(&self) -> Vec<(u32, u32)> {
        let Some(c) = self.layout.coupling else {
            return Vec::new();
        };
        let next: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[8])).to_vec();
        let words = engine_exec::explicit_plan::CouplingLayout::LIST_NODE_WORDS as usize;
        let base = c.list_base as usize
            + (engine_exec::explicit_plan::CouplingLayout::LIST_SHAPE_WORDS * c.cap.shapes * c.vehicles) as usize;
        self.layout
            .spans
            .iter()
            .map(|sp| {
                let (mut listed, mut over) = (0, 0);
                for i in 0..sp.nodes as usize {
                    let at = base + (sp.node_off as usize + i) * words + 4;
                    match next.get(at).copied() {
                        Some(w) if w & 0xFFFF == 0xFFFE => over += 1,
                        Some(w) if w & 0xFFFF != 0xFFFF => listed += 1,
                        _ => {}
                    }
                }
                (listed, over)
            })
            .collect()
    }

    /// Tests and measurement: the coupling's workgroup copies and ranges (on by default;
    /// lossless, `explicit_beams_rigid.slang`: the header words, the proxy spheres and
    /// each node's shape range from the count pass in workgroup memory, the shapes'
    /// step reactions a lane a shape before each body sums them in shape order, a body's
    /// shapes scanned over the range the plan wrote, a small contact group's members'
    /// coupling side by side) off: every read from the buffers, every scan over every
    /// shape, as before them.
    #[doc(hidden)]
    pub fn set_coupling_workgroup_cache(&mut self, on: bool) {
        self.coupling_cache_off = !on;
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
        // Bit 8 (measurement only, `RUBBLE_SURFACE_CAST=0`): the surface contact's casts of
        // fast proxies off (explicit_beams_surface.slang surfProxyFast).
        let casts_off = engine_softbody::explicit_surface::casts_off();
        let word = u32::from(self.coupling_on)
            | if self.coupling_on && self.coupling_cull_off { 2 } else { 0 }
            | if self.coupling_on && self.coupling_cache_off { 4 } else { 0 }
            | if self.coupling_on && casts_off { 8 } else { 0 }
            | if self.coupling_on && self.surface_pairs_off {
                16
            } else {
                0
            }
            | if self.coupling_on && self.surface_culls_off {
                32
            } else {
                0
            }
            | if self.coupling_on && self.coupling_ranges_off { 64 } else { 0 }
            | if self.coupling_on && self.coupling_list_off { COUPLING_LIST_OFF } else { 0 }
            | if self.coupling_on && self.coupling_fold_off { COUPLING_FOLD_OFF } else { 0 }
            | if self.coupling_on { self.debug_coupling_bits } else { 0 };
        self.queue.write_buffer(
            &self.misc,
            self.layout.coupling_on_offset(),
            bytemuck::bytes_of(&word),
        );
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
        let read = self.encode(&mut enc, steps, None, None).0;
        self.queue.submit([enc.finish()]);
        if read {
            self.contact.map_status();
        }
    }

    /// [`Self::frame`] encoded into `enc` (a host's own submission, such as the World's
    /// step). Submit it, then call [`Self::frame_submitted`] with the result.
    pub fn frame_encode(&mut self, enc: &mut wgpu::CommandEncoder, steps: u32) -> bool {
        self.frame_encode_timed(enc, steps, None)
    }

    /// [`Self::frame_encode`], its GPU time written to a host's query set: timestamp
    /// `stamp.1` as the frame's first pass begins and `stamp.1 + 1` as its last ends
    /// (wasm-safe: the host resolves and reads them with its own readback).
    pub fn frame_encode_timed(&mut self, enc: &mut wgpu::CommandEncoder, steps: u32, stamp: Option<(&wgpu::QuerySet, u32)>) -> bool {
        if self.layout.spans.is_empty() || steps == 0 {
            return false;
        }
        self.encode(enc, steps, stamp, None).0
    }

    /// [`Self::frame_encode`] with each dispatch in its own pass, timed in a host's
    /// query set: dispatch i (while i < `cap`) writes timestamps `2 (first + i)` and
    /// `2 (first + i) + 1`. Returns the status-read flag and the timed dispatches'
    /// kernel labels, in order (the passes add their own cost: a diagnostic).
    pub fn frame_encode_per_dispatch(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        steps: u32,
        set: &wgpu::QuerySet,
        first: u32,
        cap: u32,
    ) -> (bool, Vec<&'static str>) {
        if self.layout.spans.is_empty() || steps == 0 {
            return (false, Vec::new());
        }
        self.encode(enc, steps, None, Some((set, first, cap)))
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
    fn encode(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        steps: u32,
        ext: Option<(&wgpu::QuerySet, u32)>,
        per: Option<(&wgpu::QuerySet, u32, u32)>,
    ) -> (bool, Vec<&'static str>) {
        let mut labels = Vec::new();
        self.check_contact_status();
        self.write_wakes();
        let wide_allowed = self.alt_pipes().is_none();
        let mut cmds = self.write_params(steps, wide_allowed);
        // Timestamps at the start of the frame's first pass and the end of its last.
        let slot = self.timing.as_mut().and_then(|t| {
            (t.used < t.capacity).then(|| {
                t.used += 1;
                2 * t.used - 2
            })
        });
        let alt = self.alt_pipes();
        let vv = self.self_split_vv.get();
        let split = vv || self.group_split_on(steps);
        if split {
            // The solo frame as planned (every vehicle when no group formed, else none),
            // then the split path's dispatches (below) in place of the groups' frames:
            // with a group, every vehicle a step a dispatch. With self-collision split,
            // every vehicle a step a dispatch in every frame (no solo frame).
            cmds.retain(|c| match c {
                Cmd::Indirect { kernel: Kernel::Contact | Kernel::ContactBig, .. } => false,
                Cmd::Indirect { kernel: Kernel::Solo, .. } => !vv,
                _ => true,
            });
        }
        let set = self.timing.as_ref().map(|t| &t.set);
        let stamps = |begin: bool, end: bool| {
            ext.or(set.zip(slot))
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
            let last = i + 1 == runs.len();
            if let Some((q, first, cap)) = per {
                // Each dispatch its own timed pass (past `cap`, untimed), labelled.
                let mut timed_pass = |enc: &mut wgpu::CommandEncoder, label: &'static str, f: &mut dyn FnMut(&mut wgpu::ComputePass)| {
                    let k = first + labels.len() as u32;
                    let timed = (labels.len() as u32) < cap;
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("explicit_beams"),
                        timestamp_writes: timed.then_some(wgpu::ComputePassTimestampWrites {
                            query_set: q,
                            beginning_of_pass_write_index: Some(2 * k),
                            end_of_pass_write_index: Some(2 * k + 1),
                        }),
                    });
                    f(&mut pass);
                    if timed {
                        labels.push(label);
                    }
                };
                // (Split frames: each of their dispatches timed on its own only with
                // RUBBLE_SPLIT_DISPATCH_TIMING=1: a hundred-odd passes a frame cost more
                // than the dispatches they time, and empty indirect passes time nothing.)
                let each = self.split_dispatch_timing;
                for cmd in *run {
                    let frame = matches!(cmd, Cmd::Dispatch { kernel: Kernel::Frame, .. } | Cmd::Indirect { kernel: Kernel::Frame, .. });
                    if frame && self.self_split_frame.get() > 0 {
                        if each {
                            // Self-collision split: its step, walk and pairs dispatches.
                            for d in self_split_schedule(self.planner.vehicles(), self.self_split_frame.get()) {
                                timed_pass(enc, d.label(), &mut |pass| self.dispatch_self_one(pass, d));
                            }
                        } else {
                            timed_pass(enc, "explicit self split frame", &mut |pass| self.dispatch(pass, cmd, alt));
                        }
                        continue;
                    }
                    timed_pass(enc, kernel_name(cmd), &mut |pass| self.dispatch(pass, cmd, alt));
                }
                if split && last {
                    if each {
                        for op in self.group_split_ops(steps) {
                            timed_pass(enc, op.label(), &mut |pass| self.run_group_op(pass, op));
                        }
                    } else {
                        let label = if self.self_split_vv.get() { "explicit self split groups" } else { "explicit split groups" };
                        timed_pass(enc, label, &mut |pass| self.dispatch_group_split(pass, steps));
                    }
                }
                continue;
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
                if split && last {
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("explicit_beams"),
                            timestamp_writes: None,
                        });
                        self.dispatch_group_split(&mut pass, steps);
                    }
                    m.mark(enc, "explicit split groups");
                }
                continue;
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("explicit_beams"),
                timestamp_writes: stamps(i == 0, last),
            });
            for cmd in *run {
                self.dispatch(&mut pass, cmd, alt);
            }
            if split && last {
                self.dispatch_group_split(&mut pass, steps);
            }
        }
        // The status every STATUS_EVERY frames (the counters and the loop-cap
        // flag keep an error), with or without contact.
        let read = self.planner.status_due();
        if read {
            self.contact.read_status(&self.device, enc);
        }
        (read, labels)
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
        if kernel == Kernel::Frame && self.self_split_frame.get() > 0 {
            self.dispatch_self_split(pass, self.self_split_frame.get());
            return;
        }
        // A wide frame with the reused pair list: no triangles' or hits' pass of its own;
        // after a step's beams the list kernel's part, the walk, the pairs and the hits'
        // part; before the frame's end the list's end.
        let wl = self.wide_list_frame.get();
        // (profile_wide_frame: one of a beams command's list, walk and pairs dispatches.)
        let part = self.wide_part.get();
        if wl > 0 && part >= 2 {
            if let Cmd::Dispatch { kernel: Kernel::Wide(kind), param_copy: k, .. } = *cmd {
                if kind == plan::WIDE_BEAMS_PASS {
                    let nv = self.planner.vehicles();
                    let slot = u64::from(if k == 0 { 2 } else { k & 1 });
                    self.dispatch_self_one(pass, match part {
                        2 => SelfDispatch::List { copy: k, groups: nv },
                        3 => SelfDispatch::Walk { copy: k, offset: 12 * (3 + slot) },
                        _ => SelfDispatch::Pairs { copy: k, offset: 12 * slot },
                    });
                }
            }
            return;
        }
        if wl > 0 {
            if let Cmd::Dispatch { kernel: Kernel::Wide(kind), .. } = *cmd {
                if kind == plan::WIDE_TRIS {
                    return;
                }
                // (PERF_WIDE_FUSED: steps 1..'s coupling and wheels ran on the step
                // before's list and hits kernels.)
                if let Cmd::Dispatch { param_copy, .. } = *cmd {
                    if kind == plan::WIDE_AUX && self.wide_fused && param_copy >= 1 && param_copy < wl {
                        return;
                    }
                }
                if kind == plan::WIDE_END {
                    self.dispatch_self_one(pass, SelfDispatch::List { copy: wl, groups: self.planner.vehicles() });
                }
            }
        }
        match kernel {
            Kernel::Frame => {
                let (p, g) = alt.map_or_else(
                    || {
                        let f = self.frame_pipe();
                        (&f.0, &f.1)
                    },
                    |a| (&a.frame, &a.group),
                );
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
                let dbg = if kernel == Kernel::Solo { &self.debug_solo } else { &self.debug_groups };
                if let (Some((p, dg)), None) = (dbg, alt) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, dg, &[]);
                } else {
                    pass.set_pipeline(if kernel == Kernel::Solo { solo } else { groups });
                    pass.set_bind_group(0, g, &[]);
                }
            }
            Kernel::ContactBig => {
                // A test variant's contact module steps every group (no split): nothing
                // to dispatch here.
                if alt.is_some() {
                    return;
                }
                if let Some((p, g)) = &self.debug_big {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, g, &[]);
                } else {
                    let p = c.pipes.as_ref().expect("explicit beams: contact passes for one vehicle");
                    // At most VV_PART vehicles: no group this dispatch steps can
                    // exist (its indirect count is always 0), and it has no pipeline.
                    let Some(big) = &p.big else { return };
                    pass.set_pipeline(big);
                    pass.set_bind_group(0, &c.contact_group, &[]);
                }
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
            Kernel::Sleep => {
                let p = c.pipes.as_ref().expect("explicit beams: the sleep list for one vehicle");
                pass.set_bind_group(0, &c.plan_group, &[]);
                pass.set_pipeline(&p.sleep);
            }
            Kernel::FrameSeg | Kernel::Surf(_) => {
                let s = self.split_pipes();
                let Cmd::Dispatch { param_copy, .. } = *cmd else { unreachable!() };
                pass.set_pipeline(match kernel {
                    Kernel::Surf(i) => &s.surf[i],
                    _ => &s.frame,
                });
                pass.set_bind_group(0, &s.group, &[param_copy * PARAM_STRIDE as u32]);
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
        if wl > 0 && part == 0 {
            if let Cmd::Dispatch { kernel: Kernel::Wide(kind), param_copy: k, .. } = *cmd {
                if kind == plan::WIDE_BEAMS_PASS {
                    let nv = self.planner.vehicles();
                    let slot = u64::from(if k == 0 { 2 } else { k & 1 });
                    self.dispatch_self_one(pass, SelfDispatch::List { copy: k, groups: nv });
                    self.dispatch_self_one(pass, SelfDispatch::Walk { copy: k, offset: 12 * (3 + slot) });
                    self.dispatch_self_one(pass, SelfDispatch::Pairs { copy: k, offset: 12 * slot });
                }
            }
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
        assert!(
            !(variant == KernelVariant::CurrentDet && self.layout.surface),
            "explicit beams: the EB_DET_HITS build has no surface pass (EB_NO_SURFACE): a fleet without surface contact only"
        );
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
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e} (the test kernels are built with engine-webgpu's `test-kernels` feature)"))
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

    /// Measurement only (time; results are then meaningless): the ablation build's second
    /// mask (`EB_A2`, explicit_beams.slang: the solo frame's smaller phases), read only by
    /// the ablation kernels ([`Self::set_ablation`] with a nonzero mask selects them).
    #[doc(hidden)]
    pub fn set_ablation2(&mut self, mask: u32) {
        self.planner.set_ablation2(mask);
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

    /// Tests only: a [`Self::snapshot`] as bytes (blocking), for [`Self::snapshot_from_bytes`]
    /// in another process: a kernel's frames compared with another build's from one state.
    #[doc(hidden)]
    pub fn snapshot_to_bytes(&self, snap: &GpuSnapshot) -> Vec<u8> {
        let mut out: Vec<u32> = snap.state.to_words().to_vec();
        out.push(snap.copies.len() as u32);
        for c in &snap.copies {
            let words: Vec<u32> = bytemuck::pod_collect_to_vec(&self.read(c));
            out.push(words.len() as u32);
            out.extend(words);
        }
        bytemuck::cast_slice(&out).to_vec()
    }

    /// Tests only: a snapshot from [`Self::snapshot_to_bytes`] (the same fleet and layout).
    #[doc(hidden)]
    pub fn snapshot_from_bytes(&self, bytes: &[u8]) -> GpuSnapshot {
        let w: Vec<u32> = bytemuck::pod_collect_to_vec(bytes);
        let state = plan::PlannerState::from_words([w[0], w[1], w[2], w[3], w[4]]);
        let n = w[5] as usize;
        let mut at = 6;
        let mut copies = Vec::new();
        for b in self.snapshot_sources().iter().take(n) {
            let len = w[at] as usize;
            at += 1;
            // (A shorter buffer: the indirect arguments of a build with fewer dispatches,
            // the rest zero.)
            assert!(len as u64 * 4 <= b.size(), "explicit beams: a snapshot of another layout");
            let c = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("explicit-beams-snapshot"),
                size: b.size(),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&c, 0, bytemuck::cast_slice(&w[at..at + len]));
            at += len;
            copies.push(c);
        }
        GpuSnapshot { copies, state }
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
        let words = |b: &wgpu::Buffer| -> Vec<u32> { bytemuck::cast_slice(&self.read(b)).to_vec() };
        plan::state_words(
            &self.layout,
            &self.planner.words,
            words(&self.bufs[3]),
            &words(&self.bufs[4]),
            &words(&self.bufs[5]),
            words(&self.bufs[6]),
            words(&self.bufs[7]),
            &words(&self.bufs[8]),
        )
    }

    /// Tests only: the words of [`Self::state_words`]' `misc` holding the vehicle-contact
    /// plan's records (frame boxes, flags, margins: planning state, which a margin
    /// changes): `start..end`.
    #[doc(hidden)]
    pub fn vv_record_words(&self) -> std::ops::Range<usize> {
        let at = self.planner.words[43] as usize * 4;
        at..at + self.layout.spans.len() * VV_RECS * 4
    }

    /// Tests only: the World coupling's state, beside [`Self::state_words`]: every proxy
    /// record (bodies as integrated in the frames, shapes, planes) and the coupling's
    /// gAcc sums (per body its count, per shape its step and window reactions). Empty
    /// without the coupling.
    #[doc(hidden)]
    pub fn coupling_words(&self) -> Vec<(&'static str, Vec<u32>)> {
        if self.layout.coupling.is_none() {
            return Vec::new();
        }
        let misc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.misc)).to_vec();
        let acc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[5])).to_vec();
        coupling_words_of(&self.layout, &misc, &acc)
    }

    /// Tests and the browser A/B: the state buffers copied into one MAP_READ buffer (the
    /// copy submitted now, after the work submitted so far), to be read once mapped
    /// ([`StagedState::ab_words`]: [`Self::state_words`] with the vehicle-contact plan's
    /// records masked, then [`Self::coupling_words`]). Never blocks.
    #[doc(hidden)]
    pub fn stage_state(&self) -> StagedState {
        let src = &self.bufs[3..9];
        let sizes: [u64; 6] = std::array::from_fn(|i| src[i].size());
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-staged-state"),
            size: sizes.iter().sum(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut at = 0;
        for (b, s) in src.iter().zip(sizes) {
            enc.copy_buffer_to_buffer(b, 0, &staging, at, s);
            at += s;
        }
        self.queue.submit([enc.finish()]);
        StagedState {
            staging,
            sizes,
            layout: self.layout.clone(),
            words: self.planner.words.to_vec(),
            vv: self.vv_record_words(),
        }
    }

    /// Tests and the browser A/B: [`Self::snapshot_to_bytes`] without blocking (the copy
    /// submitted now; [`StagedSnapshot::bytes`] once mapped).
    #[doc(hidden)]
    pub fn stage_snapshot(&self, snap: &GpuSnapshot) -> StagedSnapshot {
        let sizes: Vec<u64> = snap.copies.iter().map(wgpu::Buffer::size).collect();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-staged-snapshot"),
            size: sizes.iter().sum::<u64>().max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut at = 0;
        for (c, s) in snap.copies.iter().zip(&sizes) {
            enc.copy_buffer_to_buffer(c, 0, &staging, at, *s);
            at += s;
        }
        self.queue.submit([enc.finish()]);
        StagedSnapshot {
            staging,
            sizes,
            state: snap.state.to_words(),
        }
    }

    /// Tests and measurement: per vehicle, the surface pair path's last step (pairs
    /// listed, hits found; `explicit_beams_surface.slang` surfPairFind). A vehicle that
    /// took the per-element loops that step reads as it was last listed.
    #[doc(hidden)]
    pub fn surface_pair_stats(&self) -> Vec<(u32, u32)> {
        let Some(c) = self.layout.coupling else {
            return Vec::new();
        };
        let misc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.misc)).to_vec();
        let h = (self.layout.couple_rec as usize + 1) * 4;
        let (body, pairs) = (misc[h + 2] as usize, misc[h + 3] as usize);
        if pairs == 0 {
            return Vec::new();
        }
        let acc: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[5])).to_vec();
        let nv = self.layout.spans.len();
        let at = body + 3 * c.cap.bodies as usize * nv;
        (0..nv).map(|v| (acc[at + v], acc[at + nv + v])).collect()
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
            // That vehicle's own margin doubles (its frame bounds cover more travel); the
            // others keep theirs, so one fast vehicle does not group the whole fleet. A
            // margin only changes which vehicles step together, never results.
            match left_vehicle(&e).filter(|&v| (v as usize) < self.vehicle_margins.len()) {
                Some(v) => {
                    let own = self.vehicle_contact_margin(v as usize);
                    self.set_vehicle_contact_margin(v as usize, (2.0 * own).clamp(0.5, MAX_AUTO_CONTACT_MARGIN));
                }
                None => {
                    let m = (2.0 * self.planner.contact_margin()).clamp(0.5, MAX_AUTO_CONTACT_MARGIN);
                    self.planner.set_contact_margin(m);
                }
            }
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
    /// The frame's parameters and its commands: only the parameter copies the commands
    /// read are uploaded (a dispatch reads copy `param_copy`; the per-vehicle frame and
    /// contact kernels copy 0), so a frame on per-vehicle workgroups uploads one copy,
    /// not one a step. The split group path reads every step's copy and copy steps + 1
    /// (the groups' frame start, the kernel's SPLIT_BEGIN).
    fn write_params(&mut self, steps: u32, wide_allowed: bool) -> Vec<Cmd> {
        let mut bytes = self.planner.write_params(steps);
        let cmds = self.planner.commands(steps, wide_allowed);
        let selfs = self.self_split_on(steps, &cmds);
        self.self_split_frame.set(if selfs { steps } else { 0 });
        let vv = !selfs && self.self_split_vv_on(steps, &cmds);
        self.self_split_vv.set(vv);
        // A wide frame with the reused pair list: every copy (step k: the wide passes, the
        // list kernel, the walk and the pairs) carries PERF_WIDE_LIST.
        let wide = cmds.iter().any(|c| matches!(c, Cmd::Dispatch { kernel: Kernel::Wide(_), .. }));
        let wl = wide
            && self.wide_list
            && steps > 0
            && !self.planner.has_surface()
            && self.variant == KernelVariant::Current
            && self.self_split_ready(false)
            && crate::pipeline_ready(wide_list_module(self.module_kind.2));
        self.wide_list_frame.set(if wl { steps } else { 0 });
        if wl {
            let stride = PARAM_STRIDE as usize;
            let at = 4 * plan::W_PERF_FLAGS;
            let flags = self.planner.words[plan::W_PERF_FLAGS];
            for k in 0..=steps as usize {
                let fused = if self.wide_fused { plan::PERF_WIDE_FUSED } else { 0 };
                bytes[stride * k + at..stride * k + at + 4].copy_from_slice(&(flags | plan::PERF_WIDE_LIST | fused).to_le_bytes());
            }
            self.queue.write_buffer(&self.params, 0, &bytes);
            return cmds;
        }
        if selfs {
            // A step dispatch reads copy s (s = 0 ..= steps), the pairs dispatch too.
            self.queue.write_buffer(&self.params, 0, &bytes);
            return cmds;
        }
        if vv || self.group_split_on(steps) {
            // Copy steps + 1: the groups' frame start (the kernel's SPLIT_BEGIN); with the
            // tests split, copy steps + 2 + s step s's second phase (SPLIT_LEVELS) and copy
            // 2 steps + 2 + s its first group dispatch and walk (SPLIT_TESTS). The
            // vehicles' steps read copy s (the step alone).
            // (The members' rest dispatches read the SPLIT_LEVELS copies too.)
            let rest = self.member_rest_on(vv);
            let extra = if self.group_split_tests { 2 * steps + 1 } else if rest { steps + 1 } else { 1 };
            bytes.resize((PARAM_STRIDE * u64::from(steps + 1 + extra)) as usize, 0);
            // The owners' state by vehicle only where its dispatches and regions are (the
            // groups' dispatches' copies carry the flags that hold this frame).
            let flags = self.fleet_flags();
            let mut put = |k: u32, word: u32| {
                let mut w = self.planner.words;
                w[plan::W_PERF_FLAGS] = flags;
                w[11] = word;
                let at = (PARAM_STRIDE * u64::from(k)) as usize;
                bytes[at..at + 4 * w.len()].copy_from_slice(bytemuck::cast_slice(&w));
            };
            put(steps + 1, SPLIT_BEGIN);
            for s in 0..steps {
                if self.group_split_tests || rest {
                    put(steps + 2 + s, s | SPLIT_LEVELS);
                }
                if self.group_split_tests {
                    put(2 * steps + 2 + s, s | SPLIT_TESTS);
                }
            }
            self.queue.write_buffer(&self.params, 0, &bytes);
            return cmds;
        }
        let used = cmds
            .iter()
            .map(|c| match *c {
                Cmd::Dispatch { param_copy, .. } => param_copy,
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        let n = ((u64::from(used) + 1) * PARAM_STRIDE).min(bytes.len() as u64) as usize;
        self.queue.write_buffer(&self.params, 0, &bytes[..n]);
        cmds
    }

    /// Contact groups run split across workgroups (contact-groups-design.md) on, or off
    /// (each group in one workgroup, as before): the same results bit for bit. Off by
    /// default (`RUBBLE_GROUP_SPLIT=1` on). Not with vehicle sleep, the test kernel
    /// variants or the diagnostic pipelines (those frames keep the groups' workgroup).
    pub fn set_group_split(&mut self, on: bool) {
        self.group_split = on;
    }

    /// The wide passes with the reused pair list (on) or with their own grid walk and
    /// tests every step (off, the default; `RUBBLE_WIDE_LIST=1` on): the self split's
    /// list, walk and pairs kernels in the wide schedule, the same results bit for bit.
    pub fn set_wide_list(&mut self, on: bool) {
        self.wide_list = on;
    }

    /// With the wide passes' pair list: step s + 1's coupling phases and wheels' rays and
    /// bars beside step s's list phases, its tyres after step s's hits, in place of an
    /// aux dispatch a step (on) or that dispatch (off, the default; `RUBBLE_WIDE_FUSED=1`
    /// on). The same results bit for bit.
    pub fn set_wide_fused(&mut self, on: bool) {
        self.wide_fused = on;
    }

    /// Self-collision's listed pairs across workgroups (selfcol-design.md) on, or tested
    /// in each vehicle's workgroup (off, as before): the same results bit for bit. Off by
    /// default (`RUBBLE_SELF_SPLIT=1` on). For frames on the per-vehicle frame kernel
    /// alone: no vehicle contact, sleep, surface contact, wide passes, test variants or
    /// diagnostic pipelines (those frames keep the tests in the workgroup).
    pub fn set_self_split(&mut self, on: bool) {
        self.self_split = on;
    }

    /// Whether the last frame ran self-collision split ([`Self::set_self_split`]), with
    /// or without vehicle contact.
    pub fn self_split_used(&self) -> bool {
        self.self_split_frame.get() > 0 || self.self_split_vv.get()
    }

    /// Whether this frame (its commands) runs self-collision split with vehicle contact:
    /// on the split group path's dispatches, every vehicle's step in every substep's
    /// vehicles' dispatch (a solo vehicle's cut at its tests, a contact group's member's
    /// whole), then the walk and pairs dispatches over the solo vehicles, then the
    /// groups' contact passes (none without a group). The same conditions as that path
    /// (no sleep, the shipped kernels, the parameter copies, the fleet bound), and no
    /// surface contact (the self split's modules are built without it).
    fn self_split_vv_on(&self, steps: u32, cmds: &[Cmd]) -> bool {
        self.self_split
            && self.planner.contact_active()
            && !self.planner.sleep_on()
            && !self.planner.has_surface()
            && self.variant == KernelVariant::Current
            && self.debug_big.is_none()
            && self.debug_solo.is_none()
            && self.debug_groups.is_none()
            && steps > 0
            && 3 * steps + 2 <= plan::PARAM_COPIES
            && self.layout.spans.len() * 64 <= 65_535
            && self.self_split_ready(true)
            && cmds.iter().all(|c| match c {
                Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => {
                    matches!(kernel, Kernel::Vv(_) | Kernel::Solo | Kernel::Contact | Kernel::ContactBig)
                }
                Cmd::Copy { .. } => true,
            })
    }

    /// Whether this frame (its commands) runs self-collision split.
    fn self_split_on(&self, steps: u32, cmds: &[Cmd]) -> bool {
        self.self_split
            && !self.planner.contact_active()
            && !self.planner.sleep_on()
            && !self.planner.has_surface()
            // (Ablate: measurement only, the step module's ablation build.)
            && matches!(self.variant, KernelVariant::Current | KernelVariant::Ablate)
            && self.debug_big.is_none()
            && self.debug_solo.is_none()
            && self.debug_groups.is_none()
            && steps > 0
            && steps < plan::PARAM_COPIES
            && self.self_split_ready(false)
            && cmds.iter().any(|c| matches!(c, Cmd::Dispatch { kernel: Kernel::Frame, .. } | Cmd::Indirect { kernel: Kernel::Frame, .. }))
            && cmds.iter().all(|c| match c {
                Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => *kernel == Kernel::Frame,
                Cmd::Copy { .. } => true,
            })
    }

    /// Whether the self split's modules (with vehicle contact or not) are compiled
    /// (a browser host's async compiles, [`crate::pipeline_ready`]; natively always):
    /// until then frames keep the tests in the workgroup, the same results.
    fn self_split_ready(&self, vv: bool) -> bool {
        let (aero, _, lanes) = self.module_kind;
        self_split_modules(aero, lanes, vv).iter().all(|(m, _, _)| crate::pipeline_ready(m))
    }

    fn self_split_pipes(&self) -> &SelfSplit {
        self.self_split_pipes.get_or_init(|| {
            let (aero, _, lanes) = self.module_kind;
            let zero = std::env::var("RUBBLE_SELF_SPLIT_ZERO").as_deref() == Ok("1");
            SelfSplit::new(&self.device, &self.bufs, aero, lanes, zero)
        })
    }

    /// A solo frame of `steps` steps with self-collision split: per step its step
    /// dispatch (a workgroup a vehicle), its walk dispatch (indirect: the vehicles that
    /// build their pair list) and its pairs dispatch (indirect: the listed pairs of the
    /// vehicles that reuse it), then the last
    /// step's rest and the frame's end (step dispatch `steps`).
    fn dispatch_self_split(&self, pass: &mut wgpu::ComputePass, steps: u32) {
        for d in self_split_schedule(self.planner.vehicles(), steps) {
            match d {
                SelfDispatch::Walk { .. } if self.self_split_skip & 1 != 0 => continue,
                SelfDispatch::Pairs { .. } if self.self_split_skip & 2 != 0 => continue,
                _ => self.dispatch_self_one(pass, d),
            }
        }
    }

    /// Measurement only (time; the results are then meaningless): leave the self
    /// split's walk dispatches (bit 0) and/or pairs dispatches (bit 1) out of its frames.
    #[doc(hidden)]
    pub fn set_self_split_skip(&mut self, mask: u32) {
        self.self_split_skip = mask;
    }

    fn dispatch_self_one(&self, pass: &mut wgpu::ComputePass, d: SelfDispatch) {
        let p = self.self_split_pipes();
        let at = |k: u32| [k * PARAM_STRIDE as u32];
        match d {
            SelfDispatch::Step { copy, groups } => {
                pass.set_pipeline(if self.variant == KernelVariant::Ablate { &p.ablate_pipes()[0] } else { &p.solo_pipes()[0] });
                pass.set_bind_group(0, &p.step_bind, &at(copy));
                pass.dispatch_workgroups(groups, 1, 1);
            }
            SelfDispatch::List { copy, groups } => {
                pass.set_pipeline(p.list_pipe());
                pass.set_bind_group(0, &p.step_bind, &at(copy));
                pass.dispatch_workgroups(groups, 1, 1);
            }
            SelfDispatch::Walk { copy, offset } | SelfDispatch::Pairs { copy, offset } => {
                let walk = matches!(d, SelfDispatch::Walk { .. });
                pass.set_pipeline(match (self.variant == KernelVariant::Ablate, walk) {
                    (true, true) => &p.ablate_pipes()[1],
                    (true, false) => &p.ablate_pipes()[2],
                    (false, true) => &p.solo_pipes()[1],
                    (false, false) => &p.solo_pipes()[2],
                });
                pass.set_bind_group(0, &p.pairs_bind, &at(copy));
                pass.dispatch_workgroups_indirect(&p.args, offset);
            }
        }
    }

    /// Measurement: one frame of `steps` steps as it would run now (whichever path: the
    /// fleet-wide passes, self split, the split group path, one workgroup a vehicle),
    /// every dispatch in its own timed pass: per label the GPU ms summed over the frame
    /// and its dispatches, in first-seen order, then the span (the first pass's begin to
    /// the last's end; with the labels' sum, the time between passes). The passes add
    /// their own cost: compare the span with an untimed frame's wall time. (A wide frame
    /// with the list kernel: its list, walk and pairs dispatches inside the beams pass.)
    /// `None` without timestamp queries or with more dispatches than a query set holds.
    /// Waits.
    #[doc(hidden)]
    pub fn profile_dispatches(&mut self, steps: u32) -> Option<(Vec<(&'static str, f64, u32)>, f64)> {
        if !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) || self.layout.spans.is_empty() || steps == 0 {
            return None;
        }
        const CAP: u32 = 2048;
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("explicit-beams-dispatch-profile"),
            ty: wgpu::QueryType::Timestamp,
            count: 2 * CAP,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-dispatch-profile"),
            size: u64::from(2 * CAP) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let each = std::mem::replace(&mut self.split_dispatch_timing, true);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let (read, labels) = self.encode(&mut enc, steps, None, Some((&set, 0, CAP)));
        self.split_dispatch_timing = each;
        let n = labels.len() as u32;
        if n > 0 {
            enc.resolve_query_set(&set, 0..2 * n, &resolve, 0);
        }
        self.queue.submit([enc.finish()]);
        self.frame_submitted(read);
        if n == 0 || n == CAP {
            self.wait();
            return None;
        }
        let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
        let raw = self.read(&resolve);
        let ticks: &[u64] = bytemuck::cast_slice(&raw);
        // A pass whose indirect dispatch was empty has no work to stamp (Metal: its
        // stamps land on its neighbours' work, or are stale). The passes run one after
        // another, so a real pass's interval never holds another's: a pass outside the
        // frame (from the first begin, at most a minute) or whose interval holds a
        // shorter pass's is counted as an empty dispatch, time 0. (An empty pass with a
        // short interval inside a real one is not caught: the table's "between passes"
        // row goes negative by it.)
        let t0 = ticks[0];
        let limit = (60_000.0 / period) as u64;
        let at = |i: usize| (ticks[2 * i], ticks[2 * i + 1]);
        let inside = |i: usize| {
            let (b, e) = at(i);
            b >= t0 && e >= b && e - t0 < limit
        };
        let mut order: Vec<usize> = (0..labels.len()).filter(|&i| inside(i) && at(i).1 > at(i).0).collect();
        order.sort_by_key(|&i| at(i).0);
        let mut keep: Vec<bool> = (0..labels.len()).map(inside).collect();
        for (k, &i) in order.iter().enumerate() {
            let (b, e) = at(i);
            // (Sorted by begin: a pass held by i begins at or after b and before e.)
            for &j in order[k + 1..].iter().take_while(|&&j| at(j).0 < e) {
                if at(j).1 <= e && at(j).1 - at(j).0 < e - b {
                    keep[i] = false;
                    break;
                }
            }
        }
        let mut out: Vec<(&'static str, f64, u32)> = Vec::new();
        let (mut end, mut empty) = (t0, 0u32);
        for (i, label) in labels.iter().enumerate() {
            if !keep[i] {
                empty += 1;
                continue;
            }
            end = end.max(ticks[2 * i + 1]);
            let ms = (ticks[2 * i + 1] - ticks[2 * i]) as f64 * period;
            match out.iter_mut().find(|r| r.0 == *label) {
                Some(r) => {
                    r.1 += ms;
                    r.2 += 1;
                }
                None => out.push((label, ms, 1)),
            }
        }
        if empty > 0 {
            out.push(("(empty dispatches, untimed)", 0.0, empty));
        }
        Some((out, (end - t0) as f64 * period))
    }

    /// Measurement: one frame of `steps` steps with self-collision split (when this frame
    /// runs it: [`Self::set_self_split`]), each of its dispatches in its own timed pass
    /// (the passes add their own cost): per kind (step, walk, pairs) the GPU ms summed
    /// over the frame and the dispatches, then ("span", first begin to last end, 1).
    /// `None` when the frame does not run split or the device has no timestamps. Waits.
    #[doc(hidden)]
    pub fn profile_self_split_frame(&mut self, steps: u32) -> Option<Vec<(&'static str, f64, u32)>> {
        if !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) || self.layout.spans.is_empty() || steps == 0 {
            return None;
        }
        self.check_contact_status();
        self.write_wakes();
        let cmds = self.write_params(steps, self.alt_pipes().is_none());
        if self.self_split_frame.get() == 0 {
            return None;
        }
        let sched = self_split_schedule(self.planner.vehicles(), steps);
        let count = 2 * sched.len() as u32;
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("explicit-beams-self-split-profile"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit-beams-self-split-profile"),
            size: u64::from(count) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for cmd in &cmds {
            if let Cmd::Copy { .. } = cmd {
                self.copy(&mut enc, cmd);
                continue;
            }
            for (i, &d) in sched.iter().enumerate() {
                let k = 2 * i as u32;
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("explicit_beams_self_split_profile"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &set,
                        beginning_of_pass_write_index: Some(k),
                        end_of_pass_write_index: Some(k + 1),
                    }),
                });
                self.dispatch_self_one(&mut pass, d);
            }
        }
        enc.resolve_query_set(&set, 0..count, &resolve, 0);
        if self.planner.status_due() {
            self.contact.read_status(&self.device, &mut enc);
        }
        self.queue.submit([enc.finish()]);
        let period = f64::from(self.queue.get_timestamp_period()) * 1e-6;
        let raw = self.read(&resolve);
        let ticks: &[u64] = bytemuck::cast_slice(&raw);
        let mut out = vec![("step", 0.0, 0u32), ("walk", 0.0, 0), ("pairs", 0.0, 0)];
        for (i, d) in sched.iter().enumerate() {
            let k = match d {
                SelfDispatch::Step { .. } | SelfDispatch::List { .. } => 0,
                SelfDispatch::Walk { .. } => 1,
                SelfDispatch::Pairs { .. } => 2,
            };
            out[k].1 += ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period;
            out[k].2 += 1;
        }
        let span = ticks[2 * sched.len() - 1].saturating_sub(ticks[0]) as f64 * period;
        out.push(("span", span, 1));
        Some(out)
    }

    /// The perf flags of a frame on the split group path: vehicle contact's state by owner
    /// vehicle ([`plan::PERF_FLEET_STATE`], contact-groups-design.md §8) only with the
    /// tests in their own dispatches, the candidates' grid, the pair list and the state's
    /// regions (a fleet of two or more); otherwise the groups' slots, the same results.
    fn fleet_flags(&self) -> u32 {
        let w = &self.planner.words;
        let ok = self.group_split_tests
            && self.vv_grid
            && w[plan::W_VV_TRI_PAIRS] != 0
            && w[plan::W_VV_FLEET] != 0
            && w[plan::W_VV_FLEET_NEXT] != 0;
        // (The front as fleet-wide dispatches: with the state by vehicle, once its modules
        // are compiled; a browser host's async compiles.)
        let front = ok && self.planner.perf_flags() & plan::PERF_FLEET_STATE != 0 && {
            let (aero, _, lanes) = self.module_kind;
            ["fleetfront", "fleetplan"].iter().all(|w| crate::pipeline_ready(split_module(w, aero, lanes)))
        };
        let mut flags = self.planner.perf_flags();
        if !ok {
            flags &= !plan::PERF_FLEET_STATE;
        }
        if !front {
            // (The kept candidates are the fleet-wide front's.)
            flags &= !(plan::PERF_FLEET_FRONT | plan::PERF_FLEET_KEEP);
        }
        flags
    }

    /// The split path's small groups' triangle tests in dispatches of their own, across
    /// workgroups (on, the default; `RUBBLE_GROUP_SPLIT_TESTS=0` off: each group's pass
    /// whole in its workgroup). The same results bit for bit.
    pub fn set_group_split_tests(&mut self, on: bool) {
        self.group_split_tests = on;
    }

    /// Measurement: the split path's pipelines with workgroup memory zeroed at each
    /// workgroup's start (the shader translator's default) or not (ours natively).
    #[doc(hidden)]
    pub fn set_group_split_zero(&mut self, zero: bool) {
        if zero != self.group_split_zero {
            self.group_split_zero = zero;
            self.group_split_pipes = std::sync::OnceLock::new();
        }
    }

    /// Measurement only (time): the ablation word alone, for the split path (its modules
    /// read bit 512: the groups' contact passes skipped), keeping the shipped modules.
    #[doc(hidden)]
    pub fn set_split_ablation(&mut self, mask: u32) {
        self.planner.set_ablation(mask);
    }

    /// This frame (after the parameters' plan) runs the split group path.
    fn group_split_on(&self, steps: u32) -> bool {
        self.group_split
            && self.planner.contact_active()
            && !self.planner.sleep_on()
            && self.variant == KernelVariant::Current
            && self.debug_big.is_none()
            && self.debug_solo.is_none()
            && self.debug_groups.is_none()
            && steps > 0
            && 3 * steps + 2 <= plan::PARAM_COPIES
            // The tests' dispatch is 64 workgroups a group (at most one group per two
            // vehicles), the vehicles' one a workgroup each: within WebGPU's 65,535 a
            // dimension. A larger fleet keeps the groups' workgroup (the same results).
            && self.layout.spans.len() * 64 <= 65_535
            // (A browser host's async compiles: the groups' workgroup until its modules
            // are compiled, the same results.)
            && {
                let (aero, _, lanes) = self.module_kind;
                ["member", "group", "tests"].iter().all(|w| crate::pipeline_ready(split_module(w, aero, lanes)))
            }
    }

    fn group_split_pipes(&self) -> &GroupSplit {
        self.group_split_pipes.get_or_init(|| {
            let (aero, _, lanes) = self.module_kind;
            GroupSplit::new(&self.device, &self.bufs, aero, lanes, self.group_split_zero)
        })
    }

    /// The split group path's dispatches of a frame of `steps` steps: the groups' frame
    /// start, then per step the members' steps and the groups' passes (the last with the
    /// groups' frame end). Each from the indirect arguments the plan wrote (none without
    /// a group).
    fn dispatch_group_split(&self, pass: &mut wgpu::ComputePass, steps: u32) {
        for op in self.group_split_ops(steps) {
            match op {
                GroupOp::SelfWalk { .. } if self.self_split_skip & 1 != 0 => continue,
                GroupOp::SelfPairs { .. } if self.self_split_skip & 2 != 0 => continue,
                _ => self.run_group_op(pass, op),
            }
        }
    }

    /// Whether a frame with self-collision split and vehicle contact (`vv`) runs the
    /// members' rest dispatches: with [`plan::PERF_MEMBER_LIST`] a member's step ends at
    /// its self-collision tests (the kernel's contactMemberStep), its hits applied in a
    /// dispatch of their own after the step's walk and pairs (memberRest).
    fn member_rest_on(&self, vv: bool) -> bool {
        vv && self.planner.perf_flags() & plan::PERF_MEMBER_LIST != 0
    }

    /// The split group path's dispatches of a frame of `steps` steps, in order (with
    /// self-collision split, [`Self::self_split_vv_on`], every vehicle's step, the walk
    /// and the pairs in each substep).
    fn group_split_ops(&self, steps: u32) -> Vec<GroupOp> {
        let mut out = vec![GroupOp::Group { copy: steps + 1, tests: false, args: SPLIT_GROUP_ARGS, part: 0 }];
        // With self-collision split: every vehicle's step (copy s, s = 0 ..= steps: the
        // last the solo vehicles' last rest and frame end), each step's walk and pairs
        // over the solo vehicles (explicit_beams_selfsplit.slang), then the groups.
        let vv = self.self_split_vv.get();
        for s in 0..steps + u32::from(vv) {
            if vv {
                out.push(GroupOp::SelfMember(s));
                if s == steps {
                    break;
                }
                let slot = u64::from(if s == 0 { 2 } else { s & 1 });
                out.push(GroupOp::SelfWalk { copy: s, offset: 12 * (3 + slot) });
                out.push(GroupOp::SelfPairs { copy: s, offset: 12 * slot });
                if self.member_rest_on(vv) {
                    out.push(GroupOp::SelfRest { copy: steps + 2 + s });
                }
            } else {
                out.push(GroupOp::Member(s));
            }
            if self.group_split_tests && self.fleet_flags() & plan::PERF_FLEET_FRONT != 0 {
                // The front and the tests as fleet-wide dispatches (explicit_beams_fleet.slang),
                // then the groups' levels. The step's three copies are the phases.
                let (c0, ct, cl) = (s, 2 * steps + 2 + s, steps + 2 + s);
                // (A group past VV_PART members: its whole pass in its own workgroup,
                // no workgroups without one.)
                out.push(GroupOp::Group { copy: ct, tests: false, args: FLEET_LARGER_ARGS, part: 5 });
                out.push(GroupOp::Fleet { copy: c0, kernel: 0, args: FLEET_SCAN_ARGS, from: 0, part: 0 });
                out.push(GroupOp::Fleet { copy: c0, kernel: 1, args: FLEET_FIRST_ARGS, from: 0, part: 1 });
                out.push(GroupOp::Fleet { copy: ct, kernel: 0, args: FLEET_ELEMENT_ARGS, from: 1, part: 2 });
                out.push(GroupOp::Fleet { copy: c0, kernel: 2, args: FLEET_GRID_ARGS, from: 1, part: 3 });
                out.push(GroupOp::Fleet { copy: cl, kernel: 0, args: FLEET_SORT_ARGS, from: 1, part: 4 });
                out.push(GroupOp::Fleet { copy: ct, kernel: 1, args: FLEET_ONE_ARGS, from: 1, part: 5 });
                out.push(GroupOp::Fleet { copy: ct, kernel: 2, args: FLEET_WALK_ARGS, from: 2, part: 6 });
                out.push(GroupOp::Fleet { copy: cl, kernel: 1, args: FLEET_ONE_ARGS, from: 1, part: 7 });
                out.push(GroupOp::Fleet { copy: cl, kernel: 2, args: FLEET_PAIRS_ARGS, from: 2, part: 8 });
                out.push(GroupOp::Group { copy: cl, tests: false, args: SPLIT_GROUP_ARGS, part: 4 });
            } else if self.group_split_tests {
                // The front, the tests across workgroups (the walk, then the listed
                // pairs), then the levels.
                out.push(GroupOp::Group { copy: 2 * steps + 2 + s, tests: false, args: SPLIT_GROUP_ARGS, part: 1 });
                out.push(GroupOp::Group { copy: 2 * steps + 2 + s, tests: true, args: SPLIT_TESTS_ARGS, part: 2 });
                out.push(GroupOp::Group { copy: steps + 2 + s, tests: true, args: SPLIT_TESTS_ARGS, part: 3 });
                out.push(GroupOp::Group { copy: steps + 2 + s, tests: false, args: SPLIT_GROUP_ARGS, part: 4 });
            } else {
                out.push(GroupOp::Group { copy: s, tests: false, args: SPLIT_GROUP_ARGS, part: 5 });
            }
        }
        out
    }

    /// One of [`Self::group_split_ops`]: its pipeline, bind group and indirect dispatch.
    fn run_group_op(&self, pass: &mut wgpu::ComputePass, op: GroupOp) {
        let g = self.group_split_pipes();
        let args = &self.contact.args;
        let at = |k: u32| [k * PARAM_STRIDE as u32];
        match op {
            GroupOp::Group { copy, tests, args: off, .. } => {
                pass.set_pipeline(if tests { &g.tests } else { &g.group });
                pass.set_bind_group(0, if tests { &g.bind } else { &g.bind_args }, &at(copy));
                pass.dispatch_workgroups_indirect(args, off);
            }
            GroupOp::Fleet { copy, kernel, args: off, from, .. } => {
                // (The plan kernel writes the arguments after the ones it runs from: the
                // first buffer when run from the contact plan's, the second from the first.)
                pass.set_pipeline(match kernel {
                    0 => &g.fleet()[0],
                    1 => &g.fleet()[1],
                    _ => &g.tests,
                });
                let bind = match (kernel, from) {
                    (1, 0) => &g.bind_first,
                    (1, _) => &g.bind_second,
                    _ => &g.bind,
                };
                pass.set_bind_group(0, bind, &at(copy));
                pass.dispatch_workgroups_indirect(if from == 0 { args } else { &g.fleet_args[from as usize - 1] }, off);
            }
            GroupOp::Member(s) => {
                pass.set_pipeline(g.member());
                pass.set_bind_group(0, &g.bind, &at(s));
                pass.dispatch_workgroups_indirect(args, SPLIT_MEMBER_ARGS);
            }
            GroupOp::SelfMember(s) => {
                let p = self.self_split_pipes();
                pass.set_pipeline(&p.vv_pipes()[0]);
                pass.set_bind_group(0, &p.step_bind, &at(s));
                pass.dispatch_workgroups_indirect(args, SPLIT_ALL_ARGS);
            }
            GroupOp::SelfWalk { copy, offset } | GroupOp::SelfPairs { copy, offset } => {
                let p = self.self_split_pipes();
                pass.set_pipeline(&p.vv_pipes()[if matches!(op, GroupOp::SelfWalk { .. }) { 1 } else { 2 }]);
                pass.set_bind_group(0, &p.pairs_bind, &at(copy));
                pass.dispatch_workgroups_indirect(&p.args, offset);
            }
            GroupOp::SelfRest { copy } => {
                // The vehicles' module again, a workgroup a vehicle when the plan formed
                // a group (none otherwise): a solo vehicle's returns at once.
                let p = self.self_split_pipes();
                pass.set_pipeline(&p.vv_pipes()[0]);
                pass.set_bind_group(0, &p.step_bind, &at(copy));
                pass.dispatch_workgroups_indirect(args, SPLIT_MEMBER_ARGS);
            }
        }
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
            let (pipeline, group) = self.frame_pipe();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
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
        // Vehicle sleep: a grabbed vehicle is kept awake (and one let go wakes).
        if let Some(v) = self.sleep_grab.take() {
            self.wake(v);
        }
        self.sleep_grab = grab.map(|(v, _)| v);
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
        // Vehicle sleep: a changed input (brake released, steering, throttle) wakes it.
        if self.sleep_inputs.get(vehicle).copied().flatten() != Some(input) {
            self.wake(vehicle);
        }
        if let Some(i) = self.sleep_inputs.get_mut(vehicle) {
            *i = Some(input);
        }
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

    /// Set a vehicle's parking brake input (0 released to 1 applied, as
    /// `ExplicitBeams::parking_brake`; jbeam-semantics-spec.md §25.1).
    pub fn set_parking_brake(&self, vehicle: usize, input: f32) {
        let Some(&(off, count)) = self.layout.brake_spans.get(vehicle) else {
            return;
        };
        let u = input.min(1.0);
        for k in 0..count {
            let rec = u64::from(self.layout.brake_base + 2 * (off + k) + 1);
            self.queue
                .write_buffer(&self.bufs[1], rec * 16 + 8, bytemuck::bytes_of(&u.to_bits()));
        }
    }

    /// Vehicle `vehicle`'s anti-clip box padding by each contacter's own travel
    /// (`ExplicitBeams::own_travel_reach`, its flags word's FLAG_OWN_TRAVEL) from the
    /// next frame on. The upload takes it from the vehicle; this changes it in place.
    pub fn set_own_travel_reach(&mut self, vehicle: usize, on: bool) {
        if vehicle >= self.layout.vehicles() {
            return;
        }
        let at = (vehicle * 4 + 3) * 4;
        let words: Vec<u32> = bytemuck::cast_slice(&self.read(&self.bufs[1])).to_vec();
        let f = if on {
            words[at] | plan::FLAG_OWN_TRAVEL
        } else {
            words[at] & !plan::FLAG_OWN_TRAVEL
        };
        self.queue.write_buffer(&self.bufs[1], (at * 4) as u64, bytemuck::bytes_of(&f));
        // The wide passes' copy of the static records (the first vehicles').
        if let Some(w) = &self.wide {
            if ((at + 1) * 4) as u64 <= w.table.size() {
                self.queue.write_buffer(&w.table, (at * 4) as u64, bytemuck::bytes_of(&f));
            }
        }
    }

    /// Set one node's velocity (vehicle index, node index within it).
    pub fn set_velocity(&self, vehicle: usize, node: u32, v: Vec3) {
        let offset = self.layout.velocity_offset(vehicle, node);
        self.queue
            .write_buffer(&self.nodes, offset, bytemuck::cast_slice(&v.to_array()));
        self.wake(vehicle);
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
                    // No zeroing of workgroup memory natively (naga's MSL writer zeroes it
                    // from one thread on every dispatch: 70 us each, 7 ms a frame). Every
                    // workgroup value a wide pass reads it wrote earlier in the same
                    // dispatch: the coupling's header and table (coupleBegin, cplGsClear on
                    // lane 0 before a barrier: wide_nodes, wide_aux), the hit flags (lane 0
                    // before a barrier: wide_hits), the drive's word (driveBegin); wide_tris
                    // only writes its two. vehicle-perf-first-principles.md §8.8.
                    // RUBBLE_WIDE_ZERO=1 zeroes (a check: the same results).
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: std::env::var("RUBBLE_WIDE_ZERO").as_deref() == Ok("1"),
                        ..Default::default()
                    },
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
    for (n, w) in crate::EXPLICIT_BEAMS_NOSURF.iter().chain(crate::EXPLICIT_BEAMS_NOCPL.iter()).chain(crate::EXPLICIT_BEAMS_CONTACT_BIG.iter()) {
        let entry = if n.contains("contact_big") { plan::CONTACT_ENTRY } else { plan::FRAME_ENTRY };
        out.push((n.to_string(), entry, u(w)));
    }
    out.push((plan::SLEEP_LIST.into(), plan::SLEEP_LIST, u(crate::EXPLICIT_BEAMS_SLEEP_LIST_WGSL)));
    for (n, w) in crate::EXPLICIT_BEAMS_SURF_SPLIT.iter() {
        let entry = plan::SURF_PASSES.iter().find(|(m, _)| m == n).map_or(plan::FRAME_ENTRY, |(_, e)| *e);
        out.push((n.to_string(), entry, u(w)));
    }
    for (n, w) in crate::EXPLICIT_BEAMS_SPLIT_GROUPS.iter() {
        let entry = if n.contains("member") {
            SPLIT_MEMBER_ENTRY
        } else if n.contains("tests") {
            SPLIT_TESTS_ENTRY
        } else {
            SPLIT_GROUP_ENTRY
        };
        out.push((n.to_string(), entry, u(w)));
    }
    for (n, w) in crate::EXPLICIT_BEAMS_SELF_SPLIT.iter() {
        out.push((n.to_string(), self_split_entry(n), u(w)));
    }
    out
}

/// A self-collision split module's entry point (`explicit_beams_selfsplit_{step,walk,pairs}*`).
fn self_split_entry(name: &str) -> &'static str {
    if name.contains("_member") {
        SPLIT_MEMBER_ENTRY
    } else if name.contains("_walk") {
        "explicit_beams_self_walk"
    } else if name.contains("_pairs") {
        "explicit_beams_self_pairs"
    } else {
        "explicit_beams_self_step"
    }
}

/// What a self-collision split frame dispatches ([`ExplicitBeamsGpu::set_self_split`]), in
/// order: per step its step dispatch (`nv` workgroups, binding the indirect arguments as
/// storage), its walk and its pairs dispatch (indirect from the arguments, which they do
/// not bind), then the last step's rest and the frame's end.
/// A dispatch of the split group path ([`ExplicitBeamsGpu::group_split_ops`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupOp {
    /// A groups' dispatch (`tests`: the tests across workgroups) of parameter copy `copy`,
    /// its size at byte `args` of the plan's arguments.
    /// `part` names it in timings: 0 the frame's start, 1 a step's front (candidates,
    /// grids, lists), 2 the tests' walk, 3 the listed pairs' tests, 4 the levels (the
    /// hits applied; the last with the frame's end), 5 a whole pass in the group's workgroup.
    Group { copy: u32, tests: bool, args: u64, part: u8 },
    /// A dispatch of the contact pass's front over the fleet ([`plan::PERF_FLEET_FRONT`]):
    /// `kernel` 0 the thread kernels' module, 1 the plan module, 2 the tests' module;
    /// parameter copy `copy`; its size at byte `args` of the contact plan's arguments
    /// (`from` 0) or of the fleet's first or second (1, 2). `part` names it: 0 the pairs'
    /// scans, 1 the owners' parts, 2 the triangles and contact points, 3 the grids'
    /// offsets, 4 the sorted copy, 5 the listed triangles' prefix, 6 the walk, 7 the
    /// pairs' prefix, 8 the listed pairs.
    Fleet { copy: u32, kernel: u8, args: u64, from: u8, part: u8 },
    /// The vehicles' step s (a group's members; with no group, none).
    Member(u32),
    /// Self-collision split: every vehicle's step s.
    SelfMember(u32),
    /// Self-collision split: step `copy`'s walk and pairs over the solo vehicles.
    SelfWalk { copy: u32, offset: u64 },
    SelfPairs { copy: u32, offset: u64 },
    /// Self-collision split with the members' kept pair lists
    /// ([`plan::PERF_MEMBER_LIST`]): the members' rest of a step (their self-collision
    /// hits applied) after its walk and pairs, before the groups' pass; parameter copy
    /// `copy` (the step with the kernel's SPLIT_LEVELS). No workgroups without a group.
    SelfRest { copy: u32 },
}

impl GroupOp {
    fn label(self) -> &'static str {
        match self {
            GroupOp::Group { part: 0, .. } => "explicit split begin",
            GroupOp::Group { part: 1, .. } => "explicit split front",
            GroupOp::Group { part: 2, .. } => "explicit split tests walk",
            GroupOp::Group { part: 3, .. } => "explicit split tests pairs",
            GroupOp::Group { part: 4, .. } => "explicit split levels",
            GroupOp::Group { .. } => "explicit split pass",
            GroupOp::Fleet { part: 0, .. } => "explicit fleet scans",
            GroupOp::Fleet { part: 1, .. } => "explicit fleet owners' parts",
            GroupOp::Fleet { part: 2, .. } => "explicit fleet triangles, points",
            GroupOp::Fleet { part: 3, .. } => "explicit fleet grids",
            GroupOp::Fleet { part: 4, .. } => "explicit fleet sorted copy",
            GroupOp::Fleet { part: 5, .. } => "explicit fleet listed prefix",
            GroupOp::Fleet { part: 6, .. } => "explicit fleet walk",
            GroupOp::Fleet { part: 7, .. } => "explicit fleet pairs' prefix",
            GroupOp::Fleet { .. } => "explicit fleet pairs",
            GroupOp::Member(_) => "explicit split member",
            GroupOp::SelfMember(_) => "explicit self step",
            GroupOp::SelfWalk { .. } => "explicit self walk",
            GroupOp::SelfPairs { .. } => "explicit self pairs",
            GroupOp::SelfRest { .. } => "explicit self members' hits",
        }
    }
}

impl SelfDispatch {
    fn label(self) -> &'static str {
        match self {
            SelfDispatch::Step { .. } => "explicit self step",
            SelfDispatch::List { .. } => "explicit wide list",
            SelfDispatch::Walk { .. } => "explicit self walk",
            SelfDispatch::Pairs { .. } => "explicit self pairs",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelfDispatch {
    /// The step dispatch of parameter copy `copy` over `groups` workgroups.
    Step { copy: u32, groups: u32 },
    /// The wide passes' list dispatch (explicit_beams_wide_list) of copy `copy`.
    List { copy: u32, groups: u32 },
    /// The walk dispatch of copy `copy`, its size at byte `offset` of the arguments.
    Walk { copy: u32, offset: u64 },
    /// The pairs dispatch of copy `copy`, its size at byte `offset` of the arguments.
    Pairs { copy: u32, offset: u64 },
}

/// Bytes of the self-collision split's indirect arguments (two kinds, three slots of
/// three words: explicit_beams_selfsplit.slang selfArgsAt, selfWalkArgsAt).
const SELF_ARGS_BYTES: u64 = 128;

impl SelfDispatch {
    /// Whether the dispatch binds the arguments buffer as storage (the step dispatch
    /// alone: a buffer cannot be storage and indirect in one dispatch).
    fn binds_args(self) -> bool {
        matches!(self, SelfDispatch::Step { .. } | SelfDispatch::List { .. })
    }
}

/// A self-collision split frame's dispatches for `nv` vehicles and `steps` steps.
fn self_split_schedule(nv: u32, steps: u32) -> Vec<SelfDispatch> {
    let mut out = Vec::new();
    for s in 0..=steps {
        out.push(SelfDispatch::Step { copy: s, groups: nv });
        if s == steps {
            break;
        }
        // Step 0's slot 2, then odd and even steps' (selfArgsAt); the walk's after the
        // pairs' three.
        let slot = u64::from(if s == 0 { 2 } else { s & 1 });
        out.push(SelfDispatch::Walk { copy: s, offset: 12 * (3 + slot) });
        out.push(SelfDispatch::Pairs { copy: s, offset: 12 * slot });
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

#[cfg(test)]
mod self_split_tests {
    use super::*;

    /// The split frame's dispatches for 1, 4 and 64 vehicles, 1 to 34 steps (and the
    /// most a parameter buffer holds): a step dispatch for each parameter copy 0 ..= steps
    /// over every vehicle; between them the walk and pairs dispatches of that step, each
    /// sized from three words inside the arguments buffer, a slot its own for each of the
    /// two dispatches of consecutive steps; no indirect dispatch binds the arguments (wgpu
    /// rejects a buffer used as storage and indirect in one dispatch).
    #[test]
    fn self_split_schedule_is_bounded_and_never_binds_its_arguments_indirectly() {
        for nv in [1u32, 4, 64] {
            for steps in [1u32, 2, 3, 34, plan::PARAM_COPIES - 1] {
                let d = self_split_schedule(nv, steps);
                assert_eq!(d.len() as u32, 3 * steps + 1, "nv {nv}, steps {steps}");
                let mut last: Option<(u64, u64)> = None;
                for (k, x) in d.iter().enumerate() {
                    match *x {
                        SelfDispatch::Step { copy, groups } => {
                            assert_eq!(copy as usize, k / 3);
                            assert_eq!(groups, nv);
                            assert!(copy <= steps && copy < plan::PARAM_COPIES);
                            assert!(x.binds_args());
                        }
                        SelfDispatch::Walk { copy, offset } | SelfDispatch::Pairs { copy, offset } => {
                            assert!(!x.binds_args(), "an indirect dispatch binds its arguments");
                            assert!(copy < steps);
                            assert_eq!(offset % 4, 0);
                            assert!(offset + 12 <= SELF_ARGS_BYTES, "arguments past the buffer at {offset}");
                        }
                    }
                }
                // The two kinds never share a slot, and consecutive steps never share one
                // (a step dispatch clears the next step's while the last is still read).
                for s in 0..steps as usize {
                    let (SelfDispatch::Walk { offset: w, .. }, SelfDispatch::Pairs { offset: p, .. }) =
                        (d[3 * s + 1], d[3 * s + 2])
                    else {
                        panic!("step {s}: not a walk then a pairs dispatch");
                    };
                    assert_ne!(w, p);
                    if let Some((lw, lp)) = last {
                        assert!(lw != w && lp != p, "steps {} and {s} share a slot", s - 1);
                    }
                    last = Some((w, p));
                }
                assert!(matches!(d.last(), Some(SelfDispatch::Step { copy, .. }) if *copy == steps));
            }
        }
    }

    /// The twelve split modules are shipped with their entry points.
    #[test]
    fn self_split_modules_are_shipped() {
        let shipped = explicit_beams_shipped_wgsl();
        for (n, _) in crate::EXPLICIT_BEAMS_SELF_SPLIT.iter() {
            let (_, e, w) = shipped.iter().find(|(s, _, _)| s == n).unwrap_or_else(|| panic!("{n} not shipped"));
            assert!(w.contains(&format!("fn {e}(")), "{n}: no entry {e}");
        }
    }
}

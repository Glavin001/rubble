//! One explicit-beams command plan for every host (`slang/kernels/explicit_beams.slang`).
//!
//! The explicit node–beam solver runs the same Slang kernels on WebGPU (native wgpu
//! and the browser), native Metal and CUDA. This module is everything about it that
//! does not depend on the GPU API, so each host is a thin replay:
//!
//! * [`MODULES`]: the kernel modules (name, Slang entry, `-D` defines). Every host's
//!   build compiles exactly these from the one Slang source (WGSL, MSL, CUDA).
//! * [`ExplicitUpload::new`]: the buffers' contents (bindings 0..10, [`BIND_COUNT`])
//!   and the [`ExplicitLayout`] that says where each record lives, from spawned
//!   [`ExplicitBeams`].
//! * [`FramePlanner`]: per frame, the parameter copies ([`FramePlanner::write_params`])
//!   and the ordered command list ([`FramePlanner::commands`]): dispatches, indirect
//!   dispatches and buffer copies, with the vehicle-contact plan's and the wide
//!   passes' scheduling decided here.
//! * [`ExplicitLayout`]'s readback decoders (positions, beams, ...) and the host
//!   writes' offsets (a node's velocity, the grab, the coupling switch).
//!
//! Hosts own only buffers, pipelines, submission and readback
//! (`engine-webgpu::ExplicitBeamsGpu`, `engine-metal::MetalExplicitBeams`,
//! `engine-cuda::CudaExplicitBeams`).

pub mod coupling;
pub mod drive;

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitKind, StaticShape};
use glam::Vec3;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// A vehicle's place in the shared buffers.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub node_off: u32,
    pub nodes: u32,
    pub beam_off: u32,
    pub beams: u32,
    /// First slide node and rail segment (global) and the slide node count.
    pub slide_off: u32,
    pub slides: u32,
    pub seg_off: u32,
    pub tris: u32,
}

/// Room for rigid-body proxies per vehicle (`ExplicitBeamsGpu::with_coupling`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CouplingCapacity {
    /// Proxy bodies, shapes (at most 1024, at least the bodies) and hull planes (at most
    /// 65535) per vehicle.
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
        assert!(
            cap.bodies <= cap.shapes && cap.shapes <= 1024 && cap.planes < 1 << 16,
            "explicit coupling: at most 1024 shapes (and no more bodies) and 65535 planes per vehicle"
        );
        Self {
            cap,
            vehicles,
            misc_base,
            acc_base,
        }
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
        self.vehicle(v)
            + 1
            + self.cap.bodies * Self::BODY_RECS
            + self.cap.shapes * Self::SHAPE_RECS
            + i
    }
    pub fn acc(&self, v: u32) -> u32 {
        self.acc_base + v * self.acc_per_vehicle()
    }
}

// --- Bindings (the kernel's `[[vk::binding(n, 0)]]`, one numbering for every host).

/// The parameters (uniform): [`PARAM_COPIES`] copies of [`PARAM_WORDS`] words at
/// [`PARAM_STRIDE`] bytes.
pub const BIND_PARAMS: u32 = 0;
/// Static records (read only): vehicles, node info, beams, triangles, boxes, aero, ...
pub const BIND_STATIC: u32 = 1;
/// Lists (read only): incidence, candidates, contact points, slots, ...
pub const BIND_LISTS: u32 = 2;
/// Node state: 3 `vec4<f32>` per node (position and mass, velocity and friction, force).
pub const BIND_NODES: u32 = 3;
/// Beam state: 2 `vec4<f32>` per beam, then the kernel's scratch and aero state.
pub const BIND_BEAMS: u32 = 4;
/// Accumulators (atomic i32): node forces, counters, the status words.
pub const BIND_ACC: u32 = 5;
/// Group slots (atomic u32), two parities, then the deform group words.
pub const BIND_GROUPS: u32 = 6;
/// Per-vehicle records (origin, steps), triangle throttles, contact records, ...
pub const BIND_MISC: u32 = 7;
/// Grid ranks, wide-pass scratch, trigger inputs, contact lists, hit lists.
pub const BIND_NEXT: u32 = 8;
/// The wide passes' vehicle table (uniform, [`TABLE_BYTES`]).
pub const BIND_TABLE: u32 = 9;
/// The contact plan's indirect arguments ([`ARGS_BYTES`]): solo vehicles at byte 0,
/// contact groups at byte 16.
pub const BIND_ARGS: u32 = 10;
/// Bindings a host allocates (0..BIND_COUNT; 11 and 12 are test-only variants').
pub const BIND_COUNT: usize = 11;
/// The state bindings a frame writes (snapshots): 3..=8.
pub const STATE_BINDINGS: std::ops::RangeInclusive<u32> = 3..=8;

/// Parameter copies for the wide passes: one per step of a frame (its index in the
/// step word) plus the frame's end, at the uniform offset alignment.
pub const PARAM_STRIDE: u64 = 512;
pub const PARAM_COPIES: u32 = 129;
/// Words of the kernel's Params (85 used).
pub const PARAM_WORDS: usize = 88;
/// Two-body mass splitting's words per node (the kernel's 1 + VV_PART).
pub const VV_SPLIT_SLOTS: usize = 5;
/// Bytes of the indirect arguments buffer.
pub const ARGS_BYTES: u64 = 32;
/// Params word: the phases a measurement build skips (`EB_ABLATE`).
pub const W_ABLATE: usize = 63;
/// Params word: the steps the contact plan's frame bounds cover.
const W_VV_PLAN_STEPS: usize = 60;
/// Steps a contact plan covers at least (frames shorter than this share a plan):
/// fewer plan passes, frame bounds a little wider.
const VV_PLAN_STEPS: u32 = 34;
/// Params words of vehicle contact: on, the frame bounds margin.
const W_VV_ON: usize = 39;
const W_VV_MARGIN: usize = 57;
/// The kernel's VV_RECS, VV_SLOTS and the counters' words (VV_CTL_WORDS).
pub const VV_RECS: usize = 6;
const VV_SLOTS: usize = 8;
const VV_CTL_WORDS: u32 = 8;
/// Frame bounds margin past the travel of a vehicle's fastest node (m): see
/// [`FramePlanner::set_contact_margin`].
pub const DEFAULT_CONTACT_MARGIN: f32 = 1.0;
/// The kernel's KEYS (trigger inputs per vehicle and step parity), NONE and node
/// info flags shift of the slide slots.
pub const KEYS: usize = engine_softbody::explicit_triggers::KEYS;
const NONE: u32 = u32::MAX;
const NODE_SLOT_SHIFT: u32 = 8;
/// Per vehicle the kernel's self-collision hit list (HIT_CAP records of HIT_WORDS
/// words) and pending-beam list (PEND_CAP) in gNext.
const SELF_HIT_WORDS: usize = 256 * 24 + 64;
/// The kernel's WIDE_MAX_VEHICLES: more vehicles run one workgroup each.
pub const WIDE_MAX_VEHICLES: usize = 256;
/// Bytes of the wide passes' vehicle table (the kernel's VehicleTable).
pub const TABLE_BYTES: u64 = (WIDE_MAX_VEHICLES * 5 * 16) as u64;
/// Fleets up to this size find their overlapping pairs in the grouping pass (one
/// dispatch fewer per frame).
pub const FUSED_PAIRS_MAX: usize = 2048;
/// Frames between status reads.
pub const STATUS_EVERY: u64 = 16;
/// Beams above which a vehicle runs on the 512-lane module, and wide passes are the
/// default.
pub const WIDE_BEAMS: usize = 2048;
/// The kernel's GRID_BUCKETS and GRID_SUBS.
const WIDE_GRID_BUCKETS: usize = 1024;
const WIDE_SUBS: u32 = 8;
/// The kernel's WIDE_SLOTS (grid bucket slots of the wide passes).
const WIDE_SLOTS: usize = 16;

// --- Kernel modules.

/// One compiled kernel module: its name (the WGSL file / MSL / CUDA module), the Slang
/// entry point and the `-D` defines. Every host compiles [`MODULES`] from
/// `slang/kernels/explicit_beams.slang`.
#[derive(Clone, Copy, Debug)]
pub struct Module {
    pub name: &'static str,
    pub entry: &'static str,
    pub defines: &'static [(&'static str, &'static str)],
}

/// The frame entry point (frame, solo and contact modules).
pub const FRAME_ENTRY: &str = "explicit_beams_frame";
/// The wide passes, in dispatch-kind order ([`Kernel::Wide`]).
pub const WIDE_PASSES: [&str; 10] = [
    "explicit_beams_wide_begin",
    "explicit_beams_wide_rebase",
    "explicit_beams_wide_nodes",
    "explicit_beams_wide_beams",
    "explicit_beams_wide_tris",
    "explicit_beams_wide_end",
    "explicit_beams_wide_close",
    "explicit_beams_wide_hits",
    "explicit_beams_wide_centred_beams",
    "explicit_beams_wide_centred_nodes",
];
/// The vehicle-contact plan passes ([`Kernel::Vv`]): bounds, pairs, group.
pub const VV_PASSES: [&str; 3] = [
    "explicit_beams_vv_bounds",
    "explicit_beams_vv_pairs",
    "explicit_beams_vv_group",
];
const WIDE_BEGIN: usize = 0;
const WIDE_REBASE: usize = 1;
const WIDE_NODES: usize = 2;
const WIDE_BEAMS_PASS: usize = 3;
const WIDE_TRIS: usize = 4;
const WIDE_END: usize = 5;
const WIDE_CLOSE: usize = 6;
const WIDE_HITS: usize = 7;
// Centred damping (jbeam-semantics-spec.md §39): after a step's nodes, when a
// vehicle has it.
const WIDE_CENTRED_BEAMS: usize = 8;
const WIDE_CENTRED_NODES: usize = 9;
/// The Params word of the centred damping's Δv region (nonzero: some vehicle has it).
const W_CENTRED_DV: usize = 84;

const W: (&str, &str) = ("ENGINE_WGSL", "1");
const L256: (&str, &str) = ("EB_LANES", "256");
const L512: (&str, &str) = ("EB_LANES", "512");
const AERO: (&str, &str) = ("EB_AERO", "1");
const SOLO: (&str, &str) = ("EB_VV_SOLO", "1");
const GROUPS: (&str, &str) = ("EB_VV_GROUPS", "1");

const fn frame(name: &'static str, defines: &'static [(&'static str, &'static str)]) -> Module {
    Module {
        name,
        entry: FRAME_ENTRY,
        defines,
    }
}
const fn pass(name: &'static str) -> Module {
    Module {
        name,
        entry: name,
        defines: &[W],
    }
}

/// Every shipped explicit-beams module (the defines are those engine-webgpu's
/// build.rs passes for the WGSL of the same name).
pub const MODULES: &[Module] = &[
    frame("explicit_beams", &[W]),
    frame("explicit_beams_wide", &[W, L512]),
    frame("explicit_beams_aero", &[W, L256, AERO]),
    frame("explicit_beams_aero_wide", &[W, L512, AERO]),
    frame("explicit_beams_solo", &[W, L256, SOLO]),
    frame("explicit_beams_solo_wide", &[W, L512, SOLO]),
    frame("explicit_beams_solo_aero", &[W, L256, AERO, SOLO]),
    frame("explicit_beams_solo_aero_wide", &[W, L512, AERO, SOLO]),
    frame("explicit_beams_contact", &[W, L256, SOLO, GROUPS]),
    frame("explicit_beams_contact_wide", &[W, L512, SOLO, GROUPS]),
    frame(
        "explicit_beams_contact_aero",
        &[W, L256, AERO, SOLO, GROUPS],
    ),
    frame(
        "explicit_beams_contact_aero_wide",
        &[W, L512, AERO, SOLO, GROUPS],
    ),
    pass(VV_PASSES[0]),
    pass(VV_PASSES[1]),
    pass(VV_PASSES[2]),
    pass(WIDE_PASSES[0]),
    pass(WIDE_PASSES[1]),
    pass(WIDE_PASSES[2]),
    pass(WIDE_PASSES[3]),
    pass(WIDE_PASSES[4]),
    pass(WIDE_PASSES[5]),
    pass(WIDE_PASSES[6]),
    pass(WIDE_PASSES[7]),
    pass(WIDE_PASSES[8]),
    pass(WIDE_PASSES[9]),
];

/// A dispatch's kernel. The frame kernels' module depends on the fleet
/// ([`ExplicitLayout::aero`], [`ExplicitLayout::lanes`]): [`Kernel::module`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    /// One workgroup per vehicle runs the whole frame (no vehicle contact).
    Frame,
    /// Contact active: the solo vehicles' frame (`EB_VV_SOLO`).
    Solo,
    /// Contact active: the frame with the contact groups past the vehicles
    /// (`EB_VV_GROUPS`).
    Contact,
    /// A contact plan pass ([`VV_PASSES`] index).
    Vv(usize),
    /// A wide pass ([`WIDE_PASSES`] index).
    Wide(usize),
}

impl Kernel {
    /// The module this kernel runs from, for a fleet with or without aero at `lanes`.
    pub fn module(self, aero: bool, lanes: u32) -> &'static str {
        let pick = |plain: [&'static str; 4]| match (aero, lanes) {
            (false, 512) => plain[1],
            (false, _) => plain[0],
            (true, 512) => plain[3],
            (true, _) => plain[2],
        };
        match self {
            Kernel::Frame => pick([
                "explicit_beams",
                "explicit_beams_wide",
                "explicit_beams_aero",
                "explicit_beams_aero_wide",
            ]),
            Kernel::Solo => pick([
                "explicit_beams_solo",
                "explicit_beams_solo_wide",
                "explicit_beams_solo_aero",
                "explicit_beams_solo_aero_wide",
            ]),
            Kernel::Contact => pick([
                "explicit_beams_contact",
                "explicit_beams_contact_wide",
                "explicit_beams_contact_aero",
                "explicit_beams_contact_aero_wide",
            ]),
            Kernel::Vv(i) => VV_PASSES[i],
            Kernel::Wide(i) => WIDE_PASSES[i],
        }
    }

    /// The Slang entry point.
    pub fn entry(self) -> &'static str {
        match self {
            Kernel::Frame | Kernel::Solo | Kernel::Contact => FRAME_ENTRY,
            Kernel::Vv(i) => VV_PASSES[i],
            Kernel::Wide(i) => WIDE_PASSES[i],
        }
    }

    /// Bindings beyond the state ones (0..=8) the kernel's bind group carries
    /// (the WebGPU layouts; Metal and CUDA bind what the reflection lists).
    pub fn extra_binding(self) -> Option<u32> {
        match self {
            Kernel::Vv(_) => Some(BIND_ARGS),
            Kernel::Wide(_) => Some(BIND_TABLE),
            _ => None,
        }
    }

    /// A profiling label.
    pub fn label(self) -> &'static str {
        match self {
            Kernel::Frame => "frame",
            Kernel::Solo => "solo frame",
            Kernel::Contact => "frame with groups",
            Kernel::Vv(0) => "vv_bounds",
            Kernel::Vv(1) => "vv_pairs",
            Kernel::Vv(_) => "vv_group",
            Kernel::Wide(i) => WIDE_PASSES[i],
        }
    }
}

/// The module of `name` in [`MODULES`].
pub fn module(name: &str) -> Option<&'static Module> {
    MODULES.iter().find(|m| m.name == name)
}

/// One command of a frame, in order. A copy ends the compute pass it follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    /// `groups` workgroups of `kernel` with the parameters at copy `param_copy`
    /// (byte `param_copy * PARAM_STRIDE` of binding 0; 0 for the frame kernels,
    /// which bind the whole buffer).
    Dispatch {
        kernel: Kernel,
        groups: u32,
        param_copy: u32,
    },
    /// `kernel` with the workgroup count at byte `offset` of the indirect arguments
    /// ([`BIND_ARGS`]: x, y, z words).
    Indirect { kernel: Kernel, offset: u64 },
    /// Copy `size` bytes between bindings.
    Copy {
        src: u32,
        src_offset: u64,
        dst: u32,
        dst_offset: u64,
        size: u64,
    },
}

/// The status a host reads every [`STATUS_EVERY`] frames: 4 contact counters
/// (outside, overflow, unconverged, too big) at byte `base + 16` of gAcc, and the
/// loop-cap flag at byte `cap` (the kernel's `EB_LOOP_CAP`, every target).
#[derive(Clone, Copy, Debug)]
pub struct StatusWords {
    pub base: u64,
    pub cap: u64,
}

impl StatusWords {
    /// Byte offset of the 4 contact counters in gAcc.
    pub fn counters(&self) -> u64 {
        self.base + 16
    }
}

/// The contact error the status counters (outside, overflow, unconverged, too big)
/// report.
pub fn contact_error(w: &[i32]) -> Result<(), String> {
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

/// Every buffer's initial contents ([`ExplicitUpload::bytes`]) and the layout.
pub struct ExplicitUpload {
    pub layout: ExplicitLayout,
    statics: Vec<[u32; 4]>,
    lists: Vec<u32>,
    nodes: Vec<[f32; 4]>,
    beams: Vec<[f32; 4]>,
    acc: Vec<i32>,
    groups: Vec<u32>,
    misc: Vec<[u32; 4]>,
    next: Vec<u32>,
    /// The wide passes' vehicle table: the first vehicles' static records.
    table: Vec<[u32; 4]>,
}

/// Where everything lives, and the fleet's module choice.
#[derive(Clone, Debug)]
pub struct ExplicitLayout {
    /// The contact-force log in gAcc (word index; 0: no vehicle has it): per
    /// node (sum, last, peak), then per group slot its contact-break state.
    pub contact_force: u32,
    /// Per vehicle its break groups' (offset, count) in the group slots.
    pub group_spans: Vec<(u32, u32)>,
    /// Per vehicle whether it has the contact-force log.
    pub contact_force_on: Vec<bool>,
    pub spans: Vec<Span>,
    /// Spawn rest lengths of every beam, in buffer order.
    pub rest0: Vec<f32>,
    /// The parameter words at upload.
    pub words: [u32; PARAM_WORDS],
    /// JBeam deform group words (spec §19): gGroups word of the first, and per
    /// vehicle (first, count).
    pub deform_base: u32,
    pub deform_spans: Vec<(u32, u32)>,
    /// gBeams record of the aero state.
    pub aero_state: u32,
    /// gBeams record of the slide node states; gNext word of the trigger inputs.
    pub slide_state: u32,
    pub key_base: u32,
    /// Per vehicle its first wing and count.
    pub wing_spans: Vec<(u32, u32)>,
    /// gStatic record of the parking brakes (2 records each) and per vehicle
    /// its first brake and count (spec §25.1).
    pub brake_base: u32,
    pub brake_spans: Vec<(u32, u32)>,
    /// Nodes, beams, triangles, candidates, group slots over the fleet.
    pub total: [u32; 5],
    /// gMisc record of the rigid coupling's header.
    pub couple_rec: u32,
    pub coupling: Option<CouplingLayout>,
    /// The raycast wheels' records (None: no vehicle has them).
    pub drive: Option<drive::DriveLayout>,
    /// gMisc record of the first triangle's contact throttle.
    pub vv_tri_base: u32,
    /// The fleet runs the aero modules (`EB_AERO`).
    pub aero: bool,
    /// A vehicle has more than [`WIDE_BEAMS`] beams.
    pub big: bool,
    /// Lanes per vehicle workgroup of the frame modules (256 or 512).
    pub lanes: u32,
    /// The wide passes can run this fleet, and are on by default.
    pub wide_ok: bool,
    pub wide_default: bool,
    pub status: StatusWords,
}

impl ExplicitUpload {
    /// Pack `vehicles` (their current state: positions, velocities, forces, beam
    /// state, throttles, origin), with room for rigid proxies when `coupling` is set.
    pub fn new(vehicles: &[ExplicitBeams], coupling: Option<CouplingCapacity>) -> Self {
        let nv = vehicles.len();
        // Anti-clip (jbeam-semantics-spec.md §36): its side memory names a
        // triangle by owner vehicle (10 bits) and index (20 bits).
        let any_anti = vehicles.iter().any(|v| v.anti_clip);
        // Dissipative contact's position steps (§37) on every vehicle of the upload.
        // (Surface contact's position steps too, §38.)
        let any_shift = vehicles.iter().any(|v| v.dissipative_contact || v.surface_contact);
        let any_surface = vehicles.iter().any(|v| v.surface_contact);
        if any_surface {
            assert!(
                coupling.is_none(),
                "explicit vehicles: surface contact (jbeam-semantics-spec.md §38) runs against static boxes and shapes on the GPU; with rigid proxies it is the CPU reference's only (ExplicitBeams::rigid)"
            );
        }
        // Centred damping (jbeam-semantics-spec.md §39): some vehicle has it.
        let any_centred = vehicles.iter().any(|v| v.settings.centred_damping);
        if any_anti {
            assert!(
                nv < engine_softbody::explicit_contact::CLIP_SELF as usize,
                "explicit vehicles: anti-clip takes at most {} vehicles in one upload ({nv})",
                engine_softbody::explicit_contact::CLIP_SELF
            );
            for v in vehicles {
                assert!(
                    v.triangles.len() <= engine_softbody::explicit_contact::CLIP_MAX_TRIANGLES as usize,
                    "explicit vehicles: anti-clip takes at most {} collision triangles per vehicle ({})",
                    engine_softbody::explicit_contact::CLIP_MAX_TRIANGLES,
                    v.triangles.len()
                );
            }
        }
        // Beam records carry the vehicle index in 16 bits.
        assert!(
            nv <= 1 << 16,
            "explicit beams GPU: at most 65536 vehicles, got {nv}"
        );
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
        // Per vehicle its break groups' (offset, count) in the group slots.
        let mut group_spans: Vec<(u32, u32)> = Vec::with_capacity(nv);
        let mut vv_static: Vec<[u32; 4]> = Vec::with_capacity(nv);
        // Per lists position (incidence entries) the entry's far end.
        let mut nbr: Vec<u32> = Vec::new();
        let mut vv_tri: Vec<[u32; 4]> = Vec::with_capacity(total_tris);
        // JBeam Part II (spec §16-§19): extension records, per beam its record + 1,
        // the extension states, the triangle break lists, the deform group words.
        let any_ext = vehicles.iter().any(|v| !v.beam_ext.is_empty());
        let any_tri_break = vehicles.iter().any(|v| !v.tri_breaks.is_empty());
        // Self-collision spawn exclusions (spec §10.1): per triangle (start, count) in
        // the lists, the excluded nodes global.
        let any_tri_excl = vehicles.iter().any(|v| !v.tri_excl.is_empty());
        let mut tri_excl_recs: Vec<[u32; 4]> = Vec::new();
        let mut tri_excl_list: Vec<u32> = Vec::new();
        let mut ext_recs: Vec<[u32; 4]> = Vec::new();
        let mut ext_state: Vec<[f32; 4]> = Vec::new();
        let mut ext_index: Vec<u32> = Vec::new();
        let mut tri_break_recs: Vec<[u32; 4]> = Vec::new();
        let mut deform_spans: Vec<(u32, u32)> = Vec::with_capacity(nv);
        let mut deform_total = 0u32;
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
                .chain(
                    v.beam_ext
                        .iter()
                        .flat_map(|x| x.extra_groups)
                        .map(|g| g.unsigned_abs() + 1),
                )
                .max()
                .filter(|&g| g > 1)
                .unwrap_or(0);
            let group_off = groups.len() as u32;
            groups.extend(std::iter::repeat_n(u32::MAX, group_count as usize));
            group_spans.push((group_off, group_count));
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
                | (!v.shapes.is_empty() as u32) << 6
                | (v.two_body_contact as u32) << 7
                | (v.contact_force.is_some() as u32) << 8
                | (!v.tread.is_empty() as u32) << 9
                | (!v.settings.triangle_throttle as u32) << 10
                | (v.anti_clip as u32) << 11
                | (any_anti as u32) << 12
                | (!v.plastic_strength_drop as u32) << 13
                | (v.exact_friction as u32) << 14
                | (v.dissipative_contact as u32) << 15
                | (any_shift as u32) << 16
                | (v.surface_contact as u32) << 17
                | (v.settings.centred_damping as u32) << 18;
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
            // Extension records, globalised (spec §16).
            let ext_first = (ext_recs.len() / 4) as u32;
            let deform_first = deform_total;
            deform_spans.push((deform_first, v.deform_state.len() as u32));
            deform_total += v.deform_state.len() as u32;
            for x in &v.beam_ext {
                use engine_softbody::explicit_beams::ExplicitKind as K;
                let kind = v.beams[x.beam as usize].kind;
                let mut e1 = x.e1.map(f32::to_bits);
                match kind {
                    K::JbeamLbeam => e1[0] += node_off,
                    K::JbeamPressureTri => {
                        e1[0] += node_off;
                        e1[1] += beam_off;
                    }
                    K::JbeamPressureGroup => e1[0] += beam_off,
                    K::JbeamAeroTri => e1[0] += node_off,
                    _ => {}
                }
                let dg = |g: u32| if g == 0 { 0 } else { deform_first + g };
                ext_recs.push([
                    x.mask,
                    dg(x.deform[0]),
                    dg(x.deform[1]),
                    x.trigger.to_bits(),
                ]);
                ext_recs.push(e1);
                ext_recs.push(x.e2.map(f32::to_bits));
                ext_recs.push([
                    x.ramp_delta.to_bits(),
                    x.ramp_steps,
                    x.extra_groups[0] as u32,
                    x.extra_groups[1] as u32,
                ]);
                ext_state.push([x.ramp_left as f32, x.pressure[0], x.pressure[1], 0.0]);
            }
            if any_ext {
                ext_index.extend((0..v.beams.len()).map(|j| match v.ext_of.get(j) {
                    Some(&x) if x != 0 => ext_first + x,
                    _ => 0,
                }));
            }
            for (j, b) in v.beams.iter().enumerate() {
                let ext = if v.ext_of.get(j).is_some_and(|&x| x != 0) {
                    engine_softbody::explicit_jbeam_ext::TYPE_EXT
                } else {
                    0
                };
                beam_recs.push([
                    node_off + b.a,
                    node_off + b.b,
                    b.kind.code() | (b.ty.code() | ext) << 8 | (vi as u32) << 16,
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
                // Anti-clip only (NONCOLLIDABLE, §36): bit 30.
                if v.tri_anti_only(t) {
                    tri_misc[first_tri + t][2] |= 0x4000_0000;
                }
            }
            // Breakable triangles (spec §18): their four beams, bit 31 of the word.
            if any_tri_break {
                for t in 0..v.triangles.len() {
                    let l = v.tri_breaks.get(t).copied().unwrap_or([u32::MAX; 4]);
                    let g = l.map(|j| if j == u32::MAX { j } else { beam_off + j });
                    if g.iter().any(|&j| j != u32::MAX) {
                        tri_misc[first_tri + t][2] |= 0x8000_0000;
                    }
                    tri_break_recs.push(g);
                }
            }
            if any_tri_excl {
                for t in 0..v.triangles.len() {
                    let l = v.tri_excl.get(t).map_or(&[][..], |l| &l[..]);
                    tri_excl_recs.push([tri_excl_list.len() as u32, l.len() as u32, 0, 0]);
                    tri_excl_list.extend(l.iter().map(|&n| node_off + n));
                }
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
            // Its slots: its boxes, then its static shapes (each a slot).
            box_off += (v.boxes.len() + v.shapes.len()) as u32;
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
                || !v.brakes.is_empty()
        });
        // Parking brakes (jbeam-semantics-spec.md §25.1): per brake 2 records,
        // (node1, node2, arm, first hub node) global, (hub nodes, torque bits,
        // input bits, 0); per vehicle its (first brake, count) in the third aero
        // record's .z, .w.
        let mut brake_recs: Vec<[u32; 4]> = Vec::new();
        let mut brake_spans: Vec<(u32, u32)> = Vec::with_capacity(nv);
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
            let brake_off = (brake_recs.len() / 2) as u32;
            aero_recs.push([slide_off, v.slides.len() as u32, brake_off, v.brakes.len() as u32]);
            brake_spans.push((brake_off, v.brakes.len() as u32));
            let u = v.parking_brake.min(1.0);
            for b in &v.brakes {
                brake_recs.push([
                    node_off + b.axle[0],
                    node_off + b.axle[1],
                    node_off + b.arm,
                    node_off + b.first,
                ]);
                brake_recs.push([b.count, b.torque.to_bits(), u.to_bits(), 0]);
            }
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
                        if i.side == engine_softbody::explicit_aero::WingSide::Left {
                            1
                        } else {
                            2
                        },
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
                        Some(engine_softbody::explicit_aero::WingSide::Left) => {
                            order.extend([(n(6), 0), (n(4), 0)])
                        }
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
        // Centred damping (§39): each record's damping slope of the last beam pass in
        // its first incidence entry's .w (the kernel's beam phase keeps it there).
        {
            let mut off = 0usize;
            for v in vehicles {
                if v.settings.centred_damping {
                    for (j, &slope) in v.damp_slopes.iter().enumerate().take(v.beams.len()) {
                        let at = slots[off + j][0];
                        if at != u32::MAX {
                            beams[inc_base as usize + at as usize][3] = slope;
                        }
                    }
                }
                off += v.beams.len();
            }
        }
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
        // planes << 16), (gAcc region, 0, 0, 0); the regions are placed below.
        let couple_rec = misc.len();
        misc.extend(std::iter::repeat_n([0u32; 4], 2));
        // The raycast wheels' header (the kernel's driveRec(), after the coupling's):
        // (on, region, stride, 0); the region is placed below. Off without wheels.
        let drive_rec = misc.len();
        misc.push([0u32; 4]);
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
                acc_init[vv_acc as usize + 12 * v + k] =
                    if k % 6 < 3 { i32::MAX } else { i32::MIN };
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
        // Two-body contact (jbeam-semantics-spec.md §10.3): per node its force at
        // the contact pass's start.
        let two_body = nv > 1 && vehicles.iter().any(|v| v.two_body_contact);
        let vv_snap = if two_body {
            let at = beams.len() as u32;
            beams.extend(std::iter::repeat_n([0.0f32; 4], total_nodes));
            at
        } else {
            0
        };
        // Surface contact (§38): per node its force at the surface pass's start.
        let surf_snap = if any_surface {
            let at = beams.len() as u32;
            beams.extend(std::iter::repeat_n([0.0f32; 4], total_nodes));
            at
        } else {
            0
        };
        // The tread contact (jbeam-semantics-spec.md §34): per node (z, the depth
        // the step before, 0, 0) for every vehicle when any has tread nodes.
        // Centred damping (§39): per node its step-start velocity, then its Δv.
        let centred_dv = if any_centred {
            let at = beams.len() as u32;
            beams.extend(std::iter::repeat_n([0.0f32; 4], total_nodes));
            at
        } else {
            0
        };
        let tread_state = if vehicles.iter().any(|v| !v.tread.is_empty()) {
            let at = beams.len() as u32;
            for v in vehicles {
                beams.extend((0..v.nodes.len()).map(|i| {
                    let t = v.tread.get(i).copied().unwrap_or([0.0; 2]);
                    [t[0], t[1], 0.0, 0.0]
                }));
            }
            at
        } else {
            0
        };
        // The contact-force log (jbeam-semantics-spec.md §32): in gLists per node
        // its contact break (break group, 0 none; force N's bits).
        let any_cf = vehicles.iter().any(|v| v.contact_force.is_some());
        let cf_node = if any_cf {
            let at = lists.len() as u32;
            for v in vehicles {
                match v.contact_force.as_ref() {
                    Some(l) => {
                        for &(g, f) in &l.node_break {
                            lists.extend([g, f.to_bits()]);
                        }
                    }
                    None => lists.extend(std::iter::repeat_n(0, 2 * v.nodes.len())),
                }
            }
            at
        } else {
            0
        };
        // JBeam damping filters: (last rate, last output, a, b) each.
        let filter_base = beams.len();
        beams.extend(filters.iter().map(|f| [f.u, f.y, f.a, f.b]));
        for &(rec, i) in &filter_patches {
            beam_recs[rec][3] = (filter_base + i + 1) as u32;
        }
        let ext_state_base = beams.len() as u32;
        beams.extend(ext_state);
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
        let mut next_len = hit_base + nv * SELF_HIT_WORDS;
        let hit_acc = acc_init.len();
        acc_init.extend(std::iter::repeat_n(0, 2 * nv));
        // Last: the loop-cap flag (the kernel's `EB_LOOP_CAP`, at the parameter
        // `loopCapWord`; the frozen WGSL test modules' post-process writes gAcc's
        // last word, the same word).
        let coupling = coupling.map(|cap| {
            let layout =
                CouplingLayout::new(cap, nv as u32, misc.len() as u32, acc_init.len() as u32);
            misc.extend(std::iter::repeat_n([0u32; 4], layout.misc_words() as usize));
            acc_init.extend(std::iter::repeat_n(0, layout.acc_words() as usize));
            // The shape cull's spheres (`explicit_beams_rigid.slang` `cplSpheres`): per
            // vehicle and shape 4 words of gNext, rewritten every step.
            let cull_base = next_len as u32;
            next_len += (4 * cap.shapes * nv as u32) as usize;
            misc[couple_rec] = [
                0,
                layout.misc_base,
                cap.bodies,
                cap.shapes | cap.planes << 16,
            ];
            misc[couple_rec + 1] = [layout.acc_base, cull_base, 0, 0];
            layout
        });
        // Raycast wheels (explicit_plan::drive): a region per vehicle when any has them.
        let drive_wheels = vehicles.iter().filter_map(|v| v.drive.as_ref()).map(|d| d.config.wheels.len()).max();
        let drive = drive_wheels.map(|w| {
            let layout = drive::DriveLayout {
                header: drive_rec as u32,
                base: misc.len() as u32,
                stride: drive::DriveLayout::stride_for(w),
            };
            for v in vehicles {
                misc.extend(drive::records(v.drive.as_deref(), layout.stride));
            }
            misc[drive_rec] = layout.header_words();
            layout
        });
        // Vehicle contact's two-body mass splitting (jbeam-semantics-spec.md §10.3):
        // per node its triangle-hit count, then VV_SPLIT_SLOTS − 1 point-hit slots.
        let vv_split = if two_body {
            let at = acc_init.len() as u32;
            acc_init.extend(std::iter::repeat_n(0, VV_SPLIT_SLOTS * total_nodes));
            at
        } else {
            0
        };
        // The contact-force log's state in gAcc: per node (this step's sum, the last
        // step's, the peak), fixed point; then per group slot its contact-break
        // state (0 idle, 1 set off, 2 done).
        let (cf_acc, cf_pend) = if any_cf {
            let at = acc_init.len() as u32;
            for v in vehicles {
                match v.contact_force.as_ref() {
                    Some(l) => {
                        for i in 0..v.nodes.len() {
                            acc_init.extend([l.sum[i], l.last[i], l.peak[i]]);
                        }
                    }
                    None => acc_init.extend(std::iter::repeat_n(0, 3 * v.nodes.len())),
                }
            }
            let pend = acc_init.len() as u32;
            for (v, &(_, count)) in vehicles.iter().zip(&group_spans) {
                for g in 0..count as usize {
                    let p = v
                        .contact_force
                        .as_ref()
                        .and_then(|l| l.pending.get(g))
                        .copied();
                    acc_init.push(i32::from(p.unwrap_or(0)));
                }
            }
            (at, pend)
        } else {
            (0, 0)
        };
        // Self-collision's two-body rule (jbeam-semantics-spec.md §36): per node its
        // hit count this step (cleared by its node phase).
        // Dissipative contact's position steps (§37): per node 3 words in SHIFT_FIXED
        // units, right after the counts (the kernel's shiftAt: selfSplit +
        // totalNodes), carried over from the CPU state (zero without them).
        let self_split = if vehicles.iter().any(|v| v.two_body_contact) || any_shift {
            let at = acc_init.len() as u32;
            acc_init.extend(std::iter::repeat_n(0, total_nodes));
            for v in vehicles {
                for i in 0..v.nodes.len() {
                    let w = v.contact_shift.get(i).copied().unwrap_or([0; 3]);
                    acc_init.extend(w);
                }
            }
            // §37.1's residual pass and work guard (the kernel's resAt: selfSplit +
            // 4·totalNodes): per node 8 words (the residual's staged sums B, the
            // totals' deltas D), zero between steps.
            acc_init.extend(std::iter::repeat_n(0, 8 * total_nodes));
            at
        } else {
            0
        };
        // Anti-clip (§36): per node its side memory (held, recorded), then per vehicle
        // two parities of its fastest node's speed.
        let (clip_mem, clip_vmax) = if any_anti {
            let empty = engine_softbody::explicit_contact::CLIP_EMPTY;
            let mem = acc_init.len() as u32;
            for v in vehicles {
                for i in 0..v.nodes.len() {
                    let c = v.clip_sides.get(i).copied().unwrap_or([empty; 4]);
                    acc_init.extend(c.map(|w| w as i32));
                }
            }
            let vmax = acc_init.len() as u32;
            acc_init.extend(std::iter::repeat_n(0, 2 * nv));
            (mem, vmax)
        } else {
            (0, 0)
        };
        // Surface contact (§38): per node its hit count this step.
        let surf_split = if any_surface {
            let at = acc_init.len() as u32;
            acc_init.extend(std::iter::repeat_n(0, total_nodes));
            at
        } else {
            0
        };
        // Surface contact's segments per vehicle (beam-edge contact; none without
        // the flag), and the find's slot masks (gAcc): per triangle, then per
        // segment, then per node, each the slots it hit (bit b slot b, bit 31 any
        // slot from 31 on), so the apply recomputes only those.
        let surf_segs: Vec<Vec<u32>> = vehicles
            .iter()
            .map(|v| {
                if any_surface && v.surface_contact {
                    engine_softbody::explicit_surface::surface_segments(v)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let surf_seg_total: usize = surf_segs.iter().map(Vec::len).sum();
        let surf_mark = if any_surface {
            let at = acc_init.len() as u32;
            acc_init.extend(std::iter::repeat_n(0, total_tris + surf_seg_total + total_nodes));
            at
        } else {
            0
        };
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
        let brake_base = statics.len() as u32;
        statics.extend(brake_recs);
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
        let ext_base = statics.len() as u32;
        statics.extend(ext_recs);
        let tri_break_base = statics.len() as u32;
        statics.extend(tri_break_recs);
        let ext_index_base = lists.len() as u32;
        lists.extend(ext_index);
        let tri_excl_base = if any_tri_excl {
            let at = lists.len() as u32;
            lists.extend(tri_excl_list);
            let base = statics.len() as u32;
            statics.extend(tri_excl_recs.iter().map(|r| [r[0] + at, r[1], 0, 0]));
            base
        } else {
            0
        };
        // Surface contact's segments (§38, beam-edge contact): per vehicle (start in
        // gLists, count, its first segment's mask in gAcc, its first node's mask in
        // gAcc), the global beam indices in gLists.
        let surf_seg = if any_surface {
            let base = statics.len() as u32;
            let mut beam_at = 0u32;
            let mut seg_mark = surf_mark + total_tris as u32;
            let mut node_mark = seg_mark + surf_seg_total as u32;
            for (v, segs) in vehicles.iter().zip(&surf_segs) {
                statics.push([lists.len() as u32, segs.len() as u32, seg_mark, node_mark]);
                lists.extend(segs.iter().map(|&j| beam_at + j));
                beam_at += v.beams.len() as u32;
                seg_mark += segs.len() as u32;
                node_mark += v.nodes.len() as u32;
            }
            base
        } else {
            0
        };

        let big = vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS);
        let lanes = frame_lanes(vehicles);
        // Two parities of group slots.
        let group_total = groups.len() as u32;
        let mut groups: Vec<u32> = groups.iter().chain(groups.iter()).copied().collect();
        let deform_base = groups.len() as u32;
        groups.extend(std::iter::repeat_n(0u32, deform_total as usize));
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
            ext_index_base,
            ext_base,
            ext_state_base,
            tri_break_base,
            deform_base,
            tri_excl_base,
            cap_at as u32,
            vv_split,
            vv_snap,
            cf_acc,
            cf_node,
            cf_pend,
            tread_state,
            self_split,
            clip_mem,
            clip_vmax,
            surf_snap,
            surf_split,
            surf_seg,
            surf_mark,
            centred_dv,
            brake_base,
        ];
        let words: [u32; PARAM_WORDS] = std::array::from_fn(|i| words.get(i).copied().unwrap_or(0));
        let total = [
            total_nodes as u32,
            total_beams as u32,
            total_tris as u32,
            total_cands,
            group_total,
        ];
        // The wide passes do not run wings, the fuselage drag, commands or shocks2.
        // Raycast wheels run in the vehicle workgroups only.
        // (Surface contact runs in the frame modules: not in the wide passes.)
        let wide_ok = nv <= WIDE_MAX_VEHICLES && !aero && drive.is_none() && !any_surface;
        // Every vehicle must have run the same steps for the wide passes (one
        // step parity for the group slots).
        let same_steps = vehicles.windows(2).all(|w| w[0].steps == w[1].steps);
        let wide_default = same_steps && big;
        let n = table_recs.len().min(WIDE_MAX_VEHICLES * 4);
        let mut table = table_recs[..n].to_vec();
        table.resize(WIDE_MAX_VEHICLES * 5, [0; 4]);
        let layout = ExplicitLayout {
            spans,
            rest0,
            words,
            deform_base,
            deform_spans,
            aero_state,
            slide_state: slide_state_base,
            key_base,
            wing_spans,
            brake_base,
            brake_spans,
            total,
            couple_rec: couple_rec as u32,
            coupling,
            drive,
            vv_tri_base,
            aero,
            big,
            lanes,
            wide_ok,
            wide_default,
            contact_force: cf_acc,
            group_spans: group_spans.clone(),
            contact_force_on: vehicles.iter().map(|v| v.contact_force.is_some()).collect(),
            status: StatusWords {
                base: u64::from(vv_acc + 12 * nv as u32) * 4,
                cap: cap_at as u64 * 4,
            },
        };
        Self {
            layout,
            statics,
            lists,
            nodes,
            beams,
            acc: acc_init,
            groups,
            misc,
            next: next_init,
            table,
        }
    }

    /// The initial contents of binding `b` (an empty buffer as 16 zero bytes). The
    /// parameters: the upload's words at copy 0 of a [`params_bytes`] buffer.
    pub fn bytes(&self, b: u32) -> Vec<u8> {
        let pad = |b: &[u8]| {
            if b.is_empty() {
                vec![0u8; 16]
            } else {
                b.to_vec()
            }
        };
        let cast = |x: &[u8]| pad(x);
        match b {
            BIND_PARAMS => {
                let mut out = vec![0u8; params_bytes() as usize];
                out[..4 * PARAM_WORDS].copy_from_slice(bytemuck::cast_slice(&self.layout.words));
                out
            }
            BIND_STATIC => cast(bytemuck::cast_slice(&self.statics)),
            BIND_LISTS => cast(bytemuck::cast_slice(&self.lists)),
            BIND_NODES => cast(bytemuck::cast_slice(&self.nodes)),
            BIND_BEAMS => cast(bytemuck::cast_slice(&self.beams)),
            BIND_ACC => cast(bytemuck::cast_slice(&self.acc)),
            BIND_GROUPS => cast(bytemuck::cast_slice(&self.groups)),
            BIND_MISC => cast(bytemuck::cast_slice(&self.misc)),
            BIND_NEXT => cast(bytemuck::cast_slice(&self.next)),
            BIND_TABLE => bytemuck::cast_slice(&self.table).to_vec(),
            BIND_ARGS => vec![0u8; ARGS_BYTES as usize],
            _ => panic!("explicit plan: no binding {b}"),
        }
    }

    /// The wide table's records a WebGPU host writes at creation (the rest zero).
    pub fn table_records(&self) -> &[[u32; 4]] {
        &self.table
    }
}

/// Bytes of the parameter buffer ([`PARAM_COPIES`] copies).
pub const fn params_bytes() -> u64 {
    PARAM_STRIDE * PARAM_COPIES as u64
}

// --- Per-frame planning.

/// The vehicle-contact plan's host state.
#[derive(Debug, Default)]
struct ContactPlan {
    /// On (the default); it runs only with more than one vehicle.
    on: bool,
    /// Steps the current plan still covers; a new plan is due when a frame needs
    /// more (or `replan` is set: velocities written by the host).
    left: u32,
    replan: AtomicBool,
    /// This frame runs the plan passes.
    planning: bool,
    /// No plan has run since contact (re)started.
    fresh: bool,
}

/// The planner's state a host saves with a GPU snapshot (tests).
#[derive(Clone, Copy, Debug)]
pub struct PlannerState {
    left: u32,
    fresh: bool,
    frames: u64,
    replan: bool,
}

impl PlannerState {
    /// As words (tests: a snapshot saved to a file).
    pub fn to_words(self) -> [u32; 5] {
        [self.left, u32::from(self.fresh), self.frames as u32, (self.frames >> 32) as u32, u32::from(self.replan)]
    }
    /// From [`Self::to_words`].
    pub fn from_words(w: [u32; 5]) -> Self {
        Self {
            left: w[0],
            fresh: w[1] != 0,
            frames: u64::from(w[2]) | (u64::from(w[3]) << 32),
            replan: w[4] != 0,
        }
    }
}

/// Plans each frame of one uploaded fleet: parameters and commands. Hosts call
/// [`Self::write_params`], then [`Self::commands`], then [`Self::status_due`].
#[derive(Debug)]
pub struct FramePlanner {
    pub words: [u32; PARAM_WORDS],
    nv: u32,
    total: [u32; 5],
    contact: ContactPlan,
    /// Wide passes possible ([`ExplicitLayout::wide_ok`]) and on.
    wide_ok: bool,
    wide_on: bool,
    /// Frames submitted (every STATUS_EVERY-th reads the status).
    frames: u64,
}

impl FramePlanner {
    pub fn new(layout: &ExplicitLayout) -> Self {
        Self {
            words: layout.words,
            nv: layout.spans.len() as u32,
            total: layout.total,
            contact: ContactPlan {
                on: true,
                fresh: true,
                ..Default::default()
            },
            wide_ok: layout.wide_ok,
            wide_on: layout.wide_default,
            frames: 0,
        }
    }

    /// Vehicles in the fleet.
    pub fn vehicles(&self) -> u32 {
        self.nv
    }

    /// Vehicle–vehicle contact between every vehicle of the fleet
    /// (`engine/docs/explicit-beams-vehicle-contact-spec.md`): on by default; it runs
    /// when the fleet has more than one vehicle.
    pub fn set_vehicle_contact(&mut self, on: bool) {
        if on && !self.contact.on {
            self.contact.fresh = true;
        }
        self.contact.on = on;
        self.contact.left = 0;
    }

    pub fn vehicle_contact(&self) -> bool {
        self.contact.on
    }

    /// Whether frames run the vehicle contact passes (on, more than one vehicle).
    pub fn contact_active(&self) -> bool {
        self.contact.on && self.nv > 1
    }

    /// The frame bounds margin (m, default [`DEFAULT_CONTACT_MARGIN`]).
    pub fn set_contact_margin(&mut self, metres: f32) {
        assert!(
            metres >= 0.0 && metres.is_finite(),
            "contact margin {metres}"
        );
        self.words[W_VV_MARGIN] = metres.to_bits();
        self.contact.left = 0;
    }

    /// The frame bounds margin (m).
    pub fn contact_margin(&self) -> f32 {
        f32::from_bits(self.words[W_VV_MARGIN])
    }

    /// A new contact plan is due next frame (the host wrote velocities or forces the
    /// current plan's frame bounds did not assume).
    pub fn request_replan(&self) {
        self.contact.replan.store(true, Ordering::Relaxed);
    }

    /// Wide passes on (and possible), and contact not active.
    pub fn is_wide(&self) -> bool {
        self.wide_ok && self.wide_on && !self.contact_active()
    }

    /// Run frames as the wide passes or one workgroup per vehicle; returns
    /// [`Self::is_wide`].
    pub fn set_wide(&mut self, on: bool) -> bool {
        if self.wide_ok {
            self.wide_on = on;
        }
        self.is_wide()
    }

    /// Measurement only: the phases `EB_ABLATE` builds skip.
    pub fn set_ablation(&mut self, mask: u32) {
        self.words[W_ABLATE] = mask;
    }

    /// The parameters for a frame of `steps` steps, from byte 0 of binding 0: copy k
    /// (at k × PARAM_STRIDE) carries step k of the frame (k = steps: the frame's
    /// end). Decides whether this frame runs the contact plan passes.
    pub fn write_params(&mut self, steps: u32) -> Vec<u8> {
        assert!(
            steps < PARAM_COPIES,
            "at most {} steps per frame",
            PARAM_COPIES - 1
        );
        self.words[0] = steps;
        self.words[W_VV_ON] = self.contact_active() as u32;
        self.contact.planning = false;
        if self.contact_active() {
            let replan = self.contact.replan.swap(false, Ordering::Relaxed);
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
        bytes
    }

    /// The parameters of [`Self::frame_split`]-style measurements: the words alone,
    /// with `steps` and contact off.
    pub fn split_params(&mut self, steps: u32) -> Vec<u8> {
        self.words[0] = steps;
        self.words[W_VV_ON] = 0;
        bytemuck::cast_slice(&self.words).to_vec()
    }

    /// This frame runs the contact plan passes (after [`Self::write_params`]).
    pub fn planning(&self) -> bool {
        self.contact.planning
    }

    /// The frame's commands after [`Self::write_params`] for the same `steps`.
    /// `wide_allowed` false keeps the per-vehicle frame (test kernel variants).
    pub fn commands(&self, steps: u32, wide_allowed: bool) -> Vec<Cmd> {
        let nv = self.nv;
        let d = |kernel, groups, param_copy| Cmd::Dispatch {
            kernel,
            groups,
            param_copy,
        };
        if self.contact_active() {
            let mut out = Vec::with_capacity(5);
            if self.contact.planning {
                out.push(d(Kernel::Vv(0), nv, 0));
                if nv as usize > FUSED_PAIRS_MAX {
                    out.push(d(Kernel::Vv(1), nv.div_ceil(256), 0));
                }
                out.push(d(Kernel::Vv(2), 1, 0));
            }
            // The vehicles (grouped ones return at once) and, past them, the
            // contact groups, side by side. The plan wrote one of the two non-empty.
            out.push(Cmd::Indirect {
                kernel: Kernel::Solo,
                offset: 0,
            });
            out.push(Cmd::Indirect {
                kernel: Kernel::Contact,
                offset: 16,
            });
            out
        } else if self.is_wide() && wide_allowed {
            let plan = self.wide_plan(steps);
            let mut out: Vec<Cmd> = plan
                .iter()
                .map(|&(k, kind, x)| d(Kernel::Wide(kind), x, k))
                .collect();
            // Begin and rebase, then the vehicles' origins and steps into the
            // table, then the steps.
            out.insert(
                2,
                Cmd::Copy {
                    src: BIND_MISC,
                    src_offset: 0,
                    dst: BIND_TABLE,
                    dst_offset: (WIDE_MAX_VEHICLES * 4 * 16) as u64,
                    size: u64::from(nv) * 16,
                },
            );
            out
        } else {
            vec![d(Kernel::Frame, nv, 0)]
        }
    }

    /// Whether this frame reads the status (every [`STATUS_EVERY`] frames, the first
    /// included); counts the frame.
    pub fn status_due(&mut self) -> bool {
        self.frames += 1;
        self.frames % STATUS_EVERY == 1
    }

    pub fn state(&self) -> PlannerState {
        PlannerState {
            left: self.contact.left,
            fresh: self.contact.fresh,
            frames: self.frames,
            replan: self.contact.replan.load(Ordering::Relaxed),
        }
    }

    pub fn restore(&mut self, s: PlannerState) {
        self.contact.left = s.left;
        self.contact.fresh = s.fresh;
        self.frames = s.frames;
        self.contact.replan.store(s.replan, Ordering::Relaxed);
    }

    /// The wide passes of a frame: (parameter copy, pass, workgroups).
    pub fn wide_plan(&self, steps: u32) -> Vec<(u32, usize, u32)> {
        let groups = |n: u32| n.div_ceil(256).max(1);
        let nv = self.nv;
        let [nodes, beams, tris, cands, slots] = self.total;
        let buckets = nv * WIDE_GRID_BUCKETS as u32;
        let centred = self.words[W_CENTRED_DV] != 0;
        let mut plan = vec![(0, WIDE_BEGIN, groups(nv)), (0, WIDE_REBASE, groups(nodes))];
        for k in 0..steps {
            plan.push((k, WIDE_NODES, groups(nodes + slots + buckets)));
            if centred {
                plan.push((k, WIDE_CENTRED_BEAMS, groups(beams)));
                plan.push((k, WIDE_CENTRED_NODES, groups(nodes)));
            }
            plan.push((k, WIDE_BEAMS_PASS, groups(beams + cands + tris)));
            plan.push((k, WIDE_TRIS, groups(tris * WIDE_SUBS)));
            // One workgroup per vehicle: deferred breaks and self-collision hits.
            plan.push((k, WIDE_HITS, nv));
        }
        plan.push((steps, WIDE_END, groups(tris + beams)));
        plan.push((steps, WIDE_CLOSE, groups(nv)));
        plan
    }
}

// --- Host writes and readback decoding.

impl ExplicitLayout {
    pub fn vehicles(&self) -> usize {
        self.spans.len()
    }

    /// Byte offset in [`BIND_NODES`] of a node's velocity (3 f32).
    pub fn velocity_offset(&self, vehicle: usize, node: u32) -> u64 {
        ((self.spans[vehicle].node_off + node) as u64 * 3 + 1) * 16
    }

    /// Byte offset in [`BIND_MISC`] of the rigid coupling's `on` word (its header).
    pub fn coupling_on_offset(&self) -> u64 {
        u64::from(self.couple_rec) * 16
    }

    /// gMisc record of the grab (the kernel's grabRec()).
    pub fn grab_rec(&self) -> u64 {
        u64::from(self.words[16]) + 1 + self.spans.len() as u64
    }

    /// The grab's two gMisc records ([`Self::grab_rec`]); `None` releases it.
    pub fn grab_records(
        &self,
        grab: Option<(usize, engine_softbody::explicit_beams::Grab)>,
    ) -> [u32; 8] {
        match grab {
            Some((vehicle, g))
                if vehicle < self.spans.len() && g.node < self.spans[vehicle].nodes =>
            {
                [
                    1,
                    self.spans[vehicle].node_off + g.node,
                    g.k.to_bits(),
                    g.c.to_bits(),
                    g.target.x.to_bits(),
                    g.target.y.to_bits(),
                    g.target.z.to_bits(),
                    0,
                ]
            }
            _ => [0u32; 8],
        }
    }

    fn node_field(&self, nodes: &[[f32; 4]], field: usize) -> Vec<Vec<Vec3>> {
        self.spans
            .iter()
            .map(|s| {
                (0..s.nodes)
                    .map(|i| {
                        let r = nodes[(s.node_off + i) as usize * 3 + field];
                        Vec3::new(r[0], r[1], r[2])
                    })
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, the origins nodes are relative to (from gMisc).
    pub fn origins(&self, misc: &[[u32; 4]]) -> Vec<Vec3> {
        (0..self.spans.len())
            .map(|i| {
                Vec3::new(
                    f32::from_bits(misc[i][0]),
                    f32::from_bits(misc[i][1]),
                    f32::from_bits(misc[i][2]),
                )
            })
            .collect()
    }

    /// Per vehicle, world node positions (from gNodes and gMisc).
    pub fn positions(&self, nodes: &[[f32; 4]], misc: &[[u32; 4]]) -> Vec<Vec<Vec3>> {
        let origins = self.origins(misc);
        let mut rel = self.node_field(nodes, 0);
        for (o, nodes) in origins.iter().zip(&mut rel) {
            for p in nodes.iter_mut() {
                *p += *o;
            }
        }
        rel
    }

    /// Per vehicle, node velocities (from gNodes).
    pub fn velocities(&self, nodes: &[[f32; 4]]) -> Vec<Vec<Vec3>> {
        self.node_field(nodes, 1)
    }

    /// Per vehicle, every beam's broken flag and rest length (from gBeams).
    pub fn beams(&self, beams: &[[f32; 4]]) -> Vec<Vec<GpuBeam>> {
        self.spans
            .iter()
            .map(|s| {
                (0..s.beams)
                    .map(|j| {
                        let g = (s.beam_off + j) as usize;
                        GpuBeam {
                            broken: beams[g * 2 + 1][2] != 0.0,
                            rest: beams[g * 2][0],
                            rest0: self.rest0[g],
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, its wings' broken flags (file order; from gBeams).
    pub fn wings_broken(&self, beams: &[[f32; 4]]) -> Vec<Vec<bool>> {
        let nv = self.spans.len() as u32;
        self.wing_spans
            .iter()
            .map(|&(first, count)| {
                (first..first + count)
                    .map(|gw| beams[(self.aero_state + nv + 4 * gw) as usize][0] != 0.0)
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, the fuselage drag force (from gBeams).
    pub fn fuse_forces(&self, beams: &[[f32; 4]]) -> Vec<Vec3> {
        (0..self.spans.len())
            .map(|i| {
                let r = beams[self.aero_state as usize + i];
                Vec3::new(r[0], r[1], r[2])
            })
            .collect()
    }

    /// Per vehicle, each slide node's current segment and broken-away flag.
    pub fn slides(&self, beams: &[[f32; 4]]) -> Vec<Vec<(Option<u32>, bool)>> {
        self.spans
            .iter()
            .map(|s| {
                (0..s.slides)
                    .map(|k| {
                        let r = beams[(self.slide_state + s.slide_off + k) as usize];
                        let seg = (r[0] >= 0.0).then(|| r[0] as u32 - s.seg_off);
                        (seg, r[1] != 0.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// Per vehicle, the trigger inputs the next step's command pass reads (from gNext).
    pub fn key_inputs(&self, next: &[u32], steps: u64) -> Vec<[f32; KEYS]> {
        let parity = (steps as usize & 1) ^ 1;
        (0..self.spans.len())
            .map(|vi| {
                let mut out = [0.0f32; KEYS];
                let at = self.key_base as usize + (vi * 2 + parity) * KEYS;
                if at + KEYS <= next.len() {
                    for (k, o) in out.iter_mut().enumerate() {
                        *o = (next[at + k] != 0) as u32 as f32;
                    }
                }
                out
            })
            .collect()
    }

    /// Per vehicle its triggered JBeam deform groups (from gGroups).
    pub fn deform_triggered(&self, groups: &[u32]) -> Vec<Vec<u32>> {
        self.deform_spans
            .iter()
            .map(|&(first, n)| {
                let at = (self.deform_base + first) as usize;
                groups
                    .get(at..at + n as usize)
                    .map_or_else(|| vec![0; n as usize], <[u32]>::to_vec)
            })
            .collect()
    }

    /// Per vehicle, its triangles' contact throttles (cwait, cidle; from gMisc).
    pub fn contact_throttles(&self, misc: &[[u32; 4]]) -> Vec<Vec<[i32; 2]>> {
        self.throttles(misc, self.vv_tri_base as usize)
    }

    /// Per vehicle, its triangles' self-collision throttles (wait, idle; from gMisc).
    pub fn self_collision_throttles(&self, misc: &[[u32; 4]]) -> Vec<Vec<[i32; 2]>> {
        self.throttles(misc, self.spans.len())
    }

    fn throttles(&self, misc: &[[u32; 4]], first: usize) -> Vec<Vec<[i32; 2]>> {
        let mut out = Vec::with_capacity(self.spans.len());
        let mut t0 = first;
        for s in &self.spans {
            let n = s.tris as usize;
            out.push(
                misc[t0..t0 + n]
                    .iter()
                    .map(|r| [r[0] as i32, r[1] as i32])
                    .collect(),
            );
            t0 += n;
        }
        out
    }

    /// The contact status counters' error (from gAcc).
    /// The gAcc words of the contact-force peaks (word index, count) of every node,
    /// three words apart (to clear them), or `None` when no vehicle has the log.
    pub fn contact_force_peaks(&self) -> Option<(u32, u32)> {
        let total: u32 = self.spans.iter().map(|s| s.nodes).sum();
        (self.contact_force != 0).then_some((self.contact_force + 2, total))
    }

    /// Per vehicle with the contact-force log (spec §32), from gAcc: its nodes'
    /// last complete step's contact force and the peak (N), and its contact-break
    /// states by group; `None` for the others.
    #[allow(clippy::type_complexity)]
    pub fn contact_forces(&self, acc: &[i32]) -> Vec<Option<(Vec<f32>, Vec<f32>, Vec<u8>)>> {
        let has = &self.contact_force_on;
        if self.contact_force == 0 {
            return vec![None; self.spans.len()];
        }
        let base = self.contact_force as usize;
        let total: usize = self.spans.iter().map(|s| s.nodes as usize).sum();
        let pend = base + 3 * total;
        self.spans
            .iter()
            .zip(&self.group_spans)
            .zip(has)
            .map(|((s, &(goff, gcount)), &on)| {
                on.then(|| {
                    let at = |i: usize, k: usize| acc[base + 3 * (s.node_off as usize + i) + k];
                    let f = |q: i32| q as f32 / 32.0;
                    (
                        (0..s.nodes as usize).map(|i| f(at(i, 1))).collect(),
                        (0..s.nodes as usize).map(|i| f(at(i, 2))).collect(),
                        (0..gcount as usize)
                            .map(|g| acc[pend + goff as usize + g] as u8)
                            .collect(),
                    )
                })
            })
            .collect()
    }

    pub fn contact_status(&self, acc: &[i32]) -> Result<(), String> {
        let at = (self.status.base / 4) as usize;
        contact_error(&acc[at + 4..at + 8])
    }

    /// The loop-cap flag (from gAcc): nonzero when a kernel loop was held to its
    /// cap ([`explicit_loop_cap_hit`]). Every target (WGSL, MSL, CUDA) writes it.
    pub fn loop_cap_flag(&self, acc: &[i32]) -> i32 {
        acc[(self.status.cap / 4) as usize]
    }

    /// The kernel status (from gAcc): the loop-cap flag, then the contact
    /// counters ([`Self::contact_status`]).
    pub fn kernel_status(&self, acc: &[i32]) -> Result<(), String> {
        explicit_loop_cap_hit(self.loop_cap_flag(acc))
            .map_err(|e| format!("explicit beams kernel: {e}"))?;
        self.contact_status(acc)
    }
}

/// The explicit kernels' loop cap (`EB_LOOP_CAP` in `explicit_beams.slang`, every
/// target): a loop whose bound is not a constant runs at most this many passes.
/// Far above any legitimate count (a few loops run on one lane over a whole
/// vehicle's beams: tens of thousands for a large vehicle), so a capped loop
/// computes what an uncapped one does.
pub const EXPLICIT_LOOP_CAP: u32 = 1 << 20;

/// The kernel's tag on a loop-cap flag (`EB_LOOP_CAP_TAG`): the flag is the tag
/// | the stopped loop's source line. Untagged, the flag is the loop's number in
/// its WGSL module (counting from 1): the frozen test modules
/// (`explicit_beams_baseline.slang`, `explicit_beams_ref.slang`), capped by
/// engine-webgpu's WGSL post-process.
pub const EXPLICIT_LOOP_CAP_TAG: i32 = 1 << 24;

/// The error for a loop-cap flag read back from gAcc ([`ExplicitLayout::loop_cap_flag`];
/// 0: none).
pub fn explicit_loop_cap_hit(flag: i32) -> Result<(), String> {
    if flag == 0 {
        return Ok(());
    }
    let place = if flag & EXPLICIT_LOOP_CAP_TAG != 0 {
        format!(
            "the loop at line {} of explicit_beams.slang (or of an explicit_beams_*.slang file it includes)",
            flag & !EXPLICIT_LOOP_CAP_TAG
        )
    } else {
        format!("a kernel loop (number {flag} in its WGSL module, counting from 1)")
    };
    Err(format!(
        "{place} ran {EXPLICIT_LOOP_CAP} passes, its cap, and was stopped: that frame's results \
         are invalid (a runaway loop: a corrupt count or bound)"
    ))
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
pub fn frame_lanes(vehicles: &[ExplicitBeams]) -> u32 {
    let big = vehicles.iter().any(|v| v.beams.len() > WIDE_BEAMS);
    let default = if vehicles.len() <= SMALL_FLEET || big {
        512
    } else {
        256
    };
    match std::env::var("RUBBLE_EXPLICIT_LANES")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(n @ (256 | 512)) => n,
        _ => default,
    }
}

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
        } => vec![
            rec(centre, radius),
            rec(Vec3::new(half_height, 0.0, 0.0), 0.0),
        ],
        StaticShape::Capsule { a, b, radius } => vec![rec(a, radius), rec(b, 0.0)],
        StaticShape::Sphere { centre, radius } => vec![rec(centre, radius), [0; 4]],
    }
}

/// The self-collision grid's cell size for a vehicle (0: all pairs): the
/// solver's own ([`ExplicitBeams::grid_cell`]), so CPU and GPU share it.
fn grid_cell(v: &ExplicitBeams) -> f32 {
    v.grid_cell
}

/// The vehicle grid's cell size: the largest vehicle extent (at upload), plus room
/// for the frame bounds and the box margin.
fn vv_cell(vehicles: &[ExplicitBeams]) -> f32 {
    let extent = vehicles
        .iter()
        .map(|v| {
            let (lo, hi) = v.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
                (a.min(n.pos), b.max(n.pos))
            });
            (hi - lo).max_element()
        })
        .filter(|e| e.is_finite())
        .fold(0.0f32, f32::max);
    extent + 2.0 * (DEFAULT_CONTACT_MARGIN + 0.05) + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each vehicle's box slots (its boxes, then its static shapes) start after every
    /// earlier vehicle's: two vehicles with a box and a cylinder each, the second's
    /// first slot is its own box.
    #[test]
    fn box_slots_of_later_vehicles_skip_earlier_shapes() {
        use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape};
        let text = "slots\nglobals\n40, 0, tracks/semi\nnodes\n0, 0.0, 0.0, 0.0\n1, 0.3, 0.0, 0.0\nbeams\n0, 1\nend\n";
        let settings = ExplicitBeamsSettings {
            drag: false,
            turbulence: false,
            fast_inv_sqrt: true,
            triangle_throttle: true,
            centred_damping: false,
        };
        let mut vs = Vec::new();
        for k in 0..2 {
            let mut v = ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
            let c = Vec3::new(10.0 * k as f32 + 1.0, 0.5, 0.0);
            v.boxes.push(StaticBox::centered(c, Vec3::splat(0.5)));
            v.shapes.push(StaticShape::cylinder(c + Vec3::X * 3.0, 0.2, 1.0));
            vs.push(v);
        }
        let up = ExplicitUpload::new(&vs, None);
        let box_base = up.layout.words[5] as usize;
        for (k, v) in vs.iter().enumerate() {
            let r1 = up.statics[k * 4 + 1];
            assert_eq!(r1[3], 2, "vehicle {k}: its box and its cylinder");
            let lo = up.statics[box_base + r1[2] as usize * 2];
            let want = v.boxes[0].lo;
            assert_eq!(
                [f32::from_bits(lo[0]), f32::from_bits(lo[1]), f32::from_bits(lo[2])],
                [want.x, want.y, want.z],
                "vehicle {k}: its first slot (offset {}) is its own box",
                r1[2]
            );
        }
    }

    /// Every kernel a plan can name has a module, with its entry.
    #[test]
    fn kernels_have_modules() {
        let mut ks = vec![Kernel::Frame, Kernel::Solo, Kernel::Contact];
        ks.extend((0..3).map(Kernel::Vv));
        ks.extend((0..WIDE_PASSES.len()).map(Kernel::Wide));
        for k in ks {
            for aero in [false, true] {
                for lanes in [256, 512] {
                    let m = module(k.module(aero, lanes)).unwrap_or_else(|| panic!("{k:?}"));
                    assert_eq!(m.entry, k.entry());
                }
            }
        }
        let mut names: Vec<_> = MODULES.iter().map(|m| m.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), MODULES.len());
    }
}

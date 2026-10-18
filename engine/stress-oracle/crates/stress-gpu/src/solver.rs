//! The GPU stress solver with the full model: explicit substeps of every cluster with
//! the joint law, fracture, rigid motion, drift removal, scripted and replacement loads,
//! probes, and penalty contacts with impactors and the ground, all in
//! `shaders/world.slang` (one binding layout for every kernel).
//!
//! Islands that touch nothing this segment run all their substeps in one dispatch; the
//! others (and impactors) advance in lockstep through a per-substep contact pipeline.
//!
//! A `ReferenceSolver` is kept as the host mirror of topology and history. Rare
//! operations run the reference's own code on it after the GPU state is downloaded:
//! when a bond disconnects, its island stops after that substep, and the host splits the
//! cluster with `ReferenceSolver::split_cluster` (as the reference does at the end of
//! that substep), re-uploads, and lets the children finish the parent's remaining
//! substeps.

use std::collections::HashMap;

use stress_ref::bond::Local6;
use stress_ref::math::{Mat3, Quat, Vec3};
use stress_ref::scene::{ImpactorShape, Scene, SolveMode, Support};
use stress_ref::solver::{Activity, ChunkLoads, ReferenceSolver, SolverEvent};
use stress_ref::statics::StaticOptions;
use stress_ref::world::Impactor;

use crate::coverage;
use crate::contacts::{modulus, plan, sample_count, PairKey, PairState, Plan, NAN_STATE};
use crate::gpu::{Binding, Gpu, Kernel};
use crate::joint::{from_gpu_state, mode_of, to_gpu_state, GpuJointMaterial};
pub use stress_layout::{BondDyn, Island};
use stress_layout::{joined, rows, split, split1, split_quat, v4, vec3, StructureTables, ISLAND_ANCHORED, ISLAND_SETTLED, MAX_MATERIALS};
use crate::loads::{Function, LoadTerm, ProbeItem, Target, WorldLoads};
use crate::shaders;


const ISLAND_CONTACT: u32 = 4;
const ISLAND_WIDE: u32 = 8;
const ISLAND_SOLVE: u32 = 8;
const ISLAND_QUIET: u32 = 32;
const ISLAND_DAMAGED: u32 = 16;
const ISLAND_DISCONNECTED: u32 = 32;
/// Threads per group of the wide kernels (chunks or bonds per group).
const WIDE_GROUP: usize = 256;

/// Islands with at least this many chunks spread over many threadgroups
/// (`STRESS_GPU_WIDE_CHUNKS` overrides; 0 disables): above what a narrow island's own
/// connectivity check holds (`CONNECT_NARROW` labels in world.slang). A narrow island
/// takes fewer dispatches per contact substep (three against four to seven), and its
/// contact-path bonds run in groups of their own, so only its chunk updates share its
/// group's threads.
fn wide_threshold() -> usize {
    std::env::var("STRESS_GPU_WIDE_CHUNKS").ok().and_then(|v| v.parse().ok()).unwrap_or(CONNECT_NARROW + 1)
}

/// world.slang `CONNECT_NARROW`: the chunks a narrow island's connectivity check holds.
const CONNECT_NARROW: usize = 512;
/// Scheduling knobs (GPU only, no physics change), read per call like
/// `STRESS_GPU_WIDE_CHUNKS`: they force rare scheduling paths in tests (`gate::Knobs`).
///
/// `STRESS_GPU_BATCH_ROUNDS=n`: at most `n` per-substep rounds per submission (1: a batch
/// boundary after every round).
fn batch_cap() -> usize {
    std::env::var("STRESS_GPU_BATCH_ROUNDS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0).map_or(MAX_BATCH_ROUNDS, |n| n.min(MAX_BATCH_ROUNDS))
}

/// The island kernel with each island's bonds over 512 threads (world.slang `BOND_GROUP`);
/// `STRESS_GPU_ISLAND_BONDS=256`: over the 256 its chunks use.
fn wide_island_bonds() -> bool {
    std::env::var("STRESS_GPU_ISLAND_BONDS").map_or(true, |v| v != "256")
}

/// Contact islands' bonds outside the island kernel (`STRESS_GPU_INLINE_BONDS=1`: inline).
fn external_bonds() -> bool {
    std::env::var("STRESS_GPU_INLINE_BONDS").map_or(true, |v| v != "1")
}

/// The contact pipeline in two dispatches per substep (`STRESS_GPU_FUSE=0`: three).
fn fuse_pipeline() -> bool {
    std::env::var("STRESS_GPU_FUSE").map_or(true, |v| v != "0")
}

/// The batch length after a split (and at the start), capped by `batch_cap`.
fn batch_start() -> usize {
    16.min(batch_cap())
}

/// `STRESS_GPU_PLAN_STEPS=x`: steps a contact plan covers (1: re-plan every step).
fn plan_steps() -> f64 {
    std::env::var("STRESS_GPU_PLAN_STEPS").ok().and_then(|v| v.parse::<f64>().ok()).filter(|&x| x > 0.0).unwrap_or(PLAN_STEPS)
}

/// `STRESS_GPU_BLOWUP_CHECK=<m/s>` (diagnostic, no physics change): report the mirror's
/// fastest chunk when above this speed, or with a hidden displacement over a metre or
/// turn over a radian, at the coupled frame's checkpoints (`GpuSolver::mirror_check`),
/// and each adaptive settle's static solve beside the reference's on the same state.
/// Read once.
pub fn blowup_limit() -> Option<f64> {
    static LIMIT: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| std::env::var("STRESS_GPU_BLOWUP_CHECK").ok().and_then(|v| v.parse::<f64>().ok()))
}

/// `STRESS_GPU_SYNC_EACH_STEP=1`: bring the mirror's bonds up to the GPU's after every
/// `step` instead of lazily.
fn sync_each_step() -> bool {
    std::env::var("STRESS_GPU_SYNC_EACH_STEP").is_ok_and(|v| v == "1")
}
const ISLAND_HALTED: u32 = 1;
const STOP_NOW: u32 = 1;

const GROUND_ENTRY: u32 = 0x8000_0000;

/// The parameters every kernel of `world.slang` shares (scratch offsets in float4).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    gravity: [f32; 4],
    dt: f32,
    fracture: u32,
    rigid_motion_loads: u32,
    step_start: u32,
    t_hi: f32,
    t_lo: f32,
    max_steps: u32,
    contact_mode: u32,
    halt_index: u32,
    chunk_count: u32,
    pair_count: u32,
    impactor_count: u32,
    cand_count: u32,
    probe_base: u32,
    probe_stride: u32,
    slot_base: u32,
    ledger_base: u32,
    record_base: u32,
    record_stride: u32,
    cand_base: u32,
    pair_index: u32,
    cand_index: u32,
    zeta: f32,
    pair_friction: f32,
    ground_hi: f32,
    ground_lo: f32,
    ground_friction: f32,
    ground_modulus: f32,
    has_ground: u32,
    wide_bond_groups: u32,
    wide_bond_table: u32,
    wide_chunk_table: u32,
    wide_base: u32,
    seg_index: u32,
    seg_count: u32,
    seg_base: u32,
    solve_mode: u32,
    cload_base: u32,
    cframe_base: u32,
    statics_base: u32,
    statics_bonds: u32,
    statics_newton: u32,
    statics_cg: u32,
    statics_tol: f32,
    /// Lanes per candidate pair in `contact_forces`: a SIMD group of 32 on the Metal
    /// shaders (lanes over the pair's sample points), one thread on WGSL.
    pair_lanes: u32,
    /// Islands check a disconnection's connectivity themselves (0:
    /// `STRESS_GPU_DEVICE_CONNECT=0`, always the host's).
    device_connect: u32,
    /// Scratch: each chunk's contact plan (start position hi with its budget, lo with
    /// the plan time's rounding, orientation, velocity with the plan time), 4 float4.
    cplan_base: u32,
    /// Scratch: each impactor's contact plan, 4 float4 as a chunk's (`cplan_base`); a zero
    /// budget: none (driven).
    iplan_base: u32,
    /// Narrow contact islands' bond groups run in the sums dispatch this batch (0: their
    /// bonds run inline in the island kernel), and their table in `index`.
    contact_bond_groups: u32,
    contact_bond_table: u32,
    /// Scratch: each chunk's external load for the static solves (2 float4, contiguous
    /// so a cluster's loads upload in one write).
    sload_base: u32,
}




#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuImpactor {
    position: [f32; 4],
    position_err: [f32; 4],
    velocity: [f32; 4],
    velocity_err: [f32; 4],
    angular_velocity: [f32; 4],
    rotation: [f32; 4],
    inertia: [[f32; 4]; 3],
    inv: [[f32; 4]; 3],
    shape: [f32; 4],
    half: [f32; 4],
    mat: [f32; 4],
    crush: [f32; 4],
    geom: [f32; 4],
    rotation_err: [f32; 4],
    ledger: [f32; 4],
    cand: [u32; 4],
    momentum: [f32; 4],
    momentum_err: [f32; 4],
}


fn damping_ratio(restitution: f64) -> f64 {
    let l = restitution.max(1e-6).ln();
    -l / (std::f64::consts::PI.powi(2) + l * l).sqrt()
}

/// The GPU form of a load term (5 float4) and its table data.
fn pack_term(t: &LoadTerm, chunk: u32, data: &mut Vec<[f32; 4]>, data_base: u32) -> [[f32; 4]; 5] {
    let (fn_kind, origin, p, offset): (u32, f64, [f32; 4], u32) = match &t.function {
        Function::Constant(_) => (0, 0.0, [0.0; 4], 0),
        Function::Ramp { t0, t1, value } => (1, *t0, [(t1 - t0) as f32, *value as f32, 0.0, 0.0], 0),
        Function::HalfSine { start, duration, peak } => (2, *start, [*duration as f32, *peak as f32, 0.0, 0.0], 0),
        Function::Friedlander { arrival, peak, duration, decay } => (3, *arrival, [*peak as f32, *duration as f32, *decay as f32, 0.0], 0),
        Function::Table(points) => {
            let origin = points.first().map(|p| p[0]).unwrap_or(0.0);
            let offset = data_base + data.len() as u32;
            for p in points {
                data.push([(p[0] - origin) as f32, p[1] as f32, 0.0, 0.0]);
            }
            if points.is_empty() {
                data.push([0.0; 4]);
            }
            (4, origin, [f32::from_bits(points.len().max(1) as u32), 0.0, 0.0, 0.0], offset)
        }
        Function::Blast(fb) => (5, fb.arrival, [fb.duration as f32, fb.decay as f32, fb.peak as f32, fb.stagnation as f32], 0),
        Function::Replacement { start, duration } => (6, *start, [*duration as f32, 0.0, 0.0, 0.0], 0),
        Function::Pulse { start, plateau, plateau_time, peak, duration } => (7, *start, [*plateau as f32, *plateau_time as f32, *peak as f32, *duration as f32], 0),
    };
    let (o_hi, o_lo) = split1(origin);
    let constant = match &t.function {
        Function::Constant(v) => *v as f32,
        _ => 0.0,
    };
    let clearing = match &t.function {
        Function::Blast(fb) => fb.clearing_time as f32,
        _ => 0.0,
    };
    [
        [f32::from_bits(chunk), f32::from_bits(t.kind), f32::from_bits(fn_kind), f32::from_bits(offset)],
        v4(t.dir, t.area),
        v4(t.arm, 0.0),
        [o_hi, o_lo, constant, clearing],
        p,
    ]
}

const K_ISLAND: usize = 0;
const K_FORCES: usize = 1;
const K_SHARES: usize = 2;
const K_CRUSH: usize = 3;
const K_WIDE: [usize; 5] = [5, 6, 7, 8, 9];
const K_SUMS: usize = 10;
const K_WAKE: usize = 11;
/// `contact_sums` and `impactor_integrate` in one dispatch.
const K_SUMS_INTEGRATE: usize = 12;
/// `island_frame` and `wide_chunks` in one dispatch.
const K_ISLAND_WIDE: usize = 13;
/// `contact_sums`, `impactor_integrate` and the wide bond groups in one dispatch.
const K_SUMS_BONDS: usize = 14;
/// The contact path's narrow islands without their bonds (run in the sums dispatch), and
/// that with the wide islands' chunks.
const K_CONTACT_ISLANDS: usize = 15;
const K_CONTACT_ISLAND_WIDE: usize = 16;
/// The contact pipeline in two dispatches: the forces with the narrow contact islands'
/// bonds, then the islands (with their contact sums) with the impactors' integration.
const K_FORCES_BONDS: usize = 17;
const K_ISLANDS_SUMS: usize = 18;
/// Its travel checks before a batch's first substep (later ones run at each substep's end).
const K_TRAVEL: usize = 19;
/// `K_ISLANDS_SUMS` able to wake a settled contact island (with its joint law).
const K_ISLANDS_SUMS_WAKE: usize = 20;
/// Kernel names by `K_*` index.
pub const KERNEL_NAMES: [&str; 21] = [
    "island_frame",
    "contact_forces",
    "impactor_shares",
    "impactor_crush",
    "impactor_integrate",
    "wide_bonds",
    "wide_chunks",
    "wide_drift",
    "wide_rigid",
    "wide_end",
    "contact_sums",
    "wide_wake",
    "sums_and_impactors",
    "islands_and_wide_chunks",
    "sums_impactors_bonds",
    "contact_islands",
    "contact_islands_and_wide_chunks",
    "forces_and_bonds",
    "islands_sums_impactors",
    "travel_start",
    "islands_sums_impactors_wake",
];
/// Contact contributions per segment (`contact_sums`, one thread each).
const SEGMENT: usize = 8;
/// Steps a contact plan covers (re-planned sooner when a chunk leaves its budget;
/// `STRESS_GPU_PLAN_STEPS` overrides, `plan_steps`).
const PLAN_STEPS: f64 = 4.0;
/// Most per-substep rounds per submission.
const MAX_BATCH_ROUNDS: usize = 2048;
/// Dispatches per submission while profiling (2 timestamps each).
const TIMED_DISPATCHES: usize = 2048;

/// GPU timestamps of every dispatch (`STRESS_GPU_TIMING=1`, where supported).
struct Timing {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
}

struct Kernels {
    island: Kernel,
    forces: Kernel,
    shares: Kernel,
    crush: Kernel,
    integrate: Kernel,
    wide: [Kernel; 5],
    sums: Kernel,
    wake: Kernel,
    sums_integrate: Kernel,
    island_wide: Kernel,
    sums_bonds: Kernel,
    contact_islands: Kernel,
    contact_island_wide: Kernel,
    forces_bonds: Kernel,
    islands_sums: Kernel,
    travel: Kernel,
    islands_sums_wake: Kernel,
    statics: Kernel,
    fatigue: Kernel,
}

/// The bindings of every `world.slang` kernel.
const WORLD_BINDINGS: [Binding; 12] = [
    Binding::Uniform,   // params
    Binding::Uniform,   // materials
    Binding::Storage,   // bonds
    Binding::Storage,   // chunks
    Binding::Storage,   // index
    Binding::Storage,   // loads
    Binding::StorageRw, // state
    Binding::StorageRw, // bond_dyn
    Binding::StorageRw, // scratch
    Binding::StorageRw, // islands
    Binding::StorageRw, // impactors
    Binding::StorageRw, // contact_state
];

struct Buffers {
    /// Static bond data and the material table (unchanged while the topology is).
    bonds_static: wgpu::Buffer,
    materials: wgpu::Buffer,
    /// Dispatch params: segment mode (contact_mode 0) and per-substep mode (1).
    params_segment: wgpu::Buffer,
    params_contact: wgpu::Buffer,
    /// The static part of the params (offsets, counts, contact constants).
    params: Params,
    state: wgpu::Buffer,
    bond_dyn: wgpu::Buffer,
    scratch: wgpu::Buffer,
    islands: wgpu::Buffer,
    impactors: wgpu::Buffer,
    contact_state: wgpu::Buffer,
    /// Rebuilt by every `build`; kept to be rewritten in place when the size holds.
    chunk_static: wgpu::Buffer,
    index: wgpu::Buffer,
    loads: wgpu::Buffer,
    bind_segment: wgpu::BindGroup,
    bind_contact: wgpu::BindGroup,
    /// Some impactor has a crush cap (the crush passes run each substep).
    crush: bool,
    /// Chunk groups of the wide islands (0: none).
    wide_chunk_groups: u32,
    /// Some wide island is free (not anchored): its rounds need the drift and rigid
    /// passes, which anchored islands skip.
    wide_free: bool,
    /// Some wide island is settled (its loads checked by `wide_wake` every substep). The
    /// build never makes a settled island wide, so this is false today and `wide_wake` is
    /// not dispatched.
    wide_settled: bool,
    /// Bond groups of the narrow contact islands (for the sums dispatch), and whether
    /// one of them is settled (its wake check must precede its bonds: they run inline).
    contact_bond_groups: u32,
    contact_settled: bool,
}

/// The chunk and bond tables as uploaded (`Layout::host`), and the chunk-to-bond
/// incidence's length at the start of `index`.
struct HostTables {
    chunk_static: Vec<stress_layout::ChunkStatic>,
    bond_static: Vec<stress_layout::BondStatic>,
    /// The incidence (the start of `index`).
    csr: Vec<u32>,
}

/// The resident cascade's repeat gate (cascade.slang) for one layout: its buffer, and the
/// clusters, island ranges and chunks it was made for (entries follow their clusters
/// across layouts).
struct CascadeGate {
    generation: u64,
    buffer: wgpu::Buffer,
    ids: Vec<u64>,
    ranges: Vec<[u32; 4]>,
    chunks: Vec<(usize, usize)>,
}

/// Where each GPU chunk and bond lives in the mirror, the probe items by slot, the
/// contact plan.
struct Layout {
    /// Islands the buffers hold room for (`island_headroom` past the layout's).
    island_cap: usize,
    /// With island headroom: the static tables as uploaded, for a split in place.
    host: Option<HostTables>,
    chunks: Vec<(usize, usize)>,
    bonds: Vec<(usize, usize)>,
    islands: Vec<Island>,
    items: Vec<ProbeItem>,
    plan: Plan,
    /// Per pair: its sample-point entries in the contact state (offset, count).
    pair_state: Vec<(u32, u32)>,
    gpu_impactors: Vec<GpuImpactor>,
}

/// Host-side time spent per activity (seconds) and counts, for profiling.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub build: f64,
    pub gpu: f64,
    pub readback: f64,
    pub download: f64,
    pub split: f64,
    /// Host time (s) writing a batch's islands and params and encoding its dispatches.
    pub encode: f64,
    pub batches: u64,
    pub substeps: u64,
    /// Per-substep rounds encoded, and those some island or impactor ran (the rest
    /// launch empty after a halt).
    pub rounds_encoded: u64,
    pub rounds_used: u64,
    /// Largest contact plan seen: pair candidates, impactor candidates.
    pub max_pairs: usize,
    pub max_impactor_candidates: usize,
    /// Longest contact contribution list of a chunk, and of an island (sum over its chunks).
    pub max_chunk_contacts: usize,
    pub max_island_contacts: usize,
    /// GPU time (s) and dispatches per kernel (`KERNEL_NAMES`), with `STRESS_GPU_TIMING=1`.
    pub kernels: [f64; 21],
    pub kernel_dispatches: [u64; 21],
    /// `tangent_statics`: static solves delegated to the mirror (`equilibrate`), and the
    /// host time they took (s).
    pub statics_delegated: u64,
    pub statics_delegated_seconds: f64,
    /// `STRESS_GPU_CHECK_STATICS=1`: static solves compared with the oracle's, and those
    /// whose outcome differed.
    pub statics_checked: u64,
    pub statics_mismatched: u64,
}

/// The solver's buffers a renderer reads (`GpuSolver::render_buffers`), in the binding
/// order of `destruction_render.slang` after its camera.
pub struct RenderBuffers<'a> {
    pub chunks: &'a wgpu::Buffer,
    pub islands: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    pub index: &'a wgpu::Buffer,
    pub bond_dyn: &'a wgpu::Buffer,
    pub impactors: &'a wgpu::Buffer,
    pub chunk_count: u32,
    pub impactor_count: u32,
}

/// The host copy of everything a frame changes (`GpuSolver::snapshot`): to solve the
/// frame again after an engine-coupled frame's verification found a body the engine
/// moved fractured (world_coupling.rs `step_frame_coupled`).
#[derive(Clone)]
pub struct HostSnapshot {
    mirror: ReferenceSolver,
    /// Structures left out of `mirror` (their chunk, bond and adjacency tables are empty
    /// there): every cluster of theirs was a quiet cluster off the GPU, so nothing a
    /// segment does can change them, and `restore` takes them from the live mirror.
    omitted: Vec<usize>,
    impactors: Vec<Impactor>,
    contact_dissipated: f64,
    crush_energy: f64,
    last_loads: ChunkLoads,
    frame_load_sum: ChunkLoads,
    pair_memory: HashMap<PairKey, PairState>,
    /// Resident segments: the chunk state and bond records, copied on the GPU.
    gpu_state: Option<(wgpu::Buffer, wgpu::Buffer)>,
}

impl HostSnapshot {
    /// The mirror at the snapshot.
    pub fn mirror(&self) -> &ReferenceSolver {
        &self.mirror
    }
}

pub struct GpuSolver {
    /// The host mirror: topology and history, current after `download`.
    pub mirror: ReferenceSolver,
    /// The scene's loads and probes (none: gravity only, no probes).
    pub loads: Option<WorldLoads>,
    /// Impactors (host mirror) and contact energies.
    pub impactors: Vec<Impactor>,
    pub contact_dissipated: f64,
    /// External chunk loads (world.rs `loads`, adaptive and quasi-static modes): the last
    /// substep's, and their sum since `frame_load_sum` was last cleared.
    pub last_loads: ChunkLoads,
    pub frame_load_sum: ChunkLoads,
    pub crush_energy: f64,
    /// Pre-existing overlap of the pairs in contact (world.rs `pair_offsets`).
    pair_memory: HashMap<PairKey, PairState>,
    /// Ids of the quiet clusters left out of the chunk and bond tables (`quiet_clusters`):
    /// chosen by `refresh`, kept by every other build.
    excluded: Vec<u64>,
    /// The mirror's chunk state and cluster motion equal the GPU's (after a download,
    /// before the next compute submission).
    state_current: bool,
    kernels: Kernels,
    timing: Option<Timing>,
    layout: Layout,
    buffers: Buffers,
    /// Time after each absolute substep (index: substep count since the start).
    step_times: Vec<f64>,
    /// GPU submissions, host splits and contact re-plans so far.
    pub dispatches: u64,
    pub host_splits: u64,
    /// Halts whose disconnections left the cluster whole (resumed on the GPU).
    pub resumed_halts: u64,
    /// The GPU's bond history is ahead of the mirror's (synced lazily, `sync`).
    bonds_stale: bool,
    /// Time until which the current contact plan covers the motion.
    plan_until: f64,
    /// Load terms besides the scene's (an engine's contact loads, `coupled`), applied
    /// from the next build.
    pub extra_terms: Vec<LoadTerm>,
    /// Bumped by every new layout (`build`): tables keyed by GPU chunk follow it.
    pub layout_generation: u64,
    /// Islands the tables leave room for past the layout's own (0: none), so a split can
    /// be laid out in place (`split_resident`).
    pub island_headroom: usize,
    /// Settled islands' chunk state changes on the GPU too (their static solves run
    /// there: the resident frame), so a sync reads it back.
    pub state_on_gpu: bool,
    /// Island segments resident (opt-in): re-plans from the GPU's own plan
    /// (`resident_plan`), segment snapshots without the chunk state or bond download
    /// (their buffers copied on the GPU instead).
    pub resident_contacts: bool,
    /// The layout was changed in place (`split_resident`): its order is not a fresh
    /// pack's, so a refresh syncs and rebuilds.
    layout_in_place: bool,
    /// Whether the solver resolves contacts itself (standalone world); off under an
    /// engine, which owns them.
    pub own_contacts: bool,
    /// Per-substep rounds per submission: small after a split, doubling while none
    /// occurs (kept across steps).
    batch_rounds: usize,
    pub replans: u64,
    /// When set, every contact plan built inside `step` (re-plans, host splits) is
    /// recorded here, for a step bound to be checked against them.
    pub plan_log: Option<Vec<crate::contacts::PlanView>>,
    /// When set, `step` returns as soon as it makes a contact plan within the step (a
    /// re-plan or a host split) while every island stands at the same substep: nothing has
    /// run under the plan yet, so the caller may pick the step it needs.
    /// `stopped_at_plan` tells whether the last `step` did.
    pub stop_at_plan: bool,
    /// Engine-coupled quasi-static mode: clusters whose last static solve repeated and
    /// whose damage commit changed nothing (sorted ids; `GpuWorld::solve_static_all`),
    /// the only ones it may leave out as quiet.
    pub static_stable: Vec<u64>,
    /// Quasi-static mode under an engine (`GpuWorld::with_engine`): quiet clusters (by
    /// `static_stable`) may be left out.
    pub quiet_quasi_static: bool,
    /// `equilibrate_resident`'s copy of the state from before a solve.
    state_backup: Option<wgpu::Buffer>,
    /// The resident cascade's kernels, status record and the layout's island records
    /// (by layout generation), made on first use (`resident_cascade`).
    cascade: Option<crate::resident::CascadeKernels>,
    /// The GPU contact planner (`gpu_plan.rs`) and how its plans compared with the host's.
    planner: Option<crate::gpu_plan::GpuPlanner>,
    pub plan_check: crate::gpu_plan::PlanCheck,
    cascade_islands: Option<(u64, wgpu::Buffer)>,
    cascade_status: Option<wgpu::Buffer>,
    /// The repeat gate's loads and flags, by layout generation (cascade.slang).
    cascade_gate: Option<CascadeGate>,
    bond_backup: Option<wgpu::Buffer>,
    pub stopped_at_plan: bool,
    pub profile: Profile,
    /// The code paths that ran so far (`coverage`): the device's words OR-ed after every
    /// readback, and the host's rare paths. Tests only; nothing reads it.
    pub coverage: u64,
}

impl GpuSolver {
    pub fn new(gpu: &Gpu, mirror: ReferenceSolver, loads: Option<WorldLoads>, impactors: Vec<Impactor>) -> Result<GpuSolver, String> {
        let kernels = Kernels {
            island: gpu.compute(&shaders::WORLD, if wide_island_bonds() { "island_frame_bonds_wide" } else { "island_frame" }, &WORLD_BINDINGS),
            forces: gpu.compute(&shaders::WORLD, "contact_forces", &WORLD_BINDINGS),
            shares: gpu.compute(&shaders::WORLD, "impactor_shares", &WORLD_BINDINGS),
            crush: gpu.compute(&shaders::WORLD, "impactor_crush", &WORLD_BINDINGS),
            integrate: gpu.compute(&shaders::WORLD, "impactor_integrate", &WORLD_BINDINGS),
            wide: ["wide_bonds", "wide_chunks", "wide_drift", "wide_rigid", "wide_end"].map(|e| gpu.compute(&shaders::WORLD, e, &WORLD_BINDINGS)),
            sums: gpu.compute(&shaders::WORLD, "contact_sums", &WORLD_BINDINGS),
            wake: gpu.compute(&shaders::WORLD, "wide_wake", &WORLD_BINDINGS),
            sums_integrate: gpu.compute(&shaders::WORLD, "sums_and_impactors", &WORLD_BINDINGS),
            island_wide: gpu.compute(&shaders::WORLD, "islands_and_wide_chunks", &WORLD_BINDINGS),
            sums_bonds: gpu.compute(&shaders::WORLD, "sums_impactors_bonds", &WORLD_BINDINGS),
            contact_islands: gpu.compute(&shaders::WORLD, "contact_islands", &WORLD_BINDINGS),
            contact_island_wide: gpu.compute(&shaders::WORLD, "contact_islands_and_wide_chunks", &WORLD_BINDINGS),
            forces_bonds: gpu.compute(&shaders::WORLD, "forces_and_bonds", &WORLD_BINDINGS),
            islands_sums: gpu.compute(&shaders::WORLD, "islands_sums_impactors", &WORLD_BINDINGS),
            travel: gpu.compute(&shaders::WORLD, "travel_start", &WORLD_BINDINGS),
            islands_sums_wake: gpu.compute(&shaders::WORLD, "islands_sums_impactors_wake", &WORLD_BINDINGS),
            statics: gpu.compute(&shaders::WORLD, "island_statics", &WORLD_BINDINGS),
            fatigue: gpu.compute(&shaders::WORLD, "settled_fatigue", &WORLD_BINDINGS),
        };
        let timing = (std::env::var("STRESS_GPU_TIMING").is_ok_and(|v| v == "1") && gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY)).then(|| Timing {
            queries: gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 * TIMED_DISPATCHES as u32 }),
            resolve: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamps"),
                size: 16 * TIMED_DISPATCHES as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        });
        let step_times = vec![mirror.time];
        let mut loads = loads;
        let pair_memory = HashMap::new();
        let (layout, buffers) = Self::build(gpu, &kernels, &mirror, loads.as_mut(), &impactors, &pair_memory, 4, 0.0, mirror.time, None, &[], &[], 0, None)?;
        Ok(GpuSolver {
            mirror,
            loads,
            impactors,
            contact_dissipated: 0.0,
            last_loads: ChunkLoads::default(),
            frame_load_sum: ChunkLoads::default(),
            crush_energy: 0.0,
            pair_memory,
            excluded: Vec::new(),
            state_current: true,
            kernels,
            timing,
            layout,
            buffers,
            step_times,
            dispatches: 0,
            host_splits: 0,
            resumed_halts: 0,
            bonds_stale: false,
            plan_log: None,
            stop_at_plan: false,
            static_stable: Vec::new(),
            quiet_quasi_static: false,
            state_backup: None,
            cascade: None,
            planner: None,
            plan_check: Default::default(),
            cascade_islands: None,
            cascade_status: None,
            cascade_gate: None,
            bond_backup: None,
            stopped_at_plan: false,
            plan_until: f64::NEG_INFINITY,
            extra_terms: Vec::new(),
            layout_generation: 0,
            island_headroom: 0,
            state_on_gpu: false,
            resident_contacts: false,
            layout_in_place: false,
            own_contacts: true,
            batch_rounds: batch_start(),
            replans: 0,
            profile: Profile::default(),
            coverage: 0,
        })
    }

    /// Rebuild every buffer from the mirror (after the host changed topology or loads),
    /// planning contacts for the next `horizon` seconds.
    /// The mirror must be current (`sync` before changing it): everything is uploaded
    /// from it.
    pub fn rebuild(&mut self, gpu: &Gpu, horizon: f64) -> Result<(), String> {
        if self.bonds_stale {
            return Err("rebuild from a mirror whose bonds lag the GPU (sync first)".into());
        }
        let stride = self.buffers.params.probe_stride;
        self.plan_until = self.mirror.time + horizon;
        let (layout, buffers) = Self::build(gpu, &self.kernels, &self.mirror, self.loads.as_mut(), &self.impactors, &self.pair_memory, stride, horizon, self.mirror.time, None, &self.extra_terms, &self.excluded, self.island_headroom, None)?;
        self.layout_in_place = false;
        self.layout = layout;
        self.layout_generation += 1;
        self.buffers = buffers;
        Ok(())
    }

    /// Contacts planned afresh for the next `horizon` seconds from the current state,
    /// keeping the GPU's state and bond history resident (as `step` re-plans).
    pub fn plan_for(&mut self, gpu: &Gpu, horizon: f64) -> Result<(), String> {
        let stride = self.buffers.params.probe_stride;
        let horizon = if self.contacts_possible() { horizon } else { 0.0 };
        self.plan_until = self.mirror.time + horizon;
        // `STRESS_GPU_PLAN_CHECK=1`: the GPU's plan from the same state, to compare.
        let gpu_plan = (horizon > 0.0 && std::env::var("STRESS_GPU_PLAN_CHECK").is_ok_and(|v| v == "1")).then(|| (self.gpu_plan(gpu, horizon), self.layout.chunks.clone()));
        let resident = (self.resident_contacts && horizon > 0.0).then(|| self.resident_plan(gpu, horizon));
        let (layout, buffers) =
            Self::build(gpu, &self.kernels, &self.mirror, self.loads.as_mut(), &self.impactors, &self.pair_memory, stride, horizon, self.mirror.time, Some(&self.buffers), &self.extra_terms, &self.excluded, self.island_headroom, resident.as_ref())?;
        self.layout = layout;
        self.layout_generation += 1;
        self.buffers = buffers;
        if let Some((out, chunks)) = gpu_plan {
            self.compare_plans(&out, &chunks);
        }
        Ok(())
    }

    /// The contact plan over `horizon` made on the GPU from its current state
    /// (`gpu_plan.rs`), read back.
    pub fn gpu_plan(&mut self, gpu: &Gpu, horizon: f64) -> crate::gpu_plan::GpuPlanOut {
        if self.planner.is_none() {
            self.planner = Some(crate::gpu_plan::GpuPlanner::new(gpu));
        }
        let m = &self.mirror;
        let owned: Vec<bool> = m.clusters.iter().map(|c| !c.driven).collect();
        let anchored: Vec<bool> = m.clusters.iter().map(|c| c.anchored).collect();
        let sorted: Vec<Vec<u32>> = self
            .layout
            .islands
            .iter()
            .map(|isl| {
                let mut ks: Vec<u32> = (isl.range[0]..isl.range[1]).collect();
                ks.sort_by_key(|&k| self.layout.chunks[k as usize]);
                ks
            })
            .collect();
        let g = m.config.gravity;
        let ground = self.scene().and_then(|s| s.ground.as_ref().map(|g| g.height));
        self.planner.as_ref().expect("planner").plan(
            gpu,
            (&self.buffers.chunk_static, &self.buffers.islands, &self.buffers.state, &self.buffers.impactors),
            self.layout.chunks.len(),
            &owned,
            &anchored,
            &sorted,
            self.impactors.len(),
            horizon,
            [g.x, g.y, g.z],
            ground,
        )
    }

    fn compare_plans(&mut self, out: &crate::gpu_plan::GpuPlanOut, chunks: &[(usize, usize)]) {
        use std::collections::HashSet;
        let plan = &self.layout.plan;
        let key = |k: u32| chunks[k as usize];
        let nkey = |k: u32| self.layout.chunks[k as usize];
        let c = &mut self.plan_check;
        c.plans += 1;
        let host: HashSet<((usize, usize), (usize, usize))> = plan.pairs.iter().map(|p| (nkey(p.1), nkey(p.2))).collect();
        let gpu: HashSet<((usize, usize), (usize, usize))> = out.pairs.iter().map(|&(a, b)| (key(a), key(b))).collect();
        c.host_pairs += host.len() as u64;
        c.missed_pairs += host.difference(&gpu).count() as u64;
        let host_seq: Vec<((usize, usize), (usize, usize))> = plan.pairs.iter().map(|p| (nkey(p.1), nkey(p.2))).collect();
        let gpu_seq: Vec<((usize, usize), (usize, usize))> = out.pairs.iter().map(|&(a, b)| (key(a), key(b))).collect();
        if host == gpu && host_seq != gpu_seq {
            c.order_mismatch += 1;
        }
        c.extra_pairs += gpu.difference(&host).count() as u64;
        let hg: HashSet<(usize, usize)> = plan.ground.iter().map(|&k| nkey(k)).collect();
        let gg: HashSet<(usize, usize)> = (0..out.ground.len()).filter(|&k| out.ground[k]).map(|k| key(k as u32)).collect();
        c.missed_ground += hg.difference(&gg).count() as u64;
        c.extra_ground += gg.difference(&hg).count() as u64;
        for (i, list) in plan.impactor_chunks.iter().enumerate() {
            let hi: HashSet<(usize, usize)> = list.iter().map(|&k| nkey(k)).collect();
            let gi: HashSet<(usize, usize)> = out.impactor.get(i).map(|f| (0..f.len()).filter(|&k| f[k]).map(|k| key(k as u32)).collect()).unwrap_or_default();
            c.missed_impactor += hi.difference(&gi).count() as u64;
            c.extra_impactor += gi.difference(&hi).count() as u64;
        }
        let host_rot: HashMap<(usize, usize), bool> = plan.rotation_checked.iter().enumerate().map(|(k, &r)| (nkey(k as u32), r)).collect();
        let host_budget: HashMap<(usize, usize), f64> = plan.budget.iter().enumerate().map(|(k, &b)| (nkey(k as u32), b)).collect();
        for k in 0..out.budget.len() {
            let sc = key(k as u32);
            if host_rot.get(&sc).copied().unwrap_or(false) != out.rotation_checked[k] {
                c.rotation_mismatch += 1;
            }
            if let Some(&b) = host_budget.get(&sc) {
                c.budget = c.budget.max((out.budget[k] as f64 - b).abs() / b.abs().max(1e-12));
            }
        }
    }

    /// The contact plan `plan_for(horizon)` would make, on the host only (nothing is
    /// built or uploaded): for weighing horizons before building one.
    pub fn plan_preview(&self, horizon: f64) -> crate::contacts::PlanView {
        let chunks = &self.layout.chunks;
        let plan = match self.scene() {
            Some(scene) if self.contacts_possible() && horizon > 0.0 => crate::contacts::plan(scene, &self.mirror, &self.impactors, chunks, horizon),
            _ => crate::contacts::Plan::default(),
        };
        crate::contacts::PlanView::of(&plan, chunks)
    }

    /// The current contact plan by (structure, chunk).
    pub fn plan_view(&self) -> crate::contacts::PlanView {
        crate::contacts::PlanView::of(&self.layout.plan, &self.layout.chunks)
    }

    fn log_plan(&mut self) {
        if self.plan_log.is_some() {
            let view = self.plan_view();
            self.plan_log.as_mut().unwrap().push(view);
        }
    }

    fn scene(&self) -> Option<&Scene> {
        self.loads.as_ref().map(|l| &l.scene)
    }

    /// Every buffer from the mirror's current state.
    #[allow(clippy::too_many_arguments)]
    fn build(
        gpu: &Gpu,
        kernels: &Kernels,
        m: &ReferenceSolver,
        mut loads: Option<&mut WorldLoads>,
        impactors: &[Impactor],
        pair_memory: &HashMap<PairKey, PairState>,
        probe_stride: u32,
        horizon: f64,
        plan_time: f64,
        keep: Option<&Buffers>,
        extra_terms: &[LoadTerm],
        excluded: &[u64],
        island_headroom: usize,
        resident: Option<&ResidentPlan>,
    ) -> Result<(Layout, Buffers), String> {
        if m.config.mode == SolveMode::Implicit {
            return Err("the implicit solve mode is not on the GPU yet".into());
        }
        let prof = std::env::var("STRESS_GPU_BUILD_PROFILE").is_ok();
        let t_build = std::time::Instant::now();
        let mut t_last = 0.0f64;
        let mut mark = |what: &str| {
            if prof {
                let now = t_build.elapsed().as_secs_f64() * 1e3;
                eprintln!("build {what}: {:.1} ms", now - t_last);
                t_last = now;
            }
        };
        let (mut terms, items) = match loads {
            Some(ref mut l) => (l.terms(m, m.time), l.probe_items(m)),
            None => (Vec::new(), Vec::new()),
        };
        mark("terms");
        let scene: Option<&Scene> = loads.as_deref().map(|l| &l.scene);
        terms.extend_from_slice(extra_terms);
        let StructureTables { chunk_index, chunks, mut chunk_static, state, mut islands, table, bonds, bond_index, bond_static, bond_dyn, csr } = stress_layout::pack_with(m, &m.clusters.iter().map(|cl| excluded.binary_search(&cl.id).is_ok()).collect::<Vec<_>>())?;
        mark("pack");
        let n = chunks.len();

        // Load terms by GPU chunk (stable: the reference's order within a chunk), then
        // table data, then probe items grouped by island.
        let mut by_chunk: Vec<Vec<&LoadTerm>> = vec![Vec::new(); n];
        for t in &terms {
            let k = chunk_index.get(t.structure).and_then(|v| v.get(t.chunk)).copied().unwrap_or(u32::MAX);
            if k != u32::MAX {
                by_chunk[k as usize].push(t);
            }
        }
        let term_count: usize = by_chunk.iter().map(|v| v.len()).sum();
        let data_base = (5 * term_count) as u32;
        let mut data = Vec::new();
        let mut packed: Vec<[f32; 4]> = Vec::with_capacity(5 * term_count);
        for (k, list) in by_chunk.iter().enumerate() {
            let begin = (packed.len() / 5) as u32;
            for t in list {
                packed.extend(pack_term(t, k as u32, &mut data, data_base));
            }
            chunk_static[k].load_range = [begin, (packed.len() / 5) as u32, 0, 0];
        }
        packed.extend(data);
        let item_island = |it: &ProbeItem| -> Option<(u32, u32)> {
            match it.target {
                Target::Chunk(s, c) => {
                    let k = chunk_index.get(s).and_then(|v| v.get(c)).copied().unwrap_or(u32::MAX);
                    (k != u32::MAX).then(|| (chunk_static[k as usize].info[1], k))
                }
                Target::Bond(s, b) => bond_index.get(s).and_then(|v| v.get(b)).copied().filter(|&i| i != u32::MAX).map(|i| (bond_static[i as usize].law.ids[3], i)),
            }
        };
        let mut placed: Vec<(u32, u32, ProbeItem)> = items.into_iter().filter_map(|it| item_island(&it).map(|(isl, idx)| (isl, idx, it))).collect();
        placed.sort_by_key(|p| p.0);
        let mut layout_items = Vec::new();
        let items_base = packed.len() as u32;
        let mut cursor = 0;
        for (ci, island) in islands.iter_mut().enumerate() {
            let begin = items_base + 4 * layout_items.len() as u32;
            while cursor < placed.len() && placed[cursor].0 == ci as u32 {
                let (_, idx, it) = &placed[cursor];
                let slot = layout_items.len() as u32;
                packed.push([f32::from_bits(it.kind), f32::from_bits(*idx), f32::from_bits(it.side), f32::from_bits(slot)]);
                // Displacement items: the chunk centre's f32 rounding error (body frame) in
                // the w lanes, so the probe resolves displacements far below the ulp of
                // the centre (world.slang `record_probes`).
                let centre_lo = match it.target {
                    Target::Chunk(s, c) if it.kind == crate::loads::PROBE_DISPLACEMENT => {
                        let x = m.structures[s].chunks[c].center;
                        x - vec3(v4(x, 0.0))
                    }
                    _ => Vec3::ZERO,
                };
                packed.push(v4(it.a, centre_lo.x));
                let (mut hi, mut lo) = split(it.x0);
                hi[3] = centre_lo.y as f32;
                lo[3] = centre_lo.z as f32;
                packed.push(if it.kind == crate::loads::PROBE_SECTION_BOND { v4(it.b, 0.0) } else { hi });
                packed.push(lo);
                layout_items.push(it.clone());
                cursor += 1;
            }
            island.probes = [begin, items_base + 4 * layout_items.len() as u32, 0, 0];
        }
        // Convex hulls (chunk frame): vertices, face planes (normal, offset), face
        // centroids; each chunk's at `cmat.z`, counts in `cmat.w` (vertices | faces << 8).
        let mut hull_of = vec![(0u32, 0u32); n];
        for (k, &(s, c)) in chunks.iter().enumerate() {
            if let Some(h) = &m.structures[s].chunks[c].hull {
                let at = packed.len() as u32;
                packed.extend(h.vertices.iter().map(|v| v4(*v, 0.0)));
                packed.extend(h.faces.iter().map(|f| v4(f.normal, f.offset)));
                packed.extend(h.faces.iter().map(|f| v4(f.centroid, 0.0)));
                hull_of[k] = (at, h.vertices.len() as u32 | (h.faces.len() as u32) << 8);
            }
        }
        if packed.is_empty() {
            packed.push([0.0; 4]);
        }

        mark("loads+probes+hulls");
        // ---------------------------------------------------------------- contacts
        let plan = match (&scene, resident) {
            // A plan made on the GPU from its own state (a resident segment).
            (_, Some(r)) => r.plan.clone(),
            (Some(scene), None) if horizon > 0.0 && (m.clusters.len() > 1 || !impactors.is_empty() || scene.ground.is_some()) => plan(scene, m, impactors, &chunks, horizon),
            _ => Plan { contact_clusters: vec![false; m.clusters.len()], budget: vec![0.0; n], ..Default::default() },
        };
        // With any contact or impactor, every island advances in lockstep through the
        // per-substep pipeline (a split then stops all of them at the same substep); only
        // the planned chunks receive contact loads.
        // Impactors the solver integrates (under an engine, those in impact islands).
        let pipeline = impactors.iter().any(|i| !i.driven) || plan.contact_clusters.iter().any(|&c| c);
        if pipeline {
            for island in islands.iter_mut() {
                island.info[0] |= ISLAND_CONTACT;
            }
        }
        // Output slots (force, torque): 2 per pair (side a, side b), 1 per impactor
        // candidate. Index data after the CSR: pair candidates, impactor candidates,
        // then each chunk's contributions in the reference's order (impactors, ground,
        // chunk pairs).
        let pair_slots = 2 * plan.pairs.len();
        let cand_count: usize = plan.impactor_chunks.iter().map(|l| l.len()).sum();
        let slots = pair_slots + cand_count;
        let csr_host = (island_headroom > 0).then(|| csr.clone());
        let mut index = csr;
        let pair_index = index.len() as u32;
        let samples = |k: u32| sample_count(&m.structures[chunks[k as usize].0].chunks[chunks[k as usize].1]) as u32;
        let mut pair_state = Vec::with_capacity(plan.pairs.len());
        let mut state_len = 0u32;
        for (i, (_, a, b, m_red, mu)) in plan.pairs.iter().enumerate() {
            let len = samples(*a) + samples(*b);
            pair_state.push((state_len, len));
            index.extend([*a, *b, (2 * i) as u32, state_len, (*m_red as f32).to_bits(), (*mu as f32).to_bits()]);
            state_len += len;
        }
        let cand_index = index.len() as u32;
        let mut contributions: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut imp_ranges = Vec::new();
        let mut cand = 0u32;
        for (ii, list) in plan.impactor_chunks.iter().enumerate() {
            let begin = cand;
            for &k in list {
                let slot = pair_slots as u32 + cand;
                index.extend([k, slot, ii as u32]);
                contributions[k as usize].push(slot);
                cand += 1;
            }
            imp_ranges.push([begin, cand]);
        }
        for &k in &plan.ground {
            contributions[k as usize].push(GROUND_ENTRY);
        }
        for (i, (_, a, b, _, _)) in plan.pairs.iter().enumerate() {
            contributions[*a as usize].push((2 * i) as u32);
            contributions[*b as usize].push((2 * i + 1) as u32);
        }
        let mut segments: Vec<u32> = Vec::new();
        // Each chunk's contact plan, for the scratch (`cplan_base`).
        let mut chunk_plan = vec![[[0f32; 4]; 4]; chunks.len()];
        // Contact modulus and friction by material name (looked up once per name).
        let mut contact_material: HashMap<&str, (f64, f64)> = HashMap::new();
        for (k, &(s, c)) in chunks.iter().enumerate() {
            let ci = m.chunks[s][c].cluster;
            let data = &m.structures[s].chunks[c];
            let entries = index.len() as u32;
            index.extend_from_slice(&contributions[k]);
            let begin = (segments.len() / 3) as u32;
            for at in (0..contributions[k].len()).step_by(SEGMENT) {
                let end = (at + SEGMENT).min(contributions[k].len());
                segments.extend([k as u32, entries + at as u32, entries + end as u32]);
            }
            let (e, mu) = match &scene {
                Some(scene) => *contact_material.entry(data.material.as_str()).or_insert_with(|| (modulus(scene, &data.material), scene.material(&data.material).friction)),
                None => (1.0, 0.0),
            };
            let (start_hi, start_lo) = match resident {
                Some(r) => (r.starts[k].0, r.starts[k].1),
                None => split(m.chunk_position(s, c)),
            };
            let cs = &mut chunk_static[k];
            cs.half = v4(data.half_extents, data.half_extents.norm());
            cs.crot = rows(&data.rotation);
            cs.cmat = [e as f32, mu as f32, f32::from_bits(hull_of[k].0), f32::from_bits(hull_of[k].1)];
            let mut start_lo = start_lo;
            let falls = plan.falls.get(k).copied().unwrap_or(false);
            cs.cinfo = [begin, (segments.len() / 3) as u32, plan.contact_clusters[ci] as u32, plan.rotation_checked.get(k).copied().unwrap_or(false) as u32 | (falls as u32) << 1];
            // The ballistic path the budget is measured from: velocity, and the time the
            // plan started (compensated pair, with start_lo.w).
            let (t_hi, t_lo) = split1(plan_time);
            let start_vel = v4(plan.velocity.get(k).copied().unwrap_or(Vec3::ZERO), t_hi as f64);
            start_lo[3] = t_lo;
            // Orientation at the segment start (cluster rotation x hidden rotation), x y z w.
            let st = &m.chunks[s][c];
            let q = m.clusters[ci].pose.rotation;
            let h = stress_ref::math::Quat::from_axis_angle(st.th, st.th.norm());
            let start_rot = match resident {
                Some(r) => r.starts[k].2,
                None => [
                    (q.w * h.x + q.x * h.w + q.y * h.z - q.z * h.y) as f32,
                    (q.w * h.y - q.x * h.z + q.y * h.w + q.z * h.x) as f32,
                    (q.w * h.z + q.x * h.y - q.y * h.x + q.z * h.w) as f32,
                    (q.w * h.w - q.x * h.x - q.y * h.y - q.z * h.z) as f32,
                ],
            };
            chunk_plan[k] = [[start_hi[0], start_hi[1], start_hi[2], plan.budget[k] as f32], start_lo, start_rot, start_vel];
        }
        if std::env::var("STRESS_GPU_CONTACT_STATS").is_ok() {
            let per_chunk = contributions.iter().map(|c| c.len()).max().unwrap_or(0);
            let per_island = islands.iter().map(|isl| (isl.range[0]..isl.range[1]).map(|k| contributions[k as usize].len()).sum::<usize>()).max().unwrap_or(0);
            let biggest = islands.iter().map(|isl| isl.range[1] - isl.range[0]).max().unwrap_or(0);
            eprintln!("contacts: {} pairs, longest chunk list {per_chunk}, busiest island {per_island} entries, biggest island {biggest} chunks, {} islands", plan.pairs.len(), islands.len());
        }
        mark("contact tables");
        let mut cstate: Vec<[f32; 4]> = Vec::with_capacity(state_len as usize);
        let mut remembered = Vec::new();
        for (i, (key, ..)) in plan.pairs.iter().enumerate() {
            let len = pair_state[i].1 as usize;
            match pair_memory.get(key) {
                Some(st) if st.len() == len => {
                    cstate.extend_from_slice(st);
                    remembered.push(i);
                }
                _ => cstate.extend(std::iter::repeat(NAN_STATE).take(len)),
            }
        }
        if cstate.is_empty() {
            cstate.push(NAN_STATE);
        }
        let gpu_impactors: Vec<GpuImpactor> = impactors
            .iter()
            .enumerate()
            .map(|(ii, imp)| {
                let (position, position_err) = split(imp.pose.position);
                let (velocity, velocity_err) = split(imp.velocity);
                let (rotation, rotation_err) = split_quat(imp.pose.rotation);
                let r = imp.pose.rotation.to_mat3();
                let (momentum, momentum_err) = split(r * imp.inertia * r.transpose() * imp.angular_velocity);
                let (shape, half) = match imp.shape {
                    ImpactorShape::Sphere { radius } => ([0.0, radius as f32, 0.0, 0.0], Vec3::splat(radius)),
                    ImpactorShape::Box { half_extents } => ([1.0, 0.0, 0.0, 0.0], Vec3::from_array(half_extents)),
                };
                let e = scene.as_ref().map(|s| {
                    let mm = &imp.material;
                    mm.youngs_modulus * s.sim.stiffness_scale / (1.0 - mm.poisson_ratio * mm.poisson_ratio)
                });
                let (max_force, energy) = imp.crush.unwrap_or((0.0, 0.0));
                GpuImpactor {
                    position,
                    position_err,
                    velocity,
                    velocity_err,
                    angular_velocity: v4(imp.angular_velocity, 0.0),
                    rotation,
                    rotation_err,
                    momentum,
                    momentum_err,
                    inertia: rows(&imp.inertia),
                    inv: rows(&imp.inertia.inverse().unwrap_or(Mat3::ZERO)),
                    shape,
                    half: v4(half, half.norm()),
                    mat: [e.unwrap_or(1.0) as f32, imp.material.friction as f32, imp.mass as f32, 0.0],
                    crush: [max_force as f32, energy as f32, imp.crush_used as f32, imp.crush_depth as f32],
                    // Full force at the crush depth reached, until a crush pass says otherwise.
                    geom: [1.0, imp.crush_depth as f32, 0.0, 0.0],
                    cand: [imp_ranges.get(ii).map_or(0, |r| r[0]), imp_ranges.get(ii).map_or(0, |r| r[1]), imp.driven as u32, m.substeps as u32],
                    ..Default::default()
                }
            })
            .collect();
        let crush = impactors.iter().any(|imp| !imp.driven && imp.crush.is_some_and(|(f, _)| f > 0.0));
        for isl in islands.iter_mut() {
            let loaded = (isl.range[0]..isl.range[1]).any(|k| chunk_static[k as usize].load_range[0] != chunk_static[k as usize].load_range[1]);
            let contact = (isl.range[0]..isl.range[1]).any(|k| chunk_static[k as usize].cinfo[2] != 0);
            if isl.info[0] & ISLAND_SETTLED != 0 && isl.info[0] & ISLAND_ANCHORED != 0 && !loaded && !contact {
                isl.info[0] |= ISLAND_QUIET;
            }
        }
        // Wide islands: groups of bonds and of chunks, each chunk group with its island's
        // first chunk group (where the island's partial sums start).
        let threshold = wide_threshold();
        let (mut bond_table, mut chunk_table) = (Vec::new(), Vec::new());
        for (ci, island) in islands.iter_mut().enumerate() {
            let chunk_count = (island.range[1] - island.range[0]) as usize;
            // Settled islands only check their loads and move rigidly: the narrow kernel
            // runs them a whole segment per dispatch, whatever their size.
            if threshold == 0 || chunk_count < threshold || island.info[0] & ISLAND_SETTLED != 0 {
                continue;
            }
            island.info[0] |= ISLAND_WIDE;
            for b in (island.range[2]..island.range[3]).step_by(WIDE_GROUP) {
                bond_table.extend([ci as u32, b, (b + WIDE_GROUP as u32).min(island.range[3]), 0]);
            }
            let first = (chunk_table.len() / 4) as u32;
            for c in (island.range[0]..island.range[1]).step_by(WIDE_GROUP) {
                chunk_table.extend([ci as u32, c, (c + WIDE_GROUP as u32).min(island.range[1]), first]);
            }
            island.done[2] = (chunk_table.len() / 4) as u32 - first;
        }
        let seg_count = (segments.len() / 3) as u32;
        let seg_index = index.len() as u32;
        index.extend(segments);
        let wide_bond_groups = (bond_table.len() / 4) as u32;
        let wide_chunk_groups = (chunk_table.len() / 4) as u32;
        let wide_free = islands.iter().any(|i| i.info[0] & ISLAND_WIDE != 0 && i.info[0] & ISLAND_ANCHORED == 0);
        let wide_settled = islands.iter().any(|i| i.info[0] & ISLAND_WIDE != 0 && i.info[0] & ISLAND_SETTLED != 0);
        let wide_bond_table = index.len() as u32;
        index.extend(bond_table);
        let wide_chunk_table = index.len() as u32;
        index.extend(chunk_table);
        // Narrow contact islands' bonds in groups (the contact rounds run them in the sums
        // dispatch, `contact_bonds_at`).
        let mut contact_bond_list = Vec::new();
        for (ci, island) in islands.iter().enumerate() {
            if island.info[0] & ISLAND_CONTACT == 0 || island.info[0] & ISLAND_WIDE != 0 {
                continue;
            }
            for b in (island.range[2]..island.range[3]).step_by(WIDE_GROUP) {
                contact_bond_list.extend([ci as u32, b, (b + WIDE_GROUP as u32).min(island.range[3]), 0]);
            }
        }
        let contact_settled = islands.iter().any(|i| i.info[0] & ISLAND_CONTACT != 0 && i.info[0] & ISLAND_WIDE == 0 && i.info[0] & ISLAND_SETTLED != 0);
        let contact_bond_groups = (contact_bond_list.len() / 4) as u32;
        let contact_bond_table = index.len() as u32;
        index.extend(contact_bond_list);
        // The sentinel island (batch control) after the islands.
        let halt_index = islands.len() as u32;
        // Room for `island_headroom` more islands (a split in place, `split_resident`):
        // the regions after the islands' result slots stay where they are.
        let island_cap = islands.len() + island_headroom;
        let mut islands_gpu = islands.clone();
        islands_gpu.push(Island::default());
        islands_gpu.resize(island_cap + 1, Island::default());

        let mut materials = vec![GpuJointMaterial::default(); MAX_MATERIALS];
        materials[..table.materials.len()].copy_from_slice(&table.materials);
        // Scratch: bond loads, probe outputs, contact slots, pair and chunk ledgers,
        // impactor records (2 per impactor and substep), impactor candidate data (3 each).
        let probe_base = 3 * bonds.len();
        let slot_base = probe_base + (layout_items.len() * probe_stride as usize).div_ceil(4);
        let ledger_base = slot_base + 2 * slots;
        let record_base = ledger_base + plan.pairs.len() + n;
        let cand_base = record_base + 2 * impactors.len() * probe_stride as usize;
        let wide_base = cand_base + 3 * cand_count;
        let seg_base = wide_base + 8 * wide_chunk_groups as usize;
        // Each chunk's external load: of the last substep, and summed over the frame.
        let cload_base = seg_base + 2 * seg_count as usize;
        let cframe_base = cload_base + 2 * n;
        // Static solves: 23 float4 per chunk, 2 per bond, 1 per island.
        let statics_base = cframe_base + 2 * n;
        let cplan_base = statics_base + 23 * n + 2 * bonds.len() + island_cap + 1;
        let iplan_base = cplan_base + 4 * n;
        let sload_base = iplan_base + 4 * impactors.len().max(1);
        let scratch_len = sload_base + 2 * n;

        let (gravity, ground) = (m.config.gravity, scene.as_ref().and_then(|s| s.ground.clone()));
        let (g_hi, g_lo) = split1(ground.as_ref().map_or(0.0, |g| g.height));
        let params = Params {
            gravity: v4(gravity, 0.0),
            fracture: m.config.fracture as u32,
            rigid_motion_loads: m.config.features.rigid_motion_loads as u32,
            halt_index,
            chunk_count: n as u32,
            pair_count: plan.pairs.len() as u32,
            impactor_count: impactors.len() as u32,
            cand_count: cand_count as u32,
            probe_base: probe_base as u32,
            probe_stride,
            slot_base: slot_base as u32,
            ledger_base: ledger_base as u32,
            record_base: record_base as u32,
            record_stride: probe_stride,
            cand_base: cand_base as u32,
            pair_index,
            cand_index,
            zeta: scene.as_ref().map_or(0.0, |s| damping_ratio(s.sim.contact_restitution)) as f32,
            pair_friction: scene.as_ref().and_then(|s| s.sim.contact_friction).map_or(-1.0, |f| f as f32),
            ground_hi: g_hi,
            ground_lo: g_lo,
            ground_friction: ground.as_ref().map_or(0.0, |g| scene.as_ref().unwrap().sim.contact_friction.unwrap_or(g.friction)) as f32,
            ground_modulus: ground.as_ref().map_or(1.0, |g| modulus(scene.as_ref().unwrap(), &g.material)) as f32,
            has_ground: ground.is_some() as u32,
            wide_bond_groups,
            wide_bond_table,
            wide_chunk_table,
            wide_base: wide_base as u32,
            seg_index,
            seg_count,
            seg_base: seg_base as u32,
            solve_mode: match m.config.mode {
                SolveMode::Explicit => 0,
                SolveMode::Adaptive => 1,
                _ => 2,
            },
            cload_base: cload_base as u32,
            cframe_base: cframe_base as u32,
            statics_base: statics_base as u32,
            statics_bonds: bonds.len() as u32,
            // statics.rs `equilibrate`: its secant steps (all 60 Newton iterations, or
            // SECANT_STEPS under `tangent_statics`, whose re-solve `equilibrate` delegates
            // to the mirror); the residual tolerance at f32's reach.
            // (`STRESS_GPU_STATICS_NEWTON=N`, a measurement only: at most N steps, a lossy
            // budget like Blast's per-frame iteration cap.)
            statics_newton: std::env::var("STRESS_GPU_STATICS_NEWTON")
                .ok()
                .and_then(|v| v.parse().ok())
                .or(m.config.static_budget.map(|b| b.0))
                .unwrap_or(StaticOptions::default().secant_steps(m.config.methods.tangent_statics) as u32),
            statics_cg: m.config.static_budget.map_or(20_000, |b| b.1),
            statics_tol: std::env::var("STRESS_GPU_STATICS_TOL").ok().and_then(|v| v.parse().ok()).or(m.config.static_tolerance).unwrap_or(1e-5),
            pair_lanes: if gpu.shader_path == crate::gpu::ShaderPath::Wgsl { 1 } else { 32 },
            cplan_base: cplan_base as u32,
            iplan_base: iplan_base as u32,
            contact_bond_table,
            sload_base: sload_base as u32,
            device_connect: std::env::var("STRESS_GPU_DEVICE_CONNECT").map_or(1, |v| (v != "0") as u32),
            ..Default::default()
        };

        mark("impactors+wide+params");
        let params_segment = gpu.uniform("params segment", &params);
        let params_contact = gpu.uniform("params contact", &params);
        // A re-plan (same topology) keeps the GPU's chunk state, bond history and static
        // bond data: the GPU holds the current values, the mirror may lag.
        let materials = match keep {
            Some(k) => k.materials.clone(),
            None => gpu.uniform_slice("materials", &materials),
        };
        let bonds_buf = match keep {
            Some(k) => k.bonds_static.clone(),
            None => gpu.storage("bond static", &bond_static),
        };
        let chunks_buf = gpu.storage_into("chunk static", &chunk_static, keep.map(|k| &k.chunk_static));
        let index_buf = gpu.storage_into("index", &index, keep.map(|k| &k.index));
        let state = match keep {
            Some(k) => k.state.clone(),
            None => gpu.storage("state", &state),
        };
        let bond_dyn_buf = match keep {
            Some(k) => k.bond_dyn.clone(),
            None => gpu.storage("bond dyn", &bond_dyn),
        };
        // The scratch is zero but for these regions: cleared on the GPU (a new buffer is
        // zero-initialised) and the regions written, instead of uploading the whole image.
        let mut regions: Vec<(usize, Vec<[f32; 4]>)> = Vec::new();
        regions.push((cplan_base, chunk_plan.iter().flatten().copied().collect()));
        let (t_hi, t_lo) = split1(plan_time);
        let mut iplan = vec![[0f32; 4]; 4 * plan.impactor_plan.len()];
        for (i, p) in plan.impactor_plan.iter().enumerate() {
            if let Some(p) = p {
                let (hi, lo) = split(p.position);
                let q = p.rotation;
                iplan[4 * i] = [hi[0], hi[1], hi[2], p.budget as f32];
                iplan[4 * i + 1] = [lo[0], lo[1], lo[2], t_lo];
                iplan[4 * i + 2] = [q.x as f32, q.y as f32, q.z as f32, q.w as f32];
                iplan[4 * i + 3] = v4(p.velocity, t_hi as f64);
            }
        }
        regions.push((iplan_base, iplan));
        // Bond loads from the mirror's forces: settled islands keep them (their bonds are
        // not evaluated) for the section probes.
        let mut bond_loads = Vec::with_capacity(3 * bonds.len());
        for &(s, bi) in bonds.iter() {
            let b = &m.bonds[s][bi];
            let (fa, ma, _, mb) = b.geometry.chunk_loads(&b.force);
            bond_loads.extend([v4(fa, b.measures.tension.max(b.measures.compression)), v4(ma, 0.0), v4(mb, 0.0)]);
        }
        regions.push((0, bond_loads));
        for &i in &remembered {
            // Pair ledger flags (world.slang): overlap state present.
            regions.push((ledger_base + i, vec![[0.0, 0.0, 0.0, f32::from_bits(1)]]));
        }
        let scratch_bytes = (16 * scratch_len).max(16) as u64;
        let scratch = match keep.map(|k| &k.scratch).filter(|old| old.size() == scratch_bytes) {
            Some(old) => {
                // Queued writes run before the next submission: the clear goes first.
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                encoder.clear_buffer(old, 0, None);
                gpu.queue.submit([encoder.finish()]);
                old.clone()
            }
            None => gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scratch"),
                size: scratch_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        for (at, data) in &regions {
            if !data.is_empty() {
                gpu.queue.write_buffer(&scratch, 16 * *at as u64, bytemuck::cast_slice(data));
            }
        }
        let islands_buf = gpu.storage_into("islands", &islands_gpu, keep.map(|k| &k.islands));
        let loads_buf = gpu.storage_into("loads", &packed, keep.map(|k| &k.loads));
        let cstate_buf = gpu.storage_into("contact state", &cstate, keep.map(|k| &k.contact_state));
        let impactors_buf = gpu.storage_into("impactors", &if gpu_impactors.is_empty() { vec![GpuImpactor::default()] } else { gpu_impactors.clone() }, keep.map(|k| &k.impactors));
        let bind = |params: &wgpu::Buffer| {
            gpu.bind(
                &kernels.island.group,
                &[params, &materials, &bonds_buf, &chunks_buf, &index_buf, &loads_buf, &state, &bond_dyn_buf, &scratch, &islands_buf, &impactors_buf, &cstate_buf],
            )
        };
        let bind_segment = bind(&params_segment);
        let bind_contact = bind(&params_contact);
        mark("buffers");
        Ok((
            Layout { chunks, bonds, islands, items: layout_items, plan, pair_state, gpu_impactors, island_cap, host: (island_headroom > 0).then(|| HostTables { chunk_static: chunk_static.clone(), bond_static: bond_static.clone(), csr: csr_host.unwrap_or_default() }) },
            Buffers {
                bonds_static: bonds_buf.clone(),
                materials: materials.clone(),
                params_segment,
                params_contact,
                params,
                state,
                bond_dyn: bond_dyn_buf,
                scratch,
                islands: islands_buf,
                impactors: impactors_buf,
                contact_state: cstate_buf,
                chunk_static: chunks_buf,
                index: index_buf,
                loads: loads_buf,
                bind_segment,
                bind_contact,
                crush,
                wide_chunk_groups,
                wide_free,
                wide_settled,
                contact_bond_groups,
                contact_settled,
            },
        ))
    }

    /// Whether anything can touch in the solver's own contact: an impactor, or a free
    /// cluster (anchored clusters touch neither each other nor the ground in the
    /// reference's contact). Under an engine, only the bodies it does not drive (the
    /// impact islands) take part (contacts.rs skips every pair with a driven body); with
    /// none, there is no contact pipeline to run.
    fn contacts_possible(&self) -> bool {
        if !self.own_contacts {
            return false;
        }
        let owned = self.mirror.clusters.iter().filter(|c| !c.driven).count();
        let free = self.mirror.clusters.iter().any(|c| !c.anchored && !c.driven);
        self.impactors.iter().any(|i| !i.driven) || (free && (owned > 1 || self.scene().is_some_and(|s| s.ground.is_some())))
    }

    /// Advance every cluster by `substeps` explicit substeps of `dt` (splits and
    /// contacts included). Returns each probe's value after every substep (NaN where it
    /// has no item).
    pub fn step(&mut self, gpu: &Gpu, dt: f64, substeps: usize) -> Result<Vec<Vec<f64>>, String> {
        let stride = (substeps as u32).div_ceil(4) * 4;
        let horizon = dt * substeps as f64;
        let start_time = self.mirror.time;
        // (A plan that covers this step up to the rounding of its substeps' sum stands.)
        let replan = self.contacts_possible() && start_time + horizon > self.plan_until + 1e-9 * horizon;
        let planned_now = stride > self.buffers.params.probe_stride || replan;
        if planned_now {
            // A longer probe stride re-lays the scratch: a full rebuild from the synced
            // mirror. Otherwise a contact re-plan, from the positions the GPU reached
            // (synced after every step), keeping state and bond history resident. Plans
            // cover PLAN_STEPS steps; the GPU's travel check re-plans sooner when a chunk
            // outruns its budget.
            let t = std::time::Instant::now();
            let full = stride > self.buffers.params.probe_stride;
            let horizon = if self.contacts_possible() { horizon * plan_steps() } else { 0.0 };
            self.plan_until = start_time + horizon;
            if full {
                // (Resident segments leave the mirror's chunk state behind: all of it.)
                if self.resident_contacts {
                    self.download(gpu);
                } else {
                    self.sync(gpu);
                }
            }
            let stride = stride.max(self.buffers.params.probe_stride);
            let resident = (self.resident_contacts && !full && horizon > 0.0).then(|| self.resident_plan(gpu, horizon));
            let keep = if full { None } else { Some(&self.buffers) };
            let (layout, buffers) = Self::build(gpu, &self.kernels, &self.mirror, self.loads.as_mut(), &self.impactors, &self.pair_memory, stride, horizon, start_time, keep, &self.extra_terms, &self.excluded, self.island_headroom, resident.as_ref())?;
            self.layout = layout;
            self.layout_generation += 1;
            self.buffers = buffers;
            self.log_plan();
            self.profile.build += t.elapsed().as_secs_f64();
        }
        self.profile.substeps += substeps as u64;
        self.profile.max_pairs = self.profile.max_pairs.max(self.layout.plan.pairs.len());
        self.profile.max_impactor_candidates = self.profile.max_impactor_candidates.max(self.buffers.params.cand_count as usize);
        // Substeps run (fewer when the step stops at a plan; none when it made one first).
        self.stopped_at_plan = self.stop_at_plan && planned_now && self.contacts_possible();
        let mut ran = if self.stopped_at_plan { 0 } else { substeps };
        let start = self.mirror.substeps as usize;
        // Times from `start` on are this step's: a restored state may run them again at
        // another step size.
        self.step_times.truncate(start + 1);
        while self.step_times.len() <= start + substeps {
            let t = *self.step_times.last().unwrap() + dt;
            self.step_times.push(t);
        }
        let (t_hi, t_lo) = split1(self.step_times[start]);
        let dispatch_params = |b: &Buffers, contact: bool| Params {
            dt: dt as f32,
            step_start: start as u32,
            t_hi,
            t_lo,
            max_steps: if contact { 1 } else { u32::MAX },
            contact_mode: contact as u32,
            ..b.params
        };
        // Probe values per item (by identity) and substep.
        let mut values: HashMap<(usize, u32, Target, u32), Vec<f64>> = HashMap::new();
        let mut remaining: HashMap<u64, usize> = self.mirror.clusters.iter().map(|c| (c.id, substeps)).collect();
        // Substeps the contact pipeline (impactors the solver integrates; driven ones are
        // the engine's) still has to run.
        let owned_impactors = self.impactors.iter().any(|i| !i.driven);
        let mut pipeline_left = if owned_impactors { substeps } else { 0 };
        // Pipeline substeps per submission: a halt (split, re-plan) stops the rest of a
        // submission, whose dispatches still launch; small chunks after a halt, growing
        // while none occurs.
        let mut chunk = self.batch_rounds.min(batch_cap());
        // Impactor velocity and position after every substep (for the impactor probes).
        let mut impactor_records = vec![vec![(Vec3::splat(f64::NAN), Vec3::splat(f64::NAN)); substeps]; self.impactors.len()];
        loop {
            if self.stopped_at_plan {
                break;
            }
            let mut any = false;
            let mut contact_left = 0usize;
            let mut wide_left = 0usize;
            // The coverage words start with what was seen (a path costs an atomic once).
            let seen = (self.coverage & coverage::DEVICE) as u32;
            for (ci, isl) in self.layout.islands.iter_mut().enumerate() {
                let left = remaining[&self.mirror.clusters[ci].id];
                isl.done[1] |= seen;
                isl.info[1] = left as u32;
                isl.info[2] = 0;
                isl.info[3] = (start + substeps - left) as u32;
                any |= left > 0;
                if isl.info[0] & ISLAND_CONTACT != 0 {
                    contact_left = contact_left.max(left);
                }
                if isl.info[0] & ISLAND_WIDE != 0 {
                    wide_left = wide_left.max(left);
                }
            }
            let pipeline = owned_impactors || contact_left > 0;
            let mut pipeline_steps = if owned_impactors { pipeline_left.max(contact_left) } else { contact_left }.min(chunk);
            // Rounds of per-substep dispatches: the contact pipeline and the wide islands.
            let mut rounds = pipeline_steps.max(wide_left.min(chunk));
            if self.timing.is_some() {
                rounds = rounds.min((TIMED_DISPATCHES - 1) / 12);
                pipeline_steps = pipeline_steps.min(rounds);
            }
            if !any && rounds == 0 {
                break;
            }
            let t_encode = std::time::Instant::now();
            let mut islands_gpu = self.layout.islands.clone();
            islands_gpu.push(Island { done: [0, seen, 0, 0], ..Default::default() });
            gpu.queue.write_buffer(&self.buffers.islands, 0, bytemuck::cast_slice(&islands_gpu));
            if !self.layout.gpu_impactors.is_empty() {
                // Impactors count their integrated substeps from where the pipeline is.
                let at = (start + substeps - pipeline_left) as u32;
                // Only `cand[3]` changes: written in place, without reading the buffer back
                // (nothing runs between batches, so the rest is what the GPU holds).
                let size = std::mem::size_of::<GpuImpactor>() as u64;
                let field = (std::mem::offset_of!(GpuImpactor, cand) + 3 * std::mem::size_of::<u32>()) as u64;
                for i in 0..self.buffers.impactors.size() / size {
                    gpu.queue.write_buffer(&self.buffers.impactors, i * size + field, bytemuck::bytes_of(&at));
                }
            }
            gpu.queue.write_buffer(&self.buffers.params_segment, 0, bytemuck::bytes_of(&dispatch_params(&self.buffers, false)));
            // Narrow contact islands' bonds in the sums dispatch (they read only the last
            // substep's state), unless one is settled (its wake check comes first).
            let bonds_external = self.buffers.contact_bond_groups > 0 && !self.buffers.contact_settled && external_bonds();
            let contact_params = Params { contact_bond_groups: if bonds_external { self.buffers.contact_bond_groups } else { 0 }, ..dispatch_params(&self.buffers, true) };
            gpu.queue.write_buffer(&self.buffers.params_contact, 0, bytemuck::bytes_of(&contact_params));
            // The batch's dispatches: (kernel, workgroups, per-substep params).
            let islands_n = self.layout.islands.len().max(1) as u32;
            // Islands that touch nothing: every substep in one dispatch.
            let mut ops: Vec<(usize, u32, bool)> = vec![(K_ISLAND, islands_n, false)];
            // Per substep: contact islands and impactors (the contact pipeline), and the wide
            // islands (a round of dispatches each).
            let p = &self.buffers.params;
            let groups = |count: u32| count.div_ceil(64).max(1);
            let impactors_n = self.impactors.len().max(1) as u32;
            let wide_groups = self.buffers.wide_chunk_groups;
            // The wide bond groups join the sums when no wide island is free (no net
            // loads to sum first) or settled (no wake pass first).
            let bonds_with_sums = wide_groups > 0 && !self.buffers.wide_free && !(p.solve_mode != 0 && self.buffers.wide_settled);
            // The contact pipeline in two dispatches per substep (world.slang STOP_NOW): with
            // its bonds outside the island kernel and no wide island (whose kernels keep
            // the three-dispatch order).
            // Settled contact islands too: one that wakes runs that substep's joint law
            // in the island kernel.
            let fused = pipeline_steps > 0 && wide_groups == 0 && self.buffers.contact_bond_groups > 0 && external_bonds() && fuse_pipeline();
            if fused {
                ops.push((K_TRAVEL, groups(p.chunk_count + p.impactor_count), true));
            }
            for r in 0..rounds {
                let contact = pipeline && r < pipeline_steps;
                // A round: contact_forces; the sums with the impactors' integration (both
                // read only its outputs); wide_bonds (its free islands' net loads need the
                // sums); the narrow islands with the wide islands' chunks (other islands,
                // both after the sums and wide_bonds); the wide islands' tail.
                if contact && fused {
                    if self.buffers.crush {
                        ops.push((K_SHARES, groups(p.cand_count), true));
                        ops.push((K_CRUSH, impactors_n, true));
                    }
                    let forces = (p.pair_count + p.cand_count) * p.pair_lanes.max(1);
                    ops.push((K_FORCES_BONDS, forces.div_ceil(256) + self.buffers.contact_bond_groups, true));
                    ops.push((if self.buffers.contact_settled { K_ISLANDS_SUMS_WAKE } else { K_ISLANDS_SUMS }, islands_n + impactors_n, true));
                    continue;
                }
                if contact {
                    if self.buffers.crush {
                        ops.push((K_SHARES, groups(p.cand_count), true));
                        ops.push((K_CRUSH, impactors_n, true));
                    }
                    ops.push((K_FORCES, groups((p.pair_count + p.cand_count) * p.pair_lanes.max(1) + p.chunk_count + p.impactor_count), true));
                    if bonds_with_sums || bonds_external {
                        let narrow = if bonds_external { self.buffers.contact_bond_groups } else { 0 };
                        let wide = if bonds_with_sums { p.wide_bond_groups } else { 0 };
                        ops.push((K_SUMS_BONDS, p.seg_count.div_ceil(256) + impactors_n + narrow + wide, true));
                    } else if owned_impactors {
                        ops.push((K_SUMS_INTEGRATE, p.seg_count.div_ceil(256) + impactors_n, true));
                    } else {
                        ops.push((K_SUMS, groups(p.seg_count), true));
                    }
                }
                if wide_groups > 0 {
                    // wide_wake returns at once unless its island is settled.
                    if p.solve_mode != 0 && self.buffers.wide_settled {
                        ops.push((K_WAKE, wide_groups, true));
                    }
                    if !(contact && bonds_with_sums) {
                        ops.push((K_WIDE[0], p.wide_bond_groups + wide_groups, true));
                    }
                    if contact {
                        ops.push((if bonds_external { K_CONTACT_ISLAND_WIDE } else { K_ISLAND_WIDE }, islands_n + wide_groups, true));
                    } else {
                        ops.push((K_WIDE[1], wide_groups, true));
                    }
                    for &k in &K_WIDE[2..] {
                        // wide_drift and wide_rigid return at once for anchored islands.
                        if self.buffers.wide_free || (k != K_WIDE[2] && k != K_WIDE[3]) {
                            ops.push((k, wide_groups, true));
                        }
                    }
                } else if contact {
                    ops.push((if bonds_external { K_CONTACT_ISLANDS } else { K_ISLAND }, islands_n, true));
                }
            }
            let w = &self.kernels.wide;
            let kernels = [
                &self.kernels.island,
                &self.kernels.forces,
                &self.kernels.shares,
                &self.kernels.crush,
                &self.kernels.integrate,
                &w[0],
                &w[1],
                &w[2],
                &w[3],
                &w[4],
                &self.kernels.sums,
                &self.kernels.wake,
                &self.kernels.sums_integrate,
                &self.kernels.island_wide,
                &self.kernels.sums_bonds,
                &self.kernels.contact_islands,
                &self.kernels.contact_island_wide,
                &self.kernels.forces_bonds,
                &self.kernels.islands_sums,
                &self.kernels.travel,
                &self.kernels.islands_sums_wake,
            ];
            let bind = |contact: bool| if contact { &self.buffers.bind_contact } else { &self.buffers.bind_segment };
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            match &self.timing {
                None => {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    for &(k, groups, contact) in &ops {
                        pass.set_pipeline(&kernels[k].pipeline);
                        pass.set_bind_group(0, bind(contact), &[]);
                        pass.dispatch_workgroups(groups, 1, 1);
                    }
                }
                Some(timing) => {
                    // Profiling: a pass per dispatch, timestamped at both ends.
                    for (q, &(k, groups, contact)) in ops.iter().enumerate() {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: &timing.queries,
                                beginning_of_pass_write_index: Some(2 * q as u32),
                                end_of_pass_write_index: Some(2 * q as u32 + 1),
                            }),
                        });
                        pass.set_pipeline(&kernels[k].pipeline);
                        pass.set_bind_group(0, bind(contact), &[]);
                        pass.dispatch_workgroups(groups, 1, 1);
                    }
                    encoder.resolve_query_set(&timing.queries, 0..2 * ops.len() as u32, &timing.resolve, 0);
                }
            }
            let finished = encoder.finish();
            self.profile.encode += t_encode.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            gpu.queue.submit([finished]);
        self.state_current = false;
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            self.profile.gpu += t.elapsed().as_secs_f64();
            if let Some(timing) = &self.timing {
                let stamps: Vec<u64> = gpu.read(&timing.resolve);
                let period = gpu.queue.get_timestamp_period() as f64 * 1e-9;
                let mut batch = [(0.0f64, 0.0f64, 0u32); 21];
                for (q, &(k, ..)) in ops.iter().enumerate() {
                    let secs = stamps[2 * q + 1].saturating_sub(stamps[2 * q]) as f64 * period;
                    self.profile.kernels[k] += secs;
                    self.profile.kernel_dispatches[k] += 1;
                    batch[k] = (batch[k].0 + secs, batch[k].1.max(secs), batch[k].2 + 1);
                }
                if std::env::var("STRESS_GPU_TIMING_BATCHES").is_ok() {
                    let line: Vec<String> = batch
                        .iter()
                        .enumerate()
                        .filter(|(_, b)| b.2 > 0)
                        .map(|(k, b)| format!("{} {:.1}/{:.1} us", KERNEL_NAMES[k], b.0 / b.2 as f64 * 1e6, b.1 * 1e6))
                        .collect();
                    eprintln!("batch at substep {}: {} islands, {} pairs, {} candidates: {}", start + substeps - pipeline_left, self.layout.islands.len(), self.buffers.params.pair_count, self.buffers.params.cand_count, line.join(", "));
                }
            }
            self.profile.batches += 1;
            self.dispatches += 1;
            let t_read = std::time::Instant::now();
            // One round trip: the islands, the probe outputs, the impactors' records and
            // state.
            let p = self.buffers.params;
            let f4 = std::mem::size_of::<[f32; 4]>() as u64;
            let probe_len = (self.layout.items.len() as u64 * p.probe_stride as u64).div_ceil(4) * f4;
            let record_len = if owned_impactors { 2 * self.impactors.len() as u64 * p.record_stride as u64 * f4 } else { 0 };
            let mut read = gpu
                .read_ranges(&[
                    (&self.buffers.islands, 0, std::mem::size_of::<Island>() as u64 * (self.layout.islands.len() as u64 + 1)),
                    (&self.buffers.scratch, p.probe_base as u64 * f4, probe_len),
                    (&self.buffers.scratch, p.record_base as u64 * f4, record_len),
                    (&self.buffers.impactors, 0, if owned_impactors { self.buffers.impactors.size() } else { 0 }),
                ])
                .into_iter();
            let mut islands: Vec<Island> = bytemuck::pod_collect_to_vec(&read.next().unwrap());
            self.profile.rounds_encoded += rounds as u64;
            self.profile.rounds_used += islands.iter().zip(&self.layout.islands).filter(|(_, b)| b.info[0] & (ISLAND_CONTACT | ISLAND_WIDE) != 0).map(|(a, b)| (a.info[3] - b.info[3]) as u64).max().unwrap_or(0);
            let probe_out: Vec<f32> = bytemuck::pod_collect_to_vec(&read.next().unwrap());
            let records: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read.next().unwrap());
            let imps_now: Vec<GpuImpactor> = bytemuck::pod_collect_to_vec(&read.next().unwrap());
            let sentinel = islands.pop().unwrap_or_default();
            self.coverage |= islands.iter().chain([&sentinel]).fold(0, |a, i| a | i.done[1]) as u64;
            // Probe outputs of the substeps each island ran.
            if !self.layout.items.is_empty() {
                let flat: &[f32] = &probe_out;
                for (ci, isl) in islands.iter().enumerate() {
                    let before = self.layout.islands[ci].info[3] as usize;
                    let (first, last) = (before - start, isl.info[3] as usize - start);
                    let island = &self.layout.islands[ci];
                    for at in (island.probes[0]..island.probes[1]).step_by(4) {
                        let slot = ((at - self.layout.islands[0].probes[0]) / 4) as usize;
                        let item = &self.layout.items[slot];
                        let row = values.entry(item.key()).or_insert_with(|| vec![f64::NAN; substeps]);
                        for k in first..last {
                            row[k] = flat[slot * p.probe_stride as usize + k] as f64;
                        }
                    }
                }
            }
            if owned_impactors {
                // Impactor records of the substeps the pipeline ran.
                let stride = self.buffers.params.record_stride as usize;
                let first = substeps - pipeline_left;
                for (ii, rec) in impactor_records.iter_mut().enumerate() {
                    for k in first..substeps.min(stride) {
                        let at = 2 * (ii * stride + k);
                        rec[k] = (vec3(records[at]), vec3(records[at + 1]));
                    }
                }
                let imps = imps_now;
                let done = imps[0].cand[3] as usize - (start + substeps - pipeline_left);
                pipeline_left -= done;
            }
            self.profile.readback += t_read.elapsed().as_secs_f64();
            let halted: Vec<usize> = (0..islands.len()).filter(|&i| islands[i].info[2] & ISLAND_HALTED != 0).collect();
            if let Some(limit) = blowup_limit() {
                self.batch_check(limit, &islands, dt);
            }
            for (ci, isl) in islands.iter().enumerate() {
                let id = self.mirror.clusters[ci].id;
                *remaining.get_mut(&id).unwrap() = isl.info[1] as usize;
            }
            self.layout.islands = islands;
            // The two-dispatch pipeline records the substep a travel check stopped at; a stop
            // after a split's substep is moot (the three-dispatch order never checks it).
            let replan = if fused {
                let at = sentinel.range[0];
                at != 0 && (sentinel.info[1] == 0 || at <= sentinel.info[1])
            } else {
                sentinel.info[2] & STOP_NOW != 0
            };
            if halted.is_empty() && !replan {
                if remaining.values().all(|&l| l == 0) && pipeline_left == 0 {
                    break;
                }
                // The pipeline ran a full chunk without a halt: continue with a larger one.
                chunk = (chunk * 2).min(batch_cap());
                self.batch_rounds = chunk;
                continue;
            }
            // A disconnection that leaves its cluster in one piece changes nothing
            // (`split_cluster` returns): those islands resume without a host round trip.
            // A halt (resumed or split) starts short batches again: breaks come in runs,
            // and each halt leaves the rest of its batch's dispatches launching empty.
            chunk = batch_start();
            self.batch_rounds = chunk;
            if !replan && halted.iter().all(|&ci| self.still_connected(gpu, ci)) {
                self.resumed_halts += halted.len() as u64;
                self.coverage |= coverage::RESUMED_HALT;
                continue;
            }
            // Split the halted clusters on the host mirror, at the time they reached;
            // re-plan contacts from the state reached.
            let t_split = std::time::Instant::now();
            self.download(gpu);
            self.mirror_check("before host split");
            let mut pending: Vec<usize> = halted;
            pending.sort_unstable();
            for &ci in pending.iter().rev() {
                let id = self.mirror.clusters[ci].id;
                let left = remaining[&id];
                let done = substeps - left;
                self.mirror.time = self.step_times[start + done];
                let before = self.mirror.clusters.len();
                self.mirror.split_cluster(ci);
                self.host_splits += 1;
                self.coverage |= coverage::HOST_SPLIT;
                for c in &self.mirror.clusters[before.saturating_sub(1)..] {
                    remaining.entry(c.id).or_insert(left);
                }
            }
            if !pending.is_empty() {
                if let Some(l) = self.loads.as_mut() {
                    l.topology_version += 1;
                }
            }
            if replan {
                self.replans += 1;
                self.coverage |= coverage::REPLAN;
            }
            self.mirror_check("after host split");
            self.mirror.time = self.step_times[start];
            // The live clusters' (the map keeps the ids of clusters split away).
            let left_now = self.mirror.clusters.iter().filter_map(|c| remaining.get(&c.id).copied()).max().unwrap_or(0).max(pipeline_left);
            let reached = self.step_times[start + substeps - left_now];
            self.plan_until = reached + dt * left_now as f64;
            let (layout, buffers) = Self::build(gpu, &self.kernels, &self.mirror, self.loads.as_mut(), &self.impactors, &self.pair_memory, self.buffers.params.probe_stride, dt * left_now as f64, reached, None, &self.extra_terms, &self.excluded, self.island_headroom, None)?;
            self.layout = layout;
            self.layout_generation += 1;
            self.buffers = buffers;
            self.log_plan();
            self.profile.split += t_split.elapsed().as_secs_f64();
            let lockstep = self.mirror.clusters.iter().all(|c| remaining.get(&c.id).is_none_or(|&l| l == left_now)) && (!owned_impactors || pipeline_left == left_now);
            if self.stop_at_plan && left_now > 0 && lockstep && self.contacts_possible() {
                ran = substeps - left_now;
                self.stopped_at_plan = true;
                break;
            }
        }
        let t = std::time::Instant::now();
        self.sync_motion(gpu);
        self.bonds_stale = true;
        self.profile.download += t.elapsed().as_secs_f64();
        self.mirror.time = self.step_times[start + ran];
        self.mirror.substeps = (start + ran) as u64;
        if sync_each_step() {
            self.sync(gpu);
        }
        // Each probe's value: the sum of its items (NaN without any); impactor probes from
        // the impactor records.
        let probes = self.loads.as_ref().map(|l| l.scene.probes.len()).unwrap_or(0);
        let mut out = vec![vec![f64::NAN; substeps]; probes];
        if let Some(l) = self.loads.as_ref() {
            for (p, probe) in l.scene.probes.iter().enumerate() {
                let (name, axis, velocity) = match &probe.kind {
                    stress_ref::scene::ProbeKind::ImpactorVelocity { impactor, axis } => (impactor, axis, true),
                    stress_ref::scene::ProbeKind::ImpactorPosition { impactor, axis } => (impactor, axis, false),
                    _ => continue,
                };
                let Some(ii) = self.impactors.iter().position(|i| &i.name == name) else { continue };
                let a = Vec3::from_array(*axis).normalized();
                for k in 0..substeps {
                    let (v, x) = impactor_records[ii][k];
                    out[p][k] = if velocity { v.dot(a) } else { x.dot(a) };
                }
            }
        }
        let mut seen = vec![vec![false; substeps]; probes];
        // Sums over bonds or chunks (world.rs `probe_value`): zero once nothing is left
        // to sum, e.g. after every bond across a section broke.
        if let Some(l) = self.loads.as_ref() {
            for (p, probe) in l.scene.probes.iter().enumerate() {
                if matches!(probe.kind, stress_ref::scene::ProbeKind::SectionForce { .. } | stress_ref::scene::ProbeKind::Reaction { .. }) {
                    out[p].fill(0.0);
                    seen[p].fill(true);
                }
            }
        }
        for (key, row) in &values {
            for k in 0..substeps {
                if row[k].is_nan() {
                    continue;
                }
                if !seen[key.0][k] {
                    out[key.0][k] = 0.0;
                    seen[key.0][k] = true;
                }
                out[key.0][k] += row[k];
            }
        }
        for row in &mut out {
            row.truncate(ran);
        }
        Ok(out)
    }

    /// Diagnostic (`STRESS_GPU_BLOWUP_CHECK=<m/s>`): the fastest active chunk of the
    /// mirror (cluster motion and hidden state), printed with `label` when above the limit.
    pub fn mirror_check(&self, label: &str) -> bool {
        let Some(limit) = blowup_limit() else { return false };
        let m = &self.mirror;
        let mut worst = (0.0f64, 0usize, 0usize);
        for s in 0..m.structures.len() {
            for c in 0..m.structures[s].chunks.len() {
                if !m.chunks[s][c].active {
                    continue;
                }
                let st = &m.chunks[s][c];
                // A hidden displacement of a metre or turn of a radian counts as a speed
                // over the limit (diagnostic: deformation that large is not physical).
                let v = m.chunk_velocity(s, c).0.norm().max(if st.u.norm() > 1.0 || st.th.norm() > 1.0 { f64::INFINITY } else { 0.0 });
                if !v.is_finite() || v > worst.0 {
                    worst = (v, s, c);
                }
            }
        }
        if !(worst.0 <= limit) {
            let (_, s, c) = worst;
            let ci = m.chunks[s][c].cluster;
            let cl = &m.clusters[ci];
            let st = &m.chunks[s][c];
            eprintln!(
                "MIRROR {label} at t {:.6} (substep {}): chunk ({s},{c}) speed {:.3e}; cluster {ci} (id {}, parent {:?}, {} chunks, {} bonds, m {:.3}, {:?}, anchored {}, driven {}) v {:?} w {:?}; hidden u {:?} th {:?} v {:?} w {:?}",
                m.time,
                m.substeps,
                worst.0,
                cl.id,
                cl.parent,
                cl.chunks.len(),
                cl.bonds.len(),
                cl.mass,
                cl.activity,
                cl.anchored,
                cl.driven,
                cl.velocity.to_array(),
                cl.angular_velocity.to_array(),
                st.u.to_array(),
                st.th.to_array(),
                st.v.to_array(),
                st.w.to_array()
            );
            return true;
        }
        false
    }

    /// Diagnostic (`STRESS_GPU_BLOWUP_CHECK`): after a batch, each island whose centre of
    /// mass or surface (|w| times its chunks' reach) now moves faster than `limit` and did
    /// not before it, with what the contact plan lets its chunks touch.
    fn batch_check(&self, limit: f64, islands: &[Island], dt: f64) {
        let m = &self.mirror;
        let plan = &self.layout.plan;
        let speed = |isl: &Island, ci: usize| {
            let cl = &m.clusters[ci];
            let reach = cl.chunks.iter().map(|&c| (m.structures[cl.structure].chunks[c].center - cl.com).norm() + m.structures[cl.structure].chunks[c].half_extents.norm()).fold(0.0, f64::max);
            joined(isl.velocity, isl.velocity_err).norm() + vec3(isl.angular_velocity).norm() * reach
        };
        for (ci, isl) in islands.iter().enumerate() {
            let before = &self.layout.islands[ci];
            let (s0, s1) = (speed(before, ci), speed(isl, ci));
            if !(s1 > limit) || s0 > limit {
                continue;
            }
            let cl = &m.clusters[ci];
            eprintln!(
                "BATCH island {ci} (id {}, parent {:?}, {} chunks, {} bonds, m {:.1}, {:?}, flags {:#x}): surface speed {s0:.3e} -> {s1:.3e} over substeps {}..{} (dt {dt:.3e}); v {:?} -> {:?}, w {:?} -> {:?}",
                cl.id,
                cl.parent,
                cl.chunks.len(),
                cl.bonds.len(),
                cl.mass,
                cl.activity,
                before.info[0],
                before.info[3],
                isl.info[3],
                joined(before.velocity, before.velocity_err).to_array(),
                joined(isl.velocity, isl.velocity_err).to_array(),
                vec3(before.angular_velocity).to_array(),
                vec3(isl.angular_velocity).to_array()
            );
            for &c in cl.chunks.iter().take(6) {
                let s = cl.structure;
                let k = self.layout.chunks.iter().position(|&sc| sc == (s, c)).unwrap() as u32;
                let pairs: Vec<String> = plan
                    .pairs
                    .iter()
                    .filter(|p| p.1 == k || p.2 == k)
                    .map(|p| {
                        let o = if p.1 == k { p.2 } else { p.1 } as usize;
                        let (so, co) = self.layout.chunks[o];
                        format!("({so},{co}) cl {} m {:.0}", m.chunks[so][co].cluster, m.structures[so].chunks[co].mass)
                    })
                    .collect();
                let ch = &m.structures[s].chunks[c];
                eprintln!(
                    "    chunk ({s},{c}) m {:.1} half {:?} hull {} at {:?}; ground {}, impactor {}, pairs {}: {}",
                    ch.mass,
                    ch.half_extents.to_array(),
                    ch.hull.as_ref().map_or(0, |h| h.vertices.len()),
                    m.chunk_position(s, c).to_array(),
                    plan.ground.contains(&k),
                    plan.impactor_chunks.iter().any(|l| l.contains(&k)),
                    pairs.len(),
                    pairs.join("; ")
                );
            }
        }
    }

    /// Whether cluster `ci` stays one piece with the bonds that disconnected on the GPU
    /// since the last download (their break events are pending there) removed.
    fn still_connected(&self, gpu: &Gpu, ci: usize) -> bool {
        let isl = &self.layout.islands[ci];
        let (first, last) = (isl.range[2] as usize, isl.range[3] as usize);
        let dyns: Vec<BondDyn> = gpu.read_range(&self.buffers.bond_dyn, first, last - first);
        let m = &self.mirror;
        let cl = &m.clusters[ci];
        let s = cl.structure;
        let mut broken = std::collections::HashSet::new();
        for (k, d) in dyns.iter().enumerate() {
            if d.events[2] != 0 {
                broken.insert(self.layout.bonds[first + k].1);
            }
        }
        let mut member = std::collections::HashSet::new();
        member.extend(cl.chunks.iter().copied());
        let mut seen = std::collections::HashSet::new();
        let mut queue = vec![cl.chunks[0]];
        seen.insert(cl.chunks[0]);
        while let Some(c) = queue.pop() {
            for &bi in &m.chunk_bonds[s][c] {
                let b = &m.bonds[s][bi];
                if !b.connected() || broken.contains(&bi) {
                    continue;
                }
                let o = if b.geometry.a == c { b.geometry.b } else { b.geometry.a };
                if member.contains(&o) && seen.insert(o) {
                    queue.push(o);
                }
            }
        }
        seen.len() == cl.chunks.len()
    }

    /// Everything a frame changes, downloaded into a host copy (`restore` undoes the
    /// frame).
    pub fn snapshot(&mut self, gpu: &Gpu) -> HostSnapshot {
        self.snapshot_omitting(gpu, false)
    }

    /// `snapshot` for a segment: the excluded set is fixed until the next `refresh`, so
    /// structures all of whose clusters are off the GPU tables cannot change before a
    /// `restore` in the same frame and stay out of the copy (`HostSnapshot::omitted`;
    /// its `mirror()` then lacks their chunks and bonds).
    pub fn snapshot_segment(&mut self, gpu: &Gpu) -> HostSnapshot {
        self.snapshot_omitting(gpu, true)
    }

    fn snapshot_omitting(&mut self, gpu: &Gpu, omit_quiet: bool) -> HostSnapshot {
        // Resident segments: the mirror's motion and contacts only; the chunk state and
        // bond records stay on the GPU, copied there for a restore.
        let gpu_state = if self.resident_contacts {
            self.sync_motion_light(gpu);
            let copy = |b: &wgpu::Buffer, label: &str| {
                let c = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: b.size(), usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                c
            };
            let (st, bd) = (copy(&self.buffers.state, "snapshot state"), copy(&self.buffers.bond_dyn, "snapshot bonds"));
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.buffers.state, 0, &st, 0, st.size());
            encoder.copy_buffer_to_buffer(&self.buffers.bond_dyn, 0, &bd, 0, bd.size());
            gpu.queue.submit([encoder.finish()]);
            Some((st, bd))
        } else {
            self.download(gpu);
            None
        };
        let m = &mut self.mirror;
        let mut on_gpu = vec![!omit_quiet; m.structures.len()];
        for cl in &m.clusters {
            if self.excluded.binary_search(&cl.id).is_err() {
                on_gpu[cl.structure] = true;
            }
        }
        let omitted: Vec<usize> = (0..on_gpu.len()).filter(|&s| !on_gpu[s]).collect();
        let parked: Vec<_> = omitted.iter().map(|&s| (std::mem::take(&mut m.chunks[s]), std::mem::take(&mut m.bonds[s]), std::mem::take(&mut m.chunk_bonds[s]))).collect();
        let mirror = m.clone();
        for (&s, (c, b, cb)) in omitted.iter().zip(parked) {
            (m.chunks[s], m.bonds[s], m.chunk_bonds[s]) = (c, b, cb);
        }
        HostSnapshot {
            mirror,
            omitted,
            impactors: self.impactors.clone(),
            contact_dissipated: self.contact_dissipated,
            crush_energy: self.crush_energy,
            last_loads: self.last_loads.clone(),
            frame_load_sum: self.frame_load_sum.clone(),
            pair_memory: self.pair_memory.clone(),
            gpu_state,
        }
    }

    /// Return to a snapshot: the host state, then every buffer rebuilt from it.
    pub fn restore(&mut self, gpu: &Gpu, s: HostSnapshot) -> Result<(), String> {
        let mut mirror = s.mirror;
        for &st in &s.omitted {
            mirror.chunks[st] = std::mem::take(&mut self.mirror.chunks[st]);
            mirror.bonds[st] = std::mem::take(&mut self.mirror.bonds[st]);
            mirror.chunk_bonds[st] = std::mem::take(&mut self.mirror.chunk_bonds[st]);
        }
        self.mirror = mirror;
        self.impactors = s.impactors;
        self.contact_dissipated = s.contact_dissipated;
        self.crush_energy = s.crush_energy;
        self.last_loads = s.last_loads;
        self.frame_load_sum = s.frame_load_sum;
        self.pair_memory = s.pair_memory;
        self.bonds_stale = false;
        self.rebuild(gpu, 0.0)?;
        if let Some((st, bd)) = s.gpu_state {
            // The chunk state and bond records as the GPU had them (the mirror's lag).
            if st.size() != self.buffers.state.size() || bd.size() != self.buffers.bond_dyn.size() {
                return Err("restore: the snapshot's tables do not match the rebuilt ones".into());
            }
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&st, 0, &self.buffers.state, 0, st.size());
            encoder.copy_buffer_to_buffer(&bd, 0, &self.buffers.bond_dyn, 0, bd.size());
            gpu.queue.submit([encoder.finish()]);
            self.state_current = false;
            self.bonds_stale = true;
        }
        Ok(())
    }

    /// Copy the GPU state into the mirror: hidden state, reactions, bond history and
    /// forces, cluster motion, energies (accumulated since the last download), events,
    /// impactors and the contacts' overlap memory.
    pub fn download(&mut self, gpu: &Gpu) {
        if let Some(bond_dyn) = self.sync_motion_and(gpu, true) {
            self.apply_bonds(gpu, bond_dyn);
        }
    }

    /// Bring the mirror's bonds (history, forces, energies, events) up to the GPU's, if
    /// they lag: before the host changes the mirror, and for observations.
    pub fn sync(&mut self, gpu: &Gpu) {
        if self.bonds_stale {
            self.sync_bonds(gpu);
        }
    }

    /// The mirror's motion from the GPU: chunk state, cluster poses and work, impactors,
    /// contact energies and overlap memory, recorded loads. Cheap: no bond data.
    fn sync_motion(&mut self, gpu: &Gpu) {
        self.sync_motion_and(gpu, false);
    }

    /// `sync_motion`, its reads (and the bonds' with `bonds`, returned) in one round trip.
    fn sync_motion_and(&mut self, gpu: &Gpu, bonds: bool) -> Option<Vec<BondDyn>> {
        self.sync_motion_with(gpu, bonds, true)
    }

    /// `sync_motion` without the chunk state (a resident segment's snapshot: the
    /// mirror's cluster motion, impactors, loads and contact memory).
    fn sync_motion_light(&mut self, gpu: &Gpu) {
        self.sync_motion_with(gpu, false, false);
        self.state_current = false;
    }

    fn sync_motion_with(&mut self, gpu: &Gpu, bonds: bool, chunk_state: bool) -> Option<Vec<BondDyn>> {
        // Settled islands' chunk state is frozen: read the rest only (in runs).
        let mut state: Vec<[f32; 4]> = vec![[0.0; 4]; 4 * self.layout.chunks.len()];
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for isl in self.layout.islands.iter().filter(|_| chunk_state) {
            let (c0, c1) = (isl.range[0] as usize, isl.range[1] as usize);
            if (isl.info[0] & ISLAND_SETTLED != 0 && !self.state_on_gpu) || c0 == c1 {
                continue;
            }
            match runs.last_mut() {
                Some(r) if r.1 == c0 => r.1 = c1,
                _ => runs.push((c0, c1)),
            }
        }
        let p = self.buffers.params;
        let plan = &self.layout.plan;
        let n = self.layout.chunks.len();
        let contacts = !plan.pairs.is_empty() || !plan.ground.is_empty() || p.cand_count > 0;
        let (lb, le) = (p.ledger_base as usize, p.ledger_base as usize + plan.pairs.len() + n);
        let (cb, ce) = (p.cand_base as usize, p.cand_base as usize + 3 * p.cand_count as usize);
        let f4 = std::mem::size_of::<[f32; 4]>() as u64;
        let mut parts: Vec<(&wgpu::Buffer, u64, u64)> = runs.iter().map(|&(a, b)| (&self.buffers.state, 4 * a as u64 * f4, 4 * (b - a) as u64 * f4)).collect();
        let at_islands = parts.len();
        parts.push((&self.buffers.islands, 0, std::mem::size_of::<Island>() as u64 * (self.layout.islands.len() as u64 + 1)));
        let at_imps = parts.len();
        parts.push((&self.buffers.impactors, 0, if self.impactors.is_empty() { 0 } else { self.buffers.impactors.size() }));
        let at_loads = parts.len();
        parts.push((&self.buffers.scratch, p.cload_base as u64 * f4, if p.solve_mode != 0 { 4 * n as u64 * f4 } else { 0 }));
        let at_ledgers = parts.len();
        parts.push((&self.buffers.scratch, lb as u64 * f4, if contacts { (le - lb) as u64 * f4 } else { 0 }));
        parts.push((&self.buffers.scratch, cb as u64 * f4, if contacts && ce > cb { (ce - cb) as u64 * f4 } else { 0 }));
        parts.push((&self.buffers.contact_state, 0, if contacts && !plan.pairs.is_empty() { self.buffers.contact_state.size() } else { 0 }));
        let at_bonds = parts.len();
        parts.push((&self.buffers.bond_dyn, 0, if bonds { self.buffers.bond_dyn.size() } else { 0 }));
        let mut read = gpu.read_ranges(&parts).into_iter();
        for &(a, b) in &runs {
            let part: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read.next().unwrap());
            state[4 * a..4 * b].copy_from_slice(&part);
        }
        let mut fetched: Vec<Vec<u8>> = read.collect();
        let mut take = |k: usize| std::mem::take(&mut fetched[k - at_islands]);
        let mut islands: Vec<Island> = bytemuck::pod_collect_to_vec(&take(at_islands));
        let sentinel = islands.pop().unwrap_or_default();
        self.coverage |= islands.iter().fold(0, |a, i| a | i.done[1]) as u64;
        let m = &mut self.mirror;
        // Settled islands' chunks and bonds never change on the GPU: the mirror keeps its
        // own (f64) equilibrium for them, which the next static solve starts from.
        let frozen: Vec<bool> = self.layout.islands.iter().map(|isl| isl.info[0] & ISLAND_SETTLED != 0 && !self.state_on_gpu).collect();
        for (k, &(s, c)) in self.layout.chunks.iter().enumerate() {
            if !chunk_state || frozen[m.chunks[s][c].cluster] {
                continue;
            }
            let cs = &mut m.chunks[s][c];
            cs.u = vec3(state[4 * k]);
            cs.th = vec3(state[4 * k + 1]);
            cs.v = vec3(state[4 * k + 2]);
            cs.w = vec3(state[4 * k + 3]);
            if m.structures[s].chunks[c].support != Support::None {
                cs.reaction = (Vec3::new(state[4 * k + 1][3] as f64, state[4 * k + 2][3] as f64, state[4 * k + 3][3] as f64), Vec3::ZERO);
            }
        }
        for (ci, isl) in islands.iter_mut().enumerate() {
            m.energy.external_work += isl.energy[0] as f64 + isl.energy[1] as f64;
            isl.energy[0] = 0.0;
            isl.energy[1] = 0.0;
            let cl = &mut m.clusters[ci];
            if m.config.mode == SolveMode::Adaptive && cl.activity == Activity::Settled && isl.info[0] & ISLAND_SETTLED == 0 {
                // Woke on the GPU (world.rs `wake_loaded_clusters`).
                cl.activity = Activity::Active;
                cl.active_timer = m.config.active_time;
            }
            if !cl.anchored {
                let (q, e) = (isl.rotation, isl.rotation_err);
                let j = |k: usize| q[k] as f64 + e[k] as f64;
                cl.pose.rotation = Quat { x: j(0), y: j(1), z: j(2), w: j(3) }.normalized();
                cl.pose.position = joined(isl.position, isl.position_err);
                cl.velocity = joined(isl.velocity, isl.velocity_err);
                // From the momentum the GPU integrates (its `angular_velocity` is derived).
                let l = joined(isl.momentum, isl.momentum_err);
                cl.angular_velocity = cl.world_inertia().inverse().map_or(vec3(isl.angular_velocity), |inv| inv * l);
            }
        }
        let mut with_sentinel = islands.clone();
        with_sentinel.push(sentinel);
        gpu.queue.write_buffer(&self.buffers.islands, 0, bytemuck::cast_slice(&with_sentinel));
        self.layout.islands = islands;
        // Impactors and contact energies.
        let imps_bytes = take(at_imps);
        let loads_bytes = take(at_loads);
        let ledgers_bytes = take(at_ledgers);
        let cands_bytes = take(at_ledgers + 1);
        let cstate_bytes = take(at_ledgers + 2);
        let bond_bytes = take(at_bonds);
        if !self.impactors.is_empty() {
            let mut imps: Vec<GpuImpactor> = bytemuck::pod_collect_to_vec(&imps_bytes);
            for (imp, g) in self.impactors.iter_mut().zip(imps.iter_mut()) {
                imp.pose.position = joined(g.position, g.position_err);
                let (q, e) = (g.rotation, g.rotation_err);
                let j = |k: usize| q[k] as f64 + e[k] as f64;
                imp.pose.rotation = Quat { x: j(0), y: j(1), z: j(2), w: j(3) }.normalized();
                imp.velocity = joined(g.velocity, g.velocity_err);
                // From the momentum the GPU integrates (its angular velocity is derived).
                let r = imp.pose.rotation.to_mat3();
                let l = joined(g.momentum, g.momentum_err);
                imp.angular_velocity = (r * imp.inertia * r.transpose()).inverse().map_or(vec3(g.angular_velocity), |inv| inv * l);
                if g.crush[2] > imp.crush_used as f32 {
                    // Only the crush cap (impactor_crush) spends crush energy.
                    self.coverage |= coverage::CRUSH;
                }
                imp.crush_used = g.crush[2] as f64;
                imp.crush_depth = g.crush[3] as f64;
                self.contact_dissipated += g.ledger[0] as f64 + g.ledger[1] as f64;
                self.crush_energy += g.ledger[2] as f64 + g.ledger[3] as f64;
                g.ledger = [0.0; 4];
            }
            gpu.queue.write_buffer(&self.buffers.impactors, 0, bytemuck::cast_slice(&imps));
        }
        if p.solve_mode != 0 {
            // External loads (world.rs `loads`): the last substep's, and the frame's sum
            // (the GPU's is cleared once added here).
            // The last-substep loads, then the frame sums (adjacent in the scratch).
            let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&loads_bytes);
            self.last_loads.ensure_shape(m);
            self.frame_load_sum.ensure_shape(m);
            for (k, &(s, c)) in self.layout.chunks.iter().enumerate() {
                let (cl, cf) = (2 * k, 2 * n + 2 * k);
                self.last_loads.force[s][c] = vec3(out[cl]);
                self.last_loads.torque[s][c] = vec3(out[cl + 1]);
                self.frame_load_sum.force[s][c] += vec3(out[cf]);
                self.frame_load_sum.torque[s][c] += vec3(out[cf + 1]);
            }
            gpu.queue.write_buffer(&self.buffers.scratch, (16 * p.cframe_base as usize) as u64, bytemuck::cast_slice(&vec![[0f32; 4]; 2 * n]));
        }
        let plan = &self.layout.plan;
        if contacts {
            // Contact dissipation: the pair and chunk ledgers, the impactor candidates'.
            let mut ledgers: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&ledgers_bytes);
            for l in ledgers.iter_mut() {
                self.contact_dissipated += l[1] as f64 + l[2] as f64;
                l[1] = 0.0;
                l[2] = 0.0;
            }
            gpu.queue.write_buffer(&self.buffers.scratch, (16 * lb) as u64, bytemuck::cast_slice(&ledgers));
            if ce > cb {
                let mut cands: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&cands_bytes);
                for l in cands.iter_mut().step_by(3) {
                    self.contact_dissipated += l[2] as f64 + l[3] as f64;
                    l[2] = 0.0;
                    l[3] = 0.0;
                }
                gpu.queue.write_buffer(&self.buffers.scratch, (16 * cb) as u64, bytemuck::cast_slice(&cands));
            }
            if !plan.pairs.is_empty() {
                let cstate: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&cstate_bytes);
                for (i, (key, ..)) in plan.pairs.iter().enumerate() {
                    let (at, len) = self.layout.pair_state[i];
                    let st: PairState = cstate[at as usize..(at + len) as usize].to_vec();
                    if st.iter().all(|e| e[0].is_nan()) {
                        self.pair_memory.remove(key);
                    } else {
                        self.pair_memory.insert(*key, st);
                    }
                }
            }
        }
        self.state_current = true;
        bonds.then(|| bytemuck::pod_collect_to_vec(&bond_bytes))
    }

    /// The mirror's bonds from the GPU (and the GPU's per-sync accumulators reset).
    fn sync_bonds(&mut self, gpu: &Gpu) {
        let bond_dyn: Vec<BondDyn> = gpu.read(&self.buffers.bond_dyn);
        self.apply_bonds(gpu, bond_dyn);
    }

    fn apply_bonds(&mut self, gpu: &Gpu, bond_dyn: Vec<BondDyn>) {
        self.bonds_stale = false;
        self.apply_bond_range(gpu, 0, bond_dyn);
    }

    /// The GPU's records of bonds `first..first + bond_dyn.len()` into the mirror (their
    /// history, forces, energies and events), the per-download accumulators reset there.
    fn apply_bond_range(&mut self, gpu: &Gpu, first: usize, bond_dyn: Vec<BondDyn>) {
        let m = &mut self.mirror;
        // Events in the reference's order: by substep, then cluster, then bond.
        let mut events: Vec<(u32, usize, usize, u8, SolverEvent)> = Vec::new();
        let mut cleared = Vec::new();
        let (mut dissipated, mut overshoot, mut damped) = (0.0, 0.0, 0.0);
        // (A buffer for no bonds still holds one record: only the layout's are taken.)
        let count = bond_dyn.len().min(self.layout.bonds.len().saturating_sub(first));
        for (j, &(s, bi)) in self.layout.bonds[first..first + count].iter().enumerate() {
            let k = first + j;
            let bd = &bond_dyn[j];
            let b = &mut m.bonds[s][bi];
            let previous = b.joint.clone();
            b.joint = from_gpu_state(&bd.js, &previous);
            b.force = Local6 { lin: vec3(bd.force_lin), ang: vec3(bd.force_ang) };
            b.stored = bd.force_lin[3] as f64;
            m.max_utilization = m.max_utilization.max(bd.force_ang[3] as f64);
            dissipated += bd.sums[0] as f64 + bd.comps[0] as f64;
            overshoot += bd.sums[1] as f64 + bd.comps[1] as f64;
            damped += bd.sums[2] as f64 + bd.comps[2] as f64;
            let ci = m.chunks[s][b.geometry.a].cluster;
            let pose = m.clusters[ci].pose;
            let position = pose.transform_point(b.geometry.centroid).to_array();
            let time = |step: u32| self.step_times[step as usize];
            if bd.events[0] != 0 {
                let mode = mode_of(bd.events[3]);
                events.push((bd.events[0], ci, k, 0, SolverEvent::Cracked { time: time(bd.events[0]), structure: s, bond: bi, mode, position }));
            }
            if bd.events[1] != 0 {
                events.push((bd.events[1], ci, k, 1, SolverEvent::Creaked { time: time(bd.events[1]), structure: s, bond: bi, position }));
            }
            if bd.events[2] != 0 {
                let b = &m.bonds[s][bi];
                events.push((
                    bd.events[2],
                    ci,
                    k,
                    2,
                    SolverEvent::Broken {
                        time: time(bd.events[2]),
                        structure: s,
                        bond: bi,
                        mode: b.joint.mode,
                        position,
                        normal: pose.transform_vector(b.geometry.normal).to_array(),
                        dissipated: b.joint.dissipated,
                    },
                ));
            }
            if bd.events != [0; 4] || bd.sums != [0.0; 4] {
                cleared.push(j);
            }
        }
        events.sort_by_key(|e| (e.0, e.1, e.2, e.3));
        m.events.extend(events.into_iter().map(|e| e.4));
        m.energy.bond_dissipation += dissipated;
        m.energy.softening_overshoot += overshoot;
        m.energy.damping_dissipation += damped;
        // Reset the per-download accumulators and events on the GPU.
        if !cleared.is_empty() {
            let mut reset = bond_dyn;
            for &j in &cleared {
                reset[j].sums = [0.0; 4];
                reset[j].comps = [0.0; 4];
                reset[j].events = [0; 4];
            }
            gpu.queue.write_buffer(&self.buffers.bond_dyn, (std::mem::size_of::<BondDyn>() * first) as u64, bytemuck::cast_slice(&reset));
        }
        let _ = &self.layout.gpu_impactors;
    }

    /// Diagnostics: the time of each pipeline kernel, repeated `reps` times on the
    /// current buffers (the state is advanced meaningfully only by the island kernel).
    pub fn bench_kernels(&mut self, gpu: &Gpu, dt: f64, reps: usize) -> Vec<(&'static str, f64)> {
        let p = Params {
            dt: dt as f32,
            step_start: self.mirror.substeps as u32,
            t_hi: self.mirror.time as f32,
            t_lo: 0.0,
            max_steps: 1,
            contact_mode: 1,
            ..self.buffers.params
        };
        gpu.queue.write_buffer(&self.buffers.params_contact, 0, bytemuck::bytes_of(&p));
        let groups = |count: u32| count.div_ceil(64).max(1);
        let impactors_n = self.impactors.len().max(1) as u32;
        let mut out = Vec::new();
        eprintln!("  plan: {} pairs, {} impactor candidates, {} chunks, {} islands", p.pair_count, p.cand_count, p.chunk_count, self.layout.islands.len());
        let kernels: [(&'static str, &Kernel, u32); 5] = [
            ("impactor_shares", &self.kernels.shares, groups(p.cand_count)),
            ("impactor_crush", &self.kernels.crush, impactors_n),
            ("contact_forces", &self.kernels.forces, groups((p.pair_count + p.cand_count) * p.pair_lanes.max(1) + p.chunk_count + p.impactor_count)),
            ("island_frame (1 substep)", &self.kernels.island, self.layout.islands.len().max(1) as u32),
            ("impactor_integrate", &self.kernels.integrate, impactors_n),
        ];
        for (name, kernel, count) in kernels {
            let mut islands = self.layout.islands.clone();
            for isl in islands.iter_mut() {
                isl.info[1] = reps as u32;
                isl.info[2] = 0;
                isl.info[0] |= ISLAND_CONTACT;
            }
            islands.push(Island::default());
            gpu.queue.write_buffer(&self.buffers.islands, 0, bytemuck::cast_slice(&islands));
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            let t = std::time::Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&kernel.pipeline);
                pass.set_bind_group(0, &self.buffers.bind_contact, &[]);
                for _ in 0..reps {
                    pass.dispatch_workgroups(count, 1, 1);
                }
            }
            gpu.queue.submit([encoder.finish()]);
        self.state_current = false;
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            out.push((name, t.elapsed().as_secs_f64() / reps as f64));
        }
        out
    }

    /// Clusters the GPU would not touch, by id (sorted): settled, anchored and moved by
    /// the engine (`driven`: never in a contact plan), in a scene without scripted loads, events or probes,
    /// with no engine load on any chunk (`extra_terms`), whose every bond's static fatigue
    /// over a frame stays a thousandth of half an f32 ulp of its life (so the GPU's
    /// `settled_fatigue` leaves it bit for bit). The kernels skip such an island's phases
    /// (`ISLAND_QUIET`), so leaving its chunks and bonds out changes nothing else.
    /// `STRESS_GPU_QUIET_EXCLUDE=0`: none.
    fn quiet_clusters(&self) -> Vec<u64> {
        let m = &self.mirror;
        let Some(l) = self.loads.as_ref() else { return Vec::new() };
        let sc = &l.scene;
        // Adaptive mode, and the engine-coupled quasi-static mode (`static_stable`: a quiet
        // cluster's static solve repeats bit for bit and its damage commit changes nothing).
        let tiered_qs = m.config.mode == SolveMode::QuasiStatic && self.quiet_quasi_static;
        if (m.config.mode != SolveMode::Adaptive && !tiered_qs)
            || !sc.loads.is_empty()
            || !sc.events.is_empty()
            || !sc.probes.is_empty()
            || std::env::var("STRESS_GPU_QUIET_EXCLUDE").is_ok_and(|v| v == "0")
        {
            return Vec::new();
        }
        let frame_dt = sc.sim.frame_dt;
        let mut loaded = vec![false; m.clusters.len()];
        for t in &self.extra_terms {
            if let Some(cs) = m.chunks.get(t.structure).and_then(|v| v.get(t.chunk)) {
                if let Some(x) = loaded.get_mut(cs.cluster) {
                    *x = true;
                }
            }
        }
        let fatigue_still = |s: usize, bi: usize| {
            let b = &m.bonds[s][bi];
            let Some(f) = b.strength.static_fatigue else { return true };
            let idx = stress_ref::joint::failure_indices(&b.strength, &stress_ref::joint::stress_measures(&b.geometry, &b.force), b.weibull);
            let ratio = idx.tension.max(idx.shear).max(idx.compression);
            let life = 1.0 - b.joint.fatigue;
            f.life_rate(ratio) * frame_dt < 1e-3 * life * (-27f64).exp2()
        };
        let mut ids: Vec<u64> = m
            .clusters
            .iter()
            .enumerate()
            .filter(|(ci, cl)| {
                cl.activity == Activity::Settled
                    && cl.anchored
                    // Moved by the engine, not owned by an impact island: never in a
                    // contact plan (`integrate_rigid` off makes every cluster count as
                    // driven to the kernels, island members included, so it is not enough).
                    && cl.driven
                    && !loaded[*ci]
                    // An excluded cluster's bonds are not read back, so its verdict holds.
                    && (self.excluded.binary_search(&cl.id).is_ok() || cl.bonds.iter().all(|&bi| fatigue_still(cl.structure, bi)))
                    && (!tiered_qs || self.excluded.binary_search(&cl.id).is_ok() || self.static_stable.binary_search(&cl.id).is_ok())
            })
            .map(|(_, cl)| cl.id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// The ids of the clusters left out of the GPU tables as quiet (sorted).
    /// Each island's last static solve result (residual, Newton steps, CG iterations,
    /// converged), with its chunk and bond counts: a diagnostic read.
    pub fn statics_results(&self, gpu: &Gpu) -> Vec<(f32, u32, u32, bool, u32, u32)> {
        let p = self.buffers.params;
        let at = p.statics_base as usize + 23 * self.layout.chunks.len() + 2 * self.layout.bonds.len();
        let r: Vec<[f32; 4]> = gpu.read_range(&self.buffers.scratch, at, self.layout.islands.len());
        r.iter().zip(&self.layout.islands).map(|(v, isl)| (v[0], v[1].to_bits(), v[2].to_bits(), v[3] != 0.0, isl.range[1] - isl.range[0], isl.range[3] - isl.range[2])).collect()
    }

    /// Islands in the current layout.
    pub fn layout_islands(&self) -> usize {
        self.layout.islands.len()
    }

    /// The GPU chunks' (structure, chunk), by GPU chunk index.
    pub fn gpu_chunks(&self) -> &[(usize, usize)] {
        &self.layout.chunks
    }

    /// The buffers the resident frame's kernels share (`resident.rs`): islands, chunk
    /// statics, chunk state, scratch; and the static solve's load slots in scratch.
    pub(crate) fn resident_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u32) {
        (&self.buffers.islands, &self.buffers.chunk_static, &self.buffers.state, &self.buffers.scratch, self.buffers.params.sload_base)
    }

    pub fn excluded_ids(&self) -> &[u64] {
        &self.excluded
    }

    /// How many clusters are left out of the GPU tables as quiet (`quiet_clusters`).
    pub fn quiet_count(&self) -> usize {
        self.excluded.len()
    }

    /// `download` for a relayout, keeping what the host set since the GPU last ran:
    /// cluster motion and activity, and the impactors.
    fn download_keeping_host_motion(&mut self, gpu: &Gpu) {
        let host: Vec<_> = self
            .mirror
            .clusters
            .iter()
            .map(|cl| (cl.pose, cl.velocity, cl.angular_velocity, cl.driven, cl.activity, cl.active_timer, cl.settled_load_norm))
            .collect();
        let impactors = self.impactors.clone();
        self.download(gpu);
        for (cl, h) in self.mirror.clusters.iter_mut().zip(host) {
            (cl.pose, cl.velocity, cl.angular_velocity, cl.driven, cl.activity, cl.active_timer, cl.settled_load_norm) = h;
        }
        self.impactors = impactors;
    }

    /// Re-lay the per-step data (load terms, island poses) from the mirror, keeping the
    /// GPU's chunk state and bond history: for an engine's new frame (poses, contact
    /// loads). The mirror's bonds may lag (nothing is taken from them).
    ///
    /// Also where the quiet clusters left out of the tables are chosen (`quiet_clusters`):
    /// when that set changes, everything is laid out again from the mirror, brought up
    /// to the GPU's state first.
    pub fn refresh(&mut self, gpu: &Gpu) -> Result<(), String> {
        let stride = self.buffers.params.probe_stride;
        self.plan_until = self.mirror.time;
        let quiet = self.quiet_clusters();
        if quiet != self.excluded || self.layout_in_place {
            self.layout_in_place = false;
            if !self.state_current {
                self.download_keeping_host_motion(gpu);
            }
            self.sync(gpu);
            self.excluded = quiet;
            let (layout, buffers) =
                Self::build(gpu, &self.kernels, &self.mirror, self.loads.as_mut(), &self.impactors, &self.pair_memory, stride, 0.0, self.mirror.time, None, &self.extra_terms, &self.excluded, self.island_headroom, None)?;
            self.layout = layout;
            self.layout_generation += 1;
            self.buffers = buffers;
            return Ok(());
        }
        let (layout, buffers) =
            Self::build(gpu, &self.kernels, &self.mirror, self.loads.as_mut(), &self.impactors, &self.pair_memory, stride, 0.0, self.mirror.time, Some(&self.buffers), &self.extra_terms, &self.excluded, self.island_headroom, None)?;
        self.layout = layout;
        self.layout_generation += 1;
        self.buffers = buffers;
        Ok(())
    }

    /// Solve clusters to static equilibrium on the GPU (statics.rs `equilibrate_or_keep`)
    /// under the given external loads: all in one dispatch, one threadgroup each. The
    /// buffers must hold the mirror's state (as after `step` or `rebuild`). Converged
    /// clusters' equilibrium (displacements, zero velocities, reactions, bond forces) is
    /// copied into the mirror; the others keep their previous state on both sides.
    ///
    /// `sim.methods.tangent_statics`: the GPU runs the oracle's secant iteration for its
    /// SECANT_STEPS (`StaticOptions::secant_steps`, the `statics_newton` parameter). A
    /// cluster it leaves unconverged is solved from the same start by the oracle's own
    /// `equilibrate_or_keep` on the mirror (its secant, the globalised tangent re-solve,
    /// the secant's remaining steps), whose outcome, equilibrium or the kept state, is
    /// uploaded: exact delegation (`profile.statics_delegated`).
    /// `equilibrate` with everything left on the GPU (the fast quasi-static frame,
    /// `GpuWorld::fast_qs`): no read-back of the state or the bond history and no mirror
    /// update; an unconverged cluster's chunks get their state from before the solve back
    /// on the GPU (`equilibrate_or_keep`). Whether each converged.
    pub fn equilibrate_resident(&mut self, gpu: &Gpu, clusters: &[usize], loads: &ChunkLoads) -> Vec<bool> {
        self.equilibrate_resident_with(gpu, clusters, Some(loads), &|_, _| {})
    }

    /// world.rs `fast_static_tail`'s cascade of solves and damage commits from pass
    /// `first` on, every pass encoded in one submission and the next one's run decided on
    /// the GPU (`shaders/cascade.slang`); `pre(solver, encoder, pass)` forms each pass's
    /// loads (encoded before its solve). Nothing is read: the caller reads the status
    /// record (`CascadeRun`), with whatever else the frame reports. The same work as
    /// `equilibrate_resident` and `commit_resident` per pass (bit for bit), with the
    /// unconverged solves' states restored on the GPU.
    pub fn resident_cascade(&mut self, gpu: &Gpu, frame_dt: f64, first: usize, passes: usize, pre: &dyn Fn(&GpuSolver, &mut wgpu::CommandEncoder, usize)) -> crate::resident::CascadeRun {
        let n_islands = self.layout.islands.len();
        if self.cascade.is_none() {
            self.cascade = Some(crate::resident::CascadeKernels::new(gpu));
        }
        let generation = self.layout_generation;
        if self.cascade_islands.as_ref().is_none_or(|(g, _)| *g != generation) {
            let mut with_sentinel = self.layout.islands.clone();
            with_sentinel.push(Island::default());
            self.cascade_islands = Some((generation, gpu.storage("cascade layout islands", &with_sentinel)));
        }
        let words = crate::resident::CASCADE_ST_ISLANDS + n_islands + 1;
        if self.cascade_status.as_ref().is_none_or(|b| b.size() < 4 * words as u64) {
            self.cascade_status = Some(gpu.storage("cascade status", &vec![0u32; words]));
        }
        let size = self.buffers.state.size();
        if self.state_backup.as_ref().is_none_or(|b| b.size() != size) {
            self.state_backup = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("state backup"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        }
        // `STRESS_QS_REPEAT_GATE=0`: every flagged island solved, repeats included.
        let gate_on = std::env::var("STRESS_QS_REPEAT_GATE").map_or(true, |v| v != "0");
        // `STRESS_QS_CERTIFY=0`: a converged solve's repeat is solved again (one polishing
        // Newton step at f32's floor) instead of certified (cascade.slang).
        let certify = gate_on && std::env::var("STRESS_QS_CERTIFY").map_or(true, |v| v != "0");
        let n_chunks = self.layout.chunks.len();
        if self.cascade_gate.as_ref().is_none_or(|g| g.generation != generation) {
            // A new layout (a split, a rebuild): every island whose cluster kept its chunks
            // keeps its gate entry (its loads, chunk states and bond records are where they
            // were: the tables were rebuilt from the synced mirror).
            let buffer = gpu.storage("cascade gate", &vec![[0f32; 4]; 2 * n_chunks + n_islands + 1]);
            let ids: Vec<u64> = self.mirror.clusters.iter().map(|c| c.id).collect();
            let carry = std::env::var("STRESS_QS_GATE_CARRY").map_or(true, |v| v != "0");
            if let Some(old) = self.cascade_gate.take().filter(|_| carry) {
                let at: HashMap<u64, usize> = old.ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                for (i, isl) in self.layout.islands.iter().enumerate() {
                    let Some(&j) = at.get(&ids[i]) else { continue };
                    let (r, q) = (isl.range, old.ranges[j]);
                    let (new_chunks, old_chunks) = (&self.layout.chunks[r[0] as usize..r[1] as usize], &old.chunks[q[0] as usize..q[1] as usize]);
                    if new_chunks != old_chunks || r[3] - r[2] != q[3] - q[2] {
                        continue;
                    }
                    if r[1] > r[0] {
                        encoder.copy_buffer_to_buffer(&old.buffer, 16 * 2 * q[0] as u64, &buffer, 16 * 2 * r[0] as u64, 16 * 2 * (r[1] - r[0]) as u64);
                    }
                    encoder.copy_buffer_to_buffer(&old.buffer, 16 * (2 * old.chunks.len() + j) as u64, &buffer, 16 * (2 * n_chunks + i) as u64, 16);
                }
                gpu.queue.submit([encoder.finish()]);
            }
            self.cascade_gate = Some(CascadeGate { generation, buffer, ids, ranges: self.layout.islands.iter().map(|i| i.range).collect(), chunks: self.layout.chunks.clone() });
        }
        let gate = self.cascade_gate.as_ref().expect("cascade gate").buffer.clone();
        let bond_bytes = self.buffers.bond_dyn.size();
        if self.bond_backup.as_ref().is_none_or(|b| b.size() != bond_bytes) {
            self.bond_backup = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("bond backup"), size: bond_bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        }
        let bond_backup = self.bond_backup.clone().expect("bond backup");
        let status = self.cascade_status.clone().expect("cascade status");
        let layout_islands = self.cascade_islands.as_ref().expect("cascade islands").1.clone();
        let backup = self.state_backup.clone().expect("state backup");
        // The status record: the first pass runs.
        let mut init = vec![0u32; crate::resident::CASCADE_ST_ISLANDS];
        init[0] = 1;
        gpu.queue.write_buffer(&status, 0, bytemuck::cast_slice(&init));
        let p = self.buffers.params;
        let result_base = (p.statics_base as usize + 23 * self.layout.chunks.len() + 2 * self.layout.bonds.len()) as u32;
        let commit_params = |dt: f64| gpu.storage("cascade commit params", std::slice::from_ref(&Params { dt: dt as f32, ..p }));
        let commit_first = commit_params(frame_dt);
        let commit_later = commit_params(0.0);
        let params_bytes = std::mem::size_of::<Params>() as u64;
        let orig = gpu.storage("cascade params", std::slice::from_ref(&p));
        let kernels = self.cascade.as_ref().expect("cascade kernels");
        let islands_bytes = std::mem::size_of::<Island>() as u64 * (n_islands as u64 + 1);
        // `STRESS_GPU_TIMING=1`: the solves' and commits' GPU times (a diagnostic read).
        let timing = (std::env::var("STRESS_GPU_TIMING").is_ok_and(|v| v == "1") && gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY))
            .then(|| gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("cascade timestamps"), ty: wgpu::QueryType::Timestamp, count: 4 * passes as u32 }));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for pass in first..first + passes {
            let cp = gpu.uniform(
                "cascade pass",
                &crate::resident::CascadeParams {
                    island_count: n_islands as u32,
                    result_base,
                    pass_index: pass as u32,
                    sload_base: p.sload_base,
                    gate_flag_base: 2 * n_chunks as u32,
                    bond_words: (std::mem::size_of::<BondDyn>() / 16) as u32,
                    certify: certify as u32,
                    unconverged: match self.mirror.config.static_unconverged {
                        stress_ref::solver::UnconvergedSolve::Restore => 0,
                        stress_ref::solver::UnconvergedSolve::KeepIterate => 1,
                        stress_ref::solver::UnconvergedSolve::CommitIterate => 2,
                    },
                },
            );
            let bind = gpu.bind(&kernels.begin.group, &[&cp, &self.buffers.islands, &self.buffers.scratch, &self.buffers.state, &backup, &status, &gate, &self.buffers.bond_dyn, &bond_backup]);
            let run = |encoder: &mut wgpu::CommandEncoder, k: &crate::gpu::Kernel, groups: u32| {
                let mut c = encoder.begin_compute_pass(&Default::default());
                c.set_pipeline(&k.pipeline);
                c.set_bind_group(0, &bind, &[]);
                c.dispatch_workgroups(groups, 1, 1);
            };
            let world = |encoder: &mut wgpu::CommandEncoder, k: &Kernel, slot: u32| {
                let stamps = timing.as_ref().map(|q| wgpu::ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(2 * slot), end_of_pass_write_index: Some(2 * slot + 1) });
                let mut c = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: stamps });
                c.set_pipeline(&k.pipeline);
                c.set_bind_group(0, &self.buffers.bind_segment, &[]);
                c.dispatch_workgroups(n_islands.max(1) as u32, 1, 1);
            };
            // equilibrate_resident: the islands flagged, the states backed up, the loads,
            // the solves, the unconverged ones restored.
            encoder.copy_buffer_to_buffer(&layout_islands, 0, &self.buffers.islands, 0, islands_bytes);
            run(&mut encoder, &kernels.begin, crate::gpu::groups(n_islands, 64));
            encoder.copy_buffer_to_buffer(&self.buffers.state, 0, &backup, 0, size);
            if gate_on {
                encoder.copy_buffer_to_buffer(&self.buffers.bond_dyn, 0, &bond_backup, 0, bond_bytes);
            }
            pre(self, &mut encoder, pass);
            if gate_on {
                run(&mut encoder, &kernels.gate, n_islands.max(1) as u32);
            }
            world(&mut encoder, &self.kernels.statics, 2 * (pass - first) as u32);
            run(&mut encoder, &kernels.restore, n_islands.max(1) as u32);
            // commit_resident: the converged islands flagged, their damage committed with
            // the frame's duration (the first pass) or none.
            encoder.copy_buffer_to_buffer(&layout_islands, 0, &self.buffers.islands, 0, islands_bytes);
            run(&mut encoder, &kernels.settle, crate::gpu::groups(n_islands, 64));
            encoder.copy_buffer_to_buffer(if pass == 0 { &commit_first } else { &commit_later }, 0, &self.buffers.params_segment, 0, params_bytes);
            world(&mut encoder, &self.kernels.fatigue, 2 * (pass - first) as u32 + 1);
            encoder.copy_buffer_to_buffer(&orig, 0, &self.buffers.params_segment, 0, params_bytes);
            if gate_on {
                run(&mut encoder, &kernels.fixed, n_islands.max(1) as u32);
            }
            run(&mut encoder, &kernels.end, 1);
        }
        let resolve = timing.as_ref().map(|q| {
            let b = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("cascade timestamps"), size: 8 * 4 * passes as u64, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            encoder.resolve_query_set(q, 0..4 * passes as u32, &b, 0);
            b
        });
        gpu.queue.submit([encoder.finish()]);
        if let Some(b) = resolve {
            let t: Vec<u64> = gpu.read(&b);
            let period = gpu.queue.get_timestamp_period() as f64 * 1e-6;
            let ms: Vec<String> = (0..2 * passes).map(|q| format!("{:.2}", t[2 * q + 1].saturating_sub(t[2 * q]) as f64 * period)).collect();
            eprintln!("cascade GPU ms (solve, commit per pass): {}", ms.join(" "));
        }
        self.state_current = false;
        self.bonds_stale = true;
        self.coverage |= coverage::STATICS | coverage::SETTLED_FATIGUE;
        crate::resident::CascadeRun { status, words, islands: n_islands }
    }

    /// `equilibrate_resident` with the loads from the host (`loads`) or, with `None`,
    /// formed on the GPU by `pre`, which is encoded after the island records are written
    /// and before the solve (the resident frame: `resident.rs`).
    pub fn equilibrate_resident_with(&mut self, gpu: &Gpu, clusters: &[usize], loads: Option<&ChunkLoads>, pre: &dyn Fn(&GpuSolver, &mut wgpu::CommandEncoder)) -> Vec<bool> {
        if clusters.is_empty() {
            return Vec::new();
        }
        let p = self.buffers.params;
        for &ci in clusters {
            let Some(loads) = loads else { break };
            let f = crate::statics::static_loads(&self.mirror, ci, loads);
            let first = self.layout.islands[ci].range[0] as usize;
            let staged: Vec<[f32; 4]> = f.iter().flat_map(|(fo, mo)| [v4(*fo, 0.0), v4(*mo, 0.0)]).collect();
            if !staged.is_empty() {
                gpu.queue.write_buffer(&self.buffers.scratch, 16 * (p.sload_base as usize + 2 * first) as u64, bytemuck::cast_slice(&staged));
            }
        }
        let mut islands = self.layout.islands.clone();
        for isl in islands.iter_mut() {
            isl.info[2] &= !ISLAND_SOLVE;
        }
        for &ci in clusters {
            islands[ci].info[2] |= ISLAND_SOLVE;
        }
        let mut with_sentinel = islands.clone();
        with_sentinel.push(Island::default());
        gpu.queue.write_buffer(&self.buffers.islands, 0, bytemuck::cast_slice(&with_sentinel));
        let size = self.buffers.state.size();
        if self.state_backup.as_ref().is_none_or(|b| b.size() != size) {
            // Storage too: the resident cascade restores from it in a kernel.
            self.state_backup = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("state backup"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        }
        let backup = self.state_backup.clone().expect("state backup");
        let range = |ci: usize| {
            let isl = self.layout.islands[ci];
            (64 * isl.range[0] as u64, 64 * (isl.range[1] - isl.range[0]) as u64)
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for &ci in clusters {
            let (at, len) = range(ci);
            if len > 0 {
                encoder.copy_buffer_to_buffer(&self.buffers.state, at, &backup, at, len);
            }
        }
        pre(self, &mut encoder);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernels.statics.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_segment, &[]);
            pass.dispatch_workgroups(islands.len().max(1) as u32, 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        self.state_current = false;
        self.bonds_stale = true;
        let result_at = p.statics_base as usize + 23 * self.layout.chunks.len() + 2 * self.layout.bonds.len();
        let results: Vec<[f32; 4]> = gpu.read_range(&self.buffers.scratch, result_at, islands.len());
        let converged: Vec<bool> = clusters.iter().map(|&ci| results[ci][3] != 0.0).collect();
        // `STRESS_QS_WARM=1` (with a budget, as Blast's solver): an unconverged solve keeps
        // its iterate, so the next frame's continues from it instead of from the start.
        let warm = std::env::var("STRESS_QS_WARM").is_ok_and(|v| v == "1");
        if !warm && converged.iter().any(|&c| !c) {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            for (&ci, &ok) in clusters.iter().zip(&converged) {
                let (at, len) = range(ci);
                if !ok && len > 0 {
                    encoder.copy_buffer_to_buffer(&backup, at, &self.buffers.state, at, len);
                }
            }
            gpu.queue.submit([encoder.finish()]);
        }
        if converged.iter().any(|&c| c) {
            self.coverage |= coverage::STATICS;
        }
        converged
    }

    /// statics.rs `commit_damage` of `clusters` on the GPU (`settled_fatigue` over just
    /// them, at their equilibrium, with `dt`): (damage grew, a bond disconnected).
    pub fn commit_resident(&mut self, gpu: &Gpu, clusters: &[usize], dt: f64) -> (Vec<usize>, Vec<usize>) {
        if clusters.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let p = Params { dt: dt as f32, ..self.buffers.params };
        gpu.queue.write_buffer(&self.buffers.params_segment, 0, bytemuck::bytes_of(&p));
        let mut islands = self.layout.islands.clone();
        for isl in islands.iter_mut() {
            isl.info[0] &= !ISLAND_SETTLED;
            isl.info[2] &= !(ISLAND_DAMAGED | ISLAND_DISCONNECTED);
        }
        for &ci in clusters {
            islands[ci].info[0] |= ISLAND_SETTLED;
        }
        islands.push(Island::default());
        gpu.queue.write_buffer(&self.buffers.islands, 0, bytemuck::cast_slice(&islands));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernels.fatigue.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_segment, &[]);
            pass.dispatch_workgroups(islands.len().max(1) as u32, 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        self.state_current = false;
        self.bonds_stale = true;
        let mut read: Vec<Island> = gpu.read_range(&self.buffers.islands, 0, self.layout.islands.len() + 1);
        read.pop();
        self.coverage |= read.iter().fold(0, |a, i| a | i.done[1]) as u64;
        // The tables' own flags back for the next dispatch.
        gpu.queue.write_buffer(&self.buffers.params_segment, 0, bytemuck::bytes_of(&self.buffers.params));
        let damaged = clusters.iter().copied().filter(|&ci| read[ci].info[2] & ISLAND_DAMAGED != 0).collect();
        let disconnected = clusters.iter().copied().filter(|&ci| read[ci].info[2] & ISLAND_DISCONNECTED != 0).collect();
        (damaged, disconnected)
    }

    pub fn equilibrate(&mut self, gpu: &Gpu, clusters: &[usize], loads: &ChunkLoads) -> Vec<crate::statics::Equilibrium> {
        if clusters.is_empty() {
            return Vec::new();
        }
        // `STRESS_GPU_CHECK_STATICS=1` (diagnostic): every solve compared with the
        // oracle's `equilibrate_or_keep` of the same cluster from the same state.
        let oracle = std::env::var("STRESS_GPU_CHECK_STATICS").is_ok_and(|v| v == "1").then(|| self.mirror.clone());
        let p = self.buffers.params;
        // External loads per chunk (body frame), contiguous by chunk: one write per cluster.
        for &ci in clusters {
            let f = crate::statics::static_loads(&self.mirror, ci, loads);
            let first = self.layout.islands[ci].range[0] as usize;
            let staged: Vec<[f32; 4]> = f.iter().flat_map(|(fo, mo)| [v4(*fo, 0.0), v4(*mo, 0.0)]).collect();
            if !staged.is_empty() {
                gpu.queue.write_buffer(&self.buffers.scratch, 16 * (p.sload_base as usize + 2 * first) as u64, bytemuck::cast_slice(&staged));
            }
        }
        let mut islands = self.layout.islands.clone();
        for isl in islands.iter_mut() {
            isl.info[2] &= !ISLAND_SOLVE;
        }
        for &ci in clusters {
            islands[ci].info[2] |= ISLAND_SOLVE;
        }
        let mut with_sentinel = islands.clone();
        with_sentinel.push(Island::default());
        gpu.queue.write_buffer(&self.buffers.islands, 0, bytemuck::cast_slice(&with_sentinel));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernels.statics.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_segment, &[]);
            pass.dispatch_workgroups(islands.len().max(1) as u32, 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        self.state_current = false;
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let result_at = p.statics_base as usize + 23 * self.layout.chunks.len() + 2 * self.layout.bonds.len();
        let results: Vec<[f32; 4]> = gpu.read_range(&self.buffers.scratch, result_at, islands.len());
        let state: Vec<[f32; 4]> = gpu.read(&self.buffers.state);
        let bond_dyn: Vec<BondDyn> = gpu.read(&self.buffers.bond_dyn);
        let tangent = self.mirror.config.methods.tangent_statics;
        let mut out = Vec::with_capacity(clusters.len());
        for &ci in clusters {
            let r = results[ci];
            let mut eq = crate::statics::Equilibrium { converged: r[3] != 0.0, residual: r[0] as f64, newton_iterations: r[1].to_bits(), cg_iterations: r[2].to_bits() };
            if eq.newton_iterations > 0 {
                self.coverage |= coverage::STATICS;
            }
            let isl = self.layout.islands[ci];
            let (c0, c1) = (isl.range[0] as usize, isl.range[1] as usize);
            if eq.converged {
                let m = &mut self.mirror;
                for k in c0..c1 {
                    let (s, c) = self.layout.chunks[k];
                    let cs = &mut m.chunks[s][c];
                    cs.u = vec3(state[4 * k]);
                    cs.th = vec3(state[4 * k + 1]);
                    cs.v = Vec3::ZERO;
                    cs.w = Vec3::ZERO;
                    if m.structures[s].chunks[c].support != Support::None {
                        cs.reaction = (Vec3::new(state[4 * k + 1][3] as f64, state[4 * k + 2][3] as f64, state[4 * k + 3][3] as f64), Vec3::ZERO);
                    }
                }
                for k in isl.range[2] as usize..isl.range[3] as usize {
                    let (s, bi) = self.layout.bonds[k];
                    let b = &mut m.bonds[s][bi];
                    b.force = Local6 { lin: vec3(bond_dyn[k].force_lin), ang: vec3(bond_dyn[k].force_ang) };
                    b.stored = bond_dyn[k].force_lin[3] as f64;
                }
            } else {
                if tangent {
                    // statics.rs `equilibrate`: the secant left the cluster unconverged
                    // within SECANT_STEPS; the oracle solves it again from the same start
                    // (the mirror's: the GPU's iterate is discarded).
                    let t = std::time::Instant::now();
                    let host = self.mirror.equilibrate_or_keep(ci, loads, &StaticOptions::default());
                    self.profile.statics_delegated += 1;
                    self.profile.statics_delegated_seconds += t.elapsed().as_secs_f64();
                    self.coverage |= coverage::STATICS_DELEGATED;
                    eq = crate::statics::Equilibrium {
                        converged: host.converged,
                        residual: host.residual,
                        newton_iterations: eq.newton_iterations + host.newton_iterations as u32,
                        cg_iterations: eq.cg_iterations + host.cg_iterations as u32,
                    };
                    // Its bond forces (refreshed at the equilibrium, or at the kept
                    // state), with the GPU's history and accumulators.
                    let (b0, b1) = (isl.range[2] as usize, isl.range[3] as usize);
                    let forces: Vec<BondDyn> = (b0..b1)
                        .map(|k| {
                            let (s, bi) = self.layout.bonds[k];
                            let b = &self.mirror.bonds[s][bi];
                            BondDyn { force_lin: v4(b.force.lin, b.stored), force_ang: v4(b.force.ang, bond_dyn[k].force_ang[3] as f64), ..bond_dyn[k] }
                        })
                        .collect();
                    if !forces.is_empty() {
                        gpu.queue.write_buffer(&self.buffers.bond_dyn, (std::mem::size_of::<BondDyn>() * b0) as u64, bytemuck::cast_slice(&forces));
                    }
                }
                // statics.rs `equilibrate_or_keep`: the mirror's state (the state before
                // the solve, or the oracle's equilibrium under `tangent_statics`).
                let m = &self.mirror;
                let back: Vec<[f32; 4]> = (c0..c1)
                    .flat_map(|k| {
                        let (s, c) = self.layout.chunks[k];
                        let cs = &m.chunks[s][c];
                        let r = cs.reaction.0;
                        [v4(cs.u, 0.0), v4(cs.th, r.x), v4(cs.v, r.y), v4(cs.w, r.z)]
                    })
                    .collect();
                gpu.queue.write_buffer(&self.buffers.state, (64 * c0) as u64, bytemuck::cast_slice(&back));
            }
            out.push(eq);
        }
        if let Some(mut oracle) = oracle {
            self.check_statics(&mut oracle, clusters, loads, &out);
        }
        out
    }

    /// `STRESS_GPU_CHECK_STATICS=1`: the oracle's outcome of each solve (`oracle`: the
    /// mirror before it) against the GPU's: converged alike, and the equilibrium within
    /// the gate's tolerance (1e-3 of the cluster's displacement range). Counted in
    /// `profile.statics_checked`, `statics_mismatched`; each mismatch printed.
    fn check_statics(&mut self, oracle: &mut ReferenceSolver, clusters: &[usize], loads: &ChunkLoads, gpu: &[crate::statics::Equilibrium]) {
        for (&ci, g) in clusters.iter().zip(gpu) {
            let r = oracle.equilibrate_or_keep(ci, loads, &StaticOptions::default());
            let cl = &oracle.clusters[ci];
            let s = cl.structure;
            let range = cl.chunks.iter().map(|&c| oracle.chunks[s][c].u.norm().max(oracle.chunks[s][c].th.norm())).fold(0.0, f64::max);
            let err = cl
                .chunks
                .iter()
                .map(|&c| {
                    let (a, b) = (&oracle.chunks[s][c], &self.mirror.chunks[s][c]);
                    (a.u - b.u).norm().max((a.th - b.th).norm())
                })
                .fold(0.0, f64::max);
            self.profile.statics_checked += 1;
            if g.converged != r.converged || err > 1e-3 * range {
                self.profile.statics_mismatched += 1;
                eprintln!(
                    "statics check: cluster {ci} ({} chunks): GPU converged {} residual {:.3e} ({} Newton) | oracle converged {} residual {:.3e} ({} Newton); displacement error {err:.3e} of {range:.3e}",
                    cl.chunks.len(),
                    g.converged,
                    g.residual,
                    g.newton_iterations,
                    r.converged,
                    r.residual,
                    r.newton_iterations
                );
            }
        }
    }

    /// world.rs `settle_quiet_clusters` (adaptive mode): clusters whose last dynamic load
    /// is older than `active_time` and whose deformation has calmed down go to static
    /// equilibrium under the external loads the GPU recorded last (all such clusters in
    /// one solve; `host`: the reference's own solve on the mirror). True if any did (or
    /// tried to). The mirror's bonds must be current (`sync`).
    pub fn settle_quiet_clusters(&mut self, gpu: &Gpu, frame_dt: f64, host: bool) -> bool {
        let loads: ChunkLoads = self.last_loads.clone();
        let m = &mut self.mirror;
        let mut candidates = Vec::new();
        for ci in 0..m.clusters.len() {
            if m.clusters[ci].activity != Activity::Active {
                continue;
            }
            m.clusters[ci].active_timer -= frame_dt;
            if m.clusters[ci].active_timer > 0.0 || m.clusters[ci].bonds.is_empty() {
                continue;
            }
            let s = m.clusters[ci].structure;
            let ke: f64 = m.clusters[ci]
                .chunks
                .iter()
                .map(|&c| {
                    let ch = &m.structures[s].chunks[c];
                    let st = &m.chunks[s][c];
                    0.5 * ch.mass * st.v.norm2() + 0.5 * st.w.dot(ch.inertia * st.w)
                })
                .sum();
            let stored: f64 = m.clusters[ci].bonds.iter().map(|&b| m.bonds[s][b].stored).sum();
            if ke > 1e-3 * stored.max(1e-9) {
                continue;
            }
            candidates.push(ci);
        }
        if candidates.is_empty() {
            return false;
        }
        let results: Vec<bool> = if host {
            let opts = stress_ref::statics::StaticOptions::default();
            candidates.iter().map(|&ci| self.mirror.equilibrate_or_keep(ci, &loads, &opts).converged).collect()
        } else {
            let diag = blowup_limit().is_some().then(|| self.mirror.clone());
            let eqs = self.equilibrate(gpu, &candidates, &loads);
            if let Some(mut oracle) = diag {
                // Diagnostic: the reference's own static solve of the same clusters from
                // the same state and loads.
                let opts = stress_ref::statics::StaticOptions::default();
                for (&ci, e) in candidates.iter().zip(&eqs) {
                    let r = oracle.equilibrate_or_keep(ci, &loads, &opts);
                    let cl = &oracle.clusters[ci];
                    let s = cl.structure;
                    let u = cl.chunks.iter().map(|&c| oracle.chunks[s][c].u.norm()).fold(0.0, f64::max);
                    let th = cl.chunks.iter().map(|&c| oracle.chunks[s][c].th.norm()).fold(0.0, f64::max);
                    let net: (Vec3, Vec3) = cl.chunks.iter().fold((Vec3::ZERO, Vec3::ZERO), |(f, t), &c| (f + loads.force[s][c], t + loads.torque[s][c]));
                    eprintln!(
                        "settle {ci}: GPU converged {} residual {:.3e} newton {} cg {}; reference converged {} residual {:.3e} newton {} cg {}, max |u| {u:.3e} |th| {th:.3e}; recorded loads net force {:?} (weight {:.1} N)",
                        e.converged, e.residual, e.newton_iterations, e.cg_iterations, r.converged, r.residual, r.newton_iterations, r.cg_iterations, net.0.to_array(), cl.mass * 9.81
                    );
                    if !r.converged {
                        // The reference kept the start state (`equilibrate_or_keep`).
                        let m0 = &oracle;
                        let st0 = |c: usize| &m0.chunks[s][c];
                        let dump = serde_json::json!({
                            "pose": [cl.pose.position.to_array(), [cl.pose.rotation.x, cl.pose.rotation.y, cl.pose.rotation.z, cl.pose.rotation.w]],
                            "chunks": cl.chunks.iter().map(|&c| serde_json::json!({
                                "index": c, "u": st0(c).u.to_array(), "th": st0(c).th.to_array(), "inertia_scale": st0(c).inertia_scale,
                                "force": loads.force[s][c].to_array(), "torque": loads.torque[s][c].to_array()
                            })).collect::<Vec<_>>(),
                            "bonds": cl.bonds.iter().map(|&bi| serde_json::json!({"source": m0.bonds[s][bi].source, "joint": m0.bonds[s][bi].joint})).collect::<Vec<_>>(),
                        });
                        eprintln!("DUMP {dump}");
                        for &c in &cl.chunks {
                            let ch = &m0.structures[s].chunks[c];
                            eprintln!("    chunk {c}: center {:?} half {:?} m {:.3} hull {} load f {:?} t {:?}", ch.center.to_array(), ch.half_extents.to_array(), ch.mass, ch.hull.is_some(), loads.force[s][c].to_array(), loads.torque[s][c].to_array());
                        }
                        for &bi in &cl.bonds {
                            let b = &m0.bonds[s][bi];
                            eprintln!("    bond {bi} ({} - {}): alive {}, area {:.4}, normal {:?}, centroid {:?}, joint {:?}", b.geometry.a, b.geometry.b, b.alive, b.geometry.area, b.geometry.normal.to_array(), b.geometry.centroid.to_array(), b.joint);
                        }
                    }
                }
            }
            eqs.iter().map(|e| e.converged).collect()
        };
        let m = &mut self.mirror;
        if blowup_limit().is_some() {
            for (&ci, &ok) in candidates.iter().zip(&results) {
                let cl = &m.clusters[ci];
                let s = cl.structure;
                let th = cl.chunks.iter().map(|&c| m.chunks[s][c].th.norm()).fold(0.0, f64::max);
                let u = cl.chunks.iter().map(|&c| m.chunks[s][c].u.norm()).fold(0.0, f64::max);
                eprintln!("settle cluster {ci} (id {}, {} chunks, {} bonds, anchored {}): converged {ok}, max |u| {u:.3e}, max |th| {th:.3e}", cl.id, cl.chunks.len(), cl.bonds.len(), cl.anchored);
            }
        }
        for (&ci, &ok) in candidates.iter().zip(&results) {
            if !ok {
                // No static equilibrium (a mechanism): it stays in the explicit solve.
                m.clusters[ci].active_timer = m.config.active_time;
                continue;
            }
            m.clusters[ci].activity = Activity::Settled;
            let cl = &m.clusters[ci];
            m.clusters[ci].settled_load_norm = cl.chunks.iter().map(|&c| loads.force[cl.structure][c].norm()).sum();
            self.coverage |= coverage::SETTLE;
        }
        true
    }

    /// The adaptive mode's frame end (world.rs `step_frame`): quiet clusters settle, then
    /// the settled clusters' static fatigue on the GPU; a bond whose damage grew wakes its
    /// cluster, a disconnection splits it. True if the activity or topology changed (the
    /// caller rebuilds).
    pub fn adaptive_tail(&mut self, gpu: &Gpu, frame_dt: f64) -> Result<bool, String> {
        let may_settle = self.mirror.clusters.iter().any(|c| c.activity == Activity::Active && !c.bonds.is_empty() && c.active_timer - frame_dt <= 0.0);
        if may_settle {
            self.sync(gpu);
        }
        // Every frame: it also counts the active clusters' timers down.
        self.mirror_check("tail start");
        if self.settle_quiet_clusters(gpu, frame_dt, false) {
            self.mirror_check("after settle_quiet_clusters");
            // The newly settled clusters take part in this frame's fatigue.
            self.rebuild(gpu, 0.0)?;
        }
        let (damaged, disconnected) = self.settled_fatigue(gpu, frame_dt);
        if damaged.is_empty() && disconnected.is_empty() {
            return Ok(false);
        }
        // A bond's damage grew: the cluster wakes; one disconnected: it splits.
        self.sync(gpu);
        let m = &mut self.mirror;
        let mut woke: Vec<usize> = damaged.iter().chain(&disconnected).copied().collect();
        woke.sort_unstable();
        woke.dedup();
        for &ci in &woke {
            m.clusters[ci].activity = Activity::Active;
            m.clusters[ci].active_timer = m.config.active_time;
        }
        if blowup_limit().is_some() {
            eprintln!("tail: damaged {damaged:?}, disconnected {disconnected:?}");
        }
        self.mirror_check("tail before split");
        let m = &mut self.mirror;
        for &ci in disconnected.iter().rev() {
            m.split_cluster(ci);
        }
        self.mirror_check("tail after split");
        Ok(true)
    }

    /// world.rs `advance_settled_fatigue` on the GPU: every settled cluster's bonds
    /// committed at its equilibrium with `frame_dt` of static fatigue. The clusters whose
    /// bonds' damage grew, and those with a disconnected bond (the mirror's bonds are then
    /// behind the GPU's until `sync`).
    pub fn settled_fatigue(&mut self, gpu: &Gpu, frame_dt: f64) -> (Vec<usize>, Vec<usize>) {
        if !self.layout.islands.iter().any(|isl| isl.info[0] & ISLAND_SETTLED != 0) {
            return (Vec::new(), Vec::new());
        }
        let p = Params { dt: frame_dt as f32, ..self.buffers.params };
        gpu.queue.write_buffer(&self.buffers.params_segment, 0, bytemuck::bytes_of(&p));
        let mut islands = self.layout.islands.clone();
        islands.iter_mut().for_each(|isl| isl.info[2] &= !(ISLAND_DAMAGED | ISLAND_DISCONNECTED));
        let mut with_sentinel = islands.clone();
        with_sentinel.push(Island::default());
        gpu.queue.write_buffer(&self.buffers.islands, 0, bytemuck::cast_slice(&with_sentinel));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.kernels.fatigue.pipeline);
            pass.set_bind_group(0, &self.buffers.bind_segment, &[]);
            pass.dispatch_workgroups(islands.len().max(1) as u32, 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        self.state_current = false;
        self.bonds_stale = true;
        let mut read: Vec<Island> = gpu.read_range(&self.buffers.islands, 0, self.layout.islands.len() + 1);
        read.pop();
        self.coverage |= read.iter().fold(0, |a, i| a | i.done[1]) as u64;
        let damaged = (0..read.len()).filter(|&ci| read[ci].info[2] & ISLAND_DAMAGED != 0).collect();
        let disconnected = (0..read.len()).filter(|&ci| read[ci].info[2] & ISLAND_DISCONNECTED != 0).collect();
        self.layout.islands = islands;
        (damaged, disconnected)
    }

    /// Write the mirror's bond state (history, forces) of `clusters` to the GPU, with the
    /// per-sync accumulators zeroed: for changes the host made to bonds that do not run
    /// on the GPU (settled clusters' static fatigue).
    pub fn upload_bonds(&mut self, gpu: &Gpu, clusters: &[usize]) {
        let m = &self.mirror;
        for &ci in clusters {
            let isl = self.layout.islands[ci];
            let (b0, b1) = (isl.range[2] as usize, isl.range[3] as usize);
            let data: Vec<BondDyn> = (b0..b1)
                .map(|k| {
                    let (s, bi) = self.layout.bonds[k];
                    let b = &m.bonds[s][bi];
                    BondDyn { js: to_gpu_state(&b.joint), force_lin: v4(b.force.lin, b.stored), force_ang: v4(b.force.ang, b.joint.utilization), ..Default::default() }
                })
                .collect();
            if !data.is_empty() {
                gpu.queue.write_buffer(&self.buffers.bond_dyn, (std::mem::size_of::<BondDyn>() * b0) as u64, bytemuck::cast_slice(&data));
            }
        }
    }

    /// The buffers a renderer draws the world from (`destruction_render.slang`): valid
    /// until the next rebuild or refresh, which replaces some of them.
    pub fn render_buffers(&self) -> RenderBuffers<'_> {
        let b = &self.buffers;
        RenderBuffers {
            chunks: &b.chunk_static,
            islands: &b.islands,
            state: &b.state,
            index: &b.index,
            bond_dyn: &b.bond_dyn,
            impactors: &b.impactors,
            chunk_count: self.layout.chunks.len() as u32,
            impactor_count: self.layout.gpu_impactors.len() as u32,
        }
    }

    /// GPU order of chunks (structure, chunk).
    pub fn chunk_order(&self) -> &[(usize, usize)] {
        &self.layout.chunks
    }

    pub fn island_count(&self) -> usize {
        self.layout.islands.len()
    }

    /// Time after `n` substeps since the start.
    pub fn time_after(&self, n: usize) -> f64 {
        self.step_times[n]
    }
}

/// Scratch float4 per chunk of a static solve (world.slang `STATICS_PER_CHUNK`).
const STATICS_PER_CHUNK: u64 = 23;

impl GpuSolver {
    /// `ReferenceSolver::split_cluster` of `parents` (cluster indices) laid out in place
    /// (docs/stress-fracture-gpu-resident.md phase 6e): only the parents' chunk states
    /// and bond records are read back; each parent's children take its chunk and bond
    /// ranges (their chunks and bonds in the order a full pack gives them), the cluster
    /// the split moves into a parent's index keeps its ranges, and only those ranges,
    /// the islands and the chunk-to-bond incidence of the parents' chunks are written.
    /// The other islands' tables (and their GPU equilibria) stay as they are.
    ///
    /// Returns false, having changed nothing, when the tables cannot take it in place
    /// (no island headroom left, contacts planned, quiet clusters left out, load terms):
    /// the caller syncs, splits and rebuilds.
    /// Hashes of the GPU chunk state and bond records (`STRESS_GPU_SPLIT_HASH`, a
    /// determinism diagnostic).
    pub(crate) fn state_hashes(&self, gpu: &Gpu) -> (u64, u64) {
        use std::hash::{Hash, Hasher};
        let hash = |bytes: &[u8]| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        };
        let st: Vec<[f32; 4]> = gpu.read(&self.buffers.state);
        let bd: Vec<u8> = gpu.read(&self.buffers.bond_dyn);
        if let Ok(dir) = std::env::var("STRESS_GPU_STATE_DUMP") {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = std::fs::write(format!("{dir}/state_{n}.bin"), bytemuck::cast_slice::<_, u8>(&st));
            let isl: Vec<[u32; 4]> = self.layout.islands.iter().map(|i| i.range).collect();
            let _ = std::fs::write(format!("{dir}/islands_{n}.json"), format!("{:?}", isl));
        }
        (hash(bytemuck::cast_slice(&st)), hash(&bd))
    }

    pub fn split_resident(&mut self, gpu: &Gpu, parents: &[usize]) -> Result<bool, String> {
        let why = |reason: &str| {
            if std::env::var("STRESS_GPU_QS_TRACE").is_ok() {
                eprintln!("split in place: not taken ({reason})");
            }
            Ok(false)
        };
        if self.layout.host.is_none() {
            return why("no host tables");
        }
        let plan = &self.layout.plan;
        if !self.excluded.is_empty() || !plan.pairs.is_empty() || !plan.ground.is_empty() || self.buffers.params.cand_count > 0 || !self.layout.items.is_empty() {
            return why("contacts planned, probes or quiet clusters");
        }
        let mut parents: Vec<usize> = parents.to_vec();
        parents.sort_unstable();
        parents.dedup();
        let old_islands = self.layout.islands.clone();
        let t_split = std::time::Instant::now();
        let trace = std::env::var("STRESS_GPU_QS_TRACE").is_ok();
        let mut stamps: Vec<(&str, f64)> = Vec::new();
        // The parents' chunk states and bond records, read back together.
        let f4 = 16u64;
        let bond_bytes = std::mem::size_of::<BondDyn>() as u64;
        let mut parts: Vec<(&wgpu::Buffer, u64, u64)> = Vec::new();
        for &ci in &parents {
            let r = old_islands[ci].range;
            parts.push((&self.buffers.state, 4 * r[0] as u64 * f4, 4 * (r[1] - r[0]) as u64 * f4));
            parts.push((&self.buffers.bond_dyn, r[2] as u64 * bond_bytes, (r[3] - r[2]) as u64 * bond_bytes));
        }
        let read = gpu.read_ranges(&parts);
        let mut parent_state: HashMap<usize, Vec<[f32; 4]>> = HashMap::new();
        let mut parent_bonds: HashMap<usize, Vec<BondDyn>> = HashMap::new();
        for (i, &ci) in parents.iter().enumerate() {
            parent_state.insert(ci, bytemuck::pod_collect_to_vec(&read[2 * i]));
            parent_bonds.insert(ci, bytemuck::pod_collect_to_vec(&read[2 * i + 1]));
        }
        if trace {
            stamps.push(("read", t_split.elapsed().as_secs_f64() * 1e3));
        }
        // Into the mirror: the parents' chunk states (sync's conversion) and bonds.
        for &ci in &parents {
            let r = old_islands[ci].range;
            let st = &parent_state[&ci];
            for k in r[0] as usize..r[1] as usize {
                let (s, c) = self.layout.chunks[k];
                let j = k - r[0] as usize;
                let m = &mut self.mirror;
                let cs = &mut m.chunks[s][c];
                cs.u = vec3(st[4 * j]);
                cs.th = vec3(st[4 * j + 1]);
                cs.v = vec3(st[4 * j + 2]);
                cs.w = vec3(st[4 * j + 3]);
                if m.structures[s].chunks[c].support != Support::None {
                    cs.reaction = (Vec3::new(st[4 * j + 1][3] as f64, st[4 * j + 2][3] as f64, st[4 * j + 3][3] as f64), Vec3::ZERO);
                }
            }
            self.apply_bond_range(gpu, r[2] as usize, parent_bonds[&ci].clone());
        }
        let host = self.layout.host.as_ref().expect("host tables");
        if trace {
            stamps.push(("apply", t_split.elapsed().as_secs_f64() * 1e3));
        }
        // Room for the children, before anything is split.
        let added: usize = parents.iter().map(|&ci| self.mirror.components(self.mirror.clusters[ci].structure, &self.mirror.clusters[ci].chunks.clone()).len().saturating_sub(1)).sum();
        if self.mirror.clusters.len() + added > self.layout.island_cap {
            return why("no island headroom left");
        }
        if trace {
            stamps.push(("components", t_split.elapsed().as_secs_f64() * 1e3));
        }
        // The split, on the host (as `split_cluster` does it: the last cluster moves into
        // a split parent's index, the children are appended).
        let old_ids: Vec<u64> = self.mirror.clusters.iter().map(|c| c.id).collect();
        let parent_ids: Vec<u64> = parents.iter().map(|&ci| old_ids[ci]).collect();
        for &ci in parents.iter().rev() {
            self.mirror.split_cluster(ci);
        }
        let m = &self.mirror;
        let n_new = m.clusters.len();
        if n_new > self.layout.island_cap {
            return Err("split in place: more children than counted".into());
        }
        if trace {
            stamps.push(("split_cluster", t_split.elapsed().as_secs_f64() * 1e3));
        }
        let old_index: HashMap<u64, usize> = old_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let old_chunk_at: HashMap<(usize, usize), usize> = parents
            .iter()
            .flat_map(|&ci| {
                let r = old_islands[ci].range;
                (r[0] as usize..r[1] as usize).map(|k| (self.layout.chunks[k], k)).collect::<Vec<_>>()
            })
            .collect();
        let old_bond_at: HashMap<(usize, usize), usize> = parents
            .iter()
            .flat_map(|&ci| {
                let r = old_islands[ci].range;
                (r[2] as usize..r[3] as usize).map(|b| (self.layout.bonds[b], b)).collect::<Vec<_>>()
            })
            .collect();
        let parent_of_chunk = |k: usize| parents.iter().copied().find(|&ci| (old_islands[ci].range[0] as usize..old_islands[ci].range[1] as usize).contains(&k));
        // The parents' records as they were (the tables are rewritten in place below).
        let mut old_chunk_rec: HashMap<usize, stress_layout::ChunkStatic> = HashMap::new();
        let mut old_bond_rec: HashMap<usize, stress_layout::BondStatic> = HashMap::new();
        let mut old_bond_key: HashMap<usize, (usize, usize)> = HashMap::new();
        for &ci in &parents {
            let r = old_islands[ci].range;
            for k in r[0] as usize..r[1] as usize {
                old_chunk_rec.insert(k, host.chunk_static[k]);
            }
            for b in r[2] as usize..r[3] as usize {
                old_bond_rec.insert(b, host.bond_static[b]);
                old_bond_key.insert(b, self.layout.bonds[b]);
            }
        }
        let host = self.layout.host.as_mut().expect("host tables");
        let mut chunks = std::mem::take(&mut self.layout.chunks);
        let mut bonds = std::mem::take(&mut self.layout.bonds);
        let mut chunk_static = std::mem::take(&mut host.chunk_static);
        let mut bond_static = std::mem::take(&mut host.bond_static);
        let old_csr = std::mem::take(&mut host.csr);
        let mut islands: Vec<Island> = Vec::with_capacity(n_new);
        // Per parent: the next free chunk and bond of its ranges.
        let mut cursor: HashMap<usize, (u32, u32)> = parents.iter().map(|&ci| (ci, (old_islands[ci].range[0], old_islands[ci].range[2]))).collect();
        // Moved chunks and bonds (old -> new index), and the ranges to upload.
        let mut chunk_moves: Vec<(usize, usize)> = Vec::new();
        let mut bond_moves: Vec<(usize, usize)> = Vec::new();
        let mut dirty_chunks: Vec<(usize, usize)> = Vec::new();
        let mut dirty_bonds: Vec<(usize, usize)> = Vec::new();
        let mut new_state: HashMap<usize, [f32; 4]> = HashMap::new();
        let mut new_bond_dyn: HashMap<usize, BondDyn> = HashMap::new();
        for (j, cl) in m.clusters.iter().enumerate() {
            match old_index.get(&cl.id) {
                Some(&oi) if !parent_ids.contains(&cl.id) => {
                    let isl = old_islands[oi];
                    if oi != j {
                        // Moved to a new index: its chunks and bonds name it.
                        for k in isl.range[0] as usize..isl.range[1] as usize {
                            chunk_static[k].info[1] = j as u32;
                        }
                        for b in isl.range[2] as usize..isl.range[3] as usize {
                            bond_static[b].law.ids[3] = j as u32;
                        }
                        dirty_chunks.push((isl.range[0] as usize, isl.range[1] as usize));
                        dirty_bonds.push((isl.range[2] as usize, isl.range[3] as usize));
                    }
                    islands.push(isl);
                }
                _ => {
                    // A child: its chunks and bonds take the next part of its parent's.
                    let s = cl.structure;
                    let first = old_chunk_at.get(&(s, cl.chunks[0])).copied().ok_or("split child outside its parent")?;
                    let parent = parent_of_chunk(first).ok_or("split child without a parent")?;
                    let (c0, b0) = cursor[&parent];
                    let mut k = c0 as usize;
                    for &c in &cl.chunks {
                        let ok = old_chunk_at[&(s, c)];
                        let mut rec = old_chunk_rec[&ok];
                        rec.info[1] = j as u32;
                        chunk_static[k] = rec;
                        chunks[k] = (s, c);
                        let pr = old_islands[parent].range;
                        let st = &parent_state[&parent];
                        for q in 0..4 {
                            new_state.insert(4 * k + q, st[4 * (ok - pr[0] as usize) + q]);
                        }
                        chunk_moves.push((ok, k));
                        k += 1;
                    }
                    let new_index: HashMap<usize, usize> = chunk_moves.iter().filter(|(_, n)| (c0 as usize..k).contains(n)).map(|&(o, n)| (o, n)).collect();
                    let mut b = b0 as usize;
                    for &bi in &cl.bonds {
                        let ob = old_bond_at[&(s, bi)];
                        let mut rec = old_bond_rec[&ob];
                        let (oa, obb) = (rec.law.ids[1] as usize, rec.law.ids[2] as usize);
                        rec.law.ids = [rec.law.ids[0], new_index[&oa] as u32, new_index[&obb] as u32, j as u32];
                        bond_static[b] = rec;
                        bonds[b] = (s, bi);
                        let pr = old_islands[parent].range;
                        new_bond_dyn.insert(b, parent_bonds[&parent][ob - pr[2] as usize]);
                        bond_moves.push((ob, b));
                        b += 1;
                    }
                    let mut isl = stress_layout::island_record(m, j, c0, k as u32);
                    isl.range[2] = b0;
                    isl.range[3] = b as u32;
                    islands.push(isl);
                    cursor.insert(parent, (k as u32, b as u32));
                }
            }
        }
        if trace {
            stamps.push(("layout", t_split.elapsed().as_secs_f64() * 1e3));
        }
        // Each parent's bonds the children do not hold (those the split cut) keep their
        // slots, in no island; their records and zero loads stay there for the
        // incidence's padding.
        let mut dead: HashMap<usize, usize> = HashMap::new();
        for &ci in &parents {
            let r = old_islands[ci].range;
            let (_, b_end) = cursor[&ci];
            let held: std::collections::HashSet<usize> = bond_moves.iter().map(|&(o, _)| o).collect();
            let mut b = b_end as usize;
            for ob in r[2] as usize..r[3] as usize {
                if held.contains(&ob) {
                    continue;
                }
                let mut rec = old_bond_rec[&ob];
                rec.law.ids[3] = u32::MAX;
                bond_static[b] = rec;
                bonds[b] = old_bond_key[&ob];
                new_bond_dyn.insert(b, parent_bonds[&ci][ob - r[2] as usize]);
                dead.entry(ci).or_insert(b);
                b += 1;
            }
            if b != r[3] as usize {
                return Err(format!("split in place: parent {ci}'s bonds do not add up"));
            }
            dirty_chunks.push((r[0] as usize, r[1] as usize));
            dirty_bonds.push((r[2] as usize, r[3] as usize));
        }
        // The parents' chunks' incidence: each chunk keeps its slots, its entries now its
        // remaining bonds (in bond order), padded with a cut bond's (zero loads).
        let mut csr = old_csr.clone();
        for &ci in &parents {
            let r = old_islands[ci].range;
            // A cut bond of this split, or the padding an earlier one left in these chunks
            // (an entry for a bond outside the parent's range: zero loads either way).
            let pad = dead.get(&ci).map(|&b| 2 * b as u32).or_else(|| {
                (old_csr[r[0] as usize] as usize..old_csr[r[1] as usize] as usize).map(|e| old_csr[e]).find(|&e| !(r[2] as usize..r[3] as usize).contains(&((e >> 1) as usize)))
            });
            let mut incident: HashMap<usize, Vec<u32>> = HashMap::new();
            for b in r[2] as usize..cursor[&ci].1 as usize {
                let ids = bond_static[b].law.ids;
                incident.entry(ids[1] as usize).or_default().push(2 * b as u32);
                incident.entry(ids[2] as usize).or_default().push(2 * b as u32 + 1);
            }
            let degree = |old_k: usize| (old_csr[old_k + 1] - old_csr[old_k]) as usize;
            let new_to_old: HashMap<usize, usize> = chunk_moves.iter().filter(|(_, nk)| (r[0] as usize..r[1] as usize).contains(nk)).map(|&(o, nk)| (nk, o)).collect();
            let mut at = old_csr[r[0] as usize];
            for k in r[0] as usize..r[1] as usize {
                csr[k] = at;
                let d = degree(new_to_old[&k]);
                let list = incident.remove(&k).unwrap_or_default();
                if list.len() > d {
                    return Err(format!("split in place: chunk {k} gained bonds"));
                }
                for q in 0..d {
                    csr[at as usize + q] = match list.get(q) {
                        Some(&e) => e,
                        None => pad.ok_or("split in place: no cut bond to pad with")?,
                    };
                }
                at += d as u32;
            }
            if at != old_csr[r[1] as usize] {
                return Err("split in place: incidence does not add up".into());
            }
        }
        if trace {
            stamps.push(("incidence", t_split.elapsed().as_secs_f64() * 1e3));
        }
        // Uploads: the changed ranges, the islands (and the sentinel), the incidence.
        let cs_bytes = std::mem::size_of::<stress_layout::ChunkStatic>() as u64;
        let bs_bytes = std::mem::size_of::<stress_layout::BondStatic>() as u64;
        for &(a, b) in &dirty_chunks {
            if b > a {
                gpu.queue.write_buffer(&self.buffers.chunk_static, a as u64 * cs_bytes, bytemuck::cast_slice(&chunk_static[a..b]));
            }
        }
        for &(a, b) in &dirty_bonds {
            if b > a {
                gpu.queue.write_buffer(&self.buffers.bonds_static, a as u64 * bs_bytes, bytemuck::cast_slice(&bond_static[a..b]));
            }
        }
        let p = &mut self.buffers.params;
        for &ci in &parents {
            let r = old_islands[ci].range;
            let (c0, c1, b0, b1) = (r[0] as usize, r[1] as usize, r[2] as usize, r[3] as usize);
            // The parent's ranges, each in one write.
            let state: Vec<[f32; 4]> = (4 * c0..4 * c1).map(|k| new_state[&k]).collect();
            gpu.queue.write_buffer(&self.buffers.state, 4 * c0 as u64 * f4, bytemuck::cast_slice(&state));
            let dyns: Vec<BondDyn> = (b0..b1).map(|b| new_bond_dyn[&b]).collect();
            if !dyns.is_empty() {
                gpu.queue.write_buffer(&self.buffers.bond_dyn, b0 as u64 * bond_bytes, bytemuck::cast_slice(&dyns));
            }
            // The children's static-solve slots cleared (a rebuild's cleared scratch: the
            // compensated displacement parts with them), and the cut bonds' loads.
            gpu.queue.write_buffer(&self.buffers.scratch, (p.statics_base as u64 + STATICS_PER_CHUNK * c0 as u64) * f4, bytemuck::cast_slice(&vec![[0f32; 4]; STATICS_PER_CHUNK as usize * (c1 - c0)]));
            let cut = cursor[&ci].1 as usize;
            if b1 > cut {
                gpu.queue.write_buffer(&self.buffers.scratch, 3 * cut as u64 * f4, bytemuck::cast_slice(&vec![[0f32; 4]; 3 * (b1 - cut)]));
                // And their static-solve secant stiffness (world.slang `statics_bond_slot`):
                // no island's solve rewrites a cut bond's, and the incidence padding still
                // reaches it, so a stale one would stiffen the children with a bond that is
                // gone (and read a chunk another thread is writing).
                let slot = p.statics_base as u64 + STATICS_PER_CHUNK * p.chunk_count as u64 + 2 * cut as u64;
                gpu.queue.write_buffer(&self.buffers.scratch, slot * f4, bytemuck::cast_slice(&vec![[0f32; 4]; 2 * (b1 - cut)]));
            }
            // The incidence of its chunks: their offsets and their entries.
            gpu.queue.write_buffer(&self.buffers.index, 4 * c0 as u64, bytemuck::cast_slice(&csr[c0..c1]));
            let (e0, e1) = (old_csr[c0] as usize, old_csr[c1] as usize);
            if e1 > e0 {
                gpu.queue.write_buffer(&self.buffers.index, 4 * e0 as u64, bytemuck::cast_slice(&csr[e0..e1]));
            }
        }
        if std::env::var("STRESS_GPU_SPLIT_HASH").is_ok() {
            use std::hash::{Hash, Hasher};
            let hash = |bytes: &[u8]| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                bytes.hash(&mut h);
                h.finish()
            };
            let mut ks: Vec<(usize, [u32; 4])> = new_state.iter().map(|(k, v)| (*k, bytemuck::cast::<[f32; 4], [u32; 4]>(*v))).collect();
            ks.sort();
            let mut kb: Vec<(usize, Vec<u8>)> = new_bond_dyn.iter().map(|(k, v)| (*k, bytemuck::bytes_of(v).to_vec())).collect();
            kb.sort();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            ks.hash(&mut h);
            kb.hash(&mut h);
            eprintln!(
                "split hash: chunks {:x} bonds {:x} csr {:x} islands {:x} state {:x} parents {parents:?}",
                hash(bytemuck::cast_slice(&chunk_static)),
                hash(bytemuck::cast_slice(&bond_static)),
                hash(bytemuck::cast_slice(&csr)),
                hash(bytemuck::cast_slice(&islands)),
                h.finish()
            );
        }
        let mut with_sentinel = islands.clone();
        with_sentinel.push(Island::default());
        gpu.queue.write_buffer(&self.buffers.islands, 0, bytemuck::cast_slice(&with_sentinel));
        p.halt_index = n_new as u32;
        let params = *p;
        gpu.queue.write_buffer(&self.buffers.params_segment, 0, bytemuck::bytes_of(&params));
        gpu.queue.write_buffer(&self.buffers.params_contact, 0, bytemuck::bytes_of(&params));
        if trace {
            stamps.push(("uploads", t_split.elapsed().as_secs_f64() * 1e3));
        }
        // The layout follows.
        let host = self.layout.host.as_mut().expect("host tables");
        host.chunk_static = chunk_static;
        host.bond_static = bond_static;
        host.csr = csr;
        self.layout.chunks = chunks;
        self.layout.bonds = bonds;
        self.layout.islands = islands;
        self.layout_generation += 1;
        self.bonds_stale = true;
        self.state_current = false;
        self.layout_in_place = true;
        if trace {
            let mut last = 0.0;
            let parts: Vec<String> = stamps.iter().map(|(n, t)| {
                let d = t - last;
                last = *t;
                format!("{n} {d:.1}")
            }).collect();
            eprintln!("split in place stages (ms): {}", parts.join(", "));
        }
        if let Some(l) = self.loads.as_mut() {
            l.topology_version += 1;
        }
        Ok(true)
    }
}

/// What a plan made on the GPU gives a rebuild (`build`'s `resident`): the plan, and each
/// chunk's position (compensated pair) and orientation at its start, for the travel
/// check's records (instead of the mirror's, which a resident segment does not sync).
pub(crate) struct ResidentPlan {
    plan: Plan,
    starts: Vec<([f32; 4], [f32; 4], [f32; 4])>,
}

impl GpuSolver {
    /// `contacts::plan` over `horizon`, made on the GPU from its own state
    /// (`gpu_plan.rs`), as the host's `Plan`, with the chunks' starts.
    fn resident_plan(&mut self, gpu: &Gpu, horizon: f64) -> ResidentPlan {
        let out = self.gpu_plan(gpu, horizon);
        let m = &self.mirror;
        let chunks = &self.layout.chunks;
        let n = chunks.len();
        let scene = self.loads.as_ref().map(|l| &l.scene);
        let g = m.config.gravity.norm();
        let mut plan = Plan {
            contact_clusters: vec![false; m.clusters.len()],
            budget: out.budget.iter().map(|&b| b as f64).collect(),
            rotation_checked: out.rotation_checked.clone(),
            velocity: out.motion.iter().map(|mo| vec3(mo[1])).collect(),
            falls: out.motion.iter().map(|mo| mo[2][3] != 0.0).collect(),
            horizon,
            ..Default::default()
        };
        let cluster_of = |k: usize| m.chunks[chunks[k].0][chunks[k].1].cluster;
        // The box rotation of a chunk in a pair (cluster x hidden x chunk rotation).
        let box_rotation = |k: usize| {
            let q = out.motion[k][3];
            let q = Quat { x: q[0] as f64, y: q[1] as f64, z: q[2] as f64, w: q[3] as f64 };
            let (s, c) = chunks[k];
            q.to_mat3() * m.structures[s].chunks[c].rotation
        };
        let mu_of = |s: usize, c: usize| scene.map_or(0.0, |sc| sc.material(&m.structures[s].chunks[c].material).friction);
        for &(ka, kb) in &out.pairs {
            let (a, b) = (ka as usize, kb as usize);
            let ((sa, ca), (sb, cb)) = (chunks[a], chunks[b]);
            let (ma, mb) = (m.structures[sa].chunks[ca].mass, m.structures[sb].chunks[cb].mass);
            let mu = scene.and_then(|sc| sc.sim.contact_friction).unwrap_or(mu_of(sa, ca).min(mu_of(sb, cb)));
            plan.pairs.push(((sa, ca, sb, cb), ka, kb, ma * mb / (ma + mb), mu));
            plan.poses.insert(ka, (vec3(out.motion[a][0]), box_rotation(a)));
            plan.poses.insert(kb, (vec3(out.motion[b][0]), box_rotation(b)));
            plan.contact_clusters[cluster_of(a)] = true;
            plan.contact_clusters[cluster_of(b)] = true;
        }
        // Impactor and ground candidates in active order (by structure, chunk).
        let mut active: Vec<u32> = (0..n as u32).collect();
        active.sort_by_key(|&k| chunks[k as usize]);
        for (i, imp) in self.impactors.iter().enumerate() {
            if imp.driven {
                plan.impactor_plan.push(None);
                plan.impactor_chunks.push(Vec::new());
                continue;
            }
            let reach = match imp.shape {
                ImpactorShape::Sphere { radius } => radius * 3f64.sqrt(),
                ImpactorShape::Box { half_extents } => Vec3::from_array(half_extents).norm(),
            };
            let budget = 2.0 * (imp.velocity.norm() + imp.angular_velocity.norm() * reach) * horizon + g * horizon * horizon + 1e-3;
            plan.impactor_plan.push(Some(crate::contacts::ImpactorPlan { position: imp.pose.position, velocity: Vec3::ZERO, rotation: imp.pose.rotation, budget }));
            let list: Vec<u32> = active.iter().copied().filter(|&k| out.impactor.get(i).is_some_and(|f| f[k as usize])).collect();
            for &k in &list {
                plan.contact_clusters[cluster_of(k as usize)] = true;
            }
            plan.impactor_chunks.push(list);
        }
        plan.ground = active.iter().copied().filter(|&k| out.ground[k as usize]).collect();
        for &k in &plan.ground {
            plan.contact_clusters[cluster_of(k as usize)] = true;
        }
        let starts = (0..n).map(|k| (out.start[k].0, out.start[k].1, out.motion[k][3])).collect();
        ResidentPlan { plan, starts }
    }
}

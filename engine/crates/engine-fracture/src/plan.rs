//! The per-tick host plan of the fracture lab, as data both hosts replay.
//!
//! Rubble's shared Slang AVBD kernels (persistent hot/cold contacts, convex
//! hull narrowphase, hierarchical broadphase) are bound to the 31 storage
//! buffers of the WGSL layout (`BIND(slot)` numbering). `Planner` owns the
//! sizes, the host-uploaded words and the host state machine (awake lists,
//! persistent broadphase, tagged warm hash, colour estimates, bond phases), and
//! emits each tick as a flat command list ([`Command`]) that a host executes in
//! order: WebGPU (`fracture/simulation.mjs`) and Metal
//! (`engine-viewer/src/fracture/sim.rs`). The schedule mirrors
//! engine-webgpu/src/gpu.rs in its persistent-contact mode.

use crate::scene::{Body, Collider, Joint, Scene, SceneParams, SHAPE_CONVEX};
use anyhow::{bail, ensure, Result};
use bytemuck::bytes_of;
use engine_schema::{BodyState, BondSpec, Collider as GpuCollider, Joint as GpuJoint, JointDesc};

pub const BUFFERS: usize = 31;
/// Host uniform buffers (params, pass table) and the indirect-args buffer.
pub const PARAMS_BUFFER: u32 = 6;
pub const COUNTERS_BUFFER: u32 = 7;
pub const HULL_BUFFER: u32 = 18;
pub const SCRATCH_BUFFER: u32 = 21;
pub const PASS_BUFFER: u32 = 22;
pub const ARGS_BUFFER: u32 = 23;
pub const JOINT_BUFFER: u32 = 5;
pub const BODY_BUFFER: u32 = 0;
pub const COLLIDER_BUFFER: u32 = 2;
pub const MATERIAL_BUFFER: u32 = 3;
/// `Command::Copy` source standing for the per-tick host upload (`host_upload`).
pub const HOST_UPLOAD: u32 = 255;
/// Pass-uniform stride (dynamic offset alignment).
pub const PASS_ALIGN: u32 = 256;
pub const PASS_SLOTS: u32 = 160;
/// First pass slot of the post-stabilization primal pass (value 256 | colour).
pub const STAB_PASS: u32 = 96;
pub const COUNTER_BYTES: u32 = 80;
pub const BODY_BYTES: usize = 112;
pub const COLLIDER_BYTES: usize = 64;
pub const JOINT_BYTES: usize = std::mem::size_of::<GpuJoint>();
const COLOR_COUNT_WORDS: usize = 128;
/// Host upload layout: scratch words, params block, counters, colour counts.
pub const UPLOAD_SCRATCH: u32 = 0;
pub const UPLOAD_PARAMS: u32 = 256;
pub const UPLOAD_COUNTERS: u32 = 512;
pub const UPLOAD_COLORS: u32 = 608;
pub const UPLOAD_BYTES: usize = 608 + COLOR_COUNT_WORDS * 4;
pub const PARAMS_BYTES: u32 = 208;
/// Threads per workgroup of every scheduled kernel.
pub const WORKGROUP: u32 = 64;

const MAX_WARM_HASH: u32 = 1 << 21;
const BOND_WAKE_ROUNDS: u32 = 8;
const SPARSE_AWAKE: u32 = 256;
/// Colour slots dispatched past the previous step's colour count.
const COLOR_HEADROOM: u32 = 4;
/// Colouring rounds dispatched past the previous step's need.
const COLOR_ROUND_HEADROOM: u32 = 2;

/// Scheduled kernels: (dispatch name, exported entry). Persistent-storage
/// entries replace their core names; the dispatch name labels profiles.
pub const KERNELS: &[(&str, &str)] = &[
    ("build_aabbs", "build_aabbs"),
    ("broadphase_clear", "broadphase_clear"),
    ("broadphase_insert", "broadphase_insert"),
    ("broadphase_query", "broadphase_query"),
    ("broadphase_query_awake", "broadphase_query_awake"),
    ("write_finer_args", "write_finer_args"),
    ("write_moved_args", "write_moved_args"),
    ("rest_clear", "rest_clear"),
    ("rest_insert", "rest_insert"),
    ("rest_publish", "rest_publish"),
    ("broadphase_insert_awake", "broadphase_insert_awake"),
    ("broadphase_clear_awake", "broadphase_clear_awake"),
    ("list_awake_colliders", "list_awake_colliders"),
    ("write_awake_collider_args", "write_awake_collider_args"),
    ("broadphase_query_finer", "broadphase_query_finer"),
    ("write_work_args", "write_work_args"),
    ("update_refs", "update_refs"),
    ("contact_hash_clear", "contact_hash_clear"),
    ("contact_hash_insert", "contact_hash_insert_persistent"),
    ("narrowphase", "narrowphase_persistent"),
    ("convex_classify", "convex_classify_persistent"),
    ("convex_bucket", "convex_bucket"),
    ("convex_write_args", "convex_write_args"),
    ("convex_sat_small", "convex_sat_small"),
    ("convex_sat_large", "convex_sat_large"),
    ("convex_manifold", "convex_manifold_persistent"),
    ("reuse_all_manifolds", "reuse_all_manifolds_persistent"),
    ("persist_contacts", "persist_contacts_persistent"),
    ("build_awake_list", "build_awake_list"),
    ("csr_clear", "csr_clear"),
    ("csr_degree_contacts", "csr_degree_contacts"),
    ("csr_prefix", "csr_prefix"),
    ("csr_prefix_blocks", "csr_prefix_blocks"),
    ("csr_prefix_block_sums", "csr_prefix_block_sums"),
    ("csr_prefix_add", "csr_prefix_add"),
    ("csr_scatter_contacts", "csr_scatter_contacts"),
    ("csr_sort", "csr_sort"),
    ("init_colors", "init_colors"),
    ("color_bodies", "color_bodies"),
    ("color_bodies_local", "color_bodies_local"),
    ("color_bodies_tail", "color_bodies_tail"),
    ("count_color_tiles", "count_color_tiles"),
    ("prefix_color_tiles_parallel", "prefix_color_tiles_parallel"),
    ("scatter_color_tiles", "scatter_color_tiles"),
    ("record_color_degrees", "record_color_degrees"),
    ("write_color_args", "write_color_args"),
    ("predict_bodies", "predict_bodies"),
    ("pack_solver_contacts", "pack_solver_pairs_persistent"),
    ("solve_primal_color", "solve_primal_color_persistent"),
    ("solve_primal_sparse", "solve_primal_sparse_persistent"),
    ("solve_dual", "solve_dual_persistent"),
    ("record_stab_velocity", "record_stab_velocity"),
    ("extract_velocity_sleep", "extract_velocity_sleep"),
    ("carry_rest_pairs", "carry_rest_pairs"),
    // Continuous collision for projectiles (engine/docs/ccd.md; scheduled only
    // once a CCD shot is fired, `Planner::set_projectile_ccd`).
    ("ccd_register", "ccd_register"),
    ("broadphase_fast", "broadphase_fast"),
    ("ccd_toi", "ccd_toi"),
    ("ccd_pick", "ccd_pick"),
    ("ccd_mark", "ccd_mark"),
    ("ccd_emit", "ccd_emit_persistent"),
    ("ccd_respond", "ccd_respond"),
    ("ccd_respond_apply", "ccd_respond_apply"),
    ("ccd_trust", "ccd_trust"),
    ("ccd_validate", "ccd_validate"),
    ("ccd_clamp", "ccd_clamp"),
    ("joint_hash_clear", "joint_hash_clear"),
    ("joint_hash_insert", "joint_hash_insert"),
    ("bond_wake", "bond_wake"),
    ("warmstart_joints", "warmstart_joints"),
    ("csr_degree_joints", "csr_degree_joints"),
    ("csr_scatter_joints", "csr_scatter_joints"),
    ("write_sleep_args", "write_sleep_args"),
    ("sleep_island_gate", "sleep_island_gate"),
    ("sleep_island_commit", "sleep_island_commit"),
];

/// Slang source (relative to `engine/slang/kernels/`) that defines `entry`
/// (as engine-webgpu's build script compiles them).
pub fn kernel_source(entry: &str) -> &'static str {
    if entry.starts_with("convex_") {
        return if entry.ends_with("_persistent") {
            "convex_persistent.slang"
        } else {
            "convex_step.slang"
        };
    }
    if entry.ends_with("_copy_persistent") {
        "reuse_copy.slang"
    } else if entry.ends_with("_persistent") {
        "persistent_solve.slang"
    } else if entry.ends_with("_compact")
        || entry == "pack_solver_contacts"
        || entry == "clear_solver_contacts"
    {
        "compact_solve.slang"
    } else {
        "world_step.slang"
    }
}

fn kernel(name: &str) -> u32 {
    KERNELS
        .iter()
        .position(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("unscheduled kernel {name}")) as u32
}

/// Profiling and slicing boundaries: a phase ends where its marker appears.
/// `slice` phases end a command buffer when a host spreads a step over frames
/// (each slice <= ~6-8 ms at the 49-house peak; same dispatches, same order).
pub const PHASES: &[(&str, bool)] = &[
    ("bounds", false),
    ("broadphaseInsert", false),
    ("broadphase", true),
    ("narrowphase", false),
    ("convexClassify", true),
    ("convexSatSmall", true),
    ("convexSat", true),
    ("convexManifold", true),
    ("reuseManifolds", false),
    ("persistence", false),
    ("graph", false),
    ("coloring", false),
    ("topology", true),
    ("solveHalf", true),
    ("solve", true),
    ("sleep", false),
];

fn phase_id(name: &str) -> u32 {
    PHASES.iter().position(|(n, _)| *n == name).unwrap() as u32
}

/// Command opcodes (first word); `Planner` output is a flat `u32` list.
pub const OP_DISPATCH: u32 = 0;
pub const OP_INDIRECT: u32 = 1;
pub const OP_COPY: u32 = 2;
pub const OP_PHASE: u32 = 3;

/// One host operation, decoded from the flat list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// `dispatchWorkgroups(x, y, 1)` of `kernel` with pass-uniform slot `pass`.
    Dispatch {
        kernel: u32,
        x: u32,
        y: u32,
        pass: u32,
    },
    /// Indirect dispatch: workgroup counts at `slot * 12` bytes of buffer 23.
    Indirect { kernel: u32, slot: u32, pass: u32 },
    /// Buffer copy; `src == HOST_UPLOAD` reads this tick's host upload.
    Copy {
        src: u32,
        src_offset: u32,
        dst: u32,
        dst_offset: u32,
        size: u32,
    },
    /// End of phase `PHASES[id]`.
    Phase { id: u32 },
}

impl Command {
    pub fn decode(words: &[u32]) -> Vec<Command> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let w = &words[i..];
            let (c, n) = match w[0] {
                OP_DISPATCH => (
                    Command::Dispatch {
                        kernel: w[1],
                        x: w[2],
                        y: w[3],
                        pass: w[4],
                    },
                    5,
                ),
                OP_INDIRECT => (
                    Command::Indirect {
                        kernel: w[1],
                        slot: w[2],
                        pass: w[3],
                    },
                    4,
                ),
                OP_COPY => (
                    Command::Copy {
                        src: w[1],
                        src_offset: w[2],
                        dst: w[3],
                        dst_offset: w[4],
                        size: w[5],
                    },
                    6,
                ),
                _ => (Command::Phase { id: w[1] }, 2),
            };
            out.push(c);
            i += n;
        }
        out
    }
}

/// GPU counters (20 words of buffer 7) the host reads back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Counters {
    pub contact: u32,
    pub candidate: u32,
    pub awake: u32,
    pub color: u32,
    pub manifold: u32,
    pub overflow: u32,
    pub convex_small: u32,
    pub convex_large: u32,
    pub color_rounds: u32,
}

impl Counters {
    pub fn from_words(c: &[u32]) -> Self {
        Self {
            contact: c[2],
            candidate: c[4],
            awake: c[5],
            color: c[6],
            manifold: c[12],
            overflow: c[7],
            convex_small: c[11],
            convex_large: c[16],
            color_rounds: c[17],
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Caps {
    pub max_bodies: u32,
    pub max_colliders: u32,
    pub max_contacts: u32,
    pub max_candidates: u32,
    pub max_joints: u32,
    pub max_colors: u32,
    pub max_commands: u32,
    pub max_hull: u32,
}

fn next_pow2(n: u64) -> u32 {
    n.max(1).next_power_of_two() as u32
}

/// Packs one body record (initial pose = pose, so frame-0 x- and warm start match).
pub fn body_record(b: &Body) -> BodyState {
    let p = b.p.map(|x| x as f32);
    let q = b.q.map(|x| x as f32);
    BodyState {
        position: p,
        inv_mass: b.inv_mass as f32,
        orientation: q,
        inv_inertia: b.inv_i.map(|x| x as f32),
        flags: b.flags,
        linear_vel: b.v.map(|x| x as f32),
        sleep_timer: 0.0,
        angular_vel: b.w.map(|x| x as f32),
        generation: 0,
        initial_pos: p,
        initial_pad: 0.0,
        initial_rot: q,
    }
}

pub fn collider_record(c: &Collider) -> GpuCollider {
    let (generation, material) = c.filter_words();
    GpuCollider {
        body: c.body,
        shape: c.shape,
        material,
        generation,
        local_pos: c.local_pos.map(|x| x as f32),
        density: c.density as f32,
        local_rot: c.local_rot.map(|x| x as f32),
        params: c.params.map(|x| x as f32),
    }
}

/// Bond joint: the D6 bond preset (`JointDesc::bond`): a weld in the bond
/// frame (normal, u, v in A's frame) about the rest pose `conj(qA)·qB`, with
/// fixed per-row penalties and Shardwright's capacities.
pub fn joint_record(j: &Joint) -> GpuJoint {
    let v3 = |v: [f64; 3]| glam::Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32);
    let rest = glam::Quat::from_xyzw(
        j.rest[0] as f32,
        j.rest[1] as f32,
        j.rest[2] as f32,
        j.rest[3] as f32,
    );
    // Scene penalties: normal, u, v, about u, about v, about normal; the D6
    // bond rows: normal, u, v, about normal, about u, about v.
    let k = [j.k[0], j.k[1], j.k[2], j.k[5], j.k[3], j.k[4]].map(|x| x as f32);
    JointDesc::bond(
        j.body_a,
        j.body_b,
        v3(j.anchor_a),
        v3(j.anchor_b),
        v3(j.normal),
        v3(j.frame_u),
        rest.normalize(),
        BondSpec {
            k,
            strength: j.strength.map(|x| x as f32),
        },
    )
    .to_gpu()
}

/// Host plan of one simulation (see module docs).
pub struct Planner {
    pub cap: Caps,
    pub params: SceneParams,
    sizes: Vec<u64>,
    body_count: u32,
    collider_count: u32,
    joint_count: u32,
    joint_hash_slots: u32,
    csr_base: u32,
    grid_hash_words: u32,
    warm_hash_words: u32,
    warm_tagged: bool,
    warm_tag_period: u32,
    grid_levels: u32,
    color_round_limit: u32,
    listed_passes: bool,
    pub incremental: bool,
    has_convex: bool,
    has_filters: bool,
    /// Shots carry `BODY_CCD` (time-of-impact contacts, fast broadphase list).
    projectile_ccd: bool,
    /// A CCD shot was fired since the last reset (its flag stays on the body).
    ccd_fired: bool,
    sleep_rounds: u32,
    color_headroom: Option<u32>,
    color_round_headroom: Option<u32>,
    moved_only_override: Option<bool>,
    pub awake_queries: bool,
    persistent_rest: bool,
    exact_sleep_gate: bool,
    bounds_sortable: bool,
    colliders: Vec<Collider>,
    body_bytes: Vec<u8>,
    collider_bytes: Vec<u8>,
    joint_bytes: Vec<u8>,
    // Projectiles.
    projectile_start: u32,
    projectile_count: u32,
    projectile_mass: f64,
    projectile_radius: f64,
    projectile_speed: Option<f64>,
    /// The shots' speed as a multiple of the scene's (`set_projectile_speed_scale`).
    projectile_speed_scale: f64,
    projectile_arc: bool,
    extent: f64,
    center: [f64; 3],
    // Host state.
    pub last: Counters,
    pub steps: u32,
    parity: u32,
    pub shots: u32,
    pub initialized: bool,
    bounds_stale: bool,
    moved_only_step: bool,
    rest_step: bool,
    rest_valid: bool,
    warm_clear: bool,
    warm_cycle: i64,
    manifolds_valid: bool,
    counted_steps: u32,
    submitted_steps: u32,
    wake_serial: u32,
    upload: Vec<u8>,
    commands: Vec<u32>,
}

/// Device limits a host checks buffer sizes against.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_binding: u64,
    pub max_buffer: u64,
}

impl Planner {
    pub fn new(scene: &Scene, limits: Limits) -> Result<Self> {
        let budget = scene.capacity;
        let nb = scene.bodies.len() as u64;
        let nc = scene.colliders.len() as u64;
        let cap = Caps {
            max_bodies: next_pow2(nb.max(64)),
            max_colliders: next_pow2(nc.max(64)),
            max_contacts: next_pow2((nc * budget.contacts_per_collider as u64).max(4096)),
            max_candidates: next_pow2((nc * budget.candidates_per_collider as u64).max(4096)),
            max_joints: (scene.joints.len() as u32).max(1),
            max_colors: budget.max_colors.unwrap_or(32),
            max_commands: 1,
            max_hull: (scene.hull_data.len() as u32).max(64),
        };
        let joint_count = scene.joints.len() as u32;
        // Joint-pair hash (world_step jointHashSize): >= 65536 slots and 2x the joints.
        let joint_hash_slots = next_pow2((2 * joint_count as u64).max(65536));
        let constraint_id_words =
            next_pow2((2 * (cap.max_contacts as u64 + cap.max_joints as u64)).max(32768));
        let csr_base = constraint_id_words.max(131072);
        // Broadphase grid heads: twice the colliders, within the pre-CSR id region.
        let grid_hash_words = next_pow2((2 * nc).max(131072)).min(csr_base);
        // Warm (previous-contact) hash slots; host field and clear dispatch must agree.
        let warm_hash_words = next_pow2((cap.max_contacts as u64 * 4).max(1024))
            .min(csr_base)
            .min(MAX_WARM_HASH);
        // Tick-tagged warm hash (world_step warmTagged): slot words carry a per-step
        // tag above log2(maxContacts) index bits, so the table is cleared only when
        // the tags wrap (every warm_tag_period steps) instead of every step.
        let warm_value_bits = cap.max_contacts.trailing_zeros();
        let warm_tagged = budget.tagged_warm_hash && warm_value_bits <= 29;
        let warm_tag_period = if warm_value_bits == 0 {
            u32::MAX
        } else {
            ((1u64 << (32 - warm_value_bits)) - 2) as u32
        };
        let mut params = scene.params;
        let (grid_base, grid_levels) = crate::scene::grid_config(&scene.colliders);
        params.grid_cell_size = grid_base as f32;
        // budget.deterministic (opt-in): run-to-run identical results. Adds the exact
        // sleep gate and drops the SAT-cache early-out (params flag 8): the cache is
        // direct-mapped, pairs that share a slot overwrite each other in scheduling
        // order, and on rare near pairs its proof of separation disagrees with the
        // full SAT, which can miss that axis (near-parallel edges are skipped) and
        // emit a fallback contact. Both change results vs the default.
        if budget.deterministic {
            params.flags |= 8;
        }
        let bounds_sortable = scene.colliders.windows(2).all(|w| w[1].body >= w[0].body)
            && scene.bodies.iter().all(|b| b.flags & 2 == 0);
        let mut p = Self {
            cap,
            params,
            sizes: vec![],
            body_count: nb as u32,
            collider_count: nc as u32,
            joint_count,
            joint_hash_slots,
            csr_base,
            grid_hash_words,
            warm_hash_words,
            warm_tagged,
            warm_tag_period,
            grid_levels,
            // Incremental colouring recolours only changed neighbourhoods but lets
            // the palette drift (more solve passes); fresh colouring is the default.
            // 64: a fast ball through a dense pile (vibe-city reinforcedTower,
            // 6,308 0.42 m cubes, at 576 m/s) needs more than 32 rounds; rounds
            // past last step's need plus headroom run only when a body is blocked.
            color_round_limit: budget.color_rounds.unwrap_or(64).min(64),
            // Per-body passes (predict, velocity/sleep, sleep commit) over the awake list.
            listed_passes: budget.awake_list_passes,
            incremental: budget.incremental_coloring,
            has_convex: scene.colliders.iter().any(|c| c.shape == SHAPE_CONVEX),
            has_filters: scene.colliders.iter().any(Collider::has_filter),
            projectile_ccd: false,
            ccd_fired: false,
            sleep_rounds: params.sleep_rounds.min(16),
            color_headroom: budget.color_headroom.unwrap_or(Some(COLOR_HEADROOM)),
            color_round_headroom: budget
                .color_round_headroom
                .unwrap_or(Some(COLOR_ROUND_HEADROOM)),
            moved_only_override: budget.moved_only_bounds,
            // Awake-only broadphase queries on moved-only steps (identical pairs).
            awake_queries: budget.awake_queries,
            // Resting colliders in a persistent table, so idle steps insert only movers.
            persistent_rest: budget.persistent_rest,
            // Opt-in: the island sleep gate moves a revocation exactly one hop per
            // round (deterministic); the default in-place gate can move it further
            // within a round depending on GPU scheduling. Changes which bodies sleep.
            exact_sleep_gate: budget.deterministic_sleep_gate || budget.deterministic,
            bounds_sortable,
            colliders: scene.colliders.clone(),
            body_bytes: vec![],
            collider_bytes: vec![],
            joint_bytes: vec![],
            projectile_start: scene.projectile_start,
            projectile_count: scene.projectile_count,
            projectile_mass: scene.projectile_mass,
            projectile_radius: scene.projectile_radius,
            projectile_speed: scene.projectile_speed,
            projectile_speed_scale: 1.0,
            projectile_arc: scene.projectile_arc,
            extent: scene.extent,
            center: scene.center,
            last: Counters::default(),
            steps: 0,
            parity: 0,
            shots: 0,
            initialized: false,
            bounds_stale: true,
            moved_only_step: false,
            rest_step: false,
            rest_valid: false,
            warm_clear: true,
            warm_cycle: -1,
            manifolds_valid: false,
            counted_steps: 0,
            submitted_steps: 0,
            wake_serial: 0,
            upload: vec![0; UPLOAD_BYTES],
            commands: Vec::new(),
        };
        p.sizes = p.buffer_sizes();
        for (i, &size) in p.sizes.iter().enumerate() {
            ensure!(
                size <= limits.max_binding && size <= limits.max_buffer,
                "GPU buffer {i} exceeds this adapter's limits"
            );
        }
        p.body_bytes = vec![0; p.sizes[0] as usize];
        for (i, b) in scene.bodies.iter().enumerate() {
            let r = body_record(b);
            for at in [i, cap.max_bodies as usize + i] {
                p.body_bytes[at * BODY_BYTES..(at + 1) * BODY_BYTES].copy_from_slice(bytes_of(&r));
            }
        }
        p.collider_bytes = vec![0; p.sizes[2] as usize];
        for (i, c) in scene.colliders.iter().enumerate() {
            p.collider_bytes[i * COLLIDER_BYTES..(i + 1) * COLLIDER_BYTES]
                .copy_from_slice(bytes_of(&collider_record(c)));
        }
        p.joint_bytes = scene
            .joints
            .iter()
            .flat_map(|j| bytes_of(&joint_record(j)).to_vec())
            .collect();
        if p.joint_bytes.is_empty() {
            p.joint_bytes = vec![0; JOINT_BYTES];
        }
        p.reset();
        Ok(p)
    }

    fn buffer_sizes(&self) -> Vec<u64> {
        let cap = &self.cap;
        let b = |n: u32, stride: u64| n as u64 * stride;
        let mut s = vec![256u64; BUFFERS];
        s[0] = b(cap.max_bodies * 2, 112);
        s[1] = b(cap.max_bodies, 112);
        s[2] = b(cap.max_colliders, 64);
        s[3] = b(64, 16);
        s[4] = b(cap.max_contacts * 2, 32); // persistent ContactMeta (cold)
        s[5] = b(cap.max_joints, JOINT_BYTES as u64);
        s[8] = b(cap.max_commands, 64);
        s[9] = b(cap.max_bodies + cap.max_colliders, 32);
        s[10] = b(cap.max_candidates, 16);
        s[11] = b(cap.max_bodies, 4);
        s[12] = b(cap.max_bodies, 4);
        s[13] = b(cap.max_bodies + 1, 4);
        // CSR index, the tagged warm hash, then the persistent resting table (world_step
        // restBase): 32 header words, resting + awake heads, one resting link per
        // collider, then the awake-collider list (one word per collider).
        s[14] = (self.csr_base as u64
            + cap.max_bodies as u64
            + 1
            + self.warm_hash_words as u64
            + 32
            + 2 * self.grid_hash_words as u64
            + 2 * cap.max_colliders as u64)
            * 4;
        // Colour lists + history rows; broadphase links reuse it per collider.
        s[15] = b(
            cap.max_colliders.max(cap.max_bodies * (cap.max_colors + 2)),
            4,
        );
        s[16] = b(cap.max_colors * cap.max_bodies.div_ceil(64).max(1), 4);
        s[17] = b(128, 4); // colour counts 0..63 + max degrees 64..127
        s[18] = b(cap.max_hull, 16);
        s[19] = self.joint_hash_slots as u64 * 3 * 4; // joint hash: index + body pair per slot
        s[20] = b(cap.max_bodies, 4);
        // Scratch: host-reset words 0..63, uniform-grid large list 64..79, then
        // the GPU-owned lists past them (large statics, CCD slots, fast list,
        // impulse records; world_step.slang CCD_BASE..).
        s[21] = engine_schema::SCRATCH_WORDS as u64 * 4;
        s[22] = (PASS_SLOTS * PASS_ALIGN) as u64;
        // Colours, work 64..73, convex 74..76, finer scan 77, moved 78, rest rebuild 79..80, awake colliders 81.
        s[23] = b(cap.max_colors.max(82) * 3, 4);
        s[24] = b(cap.max_colliders, 80);
        s[25] = b(cap.max_candidates, 32);
        s[26] = b(cap.max_bodies, 32);
        s[27] = b(cap.max_contacts * 2, 64); // persistent SolverPoint (hot)
        s[28] = b(cap.max_candidates, 32); // solver pairs
        s[29] = b(cap.max_candidates, 32); // convex SAT work list
        s[30] = next_pow2((cap.max_candidates as u64).max(1024)) as u64 * 8; // SAT separating-feature cache
        s
    }

    /// Byte size of every buffer (`AVBD 0..30`).
    pub fn buffer_sizes_bytes(&self) -> &[u64] {
        &self.sizes
    }

    /// Buffers bound as uniforms (the rest are storage); 23 is also indirect.
    pub fn is_uniform(slot: u32) -> bool {
        slot == PARAMS_BUFFER || slot == PASS_BUFFER
    }

    /// Initial contents: bodies (current and previous halves), colliders, joints.
    pub fn body_bytes(&self) -> &[u8] {
        &self.body_bytes
    }
    pub fn collider_bytes(&self) -> &[u8] {
        &self.collider_bytes
    }
    pub fn joint_bytes(&self) -> &[u8] {
        &self.joint_bytes
    }
    pub fn body_count(&self) -> u32 {
        self.body_count
    }
    pub fn collider_count(&self) -> u32 {
        self.collider_count
    }
    pub fn joint_count(&self) -> u32 {
        self.joint_count
    }

    /// Pass uniform table (buffer 22): slot i holds i; slots 48..67 the block-scan
    /// constants; 68..83 the colouring rounds 48..63 (`round_slot`); 96..159
    /// `256 | colour` for the post-stabilization primal pass.
    pub fn pass_table() -> Vec<u8> {
        let mut out = vec![0u8; (PASS_SLOTS * PASS_ALIGN) as usize];
        for i in 0..PASS_SLOTS {
            let v = if (48..68).contains(&i) {
                let k = i - 48;
                (1u32 << k) | ((k & 1) << 31)
            } else if (68..84).contains(&i) {
                i - 20
            } else if i >= STAB_PASS {
                256 | (i - STAB_PASS)
            } else if i >= 84 {
                0
            } else {
                i
            };
            let at = (i * PASS_ALIGN) as usize;
            out[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Material table (buffer 3): 64 slots of friction, restitution, offset.
    pub fn material_table(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 * 16);
        for _ in 0..64 {
            for x in [self.params.friction, 0.0, 0.01, 0.0] {
                out.extend(x.to_le_bytes());
            }
        }
        out
    }

    /// Buffers `reset` clears (all but bodies, colliders, materials, hulls, passes).
    pub fn cleared_on_reset(slot: u32) -> bool {
        ![0, 2, 3, 18, 22].contains(&slot)
    }

    /// Restore the step state (hosts clear buffers and re-upload the initial contents).
    pub fn reset(&mut self) {
        self.bounds_stale = true;
        self.manifolds_valid = false;
        self.steps = 0;
        self.counted_steps = 0;
        self.submitted_steps = 0;
        self.warm_cycle = -1;
        self.parity = 0;
        self.shots = 0;
        self.ccd_fired = false;
        self.initialized = false;
        self.last = Counters {
            awake: self.body_count.saturating_sub(1),
            color: 16,
            ..Counters::default()
        };
    }

    /// This tick's host upload (`UPLOAD_BYTES`), valid until the next encode.
    pub fn host_upload(&self) -> &[u8] {
        &self.upload
    }

    /// The commands of the last encode, as flat words.
    pub fn commands(&self) -> &[u32] {
        &self.commands
    }

    fn scratch_words(&mut self) -> [u32; 64] {
        let cap = self.cap;
        let mut s = [0u32; 64];
        s[1] = cap.max_bodies;
        s[2] = cap.max_candidates;
        s[3] = cap.max_contacts;
        s[4] = cap.max_colors.min(64);
        s[7] = if self.steps == 0 {
            0
        } else {
            self.last.contact.min(cap.max_contacts)
        };
        s[8] = if self.parity == 0 {
            cap.max_contacts
        } else {
            0
        };
        s[9] = self.warm_hash_words;
        s[10] = cap.max_candidates;
        s[11] = if self.parity == 0 {
            0
        } else {
            cap.max_contacts
        };
        let mut log = 0;
        while (1u32 << log) < self.body_count.max(1) && log < 20 {
            log += 1;
        }
        s[12] = if log == 0 { 0 } else { (log - 1) & 1 };
        s[16] = self.grid_hash_words;
        s[17] = if self.initialized {
            self.steps.max(1)
        } else {
            0
        };
        // Incremental colouring (as the Rust host): colours persist in a history row
        // and only bodies whose neighbourhood changed recolour (work slot 69); small
        // repair lists are coloured by one workgroup (local repair).
        s[26] = self.incremental as u32;
        s[27] = self.incremental as u32;
        s[18] = self.grid_levels; // hierarchical broadphase levels
        s[29] = 1; // degree-aware primal lanes (buffer 17 holds 128 words)
        s[30] = self.has_filters as u32; // collision filters present
        // Bit 2: CCD bodies; bit 4: broadphase_fast scheduled (rigid_plan's bits).
        s[30] |= (self.ccd_step() as u32) << 2 | (self.ccd_step() as u32) << 4;
                                         // Moved-only bounds: rebuild AABBs only for last step's awake bodies
                                         // (build_aabbs). Needs colliders sorted by body, no kinematic bodies, and no
                                         // host-side pose/collider write since the last step.
        s[19] = self.moved_only_bounds() as u32;
        // Persistent resting table on moved-only steps with awake queries: bit 0 on,
        // bit 1 rebuild now (it is rebuilt on the GPU after sleep events).
        let rest = s[19] != 0 && self.awake_queries && self.persistent_rest;
        s[25] = (if rest {
            if self.rest_valid {
                1
            } else {
                3
            }
        } else {
            0
        }) | if self.warm_tagged { 4 } else { 0 }
            | if self.exact_sleep_gate { 8 } else { 0 };
        // Tagged warm hash: clear once per tag cycle (HOST_S(17) is the step).
        let cycle = (s[17] / self.warm_tag_period) as i64;
        self.warm_clear = !self.warm_tagged || cycle != self.warm_cycle;
        self.warm_cycle = cycle;
        s[24] = self.csr_base;
        s[28] = self.carry_rest() as u32;
        s
    }

    /// A CCD shot is in the world: the step registers CCD bodies, walks the
    /// fast list and places time-of-impact contacts.
    fn ccd_step(&self) -> bool {
        self.ccd_fired
    }

    /// Resting-pair carry (FLAG_CARRY_REST): resting pairs keep their contacts
    /// and warm-start lambda through sleep from last step's manifolds.
    fn carry_rest(&self) -> bool {
        self.params.flags & crate::scene::FLAG_CARRY_REST != 0
    }

    /// With most bodies awake the full pass is faster (no per-body search, even
    /// load); both paths give identical bounds, so pick by last step's count.
    fn moved_only_bounds(&self) -> bool {
        self.bounds_sortable
            && !self.bounds_stale
            && self.initialized
            && self.moved_only_override != Some(false)
            && (self.moved_only_override == Some(true) || self.last.awake * 4 < self.body_count)
    }

    fn params_words(&self, scratch: &[u32; 64]) -> [u32; 52] {
        let p = &self.params;
        let f = f32::to_bits;
        let mut w = [0u32; 52];
        w[..20].copy_from_slice(&[
            f(p.dt),
            f(p.gravity_y),
            p.iterations,
            p.flags,
            f(p.alpha),
            f(p.gamma),
            f(p.beta_lin),
            f(p.beta_ang),
            f(p.k_start),
            f(p.k_max_n),
            f(p.k_max_t),
            f(p.contact_offset),
            f(p.sleep_lin),
            f(p.sleep_ang),
            f(p.sleep_time),
            f(p.residual_exit),
            f(p.trust_lin),
            f(p.trust_ang),
            f(p.pivot_floor),
            f(p.grid_cell_size),
        ]);
        w[20..].copy_from_slice(&scratch[..32]);
        w
    }

    fn copy(&mut self, src: u32, src_offset: u32, dst: u32, dst_offset: u32, size: u32) {
        self.commands
            .extend([OP_COPY, src, src_offset, dst, dst_offset, size]);
    }
    fn phase(&mut self, name: &str) {
        self.commands.extend([OP_PHASE, phase_id(name)]);
    }
    /// `dispatchWorkgroups(groups(threads))`: ceil(threads/64) groups in rows of 32768.
    fn disp(&mut self, name: &str, threads: u32, pass: u32) {
        let n = threads.max(1).div_ceil(WORKGROUP);
        self.commands.extend([
            OP_DISPATCH,
            kernel(name),
            n.min(32768),
            n.div_ceil(32768),
            pass,
        ]);
    }
    fn work(&mut self, name: &str, slot: u32, pass: u32) {
        self.commands
            .extend([OP_INDIRECT, kernel(name), slot, pass]);
    }

    fn write_host(&mut self) {
        let s = self.scratch_words();
        let params = self.params_words(&s);
        let mut counters = [0u32; COUNTER_BYTES as usize / 4];
        counters[0] = self.body_count;
        counters[1] = self.collider_count;
        counters[3] = self.joint_count;
        counters[18] = self.color_round_limit; // read by color_bodies_tail
        let up = &mut self.upload;
        up.fill(0);
        let put = |up: &mut Vec<u8>, at: u32, words: &[u32]| {
            for (k, w) in words.iter().enumerate() {
                let o = at as usize + 4 * k;
                up[o..o + 4].copy_from_slice(&w.to_le_bytes());
            }
        };
        put(up, UPLOAD_SCRATCH, &s);
        put(up, UPLOAD_PARAMS, &params);
        put(up, UPLOAD_COUNTERS, &counters);
        self.copy(HOST_UPLOAD, UPLOAD_SCRATCH, SCRATCH_BUFFER, 0, 256);
        self.copy(HOST_UPLOAD, UPLOAD_PARAMS, PARAMS_BUFFER, 0, PARAMS_BYTES);
        // Keep the previous GPU count before clearing counters (a batch of ticks
        // between readbacks would see a stale contact count otherwise).
        if self.steps > 0 {
            self.copy(COUNTERS_BUFFER, 8, PARAMS_BUFFER, 80 + 7 * 4, 4);
        }
        // Last step's awake count (its awake list is still in gAwakeList) -> HOST_S(6).
        if s[19] != 0 {
            self.copy(COUNTERS_BUFFER, 20, PARAMS_BUFFER, 80 + 6 * 4, 4);
        }
        // Last step's manifold count -> HOST_S(13), for the resting-pair carry.
        if s[28] != 0 && self.steps > 0 {
            self.copy(COUNTERS_BUFFER, 48, PARAMS_BUFFER, 80 + 13 * 4, 4);
        }
        self.moved_only_step = s[19] != 0;
        self.rest_step = s[25] & 1 != 0;
        // Last step's sleep events -> HOST_S(5): any sleeper may be missing from R.
        if self.rest_step {
            self.copy(COUNTERS_BUFFER, 36, PARAMS_BUFFER, 80 + 5 * 4, 4);
        }
        self.rest_valid = self.rest_step;
        self.bounds_stale = false;
        self.copy(
            HOST_UPLOAD,
            UPLOAD_COUNTERS,
            COUNTERS_BUFFER,
            0,
            COUNTER_BYTES,
        );
        self.copy(
            HOST_UPLOAD,
            UPLOAD_COLORS,
            17,
            0,
            (COLOR_COUNT_WORDS * 4) as u32,
        );
    }

    /// Collision only (bounds, broadphase, narrowphase incl. convex).
    fn encode_collision(&mut self, reuse: bool) {
        let n_cols = self.collider_count.max(1);
        let n_bodies = self.body_count.max(1);
        // Moved-only bounds: last step's awake bodies (work slot 78).
        if self.moved_only_step {
            self.disp("write_moved_args", 64, 0);
            self.work("build_aabbs", 78, 0);
        } else {
            self.disp("build_aabbs", n_cols, 0);
        }
        if self.joint_count > 0 {
            // Exact bonded-pair set for the broadphase (broken bonds are left out).
            self.disp("joint_hash_clear", self.joint_hash_slots, 0);
            self.disp("joint_hash_insert", self.joint_count, 0);
        }
        self.phase("bounds");
        if self.rest_step {
            // Resting table: rebuilt only after sleep events (indirect, else empty).
            self.work("rest_clear", 79, 0);
            self.work("rest_insert", 80, 0);
            self.disp("rest_publish", 64, 0);
            if self.ccd_step() {
                self.disp("ccd_register", n_bodies, 0);
            }
            // Colliders of moving bodies, then one thread per collider (slot 81).
            self.work("list_awake_colliders", 78, 0);
            self.disp("write_awake_collider_args", 64, 0);
            self.work("broadphase_insert_awake", 81, 0);
        } else {
            self.disp("broadphase_clear", self.grid_hash_words, 0);
            if self.ccd_step() {
                self.disp("ccd_register", n_bodies, 0);
            }
            self.disp("broadphase_insert", n_cols, 0);
        }
        self.phase("broadphaseInsert");
        if self.rest_step {
            self.work("broadphase_query_awake", 81, 0);
            self.disp("write_finer_args", 64, 0);
            self.work("broadphase_query_finer", 77, 0);
            if self.ccd_step() {
                // Before the awake table is emptied: a fast collider walks it too
                // (bodies woken last step are tombstones in the resting table).
                self.disp("broadphase_fast", engine_schema::FAST_CAP * 64, 0);
            }
            self.work("broadphase_clear_awake", 81, 0);
        } else if self.moved_only_step && self.awake_queries {
            // Only last step's awake bodies query; coarse movers then scan finer
            // cells for resting colliders (same pairs as every collider querying).
            self.work("list_awake_colliders", 78, 0);
            self.disp("write_awake_collider_args", 64, 0);
            self.work("broadphase_query_awake", 81, 0);
            self.disp("write_finer_args", 64, 0);
            self.work("broadphase_query_finer", 77, 0);
        } else {
            self.disp("broadphase_query", n_cols, 0);
        }
        if self.ccd_step() && !self.rest_step {
            // Sweeps longer than a cell query along their path (any table; rest
            // steps walk before the awake table is cleared, above).
            self.disp("broadphase_fast", engine_schema::FAST_CAP * 64, 0);
        }
        self.disp("write_work_args", 64, 0);
        if self.carry_rest() {
            // Last step's manifolds (slot 72) are intact until narrowphase;
            // recount candidates after the carried pairs join them.
            self.work("carry_rest_pairs", 72, 0);
            self.disp("write_work_args", 64, 0);
        }
        if self.ccd_step() {
            // Each CCD body's nearest time of impact and its winning pair; the
            // narrowphase leaves that pair to ccd_emit.
            self.work("ccd_toi", 64, 0);
            self.work("ccd_pick", 64, 0);
            self.disp("ccd_mark", engine_schema::CCD_BODY_CAP, 0);
        }
        self.phase("broadphase");
        if reuse {
            if self.moved_only_step {
                self.work("update_refs", 78, 0); // last step's awake list
            } else {
                self.disp("update_refs", n_bodies, 0);
            }
            if self.warm_clear {
                self.disp("contact_hash_clear", self.warm_hash_words, 0);
            }
            self.work("contact_hash_insert", 68, 0);
        }
        self.work("narrowphase", 64, 0);
        self.phase("narrowphase");
        if self.has_convex {
            self.work("convex_classify", 64, 0);
            self.work("convex_bucket", 64, 0);
            self.disp("convex_write_args", 64, 0);
            self.phase("convexClassify");
            self.work("convex_sat_small", 74, 0);
            self.phase("convexSatSmall");
            self.work("convex_sat_large", 75, 0);
            self.phase("convexSat");
            self.work("convex_manifold", 76, 0);
            self.phase("convexManifold");
        }
        if self.ccd_step() {
            self.work("ccd_emit", 64, 0);
        }
    }

    /// Steps to skip without GPU work: a fully sleeping world whose counters
    /// describe the latest tick. Returns true (and advances `count` steps) to skip.
    pub fn skip_asleep(&mut self, count: u32) -> bool {
        if self.initialized && self.last.awake == 0 && self.counted_steps == self.steps {
            self.steps += count;
            self.submitted_steps = self.steps;
            self.counted_steps = self.steps;
            true
        } else {
            false
        }
    }

    /// Encode one tick (host upload + commands). `reuse_manifolds`: three-avbd
    /// whole-step manifold reuse (no bounds, broadphase or narrowphase), honoured
    /// only when the previous step's manifolds are valid.
    pub fn encode_step(&mut self, reuse_manifolds: bool) -> &[u32] {
        // CCD steps re-run collision (fresh TOI slots and contacts each step).
        let reuse_manifolds = reuse_manifolds && self.manifolds_valid && !self.ccd_step();
        self.commands.clear();
        let n_bodies = self.body_count.max(1);
        self.write_host();
        let reuse = self.params.flags & 1 != 0;
        if reuse_manifolds {
            // Its sleep events never reach a rebuild check: rebuild next time.
            self.rest_valid = false;
            self.disp("write_work_args", 64, 0);
            self.work("reuse_all_manifolds", 73, 0);
            self.phase("reuseManifolds");
        } else {
            self.encode_collision(reuse);
        }
        self.disp("write_work_args", 64, 0);
        if !reuse && !reuse_manifolds {
            if self.warm_clear {
                self.disp("contact_hash_clear", self.warm_hash_words, 0);
            }
            self.work("contact_hash_insert", 68, 0);
        }
        self.work("persist_contacts", 65, 0);
        let joints = self.joint_count;
        if joints > 0 {
            // Bonded bodies wake together (one bond hop per round), then the awake
            // bonds capture their step-start violation.
            for _ in 0..BOND_WAKE_ROUNDS {
                self.disp("bond_wake", joints, 0);
            }
            self.disp("warmstart_joints", joints, 0);
        }
        self.phase("persistence");
        self.disp("build_awake_list", n_bodies, 0);
        self.disp("write_work_args", 64, 0);
        self.disp("csr_clear", n_bodies + 1, 0);
        self.work("csr_degree_contacts", 66, 0);
        if joints > 0 {
            self.disp("csr_degree_joints", joints, 0);
        }
        if self.body_count <= 512 {
            self.disp("csr_prefix", 1, 0);
        } else {
            // Three-dispatch block scan (not log N Hillis-Steele passes).
            self.disp("csr_prefix_blocks", n_bodies.div_ceil(4), 0);
            self.disp("csr_prefix_block_sums", 64, 0);
            self.disp("csr_prefix_add", n_bodies, 0);
        }
        self.work("csr_scatter_contacts", 66, 0);
        if joints > 0 {
            self.disp("csr_scatter_joints", joints, 0);
        }
        self.work("csr_sort", 67, 0);
        self.phase("graph");
        self.work("init_colors", 67, 0);
        self.disp("write_work_args", 64, 0);
        // Complete graph colouring before solving; the shader reports unfinished
        // colours. Slot 69 is zero when the local pass handled the repair list.
        // Dense debris (100k awake pieces) can need more than 32 rounds.
        if self.incremental {
            self.disp("color_bodies_local", 64, 0);
        }
        // Rounds dispatched: the previous step's need plus headroom, not the whole
        // budget (every empty round still costs a barrier). color_bodies_tail runs
        // further rounds in one workgroup only if the last dispatched round left a
        // body blocked, up to the same limit, so colouring always finishes.
        let limit = self.color_round_limit;
        let rounds = match self.color_round_headroom {
            None => limit,
            Some(h) => limit.min((self.last.color_rounds + 1 + h).max(1)),
        };
        let slot = if self.incremental { 69 } else { 67 };
        for round in 0..rounds {
            self.work("color_bodies", slot, round_slot(round));
        }
        if rounds < limit {
            self.disp("color_bodies_tail", 64, round_slot(rounds));
        }
        self.phase("coloring");
        // Stable per-colour body order (locality across every primal iteration).
        let colors = self.cap.max_colors.min(64);
        self.disp("count_color_tiles", n_bodies, 0);
        self.disp("prefix_color_tiles_parallel", colors * 64, 0);
        self.disp("scatter_color_tiles", n_bodies, 0);
        self.work("record_color_degrees", 67, 0);
        self.disp("write_color_args", 64, 0);
        if self.listed_passes {
            self.work("predict_bodies", 67, 1); // awake list (slot 67: awake count)
        } else {
            self.disp("predict_bodies", n_bodies, 0);
        }
        if self.ccd_step() {
            // Struck bodies' trust regions for this step.
            self.disp("ccd_trust", engine_schema::CCD_PAIR_CAP, 0);
        }
        self.work("pack_solver_contacts", 66, 0);
        self.phase("topology");
        // Few awake bodies: one workgroup walks every colour in order (barrier
        // between colours, same per-body solve), one dispatch per iteration instead
        // of one per palette colour. Correct for any count; the previous step's
        // awake count only chooses the faster path.
        let sparse = self.last.awake <= SPARSE_AWAKE;
        // Colour slots dispatched one by one: the previous step's colour count plus
        // headroom, not the whole palette (a scene uses ~8-14 of 64 slots, and every
        // empty dispatch still costs a barrier). Colours at or past the estimate run
        // in solve_primal_sparse starting there: same order, same per-body solve, a
        // barrier between colours. It loops to this step's real colour count on the
        // GPU, so a step with more colours than the estimate is still solved fully.
        let dispatched = match self.color_headroom {
            None => colors,
            Some(h) => colors.min((self.last.color + h).max(1)),
        };
        let iterations = self.params.iterations;
        for iter in 0..iterations {
            if sparse {
                self.disp("solve_primal_sparse", 64, 0);
            } else {
                for color in 0..dispatched {
                    self.work("solve_primal_color", color, color);
                }
                if dispatched < colors {
                    self.disp("solve_primal_sparse", 64, dispatched);
                }
            }
            self.work("solve_dual", 65, (iter + 1 == iterations) as u32);
            // Profiling / slicing boundary halfway through the iterations.
            if iter + 1 == iterations >> 1 {
                self.phase("solveHalf");
            }
        }
        // Post-stabilization: record the velocity, then one primal pass with
        // alpha = 0 (pass value 256 | colour) and no dual update.
        if self.params.flags & crate::scene::FLAG_POST_STAB != 0 {
            if self.listed_passes {
                self.work("record_stab_velocity", 67, 1); // awake list
            } else {
                self.disp("record_stab_velocity", n_bodies, 0);
            }
            if sparse {
                self.disp("solve_primal_sparse", 64, STAB_PASS);
            } else {
                for color in 0..dispatched {
                    self.work("solve_primal_color", color, STAB_PASS + color);
                }
                if dispatched < colors {
                    self.disp("solve_primal_sparse", 64, STAB_PASS + dispatched);
                }
            }
        }
        if self.ccd_step() {
            // Post-solve sweep check: no CCD body ends the step past a surface.
            self.disp("ccd_validate", engine_schema::CCD_PAIR_CAP, 0);
            self.disp("ccd_clamp", engine_schema::CCD_BODY_CAP, 0);
        }
        self.phase("solve");
        if self.listed_passes {
            self.work("extract_velocity_sleep", 67, 1); // awake list
        } else {
            self.disp("extract_velocity_sleep", n_bodies, 0);
        }
        self.disp("write_sleep_args", 64, 0);
        // Pass value = round (1..n), used by the opt-in exact gate.
        for i in 0..self.sleep_rounds {
            self.work("sleep_island_gate", 66, i + 1);
        }
        self.work("sleep_island_commit", 67, self.listed_passes as u32);
        if self.ccd_step() {
            // CCD impacts on dynamic bodies: the momentum exchange.
            self.disp("ccd_respond", engine_schema::CCD_BODY_CAP, 0);
            self.disp("ccd_respond_apply", engine_schema::CCD_BODY_CAP, 0);
        }
        self.phase("sleep");
        self.manifolds_valid = true;
        self.initialized = true;
        self.parity ^= 1;
        self.steps += 1;
        &self.commands
    }

    /// Mark the encoded steps as submitted (counter reads then describe them).
    pub fn mark_submitted(&mut self) {
        self.submitted_steps = self.steps;
    }

    /// Collision kernels only, at the current poses (diagnostics; poses do not
    /// advance). Call `after_inspect` once submitted.
    pub fn encode_inspect(&mut self, flags: u32, awake_queries: bool) -> &[u32] {
        self.commands.clear();
        let (saved_flags, saved_queries) = (self.params.flags, self.awake_queries);
        self.params.flags = flags;
        self.awake_queries = awake_queries;
        self.write_host();
        self.params.flags = saved_flags;
        self.encode_collision(false);
        self.awake_queries = saved_queries;
        self.manifolds_valid = false;
        &self.commands
    }

    /// After an inspect pass: it reset the GPU awake count, so the next step
    /// rebuilds every bound; its host words may have opened a warm-hash tag cycle
    /// without the clear.
    pub fn after_inspect(&mut self) {
        self.bounds_stale = true;
        self.warm_cycle = -1;
    }

    /// Contact base (record index) of the step encoded last (inspect) or of the
    /// last completed step (`previous`).
    pub fn contact_base(&self, previous: bool) -> u32 {
        if (self.parity == 0) != previous {
            0
        } else {
            self.cap.max_contacts
        }
    }

    /// A counter read is about to be queued: `(steps it describes, wake serial)`.
    pub fn begin_read(&self) -> (u32, u32) {
        (self.submitted_steps, self.wake_serial)
    }

    /// Apply a counter read queued at `begin_read`. A launch after that copy
    /// woke a projectile, so an older read must not report a sleeping world.
    pub fn finish_read(&mut self, token: (u32, u32), words: &[u32]) -> Counters {
        let mut c = Counters::from_words(words);
        if token.1 != self.wake_serial {
            c.awake = c.awake.max(1);
        }
        self.last = c;
        self.counted_steps = token.0;
        c
    }

    /// A counter read that reports but does not replace the step counters
    /// (inspection passes count no awake bodies).
    pub fn count_read(&mut self, token: (u32, u32), words: &[u32]) -> Counters {
        self.counted_steps = token.0;
        Counters::from_words(words)
    }

    /// A host write to bodies, colliders or hulls: rebuild every bound next step
    /// (`manifolds` also drops whole-step manifold reuse).
    pub fn mark_stale(&mut self, manifolds: bool) {
        self.bounds_stale = true;
        if manifolds {
            self.manifolds_valid = false;
        }
    }

    /// The overflow error of the last counters, if any.
    pub fn overflow_error(&mut self) -> Result<()> {
        if self.last.overflow != 0 {
            self.manifolds_valid = false;
            bail!(
                "Physics capacity exceeded (0x{:x}); choose a coarser level",
                self.last.overflow
            );
        }
        Ok(())
    }

    /// Replace collider `index` (hull swap, filter change); returns its record.
    pub fn set_collider(&mut self, index: u32, c: Collider) -> Result<Vec<u8>> {
        ensure!(index < self.collider_count, "collider out of range");
        self.mark_stale(true);
        let i = index as usize;
        self.colliders[i] = c;
        let bytes = bytes_of(&collider_record(&c)).to_vec();
        self.collider_bytes[i * COLLIDER_BYTES..(i + 1) * COLLIDER_BYTES].copy_from_slice(&bytes);
        self.has_convex |= c.shape == SHAPE_CONVEX;
        self.has_filters = self.colliders.iter().any(Collider::has_filter);
        Ok(bytes)
    }

    /// The mass (kg) of the projectiles fired from now on (the scene's default
    /// until set; the radius stays).
    pub fn set_projectile_mass(&mut self, mass: f64) -> Result<()> {
        ensure!(mass.is_finite() && mass > 0.0, "Projectile mass must be positive");
        self.projectile_mass = mass;
        Ok(())
    }

    /// The speed of the shots fired from now on, as a multiple of the scene's
    /// ([`Self::shot_speed`] at scale 1).
    pub fn set_projectile_speed_scale(&mut self, scale: f64) -> Result<()> {
        ensure!(scale.is_finite() && scale > 0.0, "Projectile speed scale must be positive");
        self.projectile_speed_scale = scale;
        Ok(())
    }

    /// Shots fired from now on carry `BODY_CCD`: a fast shot stops at a thin
    /// piece instead of passing it between steps (engine/docs/ccd.md).
    pub fn set_projectile_ccd(&mut self, on: bool) {
        self.projectile_ccd = on;
    }

    /// The scene's shot speed (m/s) before the scale ([`default_shot_speed`]).
    pub fn shot_speed(&self) -> f64 {
        default_shot_speed(self.extent, self.projectile_speed)
    }

    /// Fire a projectile along a ray: returns its body id and record
    /// (write it at the id in both body halves). Shots compensate for gravity to
    /// aim near the scene centre's depth unless `arc` is false.
    pub fn launch(
        &mut self,
        origin: [f64; 3],
        direction: [f64; 3],
        speed: Option<f64>,
        arc: Option<bool>,
    ) -> Result<(u32, Vec<u8>)> {
        self.mark_stale(false);
        ensure!(
            origin.iter().chain(&direction).all(|x| x.is_finite()),
            "Invalid projectile ray"
        );
        // No limit on shots: past the slots, each shot reuses the oldest projectile (the
        // world holds the newest `projectile_count`).
        ensure!(self.projectile_count > 0, "The scene has no projectile slots");
        let id = self.projectile_start + self.shots % self.projectile_count;
        let (m, r) = (self.projectile_mass, self.projectile_radius);
        let v = shot_velocity(
            self.center,
            self.extent,
            r,
            origin,
            direction,
            Some(speed.unwrap_or_else(|| self.shot_speed()) * self.projectile_speed_scale),
            arc.unwrap_or(self.projectile_arc),
        );
        let body = Body {
            p: origin,
            q: [0.0, 0.0, 0.0, 1.0],
            v,
            w: [0.0; 3],
            inv_mass: 1.0 / m,
            inv_i: [1.0 / (0.4 * m * r * r); 3],
            flags: if self.projectile_ccd { 9 | engine_schema::BODY_CCD } else { 9 },
        };
        self.manifolds_valid = false;
        self.last.awake = self.last.awake.max(1);
        self.wake_serial += 1;
        self.shots += 1;
        self.ccd_fired |= self.projectile_ccd;
        Ok((id, bytes_of(&body_record(&body)).to_vec()))
    }

    /// Byte offset of body `id` in the current and previous halves of buffer 0.
    pub fn body_offsets(&self, id: u32) -> [u64; 2] {
        [
            id as u64 * BODY_BYTES as u64,
            (self.cap.max_bodies + id) as u64 * BODY_BYTES as u64,
        ]
    }
}

/// Pass-uniform slot whose value is `round` (slots 48..67 hold scan constants).
pub fn round_slot(round: u32) -> u32 {
    if round < 48 {
        round
    } else {
        round + 20
    }
}

/// Phase names, in `Command::Phase` id order.
pub fn phase_names() -> Vec<&'static str> {
    PHASES.iter().map(|(n, _)| *n).collect()
}

/// Bodies-snapshot helper: the `p` and `q` of body `i` from 28-word records.
pub fn snapshot_pose(snapshot: &[f32], i: usize) -> ([f32; 3], [f32; 4]) {
    let o = i * 28;
    (
        [snapshot[o], snapshot[o + 1], snapshot[o + 2]],
        [
            snapshot[o + 4],
            snapshot[o + 5],
            snapshot[o + 6],
            snapshot[o + 7],
        ],
    )
}

/// A shot's launch velocity (asset frame, +Y up) along `direction` from `origin`: at
/// `speed`, or by default three times the scene's extent per second (5 to 18 m/s), and
/// with `arc` lifted to cancel the gravity drop over the distance to the scene centre's
/// depth. The lab's every bond mode fires by this rule (`Planner::launch` and the stress
/// mode's `stress-web`).
/// A scene's shot speed (m/s): its own, else three scene extents a second within 5–18.
pub fn default_shot_speed(extent: f64, speed: Option<f64>) -> f64 {
    speed.unwrap_or_else(|| (extent * 3.0).clamp(5.0, 18.0))
}

pub fn shot_velocity(
    center: [f64; 3],
    extent: f64,
    radius: f64,
    origin: [f64; 3],
    direction: [f64; 3],
    speed: Option<f64>,
    arc: bool,
) -> [f64; 3] {
    let speed = default_shot_speed(extent, speed);
    let distance = radius.max(
        (0..3)
            .map(|i| (center[i] - origin[i]) * direction[i])
            .sum(),
    );
    let mut v = direction.map(|x| x * speed);
    if arc {
        v[1] += 0.5 * 9.81 * distance / speed;
    }
    v
}

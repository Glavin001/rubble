//! One rigid-body step graph for every host (ENGINE_AUDIT P0).
//!
//! `encode_step` turns a step's configuration and the previous step's counters
//! into an ordered command list; the native wgpu host, the Metal host and (via
//! wasm) the browser replay the same list, so a measured win lands on all of
//! them at once. Kernel names are logical (canonical) entry points; hosts map
//! them to the compiled variant with [`resolve_entry`].

use engine_schema::{AvbdParams, StepCounters, WorldCapacity};

/// Jones-Plassmann rounds a step may use (dispatched plus tail).
pub const COLOR_ROUND_LIMIT: u32 = 32;
/// Sleep times at or above this never elapse in a run (benchmarks use 1e6 s).
pub const SLEEP_NEVER_S: f32 = 1.0e5;
/// Buffer id of the host-written counter block a pipelined step copies into
/// binding 7 (not one of the 31 GPU bindings).
pub const COUNTER_UPLOAD: u32 = u32::MAX;
/// Host binding of the counters, and of the params/host uniform.
pub const COUNTERS_BINDING: u32 = 7;
pub const HOST_UNIFORM_BINDING: u32 = 6;
/// Byte offset of `HostUniforms.s[k]` (after the 80-byte `AvbdParams`).
pub const fn host_word_offset(k: u32) -> u32 {
    80 + 4 * k
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    /// Buffer copy, encoded before the compute pass.
    Copy {
        src: u32,
        src_offset: u32,
        dst: u32,
        dst_offset: u32,
        size: u32,
    },
    /// Direct dispatch of `threads` invocations (64 per workgroup).
    Dispatch {
        kernel: &'static str,
        threads: u64,
        pass: u32,
    },
    /// End of a named phase (profiling hosts split timing here; others ignore it).
    Phase { name: &'static str },
    /// Indirect dispatch from work/colour slot `slot` of binding 23.
    Indirect {
        kernel: &'static str,
        slot: u32,
        pass: u32,
    },
}

/// Modes a host resolved for this step (already gated on each other).
#[derive(Clone, Copy, Debug, Default)]
pub struct StepModes {
    pub compact_contacts: bool,
    pub persistent_contacts: bool,
    pub manifold_persistence: bool,
    pub fused_persistence: bool,
    pub manifold_insert: bool,
    pub copy_reuse: bool,
    pub split_np: bool,
    pub listed_persist: bool,
    pub carry_rest: bool,
    pub local_repair: bool,
    pub grid_split: bool,
    /// Colours from this index always run in the sparse tail.
    pub sparse_start: u32,
    pub color_headroom: Option<u32>,
    pub color_round_headroom: Option<u32>,
    /// The sparse tail's single-workgroup loop can cover the body capacity.
    pub tail_ok: bool,
}

/// Scene sizes and last step's counters this step is planned from.
#[derive(Clone, Copy, Debug)]
pub struct StepInputs {
    pub body_count: u32,
    pub collider_count: u32,
    pub joint_count: u32,
    pub joint_hash_slots: u32,
    pub has_convex: bool,
    pub reuse_contacts: bool,
    pub max_colors: u32,
    pub iterations: u32,
    pub sleep_time: f32,
    pub sleep_island_rounds: u32,
    pub grid_hash_words: u32,
    /// Warm-hash words the clear dispatch covers (slots × 1 or 4).
    pub warm_hash_words: u64,
    pub steps_done: u32,
    pub last: StepCounters,
    /// Vehicle controller pool in use (`vehicle_step.slang`; 0 = no vehicle passes).
    pub vehicle_count: u32,
    pub wheel_count: u32,
    /// Continuous collision and step-start motion (`ccd_step.slang`).
    pub motion: MotionInputs,
}

/// What the continuous-collision, impulse, projectile and character passes
/// need this step (all off: the step graph is unchanged).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionInputs {
    /// Some body carries `BODY_CCD` (registration, TOI casts, TOI manifolds).
    pub ccd: bool,
    /// Some body is kinematic (`BODY_KINEMATIC`).
    pub kinematic: bool,
    /// `FLAG_SPEC_VELOCITY` is set (swept bounds: the fast list may fill).
    pub look_ahead: bool,
    /// The host queued impulse records, or projectiles may write some.
    pub impulses: bool,
    /// Projectile records in the character/projectile pool (`projectile_step`).
    pub projectiles: u32,
    /// Character controllers in the pool (`character_step.slang`).
    pub characters: u32,
    /// In-step scene query records (`query_batch`).
    pub queries: u32,
    /// Measurement control: never schedule the fast list (long sweeps then
    /// stay in the 27-cell stencil, as before the CCD audit's row 1).
    pub fast_list_off: bool,
}

/// Whether a step dispatches `broadphase_fast` (host word 30 bit 4 tells the
/// kernels, which divert long sweeps to the fast list only then).
pub fn fast_list_scheduled(mo: &MotionInputs) -> bool {
    (mo.ccd || mo.look_ahead || mo.kinematic) && !mo.fast_list_off
}

/// `apply_impulses` threads (one per record slot).
pub const IMPULSE_CAP: u32 = engine_schema::IMPULSE_CAP;
/// `ccd_mark` threads (one per CCD body slot).
pub const CCD_BODY_CAP: u32 = engine_schema::CCD_BODY_CAP;
/// `broadphase_fast` workgroups (they stride the fast list).
pub const FAST_GROUPS: u64 = 64;

/// JP rounds dispatched this step; later rounds run in `color_bodies_tail`.
pub fn color_rounds_dispatched(m: &StepModes, last: &StepCounters) -> u32 {
    match m.color_round_headroom {
        None => COLOR_ROUND_LIMIT,
        Some(h) => COLOR_ROUND_LIMIT.min((last.color_rounds + 1 + h).max(1)),
    }
}

/// Primal colours dispatched one by one; later colours run in the sparse tail.
pub fn primal_colors_dispatched(m: &StepModes, i: &StepInputs, max_c: u32) -> u32 {
    let estimate = match m.color_headroom {
        Some(h) if m.tail_ok && i.steps_done > 0 => (i.last.color_count + h).max(1),
        _ => max_c,
    };
    estimate.min(m.sparse_start).min(max_c)
}

/// Canonical entry → the variant a persistent-contact host compiled.
pub fn resolve_entry(name: &str, persistent: bool) -> &str {
    if !persistent {
        return name;
    }
    match name {
        "apply_commands" => "apply_commands_persistent",
        "narrowphase" => "narrowphase_persistent",
        "contact_hash_insert" => "contact_hash_insert_persistent",
        "contact_hash_insert_manifolds" => "contact_hash_insert_manifolds_persistent",
        "persist_contacts" => "persist_contacts_persistent",
        "persist_manifolds" | "persist_manifolds_compact" => "persist_manifolds_persistent",
        "pack_solver_contacts" => "pack_solver_pairs_persistent",
        // 8el: persistent hosts pack pairs before the graph; CSR reads their body ids.
        "csr_degree_contacts" => "csr_degree_contacts_persistent",
        "csr_scatter_contacts" => "csr_scatter_contacts_persistent",
        "solve_primal_color_compact" => "solve_primal_color_persistent",
        "solve_primal_sparse_compact" => "solve_primal_sparse_persistent",
        "solve_primal_one_compact" => "solve_primal_one_persistent",
        "solve_dual_compact" => "solve_dual_persistent",
        "convex_classify" => "convex_classify_persistent",
        "convex_manifold" => "convex_manifold_persistent",
        "ccd_emit" => "ccd_emit_persistent",
        "reuse_all_manifolds" => "reuse_all_manifolds_persistent",
        _ => name,
    }
}

struct Enc(Vec<Cmd>);
impl Enc {
    fn d(&mut self, kernel: &'static str, threads: u64) {
        self.0.push(Cmd::Dispatch {
            kernel,
            threads,
            pass: 0,
        });
    }
    fn dp(&mut self, kernel: &'static str, threads: u64, pass: u32) {
        self.0.push(Cmd::Dispatch {
            kernel,
            threads,
            pass,
        });
    }
    fn w(&mut self, kernel: &'static str, slot: u32) {
        self.0.push(Cmd::Indirect {
            kernel,
            slot,
            pass: 0,
        });
    }
    fn phase(&mut self, name: &'static str) {
        self.0.push(Cmd::Phase { name });
    }
    fn wp(&mut self, kernel: &'static str, slot: u32, pass: u32) {
        self.0.push(Cmd::Indirect { kernel, slot, pass });
    }
}

/// The full step. `pipelined`: no host readback between steps, so last step's
/// GPU counts are forwarded into the host uniform before the counters reset.
pub fn encode_step(
    m: &StepModes,
    i: &StepInputs,
    reuse_manifolds: bool,
    pipelined: bool,
) -> Vec<Cmd> {
    let mut e = Enc(Vec::with_capacity(256));
    let n_bodies = u64::from(i.body_count.max(1));
    let n_cols = u64::from(i.collider_count.max(1));
    let n_joints = u64::from(i.joint_count.max(1));
    let work_args = if m.listed_persist {
        "write_work_args_copy_persistent"
    } else {
        "write_work_args"
    };

    if pipelined {
        if i.steps_done > 0 {
            e.0.push(Cmd::Copy {
                src: COUNTERS_BINDING,
                src_offset: 8,
                dst: HOST_UNIFORM_BINDING,
                dst_offset: host_word_offset(7),
                size: 4,
            });
            if m.manifold_insert || m.carry_rest {
                e.0.push(Cmd::Copy {
                    src: COUNTERS_BINDING,
                    src_offset: 48,
                    dst: HOST_UNIFORM_BINDING,
                    dst_offset: host_word_offset(13),
                    size: 4,
                });
            }
        }
        e.0.push(Cmd::Copy {
            src: COUNTER_UPLOAD,
            src_offset: 0,
            dst: COUNTERS_BINDING,
            dst_offset: 0,
            size: std::mem::size_of::<StepCounters>() as u32,
        });
    }

    let vehicles = i.vehicle_count > 0;
    let mo = i.motion;
    // GPU impulses first (CCD audit row 4): scripted loads and projectile hits
    // change velocities before bounds are swept.
    if mo.impulses || mo.projectiles > 0 {
        e.d("apply_impulses", u64::from(IMPULSE_CAP));
    }
    if vehicles {
        e.d("vehicle_input", u64::from(i.vehicle_count));
    }
    if !reuse_manifolds {
        e.d("build_aabbs", n_cols);
        if i.joint_count > 0 {
            e.d("joint_hash_clear", u64::from(i.joint_hash_slots));
            e.d("joint_hash_insert", n_joints);
        }
        e.d(
            "broadphase_clear",
            u64::from(i.grid_hash_words) * if m.grid_split { 2 } else { 1 },
        );
        if mo.ccd {
            e.d("ccd_register", n_bodies);
        }
        e.d("broadphase_insert", n_cols);
        e.d("broadphase_query", n_cols);
        if fast_list_scheduled(&mo) {
            // Sweeps longer than a cell query along their path (row 1).
            e.d("broadphase_fast", FAST_GROUPS * 64);
        }
        if vehicles {
            // One workgroup per wheel, cast at x⁻ through this step's grid (the
            // contact hash reuses its heads later in the step); then, before
            // the constraint graph, link wheels standing on dynamic bodies.
            e.d("wheel_query", u64::from(i.wheel_count) * 64);
            e.d("vehicle_link", u64::from(i.wheel_count.max(1)));
        }
        // Projectiles, queries and characters walk the same grid, in the same
        // window. Projectile hits push bodies through a second (late)
        // apply_impulses pass over the records they appended.
        if mo.projectiles > 0 {
            e.d("projectile_step", u64::from(mo.projectiles) * 64);
            e.dp("apply_impulses", u64::from(IMPULSE_CAP), 1);
        }
        if mo.queries > 0 {
            e.d("query_batch", u64::from(mo.queries) * 64);
        }
        if mo.characters > 0 {
            e.d("char_move", u64::from(mo.characters) * 64);
        }
    }
    e.d(work_args, 64);
    e.phase("broadphase");
    if m.carry_rest && !reuse_manifolds {
        // Previous manifolds (slot 72) are intact until narrowphase; recount
        // candidates (slot 64) after the carried resting pairs join them.
        e.w("carry_rest_pairs", 72);
        e.d(work_args, 64);
    }
    if mo.ccd && !reuse_manifolds {
        // Each CCD body's nearest time of impact, then its winning pair
        // (rows 2-3); the narrowphase leaves that pair to ccd_emit.
        e.w("ccd_toi", 64);
        e.w("ccd_pick", 64);
        e.d("ccd_mark", u64::from(CCD_BODY_CAP));
    }
    let reuse = i.reuse_contacts;
    if reuse_manifolds {
        // Every previous run becomes this step's manifold; no hash needed.
        e.w("reuse_all_manifolds", 73);
    } else {
        if reuse {
            e.d("update_refs", n_bodies);
            e.d("contact_hash_clear", i.warm_hash_words);
            e.w(
                if m.manifold_insert {
                    "contact_hash_insert_manifolds"
                } else {
                    "contact_hash_insert"
                },
                68,
            );
        }
        // Narrowphase overwrites gManifolds; the manifold insert must precede it.
        assert!(
            reuse || !m.manifold_insert,
            "manifold hash insert requires contact reuse"
        );
        if m.split_np {
            e.w("narrowphase_defer_copy_persistent", 64);
            e.w("narrowphase_fresh_copy_persistent", 64);
        } else {
            e.w(
                if m.copy_reuse {
                    "narrowphase_copy_persistent"
                } else {
                    "narrowphase"
                },
                64,
            );
        }
        if i.has_convex {
            // Hull pairs deferred by narrowphase: classify (cheap pairs emit
            // here), bucket hull pairs, size-bucketed cooperative SAT, then clipping.
            e.w("convex_classify", 64);
            e.w("convex_bucket", 64);
            e.d("convex_write_args", 64);
            e.w("convex_sat_small", 74);
            e.w("convex_sat_large", 75);
            e.w("convex_manifold", 76);
        }
        if mo.ccd {
            e.w(if m.copy_reuse { "ccd_emit_copy_persistent" } else { "ccd_emit" }, 64);
        }
    }
    e.phase("narrowphase");
    e.d(work_args, 64);
    if !reuse && !reuse_manifolds {
        e.d("contact_hash_clear", i.warm_hash_words);
        e.w("contact_hash_insert", 68);
    }
    if m.fused_persistence && !m.persistent_contacts {
        e.w("clear_solver_contacts", 65);
    }
    if m.listed_persist && !reuse_manifolds {
        // Whole-step reuse lists nothing, so it takes the full scan below.
        // Slot 70 covers listed contacts; slot 71 runs the full scan only when
        // the list overflowed (both written after narrowphase).
        e.w("persist_list_copy_persistent", 70);
        e.w("persist_contacts_copy_persistent", 71);
    } else {
        e.w(
            if m.fused_persistence {
                "persist_manifolds_compact"
            } else if m.manifold_persistence {
                "persist_manifolds"
            } else if m.copy_reuse {
                "persist_contacts_copy_persistent"
            } else {
                "persist_contacts"
            },
            if m.fused_persistence || m.manifold_persistence {
                66
            } else {
                65
            },
        );
    }
    // Persistent CSR reads solver pairs, so pack them before the graph unless
    // manifold persistence already did.
    let early_pair_pack = m.persistent_contacts && !m.fused_persistence && !m.manifold_persistence;
    if early_pair_pack {
        e.w("pack_solver_contacts", 66);
    }
    if i.joint_count > 0 {
        e.d("warmstart_joints", n_joints);
    }
    e.phase("warm");
    e.d("build_awake_list", n_bodies);
    e.d(work_args, 64);
    e.d("csr_clear", n_bodies + 1);
    e.w("csr_degree_contacts", 66);
    if i.joint_count > 0 {
        e.d("csr_degree_joints", n_joints);
    }
    if i.body_count <= 512 {
        e.d("csr_prefix", 1);
    } else {
        e.d("csr_prefix_blocks", n_bodies.div_ceil(4));
        e.d("csr_prefix_block_sums", 64);
        e.d("csr_prefix_add", n_bodies);
    }
    e.w("csr_scatter_contacts", 66);
    if i.joint_count > 0 {
        e.d("csr_scatter_joints", n_joints);
    }
    e.w("csr_sort", 67);
    e.phase("graph");
    e.w("init_colors", 67);
    e.d(work_args, 64);
    if m.local_repair {
        // Slot 69 is zero when the local pass handled the repair list.
        e.d("color_bodies_local", 64);
    }
    // Rounds dispatched: last step's need plus headroom; color_bodies_tail runs
    // any further rounds in one workgroup up to the same limit, so colouring
    // always finishes (8cl). Pass value = round index (slots 0..47 hold i).
    let rounds = color_rounds_dispatched(m, &i.last);
    for round in 0..rounds {
        e.wp("color_bodies", 69, round);
    }
    if rounds < COLOR_ROUND_LIMIT {
        e.dp("color_bodies_tail", 64, rounds);
    }
    e.phase("coloring");
    let max_c = i.max_colors.min(32);
    e.d("count_color_tiles", n_bodies);
    e.d("prefix_color_tiles_parallel", u64::from(max_c) * 64);
    e.d("scatter_color_tiles", n_bodies);
    // Per-colour max CSR degree for degree-aware primal lanes (s[20]).
    e.w("record_color_degrees", 67);
    e.d("write_color_args", 64);
    e.d("predict_bodies", n_bodies);
    if vehicles {
        e.d("vehicle_prepare", u64::from(i.vehicle_count));
    }

    if m.compact_contacts && !m.fused_persistence && !early_pair_pack {
        if !m.persistent_contacts {
            e.w("clear_solver_contacts", 65);
        }
        e.w("pack_solver_contacts", 66);
    }
    e.phase("topology");
    // Colours dispatched one by one: last step's colour count plus headroom (an
    // empty indirect dispatch still costs a barrier). Later colours run in
    // solve_primal_sparse, which loops to this step's real GPU colour count.
    let sparse_start = primal_colors_dispatched(m, i, max_c);
    let (primal, sparse, dual) = if m.compact_contacts {
        (
            "solve_primal_color_compact",
            "solve_primal_sparse_compact",
            "solve_dual_compact",
        )
    } else {
        ("solve_primal_color", "solve_primal_sparse", "solve_dual")
    };
    for iter in 0..i.iterations {
        for color in 0..max_c.min(sparse_start) {
            e.wp(primal, color, color);
        }
        if sparse_start < max_c {
            e.dp(sparse, 64, sparse_start);
        }
        e.phase("primal");
        e.wp(dual, 65, u32::from(iter + 1 == i.iterations));
        if vehicles {
            e.d("vehicle_iterate", u64::from(i.vehicle_count));
        }
        e.phase("dual");
    }
    e.d("extract_velocity_sleep", n_bodies);
    if vehicles {
        e.d("vehicle_post", u64::from(i.vehicle_count));
    }
    if mo.characters > 0 {
        // One lane walks the characters in order (weights on shared supports).
        e.d("char_post", 64);
    }
    e.d("write_sleep_args", 64);
    // A sleep time beyond any run's length cannot produce candidates, so the
    // island gate has nothing to revoke (ring/gables benchmarks: 1e6 s).
    if i.sleep_island_rounds > 0 && i.sleep_time < SLEEP_NEVER_S {
        for _ in 0..i.sleep_island_rounds.min(16) {
            e.w("sleep_island_gate", 66);
        }
    }
    e.w("sleep_island_commit", 67);
    if mo.ccd && !reuse_manifolds {
        // CCD impacts on dynamic bodies: the momentum exchange (ccd_step.slang).
        e.d("ccd_respond", 64);
    }
    e.phase("sleep");
    e.0
}

/// Entries a persistent-contact host replays (every name `encode_step` can
/// emit, resolved with [`resolve_entry`]). Native hosts compile exactly these.
pub const PERSISTENT_ENTRIES: &[&str] = &[
    "apply_impulses",
    "ccd_register",
    "broadphase_fast",
    "ccd_toi",
    "ccd_pick",
    "ccd_mark",
    "ccd_respond",
    "ccd_emit_persistent",
    "ccd_emit_copy_persistent",
    "char_move",
    "char_post",
    "query_batch",
    "projectile_step",
    "vehicle_input",
    "wheel_query",
    "vehicle_link",
    "vehicle_prepare",
    "vehicle_iterate",
    "vehicle_post",
    "build_aabbs",
    "joint_hash_clear",
    "joint_hash_insert",
    "broadphase_clear",
    "broadphase_insert",
    "broadphase_query",
    "write_work_args",
    "write_work_args_copy_persistent",
    "carry_rest_pairs",
    "reuse_all_manifolds_persistent",
    "update_refs",
    "contact_hash_clear",
    "contact_hash_insert_persistent",
    "contact_hash_insert_manifolds_persistent",
    "narrowphase_persistent",
    "narrowphase_copy_persistent",
    "narrowphase_defer_copy_persistent",
    "narrowphase_fresh_copy_persistent",
    "convex_classify_persistent",
    "convex_bucket",
    "convex_write_args",
    "convex_sat_small",
    "convex_sat_large",
    "convex_manifold_persistent",
    "persist_contacts_persistent",
    "persist_contacts_copy_persistent",
    "persist_list_copy_persistent",
    "persist_manifolds_persistent",
    "pack_solver_pairs_persistent",
    "warmstart_joints",
    "build_awake_list",
    "csr_clear",
    "csr_degree_contacts_persistent",
    "csr_degree_joints",
    "csr_prefix",
    "csr_prefix_blocks",
    "csr_prefix_block_sums",
    "csr_prefix_add",
    "csr_scatter_contacts_persistent",
    "csr_scatter_joints",
    "csr_sort",
    "init_colors",
    "color_bodies_local",
    "color_bodies",
    "color_bodies_tail",
    "count_color_tiles",
    "prefix_color_tiles_parallel",
    "scatter_color_tiles",
    "record_color_degrees",
    "write_color_args",
    "predict_bodies",
    "solve_primal_color_persistent",
    "solve_primal_sparse_persistent",
    "solve_dual_persistent",
    "extract_velocity_sleep",
    "write_sleep_args",
    "sleep_island_gate",
    "sleep_island_commit",
];

/// Character, query and projectile entries (`character_step.slang`).
pub const CHARACTER_ENTRIES: &[&str] = &["char_move", "char_post", "query_batch", "projectile_step"];

/// Slang source (under `slang/kernels/`) that defines `entry`.
/// Vehicle controller entries (`vehicle_step.slang`).
pub const VEHICLE_ENTRIES: &[&str] = &[
    "vehicle_input",
    "wheel_query",
    "vehicle_link",
    "vehicle_prepare",
    "vehicle_iterate",
    "vehicle_post",
    "raycast_batch",
];

pub fn kernel_source(entry: &str) -> &'static str {
    if VEHICLE_ENTRIES.contains(&entry) {
        return "vehicle_step.slang";
    }
    if CHARACTER_ENTRIES.contains(&entry) {
        return "character_step.slang";
    }
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

/// Broadphase cell size and hierarchical levels for a scene. Uniform grids use
/// twice the largest small dynamic radius (larger bodies take the large list);
/// `hierarchical` (convex scenes) sizes levels from every collider.
pub fn grid_cell(
    cols: &[engine_schema::Collider],
    bodies: &[engine_schema::BodyState],
    hierarchical: bool,
) -> (f32, u32) {
    if hierarchical {
        return engine_schema::convex::hierarchical_grid(cols, bodies);
    }
    let mut max_r = 0.25f32;
    for c in cols {
        if c.shape == engine_schema::ShapeType::Plane as u32 {
            continue;
        }
        let body = c.body as usize;
        if body >= bodies.len() || bodies[body].inv_mass <= 0.0 {
            continue;
        }
        let r = if c.params[3] > 0.0 {
            c.params[3]
        } else {
            (c.params[0] * c.params[0] + c.params[1] * c.params[1] + c.params[2] * c.params[2])
                .sqrt()
        };
        if r > 2.0 {
            continue;
        }
        max_r = max_r.max(r);
    }
    ((2.0 * max_r * 1.0001).max(0.5), 0)
}

/// Warm-start pair key, bit-identical to `pairWarmKey` in world_step.slang
/// (murmur3 fmix32 finalizer over both collider ids; never the empty sentinel).
pub fn pair_warm_key(a: u32, b: u32) -> u32 {
    let mut h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_add(0x7F4A_7C15).wrapping_mul(0x85EB_CA77);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    if h == u32::MAX {
        a ^ b ^ 0xA5A5_A5A5
    } else {
        h
    }
}

/// GPU bindings every rigid host allocates (`BIND(0..30)` in world_step.slang,
/// 31 the vehicle pool, 32 the character / query / projectile pool).
pub const BIND_COUNT: usize = 33;
/// Pass-uniform slot stride (binding 22, dynamic offsets).
pub const PASS_ALIGN: u64 = 256;

/// `gConstraintIds` words before the CSR index (grid heads, warm hash, ids).
pub fn constraint_id_words(c: &WorldCapacity) -> u32 {
    let words = (2 * (c.max_contacts + c.max_joints))
        .max(32768)
        .next_power_of_two() as u64;
    words.max(131072) as u32
}

/// Broadphase grid heads: at least twice the colliders (three-avbd), so 506k
/// gables avoids ~4 colliders per bucket; small scenes keep 131072. `fixed`
/// keeps the old table (control); `split` halves the cap so both tables fit.
pub fn grid_hash_words(c: &WorldCapacity, colliders: u32, split: bool, fixed: bool) -> u32 {
    if fixed {
        return 131072;
    }
    let region = constraint_id_words(c) / if split { 2 } else { 1 };
    let cap = 1u32 << (31 - region.leading_zeros());
    (2 * colliders).max(131072).next_power_of_two().min(cap)
}

/// Warm hash size in slots: one word per slot (heads), or four-word record
/// slots at ~2x manifolds, never below 4x bodies, within the pre-CSR region.
pub fn warm_hash_slots(c: &WorldCapacity, bodies: u32, prev_contacts: u32, record: bool) -> u32 {
    let ids_len = constraint_id_words(c);
    if record {
        (prev_contacts / 2)
            .max(4 * bodies)
            .max(1024)
            .next_power_of_two()
            .min(ids_len / 4)
    } else {
        const MAX_WARM_HASH: u32 = 1 << 21;
        prev_contacts
            .saturating_mul(4)
            .max(1024)
            .next_power_of_two()
            .min(ids_len)
            .min(MAX_WARM_HASH)
    }
}

/// Byte size of every binding for a capacity and contact layout.
pub fn buffer_sizes(c: &WorldCapacity, compact: bool, persistent: bool) -> [u64; BIND_COUNT] {
    let b = |n: u32, stride: u64| n as u64 * stride;
    let body = std::mem::size_of::<engine_schema::BodyState>() as u64;
    let mut s = [256u64; BIND_COUNT];
    s[0] = b(c.max_bodies.saturating_mul(2), body);
    s[1] = b(c.max_bodies, body);
    s[2] = b(c.max_colliders, 64);
    s[3] = b(64, 16);
    s[4] = b(c.max_contacts.saturating_mul(2), 128);
    s[5] = b(c.max_joints, std::mem::size_of::<engine_schema::Joint>() as u64);
    s[6] = 256;
    s[7] = 256;
    s[8] = b(c.max_commands, 64);
    s[9] = b(c.max_bodies + c.max_colliders, 32);
    s[10] = b(c.max_candidates, 16);
    s[11] = b(c.max_bodies, 4);
    s[12] = b(c.max_bodies, 4);
    s[13] = b(c.max_bodies + 1, 4);
    s[14] = (u64::from(constraint_id_words(c)) + u64::from(c.max_bodies) + 1) * 4;
    s[15] = b(c.max_bodies * (c.max_colors + 2), 4);
    s[16] = b(c.max_colors * c.max_bodies.div_ceil(64).max(1), 4);
    s[17] = b(c.max_colors.max(128), 4);
    s[18] = b(c.max_hull_vertices, 16);
    s[19] = u64::from(engine_schema::joint_hash_size(c.max_joints)) * 3 * 4;
    s[20] = b(c.max_bodies, 4);
    // Scratch: host-reset words 0..63, uniform-grid large list 64..79, the
    // oversized-static count (80), then the event ring and the static list.
    s[21] = u64::from(engine_schema::SCRATCH_WORDS) * 4;
    s[22] = 96 * PASS_ALIGN;
    // Colour slots 0..63, work 64..73 (persist list 70/71, carry 72, previous
    // contacts 73), convex 74..76.
    s[23] = b(c.max_colors.max(77) * 3, 4);
    s[24] = b(c.max_colliders, 80);
    s[25] = b(c.max_candidates, 32);
    s[26] = b(c.max_bodies, 32);
    s[27] = b(c.max_contacts, 64);
    s[28] = b(c.max_candidates, 32);
    s[29] = b(c.max_candidates, 32);
    // SAT separating-feature cache (convex_step gSatCache): uint2 slots, power of two.
    s[30] = u64::from(c.max_candidates.max(1024).next_power_of_two()) * 8;
    // Vehicle pool (vehicle_step gVehicleData): hosts with vehicles resize it.
    if persistent {
        s[4] = u64::from(c.max_contacts) * 2 * 32;
        s[27] = u64::from(c.max_contacts) * 2 * 64;
    }
    if !compact {
        s[27] = 64;
        s[28] = 32;
    }
    s
}

/// Pass uniform table (binding 22): slot i holds i below 48, block-scan
/// constants at 48..67, zero above.
pub fn pass_table() -> Vec<u8> {
    let mut pass = vec![0u8; (96 * PASS_ALIGN) as usize];
    for i in 0..96u32 {
        let v = if i < 48 {
            i
        } else if i < 68 {
            let k = i - 48;
            (1u32 << k) | ((k & 1) << 31)
        } else {
            0
        };
        let o = (i as usize) * PASS_ALIGN as usize;
        pass[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    pass
}

/// Matches `HostUniforms` in `world_step.slang` (`AvbdParams` then `uint4 s[8]`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HostUniforms {
    pub params: AvbdParams,
    pub s: [[u32; 4]; 8],
}

/// Everything the host words depend on besides the modes.
#[derive(Clone, Copy, Debug)]
pub struct HostState {
    pub contacts_parity: u32,
    pub grid_levels: u32,
    pub has_filters: bool,
    pub record_hash: bool,
    pub np_defer_limit: u32,
    pub persist_list_limit: u32,
}

/// Host scratch words (HOST_S / binding 21) for this step.
pub fn host_words(c: &WorldCapacity, m: &StepModes, i: &StepInputs, h: &HostState) -> [u32; 64] {
    let max_c = c.max_contacts;
    let cur = if h.contacts_parity == 0 { 0 } else { max_c };
    let prev = if h.contacts_parity == 0 { max_c } else { 0 };
    let prev_n = i.last.contact_count.min(max_c).max(256);
    let mut s = [0u32; 64];
    s[1] = c.max_bodies;
    s[2] = c.max_candidates;
    s[3] = max_c;
    s[4] = c.max_colors.min(32);
    s[7] = i.last.contact_count.min(max_c);
    s[8] = prev;
    s[9] = warm_hash_slots(c, i.body_count, prev_n, h.record_hash);
    s[10] = c.max_candidates;
    s[11] = cur;
    let mut log = 0u32;
    while (1u32 << log) < i.body_count.max(1) && log < 20 {
        log += 1;
    }
    s[12] = if log == 0 { 0 } else { (log - 1) & 1 };
    if m.manifold_insert || m.carry_rest {
        s[13] = i.last.manifold_count;
    }
    if m.manifold_insert {
        s[14] = 1;
    }
    s[15] = u32::from(h.record_hash);
    s[16] = i.grid_hash_words;
    s[17] = i.steps_done;
    s[18] = h.grid_levels;
    s[20] = u32::from(m.grid_split);
    s[21] = h.np_defer_limit;
    s[22] = u32::from(m.listed_persist);
    s[23] = h.persist_list_limit;
    // CSR index region starts past the hash/id words inside gConstraintIds.
    s[24] = constraint_id_words(c);
    s[26] = 1; // incremental coloring: history row + compact work slot 69
    s[27] = u32::from(m.local_repair);
    s[28] = u32::from(m.carry_rest);
    // Degree-aware primal lanes for convex/compound scenes (high-degree bodies).
    s[29] = u32::from(i.has_convex);
    // Bit 0: category/mask/group filters. Bit 1: vehicle sever grace (severed
    // parts ignore their chassis by collider generation) unless filters are on.
    s[30] = u32::from(h.has_filters) | (u32::from(!h.has_filters && i.vehicle_count > 0) << 1);
    // Bit 2: CCD bodies (ccd_step.slang sceneHasCcd); bit 3: kinematic bodies;
    // bit 4: broadphase_fast is scheduled (encode_step's condition).
    let mo = i.motion;
    s[30] |= (u32::from(mo.ccd) << 2) | (u32::from(mo.kinematic) << 3)
        | (u32::from(fast_list_scheduled(&mo)) << 4);
    s
}

/// The step's host uniform (params + the first 32 host words).
pub fn host_uniforms(params: &AvbdParams, s: &[u32; 64]) -> HostUniforms {
    let mut packed = [[0u32; 4]; 8];
    for k in 0..32 {
        packed[k / 4][k % 4] = s[k];
    }
    HostUniforms {
        params: *params,
        s: packed,
    }
}

/// Counter block every step starts from (binding 7).
pub fn step_counters(i: &StepInputs) -> StepCounters {
    StepCounters {
        body_count: i.body_count,
        collider_count: i.collider_count,
        joint_count: i.joint_count,
        // Read by color_bodies_tail: rounds allowed this step.
        color_round_limit: COLOR_ROUND_LIMIT,
        ..StepCounters::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(colors: u32, rounds: u32) -> StepInputs {
        StepInputs {
            body_count: 9132,
            collider_count: 9132,
            joint_count: 0,
            joint_hash_slots: 65536,
            has_convex: false,
            reuse_contacts: true,
            max_colors: 32,
            iterations: 4,
            sleep_time: 1e6,
            sleep_island_rounds: 8,
            grid_hash_words: 131072,
            warm_hash_words: 65536,
            steps_done: 60,
            last: StepCounters {
                color_count: colors,
                color_rounds: rounds,
                ..StepCounters::default()
            },
            vehicle_count: 0,
            wheel_count: 0,
            motion: MotionInputs::default(),
        }
    }

    fn fast() -> StepModes {
        StepModes {
            compact_contacts: true,
            persistent_contacts: true,
            manifold_insert: true,
            copy_reuse: true,
            split_np: true,
            listed_persist: true,
            carry_rest: true,
            local_repair: true,
            grid_split: true,
            sparse_start: 32,
            color_headroom: Some(4),
            color_round_headroom: Some(2),
            tail_ok: true,
            ..StepModes::default()
        }
    }

    #[test]
    fn adaptive_counts_keep_completeness_tails() {
        let cmds = encode_step(&fast(), &inputs(10, 0), false, false);
        let primal = cmds
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    Cmd::Indirect {
                        kernel: "solve_primal_color_compact",
                        ..
                    }
                )
            })
            .count();
        assert_eq!(primal, 4 * 14, "colours used + headroom per iteration");
        let tails = cmds
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    Cmd::Dispatch {
                        kernel: "solve_primal_sparse_compact",
                        pass: 14,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            tails, 4,
            "every iteration ends with the sparse tail from the estimate"
        );
        assert!(cmds.iter().any(|c| matches!(
            c,
            Cmd::Dispatch {
                kernel: "color_bodies_tail",
                pass: 3,
                ..
            }
        )));
        assert!(!cmds.iter().any(|c| matches!(
            c,
            Cmd::Indirect {
                kernel: "sleep_island_gate",
                ..
            }
        )));
    }

    #[test]
    fn full_controls_dispatch_the_whole_palette() {
        let m = StepModes {
            color_headroom: None,
            color_round_headroom: None,
            ..fast()
        };
        let cmds = encode_step(&m, &inputs(10, 0), false, false);
        assert!(!cmds.iter().any(|c| matches!(
            c,
            Cmd::Dispatch {
                kernel: "solve_primal_sparse_compact",
                ..
            }
        )));
        assert!(!cmds.iter().any(|c| matches!(
            c,
            Cmd::Dispatch {
                kernel: "color_bodies_tail",
                ..
            }
        )));
        let rounds = cmds
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    Cmd::Indirect {
                        kernel: "color_bodies",
                        ..
                    }
                )
            })
            .count();
        assert_eq!(rounds, 32);
    }

    #[test]
    fn persistent_entries_cover_every_planned_kernel() {
        let mut m = fast();
        for split in [false, true] {
            for listed in [false, true] {
                m.split_np = split;
                m.listed_persist = listed;
                let mut i = inputs(10, 0);
                i.joint_count = 3;
                i.has_convex = true;
                for whole in [false, true] {
                    for c in encode_step(&m, &i, whole, true) {
                        let k = match c {
                            Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => kernel,
                            Cmd::Copy { .. } | Cmd::Phase { .. } => continue,
                        };
                        let e = resolve_entry(k, true);
                        assert!(PERSISTENT_ENTRIES.contains(&e), "{k} -> {e} missing");
                    }
                }
            }
        }
    }

    #[test]
    fn pipelined_steps_forward_counts_before_the_reset() {
        let cmds = encode_step(&fast(), &inputs(10, 0), false, true);
        assert!(matches!(
            cmds[0],
            Cmd::Copy {
                src: COUNTERS_BINDING,
                src_offset: 8,
                ..
            }
        ));
        assert!(matches!(
            cmds[1],
            Cmd::Copy {
                src: COUNTERS_BINDING,
                src_offset: 48,
                ..
            }
        ));
        assert!(matches!(
            cmds[2],
            Cmd::Copy {
                src: COUNTER_UPLOAD,
                dst: COUNTERS_BINDING,
                ..
            }
        ));
    }

    #[test]
    fn motion_passes_are_scheduled_only_when_needed() {
        let names = |i: &StepInputs| -> Vec<&'static str> {
            encode_step(&fast(), i, false, false)
                .into_iter()
                .filter_map(|c| match c {
                    Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => Some(kernel),
                    _ => None,
                })
                .collect()
        };
        let plain = names(&inputs(10, 0));
        for k in ["apply_impulses", "ccd_register", "broadphase_fast", "ccd_toi", "ccd_pick", "ccd_emit"] {
            assert!(!plain.contains(&k), "{k} scheduled without CCD");
        }
        let mut i = inputs(10, 0);
        i.motion.ccd = true;
        i.motion.impulses = true;
        let ccd = names(&i);
        let at = |k: &str| ccd.iter().position(|n| *n == k).unwrap_or_else(|| panic!("{k} missing"));
        assert!(at("apply_impulses") < at("build_aabbs"));
        assert!(at("broadphase_clear") < at("ccd_register") && at("ccd_register") < at("broadphase_insert"));
        assert!(at("broadphase_query") < at("broadphase_fast"));
        assert!(at("broadphase_fast") < at("ccd_toi") && at("ccd_toi") < at("ccd_pick"));
        let np = ccd.iter().position(|n| n.starts_with("narrowphase")).unwrap();
        assert!(at("ccd_pick") < np);
        let emit = ccd.iter().position(|n| n.starts_with("ccd_emit")).unwrap();
        assert!(np < emit && emit < at("predict_bodies"));
    }
}

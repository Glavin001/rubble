//! One rigid-body step graph for every host (ENGINE_AUDIT P0).
//!
//! `encode_step` turns a step's configuration and the previous step's counters
//! into an ordered command list; the native wgpu host, the Metal host and (via
//! wasm) the browser replay the same list, so a measured win lands on all of
//! them at once. Kernel names are logical (canonical) entry points; hosts map
//! them to the compiled variant with [`resolve_entry`].

use engine_schema::{AvbdParams, StepCounters, WorldCapacity};

/// Jones-Plassmann rounds a step may dispatch (more run in the tail).
pub const COLOR_ROUND_LIMIT: u32 = 32;
/// Rounds `color_bodies_tail` may run in all (the dispatched ones included). It stops as
/// soon as every body is coloured, so the limit costs nothing until a dense pile (a
/// razed ship's wreck) has longer chains of blocking neighbours than the dispatched
/// rounds; past it the step reports `OVERFLOW_COLORS`.
pub const COLOR_TAIL_ROUND_LIMIT: u32 = 1024;
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
    /// Sleep archive blocks (`world_step.slang` archiveMeta, host word 30 bit 5; 0: off). Canonical
    /// contacts only: a pair that falls asleep keeps its rows for its wake.
    pub sleep_archive_blocks: u32,
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
    /// Append `pose_pack` after the sleep commit: compact render poses and the
    /// list of bodies that moved (bindings 34-36, after BIND_COUNT; read-only on physics state).
    pub pose_output: bool,
    /// Append `contact_events` after contact persistence: list touching pairs
    /// of event-flagged colliders (read-only on physics state).
    pub contact_events: bool,
    /// Some body has motion locks: `lock_predict` after `predict_bodies` and
    /// `lock_project` after the solve.
    pub motion_locks: bool,
    /// Awake-indexed broadphase (hierarchical grids only): static and sleeping
    /// colliders stay in the persistent resting table, and only the colliders of
    /// moving bodies are bounded, inserted and queried (`list_moving_colliders`,
    /// world_step.slang's rest tables). The same candidate pairs as the full pass.
    pub awake_broadphase: bool,
    /// With `awake_broadphase`, on a uniform grid (no hierarchical levels): the split
    /// grid's other table kept across steps as the resting table, its awake table filled
    /// from the moving colliders (world_step.slang restUniform). Only with the split grid,
    /// above the pairwise path's 512 colliders and without kinematic bodies (they sit in
    /// the other table while they move); otherwise the full pass runs.
    pub rest_uniform: bool,
    /// With `awake_broadphase`: after the sleep commit, the colliders that came to rest
    /// this step go into the resting table (rest_sleepers_mark .. rest_push) instead of
    /// a rebuild the next step after every sleep event; a rebuild only when one cannot
    /// be placed (world_step.slang restIncremental).
    pub rest_incremental: bool,
    /// `ccd_wake` a thread per collider (ccd_step.slang ccd_wake_colliders), only while
    /// a CCD body sweeps fast (work slot 87); WGSL hosts. `ccd_unhold` stays per body.
    pub ccd_by_collider: bool,
    /// `predict_bodies` and `extract_velocity_sleep` a thread per entry of this step's
    /// awake list (their PASS_U = 1 mode, work slot 67) instead of a thread per body
    /// slot, where no body changes outside that list between `build_awake_list` and
    /// them ([`awake_body_passes`]): the bodies they change are exactly the listed ones.
    pub awake_body_passes: bool,
    /// The primal's headroom colours (from last step's colour count to the sparse
    /// tail's first) solved in the sparse tail's single workgroup with the colour
    /// dispatches' lanes (`solve_primal_sparse_tail`): one dispatch an iteration instead
    /// of one per headroom colour, with each body's update as its colour's dispatch
    /// gives it. WGSL hosts.
    pub primal_tail: bool,
    /// Every solver iteration in one workgroup while few bodies are awake
    /// (`solve_primal_sparse_loop`, [`solve_loop`]): one dispatch for what is otherwise a
    /// dispatch per colour and a dual per iteration, with the same updates. WGSL hosts.
    pub solve_loop: bool,
    /// With `awake_broadphase`: rebuild the resting table this step (the host wrote
    /// bodies or colliders, or the last step did not keep the table). Steps after
    /// one where bodies fell asleep rebuild it on the GPU (host word 5).
    pub rest_rebuild: bool,
    /// With `awake_broadphase`: bound every collider this step (`build_aabbs`), not
    /// only the moving ones (a host write may have moved a static).
    pub full_bounds: bool,
    /// Size `broadphase_fast` from the fast list's count (`write_fast_args`, work
    /// slot 82) instead of dispatching all FAST_CAP workgroups (the same groups do
    /// the work; the rest returned at once).
    pub fast_indirect: bool,
    /// Sleep archive: a woken pair starts cold instead of from its archived row (host
    /// word 30 bit 6); with the touch query no rows are archived then
    /// ([`archive_rows`]).
    pub archive_cold: bool,
    /// Measurement variant (sleep archive): `archive_spread` rounds before `archive_emit`,
    /// each listing the sleepers resting on listed ones (0: off); with the touch query,
    /// that many more `archive_touch_spread` hops than its one.
    pub archive_spread_rounds: u32,
    /// Sleep archive: a sleeper touched by a moving body forms all its resting pairs
    /// before this step's narrowphase (`archive_touch`, host word 30 bit 8), and its
    /// pairs at rest keep their archived rows; off: `archive_emit` emits the pairs
    /// filed in the touched sleeper's own archive (the earlier path, for A/B runs).
    pub archive_touch_query: bool,
    /// Sleep archive laid out without rows (host word 30 bit 9, world_step.slang
    /// archiveCompact): only the touch query's per-body records, for an archive that
    /// never keeps rows (cold wakes with the touch query).
    pub archive_compact: bool,
}

/// Scene sizes and last step's counters this step is planned from.
#[derive(Clone, Copy, Debug)]
pub struct StepInputs {
    pub body_count: u32,
    pub collider_count: u32,
    pub joint_count: u32,
    pub joint_hash_slots: u32,
    pub has_convex: bool,
    /// Heightfield colliders exist: their pairs take the convex stages
    /// (`convex_classify`) without the convex-only scheduling choices.
    pub has_height_field: bool,
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
    /// Soft bodies in use (`soft_step.slang`; zero bodies = no soft passes).
    pub soft: SoftCounts,
    /// Host commands in binding 8 (`apply_commands` runs first when non-zero).
    pub command_count: u32,
    /// Directed debris (`guide_step.slang`, bindings 55-59) runs before
    /// prediction. Off: the passes are not encoded.
    pub guidance: bool,
    /// Plan slots per step (`GuidanceCapacity::plan_budget`).
    pub guide_slots: u32,
    /// Some heightfield carries a soil (`engine_schema::soil`, vehicle ruts):
    /// plan `wheel_soil` and `wheel_soil_press` after the vehicle passes.
    /// Off, the step graph is unchanged.
    pub soil: bool,
    /// Queued explosions (`explode` runs after the commands when non-zero).
    pub explosion_count: u32,
    /// `FLAG_POST_STAB`: record the velocity after the iterations, then one
    /// primal pass with α = 0 (pass `STAB_PASS + colour`).
    pub post_stabilize: bool,
}

/// Largest particle colour `soft_primal_fused` (one 256-lane workgroup) takes;
/// bigger colours get one dispatch each.
pub const SOFT_FUSED_MAX_COLOR: u32 = 256;

/// Dispatch sizes of the soft-body passes (`soft_step.slang`), from the host's
/// `engine_softbody::SoftHost`. Particle colours are computed at spawn, so the
/// host knows every colour's size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftCounts {
    pub bodies: u32,
    pub beams: u32,
    /// Particle colours (the largest vehicle's colour count).
    pub colors: u32,
    /// Particles in the largest colour (threads per `soft_primal` dispatch).
    pub max_color: u32,
    /// Break-group flag slots (groups + 1).
    pub groups: u32,
    /// Solved particles (colour-list entries).
    pub particles: u32,
    /// Chebyshev acceleration enabled.
    pub chebyshev: bool,
    /// Every particle (fixed included).
    pub particles_total: u32,
    pub triangles: u32,
    /// Node–triangle / triangle–obstacle contact passes.
    pub contacts: bool,
    /// Some beam still ramps its penalty: run `soft_dual`.
    /// Otherwise the pass would change nothing and is skipped.
    pub dual: bool,
    /// Particle sweeps per solver iteration (0 = 1). Extra sweeps refine the
    /// soft bodies between the rigid passes, which couple to them only
    /// through the core joints.
    pub sweeps: u32,
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
pub const FAST_GROUPS: u64 = engine_schema::FAST_CAP as u64;

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
        Some(h) if i.steps_done > 0 => (i.last.color_count + h).max(1),
        _ => max_c,
    };
    estimate.min(m.sparse_start).min(max_c)
}

/// Colours `(p, q)` of the primal tail ([`StepModes::primal_tail`]): `p` last step's
/// colour count, `q` the first colour today's plan leaves to the sparse tail (all
/// colours from `q` are solved there with 8 lanes a body, as before), when the headroom
/// between them is not empty and no wider than [`TAIL_HEADROOM_MAX`].
pub fn primal_tail(m: &StepModes, i: &StepInputs, max_c: u32, sparse_start: u32) -> Option<(u32, u32)> {
    if !m.primal_tail || m.color_headroom.is_none() || i.steps_done == 0 {
        return None;
    }
    let q = max_c.min(sparse_start);
    let p = i.last.color_count.min(q);
    (q > p && q - p <= TAIL_HEADROOM_MAX).then_some((p, q))
}

/// Whether the step's solver iterations run in one workgroup ([`StepModes::solve_loop`]):
/// while few bodies were awake last step (the loop is one workgroup), without soft bodies
/// or vehicles (their passes run between the iterations) and for at most
/// [`LOOP_ITERATIONS_MAX`] iterations.
pub fn solve_loop(m: &StepModes, i: &StepInputs) -> bool {
    m.solve_loop
        && i.steps_done > 0
        && i.soft.bodies == 0
        && i.vehicle_count == 0
        && (1..=LOOP_ITERATIONS_MAX).contains(&i.iterations)
        && i.last.awake_count <= LOOP_AWAKE_MAX
        && i.last.contact_count.max(i.joint_count) <= LOOP_CONTACTS_MAX
}

/// The solve loop's limits: awake bodies and contacts (or joints) last step, iterations.
pub const LOOP_AWAKE_MAX: u32 = 256;
pub const LOOP_CONTACTS_MAX: u32 = 4096;
pub const LOOP_ITERATIONS_MAX: u32 = 32;
/// First pass-table slot of the solve loop: `q * (LOOP_ITERATIONS_MAX + 1) + iterations`
/// for q (the first sparse colour) in 0..=32.
pub const LOOP_PASS: u32 = TAIL_PASS + 2 * TAIL_SLOTS;
const LOOP_SLOTS: u32 = 33 * (LOOP_ITERATIONS_MAX + 1);

/// Pass-table slot of the solve loop.
pub fn loop_pass(q: u32, iterations: u32) -> u32 {
    LOOP_PASS + q * (LOOP_ITERATIONS_MAX + 1) + iterations
}

/// Widest headroom the tail's pass table covers.
pub const TAIL_HEADROOM_MAX: u32 = 8;
/// First pass-table slot of the primal tail: `p * (TAIL_HEADROOM_MAX + 1) + (q - p)`
/// for p in 0..=32, then the same again with the stabilization flag.
pub const TAIL_PASS: u32 = 160;
const TAIL_SLOTS: u32 = 33 * (TAIL_HEADROOM_MAX + 1);

/// Pass-table slot of the tail over colours [p, q) (colour lanes), then sparse.
pub fn tail_pass(p: u32, q: u32, stab: bool) -> u32 {
    TAIL_PASS + u32::from(stab) * TAIL_SLOTS + p * (TAIL_HEADROOM_MAX + 1) + (q - p)
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
        "solve_primal_sparse_tail_compact" => "solve_primal_sparse_tail_persistent",
        "solve_primal_sparse_loop_compact" => "solve_primal_sparse_loop_persistent",
        "solve_primal_one_compact" => "solve_primal_one_persistent",
        "solve_dual_compact" => "solve_dual_persistent",
        "convex_classify" => "convex_classify_persistent",
        "convex_manifold" => "convex_manifold_persistent",
        "ccd_emit" => "ccd_emit_persistent",
        "reuse_all_manifolds" => "reuse_all_manifolds_persistent",
        "contact_events" => "contact_events_persistent",
        // The one vehicle kernel that reads contacts (vehicle_step_persistent.slang).
        "vehicle_loads" => "vehicle_loads_persistent",
        // The sleep archive's passes over this step's contact rows.
        "archive_count" => "archive_count_persistent",
        "archive_scatter" => "archive_scatter_persistent",
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
            if awake_broadphase(m, i) {
                // Last step's sleep events: write_moved_args rebuilds the resting
                // table after any (a new sleeper is missing from it).
                e.0.push(Cmd::Copy {
                    src: COUNTERS_BINDING,
                    src_offset: 36,
                    dst: HOST_UNIFORM_BINDING,
                    dst_offset: host_word_offset(5),
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

    if i.command_count > 0 {
        // Host commands land before anything reads body state this step.
        e.d("apply_commands", u64::from(i.command_count));
    }
    if i.explosion_count > 0 {
        e.d("explode", n_bodies);
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
    let soft = i.soft.bodies > 0;
    if soft {
        // A soft body sleeps and wakes as a whole.
        e.d("soft_wake", u64::from(i.soft.bodies) * 64);
    }
    let rest = awake_broadphase(m, i);
    if !reuse_manifolds && rest {
        // Moving colliders (slot 81), their bounds (or every collider's after a host
        // write), then the resting table: rebuilt after sleep events or host writes
        // (slots 79/80 are empty otherwise), its lists published, CCD lists reset.
        e.d("list_moving_colliders", n_cols);
        e.d("write_awake_collider_args", 64);
        if m.full_bounds {
            e.d("build_aabbs", n_cols);
        } else {
            e.w("build_aabbs_listed", 81);
        }
        if i.joint_count > 0 {
            e.d("joint_hash_clear", u64::from(i.joint_hash_slots));
            e.d("joint_hash_insert", n_joints);
        }
        // The uniform table with incremental inserts is rebuilt only when the host asks
        // (nothing on the GPU asks for it): otherwise these three would dispatch nothing.
        if !(m.rest_uniform && m.rest_incremental && !m.rest_rebuild) {
            e.d("write_moved_args", 64);
            e.w("rest_clear", 79);
            e.w(if m.rest_uniform { "rest_insert_uniform" } else { "rest_insert" }, 80);
        }
        if m.rest_uniform {
            e.d("rest_publish_uniform", 64);
            if mo.ccd {
                e.d("ccd_register", n_bodies);
            }
            e.w("broadphase_insert_awake_uniform", 81);
            e.w("broadphase_query_awake_uniform", 81);
            // Sleepers' pairs with the large dynamics: a walk from each (slot 88), or a
            // thread per collider for a very large one (slot 85); empty without any.
            e.d("write_sleeper_large_args", 64);
            e.w("broadphase_sleeper_large_walk", 88);
            e.w("broadphase_sleeper_large", 85);
        } else {
            e.d("rest_publish", 64);
            if mo.ccd {
                e.d("ccd_register", n_bodies);
            }
            e.w("broadphase_insert_awake", 81);
            e.w("broadphase_query_awake", 81);
            // Resting colliders finer than a coarse mover: the mover scans their cells.
            e.d("write_finer_args", 64);
            e.w("broadphase_query_finer", 77);
        }
    } else if !reuse_manifolds {
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
    }
    if !reuse_manifolds {
        if fast_list_scheduled(&mo) {
            // Sweeps longer than a cell query along their path (row 1).
            e.phase("broadphaseGrid");
            if m.fast_indirect {
                // One workgroup per listed fast collider (slot 82), not FAST_CAP.
                e.d("write_fast_args", 64);
                e.w("broadphase_fast", 82);
            } else {
                e.d("broadphase_fast", FAST_GROUPS * 64);
            }
            e.phase("ccdFast");
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
        if rest {
            // Every walker of this step's tables is done: empty W's heads.
            let clear = if m.rest_uniform {
                "broadphase_clear_awake_uniform"
            } else {
                "broadphase_clear_awake"
            };
            e.w(clear, 81);
        }
    }
    if m.sleep_archive_blocks > 0 && !reuse_manifolds {
        if archive_touch(m) {
            // Touched sleepers' resting pairs, queried (world_step.slang archive_touch),
            // after listing the sleepers those woken since the broadphase rest on.
            // One hop, and `archive_spread_rounds` more (measurement variant).
            // Work slots 83 (a thread a listed sleeper) and 84 (a thread a collider, or
            // none when nothing is listed), written by archive_touch_snap / _args.
            for _ in 0..=m.archive_spread_rounds {
                e.d("archive_touch_snap", 64);
                e.w("archive_touch_seed", 83);
                e.w("archive_touch_spread", 84);
            }
            e.d("archive_touch_args", 64);
            e.w("archive_touch", 84);
        } else {
            // Measurement variant: sleepers resting on touched ones, one archived hop a round.
            for _ in 0..m.archive_spread_rounds {
                e.d("archive_spread", 4096);
            }
            // Touched sleepers' archived resting pairs (world_step.slang archive_emit).
            e.d("archive_emit", 4096);
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
        e.w("ccd_pick_key", 64);
        e.w("ccd_pick", 64);
        // Sleepers in a fast CCD body's path wake before the narrowphase.
        if m.ccd_by_collider {
            e.d("ccd_reach_args", 64);
            // A CCD slot's colliders are among the listed movers on awake-indexed steps.
            if rest {
                e.w("ccd_slot_reach_listed", 81);
            } else {
                e.w("ccd_slot_reach", 87);
            }
            e.w("ccd_wake_colliders", 87);
        } else {
            e.d("ccd_wake", n_bodies);
        }
        e.d("ccd_mark", u64::from(CCD_BODY_CAP));
        e.phase("ccdToi");
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
        if i.has_convex || i.has_height_field {
            // Hull and heightfield pairs deferred by narrowphase: classify (cheap pairs emit
            // here), bucket hull pairs, size-bucketed cooperative SAT, then clipping.
            e.w("convex_classify", 64);
            e.w("convex_bucket", 64);
            e.d("convex_write_args", 64);
            e.w("convex_sat_small", 74);
            e.w("convex_sat_large", 75);
            e.w("convex_manifold", 76);
        }
        if mo.ccd {
            e.phase("narrowphaseMain");
            e.w(if m.copy_reuse { "ccd_emit_copy_persistent" } else { "ccd_emit" }, 64);
            e.phase("ccdEmit");
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
    if m.contact_events {
        e.w("contact_events", 66);
    }
    if i.joint_count > 0 {
        e.d("warmstart_joints", n_joints);
    }
    e.phase("warm");
    // With the awake bitmap (awake-list body passes): the per-body passes below that act
    // only on listed bodies read a bit per body before its record.
    let bits = awake_body_passes(m, i).1;
    if bits {
        e.d("clear_awake_bits", n_bodies.div_ceil(32));
        e.d("build_awake_list_bits", n_bodies);
    } else {
        e.d("build_awake_list", n_bodies);
    }
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
    // any further rounds in one workgroup up to COLOR_TAIL_ROUND_LIMIT, so
    // colouring finishes (8cl). It returns at once when the dispatched rounds
    // did, so the adaptive plan always has it (a pile may need more than
    // COLOR_ROUND_LIMIT rounds); full controls dispatch exactly the 32 rounds.
    // Pass value = round index (slots 0..47 hold i).
    let rounds = color_rounds_dispatched(m, &i.last);
    for round in 0..rounds {
        e.wp("color_bodies", 69, round);
    }
    if rounds < COLOR_ROUND_LIMIT || m.color_round_headroom.is_some() {
        e.dp("color_bodies_tail", 64, rounds);
    }
    e.phase("coloring");
    let max_c = i.max_colors.min(32);
    e.d(if bits { "count_color_tiles_bits" } else { "count_color_tiles" }, n_bodies);
    e.d("prefix_color_tiles_parallel", u64::from(max_c) * 64);
    e.d(if bits { "scatter_color_tiles_bits" } else { "scatter_color_tiles" }, n_bodies);
    // Per-colour max CSR degree for degree-aware primal lanes (s[20]).
    e.w("record_color_degrees", 67);
    e.d("write_color_args", 64);
    if mo.ccd && !reuse_manifolds {
        // Bodies a fast CCD body woke this step are solved this step (ccd_step.slang).
        // Per body: it reads a body's flags and searches colliders only for the few whose
        // wake is deferred: cheaper than a scan of every collider (one measured 0.1 ms
        // more on gables506k while the ball flies).
        e.d(if bits { "ccd_unhold_bits" } else { "ccd_unhold" }, n_bodies);
    }
    if i.guidance {
        // After the graph (free flight = no contacts or joints this step) and
        // before prediction, which carries the steered velocities.
        // guide_scan and the single-group passes use 256-lane groups, the plan
        // one 64-lane group per slot (hosts size groups as threads / 64).
        let blocks = n_bodies.div_ceil(256);
        e.d("guide_scan", blocks * 64);
        e.d("guide_select", 64);
        e.d("guide_rank", n_bodies);
        e.d("guide_plan", u64::from(i.guide_slots) * 64);
        e.d("guide_commit", 64);
        e.phase("guide");
    }
    let (predict_listed, extract_listed) = awake_body_passes(m, i);
    if predict_listed {
        e.wp("predict_bodies", 67, 1);
    } else {
        e.d("predict_bodies", n_bodies);
    }
    if m.motion_locks {
        e.d("lock_predict", n_bodies);
    }
    if mo.ccd && !reuse_manifolds {
        // Struck bodies' trust regions for this step (ccd_step.slang).
        e.d("ccd_trust", u64::from(engine_schema::CCD_PAIR_CAP));
    }
    if vehicles {
        e.d("vehicle_prepare", u64::from(i.vehicle_count));
    }
    if soft {
        let n = i
            .soft
            .beams
            .max(i.soft.groups)
            .max(i.soft.particles)
            .max(i.soft.particles_total);
        e.d("soft_begin", u64::from(n.max(1)));
        if i.soft.contacts {
            // Soft contacts at the initial guess, frozen for the step.
            e.d("soft_bounds", u64::from(i.soft.bodies) * 64);
            e.d("soft_pairs", u64::from(i.soft.bodies));
            e.d("soft_tri_planes", u64::from(i.soft.triangles.max(1)));
            // SOFT_NODE_LANES (8) lanes per node.
            e.d("soft_collide_nodes", u64::from(i.soft.particles.max(1)) * 8);
            e.d("soft_collide_tris", u64::from(i.soft.triangles.max(1)));
            e.d("soft_contact_lists", u64::from(i.soft.particles.max(1)));
        }
        // Rigid mode: promote / demote before the solve (soft_step.slang).
        e.d("soft_mode", u64::from(i.soft.bodies.max(1)));
        e.phase("soft_begin");
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
    // The headroom colours (past last step's count) and the sparse tail in one
    // workgroup (`solve_primal_sparse_tail`, the colour dispatches' lanes for the
    // headroom): colours [0, solo) dispatched one by one, the rest in the tail.
    let tail = primal_tail(m, i, max_c, sparse_start);
    let solo = tail.map_or(max_c.min(sparse_start), |(p, _)| p);
    let tail_kernel = if m.compact_contacts {
        "solve_primal_sparse_tail_compact"
    } else {
        "solve_primal_sparse_tail"
    };
    let looped = solve_loop(m, i);
    if looped {
        // Every iteration in one workgroup (few bodies awake): the same colours, lanes,
        // order and dual updates as the dispatches below.
        let kernel = if m.compact_contacts {
            "solve_primal_sparse_loop_compact"
        } else {
            "solve_primal_sparse_loop"
        };
        e.dp(kernel, 64, loop_pass(max_c.min(sparse_start), i.iterations));
        // Timed as one primal phase (the duals run inside it).
        e.phase("primal");
    }
    for iter in 0..if looped { 0 } else { i.iterations } {
        for color in 0..solo {
            e.wp(primal, color, color);
        }
        if let Some((p, q)) = tail {
            e.dp(tail_kernel, 64, tail_pass(p, q, false));
        } else if sparse_start < max_c {
            e.dp(sparse, 64, sparse_start);
        }
        if soft {
            e.phase("primal");
            let sweeps = i.soft.sweeps.max(1);
            for k in 0..sweeps {
                // Chebyshev pass index across every sweep of the step (ω has
                // converged long before the pass-parameter table runs out).
                let pass = (u32::from(iter) * sweeps + k).min(95);
                // One workgroup per soft body (`soft_primal_bodies`): every body's
                // sweep at once, its particles' beams and contacts over the lanes,
                // one dispatch at any vehicle count. On the iteration's last sweep,
                // workgroups past the bodies run vehicle_iterate (64 vehicles each)
                // beside it. Hosts size workgroups as threads / 64.
                let vehicle_groups = if vehicles && k + 1 == sweeps {
                    u64::from(i.vehicle_count).div_ceil(64)
                } else {
                    0
                };
                e.dp(
                    "soft_primal_bodies",
                    (u64::from(i.soft.bodies) + vehicle_groups) * 64,
                    pass,
                );
            }
            e.phase("soft_primal");
        } else {
            e.phase("primal");
        }
        e.wp(dual, 65, u32::from(iter + 1 == i.iterations));
        if soft && i.soft.dual {
            e.d("soft_dual", u64::from(i.soft.beams.max(1)));
        }
        if vehicles && !soft {
            e.d("vehicle_iterate", u64::from(i.vehicle_count));
        }
        e.phase("dual");
    }
    if i.post_stabilize {
        e.d("record_stab_velocity", n_bodies);
        for color in 0..solo {
            e.wp(primal, color, STAB_PASS + color);
        }
        if let Some((p, q)) = tail {
            e.dp(tail_kernel, 64, tail_pass(p, q, true));
        } else if sparse_start < max_c {
            e.dp(sparse, 64, STAB_PASS + sparse_start);
        }
        e.phase("stabilize");
    }
    if soft {
        // Plasticity and breaking at the converged state, then detacher groups.
        e.d("soft_plastic", u64::from(i.soft.beams.max(1)));
        e.d("soft_break_groups", u64::from(i.soft.beams.max(1)));
        e.phase("soft_plastic");
    }
    if mo.ccd && !reuse_manifolds {
        // Post-solve sweep check: no CCD body ends the step past a surface.
        e.phase("solveMain");
        e.d("ccd_validate", u64::from(engine_schema::CCD_PAIR_CAP));
        e.d("ccd_clamp", u64::from(CCD_BODY_CAP));
        e.phase("ccdValidate");
    }
    if m.motion_locks {
        e.d("lock_project", n_bodies);
    }
    if extract_listed {
        e.wp("extract_velocity_sleep", 67, 1);
    } else {
        e.d("extract_velocity_sleep", n_bodies);
    }
    if vehicles {
        // Mount loads from the final λ and solved velocities; overloaded parts
        // and wheels are severed by the next step's vehicle_input.
        e.d("vehicle_loads", u64::from(i.vehicle_count));
        e.d("vehicle_post", u64::from(i.vehicle_count));
        if i.soil {
            // Wheels on soil sink to their load's depth: their patches are
            // pressed into the heightfield for the next step's queries, and
            // their compaction resistance waits for its vehicle_prepare.
            e.d("wheel_soil", u64::from(i.wheel_count.max(1)));
            e.d(
                "wheel_soil_press",
                u64::from(engine_schema::vehicle::SOIL_LANES),
            );
        }
    }
    if soft {
        // BDF2 velocities and history (a no-op under backward Euler).
        e.d("soft_velocity", u64::from(i.soft.particles.max(1)));
        e.d("soft_sleep", u64::from(i.soft.bodies) * 64);
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
        if soft {
            // The gate may have revoked part of a soft body: revoke it whole.
            e.d("soft_sleep", u64::from(i.soft.bodies) * 64);
        }
    }
    e.w("sleep_island_commit", 67);
    if archive_rows(m) {
        // Archive what just fell asleep (awake list, then this step's rows).
        e.w("archive_free", 67);
        e.w("archive_count", 65);
        e.w("archive_alloc", 67);
        e.w("archive_scatter", 65);
        e.w("archive_seal", 67);
    }
    if mo.ccd && !reuse_manifolds {
        // CCD impacts on dynamic bodies: the momentum exchange (ccd_step.slang).
        e.phase("sleepMain");
        e.d("ccd_respond", u64::from(CCD_BODY_CAP));
        e.d("ccd_respond_apply", u64::from(CCD_BODY_CAP));
        e.phase("ccdRespond");
    }
    if rest && m.rest_incremental && !reuse_manifolds {
        // This step's movers that came to rest join the resting table (slot 81 still
        // lists them; slot 86 one thread per bucket they leave).
        e.w("rest_sleepers_mark", 81);
        e.d("write_rest_dirty_args", 64);
        e.w("rest_unlink", 86);
        e.w("rest_push", 81);
    }
    e.phase("sleep");
    if m.pose_output {
        // Reads the final body state; writes only the pose buffers (34-36).
        e.d("pose_pack", n_bodies);
        e.phase("pose");
    }
    e.0
}

/// Whether the sleep archive keeps rows (`archive_free` .. `archive_seal` after the
/// sleep pass): only when something reads them, a warm start (not
/// [`StepModes::archive_cold`]) or the earlier `archive_emit` path (which emits nothing
/// while the carry runs).
pub fn archive_rows(m: &StepModes) -> bool {
    m.sleep_archive_blocks > 0
        && !m.archive_compact
        && (!m.archive_cold || (!m.archive_touch_query && !m.carry_rest))
}

/// Whether the sleep archive's touch query runs ([`StepModes::archive_touch_query`]):
/// not with the resting-pair carry, whose carried pairs already include a touched
/// sleeper's.
pub fn archive_touch(m: &StepModes) -> bool {
    m.sleep_archive_blocks > 0 && m.archive_touch_query && !m.carry_rest
}

/// Whether `predict_bodies` / `extract_velocity_sleep` run over the awake list
/// ([`StepModes::awake_body_passes`]): not with kinematic bodies (prediction moves them,
/// and they are not listed) for prediction, and for both not with vehicles, characters,
/// soft bodies or directed debris, whose passes may wake or move bodies after the list
/// is built.
pub fn awake_body_passes(m: &StepModes, i: &StepInputs) -> (bool, bool) {
    let both = m.awake_body_passes
        && i.vehicle_count == 0
        && i.motion.characters == 0
        && i.soft.bodies == 0
        && !i.guidance;
    (both && !i.motion.kinematic, both)
}

/// Whether this step runs the awake-indexed broadphase ([`StepModes::awake_broadphase`],
/// which hosts set for hierarchical grids only): not with characters (they move
/// kinematic bodies without a velocity, which the moving-collider list cannot see)
/// or soft bodies.
pub fn awake_broadphase(m: &StepModes, i: &StepInputs) -> bool {
    m.awake_broadphase
        && i.motion.characters == 0
        && i.soft.bodies == 0
        && (!m.rest_uniform || (m.grid_split && !i.motion.kinematic && i.collider_count > 512))
}

/// Entries a persistent-contact host replays (every name `encode_step` can
/// emit, resolved with [`resolve_entry`]). Native hosts compile exactly these.
pub const PERSISTENT_ENTRIES: &[&str] = &[
    "apply_commands_persistent",
    "apply_impulses",
    "ccd_register",
    "broadphase_fast",
    "write_fast_args",
    "ccd_toi",
    "ccd_pick_key",
    "ccd_pick",
    "ccd_wake",
    "ccd_mark",
    "ccd_respond",
    "ccd_respond_apply",
    "ccd_trust",
    "ccd_unhold",
    "ccd_unhold_bits",
    "ccd_reach_args",
    "ccd_slot_reach",
    "ccd_slot_reach_listed",
    "ccd_wake_colliders",
    "ccd_validate",
    "ccd_clamp",
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
    "vehicle_loads_persistent",
    "vehicle_post",
    "wheel_soil",
    "wheel_soil_press",
    "build_aabbs",
    "joint_hash_clear",
    "joint_hash_insert",
    "broadphase_clear",
    "broadphase_insert",
    "broadphase_query",
    "list_moving_colliders",
    "write_awake_collider_args",
    "build_aabbs_listed",
    "write_moved_args",
    "rest_clear",
    "rest_insert",
    "rest_publish",
    "broadphase_insert_awake",
    "broadphase_query_awake",
    "write_finer_args",
    "broadphase_query_finer",
    "broadphase_clear_awake",
    "rest_insert_uniform",
    "rest_publish_uniform",
    "broadphase_insert_awake_uniform",
    "broadphase_query_awake_uniform",
    "write_sleeper_large_args",
    "broadphase_sleeper_large",
    "broadphase_sleeper_large_walk",
    "broadphase_clear_awake_uniform",
    "rest_sleepers_mark",
    "write_rest_dirty_args",
    "rest_unlink",
    "rest_push",
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
    "contact_events_persistent",
    "warmstart_joints",
    "build_awake_list",
    "clear_awake_bits",
    "build_awake_list_bits",
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
    "count_color_tiles_bits",
    "scatter_color_tiles_bits",
    "prefix_color_tiles_parallel",
    "scatter_color_tiles",
    "record_color_degrees",
    "write_color_args",
    "predict_bodies",
    "solve_primal_color_persistent",
    "solve_primal_sparse_persistent",
    "solve_primal_sparse_tail_persistent",
    "solve_primal_sparse_loop_persistent",
    "solve_dual_persistent",
    "record_stab_velocity",
    "extract_velocity_sleep",
    "write_sleep_args",
    "sleep_island_gate",
    "sleep_island_commit",
];

/// Character, query and projectile entries (`character_step.slang`).
pub const CHARACTER_ENTRIES: &[&str] = &["char_move", "char_post", "query_batch", "projectile_step"];
/// Soft-body entries (`soft_step.slang`).
pub const SOFT_ENTRIES: &[&str] = &[
    "soft_wake",
    "soft_begin",
    "soft_bounds",
    "soft_tri_planes",
    "soft_pairs",
    "soft_velocity",
    "soft_contact_lists",
    "soft_collide_nodes",
    "soft_collide_tris",
    "soft_primal",
    "soft_primal_fused",
    "soft_primal_bodies",
    "soft_chebyshev",
    "soft_mode",
    "soft_dual",
    "soft_plastic",
    "soft_break_groups",
    "soft_sleep",
];

/// Slang source (under `slang/kernels/`) that defines `entry`.
/// Vehicle controller entries (`vehicle_step.slang`).
pub const VEHICLE_ENTRIES: &[&str] = &[
    "vehicle_input",
    "wheel_query",
    "vehicle_link",
    "vehicle_prepare",
    "vehicle_iterate",
    "vehicle_loads",
    "vehicle_post",
    "raycast_batch",
    "wheel_soil",
    "wheel_soil_press",
];

pub fn kernel_source(entry: &str) -> &'static str {
    if entry == "vehicle_loads_persistent" {
        return "vehicle_step_persistent.slang";
    }
    if VEHICLE_ENTRIES.contains(&entry) {
        return "vehicle_step.slang";
    }
    if CHARACTER_ENTRIES.contains(&entry) {
        return "character_step.slang";
    }
    if SOFT_ENTRIES.contains(&entry) {
        return "soft_step.slang";
    }
    if entry == "pose_pack" {
        return "pose_pack.slang";
    }
    if engine_schema::guidance::GUIDE_ENTRIES.contains(&entry) {
        return "guide_step.slang";
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
    grid_cell_with(cols, bodies, hierarchical, 0.0)
}

/// [`grid_cell`], also sized for a moving collider of radius `movable_radius` the host's
/// mirror cannot see move (engine-schema `hierarchical_grid_with`).
pub fn grid_cell_with(
    cols: &[engine_schema::Collider],
    bodies: &[engine_schema::BodyState],
    hierarchical: bool,
    movable_radius: f32,
) -> (f32, u32) {
    if hierarchical {
        return engine_schema::convex::hierarchical_grid_with(cols, bodies, movable_radius);
    }
    let mut max_r = if movable_radius <= 2.0 { movable_radius.max(0.25) } else { 0.25f32 };
    for c in cols {
        if c.shape == engine_schema::ShapeType::Plane as u32 {
            continue;
        }
        let body = c.body as usize;
        if body >= bodies.len() || !bodies[body].moves() {
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
/// 31 the vehicle pool, 32 the soft-body arena, 33 the character / query /
/// projectile pool).
pub const BIND_COUNT: usize = 34;
/// Pass-uniform slot stride (binding 22, dynamic offsets).
pub const PASS_ALIGN: u64 = 256;
/// Pass-uniform slots (binding 22): see [`pass_table`].
pub const PASS_SLOTS: u64 = 160 + 2 * 33 * 9 + 33 * 33;
/// First post-stabilization slot: `STAB_PASS + c` holds `256 | c` (colour c at α = 0).
pub const STAB_PASS: u32 = 96;

/// The bound the pre-CSR region's tables are sized within (grid heads, warm hash):
/// the constraint ids rounded up to a power of two. The region itself
/// ([`constraint_id_words`]) holds only what those tables and the ids can reach.
pub fn constraint_id_cap(c: &WorldCapacity) -> u32 {
    let words = (2 * (c.max_contacts + c.max_joints))
        .max(32768)
        .next_power_of_two() as u64;
    words.max(131072) as u32
}

/// `gConstraintIds` words before the CSR index (grid heads, warm hash, ids): the most
/// any of them can use. Their sizes are those within [`constraint_id_cap`], so the
/// tables are the same; only the words none of them can reach are not allocated
/// (the ids' rounding to a power of two: 2 x (2^21 contacts + 64 joints) took 2^23
/// words, half of them unused).
pub fn constraint_id_words(c: &WorldCapacity) -> u32 {
    let cap = constraint_id_cap(c);
    let ids = 2 * (c.max_contacts + c.max_joints);
    // Split grids: two tables (the most a grid takes, split or not).
    let grid = 2 * grid_hash_words(c, c.max_colliders, true, false);
    let fixed = 2 * grid_hash_words(c, c.max_colliders, true, true);
    // The warm hash at the most contacts and bodies its callers pass (both clamped
    // to the capacity), heads or records.
    let heads = warm_hash_slots(c, c.max_bodies, c.max_contacts.max(256), false);
    let records = 4 * warm_hash_slots(c, c.max_bodies, c.max_contacts.max(256), true);
    ids.max(grid)
        .max(fixed)
        .max(heads)
        .max(records)
        .max(131072)
        .min(cap)
}

/// Broadphase grid heads: at least twice the colliders (three-avbd), so 506k
/// gables avoids ~4 colliders per bucket; small scenes keep 131072. `fixed`
/// keeps the old table (control); `split` halves the cap so both tables fit.
pub fn grid_hash_words(c: &WorldCapacity, colliders: u32, split: bool, fixed: bool) -> u32 {
    if fixed {
        return 131072;
    }
    let region = constraint_id_cap(c) / if split { 2 } else { 1 };
    let cap = 1u32 << (31 - region.leading_zeros());
    (2 * colliders).max(131072).next_power_of_two().min(cap)
}

/// Warm hash size in slots: one word per slot (heads), or four-word record
/// slots at ~2x manifolds, never below 4x bodies, within the pre-CSR region.
/// `bodies` and `prev_contacts` are at most the capacity's ([`constraint_id_words`]
/// holds the largest table).
pub fn warm_hash_slots(c: &WorldCapacity, bodies: u32, prev_contacts: u32, record: bool) -> u32 {
    let ids_len = constraint_id_cap(c);
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
    s[5] = b(
        c.max_joints,
        std::mem::size_of::<engine_schema::Joint>() as u64,
    );
    s[6] = 256;
    s[7] = 256;
    s[8] = b(c.max_commands, 64);
    s[9] = b(c.max_bodies + c.max_colliders, 32);
    s[10] = b(c.max_candidates, 16);
    s[11] = b(c.max_bodies, 4);
    s[12] = b(c.max_bodies, 4);
    s[13] = b(c.max_bodies + 1, 4);
    // The constraint ids, the CSR index, the awake bitmap (the awake-indexed
    // broadphase's tables are added once a step takes it).
    s[14] = (u64::from(constraint_id_words(c)) + u64::from(c.max_bodies) + 1 + awake_bits_words(c)) * 4;
    s[15] = b(c.max_bodies * (c.max_colors + 2), 4);
    s[16] = b(c.max_colors * c.max_bodies.div_ceil(64).max(1), 4);
    s[17] = b(c.max_colors.max(128), 4);
    s[18] = b(c.max_hull_vertices, 16);
    s[19] = u64::from(engine_schema::joint_hash_size(c.max_joints)) * 3 * 4;
    s[20] = b(c.max_bodies, 4);
    // Scratch: host-reset words 0..63, uniform-grid large list 64..79, the
    // oversized-static count (80), then the event ring and the static list.
    s[21] = u64::from(engine_schema::SCRATCH_WORDS) * 4;
    s[22] = PASS_SLOTS * PASS_ALIGN;
    // Colour slots 0..63, work 64..73 (persist list 70/71, carry 72, previous
    // contacts 73), convex 74..76, the awake-indexed broadphase's 77..81 (finer
    // scan, moved list, resting-table rebuild, moving colliders), the fast list's 82, the
    // sleep archive's touch query 83 and 84, the uniform rest tables' sleeper scan 85,
    // the incremental rest inserts' unlink 86, the CCD wake by collider 87, the
    // sleepers' walk from the large dynamics 88.
    s[23] = b(c.max_colors.max(90) * 3, 4);
    s[24] = b(c.max_colliders, 80);
    s[25] = b(c.max_candidates, 32);
    s[26] = b(c.max_bodies, 32);
    s[27] = b(c.max_contacts, 64);
    s[28] = b(c.max_candidates, 32);
    s[29] = b(c.max_candidates, 32);
    // SAT separating-feature cache (convex_step gSatCache): uint2 slots, power of two.
    s[30] = u64::from(c.max_candidates.max(1024).next_power_of_two()) * 8;
    // Vehicle pool (vehicle_step gVehicleData) and soft-body arena (soft_step
    // gSoftData, 32): hosts with vehicles or soft bodies resize them.
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
/// constants at 48..67, zero to 95, and `256 | colour` from [`STAB_PASS`] (the
/// post-stabilization primal pass).
pub fn pass_table() -> Vec<u8> {
    let mut pass = vec![0u8; (PASS_SLOTS * PASS_ALIGN) as usize];
    for i in 0..PASS_SLOTS as u32 {
        let v = if i < 48 {
            i
        } else if i < 68 {
            let k = i - 48;
            (1u32 << k) | ((k & 1) << 31)
        } else if i >= LOOP_PASS {
            // The solve loop: first sparse colour, iterations.
            let k = i - LOOP_PASS;
            (k / (LOOP_ITERATIONS_MAX + 1)) | ((k % (LOOP_ITERATIONS_MAX + 1)) << 8)
        } else if i >= TAIL_PASS {
            // The primal tail: first colour, stabilization flag, first sparse colour.
            let k = i - TAIL_PASS;
            let (stab, r) = (k / TAIL_SLOTS, k % TAIL_SLOTS);
            let (p, d) = (r / (TAIL_HEADROOM_MAX + 1), r % (TAIL_HEADROOM_MAX + 1));
            p | (stab << 8) | ((p + d) << 16)
        } else if i >= STAB_PASS {
            256 | (i - STAB_PASS)
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
    s[0] = i.command_count;
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
    // Bit 5: the sleep archive (world_step.slang sleepArchive).
    s[30] |= u32::from(m.sleep_archive_blocks > 0) << 5;
    // Bit 6: the archive's cold-start variant (world_step.slang archiveCold).
    s[30] |= u32::from(m.archive_cold) << 6;
    // Bit 8: the archive's touch query (world_step.slang archiveTouchQuery).
    s[30] |= u32::from(archive_touch(m)) << 8;
    // Bit 9: the archive's compact layout (world_step.slang archiveCompact).
    s[30] |= u32::from(m.sleep_archive_blocks > 0 && m.archive_compact) << 9;
    if awake_broadphase(m, i) {
        // Rest tables (bit 0); bit 1 rebuilds them this step. Word 5: last step's
        // sleep events (a pipelined step copies them on the GPU), in the uniform
        // only: the scratch block's word 5 is the finer-scan count, which hosts
        // start at zero (the awake broadphase runs on WGSL hosts, whose HOST_S
        // reads the uniform).
        // Bit 4: the uniform grid's rest tables (world_step.slang restUniform).
        // Bit 5: incremental inserts (world_step.slang restIncremental).
        s[25] = 1
            | (u32::from(m.rest_rebuild) << 1)
            | (u32::from(m.rest_uniform) << 4)
            | (u32::from(m.rest_incremental) << 5);
        s[5] = i.last.sleep_events;
    }
    s
}

/// Words of binding 14 past the CSR index every host allocates: the awake bitmap
/// (world_step.slang awakeBitsBase, a word per 32 body slots).
pub fn awake_bits_words(c: &WorldCapacity) -> u64 {
    u64::from(c.max_bodies).div_ceil(32)
}

/// Words past the awake bitmap that the awake-indexed broadphase uses with up to
/// `colliders` colliders on a grid of `split` / `fixed` layout (world_step.slang
/// restBase): a 32-word header, the resting and awake heads, a resting link per
/// collider, the awake-collider list, and per collider the uniform tables' awake stamp,
/// the resting bucket and the incremental inserts' bucket list (restStampBase,
/// restAtBase, restDirtyBase). Hosts grow binding 14 for them only once a step takes
/// that broadphase (`WgpuSim::ensure_rest_tables`).
pub fn rest_table_words(c: &WorldCapacity, colliders: u32, split: bool, fixed: bool) -> u64 {
    let grid = u64::from(grid_hash_words(c, colliders, split, fixed));
    32 + 2 * grid + 5 * u64::from(colliders)
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
        color_round_limit: COLOR_TAIL_ROUND_LIMIT,
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
            has_height_field: false,
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
            soft: SoftCounts::default(),
            command_count: 0,
            guidance: false,
            guide_slots: 0,
            soil: false,
            explosion_count: 0,
            post_stabilize: false,
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
            pose_output: false,
            ..StepModes::default()
        }
    }

    /// The pre-CSR region holds every table its users can size (the grid, split or not,
    /// fixed or not; the warm hash as heads or records at any contact and body count up
    /// to the capacity; the ids), and those sizes are the ones within the power-of-two
    /// cap, so a smaller region changes no table.
    #[test]
    fn constraint_id_region_holds_every_table() {
        let caps = [
            // ring110k, gables506k (demo-config.mjs sceneCap), small worlds, joints.
            (131072, 262144, 2097152, 64),
            (524288, 1048576, 6 * 1048576, 64),
            (4096, 8192, 32768, 64),
            (65536, 65536, 1048576, 10000),
            (1024, 1024, 4096, 1),
            (262144, 524288, 524288, 0),
        ];
        for (mb, mc, mcon, mj) in caps {
            let c = WorldCapacity {
                max_bodies: mb,
                max_colliders: mc,
                max_contacts: mcon,
                max_joints: mj,
                ..WorldCapacity::small()
            };
            let words = constraint_id_words(&c);
            let cap = constraint_id_cap(&c);
            assert!(words <= cap && words >= 2 * (mcon + mj), "{words} {cap}");
            for colliders in [0, mc / 3, mc] {
                for split in [false, true] {
                    for fixed in [false, true] {
                        let g = grid_hash_words(&c, colliders, split, fixed);
                        assert!(g.is_power_of_two());
                        let tables = if split { 2 } else { 1 };
                        // The fixed control's split tables overlapped small regions before.
                        if !(fixed && split && 2 * g > cap) {
                            assert!(tables * g <= words, "grid {g} x{tables} > {words}");
                        }
                    }
                }
            }
            for bodies in [0, mb / 2, mb] {
                for prev in [256, mcon / 2, mcon.max(256)] {
                    let h = warm_hash_slots(&c, bodies, prev, false);
                    let r = warm_hash_slots(&c, bodies, prev, true);
                    assert!(h.is_power_of_two() && r.is_power_of_two());
                    assert!(h <= words && 4 * r <= words, "warm {h} / {r} records > {words}");
                }
            }
        }
        // The rounding no table reaches is not allocated: ring110k's region was 2^23 words.
        let ring = WorldCapacity {
            max_bodies: 131072,
            max_colliders: 262144,
            max_contacts: 2097152,
            max_joints: 64,
            ..WorldCapacity::small()
        };
        assert_eq!(constraint_id_cap(&ring), 1 << 23);
        assert_eq!(constraint_id_words(&ring), 2 * (2097152 + 64));
    }

    #[test]
    fn awake_broadphase_runs_over_moving_colliders_only() {
        let names = |m: &StepModes, i: &StepInputs| -> Vec<&'static str> {
            encode_step(m, i, false, true)
                .iter()
                .filter_map(|c| match *c {
                    Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => Some(kernel),
                    _ => None,
                })
                .collect()
        };
        let mut i = inputs(10, 0);
        i.motion.queries = 1;
        let m = StepModes {
            awake_broadphase: true,
            ..fast()
        };
        let k = names(&m, &i);
        let at = |n: &str| {
            k.iter()
                .position(|x| *x == n)
                .unwrap_or_else(|| panic!("{n} missing"))
        };
        // No pass over every collider but the cheap moving-collider scan.
        for full in ["build_aabbs", "broadphase_clear", "broadphase_insert", "broadphase_query"] {
            assert!(!k.contains(&full), "{full} still runs");
        }
        assert!(at("list_moving_colliders") < at("build_aabbs_listed"));
        assert!(at("build_aabbs_listed") < at("rest_insert"));
        assert!(at("rest_publish") < at("broadphase_insert_awake"));
        assert!(at("broadphase_query_awake") < at("broadphase_query_finer"));
        // W is emptied after every walker of the step's tables, before narrowphase.
        assert!(at("query_batch") < at("broadphase_clear_awake"));
        assert!(at("broadphase_clear_awake") < at("narrowphase_defer_copy_persistent"));
        // A host write bounds every collider; characters keep the full broadphase.
        let full_bounds = StepModes {
            full_bounds: true,
            ..m
        };
        assert!(names(&full_bounds, &i).contains(&"build_aabbs"));
        i.motion.characters = 1;
        assert!(names(&m, &i).contains(&"broadphase_query"));
        // The pipelined step forwards last step's sleep events to host word 5.
        i.motion.characters = 0;
        assert!(encode_step(&m, &i, false, true).iter().any(|c| matches!(
            c,
            Cmd::Copy { src_offset: 36, dst_offset, .. } if *dst_offset == host_word_offset(5)
        )));
    }

    #[test]
    fn primal_tail_takes_the_headroom_colours_in_one_dispatch() {
        let mut m = fast();
        m.primal_tail = true;
        let cmds = encode_step(&m, &inputs(10, 0), false, false);
        let colors = cmds
            .iter()
            .filter(|c| matches!(c, Cmd::Indirect { kernel: "solve_primal_color_compact", .. }))
            .count();
        assert_eq!(colors, 4 * 10, "last step's colours one by one");
        let tails: Vec<u32> = cmds
            .iter()
            .filter_map(|c| match c {
                Cmd::Dispatch { kernel: "solve_primal_sparse_tail_compact", pass, threads } => {
                    assert_eq!(*threads, 64, "one workgroup");
                    Some(*pass)
                }
                _ => None,
            })
            .collect();
        assert_eq!(tails, vec![tail_pass(10, 14, false); 4], "one tail an iteration");
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, Cmd::Dispatch { kernel: "solve_primal_sparse_compact", .. })));
        // The table: first colour, stabilization flag, first sparse colour.
        let table = pass_table();
        let at = |slot: u32| {
            let o = (slot as u64 * PASS_ALIGN) as usize;
            u32::from_le_bytes(table[o..o + 4].try_into().unwrap())
        };
        assert_eq!(at(tail_pass(10, 14, false)), 10 | (14 << 16));
        assert_eq!(at(tail_pass(32, 32 + TAIL_HEADROOM_MAX, true)), 32 | 256 | ((32 + TAIL_HEADROOM_MAX) << 16));
        assert_eq!(at(STAB_PASS + 63), 256 | 63, "the stabilization slots below the tail's");
        assert!(u64::from(tail_pass(32, 32 + TAIL_HEADROOM_MAX, true)) < PASS_SLOTS);
        // The first step, and no headroom left: as before.
        let mut i = inputs(10, 0);
        i.steps_done = 0;
        assert!(primal_tail(&m, &i, 32, primal_colors_dispatched(&m, &i, 32)).is_none());
        assert!(primal_tail(&m, &inputs(32, 0), 32, 32).is_none());
    }

    #[test]
    fn solve_loop_takes_every_iteration_in_one_dispatch() {
        let mut m = fast();
        m.solve_loop = true;
        let mut i = inputs(10, 0);
        i.last.awake_count = 40;
        i.last.contact_count = 300;
        let cmds = encode_step(&m, &i, false, false);
        let count = |k: &str| cmds.iter().filter(|c| matches!(c, Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } if *kernel == k)).count();
        assert_eq!(count("solve_primal_color_compact"), 0);
        assert_eq!(count("solve_dual_compact"), 0);
        assert_eq!(count("solve_primal_sparse_compact"), 0);
        let pass = cmds.iter().find_map(|c| match c {
            Cmd::Dispatch { kernel: "solve_primal_sparse_loop_compact", pass, threads } => {
                assert_eq!(*threads, 64);
                Some(*pass)
            }
            _ => None,
        });
        assert_eq!(pass, Some(loop_pass(14, 4)));
        let table = pass_table();
        let o = (loop_pass(14, 4) as u64 * PASS_ALIGN) as usize;
        assert_eq!(u32::from_le_bytes(table[o..o + 4].try_into().unwrap()), 14 | (4 << 8));
        assert!(u64::from(loop_pass(32, LOOP_ITERATIONS_MAX)) < PASS_SLOTS);
        // Many awake: the dispatches as before.
        i.last.awake_count = 5000;
        assert!(!solve_loop(&m, &i));
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
                for awake in [false, true] {
                    m.split_np = split;
                    m.listed_persist = listed;
                    // World's awake-indexed modes: rest tables (both grids, incremental),
                    // CCD by collider, awake-list body passes.
                    m.awake_broadphase = awake;
                    m.rest_uniform = split;
                    m.rest_incremental = listed;
                    m.ccd_by_collider = awake;
                    m.awake_body_passes = awake;
                    m.primal_tail = awake;
                    m.solve_loop = awake;
                    let mut i = inputs(10, 0);
                    i.joint_count = 3;
                    i.has_convex = true;
                    i.collider_count = 1024;
                    i.motion.ccd = true;
                    m.contact_events = true;
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
    }

    #[test]
    fn soft_passes_are_planned_in_order() {
        let mut i = inputs(10, 0);
        i.soft = SoftCounts {
            bodies: 1,
            beams: 100,
            colors: 5,
            max_color: 30,
            groups: 1,
            particles: 60,
            chebyshev: true,
            particles_total: 60,
            triangles: 40,
            contacts: true,
            dual: false,
            sweeps: 1,
        };
        let names: Vec<&str> = encode_step(&fast(), &i, false, true)
            .into_iter()
            .filter_map(|c| match c {
                Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => Some(kernel),
                _ => None,
            })
            .collect();
        let at = |k: &str| {
            names
                .iter()
                .position(|n| *n == k)
                .unwrap_or_else(|| panic!("{k} missing"))
        };
        assert!(at("soft_wake") < at("predict_bodies"));
        assert!(at("predict_bodies") < at("soft_begin"));
        assert!(
            at("soft_begin") < at("soft_collide_nodes")
                && at("soft_collide_nodes") < at("soft_collide_tris")
        );
        assert!(at("soft_collide_tris") < at("soft_primal_bodies"));
        assert!(!names.contains(&"soft_dual"), "no-op dual pass skipped");
        let sweeps = |inp: &StepInputs| -> Vec<u64> {
            encode_step(&fast(), inp, false, true)
                .into_iter()
                .filter_map(|c| match c {
                    Cmd::Dispatch {
                        kernel: "soft_primal_bodies",
                        threads,
                        ..
                    } => Some(threads),
                    Cmd::Dispatch {
                        kernel: "soft_primal" | "soft_primal_fused" | "soft_chebyshev",
                        ..
                    } => panic!("per-colour sweep planned"),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(
            sweeps(&i),
            vec![64; i.iterations as usize],
            "one sweep per iteration"
        );
        let mut two = i;
        two.soft.sweeps = 2;
        assert_eq!(
            sweeps(&two).len(),
            2 * i.iterations as usize,
            "two sweeps per iteration"
        );
        // Several bodies and vehicles: one workgroup per body, vehicle_iterate in
        // workgroups past them on the last sweep (no separate dispatch).
        let mut many = i;
        many.soft.bodies = 20;
        many.vehicle_count = 300;
        assert_eq!(sweeps(&many), vec![(20 + 5) * 64; i.iterations as usize]);
        assert!(!encode_step(&fast(), &many, false, true)
            .iter()
            .any(|c| matches!(
                c,
                Cmd::Dispatch {
                    kernel: "vehicle_iterate",
                    ..
                }
            )));
        assert!(
            at("soft_plastic") < at("soft_break_groups")
                && at("soft_break_groups") < at("extract_velocity_sleep")
        );
        assert!(at("extract_velocity_sleep") < at("soft_sleep"));
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
        assert!(at("broadphase_fast") < at("ccd_toi") && at("ccd_toi") < at("ccd_pick_key") && at("ccd_pick_key") < at("ccd_pick"));
        let np = ccd.iter().position(|n| n.starts_with("narrowphase")).unwrap();
        assert!(at("ccd_pick") < np);
        let emit = ccd.iter().position(|n| n.starts_with("ccd_emit")).unwrap();
        assert!(np < emit && emit < at("predict_bodies"));
    }

    #[test]
    fn soil_passes_follow_the_vehicle_passes_only_with_soil() {
        for m in [fast(), StepModes::default()] {
            for whole in [false, true] {
                let mut i = inputs(10, 0);
                i.has_height_field = true;
                i.vehicle_count = 2;
                i.wheel_count = 8;
                let plain = encode_step(&m, &i, whole, true);
                i.soil = true;
                let with = encode_step(&m, &i, whole, true);
                let soil = |c: &Cmd| {
                    matches!(
                        c,
                        Cmd::Dispatch {
                            kernel: "wheel_soil" | "wheel_soil_press",
                            ..
                        }
                    )
                };
                // Off: unchanged. On: the two passes inserted, nothing else moves.
                assert!(!plain.iter().any(soil));
                let stripped: Vec<Cmd> = with.iter().copied().filter(|c| !soil(c)).collect();
                assert_eq!(stripped, plain);
                let at = |k: &str| {
                    with.iter()
                        .position(|c| matches!(c, Cmd::Dispatch { kernel, .. } if *kernel == k))
                        .unwrap_or_else(|| panic!("{k} missing"))
                };
                assert!(at("vehicle_post") < at("wheel_soil"));
                assert_eq!(at("wheel_soil") + 1, at("wheel_soil_press"));
                assert!(at("wheel_soil_press") < at("write_sleep_args"));
                assert!(with.contains(&Cmd::Dispatch {
                    kernel: "wheel_soil_press",
                    threads: u64::from(engine_schema::vehicle::SOIL_LANES),
                    pass: 0,
                }));
                for k in ["wheel_soil", "wheel_soil_press"] {
                    assert!(PERSISTENT_ENTRIES.contains(&resolve_entry(k, true)));
                    assert_eq!(kernel_source(k), "vehicle_step.slang");
                }
                // No vehicles: no wheels to sink.
                i.vehicle_count = 0;
                assert!(!encode_step(&m, &i, whole, true).iter().any(soil));
            }
        }
    }

    #[test]
    fn contact_events_are_planned_only_for_flagged_worlds() {
        let names = |m: &StepModes, i: &StepInputs, whole: bool| -> Vec<&'static str> {
            encode_step(m, i, whole, true)
                .into_iter()
                .filter_map(|c| match c {
                    Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => Some(kernel),
                    _ => None,
                })
                .collect()
        };
        for mut m in [fast(), StepModes::default()] {
            for whole in [false, true] {
                m.contact_events = false;
                let mut i = inputs(10, 0);
                i.has_height_field = true;
                i.motion.ccd = true;
                let plain = encode_step(&m, &i, whole, true);
                assert!(
                    !names(&m, &i, whole)
                        .iter()
                        .any(|k| k.starts_with("contact_events")),
                    "a world without event shapes plans no contact_events pass"
                );
                m.contact_events = true;
                let with = encode_step(&m, &i, whole, true);
                // The same commands with the pass inserted: nothing else moves.
                let stripped: Vec<Cmd> = with
                    .iter()
                    .copied()
                    .filter(|c| {
                        !matches!(
                            c,
                            Cmd::Indirect {
                                kernel: "contact_events",
                                ..
                            }
                        )
                    })
                    .collect();
                assert_eq!(stripped, plain);
                let n = names(&m, &i, whole);
                let at = |k: &str| {
                    n.iter()
                        .position(|x| *x == k)
                        .unwrap_or_else(|| panic!("{k} missing"))
                };
                let last_manifold_stage = n
                    .iter()
                    .rposition(|k| {
                        k.starts_with("narrowphase")
                            || k.starts_with("convex_")
                            || k.starts_with("ccd_emit")
                            || k.starts_with("reuse_all_manifolds")
                            || k.starts_with("persist_")
                    })
                    .unwrap();
                assert!(last_manifold_stage < at("contact_events"));
                assert!(at("contact_events") < at("predict_bodies"));
                if !whole {
                    assert!(
                        at("convex_manifold") < at("contact_events"),
                        "heightfield pairs are included"
                    );
                }
            }
        }
    }

    /// The schedule properties demo.html's node tests checked on its own JS
    /// encoder (tests/persistence_schedule.test.mjs), now on the plan every host
    /// runs: each mode's contact writers, persistence, packing, CSR and solve
    /// stay in dependency order.
    #[test]
    fn every_mode_keeps_its_contact_dependencies() {
        let kernel = |c: &Cmd| match *c {
            Cmd::Dispatch { kernel, .. } | Cmd::Indirect { kernel, .. } => Some(kernel),
            _ => None,
        };
        for storage in ["canonical", "compact", "persistent"] {
            for persistence in ["contacts", "manifolds", "fused"] {
                if persistence == "fused" && storage == "canonical" {
                    continue;
                }
                for reuse in [false, true] {
                    for fast_paths in [false, true] {
                        let persistent = storage == "persistent";
                        let compact = storage != "canonical";
                        let copy = persistent && persistence == "contacts" && fast_paths;
                        let m = StepModes {
                            compact_contacts: compact,
                            persistent_contacts: persistent,
                            manifold_persistence: persistence == "manifolds",
                            fused_persistence: compact && persistence == "fused",
                            manifold_insert: fast_paths && reuse,
                            copy_reuse: copy,
                            split_np: copy,
                            listed_persist: copy,
                            carry_rest: fast_paths,
                            local_repair: fast_paths,
                            grid_split: fast_paths,
                            sparse_start: 10,
                            color_headroom: Some(4),
                            color_round_headroom: Some(2),
                            ..StepModes::default()
                        };
                        let i = StepInputs { reuse_contacts: reuse, sleep_time: 0.5, ..inputs(10, 3) };
                        let cmds = encode_step(&m, &i, false, true);
                        let label = format!("{storage} {persistence} reuse={reuse} fast={fast_paths}");
                        let names: Vec<&str> = cmds.iter().filter_map(kernel).collect();
                        let first = |pred: &dyn Fn(&str) -> bool| names.iter().position(|n| pred(n));
                        let last = |pred: &dyn Fn(&str) -> bool| names.iter().rposition(|n| pred(n));
                        let np_last = last(&|n| n.starts_with("narrowphase")).expect("narrowphase");
                        let persist_first = first(&|n| n.starts_with("persist_")).expect("a persistence pass");
                        let primal = first(&|n| n.starts_with("solve_primal")).expect("a solve");
                        assert!(np_last < persist_first, "{label}: persistence after every narrowphase pass");
                        assert!(persist_first < primal, "{label}: persistence before the solve");
                        if compact && persistence != "fused" {
                            let pack = first(&|n| n == "pack_solver_contacts").expect("pack");
                            assert_eq!(names.iter().filter(|n| **n == "pack_solver_contacts").count(), 1, "{label}: packed once");
                            assert!(persist_first < pack && pack < primal, "{label}: packed between persistence and the solve");
                            if persistent && persistence == "contacts" {
                                let csr = first(&|n| n == "csr_degree_contacts").unwrap();
                                assert!(pack < csr, "{label}: CSR reads the packed pairs");
                            }
                            if !persistent {
                                let clear = first(&|n| n == "clear_solver_contacts").expect("clear");
                                assert!(clear < pack, "{label}: cleared before packing");
                            }
                        }
                        if m.manifold_insert {
                            let insert = first(&|n| n == "contact_hash_insert_manifolds").unwrap();
                            assert!(insert < first(&|n| n.starts_with("narrowphase")).unwrap(), "{label}: manifold insert before narrowphase overwrites them");
                        }
                        if m.split_np {
                            let defer = first(&|n| n == "narrowphase_defer_copy_persistent").unwrap();
                            let fresh = first(&|n| n == "narrowphase_fresh_copy_persistent").unwrap();
                            assert!(defer < fresh && fresh < persist_first, "{label}: defer, then fresh pairs");
                        }
                        if m.listed_persist {
                            let at = |slot| cmds.iter().position(|c| matches!(c, Cmd::Indirect { kernel, slot: s, .. } if kernel.starts_with("persist_") && *s == slot));
                            let (list, scan) = (at(70).expect("list on slot 70"), at(71).expect("overflow scan on slot 71"));
                            let np = cmds.iter().rposition(|c| kernel(c).is_some_and(|k| k.starts_with("narrowphase"))).unwrap();
                            let args = cmds.iter().skip(np).position(|c| kernel(c).is_some_and(|k| k.starts_with("write_work_args"))).unwrap() + np;
                            assert!(args < list && list < scan, "{label}: list then scan, after the narrowphase work args");
                        }
                        if m.carry_rest {
                            let carry = first(&|n| n == "carry_rest_pairs").unwrap();
                            assert!(first(&|n| n == "broadphase_query").unwrap() < carry && carry < first(&|n| n.starts_with("narrowphase")).unwrap(), "{label}: carried pairs join the broadphase's before narrowphase");
                        }
                        if m.local_repair {
                            let local = first(&|n| n == "color_bodies_local").unwrap();
                            assert!(first(&|n| n == "init_colors").unwrap() < local && local < first(&|n| n == "color_bodies").unwrap(), "{label}: local repair before the global rounds");
                        }
                        assert_eq!(names.iter().filter(|n| **n == "sleep_island_gate").count(), 8, "{label}: the island gate runs with sleep on");
                    }
                }
            }
        }
        // A sleep time past any run (1e6 s, the ring and gables benchmarks) has
        // nothing for the gate to revoke.
        let cmds = encode_step(&fast(), &inputs(10, 0), false, true);
        assert!(!cmds.iter().any(|c| kernel(c) == Some("sleep_island_gate")));
    }
}

//! Single-precision GPU stress solver, diffed against the f64 reference (`stress-ref`).
//!
//! Kernels are written in Slang (`shaders/*.slang`) and compiled by
//! `scripts/compile_shaders.sh` (pinned Slang in Docker) to Metal Shading Language and
//! WGSL, both checked in. wgpu runs them: on macOS the Metal library goes straight to
//! the driver (`build.rs` builds it with Xcode), in a browser the WGSL goes to WebGPU.
//! Everything on the GPU is f32; accuracy against the reference is measured, not assumed.

pub mod colliders;

/// The GPU's split path: splits laid out on the GPU (`STRESS_GPU_LAYOUT`), and a frame's
/// cascade and splits in one submission (`STRESS_GPU_FRAME`). On when a capability profile
/// asks for it (`GpuWorld::apply_profile`); either variable overrides it (1 on, 0 off).
pub static GPU_FRAME_PROFILE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn switch(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => v == "1",
        Err(_) => GPU_FRAME_PROFILE.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// Plan 5d, work in progress (engine/docs/stress-5d-one-command-buffer.md): a resident
/// frame's engine substeps, ingest and cascade in one command buffer
/// (`STRESS_FRAME_SUBMITS=one`; off by default until it meets the plan's acceptance).
pub fn one_command_buffer() -> bool {
    std::env::var("STRESS_FRAME_SUBMITS").is_ok_and(|v| v == "one")
}

/// Splits laid out on the GPU (`STRESS_GPU_LAYOUT`).
pub fn gpu_layout() -> bool {
    switch("STRESS_GPU_LAYOUT")
}

/// A frame's cascade and splits in one submission (`STRESS_GPU_FRAME`, with the layout).
pub fn gpu_frame() -> bool {
    gpu_layout() && switch("STRESS_GPU_FRAME")
}
pub mod gpu;
pub mod gpu_plan;
pub mod gpu_pass;
pub mod gpu_split;
pub mod ingest;
pub mod resident;
pub mod routing;
pub mod gate;
pub mod contacts;
pub mod coupled;
pub mod coverage;
pub use stress_layout::joint;
pub mod loads;
pub mod methods;
pub mod micro;
pub mod gnn_probe;
pub mod learned;
pub mod learned_feat;
pub mod learned_timing;

/// What `build.rs` reflects from the generated WGSL: how many storage buffers each module's
/// group 0 declares, and which group-0 bindings each entry point uses.
pub mod reflection {
    include!(concat!(env!("OUT_DIR"), "/entry_bindings.rs"));

    /// Storage buffers `module` declares in group 0.
    pub fn module_storage(module: &str) -> Option<u32> {
        MODULE_STORAGE.iter().find(|m| m.0 == module).map(|m| m.1)
    }

    /// The group-0 bindings `entry` of `module` uses, ascending.
    pub fn entry_bindings(module: &str, entry: &str) -> Option<&'static [u32]> {
        ENTRY_BINDINGS.iter().find(|e| e.0 == module && e.1 == entry).map(|e| e.2)
    }

    /// The storage buffers `entry` of `module` uses.
    pub fn entry_storage(module: &str, entry: &str) -> Option<u32> {
        ENTRY_BINDINGS.iter().find(|e| e.0 == module && e.1 == entry).map(|e| e.3)
    }

    /// Entry points that bind more storage buffers than WebGPU allows per stage (10), and
    /// the entry points that do their work in order where a device allows fewer than they
    /// use (`Gpu::staged`): (module, entry, phases).
    pub const PHASED: &[(&str, &str, &[&str])] = &[
        ("pass", "pass_snapshot", &["pass_snapshot_chunks", "pass_snapshot_bonds", "pass_snapshot_ctab"]),
        ("passlog", "pass_log", &["pass_log_head", "pass_log_tail"]),
        (
            "layout",
            "layout_parent",
            &["layout_parent_positions", "layout_parent_chunks", "layout_parent_ctab", "layout_parent_bond_ranks", "layout_parent_bonds", "layout_parent_incidence", "layout_parent_records", "layout_parent_engine"],
        ),
        ("spawn", "spawn_bodies", &["spawn_bodies_mass", "spawn_bodies_pose", "spawn_bodies_record"]),
        ("spawn", "spawn_bodies_group", &["spawn_bodies_group_mass", "spawn_bodies_group_pose", "spawn_bodies_group_record"]),
        ("spawn", "spawn_colliders", &["spawn_collider_members", "spawn_colliders_phased"]),
    ];

    /// `entry`'s phases (`PHASED`).
    pub fn phases(module: &str, entry: &str) -> Option<&'static [&'static str]> {
        PHASED.iter().find(|p| p.0 == module && p.1 == entry).map(|p| p.2)
    }
}

/// The generated shaders of one Slang file, in every form the runtime can load.
#[derive(Clone, Copy, Debug)]
pub struct ShaderSet {
    pub name: &'static str,
    pub wgsl: &'static str,
    pub msl: &'static str,
    /// Built from `msl` by `build.rs` (empty when Xcode was unavailable): without fast
    /// math (parity), and with it (production, `STRESS_GPU_MATH=fast`).
    pub metallib: &'static [u8],
    pub metallib_fast: &'static [u8],
    /// Entry points and their workgroup sizes (render stages: `(1, 1, 1)`).
    pub entries: &'static [(&'static str, (u32, u32, u32))],
}

macro_rules! shader_set {
    ($name:literal, [$(($entry:literal, $size:expr)),* $(,)?]) => {
        ShaderSet {
            name: $name,
            wgsl: include_str!(concat!("../shaders/generated/", $name, ".wgsl")),
            msl: include_str!(concat!("../shaders/generated/", $name, ".metal")),
            metallib: include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib")),
            metallib_fast: include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".fast.metallib")),
            entries: &[$(($entry, $size)),*],
        }
    };
}

/// Every Slang file's generated shaders.
pub mod shaders {
    use super::ShaderSet;
    /// `WORLD` with implicit penalty contact (`world_ic.slang`, `STRESS_REF_IMPLICIT_CONTACT=1`).
    pub const WORLD_IC: ShaderSet = ShaderSet {
        name: "world_ic",
        wgsl: include_str!("../shaders/generated/world_ic.wgsl"),
        msl: include_str!("../shaders/generated/world_ic.metal"),
        metallib: include_bytes!(concat!(env!("OUT_DIR"), "/world_ic.metallib")),
        metallib_fast: include_bytes!(concat!(env!("OUT_DIR"), "/world_ic.fast.metallib")),
        entries: WORLD.entries,
    };
    pub const SMOKE: ShaderSet = shader_set!("smoke", [("smoke", (64, 1, 1))]);
    pub const STRESS_STEP: ShaderSet = shader_set!("stress_step", [("bond_forces", (64, 1, 1)), ("chunk_integrate", (64, 1, 1)), ("island_substeps", (256, 1, 1)), ("island_shared_substeps", (256, 1, 1))]);
    pub const JOINT_TEST: ShaderSet = shader_set!("joint_test", [("joint_eval_test", (64, 1, 1))]);
    pub const WORLD: ShaderSet =
        shader_set!("world", [
            ("contact_forces", (64, 1, 1)),
            ("contact_sums", (64, 1, 1)),
            ("impactor_shares", (64, 1, 1)),
            ("impactor_crush", (256, 1, 1)),
            ("island_frame", (256, 1, 1)),
            ("island_frame_bonds_wide", (512, 1, 1)),
            ("impactor_integrate", (256, 1, 1)),
            ("wide_bonds", (256, 1, 1)),
            ("wide_chunks", (256, 1, 1)),
            ("wide_drift", (256, 1, 1)),
            ("wide_rigid", (256, 1, 1)),
            ("wide_end", (256, 1, 1)),
            ("wide_wake", (256, 1, 1)),
            ("island_statics", (256, 1, 1)),
            ("settled_fatigue", (256, 1, 1)),
            ("sums_and_impactors", (256, 1, 1)),
            ("islands_and_wide_chunks", (256, 1, 1)),
            ("sums_impactors_bonds", (256, 1, 1)),
            ("contact_islands", (256, 1, 1)),
            ("contact_islands_and_wide_chunks", (256, 1, 1)),
            ("forces_and_bonds", (256, 1, 1)),
            ("islands_sums_impactors", (256, 1, 1)),
            ("islands_sums_impactors_wake", (256, 1, 1)),
            ("travel_start", (64, 1, 1)),
        ]);
    pub const STRESS_RENDER: ShaderSet = shader_set!("stress_render", [("stress_vs", (1, 1, 1)), ("stress_fs", (1, 1, 1))]);
    pub const DESTRUCTION_RENDER: ShaderSet = shader_set!(
        "destruction_render",
        [("chunk_vs", (1, 1, 1)), ("chunk_hull_vs", (1, 1, 1)), ("host_chunk_vs", (1, 1, 1)), ("host_hull_vs", (1, 1, 1)), ("impactor_vs", (1, 1, 1)), ("ground_vs", (1, 1, 1)), ("lit_fs", (1, 1, 1)), ("ground_fs", (1, 1, 1))]
    );
    pub const COUPLING: ShaderSet = shader_set!(
        "coupling",
        [("ingest_rows", (64, 1, 1)), ("ingest_clear", (64, 1, 1)), ("ingest_count", (64, 1, 1)), ("ingest_scan", (1024, 1, 1)), ("ingest_scatter", (64, 1, 1)), ("ingest_merge", (64, 1, 1))]
    );
    pub const RESIDENT: ShaderSet = shader_set!("resident", [("resident_poses", (64, 1, 1)), ("resident_filter", (64, 1, 1)), ("resident_static_loads", (256, 1, 1))]);
    pub const CASCADE: ShaderSet = shader_set!("cascade", [("cascade_begin", (64, 1, 1)), ("cascade_gate", (64, 1, 1)), ("cascade_fixed", (64, 1, 1)), ("cascade_restore", (64, 1, 1)), ("cascade_settle", (64, 1, 1)), ("cascade_end", (256, 1, 1)), ("cascade_backup", (64, 1, 1))]);
    pub const PLAN: ShaderSet = shader_set!("plan", [("plan_motion", (64, 1, 1)), ("plan_reach", (64, 1, 1)), ("plan_pair_count", (64, 1, 1)), ("plan_scan", (1024, 1, 1)), ("plan_pair_write", (64, 1, 1)), ("plan_side", (64, 1, 1))]);
    pub const LAYOUT: ShaderSet = shader_set!(
        "layout",
        [
            ("layout_plan", (1, 1, 1)),
            ("layout_moved", (64, 1, 1)),
            ("layout_parent", (256, 1, 1)),
            ("layout_copy", (64, 1, 1)),
            ("layout_parent_positions", (256, 1, 1)),
            ("layout_parent_chunks", (256, 1, 1)),
            ("layout_parent_ctab", (256, 1, 1)),
            ("layout_parent_bond_ranks", (256, 1, 1)),
            ("layout_parent_bonds", (256, 1, 1)),
            ("layout_parent_incidence", (256, 1, 1)),
            ("layout_parent_records", (256, 1, 1)),
            ("layout_parent_engine", (256, 1, 1)),
        ]
    );
    pub const SPAWN: ShaderSet = shader_set!(
        "spawn",
        [
            ("spawn_offsets", (1024, 1, 1)),
            ("spawn_bodies", (64, 1, 1)),
            ("spawn_colliders", (64, 1, 1)),
            ("spawn_bodies_group", (64, 1, 1)),
            ("spawn_retire", (64, 1, 1)),
            ("spawn_bodies_mass", (64, 1, 1)),
            ("spawn_bodies_pose", (64, 1, 1)),
            ("spawn_bodies_record", (64, 1, 1)),
            ("spawn_bodies_group_mass", (64, 1, 1)),
            ("spawn_bodies_group_pose", (64, 1, 1)),
            ("spawn_bodies_group_record", (64, 1, 1)),
            ("spawn_collider_members", (64, 1, 1)),
            ("spawn_colliders_phased", (64, 1, 1)),
        ]
    );
    pub const COLLIDERS: ShaderSet = shader_set!("colliders", [("collider_keys", (64, 1, 1)), ("collider_slots", (64, 1, 1)), ("collider_refresh", (64, 1, 1))]);
    pub const PASS: ShaderSet = shader_set!(
        "pass",
        [
            ("pass_parents", (1024, 1, 1)),
            ("pass_snapshot", (256, 1, 1)),
            ("pass_whole", (256, 1, 1)),
            ("pass_finish", (1, 1, 1)),
            ("pass_snapshot_chunks", (256, 1, 1)),
            ("pass_snapshot_bonds", (256, 1, 1)),
            ("pass_snapshot_ctab", (256, 1, 1)),
        ]
    );
    pub const PASSLOG: ShaderSet = shader_set!("passlog", [("pass_log", (256, 1, 1)), ("pass_log_head", (256, 1, 1)), ("pass_log_tail", (256, 1, 1))]);
    pub const ROUTING: ShaderSet = shader_set!("routing", [("route_islands", (64, 1, 1))]);
    pub const PUNCH: ShaderSet = shader_set!("punch", [("punch_islands", (256, 1, 1)), ("punch_fast", (1024, 1, 1))]);
    /// The learned impact-outcome model's engine side (`STRESS_IMPACT_PUNCH=learned`).
    pub const LEARNED: ShaderSet = shader_set!(
        "learned",
        [
            ("learned_decode", (64, 1, 1)),
            ("learned_clear", (64, 1, 1)),
            ("learned_select", (64, 1, 1)),
            ("learned_debris", (64, 1, 1)),
            ("learned_compact", (64, 1, 1)),
            ("learned_count_bonds", (64, 1, 1)),
            ("learned_gather_chunks", (64, 1, 1)),
            ("learned_gather_bonds", (64, 1, 1)),
            ("learned_gather_impactors", (32, 1, 1)),
            ("learned_finish", (64, 1, 1)),
            ("learned_stub", (64, 1, 1)),
            ("learned_gate", (64, 1, 1)),
            ("learned_check", (64, 1, 1)),
            ("learned_apply_bonds", (64, 1, 1)),
            ("learned_apply_bodies", (64, 1, 1)),
            ("learned_apply_impactors", (32, 1, 1)),
            ("learned_place", (64, 1, 1)),
            ("learned_tally_apply", (1, 1, 1)),
            ("learned_tally", (1, 1, 1)),
            ("learned_adopt", (64, 1, 1)),
        ]
    );
    /// The learned impact model's input features (`learned_feat.slang`).
    pub const LEARNED_FEAT: ShaderSet = shader_set!(
        "learned_feat",
        [
            ("feat_map", (64, 1, 1)),
            ("feat_keys", (64, 1, 1)),
            ("feat_brow", (64, 1, 1)),
            ("feat_contacts", (64, 1, 1)),
            ("feat_scales", (64, 1, 1)),
            ("feat_summ", (64, 1, 1)),
            ("feat_graph", (64, 1, 1)),
            ("feat_ic", (64, 1, 1)),
            ("feat_cracked_find", (64, 1, 1)),
            ("feat_cracked", (64, 1, 1)),
            ("feat_sc", (64, 1, 1)),
            ("feat_prefix", (1, 1, 1)),
            ("feat_nodes", (64, 1, 1)),
            ("feat_bonds", (64, 1, 1)),
            ("feat_deform", (64, 1, 1)),
        ]
    );
    pub const SPLIT: ShaderSet = shader_set!("split", [("split_components", (256, 1, 1)), ("split_children", (64, 1, 1)), ("split_frames", (64, 1, 1)), ("split_children_group", (64, 1, 1))]);
    pub const HUD_OVERLAY: ShaderSet = shader_set!("hud_overlay", [("hud_vs", (1, 1, 1)), ("hud_fs", (1, 1, 1))]);
    pub const THRESHOLD: ShaderSet = shader_set!("threshold", [("threshold_breaks", (256, 1, 1))]);
    /// The learned impact-outcome plan's P0 cost probe (`gnn_probe`), f32 and f16.
    pub const GNN_PROBE: ShaderSet = shader_set!("gnn_probe", [("gnn_round", (128, 1, 1)), ("gnn_mlp_tiled", (128, 1, 1)), ("gnn_csr_sum", (128, 1, 1))]);
    pub const GNN_PROBE_F16: ShaderSet = shader_set!("gnn_probe_f16", [("gnn_round", (128, 1, 1)), ("gnn_mlp_tiled", (128, 1, 1)), ("gnn_csr_sum", (128, 1, 1))]);

    /// Every set (each Slang file with entry points).
    pub const ALL: &[ShaderSet] = &[
        WORLD_IC, SMOKE, STRESS_STEP, JOINT_TEST, WORLD, STRESS_RENDER, DESTRUCTION_RENDER, COUPLING, RESIDENT, CASCADE, PLAN, LAYOUT, SPAWN, COLLIDERS, PASS, PASSLOG, ROUTING, PUNCH, SPLIT,
        HUD_OVERLAY, THRESHOLD, GNN_PROBE, GNN_PROBE_F16, LEARNED, LEARNED_FEAT,
    ];
}

pub use stress_ref::scenes;
pub mod solver;
pub mod statics;
pub mod static_inputs;
pub mod stability;
pub mod world;
pub mod stress;

//! The rigid step graph's kernels: the entries a persistent-contact host compiles and
//! which Slang file defines each. `engine_exec::rigid_plan` and
//! `engine_schema::guidance` re-export them.

/// The guidance entries (`guide_step.slang`), in step order.
pub const GUIDE_ENTRIES: &[&str] = &[
    "guide_scan",
    "guide_select",
    "guide_rank",
    "guide_plan",
    "guide_commit",
];

/// Entries a persistent-contact host replays (every name `encode_step` can
/// emit, resolved with `resolve_entry`). Native hosts compile exactly these.
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
    if GUIDE_ENTRIES.contains(&entry) {
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

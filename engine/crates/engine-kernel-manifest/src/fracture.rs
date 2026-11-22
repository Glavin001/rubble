//! The fracture lab's scheduled kernels and their Slang files.
//! `engine_fracture::plan` re-exports them.

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
    ("ccd_pick_key", "ccd_pick_key"),
    ("ccd_pick", "ccd_pick"),
    ("ccd_mark", "ccd_mark"),
    ("ccd_wake", "ccd_wake"),
    ("ccd_emit", "ccd_emit_persistent"),
    ("ccd_respond", "ccd_respond"),
    ("ccd_respond_apply", "ccd_respond_apply"),
    ("ccd_trust", "ccd_trust"),
    ("ccd_unhold", "ccd_unhold"),
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
    // Directed debris (guide_step.slang), encoded only while guidance is on.
    ("guide_scan", "guide_scan"),
    ("guide_select", "guide_select"),
    ("guide_rank", "guide_rank"),
    ("guide_plan", "guide_plan"),
    ("guide_commit", "guide_commit"),
];

/// Slang source (relative to `engine/slang/kernels/`) that defines `entry`
/// (as engine-webgpu's build script compiles them).
pub fn kernel_source(entry: &str) -> &'static str {
    if entry.starts_with("guide_") {
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

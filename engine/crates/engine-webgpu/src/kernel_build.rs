// The engine-webgpu kernels' compile: every Slang entry this crate runs, to SPIR-V and
// WGSL, into a directory (`spirv/`, `spirv_entries.txt`, `wgsl/`). `include!`d by
// build.rs (the embedded kernels: release, the C library, wasm) and by lib.rs's
// kernel_store (the kernels compiled at run time: the fast profile on native, so a
// kernel edit compiles no Rust): one compile, the same bytes either way. No `use` lines
// here: the including file brings anyhow's Context and Result, engine_slang_build's
// compile functions, Path and PathBuf, wgsl_loop_cap.rs and explicit_post.rs.

/// Same world_step entries as `engine-metal`, minus `broadphase_grid` (thin wrapper;
/// slangc SIGSEGV on SPIR-V — runtime aliases to `broadphase_query`).
const WORLD_ENTRIES: &[&str] = &[
    "solve_primal_sparse",
    "solve_primal_sparse_compact",
    "solve_primal_sparse_persistent",
    "solve_primal_sparse_tail",
    "solve_primal_sparse_tail_compact",
    "solve_primal_sparse_tail_persistent",
    "solve_primal_sparse_loop",
    "solve_primal_sparse_loop_compact",
    "solve_primal_sparse_loop_persistent",
    "narrowphase_copy_persistent",
    "narrowphase_defer_copy_persistent",
    "narrowphase_fresh_copy_persistent",
    "csr_degree_contacts_persistent",
    "persist_list_copy_persistent",
    "write_work_args_copy_persistent",
    "csr_scatter_contacts_persistent",
    "persist_contacts_copy_persistent",
    "apply_commands",
    "apply_commands_persistent",
    "narrowphase_persistent",
    "contact_hash_insert_persistent",
    "contact_hash_insert_manifolds_persistent",
    "persist_contacts_persistent",
    "persist_manifolds_persistent",
    "pack_solver_pairs_persistent",
    "solve_primal_color_persistent",
    "solve_primal_one_persistent",
    "solve_dual_persistent",
    "predict_bodies",
    "build_aabbs",
    "joint_hash_clear",
    "joint_hash_insert",
    "broadphase_clear",
    "broadphase_insert",
    "broadphase_query",
    "broadphase_query_awake",
    "list_moving_colliders",
    "build_aabbs_listed",
    "broadphase_query_finer",
    "write_finer_args",
    "write_moved_args",
    "list_awake_colliders",
    "write_awake_collider_args",
    "rest_clear",
    "rest_insert",
    "rest_publish",
    "broadphase_insert_awake",
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
    "narrowphase",
    "update_refs",
    "contact_hash_clear",
    "contact_hash_insert",
    "contact_hash_insert_manifolds",
    "persist_contacts",
    "persist_manifolds",
    "persist_manifolds_compact",
    "warmstart_joints",
    "bond_wake",
    "build_awake_list",
    "csr_clear",
    "csr_degree_contacts",
    "csr_degree_joints",
    "csr_prefix",
    "csr_prefix_blocks",
    "csr_prefix_block_sums",
    "csr_prefix_add",
    "csr_prefix_init",
    "csr_prefix_scan",
    "csr_prefix_swap",
    "csr_prefix_finalize",
    "csr_scatter_contacts",
    "csr_scatter_joints",
    "csr_sort",
    "csr_sort_append",
    "color_bodies",
    "color_bodies_local",
    "color_bodies_tail",
    "carry_rest_pairs",
    "archive_free",
    "archive_count",
    "archive_alloc",
    "archive_scatter",
    "archive_seal",
    "archive_emit",
    "archive_spread",
    "archive_touch",
    "archive_touch_spread",
    "archive_touch_snap",
    "archive_touch_args",
    "archive_touch_seed",
    "archive_count_persistent",
    "archive_scatter_persistent",
    "init_colors",
    "pack_color_lists",
    "count_color_tiles",
    "count_color_tiles_bits",
    "scatter_color_tiles_bits",
    "clear_awake_bits",
    "build_awake_list_bits",
    "prefix_color_tiles",
    "prefix_color_tiles_parallel",
    "scatter_color_tiles",
    "write_color_args",
    "write_work_args",
    "write_sleep_args",
    "clear_solver_contacts",
    "pack_solver_contacts",
    "solve_primal_color_compact",
    "solve_primal_one_compact",
    "solve_dual_compact",
    "solve_primal_color",
    "solve_primal_one",
    "solve_dual",
    "sleep_island_gate",
    "sleep_island_commit",
    "record_stab_velocity",
    "extract_velocity_sleep",
    "recompute_inertia",
    // Continuous collision and step-start impulses (ccd_step.slang).
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
    "ccd_emit",
    "ccd_emit_persistent",
    "ccd_emit_copy_persistent",
    "record_color_degrees",
    "reuse_all_manifolds",
    "reuse_all_manifolds_persistent",
    "convex_classify",
    "convex_classify_persistent",
    "convex_bucket",
    "convex_write_args",
    "convex_sat_small",
    "convex_sat_large",
    "convex_manifold",
    "convex_manifold_persistent",
    // Vehicle controller (slang/kernels/vehicle_step.slang).
    "vehicle_input",
    "wheel_query",
    "vehicle_link",
    "vehicle_prepare",
    "vehicle_iterate",
    "vehicle_loads",
    "vehicle_loads_persistent",
    "vehicle_post",
    "raycast_batch",
    "wheel_soil",
    "wheel_soil_press",
    // Characters, in-step queries, projectiles (slang/kernels/character_step.slang).
    "char_move",
    "char_post",
    "query_batch",
    "projectile_step",
    // Soft bodies (slang/kernels/soft_step.slang).
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
    // Render poses for hosts (slang/kernels/pose_pack.slang).
    "pose_pack",
    // Directed debris (slang/kernels/guide_step.slang).
    "guide_scan",
    "guide_select",
    "guide_rank",
    "guide_plan",
    "guide_commit",
    // Deformable terrain MPM (slang/kernels/terrain_mpm.slang).
    "mpm_clear",
    "mpm_touch",
    "mpm_write_args",
    "mpm_count",
    "mpm_scan_cells",
    "mpm_scatter",
    "mpm_cell_sort",
    "mpm_payload",
    "mpm_p2g",
    "mpm_grid",
    "mpm_g2p",
    "mpm_activate_count",
    "mpm_activate_scan",
    "mpm_activate_emit",
    "mpm_activate_lower",
    "mpm_bake_bin",
    "mpm_bake_raise",
    "mpm_bake_header",
    "mpm_couple_clear",
    "mpm_couple_bin",
    "mpm_couple_gather",
    "mpm_couple_apply",
    "mpm_couple_emit",
    "mpm_couple_finish",
    "mpm_parcels",
    "mpm_deposit_raise",
    "mpm_deposit_header",
    "mpm_source",
    "mpm_absorb_init",
    "mpm_geostatic",
    "mpm_absorb_faces",
    "mpm_bake_shift",
    "mpm_bake_edge_clear",
    "mpm_bake_edge_count",
    "mpm_bake_edge_scan",
    "mpm_bake_edge_scatter",
    "mpm_bake_edge_sort",
    "mpm_source_surface",
    "mpm_grow_shift",
    "mpm_wheel_loads",
    "mpm_wheel_mass",
    "mpm_parcels_land",
    "mpm_pass",
    "mpm_sleep_wheels",
    "mpm_sleep_ring",
    "mpm_sleep_dilate",
    "mpm_sleep_touch",
    "mpm_pool_reserve",
    "mpm_pool_top",
    "mpm_pool_trim",
    "mpm_pool_holes_count",
    "mpm_pool_holes_scan",
    "mpm_pool_holes_list",
    "mpm_pool_finish",
    "mpm_parcels_pack",
    "mpm_parcels_unpack",
    // Terrain look-ahead (slang/kernels/terrain_watch.slang).
    "terrain_watch_energy",
    "terrain_watch_pick",
    "terrain_watch_write",
    // Render-only terrain surface (slang/kernels/terrain_surface.slang).
    "terrain_surface_clear",
    "terrain_surface_shift",
    "terrain_surface_shift_mean",
    "terrain_surface_bin",
    "terrain_surface_resolve",
    "terrain_surface_debris",
    "terrain_surface_gather",
    "terrain_surface_gather_parcels",
    "terrain_surface_publish",
    // Contact events (P4), canonical and persistent storage.
    "contact_events",
    "contact_events_persistent",
    // Overlap queries (P5) and explosions.
    "overlap_query",
    "explode",
    // Motion locks (P3).
    "lock_predict",
    "lock_project",
];

const CHARACTER_ENTRIES: &[&str] = &["char_move", "char_post", "query_batch", "projectile_step"];

const VEHICLE_ENTRIES: &[&str] = &[
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

/// Slang source that defines `entry`.
fn source_for(root: &std::path::Path, entry: &str) -> PathBuf {
    let kernels = root.join("kernels");
    if entry == "vehicle_loads_persistent" {
        return kernels.join("vehicle_step_persistent.slang");
    }
    if VEHICLE_ENTRIES.contains(&entry) {
        return kernels.join("vehicle_step.slang");
    }
    if CHARACTER_ENTRIES.contains(&entry) {
        return kernels.join("character_step.slang");
    }
    if entry.starts_with("soft_") {
        return kernels.join("soft_step.slang");
    }
    if entry == "pose_pack" {
        return kernels.join("pose_pack.slang");
    }
    if entry.starts_with("guide_") {
        return kernels.join("guide_step.slang");
    }
    if entry.starts_with("mpm_") {
        return kernels.join("terrain_mpm.slang");
    }
    if entry.starts_with("terrain_surface_") {
        return kernels.join("terrain_surface.slang");
    }
    if entry.starts_with("terrain_watch_") {
        return kernels.join("terrain_watch.slang");
    }
    if entry.starts_with("convex_") {
        return kernels.join(if entry.ends_with("_persistent") {
            "convex_persistent.slang"
        } else {
            "convex_step.slang"
        });
    }
    if entry.ends_with("_copy_persistent") {
        kernels.join("reuse_copy.slang")
    } else if entry.ends_with("_persistent") {
        kernels.join("persistent_solve.slang")
    } else if entry.ends_with("_compact")
        || entry == "pack_solver_contacts"
        || entry == "clear_solver_contacts"
    {
        kernels.join("compact_solve.slang")
    } else {
        kernels.join("world_step.slang")
    }
}

/// What a compile leaves besides the files in OUT_DIR: the `cargo:warning` lines and
/// the entries whose WGSL compiled (for `direct_wgsl.rs`, which names OUT_DIR paths).
pub(crate) struct Compiled {
    pub(crate) warnings: Vec<String>,
    pub(crate) wgsl_ok: Vec<String>,
}

/// Every slangc compile of this crate into `out_dir` (`spirv/`, `spirv_entries.txt`,
/// `wgsl/`).

pub(crate) fn compile_all(slang_root: &Path, out_dir: &Path) -> Result<Compiled> {
    let out = out_dir.join("spirv");
    std::fs::create_dir_all(&out)?;
    let mut warnings = Vec::new();
    let abi = slang_root.join("kernels/abi_probe.slang");
    compile(
        &abi,
        "abi_roundtrip",
        SlangTarget::Spirv,
        &out.join("abi_roundtrip.spv"),
        &[],
    )?;

    let mut ok = vec!["abi_roundtrip".to_string()];
    let mut failed = Vec::new();
    // Direct Slang → WGSL. `ENGINE_WGSL` switches atomic slots to Atomic<uint>
    // (physics/atomics.slang); the WGSL is embedded so wasm32 needs no filesystem.
    let wgsl_dir = out_dir.join("wgsl");
    std::fs::create_dir_all(&wgsl_dir)?;
    let mut wgsl_ok = Vec::new();
    let mut wgsl_failed = Vec::new();
    let wgsl_entries: Vec<&str> = WORLD_ENTRIES
        .iter()
        .copied()
        .chain(std::iter::once("broadphase_grid"))
        .collect();
    // The SPIR-V and the WGSL of every world entry in one compile_many: the entries of
    // one source and target in a few processes, each parsing the source once (the same
    // outputs as a compile each), both targets at once.
    let spv_sources: Vec<PathBuf> = WORLD_ENTRIES.iter().map(|e| source_for(slang_root, e)).collect();
    let spv_dests: Vec<PathBuf> = WORLD_ENTRIES.iter().map(|e| out.join(format!("{e}.spv"))).collect();
    let wgsl_sources: Vec<PathBuf> = wgsl_entries.iter().map(|e| source_for(slang_root, e)).collect();
    let wgsl_dests: Vec<PathBuf> = wgsl_entries.iter().map(|e| wgsl_dir.join(format!("{e}.wgsl"))).collect();
    let mut jobs: Vec<Compile> = Vec::new();
    for (k, entry) in WORLD_ENTRIES.iter().enumerate() {
        jobs.push(Compile {
            source: &spv_sources[k],
            entry,
            stage: "compute",
            target: SlangTarget::Spirv,
            out: &spv_dests[k],
            include_dirs: &[],
            defines: &[],
        });
    }
    for (k, entry) in wgsl_entries.iter().enumerate() {
        jobs.push(Compile {
            source: &wgsl_sources[k],
            entry,
            stage: "compute",
            target: SlangTarget::Wgsl,
            out: &wgsl_dests[k],
            include_dirs: &[],
            defines: &[("ENGINE_WGSL", "1")],
        });
    }
    let mut results = compile_many(&jobs).into_iter();
    let spirv: Vec<Result<()>> = results.by_ref().take(WORLD_ENTRIES.len()).collect();
    let wgsl: Vec<Result<()>> = results.collect();
    for (entry, result) in WORLD_ENTRIES.iter().zip(spirv) {
        match result {
            Ok(()) => ok.push((*entry).to_string()),
            Err(e) => {
                warnings.push(format!("SPIR-V skip {entry}: {e}"));
                failed.push((*entry).to_string());
            }
        }
    }
    // Alias for phase-graph name used by Metal. Not conditional on the alias being
    // absent: a reused OUT_DIR already holds it, and the entry list must not change.
    let query = out.join("broadphase_query.spv");
    let grid = out.join("broadphase_grid.spv");
    if query.is_file() {
        std::fs::copy(&query, &grid)?;
        ok.push("broadphase_grid".to_string());
    }

    std::fs::write(out_dir.join("spirv_entries.txt"), ok.join("\n") + "\n")?;
    if !failed.is_empty() {
        warnings.push(format!(
            "SPIR-V failed for {} entries: {}",
            failed.len(),
            failed.join(", ")
        ));
    }
    for (entry, result) in wgsl_entries.iter().zip(wgsl) {
        match result {
            Ok(()) => wgsl_ok.push((*entry).to_string()),
            Err(e) => {
                warnings.push(format!("WGSL skip {entry}: {e}"));
                wgsl_failed.push(*entry);
            }
        }
    }
    // Explicit vehicles in the World (engine-world `explicit_vehicles.rs`): one
    // standalone module.
    let coupling = wgsl_dir.join("vehicle_coupling.wgsl");
    let coupling_entries: Vec<(&str, &str)> = [
        "cpl_bounds", "cpl_plan", "cpl_share", "cpl_loads", "cpl_loads_end", "cpl_emit", "cpl_emit_end", "cpl_query", "cpl_sleep_mask",
    ]
    .iter()
    .map(|&e| (e, "compute"))
    .collect();
    compile_entries_to(
        &slang_root.join("kernels/vehicle_coupling.slang"),
        &coupling_entries,
        SlangTarget::Wgsl,
        &coupling,
        &[],
        &[],
    )
    .context("slangc failed for vehicle_coupling.slang")?;
    // Soft-body skinning: one standalone WGSL module, three entries.
    let skin = wgsl_dir.join("soft_skin.wgsl");
    compile_entries_to(
        &slang_root.join("kernels/soft_skin.slang"),
        &[("soft_skin_vertices", "compute"), ("soft_skin_faces", "compute"), ("soft_skin_normals", "compute")],
        SlangTarget::Wgsl,
        &skin,
        &[],
        &[],
    )
    .context("slangc failed for soft_skin.slang")?;
    // Every explicit module is regenerated below: clear the previous build's, so a
    // module that no longer exists does not linger in the output directory (the loop-cap
    // test reads every `explicit_*.wgsl` there).
    for entry in std::fs::read_dir(&wgsl_dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("explicit_") && name.ends_with(".wgsl") {
            std::fs::remove_file(entry.path())?;
        }
    }
    // Vehicle–vehicle contact: the plan passes (one module each) and the contact
    // groups' entry at the frame module's four variants.
    let mut vv: Vec<(String, &str, Vec<(&str, &str)>)> = ["bounds", "pairs", "group"]
        .iter()
        .map(|p| {
            (
                format!("explicit_beams_vv_{p}"),
                "",
                vec![("ENGINE_WGSL", "1")],
            )
        })
        .collect();
    // The frame module (`ExplicitBeamsGpu`, entry `explicit_beams_frame`), the same
    // with 512 lanes per vehicle for large vehicles, and the wide passes, one module per
    // entry point (the aux pass: a workgroup a vehicle of 512 lanes). Their barriers
    // order storage only: Slang lowers DeviceMemoryBarrierWithGroupSync with a texture
    // barrier naga rejects (and an empty statement Safari's parser rejects), so
    // `explicit_post` keeps the storage and workgroup barriers alone.
    vv.push(("explicit_beams".to_string(), "explicit_beams_frame", vec![("ENGINE_WGSL", "1")]));
    vv.push(("explicit_beams_wide".to_string(), "explicit_beams_frame", vec![("ENGINE_WGSL", "1"), ("EB_LANES", "512")]));
    // With wings, the fuselage drag and command beams (EB_AERO), at both lane counts:
    // the modules above stay those of vehicles without them.
    vv.push(("explicit_beams_aero".to_string(), "explicit_beams_frame", vec![("ENGINE_WGSL", "1"), ("EB_LANES", "256"), ("EB_AERO", "1")]));
    vv.push(("explicit_beams_aero_wide".to_string(), "explicit_beams_frame", vec![("ENGINE_WGSL", "1"), ("EB_LANES", "512"), ("EB_AERO", "1")]));
    for pass in ["begin", "rebase", "nodes", "beams", "tris", "end", "close", "hits", "centred_beams", "centred_nodes", "aux"] {
        let defines = if pass == "aux" { vec![("ENGINE_WGSL", "1"), ("EB_LANES", "512")] } else { vec![("ENGINE_WGSL", "1")] };
        vv.push((format!("explicit_beams_wide_{pass}"), "", defines));
    }
    // Vehicle sleep's awake list (engine_kernel_manifest::explicit::SLEEP_LIST).
    vv.push((
        "explicit_beams_sleep_list".to_string(),
        "",
        vec![("ENGINE_WGSL", "1"), ("EB_SLEEP_LIST", "1")],
    ));
    for (name, lanes, aero) in [
        ("explicit_beams_contact", "256", false),
        ("explicit_beams_contact_wide", "512", false),
        ("explicit_beams_contact_aero", "256", true),
        ("explicit_beams_contact_aero_wide", "512", true),
    ] {
        let mut d = vec![("ENGINE_WGSL", "1"), ("EB_LANES", lanes)];
        if aero {
            d.push(("EB_AERO", "1"));
        }
        // The solo vehicles' frame (the frame entry with EB_VV_SOLO), and the same
        // with the contact groups past the vehicles (EB_VV_GROUPS).
        d.push(("EB_VV_SOLO", "1"));
        vv.push((
            name.replace("contact", "solo"),
            "explicit_beams_frame",
            d.clone(),
        ));
        d.push(("EB_VV_GROUPS", "1"));
        // The groups of at most four members here (EB_VV_SMALL_ONLY), the larger ones
        // in their own module (EB_VV_BIG_ONLY, entry explicit_beams_contact):
        // engine_exec::explicit_plan Kernel::ContactBig.
        let mut small = d.clone();
        small.push(("EB_VV_SMALL_ONLY", "1"));
        vv.push((name.to_string(), "explicit_beams_frame", small));
        d.push(("EB_VV_BIG_ONLY", "1"));
        vv.push((name.replace("contact", "contact_big"), "explicit_beams_contact", d));
    }
    // (The explicit modules compile in parallel, as the baseline's: each a slangc of its own.)
    for r in par_map(&vv, |(name, entry, defines)| -> Result<()> {
        let out = wgsl_dir.join(format!("{name}.wgsl"));
        let entry = if entry.is_empty() { name.as_str() } else { entry };
        compile_with_defines(&explicit_beams_source(slang_root), entry, SlangTarget::Wgsl, &out, &[], defines)?;
        explicit_post(&out)
    }) {
        r?;
    }
    // The sixteen frame and contact modules again without surface contact
    // (EB_NO_SURFACE, the `_nosurf` names; engine_kernel_manifest::explicit::Kernel::module_for):
    // the host runs them when no vehicle of the upload has it, so a browser's first
    // load compiles the frame kernels without the surface pass (feature_query and the
    // pair path).
    const W: (&str, &str) = ("ENGINE_WGSL", "1");
    const L256: (&str, &str) = ("EB_LANES", "256");
    const L512: (&str, &str) = ("EB_LANES", "512");
    const AERO: (&str, &str) = ("EB_AERO", "1");
    const SOLO: (&str, &str) = ("EB_VV_SOLO", "1");
    const GROUPS: (&str, &str) = ("EB_VV_GROUPS", "1");
    const SMALL: (&str, &str) = ("EB_VV_SMALL_ONLY", "1");
    const BIG: (&str, &str) = ("EB_VV_BIG_ONLY", "1");
    const F: &str = "explicit_beams_frame";
    const C: &str = "explicit_beams_contact";
    let frames: [(&str, &str, &[(&str, &str)]); 16] = [
        ("explicit_beams", F, &[W]),
        ("explicit_beams_wide", F, &[W, L512]),
        ("explicit_beams_aero", F, &[W, L256, AERO]),
        ("explicit_beams_aero_wide", F, &[W, L512, AERO]),
        ("explicit_beams_solo", F, &[W, L256, SOLO]),
        ("explicit_beams_solo_wide", F, &[W, L512, SOLO]),
        ("explicit_beams_solo_aero", F, &[W, L256, AERO, SOLO]),
        ("explicit_beams_solo_aero_wide", F, &[W, L512, AERO, SOLO]),
        ("explicit_beams_contact", F, &[W, L256, SOLO, GROUPS, SMALL]),
        ("explicit_beams_contact_wide", F, &[W, L512, SOLO, GROUPS, SMALL]),
        ("explicit_beams_contact_aero", F, &[W, L256, AERO, SOLO, GROUPS, SMALL]),
        ("explicit_beams_contact_aero_wide", F, &[W, L512, AERO, SOLO, GROUPS, SMALL]),
        ("explicit_beams_contact_big", C, &[W, L256, SOLO, GROUPS, BIG]),
        ("explicit_beams_contact_big_wide", C, &[W, L512, SOLO, GROUPS, BIG]),
        ("explicit_beams_contact_big_aero", C, &[W, L256, AERO, SOLO, GROUPS, BIG]),
        ("explicit_beams_contact_big_aero_wide", C, &[W, L512, AERO, SOLO, GROUPS, BIG]),
    ];
    for r in par_map(&frames, |&(name, entry, defines)| -> Result<()> {
        let out = wgsl_dir.join(format!("{name}_nosurf.wgsl"));
        let mut d = defines.to_vec();
        d.push(("EB_NO_SURFACE", "1"));
        compile_with_defines(&explicit_beams_source(slang_root), entry, SlangTarget::Wgsl, &out, &[], &d)?;
        explicit_post(&out)
    }) {
        r?;
    }
    // Twelve of the sixteen with surface contact but without the World's coupling
    // (EB_NO_COUPLING, the `_nocpl` names; explicit_beams_gpu::frame_module): the surface
    // pass without its proxy paths, for an upload that has no coupling room.
    for r in par_map(&frames, |&(name, entry, defines)| -> Result<()> {
        // (Not the large groups' modules: a fleet of more than four vehicles without
        // coupling runs the shipped ones.)
        if name.contains("contact_big") {
            return Ok(());
        }
        let out = wgsl_dir.join(format!("{name}_nocpl.wgsl"));
        let mut d = defines.to_vec();
        d.push(("EB_NO_COUPLING", "1"));
        compile_with_defines(&explicit_beams_source(slang_root), entry, SlangTarget::Wgsl, &out, &[], &d)?;
        explicit_post(&out)
    }) {
        r?;
    }
    // A fleet with surface contact on the frame modules (no contact groups): the frame
    // in segments, a step each (EB_FRAME_SEG, `<module>_seg`), with the surface pass
    // as its own small pipelines between them (EB_SURF_PIPE: the prep, then per kind
    // its find and its apply; engine_kernel_manifest::explicit::Kernel::FrameSeg, Surf). A
    // browser compiles these in seconds; the frame with the pass in it took minutes.
    let segs: [(&str, &[(&str, &str)]); 4] = [
        ("explicit_beams_seg", &[W]),
        ("explicit_beams_wide_seg", &[W, L512]),
        ("explicit_beams_aero_seg", &[W, L256, AERO]),
        ("explicit_beams_aero_wide_seg", &[W, L512, AERO]),
    ];
    let mut jobs: Vec<(String, &str, Vec<(&str, &str)>)> = segs
        .iter()
        .map(|(name, defines)| {
            let mut d = defines.to_vec();
            d.push(("EB_FRAME_SEG", "1"));
            (name.to_string(), F, d)
        })
        .collect();
    jobs.push(("explicit_surface_prep".into(), "explicit_surface_prep", vec![W, ("EB_SURF_PIPE", "1")]));
    for (kind, tag) in [("1", "tri"), ("2", "seg"), ("3", "point"), ("4", "pair")] {
        for phase in ["find", "apply"] {
            jobs.push((
                format!("explicit_surface_{phase}_{tag}"),
                if phase == "find" { "explicit_surface_find" } else { "explicit_surface_apply" },
                vec![W, ("EB_SURF_PIPE", "1"), ("EB_SURF_KIND", kind)],
            ));
        }
    }
    // The split paths' modules (the split group path, self-collision split, the wide
    // passes' reused pair list, self split with vehicle contact): one list for every
    // host's build (engine_kernel_manifest::explicit::split_modules).
    for m in engine_kernel_manifest::explicit::split_modules() {
        jobs.push((m.name, m.entry, m.defines));
    }
    for r in par_map(&jobs, |(name, entry, defines)| -> Result<()> {
        let out = wgsl_dir.join(format!("{name}.wgsl"));
        compile_with_defines(&explicit_beams_source(slang_root), entry, SlangTarget::Wgsl, &out, &[], defines)?;
        explicit_post(&out)
    }) {
        r?;
    }
    // (The test-only modules, the frozen baseline and reference and the deterministic-hits
    // and ablation builds, are not compiled here: engine-webgpu's test_kernels_dir
    // compiles them when a test or a measurement asks, from
    // engine_kernel_manifest::explicit::test_modules.)
    // Tests only (tests/explicit_loop_cap.rs): the explicit kernels' Slang loop caps
    // (explicit_loop_cap.slang) on a runaway, read at run time.
    compile_with_defines(
        &slang_root.join("kernels/explicit_loop_cap_test.slang"),
        "explicit_loop_cap_test",
        SlangTarget::Wgsl,
        &wgsl_dir.join("loop_cap_test.wgsl"),
        &[],
        &[],
    )?;
    // Compact snapshots' buffer extents (WgpuSim::measure_state): standalone.
    let extent = wgsl_dir.join("state_extent.wgsl");
    compile_entries_to(
        &slang_root.join("kernels/state_extent.slang"),
        &[("state_extent", "compute")],
        SlangTarget::Wgsl,
        &extent,
        &[],
        &[],
    )
    .context("slangc failed for state_extent.slang")?;
    // MPM material probe (tests/mpm_material.rs): standalone, with its own bindings,
    // so it stays out of the world's pipeline table.
    compile_with_defines(
        &slang_root.join("kernels/mpm_material_probe.slang"),
        "mpm_material_probe",
        SlangTarget::Wgsl,
        &wgsl_dir.join("mpm_material_probe.wgsl"),
        &[],
        &[("ENGINE_WGSL", "1")],
    )?;
    if !wgsl_failed.is_empty() {
        warnings.push(format!("direct WGSL failed: {}", wgsl_failed.join(", ")));
    }
    Ok(Compiled { warnings, wgsl_ok })
}

/// An explicit-beams module as slangc wrote it, made ready to ship. Slang's storage
/// barrier carries a texture barrier naga rejects (and an empty statement Safari's
/// parser rejects): keep the storage and workgroup barriers. Every loop is capped:
/// a runaway loop stops, flagged, instead of holding the GPU. The shipped kernel
/// (`explicit_beams.slang`) caps its loops in its source (`EB_LOOP_CAP`, every
/// target; its modules call `ebLoopCapHit`); a module without them (the frozen
/// baseline and reference, or a measurement build of another source) is capped
/// here (`src/wgsl_loop_cap.rs`, `cap_explicit_loops`). Measurement builds only (in
/// their own target directory): `RUBBLE_EB_REF_UNCAPPED=1` leaves the frozen
/// reference modules (`explicit_ref_*`) uncapped, so `tests/explicit_beams_bitwise.rs`
/// compares the capped kernel with the uncapped one bit for bit.
fn explicit_post(out: &Path) -> Result<()> {
    let name = out.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let text = explicit_post_text(name, &std::fs::read_to_string(out)?).map_err(|e| anyhow::anyhow!("{}: {e}", out.display()))?;
    std::fs::write(out, text)?;
    Ok(())
}

/// The shipped explicit-beams modules' source: `explicit_beams.slang`; for
/// measurement builds only (each in its own target directory), `RUBBLE_EB_REF=1`
/// the frozen reference `explicit_beams_ref.slang` and `RUBBLE_EB_SRC=<file>` any
/// other file, so a before/after timing runs both kernels through the same pipelines.
fn explicit_beams_source(slang_root: &Path) -> PathBuf {
    if let Ok(f) = std::env::var("RUBBLE_EB_SRC") {
        return PathBuf::from(f);
    }
    if std::env::var("RUBBLE_EB_REF").is_ok_and(|v| v == "1") {
        slang_root.join("kernels/explicit_beams_ref.slang")
    } else {
        slang_root.join("kernels/explicit_beams.slang")
    }
}

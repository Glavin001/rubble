use anyhow::Result;
use engine_slang_build::{compile, compile_with_defines, find_slangc, SlangTarget};
use std::env;
use std::path::PathBuf;

/// Same world_step entries as `engine-metal`, minus `broadphase_grid` (thin wrapper;
/// slangc SIGSEGV on SPIR-V — runtime aliases to `broadphase_query`).
const WORLD_ENTRIES: &[&str] = &[
    "solve_primal_sparse",
    "solve_primal_sparse_compact",
    "solve_primal_sparse_persistent",
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
    "init_colors",
    "pack_color_lists",
    "count_color_tiles",
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
    "ccd_classify",
    "ccd_sweep",
    "recompute_inertia",
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
    "vehicle_post",
    "raycast_batch",
    // Soft bodies (slang/kernels/soft_step.slang).
    "soft_wake",
    "soft_begin",
    "soft_bounds",
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

const VEHICLE_ENTRIES: &[&str] = &[
    "vehicle_input",
    "wheel_query",
    "vehicle_link",
    "vehicle_prepare",
    "vehicle_iterate",
    "vehicle_post",
    "raycast_batch",
];

/// Slang source that defines `entry`.
fn source_for(root: &std::path::Path, entry: &str) -> PathBuf {
    let kernels = root.join("kernels");
    if VEHICLE_ENTRIES.contains(&entry) {
        return kernels.join("vehicle_step.slang");
    }
    if entry.starts_with("soft_") {
        return kernels.join("soft_step.slang");
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

fn main() -> Result<()> {
    engine_slang_build::report_slangc_for_cargo();
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let slang_root = manifest.join("../../slang");
    let out = PathBuf::from(env::var("OUT_DIR")?).join("spirv");
    std::fs::create_dir_all(&out)?;
    println!("cargo:rerun-if-changed={}", slang_root.display());
    let _ = find_slangc()?;

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
    for entry in WORLD_ENTRIES {
        let dest = out.join(format!("{entry}.spv"));
        match compile(
            &source_for(&slang_root, entry),
            entry,
            SlangTarget::Spirv,
            &dest,
            &[],
        ) {
            Ok(()) => ok.push((*entry).to_string()),
            Err(e) => {
                println!("cargo:warning=SPIR-V skip {entry}: {e}");
                failed.push((*entry).to_string());
            }
        }
    }
    // Alias for phase-graph name used by Metal.
    let query = out.join("broadphase_query.spv");
    let grid = out.join("broadphase_grid.spv");
    if query.is_file() && !grid.is_file() {
        std::fs::copy(&query, &grid)?;
        ok.push("broadphase_grid".to_string());
    }

    let list_path = PathBuf::from(env::var("OUT_DIR")?).join("spirv_entries.txt");
    std::fs::write(&list_path, ok.join("\n") + "\n")?;
    if !failed.is_empty() {
        println!(
            "cargo:warning=SPIR-V failed for {} entries: {}",
            failed.len(),
            failed.join(", ")
        );
    }

    // Direct Slang → WGSL. `ENGINE_WGSL` switches atomic slots to Atomic<uint>
    // (physics/atomics.slang); the WGSL is embedded so wasm32 needs no filesystem.
    let wgsl_dir = PathBuf::from(env::var("OUT_DIR")?).join("wgsl");
    std::fs::create_dir_all(&wgsl_dir)?;
    let mut table = String::from("pub static DIRECT_WGSL: &[(&str, &str)] = &[\n");
    let mut wgsl_failed = Vec::new();
    for entry in WORLD_ENTRIES
        .iter()
        .chain(std::iter::once(&"broadphase_grid"))
    {
        let dest = wgsl_dir.join(format!("{entry}.wgsl"));
        match compile_with_defines(
            &source_for(&slang_root, entry),
            entry,
            SlangTarget::Wgsl,
            &dest,
            &[],
            &[("ENGINE_WGSL", "1")],
        ) {
            Ok(()) => table.push_str(&format!(
                "    ({entry:?}, include_str!({:?})),\n",
                dest.display().to_string()
            )),
            Err(e) => {
                println!("cargo:warning=WGSL skip {entry}: {e}");
                wgsl_failed.push(*entry);
            }
        }
    }
    table.push_str("];\n");
    std::fs::write(
        PathBuf::from(env::var("OUT_DIR")?).join("direct_wgsl.rs"),
        table,
    )?;
    // Soft-body skinning: one standalone WGSL module, three entries.
    let skin = wgsl_dir.join("soft_skin.wgsl");
    let status = std::process::Command::new(find_slangc()?)
        .args(["-target", "wgsl"])
        .args(["-entry", "soft_skin_vertices", "-stage", "compute"])
        .args(["-entry", "soft_skin_faces", "-stage", "compute"])
        .args(["-entry", "soft_skin_normals", "-stage", "compute"])
        .arg("-o")
        .arg(&skin)
        .arg(slang_root.join("kernels/soft_skin.slang"))
        .status()?;
    anyhow::ensure!(status.success(), "slangc failed for soft_skin.slang");
    if !wgsl_failed.is_empty() {
        println!(
            "cargo:warning=direct WGSL failed: {}",
            wgsl_failed.join(", ")
        );
    }
    Ok(())
}

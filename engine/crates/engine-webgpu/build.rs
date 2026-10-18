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
    "narrowphase_defer_pairs_copy_persistent",
    "narrowphase_fresh_pairs_copy_persistent",
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
    "narrowphase",
    "update_refs",
    "contact_hash_clear",
    "contact_hash_insert",
    "contact_hash_insert_manifolds",
    "persist_contacts",
    "persist_manifolds",
    "persist_manifolds_compact",
    "warmstart_joints",
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
    "extract_velocity_sleep",
    "ccd_classify",
    "ccd_sweep",
    "recompute_inertia",
];

fn main() -> Result<()> {
    engine_slang_build::report_slangc_for_cargo();
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let slang_root = manifest.join("../../slang");
    let out = PathBuf::from(env::var("OUT_DIR")?).join("spirv");
    std::fs::create_dir_all(&out)?;
    println!("cargo:rerun-if-changed={}", slang_root.display());
    let _ = find_slangc()?;

    let world = slang_root.join("kernels/world_step.slang");
    let compact = slang_root.join("kernels/compact_solve.slang");
    let persistent = slang_root.join("kernels/persistent_solve.slang");
    let reuse_copy = slang_root.join("kernels/reuse_copy.slang");
    let np_pairs = slang_root.join("kernels/np_pairs.slang");
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
            if entry.ends_with("_pairs_copy_persistent") {
                &np_pairs
            } else if entry.ends_with("_copy_persistent") {
                &reuse_copy
            } else if entry.ends_with("_persistent") {
                &persistent
            } else if entry.ends_with("_compact")
                || (entry == &"pack_solver_contacts" || entry == &"clear_solver_contacts")
            {
                &compact
            } else {
                &world
            },
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
            if entry.ends_with("_pairs_copy_persistent") {
                &np_pairs
            } else if entry.ends_with("_copy_persistent") {
                &reuse_copy
            } else if entry.ends_with("_persistent") {
                &persistent
            } else if entry.ends_with("_compact")
                || (entry == &"pack_solver_contacts" || entry == &"clear_solver_contacts")
            {
                &compact
            } else {
                &world
            },
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
    if !wgsl_failed.is_empty() {
        println!(
            "cargo:warning=direct WGSL failed: {}",
            wgsl_failed.join(", ")
        );
    }
    Ok(())
}

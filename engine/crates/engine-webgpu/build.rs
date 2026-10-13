use anyhow::Result;
use engine_slang_build::{compile, find_slangc, SlangTarget};
use std::env;
use std::path::PathBuf;

/// Same world_step entries as `engine-metal`, minus `broadphase_grid` (thin wrapper;
/// slangc SIGSEGV on SPIR-V — runtime aliases to `broadphase_query`).
const WORLD_ENTRIES: &[&str] = &[
    "apply_commands",
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
    "persist_contacts",
    "warmstart_joints",
    "build_awake_list",
    "csr_clear",
    "csr_degree_contacts",
    "csr_degree_joints",
    "csr_prefix",
    "csr_prefix_init",
    "csr_prefix_scan",
    "csr_prefix_swap",
    "csr_prefix_finalize",
    "csr_scatter_contacts",
    "csr_scatter_joints",
    "csr_sort",
    "color_bodies",
    "init_colors",
    "pack_color_lists",
    "write_color_args",
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
        match compile(&world, entry, SlangTarget::Spirv, &dest, &[]) {
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
    Ok(())
}

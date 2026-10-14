use anyhow::Result;
use engine_slang_build::{compile, find_slangc, SlangTarget};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    engine_slang_build::report_slangc_for_cargo();
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let slang_root = manifest.join("../../slang");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let msl_dir = out_dir.join("msl");
    std::fs::create_dir_all(&msl_dir)?;

    println!("cargo:rerun-if-changed={}", slang_root.display());
    let _ = find_slangc()?;

    let abi = slang_root.join("kernels/abi_probe.slang");
    compile(
        &abi,
        "abi_roundtrip",
        SlangTarget::Metal,
        &msl_dir.join("abi_roundtrip.metal"),
        &[],
    )?;

    let world = slang_root.join("kernels/world_step.slang");
    let compact = slang_root.join("kernels/compact_solve.slang");
    let entries = [
        "apply_commands",
        "predict_bodies",
        "build_aabbs",
        "joint_hash_clear",
        "joint_hash_insert",
        "broadphase_clear",
        "broadphase_insert",
        "broadphase_query",
        "broadphase_grid",
        "narrowphase",
        "update_refs",
        "contact_hash_clear",
        "contact_hash_insert",
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
        "init_colors",
        "pack_color_lists",
        "count_color_tiles",
        "prefix_color_tiles",
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
    for entry in entries {
        compile(
            if entry.ends_with("_compact")
                || (entry == "pack_solver_contacts" || entry == "clear_solver_contacts")
            {
                &compact
            } else {
                &world
            },
            entry,
            SlangTarget::Metal,
            &msl_dir.join(format!("{entry}.metal")),
            &[],
        )?;
    }

    // Also emit SPIR-V and CUDA artifacts for the port crates to consume from OUT_DIR copy.
    let spirv_dir = out_dir.join("spirv");
    let cuda_dir = out_dir.join("cuda");
    std::fs::create_dir_all(&spirv_dir)?;
    std::fs::create_dir_all(&cuda_dir)?;
    for entry in entries.iter().chain(["abi_roundtrip"].iter()) {
        let src = if *entry == "abi_roundtrip" {
            &abi
        } else if entry.ends_with("_compact")
            || (*entry == "pack_solver_contacts" || *entry == "clear_solver_contacts")
        {
            &compact
        } else {
            &world
        };
        let _ = compile(
            src,
            entry,
            SlangTarget::Spirv,
            &spirv_dir.join(format!("{entry}.spv")),
            &[],
        );
        let _ = compile(
            src,
            entry,
            SlangTarget::Cuda,
            &cuda_dir.join(format!("{entry}.cu")),
            &[],
        );
    }

    println!("cargo:rustc-env=ENGINE_MSL_DIR={}", msl_dir.display());
    Ok(())
}

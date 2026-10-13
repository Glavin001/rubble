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
        "persist_contacts",
        "warmstart_joints",
        "build_awake_list",
        "csr_clear",
        "csr_degree_contacts",
        "csr_degree_joints",
        "csr_prefix",
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
    for entry in entries {
        compile(
            &world,
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

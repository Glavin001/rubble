use anyhow::Result;
use engine_slang_build::{compile, SlangTarget};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let slang_root = manifest.join("../../slang");
    let out = PathBuf::from(env::var("OUT_DIR")?).join("spirv");
    std::fs::create_dir_all(&out)?;
    println!("cargo:rerun-if-changed={}", slang_root.display());

    let world = slang_root.join("kernels/world_step.slang");
    let abi = slang_root.join("kernels/abi_probe.slang");
    compile(&abi, "abi_roundtrip", SlangTarget::Spirv, &out.join("abi_roundtrip.spv"), &[])?;
    for entry in [
        "predict_bodies",
        "narrowphase",
        "solve_primal_color",
        "solve_dual",
        "extract_velocity_sleep",
    ] {
        compile(&world, entry, SlangTarget::Spirv, &out.join(format!("{entry}.spv")), &[])?;
    }
    Ok(())
}

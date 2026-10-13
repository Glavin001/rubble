use anyhow::Result;
use engine_slang_build::{compile, SlangTarget};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let slang_root = manifest.join("../../slang");
    let out = PathBuf::from(env::var("OUT_DIR")?).join("cuda");
    std::fs::create_dir_all(&out)?;
    println!("cargo:rerun-if-changed={}", slang_root.display());
    let world = slang_root.join("kernels/world_step.slang");
    for entry in [
        "predict_bodies",
        "broadphase_clear",
        "broadphase_insert",
        "broadphase_query",
        "broadphase_grid",
        "narrowphase",
        "solve_primal_color",
        "solve_dual",
        "extract_velocity_sleep",
        "ccd_classify",
        "ccd_sweep",
    ] {
        compile(
            &world,
            entry,
            SlangTarget::Cuda,
            &out.join(format!("{entry}.cu")),
            &[],
        )?;
    }
    Ok(())
}

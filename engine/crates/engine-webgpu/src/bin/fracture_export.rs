//! Export the opt-in convex demo kernel alongside the standard `wgsl_export` set.
use engine_slang_build::{compile_with_defines, SlangTarget};
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../../slang");
    for entry in [
        "fracture_prepare_candidates",
        "fracture_write_sat_args",
        "fracture_sat",
        "fracture_narrowphase",
        "fracture_broadphase_query",
        "fracture_build_aabbs",
        "fracture_broadphase_insert",
    ] {
        let out = manifest.join(format!("generated/wgsl/{entry}.wgsl"));
        std::fs::create_dir_all(out.parent().unwrap())?;
        compile_with_defines(
            &root.join("kernels/fracture_step.slang"),
            entry,
            SlangTarget::Wgsl,
            &out,
            &[&root.join("physics"), &root.join("kernels")],
            &[("ENGINE_WGSL", "1")],
        )?;
        let source = std::fs::read_to_string(&out)?;
        let module = naga::front::wgsl::parse_str(&source)
            .map_err(|e| anyhow::anyhow!(e.emit_to_string(&source)))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)?;
        println!("Validated {}", out.display());
    }
    Ok(())
}

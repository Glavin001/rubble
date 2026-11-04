//! Write every shipped explicit-beams WGSL module (as the host creates it) to
//! `generated/explicit-wgsl/` with a manifest, for `explicit-wgsl-check.mjs`.
fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("generated/explicit-wgsl");
    std::fs::create_dir_all(&dir)?;
    let mut manifest = Vec::new();
    for (name, entry, wgsl) in engine_webgpu::explicit_beams_shipped_wgsl() {
        std::fs::write(dir.join(format!("{name}.wgsl")), wgsl)?;
        manifest.push(serde_json::json!({ "name": name, "entry": entry }));
    }
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    println!("{} modules in {}", manifest.len(), dir.display());
    Ok(())
}

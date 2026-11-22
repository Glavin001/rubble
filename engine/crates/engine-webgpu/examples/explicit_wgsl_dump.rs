//! Write every shipped explicit-beams WGSL module (as the host creates it) to
//! `generated/explicit-wgsl/` with a manifest, for `explicit-wgsl-check.mjs`; with
//! `--fast`, the fast setting's modules (bit-exact off: `engine_webgpu::fast_float_wgsl`)
//! to `generated/explicit-wgsl-fast/`. `--raw-fast DIR`: the build's raw modules
//! (`EXPLICIT_BEAMS_WGSL_DIR`) rewritten for the fast setting into DIR, for
//! `vehicle_lab --ab-switch kernel --ab-kernel-dir DIR`.
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--raw-fast") {
        let out = std::path::PathBuf::from(args.get(i + 1).expect("--raw-fast DIR"));
        std::fs::create_dir_all(&out)?;
        let mut n = 0;
        // (The build's modules and the test-only ones, compiled on demand.)
        let dirs = [std::path::Path::new(engine_webgpu::EXPLICIT_BEAMS_WGSL_DIR), engine_webgpu::test_kernels_dir()];
        for e in dirs.iter().map(|d| std::fs::read_dir(d)).collect::<Result<Vec<_>, _>>()?.into_iter().flatten() {
            let path = e?.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            if name.starts_with("explicit_") && name.ends_with(".wgsl") {
                let text = std::fs::read_to_string(&path)?;
                std::fs::write(
                    out.join(&name),
                    engine_webgpu::fast_float_wgsl(&text).as_ref(),
                )?;
                n += 1;
            }
        }
        println!("{n} modules in {}", out.display());
        return Ok(());
    }
    let fast = args.iter().any(|a| a == "--fast");
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(if fast {
        "generated/explicit-wgsl-fast"
    } else {
        "generated/explicit-wgsl"
    });
    std::fs::create_dir_all(&dir)?;
    let mut manifest = Vec::new();
    for (name, entry, wgsl) in engine_webgpu::explicit_beams_shipped_wgsl() {
        let wgsl = engine_webgpu::explicit_wgsl(&wgsl, !fast);
        std::fs::write(dir.join(format!("{name}.wgsl")), wgsl.as_ref())?;
        manifest.push(serde_json::json!({ "name": name, "entry": entry }));
    }
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    println!("{} modules in {}", manifest.len(), dir.display());
    Ok(())
}

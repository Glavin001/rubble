//! Write Slang→WGSL sources next to this crate for the Chrome load harness, plus
//! the vehicle presets and the engine-vehicle wasm module (needs `wasm-pack` and the
//! `wasm32-unknown-unknown` target) the browser demo and node tests use.
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin wgsl_export
//! cd engine/crates/engine-webgpu && python3 -m http.server 8765
//! # Chrome → http://127.0.0.1:8765/chrome_wgsl_load.html
//! # Chrome → http://127.0.0.1:8765/vehicle.html
//! ```

use engine_vehicle::presets;
use engine_webgpu::WebGpuWorld;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("generated");
    let out = root.join("wgsl");
    std::fs::create_dir_all(&out)?;
    let report = WebGpuWorld::compile_shaders_report();
    let mut names = Vec::new();
    let mut bad = 0u32;
    for (name, r) in report {
        match r {
            Ok(wgsl) => {
                std::fs::write(out.join(format!("{name}.wgsl")), wgsl)?;
                names.push(name);
            }
            Err(e) => {
                eprintln!("FAIL {name}: {e}");
                bad += 1;
            }
        }
    }
    names.sort();
    // build.rs skips an entry slangc rejects (with only a cargo warning): fail
    // here rather than leave a stale module from an earlier build in place.
    let missing: Vec<_> = engine_exec::rigid_plan::SOFT_ENTRIES
        .iter()
        .filter(|k| !names.iter().any(|n| n == *k))
        .collect();
    anyhow::ensure!(
        missing.is_empty() && bad == 0,
        "WGSL entries failed to compile: {missing:?} (see the engine-webgpu build warnings)"
    );
    let entries = names
        .iter()
        .map(|n| format!("    \"{n}\""))
        .collect::<Vec<_>>()
        .join(",\n");
    let manifest = format!(
        "{{\n  \"note\": \"Fair bar vs three-avbd: load in Chrome WebGPU\",\n  \"count\": {},\n  \"entries\": [\n{entries}\n  ]\n}}\n",
        names.len()
    );
    std::fs::write(out.join("manifest.json"), manifest)?;
    // The fracture lab's draw shader: one module, the Metal viewer's entries.
    let draw =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../slang/render/fracture_draw.slang");
    engine_slang_build::compile_entries_to(
        &draw,
        &[("vs", "vertex"), ("fs", "fragment"), ("ground", "fragment"), ("wire", "fragment")],
        engine_slang_build::SlangTarget::Wgsl,
        &out.join("fracture_draw.wgsl"),
        &[],
        &[],
    )?;
    // The playground's render layers (slang/render/*.slang, engine/docs/playground-plan.md
    // phase 4): each page draws through these, not shaders of its own.
    for (name, entries) in [
        (
            "rigid_bodies",
            &[("vs", "vertex"), ("fs", "fragment"), ("vsGround", "vertex"), ("fsGround", "fragment")][..],
        ),
        ("heightfield", &[("vs", "vertex"), ("fs", "fragment")][..]),
        ("projectiles", &[("vs", "vertex"), ("fs", "fragment")][..]),
    ] {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../slang/render/{name}.slang"));
        engine_slang_build::compile_entries_to(
            &src,
            entries,
            engine_slang_build::SlangTarget::Wgsl,
            &out.join(format!("{name}.wgsl")),
            &[],
            &[],
        )?;
    }
    // Soft-body skinning (standalone, run after the step): the same module the
    // native SoftSkinner embeds.
    std::fs::write(out.join("soft_skin.wgsl"), engine_webgpu::SOFT_SKIN_WGSL.text())?;
    println!("Wrote {} WGSL modules to {}", names.len(), out.display());

    // Vehicle presets (fully expanded descriptors) for vehicle.html.
    let all = presets::all();
    std::fs::write(
        root.join("vehicle_presets.json"),
        serde_json::to_string_pretty(&all)?,
    )?;
    println!("Wrote {} vehicle presets", all.len());
    // engine-vehicle's packer and VehicleHost for vehicle.mjs.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-vehicle-wasm");
    // build-web.sh's RUBBLE_WEB_PROFILE: `fast` for a local build to iterate on (the
    // fast profile, no wasm-opt), else release.
    let profile: &[&str] = if std::env::var("RUBBLE_WEB_PROFILE").is_ok_and(|p| p == "fast") {
        &["--profile", "fast", "--no-opt"]
    } else {
        &["--release"]
    };
    let status = std::process::Command::new("wasm-pack")
        .args(["build", "--target", "web"])
        .args(profile)
        .arg("--out-dir")
        .arg(root.join("vehicle_wasm"))
        .arg(&crate_dir)
        .status()
        .map_err(|e| {
            anyhow::anyhow!("wasm-pack (cargo install wasm-pack) for generated/vehicle_wasm: {e}")
        })?;
    anyhow::ensure!(
        status.success(),
        "wasm-pack build of engine-vehicle-wasm failed"
    );
    if bad > 0 {
        anyhow::bail!("{bad} translations failed");
    }
    Ok(())
}

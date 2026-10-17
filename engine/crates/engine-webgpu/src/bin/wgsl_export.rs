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
    let draw = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../slang/render/fracture_draw.slang");
    let status = std::process::Command::new(engine_slang_build::find_slangc()?)
        .args(["-target", "wgsl", "-entry", "vs", "-stage", "vertex"])
        .args(["-entry", "fs", "-stage", "fragment", "-entry", "ground", "-stage", "fragment", "-entry", "wire", "-stage", "fragment"])
        .arg("-o")
        .arg(out.join("fracture_draw.wgsl"))
        .arg(&draw)
        .status()?;
    anyhow::ensure!(status.success(), "slangc failed for {}", draw.display());
    println!("Wrote {} WGSL modules to {}", names.len(), out.display());

    // Vehicle presets (fully expanded descriptors) for vehicle.html.
    let all = presets::all();
    std::fs::write(root.join("vehicle_presets.json"), serde_json::to_string_pretty(&all)?)?;
    println!("Wrote {} vehicle presets", all.len());
    // engine-vehicle's packer and VehicleHost for vehicle.mjs.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-vehicle-wasm");
    let status = std::process::Command::new("wasm-pack")
        .args(["build", "--target", "web", "--release", "--out-dir"])
        .arg(root.join("vehicle_wasm"))
        .arg(&crate_dir)
        .status()
        .map_err(|e| anyhow::anyhow!("wasm-pack (cargo install wasm-pack) for generated/vehicle_wasm: {e}"))?;
    anyhow::ensure!(status.success(), "wasm-pack build of engine-vehicle-wasm failed");
    if bad > 0 {
        anyhow::bail!("{bad} translations failed");
    }
    Ok(())
}

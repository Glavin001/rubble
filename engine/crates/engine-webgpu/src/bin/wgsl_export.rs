//! Write Slang→WGSL sources next to this crate for the Chrome load harness.
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin wgsl_export
//! cd engine/crates/engine-webgpu && python3 -m http.server 8765
//! # Chrome → http://127.0.0.1:8765/chrome_wgsl_load.html
//! ```

use engine_webgpu::WebGpuWorld;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("generated/wgsl");
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
    println!("Wrote {} WGSL modules to {}", names.len(), out.display());
    if bad > 0 {
        anyhow::bail!("{bad} translations failed");
    }
    Ok(())
}

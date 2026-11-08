//! vehiclegen: build a vehicle design to a glTF binary.
//!
//!     cargo run -p engine-vehiclegen --release --bin vehiclegen -- sedan out.glb [params.json] [--stats out.json] [--no-uv]
//!
//! params.json holds the design's parameters (any subset; the rest take the
//! defaults). --stats writes per-part triangle counts, bounding boxes and
//! vertex checksums (for comparison with the Python generator).

use std::time::Instant;

use engine_vehiclegen::export::{glb, render_parts};
use engine_vehiclegen::{build, design_from_json, BuildOpts};
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: vehiclegen <design> <out.glb> [params.json] [--stats stats.json] [--no-uv]");
        std::process::exit(2);
    }
    let opt = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let params = args.get(2).filter(|a| !a.starts_with("--")).map(|p| std::fs::read_to_string(p).expect("params file")).unwrap_or_default();
    let d = design_from_json(&args[0], &params).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let t0 = Instant::now();
    let b = build(d.as_ref(), BuildOpts { uv: !args.iter().any(|a| a == "--no-uv"), ..Default::default() });
    let t_build = t0.elapsed().as_secs_f64();
    let rp = render_parts(&b.parts);
    let bytes = glb(&rp, &b.markers, b.paint, b.metallic, b.roughness);
    std::fs::write(&args[1], &bytes).expect("write glb");
    let tris: usize = b.parts.0.iter().map(|(_, m)| m.ntris()).sum();
    eprintln!("{}: {} parts, {} markers, {} triangles in {:.2} s -> {} ({} KB)", b.name, b.parts.len(), b.markers.len(), tris, t_build, args[1], bytes.len() / 1024);
    if let Some(path) = opt("--stats") {
        let parts: Vec<_> = b
            .parts
            .0
            .iter()
            .map(|(n, m)| {
                let (lo, hi) = if m.v.is_empty() { (Default::default(), Default::default()) } else { m.bbox() };
                let s: glam::DVec3 = m.v.iter().copied().sum();
                let s2: f64 = m.v.iter().map(|p| p.length_squared()).sum();
                json!({ "name": n, "verts": m.v.len(), "quads": m.q.len(), "tris": m.t.len(), "ntris": m.ntris(),
                        "lo": [lo.x, lo.y, lo.z], "hi": [hi.x, hi.y, hi.z], "sum": [s.x, s.y, s.z], "sum2": s2 })
            })
            .collect();
        let markers: Vec<_> = b.markers.iter().map(|m| json!({ "name": m.name, "pos": [m.pos.x, m.pos.y, m.pos.z] })).collect();
        std::fs::write(&path, serde_json::to_string_pretty(&json!({ "parts": parts, "markers": markers, "seconds": t_build })).unwrap()).unwrap();
    }
}

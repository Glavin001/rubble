//! vehiclegen: build a vehicle design to a glTF binary.
//!
//!     cargo run -p engine-vehiclegen --release --bin vehiclegen -- sedan out.glb [params.json] [--stats out.json] [--no-uv]
//!
//! --sheet builds the legacy sheet parts; --solids prints every solid part's
//! closedness, volume, wall thickness and self-intersections.
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
    let uv = !args.iter().any(|a| a == "--no-uv");
    // --sheet: the legacy sheet parts (single-sided skins, hems, loose inner panels)
    let solid = !args.iter().any(|a| a == "--sheet");
    // --bench N: build N extra times (profiling, timing)
    if let Some(n) = opt("--bench") {
        let n: usize = n.parse().unwrap();
        let t = Instant::now();
        for _ in 0..n {
            std::hint::black_box(build(d.as_ref(), BuildOpts { uv, solid, ..Default::default() }));
        }
        eprintln!("bench: {:.1} ms per build", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
    }
    let t0 = Instant::now();
    let b = build(d.as_ref(), BuildOpts { uv, solid, ..Default::default() });
    let t_build = t0.elapsed().as_secs_f64();
    let rp = render_parts(&b.parts);
    let bytes = glb(&rp, &b.markers, b.paint, b.metallic, b.roughness);
    std::fs::write(&args[1], &bytes).expect("write glb");
    let tris: usize = b.parts.0.iter().map(|(_, m)| m.ntris()).sum();
    eprintln!("{}: {} parts, {} markers, {} triangles in {:.2} s -> {} ({} KB)", b.name, b.parts.len(), b.markers.len(), tris, t_build, args[1], bytes.len() / 1024);
    if args.iter().any(|a| a == "--solids") {
        println!("{:<22} {:>8} {:>6} {:>5} {:>10} {:>14} {:>6}", "part", "tris", "closed", "comp", "volume L", "thickness mm", "cross");
        for (n, m) in &b.parts.0 {
            if m.surf.is_none() && !(std::env::var("SOLID_ALL").is_ok() && (engine_vehiclegen::uvpack::is_panel(n) || n.contains("lens") || n.starts_with("glass"))) {
                continue;
            }
            let r = engine_vehiclegen::solid::check(m);
            let x = engine_vehiclegen::solid::self_intersections(m, 1);
            let skin = engine_vehiclegen::solid::self_intersections(&m.outer_only(), 1);
            let x: Vec<_> = x.into_iter().filter(|p| !skin.iter().any(|s| (*s - *p).length() < 0.03)).collect();
            print!("{:>4} skin ", skin.len());
            println!(
                "{:<22} {:>8} {:>6} {:>5} {:>10.3} {:>6.1}..{:<6.1} {:>6}{}",
                n,
                r.triangles,
                if r.closed() { "yes" } else { "NO" },
                r.components,
                r.volume * 1000.0,
                r.thickness_min * 1000.0,
                r.thickness_max * 1000.0,
                x.len(),
                x.first().map(|p| format!("  first at ({:.3}, {:.3}, {:.3})", p.x, p.y, p.z)).unwrap_or_default()
            );
            if std::env::var("SOLID_DEBUG").is_ok() && !x.is_empty() {
                // nearest vertex's tag and position histogram of the crossings
                let mut h = std::collections::BTreeMap::new();
                for p in &x {
                    let (mut best, mut bi) = (f64::INFINITY, 0);
                    for (i, v) in m.v.iter().enumerate() {
                        let d = (*v - *p).length();
                        if d < best {
                            best = d;
                            bi = i;
                        }
                    }
                    *h.entry((format!("{:?}", m.surf_of(bi)), (p.x * 10.0).round() as i32, (p.y * 10.0).round() as i32, (p.z * 10.0).round() as i32)).or_insert(0) += 1;
                }
                let mut hv: Vec<_> = h.into_iter().collect();
                hv.sort_by_key(|e| std::cmp::Reverse(e.1));
                println!("    {:?}", &hv[..hv.len().min(8)]);
            }
            if std::env::var("SOLID_DEBUG2").is_ok() {
                for (a, b, t, p) in engine_vehiclegen::solid::crossings(m, 1).into_iter().step_by(7).take(6) {
                    let d = |i: u32| format!("{:?}#{} ({:.4},{:.4},{:.4})", m.surf_of(i as usize), i, m.v[i as usize].x, m.v[i as usize].y, m.v[i as usize].z);
                    println!("    edge {} - {}\n      tri {} | {} | {}\n      at ({:.4},{:.4},{:.4})", d(a), d(b), d(t[0]), d(t[1]), d(t[2]), p.x, p.y, p.z);
                }
            }
            if !r.closed() {
                println!("    {} boundary edges, {} non-manifold", r.boundary_edges, r.nonmanifold_edges);
            }
        }
    }
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

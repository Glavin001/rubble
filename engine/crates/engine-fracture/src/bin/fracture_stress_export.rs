//! Writes a fracture GLB as the stress oracle's Shardwright input (`shardwright-stress/1`).
//!
//!   cargo run -p engine-fracture --bin fracture_stress_export -- <asset.glb> <out.json> [level=0]

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (glb, out) = match (args.first(), args.get(1)) {
        (Some(g), Some(o)) => (g, o),
        _ => anyhow::bail!("usage: fracture_stress_export <asset.glb> <out.json> [level=0]"),
    };
    let level: u8 = args.get(2).map(|l| l.parse()).transpose()?.unwrap_or(0);
    let asset = engine_fracture::parse_fracture_glb(&std::fs::read(glb)?)?;
    let json = engine_fracture::stress_export::shardwright_stress_json(&asset, level);
    std::fs::write(out, serde_json::to_string(&json)? + "\n")?;
    eprintln!(
        "{}: {} fragments (levels 0..={level}), {} bonds at level {level}",
        out,
        json["fragments"].as_array().map_or(0, |f| f.len()),
        json["bonds"].as_array().map_or(0, |b| b.len())
    );
    Ok(())
}

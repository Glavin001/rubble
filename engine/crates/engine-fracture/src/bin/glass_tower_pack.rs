//! Writes the glass tower and pavilion scene pack (`glass_tower::felling_scene_pack`, the fracture
//! lab's Generated · Glass tower) for the Unreal client's `fracture:glasstower`.
//!
//!   cargo run --release -p engine-fracture --bin glass_tower_pack -- <out.json>

fn main() -> anyhow::Result<()> {
    let out = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: glass_tower_pack <out.json>"))?;
    let pack = engine_fracture::glass_tower::felling_scene_pack();
    std::fs::write(&out, serde_json::to_string(&pack)?)?;
    eprintln!("{out}: {} nodes, {} bonds", pack["scenario"]["nodes"].as_array().map_or(0, |a| a.len()), pack["scenario"]["bonds"].as_array().map_or(0, |a| a.len()));
    Ok(())
}

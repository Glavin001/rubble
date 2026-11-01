//! Writes the town the Unreal client loads: a base ScenePack with one building replaced by a
//! glass office tower (`glass_tower`).
//!
//!   cargo run --release -p engine-fracture --bin town_pack -- <base.json> <out.json> [group=tower@tower-165]

use engine_fracture::glass_tower::{glass_tower, replace_building, GlassTowerOptions};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (base, out) = match (args.first(), args.get(1)) {
        (Some(b), Some(o)) => (b, o),
        _ => anyhow::bail!("usage: town_pack <base.json> <out.json> [group=tower@tower-165]"),
    };
    let group = args.get(2).map_or("tower@tower-165", String::as_str);
    let mut pack: serde_json::Value = serde_json::from_slice(&std::fs::read(base)?)?;
    let tower = glass_tower(&GlassTowerOptions::default());
    replace_building(&mut pack, group, &tower, "tower@glass-tower")?;
    std::fs::write(out, serde_json::to_string(&pack)?)?;
    eprintln!("{out}: {group} replaced by a glass tower ({} nodes, {} bonds)", tower.nodes.len(), tower.bonds.len());
    Ok(())
}

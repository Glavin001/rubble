//! The stress cooker on the lab's assets: every level cooks through the stress oracle,
//! every fragment becomes a compound hull chunk (no equivalent-box fallback), chunk `k`
//! is the `k`-th fragment of its level, and the `.stress.bin` sections hold the records.

use engine_fracture::cook::{cook, write_stress_bin};
use engine_fracture::parse_fracture_glb;
use stress_ref::scene::Methods;

fn asset(name: &str) -> engine_fracture::FractureAsset {
    let path = format!("{}/../engine-webgpu/fracture/assets/{name}", env!("CARGO_MANIFEST_DIR"));
    parse_fracture_glb(&std::fs::read(path).expect("asset")).expect("parse")
}

#[test]
fn lab_assets_cook_at_every_level_as_hull_chunks() {
    for (name, levels) in [("veneer-house.glb", 0..4u8), ("masonry-arch.glb", 0..4u8)] {
        let a = asset(name);
        for level in levels {
            let c = cook(&a, level, Methods::established()).unwrap_or_else(|e| panic!("{name} L{level}: {e}"));
            let fragments: Vec<u32> = a.physics.fragments.iter().filter(|f| f.level == level).map(|f| f.id).collect();
            let chunks = &c.scene.bodies[0].chunks;
            assert_eq!(chunks.len(), fragments.len(), "{name} L{level}");
            assert_eq!(c.chunk_fragment, fragments, "{name} L{level}: chunk order");
            let boxed = chunks.iter().filter(|ch| ch.compound.is_none()).count();
            assert_eq!(boxed, 0, "{name} L{level}: {boxed} fragments fell back to equivalent boxes");
            assert_eq!(c.tables.chunk_static.len(), fragments.len());
            assert_eq!(c.tables.bond_static.len(), c.tables.bond_dyn.len());
            assert!(!c.tables.bond_static.is_empty());
            let bin = write_stress_bin(&c);
            assert_eq!(&bin[..8], b"RSTRESS1");
        }
    }
}

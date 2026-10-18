//! Scene packs as lab scenes: every generated vibe-city preset builds (bodies, colliders,
//! bond welds, render mesh), supports are static, and bond capacities hold the structure's
//! own weight at its supports. Packs are generated, not committed: run
//! `bun run export.ts` in engine/scenes/vibe-city first; missing packs skip.

use engine_fracture::pack::{build_pack_scene, BLAST_DT};
use std::path::PathBuf;

fn pack_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/generated/vibe-city")
}

#[test]
fn every_vibe_city_pack_builds() {
    let dir = pack_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!(
            "skip: {} missing (run bun run export.ts in engine/scenes/vibe-city)",
            dir.display()
        );
        return;
    };
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json") && !p.ends_with("index.json"))
        .collect();
    files.sort();
    assert!(!files.is_empty());
    for path in files {
        let bytes = std::fs::read(&path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let (scene, info) =
            build_pack_scene(&bytes).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
        // Ground + one body per node, then the projectile slots.
        assert_eq!(scene.projectile_start as usize, 1 + info.nodes, "{name}");
        let statics = scene.bodies[1..=info.nodes]
            .iter()
            .filter(|b| b.inv_mass == 0.0)
            .count();
        assert_eq!(statics, info.supports, "{name}: supports are static");
        assert!(scene.joints.len() <= info.bonds, "{name}");
        assert!(!scene.vertices.is_empty(), "{name}: render mesh");
        for j in &scene.joints {
            assert!(
                j.strength.iter().all(|s| s.is_finite() && *s > 0.0),
                "{name}: {j:?}"
            );
            assert!(
                j.k.iter().all(|k| k.is_finite() && *k >= 0.0),
                "{name}: {j:?}"
            );
        }
        // Self-weight: the summed compression capacity of the world welds exceeds the weight.
        let weight: f64 = scene.bodies[1..=info.nodes]
            .iter()
            .filter(|b| b.inv_mass > 0.0)
            .map(|b| 9.81 / b.inv_mass)
            .sum();
        let held: f64 = scene
            .joints
            .iter()
            .filter(|j| j.world)
            .map(|j| j.strength[1])
            .sum();
        eprintln!(
            "{name:18} nodes {:5} supports {:4} welds {:6} (world {:4}) hulls {:4} weight {:>10.0} N  support capacity {:>12.0} N",
            info.nodes,
            info.supports,
            scene.joints.len(),
            scene.joints.iter().filter(|j| j.world).count(),
            scene.hull_count(),
            weight,
            held,
        );
        if info.supports > 0 && weight > 0.0 {
            assert!(held > weight, "{name}: support welds hold {held:.0} N < weight {weight:.0} N (BLAST_DT {BLAST_DT})");
        }
    }
}

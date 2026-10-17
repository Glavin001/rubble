//! Bundled house: flat zero-volume compound children are skipped at the fine levels
//! (same counts as the convex-hulls JS importer), and every fragment keeps a hull.

use engine_fracture::{build_scene, parse_fracture_glb};

#[test]
fn fine_levels_skip_only_flat_hulls() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/fracture/assets/veneer-house.glb");
    let asset = parse_fracture_glb(&std::fs::read(path).unwrap()).unwrap();
    assert!(build_scene(&asset, 0, None).unwrap().skipped_hulls.is_empty());
    assert!(build_scene(&asset, 1, None).unwrap().skipped_hulls.is_empty());
    for (level, flat, first) in [(2u8, 35usize, (1401u32, 3590u32)), (3, 36, (5162, 8777))] {
        let scene = build_scene(&asset, level, None).unwrap();
        assert_eq!(scene.skipped_hulls.len(), flat, "level {level}");
        let s0 = &scene.skipped_hulls[0];
        assert_eq!((s0.fragment, s0.hull), first);
        for skipped in &scene.skipped_hulls {
            let hull = skipped.hull;
            let v = &asset.physics.hulls[hull as usize].vertices;
            let flat_axis = (0..3).any(|k| {
                let (lo, hi) = v
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b): (f32, f32), p| (a.min(p[k]), b.max(p[k])));
                hi - lo < 1e-9
            });
            assert!(flat_axis, "hull {hull} skipped although it has volume");
        }
        let total: u32 = asset
            .physics
            .fragments
            .iter()
            .filter(|f| f.level == level)
            .map(|f| f.hull_count)
            .sum();
        assert_eq!(scene.hull_count(), total as usize - flat);
    }
}

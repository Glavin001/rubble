//! The bond graph's lines follow the joints: line k (render vertices 2k, 2k + 1) is joint k,
//! so the draw (`fracture_draw.slang`) hides it once that joint breaks. Every builder (GLB,
//! cuboid, scene pack) and the copies of a layout keep the pairing; each line vertex carries
//! the bond-line normal (`bond_line_normal`) the draw keys on.

mod common;

use engine_fracture::pack::build_pack_scene;
use engine_fracture::samples::{layout_scene, sample};
use engine_fracture::cuboid::{build_cuboid_house, CuboidOptions};
use engine_fracture::{build_scene, Scene};
use std::path::PathBuf;

const VERTEX: usize = 10;

/// Line k's two vertices name joint k's bodies (an anchor's both on its piece), with the
/// partner each bond-line normal encodes.
fn lines_follow_joints(scene: &Scene, name: &str) {
    assert_eq!(scene.lines.len(), 2 * VERTEX * scene.joints.len(), "{name}: one line per joint");
    let vertex = |i: usize| &scene.lines[i * VERTEX..(i + 1) * VERTEX];
    for (k, j) in scene.joints.iter().enumerate() {
        let (a, b) = (vertex(2 * k), vertex(2 * k + 1));
        let partner = |v: &[f32]| (-v[3] - 2.0).round() as u32;
        assert!(a[3] < -1.5 && b[3] < -1.5, "{name}: line {k} is a bond line");
        assert_eq!(a[9] as u32, j.body_a, "{name}: line {k}");
        if j.world {
            assert_eq!(b[9] as u32, j.body_a, "{name}: anchor line {k} ends on its piece");
            assert_eq!((partner(a), partner(b)), (0, 0), "{name}: anchor line {k}");
        } else {
            assert_eq!(b[9] as u32, j.body_b, "{name}: line {k}");
            assert_eq!((partner(a), partner(b)), (j.body_b, j.body_a), "{name}: line {k}");
        }
    }
}

#[test]
fn glb_levels() {
    for (asset, file) in [(common::house(), "veneer house"), (common::arch(), "masonry arch")] {
        for level in asset.levels() {
            let scene = build_scene(asset, level, None).unwrap();
            assert!(!scene.joints.is_empty(), "{file} level {level}");
            lines_follow_joints(&scene, &format!("{file} level {level}"));
        }
    }
}

#[test]
fn cuboid_house() {
    let scene = build_cuboid_house(&CuboidOptions::default()).unwrap();
    assert!(!scene.joints.is_empty());
    lines_follow_joints(&scene, "cuboid house");
}

/// Copy c's joint k is the house's joint k on copy c's bodies: the draw finds line k of
/// instance c at joint k + c * (joints per copy).
#[test]
fn layout_copies() {
    let s = sample("cuboids").unwrap();
    let one = build_cuboid_house(&CuboidOptions::default()).unwrap();
    let per = one.joints.len();
    let grid = layout_scene(one.clone(), s.layout, true);
    assert!(grid.copies > 1);
    assert_eq!(grid.joints.len(), per * grid.copies as usize);
    let stride = grid.instance_stride;
    for c in 0..grid.copies {
        for (k, j) in one.joints.iter().enumerate() {
            let g = &grid.joints[k + c as usize * per];
            assert_eq!(g.body_a, j.body_a + c * stride, "copy {c} joint {k}");
            assert_eq!(g.body_b, if j.world { 0 } else { j.body_b + c * stride }, "copy {c} joint {k}");
        }
    }
}

/// Generated vibe-city packs (`bun run export.ts` in engine/scenes/vibe-city); missing
/// packs skip.
#[test]
fn scene_packs() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/generated/vibe-city");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skip: {} missing", dir.display());
        return;
    };
    for path in entries.filter_map(|e| e.ok().map(|e| e.path())) {
        if path.extension().is_none_or(|x| x != "json") || path.ends_with("index.json") {
            continue;
        }
        let (scene, _) = build_pack_scene(&std::fs::read(&path).unwrap()).unwrap();
        lines_follow_joints(&scene, &path.display().to_string());
    }
}

/// ScenePack v2 gives fatal limits per material (`defaults.solver.materials[m]`, the bond's `m`),
/// with no pack-wide `limits`: each bond's strength follows its material's, finite, and scales
/// with them (a weaker material, a weaker bond of the same area).
#[test]
fn pack_v2_bonds_take_their_materials_limits() {
    let pack = |m: u32| {
        serde_json::json!({
            "version": 2, "key": "v2-limits",
            "defaults": { "solver": { "materials": [
                { "name": "strong", "tensionFatal": 4.0e6, "compressionFatal": 2.0e7, "shearFatal": 6.0e6 },
                { "name": "weak", "tensionFatal": 1.0e6, "compressionFatal": 5.0e6, "shearFatal": 1.5e6 }
            ] } },
            "scenario": {
                "nodes": [
                    { "centroid": { "x": 0.0, "y": 0.5, "z": 0.0 }, "mass": 10.0, "volume": 1.0 },
                    { "centroid": { "x": 1.0, "y": 0.5, "z": 0.0 }, "mass": 10.0, "volume": 1.0 }
                ],
                "nodeSizes": [ { "x": 1.0, "y": 1.0, "z": 1.0 }, { "x": 1.0, "y": 1.0, "z": 1.0 } ],
                "bonds": [ { "node0": 0, "node1": 1, "centroid": { "x": 0.5, "y": 0.5, "z": 0.0 },
                             "normal": { "x": 1.0, "y": 0.0, "z": 0.0 }, "area": 1.0, "m": m } ]
            }
        })
        .to_string()
    };
    let strength = |m: u32| {
        let (scene, _) = build_pack_scene(pack(m).as_bytes()).unwrap();
        assert_eq!(scene.joints.len(), 1);
        scene.joints[0].strength
    };
    let (strong, weak) = (strength(0), strength(1));
    assert!(strong.iter().all(|s| s.is_finite()), "finite: {strong:?}");
    for k in 0..3 {
        assert!(weak[k] < strong[k], "component {k}: weak {} vs strong {}", weak[k], strong[k]);
    }
    assert!((strong[0] / weak[0] - 4.0).abs() < 1e-9, "tension follows the material's");
}

/// Every pack in the repo loads (the loader refuses collider kinds it does not know), and a
/// `shape` collider collides as its library hull: no pack node of kind `shape` is left a box.
#[test]
fn repo_packs_load_with_their_library_shapes() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../stress-oracle/packs");
    let mut seen = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        if !json["scenario"]["nodes"].is_array() {
            continue; // not a scene pack (the oracle's other inputs)
        }
        let (scene, info) = build_pack_scene(&bytes).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
        let shapes = json["scenario"]["nodeColliders"]
            .as_array()
            .map_or(0, |c| c.iter().filter(|c| c["kind"] == "shape").count());
        let hulls = scene
            .colliders
            .iter()
            .filter(|c| c.shape == engine_schema::ShapeType::ConvexHull as u32)
            .count();
        assert!(hulls >= shapes, "{}: {shapes} shape colliders, {hulls} hulls", path.display());
        assert_eq!(info.node_frames.len(), info.nodes, "{}", path.display());
        seen += 1;
    }
    assert!(seen >= 5, "found {seen} packs in {dir}");
}

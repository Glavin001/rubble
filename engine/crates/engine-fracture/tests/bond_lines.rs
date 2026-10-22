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

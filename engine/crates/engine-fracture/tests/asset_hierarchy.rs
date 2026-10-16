//! The Shardwright hierarchy the stress cooker walks: fragments and bonds per level,
//! parent/child links both ways, and mass conserved from level to level.

use engine_fracture::parse_fracture_glb;

fn veneer_house() -> engine_fracture::FractureAsset {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../engine-webgpu/fracture/assets/veneer-house.glb");
    parse_fracture_glb(&std::fs::read(path).expect("veneer-house.glb")).expect("parse")
}

#[test]
fn veneer_house_levels_and_links() {
    let asset = veneer_house();
    let p = &asset.physics;
    assert_eq!(p.levels, 4);
    let per_level = |level: u8| {
        let frags = p.fragments.iter().filter(|f| f.level == level).count();
        let bonds = p.bonds.iter().filter(|b| b.level == level).count();
        let world = p.bonds.iter().filter(|b| b.level == level && b.b < 0).count();
        let mass: f64 = p.fragments.iter().filter(|f| f.level == level).map(|f| f.mass as f64).sum();
        (frags, bonds, world, mass)
    };
    let expected = [(131, 454, 1), (567, 2247, 12), (3158, 12056, 153), (5905, 24364, 415)];
    let mass0 = per_level(0).3;
    for (level, &(frags, bonds, world)) in expected.iter().enumerate() {
        let (f, b, w, m) = per_level(level as u8);
        assert_eq!((f, b, w), (frags, bonds, world), "level {level}");
        assert!((m - mass0).abs() < 1e-4 * mass0, "level {level} mass {m} vs {mass0}");
    }
    // Fragment tree: every child names its parent, one level coarser.
    for f in &p.fragments {
        for c in f.child_begin..f.child_begin + f.child_count {
            let child = &p.fragments[c as usize];
            assert_eq!(child.parent, f.id as i32);
            assert_eq!(child.level, f.level + 1);
        }
        if f.parent >= 0 {
            let parent = &p.fragments[f.parent as usize];
            assert!((parent.child_begin..parent.child_begin + parent.child_count).contains(&f.id));
        }
    }
    // Bond tree: each child bond names its parent bond and sits one level finer.
    for b in &p.bonds {
        for k in b.child_begin..b.child_begin + b.child_count {
            let child = &p.bonds[p.bond_children[k as usize] as usize];
            assert_eq!(child.parent_bond, b.id as i32);
            assert_eq!(child.level, b.level + 1);
        }
        // Sliver patches (area ~1e-9 m^2, 17 at L2) carry second moments that round to
        // tiny negatives (down to -1.3e-23): the cooker clamps them at zero.
        assert!(b.area >= 0.0 && b.i_uu > -1e-20 && b.i_vv > -1e-20, "bond {}", b.id);
        let n = (b.normal[0] * b.normal[0] + b.normal[1] * b.normal[1] + b.normal[2] * b.normal[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-3, "bond {} normal length {n}", b.id);
    }
}

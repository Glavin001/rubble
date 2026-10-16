//! Real Shardwright bakes: the reader, level selection, frame conversion and
//! the row / grid / suburb layouts (formerly fracture/asset.test.mjs).

mod common;
use common::*;
use engine_fracture::math::*;
use engine_fracture::scene::*;
use engine_fracture::{build_grid, build_row, build_scene, build_suburb, parse_fracture_glb};

#[test]
fn bake_retains_hierarchy_material_ids_and_bond_topology() {
    let asset = arch();
    let p = &asset.physics;
    assert_eq!(p.name, "masonry_arch");
    assert_eq!(p.seed, "1337");
    assert_eq!(p.fragments.len(), 751);
    assert_eq!(p.bonds.len(), 2821);
    assert_eq!(p.level_fragment_count, vec![23, 165, 171, 392]);
    let bond = p.bonds.iter().find(|b| b.composition.iter().any(|c| c.interface_material >= 0)).unwrap();
    let id = bond.composition.iter().find(|c| c.interface_material >= 0).unwrap().interface_material;
    assert_eq!(asset.interface_material(id).unwrap()["id"], "lime_mortar");
    assert_eq!(p.bonds.iter().filter(|b| b.level == 0 && b.anchor).count(), 2);
    assert!(p.fragments[0].mass > 0.0);
    let meta: serde_json::Value = serde_json::from_str(&asset.metadata_json().unwrap()).unwrap();
    assert_eq!(meta["physics"]["fragments"].as_array().unwrap().len(), 751);
    assert_eq!(meta["physics"]["levelFragmentCount"][3], 392);
    assert!(meta["materialLibrary"]["material"].is_array());
}

#[test]
fn principal_basis_reconstructs_a_tensor_with_off_diagonal_terms() {
    let inertia = [4.0, 1.0, 0.3, 1.0, 2.0, 0.2, 0.3, 0.2, 3.0];
    let (q, moments) = principal_axes(&inertia).unwrap();
    for col in 0..3 {
        let mut axis = [0.0; 3];
        axis[col] = 1.0;
        let local = inverse_rotate(q, axis);
        let v = rotate(q, [local[0] * moments[0], local[1] * moments[1], local[2] * moments[2]]);
        for row in 0..3 {
            assert!((v[row] - inertia[row * 3 + col] as f64).abs() < 1e-6, "row {row} col {col}");
        }
    }
}

#[test]
fn levels_never_mix_ancestors_and_hulls_survive_the_frame_conversion() {
    let asset = arch();
    for level in 0..4u8 {
        let s = build_scene(asset, level, None).unwrap();
        assert_eq!(s.projectile_start - 1, asset.physics.level_fragment_count[level as usize]);
        for &id in &s.selected {
            let f = &asset.physics.fragments[id as usize];
            assert_eq!(f.level, level);
            assert!(f.parent < 0 || s.body_of_fragment(f.parent as u32).is_none());
            let body_id = s.body_of_fragment(id).unwrap();
            let body = s.bodies[body_id as usize];
            let c = s.colliders.iter().find(|c| c.body == body_id).unwrap();
            let v0 = collider_vertices(&s.hull_data, c, [0.0; 3], IDENTITY)[0];
            let reconstructed = add(rotate(body.q, v0), f64x3(f.com));
            let original = f64x3(asset.physics.hulls[f.hull_begin as usize].vertices[0]);
            assert!(length(sub(reconstructed, original)) < 1e-5, "level {level} fragment {id}");
        }
    }
}

#[test]
fn rejects_corrupt_plain_or_unsupported_glbs() {
    let bytes = std::fs::read(asset_path("masonry-arch.glb")).unwrap();
    let err = |b: &[u8]| parse_fracture_glb(b).err().map(|e| e.to_string()).unwrap_or_default();
    assert!(err(&bytes[..bytes.len() - 4]).contains("header or length"));
    let mut copy = bytes.clone();
    copy[0..4].fill(0);
    assert!(err(&copy).contains("header"));
    let mut bad = bytes.clone();
    let at = bad.windows(20).position(|w| w == b"SHARDWRIGHT_fracture").unwrap();
    bad[at] = 88;
    assert!(err(&bad).contains("export"));
}

#[test]
fn floor_meets_hulls_fragments_start_asleep_and_projectiles_are_inactive() {
    let s = build_scene(arch(), 0, None).unwrap();
    let mut min_y = f64::INFINITY;
    for c in s.colliders.iter().filter(|c| c.shape == SHAPE_CONVEX) {
        let b = s.bodies[c.body as usize];
        assert_eq!(b.flags, 5);
        for v in collider_vertices(&s.hull_data, c, b.p, b.q) {
            min_y = min_y.min(v[1]);
        }
    }
    assert!(min_y.abs() < 1e-6, "{min_y}");
    assert_eq!(s.bodies.len() as u32 - s.projectile_start, 16);
    assert!(s.bodies[s.projectile_start as usize..].iter().all(|b| b.flags == 0 && b.inv_mass == 0.0));
}

#[test]
fn house_imports_parts_compound_hulls_and_finer_levels() {
    let house = house();
    assert_eq!(house.physics.name, "veneer_house");
    assert_eq!(house.physics.level_fragment_count, vec![131, 567, 3158, 5905]);
    let s = build_scene(house, 0, None).unwrap();
    assert_eq!(s.selected.len(), 131);
    assert_eq!(s.hull_count(), 163);
    assert_eq!(s.bond_count, 454);
    assert!(s.extent > 6.0 && s.height > 4.0);
    assert!(s.selected.iter().any(|&f| house.physics.fragments[f as usize].hull_count > 1));
    assert!(s.skipped_hulls.is_empty());
    let fine = build_scene(house, 1, None).unwrap();
    assert_eq!(fine.selected.len(), 567);
    assert_eq!(fine.hull_count(), 2253);
    // Finer levels bake a few flat (zero-volume) sheets into compounds. They are
    // dropped; every fragment keeps real hulls, and nothing is thickened.
    for (level, flat, first) in [(2u8, 35usize, (1401u32, 3590u32)), (3, 36, (5162, 8777))] {
        let s = build_scene(house, level, None).unwrap();
        assert_eq!(s.selected.len() as u32, house.physics.level_fragment_count[level as usize]);
        assert_eq!(s.skipped_hulls.len(), flat, "level {level}: flat hulls skipped");
        assert_eq!((s.skipped_hulls[0].fragment, s.skipped_hulls[0].hull), first);
        for h in &s.skipped_hulls {
            let v = &house.physics.hulls[h.hull as usize].vertices;
            let flat_axis = (0..3).any(|k| {
                let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p[k]), b.max(p[k])));
                hi - lo < 1e-9
            });
            assert!(flat_axis, "hull {} skipped although it has volume", h.hull);
        }
        for &f in &s.selected {
            let id = s.body_of_fragment(f).unwrap();
            assert!(s.colliders.iter().any(|c| c.body == id && c.shape == SHAPE_CONVEX), "level {level}: fragment {f} lost all hulls");
        }
        let total: u32 = s.selected.iter().map(|&f| house.physics.fragments[f as usize].hull_count).sum();
        assert_eq!(s.hull_count(), total as usize - flat);
    }
}

#[test]
fn a_row_of_level_3_houses_shares_hulls_and_the_ball_rolls_through_every_house() {
    // Re-baked by Shardwright with a 16-vertex hull cap (assets/veneer-house-hull16.toml).
    let house = load("veneer-house-hull16.glb");
    assert_eq!(house.physics.level_fragment_count, vec![142, 614, 3619, 6592]);
    assert!(house.physics.hulls.iter().all(|h| h.vertices.len() <= 16), "the bake honoured its 16-vertex cap");
    let one = build_scene(&house, 3, Some(16)).unwrap();
    assert_eq!(one.skipped_hulls.len(), 23);
    let err = build_scene(&house, 3, Some(8)).err().unwrap().to_string();
    assert!(err.contains("configured limit is 8 (hard maximum 64)"), "{err}");
    let row = build_row(&one, 20);
    let frags = one.projectile_start - 1;
    let house_cols: Vec<_> = one.colliders.iter().filter(|c| c.shape == SHAPE_CONVEX).collect();
    assert_eq!(row.instance_stride, frags);
    assert_eq!(row.copies, 20);
    assert_eq!(row.projectile_start, 1 + 20 * frags);
    assert_eq!(row.bodies.len() as u32, row.projectile_start + row.projectile_count);
    assert_eq!(row.hull_count(), 20 * house_cols.len());
    assert_eq!(row.hull_data.len(), one.hull_data.len(), "every copy reuses the single hull arena");
    // Copy k is copy 0 shifted along z only, with colliders on the matching bodies.
    for k in [0, 7, 19] {
        for i in [1, frags >> 1, frags] {
            let (a, b) = (row.bodies[i as usize], row.bodies[(i + k * frags) as usize]);
            assert_eq!([b.p[0], b.p[1]], [a.p[0], a.p[1]]);
            assert!((b.p[2] - a.p[2] - k as f64 * row.row_pitch).abs() < 1e-9);
            assert_eq!(b.q, a.q);
            assert_eq!(b.flags, 5);
        }
    }
    for k in 0..20 {
        for (j, c) in house_cols.iter().enumerate().step_by(97) {
            let r = row.colliders[1 + k * house_cols.len() + j];
            assert_eq!(r.body, c.body + k as u32 * frags);
            assert_eq!(r.params, c.params);
        }
    }
    assert!(row.row_pitch > one.extent_z, "houses do not overlap");
    // The ready-made shot starts before the first house at ground level and heads
    // down the row's axis, through the middle of every house.
    let shot = row.row_shot.unwrap();
    let zmin = row.bodies[1..row.projectile_start as usize].iter().map(|b| b.p[2]).fold(f64::INFINITY, f64::min);
    assert_eq!(shot.direction, [0.0, 0.0, 1.0]);
    assert!(shot.origin[2] + row.projectile_radius < zmin);
    assert!((shot.origin[1] - row.projectile_radius).abs() < 0.05);
    assert_eq!(shot.origin[0], 0.0);
    assert!(!row.projectile_arc);
    let mass: f64 = one.selected.iter().map(|&f| house.physics.fragments[f as usize].mass as f64).sum();
    assert!(row.projectile_mass >= 5.0 * mass, "the ball carries enough momentum for the row");
    assert!(row.colliders.iter().filter(|c| c.shape == SHAPE_SPHERE).all(|c| c.params[0] == row.projectile_radius));
    assert_eq!(row.capacity, Budget::grid());
    assert_eq!((row.capacity.candidates_per_collider, row.capacity.contacts_per_collider), (8, 8));
    assert_eq!((row.capacity.max_colors, row.capacity.color_rounds), (Some(64), Some(64)));
}

#[test]
fn a_7x7_grid_offsets_copies_in_x_and_z_and_fires_down_the_middle_column() {
    let one = build_scene(house(), 2, None).unwrap();
    let grid = build_grid(&one, &GridOptions { columns: 7, rows: 7, ..GridOptions::default() });
    let frags = one.projectile_start - 1;
    assert_eq!(grid.copies, 49);
    assert_eq!(grid.instance_stride, frags);
    assert_eq!(grid.projectile_start, 1 + 49 * frags);
    assert_eq!(grid.hull_count(), 49 * one.hull_count());
    assert!(grid.column_pitch > one.extent_x && grid.row_pitch > one.extent_z, "houses do not overlap");
    for (r, c) in [(0u32, 0u32), (3, 3), (6, 6), (2, 5)] {
        let k = r * 7 + c;
        for i in [1, frags] {
            let (a, b) = (grid.bodies[i as usize], grid.bodies[(i + k * frags) as usize]);
            assert!((b.p[0] - a.p[0] - c as f64 * grid.column_pitch).abs() < 1e-9);
            assert_eq!(b.p[1], a.p[1]);
            assert!((b.p[2] - a.p[2] - r as f64 * grid.row_pitch).abs() < 1e-9);
            assert_eq!(b.q, a.q);
        }
    }
    let shot = grid.row_shot.unwrap();
    assert_eq!(shot.origin[0], 0.0);
    assert_eq!(shot.direction, [0.0, 0.0, 1.0]);
    assert!(shot.origin[2] < -grid.row_length / 2.0);
}

/// Copy footprints (x0, z0, x1, z1) from copy 0's hull vertices.
pub fn footprints(one: &Scene, s: &Scene) -> Vec<[f64; 4]> {
    let frags = (one.projectile_start - 1) as usize;
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for c in one.colliders.iter().filter(|c| c.shape == SHAPE_CONVEX) {
        let b = one.bodies[c.body as usize];
        for v in collider_vertices(&one.hull_data, c, b.p, b.q) {
            lo = [lo[0].min(v[0]), lo[1].min(v[2])];
            hi = [hi[0].max(v[0]), hi[1].max(v[2])];
        }
    }
    (0..s.copies as usize)
        .map(|k| {
            let (dx, dz) = (s.bodies[1 + k * frags].p[0] - one.bodies[1].p[0], s.bodies[1 + k * frags].p[2] - one.bodies[1].p[2]);
            [lo[0] + dx, lo[1] + dz, hi[0] + dx, hi[1] + dz]
        })
        .collect()
}

fn overlap(a: [f64; 4], b: [f64; 4]) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

#[test]
fn a_suburb_puts_houses_on_lots_between_roads_and_shoots_one_column() {
    let one = build_scene(house(), 2, None).unwrap();
    let suburb = build_suburb(&one, &SuburbOptions::default());
    let frags = one.projectile_start - 1;
    assert_eq!(suburb.copies, 48);
    assert_eq!(suburb.projectile_start, 1 + 48 * frags);
    let boxes = footprints(&one, &suburb);
    for i in 0..48 {
        for j in i + 1..48 {
            assert!(!overlap(boxes[i], boxes[j]), "houses {i} and {j} overlap");
        }
    }
    // No house sits on a road or sidewalk (6 vertices per quad, 10 floats each).
    let r = &suburb.road_vertices;
    assert!(!r.is_empty() && r.len().is_multiple_of(60));
    for q in r.chunks(60) {
        let xs: Vec<f64> = (0..6).map(|k| q[k * 10] as f64).collect();
        let zs: Vec<f64> = (0..6).map(|k| q[k * 10 + 2] as f64).collect();
        let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
        let max = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let quad = [min(&xs), min(&zs), max(&xs), max(&zs)];
        for b in &boxes {
            assert!(!overlap(quad, *b), "a house overlaps a road surface");
        }
    }
    // The shot rolls along +z and its path (ball radius) crosses one column of lots.
    let shot = suburb.row_shot.unwrap();
    let (x, rad) = (shot.origin[0], suburb.projectile_radius);
    assert_eq!(shot.direction, [0.0, 0.0, 1.0]);
    assert_eq!(boxes.iter().filter(|b| x + rad > b[0] && x - rad < b[2]).count(), 6, "the ball crosses 3 blocks x 2 lots");
}

#[test]
fn fine_levels_of_the_arch_build_and_bonds_become_joints() {
    for level in 0..4u8 {
        let s = build_scene(arch(), level, None).unwrap();
        assert!(!s.joints.is_empty());
        assert!(s.lines.len().is_multiple_of(10) && !s.vertices.is_empty());
    }
}

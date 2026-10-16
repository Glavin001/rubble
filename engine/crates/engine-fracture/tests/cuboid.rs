//! The procedural cuboid house (formerly fracture/cuboid.test.mjs).

use engine_fracture::cuboid::*;
use engine_fracture::scene::*;

/// Penetration depth of two pieces (boxes rotated about +x): the smallest
/// overlap over the 15 separating axes, or 0 when some axis separates them.
fn penetration(a: &Piece, b: &Piece) -> f64 {
    let axes = |p: &Piece| {
        let (c, s) = (p.angle.cos(), p.angle.sin());
        [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]]
    };
    let (aa, ab) = (axes(a), axes(b));
    let mut cand: Vec<[f64; 3]> = aa.iter().chain(&ab).copied().collect();
    for u in &aa {
        for v in &ab {
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if l > 1e-6 {
                cand.push(n.map(|x| x / l));
            }
        }
    }
    let (ca, cb) = (a.corners(), b.corners());
    let mut depth = f64::INFINITY;
    for n in cand {
        let proj = |c: &[[f64; 3]; 8]| {
            c.iter()
                .map(|v| v[0] * n[0] + v[1] * n[1] + v[2] * n[2])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                    (lo.min(x), hi.max(x))
                })
        };
        let ((alo, ahi), (blo, bhi)) = (proj(&ca), proj(&cb));
        let overlap = ahi.min(bhi) - alo.max(blo);
        if overlap <= 0.0 {
            return 0.0;
        }
        depth = depth.min(overlap);
    }
    depth
}

#[test]
fn box_colliders_only_no_piece_starts_inside_another_all_asleep_above_ground() {
    let pieces = cuboid_pieces(&CuboidOptions::default());
    let house = build_cuboid_house(&CuboidOptions::default()).unwrap();
    assert!(
        pieces.len() > 700,
        "a detailed house ({} pieces)",
        pieces.len()
    );
    let boxes = &house.colliders[1..house.projectile_start as usize];
    assert!(boxes
        .iter()
        .all(|c| c.shape == SHAPE_BOX && c.params[..3].iter().all(|h| *h > 0.0)));
    assert!(house.bodies[1..house.projectile_start as usize]
        .iter()
        .all(|b| b.flags == 5 && b.inv_mass > 0.0 && b.inv_i.iter().all(|x| x.is_finite())));
    for p in &pieces {
        assert!(
            p.corners()
                .iter()
                .map(|c| c[1])
                .fold(f64::INFINITY, f64::min)
                > -1e-9,
            "a piece starts below the ground"
        );
    }
    // Touching is fine (resting contact); interpenetration over 0.1 mm is a layout bug.
    let bounds: Vec<[[f64; 2]; 3]> = pieces
        .iter()
        .map(|p| {
            let c = p.corners();
            [0, 1, 2].map(|k| {
                [
                    c.iter().map(|v| v[k]).fold(f64::INFINITY, f64::min),
                    c.iter().map(|v| v[k]).fold(f64::NEG_INFINITY, f64::max),
                ]
            })
        })
        .collect();
    let (mut worst, mut pair) = (0.0, (0, 0));
    for i in 0..pieces.len() {
        for j in i + 1..pieces.len() {
            if (0..3)
                .any(|k| bounds[i][k][1] <= bounds[j][k][0] || bounds[j][k][1] <= bounds[i][k][0])
            {
                continue;
            }
            let d = penetration(&pieces[i], &pieces[j]);
            if d > worst {
                worst = d;
                pair = (i, j);
            }
        }
    }
    assert!(
        worst < 1e-4,
        "pieces {pair:?} interpenetrate by {:.2} mm",
        worst * 1000.0
    );
}

#[test]
fn cuboid_suburb_has_48_houses_and_the_shot_crosses_one_column_of_lots() {
    let one = build_cuboid_house(&CuboidOptions::default()).unwrap();
    let suburb = build_suburb(&one, &SuburbOptions::default());
    assert_eq!(suburb.copies, 48);
    let frags = one.projectile_start - 1;
    assert_eq!(suburb.projectile_start, 1 + 48 * frags);
    let (x, r) = (suburb.row_shot.unwrap().origin[0], suburb.projectile_radius);
    let hit = (0..48)
        .map(|k| suburb.bodies[(1 + k * frags) as usize].p[0] - one.bodies[1].p[0])
        .filter(|dx| x + r > dx - one.extent_x / 2.0 && x - r < dx + one.extent_x / 2.0)
        .count();
    assert_eq!(hit, 6);
}

#[test]
fn hull_variant_cooks_every_box_as_an_eight_vertex_hull() {
    let house = build_cuboid_house(&CuboidOptions {
        hulls: true,
        ..CuboidOptions::default()
    })
    .unwrap();
    let boxes = build_cuboid_house(&CuboidOptions::default()).unwrap();
    assert_eq!(
        house.hull_count(),
        boxes
            .colliders
            .iter()
            .filter(|c| c.shape == SHAPE_BOX)
            .count()
    );
    for c in house.colliders.iter().filter(|c| c.shape == SHAPE_CONVEX) {
        assert_eq!(house.hull_data[c.hull_base() as usize][0] & 255, 8);
    }
}

//! The shared closest-feature queries (`engine_schema::feature_query`, twin of
//! `slang/physics/feature_query.slang`): the shapes the surface contact had not met
//! before (a single triangle, a heightfield cell, a hull placed from the World's hull
//! records), checked against their definitions.

use engine_schema::convex::{ConvexHull, HullArena};
use engine_schema::feature_query::{
    axes, bounds, heightfield_cell, hull_world, point_contact, segment_contact, shape_pose,
    triangle_contact, SurfaceShape, PIERCE,
};
use glam::{Quat, Vec3};

fn ground_tri() -> SurfaceShape {
    // A large triangle in y = 0.
    SurfaceShape::Triangle {
        t: [
            Vec3::new(-5.0, 0.0, -5.0),
            Vec3::new(5.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 5.0),
        ],
    }
}

#[test]
fn a_triangle_shape_meets_a_panel_by_its_edge() {
    // A ridge: a small vertical triangle whose top edge runs 1 cm under a large panel,
    // far from the panel's vertices. The contact is on the panel's face (all three
    // weights), gap 1 cm, normal down toward the ridge, at the edge's middle.
    let ridge = SurfaceShape::Triangle {
        t: [
            Vec3::new(-0.2, 0.0, 0.0),
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
        ],
    };
    let x = [
        Vec3::new(-3.0, 0.01, -2.0),
        Vec3::new(3.0, 0.01, -2.0),
        Vec3::new(0.0, 0.01, 3.0),
    ];
    let hit = triangle_contact(&x, &ridge, 0.025).expect("hit");
    assert!((hit.gap - 0.01).abs() < 1e-6, "gap {}", hit.gap);
    assert!((hit.n - Vec3::NEG_Y).length() < 1e-5, "normal {:?}", hit.n);
    assert!(hit.w.iter().all(|&w| w > 0.0), "weights {:?}", hit.w);
    assert!(
        hit.q.x.abs() < 1e-5 && hit.q.z.abs() < 1e-5,
        "point {:?}",
        hit.q
    );
    // Out of the band: none.
    assert!(triangle_contact(&x, &ridge, 0.005).is_none());
}

#[test]
fn a_vertex_nearest_a_triangle_shape_is_the_nodes() {
    // A tilted element whose lowest vertex is the nearest feature: node contact's.
    let x = [
        Vec3::new(0.0, 0.01, 0.0),
        Vec3::new(0.5, 0.6, 0.0),
        Vec3::new(-0.5, 0.6, 0.2),
    ];
    assert!(triangle_contact(&x, &ground_tri(), 0.025).is_none());
    // That vertex alone takes the band contact (point contact).
    let p = point_contact(x[0], &ground_tri(), 0.025).expect("band");
    assert!((p.gap - 0.01).abs() < 1e-6 && (p.n - Vec3::NEG_Y).length() < 1e-6);
}

#[test]
fn a_beam_through_a_triangle_shape_is_held_where_it_pierces() {
    // A beam crossing the ground triangle at x = 0.25: one crossing candidate, the
    // separating axis the triangle's normal (the beam is nearest along y below it).
    let x = [Vec3::new(0.0, 0.1, 0.0), Vec3::new(1.0, -0.3, 0.0)];
    let (hit, _) = segment_contact(&x, &ground_tri(), 0.025).expect("hit");
    assert!(hit.gap <= PIERCE);
    assert!(hit.n.y.abs() > 0.999, "normal {:?}", hit.n);
    assert!(hit.w[0] > 0.0 && hit.w[1] > 0.0);
}

#[test]
fn a_heightfield_cell_is_its_two_triangles() {
    // A sloped cell: (0,0) at 0, (1,0) at 0, (0,1) at 1, (1,1) at 1 (a ramp along z).
    let c00 = Vec3::new(0.0, 0.0, 0.0);
    let c10 = Vec3::new(1.0, 0.0, 0.0);
    let c01 = Vec3::new(0.0, 1.0, 1.0);
    let c11 = Vec3::new(1.0, 1.0, 1.0);
    let [a, b] = heightfield_cell(c00, c10, c01, c11);
    // A point 1 cm above the ramp's surface along its normal, over each triangle.
    let n = Vec3::new(0.0, 1.0, -1.0).normalize();
    for (shape, at) in [
        (&a, Vec3::new(0.7, 0.3, 0.3)),
        (&b, Vec3::new(0.3, 0.7, 0.7)),
    ] {
        let x = at + n * 0.01;
        let p = point_contact(x, shape, 0.025).expect("band");
        assert!((p.gap - 0.01).abs() < 1e-5, "gap {}", p.gap);
        assert!((p.n + n).length() < 1e-4, "normal {:?}", p.n);
    }
    let (lo, hi) = bounds(&a);
    assert!(lo.cmple(Vec3::splat(-0.01 + 1e-6)).all() && hi.x >= 1.01 - 1e-6);
}

#[test]
fn a_hull_from_the_arena_meets_a_triangle_as_the_box_it_is() {
    // A unit cube cooked into the World's hull arena, placed by the strict pose, is
    // the same surface as the box shape: one face 1 cm under a triangle edge.
    let pts: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 != 0 { 0.5 } else { -0.5 },
                if i & 2 != 0 { 0.5 } else { -0.5 },
                if i & 4 != 0 { 0.5 } else { -0.5 },
            )
        })
        .collect();
    let hull = ConvexHull::from_points(&pts).expect("hull");
    let mut arena = HullArena::new();
    let base = arena.insert(&hull).expect("insert");
    let recs = arena.records();
    let head = recs[base as usize];
    let (nv, nf, ne) = (head[0] & 255, (head[0] >> 8) & 255, (head[0] >> 16) & 255);
    let f = |r: [u32; 4]| {
        Vec3::new(
            f32::from_bits(r[0]),
            f32::from_bits(r[1]),
            f32::from_bits(r[2]),
        )
    };
    let verts: Vec<Vec3> = (0..nv).map(|i| f(recs[(base + 4 + i) as usize])).collect();
    let planes: Vec<(Vec3, f32)> = (0..nf)
        .map(|i| {
            (
                f(recs[(base + 4 + nv + i) as usize]),
                f32::from_bits(recs[(base + 4 + nv + i) as usize][3]),
            )
        })
        .collect();
    let word = |w: u32| recs[(base + w / 4) as usize][(w % 4) as usize];
    let edges: Vec<[u32; 2]> = (0..ne)
        .map(|e| {
            let x = word(head[2] + e);
            [x & 255, (x >> 8) & 255]
        })
        .collect();
    let (p, r) = shape_pose(
        Vec3::new(0.0, 0.5, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Quat::IDENTITY,
    );
    let a = axes(r);
    let h = hull_world(p, &a, &verts, &edges, &planes);
    let b = SurfaceShape::Box {
        c: p,
        axes: a,
        half: Vec3::splat(0.5),
    };
    let x = [
        Vec3::new(-3.0, 1.01, -2.0),
        Vec3::new(3.0, 1.01, -2.1),
        Vec3::new(0.1, 1.02, 3.0),
    ];
    let hh = triangle_contact(&x, &h, 0.025).expect("hull hit");
    let hb = triangle_contact(&x, &b, 0.025).expect("box hit");
    assert!(
        (hh.gap - hb.gap).abs() < 1e-6 && (hh.n - hb.n).length() < 1e-6,
        "{hh:?} {hb:?}"
    );
    assert!((hh.q - hb.q).length() < 1e-5);
}

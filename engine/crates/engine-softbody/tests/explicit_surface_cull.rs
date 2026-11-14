//! The surface pass's work on a turned shape (`engine/docs/measurements/lab-mountain-perf.md`).
//!
//! A slab on a slope holds a whole car in its bounding box, so the box cull lets every
//! collision triangle and beam edge through to the narrow phase, though only those along
//! the slab's face can touch it. The vehicle lab's mountain is such slabs: there the
//! narrow phase was nearly all of the surface pass's cost. The gate counts narrow-phase
//! queries (not time): no more than the elements within reach of the slab's face. The
//! separation cull that meets it is lossless: a car run onto the slab steps bit for bit
//! as with every query made.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticShape};
use engine_softbody::explicit_surface::{segment_contact, triangle_contact, SurfaceShape};
use glam::{Quat, Vec3};

const SLOPE: f32 = 55.0;
const HALF: Vec3 = Vec3::new(15.0, 1.5, 20.0);

fn sedan() -> ExplicitBeams {
    let text = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
        minimass: 10.0,
        ..engine_softbody::procedural::CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_truck_raycast(&text, None, Vec3::ZERO, 0.0, Vec3::ZERO, settings)
            .expect("the procedural sedan");
    s.surface_contact = true;
    s.ground = -1.0e5;
    s
}

/// A slab turned `SLOPE` degrees about x whose top face passes `gap` below the car's
/// nearest node: (its outward top normal, the face plane's offset).
fn slab_under(s: &mut ExplicitBeams, gap: f32) -> (Vec3, f32) {
    let rot = Quat::from_rotation_x(SLOPE.to_radians());
    let n = rot * Vec3::Y;
    let pts: Vec<Vec3> = s.nodes.iter().map(|x| s.origin + x.pos).collect();
    let d = pts.iter().map(|&p| n.dot(p)).fold(f32::INFINITY, f32::min) - gap;
    let mid = pts.iter().fold(Vec3::ZERO, |a, &p| a + p) / pts.len() as f32;
    let on_face = mid + n * (d - n.dot(mid));
    s.shapes
        .push(StaticShape::oriented_box(on_face - n * HALF.y, HALF, rot));
    (n, d)
}

/// The car's triangles and surface segments with a point within `reach` of the plane.
fn within(s: &ExplicitBeams, n: Vec3, d: f32, reach: f32) -> usize {
    let near = |i: u32| n.dot(s.origin + s.nodes[i as usize].pos) - d <= reach;
    let tris = s
        .triangles
        .iter()
        .filter(|t| t.nodes.iter().any(|&i| near(i)))
        .count();
    let segs = s
        .surface_segs
        .as_ref()
        .expect("the surface pass ran")
        .iter()
        .filter(|&&j| {
            let b = &s.beams[j as usize];
            near(b.a) || near(b.b)
        })
        .count();
    tris + segs
}

#[test]
fn a_slab_on_a_slope_queries_only_the_elements_along_its_face() {
    let mut s = sedan();
    let (n, d) = slab_under(&mut s, 0.005);
    s.step();
    let elements = s.triangles.len() + s.surface_segs.as_ref().unwrap().len();
    let reach = s.collision_range + 0.1;
    let bound = within(&s, n, d, reach);
    println!(
        "narrow-phase queries {} (separation cull {}); elements {elements}, within {reach} m of the face {bound}; hits {}",
        s.surface_queries, s.surface_separation, s.surface_last
    );
    assert!(s.surface_last > 0, "the car touches the slab");
    assert!(
        s.surface_queries as usize <= bound,
        "{} narrow-phase queries for {bound} elements within {reach} m of the slab's face ({elements} in all)",
        s.surface_queries
    );
}

/// Every node's position and velocity, as bits.
fn bits(s: &ExplicitBeams) -> Vec<u32> {
    s.nodes
        .iter()
        .flat_map(|x| x.pos.to_array().into_iter().chain(x.vel.to_array()))
        .map(f32::to_bits)
        .collect()
}

#[test]
fn the_separation_cull_changes_nothing() {
    // The car pushed onto the slab and sliding down it: the same bits with the cull as
    // with every query made, and every element the cull skips has no contact.
    let run = |cull: bool| {
        let mut s = sedan();
        s.surface_separation = cull;
        slab_under(&mut s, 0.02);
        let rot = Quat::from_rotation_x(SLOPE.to_radians());
        let push = rot * Vec3::new(0.0, -1.0, -2.0);
        for x in &mut s.nodes {
            x.vel = push;
        }
        let mut queries = 0u64;
        let mut hits = 0u64;
        for k in 0..400 {
            s.step();
            queries += u64::from(s.surface_queries);
            hits += u64::from(s.surface_last);
            if cull && k % 50 == 0 {
                let shapes = s.surface_shapes();
                let r = s.collision_range;
                for (sh, _, _) in &shapes {
                    for t in &s.triangles {
                        let x = t.nodes.map(|i| s.origin + s.nodes[i as usize].pos);
                        if engine_schema::feature_query::separated(&x, sh, r) {
                            assert!(
                                triangle_contact(&x, sh, r).is_none(),
                                "a culled triangle in contact"
                            );
                        }
                    }
                    for &j in s.surface_segs.as_ref().unwrap() {
                        let b = &s.beams[j as usize];
                        let x = [b.a, b.b].map(|i| s.origin + s.nodes[i as usize].pos);
                        if engine_schema::feature_query::separated(&x, sh, r) {
                            assert!(
                                segment_contact(&x, sh, r).is_none(),
                                "a culled segment in contact"
                            );
                        }
                    }
                }
            }
        }
        (bits(&s), queries, hits)
    };
    let (a, qa, ha) = run(true);
    let (b, qb, hb) = run(false);
    println!("400 steps: {qa} queries with the cull, {qb} without; {ha} hits");
    assert!(ha > 0, "the car meets the slab");
    assert_eq!(ha, hb);
    assert!(a == b, "the cull changed the car's state");
    assert!(qa * 4 < qb, "the cull skips most queries: {qa} of {qb}");
}

#[test]
fn separation_is_only_beyond_the_band() {
    // A unit cube turned about two axes; points just inside the band of a face are not
    // separated, points past it (and the allowance) are.
    let rot = Quat::from_rotation_x(0.7) * Quat::from_rotation_z(0.4);
    let cube = SurfaceShape::Box {
        c: Vec3::new(3.0, -2.0, 1.0),
        axes: [rot * Vec3::X, rot * Vec3::Y, rot * Vec3::Z],
        half: Vec3::splat(0.5),
    };
    let r = 0.025;
    let up = rot * Vec3::Y;
    let at = |h: f32| {
        let o = Vec3::new(3.0, -2.0, 1.0) + up * (0.5 + h);
        [
            o + rot * Vec3::new(-0.2, 0.0, -0.1),
            o + rot * Vec3::new(0.2, 0.0, -0.1),
            o + rot * Vec3::new(0.0, 0.0, 0.2),
        ]
    };
    let sep = |x: &[Vec3]| engine_schema::feature_query::separated(x, &cube, r);
    assert!(!sep(&at(0.02)), "in the band");
    assert!(!sep(&at(r)), "at the band's edge");
    assert!(sep(&at(r + 0.01)), "past the band and the allowance");
    assert!(triangle_contact(&at(r + 0.01), &cube, r).is_none());
    // A ball: separated past its radius, the band and the allowance, from the box of
    // the element's points (the triangle's face may pass nearer the centre than its
    // vertices do).
    let ball = SurfaceShape::Sphere {
        c: Vec3::ZERO,
        radius: 0.1,
    };
    let near_face = [
        Vec3::new(-1.0, 0.12, -1.0),
        Vec3::new(1.0, 0.12, -1.0),
        Vec3::new(0.0, 0.12, 1.0),
    ];
    assert!(
        !engine_schema::feature_query::separated(&near_face, &ball, r),
        "its face is 2 cm from the ball"
    );
    assert!(triangle_contact(&near_face, &ball, r).is_some());
    let lifted = near_face.map(|p| p + Vec3::Y * (r + 0.01));
    assert!(engine_schema::feature_query::separated(&lifted, &ball, r));
    assert!(triangle_contact(&lifted, &ball, r).is_none());
}

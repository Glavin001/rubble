//! The surface pass's work on a turned shape (`engine/docs/measurements/lab-mountain-perf.md`).
//!
//! A slab on a slope holds a whole car in its bounding box, so the box cull lets every
//! collision triangle and beam edge through to the narrow phase, though only those along
//! the slab's face can touch it. The vehicle lab's mountain is such slabs: there the
//! narrow phase was nearly all of the surface pass's cost. The gate counts narrow-phase
//! queries (not time): no more than the elements within reach of the slab's face.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticShape};
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
        "narrow-phase queries {}; elements {elements}, within {reach} m of the face {bound}; hits {}",
        s.surface_queries, s.surface_last
    );
    assert!(s.surface_last > 0, "the car touches the slab");
    assert!(
        s.surface_queries as usize <= bound,
        "{} narrow-phase queries for {bound} elements within {reach} m of the slab's face ({elements} in all)",
        s.surface_queries
    );
}

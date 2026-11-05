//! Static shapes beyond the axis-aligned box (`StaticShape`, Rubble only): the box
//! contact rule in a rotated frame (a node on a 20° ramp holds or slides as its
//! friction dictates), the radial rule of a cylinder, capsule and sphere, and an
//! identity-rotation box being the axis-aligned box bit for bit.

use engine_softbody::explicit_beams::*;
use glam::{Quat, Vec3};

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
};

/// One unit-mass node with ground contact on (the ground far below), no drag.
fn one(p: Vec3, v: Vec3, friction: f32) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -1.0e3;
    s.nodes.push(Node {
        pos: p,
        vel: v,
        force: Vec3::ZERO,
        mass: 1.0,
        friction,
        flags: 0,
    });
    s
}

const SLOPE: f32 = 20.0 * std::f32::consts::PI / 180.0;

/// A 4 m × 1 m × 2 m box turned 20° about z (its top face rises toward +x), and
/// a node resting 1 mm above the middle of its top face.
fn ramp(friction: f32) -> (ExplicitBeams, Vec3, Vec3) {
    let q = Quat::from_rotation_z(SLOPE);
    let half = Vec3::new(2.0, 0.5, 1.0);
    let shape = StaticShape::oriented_box(Vec3::ZERO, half, q);
    let up = q * Vec3::Y;
    let downhill = -(q * Vec3::X);
    let mut s = one(up * (half.y + 0.001), Vec3::ZERO, friction);
    s.shapes.push(shape);
    s.finish();
    (s, up, downhill)
}

/// Height above the ramp's top face and distance moved downhill.
fn on_ramp(s: &ExplicitBeams, up: Vec3, downhill: Vec3) -> (f32, f32) {
    let p = s.nodes[0].pos;
    (p.dot(up) - 0.5, p.dot(downhill))
}

#[test]
fn a_node_on_a_20_degree_ramp_holds_with_static_friction() {
    // Concrete's static coefficient 1.2 times the node's 1.0 is over tan 20° ≈ 0.36.
    let (mut s, up, downhill) = ramp(1.0);
    for _ in 0..4000 {
        s.step();
    }
    let (h, slid) = on_ramp(&s, up, downhill);
    assert!(h > -0.01 && h < 0.01, "left the face: {h} m");
    assert!(slid.abs() < 0.02, "slid {slid} m in 2 s");
    assert!(s.nodes[0].vel.length() < 0.05, "{}", s.nodes[0].vel);
}

#[test]
fn a_node_on_a_20_degree_ramp_slides_down_with_low_friction() {
    // 1.2 × 0.2 = 0.24 static, 0.15 sliding: under tan 20°, so it slides and keeps
    // accelerating along the face (g (sin 20° − 0.15 cos 20°) ≈ 1.97 m/s² at most).
    let (mut s, up, downhill) = ramp(0.2);
    for _ in 0..2000 {
        s.step();
    }
    let (h, slid) = on_ramp(&s, up, downhill);
    assert!(h > -0.01 && h < 0.01, "left the face: {h} m");
    assert!(slid > 0.3 && slid < 1.0, "slid {slid} m in 1 s");
    let v = s.nodes[0].vel;
    // Along the face, downhill.
    assert!(v.dot(downhill) > 0.5, "{v}");
    assert!(v.dot(up).abs() < 0.05, "{v}");
}

#[test]
fn oriented_box_faces_are_picked_nearest_first_in_the_box_frame() {
    let q = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(0.3);
    let half = Vec3::new(1.0, 0.5, 0.25);
    let c = Vec3::new(3.0, 1.0, -2.0);
    let s = StaticShape::oriented_box(c, half, q);
    let b = s.bounds();
    let at = |l: Vec3| c + q * l;
    let near = |n: Option<Vec3>, want: Vec3| {
        let n = n.expect("inside");
        assert!((n - q * want).length() < 1e-6, "{n} vs {}", q * want);
    };
    near(
        shape_face(&s, &b, at(Vec3::new(0.0, 0.0, -0.2))),
        Vec3::NEG_Z,
    );
    near(shape_face(&s, &b, at(Vec3::new(0.95, 0.0, 0.0))), Vec3::X);
    near(shape_face(&s, &b, at(Vec3::new(0.0, 0.45, 0.0))), Vec3::Y);
    assert_eq!(shape_face(&s, &b, at(Vec3::new(0.0, 0.0, 0.3))), None);
    assert_eq!(shape_face(&s, &b, at(Vec3::new(1.1, 0.0, 0.0))), None);
}

#[test]
fn a_node_pushed_into_a_cylinder_exits_radially() {
    // A frictionless node at 5 m/s along +x, 0.3 m off the axis of a 0.5 m pole:
    // the contact removes only the radial part of its velocity (along the normal
    // through the entry point), the tangential part stays.
    let c = Vec3::new(0.0, 1.0, 0.0);
    let pole = StaticShape::cylinder(c, 0.5, 1.0);
    let mut s = one(Vec3::new(-0.45, 1.0, 0.3), Vec3::new(5.0, 0.0, 0.0), 0.0);
    s.shapes.push(pole);
    s.finish();
    let b = pole.bounds();
    let mut first = None;
    let mut deepest = 0.0f32;
    for _ in 0..400 {
        let before = s.nodes[0];
        let normal = shape_face(&pole, &b, s.origin + before.pos);
        s.step();
        let d = s.nodes[0].pos - c;
        deepest = deepest.max(0.5 - (d.x * d.x + d.z * d.z).sqrt());
        if let (Some(n), None) = (normal, first) {
            // The step's velocity change in the ground plane is along the radial
            // normal at the node, outward.
            let dv = s.nodes[0].vel - before.vel;
            let dv = Vec3::new(dv.x, 0.0, dv.z);
            let r = Vec3::new(before.pos.x - c.x, 0.0, before.pos.z - c.z).normalize();
            assert!((n - r).length() < 1e-5, "normal {n} not radial {r}");
            assert!(dv.normalize().dot(r) > 0.99999, "{dv} vs {r}");
            first = Some(r);
        }
    }
    let r = first.expect("never touched the pole");
    let t = Vec3::new(-r.z, 0.0, r.x);
    let v = s.nodes[0].vel;
    // Stopped going in, kept its tangential speed (|5 · t.x|), never deep.
    assert!(deepest < 0.01, "went {deepest} m in");
    let d = s.nodes[0].pos - c;
    let now = Vec3::new(d.x, 0.0, d.z).normalize();
    assert!(v.dot(now) > -1e-3, "still going in: {v}");
    assert!(v.length() > 0.9 * (5.0 * t.x).abs(), "{v}");
}

#[test]
fn cylinder_caps_and_round_shapes_push_out_along_their_normals() {
    let c = Vec3::new(1.0, 2.0, 3.0);
    let cyl = StaticShape::cylinder(c, 0.5, 1.0);
    let b = cyl.bounds();
    // Near the top cap: +y; near the bottom: −y; near the side: radial.
    assert_eq!(
        shape_face(&cyl, &b, c + Vec3::new(0.1, 0.95, 0.0)),
        Some(Vec3::Y)
    );
    assert_eq!(
        shape_face(&cyl, &b, c + Vec3::new(0.1, -0.95, 0.0)),
        Some(Vec3::NEG_Y)
    );
    let n = shape_face(&cyl, &b, c + Vec3::new(0.0, 0.2, -0.45)).unwrap();
    assert!((n - Vec3::NEG_Z).length() < 1e-6, "{n}");
    assert_eq!(shape_face(&cyl, &b, c + Vec3::new(0.36, 0.0, 0.36)), None);
    assert_eq!(shape_face(&cyl, &b, c + Vec3::new(0.0, 1.01, 0.0)), None);

    let sph = StaticShape::sphere(c, 0.5);
    let n = shape_face(&sph, &sph.bounds(), c + Vec3::new(0.2, 0.2, 0.1)).unwrap();
    assert!((n - Vec3::new(2.0, 2.0, 1.0) / 3.0).length() < 1e-6, "{n}");
    assert_eq!(
        shape_face(&sph, &sph.bounds(), c + Vec3::new(0.3, 0.3, 0.3)),
        None
    );

    // A capsule along x: the normal is from the nearest axis point (the end cap
    // past an end).
    let cap = StaticShape::capsule(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.25);
    let cb = cap.bounds();
    let n = shape_face(&cap, &cb, Vec3::new(0.3, 0.2, 0.0)).unwrap();
    assert!((n - Vec3::Y).length() < 1e-6, "{n}");
    let n = shape_face(&cap, &cb, Vec3::new(1.1, 0.0, 0.1)).unwrap();
    assert!(
        (n - Vec3::new(1.0, 0.0, 1.0).normalize()).length() < 1e-6,
        "{n}"
    );
    assert_eq!(shape_face(&cap, &cb, Vec3::new(0.0, 0.0, 0.3)), None);
}

#[test]
fn an_identity_rotation_box_is_the_axis_aligned_box_bit_for_bit() {
    let c = Vec3::new(0.3, 0.7, 1.1);
    let half = Vec3::new(0.6, 0.9, 0.5);
    let run = |oriented: bool| {
        let mut s = one(Vec3::new(0.1, 1.2, 0.0), Vec3::new(0.4, -1.0, 6.0), 0.8);
        if oriented {
            s.shapes
                .push(StaticShape::oriented_box(c, half, Quat::IDENTITY));
        } else {
            s.boxes.push(StaticBox::centered(c, half));
        }
        s.finish();
        let mut out = Vec::new();
        for _ in 0..600 {
            s.step();
            let n = s.nodes[0];
            out.push((
                n.pos.to_array().map(f32::to_bits),
                n.vel.to_array().map(f32::to_bits),
            ));
        }
        out
    };
    let a = run(false);
    assert_eq!(a, run(true));
    // It did hit the box.
    let v = f32::from_bits(a[599].1[2]);
    assert!(v < 5.0, "never touched: vz {v}");
}

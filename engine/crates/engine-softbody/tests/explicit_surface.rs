//! Surface contact (`ExplicitBeams::surface_contact`, jbeam-semantics-spec.md §38):
//! collision triangles against rigid shapes by their surface. The narrow phase against
//! each shape kind; a pole between two node lines is caught by the skin; the equal and
//! opposite force on a free body conserves momentum to the fixed point's rounding; a
//! panel resting on a bar narrower than its node spacing gains no energy.

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape, STEP,
};
use engine_softbody::explicit_rigid::{
    hull_from_planes, ProxyGeom, ProxySet, ProxyShape, RigidProxy,
};
use engine_softbody::explicit_surface::{triangle_contact, SurfaceShape, PIERCE};
use glam::{Quat, Vec3};

const R: f32 = 0.025;

/// A 0.6 m right triangle in the z = 0 plane, normal +z, no vertex in the strip x ∈ ±0.1.
fn tri() -> [Vec3; 3] {
    [
        Vec3::new(-0.3, -0.3, 0.0),
        Vec3::new(0.3, -0.3, 0.0),
        Vec3::new(0.3, 0.3, 0.0),
    ]
}

fn pole_box(z: f32) -> SurfaceShape {
    // A 0.2 m square pole along y over the whole triangle, its −z face at `z`.
    SurfaceShape::Box {
        c: Vec3::new(0.0, 0.0, z + 0.1),
        axes: [Vec3::X, Vec3::Y, Vec3::Z],
        half: Vec3::new(0.1, 1.0, 0.1),
    }
}

fn interior(w: [f32; 3]) -> bool {
    w.iter().all(|&x| x > 0.0) && (w.iter().sum::<f32>() - 1.0).abs() < 1e-5
}

#[test]
fn a_triangle_meets_a_box_by_its_face() {
    let h = triangle_contact(&tri(), &pole_box(0.01), R).expect("in the band");
    assert!((h.gap - 0.01).abs() < 1e-6, "gap {}", h.gap);
    assert!((h.n - Vec3::Z).length() < 1e-5, "normal {:?}", h.n);
    // The pole's strip across the triangle (x in ±0.1, y from −0.3 to x): its middle.
    assert!(interior(h.w), "{:?}", h.w);
    assert!(
        h.q.x.abs() < 1e-6 && (h.q.y + 0.15).abs() < 1e-5,
        "{:?}",
        h.q
    );
    assert!(
        triangle_contact(&tri(), &pole_box(0.03), R).is_none(),
        "past the band"
    );
    // Pushed 1 cm through: touching, the normal the axis of least overlap (+z), gap 0.
    let h = triangle_contact(&tri(), &pole_box(-0.01), R).expect("crossing");
    assert!(h.gap <= PIERCE && (h.n - Vec3::Z).length() < 1e-5, "{h:?}");
    // From the other side, the normal flips.
    let below = SurfaceShape::Box {
        c: Vec3::new(0.0, 0.0, -0.11),
        axes: [Vec3::X, Vec3::Y, Vec3::Z],
        half: Vec3::new(0.1, 1.0, 0.1),
    };
    let h = triangle_contact(&tri(), &below, R).expect("below");
    assert!(
        (h.n + Vec3::Z).length() < 1e-5 && (h.gap - 0.01).abs() < 1e-6,
        "{h:?}"
    );
}

#[test]
fn a_shape_nearest_a_vertex_is_the_nodes() {
    // A cube 1 cm beyond vertex 2 (0.3, 0.3, 0): the nearest feature is the vertex.
    let cube = SurfaceShape::Box {
        c: Vec3::new(0.3, 0.41, 0.0),
        axes: [Vec3::X, Vec3::Y, Vec3::Z],
        half: Vec3::splat(0.1),
    };
    assert!(triangle_contact(&tri(), &cube, R).is_none());
    let s = SurfaceShape::Sphere {
        c: Vec3::new(0.3, 0.35, 0.0),
        radius: 0.04,
    };
    assert!(triangle_contact(&tri(), &s, R).is_none());
}

#[test]
fn a_triangle_meets_a_turned_box_by_an_edge() {
    // The pole turned 45° about y: its edge (x = 0, z = 0.01) leads.
    let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
    let m = glam::Mat3::from_quat(q);
    let half = Vec3::new(0.1, 1.0, 0.1);
    let c = Vec3::new(0.0, 0.0, 0.01 + 0.1 * std::f32::consts::SQRT_2);
    let s = SurfaceShape::Box {
        c,
        axes: [m.x_axis, m.y_axis, m.z_axis],
        half,
    };
    let h = triangle_contact(&tri(), &s, R).expect("edge in the band");
    assert!((h.gap - 0.01).abs() < 1e-5, "gap {}", h.gap);
    assert!((h.n - Vec3::Z).length() < 1e-4, "normal {:?}", h.n);
    assert!(interior(h.w) && h.q.x.abs() < 1e-5, "{h:?}");
}

#[test]
fn a_triangle_meets_a_capsule_a_sphere_a_cylinder_and_a_hull() {
    let x = tri();
    let cap = SurfaceShape::Capsule {
        a: Vec3::new(-1.0, -0.1, 0.03),
        b: Vec3::new(1.0, -0.1, 0.03),
        radius: 0.02,
    };
    let h = triangle_contact(&x, &cap, R).expect("capsule");
    assert!(
        (h.gap - 0.01).abs() < 1e-6 && (h.n - Vec3::Z).length() < 1e-5 && interior(h.w),
        "{h:?}"
    );
    assert!(
        (h.q.y + 0.1).abs() < 1e-5 && (h.q.x - 0.1).abs() < 1e-5,
        "{:?}",
        h.q
    );
    let sph = SurfaceShape::Sphere {
        c: Vec3::new(0.0, -0.05, 0.05),
        radius: 0.04,
    };
    let h = triangle_contact(&x, &sph, R).expect("sphere");
    assert!(
        (h.gap - 0.01).abs() < 1e-6 && (h.n - Vec3::Z).length() < 1e-5 && interior(h.w),
        "{h:?}"
    );
    // A vertical cylinder against a vertical triangle (the x–y plane): its side.
    let xv = [
        Vec3::new(-0.3, 0.0, 0.0),
        Vec3::new(0.3, 0.0, 0.0),
        Vec3::new(0.3, 0.6, 0.0),
    ];
    let cyl = SurfaceShape::Cylinder {
        c: Vec3::new(0.0, 0.3, 0.11),
        radius: 0.1,
        half_height: 1.0,
    };
    let h = triangle_contact(&xv, &cyl, R).expect("cylinder");
    assert!(
        (h.gap - 0.01).abs() < 1e-5 && (h.n - Vec3::Z).length() < 1e-5 && interior(h.w),
        "{h:?}"
    );
    // The pole as a hull (its six planes): the box's contact.
    let planes = vec![
        (Vec3::NEG_X, 0.1),
        (Vec3::X, 0.1),
        (Vec3::NEG_Y, 1.0),
        (Vec3::Y, 1.0),
        (Vec3::NEG_Z, -0.01),
        (Vec3::Z, 0.21),
    ];
    let (verts, edges) = hull_from_planes(&planes);
    assert_eq!((verts.len(), edges.len()), (8, 12));
    let hull = SurfaceShape::Hull {
        verts,
        edges,
        planes,
    };
    let hb = triangle_contact(&x, &pole_box(0.01), R).unwrap();
    let hh = triangle_contact(&x, &hull, R).expect("hull");
    assert!(
        (hh.gap - hb.gap).abs() < 1e-6 && (hh.n - hb.n).length() < 1e-6,
        "{hh:?} vs {hb:?}"
    );
    assert!((hh.q - hb.q).length() < 1e-5, "{hh:?} vs {hb:?}");
}

const QUIET: ExplicitBeamsSettings = ExplicitBeamsSettings {
    drag: false,
    turbulence: false,
    fast_inv_sqrt: true,
    triangle_throttle: true,
    centred_damping: false,
};

/// A 0.3 m × 0.5 m plate (node lines at z = ±0.25, braced), its two collision
/// triangles, at `pos` moving at `vel`, in the BeamNG profile's contact rules.
fn plate(pos: Vec3, vel: Vec3) -> ExplicitBeams {
    let text = "surface plate\nglobals\n40, 0, tracks/semi\nnodes\n0, 0.0, 0.0, -0.25\n1, 0.0, 0.0, 0.25\n\
2, 0.3, 0.0, -0.25\n3, 0.3, 0.0, 0.25\nbeams\n0, 1\n2, 3\n0, 2\n1, 3\n0, 3\n1, 2\n\
submesh\ncab\n0, 1, 3, c\n0, 3, 2, c\nend\n";
    let mut s = ExplicitBeams::from_truck(text, pos, vel, QUIET).unwrap();
    s.roll_tyres(vel);
    s.two_body_contact = true;
    s.fixed_contact = true;
    s.dissipative_contact = true;
    s.collision_range = R;
    s
}

/// The pole between the plate's node lines (0.2 m square at x = −0.5, z = 0).
fn pole() -> StaticBox {
    StaticBox::centered(Vec3::new(-0.5, 2.0, 0.0), Vec3::new(0.1, 2.0, 0.1))
}

fn mean_x(s: &ExplicitBeams) -> f32 {
    s.positions().iter().map(|p| p.x).sum::<f32>() / s.nodes.len() as f32
}

#[test]
fn a_pole_between_two_node_lines_is_caught_by_the_skin() {
    let run = |surface: bool| {
        let mut s = plate(Vec3::new(0.5, 1.0, 0.0), Vec3::new(-10.0, 0.0, 0.0));
        s.boxes.push(pole());
        s.surface_contact = surface;
        let mut deepest = f32::MAX;
        for _ in 0..600 {
            s.step();
            // The plate's front edge (nodes 0, 1) against the pole's +x face (x = −0.4).
            let f = 0.5 * (s.positions()[0].x + s.positions()[1].x);
            deepest = deepest.min(f + 0.4);
        }
        (s, deepest)
    };
    let (off, _) = run(false);
    assert!(
        mean_x(&off) < -2.0,
        "node-only contact passes: {}",
        mean_x(&off)
    );
    let (on, deepest) = run(true);
    assert!(mean_x(&on) > -0.5, "caught: {}", mean_x(&on));
    // The front edge's middle never more than 2 cm into the pole.
    assert!(deepest > -0.02, "deepest {deepest}");
}

#[test]
fn the_body_takes_the_equal_and_opposite_force() {
    // A free 20 kg block in the plate's path between its node lines.
    let (c, half) = (Vec3::new(-0.5, 1.0, 0.0), Vec3::new(0.1, 0.3, 0.1));
    let mass = 20.0f32;
    let i = Vec3::new(
        mass / 3.0 * (half.y * half.y + half.z * half.z),
        mass / 3.0 * (half.x * half.x + half.z * half.z),
        mass / 3.0 * (half.x * half.x + half.y * half.y),
    );
    let set = ProxySet {
        bodies: vec![RigidProxy {
            body: 0,
            pos: c,
            rot: Quat::IDENTITY,
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass: 1.0 / mass,
            inv_inertia: Vec3::ONE / i,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(
            0,
            Vec3::ZERO,
            Quat::IDENTITY,
            ProxyGeom::Box { half },
            0,
            None,
        )],
        frozen: false,
    };
    let mut s = plate(Vec3::new(0.5, 1.0, 0.0), Vec3::new(-10.0, 0.0, 0.0));
    s.surface_contact = true;
    let mut set = Box::new(set);
    set.begin_window();
    s.rigid = Some(set);
    // (In f64: the measure's own f32 sums round by more than the bound.)
    let momentum = |s: &ExplicitBeams| {
        let p: glam::DVec3 = s
            .nodes
            .iter()
            .map(|n| n.vel.as_dvec3() * f64::from(n.mass))
            .sum();
        let b = &s.rigid.as_ref().unwrap().bodies[0];
        (p + b.vel.as_dvec3() / f64::from(b.inv_mass)).as_vec3()
    };
    let m_plate: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let p0 = momentum(&s);
    let steps = 600;
    let mut hits = 0;
    for _ in 0..steps {
        s.step();
        hits += s.surface_last;
    }
    // Two more steps: the last reactions integrated on both sides.
    let saved = s.surface_contact;
    s.surface_contact = false;
    s.step();
    s.step();
    s.surface_contact = saved;
    let b = s.rigid.as_ref().unwrap().bodies[0].clone();
    assert!(
        hits > 0 && b.vel.x < -1.0,
        "the block was pushed: {hits} hits, {:?}",
        b.vel
    );
    // Gravity on the nodes only (the block is not weighed), every step but the first
    // (the spawn's force slot holds none).
    let g = Vec3::new(0.0, m_plate * -9.807 * STEP * (steps + 1) as f32, 0.0);
    let dp = momentum(&s) - g - p0;
    println!(
        "plate {m_plate} kg, block v {:?}, momentum error {dp:?}",
        b.vel
    );
    // The fixed point's rounding: each hit's share to each node and the body's sum
    // round by at most 1/64 N a component, over the step.
    let bound = hits as f32 * 4.0 * (1.0 / 64.0) * STEP * 3f32.sqrt();
    // Vertically, gravity's increments round in each node's f32 velocity as well (half an
    // ulp of its speed, at most 4 m/s here, a step a node).
    let gravity_rounding = (steps + 1) as f32 * m_plate * 0.5 * (4.0f32 * f32::EPSILON);
    assert!(
        dp.x.abs() + dp.z.abs() < bound + 1e-3 && dp.y.abs() < bound + gravity_rounding + 1e-3,
        "momentum error {dp:?} of {p0:?} (bound {bound}, gravity's rounding {gravity_rounding})"
    );
}

fn energy(s: &ExplicitBeams) -> f64 {
    let strain: f64 = s
        .beams
        .iter()
        .filter(|b| !b.broken)
        .map(|b| {
            let l = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).length();
            0.5 * f64::from(b.k) * f64::from(l - b.rest).powi(2)
        })
        .sum();
    strain
        + s.nodes
            .iter()
            .map(|n| {
                let m = f64::from(n.mass);
                0.5 * m * f64::from(n.vel.length_squared())
                    + m * 9.807 * f64::from((s.origin + n.pos).y)
            })
            .sum::<f64>()
}

#[test]
fn a_plate_resting_on_a_bar_narrower_than_its_nodes_gains_no_energy() {
    // A 4 cm bar along x under the plate's middle (z = 0), its top at y = 0.5; the
    // plate set down 1 cm above the band, at rest. Its nodes, 0.25 m to either side
    // of the bar, would fall past it: only the triangles hold it.
    let top = 0.5;
    let mut s = plate(Vec3::new(0.0, top + R + 0.01, 0.0), Vec3::ZERO);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.15, top - 0.25, 0.0),
        Vec3::new(1.0, 0.25, 0.02),
    ));
    s.surface_contact = true;
    // The energy measure (kinetic, weight, beam strain) at one step's state: its
    // vibration terms swing by about a joule step to step (positions and velocities
    // half a step apart), so it is compared in means over 0.1 s windows from the
    // first contact on.
    let mut windows: Vec<f64> = Vec::new();
    let (mut sum, mut n, mut low, mut started) = (0.0f64, 0u32, f32::MAX, false);
    for _ in 0..8000 {
        s.step();
        started |= s.surface_last > 0;
        if !started {
            continue;
        }
        sum += energy(&s);
        n += 1;
        if n == 200 {
            windows.push(sum / 200.0);
            (sum, n) = (0.0, 0);
        }
        low = low.min(s.positions().iter().map(|p| p.y).fold(f32::MAX, f32::min));
    }
    let ke: f64 = s
        .nodes
        .iter()
        .map(|n| 0.5 * f64::from(n.mass) * f64::from(n.vel.length_squared()))
        .sum();
    println!(
        "window means {:?}, lowest {low}, kinetic {ke}",
        windows.iter().map(|e| e - windows[0]).collect::<Vec<_>>()
    );
    assert!(low > top, "sank to {low}");
    // Never above the most it has held before, past the measure's rounding.
    let mut most = windows[0];
    for &e in &windows[1..] {
        assert!(e <= most + 0.02, "energy rose to {e} from {most}");
        most = most.max(e);
    }
    assert!(ke < 0.01, "still moving: {ke} J");
}

#[test]
fn off_is_bit_for_bit_node_contact() {
    // With the flag off nothing changes; on with no shape near the skin, likewise.
    let mut a = plate(Vec3::new(0.5, 1.0, 0.0), Vec3::new(-10.0, 0.0, 0.0));
    a.boxes.push(pole());
    let mut b = a.clone();
    b.surface_contact = true;
    b.boxes[0] = StaticBox::centered(Vec3::new(-0.5, 2.0, 5.0), Vec3::new(0.1, 2.0, 0.1));
    a.boxes[0] = b.boxes[0];
    b.shapes
        .push(StaticShape::sphere(Vec3::new(0.0, -5.0, 0.0), 0.5));
    a.shapes
        .push(StaticShape::sphere(Vec3::new(0.0, -5.0, 0.0), 0.5));
    for _ in 0..600 {
        a.step();
        b.step();
    }
    assert_eq!(a.positions(), b.positions());
}

#[test]
fn a_beam_meets_a_box_by_its_edges_and_its_ends_are_the_nodes() {
    use engine_softbody::explicit_surface::segment_contact;
    // A 0.6 m beam along x 1 cm under the pole's −z face: the two edges it passes
    // under are nearest, one contact at their middle shared by both nodes.
    let x = [Vec3::new(-0.3, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.0)];
    let h = segment_contact(&x, &pole_box(0.01), R)
        .expect("in the band")
        .0;
    assert!(
        (h.gap - 0.01).abs() < 1e-6 && (h.n - Vec3::Z).length() < 1e-5,
        "{h:?}"
    );
    assert!((h.w[0] - 0.5).abs() < 1e-5 && (h.w[1] - 0.5).abs() < 1e-5 && h.w[2] == 0.0);
    assert!(h.q.length() < 1e-5, "{h:?}");
    // A beam whose end is nearest (under the face): the node's.
    let end = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -0.5)];
    assert!(segment_contact(&end, &pole_box(0.01), R).is_none());
    assert!(
        segment_contact(&x, &pole_box(0.03), R).is_none(),
        "past the band"
    );
    let cap = SurfaceShape::Capsule {
        a: Vec3::new(0.0, -1.0, 0.03),
        b: Vec3::new(0.0, 1.0, 0.03),
        radius: 0.02,
    };
    let h = segment_contact(&x, &cap, R).expect("capsule").0;
    assert!(
        (h.gap - 0.01).abs() < 1e-6 && (h.n - Vec3::Z).length() < 1e-5,
        "{h:?}"
    );
    let sph = SurfaceShape::Sphere {
        c: Vec3::new(0.1, 0.0, 0.05),
        radius: 0.04,
    };
    let h = segment_contact(&x, &sph, R).expect("sphere").0;
    assert!(
        (h.gap - 0.01).abs() < 1e-6 && (h.q.x - 0.1).abs() < 1e-6,
        "{h:?}"
    );
    let cyl = SurfaceShape::Cylinder {
        c: Vec3::new(0.0, 0.0, 0.11),
        radius: 0.1,
        half_height: 1.0,
    };
    let h = segment_contact(&x, &cyl, R).expect("cylinder").0;
    assert!(
        (h.gap - 0.01).abs() < 1e-5 && (h.n - Vec3::Z).length() < 1e-4,
        "{h:?}"
    );
}

#[test]
fn a_beam_through_a_box_is_held_at_both_ends_of_its_run() {
    use engine_softbody::explicit_surface::segment_contact;
    // A beam across the pole, tilted: 2 cm inside its −z face at x = +0.1, 0.5 cm at
    // x = −0.1. The normal is the least overlap: across the beam (beam × y, mostly
    // +z); its run's ends are equally deep along it, so one contact at each (x = ±0.1).
    let x = [Vec3::new(-0.3, 0.0, -0.0025), Vec3::new(0.3, 0.0, 0.02)];
    let (a, b) = segment_contact(&x, &pole_box(0.0), R).expect("crossing");
    let b = b.expect("a second contact");
    for h in [a, b] {
        assert!(
            h.gap <= PIERCE && h.n.z > 0.99 && h.n.dot(x[1] - x[0]).abs() < 1e-5,
            "{h:?}"
        );
        assert!(h.w[0] > 0.0 && h.w[1] > 0.0);
    }
    assert!(
        (a.q.x + 0.1).abs() < 1e-4 && (b.q.x - 0.1).abs() < 1e-4,
        "{a:?} {b:?}"
    );
}

#[test]
fn a_node_in_the_band_is_met_by_point_contact() {
    use engine_softbody::explicit_surface::point_contact;
    let h = point_contact(Vec3::ZERO, &pole_box(0.01), R).expect("band");
    assert!(
        (h.gap - 0.01).abs() < 1e-6 && (h.n - Vec3::Z).length() < 1e-6,
        "{h:?}"
    );
    assert_eq!(h.w, [1.0, 0.0, 0.0]);
    assert!(
        point_contact(Vec3::new(0.0, 0.0, 0.05), &pole_box(0.01), R).is_none(),
        "inside"
    );
    assert!(
        point_contact(Vec3::new(0.0, 0.0, -0.05), &pole_box(0.01), R).is_none(),
        "past"
    );
}

#[test]
fn the_segments_are_the_beams_on_no_collision_triangle() {
    use engine_softbody::explicit_surface::surface_segments;
    // The plate: its triangles' five edges are theirs; the other diagonal 1–2 is a
    // segment.
    let s = plate(Vec3::ZERO, Vec3::ZERO);
    let segs = surface_segments(&s);
    assert_eq!(segs.len(), 1, "{segs:?}");
    let b = s.beams[segs[0] as usize];
    assert_eq!((b.a.min(b.b), b.a.max(b.b)), (1, 2));
}

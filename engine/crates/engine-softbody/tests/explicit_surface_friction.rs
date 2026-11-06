//! Surface contact's friction (jbeam-semantics-spec.md §38.8): a panel held by its skin on
//! two capsules or two bars between its node lines (no node touches them) and pushed along
//! them sticks under μk·N and slides past μs·N; set sliding, it stops within the friction law's
//! distance; on an incline it stays below the friction slope. Node contact on a wide box
//! is printed beside it.

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape, GRAVITY,
};
use glam::{Quat, Vec3};

const QUIET: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: false,
    centred_damping: false,
};
const R: f32 = 0.025;

/// A 0.6 m by 0.5 m plate in y = `y` (two triangles, its nodes at the corners).
fn plate(y: f32) -> ExplicitBeams {
    let text = "friction plate\nglobals\n40, 0, tracks/semi\nnodes\n0, -0.3, 0.0, -0.25\n1, -0.3, 0.0, 0.25\n\
2, 0.3, 0.0, -0.25\n3, 0.3, 0.0, 0.25\nbeams\n0, 1\n2, 3\n0, 2\n1, 3\n0, 3\n1, 2\n\
submesh\ncab\n0, 1, 3, c\n0, 3, 2, c\nend\n";
    let mut s = ExplicitBeams::from_truck(text, Vec3::new(0.0, y, 0.0), Vec3::ZERO, QUIET).unwrap();
    s.ground = -1.0e4;
    s.two_body_contact = true;
    s.fixed_contact = true;
    s.dissipative_contact = true;
    s.collision_range = R;
    s
}

#[derive(Clone, Copy, Debug)]
enum Support {
    /// Node contact: the nodes rest on a wide box.
    Nodes,
    /// Surface contact: the triangles rest on two 4 cm capsules along z (x = ±0.15),
    /// between its node lines (x = ±0.3).
    Capsule,
    /// Surface contact: on two 4 cm box bars along z.
    Bar,
}

/// The plate settled on its support, then pushed along x with `push` × its weight for
/// 0.2 s: how far it slid (m).
fn slide(support: Support, push: f32) -> (f32, u32) {
    let top = 0.5;
    let mut s = match support {
        Support::Nodes => {
            let mut s = plate(top + 0.005);
            s.boxes.push(StaticBox::centered(
                Vec3::new(0.0, top - 0.25, 0.0),
                Vec3::new(2.0, 0.25, 2.0),
            ));
            s
        }
        Support::Capsule => {
            let mut s = plate(top + R + 0.005);
            for x in [-0.15, 0.15] {
                s.shapes.push(StaticShape::capsule(
                    Vec3::new(x, top - 0.02, -1.0),
                    Vec3::new(x, top - 0.02, 1.0),
                    0.02,
                ));
            }
            s.surface_contact = true;
            s
        }
        Support::Bar => {
            let mut s = plate(top + R + 0.005);
            for x in [-0.15, 0.15] {
                s.boxes.push(StaticBox::centered(
                    Vec3::new(x, top - 0.25, 0.0),
                    Vec3::new(0.02, 0.25, 2.0),
                ));
            }
            s.surface_contact = true;
            s
        }
    };
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    // Settle 0.5 s.
    for _ in 0..1000 {
        s.step();
    }
    let x0 = s.positions().iter().map(|p| p.x).sum::<f32>() / 4.0;
    let f = push * m * -GRAVITY / s.nodes.len() as f32;
    let mut hits = 0;
    for _ in 0..400 {
        for n in &mut s.nodes {
            n.force.x += f;
        }
        s.step();
        hits += s.surface_last;
    }
    let x1 = s.positions().iter().map(|p| p.x).sum::<f32>() / 4.0;
    let y = s.positions().iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let _ = y;
    (x1 - x0, hits)
}

/// The plate settled on its support, then set sliding along x at `v0`: how far it goes
/// in 1 s (m) and its speed then.
fn coast(support: Support, v0: f32) -> (f32, f32) {
    let top = 0.5;
    let mut s = match support {
        Support::Nodes => {
            let mut s = plate(top + 0.005);
            s.boxes.push(StaticBox::centered(
                Vec3::new(0.0, top - 0.25, 0.0),
                Vec3::new(4.0, 0.25, 2.0),
            ));
            s
        }
        Support::Capsule => {
            let mut s = plate(top + R + 0.005);
            for x in [-0.15, 0.15] {
                s.shapes.push(StaticShape::capsule(
                    Vec3::new(x, top - 0.02, -1.0),
                    Vec3::new(x, top - 0.02, 1.0),
                    0.02,
                ));
            }
            s.surface_contact = true;
            s
        }
        Support::Bar => {
            let mut s = plate(top + R + 0.005);
            for x in [-0.15, 0.15] {
                s.boxes.push(StaticBox::centered(
                    Vec3::new(x, top - 0.25, 0.0),
                    Vec3::new(0.02, 0.25, 2.0),
                ));
            }
            s.surface_contact = true;
            s
        }
    };
    for _ in 0..1000 {
        s.step();
    }
    let x0 = s.positions().iter().map(|p| p.x).sum::<f32>() / 4.0;
    for n in &mut s.nodes {
        n.vel.x += v0;
    }
    // (Along z, the capsules' and bars' length: x would carry the plate off them.)
    for n in &mut s.nodes {
        n.vel = Vec3::new(n.vel.z, n.vel.y, n.vel.x);
    }
    let _ = x0;
    let z0 = s.positions().iter().map(|p| p.z).sum::<f32>() / 4.0;
    for _ in 0..2000 {
        s.step();
    }
    let z1 = s.positions().iter().map(|p| p.z).sum::<f32>() / 4.0;
    let v = s.nodes.iter().map(|n| n.vel.z).sum::<f32>() / 4.0;
    (z1 - z0, v)
}

/// The scene of `coast`, turned about z by `deg` about (0, 0.5, 0): the plate on an
/// incline. How far it slides down along it in 1 s after 0.5 s settling (m).
fn incline(support: Support, deg: f32) -> f32 {
    let top = 0.5;
    let q = Quat::from_rotation_z(deg.to_radians());
    let pivot = Vec3::new(0.0, top, 0.0);
    let turn = |p: Vec3| pivot + q * (p - pivot);
    let mut s = match support {
        Support::Nodes => {
            let mut s = plate(top + 0.005);
            s.shapes.push(StaticShape::oriented_box(
                turn(Vec3::new(0.0, top - 0.25, 0.0)),
                Vec3::new(4.0, 0.25, 2.0),
                q,
            ));
            s
        }
        Support::Capsule => {
            let mut s = plate(top + R + 0.005);
            for x in [-0.15, 0.15] {
                s.shapes.push(StaticShape::capsule(
                    turn(Vec3::new(x, top - 0.02, -1.0)),
                    turn(Vec3::new(x, top - 0.02, 1.0)),
                    0.02,
                ));
            }
            s.surface_contact = true;
            s
        }
        Support::Bar => {
            let mut s = plate(top + R + 0.005);
            for x in [-0.15, 0.15] {
                s.shapes.push(StaticShape::oriented_box(
                    turn(Vec3::new(x, top - 0.25, 0.0)),
                    Vec3::new(0.02, 0.25, 2.0),
                    q,
                ));
            }
            s.surface_contact = true;
            s
        }
    };
    let o = s.origin;
    for n in &mut s.nodes {
        n.pos = turn(o + n.pos) - o;
    }
    let along = |s: &ExplicitBeams| {
        s.positions()
            .iter()
            .map(|p| (q * Vec3::X).dot(*p))
            .sum::<f32>()
            / 4.0
    };
    for _ in 0..1000 {
        s.step();
    }
    let a0 = along(&s);
    for _ in 0..2000 {
        s.step();
    }
    along(&s) - a0
}

/// On an incline it stays below the kinetic limit's slope and slides past the static one.
#[test]
fn a_panel_on_an_incline_stays_below_the_friction_slope() {
    for support in [Support::Nodes, Support::Capsule, Support::Bar] {
        let row: Vec<(f32, f32)> = [10.0f32, 25.0, 40.0, 48.0, 55.0, 65.0]
            .iter()
            .map(|&d| (d, incline(support, d)))
            .collect();
        println!("incline {support:?}: {row:?}");
        // Under the kinetic limit's slope (atan 0.75 = 36.9°) it stays, set down onto it;
        // past the static limit's (atan 1.2 = 50.2°) it slides.
        for &(d, x) in &row {
            if d <= 25.0 {
                assert!(x.abs() < 0.002, "{support:?} slid {x} m at {d}°");
            }
            if d >= 55.0 {
                assert!(x < -1.0, "{support:?} held at {d}°: {x} m");
            }
        }
    }
}

/// Slid along the shapes at 0.3, 1 and 3 m/s it stops within the friction law's distances
/// (node contact on the same law, RoR's, creeps on: printed beside it).
#[test]
fn a_sliding_panel_stops_within_the_friction_laws_distance() {
    for support in [Support::Nodes, Support::Capsule, Support::Bar] {
        let row: Vec<(f32, f32, f32)> = [0.3f32, 1.0, 3.0]
            .iter()
            .map(|&v0| {
                let (d, v) = coast(support, v0);
                (v0, d, v)
            })
            .collect();
        println!("coast {support:?}: {row:?}");
        if !matches!(support, Support::Nodes) {
            // Stopped, within the distances of the kinetic curve's ends: v²/(2 μ g) for μ
            // from `ContactSurface::BOX`'s 0.75 (fast) to 1.2 (no slip).
            for &(v0, d, v) in &row {
                let (lo, hi) = (
                    v0 * v0 / (2.0 * 1.2 * -GRAVITY),
                    v0 * v0 / (2.0 * 0.75 * -GRAVITY),
                );
                assert!(
                    v.abs() < 1e-3 && d > 0.95 * lo && d < 1.05 * hi + 0.002,
                    "{support:?} from {v0} m/s: {d} m (friction's {lo}..{hi}), still {v} m/s"
                );
            }
        }
    }
}

#[test]
fn a_panel_pushed_along_a_shape_sticks_under_the_static_limit_and_slides_above() {
    for support in [Support::Nodes, Support::Capsule, Support::Bar] {
        let mut row = Vec::new();
        for push in [0.2f32, 0.5, 0.8, 1.1, 1.5, 2.0] {
            let (d, hits) = slide(support, push);
            row.push((push, d, hits));
        }
        println!("{support:?}: {row:?}");
        // Under μk·N it sticks; past μs·N (1.2) it slides.
        for &(push, d, _) in &row {
            if push <= 0.8 {
                assert!(
                    d.abs() < 0.002,
                    "{support:?} slid {d} m under {push} × its weight"
                );
            }
            if push >= 1.5 {
                assert!(
                    d > 0.05,
                    "{support:?} held under {push} × its weight: {d} m"
                );
            }
        }
    }
}

//! Explicit vehicles in the World broadphase (`engine/docs/explicit-vehicles-in-world.md`
//! §10): the coupling plan's candidates from the World's persistent resting table
//! (statics and sleepers at their cells) and the step's moving and moved colliders, not a
//! scan of every collider.
//! * lossless: from one World state the plan from the broadphase writes every proxy record
//!   word for word as the scan does, on the hierarchical grid (hulls) and the uniform grid
//!   (boxes), with statics, sleepers, falling and resting bodies near a moving vehicle;
//! * the proxy API: a sleeping vehicle's proxy is frozen, has no proxies, and is marked
//!   touched when a moving body enters its fat box; out of the broadphase it has none.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::*;

fn sedan(speed: f32, at: Vec3) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
    };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let b = |s: &ExplicitBeams| {
        s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
            (a.min(n.pos), b.max(n.pos))
        })
    };
    let (lo, hi) = b(&sim);
    if hi.x - lo.x > hi.z - lo.z {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(-n.pos.z, n.pos.y, n.pos.x);
        }
    }
    let (lo, hi) = b(&sim);
    let shift = Vec3::new(at.x - 0.5 * (lo.x + hi.x), at.y + 0.02 - lo.y, at.z - hi.z);
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = Vec3::new(0.0, 0.0, speed);
    }
    sim.ground = at.y;
    sim
}

fn def() -> ExplicitVehiclesDef {
    ExplicitVehiclesDef {
        capacity: engine_webgpu::CouplingCapacity {
            bodies: 96,
            shapes: 128,
            planes: 4096,
        },
        ..ExplicitVehiclesDef::default()
    }
}

fn cube_hull(h: f32) -> engine_schema::convex::ConvexHull {
    let pts: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 != 0 { h } else { -h },
                if i & 2 != 0 { h * 3.0 } else { -h * 3.0 },
                if i & 4 != 0 { h } else { -h },
            )
        })
        .collect();
    engine_schema::convex::ConvexHull::from_points(&pts).unwrap()
}

/// A lane of posts (hulls or boxes) both sides of the car's path, a ground slab, and
/// boxes dropped beside and in the lane (they fall, land and fall asleep, or are shoved).
fn scene(hulls: bool, posts: usize, drops: usize) -> World {
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    // The slab's top 30 cm under the car's own ground (the car does not drag on it).
    let ground = w
        .create_body(&BodyDef {
            position: [0.0, -0.8, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(ground, &ShapeDef::default(), [200.0, 0.5, 200.0])
        .unwrap();
    let hull = cube_hull(0.15);
    // Posts along both sides of the car (the nearest within its reach), then farther out.
    for k in 0..posts {
        let side = if k % 2 == 0 { -1.0 } else { 1.0 };
        let row = (k / 2) % 24;
        let ring = (k / 48) as f32;
        let x = side * (1.0 + 0.6 * ring) + 0.02 * (k % 5) as f32;
        let p = [x, 0.45, -6.0 + 0.5 * row as f32];
        let b = w
            .create_body(&BodyDef {
                position: p,
                ..BodyDef::default()
            })
            .unwrap();
        if hulls && k % 3 != 0 {
            w.create_hull_shape(b, &ShapeDef::default(), &hull).unwrap();
        } else {
            w.create_box_shape(b, &ShapeDef::default(), [0.15, 0.45, 0.15])
                .unwrap();
        }
    }
    // Boxes dropped onto and beside the car: they fall, land on it or the slab, and
    // fall asleep.
    for k in 0..drops {
        let x = -0.8 + 0.7 * (k % 3) as f32;
        let b = w
            .create_body(&BodyDef::dynamic([
                x,
                2.0 + 0.5 * (k % 4) as f32,
                -4.5 + 0.4 * (k / 2) as f32,
            ]))
            .unwrap();
        w.create_box_shape(b, &ShapeDef::default(), [0.15, 0.15, 0.15])
            .unwrap();
    }
    w
}

fn compare(hulls: bool, posts: usize) {
    let mut w = scene(hulls, posts, 24);
    let car = sedan(1.0, Vec3::new(0.0, 0.0, 0.0));
    w.add_explicit_vehicles(std::slice::from_ref(&car), &def())
        .unwrap();
    let mut checked = 0;
    let mut with_shapes = 0;
    let mut used = 0;
    for step in 0..240 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if step == 0 {
            println!("resting table: {:?}", w.sim().rest_view());
        }
        if w.explicit_plan_used_broadphase() == Some(true) {
            used += 1;
        }
        if step % 12 != 11 {
            continue;
        }
        let Ok([(a, sa), (b, sb)]) = w.debug_explicit_plans() else {
            continue;
        };
        assert_eq!(sa, 0, "step {step}: broadphase plan status {sa}");
        assert_eq!(sb, 0, "step {step}: scan plan status {sb}");
        let diff = a.iter().zip(&b).filter(|(p, q)| p != q).count();
        let shapes = b[1];
        let z = w.explicit_vehicles().unwrap().positions()[0]
            .iter()
            .map(|p| p.z)
            .fold(f32::MIN, f32::max);
        println!("step {step}: car front z {z:.2}, proxy shapes {shapes}, {diff} words differ");
        assert_eq!(
            diff, 0,
            "step {step}: the broadphase plan differs from the scan in {diff} words"
        );
        checked += 1;
        if shapes > 0 {
            with_shapes += 1;
        }
    }
    println!("hulls {hulls}: {checked} checks, {with_shapes} with proxies; broadphase used in {used} of 240 windows");
    assert!(
        checked >= 10,
        "the resting table was kept: {checked} checks"
    );
    assert!(
        with_shapes >= 5,
        "the car met proxies in {with_shapes} checks"
    );
    assert!(
        used >= 200,
        "the plan took the broadphase in {used} windows"
    );
}

#[test]
fn the_broadphase_plan_is_the_scan_on_the_hierarchical_grid() {
    compare(true, 160);
}

#[test]
fn the_broadphase_plan_is_the_scan_on_the_uniform_grid() {
    compare(false, 700);
}

#[test]
fn a_sleeping_vehicle_proxy_is_frozen_and_woken_by_a_mover() {
    let mut w = scene(true, 40, 0);
    let car = sedan(0.0, Vec3::new(0.0, 0.0, -2.0));
    w.add_explicit_vehicles(std::slice::from_ref(&car), &def())
        .unwrap();
    for _ in 0..10 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
    let p0 = w.explicit_vehicle_proxy(0).unwrap();
    assert!(p0.refattens >= 1 && !p0.asleep && p0.in_broadphase);
    assert!(p0.lo.cmplt(p0.hi).all(), "fat box {p0:?}");
    w.set_explicit_vehicle_asleep(0, true).unwrap();
    for _ in 0..10 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
    let p1 = w.explicit_vehicle_proxy(0).unwrap();
    assert!(p1.asleep && !p1.touched, "{p1:?}");
    assert_eq!(
        (p1.lo, p1.hi, p1.refattens),
        (p0.lo, p0.hi, p0.refattens),
        "frozen while asleep"
    );
    assert!(
        w.explicit_window_impulses()[0].is_empty(),
        "no proxies while asleep"
    );
    // A ball dropped onto the sleeping car: it enters the fat box and marks it.
    let ball = w.create_body(&BodyDef::dynamic([0.0, 3.5, -4.0])).unwrap();
    w.create_sphere_shape(ball, &ShapeDef::default(), [0.0; 3], 0.2)
        .unwrap();
    let mut touched = false;
    for _ in 0..60 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if w.explicit_vehicle_proxy(0).unwrap().touched {
            touched = true;
            break;
        }
    }
    assert!(
        touched,
        "the falling ball marked the sleeping vehicle's proxy"
    );
    w.set_explicit_vehicle_asleep(0, false).unwrap();
    w.set_explicit_vehicle_in_broadphase(0, false).unwrap();
    w.step().unwrap();
    w.poll_blocking().unwrap();
    assert!(
        w.explicit_window_impulses()[0].is_empty(),
        "no proxies out of the broadphase"
    );
    w.set_explicit_vehicle_in_broadphase(0, true).unwrap();
    w.step().unwrap();
    w.poll_blocking().unwrap();
    let p2 = w.explicit_vehicle_proxy(0).unwrap();
    assert!(
        p2.in_broadphase && p2.refattens == 1,
        "refattened on return: {p2:?}"
    );
}

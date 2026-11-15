//! Surface contact's lossless speed-ups against the World's proxies
//! (`jbeam-semantics-spec.md` §38.5): from one World state, frames with each on and off
//! (the pair path, the feature queries' edge culls, both) leave every vehicle state
//! word, every proxy record and every coupling sum the same.
//! The scene: a repo sedan with surface contact driving at 4 m/s between rows of posts
//! 3 cm from its sides, just past its contact band (boxes, turned boxes, hulls, turned hulls),
//! a capsule, a sphere and a resting box ahead, a free box thrown at its front and a fast
//! steel ball fired onto its roof (a fast proxy: its casts beside the pairs).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::{Quat, Vec3};
use rubble::*;

fn sedan(speed: f32) -> ExplicitBeams {
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
        centred_damping: false,
        centred_objective: false,
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
    let shift = Vec3::new(-0.5 * (lo.x + hi.x), 0.02 - lo.y, -hi.z);
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = Vec3::new(0.0, 0.0, speed);
    }
    sim.ground = 0.0;
    sim.surface_contact = true;
    sim.dissipative_contact = true;
    sim
}

/// The car's half width (x) after the placement above.
fn half_width(s: &ExplicitBeams) -> f32 {
    s.nodes.iter().map(|n| n.pos.x.abs()).fold(0.0, f32::max)
}

fn scene() -> World {
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.8, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [30.0, 0.5, 30.0])
        .unwrap();
    let car = sedan(4.0);
    let side = half_width(&car) + 0.15 + 0.03;
    let pts: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 != 0 { 0.15 } else { -0.15 },
                if i & 2 != 0 { 0.45 } else { -0.45 },
                if i & 4 != 0 { 0.15 } else { -0.15 },
            )
        })
        .collect();
    let hull = engine_schema::convex::ConvexHull::from_points(&pts).unwrap();
    let turned = Quat::from_rotation_y(0.5).to_array();
    for k in 0..8 {
        let z = -4.2 + 0.6 * k as f32;
        for (sx, kind) in [(-1.0f32, k % 4), (1.0, (k + 2) % 4)] {
            let b = w
                .create_body(&BodyDef {
                    position: [sx * side, 0.45, z],
                    rotation: if kind >= 2 {
                        turned
                    } else {
                        [0.0, 0.0, 0.0, 1.0]
                    },
                    ..BodyDef::default()
                })
                .unwrap();
            if kind % 2 == 0 {
                w.create_box_shape(b, &ShapeDef::default(), [0.15, 0.45, 0.15])
                    .unwrap();
            } else {
                w.create_hull_shape(b, &ShapeDef::default(), &hull).unwrap();
            }
        }
    }
    // Ahead: a capsule post, a sphere and a free box in the car's path.
    let b = w
        .create_body(&BodyDef {
            position: [0.4, 0.6, 1.5],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_capsule_shape(b, &ShapeDef::default(), [0.0, 0.0, 0.0], 0.1, 0.4)
        .unwrap();
    let b = w
        .create_body(&BodyDef {
            position: [-0.5, 0.25, 1.6],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_sphere_shape(b, &ShapeDef::default(), [0.0, 0.0, 0.0], 0.25)
        .unwrap();
    let b = w.create_body(&BodyDef::dynamic([0.0, -0.1, 2.0])).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.2, 0.2, 0.2])
        .unwrap();
    // A free box thrown at the car's front.
    let b = w
        .create_body(&BodyDef {
            linear_velocity: [0.0, 0.0, -6.0],
            ..BodyDef::dynamic([0.0, 0.45, 1.2])
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.2, 0.2, 0.2])
        .unwrap();
    // A fast steel ball (3 cm at 60 m/s: 3 cm a step, past half its radius) fired down
    // onto the roof: a fast proxy (its casts) beside the pair path's slow ones.
    let b = w
        .create_body(&BodyDef {
            linear_velocity: [0.0, -60.0, 0.0],
            ..BodyDef::dynamic([0.0, 3.0, -2.0])
        })
        .unwrap();
    let steel = ShapeDef {
        density: 7850.0,
        ..ShapeDef::default()
    };
    w.create_sphere_shape(b, &steel, [0.0; 3], 0.03).unwrap();
    let def = ExplicitVehiclesDef {
        capacity: engine_webgpu::CouplingCapacity {
            bodies: 64,
            shapes: 64,
            planes: 2048,
        },
        sleep: None,
        ..ExplicitVehiclesDef::default()
    };
    w.add_explicit_vehicles(&[car], &def).unwrap();
    w
}

/// Every vehicle and coupling word after `frames` frames from one saved state, with
/// `set` applied first.
fn run(
    x: &mut engine_webgpu::ExplicitBeamsGpu,
    snap: &engine_webgpu::GpuSnapshot,
    frames: usize,
    set: &dyn Fn(&mut engine_webgpu::ExplicitBeamsGpu),
) -> Vec<(&'static str, Vec<u32>)> {
    x.restore(snap);
    set(x);
    for _ in 0..frames {
        x.frame(17);
    }
    x.wait();
    x.set_surface_pairs(true);
    x.set_surface_culls(true);
    let mut out = x.state_words();
    out.extend(x.coupling_words());
    out
}

#[test]
fn surface_pairs_are_the_per_element_loops_bit_for_bit() {
    let mut w = scene();
    let mut checked = 0;
    let mut hit = 0;
    for step in 0..120 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        // Through the strike (the car meets the posts ahead within 20 steps), then resting.
        if !(step < 8 || step % 3 == 1 && step < 40 || step % 40 == 39) {
            continue;
        }
        let shapes: Vec<usize> = w
            .explicit_window_impulses()
            .iter()
            .map(|v| v.len())
            .collect();
        let x = w.explicit_vehicles_mut().unwrap();
        let snap = x.snapshot();
        let off = run(x, &snap, 3, &|x| {
            x.set_surface_pairs(false);
            x.set_surface_culls(false)
        });
        let culls = run(x, &snap, 3, &|x| x.set_surface_pairs(false));
        let pairs = run(x, &snap, 3, &|x| x.set_surface_culls(false));
        let on = run(x, &snap, 3, &|_| {});
        let stats = x.surface_pair_stats();
        x.restore(&snap);
        let front = x.positions()[0]
            .iter()
            .map(|p| p.z)
            .fold(f32::MIN, f32::max);
        println!("step {step}: car front z {front:.2}, proxy shapes {shapes:?}, surface pairs (listed, hits) {stats:?}");
        for (label, b) in [("edge culls", &culls), ("pair path", &pairs), ("both", &on)] {
            for ((name, a), (_, b)) in off.iter().zip(b) {
                let diff = a.iter().zip(b).filter(|(p, q)| p != q).count();
                assert_eq!(
                    diff, 0,
                    "step {step}: {name}: {diff} words differ with the {label}"
                );
            }
        }
        if stats.iter().any(|&(_, h)| h > 0) {
            hit += 1;
        }
        if shapes.iter().any(|&n| n >= 10) {
            checked += 1;
        }
    }
    assert!(
        checked >= 3,
        "the comparison ran with the car among the posts"
    );
    assert!(hit >= 3, "the surface met the posts on the pair path");
}

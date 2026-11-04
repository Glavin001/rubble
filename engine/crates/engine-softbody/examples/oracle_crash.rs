//! Explicit-oracle crash probe: the procedural sedan into a rigid wall.
//!
//! ```bash
//! cargo run -p engine-softbody --release --example oracle_crash -- speed=56 damp=1 hz=2000 car=sedan
//! ```
use engine_softbody::explicit_ref::{ExplicitSettings, ExplicitSim, Shape};
use engine_softbody::metrics::*;
use engine_softbody::procedural::{car_desc, CarParams};
use glam::{DQuat, DVec3, Quat, Vec3};

fn arg(name: &str, default: f64) -> f64 {
    std::env::args()
        .skip(1)
        .find_map(|a| {
            a.strip_prefix(&format!("{name}="))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(default)
}

fn main() {
    let car = std::env::args()
        .skip(1)
        .find_map(|a| a.strip_prefix("car=").map(str::to_string));
    let p = match car.as_deref() {
        Some("default") => CarParams::default(),
        _ => CarParams::sedan(),
    };
    let (d, layout) = car_desc(&p);
    let mut sim = ExplicitSim::new(ExplicitSettings {
        hz: arg("hz", 2000.0),
        damping_scale: arg("damp", 1.0),
        plasticity: arg("plastic", 1.0) != 0.0,
        breaking: arg("break", 1.0) != 0.0,
        self_collision: arg("self", 1.0) != 0.0,
        ..Default::default()
    });
    let v = arg("speed", 56.0) / 3.6;
    let spawn = Vec3::new(0.0, 0.06 - d.bounds().0[1], -1.0);
    sim.add_vehicle(&d, spawn, Quat::IDENTITY, Vec3::new(0.0, 0.0, v as f32));
    sim.add_obstacle(
        Shape::Plane {
            normal: DVec3::Y,
            offset: 0.0,
        },
        arg("mu", 0.8),
    );
    sim.add_obstacle(
        Shape::Box {
            center: DVec3::new(0.0, 1.5, 3.0),
            half: DVec3::new(3.0, 1.5, 0.2),
            rotation: DQuat::IDENTITY,
        },
        1.0,
    );
    let rest: Vec<Vec3> = d
        .nodes
        .iter()
        .map(|n| Vec3::from_array(n.position) + spawn)
        .collect();
    let body = d.part("body").unwrap().nodes.clone();
    let ring_nodes = layout.stations * 8;
    let rear: Vec<u32> = body
        .iter()
        .copied()
        .filter(|&i| (i as usize) < ring_nodes && (i as usize / 8) <= layout.cabin_first)
        .collect();
    let cabin: Vec<u32> = body
        .iter()
        .copied()
        .filter(|&i| (i as usize) < ring_nodes && layout.cabin[i as usize / 8])
        .collect();
    let per = (sim.hz() / 240.0).round() as u32;
    println!("rate {:.0} Hz, mass {:.0} kg", sim.hz(), sim.total_mass(0));
    let len = |x: &[Vec3]| {
        let c = |k: usize| (0..6).map(|j| x[k * 8 + j]).sum::<Vec3>() / 6.0;
        (c(layout.stations - 1) - c(0)).length()
    };
    let l0 = len(&rest);
    for f in 0..(arg("secs", 1.0) * 240.0) as u32 {
        for _ in 0..per {
            sim.substep();
        }
        if f % 12 == 0 {
            let x = sim.positions();
            let vz = centroid_velocity(
                &sim.body_velocities(0),
                &sim.mass.iter().map(|&m| m as f32).collect::<Vec<_>>(),
                &cabin,
            )
            .z;
            println!(
                "t {:.3} vz {:6.2} shortening {:.3} yielded {} broken {}",
                sim.time,
                vz,
                l0 - len(&x),
                sim.yielded_beams(0),
                sim.broken(0).len()
            );
        }
    }
    let x = sim.positions();
    let prof = crush_profile(&rest, &x, &body, &rear, Vec3::Z, Vec3::X, 5);
    println!("crush profile {:?}", prof);
}

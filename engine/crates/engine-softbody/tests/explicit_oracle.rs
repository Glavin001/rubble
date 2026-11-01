//! The explicit oracle on the procedural cars: stability at rest, a full-width
//! wall crash, and the crash metrics it reports (CPU only, runs in CI).

use engine_softbody::explicit_ref::{ExplicitSettings, ExplicitSim, Shape};
use engine_softbody::metrics::*;
use engine_softbody::procedural::{car_desc, CarParams};
use glam::{DQuat, DVec3, Quat, Vec3};

/// Spawn height that puts the lowest node sphere on the ground (the oracle has
/// no wheels: the car stands on its nodes).
fn standing(d: &engine_softbody::SoftVehicleDesc) -> f32 {
    0.06 - d.bounds().0[1]
}

fn broken_flags(sim: &ExplicitSim) -> Vec<Vec<bool>> {
    (0..sim.bodies.len() as u32)
        .map(|b| {
            let r = sim.bodies[b as usize];
            (0..r.beams)
                .map(|k| sim.beams[(r.first_beam + k) as usize].broken())
                .collect()
        })
        .collect()
}

fn ground(sim: &mut ExplicitSim) {
    sim.add_obstacle(
        Shape::Plane {
            normal: DVec3::Y,
            offset: 0.0,
        },
        0.8,
    );
}

#[test]
fn car_rests_without_yielding_or_clipping() {
    let (d, _) = car_desc(&CarParams::default());
    let mut sim = ExplicitSim::new(ExplicitSettings::default());
    sim.add_vehicle(
        &d,
        Vec3::new(0.0, standing(&d), 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
    );
    ground(&mut sim);
    let mut tr = PenetrationTracker::new(vec![TrackedBody::from_desc(&d, 0)]);
    let per_frame = (sim.hz() / 240.0).round() as u32;
    for _ in 0..240 {
        for _ in 0..per_frame {
            sim.substep();
        }
        tr.update(&sim.positions(), Some(&broken_flags(&sim)));
    }
    println!("rate {:.0} Hz, ke {:.3}", sim.hz(), sim.kinetic_energy(0));
    assert!(sim.x.iter().all(|p| p.is_finite()));
    assert_eq!(sim.yielded_beams(0), 0, "resting must not yield");
    assert!(sim.kinetic_energy(0) < 1.0);
    assert_eq!(tr.max.total(), 0, "no clipping at rest");
}

/// 56 km/h into a rigid wall: the front folds (about 0.2 m converged; the GPU at
/// 960 Hz × 30 agrees within 2 cm, `soft_calibrate`), the cabin keeps its shape,
/// and almost no node ends up behind a triangle it was in front of.
#[test]
fn wall_56_crushes_the_front_without_clipping() {
    let p = CarParams::sedan();
    let (d, layout) = car_desc(&p);
    let mut sim = ExplicitSim::new(ExplicitSettings::default());
    let v = 56.0 / 3.6;
    let spawn = Vec3::new(0.0, standing(&d), -1.0);
    sim.add_vehicle(&d, spawn, Quat::IDENTITY, Vec3::new(0.0, 0.0, v));
    ground(&mut sim);
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
        .map(|n| Vec3::from_array(n.position))
        .collect();
    let body = d.part("body").unwrap().nodes.clone();
    let cabin: Vec<u32> = body
        .iter()
        .copied()
        .filter(|&i| (i as usize) < layout.stations * 8 && layout.cabin[i as usize / 8])
        .collect();
    let rear: Vec<u32> = body
        .iter()
        .copied()
        .filter(|&i| (i as usize) < layout.stations * 8 && (i as usize / 8) <= layout.cabin_first)
        .collect();
    let mass: Vec<f32> = sim.mass.iter().map(|&m| m as f32).collect();
    let mut tr = PenetrationTracker::new(vec![TrackedBody::from_desc(&d, 0)]);
    let mut decel = DecelTracker::default();
    let per_frame = (sim.hz() / 240.0).round() as u32;
    let t = std::time::Instant::now();
    for _ in 0..(240.0 * 1.0) as u32 {
        for _ in 0..per_frame {
            sim.substep();
        }
        let x = sim.positions();
        tr.update(&x, Some(&broken_flags(&sim)));
        decel.push(
            sim.time,
            centroid_velocity(&sim.body_velocities(0), &mass, &cabin),
        );
    }
    let x = sim.positions();
    let world_rest: Vec<Vec3> = rest.iter().map(|p| *p + spawn).collect();
    let profile = crush_profile(&world_rest, &x, &body, &rear, Vec3::Z, Vec3::X, 5);
    let engine = d.part("engine").unwrap().nodes.clone();
    let (travel, back) = block_travel(&world_rest, &x, &cabin, &engine, Vec3::Z);
    let probes: Vec<u32> = cabin
        .iter()
        .copied()
        .filter(|&i| i as usize / 8 == layout.cabin_last)
        .collect();
    let intr = intrusion(&world_rest, &x, &cabin, &probes);
    println!(
        "rate {:.0} Hz, {:.1} s wall time; yielded {}, broken {}; crush profile {:?} cm",
        sim.hz(),
        t.elapsed().as_secs_f32(),
        sim.yielded_beams(0),
        sim.broken(0).len(),
        profile
            .iter()
            .map(|c| (c * 1000.0).round() / 10.0)
            .collect::<Vec<_>>()
    );
    println!(
        "engine travel {travel:.3} m ({back:.3} back); firewall intrusion {intr:.3} m; peak {:.1} g, stop {:?} s; penetrations max {:?}",
        decel.peak_decel_g(0.01),
        decel.time_to_stop(0.08, 0.5),
        tr.max
    );
    assert!(sim.x.iter().all(|p| p.is_finite()));
    let max_crush = profile
        .iter()
        .copied()
        .filter(|c| !c.is_nan())
        .fold(0.0, f32::max);
    assert!(max_crush > 0.12 && max_crush < 0.45, "crush {max_crush}");
    assert!(intr < 0.15, "the cell holds ({intr})");
    assert!(tr.max.total() <= 4, "clipping {:?}", tr.max);
}

/// Penetrations in the phase 2 scenes on the oracle (the reference for the GPU's
/// `soft_anticlip.rs` probe): a 100 km/h wall and a 2 × 80 km/h head-on.
#[test]
#[ignore]
fn oracle_phase2_scenes_probe() {
    let (d, _) = car_desc(&CarParams::sedan());
    let lift = standing(&d);
    for scene in ["wall100", "headon"] {
        let mut sim = ExplicitSim::new(ExplicitSettings::default());
        match scene {
            "wall100" => {
                sim.add_vehicle(
                    &d,
                    Vec3::new(0.0, lift, -1.0),
                    Quat::IDENTITY,
                    Vec3::new(0.0, 0.0, 100.0 / 3.6),
                );
                sim.add_obstacle(
                    Shape::Box {
                        center: DVec3::new(0.0, 1.5, 3.0),
                        half: DVec3::new(3.0, 1.5, 0.2),
                        rotation: DQuat::IDENTITY,
                    },
                    0.894,
                );
            }
            _ => {
                let v = 80.0 / 3.6;
                sim.add_vehicle(
                    &d,
                    Vec3::new(0.0, lift, -2.6),
                    Quat::IDENTITY,
                    Vec3::new(0.0, 0.0, v),
                );
                sim.add_vehicle(
                    &d,
                    Vec3::new(0.0, lift, 2.6),
                    Quat::from_rotation_y(std::f32::consts::PI),
                    Vec3::new(0.0, 0.0, -v),
                );
            }
        }
        ground(&mut sim);
        let n = d.nodes.len() as u32;
        let bodies = (0..sim.bodies.len() as u32)
            .map(|b| TrackedBody::from_desc(&d, b * n))
            .collect();
        let mut tr = PenetrationTracker::new(bodies);
        let per = (sim.hz() / 240.0).round() as u32;
        for _ in 0..300 {
            for _ in 0..per {
                sim.substep();
            }
            tr.update(&sim.positions(), Some(&broken_flags(&sim)));
        }
        println!("{scene}: oracle at {:.0} Hz: {:?}", sim.hz(), tr.max);
    }
}

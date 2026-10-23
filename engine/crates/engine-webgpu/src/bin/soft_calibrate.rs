//! Soft-body calibration harness (design: "Calibration and validation") on the
//! native wgpu host. Runs the standard tests on the procedural car at a reference
//! setting (a fine step and many iterations, standing in for a 2,000 Hz explicit
//! engine when no Rigs of Rods recording is available) and at candidate settings,
//! reports the metrics, and fits the global threshold scale per step rate so the
//! 60 km/h wall crush matches the reference.
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin soft_calibrate
//! ```
use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::calibrate::*;
use engine_softbody::procedural::{car_truck, CarParams};
use engine_softbody::truck::load_truck;
use engine_softbody::{
    soft_scene_params, SoftCapacity, SoftHandle, SoftSettings, SoftVehicleDesc, SpawnOptions,
};
use engine_vehicle::VehicleCapacity;
use engine_webgpu::WgpuSim;
use glam::Vec3;

#[derive(Clone, Copy)]
struct Setting {
    hz: f32,
    iterations: u32,
    threshold_scale: f32,
}

fn world(s: Setting) -> WgpuSim {
    let cap = WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 512,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_soft(cap, VehicleCapacity::default(), SoftCapacity::default())
        .expect("wgpu");
    let mut p = soft_scene_params(s.hz, s.iterations);
    p.sleep_time = 1.0e6;
    sim.set_params(p);
    sim.set_soft_settings(SoftSettings {
        threshold_scale: s.threshold_scale,
        ..Default::default()
    });
    sim.set_material(
        0,
        Material {
            friction: 1.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    sim.set_material(
        9,
        Material {
            friction: 0.8,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    let g = sim.add_body(BodyState::static_plane_support());
    sim.add_collider(Collider {
        body: g,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0, -5.0, 0.0],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [400.0, 5.0, 400.0, 566.0],
    });
    sim
}

fn positions(sim: &WgpuSim, h: SoftHandle) -> Vec<Vec3> {
    sim.read_soft_particles(h)
        .iter()
        .map(|b| Vec3::from_array(b.position))
        .collect()
}

fn kinetic(sim: &WgpuSim, h: SoftHandle) -> f32 {
    sim.read_soft_particles(h)
        .iter()
        .filter(|b| b.inv_mass > 0.0)
        .map(|b| 0.5 / b.inv_mass * Vec3::from_array(b.linear_vel).length_squared())
        .sum()
}

fn run_s(sim: &mut WgpuSim, s: Setting, seconds: f32) {
    for _ in 0..(s.hz * seconds) as u32 {
        sim.step().expect("step");
    }
}

#[derive(Default, Debug, Clone)]
struct Metrics {
    sag_rms: f32,
    tap_hz: f32,
    drop_yield: usize,
    crush30: (f32, f32),
    crush60: (f32, f32),
    broken60: Vec<u32>,
    absorbed60: f32,
}

fn rest_and_tap(desc: &SoftVehicleDesc, s: Setting) -> (f32, f32) {
    let mut sim = world(s);
    let h = sim
        .add_soft_vehicle(
            desc,
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions::default(),
        )
        .unwrap();
    let rest = positions(&sim, h);
    run_s(&mut sim, s, 3.0);
    let sag = rms_max(&residual_displacement(&rest, &positions(&sim, h))).0;
    // Tap: a downward 1 m/s kick on the front-left roof node, then record a
    // structural length (roof node to the floor node below it), which the
    // suspension's rigid bounce leaves alone.
    let first = sim.soft_host().entry(h).first_body;
    let stations = CarParams::default().stations;
    let ring = (stations - 1) * engine_softbody::procedural::CAR_RING;
    let (node, floor) = (first + (ring + 4) as u32, first + ring as u32);
    let mut b = sim.read_body(node);
    b.linear_vel = [0.0, -1.0, 0.0];
    engine_vehicle::VehicleBackend::write_body(&mut sim, node, &b);
    let mut sig = Vec::new();
    for _ in 0..(s.hz * 1.0) as u32 {
        sim.step().unwrap();
        let (p, q) = (
            Vec3::from_array(sim.read_body(node).position),
            Vec3::from_array(sim.read_body(floor).position),
        );
        sig.push((p - q).length());
    }
    (sag, dominant_frequency(&sig, 1.0 / s.hz, 0.5 * s.hz))
}

fn drop_test(desc: &SoftVehicleDesc, s: Setting) -> usize {
    let mut sim = world(s);
    let h = sim
        .add_soft_vehicle(
            desc,
            [0.0, 1.05, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions::default(),
        )
        .unwrap();
    run_s(&mut sim, s, 2.0);
    sim.read_soft_beams(h)
        .iter()
        .filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0)
        .count()
}

fn wall(desc: &SoftVehicleDesc, s: Setting, kmh: f32) -> ((f32, f32), Vec<u32>, f32) {
    let mut sim = world(s);
    let b = sim.add_body(BodyState::static_plane_support());
    sim.add_collider(Collider {
        body: b,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0, 1.5, 3.0],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [3.0, 1.5, 0.2, 3.4],
    });
    let v = kmh / 3.6;
    let h = sim
        .add_soft_vehicle(
            desc,
            [0.0, 0.05, -1.0],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions {
                velocity: [0.0, 0.0, v],
                ..Default::default()
            },
        )
        .unwrap();
    let rest = positions(&sim, h);
    let mass: f32 = sim
        .read_soft_particles(h)
        .iter()
        .filter(|b| b.inv_mass > 0.0)
        .map(|b| 1.0 / b.inv_mass)
        .sum();
    let ke0 = 0.5 * mass * v * v;
    run_s(&mut sim, s, 1.5);
    let now = positions(&sim, h);
    // The bumper may detach: measure the body (every node but the last six).
    let n = now.len() - 6;
    let crush = rms_max(&residual_displacement(&rest[..n], &now[..n]));
    let broken: Vec<u32> = sim
        .read_soft_beams(h)
        .iter()
        .enumerate()
        .filter(|(_, b)| b.broken())
        .map(|(i, _)| i as u32)
        .collect();
    let absorbed = (ke0 - kinetic(&sim, h)) / ke0;
    (crush, broken, absorbed)
}

fn metrics(desc: &SoftVehicleDesc, s: Setting) -> Metrics {
    let (sag_rms, tap_hz) = rest_and_tap(desc, s);
    let (crush30, _, _) = wall(desc, s, 30.0);
    let (crush60, broken60, absorbed60) = wall(desc, s, 60.0);
    Metrics {
        sag_rms,
        tap_hz,
        drop_yield: drop_test(desc, s),
        crush30,
        crush60,
        broken60,
        absorbed60,
    }
}

fn main() {
    let desc = load_truck(&car_truck(&CarParams::default()));
    let reference = Setting {
        hz: 960.0,
        iterations: 30,
        threshold_scale: 1.0,
    };
    let t = std::time::Instant::now();
    let r = metrics(&desc, reference);
    eprintln!("reference done in {:.1} s", t.elapsed().as_secs_f32());
    println!("| setting | sag RMS (mm) | tap (Hz) | 1 m drop: bent | crush 30 km/h RMS/max (cm) | crush 60 km/h RMS/max (cm) | broken-set Jaccard 60 | KE absorbed 60 |");
    println!("|---|---|---|---|---|---|---|---|");
    let row = |name: &str, m: &Metrics| {
        println!(
            "| {name} | {:.2} | {:.1} | {} | {:.1} / {:.1} | {:.1} / {:.1} | {:.2} | {:.0}% |",
            m.sag_rms * 1e3,
            m.tap_hz,
            m.drop_yield,
            m.crush30.0 * 100.0,
            m.crush30.1 * 100.0,
            m.crush60.0 * 100.0,
            m.crush60.1 * 100.0,
            jaccard(&m.broken60, &r.broken60),
            m.absorbed60 * 100.0
        );
    };
    row("reference 960 Hz × 30", &r);
    for (hz, it) in [(120.0, 10), (240.0, 10), (240.0, 20), (480.0, 10)] {
        let s = Setting {
            hz,
            iterations: it,
            threshold_scale: 1.0,
        };
        row(&format!("{hz} Hz × {it}"), &metrics(&desc, s));
    }
    // Fit: one global threshold scale per step rate so the 60 km/h crush depth
    // (max residual) matches the reference.
    println!();
    println!(
        "| step rate | fitted threshold scale | crush 60 km/h max (cm), reference {:.1} |",
        r.crush60.1 * 100.0
    );
    println!("|---|---|---|");
    for hz in [120.0, 240.0] {
        let target = r.crush60.1;
        let scale = minimize_1d(
            |k| {
                rel_error(
                    wall(
                        &desc,
                        Setting {
                            hz,
                            iterations: 10,
                            threshold_scale: k,
                        },
                        60.0,
                    )
                    .0
                     .1,
                    target,
                )
            },
            0.2,
            1.5,
            8,
        );
        let fitted = wall(
            &desc,
            Setting {
                hz,
                iterations: 10,
                threshold_scale: scale,
            },
            60.0,
        )
        .0
         .1;
        println!("| {hz} Hz × 10 | {scale:.2} | {:.1} |", fitted * 100.0);
    }
}

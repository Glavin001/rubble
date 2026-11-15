//! The vehicle lab's camera follows `Bench::pose`: the car's centre and velocity as last
//! read, with the World time of that reading, carried forward to the time drawn. A page
//! reads the car's buffers back asynchronously, so a reading arrives one to three frames
//! after the step it was copied at. The reading's time is the time it was TAKEN: stamped
//! with the time it arrived, the carried-forward centre falls short of the car by its
//! speed × the readback's delay, a delay that changes from frame to frame (the camera's
//! jitter on the mountain, reported 2026-11-14). CPU reference only.
#![cfg(feature = "stress")]

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::{DVec3, Vec3};
use rubble::vehicle_lab::cpu::CpuLab;
use rubble::vehicle_lab::monitor::{CarRead, ProxyFrame};
use rubble::vehicle_lab::LabOptions;

fn car() -> ExplicitBeams {
    let text = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
        minimass: 10.0,
        ..engine_softbody::procedural::CarParams::sedan()
    });
    ExplicitBeams::from_truck_raycast(
        &text,
        None,
        Vec3::ZERO,
        0.0,
        Vec3::ZERO,
        ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            ..Default::default()
        },
    )
    .unwrap()
    .0
}

/// A car in uniform motion at 20 m/s, read every 4 windows, each reading arriving 1, 2
/// or 3 windows late: the pose carried forward to "now" is the car's true centre.
#[test]
fn a_late_reading_carries_the_time_it_was_taken() {
    let mut lab = CpuLab::new(&car(), LabOptions::default()).unwrap();
    let rest: Vec<Vec3> = lab.sim.nodes.iter().map(|n| n.pos).collect();
    let v = Vec3::new(0.0, 0.0, 20.0);
    let vel = vec![v; rest.len()];
    let dt = lab.bench.dt as f64;
    let centre0 = {
        let p = lab.bench.pose();
        DVec3::new(p[0], p[1], p[2])
    };
    let mut worst = 0.0f64;
    // (taken at window, arrives at window)
    let mut pending: Vec<(u32, u32)> = Vec::new();
    for w in 0..120u32 {
        lab.bench.t = f64::from(w) * dt;
        if w % 4 == 0 {
            pending.push((w, w + 1 + (w / 4) % 3));
        }
        while pending.first().is_some_and(|p| p.1 <= w) {
            let (taken, _) = pending.remove(0);
            let t = f64::from(taken) * dt;
            lab.bench.read_at(
                &CarRead {
                    pos: &rest,
                    vel: &vel,
                    beams: None,
                    origin: DVec3::new(0.0, 0.0, 20.0 * t),
                },
                &ProxyFrame::default(),
                t,
            );
        }
        let p = lab.bench.pose();
        if p[11] < 2.0 {
            continue;
        }
        let now = lab.bench.t;
        let carried = DVec3::new(p[0], p[1], p[2]) + DVec3::new(p[3], p[4], p[5]) * (now - p[10]);
        let truth = centre0 + DVec3::new(0.0, 0.0, 20.0 * now);
        worst = worst.max((carried - truth).length());
    }
    assert!(worst < 1e-3, "the pose carried forward is {worst:.3} m from the car");
}

/// `Bench::read` (a host that reads at the step, blocking) stamps the lab's time.
#[test]
fn a_reading_at_the_step_carries_the_labs_time() {
    let mut lab = CpuLab::new(&car(), LabOptions::default()).unwrap();
    let rest: Vec<Vec3> = lab.sim.nodes.iter().map(|n| n.pos).collect();
    let vel = vec![Vec3::ZERO; rest.len()];
    lab.bench.t = 1.25;
    lab.bench.read(
        &CarRead {
            pos: &rest,
            vel: &vel,
            beams: None,
            origin: DVec3::ZERO,
        },
        &ProxyFrame::default(),
    );
    assert_eq!(lab.bench.pose()[10], 1.25);
}

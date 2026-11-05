//! Raycast wheels on explicit soft-body vehicles (`engine_softbody::explicit_drive`).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, NODE_TYRE};
use engine_softbody::explicit_drive::{spin_solve, DriveInput};
use engine_softbody::procedural::{car_truck, CarParams};
use glam::Vec3;

fn sedan() -> String {
    car_truck(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    })
}

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    }
}

fn spawn(v: Vec3) -> ExplicitBeams {
    let (mut s, _) = ExplicitBeams::from_truck_raycast(
        &sedan(),
        None,
        Vec3::new(0.0, 0.6, 0.0),
        0.0,
        v,
        settings(),
    )
    .expect("raycast sedan");
    s.finish();
    s
}

fn com(s: &ExplicitBeams) -> (Vec3, Vec3, f32) {
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let x = s
        .nodes
        .iter()
        .map(|n| (s.origin + n.pos) * n.mass)
        .sum::<Vec3>()
        / m;
    let v = s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
    (x, v, m)
}

fn forward(s: &ExplicitBeams) -> Vec3 {
    let [c, b, _] = s.drive.as_ref().unwrap().config.refs;
    let f = s.nodes[c as usize].pos - s.nodes[b as usize].pos;
    Vec3::new(f.x, 0.0, f.z).normalize()
}

/// Settle on the ground for `t` s.
fn settle(s: &mut ExplicitBeams, t: f32) {
    s.run(t);
}

#[test]
fn raycast_spawn_has_no_tyre_nodes_and_keeps_the_mass() {
    let with =
        ExplicitBeams::from_truck(&sedan(), Vec3::new(0.0, 0.6, 0.0), Vec3::ZERO, settings())
            .unwrap();
    let s = spawn(Vec3::ZERO);
    assert!(s.nodes.iter().all(|n| n.flags & NODE_TYRE == 0));
    let m0: f32 = with.nodes.iter().map(|n| n.mass).sum();
    let m1: f32 = s.nodes.iter().map(|n| n.mass).sum();
    assert!((m0 - m1).abs() < 1e-3 * m0, "{m0} vs {m1}");
    let d = s.drive.as_ref().unwrap();
    assert_eq!(d.config.wheels.len(), 4);
    assert!(d.config.wheels.iter().any(|w| w.driven));
    assert!(d.config.wheels.iter().any(|w| w.max_steer > 0.0));
}

#[test]
fn spin_solve_closed_forms() {
    let h = 0.0005;
    // Free rolling, no brake: the hub and the wheel end at one rolling speed that
    // conserves (m u + J ω / R).
    let (m, j, r) = (40.0, 1.0, 0.3);
    let (w, f) = spin_solve(m, 10.0, 0.0, j, r, 0.0, 0.0, f32::MAX, h);
    assert!((w * r - (10.0 + h * f / m)).abs() < 1e-4);
    let p0 = m * 10.0;
    let p1 = m * w * r + j * w / r;
    assert!((p0 - p1).abs() < 1e-3 * p0, "{p0} {p1}");
    // A brake that can hold the wheel: it stops (w = 0), the tyre stops the hub.
    let (w, f) = spin_solve(m, 0.01, 5.0, j, r, 0.0, 1e6, f32::MAX, h);
    assert_eq!(w, 0.0);
    assert!((f - (-(m * 0.01 / h) - 5.0)).abs() < 1e-2);
    // Locked and sliding: the tyre force is its bound.
    let (w, f) = spin_solve(m, 10.0, 0.0, j, r, 0.0, 1e6, 100.0, h);
    assert_eq!(w, 0.0);
    assert_eq!(f, -100.0);
}

/// Mechanical energy: kinetic, gravitational, the beams' elastic energy and the
/// raycast springs' ½k(L₀ − L)² (at the length the last step measured).
fn mechanical(s: &ExplicitBeams) -> (f64, f64) {
    let ke = s
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64)
        .sum::<f64>();
    let pe = s
        .nodes
        .iter()
        .map(|n| n.mass as f64 * 9.807 * (s.origin.y + n.pos.y) as f64)
        .sum::<f64>();
    let springs = s.drive.as_ref().map_or(0.0, |d| {
        d.wheels
            .iter()
            .zip(&d.config.wheels)
            .filter(|(w, _)| w.contact)
            .map(|(w, c)| 0.5 * c.k as f64 * ((c.susp_max - w.length) as f64).powi(2))
            .sum::<f64>()
    });
    (
        ke + pe + engine_softbody::explicit_contact::elastic_energy(s) + springs,
        ke,
    )
}

/// After `settle` s, over 5 s: (largest rise of the mechanical energy, its
/// change, largest kinetic energy, horizontal drift of the centre of mass).
fn rest_ledger(mut s: ExplicitBeams, settle_s: f32) -> (f64, f64, f64, f32) {
    settle(&mut s, settle_s);
    let (x0, _, _) = com(&s);
    let (e0, _) = mechanical(&s);
    let (mut rise, mut kmax) = (f64::MIN, 0.0f64);
    for _ in 0..(5 * 60) {
        s.frame();
        let (e, k) = mechanical(&s);
        rise = rise.max(e - e0);
        kmax = kmax.max(k);
    }
    let (x1, _, _) = com(&s);
    (
        rise,
        mechanical(&s).0 - e0,
        kmax,
        Vec3::new(x1.x - x0.x, 0.0, x1.z - x0.z).length(),
    )
}

#[test]
fn rests_without_creep_and_without_energy_gain() {
    // The same car on its beam tyres: the band a resting explicit vehicle's
    // step-boundary energy moves in (the integrator's own oscillation).
    let tyres =
        ExplicitBeams::from_truck(&sedan(), Vec3::new(0.0, 0.6, 0.0), Vec3::ZERO, settings())
            .unwrap();
    let (t_rise, t_de, t_k, t_drift) = rest_ledger(tyres, 6.0);
    let (rise, de, kmax, drift) = rest_ledger(spawn(Vec3::ZERO), 6.0);
    println!("rest, beam tyres: rise {t_rise:.3e} J, ΔE {t_de:.3e} J, KE max {t_k:.3e} J, drift {t_drift:.2e} m");
    println!("rest, raycast:    rise {rise:.3e} J, ΔE {de:.3e} J, KE max {kmax:.3e} J, drift {drift:.2e} m");
    assert!(drift < 2e-3, "creep {drift} m");
    assert!(de <= 0.0, "energy gained over 5 s: {de} J");
    assert!(
        rise < t_rise.max(0.05) * 2.0,
        "energy rose {rise} J (beam tyres {t_rise} J)"
    );
    // Wheels touch and carry the weight.
    let mut s = spawn(Vec3::ZERO);
    settle(&mut s, 3.0);
    let m = com(&s).2;
    let d = s.drive.as_ref().unwrap();
    let load: f32 = d.wheels.iter().map(|w| w.load).sum();
    assert!(d.wheels.iter().all(|w| w.contact));
    assert!(
        (load - m * 9.807).abs() < 0.03 * m * 9.807,
        "load {load} vs weight {}",
        m * 9.807
    );
}

#[test]
fn accelerates_with_direct_drive_and_brakes_to_a_stop() {
    let mut s = spawn(Vec3::ZERO);
    settle(&mut s, 2.0);
    let fw = forward(&s);
    let (cfg, m) = {
        let d = s.drive.as_ref().unwrap();
        (d.config.clone(), com(&s).2)
    };
    s.set_drive_input(DriveInput {
        forward: 1.0,
        ..Default::default()
    });
    // Closed form of the drive below a third of top speed: the driven wheels'
    // torque over the radius, less what spins up every wheel (J/R² each), against
    // the whole mass; traction-limited at μ·F_n on the driven wheels.
    let driven: Vec<_> = cfg.wheels.iter().filter(|w| w.driven).collect();
    let thrust: f32 = driven
        .iter()
        .map(|w| cfg.max_torque * w.drive_mult / w.radius)
        .sum();
    let m_eff = m + cfg
        .wheels
        .iter()
        .map(|w| w.inertia / (w.radius * w.radius))
        .sum::<f32>();
    let a_drive = thrust / m_eff;
    let mut speeds = Vec::new();
    for _ in 0..(3 * 60) {
        s.frame();
        speeds.push(com(&s).1.dot(fw));
    }
    let v1 = speeds[59];
    let v3 = speeds[179];
    let a = (v3 - v1) / 2.0;
    let mu_peak = 1.2f32.sqrt();
    let w_driven: f32 = s
        .drive
        .as_ref()
        .unwrap()
        .wheels
        .iter()
        .zip(&cfg.wheels)
        .filter(|(_, c)| c.driven)
        .map(|(w, _)| w.load)
        .sum();
    let a_grip = mu_peak * w_driven / m_eff;
    println!("drive: v(1 s) {v1:.2} m/s, v(3 s) {v3:.2} m/s, a {a:.2} m/s², closed form {a_drive:.2} (torque), {a_grip:.2} (grip)");
    assert!(v3 > 5.0, "too slow: {v3} m/s");
    assert!(
        a <= a_drive.min(a_grip) * 1.05,
        "faster than the model allows: {a} vs {}",
        a_drive.min(a_grip)
    );
    assert!(
        a >= 0.5 * a_drive.min(a_grip),
        "much slower than the model: {a} vs {}",
        a_drive.min(a_grip)
    );
    // Brake: full brake from v3; deceleration bounded by μ·g, and the car stops.
    s.set_drive_input(DriveInput {
        brake: 1.0,
        ..Default::default()
    });
    let mut t = 0.0;
    let v_start = com(&s).1.dot(fw);
    while com(&s).1.dot(fw) > 0.05 && t < 8.0 {
        s.frame();
        t += 1.0 / 60.0;
    }
    let decel = v_start / t;
    let g_mu = 1.2f32.sqrt() * 9.807;
    println!("brake: {v_start:.2} m/s to rest in {t:.2} s, {decel:.2} m/s² (μg {g_mu:.2})");
    assert!(t < 8.0, "did not stop");
    assert!(decel <= g_mu * 1.1, "braked harder than the grip: {decel}");
    assert!(decel > 3.0, "weak brakes: {decel}");
    // Stays stopped with the brake held.
    let (x0, _, _) = com(&s);
    for _ in 0..60 {
        s.frame();
    }
    let (x1, v, _) = com(&s);
    assert!(
        (x1 - x0).length() < 0.05 && v.length() < 0.05,
        "rolled {} m",
        (x1 - x0).length()
    );
}

#[test]
fn steady_cornering_turns_with_the_steer() {
    let mut s = spawn(Vec3::ZERO);
    settle(&mut s, 2.0);
    let fw0 = forward(&s);
    s.set_drive_input(DriveInput {
        forward: 0.5,
        ..Default::default()
    });
    for _ in 0..(3 * 60) {
        s.frame();
    }
    // Hold a speed (cruise at partial throttle) and steer left.
    s.set_drive_input(DriveInput {
        forward: 0.25,
        steer: -0.6,
        ..Default::default()
    });
    let mut yaw = Vec::new();
    for _ in 0..(4 * 60) {
        s.frame();
        let f = forward(&s);
        // Unwrapped: the step's increment added to the last angle.
        let last = yaw.last().copied().unwrap_or(0.0f32);
        let a = f32::atan2(fw0.cross(f).y, fw0.dot(f));
        let mut da = a - last.rem_euclid(std::f32::consts::TAU);
        if da > std::f32::consts::PI {
            da -= std::f32::consts::TAU
        }
        if da < -std::f32::consts::PI {
            da += std::f32::consts::TAU
        }
        yaw.push(last + da);
    }
    let (_, v, _) = com(&s);
    let speed = Vec3::new(v.x, 0.0, v.z).length();
    // Steady state over the last second: yaw rate against v / R_kinematic.
    let rate = (yaw[239] - yaw[179]) / 1.0;
    let d = s.drive.as_ref().unwrap();
    let steer_avg: f32 = d
        .wheels
        .iter()
        .filter(|w| w.steer != 0.0)
        .map(|w| w.steer)
        .sum::<f32>()
        / d.wheels.iter().filter(|w| w.steer != 0.0).count().max(1) as f32;
    let kin = speed * steer_avg.tan() / d.config.wheelbase;
    let lat = speed * rate.abs();
    println!("corner: speed {speed:.2} m/s, yaw rate {rate:.3} rad/s, kinematic {kin:.3}, lateral {lat:.2} m/s²");
    assert!(rate > 0.05, "left steer must yaw left (+): {rate}");
    assert!(
        rate <= kin * 1.15,
        "turns tighter than the wheels point: {rate} vs {kin}"
    );
    assert!(lat < 1.2f32.sqrt() * 9.807 * 1.1, "beyond the grip: {lat}");
}

#[test]
#[ignore]
fn debug_corner() {
    let mut s = spawn(Vec3::ZERO);
    settle(&mut s, 2.0);
    let fw0 = forward(&s);
    println!(
        "fw0 {fw0:?} refs {:?}",
        s.drive.as_ref().unwrap().config.refs
    );
    for w in &s.drive.as_ref().unwrap().config.wheels {
        println!(
            "wheel n{} n{} steer {} driven {} side {} hb {}",
            w.node1, w.node2, w.max_steer, w.driven, w.side, w.handbrake
        );
    }
    s.set_drive_input(DriveInput {
        forward: 0.5,
        ..Default::default()
    });
    for _ in 0..(3 * 60) {
        s.frame();
    }
    s.set_drive_input(DriveInput {
        forward: 0.25,
        steer: -0.6,
        ..Default::default()
    });
    for i in 0..(2 * 60) {
        s.frame();
        if i % 6 == 0 {
            let f = forward(&s);
            let d = s.drive.as_ref().unwrap();
            let v = com(&s).1;
            println!(
                "t {:.2} yaw {:.3} v {:.2} vlat {:.2} | {}",
                i as f32 / 60.0,
                f32::atan2(fw0.cross(f).y, fw0.dot(f)),
                v.dot(f),
                v.dot(Vec3::Y.cross(f)),
                d.wheels
                    .iter()
                    .map(|w| format!(
                        "[st {:.2} Fn {:.0} k {:.2} a {:.1} om {:.1}]",
                        w.steer, w.load, w.kappa, w.alpha, w.omega
                    ))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
}

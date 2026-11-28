//! Cars never fuse (owner, 2026-11-28): after a head-on crash both drivers reverse at full
//! throttle and the cars must back apart. Generated sedan, 100 + 100 km/h, full overlap, edge
//! contact on, CPU. Measured with `vehicle_c2c_overlap sedan 100 0.6` and `REVERSE=2.0`
//! (`docs/continuous-contact-design.md`, "Do the cars back apart?"): the default contact fuses
//! (3.7 m apart, 0.01 m/s after 2 s), velocity contact backs apart (42 m, 26 m/s).
//!
//! One process per test binary: the contact mode is read from the environment when the world
//! is built, so each test here sets it before building and the tests must not run together
//! (`backs_apart_default` is ignored).

use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody::explicit_contact::{Dat, ExplicitBeamsWorld};
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody_authoring::authoring::rig;
use engine_softbody_authoring::vehicle_compare::settings;
use glam::Vec3;

/// The two cars' centre distance (m) and separating speed (m/s) after `reverse` s of full
/// reverse following a `crash` s head-on at `kmh` each.
fn crash_then_reverse(kmh: f32, crash: f32, reverse: f32, continuous: bool) -> (f32, f32) {
    let built = rig::rig_vehicle("sedan", None, None).expect("the generated sedan");
    let files = rig::mod_files(&built);
    let refs: Vec<(String, String)> = files.iter().filter(|(p, _)| p.ends_with(".jbeam")).map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned())).collect();
    let pc = files.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let r: Vec<(&str, &str)> = refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let d = engine_softbody::jbeam::load_jbeam(&r, pc.as_deref()).expect("the sedan's jbeam");
    let zmax = d.nodes.iter().map(|n| n.position[2]).fold(f32::MIN, f32::max);
    let mut tb = d.clone();
    for n in &mut tb.nodes {
        n.position = [-n.position[0], n.position[1], -n.position[2]];
    }
    let spawn = |desc: &engine_softbody::desc::SoftVehicleDesc, at: Vec3, vel: Vec3| {
        let lift = 0.02 - desc.nodes.iter().map(|n| n.position[1]).fold(f32::MAX, f32::min);
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(desc, at + Vec3::Y * lift, vel, settings()).expect("spawn");
        s.surface_contact = true;
        s.edge_contact = std::env::var("EDGE_CONTACT").map_or(true, |v| v != "0");
        s
    };
    let v = kmh / 3.6;
    let sa = spawn(&d, Vec3::new(0.0, 0.0, -0.025 - zmax), Vec3::Z * v);
    let sb = spawn(&tb, Vec3::new(0.0, 0.0, 0.025 + zmax), -Vec3::Z * v);
    let mut w = ExplicitBeamsWorld::new(vec![sa, sb]);
    if continuous {
        w.continuous = Some(Dat::new(0.08, 0.85));
    }
    let step = engine_softbody::explicit_beams::STEP * engine_softbody::STEP_DIV as f32;
    for _ in 0..(crash / step).round() as usize {
        w.step();
    }
    for s in &mut w.vehicles {
        s.set_drive_input(DriveInput { forward: -1.0, ..Default::default() });
    }
    for _ in 0..(reverse / step).round() as usize {
        w.step();
    }
    let com = |s: &ExplicitBeams| {
        let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
        (s.origin + s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / m, s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m)
    };
    let ((ca, va), (cb, vb)) = (com(&w.vehicles[0]), com(&w.vehicles[1]));
    let axis = (cb - ca).normalize_or_zero();
    ((cb - ca).length(), (vb - va).dot(axis))
}

fn backs_apart(continuous: bool) {
    let (gap, speed) = crash_then_reverse(100.0, 0.6, 1.0, continuous);
    println!("| backs apart | 100 + 100 km/h, then 1 s of full reverse | centres {gap:.2} m apart, separating at {speed:.2} m/s |");
    // After the crash the centres are about 3.7 m apart; fused cars stay there (0.01 m/s).
    assert!(gap > 6.0 && speed > 2.0, "fused: centres {gap:.2} m apart, separating at {speed:.2} m/s after 1 s of full reverse");
}

/// Velocity contact (`DAT_RESPONSE=vel`): the cars back apart.
#[test]
fn backs_apart_velocity_contact() {
    std::env::set_var("DAT_RESPONSE", "vel");
    backs_apart(true);
}

/// The default contact, a control (run alone; `EDGE_CONTACT=0` turns edge contact off). Measured
/// 2026-11-28: without edge contact it FAILS (fused: 3.57 m, −0.01 m/s); with edge contact it
/// backed apart here (10.1 m, 5.1 m/s) but fused in `vehicle_c2c_overlap` (3.63 m after 2 s), so
/// whether edge contact alone avoids the lock depends on small differences. Velocity contact
/// backed apart in both (here 20.8 m, 17.0 m/s).
#[test]
#[ignore]
fn backs_apart_default() {
    backs_apart(false);
}

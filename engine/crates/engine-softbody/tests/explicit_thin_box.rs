//! A static box narrower than a body's node spacing (`engine/docs/explicit-beams.md`,
//! "1. Nodes", and `explicit-beams-static-box-spec.md`). Box contact is per node,
//! at the node's position, with no swept, beam or triangle test, which is also what
//! Rigs of Rods (RoR) does:
//! * a box between two node lines touches nothing: the body passes untouched;
//! * a node line inside the box, with a lateral margin above the node's entry depth
//!   (about 1.25·u·h), is held on the entry face and stops the body;
//! * a node line inside by less than its entry depth switches to the side face and
//!   slides through.
//!
//! The first tests are self-contained (a four-node plate, `tests/data/
//! thin_box_plate.truck`, the same file RoR's harness reads). The last ones are
//! opt-in: with `RUBBLE_INSPIRATION=<dir>` they run the harness
//! (`<dir>/ror-harness/bin/ror_ref`, the `free` scene) on the E36 against a 20 cm
//! pole, front and side, and check that Rubble stops where RoR stops and passes
//! where RoR passes, with a comparable node distance.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox, STEP};
use glam::Vec3;

const PLATE: &str = include_str!("data/thin_box_plate.truck");

/// The harness's `free` scene: spawn at `pos` with `vel`, an optional box
/// (centre, half extents), `secs` of steps with the origin rebased every 33, no air
/// drag, tyres rolling.
fn free(text: &str, pos: Vec3, vel: Vec3, pole: Option<(Vec3, Vec3)>, secs: f32) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        fast_inv_sqrt: true,
    };
    let mut sim = ExplicitBeams::from_truck(text, pos, vel, settings).expect("truck");
    sim.roll_tyres(vel);
    if let Some((c, e)) = pole {
        sim.boxes.push(StaticBox::centered(c, e));
    }
    let steps = (secs / STEP).round() as u64;
    for i in 0..steps {
        if i % 33 == 0 {
            sim.rebase_origin();
        }
        sim.step();
    }
    sim
}

fn mean(p: &[Vec3]) -> Vec3 {
    p.iter().copied().sum::<Vec3>() / p.len() as f32
}

// The plate: node lines at z = ±0.25, flying along −x at 10 m/s from x = 0.5, 1 m
// up, for 0.3 s (it falls 0.44 m, never reaching the ground). The pole is 0.2 m
// square, centred at x = −0.5, from the ground up to 4 m.
const PLATE_POS: Vec3 = Vec3::new(0.5, 1.0, 0.0);
const PLATE_VEL: Vec3 = Vec3::new(-10.0, 0.0, 0.0);
const PLATE_SECS: f32 = 0.3;

fn plate(pole_z: Option<f32>) -> ExplicitBeams {
    let pole = pole_z.map(|z| (Vec3::new(-0.5, 2.0, z), Vec3::new(0.1, 2.0, 0.1)));
    free(PLATE, PLATE_POS, PLATE_VEL, pole, PLATE_SECS)
}

#[test]
fn a_box_between_node_lines_lets_the_body_pass_untouched() {
    let alone = plate(None).positions();
    let past = plate(Some(0.0)).positions();
    // No node is ever strictly inside: the same run as without the box.
    assert_eq!(past, alone);
    assert!(
        mean(&past).x < -2.0,
        "should have flown past: {:?}",
        mean(&past)
    );
}

#[test]
fn a_node_line_inside_the_box_is_held_on_the_entry_face_and_stops_the_body() {
    // Pole centred on the z = +0.25 line: 0.1 m margin to either side face, far
    // above the entry depth (about 1.25 · 10 m/s · 0.5 ms ≈ 6 mm).
    let face = -0.4; // the pole's +x face
    let mut sim = {
        let pole = (Vec3::new(-0.5, 2.0, 0.25), Vec3::new(0.1, 2.0, 0.1));
        let settings = ExplicitBeamsSettings {
            drag: false,
            turbulence: false,
            fast_inv_sqrt: true,
        };
        let mut s = ExplicitBeams::from_truck(PLATE, PLATE_POS, PLATE_VEL, settings).unwrap();
        s.roll_tyres(PLATE_VEL);
        s.boxes.push(StaticBox::centered(pole.0, pole.1));
        s
    };
    // Node 1 (the front node of the +z line) never gets more than its entry depth
    // past the face while it is inside the pole's lane.
    let steps = (PLATE_SECS / STEP).round() as u64;
    for i in 0..steps {
        if i % 33 == 0 {
            sim.rebase_origin();
        }
        sim.step();
        let p = sim.positions()[1];
        if (0.15..0.35).contains(&p.z) {
            assert!(
                p.x > face - 0.0075,
                "step {i}: node 1 at {p:?} sliced into the pole"
            );
        }
    }
    let end = sim.positions();
    let free_flight = mean(&plate(None).positions()).x;
    // Stopped and turned about the pole, not flown past (free flight ends at −2.35).
    assert!(
        mean(&end).x > -1.0,
        "not stopped: {:?} (free {free_flight})",
        mean(&end)
    );
    assert_eq!(sim.damage(), (0, 0));
}

#[test]
fn a_node_line_inside_by_less_than_its_entry_depth_slides_through() {
    // Pole edge 0.2 mm past the z = +0.25 line: node 1 is strictly inside, nearest
    // the +z face, so the push is sideways only and the plate flies on.
    let alone = mean(&plate(None).positions());
    let past = mean(&plate(Some(0.1502)).positions());
    assert!((past - alone).length() < 1e-3, "{past:?} vs free {alone:?}");
    // A margin above the entry depth holds it (5 mm here).
    let held = mean(&plate(Some(0.155)).positions());
    assert!(held.x > -1.0, "should be held: {held:?}");
}

// --- opt-in: the E36 against Rigs of Rods' harness ---

/// `(ror_ref, E36 truck path)` under `RUBBLE_INSPIRATION`, if set and present.
fn harness() -> Option<(String, String)> {
    let Ok(dir) = std::env::var("RUBBLE_INSPIRATION") else {
        eprintln!("skip: RUBBLE_INSPIRATION not set");
        return None;
    };
    let bin = format!("{dir}/ror-harness/bin/ror_ref");
    let truck = format!("{dir}/ror-repo/e36/E36Sedan.truck");
    if !std::path::Path::new(&bin).exists() || !std::path::Path::new(&truck).exists() {
        eprintln!("skip: {bin} or {truck} not found");
        return None;
    }
    Some((bin, truck))
}

fn v(p: Vec3) -> String {
    format!("{},{},{}", p.x, p.y, p.z)
}

/// RoR's end positions from the harness's `free` scene (1.5 s).
fn ror_free(bin: &str, truck: &str, pos: Vec3, vel: Vec3, pole: (Vec3, Vec3)) -> Vec<Vec3> {
    let out = std::env::temp_dir().join(format!(
        "rubble_thin_box_{}_{}.csv",
        std::process::id(),
        v(pole.0)
    ));
    let status = std::process::Command::new(bin)
        .args([
            truck,
            "free",
            &format!("pos={}", v(pos)),
            &format!("vel={}", v(vel)),
            &format!("box={},{}", v(pole.0), v(pole.1)),
            "roll=1",
            "drag=0",
            "secs=1.5",
            &format!("out={}", out.display()),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("ror_ref");
    assert!(status.success(), "ror_ref failed");
    let text = std::fs::read_to_string(&out).unwrap();
    let _ = std::fs::remove_file(&out);
    text.lines()
        .skip(1)
        .map(|l| {
            let c: Vec<f32> = l.split(',').map(|s| s.parse().unwrap()).collect();
            Vec3::new(c[5], c[6], c[7])
        })
        .collect()
}

/// Rubble against RoR on one pole: the same outcome along `axis` (mean node
/// position within `along` metres: both stopped or both through) and node RMS
/// within `rms` metres.
fn same_outcome(pos: Vec3, vel: Vec3, pole: (Vec3, Vec3), axis: Vec3, along: f32, rms: f32) {
    let Some((bin, truck)) = harness() else {
        return;
    };
    let text = std::fs::read_to_string(&truck).unwrap();
    let ror = ror_free(&bin, &truck, pos, vel, pole);
    let ours = free(&text, pos, vel, Some(pole), 1.5).positions();
    assert_eq!(ours.len(), ror.len());
    let d = (ours
        .iter()
        .zip(&ror)
        .map(|(a, b)| a.distance_squared(*b))
        .sum::<f32>()
        / ours.len() as f32)
        .sqrt();
    let (a, b) = (mean(&ours).dot(axis), mean(&ror).dot(axis));
    eprintln!(
        "pole {:?}: mean along {axis:?} rubble {a:.2} ror {b:.2}, node RMS {d:.3} m",
        pole.0
    );
    assert!(
        (a - b).abs() < along,
        "outcome differs: rubble {a:.2}, ror {b:.2}"
    );
    assert!(d < rms, "node RMS {d:.3} m");
}

const POLE: Vec3 = Vec3::new(0.1, 2.0, 0.1);
/// The coordinator's front case: 60 km/h along −x, spawned 6 cm low (RoR's launch).
const FRONT_POS: Vec3 = Vec3::new(120.0, -0.059531, 100.0);
const FRONT_VEL: Vec3 = Vec3::new(-16.67, 0.0, 0.0);

#[test]
fn e36_front_pole_matches_ror_at_offsets_0_016_0233() {
    // RoR (and Rubble) stop the car at each; a pass-through would end ~10 m further.
    for dz in [0.0, 0.16, 0.233] {
        let pole = (Vec3::new(108.19, 2.0, 100.0 + dz), POLE);
        same_outcome(FRONT_POS, FRONT_VEL, pole, Vec3::X, 1.0, 0.6);
    }
}

#[test]
fn e36_side_pole_matches_ror_at_door_and_between_door_nodes() {
    // 32 km/h sideways (+z), on the ground; the pole beside the +z side. At x = 120
    // a door node line meets it and both engines stop; at x = 120.2 it lies between
    // the door's node lines (119.73 and 120.38) and both pass it untouched.
    let pos = Vec3::new(120.0, 0.1225, 100.0);
    let vel = Vec3::new(0.0, 0.0, 8.89);
    for x in [120.0, 120.2] {
        let pole = (Vec3::new(x, 2.0, 101.6), POLE);
        same_outcome(pos, vel, pole, Vec3::Z, 0.5, 0.3);
    }
}

//! What is seen: one generated sedan's paint never cuts through the other's
//! (`paint_overlap`: the length of the lines where painted triangles cross, m). Two runs a
//! scene, the second car moved 0 and 1 mm (a crash is chaotic: one run proves little); the
//! largest counts. Bound: exact, 0 (two solids share no space), past 1 mm of line.

use engine_softbody_authoring::paint_overlap::{crash, crash_measuring};

/// (the largest crossing at any sample, the largest at the end) of two runs.
fn largest(scene: &str, kmh: f32, secs: f32, edge: bool, press: bool) -> (f32, f32) {
    let r = crash(scene, kmh, secs, 2, edge, press, true).expect("the scene");
    assert!(r.iter().all(|x| x[0] == 0.0 && x.iter().all(|v| v.is_finite())), "clear at spawn and finite: {r:?}");
    let m = |c: usize| r.iter().map(|x| x[c]).fold(0.0f32, f32::max);
    println!("| paint cuts paint | {scene} {kmh} km/h, edge {edge}, press {press}, {secs} s | worst {:.2} m, at the end {:.2} m | 0 |", m(1), m(2));
    (m(1), m(2))
}
const CLEAR: f32 = 1e-3;

/// The measure sees an overlap: the same head-on with vehicle contact off.
#[test]
fn the_measure_sees_cars_through_each_other() {
    let r = crash("head", 60.0, 0.2, 1, false, false, false).expect("the scene");
    assert!(r[0][1] > 50.0, "{r:?}");
}

#[test]
fn edge_contact_side_30_paint_never_cuts() {
    assert!(largest("side", 30.0, 1.0, true, false).0 <= CLEAR);
}

/// The control for a car's own paint: driving at 60 km/h nothing new cuts (past 5 cm of
/// line: shut lines a few millimetres wide rub as the car flexes).
#[test]
fn driving_own_paint_stays_apart() {
    let r = crash("drive", 60.0, 0.5, 1, false, false, true).expect("the scene");
    println!("| own paint | drive 60 km/h | worst {:.3} m | 0.05 |", r[0][4]);
    assert!(r[0][4] <= 0.05, "{r:?}");
}

/// FAILS today (2026-11-25): a car's own panels through each other after a crash (about 5 m
/// of line at a pole, 4 to 8 m after a head-on): self contact is node against face.
#[test]
fn own_paint_ends_apart() {
    for (scene, kmh) in [("pole", 64.0), ("head", 60.0)] {
        let r = crash(scene, kmh, 1.0, 2, true, false, true).expect("the scene");
        let m = r.iter().map(|x| x[5]).fold(0.0f32, f32::max);
        println!("| own paint | {scene} {kmh} km/h | at the end {m:.2} m | 0.05 |");
        assert!(m <= 0.05, "{scene} {kmh}: {m}");
    }
}

/// The owner's bar (2026-11-25), edge contact on, 8 runs: one car's paint inside the other
/// (`paint_overlap::depth`) at most 5 cm at any sample (every 20 ms), above 1 cm for at most
/// 0.2 s before 0.3 s after the hit, and at most 1 cm from then on.
fn accept(scene: &str, kmh: f32, secs: f32, press: bool) {
    let r = crash_measuring(scene, kmh, secs, 8, true, press, true, false, false).expect("the scene");
    let m = |c: usize| r.iter().map(|x| x[c]).fold(0.0f32, f32::max);
    let pass = m(6) <= 0.05 && m(9) <= 0.2 && m(8) <= 0.01;
    println!("| paint inside the other car | {scene} {kmh} km/h, press {press}, 8 runs | worst {:.0} mm, above 1 cm {:.2} s, from 0.3 s after the hit {:.0} mm | 50 mm, 0.2 s, 10 mm | {} |", m(6) * 1e3, m(9), m(8) * 1e3, if pass { "pass" } else { "FAIL" });
    assert!(pass);
}

#[test]
fn accept_side_30() { accept("side", 30.0, 1.0, false); }
#[test]
fn accept_side_60() { accept("side", 60.0, 1.0, false); }
#[test]
fn accept_head_30_pressed() { accept("head", 30.0, 2.0, true); }
#[test]
fn accept_head_60_pressed() { accept("head", 60.0, 2.0, true); }
/// FAILS today (2026-11-25): 52 mm at the hood's front edge.
#[test]
fn accept_head_60() { accept("head", 60.0, 1.0, false); }
/// FAILS today (2026-11-25): 54 mm at the hood's front edge.
#[test]
fn accept_head_100() { accept("head", 100.0, 1.0, false); }
/// FAILS today (2026-11-25): 68 mm, the door's top front corner in the other's hood.
#[test]
fn accept_side_90() { accept("side", 90.0, 1.0, false); }
/// FAILS today (2026-11-25): 34 mm from 0.94 s (a second hit, pressing on).
#[test]
fn accept_side_60_pressed() { accept("side", 60.0, 2.0, true); }

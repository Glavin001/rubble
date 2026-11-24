//! What is seen: one generated sedan's paint never cuts through the other's
//! (`paint_overlap`: the length of the lines where painted triangles cross, m). Two runs a
//! scene, the second car moved 0 and 1 mm (a crash is chaotic: one run proves little); the
//! largest counts. Bound: exact, 0 (two solids share no space), past 1 mm of line.

use engine_softbody_authoring::paint_overlap::crash;

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

#[test]
fn edge_contact_crashes_end_with_paint_clear() {
    for (scene, kmh, secs) in [("head", 60.0, 1.5), ("head", 100.0, 1.5), ("side", 60.0, 2.0)] {
        assert!(largest(scene, kmh, secs, true, false).1 <= CLEAR, "{scene} {kmh}");
    }
}

/// FAILS today (2026-11-24): the crush itself cuts (head on 60: about 3 m, side 60: 1.4 m).
#[test]
fn edge_contact_crashes_paint_never_cuts() {
    for (scene, kmh) in [("head", 60.0), ("head", 100.0), ("side", 60.0)] {
        assert!(largest(scene, kmh, 0.6, true, false).0 <= CLEAR, "{scene} {kmh}");
    }
}

/// FAILS today (2026-11-24): one car pressing on at full throttle (the user's screenshots).
#[test]
fn edge_contact_pressing_ends_with_paint_clear() {
    for (scene, kmh, secs) in [("head", 30.0, 2.5), ("head", 60.0, 2.5), ("side", 60.0, 2.0)] {
        assert!(largest(scene, kmh, secs, true, true).1 <= CLEAR, "{scene} {kmh}");
    }
}

/// FAILS today (2026-11-24): the default (edge contact off), side 60 pressed on: about 2.4 m.
#[test]
fn default_pressing_ends_with_paint_clear() {
    assert!(largest("side", 60.0, 2.0, false, true).1 <= CLEAR);
}

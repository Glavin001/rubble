//! Continuous contact (`docs/continuous-contact-design.md`): the owner's bar (2026-11-25) in
//! every acceptance crash, 8 runs, and the mode's invariant: no edge of one car ever passes
//! through the other's collision triangles (checked every 20 ms, every run). Generated sedan,
//! edge contact on.

use engine_softbody_authoring::paint_overlap::crash_mode;

fn accept(scene: &str, kmh: f32, secs: f32, press: bool) {
    let r = crash_mode(scene, kmh, secs, 8, true, press, true, false, false, Some(true)).expect("the scene");
    let m = |c: usize| r.iter().map(|x| x.0[c]).fold(0.0f32, f32::max);
    let crossed = r.iter().map(|x| x.1).max().unwrap_or(0);
    let pass = m(6) <= 0.05 && m(9) <= 0.2 && m(8) <= 0.01 && crossed == 0;
    println!("| continuous | {scene} {kmh} km/h, press {press}, 8 runs | worst {:.0} mm, above 1 cm {:.2} s, from 0.3 s after the hit {:.0} mm, crossings {crossed}, jitter {:.2} m/s | {} |", m(6) * 1e3, m(9), m(8) * 1e3, m(12), if pass { "pass" } else { "FAIL" });
    assert!(r.iter().all(|x| x.0.iter().all(|v| v.is_finite())), "{r:?}");
    assert!(pass);
}

#[test]
fn continuous_side_30() { accept("side", 30.0, 1.0, false); }
#[test]
fn continuous_side_60() { accept("side", 60.0, 1.0, false); }
#[test]
fn continuous_head_30_pressed() { accept("head", 30.0, 2.0, true); }
#[test]
fn continuous_head_60_pressed() { accept("head", 60.0, 2.0, true); }
#[test]
fn continuous_head_60() { accept("head", 60.0, 1.0, false); }
/// FAILS today (2026-11-26): one run of 8 keeps 33 mm from 0.3 s on (the hoods' front edges
/// wedged inside each other's paint; no crossing of the collision triangles).
#[test]
fn continuous_head_100() { accept("head", 100.0, 1.0, false); }
#[test]
fn continuous_side_90() { accept("side", 90.0, 1.0, false); }
#[test]
fn continuous_side_60_pressed() { accept("side", 60.0, 2.0, true); }
/// Extremes: both cars at 130 km/h head on, and at 100 km/h with 40 % of the width overlapping.
#[test]
fn continuous_head_130() { accept("head", 130.0, 1.0, false); }
#[test]
fn continuous_offset_100() { accept("offset", 100.0, 1.0, false); }

//! Continuous contact (`docs/continuous-contact-design.md`): the owner's bar (2026-11-25) in
//! every acceptance crash, 8 runs, and the mode's invariant: no edge of one car ever passes
//! through the other's collision triangles, and none through its own (checked every 20 ms,
//! every run). Generated sedan,
//! edge contact on.

use engine_softbody_authoring::paint_overlap::crash_mode;

fn accept(scene: &str, kmh: f32, secs: f32, press: bool) {
    let r = crash_mode(scene, kmh, secs, 8, true, press, true, false, false, Some(true)).expect("the scene");
    let m = |c: usize| r.iter().map(|x| x.0[c]).fold(0.0f32, f32::max);
    let crossed = r.iter().map(|x| x.1).max().unwrap_or(0);
    let pass = m(6) <= 0.05 && m(9) <= 0.2 && m(8) <= 0.01 && crossed == 0;
    println!("| continuous | {scene} {kmh} km/h, press {press}, 8 runs | worst {:.0} mm, above 1 cm {:.2} s, from 0.3 s after the hit {:.0} mm, crossings {crossed}, jitter {:.2} m/s | {} |", m(6) * 1e3, m(9), m(8) * 1e3, m(12), if pass { "pass" } else { "FAIL" });
    assert!(r.iter().all(|x| x.0.iter().all(|v| v.is_finite())), "{r:?}");
    // Physics in every run: every impact resolved, the response adds no kinetic energy, and the
    // vehicles' total momentum changes by rounding only (N·s and N·m·s in a step).
    for (k, x) in r.iter().enumerate() {
        let c = x.2.expect("continuous").ccd;
        println!("|   run {k} | unresolved {}, zones {} ({} nodes), soft rounds {} rows {} left {}, response energy {:.0} J, momentum {:.1e} N*s, {:.1e} N*m*s |", c.unresolved, c.zones, c.zone_nodes, c.soft_rounds, c.soft_rows, c.soft_left, c.ke, c.dp, c.dl);
        assert_eq!(c.unresolved, 0, "run {k}");
        assert!(c.ke <= 1e-3, "run {k}: {} J", c.ke);
        assert!(c.dp < 0.01 && c.dl < 2.0, "run {k}: {} {}", c.dp, c.dl);
    }
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
#[test]
fn continuous_head_100() { accept("head", 100.0, 1.0, false); }
#[test]
fn continuous_side_90() { accept("side", 90.0, 1.0, false); }
#[test]
fn continuous_side_60_pressed() { accept("side", 60.0, 2.0, true); }
/// Extremes: both cars at 130 km/h head on, and at 100 km/h with 40 % of the width overlapping.
/// FAILS today (2026-11-26): one run of 8 keeps 17 mm of hood paint drawn past the car's
/// contact surface (inner node `hd_i32`); no crossing.
#[test]
fn continuous_head_130() { accept("head", 130.0, 1.0, false); }
#[test]
fn continuous_offset_100() { accept("offset", 100.0, 1.0, false); }
/// A fast rigid body into a car: a 0.15 m steel ball (111 kg) into the door at 100 km/h.
#[test]
fn continuous_ball_100() { accept("ball", 100.0, 0.5, false); }

/// A measurement (ignored): one run of `CC_SCENE` at `CC_KMH` (km/h) for `CC_SECS` s, its
/// depths, crossings and response counters (`DAT_SOFT=N` compares non-rigid zones).
#[test]
#[ignore]
fn continuous_probe() {
    let e = |k: &str, d: &str| std::env::var(k).unwrap_or(d.into());
    let (scene, kmh, secs) = (e("CC_SCENE", "head"), e("CC_KMH", "100").parse().unwrap(), e("CC_SECS", "1").parse().unwrap());
    let t = std::time::Instant::now();
    let r = crash_mode(&scene, kmh, secs, 1, true, false, true, false, false, Some(true)).expect("the scene");
    let (x, c) = (&r[0].0, r[0].2.expect("continuous").ccd);
    println!("PROBE {scene} {kmh} soft {} | worst {:.1} mm, after 0.3 s {:.1} mm, crossings {}, jitter {:.2} m/s | zones {} ({} nodes), soft rounds {} rows {} left {}, unresolved {}, energy {:.0} J, dp {:.1e}, dl {:.1e} | {:.0} s",
        e("DAT_SOFT", "0"), x[6] * 1e3, x[8] * 1e3, r[0].1, x[12], c.zones, c.zone_nodes, c.soft_rounds, c.soft_rows, c.soft_left, c.unresolved, c.ke, c.dp, c.dl, t.elapsed().as_secs_f32());
}

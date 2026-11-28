//! Velocity contact (`DAT_RESPONSE=vel`, `explicit_contact/ccd_vel.rs`; off by default) held in CI
//! on three short crashes, 2 runs each, generated sedan, edge contact on: a car into a car
//! (head on, 100 km/h, through the crush's first 0.3 s), a car's side (60 km/h, 1 s) and a fast
//! rigid body into a car (a steel ball at 100 km/h, 0.5 s). Its own test binary: the response is
//! chosen once a process, from the environment.
//!
//! Each run: no collision edge through the other car's or its own triangles; the paint guards
//! of the acceptance crashes (`continuous_common`); every impact resolved; the vehicles' linear
//! momentum changed by rounding only and their angular momentum by under 2 N·m·s in a step; no
//! fixed-point sum past its range. (Velocity contact's two CCD holes, 2026-11-27, crossed in
//! head 100's first 0.3 s: `docs/continuous-contact-design.md`.)

mod continuous_common;
use continuous_common::{PAINT_AFTER, PAINT_DURING};
use engine_softbody_authoring::paint_overlap::crash_mode;

fn velocity(scene: &str, kmh: f32, secs: f32) {
    std::env::set_var("DAT_RESPONSE", "vel");
    let r = crash_mode(scene, kmh, secs, 2, true, false, true, false, false, Some(true)).expect("the scene");
    for (k, x) in r.iter().enumerate() {
        let c = x.2.expect("continuous").ccd;
        println!("| velocity | {scene} {kmh} km/h run {k} | paint {:.0} mm, from 0.3 s {:.0} mm, above 1 cm {:.2} s, crossings {} | zones {}, unresolved {}, momentum {:.1e} N*s, {:.1e} N*m*s |", x.0[6] * 1e3, x.0[8] * 1e3, x.0[9], x.1, c.zones, c.unresolved, c.dp, c.dl);
        assert!(x.0.iter().all(|v| v.is_finite()), "run {k}: {:?}", x.0);
        assert_eq!(x.1, 0, "run {k}: collision meshes crossed");
        assert!(x.0[6] <= PAINT_DURING && x.0[8] <= PAINT_AFTER && x.0[9] <= 0.2, "run {k}: paint {:?}", x.0);
        assert_eq!(c.unresolved, 0, "run {k}");
        assert!(c.dp < 0.01 && c.dl < 2.0, "run {k}: momentum {} {}", c.dp, c.dl);
    }
    assert_eq!(engine_softbody::explicit_contact::fixed_overflows(), 0, "fixed-point sums overflowed");
}

#[test]
fn velocity_head_100() { velocity("head", 100.0, 0.3); }
#[test]
fn velocity_side_60() { velocity("side", 60.0, 1.0); }
#[test]
fn velocity_ball_100() { velocity("ball", 100.0, 0.5); }

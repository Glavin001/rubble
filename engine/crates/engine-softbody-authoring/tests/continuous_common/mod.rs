//! The acceptance crashes' bar, shared by `continuous_contact.rs` and `continuous_contact_hard.rs`
//! (two test binaries, so CI's shards can hold them: one ran past a job's 25 minutes).
#![allow(dead_code)]

use engine_softbody_authoring::paint_overlap::crash_mode;

/// The paint's guard during a hit (m): the worst measured, head 130's 66 mm, with a little room.
pub const PAINT_DURING: f32 = 0.07;
/// The paint's guard from 0.3 s after the hit (m): the worst measured, offset 100's 25 mm.
pub const PAINT_AFTER: f32 = 0.03;

pub fn accept(scene: &str, kmh: f32, secs: f32, press: bool) {
    let r = crash_mode(scene, kmh, secs, 8, true, press, true, false, false, Some(true)).expect("the scene");
    let m = |c: usize| r.iter().map(|x| x.0[c]).fold(0.0f32, f32::max);
    let crossed = r.iter().map(|x| x.1).max().unwrap_or(0);
    let pass = crossed == 0 && m(6) <= PAINT_DURING && m(9) <= 0.2 && m(8) <= PAINT_AFTER;
    println!("| continuous | {scene} {kmh} km/h, press {press}, 8 runs | worst {:.0} mm, above 1 cm {:.2} s, from 0.3 s after the hit {:.0} mm, crossings {crossed}, jitter {:.2} m/s | {} |", m(6) * 1e3, m(9), m(8) * 1e3, m(12), if pass { "pass" } else { "FAIL" });
    assert!(r.iter().all(|x| x.0.iter().all(|v| v.is_finite())), "{r:?}");
    // Physics in every run: every impact resolved, the response adds no kinetic energy, and the
    // vehicles' total momentum changes by rounding only (N·s and N·m·s in a step).
    for (k, x) in r.iter().enumerate() {
        let c = x.2.expect("continuous").ccd;
        println!("|   run {k} | depth {:.1} mm (own {:.1}), above 1 cm {:.2} s | unresolved {}, zones {} ({} nodes), soft rounds {} rows {} left {}, response energy {:.0} J, momentum {:.1e} N*s, {:.1e} N*m*s |", x.0[6] * 1e3, x.0[4] * 1e3, x.0[9], c.unresolved, c.zones, c.zone_nodes, c.soft_rounds, c.soft_rows, c.soft_left, c.ke, c.dp, c.dl);
        assert_eq!(c.unresolved, 0, "run {k}");
        assert!(c.ke <= 1e-3, "run {k}: {} J", c.ke);
        assert!(c.dp < 0.01 && c.dl < 2.0, "run {k}: {} {}", c.dp, c.dl);
    }
    // No fixed-point sum past its range anywhere in the run (counted, not wrapped silently).
    assert_eq!(engine_softbody::explicit_contact::fixed_overflows(), 0, "fixed-point sums overflowed");
    assert!(pass);
}


//! Continuous contact (`docs/continuous-contact-design.md`) in every acceptance crash, 8 runs,
//! generated sedan, edge contact on. The owner's bar (2026-11-26):
//! * **hard**: the collision mesh never overlaps: no edge of one car passes through the
//!   other's collision triangles, and none through its own (checked every 20 ms, every run);
//! * **paint, a guard**: the drawn paint (a flexbody over the coarser node-beam mesh) may
//!   stand a little into the other car while the collision meshes are apart; it is held at
//!   or under [`PAINT_DURING`] during a hit, [`PAINT_AFTER`] from 0.3 s after it and over
//!   1 cm for at most 0.2 s, so it cannot quietly get worse (the owner may tighten them).

use engine_softbody_authoring::paint_overlap::crash_mode;

/// The paint's guard during a hit (m): the worst measured, head 130's 66 mm, with a little room.
const PAINT_DURING: f32 = 0.07;
/// The paint's guard from 0.3 s after the hit (m): the worst measured, offset 100's 25 mm.
const PAINT_AFTER: f32 = 0.03;

fn accept(scene: &str, kmh: f32, secs: f32, press: bool) {
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
    println!("PROBE {scene} {kmh} soft {} | worst {:.1} mm, after 0.3 s {:.1} mm, crossings {}, jitter {:.2} m/s | zones {} ({} nodes), soft rounds {} rows {} left {}, unresolved {}, energy {:.0} J, dp {:.1e}, dl {:.1e}, fixed overflows {} | {:.0} s",
        e("DAT_SOFT", "0"), x[6] * 1e3, x[8] * 1e3, r[0].1, x[12], c.zones, c.zone_nodes, c.soft_rounds, c.soft_rows, c.soft_left, c.unresolved, c.ke, c.dp, c.dl, engine_softbody::explicit_contact::fixed_overflows(), t.elapsed().as_secs_f32());
}

/// Contact never stands in for metal (AGENTS.md): over a crash the contact pass changes the
/// vehicles' mechanical energy (kinetic, gravitational and stored in the beams: what it did by
/// moving velocities and positions both) by at most 1 % of their starting kinetic energy, and
/// never raises it; the crash's energy goes into the beams. 2 runs. (The kinetic-only counter,
/// `spec_ke`, also counts energy the pass moves into or out of the beams by its position
/// steps; it is printed beside.)
fn energy(scene: &str, kmh: f32, secs: f32) {
    // (The measurement of the pass's mechanical energy, `ExplicitBeamsWorld::dat_truncate_measured`.)
    std::env::set_var("DAT_ENERGY", "1");
    let r = crash_mode(scene, kmh, secs, 2, true, false, true, false, false, Some(true)).expect("the scene");
    let mut worst = 0.0f64;
    for (k, x) in r.iter().enumerate() {
        let d = x.2.expect("continuous");
        let share = -d.contact_de / d.ke_start.max(1.0);
        println!("| energy | {scene} {kmh} km/h run {k} | mechanical energy changed by contact {:.1} kJ ({:.2} % of the start's kinetic); contact's work {:.1} kJ (DAT_RESPONSE=vel) |", d.contact_de * 1e-3, -100.0 * share, d.contact_work * 1e-3);
        assert!(d.contact_de <= 1e-3 * d.ke_start, "{scene} {kmh} run {k}: contact created {} J", d.contact_de);
        println!("| energy | {scene} {kmh} km/h run {k} | start {:.0} kJ, contact removed {:.1} kJ ({:.1} %; CCD response {:.1} kJ; guard scaled e down in {} of {} steps; zones {}) |", d.ke_start * 1e-3, -d.spec_ke * 1e-3, 100.0 * share, -d.ccd.ke * 1e-3, d.guarded, d.steps, d.ccd.zones);
        worst = worst.max(share);
    }
    assert!(worst <= 0.01, "{scene} {kmh}: contact removed {:.1} % of the starting kinetic energy (at most 1 %)", 100.0 * worst);
}

/// FAILS today (2026-11-26): the inelastic impacts delete the struck nodes' closing energy.
#[test]
fn continuous_energy_head_100() { energy("head", 100.0, 0.6); }
/// FAILS today (2026-11-26), as `continuous_energy_head_100`.
#[test]
fn continuous_energy_side_60() { energy("side", 60.0, 0.6); }
/// FAILS today (2026-11-26), as `continuous_energy_head_100`.
#[test]
fn continuous_energy_ball_100() { energy("ball", 100.0, 0.5); }

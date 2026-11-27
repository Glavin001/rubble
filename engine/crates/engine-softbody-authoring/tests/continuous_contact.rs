//! Continuous contact (`docs/continuous-contact-design.md`) in every acceptance crash, 8 runs,
//! generated sedan, edge contact on. The owner's bar (2026-11-26):
//! * **hard**: the collision mesh never overlaps: no edge of one car passes through the
//!   other's collision triangles, and none through its own (checked every 20 ms, every run);
//! * **paint, a guard**: the drawn paint (a flexbody over the coarser node-beam mesh) may
//!   stand a little into the other car while the collision meshes are apart; it is held at
//!   or under [`PAINT_DURING`] during a hit, [`PAINT_AFTER`] from 0.3 s after it and over
//!   1 cm for at most 0.2 s, so it cannot quietly get worse (the owner may tighten them).

mod continuous_common;
use continuous_common::*;
use engine_softbody_authoring::paint_overlap::crash_mode;

#[test]
fn continuous_side_30() { accept("side", 30.0, 1.0, false); }
#[test]
fn continuous_side_60() { accept("side", 60.0, 1.0, false); }
#[test]
fn continuous_head_30_pressed() { accept("head", 30.0, 2.0, true); }
#[test]
fn continuous_head_60() { accept("head", 60.0, 1.0, false); }
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
    let press = e("CC_PRESS", "0") == "1";
    let runs: usize = e("CC_RUNS", "1").parse().unwrap();
    let r = crash_mode(&scene, kmh, secs, runs, true, press, true, false, false, Some(true)).expect("the scene");
    let (x, c) = (&r[0].0, r[0].2.expect("continuous").ccd);
    println!("PROBE {scene} {kmh} soft {} | worst {:.1} mm, after 0.3 s {:.1} mm, crossings {}, jitter {:.2} m/s | zones {} ({} nodes, {} too spun), soft rounds {} rows {} left {}, unresolved {}, energy {:.0} J, dp {:.1e}, dl {:.1e}, fixed overflows {} | exact {} ({} rows, {} pivots, {} too big, {} left) | {:.0} s",
        e("DAT_SOFT", "0"), x[6] * 1e3, x[8] * 1e3, r[0].1, x[12], c.zones, c.zone_nodes, c.zone_spin, c.soft_rounds, c.soft_rows, c.soft_left, c.unresolved, c.ke, c.dp, c.dl, engine_softbody::explicit_contact::fixed_overflows(), c.exact, c.exact_rows, c.exact_pivots, c.exact_big, c.exact_left, t.elapsed().as_secs_f32());
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
        println!("| energy | {scene} {kmh} km/h run {k} | mechanical energy changed by contact {:.1} kJ ({:.2} % of the start's kinetic); contact's work {:.1} kJ (DAT_RESPONSE=vel: rows {:.1} kJ, zones {:.1} kJ, zones {}, passes e lowered {}, rows restituted {}) |", d.contact_de * 1e-3, -100.0 * share, d.contact_work * 1e-3, d.ccd.work_rows * 1e-3, d.ccd.work_zones * 1e-3, d.ccd.zones, d.ccd.e_scaled, d.ccd.restituted);
        // Velocity contact (DAT_RESPONSE=vel) moves positions only by its velocity change, so the
        // integrator's conserved energy is its measure, the contact's work (as the elastic bars':
        // `contact_known_solutions.rs`); the mechanical energy also counts the stiff beams' h·Δv
        // share of the step.
        let vel = std::env::var("DAT_RESPONSE").is_ok_and(|v| v == "vel");
        let (made, share) = if vel { (d.contact_work, -d.contact_work / d.ke_start.max(1.0)) } else { (d.contact_de, share) };
        assert!(made <= 1e-3 * d.ke_start, "{scene} {kmh} run {k}: contact created {made} J");
        println!("| energy | {scene} {kmh} km/h run {k} | start {:.0} kJ, contact removed {:.1} kJ ({:.1} %; CCD response {:.1} kJ; guard scaled e down in {} of {} steps; zones {}) |", d.ke_start * 1e-3, -d.spec_ke * 1e-3, 100.0 * share, -d.ccd.ke * 1e-3, d.guarded, d.steps, d.ccd.zones);
        worst = worst.max(share);
    }
    // (Velocity contact: the owner's target, rule 8 of the deciding-tradeoffs skill, 2 %.)
    let bar = if std::env::var("DAT_RESPONSE").is_ok_and(|v| v == "vel") { 0.02 } else { 0.01 };
    assert!(worst <= bar, "{scene} {kmh}: contact removed {:.1} % of the starting kinetic energy (at most {:.0} %)", 100.0 * worst, 100.0 * bar);
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

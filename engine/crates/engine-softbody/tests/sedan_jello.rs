//! The generated sedan is not jello: its panels' first modes, its wobble over
//! a kerb, how fast that dies and how still it stands, against the ETK
//! K-series measured in the same harness (`examples/sedan_stiffness.rs`, whose
//! measurements this test calls; CPU reference, raycast wheels, the centred
//! damping's objective form on).
//!
//! **The bar (the user's, plan 2e): the ETK's level, no jello.**
//!
//! - **First modes** (each panel on the car, its neighbours held; bilateral
//!   springs only): at least the lower end of the band the five BeamNG cars
//!   span (`beamng-structure-study.md` §10.6).
//! - **Kerb** (5 cm × 0.5 m at 30 km/h, both axles): the peak ripple of each
//!   category's nodes about the body's rigid fit (√2·σ per 0.1 s window) at
//!   most [`LEVEL`] times the ETK's, and under 0.2 mm no later than the ETK
//!   plus one window.
//! - **Rest** (4–5 s after spawn): no node moves about its own mean (the ETK:
//!   0.00 mm).
//!
//! The ETK is community content and never in the repository: its numbers are
//! recorded here from `RUBBLE_CENTRED_OBJECTIVE=1 RAYCAST=1 sedan_stiffness
//! etkk` (2026-11-13, `kr6.pc`).
#[allow(dead_code)]
#[path = "../examples/sedan_stiffness.rs"]
mod harness;

use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::vehicle_compare::{Car, CarSpec};
use harness::CATS;
use std::sync::OnceLock;

/// "At the ETK's level": within a quarter of its measured value.
const LEVEL: f64 = 1.25;
/// The ETK's kerb ripple about its body (mm), by [`CATS`]: body, door, hood,
/// trunk, fender, bumper.
const ETK_KERB_MM: [f64; 6] = [0.09, 0.15, 0.23, 0.21, 0.09, 0.18];
/// The ETK's time to stay under 0.2 mm after the kerb (s), by [`CATS`].
const ETK_SETTLE_S: [f64; 6] = [0.0, 0.0, 0.3, 0.6, 0.0, 0.0];
/// First-mode floors (Hz): the study's bands' lower ends (§10.6); the ETK's
/// are 34.9, 12.5, 27.0, 41.8 and 23.8 / 9.1.
const MODE_FLOOR_HZ: [f64; 6] = [0.0, 16.0, 11.0, 23.0, 17.0, 9.0];
/// Rest: the largest move of a node about its own mean (mm).
const REST_MAX_MM: f64 = 0.05;

struct Measured {
    modes: Vec<(usize, String, f64)>,
    kerb_mm: [f64; 6],
    settle_s: [f64; 6],
    rest_max_mm: f64,
    rest_rms_mm: f64,
}

fn measured() -> &'static Measured {
    static M: OnceLock<Measured> = OnceLock::new();
    M.get_or_init(|| {
        // The sedan as the rigger exports it, as a mod folder the harness loads.
        let inp = rig::DesignInput::find("sedan", None, None).expect("the sedan's design");
        let r = lattice::rig_with(&inp);
        let mods = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("sedan_jello_mods");
        let dir = mods.join("sedan_jello");
        let _ = std::fs::remove_dir_all(&dir);
        for (p, data) in vg_input::physics_files(&inp, &r) {
            let path = dir.join(&p);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, data).unwrap();
        }
        std::env::set_var("RUBBLE_COMPARE_MODS", &mods);
        std::env::set_var("RUBBLE_CENTRED_OBJECTIVE", "1");
        std::env::set_var("RAYCAST", "1");
        let spec: CarSpec = serde_json::from_value(serde_json::json!({
            "name": "sedan", "mod": "sedan_jello", "vehicle": "sedan", "pc": "sedan.pc"
        }))
        .unwrap();
        let car = Car::load(&spec).expect("the sedan loads");
        let (_, cats) = harness::classify(&car);
        let modes = harness::panel_first_modes(&car, &cats);
        let sets = harness::sets_of(&car, &cats);
        let zero = |_: f32| glam::Vec3::ZERO;
        let mut base = harness::spawn(&car);
        harness::run(&car, &sets, &mut base, 4.0, &zero, false);
        let (rms, max) = harness::rest_wobble(&base, &sets.nontyre);
        let mut s2 = harness::kerb_start(&base);
        let k = harness::run(&car, &sets, &mut s2, 3.0, &zero, false);
        let mut kerb_mm = [0.0; 6];
        let mut settle_s = [0.0; 6];
        for c in 0..CATS.len() {
            kerb_mm[c] = harness::peak(&k.body[c]) * 1e3;
            settle_s[c] = harness::settle_time(&k.body[c], 0, 2e-4);
        }
        let m = Measured { modes, kerb_mm, settle_s, rest_max_mm: max * 1e3, rest_rms_mm: rms * 1e3 };
        println!("first modes (Hz): {}", m.modes.iter().map(|(_, p, f)| format!("{p} {f:.1}")).collect::<Vec<_>>().join(", "));
        println!("kerb ripple about the body (mm) {CATS:?}: {:.2?}; the ETK {ETK_KERB_MM:?}", m.kerb_mm);
        println!("kerb settle under 0.2 mm (s): {:.1?}; the ETK {ETK_SETTLE_S:?}", m.settle_s);
        println!("rest 4-5 s: RMS {:.3} mm, max {:.2} mm", m.rest_rms_mm, m.rest_max_mm);
        m
    })
}

#[test]
fn panel_first_modes_reach_the_beamng_bands() {
    let m = measured();
    let low: Vec<String> = m
        .modes
        .iter()
        .filter(|(c, _, f)| *f < MODE_FLOOR_HZ[*c])
        .map(|(c, p, f)| format!("{p} {f:.1} Hz (floor {})", MODE_FLOOR_HZ[*c]))
        .collect();
    assert!(low.is_empty(), "first modes under the band: {}", low.join(", "));
}

#[test]
fn kerb_wobble_is_at_the_etk_level() {
    let m = measured();
    let over: Vec<String> = (0..CATS.len())
        .filter(|&c| m.kerb_mm[c] > LEVEL * ETK_KERB_MM[c])
        .map(|c| format!("{} {:.2} mm (the ETK {:.2})", CATS[c], m.kerb_mm[c], ETK_KERB_MM[c]))
        .collect();
    assert!(over.is_empty(), "kerb ripple over {LEVEL} x the ETK's: {}", over.join(", "));
}

#[test]
fn kerb_wobble_dies_as_fast_as_the_etks() {
    let m = measured();
    let over: Vec<String> = (0..CATS.len())
        .filter(|&c| m.settle_s[c] > ETK_SETTLE_S[c] + 0.1 + 1e-9)
        .map(|c| format!("{} {:.1} s (the ETK {:.1})", CATS[c], m.settle_s[c], ETK_SETTLE_S[c]))
        .collect();
    assert!(over.is_empty(), "still over 0.2 mm after the ETK has settled: {}", over.join(", "));
}

#[test]
fn the_parked_car_stands_still() {
    let m = measured();
    assert!(
        m.rest_max_mm <= REST_MAX_MM,
        "a node moves {:.2} mm about its mean 4-5 s after spawn (limit {REST_MAX_MM}; the ETK 0.00)",
        m.rest_max_mm
    );
}

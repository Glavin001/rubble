//! The fracture's quiet gate skips only what would change nothing.
//!
//! A step in which no contact row touches a chunk and no chunk carries a pulse, after two
//! such steps, dispatches none of the ingest merge's, the loads' and the threshold breaks'
//! groups (stress-gpu `GpuFracture::encode_frame_merged`). That is lossless only if every
//! word those passes write already holds what they would write. Here every skipped frame of
//! a small city is read back and checked for exactly that, through a drive past sleeping
//! houses, a war
//! (shots and breaks: the gate must stand aside) and the quiet after it.
//!
//!   cargo test --release -p engine-world --test quiet_gate -- --test-threads=1 --nocapture

use rubble::city::*;
use rubble::*;

fn step(w: &mut World, broken: &mut u32) -> (bool, usize) {
    w.step().unwrap();
    for f in w.poll_blocking().unwrap() {
        *broken += bonds_broken(&f.destruction_events);
    }
    w.stress_quiet_frame().expect("a shared fracture")
}

#[test]
fn a_skipped_frame_would_have_changed_nothing() {
    let asset = city_house().unwrap();
    let def = CityDef {
        houses: 4,
        map: 240.0,
        sleep: true,
        pool: Some(4096),
    };
    let mut w = World::new(GpuContext::Default, &def.world_def(&asset).unwrap()).unwrap();
    let mut c = City::begin(&mut w, &asset, def).unwrap();
    c.place(&mut w, 4).unwrap();
    let mut broken = 0;
    let (mut skipped, mut ran) = (0u32, 0u32);
    let mut tally = |(skip, wrong): (bool, usize), step: u32, phase: &str| {
        assert_eq!(wrong, 0, "{phase} step {step}: a skipped frame would have written {wrong} words differently");
        if skip {
            skipped += 1;
        } else {
            ran += 1;
        }
        skip
    };
    // Settling, then a car driving down the street: the houses sleep, the car keeps the
    // fracture's frame running, and nothing touches a chunk.
    for i in 0..CITY_SETTLE + 60 {
        tally(step(&mut w, &mut broken), i, "settle");
    }
    let settled = broken;
    c.start(&mut w, CityScenario::Drive).unwrap();
    let mut quiet_before = 0;
    for i in 0..120 {
        quiet_before += tally(step(&mut w, &mut broken), i, "drive") as u32;
    }
    assert!(quiet_before > 100, "{quiet_before} of 120 frames were skipped while a car drove past sleeping houses");
    assert_eq!(broken, settled, "the drive broke bonds");
    // War: shots strike the houses, bonds break, pieces fall on chunks.
    c.start(&mut w, CityScenario::War).unwrap();
    let mut loud = 0;
    for i in 0..360 {
        loud += !tally(step(&mut w, &mut broken), i, "war") as u32;
    }
    assert!(broken > settled, "the war broke no bond");
    assert!(loud > 0, "every frame of the war was skipped");
    // And the quiet after it, once the rubble sleeps.
    let mut quiet_after = 0;
    for i in 0..900 {
        quiet_after += tally(step(&mut w, &mut broken), i, "after") as u32;
    }
    eprintln!("skipped {skipped} frames, ran {ran}; broken {broken} ({settled} settling); quiet before {quiet_before}, loud {loud}, quiet after {quiet_after}");
}

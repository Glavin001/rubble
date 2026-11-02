//! The threshold fracture's listed loads and breaks (`GpuFracture::set_listed`, the
//! default) give exactly the full passes' results, on the same GPU state, step after step.
//!
//! A small city (`rubble::city`) with the large-map sleep policy: houses settle and fall
//! asleep, then 20 shots strike them (`war`), more houses are placed mid-run (a relist:
//! placement reuses no state, but the listed passes must start over), and the debris
//! settles. On every checked step `World::set_fracture_check` runs the ingest's full
//! merge and the full `fr_loads` / `fr_breaks` on a copy of the frame's state before the
//! frame runs listed; `World::fracture_check` then compares every chunk's load and pulses,
//! every bond's state, the counters, the bodies marked for a split and the bonds broken.
#![cfg(feature = "stress")]

use rubble::city::*;
use rubble::*;

#[test]
fn listed_fracture_passes_match_the_full_passes() {
    let house = city_house().unwrap();
    let def = CityDef {
        houses: 24,
        map: 160.0,
        sleep: true,
        pool: Some(4096),
    };
    let mut w = World::new(GpuContext::Default, &def.world_def(&house).unwrap()).unwrap();
    let mut c = City::begin(&mut w, &house, def).unwrap();
    c.place(&mut w, 16).unwrap();
    let (mut checked, mut broken, mut n) = (0, 0, 0);
    let mut step = |w: &mut World, check: bool| {
        w.set_fracture_check(check);
        w.step().unwrap();
        n += 1;
        for f in w.poll_blocking().unwrap() {
            broken += bonds_broken(&f.destruction_events);
        }
        if w.fracture_check()
            .unwrap_or_else(|e| panic!("step {n}: {e}"))
        {
            checked += 1;
        }
    };
    for i in 0..60 {
        step(&mut w, i % 4 == 0);
    }
    c.start(&mut w, CityScenario::War).unwrap();
    for i in 0..200 {
        if i == 40 {
            // More houses mid-run: the next frame relists every live chunk.
            c.place(&mut w, 8).unwrap();
        }
        step(&mut w, i < 120 || i % 3 == 0);
    }
    assert!(checked > 100, "{checked} steps checked");
    assert!(broken > 100, "the shots broke {broken} bonds");
    eprintln!("{checked} steps checked, {broken} bonds broken");
}

/// The pose readback sized to the last step's moved count plus headroom (the rest read
/// from the slot's spill when a step moves more) delivers every frame's moved records
/// byte for byte as the full copy (`World::set_readback_check`): the same city, settling,
/// struck by 20 shots (a step that wakes hundreds of bodies overflows its slot's copy).
#[test]
fn sized_pose_readback_matches_the_full_copy() {
    let house = city_house().unwrap();
    let def = CityDef {
        houses: 16,
        map: 160.0,
        sleep: true,
        pool: Some(4096),
    };
    let mut w = World::new(GpuContext::Default, &def.world_def(&house).unwrap()).unwrap();
    let mut c = City::begin(&mut w, &house, def).unwrap();
    c.place(&mut w, 16).unwrap();
    w.set_readback_check(true);
    let mut moved = 0;
    for i in 0..300 {
        if i == 60 {
            c.start(&mut w, CityScenario::War).unwrap();
        }
        w.step().unwrap();
        // Every other step polled without waiting: frames left in flight.
        let frames = if i % 2 == 0 {
            w.poll().unwrap()
        } else {
            w.poll_blocking().unwrap()
        };
        moved += frames
            .iter()
            .map(|f| f.body_events.moves.len())
            .sum::<usize>();
    }
    w.poll_blocking().unwrap();
    let (checked, differing, spills) = w.readback_check();
    eprintln!(
        "{checked} frames checked, {differing} differing, {spills} spilled, {moved} body moves"
    );
    assert_eq!(
        differing, 0,
        "{differing} of {checked} frames differ from the full copy"
    );
    assert!(checked > 250, "{checked} frames checked");
    assert!(spills > 50, "{spills} steps overflowed their slot's copy");
}

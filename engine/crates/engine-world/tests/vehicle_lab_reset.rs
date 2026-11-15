//! The vehicle lab's in-place restart (`vehicle_lab::Lab::restart`,
//! `World::reset_explicit_vehicles`, `ExplicitBeamsGpu::reset_state`): a lab that has been
//! used (a shot fired into the car, a crash preset run) and then restarted steps bit for
//! bit as a lab just built, without a new device, buffer or pipeline.
#![cfg(feature = "stress")]

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::explicit_coupling::read_words;
use rubble::vehicle_lab::{Lab, LabOptions};
use rubble::*;
use serde_json::json;

fn car() -> ExplicitBeams {
    let text = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
        minimass: 10.0,
        ..engine_softbody::procedural::CarParams::sedan()
    });
    ExplicitBeams::from_truck_raycast(
        &text,
        None,
        glam::Vec3::ZERO,
        0.0,
        glam::Vec3::ZERO,
        ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            ..Default::default()
        },
    )
    .unwrap()
    .0
}

fn run(world: &mut World, lab: &mut Lab, windows: u32) {
    for _ in 0..windows {
        lab.before_step(world).unwrap();
        world.step().unwrap();
        world.poll().unwrap();
        lab.read_blocking(world, true).unwrap();
    }
    world.poll_blocking().unwrap();
}

/// The car's node and beam records and its origin, as words.
fn state(world: &World, lab: &Lab) -> Vec<u32> {
    let x = world.explicit_vehicles().unwrap();
    let p = lab.read_plan();
    let (d, q) = (world.device(), world.queue());
    let mut w = read_words(d, q, x.nodes_buffer(), p.nodes.0, p.nodes.1);
    w.extend(read_words(d, q, x.beams_buffer(), p.beams.0, p.beams.1));
    w.extend(read_words(d, q, x.misc_buffer(), p.origin.0, p.origin.1));
    w
}

#[test]
fn a_restarted_lab_steps_bit_for_bit_as_a_new_one() {
    let template = car();
    let (mut wa, mut la) = Lab::build(
        GpuContext::Default,
        &template,
        LabOptions::default(),
        |_| {},
    )
    .unwrap();
    run(&mut wa, &mut la, 40);
    let fresh = state(&wa, &la);

    let (mut wb, mut lb) = Lab::build(
        GpuContext::Default,
        &template,
        LabOptions::default(),
        |_| {},
    )
    .unwrap();
    lb.read_blocking(&wb, true).unwrap();
    // Use it: a steel box into the door, then a wall at 40 km/h.
    lb.command(&mut wb, &json!({ "fire": { "shape": "box", "size": [0.3, 0.2, 0.4], "speed": 30, "aim": { "side": "left", "distance": 3 } } })).unwrap();
    run(&mut wb, &mut lb, 30);
    assert!(lb.bench.shots[0].impact.is_some(), "the shot struck");
    lb.restart(&mut wb, &json!({ "crash": { "kind": "wall", "kmh": 40 } }))
        .unwrap();
    run(&mut wb, &mut lb, 30);
    assert!(
        lb.bench.car.beams_broken + lb.bench.car.beams_yielded > 0,
        "the wall damaged the car"
    );
    let arrival = lb.bench.arrival.expect("the car reached the wall");
    assert!(
        (arrival * 3.6 - 40.0).abs() < 1.5,
        "it arrived at the stated speed: {} km/h",
        arrival * 3.6
    );
    let bodies = wb.body_count();
    // Back to the lab's own start, in place.
    let t = std::time::Instant::now();
    lb.restart(&mut wb, &json!({})).unwrap();
    let took = t.elapsed();
    assert!(wb.body_count() < bodies, "the wall is gone");
    run(&mut wb, &mut lb, 40);
    let again = state(&wb, &lb);
    assert_eq!(fresh.len(), again.len());
    let differ = fresh.iter().zip(&again).filter(|(a, b)| a != b).count();
    assert_eq!(
        differ,
        0,
        "{differ} of {} words differ after the restart",
        fresh.len()
    );
    assert_eq!(lb.bench.car.beams_broken, 0);
    assert!(took.as_millis() < 500, "a restart took {took:?}");
}

#[test]
fn a_restart_of_another_fleet_is_refused() {
    let template = car();
    let (mut w, _lab) = Lab::build(
        GpuContext::Default,
        &template,
        LabOptions::default(),
        |_| {},
    )
    .unwrap();
    assert!(w
        .reset_explicit_vehicles(&[template.clone(), template.clone()])
        .is_err());
}

#[test]
fn the_side_pole_sled_arrives_at_its_stated_speed() {
    let template = car();
    let (mut w, mut lab) = Lab::build(
        GpuContext::Default,
        &template,
        LabOptions::default(),
        |_| {},
    )
    .unwrap();
    lab.restart(
        &mut w,
        &json!({ "crash": { "kind": "sidepole", "kmh": 32 } }),
    )
    .unwrap();
    run(&mut w, &mut lab, 30);
    let arrival = lab.bench.arrival.expect("the car reached the pole") * 3.6;
    assert!(
        (arrival - 32.0).abs() < 1.0,
        "the sled delivered the car at {arrival} km/h"
    );
}

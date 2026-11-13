//! Destruction approaches side by side (`rubble::play::compare`): each scenario under each
//! approach, and what tells them apart.
//!
//!   cargo test --release -p engine-world --features stress --test compare -- --test-threads=1 --nocapture
#![cfg(feature = "stress")]

use rubble::play::compare::*;
use rubble::*;

fn run(scenario: Scenario, approach: Approach) -> Verdict {
    let (mut world, mut c) = Compare::new(GpuContext::Default, scenario, approach).unwrap_or_else(|e| panic!("{} / {}: {e:#}", scenario.name(), approach.name()));
    c.queue_script(&world).unwrap();
    let steps = (scenario.seconds() * scenario.physics_hz() as f64) as usize;
    let mut v = None;
    for _ in 0..steps {
        c.step(&mut world).unwrap();
        v = Some(c.verdict(&world).unwrap());
    }
    let v = v.unwrap();
    println!("{:>15} / {:<16}: {} of {} bonds broken, first at {:?} s, drop {:.2} m{}", scenario.name(), approach.name(), v.bonds_broken, v.bonds, v.first_break.map(|t| (t * 100.0).round() / 100.0), v.drop, if v.fell { " (fell)" } else { "" });
    let r = v.report.as_ref().unwrap();
    println!("{:>34}  bounce {} crater {} spall {} perforate {}; peak utilization {:.2}; {:?}", "", r.bounces, r.craters, r.spalls, r.perforations, r.peak_utilization, r.strikes);
    for s in &v.shots {
        println!("{:>34}: in {:.0} m/s, now {:.1} m/s ({:+.1} along), broke {}{}", s.label, s.launch_speed, s.speed, s.along, s.broke, if s.through { ", through" } else { "" }.to_string() + if s.wall_fell { ", wall down" } else { "" });
    }
    v
}

#[test]
fn print_the_matrix() {
    let only = std::env::var("COMPARE_SCENARIO").ok();
    for s in Scenario::ALL {
        if only.as_deref().is_some_and(|o| o != s.name()) {
            continue;
        }
        let one = std::env::var("COMPARE_APPROACH").ok();
        for a in Approach::ALL {
            if one.as_deref().is_some_and(|o| o != a.name()) {
                continue;
            }
            run(s, a);
        }
    }
}

/// The stress view: on the balcony under the load path, the root chunks carry the largest
/// load over strength and it rises when the weight lands; the breaks are the same with the
/// view on or off.
#[test]
fn the_stress_view_shows_the_root_and_changes_nothing() {
    let run = |view: bool| -> (u32, Option<f64>, f32, f32) {
        let (mut world, mut c) = Compare::new(GpuContext::Default, Scenario::Balcony, Approach::StressSolver).unwrap();
        world.set_stress_view(view);
        c.queue_script(&world).unwrap();
        let d = c.lab.destructibles()[0];
        let (mut idle, mut loaded) = (0.0f32, 0.0f32);
        for step in 0..360 {
            c.step(&mut world).unwrap();
            let peak = world.destructible_chunk_stress(d).unwrap().map_or(0.0, |v| v.iter().flatten().fold(0.0f32, |a, &b| a.max(b)));
            if step == 40 {
                idle = peak;
            }
            loaded = loaded.max(peak);
        }
        let v = c.verdict(&world).unwrap();
        (v.bonds_broken, v.first_break, idle, loaded)
    };
    let off = run(false);
    let on = run(true);
    println!("view off {off:?}; on {on:?}");
    assert_eq!((off.0, off.1), (on.0, on.1), "the same breaks at the same time");
    assert_eq!((off.2, off.3), (0.0, 0.0), "no view, no numbers");
    assert!(on.2 > 0.05 && on.2 < 1.0, "the balcony's own weight loads its root below strength ({})", on.2);
    assert!(on.3 >= 1.0, "the wrecking ball takes a bond to its strength ({})", on.3);
}

/// The contact-force rule's stress view: nothing until something presses on the wall, the
/// struck chunks above their strength when the shot lands, and the same breaks with the
/// view on or off.
#[test]
fn the_threshold_stress_view_shows_the_strike_and_changes_nothing() {
    let run = |view: bool| -> (u32, Option<f64>, f32, f32) {
        let (mut world, mut c) = Compare::new(GpuContext::Default, Scenario::EqualMomentum, Approach::ImpactThreshold).unwrap();
        world.set_stress_view(view);
        c.queue_script(&world).unwrap();
        let d = c.lab.destructibles()[0];
        let (mut idle, mut struck) = (0.0f32, 0.0f32);
        for step in 0..150 {
            c.step(&mut world).unwrap();
            let peak = world.destructible_chunk_stress(d).unwrap().map_or(0.0, |v| v.iter().flatten().fold(0.0f32, |a, &b| a.max(b)));
            if step == 30 {
                idle = peak;
            }
            struck = struck.max(peak);
        }
        let v = c.verdict(&world).unwrap();
        (v.bonds_broken, v.first_break, idle, struck)
    };
    let off = run(false);
    let on = run(true);
    println!("view off {off:?}; on {on:?}");
    assert_eq!((off.0, off.1), (on.0, on.1), "the same breaks at the same time");
    assert!(on.2 < 1.0, "a wall at rest is below its strength ({})", on.2);
    assert!(on.3 >= 1.0, "the shot takes a bond past its strength ({})", on.3);
}

/// The GPU stress solver's stress view (`BlastGpu`) on the balcony: loaded below its fatal
/// limit by its own weight, and the same breaks with the view on or off.
#[test]
fn the_blast_gpu_stress_view_shows_the_load_and_changes_nothing() {
    use rubble::lab::{Lab, LabBonds, LabPlan};
    let run = |view: bool| -> (u32, f32, f32) {
        let mut def = Scenario::Balcony.lab_def(Approach::StressSolver);
        def.settings.bonds = LabBonds::Model(StrengthModel::BlastGpu);
        def.settings.fracture = FractureOptions::default();
        let plan = LabPlan::new(&def).unwrap();
        let mut world = World::new(GpuContext::Default, &plan.world_def().unwrap()).unwrap();
        let mut lab = Lab::place(&mut world, &plan).unwrap();
        world.set_stress_view(view);
        queue_script(Scenario::Balcony, &mut lab, &world).unwrap();
        let d = lab.destructibles()[0];
        let (mut idle, mut loaded) = (0.0f32, 0.0f32);
        for step in 0..360 {
            lab.step(&mut world).unwrap();
            world.poll_blocking().unwrap();
            let peak = world.destructible_chunk_stress(d).unwrap().map_or(0.0, |v| v.iter().flatten().fold(0.0f32, |a, &b| a.max(b)));
            if step == 40 {
                idle = peak;
            }
            loaded = loaded.max(peak);
        }
        let st = lab.stats(&world).unwrap();
        (st.bonds - st.bonds_intact, idle, loaded)
    };
    let off = run(false);
    let on = run(true);
    println!("view off {off:?}; on {on:?}");
    assert_eq!(off.0, on.0, "the same breaks");
    assert!(on.1 > 0.0 && on.1 < 1.0, "its own weight loads it below the fatal limit ({})", on.1);
    assert!(on.2 > on.1, "the weights load it further ({} after {})", on.2, on.1);
}

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

/// What tells the approaches apart, scenario by scenario: the claims the compare page makes.
fn check(scenario: Scenario, approach: Approach, v: &Verdict) {
    use Approach::*;
    let at = format!("{} / {}", scenario.name(), approach.name());
    let through: Vec<bool> = v.shots.iter().map(|s| s.through).collect();
    // Whether the approach solves sustained load through the bonds' own strengths.
    let solves_load = matches!(approach, StressSolver | Rubble);
    match scenario {
        Scenario::Balcony => {
            if solves_load {
                // Beam theory: the root is at 0.91 of its strength under seven weights and
                // 1.06 under eight. No break under the first five; down by the end.
                let first = v.shots.iter().position(|s| s.broke > 0).unwrap_or_else(|| panic!("{at}: the balcony never cracked"));
                assert!((5..=7).contains(&first), "{at}: first crack at weight {}", first + 1);
                assert!(v.fell, "{at}: the balcony came down");
            } else {
                assert_eq!(v.bonds_broken, 0, "{at}: nothing weighs on a bond");
                assert!(!v.fell, "{at}");
            }
        }
        // The canopy comes down where its weight is solved through the bonds (bending
        // included: the radial column's stress solver too); the threshold rule leaves it
        // hanging off the other pillar. The support graph is not asserted: its weight flow
        // carries no bending, and the canopy stays up there as well.
        Scenario::Pillar => match approach {
            ImpactThreshold => assert!(!v.fell, "{at}"),
            SupportFlow => {}
            StressSolver | StressRadial | Rubble => assert!(v.fell, "{at}: the canopy comes down"),
        },
        Scenario::SpeedLadder => match approach {
            // The engine has stopped the shot before anything breaks.
            ImpactThreshold | SupportFlow | StressSolver => assert_eq!(through, [false; 4], "{at}"),
            // The same hole at 20 and at 100 m/s, and every shot through it.
            StressRadial => {
                assert_eq!(through, [true; 4], "{at}");
                assert_eq!(v.shots[0].broke, v.shots[1].broke, "{at}: speed does not enter");
            }
            // Below the ballistic limit it stops and the wall takes its momentum; above, it
            // goes through, faster the faster it came.
            Rubble => {
                assert_eq!(through, [false, false, true, true], "{at}");
                assert!(v.shots[0].broke > 0 && v.shots[1].broke > v.shots[0].broke, "{at}: a harder stop breaks more");
                assert!(v.shots[3].along > v.shots[2].along && v.shots[2].along > 5.0, "{at}: exit speeds {} and {}", v.shots[2].along, v.shots[3].along);
            }
        },
        Scenario::EqualMomentum => match approach {
            ImpactThreshold | SupportFlow | StressSolver => assert_eq!(through, [false; 2], "{at}"),
            StressRadial => assert_eq!(through, [true; 2], "{at}"),
            // The fast light shot holes its wall; the slow heavy one stops and wrecks its.
            Rubble => {
                assert_eq!(through, [true, false], "{at}");
                assert!(v.shots[1].broke > v.shots[0].broke, "{at}: the slow heavy shot breaks more of its wall");
            }
        },
    }
}

/// Every scenario under every approach, printed and checked (`COMPARE_SCENARIO`,
/// `COMPARE_APPROACH`: one of them).
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
            let v = run(s, a);
            check(s, a, &v);
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

/// Each approach alone on the scale scene (10,240 pieces, a shot at every wall), one world on
/// the GPU at a time: wall time a step with the step's frame read back, against the 8.33 ms
/// a 120 Hz step has. A measurement, not a check; run it with nothing else on the GPU:
///
///   cargo test --release -p engine-world --features stress --test compare scale_solo -- --ignored --nocapture
///
/// `COMPARE_APPROACH`: one of them; `COMPARE_WALLS`: the wall count (default 40).
#[test]
#[ignore = "a solo performance measurement"]
fn scale_solo() {
    use rubble::lab::{Lab, LabAction, LabPlan};
    let walls: usize = std::env::var("COMPARE_WALLS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let one = std::env::var("COMPARE_APPROACH").ok();
    println!("{:<17} {:>7} {:>7} {:>7} {:>8} {:>9} {:>9} {:>9} {:>10}", "approach", "pieces", "bonds", "broken", "bodies", "mean ms", "p95 ms", "worst ms", "real time");
    for a in Approach::ALL {
        if one.as_deref().is_some_and(|o| o != a.name()) {
            continue;
        }
        let plan = LabPlan::new(&scale_def(a, walls)).unwrap();
        let mut world = World::new(GpuContext::Default, &plan.world_def().unwrap()).unwrap();
        let mut lab = Lab::place(&mut world, &plan).unwrap();
        let shots: Vec<f64> = lab.pack_shots().iter().map(|s| s.launch_at).collect();
        for (i, at) in shots.iter().enumerate() {
            lab.queue(&world, &LabAction::PackShot(i as u32), (at * 120.0).round() as u64).unwrap();
        }
        // Through the last launch and two seconds on.
        let steps = ((shots.last().unwrap() + 2.0) * 120.0) as usize;
        let mut ms = Vec::with_capacity(steps);
        let mut bodies = 0;
        for _ in 0..steps {
            let t = std::time::Instant::now();
            lab.step(&mut world).unwrap();
            world.poll_blocking().unwrap();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            bodies = bodies.max(lab.stats(&world).unwrap().bodies);
        }
        let st = lab.stats(&world).unwrap();
        // The first steps build pipelines and upload the scene: not the steady cost.
        ms.drain(..10);
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        ms.sort_by(f64::total_cmp);
        println!("{:<17} {:>7} {:>7} {:>7} {:>8} {:>9.2} {:>9.2} {:>9.2} {:>9.2}x", a.name(), st.chunks, st.bonds, st.bonds - st.bonds_intact, bodies, mean, ms[ms.len() * 95 / 100], ms[ms.len() - 1], 1e3 / 120.0 / mean);
    }
}

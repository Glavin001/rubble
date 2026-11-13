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
        world.poll_blocking().unwrap();
        v = Some(c.verdict(&world).unwrap());
    }
    let v = v.unwrap();
    println!("{:>15} / {:<16}: {} of {} bonds broken, first at {:?} s, drop {:.2} m{}", scenario.name(), approach.name(), v.bonds_broken, v.bonds, v.first_break.map(|t| (t * 100.0).round() / 100.0), v.drop, if v.fell { " (fell)" } else { "" });
    let r = &v.report;
    println!("{:>34}  bounce {} crater {} spall {} perforate {}; peak utilization {:.2}; {:?}", "", r.bounces, r.craters, r.spalls, r.perforations, r.peak_utilization, r.strikes);
    for s in &v.shots {
        println!("{:>34}: in {:.0} m/s, now {:.1} m/s ({:+.1} along), broke {}{}", s.label, s.launch_speed, s.speed, s.along, s.broke, if s.through { ", through" } else { "" });
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

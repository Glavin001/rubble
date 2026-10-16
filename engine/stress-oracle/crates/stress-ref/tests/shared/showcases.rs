//! Showcase behaviours (the spec's comparison table, "Ours" column). Each test runs a
//! showcase scene and a control that differs in one physical respect, and asserts the
//! qualitative outcome that threshold or quasi-static destruction gets wrong.
//!
//! Shared by the standalone world (`tests/showcases.rs`) and the world coupled to PhysX
//! (`stress-physx/tests/showcases_physx.rs`): the including file defines
//! `make_world(&Scene) -> World`.

use stress_ref::builders::blast_two_walls;
use stress_ref::observation::Observation;
use stress_ref::scene::Scene;
use stress_ref::showcases::*;
use stress_ref::solver::SolverEvent;

use super::make_world;

struct Run {
    obs: Observation,
    events: Vec<SolverEvent>,
}

fn run(scene: &Scene) -> Run {
    let mut w = make_world(scene);
    let obs = w.run();
    let events = w.solver.events.clone();
    let r = Run { obs, events };
    println!(
        "{}: broken {} fragments {} first crack {:?} first creak {:?} first break {:?} bond dissipation {:.1} J",
        scene.name,
        r.value("broken_bonds"),
        r.value("fragments"),
        r.first(|e| matches!(e, SolverEvent::Cracked { .. })),
        r.first(|e| matches!(e, SolverEvent::Creaked { .. })),
        r.first(|e| matches!(e, SolverEvent::Broken { .. })),
        r.value("bond_dissipation"),
    );
    r
}

impl Run {
    fn value(&self, key: &str) -> f64 {
        self.obs.values.get(key).copied().unwrap_or(0.0)
    }

    fn first(&self, pred: impl Fn(&SolverEvent) -> bool) -> Option<f64> {
        self.events.iter().filter(|e| pred(e)).map(event_time).reduce(f64::min)
    }

    fn broken(&self, body: &str) -> bool {
        self.obs.bodies[body].iter().any(|c| c.detached)
    }

    fn probe_final(&self, name: &str) -> f64 {
        *self.obs.probes[name].v.last().unwrap()
    }
}

fn event_time(e: &SolverEvent) -> f64 {
    match e {
        SolverEvent::Cracked { time, .. }
        | SolverEvent::Creaked { time, .. }
        | SolverEvent::Broken { time, .. }
        | SolverEvent::Split { time, .. }
        | SolverEvent::Refined { time, .. } => *time,
    }
}

/// Overhang on a thin connection: the neck holds at first, creaks under sustained load
/// (static fatigue),
/// cracks and snaps; the slab swings down. A neck at 40% utilization holds.
#[test]
fn overhang_on_thin_neck_creaks_cracks_and_swings_down() {
    let thin = run(&overhang(0.97, "s_overhang_thin"));
    let creak = thin.first(|e| matches!(e, SolverEvent::Creaked { .. })).expect("thin neck never creaked");
    let crack = thin.first(|e| matches!(e, SolverEvent::Cracked { .. })).expect("thin neck never cracked");
    let snap = thin.first(|e| matches!(e, SolverEvent::Broken { .. })).expect("thin neck never snapped");
    assert!(creak > 0.05 && creak <= crack && crack <= snap, "order creak {creak} crack {crack} break {snap}");
    assert!(snap > 0.2, "neck snapped at {snap} s: no sustained-load delay");
    assert!(thin.broken("overhang"), "slab still attached");
    assert!(thin.probe_final("slab_tip") < -0.5, "slab tip dropped only {} m", thin.probe_final("slab_tip"));

    let thick = run(&overhang(0.4, "s_overhang_thick"));
    assert_eq!(thick.value("broken_bonds"), 0.0);
    assert!(thick.probe_final("slab_tip").abs() < 0.01);
}

/// Supports removed one by one: the first gradual removal is carried by redistribution;
/// after the second (line load set for 0.97 utilization) the overhang root creaks, cracks
/// and gives way over seconds, not at the moment the post goes.
#[test]
fn supports_removed_one_by_one_fail_progressively() {
    let r = run(&supports_one_by_one(410.0, "s_supports_one_by_one"));
    let creak = r.first(|e| matches!(e, SolverEvent::Creaked { .. })).expect("no creak");
    let crack = r.first(|e| matches!(e, SolverEvent::Cracked { .. })).expect("no crack");
    let snap = r.first(|e| matches!(e, SolverEvent::Broken { .. })).expect("beam never broke");
    assert!(creak > 0.8, "creak at {creak} s, before the second post went");
    assert!(creak <= crack && crack <= snap, "order creak {creak} crack {crack} break {snap}");
    assert!(snap > 1.1 + 0.3, "beam broke at {snap} s: no delay after the second removal (ends 1.1 s)");
    assert!(r.broken("beam"));

    // Same structure with one post removed: redistribution carries it.
    let mut one = supports_one_by_one(410.0, "s_supports_one_removed");
    one.events.truncate(1);
    let r1 = run(&one);
    assert_eq!(r1.value("broken_bonds"), 0.0);
}

/// Steel vs brick, same car: the ductile wall absorbs more of the car's energy in its
/// joints before letting it through.
#[test]
fn ductile_wall_absorbs_more_than_brittle_wall() {
    let brick = run(&car_into_wall(false, "s_car_brick"));
    let ductile = run(&car_into_wall(true, "s_car_ductile"));
    let lost = |r: &Run| 8.0 - r.probe_final("car_velocity");
    println!("car speed lost: brick {:.2} m/s, ductile {:.2} m/s", lost(&brick), lost(&ductile));
    assert!(brick.broken("wall"), "brick wall did not breach");
    assert!(ductile.value("bond_dissipation") > 3.0 * brick.value("bond_dissipation"));
    assert!(lost(&ductile) > lost(&brick) + 0.5);
}

/// Floor onto floor: a slab dropped 1 m breaks the slab below (impact, not weight);
/// the same weight applied statically does not.
#[test]
fn dropped_floor_breaks_floor_below_static_weight_does_not() {
    let drop = run(&floor_onto_floor(1.0, "s_floor_drop"));
    let stat = run(&floor_onto_floor(0.0, "s_floor_static"));
    assert!(drop.value("broken_bonds") > 0.0 && drop.broken("lower"), "lower slab survived the impact");
    assert_eq!(stat.value("broken_bonds"), 0.0, "static weight broke the slab");
}

/// Masonry arch: stands in compression under its own weight; with the keystone removed
/// it collapses.
#[test]
fn arch_stands_until_keystone_is_removed() {
    let standing = run(&arch(None, "s_arch"));
    assert!(!standing.broken("arch"), "arch fell without intervention");
    assert!(standing.obs.bodies["arch"].iter().all(|c| c.displacement[2].abs() < 0.01));

    let removed = run(&arch(Some(0.1), "s_arch_keystone_removed"));
    assert!(removed.broken("arch"), "arch stood without its keystone");
    let crown_drop = removed.obs.bodies["arch"].iter().map(|c| c.displacement[2]).fold(f64::INFINITY, f64::min);
    assert!(crown_drop < -0.1, "voussoirs dropped only {crown_drop} m");
}

/// Explosion beside a building: the facing wall breaches, the shadowed wall behind it
/// survives; the same back wall alone breaches.
#[test]
fn blast_breaches_front_wall_and_shadowed_wall_survives() {
    let scene = blast_two_walls(30.0, "s_blast_two_walls");
    let both = run(&scene);
    assert!(both.broken("front"), "front wall did not breach");
    assert!(!both.broken("back"), "shadowed back wall breached");

    let mut alone = scene.clone();
    alone.name = "s_blast_back_alone".into();
    alone.bodies.retain(|b| b.name == "back");
    alone.metrics.retain(|m| m.name == "back_breach");
    let r = run(&alone);
    assert!(r.broken("back"), "back wall alone did not breach: shadowing test is vacuous");
}

/// The brick house stands under its roof load with stress paths around the openings;
/// a car through the side wall breaches it; a corner settlement brings the corner down.
#[test]
fn masonry_house_stands_and_fails_under_impact() {
    let standing = run(&masonry_house(HouseEvent::None, "s_house"));
    assert_eq!(standing.value("broken_bonds"), 0.0);
    assert!(standing.value("max_failure_index") > 0.1 && standing.value("max_failure_index") < 0.5);
    let car = run(&masonry_house(HouseEvent::Impact { speed: 10.0 }, "s_house_car"));
    assert!(car.broken("house"), "the car did not breach the wall");
}

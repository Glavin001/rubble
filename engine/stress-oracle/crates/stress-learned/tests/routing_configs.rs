//! A measurement (ignored by default): on P1's evaluation recipes, stepped by the recorder's
//! own world (oracle config a12_realtime, standalone), the routing P1 records
//! (`predict_island_members`, not tiered: any rigid-assumption failure seeds) against the
//! tiered routing a coupled quasi-static world uses (only an impactor's contact seeds), and
//! debris-reach routing on and off. Per frame: the routed cluster and impactor sets.
//!
//!   cargo test --release -p stress-learned --test routing_configs -- --ignored --nocapture

use std::collections::HashSet;

use stress_learned::oracle_config::OracleConfig;
use stress_learned::procedural::{wall_recipes, with_splits_and_ensembles, EnsemblePlan, Ranges};
use stress_ref::math::Vec3;
use stress_ref::scene::SolveMode;
use stress_ref::world::World;

fn members(w: &mut World, tiered: bool, debris: bool) -> (Vec<u64>, Vec<usize>) {
    let (q, mode, d) = (w.qs_islands, w.scene.sim.solve_mode, w.debris_reach_routing);
    if tiered {
        w.qs_islands = true;
        w.scene.sim.solve_mode = SolveMode::QuasiStatic;
    }
    w.debris_reach_routing = debris;
    let dt = w.scene.sim.frame_dt;
    let m = w.predict_island_members(dt, &HashSet::new());
    (w.qs_islands, w.scene.sim.solve_mode, w.debris_reach_routing) = (q, mode, d);
    m
}

#[test]
#[ignore]
fn recorded_routing_against_tiered() {
    let oracle = OracleConfig::named("a12_realtime").unwrap();
    for family in ["free_wall", "anchored_wall"] {
        let recipes = with_splits_and_ensembles(wall_recipes(family, 1, 24, 101, &Ranges::pilot()).expect("recipes"), &EnsemblePlan::default());
        let fast = |r: &&stress_learned::outcome::OutcomeRecipe| r.scene.impactors.iter().any(|i| Vec3::from_array(i.velocity).norm() > 15.0);
        for r in recipes.iter().filter(|r| r.parameters["split"] != "train").filter(fast).take(2) {
            let mut scene = r.scene.clone();
            oracle.apply(&mut scene);
            let mut w = World::new(&scene);
            let (mut routed_frames, mut differ) = (0, 0);
            for f in 0..24 {
                let base = members(&mut w, false, false);
                let variants = [("tiered", members(&mut w, true, false)), ("debris", members(&mut w, false, true)), ("tiered+debris", members(&mut w, true, true))];
                if !base.0.is_empty() || variants.iter().any(|(_, m)| !m.0.is_empty()) {
                    routed_frames += 1;
                    let mut line = format!("{} f{f}: clusters {} recorded {:?}+imp{:?}", r.event_id, w.solver.clusters.len(), base.0.len(), base.1);
                    for (name, m) in &variants {
                        let (a, b): (HashSet<u64>, HashSet<u64>) = (base.0.iter().copied().collect(), m.0.iter().copied().collect());
                        if a != b || base.1 != m.1 {
                            differ += 1;
                            line += &format!(" | {name}: {} (+{} -{}) imp{:?}", m.0.len(), b.difference(&a).count(), a.difference(&b).count(), m.1);
                        }
                    }
                    eprintln!("{line}");
                }
                w.step_frame();
            }
            eprintln!("{}: {routed_frames} routed frames, {differ} variant differences", r.event_id);
        }
    }
}

//! The coupled reference and its ceiling are one computation (the planning session's
//! definition of 2026-10-22): a coupled frame whose island step comes from the oracle through
//! the frame-provider interface (`World::step_frame_with_frame_provider`, the oracle's
//! `coupled_substeps` on the prepared clone) is `World::step_frame`.
//!
//! - Whole runs, untiered routing (where rubble is deterministic run to run): every frame's
//!   state bit for bit.
//! - Per frame, both routings: from one prepared world (the routing and rubble's step done),
//!   the provider on a clone and the reference's own substeps give the same frame, bit for
//!   bit. This holds whatever rubble does: under tiered routing rubble integrates the
//!   structure and is not bit-reproducible run to run (GPU), so two whole runs can't be
//!   compared bitwise there, while the island step through the interface still equals the
//!   reference's.

use std::collections::HashSet;

use stress_ref::math::Vec3;
use stress_ref::scene::Scene;
use stress_ref::world::World;
use stress_rubble::scenes::realtime;
use stress_rubble::RubbleEngine;

#[path = "rewind_scenes/mod.rs"]
mod rewind_scenes;

fn coupled(scene: &Scene, tiered: bool) -> World {
    let mut e = RubbleEngine::new(Vec3::from_array(scene.gravity)).expect("rubble engine");
    let (i, n) = stress_ref::capability::Profile::BlastParityRouted.engine_policy();
    e.set_policy(i, n);
    let mut w = World::with_engine(scene, Box::new(e));
    w.routing_tiered = tiered;
    w.debris_reach_routing = true;
    w.handed_ground_overlaps = true;
    w
}

fn oracle(mut w: World, f: &stress_ref::world::CoupledFrame) -> Result<World, String> {
    w.coupled_substeps(f);
    Ok(w)
}

fn scenes() -> Vec<(Scene, usize)> {
    let row = realtime(rewind_scenes::wall_row(2, 40.0, 0.1, 0.15), 0.15);
    let mut b5 = stress_ref::builders::wall_impact(20.0, "b5_wall_impact_v20");
    b5.sim.duration = 0.25;
    vec![(row, 9), (realtime(b5, 0.25), 15)]
}

#[test]
fn whole_runs_untiered_are_the_coupled_reference_bit_for_bit() {
    for (scene, frames) in scenes() {
        let mut reference = coupled(&scene, false);
        let mut ceiling = coupled(&scene, false);
        for frame in 0..frames {
            reference.step_frame();
            ceiling.step_frame_with_frame_provider(&mut oracle).unwrap();
            assert_eq!(stress_learned::outcome::fingerprint(&reference), stress_learned::outcome::fingerprint(&ceiling), "{}: frame {frame} differs", scene.name);
        }
        assert_eq!((reference.island_frames, reference.redone_frames), (ceiling.island_frames, ceiling.redone_frames));
        eprintln!("{}: {frames} frames identical (island frames {})", scene.name, reference.island_frames);
    }
}

#[test]
fn the_island_step_through_the_interface_is_the_references_both_routings() {
    for tiered in [false, true] {
        for (scene, frames) in scenes() {
            let mut w = coupled(&scene, tiered);
            let mut routed = 0;
            for frame in 0..frames {
                w.engine_save();
                let f = w.coupled_prepare(&HashSet::new());
                routed += usize::from(!f.frozen.is_empty());
                let via = oracle(w.clone(), &f).unwrap();
                w.coupled_substeps(&f);
                assert_eq!(stress_learned::outcome::fingerprint(&w), stress_learned::outcome::fingerprint(&via), "{} tiered {tiered}: frame {frame}", scene.name);
                w.engine_discard();
                w.coupled_write_back(&f.frozen);
            }
            eprintln!("{} tiered {tiered}: {frames} frames identical ({routed} routed)", scene.name);
        }
    }
}

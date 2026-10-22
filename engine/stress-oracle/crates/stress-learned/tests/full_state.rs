//! The completeness gate of the full-state contract: a World rebuilt from a frame
//! boundary's recorded columns (plus its scene), stepped one frame, equals the oracle's own
//! next boundary bit for bit, on frames through the impact: the M5 free wall and wall row,
//! an anchored wall under gravity, and multi-impactor walls.

use stress_learned::full_state::{decode, differing_paths, encode, oracle_frame, restore, shred_world, unshred};
use stress_learned::oracle_config::OracleConfig;
use stress_learned::procedural::{m5_free_wall, m5_wall_row, wall_recipes, Ranges};
use stress_ref::scene::Scene;
use stress_ref::world::World;

/// Steps `frames` oracle frames; at every boundary checks the shredded state round-trips
/// (columns -> tree -> World -> columns, and through the file encoding), and that the
/// restored world's next frame equals the oracle's. Returns the frames with a contact or a
/// break (the gate must cover the impact).
fn gate(name: &str, mut scene: Scene, frames: usize) -> usize {
    OracleConfig::named("a12_realtime").unwrap().apply(&mut scene);
    scene.sim.duration = frames as f64 * scene.sim.frame_dt;
    let mut w = World::new(&scene);
    let mut active = 0;
    for f in 0..frames {
        let s0 = shred_world(&w).unwrap();
        assert!(differing_paths(&decode(&encode(&s0).unwrap()).unwrap(), &s0, |_| false).is_empty(), "{name} frame {f}: file encoding round-trips");
        let mut r = restore(&scene, &s0).unwrap();
        assert!(differing_paths(&s0, &shred_world(&r).unwrap(), |_| false).is_empty(), "{name} frame {f}: restore round-trips");
        let _ = unshred(&s0).unwrap();
        let events = w.solver.events.len();
        let mut touched = false;
        oracle_frame(&mut w, |x| touched |= x.chunk_contact_hits().iter().flatten().any(|&h| h > 0.0)).unwrap();
        oracle_frame(&mut r, |_| {}).unwrap();
        let diff = differing_paths(&shred_world(&w).unwrap(), &shred_world(&r).unwrap(), |_| false);
        assert!(diff.is_empty(), "{name} frame {f}: the restored world's next frame differs in {} paths: {:?}", diff.len(), &diff[..diff.len().min(20)]);
        active += usize::from(touched || w.solver.events.len() > events);
    }
    active
}

#[test]
fn m5_free_wall_resumes_bit_for_bit_through_the_impact() {
    let n = gate("m5_free_wall", m5_free_wall(), 20);
    assert!(n >= 1, "{n} active frames: the strike happens within one frame");
}

#[test]
fn m5_wall_row_resumes_bit_for_bit_through_the_impact() {
    assert!(gate("m5_wall_row", m5_wall_row(), 20) >= 2);
}

#[test]
fn an_anchored_wall_under_gravity_with_several_impactors_resumes_bit_for_bit() {
    let r = wall_recipes("anchored_wall", 2, 1, 7001, &Ranges::pilot()).unwrap().remove(0);
    assert!(r.scene.gravity != [0.0; 3] && r.scene.impactors.len() >= 2);
    assert!(gate(&r.event_id, r.scene, 30) >= 2);
}

#[test]
fn a_free_wall_with_several_impactors_resumes_bit_for_bit() {
    let r = wall_recipes("free_wall", 2, 1, 7002, &Ranges::pilot()).unwrap().remove(0);
    assert!(r.scene.impactors.len() >= 2);
    assert!(gate(&r.event_id, r.scene, 30) >= 2);
}

#[test]
fn an_anchored_wall_resumes_bit_for_bit_while_it_settles() {
    let r = wall_recipes("anchored_wall", 1, 1, 7003, &Ranges::pilot()).unwrap().remove(0);
    gate(&r.event_id, r.scene, 40);
}

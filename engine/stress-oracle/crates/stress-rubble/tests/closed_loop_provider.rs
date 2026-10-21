//! The closed-loop harness's window-end handoff (`World::step_frame_with_provider`,
//! `apply_decoded`): with the oracle as its provider, one routed frame of the wall row
//! leaves exactly the oracle's breaks and the impactor's frame-end motion, and the decoded
//! state survives a JSON round trip bit for bit (`learned-impact/PROVIDER_PROTOCOL.md`).

use stress_ref::math::Vec3;
use stress_ref::world::{DecodedState, World};
use stress_rubble::scenes::realtime;
use stress_rubble::RubbleEngine;

#[path = "rewind_scenes/mod.rs"]
mod rewind_scenes;

#[test]
fn oracle_provider_applies_the_oracles_frame() {
    let scene = realtime(rewind_scenes::wall_row(2, 40.0, 0.1, 0.05), 0.05);
    let mut oracle = World::new(&scene);
    oracle.step_frame();
    let mut w = World::with_engine(&scene, Box::new(RubbleEngine::new(Vec3::from_array(scene.gravity)).expect("rubble engine")));
    let mut provider = |view: &World, keys: &stress_ref::world::IslandKeys| -> Result<DecodedState, String> {
        let mut model = view.standalone_copy();
        model.step_frame();
        let d = view.decoded_from_oracle(&model, keys, 0, 0);
        // The file path's round trip changes nothing.
        let back: DecodedState = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
        assert_eq!(serde_json::to_string(&back).unwrap(), serde_json::to_string(&d).unwrap());
        Ok(back)
    };
    let keys = w.step_frame_with_provider(&mut provider).expect("provider frame");
    assert!(!keys.chunks.is_empty() && keys.impactors == vec![0], "the strike frame routes the walls and the ram");
    let broken = |sv: &stress_ref::solver::ReferenceSolver| -> Vec<(usize, usize)> {
        (0..sv.bonds.len()).flat_map(|s| (0..sv.bonds[s].len()).filter(move |&b| !sv.bonds[s][b].connected()).map(move |b| (s, b))).collect()
    };
    assert_eq!(broken(&w.solver), broken(&oracle.solver), "the oracle's breaks");
    assert_eq!(w.impactors[0].velocity, oracle.impactors[0].velocity, "the ram's frame-end velocity");
    assert!((w.impactors[0].pose.position - oracle.impactors[0].pose.position).norm() < 1e-12, "the ram's frame-end position");
    assert!((w.solver.time - oracle.solver.time).abs() < 1e-12);
}

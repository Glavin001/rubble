//! Harness A relabelling (`stress_learned::relabel`): a round trip on the oracle's own state is
//! bit-identical, the label is the native oracle's next frame, files round-trip exactly, and
//! two runs give the same bytes.

use stress_learned::relabel::{read_state, relabel_event, write_state, Provided};
use stress_ref::world::World;

fn event_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("relabel-test-{}-{:?}", std::process::id(), std::thread::current().id()).replace(['(', ')'], ""));
    std::fs::create_dir_all(&dir).unwrap();
    let mut scene = stress_ref::builders::wall_impact(20.0, "b5_wall_impact_v20");
    scene.sim.duration = 0.2;
    std::fs::write(dir.join("scene.json"), serde_json::to_vec(&scene).unwrap()).unwrap();
    dir
}

#[test]
fn relabelling_the_oracles_own_state_is_bit_identical() {
    let dir = event_dir();
    let bytes = |s: &stress_ref::world::IslandState| stress_learned::relabel::state_bytes(s).unwrap();
    let own = relabel_event(&dir, &[(2, Provided::Own), (5, Provided::Own)], true).unwrap();
    assert!(own.iter().all(|r| !r.members.chunks.is_empty()), "the frames are routed");
    // the calls carry the oracle's frame as targets (the recorder's capture of the same step:
    // relabel_event checks it against the island step bit for bit)
    for r in &own {
        let c = r.call.as_ref().unwrap();
        assert_eq!(c.manifest["schema_version"], stress_learned::outcome_calls::CALLS_SCHEMA);
        assert_eq!(c.calls.len(), 1);
        assert_eq!(c.calls[0].frame, r.frame);
    }
    // the oracle's own start state handed back in: the same label, bit for bit
    let back = relabel_event(&dir, &[(5, Provided::Start(own[1].start.clone())), (2, Provided::Start(own[0].start.clone()))], false).unwrap();
    assert_eq!(bytes(&back[1].start), bytes(&own[0].start));
    assert_eq!(bytes(&back[1].end), bytes(&own[0].end));
    assert_eq!(bytes(&back[0].end), bytes(&own[1].end));
    // the label is the native oracle's next frame
    let scene = stress_learned::relabel::event_scene(&dir).unwrap();
    let mut w = World::new(&scene);
    for _ in 0..6 {
        w.solver.reset_peak_utilizations();
        w.step_frame();
    }
    assert_eq!(bytes(&w.extract_island(&own[1].members)), bytes(&own[1].end), "frame 5's label is the native frame 6's island");
    // files and determinism
    let p = dir.join("end.bin");
    write_state(&p, &own[1].end).unwrap();
    assert_eq!(bytes(&read_state(&p).unwrap()), bytes(&own[1].end));
    let again = relabel_event(&dir, &[(2, Provided::Own), (5, Provided::Own)], false).unwrap();
    assert_eq!(bytes(&again[1].end), bytes(&own[1].end));
    std::fs::remove_dir_all(&dir).ok();
}

/// A decoded outcome (`learned-impact-decoded-state/1`) as harness A takes it: the call is the
/// next frame's, from the world the decoded island reaches. The oracle's own decoded outcome
/// is the /3 boundary's ceiling: the next frame's island is close to the native one, not
/// equal (the decoded state drops the hidden deformation and the joint history it does not
/// carry).
#[test]
fn a_decoded_outcome_relabels_the_next_frame() {
    let dir = event_dir();
    let scene = stress_learned::relabel::event_scene(&dir).unwrap();
    let mut w = World::new(&scene);
    for _ in 0..2 {
        w.solver.reset_peak_utilizations();
        w.step_frame();
    }
    let mut view = w.clone();
    let m = view.island_members();
    let island: std::collections::HashSet<u64> = view.solver.clusters.iter().filter(|c| c.chunks.first().is_some_and(|&k| m.chunks.contains(&(c.structure, k)))).map(|c| c.id).collect();
    for cl in &mut view.solver.clusters {
        cl.driven = !island.contains(&cl.id);
    }
    for (i, imp) in view.impactors.iter_mut().enumerate() {
        imp.driven = !m.impactors.contains(&i);
    }
    let keys = view.routed_island_keys();
    let mut model = view.standalone_copy();
    model.step_frame();
    let d = view.decoded_from_oracle(&model, &keys, 2, 0);
    let r = relabel_event(&dir, &[(2, Provided::Decoded(d.clone()))], true).unwrap();
    assert_eq!(r[0].frame, 3);
    let c = r[0].call.as_ref().unwrap();
    assert_eq!(c.calls[0].frame, 3);
    let again = relabel_event(&dir, &[(2, Provided::Decoded(d))], true).unwrap();
    let bytes = |s: &stress_ref::world::IslandState| stress_learned::relabel::state_bytes(s).unwrap();
    assert_eq!(bytes(&again[0].end), bytes(&r[0].end), "deterministic");
    std::fs::remove_dir_all(&dir).ok();
}

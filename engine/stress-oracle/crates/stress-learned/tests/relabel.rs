//! Harness A relabelling (`stress_learned::relabel`): a round trip on the oracle's own state is
//! bit-identical, the label is the native oracle's next frame, files round-trip exactly, and
//! two runs give the same bytes.

use stress_learned::relabel::{read_state, relabel_event, write_state};
use stress_ref::world::World;

fn event_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("relabel-test-{}", std::process::id()));
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
    let own = relabel_event(&dir, &[(2, None), (5, None)], true).unwrap();
    assert!(own.iter().all(|r| !r.members.chunks.is_empty()), "the frames are routed");
    assert!(own.iter().all(|r| r.call.is_some()));
    // the oracle's own start state handed back in: the same label, bit for bit
    let back = relabel_event(&dir, &[(5, Some(own[1].start.clone())), (2, Some(own[0].start.clone()))], false).unwrap();
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
    let again = relabel_event(&dir, &[(2, None), (5, None)], false).unwrap();
    assert_eq!(bytes(&again[1].end), bytes(&own[1].end));
    std::fs::remove_dir_all(&dir).ok();
}

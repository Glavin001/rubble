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

/// The relabelled call of the oracle's own state is the recorder's call of that frame, inputs
/// and targets, bit for bit: the exceptions are the event-level `y_impactor_regime` (the
/// event's final regime), unknown (-1) to a one-frame relabel, and the punch baseline.
#[test]
fn an_own_state_relabel_is_the_recorded_call() {
    use serde_json::json;
    use stress_learned::oracle_config::OracleConfig;
    use stress_learned::outcome::{record_outcome_event, OutcomeRecipe, RecordOptions};
    use stress_learned::outcome_calls::{build_event, Arr};
    use stress_learned::procedural::{wall_body, Layout};
    use stress_ref::builders;
    use stress_ref::scene::{ImpactorDesc, ImpactorShape};

    let mut s = builders::new_scene("relabel", "relabel test wall");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), builders::oracle_concrete(None));
    s.materials.insert("steel".into(), builders::elastic_steel());
    s.bodies.push(wall_body(&Layout { counts: [4, 2, 4], chunk: 0.1, opening: None, supports: "free" }, "wall", 0.0));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.06, 0.06, 0.06] },
        mass: 80.0,
        position: [0.03, -0.065 - 25.0 * 0.004, 0.21],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 25.0, 0.0],
        angular_velocity: [0.0, 0.0, 3.0],
        material: "steel".into(),
        crush: None,
    });
    let out = std::env::temp_dir().join(format!("relabel-recorded-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let recipe = OutcomeRecipe { event_id: "ev".into(), family: "test".into(), tier: 1, split_group: "ev".into(), layout_group: "L".into(), parameters: json!({}), scene: s };
    let mut opts = RecordOptions::new(OracleConfig::named("a12_realtime").unwrap());
    opts.stop.max_seconds = 0.5;
    record_outcome_event(&recipe, &out, &opts, &|_| {}).unwrap();
    let dir = out.join("ev");
    let recorded = build_event(&dir).unwrap();
    assert!(recorded.calls.len() >= 3, "the strike routes several frames");
    let jobs: Vec<(usize, Provided)> = recorded.calls.iter().map(|c| (c.frame, Provided::Own)).collect();
    let relabelled = relabel_event(&dir, &jobs, true).unwrap();
    for (rec, r) in recorded.calls.iter().zip(&relabelled) {
        let call = &r.call.as_ref().expect("a call").calls[0];
        // (The punch baseline, `base_punch_*`, is the recorder's comparison branch: not a
        // model input or target, and not built for a live call.)
        let names = |c: &stress_learned::outcome_calls::Call| c.arrays.iter().map(|(k, _)| k.clone()).filter(|k| !k.starts_with("base_punch_")).collect::<Vec<_>>();
        assert_eq!(names(call), names(rec), "frame {}: the same arrays", rec.frame);
        for (k, a) in rec.arrays.iter().filter(|(k, _)| !k.starts_with("base_punch_")) {
            let b = call.array(k).unwrap();
            if k == "y_impactor_regime" {
                let Arr::I32(_, v) = b else { panic!() };
                assert!(v.iter().all(|&x| x == -1), "frame {}: the regime is unknown to a relabel", rec.frame);
                continue;
            }
            assert!(a == b, "frame {}: {k} differs", rec.frame);
        }
    }
    let _ = std::fs::remove_dir_all(&out);
}

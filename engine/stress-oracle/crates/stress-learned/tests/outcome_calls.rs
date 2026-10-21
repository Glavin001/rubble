//! The call builder (frames/1 → P2's outcome/2 calls): a live world's call equals the
//! recorded call of the same state, features do not see a rotation of the whole scene,
//! and edges come as reverse twins.

use std::path::PathBuf;

use serde_json::json;
use stress_learned::oracle_config::OracleConfig;
use stress_learned::outcome::{record_outcome_event, OutcomeRecipe, RecordOptions};
use stress_learned::outcome_calls::{build_event, live_call, write_event, Arr, Call};
use stress_learned::procedural::{wall_body, Layout};
use stress_ref::builders;
use stress_ref::math::{Quat, Vec3};
use stress_ref::scene::{ImpactorDesc, ImpactorShape, Scene};
use stress_ref::world::World;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("stress-learned-calls-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A free 4 x 2 x 4 wall struck by a box ram hard enough to break it.
fn scene() -> Scene {
    let mut s = builders::new_scene("calls", "call builder test wall");
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
    s
}

fn record(name: &str) -> (PathBuf, OutcomeRecipe) {
    let recipe = OutcomeRecipe { event_id: name.into(), family: "test".into(), tier: 1, split_group: name.into(), layout_group: "L".into(), parameters: json!({}), scene: scene() };
    let out = tmp(name);
    let mut opts = RecordOptions::new(OracleConfig::named("a12_realtime").unwrap());
    opts.stop.max_seconds = 0.5;
    record_outcome_event(&recipe, &out, &opts, &|_| {}).unwrap();
    (out.join(name), recipe)
}

fn keys_of(c: &Call, name: &str) -> Vec<i64> {
    match c.array(name) {
        Some(Arr::I64(_, v)) => v.clone(),
        _ => panic!("{name}"),
    }
}

#[test]
fn a_live_call_equals_the_recorded_call_of_the_same_state() {
    let (ev, recipe) = record("live");
    let built = build_event(&ev).unwrap();
    assert!(built.calls.len() >= 3, "the strike routes several frames");
    let ctx: serde_json::Value = serde_json::from_slice(&std::fs::read(ev.join("context.json")).unwrap()).unwrap();
    // Replay the oracle frame by frame as the recorder stepped it.
    let mut s = recipe.scene.clone();
    OracleConfig::named("a12_realtime").unwrap().apply(&mut s);
    s.sim.duration = 1.0;
    let mut w = World::new(&s);
    let mut checked = 0;
    for f in 0..=built.calls.last().unwrap().frame {
        if let Some(rec) = built.calls.iter().find(|c| c.frame == f) {
            let live = live_call(&mut w, "live", &ctx, &keys_of(rec, "chunk_key"), &keys_of(rec, "impactor_key"), f, true).unwrap();
            let lc = &live.calls[0];
            for (name, a) in &lc.arrays {
                let b = rec.array(name).unwrap_or_else(|| panic!("frame {f}: recorded call lacks {name}"));
                assert_eq!(a, b, "frame {f}: {name} differs between the live and the recorded call");
            }
            assert_eq!(lc.scales, rec.scales, "frame {f}: scales");
            assert_eq!(lc.counts, rec.counts, "frame {f}: counts");
            checked += 1;
        }
        w.solver.reset_peak_utilizations();
        w.step_frame();
    }
    assert_eq!(checked, built.calls.len());
    // Writing round-trips: compressed and plain hold the same arrays.
    let out = tmp("write");
    write_event(&out.join("z"), &built, true).unwrap();
    write_event(&out.join("p"), &built, false).unwrap();
    let z = zstd::decode_all(std::fs::File::open(out.join("z/arrays.bin.zst")).unwrap()).unwrap();
    assert_eq!(z, std::fs::read(out.join("p/arrays.bin")).unwrap(), "arrays.bin.zst is one zstd frame of arrays.bin");
}

fn feats(c: &Call, name: &str) -> Vec<f32> {
    match c.array(name) {
        Some(Arr::F32(_, v)) => v.clone(),
        _ => panic!("{name}"),
    }
}

#[test]
fn features_do_not_see_a_rotation_of_the_whole_scene() {
    let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalized(), 0.7);
    let rot = |v: [f64; 3]| q.rotate(Vec3::from_array(v)).to_array();
    let base = scene();
    let mut turned = base.clone();
    for b in &mut turned.bodies {
        b.position = rot(b.position);
        b.orientation = (q * Quat::from_wxyz(b.orientation)).to_wxyz();
    }
    for i in &mut turned.impactors {
        i.position = rot(i.position);
        i.velocity = rot(i.velocity);
        i.angular_velocity = rot(i.angular_velocity);
        i.orientation = (q * Quat::from_wxyz(i.orientation)).to_wxyz();
    }
    let ctx = json!({"family": "test"});
    let call = |s: &Scene| {
        let mut s = s.clone();
        OracleConfig::named("a12_realtime").unwrap().apply(&mut s);
        let mut w = World::new(&s);
        let chunks: Vec<i64> = (0..w.solver.structures[0].chunks.len()).map(|c| stress_ref::keys::chunk_key(0, c)).collect();
        live_call(&mut w, "rot", &ctx, &chunks, &[0], 0, true).unwrap().calls.remove(0)
    };
    let (a, b) = (call(&base), call(&turned));
    assert_eq!(a.counts, b.counts);
    assert!(a.counts["impactor_contact"] > 0 && a.counts["bond"] > 0);
    for name in ["chunk_feat", "impactor_feat", "global_feat", "bond_feat", "impactor_contact_feat", "bond_qs_feat"] {
        let (x, y) = (feats(&a, name), feats(&b, name));
        for (i, (u, v)) in x.iter().zip(&y).enumerate() {
            assert!((u - v).abs() <= 2e-3 * (1.0 + u.abs()), "{name}[{i}]: {u} vs {v} under rotation");
        }
    }
    // Twins: row 2k + 1 is row 2k reversed.
    for t in ["bond", "impactor_contact"] {
        let (s, d) = match (a.array(&format!("{t}_src")), a.array(&format!("{t}_dst"))) {
            (Some(Arr::I32(_, s)), Some(Arr::I32(_, d))) => (s.clone(), d.clone()),
            _ => panic!("{t}"),
        };
        for k in 0..s.len() / 2 {
            assert_eq!((s[2 * k], d[2 * k]), (d[2 * k + 1], s[2 * k + 1]), "{t} twin {k}");
        }
    }
}

#[test]
fn a_thinned_training_event_regenerates_byte_identically_and_builds_the_same_calls() {
    let mut recipe = OutcomeRecipe { event_id: "thin".into(), family: "test".into(), tier: 1, split_group: "thin".into(), layout_group: "L".into(), parameters: json!({"split": "train"}), scene: scene() };
    recipe.scene.name = "thin".into();
    let out = tmp("thin");
    let opts = RecordOptions::new(OracleConfig::named("a12_realtime").unwrap());
    let sum = record_outcome_event(&recipe, &out, &opts, &|_| {}).unwrap();
    let ev = out.join("thin");
    let ctx: serde_json::Value = serde_json::from_slice(&std::fs::read(ev.join("context.json")).unwrap()).unwrap();
    let dropped = ctx["storage"]["dropped_frames"].as_u64().unwrap();
    assert!(dropped > 0, "a training event drops settling frames ({} frames)", sum.frames);
    assert!(std::fs::read_dir(ev.join("arrays")).unwrap().all(|e| e.unwrap().path().to_string_lossy().ends_with(".zst")), "arrays are compressed");
    // Regenerate every frame and check each stored row.
    let regen = stress_learned::outcome::regenerate(&ev, &tmp("thin-full")).unwrap();
    assert_eq!(regen.frames, sum.frames);
    assert!(regen.arrays_checked > 40 && regen.kept_frames < regen.frames);
    // The calls of the kept frames are the same whether built from the thinned or the full record.
    let (thin, full) = (build_event(&ev).unwrap(), build_event(&regen.event).unwrap());
    assert!(full.calls.len() >= thin.calls.len() && !thin.calls.is_empty());
    for c in &thin.calls {
        let f = full.calls.iter().find(|x| x.frame == c.frame).expect("a full call for each kept frame");
        for (name, a) in &c.arrays {
            assert_eq!(Some(a), f.array(name), "frame {}: {name}", c.frame);
        }
        assert_eq!(c.energy, f.energy, "frame {}", c.frame);
    }
}

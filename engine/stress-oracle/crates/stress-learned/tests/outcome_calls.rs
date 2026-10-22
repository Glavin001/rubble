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
    let mut opts = RecordOptions::new(OracleConfig::named("a12_realtime").unwrap());
    opts.storage.select_training_frames = true; // off by default since revision 8; old events are thinned
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
            // bitwise: the /5 columns hold NaN (absent values)
            let bits = |x: &Arr| match x {
                Arr::F64(s, d) => (s.clone(), d.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                Arr::F32(s, d) => (s.clone(), d.iter().map(|v| u64::from(v.to_bits())).collect()),
                Arr::I32(s, d) => (s.clone(), d.iter().map(|&v| v as u64).collect()),
                Arr::I64(s, d) => (s.clone(), d.iter().map(|&v| v as u64).collect()),
            };
            assert_eq!(Some(bits(a)), f.array(name).map(bits), "frame {}: {name}", c.frame);
        }
        assert_eq!(c.energy, f.energy, "frame {}", c.frame);
    }
}

#[test]
fn the_calls_hidden_state_is_the_solvers() {
    let mut s = scene();
    OracleConfig::named("a12_realtime").unwrap().apply(&mut s);
    s.sim.duration = 1.0;
    let mut w = World::new(&s);
    for _ in 0..2 {
        w.step_frame();
    }
    let chunks: Vec<i64> = (0..w.solver.structures[0].chunks.len()).map(|c| stress_ref::keys::chunk_key(0, c)).collect();
    let call = live_call(&mut w, "hidden", &json!({}), &chunks, &[0], 2, false).unwrap().calls.remove(0);
    let def = feats(&call, "chunk_def");
    let (mut worst_u, mut worst_v, mut scale_v) = (0.0f64, 0.0f64, 0.0f64);
    for (j, &key) in keys_of(&call, "chunk_key").iter().enumerate() {
        let (_, c) = stress_ref::keys::split_key(key);
        let st = &w.solver.chunks[0][c];
        let d: Vec<f64> = def[12 * j..12 * j + 12].iter().map(|&x| x as f64).collect();
        let u = [st.u.x, st.u.y, st.u.z, st.th.x, st.th.y, st.th.z];
        let vh = [st.v.x, st.v.y, st.v.z, st.w.x, st.w.y, st.w.z];
        for k in 0..6 {
            worst_u = worst_u.max((d[k] - u[k]).abs());
            worst_v = worst_v.max((d[6 + k] - vh[k]).abs());
            scale_v = scale_v.max(vh[k].abs());
        }
    }
    eprintln!("hidden u/theta max |diff| {worst_u:.3e}; hidden rates max |diff| {worst_v:.3e} of max {scale_v:.3e}");
    assert!(worst_u <= 1e-6, "u and theta are the solver's (f32)");
    // Each chunk's body and its rotation: the cluster's.
    let (body, bq) = match (call.array("chunk_body"), call.array("chunk_body_q")) {
        (Some(Arr::I32(_, b)), Some(Arr::F32(_, q))) => (b.clone(), q.clone()),
        _ => panic!("chunk_body / chunk_body_q"),
    };
    for (j, &key) in keys_of(&call, "chunk_key").iter().enumerate() {
        let (_, c) = stress_ref::keys::split_key(key);
        let cl = &w.solver.clusters[w.solver.chunks[0][c].cluster];
        let q = cl.pose.rotation.to_wxyz();
        assert!((0..4).all(|k| (bq[4 * j + k] as f64 - q[k]).abs() < 1e-6));
        for (i, &key2) in keys_of(&call, "chunk_key").iter().enumerate() {
            let (_, c2) = stress_ref::keys::split_key(key2);
            assert_eq!(body[i] == body[j], w.solver.chunks[0][c2].cluster == w.solver.chunks[0][c].cluster);
        }
    }
    assert!(worst_v <= 1e-3 * scale_v.max(1e-6) + 1e-5, "the fitted rates are the solver's hidden rates");
}

/// A full-state boundary stored as a delta (of the previous one) decodes through its chain
/// and is the oracle's next state from the boundary before it, bit for bit.
#[test]
fn a_delta_full_state_boundary_is_the_next_state_bit_for_bit() {
    let (ev, _) = record("fullstate");
    let ctx: serde_json::Value = serde_json::from_slice(&std::fs::read(ev.join("context.json")).unwrap()).unwrap();
    let list = ctx["full_state"]["boundaries"].as_array().unwrap();
    assert!(list.iter().any(|s| !s["delta_of"].is_null()));
    let scene: Scene = serde_json::from_slice(&std::fs::read(ev.join("scene.json")).unwrap()).unwrap();
    let b = 9; // a delta, the end of a nine-delta chain
    assert_eq!(list[b]["delta_of"].as_u64(), Some(8));
    let mut w = stress_learned::full_state::restore(&scene, &stress_learned::full_state::load_frame(&ev, b - 1).unwrap()).unwrap();
    stress_learned::full_state::oracle_frame(&mut w, |_| {}).unwrap();
    let diff = stress_learned::full_state::differing_paths(&stress_learned::full_state::shred_world(&w).unwrap(), &stress_learned::full_state::load_frame(&ev, b).unwrap(), |_| false);
    assert!(diff.is_empty(), "{diff:?}");
}

/// Calls `/5`: the island's full state at the frame's start and end, by name; its u is the
/// `/4` deformation's, a bond the oracle broke in the frame is dead at its end, and the
/// static columns hold one row per entity.
#[test]
fn full_state_calls_carry_the_islands_complete_state() {
    let (ev, _) = record("fs5");
    let e = build_event(&ev).unwrap();
    assert_eq!(e.manifest["schema_version"], "learned-impact-outcome/5");
    let cols = &e.manifest["full_state_columns"];
    let idx = |kind: &str, name: &str| cols[kind].as_array().unwrap().iter().position(|c| c == name).unwrap_or_else(|| panic!("{kind} lacks {name}"));
    let (ux, alive, kappa) = (idx("chunk", "u.x"), idx("bond", "alive"), idx("bond", "joint.kappa"));
    let _ = kappa;
    assert!(cols["static_bond"].as_array().unwrap().iter().any(|c| c == "geometry.area"));
    let f64s = |c: &Call, n: &str| match c.array(n) {
        Some(Arr::F64(s, d)) => (s.clone(), d.clone()),
        _ => panic!("{n}"),
    };
    let mut checked_breaks = 0;
    for c in &e.calls {
        let (s, d) = f64s(c, "fs_chunk_in");
        let def = match c.array("chunk_def") {
            Some(Arr::F32(_, d)) => d.clone(),
            _ => panic!(),
        };
        for j in 0..s[0] {
            assert_eq!(d[j * s[1] + ux] as f32, def[j * 12], "call {} chunk {j}: u.x", c.frame);
        }
        let (bs, bd) = f64s(c, "fs_bond_out");
        let (_, bi) = f64s(c, "fs_bond_in");
        let keys = match c.array("fs_bond_key") {
            Some(Arr::I64(_, k)) => k.clone(),
            _ => panic!(),
        };
        assert_eq!(keys.len(), bs[0]);
        for j in 0..bs[0] {
            assert!(bd[j * bs[1] + alive] <= bi[j * bs[1] + alive], "a bond never heals");
            checked_breaks += usize::from(bd[j * bs[1] + alive] < bi[j * bs[1] + alive]);
        }
    }
    assert!(checked_breaks > 0, "some bond breaks in a call");
}

/// A live `/5` call (harness A: `augment_live` on a world rebuilt from the recorded
/// boundaries, with the training manifest's columns) equals the recorded call bit for bit,
/// and its static rows are the event's for those entities.
#[test]
fn a_live_full_state_call_equals_the_recorded_one() {
    use stress_learned::full_state::{load_frame, restore};
    let (ev, _) = record("fs5live");
    let e = build_event(&ev).unwrap();
    let cols = e.manifest["full_state_columns"].clone();
    let ctx: serde_json::Value = serde_json::from_slice(&std::fs::read(ev.join("context.json")).unwrap()).unwrap();
    let scene: Scene = serde_json::from_slice(&std::fs::read(ev.join("scene.json")).unwrap()).unwrap();
    let bits = |a: &Arr| match a {
        Arr::F64(s, d) => (s.clone(), d.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
        Arr::F32(s, d) => (s.clone(), d.iter().map(|v| u64::from(v.to_bits())).collect()),
        Arr::I32(s, d) => (s.clone(), d.iter().map(|&v| v as u64).collect()),
        Arr::I64(s, d) => (s.clone(), d.iter().map(|&v| v as u64).collect()),
    };
    let ev_arr = |n: &str| e.event_arrays.iter().find(|(k, _)| k == n).map(|(_, a)| a.clone()).unwrap();
    for c in e.calls.iter().step_by(3).take(4) {
        let (w0, w1) = (restore(&scene, &load_frame(&ev, c.frame).unwrap()).unwrap(), restore(&scene, &load_frame(&ev, c.frame + 1).unwrap()).unwrap());
        let mut live = c.clone();
        live.arrays.retain(|(k, _)| !k.starts_with("fs_"));
        let (_, statics) = stress_learned::full_state_calls::augment_live(&w0, Some(&w1), &ctx, std::slice::from_mut(&mut live), Some(&cols)).unwrap();
        for (k, a) in c.arrays.iter().filter(|(k, _)| k.starts_with("fs_")) {
            assert_eq!(live.array(k).map(bits), Some(bits(a)), "frame {}: {k}", c.frame);
        }
        // static rows: the event's rows of the same keys
        for kind in ["chunk", "bond", "impactor"] {
            let (Arr::I64(_, lk), Arr::F64(ls, lv)) = (statics.iter().find(|(k, _)| *k == format!("fs_static_{kind}_key")).unwrap().1.clone(), statics.iter().find(|(k, _)| *k == format!("fs_static_{kind}")).unwrap().1.clone()) else { panic!() };
            let (Arr::I64(_, ek), Arr::F64(_, evv)) = (ev_arr(&format!("fs_static_{kind}_key")), ev_arr(&format!("fs_static_{kind}"))) else { panic!() };
            let w = ls[1];
            for (j, key) in lk.iter().enumerate() {
                let r = ek.iter().position(|x| x == key).unwrap();
                assert!(lv[j * w..(j + 1) * w].iter().zip(&evv[r * w..(r + 1) * w]).all(|(a, b)| a.to_bits() == b.to_bits()), "{kind} {key}: static row");
            }
        }
    }
}

/// Written calls store identical arrays once (shared offsets) and read back bit for bit.
#[test]
fn written_calls_share_identical_arrays_and_read_back_exactly() {
    let (ev, _) = record("dedup");
    let e = build_event(&ev).unwrap();
    let out = tmp("dedup-out");
    write_event(&out, &e, true).unwrap();
    let blob = zstd::decode_all(std::fs::File::open(out.join("arrays.bin.zst")).unwrap()).unwrap();
    let entries: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(out.join("calls.json")).unwrap()).unwrap();
    let bytes_of = |a: &Arr| -> Vec<u8> {
        match a {
            Arr::F32(_, d) => d.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Arr::I32(_, d) => d.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Arr::I64(_, d) => d.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Arr::F64(_, d) => d.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    };
    let (mut offsets, mut total) = (std::collections::HashSet::new(), 0usize);
    for (c, en) in e.calls.iter().zip(&entries) {
        for (k, a) in &c.arrays {
            let m = &en["arrays"][k];
            let off = m["offset"].as_u64().unwrap() as usize;
            let b = bytes_of(a);
            assert_eq!(&blob[off..off + b.len()], &b[..], "frame {}: {k}", c.frame);
            offsets.insert(off);
            total += 1;
        }
    }
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("manifest.json")).unwrap()).unwrap();
    for (k, a) in &e.event_arrays {
        let off = manifest["event_arrays"][k]["offset"].as_u64().unwrap() as usize;
        let b = bytes_of(a);
        assert_eq!(&blob[off..off + b.len()], &b[..], "event array {k}");
    }
    assert!(offsets.len() < total, "some arrays share their bytes");
}

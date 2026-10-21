//! The impact-outcome recorder and generator (plan P1): the M5 scenes are the ones P0
//! pinned, recordings are causal, complete and replayable, and recipes are deterministic,
//! independent draws.

use std::collections::HashSet;
use std::path::PathBuf;

use stress_learned::catalog::{read_f32, read_i32, Catalog};
use stress_learned::oracle_config::OracleConfig;
use stress_learned::outcome::{record_outcome_event, replay_fingerprint, scene_hash, OutcomeRecipe, RecordOptions};
use stress_learned::procedural::{latin_hypercube, m5_free_wall, m5_wall_row, wall_body, wall_recipes, Layout, Ranges, Rng};
use stress_ref::builders;
use stress_ref::scene::{ImpactorDesc, ImpactorShape};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("stress-learned-outcome-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The M5 scenes rebuilt here are byte for byte the ones P0 pinned (scene hashes of
/// `stress_rubble::scenes::realtime(.., 0.15)`, p0-feasibility-2026-10-20.md step 1).
#[test]
fn m5_scenes_are_p0s() {
    let oracle = OracleConfig::named("a12_realtime").unwrap();
    let mut s = m5_free_wall();
    oracle.apply(&mut s);
    s.sim.duration = 0.15;
    assert_eq!(scene_hash(&s), "3657bced087a0e78");
    let mut s = m5_wall_row();
    oracle.apply(&mut s);
    s.sim.duration = 0.15;
    assert_eq!(scene_hash(&s), "f9665b3652139750");
}

#[test]
fn oracle_config_is_explicit_and_round_trips() {
    let c = OracleConfig::named("a12_realtime").unwrap();
    let v = serde_json::to_value(&c).unwrap();
    // Every Methods switch is written, not only those off the built-in default.
    assert_eq!(v["methods"].as_object().unwrap().len(), stress_ref::scene::Methods::NAMES.len());
    assert_eq!(serde_json::from_value::<OracleConfig>(v).unwrap(), c);
    assert!(OracleConfig::named("nope").is_err());
}

/// A small free wall struck hard enough to break: the recording is complete, causal and
/// replays to the same end state.
#[test]
fn a_recording_is_complete_causal_and_replays() {
    let mut s = builders::new_scene("tiny", "tiny wall");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), builders::oracle_concrete(None));
    s.materials.insert("steel".into(), builders::elastic_steel());
    let layout = Layout { counts: [4, 1, 4], chunk: 0.1, opening: None, supports: "free" };
    s.bodies.push(wall_body(&layout, "wall", 0.0));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Sphere { radius: 0.05 },
        mass: 50.0,
        position: [0.0, -0.06 - 25.0 * 0.004, 0.2],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 25.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    let recipe = OutcomeRecipe { event_id: "tiny".into(), family: "test".into(), tier: 1, split_group: "tiny".into(), layout_group: layout.key(), parameters: serde_json::json!({}), scene: s };
    let out = tmp("record");
    let opts = RecordOptions::new(OracleConfig::named("a12_realtime").unwrap());
    let sum = record_outcome_event(&recipe, &out, &opts, &|_| {}).unwrap();
    assert!(!sum.truncated, "{}", sum.stop_reason);
    assert!(sum.broken_bonds > 0, "the strike breaks bonds");
    assert_eq!(sum.breaks_last_half_second, 0);
    let ev = out.join("tiny");
    assert!(!out.join("tiny.partial").exists());
    let cat = Catalog::load(&ev.join("catalog.json")).unwrap();
    let get_i = |n: &str| read_i32(&ev, &cat.arrays[n]).unwrap();
    let shape = |n: &str| cat.arrays[n].shape.clone();
    let (f, e) = (sum.frames, shape("static_bond_endpoints")[1]);
    assert_eq!(shape("state_bond_connected"), vec![f + 1, e]);
    assert_eq!(shape("frame_bond_break_bin"), vec![f, e]);
    assert_eq!(shape("qs_bond_broken"), vec![f, e]);
    // Breaks are exactly the frame's disconnections, and only of bonds connected at its start.
    let (conn, bins) = (get_i("state_bond_connected"), get_i("frame_bond_break_bin"));
    let mut breaks = 0;
    for k in 0..f {
        for b in 0..e {
            let (before, after, bin) = (conn[k * e + b], conn[(k + 1) * e + b], bins[k * e + b]);
            assert_eq!(bin >= 0, before == 1 && after == 0, "frame {k} bond {b}");
            assert!((-1..8).contains(&bin));
            breaks += usize::from(bin >= 0);
        }
    }
    assert_eq!(breaks, sum.broken_bonds);
    // The causal inputs: the strike was predicted before the frame it happens in.
    let contacts = get_i("contact_i32");
    assert!(!contacts.is_empty() && contacts[0] == 0, "a predicted contact in frame 0");
    let prior = read_f32(&ev, &cat.arrays["frame_bond_qs_prior_util"]).unwrap();
    assert!(prior[..e].iter().any(|&u| u > 0.0), "the frame-0 QS prior loads the wall");
    let ctx: serde_json::Value = serde_json::from_slice(&std::fs::read(ev.join("context.json")).unwrap()).unwrap();
    assert_eq!(ctx["oracle_config"]["name"], "a12_realtime");
    assert_eq!(ctx["split_group"], "tiny");
    let (fp, frames) = replay_fingerprint(&ev).unwrap();
    assert_eq!((fp, frames), (sum.fingerprint.clone(), sum.frames));
}

#[test]
fn recipes_are_deterministic_independent_draws() {
    let r = Ranges::pilot();
    let a = wall_recipes("free_wall", 1, 24, 7, &r).unwrap();
    let b = wall_recipes("free_wall", 1, 24, 7, &r).unwrap();
    assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    let ids: HashSet<_> = a.iter().map(|r| r.split_group.clone()).collect();
    assert_eq!(ids.len(), a.len(), "each recipe is its own split group");
    for x in &a {
        assert_eq!(x.scene.impactors.len(), 1);
        let v = x.parameters["impactors"][0]["speed"].as_f64().unwrap();
        assert!((r.speed.0..=r.speed.1).contains(&v), "speed {v}");
        assert_eq!(x.scene.gravity, [0.0; 3]);
    }
    let t2 = wall_recipes("anchored_wall", 2, 12, 7, &r).unwrap();
    for x in &t2 {
        assert!((2..=4).contains(&x.scene.impactors.len()));
        assert!(x.scene.ground.is_some());
        assert!(x.scene.bodies[0].chunks.iter().any(|c| c.support != stress_ref::scene::Support::None));
    }
    assert!(wall_recipes("house", 1, 1, 7, &r).is_err());
}

#[test]
fn latin_hypercube_uses_every_stratum_once() {
    let mut rng = Rng::new(3);
    let n = 17;
    let pts = latin_hypercube(n, 5, &mut rng);
    for d in 0..5 {
        let mut strata: Vec<usize> = pts.iter().map(|p| (p[d] * n as f64) as usize).collect();
        strata.sort_unstable();
        assert_eq!(strata, (0..n).collect::<Vec<_>>());
    }
}

#[test]
fn impactors_start_clear_of_ground_wall_and_each_other() {
    let r = Ranges::pilot();
    for fam in ["free_wall", "anchored_wall"] {
        for x in wall_recipes(fam, 2, 40, 11, &r).unwrap() {
            let imps = &x.scene.impactors;
            let radius = |d: &ImpactorDesc| match d.shape {
                ImpactorShape::Sphere { radius } => radius,
                ImpactorShape::Box { half_extents } => (half_extents.iter().map(|h| h * h).sum::<f64>()).sqrt(),
            };
            for (i, a) in imps.iter().enumerate() {
                if x.scene.ground.is_some() {
                    assert!(a.position[2] - radius(a) > 0.0, "{} impactor {i} starts in the ground", x.event_id);
                }
                for b in &imps[i + 1..] {
                    let d: f64 = (0..3).map(|k| (a.position[k] - b.position[k]).powi(2)).sum::<f64>().sqrt();
                    assert!(d >= radius(a) + radius(b), "{} impactors overlap", x.event_id);
                }
            }
        }
    }
}

#[test]
fn ensembles_cover_evaluation_splits_and_keep_the_parent_split_group() {
    use stress_learned::procedural::{split_of, with_splits_and_ensembles, EnsemblePlan};
    let plan = EnsemblePlan { seed: 5, ..EnsemblePlan::default() };
    let base = wall_recipes("free_wall", 1, 60, 5, &Ranges::pilot()).unwrap();
    let out = with_splits_and_ensembles(base.clone(), &plan);
    let mut by_parent: std::collections::BTreeMap<String, Vec<&OutcomeRecipe>> = Default::default();
    for r in &out {
        by_parent.entry(r.split_group.clone()).or_default().push(r);
    }
    assert_eq!(by_parent.len(), base.len(), "members keep their parent's split group");
    let mut train_ens = 0;
    for b in &base {
        let members = &by_parent[&b.split_group];
        let split = split_of(&b.layout_group, &b.family, &plan);
        assert!(members.iter().all(|m| m.parameters["split"] == split));
        if split != "train" {
            assert_eq!(members.len(), 8, "every evaluation configuration is an ensemble");
        }
        if members.len() > 1 {
            train_ens += usize::from(split == "train");
            for (k, m) in members.iter().enumerate() {
                assert_eq!(m.parameters["ensemble"]["member"], k);
                assert_eq!(m.parameters["ensemble"]["id"], b.event_id.as_str());
                let (p0, p1) = (b.scene.impactors[0].position, m.scene.impactors[0].position);
                let off: f64 = (0..3).map(|i| (p0[i] - p1[i]).powi(2)).sum::<f64>().sqrt();
                assert!(off > 0.0 && off <= 1.5e-3, "offset {off}");
                let (v0, v1) = (b.scene.impactors[0].velocity, m.scene.impactors[0].velocity);
                let along: f64 = (0..3).map(|i| (p1[i] - p0[i]) * v0[i]).sum();
                assert!(along.abs() < 1e-9, "the offset is across the approach");
                let s = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                assert!((s(v1) / s(v0) - 1.0).abs() <= 1e-3);
            }
        }
    }
    assert!(train_ens > 0, "some training configurations are ensembles");
}

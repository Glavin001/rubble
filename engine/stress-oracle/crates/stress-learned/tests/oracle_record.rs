//! Record two frames from a tiny builder scene and validate the event.

use std::path::PathBuf;

use stress_learned::{validate_event, OracleSession, RecordMeta};
use stress_ref::builders;
use stress_ref::world::World;

#[test]
fn record_bond_tension_two_frames() {
    let mut scene = builders::bond_tension();
    scene.sim.record_peak_utilization = true;
    scene.sim.duration = scene.sim.frame_dt * 2.0;
    let mut world = World::new(&scene);
    let initial_joint=world.solver.bonds.iter().flat_map(|s|s.iter().map(|b|stress_learned::joint_history::pack(&b.joint))).collect::<Vec<_>>();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("learned_impact_oracle_tension");
    let meta = RecordMeta {
        event_id: "oracle_tension".into(),
        structure_type: "bond_tension".into(),
        load_type: "axial_pull".into(),
        notes: "integration test".into(),
    };
    let mut session = OracleSession::begin(&dir, &world, meta, true).expect("begin");
    for _ in 0..2 {
        session.begin_frame(&mut world).expect("begin_frame");
        world.step_frame();
        session.end_frame(&mut world).expect("end_frame");
    }
    let path = session.finish().expect("finish");
    let man = validate_event(&path).expect("validate");
    assert_eq!(man.n_frames, 2);
    assert!(man.n_chunks >= 2);
    assert!(man.n_bonds >= 1);
    let replay: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("replay.json")).unwrap()).unwrap();
    assert_eq!(replay["format"], "learned-impact-replay/1");
    assert_eq!(replay["world_state_format"], "learned-world-state/1");
    assert_eq!(replay["frames"].as_array().unwrap().len(), 3);
    assert_eq!(replay["frames"][0]["time"], 0.0);
    assert_eq!(
        replay["frames"][2]["centers"].as_array().unwrap().len(),
        man.n_chunks as usize
    );
    let center: [f64; 3] =
        serde_json::from_value(replay["frames"][2]["centers"][0].clone()).unwrap();
    assert_eq!(center, world.solver.chunk_position(0, 0).to_array());
    assert!(path.join("scene.json").exists());
    let context: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("context.json")).unwrap()).unwrap();
    assert_eq!(context["format"], "learned-impact-context/2");
    assert_eq!(context["load_basis"], "frame_start_cluster");
    assert_eq!(context["qs_probe"], true);
    assert_eq!(context["joint_history"]["format"],stress_learned::joint_history::FORMAT);
    assert_eq!(context["bond_response"]["format"],stress_learned::response_history::FORMAT);
    let keys:Vec<(usize,usize)>=serde_json::from_value(context["bond_keys"].clone()).unwrap();
    let (first,_)=stress_learned::joint_history::read_snapshot(&path.join("joint_history"),0,keys.len()).unwrap();
    assert_eq!(first.iter().map(|r|r.map(f64::to_bits)).collect::<Vec<_>>(),initial_joint.iter().map(|r|r.map(f64::to_bits)).collect::<Vec<_>>());
    let (last,flags)=stress_learned::joint_history::read_snapshot(&path.join("joint_history"),2,keys.len()).unwrap();
    for ((row,flag),&(s,b)) in last.iter().zip(&flags).zip(&keys) {
        let bond=&world.solver.bonds[s][b];
        assert_eq!(row.map(f64::to_bits),stress_learned::joint_history::pack(&bond.joint).map(f64::to_bits));
        assert_eq!(stress_learned::joint_history::unpack(*row,flag[0]==1,flag[1]).unwrap(),bond.joint);
        assert_eq!(flag[2],bond.alive as i32);assert_eq!(flag[3],bond.connected() as i32);
    }
    let (response,response_flags)=stress_learned::response_history::read_snapshot(&path.join("bond_response"),2,keys.len()).unwrap();
    for ((row,flag),&(s,b)) in response.iter().zip(response_flags).zip(&keys) {
        let bond=&world.solver.bonds[s][b];
        assert_eq!(row[..3],bond.force.lin.to_array());
        assert_eq!(row[6..9],bond.elastic.lin.to_array());
        assert_eq!(row[12].to_bits(),bond.stored.to_bits());
        assert_eq!(flag,[bond.alive as i32,bond.connected() as i32]);
    }
    let saved_scene: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join("scene.json")).unwrap()).unwrap();
    let methods = serde_json::to_value(world.scene.sim.methods).unwrap();
    assert_eq!(saved_scene["sim"]["methods"], methods);
    assert_eq!(context["solver_methods"], methods);
    assert_eq!(
        context["bond_frames"].as_array().unwrap().len(),
        man.n_bonds as usize
    );
    let frame = &replay["frames"][2];
    let velocity: [f64; 3] = serde_json::from_value(frame["velocities"][0].clone()).unwrap();
    for (a, b) in velocity
        .into_iter()
        .zip(world.solver.chunk_velocity(0, 0).0.to_array())
    {
        assert!((a - b).abs() < 1e-10);
    }
    assert!(
        (frame["mechanical_energy"].as_f64().unwrap() - world.mechanical_energy()).abs() < 1e-8
    );
    assert!(
        (frame["dissipated_energy"].as_f64().unwrap() - world.dissipated_energy()).abs() < 1e-8
    );
}

#[test]
fn wall_impact_records_contact_load_bins() {
    use stress_learned::catalog::{self, Catalog};

    let mut scene = builders::wall_impact(20.0, "bin_wall");
    scene.sim.record_peak_utilization = true;
    scene.sim.duration = scene.sim.frame_dt; // one frame
    let mut world = World::new(&scene);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("learned_impact_contact_bins");
    let meta = RecordMeta {
        event_id: "bin_wall".into(),
        structure_type: "wall_b5".into(),
        load_type: "ram_20mps".into(),
        notes: "contact bin test".into(),
    };
    let mut session = OracleSession::begin(&dir, &world, meta, false).expect("begin");
    session.begin_frame(&mut world).expect("begin_frame");
    world.step_frame();
    session.end_frame(&mut world).expect("end_frame");
    let path = session.finish().expect("finish");
    let frame = path.join("frames/000000");
    let cat = Catalog::load(&frame.join("catalog.json")).expect("catalog");
    let loads =
        catalog::read_f32(&frame, cat.arrays.get("chunk_load_bins").unwrap()).expect("loads");
    let contact =
        catalog::read_i32(&frame, cat.arrays.get("chunk_contact").unwrap()).expect("contact");
    let load_norm: f32 = loads.iter().map(|x| x.abs()).sum();
    let n_contact: i32 = contact.iter().sum();
    assert!(
        n_contact > 0,
        "expected some chunks marked contacted, got {n_contact}"
    );
    assert!(
        load_norm > 0.0,
        "expected non-zero contact/scripted load bins, got {load_norm}"
    );
    let replay: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("replay.json")).unwrap()).unwrap();
    let initial: [f64; 3] =
        serde_json::from_value(replay["frames"][0]["impactors"][0]["center"].clone()).unwrap();
    let end: [f64; 3] =
        serde_json::from_value(replay["frames"][1]["impactors"][0]["center"].clone()).unwrap();
    assert_ne!(initial, end, "projectile must move during impact");
    let recorded = &replay["frames"][1]["impactors"][0];
    let actual = &world.impactors[0];
    assert_eq!(recorded["angular_velocity"], serde_json::json!(actual.angular_velocity.to_array()));
    assert_eq!(recorded["mass"], serde_json::json!(actual.mass));
    assert_eq!(recorded["inertia"], serde_json::json!(actual.inertia.m));
    assert_eq!(recorded["crush_used"], serde_json::json!(actual.crush_used));
    assert_eq!(recorded["crush_depth"], serde_json::json!(actual.crush_depth));
    assert_eq!(recorded["material"], serde_json::to_value(&actual.material).unwrap());
    for (actual, expected) in end
        .into_iter()
        .zip(world.impactors[0].pose.position.to_array())
    {
        assert!(
            (actual - expected).abs() < 1e-12,
            "JSON roundtrip changed the projectile pose"
        );
    }
}

#[test]
fn replay_preserves_current_projectile_dynamics_not_initial_scene_values() {
    use stress_ref::math::Vec3;
    let scene = builders::wall_impact(20.0, "replay_dynamics");
    let mut world = World::new(&scene);
    let p = &mut world.impactors[0];
    p.angular_velocity = Vec3::new(1.0, 2.0, 3.0);
    p.crush = Some((400.0, 1000.0));
    p.crush_used = 123.0;
    p.crush_depth = 0.25;
    p.driven = true;
    let replay = stress_learned::replay::Replay::new(&world, &[(0,0)]);
    let value = serde_json::to_value(replay).unwrap();
    let saved = &value["frames"][0]["impactors"][0];
    assert_eq!(saved["angular_velocity"], serde_json::json!([1.0,2.0,3.0]));
    assert_eq!(saved["crush"], serde_json::json!([400.0,1000.0]));
    assert_eq!(saved["crush_used"], 123.0);
    assert_eq!(saved["crush_depth"], 0.25);
    assert_eq!(saved["driven"], true);
    assert_eq!(value["frames"][0]["cluster_rotations"].as_array().unwrap().len(), 1);
    assert_eq!(world.impactors[0].crush_used, 123.0); // capture is read-only
}

#[test]
fn recorder_rejects_unrecorded_refinement_nodes() {
    let scene = builders::bond_tension();
    let mut world = World::new(&scene);
    world.solver.chunks[0][1].active = false;
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("learned_impact_topology_guard");
    let meta = RecordMeta {
        event_id: "topology_guard".into(),
        structure_type: "bond_tension".into(),
        load_type: "test".into(),
        notes: String::new(),
    };
    let mut session = OracleSession::begin(&dir, &world, meta, false).unwrap();
    session.begin_frame(&mut world).unwrap();
    world.solver.chunks[0][1].active = true;
    assert!(session
        .end_frame(&mut world)
        .unwrap_err()
        .contains("new active chunk"));
}


#[test]
fn rigid_preview_records_causal_inputs_without_changing_oracle() {
    use stress_learned::oracle::{record_event_options, LoadBasis};
    use stress_learned::catalog::{self, Catalog};
    let scene = builders::wall_impact(20.0, "preview_wall");
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("rigid_preview_parity");
    for preview in [false, true] {
        record_event_options(scene.clone(), base.join(preview.to_string()), RecordMeta {
            event_id: "preview_wall".into(), structure_type: "wall".into(),
            load_type: "ram".into(), notes: "preview parity".into(),
        }, 2, false, LoadBasis::FrameStart, preview).unwrap();
    }
    // Exact oracle labels, input history and replay must survive instrumentation.
    assert_eq!(std::fs::read(base.join("false/replay.json")).unwrap(), std::fs::read(base.join("true/replay.json")).unwrap());
    for fi in 0..2 {
        let a = base.join(format!("false/frames/{fi:06}"));
        let b = base.join(format!("true/frames/{fi:06}"));
        let cat = Catalog::load(&a.join("catalog.json")).unwrap();
        for entry in cat.arrays.values() {
            assert_eq!(std::fs::read(a.join(&entry.path)).unwrap(), std::fs::read(b.join(&entry.path)).unwrap(), "oracle changed: {}", entry.path);
        }
        let cat = Catalog::load(&b.join("catalog.json")).unwrap();
        let loads = catalog::read_f32(&b, &cat.arrays["chunk_preview_load_bins"]).unwrap();
        assert!(loads.iter().all(|x| x.is_finite()));
        if fi == 0 { assert!(loads.iter().any(|x| x.abs() > 0.0)); }
    }
    let mut world = World::new(&scene);
    world.scene.sim.refine_utilization = Some(0.1);
    assert!(world.learned_rigid_preview(4).is_err());
}


#[test]
fn resume_does_not_accept_a_valid_but_truncated_clip() {
    use stress_learned::generate::{run_jobs_with, Job, RunJobsOpts};
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("resume_complete_frames");
    let job = Job { event_id: "tension".into(), structure_type: "bond_tension".into(),
        load_type: "pull".into(), frames: 2, qs_probe: false, scene: builders::bond_tension() };
    run_jobs_with(&root, "test", &[job.clone()], &RunJobsOpts { resume: false, ..Default::default() }).unwrap();
    let path = root.join("tension/manifest.json");
    let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["n_frames"] = serde_json::json!(1);
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    // A valid prefix must truncate every snapshot contract, not just the frame count.
    for sidecar in ["joint_history","bond_response"] {
        let history_path=root.join(format!("tension/{sidecar}/manifest.json"));
        let mut history:serde_json::Value=serde_json::from_slice(&std::fs::read(&history_path).unwrap()).unwrap();
        history["snapshots"].as_array_mut().unwrap().truncate(2);
        std::fs::write(history_path,serde_json::to_vec(&history).unwrap()).unwrap();
    }
    let replay_path=root.join("tension/replay.json");
    let mut replay:serde_json::Value=serde_json::from_slice(&std::fs::read(&replay_path).unwrap()).unwrap();
    replay["frames"].as_array_mut().unwrap().truncate(2);
    std::fs::write(replay_path,serde_json::to_vec(&replay).unwrap()).unwrap();
    assert_eq!(validate_event(&root.join("tension")).unwrap().n_frames, 1);
    let result = run_jobs_with(&root, "test", &[job], &RunJobsOpts::default()).unwrap();
    assert_eq!(result.events[0].n_frames, 2);
    assert_eq!(validate_event(&root.join("tension")).unwrap().n_frames, 2);
}

#[test]
fn declared_joint_history_rejects_missing_bytes_and_legacy_resume() {
    use stress_learned::generate::{run_jobs_with,Job,RunJobsOpts};
    let root=PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("joint_history_integrity");
    let job=Job {event_id:"tension".into(),structure_type:"bond_tension".into(),load_type:"pull".into(),
        frames:1,qs_probe:false,scene:builders::bond_tension()};
    run_jobs_with(&root,"test",&[job.clone()],&RunJobsOpts {resume:false,..Default::default()}).unwrap();
    let event=root.join("tension");validate_event(&event).unwrap();
    let file=event.join("joint_history/000001.f64");let bytes=std::fs::read(&file).unwrap();
    std::fs::write(&file,&bytes[..bytes.len()-1]).unwrap();assert!(validate_event(&event).is_err());
    std::fs::write(file,bytes).unwrap();validate_event(&event).unwrap();
    let path=event.join("context.json");let mut context:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    context.as_object_mut().unwrap().remove("joint_history");
    std::fs::write(path,serde_json::to_vec(&context).unwrap()).unwrap();
    assert!(run_jobs_with(&root,"test",&[job],&RunJobsOpts::default()).unwrap_err().contains("joint-history recording contract changed"));
}

#[test]
fn declared_response_rejects_missing_bytes_and_legacy_resume() {
    use stress_learned::generate::{run_jobs_with,Job,RunJobsOpts};
    let root=PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("response_history_integrity");
    let job=Job {event_id:"tension".into(),structure_type:"bond_tension".into(),load_type:"pull".into(),
        frames:1,qs_probe:false,scene:builders::bond_tension()};
    run_jobs_with(&root,"test",&[job.clone()],&RunJobsOpts {resume:false,..Default::default()}).unwrap();
    let event=root.join("tension");validate_event(&event).unwrap();
    let file=event.join("bond_response/000001.f64");let bytes=std::fs::read(&file).unwrap();
    std::fs::write(&file,&bytes[..bytes.len()-1]).unwrap();assert!(validate_event(&event).is_err());
    std::fs::write(&file,&bytes).unwrap();validate_event(&event).unwrap();
    let path=event.join("context.json");let mut context:serde_json::Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    context.as_object_mut().unwrap().remove("bond_response");
    std::fs::write(path,serde_json::to_vec(&context).unwrap()).unwrap();
    assert!(run_jobs_with(&root,"test",&[job],&RunJobsOpts::default()).unwrap_err().contains("bond-response recording contract changed"));
    assert_eq!(std::fs::read(file).unwrap(),bytes);
}

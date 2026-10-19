use std::fs;
use stress_learned::{phase_history,runtime_snapshot::RuntimeGraph};
use stress_ref::{builders,material::Material,scene::{SolveMode,TimeFunction},solver::SolverEvent,world::World};

#[test]
fn dense_phase_recording_preserves_native_state_and_exact_break_time() {
    let root=std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("dense_stress_phase");
    if root.exists(){fs::remove_dir_all(&root).unwrap();}fs::create_dir(&root).unwrap();
    let m=Material::analytic_test();let force=4.*m.tensile_strength*0.01;
    let mut scene=builders::bond_pull("phase-fracture",m,0.05,TimeFunction::Ramp{t0:0.,t1:0.005,value:force},0.02);
    scene.sim.solve_mode=SolveMode::Explicit;scene.sim.frame_dt=0.001;
    let mut native=World::new(&scene);let mut recorded=native.clone();let graph=RuntimeGraph::new(&native).unwrap();
    let mut rec=phase_history::Recorder::new(&root,&graph).unwrap();
    // An unfinished recording has no publishable completed manifest.
    assert!(!root.join("stress_phases/manifest.json").exists());
    for frame in 0..20 {
        rec.frame=frame;native.step_frame();recorded.step_frame_with_stress_provider(&mut rec).unwrap();
        assert_eq!(serde_json::to_value(graph.capture(&native).unwrap()).unwrap(),serde_json::to_value(graph.capture(&recorded).unwrap()).unwrap());
        assert_eq!(native.solver.events,recorded.solver.events);
        assert_eq!(serde_json::to_value(native.solver.energy).unwrap(),serde_json::to_value(recorded.solver.energy).unwrap());
    }
    let rows=rec.rows as usize;let phases=rec.phases.len();assert_eq!(rec.rows,native.solver.bond_evaluations);
    assert_eq!(recorded.solver.bond_evaluations,0);assert_eq!(phases as u64,native.solver.substeps);rec.finish().unwrap();
    let dir=root.join("stress_phases");let meta:serde_json::Value=serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(meta["status"],"completed");
    for (name,bytes) in [("inputs.f64",rows*51*8),("targets.f64",rows*42*8),("input_flags.i32",rows*4*4),
        ("target_flags.i32",rows*4*4),("bond_slots.u32",rows*4),("chunks.f64",phases*2*18*8),("chunk_flags.u64",phases*2*2*8)] {
        assert_eq!(fs::metadata(dir.join(name)).unwrap().len() as usize,bytes,"{name}");
    }
    let tags=fs::read(dir.join("target_flags.i32")).unwrap();
    let flags:Vec<i32>=tags.chunks_exact(4).map(|a|i32::from_le_bytes(a.try_into().unwrap())).collect();
    let mut times=Vec::new();
    for phase in meta["phases"].as_array().unwrap() {
        let start=phase["row_start"].as_u64().unwrap() as usize;let count=phase["rows"].as_u64().unwrap() as usize;
        for row in start..start+count {if flags[row*4+3]==1 {times.push(phase["time"].as_f64().unwrap()+phase["dt"].as_f64().unwrap());}}
    }
    let expected:Vec<_>=native.solver.events.iter().filter_map(|e|if let SolverEvent::Broken{time,..}=e{Some(*time)}else{None}).collect();
    assert!(!times.is_empty());assert_eq!(times,expected);
    // Unset ductility is retained as positive infinity in the raw f64 contract.
    let inputs=fs::read(dir.join("inputs.f64")).unwrap();
    assert_eq!(f64::from_le_bytes(inputs[16*8..17*8].try_into().unwrap()),f64::INFINITY);
}

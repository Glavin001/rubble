//! Record every native stress response, alongside causal frame-start snapshots.
use std::{fs,path::PathBuf,time::Instant};
use serde_json::json;
use stress_learned::{oracle::prepare_scene,phase_history::Recorder,runtime_snapshot::RuntimeGraph};
use stress_ref::{scene::Scene,world::World};
fn run()->Result<(),String>{
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=3{return Err("usage: learned_impact_phase_record SCENE NEW_OUTPUT FRAMES".into())}
    let frames:u32=args[2].parse().map_err(|_|"invalid frame count")?;if frames==0{return Err("positive frames required".into())}
    let source=fs::read(&args[0]).map_err(|e|e.to_string())?;
    let scene:Scene=serde_json::from_slice(&source).map_err(|e|e.to_string())?;
    let mut w=World::new(&prepare_scene(scene,frames));let graph=RuntimeGraph::new(&w)?;
    let out=PathBuf::from(&args[1]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    fs::create_dir(out.join("snapshots")).map_err(|e|e.to_string())?;
    fs::write(out.join("source-scene.json"),source).map_err(|e|e.to_string())?;
    fs::write(out.join("graph.json"),serde_json::to_vec(&graph).unwrap()).map_err(|e|e.to_string())?;
    fs::write(out.join("snapshots/000000.json"),serde_json::to_vec(&graph.capture(&w)?).unwrap()).map_err(|e|e.to_string())?;
    let mut recorder=Recorder::new(&out,&graph)?;let start=Instant::now();
    for frame in 0..frames {
        recorder.frame=frame;w.step_frame_with_stress_provider(&mut recorder)?;
        fs::write(out.join(format!("snapshots/{:06}.json",frame+1)),serde_json::to_vec(&graph.capture(&w)?).unwrap()).map_err(|e|e.to_string())?;
        println!("{}",json!({"frame":frame,"phases":recorder.phases.len(),"oracle_response_rows":recorder.rows,"seconds":start.elapsed().as_secs_f64()}));
    }
    let phases=recorder.phases.len();let rows=recorder.rows;recorder.finish()?;
    let report=json!({"status":"completed","frames":frames,"phases":phases,"oracle_response_rows":rows,
        "engine_fallback_evaluations":w.solver.bond_evaluations,"seconds":start.elapsed().as_secs_f64(),
        "scope":"Oracle-only full native stress-phase recording. All native steps retained; no trained model or speed claim. Frame inputs are snapshots; later phase observations are not causal frame inputs."});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("stress phase recording failed: {e}");std::process::exit(1)}}

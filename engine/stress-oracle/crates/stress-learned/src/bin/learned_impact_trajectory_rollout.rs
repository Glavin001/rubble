//! Persistent native world driven by complete learned stress trajectories.
//! No oracle recording is accepted or read. The caller may compare a separate run.
use std::{fs,path::PathBuf,io::{self,BufRead,Write}};
use serde_json::json;
use stress_learned::{runtime_snapshot::RuntimeGraph,runtime_trajectory::Prediction,oracle::prepare_scene};
use stress_ref::{scene::Scene,world::World};
fn emit(value:serde_json::Value)->Result<(),String> {
    let mut out=io::stdout().lock();serde_json::to_writer(&mut out,&value).map_err(|e|e.to_string())?;
    writeln!(out).map_err(|e|e.to_string())?;out.flush().map_err(|e|e.to_string())
}
fn run()->Result<(),String> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=3 {return Err("usage: learned_impact_trajectory_rollout SCENE_JSON NEW_OUTPUT FRAMES".into())}
    let scene:Scene=serde_json::from_slice(&fs::read(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let frames:u32=args[2].parse().map_err(|_|"invalid frames")?;if frames==0 {return Err("frames must be positive".into())}
    let scene=prepare_scene(scene,frames);let mut world=World::new(&scene);let graph=RuntimeGraph::new(&world)?;
    let out=PathBuf::from(&args[1]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    for d in ["snapshots","predictions"] {fs::create_dir(out.join(d)).map_err(|e|e.to_string())?}
    fs::write(out.join("graph.json"),serde_json::to_vec(&graph).unwrap()).map_err(|e|e.to_string())?;
    emit(json!({"kind":"graph","graph":graph}))?;
    let mut input=io::stdin().lock();let mut reports=Vec::new();
    for i in 0..frames {
        let (snapshot,kinematics)=graph.capture_with_kinematics(&world)?;
        fs::write(out.join(format!("snapshots/{i:06}.json")),serde_json::to_vec(&snapshot).unwrap()).map_err(|e|e.to_string())?;
        emit(json!({"kind":"snapshot","index":i,"snapshot":snapshot,"kinematics":kinematics}))?;
        let mut line=String::new();if input.read_line(&mut line).map_err(|e|e.to_string())?==0 {return Err("inference process closed before providing trajectory".into())}
        let prediction:Prediction=serde_json::from_str(&line).map_err(|e|e.to_string())?;
        let mut provider=prediction.decode(&graph,&world)?;
        fs::write(out.join(format!("predictions/{i:06}.json")),&line).map_err(|e|e.to_string())?;
        let before=world.solver.substeps;
        world.step_frame_with_stress_provider(&mut provider)?;
        if world.solver.bond_evaluations!=0 {return Err("unexpected analytical stress evaluation".into())}
        let report=json!({"analytic_bond_evaluations":world.solver.bond_evaluations,"stress_substeps":world.solver.substeps-before,
            "provider_calls":provider.calls,"provider_responses":provider.responses});
        let value=json!({"interval":i,"report":report,"mechanical_energy_j":world.mechanical_energy(),"dissipated_energy_j":world.dissipated_energy(),"external_work_j":world.solver.energy.external_work});
        reports.push(value);fs::write(out.join("reports.json"),serde_json::to_vec_pretty(&reports).unwrap()).map_err(|e|e.to_string())?;
    }
    let final_state=graph.capture(&world)?;
    fs::write(out.join(format!("snapshots/{frames:06}.json")),serde_json::to_vec(&final_state).unwrap()).map_err(|e|e.to_string())?;
    emit(json!({"kind":"completed","frames":frames,"reports":reports,"snapshot":final_state}))
}
fn main() {if let Err(e)=run() {eprintln!("independent trajectory rollout failed: {e}");std::process::exit(1)}}

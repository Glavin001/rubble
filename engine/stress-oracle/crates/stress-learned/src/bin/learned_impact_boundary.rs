//! Record causal bond kinematics before each native frame, without phase labels.
use std::{fs,path::PathBuf};
use stress_learned::{oracle::prepare_scene,runtime_snapshot::RuntimeGraph};
use stress_ref::{scene::Scene,world::World};
fn run()->Result<(),String>{
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=3{return Err("usage: learned_impact_boundary SCENE NEW_OUTPUT FRAMES".into())}
    let frames:u32=args[2].parse().map_err(|_|"invalid frame count")?;
    if frames==0{return Err("positive frames required".into())}
    let source=fs::read(&args[0]).map_err(|e|e.to_string())?;
    let scene:Scene=serde_json::from_slice(&source).map_err(|e|e.to_string())?;
    let mut world=World::new(&prepare_scene(scene,frames));let graph=RuntimeGraph::new(&world)?;
    let out=PathBuf::from(&args[1]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    for name in ["snapshots","kinematics"]{fs::create_dir(out.join(name)).map_err(|e|e.to_string())?;}
    fs::write(out.join("graph.json"),serde_json::to_vec(&graph).unwrap()).map_err(|e|e.to_string())?;
    fs::write(out.join("source-scene.json"),source).map_err(|e|e.to_string())?;
    for frame in 0..=frames {
        let (snapshot,kinematics)=graph.capture_with_kinematics(&world)?;
        fs::write(out.join(format!("snapshots/{frame:06}.json")),serde_json::to_vec(&snapshot).unwrap()).map_err(|e|e.to_string())?;
        fs::write(out.join(format!("kinematics/{frame:06}.json")),serde_json::to_vec(&kinematics).unwrap()).map_err(|e|e.to_string())?;
        if frame<frames{world.step_frame();}
    }
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&serde_json::json!({"status":"completed","frames":frames,
        "scope":"Current-state kinematics captured before native stepping. No future labels read. No learned inference or speed claim."})).unwrap()).map_err(|e|e.to_string())?;
    Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("boundary recording failed: {e}");std::process::exit(1)}}

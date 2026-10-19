//! Dense oracle trajectory probe for designing split-aware learned targets.
//! Observes the unchanged native substeps; never a model performance claim.
use std::{fs,path::PathBuf,io::{BufWriter,Write}};
use serde_json::json;
use stress_learned::{oracle::prepare_scene,replay,runtime_snapshot::RuntimeGraph};
use stress_ref::{scene::Scene,world::World};
fn run()->Result<(),String> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=3{return Err("usage: learned_impact_trajectory_probe SCENE NEW_OUTPUT FRAMES".into())}
    let frames:u32=args[2].parse().map_err(|_|"invalid frame count")?;if frames==0{return Err("positive frame count required".into())}
    let scene:Scene=serde_json::from_slice(&fs::read(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut world=World::new(&prepare_scene(scene,frames));let graph=RuntimeGraph::new(&world)?;
    let out=PathBuf::from(&args[1]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    fs::write(out.join("graph.json"),serde_json::to_vec(&graph).unwrap()).map_err(|e|e.to_string())?;
    let mut file=BufWriter::new(fs::File::create(out.join("trajectory.jsonl")).map_err(|e|e.to_string())?);
    let mut count=0;let mut error=None;
    let mut capture=|world:&World|{
        if error.is_some(){return}
        let frame=replay::capture(world,graph.chunk_keys());
        let local:Vec<_>=graph.chunk_keys().iter().map(|&(s,c)|{let st=&world.solver.chunks[s][c];[st.u.to_array(),st.th.to_array(),st.v.to_array(),st.w.to_array()]}).collect();
        let connected:Vec<_>=graph.bond_keys().iter().map(|&(s,b)|world.solver.bonds[s][b].connected()).collect();
        let row=json!({"substeps":world.solver.substeps,"world":frame,"local":local,"connected":connected});
        let result=serde_json::to_writer(&mut file,&row).map_err(|e|e.to_string()).and_then(|_|writeln!(file).map_err(|e|e.to_string()));
        match result {Ok(())=>count+=1,Err(e)=>error=Some(e)}
    };
    capture(&world);
    for _ in 0..frames {world.step_frame_observed(&mut capture)?;}
    drop(capture);if let Some(error)=error{return Err(error)}file.flush().map_err(|e|e.to_string())?;
    fs::write(out.join("endpoint.json"),serde_json::to_vec(&graph.capture(&world)?).unwrap()).map_err(|e|e.to_string())?;
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&json!({"status":"completed","frames":frames,"snapshots":count,
        "scope":"Read-only native oracle substep probe: physical poses/velocities, cluster IDs, hidden local state, connectivity and projectile state. Diagnostic target design, not a complete training dataset or learned inference."})).unwrap()).map_err(|e|e.to_string())?;
    println!("captured {count} native snapshots");Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("trajectory probe failed: {e}");std::process::exit(1)}}

//! Oracle-assisted causal ablation of one learned frame. Never learned inference
//! or a physically consistent hybrid model: channels are deliberately mixed to
//! diagnose which errors leave a damaging state for subsequent native evolution.
use std::{fs,path::Path};
use serde_json::{json,Value};
use stress_learned::{runtime_snapshot::RuntimeGraph,runtime_trajectory::{Prediction,TrajectoryProvider}};
use stress_ref::{scene::Scene,world::World,solver::{ReferenceSolver,ChunkLoads},stress_provider::{StressProvider,SuppliedStress,BondJob}};

struct Mixture { learned:TrajectoryProvider, mode:String, oracle_rows:u64 }
impl StressProvider for Mixture {
    fn responses(&mut self,s:&ReferenceSolver,loads:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)->Result<Vec<SuppliedStress>,String> {
        let predicted=self.learned.responses(s,loads,jobs,dt,fracture)?;
        if self.mode=="learned" {return Ok(predicted)}
        Ok(predicted.into_iter().zip(jobs).map(|(mut p,&job)| {
            let o=s.oracle_stress_response(job,dt,fracture);self.oracle_rows+=1;
            match self.mode.as_str() {
                "oracle"=>o,
                "oracle-forces"=>{p.total_force=o.total_force;p.damping_dissipated=o.damping_dissipated;
                    p.joint.force=o.joint.force;p.joint.stored=o.joint.stored;p.joint.measures=o.joint.measures;p},
                "oracle-history"=>{p.joint.state=o.joint.state;p.joint.dissipated=o.joint.dissipated;
                    p.joint.overshoot=o.joint.overshoot;p.joint.disconnected=o.joint.disconnected;p},
                _=>unreachable!(),
            }
        }).collect())
    }
}
fn save(path:impl AsRef<Path>,value:&impl serde::Serialize)->Result<(),String>{
    fs::write(path,serde_json::to_vec(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
}
fn run()->Result<(),String>{
    let a:Vec<_>=std::env::args().skip(1).collect();
    if a.len()!=4{return Err("usage: learned_impact_channel_probe SAVED_NATIVE_RUN NEW_OUTPUT MODE ORACLE_TAIL_FRAMES".into())}
    if !["learned","oracle","oracle-forces","oracle-history"].contains(&a[2].as_str()){return Err("invalid diagnostic mode".into())}
    let tail:u32=a[3].parse().map_err(|_|"invalid tail frames")?;
    let source=Path::new(&a[0]);let out=Path::new(&a[1]);
    let graph_bytes=fs::read(source.join("graph.json")).map_err(|e|e.to_string())?;
    let raw:Value=serde_json::from_slice(&graph_bytes).map_err(|e|e.to_string())?;
    let scene:Scene=serde_json::from_value(raw["scene"].clone()).map_err(|e|e.to_string())?;
    let mut world=World::new(&scene);let graph=RuntimeGraph::new(&world)?;
    if serde_json::to_vec(&graph).unwrap()!=graph_bytes{return Err("initial graph mismatch".into())}
    let initial=graph.capture(&world)?;
    if serde_json::to_vec(&initial).unwrap()!=fs::read(source.join("snapshots/000000.json")).map_err(|e|e.to_string())?{return Err("initial state mismatch".into())}
    let prediction:Prediction=serde_json::from_slice(&fs::read(source.join("predictions/000000.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut provider=Mixture{learned:prediction.decode(&graph,&world)?,mode:a[2].clone(),oracle_rows:0};
    fs::create_dir(out).map_err(|e|e.to_string())?;fs::create_dir(out.join("snapshots")).map_err(|e|e.to_string())?;
    save(out.join("graph.json"),&graph)?;save(out.join("snapshots/000000.json"),&initial)?;
    world.step_frame_with_stress_provider(&mut provider)?;
    let first=graph.capture(&world)?;
    let learned_control_exact=if a[2]=="learned" {
        if serde_json::to_vec(&first).unwrap()!=fs::read(source.join("snapshots/000001.json")).map_err(|e|e.to_string())?{return Err("learned control differs".into())}Some(true)
    }else{None};
    save(out.join("snapshots/000001.json"),&first)?;
    for i in 0..tail {world.step_frame();save(out.join(format!("snapshots/{:06}.json",i+2)),&graph.capture(&world)?)?;}
    save(out.join("events.json"),&world.solver.events)?;
    save(out.join("report.json"),&json!({"status":"completed","mode":a[2],"frames":tail+1,"oracle_tail_frames":tail,
        "first_frame_oracle_response_rows":provider.oracle_rows,"native_tail_analytic_evaluations":world.solver.bond_evaluations,
        "learned_first_frame_exact":learned_control_exact,"source_native":source,
        "scope":"Diagnostic channel intervention on first frame, followed by native oracle evolution. Oracle-forces replaces total/elastic force, damping work, stored energy and measures. Oracle-history replaces joint state, dissipated/signed work and disconnection. Mixtures need not be physically consistent; never a learned accuracy, inference or speed claim. No target training admission."}))
}
fn main(){if let Err(e)=run(){eprintln!("channel probe failed: {e}");std::process::exit(1)}}

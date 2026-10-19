//! Exact oracle-assisted control of the supplied-stress integration boundary.
//! This program performs analytical stress work explicitly; it is NOT inference.
use std::{fs,path::PathBuf,time::Instant};
use serde_json::json;
use stress_learned::{oracle::prepare_scene,runtime_snapshot::RuntimeGraph};
use stress_ref::{scene::Scene,world::World,solver::{ReferenceSolver,ChunkLoads},
    stress_provider::{StressProvider,SuppliedStress,BondJob}};
struct Oracle {calls:u64,evaluations:u64,negative_residuals:u64,min_residual:f64,max_negative_relative:f64}
impl StressProvider for Oracle {
    fn responses(&mut self,s:&ReferenceSolver,_:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)->Result<Vec<SuppliedStress>,String> {
        self.calls+=1;self.evaluations+=jobs.len() as u64;
        Ok(jobs.iter().map(|&job|{
            let response=s.oracle_stress_response(job,dt,fracture);
            if response.joint.overshoot<0. {
                self.negative_residuals+=1;
                self.min_residual=self.min_residual.min(response.joint.overshoot);
                self.max_negative_relative=self.max_negative_relative.max(-response.joint.overshoot/response.joint.stored.max(f64::MIN_POSITIVE));
            }
            response
        }).collect())
    }
}
fn run()->Result<(),String> {
    let args:Vec<_>=std::env::args().skip(1).collect();if args.len()!=3{return Err("usage: learned_impact_provider_control SCENE NEW_OUTPUT FRAMES".into())}
    let frames:u32=args[2].parse().map_err(|_|"invalid frames")?;if frames==0{return Err("positive frames required".into())}
    let scene:Scene=serde_json::from_slice(&fs::read(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut reference=World::new(&prepare_scene(scene,frames));let mut supplied=reference.clone();let graph=RuntimeGraph::new(&reference)?;
    let out=PathBuf::from(&args[1]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    fs::write(out.join("graph.json"),serde_json::to_vec(&graph).unwrap()).map_err(|e|e.to_string())?;
    fs::write(out.join("initial.json"),serde_json::to_vec(&graph.capture(&reference)?).unwrap()).map_err(|e|e.to_string())?;
    let mut oracle=Oracle{calls:0,evaluations:0,negative_residuals:0,min_residual:0.,max_negative_relative:0.};let mut rows=Vec::new();let start=Instant::now();
    for frame in 0..frames {
        stress_ref::profile::enable();stress_ref::profile::reset();
        let time=Instant::now();reference.step_frame();let native_seconds=time.elapsed().as_secs_f64();
        fs::write(out.join(format!("native-profile-{frame:06}.txt")),stress_ref::profile::report()).map_err(|e|e.to_string())?;
        stress_ref::profile::reset();
        let time=Instant::now();supplied.step_frame_with_stress_provider(&mut oracle)?;let supplied_seconds=time.elapsed().as_secs_f64();
        fs::write(out.join(format!("supplied-profile-{frame:06}.txt")),stress_ref::profile::report()).map_err(|e|e.to_string())?;
        let a=serde_json::to_value(graph.capture(&reference)?).unwrap();let b=serde_json::to_value(graph.capture(&supplied)?).unwrap();
        fs::write(out.join(format!("native-{frame:06}.json")),serde_json::to_vec(&a).unwrap()).map_err(|e|e.to_string())?;
        fs::write(out.join(format!("supplied-{frame:06}.json")),serde_json::to_vec(&b).unwrap()).map_err(|e|e.to_string())?;
        let state_equal=a==b;let events_equal=reference.solver.events==supplied.solver.events;
        let energy_equal=serde_json::to_value(reference.solver.energy).unwrap()==serde_json::to_value(supplied.solver.energy).unwrap();
        let row=json!({"frame":frame,"complete_snapshot_exact":state_equal,"events_exact":events_equal,"energy_ledger_exact":energy_equal,
            "native_analytic_evaluations":reference.solver.bond_evaluations,"oracle_provider_evaluations":oracle.evaluations,
            "engine_analytic_fallback_evaluations":supplied.solver.bond_evaluations,"provider_calls":oracle.calls,
            "negative_native_residuals":oracle.negative_residuals,"min_native_residual_j":oracle.min_residual,"max_negative_residual_over_stored":oracle.max_negative_relative,
            "native_seconds":native_seconds,"supplied_oracle_seconds":supplied_seconds});
        println!("{row}");rows.push(row);
        let exact=state_equal&&events_equal&&energy_equal&&supplied.solver.bond_evaluations==0&&oracle.evaluations==reference.solver.bond_evaluations;
        fs::write(out.join("report.json"),serde_json::to_vec_pretty(&json!({"status":if !exact{"failed_parity"}else if frame+1==frames{"completed_exact"}else{"running"},
            "scope":"Exact oracle-assisted native stress-phase control. Provider performs reference evaluations; no learned model or speedup claim. Compare full captured state, event sequence/times and complete stress energy ledger at every frame.",
            "seconds":start.elapsed().as_secs_f64(),"rows":rows})).unwrap()).map_err(|e|e.to_string())?;
        if !exact{return Err("supplied response path differs from native reference; saved both snapshots".into())}
    }
    Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("stress provider control failed: {e}");std::process::exit(1)}}

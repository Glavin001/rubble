//! Read-only oracle joint-energy witnesses at the native stress phase.
//! No learned predictions, output clamping, or changes to native integration.
use std::{fs,path::PathBuf};
use serde_json::{json,Value};
use stress_learned::{joint_history,oracle::prepare_scene,runtime_snapshot::RuntimeGraph};
use stress_ref::{scene::Scene,world::World,solver::{ReferenceSolver,ChunkLoads},
    stress_provider::{StressProvider,SuppliedStress,BondJob}};

struct Probe {calls:u64,evaluations:u64,witnesses:Vec<Value>}
impl StressProvider for Probe {
    fn responses(&mut self,s:&ReferenceSolver,_:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)->Result<Vec<SuppliedStress>,String> {
        self.calls+=1;self.evaluations+=jobs.len() as u64;
        let mut responses=Vec::with_capacity(jobs.len());
        for &job in jobs {
            let response=s.oracle_stress_response(job,dt,fracture);
            if response.joint.overshoot<0. {
                let b=&s.bonds[job.0][job.1];let g=&b.geometry;
                let a=s.chunks[job.0][g.a];let c=s.chunks[job.0][g.b];
                let d=if s.config.methods.corotational_bonds {
                    g.corotation(a.u,a.th,c.u,c.th).deformation()
                } else {g.kinematics(a.u,a.th,c.u,c.th)};
                // Verify that the witness carries the exact constitutive input,
                // independent of the private cached-corotation implementation.
                let direct=b.model().evaluate(&b.joint,&d,dt*job.2,fracture);
                if direct.state!=response.joint.state || direct.force!=response.joint.force
                    || direct.stored!=response.joint.stored || direct.overshoot!=response.joint.overshoot {
                    return Err("energy witness kinematics differs from native response".into())
                }
                let values=|st:&stress_ref::joint::JointState|joint_history::pack(st).map(|v|if v==f64::INFINITY{None}else{Some(v)});
                self.witnesses.push(json!({"key":[job.0,job.1],"substep":s.substeps,"time":s.time,"dt":dt*job.2,
                    "deformation":d.to_array(),"old_values":values(&b.joint),"new_values":values(&direct.state),
                    "old_rebar_broken":b.joint.rebar_broken,"old_mode":joint_history::mode_code(b.joint.mode),
                    "new_rebar_broken":direct.state.rebar_broken,"new_mode":joint_history::mode_code(direct.state.mode),
                    "force":direct.force.to_array(),"stored":direct.stored,"dissipated":direct.dissipated,"overshoot":direct.overshoot,
                    "stiffness":b.stiffness.as_local().to_array(),"geometry":format!("{g:?}"),
                    "strength":format!("{:?}",b.strength),"rebar":format!("{:?}",b.rebar),"weibull":b.weibull}));
            }
            responses.push(response);
        }
        Ok(responses)
    }
}
fn run()->Result<(),String>{
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=3{return Err("usage: learned_impact_energy_probe SCENE NEW_OUTPUT FRAMES".into())}
    let frames:u32=args[2].parse().map_err(|_|"invalid frames")?;if frames==0{return Err("positive frames required".into())}
    let scene:Scene=serde_json::from_slice(&fs::read(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut w=World::new(&prepare_scene(scene,frames));let graph=RuntimeGraph::new(&w)?;
    let out=PathBuf::from(&args[1]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    let mut probe=Probe{calls:0,evaluations:0,witnesses:Vec::new()};
    for _ in 0..frames {w.step_frame_with_stress_provider(&mut probe)?;}
    fs::write(out.join("endpoint.json"),serde_json::to_vec(&graph.capture(&w)?).unwrap()).map_err(|e|e.to_string())?;
    let report=json!({"scope":"Oracle-only diagnostic of signed joint energy residuals, with exact direct-law witness checks. Not learned inference or a training corpus.",
        "status":"completed","frames":frames,"provider_calls":probe.calls,"oracle_evaluations":probe.evaluations,
        "engine_fallback_evaluations":w.solver.bond_evaluations,"fields":joint_history::FIELDS,"witnesses":probe.witnesses});
    fs::write(out.join("witnesses.json"),serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
    println!("{}",json!({"status":"completed","witnesses":report["witnesses"].as_array().unwrap().len(),"oracle_evaluations":probe.evaluations}));Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("oracle energy probe failed: {e}");std::process::exit(1)}}

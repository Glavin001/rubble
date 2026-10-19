//! Diagnostic only: exact oracle endpoints supplied to the experimental adapter.
//! Each oracle step starts from a clone of the current independent world. This
//! deliberately uses an oracle and must never be reported as learned inference.
use std::{fs,path::PathBuf,time::Instant};
use serde_json::json;
use stress_learned::{oracle::prepare_scene,replay,runtime_snapshot::RuntimeGraph};
use stress_ref::{world::World,scene::Scene,external_stress::{StressEndpoint,ChunkEndpoint,BondEndpoint}};

fn endpoint(start:&World,end:&World,g:&RuntimeGraph)->StressEndpoint {
    let chunks=g.chunk_keys().iter().filter_map(|&(s,c)|{
        let a=&start.solver.chunks[s][c];let b=&end.solver.chunks[s][c];
        a.active.then_some(ChunkEndpoint{key:(s,c),delta:[b.u-a.u,b.th-a.th,b.v-a.v,b.w-a.w]})
    }).collect();
    let bonds=g.bond_keys().iter().filter_map(|&(s,i)|{
        let a=&start.solver.bonds[s][i];let b=&end.solver.bonds[s][i];
        a.alive.then_some(BondEndpoint{key:(s,i),state:b.joint.clone(),force:b.force,elastic:b.elastic,
            stored:b.stored,measures:b.measures,connected:b.connected()})
    }).collect();
    StressEndpoint{start_time:start.solver.time,dt:start.scene.sim.frame_dt,chunks,bonds}
}
fn run()->Result<(),String> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=3 {return Err("usage: learned_impact_endpoint_control SCENE_JSON NEW_OUTPUT FRAMES".into())}
    let frames:u32=args[2].parse().map_err(|_|"invalid frame count")?;
    if frames==0 {return Err("positive frames required".into())}
    let scene:Scene=serde_json::from_slice(&fs::read(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut world=World::new(&prepare_scene(scene,frames));let graph=RuntimeGraph::new(&world)?;
    let out=PathBuf::from(&args[1]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    fs::write(out.join("graph.json"),serde_json::to_vec(&graph).unwrap()).map_err(|e|e.to_string())?;
    let start=Instant::now();let mut reports=Vec::new();
    for i in 0..frames {
        let mut reference=world.clone();let before=reference.solver.bond_evaluations;
        let initial=replay::capture(&world,graph.chunk_keys());
        reference.step_frame();let exact=endpoint(&world,&reference,&graph);
        let oracle_evaluations=reference.solver.bond_evaluations-before;
        let report=world.step_frame_external_endpoint(&exact)?;
        let expected=replay::capture(&reference,graph.chunk_keys());let actual=replay::capture(&world,graph.chunk_keys());
        let rmse=|a:&Vec<[f64;3]>,b:&Vec<[f64;3]>|->f64 {
            (a.iter().zip(b).flat_map(|(a,b)|a.iter().zip(b).map(|(x,y)|(x-y).powi(2))).sum::<f64>()/(a.len()*3) as f64).sqrt()
        };
        let reset_chunks=initial.clusters.iter().zip(&expected.clusters).filter(|(a,b)|a!=b).count();
        let value=json!({"interval":i,"position_coordinate_rmse_m":rmse(&actual.centers,&expected.centers),
            "velocity_coordinate_rmse_mps":rmse(&actual.velocities,&expected.velocities),
            "angular_velocity_coordinate_rmse_radps":rmse(&actual.angular_velocities,&expected.angular_velocities),
            "oracle_split_reset_chunks":reset_chunks,"oracle_bond_evaluations":oracle_evaluations,
            "oracle_mechanical_energy_j":expected.mechanical_energy,"adapter_mechanical_energy_j":actual.mechanical_energy,
            "oracle_fragments":reference.solver.clusters.len(),"adapter_fragments":world.solver.clusters.len(),"adapter":report});
        fs::write(out.join(format!("interval-{i:06}.json")),serde_json::to_vec(&json!({"start":initial,"oracle":expected,"adapter":actual})).unwrap()).map_err(|e|e.to_string())?;
        println!("{value}");reports.push(value);
        fs::write(out.join("report.json"),serde_json::to_vec_pretty(&json!({"status":if i+1==frames {"completed"}else{"running"},
            "scope":"Oracle-assisted integration control, NOT learned inference. Exact f64 endpoint targets from an oracle clone of each live start, including local-frame resets. Measures adapter error even with perfect endpoint values; no claim these split-reset targets were supervised by current models.",
            "seconds":start.elapsed().as_secs_f64(),"reports":reports})).unwrap()).map_err(|e|e.to_string())?;
    }
    Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("endpoint control failed: {e}");std::process::exit(1)}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_endpoint_control_matches_static_zero_load_world() {
        let mut scene=stress_ref::builders::bond_tension();scene.loads.clear();scene.gravity=[0.;3];
        scene.sim.solve_mode=stress_ref::scene::SolveMode::Explicit;
        let mut a=World::new(&scene);let mut b=a.clone();let g=RuntimeGraph::new(&a).unwrap();b.step_frame();
        let e=endpoint(&a,&b,&g);let r=a.step_frame_external_endpoint(&e).unwrap();assert_eq!(r.analytic_bond_evaluations,0);
        for &(s,c) in g.chunk_keys(){assert!((a.solver.chunk_position(s,c)-b.solver.chunk_position(s,c)).norm()<1e-12)}
    }
}

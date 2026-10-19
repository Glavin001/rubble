//! Causal learned material update, native integration and native contacts.
use std::{fs,path::PathBuf,time::Instant};
use serde_json::json;
use stress_ref::{world::World,scene::{Scene,Support},math::Vec3,builders::*};
use stress_learned::{oracle::prepare_scene,runtime_snapshot::RuntimeGraph,causal_state::{Teacher,Provider,Model,FORMAT,NY,NG}};
fn run()->Result<(),String>{
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.first().is_some_and(|a|a=="fixture"){
        if args.len()!=6{return Err("fixture NEW_SCENE SPEED MASS X_OFFSET STRENGTH_SCALE".into())}
        let parse=|i:usize|args[i].parse::<f64>().map_err(|e|e.to_string());
        let speed=parse(2)?;let mass=parse(3)?;let offset=parse(4)?;let strength=parse(5)?;
        let mut scene=wall_impact(speed,"causal-small-wall");let counts=[8,1,8];
        let mut chunks=grid(Vec3::new(-0.4,0.,-0.1),counts,Vec3::splat(0.1),"concrete");
        for i in 0..counts[0]{chunks[grid_index(counts,i,0,0)].support=Support::Fixed;}
        let bonds=auto_bonds(&chunks,|_,_|"concrete".into());scene.bodies=vec![body("wall",chunks,bonds)];
        scene.impactors[0].mass=mass;scene.impactors[0].position=[offset,-0.121,0.3];
        scene.impactors[0].shape=stress_ref::scene::ImpactorShape::Box{half_extents:[0.12,0.12,0.12]};
        let m=scene.materials.get_mut("concrete").unwrap();m.tensile_strength*=strength;m.compressive_strength*=strength;m.cohesion*=strength;
        scene.sim.frame_dt=0.001;scene.metrics.clear();scene.probes.clear();
        if std::path::Path::new(&args[1]).exists(){return Err("scene already exists".into())}
        fs::write(&args[1],serde_json::to_vec_pretty(&scene).unwrap()).map_err(|e|e.to_string())?;return Ok(())
    }
    if args.len()<4||args.len()>5{return Err("teacher SCENE NEW_OUTPUT FRAMES | model SCENE NEW_OUTPUT FRAMES WEIGHTS".into())}
    let training=args[0]=="teacher";let collecting=args[0]=="collect"||args[0]=="collect-energy";let energy_decoder=args[0]=="energy"||args[0]=="collect-energy";
    if !training&&!collecting&&!energy_decoder&&args[0]!="model"{return Err("unknown mode".into())}
    if training&&args.len()!=4||!training&&args.len()!=5{return Err("wrong mode arguments".into())}
    let frames:u32=args[3].parse().map_err(|_|"frames")?;if frames==0{return Err("positive frames".into())}
    let source=fs::read(&args[1]).map_err(|e|e.to_string())?;
    let scene:Scene=serde_json::from_slice(&source).map_err(|e|e.to_string())?;
    let mut w=World::new(&prepare_scene(scene,frames));let graph=RuntimeGraph::new(&w)?;
    let out=PathBuf::from(&args[2]);fs::create_dir(&out).map_err(|e|e.to_string())?;
    let native=if training{out.clone()}else{out.join("native")};if !training{fs::create_dir(&native).map_err(|e|e.to_string())?;}
    fs::create_dir(native.join("snapshots")).map_err(|e|e.to_string())?;
    fs::write(native.join("source-scene.json"),source).map_err(|e|e.to_string())?;
    fs::write(native.join("graph.json"),serde_json::to_vec(&graph).unwrap()).map_err(|e|e.to_string())?;
    let mut teacher=if training{Some(Teacher::new(&out.join("rows.f64"))?)}else{None};
    let mut provider=if training{None}else{Some(Provider{model:Model::load(std::path::Path::new(&args[4]))?,calls:0,rows:0,
        labels:if collecting{Some(Teacher::new(&out.join("rows.f64"))?)}else{None},energy_decoder})};
    let start=Instant::now();let mut reports=Vec::new();
    for i in 0..=frames{
        fs::write(native.join(format!("snapshots/{i:06}.json")),serde_json::to_vec(&graph.capture(&w)?).unwrap()).map_err(|e|e.to_string())?;
        if i==frames{break}
        let before=w.solver.substeps;
        if let Some(p)=&mut teacher{w.step_frame_with_stress_provider(p)?}else{w.step_frame_with_stress_provider(provider.as_mut().unwrap())?}
        if w.solver.bond_evaluations!=0{return Err("unexpected fallback evaluator".into())}
        reports.push(json!({"interval":i,"report":{"analytic_bond_evaluations":0,"stress_substeps":w.solver.substeps-before},
            "mechanical_energy_j":w.mechanical_energy(),"dissipated_energy_j":w.dissipated_energy(),"external_work_j":w.solver.energy.external_work}));
        if i%10==0{println!("{}",json!({"frame":i,"seconds":start.elapsed().as_secs_f64()}));}
    }
    fs::write(native.join("reports.json"),serde_json::to_vec_pretty(&reports).unwrap()).map_err(|e|e.to_string())?;
    let (rows,calls,queries,nx)=if let Some(p)=&mut teacher{p.flush()?;(p.rows,p.calls,p.rows,p.input_width)}else{let mut p=provider.unwrap();
        let queries=if let Some(l)=&mut p.labels{l.flush()?;l.rows}else{0};(p.rows,p.calls,queries,69)};
    let report=json!({"status":"completed","format":FORMAT,"frames":frames,"rows":rows,"calls":calls,"oracle_queries":queries,"energy_decoder":energy_decoder,
        "analytic_fallback_evaluations":w.solver.bond_evaluations,"input_width":nx,"output_width":NY,"gate_width":NG,"seconds":start.elapsed().as_secs_f64(),
        "scope":"Causal material-state model. All native contact and mechanical substeps retained. No runtime speedup claim."});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
    if !training&&!collecting{fs::write(out.join("plan.json"),serde_json::to_vec_pretty(&json!({"status":"completed","frames":frames,"checkpoint":{"sha256":args[4]},"oracle_queries":0})).unwrap()).map_err(|e|e.to_string())?;}
    println!("{report}");Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("causal material experiment failed: {e}");std::process::exit(1)}}

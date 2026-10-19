//! Record causal whole-graph inputs and matching pre-transfer physical targets.
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};
use stress_learned::{
    graph_transition::Recorder, oracle::prepare_scene, runtime_snapshot::RuntimeGraph,
};
use stress_ref::{scene::Scene, world::World};
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: learned_impact_graph_record SCENE NEW_OUTPUT FRAMES".into());
    }
    let frames: u32 = args[2].parse().map_err(|_| "invalid frames")?;
    if frames == 0 {
        return Err("positive frames required".into());
    }
    let source = fs::read(&args[0]).map_err(|e| e.to_string())?;
    let scene: Scene = serde_json::from_slice(&source).map_err(|e| e.to_string())?;
    let mut w = World::new(&prepare_scene(scene, frames));
    let graph = RuntimeGraph::new(&w)?;
    let out = PathBuf::from(&args[1]);
    fs::create_dir(&out).map_err(|e| e.to_string())?;
    fs::create_dir(out.join("snapshots")).map_err(|e| e.to_string())?;
    fs::write(out.join("source-scene.json"), source).map_err(|e| e.to_string())?;
    fs::write(out.join("graph.json"), serde_json::to_vec(&graph).unwrap())
        .map_err(|e| e.to_string())?;
    let mut recorder = Recorder::new(&out)?;
    let start = Instant::now();
    for i in 0..=frames {
        fs::write(
            out.join(format!("snapshots/{i:06}.json")),
            serde_json::to_vec(&graph.capture(&w)?).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        if i == frames {
            break;
        }
        recorder.frame = i;
        w.step_frame_with_stress_provider(&mut recorder)?;
        if i % 10 == 0 {
            println!(
                "{}",
                json!({"frame":i,"phases":recorder.phases.len(),"seconds":start.elapsed().as_secs_f64()})
            );
        }
    }
    let report = json!({"status":"completed","frames":frames,"phases":recorder.phases.len(),"oracle_queries":recorder.edges,
        "node_rows":recorder.nodes,"edge_rows":recorder.edges,"numerical_chunk_updates":w.solver.numerical_chunk_updates,
        "supplied_chunk_updates":w.solver.supplied_chunk_updates,"engine_fallback_bond_evaluations":w.solver.bond_evaluations,
        "seconds":start.elapsed().as_secs_f64(),"scope":"Oracle training capture. No learned inference. Targets before frame transfers/splits."});
    recorder.finish()?;
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("graph transition recording failed: {e}");
        std::process::exit(1)
    }
}

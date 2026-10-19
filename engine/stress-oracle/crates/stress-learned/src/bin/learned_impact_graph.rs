//! Independent graph-state inference. No oracle frames or labels are accepted.
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use stress_learned::{
    graph_live::GraphProvider, oracle::prepare_scene, runtime_snapshot::RuntimeGraph,
};
use stress_ref::{scene::Scene, world::World};
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("usage: learned_impact_graph PYTHON GRAPH_CHECKPOINT MATERIAL_CHECKPOINT SCENE NEW_OUTPUT FRAMES".into());
    }
    let frames: u32 = args[5].parse().map_err(|_| "invalid frames")?;
    if frames == 0 {
        return Err("positive frames required".into());
    }
    let source = fs::read(&args[3]).map_err(|e| e.to_string())?;
    let scene: Scene = serde_json::from_slice(&source).map_err(|e| e.to_string())?;
    let mut w = World::new(&prepare_scene(scene, frames));
    let graph = RuntimeGraph::new(&w)?;
    let out = PathBuf::from(&args[4]);
    fs::create_dir(&out).map_err(|e| e.to_string())?;
    let native = out.join("native");
    fs::create_dir(&native).map_err(|e| e.to_string())?;
    fs::create_dir(native.join("snapshots")).map_err(|e| e.to_string())?;
    fs::write(native.join("source-scene.json"), source).map_err(|e| e.to_string())?;
    fs::write(
        native.join("graph.json"),
        serde_json::to_vec(&graph).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    fs::copy(&args[1], out.join("graph-model.pt")).map_err(|e| e.to_string())?;
    fs::copy(&args[2], out.join("material-model.json")).map_err(|e| e.to_string())?;
    let mut provider = GraphProvider::new(
        Path::new(&args[0]),
        &out.join("graph-model.pt"),
        &out.join("material-model.json"),
    )?;
    let start = Instant::now();
    let mut reports = Vec::new();
    for i in 0..=frames {
        fs::write(
            native.join(format!("snapshots/{i:06}.json")),
            serde_json::to_vec(&graph.capture(&w)?).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        if i == frames {
            break;
        }
        let before = w.solver.substeps;
        w.step_frame_with_stress_provider(&mut provider)?;
        if w.solver.bond_evaluations != 0 || w.solver.numerical_chunk_updates != 0 {
            return Err("numerical stress/state fallback during graph inference".into());
        }
        reports.push(json!({"interval":i,"report":{"analytic_bond_evaluations":0,"numerical_chunk_updates":0,
            "supplied_chunk_updates":w.solver.supplied_chunk_updates,"stress_substeps":w.solver.substeps-before}}));
        if i % 10 == 0 {
            println!(
                "{}",
                json!({"frame":i,"seconds":start.elapsed().as_secs_f64()})
            );
        }
    }
    let report = json!({"status":"completed","frames":frames,"oracle_queries":0,"numerical_chunk_updates":w.solver.numerical_chunk_updates,
        "supplied_chunk_updates":w.solver.supplied_chunk_updates,"analytic_bond_evaluations":w.solver.bond_evaluations,
        "graph_calls":provider.calls,"graph_chunk_predictions":provider.chunk_predictions,"seconds":start.elapsed().as_secs_f64(),
        "scope":"Learned graph internal-velocity transition plus learned material state. Native cadence, contacts, rigid motion, kinematics and frame/topology transfers retained. No oracle resets or numerical internal kick."});
    for (path, value) in [
        (native.join("reports.json"), json!(reports)),
        (out.join("report.json"), report.clone()),
        (
            out.join("plan.json"),
            json!({"status":"completed","frames":frames,"oracle_queries":0,
            "checkpoint":{"path":"graph-model.pt","sha256":null},"material_checkpoint":"material-model.json"}),
        ),
    ] {
        fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).map_err(|e| e.to_string())?;
    }
    println!("{report}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("graph inference failed: {e}");
        std::process::exit(1)
    }
}

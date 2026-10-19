//! Reconstruct a learned World by deterministic replay, then ask the oracle to
//! advance that same state. This is diagnostic supervision, never inference.
use std::{fs, path::Path};
use serde_json::{json, Value};
use stress_learned::{phase_history::Recorder, runtime_snapshot::RuntimeGraph, runtime_trajectory::Prediction};
use stress_ref::{scene::Scene, world::World};

fn read(path: impl AsRef<Path>) -> Result<Vec<u8>, String> {
    fs::read(path.as_ref()).map_err(|e| format!("{}: {e}", path.as_ref().display()))
}
fn save(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<(), String> {
    fs::write(path, serde_json::to_vec(value).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
}
fn verify(world: &World, graph: &RuntimeGraph, run: &Path, frame: u32) -> Result<(), String> {
    let actual = serde_json::to_vec(&graph.capture(world)?).map_err(|e|e.to_string())?;
    if actual != read(run.join(format!("snapshots/{frame:06}.json")))? {
        return Err(format!("learned replay snapshot differs at frame {frame}"));
    }
    Ok(())
}
fn advance(world: &mut World, graph: &RuntimeGraph, run: &Path, frame: u32) -> Result<(), String> {
    let p: Prediction = serde_json::from_slice(&read(run.join(format!("predictions/{frame:06}.json")))?).map_err(|e|e.to_string())?;
    let mut provider = p.decode(graph, world)?;
    world.step_frame_with_stress_provider(&mut provider)?;
    if world.solver.bond_evaluations != 0 {return Err("unexpected analytical fallback during learned replay".into())}
    verify(world, graph, run, frame+1)
}
fn branch(run: &Path, out: &Path, prefix: u32) -> Result<(), String> {
    let source: Value = serde_json::from_slice(&read(run.join("graph.json"))?).map_err(|e|e.to_string())?;
    let scene: Scene = serde_json::from_value(source["scene"].clone()).map_err(|e|e.to_string())?;
    let mut world = World::new(&scene);
    let graph = RuntimeGraph::new(&world)?;
    if serde_json::to_vec(&graph).map_err(|e|e.to_string())? != read(run.join("graph.json"))? {return Err("reconstructed graph differs".into())}
    verify(&world, &graph, run, 0)?;
    for frame in 0..prefix {advance(&mut world, &graph, run, frame)?;}
    // A snapshot alone cannot restore hidden World state. Replaying a further
    // saved learned step checks that the reconstructed state also continues
    // identically before accepting any oracle labels from the branch.
    let mut learned_control = world.clone();
    advance(&mut learned_control, &graph, run, prefix)?;
    fs::create_dir(out).map_err(|e|e.to_string())?;
    fs::create_dir(out.join("snapshots")).map_err(|e|e.to_string())?;
    let (initial, kinematics) = graph.capture_with_kinematics(&world)?;
    save(out.join("graph.json"), &graph)?;
    save(out.join("snapshots/000000.json"), &initial)?;
    save(out.join("boundary-kinematics.json"), &kinematics)?;
    save(out.join("learned-control.json"), &graph.capture(&learned_control)?)?;
    let event_start=world.solver.events.len();
    let mut recorder = Recorder::new(out, &graph)?;
    world.step_frame_with_stress_provider(&mut recorder)?;
    let phases=recorder.phases.len();let rows=recorder.rows;
    recorder.finish()?;
    save(out.join("snapshots/000001.json"), &graph.capture(&world)?)?;
    save(out.join("oracle-events.json"), &world.solver.events[event_start..].to_vec())?;
    save(out.join("report.json"), &json!({"status":"completed", "frames":1,
        "phases":phases,"oracle_response_rows":rows,"engine_fallback_evaluations":world.solver.bond_evaluations,
        "branch_from_frame":prefix,"branch_time":initial.world.time,
        "source_native":run,"replayed_snapshots_exact":prefix+1,"learned_continuation_exact":true,
        "scope":"Oracle labels from a deterministically reconstructed model-generated World. Prior frames and one further learned frame match saved snapshot bytes. The oracle continuation is diagnostic supervision, not learned inference. No original oracle frame resets. Retain parent scenario partition; do not admit diagnostic holdouts as training data."}))
}
fn main() {
    let a:Vec<_>=std::env::args().skip(1).collect();
    let result=if a.len()==3 {a[2].parse::<u32>().map_err(|e|e.to_string()).and_then(|n|branch(Path::new(&a[0]),Path::new(&a[1]),n))}
        else {Err("usage: learned_impact_oracle_branch SAVED_NATIVE_RUN NEW_OUTPUT PREFIX_FRAMES".into())};
    if let Err(e)=result {eprintln!("oracle branch failed: {e}");std::process::exit(1)}
}

#[cfg(test)]
mod tests {
    use super::*;
    use stress_learned::runtime_trajectory::{BondForecast,HistoryEncoding,FORMAT};
    use stress_ref::{builders,scene::SolveMode};
    #[test]
    fn branch_preserves_learned_state_and_rejects_tampered_continuation() {
        let root=std::env::temp_dir().join(format!("oracle-branch-test-{}",std::process::id()));
        fs::create_dir(&root).unwrap();let run=root.join("native");fs::create_dir(&run).unwrap();
        for d in ["snapshots","predictions"] {fs::create_dir(run.join(d)).unwrap();}
        let mut scene=builders::bond_tension();scene.sim.solve_mode=SolveMode::Explicit;scene.sim.frame_dt=0.001;
        let mut w=World::new(&scene);let g=RuntimeGraph::new(&w).unwrap();save(run.join("graph.json"),&g).unwrap();
        let mut oracle_from_learned=None;
        for i in 0..3 {
            let s=g.capture(&w).unwrap();save(run.join(format!("snapshots/{i:06}.json")),&s).unwrap();
            if i==1 {let mut reference=w.clone();reference.step_frame();oracle_from_learned=Some(g.capture(&reference).unwrap());}
            if i==2 {break;}
            let p=Prediction{format:FORMAT.into(),start_time:w.solver.time,dt:scene.sim.frame_dt,history_encoding:HistoryEncoding::Physical,
                bonds:s.joint_values.iter().enumerate().map(|(slot,h)| {
                    let mut response=[0.;18];response[2]=123.;response[8]=123.;
                    BondForecast{slot,history:vec![*h;2],response:vec![response;2],work_rate:vec![[0.;3];2],mode:vec![0;2],damage_complete_time:None,rebar_break_time:None,history_origin_encoded:None,ductility_set_time:[None;2],initial_work:None}
                }).collect()};
            save(run.join(format!("predictions/{i:06}.json")),&p).unwrap();
            w.step_frame_with_stress_provider(&mut p.decode(&g,&w).unwrap()).unwrap();
        }
        let out=root.join("branch");branch(&run,&out,1).unwrap();
        assert_eq!(read(out.join("snapshots/000000.json")).unwrap(),read(run.join("snapshots/000001.json")).unwrap());
        assert_eq!(read(out.join("snapshots/000001.json")).unwrap(),serde_json::to_vec(&oracle_from_learned.unwrap()).unwrap());
        // Valid JSON with altered physics must fail parity, before label output.
        let bad=run.join("predictions/000001.json");let mut p:Prediction=serde_json::from_slice(&read(&bad).unwrap()).unwrap();
        p.bonds[0].response[0][2]+=1.;save(bad,&p).unwrap();
        let rejected=root.join("rejected");assert!(branch(&run,&rejected,1).unwrap_err().contains("snapshot differs"));assert!(!rejected.exists());
        fs::remove_dir_all(root).unwrap();
    }
}

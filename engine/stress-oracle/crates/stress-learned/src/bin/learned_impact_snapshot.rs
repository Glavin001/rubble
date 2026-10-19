//! Emit current native inputs BEFORE each oracle interval, plus a separate
//! recording for parity verification. No model output is applied to the World.
use std::{fs, path::PathBuf};
use stress_learned::{
    oracle::prepare_scene, runtime_snapshot::RuntimeGraph, OracleSession, RecordMeta,
};
use stress_ref::{scene::Scene, world::World};

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut source, mut out, mut frames) = (None, None, 8u32);
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or("expected --scene PATH --out NEW_DIRECTORY [--frames N]")?;
        match arg.as_str() {
            "--scene" => source = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--frames" => frames = value.parse().map_err(|_| "invalid frames")?,
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    let source = source.ok_or("missing --scene")?;
    let out = out.ok_or("missing --out")?;
    if frames == 0 {
        return Err("frames must be positive".into());
    }
    let scene: Scene = serde_json::from_slice(&fs::read(source).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let scene = prepare_scene(scene, frames as u32);
    let mut world = World::new(&scene);
    let graph = RuntimeGraph::new(&world)?;
    fs::create_dir(&out).map_err(|e| e.to_string())?;
    fs::create_dir(out.join("snapshots")).map_err(|e| e.to_string())?;
    fs::write(
        out.join("graph.json"),
        serde_json::to_vec(&graph).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let meta = RecordMeta {
        event_id: "native_snapshot_parity".into(),
        structure_type: "parity".into(),
        load_type: "source_scene".into(),
        notes: "Read-only native inputs captured before oracle stepping".into(),
    };
    let mut recording = OracleSession::begin(out.join("recording"), &world, meta, false)?;
    for i in 0..frames {
        let snapshot = graph.capture(&world)?;
        fs::write(
            out.join(format!("snapshots/{i:06}.json")),
            serde_json::to_vec(&snapshot).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        recording.begin_frame(&mut world)?;
        world.step_frame();
        recording.end_frame(&mut world)?;
    }
    recording.finish()?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

//! Shared pieces of the cross-engine benchmark: the scene format, the timing protocol, the
//! command line and the result/pose files. Every adapter links this so all engines are timed
//! by the same code (the three-avbd adapter, in TypeScript, mirrors `protocol::run`).

pub mod cli;
pub mod machine;
pub mod protocol;
pub mod scene;

pub use cli::Args;
pub use protocol::{BodySample, Engine, EngineInfo, Mode, RunOptions};
pub use scene::{Body, Joint, Scene, Shape};

use anyhow::Result;
use std::time::Instant;

/// The whole adapter `main`: parse args, build the engine (timed), run the protocol, save.
pub fn main_with<E: Engine>(build: impl FnOnce(&Scene, &mut Args) -> Result<E>) -> Result<()> {
    let mut args = Args::parse()?;
    let mut scene = Scene::load(&args.scene)?;
    if args.smoke {
        scene.meta.protocol.smoke();
    }
    let t = Instant::now();
    let mut engine = build(&scene, &mut args)?;
    args.reject_unknown()?;
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut doc = protocol::run(&mut engine, &scene, build_ms, &args.run)?;
    doc["protocol"]["smoke"] = serde_json::json!(args.smoke);
    protocol::write_result(&args.out, &doc)?;
    if !args.run.quiet {
        eprintln!("wrote {}", args.out.display());
    }
    Ok(())
}

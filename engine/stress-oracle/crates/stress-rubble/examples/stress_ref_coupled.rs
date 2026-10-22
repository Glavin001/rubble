//! The oracle's CLI (`stress-ref`: `check scenes golden`, `run`, `compare`, `seeds`, …)
//! with every world coupled to rubble (`World::with_engine`), the learned-impact plan's
//! coupled reference (2026-10-22): rubble with production's policy (`blast_parity_routed`'s
//! `Profile::engine_policy`: 4 iterations, one step a frame; `STRESS_P0_ENGINE_POLICY=default`
//! for the engine's own), the routing as the environment sets it (`STRESS_ROUTING_TIERED`,
//! `STRESS_DEBRIS_ROUTING`, `STRESS_HANDED_INDENTS`). The PhysX counterpart is
//! `stress-physx check scenes golden`.
//!
//!   cargo run --release -p stress-rubble --example stress_ref_coupled -- check scenes golden

use std::process::ExitCode;

use stress_ref::math::Vec3;
use stress_ref::scene::Scene;
use stress_ref::world::World;

fn coupled(scene: &Scene) -> World {
    let mut e = stress_rubble::RubbleEngine::new(Vec3::from_array(scene.gravity)).expect("rubble engine");
    if !std::env::var("STRESS_P0_ENGINE_POLICY").is_ok_and(|v| v == "default") {
        let (iterations, substeps) = stress_ref::capability::Profile::BlastParityRouted.engine_policy();
        e.set_policy(iterations, substeps);
    }
    World::with_engine(scene, Box::new(e))
}

fn main() -> ExitCode {
    stress_ref::cli::main(&coupled, None, std::env::args().skip(1).collect())
}

//! The showcase behaviours (`stress-ref/tests/shared/showcases.rs`) with rubble as the
//! rigid-body engine: the same assertions must hold (ported from
//! `stress-physx/tests/showcases_physx.rs`).
//!
//! Slow (every case runs its scene and a control through the explicit stress solve),
//! so behind a feature:
//!   cargo test --release -p stress-rubble --features showcases --test showcases_rubble -- --nocapture
#![cfg(feature = "showcases")]

use stress_ref::math::Vec3;
use stress_ref::scene::Scene;
use stress_ref::world::World;
use stress_rubble::RubbleEngine;

fn make_world(scene: &Scene) -> World {
    let engine = RubbleEngine::boxed(Vec3::from_array(scene.gravity)).expect("wgpu adapter for rubble");
    World::with_engine(scene, engine)
}

#[path = "../../stress-ref/tests/shared/showcases.rs"]
mod cases;

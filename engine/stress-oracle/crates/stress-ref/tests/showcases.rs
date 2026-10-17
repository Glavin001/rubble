//! Showcase behaviours in the standalone world; the cases are in `shared/showcases.rs`.

use stress_ref::scene::Scene;
use stress_ref::world::World;

fn make_world(scene: &Scene) -> World {
    World::new(scene)
}

#[path = "shared/showcases.rs"]
mod cases;

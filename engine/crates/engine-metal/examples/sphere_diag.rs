use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::WorldCapacity;
use glam::Vec3;
fn main() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    { let mut b = SceneBuilder::new(&mut world); b.add_ground_plane(0.0); b.add_sphere(Vec3::new(0.0, 3.0, 0.0), 0.5, 1.0); }
    for s in 0..300 {
        world.step().unwrap();
        if s % 30 == 0 || s == 299 {
            let b = world.body(1);
            let c = world.last_counters();
            println!("step={s} y={} v={} contacts={}", b.position[1], b.linear_vel[1], c.contact_count);
        }
    }
}

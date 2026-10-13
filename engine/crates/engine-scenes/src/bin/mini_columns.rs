use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::{AvbdParams, WorldCapacity};
use glam::Vec3;

fn main() {
    let n: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let h: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let steps: u32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(120);
    let iters: u32 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(6);
    let dy: f32 = std::env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let mut p = AvbdParams::default();
    p.iterations = iters;
    let mut world = MetalWorld::new(WorldCapacity::scale()).unwrap();
    world.set_params(p);
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.0);
        for x in 0..n {
            for z in 0..n {
                for y in 0..h {
                    b.add_box(
                        Vec3::new(
                            (x as f32 - n as f32 / 2.0) * 1.5,
                            0.5 + y as f32 * dy,
                            (z as f32 - n as f32 / 2.0) * 1.5,
                        ),
                        Vec3::splat(0.5),
                        1.0,
                    );
                }
            }
        }
    }
    println!("mini_columns {n}x{n}x{h} iters={iters} dy={dy} bodies={}", world.body_count());
    for s in 0..steps {
        world.step().unwrap();
        if s % 30 == 29 || s + 1 == steps {
            let bodies = world.download_bodies();
            let mut ymin = f32::INFINITY;
            let mut ymax = f32::NEG_INFINITY;
            let mut ke = 0.0f32;
            for b in &bodies {
                if !b.is_dynamic() { continue; }
                ymin = ymin.min(b.position[1]);
                ymax = ymax.max(b.position[1]);
                let m = if b.inv_mass > 1e-12 { 1.0 / b.inv_mass } else { 0.0 };
                ke += 0.5 * m * glam::Vec3::from_array(b.linear_vel).length_squared();
            }
            let c = world.last_counters();
            println!("  t={s}: y=[{ymin:.2},{ymax:.2}] ke={ke:.1} awake={} contacts={} colors={}",
                c.awake_count, c.contact_count, c.color_count);
        }
    }
}

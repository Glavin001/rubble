use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::{AvbdParams, WorldCapacity};
use glam::Vec3;

fn main() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    let mut p = AvbdParams::default();
    p.iterations = 10;
    world.set_params(p);
    let n = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5u32);
    let steps = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200usize);
    {
        let mut b = SceneBuilder::new(&mut world);
        let spaced = std::env::var("STACK_SPACED").is_ok();
        if spaced {
            b.add_ground_box(0.5);
        } else {
            b.add_ground_plane(0.0);
        }
        for i in 0..n {
            let y = if spaced {
                1.0 + i as f32 * 1.5
            } else {
                0.5 + i as f32
            };
            b.add_box(Vec3::new(0.0, y, 0.0), Vec3::splat(0.5), 1.0);
        }
    }
    for s in 0..steps {
        world.step().unwrap();
        if s % 100 == 0 || s + 1 == steps {
            let bodies = world.download_bodies();
            let dynb: Vec<_> = bodies.iter().filter(|b| b.is_dynamic()).collect();
            let ys: Vec<_> = dynb.iter().map(|b| b.position[1]).collect();
            let xz: Vec<_> = dynb
                .iter()
                .map(|b| (b.position[0], b.position[2]))
                .collect();
            let spd: Vec<_> = dynb
                .iter()
                .map(|b| {
                    (b.linear_vel[0] * b.linear_vel[0]
                        + b.linear_vel[1] * b.linear_vel[1]
                        + b.linear_vel[2] * b.linear_vel[2])
                        .sqrt()
                })
                .collect();
            let c = world.last_counters();
            println!(
                "step={s} contacts={} colors={} awake={} ys={:?} xz={:?} spd={:?}",
                c.contact_count, c.color_count, c.awake_count, ys, xz, spd
            );
        }
    }
}

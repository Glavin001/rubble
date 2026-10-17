use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::WorldCapacity;
use glam::Vec3;
use std::time::Instant;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn run_scene(name: &str, capacity: WorldCapacity, build: impl FnOnce(&mut SceneBuilder), steps: u32) {
    let mut world = MetalWorld::new(capacity).expect("metal world");
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.0);
        build(&mut b);
    }
    for _ in 0..5 {
        let _ = world.step();
    }
    let mut times = Vec::new();
    let mut cpu = Vec::new();
    for _ in 0..steps {
        let _ = world.step();
        let t = world.last_timings();
        times.push(t.total_gpu);
        cpu.push(t.cpu_submit);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let c = world.last_counters();
    println!(
        "scene={name} chip={} bodies={} contacts={} gpu_p50={:.3}ms gpu_p95={:.3}ms cpu_submit_p95={:.3}ms sleep_events={} overflow=0x{:x}",
        world.chip,
        world.body_count(),
        c.contact_count,
        percentile(&times, 0.50),
        percentile(&times, 0.95),
        percentile(&cpu, 0.95),
        c.sleep_events,
        c.overflow_flags
    );
}

fn main() {
    let _ = Instant::now();
    env_logger::init();
    println!("engine-metal scale_bench (Slang→Metal)");

    run_scene("pile_1k", WorldCapacity::small(), |b| b.pile_spheres(1000, 0.35), 40);

    run_scene("pile_10k", WorldCapacity::default(), |b| b.pile_spheres(10_000, 0.3), 30);

    run_scene(
        "collapse_400",
        WorldCapacity::default(),
        |b| {
            for y in 0..20 {
                for x in 0..20 {
                    b.add_box(
                        Vec3::new(x as f32 * 0.55 - 5.0, 1.0 + y as f32 * 0.55, 0.0),
                        Vec3::splat(0.25),
                        1.0,
                    );
                }
            }
        },
        30,
    );

    run_scene(
        "asleep_50k_awake_1k",
        WorldCapacity {
            max_bodies: 65_536,
            max_colliders: 131_072,
            ..WorldCapacity::default()
        },
        |b| {
            b.sleeping_field(50_000, Vec3::new(100.0, 0.0, 100.0), 2.0);
            b.pile_spheres(1000, 0.35);
        },
        20,
    );

    run_scene(
        "rope_128",
        WorldCapacity::small(),
        |b| b.rope_capsules(128, Vec3::new(0.0, 20.0, 0.0), 0.2),
        40,
    );

    run_scene(
        "wall_smash_2k",
        WorldCapacity::default(),
        |b| {
            // Inline compact wall smash (matches engine-scenes::wall_smash_2k proportions)
            for z in 0..25u32 {
                let offset = if z % 2 == 0 { 0.0 } else { 0.5 };
                for x in 0..40u32 {
                    for y in 0..2u32 {
                        b.add_box(
                            Vec3::new(
                                (x as f32 - 20.0) + offset,
                                0.75 + z as f32 * 0.5,
                                y as f32 * 0.5,
                            ),
                            Vec3::new(0.5, 0.25, 0.25),
                            1.0,
                        );
                    }
                }
            }
            let id = b.add_sphere(Vec3::new(0.0, 5.0, -25.0), 2.0, 670.0);
            let mut body = b.world_mut().body(id);
            body.linear_vel = [0.0, 4.0, 30.0];
            b.world_mut().set_body(id, body);
        },
        30,
    );
}

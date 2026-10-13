use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::{AvbdParams, WorldCapacity};
use glam::Vec3;

fn run(label: &str, iters: u32, sleep_time: f32, n: u32, frames: u32) {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    let mut p = AvbdParams::default();
    p.iterations = iters;
    p.sleep_time = sleep_time; // huge => effectively no sleep
    world.set_params(p);
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_box(0.5);
        for i in 0..n {
            b.add_box(Vec3::new(0.0, 1.0 + i as f32 * 1.5, 0.0), Vec3::splat(0.5), 1.0);
        }
    }
    for s in 0..frames {
        world.step().unwrap();
        if s + 1 == frames || s + 1 == 200 {
            let dynb: Vec<_> = world.download_bodies().into_iter().filter(|b| b.is_dynamic()).collect();
            let sleeping = dynb.iter().filter(|b| b.flags & 4 != 0).count();
            let lat = dynb.iter().map(|b| b.position[0].abs().max(b.position[2].abs())).fold(0.0f32, f32::max);
            let ys: Vec<_> = dynb.iter().map(|b| format!("{:.2}", b.position[1])).collect();
            let ok = dynb.iter().enumerate().all(|(i, b)| {
                let expected = 0.99 + i as f32;
                b.position[0].abs() < 0.12
                    && b.position[2].abs() < 0.12
                    && (b.position[1] - expected).abs() < 0.25
            });
            println!(
                "{label} iters={iters} frame={} ok={ok} sleep={sleeping}/{n} lat={lat:.3} ys={}",
                s + 1,
                ys.join(",")
            );
        }
    }
}

fn main() {
    run("nosleep", 40, 1.0e9, 5, 600);
    run("nosleep", 80, 1.0e9, 5, 600);
    run("nosleep10", 40, 1.0e9, 10, 900);
    run("normalsleep", 40, 1.0, 5, 600);
}

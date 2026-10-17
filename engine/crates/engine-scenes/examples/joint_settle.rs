//! Settle probe for joint scenes: kinetic energy, max speed and sleepers over time.
//!
//! ```bash
//! cargo run -p engine-scenes --release --example joint_settle -- "Ragdolls (40)" 900
//! ```

use engine_exec::PhysicsBackend;
use engine_scenes::build_world;
use glam::Vec3;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map(String::as_str).unwrap_or("Ragdolls (40)");
    let steps: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(900);
    let every: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let from: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let (mut world, _) = build_world(name)?;
    for step in 1..=steps {
        world.step().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        if step >= from && step % every == 0 {
            let bodies = world.download_bodies();
            let (mut ke, mut vmax, mut sleeping, mut dynamic) = (0.0f32, 0.0f32, 0, 0);
            let mut fastest = 0usize;
            for (i, b) in bodies.iter().enumerate().filter(|(_, b)| b.is_dynamic()) {
                dynamic += 1;
                if b.is_sleeping() {
                    sleeping += 1;
                }
                let v = Vec3::from_array(b.linear_vel);
                ke += 0.5 * v.length_squared() / b.inv_mass;
                if v.length() > vmax {
                    vmax = v.length();
                    fastest = i;
                }
            }
            let gaps = world.joints().iter().filter(|j| !j.is_broken()).count();
            if std::env::var("JOINT_DUMP").is_ok() {
                let mut js: Vec<(usize, engine_schema::Joint)> =
                    world.joints().iter().copied().enumerate().collect();
                js.sort_by(|a, b| {
                    let m = |j: &engine_schema::Joint| {
                        j.lambda.iter().fold(0.0f32, |m, x| m.max(x.abs()))
                    };
                    m(&b.1).partial_cmp(&m(&a.1)).unwrap()
                });
                for (i, j) in js.iter().take(3) {
                    println!(
                        "  joint {i} ({}-{}) preset {} motion {:03x} lambda {:?} c0 {:?}",
                        j.body_a,
                        j.body_b,
                        j.preset(),
                        j.motion,
                        j.lambda,
                        j.c0
                    );
                }
            }
            println!(
                "step {step:5} KE {ke:10.3} max|v| {vmax:7.3} (body {fastest}) sleeping {sleeping}/{dynamic} gpu {:.2} ms joints {gaps}",
                world.last_timings().total_gpu
            );
        }
    }
    Ok(())
}

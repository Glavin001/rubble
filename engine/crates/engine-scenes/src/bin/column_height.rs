use engine_exec::PhysicsBackend;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "Box Columns (100k)".into());
    let steps: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let (mut world, info) = engine_scenes::build_world(&name).unwrap();
    println!("{name} iters={} steps={steps}", info.iterations);
    for s in 0..steps {
        world.step().unwrap();
        if s < 5 || s % 10 == 9 || s + 1 == steps {
            let bodies = world.download_bodies();
            let mut ymin = f32::INFINITY;
            let mut ymax = f32::NEG_INFINITY;
            let mut ke = 0.0f32;
            let mut n = 0u32;
            for b in &bodies {
                if !b.is_dynamic() { continue; }
                n += 1;
                ymin = ymin.min(b.position[1]);
                ymax = ymax.max(b.position[1]);
                let m = if b.inv_mass > 1e-12 { 1.0 / b.inv_mass } else { 0.0 };
                ke += 0.5 * m * glam::Vec3::from_array(b.linear_vel).length_squared();
            }
            let c = world.last_counters();
            println!(
                "  t={s}: y=[{ymin:.2},{ymax:.2}] ke={ke:.1} awake={} contacts={} colors={} ov=0x{:x} n={n}",
                c.awake_count, c.contact_count, c.color_count, c.overflow_flags
            );
        }
    }
}

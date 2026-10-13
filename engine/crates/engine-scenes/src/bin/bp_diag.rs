use engine_exec::PhysicsBackend;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "Brick Ring (9k)".into());
    let steps: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let (mut world, _) = engine_scenes::build_world(&name).expect("build");
    for _ in 0..steps {
        world.step().expect("step");
    }
    let c = world.last_counters();
    let bodies = world.download_bodies();
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    let mut ke = 0.0f32;
    let mut below = 0u32;
    let mut n = 0u32;
    for b in &bodies {
        if !b.is_dynamic() { continue; }
        n += 1;
        ymin = ymin.min(b.position[1]);
        ymax = ymax.max(b.position[1]);
        let m = if b.inv_mass > 1e-12 { 1.0 / b.inv_mass } else { 0.0 };
        ke += 0.5 * m * glam::Vec3::from_array(b.linear_vel).length_squared();
        if b.position[1] < 1.0 { below += 1; }
    }
    let ball = &bodies[n as usize]; // ground + n dynamics; last dynamic
    println!(
        "steps={steps} awake={} contacts={} cand={} y=[{ymin:.2},{ymax:.2}] below1m={below}/{n} ke={ke:.0} ball_z={:.2} ball_vz={:.2}",
        c.awake_count, c.contact_count, c.candidate_count, ball.position[2], ball.linear_vel[2]
    );
}

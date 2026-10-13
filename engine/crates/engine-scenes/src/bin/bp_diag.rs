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
    let mut ball_i = 0usize;
    let mut best_v = 0.0f32;
    let mut flying = 0u32;
    let mut below = 0u32;
    let mut n = 0u32;
    let mut brick_ke = 0.0f32;
    for (i, b) in bodies.iter().enumerate() {
        if !b.is_dynamic() { continue; }
        n += 1;
        let v = glam::Vec3::from_array(b.linear_vel).length();
        if v > best_v {
            best_v = v;
            ball_i = i;
        }
        if b.position[1] < 1.0 { below += 1; }
        if v > 1.0 { flying += 1; }
        let m = if b.inv_mass > 1e-12 { 1.0 / b.inv_mass } else { 0.0 };
        // skip heaviest (ball) later
        brick_ke += 0.5 * m * v * v;
    }
    let ball = &bodies[ball_i];
    let bm = if ball.inv_mass > 1e-12 { 1.0 / ball.inv_mass } else { 0.0 };
    let ball_ke = 0.5 * bm * glam::Vec3::from_array(ball.linear_vel).length_squared();
    println!(
        "steps={steps} awake={} contacts={} manifolds={} cand={} below1m={below}/{n} flying(v>1)={flying} ball_z={:.2} vz={:.2} ball_ke={ball_ke:.0} total_ke={:.0}",
        c.awake_count, c.contact_count, c.manifold_count, c.candidate_count,
        ball.position[2], ball.linear_vel[2], brick_ke
    );
}

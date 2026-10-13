use engine_exec::PhysicsBackend;

fn main() {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Brick Ring (9k)".into());
    let steps: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let (mut world, _) = engine_scenes::build_world(&name).expect("build");
    let bodies0 = world.download_bodies();
    // Smash demos add the ball last among dynamics.
    let ball_body = bodies0
        .iter()
        .enumerate()
        .filter(|(_, b)| b.is_dynamic())
        .max_by(|a, b| {
            a.1.linear_vel[2]
                .abs()
                .partial_cmp(&b.1.linear_vel[2].abs())
                .unwrap()
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    let b0 = &bodies0[ball_body as usize];
    println!(
        "ball body={ball_body} mass={:.1} pos={:?} vel={:?}",
        if b0.inv_mass > 1e-12 {
            1.0 / b0.inv_mass
        } else {
            0.0
        },
        b0.position,
        b0.linear_vel
    );
    for step in 1..=steps {
        world.step().expect("step");
        if step % 10 == 0 || step == steps {
            let c = world.last_counters();
            let bodies = world.download_bodies();
            let b = &bodies[ball_body as usize];
            let m = if b.inv_mass > 1e-12 {
                1.0 / b.inv_mass
            } else {
                0.0
            };
            let ke = 0.5 * m * glam::Vec3::from_array(b.linear_vel).length_squared();
            let mut below = 0u32;
            let mut n = 0u32;
            for bb in &bodies {
                if !bb.is_dynamic() {
                    continue;
                }
                n += 1;
                if bb.position[1] < 1.0 {
                    below += 1;
                }
            }
            println!(
                "t={step:3} awake={} contacts={} manifolds={} cand={} below1m={below}/{n} ball_xyz=({:.2},{:.2},{:.2}) v=({:.2},{:.2},{:.2}) ke={ke:.0}",
                c.awake_count,
                c.contact_count,
                c.manifold_count,
                c.candidate_count,
                b.position[0],
                b.position[1],
                b.position[2],
                b.linear_vel[0],
                b.linear_vel[1],
                b.linear_vel[2],
            );
        }
    }
}

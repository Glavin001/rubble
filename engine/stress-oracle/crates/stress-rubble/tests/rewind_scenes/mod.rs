//! The scenes of `findings/rewind-scaling.md`, shared by `examples/rewind_scaling.rs`
//! and `tests/rewind_scaling.rs`.
// Each user takes what it needs.
#![allow(dead_code)]

use stress_ref::scene::Scene;

/// `n` free concrete walls (each 1.0 x 0.2 x 1.1 m, 10 x 2 x 11 chunks of 0.1 m, 528 kg,
/// as `stress_bench`'s `wall_ram`) one behind another along +y with `gap` between them,
/// and a 200 kg steel ram at `speed` m/s along +y, 5 cm short of the first. No gravity,
/// so nothing but the ram loads the walls. In A12's real-time configuration.
pub fn wall_row(n: usize, speed: f64, gap: f64, seconds: f64) -> Scene {
    use stress_ref::builders::*;
    use stress_ref::math::Vec3;
    use stress_ref::scene::*;
    let mut s = new_scene(&format!("wall_row_{n}_v{speed}"), "a steel ram through a row of free concrete walls");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    for k in 0..n {
        let y = k as f64 * (0.2 + gap);
        let chunks = grid(Vec3::new(-0.5, y, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        s.bodies.push(body(&format!("wall{k}"), chunks, bonds));
    }
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.1, 0.1, 0.1] },
        mass: 200.0,
        position: [0.0, -0.15, 0.5],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.methods = Methods::established();
    s.sim.methods.true_step = true;
    s.sim.courant_safety = 0.35;
    s.sim.stress_courant_safety = Some(0.5);
    s.sim.solve_mode = SolveMode::Adaptive;
    s.sim.duration = seconds;
    s
}

/// Where routing misses: the ram strikes a resting, unbreakable steel bar (one chunk, no
/// bonds: 0.2 x 1.0 x 0.2 m, 314 kg) 1 cm in front of it, whose far end is 1 cm short of
/// one free wall. The ram's swept bounds over a frame (0.67 m at 40 m/s) do not reach
/// the wall, the bar is at rest at the frame start, and a bond-less body never makes the
/// rigid assumption fail, so no island is predicted; within the frame the bar carries the
/// ram's momentum into the wall under the engine's rigid contact.
pub fn striker(speed: f64, seconds: f64) -> Scene {
    use stress_ref::builders::*;
    use stress_ref::math::Vec3;
    let mut s = wall_row(0, speed, 0.0, seconds);
    s.name = format!("striker_v{speed}");
    s.description = "a steel ram strikes a resting steel bar against a free concrete wall".into();
    let chunks = grid(Vec3::new(-0.5, 0.97, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall0", chunks, bonds));
    let bar = chunk(Vec3::new(0.0, 0.46, 0.5), Vec3::new(0.1, 0.5, 0.1), "steel");
    s.bodies.push(body("bar", vec![bar], Vec::new()));
    s
}

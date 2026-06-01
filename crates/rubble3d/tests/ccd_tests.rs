//! Continuous-collision / tunneling characterization.
//!
//! Most discrete-time engines have no continuous collision detection (CCD): a
//! body moving faster than ~(feature size)/dt can skip across a thin obstacle in
//! one step. This test measures *where* that happens for Rubble, and asserts the
//! correctness floor — a slow body must still be stopped by a solid wall (if even
//! a slow body tunnels, that's a narrowphase/contact failure, not merely missing
//! CCD).

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

/// Fire a sphere at a thin static wall at `speed` (+X) and return its final x.
/// The wall spans x∈[-0.1, 0.1]; "stopped" means it ends on the near side.
fn final_x_after_wall(speed: f32) -> Option<f32> {
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        max_bodies: 8,
        ..Default::default()
    })
    .ok()?;
    // Static wall: thin in X, large in Y/Z.
    world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        friction: 0.3,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(0.1, 3.0, 3.0),
        },
        ..Default::default()
    });
    // Dynamic sphere approaching from -X.
    let s = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-1.5, 0.0, 0.0),
        linear_velocity: Vec3::new(speed, 0.0, 0.0),
        mass: 1.0,
        friction: 0.3,
        shape: ShapeDesc::Sphere { radius: 0.2 },
        ..Default::default()
    });
    for _ in 0..120 {
        world.step();
    }
    Some(world.get_position(s)?.x)
}

#[test]
fn fast_body_tunneling_characterization() {
    let speeds = [5.0f32, 15.0, 40.0, 80.0];
    let mut slow_caught: Option<bool> = None;
    let mut threshold: Option<f32> = None;

    for &v in &speeds {
        let Some(x) = final_x_after_wall(v) else {
            eprintln!("SKIP: no GPU adapter");
            return;
        };
        // Wall right face is at +0.1; ending well past it means it passed through.
        let tunneled = x > 0.5;
        println!(
            "speed {v:>5} m/s: final_x = {x:>8.3}  ->  {}",
            if tunneled { "TUNNELED" } else { "stopped" }
        );
        if (v - 5.0).abs() < 1e-6 {
            slow_caught = Some(!tunneled);
        }
        if tunneled && threshold.is_none() {
            threshold = Some(v);
        }
    }

    match threshold {
        Some(v) => println!(
            "=> tunneling begins around {v} m/s ({:.3} m/step at dt=1/120) — no CCD",
            v / 120.0
        ),
        None => println!("=> no tunneling up to {} m/s", speeds.last().unwrap()),
    }

    // Correctness floor: a slow body must be stopped by a solid wall.
    assert_eq!(
        slow_caught,
        Some(true),
        "a slow (5 m/s) sphere passed through a solid wall — narrowphase/contact failure"
    );
}

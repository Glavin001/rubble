//! GPU performance and stress tests for rubble3d.
//!
//! These tests verify the engine handles large body counts, sustained simulation,
//! buffer overflow recovery, and extreme scenarios without diverging or crashing.

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

fn gpu_world(config: SimConfig) -> World {
    World::new(config).expect("GPU required for performance tests")
}

fn step_n(world: &mut World, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

// ---------------------------------------------------------------------------
// Sustained simulation stability
// ---------------------------------------------------------------------------

#[test]
fn sustained_1000_steps_single_body() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 100.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 1000);

    let pos = world.get_position(h).unwrap();
    let vel = world.get_velocity(h).unwrap();
    assert!(pos.is_finite(), "Position diverged after 1000 steps: {pos}");
    assert!(vel.is_finite(), "Velocity diverged after 1000 steps: {vel}");
}

#[test]
fn sustained_500_steps_with_contacts() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 8,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 1.0, 20.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..10 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(
                (i % 5) as f32 * 3.0 - 6.0,
                3.0 + (i / 5) as f32 * 3.0,
                0.0,
            ),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 500);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.is_finite(),
            "Body {i} diverged after 500 steps with contacts: {pos}"
        );
        assert!(
            pos.y > -10.0,
            "Body {i} fell far through floor after 500 steps: y={}",
            pos.y
        );
    }
}

// ---------------------------------------------------------------------------
// Scaling tests
// ---------------------------------------------------------------------------

#[test]
fn scale_16_bodies() {
    run_body_count_test(16);
}

#[test]
fn scale_32_bodies() {
    run_body_count_test(32);
}

#[test]
fn scale_64_bodies() {
    run_body_count_test(64);
}

#[test]
fn scale_128_bodies() {
    run_body_count_test(128);
}

#[test]
fn scale_256_bodies() {
    run_body_count_test(256);
}

fn run_body_count_test(count: usize) {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: count + 16,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -2.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(100.0, 1.0, 100.0),
        },
        ..Default::default()
    });

    let cols = (count as f32).sqrt().ceil() as usize;
    let mut handles = Vec::new();
    for i in 0..count {
        let x = (i % cols) as f32 * 3.0 - (cols as f32 * 1.5);
        let z = (i / cols) as f32 * 3.0 - (cols as f32 * 1.5);
        let shape = match i % 3 {
            0 => ShapeDesc::Sphere { radius: 0.5 },
            1 => ShapeDesc::Box {
                half_extents: Vec3::splat(0.4),
            },
            _ => ShapeDesc::Capsule {
                half_height: 0.4,
                radius: 0.3,
            },
        };
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(x, 5.0 + (i as f32) * 0.05, z),
            mass: 1.0,
            shape,
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 60);

    let mut finite = 0;
    for &h in &handles {
        let pos = world.get_position(h).unwrap();
        if pos.is_finite() {
            finite += 1;
        }
    }
    assert_eq!(
        finite, count,
        "All {count} bodies should have finite positions, got {finite}"
    );
}

// ---------------------------------------------------------------------------
// Dynamic body addition/removal under load
// ---------------------------------------------------------------------------

#[test]
fn dynamic_add_remove_during_simulation() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 1.0, 20.0),
        },
        ..Default::default()
    });

    let mut active_handles = Vec::new();

    // Add bodies progressively
    for frame in 0..180 {
        if frame < 60 && frame % 5 == 0 {
            let h = world.add_body(&RigidBodyDesc {
                position: Vec3::new(
                    (active_handles.len() as f32) * 3.0 - 15.0,
                    5.0,
                    0.0,
                ),
                mass: 1.0,
                shape: ShapeDesc::Sphere { radius: 0.5 },
                ..Default::default()
            });
            active_handles.push(h);
        }

        // Remove oldest bodies later in simulation
        if frame == 90 && !active_handles.is_empty() {
            let h = active_handles.remove(0);
            world.remove_body(h);
        }
        if frame == 120 && !active_handles.is_empty() {
            let h = active_handles.remove(0);
            world.remove_body(h);
        }

        world.step();
    }

    for &h in &active_handles {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.is_finite(),
            "Dynamic add/remove body diverged: {pos}"
        );
    }
}

// ---------------------------------------------------------------------------
// Buffer overflow recovery
// ---------------------------------------------------------------------------

#[test]
fn many_contacts_trigger_overflow_recovery() {
    // Create many overlapping bodies to trigger contact buffer overflow.
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 128,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    // Pack spheres tightly to generate many contacts
    let mut handles = Vec::new();
    for x in 0..8 {
        for z in 0..8 {
            let h = world.add_body(&RigidBodyDesc {
                position: Vec3::new(
                    x as f32 * 1.5 - 5.0,
                    2.0,
                    z as f32 * 1.5 - 5.0,
                ),
                mass: 1.0,
                shape: ShapeDesc::Sphere { radius: 0.8 },
                ..Default::default()
            });
            handles.push(h);
        }
    }

    // Should not crash even with many contacts
    step_n(&mut world, 30);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.is_finite(),
            "Overflow recovery: body {i} diverged: {pos}"
        );
    }
}

// ---------------------------------------------------------------------------
// High velocity stability
// ---------------------------------------------------------------------------

#[test]
fn high_velocity_impact_on_wall() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let _wall = world.add_body(&RigidBodyDesc {
        position: Vec3::new(10.0, 0.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(1.0, 10.0, 10.0),
        },
        ..Default::default()
    });

    let projectile = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        linear_velocity: Vec3::new(100.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pos = world.get_position(projectile).unwrap();
    let vel = world.get_velocity(projectile).unwrap();
    assert!(pos.is_finite(), "High velocity impact: pos diverged: {pos}");
    assert!(vel.is_finite(), "High velocity impact: vel diverged: {vel}");
}

#[test]
fn tunneling_prevention_fast_sphere() {
    // Fast sphere should still collide (not tunnel through) a thick wall.
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _wall = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(2.0, 10.0, 10.0),
        },
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-2.0, 0.0, 0.0),
        linear_velocity: Vec3::new(20.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(sphere).unwrap();
    assert!(pos.is_finite(), "Fast sphere diverged: {pos}");
    // Should not tunnel completely through 4-unit thick wall
    assert!(
        pos.x < 10.0,
        "Sphere may have tunneled: x={}",
        pos.x
    );
}

// ---------------------------------------------------------------------------
// Broadphase edge cases
// ---------------------------------------------------------------------------

#[test]
fn bodies_far_apart_no_interaction() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-1000.0, 0.0, 0.0),
        linear_velocity: Vec3::new(1.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(1000.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-1.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    // Velocities should be preserved (no collision)
    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    assert!(
        (va.x - 1.0).abs() < 0.01,
        "Far-apart body A velocity changed: vx={}",
        va.x
    );
    assert!(
        (vb.x - (-1.0)).abs() < 0.01,
        "Far-apart body B velocity changed: vx={}",
        vb.x
    );
}

#[test]
fn many_static_bodies_no_crash() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    // 50 static boxes as terrain
    for i in 0..50 {
        let x = (i % 10) as f32 * 4.0 - 20.0;
        let z = (i / 10) as f32 * 4.0 - 10.0;
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(x, -1.0, z),
            mass: 0.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(2.0, 0.5, 2.0),
            },
            ..Default::default()
        });
    }

    // One dynamic sphere
    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pos = world.get_position(sphere).unwrap();
    assert!(
        pos.is_finite(),
        "Sphere with many statics diverged: {pos}"
    );
}

// ---------------------------------------------------------------------------
// Warm-starting over many frames
// ---------------------------------------------------------------------------

#[test]
fn warm_starting_long_simulation() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 8,
        warmstart_decay: 0.95,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 1.0, 20.0),
        },
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 600);

    let pos = world.get_position(sphere).unwrap();
    let vel = world.get_velocity(sphere).unwrap();
    assert!(pos.is_finite(), "Long warmstart simulation: pos={pos}");
    assert!(vel.is_finite(), "Long warmstart simulation: vel={vel}");
    assert!(
        pos.y > -5.0,
        "Sphere fell too far after long simulation: y={}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Edge case: all-static world
// ---------------------------------------------------------------------------

#[test]
fn all_static_bodies_no_crash() {
    let mut world = gpu_world(SimConfig::default());

    for i in 0..10 {
        world.add_body(&RigidBodyDesc {
            position: Vec3::new(i as f32 * 3.0, 0.0, 0.0),
            mass: 0.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });
    }

    step_n(&mut world, 60);
    assert_eq!(world.body_count(), 10);
}

// ---------------------------------------------------------------------------
// Gravity directions
// ---------------------------------------------------------------------------

#[test]
fn negative_x_gravity() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(-9.81, 0.0, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(10.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    assert!(
        pos.x < 10.0,
        "Should fall in -X direction: x={}",
        pos.x
    );
    assert!(
        pos.y.abs() < 0.01,
        "No Y drift with X gravity: y={}",
        pos.y
    );
}

#[test]
fn diagonal_gravity() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(-5.0, -5.0, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    assert!(pos.x < 0.0, "Should fall in -X: x={}", pos.x);
    assert!(pos.y < 0.0, "Should fall in -Y: y={}", pos.y);
    assert!(pos.z.abs() < 0.01, "No Z drift: z={}", pos.z);
}

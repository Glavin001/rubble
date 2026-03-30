//! GPU integration tests for rubble3d.
//!
//! These tests verify that the GPU compute pipeline (AVBD solver) produces
//! physically reasonable results. Each test creates a World with GPU enabled,
//! runs the simulation, and checks invariants.

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

/// Helper to create a GPU-backed world, panicking if no GPU is available.
fn gpu_world(config: SimConfig) -> World {
    World::new(config).expect(
        "FATAL: No GPU adapter found. Install mesa-vulkan-drivers for lavapipe software Vulkan.",
    )
}

#[test]
fn gpu_world_creation() {
    let world = gpu_world(SimConfig::default());
    assert_eq!(world.body_count(), 0);
}

#[test]
fn gpu_single_body_free_fall() {
    let dt = 1.0 / 60.0;
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let steps = 60;
    for _ in 0..steps {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    let t = steps as f32 * dt;
    let expected_y = 10.0 - 0.5 * 9.81 * t * t;
    assert!(pos.y < 10.0, "GPU: Body should have fallen: y = {}", pos.y);
    assert!(
        (pos.y - expected_y).abs() < 2.0,
        "GPU: Free fall y={}, expected ~{}, error={}",
        pos.y,
        expected_y,
        (pos.y - expected_y).abs()
    );
    assert!(pos.x.abs() < 0.1, "GPU: X drift = {}", pos.x);
    assert!(pos.z.abs() < 0.1, "GPU: Z drift = {}", pos.z);
}

#[test]
fn gpu_static_body_does_not_move() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 5.0, 5.0),
        mass: 0.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos - Vec3::new(5.0, 5.0, 5.0)).length() < 0.01,
        "GPU: Static body moved to {:?}",
        pos
    );
}

#[test]
fn gpu_two_sphere_collision() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        max_bodies: 256,
    });

    let h1 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-2.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let h2 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(2.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    for _ in 0..120 {
        world.step();
    }

    let p1 = world.get_position(h1).unwrap();
    let p2 = world.get_position(h2).unwrap();
    let dist = (p2 - p1).length();

    assert!(
        dist >= 1.5,
        "GPU: Spheres should not overlap: distance = {}",
        dist
    );
}

#[test]
fn gpu_box_free_fall() {
    let dt = 1.0 / 60.0;
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(pos.y < 10.0, "GPU: Box should have fallen: y = {}", pos.y);
    assert!(
        pos.y > -20.0,
        "GPU: Box fell too far (not finite?): y = {}",
        pos.y
    );
}

#[test]
fn gpu_multiple_bodies_no_crash() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let mut handles = Vec::new();
    for i in 0..10 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(i as f32 * 3.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        handles.push(h);
    }

    for _ in 0..60 {
        world.step();
    }

    for h in &handles {
        let pos = world.get_position(*h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "GPU: Non-finite position: {:?}",
            pos
        );
    }
}

#[test]
fn gpu_sphere_box_collision() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        max_bodies: 256,
    });

    let _box_h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        mass: 0.0, // static
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        },
        ..Default::default()
    });

    let sphere_h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..120 {
        world.step();
    }

    let pos = world.get_position(sphere_h).unwrap();
    assert!(
        pos.x.is_finite(),
        "GPU: Sphere position is not finite: {:?}",
        pos
    );
}

#[test]
fn gpu_zero_gravity_no_drift() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 5.0, 5.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos - Vec3::new(5.0, 5.0, 5.0)).length() < 0.01,
        "GPU: Body drifted without gravity or velocity: {:?}",
        pos
    );
}

#[test]
fn gpu_velocity_preserved_without_collision() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        linear_velocity: Vec3::new(3.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    let vel = world.get_velocity(h).unwrap();

    assert!(
        (pos.x - 3.0).abs() < 1.0,
        "GPU: Expected x~3.0, got {}",
        pos.x
    );
    assert!(
        (vel.x - 3.0).abs() < 1.0,
        "GPU: Expected vx~3.0, got {}",
        vel.x
    );
}

#[test]
fn gpu_add_remove_body_stability() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h1 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..30 {
        world.step();
    }

    let h2 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    world.remove_body(h1);

    for _ in 0..30 {
        world.step();
    }

    let pos2 = world.get_position(h2).unwrap();
    assert!(pos2.y < 5.0, "GPU: h2 should have fallen: y = {}", pos2.y);
    assert!(pos2.y.is_finite(), "GPU: h2 position not finite");
}

//! GPU integration tests for rubble2d.
//!
//! These tests verify that the GPU compute pipeline (2D AVBD solver) produces
//! physically reasonable results. Each test creates a World2D with GPU enabled,
//! runs the simulation, and checks invariants.

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D, World2D};

/// Helper to create a GPU-backed 2D world.
fn gpu_world(config: SimConfig2D) -> World2D {
    World2D::new(config).expect(
        "FATAL: No GPU adapter found. Install mesa-vulkan-drivers for lavapipe software Vulkan.",
    )
}

#[test]
fn gpu_2d_world_creation() {
    let world = gpu_world(SimConfig2D::default());
    // GPU is always active now
    assert_eq!(world.body_count(), 0);
}

#[test]
fn gpu_2d_free_fall() {
    let dt = 1.0 / 60.0;
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    let steps = 60;
    for _ in 0..steps {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    let t = steps as f32 * dt;
    let expected_y = 10.0 - 0.5 * 9.81 * t * t;
    assert!(
        pos.y < 10.0,
        "GPU 2D: Body should have fallen: y = {}",
        pos.y
    );
    assert!(
        (pos.y - expected_y).abs() < 2.0,
        "GPU 2D: Free fall y={}, expected ~{}, error={}",
        pos.y,
        expected_y,
        (pos.y - expected_y).abs()
    );
    assert!(pos.x.abs() < 0.1, "GPU 2D: X drift = {}", pos.x);
}

#[test]
fn gpu_2d_static_body_immovable() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 5.0,
        mass: 0.0, // static
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos - Vec2::new(5.0, 5.0)).length() < 0.01,
        "GPU 2D: Static body moved to {:?}",
        pos
    );
}

#[test]
fn gpu_2d_two_circle_collision() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        max_bodies: 256,
    });

    let h1 = world.add_body(&RigidBodyDesc2D {
        x: -2.0,
        y: 0.0,
        vx: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    let h2 = world.add_body(&RigidBodyDesc2D {
        x: 2.0,
        y: 0.0,
        vx: -5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
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
        "GPU 2D: Circles should not overlap: distance = {}",
        dist
    );
}

#[test]
fn gpu_2d_zero_gravity_no_drift() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 7.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos - Vec2::new(3.0, 7.0)).length() < 0.01,
        "GPU 2D: Body drifted without forces: {:?}",
        pos
    );
}

#[test]
fn gpu_2d_velocity_preserved() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        vx: 4.0,
        vy: 0.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos.x - 4.0).abs() < 1.0,
        "GPU 2D: Expected x~4.0, got {}",
        pos.x
    );
}

#[test]
fn gpu_2d_multiple_bodies_stability() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let mut handles = Vec::new();
    for i in 0..8 {
        let h = world.add_body(&RigidBodyDesc2D {
            x: i as f32 * 3.0,
            y: 5.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
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
            pos.x.is_finite() && pos.y.is_finite(),
            "GPU 2D: Non-finite position: {:?}",
            pos
        );
    }
}

#[test]
fn gpu_2d_rect_free_fall() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        mass: 1.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(0.5, 0.5),
        },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(
        pos.y < 10.0,
        "GPU 2D: Rect should have fallen: y = {}",
        pos.y
    );
    assert!(pos.y > -20.0, "GPU 2D: Rect fell too far: y = {}", pos.y);
}

#[test]
fn gpu_2d_add_remove_body() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    });

    let h1 = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..30 {
        world.step();
    }

    let h2 = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    world.remove_body(h1);

    for _ in 0..30 {
        world.step();
    }

    let pos2 = world.get_position(h2).unwrap();
    assert!(
        pos2.y < 5.0,
        "GPU 2D: h2 should have fallen: y = {}",
        pos2.y
    );
    assert!(pos2.y.is_finite(), "GPU 2D: h2 position not finite");
}

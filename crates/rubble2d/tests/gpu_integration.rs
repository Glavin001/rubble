//! GPU integration tests for rubble2d.
//!
//! These tests verify that the GPU compute pipeline (2D AVBD solver) produces
//! physically reasonable results. Each test creates a World2D with GPU enabled,
//! runs the simulation, and checks invariants.

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D, World2D};
use std::f32::consts::FRAC_PI_6;

macro_rules! gpu_world {
    ($config:expr) => {
        match World2D::new($config) {
            Ok(w) => w,
            Err(_) => { eprintln!("SKIP: No GPU adapter found"); return; }
        }
    };
}

#[test]
fn gpu_2d_world_creation() {
    let world = gpu_world!(SimConfig2D::default());
    assert_eq!(world.body_count(), 0);
}

#[test]
fn gpu_2d_free_fall() {
    let dt = 1.0 / 60.0;
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
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
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
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
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        max_bodies: 256,
        ..Default::default()
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
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
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
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
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
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
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

    let initial_y = 5.0_f32;
    for h in &handles {
        let pos = world.get_position(*h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite(),
            "GPU 2D: Non-finite position: {:?}",
            pos
        );
        assert!(
            pos.y < initial_y,
            "GPU 2D: Dynamic body should have fallen due to gravity: y = {} (started at {})",
            pos.y,
            initial_y
        );
    }
}

#[test]
fn gpu_2d_rect_free_fall() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
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
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
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

#[test]
fn gpu_2d_empty_world_step() {
    let mut world = gpu_world!(SimConfig2D::default());
    assert_eq!(world.body_count(), 0);
    // Stepping an empty world should not crash or panic.
    for _ in 0..10 {
        world.step();
    }
    assert_eq!(world.body_count(), 0);
}

#[test]
fn gpu_2d_get_angle() {
    let mut world = gpu_world!(SimConfig2D::default());
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        angle: FRAC_PI_6,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 0.5),
        },
        ..Default::default()
    });

    let angle = world.get_angle(h).unwrap();
    assert!(
        (angle - FRAC_PI_6).abs() < 0.01,
        "Initial angle should be PI/6, got {angle}"
    );
}

#[test]
fn gpu_2d_high_velocity_stability() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        vx: 500.0,
        vy: -300.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..120 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    let vel = world.get_velocity(h).unwrap();
    assert!(
        pos.x.is_finite() && pos.y.is_finite(),
        "High-velocity body position diverged: {pos}"
    );
    assert!(
        vel.x.is_finite() && vel.y.is_finite(),
        "High-velocity body velocity diverged: {vel}"
    );
    let pos_magnitude = pos.length();
    assert!(
        pos_magnitude > 10.0,
        "High-velocity body should have moved significantly from origin: magnitude = {}",
        pos_magnitude
    );
}

#[test]
fn gpu_2d_convex_polygon_free_fall() {
    let dt = 1.0 / 60.0;
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
    });

    // Pentagon
    let verts: Vec<Vec2> = (0..5)
        .map(|i| {
            let angle = i as f32 * std::f32::consts::TAU / 5.0;
            Vec2::new(angle.cos(), angle.sin())
        })
        .collect();

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        mass: 1.0,
        shape: ShapeDesc2D::ConvexPolygon { vertices: verts },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(pos.y < 10.0, "Polygon should have fallen: y = {}", pos.y);
    assert!(pos.y > -20.0, "Polygon fell too far: y = {}", pos.y);
    assert!(pos.x.abs() < 0.5, "X drift: {}", pos.x);
}

#[test]
fn gpu_2d_poly_poly_collision() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        max_bodies: 256,
        ..Default::default()
    });

    // Two squares as convex polygons
    let square_verts = vec![
        Vec2::new(-0.5, -0.5),
        Vec2::new(0.5, -0.5),
        Vec2::new(0.5, 0.5),
        Vec2::new(-0.5, 0.5),
    ];

    let h1 = world.add_body(&RigidBodyDesc2D {
        x: -2.0,
        y: 0.0,
        vx: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::ConvexPolygon {
            vertices: square_verts.clone(),
        },
        ..Default::default()
    });

    let h2 = world.add_body(&RigidBodyDesc2D {
        x: 2.0,
        y: 0.0,
        vx: -5.0,
        mass: 1.0,
        shape: ShapeDesc2D::ConvexPolygon {
            vertices: square_verts,
        },
        ..Default::default()
    });

    for _ in 0..120 {
        world.step();
    }

    let p1 = world.get_position(h1).unwrap();
    let p2 = world.get_position(h2).unwrap();
    assert!(
        p1.is_finite() && p2.is_finite(),
        "Polygon positions should be finite: {p1}, {p2}"
    );
    let dist = (p2 - p1).length();
    assert!(dist > 0.3, "Polygons should have separated: dist = {dist}");
    let init_p1 = Vec2::new(-2.0, 0.0);
    let init_p2 = Vec2::new(2.0, 0.0);
    assert!(
        (p1 - init_p1).length() > 0.1,
        "Polygon 1 should have moved from initial position: {p1}"
    );
    assert!(
        (p2 - init_p2).length() > 0.1,
        "Polygon 2 should have moved from initial position: {p2}"
    );
}

#[test]
fn gpu_2d_circle_poly_collision() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        max_bodies: 256,
        ..Default::default()
    });

    let square_verts = vec![
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 1.0),
    ];

    // Static polygon
    let _poly_h = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        mass: 0.0,
        shape: ShapeDesc2D::ConvexPolygon {
            vertices: square_verts,
        },
        ..Default::default()
    });

    // Moving circle
    let circle_h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        vx: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..120 {
        world.step();
    }

    let pos = world.get_position(circle_h).unwrap();
    assert!(
        pos.x.is_finite() && pos.y.is_finite(),
        "Circle position should be finite: {pos}"
    );
}

#[test]
fn gpu_2d_rect_poly_collision() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        max_bodies: 256,
        ..Default::default()
    });

    let triangle_verts = vec![
        Vec2::new(0.0, 1.0),
        Vec2::new(-0.866, -0.5),
        Vec2::new(0.866, -0.5),
    ];

    // Static rect
    let _rect_h = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 1.0),
        },
        ..Default::default()
    });

    // Moving polygon
    let poly_h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        vx: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::ConvexPolygon {
            vertices: triangle_verts,
        },
        ..Default::default()
    });

    for _ in 0..120 {
        world.step();
    }

    let pos = world.get_position(poly_h).unwrap();
    assert!(
        pos.x.is_finite() && pos.y.is_finite(),
        "Polygon position should be finite: {pos}"
    );
}

#[test]
fn gpu_2d_convex_polygon_static_no_move() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
    });

    let verts = vec![
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 1.0),
    ];

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 5.0,
        mass: 0.0,
        shape: ShapeDesc2D::ConvexPolygon { vertices: verts },
        ..Default::default()
    });

    for _ in 0..60 {
        world.step();
    }

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos - Vec2::new(0.0, 5.0)).length() < 0.01,
        "Static polygon moved: {pos}"
    );
}

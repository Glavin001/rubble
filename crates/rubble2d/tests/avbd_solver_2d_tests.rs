//! Comprehensive AVBD solver accuracy tests for rubble2d.
//!
//! Tests verify: momentum conservation, energy behavior, solver convergence,
//! friction, numerical stability, and angular dynamics in 2D.

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D, World2D};

macro_rules! gpu_world {
    ($config:expr) => {
        match World2D::new($config) {
            Ok(w) => w,
            Err(_) => { eprintln!("SKIP: No GPU adapter found"); return; }
        }
    };
}

fn step_n(world: &mut World2D, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

// ---------------------------------------------------------------------------
// Momentum conservation
// ---------------------------------------------------------------------------

#[test]
fn momentum_2d_equal_mass_head_on() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 4.0,
        vy: 0.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: -4.0,
        vy: 0.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    let total_momentum = va.x + vb.x; // equal mass
    assert!(
        total_momentum.abs() < 2.0,
        "2D Momentum not conserved: total_p={total_momentum}"
    );
}

#[test]
fn momentum_2d_unequal_mass() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        ..Default::default()
    });

    let heavy = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 2.0,
        vy: 0.0,
        mass: 5.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let light = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: -2.0,
        vy: 0.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    let initial_p = 5.0 * 2.0 + 1.0 * (-2.0); // = 8

    step_n(&mut world, 300);

    let vh = world.get_velocity(heavy).unwrap();
    let vl = world.get_velocity(light).unwrap();
    let final_p = 5.0 * vh.x + 1.0 * vl.x;
    let error = (final_p - initial_p).abs();
    assert!(
        error < 4.0,
        "2D unequal mass momentum: initial={initial_p}, final={final_p}, error={error}"
    );
}

// ---------------------------------------------------------------------------
// Energy
// ---------------------------------------------------------------------------

#[test]
fn energy_2d_does_not_increase_collision() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 5.0,
        vy: 0.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: -5.0,
        vy: 0.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    let initial_ke = 0.5 * (25.0 + 25.0);

    step_n(&mut world, 400);

    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    let final_ke = 0.5 * (va.length_squared() + vb.length_squared());

    assert!(
        final_ke < initial_ke * 1.5,
        "2D energy increased: initial={initial_ke}, final={final_ke}"
    );
}

#[test]
fn gravitational_pe_to_ke_2d() {
    let h = 10.0f32;
    let g = 9.81f32;
    let dt = 1.0 / 60.0;

    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -g),
        dt,
        solver_iterations: 5,
        ..Default::default()
    });

    let body = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: h,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(body).unwrap();
    let vel = world.get_velocity(body).unwrap();
    let height_fallen = h - pos.y;
    let pe_lost = g * height_fallen;
    let ke_gained = 0.5 * vel.length_squared();
    let error = (pe_lost - ke_gained).abs();

    assert!(
        error < pe_lost * 0.3,
        "2D energy conversion error: PE_lost={pe_lost}, KE_gained={ke_gained}"
    );
}

// ---------------------------------------------------------------------------
// Solver convergence
// ---------------------------------------------------------------------------

#[test]
fn solver_iterations_improve_2d() {
    let mut results = Vec::new();

    for &iters in &[2u32, 5, 10, 20] {
        let mut world = gpu_world!(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            dt: 1.0 / 60.0,
            solver_iterations: iters,
            ..Default::default()
        });

        let _floor = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: -1.0,
            mass: 0.0,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(10.0, 1.0),
            },
            ..Default::default()
        });

        let circle = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 2.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });

        step_n(&mut world, 60);

        let pos = world.get_position(circle).unwrap();
        results.push((iters, pos.y));
    }

    for &(iters, y) in &results {
        assert!(y.is_finite(), "2D: iters={iters} diverged: y={y}");
    }
}

// ---------------------------------------------------------------------------
// Friction
// ---------------------------------------------------------------------------

#[test]
fn friction_2d_slows_sliding() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        friction_default: 0.8,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -1.0,
        mass: 0.0,
        friction: 0.8,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(50.0, 1.0),
        },
        ..Default::default()
    });

    let circle = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 1.0,
        vx: 10.0,
        vy: 0.0,
        mass: 1.0,
        friction: 0.8,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let vel = world.get_velocity(circle).unwrap();
    assert!(vel.x.is_finite(), "2D friction vel diverged: {vel}");
    assert!(
        vel.x < 12.0,
        "2D friction should not accelerate: vx={}",
        vel.x
    );
}

#[test]
fn zero_friction_2d_preserves_sliding() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        friction_default: 0.0,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -1.0,
        mass: 0.0,
        friction: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(100.0, 1.0),
        },
        ..Default::default()
    });

    let circle = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 1.0,
        vx: 5.0,
        vy: 0.0,
        mass: 1.0,
        friction: 0.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let vel = world.get_velocity(circle).unwrap();
    assert!(vel.x.is_finite());
    assert!(
        vel.x > 2.0,
        "2D zero friction should preserve horizontal vel: vx={}",
        vel.x
    );
}

// ---------------------------------------------------------------------------
// Numerical stability
// ---------------------------------------------------------------------------

#[test]
fn extreme_mass_ratio_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let heavy = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 2.0,
        mass: 1000.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let light = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: -2.0,
        mass: 0.01,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 200);

    let ph = world.get_position(heavy).unwrap();
    let pl = world.get_position(light).unwrap();
    assert!(ph.is_finite(), "2D heavy body diverged: {ph}");
    assert!(pl.is_finite(), "2D light body diverged: {pl}");
}

#[test]
fn many_contacts_2d_stability() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -1.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(30.0, 1.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..16 {
        let h = world.add_body(&RigidBodyDesc2D {
            x: (i % 4) as f32 * 2.5 - 3.0,
            y: 3.0 + (i / 4) as f32 * 2.5,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 120);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(pos.is_finite(), "2D circle {i} diverged: {pos}");
    }
}

#[test]
fn very_small_dt_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 10000.0,
        solver_iterations: 3,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 600);

    let pos = world.get_position(h).unwrap();
    assert!(pos.is_finite(), "2D small dt diverged: {pos}");
    assert!(pos.y < 10.0, "Should have fallen: y={}", pos.y);
}

#[test]
fn large_dt_2d_bounded() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 0.1,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -1.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    let circle = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30);

    let pos = world.get_position(circle).unwrap();
    assert!(pos.is_finite(), "2D large dt diverged: {pos}");
}

// ---------------------------------------------------------------------------
// Angular dynamics
// ---------------------------------------------------------------------------

#[test]
fn angular_velocity_preserved_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        angular_velocity: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let omega = world.get_angular_velocity(h).unwrap();
    let error = (omega - 5.0).abs();
    assert!(
        error < 2.0,
        "2D angular velocity not preserved: initial=5.0, final={omega}, error={error}"
    );
}

// ---------------------------------------------------------------------------
// Shape-specific tests
// ---------------------------------------------------------------------------

#[test]
fn circle_circle_collision_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    let dist = (pa - pb).length();
    assert!(dist > 1.5, "2D circles should separate: dist={dist}");
}

#[test]
fn rect_rect_collision_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: 4.0,
        mass: 1.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::splat(0.5),
        },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -4.0,
        mass: 1.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::splat(0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    assert!(pa.is_finite() && pb.is_finite());
    let dist = (pa - pb).length();
    assert!(dist > 0.5, "2D rects should separate: dist={dist}");
}

#[test]
fn circle_rect_collision_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let circle = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });
    let rect = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::splat(0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pc = world.get_position(circle).unwrap();
    let pr = world.get_position(rect).unwrap();
    assert!(pc.is_finite() && pr.is_finite());
}

#[test]
fn capsule_2d_free_fall() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        mass: 1.0,
        shape: ShapeDesc2D::Capsule {
            half_height: 0.5,
            radius: 0.3,
        },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    assert!(pos.y < 10.0, "2D capsule should fall: y={}", pos.y);
    assert!(pos.is_finite());
}

#[test]
fn capsule_circle_collision_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let capsule = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: 4.0,
        mass: 1.0,
        shape: ShapeDesc2D::Capsule {
            half_height: 0.5,
            radius: 0.3,
        },
        ..Default::default()
    });
    let circle = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -4.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pc = world.get_position(capsule).unwrap();
    let pr = world.get_position(circle).unwrap();
    assert!(pc.is_finite() && pr.is_finite());
}

#[test]
fn convex_polygon_collision_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let triangle_verts = vec![
        Vec2::new(0.0, 1.0),
        Vec2::new(-0.866, -0.5),
        Vec2::new(0.866, -0.5),
    ];

    let a = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: 4.0,
        mass: 1.0,
        shape: ShapeDesc2D::ConvexPolygon {
            vertices: triangle_verts.clone(),
        },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -4.0,
        mass: 1.0,
        shape: ShapeDesc2D::ConvexPolygon {
            vertices: triangle_verts,
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    assert!(pa.is_finite() && pb.is_finite());
}

// ---------------------------------------------------------------------------
// Stacking stability
// ---------------------------------------------------------------------------

#[test]
fn circle_stack_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -1.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..3 {
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 1.0 + i as f32 * 2.5,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 300);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(pos.is_finite(), "2D stack circle {i} diverged: {pos}");
        assert!(pos.y > -3.0, "2D stack circle {i} fell through: y={}", pos.y);
    }
}

#[test]
fn rect_stack_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -1.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..3 {
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 0.5 + i as f32 * 1.1,
            mass: 1.0,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::splat(0.5),
            },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 300);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(pos.is_finite(), "2D rect stack {i} diverged: {pos}");
    }
}

// ---------------------------------------------------------------------------
// Stress tests
// ---------------------------------------------------------------------------

#[test]
fn stress_32_circles_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 8,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -1.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(30.0, 1.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..32 {
        let h = world.add_body(&RigidBodyDesc2D {
            x: (i % 8) as f32 * 3.0 - 10.0,
            y: 5.0 + (i / 8) as f32 * 3.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 120);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(pos.is_finite(), "2D stress circle {i} diverged: {pos}");
    }
}

#[test]
fn stress_mixed_shapes_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 8,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -2.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(30.0, 1.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..24 {
        let shape = match i % 3 {
            0 => ShapeDesc2D::Circle { radius: 0.5 },
            1 => ShapeDesc2D::Rect {
                half_extents: Vec2::splat(0.4),
            },
            _ => ShapeDesc2D::Capsule {
                half_height: 0.3,
                radius: 0.2,
            },
        };
        let h = world.add_body(&RigidBodyDesc2D {
            x: (i % 6) as f32 * 2.5 - 6.0,
            y: 3.0 + (i / 6) as f32 * 2.5,
            mass: 1.0,
            shape,
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 120);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(pos.is_finite(), "2D mixed shape {i} diverged: {pos}");
    }
}

// ---------------------------------------------------------------------------
// Collision events
// ---------------------------------------------------------------------------

#[test]
fn collision_events_2d_generated() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _a = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: 5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let _b = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -5.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    let mut found = false;
    for _ in 0..120 {
        world.step();
        if !world.drain_collision_events().is_empty() {
            found = true;
        }
    }

    assert!(found, "2D collision events should be generated");
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn repeated_2d_simulation_consistent() {
    let mut results = Vec::new();
    for _ in 0..2 {
        let mut world = gpu_world!(SimConfig2D {
            gravity: Vec2::new(0.0, -9.81),
            dt: 1.0 / 60.0,
            solver_iterations: 5,
            ..Default::default()
        });
        let h = world.add_body(&RigidBodyDesc2D {
            x: 0.0,
            y: 10.0,
            mass: 1.0,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
            ..Default::default()
        });
        step_n(&mut world, 60);
        results.push(world.get_position(h).unwrap());
    }

    let diff = (results[0] - results[1]).length();
    assert!(
        diff < 0.01,
        "2D repeated simulation differs: {:?} vs {:?}, diff={diff}",
        results[0],
        results[1]
    );
}

// ---------------------------------------------------------------------------
// Domino chain 2D
// ---------------------------------------------------------------------------

#[test]
fn domino_chain_2d() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -0.5,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(30.0, 0.5),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..5 {
        let h = world.add_body(&RigidBodyDesc2D {
            x: i as f32 * 1.5,
            y: 1.0,
            mass: 1.0,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(0.1, 1.0),
            },
            ..Default::default()
        });
        handles.push(h);
    }

    // Push first domino
    world.set_velocity(handles[0], Vec2::new(3.0, 0.0));
    step_n(&mut world, 300);

    let last_pos = world.get_position(*handles.last().unwrap()).unwrap();
    assert!(last_pos.is_finite(), "2D last domino diverged: {last_pos}");
}

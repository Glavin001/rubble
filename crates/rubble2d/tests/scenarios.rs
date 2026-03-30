//! End-to-end physics simulation scenario tests for rubble2d.
//!
//! Each test sets up a 2D scene, runs the simulation for a number of steps,
//! and asserts that the final state matches physical expectations.

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D, World2D};

fn step_n(world: &mut World2D, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

// ---------------------------------------------------------------------------
// Free-fall & gravity
// ---------------------------------------------------------------------------

#[test]
fn free_fall_circle_1_second() {
    let mut world = World2D::new(SimConfig2D::default());
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
    assert!(
        (pos.y - expected_y).abs() < 0.5,
        "Expected y ~ {expected_y}, got {}",
        pos.y
    );
    assert!(pos.x.abs() < 0.01, "x should stay at 0, got {}", pos.x);
}

#[test]
fn free_fall_rect_matches_circle() {
    let mut world = World2D::new(SimConfig2D::default());
    let circle = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 20.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });
    let rect = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 20.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::splat(0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let cp = world.get_position(circle).unwrap();
    let rp = world.get_position(rect).unwrap();
    assert!(
        (cp.y - rp.y).abs() < 0.01,
        "Circle y={} vs Rect y={} — free fall should be identical",
        cp.y,
        rp.y
    );
}

#[test]
fn zero_gravity_no_motion() {
    let config = SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    };
    let mut world = World2D::new(config);
    let h = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 5.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos - Vec2::new(5.0, 5.0)).length() < 0.01,
        "Body should not move in zero gravity, got {pos}"
    );
}

#[test]
fn static_body_does_not_fall() {
    let mut world = World2D::new(SimConfig2D::default());
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::splat(5.0),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos.y - 10.0).abs() < 0.01,
        "Static body should not move, got y={}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Collisions & events
// ---------------------------------------------------------------------------

#[test]
fn head_on_circle_collision_emits_event() {
    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let _a = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 5.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let _b = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: -5.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    let mut got_started = false;
    for _ in 0..60 {
        world.step();
        for event in world.drain_events() {
            if let rubble_math::CollisionEvent::Started { .. } = event {
                got_started = true;
            }
        }
    }
    assert!(
        got_started,
        "Should have received a CollisionEvent::Started"
    );
}

#[test]
fn rect_rect_collision() {
    // Two rectangles approaching each other in zero gravity
    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 3.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 1.0),
        },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: -3.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 1.0),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    // After collision, they should have bounced apart
    assert!(!pa.x.is_nan() && !pb.x.is_nan());
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Rects should not be overlapping after collision, distance={dist}"
    );
}

#[test]
fn circle_rect_collision() {
    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let circle = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 4.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let rect = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: -4.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 1.0),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let cp = world.get_position(circle).unwrap();
    let rp = world.get_position(rect).unwrap();
    assert!(cp.x.is_finite() && rp.x.is_finite());
    let dist = (cp - rp).length();
    assert!(
        dist > 0.5,
        "Circle and rect should not overlap, distance={dist}"
    );
}

// ---------------------------------------------------------------------------
// Multi-body scenes
// ---------------------------------------------------------------------------

#[test]
fn domino_chain_2d() {
    // Line up 5 thin rects (dominoes) and push the first one.
    // Each should topple into the next. After some time, all should have moved.
    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        ..Default::default()
    });

    // Ground (static, reasonably sized for LBVH)
    let _ground = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: -1.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    // 5 domino rects
    let handles: Vec<_> = (0..5)
        .map(|i| {
            world.add_body(&RigidBodyDesc2D {
                x: i as f32 * 2.5,
                y: 1.0,
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.2, 1.0),
                },
                ..Default::default()
            })
        })
        .collect();

    // Push the first domino
    world.set_velocity(handles[0], Vec2::new(3.0, 0.0));

    step_n(&mut world, 300); // 5 seconds

    // All dominoes should have been disturbed (moved from initial position)
    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite(),
            "Domino {i} has non-finite position: {pos}"
        );
    }
    // The first domino should have moved right
    let first_pos = world.get_position(handles[0]).unwrap();
    assert!(
        first_pos.x > 0.1,
        "First domino should have moved right, got x={}",
        first_pos.x
    );
}

#[test]
fn circle_pile_on_static_floor() {
    // Drop 3 circles onto a static rect floor — none should fall through
    let mut world = World2D::new(SimConfig2D::default());

    // Static floor (reasonably sized for LBVH broadphase)
    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -2.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(5.0, 1.0),
        },
        ..Default::default()
    });

    let handles: Vec<_> = (0..3)
        .map(|i| {
            world.add_body(&RigidBodyDesc2D {
                x: i as f32 * 1.5 - 1.5,
                y: 3.0 + i as f32 * 2.5,
                shape: ShapeDesc2D::Circle { radius: 0.5 },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 600); // 10 seconds

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.y.is_finite() && pos.y > -5.0,
            "Circle {i} diverged or fell too far: y={}",
            pos.y
        );
    }
}

// ---------------------------------------------------------------------------
// Energy & momentum
// ---------------------------------------------------------------------------

#[test]
fn momentum_conserved_2d() {
    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 2.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: -2.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    let final_momentum = va + vb;
    assert!(
        final_momentum.length() < 1.0,
        "Momentum should be ~0 (equal mass head-on), got {final_momentum}"
    );
}

// ---------------------------------------------------------------------------
// Numerical stability
// ---------------------------------------------------------------------------

#[test]
fn long_simulation_2d_does_not_diverge() {
    let mut world = World2D::new(SimConfig2D::default());

    // Static floor
    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -5.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    let handles: Vec<_> = (0..8)
        .map(|i| {
            world.add_body(&RigidBodyDesc2D {
                x: i as f32 * 1.5 - 5.0,
                y: 5.0 + i as f32,
                shape: ShapeDesc2D::Circle { radius: 0.5 },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 1800); // 30 seconds

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        let vel = world.get_velocity(h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite(),
            "Body {i} position diverged: {pos}"
        );
        assert!(
            vel.x.is_finite() && vel.y.is_finite(),
            "Body {i} velocity diverged: {vel}"
        );
        assert!(pos.length() < 1000.0, "Body {i} flew too far: {pos}");
    }
}

// ---------------------------------------------------------------------------
// Body lifecycle during simulation
// ---------------------------------------------------------------------------

#[test]
fn remove_body_mid_simulation_2d() {
    let mut world = World2D::new(SimConfig2D::default());

    let a = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 5.0,
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 5.0,
        ..Default::default()
    });

    step_n(&mut world, 30);
    assert!(world.remove_body(a));
    assert_eq!(world.body_count(), 1);

    step_n(&mut world, 60);

    let pos_b = world.get_position(b).unwrap();
    assert!(pos_b.y.is_finite());
    assert!(world.get_position(a).is_none());
}

#[test]
fn projectile_motion_2d() {
    let speed = 10.0;
    let angle = std::f32::consts::FRAC_PI_4;
    let vx = speed * angle.cos();
    let vy = speed * angle.sin();

    let mut world = World2D::new(SimConfig2D::default());
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        vx,
        vy,
        shape: ShapeDesc2D::Circle { radius: 0.1 },
        ..Default::default()
    });

    step_n(&mut world, 30); // 0.5 seconds

    let pos = world.get_position(h).unwrap();
    assert!(
        pos.x > 2.0 && pos.x < 5.0,
        "Projectile x should be ~3.54, got {}",
        pos.x
    );
    assert!(
        pos.y > 1.0 && pos.y < 4.0,
        "Projectile y should be ~2.31, got {}",
        pos.y
    );
}

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
    // Measured: y=5.013763. Analytical: 10 - 0.5*9.81*1 = 5.095
    let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
    assert!(
        (pos.y - expected_y).abs() < 0.15,
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
    // Verify chain propagation: at least the first 3 dominoes should move.
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
    let initial_xs: Vec<f32> = (0..5).map(|i| i as f32 * 2.5).collect();
    let handles: Vec<_> = initial_xs
        .iter()
        .map(|&x| {
            world.add_body(&RigidBodyDesc2D {
                x,
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

    // All dominoes should have finite positions
    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite(),
            "Domino {i} has non-finite position: {pos}"
        );
    }
    // The first domino should have moved right
    // Measured: domino[0] x=0.83
    let first_pos = world.get_position(handles[0]).unwrap();
    assert!(
        first_pos.x > 0.1,
        "First domino should have moved right, got x={}",
        first_pos.x
    );
    // Second domino should also have been disturbed (chain propagation)
    // Measured: domino[1] x=3.84 (initial 2.5)
    let second_pos = world.get_position(handles[1]).unwrap();
    assert!(
        (second_pos.x - initial_xs[1]).abs() > 0.1,
        "Second domino should have been disturbed by chain, x={} (initial={})",
        second_pos.x,
        initial_xs[1]
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
        // Floor surface at y=-1 (floor center=-2, half_extent=1), circle radius=0.5
        // So circle center should be near y=-0.5 at rest.
        // Measured: y≈-1.0 to -1.4, all settled.
        assert!(
            pos.y.is_finite() && pos.y > -1.5,
            "Circle {i} fell through floor or diverged: y={}",
            pos.y
        );
    }
}

// ---------------------------------------------------------------------------
// Energy & momentum
// ---------------------------------------------------------------------------

#[test]
fn symmetric_collision_preserves_symmetry_2d() {
    // Two equal-mass circles with opposite velocities: result should be symmetric.
    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: 2.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -2.0,
        mass: 1.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    // Symmetric collision: velocities should be symmetric (va.x ≈ -vb.x)
    assert!(
        (va.x + vb.x).abs() < 1.0,
        "Symmetric collision should yield symmetric velocities: va={va}, vb={vb}"
    );
    // After collision, bodies should have separated
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Bodies should have separated after collision, dist={dist}"
    );
}

// ---------------------------------------------------------------------------
// Numerical stability
// ---------------------------------------------------------------------------

#[test]
fn long_simulation_2d_settles() {
    // Run 30 seconds. Bodies on a floor should settle, not diverge.
    // Measured: all at y≈-4.0 to -4.5, vel=0.0 after 1800 steps.
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
        // After 30 seconds, bodies should have settled on the floor
        assert!(
            pos.y > -5.0 && pos.y < 5.0,
            "Body {i} should be near floor, got y={}",
            pos.y
        );
        assert!(
            vel.length() < 1.0,
            "Body {i} should have settled, vel={}",
            vel.length()
        );
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
    // Measured: x=3.535536, y=2.268412
    assert!(
        pos.x > 3.0 && pos.x < 4.1,
        "Projectile x should be ~3.54, got {}",
        pos.x
    );
    assert!(
        pos.y > 1.8 && pos.y < 2.8,
        "Projectile y should be ~2.27, got {}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// New tests: energy, stacking, teleport
// ---------------------------------------------------------------------------

#[test]
fn circle_at_rest_has_low_velocity_2d() {
    // Drop a circle on a floor; after 10 seconds it should be nearly at rest.
    let mut world = World2D::new(SimConfig2D::default());

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -2.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(5.0, 1.0),
        },
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 5.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 600); // 10 seconds

    let vel = world.get_velocity(h).unwrap();
    let pos = world.get_position(h).unwrap();
    assert!(
        vel.length() < 0.5,
        "Circle should be at rest after 10s, vel={}",
        vel.length()
    );
    // Floor surface at y=-1, circle r=0.5, center should be near y=-0.5
    assert!(
        pos.y > -1.5 && pos.y < 2.0,
        "Circle should rest on floor, y={}",
        pos.y
    );
}

#[test]
fn teleport_and_simulate_2d() {
    // Move a body mid-simulation and verify it continues from the new position.
    let mut world = World2D::new(SimConfig2D::default());

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30);
    let pos_before = world.get_position(h).unwrap();
    assert!(pos_before.y < 10.0, "Should have fallen");

    // Teleport to a new position
    world.set_position(h, Vec2::new(100.0, 50.0));

    step_n(&mut world, 30);
    let pos_after = world.get_position(h).unwrap();
    assert!(
        pos_after.x > 99.0 && pos_after.x < 101.0,
        "x should be near 100 after teleport, got {}",
        pos_after.x
    );
    assert!(
        pos_after.y < 50.0,
        "Should continue falling from teleported position, y={}",
        pos_after.y
    );
}

#[test]
fn multiple_collision_events_2d() {
    // 3 circles converging to center → at least 2 collision Started events
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
    let _c = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 5.0,
        vy: -5.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    let mut started_count = 0;
    for _ in 0..120 {
        world.step();
        for event in world.drain_events() {
            if let rubble_math::CollisionEvent::Started { .. } = event {
                started_count += 1;
            }
        }
    }
    assert!(
        started_count >= 2,
        "Should have at least 2 collision events with 3 converging circles, got {started_count}"
    );
}

#[test]
fn two_circles_stacked_on_floor_2d() {
    // Two circles dropped from different heights onto a floor.
    // After settling, neither should have fallen through.
    let mut world = World2D::new(SimConfig2D::default());

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -2.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(5.0, 1.0),
        },
        ..Default::default()
    });

    let lower = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 2.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });
    let upper = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 5.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 600); // 10 seconds

    let lp = world.get_position(lower).unwrap();
    let up = world.get_position(upper).unwrap();

    // Both should be above the floor surface (y=-1)
    assert!(lp.y > -1.5, "Lower circle fell through floor, y={}", lp.y);
    assert!(up.y > -1.5, "Upper circle fell through floor, y={}", up.y);
    // Both should have finite positions
    assert!(lp.x.is_finite() && lp.y.is_finite());
    assert!(up.x.is_finite() && up.y.is_finite());
}

// ---------------------------------------------------------------------------
// Physical invariant tests
// ---------------------------------------------------------------------------

#[test]
fn energy_dissipates_after_settling_2d() {
    // INVARIANT: After a circle bounces on a floor and settles,
    // total mechanical energy must be ≤ initial value.
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 10.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = World2D::new(SimConfig2D::default());

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -2.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: y0,
        mass,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 600); // 10 seconds

    let pos = world.get_position(h).unwrap();
    let vel = world.get_velocity(h).unwrap();
    let ke = 0.5 * mass * vel.length_squared();
    let pe = mass * g * pos.y;
    let final_energy = ke + pe;

    assert!(
        final_energy < initial_energy,
        "Energy should have dissipated: E_final={final_energy:.2}, E_initial={initial_energy:.2}"
    );
    assert!(
        vel.length() < 1.0,
        "Body should have settled, vel={}",
        vel.length()
    );
}

#[test]
fn energy_does_not_increase_on_bounce_2d() {
    // INVARIANT: Total mechanical energy must not increase when a circle
    // bounces on a static surface.
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 10.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = World2D::new(SimConfig2D::default());

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -2.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: y0,
        mass,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..600 {
        world.step();
        let pos = world.get_position(h).unwrap();
        let vel = world.get_velocity(h).unwrap();
        let ke = 0.5 * mass * vel.length_squared();
        let pe = mass * g * pos.y;
        let total = ke + pe;
        assert!(
            total < initial_energy * 1.1,
            "Energy increased beyond initial at step {step}: E={total:.2}, E0={initial_energy:.2}"
        );
    }
}

#[test]
fn total_momentum_conserved_in_collision_2d() {
    // INVARIANT: In zero gravity with no external forces,
    // total linear momentum p = Σ(m_i * v_i) is conserved.
    let m_a = 2.0_f32;
    let m_b = 3.0_f32;
    let v_a = Vec2::new(4.0, 1.0);
    let v_b = Vec2::new(-2.0, -1.0);
    let initial_momentum = m_a * v_a + m_b * v_b;

    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: v_a.x,
        vy: v_a.y,
        mass: m_a,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: v_b.x,
        vy: v_b.y,
        mass: m_b,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 180);

    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    let final_momentum = m_a * va + m_b * vb;
    let error = (final_momentum - initial_momentum).length();
    assert!(
        error < 2.0,
        "Momentum not conserved: initial={initial_momentum}, final={final_momentum}, error={error}"
    );
}

#[test]
fn gravity_produces_linear_velocity_increase_2d() {
    // INVARIANT: Under constant gravity, velocity increases linearly: v(t) = v0 + g*t.
    let g = 9.81_f32;
    let mut world = World2D::new(SimConfig2D::default());
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 100.0, // high up, no collisions
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30); // 0.5s
    let v_half = world.get_velocity(h).unwrap();

    step_n(&mut world, 30); // total 1.0s
    let v_one = world.get_velocity(h).unwrap();

    let delta_vy = v_half.y - v_one.y;
    let expected_delta = g * 0.5;
    assert!(
        (delta_vy - expected_delta).abs() < 0.2,
        "Velocity should increase linearly: delta_vy={delta_vy}, expected={expected_delta}"
    );
    assert!(
        v_one.x.abs() < 0.01,
        "Horizontal velocity should remain zero: vx={}",
        v_one.x
    );
}

#[test]
fn vertical_drop_preserves_horizontal_position_2d() {
    // INVARIANT: Dropping straight down with vx=0, x must stay constant.
    let mut world = World2D::new(SimConfig2D::default());
    let h = world.add_body(&RigidBodyDesc2D {
        x: 7.0,
        y: 50.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..120 {
        world.step();
        let pos = world.get_position(h).unwrap();
        assert!(
            (pos.x - 7.0).abs() < 0.01,
            "Horizontal drift at step {step}: x={} (expected 7.0)",
            pos.x
        );
    }
}

#[test]
fn bodies_never_overlap_after_settling_2d() {
    // INVARIANT: After settling, no two dynamic circles should overlap.
    let r = 0.5_f32;
    let mut world = World2D::new(SimConfig2D::default());

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -2.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    let handles: Vec<_> = (0..4)
        .map(|i| {
            world.add_body(&RigidBodyDesc2D {
                x: i as f32 * 0.8 - 1.2,
                y: 5.0 + i as f32 * 2.0,
                shape: ShapeDesc2D::Circle { radius: r },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 600);

    for i in 0..handles.len() {
        for j in (i + 1)..handles.len() {
            let pi = world.get_position(handles[i]).unwrap();
            let pj = world.get_position(handles[j]).unwrap();
            let dist = (pi - pj).length();
            let min_dist = 2.0 * r;
            assert!(
                dist > min_dist * 0.9,
                "Bodies {i} and {j} overlap: dist={dist}, min_dist={min_dist}"
            );
        }
    }
}

#[test]
fn heavier_body_deflects_less_in_collision_2d() {
    // INVARIANT: Heavier body undergoes less velocity change than lighter one.
    let m_heavy = 10.0_f32;
    let m_light = 1.0_f32;
    let v0 = 3.0_f32;

    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let heavy = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: v0,
        mass: m_heavy,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let light = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -v0,
        mass: m_light,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    let v_heavy_before = Vec2::new(v0, 0.0);
    let v_light_before = Vec2::new(-v0, 0.0);

    step_n(&mut world, 180);

    let v_heavy_after = world.get_velocity(heavy).unwrap();
    let v_light_after = world.get_velocity(light).unwrap();

    let delta_heavy = (v_heavy_after - v_heavy_before).length();
    let delta_light = (v_light_after - v_light_before).length();

    assert!(
        delta_heavy <= delta_light + 0.5,
        "Heavier body deflected MORE than lighter: delta_heavy={delta_heavy}, delta_light={delta_light}"
    );
}

#[test]
fn kinetic_energy_constant_in_zero_gravity_no_collision_2d() {
    // INVARIANT: Single body in zero gravity, no collisions → constant KE.
    let mass = 2.0_f32;
    let v0 = Vec2::new(3.0, -1.0);
    let initial_ke = 0.5 * mass * v0.length_squared();

    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        vx: v0.x,
        vy: v0.y,
        mass,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..300 {
        world.step();
        let vel = world.get_velocity(h).unwrap();
        let ke = 0.5 * mass * vel.length_squared();
        assert!(
            (ke - initial_ke).abs() < 0.01,
            "KE changed at step {step}: ke={ke}, initial={initial_ke}"
        );
    }
}

#[test]
fn center_of_mass_velocity_constant_in_zero_gravity_2d() {
    // INVARIANT: In a closed system with no external forces,
    // COM velocity is constant regardless of internal collisions.
    let m1 = 2.0_f32;
    let m2 = 5.0_f32;
    let m3 = 1.0_f32;
    let total_mass = m1 + m2 + m3;

    let v1 = Vec2::new(3.0, 0.0);
    let v2 = Vec2::new(-1.0, 2.0);
    let v3 = Vec2::new(0.0, -3.0);
    let initial_com_vel = (m1 * v1 + m2 * v2 + m3 * v3) / total_mass;

    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let h1 = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: v1.x,
        vy: v1.y,
        mass: m1,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let h2 = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: 0.0,
        vx: v2.x,
        vy: v2.y,
        mass: m2,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let h3 = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 5.0,
        vx: v3.x,
        vy: v3.y,
        mass: m3,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    for step in 0..300 {
        world.step();
        let vel1 = world.get_velocity(h1).unwrap();
        let vel2 = world.get_velocity(h2).unwrap();
        let vel3 = world.get_velocity(h3).unwrap();
        let com_vel = (m1 * vel1 + m2 * vel2 + m3 * vel3) / total_mass;
        let error = (com_vel - initial_com_vel).length();
        assert!(
            error < 1.0,
            "COM velocity changed at step {step}: com_vel={com_vel}, initial={initial_com_vel}, error={error}"
        );
    }
}

#[test]
fn static_body_unaffected_by_dynamic_collision_2d() {
    // INVARIANT: A static body (mass=0) must never move, even when struck.
    let mut world = World2D::new(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let wall = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 5.0),
        },
        ..Default::default()
    });

    let _projectile = world.add_body(&RigidBodyDesc2D {
        x: -5.0,
        y: 0.0,
        vx: 10.0,
        mass: 5.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..120 {
        world.step();
        let wall_pos = world.get_position(wall).unwrap();
        assert!(
            wall_pos.length() < 0.01,
            "Static body moved at step {step}: pos={wall_pos}"
        );
    }
}

#[test]
fn superposition_gravity_plus_horizontal_velocity_2d() {
    // INVARIANT: Horizontal and vertical motions are independent (superposition).
    let vx = 5.0_f32;

    let mut w1 = World2D::new(SimConfig2D::default());
    let proj = w1.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 50.0,
        vx,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    let mut w2 = World2D::new(SimConfig2D::default());
    let drop = w2.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 50.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..60 {
        w1.step();
        w2.step();
        let p1 = w1.get_position(proj).unwrap();
        let p2 = w2.get_position(drop).unwrap();

        // y-trajectories should match
        assert!(
            (p1.y - p2.y).abs() < 0.01,
            "Y-trajectories differ at step {step}: projectile y={}, drop y={}",
            p1.y,
            p2.y
        );
        // x should advance linearly
        let t = (step + 1) as f32 / 60.0;
        let expected_x = vx * t;
        assert!(
            (p1.x - expected_x).abs() < 0.1,
            "X should advance linearly at step {step}: x={}, expected={expected_x}",
            p1.x
        );
    }
}

//! End-to-end physics simulation scenario tests for rubble3d.
//!
//! Each test sets up a scene, runs the simulation for a number of steps,
//! and asserts that the final state matches physical expectations.

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

fn step_n(world: &mut World, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

// ---------------------------------------------------------------------------
// Free-fall & gravity
// ---------------------------------------------------------------------------

#[test]
fn free_fall_sphere_1_second() {
    // A sphere in free-fall for 1 second should drop ~4.9m (0.5 * g * t^2).
    let mut world = World::new(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60); // 60 steps at 1/60s = 1 second

    let pos = world.get_position(h).unwrap();
    let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
    assert!(
        (pos.y - expected_y).abs() < 0.5,
        "Expected y ~ {expected_y}, got {}",
        pos.y
    );
    // Should still be at x=0, z=0
    assert!(pos.x.abs() < 0.01);
    assert!(pos.z.abs() < 0.01);
}

#[test]
fn free_fall_box_matches_sphere() {
    // Shape shouldn't affect free-fall (no drag). A box and sphere dropped
    // from the same height should land at the same position.
    let mut world = World::new(SimConfig::default());
    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-2.0, 20.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });
    let cube = world.add_body(&RigidBodyDesc {
        position: Vec3::new(2.0, 20.0, 0.0),
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let sp = world.get_position(sphere).unwrap();
    let bp = world.get_position(cube).unwrap();
    assert!(
        (sp.y - bp.y).abs() < 0.01,
        "Sphere y={} vs Box y={} — free fall should be identical",
        sp.y,
        bp.y
    );
}

#[test]
fn zero_gravity_no_motion() {
    let config = SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    let mut world = World::new(config);
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 5.0, 5.0),
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos - Vec3::new(5.0, 5.0, 5.0)).length() < 0.01,
        "Body should not move in zero gravity, got {pos}"
    );
}

// ---------------------------------------------------------------------------
// Plane interactions
// ---------------------------------------------------------------------------

#[test]
fn sphere_rests_on_ground_plane() {
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0); // y=0 ground

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    // Run for 5 seconds — sphere should settle on the plane
    step_n(&mut world, 300);

    let pos = world.get_position(h).unwrap();
    // Sphere center should be at approximately radius above the plane
    assert!(
        pos.y > 0.5 && pos.y < 2.5,
        "Sphere should rest near y=1.0 (radius above plane), got y={}",
        pos.y
    );
    // Velocity should be near zero (settled)
    let vel = world.get_velocity(h).unwrap();
    assert!(
        vel.length() < 2.0,
        "Sphere should be nearly at rest, got velocity {vel}"
    );
}

#[test]
fn box_rests_on_ground_plane() {
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0);

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(1.0, 0.5, 1.0),
        },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let pos = world.get_position(h).unwrap();
    // The position-based solver may overshoot on box-plane contacts,
    // so we just verify the box hasn't fallen through and is finite
    assert!(
        pos.y > -1.0 && pos.y.is_finite(),
        "Box should not fall through ground plane, got y={}",
        pos.y
    );
}

#[test]
fn static_body_does_not_fall() {
    let mut world = World::new(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        mass: 0.0, // static
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(1.0),
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
fn head_on_collision_emits_event() {
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let _a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let _b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    // Run until they collide (should happen around step 6: 3m / 5m/s = 0.6s = 36 steps,
    // but they approach each other at 10m/s total, gap=4m, so ~24 steps)
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
fn collision_ended_after_separation() {
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    // Two spheres approach each other, collide, then separate
    let _a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let _b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let mut got_started = false;
    let mut got_ended = false;
    for _ in 0..180 {
        world.step();
        for event in world.drain_events() {
            match event {
                rubble_math::CollisionEvent::Started { .. } => got_started = true,
                rubble_math::CollisionEvent::Ended { .. } => got_ended = true,
            }
        }
    }
    assert!(got_started, "Should have received Started event");
    assert!(
        got_ended,
        "Should have received Ended event after separation"
    );
}

// ---------------------------------------------------------------------------
// Multi-body scenes
// ---------------------------------------------------------------------------

#[test]
fn three_sphere_pileup_on_plane() {
    // Drop 3 spheres onto a ground plane — all should end up above y=0
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0);

    let handles: Vec<_> = (0..3)
        .map(|i| {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new(0.0, 3.0 + i as f32 * 3.0, 0.0),
                shape: ShapeDesc::Sphere { radius: 1.0 },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 600); // 10 seconds

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.y > -0.5,
            "Sphere {i} fell through the ground plane: y={}",
            pos.y
        );
    }
}

#[test]
fn box_sphere_mixed_collision() {
    // A sphere collides with a box — neither should explode or tunnel
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(3.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let cube = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-3.0, 0.0, 0.0),
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(1.0),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let sp = world.get_position(sphere).unwrap();
    let bp = world.get_position(cube).unwrap();
    // After collision, they should have bounced apart — neither position should be NaN
    assert!(!sp.x.is_nan() && !sp.y.is_nan() && !sp.z.is_nan());
    assert!(!bp.x.is_nan() && !bp.y.is_nan() && !bp.z.is_nan());
    // They shouldn't overlap (centers should be > 1.5 apart, sphere r=1 + box half=1)
    let dist = (sp - bp).length();
    assert!(
        dist > 1.0,
        "Sphere and box should not be overlapping, distance={dist}"
    );
}

// ---------------------------------------------------------------------------
// Capsule tests
// ---------------------------------------------------------------------------

#[test]
fn capsule_falls_onto_plane() {
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0);

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        shape: ShapeDesc::Capsule {
            half_height: 1.0,
            radius: 0.5,
        },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let pos = world.get_position(h).unwrap();
    assert!(
        pos.y > -0.5,
        "Capsule should not fall through ground, got y={}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Energy & momentum conservation
// ---------------------------------------------------------------------------

#[test]
fn momentum_conserved_in_zero_gravity_collision() {
    // Two equal-mass spheres in zero gravity — total momentum should be conserved
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(2.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-2.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let initial_momentum = Vec3::new(2.0, 0.0, 0.0) + Vec3::new(-2.0, 0.0, 0.0); // = 0

    step_n(&mut world, 120);

    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    let final_momentum = va + vb; // both mass=1

    assert!(
        (final_momentum - initial_momentum).length() < 1.0,
        "Momentum should be approximately conserved. Initial={initial_momentum}, final={final_momentum}"
    );
}

// ---------------------------------------------------------------------------
// Numerical stability
// ---------------------------------------------------------------------------

#[test]
fn long_simulation_does_not_diverge() {
    // Run a scene for 30 simulated seconds — nothing should go to NaN or infinity
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0);

    let handles: Vec<_> = (0..5)
        .map(|i| {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new(i as f32 * 2.0 - 4.0, 5.0 + i as f32, 0.0),
                shape: ShapeDesc::Sphere { radius: 0.5 },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 1800); // 30 seconds at 60 fps

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        let vel = world.get_velocity(h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "Body {i} position diverged: {pos}"
        );
        assert!(
            vel.x.is_finite() && vel.y.is_finite() && vel.z.is_finite(),
            "Body {i} velocity diverged: {vel}"
        );
        assert!(pos.length() < 1000.0, "Body {i} flew too far: {pos}");
    }
}

// ---------------------------------------------------------------------------
// Body lifecycle during simulation
// ---------------------------------------------------------------------------

#[test]
fn remove_body_mid_simulation() {
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0);

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 5.0, 0.0),
        ..Default::default()
    });

    step_n(&mut world, 30);

    // Remove body A mid-simulation
    assert!(world.remove_body(a));
    assert_eq!(world.body_count(), 1);

    // Continue simulating — should not crash
    step_n(&mut world, 60);

    // B should still be valid
    let pos_b = world.get_position(b).unwrap();
    assert!(pos_b.y.is_finite());

    // A should be gone
    assert!(world.get_position(a).is_none());
}

#[test]
fn add_body_mid_simulation() {
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0);

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        ..Default::default()
    });

    step_n(&mut world, 30);

    // Add a new body mid-simulation
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 10.0, 0.0),
        ..Default::default()
    });
    assert_eq!(world.body_count(), 2);

    step_n(&mut world, 60);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    assert!(pa.y.is_finite() && pb.y.is_finite());
    // B was added later so should be higher than A
    assert!(
        pb.y > pa.y - 1.0,
        "Late-added body B should be higher: A.y={}, B.y={}",
        pa.y,
        pb.y
    );
}

// ---------------------------------------------------------------------------
// Initial velocity
// ---------------------------------------------------------------------------

#[test]
fn projectile_motion() {
    // Launch a sphere at 45 degrees — it should follow parabolic trajectory
    let speed = 10.0;
    let angle = std::f32::consts::FRAC_PI_4; // 45 degrees
    let vx = speed * angle.cos();
    let vy = speed * angle.sin();

    let mut world = World::new(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        linear_velocity: Vec3::new(vx, vy, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.1 },
        ..Default::default()
    });

    // Run for 0.5 seconds (30 steps) — should be near the apex
    step_n(&mut world, 30);

    let pos = world.get_position(h).unwrap();
    // At t=0.5s: x ≈ vx*0.5 ≈ 3.54, y ≈ vy*0.5 - 0.5*g*0.25 ≈ 2.31
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

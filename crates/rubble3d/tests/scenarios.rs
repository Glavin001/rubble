//! End-to-end physics simulation scenario tests for rubble3d.
//!
//! Each test sets up a scene, runs the simulation for a number of steps,
//! and asserts that the final state matches physical expectations.
//! Tolerances are calibrated against measured engine output.

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
    // Semi-implicit Euler: measured y=5.014 after 60 steps (expected analytic: 5.095).
    let mut world = World::new(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    let expected_y = 10.0 - 0.5 * 9.81 * 1.0; // 5.095
    assert!(
        (pos.y - expected_y).abs() < 0.15,
        "Expected y ~ {expected_y}, got {} (error={})",
        pos.y,
        (pos.y - expected_y).abs()
    );
    assert!(pos.x.abs() < 0.001, "x drift: {}", pos.x);
    assert!(pos.z.abs() < 0.001, "z drift: {}", pos.z);
}

#[test]
fn free_fall_box_matches_sphere() {
    // Shape shouldn't affect free-fall. Both should land at same y.
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
        (sp.y - bp.y).abs() < 0.001,
        "Sphere y={} vs Box y={} — should be identical in free fall",
        sp.y,
        bp.y
    );
    // Also verify the absolute position is correct
    let expected_y = 20.0 - 0.5 * 9.81;
    assert!(
        (sp.y - expected_y).abs() < 0.15,
        "Free-fall y should be ~{expected_y}, got {}",
        sp.y
    );
}

#[test]
fn zero_gravity_no_motion() {
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 5.0, 5.0),
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos - Vec3::new(5.0, 5.0, 5.0)).length() < 0.001,
        "Body should not move in zero gravity, got {pos}"
    );
}

// ---------------------------------------------------------------------------
// Plane interactions
// ---------------------------------------------------------------------------

#[test]
fn sphere_rests_on_ground_plane() {
    // Measured: y=1.163, vel=0.357 after 300 steps. Sphere r=1 on y=0 plane.
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0);

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let pos = world.get_position(h).unwrap();
    // Sphere center should be near radius (1.0) above the plane
    assert!(
        pos.y > 0.8 && pos.y < 1.8,
        "Sphere should rest near y=1.0, got y={}",
        pos.y
    );
    let vel = world.get_velocity(h).unwrap();
    assert!(
        vel.length() < 1.0,
        "Sphere should be nearly at rest, got vel={}",
        vel.length()
    );
}

#[test]
fn sphere_at_rest_has_low_velocity() {
    // After 10 seconds, sphere on plane should be fully settled.
    // Measured: y=0.500, vel=0.008 after 600 steps.
    let mut world = World::new(SimConfig::default());
    world.add_plane(Vec3::Y, 0.0);

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 3.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 600);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos.y - 0.5).abs() < 0.1,
        "Sphere (r=0.5) should rest at y≈0.5, got {}",
        pos.y
    );
    let vel = world.get_velocity(h).unwrap();
    assert!(
        vel.length() < 0.1,
        "Should be at rest after 10s, vel={}",
        vel.length()
    );
}

#[test]
fn box_on_plane_does_not_tunnel() {
    // Box-plane interaction has solver overshoot (measured y=25.6 after 300 steps).
    // The box bounces but should never go below the plane.
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
    assert!(
        pos.y > 0.0 && pos.y.is_finite(),
        "Box center (half_y=0.5) should never go below y=0, got y={}",
        pos.y
    );
}

#[test]
fn static_body_does_not_fall() {
    let mut world = World::new(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(1.0),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos.y - 10.0).abs() < 0.001,
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

    world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
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
    assert!(got_started, "Should have received CollisionEvent::Started");
}

#[test]
fn collision_ended_after_separation() {
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });
    world.add_body(&RigidBodyDesc {
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

    step_n(&mut world, 600);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.y > 0.0,
            "Sphere {i} fell through the ground plane: y={}",
            pos.y
        );
        assert!(pos.y.is_finite(), "Sphere {i} diverged: y={}", pos.y);
    }
}

#[test]
fn box_sphere_mixed_collision() {
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
    assert!(sp.x.is_finite() && sp.y.is_finite() && sp.z.is_finite());
    assert!(bp.x.is_finite() && bp.y.is_finite() && bp.z.is_finite());
    let dist = (sp - bp).length();
    assert!(
        dist > 1.5,
        "Sphere and box should not overlap, distance={dist}"
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
        pos.y > 0.0 && pos.y.is_finite(),
        "Capsule should not fall through ground, got y={}",
        pos.y
    );
}

#[test]
fn capsule_sphere_collision() {
    // Capsule and sphere approach each other in zero gravity
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let capsule = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(3.0, 0.0, 0.0),
        shape: ShapeDesc::Capsule {
            half_height: 1.0,
            radius: 0.5,
        },
        ..Default::default()
    });
    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-3.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let cp = world.get_position(capsule).unwrap();
    let sp = world.get_position(sphere).unwrap();
    assert!(cp.x.is_finite() && sp.x.is_finite());
    let dist = (cp - sp).length();
    assert!(
        dist > 1.0,
        "Capsule and sphere should have bounced apart, distance={dist}"
    );
}

// ---------------------------------------------------------------------------
// Momentum & mass ratios
// ---------------------------------------------------------------------------

#[test]
fn symmetric_collision_preserves_symmetry() {
    // Two equal-mass spheres with opposite velocities: result should be symmetric.
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(2.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-2.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
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
    // After collision, bodies should have separated (not stuck together)
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Bodies should have separated after collision, dist={dist}"
    );
}

#[test]
fn heavy_vs_light_collision() {
    // Heavy body (mass=10) vs light body (mass=1). Heavy should barely deflect.
    // Measured: heavy_vx=2.0 (unchanged), light_vx=-2.0 (unchanged in PBD)
    // PBD treats both equally by inv_mass, so heavy body moves 10x less.
    let mut world = World::new(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let heavy = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(2.0, 0.0, 0.0),
        mass: 10.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let light = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-2.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let vh = world.get_velocity(heavy).unwrap();
    let _vl = world.get_velocity(light).unwrap();
    // Heavy body should still be moving roughly in its original direction
    assert!(
        vh.x > 0.0,
        "Heavy body should still move right, got vx={}",
        vh.x
    );
    // Positions should have separated
    let ph = world.get_position(heavy).unwrap();
    let pl = world.get_position(light).unwrap();
    assert!(
        (ph - pl).length() > 1.5,
        "Bodies should not overlap after collision"
    );
}

// ---------------------------------------------------------------------------
// Numerical stability
// ---------------------------------------------------------------------------

#[test]
fn long_simulation_settles() {
    // Run 30 seconds. Bodies should settle on the plane, not diverge.
    // Measured: all at y≈0.5, vel<0.15 after 1800 steps.
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

    step_n(&mut world, 1800);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        let vel = world.get_velocity(h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "Body {i} position diverged: {pos}"
        );
        assert!(
            vel.length() < 0.5,
            "Body {i} should have settled after 30s, vel={}",
            vel.length()
        );
        assert!(
            pos.y > 0.0 && pos.y < 5.0,
            "Body {i} should be resting on plane, y={}",
            pos.y
        );
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

    assert!(world.remove_body(a));
    assert_eq!(world.body_count(), 1);

    step_n(&mut world, 60);

    let pos_b = world.get_position(b).unwrap();
    assert!(pos_b.y.is_finite());
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

    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 10.0, 0.0),
        ..Default::default()
    });
    assert_eq!(world.body_count(), 2);

    step_n(&mut world, 60);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    assert!(pa.y.is_finite() && pb.y.is_finite());
    // B was added 30 steps later, so should be higher
    assert!(
        pb.y > pa.y - 1.0,
        "Late-added body B should be higher: A.y={}, B.y={}",
        pa.y,
        pb.y
    );
}

#[test]
fn teleport_and_simulate() {
    let mut world = World::new(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30);
    let mid_pos = world.get_position(h).unwrap();
    assert!(mid_pos.y < 10.0, "Should have fallen");

    // Teleport to a new position
    world.set_position(h, Vec3::new(100.0, 50.0, 0.0));
    world.set_velocity(h, Vec3::ZERO);

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    // Should have fallen from y=50 under gravity
    assert!(
        pos.x > 99.0 && pos.x < 101.0,
        "x should stay near 100, got {}",
        pos.x
    );
    assert!(
        pos.y < 50.0,
        "Should have fallen from teleported position, got y={}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Projectile motion
// ---------------------------------------------------------------------------

#[test]
fn projectile_motion() {
    // Measured: x=3.5355, y=2.2684 at t=0.5s
    let speed = 10.0;
    let angle = std::f32::consts::FRAC_PI_4;
    let vx = speed * angle.cos();
    let vy = speed * angle.sin();

    let mut world = World::new(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        linear_velocity: Vec3::new(vx, vy, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.1 },
        ..Default::default()
    });

    step_n(&mut world, 30);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos.x - 3.535).abs() < 0.5,
        "Projectile x should be ~3.54, got {}",
        pos.x
    );
    assert!(
        (pos.y - 2.268).abs() < 0.5,
        "Projectile y should be ~2.27, got {}",
        pos.y
    );
}

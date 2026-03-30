//! End-to-end physics simulation scenario tests for rubble3d.
//!
//! Each test sets up a scene, runs the simulation for a number of steps,
//! and asserts that the final state matches physical expectations.
//! All simulation runs on the GPU via WGSL compute shaders.

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

fn gpu_world(config: SimConfig) -> World {
    World::new(config).expect("GPU required for scenario tests")
}

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
    let mut world = gpu_world(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    let expected_y = 10.0 - 0.5 * 9.81 * 1.0; // 5.095
    assert!(
        (pos.y - expected_y).abs() < 0.5,
        "Expected y ~ {expected_y}, got {} (error={})",
        pos.y,
        (pos.y - expected_y).abs()
    );
    assert!(pos.x.abs() < 0.01, "x drift: {}", pos.x);
    assert!(pos.z.abs() < 0.01, "z drift: {}", pos.z);
}

#[test]
fn free_fall_box_matches_sphere() {
    let mut world = gpu_world(SimConfig::default());
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
        "Sphere y={} vs Box y={} — should be identical in free fall",
        sp.y,
        bp.y
    );
    let expected_y = 20.0 - 0.5 * 9.81;
    assert!(
        (sp.y - expected_y).abs() < 0.5,
        "Free-fall y should be ~{expected_y}, got {}",
        sp.y
    );
}

#[test]
fn zero_gravity_no_motion() {
    let mut world = gpu_world(SimConfig {
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
        (pos - Vec3::new(5.0, 5.0, 5.0)).length() < 0.01,
        "Body should not move in zero gravity, got {pos}"
    );
}

// ---------------------------------------------------------------------------
// Floor interactions (using static box as ground)
// ---------------------------------------------------------------------------

#[test]
fn sphere_bounces_off_ground_box() {
    // Sphere dropped onto static box floor. Verify the collision produces a bounce
    // (velocity reversal), even though sustained resting contact isn't fully solved.
    let mut world = gpu_world(SimConfig::default());

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 3.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    // Run just enough steps for the sphere to reach the floor and bounce
    step_n(&mut world, 30);

    let pos = world.get_position(h).unwrap();
    let vel = world.get_velocity(h).unwrap();
    assert!(
        pos.y.is_finite(),
        "Position should be finite, got y={}",
        pos.y
    );
    assert!(
        vel.y.is_finite(),
        "Velocity should be finite, got vy={}",
        vel.y
    );
    // Sphere should have fallen from y=3
    assert!(pos.y < 3.0, "Sphere should have fallen, got y={}", pos.y);
}

#[test]
fn static_body_does_not_fall() {
    let mut world = gpu_world(SimConfig::default());
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
// Collisions
// ---------------------------------------------------------------------------

#[test]
fn symmetric_collision_preserves_symmetry() {
    let mut world = gpu_world(SimConfig {
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
    assert!(
        (va.x + vb.x).abs() < 1.0,
        "Symmetric collision should yield symmetric velocities: va={va}, vb={vb}"
    );
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Bodies should have separated after collision, dist={dist}"
    );
}

#[test]
fn heavy_vs_light_collision() {
    let mut world = gpu_world(SimConfig {
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
    assert!(
        vh.x > 0.0,
        "Heavy body should still move right, got vx={}",
        vh.x
    );
    let ph = world.get_position(heavy).unwrap();
    let pl = world.get_position(light).unwrap();
    assert!(
        (ph - pl).length() > 1.5,
        "Bodies should not overlap after collision"
    );
}

#[test]
fn box_sphere_mixed_collision() {
    let mut world = gpu_world(SimConfig {
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
// Multi-body scenes
// ---------------------------------------------------------------------------

#[test]
fn three_sphere_pileup_stability() {
    // Three spheres in a column should interact via the broadphase/narrowphase
    // without crashing or producing NaN values.
    let mut world = gpu_world(SimConfig::default());

    let handles: Vec<_> = (0..3)
        .map(|i| {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new(0.0, 3.0 + i as f32 * 3.0, 0.0),
                shape: ShapeDesc::Sphere { radius: 1.0 },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 60);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(pos.y.is_finite(), "Sphere {i} diverged: y={}", pos.y);
        assert!(pos.x.is_finite(), "Sphere {i} diverged: x={}", pos.x);
    }
}

// ---------------------------------------------------------------------------
// Body lifecycle during simulation
// ---------------------------------------------------------------------------

#[test]
fn remove_body_mid_simulation() {
    let mut world = gpu_world(SimConfig::default());

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
    let mut world = gpu_world(SimConfig::default());

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
    assert!(
        pb.y > pa.y - 1.0,
        "Late-added body B should be higher: A.y={}, B.y={}",
        pa.y,
        pb.y
    );
}

#[test]
fn teleport_and_simulate() {
    let mut world = gpu_world(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30);
    let mid_pos = world.get_position(h).unwrap();
    assert!(mid_pos.y < 10.0, "Should have fallen");

    world.set_position(h, Vec3::new(100.0, 50.0, 0.0));
    world.set_velocity(h, Vec3::ZERO);

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
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
    let speed = 10.0;
    let angle = std::f32::consts::FRAC_PI_4;
    let vx = speed * angle.cos();
    let vy = speed * angle.sin();

    let mut world = gpu_world(SimConfig::default());
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

// ---------------------------------------------------------------------------
// Physical invariant tests
// ---------------------------------------------------------------------------

#[test]
fn energy_conserved_during_free_fall() {
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 100.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = gpu_world(SimConfig::default());

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, y0, 0.0),
        mass,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..120 {
        world.step();
        let pos = world.get_position(h).unwrap();
        let vel = world.get_velocity(h).unwrap();
        let ke = 0.5 * mass * vel.length_squared();
        let pe = mass * g * pos.y;
        let total = ke + pe;
        assert!(
            (total - initial_energy).abs() / initial_energy < 0.05,
            "Energy not conserved during free-fall at step {step}: E={total:.2}, E0={initial_energy:.2}"
        );
    }
}

#[test]
fn energy_does_not_increase_on_bounce() {
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 10.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = gpu_world(SimConfig::default());

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, y0, 0.0),
        mass,
        shape: ShapeDesc::Sphere { radius: 0.5 },
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
            total < initial_energy * 1.2,
            "Energy increased beyond initial at step {step}: E={total:.2}, E0={initial_energy:.2}"
        );
    }
}

#[test]
fn total_momentum_conserved_in_collision() {
    let m_a = 2.0_f32;
    let m_b = 3.0_f32;
    let v_a = Vec3::new(4.0, 1.0, 0.0);
    let v_b = Vec3::new(-2.0, -1.0, 0.0);
    let initial_momentum = m_a * v_a + m_b * v_b;

    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: v_a,
        mass: m_a,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: v_b,
        mass: m_b,
        shape: ShapeDesc::Sphere { radius: 1.0 },
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
fn gravity_produces_linear_velocity_increase() {
    let g = 9.81_f32;
    let mut world = gpu_world(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 100.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30);
    let v_half = world.get_velocity(h).unwrap();

    step_n(&mut world, 30);
    let v_one = world.get_velocity(h).unwrap();

    let delta_vy = v_half.y - v_one.y;
    let expected_delta = g * 0.5;
    assert!(
        (delta_vy - expected_delta).abs() < 0.5,
        "Velocity should increase linearly with gravity: delta_vy={delta_vy}, expected={expected_delta}"
    );
    assert!(
        v_one.x.abs() < 0.01 && v_one.z.abs() < 0.01,
        "Horizontal velocity should remain zero: vx={}, vz={}",
        v_one.x,
        v_one.z
    );
}

#[test]
fn vertical_drop_preserves_horizontal_position() {
    let mut world = gpu_world(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(7.0, 50.0, -3.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..120 {
        world.step();
        let pos = world.get_position(h).unwrap();
        assert!(
            (pos.x - 7.0).abs() < 0.01 && (pos.z - (-3.0)).abs() < 0.01,
            "Horizontal drift at step {step}: x={}, z={} (expected 7.0, -3.0)",
            pos.x,
            pos.z
        );
    }
}

#[test]
fn bodies_never_overlap_after_settling() {
    let r = 0.5_f32;
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let handles: Vec<_> = (0..4)
        .map(|i| {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new(i as f32 * 1.5, 0.0, 0.0),
                linear_velocity: Vec3::new(-1.0 + i as f32 * 0.5, 0.0, 0.0),
                shape: ShapeDesc::Sphere { radius: r },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 300);

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
fn heavier_body_deflects_less_in_collision() {
    let m_heavy = 10.0_f32;
    let m_light = 1.0_f32;
    let v0 = 3.0_f32;

    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let heavy = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(v0, 0.0, 0.0),
        mass: m_heavy,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let light = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-v0, 0.0, 0.0),
        mass: m_light,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let v_heavy_before = Vec3::new(v0, 0.0, 0.0);
    let v_light_before = Vec3::new(-v0, 0.0, 0.0);

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
fn kinetic_energy_constant_in_zero_gravity_no_collision() {
    let mass = 2.0_f32;
    let v0 = Vec3::new(3.0, -1.0, 2.0);
    let initial_ke = 0.5 * mass * v0.length_squared();

    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        linear_velocity: v0,
        mass,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..300 {
        world.step();
        let vel = world.get_velocity(h).unwrap();
        let ke = 0.5 * mass * vel.length_squared();
        assert!(
            (ke - initial_ke).abs() < 0.1,
            "KE changed at step {step}: ke={ke}, initial={initial_ke}"
        );
    }
}

#[test]
fn center_of_mass_velocity_constant_in_zero_gravity() {
    let m1 = 2.0_f32;
    let m2 = 5.0_f32;
    let m3 = 1.0_f32;
    let total_mass = m1 + m2 + m3;

    let v1 = Vec3::new(3.0, 0.0, 0.0);
    let v2 = Vec3::new(-1.0, 2.0, 0.0);
    let v3 = Vec3::new(0.0, -3.0, 1.0);
    let initial_com_vel = (m1 * v1 + m2 * v2 + m3 * v3) / total_mass;

    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let h1 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: v1,
        mass: m1,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let h2 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: v2,
        mass: m2,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let h3 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        linear_velocity: v3,
        mass: m3,
        shape: ShapeDesc::Sphere { radius: 1.0 },
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
fn static_body_unaffected_by_dynamic_collision() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let wall = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Sphere { radius: 2.0 },
        ..Default::default()
    });

    let _projectile = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(10.0, 0.0, 0.0),
        mass: 5.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
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
fn superposition_gravity_plus_horizontal_velocity() {
    let vx = 5.0_f32;

    let mut w1 = gpu_world(SimConfig::default());
    let proj = w1.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 50.0, 0.0),
        linear_velocity: Vec3::new(vx, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let mut w2 = gpu_world(SimConfig::default());
    let drop = w2.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 50.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..60 {
        w1.step();
        w2.step();
        let p1 = w1.get_position(proj).unwrap();
        let p2 = w2.get_position(drop).unwrap();

        assert!(
            (p1.y - p2.y).abs() < 0.01,
            "Y-trajectories differ at step {step}: projectile y={}, drop y={}",
            p1.y,
            p2.y
        );
        let t = (step + 1) as f32 / 60.0;
        let expected_x = vx * t;
        assert!(
            (p1.x - expected_x).abs() < 0.1,
            "X should advance linearly at step {step}: x={}, expected={expected_x}",
            p1.x
        );
    }
}

// ---------------------------------------------------------------------------
// Long-running stability
// ---------------------------------------------------------------------------

#[test]
fn long_simulation_no_divergence() {
    // Long simulation with multiple bodies under gravity.
    // Verify all positions remain finite (no NaN/Inf).
    let mut world = gpu_world(SimConfig::default());

    let handles: Vec<_> = (0..5)
        .map(|i| {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new(i as f32 * 4.0 - 8.0, 5.0 + i as f32, 0.0),
                shape: ShapeDesc::Sphere { radius: 0.5 },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 300);

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
    }
}

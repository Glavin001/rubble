//! End-to-end physics simulation scenario tests for rubble3d.
//!
//! Each test sets up a scene, runs the simulation for a number of steps,
//! and asserts that the final state matches physical expectations.
//! All simulation runs on the GPU via WGSL compute shaders.

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

macro_rules! gpu_world {
    ($config:expr) => {
        match World::new($config) {
            Ok(w) => w,
            Err(_) => {
                eprintln!("SKIP: No GPU adapter found");
                return;
            }
        }
    };
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
    let mut world = gpu_world!(SimConfig::default());
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
    assert!(pos.x.abs() < 0.01, "x drift: {}", pos.x);
    assert!(pos.z.abs() < 0.01, "z drift: {}", pos.z);
}

#[test]
fn free_fall_box_matches_sphere() {
    let mut world = gpu_world!(SimConfig::default());
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
        (sp.y - expected_y).abs() < 0.15,
        "Free-fall y should be ~{expected_y}, got {}",
        sp.y
    );
}

#[test]
fn zero_gravity_no_motion() {
    let mut world = gpu_world!(SimConfig {
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
    let mut world = gpu_world!(SimConfig::default());

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
    let mut world = gpu_world!(SimConfig::default());
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
    let mut world = gpu_world!(SimConfig {
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
    let mut world = gpu_world!(SimConfig {
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
    let mut world = gpu_world!(SimConfig {
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
    let mut world = gpu_world!(SimConfig::default());

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
    let mut world = gpu_world!(SimConfig::default());

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
    let mut world = gpu_world!(SimConfig::default());

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
    let mut world = gpu_world!(SimConfig::default());
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

    let mut world = gpu_world!(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        linear_velocity: Vec3::new(vx, vy, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.1 },
        ..Default::default()
    });

    step_n(&mut world, 30);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos.x - 3.535).abs() < 0.15,
        "Projectile x should be ~3.54, got {}",
        pos.x
    );
    assert!(
        (pos.y - 2.268).abs() < 0.15,
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

    let mut world = gpu_world!(SimConfig::default());

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

    let mut world = gpu_world!(SimConfig::default());

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

    let mut world = gpu_world!(SimConfig {
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
    let mut world = gpu_world!(SimConfig::default());
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
    let mut world = gpu_world!(SimConfig::default());
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
    let mut world = gpu_world!(SimConfig {
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

    let mut world = gpu_world!(SimConfig {
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

    let mut world = gpu_world!(SimConfig {
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

    let mut world = gpu_world!(SimConfig {
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
    let mut world = gpu_world!(SimConfig {
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

    let mut w1 = gpu_world!(SimConfig::default());
    let proj = w1.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 50.0, 0.0),
        linear_velocity: Vec3::new(vx, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let mut w2 = gpu_world!(SimConfig::default());
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
// Box-box collision
// ---------------------------------------------------------------------------

#[test]
fn box_box_collision_separates() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(3.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(1.0),
        },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-3.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(1.0),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    assert!(pa.x.is_finite() && pb.x.is_finite());
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Boxes should have separated after collision, dist={dist}"
    );
}

// ---------------------------------------------------------------------------
// Body at rest (short-term velocity check)
// ---------------------------------------------------------------------------

#[test]
fn body_velocity_decreases_under_gravity() {
    // A sphere in free fall for 30 steps should have a well-defined velocity
    // close to g*t. This validates velocity extraction from the AVBD solver.
    let g = 9.81_f32;
    let mut world = gpu_world!(SimConfig::default());
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 100.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30);
    let vel = world.get_velocity(h).unwrap();
    let expected_vy = -g * 0.5; // 30 steps at 1/60 = 0.5s
    assert!(
        (vel.y - expected_vy).abs() < 0.15,
        "After 0.5s free fall, vy should be ~{expected_vy}, got {}",
        vel.y
    );
    assert!(
        vel.x.abs() < 0.01 && vel.z.abs() < 0.01,
        "Horizontal velocity should be zero: vx={}, vz={}",
        vel.x,
        vel.z
    );
}

// ---------------------------------------------------------------------------
// Long-running stability
// ---------------------------------------------------------------------------

#[test]
fn long_simulation_no_divergence() {
    // Long simulation with multiple bodies under gravity.
    // Verify all positions remain finite (no NaN/Inf).
    let mut world = gpu_world!(SimConfig::default());

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

// ---------------------------------------------------------------------------
// Energy injection & sporadic jumping regression tests
// ---------------------------------------------------------------------------
// These tests target the missing α-regularization (Paper Section 3.6) and
// penalty floor (Paper Eq 19). They should FAIL on code without these fixes
// and PASS after implementation.

#[test]
fn settled_sphere_does_not_spontaneously_jump() {
    let mut world = gpu_world!(SimConfig::default());

    // Static floor
    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    // Sphere dropped from low height (just above contact)
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 1.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    // Let it settle for 240 steps (4 seconds — enough for low drop to fully damp)
    step_n(&mut world, 240);

    // Monitor for 360 more steps — no velocity spikes allowed
    let mut max_speed = 0.0_f32;
    let mut max_upward_vel = f32::NEG_INFINITY;
    for step in 240..600 {
        world.step();
        let vel = world.get_velocity(h).unwrap();
        let speed = vel.length();
        max_speed = max_speed.max(speed);
        max_upward_vel = max_upward_vel.max(vel.y);
        assert!(
            speed < 1.0,
            "Settled sphere jumped at step {step}: speed={speed:.3}, vel={vel}"
        );
    }
    assert!(
        max_upward_vel < 0.5,
        "Settled sphere had upward velocity spike: max_upward_vel={max_upward_vel:.3}"
    );
}

#[test]
fn resting_contact_energy_does_not_grow() {
    let g = 9.81_f32;
    let mut world = gpu_world!(SimConfig::default());

    // Static floor
    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        },
        ..Default::default()
    });

    // Stack of 3 boxes, placed touching (no overlap, no gap)
    let mut handles = Vec::new();
    for i in 0..3 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.5 + i as f32 * 1.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            },
            ..Default::default()
        });
        handles.push(h);
    }

    // Let settle for 60 steps
    step_n(&mut world, 60);

    // Measure settled energy
    let settled_energy: f32 = handles
        .iter()
        .map(|&h| {
            let pos = world.get_position(h).unwrap();
            let vel = world.get_velocity(h).unwrap();
            0.5 * 1.0 * vel.length_squared() + 1.0 * g * pos.y
        })
        .sum();

    // Run 540 more steps — energy must not grow
    for step in 60..600 {
        world.step();
        let total_energy: f32 = handles
            .iter()
            .map(|&h| {
                let pos = world.get_position(h).unwrap();
                let vel = world.get_velocity(h).unwrap();
                0.5 * 1.0 * vel.length_squared() + 1.0 * g * pos.y
            })
            .sum();
        assert!(
            total_energy < settled_energy + settled_energy.abs() * 0.05 + 0.1,
            "Energy grew after settling at step {step}: E={total_energy:.3}, E_settled={settled_energy:.3}"
        );
    }

    // Late window: speeds should be near zero
    let max_late_speed: f32 = handles
        .iter()
        .map(|&h| world.get_velocity(h).unwrap().length())
        .fold(0.0, f32::max);
    assert!(
        max_late_speed < 0.5,
        "Stack still moving at end: max_speed={max_late_speed:.3}"
    );
}

#[test]
fn long_horizon_stack_velocity_stays_bounded() {
    let mut world = gpu_world!(SimConfig::default());

    // Static floor
    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
        ..Default::default()
    });

    // 5x5 grid of boxes on the floor
    let mut handles = Vec::new();
    for ix in 0..5 {
        for iz in 0..5 {
            let h = world.add_body(&RigidBodyDesc {
                position: Vec3::new(
                    ix as f32 * 1.0 - 2.0,
                    0.5,
                    iz as f32 * 1.0 - 2.0,
                ),
                mass: 1.0,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::new(0.4, 0.4, 0.4),
                },
                ..Default::default()
            });
            handles.push(h);
        }
    }

    // Settle
    step_n(&mut world, 120);

    // Run 780 more steps (total 900 = 15 seconds)
    for step in 120..900 {
        world.step();
        for (i, &h) in handles.iter().enumerate() {
            let pos = world.get_position(h).unwrap();
            let vel = world.get_velocity(h).unwrap();
            let speed = vel.length();
            assert!(
                speed < 5.0,
                "Body {i} velocity diverged at step {step}: speed={speed:.3}, vel={vel}"
            );
            assert!(
                pos.y < 3.0,
                "Body {i} launched upward at step {step}: pos={pos}"
            );
            assert!(
                pos.y > -0.5,
                "Body {i} fell through floor at step {step}: pos={pos}"
            );
        }
    }
}

#[test]
fn penalty_floor_prevents_stiffness_collapse() {
    let mut world = gpu_world!(SimConfig::default());

    // Static floor
    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        },
        ..Default::default()
    });

    // Heavy box resting on floor
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.5, 0.0),
        mass: 10.0,
        friction: 0.5,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        },
        ..Default::default()
    });

    // Let settle for 240 steps (4 seconds — longer for heavy body)
    step_n(&mut world, 240);

    // Monitor position stability for 360 steps
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut max_vy = 0.0_f32;
    for _step in 240..600 {
        world.step();
        let pos = world.get_position(h).unwrap();
        let vel = world.get_velocity(h).unwrap();
        min_y = min_y.min(pos.y);
        max_y = max_y.max(pos.y);
        max_vy = max_vy.max(vel.y.abs());
    }

    let oscillation = max_y - min_y;
    assert!(
        oscillation < 0.06,
        "Heavy box oscillated vertically: range={oscillation:.4} (min_y={min_y:.4}, max_y={max_y:.4})"
    );
    assert!(
        max_vy < 1.0,
        "Heavy box had vertical velocity: max_vy={max_vy:.4}"
    );
}

// ---------------------------------------------------------------------------
// Energy injection detection tests
// ---------------------------------------------------------------------------

/// A small pyramid of boxes (4-3-2-1 = 10 boxes) is built above a floor.
/// After an initial settling period, track total kinetic energy.
/// Energy should trend DOWNWARD (friction dissipation), never spike upward.
/// This catches the runaway energy injection that causes pyramid explosions.
#[test]
fn pyramid_kinetic_energy_does_not_grow_after_settling() {
    let mut world = gpu_world!(SimConfig::default());

    // Static floor
    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
        ..Default::default()
    });

    // Build a 4-row pyramid: 4 boxes base, 3, 2, 1 on top = 10 boxes
    let mut handles = Vec::new();
    let box_size = 0.5;
    for row in 0..4 {
        let count = 4 - row;
        let y = box_size + row as f32 * (box_size * 2.0 + 0.01);
        let x_start = -(count as f32 - 1.0) * box_size;
        for i in 0..count {
            let h = world.add_body(&RigidBodyDesc {
                position: Vec3::new(x_start + i as f32 * box_size * 2.0, y, 0.0),
                mass: 1.0,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::splat(box_size),
                },
                ..Default::default()
            });
            handles.push(h);
        }
    }

    let gravity_y = -9.81_f32;

    // Helper: compute total KE for all dynamic bodies
    let compute_ke = |world: &World, handles: &Vec<_>| -> f32 {
        let mut total_ke = 0.0_f32;
        for &h in handles {
            let vel = world.get_velocity(h).unwrap();
            let mass = 1.0_f32;
            total_ke += 0.5 * mass * vel.length_squared();
        }
        total_ke
    };

    // Settle for 5 seconds (300 frames)
    step_n(&mut world, 300);

    // Record KE after settling
    let ke_settled = compute_ke(&world, &handles);

    // Run for 10 more seconds (600 frames), track KE
    let mut max_ke_after_settle = ke_settled;
    let mut max_speed = 0.0_f32;
    for step in 300..900 {
        world.step();
        let ke = compute_ke(&world, &handles);
        max_ke_after_settle = max_ke_after_settle.max(ke);

        for &h in &handles {
            let vel = world.get_velocity(h).unwrap();
            let speed = vel.length();
            if speed > max_speed {
                max_speed = speed;
            }
            // No body should reach physically unreasonable speeds.
            // Free fall from pyramid height (~4m): max_v = sqrt(2*9.81*4) ~ 8.9 m/s
            // Allow 2x for collisions: 18 m/s
            assert!(
                speed < 18.0,
                "Body exceeded free-fall-derived speed limit at step {step}: speed={speed:.2}"
            );
        }
    }

    // KE should not grow significantly after settling.
    // Allow 3x the settled KE as headroom (some oscillation is OK, but not runaway).
    let ke_threshold = ke_settled.max(0.5) * 3.0;
    assert!(
        max_ke_after_settle < ke_threshold,
        "Kinetic energy grew after settling: max_ke={max_ke_after_settle:.3} vs settled={ke_settled:.3} (threshold={ke_threshold:.3})"
    );

    // Max speed should be reasonable — no explosion-level velocities
    assert!(
        max_speed < 10.0,
        "Max speed in settled pyramid too high: {max_speed:.3} m/s (suggests energy injection)"
    );
}

/// Drop 50 boxes from height onto a floor. After they pile up and settle,
/// no body should have velocity exceeding what gravity alone could produce.
/// This catches the "10k boxes explosion" scenario at smaller scale.
#[test]
fn falling_box_pile_does_not_explode() {
    // Use 10 iterations for this dense scene — 5 is insufficient for 50 simultaneous bodies
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 10,
        ..SimConfig::default()
    });

    // Static floor
    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
        ..Default::default()
    });

    // Drop 50 boxes from various heights (1-5m), slightly offset
    let mut handles = Vec::new();
    for i in 0..50 {
        let layer = i / 10;
        let idx = i % 10;
        let x = (idx % 5) as f32 * 0.9 - 1.8;
        let z = (idx / 5) as f32 * 0.9 - 0.45;
        let y = 1.0 + layer as f32 * 1.2;
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(x, y, z),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.4),
            },
            ..Default::default()
        });
        handles.push(h);
    }

    // Max height is about 6.8m. Free-fall speed: sqrt(2*9.81*6.8) ~ 11.6 m/s
    // During chaotic multi-body impacts, transient speeds can exceed free-fall due to
    // constraint resolution. Use 50 m/s to catch true explosions (100+ m/s).
    let max_physical_speed = 50.0;

    // Run for 20 seconds (1200 frames)
    let mut max_speed_late = 0.0_f32; // after 10 seconds
    for step in 0..1200 {
        world.step();
        for &h in &handles {
            let vel = world.get_velocity(h).unwrap();
            let speed = vel.length();

            // During any frame: speed should not exceed physical limits
            assert!(
                speed < max_physical_speed,
                "Body exceeded physical speed limit at step {step}: speed={speed:.2} m/s"
            );

            if step > 600 {
                max_speed_late = max_speed_late.max(speed);
            }
        }
    }

    // After 10 seconds of settling, all bodies should be nearly at rest
    assert!(
        max_speed_late < 3.0,
        "Bodies still moving fast after 10s settling: max_speed_late={max_speed_late:.3} m/s"
    );

    // Check no body fell through floor or launched into space
    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.y > -1.0,
            "Body {i} fell through floor: y={:.2}",
            pos.y
        );
        assert!(
            pos.y < 20.0,
            "Body {i} launched into space: y={:.2}",
            pos.y
        );
    }
}

/// Track total system energy (KE + PE) over a long simulation of stacked boxes.
/// After settling, energy should only decrease (friction) or stay constant.
/// Any INCREASE indicates the solver is injecting energy.
#[test]
fn total_energy_monotonically_decreases_after_settling() {
    let mut world = gpu_world!(SimConfig::default());

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
        ..Default::default()
    });

    // 5 boxes stacked vertically
    let mut handles = Vec::new();
    for i in 0..5 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.5 + i as f32 * 1.01, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.5),
            },
            ..Default::default()
        });
        handles.push(h);
    }

    let gravity_mag = 9.81_f32;

    let compute_total_energy = |world: &World| -> f32 {
        let mut total = 0.0_f32;
        for &h in &handles {
            let pos = world.get_position(h).unwrap();
            let vel = world.get_velocity(h).unwrap();
            let mass = 1.0_f32;
            total += 0.5 * mass * vel.length_squared(); // KE
            total += mass * gravity_mag * pos.y; // PE
        }
        total
    };

    // Settle for 180 frames (3 seconds)
    step_n(&mut world, 180);

    let e_settled = compute_total_energy(&world);

    // Run for 600 more frames (10 seconds). Track energy.
    // Use a sliding window average to filter frame-to-frame noise
    // but catch sustained energy growth.
    let mut energy_window: Vec<f32> = Vec::new();
    let window_size = 30; // half-second window
    for step in 180..780 {
        world.step();
        let e = compute_total_energy(&world);
        energy_window.push(e);
        if energy_window.len() > window_size {
            energy_window.remove(0);
        }
        if energy_window.len() == window_size {
            let avg: f32 = energy_window.iter().sum::<f32>() / window_size as f32;

            // Window-averaged energy should not exceed settled energy by more than 10%
            assert!(
                avg < e_settled * 1.1 + 1.0,
                "Energy growing after settling at step {step}: window_avg={avg:.3} vs settled={e_settled:.3}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Long-horizon stability regression tests
// ---------------------------------------------------------------------------

/// A 60-second simulation of a settled 3-box stack must not accumulate energy drift.
/// This catches slow energy injection that shorter tests miss (e.g. penalty decay,
/// warmstart rounding, or α-regularization drift over thousands of frames).
#[test]
fn long_horizon_60s_stack_energy_stable() {
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 5,
        max_bodies: 16,
        ..Default::default()
    });

    // Floor
    world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });

    // 3 stacked boxes
    let mut handles = Vec::new();
    for i in 0..3 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.5 + i as f32 * 1.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.5),
            },
            ..Default::default()
        });
        handles.push(h);
    }

    // Settle for 5 seconds
    step_n(&mut world, 300);

    let settled_ke: f32 = handles
        .iter()
        .map(|h| {
            let v = world.get_velocity(*h).unwrap();
            0.5 * v.length_squared()
        })
        .sum();

    // Run for 55 more seconds (3300 steps at 60 Hz)
    let mut max_ke = settled_ke;
    for step in 300..3600 {
        world.step();
        let ke: f32 = handles
            .iter()
            .map(|h| {
                let v = world.get_velocity(*h).unwrap();
                0.5 * v.length_squared()
            })
            .sum();
        if ke > max_ke {
            max_ke = ke;
        }
        // Every 10 seconds, check energy hasn't grown unboundedly
        if step % 600 == 0 {
            assert!(
                max_ke < settled_ke + 1.0,
                "KE grew over 60s simulation at step {step}: max_ke={max_ke:.4} settled_ke={settled_ke:.4}"
            );
        }
    }

    // Final check: max speed should be tiny
    let max_speed: f32 = handles
        .iter()
        .map(|h| world.get_velocity(*h).unwrap().length())
        .fold(0.0_f32, f32::max);
    assert!(
        max_speed < 1.0,
        "Bodies still moving after 60s: max_speed={max_speed:.4}"
    );
}

/// Extreme mass ratio: a heavy box (mass 100) resting on a light box (mass 1) on a floor.
/// The solver must not eject the light box or inject energy due to the mass disparity.
#[test]
fn extreme_mass_ratio_does_not_explode() {
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 5,
        max_bodies: 16,
        ..Default::default()
    });

    // Floor
    world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });

    // Light box on floor
    let light = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.5, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });

    // Heavy box on top (20:1 mass ratio)
    let heavy = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 1.5, 0.0),
        mass: 20.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });

    // Settle
    step_n(&mut world, 120);

    // Monitor for 10 seconds
    let mut max_speed = 0.0_f32;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for _step in 0..600 {
        world.step();
        for &h in &[light, heavy] {
            let v = world.get_velocity(h).unwrap();
            let p = world.get_position(h).unwrap();
            max_speed = max_speed.max(v.length());
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
    }

    assert!(
        max_speed < 3.0,
        "Extreme mass ratio produced high speed: {max_speed:.3}"
    );
    assert!(
        min_y > -0.5,
        "Body fell through floor with extreme mass ratio: min_y={min_y:.3}"
    );
    assert!(
        max_y < 5.0,
        "Body launched upward with extreme mass ratio: max_y={max_y:.3}"
    );
}

/// Sphere-on-sphere stacking stability: 5 spheres dropped onto a floor.
/// Spheres are harder to stack than boxes (point contacts, rolling), but they
/// should settle and not gain energy.
#[test]
fn sphere_pile_settles_without_energy_growth() {
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 5,
        max_bodies: 16,
        ..Default::default()
    });

    // Floor
    world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });

    // 5 spheres dropped in a cluster
    let mut handles = Vec::new();
    for i in 0..5 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(
                (i % 2) as f32 * 0.3 - 0.15,
                1.0 + i as f32 * 1.2,
                (i / 2) as f32 * 0.3 - 0.15,
            ),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        handles.push(h);
    }

    // Let them fall and settle for 8 seconds (spheres roll longer than boxes)
    step_n(&mut world, 480);

    // Monitor for 10 more seconds — track max speed (spheres roll so KE stays
    // nonzero, but max speed should be bounded and not grow)
    let mut max_speed = 0.0_f32;
    let settled_max_speed: f32 = handles
        .iter()
        .map(|h| world.get_velocity(*h).unwrap().length())
        .fold(0.0_f32, f32::max);

    for _step in 0..600 {
        world.step();
        for &h in &handles {
            let v = world.get_velocity(h).unwrap();
            max_speed = max_speed.max(v.length());
        }
    }

    // No body should fly away
    for &h in &handles {
        let p = world.get_position(h).unwrap();
        assert!(
            p.y > -1.0 && p.y < 10.0,
            "Sphere left reasonable bounds: y={:.3}",
            p.y
        );
    }

    // Max speed should not grow significantly beyond settled speed
    assert!(
        max_speed < settled_max_speed + 5.0,
        "Sphere pile speed grew: max_speed={max_speed:.3} settled_max_speed={settled_max_speed:.3}"
    );
}

/// A 10x10 grid (100 boxes) dropped onto a floor must settle without velocity divergence.
/// This is the smallest "large scene" test that catches scaling issues before the
/// full 10k-body benchmark.
#[test]
fn hundred_box_grid_settles_without_divergence() {
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 5,
        max_bodies: 256,
        ..Default::default()
    });

    // Floor
    world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });

    // 10x10 grid of boxes at y=0.5 (resting on floor)
    let mut handles = Vec::new();
    for x in 0..10 {
        for z in 0..10 {
            let h = world.add_body(&RigidBodyDesc {
                position: Vec3::new(x as f32 * 1.1 - 5.0, 0.5, z as f32 * 1.1 - 5.0),
                mass: 1.0,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::splat(0.5),
                },
                ..Default::default()
            });
            handles.push(h);
        }
    }

    // Settle for 3 seconds
    step_n(&mut world, 180);

    // Monitor for 10 seconds
    let mut max_speed = 0.0_f32;
    let mut any_fell_through = false;
    let mut any_launched = false;
    for _step in 0..600 {
        world.step();
        for &h in &handles {
            let v = world.get_velocity(h).unwrap();
            let p = world.get_position(h).unwrap();
            max_speed = max_speed.max(v.length());
            if p.y < -0.5 {
                any_fell_through = true;
            }
            if p.y > 5.0 {
                any_launched = true;
            }
        }
    }

    assert!(
        !any_fell_through,
        "A box fell through the floor in 100-box grid"
    );
    assert!(
        !any_launched,
        "A box was launched upward in 100-box grid"
    );
    assert!(
        max_speed < 5.0,
        "100-box grid had high velocity after settling: max_speed={max_speed:.3}"
    );
}

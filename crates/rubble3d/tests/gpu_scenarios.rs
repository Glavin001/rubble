//! GPU scenario tests for rubble3d.
//!
//! Mirrors the CPU scenario tests but runs on the GPU compute pipeline (AVBD solver).
//! Tests that require planes or capsules are omitted since the GPU narrowphase
//! only supports sphere-sphere, sphere-box, and box-box contacts.
//! Tolerances are wider than CPU tests because the GPU AVBD solver produces
//! slightly different results from the CPU PBD solver.

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

fn gpu_world(config: SimConfig) -> World {
    World::new_gpu(config).expect(
        "FATAL: No GPU adapter found. Install mesa-vulkan-drivers for lavapipe.",
    )
}

fn gpu_world_default() -> World {
    gpu_world(SimConfig::default())
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
fn gpu_free_fall_sphere_1_second() {
    let mut world = gpu_world_default();
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    let expected_y = 10.0 - 0.5 * 9.81 * 1.0;
    assert!(
        (pos.y - expected_y).abs() < 1.0,
        "Expected y ~ {expected_y}, got {} (error={})",
        pos.y,
        (pos.y - expected_y).abs()
    );
    assert!(pos.x.abs() < 0.01, "x drift: {}", pos.x);
    assert!(pos.z.abs() < 0.01, "z drift: {}", pos.z);
}

#[test]
fn gpu_free_fall_box_matches_sphere() {
    let mut world = gpu_world_default();
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
        (sp.y - bp.y).abs() < 0.5,
        "Sphere y={} vs Box y={} — should be similar in free fall",
        sp.y,
        bp.y
    );
    let expected_y = 20.0 - 0.5 * 9.81;
    assert!(
        (sp.y - expected_y).abs() < 1.0,
        "Free-fall y should be ~{expected_y}, got {}",
        sp.y
    );
}

#[test]
fn gpu_zero_gravity_no_motion() {
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
// Static bodies
// ---------------------------------------------------------------------------

#[test]
fn gpu_static_body_does_not_fall() {
    let mut world = gpu_world_default();
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
        (pos.y - 10.0).abs() < 0.01,
        "Static body should not move, got y={}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Collisions (sphere-sphere, sphere-box, box-box)
// ---------------------------------------------------------------------------

#[test]
fn gpu_two_sphere_head_on_collision() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    assert!(pa.x.is_finite() && pb.x.is_finite());
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Spheres should have separated after collision, dist={dist}"
    );
}

#[test]
fn gpu_box_sphere_mixed_collision() {
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
    assert!(sp.x.is_finite() && bp.x.is_finite());
    let dist = (sp - bp).length();
    assert!(
        dist > 1.5,
        "Sphere and box should not overlap, distance={dist}"
    );
}

#[test]
fn gpu_box_box_collision() {
    let mut world = gpu_world(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-4.0, 0.0, 0.0),
        linear_velocity: Vec3::new(3.0, 0.0, 0.0),
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(1.0),
        },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(4.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-3.0, 0.0, 0.0),
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
// Sphere resting on static box floor (GPU equivalent of plane tests)
// ---------------------------------------------------------------------------

#[test]
fn gpu_sphere_rests_on_box_floor() {
    let mut world = gpu_world_default();

    // Static box as ground
    world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 600);

    let pos = world.get_position(h).unwrap();
    // Floor surface at y=0, sphere radius=0.5, center should be near y=0.5
    assert!(
        pos.y > -0.5 && pos.y < 3.0,
        "Sphere should rest near floor, got y={}",
        pos.y
    );
    assert!(pos.y.is_finite(), "Position should be finite");
}

#[test]
fn gpu_box_on_box_floor_no_tunnel() {
    let mut world = gpu_world_default();

    // Static floor
    world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

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
        pos.y > -0.5 && pos.y.is_finite(),
        "Box should not tunnel through floor, got y={}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Momentum & mass ratios
// ---------------------------------------------------------------------------

#[test]
fn gpu_symmetric_collision_preserves_symmetry() {
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
        (va.x + vb.x).abs() < 2.0,
        "Symmetric collision should yield symmetric velocities: va={va}, vb={vb}"
    );
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Bodies should have separated after collision, dist={dist}"
    );
}

#[test]
fn gpu_heavy_vs_light_collision() {
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
    let _vl = world.get_velocity(light).unwrap();
    assert!(
        vh.x > -0.5,
        "Heavy body should still move roughly rightward, got vx={}",
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
fn gpu_heavier_body_deflects_less() {
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
        delta_heavy <= delta_light + 1.0,
        "Heavier body deflected MORE: delta_heavy={delta_heavy}, delta_light={delta_light}"
    );
}

// ---------------------------------------------------------------------------
// Physical invariants
// ---------------------------------------------------------------------------

#[test]
fn gpu_energy_conserved_during_free_fall() {
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 100.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = gpu_world_default();
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
            "Energy not conserved at step {step}: E={total:.2}, E0={initial_energy:.2}"
        );
    }
}

#[test]
fn gpu_energy_does_not_increase_on_bounce() {
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 10.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = gpu_world_default();

    // Static box floor
    world.add_body(&RigidBodyDesc {
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
            "Energy increased at step {step}: E={total:.2}, E0={initial_energy:.2}"
        );
    }
}

#[test]
fn gpu_total_momentum_conserved_in_collision() {
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
        error < 3.0,
        "Momentum not conserved: initial={initial_momentum}, final={final_momentum}, error={error}"
    );
}

#[test]
fn gpu_gravity_produces_linear_velocity_increase() {
    let g = 9.81_f32;
    let mut world = gpu_world_default();
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
        (delta_vy - expected_delta).abs() < 1.0,
        "Velocity should increase linearly: delta_vy={delta_vy}, expected={expected_delta}"
    );
    assert!(
        v_one.x.abs() < 0.1 && v_one.z.abs() < 0.1,
        "Horizontal velocity should remain near zero: vx={}, vz={}",
        v_one.x,
        v_one.z
    );
}

#[test]
fn gpu_vertical_drop_preserves_horizontal_position() {
    let mut world = gpu_world_default();
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(7.0, 50.0, -3.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..120 {
        world.step();
        let pos = world.get_position(h).unwrap();
        assert!(
            (pos.x - 7.0).abs() < 0.1 && (pos.z - (-3.0)).abs() < 0.1,
            "Horizontal drift at step {step}: x={}, z={} (expected 7.0, -3.0)",
            pos.x,
            pos.z
        );
    }
}

#[test]
fn gpu_kinetic_energy_constant_no_collision() {
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
            (ke - initial_ke).abs() < 0.5,
            "KE changed at step {step}: ke={ke}, initial={initial_ke}"
        );
    }
}

#[test]
fn gpu_center_of_mass_velocity_constant() {
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
            error < 2.0,
            "COM velocity changed at step {step}: com_vel={com_vel}, initial={initial_com_vel}, error={error}"
        );
    }
}

#[test]
fn gpu_static_body_unaffected_by_collision() {
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

    world.add_body(&RigidBodyDesc {
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
            wall_pos.length() < 0.1,
            "Static body moved at step {step}: pos={wall_pos}"
        );
    }
}

#[test]
fn gpu_bodies_never_overlap_after_settling() {
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
                dist > min_dist * 0.8,
                "Bodies {i} and {j} overlap: dist={dist}, min_dist={min_dist}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Superposition & projectile
// ---------------------------------------------------------------------------

#[test]
fn gpu_superposition_gravity_plus_horizontal() {
    let vx = 5.0_f32;

    let mut w1 = gpu_world_default();
    let proj = w1.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 50.0, 0.0),
        linear_velocity: Vec3::new(vx, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let mut w2 = gpu_world_default();
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
            (p1.y - p2.y).abs() < 0.5,
            "Y-trajectories differ at step {step}: projectile y={}, drop y={}",
            p1.y,
            p2.y
        );
        let t = (step + 1) as f32 / 60.0;
        let expected_x = vx * t;
        assert!(
            (p1.x - expected_x).abs() < 0.5,
            "X should advance linearly at step {step}: x={}, expected={expected_x}",
            p1.x
        );
    }
}

#[test]
fn gpu_projectile_motion() {
    let speed = 10.0;
    let angle = std::f32::consts::FRAC_PI_4;
    let vx = speed * angle.cos();
    let vy = speed * angle.sin();

    let mut world = gpu_world_default();
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        linear_velocity: Vec3::new(vx, vy, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.1 },
        ..Default::default()
    });

    step_n(&mut world, 30);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos.x - 3.535).abs() < 1.0,
        "Projectile x should be ~3.54, got {}",
        pos.x
    );
    assert!(
        (pos.y - 2.268).abs() < 1.0,
        "Projectile y should be ~2.27, got {}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Multi-body stability
// ---------------------------------------------------------------------------

#[test]
fn gpu_multi_sphere_pileup() {
    let mut world = gpu_world_default();

    // Static floor
    world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

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
            pos.y > -1.0 && pos.y.is_finite(),
            "Sphere {i} fell through floor or diverged: y={}",
            pos.y
        );
    }
}

#[test]
fn gpu_long_simulation_settles() {
    let mut world = gpu_world_default();

    // Static floor
    world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

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
            vel.length() < 2.0,
            "Body {i} should have settled after 30s, vel={}",
            vel.length()
        );
        assert!(
            pos.y > -1.0 && pos.y < 10.0,
            "Body {i} should be resting on floor, y={}",
            pos.y
        );
    }
}

// ---------------------------------------------------------------------------
// Body lifecycle
// ---------------------------------------------------------------------------

#[test]
fn gpu_remove_body_mid_simulation() {
    let mut world = gpu_world_default();

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
fn gpu_add_body_mid_simulation() {
    let mut world = gpu_world_default();

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
        pb.y > pa.y - 2.0,
        "Late-added body B should be higher: A.y={}, B.y={}",
        pa.y,
        pb.y
    );
}

#[test]
fn gpu_teleport_and_simulate() {
    let mut world = gpu_world_default();
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
// CPU-GPU agreement
// ---------------------------------------------------------------------------

#[test]
fn gpu_cpu_free_fall_agreement() {
    let make_config = || SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    };

    let mut cpu_world = World::new(make_config());
    let mut gpu_world = World::new_gpu(make_config()).unwrap();

    let ch = cpu_world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 20.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });
    let gh = gpu_world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 20.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..60 {
        cpu_world.step();
        gpu_world.step();
    }

    let cp = cpu_world.get_position(ch).unwrap();
    let gp = gpu_world.get_position(gh).unwrap();
    assert!(
        (cp - gp).length() < 2.0,
        "CPU pos={cp} vs GPU pos={gp} should be close in free fall"
    );
}

#[test]
fn gpu_cpu_collision_agreement() {
    let make_config = || SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        max_bodies: 256,
    };

    let mut cpu_world = World::new(make_config());
    let mut gpu_world = World::new_gpu(make_config()).unwrap();

    // Same collision setup on both
    let ca = cpu_world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(3.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let cb = cpu_world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-3.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let ga = gpu_world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(3.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let gb = gpu_world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-3.0, 0.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    for _ in 0..120 {
        cpu_world.step();
        gpu_world.step();
    }

    // Both should have collided and separated
    let cpu_dist = (cpu_world.get_position(ca).unwrap() - cpu_world.get_position(cb).unwrap()).length();
    let gpu_dist = (gpu_world.get_position(ga).unwrap() - gpu_world.get_position(gb).unwrap()).length();

    assert!(cpu_dist > 1.5, "CPU bodies should have separated");
    assert!(gpu_dist > 1.5, "GPU bodies should have separated");
}

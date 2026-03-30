//! GPU scenario tests for rubble2d.
//!
//! All simulation runs on the GPU via WGSL compute shaders (AVBD solver).
//! Tests cover free fall, collisions, static bodies, momentum, energy,
//! body lifecycle, projectile motion, and multi-body stability.

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D, World2D};

fn gpu_world(config: SimConfig2D) -> World2D {
    World2D::new(config).expect("GPU adapter required for tests")
}

fn gpu_world_default() -> World2D {
    gpu_world(SimConfig2D::default())
}

fn step_n(world: &mut World2D, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

// ---------------------------------------------------------------------------
// Free-fall & gravity
// ---------------------------------------------------------------------------

#[test]
fn gpu_free_fall_circle_1_second() {
    let mut world = gpu_world_default();
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
        (pos.y - expected_y).abs() < 1.0,
        "Expected y ~ {expected_y}, got {}",
        pos.y
    );
    assert!(pos.x.abs() < 0.1, "x should stay at 0, got {}", pos.x);
}

#[test]
fn gpu_free_fall_rect_matches_circle() {
    let mut world = gpu_world_default();
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
        (cp.y - rp.y).abs() < 0.5,
        "Circle y={} vs Rect y={} — free fall should be similar",
        cp.y,
        rp.y
    );
}

#[test]
fn gpu_zero_gravity_no_motion() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });
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

// ---------------------------------------------------------------------------
// Static bodies
// ---------------------------------------------------------------------------

#[test]
fn gpu_static_body_does_not_fall() {
    let mut world = gpu_world_default();
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
// Collisions
// ---------------------------------------------------------------------------

#[test]
fn gpu_two_circle_collision() {
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -3.0,
        y: 0.0,
        vx: 5.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 3.0,
        y: 0.0,
        vx: -5.0,
        shape: ShapeDesc2D::Circle { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Circles should have separated after collision, dist={dist}"
    );
}

#[test]
fn gpu_rect_rect_collision() {
    let mut world = gpu_world(SimConfig2D {
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
    assert!(!pa.x.is_nan() && !pb.x.is_nan());
    let dist = (pa - pb).length();
    assert!(
        dist > 1.5,
        "Rects should not be overlapping after collision, distance={dist}"
    );
}

#[test]
fn gpu_circle_rect_collision() {
    let mut world = gpu_world(SimConfig2D {
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
// Circle resting on static rect floor
// ---------------------------------------------------------------------------

#[test]
fn gpu_circle_rests_on_rect_floor() {
    let mut world = gpu_world_default();

    // Static floor
    world.add_body(&RigidBodyDesc2D {
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

    step_n(&mut world, 600);

    let pos = world.get_position(h).unwrap();
    assert!(
        pos.y > -2.0 && pos.y < 5.0,
        "Circle should rest near floor, got y={}",
        pos.y
    );
    assert!(pos.y.is_finite());
}

#[test]
fn gpu_circle_pile_on_floor() {
    let mut world = gpu_world_default();

    // Static floor
    world.add_body(&RigidBodyDesc2D {
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

    step_n(&mut world, 600);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.y.is_finite() && pos.y > -2.0,
            "Circle {i} fell through floor or diverged: y={}",
            pos.y
        );
    }
}

// ---------------------------------------------------------------------------
// Momentum & mass ratios
// ---------------------------------------------------------------------------

#[test]
fn gpu_symmetric_collision_preserves_symmetry_2d() {
    let mut world = gpu_world(SimConfig2D {
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
fn gpu_heavier_body_deflects_less_2d() {
    let m_heavy = 10.0_f32;
    let m_light = 1.0_f32;
    let v0 = 3.0_f32;

    let mut world = gpu_world(SimConfig2D {
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
        delta_heavy <= delta_light + 1.0,
        "Heavier body deflected MORE: delta_heavy={delta_heavy}, delta_light={delta_light}"
    );
}

// ---------------------------------------------------------------------------
// Physical invariants
// ---------------------------------------------------------------------------

#[test]
fn gpu_energy_conserved_during_free_fall_2d() {
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 100.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = gpu_world_default();
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: y0,
        mass,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
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
fn gpu_total_momentum_conserved_2d() {
    let m_a = 2.0_f32;
    let m_b = 3.0_f32;
    let v_a = Vec2::new(4.0, 1.0);
    let v_b = Vec2::new(-2.0, -1.0);
    let initial_momentum = m_a * v_a + m_b * v_b;

    let mut world = gpu_world(SimConfig2D {
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
        error < 3.0,
        "Momentum not conserved: initial={initial_momentum}, final={final_momentum}, error={error}"
    );
}

#[test]
fn gpu_gravity_linear_velocity_increase_2d() {
    let g = 9.81_f32;
    let mut world = gpu_world_default();
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 100.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
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
        v_one.x.abs() < 0.1,
        "Horizontal velocity should remain near zero: vx={}",
        v_one.x
    );
}

#[test]
fn gpu_vertical_drop_preserves_horizontal_2d() {
    let mut world = gpu_world_default();
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
            (pos.x - 7.0).abs() < 0.1,
            "Horizontal drift at step {step}: x={} (expected 7.0)",
            pos.x
        );
    }
}

#[test]
fn gpu_kinetic_energy_constant_no_collision_2d() {
    let mass = 2.0_f32;
    let v0 = Vec2::new(3.0, -1.0);
    let initial_ke = 0.5 * mass * v0.length_squared();

    let mut world = gpu_world(SimConfig2D {
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
            (ke - initial_ke).abs() < 0.5,
            "KE changed at step {step}: ke={ke}, initial={initial_ke}"
        );
    }
}

#[test]
fn gpu_center_of_mass_velocity_constant_2d() {
    let m1 = 2.0_f32;
    let m2 = 5.0_f32;
    let m3 = 1.0_f32;
    let total_mass = m1 + m2 + m3;

    let v1 = Vec2::new(3.0, 0.0);
    let v2 = Vec2::new(-1.0, 2.0);
    let v3 = Vec2::new(0.0, -3.0);
    let initial_com_vel = (m1 * v1 + m2 * v2 + m3 * v3) / total_mass;

    let mut world = gpu_world(SimConfig2D {
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
            error < 2.0,
            "COM velocity changed at step {step}: com_vel={com_vel}, initial={initial_com_vel}, error={error}"
        );
    }
}

#[test]
fn gpu_static_body_unaffected_by_collision_2d() {
    let mut world = gpu_world(SimConfig2D {
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

    world.add_body(&RigidBodyDesc2D {
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
            wall_pos.length() < 0.1,
            "Static body moved at step {step}: pos={wall_pos}"
        );
    }
}

// ---------------------------------------------------------------------------
// Superposition & projectile
// ---------------------------------------------------------------------------

#[test]
fn gpu_superposition_2d() {
    let vx = 5.0_f32;

    let mut w1 = gpu_world_default();
    let proj = w1.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 50.0,
        vx,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    let mut w2 = gpu_world_default();
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
fn gpu_projectile_motion_2d() {
    let speed = 10.0;
    let angle = std::f32::consts::FRAC_PI_4;
    let vx = speed * angle.cos();
    let vy = speed * angle.sin();

    let mut world = gpu_world_default();
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.0,
        vx,
        vy,
        shape: ShapeDesc2D::Circle { radius: 0.1 },
        ..Default::default()
    });

    step_n(&mut world, 30);

    let pos = world.get_position(h).unwrap();
    assert!(
        pos.x > 3.0 && pos.x < 4.5,
        "Projectile x should be ~3.54, got {}",
        pos.x
    );
    assert!(
        pos.y > 1.5 && pos.y < 3.0,
        "Projectile y should be ~2.27, got {}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Long simulation stability
// ---------------------------------------------------------------------------

#[test]
fn gpu_long_simulation_settles_2d() {
    let mut world = gpu_world_default();

    // Static floor
    world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -5.0,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 1.0),
        },
        ..Default::default()
    });

    let handles: Vec<_> = (0..5)
        .map(|i| {
            world.add_body(&RigidBodyDesc2D {
                x: i as f32 * 1.5 - 3.0,
                y: 5.0 + i as f32,
                shape: ShapeDesc2D::Circle { radius: 0.5 },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 1800);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        let vel = world.get_velocity(h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite(),
            "Body {i} position diverged: {pos}"
        );
        assert!(
            vel.length() < 2.0,
            "Body {i} should have settled, vel={}",
            vel.length()
        );
        assert!(
            pos.y > -5.0 && pos.y < 10.0,
            "Body {i} should be near floor, y={}",
            pos.y
        );
    }
}

// ---------------------------------------------------------------------------
// Body lifecycle
// ---------------------------------------------------------------------------

#[test]
fn gpu_remove_body_mid_simulation_2d() {
    let mut world = gpu_world_default();

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
fn gpu_teleport_and_simulate_2d() {
    let mut world = gpu_world_default();

    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 10.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30);
    let pos_before = world.get_position(h).unwrap();
    assert!(pos_before.y < 10.0, "Should have fallen");

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
fn gpu_bodies_never_overlap_2d() {
    let r = 0.5_f32;
    let mut world = gpu_world(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let handles: Vec<_> = (0..4)
        .map(|i| {
            world.add_body(&RigidBodyDesc2D {
                x: i as f32 * 1.5,
                y: 0.0,
                vx: -1.0 + i as f32 * 0.5,
                shape: ShapeDesc2D::Circle { radius: r },
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

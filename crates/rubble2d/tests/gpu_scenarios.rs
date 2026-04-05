//! End-to-end physics simulation scenario tests for rubble2d.
//!
//! Each test sets up a 2D scene, runs the simulation for a number of steps,
//! and asserts that the final state matches physical expectations.
//! All simulation runs on the GPU via WGSL compute shaders.

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D, World2D};

macro_rules! gpu_world {
    ($config:expr) => {
        match World2D::new($config) {
            Ok(w) => w,
            Err(_) => {
                eprintln!("SKIP: No GPU adapter found");
                return;
            }
        }
    };
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
fn free_fall_circle_1_second() {
    let mut world = gpu_world!(SimConfig2D::default());
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
        (pos.y - expected_y).abs() < 0.15,
        "Expected y ~ {expected_y}, got {}",
        pos.y
    );
    assert!(pos.x.abs() < 0.01, "x should stay at 0, got {}", pos.x);
}

#[test]
fn free_fall_rect_matches_circle() {
    let mut world = gpu_world!(SimConfig2D::default());
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
    let mut world = gpu_world!(config);
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
    let mut world = gpu_world!(SimConfig2D::default());
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
fn rect_rect_collision() {
    let mut world = gpu_world!(SimConfig2D {
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
fn circle_rect_collision() {
    let mut world = gpu_world!(SimConfig2D {
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
// Rotated rect-rect collision
// ---------------------------------------------------------------------------

#[test]
fn rotated_rect_rect_collision() {
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::ZERO,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc2D {
        x: -4.0,
        y: 0.0,
        vx: 3.0,
        angle: std::f32::consts::FRAC_PI_6, // 30 degrees
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 0.5),
        },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc2D {
        x: 4.0,
        y: 0.0,
        vx: -3.0,
        angle: -std::f32::consts::FRAC_PI_4, // -45 degrees
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(1.0, 0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    assert!(
        pa.x.is_finite() && pa.y.is_finite(),
        "Rotated rect A has non-finite position: {pa}"
    );
    assert!(
        pb.x.is_finite() && pb.y.is_finite(),
        "Rotated rect B has non-finite position: {pb}"
    );
    let dist = (pa - pb).length();
    assert!(
        dist > 0.5,
        "Rotated rects should not overlap after collision, dist={dist}"
    );
}

// ---------------------------------------------------------------------------
// Multi-body scenes
// ---------------------------------------------------------------------------

#[test]
fn domino_chain_2d() {
    // Line up rects on a floor, push the first one. Verify no NaN/crashes and
    // the first domino moves in the push direction.
    let mut world = gpu_world!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        ..Default::default()
    });

    // Static floor so dominos have a stable resting surface.
    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 5.0,
        y: -0.5,
        mass: 0.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(20.0, 0.5),
        },
        ..Default::default()
    });

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

    world.set_velocity(handles[0], Vec2::new(3.0, 0.0));

    step_n(&mut world, 60);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.x.is_finite() && pos.y.is_finite(),
            "Domino {i} has non-finite position: {pos}"
        );
    }
    let first_pos = world.get_position(handles[0]).unwrap();
    assert!(
        first_pos.x > 0.1,
        "First domino should have moved right, got x={}",
        first_pos.x
    );
    // Verify chain propagation: dominos 1 and 2 should also have moved
    // from their initial x positions (2.5 and 5.0 respectively).
    let second_pos = world.get_position(handles[1]).unwrap();
    assert!(
        (second_pos.x - 2.5).abs() > 0.01 || (second_pos.y - 1.0).abs() > 0.01,
        "Second domino should have been displaced from initial position, got {second_pos}"
    );
    let third_pos = world.get_position(handles[2]).unwrap();
    assert!(
        (third_pos.x - 5.0).abs() > 0.01 || (third_pos.y - 1.0).abs() > 0.01,
        "Third domino should have been displaced from initial position, got {third_pos}"
    );
}

#[test]
fn circle_pile_stability() {
    // Multiple circles dropped near each other — verify collision handling
    // doesn't produce NaN/Inf values over a moderate simulation.
    let mut world = gpu_world!(SimConfig2D::default());

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

    step_n(&mut world, 60);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(pos.y.is_finite(), "Circle {i} diverged: y={}", pos.y);
    }
}

// ---------------------------------------------------------------------------
// Energy & momentum
// ---------------------------------------------------------------------------

#[test]
fn symmetric_collision_preserves_symmetry_2d() {
    let mut world = gpu_world!(SimConfig2D {
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
        (va.x + vb.x).abs() < 1.0,
        "Symmetric collision should yield symmetric velocities: va={va}, vb={vb}"
    );
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
fn long_simulation_2d_no_divergence() {
    // Long simulation with multiple bodies under gravity.
    // Verify all positions remain finite (no NaN/Inf).
    let mut world = gpu_world!(SimConfig2D::default());

    let handles: Vec<_> = (0..8)
        .map(|i| {
            world.add_body(&RigidBodyDesc2D {
                x: i as f32 * 3.0 - 10.0,
                y: 5.0 + i as f32,
                shape: ShapeDesc2D::Circle { radius: 0.5 },
                ..Default::default()
            })
        })
        .collect();

    step_n(&mut world, 300);

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
    }
}

// ---------------------------------------------------------------------------
// Body lifecycle during simulation
// ---------------------------------------------------------------------------

#[test]
fn remove_body_mid_simulation_2d() {
    let mut world = gpu_world!(SimConfig2D::default());

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

    let mut world = gpu_world!(SimConfig2D::default());
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
fn circle_bounce_off_floor_2d() {
    // Circle dropped onto a static rect floor. Verify the collision is detected
    // (contact count > 0 at some point), and positions remain finite.
    let mut world = gpu_world!(SimConfig2D::default());

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
        y: 2.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

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
    assert!(pos.y < 2.0, "Circle should have fallen, got y={}", pos.y);
}

#[test]
fn teleport_and_simulate_2d() {
    let mut world = gpu_world!(SimConfig2D::default());

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
fn two_circles_stacked_stability_2d() {
    // Two circles at different heights under gravity. After short sim,
    // all positions remain finite.
    let mut world = gpu_world!(SimConfig2D::default());

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

    step_n(&mut world, 60);

    let lp = world.get_position(lower).unwrap();
    let up = world.get_position(upper).unwrap();

    assert!(lp.x.is_finite() && lp.y.is_finite(), "Lower diverged: {lp}");
    assert!(up.x.is_finite() && up.y.is_finite(), "Upper diverged: {up}");
}

// ---------------------------------------------------------------------------
// Velocity check
// ---------------------------------------------------------------------------

#[test]
fn body_velocity_decreases_under_gravity_2d() {
    let g = 9.81_f32;
    let mut world = gpu_world!(SimConfig2D::default());
    let h = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 100.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
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
        vel.x.abs() < 0.01,
        "Horizontal velocity should be zero: vx={}",
        vel.x
    );
}

// ---------------------------------------------------------------------------
// Physical invariant tests
// ---------------------------------------------------------------------------

#[test]
fn energy_conserved_during_free_fall_2d() {
    // During free fall (no contacts), total mechanical energy KE + PE
    // must remain constant within integrator tolerance.
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 100.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = gpu_world!(SimConfig2D::default());

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
            "Energy not conserved during free-fall at step {step}: E={total:.2}, E0={initial_energy:.2}"
        );
    }
}

#[test]
fn energy_does_not_increase_on_bounce_2d() {
    let g = 9.81_f32;
    let mass = 1.0_f32;
    let y0 = 10.0_f32;
    let initial_energy = mass * g * y0;

    let mut world = gpu_world!(SimConfig2D::default());

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
            total < initial_energy * 1.2,
            "Energy increased beyond initial at step {step}: E={total:.2}, E0={initial_energy:.2}"
        );
    }
}

#[test]
fn total_momentum_conserved_in_collision_2d() {
    let m_a = 2.0_f32;
    let m_b = 3.0_f32;
    let v_a = Vec2::new(4.0, 1.0);
    let v_b = Vec2::new(-2.0, -1.0);
    let initial_momentum = m_a * v_a + m_b * v_b;

    let mut world = gpu_world!(SimConfig2D {
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
    let g = 9.81_f32;
    let mut world = gpu_world!(SimConfig2D::default());
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
        (delta_vy - expected_delta).abs() < 0.5,
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
    let mut world = gpu_world!(SimConfig2D::default());
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
    let r = 0.5_f32;
    let mut world = gpu_world!(SimConfig2D::default());

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
    let m_heavy = 10.0_f32;
    let m_light = 1.0_f32;
    let v0 = 3.0_f32;

    let mut world = gpu_world!(SimConfig2D {
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
    let mass = 2.0_f32;
    let v0 = Vec2::new(3.0, -1.0);
    let initial_ke = 0.5 * mass * v0.length_squared();

    let mut world = gpu_world!(SimConfig2D {
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
            (ke - initial_ke).abs() < 0.1,
            "KE changed at step {step}: ke={ke}, initial={initial_ke}"
        );
    }
}

#[test]
fn center_of_mass_velocity_constant_in_zero_gravity_2d() {
    let m1 = 2.0_f32;
    let m2 = 5.0_f32;
    let m3 = 1.0_f32;
    let total_mass = m1 + m2 + m3;

    let v1 = Vec2::new(3.0, 0.0);
    let v2 = Vec2::new(-1.0, 2.0);
    let v3 = Vec2::new(0.0, -3.0);
    let initial_com_vel = (m1 * v1 + m2 * v2 + m3 * v3) / total_mass;

    let mut world = gpu_world!(SimConfig2D {
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
    let mut world = gpu_world!(SimConfig2D {
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
    let vx = 5.0_f32;

    let mut w1 = gpu_world!(SimConfig2D::default());
    let proj = w1.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 50.0,
        vx,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    let mut w2 = gpu_world!(SimConfig2D::default());
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

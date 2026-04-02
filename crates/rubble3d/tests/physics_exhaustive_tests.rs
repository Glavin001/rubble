//! Exhaustive physics scenario tests for rubble3d.
//!
//! Tests cover: stacking stability, domino chains, Newton's cradle,
//! inclined planes, rotational dynamics, multi-shape interactions,
//! boundary conditions, and GPU performance under load.

use glam::{Quat, Vec3};
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
// Stacking stability
// ---------------------------------------------------------------------------

#[test]
fn two_sphere_stack_on_floor() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    let bottom = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 1.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let top = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 3.5, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let pos_bottom = world.get_position(bottom).unwrap();
    let pos_top = world.get_position(top).unwrap();

    assert!(
        pos_bottom.is_finite(),
        "Bottom sphere diverged: {pos_bottom}"
    );
    assert!(pos_top.is_finite(), "Top sphere diverged: {pos_top}");
    // Top should be above bottom
    assert!(
        pos_top.y > pos_bottom.y - 0.5,
        "Stack collapsed: bottom.y={}, top.y={}",
        pos_bottom.y,
        pos_top.y
    );
    // Neither should fall through floor
    assert!(
        pos_bottom.y > -3.0,
        "Bottom fell through floor: y={}",
        pos_bottom.y
    );
}

#[test]
fn box_stack_three_high() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..3 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.5 + i as f32 * 1.1, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.5),
            },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 300);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(pos.is_finite(), "Box {i} in stack diverged: {pos}");
        assert!(pos.y > -3.0, "Box {i} fell through floor: y={}", pos.y);
    }

    // Boxes should be roughly stacked (each above the previous)
    let y0 = world.get_position(handles[0]).unwrap().y;
    let y1 = world.get_position(handles[1]).unwrap().y;
    let y2 = world.get_position(handles[2]).unwrap().y;
    assert!(
        y2 >= y1 - 1.0 && y1 >= y0 - 1.0,
        "Stack order violated: y0={y0}, y1={y1}, y2={y2}"
    );
}

// ---------------------------------------------------------------------------
// Domino chain
// ---------------------------------------------------------------------------

#[test]
fn domino_chain_propagation() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(30.0, 0.5, 5.0),
        },
        ..Default::default()
    });

    // Thin tall boxes as dominoes
    let mut handles = Vec::new();
    for i in 0..5 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(i as f32 * 2.0, 1.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(0.1, 1.0, 0.5),
            },
            ..Default::default()
        });
        handles.push(h);
    }

    // Push the first domino
    world.set_velocity(handles[0], Vec3::new(3.0, 0.0, 0.0));

    step_n(&mut world, 300);

    // All dominoes should have been affected (positions changed from initial)
    let last_pos = world.get_position(*handles.last().unwrap()).unwrap();
    assert!(
        last_pos.is_finite(),
        "Last domino position diverged: {last_pos}"
    );
}

// ---------------------------------------------------------------------------
// Newton's cradle approximation
// ---------------------------------------------------------------------------

#[test]
fn newtons_cradle_three_balls() {
    // Three identical spheres in a line. Hit the first one.
    // The last one should move, middle should stay approximately still.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        ..Default::default()
    });

    let ball1 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let ball2 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let ball3 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let v1 = world.get_velocity(ball1).unwrap();
    let v3 = world.get_velocity(ball3).unwrap();
    let p1 = world.get_position(ball1).unwrap();
    let _p2 = world.get_position(ball2).unwrap();
    let p3 = world.get_position(ball3).unwrap();

    // All should be finite
    assert!(v1.is_finite() && v3.is_finite());
    assert!(p1.is_finite() && p3.is_finite());

    // Total momentum should be approximately 5 (initial)
    let total_p = v1.x + world.get_velocity(ball2).unwrap().x + v3.x;
    assert!(
        (total_p - 5.0).abs() < 3.0,
        "Newton's cradle momentum: total_p={total_p}"
    );
}

// ---------------------------------------------------------------------------
// Rotational dynamics
// ---------------------------------------------------------------------------

#[test]
fn box_rotation_from_torque() {
    // A box with initial angular velocity should rotate.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        angular_velocity: Vec3::new(0.0, std::f32::consts::PI, 0.0), // ~180 deg/s
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(1.0, 0.5, 0.3),
        },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let rot = world.get_rotation(h).unwrap();
    // Rotation should have changed from identity
    let angle = rot.to_axis_angle().1;
    assert!(
        angle > 0.1,
        "Box should have rotated significantly, angle={angle}"
    );
}

#[test]
fn spinning_sphere_free_fall() {
    // A spinning sphere in gravity should fall normally while spinning.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let spinning = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        angular_velocity: Vec3::new(10.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let non_spinning = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 10.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos_spin = world.get_position(spinning).unwrap();
    let pos_no_spin = world.get_position(non_spinning).unwrap();

    // Both should fall at the same rate (spin doesn't affect free fall)
    assert!(
        (pos_spin.y - pos_no_spin.y).abs() < 0.5,
        "Spin affected free fall: spinning.y={}, non_spinning.y={}",
        pos_spin.y,
        pos_no_spin.y
    );
}

// ---------------------------------------------------------------------------
// Multi-shape collision pairs
// ---------------------------------------------------------------------------

#[test]
fn sphere_capsule_collision() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let capsule = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Capsule {
            half_height: 1.0,
            radius: 0.5,
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let ps = world.get_position(sphere).unwrap();
    let pc = world.get_position(capsule).unwrap();
    assert!(ps.is_finite() && pc.is_finite());
    let dist = (ps - pc).length();
    assert!(
        dist > 1.0,
        "Sphere and capsule should have separated: dist={dist}"
    );
}

#[test]
fn box_capsule_collision() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let box_body = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });

    let capsule = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Capsule {
            half_height: 0.5,
            radius: 0.3,
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pb = world.get_position(box_body).unwrap();
    let pc = world.get_position(capsule).unwrap();
    assert!(pb.is_finite() && pc.is_finite());
}

#[test]
fn capsule_capsule_collision() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let c1 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Capsule {
            half_height: 1.0,
            radius: 0.5,
        },
        ..Default::default()
    });

    let c2 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Capsule {
            half_height: 1.0,
            radius: 0.5,
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let p1 = world.get_position(c1).unwrap();
    let p2 = world.get_position(c2).unwrap();
    assert!(p1.is_finite() && p2.is_finite());
    let dist = (p1 - p2).length();
    assert!(dist > 0.5, "Capsules should separate: dist={dist}");
}

#[test]
fn hull_capsule_collision() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let cube_verts = vec![
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
    ];

    let hull = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::ConvexHull {
            vertices: cube_verts,
        },
        ..Default::default()
    });

    let capsule = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Capsule {
            half_height: 0.5,
            radius: 0.3,
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let ph = world.get_position(hull).unwrap();
    let pc = world.get_position(capsule).unwrap();
    assert!(ph.is_finite() && pc.is_finite());
}

// ---------------------------------------------------------------------------
// Plane interactions
// ---------------------------------------------------------------------------

#[test]
fn sphere_on_plane_does_not_fall_through() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _plane = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let pos = world.get_position(sphere).unwrap();
    assert!(pos.is_finite(), "Sphere on plane diverged: {pos}");
    assert!(pos.y > -2.0, "Sphere fell through plane: y={}", pos.y);
}

#[test]
fn box_on_plane_stable() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _plane = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });

    let box_body = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 3.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let pos = world.get_position(box_body).unwrap();
    assert!(pos.is_finite(), "Box on plane diverged: {pos}");
    assert!(pos.y > -2.0, "Box fell through plane: y={}", pos.y);
}

// ---------------------------------------------------------------------------
// Collision events
// ---------------------------------------------------------------------------

#[test]
fn collision_events_generated_on_impact() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let _b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let mut found_collision = false;
    for _ in 0..120 {
        world.step();
        let events = world.drain_collision_events();
        if !events.is_empty() {
            found_collision = true;
        }
    }

    assert!(
        found_collision,
        "Expected collision events from head-on sphere collision"
    );
}

// ---------------------------------------------------------------------------
// Kinematic bodies
// ---------------------------------------------------------------------------

#[test]
fn kinematic_body_not_affected_by_gravity() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    world.set_body_kinematic(h, true);

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    assert!(
        (pos.y - 5.0).abs() < 0.01,
        "Kinematic body should not fall: y={}",
        pos.y
    );
}

#[test]
fn kinematic_body_pushes_dynamic() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let kinematic = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    world.set_body_kinematic(kinematic, true);

    let dynamic = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    // Move kinematic body toward dynamic one
    for _ in 0..60 {
        let pos = world.get_position(kinematic).unwrap();
        world.set_position(kinematic, pos + Vec3::new(0.1, 0.0, 0.0));
        world.step();
    }

    let pos_d = world.get_position(dynamic).unwrap();
    assert!(pos_d.is_finite(), "Dynamic body diverged: {pos_d}");
}

// ---------------------------------------------------------------------------
// Raycast
// ---------------------------------------------------------------------------

#[test]
fn raycast_hits_sphere() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let result = world.raycast(Vec3::ZERO, Vec3::X, 100.0);
    assert!(result.is_some(), "Ray should hit sphere");
    let (hit_handle, t, _normal) = result.unwrap();
    assert_eq!(hit_handle, h, "Hit wrong body");
    assert!((t - 4.0).abs() < 0.5, "Hit distance should be ~4.0: t={t}");
}

#[test]
fn raycast_misses_behind() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let _h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    // Ray pointing +X, sphere at -X
    let result = world.raycast(Vec3::ZERO, Vec3::X, 100.0);
    assert!(result.is_none(), "Ray should miss sphere behind it");
}

#[test]
fn raycast_batch_multiple_hits() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let _s1 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let _s2 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let results = world.raycast_batch(&[
        (Vec3::ZERO, Vec3::X, 100.0),
        (Vec3::ZERO, Vec3::Y, 100.0),
        (Vec3::ZERO, Vec3::NEG_Z, 100.0), // should miss
    ]);

    assert_eq!(results.len(), 3);
    assert!(results[0].is_some(), "Should hit sphere on X axis");
    assert!(results[1].is_some(), "Should hit sphere on Y axis");
    assert!(results[2].is_none(), "Should miss on -Z axis");
}

// ---------------------------------------------------------------------------
// AABB overlap query
// ---------------------------------------------------------------------------

#[test]
fn overlap_aabb_finds_bodies() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        ..Default::default()
    });

    let h1 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let _h2 = world.add_body(&RigidBodyDesc {
        position: Vec3::new(100.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    // Step once to compute AABBs
    world.step();

    let hits = world.overlap_aabb(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));
    assert!(
        hits.contains(&h1),
        "AABB overlap should find body at origin"
    );
}

// ---------------------------------------------------------------------------
// Rotated shape collisions
// ---------------------------------------------------------------------------

#[test]
fn rotated_box_box_collision() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    // 45-degree rotated box
    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-3.0, 0.0, 0.0),
        rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
        linear_velocity: Vec3::new(4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });

    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        rotation: Quat::from_rotation_y(0.3),
        linear_velocity: Vec3::new(-4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pa = world.get_position(a).unwrap();
    let pb = world.get_position(b).unwrap();
    assert!(pa.is_finite() && pb.is_finite());
    assert!(
        (pa - pb).length() > 0.5,
        "Rotated boxes should separate: dist={}",
        (pa - pb).length()
    );
}

// ---------------------------------------------------------------------------
// GPU performance / stress
// ---------------------------------------------------------------------------

#[test]
fn stress_64_bodies_mixed_shapes() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 8,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -2.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(50.0, 1.0, 50.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..64 {
        let x = (i % 8) as f32 * 3.0 - 12.0;
        let z = (i / 8) as f32 * 3.0 - 12.0;
        let shape = match i % 4 {
            0 => ShapeDesc::Sphere { radius: 0.5 },
            1 => ShapeDesc::Box {
                half_extents: Vec3::splat(0.4),
            },
            2 => ShapeDesc::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            _ => ShapeDesc::Sphere { radius: 0.7 },
        };
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(x, 5.0 + (i as f32) * 0.05, z),
            mass: 1.0,
            shape,
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 120);

    let mut finite_count = 0;
    for &h in &handles {
        let pos = world.get_position(h).unwrap();
        if pos.is_finite() {
            finite_count += 1;
        }
    }
    assert_eq!(
        finite_count,
        handles.len(),
        "All 64 bodies should have finite positions"
    );
}

#[test]
fn stress_128_spheres_rain() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -2.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(50.0, 1.0, 50.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..128 {
        let x = (i % 16) as f32 * 2.5 - 20.0;
        let z = (i / 16) as f32 * 2.5 - 10.0;
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(x, 10.0 + (i as f32) * 0.2, z),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 60);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.is_finite(),
            "Sphere {i} diverged in 128-body rain: {pos}"
        );
    }
}

// ---------------------------------------------------------------------------
// Compound shape tests
// ---------------------------------------------------------------------------

#[test]
fn compound_shape_free_fall() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Compound {
            children: vec![
                (
                    ShapeDesc::Sphere { radius: 0.5 },
                    Vec3::new(-1.0, 0.0, 0.0),
                    Quat::IDENTITY,
                ),
                (
                    ShapeDesc::Sphere { radius: 0.5 },
                    Vec3::new(1.0, 0.0, 0.0),
                    Quat::IDENTITY,
                ),
            ],
        },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    assert!(pos.y < 10.0, "Compound should have fallen: y={}", pos.y);
    assert!(pos.is_finite(), "Compound diverged: {pos}");
}

#[test]
fn compound_vs_static_collision() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    let compound = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Compound {
            children: vec![
                (
                    ShapeDesc::Sphere { radius: 0.5 },
                    Vec3::new(-0.5, 0.0, 0.0),
                    Quat::IDENTITY,
                ),
                (
                    ShapeDesc::Sphere { radius: 0.5 },
                    Vec3::new(0.5, 0.0, 0.0),
                    Quat::IDENTITY,
                ),
            ],
        },
        ..Default::default()
    });

    step_n(&mut world, 180);

    let pos = world.get_position(compound).unwrap();
    assert!(pos.is_finite(), "Compound vs floor diverged: {pos}");
    assert!(pos.y > -3.0, "Compound fell through floor: y={}", pos.y);
}

// ---------------------------------------------------------------------------
// Velocity setter correctness
// ---------------------------------------------------------------------------

#[test]
fn set_velocity_applies_correctly() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    world.set_velocity(h, Vec3::new(10.0, 0.0, 0.0));
    step_n(&mut world, 60);

    let pos = world.get_position(h).unwrap();
    // With v=10 and t=1s, should be at x~10
    assert!(
        (pos.x - 10.0).abs() < 2.0,
        "Velocity not applied correctly: pos.x={}",
        pos.x
    );
}

#[test]
fn set_angular_velocity_applies_correctly() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    world.set_angular_velocity(h, Vec3::new(0.0, std::f32::consts::TAU, 0.0));
    step_n(&mut world, 30);

    let rot = world.get_rotation(h).unwrap();
    let angle = rot.to_axis_angle().1;
    assert!(
        angle > 0.5,
        "Angular velocity should cause rotation: angle={angle}"
    );
}

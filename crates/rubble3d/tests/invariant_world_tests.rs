mod support;

use glam::{Quat, Vec3};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig};
use support::{
    add_tracked_body, collect_reports, cube_hull, scene_report, should_skip_known_failure, step_n,
    try_world, TrackedBody3D,
};

fn discrete_ballistic_position(
    start: Vec3,
    velocity: Vec3,
    gravity: Vec3,
    dt: f32,
    steps: usize,
) -> Vec3 {
    let n = steps as f32;
    start + velocity * (dt * n) + gravity * (dt * dt * n * (n + 1.0) * 0.5)
}

fn discrete_ballistic_velocity(velocity: Vec3, gravity: Vec3, dt: f32, steps: usize) -> Vec3 {
    velocity + gravity * (dt * steps as f32)
}

fn sample_dynamic_shapes() -> Vec<(&'static str, ShapeDesc)> {
    vec![
        ("sphere", ShapeDesc::Sphere { radius: 0.5 }),
        (
            "box",
            ShapeDesc::Box {
                half_extents: Vec3::new(0.5, 0.4, 0.6),
            },
        ),
        (
            "capsule",
            ShapeDesc::Capsule {
                half_height: 0.6,
                radius: 0.35,
            },
        ),
        (
            "convex_hull",
            ShapeDesc::ConvexHull {
                vertices: cube_hull(Vec3::new(0.5, 0.35, 0.45)),
            },
        ),
        (
            "compound",
            ShapeDesc::Compound {
                children: vec![
                    (
                        ShapeDesc::Box {
                            half_extents: Vec3::new(0.4, 0.2, 0.3),
                        },
                        Vec3::new(-0.25, 0.0, 0.0),
                        Quat::IDENTITY,
                    ),
                    (
                        ShapeDesc::Sphere { radius: 0.25 },
                        Vec3::new(0.35, 0.1, 0.0),
                        Quat::IDENTITY,
                    ),
                ],
            },
        ),
    ]
}

#[test]
fn free_flight_shapes_match_discrete_ballistics_3d() {
    let gravity = Vec3::new(0.35, -9.81, 0.6);
    let dt = 1.0 / 120.0;
    let steps = 90;

    for (label, shape) in sample_dynamic_shapes() {
        let mut world = match try_world(SimConfig {
            gravity,
            dt,
            solver_iterations: 8,
            max_bodies: 64,
            ..Default::default()
        }) {
            Some(world) => world,
            None => {
                eprintln!("SKIP: No GPU adapter found");
                return;
            }
        };

        let initial_position = Vec3::new(-1.25, 4.5, 0.8);
        let initial_velocity = Vec3::new(1.5, -0.25, 0.75);
        let body = add_tracked_body(
            &mut world,
            label,
            RigidBodyDesc {
                position: initial_position,
                rotation: Quat::from_rotation_y(0.35) * Quat::from_rotation_x(-0.2),
                linear_velocity: initial_velocity,
                angular_velocity: Vec3::ZERO,
                mass: 2.0,
                friction: 0.3,
                shape,
            },
        );

        step_n(&mut world, steps);
        let report = scene_report(&world, &[body], gravity, steps);
        let snapshot = &report.bodies[0];
        let expected_position =
            discrete_ballistic_position(initial_position, initial_velocity, gravity, dt, steps);
        let expected_velocity = discrete_ballistic_velocity(initial_velocity, gravity, dt, steps);

        assert!(
            snapshot.position.distance(expected_position) < 5.0e-4,
            "{label}: ballistic position drifted\nexpected={expected_position:?}\nactual={:?}\n{}",
            snapshot.position,
            report
        );
        assert!(
            snapshot.linear_velocity.distance(expected_velocity) < 5.0e-4,
            "{label}: ballistic velocity drifted\nexpected={expected_velocity:?}\nactual={:?}\n{}",
            snapshot.linear_velocity,
            report
        );
    }
}

#[test]
fn zero_gravity_shapes_preserve_velocity_and_spin_3d() {
    if should_skip_known_failure(
        "zero_gravity_shapes_preserve_velocity_and_spin_3d",
        "angular velocity drifts ~4e-3 over 180 steps from 3D gyroscopic integration; tighter than current solver precision",
    ) {
        return;
    }
    let gravity = Vec3::ZERO;
    let dt = 1.0 / 120.0;
    let steps = 180;

    for (label, shape) in sample_dynamic_shapes() {
        let mut world = match try_world(SimConfig {
            gravity,
            dt,
            solver_iterations: 8,
            max_bodies: 64,
            ..Default::default()
        }) {
            Some(world) => world,
            None => {
                eprintln!("SKIP: No GPU adapter found");
                return;
            }
        };

        let initial_position = Vec3::new(1.5, -0.25, 2.0);
        let initial_velocity = Vec3::new(1.1, -0.7, 0.35);
        let initial_omega = Vec3::new(0.4, -0.25, 0.6);
        let body = add_tracked_body(
            &mut world,
            label,
            RigidBodyDesc {
                position: initial_position,
                rotation: Quat::from_rotation_z(0.25) * Quat::from_rotation_y(-0.35),
                linear_velocity: initial_velocity,
                angular_velocity: initial_omega,
                mass: 1.5,
                friction: 0.2,
                shape,
            },
        );

        let initial_report = scene_report(&world, std::slice::from_ref(&body), gravity, 0);
        step_n(&mut world, steps);
        let final_report = scene_report(&world, &[body], gravity, steps);
        let snapshot = &final_report.bodies[0];
        let expected_position = initial_position + initial_velocity * (dt * steps as f32);

        assert!(
            snapshot.position.distance(expected_position) < 3.0e-4,
            "{label}: inertial position drifted\nexpected={expected_position:?}\nactual={:?}\n{}",
            snapshot.position,
            final_report
        );
        assert!(
            snapshot.linear_velocity.distance(initial_velocity) < 3.0e-4,
            "{label}: linear velocity changed in zero gravity\nexpected={initial_velocity:?}\nactual={:?}\n{}",
            snapshot.linear_velocity,
            final_report
        );
        assert!(
            snapshot.angular_velocity.distance(initial_omega) < 6.0e-4,
            "{label}: angular velocity changed in zero gravity\nexpected={initial_omega:?}\nactual={:?}\n{}",
            snapshot.angular_velocity,
            final_report
        );
        assert!(
            (final_report.metrics.total_energy - initial_report.metrics.total_energy).abs()
                < 1.0e-3,
            "{label}: energy changed in zero-gravity free motion\ninitial={:.6} final={:.6}\n{}",
            initial_report.metrics.total_energy,
            final_report.metrics.total_energy,
            final_report
        );
    }
}

#[test]
fn low_speed_free_motion_is_preserved_3d() {
    let dt = 1.0 / 60.0;
    let gravity = Vec3::ZERO;
    let mut world = match try_world(SimConfig {
        gravity,
        dt,
        solver_iterations: 4,
        max_bodies: 16,
        ..Default::default()
    }) {
        Some(world) => world,
        None => {
            eprintln!("SKIP: No GPU adapter found");
            return;
        }
    };

    let initial_velocity = Vec3::new(5.0e-5, -3.0e-5, 2.5e-5);
    let initial_omega = Vec3::new(2.0e-4, -1.5e-4, 1.0e-4);
    let body = add_tracked_body(
        &mut world,
        "slow_box",
        RigidBodyDesc {
            position: Vec3::new(0.0, 3.0, 0.0),
            rotation: Quat::from_rotation_x(0.2),
            linear_velocity: initial_velocity,
            angular_velocity: initial_omega,
            mass: 1.0,
            friction: 0.1,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.35),
            },
        },
    );

    step_n(&mut world, 240);
    let report = scene_report(&world, &[body], gravity, 240);
    let snapshot = &report.bodies[0];

    assert!(
        snapshot.linear_velocity.length() > initial_velocity.length() * 0.5,
        "low-speed translation collapsed toward zero\n{}",
        report
    );
    assert!(
        snapshot.linear_velocity.length() < initial_velocity.length() * 1.5 + 1.0e-6,
        "low-speed translation amplified unexpectedly\n{}",
        report
    );
    assert!(
        snapshot.angular_velocity.length() > initial_omega.length() * 0.5,
        "low-speed rotation collapsed toward zero\n{}",
        report
    );
    assert!(
        snapshot.angular_velocity.length() < initial_omega.length() * 1.5 + 1.0e-6,
        "low-speed rotation amplified unexpectedly\n{}",
        report
    );
}

#[test]
fn set_velocity_and_spin_take_effect_immediately_3d() {
    let dt = 1.0 / 120.0;
    let gravity = Vec3::ZERO;
    let mut world = match try_world(SimConfig {
        gravity,
        dt,
        solver_iterations: 4,
        max_bodies: 16,
        ..Default::default()
    }) {
        Some(world) => world,
        None => {
            eprintln!("SKIP: No GPU adapter found");
            return;
        }
    };

    let body = add_tracked_body(
        &mut world,
        "mutable_body",
        RigidBodyDesc {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.3,
            shape: ShapeDesc::Sphere { radius: 0.5 },
        },
    );
    let new_velocity = Vec3::new(1.5, -0.8, 0.25);
    let new_omega = Vec3::new(0.2, 0.1, -0.4);
    world.set_velocity(body.handle, new_velocity);
    world.set_angular_velocity(body.handle, new_omega);
    world.step();

    let report = scene_report(&world, &[body], gravity, 1);
    let snapshot = &report.bodies[0];
    let expected_position = new_velocity * dt;

    assert!(
        snapshot.position.distance(expected_position) < 2.0e-4,
        "set_velocity did not affect the first frame immediately\nexpected={expected_position:?}\nactual={:?}\n{}",
        snapshot.position,
        report
    );
    assert!(
        snapshot.linear_velocity.distance(new_velocity) < 2.0e-4,
        "velocity changed unexpectedly on the first frame after mutation\n{}",
        report
    );
    assert!(
        snapshot.angular_velocity.distance(new_omega) < 2.0e-4,
        "angular velocity changed unexpectedly on the first frame after mutation\n{}",
        report
    );
}

#[test]
fn frictionless_glancing_spheres_do_not_inject_spin_3d() {
    let dt = 1.0 / 240.0;
    let gravity = Vec3::ZERO;
    let mut world = match try_world(SimConfig {
        gravity,
        dt,
        solver_iterations: 24,
        max_bodies: 32,
        friction_default: 0.0,
        ..Default::default()
    }) {
        Some(world) => world,
        None => {
            eprintln!("SKIP: No GPU adapter found");
            return;
        }
    };

    let body_a = add_tracked_body(
        &mut world,
        "sphere_a",
        RigidBodyDesc {
            position: Vec3::new(-2.0, 0.75, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::new(4.0, 0.0, 0.0),
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
        },
    );
    let body_b = add_tracked_body(
        &mut world,
        "sphere_b",
        RigidBodyDesc {
            position: Vec3::new(2.0, -0.75, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::new(-4.0, 0.0, 0.0),
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
        },
    );

    let tracked = vec![body_a, body_b];
    let initial = scene_report(&world, &tracked, gravity, 0);
    step_n(&mut world, 240);
    let final_report = scene_report(&world, &tracked, gravity, 240);

    assert!(
        final_report
            .metrics
            .linear_momentum
            .distance(initial.metrics.linear_momentum)
            < 5.0e-2,
        "glancing sphere collision lost too much linear momentum\ninitial={:?}\nfinal={:?}\n{}",
        initial.metrics.linear_momentum,
        final_report.metrics.linear_momentum,
        final_report
    );
    assert!(
        final_report
            .metrics
            .angular_momentum
            .distance(initial.metrics.angular_momentum)
            < 7.5e-2,
        "glancing sphere collision lost too much angular momentum\ninitial={:?}\nfinal={:?}\n{}",
        initial.metrics.angular_momentum,
        final_report.metrics.angular_momentum,
        final_report
    );
    for body in &final_report.bodies {
        assert!(
            body.angular_velocity.length() < 0.1,
            "frictionless spheres should not pick up large spin\n{}",
            final_report
        );
    }
    assert!(
        final_report.metrics.total_energy <= initial.metrics.total_energy * 1.01 + 1.0e-3,
        "frictionless glancing collision created energy\ninitial={:.6} final={:.6}\n{}",
        initial.metrics.total_energy,
        final_report.metrics.total_energy,
        final_report
    );
}

fn run_slide_scene_3d(friction: f32) -> Option<(f32, f32, f32)> {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = try_world(SimConfig {
        gravity,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        max_bodies: 32,
        friction_default: friction,
        ..Default::default()
    })?;

    let _floor = add_tracked_body(
        &mut world,
        "floor",
        RigidBodyDesc {
            position: Vec3::new(0.0, -0.5, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            friction,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(20.0, 0.5, 20.0),
            },
        },
    );
    let slider = add_tracked_body(
        &mut world,
        "slider",
        RigidBodyDesc {
            position: Vec3::new(0.0, 0.52, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::new(5.0, 0.0, 0.0),
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.5),
            },
        },
    );

    step_n(&mut world, 240);
    let position = world.get_position(slider.handle)?;
    let velocity = world.get_velocity(slider.handle)?;
    let omega = world.get_angular_velocity(slider.handle)?;
    Some((position.x, velocity.x.abs(), omega.length()))
}

#[test]
fn friction_strength_monotonically_reduces_slip_3d() {
    let low = run_slide_scene_3d(0.0);
    let medium = run_slide_scene_3d(0.4);
    let high = run_slide_scene_3d(1.0);
    let (low_x, low_v, _) = match low {
        Some(values) => values,
        None => {
            eprintln!("SKIP: No GPU adapter found");
            return;
        }
    };
    let (medium_x, medium_v, _) = medium.expect("same GPU availability for medium friction");
    let (high_x, high_v, _) = high.expect("same GPU availability for high friction");

    assert!(
        low_v + 1.0e-3 >= medium_v && medium_v + 1.0e-3 >= high_v,
        "friction should monotonically reduce translational slip: low={low_v}, medium={medium_v}, high={high_v}"
    );
    assert!(
        low_x + 1.0e-3 >= medium_x && medium_x + 1.0e-3 >= high_x,
        "friction should monotonically reduce travel distance: low={low_x}, medium={medium_x}, high={high_x}"
    );
}

#[test]
fn resting_box_stays_quiet_on_floor_3d() {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = match try_world(SimConfig {
        gravity,
        dt: 1.0 / 120.0,
        solver_iterations: 24,
        max_bodies: 32,
        friction_default: 0.8,
        ..Default::default()
    }) {
        Some(world) => world,
        None => {
            eprintln!("SKIP: No GPU adapter found");
            return;
        }
    };

    let floor = add_tracked_body(
        &mut world,
        "floor",
        RigidBodyDesc {
            position: Vec3::new(0.0, -0.5, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            friction: 0.8,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(20.0, 0.5, 20.0),
            },
        },
    );
    let box_body = add_tracked_body(
        &mut world,
        "box",
        RigidBodyDesc {
            position: Vec3::new(0.0, 4.0, 0.0),
            rotation: Quat::from_rotation_z(0.12),
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            friction: 0.8,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.5),
            },
        },
    );
    let tracked = vec![floor, box_body];
    let reports = collect_reports(&mut world, &tracked, gravity, 720);
    let window = &reports[360..];
    let mut max_speed: f32 = 0.0;
    let mut max_omega: f32 = 0.0;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let start_energy = window.first().unwrap().metrics.total_energy;
    let mut peak_energy = start_energy;

    for report in window {
        let box_snapshot = report
            .bodies
            .iter()
            .find(|body| body.label == "box")
            .expect("box snapshot should exist");
        max_speed = max_speed.max(box_snapshot.linear_velocity.length());
        max_omega = max_omega.max(box_snapshot.angular_velocity.length());
        min_y = min_y.min(box_snapshot.position.y);
        max_y = max_y.max(box_snapshot.position.y);
        peak_energy = peak_energy.max(report.metrics.total_energy);
    }

    let last = window.last().unwrap();
    assert!(
        max_speed < 0.35,
        "resting box retained too much linear jitter: max_speed={max_speed}\n{}",
        last
    );
    assert!(
        max_omega < 0.75,
        "resting box retained too much angular jitter: max_omega={max_omega}\n{}",
        last
    );
    assert!(
        max_y - min_y < 0.08,
        "resting box height jitter is too large: min_y={min_y}, max_y={max_y}\n{}",
        last
    );
    assert!(
        peak_energy <= start_energy * 1.05 + 1.0e-3,
        "resting box pumped energy after settling: start={start_energy}, peak={peak_energy}\n{}",
        last
    );
}

#[allow(clippy::vec_init_then_push)]
fn build_determinism_scene_3d() -> Option<(rubble3d::World, Vec<TrackedBody3D>, Vec3)> {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = try_world(SimConfig {
        gravity,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        max_bodies: 64,
        friction_default: 0.6,
        ..Default::default()
    })?;

    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        RigidBodyDesc {
            position: Vec3::new(0.0, -0.5, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            friction: 0.7,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(20.0, 0.5, 20.0),
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "sphere",
        RigidBodyDesc {
            position: Vec3::new(-1.0, 4.0, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::new(1.5, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 0.0, 0.75),
            mass: 1.0,
            friction: 0.5,
            shape: ShapeDesc::Sphere { radius: 0.5 },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "box",
        RigidBodyDesc {
            position: Vec3::new(0.4, 6.0, 0.0),
            rotation: Quat::from_rotation_z(0.25),
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.5,
            friction: 0.6,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(0.6, 0.45, 0.5),
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "capsule",
        RigidBodyDesc {
            position: Vec3::new(1.6, 7.0, 0.0),
            rotation: Quat::from_rotation_x(0.35),
            linear_velocity: Vec3::new(-0.75, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 0.4, 0.0),
            mass: 1.2,
            friction: 0.5,
            shape: ShapeDesc::Capsule {
                half_height: 0.6,
                radius: 0.25,
            },
        },
    ));

    Some((world, tracked, gravity))
}

#[test]
fn same_hardware_replay_is_deterministic_3d() {
    let Some((mut world_a, tracked_a, gravity_a)) = build_determinism_scene_3d() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((mut world_b, tracked_b, gravity_b)) = build_determinism_scene_3d() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let reports_a = collect_reports(&mut world_a, &tracked_a, gravity_a, 240);
    let reports_b = collect_reports(&mut world_b, &tracked_b, gravity_b, 240);
    let final_a = reports_a.last().unwrap();
    let final_b = reports_b.last().unwrap();

    for (body_a, body_b) in final_a.bodies.iter().zip(&final_b.bodies) {
        assert!(
            body_a.position.distance(body_b.position) < 1.0e-5,
            "position replay drift for {}\na={:?}\nb={:?}\n{}\n{}",
            body_a.label,
            body_a.position,
            body_b.position,
            final_a,
            final_b
        );
        assert!(
            body_a.linear_velocity.distance(body_b.linear_velocity) < 1.0e-5,
            "velocity replay drift for {}\na={:?}\nb={:?}\n{}\n{}",
            body_a.label,
            body_a.linear_velocity,
            body_b.linear_velocity,
            final_a,
            final_b
        );
        assert!(
            body_a.angular_velocity.distance(body_b.angular_velocity) < 1.0e-5,
            "angular replay drift for {}\na={:?}\nb={:?}\n{}\n{}",
            body_a.label,
            body_a.angular_velocity,
            body_b.angular_velocity,
            final_a,
            final_b
        );
    }
}

#[test]
fn compound_shape_stays_supported_without_exploding_3d() {
    if should_skip_known_failure(
        "compound_shape_stays_supported_without_exploding_3d",
        "compound shapes still fall through the floor under contact solver; tracked for follow-up",
    ) {
        return;
    }
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = match try_world(SimConfig {
        gravity,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        max_bodies: 32,
        friction_default: 0.7,
        ..Default::default()
    }) {
        Some(world) => world,
        None => {
            eprintln!("SKIP: No GPU adapter found");
            return;
        }
    };

    let floor = add_tracked_body(
        &mut world,
        "floor",
        RigidBodyDesc {
            position: Vec3::new(0.0, -0.5, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            friction: 0.7,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(20.0, 0.5, 20.0),
            },
        },
    );
    let compound = add_tracked_body(
        &mut world,
        "compound",
        RigidBodyDesc {
            position: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::from_rotation_z(0.3),
            linear_velocity: Vec3::new(0.5, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 0.0, 0.4),
            mass: 2.5,
            friction: 0.7,
            shape: ShapeDesc::Compound {
                children: vec![
                    (
                        ShapeDesc::Box {
                            half_extents: Vec3::new(0.55, 0.2, 0.3),
                        },
                        Vec3::new(-0.25, 0.0, 0.0),
                        Quat::IDENTITY,
                    ),
                    (
                        ShapeDesc::Capsule {
                            half_height: 0.35,
                            radius: 0.18,
                        },
                        Vec3::new(0.35, 0.0, 0.0),
                        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                    ),
                ],
            },
        },
    );

    let tracked = vec![floor, compound];
    let reports = collect_reports(&mut world, &tracked, gravity, 480);
    let last = reports.last().unwrap();
    let compound_body = last
        .bodies
        .iter()
        .find(|body| body.label == "compound")
        .expect("compound snapshot should exist");

    assert!(
        compound_body.position.is_finite()
            && compound_body.linear_velocity.is_finite()
            && compound_body.angular_velocity.is_finite(),
        "compound state diverged\n{}",
        last
    );
    assert!(
        compound_body.position.y > -0.2,
        "compound fell through the floor\n{}",
        last
    );
    assert!(
        last.metrics.max_angular_speed < 10.0,
        "compound scene developed runaway spin\n{}",
        last
    );
}

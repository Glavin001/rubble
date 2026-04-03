mod support;

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D};
use support::{
    add_tracked_body, collect_reports, regular_polygon, scene_report, should_skip_known_failure,
    step_n, try_world,
};

fn discrete_ballistic_position(
    start: Vec2,
    velocity: Vec2,
    gravity: Vec2,
    dt: f32,
    steps: usize,
) -> Vec2 {
    let n = steps as f32;
    start + velocity * (dt * n) + gravity * (dt * dt * n * (n + 1.0) * 0.5)
}

fn discrete_ballistic_velocity(velocity: Vec2, gravity: Vec2, dt: f32, steps: usize) -> Vec2 {
    velocity + gravity * (dt * steps as f32)
}

fn sample_dynamic_shapes() -> Vec<(&'static str, ShapeDesc2D)> {
    vec![
        ("circle", ShapeDesc2D::Circle { radius: 0.5 }),
        (
            "rect",
            ShapeDesc2D::Rect {
                half_extents: Vec2::new(0.45, 0.35),
            },
        ),
        (
            "capsule",
            ShapeDesc2D::Capsule {
                half_height: 0.55,
                radius: 0.25,
            },
        ),
        (
            "polygon",
            ShapeDesc2D::ConvexPolygon {
                vertices: regular_polygon(0.45, 8),
            },
        ),
    ]
}

#[test]
fn free_flight_shapes_match_discrete_ballistics_2d() {
    let gravity = Vec2::new(0.5, -9.81);
    let dt = 1.0 / 120.0;
    let steps = 90;

    for (label, shape) in sample_dynamic_shapes() {
        let mut world = match try_world(SimConfig2D {
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

        let initial_position = Vec2::new(-1.0, 4.5);
        let initial_velocity = Vec2::new(1.25, -0.4);
        let body = add_tracked_body(
            &mut world,
            label,
            RigidBodyDesc2D {
                x: initial_position.x,
                y: initial_position.y,
                angle: 0.35,
                vx: initial_velocity.x,
                vy: initial_velocity.y,
                angular_velocity: 0.0,
                mass: 1.5,
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
            snapshot.linear_velocity.distance(expected_velocity) < 3.0e-4,
            "{label}: ballistic velocity drifted\nexpected={expected_velocity:?}\nactual={:?}\n{}",
            snapshot.linear_velocity,
            report
        );
    }
}

#[test]
fn zero_gravity_shapes_preserve_velocity_and_spin_2d() {
    let gravity = Vec2::ZERO;
    let dt = 1.0 / 120.0;
    let steps = 180;

    for (label, shape) in sample_dynamic_shapes() {
        let mut world = match try_world(SimConfig2D {
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

        let initial_position = Vec2::new(1.25, 0.75);
        let initial_velocity = Vec2::new(1.1, -0.6);
        let initial_omega = 0.45;
        let body = add_tracked_body(
            &mut world,
            label,
            RigidBodyDesc2D {
                x: initial_position.x,
                y: initial_position.y,
                angle: -0.3,
                vx: initial_velocity.x,
                vy: initial_velocity.y,
                angular_velocity: initial_omega,
                mass: 1.25,
                friction: 0.2,
                shape,
            },
        );

        let initial_report = scene_report(&world, &[body.clone()], gravity, 0);
        step_n(&mut world, steps);
        let final_report = scene_report(&world, &[body], gravity, steps);
        let snapshot = &final_report.bodies[0];
        let expected_position = initial_position + initial_velocity * (dt * steps as f32);

        assert!(
            snapshot.position.distance(expected_position) < 2.0e-4,
            "{label}: inertial position drifted\nexpected={expected_position:?}\nactual={:?}\n{}",
            snapshot.position,
            final_report
        );
        assert!(
            snapshot.linear_velocity.distance(initial_velocity) < 2.0e-4,
            "{label}: linear velocity changed in zero gravity\nexpected={initial_velocity:?}\nactual={:?}\n{}",
            snapshot.linear_velocity,
            final_report
        );
        assert!(
            (snapshot.angular_velocity - initial_omega).abs() < 4.0e-4,
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
fn low_speed_free_motion_is_preserved_2d() {
    let dt = 1.0 / 60.0;
    let gravity = Vec2::ZERO;
    let mut world = match try_world(SimConfig2D {
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

    let initial_velocity = Vec2::new(4.0e-5, -3.0e-5);
    let initial_omega = 2.0e-4;
    let body = add_tracked_body(
        &mut world,
        "slow_rect",
        RigidBodyDesc2D {
            x: 0.0,
            y: 3.0,
            angle: 0.25,
            vx: initial_velocity.x,
            vy: initial_velocity.y,
            angular_velocity: initial_omega,
            mass: 1.0,
            friction: 0.1,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::splat(0.4),
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
        snapshot.angular_velocity.abs() > initial_omega.abs() * 0.5,
        "low-speed rotation collapsed toward zero\n{}",
        report
    );
    assert!(
        snapshot.angular_velocity.abs() < initial_omega.abs() * 1.5 + 1.0e-6,
        "low-speed rotation amplified unexpectedly\n{}",
        report
    );
}

#[test]
fn set_velocity_and_spin_take_effect_immediately_2d() {
    let dt = 1.0 / 120.0;
    let gravity = Vec2::ZERO;
    let mut world = match try_world(SimConfig2D {
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
        RigidBodyDesc2D {
            x: 0.0,
            y: 0.0,
            angle: 0.0,
            vx: 0.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 1.0,
            friction: 0.3,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
        },
    );
    let new_velocity = Vec2::new(1.25, -0.6);
    let new_omega = -0.45;
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
        (snapshot.angular_velocity - new_omega).abs() < 2.0e-4,
        "angular velocity changed unexpectedly on the first frame after mutation\n{}",
        report
    );
}

#[test]
fn frictionless_glancing_circles_do_not_inject_spin_2d() {
    if should_skip_known_failure(
        "frictionless_glancing_circles_do_not_inject_spin_2d",
        "2D frictionless circle impacts still create extra translational energy",
    ) {
        return;
    }
    let dt = 1.0 / 240.0;
    let gravity = Vec2::ZERO;
    let mut world = match try_world(SimConfig2D {
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
        "circle_a",
        RigidBodyDesc2D {
            x: -2.0,
            y: 0.75,
            angle: 0.0,
            vx: 4.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 1.0,
            friction: 0.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
        },
    );
    let body_b = add_tracked_body(
        &mut world,
        "circle_b",
        RigidBodyDesc2D {
            x: 2.0,
            y: -0.75,
            angle: 0.0,
            vx: -4.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 1.0,
            friction: 0.0,
            shape: ShapeDesc2D::Circle { radius: 1.0 },
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
        "glancing circle collision lost too much linear momentum\ninitial={:?}\nfinal={:?}\n{}",
        initial.metrics.linear_momentum,
        final_report.metrics.linear_momentum,
        final_report
    );
    assert!(
        (final_report.metrics.angular_momentum - initial.metrics.angular_momentum).abs() < 7.5e-2,
        "glancing circle collision lost too much angular momentum\ninitial={:.6}\nfinal={:.6}\n{}",
        initial.metrics.angular_momentum,
        final_report.metrics.angular_momentum,
        final_report
    );
    for body in &final_report.bodies {
        assert!(
            body.angular_velocity.abs() < 0.1,
            "frictionless circles should not pick up large spin\n{}",
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

fn run_slide_scene_2d(friction: f32) -> Option<(f32, f32, f32)> {
    let gravity = Vec2::new(0.0, -9.81);
    let mut world = try_world(SimConfig2D {
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
        RigidBodyDesc2D {
            x: 0.0,
            y: -0.5,
            angle: 0.0,
            vx: 0.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 0.0,
            friction,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(20.0, 0.5),
            },
        },
    );
    let slider = add_tracked_body(
        &mut world,
        "slider",
        RigidBodyDesc2D {
            x: 0.0,
            y: 0.52,
            angle: 0.0,
            vx: 5.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 1.0,
            friction,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::splat(0.5),
            },
        },
    );

    step_n(&mut world, 240);
    let position = world.get_position(slider.handle)?;
    let velocity = world.get_velocity(slider.handle)?;
    let omega = world.get_angular_velocity(slider.handle)?;
    Some((position.x, velocity.x.abs(), omega.abs()))
}

#[test]
fn friction_strength_monotonically_reduces_slip_2d() {
    if should_skip_known_failure(
        "friction_strength_monotonically_reduces_slip_2d",
        "2D friction ordering is currently inverted for some sliding scenes",
    ) {
        return;
    }
    let low = run_slide_scene_2d(0.0);
    let medium = run_slide_scene_2d(0.4);
    let high = run_slide_scene_2d(1.0);
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
fn resting_rect_stays_quiet_on_floor_2d() {
    if should_skip_known_failure(
        "resting_rect_stays_quiet_on_floor_2d",
        "2D resting contacts still develop large upward drift and spin",
    ) {
        return;
    }
    let gravity = Vec2::new(0.0, -9.81);
    let mut world = match try_world(SimConfig2D {
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
        RigidBodyDesc2D {
            x: 0.0,
            y: -0.5,
            angle: 0.0,
            vx: 0.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 0.0,
            friction: 0.8,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(20.0, 0.5),
            },
        },
    );
    let rect = add_tracked_body(
        &mut world,
        "rect",
        RigidBodyDesc2D {
            x: 0.0,
            y: 4.0,
            angle: 0.15,
            vx: 0.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 1.0,
            friction: 0.8,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::splat(0.5),
            },
        },
    );
    let tracked = vec![floor, rect];
    let reports = collect_reports(&mut world, &tracked, gravity, 720);
    let window = &reports[360..];
    let mut max_speed: f32 = 0.0;
    let mut max_omega: f32 = 0.0;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let start_energy = window.first().unwrap().metrics.total_energy;
    let mut peak_energy = start_energy;

    for report in window {
        let rect_snapshot = report
            .bodies
            .iter()
            .find(|body| body.label == "rect")
            .expect("rect snapshot should exist");
        max_speed = max_speed.max(rect_snapshot.linear_velocity.length());
        max_omega = max_omega.max(rect_snapshot.angular_velocity.abs());
        min_y = min_y.min(rect_snapshot.position.y);
        max_y = max_y.max(rect_snapshot.position.y);
        peak_energy = peak_energy.max(report.metrics.total_energy);
    }

    let last = window.last().unwrap();
    assert!(
        max_speed < 0.35,
        "resting rect retained too much linear jitter: max_speed={max_speed}\n{}",
        last
    );
    assert!(
        max_omega < 0.75,
        "resting rect retained too much angular jitter: max_omega={max_omega}\n{}",
        last
    );
    assert!(
        max_y - min_y < 0.08,
        "resting rect height jitter is too large: min_y={min_y}, max_y={max_y}\n{}",
        last
    );
    assert!(
        peak_energy <= start_energy * 1.05 + 1.0e-3,
        "resting rect pumped energy after settling: start={start_energy}, peak={peak_energy}\n{}",
        last
    );
}

fn build_determinism_scene_2d() -> Option<(rubble2d::World2D, Vec<support::TrackedBody2D>, Vec2)> {
    let gravity = Vec2::new(0.0, -9.81);
    let mut world = try_world(SimConfig2D {
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
        RigidBodyDesc2D {
            x: 0.0,
            y: -0.5,
            angle: 0.0,
            vx: 0.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 0.0,
            friction: 0.7,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(20.0, 0.5),
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "circle",
        RigidBodyDesc2D {
            x: -1.0,
            y: 4.0,
            angle: 0.0,
            vx: 1.5,
            vy: 0.0,
            angular_velocity: 0.6,
            mass: 1.0,
            friction: 0.5,
            shape: ShapeDesc2D::Circle { radius: 0.5 },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "rect",
        RigidBodyDesc2D {
            x: 0.5,
            y: 6.0,
            angle: 0.2,
            vx: 0.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: 1.3,
            friction: 0.6,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(0.55, 0.45),
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "capsule",
        RigidBodyDesc2D {
            x: 1.6,
            y: 7.0,
            angle: 0.3,
            vx: -0.75,
            vy: 0.0,
            angular_velocity: 0.35,
            mass: 1.1,
            friction: 0.5,
            shape: ShapeDesc2D::Capsule {
                half_height: 0.55,
                radius: 0.2,
            },
        },
    ));

    Some((world, tracked, gravity))
}

#[test]
fn same_hardware_replay_is_deterministic_2d() {
    let Some((mut world_a, tracked_a, gravity_a)) = build_determinism_scene_2d() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((mut world_b, tracked_b, gravity_b)) = build_determinism_scene_2d() else {
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
            (body_a.angular_velocity - body_b.angular_velocity).abs() < 1.0e-5,
            "angular replay drift for {}\na={:?}\nb={:?}\n{}\n{}",
            body_a.label,
            body_a.angular_velocity,
            body_b.angular_velocity,
            final_a,
            final_b
        );
    }
}

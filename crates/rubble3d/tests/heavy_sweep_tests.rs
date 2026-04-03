mod support;

use glam::{Quat, Vec3};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig};
use support::{add_tracked_body, collect_reports, cube_hull, octagon_hull, TrackedBody3D};

fn build_sweep_scene_3d(
    config: SimConfig,
    mass_ratio: f32,
    friction: f32,
) -> Option<(rubble3d::World, Vec<TrackedBody3D>, Vec3)> {
    let gravity = config.gravity;
    let mut world = support::try_world(config)?;
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
            friction,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(40.0, 0.5, 40.0),
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "heavy_box",
        RigidBodyDesc {
            position: Vec3::new(0.0, 0.52, 0.0),
            rotation: Quat::from_rotation_z(0.08),
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: mass_ratio,
            friction,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(0.7, 0.5, 0.6),
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "light_sphere",
        RigidBodyDesc {
            position: Vec3::new(-3.0, 3.5, 0.0),
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::new(5.0, -0.2, 0.0),
            angular_velocity: Vec3::new(0.0, 0.0, 0.5),
            mass: 1.0,
            friction,
            shape: ShapeDesc::Sphere { radius: 0.45 },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "capsule",
        RigidBodyDesc {
            position: Vec3::new(2.5, 5.0, 0.4),
            rotation: Quat::from_rotation_x(0.25),
            linear_velocity: Vec3::new(-2.5, 0.0, -0.25),
            angular_velocity: Vec3::new(0.0, 0.3, 0.0),
            mass: 1.5,
            friction,
            shape: ShapeDesc::Capsule {
                half_height: 0.7,
                radius: 0.2,
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "hull",
        RigidBodyDesc {
            position: Vec3::new(0.8, 6.0, -0.4),
            rotation: Quat::from_rotation_y(0.35),
            linear_velocity: Vec3::new(-0.5, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 0.5, 0.0),
            mass: 1.25,
            friction,
            shape: ShapeDesc::ConvexHull {
                vertices: octagon_hull(0.35, 0.55),
            },
        },
    ));

    Some((world, tracked, gravity))
}

#[test]
#[ignore = "heavy"]
fn parameter_sweep_stays_bounded_3d() {
    let dts = [1.0 / 120.0, 1.0 / 60.0];
    let iterations = [8, 20];
    let mass_ratios = [1.0, 50.0];
    let frictions = [0.0, 0.8];
    let betas = [5.0, 10.0];
    let k_starts = [1.0e3, 1.0e4];
    let warmstarts = [0.8, 0.95];

    for &dt in &dts {
        for &solver_iterations in &iterations {
            for &mass_ratio in &mass_ratios {
                for &friction in &frictions {
                    for &beta in &betas {
                        for &k_start in &k_starts {
                            for &warmstart_decay in &warmstarts {
                                let Some((mut world, tracked, gravity)) = build_sweep_scene_3d(
                                    SimConfig {
                                        gravity: Vec3::new(0.0, -9.81, 0.0),
                                        dt,
                                        solver_iterations,
                                        max_bodies: 64,
                                        beta,
                                        k_start,
                                        warmstart_decay,
                                        friction_default: friction,
                                    },
                                    mass_ratio,
                                    friction,
                                ) else {
                                    eprintln!("SKIP: No GPU adapter found");
                                    return;
                                };

                                let label = format!(
                                    "dt={dt:.6} iters={solver_iterations} mass_ratio={mass_ratio} friction={friction} beta={beta} k_start={k_start} warm={warmstart_decay}"
                                );
                                let reports = collect_reports(&mut world, &tracked, gravity, 240);
                                let initial_energy = reports.first().unwrap().metrics.total_energy;
                                let mut peak_energy = initial_energy;
                                let mut max_speed: f32 = 0.0;
                                let mut max_omega: f32 = 0.0;
                                let mut min_height = f32::INFINITY;

                                for report in &reports {
                                    peak_energy = peak_energy.max(report.metrics.total_energy);
                                    max_speed = max_speed.max(report.metrics.max_speed);
                                    max_omega = max_omega.max(report.metrics.max_angular_speed);
                                    for body in &report.bodies {
                                        assert!(
                                            body.position.is_finite()
                                                && body.linear_velocity.is_finite()
                                                && body.angular_velocity.is_finite(),
                                            "{label}: non-finite state\n{}",
                                            report
                                        );
                                        min_height = min_height.min(body.position.y);
                                    }
                                }

                                let last = reports.last().unwrap();
                                assert!(
                                    peak_energy <= initial_energy * 1.2 + 2.0,
                                    "{label}: mechanical energy grew too much: initial={initial_energy}, peak={peak_energy}\n{}",
                                    last
                                );
                                assert!(
                                    max_speed < 80.0,
                                    "{label}: translational speed blew up: max_speed={max_speed}\n{}",
                                    last
                                );
                                assert!(
                                    max_omega < 120.0,
                                    "{label}: angular speed blew up: max_omega={max_omega}\n{}",
                                    last
                                );
                                assert!(
                                    min_height > -5.0,
                                    "{label}: body fell too far below the floor: min_height={min_height}\n{}",
                                    last
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn build_chaos_scene_3d() -> Option<(rubble3d::World, Vec<TrackedBody3D>, Vec3)> {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = support::try_world(SimConfig {
        gravity,
        dt: 1.0 / 120.0,
        solver_iterations: 24,
        max_bodies: 128,
        beta: 10.0,
        k_start: 1.0e4,
        warmstart_decay: 0.95,
        friction_default: 0.7,
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
                half_extents: Vec3::new(40.0, 0.5, 40.0),
            },
        },
    ));

    for i in 0..12 {
        let x = (i % 4) as f32 * 1.4 - 2.1;
        let y = 2.0 + (i / 4) as f32 * 1.5;
        let z = if i % 2 == 0 { -0.35 } else { 0.35 };
        let label = Box::leak(format!("body_{i}").into_boxed_str());
        let shape = match i % 4 {
            0 => ShapeDesc::Sphere { radius: 0.4 },
            1 => ShapeDesc::Box {
                half_extents: Vec3::new(0.45, 0.35, 0.4),
            },
            2 => ShapeDesc::Capsule {
                half_height: 0.5,
                radius: 0.18,
            },
            _ => ShapeDesc::ConvexHull {
                vertices: cube_hull(Vec3::new(0.4, 0.3, 0.35)),
            },
        };
        tracked.push(add_tracked_body(
            &mut world,
            label,
            RigidBodyDesc {
                position: Vec3::new(x, y, z),
                rotation: Quat::from_rotation_z(0.1 * i as f32),
                linear_velocity: Vec3::new(0.1 * (i as f32 - 5.0), 0.0, 0.05 * (i as f32 % 3.0)),
                angular_velocity: Vec3::new(0.0, 0.1 * i as f32, 0.05 * i as f32),
                mass: 1.0 + 0.1 * i as f32,
                friction: 0.7,
                shape,
            },
        ));
    }

    Some((world, tracked, gravity))
}

#[test]
#[ignore = "heavy"]
fn long_horizon_chaos_scene_does_not_explode_3d() {
    let Some((mut world, tracked, gravity)) = build_chaos_scene_3d() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let reports = collect_reports(&mut world, &tracked, gravity, 1200);
    let settle_energy = reports[300].metrics.total_energy;
    let mut peak_energy = settle_energy;
    let mut max_speed: f32 = 0.0;
    let mut max_omega: f32 = 0.0;
    let mut min_height = f32::INFINITY;

    for report in &reports[300..] {
        peak_energy = peak_energy.max(report.metrics.total_energy);
        max_speed = max_speed.max(report.metrics.max_speed);
        max_omega = max_omega.max(report.metrics.max_angular_speed);
        for body in &report.bodies {
            assert!(
                body.position.is_finite()
                    && body.linear_velocity.is_finite()
                    && body.angular_velocity.is_finite(),
                "chaos scene diverged\n{}",
                report
            );
            min_height = min_height.min(body.position.y);
        }
    }

    let last = reports.last().unwrap();
    assert!(
        peak_energy <= settle_energy * 2.0 + 5.0,
        "chaos scene pumped too much energy after settling: settle={settle_energy}, peak={peak_energy}\n{}",
        last
    );
    assert!(
        max_speed < 120.0,
        "chaos scene developed runaway linear speed: max_speed={max_speed}\n{}",
        last
    );
    assert!(
        max_omega < 160.0,
        "chaos scene developed runaway angular speed: max_omega={max_omega}\n{}",
        last
    );
    assert!(
        min_height > -8.0,
        "chaos scene fell too far below the floor: min_height={min_height}\n{}",
        last
    );
}

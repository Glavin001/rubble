mod support;

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D};
use support::{add_tracked_body, collect_reports, regular_polygon, TrackedBody2D};

fn build_sweep_scene_2d(
    config: SimConfig2D,
    mass_ratio: f32,
    friction: f32,
) -> Option<(rubble2d::World2D, Vec<TrackedBody2D>, Vec2)> {
    let gravity = config.gravity;
    let mut world = support::try_world(config)?;
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
            friction,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(40.0, 0.5),
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "heavy_rect",
        RigidBodyDesc2D {
            x: 0.0,
            y: 0.52,
            angle: 0.08,
            vx: 0.0,
            vy: 0.0,
            angular_velocity: 0.0,
            mass: mass_ratio,
            friction,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(0.7, 0.45),
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "light_circle",
        RigidBodyDesc2D {
            x: -3.0,
            y: 3.5,
            angle: 0.0,
            vx: 5.0,
            vy: -0.2,
            angular_velocity: 0.5,
            mass: 1.0,
            friction,
            shape: ShapeDesc2D::Circle { radius: 0.45 },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "capsule",
        RigidBodyDesc2D {
            x: 2.5,
            y: 5.0,
            angle: 0.25,
            vx: -2.5,
            vy: 0.0,
            angular_velocity: 0.25,
            mass: 1.5,
            friction,
            shape: ShapeDesc2D::Capsule {
                half_height: 0.65,
                radius: 0.2,
            },
        },
    ));
    tracked.push(add_tracked_body(
        &mut world,
        "polygon",
        RigidBodyDesc2D {
            x: 0.8,
            y: 6.0,
            angle: 0.35,
            vx: -0.5,
            vy: 0.0,
            angular_velocity: 0.4,
            mass: 1.25,
            friction,
            shape: ShapeDesc2D::ConvexPolygon {
                vertices: regular_polygon(0.45, 7),
            },
        },
    ));

    Some((world, tracked, gravity))
}

#[test]
#[ignore = "heavy"]
fn parameter_sweep_stays_bounded_2d() {
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
                                let Some((mut world, tracked, gravity)) = build_sweep_scene_2d(
                                    SimConfig2D {
                                        gravity: Vec2::new(0.0, -9.81),
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
                                                && body.angle.is_finite()
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

fn build_chaos_scene_2d() -> Option<(rubble2d::World2D, Vec<TrackedBody2D>, Vec2)> {
    let gravity = Vec2::new(0.0, -9.81);
    let mut world = support::try_world(SimConfig2D {
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
                half_extents: Vec2::new(40.0, 0.5),
            },
        },
    ));

    for i in 0..14 {
        let x = (i % 5) as f32 * 1.2 - 2.4;
        let y = 2.0 + (i / 5) as f32 * 1.3;
        let label = Box::leak(format!("body_{i}").into_boxed_str());
        let shape = match i % 4 {
            0 => ShapeDesc2D::Circle { radius: 0.38 },
            1 => ShapeDesc2D::Rect {
                half_extents: Vec2::new(0.42, 0.32),
            },
            2 => ShapeDesc2D::Capsule {
                half_height: 0.48,
                radius: 0.16,
            },
            _ => ShapeDesc2D::ConvexPolygon {
                vertices: regular_polygon(0.4, 6),
            },
        };
        tracked.push(add_tracked_body(
            &mut world,
            label,
            RigidBodyDesc2D {
                x,
                y,
                angle: 0.1 * i as f32,
                vx: 0.12 * (i as f32 - 6.0),
                vy: 0.0,
                angular_velocity: 0.08 * i as f32,
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
fn long_horizon_chaos_scene_does_not_explode_2d() {
    let Some((mut world, tracked, gravity)) = build_chaos_scene_2d() else {
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
                    && body.angle.is_finite()
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

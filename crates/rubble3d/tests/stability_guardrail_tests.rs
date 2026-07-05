mod support;

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};
use support::{
    add_tracked_body, collect_reports, scene_report, step_n, try_world, SceneReport3D,
    TrackedBody3D,
};

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

#[allow(clippy::too_many_arguments)]
fn box_desc(
    x: f32,
    y: f32,
    z: f32,
    width: f32,
    height: f32,
    depth: f32,
    mass: f32,
    friction: f32,
) -> RigidBodyDesc {
    RigidBodyDesc {
        position: Vec3::new(x, y, z),
        mass,
        friction,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
        },
        ..Default::default()
    }
}

fn demo_world(max_bodies: usize) -> Option<(World, Vec3)> {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let world = try_world(SimConfig {
        gravity,
        max_bodies,
        ..Default::default()
    })?;
    Some((world, gravity))
}

fn build_demo_stack_scene() -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    let (mut world, gravity) = demo_world(32)?;
    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5),
    ));
    for _ in 0..10 {
        let i = tracked.len() - 1;
        tracked.push(add_tracked_body(
            &mut world,
            "box",
            box_desc(0.0, i as f32 * 1.5 + 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5),
        ));
    }
    Some((world, tracked, gravity))
}

fn build_demo_stack_ratio_scene() -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    let (mut world, gravity) = demo_world(16)?;
    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5),
    ));

    let mut top_y = 0.5;
    let mut size = 1.0;
    for _ in 0..4 {
        let half = size * 0.5;
        let center_y = top_y + half;
        tracked.push(add_tracked_body(
            &mut world,
            "box",
            box_desc(0.0, center_y, 0.0, size, size, size, 1.0, 0.5),
        ));
        top_y = center_y + half;
        size *= 2.0;
    }

    Some((world, tracked, gravity))
}

fn dynamic_centers(report: &SceneReport3D) -> Vec<f32> {
    let mut centers: Vec<f32> = report
        .bodies
        .iter()
        .filter(|body| body.mass > 0.0)
        .map(|body| body.position.y)
        .collect();
    centers.sort_by(f32::total_cmp);
    centers
}

fn tail_max_upward_speed(reports: &[SceneReport3D]) -> f32 {
    reports
        .iter()
        .flat_map(|report| report.bodies.iter())
        .filter(|body| body.mass > 0.0)
        .map(|body| body.linear_velocity.y)
        .fold(f32::NEG_INFINITY, f32::max)
}

fn tail_max_abs_vertical_speed(reports: &[SceneReport3D]) -> f32 {
    reports
        .iter()
        .flat_map(|report| report.bodies.iter())
        .filter(|body| body.mass > 0.0)
        .map(|body| body.linear_velocity.y.abs())
        .fold(0.0, f32::max)
}

#[test]
fn box_stack_three_high_guardrail() {
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
        handles.push(world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.5 + i as f32 * 1.1, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.5),
            },
            ..Default::default()
        }));
    }

    step_n(&mut world, 300);

    for (i, &handle) in handles.iter().enumerate() {
        let pos = world.get_position(handle).unwrap();
        assert!(pos.is_finite(), "Box {i} diverged: {pos}");
        assert!(pos.y > -3.0, "Box {i} fell through floor: y={}", pos.y);
    }

    let y0 = world.get_position(handles[0]).unwrap().y;
    let y1 = world.get_position(handles[1]).unwrap().y;
    let y2 = world.get_position(handles[2]).unwrap().y;
    assert!(
        y2 >= y1 - 1.0 && y1 >= y0 - 1.0,
        "Stack order violated: y0={y0}, y1={y1}, y2={y2}"
    );
}

#[test]
fn demo_stack_settles_without_support_loss_or_ejection_guardrail() {
    let Some((mut world, tracked, gravity)) = build_demo_stack_scene() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let reports = collect_reports(&mut world, &tracked, gravity, 480);
    let window = &reports[180..];
    let mut min_bottom_y = f32::INFINITY;
    let mut min_gap = f32::INFINITY;

    for report in window {
        let centers = dynamic_centers(report);
        min_bottom_y = min_bottom_y.min(centers[0]);
        for pair in centers.windows(2) {
            min_gap = min_gap.min(pair[1] - pair[0]);
        }
    }

    let max_upward_speed = tail_max_upward_speed(window);
    let last = window.last().unwrap();
    assert!(
        min_bottom_y > 0.9,
        "stack lost floor support in settle window: min_bottom_y={min_bottom_y}\n{}",
        last
    );
    assert!(
        min_gap > 0.9,
        "stack boxes interpenetrated too deeply while settling: min_gap={min_gap}\n{}",
        last
    );
    assert!(
        max_upward_speed < 0.6,
        "stack produced a large upward correction spike: max_upward_speed={max_upward_speed}\n{}",
        last
    );
}

#[test]
fn demo_stack_ratio_settles_without_vertical_oscillation_guardrail() {
    let Some((mut world, tracked, gravity)) = build_demo_stack_ratio_scene() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let reports = collect_reports(&mut world, &tracked, gravity, 720);
    let window = &reports[360..];
    let mut lowest_min = f32::INFINITY;
    let mut lowest_max = f32::NEG_INFINITY;
    let mut highest_min = f32::INFINITY;
    let mut highest_max = f32::NEG_INFINITY;

    for report in window {
        let centers = dynamic_centers(report);
        lowest_min = lowest_min.min(centers[0]);
        lowest_max = lowest_max.max(centers[0]);
        highest_min = highest_min.min(centers[3]);
        highest_max = highest_max.max(centers[3]);
    }

    let max_abs_vertical_speed = tail_max_abs_vertical_speed(window);
    let last = window.last().unwrap();
    assert!(
        lowest_max - lowest_min < 0.08,
        "bottom box kept bouncing in settle window: band={}\n{}",
        lowest_max - lowest_min,
        last
    );
    assert!(
        highest_max - highest_min < 0.12,
        "top box kept oscillating in settle window: band={}\n{}",
        highest_max - highest_min,
        last
    );
    assert!(
        max_abs_vertical_speed < 0.25,
        "stack-ratio never decayed to near-rest vertical motion: max_abs_vertical_speed={max_abs_vertical_speed}\n{}",
        last
    );
}

#[test]
fn resting_contact_energy_does_not_grow_guardrail() {
    let gravity_mag = 9.81_f32;
    let mut world = gpu_world!(SimConfig::default());

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..3 {
        handles.push(world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 0.5 + i as f32 * 1.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            },
            ..Default::default()
        }));
    }

    step_n(&mut world, 60);

    let settled_energy: f32 = handles
        .iter()
        .map(|&handle| {
            let pos = world.get_position(handle).unwrap();
            let vel = world.get_velocity(handle).unwrap();
            0.5 * vel.length_squared() + gravity_mag * pos.y
        })
        .sum();

    for step in 60..600 {
        world.step();
        let total_energy: f32 = handles
            .iter()
            .map(|&handle| {
                let pos = world.get_position(handle).unwrap();
                let vel = world.get_velocity(handle).unwrap();
                0.5 * vel.length_squared() + gravity_mag * pos.y
            })
            .sum();
        assert!(
            total_energy < settled_energy + settled_energy.abs() * 0.05 + 0.1,
            "Energy grew after settling at step {step}: E={total_energy:.3}, E_settled={settled_energy:.3}"
        );
    }

    let max_late_speed = handles
        .iter()
        .map(|&handle| world.get_velocity(handle).unwrap().length())
        .fold(0.0, f32::max);
    assert!(
        max_late_speed < 0.5,
        "Stack still moving at end: max_speed={max_late_speed:.3}"
    );
}

#[test]
fn long_horizon_stack_velocity_stays_bounded_guardrail() {
    let mut world = gpu_world!(SimConfig::default());

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for ix in 0..5 {
        for iz in 0..5 {
            handles.push(world.add_body(&RigidBodyDesc {
                position: Vec3::new(ix as f32 * 1.0 - 2.0, 0.5, iz as f32 * 1.0 - 2.0),
                mass: 1.0,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::new(0.4, 0.4, 0.4),
                },
                ..Default::default()
            }));
        }
    }

    step_n(&mut world, 120);

    for step in 120..900 {
        world.step();
        for (i, &handle) in handles.iter().enumerate() {
            let pos = world.get_position(handle).unwrap();
            let vel = world.get_velocity(handle).unwrap();
            let speed = vel.length();
            assert!(
                speed < 5.0,
                "Body {i} velocity diverged at step {step}: speed={speed:.3}, vel={vel}"
            );
            assert!(pos.y < 3.0, "Body {i} launched upward at step {step}: pos={pos}");
            assert!(pos.y > -0.5, "Body {i} fell through floor at step {step}: pos={pos}");
        }
    }

    let report = scene_report(
        &world,
        &handles
            .iter()
            .map(|&handle| TrackedBody3D {
                label: "box",
                handle,
                mass: 1.0,
                friction: 0.5,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::new(0.4, 0.4, 0.4),
                },
            })
            .collect::<Vec<_>>(),
        Vec3::new(0.0, -9.81, 0.0),
        900,
    );
    assert!(
        report.metrics.max_speed < 0.6,
        "Grid failed to settle near rest by end of guardrail run\n{}",
        report
    );
}

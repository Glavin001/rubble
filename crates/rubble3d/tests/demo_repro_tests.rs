mod support;

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};
use support::{add_tracked_body, collect_reports, try_world, SceneReport3D, TrackedBody3D};

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

fn build_demo_pyramid_scene() -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    const SIZE: usize = 16;

    let (mut world, gravity) = demo_world(256)?;
    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        box_desc(0.0, -0.5, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5),
    ));
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            tracked.push(add_tracked_body(
                &mut world,
                "box",
                box_desc(
                    x as f32 * 1.01 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                    y as f32 * 0.85 + 0.5,
                    0.0,
                    1.0,
                    0.5,
                    0.5,
                    1.0,
                    0.5,
                ),
            ));
        }
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
fn demo_stack_settles_without_support_loss_or_ejection_3d() {
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
        "stack lost floor support in the settle window: min_bottom_y={min_bottom_y}\n{}",
        last
    );
    assert!(
        min_gap > 0.9,
        "stack boxes interpenetrated too deeply while settling: min_gap={min_gap}\n{}",
        last
    );
    assert!(
        max_upward_speed < 0.5,
        "stack produced a large upward correction spike: max_upward_speed={max_upward_speed}\n{}",
        last
    );
}

#[test]
fn demo_stack_ratio_settles_without_persistent_vertical_oscillation_3d() {
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
        "bottom box kept bouncing in the settle window: band={}\n{}",
        lowest_max - lowest_min,
        last
    );
    assert!(
        highest_max - highest_min < 0.12,
        "top box kept oscillating in the settle window: band={}\n{}",
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
fn demo_pyramid_stays_supported_by_floor_without_exploding_3d() {
    let Some((mut world, tracked, gravity)) = build_demo_pyramid_scene() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let reports = collect_reports(&mut world, &tracked, gravity, 360);
    let window = &reports[120..];
    let min_center_y = reports
        .iter()
        .flat_map(|report| report.bodies.iter())
        .filter(|body| body.mass > 0.0)
        .map(|body| body.position.y)
        .fold(f32::INFINITY, f32::min);
    let tail_max_speed = window
        .iter()
        .map(|report| report.metrics.max_speed)
        .fold(0.0, f32::max);

    let last = window.last().unwrap();
    assert!(
        min_center_y > 0.15,
        "pyramid let bodies push substantially into or through the floor: min_center_y={min_center_y}\n{}",
        last
    );
    assert!(
        tail_max_speed < 10.0,
        "pyramid developed explosion-like speeds instead of damping out: tail_max_speed={tail_max_speed}\n{}",
        last
    );
}

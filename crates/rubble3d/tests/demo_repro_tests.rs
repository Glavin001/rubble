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

// ---------------------------------------------------------------------------
// Slanted grid — regression tests for explosion bug
// ---------------------------------------------------------------------------

/// Build a small slanted grid (4×8×4 = 128 boxes) that reproduces the
/// explosion behavior of the full 12×22×12 demo scene at lower cost.
fn build_slanted_grid_scene() -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = try_world(SimConfig {
        gravity,
        max_bodies: 256,
        ..Default::default()
    })?;
    let mut tracked = Vec::new();

    // Ground plane
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        RigidBodyDesc {
            position: Vec3::ZERO,
            mass: 0.0,
            shape: ShapeDesc::Plane {
                normal: Vec3::Y,
                distance: 0.0,
            },
            ..Default::default()
        },
    ));

    const NX: usize = 4;
    const NY: usize = 8;
    const NZ: usize = 4;
    let side = 0.42_f32;
    let gap = 0.08_f32;
    let pitch = side + gap;
    let half = side * 0.5;
    let ox = -((NX - 1) as f32 * pitch) * 0.5;
    let oz = -((NZ - 1) as f32 * pitch) * 0.5;
    let base_y = half + 0.03;
    let layer_shift = Vec3::new(0.038, 0.0, 0.026);

    for j in 0..NY {
        let shift = layer_shift * j as f32;
        for i in 0..NX {
            for k in 0..NZ {
                let x = ox + i as f32 * pitch + shift.x;
                let y = base_y + j as f32 * pitch;
                let z = oz + k as f32 * pitch + shift.z;
                tracked.push(add_tracked_body(
                    &mut world,
                    "box",
                    box_desc(x, y, z, side, side, side, 1.0, 0.5),
                ));
            }
        }
    }

    Some((world, tracked, gravity))
}

/// Build a matching non-slanted grid (4×8×4) as a control baseline.
fn build_regular_grid_scene() -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = try_world(SimConfig {
        gravity,
        max_bodies: 256,
        ..Default::default()
    })?;
    let mut tracked = Vec::new();

    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        RigidBodyDesc {
            position: Vec3::ZERO,
            mass: 0.0,
            shape: ShapeDesc::Plane {
                normal: Vec3::Y,
                distance: 0.0,
            },
            ..Default::default()
        },
    ));

    const NX: usize = 4;
    const NY: usize = 8;
    const NZ: usize = 4;
    let side = 0.42_f32;
    let gap = 0.08_f32;
    let pitch = side + gap;
    let half = side * 0.5;
    let ox = -((NX - 1) as f32 * pitch) * 0.5;
    let oz = -((NZ - 1) as f32 * pitch) * 0.5;
    let base_y = half + 0.03;

    for j in 0..NY {
        for i in 0..NX {
            for k in 0..NZ {
                let x = ox + i as f32 * pitch;
                let y = base_y + j as f32 * pitch;
                let z = oz + k as f32 * pitch;
                tracked.push(add_tracked_body(
                    &mut world,
                    "box",
                    box_desc(x, y, z, side, side, side, 1.0, 0.5),
                ));
            }
        }
    }

    Some((world, tracked, gravity))
}

#[test]
fn slanted_grid_boxes_stay_bounded() {
    let Some((mut world, tracked, gravity)) = build_slanted_grid_scene() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let reports = collect_reports(&mut world, &tracked, gravity, 480);

    // Check that no body has flown far from the origin (explosion detection).
    let max_distance = reports
        .iter()
        .flat_map(|r| r.bodies.iter())
        .filter(|b| b.mass > 0.0)
        .map(|b| b.position.length())
        .fold(0.0_f32, f32::max);

    let last = reports.last().unwrap();
    assert!(
        max_distance < 50.0,
        "slanted grid body flew too far from origin (explosion): max_distance={max_distance}\n{}",
        last
    );
}

#[test]
fn slanted_grid_max_speed_stays_reasonable() {
    let Some((mut world, tracked, gravity)) = build_slanted_grid_scene() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let reports = collect_reports(&mut world, &tracked, gravity, 480);

    // After initial settling (first 60 frames), max speed should stay bounded.
    // With explosion bug, speeds would exceed 50+ m/s.
    let window = &reports[60..];
    let tail_max_speed = window
        .iter()
        .map(|r| r.metrics.max_speed)
        .fold(0.0_f32, f32::max);

    let last = reports.last().unwrap();
    assert!(
        tail_max_speed < 30.0,
        "slanted grid developed explosion-like speeds: tail_max_speed={tail_max_speed}\n{}",
        last
    );
}

#[test]
fn slanted_grid_energy_does_not_diverge() {
    let Some((mut world, tracked, gravity)) = build_slanted_grid_scene() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let reports = collect_reports(&mut world, &tracked, gravity, 480);

    // Kinetic energy should not grow unbounded. With the explosion bug, KE
    // would increase dramatically as boxes accelerate outward.
    let initial_ke = reports[0].metrics.translational_ke + reports[0].metrics.rotational_ke;
    let max_ke = reports
        .iter()
        .map(|r| r.metrics.translational_ke + r.metrics.rotational_ke)
        .fold(0.0_f32, f32::max);

    // Allow KE to grow from gravitational potential conversion, but not
    // explosively. The initial PE for 128 boxes at various heights is large,
    // so we use an absolute bound rather than relative.
    let last = reports.last().unwrap();
    assert!(
        max_ke < 5000.0,
        "slanted grid kinetic energy diverged (explosion): initial_ke={initial_ke}, max_ke={max_ke}\n{}",
        last
    );
}

#[test]
fn slanted_grid_no_worse_than_regular_grid() {
    let Some((mut slanted_world, slanted_tracked, gravity)) = build_slanted_grid_scene() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((mut grid_world, grid_tracked, _)) = build_regular_grid_scene() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let slanted_reports = collect_reports(&mut slanted_world, &slanted_tracked, gravity, 480);
    let grid_reports = collect_reports(&mut grid_world, &grid_tracked, gravity, 480);

    // Compare max speeds in the settling window (after frame 120).
    let slanted_max = slanted_reports[120..]
        .iter()
        .map(|r| r.metrics.max_speed)
        .fold(0.0_f32, f32::max);
    let grid_max = grid_reports[120..]
        .iter()
        .map(|r| r.metrics.max_speed)
        .fold(0.0_f32, f32::max);

    // Slanted grid speeds should not be wildly higher than grid speeds.
    // With the explosion bug, slanted_max would be 10-100× higher.
    // Allow 5× margin for the expected slightly worse behavior of offset stacking.
    let ratio = if grid_max > 0.01 {
        slanted_max / grid_max
    } else {
        slanted_max
    };
    assert!(
        ratio < 5.0,
        "slanted grid is much worse than regular grid: slanted_max={slanted_max}, grid_max={grid_max}, ratio={ratio}"
    );
}

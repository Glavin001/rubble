use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

fn try_world(config: SimConfig) -> Option<World> {
    World::new(config).ok()
}

#[test]
fn gpu_render_step_builds_expected_batches() {
    let Some(mut world) = try_world(SimConfig::default()) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    world.add_body(&RigidBodyDesc {
        position: Vec3::new(-2.0, 3.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });
    world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 3.0, 0.0),
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });
    world.add_body(&RigidBodyDesc {
        position: Vec3::new(2.0, 3.0, 0.0),
        shape: ShapeDesc::Capsule {
            half_height: 0.6,
            radius: 0.25,
        },
        ..Default::default()
    });
    world.add_body(&RigidBodyDesc {
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });

    let _ = world.step_for_gpu_render();
    let counts = world.debug_render_instance_counts();
    assert_eq!(counts[0], 1);
    assert_eq!(counts[1], 1);
    assert_eq!(counts[2], 1);
    assert_eq!(world.last_step_timings().contact_fetch_ms, 0.0);
}

#[test]
fn cpu_sync_is_explicit_after_gpu_render_step() {
    let Some(mut world) = try_world(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        ..Default::default()
    }) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let body = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let y_before = world.get_position(body).unwrap().y;
    for _ in 0..10 {
        let _ = world.step_for_gpu_render();
    }

    let stale_y = world.get_position(body).unwrap().y;
    assert!(
        (stale_y - y_before).abs() < 1.0e-6,
        "CPU pose should stay unchanged until explicit sync"
    );

    world.sync_body_states_from_gpu();
    let synced_y = world.get_position(body).unwrap().y;
    assert!(
        synced_y < y_before,
        "Explicit sync should pull GPU pose to CPU"
    );
}

#[test]
fn gpu_render_path_handles_10k_grid_boxes() {
    let Some(mut world) = try_world(SimConfig::default()) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    world.add_body(&RigidBodyDesc {
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });

    let side = 100;
    for z in 0..side {
        for x in 0..side {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new(
                    x as f32 * 1.05 - side as f32 * 0.5,
                    0.5,
                    z as f32 * 1.05 - side as f32 * 0.5,
                ),
                shape: ShapeDesc::Box {
                    half_extents: Vec3::splat(0.5),
                },
                ..Default::default()
            });
        }
    }

    let _ = world.step_for_gpu_render();
    let counts = world.debug_render_instance_counts();
    assert_eq!(counts[1], 10_000);
    assert_eq!(counts[0], 0);
    assert_eq!(counts[2], 0);
}

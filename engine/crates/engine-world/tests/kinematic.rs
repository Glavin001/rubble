//! Kinematic bodies: infinite mass that moves at a set velocity or toward a
//! target pose, pushing and carrying dynamic bodies.

use glam::Quat;
use rubble::*;

fn world() -> World {
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [50.0, 0.5, 50.0])
        .unwrap();
    w
}

fn kinematic(w: &mut World, position: [f32; 3], velocity: [f32; 3]) -> BodyId {
    w.create_body(&BodyDef {
        body_type: BodyType::Kinematic,
        position,
        linear_velocity: velocity,
        ..BodyDef::default()
    })
    .unwrap()
}

fn run(w: &mut World, steps: u32) {
    for _ in 0..steps {
        w.step().unwrap();
        w.poll().unwrap();
    }
    w.poll_blocking().unwrap();
}

fn pos(w: &World, b: BodyId) -> [f32; 3] {
    w.read_body_state(b).unwrap().transform.position
}

#[test]
fn a_box_rides_a_moving_platform() {
    let mut w = world();
    let platform = kinematic(&mut w, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    let grippy = ShapeDef {
        material: SurfaceMaterial {
            friction: 0.8,
            ..SurfaceMaterial::default()
        },
        ..ShapeDef::default()
    };
    w.create_box_shape(platform, &grippy, [2.0, 0.25, 2.0])
        .unwrap();
    let rider = w.create_body(&BodyDef::dynamic([0.0, 1.76, 0.0])).unwrap();
    w.create_box_shape(rider, &grippy, [0.5; 3]).unwrap();
    run(&mut w, 120);
    let (p, r) = (pos(&w, platform), pos(&w, rider));
    assert!((p[0] - 2.0).abs() < 0.02, "the platform moved 2 m: {p:?}");
    assert!((p[1] - 1.0).abs() < 1e-4, "no gravity on it: {p:?}");
    assert!(
        (r[0] - p[0]).abs() < 0.1,
        "the rider came along: {r:?} on {p:?}"
    );
    assert!((r[1] - 1.75).abs() < 0.06, "still on top: {r:?}");
    let v = w.read_body_state(platform).unwrap().linear_velocity;
    assert!(
        (v[0] - 1.0).abs() < 1e-5 && v[1].abs() < 1e-6,
        "it keeps its velocity: {v:?}"
    );
}

#[test]
fn a_paddle_pushes_boxes_without_slowing() {
    let mut w = world();
    let paddle = kinematic(&mut w, [-3.0, 0.6, 0.0], [2.0, 0.0, 0.0]);
    w.create_box_shape(paddle, &ShapeDef::default(), [0.2, 0.5, 1.0])
        .unwrap();
    let boxes: Vec<BodyId> = (0..3)
        .map(|k| {
            let b = w
                .create_body(&BodyDef::dynamic([0.0, 0.5 + k as f32 * 1.01, 0.0]))
                .unwrap();
            w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])
                .unwrap();
            b
        })
        .collect();
    run(&mut w, 180);
    let p = pos(&w, paddle);
    assert!((p[0] - 3.0).abs() < 0.02, "the paddle kept its pace: {p:?}");
    assert!(pos(&w, boxes[0])[0] > 1.5, "it shoved the bottom box");
}

#[test]
fn target_transforms_track_without_drift() {
    let mut w = world();
    let body = kinematic(&mut w, [0.0, 2.0, 0.0], [0.0; 3]);
    w.create_box_shape(body, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    let dt = w.solver().dt;
    let mut target = Transform::IDENTITY;
    for k in 1..=240 {
        let t = k as f32 * dt;
        let yaw = Quat::from_rotation_y(0.8 * t);
        target = Transform::new([3.0 * (0.7 * t).sin(), 2.0 + 0.5 * t, 0.0], yaw.to_array());
        w.body_set_target_transform(body, target).unwrap();
        w.step().unwrap();
        w.poll().unwrap();
    }
    w.poll_blocking().unwrap();
    let s = w.read_body_state(body).unwrap().transform;
    for k in 0..3 {
        assert!(
            (s.position[k] - target.position[k]).abs() < 1e-3,
            "at the last target: {s:?} vs {target:?}"
        );
    }
    let dot: f32 = (0..4).map(|k| s.rotation[k] * target.rotation[k]).sum();
    assert!(dot.abs() > 0.9999, "facing the target: {s:?} vs {target:?}");
}

#[test]
fn impulses_and_mass_do_not_apply_to_kinematic_bodies() {
    let mut w = world();
    let body = kinematic(&mut w, [0.0, 3.0, 0.0], [0.0; 3]);
    w.create_box_shape(body, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    w.body_apply_linear_impulse_to_center(body, [100.0, 0.0, 0.0], true)
        .unwrap();
    assert!(w.body_apply_mass_from_shapes(body).is_err());
    let dynamic = w.create_body(&BodyDef::dynamic([5.0, 3.0, 0.0])).unwrap();
    w.create_box_shape(dynamic, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    assert!(
        w.body_set_target_transform(dynamic, Transform::IDENTITY)
            .is_err(),
        "targets are for kinematic bodies"
    );
    run(&mut w, 60);
    let p = pos(&w, body);
    assert!(
        p[0].abs() < 1e-6 && (p[1] - 3.0).abs() < 1e-6,
        "unmoved: {p:?}"
    );
}

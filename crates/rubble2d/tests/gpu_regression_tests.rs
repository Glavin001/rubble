use glam::{Vec2, Vec4};
use rubble2d::{
    gpu::{GpuPipeline2D, ShapeInfo},
    RigidBodyDesc2D, ShapeDesc2D, SimConfig2D, World2D,
};
use rubble_math::{Contact2D, RigidBodyState2D};
use rubble_shapes2d::{CapsuleData2D, CircleData, ConvexPolygonData, ConvexVertex2D, RectData};

const SHAPE_CIRCLE_2D: u32 = 0;
const SHAPE_RECT_2D: u32 = 1;
const INITIAL_PENALTY: f32 = 1.0e4;

macro_rules! gpu_pipeline_2d {
    ($max_bodies:expr) => {
        match GpuPipeline2D::try_new($max_bodies) {
            Some(p) => p,
            None => {
                eprintln!("SKIP: No GPU adapter found");
                return;
            }
        }
    };
}

macro_rules! gpu_world_2d {
    ($config:expr) => {
        match World2D::new($config) {
            Ok(w) => w,
            Err(_) => {
                eprintln!("SKIP: No GPU adapter found");
                return;
            }
        }
    };
}

fn step_n(world: &mut World2D, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

fn rect_inv_inertia(mass: f32, half_extents: Vec2) -> f32 {
    if mass <= 0.0 {
        return 0.0;
    }
    let w = 2.0 * half_extents.x;
    let h = 2.0 * half_extents.y;
    12.0 / (mass * (w * w + h * h))
}

fn rect_state(
    pos: Vec2,
    angle: f32,
    inv_mass: f32,
    inv_inertia: f32,
    lin_vel: Vec2,
    omega: f32,
    friction: f32,
) -> RigidBodyState2D {
    let mut state = RigidBodyState2D::new(pos.x, pos.y, angle, inv_mass, lin_vel.x, lin_vel.y, omega);
    state._pad0 = Vec4::new(friction, inv_inertia, 0.0, 0.0);
    state
}

fn circle_state(
    pos: Vec2,
    inv_mass: f32,
    lin_vel: Vec2,
    omega: f32,
    friction: f32,
    inv_inertia: f32,
) -> RigidBodyState2D {
    let mut state = RigidBodyState2D::new(pos.x, pos.y, 0.0, inv_mass, lin_vel.x, lin_vel.y, omega);
    state._pad0 = Vec4::new(friction, inv_inertia, 0.0, 0.0);
    state
}

fn run_rect_floor_step(
    body_pos: Vec2,
    body_angle: f32,
    inv_mass: f32,
    inv_inertia: f32,
    body_velocity: Vec2,
    body_omega: f32,
    gravity: Vec2,
    solver_iterations: u32,
    warm_contacts: Option<&[Contact2D]>,
) -> Option<(Vec<RigidBodyState2D>, Vec<Contact2D>)> {
    let mut pipeline = GpuPipeline2D::try_new(2)?;
    let floor_he = Vec2::new(4.0, 0.5);
    let body_he = Vec2::new(1.0, 0.5);
    let states = vec![
        rect_state(Vec2::new(0.0, -0.5), 0.0, 0.0, 0.0, Vec2::ZERO, 0.0, 1.0),
        rect_state(
            body_pos,
            body_angle,
            inv_mass,
            inv_inertia,
            body_velocity,
            body_omega,
            1.0,
        ),
    ];
    let shape_infos = vec![
        ShapeInfo {
            shape_type: SHAPE_RECT_2D,
            shape_index: 0,
        },
        ShapeInfo {
            shape_type: SHAPE_RECT_2D,
            shape_index: 1,
        },
    ];
    let rects = vec![
        RectData {
            half_extents: floor_he.extend(0.0).extend(0.0),
        },
        RectData {
            half_extents: body_he.extend(0.0).extend(0.0),
        },
    ];

    pipeline.upload(
        &states,
        &shape_infos,
        &[],
        &rects,
        &[],
        &[],
        &[],
        gravity,
        1.0 / 60.0,
        solver_iterations,
    );
    Some(pipeline.step_with_contacts(
        states.len() as u32,
        solver_iterations,
        warm_contacts,
    ))
}

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn rect_angular_response_matches_inertia() {
    let Some((low_states, _)) = run_rect_floor_step(
        Vec2::new(0.0, 0.62),
        0.45,
        1.0,
        0.2,
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        12,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((high_states, _)) = run_rect_floor_step(
        Vec2::new(0.0, 0.62),
        0.45,
        1.0,
        4.0,
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        12,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let low_omega = low_states[1].angular_velocity().abs();
    let high_omega = high_states[1].angular_velocity().abs();
    let low_angle_delta = (low_states[1].angle() - 0.45).abs();
    let high_angle_delta = (high_states[1].angle() - 0.45).abs();

    assert!(low_omega.is_finite() && high_omega.is_finite());
    assert!(
        high_omega > low_omega * 1.5,
        "higher inverse inertia should create more angular response: low={low_omega}, high={high_omega}"
    );
    assert!(
        high_angle_delta > low_angle_delta * 1.3,
        "higher inverse inertia should rotate farther during solve: low={low_angle_delta}, high={high_angle_delta}"
    );
}

#[test]
fn circle_vs_rect_rotation_ratio() {
    let circle_config = SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 12,
        friction_default: 1.0,
        ..Default::default()
    };

    let mut circle_world = gpu_world_2d!(circle_config);
    let _floor_circle = circle_world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -0.5,
        mass: 0.0,
        friction: 1.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 0.5),
        },
        ..Default::default()
    });
    let circle = circle_world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.52,
        vx: 5.0,
        mass: 1.0,
        friction: 1.0,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });
    step_n(&mut circle_world, 120);
    let circle_omega = circle_world.get_angular_velocity(circle).unwrap().abs();

    let mut rect_world = gpu_world_2d!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 12,
        friction_default: 1.0,
        ..Default::default()
    });
    let _floor_rect = rect_world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -0.5,
        mass: 0.0,
        friction: 1.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(10.0, 0.5),
        },
        ..Default::default()
    });
    let rect = rect_world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 0.52,
        vx: 5.0,
        mass: 1.0,
        friction: 1.0,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::splat(0.5),
        },
        ..Default::default()
    });
    step_n(&mut rect_world, 120);
    let rect_omega = rect_world.get_angular_velocity(rect).unwrap().abs();

    assert!(circle_omega.is_finite() && rect_omega.is_finite());
    assert!(
        circle_omega > rect_omega + 0.5,
        "circles should pick up more rolling angular velocity than flat rectangles: circle={circle_omega}, rect={rect_omega}"
    );
}

#[test]
fn heavy_rect_light_rect_angular() {
    let body_he = Vec2::new(1.0, 0.5);
    let light_inv_mass = 1.0 / 1.0;
    let heavy_inv_mass = 1.0 / 10.0;
    let light_inv_inertia = rect_inv_inertia(1.0, body_he);
    let heavy_inv_inertia = rect_inv_inertia(10.0, body_he);

    let Some((light_states, _)) = run_rect_floor_step(
        Vec2::new(0.0, 0.62),
        0.45,
        light_inv_mass,
        light_inv_inertia,
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        12,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((heavy_states, _)) = run_rect_floor_step(
        Vec2::new(0.0, 0.62),
        0.45,
        heavy_inv_mass,
        heavy_inv_inertia,
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        12,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let light_omega = light_states[1].angular_velocity().abs();
    let heavy_omega = heavy_states[1].angular_velocity().abs();
    let ratio = if heavy_omega > 1e-5 {
        light_omega / heavy_omega
    } else {
        f32::INFINITY
    };

    assert!(light_omega.is_finite() && heavy_omega.is_finite());
    assert!(
        ratio > 0.6 && ratio < 1.4,
        "same-shape off-center angular response should be approximately mass invariant: light={light_omega}, heavy={heavy_omega}, ratio={ratio}"
    );
}

#[test]
fn rect_on_rect_generates_two_contacts() {
    let Some((_, contacts)) = run_rect_floor_step(
        Vec2::new(0.0, 0.45),
        0.0,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        0,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    assert_eq!(
        contacts.len(),
        2,
        "axis-aligned face contact should emit two manifold points, got {}",
        contacts.len()
    );
    assert!(contacts.iter().all(|c| c.feature_id != 0));
}

#[test]
fn rotated_rect_on_rect_manifold() {
    let Some((_, contacts)) = run_rect_floor_step(
        Vec2::new(0.0, 0.445),
        0.03,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        0,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    assert!(
        contacts.len() >= 2,
        "slightly rotated face contact should retain a manifold, got {} contact(s)",
        contacts.len()
    );
}

#[test]
fn feature_ids_stable_under_small_motion_2d() {
    let Some((_, contacts_a)) = run_rect_floor_step(
        Vec2::new(0.0, 0.45),
        0.0,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        0,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((_, contacts_b)) = run_rect_floor_step(
        Vec2::new(0.05, 0.45),
        0.0,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        0,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let ids_a: std::collections::BTreeSet<u32> = contacts_a.iter().map(|c| c.feature_id).collect();
    let ids_b: std::collections::BTreeSet<u32> = contacts_b.iter().map(|c| c.feature_id).collect();
    let overlap: Vec<u32> = ids_a.intersection(&ids_b).copied().collect();

    assert!(
        !overlap.is_empty(),
        "small translations should preserve manifold feature ids: a={ids_a:?}, b={ids_b:?}"
    );
}

#[test]
fn stiffness_ramp_conditional_2d() {
    let Some((_, contacts)) = run_rect_floor_step(
        Vec2::new(0.0, 0.43),
        0.0,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        4,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    assert!(!contacts.is_empty());
    assert!(
        contacts.iter().any(|c| c.lambda_penalty.z > INITIAL_PENALTY),
        "normal penalty should ramp under penetration: {contacts:?}"
    );
    assert!(
        contacts
            .iter()
            .all(|c| approx_eq(c.lambda_penalty.w, INITIAL_PENALTY, 1.0)),
        "purely normal resting contacts should not ramp tangential stiffness: {contacts:?}"
    );
}

#[test]
fn lambda_accumulates_correctly_2d() {
    let Some((_, contacts_one_iter)) = run_rect_floor_step(
        Vec2::new(0.0, 0.43),
        0.0,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        1,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((_, contacts_many_iters)) = run_rect_floor_step(
        Vec2::new(0.0, 0.43),
        0.0,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        6,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let lambda_one: f32 = contacts_one_iter.iter().map(|c| c.lambda_penalty.x).sum::<f32>().abs();
    let lambda_many: f32 = contacts_many_iters
        .iter()
        .map(|c| c.lambda_penalty.x)
        .sum::<f32>()
        .abs();

    assert!(
        lambda_many >= lambda_one,
        "more dual updates should accumulate at least as much normal lambda: one_iter={lambda_one}, many_iters={lambda_many}"
    );
}

#[test]
fn warm_start_decay_2d() {
    let Some((_, first_contacts)) = run_rect_floor_step(
        Vec2::new(0.0, 0.43),
        0.0,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        6,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    assert!(!first_contacts.is_empty());

    let Some((_, second_contacts)) = run_rect_floor_step(
        Vec2::new(0.03, 0.43),
        0.0,
        1.0,
        rect_inv_inertia(1.0, Vec2::new(1.0, 0.5)),
        Vec2::ZERO,
        0.0,
        Vec2::ZERO,
        0,
        Some(&first_contacts),
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    assert!(!second_contacts.is_empty());

    let first_by_feature: std::collections::HashMap<u32, &Contact2D> =
        first_contacts.iter().map(|c| (c.feature_id, c)).collect();

    let mut matched = 0usize;
    for contact in &second_contacts {
        if let Some(prev) = first_by_feature.get(&contact.feature_id) {
            matched += 1;
            let expected = prev.lambda_penalty * 0.95;
            let delta = (contact.lambda_penalty - expected).abs();
            assert!(
                delta.max_element() < 1e-3,
                "warm-start should decay cached lambdas and penalties by gamma: feature={}, expected={expected:?}, actual={:?}",
                contact.feature_id,
                contact.lambda_penalty
            );
        }
    }

    assert!(matched > 0, "expected at least one persisted contact feature");
}

#[test]
fn circle_contact_warm_start_keeps_feature_ids() {
    let mut pipeline = gpu_pipeline_2d!(2);
    let circle_radius = 0.5;
    let floor_he = Vec2::new(5.0, 0.5);
    let circle_inv_inertia = 1.0 / (0.5 * 1.0 * circle_radius * circle_radius);

    let states = vec![
        rect_state(Vec2::new(0.0, -0.5), 0.0, 0.0, 0.0, Vec2::ZERO, 0.0, 1.0),
        circle_state(
            Vec2::new(0.0, 0.48),
            1.0,
            Vec2::new(2.0, 0.0),
            0.0,
            1.0,
            circle_inv_inertia,
        ),
    ];
    let shape_infos = vec![
        ShapeInfo {
            shape_type: SHAPE_RECT_2D,
            shape_index: 0,
        },
        ShapeInfo {
            shape_type: SHAPE_CIRCLE_2D,
            shape_index: 0,
        },
    ];
    let rects = vec![RectData {
        half_extents: floor_he.extend(0.0).extend(0.0),
    }];
    let circles = vec![CircleData {
        radius: circle_radius,
        _pad: [0.0; 3],
    }];

    pipeline.upload(
        &states,
        &shape_infos,
        &circles,
        &rects,
        &Vec::<ConvexPolygonData>::new(),
        &Vec::<ConvexVertex2D>::new(),
        &Vec::<CapsuleData2D>::new(),
        Vec2::new(0.0, -9.81),
        1.0 / 60.0,
        4,
    );
    let (_, first_contacts) = pipeline.step_with_contacts(2, 4, None);

    pipeline.upload(
        &states,
        &shape_infos,
        &circles,
        &rects,
        &Vec::<ConvexPolygonData>::new(),
        &Vec::<ConvexVertex2D>::new(),
        &Vec::<CapsuleData2D>::new(),
        Vec2::new(0.0, -9.81),
        1.0 / 60.0,
        0,
    );
    let (_, second_contacts) = pipeline.step_with_contacts(2, 0, Some(&first_contacts));

    assert_eq!(first_contacts.len(), second_contacts.len());
    assert_eq!(
        first_contacts
            .iter()
            .map(|c| c.feature_id)
            .collect::<Vec<_>>(),
        second_contacts
            .iter()
            .map(|c| c.feature_id)
            .collect::<Vec<_>>()
    );
}

#[test]
fn circle_on_rect_remains_supported() {
    let mut world = gpu_world_2d!(SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        friction_default: 0.8,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: -1.0,
        mass: 0.0,
        friction: 0.8,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(50.0, 1.0),
        },
        ..Default::default()
    });
    let circle = world.add_body(&RigidBodyDesc2D {
        x: 0.0,
        y: 1.0,
        vx: 6.0,
        vy: 0.0,
        mass: 1.0,
        friction: 0.8,
        shape: ShapeDesc2D::Circle { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..180 {
        world.step();
        let pos = world.get_position(circle).unwrap();
        let vel = world.get_velocity(circle).unwrap();
        assert!(pos.is_finite(), "circle on rect diverged at step {step}: {pos}");
        assert!(
            vel.is_finite(),
            "circle on rect velocity diverged at step {step}: {vel}"
        );
    }

    let pos = world.get_position(circle).unwrap();
    assert!(pos.y > -1.0, "circle fell through rect floor: y={}", pos.y);
}

use glam::{Mat3, Quat, Vec3, Vec4};
use rubble3d::{gpu::GpuPipeline, RigidBodyDesc, ShapeDesc, SimConfig, World};
use rubble_math::{
    Contact3D, RigidBodyProps3D, RigidBodyState3D, FLAG_STATIC, SHAPE_BOX, SHAPE_PLANE,
    SHAPE_SPHERE,
};
use rubble_shapes3d::{BoxData, CompoundShape, CompoundShapeGpu, SphereData};

const INITIAL_PENALTY: f32 = 1.0e4;

macro_rules! gpu_world_3d {
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

fn step_n(world: &mut World, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

fn box_inv_inertia(mass: f32, half_extents: Vec3) -> Mat3 {
    if mass <= 0.0 {
        return Mat3::ZERO;
    }
    let size = 2.0 * half_extents;
    let i_x = mass * (size.y * size.y + size.z * size.z) / 12.0;
    let i_y = mass * (size.x * size.x + size.z * size.z) / 12.0;
    let i_z = mass * (size.x * size.x + size.y * size.y) / 12.0;
    Mat3::from_diagonal(Vec3::new(1.0 / i_x, 1.0 / i_y, 1.0 / i_z))
}

fn run_box_floor_step(
    body_pos: Vec3,
    body_rot: Quat,
    body_lin_vel: Vec3,
    body_ang_vel: Vec3,
    body_mass: f32,
    friction: f32,
    solver_iterations: u32,
    warm_contacts: Option<&[Contact3D]>,
) -> Option<(Vec<RigidBodyState3D>, Vec<Contact3D>)> {
    let mut pipeline = GpuPipeline::try_new(2)?;
    let floor_he = Vec3::new(4.0, 0.5, 4.0);
    let body_he = Vec3::new(1.0, 0.5, 1.0);
    let inv_mass = if body_mass > 0.0 { 1.0 / body_mass } else { 0.0 };
    let states = vec![
        RigidBodyState3D::new(Vec3::new(0.0, -0.5, 0.0), 0.0, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO),
        RigidBodyState3D::new(body_pos, inv_mass, body_rot, body_lin_vel, body_ang_vel),
    ];
    let props = vec![
        RigidBodyProps3D::new(Mat3::ZERO, 1.0, SHAPE_BOX, 0, FLAG_STATIC),
        RigidBodyProps3D::new(box_inv_inertia(body_mass, body_he), friction, SHAPE_BOX, 1, 0),
    ];
    let boxes = vec![
        BoxData {
            half_extents: floor_he.extend(0.0),
        },
        BoxData {
            half_extents: body_he.extend(0.0),
        },
    ];

    pipeline.upload(
        &states,
        &props,
        &[],
        &boxes,
        &[],
        &[],
        &[],
        &[],
        &Vec::<CompoundShapeGpu>::new(),
        &[],
        &Vec::<CompoundShape>::new(),
        Vec3::ZERO,
        1.0 / 60.0,
        solver_iterations,
    );
    Some(pipeline.step_with_contacts(
        states.len() as u32,
        solver_iterations,
        warm_contacts,
    ))
}

fn run_sphere_plane_step(
    sphere_pos: Vec3,
    sphere_lin_vel: Vec3,
    sphere_mass: f32,
    solver_iterations: u32,
    warm_contacts: Option<&[Contact3D]>,
) -> Option<(Vec<RigidBodyState3D>, Vec<Contact3D>)> {
    let mut pipeline = GpuPipeline::try_new(2)?;
    let inv_mass = if sphere_mass > 0.0 {
        1.0 / sphere_mass
    } else {
        0.0
    };
    let radius = 0.5;
    let inv_inertia = if sphere_mass > 0.0 {
        let inertia = 2.0 * sphere_mass * radius * radius / 5.0;
        Mat3::from_diagonal(Vec3::splat(1.0 / inertia))
    } else {
        Mat3::ZERO
    };
    let states = vec![
        RigidBodyState3D::new(Vec3::ZERO, 0.0, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO),
        RigidBodyState3D::new(
            sphere_pos,
            inv_mass,
            Quat::IDENTITY,
            sphere_lin_vel,
            Vec3::ZERO,
        ),
    ];
    let props = vec![
        RigidBodyProps3D::new(Mat3::ZERO, 1.0, SHAPE_PLANE, 0, FLAG_STATIC),
        RigidBodyProps3D::new(inv_inertia, 1.0, SHAPE_SPHERE, 0, 0),
    ];
    let spheres = vec![SphereData {
        radius,
        _pad: [0.0; 3],
    }];
    let planes = vec![Vec4::new(0.0, 1.0, 0.0, 0.0)];

    pipeline.upload(
        &states,
        &props,
        &spheres,
        &[],
        &[],
        &[],
        &[],
        &planes,
        &Vec::<CompoundShapeGpu>::new(),
        &[],
        &Vec::<CompoundShape>::new(),
        Vec3::ZERO,
        1.0 / 60.0,
        solver_iterations,
    );
    Some(pipeline.step_with_contacts(
        states.len() as u32,
        solver_iterations,
        warm_contacts,
    ))
}

fn simulate_sphere_plane_frames(
    steps: usize,
    solver_iterations: u32,
    use_warmstart: bool,
) -> Option<Vec<RigidBodyState3D>> {
    let mut pipeline = GpuPipeline::try_new(2)?;
    let radius = 0.5;
    let mass = 1.0;
    let inertia = 2.0 * mass * radius * radius / 5.0;
    let props = vec![
        RigidBodyProps3D::new(Mat3::ZERO, 1.0, SHAPE_PLANE, 0, FLAG_STATIC),
        RigidBodyProps3D::new(
            Mat3::from_diagonal(Vec3::splat(1.0 / inertia)),
            1.0,
            SHAPE_SPHERE,
            0,
            0,
        ),
    ];
    let spheres = vec![SphereData {
        radius,
        _pad: [0.0; 3],
    }];
    let planes = vec![Vec4::new(0.0, 1.0, 0.0, 0.0)];
    let mut states = vec![
        RigidBodyState3D::new(Vec3::ZERO, 0.0, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO),
        RigidBodyState3D::new(
            Vec3::new(0.0, 2.0, 0.0),
            1.0 / mass,
            Quat::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
        ),
    ];
    let mut prev_contacts = Vec::new();

    for _ in 0..steps {
        pipeline.upload(
            &states,
            &props,
            &spheres,
            &[],
            &[],
            &[],
            &[],
            &planes,
            &Vec::<CompoundShapeGpu>::new(),
            &[],
            &Vec::<CompoundShape>::new(),
            Vec3::new(0.0, -9.81, 0.0),
            1.0 / 60.0,
            solver_iterations,
        );
        let warm = if use_warmstart && !prev_contacts.is_empty() {
            Some(prev_contacts.as_slice())
        } else {
            None
        };
        let (next_states, contacts) =
            pipeline.step_with_contacts(states.len() as u32, solver_iterations, warm);
        states = next_states;
        prev_contacts = contacts;
    }

    Some(states)
}

#[test]
fn box_on_box_generates_multi_contact_manifold() {
    let Some((_, contacts)) = run_box_floor_step(
        Vec3::new(0.0, 0.45, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
        0,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    assert!(
        contacts.len() >= 2,
        "face-face box contact should produce a manifold, got {} contact(s)",
        contacts.len()
    );
    assert!(contacts.iter().all(|c| c.feature_id != 0));
}

#[test]
fn rotated_box_on_box_manifold() {
    let Some((_, contacts)) = run_box_floor_step(
        Vec3::new(0.0, 0.46, 0.0),
        Quat::from_rotation_z(0.08),
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
        0,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    assert!(
        contacts.len() >= 2,
        "slightly rotated box should still keep a manifold, got {} contact(s)",
        contacts.len()
    );
}

#[test]
fn feature_ids_stable_under_small_motion_3d() {
    let Some((_, contacts_a)) = run_box_floor_step(
        Vec3::new(0.0, 0.45, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
        0,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((_, contacts_b)) = run_box_floor_step(
        Vec3::new(0.05, 0.45, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
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
        "small translations should preserve at least part of the manifold identity: a={ids_a:?}, b={ids_b:?}"
    );
}

#[test]
fn stiffness_ramp_conditional_3d() {
    let Some((_, contacts)) = run_box_floor_step(
        Vec3::new(0.0, 0.43, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
        4,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    assert!(!contacts.is_empty());
    assert!(
        contacts.iter().any(|c| c.penalty.x > INITIAL_PENALTY),
        "normal penalty should ramp under penetration: {contacts:?}"
    );
    assert!(
        contacts
            .iter()
            .all(|c| (c.penalty.y - INITIAL_PENALTY).abs() <= 1.0 && (c.penalty.z - INITIAL_PENALTY).abs() <= 1.0),
        "purely normal resting contacts should not ramp tangential stiffness: {contacts:?}"
    );
}

#[test]
fn lambda_accumulates_correctly_3d() {
    let Some((_, contacts_one_iter)) = run_box_floor_step(
        Vec3::new(0.0, 0.43, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
        1,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let Some((_, contacts_many_iters)) = run_box_floor_step(
        Vec3::new(0.0, 0.43, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
        6,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let lambda_one: f32 = contacts_one_iter.iter().map(|c| c.lambda.x).sum::<f32>().abs();
    let lambda_many: f32 = contacts_many_iters.iter().map(|c| c.lambda.x).sum::<f32>().abs();

    assert!(
        lambda_many >= lambda_one,
        "more dual updates should accumulate at least as much normal lambda: one_iter={lambda_one}, many_iters={lambda_many}"
    );
}

#[test]
fn warm_start_matches_by_feature_not_distance_3d() {
    let Some((_, first_contacts)) = run_box_floor_step(
        Vec3::new(0.0, 0.43, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
        6,
        None,
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    assert!(!first_contacts.is_empty());

    let Some((_, second_contacts)) = run_box_floor_step(
        Vec3::new(0.05, 0.43, 0.0),
        Quat::IDENTITY,
        Vec3::ZERO,
        Vec3::ZERO,
        1.0,
        1.0,
        0,
        Some(&first_contacts),
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    assert!(!second_contacts.is_empty());

    let first_by_feature: std::collections::HashMap<u32, &Contact3D> =
        first_contacts.iter().map(|c| (c.feature_id, c)).collect();

    let mut matched = 0usize;
    for contact in &second_contacts {
        if let Some(prev) = first_by_feature.get(&contact.feature_id) {
            matched += 1;
            let expected_lambda = prev.lambda * 0.95;
            let expected_penalty = prev.penalty * 0.95;
            let lambda_delta = (contact.lambda - expected_lambda).abs();
            let penalty_delta = (contact.penalty - expected_penalty).abs();
            assert!(
                lambda_delta.max_element() < 1e-3,
                "warm-start lambda should track by feature id: feature={}, expected={expected_lambda:?}, actual={:?}",
                contact.feature_id,
                contact.lambda
            );
            assert!(
                penalty_delta.max_element() < 1e-3,
                "warm-start penalty should track by feature id: feature={}, expected={expected_penalty:?}, actual={:?}",
                contact.feature_id,
                contact.penalty
            );
        }
    }

    assert!(matched > 0, "expected at least one persisted feature after small motion");
}

#[test]
fn friction_induces_angular_velocity() {
    let mut world = gpu_world_3d!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 12,
        friction_default: 1.0,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        friction: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
        ..Default::default()
    });
    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.52, 0.0),
        linear_velocity: Vec3::new(6.0, 0.0, 0.0),
        mass: 1.0,
        friction: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 120);
    let omega = world.get_angular_velocity(sphere).unwrap();

    assert!(omega.is_finite(), "angular velocity diverged: {omega}");
    assert!(
        omega.length() > 0.5 && omega.z.abs() > 0.2,
        "friction should convert sliding into spin: omega={omega}"
    );
}

#[test]
fn friction_torque_stops_spinning() {
    let mut world = gpu_world_3d!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 12,
        friction_default: 1.0,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        friction: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
        ..Default::default()
    });
    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.52, 0.0),
        angular_velocity: Vec3::new(0.0, 0.0, 15.0),
        mass: 1.0,
        friction: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let initial = world.get_angular_velocity(sphere).unwrap().length();
    step_n(&mut world, 120);
    let final_omega = world.get_angular_velocity(sphere).unwrap().length();

    assert!(final_omega.is_finite());
    assert!(
        final_omega < initial * 0.85,
        "contact friction should bleed off spinning energy: initial={initial}, final={final_omega}"
    );
}

#[test]
fn sphere_plane_contact_generated_without_nan() {
    let Some((states, contacts)) =
        run_sphere_plane_step(Vec3::new(0.0, 0.45, 0.0), Vec3::ZERO, 1.0, 0, None)
    else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    assert!(!contacts.is_empty(), "sphere-plane should generate a contact");
    assert!(
        contacts.iter().all(|c| c.point.is_finite() && c.normal.is_finite()),
        "sphere-plane contact contains non-finite values: {contacts:?}"
    );
    assert!(
        states.iter().all(|s| s.position().is_finite() && s.linear_velocity().is_finite()),
        "narrowphase-only sphere-plane step produced non-finite state: {states:?}"
    );
}

#[test]
fn sphere_on_plane_without_warmstart_stays_finite() {
    let Some(states) = simulate_sphere_plane_frames(180, 12, false) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let pos = states[1].position();
    let vel = states[1].linear_velocity();
    assert!(pos.is_finite(), "sphere without warmstart diverged: {pos}");
    assert!(vel.is_finite(), "sphere without warmstart velocity diverged: {vel}");
}

#[test]
fn sphere_on_plane_remains_supported() {
    let mut world = gpu_world_3d!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 12,
        ..Default::default()
    });

    let _plane = world.add_body(&RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    });
    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 2.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for step in 0..180 {
        world.step();
        let pos = world.get_position(sphere).unwrap();
        let vel = world.get_velocity(sphere).unwrap();
        assert!(pos.is_finite(), "sphere on plane diverged at step {step}: {pos}");
        assert!(
            vel.is_finite(),
            "sphere on plane velocity diverged at step {step}: {vel}"
        );
    }

    let pos = world.get_position(sphere).unwrap();
    assert!(pos.y > -1.0, "sphere fell through plane: y={}", pos.y);
}

#[test]
fn sphere_stack_on_box_remains_supported() {
    let mut world = gpu_world_3d!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        warmstart_decay: 0.95,
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
            position: Vec3::new(0.0, 1.0 + i as f32 * 2.5, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        }));
    }

    for step in 0..180 {
        world.step();
        for (idx, &handle) in handles.iter().enumerate() {
            let pos = world.get_position(handle).unwrap();
            let vel = world.get_velocity(handle).unwrap();
            assert!(
                pos.is_finite(),
                "stacked sphere {idx} diverged at step {step}: {pos}"
            );
            assert!(
                vel.is_finite(),
                "stacked sphere {idx} velocity diverged at step {step}: {vel}"
            );
        }
    }

    for (idx, &handle) in handles.iter().enumerate() {
        let pos = world.get_position(handle).unwrap();
        assert!(
            pos.y > -10.0,
            "stacked sphere {idx} fell through box floor: y={}",
            pos.y
        );
    }
}

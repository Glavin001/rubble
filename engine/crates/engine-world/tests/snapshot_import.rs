//! A scene stored as raw GPU records (`engine_schema::WorldSnapshot`, the
//! browser demo's RUBSNAP1 files) rebuilt through `World::import_snapshot`:
//! the world's records come out as the snapshot's.

use engine_schema::{BodyState, Collider, Joint, Material, ShapeType, BODY_CCD, JOINT_FLAG_BROKEN};
use glam::{Quat, Vec3};
use rubble::*;

fn collider(
    body: u32,
    shape: ShapeType,
    params: [f32; 4],
    material: u32,
    local_pos: [f32; 3],
) -> Collider {
    Collider {
        body,
        shape: shape as u32,
        material,
        generation: 0,
        local_pos,
        density: 2.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params,
    }
}

fn scene() -> engine_schema::WorldSnapshot {
    let mut ground = BodyState::dynamic(Vec3::new(0.0, -0.5, 0.0), Quat::IDENTITY, 1.0, Vec3::ONE);
    ground.inv_mass = 0.0;
    ground.inv_inertia = [0.0; 3];
    ground.flags = 0;
    let q = Quat::from_rotation_y(0.3);
    let mut boxed = BodyState::dynamic(Vec3::new(0.0, 1.0, 0.0), q, 2.0, Vec3::new(0.1, 0.2, 0.3));
    boxed.linear_vel = [0.5, 0.0, -0.25];
    boxed.initial_pos = boxed.position;
    boxed.initial_rot = boxed.orientation;
    let mut ball = BodyState::dynamic(
        Vec3::new(2.0, 1.0, 0.0),
        Quat::IDENTITY,
        1.5,
        Vec3::splat(0.06),
    );
    ball.flags |= BODY_CCD;
    ball.angular_vel = [0.0, 3.0, 0.0];
    ball.initial_pos = ball.position;
    let capsule = BodyState::dynamic(
        Vec3::new(-2.0, 1.0, 0.0),
        Quat::IDENTITY,
        1.0,
        Vec3::splat(0.1),
    );
    let mut mouse = ground;
    mouse.position = [0.0; 3];
    let bodies = vec![ground, boxed, ball, capsule, mouse];
    let colliders = vec![
        collider(
            0,
            ShapeType::Box,
            [20.0, 0.5, 20.0, Vec3::new(20.0, 0.5, 20.0).length()],
            0,
            [0.0; 3],
        ),
        collider(
            1,
            ShapeType::Box,
            [0.5, 0.25, 0.25, Vec3::new(0.5, 0.25, 0.25).length()],
            3,
            [0.0; 3],
        ),
        collider(2, ShapeType::Sphere, [0.4, 0.0, 0.0, 0.4], 3, [0.0; 3]),
        collider(
            3,
            ShapeType::Capsule,
            [0.2, 0.3, 0.0, 0.5],
            0,
            [0.0, 0.1, 0.0],
        ),
    ];
    let mut materials = vec![
        Material {
            friction: 0.5,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0
        };
        64
    ];
    materials[3].friction = 0.7;
    let hinge = JointDesc::hinge(
        1,
        2,
        JointFrame::with_axis(Vec3::new(1.0, 0.0, 0.0), Vec3::Z),
        JointFrame::at(Vec3::new(-1.0, 0.0, 0.0)),
    )
    .with_drive(engine_schema::JointDrive::motor(Dof::Twist, 1.5, 40.0))
    .with_torque_arm(1.0)
    .to_gpu();
    let mut drag = JointDesc::ball(4, 0, Vec3::ZERO, Vec3::ZERO)
        .with_stiffness(5000.0, 0.0)
        .with_torque_arm(1.0)
        .to_gpu();
    drag.flags |= JOINT_FLAG_BROKEN;
    engine_schema::WorldSnapshot {
        bodies,
        colliders,
        materials,
        joints: vec![hinge, drag],
        params: None,
    }
}

fn same_body(a: &BodyState, b: &BodyState) -> bool {
    a.position == b.position
        && a.orientation == b.orientation
        && a.inv_mass == b.inv_mass
        && a.inv_inertia == b.inv_inertia
        && a.linear_vel == b.linear_vel
        && a.angular_vel == b.angular_vel
        && (a.flags & !(BODY_CCD | 0xF)) == (b.flags & !(BODY_CCD | 0xF))
        && (a.flags & BODY_CCD) == (b.flags & BODY_CCD)
        && a.is_dynamic() == b.is_dynamic()
}

#[test]
fn import_gives_back_the_snapshot_records() {
    let snap = scene();
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: WorldCapacity::small(),
            ..WorldDef::default()
        },
    )
    .unwrap();
    let import = w.import_snapshot(&snap).unwrap();
    assert_eq!(import.bodies.len(), 5);
    assert_eq!(import.shapes.len(), 4);
    assert!(
        import.joints[0].is_some() && import.joints[1].is_none(),
        "the broken drag slot is skipped"
    );
    for (i, id) in import.bodies.iter().enumerate() {
        assert_eq!(id.index() as usize, i, "bodies keep their record slots");
        let got = w.sim().read_body(i as u32);
        assert!(
            same_body(&got, &snap.bodies[i]),
            "body {i}: {got:?} vs {:?}",
            snap.bodies[i]
        );
    }
    assert!(w.body_is_bullet(import.bodies[2]).unwrap());
    assert!(w.sim().has_ccd());
    for (k, s) in import.shapes.iter().enumerate() {
        let c = w.shape_collider(*s).unwrap();
        assert_eq!(c as usize, k, "colliders keep their record slots");
        let (got, want) = (w.sim().collider_record(c), snap.colliders[k]);
        let mut got_cmp = got;
        got_cmp.material = want.material;
        assert_eq!(
            bytemuck::bytes_of(&got_cmp),
            bytemuck::bytes_of(&want),
            "collider {k}"
        );
        let friction = w.shape_surface_material(*s).unwrap().friction;
        assert_eq!(
            friction,
            snap.materials[(want.material & 0xFFF) as usize].friction
        );
    }
    let j: Joint = w.sim().read_joint(0);
    assert_eq!(
        bytemuck::bytes_of(&j),
        bytemuck::bytes_of(&snap.joints[0]),
        "hinge record"
    );
    for _ in 0..10 {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
}

#[test]
fn records_a_description_cannot_hold_are_refused() {
    let mut snap = scene();
    snap.colliders[3].local_rot = Quat::from_rotation_x(0.5).to_array();
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let e = w.import_snapshot(&snap).unwrap_err().to_string();
    assert!(e.contains("rotated capsule"), "{e}");
    let mut snap = scene();
    snap.joints[0].lambda[0] = 1.0;
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let e = w.import_snapshot(&snap).unwrap_err().to_string();
    assert!(e.contains("joint record 0"), "{e}");
}

/// A parked (static) projectile marked a bullet launches with continuous
/// collision; a dynamic body's flag is fixed at creation.
#[test]
fn bullet_set_while_parked_applies_at_launch() {
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let p = w.create_body(&BodyDef::default()).unwrap();
    w.create_sphere_shape(p, &ShapeDef::default(), [0.0; 3], 0.3)
        .unwrap();
    w.body_set_bullet(p, true).unwrap();
    assert!(!w.sim().has_ccd(), "inert while parked");
    let t = Transform::new([0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
    w.body_set_type(p, BodyType::Dynamic, t, [0.0, 0.0, 30.0], [0.0; 3])
        .unwrap();
    w.body_set_mass_data(p, MassData::principal(8.0, [0.0; 3], [0.3; 3]))
        .unwrap();
    assert!(w.sim().has_ccd());
    assert_ne!(w.sim().read_body(p.index()).flags & BODY_CCD, 0);
    assert!(w.body_set_bullet(p, false).is_err());
}

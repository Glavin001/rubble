//! The public world API: same physics as a raw `WgpuSim`, generational
//! handles, Box3D-style bodies and shapes, asynchronous frames.

use engine_schema::{BodyState, Collider, ShapeType};
use engine_vehicle::presets;
use glam::{Quat, Vec3};
use rubble::*;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 4096,
        max_contacts: 32_768,
        max_joints: 2048,
        max_candidates: 65_536,
        max_colors: 32,
        max_commands: 1024,
        max_hull_vertices: 1024,
    }
}

fn def(contacts: ContactStorage) -> WorldDef {
    WorldDef {
        capacity: capacity(),
        contacts,
        ..WorldDef::default()
    }
}

fn world() -> World {
    World::new(GpuContext::Default, &def(ContactStorage::Persistent)).unwrap()
}

fn zero_g() -> World {
    let d = WorldDef {
        gravity: [0.0; 3],
        ..def(ContactStorage::Persistent)
    };
    World::new(GpuContext::Default, &d).unwrap()
}

fn ground(w: &mut World) -> BodyId {
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [40.0, 0.5, 40.0])
        .unwrap();
    g
}

fn cube(w: &mut World, p: [f32; 3], user_data: u64) -> BodyId {
    let b = w
        .create_body(&BodyDef {
            user_data,
            ..BodyDef::dynamic(p)
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    b
}

/// Ground plus an `n`×`n`×`n` pile of unit boxes.
fn pile(w: &mut World, n: u32) -> Vec<BodyId> {
    ground(w);
    let mut ids = Vec::new();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let p = [x as f32 * 1.1, 0.5 + y as f32 * 1.02, z as f32 * 1.1];
                let id = cube(w, p, 1000 + ids.len() as u64);
                ids.push(id);
            }
        }
    }
    ids
}

fn run(w: &mut World, steps: u32) -> Vec<Frame> {
    let mut frames = Vec::new();
    for _ in 0..steps {
        w.step().unwrap();
        frames.extend(w.poll().unwrap());
    }
    frames.extend(w.poll_blocking().unwrap());
    frames
}

/// The collider record `World` writes for a default box shape of half `h`.
fn box_collider(body: u32, h: [f32; 3]) -> Collider {
    Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 1.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [h[0], h[1], h[2], Vec3::from(h).length()],
    }
}

fn bytes(bodies: &[BodyState]) -> Vec<u8> {
    bytemuck::cast_slice::<BodyState, u8>(bodies).to_vec()
}

fn unsupported(e: anyhow::Error) -> bool {
    e.downcast_ref::<Unsupported>().is_some()
}

fn hanging_pair(w: &mut World, x: f32, z: f32, break_force: f32, user_data: u64) -> JointId {
    let a = w
        .create_body(&BodyDef {
            position: [x, 10.0, z],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(a, &ShapeDef::default(), [0.25; 3])
        .unwrap();
    let b = w.create_body(&BodyDef::dynamic([x, 9.0, z])).unwrap();
    let heavy = ShapeDef {
        density: 1000.0,
        ..ShapeDef::default()
    };
    w.create_box_shape(b, &heavy, [0.25; 3]).unwrap();
    // 125 kg: 1250 N on the joint.
    let mut base = JointBase::at(a, [0.0, -0.5, 0.0], b, [0.0, 0.5, 0.0]);
    base.break_force = break_force;
    base.user_data = user_data;
    let joint = SphericalJointDef {
        base,
        cone_angle: None,
        twist_limit: None,
    };
    w.create_joint(&JointDef::Spherical(joint)).unwrap()
}

#[test]
fn world_physics_is_the_raw_simulation_bit_for_bit() {
    for contacts in [
        ContactStorage::Canonical,
        ContactStorage::Compact,
        ContactStorage::Persistent,
    ] {
        for vehicle in [false, true] {
            let mut w = World::new(GpuContext::Default, &def(contacts)).unwrap();
            pile(&mut w, 4);
            let initial = w.sim().download_bodies();
            let options = engine_webgpu::SimOptions {
                contacts,
                ..Default::default()
            };
            let mut raw = WgpuSim::new_with_options(capacity(), options).unwrap();
            raw.set_params(w.sim().params());
            raw.add_bodies(&initial);
            raw.add_collider(box_collider(0, [40.0, 0.5, 40.0]));
            for b in 1..initial.len() as u32 {
                raw.add_collider(box_collider(b, [0.5; 3]));
            }
            if vehicle {
                let at = ([12.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]);
                let v = w.add_vehicle(&presets::sedan(), at.0, at.1).unwrap();
                let r = raw.add_vehicle(&presets::sedan(), at.0, at.1).unwrap();
                let input = VehicleInput {
                    forward: 0.6,
                    steer: 0.2,
                    ..VehicleInput::default()
                };
                w.set_vehicle_input(v, input);
                raw.set_vehicle_input(r, input);
            }
            run(&mut w, 300);
            for _ in 0..300 {
                raw.step().unwrap();
            }
            assert!(
                bytes(&w.sim().download_bodies()) == bytes(&raw.download_bodies()),
                "{contacts:?} vehicle={vehicle}: World diverged from WgpuSim"
            );
        }
    }
}

#[test]
fn frames_arrive_in_step_order_with_moves_and_previous_poses() {
    let mut w = world();
    let ids = pile(&mut w, 3);
    let frames = run(&mut w, 120);
    let steps: Vec<u64> = frames.iter().map(|f| f.step).collect();
    assert_eq!(steps, (0..120).collect::<Vec<_>>());
    let first = &frames[0].body_events.moves;
    assert_eq!(
        first.len(),
        ids.len() + 1,
        "every body is reported once at first"
    );
    assert!(first.iter().all(|m| m.previous.is_none()));
    let later = frames[1]
        .body_events
        .moves
        .iter()
        .find(|m| m.body == ids[0])
        .unwrap();
    let first_pose = first.iter().find(|m| m.body == ids[0]).map(|m| m.transform);
    assert_eq!(later.previous, first_pose);
    assert_eq!(later.user_data, 1000);
    let last = frames
        .iter()
        .rev()
        .flat_map(|f| f.body_events.moves.iter())
        .find(|m| m.body == ids[0])
        .unwrap();
    assert_eq!(w.body_transform(ids[0]).unwrap(), Some(last.transform));
}

#[test]
fn every_box_sleeps_once_and_wakes_on_command() {
    let mut w = world();
    let ids = pile(&mut w, 3);
    let frames = run(&mut w, 1500);
    let mut slept: Vec<BodyId> = frames
        .iter()
        .flat_map(|f| f.body_events.moves.iter())
        .filter(|m| m.fell_asleep)
        .map(|m| m.body)
        .collect();
    slept.sort();
    assert_eq!(slept, ids, "one fell_asleep per box");
    assert!(
        !frames.last().unwrap().ran,
        "an asleep world skips its steps"
    );
    assert!(!w.body_is_awake(ids[4]).unwrap());
    w.body_set_awake(ids[4], true).unwrap();
    let f = w.step_sync().unwrap();
    assert!(f
        .body_events
        .moves
        .iter()
        .any(|m| m.body == ids[4] && m.woke));
    assert!(w.body_is_awake(ids[4]).unwrap());
}

#[test]
fn destroyed_bodies_invalidate_handles_and_slots_are_reused_cleanly() {
    let run_once = || {
        let mut w = world();
        let mut live = pile(&mut w, 3);
        let mut stale = Vec::new();
        for k in 0..500u32 {
            let victim = live.swap_remove((k as usize * 7) % live.len());
            w.destroy_body(victim).unwrap();
            stale.push(victim);
            w.step().unwrap();
            let p = [
                ((k % 5) as f32) * 1.2 - 2.0,
                6.0 + (k % 3) as f32,
                ((k % 7) as f32) - 3.0,
            ];
            live.push(cube(&mut w, p, 0));
            w.step().unwrap();
            w.poll().unwrap();
        }
        w.poll_blocking().unwrap();
        (w, live, stale)
    };
    let (mut w, live, stale) = run_once();
    for id in &stale {
        assert!(!w.body_is_valid(*id), "stale handle {id:?} still valid");
        let e = w
            .body_set_linear_velocity(*id, [0.0, 1.0, 0.0])
            .unwrap_err();
        assert!(e.downcast_ref::<StaleHandle>().is_some());
    }
    assert_eq!(w.body_count(), live.len() + 1);
    let bodies = w.sim().download_bodies();
    assert!(bodies
        .iter()
        .all(|b| b.position.iter().all(|v| v.is_finite())));
    for id in &live {
        let p = bodies[id.index() as usize].position;
        assert!(
            p[1] > -1.0 && p[1] < 50.0 && p[0].abs() < 50.0,
            "{id:?} at {p:?}"
        );
    }
    let (w2, ..) = run_once();
    assert!(bytes(&w.sim().download_bodies()) == bytes(&w2.sim().download_bodies()));
    let _ = w.poll_blocking();
}

/// Shapes off the body origin: the solver body sits at the centre of mass,
/// the reported transform stays at the origin.
#[test]
fn transforms_are_reported_at_the_body_origin_not_the_centre_of_mass() {
    let mut w = world();
    let b = w.create_body(&BodyDef::dynamic([0.0, 20.0, 0.0])).unwrap();
    let s = ShapeDef::default();
    w.create_sphere_shape(b, &s, [1.0, 0.0, 0.0], 0.5).unwrap();
    w.create_sphere_shape(b, &s, [3.0, 0.0, 0.0], 0.5).unwrap();
    assert_eq!(w.body_local_center(b).unwrap(), [2.0, 0.0, 0.0]);
    let gpu = w.sim().read_body(b.index());
    assert!(
        (gpu.position[0] - 2.0).abs() < 1e-6,
        "solver body at the centre: {:?}",
        gpu.position
    );
    for _ in 0..20 {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    let t = w.body_transform(b).unwrap().unwrap();
    assert!(
        t.position[0].abs() < 1e-4,
        "the origin stays at x = 0 in free fall: {t:?}"
    );
    assert!(t.position[1] < 20.0);
    let read = w.read_body_state(b).unwrap();
    assert!((Vec3::from(read.transform.position) - Vec3::from(t.position)).length() < 1e-5);
}

/// Adding a shape to a body that has simulated re-centres it on the GPU: the
/// origin does not jump.
#[test]
fn adding_a_shape_to_a_live_body_keeps_its_origin() {
    let mut w = world();
    let b = w.create_body(&BodyDef::dynamic([0.0, 50.0, 0.0])).unwrap();
    w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.5)
        .unwrap();
    for _ in 0..10 {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    let before = w.read_body_state(b).unwrap();
    w.create_sphere_shape(b, &ShapeDef::default(), [2.0, 0.0, 0.0], 0.5)
        .unwrap();
    assert_eq!(w.body_local_center(b).unwrap(), [1.0, 0.0, 0.0]);
    let f = w.step_sync().unwrap();
    let after = f
        .body_events
        .moves
        .iter()
        .find(|m| m.body == b)
        .unwrap()
        .transform;
    assert!(
        after.position[0].abs() < 1e-4,
        "no sideways jump: {after:?}"
    );
    let dy = before.transform.position[1] - after.position[1];
    assert!(dy > 0.0 && dy < 0.2, "one more step of free fall: {dy}");
    let gpu = w.sim().read_body(b.index());
    assert!(
        (gpu.position[0] - 1.0).abs() < 1e-4,
        "the solver body moved to the new centre"
    );
}

#[test]
fn box3d_commands_reach_the_body() {
    let mut w = zero_g();
    let b = w.create_body(&BodyDef::dynamic([0.0, 5.0, 0.0])).unwrap();
    w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.5)
        .unwrap();
    let m = w.body_mass_data(b).unwrap().unwrap().mass;
    w.body_set_angular_velocity(b, [0.0, 1.0, 0.0]).unwrap();
    w.body_set_linear_velocity(b, [2.0, 0.0, 0.0]).unwrap();
    w.step_sync().unwrap();
    let s = w.read_body_state(b).unwrap();
    // The solver re-derives velocity from the step's pose change, so compare
    // within a step's integration error.
    let near = |a: [f32; 3], b: [f32; 3]| (Vec3::from(a) - Vec3::from(b)).length() < 1e-3;
    assert!(
        near(s.linear_velocity, [2.0, 0.0, 0.0]),
        "linear set: {s:?}"
    );
    assert!(
        near(s.angular_velocity, [0.0, 1.0, 0.0]),
        "angular kept by the linear-only set: {s:?}"
    );
    w.body_apply_linear_impulse_to_center(b, [0.0, m, 0.0], true)
        .unwrap();
    w.step_sync().unwrap();
    let s = w.read_body_state(b).unwrap();
    assert!(
        (s.linear_velocity[1] - 1.0).abs() < 1e-3,
        "{:?}",
        s.linear_velocity
    );
    // A force off the centre also turns the body.
    w.body_set_velocity(b, [0.0; 3], [0.0; 3]).unwrap();
    let centre = w.read_body_state(b).unwrap().transform.position;
    let point = [centre[0] + 1.0, centre[1], centre[2]];
    w.body_apply_force(b, [0.0, 0.0, 10.0], point, true)
        .unwrap();
    w.step_sync().unwrap();
    let s = w.read_body_state(b).unwrap();
    assert!(
        s.linear_velocity[2] > 0.0 && s.angular_velocity[1].abs() > 0.0,
        "{s:?}"
    );
    w.body_apply_angular_impulse(b, [0.0, 0.0, 1.0], true)
        .unwrap();
    w.step_sync().unwrap();
    assert!(w.read_body_state(b).unwrap().angular_velocity[2] > 0.0);
    let to = Transform::new([10.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
    w.body_set_transform(b, to).unwrap();
    let f = w.step_sync().unwrap();
    let t = f
        .body_events
        .moves
        .iter()
        .find(|m| m.body == b)
        .unwrap()
        .transform;
    assert!((t.position[0] - 10.0).abs() < 0.2, "{t:?}");
}

#[test]
fn wake_false_leaves_a_sleeping_body_alone() {
    let mut w = world();
    ground(&mut w);
    let b = cube(&mut w, [0.0, 0.5, 0.0], 0);
    run(&mut w, 400);
    assert!(!w.body_is_awake(b).unwrap());
    w.body_apply_linear_impulse_to_center(b, [0.0, 1000.0, 0.0], false)
        .unwrap();
    w.step().unwrap();
    w.poll_blocking().unwrap();
    assert!(
        !w.read_body_state(b).unwrap().awake,
        "a no-wake command must not wake it"
    );
    w.body_apply_linear_impulse_to_center(b, [0.0, 1000.0, 0.0], true)
        .unwrap();
    w.step().unwrap();
    w.poll_blocking().unwrap();
    let s = w.read_body_state(b).unwrap();
    assert!(s.awake && s.linear_velocity[1] > 0.0, "{s:?}");
}

/// A revolute joint turns about frame z (Box3D's convention).
#[test]
fn revolute_joints_turn_about_frame_z() {
    let mut w = zero_g();
    let anchor = w.create_body(&BodyDef::default()).unwrap();
    let wheel = w.create_body(&BodyDef::dynamic([0.0; 3])).unwrap();
    w.create_sphere_shape(wheel, &ShapeDef::default(), [0.0; 3], 0.5)
        .unwrap();
    let hinge = RevoluteJointDef {
        base: JointBase::new(anchor, wheel),
        target_angle: 0.0,
        spring: None,
        limit: None,
        motor: None,
    };
    w.create_joint(&JointDef::Revolute(hinge)).unwrap();
    w.body_set_angular_velocity(wheel, [3.0, 3.0, 3.0]).unwrap();
    for _ in 0..30 {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    let s = w.read_body_state(wheel).unwrap();
    let axis = Quat::from_array(s.transform.rotation) * Vec3::Z;
    assert!(
        (axis - Vec3::Z).length() < 0.05,
        "z is the hinge axis: {axis:?}"
    );
    assert!(
        s.angular_velocity[2].abs() > 1.0 && s.angular_velocity[0].abs() < 0.3,
        "{s:?}"
    );
}

#[test]
fn joint_breaks_arrive_with_their_handles() {
    let mut w = world();
    let joints: Vec<JointId> = (0..8)
        .map(|k| hanging_pair(&mut w, k as f32 * 3.0, 0.0, 100.0, k))
        .collect();
    let frames = run(&mut w, 30);
    let mut broke = Vec::new();
    for e in frames.iter().flat_map(|f| f.destruction_events.iter()) {
        if let DestructionEvent::JointBroke {
            joint,
            force,
            user_data,
            ..
        } = e
        {
            assert!(*force > 100.0);
            assert_eq!(w.joint_user_data(*joint).unwrap(), *user_data);
            broke.push(*joint);
        }
    }
    broke.sort();
    assert_eq!(broke, joints);
    for j in joints {
        w.destroy_joint(j).unwrap();
        assert!(!w.joint_is_valid(j));
        assert!(w.destroy_joint(j).is_err());
    }
}

#[test]
fn more_breaks_than_the_ring_holds_are_reported_lost() {
    let mut w = world();
    for k in 0..1100 {
        hanging_pair(
            &mut w,
            (k % 64) as f32 * 3.0,
            (k / 64) as f32 * 3.0,
            100.0,
            0,
        );
    }
    let frames = run(&mut w, 5);
    let lost: u32 = frames.iter().map(|f| f.lost_events).sum();
    let seen = frames
        .iter()
        .flat_map(|f| f.destruction_events.iter())
        .filter(|e| matches!(e, DestructionEvent::JointBroke { .. }))
        .count() as u32;
    assert!(
        lost > 0,
        "1100 breaks in one step overflow the 1024-record ring"
    );
    assert_eq!(seen + lost, 1100);
}

#[test]
fn shapes_are_first_class() {
    let mut w = world();
    let b = w.create_body(&BodyDef::dynamic([0.0, 3.0, 0.0])).unwrap();
    let wood = SurfaceMaterial {
        friction: 0.8,
        user_material_id: 7,
        ..SurfaceMaterial::default()
    };
    let wooden = ShapeDef {
        material: wood,
        user_data: 11,
        ..ShapeDef::default()
    };
    let a = w.create_box_shape(b, &wooden, [0.5; 3]).unwrap();
    let c = w
        .create_sphere_shape(b, &ShapeDef::default(), [0.0, 1.0, 0.0], 0.3)
        .unwrap();
    assert_eq!(w.body_shapes(b).unwrap(), vec![a, c]);
    assert_eq!(w.shape_body(a).unwrap(), b);
    assert_eq!(w.shape_user_data(a).unwrap(), 11);
    assert_eq!(w.shape_surface_material(a).unwrap().user_material_id, 7);
    let with_both = w.body_mass_data(b).unwrap().unwrap().mass;
    w.destroy_shape(c, true).unwrap();
    assert!(!w.shape_is_valid(c));
    assert!(w.body_mass_data(b).unwrap().unwrap().mass < with_both);
    // The retired collider is reused after the quarantine, under a new id.
    w.step_sync().unwrap();
    w.step_sync().unwrap();
    let d = w
        .create_sphere_shape(b, &ShapeDef::default(), [0.0, -1.0, 0.0], 0.3)
        .unwrap();
    assert_ne!(d, c);
    assert_eq!(
        w.sim().collider_count(),
        2,
        "the retired collider was reused"
    );
    let g = ground(&mut w);
    let ray = RayInput {
        origin: [5.0, 4.0, 5.0],
        translation: [0.0, -10.0, 0.0],
        ignore_body: None,
    };
    let hit = w.cast_ray_blocking(&[ray])[0].unwrap();
    assert_eq!(hit.body, g);
    assert_eq!(
        w.shape_body(hit.shape).unwrap(),
        g,
        "ray hits name the shape"
    );
}

#[test]
fn materials_are_interned_and_capped_loudly() {
    let mut w = world();
    let b = w.create_body(&BodyDef::default()).unwrap();
    for k in 0..63 {
        let material = SurfaceMaterial {
            friction: 0.01 * (k + 1) as f32 + 0.001,
            ..SurfaceMaterial::default()
        };
        let s = ShapeDef {
            material,
            ..ShapeDef::default()
        };
        w.create_box_shape(b, &s, [0.1; 3]).unwrap();
    }
    // An interned material is reused; a 65th distinct one fails.
    w.create_box_shape(b, &ShapeDef::default(), [0.1; 3])
        .unwrap();
    let extra = ShapeDef {
        material: SurfaceMaterial {
            friction: 9.0,
            ..SurfaceMaterial::default()
        },
        ..ShapeDef::default()
    };
    let e = w.create_box_shape(b, &extra, [0.1; 3]).unwrap_err();
    assert!(e.to_string().contains("64 distinct"), "{e}");
}

#[test]
fn unsupported_fields_are_refused_not_ignored() {
    let mut w = world();
    let kinematic = BodyDef {
        body_type: BodyType::Kinematic,
        ..BodyDef::default()
    };
    assert!(unsupported(w.create_body(&kinematic).unwrap_err()));
    let damped = BodyDef {
        linear_damping: 0.1,
        ..BodyDef::dynamic([0.0; 3])
    };
    assert!(unsupported(w.create_body(&damped).unwrap_err()));
    let b = w.create_body(&BodyDef::dynamic([0.0; 3])).unwrap();
    let sensor = ShapeDef {
        is_sensor: true,
        ..ShapeDef::default()
    };
    assert!(unsupported(
        w.create_box_shape(b, &sensor, [0.5; 3]).unwrap_err()
    ));
    let rolling = ShapeDef {
        material: SurfaceMaterial {
            rolling_resistance: 0.1,
            ..SurfaceMaterial::default()
        },
        ..ShapeDef::default()
    };
    assert!(unsupported(
        w.create_box_shape(b, &rolling, [0.5; 3]).unwrap_err()
    ));
    let mut base = JointBase::new(b, b);
    base.force_threshold = 10.0;
    let weld = WeldJointDef {
        base,
        linear_stiffness: None,
        angular_stiffness: None,
    };
    assert!(unsupported(
        w.create_joint(&JointDef::Weld(weld)).unwrap_err()
    ));
    let tilted = WorldDef {
        gravity: [1.0, -9.8, 0.0],
        ..def(ContactStorage::Persistent)
    };
    assert!(World::new(GpuContext::Default, &tilted).is_err());
    let soft = WorldDef {
        soft: Some(SoftCapacity::default()),
        ..def(ContactStorage::Persistent)
    };
    assert!(World::new(GpuContext::Default, &soft).is_err());
}

#[test]
fn dropped_readbacks_are_marked_and_transforms_resynchronise() {
    let d = WorldDef {
        readback_frames: 1,
        backpressure: Backpressure::Drop,
        ..def(ContactStorage::Persistent)
    };
    let mut w = World::new(GpuContext::Default, &d).unwrap();
    let ids = pile(&mut w, 2);
    for _ in 0..10 {
        w.step().unwrap();
    }
    let frames = w.poll_blocking().unwrap();
    assert_eq!(frames.len(), 10);
    assert!(!frames[0].dropped);
    assert!(frames[1..]
        .iter()
        .all(|f| f.dropped && f.body_events.moves.is_empty()));
    let poses = w.read_transforms().unwrap();
    let bodies = w.sim().download_bodies();
    assert_eq!(poses.len(), ids.len() + 1);
    for (id, t) in poses {
        assert_eq!(t.position, bodies[id.index() as usize].position);
    }
}

#[test]
fn batch_creation_matches_one_by_one() {
    let defs: Vec<BodyDef> = (0..50)
        .map(|k| BodyDef::dynamic([k as f32, 2.0, 0.0]))
        .collect();
    let mut a = world();
    let ids = a.create_bodies(&defs).unwrap();
    let mut b = world();
    let ids_b: Vec<BodyId> = defs.iter().map(|d| b.create_body(d).unwrap()).collect();
    assert_eq!(ids, ids_b);
    assert!(bytes(&a.sim().download_bodies()) == bytes(&b.sim().download_bodies()));
}

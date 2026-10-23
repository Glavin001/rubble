//! The public world API: same physics as a raw `WgpuSim`, generational handles,
//! asynchronous frames with moved bodies and events.

use engine_schema::{BodyState, Collider, JointFrame, ShapeType};
use engine_vehicle::presets;
use glam::Vec3;
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

fn config(contacts: ContactStorage) -> WorldConfig {
    WorldConfig {
        capacity: capacity(),
        contacts,
        ..WorldConfig::default()
    }
}

fn boxed(half: f32) -> ShapeDef {
    ShapeDef::new(Shape::Box { half: [half; 3] })
}

fn ground(w: &mut World) -> BodyId {
    w.create_body(
        &BodyDef {
            kind: BodyKind::Static,
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        },
        &[ShapeDef::new(Shape::Box {
            half: [40.0, 0.5, 40.0],
        })],
    )
    .unwrap()
}

/// Ground plus an `n`×`n`×`n` pile of boxes.
fn pile(w: &mut World, n: u32) -> Vec<BodyId> {
    ground(w);
    let mut ids = Vec::new();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let p = [x as f32 * 1.1, 0.5 + y as f32 * 1.02, z as f32 * 1.1];
                ids.push(
                    w.create_body(
                        &BodyDef {
                            position: p,
                            user_data: 1000 + ids.len() as u64,
                            ..BodyDef::default()
                        },
                        &[boxed(0.5)],
                    )
                    .unwrap(),
                );
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
        density: 1000.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [h[0], h[1], h[2], Vec3::from(h).length()],
    }
}

fn bytes(bodies: &[BodyState]) -> Vec<u8> {
    bytemuck::cast_slice::<BodyState, u8>(bodies).to_vec()
}

#[test]
fn world_physics_is_the_raw_simulation_bit_for_bit() {
    for contacts in [
        ContactStorage::Canonical,
        ContactStorage::Compact,
        ContactStorage::Persistent,
    ] {
        for vehicle in [false, true] {
            let mut w = World::new(GpuContext::Default, config(contacts)).unwrap();
            pile(&mut w, 4);
            let initial = w.sim().download_bodies();
            let mut raw = WgpuSim::new_with_options(
                capacity(),
                engine_webgpu::SimOptions {
                    contacts,
                    ..Default::default()
                },
            )
            .unwrap();
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
fn frames_arrive_in_step_order_with_moved_bodies_and_previous_poses() {
    let mut w = World::new(GpuContext::Default, config(ContactStorage::Persistent)).unwrap();
    let ids = pile(&mut w, 3);
    let frames = run(&mut w, 120);
    let steps: Vec<u64> = frames.iter().map(|f| f.step).collect();
    assert_eq!(steps, (0..120).collect::<Vec<_>>());
    let first = &frames[0];
    assert!(first.ran);
    assert_eq!(
        first.moved.len(),
        ids.len() + 1,
        "every body is reported once at first"
    );
    assert!(first.moved.iter().all(|m| m.previous.is_none()));
    let later = frames[1]
        .moved
        .iter()
        .find(|m| m.body == ids[0])
        .expect("falling box moves");
    assert_eq!(
        later.previous,
        first
            .moved
            .iter()
            .find(|m| m.body == ids[0])
            .map(|m| m.pose)
    );
    assert_eq!(later.user_data, 1000);
    // The last reported pose is what `pose` returns.
    assert_eq!(
        w.pose(ids[0]).unwrap(),
        Some(
            frames
                .iter()
                .rev()
                .flat_map(|f| f.moved.iter())
                .find(|m| m.body == ids[0])
                .unwrap()
                .pose
        )
    );
}

#[test]
fn every_box_sleeps_once_and_wakes_on_command() {
    let mut w = World::new(GpuContext::Default, config(ContactStorage::Persistent)).unwrap();
    let ids = pile(&mut w, 3);
    let frames = run(&mut w, 1500);
    let slept: Vec<BodyId> = frames
        .iter()
        .flat_map(|f| f.events.iter())
        .filter_map(|e| match e {
            Event::BodySlept { body, .. } => Some(*body),
            _ => None,
        })
        .collect();
    let mut sorted = slept.clone();
    sorted.sort();
    assert_eq!(sorted, ids, "one BodySlept per box");
    assert!(
        !frames.last().unwrap().ran,
        "an asleep world skips its steps"
    );
    w.wake(ids[4]).unwrap();
    let frames = run(&mut w, 1);
    assert!(frames[0]
        .events
        .iter()
        .any(|e| matches!(e, Event::BodyWoke { body, .. } if *body == ids[4])));
}

#[test]
fn removed_bodies_invalidate_handles_and_slots_are_reused_cleanly() {
    let run_once = || {
        let mut w = World::new(GpuContext::Default, config(ContactStorage::Persistent)).unwrap();
        let ids = pile(&mut w, 3);
        let mut live = ids.clone();
        let mut stale = Vec::new();
        for k in 0..500u32 {
            let victim = live.swap_remove((k as usize * 7) % live.len());
            w.remove_body(victim).unwrap();
            stale.push(victim);
            w.step().unwrap();
            let p = [
                ((k % 5) as f32) * 1.2 - 2.0,
                6.0 + (k % 3) as f32,
                ((k % 7) as f32) - 3.0,
            ];
            live.push(
                w.create_body(
                    &BodyDef {
                        position: p,
                        ..BodyDef::default()
                    },
                    &[boxed(0.5)],
                )
                .unwrap(),
            );
            w.step().unwrap();
            w.poll().unwrap();
        }
        let frames = w.poll_blocking().unwrap();
        (w, live, stale, frames)
    };
    let (mut w, live, stale, _) = run_once();
    for id in &stale {
        assert!(!w.is_valid(*id), "stale handle {id:?} still valid");
        assert!(w.set_velocity(*id, [0.0, 1.0, 0.0], [0.0; 3]).is_err());
    }
    assert!(w.body_count() == live.len() + 1);
    let bodies = w.sim().download_bodies();
    assert!(bodies
        .iter()
        .all(|b| b.position.iter().all(|v| v.is_finite())));
    // Nothing lives in the graveyard but parked slots, and no live box was
    // flung away by a ghost contact with a reused slot's old colliders.
    for id in &live {
        let p = bodies[id.index() as usize].position;
        assert!(
            p[1] > -1.0 && p[1] < 50.0 && p[0].abs() < 50.0,
            "{id:?} at {p:?}"
        );
    }
    // Deterministic: a second run lands bit for bit on the same state.
    let (w2, ..) = run_once();
    assert!(bytes(&w.sim().download_bodies()) == bytes(&w2.sim().download_bodies()));
    let _ = w.poll_blocking();
}

fn breakable_pairs(w: &mut World, n: u32, threshold: f32) -> Vec<JointId> {
    let mut joints = Vec::new();
    for k in 0..n {
        let x = (k % 64) as f32 * 3.0;
        let z = (k / 64) as f32 * 3.0;
        let anchor = w
            .create_body(
                &BodyDef {
                    kind: BodyKind::Static,
                    position: [x, 10.0, z],
                    ..BodyDef::default()
                },
                &[boxed(0.25)],
            )
            .unwrap();
        let bob = w
            .create_body(
                &BodyDef {
                    position: [x, 9.0, z],
                    ..BodyDef::default()
                },
                &[boxed(0.25)],
            )
            .unwrap();
        let desc = JointDesc::ball(0, 0, Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 0.5, 0.0))
            .with_break(threshold, 0.0);
        joints.push(
            w.create_joint(&JointDef {
                body_a: anchor,
                body_b: bob,
                desc,
            })
            .unwrap(),
        );
    }
    let _ = JointFrame::at(Vec3::ZERO);
    joints
}

#[test]
fn joint_breaks_arrive_with_their_handles() {
    let mut w = World::new(GpuContext::Default, config(ContactStorage::Persistent)).unwrap();
    // A 0.25 m box of 1000 kg/m³ weighs 125 kg: 1250 N on a 100 N joint.
    let joints = breakable_pairs(&mut w, 8, 100.0);
    let frames = run(&mut w, 30);
    let mut broke: Vec<JointId> = frames
        .iter()
        .flat_map(|f| f.events.iter())
        .filter_map(|e| match e {
            Event::JointBroke { joint, force, .. } => {
                assert!(*force > 100.0);
                Some(*joint)
            }
            _ => None,
        })
        .collect();
    broke.sort();
    assert_eq!(broke, joints);
    assert!(frames.iter().all(|f| f.lost_events == 0));
    for j in joints {
        w.remove_joint(j).unwrap();
        assert!(!w.is_joint_valid(j));
        assert!(w.remove_joint(j).is_err());
    }
}

#[test]
fn more_breaks_than_the_ring_holds_are_reported_lost() {
    let mut w = World::new(GpuContext::Default, config(ContactStorage::Persistent)).unwrap();
    breakable_pairs(&mut w, 1100, 100.0);
    let frames = run(&mut w, 5);
    let lost: u32 = frames.iter().map(|f| f.lost_events).sum();
    let seen = frames
        .iter()
        .flat_map(|f| f.events.iter())
        .filter(|e| matches!(e, Event::JointBroke { .. }))
        .count() as u32;
    assert!(
        lost > 0,
        "1100 breaks in one step overflow the 1024-record ring"
    );
    assert_eq!(seen + lost, 1100);
}

#[test]
fn commands_reach_the_body_through_the_world() {
    let mut w = World::new(GpuContext::Default, config(ContactStorage::Persistent)).unwrap();
    ground(&mut w);
    let b = w
        .create_body(
            &BodyDef {
                position: [0.0, 5.0, 0.0],
                ..BodyDef::default()
            },
            &[ShapeDef::new(Shape::Sphere { radius: 0.5 })],
        )
        .unwrap();
    w.set_velocity(b, [3.0, 0.0, 0.0], [0.0; 3]).unwrap();
    w.step().unwrap();
    let frames = w.poll_blocking().unwrap();
    let m = frames[0].moved.iter().find(|m| m.body == b).unwrap();
    assert!(m.pose.position[0] > 0.04, "{:?}", m.pose);
    w.teleport(b, [10.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        .unwrap();
    w.step().unwrap();
    let frames = w.poll_blocking().unwrap();
    let m = frames[0].moved.iter().find(|m| m.body == b).unwrap();
    assert!((m.pose.position[0] - 10.0).abs() < 0.2, "{:?}", m.pose);
}

#[test]
fn config_errors_are_explicit() {
    let bad = WorldConfig {
        gravity: [1.0, -9.8, 0.0],
        ..config(ContactStorage::Persistent)
    };
    assert!(World::new(GpuContext::Default, bad).is_err());
    let soft = WorldConfig {
        soft: Some(SoftCapacity::default()),
        ..config(ContactStorage::Persistent)
    };
    assert!(World::new(GpuContext::Default, soft).is_err());
    let mut w = World::new(GpuContext::Default, config(ContactStorage::Persistent)).unwrap();
    let off_centre = ShapeDef {
        local_position: [1.0, 0.0, 0.0],
        ..boxed(0.5)
    };
    assert!(w.create_body(&BodyDef::default(), &[off_centre]).is_err());
}

#[test]
fn dropped_readbacks_are_marked_and_poses_resynchronise() {
    let mut w = World::new(
        GpuContext::Default,
        WorldConfig {
            readback_frames: 1,
            backpressure: Backpressure::Drop,
            ..config(ContactStorage::Persistent)
        },
    )
    .unwrap();
    let ids = pile(&mut w, 2);
    for _ in 0..10 {
        w.step().unwrap();
    }
    let frames = w.poll_blocking().unwrap();
    assert_eq!(frames.len(), 10);
    assert!(!frames[0].dropped && frames[1..].iter().all(|f| f.dropped && f.moved.is_empty()));
    let poses = w.read_poses();
    let bodies = w.sim().download_bodies();
    assert_eq!(poses.len(), ids.len() + 1);
    for (id, pose) in poses {
        assert_eq!(pose.position, bodies[id.index() as usize].position);
    }
}

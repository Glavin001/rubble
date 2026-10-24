//! `pose_pack` (SimOptions::pose_output): compact render poses and the list of
//! bodies that moved, without changing the simulation.

mod support;

use engine_schema::{BodyState, GpuCommand, WorldCapacity};
use engine_webgpu::{ContactStorage, PoseRecord, SimOptions, WgpuSim};
use std::collections::BTreeSet;

const ID: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn pile(contacts: ContactStorage, pose_output: bool) -> WgpuSim {
    let cap = WorldCapacity {
        max_bodies: 512,
        max_colliders: 512,
        max_contacts: 16_384,
        max_joints: 16,
        max_candidates: 32_768,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_options(
        cap,
        SimOptions {
            contacts,
            pose_output,
            ..SimOptions::default()
        },
    )
    .expect("wgpu");
    support::add(
        &mut sim,
        [0.0, -0.5, 0.0],
        [20.0, 0.5, 20.0],
        0.0,
        ID,
        false,
    );
    for x in 0..3 {
        for y in 0..4 {
            for z in 0..3 {
                let p = [x as f32 * 1.1, 0.5 + y as f32 * 1.05, z as f32 * 1.1];
                support::add(&mut sim, p, [0.5; 3], 1.0, ID, false);
            }
        }
    }
    sim
}

fn bytes(sim: &WgpuSim) -> Vec<u8> {
    bytemuck::cast_slice::<BodyState, u8>(&sim.download_bodies()).to_vec()
}

#[test]
fn pose_output_does_not_change_the_simulation() {
    for contacts in [
        ContactStorage::Canonical,
        ContactStorage::Compact,
        ContactStorage::Persistent,
    ] {
        let mut off = pile(contacts, false);
        let mut on = pile(contacts, true);
        for _ in 0..300 {
            off.step().expect("step");
            on.step().expect("step");
        }
        assert!(
            bytes(&off) == bytes(&on),
            "{contacts:?}: pose_pack changed the bodies"
        );
    }
}

#[test]
fn moved_list_is_exactly_the_bodies_whose_pose_or_sleep_changed() {
    let mut sim = pile(ContactStorage::Persistent, true);
    let mut published: Vec<Option<BodyState>> = vec![None; sim.body_count() as usize];
    let mut total = 0;
    for step in 0..400 {
        sim.step().expect("step");
        let bodies = sim.download_bodies();
        let moved = sim.read_moved_poses();
        if !sim.last_step_ran() {
            assert!(moved.is_empty());
            continue;
        }
        let mut expect = BTreeSet::new();
        for (i, b) in bodies.iter().enumerate() {
            let changed = match &published[i] {
                None => true,
                Some(p) => {
                    bytemuck::bytes_of(&p.position) != bytemuck::bytes_of(&b.position)
                        || bytemuck::bytes_of(&p.orientation) != bytemuck::bytes_of(&b.orientation)
                        || (p.flags & 4) != (b.flags & 4)
                }
            };
            if changed {
                expect.insert(i as u32);
            }
        }
        let got: BTreeSet<u32> = moved.iter().map(PoseRecord::body).collect();
        assert_eq!(got, expect, "step {step}");
        for r in &moved {
            let b = &bodies[r.body() as usize];
            assert_eq!(r.position, b.position);
            assert_eq!(r.rotation, b.orientation);
            assert_eq!(r.first(), published[r.body() as usize].is_none());
            published[r.body() as usize] = Some(*b);
        }
        total += moved.len();
    }
    assert!(total > 0);
}

#[test]
fn each_body_reports_one_sleep_and_one_wake() {
    let mut sim = pile(ContactStorage::Persistent, true);
    let n = sim.body_count();
    let mut slept = vec![0u32; n as usize];
    for _ in 0..1500 {
        sim.step().expect("step");
        for r in sim.read_moved_poses() {
            slept[r.body() as usize] += u32::from(r.slept());
            assert!(!r.woke(), "nothing wakes a settling pile");
        }
    }
    assert!(
        !sim.last_step_ran(),
        "the pile should be asleep and skipping steps"
    );
    assert_eq!(slept[0], 0, "the static ground never sleeps");
    assert!(slept[1..].iter().all(|&k| k == 1), "{slept:?}");
    // Wake every box: each reports one WOKE (and moves again).
    let wakes: Vec<GpuCommand> = (1..n).map(GpuCommand::wake).collect();
    sim.push_body_commands(&wakes).expect("commands");
    sim.step().expect("step");
    let woke: BTreeSet<u32> = sim
        .read_moved_poses()
        .iter()
        .filter(|r| r.woke())
        .map(PoseRecord::body)
        .collect();
    assert_eq!(woke, (1..n).collect::<BTreeSet<u32>>());
}

#[test]
fn forgotten_pose_reports_first_again() {
    let mut sim = pile(ContactStorage::Canonical, true);
    sim.step().expect("step");
    assert!(sim.read_moved_poses().iter().all(PoseRecord::first));
    sim.step().expect("step");
    assert!(sim.read_moved_poses().iter().all(|r| !r.first()));
    sim.forget_pose(0);
    sim.step().expect("step");
    let ground = sim.read_moved_poses().into_iter().find(|r| r.body() == 0);
    assert!(ground.is_some_and(|r| r.first()), "{ground:?}");
}

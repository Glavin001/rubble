//! `with_device_options` is the one constructor every mode goes through: a
//! caller's device builds the same world, bit for bit, as the `new_with_*`
//! wrappers that create their own.

mod support;

use engine_schema::{BodyState, WorldCapacity};
use engine_webgpu::{default_device, ContactStorage, SimOptions, WgpuSim};

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 512,
        max_colliders: 512,
        max_contacts: 16_384,
        max_joints: 64,
        max_candidates: 32_768,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

/// A ground slab and a 4×4×4 pile of boxes, stepped 120 times.
fn run(mut sim: WgpuSim) -> Vec<u8> {
    let id = [0.0, 0.0, 0.0, 1.0];
    support::add(
        &mut sim,
        [0.0, -0.5, 0.0],
        [20.0, 0.5, 20.0],
        0.0,
        id,
        false,
    );
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let p = [x as f32 * 1.1, 0.5 + y as f32 * 1.05, z as f32 * 1.1];
                support::add(&mut sim, p, [0.5; 3], 1.0, id, false);
            }
        }
    }
    for _ in 0..120 {
        sim.step().expect("step");
    }
    bytemuck::cast_slice::<BodyState, u8>(&sim.download_bodies()).to_vec()
}

fn shared(contacts: ContactStorage) -> WgpuSim {
    let (info, device, queue) = default_device().expect("wgpu");
    WgpuSim::with_device_options(
        capacity(),
        SimOptions {
            contacts,
            ..SimOptions::from_env()
        },
        info,
        device,
        queue,
    )
    .expect("sim")
}

#[test]
fn caller_device_matches_persistent_wrapper() {
    let own = run(WgpuSim::new_with_persistent_contacts(capacity()).expect("sim"));
    assert!(own == run(shared(ContactStorage::Persistent)));
}

#[test]
fn caller_device_matches_canonical_wrapper() {
    let own = run(WgpuSim::new_with_compact_contacts(capacity(), false).expect("sim"));
    assert!(own == run(shared(ContactStorage::Canonical)));
}

#[test]
fn explicit_defaults_match_the_environment_free_resolution() {
    // No ENGINE_* variable is set under `cargo test`, so the shipped defaults
    // and the environment resolution build the same world.
    let (info, device, queue) = default_device().expect("wgpu");
    let explicit =
        WgpuSim::with_device_options(capacity(), SimOptions::default(), info, device, queue)
            .expect("sim");
    assert!(run(explicit) == run(shared(ContactStorage::Persistent)));
}

#[test]
fn soft_needs_canonical_contacts() {
    let opts = SimOptions {
        contacts: ContactStorage::Persistent,
        soft: Some(Default::default()),
        ..SimOptions::default()
    };
    let err = WgpuSim::new_with_options(capacity(), opts)
        .err()
        .expect("must fail");
    assert!(err.to_string().contains("Canonical"), "{err}");
}

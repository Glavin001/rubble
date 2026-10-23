//! Gate for an asynchronous host: do pipelined steps (`step_submit`, counters
//! forwarded on the GPU, read once at the end) give the bodies `step()` gives?

mod support;

use engine_schema::{BodyState, WorldCapacity};
use engine_webgpu::{ContactStorage, SimOptions, WgpuSim};

fn pile(contacts: ContactStorage) -> WgpuSim {
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
            ..SimOptions::default()
        },
    )
    .expect("wgpu");
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
        for y in 0..5 {
            for z in 0..4 {
                let p = [x as f32 * 1.1, 0.5 + y as f32 * 1.05, z as f32 * 1.1];
                support::add(&mut sim, p, [0.5; 3], 1.0, id, false);
            }
        }
    }
    sim
}

#[test]
fn pipelined_steps_match_synchronous_steps() {
    for contacts in [ContactStorage::Canonical, ContactStorage::Persistent] {
        let mut sync = pile(contacts);
        let mut piped = pile(contacts);
        for _ in 0..200 {
            sync.step().expect("step");
        }
        for k in 0..200 {
            piped.step_submit().expect("submit");
            if k % 3 == 2 {
                piped.drain().expect("drain");
            }
        }
        piped.drain().expect("drain");
        let a = bytemuck::cast_slice::<BodyState, u8>(&sync.download_bodies()).to_vec();
        let b = bytemuck::cast_slice::<BodyState, u8>(&piped.download_bodies()).to_vec();
        assert!(a == b, "{contacts:?}: pipelined steps differ");
    }
}

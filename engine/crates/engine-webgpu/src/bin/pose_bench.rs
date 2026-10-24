//! Cost of `pose_pack` (SimOptions::pose_output) at scale: the same 10k-box
//! pile, sleep off (all awake), stepped with pose output off and on, timed per
//! step (wall clock around a synchronous `step`, which waits for the GPU).
//! Asleep worlds skip their steps entirely, so this is the worst case. Run
//! alone on the GPU.
//!
//! cargo run -p engine-webgpu --release --bin pose_bench [-- bodies steps]

use engine_schema::{BodyState, Collider, ShapeType, WorldCapacity};
use engine_webgpu::{ContactStorage, SimOptions, WgpuSim};
use glam::{Quat, Vec3};
use std::time::Instant;

fn build(n: u32, pose_output: bool) -> WgpuSim {
    let cap = WorldCapacity {
        max_bodies: n + 16,
        max_colliders: n + 16,
        max_contacts: n * 16,
        max_joints: 64,
        max_candidates: n * 32,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_options(
        cap,
        SimOptions {
            contacts: ContactStorage::Persistent,
            pose_output,
            ..SimOptions::default()
        },
    )
    .expect("wgpu");
    // Sleep off: every body stays awake, the worst case for pose_pack (each
    // one is read, and each moving one appended).
    let mut params = engine_schema::AvbdParams::default();
    params.sleep_time = 1.0e6;
    sim.set_params(params);
    let mut ground = BodyState::static_plane_support();
    ground.position = [0.0, -0.5, 0.0];
    ground.initial_pos = ground.position;
    sim.add_body(ground);
    let side = (n as f32).cbrt().ceil() as u32;
    let mut bodies = Vec::new();
    let mut cols = vec![collider(0, [400.0, 0.5, 400.0])];
    'fill: for y in 0..side * 4 {
        for x in 0..side {
            for z in 0..side {
                if bodies.len() as u32 == n {
                    break 'fill;
                }
                let p = Vec3::new(x as f32 * 1.05, 0.5 + y as f32 * 1.02, z as f32 * 1.05);
                let i = Vec3::splat(1.0 / 6.0);
                bodies.push(BodyState::dynamic(p, Quat::IDENTITY, 1.0, i));
                cols.push(collider(bodies.len() as u32, [0.5; 3]));
            }
        }
    }
    sim.add_bodies(&bodies);
    sim.add_colliders(&cols);
    sim
}

fn collider(body: u32, h: [f32; 3]) -> Collider {
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

/// Mean and median step time (ms) over `steps` steps, after `warm` steps.
fn time(sim: &mut WgpuSim, warm: u32, steps: u32) -> (f64, f64, u32) {
    for _ in 0..warm {
        sim.step().expect("step");
    }
    let mut ms = Vec::with_capacity(steps as usize);
    let mut awake = 0;
    for _ in 0..steps {
        let t = Instant::now();
        let c = sim.step().expect("step");
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        awake = c.awake_count;
    }
    ms.sort_by(|a, b| a.total_cmp(b));
    (
        ms.iter().sum::<f64>() / ms.len() as f64,
        ms[ms.len() / 2],
        awake,
    )
}

fn main() {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = args.first().copied().unwrap_or(10_000);
    let steps = args.get(1).copied().unwrap_or(300);
    for (phase, warm) in [("falling", 10), ("resting, awake", 400)] {
        // Interleave off/on twice to average out clocks and thermal drift.
        let mut rows = Vec::new();
        for round in 0..2 {
            for pose in [false, true] {
                let mut sim = build(n, pose);
                let (mean, median, awake) = time(&mut sim, warm, steps);
                rows.push((round, pose, mean, median, awake));
            }
        }
        println!("{n} bodies, {phase}, {steps} steps after {warm}:");
        for (round, pose, mean, median, awake) in rows {
            println!(
                "  round {round} pose_output={pose:<5} mean {mean:7.3} ms  median {median:7.3} ms  awake {awake}"
            );
        }
    }
}

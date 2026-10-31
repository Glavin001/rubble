//! Cost of directed debris (`guide_step.slang`) at scale: `n` boxes in free
//! flight (spaced so none touch), every one guided and every one becoming
//! free in the same step, with `zones` zones. That is the worst burst: the
//! queue holds all of them and drains `plan_budget` per step. Stepped with
//! guidance off and on, timed per step (wall clock around a synchronous
//! `step`, which waits for the GPU). Run alone on the GPU; set
//! `RUBBLE_GPU_TIMING=1` for the per-phase split (the "guide" phase).
//!
//! cargo run -p engine-webgpu --release --bin guide_bench [-- bodies steps zones guided]
//!
//! `guided` (default: all) arms only the first bodies: debris in a big world.
//! `GUIDE_BENCH_PATHS=1` gives every zone a full path (`GUIDE_PATH_KEYS`
//! keys), the most a plan weighs per zone.

use engine_schema::guidance::{
    GuideBodyGpu, GuideParamsGpu, GuideZoneGpu, GuidanceCapacity, GUIDE_PATH_KEYS, ZONE_NO_ANCHOR,
    ZONE_RING,
};
use engine_schema::{BodyState, Collider, ShapeType, WorldCapacity};
use engine_webgpu::{ContactStorage, SimOptions, WgpuSim};
use glam::{Quat, Vec3};
use std::time::Instant;

fn build(n: u32, zones: u32, guided: bool, armed: u32) -> WgpuSim {
    let cap = WorldCapacity {
        max_bodies: n + 16,
        max_colliders: n + 16,
        max_contacts: n * 4,
        max_joints: 64,
        max_candidates: n * 8,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_options(
        cap,
        SimOptions {
            contacts: ContactStorage::Persistent,
            guidance: Some(GuidanceCapacity::default()),
            ..SimOptions::default()
        },
    )
    .expect("wgpu");
    let mut params = engine_schema::AvbdParams::default();
    params.sleep_time = 1.0e6;
    sim.set_params(params);
    let side = (n as f32).cbrt().ceil() as u32;
    let mut bodies = Vec::new();
    let mut cols = Vec::new();
    'fill: for y in 0..side {
        for x in 0..side {
            for z in 0..side {
                if bodies.len() as u32 == n {
                    break 'fill;
                }
                let p = Vec3::new(x as f32 * 2.0, 40.0 + y as f32 * 2.0, z as f32 * 2.0);
                let mut b = BodyState::dynamic(p, Quat::IDENTITY, 1.0, Vec3::splat(1.0 / 6.0));
                b.linear_vel = [6.0, 12.0, 1.0];
                cols.push(collider(bodies.len() as u32, [0.25; 3]));
                bodies.push(b);
            }
        }
    }
    sim.add_bodies(&bodies);
    sim.add_colliders(&cols);
    let extent = side as f32 * 2.0;
    let paths = std::env::var("GUIDE_BENCH_PATHS").is_ok_and(|v| v == "1");
    let zone_list: Vec<GuideZoneGpu> = (0..zones)
        .map(|k| {
            let a = k as f32 * 0.7;
            let center = [extent * 0.5 + 40.0 + 15.0 * a.cos(), 0.0, extent * 0.5 + 15.0 * a.sin()];
            // Zig-zags 2 m either side of the centre, a key per second.
            let mut path = [[0.0; 4]; GUIDE_PATH_KEYS];
            for (j, key) in path.iter_mut().enumerate() {
                let side = if j % 2 == 0 { -2.0 } else { 2.0 };
                *key = [center[0] + side, 0.0, center[2], j as f32];
            }
            GuideZoneGpu {
                center,
                path_count: if paths { GUIDE_PATH_KEYS as u32 } else { 0 },
                path,
                shape: ZONE_RING,
                extent: [1.0, 3.0, 0.0],
                anchor: ZONE_NO_ANCHOR,
                weight: 1.0,
                t_start: f32::MIN,
                t_end: f32::MAX,
                tags: 1,
                ..GuideZoneGpu::default()
            }
        })
        .collect();
    sim.set_guide_zones(&zone_list).expect("zones");
    let g = params.gravity_y.abs();
    sim.set_guidance(
        guided,
        GuideParamsGpu {
            settle_frames: 2,
            replan_frames: 15,
            kick_frac: 0.05,
            kick_window: 0.1,
            accel_max: 0.03 * g,
            max_jerk: 1.0,
            t_min: 0.6,
            t_max: 8.0,
            participation: 1.0,
            kick_frames: 3,
            min_speed: 1.0,
            impact_dv: 0.02,
            ..GuideParamsGpu::default()
        },
    )
    .expect("guidance");
    // The first `armed` bodies are guided (debris); the rest are not.
    let records = vec![GuideBodyGpu::armed(1, u32::MAX); armed.min(n) as usize];
    sim.write_guide_bodies(0, &records).expect("records");
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

/// Mean, median and worst step time (ms) over `steps` steps.
fn time(sim: &mut WgpuSim, steps: u32) -> (f64, f64, f64) {
    let mut ms = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let t = Instant::now();
        sim.step().expect("step");
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    let worst = ms.iter().cloned().fold(0.0, f64::max);
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (mean, ms[ms.len() / 2], worst)
}

fn main() {
    let args: Vec<u32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let n = args.first().copied().unwrap_or(10_000);
    let steps = args.get(1).copied().unwrap_or(120);
    let zones = args.get(2).copied().unwrap_or(256);
    let armed = args.get(3).copied().unwrap_or(n);
    for guided in [false, true] {
        let mut sim = build(n, zones, guided, armed);
        // The first step compiles nothing but uploads the scene.
        sim.step().expect("step");
        let (mean, median, worst) = time(&mut sim, steps);
        let claimed: u32 = sim.read_guide_claims().iter().sum();
        println!(
            "{n} bodies ({armed} guided), {zones} zones, guidance {}: mean {mean:.3} ms, median {median:.3} ms, worst {worst:.3} ms per step ({claimed} planned)",
            if guided { "on " } else { "off" }
        );
    }
}

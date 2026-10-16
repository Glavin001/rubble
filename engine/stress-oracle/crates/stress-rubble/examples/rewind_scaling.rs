//! Rewind scaling: a ram through a row of N free walls, the scene family of
//! `findings/rewind-scaling.md`. Each wall the ram crosses within one frame is one more
//! failure front a rewind-based coupling has to correct; rubble's pipeline routes every
//! wall the ram's swept bounds reach into the frame's impact island before the engine
//! steps, so it never rewinds whatever N.
//!
//! One JSON line per (solver, walls, speed) on stdout.
//!
//! Solvers:
//!   oracle            the stress-ref standalone world (CPU f64, its own rigid bodies): truth
//!   rubble-gpu        rubble's integrated pipeline (AVBD engine and stress-gpu on one
//!                     device, impact islands before the step, no frame rewind)
//!   rubble-gpu-redo   the same with the reference's verify-and-redo switched on
//!
//! The PhysX side (Blast-style correction passes at limits 0..100, and the same coupled
//! frame with PhysX as the engine) is the untracked
//! `PhysX/destruction/stress-oracle/crates/stress-physx/examples/rewind_scaling.rs`; its
//! `wall_row` is this one, restated.
//!
//!   cargo run --release -p stress-rubble --features gpu --example rewind_scaling -- \
//!       --solver rubble-gpu --walls 1,2,4,8 [--speed 40] [--gap 0.1] [--seconds 0.15] \
//!       [--scene wall_row|striker]
//!
//! `REWIND_SCALING_TRACE=1` prints every cluster's and the ram's velocity after each
//! frame (GPU solvers).

use std::time::Instant;

use stress_ref::scene::Scene;

#[path = "../tests/rewind_scenes/mod.rs"]
mod rewind_scenes;
use rewind_scenes::{striker, wall_row};

struct Run {
    frames: Vec<f64>,
    ram: f64,
    fragments: f64,
    broken: f64,
    /// Fragments and broken bonds of each wall.
    per_wall: Vec<(usize, usize)>,
    redone: u64,
    unverified: u64,
    island_frames: f64,
    substeps: u64,
}

fn per_wall(m: &stress_ref::solver::ReferenceSolver) -> Vec<(usize, usize)> {
    (0..m.structures.len())
        .map(|s| {
            let clusters = m.clusters.iter().filter(|c| c.structure == s).count();
            let broken = m.bonds[s].iter().filter(|b| !b.joint.connected(b.rebar.is_some())).count();
            (clusters, broken)
        })
        .collect()
}

fn run_oracle(scene: &Scene) -> Run {
    let mut w = stress_ref::world::World::new(scene);
    let n = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    let mut frames = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        w.step_frame();
        frames.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let o = w.observation();
    Run {
        frames,
        ram: w.impactors[0].velocity.y,
        fragments: o.values["fragments"],
        broken: o.values["broken_bonds"],
        per_wall: per_wall(&w.solver),
        redone: w.redone_frames,
        unverified: 0,
        island_frames: 0.0,
        substeps: w.solver.substeps,
    }
}

#[cfg(feature = "gpu")]
fn run_rubble_gpu(gpu: &stress_gpu::gpu::Gpu, scene: &Scene, redo: bool) -> Run {
    use stress_ref::math::Vec3;
    let engine = stress_rubble::RubbleEngine::with_device(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone()).expect("rubble engine");
    let mut w = stress_gpu::world::GpuWorld::with_engine(gpu, scene, Box::new(engine)).expect("GPU coupled world");
    w.redo_fractured_frames = redo;
    let n = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    let mut frames = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        w.step_frame(gpu).expect("frame");
        frames.push(t.elapsed().as_secs_f64() * 1e3);
        if std::env::var("REWIND_SCALING_TRACE").is_ok() {
            // Diagnostic: every structure-0 cluster's and the ram's velocity after the frame.
            w.solver.sync(gpu);
            let m = &w.solver.mirror;
            let bodies: Vec<String> = m.clusters.iter().map(|c| format!("s{} {:.1}kg vy {:.2}", c.structure, c.mass, c.velocity.y)).collect();
            eprintln!("frame {}: ram vy {:.2}; {}", frames.len(), w.solver.impactors[0].velocity.y, bodies.join(", "));
        }
    }
    w.solver.sync(gpu);
    let o = w.observation();
    Run {
        frames,
        ram: w.solver.impactors[0].velocity.y,
        fragments: o.values["fragments"],
        broken: o.values["broken_bonds"],
        per_wall: per_wall(&w.solver.mirror),
        redone: w.redone_frames(),
        unverified: w.unverified_fractures,
        island_frames: o.values.get("island_frames").copied().unwrap_or(0.0),
        substeps: w.solver.profile.substeps as u64,
    }
}

fn main() {
    let mut solver = "rubble-gpu".to_string();
    let mut walls = vec![1usize, 2, 4, 8];
    let mut speeds = vec![40.0f64];
    let mut gap = 0.1;
    let mut scene_name = "wall_row".to_string();
    let mut seconds = 0.15;
    let mut args = std::env::args().skip(1);
    let list = |v: String| -> Vec<f64> { v.split(',').map(|x| x.parse().expect("number")).collect() };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--solver" => solver = args.next().expect("--solver NAME"),
            "--walls" => walls = list(args.next().expect("--walls N,N")).into_iter().map(|x| x as usize).collect(),
            "--speed" => speeds = list(args.next().expect("--speed V,V")),
            "--scene" => scene_name = args.next().expect("--scene wall_row|striker"),
            "--gap" => gap = args.next().and_then(|v| v.parse().ok()).expect("--gap M"),
            "--seconds" => seconds = args.next().and_then(|v| v.parse().ok()).expect("--seconds S"),
            other => panic!("unknown argument {other}"),
        }
    }
    #[cfg(feature = "gpu")]
    let gpu = (solver != "oracle").then(|| stress_gpu::gpu::Gpu::new().expect("GPU"));
    for &speed in &speeds {
        for &n in &walls {
            let scene = if scene_name == "striker" { striker(speed, seconds) } else { wall_row(n, speed, gap, seconds) };
            let t = Instant::now();
            let run = match solver.as_str() {
                "oracle" => run_oracle(&scene),
                #[cfg(feature = "gpu")]
                "rubble-gpu" => run_rubble_gpu(gpu.as_ref().unwrap(), &scene, false),
                #[cfg(feature = "gpu")]
                "rubble-gpu-redo" => run_rubble_gpu(gpu.as_ref().unwrap(), &scene, true),
                other => panic!("unknown solver {other} (rubble-gpu needs --features gpu)"),
            };
            let total = t.elapsed().as_secs_f64();
            println!(
                "{}",
                serde_json::json!({"solver": solver, "scene": scene_name, "walls": n, "speed": speed, "gap": gap, "seconds": seconds, "frame_dt": scene.sim.frame_dt,
                    "ram_vy": run.ram, "fragments": run.fragments, "broken_bonds": run.broken, "per_wall": run.per_wall,
                    "redone_frames": run.redone, "unverified_fractures": run.unverified, "island_frames": run.island_frames,
                    "substeps": run.substeps, "total_s": total, "frame_ms": run.frames})
            );
        }
    }
}

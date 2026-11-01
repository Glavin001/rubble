//! What the engine's shapes following the chunks' deformation costs at scale
//! (`World::engine_shapes_deformed`): N free concrete beams of `--chunks` chunks each,
//! dropped onto the ground at staggered heights, so every body is active and its chunks
//! deform (landing, then resting under their own weight). Each frame timed end to end, in
//! the host-coupled fast quasi-static frame (`World::shape_poses`, rubble `move_shapes`)
//! and the resident frame (stress-gpu `shapes.rs`), with the shapes undeformed and
//! deformed. One JSON line per (path, shapes) on stdout.
//!
//!   cargo run --release -p stress-rubble --features gpu --example shape_deform_cost -- \
//!       [--bodies 10000] [--chunks 2] [--frames 60] [--path fast,resident] [--anchored]
//!
//! `--anchored`: each beam fixed at its first chunk (a cantilever bending under its own
//! weight, where free beams resting on the ground barely deform).
//!
//! `STRESS_RESIDENT_PHASES=1` prints the resident frame's phases, the shape pass's
//! included (its own wait, so the frame times then overstate).

use std::time::Instant;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_rubble::RubbleEngine;

fn scene(bodies: usize, chunks: usize, anchored: bool) -> Scene {
    let mut s = new_scene("shape_deform_cost", "free beams dropped onto the ground");
    s.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let side = (bodies as f64).sqrt().ceil() as usize;
    let size = 0.1;
    let pitch = size * chunks as f64 + 0.15;
    for i in 0..bodies {
        let (x, y) = ((i % side) as f64 * pitch, (i / side) as f64 * 0.3);
        // Staggered drops: the bodies land over the run's first frames.
        let z = 0.02 + 0.01 * (i % 7) as f64;
        let mut c = grid(Vec3::new(x, y, z), [chunks, 1, 1], Vec3::splat(size), "concrete");
        if anchored {
            c[0].support = Support::Fixed;
        }
        let b = auto_bonds(&c, |_, _| "concrete".into());
        s.bodies.push(body(&format!("beam{i}"), c, b));
    }
    let mut s = stress_rubble::scenes::realtime(s, 1.0);
    for f in ["softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping"] {
        s.sim.features.set(f, false).unwrap();
    }
    s.sim.solve_mode = SolveMode::QuasiStatic;
    s
}

struct Run {
    frames: Vec<f64>,
    resident_frames: u64,
    moved_bodies: usize,
}

fn run(gpu: &Gpu, scene: &Scene, resident: bool, deformed: bool, frames: usize) -> Run {
    std::env::set_var("STRESS_GPU_QUIET_EXCLUDE", "0");
    let engine = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(scene)).expect("rubble");
    let mut w = if resident { GpuWorld::with_resident_engine(gpu, scene, engine) } else { GpuWorld::with_engine(gpu, scene, Box::new(engine)) }.expect("coupled world");
    w.fast_qs = !resident;
    w.set_engine_shapes_deformed(deformed);
    w.redo_fractured_frames = false;
    w.set_impact_islands(false);
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let t = Instant::now();
        w.step_frame(gpu).expect("frame");
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    w.sync(gpu);
    let m = &w.solver.mirror;
    // Bodies whose chunks the solve displaced (the shapes that move).
    let moved_bodies = m.clusters.iter().filter(|cl| cl.chunks.iter().any(|&c| m.chunks[cl.structure][c].u.norm() > 0.0)).count();
    Run { frames: times, resident_frames: w.resident_frames, moved_bodies }
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    (s[s.len() / 2], mean, s[s.len() * 95 / 100])
}

fn main() {
    let mut bodies = 10_000;
    let mut chunks = 2;
    let mut frames = 60;
    let mut paths = vec!["fast".to_string(), "resident".to_string()];
    let mut anchored = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bodies" => bodies = args.next().and_then(|v| v.parse().ok()).expect("--bodies N"),
            "--chunks" => chunks = args.next().and_then(|v| v.parse().ok()).expect("--chunks N"),
            "--frames" => frames = args.next().and_then(|v| v.parse().ok()).expect("--frames N"),
            "--anchored" => anchored = true,
            "--path" => paths = args.next().expect("--path fast,resident").split(',').map(String::from).collect(),
            other => panic!("unknown argument {other}"),
        }
    }
    let gpu = Gpu::new().expect("GPU");
    let s = scene(bodies, chunks, anchored);
    for path in &paths {
        let resident = match path.as_str() {
            "fast" => false,
            "resident" => true,
            other => panic!("unknown path {other} (fast, resident)"),
        };
        for deformed in [false, true] {
            let r = run(&gpu, &s, resident, deformed, frames);
            // The first frames build the engine's bodies and the tables: skipped.
            let warm = &r.frames[frames.min(5)..];
            let (median, mean, p95) = stats(warm);
            println!(
                "{{\"path\":\"{path}\",\"anchored\":{anchored},\"shapes\":\"{}\",\"bodies\":{bodies},\"colliders\":{},\"frames\":{},\"resident_frames\":{},\"moved_bodies\":{},\"median_ms\":{median:.3},\"mean_ms\":{mean:.3},\"p95_ms\":{p95:.3},\"first_ms\":{:.1}}}",
                if deformed { "deformed" } else { "undeformed" },
                bodies * chunks,
                warm.len(),
                r.resident_frames,
                r.moved_bodies,
                r.frames[0]
            );
        }
    }
}

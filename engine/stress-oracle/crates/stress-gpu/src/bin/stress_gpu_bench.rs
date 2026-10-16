//! Substep throughput of the two kernel shapes on growing scenes:
//!   per-substep  two dispatches per substep (all chunks)
//!   island       one dispatch for all substeps, one threadgroup per island
//!
//!   full         the full model (`GpuSolver`: joint law, rigid motion, drift removal)
//!
//!   world        the GPU world frame by frame (`--world MODE`: explicit, adaptive, ...)
//!
//!   versus       the full model against the f64 reference, time per substep
//!
//!   stress-gpu-bench [--substeps N] [--full] [--versus] [--world MODE --frames N] [--pack PATH ...]

use std::time::Instant;

use stress_gpu::gpu::Gpu;
use stress_gpu::scenes;
use stress_gpu::solver::GpuSolver;
use stress_gpu::stress::GpuStress;
use stress_ref::scene::{Scene, SolveMode};
use stress_ref::solver::ReferenceSolver;

fn main() {
    let mut substeps = 2000usize;
    let mut full = false;
    let mut vs = false;
    let mut world_mode: Option<String> = None;
    let mut frames = 60usize;
    let mut packs: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--substeps" => substeps = args.next().and_then(|v| v.parse().ok()).expect("--substeps N"),
            "--full" => full = true,
            "--versus" => vs = true,
            "--world" => world_mode = args.next(),
            "--frames" => frames = args.next().and_then(|v| v.parse().ok()).expect("--frames N"),
            "--pack" => packs.push(args.next().expect("--pack PATH")),
            other => panic!("unknown argument {other}"),
        }
    }
    let gpu = Gpu::new().expect("GPU");
    eprintln!("{} ({:?}), shaders {:?}, {substeps} substeps per run", gpu.adapter_info.name, gpu.adapter_info.backend, gpu.shader_path);
    let mut cases: Vec<(&str, Scene)> = vec![
        ("slab 12x6x2", scenes::slab([12, 6, 2])),
        ("slab 16x8x2", scenes::slab([16, 8, 2])),
        ("slab 24x12x2", scenes::slab([24, 12, 2])),
        ("slab 32x16x2", scenes::slab([32, 16, 2])),
        ("slab 40x20x4", scenes::slab([40, 20, 4])),
        ("slab 100x50x4", scenes::slab([100, 50, 4])),
        ("town 64 x (12x6x2)", scenes::town(64, [12, 6, 2])),
        ("town 512 x (12x6x2)", scenes::town(512, [12, 6, 2])),
    ];
    // `--pack PATH`: scene packs instead of the built-in slabs.
    if !packs.is_empty() {
        cases.clear();
        for p in &packs {
            let path = std::path::Path::new(p);
            let name: &'static str = Box::leak(path.file_stem().unwrap().to_string_lossy().to_string().into_boxed_str());
            cases.push((name, stress_ref::scene_pack::import(path, name, "structure").expect("pack")));
        }
    }
    if full {
        return full_model(&gpu, cases, substeps);
    }
    if vs {
        return versus(&gpu, cases, substeps);
    }
    if let Some(mode) = world_mode {
        return world_frames(&gpu, cases, &mode, frames);
    }
    println!("{:<22} {:>8} {:>8} {:>8}  {:>12} {:>12}  {:>8}", "scene", "islands", "chunks", "bonds", "per-substep", "island", "speedup");
    for (name, mut scene) in cases {
        scene.sim.fracture = false;
        scene.sim.solve_mode = SolveMode::Explicit;
        // The intact kernels have no rebar: timing only, rebar adds only stiffness.
        scene.bodies.iter_mut().flat_map(|b| b.bonds.iter_mut()).for_each(|b| b.rebar = None);
        let reference = ReferenceSolver::new(&scene);
        let mut solver = GpuStress::new(&gpu, &reference).expect("GPU solver");
        let time = |f: &mut dyn FnMut()| {
            f(); // warm-up
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let t = Instant::now();
            f();
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            t.elapsed().as_secs_f64()
        };
        let per = time(&mut || solver.run(&gpu, substeps));
        let isl = time(&mut || solver.run_islands(&gpu, substeps));
        let rate = |secs: f64| substeps as f64 / secs;
        println!(
            "{:<22} {:>8} {:>8} {:>8}  {:>9.0} /s {:>9.0} /s  {:>7.1}x",
            name,
            solver.islands.len(),
            solver.chunks.len(),
            solver.bond_count,
            rate(per),
            rate(isl),
            per / isl
        );
    }
}

/// The full model on intact scenes (fracture on, nothing breaks): substeps per second and
/// the cost per bond-substep.
fn full_model(gpu: &Gpu, cases: Vec<(&str, Scene)>, substeps: usize) {
    println!("{:<22} {:>8} {:>8} {:>8}  {:>12} {:>14}", "scene", "islands", "chunks", "bonds", "substeps/s", "ns/bond-step");
    for (name, mut scene) in cases {
        scene.sim.solve_mode = SolveMode::Explicit;
        let reference = ReferenceSolver::new(&scene);
        let dt = reference.stable_dt();
        let bonds: usize = reference.clusters.iter().map(|c| c.bonds.len()).sum();
        let chunks: usize = reference.clusters.iter().map(|c| c.chunks.len()).sum();
        let islands = reference.clusters.len();
        let mut solver = GpuSolver::new(gpu, reference, None, Vec::new()).expect("GPU solver");
        solver.step(gpu, dt, substeps).expect("warm-up");
        let t = Instant::now();
        solver.step(gpu, dt, substeps).expect("step");
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{:<22} {:>8} {:>8} {:>8}  {:>9.0} /s {:>14.2}",
            name,
            islands,
            chunks,
            bonds,
            substeps as f64 / secs,
            secs / substeps as f64 / bonds as f64 * 1e9
        );
    }
}

/// The GPU world on each scene in a solve mode: setup, and the frame time (first frame
/// and steady state) over `frames` frames, with the clusters still active at the end.
fn world_frames(gpu: &Gpu, cases: Vec<(&str, Scene)>, mode: &str, frames: usize) {
    println!("{:<22} {:>8} {:>8}  {:>9} {:>11} {:>11} {:>8}", "scene", "chunks", "bonds", "setup", "first frame", "per frame", "active");
    for (name, scene) in cases {
        let scene = scene.with_override("sim.solve_mode", &format!("\"{mode}\"")).expect("solve mode");
        let t = Instant::now();
        let mut world = stress_gpu::world::GpuWorld::new(gpu, &scene).expect("GPU world");
        let setup = t.elapsed().as_secs_f64();
        let t = Instant::now();
        world.step_frame(gpu).expect("frame");
        let first = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 1..frames {
            world.step_frame(gpu).expect("frame");
        }
        let per = t.elapsed().as_secs_f64() / (frames - 1).max(1) as f64;
        let m = &world.solver.mirror;
        let chunks: usize = m.clusters.iter().map(|c| c.chunks.len()).sum();
        let bonds: usize = m.clusters.iter().map(|c| c.bonds.len()).sum();
        let active = m.clusters.iter().filter(|c| c.activity == stress_ref::solver::Activity::Active).count();
        println!(
            "{name:<22} {chunks:>8} {bonds:>8}  {:>7.2} s {:>8.2} ms {:>8.2} ms {active:>4}/{}  ({:.0} substeps/frame; host tails {:.1} ms/frame)",
            setup,
            first * 1e3,
            per * 1e3,
            m.clusters.len(),
            m.substeps as f64 / frames as f64,
            world.tail_seconds / frames as f64 * 1e3
        );
        let p = &world.solver.profile;
        let f = frames as f64 / 1e3;
        println!(
            "{:<22} per frame: build {:.1} ms, gpu {:.1} ms, readback {:.1} ms, sync {:.1} ms, split {:.1} ms; {} batches",
            "",
            p.build / f,
            p.gpu / f,
            p.readback / f,
            p.download / f,
            p.split / f,
            p.batches
        );
    }
}

/// The same substeps on the same intact scenes by the f64 reference (`ReferenceSolver`,
/// every CPU thread) and the GPU solver: time per substep and the speed-up. The reference
/// runs fewer substeps on big scenes (its cost is linear in them).
fn versus(gpu: &Gpu, cases: Vec<(&str, Scene)>, substeps: usize) {
    println!(
        "{:<22} {:>8} {:>8}  {:>14} {:>14} {:>9}",
        "scene", "chunks", "bonds", "reference/sub", "gpu/sub", "speed-up"
    );
    for (name, mut scene) in cases {
        scene.sim.solve_mode = SolveMode::Explicit;
        let mut reference = ReferenceSolver::new(&scene);
        let dt = reference.stable_dt();
        let bonds: usize = reference.clusters.iter().map(|c| c.bonds.len()).sum();
        let chunks: usize = reference.clusters.iter().map(|c| c.chunks.len()).sum();
        let mut solver = GpuSolver::new(gpu, reference.clone(), None, Vec::new()).expect("GPU solver");
        solver.step(gpu, dt, substeps).expect("warm-up");
        let t = Instant::now();
        solver.step(gpu, dt, substeps).expect("step");
        let gpu_sub = t.elapsed().as_secs_f64() / substeps as f64;
        // About two seconds of reference work, at least 5 substeps.
        let loads = stress_ref::solver::ChunkLoads::new(&reference);
        reference.substep(dt, &loads);
        let t = Instant::now();
        reference.substep(dt, &loads);
        let probe = t.elapsed().as_secs_f64();
        let n = ((2.0 / probe) as usize).clamp(5, substeps);
        let t = Instant::now();
        for _ in 0..n {
            reference.substep(dt, &loads);
        }
        let ref_sub = t.elapsed().as_secs_f64() / n as f64;
        let us = |s: f64| format!("{:.1} us", s * 1e6);
        println!("{name:<22} {chunks:>8} {bonds:>8}  {:>14} {:>14} {:>8.0}x", us(ref_sub), us(gpu_sub), ref_sub / gpu_sub);
    }
}

//! Run a scene on the GPU world and write its observation (`stress-observation/1`).
//!
//!   stress-gpu-run SCENE.json|catalogue-name [--pack] [--true-step] [--out OBS.json] [--duration S] [--profile]
//!
//! `--pack`: the file is an authored scene pack (`stress_ref::scene_pack`).
//! `--true-step`: run at the true stable step (`stability::with_true_step`, safety 0.9).
//! `--set path=value`: override a scene field (`Scene::with_override`), e.g.
//! `--set sim.solve_mode=adaptive`.

use std::time::Instant;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::scene::Scene;

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("scene file or catalogue name");
    let (mut out, mut duration, mut profile, mut pack, mut true_step) = (None, None, false, false, false);
    let mut overrides: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = args.next(),
            "--duration" => duration = args.next().and_then(|v| v.parse::<f64>().ok()),
            "--profile" => profile = true,
            "--pack" => pack = true,
            "--true-step" => true_step = true,
            "--set" => overrides.push(args.next().expect("--set path=value")),
            other => panic!("unknown argument {other}"),
        }
    }
    let mut scene = if pack {
        let stem = std::path::Path::new(&name).file_stem().unwrap().to_string_lossy().to_string();
        stress_ref::scene_pack::import(std::path::Path::new(&name), &stem, "structure").expect("scene pack")
    } else if name.ends_with(".json") {
        Scene::load(std::path::Path::new(&name)).expect("scene")
    } else {
        stress_ref::builders::catalog()
            .into_iter()
            .chain(stress_ref::showcases::catalog())
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no catalogue scene {name}"))
    };
    for o in &overrides {
        let (path, value) = o.split_once('=').expect("--set path=value");
        scene = scene.with_override(path, value).unwrap_or_else(|e| panic!("--set {o}: {e}"));
    }
    if let Some(d) = duration {
        scene.sim.duration = d;
    }
    if true_step {
        let (s, ratio) = stress_gpu::stability::with_true_step(&scene, 0.9);
        eprintln!("true step: {ratio:.1}x the reference's stress step");
        scene = s;
    }
    let gpu = Gpu::new().expect("GPU");
    let t = Instant::now();
    let mut world = GpuWorld::new(&gpu, &scene).unwrap_or_else(|e| panic!("{}: {e}", scene.name));
    let setup = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let obs = world.run(&gpu).expect("run");
    let run = t.elapsed().as_secs_f64();
    eprintln!(
        "{}: {:.3} s simulated in {run:.2} s (setup {setup:.2} s); {} chunks, {} clusters; broken bonds {}, fragments {}",
        scene.name,
        scene.sim.duration,
        world.solver.chunk_order().len(),
        world.solver.island_count(),
        obs.values.get("broken_bonds").copied().unwrap_or(0.0),
        obs.values.get("fragments").copied().unwrap_or(0.0)
    );
    if profile {
        let p = &world.solver.profile;
        eprintln!(
            "profile: substeps {}, batches {}, splits {} (+{} resumed), re-plans {}, max pairs {}, max impactor candidates {}; build {:.2} s, gpu {:.2} s, readback {:.2} s, split {:.2} s, download {:.2} s",
            p.substeps,
            p.batches,
            world.solver.host_splits,
            world.solver.resumed_halts,
            world.solver.replans,
            p.max_pairs,
            p.max_impactor_candidates,
            p.build,
            p.gpu,
            p.readback,
            p.split,
            p.download
        );
        eprintln!("frame tails (host static solves, rebuilds): {:.2} s", world.tail_seconds);
        eprintln!("static solves delegated to the mirror (tangent_statics): {} in {:.3} s", p.statics_delegated, p.statics_delegated_seconds);
        if p.statics_checked > 0 {
            eprintln!("static solves checked against the oracle (STRESS_GPU_CHECK_STATICS): {}, mismatched {}", p.statics_checked, p.statics_mismatched);
        }
        let m = &world.solver.mirror;
        for (ci, cl) in m.clusters.iter().enumerate().take(12) {
            eprintln!("  cluster {ci}: {} chunks, {} bonds, anchored {}, {:?}", cl.chunks.len(), cl.bonds.len(), cl.anchored, cl.activity);
        }
        for (k, name) in stress_gpu::solver::KERNEL_NAMES.iter().enumerate() {
            if p.kernel_dispatches[k] > 0 {
                eprintln!("  {name:<20} {:8.3} s GPU over {:>7} dispatches ({:6.1} us each)", p.kernels[k], p.kernel_dispatches[k], p.kernels[k] / p.kernel_dispatches[k] as f64 * 1e6);
            }
        }
    }
    if std::env::var("STRESS_GPU_BENCH_KERNELS").is_ok() {
        let dt = 1.0 / 60.0 / 400.0;
        world.solver.step(&gpu, dt, 1).expect("step");
        // Contact candidates for a realistic horizon (one frame), as a frame step plans.
        world.solver.rebuild(&gpu, 1.0 / 60.0).expect("rebuild");
        for (name, secs) in world.solver.bench_kernels(&gpu, dt, 200) {
            eprintln!("  {name:<28} {:8.1} us", secs * 1e6);
        }
    }
    if let Some(path) = out {
        std::fs::write(&path, serde_json::to_string_pretty(&obs).expect("json")).expect("write");
    }
}

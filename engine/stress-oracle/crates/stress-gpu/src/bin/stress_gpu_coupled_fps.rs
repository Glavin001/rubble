//! Frame times of the engine-coupled pattern (the PhysX integration): the engine owns
//! rigid motion and contacts, the stress solver takes one engine frame (1/60 s) of
//! contact impulses and substeps only at its own stress step. The GPU coupled solver
//! against stress-ref's `EngineCoupledSolver`, frame by frame, in two regimes:
//!   idle         a fresh intact structure, no contacts;
//!   destruction  a ram impact on frames 0-2, then a sustained push.
//! at two stress steps: the reference's own, and the true stable step (power iteration,
//! safety 0.9). Contacts are the engine's, so the true step changes the stress step only.
//! Accuracy is each run's broken bonds and largest damage difference against the
//! reference at the reference step (the truth).
//!
//!   stress-gpu-coupled-fps [--frames N] [--ref-cap SECONDS] [--only NAME] [--gpu-only]
//!
//! The reference runs (the truth) are cached (`gate::cached`).
//!
//! `--gpu-only`: skip the stress-ref runs (timing only; accuracy is the tests' job).
//! `STRESS_GPU_FRAME_TIMES=1`: print every frame's time (stderr).
//! `--pace`: start each frame on its 1/60 s tick (a game loop: the GPU idles between
//! frames); without it frames run back to back.

use std::time::Instant;

use stress_gpu::coupled::GpuStressSolverApi;
use stress_gpu::scenes;
use stress_ref::api::{ContactImpulse, EngineCoupledSolver, FrameInput, StressSolverApi};
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, Support};

#[derive(serde::Serialize, serde::Deserialize)]
struct Run {
    times: Vec<f64>,
    setup: f64,
    rows: Vec<[f64; 2]>,
}

/// The ram: the chunk nearest the middle of the structure's face across its thinnest
/// extent, pushed inward along that axis.
fn ram_target(scene: &Scene) -> (usize, Vec3) {
    let chunks = &scene.bodies[0].chunks;
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for c in chunks {
        let p = Vec3::from_array(c.center);
        lo = lo.component_min(p);
        hi = hi.component_max(p);
    }
    let ext = hi - lo;
    let axis = if ext.x <= ext.y && ext.x <= ext.z { 0 } else if ext.y <= ext.z { 1 } else { 2 };
    let mut target = (lo + hi) * 0.5;
    let mut dir = Vec3::ZERO;
    match axis {
        0 => (target.x, dir.x) = (lo.x, 1.0),
        1 => (target.y, dir.y) = (lo.y, 1.0),
        _ => (target.z, dir.z) = (lo.z, 1.0),
    }
    let c = (0..chunks.len()).min_by(|&a, &b| (Vec3::from_array(chunks[a].center) - target).norm().total_cmp(&(Vec3::from_array(chunks[b].center) - target).norm())).unwrap();
    (c, dir)
}

fn contacts(scene: &Scene, frame: usize, destruction: bool) -> Vec<ContactImpulse> {
    if !destruction {
        return Vec::new();
    }
    let (chunk, dir) = ram_target(scene);
    let center = Vec3::from_array(scene.bodies[0].chunks[chunk].center);
    let (impulse, approach) = if frame < 3 { (6000.0 / (frame as f64 + 1.0), 12.0) } else { (400.0, 0.0) };
    vec![ContactImpulse {
        structure: 0,
        chunk,
        point: center - dir * 0.09,
        impulse: dir * impulse,
        other_mass: 800.0,
        approach_speed: approach,
        other_modulus: 30e9,
        other_radius: 0.25,
        crush: None,
        other: Some(1),
    }]
}

/// Paced frames (`--pace`): each starts on its 1/60 s tick, as in a game loop, so the
/// GPU idles between frames (and may lower its clock) unless a frame overruns.
static PACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn run(scene: &Scene, solver: &mut dyn StressSolverApi, setup: f64, frames: usize, destruction: bool, cap: f64) -> Run {
    let mut times = Vec::new();
    let start = Instant::now();
    for f in 0..frames {
        if PACE.load(std::sync::atomic::Ordering::Relaxed) {
            let tick = std::time::Duration::from_secs_f64(f as f64 / 60.0);
            if let Some(wait) = tick.checked_sub(start.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        let input = FrameInput { dt: 1.0 / 60.0, motion: Vec::new(), contacts: contacts(scene, f, destruction) };
        let t = Instant::now();
        solver.step(&input);
        times.push(t.elapsed().as_secs_f64());
        if std::env::var("STRESS_GPU_FRAME_TIMES").is_ok() {
            eprintln!("frame {f}: {:.2} ms", times[f] * 1e3);
        }
        if start.elapsed().as_secs_f64() > cap {
            break;
        }
    }
    let rows = solver.bond_reports(0).iter().map(|r| [r.damage, if r.broken { 1.0 } else { 0.0 }]).collect();
    Run { times, setup, rows }
}

fn report(scene: &str, regime: &str, step: &str, who: &str, r: &Run, truth: Option<&Run>, substeps: usize) {
    let first = r.times[0];
    let mut rest: Vec<f64> = r.times[1..].to_vec();
    rest.sort_by(f64::total_cmp);
    let median = rest.get(rest.len() / 2).copied().unwrap_or(first);
    let worst = rest.last().copied().unwrap_or(first);
    let over = |b: f64| r.times[1..].iter().filter(|&&t| t > b).count();
    let broken = r.rows.iter().filter(|x| x[1] != 0.0).count();
    let accuracy = match truth {
        Some(t) if t.times.len() == r.times.len() => {
            let dmg = r.rows.iter().zip(&t.rows).map(|(a, b)| (a[0] - b[0]).abs()).fold(0.0, f64::max);
            let differ = r.rows.iter().zip(&t.rows).filter(|(a, b)| a[1] != b[1]).count();
            format!("{differ} differ, damage {dmg:.1e}")
        }
        Some(_) => "(reference capped)".into(),
        None => "-".into(),
    };
    println!(
        "| {scene} | {regime} | {step} | {who} | {} | {broken} | {accuracy} | {:.1} | {:.2} | {:.2} | {:.0} | {} | {} | {} | {:.2} s |",
        r.times.len(),
        first * 1e3,
        median * 1e3,
        worst * 1e3,
        1.0 / worst,
        over(1.0 / 120.0),
        over(1.0 / 60.0),
        substeps,
        r.setup
    );
}

fn main() {
    let mut frames = 60usize;
    let mut cap: f64 = 60.0;
    let mut only: Option<String> = None;
    let mut gpu_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => frames = args.next().and_then(|v| v.parse().ok()).expect("--frames N"),
            "--ref-cap" => cap = args.next().and_then(|v| v.parse().ok()).expect("--ref-cap S"),
            "--only" => only = args.next(),
            "--gpu-only" => gpu_only = true,
            "--pace" => PACE.store(true, std::sync::atomic::Ordering::Relaxed),
            other => panic!("unknown argument {other}"),
        }
    }
    let wall = |n: [usize; 3]| {
        let mut s = scenes::slab(n);
        for c in s.bodies[0].chunks.iter_mut() {
            c.support = if c.center[2] < 0.15 { Support::Fixed } else { Support::None };
        }
        s.name = format!("wall {}x{}x{}", n[0], n[1], n[2]);
        s
    };
    let packs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs");
    let pack = |p: &str| stress_ref::scene_pack::import(&packs.join(format!("{p}.json")), p, "structure").expect("pack");
    // villa-savoye is left out: as authored (plain concrete with code-minimum
    // reinforcement) 25 of its joints are over strength at rest, in exact equilibrium
    // (a stair landing on two slender posts, rooftop walls on small bearing patches),
    // so it fails under its own weight; it is not an idle scene.
    let cases: Vec<Scene> = vec![wall([16, 2, 8]), wall([40, 2, 20]), pack("house-1story")];
    println!("Engine-coupled frames (engine owns motion and contacts); {frames} frames of 1/60 s");
    println!("| scene | regime | stress step | solver | frames | broken | vs reference@ref-step | first ms | median ms | worst ms | worst FPS | >8 ms | >16.7 ms | substeps/frame | setup |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for scene in cases {
        if only.as_ref().is_some_and(|o| !scene.name.starts_with(o.as_str())) {
            continue;
        }
        let (true_scene, ratio) = stress_gpu::stability::with_true_step(&scene, 0.9);
        // Sleeping: adaptive mode (intact structures settle; contacts and damage wake them).
        let sleep_scene = true_scene.with_override("sim.solve_mode", "\"adaptive\"").expect("adaptive");
        let sub = |s: &Scene| {
            let m = stress_ref::solver::ReferenceSolver::new(s);
            (1.0 / 60.0 / m.stable_dt().min(1.0 / 60.0)).ceil() as usize
        };
        let (n_ref, n_true) = (sub(&scene), sub(&true_scene));
        eprintln!("{}: true step {ratio:.1}x the reference's ({n_ref} -> {n_true} substeps per frame)", scene.name);
        for (regime, destruction) in [("idle", false), ("destruction", true)] {
            let truth = (!gpu_only).then(|| {
                // The reference at its own step: cached (the contact inputs are this
                // tool's; bump the version when `contacts` changes).
                const INPUTS_VERSION: u32 = 1;
                let key = (&scene, frames, destruction, cap.to_bits(), INPUTS_VERSION);
                let truth: Run = stress_gpu::gate::cached(&format!("coupled-fps-{}-{regime}", scene.name), &key, || {
                    let t = Instant::now();
                    let mut reference = EngineCoupledSolver::new(&scene);
                    run(&scene, &mut reference, t.elapsed().as_secs_f64(), frames, destruction, cap)
                });
                report(&scene.name, regime, "reference", "stress-ref", &truth, None, n_ref);
                truth
            });
            let steps: Vec<(String, &Scene, usize)> = vec![("reference".into(), &scene, n_ref), ("true".into(), &true_scene, n_true), ("true+sleep".into(), &sleep_scene, n_true)];
            for (step, s, n) in steps.iter().map(|(a, b, c)| (a.as_str(), *b, *c)) {
                let t = Instant::now();
                match GpuStressSolverApi::new(s) {
                    Ok(mut gpu) => {
                        let r = run(s, &mut gpu, t.elapsed().as_secs_f64(), frames, destruction, f64::INFINITY);
                        report(&scene.name, regime, step, "GPU", &r, truth.as_ref(), n);
                        let p = &gpu.coupled.profile;
                        let per = |x: f64| x / p.frames.max(1) as f64 * 1e3;
                        eprintln!(
                            "  {} {regime} {step}: {:.0} substeps per frame; per frame sync {:.2} ms, filter {:.2}, refresh {:.2}, step {:.2} (gpu {:.2}, readback {:.2}), output {:.2}",
                            scene.name,
                            gpu.coupled.solver.profile.substeps as f64 / p.frames.max(1) as f64,
                            per(p.sync),
                            per(p.filter),
                            per(p.refresh),
                            per(p.step),
                            per(gpu.coupled.solver.profile.gpu),
                            per(gpu.coupled.solver.profile.readback),
                            per(p.output)
                        );
                    }
                    Err(e) => println!("| {} | {regime} | {step} | GPU | error: {e} |", scene.name),
                }
                if step == "true" && !gpu_only {
                    let t = Instant::now();
                    let mut cpu = EngineCoupledSolver::new(s);
                    let r = run(s, &mut cpu, t.elapsed().as_secs_f64(), frames, destruction, cap);
                    report(&scene.name, regime, step, "stress-ref", &r, truth.as_ref(), n);
                }
            }
        }
    }
}

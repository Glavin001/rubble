//! Measurement only: where one explicit-beams vehicle's GPU step time goes. Runs one
//! vehicle (the plain frame, no vehicle contact) from one saved state with phases of
//! the step skipped (`ExplicitBeamsGpu::set_ablation`, a separately compiled
//! `EB_ABLATE` module whose results are meaningless), and reports per mask the GPU
//! time per step: the slope between frames of 17 steps and frames of 1 step (GPU
//! timestamps around each frame), so the fixed per-frame cost drops out.
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin explicit_beams_ablate -- car.truck \
//!     scenario=impact frames=40 reps=3 max_load=8 max_gpu=10
//! ```
//! Scenarios as `explicit_beams_scaling` (spawned 2 cm above flat ground, tyres
//! rolling, `drag=0`, 27.78 m/s): idle, drive, impact (its own wall 1 + 0.5 v m
//! ahead; the timed frames start 0.5 s in, around the hit). Masks: see
//! `set_ablation`; 2147483648 runs the frozen baseline kernel instead (before/after
//! in one session).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use engine_softbody::truck::parse_truck;
use engine_webgpu::{default_device_with, ExplicitBeamsGpu};
use glam::Vec3;
use std::collections::HashMap;
use std::time::Instant;

fn spawn(text: &str, at: Vec3, vel: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let mut sim = ExplicitBeams::from_truck_config(text, None, at, vel, settings).expect("spawn");
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    sim.roll_tyres(vel);
    sim
}

fn load_average() -> f64 {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    out.split_whitespace().find_map(|t| t.parse().ok()).unwrap_or(f64::NAN)
}

/// Other processes' GPU share in percent over one second (macOS; NaN elsewhere).
fn gpu_busy() -> f64 {
    let sample = || -> Option<HashMap<String, u64>> {
        let xml = std::process::Command::new("ioreg")
            .args(["-r", "-c", "AGXDeviceUserClient", "-w", "0", "-a"])
            .output()
            .ok()?
            .stdout;
        let mut plutil = std::process::Command::new("plutil")
            .args(["-convert", "json", "-o", "-", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .ok()?;
        use std::io::Write;
        plutil.stdin.take()?.write_all(&xml).ok()?;
        let out = plutil.wait_with_output().ok()?;
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
        let me = format!("pid {},", std::process::id());
        let mut t = HashMap::new();
        for c in v.as_array()? {
            let who = c["IOUserClientCreator"].as_str().unwrap_or("?").to_string();
            if who.starts_with(&me) {
                continue;
            }
            let ns: u64 = c["AppUsage"]
                .as_array()
                .map(|u| u.iter().filter_map(|x| x["accumulatedGPUTime"].as_u64()).sum())
                .unwrap_or(0);
            *t.entry(who).or_insert(0) += ns;
        }
        Some(t)
    };
    let Some(a) = sample() else { return f64::NAN };
    let t0 = Instant::now();
    std::thread::sleep(std::time::Duration::from_secs(1));
    let Some(b) = sample() else { return f64::NAN };
    let dt = t0.elapsed().as_secs_f64();
    let used: u64 = b.iter().map(|(k, v)| v.saturating_sub(*a.get(k).unwrap_or(v))).sum();
    100.0 * used as f64 * 1e-9 / dt
}

fn wait_quiet(max_load: f64, max_gpu: f64) -> (f64, f64) {
    let t = Instant::now();
    loop {
        let (l, g) = (load_average(), gpu_busy());
        if !(l >= max_load) && !(g >= max_gpu) {
            return (l, g);
        }
        if t.elapsed().as_secs() > 600 {
            eprintln!("machine not quiet: stopping");
            std::process::exit(2);
        }
        eprintln!("load {l:.2}, GPU busy {g:.0} %: waiting");
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}

/// Mask value: the frozen baseline kernel (the shipped kernel before this work).
const BASELINE: u32 = 1 << 31;

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: explicit_beams_ablate <file.truck> key=value…");
    let kv: HashMap<&str, &str> = args.iter().skip(1).filter_map(|a| a.split_once('=')).collect();
    let get = |k: &str, d: f64| kv.get(k).map_or(d, |v| v.parse().expect("number"));
    let scenario = kv.get("scenario").copied().unwrap_or("drive");
    let frames = get("frames", 40.0) as u32;
    let reps = get("reps", 3.0) as usize;
    let speed = get("speed", 27.78) as f32;
    let (max_load, max_gpu) = (get("max_load", 8.0), get("max_gpu", 10.0));
    let masks: Vec<u32> = kv
        .get("masks")
        .map(|m| m.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![0, 1, 256, 257, 8, 16, 4, 2, 32, 64, 128]);
    let text = std::fs::read_to_string(path)?;
    let file = parse_truck(&text);
    let probe = spawn(&text, Vec3::ZERO, Vec3::ZERO);
    let p = |i: u32| probe.nodes[i as usize].pos;
    let forward = file
        .cameras
        .as_ref()
        .map(|c| {
            let d = p(c.center) - p(c.back);
            Vec3::new(d.x, 0.0, d.z).normalize()
        })
        .unwrap_or(-Vec3::X);
    let at = Vec3::new(100.0, 0.0, 100.0);
    let v = if scenario == "idle" { 0.0 } else { speed };
    let mut sim = spawn(&text, at, forward * v);
    if scenario == "impact" {
        let lateral = Vec3::Y.cross(forward).normalize();
        let (mut lo, mut hi, mut alo, mut ahi) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for n in &probe.nodes {
            lo = lo.min(n.pos.dot(forward));
            hi = hi.max(n.pos.dot(forward));
            alo = alo.min(n.pos.dot(lateral));
            ahi = ahi.max(n.pos.dot(lateral));
        }
        let _ = lo;
        let face = at.dot(forward) + hi + 1.0 + 0.5 * speed;
        let mid = at.dot(lateral) + 0.5 * (alo + ahi);
        let side = 0.5 * (ahi - alo) + 1.0;
        let centre = forward * (face + 0.5) + lateral * mid + Vec3::Y * 3.0;
        let half = if forward.x.abs() > 0.5 {
            Vec3::new(0.5, 3.0, side)
        } else {
            Vec3::new(side, 3.0, 0.5)
        };
        sim.boxes.push(StaticBox::centered(centre, half));
    }
    eprintln!(
        "{} nodes, {} beams, {} triangles, {} contacters, grid cell {}, {} wings",
        sim.nodes.len(),
        sim.beams.len(),
        sim.triangles.len(),
        sim.candidates.len(),
        sim.grid_cell,
        sim.wings.len()
    );
    let (info, device, queue) = default_device_with(true)?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &[sim]);
    gpu.set_wide(false);
    // 0.5 s in (60 frames at 120 fps): the impact's hit lands about 0.04 s later.
    for _ in 0..60 {
        gpu.frame(17);
    }
    gpu.wait();
    let snap = gpu.snapshot();
    let (load, busy) = wait_quiet(max_load, max_gpu);
    eprintln!("{} ({:?}); load {load:.2}, GPU busy {busy:.1} %", info.name, info.backend);
    // Warm the GPU's clocks (the first timed configuration otherwise pays the ramp).
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 2.0 {
        gpu.restore(&snap);
        for _ in 0..frames {
            gpu.frame(17);
        }
        gpu.wait();
    }
    // Per mask and repetition: mean GPU ms per frame at 17 and at 1 step.
    let mut res: Vec<Vec<(f64, f64)>> = vec![Vec::new(); masks.len()];
    for _ in 0..reps {
        for (mi, &m) in masks.iter().enumerate() {
            let mut run = |steps: u32| {
                gpu.restore(&snap);
                if m == BASELINE {
                    gpu.set_ablation(0);
                    gpu.set_kernel_variant(engine_webgpu::KernelVariant::Baseline);
                } else {
                    gpu.set_ablation(m);
                }
                gpu.wait();
                gpu.enable_frame_timing(frames);
                for _ in 0..frames {
                    gpu.frame(steps);
                }
                gpu.wait();
                // The first frame warms caches after the restore.
                mean(&gpu.frame_times_ms()[1..])
            };
            let t17 = run(17);
            let t1 = run(1);
            res[mi].push((t17, t1));
        }
    }
    gpu.set_ablation(0);
    println!("{path} {scenario}: GPU µs per step (slope 1→17 steps), per frame of 17 steps, fixed part");
    println!("| mask | per step µs | frame (17) ms | frame (1) ms | Δ step vs base µs |");
    println!("|---|---|---|---|---|");
    let base = {
        let r = &res[0];
        (mean(&r.iter().map(|x| x.0).collect::<Vec<_>>()) - mean(&r.iter().map(|x| x.1).collect::<Vec<_>>())) / 16.0
    };
    for (mi, &m) in masks.iter().enumerate() {
        let r = &res[mi];
        let t17 = mean(&r.iter().map(|x| x.0).collect::<Vec<_>>());
        let t1 = mean(&r.iter().map(|x| x.1).collect::<Vec<_>>());
        let step = (t17 - t1) / 16.0;
        println!(
            "| {m} | {:.2} | {t17:.4} | {t1:.4} | {:+.2} |",
            1e3 * step,
            1e3 * (step - base)
        );
    }
    Ok(())
}

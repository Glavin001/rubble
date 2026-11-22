//! A saved moment of a World with explicit vehicles replayed on CUDA: the scenario
//! matrix's `--world` gate on this host (`tools/scenario-matrix/run.sh --world --cuda`).
//!
//! `--fixture FILE`: a portable World (`engine_exec::world_portable`, saved by
//! `vehicle_lab ... --save-world FILE` on a wgpu host): the rigid scene, the coupling's
//! carried state, the explicit vehicles' portable state and the drive inputs of the steps
//! after the moment, so no scene, vehicle or World is built here. The whole World step
//! runs on the GPU in one context on one stream (`CudaWorld`): the coupling's emit, the
//! rigid step, the plan, the explicit frame, the loads.
//!
//! The rigid contacts start cold (the file holds the scene, not the solver's caches), so
//! `--warm W` steps (default 5) run first, untimed. Then `--rounds R` (default 3) rounds
//! of `--steps N` World steps (default 20, at most the inputs the file holds), each round
//! from the vehicles' state after the warm-up (the bodies go on, as in `vehicle_lab
//! --end-world`: time only), every step waited for and timed: its GPU time from events on
//! the stream (whole, and by part) and its wall time. Then the same explicit steps as
//! frames of the vehicles' solver alone. Printed: every step's GPU ms; the median, p95
//! and worst step (and which); the split by part; `end world: ...` (the gate's line:
//! World step, frame alone, the World adds; wall medians); a hash of the state after the
//! first round, and whether a second replay from the file ends bit for bit the same
//! (`--no-repeat` skips it).
//!
//! Before the first file's replay the same replay runs once, unreported, to bring the
//! GPU up to speed (`warm_gpu`).
//!
//! `--dir DIR [ROW ...]`: every `DIR/ROW.wfx` (all of them when no row is named), then
//! one table and a verdict: PASS when every row ran, its status words stayed zero and its
//! repeat ended bit for bit (and, with `WORLD_MS` set, the World adds no more than that).
//!
//! `--modules FILE ...`: only print the explicit modules a replay of those files may
//! launch (`ENGINE_CUDA_EXPLICIT_MODULES` for the build; no device needed).

use engine_cuda::{CudaExplicitBeams, CudaWorld, StepTimes};
use engine_exec::explicit_plan::coupling::world as cw;
use engine_exec::world_portable::PortableWorld;
use std::time::Instant;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
}

fn num<T: std::str::FromStr>(name: &str, default: T) -> T {
    arg(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn read(file: &str) -> anyhow::Result<PortableWorld> {
    PortableWorld::from_bytes(&std::fs::read(file).map_err(|e| anyhow::anyhow!("{file}: {e}"))?)
        .map_err(|e| anyhow::anyhow!("{file}: {e}"))
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

/// (explicit state words, bodies) hashed.
fn state_hash(w: &mut CudaWorld) -> anyhow::Result<(u64, u64)> {
    let (mut he, mut hb) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64);
    for (_, words) in w.explicit().state_words()? {
        fnv(&mut he, bytemuck::cast_slice(&words));
    }
    fnv(&mut hb, bytemuck::cast_slice(&w.sim().download_bodies()));
    Ok((he, hb))
}

/// Step `k` after the moment: its recorded drive inputs, then the step, waited for.
fn step(w: &mut CudaWorld, p: &PortableWorld, k: usize) -> anyhow::Result<()> {
    if let Some(inputs) = p.header.inputs.get(k) {
        for (v, words) in inputs.iter().enumerate() {
            if let Some(words) = words {
                w.explicit().set_drive_input_words(v, *words)?;
            }
        }
    }
    w.step(p.header.dt)?;
    w.wait()?;
    Ok(())
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn p95(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * 0.95).round() as usize]
}

/// What one file's replay found.
struct Row {
    name: String,
    note: String,
    step_ms: f64,
    frame_ms: f64,
    gpu_median: f64,
    gpu_p95: f64,
    gpu_worst: f64,
    hash: (u64, u64),
    /// The repeat ended bit for bit the same (`None`: not run).
    repeat: Option<bool>,
}

/// The first replay of a process runs once before anything is timed: a GPU that has just
/// been given to the container runs its first seconds of work slower (a T4's clocks
/// come up under load: the first row's first round read 14 to 25 ms a step against
/// 10.5 in its later rounds, and differently from run to run, with bit-identical
/// results), which would be reported as that row's worst steps.
fn warm_gpu(file: &str) -> anyhow::Result<()> {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return Ok(());
    }
    let p = read(file)?;
    let mut w = CudaWorld::from_portable(&p)?;
    w.explicit().preload()?;
    let steps = num::<usize>("--warm", 5) + num::<usize>("--steps", 20);
    for k in 0..steps.max(p.header.inputs.len()) {
        step(&mut w, &p, k)?;
    }
    Ok(())
}

fn replay(file: &str) -> anyhow::Result<Row> {
    warm_gpu(file)?;
    let p = read(file)?;
    let (warm, rounds): (usize, usize) = (num("--warm", 5), num::<usize>("--rounds", 3).max(1));
    let held = p.header.inputs.len();
    // Past the recorded inputs the last ones stand; with none recorded any count runs.
    let n = num::<usize>("--steps", 20).max(1);
    let n = if held > warm { n.min(held - warm) } else { n };
    let mut w = CudaWorld::from_portable(&p)?;
    let name = std::path::Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file)
        .to_string();
    println!(
        "{name}: {} ({} vehicles, {} bodies, {} colliders, t {:.3}) on {}",
        p.header.note,
        p.header.coupling.layout.vehicles,
        p.header.rigid.body_count,
        p.header.rigid.collider_count,
        p.header.t,
        w.device_name()
    );
    println!("  the rigid contacts start cold: {warm} warm-up steps, not timed; the coupling plans by the scan (saved: by {})", if p.header.coupling.planned_by_query { "the World broadphase" } else { "the scan" });
    // Every module loaded before a step is timed (one is otherwise loaded at its first
    // launch, inside the frame that needs it).
    w.explicit().preload()?;
    for k in 0..warm {
        step(&mut w, &p, k)?;
    }
    let snap = w.explicit().snapshot()?;
    let clock = w.clock();
    let (mut gpu, mut wall): (Vec<StepTimes>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut hash = (0, 0);
    for round in 0..rounds {
        if round > 0 {
            w.explicit().restore(&snap)?;
        }
        w.enable_step_timing(n as u32);
        for k in 0..n {
            let t = Instant::now();
            step(&mut w, &p, warm + k)?;
            wall.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let times = w.step_times_ms()?;
        println!(
            "  round {round}, GPU ms a step: {}",
            times
                .iter()
                .map(|t| format!("{:.3}", t.total))
                .collect::<Vec<_>>()
                .join(" ")
        );
        gpu.extend(times);
        if round == 0 {
            hash = state_hash(&mut w)?;
        }
    }
    let status = w.coupling_status()?;
    anyhow::ensure!(status == 0, "the coupling's status word is {status:#x}");
    // The same explicit steps as frames of the vehicles' solver alone.
    let (mut frame_gpu, mut frame_wall) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        w.explicit().restore(&snap)?;
        let mut c: cw::Clock = clock;
        w.explicit().enable_frame_timing(n as u32);
        for _ in 0..n {
            let steps = c.window_steps(p.header.dt);
            let t = Instant::now();
            w.explicit().frame(steps)?;
            w.explicit().wait()?;
            frame_wall.push(t.elapsed().as_secs_f64() * 1e3);
        }
        frame_gpu.extend(w.explicit().frame_times_ms()?);
    }
    let total: Vec<f64> = gpu.iter().map(|t| t.total).collect();
    let (worst_at, worst) = total
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("steps");
    let part = |f: fn(&StepTimes) -> f64| median(&gpu.iter().map(f).collect::<Vec<_>>());
    println!(
        "  World step, GPU ms ({} steps): median {:.3}, p95 {:.3}, worst {:.3} (round {}, step {})",
        total.len(),
        median(&total),
        p95(&total),
        worst,
        worst_at / n,
        worst_at % n
    );
    println!(
        "  by part (medians, GPU ms): emit {:.3}, rigid step {:.3}, plan {:.3}, explicit frame {:.3}, loads and status {:.3}; frame alone {:.3}; the World adds {:.3}",
        part(|t| t.emit),
        part(|t| t.rigid),
        part(|t| t.plan),
        part(|t| t.frame),
        part(|t| t.loads),
        median(&frame_gpu),
        median(&total) - median(&frame_gpu)
    );
    let (step_ms, frame_ms) = (median(&wall), median(&frame_wall));
    println!(
        "  end world: {} steps, World step {step_ms:.3} ms, frame alone {frame_ms:.3} ms, the World adds {:.3} ms (medians, wall; {rounds} rounds of {n})",
        n * rounds,
        step_ms - frame_ms
    );
    println!(
        "  ring waits {}, coupling launches {}",
        w.ring_waits, w.launches
    );
    drop(w);
    let repeat = if std::env::args().any(|a| a == "--no-repeat") {
        None
    } else {
        let mut again = CudaWorld::from_portable(&p)?;
        for k in 0..warm + n {
            step(&mut again, &p, k)?;
        }
        Some(state_hash(&mut again)? == hash)
    };
    println!(
        "  end state after {} steps: explicit {:016x} bodies {:016x}; a second replay: {}",
        warm + n,
        hash.0,
        hash.1,
        match repeat {
            Some(true) => "bit for bit",
            Some(false) => "DIFFERENT",
            None => "not run",
        }
    );
    Ok(Row {
        name,
        note: p.header.note.clone(),
        step_ms,
        frame_ms,
        gpu_median: median(&total),
        gpu_p95: p95(&total),
        gpu_worst: worst,
        hash,
        repeat,
    })
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--modules") {
        let mut all: Vec<String> = Vec::new();
        for file in &args[2..] {
            all.extend(CudaExplicitBeams::modules_for_replay(
                &read(file)?
                    .explicit()
                    .map_err(|e| anyhow::anyhow!("{file}: {e}"))?
                    .header
                    .layout,
            ));
        }
        all.sort();
        all.dedup();
        println!("{}", all.join(","));
        return Ok(());
    }
    if let Some(file) = arg("--fixture") {
        let row = replay(&file)?;
        anyhow::ensure!(
            row.repeat != Some(false),
            "{}: two replays differ",
            row.name
        );
        return Ok(());
    }
    let dir = arg("--dir").ok_or_else(|| {
        anyhow::anyhow!("--fixture FILE, --dir DIR [ROW ...] or --modules FILE ...")
    })?;
    // The rows named after the options' values, or every file of the directory.
    let at = args.iter().position(|a| a == "--dir").expect("checked") + 2;
    let mut named: Vec<String> = Vec::new();
    let mut k = at;
    while k < args.len() {
        if args[k].starts_with("--") {
            k += if args[k] == "--no-repeat" { 1 } else { 2 };
        } else {
            named.push(args[k].clone());
            k += 1;
        }
    }
    let mut files: Vec<String> = if named.is_empty() {
        std::fs::read_dir(&dir)
            .map_err(|e| anyhow::anyhow!("{dir}: {e}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wfx"))
            .map(|p| p.display().to_string())
            .collect()
    } else {
        named.iter().map(|r| format!("{dir}/{r}.wfx")).collect()
    };
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "{dir}: no portable World (*.wfx): matrix.py --make-world"
    );
    let limit: Option<f64> = std::env::var("WORLD_MS").ok().and_then(|v| v.parse().ok());
    let t_all = Instant::now();
    let mut rows = Vec::new();
    let mut bad = Vec::new();
    for file in &files {
        match replay(file) {
            Ok(r) => rows.push(r),
            Err(e) => {
                println!("{file}: FAILED: {e}");
                bad.push(file.clone());
            }
        }
    }
    println!("scenario matrix, the World's own cost on CUDA: each row's portable World, warmed up, then rounds of World steps and the same explicit steps as frames alone (wall medians; GPU ms from events)");
    println!(
        "{:<8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>8} {:>11} {:<6}  what",
        "row", "step ms", "frame ms", "World ms", "GPU med", "GPU p95", "GPU max", "repeat", ""
    );
    for r in &rows {
        let adds = r.step_ms - r.frame_ms;
        let over = limit.is_some_and(|l| adds > l);
        let differs = r.repeat == Some(false);
        if over || differs {
            bad.push(r.name.clone());
        }
        println!(
            "{:<8} {:>8.3} {:>9.3} {:>+9.3} {:>8.3} {:>8.3} {:>8.3} {:>11} {:<6}  {}",
            r.name,
            r.step_ms,
            r.frame_ms,
            adds,
            r.gpu_median,
            r.gpu_p95,
            r.gpu_worst,
            match r.repeat {
                Some(true) => "bit for bit",
                Some(false) => "DIFFERENT",
                None => "-",
            },
            if over { "OVER" } else { "" },
            r.note
        );
        let _ = r.hash;
    }
    let verdict = if bad.is_empty() {
        "PASS".to_string()
    } else {
        format!("FAIL ({})", bad.join(", "))
    };
    println!(
        "verdict: {verdict}; {} rows in {:.1} s",
        files.len(),
        t_all.elapsed().as_secs_f64()
    );
    std::process::exit(i32::from(!bad.is_empty()));
}

//! Where every microsecond of a step goes on native Metal (engine/docs/profiling-metal.md).
//!
//!   engine/scripts/metal-profile.sh [--scene NAME]... [--warm N] [--steps K] [--reps M]
//!       [--split P] [--no-replay] [--tag TAG] [--out DIR]
//!
//! (the script builds this bin and runs it under the GPU lock and a kill timer). Scenes:
//! `rigid:<scene name>` from `engine_scenes::all_scenes` on `MetalSim` (the shared rigid
//! step plan, the commands the WebGPU host runs), and `cars:<N>` for N procedural sedans
//! on `MetalExplicitBeams` (the explicit node–beam plan; `cars:<N>:wide` for its wide
//! passes). For each scene:
//!   1. warm-up, then K lockstep steps (submit, then wait), each step's host laps
//!      (encode+commit, wait) and GPU interval;
//!   2. P steps split into a command buffer per plan phase (rigid only);
//!   3. kernel replay: the next step's every dispatch alone, M times from the same input
//!      on a copy of the buffers, plus the whole step in one command buffer;
//! and writes `<out>/<scene>.json` (everything), `<scene>.txt` (the tables) and
//! `<scene>.trace.json` (Chrome/Perfetto: host laps, GPU steps, phases, replayed
//! kernels). `RUBBLE_CAPTURE_STEP=N` additionally writes a `.gputrace` of step N.

use engine_metal::profile::{self, host_seconds, percentile, OpTiming, ReplayReport, Trace};
use engine_metal::{MetalExplicitBeams, MetalSim};
use serde_json::json;
use std::fmt::Write as _;
use std::path::PathBuf;

struct Args {
    scenes: Vec<String>,
    warm: u32,
    steps: u32,
    reps: u32,
    split: u32,
    replay: bool,
    out: PathBuf,
}

fn parse() -> anyhow::Result<Args> {
    let mut a = Args {
        scenes: Vec::new(),
        warm: 60,
        steps: 60,
        reps: 7,
        split: 5,
        replay: true,
        out: profile::repo_out().join("metal-profile"),
    };
    let mut tag = None;
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{k} needs a value"))
        };
        match k.as_str() {
            "--scene" => a.scenes.push(v()?),
            "--warm" => a.warm = v()?.parse()?,
            "--steps" => a.steps = v()?.parse()?,
            "--reps" => a.reps = v()?.parse()?,
            "--split" => a.split = v()?.parse()?,
            "--no-replay" => a.replay = false,
            "--tag" => tag = Some(v()?),
            "--out" => a.out = PathBuf::from(v()?),
            "-h" | "--help" => {
                println!("see the header of engine/crates/engine-scenes/src/bin/metal_profile.rs");
                std::process::exit(0);
            }
            _ => anyhow::bail!("unknown argument {k}"),
        }
    }
    if a.scenes.is_empty() {
        a.scenes = vec!["rigid:Brick Ring (9k)".into(), "cars:8".into()];
    }
    if let Some(t) = tag {
        a.out = a.out.join(t);
    }
    Ok(a)
}

fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn stats(v: &[f64]) -> serde_json::Value {
    json!({
        "n": v.len(),
        "median": percentile(v, 0.5),
        "p10": percentile(v, 0.1),
        "p90": percentile(v, 0.9),
        "min": v.iter().copied().fold(f64::INFINITY, f64::min),
        "max": v.iter().copied().fold(0.0, f64::max),
    })
}

/// Lockstep timing: per step (encode+commit host ms, wait host ms, GPU ms, GPU idle ms
/// since the previous step's end).
#[derive(Default)]
struct Lockstep {
    encode_ms: Vec<f64>,
    wait_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    wall_ms: Vec<f64>,
    gap_ms: Vec<f64>,
}

impl Lockstep {
    fn json(&self) -> serde_json::Value {
        json!({
            "encode_commit_ms": stats(&self.encode_ms),
            "host_wait_ms": stats(&self.wait_ms),
            "gpu_ms": stats(&self.gpu_ms),
            "wall_ms": stats(&self.wall_ms),
            "gpu_idle_between_steps_ms": stats(&self.gap_ms),
        })
    }
}

struct Report {
    name: String,
    meta: serde_json::Value,
    lock: Lockstep,
    phases: Vec<(String, Vec<f64>)>,
    replay: Option<ReplayReport>,
    trace: Trace,
}

/// One step laps: run `submit`, then `wait`, record host and GPU spans on the trace.
fn lockstep(
    trace: &mut Trace,
    steps: u32,
    label: &str,
    mut submit: impl FnMut(),
    mut wait: impl FnMut() -> Vec<profile::GpuSpan>,
) -> Lockstep {
    let mut l = Lockstep::default();
    let mut last_end: Option<f64> = None;
    for i in 0..steps {
        let t0 = host_seconds();
        submit();
        let t1 = host_seconds();
        let spans = wait();
        let t2 = host_seconds();
        trace.span(1, "host", "encode+commit", t0, t1, json!({ "step": i }));
        trace.span(1, "host", "wait", t1, t2, json!({ "step": i }));
        l.encode_ms.push((t1 - t0) * 1e3);
        l.wait_ms.push((t2 - t1) * 1e3);
        l.wall_ms.push((t2 - t0) * 1e3);
        let mut gpu = 0.0;
        for s in &spans {
            trace.span(
                2,
                "gpu: steps",
                label,
                s.start_s,
                s.end_s,
                json!({ "step": i }),
            );
            gpu += s.end_s - s.start_s;
            if let Some(prev) = last_end {
                l.gap_ms.push((s.start_s - prev) * 1e3);
            }
            last_end = Some(s.end_s);
        }
        l.gpu_ms.push(gpu * 1e3);
    }
    l
}

fn rigid(name: &str, a: &Args) -> anyhow::Result<Report> {
    let (world, info) = engine_scenes::build_host(name)?;
    anyhow::ensure!(
        world.joints.is_empty(),
        "{name}: MetalSim has no joints (scene has {})",
        world.joints.len()
    );
    let mut sim = MetalSim::new(info.capacity.clone())?;
    sim.set_params(world.params);
    sim.set_materials(&world.materials);
    sim.add_bodies(&world.bodies);
    sim.add_colliders(&world.colliders);
    let chip = sim.chip().to_string();
    let t = host_seconds();
    for _ in 0..a.warm {
        sim.step_submit()?;
    }
    let c = sim.drain()?;
    eprintln!(
        "{name}: {} bodies, warm-up {} steps in {:.0} ms",
        sim.body_count(),
        a.warm,
        (host_seconds() - t) * 1e3
    );
    let mut trace = Trace::new();
    sim.set_gpu_log(true);
    let mut err = None;
    let lock = {
        let sim = std::cell::RefCell::new(&mut sim);
        lockstep(
            &mut trace,
            a.steps,
            "step",
            || {
                if let Err(e) = sim.borrow_mut().step_submit() {
                    err.get_or_insert(e);
                }
            },
            || {
                let mut s = sim.borrow_mut();
                let _ = s.drain();
                s.take_gpu_log()
            },
        )
    };
    if let Some(e) = err {
        return Err(e.into());
    }
    sim.set_gpu_log(false);
    let after = sim.drain()?;
    // Phase-split steps.
    let mut phases: Vec<(String, Vec<f64>)> = Vec::new();
    for _ in 0..a.split {
        let spans = sim.step_profiled_spans()?;
        let mut per: Vec<(&str, f64)> = Vec::new();
        for s in &spans {
            trace.span(
                2,
                "gpu: phases (split step)",
                s.label,
                s.start_s,
                s.end_s,
                json!({}),
            );
            match per.iter_mut().find(|(n, _)| *n == s.label) {
                Some(p) => p.1 += (s.end_s - s.start_s) * 1e3,
                None => per.push((s.label, (s.end_s - s.start_s) * 1e3)),
            }
        }
        for (n, ms) in per {
            match phases.iter_mut().find(|(p, _)| p == n) {
                Some(p) => p.1.push(ms),
                None => phases.push((n.to_string(), vec![ms])),
            }
        }
    }
    let replay = if a.replay {
        let ops = sim.replay_ops()?;
        let (device, queue) = sim.device_queue();
        let t0 = host_seconds();
        let r = profile::replay(device, queue, &sim.replay_buffers(), &ops, a.reps);
        eprintln!(
            "{name}: replayed {} ops x{} in {:.1} s",
            ops.len(),
            a.reps,
            host_seconds() - t0
        );
        Some(r)
    } else {
        None
    };
    let meta = json!({
        "host": "MetalSim (engine_exec::rigid_plan)",
        "scene": name,
        "chip": chip,
        "bodies": sim.body_count(),
        "colliders": world.colliders.len(),
        "iterations": world.params.iterations,
        "awake_after_warmup": c.awake_count,
        "contacts_after_warmup": c.contact_count,
        "awake_after_timing": after.awake_count,
        "contacts_after_timing": after.contact_count,
        "steps_done": sim.steps_done(),
    });
    Ok(Report {
        name: name.to_string(),
        meta,
        lock,
        phases,
        replay,
        trace,
    })
}

fn cars(spec: &str, a: &Args) -> anyhow::Result<Report> {
    use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, FrameClock};
    use engine_softbody::procedural::{car_truck_bare, CarParams};
    use glam::Vec3;
    let mut parts = spec.split(':');
    let n: usize = parts.next().unwrap_or("1").parse()?;
    let wide = parts.next() == Some("wide");
    // explicit_bench's fleet: procedural sedans parked 8 m apart, vehicle contact on.
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings)
        .map_err(anyhow::Error::msg)?;
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let side = (n as f32).sqrt().ceil() as usize;
    let fleet: Vec<ExplicitBeams> = (0..n)
        .map(|i| {
            let at = Vec3::new(8.0 * (i % side) as f32, 0.01 - low, 8.0 * (i / side) as f32);
            ExplicitBeams::from_truck(&text, at, Vec3::ZERO, settings).map_err(anyhow::Error::msg)
        })
        .collect::<anyhow::Result<_>>()?;
    let nodes: usize = fleet.iter().map(|v| v.nodes.len()).sum();
    let beams: usize = fleet.iter().map(|v| v.beams.len()).sum();
    let mut g = MetalExplicitBeams::new(&fleet)?;
    let wide_on = g.set_wide(wide);
    let mut clock = FrameClock::default();
    for _ in 0..a.warm {
        g.frame(clock.next());
    }
    g.wait();
    let mut trace = Trace::new();
    g.set_gpu_log(true);
    let lock = {
        let g = std::cell::RefCell::new(&mut g);
        let clock = std::cell::RefCell::new(&mut clock);
        lockstep(
            &mut trace,
            a.steps,
            "frame",
            || {
                let s = clock.borrow_mut().next();
                g.borrow_mut().frame(s);
            },
            || {
                let mut g = g.borrow_mut();
                g.wait();
                g.take_gpu_log()
            },
        )
    };
    g.set_gpu_log(false);
    let steps = clock.next();
    let replay = if a.replay {
        let ops = g.replay_ops(steps);
        let (device, queue) = g.device_queue();
        let r = profile::replay(device, queue, &g.replay_buffers(), &ops, a.reps);
        Some(r)
    } else {
        None
    };
    let meta = json!({
        "host": "MetalExplicitBeams (engine_exec::explicit_plan)",
        "scene": format!("cars:{spec}"),
        "chip": g.device_name(),
        "vehicles": n,
        "nodes": nodes,
        "beams": beams,
        "wide": wide_on,
        "contact_active": g.contact_active(),
        "steps_per_frame_replayed": steps,
        "unit": "frame (60 fps: 33 or 34 solver steps at 2 kHz)",
    });
    Ok(Report {
        name: format!("cars:{spec}"),
        meta,
        lock,
        phases: Vec::new(),
        replay,
        trace,
    })
}

/// Sum of medians by key.
fn group_by(
    ops: &[OpTiming],
    key: impl Fn(&OpTiming) -> String,
) -> Vec<(String, f64, usize, u64, u64)> {
    let mut out: Vec<(String, f64, usize, u64, u64)> = Vec::new();
    for o in ops {
        let k = key(o);
        match out.iter_mut().find(|e| e.0 == k) {
            Some(e) => {
                e.1 += o.median_ms;
                e.2 += 1;
                e.3 += o.threads;
                e.4 += o.changed_bytes;
            }
            None => out.push((k, o.median_ms, 1, o.threads, o.changed_bytes)),
        }
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    out
}

fn text(r: &Report) -> String {
    let mut s = String::new();
    let m = |v: &[f64]| percentile(v, 0.5);
    let _ = writeln!(s, "== {} ==", r.name);
    let _ = writeln!(s, "{}", serde_json::to_string(&r.meta).unwrap_or_default());
    let l = &r.lock;
    let _ = writeln!(
        s,
        "lockstep ({} steps): GPU {:.3} ms median (p90 {:.3}) | host encode+commit {:.3} ms | host wait {:.3} ms | wall {:.3} ms | GPU idle between steps {:.3} ms",
        l.gpu_ms.len(),
        m(&l.gpu_ms),
        percentile(&l.gpu_ms, 0.9),
        m(&l.encode_ms),
        m(&l.wait_ms),
        m(&l.wall_ms),
        m(&l.gap_ms)
    );
    if !r.phases.is_empty() {
        let total: f64 = r.phases.iter().map(|(_, v)| m(v)).sum();
        let _ = writeln!(
            s,
            "\nphase-split step (a command buffer per phase; {total:.3} ms summed):"
        );
        let _ = writeln!(s, "{:<22} {:>9} {:>7}", "phase", "ms", "share");
        for (n, v) in &r.phases {
            let _ = writeln!(s, "{n:<22} {:>9.3} {:>6.1}%", m(v), 100.0 * m(v) / total);
        }
    }
    if let Some(rp) = &r.replay {
        let step = m(&rp.whole_step_ms);
        let floor = m(&rp.launch_floor_ms);
        let sum: f64 = rp.ops.iter().map(|o| o.median_ms).sum();
        let n = rp.ops.len();
        let _ = writeln!(
            s,
            "\nkernel replay (x{} each): whole step in one command buffer {step:.3} ms | {n} ops alone sum {sum:.3} ms | launch floor {floor:.4} ms/op ({:.3} ms over {n}) | sum - floors {:.3} ms",
            rp.reps,
            floor * n as f64,
            sum - floor * n as f64
        );
        let _ = writeln!(s, "\nby kernel (sum of medians over its dispatches):");
        let _ = writeln!(
            s,
            "{:<38} {:>5} {:>9} {:>7} {:>11} {:>9}",
            "kernel", "n", "ms", "share", "threads", ">=GB/s"
        );
        for (k, ms, cnt, threads, bytes) in group_by(&rp.ops, |o| o.kernel.clone()).iter().take(30)
        {
            let _ = writeln!(
                s,
                "{k:<38} {cnt:>5} {ms:>9.3} {:>6.1}% {threads:>11} {:>9.1}",
                100.0 * ms / sum,
                *bytes as f64 / (ms * 1e-3) / 1e9
            );
        }
        let _ = writeln!(s, "\nby phase:");
        for (k, ms, cnt, _, _) in group_by(&rp.ops, |o| o.phase.clone()) {
            let _ = writeln!(s, "{k:<38} {cnt:>5} {ms:>9.3} {:>6.1}%", 100.0 * ms / sum);
        }
        let mut top: Vec<&OpTiming> = rp.ops.iter().collect();
        top.sort_by(|a, b| b.median_ms.partial_cmp(&a.median_ms).unwrap());
        let _ = writeln!(s, "\ntop dispatches:");
        let _ = writeln!(
            s,
            "{:>4} {:<34} {:<16} {:>10} {:>8} {:>8} {:>6} {:>10} {:>8}",
            "#", "kernel", "phase", "threads", "ms", "p90", "share", "changed", ">=GB/s"
        );
        for o in top.iter().take(40) {
            let _ = writeln!(
                s,
                "{:>4} {:<34} {:<16} {:>10} {:>8.4} {:>8.4} {:>5.1}% {:>10} {:>8.1}",
                o.index,
                o.kernel.chars().take(34).collect::<String>(),
                o.phase.chars().take(16).collect::<String>(),
                o.threads,
                o.median_ms,
                o.p90_ms,
                100.0 * o.median_ms / sum,
                o.changed_bytes,
                o.changed_bytes as f64 / (o.median_ms * 1e-3) / 1e9
            );
        }
    }
    s
}

fn main() -> anyhow::Result<()> {
    let a = parse()?;
    std::fs::create_dir_all(&a.out)?;
    let mut index = Vec::new();
    for scene in &a.scenes {
        let mut r = match scene.split_once(':') {
            Some(("rigid", name)) => rigid(name, &a)?,
            Some(("cars", spec)) => cars(spec, &a)?,
            _ => anyhow::bail!("scene {scene}: rigid:<name> or cars:<N>[:wide]"),
        };
        // The replayed ops on a track of their own, laid end to end after the run.
        if let Some(rp) = &r.replay {
            let mut t = host_seconds();
            for o in &rp.ops {
                let d = o.median_ms * 1e-3;
                r.trace.span(
                    2,
                    "gpu: kernel replay (medians, laid end to end)",
                    &o.kernel,
                    t,
                    t + d,
                    json!({ "phase": o.phase, "entry": o.entry, "threads": o.threads, "p90_ms": o.p90_ms, "changed_bytes": o.changed_bytes }),
                );
                t += d;
            }
        }
        let base = a.out.join(slug(&r.name));
        let doc = json!({
            "meta": r.meta,
            "lockstep": r.lock.json(),
            "phases": r.phases.iter().map(|(n, v)| json!({ "phase": n, "ms": stats(v) })).collect::<Vec<_>>(),
            "replay": r.replay,
        });
        std::fs::write(
            base.with_extension("json"),
            serde_json::to_string_pretty(&doc)?,
        )?;
        let t = text(&r);
        std::fs::write(base.with_extension("txt"), &t)?;
        r.trace.write(&base.with_extension("trace.json"))?;
        print!("{t}");
        println!("wrote {}.{{json,txt,trace.json}}\n", base.display());
        index.push(json!({ "scene": r.name, "files": base.display().to_string() }));
    }
    std::fs::write(
        a.out.join("index.json"),
        serde_json::to_string_pretty(&index)?,
    )?;
    Ok(())
}

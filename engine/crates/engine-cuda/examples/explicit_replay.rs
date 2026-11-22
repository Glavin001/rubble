//! One saved moment replayed on CUDA with a switch on and off: the scenario matrix's
//! row on this host (`tools/scenario-matrix/matrix.py --cuda`).
//!
//! `--fixture FILE`: a portable fixture (`engine_exec::explicit_plan::portable`, saved by
//! `vehicle_lab --from-fixture ROW.fx --save-portable FILE` on any wgpu host): every
//! binding, the layout, the planner and that host's frame paths, so no vehicle, scene or
//! World is built here. `--ab-switch SWITCH --ab-frames N` (default 20): N frames of 34
//! steps from the state with the switch on, the same from the state with it off, every
//! state word compared (the vehicle-contact plan's records outside the verdict, as the
//! lab's), each frame's wall time (launch to finish) reported as vehicle_lab reports it:
//! `AB_WARM_MS` (frames from the state for that long first), `AB_ROUNDS` (the halves in
//! turn, each frame's fastest round), `AB_STEPS` (steps a frame, default 34).
//!
//! Switches (vehicle_lab's `--ab-switch` names): `perf=MASK` (the planner's perf flags
//! with MASK set or cleared), `groupsplit` (the split group path and self split),
//! `selfsplit`, `surffast` (a fleet with surface contact on the fast schedules, its pass
//! as pipelines), `widelist` (the wide passes with their reused pair list), `fused` (its
//! fused aux pass), `wide` (the wide passes); this host's own `devindirect` (indirect
//! launches read their counts on the GPU, off: the host reads them back).
//! `--ab-profile`: each half's frame by kernel, gaps and host work, after the A/B. Exit 0 when the halves end bit for bit.
//!
//! `--run [--save-state FILE]`: this build alone, no switch: `AB_ROUNDS` rounds of N frames
//! from the state (the time: each frame's fastest round, their mean), then the end state
//! (every binding's words by name, the plan's records zeroed) into FILE. matrix.py's
//! two-build A/B (`BASE_BIN`) runs it for each build in turn and compares the files.
//!
//! `--modules FILE ...`: only print the modules a replay of those fixtures may launch.

use engine_cuda::CudaExplicitBeams;
use engine_exec::explicit_plan::{portable::Portable, VV_RECS};
use std::time::Instant;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned())
}

fn env_u(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> anyhow::Result<()> {
    // `--modules FILE ...`: the modules a replay of these fixtures may launch, comma
    // separated (ENGINE_CUDA_EXPLICIT_MODULES for the build; no device needed).
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--modules") {
        let mut all: Vec<String> = Vec::new();
        for file in &args[2..] {
            let p = Portable::from_bytes(&std::fs::read(file)?).map_err(|e| anyhow::anyhow!("{file}: {e}"))?;
            all.extend(CudaExplicitBeams::modules_for_replay(&p.header.layout));
        }
        all.sort();
        all.dedup();
        println!("{}", all.join(","));
        return Ok(());
    }
    let file = arg("--fixture").ok_or_else(|| anyhow::anyhow!("--fixture FILE (a portable fixture)"))?;
    let switch = arg("--ab-switch").unwrap_or_else(|| "perf=0".into());
    let frames = arg("--ab-frames").and_then(|v| v.parse().ok()).unwrap_or(20u32).max(1);
    let p = Portable::from_bytes(&std::fs::read(&file)?).map_err(|e| anyhow::anyhow!("{file}: {e}"))?;
    let mut x = CudaExplicitBeams::from_portable(&p)?;
    println!("{file}: {} ({} vehicles, t {:.3}) on {}", p.header.note, p.header.layout.spans.len(), p.header.t, x.device_name());
    let perf0 = x.planner_mut().perf_flags();
    let apply = |x: &mut CudaExplicitBeams, on: bool| -> anyhow::Result<()> {
        match switch.as_str() {
            s if s.starts_with("perf=") => {
                let mask: u32 = s["perf=".len()..].parse()?;
                x.planner_mut().set_perf_flags(if on { perf0 | mask } else { perf0 & !mask });
            }
            "groupsplit" => {
                x.set_group_split(on);
                x.set_self_split(on);
            }
            "selfsplit" => x.set_self_split(on),
            // Surface contact on the fast schedules (the pass as pipelines).
            "surffast" => x.set_surface_fast(on),
            "widelist" => {
                x.set_wide(true);
                x.set_wide_list(on);
            }
            "fused" => {
                x.set_wide(true);
                x.set_wide_list(true);
                x.set_wide_fused(on);
            }
            "wide" => {
                x.set_wide(on);
            }
            // The host's own: indirect launches' counts read on the GPU, or read back.
            "devindirect" => x.set_device_indirect(on),
            other => anyhow::bail!("switch {other} is not on this host (perf=MASK, groupsplit, selfsplit, surffast, widelist, fused, wide, devindirect)"),
        }
        Ok(())
    };
    let steps = env_u("AB_STEPS", 34);
    let snap = x.snapshot()?;
    // (AB_WARM_MS: frames from the state first, so a GPU still at its idle clock does not
    // slow the first half alone.)
    let warm = f64::from(env_u("AB_WARM_MS", 0));
    if warm > 0.0 {
        let t = Instant::now();
        while t.elapsed().as_secs_f64() * 1e3 < warm {
            x.restore(&snap)?;
            for _ in 0..4 {
                x.frame(steps)?;
            }
            x.wait()?;
        }
    }
    let rounds = env_u("AB_ROUNDS", 1).max(1);
    // The vehicle-contact plan's records are planning state (which vehicle overflowed a
    // bucket first differs between two runs of one path): outside the verdict, as the lab's.
    let at = x.planner_mut().words[43] as usize * 4;
    let plan = at..at + p.header.layout.spans.len() * VV_RECS * 4;
    let masked = |w: &[(&'static str, Vec<u32>)]| -> Vec<Vec<u32>> {
        w.iter()
            .map(|(name, v)| {
                let mut v = v.clone();
                if *name == "misc" && plan.end <= v.len() {
                    v[plan.clone()].fill(0);
                }
                v
            })
            .collect()
    };
    if args.iter().any(|a| a == "--run") {
        // One build, no switch (matrix.py's two-build A/B runs this binary and another,
        // process by process): the frames from the state ROUNDS times, then the end state.
        let mut runs: Vec<Vec<f64>> = Vec::new();
        for _ in 0..rounds {
            x.restore(&snap)?;
            x.frame(steps)?;
            x.wait()?;
            x.restore(&snap)?;
            let mut each = Vec::with_capacity(frames as usize);
            for _ in 0..frames {
                let t = Instant::now();
                x.frame(steps)?;
                x.wait()?;
                each.push(t.elapsed().as_secs_f64() * 1e3);
            }
            runs.push(each);
        }
        let best = (0..runs[0].len()).map(|k| runs.iter().map(|r| r[k]).fold(f64::MAX, f64::min)).sum::<f64>() / runs[0].len() as f64;
        let words = x.state_words()?;
        if let Some(path) = arg("--save-state") {
            // name, word count (u32 LE), the words; the plan's records zeroed.
            let mut out = Vec::new();
            for ((name, _), w) in words.iter().zip(masked(&words)) {
                out.extend_from_slice(name.as_bytes());
                out.push(0);
                out.extend_from_slice(&(w.len() as u32).to_le_bytes());
                for v in w {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            std::fs::write(&path, out)?;
        }
        println!("   run at t {:.1}: {best:.3} ms/frame (each frame's best of {rounds} rounds, mean)", p.header.t);
        // The peak: each frame's best round (frame k is the same work in every round),
        // the worst of them and where it is. A run is judged by its worst frame.
        let each: Vec<f64> = (0..runs[0].len()).map(|k| runs.iter().map(|r| r[k]).fold(f64::MAX, f64::min)).collect();
        let (at, worst) = each.iter().copied().enumerate().fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        println!(
            "   run peak: worst {worst:.3} ms at frame {at} of {}; frames: {}",
            each.len(),
            each.iter().map(|m| format!("{m:.2}")).collect::<Vec<_>>().join(" ")
        );
        return Ok(());
    }
    let mut pooled: [Vec<Vec<f64>>; 2] = [Vec::new(), Vec::new()];
    let mut words: [Vec<(&'static str, Vec<u32>)>; 2] = [Vec::new(), Vec::new()];
    for _ in 0..rounds {
        for (side, on) in [(0usize, true), (1, false)] {
            x.restore(&snap)?;
            apply(&mut x, on)?;
            // (One untimed frame first: a module this path loads on first use loads there.
            // Then the state again.)
            x.frame(steps)?;
            x.wait()?;
            x.restore(&snap)?;
            let mut each = Vec::with_capacity(frames as usize);
            for _ in 0..frames {
                let t = Instant::now();
                x.frame(steps)?;
                x.wait()?;
                each.push(t.elapsed().as_secs_f64() * 1e3);
            }
            pooled[side].push(each);
            words[side] = x.state_words()?;
        }
    }
    for (side, name) in [(0, "on"), (1, "off")] {
        let runs = &pooled[side];
        // Frame k is the same work in every round: its fastest round is its time without
        // the noise, their mean the half's time.
        let best = (0..runs[0].len()).map(|k| runs.iter().map(|r| r[k]).fold(f64::MAX, f64::min)).sum::<f64>() / runs[0].len() as f64;
        println!("     frames ({name}): best of {rounds} rounds a frame, mean {best:.3} ms");
        let mut each: Vec<f64> = runs.concat();
        each.sort_by(f64::total_cmp);
        println!(
            "     frames ({name}): median {:.3} ms, min {:.3}, max {:.3} ({} frames)",
            each[each.len() / 2],
            each[0],
            each[each.len() - 1],
            each.len()
        );
    }
    let same = masked(&words[0]) == masked(&words[1]);
    for ((name, a), (_, b)) in words[0].iter().zip(&words[1]) {
        let diff = a.iter().zip(b).filter(|(p, q)| p != q).count();
        if diff > 0 {
            let at: Vec<String> = a.iter().zip(b).enumerate().filter(|(_, (p, q))| p != q).take(4).map(|(i, (p, q))| format!("[{i}] {p:#x} / {q:#x}")).collect();
            println!("     {name}: {diff} of {} words differ: {}", a.len(), at.join(", "));
        }
    }
    let ms = |side: usize| pooled[side].concat().iter().sum::<f64>() / pooled[side].concat().len() as f64;
    println!(
        "   A/B at t {:.1}: {switch} on {:.2} ms/frame, off {:.2} ms/frame; state bit for bit: {same}",
        p.header.t,
        ms(0),
        ms(1)
    );
    println!("   frames on a split path: {}; argument reads (waits): {}", x.split_frames, x.args_syncs);
    if args.iter().any(|a| a == "--ab-profile") {
        profile(&mut x, &snap, steps, &p, "on", |x| apply(x, true))?;
        profile(&mut x, &snap, steps, &p, "off", |x| apply(x, false))?;
    }
    std::process::exit(i32::from(!same));
}

/// `--ab-profile` (matrix.py PROFILE_SPLIT=1): where the frame's time goes with the switch
/// on (`side`), then off, from the state. 10 frames: wall (launch to finish) and GPU
/// (events around the frame) medians, and the host's work a frame (launches, bank writes,
/// argument reads); then 10 more with an event pair around every launch: each kernel's GPU
/// time a frame. GPU minus the kernels is the stream's gaps between launches; wall minus
/// GPU the host.
fn profile(
    x: &mut CudaExplicitBeams,
    snap: &engine_cuda::CudaSnapshot,
    steps: u32,
    p: &Portable,
    side: &str,
    on: impl Fn(&mut CudaExplicitBeams) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    const N: usize = 10;
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    x.restore(snap)?;
    on(x)?;
    x.frame(steps)?;
    x.wait()?;
    x.restore(snap)?;
    let (l0, b0, s0) = (x.launches, x.bank_writes, x.args_syncs);
    x.enable_frame_timing(N as u32);
    let mut wall = Vec::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        x.frame(steps)?;
        x.wait()?;
        wall.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let gpu = med(x.frame_times_ms()?);
    let wall = med(wall);
    let per = |a: u64, b: u64| (a - b) as f64 / N as f64;
    let (launches, banks, syncs) = (per(x.launches, l0), per(x.bank_writes, b0), per(x.args_syncs, s0));
    x.restore(snap)?;
    x.set_phase_timing(true);
    for _ in 0..N {
        x.frame(steps)?;
    }
    let phases = x.phase_times_ms()?;
    x.set_phase_timing(false);
    let kernels: f64 = phases.iter().map(|k| k.1).sum::<f64>() / N as f64;
    println!(
        "   profile at t {:.1} ({side}): wall {wall:.3} ms, GPU {gpu:.3} ms a frame (medians of {N}); kernels {kernels:.3} ms, \
         gaps {:.3} ms, host {:.3} ms; a frame: {launches:.0} launches, {banks:.0} bank writes, {syncs:.1} argument reads",
        p.header.t,
        gpu - kernels,
        wall - gpu
    );
    for (k, ms, n) in &phases {
        let ms = ms / N as f64;
        println!(
            "     {k:<44} {ms:>8.3} ms {:>6.1}x {:>6.1}%",
            f64::from(*n) / N as f64,
            100.0 * ms / gpu.max(1e-9)
        );
    }
    Ok(())
}

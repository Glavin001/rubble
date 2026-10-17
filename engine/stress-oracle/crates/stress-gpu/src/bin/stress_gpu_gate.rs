//! The accuracy gate over every case (`gate::cases`) at full duration, against the
//! cached reference runs (`gate::reference_set`), and the micro and fuzz tiers
//! (`micro`).
//!
//!   stress-gpu-gate --build [--only A,B]   compute the missing reference sets (CPU only)
//!   stress-gpu-gate [--only A,B] [--quick] run the GPU world on each cached case and gate it
//!   stress-gpu-gate --build --micro        compute the micro cases' references
//!   stress-gpu-gate --micro [--only A,B]   micro tier: catalogue, knob runs (metamorphic),
//!                                          snapshots, fuzz regressions; coverage matrix
//!   stress-gpu-gate --build --fuzz N [--seed S]   compute N fuzz scenes' references
//!   stress-gpu-gate --fuzz N [--seed S]    fuzz tier: N seeded scenes (gate, batch, wide)
//!   stress-gpu-gate --accept-snapshots     quick gate in-process; if it passes, overwrite
//!                                          the micro tier's changed GPU snapshots
//!
//! `--quick`: the quick tier (`gate::QUICK`) for iterating; the full set before a commit.
//! `--explain` (with `--micro`): print each case's events, coverage and differences.
//! `--refresh` (or `STRESS_GATE_REFRESH=1`): recompute cached reference sets (with
//! `--build`) instead of reusing them. The cache is keyed by each case's scene and
//! stress-ref's source fingerprint, so a change to either recomputes on its own.
//! Exits non-zero when a case fails or has no cached reference.

use std::time::Instant;

use stress_gpu::coverage;
use stress_gpu::gate::{allowed_error, cases, difference, gate, reference_set, set_refresh, Case, QUICK};
use stress_gpu::gpu::Gpu;
use stress_gpu::micro::{self, Report, SnapshotState};
use stress_gpu::world::{unsupported, GpuWorld};

fn main() {
    let mut build = false;
    let mut quick = false;
    let mut micro_tier = false;
    let mut explain = false;
    let mut accept = false;
    let mut fuzz: Option<usize> = None;
    let mut seed: u64 = 1;
    // Diagnostic: write each selected case's scene JSON into this directory.
    let mut dump: Option<String> = None;
    let mut only: Option<Vec<String>> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--build" => build = true,
            "--quick" => quick = true,
            "--micro" => micro_tier = true,
            "--explain" => explain = true,
            "--accept-snapshots" => accept = true,
            "--fuzz" => fuzz = Some(args.next().and_then(|v| v.parse().ok()).expect("--fuzz N")),
            "--seed" => seed = args.next().and_then(|v| v.parse().ok()).expect("--seed S"),
            "--dump" => dump = args.next(),
            "--refresh" => set_refresh(true),
            "--only" => only = args.next().map(|v| v.split(',').map(str::to_string).collect()),
            other => panic!("unknown argument {other}"),
        }
    }
    if micro_tier || fuzz.is_some() || accept {
        tiers(build, micro_tier, explain, accept, fuzz, seed, only.as_deref());
        return;
    }
    let selected: Vec<_> = cases()
        .into_iter()
        .filter(|c| only.as_ref().is_none_or(|o| o.iter().any(|p| c.name.starts_with(p.as_str()))))
        .filter(|c| !quick || QUICK.contains(&c.name.as_str()))
        .collect();
    if let Some(dir) = dump {
        std::fs::create_dir_all(&dir).ok();
        for c in &selected {
            std::fs::write(format!("{dir}/{}.json", c.name), serde_json::to_string_pretty(&c.scene).unwrap()).ok();
        }
        return;
    }
    if build {
        let t = Instant::now();
        for (i, c) in selected.iter().enumerate() {
            let t_case = Instant::now();
            let hit = !std::env::var("STRESS_GATE_REFRESH").is_ok_and(|v| v == "1") && reference_set(&c.scene, false).is_some();
            if !hit {
                reference_set(&c.scene, true);
            }
            eprintln!("[{}/{}] {:<36} {}", i + 1, selected.len(), c.name, if hit { "cached".to_string() } else { format!("computed in {:.1} s", t_case.elapsed().as_secs_f64()) });
        }
        eprintln!("reference sets ready in {:.1} s", t.elapsed().as_secs_f64());
        return;
    }
    let started = Instant::now();
    let gpu = Gpu::new().expect("GPU");
    let failed = gate_cases(&gpu, &selected);
    eprintln!("{} of {} cases pass ({:.0} s)", selected.len() - failed, selected.len(), started.elapsed().as_secs_f64());
    if failed > 0 {
        std::process::exit(1);
    }
}

/// The accuracy gate over `selected` (printed as a table); the number that failed.
fn gate_cases(gpu: &Gpu, selected: &[Case]) -> usize {
    println!("| case | duration | gpu error | allowed | broken ref/gpu | fragments ref/gpu | gpu s | ref s | result |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut failed = 0;
    for c in selected {
        let Some(refs) = reference_set(&c.scene, false) else {
            println!("| {} | {:.2} s | - | - | - | - | - | - | NO REFERENCE (run --build) |", c.name, c.scene.sim.duration);
            failed += 1;
            continue;
        };
        if let Some(why) = unsupported(&c.scene) {
            println!("| {} | {:.2} s | - | - | - | - | - | - | unsupported: {why} |", c.name, c.scene.sim.duration);
            continue;
        }
        let t = Instant::now();
        let ours = match GpuWorld::new(gpu, &c.scene).and_then(|mut w| w.run(gpu)) {
            Ok(o) => o,
            Err(e) => {
                println!("| {} | {:.2} s | - | - | - | - | - | - | GPU ERROR: {e} |", c.name, c.scene.sim.duration);
                failed += 1;
                continue;
            }
        };
        let gpu_s = t.elapsed().as_secs_f64();
        let (err, _) = difference(&refs.reference, &ours);
        let allowed = allowed_error(&refs.reference, &refs.half, &refs.spreads);
        let mut fails = gate(&refs.reference, &ours, &refs.half, &refs.spreads);
        let mut verdict = "pass".to_string();
        if !fails.is_empty() && stress_gpu::gate::is_chaotic(&refs) {
            // A chaotic case: judge the GPU's distribution over the same input changes.
            let mut runs = vec![ours.clone()];
            for e in stress_gpu::gate::perturbations() {
                let s = stress_gpu::gate::perturbed_by(&c.scene, e);
                match GpuWorld::new(gpu, &s).and_then(|mut w| w.run(gpu)) {
                    Ok(o) => runs.push(o),
                    Err(e) => eprintln!("  {}: ensemble run failed: {e}", c.name),
                }
            }
            let ensemble = stress_gpu::gate::ensemble_gate(&refs, &runs);
            if ensemble.is_empty() {
                let b: Vec<String> = runs.iter().map(|o| format!("{}", o.values.get("broken_bonds").copied().unwrap_or(0.0))).collect();
                verdict = format!("pass (chaotic; GPU ensemble broken bonds {})", b.join(","));
            }
            fails = ensemble;
        }
        let v = |o: &stress_ref::observation::Observation, k: &str| o.values.get(k).copied().unwrap_or(0.0);
        println!(
            "| {} | {:.2} s | {err:.1e} | {allowed:.1e} | {}/{} | {}/{} | {gpu_s:.1} | {:.1} | {} |",
            c.name,
            c.scene.sim.duration,
            v(&refs.reference, "broken_bonds"),
            v(&ours, "broken_bonds"),
            v(&refs.reference, "fragments"),
            v(&ours, "fragments"),
            refs.reference_seconds,
            if fails.is_empty() { verdict } else { format!("FAIL: {}", fails.join("; ")) }
        );
        if !fails.is_empty() {
            // Where it differs, beside the reference's own spread and half step.
            let (_, notes) = difference(&refs.reference, &ours);
            let (_, spread) = stress_gpu::gate::spread_difference(&refs.reference, &refs.spreads);
            let (_, half) = difference(&refs.reference, &refs.half);
            eprintln!("  {}: gpu: {}", c.name, notes.join("; "));
            eprintln!("  {}: ref spread: {}", c.name, spread.join("; "));
            eprintln!("  {}: ref half step: {}", c.name, half.join("; "));
        }
        failed += !fails.is_empty() as usize;
    }
    failed
}

/// The micro and fuzz tiers (and accepting snapshots).
fn tiers(build: bool, micro_tier: bool, explain: bool, accept: bool, fuzz: Option<usize>, seed: u64, only: Option<&[String]>) {
    let started = Instant::now();
    let seeds = fuzz.map(|n| micro::fuzz_seeds(seed, n)).unwrap_or_default();
    if build {
        // References in parallel on CPU threads (each set runs its own runs in parallel).
        std::thread::scope(|sc| {
            if micro_tier {
                for case in micro::micro_cases().into_iter().filter(|c| only.is_none_or(|o| o.iter().any(|p| c.name.starts_with(p.as_str())))) {
                    sc.spawn(move || {
                        let t = Instant::now();
                        let hit = micro::micro_reference(&case, false).is_some();
                        if !hit {
                            micro::micro_reference(&case, true);
                        }
                        eprintln!("{:<24} {}", case.name, if hit { "cached".to_string() } else { format!("computed in {:.1} s", t.elapsed().as_secs_f64()) });
                    });
                }
            }
            for &s in seeds.iter().chain(if micro_tier { stress_gpu::gate::FUZZ_REGRESSIONS } else { &[] }) {
                sc.spawn(move || {
                    let scene = micro::fuzz_scene(s);
                    let hit = reference_set(&scene, false).is_some();
                    if !hit {
                        reference_set(&scene, true);
                    }
                    eprintln!("{:<24} {}", scene.name, if hit { "cached" } else { "computed" });
                });
            }
        });
        eprintln!("references ready in {:.1} s", started.elapsed().as_secs_f64());
        return;
    }
    let gpu = Gpu::new().expect("GPU");
    let mut failed = 0;
    if accept {
        // Snapshots are accepted only from a GPU that passes the quick gate.
        let quick: Vec<Case> = cases().into_iter().filter(|c| QUICK.contains(&c.name.as_str())).collect();
        let f = gate_cases(&gpu, &quick);
        if f > 0 {
            eprintln!("quick gate: {f} cases fail; snapshots not accepted");
            std::process::exit(1);
        }
    }
    if micro_tier || accept {
        let report = micro::run_micro(&gpu, only, accept, explain);
        print_report("micro", &report);
        print_coverage(&report);
        failed += report.failures() + report.missing();
    }
    if !seeds.is_empty() {
        let report = micro::run_fuzz(&gpu, &seeds, accept);
        print_report("fuzz", &report);
        let bad: Vec<String> = report.rows.iter().filter(|r| !r.fails.is_empty()).map(|r| r.case.trim_start_matches("fuzz-").to_string()).collect();
        if !bad.is_empty() {
            let mut bad = bad;
            bad.dedup();
            eprintln!("failing fuzz seeds (add to gate::FUZZ_REGRESSIONS): {}", bad.iter().map(|h| format!("0x{h}")).collect::<Vec<_>>().join(", "));
        }
        failed += report.failures() + report.missing();
    }
    eprintln!("{} ({:.1} s)", if failed == 0 { "all pass".to_string() } else { format!("{failed} failing or missing") }, started.elapsed().as_secs_f64());
    if failed > 0 {
        std::process::exit(1);
    }
}

fn print_report(tier: &str, report: &Report) {
    println!("| {tier} case | knobs | gpu s | snapshot | result |");
    println!("|---|---|---|---|---|");
    for r in &report.rows {
        let snap = match &r.snapshot {
            None => "-".to_string(),
            Some(SnapshotState::New) => "new".into(),
            Some(SnapshotState::Unchanged) => "unchanged".into(),
            Some(SnapshotState::Written) => "accepted".into(),
            Some(SnapshotState::Changed(what)) => format!("CHANGED: {}", what.join(",")),
        };
        let result = if r.missing { "NO REFERENCE (run --build)".to_string() } else if r.fails.is_empty() { "pass".into() } else { format!("FAIL: {}", r.fails.join("; ")) };
        println!("| {} | {} | {:.2} | {snap} | {result} |", r.case, r.knobs, r.seconds);
    }
    let changed = report.changed_snapshots();
    if changed > 0 {
        eprintln!("{changed} GPU snapshots changed (the GPU is deterministic: a kernel or schedule changed; --accept-snapshots once intended)");
    }
}

/// Which micro cases cover each path, and the paths none covers.
fn print_coverage(report: &Report) {
    println!();
    println!("| path | micro cases |");
    println!("|---|---|");
    let mut uncovered = Vec::new();
    for (bit, name) in coverage::NAMES {
        let by: Vec<&str> = report.coverage.iter().filter(|(_, c)| c & bit != 0).map(|(n, _)| n.as_str()).collect();
        if by.is_empty() {
            uncovered.push(name);
        }
        println!("| {name} | {} |", if by.is_empty() { "UNCOVERED".to_string() } else { by.join(", ") });
    }
    if !uncovered.is_empty() {
        eprintln!("uncovered paths: {}", uncovered.join(", "));
    }
}

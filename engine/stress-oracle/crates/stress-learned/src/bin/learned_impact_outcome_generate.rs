//! Plan and record an impact-outcome dataset (plan P1).
//!
//!   learned_impact_outcome_generate --job NAME [--runs DIR] [--catalog p1-walls|m5]
//!       [--family free_wall|anchored_wall] [--tier 1|2] [--n N] [--seed S] [--ranges pilot|bulk]
//!       [--oracle a12_realtime] [--jobs N] [--limit N] [--plan-only] [--no-baselines]
//!       [--max-seconds S] [--max-failure-fraction F] [--pause-file PATH]
//!       [--members M] [--ensemble-train-share F] [--no-ensembles]
//!       [--budget-gb G] [--min-free-gb F] [--no-thinning]
//!       [--split sealed|train|val|test] [--extrapolation speed|mass|size|loads] [--corner-share F]
//!
//! Stage 2: `--catalog p1-stage2 --family cuboid_house|masonry_house|arch --tier 3|4|5`.
//!
//! The sealed set: `--split sealed` (every configuration an ensemble, every frame kept), with
//! `--extrapolation` for the splits beyond the training ranges.
//!
//! Storage (default): arrays compressed with zstd, and training events that are not
//! ensemble members keep only the frames training uses (dropped frames regenerate
//! bit-identically with `learned_impact_outcome_record --regenerate`). `--budget-gb` caps
//! the whole `--runs` tree (default 19) and `--min-free-gb` the disk's free space (default
//! 100): when the next events could cross either, no new event starts, and the job ends
//! with state "budget" and a clear message.
//!
//! Every recipe gets a split (train/val/test by a seeded hash of its layout group; the
//! `m5_reference` family is sealed). Every val/test/sealed configuration, and a share of the
//! training ones (default 15%), is recorded as an ensemble of `--members` (default 8)
//! perturbed members (impact point ±1 mm across the approach, speed ±0.1%, a Weibull seed
//! each), which share the parent's split group and record their ensemble id.
//!
//! Writes `<runs>/<job>/recipes.json` before anything runs (an existing one must match
//! byte for byte: a changed plan needs a new job), then records every recipe into
//! `<runs>/<job>/events/<event_id>/`, skipping finished events (resume). One line per
//! finished event and a 30 s heartbeat go to `progress.log`; `status.json` is rewritten
//! every 30 s. Failed events are logged with their recipe and seed in `failures.jsonl`, and
//! generation continues; the job fails if failures exceed `--max-failure-fraction`. While
//! the `--pause-file` exists, no new event starts (running ones finish).
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;
use serde_json::json;
use stress_learned::oracle_config::OracleConfig;
use stress_learned::outcome::{record_outcome_event, OutcomeRecipe, RecordOptions};
use stress_learned::procedural::{m5_recipes, wall_recipes, with_splits_and_ensembles, EnsemblePlan, Ranges};
use stress_learned::progress::{duration_str, Progress};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let val = |s: &str| a.windows(2).find(|v| v[0] == s).map(|v| v[1].clone());
    let flag = |s: &str| a.iter().any(|x| x == s);
    let job = val("--job").expect("--job NAME required");
    let runs = PathBuf::from(val("--runs").unwrap_or_else(|| "learned-impact/runs".into()));
    let dir = runs.join(&job);
    let catalog = val("--catalog").unwrap_or_else(|| "p1-walls".into());
    let seed: u64 = val("--seed").map_or(1, |x| x.parse().expect("--seed"));
    let oracle = OracleConfig::named(&val("--oracle").unwrap_or_else(|| "a12_realtime".into())).expect("--oracle");
    let recipes: Vec<OutcomeRecipe> = match catalog.as_str() {
        "p1-walls" => {
            let family = val("--family").unwrap_or_else(|| "free_wall".into());
            let tier: u32 = val("--tier").map_or(1, |x| x.parse().expect("--tier"));
            let n: usize = val("--n").map_or(20, |x| x.parse().expect("--n"));
            let ranges = match (val("--extrapolation"), val("--ranges").as_deref().unwrap_or("pilot")) {
                (Some(x), _) => Ranges::extrapolation(&x).expect("--extrapolation"),
                (None, "pilot") => Ranges::pilot(),
                (None, "bulk") => Ranges::bulk(),
                (None, r) => panic!("unknown --ranges {r}"),
            };
            let mut ranges = ranges;
            if let Some(x) = val("--corner-share") {
                ranges.corner_share = x.parse().expect("--corner-share");
            }
            wall_recipes(&family, tier, n, seed, &ranges).expect("recipes")
        }
        "m5" => m5_recipes(),
        "p1-stage2" => {
            let family = val("--family").expect("--family cuboid_house|masonry_house|arch");
            let tier: u32 = val("--tier").map_or(3, |x| x.parse().expect("--tier"));
            let n: usize = val("--n").map_or(6, |x| x.parse().expect("--n"));
            stress_learned::stage2::stage2_recipes(&family, tier, n, seed).expect("stage-2 recipes")
        }
        c => panic!("unknown catalog {c}"),
    };
    // Splits by layout group, and ensembles on every validation/test/sealed configuration
    // and on a share of the training ones (`--members 1` or `--no-ensembles`: none).
    let mut plan = EnsemblePlan { seed, force_split: val("--split"), ..EnsemblePlan::default() };
    if let Some(m) = val("--members") {
        plan.members = m.parse().expect("--members");
    }
    if let Some(x) = val("--ensemble-train-share") {
        plan.train_share = x.parse().expect("--ensemble-train-share");
    }
    if flag("--no-ensembles") {
        plan.members = 1;
    }
    let mut recipes = with_splits_and_ensembles(recipes, &plan);
    if let Some(n) = val("--limit") {
        recipes.truncate(n.parse().expect("--limit"));
    }
    assert!(!recipes.is_empty(), "no recipes selected");
    let mut opts = RecordOptions::new(oracle.clone());
    opts.baselines = !flag("--no-baselines");
    if let Some(s) = val("--max-seconds") {
        opts.stop.max_seconds = s.parse().expect("--max-seconds");
    }
    if flag("--no-thinning") {
        opts.storage.select_training_frames = false;
    }
    let budget = (val("--budget-gb").map_or(19.0, |x| x.parse::<f64>().expect("--budget-gb")) * 1e9) as u64;
    let min_free = (val("--min-free-gb").map_or(100.0, |x| x.parse::<f64>().expect("--min-free-gb")) * 1e9) as u64;
    fs::create_dir_all(&dir).unwrap();
    let plan_json = json!({"job": job, "catalog": catalog, "seed": seed, "oracle_config": oracle, "baselines": opts.baselines, "stop_rule": opts.stop, "storage": opts.storage, "recipes": recipes});
    let bytes = serde_json::to_vec_pretty(&plan_json).unwrap();
    let path = dir.join("recipes.json");
    if path.exists() {
        assert!(fs::read(&path).unwrap() == bytes, "recipes.json differs from this plan: choose a new --job");
    } else {
        fs::write(&path, &bytes).unwrap();
    }
    eprintln!("{} recipes planned in {}", recipes.len(), path.display());
    if flag("--plan-only") {
        return;
    }

    let events = dir.join("events");
    fs::create_dir_all(&events).unwrap();
    let total = recipes.len();
    let progress = Progress::open(&dir, total).expect("progress log");
    let threads: usize = val("--jobs").map_or(8, |x| x.parse().expect("--jobs"));
    let max_fail: f64 = val("--max-failure-fraction").map_or(0.05, |x| x.parse().expect("--max-failure-fraction"));
    let pause = val("--pause-file").map(PathBuf::from);
    progress.line(&format!("[gen] start {job}: {total} events, {threads} threads, oracle {} (provisional {}), baselines {}", oracle.name, oracle.provisional, opts.baselines));
    let used = std::sync::atomic::AtomicU64::new(dir_bytes(&runs));
    let over = std::sync::atomic::AtomicBool::new(false);
    progress.line(&format!("[gen] budget: {:.2} GB used of {:.2} GB under {}, at least {:.0} GB free kept", used.load(Ordering::SeqCst) as f64 / 1e9, budget as f64 / 1e9, runs.display(), min_free as f64 / 1e9));
    let finished = AtomicUsize::new(0);
    let failures = AtomicUsize::new(0);
    let fail_log = Mutex::new(fs::OpenOptions::new().create(true).append(true).open(dir.join("failures.jsonl")).unwrap());
    let summaries = Mutex::new(fs::OpenOptions::new().create(true).append(true).open(dir.join("summaries.jsonl")).unwrap());
    let largest_event = std::sync::atomic::AtomicU64::new(0);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        recipes.par_iter().for_each(|r| {
            if events.join(&r.event_id).join("outcome.json").exists() {
                progress.update(|s| s.skipped += 1);
                return;
            }
            // The budget: the next `threads` events at the largest size seen so far must fit.
            {
                let largest = largest_event.load(Ordering::SeqCst).max(5_000_000);
                let need = used.load(Ordering::SeqCst) + largest * threads as u64;
                let free = free_bytes(&runs);
                if need > budget || free < min_free + largest * threads as u64 {
                    if !over.swap(true, Ordering::SeqCst) {
                        progress.line(&format!("[gen] STOP (budget): {:.2} GB used + {:.2} GB reserve > {:.2} GB budget, or {:.0} GB free < {:.0} GB minimum; no new events start", used.load(Ordering::SeqCst) as f64 / 1e9, (largest * threads as u64) as f64 / 1e9, budget as f64 / 1e9, free as f64 / 1e9, min_free as f64 / 1e9));
                    }
                    return;
                }
            }
            if let Some(p) = &pause {
                let mut said = false;
                while p.exists() {
                    if !said {
                        progress.line(&format!("[gen] paused by {} before {}", p.display(), r.event_id));
                        said = true;
                    }
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
            }
            let beat = |b: &stress_learned::outcome::Beat| progress.beat(b.event_id, b.frame, b.time, b.phase, b.breaks);
            let t = std::time::Instant::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| record_outcome_event(r, &events, &opts, &beat)))
                .unwrap_or_else(|e| Err(format!("panic: {}", e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())));
            progress.finish_event(&r.event_id);
            match result {
                Ok(s) => {
                    let bytes = dir_bytes(&events.join(&s.event_id));
                    used.fetch_add(bytes, Ordering::SeqCst);
                    largest_event.fetch_max(bytes, Ordering::SeqCst);
                    let k = finished.fetch_add(1, Ordering::SeqCst) + 1;
                    progress.update(|st| st.done += 1);
                    let el = progress.elapsed();
                    let rate = k as f64 / el * 60.0;
                    let left = total - k - failures.load(Ordering::SeqCst);
                    let phase = if s.truncated { "truncated" } else { "settled" };
                    progress.line(&format!(
                        "[gen] {k}/{total} events ({:.1}%) | {} done {:.2} s sim, {:.1} s wall, {} breaks, {} fragments, phase={phase} | {:.1} events/min | ETA {} | failures {}",
                        100.0 * k as f64 / total as f64, s.event_id, s.sim_seconds, t.elapsed().as_secs_f64(), s.broken_bonds, s.fragments, rate,
                        duration_str(left as f64 / (rate / 60.0)), failures.load(Ordering::SeqCst)
                    ));
                    let mut f = summaries.lock().unwrap();
                    writeln!(f, "{}", serde_json::to_string(&s).unwrap()).unwrap();
                }
                Err(e) => {
                    let n = failures.fetch_add(1, Ordering::SeqCst) + 1;
                    progress.update(|st| {
                        st.failures += 1;
                        st.last_error = Some(format!("{}: {e}", r.event_id));
                    });
                    progress.line(&format!("[gen] FAILED {} (seed {}, failure {n}): {e}", r.event_id, r.parameters["seed"]));
                    let mut f = fail_log.lock().unwrap();
                    writeln!(f, "{}", serde_json::to_string(&json!({"event_id": r.event_id, "error": e, "parameters": r.parameters, "family": r.family, "tier": r.tier})).unwrap()).unwrap();
                }
            }
        });
    });
    let fails = failures.load(Ordering::SeqCst);
    let failed = fails as f64 > max_fail * total as f64;
    progress.line(&format!("[gen] end {job}: {} recorded, {fails} failures, {:.0} s", finished.load(Ordering::SeqCst), progress.elapsed()));
    progress.update(|st| st.extra = json!({"bytes_used": used.load(Ordering::SeqCst), "budget_bytes": budget, "stopped_by_budget": over.load(Ordering::SeqCst)}));
    progress.close(if failed { "failed" } else if over.load(Ordering::SeqCst) { "budget" } else { "done" });
    if failed {
        eprintln!("failures {fails}/{total} exceed {max_fail}");
        std::process::exit(1);
    }
}

/// Bytes of every file under `p`.
fn dir_bytes(p: &std::path::Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![p.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).into_iter().flatten().flatten() {
            match e.metadata() {
                Ok(m) if m.is_dir() => stack.push(e.path()),
                Ok(m) => total += m.len(),
                Err(_) => {}
            }
        }
    }
    total
}

/// Free bytes on the volume holding `p` (`df -k`), or u64::MAX if unknown.
fn free_bytes(p: &std::path::Path) -> u64 {
    std::process::Command::new("df")
        .arg("-k")
        .arg(p)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|t| t.lines().nth(1).and_then(|l| l.split_whitespace().nth(3)).and_then(|x| x.parse::<u64>().ok()))
        .map_or(u64::MAX, |k| k * 1024)
}

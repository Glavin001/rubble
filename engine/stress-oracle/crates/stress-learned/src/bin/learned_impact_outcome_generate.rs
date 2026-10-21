//! Plan and record an impact-outcome dataset (plan P1).
//!
//!   learned_impact_outcome_generate --job NAME [--runs DIR] [--catalog p1-walls|m5]
//!       [--family free_wall|anchored_wall] [--tier 1|2] [--n N] [--seed S] [--ranges pilot|bulk]
//!       [--oracle a12_realtime] [--jobs N] [--limit N] [--plan-only] [--no-baselines]
//!       [--max-seconds S] [--max-failure-fraction F] [--pause-file PATH]
//!       [--members M] [--ensemble-train-share F] [--no-ensembles]
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
            let ranges = match val("--ranges").as_deref().unwrap_or("pilot") {
                "pilot" => Ranges::pilot(),
                "bulk" => Ranges::bulk(),
                r => panic!("unknown --ranges {r}"),
            };
            wall_recipes(&family, tier, n, seed, &ranges).expect("recipes")
        }
        "m5" => m5_recipes(),
        c => panic!("unknown catalog {c}"),
    };
    // Splits by layout group, and ensembles on every validation/test/sealed configuration
    // and on a share of the training ones (`--members 1` or `--no-ensembles`: none).
    let mut plan = EnsemblePlan { seed, ..EnsemblePlan::default() };
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
    fs::create_dir_all(&dir).unwrap();
    let plan_json = json!({"job": job, "catalog": catalog, "seed": seed, "oracle_config": oracle, "baselines": opts.baselines, "stop_rule": opts.stop, "recipes": recipes});
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
    let finished = AtomicUsize::new(0);
    let failures = AtomicUsize::new(0);
    let fail_log = Mutex::new(fs::OpenOptions::new().create(true).append(true).open(dir.join("failures.jsonl")).unwrap());
    let summaries = Mutex::new(fs::OpenOptions::new().create(true).append(true).open(dir.join("summaries.jsonl")).unwrap());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        recipes.par_iter().for_each(|r| {
            if events.join(&r.event_id).join("outcome.json").exists() {
                progress.update(|s| s.skipped += 1);
                return;
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
    progress.close(if failed { "failed" } else { "done" });
    if failed {
        eprintln!("failures {fails}/{total} exceed {max_fail}");
        std::process::exit(1);
    }
}

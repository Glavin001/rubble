//! Build P2's per-call records (`learned-impact-outcome/2`) from P1's frame records.
//!
//!   learned_impact_outcome_calls --out DIR [--plain] [--jobs N] [--eval-only | --train-only] RUN_OR_EVENT_DIR ...
//!
//! `--eval-only` takes the evaluation events (validation, test, sealed), whose calls are cached
//! permanently; `--train-only` every training event (ensemble members included), whose calls
//! are built on demand (`learned_impact/outcome_calls_cache.py`).
//!
//! Each argument is an event directory or a generator job (its `events/`). Writes
//! `DIR/<event_id>/` (manifest.json, calls.json, arrays.bin.zst; `--plain`: arrays.bin) and
//! a line per event; fails loud per event and exits non-zero if any failed.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use stress_learned::outcome_calls::{build_event, write_event};

fn events_in(p: &Path) -> Vec<PathBuf> {
    if p.join("context.json").exists() {
        return vec![p.to_path_buf()];
    }
    let root = if p.join("events").is_dir() { p.join("events") } else { p.to_path_buf() };
    let mut v: Vec<PathBuf> = std::fs::read_dir(&root).into_iter().flatten().flatten().map(|e| e.path()).filter(|q| q.join("outcome.json").exists()).collect();
    v.sort();
    v
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut out = None;
    let mut plain = false;
    let mut jobs = 8;
    let mut only: Option<bool> = None;
    let mut inputs = Vec::new();
    let mut it = a.into_iter();
    while let Some(x) = it.next() {
        match x.as_str() {
            "--out" => out = it.next().map(PathBuf::from),
            "--plain" => plain = true,
            "--eval-only" => only = Some(true),
            "--train-only" => only = Some(false),
            "--jobs" => jobs = it.next().and_then(|v| v.parse().ok()).expect("--jobs N"),
            _ => inputs.push(PathBuf::from(x)),
        }
    }
    let out = out.expect("--out DIR");
    let events: Vec<PathBuf> = inputs.iter().flat_map(|p| events_in(p)).filter(|e| only.map_or(true, |want| is_eval(e) == want)).collect();
    assert!(!events.is_empty(), "no events");
    let (done, failed) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let total = events.len();
    rayon::ThreadPoolBuilder::new().num_threads(jobs).build().unwrap().install(|| {
        events.par_iter().for_each(|ev| {
            let name = ev.file_name().unwrap().to_string_lossy().to_string();
            match build_event(ev).and_then(|c| write_event(&out.join(&name), &c, !plain).map(|_| c.calls.len())) {
                Ok(n) => {
                    let k = done.fetch_add(1, Ordering::SeqCst) + 1;
                    eprintln!("[calls] {k}/{total} {name}: {n} calls");
                }
                Err(e) => {
                    failed.fetch_add(1, Ordering::SeqCst);
                    eprintln!("[calls] FAILED {name}: {e}");
                }
            }
        });
    });
    let f = failed.load(Ordering::SeqCst);
    eprintln!("[calls] {} built, {f} failed", done.load(Ordering::SeqCst));
    if f > 0 {
        std::process::exit(1);
    }
}

/// An evaluation event (validation, test, sealed), whose calls are cached; every training
/// event, ensemble members included, has its calls built on demand (the 20 GB budget).
fn is_eval(ev: &Path) -> bool {
    let ctx: serde_json::Value = std::fs::read(ev.join("context.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    ctx["split"].as_str() != Some("train")
}

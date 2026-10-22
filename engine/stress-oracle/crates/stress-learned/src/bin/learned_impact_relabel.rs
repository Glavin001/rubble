//! Harness A relabelling for DAgger (`stress_learned::relabel`), batched and deterministic.
//!
//!   learned_impact_relabel --jobs JOBS.json --out DIR [--no-calls] [--threads N]
//!
//! JOBS.json: `[{"event": EVENT_DIR, "frame": F, "state": STATE.bin | null}, …]` (`state`: a
//! MessagePack `IslandState`, e.g. a model's prediction; null: the oracle's own state of the
//! routing's island at F). Per job `i` writes `DIR/<i>_start.bin` and `DIR/<i>_end.bin`
//! (the oracle's frame-end island state: the label), `DIR/<i>_call/` (the call's inputs,
//! `learned-impact-outcome/3`), and one JSON line on stdout. Events run in parallel; each
//! event's jobs share one replay.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rayon::prelude::*;
use serde_json::{json, Value};
use stress_learned::outcome_calls::write_event;
use stress_learned::relabel::{read_state, relabel_event, write_state};

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut jobs, mut out, mut calls, mut threads) = (None, None, true, 0usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--jobs" => jobs = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--no-calls" => calls = false,
            "--threads" => threads = args.next().and_then(|v| v.parse().ok()).expect("--threads N"),
            other => panic!("unknown argument {other}"),
        }
    }
    let (jobs, out) = (jobs.expect("--jobs JOBS.json"), out.expect("--out DIR"));
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    }
    std::fs::create_dir_all(&out).expect("out dir");
    let list: Vec<Value> = serde_json::from_slice(&std::fs::read(&jobs).expect("jobs file")).expect("jobs json");
    let mut by_event: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, j) in list.iter().enumerate() {
        by_event.entry(j["event"].as_str().expect("event").to_string()).or_default().push(i);
    }
    let results: Vec<Result<Vec<Value>, String>> = by_event
        .par_iter()
        .map(|(event, idx)| -> Result<Vec<Value>, String> {
            let js: Vec<(usize, Option<stress_ref::world::IslandState>)> = idx
                .iter()
                .map(|&i| -> Result<_, String> {
                    let f = list[i]["frame"].as_u64().ok_or("frame")? as usize;
                    let st = match list[i]["state"].as_str() {
                        Some(p) => Some(read_state(std::path::Path::new(p))?),
                        None => None,
                    };
                    Ok((f, st))
                })
                .collect::<Result<_, _>>()?;
            let done = relabel_event(std::path::Path::new(event), &js, calls)?;
            let mut lines = Vec::new();
            for (&i, r) in idx.iter().zip(&done) {
                write_state(&out.join(format!("{i}_start.bin")), &r.start)?;
                write_state(&out.join(format!("{i}_end.bin")), &r.end)?;
                if let Some(c) = &r.call {
                    write_event(&out.join(format!("{i}_call")), c, true)?;
                }
                lines.push(json!({"job": i, "event": event, "frame": r.frame, "chunks": r.members.chunks.len(), "impactors": r.members.impactors.len()}));
            }
            Ok(lines)
        })
        .collect();
    let mut failed = 0;
    for r in results {
        match r {
            Ok(ls) => ls.iter().for_each(|l| println!("{l}")),
            Err(e) => {
                failed += 1;
                eprintln!("relabel failed: {e}");
            }
        }
    }
    std::process::exit(i32::from(failed > 0));
}

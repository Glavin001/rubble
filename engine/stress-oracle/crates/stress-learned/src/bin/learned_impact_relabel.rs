//! Harness A relabelling for DAgger (`stress_learned::relabel`), batched and deterministic.
//!
//!   learned_impact_relabel --jobs JOBS.json --out DIR [--no-calls] [--threads N] [--calls v3]
//!
//! `--full-state-columns M`: `learned-impact-outcome/5` calls with the columns and static split
//! of M (a calls manifest's `full_state_columns`, a model.json's `full_state.columns`, or the
//! object itself); `own`: the call's own split.
//! `--calls v3`: the calls downgraded to `learned-impact-outcome/3` (for models trained on /3;
//! exact only from a builder revision-5 build, refused otherwise).
//!
//! JOBS.json: `[{"event": EVENT_DIR, "frame": F, "decoded": D.json | "state": S.msgpack | neither,
//! "parent_call": any, "split_group": G}, …]`:
//! - `decoded`: a model's `learned-impact-decoded-state/1` outcome of frame F, taken as
//!   harness A takes it; the relabelled call is frame F + 1's;
//! - `state`: a MessagePack `IslandState` imposed at frame F's start (exact);
//! - neither: the oracle's own state of the routing's island at F.
//!
//! Per job `i`: `DIR/<i>_start.msgpack`, `DIR/<i>_end.msgpack` (the oracle's frame-end island
//! state: the label), `DIR/<i>_call/` (one `learned-impact-outcome/4` call, inputs from that
//! start, targets the oracle's frame; manifest `source` "dagger", `dagger` the provenance,
//! (a job the oracle or the call builder rejects, e.g. a blown-up model state, is a line with
//! `error`, listed in `DIR/failures.jsonl`, and skipped: the others go on),
//! `split_group` the job's or the event's), and one JSON line on stdout. Events run in
//! parallel; each event's jobs share one replay of its native oracle.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rayon::prelude::*;
use serde_json::{json, Value};
use stress_learned::outcome_calls::write_event;
use stress_ref::world::DecodedState;
use stress_learned::relabel::{read_state, relabel_event_jobs, write_state, Provided};

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut jobs, mut out, mut calls, mut threads, mut v3) = (None, None, true, 0usize, false);
    let mut full_state: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--jobs" => jobs = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--no-calls" => calls = false,
            "--full-state-columns" => full_state = Some(args.next().expect("--full-state-columns MANIFEST.json|own")),
            "--calls" => v3 = match args.next().as_deref() {
                Some("v3") => true,
                Some("v4") => false,
                other => panic!("--calls v3|v4, not {other:?}"),
            },
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
    // `--full-state-columns`: /5 calls, with a training manifest's (or model.json's
    // "full_state") columns and static split, or `own` for the call's own split.
    let fs_columns: Option<Value> = full_state.map(|p| {
        if p == "own" {
            return Value::Null;
        }
        let m: Value = serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))).expect("columns json");
        let cols = if !m["full_state_columns"].is_null() { m["full_state_columns"].clone() } else if !m["full_state"]["columns"].is_null() { m["full_state"]["columns"].clone() } else { m };
        assert!(cols.is_object(), "{p}: no full_state_columns");
        cols
    });
    let mut by_event: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, j) in list.iter().enumerate() {
        by_event.entry(j["event"].as_str().expect("event").to_string()).or_default().push(i);
    }
    let results: Vec<Result<Vec<Value>, String>> = by_event
        .par_iter()
        .map(|(event, idx)| -> Result<Vec<Value>, String> {
            let t = std::time::Instant::now();
            let js: Vec<(usize, Provided)> = idx
                .iter()
                .map(|&i| -> Result<_, String> {
                    let j = &list[i];
                    let f = j["frame"].as_u64().ok_or("frame")? as usize;
                    let p = if let Some(p) = j["decoded"].as_str() {
                        let d: DecodedState = serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?).map_err(|e| format!("{p}: {e}"))?;
                        Provided::Decoded(d)
                    } else if let Some(p) = j["state"].as_str() {
                        Provided::Start(read_state(std::path::Path::new(p))?)
                    } else {
                        Provided::Own
                    };
                    Ok((f, p))
                })
                .collect::<Result<_, _>>()?;
            let done = relabel_event_jobs(std::path::Path::new(event), &js, calls, fs_columns.as_ref())?;
            let secs = t.elapsed().as_secs_f64();
            let mut lines = Vec::new();
            for (&i, r) in idx.iter().zip(done) {
                // A job the oracle or the call builder rejects is reported and skipped (not
                // admissible); the batch goes on.
                let mut r = match r {
                    Ok(r) => r,
                    Err(e) => {
                        lines.push(json!({"job": i, "event": event, "frame": list[i]["frame"], "error": e}));
                        continue;
                    }
                };
                let j = &list[i];
                write_state(&out.join(format!("{i}_start.msgpack")), &r.start)?;
                write_state(&out.join(format!("{i}_end.msgpack")), &r.end)?;
                if let Some(c) = r.call.as_mut() {
                    let provided = if j["decoded"].is_string() { "decoded" } else if j["state"].is_string() { "state" } else { "oracle" };
                    c.manifest["source"] = json!("dagger");
                    c.manifest["dagger"] = json!({"parent_event": event, "parent_frame": j["frame"], "parent_call": j["parent_call"], "provided": provided,
                        "decoded": j["decoded"], "state": j["state"], "call_frame": r.frame,
                        "decoded_keeps_deformation": std::env::var("STRESS_DECODED_DEFORMATION").is_ok_and(|v| v == "1")});
                    if !j["split_group"].is_null() {
                        c.manifest["split_group"] = j["split_group"].clone();
                    }
                    let parent = c.manifest["event_id"].as_str().unwrap_or_default().to_string();
                    let tag = if j["parent_call"].is_null() { format!("j{i}") } else { format!("c{}", j["parent_call"]) };
                    c.manifest["parent_event_id"] = json!(parent);
                    c.manifest["parent_call"] = j["parent_call"].clone();
                    c.manifest["event_id"] = json!(format!("{parent}__dagger_{tag}_f{}", r.frame));
                    if v3 {
                        stress_learned::outcome_calls::downgrade_to_v3(c)?;
                    }
                    write_event(&out.join(format!("{i}_call")), c, true)?;
                }
                lines.push(json!({"job": i, "event": event, "frame": r.frame, "chunks": r.members.chunks.len(), "impactors": r.members.impactors.len(), "event_seconds": secs, "event_jobs": idx.len()}));
            }
            Ok(lines)
        })
        .collect();
    let (mut failed, mut rejected) = (0, Vec::new());
    for r in results {
        match r {
            Ok(ls) => {
                for l in ls {
                    if l.get("error").is_some() {
                        rejected.push(l.to_string());
                    }
                    println!("{l}");
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("relabel failed: {e}");
            }
        }
    }
    // Rejected jobs (not admissible) are listed in DIR/failures.jsonl; they do not fail the run.
    if !rejected.is_empty() {
        std::fs::write(out.join("failures.jsonl"), rejected.join("\n") + "\n").expect("failures.jsonl");
        eprintln!("relabel: {} job(s) rejected (not admissible), listed in {}", rejected.len(), out.join("failures.jsonl").display());
    }
    std::process::exit(i32::from(failed > 0));
}

//! Harness A relabelling for DAgger (`stress_learned::relabel`), batched and deterministic.
//!
//!   learned_impact_relabel --jobs JOBS.json --out DIR [--no-calls] [--threads N]
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
//! `split_group` the job's or the event's), and one JSON line on stdout. Events run in
//! parallel; each event's jobs share one replay of its native oracle.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rayon::prelude::*;
use serde_json::{json, Value};
use stress_learned::outcome_calls::write_event;
use stress_ref::world::DecodedState;
use stress_learned::relabel::{read_state, relabel_event, write_state, Provided};

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
            let done = relabel_event(std::path::Path::new(event), &js, calls)?;
            let secs = t.elapsed().as_secs_f64();
            let mut lines = Vec::new();
            for (&i, mut r) in idx.iter().zip(done) {
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
                    write_event(&out.join(format!("{i}_call")), c, true)?;
                }
                lines.push(json!({"job": i, "event": event, "frame": r.frame, "chunks": r.members.chunks.len(), "impactors": r.members.impactors.len(), "event_seconds": secs, "event_jobs": idx.len()}));
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

//! The completeness gate on recorded events: every checked boundary f, rebuilt from its
//! full state (plus `scene.json`) and stepped one oracle frame, must equal boundary f + 1
//! bit for bit.
//!
//!   learned_impact_full_state_check [--jobs N] [--every K] EVENT_DIR ...
//!
//! One JSON line per event (boundaries checked, the first differing one and its paths);
//! exits non-zero if any event differs.
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use serde_json::json;
use stress_learned::full_state::{differing_paths, load_frame, oracle_frame, restore, shred_world};
use stress_ref::scene::Scene;

fn check(ev: &PathBuf, every: usize) -> Result<serde_json::Value, String> {
    let scene: Scene = serde_json::from_slice(&std::fs::read(ev.join("scene.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let ctx: serde_json::Value = serde_json::from_slice(&std::fs::read(ev.join("context.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let n = ctx["full_state"]["boundaries"].as_array().ok_or("no full state")?.len();
    let mut checked = 0;
    for b in (0..n.saturating_sub(1)).step_by(every.max(1)) {
        let mut w = restore(&scene, &load_frame(ev, b)?)?;
        oracle_frame(&mut w, |_| {})?;
        let diff = differing_paths(&shred_world(&w)?, &load_frame(ev, b + 1)?, |_| false);
        if !diff.is_empty() {
            return Ok(json!({"event": ev, "ok": false, "boundary": b, "paths": diff.iter().take(10).collect::<Vec<_>>(), "checked": checked}));
        }
        checked += 1;
    }
    Ok(json!({"event": ev, "ok": true, "boundaries": n, "checked": checked}))
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (mut jobs, mut every, mut events) = (4usize, 1usize, Vec::new());
    let mut it = a.into_iter();
    while let Some(x) = it.next() {
        match x.as_str() {
            "--jobs" => jobs = it.next().and_then(|v| v.parse().ok()).expect("--jobs N"),
            "--every" => every = it.next().and_then(|v| v.parse().ok()).expect("--every K"),
            _ => events.push(PathBuf::from(x)),
        }
    }
    let bad = AtomicUsize::new(0);
    rayon::ThreadPoolBuilder::new().num_threads(jobs).build().unwrap().install(|| {
        events.par_iter().for_each(|ev| {
            let r = check(ev, every).unwrap_or_else(|e| json!({"event": ev, "ok": false, "error": e}));
            if r["ok"] != true {
                bad.fetch_add(1, Ordering::SeqCst);
            }
            println!("{r}");
        })
    });
    let bad = bad.into_inner();
    eprintln!("[check] {} events, {bad} differ", events.len());
    std::process::exit(i32::from(bad > 0));
}

//! Classify every field of the full-state contract (P2's request): at chosen frame
//! boundaries, reset one field path at a time to its kind's zero (0, false, an empty list or
//! map), rebuild the World, step one oracle frame and compare with the untouched step.
//!
//!   learned_impact_full_state_classify --out DIR [--jobs N] [--timeout-s S]
//!
//! Per path and trial: `unexercised` (already zero there), `restore_fails`, `step_fails`
//! (a panic or the timeout), `state` (another field's next value differs), `derived`
//! (nothing differs: recomputed from the rest within the frame) or `diagnostic` (only the
//! field's own next value differs: nothing reads it). A path is classified `state` when any
//! trial says so (or fails), else `diagnostic` when any trial says so, else `derived`; a
//! path no trial exercised stays `unexercised`. Writes `DIR/classification.json` and
//! `DIR/classification.md`.
//!
//! Trials run in child processes (`--trial SET --path P`) so a reset that stalls the solver
//! is cut off by the timeout.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use stress_learned::full_state::{decode, differing_paths, encode, oracle_frame, restore, shred_world, under, unshred};
use stress_learned::oracle_config::OracleConfig;
use stress_learned::procedural::{m5_free_wall, m5_wall_row, wall_recipes, Ranges};
use stress_ref::scene::Scene;
use stress_ref::world::World;

fn trial(set: &Path, path: &str) -> Value {
    let scene: Scene = serde_json::from_slice(&std::fs::read(set.join("scene.json")).unwrap()).unwrap();
    let s0 = decode(&std::fs::read(set.join("s0.bin")).unwrap()).unwrap();
    let reference = decode(&std::fs::read(set.join("ref.bin")).unwrap()).unwrap();
    let v0 = unshred(&s0).unwrap();
    let mut v = v0.clone();
    stress_learned::full_state::reset_path(&mut v, path);
    if v == v0 {
        return json!({"result": "unexercised"});
    }
    let frame = stress_learned::full_state::shred_value(&v).unwrap();
    let mut w = match restore(&scene, &frame) {
        Ok(w) => w,
        Err(e) => return json!({"result": "restore_fails", "error": e.chars().take(300).collect::<String>()}),
    };
    std::panic::set_hook(Box::new(|_| {}));
    let stepped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| oracle_frame(&mut w, |_| {}).map(|_| shred_world(&w))));
    let got = match stepped {
        Ok(Ok(Ok(f))) => f,
        Ok(Ok(Err(e))) | Ok(Err(e)) => return json!({"result": "step_fails", "error": e}),
        Err(_) => return json!({"result": "step_fails", "error": "panic"}),
    };
    let others = differing_paths(&reference, &got, |q| under(q, path));
    let own = differing_paths(&reference, &got, |q| !under(q, path));
    let result = if !others.is_empty() { "state" } else if own.is_empty() { "derived" } else { "diagnostic" };
    json!({"result": result, "others_differing": others.len(), "first_others": others.iter().take(8).collect::<Vec<_>>()})
}

/// A trial set: a scene stepped to `frame` (the first boundary at which `pick` holds, plus
/// `after` frames), its state there and the untouched next state.
fn make_set(dir: &Path, name: &str, mut scene: Scene, max_frames: usize, after: usize, pick: &dyn Fn(&World, bool) -> bool) -> Option<PathBuf> {
    OracleConfig::named("a12_realtime").unwrap().apply(&mut scene);
    scene.sim.duration = max_frames as f64 * scene.sim.frame_dt;
    let mut w = World::new(&scene);
    let mut hit: Option<usize> = None;
    for f in 0..max_frames {
        if hit.is_some_and(|h| f >= h + after) {
            let set = dir.join(name);
            std::fs::create_dir_all(&set).unwrap();
            std::fs::write(set.join("scene.json"), serde_json::to_vec(&scene).unwrap()).unwrap();
            let s0 = shred_world(&w).unwrap();
            std::fs::write(set.join("s0.bin"), encode(&s0).unwrap()).unwrap();
            let t = Instant::now();
            let mut r = restore(&scene, &s0).unwrap();
            oracle_frame(&mut r, |_| {}).unwrap();
            std::fs::write(set.join("ref.bin"), encode(&shred_world(&r).unwrap()).unwrap()).unwrap();
            std::fs::write(set.join("meta.json"), serde_json::to_vec(&json!({"frame": f, "step_s": t.elapsed().as_secs_f64(), "paths": s0.columns.len()})).unwrap()).unwrap();
            eprintln!("[classify] set {name}: frame {f}, {} paths, step {:.2} s", s0.columns.len(), t.elapsed().as_secs_f64());
            return Some(set);
        }
        let events = w.solver.events.len();
        let mut touched = false;
        oracle_frame(&mut w, |x| touched |= x.chunk_contact_hits().iter().flatten().any(|&h| h > 0.0)).unwrap();
        if hit.is_none() && pick(&w, touched || w.solver.events.len() > events) {
            // the boundary at the start of the next frame
            hit = Some(f + 1);
        }
    }
    eprintln!("[classify] set {name}: condition never held");
    None
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let val = |s: &str| a.windows(2).find(|v| v[0] == s).map(|v| v[1].clone());
    if let Some(set) = val("--trial") {
        println!("{}", trial(Path::new(&set), &val("--path").expect("--path")));
        return;
    }
    let out = PathBuf::from(val("--out").expect("--out DIR"));
    let jobs: usize = val("--jobs").map_or(4, |x| x.parse().unwrap());
    let timeout = val("--timeout-s").map_or(120.0, |x| x.parse().unwrap());
    std::fs::create_dir_all(&out).unwrap();
    let active = |_: &World, a: bool| a;
    let mut sets = Vec::new();
    // Mid-impact boundaries (the frame after a contact or break) and a later one.
    sets.extend(make_set(&out, "m5_free_wall_after_strike", m5_free_wall(), 30, 0, &active));
    sets.extend(make_set(&out, "m5_wall_row_after_strike", m5_wall_row(), 30, 0, &active));
    let anchored = wall_recipes("anchored_wall", 2, 1, 7001, &Ranges::pilot()).unwrap().remove(0);
    sets.extend(make_set(&out, "anchored_t2_contact", anchored.scene.clone(), 60, 0, &active));
    sets.extend(make_set(&out, "anchored_t2_later", anchored.scene, 90, 12, &active));
    let free = wall_recipes("free_wall", 2, 1, 7002, &Ranges::pilot()).unwrap().remove(0);
    sets.extend(make_set(&out, "free_t2_contact", free.scene, 60, 0, &active));

    let exe = std::env::current_exe().unwrap();
    let mut work: Vec<(PathBuf, String)> = Vec::new();
    for set in &sets {
        let s0 = decode(&std::fs::read(set.join("s0.bin")).unwrap()).unwrap();
        work.extend(s0.columns.keys().filter(|p| !p.is_empty()).map(|p| (set.clone(), p.clone())));
    }
    let total = work.len();
    eprintln!("[classify] {total} trials over {} sets, {jobs} at a time", sets.len());
    let queue = Mutex::new(work.into_iter());
    let results: Mutex<BTreeMap<String, BTreeMap<String, Value>>> = Mutex::new(BTreeMap::new());
    let done = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..jobs {
            sc.spawn(|| loop {
                let Some((set, path)) = queue.lock().unwrap().next() else { break };
                let mut child = Command::new(&exe).args(["--trial", set.to_str().unwrap(), "--path", &path]).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
                let t = Instant::now();
                let r = loop {
                    if let Some(st) = child.try_wait().unwrap() {
                        let mut s = String::new();
                        use std::io::Read;
                        child.stdout.take().unwrap().read_to_string(&mut s).unwrap();
                        break serde_json::from_str::<Value>(s.trim()).unwrap_or_else(|_| json!({"result": "step_fails", "error": format!("exit {st}")}));
                    }
                    if t.elapsed().as_secs_f64() > timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        break json!({"result": "step_fails", "error": format!("timeout {timeout} s")});
                    }
                    std::thread::sleep(Duration::from_millis(20));
                };
                let name = set.file_name().unwrap().to_string_lossy().to_string();
                results.lock().unwrap().entry(path).or_default().insert(name, r);
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n % 100 == 0 {
                    eprintln!("[classify] {n}/{total}");
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    let mut table = BTreeMap::new();
    for (path, trials) in &results {
        let rs: Vec<&str> = trials.values().map(|t| t["result"].as_str().unwrap_or("")).collect();
        let class = if rs.iter().any(|r| matches!(*r, "state" | "restore_fails" | "step_fails")) {
            "state"
        } else if rs.contains(&"diagnostic") {
            "diagnostic"
        } else if rs.contains(&"derived") {
            "derived"
        } else {
            "unexercised"
        };
        table.insert(path.clone(), json!({"class": class, "trials": trials}));
    }
    std::fs::write(out.join("classification.json"), serde_json::to_vec_pretty(&table).unwrap()).unwrap();
    let mut md = String::from("| path | class | trials (set: result) |\n|---|---|---|\n");
    for (path, t) in &table {
        let trials: Vec<String> = t["trials"].as_object().unwrap().iter().map(|(k, v)| format!("{k}: {}", v["result"].as_str().unwrap_or(""))).collect();
        md += &format!("| `{path}` | {} | {} |\n", t["class"].as_str().unwrap(), trials.join(", "));
    }
    std::fs::write(out.join("classification.md"), md).unwrap();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for t in table.values() {
        *counts.entry(t["class"].as_str().unwrap()).or_default() += 1;
    }
    eprintln!("[classify] {} paths: {counts:?}", table.len());
}

//! Benchmark (ignored): what a step costs as a map fills with buildings
//! (engine/docs/large-maps.md, "Gates and benchmark").
//!
//! K threshold cuboid houses (`rubble::city`, the builder `city.html` shares) are laid on a grid over the same 2.2 km square, and the
//! per-step cost of four scenarios is measured for each K:
//! - `still`: nothing moves after settling;
//! - `drive`: one kinematic car drives a street, touching nothing;
//! - `demolish`: one house is struck;
//! - `war`: 20 houses are struck at once.
//! The goal is a cost that does not grow with K in any of them.
//!
//! CITY_K=16,64,256 cargo test -p engine-world --release --test city_scale -- --ignored --nocapture
//! CITY_SLEEP=1: the large-map sleep policy (`sleeping_solver_for`; the threshold solver's
//! own suggestion is never asleep).
//! CITY_JSON=path appends one JSON line per run. RUBBLE_GPU_TIMING=1: the GPU column is every
//! pass of the step (its timestamp marks; the waits they add inflate the wall column).
#![cfg(feature = "stress")]

use rubble::city::*;
use rubble::*;
use std::time::Instant;

struct Run {
    k: usize,
    scenario: &'static str,
    wall_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    /// `World::step` alone: the host's part of a step.
    host_ms: Vec<f64>,
    awake: u32,
    broken: u32,
}

fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p).round() as usize]
}

/// The city of the env's settings (`rubble::city`, shared with `city.html`).
fn def(k: usize, sleep: bool) -> CityDef {
    CityDef {
        houses: k,
        map: std::env::var("CITY_MAP")
            .map(|s| s.parse().unwrap())
            .unwrap_or(CITY_MAP),
        sleep,
        // CITY_POOL=n: a world piece pool of n slots (`WorldDef::fracture_pieces`).
        pool: std::env::var("CITY_POOL").ok().map(|v| v.parse().unwrap()),
    }
}

fn city(asset: &DestructibleAssetDesc, k: usize, sleep: bool) -> (World, City) {
    let def = def(k, sleep);
    let mut w = World::new(GpuContext::Default, &def.world_def(asset).unwrap()).unwrap();
    let t = Instant::now();
    let mut c = City::begin(&mut w, asset, def).unwrap();
    c.place(&mut w, k).unwrap();
    if std::env::var("CITY_DEBUG").is_ok() {
        eprintln!(
            "placed {k} houses in {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    (w, c)
}

fn frame(w: &mut World, broken: &mut u32) -> f64 {
    let t = Instant::now();
    w.step().unwrap();
    t.elapsed().as_secs_f64() * 1e3 + finish(w, broken)
}

/// Wait for the step's frame and count its broken bonds: ms.
fn finish(w: &mut World, broken: &mut u32) -> f64 {
    let t = Instant::now();
    for f in w.poll_blocking().unwrap() {
        let n = bonds_broken(&f.destruction_events);
        if n > 0 && std::env::var("CITY_DEBUG").is_ok() {
            eprintln!("step {}: broke {n}", f.step);
        }
        *broken += n;
    }
    // CITY_COUNTERS=1: the engine's counters after each frame (bodies, awake, contacts solved).
    if std::env::var("CITY_COUNTERS").is_ok() {
        if let Some(c) = w.counters() {
            eprintln!(
                "counters bodies {} awake {} contacts {} manifolds {} colors {} rounds {} cand {}",
                c.body_count,
                c.awake_count,
                c.contact_count,
                c.manifold_count,
                c.color_count,
                c.color_rounds,
                c.candidate_count
            );
        }
    }
    t.elapsed().as_secs_f64() * 1e3
}

fn run(asset: &DestructibleAssetDesc, k: usize, scenario: &'static str, sleep: bool) -> Run {
    let (mut w, c) = city(asset, k, sleep);
    let mut broken = 0;
    for _ in 0..CITY_SETTLE {
        frame(&mut w, &mut broken);
    }
    c.start(&mut w, CityScenario::from_name(scenario).unwrap())
        .unwrap();
    let broken_before = broken;
    let (mut wall_ms, mut gpu_ms, mut host_ms) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..CITY_MEASURE {
        let t = Instant::now();
        w.step().unwrap();
        host_ms.push(t.elapsed().as_secs_f64() * 1e3);
        wall_ms.push(host_ms.last().unwrap() + finish(&mut w, &mut broken));
        // RUBBLE_GPU_TIMING=1: every pass's GPU time (the step's marks), else the sim's.
        gpu_ms.push(w.last_step_gpu_ms().unwrap_or_else(|| w.profile_ms()));
    }
    let awake = w.counters().map(|c| c.awake_count).unwrap_or(0);
    Run {
        k,
        scenario,
        wall_ms,
        gpu_ms,
        host_ms,
        awake,
        broken: broken - broken_before,
    }
}

#[test]
#[ignore]
fn city_scale() {
    let ks: Vec<usize> = std::env::var("CITY_K")
        .unwrap_or_else(|_| "16,64,256".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let scenarios: Vec<String> = std::env::var("CITY_SCENARIOS")
        .unwrap_or_else(|_| "still,drive,demolish,war".into())
        .split(',')
        .map(String::from)
        .collect();
    let sleep = std::env::var("CITY_SLEEP").is_ok();
    let asset = city_house().unwrap();
    let chunks = asset.chunks.len();
    println!(
        "cuboid house: {chunks} chunks, {} bonds; sleep {sleep}",
        asset.bonds.len()
    );
    println!("| K | chunks | scenario | wall median ms | wall p95 ms | host median ms | gpu median ms | awake | bonds broken |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for &k in &ks {
        for s in &scenarios {
            let s: &'static str = Box::leak(s.clone().into_boxed_str());
            let r = run(&asset, k, s, sleep);
            let (wm, wp, hm, gm) = (
                percentile(&r.wall_ms, 0.5),
                percentile(&r.wall_ms, 0.95),
                percentile(&r.host_ms, 0.5),
                percentile(&r.gpu_ms, 0.5),
            );
            println!(
                "| {} | {} | {} | {wm:.2} | {wp:.2} | {hm:.2} | {gm:.2} | {} | {} |",
                r.k,
                r.k * chunks,
                r.scenario,
                r.awake,
                r.broken
            );
            if let Ok(path) = std::env::var("CITY_JSON") {
                use std::io::Write;
                let line = format!(
                    "{{\"k\":{},\"chunks\":{},\"scenario\":\"{}\",\"sleep\":{sleep},\"wall_median_ms\":{wm:.3},\"wall_p95_ms\":{wp:.3},\"host_median_ms\":{hm:.3},\"gpu_median_ms\":{gm:.3},\"awake\":{},\"broken\":{}}}\n",
                    r.k, r.k * chunks, r.scenario, r.awake, r.broken
                );
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .unwrap()
                    .write_all(line.as_bytes())
                    .unwrap();
            }
        }
    }
}

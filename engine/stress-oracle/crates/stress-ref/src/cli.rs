//! The `stress-ref` command line, shared by every rigid-body engine's binary: the
//! `stress-ref` binary runs scenes in the standalone world (the oracle), `stress-physx`
//! with PhysX as the rigid-body engine and the standalone world as its baseline.
//!
//! ```text
//! stress-ref gen-scenes <dir>                       write the scene catalogue (with derived bond data;
//!                                                    showcases into <dir>/showcases)
//! stress-ref with-seed <scene.json> <seed> <out.json>  same scene, Weibull strengths for another seed
//! stress-ref run <scene.json> [--seed N] [--out obs.json] [--mode explicit|adaptive|quasi_static]
//!                [--stiffness-scale S] [--max-substep DT] [--frame-dt DT] [--no-fracture]
//!                [--feature NAME=on|off]... [--set dotted.path=JSON]...
//! stress-ref compare <scene.json> <ours.json> [<oracle.json> ...] [--json report.json]
//! stress-ref seeds <scene.json> --out <dir> [--from 0] [--count 20]  run a scene for many Weibull seeds
//! stress-ref check <scenes-dir> <golden-dir> [--only NAME] [--json report.json]
//! ```
//!
//! `check` runs every scene, compares against analytic expectations and every golden
//! oracle observation in `<golden-dir>/<scene>/*.json`, and exits non-zero on failure.
//! Goldens with a non-zero seed (`<tool>_seedN.json`) are compared as distributions:
//! ours is run for the same seeds.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::metrics::{compare, format_table, Comparison};
use crate::observation::Observation;
use crate::scene::Scene;
use crate::world::World;
use crate::{builders, showcases};

/// Builds the world a scene runs in (standalone, or coupled to an engine).
pub type WorldFactory = dyn Fn(&Scene) -> World;

fn usage() -> ExitCode {
    eprintln!("usage: stress-ref gen-scenes <dir> | with-seed <scene> <seed> <out> | run <scene> [opts] | seeds <scene> --out <dir> | compare <scene> <ours> [oracles..] | check <scenes> <golden>");
    ExitCode::from(2)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn positional(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
            continue;
        }
        if a.starts_with("--") {
            skip = a != "--no-fracture";
            continue;
        }
        out.push(a.clone());
    }
    out
}

fn apply_overrides(scene: &mut Scene, args: &[String]) -> Result<(), String> {
    if let Some(v) = flag(args, "--seed") {
        scene.sim.seed = v.parse().map_err(|e| format!("--seed: {e}"))?;
    }
    if let Some(v) = flag(args, "--mode") {
        scene.sim.solve_mode = serde_json::from_value(serde_json::Value::String(v)).map_err(|e| format!("--mode: {e}"))?;
    }
    if let Some(v) = flag(args, "--stiffness-scale") {
        scene.sim.stiffness_scale = v.parse().map_err(|e| format!("--stiffness-scale: {e}"))?;
    }
    if let Some(v) = flag(args, "--max-substep") {
        scene.sim.max_substep = Some(v.parse().map_err(|e| format!("--max-substep: {e}"))?);
    }
    if args.iter().any(|a| a == "--no-fracture") {
        scene.sim.fracture = false;
    }
    if let Some(v) = flag(args, "--frame-dt") {
        scene.sim.frame_dt = v.parse().map_err(|e| format!("--frame-dt: {e}"))?;
    }
    for (i, a) in args.iter().enumerate() {
        let Some(v) = args.get(i + 1) else { continue };
        match a.as_str() {
            "--feature" => {
                let (name, state) = v.split_once('=').ok_or("--feature takes name=on|off")?;
                let on = match state {
                    "on" | "true" => true,
                    "off" | "false" => false,
                    _ => return Err(format!("--feature {v}: expected on or off")),
                };
                *scene = scene.with_feature(name, on)?;
            }
            "--set" => {
                let (path, value) = v.split_once('=').ok_or("--set takes path=json")?;
                *scene = scene.with_override(path, value)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn run_scene(make: &WorldFactory, scene: &Scene) -> Observation {
    let start = std::time::Instant::now();
    let mut world = make(scene);
    let mut obs = world.run();
    obs.values.insert("wall_seconds".into(), start.elapsed().as_secs_f64());
    obs
}

fn load_goldens(dir: &Path) -> Vec<Observation> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        paths.sort();
        for p in paths {
            if p.extension().is_some_and(|e| e == "json") && !p.file_name().unwrap().to_string_lossy().starts_with("provenance") {
                match Observation::load(&p) {
                    Ok(o) => out.push(o),
                    Err(e) => eprintln!("warning: {e}"),
                }
            }
        }
    }
    out
}

/// For an engine-coupled run: how much of the movable bodies' motion the engine
/// integrated, and how much the world simulated in impact islands.
fn engine_share(obs: &Observation) -> String {
    let (Some(e), Some(i)) = (obs.values.get("engine_body_frames"), obs.values.get("island_body_frames")) else {
        return String::new();
    };
    if e + i == 0.0 {
        return "  engine: no movable bodies (all motion is the stress solve's)\n".into();
    }
    format!("  engine: {e} body-frames integrated by {}, {i} in impact islands ({:.0}% engine)\n", obs.solver, 100.0 * e / (e + i))
}

/// Rows comparing `ours` with a run of the same scene in the `baseline` world (the
/// standalone reference), each metric gated at the scene's own tolerance for it.
fn compare_to_baseline(scene: &Scene, ours: &Observation, baseline: &Observation) -> Vec<Comparison> {
    let mut gate = scene.clone();
    for m in &mut gate.metrics {
        m.expected = None;
        m.oracles.clear();
        m.oracle_tolerance = None;
    }
    let mut rows = compare(&gate, ours, std::slice::from_ref(baseline));
    for r in &mut rows {
        r.reference_source = format!("{} (standalone)", baseline.solver);
    }
    rows
}

/// Run the command line `args` (without the program name), with scenes run in the
/// worlds `make` builds. With a `baseline` (another engine's binary), `run` and `check`
/// also run every scene in the baseline world and gate the two against each other.
pub fn main(make: &WorldFactory, baseline: Option<&WorldFactory>, args: Vec<String>) -> ExitCode {
    let Some(cmd) = args.first().cloned() else { return usage() };
    let rest = &args[1..];
    let pos = positional(rest);
    let result: Result<bool, String> = (|| match cmd.as_str() {
        "gen-scenes" => {
            let dir = PathBuf::from(pos.first().ok_or("missing <dir>")?);
            // Showcases go to a subdirectory: `check` runs only the benchmarks.
            let show_dir = dir.join("showcases");
            std::fs::create_dir_all(&show_dir).map_err(|e| e.to_string())?;
            let scenes = builders::catalog().into_iter().map(|s| (dir.clone(), s));
            for (dir, scene) in scenes.chain(showcases::catalog().into_iter().map(|s| (show_dir.clone(), s))) {
                scene.validate()?;
                let path = dir.join(format!("{}.json", scene.name));
                // Large scenes are written compactly; both forms parse identically.
                let derived = scene.with_derived();
                let chunks: usize = derived.bodies.iter().map(|b| b.chunks.len()).sum();
                let text = if chunks < 300 { derived.to_json_pretty() } else { serde_json::to_string(&derived).unwrap() };
                std::fs::write(&path, text + "\n").map_err(|e| e.to_string())?;
                println!("wrote {}", path.display());
            }
            Ok(true)
        }
        "with-seed" => {
            // Re-derive a scene for another Weibull seed (exporters read derived.weibull).
            let mut scene = Scene::load(Path::new(pos.first().ok_or("missing <scene>")?))?;
            scene.sim.seed = pos.get(1).ok_or("missing <seed>")?.parse().map_err(|e| format!("seed: {e}"))?;
            let out = pos.get(2).ok_or("missing <out>")?;
            std::fs::write(out, serde_json::to_string(&scene.with_derived()).unwrap() + "\n").map_err(|e| e.to_string())?;
            Ok(true)
        }
        "seeds" => {
            // Run a scene for seeds first..first+count (randomized strengths).
            let scene = Scene::load(Path::new(pos.first().ok_or("missing <scene>")?))?;
            let first: u64 = flag(rest, "--from").map(|v| v.parse().unwrap_or(0)).unwrap_or(0);
            let count: u64 = flag(rest, "--count").map(|v| v.parse().unwrap_or(20)).unwrap_or(20);
            let out = PathBuf::from(flag(rest, "--out").ok_or("missing --out <dir>")?);
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            for seed in first..first + count {
                let mut sc = scene.clone();
                sc.sim.seed = seed;
                apply_overrides(&mut sc, rest)?;
                let obs = run_scene(make, &sc);
                let path = out.join(format!("stress-ref_seed{seed}.json"));
                std::fs::write(&path, serde_json::to_string(&obs).unwrap() + "\n").map_err(|e| e.to_string())?;
                println!("wrote {}", path.display());
            }
            Ok(true)
        }
        "run" => {
            let mut scene = Scene::load(Path::new(pos.first().ok_or("missing <scene>")?))?;
            apply_overrides(&mut scene, rest)?;
            if rest.iter().any(|a| a == "--profile") {
                crate::profile::enable();
            }
            crate::profile::reset();
            let obs = run_scene(make, &scene);
            if crate::profile::enabled() {
                eprint!("{}", crate::profile::report());
            }
            let text = obs.to_json_pretty();
            match flag(rest, "--out") {
                Some(p) => std::fs::write(&p, text + "\n").map_err(|e| e.to_string())?,
                None => println!("{text}"),
            }
            let mut rows = compare(&scene, &obs, &[]);
            if let Some(b) = baseline {
                rows.extend(compare_to_baseline(&scene, &obs, &run_scene(b, &scene)));
            }
            eprint!("{}", format_table(&scene.name, &rows));
            Ok(rows.iter().all(|r| r.pass != Some(false)))
        }
        "compare" => {
            let scene = Scene::load(Path::new(pos.first().ok_or("missing <scene>")?))?;
            let ours = Observation::load(Path::new(pos.get(1).ok_or("missing <ours>")?))?;
            let oracles: Vec<Observation> = pos[2..].iter().map(|p| Observation::load(Path::new(p))).collect::<Result<_, _>>()?;
            let rows = compare(&scene, &ours, &oracles);
            print!("{}", format_table(&scene.name, &rows));
            if let Some(p) = flag(rest, "--json") {
                std::fs::write(p, serde_json::to_string_pretty(&rows).unwrap()).map_err(|e| e.to_string())?;
            }
            Ok(rows.iter().all(|r| r.pass != Some(false)))
        }
        "check" => {
            let scenes = PathBuf::from(pos.first().ok_or("missing <scenes-dir>")?);
            let golden = PathBuf::from(pos.get(1).ok_or("missing <golden-dir>")?);
            let only = flag(rest, "--only");
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&scenes)
                .map_err(|e| format!("{}: {e}", scenes.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
                .collect();
            paths.sort();
            let mut all: Vec<(String, Vec<Comparison>)> = Vec::new();
            let mut ok = true;
            for p in paths {
                let mut scene = Scene::load(&p)?;
                if only.as_ref().is_some_and(|o| !scene.name.contains(o.as_str())) {
                    continue;
                }
                apply_overrides(&mut scene, rest)?;
                {
                    let all_goldens = load_goldens(&golden.join(&scene.name));
                    let (seeded, oracles): (Vec<Observation>, Vec<Observation>) =
                        all_goldens.into_iter().partition(|o| o.notes.iter().any(|n| n == "seeded") || o.seed != 0);
                    let obs = run_scene(make, &scene);
                    let mut rows = compare(&scene, &obs, &oracles);
                    if let Some(b) = baseline {
                        rows.extend(compare_to_baseline(&scene, &obs, &run_scene(b, &scene)));
                    }
                    // Distributions: run ours for the same seeds as each oracle's seeded goldens.
                    let mut by_tool: std::collections::BTreeMap<String, Vec<Observation>> = Default::default();
                    for o in seeded {
                        by_tool.entry(o.solver.clone()).or_default().push(o);
                    }
                    for (tool, theirs) in by_tool {
                        let ours: Vec<Observation> = theirs
                            .iter()
                            .map(|o| {
                                let mut sc = scene.clone();
                                sc.sim.seed = o.seed;
                                run_scene(make, &sc)
                            })
                            .collect();
                        rows.extend(crate::metrics::compare_distributions(&scene, &ours, &theirs, &tool));
                    }
                    print!("{}", format_table(&scene.name, &rows));
                    print!("{}", engine_share(&obs));
                    ok &= rows.iter().all(|r| r.pass != Some(false));
                    all.push((scene.name.clone(), rows));
                }
            }
            if let Some(p) = flag(rest, "--json") {
                std::fs::write(p, serde_json::to_string_pretty(&all).unwrap()).map_err(|e| e.to_string())?;
            }
            Ok(ok)
        }
        "ab" => {
            // A/B of method switches (IMPROVEMENTS.md): every scene as loaded (A) and
            // with the switches given by `--feature NAME=on|off` (B), gated against the
            // oracles both ways, with every recorded value's change.
            let scenes = PathBuf::from(pos.first().ok_or("missing <scenes-dir>")?);
            let golden = PathBuf::from(pos.get(1).ok_or("missing <golden-dir>")?);
            let only = flag(rest, "--only");
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&scenes)
                .map_err(|e| format!("{}: {e}", scenes.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
                .collect();
            paths.sort();
            let switches: Vec<(String, bool)> = rest
                .iter()
                .zip(rest.iter().skip(1))
                .filter(|(f, _)| f.as_str() == "--feature")
                .map(|(_, v)| {
                    let (n, s) = v.split_once('=').ok_or(format!("--feature {v}: expected NAME=on|off"))?;
                    Ok((n.to_string(), matches!(s, "on" | "true")))
                })
                .collect::<Result<_, String>>()?;
            if switches.is_empty() {
                return Err("ab needs at least one --feature NAME=on|off (the B side)".into());
            }
            let base_args: Vec<String> = {
                // `--set` overrides apply to both sides; `--feature` only to B.
                let mut v = Vec::new();
                let mut it = rest.iter();
                while let Some(a) = it.next() {
                    if a == "--feature" {
                        it.next();
                    } else {
                        v.push(a.clone());
                    }
                }
                v
            };
            let (mut pass_a, mut pass_b, mut gated, mut regressions) = (0, 0, 0, Vec::new());
            let (mut wall_a, mut wall_b) = (0.0, 0.0);
            let mut report = Vec::new();
            for p in paths {
                let mut a = Scene::load(&p)?;
                if only.as_ref().is_some_and(|o| !a.name.contains(o.as_str())) {
                    continue;
                }
                apply_overrides(&mut a, &base_args)?;
                let mut b = a.clone();
                for (n, on) in &switches {
                    b = b.with_feature(n, *on)?;
                }
                let oracles: Vec<Observation> = load_goldens(&golden.join(&a.name)).into_iter().filter(|o| !o.notes.iter().any(|n| n == "seeded") && o.seed == 0).collect();
                let (oa, ob) = (run_scene(make, &a), run_scene(make, &b));
                let (ra, rb) = (compare(&a, &oa, &oracles), compare(&b, &ob, &oracles));
                println!("scene {}  (A {:.2} s, {} substeps; B {:.2} s, {} substeps)", a.name, oa.values["wall_seconds"], oa.values.get("substeps").copied().unwrap_or(0.0), ob.values["wall_seconds"], ob.values.get("substeps").copied().unwrap_or(0.0));
                wall_a += oa.values["wall_seconds"];
                wall_b += ob.values["wall_seconds"];
                for (x, y) in ra.iter().zip(&rb) {
                    let fmt = |v: &Option<crate::metrics::MetricValue>| v.as_ref().map_or("-".into(), |v| v.to_string());
                    let tag = |r: &Comparison| match r.pass {
                        Some(true) => "PASS",
                        Some(false) => "FAIL",
                        None => "----",
                    };
                    let err = |r: &Comparison| r.relative_error.map_or(String::new(), |e| format!(" {:.1}%", 100.0 * e));
                    println!("  {:28} vs {:14} A {} {}{}  B {} {}{}", x.metric, x.reference_source, tag(x), fmt(&x.ours), err(x), tag(y), fmt(&y.ours), err(y));
                    if x.pass.is_some() {
                        gated += 1;
                        pass_a += usize::from(x.pass == Some(true));
                        pass_b += usize::from(y.pass == Some(true));
                        if x.pass == Some(true) && y.pass == Some(false) {
                            regressions.push(format!("{}: {} vs {}", a.name, x.metric, x.reference_source));
                        }
                    }
                }
                let mut changes = Vec::new();
                for (k, va) in &oa.values {
                    if k == "wall_seconds" {
                        continue;
                    }
                    let vb = ob.values.get(k).copied().unwrap_or(f64::NAN);
                    let rel = if *va != 0.0 { (vb - va) / va.abs() } else if vb == 0.0 { 0.0 } else { f64::INFINITY };
                    if rel != 0.0 {
                        changes.push(format!("{k} {va:.6e} -> {vb:.6e} ({:+.2}%)", 100.0 * rel));
                    }
                }
                for (k, fa) in &oa.flags {
                    if ob.flags.get(k) != Some(fa) {
                        changes.push(format!("flag {k} {fa} -> {:?}", ob.flags.get(k)));
                    }
                }
                for c in &changes {
                    println!("    {c}");
                }
                report.push(serde_json::json!({ "scene": a.name, "a": ra, "b": rb, "a_values": oa.values, "b_values": ob.values }));
            }
            println!("\nA/B {}: gated metrics passing A {pass_a}/{gated}, B {pass_b}/{gated}; wall A {wall_a:.1} s, B {wall_b:.1} s", switches.iter().map(|(n, o)| format!("{n}={}", if *o { "on" } else { "off" })).collect::<Vec<_>>().join(" "));
            for r in &regressions {
                println!("  REGRESSION {r}");
            }
            if let Some(p) = flag(rest, "--json") {
                std::fs::write(p, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
            }
            Ok(regressions.is_empty())
        }
        _ => Err(format!("unknown command '{cmd}'")),
    })();
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("some metrics failed");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

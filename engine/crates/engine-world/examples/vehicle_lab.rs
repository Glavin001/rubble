//! The vehicle lab offline (`rubble::vehicle_lab`, `engine/docs/vehicle-lab.md`): the
//! page's scripted scenarios without a browser, and the lab's engine measurements, each
//! ending in a table of numbers.
//!
//! ```text
//! cargo run --release -p engine-world --example vehicle_lab -- --script <file.json> [--mode cpu|gpu]
//! cargo run --release -p engine-world --example vehicle_lab -- --measure plates|spheres|mountain [--mode cpu|gpu]
//! ```
//! - `--script`: a scenario of `engine-webgpu/vehicle-lab-scripts/` (the page's `?script=`).
//! - `--mode cpu` (default): the CPU reference (`vehicle_lab::cpu`): no GPU. `--mode gpu`:
//!   the GPU World (takes the machine's GPU lock; run it under a kill timer,
//!   `perl -e 'alarm 600; exec @ARGV'`, one GPU job at a time).
//! - `--car showcase` (default: the page's authored sedan, low resolution) | `sedan` (the
//!   explicit solver's procedural sedan).
//! - `--surface 0|1`: the car's opt-in surface contact (default: the script's, else off).
//! - `--secs S`: the run's length (default: the script's duration).
//! - `--seed N`, `--steepness D`, `--roughness R`: the mountain's.
//! - `--json <file>`: the stats as JSON.
//! - `--every N`: a progress line every N windows.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticShape, STEP};
use glam::{Quat, Vec3};
use rubble::vehicle_lab::cpu::CpuLab;
use rubble::vehicle_lab::monitor::{ShapeView, ViewGeom};
use rubble::vehicle_lab::{Lab, LabOptions, Scene};
use rubble::*;
use serde_json::{json, Value};

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn car(name: &str) -> anyhow::Result<ExplicitBeams> {
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        ..Default::default()
    };
    let text = match name {
        "showcase" => {
            use engine_softbody::authoring::{build, style, BuildParams, Resolution};
            let def = style("sedan").ok_or_else(|| anyhow::anyhow!("no sedan style"))?;
            build(
                &def,
                &BuildParams {
                    resolution: Resolution::Low,
                    ..BuildParams::default()
                },
            )
            .map_err(anyhow::Error::msg)?
            .truck
        }
        "sedan" => {
            engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
                minimass: 10.0,
                ..engine_softbody::procedural::CarParams::sedan()
            })
        }
        other => anyhow::bail!("unknown --car {other} (showcase, sedan)"),
    };
    let (s, _) =
        ExplicitBeams::from_truck_raycast(&text, None, Vec3::ZERO, 0.0, Vec3::ZERO, settings)
            .map_err(anyhow::Error::msg)?;
    Ok(s)
}

fn options(script: &Value) -> LabOptions {
    let mut v = script.clone();
    let o = v.as_object_mut().expect("a script object");
    if let Some(s) = arg("--surface") {
        o.insert("surface".into(), json!(s == "1"));
    }
    let mut m = o.get("mountain").cloned().unwrap_or_else(|| json!({}));
    for (flag, key) in [
        ("--seed", "seed"),
        ("--steepness", "steepness"),
        ("--roughness", "roughness"),
        ("--loose", "loose"),
        ("--start", "start"),
    ] {
        if let Some(x) = arg(flag).and_then(|x| x.parse::<f64>().ok()) {
            m[key] = if key == "seed" || key == "loose" {
                json!(x as u64)
            } else {
                json!(x)
            };
        }
    }
    o.insert("mountain".into(), m);
    LabOptions::from_json(&v)
}

/// A scripted run; returns the lab's stats and table.
fn run(
    template: &ExplicitBeams,
    script: &Value,
    gpu: bool,
    secs: f64,
) -> anyhow::Result<(Value, String)> {
    let opts = options(script);
    let windows = (secs * 60.0).round() as u64;
    let every = arg("--every")
        .and_then(|x| x.parse::<u64>().ok())
        .unwrap_or(0);
    let t0 = std::time::Instant::now();
    let (stats, table) = if gpu {
        let (mut world, mut lab) = Lab::build(GpuContext::Default, template, opts, |_| {})?;
        lab.read_blocking(&world, true)?;
        // `restart` in the script (the page's crash presets and restarts): applied in
        // place, as the page does, after the build.
        if let Some(r) = script.get("restart") {
            let t = std::time::Instant::now();
            lab.restart(&mut world, r)?;
            lab.read_blocking(&world, true)?;
            println!(
                "restart in place: {:.1} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        lab.bench.set_script(script)?;
        for k in 0..windows {
            lab.before_step(&mut world)?;
            world.step()?;
            world.poll()?;
            lab.read_blocking(&world, k % 6 == 5 || k + 1 == windows)?;
            anyhow::ensure!(
                !lab.bench.car.non_finite,
                "non-finite node state at t = {:.3} s",
                lab.bench.t
            );
            if every > 0 && k % every == 0 {
                let c = &lab.bench.car;
                println!(
                    "  t {:6.2}  y {:9.2}  {:6.1} km/h  broken {:4}  depth {:.3}  proxies {:?}",
                    lab.bench.t,
                    c.centre.y,
                    c.speed * 3.6,
                    c.beams_broken,
                    c.node_depth,
                    c.proxies
                );
            }
        }
        world.poll_blocking()?;
        if let Some(x) = world.explicit_vehicles() {
            x.kernel_status().map_err(anyhow::Error::msg)?;
        }
        (lab.bench.stats(), lab.bench.table())
    } else {
        let mut lab = CpuLab::new(template, opts)?;
        lab.bench.set_script(script)?;
        for k in 0..windows {
            lab.step()?;
            if every > 0 && k % every == 0 {
                let c = &lab.bench.car;
                // The load a GPU frame of this state carries (engine/docs/vehicle-lab.md,
                // "GPU overload"): the node box that sizes the proxy query, and the
                // triangles past the grid walk's 64 cells (each tests every candidate).
                let (live, every_tri, extent) = lab.sim.self_collision_load();
                println!(
                    "  t {:6.2}  y {:9.2}  {:6.1} km/h  broken {:4}  depth {:.3}  proxies {:?}  box {:8.1} m  triangles {live} ({every_tri} test every one of {} candidates)",
                    lab.bench.t,
                    c.centre.y,
                    c.speed * 3.6,
                    c.beams_broken,
                    c.node_depth,
                    c.proxies,
                    extent,
                    lab.sim.candidates.len()
                );
            }
        }
        (lab.bench.stats(), lab.bench.table())
    };
    let wall = t0.elapsed().as_secs_f64();
    Ok((
        stats,
        format!(
            "{table}wall {wall:.1} s for {secs:.1} s simulated ({:.2}× real time)\n",
            secs / wall.max(1e-9)
        ),
    ))
}

fn flat(events: Value, surface: bool) -> Value {
    json!({ "scene": "flat", "surface": surface, "events": events })
}

/// (a) alternative: the plate as the car's own solver's static shape, moved by the host
/// each 2 kHz step (the solver has no moving static shapes: the shape is teleported, its
/// contact knows no velocity). Returns the deepest a node gets inside it.
fn static_shape_plate(template: &ExplicitBeams, speed: f32, secs: f32) -> (f32, bool) {
    let mut sim = vehicle_lab::lab_car(template, &LabOptions::default(), None);
    sim.run(0.5);
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    for n in &sim.nodes {
        lo = lo.min(sim.origin + n.pos);
        hi = hi.max(sim.origin + n.pos);
    }
    let half = Vec3::new(0.5 * (hi.x - lo.x) + 0.5, 0.15, 0.5 * (hi.z - lo.z) + 0.5);
    let mut centre = Vec3::new(0.5 * (lo.x + hi.x), hi.y + 0.15 + 0.15, 0.5 * (lo.z + hi.z));
    let at = sim.shapes.len();
    sim.shapes
        .push(StaticShape::oriented_box(centre, half, Quat::IDENTITY));
    let (mut deepest, mut finite) = (0.0f32, true);
    let stop = hi.y * 0.5;
    for k in 0..(secs / STEP) as u32 {
        if centre.y - 0.15 > stop {
            centre.y -= speed * STEP;
        }
        sim.shapes[at] = StaticShape::oriented_box(centre, half, Quat::IDENTITY);
        sim.step();
        if k % 33 == 0 {
            let view = ShapeView {
                collider: 0,
                body: 0,
                geom: ViewGeom::Box { half },
                pos: centre - sim.origin,
                rot: Quat::IDENTITY,
                impulse: Vec3::ZERO,
                peak: 0.0,
            };
            for n in &sim.nodes {
                finite &= n.pos.is_finite();
                if let Some(d) = view.depth(n.pos) {
                    deepest = deepest.max(d);
                }
            }
        }
    }
    (deepest, finite)
}

fn measure(template: &ExplicitBeams, what: &str, gpu: bool) -> anyhow::Result<()> {
    let mode = if gpu { "GPU World" } else { "CPU reference" };
    match what {
        "plates" => {
            println!("(a) kinematic plates, {mode}: deepest car node inside a plate against the plate's speed");
            println!(
                "{:<6} {:>7} {:>10} {:>10} {:>10} {:>10} {:>9} {:>8} {:>8}",
                "axis",
                "m/s",
                "travel m",
                "depth m",
                "peak kN",
                "work kJ",
                "broken",
                "deform",
                "finite"
            );
            for axis in ["top", "sides"] {
                for speed in [0.02f32, 0.05, 0.1, 0.2, 0.5] {
                    // The same travel at every speed (the run's length sets it, not the stop
                    // gap): 0.15 m of clearance, then 0.3 m of crush.
                    let secs = f64::from(0.5 + 0.45 / speed);
                    let gap = 0.2;
                    let script = flat(
                        json!([{ "t": 0.5, "plates": { "axis": axis, "speed": speed, "gap": gap } }]),
                        false,
                    );
                    match run(template, &script, gpu, secs) {
                        Ok((s, _)) => {
                            for p in s["plates"].as_array().unwrap() {
                                println!(
                                    "{:<6} {:>7.2} {:>10.3} {:>10.3} {:>10.1} {:>10.2} {:>9} {:>8.3} {:>8}",
                                    p["name"].as_str().unwrap(),
                                    speed,
                                    p["travel"].as_f64().unwrap(),
                                    p["nodeDepth"].as_f64().unwrap(),
                                    p["forcePeak"].as_f64().unwrap() / 1e3,
                                    p["work"].as_f64().unwrap() / 1e3,
                                    s["car"]["beams"]["broken"],
                                    s["car"]["deformPeak"].as_f64().unwrap(),
                                    !s["car"]["nonFinite"].as_bool().unwrap()
                                );
                            }
                        }
                        Err(e) => println!("{axis:<6} {speed:>7.2} failed: {e}"),
                    }
                }
            }
            if !gpu {
                println!("(a) alternative, CPU: the top plate as the car's own static shape, teleported each 2 kHz step");
                println!("{:>7} {:>10} {:>8}", "m/s", "depth m", "finite");
                for speed in [0.02f32, 0.05, 0.1, 0.2, 0.5] {
                    let (d, ok) = static_shape_plate(template, speed, 0.5 + 0.6 / speed);
                    println!("{speed:>7.2} {d:>10.3} {ok:>8}");
                }
            }
        }
        "spheres" => {
            println!("(b) a 0.1 m steel sphere (32.9 kg) into the left side, {mode}");
            println!(
                "{:<8} {:>6} {:>8} {:>10} {:>9} {:>9} {:>9} {:>8} {:>7}  outcome",
                "surface",
                "m/s",
                "impact",
                "p given %",
                "peak kN",
                "exit m/s",
                "depth m",
                "in hull",
                "broken"
            );
            for surface in [false, true] {
                for speed in [20.0f32, 40.0, 60.0] {
                    for along in [0.0f32, 0.45] {
                        let script = flat(
                            json!([{ "t": 0.5, "fire": { "shape": "sphere", "radius": 0.1, "material": "steel", "speed": speed,
                            "aim": { "side": "left", "height": 0.8, "along": along, "distance": 3 } } }]),
                            surface,
                        );
                        match run(template, &script, gpu, 1.5) {
                            Ok((s, _)) => {
                                let h = &s["shots"][0];
                                println!(
                                    "{:<8} {:>6.0} {:>8} {:>10.1} {:>9.1} {:>9.1} {:>9.3} {:>8} {:>7}  {} (aim along {along})",
                                    if surface { "on" } else { "off" },
                                    speed,
                                    h["impactSpeed"].as_f64().map_or("-".into(), |x| format!("{x:.1}")),
                                    100.0 * h["momentumAlong"].as_f64().unwrap() / h["momentumOwn"].as_f64().unwrap(),
                                    h["peakForce"].as_f64().unwrap() / 1e3,
                                    h["exitAlong"].as_f64().unwrap_or(f64::NAN),
                                    h["nodeDepth"].as_f64().unwrap(),
                                    h["insideHull"],
                                    s["car"]["beams"]["broken"],
                                    h["outcome"].as_str().unwrap()
                                );
                            }
                            Err(e) => println!(
                                "{:<8} {speed:>6.0} failed: {e}",
                                if surface { "on" } else { "off" }
                            ),
                        }
                    }
                }
            }
        }
        "mountain" => {
            let secs = arg("--secs").and_then(|x| x.parse().ok()).unwrap_or(20.0);
            let script = json!({ "scene": "mountain", "surface": arg("--surface").as_deref() == Some("1"), "events": [] });
            let (s, table) = run(template, &script, gpu, secs)?;
            println!("mountain, {mode}, {secs} s\n{table}");
            println!("{:>7} {:>8} {:>8}", "t s", "broken", "yielded");
            for d in s["damage"].as_array().unwrap().iter().step_by(4) {
                println!("{:>7.1} {:>8} {:>8}", d[0].as_f64().unwrap(), d[1], d[2]);
            }
        }
        other => anyhow::bail!("unknown --measure {other} (plates, spheres, mountain)"),
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let gpu = match arg("--mode").as_deref() {
        None | Some("cpu") => false,
        Some("gpu") => true,
        Some(other) => anyhow::bail!("unknown --mode {other} (cpu, gpu)"),
    };
    let name = arg("--car").unwrap_or_else(|| "showcase".into());
    let template = car(&name)?;
    println!(
        "car {name}: {} nodes, {} beams, {} collision triangles, {:.0} kg",
        template.nodes.len(),
        template.beams.len(),
        template.triangles.len(),
        template.nodes.iter().map(|n| n.mass).sum::<f32>()
    );
    if let Some(what) = arg("--measure") {
        return measure(&template, &what, gpu);
    }
    if let Some(kind) = arg("--crash") {
        // A crash preset (the page's): --crash wall|offset|pole|sidepole [--kmh 50]
        // [--overlap 0.4] [--offset 0] [--angle 0]; GPU World only (the restart is its).
        let f = |k: &str, d: f64| arg(k).and_then(|x| x.parse().ok()).unwrap_or(d);
        let script = json!({ "scene": "flat", "surface": arg("--surface").as_deref() == Some("1"), "events": [],
            "restart": { "crash": { "kind": kind, "kmh": f("--kmh", 50.0), "overlap": f("--overlap", 0.4), "offset": f("--offset", 0.0), "angle": f("--angle", 0.0) } } });
        anyhow::ensure!(gpu, "--crash runs in the GPU World (--mode gpu)");
        let (stats, table) = run(&template, &script, true, f("--secs", 1.0))?;
        print!("{table}");
        println!(
            "stated {:?} km/h, arrived at {:?} km/h",
            stats["targetKmh"], stats["arrivalKmh"]
        );
        return Ok(());
    }
    let path = arg("--script").ok_or_else(|| {
        anyhow::anyhow!("--script <file.json> or --measure plates|spheres|mountain")
    })?;
    let script: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let opts = options(&script);
    let secs = arg("--secs")
        .and_then(|x| x.parse().ok())
        .or_else(|| script.get("duration").and_then(Value::as_f64))
        .unwrap_or(5.0);
    println!(
        "{}: {} ({}), surface contact {}, {secs} s",
        path,
        script.get("title").and_then(Value::as_str).unwrap_or(""),
        if gpu { "GPU World" } else { "CPU reference" },
        if opts.surface_contact { "on" } else { "off" }
    );
    let _ = Scene::Flat;
    let (stats, table) = run(&template, &script, gpu, secs)?;
    print!("{table}");
    if let Some(out) = arg("--json") {
        std::fs::write(&out, serde_json::to_string_pretty(&stats)?)?;
        println!("stats: {out}");
    }
    Ok(())
}

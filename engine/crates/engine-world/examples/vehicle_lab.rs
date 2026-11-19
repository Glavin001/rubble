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
//! - `--seed N`, `--steepness D`, `--roughness R`, `--terrain ledges|rolling`: the mountain's.
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
    // A local JBeam export (a directory with `files.json`, as the page's `__mods/…`): the
    // page's generated sedan, with the page's settings (air drag on).
    // A `.truck` file of nodes and beams with no wheels (a node-beam structure: the
    // scenario matrix's truss), as the lab's body.
    if name.ends_with(".truck") {
        return ExplicitBeams::from_truck(&std::fs::read_to_string(name)?, Vec3::ZERO, Vec3::ZERO, settings).map_err(anyhow::Error::msg);
    }
    let dir = std::path::Path::new(name);
    if dir.join("files.json").is_file() {
        let list: Vec<String> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
        let mut texts = Vec::new();
        let mut pc = None;
        for f in &list {
            if f.ends_with(".jbeam") {
                texts.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
            } else if f.ends_with(".pc") {
                pc = Some(std::fs::read_to_string(dir.join(f))?);
            }
        }
        let files: Vec<(&str, &str)> = texts
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        // `--drag 0`: air drag off (the page's `drag=0`).
        let drag = arg("--drag").as_deref() != Some("0");
        let settings = ExplicitBeamsSettings {
            drag,
            turbulence: drag,
            ..settings
        };
        return Ok(rubble::showcase_scene::jbeam_vehicle_on(
            &files,
            pc.as_deref(),
            settings,
            engine_softbody::explicit_beams::JbeamWheels::Raycast,
        )?
        .0);
    }
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
    // `--own-travel 0|1`: anti-clip's padding by each contacter's own travel (on in the lab).
    // `--debris 0|1`, `--debris-ttl S`, `--debris-kg M`, `--debris-nodes N`: the debris time
    // to live (on in the lab: 5 s, 5 kg, 40 nodes).
    if arg("--debris").is_some() || arg("--debris-ttl").is_some() {
        let mut d = json!({ "on": arg("--debris").as_deref() != Some("0") });
        for (flag, key) in [("--debris-ttl", "seconds"), ("--debris-kg", "maxKg"), ("--debris-nodes", "maxNodes")] {
            if let Some(x) = arg(flag).and_then(|x| x.parse::<f64>().ok()) {
                d[key] = json!(x);
            }
        }
        o.insert("debris".into(), d);
    }
    // `--vehicle-sleep 0|1`, `--sleep-speed V` (m/s), `--sleep-reach-mm R`, `--sleep-dwell N`
    // (steps): vehicle sleep and its thresholds (the lab's `vehicleSleep`).
    if arg("--vehicle-sleep").is_some() || arg("--sleep-speed").is_some() || arg("--sleep-reach-mm").is_some() || arg("--sleep-dwell").is_some() {
        let mut d = json!({ "on": arg("--vehicle-sleep").as_deref() != Some("0") });
        for (flag, key) in [("--sleep-speed", "speed"), ("--sleep-reach-mm", "reachMm"), ("--sleep-dwell", "dwell")] {
            if let Some(x) = arg(flag).and_then(|x| x.parse::<f64>().ok()) {
                d[key] = json!(x);
            }
        }
        o.insert("vehicleSleep".into(), d);
    }
    // `--bit-exact 0|1`: bit-exact results (the lab's `bitExact`; off by default,
    // `RUBBLE_BIT_EXACT=1` turns it on: the bit-for-bit harnesses).
    if let Some(s) = arg("--bit-exact") {
        o.insert("bitExact".into(), json!(s == "1"));
    }
    // `--pieces 0|1`: the GPU pair list's piece labels (on in the lab).
    if let Some(s) = arg("--pieces") {
        o.insert("pieceLabels".into(), json!(s == "1"));
    }
    if let Some(s) = arg("--own-travel") {
        o.insert("ownTravel".into(), json!(s == "1"));
    }
    let mut m = o.get("mountain").cloned().unwrap_or_else(|| json!({}));
    for (flag, key) in [
        ("--seed", "seed"),
        ("--steepness", "steepness"),
        ("--roughness", "roughness"),
        ("--loose", "loose"),
        ("--start", "start"),
        ("--launch", "launch"),
    ] {
        if let Some(x) = arg(flag).and_then(|x| x.parse::<f64>().ok()) {
            m[key] = if key == "seed" || key == "loose" {
                json!(x as u64)
            } else {
                json!(x)
            };
        }
    }
    // `--terrain ledges|rolling`: the mountain's ground (the offline default is ledges, the
    // page's rolling).
    if let Some(t) = arg("--terrain") {
        m["terrain"] = json!(t);
    }
    if let Some(r) = arg("--rockslide") {
        m["rockslide"] = json!(r == "1");
    }
    // `--terrain ledges|rolling` (the default without it: ledges, the measurements' scene).
    if let Some(t) = arg("--terrain") {
        m["terrain"] = json!(t);
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
            // `--tri-census N`: every N windows, the self-collision walk's load.
            if let Some(n) = arg("--tri-census").and_then(|x| x.parse::<u64>().ok()) {
                if k % n == n - 1 {
                    let pos: Vec<glam::Vec3> = lab.sim.nodes.iter().map(|n| n.pos).collect();
                    let broken: Vec<bool> = lab.sim.beams.iter().map(|b| b.broken).collect();
                    let vel: Vec<glam::Vec3> = lab.sim.nodes.iter().map(|n| n.vel).collect();
                    println!("  t {:5.1}: {}", lab.bench.t, tri_census(&lab.sim, &pos, &vel, &broken));
                }
            }
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
        "mountain-perf" => mountain_perf(template, gpu)?,
        "rest-ledger" => rest_ledger(template)?,
        "beam-census" => beam_census(template),
        other => {
            anyhow::bail!("unknown --measure {other} (plates, spheres, mountain, mountain-perf)")
        }
    }
    Ok(())
}

/// The self-collision walk's load for the car `s` with its nodes at `pos` and beams
/// `broken`: live collision triangles (none of the beams that kill it broken) by the
/// largest side of their box, how many cover more than the grid's 64 cells (they test
/// every candidate), and the (triangle, candidate) pairs inside the boxes (the kernel's
/// pair list holds 8,192; anti-clip's speed padding left out, so a lower bound).
fn tri_census(s: &ExplicitBeams, pos: &[glam::Vec3], vel: &[glam::Vec3], broken: &[bool]) -> String {
    let (r, cell) = (s.collision_range, s.grid_cell);
    // Anti-clip pads every triangle's box by its own and the car's fastest node's travel.
    let vmax = vel.iter().map(|v| v.length()).fold(0.0f32, f32::max);
    let fast = vel.iter().filter(|v| v.length() > 100.0).count();
    let (mut padded_pairs, mut padded_every, mut own_pairs) = (0u64, 0u32, 0u64);
    let (mut live, mut dead, mut over1, mut over3, mut every_c, mut cells, mut pairs, mut longest) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u64, 0u64, 0.0f32);
    let (mut never, mut big_pairs) = (0u32, 0u64);
    for (t, tri) in s.triangles.iter().enumerate() {
        let kills = s.tri_breaks.get(t).copied().unwrap_or([u32::MAX; 4]);
        if kills.iter().any(|&b| b != u32::MAX && broken.get(b as usize).copied().unwrap_or(false)) {
            dead += 1;
            continue;
        }
        live += 1;
        let p = tri.nodes.map(|i| pos[i as usize]);
        let lo = p[0].min(p[1]).min(p[2]) - glam::Vec3::splat(r);
        let hi = p[0].max(p[1]).max(p[2]) + glam::Vec3::splat(r);
        let ext = (hi - lo).max_element() - 2.0 * r;
        longest = longest.max(ext);
        over1 += (ext > 1.0) as u32;
        over3 += (ext > 3.0) as u32;
        never += (ext > 1.0 && kills.iter().all(|&x| x == u32::MAX)) as u32;
        let fs = ((hi / cell).floor() - (lo / cell).floor() + glam::Vec3::ONE).max(glam::Vec3::ONE);
        let c = fs.x * fs.y * fs.z;
        if c > 64.0 || fs.max_element() > 64.0 {
            every_c += 1;
        } else {
            cells += c as u64;
        }
        let n = s.candidates.iter().filter(|&&h| {
            let q = pos[h as usize];
            !(q.cmplt(lo).any() || q.cmpgt(hi).any())
        }).count() as u64;
        pairs += n;
        if ext > 1.0 {
            big_pairs += n;
        }
        if s.anti_clip {
            let tv = tri.nodes.iter().map(|&i| vel[i as usize].length()).fold(0.0f32, f32::max);
            let pad = glam::Vec3::splat((tv + vmax) * engine_softbody::explicit_beams::STEP + engine_softbody::explicit_contact::CLIP_DEPTH);
            let (lo, hi) = (lo - pad, hi + pad);
            let fs = ((hi / cell).floor() - (lo / cell).floor() + glam::Vec3::ONE).max(glam::Vec3::ONE);
            padded_every += (fs.x * fs.y * fs.z > 64.0 || fs.max_element() > 64.0) as u32;
            padded_pairs += s.candidates.iter().filter(|&&h| {
                let q = pos[h as usize];
                !(q.cmplt(lo).any() || q.cmpgt(hi).any())
            }).count() as u64;
            // Each candidate against the box padded by its own travel, not the fastest node's.
            let (lo0, hi0) = (lo + pad, hi - pad);
            own_pairs += s.candidates.iter().filter(|&&h| {
                let q = pos[h as usize];
                let pad = glam::Vec3::splat((tv + vel[h as usize].length()) * engine_softbody::explicit_beams::STEP + engine_softbody::explicit_contact::CLIP_DEPTH);
                !(q.cmplt(lo0 - pad).any() || q.cmpgt(hi0 + pad).any())
            }).count() as u64;
        }
    }
    format!(
        "anti-clip {}: fastest node {vmax:.0} m/s ({fast} over 100 m/s), padded boxes hold {padded_pairs} pairs ({own_pairs} with each node's own travel), {padded_every} past 64 cells; triangles live {live} dead {dead}; box side > 1 m: {over1}, > 3 m: {over3} (longest {longest:.1} m; {never} that no beam can kill); past 64 cells {every_c}; cells walked {cells}; pairs in boxes {pairs} ({big_pairs} from the > 1 m boxes); candidates {}; broken beams {}",
        s.anti_clip,
        s.candidates.len(),
        broken.iter().filter(|&&b| b).count()
    )
}

/// What the reused pair list's movement bound sees: the tracked nodes (contacters and live
/// collision triangles' nodes) split into the main body (the heaviest piece the unbroken
/// beams leave) and the rest; per group, the per-axis spread of their velocities (m/s) and
/// the steps a 0.05 m list survives at that spread (M / (spread · h)); the detached ones'
/// speeds and whether they carry live triangles.
fn reuse_census(s: &ExplicitBeams, pos: &[glam::Vec3], vel: &[glam::Vec3], broken: &[bool]) -> String {
    let n = s.nodes.len().min(pos.len());
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for (j, b) in s.beams.iter().enumerate() {
        if !broken.get(j).copied().unwrap_or(false) {
            let (x, y) = (find(&mut parent, b.a as usize), find(&mut parent, b.b as usize));
            if x != y {
                parent[x] = y;
            }
        }
    }
    let mut mass = vec![0.0f32; n];
    for i in 0..n {
        let r = find(&mut parent, i);
        mass[r] += s.nodes[i].mass;
    }
    let main = (0..n).fold(0, |b, i| if mass[i] > mass[b] { i } else { b });
    let mut tracked = vec![false; n];
    for &c in &s.candidates {
        tracked[c as usize] = true;
    }
    let mut live_tri_detached = 0u32;
    for (t, tri) in s.triangles.iter().enumerate() {
        let kills = s.tri_breaks.get(t).copied().unwrap_or([u32::MAX; 4]);
        let dead = kills.iter().any(|&b| b != u32::MAX && broken.get(b as usize).copied().unwrap_or(false));
        for &v in &tri.nodes {
            tracked[v as usize] = true;
        }
        if !dead && tri.nodes.iter().any(|&v| find(&mut parent, v as usize) != main) {
            live_tri_detached += 1;
        }
    }
    let spread = |sel: &dyn Fn(usize) -> bool| -> (f32, u32) {
        let (mut lo, mut hi, mut k) = (glam::Vec3::MAX, glam::Vec3::MIN, 0u32);
        for i in (0..n).filter(|&i| tracked[i] && sel(i)) {
            lo = lo.min(vel[i]);
            hi = hi.max(vel[i]);
            k += 1;
        }
        (if k > 0 { (hi - lo).max_element() } else { 0.0 }, k)
    };
    let mut is_main = vec![false; n];
    for i in 0..n {
        is_main[i] = find(&mut parent, i) == main;
    }
    let (all, ka) = spread(&|_| true);
    let (m0, k0) = spread(&|i| is_main[i]);
    let det: Vec<f32> = (0..n).filter(|&i| tracked[i] && !is_main[i]).map(|i| vel[i].length()).collect();
    let pieces = {
        let mut r: Vec<usize> = (0..n).filter(|&i| !is_main[i]).map(|i| find(&mut parent, i)).collect();
        r.sort();
        r.dedup();
        r.len()
    };
    let steps = |sp: f32| if sp > 0.0 { 0.05 / (sp * STEP) } else { f32::INFINITY };
    // The pieces off the main body: (nodes, kg), heaviest first.
    let mut shed: Vec<(u32, f32)> = Vec::new();
    {
        let mut seen = vec![false; n];
        for i in 0..n {
            let r = find(&mut parent, i);
            if r != main && !seen[r] {
                seen[r] = true;
                let k = (0..n).filter(|&j| find(&mut parent, j) == r).count() as u32;
                shed.push((k, mass[r]));
            }
        }
        shed.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    let shed: Vec<String> = shed.iter().map(|(k, m)| format!("{k}n {m:.2}kg")).collect();
    // The pair list's volatile triangles under the lab's labels: live and dead.
    let (vol_live, vol_dead, vol_big) = {
        use engine_softbody::explicit_debris::{piece_labels, Pieces};
        let pieces = Pieces::of(s, broken);
        let labels = piece_labels(s, &pieces, broken);
        let (mut l, mut d, mut big) = (0u32, 0u32, 0u32);
        for (t, tri) in s.triangles.iter().enumerate() {
            let a = labels[tri.nodes[0] as usize];
            if !(a == 7 || tri.nodes.iter().any(|&v| labels[v as usize] != a)) {
                continue;
            }
            let kills = s.tri_breaks.get(t).copied().unwrap_or([u32::MAX; 4]);
            if kills.iter().any(|&b| b != u32::MAX && broken.get(b as usize).copied().unwrap_or(false)) {
                d += 1;
            } else {
                l += 1;
                let p = tri.nodes.map(|i| pos[i as usize]);
                big += ((p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2])).max_element() > 3.0) as u32;
            }
        }
        (l, d, big)
    };
    format!(
        "volatile triangles live {vol_live} ({vol_big} over 3 m) dead {vol_dead}; pieces off [{}]; tracked {ka} (main {k0}); detached nodes {} in {pieces} pieces ({} tracked, fastest {:.0} m/s, {} over 20 m/s), live triangles off the main body {live_tri_detached}; velocity spread all {all:.1} m/s ({:.0} steps a list), main only {m0:.1} m/s ({:.0} steps)",
        shed.join(", "),
        (0..n).filter(|&i| !is_main[i]).count(),
        det.len(),
        det.iter().fold(0.0f32, |a, &b| a.max(b)),
        det.iter().filter(|&&v| v > 20.0).count(),
        steps(all),
        steps(m0)
    )
}

/// What spreads the main body's (label 0) tracked nodes' displacements, the pair list's
/// reuse bound: their velocity spread (the bound's rate), the same after removing the best
/// rigid motion (translation and rotation: what rotation alone costs), the nodes far off
/// the rigid motion, and the spread among nodes within 1 m of each other (what a local
/// bound would see).
fn reuse_cause(s: &ExplicitBeams, pos: &[glam::Vec3], vel: &[glam::Vec3], broken: &[bool]) -> String {
    use engine_softbody::explicit_debris::{piece_labels, Pieces};
    let n = s.nodes.len().min(pos.len());
    let pieces = Pieces::of(s, broken);
    let labels = piece_labels(s, &pieces, broken);
    let mut tracked = vec![false; n];
    for &c in &s.candidates {
        tracked[c as usize] = true;
    }
    for tri in &s.triangles {
        for &v in &tri.nodes {
            tracked[v as usize] = true;
        }
    }
    let m: Vec<usize> = (0..n).filter(|&i| tracked[i] && labels[i] == 0).collect();
    if m.is_empty() {
        return "no main nodes".into();
    }
    let k = m.len() as f32;
    let c = m.iter().map(|&i| pos[i]).sum::<glam::Vec3>() / k;
    let v0 = m.iter().map(|&i| vel[i]).sum::<glam::Vec3>() / k;
    let mut inertia = glam::Mat3::ZERO;
    let mut l = glam::Vec3::ZERO;
    for &i in &m {
        let r = pos[i] - c;
        inertia += glam::Mat3::from_diagonal(glam::Vec3::splat(r.length_squared()))
            - glam::Mat3::from_cols(r * r.x, r * r.y, r * r.z);
        l += r.cross(vel[i] - v0);
    }
    let w = if inertia.determinant().abs() > 1e-6 { inertia.inverse() * l } else { glam::Vec3::ZERO };
    let spread = |f: &dyn Fn(usize) -> glam::Vec3| {
        let (mut lo, mut hi) = (glam::Vec3::MAX, glam::Vec3::MIN);
        for &i in &m {
            lo = lo.min(f(i));
            hi = hi.max(f(i));
        }
        (hi - lo).max_element()
    };
    let all = spread(&|i| vel[i]);
    let rot = spread(&|i| w.cross(pos[i] - c));
    let res = |i: usize| vel[i] - v0 - w.cross(pos[i] - c);
    let resid = spread(&res);
    let mut far: Vec<f32> = m.iter().map(|&i| res(i).abs().max_element()).collect();
    far.sort_by(|a, b| b.total_cmp(a));
    let over = |x: f32| far.iter().filter(|&&f| f > x).count();
    // Without the nodes more than 2 m/s off the rigid motion: the rest's spread.
    let (mut lo, mut hi) = (glam::Vec3::MAX, glam::Vec3::MIN);
    for &i in &m {
        if res(i).abs().max_element() <= 2.0 {
            lo = lo.min(vel[i]);
            hi = hi.max(vel[i]);
        }
    }
    let calm = (hi - lo).max_element();
    let mut local = 0.0f32;
    for (a, &i) in m.iter().enumerate() {
        for &j in &m[a + 1..] {
            if (pos[i] - pos[j]).abs().max_element() < 1.0 {
                local = local.max((vel[i] - vel[j]).abs().max_element());
            }
        }
    }
    let steps = |sp: f32| if sp > 0.0 { 0.1 / (sp * STEP) } else { f32::INFINITY };
    format!(
        "main tracked {}: spread {all:.2} m/s ({:.0} steps at 0.1 m), rotation {:.2} rad/s gives {rot:.2}, off the rigid motion {resid:.2} (nodes over 0.5 / 2 / 10 m/s: {} / {} / {}; top {:.1} {:.1} {:.1}), spread without those over 2 m/s {calm:.2} ({:.0} steps), within 1 m {local:.2} ({:.0} steps)",
        m.len(),
        steps(all),
        w.length(),
        over(0.5),
        over(2.0),
        over(10.0),
        far.first().copied().unwrap_or(0.0),
        far.get(1).copied().unwrap_or(0.0),
        far.get(2).copied().unwrap_or(0.0),
        steps(calm),
        steps(local)
    )
}

/// The page's mountain (`vehicle-lab.html?mode=mountain&…`) for its cost: per second of
/// simulated time, the wall time of a World step (each step waited on: its GPU time and
/// the readback), with the car's speed, node box, coupling use and the surface pair
/// counts. `--steepness 55 --roughness 0.25 --launch 12 --rockslide 1 --surface 1` is
/// the owner's report. Measurement switches (one at a time): `--ablate MASK` from
/// `--ablate-at T` (results meaningless), `--surface-pairs 0`, `--surface-culls 0`.
/// The beam phase's shape on the GPU (`--measure beam-census`, CPU only): beams by kind
/// code, extension records and damping filters, and per 32-beam SIMD group (lane j takes
/// beam j, j + LANES, …) the distinct code paths it runs; the nodes' incidence counts
/// (a node lane's loads in its beam sum).
fn beam_census(template: &ExplicitBeams) {
    let path = |j: usize| {
        let b = &template.beams[j];
        let ext = template.ext_of.get(j).is_some_and(|&x| x != 0);
        (b.kind.code(), ext, b.shock != 0 && b.kind.is_jbeam())
    };
    let mut kinds = std::collections::BTreeMap::<(u32, bool, bool), usize>::new();
    for j in 0..template.beams.len() {
        *kinds.entry(path(j)).or_default() += 1;
    }
    println!("beams {} by (kind code, extension, filter): {kinds:?}", template.beams.len());
    let groups = template.beams.len().div_ceil(32);
    let (mut paths, mut mixed) = (0usize, 0usize);
    for g in 0..groups {
        let mut s: Vec<_> = (g * 32..((g + 1) * 32).min(template.beams.len())).map(path).collect();
        s.sort();
        s.dedup();
        paths += s.len();
        mixed += usize::from(s.len() > 1);
    }
    println!(
        "SIMD groups of 32 beams: {groups}, {:.2} distinct paths each on average, {mixed} mixed",
        paths as f64 / groups as f64
    );
    let mut inc: Vec<usize> = template.incidence.iter().map(Vec::len).collect();
    let total: usize = inc.iter().sum();
    inc.sort_unstable();
    println!(
        "nodes {}: incidence entries {total}, mean {:.1}, median {}, p90 {}, max {}",
        inc.len(),
        total as f64 / inc.len() as f64,
        inc[inc.len() / 2],
        inc[inc.len() * 9 / 10],
        inc[inc.len() - 1]
    );
}

const TRACE_HEADER: &str = "t,cx,cy,cz,s,n_com,n_top,n_low,x_lo,x_hi,ground,speed,wx,wy,wz,w,lid_gap,vx,vy,vz,lx,ly,lz,fleet_top,fleet_x,fleet_vn\n";

/// One `--trace` row for a car: the time; its centre of mass (world); its arc `s` down the
/// fall line and its height `n_com` over the fall line along the slope's normal; its
/// highest and lowest nodes' heights over the line (`n_top`, `n_low`); its nodes' extent
/// across the canyon (`x_lo`, `x_hi`); the drawn ground's distance under the centre
/// along the normal; its centre's speed; its angular velocity about the centre of mass
/// (rad/s, `ω = I⁻¹ L` from every node's mass); `lid_gap`, the gap between its highest
/// node and the drawn walls' top (`WALL_DRAWN_TOP`, 13 m over the line, where the
/// canyon's lid was until 2026-11-17; none on the plateau); then the centre's velocity and the angular momentum `L` about the centre
/// (kg m²/s). In free flight `L` stays (only air drag acts) while `|ω|` may not: a
/// tumbling body hands its spin between axes. Step to step, `Δv/dt − g` and `ΔL/dt` are
/// the contact force per kg and the contact torque: a touch shows in them. (The caller
/// appends the fleet's highest node over the fall line, its widest |x| and its fastest
/// node away from the slope.)
fn trace_row(out: &mut String, lab: &Lab, template: &ExplicitBeams, p: &[Vec3], v: &[Vec3]) {
    use glam::{DMat3, DVec3};
    let spec = lab.bench.opts.mountain;
    let th = f64::from(spec.steepness).to_radians();
    let up = DVec3::new(0.0, th.cos(), th.sin());
    let along = DVec3::new(0.0, -th.sin(), th.cos());
    // The car's body: nodes within 4 m of the nodes' median point (debris broken off
    // and retired nodes parked away are left out).
    let med = |f: fn(&Vec3) -> f32| {
        let mut a: Vec<f32> = p.iter().map(f).collect();
        a.sort_by(f32::total_cmp);
        a[a.len() / 2]
    };
    let mid = Vec3::new(med(|q| q.x), med(|q| q.y), med(|q| q.z));
    let near: Vec<bool> = p.iter().map(|q| q.distance(mid) < 4.0).collect();
    let (mut m, mut c, mut vc) = (0.0f64, DVec3::ZERO, DVec3::ZERO);
    for (((q, w), n), _) in p.iter().zip(v).zip(&template.nodes).zip(&near).filter(|x| *x.1) {
        let mi = f64::from(n.mass);
        m += mi;
        c += q.as_dvec3() * mi;
        vc += w.as_dvec3() * mi;
    }
    c /= m;
    vc /= m;
    let (mut l, mut i) = (DVec3::ZERO, DMat3::ZERO);
    let (mut top, mut low, mut xlo, mut xhi) = (f64::MIN, f64::MAX, f64::MAX, f64::MIN);
    for (((q, w), n), _) in p.iter().zip(v).zip(&template.nodes).zip(&near).filter(|x| *x.1) {
        let mi = f64::from(n.mass);
        let r = q.as_dvec3() - c;
        l += r.cross(w.as_dvec3() - vc) * mi;
        i += (DMat3::IDENTITY * r.length_squared() - DMat3::from_cols(r * r.x, r * r.y, r * r.z)) * mi;
        let h = q.as_dvec3().dot(up);
        top = top.max(h);
        low = low.min(h);
        xlo = xlo.min(f64::from(q.x));
        xhi = xhi.max(f64::from(q.x));
    }
    let w = i.inverse() * l;
    let ground = lab.bench.mountain_ray(c, (-up).as_vec3()).0.map_or(f64::NAN, f64::from);
    let s = c.dot(along);
    let lid = if s > 0.0 { f64::from(rubble::vehicle_lab::mountain::WALL_DRAWN_TOP) - top } else { f64::NAN };
    out.push_str(&format!(
        "{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.4},{:.3},{:.4},{:.4},{:.4},{:.2},{:.2},{:.2}\n",
        lab.bench.t, c.x, c.y, c.z, s, c.dot(up), top, low, xlo, xhi, ground, vc.length(), w.x, w.y, w.z, w.length(), lid,
        vc.x, vc.y, vc.z, l.x, l.y, l.z
    ));
}

/// `--measure rest-ledger` (CPU reference, one car on the mountain; `--steepness`,
/// `--rockslide 0` …): the car's energy ledger by term while it rests. Each 2 kHz step's
/// velocity change is split as `EnergyLedger::account` splits it: the beams' and
/// self-contact's forces (the step's force trace), and the node pass's contacts (ground,
/// the coupling's rock, boxes: what the node pass added to the accumulated force), each
/// worked over the step at the node's mean velocity; the contact position steps (§37)
/// against the held forces. Per window of `--ledger-every` World windows (default 15,
/// 0.25 s) from `--ledger-from` s: kinetic energy, the nodes past 0.25 m/s, each source's
/// net work and the sum of its per-node per-step positive parts (energy a contact put into
/// some node, whatever it took from others), the elastic energy's change and the beams'
/// dissipation. `--rest-ablate friction|anticlip|selfcol|ror` from `--ablate-from` s: every
/// node's friction 0; anti-clip off; self-collision off (no candidates); the RoR contact
/// rules (not dissipative, bit-trick friction).
fn rest_ledger(template: &ExplicitBeams) -> anyhow::Result<()> {
    let secs: f64 = arg("--secs").and_then(|x| x.parse().ok()).unwrap_or(20.0);
    let from: f64 = arg("--ledger-from").and_then(|x| x.parse().ok()).unwrap_or(secs - 3.0);
    let script = json!({ "scene": "mountain", "cars": 1, "events": [] });
    let opts = options(&script);
    println!("rest ledger: CPU, {:?}", opts.mountain);
    let mut lab = CpuLab::new(template, opts)?;
    lab.bench.set_script(&script)?;
    for _ in 0..(from * 60.0).round() as u64 {
        lab.step()?;
    }
    ledger_loop(&mut lab, secs - from)
}

/// A CPU lab with a GPU car's state (`--cpu-ledger V` on `--from-fixture`): the vehicle's
/// node positions and velocities, beams' broken flags and rest lengths, on the CPU
/// reference's mountain (the same options); then [`ledger_loop`]. The GPU keeps more
/// beam state than it reads back (yield records), so this is the GPU car's shape and
/// motion, not its state word for word: evidence for where energy goes, not a parity.
fn cpu_ledger_from(template: &ExplicitBeams, pos: &[glam::Vec3], vel: &[glam::Vec3], beams: &[engine_webgpu::GpuBeam], clip: Option<&[[u32; 4]]>) -> anyhow::Result<()> {
    let secs: f64 = arg("--ledger-secs").and_then(|x| x.parse().ok()).unwrap_or(3.0);
    let script = json!({ "scene": "mountain", "cars": 1, "events": [] });
    let opts = options(&script);
    let mut lab = CpuLab::new(template, opts)?;
    lab.bench.set_script(&script)?;
    let put = |lab: &mut CpuLab| {
        let o = lab.sim.origin;
        for (i, n) in lab.sim.nodes.iter_mut().enumerate() {
            n.pos = pos[i] - o;
            n.vel = vel[i];
        }
        for (b, g) in lab.sim.beams.iter_mut().zip(beams) {
            b.broken = g.broken;
            b.rest = g.rest;
        }
        // `--cpu-clip 1`: anti-clip's side memory too (§36; the GPU's words).
        if let Some(c) = clip.filter(|c| c.len() == lab.sim.nodes.len()) {
            lab.sim.clip_sides = c.to_vec();
        }
    };
    if let Some(c) = clip {
        let e = engine_softbody::explicit_contact::CLIP_EMPTY;
        println!("anti-clip memory: {} nodes, {} holding a side, {} recording one", c.len(), c.iter().filter(|w| w[0] != e || w[1] != e).count(), c.iter().filter(|w| w[2] != e || w[3] != e).count());
    }
    // Twice: the first window streams the mountain's chunks to where the car now is.
    put(&mut lab);
    lab.step()?;
    put(&mut lab);
    println!("cpu ledger from the GPU state: {} nodes, {} broken beams", pos.len(), beams.iter().filter(|b| b.broken).count());
    if std::env::args().any(|a| a == "--gpu-twin") {
        return gpu_twin(&mut lab, secs);
    }
    ledger_loop(&mut lab, secs)
}

/// `--cpu-world`: GPU cars' states (positions, velocities, broken flags, rest lengths,
/// anti-clip memory) as CPU vehicles, each on its own CPU lab's mountain for one window to
/// take that window's rock proxies, then all in one `ExplicitBeamsWorld` with vehicle
/// contact and the energy ledger. Per quarter second (`--ledger-secs`, default 3 s): per
/// car its kinetic energy and nodes past 0.25 m/s, and the ledger's terms over the
/// quarter: vehicle contact's normal and friction work, its position steps, self-contact,
/// the node pass's contacts (ground, rock), the beams' dissipation.
fn cpu_world_from(
    template: &ExplicitBeams,
    cars: &[usize],
    pos: &[Vec<glam::Vec3>],
    vel: &[Vec<glam::Vec3>],
    beams: &[Vec<engine_webgpu::GpuBeam>],
    clip: &[Vec<[u32; 4]>],
) -> anyhow::Result<()> {
    use engine_softbody::explicit_contact::ExplicitBeamsWorld;
    let secs: f64 = arg("--ledger-secs").and_then(|x| x.parse().ok()).unwrap_or(3.0);
    let script = json!({ "scene": "mountain", "cars": 1, "events": [] });
    let mut sims = Vec::new();
    for &v in cars {
        let mut lab = CpuLab::new(template, options(&script))?;
        lab.bench.set_script(&script)?;
        let put = |lab: &mut CpuLab| {
            let o = lab.sim.origin;
            for (i, n) in lab.sim.nodes.iter_mut().enumerate() {
                n.pos = pos[v][i] - o;
                n.vel = vel[v][i];
            }
            for (b, g) in lab.sim.beams.iter_mut().zip(&beams[v]) {
                b.broken = g.broken;
                b.rest = g.rest;
            }
            if clip[v].len() == lab.sim.nodes.len() {
                lab.sim.clip_sides = clip[v].clone();
            }
        };
        put(&mut lab);
        lab.step()?;
        put(&mut lab);
        let mut set = None;
        // (One step's window: its proxies; then the state again.)
        lab.step_observed(&mut |s: &ExplicitBeams| {
            set = s.rigid.as_deref().cloned();
        })?;
        put(&mut lab);
        let mut sim = lab.sim.clone();
        sim.rigid = set.map(Box::new);
        sim.trace = None;
        sims.push(sim);
    }
    let mut w = ExplicitBeamsWorld::new(sims);
    w.final_guard = arg("--final-guard").as_deref() == Some("1");
    w.enable_ledger();
    println!("cpu world of cars {cars:?}: final guard {}", w.final_guard);
    let steps_q = 500u32;
    let quarters = (secs * 4.0).round() as u32;
    let snap = |l: &engine_softbody::explicit_contact::EnergyLedger| [l.w_contact_normal, l.w_contact_friction, l.w_contact_shift, l.w_self_contact, l.w_ground_vertical + l.w_ground_horizontal, l.beam_dissipation(), l.mechanical()];
    let mut last = snap(w.ledger.as_ref().unwrap());
    for q in 0..quarters {
        let mut fast = vec![std::collections::HashSet::new(); cars.len()];
        let mut hits = 0u64;
        for _ in 0..steps_q {
            w.step();
            hits += u64::from(w.last.hits);
            for (k, v) in w.vehicles.iter().enumerate() {
                for (i, n) in v.nodes.iter().enumerate() {
                    if n.vel.abs().max_element() > 0.25 {
                        fast[k].insert(i);
                    }
                }
            }
        }
        let now = snap(w.ledger.as_ref().unwrap());
        let d: Vec<f64> = now.iter().zip(&last).map(|(a, b)| a - b).collect();
        let per: Vec<String> = w
            .vehicles
            .iter()
            .zip(&fast)
            .zip(cars)
            .map(|((v, f), c)| {
                let ke: f64 = v.nodes.iter().map(|n| 0.5 * f64::from(n.mass) * f64::from(n.vel.length_squared())).sum();
                format!("car {c} {ke:.2} J {}n", f.len())
            })
            .collect();
        println!(
            "  +{:.2} s: {}; vehicle contact normal {:+.3} friction {:+.3} shift {:+.3}, self {:+.3}, rock/ground {:+.3}, beams dissipated {:.3}, mechanical {:+.3} J; hits {}",
            f64::from(q + 1) * 0.25,
            per.join(", "),
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
            d[5],
            d[6],
            hits
        );
        last = now;
    }
    Ok(())
}

/// `--gpu-twin`: from the CPU lab's state now, the same vehicle uploaded to a standalone
/// GPU fleet (`ExplicitBeamsGpu::with_coupling`, the window's proxies written as the CPU
/// holds them: the mountain's static rock) and both stepped `secs`, 17 steps a frame.
/// The same starting state word for word; per half second each one's kinetic energy, the
/// nodes past 0.25 m/s at the line, its fastest node, and the largest node distance
/// between the two. `--twin-self-split 1`: the GPU's self-collision split, as the lab.
fn gpu_twin(lab: &mut CpuLab, secs: f64) -> anyhow::Result<()> {
    use engine_softbody::explicit_rigid::ProxySet;
    let mut set: Option<ProxySet> = None;
    lab.step_observed(&mut |s: &ExplicitBeams| {
        set = s.rigid.as_deref().cloned();
    })?;
    let set = set.ok_or_else(|| anyhow::anyhow!("no proxies"))?;
    let mut cpu = lab.sim.clone();
    cpu.rigid = None;
    cpu.trace = None;
    let (_, device, queue) = engine_webgpu::default_device()?;
    let cap = engine_webgpu::CouplingCapacity { bodies: 1024, shapes: 1024, planes: 4096 };
    let mut gpu = engine_webgpu::ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(&cpu), Some(cap));
    engine_webgpu::explicit_coupling::write_proxies(&gpu, &queue, 0, &set).map_err(anyhow::Error::msg)?;
    gpu.set_coupling_on(true);
    if arg("--twin-self-split").as_deref() == Some("1") {
        gpu.set_self_split(true);
    }
    cpu.rigid = Some(Box::new(set.clone()));
    println!("gpu twin: {} proxy shapes, {} bodies; wide {}", set.shapes.len(), set.bodies.len(), gpu.is_wide());
    let stat = |pos: &[glam::Vec3], vel: &[glam::Vec3], m: &[f32]| {
        let ke: f64 = vel.iter().zip(m).map(|(v, m)| 0.5 * f64::from(*m) * f64::from(v.length_squared())).sum();
        let fast = vel.iter().filter(|v| v.abs().max_element() > 0.25).count();
        let vmax = vel.iter().map(|v| v.length()).fold(0.0f32, f32::max);
        let _ = pos;
        (ke, fast, vmax)
    };
    let m: Vec<f32> = cpu.nodes.iter().map(|n| n.mass).collect();
    let frames = (secs * 2000.0 / 17.0).round() as u32;
    let (mut gfast, mut cfast) = (std::collections::HashSet::new(), std::collections::HashSet::new());
    for f in 0..frames {
        gpu.frame(17);
        for _ in 0..17 {
            cpu.step();
        }
        let (gp, gv) = (gpu.positions().remove(0), gpu.velocities().remove(0));
        let cp: Vec<glam::Vec3> = cpu.nodes.iter().map(|n| cpu.origin + n.pos).collect();
        let cv: Vec<glam::Vec3> = cpu.nodes.iter().map(|n| n.vel).collect();
        for (i, v) in gv.iter().enumerate() {
            if v.abs().max_element() > 0.25 {
                gfast.insert(i);
            }
        }
        for (i, v) in cv.iter().enumerate() {
            if v.abs().max_element() > 0.25 {
                cfast.insert(i);
            }
        }
        if (f + 1) % 59 == 0 {
            let (gk, _, gm) = stat(&gp, &gv, &m);
            let (ck, _, cm) = stat(&cp, &cv, &m);
            let d = gp.iter().zip(&cp).map(|(a, b)| (*a - *b).length()).fold(0.0f32, f32::max);
            println!(
                "  +{:.2} s: GPU KE {gk:.3} J, {} nodes past 0.25 m/s, fastest {gm:.2}; CPU KE {ck:.3} J, {}, fastest {cm:.2}; largest node distance {:.2} mm",
                f64::from(f + 1) * 17.0 * STEP as f64,
                gfast.len(),
                cfast.len(),
                d * 1e3
            );
            gfast.clear();
            cfast.clear();
        }
    }
    Ok(())
}

/// The ledger's windows (see [`rest_ledger`]) for `secs` from the lab's state now.
fn ledger_loop(lab: &mut CpuLab, secs: f64) -> anyhow::Result<()> {
    use engine_softbody::explicit_beams::{ForceTrace, STEP};
    let ablate_after: f64 = arg("--ablate-after").and_then(|x| x.parse().ok()).unwrap_or(0.0);
    let every: u64 = arg("--ledger-every").and_then(|x| x.parse().ok()).unwrap_or(15);
    let ablate = arg("--rest-ablate");
    println!("ledger: {secs} s; ablate {ablate:?} after {ablate_after} s");
    lab.sim.trace = Some(Box::default());
    lab.step()?;
    #[derive(Default)]
    struct Acc {
        v0: Vec<glam::Vec3>,
        facc: Vec<glam::Vec3>,
        prev: Option<ForceTrace>,
        st: (f64, f64),
        sc: (f64, f64),
        bm: (f64, f64),
        shift: f64,
        ga: f64,
        fast: std::collections::HashSet<usize>,
        vmax: f32,
    }
    let mut acc = Acc::default();
    let ke = |s: &ExplicitBeams| s.nodes.iter().map(|n| 0.5 * f64::from(n.mass) * f64::from(n.vel.length_squared())).sum::<f64>();
    let pe = |s: &ExplicitBeams| s.nodes.iter().map(|n| f64::from(n.mass) * 9.81 * f64::from(s.origin.y + n.pos.y)).sum::<f64>();
    acc.v0 = lab.sim.nodes.iter().map(|n| n.vel).collect();
    acc.facc = (0..lab.sim.nodes.len()).map(|i| lab.sim.force_now(i)).collect();
    acc.prev = lab.sim.trace.as_deref().cloned();
    let (mut ke0, mut pe0, mut u0) = (ke(&lab.sim), pe(&lab.sim), engine_softbody::explicit_contact::elastic_energy(&lab.sim));
    println!("{:>6} {:>8} {:>5} {:>6} | {:>16} {:>16} {:>16} {:>8} {:>8} | {:>8} {:>8} {:>8}", "t s", "KE J", "fast", "vmax", "rock/ground W(+)", "self W(+)", "beams W(+)", "shift", "grav+air", "dU", "dPE", "beam diss");
    let windows = (secs * 60.0).round() as u64;
    let mut ablated = false;
    for k in 0..windows {
        if !ablated && (k as f64) / 60.0 >= ablate_after {
            ablated = true;
            match ablate.as_deref() {
                Some("friction") => lab.sim.nodes.iter_mut().for_each(|n| n.friction = 0.0),
                Some("anticlip") => lab.sim.anti_clip = false,
                Some("selfcol") => lab.sim.candidates.clear(),
                Some("ror") => {
                    lab.sim.dissipative_contact = false;
                    lab.sim.exact_friction = false;
                }
                Some(other) => anyhow::bail!("--rest-ablate {other}: friction, anticlip, selfcol, ror"),
                None => {}
            }
        }
        lab.step_observed(&mut |s: &ExplicitBeams| {
            let Some(prev) = acc.prev.as_ref() else { return };
            for (i, n) in s.nodes.iter().enumerate() {
                if i >= prev.parts[0].len() || i >= acc.v0.len() {
                    continue;
                }
                let v0 = acc.v0[i];
                let dx = ((n.vel + v0) * (0.5 * STEP)).as_dvec3();
                let w = |f: glam::Vec3| f.as_dvec3().dot(dx);
                let f_tot = (n.vel - v0) * (n.mass / STEP);
                let ground = w(f_tot - acc.facc[i]);
                let (sc, bm) = (w(prev.parts[2][i]), w(prev.parts[1][i]));
                acc.st.0 += ground;
                acc.st.1 += ground.max(0.0);
                acc.sc.0 += sc;
                acc.sc.1 += sc.max(0.0);
                acc.bm.0 += bm;
                acc.bm.1 += bm.max(0.0);
                acc.ga += w(prev.parts[0][i]) + w(prev.parts[3][i]);
                let held = (prev.parts[0][i] + prev.parts[1][i] + prev.parts[3][i]).as_dvec3();
                if let Some(&d) = s.shift_applied.get(i) {
                    acc.shift -= held.dot(d.as_dvec3());
                }
                if let Some(&y) = s.ground_lift.get(i) {
                    acc.shift -= held.y * f64::from(y);
                }
                if n.vel.abs().max_element() > 0.25 {
                    acc.fast.insert(i);
                }
                acc.vmax = acc.vmax.max(n.vel.length());
            }
            acc.v0 = s.nodes.iter().map(|n| n.vel).collect();
            acc.facc = (0..s.nodes.len()).map(|i| s.force_now(i)).collect();
            acc.prev = s.trace.as_deref().cloned();
        })?;
        if (k + 1) % every == 0 {
            let (e, u, p) = (ke(&lab.sim), engine_softbody::explicit_contact::elastic_energy(&lab.sim), pe(&lab.sim));
            let du = u - u0;
            let mut fast: Vec<usize> = acc.fast.iter().copied().collect();
            fast.sort_unstable();
            fast.truncate(8);
            println!(
                "{:>6.2} {:>8.3} {:>5} {:>6.2} | {:>7.3} ({:>6.3}) {:>7.3} ({:>6.3}) {:>7.3} ({:>6.3}) {:>8.3} {:>8.3} | {:>8.3} {:>8.3} {:>8.3}  [dKE {:+.3}] fast {:?}",
                (k + 1) as f64 / 60.0,
                e,
                acc.fast.len(),
                acc.vmax,
                acc.st.0,
                acc.st.1,
                acc.sc.0,
                acc.sc.1,
                acc.bm.0,
                acc.bm.1,
                acc.shift,
                acc.ga,
                du,
                p - pe0,
                -(acc.bm.0 + du),
                e - ke0,
                fast
            );
            (acc.st, acc.sc, acc.bm, acc.shift, acc.ga) = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 0.0, 0.0);
            acc.fast.clear();
            acc.vmax = 0.0;
            (ke0, u0, pe0) = (e, u, p);
        }
    }
    Ok(())
}

fn mountain_perf(template: &ExplicitBeams, gpu: bool) -> anyhow::Result<()> {
    anyhow::ensure!(gpu, "mountain-perf runs in the GPU World (--mode gpu)");
    let secs: f64 = arg("--secs").and_then(|x| x.parse().ok()).unwrap_or(40.0);
    let scene = arg("--scene").unwrap_or_else(|| "mountain".into());
    // `--self-split 1`: the page's default (self-collision across the GPU).
    let script = json!({ "scene": scene, "surface": arg("--surface").as_deref() == Some("1"),
        "selfSplit": arg("--self-split").as_deref() == Some("1"),
        // (The lab's default is the wide passes; `--wide 0` a workgroup a car.)
        "wide": arg("--wide").as_deref() != Some("0"), "wideList": arg("--wide-list").as_deref() != Some("0"), "wideFused": arg("--wide-fused").as_deref() != Some("0"), "groupSplit": arg("--group-split").as_deref() != Some("0"),
        "cars": arg("--cars").and_then(|x| x.parse::<u32>().ok()).unwrap_or(1), "events": [] });
    // `--scenario FILE`: a JSON object whose keys replace the script's (the scenario
    // matrix's rows, engine/tools/scenario-matrix/scenes/: `scene`, `cars`, `parked`,
    // `others`, `events`, `restart` (a start or crash preset, applied after the build as
    // the page does), `structures` with `truckFile` read relative to the file).
    let mut script = script;
    if let Some(file) = arg("--scenario") {
        let text = std::fs::read_to_string(&file)?;
        let v: Value = serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{file}: {e}"))?;
        let dir = std::path::Path::new(&file).parent().map(std::path::Path::to_path_buf).unwrap_or_default();
        for (k, x) in v.as_object().ok_or_else(|| anyhow::anyhow!("{file}: not a JSON object"))? {
            let mut x = x.clone();
            if k == "structures" {
                for s in x.as_array_mut().into_iter().flatten() {
                    if let Some(f) = s.get("truckFile").and_then(Value::as_str) {
                        s["truck"] = json!(std::fs::read_to_string(dir.join(f))?);
                    }
                }
            }
            script[k.as_str()] = x;
        }
    }
    let scene = script["scene"].as_str().unwrap_or("mountain").to_string();
    let opts = options(&script);
    println!(
        "options: {scene}, surface {} {:?}",
        opts.surface_contact, opts.mountain
    );
    println!(
        "features: wings {}, fusedrag {}, centering {}, shocks2 {}, triggers {}, commands {}, slides {}, brakes {}, drive {}, static shapes {}",
        template.wings.len(), template.fusedrag.is_some(), template.centering.len(), template.shocks2.len(), template.triggers.len(),
        template.commands.len(), template.slides.len(), template.brakes.len(), template.drive.is_some(), template.shapes.len()
    );
    let timing = arg("--timing");
    // `--trace FILE`: a CSV row a World step of the first car's rigid motion (see
    // `trace_row`): its spin, its height over the fall line and what it can touch.
    let mut trace = arg("--trace").map(|_| String::from(TRACE_HEADER));
    let mut escaped: Vec<usize> = Vec::new();
    let (mut world, mut lab) = Lab::build(GpuContext::Default, template, opts, |d| {
        d.gpu_timing = match timing.as_deref() {
            Some("phases") => GpuTiming::Phases,
            Some("step") => GpuTiming::Step,
            _ => GpuTiming::Off,
        }
    })?;
    if let Some(r) = script.get("restart") {
        lab.restart(&mut world, r)?;
    }
    lab.read_blocking(&world, true)?;
    lab.bench.set_script(&script)?;
    println!("debris time to live {:?}; notes {:?}", lab.bench.debris.ttl, lab.bench.notes);
    let flag = |k: &str| arg(k).map(|x| x == "1");
    if let Some(x) = world.explicit_vehicles_mut() {
        // `--vv 0`: vehicle–vehicle contact off (a measurement: the cars pass through
        // each other).
        if let Some(on) = flag("--vv") {
            x.set_vehicle_contact(on);
        }
        // `--vv-grid 0`: vehicle contact's candidates without its grid (lossless).
        if let Some(on) = flag("--vv-grid") {
            x.set_vv_grid(on);
        }
        if let Some(on) = flag("--surface-pairs") {
            x.set_surface_pairs(on);
        }
        if let Some(on) = flag("--surface-culls") {
            x.set_surface_culls(on);
        }
        // `--pair-reuse 0|1`, `--pair-margin M` (m): the reused self-collision pair list.
        if let Some(on) = flag("--pair-reuse") {
            x.set_pair_reuse(on);
        }
        if let Some(m) = arg("--pair-margin").and_then(|m| m.parse::<f32>().ok()) {
            x.set_pair_reuse_margin(m);
        }
        if let Some(n) = arg("--pair-streak").and_then(|n| n.parse::<u32>().ok()) {
            x.set_pair_reuse_streak(n);
        }
        // `--wide 1`: the fleet-wide passes (`set_wide`); `--wide-list 1`: with the reused
        // pair list (`set_wide_list`).
        if let Some(on) = flag("--wide") {
            let got = x.set_wide(on);
            println!("wide passes: {got}");
        }
        if let Some(on) = flag("--wide-list") {
            x.set_wide_list(on);
        }
        x.take_pair_reuse_stats();
    }
    let ablate: u32 = arg("--ablate").and_then(|x| x.parse().ok()).unwrap_or(0);
    let ablate_at: f64 = arg("--ablate-at")
        .and_then(|x| x.parse().ok())
        .unwrap_or(0.0);
    let windows = (secs * 60.0).round() as u64;
    let (mut sum, mut worst, mut n) = (0.0f64, 0.0f64, 0u32);
    // `--alternate pairs|cull|ablate=MASK`: that switch on odd windows only, each
    // window's wall time kept by parity (the same run's states, side by side).
    let alternate = arg("--alternate");
    let mut alt = [(0.0f64, 0u32); 2];
    // (t, the step's GPU span, the vehicles' segments, the explicit frame's), ms.
    let mut gpu_steps: Vec<(f64, f64, f64, f64)> = Vec::new();
    // (t, the World step's wall time: `world.step` and its `poll_blocking`, ms.)
    let mut wall_steps: Vec<(f64, f64)> = Vec::new();
    // Vehicle sleep, per car: (the time it first slept, its wakes, asleep now).
    let mut sleep_log: Vec<(Option<f64>, u32, bool)> = Vec::new();
    let t_all = std::time::Instant::now();
    println!(
        "{:>6} {:>8} {:>8} {:>8} {:>7} {:>18} {:>12} {:>16}",
        "t s", "ms/step", "worst", "km/h", "drop m", "box m", "proxies", "surf pairs·hits"
    );
    // `--ab T1,T2,…`: at each time, the car's solver alone from one snapshot, the surface
    // culls on then off (`ExplicitBeamsGpu::set_surface_culls`; the coupling's proxies as
    // the World last wrote them): 20 frames of 34 steps each way, their wall time and
    // whether every state word is the same. `--ab-switch reuse`: the reused pair list
    // (`set_pair_reuse`) on then off instead.
    let reuse_ab = arg("--ab-switch").as_deref() == Some("reuse");
    // `--ab-switch pieces`: the pair list's piece labels (the lab's) against none.
    let pieces_ab = arg("--ab-switch").as_deref() == Some("pieces");
    // `--ab-switch coupling`: the coupling's candidate list and folded step
    // (`set_coupling_candidate_list`) against every node testing every shape in four
    // phases; `--ab-switch fold`: the folded step against the four phases, the list on.
    let coupling_ab = arg("--ab-switch").as_deref() == Some("coupling");
    let fold_ab = arg("--ab-switch").as_deref() == Some("fold");
    // `--ab-switch selfsplit`: self-collision's listed pairs across workgroups
    // (`set_self_split`, selfcol-design.md) against the tests in the car's workgroup.
    let selfsplit_ab = arg("--ab-switch").as_deref() == Some("selfsplit");
    // `--ab-switch beams`: the beam phase on its records sorted by kind against file order.
    let beams_ab = arg("--ab-switch").as_deref() == Some("beams");
    // `--ab-switch kernel --ab-kernel-dir DIR`: this build's explicit kernel (A) against
    // the modules' WGSL in DIR (B: e.g. a copy of an older build's OUT_DIR/wgsl;
    // `engine_webgpu::set_explicit_module_override`). B is a second lab of the same
    // options whose host creates its pipelines from DIR, given A's state at each --ab
    // time (the snapshot's bytes, the car's retired nodes and piece labels): every state
    // word compared bit for bit, and each one's ms a frame (B's pipelines compiled
    // before its timed frames).
    // `--ab-switch perf=MASK`: the kernels' lossless schedule switches
    // (`ExplicitBeamsGpu::set_perf_flags`, `explicit_plan::PERF_*`): the run's flags with
    // MASK's bits on against off.
    let perf_ab: Option<u32> = arg("--ab-switch").and_then(|s| s.strip_prefix("perf=").and_then(|m| m.parse().ok()));
    // `--ab-switch wide`: the fleet-wide passes (`set_wide`) against a workgroup a vehicle.
    let wide_ab = arg("--ab-switch").as_deref() == Some("wide");
    // `--ab-switch widelist`: the wide passes with the reused pair list against their own
    // grid walk every step.
    let widelist_ab = arg("--ab-switch").as_deref() == Some("widelist");
    // `--ab-switch fused`: the wide passes' aux work on the list and hits kernels' lanes
    // against an aux dispatch a step (both wide with the list).
    let fused_ab = arg("--ab-switch").as_deref() == Some("fused");
    let wide_ab = wide_ab || widelist_ab || fused_ab;
    // `--ab-switch groupsplit`: contact groups across workgroups with self-collision
    // split (`set_group_split`, `set_self_split`) against each group on one workgroup.
    let groupsplit_ab = arg("--ab-switch").as_deref() == Some("groupsplit");
    // `--ab-switch sleep`: vehicle sleep on (every vehicle awake at the snapshot, its still
    // count 0) against off: with nobody asleep the physics state must be the same bit for
    // bit (the sleep records, anchors and flags are sleep's own, outside the state words).
    // `--ab-switch sleeplist`: with sleep on (`--vehicle-sleep 1`), the awake-indexed
    // dispatch against every vehicle dispatched (a sleeping one's workgroup returning at
    // once): every state word and every sleep word. With sleep on, `--ab-switch
    // groupsplit` compares the sleep words too (the split paths against a workgroup a
    // group, the reference for sleep's rule).
    let sleep_ab = arg("--ab-switch").as_deref() == Some("sleep");
    let sleeplist_ab = arg("--ab-switch").as_deref() == Some("sleeplist");
    let sleep_cfg = options(&script).vehicle_sleep_cfg;
    // `--ab-switch bitexact`: bit-exact results (`set_bit_exact`) on against off (the
    // strict arithmetic's cost; the states differ in their last bits by design).
    let bitexact_ab = arg("--ab-switch").as_deref() == Some("bitexact");
    let kernel_ab = arg("--ab-switch").as_deref() == Some("kernel");
    let kernel_dir = arg("--ab-kernel-dir").map(std::path::PathBuf::from);
    anyhow::ensure!(!kernel_ab || kernel_dir.is_some(), "--ab-switch kernel needs --ab-kernel-dir DIR");
    let mut other = None;
    let n_nodes = template.nodes.len();
    let switch = |x: &mut engine_webgpu::ExplicitBeamsGpu, on: bool, labels: &[u8]| {
        if bitexact_ab {
            x.set_bit_exact(on);
        } else if groupsplit_ab {
            x.set_group_split(on);
            x.set_self_split(on);
        } else if sleep_ab {
            x.set_sleep_config(sleep_cfg);
            x.set_sleep(on);
        } else if sleeplist_ab {
            x.set_sleep_listing(on);
        } else if fused_ab {
            x.set_wide(true);
            x.set_wide_list(true);
            x.set_wide_fused(on);
        } else if widelist_ab {
            // (Against the wide passes' own walk: both wide.)
            x.set_wide(true);
            x.set_wide_list(on);
        } else if wide_ab {
            x.set_wide(on);
        } else if let Some(mask) = perf_ab {
            let f = x.perf_flags();
            x.set_perf_flags(if on { f | mask } else { f & !mask });
        } else if beams_ab {
            x.set_beam_order(on);
        } else if selfsplit_ab {
            x.set_self_split(on);
        } else if coupling_ab {
            x.set_coupling_candidate_list(on);
        } else if fold_ab {
            x.set_coupling_fold(on);
        } else if pieces_ab {
            let zeros = vec![0u8; n_nodes];
            x.set_piece_labels(0, if on && !labels.is_empty() { labels } else { &zeros });
        } else if reuse_ab {
            x.set_pair_reuse(on);

        } else {
            x.set_surface_culls(on);
        }
    };
    let mut ab: Vec<f64> = arg("--ab")
        .map(|x| x.split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or_default();
    // `--save-fixtures DIR`: at each `--ab` time the state as a file
    // (DIR/cars<N>_t<T>.fx: the snapshot's bytes with every car's piece labels and
    // retired nodes). `--from-fixture FILE`: that state restored in this process and the
    // `--ab` block run on it at once, with no step before it (the same fleet options):
    // a change checked on a saved moment in seconds, not after driving up to it.
    let save_dir = arg("--save-fixtures").map(std::path::PathBuf::from);
    let from_fixture = arg("--from-fixture");
    if let Some(file) = &from_fixture {
        let bytes = std::fs::read(file)?;
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let t = f64::from_bits(u64::from_le_bytes(bytes[0..8].try_into()?));
        let nv = u32_at(8);
        let mut at = 12;
        let x = world.explicit_vehicles_mut().expect("vehicles");
        anyhow::ensure!(nv <= x.vehicle_count(), "{file}: a fixture of {nv} cars, this run has {}", x.vehicle_count());
        for v in 0..nv {
            let n = u32_at(at);
            let labels = &bytes[at + 4..at + 4 + n];
            at += 4 + n;
            let m = u32_at(at);
            let retired: Vec<bool> = bytes[at + 4..at + 4 + m].iter().map(|b| *b != 0).collect();
            at += 4 + m;
            // (As the lab does: with vehicle contact on the engine takes no retired
            // nodes, and the lab's run went on without them.)
            if retired.iter().any(|&r| r) {
                let _ = x.set_retired(v, &retired);
            }
            if !labels.is_empty() {
                x.set_piece_labels(v, labels);
            }
        }
        let snap = x.snapshot_from_bytes(&bytes[at..]);
        x.restore(&snap);
        // (The fixture's records hold the thresholds it was saved with: this run's.)
        x.set_sleep_config(options(&script).vehicle_sleep_cfg);
        x.wait();
        lab.bench.t = t;
        // (AB_PRE=k: k frames of 34 steps with the run's own flags first, the A/B from the
        // state after them: a fixture's first differing frame found, then checked alone.)
        let pre: u32 = std::env::var("AB_PRE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        for _ in 0..pre {
            x.frame(34);
        }
        x.wait();
        lab.bench.t = t + f64::from(pre) / 60.0;
        let t = lab.bench.t;
        ab = vec![t];
        // `--compare-fixture B`: this fixture's state words against B's (two runs' states at
        // one time, e.g. the page's flag-on and flag-off runs saved by vehicle-lab-record.mjs
        // --abSaveAll), the verdict's masking; host state (labels, retired nodes) compared
        // as bytes. Exits.
        if let Some(other) = arg("--compare-fixture") {
            let words = |x: &mut engine_webgpu::ExplicitBeamsGpu| {
                let s = x.stage_state();
                let (tx, rx) = std::sync::mpsc::channel();
                s.buffer().slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                x.wait();
                rx.recv().expect("map").expect("map ok");
                s.ab_words()
            };
            let a = words(x);
            let b_bytes = std::fs::read(&other)?;
            let snap_b = x.snapshot_from_bytes(&b_bytes[at..]);
            x.restore(&snap_b);
            x.wait();
            let b = words(x);
            let tb = f64::from_bits(u64::from_le_bytes(b_bytes[0..8].try_into()?));
            println!("compare {file} (t {t:.3}) with {other} (t {tb:.3}); host state the same: {}", bytes[8..at] == b_bytes[8..at]);
            let d = engine_webgpu::state_differences(&a, &b, 64);
            for l in &d {
                println!("   {l}");
            }
            println!("   state bit for bit: {}", d.is_empty());
            std::process::exit(i32::from(!d.is_empty()));
        }
    }
    for k in 0..windows {
        if ab.first().is_some_and(|&t| lab.bench.t >= t) {
            ab.remove(0);
            if let Some(dir) = &save_dir {
                std::fs::create_dir_all(dir)?;
                let x = world.explicit_vehicles().expect("vehicles");
                let snap = x.snapshot();
                let out = lab.fixture_bytes(&x.snapshot_to_bytes(&snap));
                let file = dir.join(format!("cars{}_t{:.1}.fx", lab.fleet_host_state().len(), lab.bench.t));
                std::fs::write(&file, &out)?;
                println!("   fixture {} ({:.1} MB)", file.display(), out.len() as f64 / 1e6);
            }
            let labels = lab.piece_labels().to_vec();
            let x = world.explicit_vehicles_mut().expect("vehicles");
            x.wait();
            let snap = x.snapshot();
            // `--ab-ablate M1,M2,…`: after the A/B, the same frames under each ablation
            // mask (time only; 2147483648 is the ablation kernel with nothing skipped).
            if let Some(masks) = arg("--ab-ablate") {
                let mut line = Vec::new();
                let mut first_words = None;
                // (`M:M2[:SKIP]`: with the second mask M2, `set_ablation2`, and the self
                // split's walk (1) and pairs (2) dispatches left out, `set_self_split_skip`.)
                for (m, m2, skip) in masks.split(',').filter_map(|m| {
                    let mut p = m.split(':');
                    Some((
                        p.next()?.parse::<u32>().ok()?,
                        p.next().and_then(|x| x.parse::<u32>().ok()).unwrap_or(0),
                        p.next().and_then(|x| x.parse::<u32>().ok()).unwrap_or(0),
                    ))
                }) {
                    x.restore(&snap);
                    x.set_self_split_skip(skip);
                    x.set_ablation2(m2);
                    x.set_ablation(m);
                    x.wait();
                    let t = std::time::Instant::now();
                    for _ in 0..20 {
                        x.frame(34);
                        x.wait();
                    }
                    let ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
                    // (Whether the frames ended in the first mask's state: a mask that
                    // skips nothing the results depend on.)
                    let words = x.state_words();
                    let same = *first_words.get_or_insert_with(|| words.clone()) == words;
                    line.push(format!("{m}:{m2}:{skip}: {ms:.2}{}", if same { " =" } else { "" }));
                }
                x.set_self_split_skip(0);
                x.set_ablation2(0);
                x.set_ablation(0);
                println!(
                    "   ablations at t {:.1} (ms/frame): {}",
                    lab.bench.t,
                    line.join(", ")
                );
            }
            // `--ab-split-ablate M1,M2,…`: the same 20 frames with the split path's
            // measurement word M (time only; the shipped modules, `set_split_ablation`):
            // 0 nothing, 512 no contact pass, and the levels' 16384 residual phases,
            // 32768 buckets and application, 65536 snapshot, 131072 throttle updates.
            if let Some(masks) = arg("--ab-split-ablate") {
                let mut line = Vec::new();
                // (`M:M2`: with the second measurement word M2 as well, `set_ablation2`.)
                for entry in masks.split(',') {
                    let mut parts = entry.split(':').map(|p| p.parse::<u32>().ok());
                    let Some(Some(m)) = parts.next() else { continue };
                    let m2 = parts.next().flatten().unwrap_or(0);
                    x.restore(&snap);
                    x.set_split_ablation(m);
                    x.set_ablation2(m2);
                    x.wait();
                    let t = std::time::Instant::now();
                    for _ in 0..20 {
                        x.frame(34);
                        x.wait();
                    }
                    line.push(format!("{entry}: {:.2}", t.elapsed().as_secs_f64() * 1e3 / 20.0));
                }
                x.set_split_ablation(0);
                x.set_ablation2(0);
                x.restore(&snap);
                println!("   split ablation at t {:.1} (ms a frame): {}", lab.bench.t, line.join(", "));
            }
            // `--ab-perf-masks M1,M2,…`: the same 20 frames with the run's perf flags XOR
            // each mask, three times round the list: per mask the median and the fastest
            // tenth of its 60 frames' wall ms (a frame's wall time rises with any other
            // load on the machine: the low figures are the comparable ones); "=": the
            // frames ended in the first mask's state, bit for bit. A lossless switch's
            // cost or gain, and a kernel's time-only measurement bits when a build has any.
            // Mask bit 31 (2147483648) runs that mask with bit-exact results off.
            if let Some(masks) = arg("--ab-perf-masks") {
                let f = x.perf_flags();
                let bit_exact0 = x.bit_exact();
                let masks: Vec<u32> = masks.split(',').filter_map(|m| m.parse().ok()).collect();
                let mut ms = vec![Vec::new(); masks.len()];
                // (On the fleet-wide passes: each mask's GPU ms a frame by pass, every
                // dispatch a timed pass of its own, 8 frames a round.)
                let mut passes: Vec<Vec<(&str, f64)>> = vec![Vec::new(); masks.len()];
                let mut same = vec![true; masks.len()];
                let mut first_words = None;
                for _ in 0..3 {
                    for (k, m) in masks.iter().enumerate() {
                        x.restore(&snap);
                        // (Mask bit 31: bit-exact results off, the fast setting.)
                        x.set_bit_exact(m & 0x8000_0000 == 0 && bit_exact0);
                        x.set_perf_flags(f ^ (m & 0x7FFF_FFFF));
                        x.wait();
                        for _ in 0..3 {
                            x.frame(34);
                            x.wait();
                        }
                        for _ in 0..20 {
                            let t = std::time::Instant::now();
                            x.frame(34);
                            x.wait();
                            ms[k].push(t.elapsed().as_secs_f64() * 1e3);
                        }
                        let words = x.state_words();
                        same[k] &= *first_words.get_or_insert_with(|| words.clone()) == words;
                        for _ in 0..8 {
                            for (i, (name, t)) in x.profile_wide_frame(34).into_iter().enumerate() {
                                if passes[k].len() <= i {
                                    passes[k].push((name, 0.0));
                                }
                                passes[k][i].1 += t / 24.0;
                            }
                        }
                    }
                }
                println!("   self-collision hits of the last step, per vehicle: {:?}", x.self_hit_counts());
                x.set_bit_exact(bit_exact0);
                x.set_perf_flags(f);
                x.restore(&snap);
                println!(
                    "   perf masks at t {:.1} (flags {f} xor mask: median / fastest tenth, ms a frame of 60): {}",
                    lab.bench.t,
                    masks
                        .iter()
                        .enumerate()
                        .map(|(k, m)| {
                            ms[k].sort_by(f64::total_cmp);
                            format!("{m}: {:.2} / {:.2}{}", ms[k][ms[k].len() / 2], ms[k][ms[k].len() / 10], if same[k] { " =" } else { "" })
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                for (k, m) in masks.iter().enumerate() {
                    let row: Vec<String> = passes[k]
                        .iter()
                        .filter(|p| p.1 >= 0.005 && p.0 != "walk")
                        .map(|(n, t)| format!("{} {t:.3}", n.trim_start_matches("explicit_beams_wide_")))
                        .collect();
                    if !row.is_empty() {
                        println!("     {m}: by pass, GPU ms a frame, timed apart: {}", row.join(", "));
                    }
                }
            }
            // `--sleep-loss "off;V,R,N;V,R,N;…"` (speed m/s, reach mm, dwell steps), with
            // `--loss-frames F` (default 600 frames of 34 steps: 10.2 s): from this
            // snapshot the vehicles' solver alone, F frames under each setting (the
            // coupling's proxies as the World last wrote them: a pile at rest on static
            // ground). Per setting: when each car first sleeps and how often it wakes, the
            // frame's wall time early and late, the fleet's kinetic + potential energy at
            // the end, and each car's largest node displacement from the first setting's
            // end state (mm). The first setting is the reference (give `off` first).
            // `--rest-probe F`: from this snapshot with sleep off, F frames of 34 steps of
            // the solver alone: what the sleep rule would see of each car. Per car its
            // fastest node along an axis at a frame's end (the worst frame; the rule's
            // speed bound looks at every step, so this is a lower bound), how many of its
            // nodes were ever past 0.25 m/s at a frame's end, the node that travelled
            // farthest along an axis over a dwell (30 frames, from each dwell's start)
            // and how many nodes travelled past 0.1 mm in one.
            if let Some(frames) = arg("--rest-probe").and_then(|v| v.parse::<u32>().ok()) {
                x.restore(&snap);
                let was = x.sleep_on();
                x.set_sleep(false);
                x.wait();
                let nv = x.vehicle_count();
                let mut anchor = x.positions();
                let mut fast = vec![(0.0f32, 0usize); nv];
                let mut ever = vec![std::collections::HashSet::<usize>::new(); nv];
                let mut far = vec![(0.0f32, 0usize); nv];
                let mut moved = vec![std::collections::HashSet::<usize>::new(); nv];
                for f in 0..frames {
                    x.frame(34);
                    x.wait();
                    let (pos, vel) = (x.positions(), x.velocities());
                    for v in 0..nv {
                        for (i, w) in vel[v].iter().enumerate() {
                            let u = w.abs().max_element();
                            if u > fast[v].0 || u.is_nan() {
                                fast[v] = (u, i);
                            }
                            if !(u <= 0.25) {
                                ever[v].insert(i);
                            }
                        }
                        for (i, (p, a)) in pos[v].iter().zip(&anchor[v]).enumerate() {
                            let d = (*p - *a).abs().max_element();
                            if d > far[v].0 {
                                far[v] = (d, i);
                            }
                            if !(d <= 1.0e-4) {
                                moved[v].insert(i);
                            }
                        }
                    }
                    if f % 30 == 29 {
                        anchor = pos;
                    }
                }
                println!("   rest probe at t {:.1}, {frames} frames ({} nodes a car):", lab.bench.t, n_nodes);
                for v in 0..nv {
                    println!(
                        "     car {v}: fastest node at a frame's end {:.3} m/s (node {}), {} nodes ever past 0.25 m/s; farthest in a dwell {:.3} mm (node {}), {} nodes past 0.1 mm",
                        fast[v].0,
                        fast[v].1,
                        ever[v].len(),
                        far[v].0 * 1e3,
                        far[v].1,
                        moved[v].len()
                    );
                }
                x.restore(&snap);
                if was {
                    x.set_sleep(true);
                }
                x.wait();
            }
            // `--rest-parts F` (`--rest-every K`, default 30 frames; `--rest-parking P`: every
            // car's parking brake input set to P first): from this snapshot with sleep off,
            // F frames of 34 steps of the solver alone, and per car, every K frames, where
            // its motion is. Its nodes split into pieces over its unbroken beams (the main
            // body: the heaviest piece). The body's motion as a rigid body (its centre of
            // mass's velocity, its angular velocity from its angular momentum and inertia)
            // and, past that, each node's own velocity relative to the rigid motion, for
            // the wheels (tyre, tread, friction-limited and hub nodes) and the rest of the
            // body apart; the loose pieces' speeds. Kinetic energy split the same way, and
            // the potential energy: the fleet's total must not rise (nothing outside the
            // vehicles moves; the ground is static). Speeds as sleep's rule sees them: a
            // node's fastest axis, max over every frame of the window.
            if let Some(frames) = arg("--rest-parts").and_then(|v| v.parse::<u32>().ok()) {
                use engine_softbody::explicit_beams::{NODE_FRICTION_LIMIT, NODE_TREAD, NODE_TYRE};
                x.restore(&snap);
                let was = x.sleep_on();
                x.set_sleep(false);
                let nv = x.vehicle_count();
                if let Some(p) = arg("--rest-parking").and_then(|v| v.parse::<f32>().ok()) {
                    for v in 0..nv {
                        x.set_parking_brake(v, p);
                    }
                    println!("   parking brake {p} on every car ({} brakes a car)", template.brakes.len());
                }
                // `--rest-vv 0`: vehicle contact off for the probe (an ablation: which
                // motion the cars' contact with each other keeps up).
                if arg("--rest-vv").as_deref() == Some("0") {
                    x.set_vehicle_contact(false);
                    println!("   vehicle contact off for the probe");
                }
                // `--rest-coupling 0`: the rigid coupling off (no rock, no ground: the cars
                // fall free), an ablation of the static contact.
                if arg("--rest-coupling").as_deref() == Some("0") {
                    x.set_coupling_on(false);
                    println!("   rigid coupling off for the probe (free fall)");
                }
                // Ablations of the lab's lossless switches (each claimed bit for bit; here
                // only what the resting cars do): `--rest-reuse 0` the reused
                // self-collision pair list, `--rest-candlist 0` the coupling's candidate
                // list, `--rest-fold 0` its folded step, `--rest-groupsplit 0` contact
                // groups on one workgroup, `--rest-selfsplit 0` self-collision in the
                // car's workgroup.
                for (flag, name) in [("--rest-reuse", 0), ("--rest-candlist", 1), ("--rest-fold", 2), ("--rest-groupsplit", 3), ("--rest-selfsplit", 4)] {
                    if arg(flag).as_deref() == Some("0") {
                        match name {
                            0 => x.set_pair_reuse(false),
                            1 => x.set_coupling_candidate_list(false),
                            2 => x.set_coupling_fold(false),
                            3 => x.set_group_split(false),
                            _ => x.set_self_split(false),
                        }
                        println!("   {flag} 0 for the probe");
                    }
                }
                x.wait();
                let every: u32 = arg("--rest-every").and_then(|v| v.parse().ok()).unwrap_or(30).max(1);
                let nn = template.nodes.len();
                let mass: Vec<f64> = template.nodes.iter().map(|n| f64::from(n.mass)).collect();
                let mut wheel = vec![false; nn];
                for (i, n) in template.nodes.iter().enumerate() {
                    wheel[i] = n.flags & (NODE_TYRE | NODE_TREAD | NODE_FRICTION_LIMIT) != 0;
                }
                for b in &template.brakes {
                    for i in b.first..b.first + b.count {
                        wheel[i as usize] = true;
                    }
                    for i in b.axle {
                        wheel[i as usize] = true;
                    }
                }
                for t in &template.tyres {
                    for i in t.first..t.first + t.count {
                        wheel[i as usize] = true;
                    }
                }
                let degree0: Vec<u32> = {
                    let mut d = vec![0u32; nn];
                    for b in &template.beams {
                        d[b.a as usize] += 1;
                        d[b.b as usize] += 1;
                    }
                    d
                };
                let mut fr: Vec<f32> = template.nodes.iter().map(|n| n.friction).collect();
                fr.sort_by(f32::total_cmp);
                println!(
                    "   rest parts at t {:.1}: {frames} frames, a line every {every} ({:.2} s); {} wheel nodes of {nn}; node friction min {:.2} median {:.2} max {:.2}; body node / wheel node / loose: fastest axis m/s",
                    lab.bench.t,
                    f64::from(every) * 34.0 * 0.0005,
                    wheel.iter().filter(|w| **w).count(),
                    fr[0],
                    fr[nn / 2],
                    fr[nn - 1]
                );
                // Per car, per window: |v_com| max, |ω| max, body rel max (node), wheel rel
                // max, loose max, body/wheel/loose node counts past 0.25 m/s.
                #[derive(Clone, Default)]
                struct Win {
                    vc: f64,
                    w: f64,
                    body: (f64, usize),
                    wheel: f64,
                    loose: f64,
                    fast: [std::collections::HashSet<usize>; 3],
                }
                let mut win = vec![Win::default(); nv];
                let mut com0: Vec<glam::DVec3> = Vec::new();
                // (Per car: its nodes at the window's start, its KE + PE at the last line.)
                let mut wpos: Vec<Vec<glam::Vec3>> = x.positions();
                let mut e_car: Vec<Option<f64>> = vec![None; nv];
                let adjacent: Vec<std::collections::HashSet<u32>> = {
                    let mut a = vec![std::collections::HashSet::new(); nn];
                    for b in &template.beams {
                        a[b.a as usize].insert(b.b);
                        a[b.b as usize].insert(b.a);
                    }
                    a
                };
                let reach_self = 2.0 * template.collision_range;
                let mut e_prev: Option<f64> = None;
                for f in 0..frames {
                    x.frame(34);
                    x.wait();
                    let sample = (f + 1) % every == 0;
                    let (pos, vel) = (x.positions(), x.velocities());
                    let beams = x.beams();
                    let mut e_total = 0.0f64;
                    let mut lines = Vec::new();
                    for v in 0..nv {
                        // Pieces over the unbroken beams (union-find).
                        let mut parent: Vec<usize> = (0..nn).collect();
                        fn find(p: &mut [usize], mut i: usize) -> usize {
                            while p[i] != i {
                                p[i] = p[p[i]];
                                i = p[i];
                            }
                            i
                        }
                        let mut degree = vec![0u32; nn];
                        for (b, g) in template.beams.iter().zip(&beams[v]) {
                            if !g.broken {
                                let (ra, rb) = (find(&mut parent, b.a as usize), find(&mut parent, b.b as usize));
                                parent[ra] = rb;
                                degree[b.a as usize] += 1;
                                degree[b.b as usize] += 1;
                            }
                        }
                        let root: Vec<usize> = (0..nn).map(|i| find(&mut parent, i)).collect();
                        let mut pm = std::collections::HashMap::<usize, (f64, usize)>::new();
                        for i in 0..nn {
                            let e = pm.entry(root[i]).or_default();
                            e.0 += mass[i];
                            e.1 += 1;
                        }
                        let main = pm.iter().max_by(|a, b| a.1 .0.total_cmp(&b.1 .0)).map(|(k, _)| *k).unwrap();
                        let p = |i: usize| pos[v][i].as_dvec3();
                        let u = |i: usize| vel[v][i].as_dvec3();
                        let (mut m, mut c, mut vc) = (0.0f64, glam::DVec3::ZERO, glam::DVec3::ZERO);
                        for i in (0..nn).filter(|&i| root[i] == main) {
                            m += mass[i];
                            c += mass[i] * p(i);
                            vc += mass[i] * u(i);
                        }
                        c /= m;
                        vc /= m;
                        let (mut l, mut inertia) = (glam::DVec3::ZERO, glam::DMat3::ZERO);
                        for i in (0..nn).filter(|&i| root[i] == main) {
                            let r = p(i) - c;
                            l += mass[i] * r.cross(u(i) - vc);
                            inertia += glam::DMat3::from_diagonal(glam::DVec3::splat(r.dot(r))) * mass[i]
                                - glam::DMat3::from_cols(r * r.x, r * r.y, r * r.z) * mass[i];
                        }
                        let w = if inertia.determinant().abs() > 1e-12 { inertia.inverse() * l } else { glam::DVec3::ZERO };
                        let (mut ke_def, mut ke_loose, mut pe, mut ke_wheel) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                        let wv = &mut win[v];
                        wv.vc = wv.vc.max(vc.length());
                        wv.w = wv.w.max(w.length());
                        for i in 0..nn {
                            pe += mass[i] * 9.81 * p(i).y;
                            let ax = |q: glam::DVec3| q.abs().max_element();
                            if root[i] == main {
                                let rel = u(i) - vc - w.cross(p(i) - c);
                                let ke = 0.5 * mass[i] * rel.length_squared();
                                if wheel[i] {
                                    ke_wheel += ke;
                                    wv.wheel = wv.wheel.max(ax(rel));
                                    if ax(u(i)) > 0.25 {
                                        wv.fast[1].insert(i);
                                    }
                                } else {
                                    ke_def += ke;
                                    if ax(rel) > wv.body.0 {
                                        wv.body = (ax(rel), i);
                                    }
                                    if ax(u(i)) > 0.25 {
                                        wv.fast[0].insert(i);
                                    }
                                }
                            } else {
                                ke_loose += 0.5 * mass[i] * u(i).length_squared();
                                wv.loose = wv.loose.max(ax(u(i)));
                                if ax(u(i)) > 0.25 {
                                    wv.fast[2].insert(i);
                                }
                            }
                        }
                        let ke_t = 0.5 * m * vc.length_squared();
                        let ke_r = 0.5 * w.dot(l);
                        e_total += ke_t + ke_r + ke_def + ke_wheel + ke_loose + pe;
                        if com0.len() < nv {
                            com0.push(c);
                        }
                        if sample {
                            let loose_n: usize = pm.iter().filter(|(k, _)| **k != main).map(|(_, e)| e.1).sum();
                            let loose_m: f64 = pm.iter().filter(|(k, _)| **k != main).map(|(_, e)| e.0).sum();
                            let bn = wv.body.1;
                            lines.push(format!(
                                "     car {v}: com {:.3} m/s (moved {:.0} mm), ω {:.3} rad/s; rel body {:.3} (node {bn}, beams {}/{}, {}), wheels {:.3}; loose {} pieces {} nodes {:.1} kg {:.3}; nodes past 0.25 body/wheel/loose {}/{}/{}; KE J trans {:.1} rot {:.1} body {:.2} wheels {:.2} loose {:.2}",
                                wv.vc,
                                (c - com0[v]).length() * 1e3,
                                wv.w,
                                wv.body.0,
                                degree[bn],
                                degree0[bn],
                                if wheel[bn] { "wheel" } else { "body" },
                                wv.wheel,
                                pm.len() - 1,
                                loose_n,
                                loose_m,
                                wv.loose,
                                wv.fast[0].len(),
                                wv.fast[1].len(),
                                wv.fast[2].len(),
                                ke_t,
                                ke_r,
                                ke_def,
                                ke_wheel,
                                ke_loose
                            ));
                            // The body's fast nodes now (relative speed past 0.25 m/s) by
                            // what is near them: another car's node within 0.15 m, a node
                            // of its own not beamed to it within twice the collision range
                            // (a self-contact), or neither (the ground, rocks or the beams
                            // alone). And the node that travelled farthest in the window.
                            let (mut near_car, mut near_self, mut alone) = (0u32, 0u32, 0u32);
                            for i in (0..nn).filter(|&i| root[i] == main && !wheel[i]) {
                                let rel = u(i) - vc - w.cross(p(i) - c);
                                if rel.abs().max_element() <= 0.25 {
                                    continue;
                                }
                                let q = pos[v][i];
                                let other = (0..nv).filter(|&o| o != v).any(|o| pos[o].iter().any(|r| r.distance_squared(q) < 0.0225));
                                let own = (0..nn).any(|j| j != i && !adjacent[i].contains(&(j as u32)) && pos[v][j].distance(q) < reach_self);
                                if other {
                                    near_car += 1;
                                } else if own {
                                    near_self += 1;
                                } else {
                                    alone += 1;
                                }
                            }
                            let (far, fi) = (0..nn).map(|i| ((pos[v][i] - wpos[v][i]).abs().max_element(), i)).fold((0.0f32, 0), |a, b| if b.0 > a.0 { b } else { a });
                            let e = ke_t + ke_r + ke_def + ke_wheel + ke_loose + pe;
                            lines.push(format!(
                                "       fast body nodes near another car / a self-contact / neither: {near_car} / {near_self} / {alone}; farthest travel {:.1} mm (node {fi}, beams {}/{}, {}); car KE+PE {:+.2} J since the last line",
                                far * 1e3,
                                degree[fi],
                                degree0[fi],
                                if root[fi] == main { "body" } else { "loose" },
                                e_car[v].map_or(0.0, |x| e - x)
                            ));
                            e_car[v] = Some(e);
                            com0[v] = c;
                            *wv = Win::default();
                        }
                    }
                    if sample {
                        println!(
                            "   +{:.2} s: fleet KE+PE {:.3} kJ ({:+.2} J since the last line); contact groups {:?}",
                            f64::from(f + 1) * 34.0 * 0.0005,
                            e_total * 1e-3,
                            e_prev.map_or(0.0, |e| e_total - e),
                            x.contact_group_sizes()
                        );
                        for l in lines {
                            println!("{l}");
                        }
                        e_prev = Some(e_total);
                        wpos = pos;
                    }
                }
                x.restore(&snap);
                if was {
                    x.set_sleep(true);
                }
                x.wait();
                if from_fixture.is_some() && arg("--sleep-loss").is_none() {
                    std::process::exit(0);
                }
            }
            // `--pair-gaps`: per pair of cars whose boxes (grown 0.1 m) overlap, the
            // nearest any node of one comes to any unbroken triangle's box of the other,
            // both ways (m): how far apart the members of a contact group really are.
            if std::env::args().any(|a| a == "--pair-gaps") {
                x.restore(&snap);
                x.wait();
                let pos = x.positions();
                let beams = x.beams();
                let nv = pos.len();
                let boxes: Vec<(glam::Vec3, glam::Vec3)> = pos.iter().map(|p| p.iter().fold((glam::Vec3::MAX, glam::Vec3::MIN), |(a, b), q| (a.min(*q), b.max(*q)))).collect();
                let tris: Vec<Vec<(glam::Vec3, glam::Vec3)>> = (0..nv)
                    .map(|v| {
                        template
                            .triangles
                            .iter()
                            .enumerate()
                            .filter(|(t, _)| {
                                let kills = template.tri_breaks.get(*t).copied().unwrap_or([u32::MAX; 4]);
                                !kills.iter().any(|&b| b != u32::MAX && beams[v].get(b as usize).is_some_and(|g| g.broken))
                            })
                            .map(|(_, tr)| {
                                let p = tr.nodes.map(|i| pos[v][i as usize]);
                                (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]))
                            })
                            .collect()
                    })
                    .collect();
                let gap = |a: usize, b: usize| {
                    pos[a]
                        .iter()
                        .flat_map(|q| tris[b].iter().map(move |(lo, hi)| (*q - q.clamp(*lo, *hi)).length()))
                        .fold(f32::MAX, f32::min)
                };
                println!("   pair gaps at t {:.1} (m, node to triangle box, each way):", lab.bench.t);
                for a in 0..nv {
                    for b in a + 1..nv {
                        let pad = glam::Vec3::splat(0.1);
                        let (al, ah) = (boxes[a].0 - pad, boxes[a].1 + pad);
                        let (bl, bh) = (boxes[b].0 - pad, boxes[b].1 + pad);
                        if ah.cmplt(bl).any() || bh.cmplt(al).any() {
                            continue;
                        }
                        println!("     cars {a}-{b}: boxes overlap; gaps {:.3} / {:.3}", gap(a, b), gap(b, a));
                    }
                }
                println!("   contact groups {:?}", x.contact_group_sizes());
                if from_fixture.is_some() {
                    std::process::exit(0);
                }
            }
            // `--cpu-world A,B,…`: those vehicles' states (as `--cpu-ledger` carries them)
            // in one CPU reference fleet with vehicle contact (`ExplicitBeamsWorld`, each
            // with its own window's rock proxies) and its energy ledger by term:
            // `cpu_world_from`. `--final-guard 1`: §37.1's work guard on the final
            // velocities for vehicle contact (`ExplicitBeamsWorld::final_guard`).
            if let Some(list) = arg("--cpu-world") {
                x.restore(&snap);
                x.wait();
                let (pos, vel, beams, clip) = (x.positions(), x.velocities(), x.beams(), x.clip_sides());
                let cars: Vec<usize> = list.split(',').filter_map(|v| v.parse().ok()).collect();
                cpu_world_from(template, &cars, &pos, &vel, &beams, &clip)?;
                std::process::exit(0);
            }
            // `--cpu-ledger V`: vehicle V's state on the CPU reference, its energy ledger
            // (`cpu_ledger_from`), then exit.
            if let Some(v) = arg("--cpu-ledger").and_then(|v| v.parse::<usize>().ok()) {
                x.restore(&snap);
                x.wait();
                let (pos, vel, beams) = (x.positions(), x.velocities(), x.beams());
                let clip = x.clip_sides();
                let with_clip = arg("--cpu-clip").as_deref() == Some("1");
                cpu_ledger_from(template, &pos[v], &vel[v], &beams[v], if with_clip { Some(clip[v].as_slice()) } else { None })?;
                std::process::exit(0);
            }
            if let Some(spec) = arg("--sleep-loss") {
                let frames: u32 = arg("--loss-frames").and_then(|v| v.parse().ok()).unwrap_or(600);
                let was = (x.sleep_on(), sleep_cfg);
                let mut reference: Option<Vec<Vec<glam::Vec3>>> = None;
                println!("   sleep loss at t {:.1}, {frames} frames of 34 steps ({:.1} s) each:", lab.bench.t, f64::from(frames) * 34.0 * 0.0005);
                for setting in spec.split(';') {
                    let cfg = (setting != "off").then(|| {
                        let p: Vec<f64> = setting.split(',').filter_map(|v| v.parse().ok()).collect();
                        assert!(p.len() == 3, "--sleep-loss: `off` or `speed,reachMm,dwell`, got {setting}");
                        engine_softbody::explicit_sleep::VehicleSleep { speed: p[0] as f32, reach: (p[1] * 1e-3) as f32, dwell: p[2] as u32 }
                    });
                    x.restore(&snap);
                    x.set_sleep(false);
                    if let Some(c) = cfg {
                        anyhow::ensure!(x.set_sleep_config(c), "--sleep-loss: the fleet has no room for vehicle sleep");
                        x.set_sleep(true);
                    }
                    x.wait();
                    let nv = x.vehicle_count();
                    let (mut first, mut wakes, mut asleep) = (vec![None::<u32>; nv], vec![0u32; nv], vec![false; nv]);
                    let mut ms = Vec::with_capacity(frames as usize);
                    for f in 0..frames {
                        let t = std::time::Instant::now();
                        x.frame(34);
                        x.wait();
                        ms.push(t.elapsed().as_secs_f64() * 1e3);
                        if cfg.is_some() {
                            for (v, (_, a)) in x.sleep_states().into_iter().enumerate() {
                                if a && !asleep[v] {
                                    first[v].get_or_insert(f + 1);
                                }
                                if !a && asleep[v] {
                                    wakes[v] += 1;
                                }
                                asleep[v] = a;
                            }
                        }
                    }
                    let med = |v: &[f64]| {
                        let mut v = v.to_vec();
                        v.sort_by(f64::total_cmp);
                        v[v.len() / 2]
                    };
                    let n = ms.len();
                    let (early, late) = (med(&ms[..n.min(30)]), med(&ms[n - n.min(60)..]));
                    let pos = x.positions();
                    let vel = x.velocities();
                    let mut energy = 0.0f64;
                    for (p, v) in pos.iter().zip(&vel) {
                        for ((q, w), node) in p.iter().zip(v).zip(&template.nodes) {
                            let m = f64::from(node.mass);
                            energy += 0.5 * m * f64::from(w.length_squared()) + m * 9.81 * f64::from(q.y);
                        }
                    }
                    let diff: Vec<String> = match &reference {
                        Some(r) => pos
                            .iter()
                            .zip(r)
                            .map(|(a, b)| format!("{:.3}", a.iter().zip(b).map(|(p, q)| (*p - *q).length()).fold(0.0f32, f32::max) * 1e3))
                            .collect(),
                        None => vec!["reference".into()],
                    };
                    let secs = |f: u32| f64::from(f) * 34.0 * 0.0005;
                    println!(
                        "     {setting:>16}: first asleep after s {}; wakes {:?}; asleep at the end {} of {nv}; ms a frame first 30 / last 60 {early:.2} / {late:.2}; energy {:.3} kJ; largest node difference a car, mm: {}; contact groups {:?}; path split {}",
                        first.iter().map(|f| f.map_or("-".to_string(), |f| format!("{:.2}", secs(f)))).collect::<Vec<_>>().join(" "),
                        wakes,
                        asleep.iter().filter(|a| **a).count(),
                        energy * 1e-3,
                        diff.join(" "),
                        x.contact_group_sizes(),
                        x.split_path_used(),
                    );
                    reference.get_or_insert(pos);
                }
                x.restore(&snap);
                x.set_sleep(false);
                x.set_sleep_config(was.1);
                if was.0 {
                    x.set_sleep(true);
                }
                x.wait();
                if from_fixture.is_some() {
                    std::process::exit(0);
                }
            }
            // `--ab-profile`: the frame as it runs now, by dispatch kind (time only): 10
            // frames untimed (wall), then the same 10 from the same snapshot with every
            // dispatch in its own timed pass (`profile_dispatches`). The table closes: the
            // kinds' sum + the time between passes = the timed span; the span against the
            // untimed frame is what the timing itself costs.
            if std::env::args().any(|a| a == "--ab-profile") {
                const N: u32 = 10;
                let mut plain = 0.0;
                x.restore(&snap);
                // (AB_PROFILE_A2=M2: with the second measurement word, time only.)
                let a2: u32 = std::env::var("AB_PROFILE_A2").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                x.set_ablation2(a2);
                for i in 0..3 + N {
                    let t = std::time::Instant::now();
                    x.frame(34);
                    x.wait();
                    if i >= 3 {
                        plain += t.elapsed().as_secs_f64() * 1e3 / f64::from(N);
                    }
                }
                x.restore(&snap);
                for _ in 0..3 {
                    x.frame(34);
                    x.wait();
                }
                let mut rows: Vec<(&'static str, f64, u32)> = Vec::new();
                let (mut span, mut wall, mut got) = (0.0, 0.0, 0u32);
                for _ in 0..N {
                    let t = std::time::Instant::now();
                    let Some((r, sp)) = x.profile_dispatches(34) else { break };
                    x.wait();
                    wall += t.elapsed().as_secs_f64() * 1e3;
                    span += sp;
                    got += 1;
                    for (label, ms, n) in r {
                        match rows.iter_mut().find(|q| q.0 == label) {
                            Some(q) => {
                                q.1 += ms;
                                q.2 += n;
                            }
                            None => rows.push((label, ms, n)),
                        }
                    }
                }
                for (k, points, per) in x.contact_group_counts() {
                    println!("   group of {k}, {points} points; per member [listed pairs, hits, listed triangles, candidates] (hit list room {}): {per:?}", 2 * points);
                }
                // (AB_LEVELS=n: n frames again from the snapshot, each one's last pass's
                // owners with hits by level: the levels in use and their hits.)
                let levels_frames: u32 = std::env::var("AB_LEVELS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                if levels_frames > 0 {
                    x.restore(&snap);
                    let mut lines = Vec::new();
                    for _ in 0..levels_frames {
                        x.frame(34);
                        x.wait();
                        let lv = x.contact_levels();
                        let top = lv.iter().map(|p| p.1).max().unwrap_or(0) as usize;
                        let mut by = vec![(0u32, 0u32); top];
                        for (h, l) in lv {
                            by[l as usize - 1].0 += 1;
                            by[l as usize - 1].1 += h;
                        }
                        lines.push(format!("{top}:{}", by.iter().map(|(o, h)| format!("{o}o/{h}h")).collect::<Vec<_>>().join(",")));
                    }
                    println!("   levels at t {:.1} (each frame's last pass: top: owners/hits by level): {}", lab.bench.t, lines.join("  "));
                }
                x.set_ablation2(0);
                if got == 0 {
                    println!("   profile at t {:.1}: no timestamps, or too many dispatches", lab.bench.t);
                } else {
                    let f = f64::from(got);
                    let (span, wall) = (span / f, wall / f);
                    let sum: f64 = rows.iter().map(|r| r.1).sum::<f64>() / f;
                    let count: u32 = rows.iter().map(|r| r.2).sum::<u32>() / got;
                    println!("   profile at t {:.1}: GPU ms a frame by dispatch kind ({got} frames, each dispatch a timed pass)", lab.bench.t);
                    println!("     {:<30} {:>9} {:>7} {:>10} {:>7}", "kind", "ms/frame", "count", "us each", "share");
                    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for (label, ms, n) in &rows {
                        let (ms, n) = (ms / f, f64::from(*n) / f);
                        println!("     {label:<30} {ms:>9.3} {n:>7.0} {:>10.1} {:>6.1}%", ms / n * 1e3, 100.0 * ms / span);
                    }
                    println!("     {:<30} {:>9.3} {:>7} {:>10} {:>6.1}%", "between passes", span - sum, "", "", 100.0 * (span - sum) / span);
                    println!("     {:<30} {span:>9.3} {count:>7} {:>10} {:>6.1}%", "= timed span", "", 100.0);
                    println!(
                        "     untimed frame (wall) {plain:.3} ms; timed frame (wall) {wall:.3} ms: the timing adds {:+.3} ms ({:+.1} %)",
                        span - plain,
                        100.0 * (span - plain) / plain
                    );
                }
            }
            // `--ab-wide-profile`: 10 frames on the fleet-wide passes, each dispatch in its
            // own timed pass (`profile_wide_frame`): GPU ms a frame per pass kind (time
            // only; needs `--timing step`).
            if std::env::args().any(|a| a == "--ab-wide-profile") {
                x.restore(&snap);
                let was = x.is_wide();
                x.set_wide(true);
                for _ in 0..3 {
                    x.frame(34);
                }
                x.wait();
                let mut sums: Vec<(&str, f64)> = Vec::new();
                for _ in 0..10 {
                    for (k, (name, ms)) in x.profile_wide_frame(34).into_iter().enumerate() {
                        if sums.len() <= k {
                            sums.push((name, 0.0));
                        }
                        sums[k].1 += ms / 10.0;
                    }
                }
                x.set_wide(was);
                println!(
                    "   wide profile at t {:.1} (GPU ms a frame, each dispatch timed apart): {}; sum {:.2}",
                    lab.bench.t,
                    sums.iter().filter(|x| x.1 > 0.0).map(|(n, ms)| format!("{} {ms:.2}", n.trim_start_matches("explicit_beams_wide_"))).collect::<Vec<_>>().join(", "),
                    sums.iter().map(|x| x.1).sum::<f64>()
                );
            }
            // `--ab-profile`: the same 20 frames with self-collision split, each of its
            // dispatches in its own timed pass (`profile_self_split_frame`): GPU ms a frame
            // per dispatch kind and the span, beside the frames' wall time (time only).
            if std::env::args().any(|a| a == "--ab-profile") {
                x.restore(&snap);
                x.set_self_split(true);
                // (AB_PROFILE_A2=M2: with the second measurement word, time only.)
                x.set_ablation2(std::env::var("AB_PROFILE_A2").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
                x.wait();
                let mut sums: Vec<(&str, f64, u32)> = Vec::new();
                let t = std::time::Instant::now();
                let mut frames = 0u32;
                for _ in 0..20 {
                    let Some(p) = x.profile_self_split_frame(34) else { break };
                    frames += 1;
                    for (k, (name, ms, n)) in p.into_iter().enumerate() {
                        if sums.len() <= k {
                            sums.push((name, 0.0, 0));
                        }
                        sums[k].1 += ms;
                        sums[k].2 += n;
                    }
                }
                let wall = t.elapsed().as_secs_f64() * 1e3 / f64::from(frames.max(1));
                x.set_ablation2(0);
                let f = f64::from(frames.max(1));
                println!(
                    "   self split profile at t {:.1} ({frames} frames, ms/frame, dispatches/frame): {}; wall {wall:.2}",
                    lab.bench.t,
                    sums.iter()
                        .map(|(n, ms, c)| format!("{n} {:.3} ×{:.0}", ms / f, f64::from(*c) / f))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            // `--ab-skip`: the same 20 frames with self-collision split, its walk and/or
            // pairs dispatches left out (time only: 0 all, 1 no walks, 2 no pairs, 3 neither).
            if std::env::args().any(|a| a == "--ab-skip") {
                let mut line = Vec::new();
                for (k, m) in [0u32, 1, 2, 3, 0].into_iter().enumerate() {
                    x.restore(&snap);
                    x.set_self_split(true);
                    x.set_self_split_skip(m);
                    x.wait();
                    x.take_pair_reuse_stats();
                    let t = std::time::Instant::now();
                    for _ in 0..20 {
                        x.frame(34);
                        x.wait();
                    }
                    line.push(format!("{m}: {:.2}", t.elapsed().as_secs_f64() * 1e3 / 20.0));
                    let st = x.take_pair_reuse_stats();
                    if k == 0 {
                        line.push(format!(
                            "(builds {:.1}/frame, reused steps {:.1}/frame, pairs a build {:.0})",
                            st[0] as f64 / 20.0,
                            st[1] as f64 / 20.0,
                            st[2] as f64 / st[0].max(1) as f64
                        ));
                    }
                }
                x.set_self_split_skip(0);
                println!("   self split skips at t {:.1} (ms/frame): {}", lab.bench.t, line.join(", "));
            }
            let mut out = Vec::new();
            // (Each side's world positions, broken beams, nodes' kinetic energy: the
            // tolerance report below, for a switch that is not bit for bit by design.)
            let mut physical: Vec<(Vec<Vec<glam::Vec3>>, usize, f64)> = Vec::new();
            if kernel_ab {
                let timed = |y: &mut engine_webgpu::ExplicitBeamsGpu, snap: &engine_webgpu::GpuSnapshot| {
                    // Two untimed frames first (a lazily created pipeline compiles there).
                    for timed in [false, true] {
                        y.restore(snap);
                        y.wait();
                        let t = std::time::Instant::now();
                        for _ in 0..if timed { 20 } else { 2 } {
                            y.frame(34);
                            y.wait();
                        }
                        if timed {
                            return t.elapsed().as_secs_f64() * 1e3 / 20.0;
                        }
                    }
                    unreachable!()
                };
                let ms = timed(x, &snap);
                out.push((ms, x.state_words(), x.coupling_words()));
                let bytes = x.snapshot_to_bytes(&snap);
                let retired = lab.bench.debris.retired.clone();
                engine_webgpu::set_explicit_module_override(kernel_dir.clone());
                if other.is_none() {
                    other = Some(Lab::build(GpuContext::Default, template, options(&script), |_| {})?);
                }
                let (wb, _) = other.as_mut().expect("lab B");
                let y = wb.explicit_vehicles_mut().expect("vehicles B");
                if retired.iter().any(|&r| r) {
                    y.set_retired(0, &retired).map_err(anyhow::Error::msg)?;
                }
                let zeros = vec![0u8; n_nodes];
                y.set_piece_labels(0, if labels.is_empty() { &zeros } else { &labels });
                let snap_b = y.snapshot_from_bytes(&bytes);
                let ms = timed(y, &snap_b);
                out.push((ms, y.state_words(), y.coupling_words()));
                engine_webgpu::set_explicit_module_override(None);
            }
            // (AB_WARM_MS=n: frames from the snapshot for n ms before the timed halves, so a
            // GPU still at its idle clock after the process's start does not slow the
            // first half alone: a small scene's 20 ms of frames is over before it ramps.)
            let warm: f64 = std::env::var("AB_WARM_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            if warm > 0.0 && !kernel_ab {
                let t = std::time::Instant::now();
                while t.elapsed().as_secs_f64() * 1e3 < warm {
                    x.restore(&snap);
                    for _ in 0..4 {
                        x.frame(34);
                    }
                    x.wait();
                }
                x.restore(&snap);
                x.wait();
            }
            // `--ab-same`: both runs with the culls on (the harness's own repeatability).
            let second = std::env::args().any(|a| a == "--ab-same");
            // (AB_ROUNDS=r: the two halves r times in turn, on, off, on, off, ..., each from
            // the snapshot, every frame's time pooled by half: a drift of the GPU's clock
            // or a neighbour's load falls on both halves. The state compared is the last
            // round's.)
            let rounds: u32 = std::env::var("AB_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
            let sides = if kernel_ab { vec![] } else { vec![true, second] };
            let mut pooled: Vec<Vec<Vec<f64>>> = vec![Vec::new(); sides.len()];
            for (round, (side, on)) in (0..rounds).flat_map(|r| sides.clone().into_iter().enumerate().map(move |s| (r, s))) {
                x.restore(&snap);
                switch(x, on, &labels);
                x.wait();
                // (One untimed frame first: a pipeline this path creates on first use
                // compiles there, not in the timing. Then the state again.)
                x.frame(34);
                x.wait();
                x.restore(&snap);
                if sleeplist_ab {
                    // (The listing's switch asks for a new contact plan, which the
                    // restored snapshot's planner state forgot.)
                    switch(x, on, &labels);
                }
                x.wait();
                // (AB_GAP_MS=n: the GPU idle n ms before each frame, as a display's frame
                // loop leaves it; the frames' own time alone is summed.)
                let gap: u64 = std::env::var("AB_GAP_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                // (`--ab-frames N`: the frames each way, 20 unless given.)
                let frames: u32 = arg("--ab-frames").and_then(|v| v.parse().ok()).unwrap_or(20).max(1);
                let mut sum = 0.0;
                let mut each: Vec<f64> = Vec::new();
                // (AB_STEPS=s: frames of s steps in place of 34, to find the first step two
                // paths differ at; AB_COUNTS=1: each path's contact groups' counts after
                // its frames.)
                let env_u = |k: &str, d: u32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
                let ab_steps = env_u("AB_STEPS", 34);
                for _ in 0..frames {
                    if gap > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(gap));
                    }
                    let t = std::time::Instant::now();
                    x.frame(ab_steps);
                    x.wait();
                    let dt = t.elapsed().as_secs_f64();
                    sum += dt;
                    each.push(dt * 1e3);
                    // (AB_EACH=1: each frame's wall ms and, with the kept candidates, its
                    // builds, reused passes and passes on the front's own path; the
                    // summary's counts are then of the last frame alone.)
                    if env_u("AB_EACH", 0) != 0 {
                        let fk = x.take_fleet_keep_stats();
                        println!("     frame ({}): {:.2} ms, builds {} (calm {}, between {}, cover {}), reused {}, own path {}", if on { "on" } else { "off" }, dt * 1e3, fk[0], fk[5], fk[9], fk[6], fk[1], fk[3]);
                    }
                }
                let _ = sum;
                pooled[side].push(each);
                if round + 1 < rounds {
                    continue;
                }
                let runs = std::mem::take(&mut pooled[side]);
                // Frame k is the same work in every round (each starts from the snapshot):
                // its fastest round is its time without the noise, and their mean the
                // half's time with the frames' own mix kept (a crash's frames differ).
                let best = (0..runs[0].len()).map(|k| runs.iter().map(|r| r[k]).fold(f64::MAX, f64::min)).sum::<f64>() / runs[0].len() as f64;
                println!("     frames ({}): best of {rounds} rounds a frame, mean {best:.3} ms", if on { "on" } else { "off" });
                let mut each: Vec<f64> = runs.concat();
                let ms = each.iter().sum::<f64>() / each.len() as f64;
                let frames = each.len();
                // Each frame's own time: the median, the fastest and the slowest (the
                // scenario matrix's numbers).
                each.sort_by(f64::total_cmp);
                println!(
                    "     frames ({}): median {:.3} ms, min {:.3}, max {:.3} ({frames} frames)",
                    if on { "on" } else { "off" },
                    each[each.len() / 2],
                    each[0],
                    each[each.len() - 1]
                );
                // Vehicle contact's kept candidates (PERF_FLEET_KEEP), when this run used
                // them (counted from the snapshot's own counts: a fixture's are zero).
                let fk = x.take_fleet_keep_stats();
                if fk[0] + fk[1] > 0 {
                    println!(
                        "     kept candidates ({}): {} builds ({} for a calm node, {} for two vehicles, {} for cover), {} passes reused ({:.1} %), {} passes on the front's own path; {:.0} pairs a build, most an owner {}; largest bound at a build {:.3} m, allowance between vehicles {:.3} m",
                        if on { "on" } else { "off" },
                        fk[0],
                        fk[5],
                        fk[9],
                        fk[6],
                        fk[1],
                        100.0 * fk[1] as f64 / (fk[0] + fk[1] + fk[3]).max(1) as f64,
                        fk[3],
                        fk[2] as f64 / fk[0].max(1) as f64,
                        fk[4],
                        f32::from_bits(fk[7] as u32),
                        f32::from_bits(fk[8] as u32)
                    );
                }
                if env_u("AB_COUNTS", 0) != 0 {
                    println!("     counts ({}): {:?}", if on { "on" } else { "off" }, x.contact_group_counts());
                }
                // (FK_MOTION=1: the kept candidates' reference motions after the frames.)
                if on && env_u("FK_MOTION", 0) != 0 {
                    for (v, m) in x.fleet_keep_motion().iter().enumerate() {
                        println!("     kept motion, car {v}: reference {:?}, mean {:?}, origin change {:?}", &m[0..3], &m[3..6], &m[6..9]);
                    }
                }
                // (Vehicle sleep's own words, before the switch is put back: the records,
                // the anchors. Not for the sleep switch itself, which they are.)
                let sleep_words = if sleep_ab { Vec::new() } else { x.sleep_words() };
                if std::env::var("AB_SLEEP_STATES").is_ok() {
                    println!("     sleep ({}): {:?}, split path used {}", if on { "on" } else { "off" }, x.sleep_states(), x.self_split_used());
                }
                // The switch's own header bit back, so the words compare the state alone.
                switch(x, true, &labels);
                x.wait();
                let mut extra = x.coupling_words();
                extra.extend(sleep_words);
                out.push((ms, x.state_words(), extra));
                let broken = x.beams().iter().flatten().filter(|b| b.broken).count();
                let masses = x.masses();
                let ke: f64 = x
                    .velocities()
                    .iter()
                    .zip(&masses)
                    .flat_map(|(v, m)| v.iter().zip(m).map(|(v, m)| 0.5 * f64::from(*m) * f64::from(v.length_squared())))
                    .sum();
                physical.push((x.positions(), broken, ke));
            }
            // The vehicle-contact plan's records are planning state, as in
            // tests/explicit_vehicles.rs: a vehicle's "big" flag is whichever vehicle
            // overflowed a broadphase bucket first (the same pairs either way; it differs
            // between two runs of one path, --ab-same). Reported below, outside the verdict.
            let plan = x.vv_record_words();
            let masked = |w: &[(&'static str, Vec<u32>)]| -> Vec<Vec<u32>> {
                w.iter()
                    .map(|(name, v)| {
                        let mut v = v.clone();
                        if *name == "misc" {
                            v[plan.clone()].fill(0);
                        }
                        v
                    })
                    .collect()
            };
            let same = masked(&out[0].1) == masked(&out[1].1) && out[0].2 == out[1].2;
            for ((name, a), (_, b)) in out[0]
                .1
                .iter()
                .zip(&out[1].1)
                .chain(out[0].2.iter().zip(&out[1].2))
            {
                let diff = a.iter().zip(b).filter(|(p, q)| p != q).count();
                if diff > 0 || a.len() != b.len() {
                    let at: Vec<String> = a
                        .iter()
                        .zip(b)
                        .enumerate()
                        .filter(|(_, (p, q))| p != q)
                        .take(4)
                        .map(|(i, (p, q))| format!("[{i}] {p:#x} / {q:#x}"))
                        .collect();
                    println!(
                        "     {name}: {diff} of {} words differ: {}",
                        a.len(),
                        at.join(", ")
                    );
                    if *name == "misc" {
                        // Which planning record a differing word is (6 records of 4 words a vehicle).
                        let r = x.vv_record_words();
                        let inside = a.iter().zip(b).enumerate().filter(|(i, (p, q))| p != q && r.contains(i)).count();
                        println!("       (vehicle-contact planning records, words {}..{}, 24 a vehicle: {inside} of the {diff}; outside the verdict)", r.start, r.end);
                    }
                }
            }
            println!(
                "   A/B at t {:.1}: {} on {:.2} ms/frame, off {:.2} ms/frame; state bit for bit: {same}",
                lab.bench.t,
                if kernel_ab {
                    "this kernel (on) / --ab-kernel-dir's (off)"
                } else if bitexact_ab {
                    "bit-exact (on) / fast (off)"
                } else if groupsplit_ab {
                    "group split"
                } else if sleep_ab {
                    "vehicle sleep"
                } else if sleeplist_ab {
                    "sleep listing"
                } else if wide_ab {
                    "wide passes"
                } else if perf_ab.is_some() {
                    "perf flags"
                } else if beams_ab {
                    "sorted beams"
                } else if selfsplit_ab {
                    "self split"
                } else if coupling_ab {
                    "coupling list"
                } else if fold_ab {
                    "coupling fold"
                } else if pieces_ab {
                    "piece labels"
                } else if reuse_ab {
                    "pair reuse"
                } else {
                    "surface culls"
                },
                out[0].0,
                out[1].0
            );
            // (Wide against the workgroup frame: the two keep different scratch in the
            // accumulators and misc records; the nodes, beams and groups are the state.)
            // (And vehicle sleep's own words, when it is on: the records and anchors.)
            let physics = out[0].1.iter().zip(&out[1].1).filter(|((n, _), _)| ["nodes", "beams", "groups"].contains(n)).all(|((_, a), (_, b))| a == b)
                && out[0].2.iter().zip(&out[1].2).filter(|((n, _), _)| n.starts_with("sleep")).all(|((_, a), (_, b))| a == b);
            if wide_ab {
                println!("   wide: nodes, beams and groups bit for bit: {physics}");
            }
            // The two sides' physical difference (for a switch not bit for bit by design,
            // as bitexact): the largest node distance per car, the broken beams and the
            // nodes' kinetic energy each side, and whether every position is finite.
            if physical.len() == 2 {
                let (pa, ba, ka) = &physical[0];
                let (pb, bb, kb) = &physical[1];
                let worst: Vec<f32> = pa
                    .iter()
                    .zip(pb)
                    .map(|(a, b)| a.iter().zip(b).map(|(p, q)| (*p - *q).length()).fold(0.0f32, f32::max))
                    .collect();
                let finite = pa.iter().chain(pb).flatten().all(|p| p.is_finite());
                println!(
                    "   A/B physical: largest node difference {:.3} mm (cars: {}); broken beams on {ba} / off {bb}; kinetic energy on {:.1} / off {:.1} J; finite {finite}",
                    worst.iter().fold(0.0f32, |m, w| m.max(*w)) * 1e3,
                    worst.iter().map(|w| format!("{:.2}", w * 1e3)).collect::<Vec<_>>().join(" "),
                    ka,
                    kb
                );
            }
            // (AB_TIME_ONLY=1: a measurement switch that changes the results on purpose.)
            // (bitexact: not bit for bit by design; the physical line above is its verdict.)
            anyhow::ensure!(same || bitexact_ab || (wide_ab && physics) || std::env::var("AB_TIME_ONLY").is_ok(), "the A/B switch changed the state");
            x.restore(&snap);
            switch(x, true, &labels);
            x.wait();
            if from_fixture.is_some() {
                // (A replayed fixture: its check is the whole run.)
                std::process::exit(0);
            }
        }
        if ablate != 0 && lab.bench.t >= ablate_at {
            if let Some(x) = world.explicit_vehicles_mut() {
                x.set_ablation(ablate);
            }
        }
        let odd = k % 2 == 1;
        if let (Some(a), Some(x)) = (alternate.as_deref(), world.explicit_vehicles_mut()) {
            match a {
                "pairs" => x.set_surface_pairs(!odd),
                "cull" => x.set_coupling_cull(!odd),
                "culls" => x.set_surface_culls(!odd),
                m if m.starts_with("ablate=") => {
                    x.set_ablation(if odd { m[7..].parse()? } else { 0 })
                }
                other => anyhow::bail!("unknown --alternate {other}"),
            }
        }
        lab.before_step(&mut world)?;
        let t = std::time::Instant::now();
        world.step()?;
        let frames = world.poll_blocking()?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        wall_steps.push((lab.bench.t, ms));
        // Vehicle sleep: each car's first sleep, its wakes, and who sleeps now.
        for e in frames.iter().flat_map(|f| f.explicit_sleep.iter()) {
            let v = e.vehicle as usize;
            if sleep_log.len() <= v {
                sleep_log.resize(v + 1, (None, 0, false));
            }
            if e.asleep {
                sleep_log[v].0.get_or_insert(lab.bench.t);
            } else {
                sleep_log[v].1 += 1;
            }
            sleep_log[v].2 = e.asleep;
        }
        alt[usize::from(odd)].0 += ms;
        alt[usize::from(odd)].1 += 1;
        // With `--timing step|phases`: this step's GPU time, the vehicles' part of it (the
        // explicit frame's dispatches and the coupling's passes) and the frame alone.
        if let Some(g) = world.last_step_gpu_time() {
            let veh: f64 = g.segments.iter().filter(|(l, _)| l.starts_with("explicit") || l.starts_with("cpl_")).map(|x| x.1).sum();
            let frame: f64 = g.segments.iter().filter(|(l, _)| l.starts_with("explicit")).map(|x| x.1).sum();
            gpu_steps.push((lab.bench.t, g.span_ms, veh, frame));
        }
        lab.read_blocking(&world, k % 6 == 5)?;
        if let (Some(tr), Some(x)) = (trace.as_mut(), world.explicit_vehicles()) {
            let (ps, vs) = (x.positions(), x.velocities());
            trace_row(tr, &lab, template, &ps[0], &vs[0]);
            // The fleet's every node (debris included): the highest over the fall line,
            // the farthest across, and the fastest away from the slope.
            let th = f64::from(lab.bench.opts.mountain.steepness).to_radians();
            let up = glam::DVec3::new(0.0, th.cos(), th.sin());
            let (mut top, mut wide, mut vn) = (f64::MIN, 0.0f64, f64::MIN);
            for (p, v) in ps.iter().flatten().zip(vs.iter().flatten()) {
                top = top.max(p.as_dvec3().dot(up));
                wide = wide.max(f64::from(p.x.abs()));
                vn = vn.max(v.as_dvec3().dot(up));
            }
            // (`--watch-from T`: from T s, each car's fastest node every step.)
            if arg("--watch-from").and_then(|x| x.parse::<f64>().ok()).is_some_and(|t| lab.bench.t >= t) {
                let w: Vec<String> = vs
                    .iter()
                    .zip(&ps)
                    .map(|(v, p)| {
                        let (i, s) = v.iter().enumerate().fold((0, 0.0f32), |b, (i, q)| if q.length() > b.1 { (i, q.length()) } else { b });
                        format!("{i}:{s:.0}@{:.0},{:.0},{:.0}", p[i].x, p[i].y, p[i].z)
                    })
                    .collect();
                println!("   watch t {:.4}: {}", lab.bench.t, w.join(" "));
            }
            // (A node past a wall's inner face: where, the first time each car has one.)
            for (car, (p, v)) in ps.iter().zip(&vs).enumerate() {
                if escaped.contains(&car) {
                    continue;
                }
                // (Or under the ground: deeper than the slabs under the fall line.)
                if let Some(i) = p.iter().position(|q| q.x.abs() > 18.5 || q.as_dvec3().dot(up) < -6.0) {
                    escaped.push(car);
                    let along = glam::DVec3::new(0.0, -th.sin(), th.cos());
                    let q = p[i].as_dvec3();
                    println!(
                        "   escape: t {:.3} car {car} node {i} at x {:.2}, {:.2} m over the line, {:.1} m along; v {:?}; its car's median node x {:.1}, {:.1} m along",
                        lab.bench.t, q.x, q.dot(up), q.dot(along), v[i], p[p.len() / 2].x, p[p.len() / 2].as_dvec3().dot(along)
                    );
                }
            }
            tr.pop();
            tr.push_str(&format!(",{top:.3},{wide:.3},{vn:.3}\n"));
            // (Written every second as well: a run that stops early keeps its trace.)
            if k % 60 == 59 {
                if let Some(path) = arg("--trace") {
                    std::fs::write(path, tr.as_str())?;
                }
            }
        }
        if lab.bench.car.non_finite {
            // (The coupling's status is read every few windows: a room it overflowed
            // shows a few steps on, as the step's error.)
            for _ in 0..16 {
                if let Err(e) = world.step().and_then(|_| world.poll_blocking()) {
                    println!("   after the non-finite state: {e}");
                    break;
                }
            }
        }
        anyhow::ensure!(
            !lab.bench.car.non_finite,
            "non-finite node state at t = {:.3} s",
            lab.bench.t
        );
        sum += ms;
        worst = worst.max(ms);
        n += 1;
        if (k + 1) % 60 == 0 {
            if let Some(g) = world.last_step_gpu_time() {
                let mut by: Vec<(&str, f64, u32)> = Vec::new();
                for &(l, ms) in &g.segments {
                    match by.iter_mut().find(|x| x.0 == l) {
                        Some(x) => {
                            x.1 += ms;
                            x.2 += 1;
                        }
                        None => by.push((l, ms, 1)),
                    }
                }
                by.sort_by(|a, b| b.1.total_cmp(&a.1));
                println!(
                    "   gpu: total {:.2} ms, span {:.2} ms, {} segments ({} unmeasured); {}",
                    g.total_ms,
                    g.span_ms,
                    g.segments.len(),
                    g.unmeasured.len(),
                    by.iter()
                        .take(12)
                        .map(|(l, ms, n)| format!("{l} {ms:.2}×{n}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            // `--piece-boxes`: each car's coupling piece boxes (label: its size, m) and its
            // live chunks.
            if std::env::args().any(|a| a == "--piece-boxes") {
                if let Some(x) = world.explicit_vehicles() {
                    let (_, rec) = x.piece_window_records();
                    if rec != 0 {
                        use engine_webgpu::explicit_coupling::read_words;
                        let recs = 25u64; // (engine_exec::explicit_plan::PIECE_BOX_RECS)
                        let mut line = Vec::new();
                        for v in 0..x.vehicle_count() as u64 {
                            let w = read_words(world.device(), world.queue(), x.misc_buffer(), (u64::from(rec) + v * recs) * 16, recs * 16);
                            let un = |k: u32| f32::from_bits(if k & 0x8000_0000 != 0 { k & 0x7FFF_FFFF } else { !k });
                            let boxes: Vec<String> = (0..(w.len() - 4) / 6)
                                .filter(|g| w[6 * g] != u32::MAX)
                                .map(|g| {
                                    let lo = glam::Vec3::new(un(w[6 * g]), un(w[6 * g + 1]), un(w[6 * g + 2]));
                                    let hi = glam::Vec3::new(un(w[6 * g + 3]), un(w[6 * g + 4]), un(w[6 * g + 5]));
                                    format!("{g}:{:.0}", (hi - lo).length())
                                })
                                .collect();
                            line.push(format!("[{}]", boxes.join(" ")));
                        }
                        println!("   piece boxes m: {}; chunks live {}", line.join(" "), lab.bench.stats()["mountain"]["chunksLive"]);
                    }
                }
            }
            if let Some(x) = world.explicit_vehicles_mut() {
                let rc = x.take_reuse_cause();
                if rc.iter().any(|&c| c > 0) {
                    println!(
                        "   reuse causes: no list {}, main spread {}, with growth {}, pieces {}, backoff walks {}, non-finite {}, reused {}, mean main spread {:.1} mm",
                        rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6],
                        rc[7] as f64 / 10.0 / (rc[1] + rc[2] + rc[3] + rc[6]).max(1) as f64
                    );
                }
                let st = x.take_pair_reuse_stats();
                if st[0] > 0 {
                    println!(
                        "   pair list: built {} times, reused on {} steps ({:.0} %), {:.0} pairs a build, most {}",
                        st[0],
                        st[1],
                        100.0 * st[1] as f64 / (st[0] + st[1]) as f64,
                        st[2] as f64 / st[0] as f64,
                        st[3]
                    );
                }
                let fk = x.take_fleet_keep_stats();
                if fk[0] + fk[1] > 0 {
                    println!(
                        "   vehicle contact's kept candidates: {} builds ({} for a calm node, {} for two vehicles, {} for cover), {} passes reused ({:.1} %), {} passes on the front's own path; {:.0} pairs a build, most an owner {}; largest bound at a build {:.3} m, allowance between vehicles {:.3} m",
                        fk[0],
                        fk[5],
                        fk[9],
                        fk[6],
                        fk[1],
                        100.0 * fk[1] as f64 / (fk[0] + fk[1] + fk[3]).max(1) as f64,
                        fk[3],
                        fk[2] as f64 / fk[0].max(1) as f64,
                        fk[4],
                        f32::from_bits(fk[7] as u32),
                        f32::from_bits(fk[8] as u32)
                    );
                }
            }
            if alternate.is_some() {
                println!(
                    "   alternate: off {:.2} ms/step, on {:.2} ms/step",
                    alt[0].0 / f64::from(alt[0].1.max(1)),
                    alt[1].0 / f64::from(alt[1].1.max(1))
                );
                alt = [(0.0, 0); 2];
            }
            if arg("--tri-census").is_some() {
                use engine_webgpu::explicit_coupling::read_words;
                let x = world.explicit_vehicles().expect("vehicles");
                let p = lab.read_plan();
                let f = |w: u32| f32::from_bits(w);
                let nodes = read_words(world.device(), world.queue(), x.nodes_buffer(), p.nodes.0, p.nodes.1);
                let beams = read_words(world.device(), world.queue(), x.beams_buffer(), p.beams.0, p.beams.1);
                let pos: Vec<glam::Vec3> = (0..nodes.len() / 12).map(|i| glam::Vec3::new(f(nodes[12 * i]), f(nodes[12 * i + 1]), f(nodes[12 * i + 2]))).collect();
                let broken: Vec<bool> = (0..beams.len() / 8).map(|j| f(beams[8 * j + 6]) != 0.0).collect();
                let vel: Vec<glam::Vec3> = (0..nodes.len() / 12).map(|i| glam::Vec3::new(f(nodes[12 * i + 4]), f(nodes[12 * i + 5]), f(nodes[12 * i + 6]))).collect();
                println!("   census: {}", tri_census(template, &pos, &vel, &broken));
                println!("   reuse census: {}", reuse_census(template, &pos, &vel, &broken));
                println!("   reuse cause: {}", reuse_cause(template, &pos, &vel, &broken));
            }
            // Every car's box diagonal (m), when there is more than one: a car that has
            // blown up shows here (the page's stability check is the wasm host's).
            if let Some(x) = world.explicit_vehicles() {
                let all = x.positions();
                if all.len() > 1 {
                    let d: Vec<String> = all
                        .iter()
                        .map(|p| {
                            let (lo, hi) = p.iter().fold((glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN)), |(a, b), q| (a.min(*q), b.max(*q)));
                            format!("{:.1}", (hi - lo).length())
                        })
                        .collect();
                    println!("   boxes m: {}", d.join(" "));
                    // (median node speed / fastest node, km/h: a body that has blown up
                    // has nodes far faster than its own motion.)
                    let v: Vec<String> = x
                        .velocities()
                        .iter()
                        .map(|vs| {
                            let mut sp: Vec<f32> = vs.iter().map(|q| q.length() * 3.6).collect();
                            sp.sort_by(f32::total_cmp);
                            format!("{:.0}/{:.0}", sp[sp.len() / 2], sp[sp.len() - 1])
                        })
                        .collect();
                    println!("   speeds km/h: {}", v.join(" "));
                    // (Where each car's fastest node is, and how many of its nodes are
                    // past 200 km/h: debris in free fall has left the ground.)
                    let w: Vec<String> = all
                        .iter()
                        .zip(x.velocities().iter())
                        .map(|(p, vs)| {
                            let (i, top) = vs.iter().enumerate().fold((0, 0.0f32), |(bi, b), (i, q)| if q.length() > b { (i, q.length()) } else { (bi, b) });
                            let fast = vs.iter().filter(|q| q.length() * 3.6 > 200.0).count();
                            let med = p[p.len() / 2];
                            format!("{fast}@x{:.0},y{:.0},z{:.0}(car x{:.0},y{:.0},z{:.0};{:.0})", p[i].x, p[i].y, p[i].z, med.x, med.y, med.z, top * 3.6)
                        })
                        .collect();
                    println!("   fast nodes: {}", w.join(" "));
                    // The fleet's energy ledger (kJ): kinetic, potential (m g y) and their
                    // sum, per car and in all. Contact and yielding only take energy out:
                    // the total may not rise, and a car's own sum rises only by what
                    // another car hands it.
                    let vel = x.velocities();
                    // (Every car is the template: its nodes' masses.)
                    let (mut total, mut per) = (0.0f64, Vec::new());
                    for (p, v) in all.iter().zip(&vel) {
                        let (mut ke, mut pe) = (0.0f64, 0.0f64);
                        for ((q, w), n) in p.iter().zip(v).zip(&template.nodes) {
                            let m = f64::from(n.mass);
                            ke += 0.5 * m * f64::from(w.length_squared());
                            pe += m * 9.81 * f64::from(q.y);
                        }
                        total += ke + pe;
                        per.push(format!("{:.0}+{:.0}", ke * 1e-3, pe * 1e-3));
                    }
                    println!("   energy kJ (kinetic+potential a car): {}; total {:.3}", per.join(" "), total * 1e-3);
                }
            }
            if world.explicit_vehicle_sleep().is_some() {
                println!(
                    "   vehicle sleep: asleep {}; first slept at s {}; wakes {:?}",
                    sleep_log.iter().map(|s| if s.2 { "1" } else { "0" }).collect::<Vec<_>>().join(""),
                    sleep_log.iter().map(|s| s.0.map_or("-".to_string(), |t| format!("{t:.2}"))).collect::<Vec<_>>().join(" "),
                    sleep_log.iter().map(|s| s.1).collect::<Vec<_>>()
                );
            }
            if lab.bench.debris.ttl.is_some() || lab.bench.debris.pieces > 0 {
                let d = &lab.bench.debris;
                println!("   debris retired: {} pieces, {} nodes, {:.2} kg", d.pieces, d.nodes, d.mass);
            }
            let c = &lab.bench.car;
            let size = c.hi - c.lo;
            let pairs = world
                .explicit_vehicles()
                .map(|x| x.surface_pair_stats())
                .unwrap_or_default();
            println!(
                "{:>6.1} {:>8.2} {:>8.2} {:>8.1} {:>7.0} {:>18} {:>12} {:>16}",
                lab.bench.t,
                sum / f64::from(n),
                worst,
                c.speed * 3.6,
                -c.centre.y,
                format!("{:.2}×{:.2}×{:.2}", size.x, size.y, size.z),
                format!("{:?}", c.proxies),
                pairs
                    .first()
                    .map_or("-".into(), |p| format!("{}·{}", p.0, p.1)),
            );
            (sum, worst, n) = (0.0, 0.0, 0);
        }
    }
    if let (Some(tr), Some(path)) = (trace, arg("--trace")) {
        std::fs::write(path, tr)?;
    }
    if let Some(x) = world.explicit_vehicles() {
        x.kernel_status().map_err(anyhow::Error::msg)?;
    }
    // `--end-world N`: after the run, the World's own cost a step. N World steps timed one
    // by one (the step and its poll, as the run's), then the vehicles put back to the
    // state before them and the same explicit steps run as N frames of the vehicles' solver
    // alone (33, 33, 34, ... steps, as the World's clock gives them); `--end-world-rounds R`
    // (default 3) times, each round from that same vehicle state, every step and frame
    // pooled. The difference of the medians is what the World adds to the frame: the rigid
    // step, the coupling's passes and the host's encoding. (Time only: the restore puts
    // back the vehicles, not the rigid bodies.)
    if let Some(n) = arg("--end-world").and_then(|x| x.parse::<usize>().ok()).filter(|&n| n > 0) {
        let rounds: usize = arg("--end-world-rounds").and_then(|x| x.parse().ok()).unwrap_or(3).max(1);
        let snap = {
            let x = world.explicit_vehicles_mut().expect("vehicles");
            x.wait();
            x.snapshot()
        };
        let (mut steps_ms, mut frames_ms) = (Vec::new(), Vec::new());
        for _ in 0..rounds {
            {
                let x = world.explicit_vehicles_mut().expect("vehicles");
                x.restore(&snap);
                x.wait();
            }
            for _ in 0..n {
                lab.before_step(&mut world)?;
                let t = std::time::Instant::now();
                world.step()?;
                world.poll_blocking()?;
                steps_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let x = world.explicit_vehicles_mut().expect("vehicles");
            x.wait();
            x.restore(&snap);
            x.wait();
            let mut carry = 0.0f64;
            for _ in 0..n {
                carry += (1.0 / 60.0) / 0.0005;
                let k = (carry + 1e-6).floor();
                carry -= k;
                let t = std::time::Instant::now();
                x.frame(k as u32);
                x.wait();
                frames_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (w, f) = (med(&mut steps_ms), med(&mut frames_ms));
        println!(
            "end world: {} steps, World step {w:.3} ms, frame alone {f:.3} ms, the World adds {:.3} ms (medians, wall; {rounds} rounds of {n})",
            n * rounds,
            w - f
        );
    }
    // `--end-frames`: after the run, 20 frames of the vehicles' solver alone from the final
    // state (wall ms a frame), then the wide passes' profile when the frame is wide.
    if std::env::args().any(|a| a == "--end-frames") {
        let x = world.explicit_vehicles_mut().expect("vehicles");
        x.wait();
        let t = std::time::Instant::now();
        for _ in 0..20 {
            x.frame(34);
            x.wait();
        }
        println!("end frames: {:.2} ms a frame (wall), wide {}", t.elapsed().as_secs_f64() * 1e3 / 20.0, x.is_wide());
        let p = x.profile_wide_frame(34);
        println!("end wide profile: {}", p.iter().filter(|x| x.1 > 0.0).map(|(n, ms)| format!("{} {ms:.2}", n.trim_start_matches("explicit_beams_wide_"))).collect::<Vec<_>>().join(", "));
    }
    // The World step's wall time (the step and its poll; what a page waits for): median,
    // p95 and the worst step, from t >= 0.5 s (past the pipelines' first compiles).
    // `--wall-csv FILE`: every step's.
    {
        let mut v: Vec<(f64, f64)> = wall_steps.iter().filter(|x| x.0 >= 0.5).map(|x| (x.1, x.0)).collect();
        if !v.is_empty() {
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize].0;
            let w = v[v.len() - 1];
            println!("wall ms a World step (t >= 0.5 s, {} steps): median {:.3}, p95 {:.3}, worst {:.3} at t {:.2}", v.len(), q(0.5), q(0.95), w.0, w.1);
        }
        if let Some(path) = arg("--wall-csv") {
            let mut out = String::from("t,wall_ms\n");
            for x in &wall_steps {
                out.push_str(&format!("{:.4},{:.4}\n", x.0, x.1));
            }
            std::fs::write(path, out)?;
        }
    }
    if !gpu_steps.is_empty() {
        // The run's GPU time a World step: median, p95 and the worst step (and when).
        let stat = |name: &str, f: &dyn Fn(&(f64, f64, f64, f64)) -> f64, from: f64| {
            let mut v: Vec<(f64, f64)> = gpu_steps.iter().filter(|x| x.0 >= from).map(|x| (f(x), x.0)).collect();
            if v.is_empty() {
                return;
            }
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize].0;
            let w = v[v.len() - 1];
            println!("gpu ms a step, {name} (t >= {from:.1} s, {} steps): median {:.3}, p95 {:.3}, p99 {:.3}, worst {:.3} at t {:.2}", v.len(), q(0.5), q(0.95), q(0.99), w.0, w.1);
        };
        for from in [0.0, 0.5] {
            stat("whole step (span)", &|x| x.1, from);
            stat("vehicles (frame + coupling passes)", &|x| x.2, from);
            stat("explicit frame", &|x| x.3, from);
        }
        if let Some(path) = arg("--gpu-csv") {
            let mut out = String::from("t,span_ms,vehicles_ms,frame_ms\n");
            for x in &gpu_steps {
                out.push_str(&format!("{:.4},{:.4},{:.4},{:.4}\n", x.0, x.1, x.2, x.3));
            }
            std::fs::write(path, out)?;
        }
    }
    // `--save-final FILE`: every car's node positions at the run's end (f32 x, y, z a
    // node, the cars in order). `--compare-final FILE`: against another run's: per car
    // the largest node displacement between the two (mm), and the fleet's energy. Two
    // runs of the World are not bit for bit the same (a crash is chaotic and native
    // runs differ in timing): compare a setting's difference with two same-setting runs'.
    if let Some(x) = world.explicit_vehicles() {
        let pos = x.positions();
        if let Some(file) = arg("--compare-final") {
            let raw = std::fs::read(&file)?;
            let other: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let mut at = 0;
            let mut out = Vec::new();
            for p in &pos {
                let mut worst = 0.0f32;
                for q in p {
                    if at + 3 <= other.len() {
                        worst = worst.max((*q - glam::Vec3::new(other[at], other[at + 1], other[at + 2])).length());
                    }
                    at += 3;
                }
                out.push(format!("{:.1}", worst * 1e3));
            }
            println!("final positions against {file}: largest node difference a car, mm: {}", out.join(" "));
        }
        if let Some(file) = arg("--save-final") {
            let bytes: Vec<u8> = pos.iter().flatten().flat_map(|q| q.to_array()).flat_map(f32::to_le_bytes).collect();
            std::fs::write(&file, bytes)?;
        }
        let vel = x.velocities();
        let (mut ke, mut pe) = (0.0f64, 0.0f64);
        for (p, v) in pos.iter().zip(&vel) {
            for ((q, w), n) in p.iter().zip(v).zip(&template.nodes) {
                let m = f64::from(n.mass);
                ke += 0.5 * m * f64::from(w.length_squared());
                pe += m * 9.81 * f64::from(q.y);
            }
        }
        println!("final energy kJ: kinetic {:.3}, potential {:.3}, total {:.3}", ke * 1e-3, pe * 1e-3, (ke + pe) * 1e-3);
        if world.explicit_vehicle_sleep().is_some() {
            println!(
                "final vehicle sleep: asleep {}; first slept at s {}; wakes {:?}",
                sleep_log.iter().map(|s| if s.2 { "1" } else { "0" }).collect::<Vec<_>>().join(""),
                sleep_log.iter().map(|s| s.0.map_or("-".to_string(), |t| format!("{t:.2}"))).collect::<Vec<_>>().join(" "),
                sleep_log.iter().map(|s| s.1).collect::<Vec<_>>()
            );
        }
    }
    println!(
        "wall {:.1} s for {secs:.1} s simulated",
        t_all.elapsed().as_secs_f64()
    );
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
    // `--node-masses I,J,…`: those nodes' masses and the car's mass distribution (kg), then
    // exit (no GPU).
    if let Some(list) = arg("--node-masses") {
        let mut m: Vec<f32> = template.nodes.iter().map(|n| n.mass).collect();
        for i in list.split(',').filter_map(|i| i.parse::<usize>().ok()) {
            println!("node {i}: {:.3} kg", template.nodes[i].mass);
        }
        m.sort_by(f32::total_cmp);
        let n = m.len();
        println!("node masses kg: min {:.3}, p10 {:.3}, median {:.3}, p90 {:.3}, max {:.3}, mean {:.3}", m[0], m[n / 10], m[n / 2], m[n * 9 / 10], m[n - 1], m.iter().sum::<f32>() / n as f32);
        return Ok(());
    }
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

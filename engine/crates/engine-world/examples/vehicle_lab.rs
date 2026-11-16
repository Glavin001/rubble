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
    // A local JBeam export (a directory with `files.json`, as the page's `__mods/…`): the
    // page's generated sedan, with the page's settings (air drag on).
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
    if let Some(r) = arg("--rockslide") {
        m["rockslide"] = json!(r == "1");
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
    let (mut world, mut lab) = Lab::build(GpuContext::Default, template, opts, |d| {
        d.gpu_timing = match timing.as_deref() {
            Some("phases") => GpuTiming::Phases,
            Some("step") => GpuTiming::Step,
            _ => GpuTiming::Off,
        }
    })?;
    lab.read_blocking(&world, true)?;
    lab.bench.set_script(&script)?;
    println!("debris time to live {:?}; notes {:?}", lab.bench.debris.ttl, lab.bench.notes);
    let flag = |k: &str| arg(k).map(|x| x == "1");
    if let Some(x) = world.explicit_vehicles_mut() {
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
    let kernel_ab = arg("--ab-switch").as_deref() == Some("kernel");
    let kernel_dir = arg("--ab-kernel-dir").map(std::path::PathBuf::from);
    anyhow::ensure!(!kernel_ab || kernel_dir.is_some(), "--ab-switch kernel needs --ab-kernel-dir DIR");
    let mut other = None;
    let n_nodes = template.nodes.len();
    let switch = |x: &mut engine_webgpu::ExplicitBeamsGpu, on: bool, labels: &[u8]| {
        if groupsplit_ab {
            x.set_group_split(on);
            x.set_self_split(on);
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
        x.wait();
        lab.bench.t = t;
        ab = vec![t];
    }
    for k in 0..windows {
        if ab.first().is_some_and(|&t| lab.bench.t >= t) {
            ab.remove(0);
            if let Some(dir) = &save_dir {
                std::fs::create_dir_all(dir)?;
                let host = lab.fleet_host_state();
                let x = world.explicit_vehicles().expect("vehicles");
                let snap = x.snapshot();
                let mut out = lab.bench.t.to_bits().to_le_bytes().to_vec();
                out.extend((host.len() as u32).to_le_bytes());
                for (labels, retired) in &host {
                    out.extend((labels.len() as u32).to_le_bytes());
                    out.extend(labels);
                    out.extend((retired.len() as u32).to_le_bytes());
                    out.extend(retired.iter().map(|r| u8::from(*r)));
                }
                out.extend(x.snapshot_to_bytes(&snap));
                let file = dir.join(format!("cars{}_t{:.1}.fx", host.len(), lab.bench.t));
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
                for m in masks.split(',').filter_map(|m| m.parse::<u32>().ok()) {
                    x.restore(&snap);
                    x.set_split_ablation(m);
                    x.wait();
                    let t = std::time::Instant::now();
                    for _ in 0..20 {
                        x.frame(34);
                        x.wait();
                    }
                    line.push(format!("{m}: {:.2}", t.elapsed().as_secs_f64() * 1e3 / 20.0));
                }
                x.set_split_ablation(0);
                x.restore(&snap);
                println!("   split ablation at t {:.1} (ms a frame): {}", lab.bench.t, line.join(", "));
            }
            // `--ab-perf-masks M1,M2,…`: the same 20 frames with the run's perf flags XOR
            // each mask, three times round the list: per mask the median and the fastest
            // tenth of its 60 frames' wall ms (a frame's wall time rises with any other
            // load on the machine: the low figures are the comparable ones); "=": the
            // frames ended in the first mask's state, bit for bit. A lossless switch's
            // cost or gain, and a kernel's time-only measurement bits when a build has any.
            if let Some(masks) = arg("--ab-perf-masks") {
                let f = x.perf_flags();
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
                        x.set_perf_flags(f ^ m);
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
            // `--ab-profile`: the frame as it runs now, by dispatch kind (time only): 10
            // frames untimed (wall), then the same 10 from the same snapshot with every
            // dispatch in its own timed pass (`profile_dispatches`). The table closes: the
            // kinds' sum + the time between passes = the timed span; the span against the
            // untimed frame is what the timing itself costs.
            if std::env::args().any(|a| a == "--ab-profile") {
                const N: u32 = 10;
                let mut plain = 0.0;
                x.restore(&snap);
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
            // `--ab-same`: both runs with the culls on (the harness's own repeatability).
            let second = std::env::args().any(|a| a == "--ab-same");
            for on in if kernel_ab { vec![] } else { vec![true, second] } {
                x.restore(&snap);
                switch(x, on, &labels);
                x.wait();
                // (One untimed frame first: a pipeline this path creates on first use
                // compiles there, not in the timing. Then the state again.)
                x.frame(34);
                x.wait();
                x.restore(&snap);
                x.wait();
                // (AB_GAP_MS=n: the GPU idle n ms before each frame, as a display's frame
                // loop leaves it; the frames' own time alone is summed.)
                let gap: u64 = std::env::var("AB_GAP_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                // (`--ab-frames N`: the frames each way, 20 unless given.)
                let frames: u32 = arg("--ab-frames").and_then(|v| v.parse().ok()).unwrap_or(20).max(1);
                let mut sum = 0.0;
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
                    sum += t.elapsed().as_secs_f64();
                }
                let ms = sum * 1e3 / f64::from(frames);
                if env_u("AB_COUNTS", 0) != 0 {
                    println!("     counts ({}): {:?}", if on { "on" } else { "off" }, x.contact_group_counts());
                }
                // The switch's own header bit back, so the words compare the state alone.
                switch(x, true, &labels);
                x.wait();
                out.push((ms, x.state_words(), x.coupling_words()));
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
                } else if groupsplit_ab {
                    "group split"
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
            let physics = out[0].1.iter().zip(&out[1].1).filter(|((n, _), _)| ["nodes", "beams", "groups"].contains(n)).all(|((_, a), (_, b))| a == b);
            if wide_ab {
                println!("   wide: nodes, beams and groups bit for bit: {physics}");
            }
            // (AB_TIME_ONLY=1: a measurement switch that changes the results on purpose.)
            anyhow::ensure!(same || (wide_ab && physics) || std::env::var("AB_TIME_ONLY").is_ok(), "the A/B switch changed the state");
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
        world.poll_blocking()?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
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
                    println!("   energy kJ (kinetic+potential a car): {}; total {:.0}", per.join(" "), total * 1e-3);
                }
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
    if let Some(x) = world.explicit_vehicles() {
        x.kernel_status().map_err(anyhow::Error::msg)?;
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

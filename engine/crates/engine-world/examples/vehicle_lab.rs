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

/// The page's mountain (`vehicle-lab.html?mode=mountain&…`) for its cost: per second of
/// simulated time, the wall time of a World step (each step waited on: its GPU time and
/// the readback), with the car's speed, node box, coupling use and the surface pair
/// counts. `--steepness 55 --roughness 0.25 --launch 12 --rockslide 1 --surface 1` is
/// the owner's report. Measurement switches (one at a time): `--ablate MASK` from
/// `--ablate-at T` (results meaningless), `--surface-pairs 0`, `--surface-culls 0`.
fn mountain_perf(template: &ExplicitBeams, gpu: bool) -> anyhow::Result<()> {
    anyhow::ensure!(gpu, "mountain-perf runs in the GPU World (--mode gpu)");
    let secs: f64 = arg("--secs").and_then(|x| x.parse().ok()).unwrap_or(40.0);
    let scene = arg("--scene").unwrap_or_else(|| "mountain".into());
    let script = json!({ "scene": scene, "surface": arg("--surface").as_deref() == Some("1"), "events": [] });
    let opts = options(&script);
    println!(
        "options: {scene}, surface {} {:?}",
        opts.surface_contact, opts.mountain
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
    let n_nodes = template.nodes.len();
    let switch = |x: &mut engine_webgpu::ExplicitBeamsGpu, on: bool, labels: &[u8]| {
        if pieces_ab {
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
    for k in 0..windows {
        if ab.first().is_some_and(|&t| lab.bench.t >= t) {
            ab.remove(0);
            let labels = lab.piece_labels().to_vec();
            let x = world.explicit_vehicles_mut().expect("vehicles");
            x.wait();
            let snap = x.snapshot();
            // `--ab-ablate M1,M2,…`: after the A/B, the same frames under each ablation
            // mask (time only; 2147483648 is the ablation kernel with nothing skipped).
            if let Some(masks) = arg("--ab-ablate") {
                let mut line = Vec::new();
                for m in masks.split(',').filter_map(|m| m.parse::<u32>().ok()) {
                    x.restore(&snap);
                    x.set_ablation(m);
                    x.wait();
                    let t = std::time::Instant::now();
                    for _ in 0..20 {
                        x.frame(34);
                        x.wait();
                    }
                    line.push(format!(
                        "{m}: {:.2}",
                        t.elapsed().as_secs_f64() * 1e3 / 20.0
                    ));
                }
                x.set_ablation(0);
                println!(
                    "   ablations at t {:.1} (ms/frame): {}",
                    lab.bench.t,
                    line.join(", ")
                );
            }
            let mut out = Vec::new();
            // `--ab-same`: both runs with the culls on (the harness's own repeatability).
            let second = std::env::args().any(|a| a == "--ab-same");
            for on in [true, second] {
                x.restore(&snap);
                switch(x, on, &labels);
                x.wait();
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    x.frame(34);
                    x.wait();
                }
                let ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
                // The switch's own header bit back, so the words compare the state alone.
                switch(x, true, &labels);
                x.wait();
                out.push((ms, x.state_words(), x.coupling_words()));
            }
            let same = out[0].1 == out[1].1 && out[0].2 == out[1].2;
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
                }
            }
            println!(
                "   A/B at t {:.1}: {} on {:.2} ms/frame, off {:.2} ms/frame; state bit for bit: {same}",
                lab.bench.t,
                if pieces_ab { "piece labels" } else if reuse_ab { "pair reuse" } else { "surface culls" },
                out[0].0,
                out[1].0
            );
            anyhow::ensure!(same, "the A/B switch changed the state");
            x.restore(&snap);
            switch(x, true, &labels);
            x.wait();
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

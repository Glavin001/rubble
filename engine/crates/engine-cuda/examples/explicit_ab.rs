//! Old against new on the CUDA host, from one state: a scenario, two (or more) switch
//! settings, every state word compared bit for bit between them, each against the CPU
//! reference within the parity tests' tolerances, and per-kernel GPU timings.
//! Device code is built locally (engine-cuda's build.rs: Slang → CUDA → fatbin), so a
//! Modal GPU only runs it (seconds); `engine/tools/modal/gpu_run.sh` ships it.
//!
//! ```text
//! cargo run --release -p engine-cuda --example explicit_ab -- \
//!     --scenario contact --count 8 --frames 30 --a pair_reuse=0 --b pair_reuse=0.02
//! cargo run --release -p engine-cuda --example explicit_ab -- --spec ab.json
//! ```
//! Scenario (`--scenario` / "scenario"):
//! - `parked`: one car at rest on the ground;
//! - `crash`: one car at 56 km/h into a static wall;
//! - `tumble`: one car dropped 1.5 m, pitched and rolled, spinning (a mountain-like tumble);
//! - `apart`: `count` cars side by side, 6 m apart, rolling at 10 m/s (no contact);
//! - `contact`: `count` cars nose to tail, alternate ones driving into the next at 8 m/s;
//! - `overlimit`: a chain of `count` (at least VV_PART + 2) cars rear-ended into one
//!   contact group larger than the small groups' limit (the big-group module).
//!
//! `--car procedural` (default: the parity tests' crash sedan) | `repo` (the
//! repository's `rubble_sedan_ror.truck`) | a JBeam export directory (`files.json`,
//! `*.jbeam`, `*.pc`; `@sedan` with gpu_run.sh). `--at S`: the CPU reference runs S
//! seconds first and both hosts start from its state (a crash or tumble mid-way).
//! `--frames F` (default 30) of `--steps N` (default 17: 120 fps at 2 kHz).
//!
//! Switches (`--a` / `--b` / `--c` ..., `k=v,k2=v2`; JSON: "variants": [{"name",
//! "switches": {k: v}}]), each applied to a host of its own built from the same state:
//! - `pair_reuse=M`: the self-collision pair list's reuse margin in metres (0: list
//!   every step);
//! - `vv_grid=0|1`: vehicle contact's candidate grid;
//! - `wide=0|1`: the wide passes (several workgroups a vehicle) where the fleet allows;
//! - `contact=0|1`: vehicle contact; `contact_margin=M` (metres);
//! - `env.NAME=V`: an environment variable set while that variant's host is built
//!   (switches the plan reads at upload, e.g. `env.RUBBLE_PAIR_REUSE=0`);
//! - `param.W=V`: the planner's parameter word W (a switch that is a word, e.g. a
//!   branch's `W_BEAM_ORDER`).
//!
//! Not on this host (rejected with the reason): `coupling_list`, `coupling_fold`,
//! `piece_windows` (the scenarios have no World coupling or piece labels), `ablate`
//! (no ablation modules are built for CUDA).
//!
//! Exit status: 0 when every variant is bit for bit the first and each is within the
//! CPU tolerance; 1 otherwise (the first differing words are printed).

use engine_cuda::CudaExplicitBeams;
use engine_exec::explicit_plan::VV_PART;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, StaticBox};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::{Quat, Vec3};
use serde_json::{json, Value};

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn kv(spec: &str) -> serde_json::Map<String, Value> {
    let mut m = serde_json::Map::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (k, v) = part.split_once('=').unwrap_or((part, "1"));
        m.insert(k.to_string(), Value::String(v.to_string()));
    }
    m
}

/// The run's description: `--spec FILE` (JSON) or the flags.
fn spec() -> anyhow::Result<Value> {
    if let Some(path) = arg("--spec") {
        return Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?);
    }
    let mut variants = Vec::new();
    for (i, flag) in ["--a", "--b", "--c", "--d"].iter().enumerate() {
        if let Some(s) = arg(flag) {
            let name = ["a", "b", "c", "d"][i];
            variants.push(json!({ "name": name, "switches": kv(&s) }));
        }
    }
    if variants.is_empty() {
        variants = vec![json!({ "name": "a", "switches": {} }), json!({ "name": "same", "switches": {} })];
    }
    Ok(json!({
        "scenario": arg("--scenario").unwrap_or_else(|| "parked".into()),
        "car": arg("--car").unwrap_or_else(|| "procedural".into()),
        "count": arg("--count").map(|c| c.parse::<u64>().unwrap()),
        "frames": arg("--frames").map_or(30, |f| f.parse::<u64>().unwrap()),
        "steps": arg("--steps").map_or(17, |f| f.parse::<u64>().unwrap()),
        "at": arg("--at").map_or(0.0, |f| f.parse::<f64>().unwrap()),
        "variants": variants,
    }))
}

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings { turbulence: false, ..Default::default() }
}

/// One car at the origin, its lowest node 1 cm above the ground.
fn car(kind: &str) -> anyhow::Result<ExplicitBeams> {
    let mut c = match kind {
        "procedural" => {
            let text = car_truck_bare(&CarParams { minimass: 10.0, ..CarParams::sedan() });
            ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).map_err(anyhow::Error::msg)?
        }
        "repo" => {
            let text = std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
            ))?;
            ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).map_err(anyhow::Error::msg)?
        }
        dir => {
            let dir = std::path::Path::new(dir);
            let list: Vec<String> = serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
            let mut texts = Vec::new();
            let mut pc = None;
            for f in &list {
                if f.ends_with(".jbeam") {
                    texts.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
                } else if f.ends_with(".pc") {
                    pc = Some(std::fs::read_to_string(dir.join(f))?);
                }
            }
            let refs: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).map_err(anyhow::Error::msg)?;
            let s = ExplicitBeamsSettings { turbulence: false, drag: false, ..Default::default() };
            ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Pressure, Vec3::ZERO, Vec3::ZERO, s)
                .map_err(anyhow::Error::msg)?
                .0
        }
    };
    let low = c.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut c.nodes {
        n.pos.y += 0.01 - low;
    }
    Ok(c)
}

fn moved(c: &ExplicitBeams, by: Vec3, vel: Vec3) -> ExplicitBeams {
    let mut c = c.clone();
    for n in &mut c.nodes {
        n.pos += by;
        n.vel = vel;
    }
    if vel.length() > 0.0 {
        c.roll_tyres(vel);
    }
    c
}

fn z_extent(c: &ExplicitBeams) -> (f32, f32) {
    c.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| (a.min(n.pos.z), b.max(n.pos.z)))
}

fn scenario(name: &str, base: &ExplicitBeams, count: Option<usize>) -> anyhow::Result<Vec<ExplicitBeams>> {
    let (z0, z1) = z_extent(base);
    let len = z1 - z0;
    Ok(match name {
        "parked" => vec![base.clone()],
        "crash" => {
            let v = Vec3::new(0.0, 0.0, 56.0 / 3.6);
            let mut c = moved(base, Vec3::ZERO, v);
            c.boxes.push(StaticBox::centered(Vec3::new(0.0, 1.0, z1 + 0.05 + 1.0), Vec3::new(3.0, 2.0, 1.0)));
            vec![c]
        }
        "tumble" => {
            let mut c = base.clone();
            let centre = c.nodes.iter().map(|n| n.pos).sum::<Vec3>() / c.nodes.len() as f32;
            let q = Quat::from_rotation_x(0.5) * Quat::from_rotation_z(0.7);
            let spin = Vec3::new(2.0, 0.5, 3.0);
            for n in &mut c.nodes {
                let r = q * (n.pos - centre);
                n.pos = centre + r + Vec3::new(0.0, 1.5, 0.0);
                n.vel = Vec3::new(0.0, -2.0, 6.0) + spin.cross(r);
            }
            vec![c]
        }
        "apart" => {
            let v = Vec3::new(0.0, 0.0, 10.0);
            (0..count.unwrap_or(8)).map(|i| moved(base, Vec3::new(6.0 * i as f32, 0.0, 0.0), v)).collect()
        }
        "contact" => (0..count.unwrap_or(8))
            .map(|i| {
                let v = if i % 2 == 0 { Vec3::new(0.0, 0.0, 8.0) } else { Vec3::ZERO };
                moved(base, Vec3::new(0.0, 0.0, i as f32 * (len + 0.4)), v)
            })
            .collect(),
        "overlimit" => {
            let n = count.unwrap_or(VV_PART + 2).max(VV_PART + 2);
            (0..n)
                .map(|i| {
                    let v = if i == 0 { Vec3::new(0.0, 0.0, 14.0) } else { Vec3::ZERO };
                    moved(base, Vec3::new(0.0, 0.0, i as f32 * (len + 0.3)), v)
                })
                .collect()
        }
        other => anyhow::bail!("unknown scenario {other} (parked, crash, tumble, apart, contact, overlimit)"),
    })
}

/// A host for `fleet` with `switches` applied; `env.*` are set while it is built.
fn host(fleet: &[ExplicitBeams], switches: &serde_json::Map<String, Value>) -> anyhow::Result<CudaExplicitBeams> {
    let s = |v: &Value| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
    let mut set = Vec::new();
    for (k, v) in switches {
        if let Some(name) = k.strip_prefix("env.") {
            set.push((name.to_string(), std::env::var(name).ok()));
            std::env::set_var(name, s(v));
        }
    }
    let built = CudaExplicitBeams::new(fleet);
    for (name, old) in set {
        match old {
            Some(o) => std::env::set_var(&name, o),
            None => std::env::remove_var(&name),
        }
    }
    let mut g = built?;
    for (k, v) in switches {
        let v = s(v);
        let on = || v != "0";
        match k.as_str() {
            k if k.starts_with("env.") => {}
            // A parameter word by index (a switch that is a planner word, e.g. a
            // stream branch's W_BEAM_ORDER: `param.96=0`).
            k if k.starts_with("param.") => {
                let w: usize = k["param.".len()..].parse()?;
                g.planner_mut().words[w] = v.parse()?;
            }
            "pair_reuse" => g.set_pair_reuse_margin(v.parse()?),
            "vv_grid" => g.set_vv_grid(on())?,
            "wide" => {
                g.set_wide(on());
            }
            "contact" => g.set_vehicle_contact(on()),
            "contact_margin" => g.planner_mut().set_contact_margin(v.parse()?),
            "coupling_list" | "coupling_fold" | "piece_windows" => anyhow::bail!(
                "{k}: the World coupling's switch; these scenarios have no coupling proxies (run it \
                 through the WebGPU route: vehicle_lab --ab-switch)"
            ),
            "ablate" => anyhow::bail!("ablate: no EB_ABLATE modules are built for CUDA (WebGPU route only)"),
            other => anyhow::bail!("unknown switch {other}"),
        }
    }
    Ok(g)
}

fn rms_max(a: &[Vec3], b: &[Vec3]) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for (p, q) in a.iter().zip(b) {
        let d = p.distance(*q);
        sum += f64::from(d).powi(2);
        max = max.max(d);
    }
    ((sum / a.len().max(1) as f64).sqrt() as f32, max)
}

fn first_diff(a: &[(&str, Vec<u32>)], b: &[(&str, Vec<u32>)]) -> Option<String> {
    for ((name, x), (_, y)) in a.iter().zip(b) {
        if x.len() != y.len() {
            return Some(format!("{name}: {} words against {}", x.len(), y.len()));
        }
        let diff: Vec<usize> = (0..x.len()).filter(|&i| x[i] != y[i]).collect();
        if !diff.is_empty() {
            let at: Vec<String> =
                diff.iter().take(4).map(|&i| format!("[{i}] {:#x} / {:#x}", x[i], y[i])).collect();
            return Some(format!("{name}: {} of {} words differ: {}", diff.len(), x.len(), at.join(", ")));
        }
    }
    None
}

/// The modules a fleet's host loads (as `CudaExplicitBeams::with_coupling` picks them).
fn modules_for(fleet: &[ExplicitBeams]) -> Vec<&'static str> {
    use engine_exec::explicit_plan::{ExplicitUpload, Kernel, VV_PASSES, WIDE_PASSES};
    let l = ExplicitUpload::new(fleet, None).layout;
    let mut k = vec![Kernel::Frame, Kernel::Solo, Kernel::Contact, Kernel::ContactBig];
    k.extend((0..VV_PASSES.len()).map(Kernel::Vv));
    if l.wide_ok {
        k.extend((0..WIDE_PASSES.len()).map(Kernel::Wide));
    }
    let mut names: Vec<&'static str> = k.into_iter().map(|k| k.module_for(l.aero, l.lanes, l.surface)).collect();
    names.dedup();
    names
}

fn main() -> anyhow::Result<()> {
    let spec = spec()?;
    // `--modules` (no GPU): the modules this run loads, comma-separated, for
    // ENGINE_CUDA_EXPLICIT_MODULES (a build of only those is minutes, not an hour).
    if std::env::args().any(|a| a == "--modules") {
        let base = car(spec["car"].as_str().unwrap_or("procedural"))?;
        let fleet = scenario(
            spec["scenario"].as_str().unwrap_or("parked"),
            &base,
            spec["count"].as_u64().map(|c| c as usize),
        )?;
        println!("{}", modules_for(&fleet).join(","));
        return Ok(());
    }
    anyhow::ensure!(
        CudaExplicitBeams::available(),
        "no CUDA device or no embedded device code (build with nvcc; run on an NVIDIA GPU)"
    );
    let get = |k: &str| spec.get(k).cloned().unwrap_or(Value::Null);
    let name = get("scenario").as_str().unwrap_or("parked").to_string();
    let car_kind = get("car").as_str().unwrap_or("procedural").to_string();
    let frames = get("frames").as_u64().unwrap_or(30) as u32;
    let steps = get("steps").as_u64().unwrap_or(17) as u32;
    let at = get("at").as_f64().unwrap_or(0.0);
    let count = get("count").as_u64().map(|c| c as usize);
    let variants = get("variants").as_array().cloned().unwrap_or_default();
    anyhow::ensure!(!variants.is_empty(), "no variants");

    let base = car(&car_kind)?;
    let fleet = scenario(&name, &base, count)?;
    let mut cpu = ExplicitBeamsWorld::new(fleet);
    let prefix = (at * 2000.0 / f64::from(steps)).round() as u32;
    for _ in 0..prefix {
        cpu.frame(steps);
    }
    let start = cpu.vehicles.clone();
    println!(
        "scenario {name}: {} × {car_kind} ({} nodes, {} beams each), from t = {:.3} s, {frames} frames of {steps} steps",
        start.len(),
        start[0].nodes.len(),
        start[0].beams.len(),
        f64::from(prefix * steps) / 2000.0
    );
    let t_cpu = std::time::Instant::now();
    for _ in 0..frames {
        cpu.frame(steps);
    }
    println!("CPU reference: {:.2} s", t_cpu.elapsed().as_secs_f64());

    let mut first: Option<Vec<(&'static str, Vec<u32>)>> = None;
    let mut ok = true;
    for v in &variants {
        let vname = v.get("name").and_then(Value::as_str).unwrap_or("?").to_string();
        let switches = v.get("switches").and_then(Value::as_object).cloned().unwrap_or_default();
        let mut g = host(&start, &switches)?;
        let snap = g.snapshot()?;
        let t = std::time::Instant::now();
        for _ in 0..frames {
            g.frame(steps)?;
        }
        g.wait()?;
        let wall = t.elapsed().as_secs_f64() * 1e3 / f64::from(frames);
        let words = g.state_words()?;
        // Again from the snapshot, timed: GPU time a frame, and per kernel.
        g.restore(&snap)?;
        g.enable_frame_timing(frames);
        g.set_phase_timing(true);
        for _ in 0..frames {
            g.frame(steps)?;
        }
        let ft = g.frame_times_ms()?;
        let phases = g.phase_times_ms()?;
        g.set_phase_timing(false);
        let again = g.state_words()?;
        let repeat = first_diff(&words, &again);
        // Against the CPU reference.
        let gp = g.positions()?;
        let gb = g.beams()?;
        let mut worst = (0.0f32, 0.0f32);
        let mut damage = Vec::new();
        let mut within = true;
        for (k, veh) in cpu.vehicles.iter().enumerate() {
            let (rms, max) = rms_max(&veh.positions(), &gp[k]);
            worst = (worst.0.max(rms), worst.1.max(max));
            let (cb, cbent) = veh.damage();
            let gbk = gb[k].iter().filter(|b| b.broken).count();
            let gbent = gb[k].iter().filter(|b| b.bent()).count();
            let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.15 * a.max(b) as f32 + 3.0;
            within &= close(cb, gbk) && close(cbent, gbent);
            damage.push(format!("{cb}/{gbk} broken, {cbent}/{gbent} bent"));
        }
        // Rest: positions within the parity test's tolerance; motion: damage counts.
        if name == "parked" {
            within &= worst.0 < 3e-3 && worst.1 < 1e-2;
        }
        ok &= within && repeat.is_none();
        let frame_ms = ft.iter().sum::<f64>() / ft.len().max(1) as f64;
        println!(
            "[{vname}] {switches:?}\n  CPU vs GPU: position rms {:.3} mm, max {:.3} mm; damage CPU/GPU per car: {}; within tolerance: {within}",
            worst.0 * 1e3,
            worst.1 * 1e3,
            damage.join("; ")
        );
        println!(
            "  timing: {frame_ms:.3} ms GPU a frame (events), {wall:.3} ms wall a frame; rerun from snapshot bit for bit: {}",
            repeat.as_deref().unwrap_or("yes")
        );
        println!(
            "  per kernel (ms over {frames} frames, launches): {}",
            phases.iter().map(|(k, ms, n)| format!("{k} {ms:.2}×{n}")).collect::<Vec<_>>().join(", ")
        );
        match &first {
            None => first = Some(words),
            Some(f) => match first_diff(f, &words) {
                None => println!("  bit for bit against [{}]: yes (every state word)", variants[0]["name"].as_str().unwrap_or("a")),
                Some(d) => {
                    ok = false;
                    println!("  bit for bit against the first variant: NO, {d}");
                }
            },
        }
    }
    println!("verdict: {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}

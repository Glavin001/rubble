//! When and why a JBeam car's break groups break on the vehicle lab's mountain (the
//! page's `vehicle-lab.html?mode=mountain&…`), on the CPU reference
//! (`vehicle_lab::cpu`; no GPU). For every break group the step it broke at, the beam
//! that set it off (the one past its strength), that beam's force, stretch, length and
//! strength, the group's other beams, and what the panel the group holds was doing
//! then: its own beams yielded, their largest plastic strain, broken. Then when each
//! named part left the car's main body (no unbroken beam joins them) and the share of
//! the car's mass still on it over the run.
//!
//! ```text
//! cargo run --release -p engine-world --example hinge_breaks -- \
//!   [--car DIR] [--secs 30] [--seed 1] [--steepness 55] [--roughness 0.25] \
//!   [--launch 12] [--surface 0] [--drag 0] [--groups hinge,latch] [--json FILE]
//! ```
//! `--car`: a JBeam export (a directory with `files.json`; default the lab's generated
//! sedan, `~/Development/rubble-artifacts/sedan-live/mods/sedan_new`). The other
//! options are the page's (rockslide off, debris on as the page). `--groups`: name
//! fragments of the groups reported in full (default `hinge,latch`; every group's
//! break time is listed). Measurement doc: `engine/docs/measurements/hinge-breaks.md`.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use rubble::vehicle_lab::cpu::CpuLab;
use rubble::vehicle_lab::LabOptions;
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn num(k: &str, d: f64) -> f64 {
    arg(k).and_then(|x| x.parse().ok()).unwrap_or(d)
}

/// One break group: its beams (indices into the solver's beams) and the part it holds.
struct Group {
    name: String,
    beams: Vec<usize>,
    /// Original strength per beam (N), from the JBeam.
    strength0: Vec<f32>,
    /// The named part most of the group's nodes are in (not the body), if any.
    part: Option<usize>,
    broke: Option<Broke>,
    /// The largest |force| / strength any of its beams reached while intact.
    peak_ratio: f32,
}

#[derive(Clone)]
struct BeamAt {
    beam: usize,
    a: String,
    b: String,
    force: f32,
    strength: f32,
    strength0: f32,
    rest0: f32,
    rest: f32,
    len: f32,
    k: f32,
    deform: f32,
}

#[derive(Clone)]
struct Broke {
    t: f64,
    step: u64,
    /// The beam past its strength (largest |force| / strength); the rest went with it.
    trigger: BeamAt,
    others: Vec<BeamAt>,
    /// The panel's own beams then: (beams, yielded, broken, largest plastic strain).
    panel: Option<(u32, u32, u32, f32)>,
    /// The car's centre speed then (m/s).
    speed: f32,
}

fn main() -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = arg("--car").unwrap_or(format!(
        "{home}/Development/rubble-artifacts/sedan-live/mods/sedan_new"
    ));
    let dir = std::path::Path::new(&dir);
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
    let drag = num("--drag", 0.0) != 0.0;
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag,
        turbulence: drag,
        ..Default::default()
    };
    let desc =
        engine_softbody::jbeam::load_jbeam(&files, pc.as_deref()).map_err(anyhow::Error::msg)?;
    let (template, _) = rubble::showcase_scene::jbeam_vehicle_on(
        &files,
        pc.as_deref(),
        settings,
        JbeamWheels::Raycast,
    )?;
    // The solver keeps the desc's node order and beam order (desc beams first).
    for (i, b) in desc.beams.iter().enumerate() {
        let s = &template.beams[i];
        anyhow::ensure!(
            (s.a, s.b) == (b.a, b.b),
            "beam {i}: the solver's order is not the JBeam's"
        );
    }
    let secs = num("--secs", 30.0);
    let surface = num("--surface", 0.0) != 0.0;
    let script = json!({
        "scene": "mountain",
        "surface": surface,
        "mountain": {
            "seed": num("--seed", 1.0) as u64,
            "steepness": num("--steepness", 55.0),
            "roughness": num("--roughness", 0.25),
            "launch": num("--launch", 12.0),
            "loose": 0,
            "start": 0.0,
            "rockslide": false,
        },
        "events": [],
    });
    let opts = LabOptions::from_json(&script);
    let mut lab = CpuLab::new(&template, opts)?;
    lab.bench.set_script(&script)?;
    anyhow::ensure!(
        lab.sim.beams.len() >= desc.beams.len() && lab.sim.nodes.len() >= desc.nodes.len(),
        "the lab's car is not the JBeam's"
    );

    // Groups by name (solver group = desc break group index + 1; couplers too).
    let parts = &desc.parts;
    let mut part_of: Vec<Vec<usize>> = vec![Vec::new(); lab.sim.nodes.len()];
    for (p, part) in parts.iter().enumerate() {
        for &n in &part.nodes {
            if let Some(v) = part_of.get_mut(n as usize) {
                v.push(p);
            }
        }
    }
    let is_body = |name: &str| name.ends_with("_body") || name == "body";
    let mut by_index: BTreeMap<usize, Group> = BTreeMap::new();
    for (j, b) in lab.sim.beams.iter().enumerate() {
        if b.group == 0 {
            continue;
        }
        let g = b.group.unsigned_abs() as usize;
        let Some(name) = desc.break_groups.get(g - 1) else {
            continue;
        };
        let e = by_index.entry(g).or_insert_with(|| Group {
            name: name.clone(),
            beams: Vec::new(),
            strength0: Vec::new(),
            part: None,
            broke: None,
            peak_ratio: 0.0,
        });
        e.beams.push(j);
        e.strength0.push(b.strength);
    }
    for g in by_index.values_mut() {
        let mut count: BTreeMap<usize, u32> = BTreeMap::new();
        for &j in &g.beams {
            let b = &lab.sim.beams[j];
            for n in [b.a, b.b] {
                for &p in &part_of[n as usize] {
                    if !is_body(&parts[p].name) {
                        *count.entry(p).or_default() += 1;
                    }
                }
            }
        }
        g.part = count.into_iter().max_by_key(|&(_, c)| c).map(|(p, _)| p);
    }
    let detail: Vec<String> = arg("--groups")
        .unwrap_or("hinge,latch".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let watched = |name: &str| detail.iter().any(|d| name.contains(d.as_str()));
    // Per part: its own beams (both ends in it).
    let part_beams: Vec<Vec<usize>> = parts
        .iter()
        .map(|p| {
            let set: std::collections::HashSet<u32> = p.nodes.iter().copied().collect();
            lab.sim
                .beams
                .iter()
                .enumerate()
                .filter(|(_, b)| b.group == 0 && set.contains(&b.a) && set.contains(&b.b))
                .map(|(j, _)| j)
                .collect()
        })
        .collect();
    let node_name = |n: u32| -> String {
        desc.nodes
            .get(n as usize)
            .map_or(format!("#{n}"), |x| x.name.clone())
    };

    let mut step: u64 = 0;
    let mut t = 0.0f64;
    let mut speed = 0.0f32;
    let h = engine_softbody::explicit_beams::STEP as f64;
    let windows = (secs * 60.0).round() as u64;
    // Per part: when it left the main body; and the attached share over time.
    let mut left: Vec<Option<f64>> = vec![None; parts.len()];
    let mut attached: Vec<(f64, f32)> = Vec::new();
    let mass: Vec<f32> = lab.sim.nodes.iter().map(|n| n.mass).collect();
    let t0 = std::time::Instant::now();
    for w in 0..windows {
        let groups = &mut by_index;
        let mut observe = |sim: &ExplicitBeams| {
            step += 1;
            t += h;
            for g in groups.values_mut() {
                if g.broke.is_some() {
                    continue;
                }
                let at = |j: usize, s0: f32| {
                    let b = &sim.beams[j];
                    let (pa, pb) = (sim.nodes[b.a as usize].pos, sim.nodes[b.b as usize].pos);
                    BeamAt {
                        beam: j,
                        a: node_name(b.a),
                        b: node_name(b.b),
                        force: b.force,
                        strength: b.strength,
                        strength0: s0,
                        rest0: b.rest0,
                        rest: b.rest,
                        len: (pa - pb).length(),
                        k: b.k,
                        deform: b.tt.min(b.tc),
                    }
                };
                if g.beams.iter().any(|&j| sim.beams[j].broken) {
                    let all: Vec<BeamAt> = g
                        .beams
                        .iter()
                        .zip(&g.strength0)
                        .map(|(&j, &s0)| at(j, s0))
                        .collect();
                    let ratio = |x: &BeamAt| x.force.abs() / x.strength.max(1e-6);
                    let (ti, _) = all
                        .iter()
                        .enumerate()
                        .max_by(|a, b| ratio(a.1).total_cmp(&ratio(b.1)))
                        .expect("a group has beams");
                    let panel = g.part.map(|p| {
                        let mut y = (0u32, 0u32, 0u32, 0.0f32);
                        for &j in &part_beams[p] {
                            let b = &sim.beams[j];
                            y.0 += 1;
                            if b.broken {
                                y.2 += 1;
                            }
                            let strain = (b.rest / b.rest0 - 1.0).abs();
                            if strain > 1e-5 {
                                y.1 += 1;
                            }
                            y.3 = y.3.max(strain);
                        }
                        y
                    });
                    g.broke = Some(Broke {
                        t,
                        step,
                        trigger: all[ti].clone(),
                        others: all
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| i != ti)
                            .map(|(_, x)| x.clone())
                            .collect(),
                        panel,
                        speed,
                    });
                } else {
                    for &j in &g.beams {
                        let b = &sim.beams[j];
                        g.peak_ratio = g.peak_ratio.max(b.force.abs() / b.strength.max(1e-6));
                    }
                }
            }
        };
        lab.step_observed(&mut observe)?;
        speed = lab.bench.car.speed;
        if w % 15 == 14 || w + 1 == windows {
            // Pieces by unbroken beams; the main body is the heaviest.
            let n = lab.sim.nodes.len();
            let mut parent: Vec<u32> = (0..n as u32).collect();
            fn find(p: &mut [u32], mut x: u32) -> u32 {
                while p[x as usize] != x {
                    p[x as usize] = p[p[x as usize] as usize];
                    x = p[x as usize];
                }
                x
            }
            for b in &lab.sim.beams {
                if !b.broken && (b.a as usize) < n && (b.b as usize) < n {
                    let (ra, rb) = (find(&mut parent, b.a), find(&mut parent, b.b));
                    if ra != rb {
                        parent[ra as usize] = rb;
                    }
                }
            }
            let mut piece: BTreeMap<u32, f32> = BTreeMap::new();
            for i in 0..n {
                let r = find(&mut parent, i as u32);
                *piece.entry(r).or_default() += mass[i];
            }
            let (main, main_mass) = piece
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(&r, &m)| (r, m))
                .unwrap();
            let total: f32 = mass.iter().sum();
            attached.push((t, main_mass / total));
            for (p, part) in parts.iter().enumerate() {
                if left[p].is_some() || part.nodes.is_empty() {
                    continue;
                }
                let (mut on, mut all) = (0.0f32, 0.0f32);
                for &i in &part.nodes {
                    let m = mass[i as usize];
                    all += m;
                    if find(&mut parent, i) == main {
                        on += m;
                    }
                }
                if on < 0.5 * all {
                    left[p] = Some(t);
                }
            }
        }
    }
    let wall = t0.elapsed().as_secs_f64();

    println!(
        "mountain on the CPU reference: {} ({} nodes, {} beams), seed {}, steepness {}, roughness {}, launch {} m/s, surface {}, drag {}; {secs} s in {wall:.0} s",
        dir.display(),
        lab.sim.nodes.len(),
        lab.sim.beams.len(),
        num("--seed", 1.0),
        num("--steepness", 55.0),
        num("--roughness", 0.25),
        num("--launch", 12.0),
        surface as u8,
        drag as u8
    );
    let mut rows: Vec<&Group> = by_index.values().collect();
    rows.sort_by(|a, b| {
        let ta = a.broke.as_ref().map_or(f64::INFINITY, |x| x.t);
        let tb = b.broke.as_ref().map_or(f64::INFINITY, |x| x.t);
        ta.total_cmp(&tb).then(a.name.cmp(&b.name))
    });
    let mut out = Vec::new();
    println!(
        "\nbreak groups ({} of {} broke):",
        rows.iter().filter(|g| g.broke.is_some()).count(),
        rows.len()
    );
    for g in &rows {
        let part = g.part.map_or("-".to_string(), |p| parts[p].name.clone());
        match &g.broke {
            None => {
                if watched(&g.name) {
                    println!(
                        "  {:<24} intact            (peak |F|/S {:.2}; part {part})",
                        g.name, g.peak_ratio
                    );
                }
                out.push(json!({ "group": g.name, "part": part, "broke": Value::Null, "peakRatio": g.peak_ratio }));
            }
            Some(b) => {
                let x = &b.trigger;
                let stretch = x.len - x.rest;
                println!(
                    "  {:<24} t {:6.3} s  by {}-{}: |F| {:6.1} kN of S {:5.1} kN (JBeam {:5.1}), {} {:5.1} mm (L0 {:.3} m, now {:.3} m), k {:.2e}, deform {}; part {part}{}",
                    g.name,
                    b.t,
                    x.a,
                    x.b,
                    x.force.abs() / 1e3,
                    x.strength / 1e3,
                    x.strength0 / 1e3,
                    if stretch >= 0.0 { "stretch" } else { "squeeze" },
                    stretch.abs() * 1e3,
                    x.rest0,
                    x.len,
                    x.k,
                    if x.deform.is_finite() && x.deform < 1e30 { format!("{:.1} kN", x.deform / 1e3) } else { "none".into() },
                    b.panel.map_or(String::new(), |(n, y, br, s)| format!(
                        "; panel {y}/{n} beams yielded, {br} broken, largest plastic strain {:.1} %",
                        100.0 * s
                    ))
                );
                if watched(&g.name) {
                    for o in &b.others {
                        println!(
                            "      {:<12}-{:<12} |F| {:6.1} kN of {:5.1} kN, L0 {:.3} m, {:+6.1} mm",
                            o.a,
                            o.b,
                            o.force.abs() / 1e3,
                            o.strength / 1e3,
                            o.rest0,
                            (o.len - o.rest) * 1e3
                        );
                    }
                }
                let beam = |x: &BeamAt| json!({ "beam": x.beam, "a": x.a, "b": x.b, "force": x.force, "strength": x.strength, "strength0": x.strength0, "rest0": x.rest0, "rest": x.rest, "len": x.len, "k": x.k });
                out.push(json!({
                    "group": g.name, "part": part, "t": b.t, "step": b.step, "speed": b.speed,
                    "trigger": beam(x), "others": b.others.iter().map(beam).collect::<Vec<_>>(),
                    "panel": b.panel.map(|(n, y, br, s)| json!({ "beams": n, "yielded": y, "broken": br, "plasticStrainMax": s })),
                }));
            }
        }
    }
    println!("\nparts that left the main body:");
    for (p, part) in parts.iter().enumerate() {
        if let Some(t) = left[p] {
            println!("  {:<28} t {t:6.2} s", part.name);
        }
    }
    println!("\nattached share of the car's mass:");
    for (t, a) in attached.iter().step_by(4) {
        println!("  t {t:6.2} s  {:5.1} %", 100.0 * a);
    }
    let stats = lab.bench.stats();
    println!(
        "\nend: beams broken {}, yielded {}",
        stats["car"]["beams"]["broken"], stats["car"]["beams"]["yielded"]
    );
    if let Some(path) = arg("--json") {
        let v = json!({
            "car": dir.display().to_string(),
            "secs": secs,
            "groups": out,
            "left": parts.iter().enumerate().filter_map(|(p, x)| left[p].map(|t| json!({ "part": x.name, "t": t }))).collect::<Vec<_>>(),
            "attached": attached,
            "stats": stats,
        });
        if let Some(d) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(&v)?)?;
        println!("wrote {path}");
    }
    Ok(())
}

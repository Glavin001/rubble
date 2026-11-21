//! How much of a vehicle needs the explicit solver's 2 kHz step: each beam's and each
//! node's own stability limit, split by step rate, part, beam kind and place
//! (`engine/docs/measurements/beam-rate-histogram.md`). A measurement on the spawn state:
//! nothing is stepped, the solver is not changed, no GPU.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example beam_rate_histogram -- \
//!     [--car <jbeam export dir with files.json>] [--generated low|medium|high] [--worst]
//! ```
//! With no argument: the repo's generated low-resolution sedan (`authoring::build`, as
//! `engine-webgpu/tests/group_split_gpu.rs` spawns it, on its soft tyres). `--car <dir>`:
//! a JBeam export as the vehicle lab's `--car <dir>` loads it (JBeam mode, raycast wheels).
//!
//! The limits (the solver's step is `explicit_beams::STEP`; its node update is
//! v += h·F/m, then x += h·v, the forces from the pass before):
//! * one mode of stiffness k, damping c on mass m: ω = √(k/m), ζ = c / (2·√(k·m));
//! * undamped: h < 2/ω;
//! * damped, the damping read from the step's start velocity (the Rigs of Rods profile):
//!   ω·h < 2·(√(1 + ζ²) − ζ);
//! * damped, centred damping (the BeamNG profile's JBeam kinds, jbeam-semantics-spec.md
//!   §39): the smallest Ω = ω·h with (1 − ζΩ)·(Ω² + 4ζΩ) ≥ 4, and ζΩ < 1 (c·h/m < 2);
//! * a damper alone (k = 0): c·h/m < 2.
//! A beam alone: m the reduced mass of its two nodes. A node: k and c summed over its
//! beams, m its own mass (its neighbours held); the Gershgorin bound on the whole
//! vehicle doubles that k. The largest ω of the assembled vehicle at rest is found by
//! power iteration, and the largest set of nodes that could step slower by removing the
//! nodes of the stiffest mode until what is left (the removed nodes held) is slow enough.
//!
//! Rates are each beam's at rest: a bounded beam (a shock, a tyre spoke) inside its
//! bounds. `--worst`: past its bounds, at its limit spring and damping.

use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::desc::BeamKind;
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, ExplicitType, JbeamWheels, NODE_IMMOVABLE,
    NODE_TYRE, STEP,
};
use glam::{DVec3, Vec3};
use std::collections::BTreeMap;

/// Step rates a part could run at (Hz), fastest first.
const RATES: [f64; 4] = [2000.0, 1000.0, 500.0, 250.0];
const CLASS_NAMES: [&str; 5] = [
    "under margin at 2 kHz",
    "2 kHz",
    "1 kHz",
    "500 Hz",
    "250 Hz or less",
];
const SAFETY: [f64; 3] = [1.0, 2.0, 4.0];

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

/// The largest stable ω·h of one damped mode (module docs).
fn omega_h_limit(zeta: f64, centred: bool) -> f64 {
    if !centred {
        return 2.0 * ((1.0 + zeta * zeta).sqrt() - zeta);
    }
    if zeta <= 0.0 {
        return 2.0;
    }
    let g = |o: f64| (1.0 - zeta * o) * (o * o + 4.0 * zeta * o) - 4.0;
    let top = (1.0 / zeta).min(8.0);
    let n = 4000;
    let mut lo = 0.0;
    for i in 1..=n {
        let o = top * i as f64 / n as f64;
        if g(o) >= 0.0 {
            let (mut a, mut b) = (lo, o);
            for _ in 0..60 {
                let m = 0.5 * (a + b);
                if g(m) >= 0.0 {
                    b = m;
                } else {
                    a = m;
                }
            }
            return a;
        }
        lo = o;
    }
    top
}

/// The largest stable step (s) of stiffness `k`, damping `c` on mass `m`; `damped` off: c
/// ignored.
fn h_limit(k: f64, c: f64, m: f64, damped: bool, centred: bool) -> f64 {
    if !m.is_finite() {
        return f64::INFINITY;
    }
    let c = if damped { c } else { 0.0 };
    if k <= 0.0 {
        return if c > 0.0 { 2.0 * m / c } else { f64::INFINITY };
    }
    let w = (k / m).sqrt();
    omega_h_limit(c / (2.0 * (k * m).sqrt()), centred) / w
}

/// 0: under the margin even at 2 kHz; 1..=4: the slowest of `RATES` the limit allows.
fn class(h: f64, safety: f64) -> usize {
    let mut c = 0;
    for (i, r) in RATES.iter().enumerate() {
        if h >= safety / r {
            c = i + 1;
        }
    }
    c
}

/// A beam record as a linear spring–damper: its stiffest state's k and c.
#[derive(Clone, Copy)]
struct Spring {
    a: usize,
    b: usize,
    k: f64,
    c: f64,
    /// Its damping takes the centred correction.
    centred: bool,
    /// Acts in every direction (a coupler), not along the beam.
    isotropic: bool,
    record: usize,
}

struct Car {
    title: String,
    sim: ExplicitBeams,
    node_name: Vec<String>,
    node_part: Vec<String>,
    /// Per beam record: its kind's label, its break group's name ("" none).
    beam_kind: Vec<String>,
    beam_group: Vec<String>,
    /// Per beam record a second stiffness it may take (anisotropic beams' expansion
    /// spring; 0 none).
    beam_k2: Vec<f64>,
}

fn kind_label(sim: &ExplicitBeams, j: usize, desc_kind: Option<BeamKind>) -> String {
    let b = &sim.beams[j];
    let tyre = |i: u32| sim.nodes[i as usize].has(NODE_TYRE);
    let wheel = match (tyre(b.a), tyre(b.b)) {
        (true, true) => " (tyre: tread to tread)",
        (true, false) | (false, true) => " (tyre: tread to rim)",
        _ => "",
    };
    let base = match (b.kind, desc_kind) {
        (_, Some(BeamKind::Hydro)) => "hydro".into(),
        (ExplicitKind::Plain | ExplicitKind::JbeamPlain, _) if b.ty == ExplicitType::Virtual => {
            "wheel rigidity (virtual)".into()
        }
        (ExplicitKind::Plain | ExplicitKind::JbeamPlain, _) if b.k == 0.0 => "damper only".into(),
        (ExplicitKind::Plain | ExplicitKind::JbeamPlain, _) => "normal".into(),
        (ExplicitKind::Support | ExplicitKind::JbeamSupport, _) => "support".into(),
        (ExplicitKind::LimitedSpring | ExplicitKind::JbeamBounded | ExplicitKind::Shock2, _) => {
            "bounded (shock, limiter)".into()
        }
        (ExplicitKind::Rope | ExplicitKind::JbeamRope, _) => "rope".into(),
        (ExplicitKind::JbeamAniso, _) => "anisotropic".into(),
        (ExplicitKind::JbeamCoupler, _) => "coupler".into(),
        (k, _) => format!("{k:?}"),
    };
    format!("{base}{wheel}")
}

fn generated(res: Resolution) -> Car {
    let def = style("sedan").expect("sedan style");
    let built = build(
        &def,
        &BuildParams {
            resolution: res,
            ..BuildParams::default()
        },
    )
    .expect("build");
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let sim =
        ExplicitBeams::from_truck_config(&built.truck, None, Vec3::ZERO, Vec3::ZERO, settings)
            .expect("spawn");
    let n = sim.nodes.len();
    // A node's part: the smallest named group (member, block or corner) holding it.
    let mut part = vec![String::new(); n];
    let mut size = vec![usize::MAX; n];
    let mut groups: Vec<(&String, &Vec<u32>)> = built.groups.iter().collect();
    groups.sort();
    for (name, nodes) in groups {
        for &i in nodes {
            let i = i as usize;
            if i < n && nodes.len() < size[i] {
                size[i] = nodes.len();
                part[i] = part_family(name);
            }
        }
    }
    for (i, p) in part.iter_mut().enumerate() {
        if p.is_empty() {
            *p = if sim.nodes[i].has(NODE_TYRE) {
                "tyres".into()
            } else {
                "(no group)".into()
            };
        }
    }
    let beam_kind = (0..sim.beams.len())
        .map(|j| kind_label(&sim, j, None))
        .collect();
    Car {
        title: format!("repo generated sedan, {res:?} resolution (`.truck`, soft tyres)"),
        node_name: (0..n).map(|i| format!("n{i}")).collect(),
        node_part: part,
        beam_group: sim
            .beams
            .iter()
            .map(|b| {
                if b.group == 0 {
                    String::new()
                } else {
                    format!("g{}", b.group.abs())
                }
            })
            .collect(),
        beam_kind,
        beam_k2: vec![0.0; sim.beams.len()],
        sim,
    }
}

/// A group name without its trailing index or side, so the table stays short.
fn part_family(name: &str) -> String {
    let t = name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '_' || c == '.');
    if t.is_empty() {
        name.to_string()
    } else {
        t.to_string()
    }
}

fn jbeam_car(dir: &std::path::Path) -> Car {
    let list: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("files.json")).expect("files.json"))
            .expect("files.json");
    let mut texts = Vec::new();
    let mut pc = None;
    for f in &list {
        if f.ends_with(".jbeam") {
            texts.push((
                f.clone(),
                std::fs::read_to_string(dir.join(f)).expect("jbeam"),
            ));
        } else if f.ends_with(".pc") {
            pc = Some(std::fs::read_to_string(dir.join(f)).expect("pc"));
        }
    }
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let desc = engine_softbody::jbeam::load_jbeam(&files, pc.as_deref()).expect("load");
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        ..Default::default()
    };
    let wheels = arg("--wheels").unwrap_or_else(|| "raycast".into());
    let (sim, _) = match wheels.as_str() {
        "pressure" => ExplicitBeams::from_desc_jbeam_with(
            &desc,
            JbeamWheels::Pressure,
            Vec3::ZERO,
            Vec3::ZERO,
            settings,
        ),
        _ => ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings),
    }
    .expect("spawn");
    let n = sim.nodes.len();
    let mut part = vec![String::new(); n];
    let mut size = vec![usize::MAX; n];
    for p in &desc.parts {
        for &i in &p.nodes {
            let i = i as usize;
            if i < n && p.nodes.len() < size[i] {
                size[i] = p.nodes.len();
                part[i] = p.name.clone();
            }
        }
    }
    for (i, p) in part.iter_mut().enumerate() {
        if p.is_empty() {
            *p = if i >= desc.nodes.len() {
                "wheels (spawned)".into()
            } else {
                "(no part)".into()
            };
        }
    }
    let beam_kind = (0..sim.beams.len())
        .map(|j| kind_label(&sim, j, desc.beams.get(j).map(|b| b.kind)))
        .collect();
    let beam_group = sim
        .beams
        .iter()
        .map(|b| {
            let g = b.group.unsigned_abs() as usize;
            if g == 0 {
                String::new()
            } else {
                desc.break_groups
                    .get(g - 1)
                    .cloned()
                    .unwrap_or_else(|| format!("g{g}"))
            }
        })
        .collect();
    Car {
        title: format!("{} (JBeam mode, {wheels} wheels)", dir.display()),
        node_name: (0..n)
            .map(|i| {
                desc.nodes
                    .get(i)
                    .map(|x| x.name.clone())
                    .unwrap_or_else(|| format!("n{i}"))
            })
            .collect(),
        node_part: part,
        beam_kind,
        beam_group,
        beam_k2: (0..sim.beams.len())
            .map(|j| {
                desc.beams
                    .get(j)
                    .and_then(|b| b.jbeam)
                    .and_then(|j| j.spring_expansion)
                    .unwrap_or(0.0) as f64
            })
            .collect(),
        sim,
    }
}

/// The records that are linear spring–dampers between two nodes, and the count of each
/// kind left out (angle, pressure and aerodynamic records, triggers, second records).
fn springs(car: &Car, worst: bool) -> (Vec<Spring>, BTreeMap<String, usize>) {
    let s = &car.sim;
    let centred_on = s.settings.centred_damping;
    let mut out = Vec::new();
    let mut skipped = BTreeMap::new();
    for (j, b) in s.beams.iter().enumerate() {
        if b.broken {
            continue;
        }
        let (mut k, mut c) = (b.k as f64, b.c as f64);
        let fin = |x: f32| if x.is_finite() { x as f64 } else { 0.0 };
        match b.kind {
            ExplicitKind::Plain
            | ExplicitKind::JbeamPlain
            | ExplicitKind::Support
            | ExplicitKind::JbeamSupport
            | ExplicitKind::Rope
            | ExplicitKind::JbeamRope
            | ExplicitKind::JbeamCoupler => {}
            // The stiffer of its compression and expansion springs.
            ExplicitKind::JbeamAniso => k = k.max(car.beam_k2[j]),
            // Past its bounds a limited spring moves to its limit rates (`--worst`).
            ExplicitKind::LimitedSpring => {
                if worst {
                    k = k.max(fin(b.limit_k));
                    c = c.max(fin(b.limit_c));
                }
            }
            ExplicitKind::JbeamBounded => {
                // `tt`: the fast damping past the velocity split.
                c = c.max(fin(b.tt));
                if worst {
                    k = k.max(fin(b.limit_k));
                    c = c.max(fin(b.limit_c));
                }
            }
            ExplicitKind::Shock2 => {
                if let Some(sh) = s.shocks2.get(b.shock as usize) {
                    k = k.max(sh.k_in as f64).max(sh.k_out as f64);
                    c = c.max(sh.c_in as f64).max(sh.c_out as f64);
                    if worst {
                        k = k.max(sh.k_bump as f64);
                        c = c.max(sh.c_bump as f64);
                    }
                }
            }
            other => {
                *skipped.entry(format!("{other:?}")).or_insert(0) += 1;
                continue;
            }
        }
        if b.a == b.b || (k == 0.0 && c == 0.0) {
            *skipped
                .entry(format!("{:?} with no rate", b.kind))
                .or_insert(0) += 1;
            continue;
        }
        out.push(Spring {
            a: b.a as usize,
            b: b.b as usize,
            k,
            c,
            centred: centred_on && b.kind.is_jbeam(),
            isotropic: b.kind == ExplicitKind::JbeamCoupler,
            record: j,
        });
    }
    (out, skipped)
}

fn mass(s: &ExplicitBeams, i: usize) -> f64 {
    if s.nodes[i].has(NODE_IMMOVABLE) {
        f64::INFINITY
    } else {
        s.nodes[i].mass as f64
    }
}

/// The largest ω (rad/s) of the vehicle at rest with the nodes of `free` moving and the
/// others held, by power iteration on M⁻¹K (K = Σ k·d̂d̂ᵀ per beam); and each node's share
/// of that mode's kinetic energy.
fn omega_max(s: &ExplicitBeams, sp: &[Spring], free: &[bool], iters: usize) -> (f64, Vec<f64>) {
    let n = s.nodes.len();
    let dirs: Vec<DVec3> = sp
        .iter()
        .map(|b| {
            (s.nodes[b.a].pos - s.nodes[b.b].pos)
                .as_dvec3()
                .normalize_or_zero()
        })
        .collect();
    let m: Vec<f64> = (0..n).map(|i| mass(s, i)).collect();
    let live: Vec<bool> = (0..n)
        .map(|i| free[i] && m[i].is_finite() && m[i] > 0.0)
        .collect();
    if !live.iter().any(|&x| x) {
        return (0.0, vec![0.0; n]);
    }
    // A fixed pseudo-random start.
    let mut seed = 0x9e37_79b9_u32;
    let mut x: Vec<DVec3> = (0..n)
        .map(|i| {
            let mut r = || {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f64 / (1u32 << 24) as f64 - 0.5
            };
            if live[i] {
                DVec3::new(r(), r(), r())
            } else {
                DVec3::ZERO
            }
        })
        .collect();
    let mut lambda = 0.0;
    for _ in 0..iters {
        let mut y = vec![DVec3::ZERO; n];
        for (b, d) in sp.iter().zip(&dirs) {
            let rel = x[b.a] - x[b.b];
            let f = if b.isotropic {
                rel * b.k
            } else {
                *d * (b.k * d.dot(rel))
            };
            y[b.a] += f;
            y[b.b] -= f;
        }
        let (mut num, mut den, mut norm) = (0.0, 0.0, 0.0);
        for i in 0..n {
            if !live[i] {
                y[i] = DVec3::ZERO;
                continue;
            }
            num += x[i].dot(y[i]);
            den += m[i] * x[i].length_squared();
            y[i] /= m[i];
            norm += m[i] * y[i].length_squared();
        }
        lambda = num / den;
        let inv = 1.0 / norm.sqrt().max(1e-300);
        for i in 0..n {
            x[i] = y[i] * inv;
        }
    }
    let total: f64 = (0..n)
        .filter(|&i| live[i])
        .map(|i| m[i] * x[i].length_squared())
        .sum();
    let share = (0..n)
        .map(|i| {
            if live[i] {
                m[i] * x[i].length_squared() / total
            } else {
                0.0
            }
        })
        .collect();
    (lambda.max(0.0).sqrt(), share)
}

fn pct(n: usize, of: usize) -> String {
    if of == 0 {
        "–".into()
    } else {
        format!("{:.1} %", 100.0 * n as f64 / of as f64)
    }
}

fn pctf(x: f64, of: f64) -> String {
    if of <= 0.0 {
        "–".into()
    } else {
        format!("{:.1} %", 100.0 * x / of)
    }
}

fn share_row(label: &str, classes: &[usize]) -> String {
    let mut n = [0usize; 5];
    for &c in classes {
        n[c] += 1;
    }
    let cells: Vec<String> = n.iter().map(|&x| pct(x, classes.len())).collect();
    format!("| {label} | {} | {} |", classes.len(), cells.join(" | "))
}

/// Connected components of the nodes in `set` over the springs with both ends in it:
/// (count, sizes largest first).
fn components(n: usize, sp: &[Spring], set: &[bool]) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for b in sp {
        if set[b.a] && set[b.b] {
            let (x, y) = (find(&mut parent, b.a), find(&mut parent, b.b));
            parent[x] = y;
        }
    }
    let mut size: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..n {
        if set[i] {
            *size.entry(find(&mut parent, i)).or_insert(0) += 1;
        }
    }
    let mut v: Vec<usize> = size.into_values().collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

fn report(car: &Car) {
    let s = &car.sim;
    let n = s.nodes.len();
    let worst = std::env::args().any(|a| a == "--worst");
    let (sp, skipped) = springs(car, worst);
    let total_mass: f64 = (0..n).map(|i| mass(s, i)).filter(|m| m.is_finite()).sum();
    println!("\n## {}\n", car.title);
    println!(
        "{n} nodes, {} beam records ({} linear spring–dampers measured), {total_mass:.0} kg; \
         centred damping {}; step {STEP} s; bounded beams {}.",
        s.beams.len(),
        sp.len(),
        if s.settings.centred_damping {
            "on (BeamNG profile)"
        } else {
            "off (Rigs of Rods profile)"
        },
        if worst {
            "at their limit rates (`--worst`)"
        } else {
            "at their rates inside their bounds"
        },
    );
    if !skipped.is_empty() {
        let list: Vec<String> = skipped.iter().map(|(k, v)| format!("{v} {k}")).collect();
        println!(
            "\nRecords left out (not a linear spring between two nodes): {}.",
            list.join(", ")
        );
    }

    // Per beam and per node limits.
    let red = |b: &Spring| {
        let (ma, mb) = (mass(s, b.a), mass(s, b.b));
        if !ma.is_finite() {
            mb
        } else if !mb.is_finite() {
            ma
        } else {
            ma * mb / (ma + mb)
        }
    };
    let beam_h0: Vec<f64> = sp
        .iter()
        .map(|b| h_limit(b.k, b.c, red(b), false, b.centred))
        .collect();
    let beam_h: Vec<f64> = sp
        .iter()
        .map(|b| h_limit(b.k, b.c, red(b), true, b.centred))
        .collect();
    let (mut nk, mut nc, mut ncentred, mut deg) =
        (vec![0.0; n], vec![0.0; n], vec![true; n], vec![0usize; n]);
    for b in &sp {
        for i in [b.a, b.b] {
            nk[i] += b.k;
            nc[i] += b.c;
            ncentred[i] &= b.centred;
            deg[i] += 1;
        }
    }
    let node_h0: Vec<f64> = (0..n)
        .map(|i| h_limit(nk[i], nc[i], mass(s, i), false, ncentred[i]))
        .collect();
    let node_h: Vec<f64> = (0..n)
        .map(|i| h_limit(nk[i], nc[i], mass(s, i), true, ncentred[i]))
        .collect();
    let node_hg: Vec<f64> = (0..n)
        .map(|i| h_limit(2.0 * nk[i], 2.0 * nc[i], mass(s, i), true, ncentred[i]))
        .collect();
    let h = STEP as f64;
    let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let (w, mode) = omega_max(s, &sp, &vec![true; n], 4000);
    println!("\n### Where the vehicle stands at 2 kHz\n");
    println!("| limit | tightest element's limit | limit / step (the margin at 2 kHz) |");
    println!("|---|---|---|");
    for (label, v) in [
        ("a beam alone, undamped", &beam_h0),
        ("a beam alone, damped", &beam_h),
        ("a node, Σk, undamped", &node_h0),
        ("a node, Σk and Σc, damped", &node_h),
        ("a node, Gershgorin (2Σk, 2Σc), damped", &node_hg),
    ] {
        println!("| {label} | {:.3} ms | {:.2} |", min(v) * 1e3, min(v) / h);
    }
    println!(
        "| the assembled vehicle at rest, undamped (power iteration) | {:.3} ms | {:.2} |",
        2.0 / w * 1e3,
        2.0 / w / h
    );
    let mut top: Vec<usize> = (0..n).collect();
    top.sort_by(|&a, &b| mode[b].total_cmp(&mode[a]));
    let names: Vec<String> = top
        .iter()
        .take(6)
        .map(|&i| {
            format!(
                "{} ({}, {:.2} kg, {:.0} %)",
                car.node_name[i],
                car.node_part[i],
                mass(s, i),
                100.0 * mode[i]
            )
        })
        .collect();
    println!(
        "\nThe stiffest mode (ω = {w:.0} rad/s, ω·h = {:.2}) lives on: {}.",
        w * h,
        names.join("; ")
    );

    // Rate classes.
    println!("\n### Share by the slowest rate the limit allows\n");
    println!("A rate r is allowed when limit ≥ safety / r. Columns are exclusive classes.\n");
    for &sf in &SAFETY {
        println!("Safety factor {sf}:\n");
        println!("| what | count | {} |", CLASS_NAMES.join(" | "));
        println!("|---|---|---|---|---|---|---|");
        let cl = |v: &[f64]| v.iter().map(|&x| class(x, sf)).collect::<Vec<_>>();
        println!(
            "{}",
            share_row("beams, each alone, undamped", &cl(&beam_h0))
        );
        println!("{}", share_row("beams, each alone, damped", &cl(&beam_h)));
        println!("{}", share_row("nodes, Σk, undamped", &cl(&node_h0)));
        let nd = cl(&node_h);
        println!("{}", share_row("nodes, Σk and Σc, damped", &nd));
        println!("{}", share_row("nodes, Gershgorin, damped", &cl(&node_hg)));
        // A beam steps with its faster end.
        let by_end: Vec<usize> = sp.iter().map(|b| nd[b.a].min(nd[b.b])).collect();
        println!(
            "{}",
            share_row(
                "beams, at the rate of their faster end node (damped)",
                &by_end
            )
        );
        let mut m = [0.0; 5];
        for i in 0..n {
            if mass(s, i).is_finite() {
                m[nd[i]] += mass(s, i);
            }
        }
        let cells: Vec<String> = m.iter().map(|&x| pctf(x, total_mass)).collect();
        println!(
            "| mass, by its node's class (damped) | {total_mass:.0} kg | {} |",
            cells.join(" | ")
        );
        println!();
    }

    // The work a multi-rate step would save: beam evaluations per 2 kHz step.
    println!("### Beam and node updates per frame if every element ran at its own rate\n");
    println!("Relative to all at 2 kHz (1.00 = no saving); a beam at the rate of its faster end node, classes under the margin at 2 kHz.\n");
    println!("| safety factor | node updates | beam evaluations |");
    println!("|---|---|---|");
    for &sf in &SAFETY {
        let nd: Vec<usize> = node_h.iter().map(|&x| class(x, sf)).collect();
        let cost = |c: usize| 1.0 / (1u32 << c.saturating_sub(1)) as f64;
        let nodes: f64 = nd.iter().map(|&c| cost(c)).sum::<f64>() / n as f64;
        let beams: f64 =
            sp.iter().map(|b| cost(nd[b.a].min(nd[b.b]))).sum::<f64>() / sp.len() as f64;
        println!("| {sf} | {nodes:.2} | {beams:.2} |");
    }

    // By part.
    let nd2: Vec<usize> = node_h.iter().map(|&x| class(x, 2.0)).collect();
    let nd4: Vec<usize> = node_h.iter().map(|&x| class(x, 4.0)).collect();
    let nd1: Vec<usize> = node_h.iter().map(|&x| class(x, 1.0)).collect();
    println!("\n### By part (nodes, Σk and Σc, damped)\n");
    println!("Share of the part's nodes that need 2 kHz (the two fastest classes together), then that may run at 1 kHz, 500 Hz, 250 Hz or less. Detached: the part alone as a free piece (its own nodes and the beams between them), its largest ω by power iteration, undamped.\n");
    println!("| part | nodes | kg | SF 1: 2 kHz | 1 kHz | 500 Hz | ≤ 250 Hz | SF 2: 2 kHz | 1 kHz | 500 Hz | ≤ 250 Hz | SF 4: 2 kHz | 1 kHz | 500 Hz | ≤ 250 Hz | softest node's limit | detached: largest ω (rad/s) | detached: slowest rate, safety 1 | safety 2 |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut parts: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        parts.entry(car.node_part[i].as_str()).or_default().push(i);
    }
    let mut rows: Vec<(&str, Vec<usize>)> = parts.into_iter().collect();
    rows.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for (name, ids) in &rows {
        let kg: f64 = ids
            .iter()
            .map(|&i| mass(s, i))
            .filter(|m| m.is_finite())
            .sum();
        let mut cells = Vec::new();
        for nd in [&nd1, &nd2, &nd4] {
            let c =
                |lo: usize, hi: usize| ids.iter().filter(|&&i| nd[i] >= lo && nd[i] <= hi).count();
            for (lo, hi) in [(0, 1), (2, 2), (3, 3), (4, 4)] {
                cells.push(pct(c(lo, hi), ids.len()));
            }
        }
        let soft = ids
            .iter()
            .map(|&i| node_h[i])
            .filter(|x| x.is_finite())
            .fold(0.0, f64::max);
        // The part as a detached piece: its own nodes and the beams between them.
        let mut own = vec![false; n];
        for &i in ids {
            own[i] = true;
        }
        let inner: Vec<Spring> = sp
            .iter()
            .filter(|b| own[b.a] && own[b.b])
            .cloned()
            .collect();
        let (wp, _) = omega_max(s, &inner, &own, 3000);
        let rate = |sf: f64| {
            if wp > 0.0 {
                CLASS_NAMES[class(2.0 / wp, sf)]
            } else {
                "–"
            }
        };
        println!(
            "| {name} | {} | {kg:.1} | {} | {:.2} ms | {wp:.0} | {} | {} |",
            ids.len(),
            cells.join(" | "),
            soft * 1e3,
            rate(1.0),
            rate(2.0)
        );
    }

    // By kind.
    println!("\n### By beam kind\n");
    println!("Each beam alone (damped, safety factor 2), and how much of the stiffness on the nodes that need 2 kHz (safety factor 2) each kind supplies.\n");
    println!("| kind | beams | k min / median / max (N/m) | under margin | 2 kHz | 1 kHz | 500 Hz | ≤ 250 Hz | touch a 2 kHz node | share of Σk on 2 kHz nodes |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut kinds: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (q, b) in sp.iter().enumerate() {
        kinds
            .entry(car.beam_kind[b.record].as_str())
            .or_default()
            .push(q);
    }
    let stiff_k: f64 = (0..n).filter(|&i| nd2[i] <= 1).map(|i| nk[i]).sum();
    for (name, ids) in &kinds {
        let mut ks: Vec<f64> = ids.iter().map(|&q| sp[q].k).collect();
        ks.sort_by(f64::total_cmp);
        let mut c = [0usize; 5];
        for &q in ids {
            c[class(beam_h[q], 2.0)] += 1;
        }
        let touch = ids
            .iter()
            .filter(|&&q| nd2[sp[q].a] <= 1 || nd2[sp[q].b] <= 1)
            .count();
        let on: f64 = ids
            .iter()
            .map(|&q| {
                let b = &sp[q];
                [b.a, b.b].iter().filter(|&&i| nd2[i] <= 1).count() as f64 * b.k
            })
            .sum();
        let cells: Vec<String> = c.iter().map(|&x| pct(x, ids.len())).collect();
        println!(
            "| {name} | {} | {:.3e} / {:.3e} / {:.3e} | {} | {} | {} |",
            ids.len(),
            ks[0],
            ks[ks.len() / 2],
            ks[ks.len() - 1],
            cells.join(" | "),
            pct(touch, ids.len()),
            pctf(on, stiff_k),
        );
    }

    // What makes a node need 2 kHz: stiffness or lightness.
    println!("\n### What the 2 kHz nodes look like (safety factor 2)\n");
    println!("| class | nodes | mass min / median / max (kg) | Σk min / median / max (N/m) | beams per node, median | ζ median |");
    println!("|---|---|---|---|---|---|");
    for (label, lo, hi) in [
        ("need 2 kHz", 0usize, 1usize),
        ("1 kHz", 2, 2),
        ("500 Hz", 3, 3),
        ("≤ 250 Hz", 4, 4),
    ] {
        let ids: Vec<usize> = (0..n)
            .filter(|&i| nd2[i] >= lo && nd2[i] <= hi && mass(s, i).is_finite())
            .collect();
        if ids.is_empty() {
            println!("| {label} | 0 | – | – | – | – |");
            continue;
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[0], v[v.len() / 2], v[v.len() - 1])
        };
        let m = med(ids.iter().map(|&i| mass(s, i)).collect());
        let k = med(ids.iter().map(|&i| nk[i]).collect());
        let d = med(ids.iter().map(|&i| deg[i] as f64).collect());
        let z = med(ids
            .iter()
            .map(|&i| {
                if nk[i] > 0.0 {
                    nc[i] / (2.0 * (nk[i] * mass(s, i)).sqrt())
                } else {
                    0.0
                }
            })
            .collect());
        println!(
            "| {label} | {} | {:.2} / {:.2} / {:.2} | {:.2e} / {:.2e} / {:.2e} | {:.0} | {:.3} |",
            ids.len(),
            m.0,
            m.1,
            m.2,
            k.0,
            k.1,
            k.2,
            d.1,
            z.1
        );
    }

    // Break groups (JBeam): beams by the class of their faster end.
    let by_end2: Vec<usize> = sp.iter().map(|b| nd2[b.a].min(nd2[b.b])).collect();
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (q, b) in sp.iter().enumerate() {
        groups
            .entry(car.beam_group[b.record].as_str())
            .or_default()
            .push(q);
    }
    if groups.len() > 1 {
        println!(
            "\n### By break group (beams at the rate of their faster end node, safety factor 2)\n"
        );
        let grouped: usize = groups
            .iter()
            .filter(|(k, _)| !k.is_empty())
            .map(|(_, v)| v.len())
            .sum();
        let stiff_grouped: usize = groups
            .iter()
            .filter(|(k, _)| !k.is_empty())
            .map(|(_, v)| v.iter().filter(|&&q| by_end2[q] <= 1).count())
            .sum();
        let all_stiff = groups
            .iter()
            .filter(|(k, v)| !k.is_empty() && v.iter().all(|&q| by_end2[q] <= 1))
            .count();
        let none_stiff = groups
            .iter()
            .filter(|(k, v)| !k.is_empty() && v.iter().all(|&q| by_end2[q] > 1))
            .count();
        println!(
            "{} break groups hold {grouped} beams; {} of those beams need 2 kHz. {all_stiff} groups are wholly at 2 kHz, {none_stiff} wholly slower, {} mixed.\n",
            groups.len() - groups.contains_key("") as usize,
            pct(stiff_grouped, grouped),
            groups.len() - groups.contains_key("") as usize - all_stiff - none_stiff
        );
        println!("| break group | beams | need 2 kHz | 1 kHz | 500 Hz | ≤ 250 Hz |");
        println!("|---|---|---|---|---|---|");
        let mut rows: Vec<(&str, &Vec<usize>)> = groups.iter().map(|(k, v)| (*k, v)).collect();
        rows.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        for (name, ids) in rows.iter().take(10) {
            let c = |lo: usize, hi: usize| {
                ids.iter()
                    .filter(|&&q| by_end2[q] >= lo && by_end2[q] <= hi)
                    .count()
            };
            println!(
                "| {} | {} | {} | {} | {} | {} |",
                if name.is_empty() { "(none)" } else { name },
                ids.len(),
                pct(c(0, 1), ids.len()),
                pct(c(2, 2), ids.len()),
                pct(c(3, 3), ids.len()),
                pct(c(4, 4), ids.len())
            );
        }
        if rows.len() > 10 {
            println!("| ({} smaller groups) | | | | | |", rows.len() - 10);
        }
    }

    // Is the stiff set one core or scattered?
    println!("\n### Where the 2 kHz nodes are\n");
    let pos = |i: usize| s.nodes[i].pos;
    let bbox = |set: &[bool]| {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for i in (0..n).filter(|&i| set[i]) {
            lo = lo.min(pos(i));
            hi = hi.max(pos(i));
        }
        hi - lo
    };
    let all = vec![true; n];
    let car_box = bbox(&all);
    println!("| safety factor | 2 kHz nodes | their box / the car's box (x, y, z) | connected pieces (largest) | slower nodes | slower nodes with a 2 kHz neighbour | slower pieces once the 2 kHz nodes are cut out (largest) | beams joining two rates |");
    println!("|---|---|---|---|---|---|---|---|");
    for (sf, nd) in [(1.0, &nd1), (2.0, &nd2), (4.0, &nd4)] {
        let stiff: Vec<bool> = (0..n).map(|i| nd[i] <= 1).collect();
        let soft: Vec<bool> = stiff.iter().map(|x| !x).collect();
        let ns = stiff.iter().filter(|&&x| x).count();
        let mut near = vec![false; n];
        let mut cross = 0;
        for b in &sp {
            if stiff[b.a] != stiff[b.b] {
                near[if stiff[b.a] { b.b } else { b.a }] = true;
            }
            if nd[b.a].max(1) != nd[b.b].max(1) {
                cross += 1;
            }
        }
        let cs = components(n, &sp, &stiff);
        let cf = components(n, &sp, &soft);
        let bx = if ns > 0 {
            bbox(&stiff) / car_box
        } else {
            Vec3::ZERO
        };
        println!(
            "| {sf} | {ns} ({}) | {:.2}, {:.2}, {:.2} | {} ({}) | {} | {} | {} ({}) | {} ({}) |",
            pct(ns, n),
            bx.x,
            bx.y,
            bx.z,
            cs.len(),
            cs.first().copied().unwrap_or(0),
            n - ns,
            pct(near.iter().filter(|&&x| x).count(), n - ns),
            cf.len(),
            cf.first().copied().unwrap_or(0),
            cross,
            pct(cross, sp.len()),
        );
    }

    // Lengthwise profile: share of nodes needing 2 kHz in ten slices along the longest axis.
    let axis = if car_box.x > car_box.z { 0 } else { 2 };
    let lo = (0..n).map(|i| pos(i)[axis]).fold(f32::MAX, f32::min);
    let span = car_box[axis].max(1e-6);
    let mut slice = [[0usize; 2]; 10];
    for i in 0..n {
        let q = (((pos(i)[axis] - lo) / span * 10.0) as usize).min(9);
        slice[q][0] += 1;
        slice[q][1] += (nd2[i] <= 1) as usize;
    }
    println!(
        "\nAlong the car's length in ten slices (safety factor 2), nodes needing 2 kHz / nodes:\n"
    );
    println!(
        "| slice | {} |",
        (1..=10)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!("|---|{}", "---|".repeat(10));
    println!(
        "| share | {} |",
        slice
            .iter()
            .map(|c| format!("{} of {}", c[1], c[0]))
            .collect::<Vec<_>>()
            .join(" | ")
    );

    // The largest slow set by the assembled stiffness: remove the nodes carrying the
    // stiffest mode (held: they step at 2 kHz) until what is left is slow enough.
    println!("\n### The largest set that could step slower, from the assembled vehicle\n");
    println!(
        "Nodes are removed (held, as if stepping at 2 kHz) from the stiffest mode of what is left until its largest ω is under the limit (undamped; a greedy search, so a lower bound on the set). Each row continues from the one above. A beam is slow when both its ends are.\n"
    );
    println!("| largest ω allowed (rad/s) | which is | slow nodes | slow mass | slow beams |");
    println!("|---|---|---|---|---|");
    {
        let own = 2.0 / (w * h);
        let targets: Vec<(f64, String)> = vec![
            (
                2000.0 / own,
                format!("1 kHz at the car's own margin today ({own:.2})"),
            ),
            (1000.0, "1 kHz at safety 2, or 500 Hz at safety 1".into()),
            (
                1000.0 / own,
                format!("500 Hz at the car's own margin ({own:.2})"),
            ),
            (
                500.0,
                "1 kHz at safety 4, 500 Hz at safety 2, or 250 Hz at safety 1".into(),
            ),
        ];
        let mut set: Vec<bool> = (0..n).map(|i| mass(s, i).is_finite()).collect();
        for (limit, label) in targets {
            loop {
                let (mut w, mut share) = omega_max(s, &sp, &set, 600);
                if w <= limit {
                    // Confirm with a converged iteration.
                    (w, share) = omega_max(s, &sp, &set, 4000);
                    if w <= limit {
                        break;
                    }
                }
                let top = share.iter().cloned().fold(0.0, f64::max);
                if top <= 0.0 {
                    break;
                }
                for i in 0..n {
                    set[i] &= share[i] < 0.5 * top;
                }
            }
            let count = set.iter().filter(|&&x| x).count();
            let kg: f64 = (0..n).filter(|&i| set[i]).map(|i| mass(s, i)).sum();
            let beams = sp.iter().filter(|b| set[b.a] && set[b.b]).count();
            println!(
                "| {limit:.0} | {label} | {count} ({}) | {} | {beams} ({}) |",
                pct(count, n),
                pctf(kg, total_mass),
                pct(beams, sp.len()),
            );
        }
    }

    // Does a class hold at its rate once the faster nodes are held? The exact largest ω of
    // each class's nodes with every other node fixed.
    println!("\n### Check: each class's own largest frequency (the other nodes held)\n");
    println!("| safety factor | class | nodes | largest ω (rad/s) | ω·h at the class's rate (stable below 2, undamped) |");
    println!("|---|---|---|---|---|");
    for (sf, nd) in [(2.0, &nd2), (4.0, &nd4)] {
        for (c, rate) in RATES.iter().enumerate().skip(1) {
            // The nodes that would step at `rate` or slower.
            let set: Vec<bool> = (0..n).map(|i| nd[i] > c).collect();
            let count = set.iter().filter(|&&x| x).count();
            if count == 0 {
                continue;
            }
            let (w, _) = omega_max(s, &sp, &set, 4000);
            println!(
                "| {sf} | {} and slower | {count} | {w:.0} | {:.2} |",
                CLASS_NAMES[c + 1],
                w / rate
            );
        }
    }
}

fn main() {
    println!("# Beam rate histogram");
    if let Some(dir) = arg("--car") {
        report(&jbeam_car(std::path::Path::new(&dir)));
    } else {
        let res = arg("--generated")
            .and_then(|r| Resolution::parse(&r))
            .unwrap_or(Resolution::Low);
        report(&generated(res));
    }
}

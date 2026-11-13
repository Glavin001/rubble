//! What beam damping real JBeam content uses, as a damping ratio on its own node
//! masses: for each beam of the assembled configuration,
//! ζ = beamDamp / (2·√(beamSpring · m₁m₂/(m₁+m₂))), the beam's own two-mass mode.
//! Grouped per part (its `slotType` and name), with beamSpring, beamDamp and
//! beamDeform medians, so the ratio carries over to another car's masses
//! (`vehicle-authoring.md`, "Where we differ from real material").
//!
//! Reads a vehicle directory at run time (measurement only: nothing in a test or
//! build depends on its output).
//!
//! `NODES=a,b,…` keeps only the beams between those nodes.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example jbeam_damping -- <vehicles-dir> <name> <config.pc>
//! ```
use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

fn num(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn id(f: &serde_json::Map<String, Value>, k: &str) -> Option<String> {
    let v = f.get(k).or_else(|| f.get(&format!("{k}:")))?;
    v.as_str()
        .map(str::to_string)
        .or_else(|| v.as_f64().map(|x| x.to_string()))
}

fn q(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

#[derive(Default)]
struct Acc {
    zeta: Vec<f64>,
    k: Vec<f64>,
    c: Vec<f64>,
    deform: Vec<f64>,
    m: Vec<f64>,
    support: usize,
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [dir, name, pc] = &a[..] else {
        panic!("usage: <vehicles-dir> <name> <config.pc>");
    };
    let dir = std::path::Path::new(dir);
    let texts = engine_softbody::jbeam::read_vehicle_dir(dir, name).unwrap();
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
    let slot: HashMap<String, String> = engine_softbody::jbeam::assembled_parts(&files, Some(&pc))
        .unwrap()
        .into_iter()
        .collect();
    let mut mass: HashMap<String, f64> = HashMap::new();
    for (_, f) in engine_softbody::jbeam::assembled_rows(&files, Some(&pc), "nodes").unwrap() {
        if let Some(n) = id(&f, "id") {
            mass.insert(n, num(f.get("nodeWeight")).unwrap_or(25.0));
        }
    }
    // NODES=a,b,c: only beams with both ends in that set (a member inside a part,
    // such as a body's front rails), reported as one line.
    let only: Option<Vec<String>> = std::env::var("NODES")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    let mut by: BTreeMap<String, Acc> = BTreeMap::new();
    for (part, f) in engine_softbody::jbeam::assembled_rows(&files, Some(&pc), "beams").unwrap() {
        let (Some(a), Some(b)) = (id(&f, "id1"), id(&f, "id2")) else {
            continue;
        };
        let (Some(&ma), Some(&mb)) = (mass.get(&a), mass.get(&b)) else {
            continue;
        };
        let ty = f
            .get("beamType")
            .and_then(Value::as_str)
            .unwrap_or("|NORMAL")
            .trim_start_matches('|')
            .to_ascii_uppercase();
        let k = num(f.get("beamSpring")).unwrap_or(4_300_000.0);
        let c = num(f.get("beamDamp")).unwrap_or(580.0);
        let d = num(f.get("beamDeform")).unwrap_or(220_000.0);
        if let Some(o) = &only {
            if !(o.contains(&a) && o.contains(&b)) {
                continue;
            }
        }
        let key = format!(
            "{:<14} {}",
            slot.get(&part).map(String::as_str).unwrap_or("?"),
            part
        );
        let e = by.entry(key).or_default();
        if ty == "SUPPORT" {
            e.support += 1;
        }
        if ty != "NORMAL" && ty != "SUPPORT" || k <= 0.0 {
            continue;
        }
        let me = ma * mb / (ma + mb);
        e.zeta.push(c / (2.0 * (k * me).sqrt()));
        e.k.push(k);
        e.c.push(c);
        e.deform.push(d);
        e.m.push(ma.min(mb));
    }
    println!(
        "{:<48} {:>5} {:>4} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>7}",
        "slotType part",
        "beams",
        "sup",
        "ζ p10",
        "ζ p50",
        "ζ p90",
        "k p50",
        "c p50",
        "deform",
        "m p50"
    );
    for (key, e) in &mut by {
        if e.zeta.is_empty() {
            continue;
        }
        let n = e.zeta.len();
        let (z10, z50, z90) = (
            q(&mut e.zeta, 0.1),
            q(&mut e.zeta, 0.5),
            q(&mut e.zeta, 0.9),
        );
        println!(
            "{key:<48} {n:>5} {:>4} {z10:>8.4} {z50:>8.4} {z90:>8.4} {:>9.3e} {:>9.0} {:>8.0} {:>7.2}",
            e.support,
            q(&mut e.k, 0.5),
            q(&mut e.c, 0.5),
            q(&mut e.deform, 0.5),
            q(&mut e.m, 0.5),
        );
    }
}

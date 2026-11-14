//! What yield strain real JBeam content uses: for each NORMAL or SUPPORT beam of
//! the assembled configuration, ε_y = beamDeform / (beamSpring · L) with L its
//! spawn length, the elastic stretch it holds at its yield force, and the
//! springback that implies after a plastic strain ε_p: ε_y / (ε_y + ε_p)
//! (elastic over total, the beam unloaded from its yield force). Grouped per part
//! (its `slotType` and name). Beside it, sedan1's beams from its `.truck`
//! (`SEDAN1=low|medium|high`).
//!
//! Reads a vehicle directory at run time (measurement only: nothing in a test or
//! build depends on its output).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example jbeam_yield_strain -- <vehicles-dir> <name> <config.pc>
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
    eps: Vec<f64>,
    k: Vec<f64>,
    deform: Vec<f64>,
    len: Vec<f64>,
}

/// Plastic strains at which the springback is reported.
const EP: [f64; 2] = [0.05, 0.20];

fn report(by: &mut BTreeMap<String, Acc>) {
    println!(
        "{:<48} {:>5} {:>8} {:>8} {:>8} {:>9} {:>8} {:>6} {:>9} {:>9}",
        "part",
        "beams",
        "εy p10",
        "εy p50",
        "εy p90",
        "k p50",
        "deform",
        "L p50",
        "sb@5%",
        "sb@20%"
    );
    for (key, e) in by.iter_mut() {
        if e.eps.is_empty() {
            continue;
        }
        let n = e.eps.len();
        let sb = |ep: f64, eps: &[f64]| {
            let mut v: Vec<f64> = eps.iter().map(|&y| y / (y + ep)).collect();
            q(&mut v, 0.5)
        };
        let (s5, s20) = (sb(EP[0], &e.eps), sb(EP[1], &e.eps));
        println!(
            "{key:<48} {n:>5} {:>8.4} {:>8.4} {:>8.4} {:>9.3e} {:>8.0} {:>6.3} {:>9.2} {:>9.2}",
            q(&mut e.eps, 0.1),
            q(&mut e.eps, 0.5),
            q(&mut e.eps, 0.9),
            q(&mut e.k, 0.5),
            q(&mut e.deform, 0.5),
            q(&mut e.len, 0.5),
            s5,
            s20
        );
    }
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
    let mut pos: HashMap<String, [f64; 3]> = HashMap::new();
    for (_, f) in engine_softbody::jbeam::assembled_rows(&files, Some(&pc), "nodes").unwrap() {
        if let (Some(n), Some(x), Some(y), Some(z)) = (
            id(&f, "id"),
            num(f.get("posX")),
            num(f.get("posY")),
            num(f.get("posZ")),
        ) {
            pos.insert(n, [x, y, z]);
        }
    }
    let mut by: BTreeMap<String, Acc> = BTreeMap::new();
    let mut bx_ends: BTreeMap<String, (usize, std::collections::HashSet<String>)> = BTreeMap::new();
    for (part, f) in engine_softbody::jbeam::assembled_rows(&files, Some(&pc), "beams").unwrap() {
        let (Some(a), Some(b)) = (id(&f, "id1"), id(&f, "id2")) else {
            continue;
        };
        let (Some(pa), Some(pb)) = (pos.get(&a), pos.get(&b)) else {
            continue;
        };
        let l =
            ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
        let ty = f
            .get("beamType")
            .and_then(Value::as_str)
            .unwrap_or("|NORMAL")
            .trim_start_matches('|')
            .to_ascii_uppercase();
        let k = num(f.get("beamSpring")).unwrap_or(4_300_000.0);
        let d = num(f.get("beamDeform")).unwrap_or(220_000.0);
        if ty != "NORMAL" && ty != "SUPPORT" || k <= 0.0 || l < 1e-3 || d >= 1.0e9 {
            continue;
        }
        let key = format!(
            "{:<14} {}",
            slot.get(&part).map(String::as_str).unwrap_or("?"),
            part
        );
        let be = bx_ends.entry(key.clone()).or_default();
        be.0 += 2;
        be.1.insert(a.clone());
        be.1.insert(b.clone());
        let e = by.entry(key).or_default();
        e.eps.push(d / (k * l));
        e.k.push(k);
        e.deform.push(d);
        e.len.push(l);
    }
    report(&mut by);
    for (k, (n, set)) in &bx_ends {
        println!(
            "BX {k}: {} beams, {} nodes, {:.1} beam ends per node",
            n / 2,
            set.len(),
            *n as f64 / set.len() as f64
        );
    }
    if let Ok(res) = std::env::var("SEDAN1") {
        use engine_softbody::authoring::{build, style, BuildParams, Resolution};
        let b = build(
            &style("sedan1").unwrap(),
            &BuildParams {
                resolution: Resolution::parse(&res).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let s = engine_softbody::explicit_beams::ExplicitBeams::from_truck(
            &b.truck,
            glam::Vec3::ZERO,
            glam::Vec3::ZERO,
            Default::default(),
        )
        .unwrap();
        let sk = &b.skeleton;
        let mut by: BTreeMap<String, Acc> = BTreeMap::new();
        for bm in &s.beams {
            let (a, c) = (bm.a as usize, bm.b as usize);
            if a >= sk.nodes.len()
                || c >= sk.nodes.len()
                || bm.ty != engine_softbody::explicit_beams::ExplicitType::Deformable
                || bm.k <= 0.0
                || bm.tc >= 1.0e9
            {
                continue;
            }
            let (pa, pc) = (
                &sk.parts[sk.nodes[a].part as usize],
                &sk.parts[sk.nodes[c].part as usize],
            );
            let key = if pa == pc {
                pa.clone()
            } else {
                format!("{pa}~{pc}")
            };
            for (side, y) in [("compr", bm.tc), ("tens", bm.tt)] {
                let e = by.entry(format!("sedan1 {key} {side}")).or_default();
                e.eps.push((y / (bm.k * bm.rest)) as f64);
                e.k.push(bm.k as f64);
                e.deform.push(y as f64);
                e.len.push(bm.rest as f64);
            }
        }
        report(&mut by);
        // Beam ends per node inside each part (the bound shares a node's
        // stiffness among its beams).
        let mut ends: BTreeMap<String, (usize, std::collections::HashSet<usize>)> = BTreeMap::new();
        for bm in &s.beams {
            let (a, c) = (bm.a as usize, bm.b as usize);
            if a >= sk.nodes.len() || c >= sk.nodes.len() || sk.nodes[a].part != sk.nodes[c].part {
                continue;
            }
            let e = ends
                .entry(sk.parts[sk.nodes[a].part as usize].clone())
                .or_default();
            e.0 += 2;
            e.1.insert(a);
            e.1.insert(c);
        }
        for (k, (n, set)) in &ends {
            println!(
                "sedan1 {k}: {} beams, {} nodes, {:.1} beam ends per node",
                n / 2,
                set.len(),
                *n as f64 / set.len() as f64
            );
        }
    }
}

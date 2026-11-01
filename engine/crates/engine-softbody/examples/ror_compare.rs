//! Compares Rubble against the Rigs of Rods reference harness (`inspiration/ror-harness`,
//! local only) on the same `.truck` file. Reads the harness's CSV dumps; nothing here
//! depends on Rigs of Rods code.
//!
//! ```bash
//! cargo run -p engine-softbody --release --example ror_compare -- spawn <file.truck> <nodes.csv> <beams.csv>
//! # the body without wheels/engine/brakes, for both sides
//! cargo run -p engine-softbody --release --example ror_compare -- bare <in.truck> <out.truck>
//! # engine_softbody::ror on the harness's `free` scene, against its node dump
//! cargo run -p engine-softbody --release --example ror_compare -- free <file.truck> <nodes.csv> pos=x,y,z vel=… box=… drag=0 secs=1
//! ```
use engine_softbody::procedural::strip_truck_sections;
use engine_softbody::ror::{RorBox, RorSettings, RorSim};
use engine_softbody::truck::{load_truck, parse_truck};
use glam::Vec3;

fn csv(path: &str) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .skip(1)
        .map(|l| {
            l.split(',')
                .map(|v| v.parse().unwrap_or(f64::NAN))
                .collect()
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("spawn") => spawn(&a[1], &a[2], a.get(3).map(String::as_str)),
        Some("bare") => {
            let t = std::fs::read_to_string(&a[1]).unwrap();
            std::fs::write(
                &a[2],
                strip_truck_sections(&t, &["wheels", "engine", "brakes"]),
            )
            .unwrap();
        }
        Some("free") => free(&a[1], &a[2], &a[3..]),
        _ => eprintln!("usage: ror_compare spawn <file.truck> <nodes.csv> [beams.csv]"),
    }
}

/// Node masses and positions, then beam parameters, as spawned.
fn spawn(truck: &str, nodes_csv: &str, beams_csv: Option<&str>) {
    let src = std::fs::read_to_string(truck).unwrap();
    let file = parse_truck(&src);
    for w in &file.warnings {
        println!("rubble loader warning: {w}");
    }
    let desc = load_truck(&src);
    let ror = csv(nodes_csv);
    println!("nodes: rubble {} ror {}", desc.nodes.len(), ror.len());
    let full = RorSim::from_truck(&src, Vec3::ZERO, Vec3::ZERO, RorSettings::default());
    if full.nodes.len() != desc.nodes.len() {
        let rm: f64 = full.nodes.iter().map(|n| f64::from(n.mass)).sum();
        let worst = full
            .nodes
            .iter()
            .zip(&ror)
            .map(|(n, r)| (f64::from(n.mass) - r[1]).abs() / r[1].max(1e-9))
            .fold(0.0, f64::max);
        println!(
            "with RoR tyres: nodes {} (ror {}), mass {rm:.2} kg, worst node mass {:.3}%",
            full.nodes.len(),
            ror.len(),
            worst * 100.0
        );
    }
    let (mut rm, mut om, mut worst) = (0.0f64, 0.0f64, (0.0f64, 0usize));
    for (i, n) in desc.nodes.iter().enumerate() {
        let r = &ror[i];
        rm += f64::from(n.mass);
        om += r[1];
        let d = (f64::from(n.mass) - r[1]).abs() / r[1].max(1e-9);
        if d > worst.0 {
            worst = (d, i);
        }
    }
    println!("total mass: rubble {rm:.2} kg, ror {om:.2} kg");
    let i = worst.1;
    println!(
        "worst node mass: node {i} rubble {:.4} ror {:.4} ({:.2}%)",
        desc.nodes[i].mass,
        ror[i][1],
        worst.0 * 100.0
    );
    let Some(bc) = beams_csv else { return };
    let rb = csv(bc);
    println!("beams: rubble {} ror {}", desc.beams.len(), rb.len());
    // columns: i,p1,p2,k,d,L,refL,maxpos,maxneg,strength,plastic,bounded,type,detacher
    // Matched by node pair: RoR spawns cinecam, wheels, beams, shocks, hydros in that
    // order, the file may interleave them.
    let sim = if full.nodes.len() == ror.len() && full.nodes.len() != desc.nodes.len() {
        full.clone()
    } else {
        RorSim::new(&desc, Vec3::ZERO, Vec3::ZERO, RorSettings::default())
    };
    let mut by_pair: std::collections::HashMap<(u32, u32), Vec<usize>> = Default::default();
    for (i, r) in rb.iter().enumerate() {
        let (a, b) = (r[1] as u32, r[2] as u32);
        by_pair.entry((a.min(b), a.max(b))).or_default().push(i);
    }
    let mut bad = 0;
    for (k, b) in sim.beams.iter().enumerate() {
        let Some(list) = by_pair.get_mut(&(b.a.min(b.b), b.a.max(b.b))) else {
            println!("beam {k} ({}-{}) not in RoR", b.a, b.b);
            bad += 1;
            continue;
        };
        let r = &rb[list.remove(0)];
        let checks = [
            ("k", f64::from(b.k), r[3]),
            ("d", f64::from(b.d), r[4]),
            ("L", f64::from(b.l), r[5]),
            (
                "maxpos",
                f64::from(b.maxposstress),
                if b.normal { r[7] } else { f64::from(f32::MAX) },
            ),
            ("strength", f64::from(b.strength), r[9]),
            ("plastic", f64::from(b.plastic_coef), r[10]),
            ("bounded", f64::from(b.bounded as u8), r[11]),
            (
                "normal",
                f64::from(u8::from(b.normal)),
                f64::from(u8::from(r[12] == 0.0)),
            ),
            ("detacher", f64::from(b.detacher), r[13]),
        ];
        for (name, ours, theirs) in checks {
            let tol = 1e-5 * theirs.abs().max(1.0);
            if (ours - theirs).abs() > tol && bad < 30 {
                bad += 1;
                println!(
                    "beam {k} ({}-{}) {name}: rubble {ours} ror {theirs}",
                    b.a, b.b
                );
            }
        }
    }
    if false {
        for (k, (b, r)) in desc.beams.iter().zip(&rb).enumerate() {
            let checks = [
                ("node a", f64::from(b.a), r[1]),
                ("node b", f64::from(b.b), r[2]),
                ("spring", f64::from(b.spring), r[3]),
                ("damp", f64::from(b.damp), r[4]),
                ("deform", f64::from(b.deform), r[7]),
                ("strength", f64::from(b.strength), r[9]),
                ("plastic", f64::from(b.plastic_coef), r[10]),
            ];
            for (name, ours, theirs) in checks {
                let tol = 1e-4 * theirs.abs().max(1.0);
                if (ours - theirs).abs() > tol && bad < 20 {
                    bad += 1;
                    println!("beam {k} {name}: rubble {ours} ror {theirs}");
                }
            }
        }
    }
    println!("beam parameter mismatches shown: {bad} (max 30)");
}

fn kv<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(&format!("{key}=")))
}

fn floats(s: &str) -> Vec<f32> {
    s.split(',').map(|v| v.parse().unwrap()).collect()
}

/// The harness's `free` scene with [`RorSim`]; per-node distance to RoR's result.
fn free(truck: &str, nodes_csv: &str, args: &[String]) {
    let desc = load_truck(&std::fs::read_to_string(truck).unwrap());
    let v3 = |k: &str| kv(args, k).map_or(Vec3::ZERO, |s| Vec3::from_slice(&floats(s)));
    let drag = kv(args, "drag") != Some("0");
    let secs: f32 = kv(args, "secs").map_or(1.0, |s| s.parse().unwrap());
    let settings = RorSettings {
        drag,
        turbulence: drag,
        fast_inv_sqrt: kv(args, "fastinv") != Some("0"),
    };
    // With `wheels`, RoR's soft tyres (as the harness spawns them).
    let text = std::fs::read_to_string(truck).unwrap();
    let mut sim = RorSim::from_truck(&text, v3("pos"), v3("vel"), settings);
    if let Some(b) = kv(args, "box") {
        let c = floats(b);
        let (centre, half) = (Vec3::new(c[0], c[1], c[2]), Vec3::new(c[3], c[4], c[5]));
        sim.boxes.push(RorBox {
            lo: centre - half,
            hi: centre + half,
        });
    }
    let t = std::time::Instant::now();
    sim.run(secs);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let ror = csv(nodes_csv);
    let x = sim.positions();
    let d: Vec<f32> = x
        .iter()
        .zip(&ror)
        .map(|(p, r)| (*p - Vec3::new(r[5] as f32, r[6] as f32, r[7] as f32)).length())
        .collect();
    let rms = (d.iter().map(|v| v * v).sum::<f32>() / d.len() as f32).sqrt();
    let max = d.iter().cloned().fold(0.0, f32::max);
    let worst = (0..d.len())
        .max_by(|&a, &b| d[a].total_cmp(&d[b]))
        .unwrap_or(0);
    let mut order: Vec<usize> = (0..d.len()).collect();
    order.sort_by(|&a, &b| d[b].total_cmp(&d[a]));
    for &i in order.iter().take(6) {
        println!("  node {i}: {:.4} mm", d[i] * 1e3);
    }
    println!(
        "worst node {worst}: rubble {:?} ror {:?}",
        x[worst],
        &ror[worst][5..8]
    );
    let travel = x
        .iter()
        .zip(&ror)
        .map(|(p, r)| (*p - Vec3::new(r[2] as f32, r[3] as f32, r[4] as f32)).length())
        .fold(0.0, f32::max);
    // beams_csv=<harness beams_out>: per-beam stress (last step) by node pair.
    if let Some(bc) = kv(args, "beams_csv") {
        let rb = csv(bc);
        let mut by_pair: std::collections::HashMap<(u32, u32), Vec<usize>> = Default::default();
        for (i, r) in rb.iter().enumerate() {
            let (a, b) = (r[1] as u32, r[2] as u32);
            by_pair.entry((a.min(b), a.max(b))).or_default().push(i);
        }
        let mut shown = 0;
        for (k, b) in sim.beams.iter().enumerate() {
            let r = &rb[by_pair
                .get_mut(&(b.a.min(b.b), b.a.max(b.b)))
                .unwrap()
                .remove(0)];
            let d = (f64::from(b.stress) - r[3]).abs();
            if d > 1e-3 * r[3].abs().max(1.0) && shown < 12 {
                shown += 1;
                println!(
                    "  beam {k} ({}-{}) stress rubble {} ror {} (L {} / {})",
                    b.a, b.b, b.stress, r[3], b.l, r[4]
                );
            }
        }
    }
    let broken = sim.beams.iter().filter(|b| b.broken).count();
    let deformed = sim
        .beams
        .iter()
        .filter(|b| (b.l - b.ref_l).abs() > 1e-4)
        .count();
    println!(
        "engine_softbody::ror vs Rigs of Rods after {secs} s: node distance RMS {:.3} mm, max {:.3} mm (largest travel {:.2} m); rubble: broken {broken}, deformed {deformed}; {ms:.1} ms CPU",
        rms * 1e3,
        max * 1e3,
        travel
    );
}

//! Compares Rubble against the Rigs of Rods reference harness (`inspiration/ror-harness`,
//! local only) on the same `.truck` file. Reads the harness's CSV dumps; nothing here
//! depends on Rigs of Rods code.
//!
//! ```bash
//! # node masses and beam parameters of the loaded desc against RoR's spawn
//! cargo run -p engine-softbody --release --example ror_compare -- spawn <file.truck> <nodes.csv> [beams.csv]
//! # the body without wheels/engine/brakes, for both sides
//! cargo run -p engine-softbody --release --example ror_compare -- bare <in.truck> <out.truck>
//! # Rubble's explicit-beams solver on the harness's `free` scene, against RoR's node dump
//! cargo run -p engine-softbody --release --example ror_compare -- free <file.truck> <nodes.csv> pos=x,y,z vel=… box=… drag=0 secs=1
//! ```
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use engine_softbody::procedural::strip_truck_sections;
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
        _ => eprintln!(
            "usage: ror_compare spawn <file.truck> <nodes.csv> [beams.csv] | bare <in> <out>"
        ),
    }
}

/// Node masses, then beam parameters (matched by node pair: RoR's spawn order
/// differs from the file's), of the loaded desc against RoR's spawn dumps.
fn spawn(truck: &str, nodes_csv: &str, beams_csv: Option<&str>) {
    let src = std::fs::read_to_string(truck).unwrap();
    for w in &parse_truck(&src).warnings {
        println!("rubble loader warning: {w}");
    }
    let desc = load_truck(&src);
    let ror = csv(nodes_csv);
    println!("nodes: rubble {} ror {}", desc.nodes.len(), ror.len());
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
    println!("total mass (desc nodes): rubble {rm:.2} kg, ror {om:.2} kg");
    let i = worst.1;
    println!(
        "worst node mass: node {i} rubble {:.4} ror {:.4} ({:.2}%)",
        desc.nodes[i].mass,
        ror[i][1],
        worst.0 * 100.0
    );
    let Some(bc) = beams_csv else { return };
    // columns: i,p1,p2,k,d,L,refL,maxpos,maxneg,strength,plastic,bounded,type,detacher
    let rb = csv(bc);
    println!("beams: rubble {} ror {}", desc.beams.len(), rb.len());
    let mut by_pair: std::collections::HashMap<(u32, u32), Vec<usize>> = Default::default();
    for (i, r) in rb.iter().enumerate() {
        let (a, b) = (r[1] as u32, r[2] as u32);
        by_pair.entry((a.min(b), a.max(b))).or_default().push(i);
    }
    let mut bad = 0;
    for (k, b) in desc.beams.iter().enumerate() {
        let Some(list) = by_pair.get_mut(&(b.a.min(b.b), b.a.max(b.b))) else {
            println!("beam {k} ({}-{}) not in RoR", b.a, b.b);
            bad += 1;
            continue;
        };
        let r = &rb[list.remove(0)];
        let checks = [
            ("spring", f64::from(b.spring), r[3]),
            ("damp", f64::from(b.damp), r[4]),
            ("strength", f64::from(b.strength), r[9]),
            ("plastic", f64::from(b.plastic_coef), r[10]),
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
    println!("beam parameter mismatches shown: {bad} (max 30)");
}

fn kv<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(&format!("{key}=")))
}

fn floats(s: &str) -> Vec<f32> {
    s.split(',').map(|v| v.parse().unwrap()).collect()
}

/// The harness's `free` scene with Rubble's explicit-beams solver; per-node distance
/// to RoR's result and the damage counts.
fn free(truck: &str, nodes_csv: &str, args: &[String]) {
    let text = std::fs::read_to_string(truck).unwrap();
    let v3 = |k: &str| kv(args, k).map_or(Vec3::ZERO, |s| Vec3::from_slice(&floats(s)));
    let drag = kv(args, "drag") != Some("0");
    let secs: f32 = kv(args, "secs").map_or(1.0, |s| s.parse().unwrap());
    let settings = ExplicitBeamsSettings {
        drag,
        turbulence: drag,
        fast_inv_sqrt: kv(args, "fastinv") != Some("0"),
    };
    let mut sim = ExplicitBeams::from_truck(&text, v3("pos"), v3("vel"), settings).unwrap();
    if let Some(b) = kv(args, "box") {
        let c = floats(b);
        sim.boxes.push(StaticBox::centered(
            Vec3::new(c[0], c[1], c[2]),
            Vec3::new(c[3], c[4], c[5]),
        ));
    }
    // The harness steps exactly round(secs / 0.5 ms) steps, rebasing every 33.
    let steps = (secs / engine_softbody::explicit_beams::STEP).round() as u64;
    let t = std::time::Instant::now();
    for i in 0..steps {
        if i % 33 == 0 {
            sim.rebase_origin();
        }
        sim.step();
    }
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
    let (broken, bent) = sim.damage();
    println!(
        "explicit_beams vs Rigs of Rods after {secs} s: nodes {} / {}, node distance RMS {:.3} mm, max {:.3} mm; rubble: broken {broken}, bent {bent}; {ms:.1} ms CPU",
        x.len(),
        ror.len(),
        rms * 1e3,
        max * 1e3
    );
}

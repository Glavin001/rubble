//! Compares Rubble against the Rigs of Rods reference harness (`inspiration/ror-harness`,
//! local only) on the same `.truck` file. Reads the harness's CSV dumps; nothing here
//! depends on Rigs of Rods code.
//!
//! ```bash
//! # node masses and beam parameters of the loaded desc against RoR's spawn
//! cargo run -p engine-softbody --release --example ror_compare -- spawn <file.truck> <nodes.csv> [beams.csv]
//! # the body without wheels/engine/brakes, for both sides
//! cargo run -p engine-softbody --release --example ror_compare -- bare <in.truck> <out.truck>
//! ```
use engine_softbody::procedural::strip_truck_sections;
use engine_softbody::truck::{load_truck, parse_truck};

fn csv(path: &str) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(|v| v.parse().unwrap_or(f64::NAN)).collect())
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("spawn") => spawn(&a[1], &a[2], a.get(3).map(String::as_str)),
        Some("bare") => {
            let t = std::fs::read_to_string(&a[1]).unwrap();
            std::fs::write(&a[2], strip_truck_sections(&t, &["wheels", "engine", "brakes"]))
                .unwrap();
        }
        _ => eprintln!("usage: ror_compare spawn <file.truck> <nodes.csv> [beams.csv] | bare <in> <out>"),
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
                println!("beam {k} ({}-{}) {name}: rubble {ours} ror {theirs}", b.a, b.b);
            }
        }
    }
    println!("beam parameter mismatches shown: {bad} (max 30)");
}

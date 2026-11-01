//! Compares Rubble against the Rigs of Rods reference harness (`inspiration/ror-harness`,
//! local only) on the same `.truck` file. Reads the harness's CSV dumps; nothing here
//! depends on Rigs of Rods code.
//!
//! ```bash
//! cargo run -p engine-softbody --release --example ror_compare -- spawn <file.truck> <nodes.csv> <beams.csv>
//! ```
use engine_softbody::truck::{load_truck, parse_truck};

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
    let mut bad = 0;
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
    println!("beam parameter mismatches shown: {bad} (max 20)");
}

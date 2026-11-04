//! Compares Rubble against the Rigs of Rods reference harness (`inspiration/ror-harness`,
//! local only) on the same `.truck` file. Reads the harness's CSV dumps; nothing here
//! depends on Rigs of Rods code.
//!
//! ```bash
//! # node masses and beam parameters of the loaded desc against RoR's spawn
//! cargo run -p engine-softbody --release --example ror_compare -- spawn <file.truck> <nodes.csv> [beams.csv]
//! # the body without wheels/engine/brakes, for both sides
//! cargo run -p engine-softbody --release --example ror_compare -- bare <in.truck> <out.truck>
//! # the procedural sedan as a .truck file for both sides: stations along its length
//! # (21 by default; more for a higher resolution), minimum node mass, bare=1 body only
//! cargo run -p engine-softbody --release --example ror_compare -- sedan <out.truck> stations=61 minimass=10
//! # a BeamNG JBeam vehicle (a directory of .jbeam files, optional .pc config) as .truck
//! cargo run -p engine-softbody --release --example ror_compare -- jbeam <dir-or-file> <out.truck> [config.pc]
//! # Rubble's high-resolution test car (JBeam) as .truck
//! cargo run -p engine-softbody --release --example ror_compare -- hires <out.truck> [node_weight=1] [spring=3e6] [damp=250] [stiffness_length=0.03]
//! # Rubble's explicit-beams solver on the harness's `free` scene, against RoR's node dump
//! cargo run -p engine-softbody --release --example ror_compare -- free <file.truck> <nodes.csv> pos=x,y,z vel=… box=… drag=0 roll=1 secs=1
//! ```
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use engine_softbody::hires::{hires_car_jbeam, HiresParams};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::procedural::{car_truck, car_truck_bare, strip_truck_sections, CarParams};
use engine_softbody::truck::{load_truck, parse_truck};
use engine_softbody::truck_export::{desc_to_truck, TyreDefaults};
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
        // The vehicle standing on the ground (lowest node 2 cm up) for secs=: the
        // beams that broke or bent, with their nodes, kind and strength.
        Some("rest") => {
            let text = std::fs::read_to_string(&a[1]).unwrap();
            let secs: f32 = kv(&a[2..], "secs").map_or(2.0, |s| s.parse().unwrap());
            let mut sim =
                ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, Default::default())
                    .unwrap();
            let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            for n in &mut sim.nodes {
                n.pos.y += 0.02 - lowest;
            }
            let steps = (secs / engine_softbody::explicit_beams::STEP).round() as u32;
            let (mut total, mut most) = (0u64, 0u32);
            for i in 0..steps {
                if i % 33 == 0 {
                    sim.rebase_origin();
                }
                sim.step();
                total += u64::from(sim.last_contacts);
                most = most.max(sim.last_contacts);
            }
            let (broken, bent) = sim.damage();
            println!(
                "after {secs} s: broken {broken}, bent {bent}; self-collision contacts per step: mean {:.1}, max {most} ({} triangles, {} contacters)",
                total as f64 / f64::from(steps.max(1)),
                sim.triangles.len(),
                sim.candidates.len()
            );
            for (i, b) in sim
                .beams
                .iter()
                .enumerate()
                .filter(|(_, b)| b.broken)
                .take(40)
            {
                println!(
                    "broken beam {i}: nodes {}-{} ({:.2} / {:.2} kg) kind {:?} strength {:.0}",
                    b.a,
                    b.b,
                    sim.nodes[b.a as usize].mass,
                    sim.nodes[b.b as usize].mass,
                    b.kind,
                    b.strength
                );
            }
        }
        Some("jbeam") | Some("hires") => {
            let (owned, out): (Vec<(String, String)>, &str) = if a[0] == "hires" {
                // Tunables: node_weight= (kg), spring= (N/m for a 0.15 m beam), damp=.
                let d = HiresParams::default();
                let p = HiresParams {
                    node_weight: kv(&a[2..], "node_weight")
                        .map_or(d.node_weight, |s| s.parse().unwrap()),
                    spring: kv(&a[2..], "spring").map_or(d.spring, |s| s.parse().unwrap()),
                    damp: kv(&a[2..], "damp").map_or(d.damp, |s| s.parse().unwrap()),
                    stiffness_length: kv(&a[2..], "stiffness_length")
                        .map_or(d.stiffness_length, |s| s.parse().unwrap()),
                    ..d
                };
                let (text, _) = hires_car_jbeam(&p);
                (vec![("hires.jbeam".into(), text)], &a[1])
            } else {
                let path = std::path::Path::new(&a[1]);
                let files: Vec<std::path::PathBuf> = if path.is_dir() {
                    let mut v: Vec<_> = std::fs::read_dir(path)
                        .unwrap()
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| p.extension().is_some_and(|x| x == "jbeam"))
                        .collect();
                    v.sort();
                    v
                } else {
                    vec![path.to_path_buf()]
                };
                (
                    files
                        .iter()
                        .map(|f| (f.display().to_string(), std::fs::read_to_string(f).unwrap()))
                        .collect(),
                    &a[2],
                )
            };
            let config = a
                .get(3)
                .filter(|_| a[0] == "jbeam")
                .map(|p| std::fs::read_to_string(p).unwrap());
            let files: Vec<(&str, &str)> = owned
                .iter()
                .map(|(n, t)| (n.as_str(), t.as_str()))
                .collect();
            let desc = load_jbeam(&files, config.as_deref()).unwrap();
            for w in &desc.warnings {
                println!("jbeam warning: {w}");
            }
            let (text, warnings) = desc_to_truck(&desc, TyreDefaults::for_vehicle(&desc));
            for w in &warnings {
                println!("conversion: {w}");
            }
            let sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, Default::default())
                .unwrap();
            println!(
                "{out}: {} nodes ({} in the JBeam), {} beams, {} collision triangles, {:.1} kg",
                sim.nodes.len(),
                desc.nodes.len(),
                sim.beams.len(),
                sim.triangles.len(),
                sim.nodes.iter().map(|n| n.mass).sum::<f32>()
            );
            std::fs::write(out, text).unwrap();
        }
        Some("sedan") => {
            let p = CarParams {
                stations: kv(&a[2..], "stations").map_or(21, |s| s.parse().unwrap()),
                minimass: kv(&a[2..], "minimass").map_or(10.0, |s| s.parse().unwrap()),
                ..CarParams::sedan()
            };
            let text = if kv(&a[2..], "bare") == Some("1") {
                car_truck_bare(&p)
            } else {
                car_truck(&p)
            };
            let sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, Default::default())
                .unwrap();
            println!(
                "{}: {} stations, {} nodes, {} beams, {} collision triangles",
                a[1],
                p.stations,
                sim.nodes.len(),
                sim.beams.len(),
                sim.triangles.len()
            );
            std::fs::write(&a[1], text).unwrap();
        }
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
            // RoR's yield threshold at spawn (max positive and negative stress).
            ("deform+", f64::from(b.deform.min(1e30)), r[7].min(1e30)),
            ("deform-", f64::from(b.deform.min(1e30)), (-r[8]).min(1e30)),
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
    if kv(args, "roll") == Some("1") {
        sim.roll_tyres(v3("vel"));
    }
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
    // `out=<file.csv>`: Rubble's end positions in the harness's column layout
    // (rest and mass copied from Rigs of Rods' rows), for ensemble statistics.
    if let Some(path) = kv(args, "out") {
        let mut text = String::from("node,mass,rest_x,rest_y,rest_z,x,y,z\n");
        for (i, (p, r)) in x.iter().zip(&ror).enumerate() {
            text += &format!(
                "{i},{},{},{},{},{},{},{}\n",
                r[1], r[2], r[3], r[4], p.x, p.y, p.z
            );
        }
        std::fs::write(path, text).unwrap();
    }
    let d: Vec<f32> = x
        .iter()
        .zip(&ror)
        .map(|(p, r)| (*p - Vec3::new(r[5] as f32, r[6] as f32, r[7] as f32)).length())
        .collect();
    let rms = (d.iter().map(|v| v * v).sum::<f32>() / d.len() as f32).sqrt();
    let max = d.iter().cloned().fold(0.0, f32::max);
    // `worst=N`: the N nodes furthest from Rigs of Rods' (index, distance, both positions).
    if let Some(n) = kv(args, "worst").map(|s| s.parse::<usize>().unwrap()) {
        let mut idx: Vec<usize> = (0..d.len()).collect();
        idx.sort_by(|&a, &b| d[b].total_cmp(&d[a]));
        for &i in idx.iter().take(n) {
            let r = &ror[i];
            println!(
                "node {i}: {:.1} mm, rubble {:?} ror ({:.4}, {:.4}, {:.4})",
                d[i] * 1e3,
                x[i],
                r[5],
                r[6],
                r[7]
            );
        }
    }
    let (broken, bent) = sim.damage();
    // Mean node x of each engine: tells a stop from a pass-through at a glance.
    let mean_x = |v: &mut dyn Iterator<Item = f64>| {
        let (s, n) = v.fold((0.0, 0.0), |(s, n), x| (s + x, n + 1.0));
        s / n
    };
    let rubble_x = mean_x(&mut x.iter().map(|p| f64::from(p.x)));
    let ror_x = mean_x(&mut ror.iter().map(|r| r[5]));
    println!(
        "explicit_beams vs Rigs of Rods after {secs} s: nodes {} / {}, node distance RMS {:.3} mm, max {:.3} mm; rubble: broken {broken}, bent {bent}; mean node x rubble {rubble_x:.2} ror {ror_x:.2}; {ms:.1} ms CPU",
        x.len(),
        ror.len(),
        rms * 1e3,
        max * 1e3
    );
}

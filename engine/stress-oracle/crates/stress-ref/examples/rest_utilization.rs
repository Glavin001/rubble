//! How loaded each bond is at rest: the scene's gravity prestress (the world's own
//! static equilibrium at t = 0), then every bond's failure index there, evaluated
//! without committing anything (no fracture, no history). The distribution decides
//! which skip certificates can ever pass: an energy certificate needs the island's
//! bonds well below their strength at rest.
//!
//!   cargo run --release --example rest_utilization -- <pack.json | scene.json> ...

use stress_ref::scene::Scene;
use stress_ref::world::World;

const EDGES: [f64; 8] = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, f64::INFINITY];

fn load(path: &str) -> Result<Scene, String> {
    let p = std::path::Path::new(path);
    let name = p.file_stem().unwrap().to_string_lossy().to_string();
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if json.get("scenario").is_some() || json["format"] == "shardwright-stress/1" {
        stress_ref::scene_pack::import(p, &name, "structure")
    } else {
        Scene::load(p)
    }
}

fn main() {
    for path in std::env::args().skip(1) {
        let mut scene = match load(&path) {
            Ok(s) => s,
            Err(e) => {
                println!("{path}: {e}");
                continue;
            }
        };
        scene.sim.gravity_prestress = true;
        let t = std::time::Instant::now();
        let world = World::new(&scene);
        let m = &world.solver;
        let mut counts = [0usize; EDGES.len()];
        let mut worst: Vec<(f64, String)> = Vec::new();
        let mut n = 0usize;
        for (s, bonds) in m.bonds.iter().enumerate() {
            let st = &m.structures[s];
            for b in bonds.iter().filter(|b| b.connected()) {
                let (a, c) = (&m.chunks[s][b.geometry.a], &m.chunks[s][b.geometry.b]);
                let d = b.geometry.kinematics(a.u, a.th, c.u, c.th);
                let u = b.model().evaluate(&b.joint, &d, 0.0, false).state.utilization;
                counts[EDGES.iter().position(|&e| u < e).unwrap()] += 1;
                n += 1;
                let kind = |c: usize| st.chunks[c].groups.first().cloned().unwrap_or_else(|| st.chunks[c].material.clone());
                worst.push((u, format!("{} {}-{}", st.chunks[b.geometry.a].material, kind(b.geometry.a), kind(b.geometry.b))));
            }
        }
        worst.sort_by(|x, y| y.0.total_cmp(&x.0));
        let share = |k: usize| 100.0 * counts[k..].iter().sum::<usize>() as f64 / n.max(1) as f64;
        println!(
            "{:<24} {:>6} bonds  >0.3 {:>5.1}%  >0.4 {:>5.1}%  >0.6 {:>5.1}%  >1 {:>5.1}% ({} bonds)  max {:.2} ({})  [{:.1} s]",
            scene.name,
            n,
            share(3),
            share(4),
            share(5),
            share(7),
            counts[7],
            worst.first().map_or(0.0, |w| w.0),
            worst.first().map_or(String::new(), |w| w.1.clone()),
            t.elapsed().as_secs_f64()
        );
        let hist: Vec<String> = EDGES.iter().zip(counts).map(|(e, k)| format!("<{e}: {k}")).collect();
        println!("    {}", hist.join("  "));
        let top: Vec<String> = worst.iter().take(5).map(|(u, w)| format!("{u:.2} {w}")).collect();
        println!("    most loaded: {}", top.join("; "));
    }
}

//! What static condensation buys on real buildings, two ways:
//!
//! 1. Partitions: recursive coordinate bisection into parts of N chunks; each part's
//!    chunks with no bond leaving it are eliminated. Reported: degrees of freedom kept,
//!    storage (f32), static PCG iterations full against reduced, and the explicit step
//!    gain of the Guyan dynamics (`sqrt(lambda_full / lambda_reduced)`).
//! 2. Slivers: eliminate, one at a time, the retained chunk carrying most of the current
//!    top mode's kinetic energy (the few light chunks that set the step), and report the
//!    step gain after 1, 2, 4, ... chunks.
//!
//!   cargo run --release -p stress-reduce --example condense_report -- [--iters N] [--slivers N] <pack.json> ...

use std::time::Instant;

use stress_reduce::{interior_of, pcg, rcb, top_eigen, Network, Reduced};
use stress_ref::math::Vec3;
use stress_ref::solver::ReferenceSolver;

fn main() {
    let mut iters = 1500usize;
    let mut max_slivers = 16usize;
    let mut paths = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).expect("--iters N"),
            "--slivers" => max_slivers = args.next().and_then(|v| v.parse().ok()).expect("--slivers N"),
            _ => paths.push(a),
        }
    }
    for path in paths {
        let p = std::path::Path::new(&path);
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let scene = match stress_ref::scene_pack::import(p, &name, "structure") {
            Ok(s) => s,
            Err(e) => {
                println!("{name}: {e}");
                continue;
            }
        };
        let m = ReferenceSolver::new(&scene);
        // The largest anchored bonded cluster.
        let Some(ci) = (0..m.clusters.len()).filter(|&c| m.clusters[c].anchored && !m.clusters[c].bonds.is_empty()).max_by_key(|&c| m.clusters[c].chunks.len()) else {
            println!("{name}: no anchored bonded cluster");
            continue;
        };
        let t = Instant::now();
        let net = Network::of_cluster(&m, ci, Vec3::from_array(scene.gravity));
        let (lam_full, x_full) = top_eigen(&net.k, &net.m, iters);
        // The chunks carrying the full model's top mode (kinetic share), largest first.
        let mx_full = net.m.mul(&x_full);
        let mut carriers: Vec<(usize, f64)> = (0..net.chunks.len()).map(|c| (c, net.dofs_of(c).map(|d| x_full[d] * mx_full[d]).sum::<f64>())).collect();
        carriers.sort_by(|a, b| b.1.total_cmp(&a.1));
        let carriers: Vec<usize> = carriers.iter().take_while(|e| e.1 > 0.01).map(|e| e.0).collect();
        let (_, full_cg) = pcg(&net.k, &net.gravity, 1e-10, 1_000_000);
        println!(
            "{name}: {} chunks, {} bonds, {} free dofs; full top {:.0} Hz, gravity PCG {full_cg} iterations [{:.1} s]",
            net.chunks.len(),
            net.bonds.len(),
            net.n,
            lam_full.sqrt() / std::f64::consts::TAU,
            t.elapsed().as_secs_f64()
        );
        println!("  top-mode carriers (>1% of its kinetic energy): {} chunks", carriers.len());
        println!("  | parts of | dofs kept | components | storage MB (f32) | gravity PCG | step gain | carriers eliminated | build s |");
        println!("  |---|---|---|---|---|---|---|---|");
        // Authored partitions where the asset has them (Shardwright `member:` = L0,
        // `l1:`; scene packs' pieces would be `piece:`), then coordinate bisection.
        let st0 = &m.structures[m.clusters[ci].structure];
        let mut partitions: Vec<(String, Vec<usize>)> = Vec::new();
        for prefix in ["member:", "l1:", "piece:"] {
            let keys: Vec<Option<String>> = net.chunks.iter().map(|&c| st0.chunks[c].groups.iter().find(|g| g.starts_with(prefix)).cloned()).collect();
            if keys.iter().all(Option::is_some) {
                let mut ids = std::collections::BTreeMap::new();
                let part: Vec<usize> = keys.iter().map(|k| {
                    let n = ids.len();
                    *ids.entry(k.clone().unwrap()).or_insert(n)
                }).collect();
                partitions.push((format!("{} ({} parts)", prefix.trim_end_matches(':'), ids.len()), part));
            }
        }
        for target in [8usize, 16, 32, 64, 128, 256] {
            if target < 2 * net.chunks.len() {
                partitions.push((target.to_string(), rcb(&net.centers, target)));
            }
        }
        let mut interiors = Vec::new();
        for (label, part) in partitions {
            let t = Instant::now();
            let interior = interior_of(&net, &part);
            let red = Reduced::new(&net, &interior);
            let hit = carriers.iter().filter(|&&c| interior[c]).count();
            let build = t.elapsed().as_secs_f64();
            let (_, cg) = pcg(&red.k, &red.load(&net.gravity), 1e-10, 1_000_000);
            let (lam, _) = top_eigen(&red.k, &red.m, iters);
            println!(
                "  | {label} | {} ({:.0}%) | {} | {:.1} | {cg} | x{:.2} | {hit}/{} | {build:.1} |",
                red.nr,
                100.0 * red.nr as f64 / net.n as f64,
                red.components.len(),
                red.floats() as f64 * 4.0 / 1e6,
                (lam_full / lam).sqrt(),
                carriers.len()
            );
            interiors.push((label, interior));
        }
        // Slivers: greedy elimination of the top mode's carriers.
        let mut eliminate = vec![false; net.chunks.len()];
        let mut report_at = 1usize;
        let mut line = Vec::new();
        let st = &m.structures[m.clusters[ci].structure];
        let mut first = Vec::new();
        for count in 1..=max_slivers {
            let red = Reduced::new(&net, &eliminate);
            let (_, x) = top_eigen(&red.k, &red.m, iters);
            let mx = red.m.mul(&x);
            // Kinetic share per retained chunk.
            let best = (0..net.chunks.len())
                .filter(|&c| !eliminate[c])
                .map(|c| (c, net.dofs_of(c).filter_map(|d| red.keep[d]).map(|r| x[r] * mx[r]).sum::<f64>()))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            let Some((c, _)) = best else { break };
            eliminate[c] = true;
            if first.len() < 6 {
                let ch = &st.chunks[net.chunks[c]];
                first.push(format!("{:.1} kg {} {}", ch.mass, ch.material, ch.groups.first().cloned().unwrap_or_default()));
            }
            if count == report_at {
                let red = Reduced::new(&net, &eliminate);
                let (lam, _) = top_eigen(&red.k, &red.m, iters);
                line.push(format!("{count}: x{:.2}", (lam_full / lam).sqrt()));
                report_at *= 2;
            }
        }
        println!("  slivers eliminated -> step gain  {}", line.join("   "));
        // Partitions and slivers together: does condensing the interiors add to the
        // slivers' gain (the eliminated set grows, the step can only improve)?
        let mut combined = Vec::new();
        for (label, interior) in &interiors {
            let both: Vec<bool> = interior.iter().zip(&eliminate).map(|(a, b)| *a || *b).collect();
            let red = Reduced::new(&net, &both);
            let (lam, _) = top_eigen(&red.k, &red.m, iters);
            combined.push(format!("{label}: x{:.2} ({:.0}% kept)", (lam_full / lam).sqrt(), 100.0 * red.nr as f64 / net.n as f64));
        }
        println!("  partition + the {} slivers -> {}", eliminate.iter().filter(|&&e| e).count(), combined.join("   "));
        println!("  first eliminated: {}", first.join("; "));
    }
}

//! Work per 60 Hz frame of `sim.methods.multirate` against single rate, from the rates
//! the integrator itself assigns (`ReferenceSolver::bond_rates`), at the world's substep
//! after its prestress:
//!
//!   cargo run --release --example multirate_work -- [--methods LIST] <scene.json | pack.json> ...
//!
//! `--methods` is added to `STRESS_METHODS`' list (e.g. `true_step`). Columns, per frame:
//! - `ticks`: substeps (single rate and multirate take the same);
//! - `bonds`: bond evaluations, single over multirate (`sum_b ceil(n / r_b)`);
//! - `chunks`: chunk updates if a chunk is updated only at its finest bond's rate (the
//!   GPU design's chunk kernels; the oracle drifts every chunk every substep);
//! - `A1`: REALTIME_ROADMAP A1's multirate column on this substep: each chunk at its
//!   own native step binned in powers of two, with no neighbour taking the finer rate;
//! - `bonds w/o contact`: the bond ratio with the chunks' contact bounds left out (what
//!   the bond network alone allows);
//! - `binds`: whether the stress or the contact bound sets the substep;
//! - the bond count per rate.

use stress_ref::scene::Scene;
use stress_ref::world::World;

fn load(path: &str) -> Result<Scene, String> {
    let p = std::path::Path::new(path);
    let name = p.file_stem().unwrap().to_string_lossy().to_string();
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if json.get("scenario").is_some() || json["format"] == "shardwright-stress/1" || json.get("bond_level").is_some() {
        stress_ref::scene_pack::import(p, &name, "structure")
    } else {
        Scene::load(p)
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(k) = args.iter().position(|a| a == "--methods") {
        let list = args.remove(k + 1);
        args.remove(k);
        let mut methods = std::env::var("STRESS_METHODS").unwrap_or_default();
        methods = if methods.is_empty() { list } else { format!("{methods},{list}") };
        std::env::set_var("STRESS_METHODS", methods);
    }
    println!(
        "{:<28} {:>7} {:>7} {:>8} {:>7} {:>7} {:>7} {:>9}  {:<7} rates (bonds per rate)",
        "scene", "chunks", "bonds", "ticks", "bonds", "chunks", "A1", "w/o cont", "binds"
    );
    for path in &args {
        let mut scene = match load(path) {
            Ok(s) => s,
            Err(e) => {
                println!("{path}: {e}");
                continue;
            }
        };
        scene.sim.methods.multirate = true;
        scene.sim.frame_dt = 1.0 / 60.0;
        let world = World::new(&scene);
        let sv = &world.solver;
        let dt = world.substep_dt();
        let n = (scene.sim.frame_dt / dt).round().max(1.0) as u32;
        let cap = n.next_power_of_two();
        let contact = world.chunk_contact_dts();
        let with = |s: usize, c: usize| contact[s][c];
        let without = |_: usize, _: usize| f64::INFINITY;
        let per = |r: u32| (n as f64 / r as f64).ceil();
        let (mut bonds, mut single, mut bare) = (0.0, 0.0, 0.0);
        let (mut chunk_mr, mut chunk_single, mut a1) = (0.0, 0.0, 0.0);
        let mut histogram = std::collections::BTreeMap::<u32, usize>::new();
        let mut n_bonds = 0usize;
        for ci in 0..sv.clusters.len() {
            let cl = &sv.clusters[ci];
            if cl.bonds.is_empty() {
                continue;
            }
            let s = cl.structure;
            let rates = sv.bond_rates(ci, dt, cap, &with);
            let mut chunk_rate = vec![u32::MAX; sv.structures[s].chunks.len()];
            for &(bi, r) in &rates {
                bonds += per(r);
                single += n as f64;
                *histogram.entry(r).or_default() += 1;
                let g = &sv.bonds[s][bi].geometry;
                chunk_rate[g.a] = chunk_rate[g.a].min(r);
                chunk_rate[g.b] = chunk_rate[g.b].min(r);
            }
            n_bonds += rates.len();
            for (_, r) in sv.bond_rates(ci, dt, cap, &without) {
                bare += per(r);
            }
            for (c, native) in sv.chunk_native_dts(ci, &with) {
                chunk_single += n as f64;
                chunk_mr += per(if chunk_rate[c] == u32::MAX { 1 } else { chunk_rate[c] });
                let level = (native / dt).log2().floor().max(0.0).min(cap.trailing_zeros() as f64);
                a1 += (n as f64 / 2f64.powf(level)).ceil();
            }
        }
        let chunks: usize = sv.structures.iter().map(|s| s.chunks.len()).sum();
        let binds = if world.contact_dt() < sv.stable_dt() { "contact" } else { "stress" };
        let hist: Vec<String> = histogram.iter().map(|(r, k)| format!("{r}:{k}")).collect();
        let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 1.0 };
        println!(
            "{:<28} {:>7} {:>7} {:>8} {:>6.2}x {:>6.2}x {:>6.2}x {:>8.2}x  {:<7} {}",
            scene.name,
            chunks,
            n_bonds,
            n,
            ratio(single, bonds),
            ratio(chunk_single, chunk_mr),
            ratio(chunk_single, a1),
            ratio(single, bare),
            binds,
            hist.join(" ")
        );
    }
}

//! The reference's stable substep against the true stable step (power iteration) on the
//! catalogue and the scene packs: how many substeps per 60 Hz frame each needs.
//!
//!   stress-gpu-steps [--levers] [--iters N] [pack.json ...]
//!
//! `--levers`: per scene, what would change the substep count, each as a ratio of
//! substeps per frame (at safety 0.9, the power-iteration estimate with its 3 % margin):
//!   certified    the oracle's certified frequency bound (`sim.methods.true_step`), with
//!                its damping at the largest dashpot-to-stiffness ratio of any bond
//!   per island   each island at its own step instead of the scene's smallest
//!                (chunk-substeps per frame, global over per island)
//!   damping off  the top mode's step without its damping (zeta_top)
//!   rot x4..x100 every chunk's rotational inertia scaled (the top mode recomputed)
//!   multirate    each chunk at its own Gershgorin step, floored at the island's true
//!                step and binned in powers of two (chunk-substeps, global over binned)
//!   bond rule    the oracle's `sim.methods.multirate` rule at the same substep: each
//!                bond at its finer chunk's binned step (`ReferenceSolver::bond_rates`
//!                without contact), as bond evaluations and as chunk updates at each
//!                chunk's finest bond (both global over multirate)
//! and the chunks that carry the top mode (its kinetic energy share at the mode shape):
//! the slivers that set the step.

use stress_gpu::stability::{stable_step, top_mode_detail, true_stable_dt, TopMode};
use stress_ref::scene::Scene;
use stress_ref::solver::ReferenceSolver;

const FRAME: f64 = 1.0 / 60.0;
const SAFETY: f64 = 0.9;

fn per_frame(dt: f64) -> f64 {
    (FRAME / dt).ceil()
}

fn report(scene: &Scene) {
    let m = ReferenceSolver::new(scene);
    let chunks: usize = m.structures.iter().map(|s| s.chunks.len()).sum();
    let reference = m.stable_dt();
    let t = std::time::Instant::now();
    let truth = true_stable_dt(&m, 400, SAFETY);
    let took = t.elapsed().as_secs_f64();
    println!(
        "{:<30} {:>7} chunks  reference {:>10.3e} s ({:>10.0}/frame)  true {:>10.3e} s ({:>8.0}/frame)  x{:>8.1}  ({took:.2} s)",
        scene.name,
        chunks,
        reference,
        per_frame(reference),
        truth,
        per_frame(truth),
        truth / reference
    );
}

/// One island's top mode under each lever.
struct Island {
    ci: usize,
    chunks: usize,
    mode: TopMode,
    dt: f64,
    undamped_dt: f64,
    rot: [f64; 3],
    /// (chunk, share of the mode's kinetic energy), largest first.
    carriers: Vec<(usize, f64)>,
    /// Chunk-substeps per frame with each chunk at its binned own step.
    multirate_work: f64,
    /// The oracle's bond rule: bond evaluations and chunk updates per frame, single rate
    /// and multirate, and the bonds per rate.
    bond_rule: [f64; 4],
    bond_rates: std::collections::BTreeMap<u32, usize>,
}

fn island(m: &ReferenceSolver, rule: &ReferenceSolver, ci: usize, iters: usize) -> Option<Island> {
    let (mut mode, x) = top_mode_detail(m, ci, iters, 1.0)?;
    mode.omega *= 1.03;
    let dt = stable_step(mode, SAFETY);
    let undamped_dt = stable_step(TopMode { omega: mode.omega, zeta: 0.0 }, SAFETY);
    let rot = [4.0, 16.0, 100.0].map(|r| {
        top_mode_detail(m, ci, iters, r).map_or(f64::INFINITY, |(mut md, _)| {
            md.omega *= 1.03;
            stable_step(md, SAFETY)
        })
    });
    let cl = &m.clusters[ci];
    let st = &m.structures[cl.structure];
    let mut carriers: Vec<(usize, f64)> = cl
        .chunks
        .iter()
        .map(|&c| {
            let ch = &st.chunks[c];
            let mu = m.chunks[cl.structure][c].inertia_scale;
            (c, mu * (x[c].0.norm2() * ch.mass + x[c].1.dot(ch.inertia * x[c].1)))
        })
        .collect();
    let total: f64 = carriers.iter().map(|e| e.1).sum();
    carriers.iter_mut().for_each(|e| e.1 /= total.max(f64::MIN_POSITIVE));
    carriers.sort_by(|a, b| b.1.total_cmp(&a.1));
    carriers.truncate(5);
    // Each chunk's own (Gershgorin) step, no lower than the island's: binned 2^k.
    let n_global = per_frame(dt);
    let multirate_work = m
        .chunk_stable_dts(ci)
        .iter()
        .map(|&(_, own)| {
            let own = (own * SAFETY).max(dt);
            let level = (own / dt).log2().floor().max(0.0);
            (n_global / 2f64.powf(level)).ceil()
        })
        .sum::<f64>();
    // The oracle's rule at this substep: a bond at its finer chunk's rate, a chunk at its
    // finest bond's.
    let rates = rule.bond_rates(ci, dt, (n_global as u32).next_power_of_two(), &|_, _| f64::INFINITY);
    let mut chunk_rate = vec![u32::MAX; st.chunks.len()];
    let mut bond_rule = [0.0; 4];
    let mut bond_rates = std::collections::BTreeMap::new();
    for &(bi, r) in &rates {
        *bond_rates.entry(r).or_insert(0) += 1;
        bond_rule[0] += n_global;
        bond_rule[1] += (n_global / r as f64).ceil();
        let g = &m.bonds[cl.structure][bi].geometry;
        chunk_rate[g.a] = chunk_rate[g.a].min(r);
        chunk_rate[g.b] = chunk_rate[g.b].min(r);
    }
    for &c in &cl.chunks {
        bond_rule[2] += n_global;
        bond_rule[3] += (n_global / chunk_rate[c].min(n_global as u32).max(1) as f64).ceil();
    }
    Some(Island { ci, chunks: cl.chunks.len(), mode, dt, undamped_dt, rot, carriers, multirate_work, bond_rule, bond_rates })
}

fn levers(scene: &Scene, iters: usize) {
    let t = std::time::Instant::now();
    let m = ReferenceSolver::new(scene);
    let chunks: usize = m.structures.iter().map(|s| s.chunks.len()).sum();
    let bonds: usize = scene.bodies.iter().map(|b| b.bonds.len()).sum();
    let reference = m.stable_dt();
    // The bond rule's solver: the same scene at this tool's safety, Gershgorin steps only.
    let rule = {
        let mut s = scene.clone();
        s.sim.courant_safety = SAFETY;
        s.sim.stress_courant_safety = None;
        s.sim.methods.true_step = false;
        ReferenceSolver::new(&s)
    };
    let islands: Vec<Island> = (0..m.clusters.len()).filter_map(|ci| island(&m, &rule, ci, iters)).collect();
    if islands.is_empty() {
        println!("{}: no bonded islands", scene.name);
        return;
    }
    let global = islands.iter().map(|i| i.dt).fold(f64::INFINITY, f64::min);
    let certified = (0..m.structures.len()).map(|s| m.structure_true_step(s) * SAFETY).fold(f64::INFINITY, f64::min);
    // The certified frequency alone (no damping): separates the bound's two parts.
    let certified_w = (0..m.structures.len()).filter_map(|s| m.structure_frequency_bound(s)).map(|(w2, _)| w2.sqrt()).fold(0.0f64, f64::max);
    let n = per_frame(global);
    let island_chunks: f64 = islands.iter().map(|i| i.chunks as f64).sum();
    let work_global = island_chunks * n;
    let work_per_island: f64 = islands.iter().map(|i| i.chunks as f64 * per_frame(i.dt)).sum();
    let work_multirate: f64 = islands.iter().map(|i| i.multirate_work).sum();
    let rule: [f64; 4] = std::array::from_fn(|k| islands.iter().map(|i| i.bond_rule[k]).sum());
    let mut rates = std::collections::BTreeMap::<u32, usize>::new();
    for i in &islands {
        for (&r, &k) in &i.bond_rates {
            *rates.entry(r).or_insert(0) += k;
        }
    }
    let rates: Vec<String> = rates.iter().map(|(r, k)| format!("{r}:{k}")).collect();
    // The island that sets the scene's step decides the scene's levers.
    let top = islands.iter().min_by(|a, b| a.dt.total_cmp(&b.dt)).unwrap();
    let mut dts: Vec<f64> = islands.iter().map(|i| i.dt).collect();
    dts.sort_by(f64::total_cmp);
    let ratio = |dt: f64| per_frame(global) / per_frame(dt);
    println!("{} ({chunks} chunks, {bonds} bonds, {} bonded islands; {:.1} s)", scene.name, islands.len(), t.elapsed().as_secs_f64());
    println!(
        "  substeps/frame   reference {:>8.0}   true {:>6.0}   certified {:>6.0} (undamped {:.0}: w {:.3e} rad/s)",
        per_frame(reference),
        n,
        per_frame(certified),
        per_frame(SAFETY * 2.0 / certified_w),
        certified_w
    );
    let td = std::time::Instant::now();
    let damped = (0..m.structures.len()).map(|s| stress_gpu::stability::certified_damped_step(&m, s) * SAFETY).fold(f64::INFINITY, f64::min);
    println!("  certified with each bond's own dashpots {:>6.0} substeps/frame ({:.2} s)", per_frame(damped), td.elapsed().as_secs_f64());
    println!(
        "  island steps     min {:.3e}  median {:.3e}  max {:.3e} s   per-island stepping saves x{:.2}",
        dts[0],
        dts[dts.len() / 2],
        dts[dts.len() - 1],
        work_global / work_per_island
    );
    println!(
        "  top mode         w {:.3e} rad/s ({:.0} Hz)  zeta {:.3}   damping off x{:.2}   rot x4 x{:.2}  x16 x{:.2}  x100 x{:.2}   multirate x{:.2}",
        top.mode.omega,
        top.mode.omega / std::f64::consts::TAU,
        top.mode.zeta,
        ratio(top.undamped_dt),
        ratio(top.rot[0]),
        ratio(top.rot[1]),
        ratio(top.rot[2]),
        work_global / work_multirate
    );
    println!("  bond rule        bond evaluations x{:.2}   chunk updates x{:.2}   (multirate's rule at this substep; bonds per rate {})", rule[0] / rule[1].max(1.0), rule[2] / rule[3].max(1.0), rates.join(" "));
    let st = &m.structures[m.clusters[top.ci].structure];
    let carriers: Vec<String> = top
        .carriers
        .iter()
        .map(|&(c, share)| {
            let ch = &st.chunks[c];
            let kind = ch.groups.iter().find(|g| !g.starts_with("piece:")).cloned().unwrap_or_default();
            format!("#{c} {:.1} kg {} {kind} {:.0}%", ch.mass, ch.material, 100.0 * share)
        })
        .collect();
    println!("  top mode carried by  {}", carriers.join(", "));
}

fn main() {
    let mut paths = Vec::new();
    let mut with_levers = false;
    let mut iters = 400usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--levers" => with_levers = true,
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).expect("--iters N"),
            _ => paths.push(a),
        }
    }
    let run = |scene: &Scene| if with_levers { levers(scene, iters) } else { report(scene) };
    if paths.is_empty() {
        for scene in stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()) {
            run(&scene);
        }
    }
    for path in paths {
        let name = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        match stress_ref::scene_pack::import(std::path::Path::new(&path), &name, &name) {
            Ok(scene) => run(&scene),
            Err(e) => println!("{name}: {e}"),
        }
    }
}

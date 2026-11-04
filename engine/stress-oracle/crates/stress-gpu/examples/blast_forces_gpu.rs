//! `stress-ref`'s `blast_forces` on the GPU solver: a scene pack's gravity solve, fracture
//! off, the same JSON line (each bond's chunks, area, normal, force and moment), for the
//! accuracy comparison with Blast at a matched iteration budget
//! (`scripts/blast_force_agreement.py`). `--budget N,CG`: N Newton steps of at most CG
//! conjugate-gradient iterations from rest (Blast's cold first frame runs
//! `maxSolverIterationsPerFrame` = 25), the iterate kept whether or not it converged;
//! absent: to convergence.
//!
//!   cargo run --release -p stress-gpu --example blast_forces_gpu -- packs/house-1story.json [--blast-inputs] [--blast-model] [--budget 1,25]

use stress_gpu::gpu::Gpu;
use stress_gpu::solver::GpuSolver;
use stress_ref::solver::{ChunkLoads, ReferenceSolver};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let p = std::path::Path::new(path);
    let name = p.file_stem().unwrap().to_string_lossy().to_string();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let blast_model = has("--blast-model");
    let options = stress_ref::scene_pack::ImportOptions { authored_strengths: blast_model, authored_fatal: blast_model, pack_masses: has("--blast-inputs"), pack_modulus: blast_model, blast_authoring: false };
    let mut scene = stress_ref::scene_pack::import_with(p, &name, "structure", options).expect("pack");
    scene.sim.methods.blast_model = blast_model;
    scene.sim.fracture = false;
    scene.sim.gravity_prestress = false;
    let budget = args.iter().position(|a| a == "--budget").and_then(|i| args.get(i + 1)).and_then(|v| v.split_once(',')).map(|(a, b)| (a.parse::<u32>().expect("newton"), b.parse::<u32>().expect("cg")));
    // An unconverged iterate is kept (as Blast's frame publishes its own).
    std::env::set_var("STRESS_QS_WARM", "1");
    let gpu = Gpu::new().expect("GPU");
    let mut mirror = ReferenceSolver::new(&scene);
    mirror.config.static_budget = budget;
    let mut solver = GpuSolver::new(&gpu, mirror, None, Vec::new()).expect("GPU solver");
    let clusters: Vec<usize> = (0..solver.mirror.clusters.len()).filter(|&ci| solver.mirror.clusters[ci].anchored && !solver.mirror.clusters[ci].bonds.is_empty()).collect();
    let loads = ChunkLoads::new(&solver.mirror);
    let t = std::time::Instant::now();
    // The GPU's own iterate (`equilibrate` would finish an unconverged solve on the host).
    let converged_flags = solver.equilibrate_resident(&gpu, &clusters, &loads);
    let solve_s = t.elapsed().as_secs_f64();
    let results = solver.statics_results(&gpu);
    solver.sync(&gpu);
    let m = &solver.mirror;
    let bonds: Vec<_> = m.bonds[0]
        .iter()
        .map(|b| {
            let g = &b.geometry;
            let f = g.to_body(b.force.lin);
            let mo = g.to_body(b.force.ang);
            serde_json::json!([g.a, g.b, g.area, g.normal.to_array(), f.to_array(), mo.to_array()])
        })
        .collect();
    let converged = converged_flags.iter().filter(|&&c| c).count();
    let residual = clusters.iter().map(|&ci| results[ci].0 as f64).fold(0.0, f64::max);
    let cg: u32 = clusters.iter().map(|&ci| results[ci].2).max().unwrap_or(0);
    println!("{}", serde_json::json!({"pack": name, "solve_s": solve_s, "clusters": clusters.len(), "converged": converged, "residual": residual, "cg": cg, "budget": budget, "bonds": bonds}));
}

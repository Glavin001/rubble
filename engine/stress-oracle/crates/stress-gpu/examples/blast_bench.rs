//! Blast's GPU stress solve alone (`stress_gpu::blast`) on a pack under gravity: each frame's
//! solve submitted and waited for, timed, with the islands, iterations and how many converged.
//!
//!   cargo run --release -p stress-gpu --example blast_bench -- packs/villa-savoye.json [frames]
//!
//! `STRESS_BLAST_ITERATIONS` (32), `STRESS_BLAST_SKIP=0` (no settled-island skip),
//! `STRESS_BLAST_MASS=native` (the resident path's weighting).

use stress_gpu::blast::GpuBlastSolver;
use stress_gpu::gpu::Gpu;
use stress_ref::blast_bond::{AngLin, BlastAuthoring, BlastSettings, MassWeighting};
use stress_ref::math::Vec3;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("a scene pack");
    let frames: usize = args.next().map_or(60, |f| f.parse().expect("frames"));
    let options = stress_ref::capability::Profile::BlastGpu.import_options();
    let scene = stress_ref::scene_pack::import_with(
        std::path::Path::new(&path),
        "pack",
        "structure",
        options,
    )
    .expect("pack");
    let mut settings = BlastSettings::default();
    if let Some(i) = std::env::var("STRESS_BLAST_ITERATIONS").ok() {
        settings.max_iterations = i.parse().expect("STRESS_BLAST_ITERATIONS");
    }
    if std::env::var("STRESS_BLAST_MASS").is_ok_and(|v| v == "native") {
        settings.mass_weighting = MassWeighting::Native;
    }
    let skip = !std::env::var("STRESS_BLAST_SKIP").is_ok_and(|v| v == "0");
    let (graph, materials) = BlastAuthoring::graph(&scene, settings).expect("graph");
    let gpu = Gpu::new().expect("GPU");
    println!(
        "{path}: {} nodes, {} bonds, {} iterations a frame, skip {skip}",
        graph.node_count(),
        graph.bond_count(),
        settings.max_iterations
    );
    let accel = vec![
        AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::from_array(scene.gravity)
        };
        graph.node_count()
    ];
    let mut solver = GpuBlastSolver::new(&gpu, graph, materials);
    let mut ms = Vec::with_capacity(frames);
    for frame in 0..frames {
        let t = std::time::Instant::now();
        let reports = solver.solve(&accel, skip);
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        ms.push(elapsed);
        if frame < 3 || frame + 1 == frames {
            let converged = reports.iter().filter(|r| r.converged).count();
            let skipped = reports.iter().filter(|r| r.skipped).count();
            let largest = reports.iter().map(|r| r.iterations).max().unwrap_or(0);
            println!("frame {frame}: {elapsed:.2} ms, {} islands, {converged} converged, {skipped} skipped, at most {largest} iterations", reports.len());
        }
    }
    ms.sort_by(f64::total_cmp);
    println!("solve ms (submitted and waited, with the report read): median {:.2}, p95 {:.2}, worst {:.2}", ms[ms.len() / 2], ms[ms.len() * 95 / 100], ms[ms.len() - 1]);
}

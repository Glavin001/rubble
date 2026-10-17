//! Which catalogue and showcase scenes (and scene packs) the GPU world runs, and why
//! not the others; with their size.
//!
//!   stress-gpu-support [pack.json ...]

use stress_gpu::world::unsupported;
use stress_ref::scene::Scene;

fn report(scene: &Scene) {
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let bonds: usize = scene.bodies.iter().map(|b| b.bonds.len()).sum();
    let why = unsupported(scene).unwrap_or_else(|| "ok".into());
    // Hull sizes (as the solver builds them): vertices, faces, samples.
    let m = stress_ref::solver::ReferenceSolver::new(scene);
    let hulls: Vec<(usize, usize)> = m.structures.iter().flat_map(|st| st.chunks.iter().filter_map(|c| c.hull.as_ref().map(|h| (h.vertices.len(), h.faces.len())))).collect();
    let hull_note = if hulls.is_empty() {
        String::new()
    } else {
        let max_v = hulls.iter().map(|h| h.0).max().unwrap();
        let max_f = hulls.iter().map(|h| h.1).max().unwrap();
        let max_s = hulls.iter().map(|h| h.0 + h.1).max().unwrap();
        let boxes = hulls.iter().filter(|h| **h == (8, 6)).count();
        format!(" | {} hulls ({boxes} box-like), max {max_v} vertices, {max_f} faces, {max_s} samples", hulls.len())
    };
    println!("{:<34} {:>7} chunks {:>8} bonds  {:>6.2} s  {:?}  {why}{hull_note}", scene.name, chunks, bonds, scene.sim.duration, scene.sim.solve_mode);
}

fn main() {
    let packs: Vec<String> = std::env::args().skip(1).collect();
    if packs.is_empty() {
        for scene in stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()) {
            report(&scene);
        }
    }
    for path in packs {
        let name = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        match stress_ref::scene_pack::import(std::path::Path::new(&path), &name, &name) {
            Ok(scene) => report(&scene),
            Err(e) => println!("{name}: {e}"),
        }
    }
}

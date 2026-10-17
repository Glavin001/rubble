//! The GPU explicit substep against the f64 reference solver on identical model data,
//! the identical substep and the identical number of substeps, through every kernel
//! shape (one dispatch per phase, island kernels in device and threadgroup memory).

use stress_gpu::gpu::Gpu;
use stress_gpu::scenes;
use stress_gpu::stress::GpuStress;
use stress_ref::scene::{Scene, SolveMode};
use stress_ref::solver::{ChunkLoads, ReferenceSolver};

#[derive(Clone, Copy, Debug)]
enum Shape {
    PerSubstep,
    Islands,
}

fn compare(name: &str, mut scene: Scene, substeps: usize) {
    scene.sim.fracture = false;
    scene.sim.solve_mode = SolveMode::Explicit;
    let mut reference = ReferenceSolver::new(&scene);
    let gpu = Gpu::new().expect("GPU");
    let mut runs = Vec::new();
    for shape in [Shape::PerSubstep, Shape::Islands] {
        let mut solver = GpuStress::new(&gpu, &reference).expect("GPU solver");
        match shape {
            Shape::PerSubstep => solver.run(&gpu, substeps),
            Shape::Islands => solver.run_islands(&gpu, substeps),
        }
        runs.push((shape, solver.read_state(&gpu), solver));
    }
    let dt = runs[0].2.dt as f64;
    let loads = ChunkLoads::new(&reference);
    for _ in 0..substeps {
        reference.substep(dt, &loads);
    }
    for (shape, state, solver) in &runs {
        let (mut worst_u, mut scale_u, mut worst_th, mut scale_th) = (0f64, 0f64, 0f64, 0f64);
        for (k, &(s, c)) in solver.chunks.iter().enumerate() {
            let st = &reference.chunks[s][c];
            for d in 0..3 {
                let (gu, gt) = (state[4 * k][d] as f64, state[4 * k + 1][d] as f64);
                let (ru, rt) = ([st.u.x, st.u.y, st.u.z][d], [st.th.x, st.th.y, st.th.z][d]);
                worst_u = worst_u.max((gu - ru).abs());
                worst_th = worst_th.max((gt - rt).abs());
                scale_u = scale_u.max(ru.abs());
                scale_th = scale_th.max(rt.abs());
            }
        }
        eprintln!(
            "{name} [{shape:?}]: {} islands, {} chunks, {} bonds, dt {:.3e} s x {substeps}; |u| {scale_u:.3e} m, err {:.2e} rel; |theta| {scale_th:.3e}, err {:.2e} rel",
            solver.islands.len(),
            solver.chunks.len(),
            solver.bond_count,
            dt,
            worst_u / scale_u,
            worst_th / scale_th
        );
        assert!(scale_u > 0.0, "{name}: the reference moved");
        // Provisional: to be replaced by a bound derived from the f32 error propagation.
        assert!(worst_u <= 1e-3 * scale_u, "{name} {shape:?}: displacement error {worst_u:e} of {scale_u:e}");
        assert!(worst_th <= 1e-3 * scale_th, "{name} {shape:?}: rotation error {worst_th:e} of {scale_th:e}");
    }
    // The kernel shapes share their arithmetic: report whether they agree bit for bit.
    let identical = runs[0].1 == runs[1].1;
    eprintln!("{name}: per-substep and island kernels bit-identical: {identical}");
}

#[test]
fn overhang_matches_the_reference() {
    compare("overhang", stress_ref::showcases::overhang(0.5, "overhang"), 4000);
}

#[test]
fn slab_matches_the_reference() {
    compare("slab", scenes::slab([12, 6, 2]), 4000);
}

#[test]
fn big_slab_matches_the_reference() {
    compare("big slab", scenes::slab([40, 20, 4]), 1000);
}

#[test]
fn town_matches_the_reference() {
    compare("town", scenes::town(9, [12, 6, 2]), 2000);
}

//! The drift removal's frame fold on the GPU (world.slang `drift_rigid`): a change of the
//! floating frame that leaves every chunk where it is, whatever the body origin. A free
//! one-chunk cluster and a free 3-chunk cluster with a finite hidden rotation (as a split
//! leaves a child: 0.09 rad in house-1story rammed and fractured-bridge) take one tiny
//! substep with their body origin beside them and 25 m away; against the reference's
//! substep (stress-ref `remove_rigid_drift`) in f64.

use stress_gpu::gpu::Gpu;
use stress_gpu::solver::GpuSolver;
use stress_ref::math::{Quat, Vec3};
use stress_ref::scene::{Scene, SolveMode};
use stress_ref::solver::{ChunkLoads, ReferenceSolver};

/// stress-ref solver.rs `fragments`: the two clusters at the same world place, their
/// body origin beside them or 25 m away.
fn fragments(origin_far: bool) -> (Scene, ReferenceSolver) {
    use stress_ref::builders::*;
    let mut s = new_scene("fold", "rigid drift fold");
    s.gravity = [0.0; 3];
    s.sim.solve_mode = SolveMode::Explicit;
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let world = Vec3::new(25.0, -3.0, 1.0);
    let (offset, position) = if origin_far { (world, [0.0; 3]) } else { (Vec3::ZERO, world.to_array()) };
    let mut one = body("one", grid(offset + Vec3::new(-0.5, -0.5, -0.5), [1, 1, 1], Vec3::splat(1.0), "concrete"), Vec::new());
    one.position = position;
    let chunks = grid(offset + Vec3::new(-1.5, 2.0, -0.5), [3, 1, 1], Vec3::splat(1.0), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    let mut three = body("three", chunks, bonds);
    three.position = position;
    s.bodies = vec![one, three];
    let mut sv = ReferenceSolver::new(&s);
    for (k, st) in sv.chunks.iter_mut().flatten().enumerate() {
        let k = k as f64;
        st.th = Vec3::new(0.02 - 0.01 * k, 0.09, -0.01 + 0.005 * k);
        st.u = Vec3::new(1e-3 * k, -2e-3, 5e-4 * k);
    }
    (s, sv)
}

fn chunk_poses(sv: &ReferenceSolver) -> Vec<(Vec3, Quat)> {
    let mut out = Vec::new();
    for s in 0..sv.structures.len() {
        for c in 0..sv.structures[s].chunks.len() {
            let st = &sv.chunks[s][c];
            let q = sv.clusters[st.cluster].pose.rotation * Quat::from_axis_angle(st.th, st.th.norm());
            out.push((sv.chunk_position(s, c), q));
        }
    }
    out
}

/// One substep of 1e-7 s (the bonds barely move the chunks: (F/m) dt^2 ~ 1e-9 m) on the
/// GPU and the reference. The one-chunk cluster stays where it is; every chunk moves
/// alike with the origin beside it or 25 m away; the GPU's chunks are the reference's.
/// Tolerances: f32 rounding of the coordinates involved (gamma_n, n = 64, of 55 m: the
/// position, plus the 25 m origin offset and the chunk's place in the frame).
#[test]
fn frame_fold_is_independent_of_the_body_origin() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let u32_round = f32::EPSILON as f64 / 2.0;
    let tol = 64.0 * u32_round / (1.0 - 64.0 * u32_round) * 55.0;
    let dt = 1e-7;
    let mut moved: Vec<Vec<Vec3>> = Vec::new();
    let mut failures = Vec::new();
    for far in [false, true] {
        let (_, mut reference) = fragments(far);
        let before = chunk_poses(&reference);
        let mut solver = GpuSolver::new(&gpu, reference.clone(), None, Vec::new()).expect("GPU solver");
        solver.step(&gpu, dt, 1).expect("GPU step");
        solver.sync(&gpu);
        let gpu_now = chunk_poses(&solver.mirror);
        reference.substep(dt, &ChunkLoads::new(&reference));
        let ref_now = chunk_poses(&reference);
        let one = (gpu_now[0].0 - before[0].0).norm();
        eprintln!("origin far {far}: the GPU's step moved the one-chunk cluster by {one:.3e} m, the reference's by {:.3e} m", (ref_now[0].0 - before[0].0).norm());
        if one > tol {
            failures.push(format!("origin far {far}: the GPU's step moved a one-chunk cluster at rest by {one:.3e} m (f32 rounding {tol:.1e} m)"));
        }
        for k in 0..before.len() {
            let d = (gpu_now[k].0 - ref_now[k].0).norm();
            if d > tol {
                failures.push(format!("origin far {far}: chunk {k} is {d:.3e} m from the reference's"));
            }
        }
        moved.push((0..before.len()).map(|k| gpu_now[k].0 - before[k].0).collect());
    }
    for k in 0..moved[0].len() {
        let d = (moved[0][k] - moved[1][k]).norm();
        if d > tol {
            failures.push(format!("chunk {k}: moved {:?} with the origin beside it, {:?} with it 25 m away", moved[0][k].to_array(), moved[1][k].to_array()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

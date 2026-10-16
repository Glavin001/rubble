//! The full GPU solver (joint law, fracture, splits, rigid motion, drift removal)
//! against `ReferenceSolver` in f64: same scene, same substep, same substep count.

use stress_gpu::gpu::Gpu;
use stress_gpu::scenes;
use stress_gpu::solver::GpuSolver;
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, SolveMode};
use stress_ref::solver::{ChunkLoads, ReferenceSolver, SolverEvent};

fn event_key(e: &SolverEvent) -> String {
    match e {
        SolverEvent::Cracked { structure, bond, mode, .. } => format!("cracked s{structure} b{bond} {mode:?}"),
        SolverEvent::Creaked { structure, bond, .. } => format!("creaked s{structure} b{bond}"),
        SolverEvent::Broken { structure, bond, mode, .. } => format!("broken s{structure} b{bond} {mode:?}"),
        SolverEvent::Split { children, .. } => format!("split into {}", children.len()),
        SolverEvent::Refined { structure, chunk, .. } => format!("refined s{structure} c{chunk}"),
    }
}

fn event_time(e: &SolverEvent) -> f64 {
    match e {
        SolverEvent::Cracked { time, .. } | SolverEvent::Creaked { time, .. } | SolverEvent::Broken { time, .. } | SolverEvent::Split { time, .. } | SolverEvent::Refined { time, .. } => *time,
    }
}

/// Run both for `frames` x `per_frame` substeps; compare events, clusters and motion.
fn compare(name: &str, mut scene: Scene, frames: usize, per_frame: usize) {
    scene.sim.solve_mode = SolveMode::Explicit;
    let mut reference = ReferenceSolver::new(&scene);
    let gpu = Gpu::new().expect("GPU");
    let mut solver = GpuSolver::new(&gpu, reference.clone(), None, Vec::new()).expect("GPU solver");
    let dt = reference.stable_dt();
    let loads = ChunkLoads::new(&reference);
    let initial: Vec<Vec<Vec3>> = (0..reference.structures.len())
        .map(|s| (0..reference.structures[s].chunks.len()).map(|c| reference.chunk_position(s, c)).collect())
        .collect();
    // Margin of each reference crack: how far its failure index stood from 1 on either
    // side of the crossing (the substep before, and at the crack).
    let mut margins: std::collections::HashMap<(usize, usize), (f64, f64)> = Default::default();
    for _ in 0..frames {
        for _ in 0..per_frame {
            let before: Vec<Vec<f64>> = reference.bonds.iter().map(|bs| bs.iter().map(|b| b.joint.utilization).collect()).collect();
            let n = reference.events.len();
            reference.substep(dt, &loads);
            for e in &reference.events[n..] {
                if let SolverEvent::Cracked { structure, bond, .. } = e {
                    let after = reference.bonds[*structure][*bond].joint.kappa.max(reference.bonds[*structure][*bond].joint.kappa_c);
                    margins.insert((*structure, *bond), (1.0 - before[*structure][*bond], after - 1.0));
                }
            }
        }
        solver.step(&gpu, dt, per_frame).expect("GPU step");
    }
    solver.sync(&gpu);
    let gpu_m = &solver.mirror;
    // Events: same kinds, bonds and order; times equal.
    let ref_events: Vec<String> = reference.events.iter().map(event_key).collect();
    let gpu_events: Vec<String> = gpu_m.events.iter().map(event_key).collect();
    let first_diff = ref_events.iter().zip(&gpu_events).position(|(a, b)| a != b);
    let time_diff = reference.events.iter().zip(&gpu_m.events).map(|(a, b)| (event_time(a) - event_time(b)).abs()).fold(0.0, f64::max);
    // Motion: chunk world displacement from the start.
    let (mut worst, mut scale) = (0f64, 0f64);
    for s in 0..reference.structures.len() {
        for c in 0..reference.structures[s].chunks.len() {
            if !reference.chunks[s][c].active || !gpu_m.chunks[s][c].active {
                continue;
            }
            let (r, g) = (reference.chunk_position(s, c), gpu_m.chunk_position(s, c));
            worst = worst.max((r - g).norm());
            scale = scale.max((r - initial[s][c]).norm());
        }
    }
    let (er, eg) = (reference.energy, gpu_m.energy);
    eprintln!(
        "{name}: {} substeps of {dt:.3e} s; clusters ref {} gpu {}; events ref {} gpu {} (first difference {first_diff:?}, worst time difference {time_diff:.2e} s); \
         position error {worst:.3e} m of motion {scale:.3e} m; dissipated ref {:.6e} gpu {:.6e}; damped ref {:.6e} gpu {:.6e}; dispatches {}, host splits {}",
        frames * per_frame,
        reference.clusters.len(),
        gpu_m.clusters.len(),
        ref_events.len(),
        gpu_events.len(),
        er.bond_dissipation,
        eg.bond_dissipation,
        er.damping_dissipation,
        eg.damping_dissipation,
        solver.dispatches,
        solver.host_splits,
    );
    for (i, (a, b)) in reference.events.iter().zip(&gpu_m.events).enumerate().take(12) {
        let margin = match a {
            SolverEvent::Cracked { structure, bond, .. } => margins.get(&(*structure, *bond)).map(|m| format!(" (index margin before {:.2e}, after {:.2e})", m.0, m.1)).unwrap_or_default(),
            _ => String::new(),
        };
        eprintln!("  event {i}: ref {} t={:.9} | gpu {} t={:.9}{margin}", event_key(a), event_time(a), event_key(b), event_time(b));
    }
    for (i, (a, b)) in ref_events.iter().zip(&gpu_events).enumerate().take(12) {
        if a != b {
            eprintln!("  event {i}: ref {a} | gpu {b}");
        }
    }
    assert_eq!(ref_events, gpu_events, "{name}: event sequences differ");
    assert_eq!(reference.clusters.len(), gpu_m.clusters.len(), "{name}: cluster counts differ");
    // Event times: identical, except a threshold crossing whose reference margin lies
    // within the f32 state error band (10x the measured relative position error) may
    // move by one substep.
    let band = 10.0 * (worst / scale.max(1e-12)).max(1e-6);
    for (a, b) in reference.events.iter().zip(&gpu_m.events) {
        let shift = event_time(b) - event_time(a);
        if shift.abs() < 0.5 * dt {
            continue;
        }
        let margin = match a {
            SolverEvent::Cracked { structure, bond, .. } => margins.get(&(*structure, *bond)).map(|m| if shift > 0.0 { m.1 } else { m.0 }),
            _ => None,
        };
        assert!(shift.abs() < 1.5 * dt, "{name}: {} moved by {shift:e} s", event_key(a));
        assert!(margin.is_some_and(|m| m <= band), "{name}: {} moved by one substep with margin {margin:?} outside the f32 band {band:e}", event_key(a));
    }
    assert!(worst <= 1e-3 * scale.max(1e-6), "{name}: positions differ by {worst} of {scale}");
}

#[test]
fn intact_slab_matches() {
    let mut scene = scenes::slab([12, 6, 2]);
    scene.sim.fracture = false;
    compare("intact slab", scene, 2, 1500);
}

#[test]
fn overhang_snaps_like_the_reference() {
    // Over-utilized neck: it cracks, snaps, and the slab falls freely.
    let scene = stress_ref::showcases::overhang(1.6, "overhang_snap");
    compare("overhang snap", scene, 10, 1500);
}

#[test]
fn spinning_free_slab_matches() {
    // A free slab thrown and spinning under gravity: rigid motion, frame loads, drift.
    let mut scene = scenes::slab([6, 4, 2]);
    for c in &mut scene.bodies[0].chunks {
        c.support = stress_ref::scene::Support::None;
    }
    scene.bodies[0].linear_velocity = [1.0, 0.5, 3.0];
    scene.bodies[0].angular_velocity = [0.5, 2.0, 1.0];
    scene.sim.fracture = false;
    compare("spinning slab", scene, 4, 1500);
}

#[test]
fn overloaded_slab_breaks_like_the_reference() {
    // Gravity far beyond the slab's strength: progressive cracking, many splits, falling pieces.
    let mut scene = scenes::slab([12, 6, 2]);
    scene.gravity = [0.0, 0.0, -9.81 * 300.0];
    compare("overloaded slab", scene, 6, 1500);
}

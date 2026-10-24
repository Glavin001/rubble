//! The resident quasi-static frame (`GpuWorld::with_resident_engine`: rubble stepped
//! without read-backs, its contacts turned into static loads on the GPU) against the fast
//! quasi-static frame (`STRESS_QS_FAST`: the same solves, loads formed on the host from
//! read-back contacts): the same outcome bit for bit, on a rammed house that cracks.
//!
//!   cargo test --release -p stress-rubble --features gpu --test resident_gpu
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, SolveMode};
use stress_rubble::scenes::{house, realtime};
use stress_rubble::RubbleEngine;

/// house-1story rammed at 12 m/s, quasi-static with the base material model (Blast's:
/// the bench's `--config blast`).
fn scene() -> Scene {
    let mut s = realtime(stress_ref::scenes::rammed(house()), 0.25);
    for f in ["softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping"] {
        s.sim.features.set(f, false).unwrap();
    }
    s.sim.solve_mode = SolveMode::QuasiStatic;
    s
}

/// A weak concrete cantilever (8 chunks of 0.1 m, fixed at one end, 0.3 m above the
/// ground) that breaks at its root under its own weight: the resident frame's split path
/// (the mirror synced, the cluster split, the tables rebuilt, the cascade resumed, the
/// new body given to the engine), then the fragment's fall onto the ground.
fn weak_cantilever() -> Scene {
    use stress_ref::builders::*;
    use stress_ref::scene::*;
    let mut s = new_scene("weak_cantilever", "a cantilever that breaks under its own weight");
    let mut m = oracle_concrete(None);
    m.tensile_strength = 2e4;
    m.cohesion = 2e4;
    s.materials.insert("weak".into(), m);
    s.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "weak".into() });
    let mut chunks = grid(Vec3::new(-0.05, -0.05, 0.3), [8, 1, 1], Vec3::splat(0.1), "weak");
    chunks[0].support = Support::Fixed;
    let bonds = auto_bonds(&chunks, |_, _| "weak".into());
    s.bodies.push(body("beam", chunks, bonds));
    let mut s = realtime(s, 0.4);
    for f in ["softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping"] {
        s.sim.features.set(f, false).unwrap();
    }
    s.sim.solve_mode = SolveMode::QuasiStatic;
    s
}

/// (bond damage bits, alive bonds, clusters, impactor speed bits, resident frames, free
/// clusters' heights) after the run.
fn run(gpu: &Gpu, scene: &Scene, resident: bool) -> (Vec<u64>, usize, usize, u64, u64, Vec<u64>) {
    run_with(gpu, scene, resident, None)
}

fn run_with(gpu: &Gpu, scene: &Scene, resident: bool, budget: Option<(u32, u32)>) -> (Vec<u64>, usize, usize, u64, u64, Vec<u64>) {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_GPU_QUIET_EXCLUDE", "0");
    let engine = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(scene)).expect("rubble");
    let mut w = if resident { GpuWorld::with_resident_engine(gpu, scene, engine) } else { GpuWorld::with_engine(gpu, scene, Box::new(engine)) }.expect("coupled world");
    w.fast_qs = !resident;
    // The resident frame keeps the engine's shapes undeformed; the fast frame matches it.
    w.set_engine_shapes_deformed(false);
    w.redo_fractured_frames = false;
    w.set_impact_islands(false);
    w.solver.mirror.config.static_budget = budget;
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.sync(gpu);
    let m = &w.solver.mirror;
    let damage: Vec<u64> = m.bonds.iter().flatten().map(|b| b.joint.damage.to_bits()).collect();
    let alive = m.bonds.iter().flatten().filter(|b| b.alive).count();
    let speed = w.solver.impactors.first().map_or(0, |i| i.velocity.norm().to_bits());
    // Every free cluster's height (bits): the fragments' fall under the engine.
    let heights: Vec<u64> = m.clusters.iter().filter(|c| !c.anchored).map(|c| c.pose.position.z.to_bits()).collect();
    (damage, alive, m.clusters.len(), speed, w.resident_frames, heights)
}

#[test]
fn resident_frame_matches_the_fast_frame() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let s = scene();
    let fast = run(&gpu, &s, false);
    let resident = run(&gpu, &s, true);
    assert_eq!(fast.4, 0, "the fast run took resident frames");
    assert!(resident.4 > 0, "no resident frame ran");
    let damaged = fast.0.iter().filter(|&&d| f64::from_bits(d) > 0.0).count();
    eprintln!("bonds damaged {damaged}, alive {}, clusters {}, resident frames {}", fast.1, fast.2, resident.4);
    assert!(damaged > 0, "nothing cracked: the test needs damage");
    assert_eq!((fast.0, fast.1, fast.2, fast.3), (resident.0, resident.1, resident.2, resident.3));
}

#[test]
fn resident_frame_splits_as_the_fast_frame() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let s = weak_cantilever();
    let fast = run(&gpu, &s, false);
    let resident = run(&gpu, &s, true);
    eprintln!("alive {} of {}, clusters {}, resident frames {}", fast.1, fast.0.len(), fast.2, resident.4);
    assert!(fast.2 > 1, "nothing split: the test needs a split");
    assert!(resident.4 > 0, "no resident frame ran");
    let fell = fast.5.iter().map(|&z| 0.3 - f64::from_bits(z)).fold(0.0, f64::max);
    eprintln!("fragments fell up to {fell:.3} m");
    assert_eq!((fast.0, fast.1, fast.2, fast.5), (resident.0, resident.1, resident.2, resident.5));
}

/// The repeat gate (cascade.slang `cascade_gate`): with a budget the solves stop short of
/// convergence, and the gate skips each repeat of an unconverged solve. The outcome is
/// the same as solving every one again.
#[test]
fn repeat_gate_is_lossless() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let s = scene();
    std::env::set_var("STRESS_QS_REPEAT_GATE", "0");
    let every = run_with(&gpu, &s, true, Some((1, 4)));
    std::env::set_var("STRESS_QS_REPEAT_GATE", "1");
    let gated = run_with(&gpu, &s, true, Some((1, 4)));
    std::env::remove_var("STRESS_QS_REPEAT_GATE");
    assert!(gated.4 > 0, "no resident frame ran");
    assert_eq!((every.0, every.1, every.2, every.3, every.5), (gated.0, gated.1, gated.2, gated.3, gated.5));
}

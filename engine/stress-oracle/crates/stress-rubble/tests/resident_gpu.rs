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

/// (bond damage bits, alive bonds, clusters, impactor speed bits) after the run.
fn run(gpu: &Gpu, scene: &Scene, resident: bool) -> (Vec<u64>, usize, usize, u64, u64) {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_GPU_QUIET_EXCLUDE", "0");
    let engine = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(scene)).expect("rubble");
    let mut w = if resident { GpuWorld::with_resident_engine(gpu, scene, engine) } else { GpuWorld::with_engine(gpu, scene, Box::new(engine)) }.expect("coupled world");
    w.fast_qs = !resident;
    w.redo_fractured_frames = false;
    w.set_impact_islands(false);
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.sync(gpu);
    let m = &w.solver.mirror;
    let damage: Vec<u64> = m.bonds.iter().flatten().map(|b| b.joint.damage.to_bits()).collect();
    let alive = m.bonds.iter().flatten().filter(|b| b.alive).count();
    let speed = w.solver.impactors.first().map_or(0, |i| i.velocity.norm().to_bits());
    (damage, alive, m.clusters.len(), speed, w.resident_frames)
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

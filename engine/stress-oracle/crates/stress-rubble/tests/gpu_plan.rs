//! The contact plan on the GPU (stress-gpu `gpu_plan`, docs/stress-fracture-gpu-resident.md
//! phase 6a) against the host's `contacts::plan`, beside every island segment of a rammed
//! house: the same candidate pairs in the same order, the same ground and impactor
//! candidates and rotation checks, budgets to f32 rounding. The GPU plan may add a
//! borderline candidate (its box test's margin is wider), never miss one.
//!
//!   cargo test --release -p stress-rubble --features gpu --test gpu_plan
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{house, realtime};
use stress_rubble::RubbleEngine;

#[test]
fn gpu_plan_matches_the_host_plan() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_GPU_PLAN_CHECK", "1");
    let mut scene = realtime(stress_ref::scenes::rammed(house()), 0.15);
    scene.name = "house_ram".into();
    let engine = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble");
    let mut w = GpuWorld::with_engine(&gpu, &scene, Box::new(engine)).expect("coupled world");
    w.redo_fractured_frames = false;
    for _ in 0..(scene.sim.duration / scene.sim.frame_dt).round() as usize {
        w.step_frame(&gpu).expect("frame");
    }
    let c = w.solver.plan_check;
    eprintln!("{c:?}");
    assert!(c.plans > 0 && c.host_pairs > 0, "no plan with pairs compared");
    assert_eq!((c.missed_pairs, c.missed_ground, c.missed_impactor), (0, 0, 0), "the GPU plan missed candidates");
    assert_eq!((c.rotation_mismatch, c.order_mismatch), (0, 0));
    assert!(c.extra_pairs * 1000 <= c.host_pairs, "too many borderline extras");
    assert!(c.budget < 1e-5, "budgets differ by {}", c.budget);
}

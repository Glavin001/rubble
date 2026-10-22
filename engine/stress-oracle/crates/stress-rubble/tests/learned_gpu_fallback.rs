//! The learned impact path's fallback (`STRESS_IMPACT_PUNCH=learned`): a call whose decoded
//! outcome is non-finite (`STRESS_LEARNED_STUB=nonfinite`) is not applied; the GPU punch
//! takes its island, in the same submission, and the host logs and counts it. A file of
//! its own: the switches are read once per process.
//!
//!   cargo test --release -p stress-rubble --features gpu --test learned_gpu_fallback
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

#[test]
fn a_nonfinite_call_falls_back_to_the_punch() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "learned");
    std::env::set_var("STRESS_LEARNED_STUB", "nonfinite");
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParityRouted;
    let mut scene = realtime(later(wall_ram_late(), 2), 0.25);
    p.apply_scene(&mut scene);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    w.set_tiered(true).expect("tiered coupling");
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(&gpu).expect("frame");
    }
    w.sync(&gpu);
    let o = w.observation();
    let t = w.learned.as_ref().expect("learned batch").totals;
    let ram = w.solver.impactors[0].velocity.norm();
    eprintln!("ram {ram:.2} m/s, {} fragments, learned {t:?}", o.values["fragments"]);
    assert!(t.calls > 0 && t.fallbacks == t.calls && t.fallback_nonfinite == t.calls, "every call fell back as non-finite: {t:?}");
    assert_eq!(t.applied, 0, "no call was applied");
    // The punch took the island: its plug broke out (`punch_gpu.rs`'s outcome).
    assert!(o.values["fragments"] >= 2.0, "the punch broke the plug out");
    assert!(ram.is_finite());
}

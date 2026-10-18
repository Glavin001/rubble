//! The GPU's predictive punch (`STRESS_IMPACT_PUNCH=gpu`, `shaders/punch.slang`) under
//! `blast_parity_routed`: every frame after the first resident (one status read; the
//! submissions are counted in `tests/frame_io.rs`), the ram's plug broken out at the end of the frame before its strike. M5's fidelity
//! bar (engine/docs/stress-baseline-goal.md): closer to the reference than the native stage's
//! correction pass, on the free wall's ram speed and fragments (reference 35.1 m/s and 6;
//! correction 10.9 and 23). A file of its own: the switch is read once per process.
//!
//!   cargo test --release -p stress-rubble --features gpu --test punch_gpu
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

#[test]
fn gpu_punch_frees_the_plug_in_resident_frames() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "gpu");
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParityRouted;
    // The ram strikes in the third frame: the second's check sees it coming.
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
    let (ram, fragments) = (w.solver.impactors[0].velocity.norm(), o.values["fragments"]);
    eprintln!("ram {ram:.2} m/s, {fragments} fragments, {} resident frames, {} coupled", w.resident_frames, w.coupled_frames);
    // Every frame after the first resident: no island frame.
    assert_eq!(w.coupled_frames, 1, "only the first frame is coupled");
    // M5's bar against the correction pass (10.9 m/s, 23 fragments).
    assert!((ram - 35.1).abs() < (10.9f64 - 35.1).abs(), "ram {ram:.2} m/s");
    assert!((fragments - 6.0).abs() < (23.0f64 - 6.0).abs(), "{fragments} fragments");
}

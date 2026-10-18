//! The free-wall ram (`scenes::wall_ram_late`: a 200 kg steel ram at 40 m/s into a free
//! 528 kg concrete wall, striking in a resident frame) under `blast_parity_routed`, as
//! `stress_bench --config blast_parity_routed` runs it: the ram must punch a plug out and
//! leave near the reference's 35.1 m/s (6 fragments), not exchange its momentum with the
//! whole wall (about 11 m/s, what a rigid contact gives). M5's fidelity bar
//! (engine/docs/stress-baseline-goal.md): closer to the reference than the native stage's
//! correction pass (10.9 m/s, 23 fragments). Both the full-frame impact island and the
//! impact window (`GpuWorld::impact_window`, its lead, new approaches only) are checked.
//!
//!   cargo test --release -p stress-rubble --features gpu --test punch_through_routed
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

/// (ram exit speed, fragments, broken bonds) after 0.25 s.
fn run(gpu: &Gpu, window: Option<f64>) -> (f64, f64, f64) {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let p = Profile::BlastParityRouted;
    let mut scene = realtime(wall_ram_late(), 0.25);
    p.apply_scene(&mut scene);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    w.set_tiered(true).expect("tiered coupling");
    if let Some(win) = window {
        std::env::set_var("STRESS_IMPACT_WINDOW_NEW_ONLY", "1");
        w.impact_window = win;
        w.impact_window_lead = true;
    }
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.sync(gpu);
    let o = w.observation();
    (w.solver.impactors[0].velocity.norm(), o.values["fragments"], o.values["broken_bonds"])
}

fn check(label: &str, (ram, fragments, broken): (f64, f64, f64)) {
    eprintln!("{label}: ram {ram:.2} m/s, {fragments} fragments, {broken} bonds broken");
    // The plug punched out: the ram keeps most of its speed (reference 35.1 m/s); a rigid
    // whole-wall exchange would leave it near 11 m/s.
    assert!((ram - 35.1).abs() < 1.5, "{label}: ram {ram:.2} m/s, the reference 35.1");
    // A few pieces, as the reference's 6 (the correction pass: 23).
    assert!((2.0..=10.0).contains(&fragments), "{label}: {fragments} fragments, the reference 6");
    assert!(broken > 0.0, "{label}: nothing broke");
}

#[test]
fn routed_free_wall_ram_punches_through() {
    let gpu = Gpu::new().expect("GPU");
    check("full-frame island", run(&gpu, None));
}

#[test]
fn routed_free_wall_ram_punches_through_in_an_impact_window() {
    let gpu = Gpu::new().expect("GPU");
    check("3 ms impact window", run(&gpu, Some(0.003)));
}

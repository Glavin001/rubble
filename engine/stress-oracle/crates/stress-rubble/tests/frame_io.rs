//! Submissions and reads per resident tick (`GpuWorld::last_tick_io`, plan 5d W1,
//! engine/docs/stress-5d-one-command-buffer.md), counted on the free-wall ram under
//! `blast_parity_routed` with the GPU punch, whose ram strikes in the third frame. The
//! default mode submits each part of the frame on its own: rubble's ingest start, its step,
//! its substep ingest and merge, the frame encoder, the cascade and the read's copy, then
//! one read. W2 brings this to one submission. A file of its own: the punch switch is read
//! once per process.
//!
//!   cargo test --release -p stress-rubble --features gpu --test frame_io
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

#[test]
fn resident_ticks_are_counted_and_strict_io_names_the_sites() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "gpu");
    std::env::remove_var("STRESS_FRAME_SUBMITS");
    std::env::remove_var("STRESS_STRICT_IO");
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
    for f in 0..frames {
        w.step_frame(&gpu).expect("frame");
        let t = w.last_tick_io();
        if f == 0 {
            assert!(!w.last_tick_resident(), "frame 0 is coupled");
            continue;
        }
        assert!(w.last_tick_resident(), "frame {f} resident");
        assert_eq!((t.submits(), t.reads(), t.waits()), (7, 1, 0), "frame {f}: {:?}", t.0);
    }
    // Strict mode fails a resident tick with more than one submission, naming the sites.
    std::env::set_var("STRESS_STRICT_IO", "1");
    let err = w.step_frame(&gpu).expect_err("strict IO fails today's seven submissions");
    std::env::remove_var("STRESS_STRICT_IO");
    assert!(err.contains("STRESS_STRICT_IO") && err.contains("7 submissions") && err.contains("solver.rs"), "{err}");
}

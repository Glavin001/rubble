//! Reclaiming the engine slots fracture splits retire (`RubbleEngine::compact_slots`,
//! `WgpuSim::compact`). The threshold profile's ram through the free wall, with every slot
//! reclaimed after every frame that retired one (`STRESS_RUBBLE_COMPACT=always`): the run
//! keeps going resident, only live bodies and colliders keep slots, and the outcome matches
//! the run that never compacts. The colouring order follows body ids, so the two agree to
//! the engine's own run-to-run spread, not bit for bit.
//!
//!   cargo test --release -p stress-rubble --features gpu --test compact_slots
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

struct Outcome {
    broken: f64,
    fragments: f64,
    exit_speed: f64,
    compactions: u64,
    resident_frames: u64,
    frames: usize,
}

fn run(gpu: &Gpu, mode: &str) -> Outcome {
    std::env::set_var("STRESS_RUBBLE_COMPACT", mode);
    let p = Profile::Threshold;
    let mut scene = realtime(wall_ram_late(), 0.25);
    p.apply_scene(&mut scene);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.apply_profile(p);
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.sync(gpu);
    let o = w.observation();
    Outcome {
        broken: o.values["broken_bonds"],
        fragments: o.values["fragments"],
        exit_speed: w.solver.impactors[0].velocity.norm(),
        compactions: w.compactions,
        resident_frames: w.resident_frames,
        frames,
    }
}

#[test]
fn reclaimed_slots_keep_the_run() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let gpu = Gpu::new().expect("GPU");
    let never = run(&gpu, "never");
    let always = run(&gpu, "always");
    for (name, o) in [("never", &never), ("always", &always)] {
        eprintln!("{name}: {} bonds broken, {} fragments, ram exit {:.2} m/s, {} compactions, {} of {} frames resident", o.broken, o.fragments, o.exit_speed, o.compactions, o.resident_frames, o.frames);
    }
    assert_eq!(never.compactions, 0);
    assert!(always.compactions > 0, "the splits retired slots and they were reclaimed");
    assert_eq!(always.resident_frames, never.resident_frames, "compaction keeps every frame resident");
    let near = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0);
    assert!(near(always.broken, never.broken, 0.1), "bonds broken: {} vs {}", always.broken, never.broken);
    assert!(near(always.fragments, never.fragments, 0.15), "fragments: {} vs {}", always.fragments, never.fragments);
    assert!(near(always.exit_speed, never.exit_speed, 0.05), "ram exit speed: {} vs {}", always.exit_speed, never.exit_speed);
}

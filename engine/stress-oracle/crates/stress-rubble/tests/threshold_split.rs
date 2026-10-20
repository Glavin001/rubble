//! The v0 stress mode (`Profile::Threshold`, stress-gpu `threshold.slang`): with no static
//! solve, a ram's contact breaks the wall's bonds, the wall splits into child bodies in the
//! resident frame (the GPU's split path), and the run repeats bit for bit.
//!
//!   cargo test --release -p stress-rubble --features gpu --test threshold_split
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
    clusters: usize,
    resident_frames: usize,
    digest: u64,
}

fn run(gpu: &Gpu) -> Outcome {
    let p = Profile::Threshold;
    let mut scene = realtime(wall_ram_late(), 0.25);
    p.apply_scene(&mut scene);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.apply_profile(p);
    assert!(w.solver.mirror.config.contact_threshold, "the profile turns the threshold rule on");
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.sync(gpu);
    let o = w.observation();
    // FNV-1a of the end state (as stress_bench's digest): cluster membership and motion.
    let mut h: u64 = 0xcbf29ce484222325;
    for cl in &w.solver.mirror.clusters {
        for b in format!("{} {:?} {:?} {:?} {:?}", cl.id, cl.chunks, cl.pose, cl.velocity, cl.angular_velocity).bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    Outcome { broken: o.values["broken_bonds"], fragments: o.values["fragments"], clusters: w.solver.mirror.clusters.len(), resident_frames: w.resident_frames as usize, digest: h }
}

#[test]
fn contact_threshold_breaks_and_splits_in_resident_frames() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let gpu = Gpu::new().expect("GPU");
    let a = run(&gpu);
    eprintln!("threshold: {} bonds broken, {} fragments, {} clusters, {} resident frames", a.broken, a.fragments, a.clusters, a.resident_frames);
    assert!(a.broken > 0.0, "the ram breaks bonds");
    assert!(a.fragments >= 2.0 && a.clusters >= 2, "the wall splits into child bodies");
    assert!(a.resident_frames > 0, "the frames run resident (the GPU's split path)");
    let b = run(&gpu);
    assert_eq!((a.broken, a.fragments, a.clusters, a.digest), (b.broken, b.fragments, b.clusters, b.digest), "the run repeats bit for bit");
}

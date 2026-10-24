//! The Wrecking Yard structures (`stress_ref::wrecking::STRUCTURES`, the Ram Raid and
//! Fly-through buildings of `stress_ref::buildings` among them) stand and settle on the GPU in
//! both of the lab's stress modes before anything hits them, set up as stress-web sets
//! them up: `blast_parity` on the resident GpuWorld (impact islands on), `threshold` on
//! the GPU-resident threshold world. Two seconds at 60 Hz with no shot: no bond breaks,
//! the structure stays one body, and no chunk has moved more than a centimetre.
//!
//!   cargo test --release -p stress-rubble --features gpu --test wrecking_yard
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::wrecking::STRUCTURES;
use stress_rubble::scenes::wrecking_yard;
use stress_rubble::RubbleEngine;

const SECONDS: f64 = 2.0;
/// How far a chunk may move while the structure settles (m).
const SETTLE: f64 = 0.01;

fn blast_parity(gpu: &Gpu, key: &str) {
    let p = Profile::BlastParity;
    let scene = wrecking_yard(key, p, SECONDS);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    w.sync(gpu);
    let fragments = w.observation().values["fragments"];
    let frames = (SECONDS / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.sync(gpu);
    let o = w.observation();
    let moved = o.bodies.values().flatten().map(|c| Vec3::from_array(c.displacement).norm()).fold(0.0, f64::max);
    println!("{key} blast_parity: {frames} frames, broken {}, fragments {} (from {fragments}), largest move {moved:.2e} m", o.values["broken_bonds"], o.values["fragments"]);
    assert_eq!(o.values["broken_bonds"], 0.0, "{key}: bonds broke while settling");
    assert_eq!(o.values["fragments"], fragments, "{key}: the structure came apart");
    assert!(moved < SETTLE, "{key}: a chunk moved {moved} m");
}

fn threshold(gpu: &Gpu, key: &str) {
    let p = Profile::Threshold;
    let scene = wrecking_yard(key, p, SECONDS);
    let ball = stress_ref::wrecking::BALLS[0];
    let mut w = stress_rubble::threshold::ThresholdWorld::new(gpu, &scene, ball.radius, ball.mass).expect("threshold world");
    let start = w.chunk_centres(gpu);
    let frames = (SECONDS / w.frame_dt).round() as usize;
    // Each frame waited for and its counters read (the ring copies them into a slot
    // the host has read).
    let mut st = w.poll(gpu);
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        st = w.poll(gpu);
    }
    let end = w.chunk_centres(gpu);
    let moved = start.iter().zip(&end).map(|((a, _), (b, _))| (Vec3::from_array(*a) - Vec3::from_array(*b)).norm()).fold(0.0, f64::max);
    println!("{key} threshold: {frames} frames (stats of frame {}), broken {}, bodies {}, errors {:#x}/{:#x}, largest move {moved:.2e} m", st.frame, st.broken, st.bodies, st.error, st.engine_overflow);
    assert_eq!(st.frame, frames as u64, "{key}: the counters are the last frame's");
    assert_eq!((st.error, st.engine_overflow), (0, 0), "{key}");
    assert_eq!(st.broken, 0, "{key}: bonds broke while settling");
    assert_eq!(st.split_frames, 0, "{key}: the structure came apart");
    assert!(moved < SETTLE, "{key}: a chunk moved {moved} m");
}

#[test]
fn structures_settle_in_both_stress_modes() {
    let gpu = Gpu::new().expect("GPU");
    for &(key, _) in STRUCTURES {
        blast_parity(&gpu, key);
        threshold(&gpu, key);
    }
}

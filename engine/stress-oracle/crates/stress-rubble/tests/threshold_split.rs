//! The v0 stress mode (`Profile::Threshold`), forward only: the ram's strike on the free
//! wall is predicted at the end of the frame before it (the routing check and punch.slang
//! `threshold_predict`), so the wall's bonds break and the wall splits into child bodies
//! before the engine's step meets it: the bonds are broken by the end of the frame before
//! the one in which the ram first slows. Every frame resident (the GPU's split path), no
//! correction pass (`STRESS_CORRECTION_LIMIT` off: no re-simulation), and the run repeats
//! bit for bit.
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
    frames: usize,
    corrected: u64,
    coupled: u64,
    /// The first frame after which bonds are broken, and the first in which the ram slows.
    first_break: Option<usize>,
    first_slow: Option<usize>,
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
    assert!(w.solver.mirror.config.contact_threshold && w.predict_strikes, "the profile decides breaks by contact force and predicts strikes");
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    let speed0 = scene.impactors[0].velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
    let (mut first_break, mut first_slow) = (None, None);
    for f in 0..frames {
        w.step_frame(gpu).expect("frame");
        w.sync(gpu);
        if first_break.is_none() && w.observation().values["broken_bonds"] > 0.0 {
            first_break = Some(f);
        }
        if first_slow.is_none() && w.solver.impactors[0].velocity.norm() < 0.99 * speed0 {
            first_slow = Some(f);
        }
    }
    let o = w.observation();
    // FNV-1a of the end state (as stress_bench's digest): cluster membership and motion.
    let mut h: u64 = 0xcbf29ce484222325;
    for cl in &w.solver.mirror.clusters {
        for b in format!("{} {:?} {:?} {:?} {:?}", cl.id, cl.chunks, cl.pose, cl.velocity, cl.angular_velocity).bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    Outcome {
        broken: o.values["broken_bonds"],
        fragments: o.values["fragments"],
        clusters: w.solver.mirror.clusters.len(),
        resident_frames: w.resident_frames as usize,
        frames,
        corrected: w.corrected_frames,
        coupled: w.coupled_frames,
        first_break,
        first_slow,
        digest: h,
    }
}

#[test]
fn predicted_strike_breaks_and_splits_before_the_engine_meets_it() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let gpu = Gpu::new().expect("GPU");
    let a = run(&gpu);
    eprintln!(
        "threshold: {} bonds broken, {} fragments, {} clusters, {} of {} frames resident, {} coupled, {} corrected, first break after frame {:?}, ram first slows in frame {:?}",
        a.broken, a.fragments, a.clusters, a.resident_frames, a.frames, a.coupled, a.corrected, a.first_break, a.first_slow
    );
    assert!(a.broken > 0.0, "the strike breaks bonds");
    assert!(a.fragments >= 2.0 && a.clusters >= 2, "the wall splits into child bodies");
    assert_eq!(a.corrected, 0, "forward only: no correction pass");
    // The first frame (the world's set-up) is not resident in any profile; every one after
    // it is: no coupled or host-orchestrated island frame.
    assert_eq!(a.resident_frames, a.frames - 1, "every frame after the first runs resident");
    assert_eq!(a.coupled, 1, "no coupled frame but the set-up's");
    let (Some(brk), Some(slow)) = (a.first_break, a.first_slow) else { panic!("the ram strikes and bonds break: {:?} {:?}", a.first_break, a.first_slow) };
    assert!(brk < slow, "the strike's breaks are made before the engine's step meets the wall (broken after frame {brk}, the ram slows in frame {slow})");
    let b = run(&gpu);
    assert_eq!((a.broken, a.fragments, a.clusters, a.digest), (b.broken, b.fragments, b.clusters, b.digest), "the run repeats bit for bit");
}

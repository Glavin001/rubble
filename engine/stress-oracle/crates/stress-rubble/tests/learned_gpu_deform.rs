//! The learned path with P0's kept-deformation boundary (`STRESS_DECODED_DEFORMATION=1`):
//! at a call's window end every chunk of the call sits at its decoded frame-end position,
//! its departure from its body's fitted rigid motion kept as hidden state (stress-ref
//! `apply_decoded_with(keep_deformation)`). Checked on the free wall with the punch-wobble stub (pieces that do not move rigidly):
//! after the window end, each chunk's position in the synced world equals its frame-start
//! position plus its decoded displacement. A file of its own: the switches are read once.
//!
//!   cargo test --release -p stress-rubble --features gpu --test learned_gpu_deform
#![cfg(feature = "gpu")]

use std::collections::HashMap;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

#[test]
fn window_end_keeps_each_chunks_decoded_position() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "learned");
    std::env::set_var("STRESS_LEARNED_STUB", "punch-wobble");
    std::env::set_var("STRESS_DECODED_DEFORMATION", "1");
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParityRouted;
    // the impact point moved across the face (not the symmetric knife edge)
    let mut scene = realtime(later(wall_ram_late(), 2), 0.25);
    scene.impactors[0].position[0] += 7.3e-4;
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
    let mut checked = 0;
    let mut pending: Option<HashMap<i64, [f32; 3]>> = None;
    for _ in 0..frames {
        w.step_frame(&gpu).expect("frame");
        if let Some(want) = pending.take() {
            // the window end of the batch exported last frame
            w.sync(&gpu);
            // the chunks' hidden state too (the mirror skips settled islands' by default)
            let all: Vec<usize> = (0..w.solver.mirror.clusters.len()).collect();
            w.solver.sync_chunk_state(&gpu, &all);
            let worst = w.with_shell(|s| {
                let mut worst = 0.0f64;
                for (&key, x1) in &want {
                    let (st, c) = ((key >> 32) as usize, (key & 0xffff_ffff) as usize);
                    let p = s.solver.chunk_position(st, c);
                    let d = (p - Vec3::new(x1[0] as f64, x1[1] as f64, x1[2] as f64)).norm();
                    worst = worst.max(d);
                }
                worst
            });
            // and the pieces really deform (the hidden displacement is not zero)
            let u_max = w.with_shell(|s| s.solver.chunks.iter().flatten().map(|c| c.u.norm()).fold(0.0f64, f64::max));
            eprintln!("window end: {} chunks, worst position error {worst:.3e} m, largest hidden displacement {u_max:.3e} m", want.len());
            assert!(u_max > 1e-5, "no deformation was kept");
            assert!(worst < 2e-4, "a chunk is {worst:.3e} m from its decoded position");
            checked += 1;
        }
        let Some(l) = w.learned.as_ref() else { continue };
        if l.pending {
            let export = l.export(&gpu);
            let mut want = HashMap::new();
            for c in &export {
                for (j, &key) in c.chunk_keys.iter().enumerate() {
                    let (p, d) = (c.chunk_pos[j], c.chunk_dx[j]);
                    want.insert(key, [p[0] + d[0], p[1] + d[1], p[2] + d[2]]);
                }
            }
            pending = Some(want);
        }
    }
    assert!(checked > 0, "no window end checked");
}

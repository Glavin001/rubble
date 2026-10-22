//! The learned impact model's engine path (`STRESS_IMPACT_PUNCH=learned`, `learned.rs`)
//! under `blast_parity_routed`, end to end with a stand-in network (`STRESS_LEARNED_STUB`):
//! each routed frame's call batch made, decoded and held, applied at its window end, its
//! children placed, in the resident frames' one submission and one read. A file of its own:
//! the switches are read once per process.
//!
//!   cargo test --release -p stress-rubble --features gpu --test learned_gpu -- --test-threads=1
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime, wall_ram_late, wall_row_late};
use stress_rubble::RubbleEngine;

struct Run {
    ram: f64,
    fragments: f64,
    coupled: u64,
    totals: stress_gpu::learned::Totals,
    digest: u64,
    /// Read-backs (`Gpu::read_ranges` / `read_range`) per frame after the first.
    reads: Vec<u64>,
}

fn run(scene: stress_ref::scene::Scene) -> Run {
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParityRouted;
    let mut scene = scene;
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
    let mut reads = Vec::new();
    for f in 0..frames {
        let before = stress_gpu::gpu::reads();
        w.step_frame(&gpu).expect("frame");
        if f > 0 {
            reads.push(stress_gpu::gpu::reads() - before);
        }
    }
    w.sync(&gpu);
    let o = w.observation();
    let ram = w.solver.impactors[0].velocity.norm();
    let fragments = o.values["fragments"];
    let totals = w.learned.as_ref().map(|l| l.totals).unwrap_or_default();
    // The run's end state: every chunk's cluster pose and the impactor's.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    for cl in &w.solver.mirror.clusters {
        for v in [cl.pose.position.x, cl.pose.position.y, cl.pose.position.z, cl.velocity.x, cl.velocity.y, cl.velocity.z] {
            v.to_bits().hash(&mut h);
        }
    }
    for imp in &w.solver.impactors {
        for v in [imp.pose.position.x, imp.pose.position.y, imp.pose.position.z, imp.velocity.x, imp.velocity.y, imp.velocity.z] {
            v.to_bits().hash(&mut h);
        }
    }
    eprintln!("ram {ram:.2} m/s, {fragments} fragments, {} resident frames, {} coupled, learned {totals:?}", w.resident_frames, w.coupled_frames);
    Run { ram, fragments, coupled: w.coupled_frames, totals, digest: h.finish(), reads }
}

#[test]
fn learned_stub_runs_the_free_wall_and_the_wall_row() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "learned");
    std::env::set_var("STRESS_LEARNED_STUB", "punch");
    std::env::set_var("STRESS_DEBRIS_ROUTING", "1");
    let free = run(realtime(later(wall_ram_late(), 2), 0.25));
    assert_eq!(free.coupled, 1, "only the first frame is coupled");
    assert!(free.totals.calls > 0, "calls were made");
    assert_eq!(free.totals.fallbacks, 0, "no call fell back");
    assert!(free.totals.applied > 0 && free.totals.broken > 0, "a call broke the plug out at its window end");
    assert!(free.ram.is_finite() && free.fragments >= 2.0);
    // Every resident frame, the call's frames included: one read (the frame's status, the
    // routing flag and the batch's control words together).
    assert!(free.reads.iter().all(|&r| r == 1), "reads per frame {:?}", free.reads);
    // Deterministic: the same run twice, the same end state.
    let again = run(realtime(later(wall_ram_late(), 2), 0.25));
    assert_eq!(free.digest, again.digest, "two runs differ");
    let row = run(realtime(later(wall_row_late(2), 2), 0.25));
    assert!(row.totals.calls > 0 && row.totals.applied > 0);
    assert!(row.ram.is_finite());
    assert!(row.reads.iter().all(|&r| r == 1), "reads per frame {:?}", row.reads);
}

//! A measurement (ignored by default): the learned routing's GPU time (`learned_route.slang`,
//! the `route` stage of `STRESS_LEARNED_TIMING=1`) at scale. Scene: the free wall struck by
//! its ram beside a field of `STRESS_ROUTE_COST_BODIES` (default 5,000) bonded two-chunk
//! bodies resting in rows on the ground, touching (10,000 chunks in contact pairs); routing
//! as `STRESS_LEARNED_ROUTE` says (`world`: tiered; `recorder`: untiered, every pair seeds
//! if its rigid assumption fails). One GPU job at a time.
//!
//!   STRESS_LEARNED_ROUTE=recorder cargo test --release -p stress-rubble --features gpu --test learned_route_cost -- --ignored --nocapture
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::builders::{auto_bonds, body, grid};
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

#[test]
#[ignore]
fn routing_cost_at_scale() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "learned");
    std::env::set_var("STRESS_LEARNED_STUB", "punch");
    std::env::set_var("STRESS_LEARNED_DRY", "1");
    std::env::set_var("STRESS_LEARNED_TIMING", "1");
    let n: usize = std::env::var("STRESS_ROUTE_COST_BODIES").ok().and_then(|v| v.parse().ok()).unwrap_or(5000);
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParityRouted;
    let mut scene = realtime(later(wall_ram_late(), 2), 0.15);
    // rows of two-chunk bodies (0.1 m cubes), 2 mm apart along x, behind the wall (y > 1 m)
    let per_row = 100;
    for b in 0..n {
        let (i, j) = (b % per_row, b / per_row);
        let corner = Vec3::new(-5.0 + 0.202 * i as f64, 1.5 + 0.102 * j as f64, scene.ground.as_ref().map_or(0.0, |g| g.height) + 0.0005);
        let chunks = grid(corner, [2, 1, 1], Vec3::splat(0.1), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        scene.bodies.push(body(&format!("piece{b}"), chunks, bonds));
    }
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
        let t = std::time::Instant::now();
        w.step_frame(&gpu).expect("frame");
        let calls = w.learned.as_ref().map_or(0, |l| l.totals.calls);
        eprintln!("frame {f}: {:.1} ms wall, calls so far {calls}", t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("{} bodies ({} chunks), {} islands", n + 1, 2 * n + 220, w.solver.mirror.clusters.len());
    // the stage times print when the learned path drops (with the world)
}

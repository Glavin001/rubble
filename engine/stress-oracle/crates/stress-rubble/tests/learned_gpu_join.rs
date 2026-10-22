//! The learned path's earlier routing (`STRESS_LEARNED_ROUTING=sweep`: the punch's sweep,
//! calls per impactor group) joins a body that would meet a call's held bodies to the call;
//! the CPU's routing (the default, `learned_route.rs`) has no such rule, as stress-ref's
//! `select_islands` has none. A file of its own: the switches are read once per process.
//!
//!   cargo test --release -p stress-rubble --features gpu --test learned_gpu_join
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

fn run(scene: stress_ref::scene::Scene) -> stress_gpu::learned::Totals {
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
    for _ in 0..frames {
        w.step_frame(&gpu).expect("frame");
    }
    let totals = w.learned.as_ref().map(|l| l.totals).unwrap_or_default();
    eprintln!("learned {totals:?}");
    totals
}

/// With the sweep, a body outside a call that would meet the call's held (static) bodies
/// during the call's frame joins the call: a slow brick (3 m/s, below the routing speed) sliding into the
/// free wall's side face while the ram's call holds the wall.
#[test]
fn a_body_reaching_a_held_island_joins_the_call() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "learned");
    std::env::set_var("STRESS_LEARNED_STUB", "punch");
    std::env::set_var("STRESS_DEBRIS_ROUTING", "1");
    std::env::set_var("STRESS_LEARNED_ROUTING", "sweep");
    use stress_ref::builders::{auto_bonds, body, grid};
    let mut scene = realtime(later(wall_ram_late(), 2), 0.25);
    // the wall's +x face is at x = 0.5; the brick's at 0.725, closing at 3 m/s: it reaches the
    // wall 75 ms in, inside the frame after the strike's call (held from 66.7 ms)
    let chunks = grid(Vec3::new(0.725, 0.05, 0.45), [1, 1, 1], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    let mut brick = body("brick", chunks, bonds);
    brick.linear_velocity = [-3.0, 0.0, 0.0];
    scene.bodies.push(brick);
    let t = run(scene);
    assert!(t.calls > 0 && t.applied > 0);
    assert!(t.joined >= 1, "the brick joined the call: {t:?}");
}

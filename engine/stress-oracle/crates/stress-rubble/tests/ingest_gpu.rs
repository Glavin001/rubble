//! The GPU contact ingest (stress-gpu `ingest`, docs/stress-fracture-gpu-resident.md
//! phase 2) against the host's: rubble's solved contact rows read on the GPU as contact
//! manifolds per (chunk, other body) must match the manifolds the host builds from the
//! same rows read back (`contacts_of_step` and stress-ref `manifolds`), in the coupled
//! world's own frames.
//!
//!   cargo test --release -p stress-rubble --features gpu --test ingest_gpu
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::math::Vec3;
use stress_ref::scene::Scene;
use stress_rubble::scenes::{house, realtime, wall_ram};
use stress_rubble::RubbleEngine;

fn check(gpu: &Gpu, scene: &Scene, frames: usize, min_manifolds: u64) {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let mut engine = RubbleEngine::with_device(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone()).expect("rubble");
    let stats = engine.attach_ingest_check(gpu.clone());
    let mut w = GpuWorld::with_engine(gpu, scene, Box::new(engine)).expect("coupled world");
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    let stats = stats.get();
    eprintln!("{}: {stats:?}", scene.name);
    assert!(stats.manifolds >= min_manifolds, "{}: {} manifolds compared", scene.name, stats.manifolds);
    assert_eq!(stats.mismatched_steps, 0, "{}: manifold sets differ", scene.name);
    assert!(stats.impulse < 1e-4, "{}: impulse error {}", scene.name, stats.impulse);
    assert!(stats.point < 1e-4, "{}: point error {} m", scene.name, stats.point);
    assert!(stats.closing < 1e-4, "{}: closing speed error {} m/s", scene.name, stats.closing);
}

/// Free concrete blocks (3 x 3 x 2 chunks of 0.1 m, unbonded from each other): two on the
/// ground, the upper one spinning, and a third falling onto them at 8 m/s (CCD substeps):
/// chunk-ground and chunk-chunk contacts the engine resolves (no anchors, no impactor).
fn blocks_on_ground() -> Scene {
    use stress_ref::builders::*;
    use stress_ref::scene::*;
    let mut s = new_scene("blocks_on_ground", "free concrete blocks on the ground, one dropped onto them");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
    for (name, x, z, vz) in [("lower", 0.0, 0.0, 0.0), ("upper", 0.0, 0.22, 0.0), ("falling", 0.12, 0.8, -8.0)] {
        let chunks = grid(Vec3::new(x - 0.15, -0.15, z), [3, 3, 2], Vec3::splat(0.1), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        let mut b = body(name, chunks, bonds);
        b.linear_velocity = [0.0, 0.0, vz];
        if name == "upper" {
            b.angular_velocity = [0.0, 0.0, 2.0];
        }
        s.bodies.push(b);
    }
    s
}

#[test]
fn gpu_ingest_matches_the_host_manifolds() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    check(&gpu, &realtime(blocks_on_ground(), 0.5), 30, 20);
    check(&gpu, &realtime(wall_ram(), 0.5), 30, 0);
    let mut house_ram = realtime(stress_ref::scenes::rammed(house()), 0.2);
    house_ram.name = "house_ram".into();
    check(&gpu, &house_ram, 12, 0);
}

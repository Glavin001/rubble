//! Chunk crushing in the fracture lab's GPU-resident threshold world
//! (`ThresholdWorld::new_with_crush`, removal only): the punch-through wall (ScenePack v3)
//! struck by its pack's ball loses the blocks it strikes, counted in the world's stats;
//! with crushing off the same strike crushes nothing.
//!
//!   cargo test -p stress-rubble --features gpu --test threshold_crush
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_ref::capability::Profile;
use stress_ref::scene::{ImpactorDesc, ImpactorShape};
use stress_rubble::threshold::ThresholdWorld;

fn wall_scene() -> stress_ref::scene::Scene {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../crates/engine-webgpu/fracture/assets/crush-wall.json");
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (scene, _) = stress_ref::scene_pack::import_value_with(&json, "crush-wall", "structure", "crush-wall", Profile::Threshold.import_options()).unwrap();
    scene.validate().unwrap();
    let mut scene = scene.with_derived();
    scene.sim.methods = stress_ref::scene::Methods::established();
    Profile::Threshold.apply_scene(&mut scene);
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    scene
}

/// Chunks crushed after the pack's ball (0.45 m, 4800 kg, 35 m/s) hits the wall's middle.
fn crushed(gpu: &Gpu, crush: bool) -> (u32, u32) {
    let scene = wall_scene();
    let mut w = ThresholdWorld::new_with_crush(gpu, &scene, 0.45, 4800.0, crush).unwrap();
    // The pack is Y-up; the stress frame Z-up: (x, y, z) -> (x, -z, y).
    w.launch(
        gpu,
        &ImpactorDesc {
            name: "ball".into(),
            shape: ImpactorShape::Sphere { radius: 0.45 },
            mass: 4800.0,
            position: [4.0, -0.2, 1.8],
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: [-35.0, 0.0, 0.0],
            angular_velocity: [0.0; 3],
            material: "projectile".into(),
            crush: None,
        },
    );
    for _ in 0..20 {
        w.step_frame(gpu).unwrap();
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let c = w.fracture.counters(gpu);
    assert_eq!(c.error, 0, "fracture errors {:#x}", c.error);
    let removed = w.fracture.crush_counts(gpu).0;
    let state = w.fracture.crush_state(gpu);
    assert_eq!(state.iter().filter(|s| s[3].to_bits() & 1 != 0).count() as u32, removed);
    (removed, c.broken)
}

#[test]
fn the_lab_crushes_the_struck_blocks() {
    let gpu = Gpu::new().expect("GPU");
    let (on, broken_on) = crushed(&gpu, true);
    let (off, broken_off) = crushed(&gpu, false);
    eprintln!("crushing on: {on} chunks crushed, {broken_on} bonds broken; off: {off}, {broken_off}");
    assert!(on > 0, "the ball crushes the blocks it strikes");
    assert_eq!(off, 0);
}

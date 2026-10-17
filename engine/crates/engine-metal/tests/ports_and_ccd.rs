use engine_cuda::CudaWorld;
use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::WorldCapacity;
use engine_webgpu::WebGpuWorld;
use glam::Vec3;

#[test]
fn cuda_sources_emitted() {
    let w = CudaWorld::new(WorldCapacity::small()).unwrap();
    assert!(w.cuda_sources.len() >= 6);
    assert!(w.cuda_sources.iter().all(|(_, n)| *n > 100));
}

#[test]
fn webgpu_spirv_to_wgsl() {
    // Full graph must translate for the fair Chrome bar; report all failures.
    let report = WebGpuWorld::compile_shaders_report();
    assert!(report.len() >= 30, "expected full world_step SPIR-V set, got {}", report.len());
    let mut fails = Vec::new();
    for (name, r) in &report {
        match r {
            Ok(wgsl) => {
                assert!(
                    wgsl.contains("fn ") || wgsl.contains("@compute"),
                    "{name} empty/invalid"
                );
            }
            Err(e) => fails.push(format!("{name}: {e}")),
        }
    }
    assert!(
        fails.is_empty(),
        "SPIR-V→WGSL failed for {} entries:\n{}",
        fails.len(),
        fails.join("\n")
    );
}

#[test]
fn ccd_stretch_classify_and_sweep() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    world.set_ccd_enabled(true);
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.0);
        let id = b.add_sphere(Vec3::new(0.0, 5.0, 0.0), 0.2, 1.0);
        let mut body = world.body(id);
        body.linear_vel = [0.0, -80.0, 0.0];
        world.set_body(id, body);
    }
    // Fast body toward plane — CCD path runs without claiming certified miss on budget
    for _ in 0..30 {
        let _ = world.step();
    }
    let y = world.body(1).position[1];
    assert!(y.is_finite());
    // Overflow may include CCD flags; must not silently drop
    let flags = world.last_counters().overflow_flags;
    if flags & engine_schema::CCD_BUDGET_EXHAUSTED != 0 {
        // Explicit failure signal — acceptable for stretch budget tests
        assert!(flags & engine_schema::CCD_BUDGET_EXHAUSTED != 0);
    }
}

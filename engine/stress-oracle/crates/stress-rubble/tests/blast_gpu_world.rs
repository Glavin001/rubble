//! vibe-land's stress model in the GPU-only world (`Profile::BlastGpu`: stress-gpu's Slang port
//! of Blast's solver in `ThresholdWorld`'s fracture frame). Under gravity alone a house is a
//! static structure on its supports: the world's Blast solve must be the f32 reference's,
//! frame for frame. Struck, it breaks and splits on the GPU with nothing read back, and the
//! run repeats bit for bit.
//!
//!   cargo test --release -p stress-rubble --features gpu --test blast_gpu_world
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_ref::blast_bond::{AngLin, BlastSolver};
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape, Scene};
use stress_rubble::threshold::ThresholdWorld;

fn house() -> Scene {
    let p = Profile::BlastGpu;
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/house-1story.json");
    let mut scene =
        stress_ref::scene_pack::import_with(&path, "house-1story", "house", p.import_options())
            .expect("house-1story")
            .with_derived();
    p.apply_scene(&mut scene);
    scene
        .materials
        .entry("projectile".into())
        .or_insert_with(stress_ref::builders::elastic_steel);
    scene
}

fn world(gpu: &Gpu, scene: &Scene) -> ThresholdWorld {
    ThresholdWorld::with_profile(gpu, scene, 0.35, 1400.0, Profile::BlastGpu)
        .expect("blast_gpu world")
}

fn difference(a: &[AngLin], b: &[AngLin]) -> f64 {
    let scale = b
        .iter()
        .map(|j| j.lin.norm().max(j.ang.norm()))
        .fold(0.0, f64::max)
        .max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.lin - y.lin).norm().max((x.ang - y.ang).norm()))
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn a_house_under_gravity_is_the_reference_solve_frame_for_frame() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let scene = house();
    let mut w = world(&gpu, &scene);
    let graph = w.fracture.blast().expect("Blast set").solver.graph.clone();
    let mut reference = BlastSolver::new(graph.clone());
    reference.device_precision = true;
    // The house is anchored: its structure frame is the world's, and gravity is its only load.
    let g = Vec3::from_array(scene.gravity);
    let accel = vec![
        AngLin {
            ang: Vec3::ZERO,
            lin: g
        };
        graph.node_count()
    ];
    for frame in 0..20 {
        w.step_frame(&gpu).expect("frame");
        reference.solve(&accel, true);
        let device = w.fracture.blast().unwrap().solver.impulses();
        let r: Vec<AngLin> = (0..graph.bond_count())
            .map(|j| reference.impulse(j))
            .collect();
        let e = difference(&device, &r);
        println!("frame {frame}: {e:.3e} from the f32 reference");
        // Bounded by f32 rounding's spread on this house (blast_host_vs_reference: 3e-3 over
        // the first frames), so the frame's inputs are the reference's.
        assert!(e < 3e-2, "frame {frame}: {e:.3e}");
    }
    let stats = w.poll(&gpu);
    assert_eq!(stats.error, 0, "{stats:?}");
}

/// A ball through the house: the bonds Blast breaks split the house on the GPU.
fn strike(gpu: &Gpu) -> (Vec<bool>, u32, u32) {
    let scene = house();
    let mut w = world(gpu, &scene);
    let all: Vec<_> = scene.bodies.iter().flat_map(|b| &b.chunks).collect();
    let c = all[all.len() / 2].center;
    let shot = ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.35 },
        mass: 1400.0,
        position: [c[0] - 6.0, c[1], c[2]],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [60.0, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "projectile".into(),
        crush: None,
    };
    for frame in 0..60 {
        if frame == 2 {
            assert!(w.launch(gpu, &shot));
        }
        w.step_frame(gpu).expect("frame");
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let stats = w.poll(gpu);
    assert_eq!(stats.error, 0, "{stats:?}");
    (w.fracture.bonds_intact(gpu), stats.broken, stats.bodies)
}

#[test]
fn a_struck_house_breaks_splits_and_repeats() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let (intact, broken, bodies) = strike(&gpu);
    let intact_count = intact.iter().filter(|b| **b).count();
    println!(
        "{broken} bonds broken ({} of {} intact), {bodies} bodies",
        intact_count,
        intact.len()
    );
    assert!(
        broken > 0 && bodies > 1,
        "the strike breaks bonds and splits the house"
    );
    // The counters are read a frame or more late (`poll` never waits); the bonds were read after
    // the GPU finished.
    assert!(broken as usize <= intact.len() - intact_count);
    let again = strike(&gpu);
    assert_eq!(again.0, intact, "the run repeats bit for bit");
}

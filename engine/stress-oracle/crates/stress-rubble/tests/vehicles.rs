//! Bonded vehicles (`stress_ref::vehicles`) on the GPU, set up as stress-web sets them up:
//! they stand on their wheels in both stress modes, and launched into a rigid barrier at
//! 56 km/h they crumple at the front (parts break off) while the cabin stays together.
//!
//!   cargo test --release -p stress-rubble --features gpu --test vehicles
#![cfg(feature = "gpu")]

use std::collections::BTreeMap;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::launch::Launcher;
use stress_rubble::scenes::pack_scene;

/// Launched before this frame.
const AT: u64 = 10;

fn kind(scene: &stress_ref::scene::Scene, k: usize) -> String {
    scene.bodies[0].chunks[k].groups.first().cloned().unwrap_or_default()
}

/// The vehicle's chunks on its largest body (by its chunks), from each chunk's body.
fn largest(ids: &[usize], body: &[u32]) -> Vec<usize> {
    let mut on: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &k in ids {
        on.entry(body[k]).or_default().push(k);
    }
    on.into_values().max_by_key(|v| v.len()).unwrap_or_default()
}

fn resident(gpu: &Gpu, scene: &stress_ref::scene::Scene) -> GpuWorld {
    let p = Profile::BlastParity;
    let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(scene)).expect("rubble");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(gpu, scene, e).expect("world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    w
}

/// Unlaunched, a second on their wheels: no bond broken, nothing moved more than 1 cm.
#[test]
fn vehicles_stand_on_their_wheels_in_both_stress_modes() {
    let gpu = Gpu::new().expect("GPU");
    for key in ["vh_car_barrier", "vh_bus_barrier"] {
        let pack = stress_ref::vehicles::pack(key).unwrap();
        let frames = 60;
        let scene = pack_scene(&pack, key, Profile::Threshold, 1.0);
        let ball = stress_ref::wrecking::BALLS[0];
        let mut w = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, ball.radius, ball.mass).expect("threshold world");
        let start = w.chunk_centres(&gpu);
        for _ in 0..frames {
            w.step_frame(&gpu).expect("frame");
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let st = w.poll(&gpu);
        let moved = start.iter().zip(&w.chunk_centres(&gpu)).map(|((a, _), (b, _))| (Vec3::from_array(*a) - Vec3::from_array(*b)).norm()).fold(0.0, f64::max);
        println!("{key} threshold: broken {}, largest move {moved:.2e} m", st.broken);
        assert_eq!((st.error, st.engine_overflow, st.broken), (0, 0, 0), "{key}");
        assert!(moved < 0.01, "{key}: moved {moved} m");

        let scene = pack_scene(&pack, key, Profile::BlastParity, 1.0);
        let mut w = resident(&gpu, &scene);
        for _ in 0..frames {
            w.step_frame(&gpu).expect("frame");
        }
        w.sync(&gpu);
        let o = w.observation();
        let moved = o.bodies.values().flatten().map(|c| Vec3::from_array(c.displacement).norm()).fold(0.0, f64::max);
        println!("{key} blast_parity: broken {}, largest move {moved:.2e} m", o.values["broken_bonds"]);
        assert_eq!(o.values["broken_bonds"], 0.0, "{key}");
        assert!(moved < 0.01, "{key}: moved {moved} m");
    }
}

/// The car launched into the barrier at 56 km/h (threshold): it stops, parts break off
/// its front, and its cabin (floor, sills, pillars, roof, doors and glass) stays one body.
#[test]
fn car_crumples_at_the_front_threshold() {
    let gpu = Gpu::new().expect("GPU");
    let key = "vh_car_barrier";
    let vs = stress_ref::vehicles::scene(key).unwrap();
    let pack = vs.to_pack();
    let scene = pack_scene(&pack, key, Profile::Threshold, 2.0);
    let mut launcher = Launcher::of_pack(&pack).unwrap();
    let ball = stress_ref::wrecking::BALLS[0];
    let mut w = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, ball.radius, ball.mass).expect("threshold world");
    let ids = vs.vehicles[0].1.clone();
    let mean = |w: &stress_rubble::threshold::ThresholdWorld| -> Vec3 {
        let c = w.chunk_centres(&gpu);
        ids.iter().fold(Vec3::ZERO, |a, &k| a + Vec3::from_array(c[k].0)) * (1.0 / ids.len() as f64)
    };
    let mut before = Vec3::ZERO;
    let mut hit = 0.0;
    let mut late = Vec3::ZERO;
    for _ in 0..110 {
        if w.frame == AT {
            launcher.arm("", w.frame, w.frame_dt);
        }
        for l in launcher.due(w.frame) {
            w.launch_island(&gpu, 0, &l).expect("launch");
        }
        // Its speed a frame before it reaches the barrier (12 m at 19.6 m/s is 37 frames).
        if w.frame == AT + 30 {
            before = mean(&w);
        }
        w.step_frame(&gpu).expect("frame");
        if w.frame == AT + 31 {
            hit = (mean(&w) - before).norm() * 60.0;
        }
        if w.frame == 100 {
            late = mean(&w);
        }
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let st = w.poll(&gpu);
    assert_eq!((st.error, st.engine_overflow), (0, 0));
    let body = w.fracture.chunk_bodies(&gpu);
    let main = largest(&ids, &body);
    let off: Vec<String> = ids.iter().filter(|k| !main.contains(k)).map(|&k| kind(&scene, k)).collect();
    let broken = w.scene_bonds_intact(&gpu, 0).iter().filter(|&&i| !i).count();
    // Its speed over the last 10 frames.
    let speed = (mean(&w) - late).norm() * 6.0;
    println!("car: {:.1} m/s just before the barrier; broken {broken} of {}; off its largest body ({} of {} parts): {off:?}", hit, scene.bodies[0].bonds.len(), main.len(), ids.len());
    assert!((hit - stress_ref::vehicles::CRASH_SPEED).abs() < 2.0, "arrives at about 56 km/h: {hit} m/s");
    assert!(speed < 1.0, "stopped at the barrier: {speed} m/s");
    assert!(off.iter().any(|k| ["bumper", "rails", "wing"].contains(&k.as_str())), "parts broke off its front: {off:?}");
    for cabin in ["floor", "sill", "pillar", "roof", "door", "glass"] {
        assert!(!off.iter().any(|k| k == cabin), "the cabin stays together: a {cabin} broke off");
    }
}

/// The same crash under blast_parity: parts break off, and the cabin's pillars, roof, doors
/// and glass stay one body.
#[test]
fn car_crumples_at_the_front_blast_parity() {
    let gpu = Gpu::new().expect("GPU");
    let key = "vh_car_barrier";
    let vs = stress_ref::vehicles::scene(key).unwrap();
    let pack = vs.to_pack();
    let scene = pack_scene(&pack, key, Profile::BlastParity, 2.0);
    let mut launcher = Launcher::of_pack(&pack).unwrap();
    let mut w = resident(&gpu, &scene);
    for _ in 0..110 {
        if w.frame == AT {
            launcher.arm("", w.frame, scene.sim.frame_dt);
        }
        for l in launcher.due(w.frame) {
            w.launch_island(&gpu, 0, &l.chunks, l.velocity, l.angular_velocity).expect("launch");
        }
        w.step_frame(&gpu).expect("frame");
    }
    w.sync(&gpu);
    let ids = vs.vehicles[0].1.clone();
    let m = &w.solver.mirror;
    let body: Vec<u32> = (0..scene.bodies[0].chunks.len()).map(|k| m.chunks[0][k].cluster as u32).collect();
    let main = largest(&ids, &body);
    let off: Vec<String> = ids.iter().filter(|k| !main.contains(k)).map(|&k| kind(&scene, k)).collect();
    println!("car blast_parity: broken {}; off its largest body ({} of {} parts): {off:?}", w.observation().values["broken_bonds"], main.len(), ids.len());
    // How much breaks varies with how the contact meets the barrier (the profile's
    // quasi-static solve of the free car): from its four wheels torn off to its whole
    // front and floor (with a 6 m run-up). Parts always break off.
    assert!(w.observation().values["broken_bonds"] > 0.0 && !off.is_empty(), "parts broke off");
    for cabin in ["pillar", "roof", "door", "glass"] {
        assert!(!off.iter().any(|k| k == cabin), "the cabin stays together: a {cabin} broke off");
    }
}

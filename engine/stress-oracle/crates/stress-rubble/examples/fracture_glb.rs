//! A fracture GLB (the Fracture lab's assets) under the GPU stress solver coupled to rubble,
//! natively: the reference the browser's stress mode is compared with.
//!
//!   cargo run --release -p stress-rubble --example fracture_glb -- [GLB] [LEVEL] [FRAMES]
//!
//! `STRESS_PROFILE=threshold`: the v0 stress mode (`Profile::Threshold`) instead of `blast_parity`.
use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape};

fn main() {
    let mut args = std::env::args().skip(1);
    let file = args.next().unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../crates/engine-webgpu/fracture/assets/veneer-house.glb").to_string());
    let level: u8 = args.next().map_or(0, |l| l.parse().expect("LEVEL"));
    let frames: usize = args.next().map_or(120, |f| f.parse().expect("FRAMES"));
    // A procedural sample key (`cuboid`, `cuboidGrid`, `cuboids`, ...): the lab's cuboid
    // houses in that layout (`engine_fracture::samples::cuboid_stress_scene`); else a GLB.
    let (json, name) = match engine_fracture::samples::sample(&file).ok().filter(|s| s.procedural()) {
        Some(s) => (engine_fracture::samples::cuboid_stress_scene(s.layout).expect("cuboid houses").1, s.key.to_string()),
        None => {
            let bytes = std::fs::read(&file).expect("GLB");
            let asset = engine_fracture::parse_fracture_glb(&bytes).expect("fracture GLB");
            (engine_fracture::stress_export::shardwright_stress_json(&asset, level), asset.physics.name.clone())
        }
    };
    let scene = stress_ref::shardwright::import_value_with(&json, &name, true).expect("stress import");
    scene.validate().expect("valid");
    let mut scene = scene.with_derived();
    // The GPU's methods (the bench and the suites run `STRESS_METHODS=established`).
    scene.sim.methods = stress_ref::scene::Methods::established();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let parts: usize = scene.bodies.iter().flat_map(|b| &b.chunks).map(|c| c.compound.as_ref().map_or(1, |p| p.len())).sum();
    println!("{}: level {level}, {chunks} chunks, {parts} hulls, {} bonds, unsupported: {:?}", name, scene.bodies.iter().map(|b| b.bonds.len()).sum::<usize>(), stress_gpu::world::unsupported(&scene));
    // `STRESS_PROFILE=threshold`: another capability profile (default `blast_parity`).
    let p = std::env::var("STRESS_PROFILE").ok().map(|v| Profile::parse(&v).unwrap_or_else(|| panic!("STRESS_PROFILE={v}: no such profile"))).unwrap_or(Profile::BlastParity);
    p.apply_scene(&mut scene);
    scene.sim.duration = frames as f64 * scene.sim.frame_dt;
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    let gpu = Gpu::new().expect("GPU");
    let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let t = std::time::Instant::now();
    let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    println!("built in {:.2} s", t.elapsed().as_secs_f64());
    // One cannonball at the structure's side, mid height, at 60 m/s.
    let all: Vec<_> = scene.bodies.iter().flat_map(|b| &b.chunks).collect();
    let lo = all.iter().fold([f64::INFINITY; 3], |a, c| [a[0].min(c.center[0]), a[1].min(c.center[1]), a[2].min(c.center[2])]);
    let hi = all.iter().fold([f64::NEG_INFINITY; 3], |a, c| [a[0].max(c.center[0]), a[1].max(c.center[1]), a[2].max(c.center[2])]);
    let target = [lo[0], 0.5 * (lo[1] + hi[1]), lo[2] + 0.4 * (hi[2] - lo[2])];
    let shot = ImpactorDesc { name: "ball".into(), shape: ImpactorShape::Sphere { radius: 0.35 }, mass: 1400.0, position: [target[0] - 6.0, target[1], target[2]], orientation: [1.0, 0.0, 0.0, 0.0], velocity: [60.0, 0.0, 0.0], angular_velocity: [0.0; 3], material: "projectile".into(), crush: None };
    // `STRESS_SHOT=px,py,pz,vx,vy,vz,mass,radius` (the stress frame, +Z up): another shot,
    // e.g. one the browser's stress mode fired (its stats' `lastShot`).
    let shot = match std::env::var("STRESS_SHOT") {
        Ok(v) => {
            let x: Vec<f64> = v.split(',').map(|x| x.parse().expect("STRESS_SHOT numbers")).collect();
            ImpactorDesc { shape: ImpactorShape::Sphere { radius: x[7] }, mass: x[6], position: [x[0], x[1], x[2]], velocity: [x[3], x[4], x[5]], ..shot }
        }
        Err(_) => shot,
    };
    let mut ms = Vec::new();
    for f in 0..frames {
        if f == 5 {
            let imp = stress_ref::world::Impactor::from_desc(&shot, &w.scene);
            w.add_impactor(&gpu, imp).expect("shot");
        }
        let t = std::time::Instant::now();
        w.step_frame(&gpu).expect("frame");
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        if std::env::var("STRESS_FRAMES").is_ok() {
            eprintln!("frame {f}: {:.1} ms, {} resident frames so far", ms[f], w.resident_frames);
        }
    }
    w.sync(&gpu);
    let o = w.observation();
    let mut sorted = ms[1..].to_vec();
    sorted.sort_by(f64::total_cmp);
    println!("fragments {}, broken bonds {}, frame ms median {:.1}, worst {:.1}, resident frames {}", o.values["fragments"], o.values["broken_bonds"], sorted[sorted.len() / 2], sorted[sorted.len() - 1], w.resident_frames);
    // Bonds to the ground (`import_value_with`'s ground bonds): how many broke.
    let m = &w.solver.mirror;
    let ground = |si: usize, b: &stress_ref::solver::RtBond| m.structures[si].bonds[b.source].groups.iter().any(|g| g == "ground");
    let (gb, gt) = m.bonds.iter().enumerate().flat_map(|(si, bs)| bs.iter().map(move |b| (si, b))).filter(|(si, b)| ground(*si, b)).fold((0, 0), |(n, t), (_, b)| (n + !b.connected() as usize, t + 1));
    println!("ground bonds broken {gb} of {gt}");
}

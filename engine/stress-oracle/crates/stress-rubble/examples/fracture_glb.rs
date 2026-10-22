//! A fracture GLB (the Fracture lab's assets) under the GPU stress solver coupled to rubble,
//! natively: the reference the browser's stress mode is compared with.
//!
//!   cargo run --release -p stress-rubble --example fracture_glb -- [GLB] [LEVEL] [FRAMES]
//!
//! A scene pack (`.json`, engine/scenes/vibe-city's export, e.g. fracturedTower) runs as the
//! lab's stress modes run it: `stress_ref::scene_pack` under the profile's import options,
//! shots with the pack's ball.
//! `STRESS_PROFILE=threshold`: the v0 stress mode (`Profile::Threshold`) instead of `blast_parity`.
//! `STRESS_SHOTS=N`: N shots of the lab's ball (its mass, radius and speed for a cuboid
//! sample), one every 30 frames from frame 5, each at another chunk from the -x side; the
//! frame times' median, p95 and worst over the frames after the first shot.
//! `STRESS_HZ=120`: the physics frame rate (the lab's Physics menu; default the scene's).
//! `STRESS_SHOT_MASS=16`: the shots' mass times this (the lab's Ball mass).
//! `STRESS_VIEW=1`: also take the browser's view (`GpuWorld::view_with`) after every frame,
//! timed apart.
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
    let mut ball = (1400.0, 0.35, 60.0);
    // `STRESS_PROFILE=threshold`: another capability profile (default `blast_parity`).
    let p = std::env::var("STRESS_PROFILE").ok().map(|v| Profile::parse(&v).unwrap_or_else(|| panic!("STRESS_PROFILE={v}: no such profile"))).unwrap_or(Profile::BlastParity);
    let scene = if file.ends_with(".json") {
        let bytes = std::fs::read(&file).expect("scene pack");
        let (lab, info) = engine_fracture::pack::build_pack_scene(&bytes).expect("scene pack");
        ball = (lab.projectile_mass, lab.projectile_radius, engine_fracture::plan::default_shot_speed(lab.extent, lab.projectile_speed));
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("scene pack JSON");
        stress_ref::scene_pack::import_value_with(&json, &info.key, "structure", &info.key, p.import_options()).expect("pack import").0
    } else {
    let (json, name) = match engine_fracture::samples::sample(&file).ok().filter(|s| s.procedural()) {
        Some(s) => {
            let (lab, json) = engine_fracture::samples::cuboid_stress_scene(s.layout).expect("cuboid houses");
            ball = (lab.projectile_mass, lab.projectile_radius, lab.projectile_speed.unwrap_or(60.0));
            (json, s.key.to_string())
        }
        None => {
            let bytes = std::fs::read(&file).expect("GLB");
            let asset = engine_fracture::parse_fracture_glb(&bytes).expect("fracture GLB");
            (engine_fracture::stress_export::shardwright_stress_json(&asset, level), asset.physics.name.clone())
        }
    };
    stress_ref::shardwright::import_value_with(&json, &name, true).expect("stress import")
    };
    let name = file.rsplit('/').next().unwrap_or(&file).to_string();
    let mut scene = scene;
    // Before `with_derived`, as stress-web sets the lab's Physics rate.
    if let Some(hz) = std::env::var("STRESS_HZ").ok().map(|v| v.parse::<f64>().expect("STRESS_HZ")) {
        scene.sim.frame_dt = 1.0 / hz;
    }
    scene.validate().expect("valid");
    let mut scene = scene.with_derived();
    // The GPU's methods (the bench and the suites run `STRESS_METHODS=established`).
    scene.sim.methods = stress_ref::scene::Methods::established();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let parts: usize = scene.bodies.iter().flat_map(|b| &b.chunks).map(|c| c.compound.as_ref().map_or(1, |p| p.len())).sum();
    println!("{}: level {level}, {chunks} chunks, {parts} hulls, {} bonds, unsupported: {:?}", name, scene.bodies.iter().map(|b| b.bonds.len()).sum::<usize>(), stress_gpu::world::unsupported(&scene));
    p.apply_scene(&mut scene);
    if let Some(k) = std::env::var("STRESS_SHOT_MASS").ok().map(|v| v.parse::<f64>().expect("STRESS_SHOT_MASS")) {
        ball.0 *= k;
    }
    scene.sim.duration = frames as f64 * scene.sim.frame_dt;
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    let gpu = Gpu::new().expect("GPU");
    // `STRESS_THRESHOLD_GPU=1`: the threshold profile's GPU-resident world (stress-rubble
    // `threshold`) instead of the resident GpuWorld, the same shots.
    if std::env::var("STRESS_THRESHOLD_GPU").is_ok_and(|v| v == "1") {
        threshold_gpu(&gpu, &scene, ball, frames);
        return;
    }
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
    let shots: usize = std::env::var("STRESS_SHOTS").ok().map_or(0, |v| v.parse().expect("STRESS_SHOTS"));
    let shot_at = |k: usize| {
        let c = all[(k * 7919 + 13) % all.len()].center;
        let (mass, radius, speed) = ball;
        ImpactorDesc { name: format!("shot{k}"), shape: ImpactorShape::Sphere { radius }, mass, position: [c[0] - 6.0, c[1], c[2].max(lo[2] + radius)], velocity: [speed, 0.0, 0.0], ..shot.clone() }
    };
    let mut ms = Vec::new();
    let view = std::env::var("STRESS_VIEW").is_ok_and(|v| v == "1");
    let mut view_ms: Vec<f64> = Vec::new();
    for f in 0..frames {
        let fired = if shots > 0 { (f >= 5 && (f - 5) % 30 == 0 && (f - 5) / 30 < shots).then(|| shot_at((f - 5) / 30)) } else { (f == 5).then(|| shot.clone()) };
        if let Some(desc) = fired {
            let imp = stress_ref::world::Impactor::from_desc(&desc, &w.scene);
            w.add_impactor(&gpu, imp).expect("shot");
        }
        let t = std::time::Instant::now();
        w.step_frame(&gpu).expect("frame");
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        if view {
            let t = std::time::Instant::now();
            // As the browser draws: the view's reads first, then the drawing's look.
            w.view_prepare(&gpu);
            let n = w.view_with(&gpu, |m, imps| m.clusters.len() + imps.len());
            std::hint::black_box(n);
            view_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        if std::env::var("STRESS_FRAMES").is_ok() {
            w.sync(&gpu);
            let o = w.observation();
            eprintln!("frame {f}: {:.1} ms, {} resident frames so far, broken {}, fragments {}", ms[f], w.resident_frames, o.values["broken_bonds"], o.values["fragments"]);
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
    let after = &mut ms[5.min(ms.len())..].to_vec();
    after.sort_by(f64::total_cmp);
    if !after.is_empty() {
        if !view_ms.is_empty() {
            view_ms.sort_by(f64::total_cmp);
            println!("view_state ms median {:.1}, p95 {:.1}, worst {:.1}", view_ms[view_ms.len() / 2], view_ms[view_ms.len() * 95 / 100], view_ms[view_ms.len() - 1]);
        }
        println!("after the first shot: frame ms median {:.1}, p95 {:.1}, worst {:.1}; {} bodies, {} compactions", after[after.len() / 2], after[after.len() * 95 / 100], after[after.len() - 1], m.clusters.len(), w.compactions);
    }
}

/// The GPU-resident threshold world on `scene`: `STRESS_SHOTS` shots of `ball` (mass,
/// radius, speed) as the GpuWorld run fires them. Each frame timed to the GPU's end (a
/// wait the browser does not do), then the whole run again with frames queued as a
/// browser queues them (no wait), and the counters.
fn threshold_gpu(gpu: &Gpu, scene: &stress_ref::scene::Scene, ball: (f64, f64, f64), frames: usize) {
    let (mass, radius, speed) = ball;
    let all: Vec<_> = scene.bodies.iter().flat_map(|b| &b.chunks).collect();
    let lo = all.iter().fold([f64::INFINITY; 3], |a, c| [a[0].min(c.center[0]), a[1].min(c.center[1]), a[2].min(c.center[2])]);
    let shots: usize = std::env::var("STRESS_SHOTS").ok().map_or(0, |v| v.parse().expect("STRESS_SHOTS"));
    let shot_at = |k: usize| {
        let c = all[(k * 7919 + 13) % all.len()].center;
        ImpactorDesc { name: format!("shot{k}"), shape: ImpactorShape::Sphere { radius }, mass, position: [c[0] - 6.0, c[1], c[2].max(lo[2] + radius)], orientation: [1.0, 0.0, 0.0, 0.0], velocity: [speed, 0.0, 0.0], angular_velocity: [0.0; 3], material: "projectile".into(), crush: None }
    };
    // `STRESS_THRESHOLD_RUNS=wait,wait`: which runs, in order (default: one of each).
    let runs = std::env::var("STRESS_THRESHOLD_RUNS").unwrap_or_else(|_| "wait,queue".into());
    for (label, wait) in runs.split(',').map(|r| if r == "wait" { ("each frame waited for", true) } else { ("queued (no wait)", false) }) {
        let t = std::time::Instant::now();
        let mut w = stress_rubble::threshold::ThresholdWorld::new(gpu, scene, radius, mass).expect("threshold world");
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        println!("threshold GPU world: {} chunks, {} bonds, {} body slots, built in {:.2} s", w.tables.chunk_count(), w.tables.bond_count(), w.tables.body_count(), t.elapsed().as_secs_f64());
        let mut ms = Vec::with_capacity(frames);
        let run = std::time::Instant::now();
        for f in 0..frames {
            if shots > 0 && f >= 5 && (f - 5) % 30 == 0 && (f - 5) / 30 < shots {
                w.launch(gpu, &shot_at((f - 5) / 30));
            }
            // `STRESS_THRESHOLD_CHECK=1`: no chunk jumps further in a frame than its body's
            // motion allows (a split re-parents bodies in place).
            let check = std::env::var("STRESS_THRESHOLD_CHECK").is_ok_and(|v| v == "1");
            let before = check.then(|| w.chunk_centres(gpu));
            let t = std::time::Instant::now();
            w.step_frame(gpu).expect("frame");
            if let Some(before) = before {
                let after = w.chunk_centres(gpu);
                let dt = w.frame_dt;
                let mut worst = (0.0f64, 0usize);
                for (k, ((a, va), (b, vb))) in before.iter().zip(&after).enumerate() {
                    let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
                    let allowed = (va.max(*vb)) * dt + 0.02;
                    if d - allowed > worst.0 {
                        worst = (d - allowed, k);
                    }
                }
                if worst.0 > 0.0 {
                    let st = w.poll(gpu);
                    eprintln!("frame {f}: chunk {} jumped {:.3} m beyond its motion ({} broken so far)", worst.1, worst.0, st.broken);
                }
            }
            if wait {
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            }
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            // `STRESS_THRESHOLD_WHY=f0,f1`: the bonds that broke in frames f0..f1, with their
            // capacity, their chunks' loads and support flags.
            if let Some((f0, f1)) = std::env::var("STRESS_THRESHOLD_WHY").ok().and_then(|v| v.split_once(',').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()))) {
                if f >= f0 && f < f1 {
                    thread_local!(static PREV: std::cell::RefCell<Vec<bool>> = const { std::cell::RefCell::new(Vec::new()) });
                    let intact = w.fracture.bonds_intact(gpu);
                    let loads: Vec<[f32; 4]> = gpu.read_range(&w.fracture.loads, 0, w.tables.chunk_count() as usize);
                    PREV.with_borrow_mut(|prev| {
                        if prev.len() == intact.len() {
                            for (i, (&now, &was)) in intact.iter().zip(prev.iter()).enumerate() {
                                if was && !now {
                                    let b = w.tables.bonds[i];
                                    let (a, c) = (b[0] as usize, b[1] as usize);
                                    eprintln!("frame {f}: bond {i} ({a}-{c}) capacity {:.3e} N, loads {:.3e} / {:.3e} N, support {} / {}, masses {:.1} / {:.1} kg", f32::from_bits(b[2]), loads[a][0], loads[c][0], w.tables.info[a], w.tables.info[c], w.tables.chunks[a][0][3], w.tables.chunks[c][0][3]);
                                }
                            }
                        }
                        *prev = intact;
                    });
                }
            }
            if std::env::var("STRESS_GPU_QS_LOADS").is_ok() {
                let loads: Vec<[f32; 4]> = gpu.read_range(&w.fracture.loads, 0, w.tables.chunk_count() as usize);
                let loaded = loads.iter().filter(|l| l[0] > 0.0).count();
                let total: f64 = loads.iter().map(|l| l[0] as f64).sum();
                let peak = loads.iter().map(|l| l[0] as f64).fold(0.0, f64::max);
                eprintln!("frame {f} loads: {loaded} chunks loaded, sum |f| {total:.4e} N, largest {peak:.4e} N");
            }
            let st = w.poll(gpu);
            if std::env::var("STRESS_FRAMES").is_ok() {
                eprintln!("frame {f}: {:.2} ms, {:?}", ms[f], st);
            }
        }
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let total = run.elapsed().as_secs_f64() * 1e3;
        let st = w.poll(gpu);
        // The exact end state (a blocking read): counters, and the engine's bodies hashed
        // (two runs must agree bit for bit).
        let exact = w.fracture.counters(gpu);
        let bodies: Vec<u8> = gpu.read(w.engine.sim.body_buffer());
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bodies.hash(&mut h);
            h.finish()
        };
        println!("  end state: broken {}, new bodies {}, split frames {}, errors {:#x}; bodies hash {hash:x}", exact.broken, exact.pieces, exact.split_frames, exact.error);
        let after = &mut ms[5.min(ms.len())..].to_vec();
        after.sort_by(f64::total_cmp);
        println!(
            "{label}: frame ms median {:.2}, p95 {:.2}, worst {:.2}; {} frames in {:.0} ms ({:.2} ms a frame); broken {}, bodies {}, split frames {}, errors {:#x}, engine overflow {:#x}",
            after[after.len() / 2],
            after[after.len() * 95 / 100],
            after[after.len() - 1],
            frames,
            total,
            total / frames as f64,
            st.broken,
            st.bodies,
            st.split_frames,
            st.error,
            st.engine_overflow
        );
    }
}

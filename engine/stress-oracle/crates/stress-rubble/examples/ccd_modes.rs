//! How the coupling's fast-body modes change a strike (measurement): the free-wall ram
//! (`scenes::wall_ram_late`, reference 35.1 m/s and 6 fragments) under a capability
//! profile, with the engine's fast impactors handled four ways. One JSON line per run.
//!
//!   cargo run --release -p stress-rubble --features gpu --example ccd_modes -- \
//!       [--profile blast_parity|blast_parity_routed] [--modes none,substeps,toi,spec] \
//!       [--scene ram|slab|house] [--speed 250] [--plain] [--decoy none|plain|ccd]
//!
//! `--plain`: the striking body not CCD. `--decoy`: a second ball far off, flying away from
//! everything (a CCD one that touches nothing: does the scene's other contact change?).
//!
//! `--scene slab`: a 2 kg steel ball of 5 cm radius at `--speed` m/s into a fixed steel
//! slab 5 cm thick (too strong to break: a ball beyond it tunnelled). `--scene house`:
//! the house-1story pack under gravity, a 300 kg ball of 0.25 m radius at `--speed` m/s
//! into its side at mid height.
//! - `oracle`: the standalone world (its own contact), the reference.
//! - `none`: the profile's policy (one engine step a frame: no CCD).
//! - `substeps`: whole-world substeps bounded by the fast bodies' travel (up to 256).
//! - `toi`: the fast bodies `BODY_CCD` in one step (`set_toi_ccd`).
//! - `refN`: N engine steps a frame (default 32) for every body, CCD off: the converged
//!   reference a one-step mode is judged against.
//! - `spec`: one step with look-ahead contacts for the whole scene
//!   (`STRESS_RUBBLE_SPECULATIVE_CCD=1`).

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::scene::Scene;
use stress_rubble::scenes::{realtime, wall_ram_late};
use stress_rubble::RubbleEngine;

fn slab(speed: f64) -> Scene {
    use stress_ref::builders::*;
    use stress_ref::scene::*;
    let mut s = new_scene("ccd_slab", "a fast steel ball into a fixed steel slab");
    s.gravity = [0.0; 3];
    s.materials.insert("steel".into(), elastic_steel());
    let mut chunks = grid(Vec3::new(-0.5, 1.0, -0.5), [5, 1, 5], Vec3::new(0.2, 0.05, 0.2), "steel");
    for c in &mut chunks {
        c.support = Support::Fixed;
    }
    let bonds = auto_bonds(&chunks, |_, _| "steel".into());
    s.bodies.push(body("slab", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.05 },
        mass: 2.0,
        position: [0.03, 1.0 - speed / 60.0 * 0.5, 0.02],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
        ccd: true,
    });
    s
}

fn house(speed: f64) -> Scene {
    use stress_ref::scene::*;
    let mut s = stress_rubble::scenes::house();
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for b in &s.bodies {
        for c in &b.chunks {
            for a in 0..3 {
                lo[a] = lo[a].min(c.center[a]);
                hi[a] = hi[a].max(c.center[a]);
            }
        }
    }
    s.materials.entry("steel".into()).or_insert_with(stress_ref::builders::elastic_steel);
    let start = lo[1] - 0.3 - speed / 60.0 * 0.5;
    s.impactors.push(ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.25 },
        mass: 300.0,
        position: [0.5 * (lo[0] + hi[0]) + 0.13, start, lo[2] + 0.4 * (hi[2] - lo[2])],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
        ccd: true,
    });
    s
}

fn main() {
    let mut profile = Profile::BlastParityRouted;
    let mut modes = vec!["none".to_string(), "substeps".into(), "toi".into(), "spec".into()];
    let mut which = "ram".to_string();
    let mut speed = 250.0;
    let mut plain = false;
    let mut decoy = "none".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--profile" => profile = Profile::parse(&args.next().expect("--profile NAME")).expect("profile"),
            "--modes" => modes = args.next().expect("--modes LIST").split(',').map(String::from).collect(),
            "--scene" => which = args.next().expect("--scene ram|slab"),
            "--speed" => speed = args.next().and_then(|v| v.parse().ok()).expect("--speed M/S"),
            "--plain" => plain = true,
            "--decoy" => decoy = args.next().expect("--decoy none|plain|ccd"),
            other => panic!("unknown argument {other}"),
        }
    }
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let gpu = Gpu::new().expect("GPU");
    for mode in &modes {
        // Read when the engine is made.
        if mode == "spec" {
            std::env::set_var("STRESS_RUBBLE_SPECULATIVE_CCD", "1");
        } else {
            std::env::remove_var("STRESS_RUBBLE_SPECULATIVE_CCD");
        }
        let base = match which.as_str() {
            "slab" => slab(speed),
            "house" => house(speed),
            _ => wall_ram_late(),
        };
        let mut scene = realtime(base, 0.25);
        scene.impactors[0].ccd = !plain;
        if decoy != "none" {
            let mut d = scene.impactors[0].clone();
            d.name = "decoy".into();
            d.position = [d.position[0] + 50.0, d.position[1] - 50.0, d.position[2] + 20.0];
            d.velocity = [0.0, -300.0, 0.0];
            d.ccd = decoy == "ccd";
            scene.impactors.push(d);
        }
        profile.apply_scene(&mut scene);
        if mode == "oracle" {
            // The standalone world: its own rigid bodies and contact (the reference).
            let mut w = stress_ref::world::World::new(&scene);
            let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
            for _ in 0..frames {
                w.step_frame();
            }
            let o = w.observation();
            let ram = &w.impactors[0];
            println!(
                "{{\"scene\":\"{which}\",\"profile\":\"{}\",\"mode\":\"oracle\",\"ram_speed\":{:.3},\"ram_y\":{:.3},\"fragments\":{},\"broken\":{}}}",
                profile.name(),
                ram.velocity.norm(),
                ram.pose.position.y,
                o.values["fragments"],
                o.values["broken_bonds"]
            );
            continue;
        }
        let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
        let (iterations, substeps) = profile.engine_policy();
        e.set_policy(iterations, if mode == "substeps" { Some(256) } else { substeps });
        e.set_toi_ccd(mode == "toi");
        if let Some(n) = mode.strip_prefix("ref") {
            // The converged reference: every frame in `n` (default 32) engine steps.
            e.substeps = n.parse().unwrap_or(32);
        }
        let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("resident world");
        w.redo_fractured_frames = false;
        w.set_impact_islands(true);
        w.apply_profile(profile);
        if profile == Profile::BlastParityRouted {
            w.set_tiered(true).expect("tiered coupling");
        }
        let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
        let mut worst = 0.0f64;
        let t0 = std::time::Instant::now();
        for _ in 0..frames {
            let t = std::time::Instant::now();
            w.step_frame(&gpu).expect("frame");
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            worst = worst.max(t.elapsed().as_secs_f64() * 1e3);
        }
        w.sync(&gpu);
        let o = w.observation();
        let ram = &w.solver.impactors[0];
        println!(
            "{{\"scene\":\"{which}\",\"plain\":{plain},\"decoy\":\"{decoy}\",\"profile\":\"{}\",\"mode\":\"{mode}\",\"ram_speed\":{:.3},\"ram_y\":{:.3},\"fragments\":{},\"broken\":{},\"frames\":{frames},\"total_ms\":{:.1},\"worst_frame_ms\":{worst:.2}}}",
            profile.name(),
            ram.velocity.norm(),
            ram.pose.position.y,
            o.values["fragments"],
            o.values["broken_bonds"],
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
}

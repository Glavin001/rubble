//! Native Metal run of `engine-webgpu/fracture/regression.mjs` on the bundled
//! arch: the shared plan and Slang kernels rest without drift, a crown impact
//! wakes and moves fragments with valid state and floor support, and reset is
//! exact. Plus every Scene-menu layout loads and steps on Metal.

use engine_fracture::samples::{base_scene, layout_scene, SAMPLES};
use engine_fracture::scene::{collider_vertices, SHAPE_CONVEX};
use engine_fracture::{build_scene, parse_fracture_glb};
use engine_viewer::fracture::FractureSim;
use metal::Device;
use std::path::PathBuf;

fn asset(file: &str) -> engine_fracture::FractureAsset {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/fracture/assets")
        .join(file);
    parse_fracture_glb(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn arch_impact_wakes_moves_and_resets() {
    let scene = build_scene(&asset("masonry-arch.glb"), 0, None)
        .unwrap()
        .without_joints();
    assert_eq!(scene.selected.len(), 23);
    let device = Device::system_default().expect("Metal device");
    let mut sim = FractureSim::new(&device, &device.new_command_queue(), &scene).unwrap();
    let (n, start) = (scene.bodies.len(), scene.projectile_start as usize);

    let before = sim.snapshot(n).unwrap();
    sim.step(120).unwrap();
    let rest = sim.snapshot(n).unwrap();
    for i in 1..start {
        for k in 0..3 {
            assert_eq!(
                rest[i * 28 + k],
                before[i * 28 + k],
                "arch moved before impact (body {i})"
            );
        }
    }
    // Aim a real sphere at the crown from in front of the arch.
    let crown = scene.bodies[1..start]
        .iter()
        .max_by(|a, b| a.p[1].total_cmp(&b.p[1]))
        .unwrap()
        .p;
    let origin = [crown[0], crown[1], crown[2] + 3.0];
    let projectile = sim.launch(origin, [0.0, 0.0, -1.0]).unwrap() as usize;
    let (mut peak_contacts, mut peak_awake) = (0, 0);
    for _ in 0..12 {
        sim.step(60).unwrap();
        let c = sim.last;
        assert_eq!(c.overflow, 0, "GPU capacity overflow");
        peak_contacts = peak_contacts.max(c.contact);
        peak_awake = peak_awake.max(c.awake);
    }
    let after = sim.snapshot(n).unwrap();
    let mut max_displacement = 0f32;
    let mut min_hull_y = f64::INFINITY;
    for i in 1..start {
        let o = i * 28;
        for k in [0, 1, 2, 4, 5, 6, 7, 12, 13, 14, 16, 17, 18] {
            assert!(after[o + k].is_finite(), "nonfinite state in body {i}");
        }
        let ql = after[o + 4..o + 8]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((ql - 1.0).abs() < 1e-3, "invalid orientation in body {i}");
        let d: f32 = (0..3)
            .map(|k| (after[o + k] - before[o + k]).powi(2))
            .sum::<f32>()
            .sqrt();
        max_displacement = max_displacement.max(d);
    }
    for c in scene.colliders.iter().filter(|c| c.shape == SHAPE_CONVEX) {
        let o = c.body as usize * 28;
        let f = |k: usize| after[o + k] as f64;
        for v in collider_vertices(
            &scene.hull_data,
            c,
            [f(0), f(1), f(2)],
            [f(4), f(5), f(6), f(7)],
        ) {
            min_hull_y = min_hull_y.min(v[1]);
        }
    }
    println!("steps {} peak contacts {peak_contacts} peak awake {peak_awake} max displacement {max_displacement:.3} min hull y {min_hull_y:.4}", sim.steps());
    assert!(
        peak_contacts > 0 && peak_awake > 1,
        "projectile did not wake fragments"
    );
    assert!(max_displacement > 0.1, "projectile did not move the arch");
    assert!(
        min_hull_y > -0.025,
        "hulls penetrated the floor: {min_hull_y} m"
    );
    assert!(
        after[projectile * 28 + 1] as f64 >= scene.projectile_radius - 0.025,
        "sphere fell through floor"
    );

    sim.reset();
    let reset = sim.snapshot(n).unwrap();
    assert!(
        reset
            .iter()
            .zip(&before)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "reset did not restore every body field"
    );
    assert_eq!(sim.shots(), 0, "reset did not reload projectiles");
}

/// Every bundled sample builds from the shared Rust code and runs on Metal:
/// the row shot wakes pieces, nothing overflows, poses stay finite.
#[test]
fn every_scene_menu_entry_runs_on_metal() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    for sample in SAMPLES {
        let asset = sample.file.map(asset);
        let level = asset
            .as_ref()
            .map_or(Some(0), |a| {
                engine_fracture::samples::initial_level(a, sample.level)
            })
            .unwrap();
        let one = base_scene(asset.as_ref(), level, sample.max_hull_vertices)
            .unwrap()
            .without_joints();
        let scene = layout_scene(one, sample.layout, asset.is_none());
        let mut sim = FractureSim::new(&device, &queue, &scene).unwrap();
        sim.step(2).unwrap();
        // Rows, grids and suburbs: the ready-made row shot; otherwise at the highest piece.
        let top = scene.bodies[1..scene.projectile_start as usize]
            .iter()
            .map(|b| b.p)
            .max_by(|a, b| a[1].total_cmp(&b[1]))
            .unwrap();
        let shot = scene.row_shot.unwrap_or(engine_fracture::scene::Ray {
            origin: [top[0], top[1], top[2] + scene.extent],
            direction: [0.0, 0.0, -1.0],
        });
        sim.launch(shot.origin, shot.direction).unwrap();
        sim.step(60).unwrap();
        let c = sim.last;
        let snap = sim.snapshot(scene.bodies.len()).unwrap();
        println!(
            "{}: {} bodies · {} awake · {} contacts · {} candidates",
            sample.key,
            scene.bodies.len(),
            c.awake,
            c.contact,
            c.candidate
        );
        assert_eq!(c.overflow, 0, "{}: overflow", sample.key);
        assert!(
            c.awake >= 1 && c.contact > 0,
            "{}: the shot found nothing",
            sample.key
        );
        assert!(
            snap.iter().all(|x| x.is_finite()),
            "{}: non-finite bodies",
            sample.key
        );
    }
}

/// A 300 m/s shot at the arch crown (2.5 m a step at 120 Hz): without CCD it
/// passes the arch between steps and barely moves it; a CCD shot
/// (`Planner::set_projectile_ccd`) strikes the crown (time-of-impact contact,
/// then the inelastic momentum exchange, `ccd_respond`) and knocks it out, then
/// carries on through the gap (engine/docs/ccd.md, in the lab's own plan).
/// Phase-through itself is checked exactly in `fracture_tower_ccd.rs`.
#[test]
fn a_ccd_shot_does_not_tunnel_through_the_arch() {
    let scene = build_scene(&asset("masonry-arch.glb"), 0, None)
        .unwrap()
        .without_joints();
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let (n, start) = (scene.bodies.len(), scene.projectile_start as usize);
    let crown = scene.bodies[1..start]
        .iter()
        .max_by(|a, b| a.p[1].total_cmp(&b.p[1]))
        .unwrap()
        .p;
    let back = scene.bodies[1..start]
        .iter()
        .map(|b| b.p[2])
        .fold(f64::INFINITY, f64::min);
    let mut moved = [0f32; 2];
    for (k, ccd) in [false, true].into_iter().enumerate() {
        let mut sim = FractureSim::new(&device, &queue, &scene).unwrap();
        sim.step(10).unwrap();
        let before = sim.snapshot(n).unwrap();
        sim.plan.set_projectile_ccd(ccd);
        let scale = 300.0 / sim.plan.shot_speed();
        sim.plan.set_projectile_speed_scale(scale).unwrap();
        let origin = [crown[0], crown[1], crown[2] + 3.0];
        let projectile = sim.launch(origin, [0.0, 0.0, -1.0]).unwrap() as usize;
        let mut min_z = f32::INFINITY;
        for _ in 0..30 {
            sim.step(1).unwrap();
            assert_eq!(sim.last.overflow, 0, "GPU capacity overflow");
            let snap = sim.snapshot(n).unwrap();
            min_z = min_z.min(snap[projectile * 28 + 2]);
        }
        let after = sim.snapshot(n).unwrap();
        moved[k] = (1..start)
            .map(|i| {
                (0..3)
                    .map(|c| (after[i * 28 + c] - before[i * 28 + c]).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .fold(0.0, f32::max);
        println!(
            "ccd {ccd}: shot reached z {min_z:.2} (arch centres from z {back:.2}), arch moved {:.3} m",
            moved[k]
        );
        if ccd {
            assert!(after[projectile * 28..projectile * 28 + 28].iter().all(|x| x.is_finite()));
        } else {
            assert!(
                (min_z as f64) < back - 2.0,
                "control: the plain shot should pass the arch (z {min_z})"
            );
        }
    }
    // The tunnelling shot only grazes the crown's look-ahead contacts.
    assert!(
        moved[1] > 5.0 && moved[1] > 20.0 * moved[0],
        "the CCD shot should knock the crown out ({} vs control {})",
        moved[1],
        moved[0]
    );
}

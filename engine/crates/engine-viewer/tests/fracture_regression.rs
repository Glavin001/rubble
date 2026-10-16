//! Native Metal port of `engine-webgpu/fracture/regression.mjs` on the bundled arch.
//! Runs the shared Slang kernels headlessly: rest without drift, a crown impact that
//! wakes and moves fragments with valid state and floor support, then exact reset.

use engine_fracture::{build_scene, parse_fracture_glb};
use engine_viewer::fracture::FractureSim;
use metal::Device;
use std::path::PathBuf;

fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let t = cross(u, v);
    let t = [t[0] + q[3] * v[0], t[1] + q[3] * v[1], t[2] + q[3] * v[2]];
    let c = cross(u, t);
    [v[0] + 2.0 * c[0], v[1] + 2.0 * c[1], v[2] + 2.0 * c[2]]
}

fn step_frames(sim: &mut FractureSim, frames: u32) {
    for _ in 0..frames {
        sim.submit(1).unwrap();
        sim.wait().unwrap();
    }
}

#[test]
fn arch_impact_wakes_moves_and_resets() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/fracture/assets/masonry-arch.glb");
    let asset = parse_fracture_glb(&std::fs::read(path).unwrap()).unwrap();
    let scene = build_scene(&asset, 0).unwrap();
    assert_eq!(scene.fragment_count, 23);
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let mut sim = FractureSim::new(&device, &queue, &scene).unwrap();
    let n = scene.bodies.len();
    let start = scene.projectile_start as usize;

    let before = sim.snapshot(n).unwrap();
    step_frames(&mut sim, 120);
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
        step_frames(&mut sim, 60);
        let c = sim.last;
        assert_eq!(c.overflow, 0, "GPU capacity overflow");
        peak_contacts = peak_contacts.max(c.contact);
        peak_awake = peak_awake.max(c.awake);
    }
    let after = sim.snapshot(n).unwrap();
    let mut max_displacement = 0f32;
    for i in 1..start {
        let o = i * 28;
        for k in [0, 1, 2, 4, 5, 6, 7, 12, 13, 14, 16, 17, 18] {
            assert!(after[o + k].is_finite(), "nonfinite state in body {i}");
        }
        let q = &after[o + 4..o + 8];
        let ql = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((ql - 1.0).abs() < 1e-3, "invalid orientation in body {i}");
        let d: f32 = (0..3)
            .map(|k| (after[o + k] - before[o + k]).powi(2))
            .sum::<f32>()
            .sqrt();
        max_displacement = max_displacement.max(d);
    }
    let mut min_hull_y = f32::INFINITY;
    for c in scene.colliders.iter().filter(|c| c.shape == 3) {
        let o = c.body as usize * 28;
        let q = [after[o + 4], after[o + 5], after[o + 6], after[o + 7]];
        for i in 0..c.params[1] as usize {
            let h = scene.hull_data[c.params[0] as usize + 1 + i];
            min_hull_y = min_hull_y.min(rotate(q, [h[0], h[1], h[2]])[1] + after[o + 1]);
        }
    }
    println!(
        "steps {} peak contacts {peak_contacts} peak awake {peak_awake} max displacement {max_displacement:.3} min hull y {min_hull_y:.4} timing {:?}",
        sim.steps, sim.timing
    );
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
        after[projectile * 28 + 1] >= scene.projectile_radius - 0.025,
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
    assert_eq!(sim.shots, 0, "reset did not reload projectiles");
}

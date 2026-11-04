//! Diagnostic: the load path's "block not positive definite" error (fracture.slang
//! `fr_lp_block`, error bit 64). Runs a scene pack under `load_path` with a shot until the
//! error appears, reads the bonds' states back, and factors every chunk's 6×6 block in
//! f64 (stress-ref `Network::block`, `cholesky6`): a block f64 factors and the GPU's f32
//! does not is precision; one f64 cannot factor is singular (its bonds' stiffness spans
//! fewer than six directions).
//!
//!   cargo run --release -p stress-rubble --features gpu --example load_path_blocks -- <pack.json> [speed scale] [radius m] [mass kg]
use stress_gpu::gpu::Gpu;
use stress_ref::capability::Profile;
use stress_ref::load_path::cholesky6;
use stress_rubble::threshold::ThresholdWorld;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("a pack path");
    let k: f64 = args.get(2).map_or(16.0, |v| v.parse().unwrap());
    let bytes = std::fs::read(path).unwrap();
    let p = Profile::LoadPath;
    let (lab, _) = engine_fracture::pack::build_pack_scene(&bytes).unwrap();
    let r: f64 = args.get(3).map_or(1.0, |v| v.parse().unwrap());
    let m: f64 = args.get(4).map_or(30_000.0, |v| v.parse().unwrap());
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let key = path.rsplit('/').next().unwrap().trim_end_matches(".json").to_string();
    let (scene, _) = stress_ref::scene_pack::import_value_with(&json, &key, "structure", &key, p.import_options()).unwrap();
    let mut scene = scene.with_derived();
    scene.sim.methods = stress_ref::scene::Methods::established();
    p.apply_scene(&mut scene);
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    let gpu = Gpu::new().unwrap();
    let mut w = ThresholdWorld::with_profile(&gpu, &scene, r, m, p).unwrap();
    // The lab's shot at the centre from in front (lab frame Y up; the stress frame is it less
    // the pack's placement, turned Z up).
    let c = lab.center;
    let origin = [c[0], c[1], c[2] + 3.0 * lab.extent.max(1.0)];
    let speed = k * engine_fracture::plan::default_shot_speed(lab.extent, lab.projectile_speed);
    let v = engine_fracture::plan::shot_velocity(c, lab.extent, r, origin, [0.0, 0.0, -1.0], Some(speed), lab.projectile_arc);
    let placement = lab.placement;
    let zup = |v: [f64; 3]| [v[0], -v[2], v[1]];
    w.launch(
        &gpu,
        &stress_ref::scene::ImpactorDesc {
            name: "shot0".into(),
            shape: stress_ref::scene::ImpactorShape::Sphere { radius: r },
            mass: m,
            position: zup([0, 1, 2].map(|i| origin[i] - placement[i])),
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: zup(v),
            angular_velocity: [0.0; 3],
            material: "projectile".into(),
            crush: None,
        },
    );
    // Each frame: every non-support chunk's block over its intact bonds, factored in f64
    // and in f32 (the GPU's precision, the same algorithm as `fr_lp_block`).
    let net = w.load_path_network();
    let mut first_f32 = None;
    for frame in 1..=240 {
        w.step_frame(&gpu).unwrap();
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        if frame % 4 != 0 {
            continue;
        }
        let c = w.fracture.counters(&gpu);
        let alive = w.fracture.bonds_intact(&gpu);
        let (mut f64_bad, mut f32_bad, mut examples) = (0, 0, Vec::new());
        for k in 0..net.chunks.len() {
            if net.chunks[k].support || !(0..net.bonds.len()).any(|i| alive[i] && (net.bonds[i].a == k || net.bonds[i].b == k)) {
                continue;
            }
            let a = net.block(k, &alive);
            let ok64 = cholesky6(&a).is_some();
            let ok32 = cholesky6_f32(&a);
            f64_bad += usize::from(!ok64);
            f32_bad += usize::from(!ok32);
            if ok64 && !ok32 && examples.len() < 3 {
                let n = (0..net.bonds.len()).filter(|&i| alive[i] && (net.bonds[i].a == k || net.bonds[i].b == k)).count();
                examples.push((k, n, (0..6).map(|i| a[i][i]).collect::<Vec<_>>()));
            }
        }
        if f64_bad + f32_bad > 0 || c.error != 0 {
            println!("frame {frame}: {} broken, error {:#x}; blocks f64 cannot factor {f64_bad}, f32 cannot factor {f32_bad}", c.broken, c.error);
            for e in &examples {
                println!("  chunk {} ({} bonds) diagonal {}", e.0, e.1, e.2.iter().map(|x| format!("{x:.2e}")).collect::<Vec<_>>().join(" "));
            }
            first_f32.get_or_insert(frame);
        }
    }
    println!("first frame with an unfactorable f32 block: {first_f32:?}");
}

/// [`cholesky6`] in f32: whether it factors.
fn cholesky6_f32(a: &[[f64; 6]; 6]) -> bool {
    let mut l = [0.0f32; 21];
    let tri = |i: usize, j: usize| i * (i + 1) / 2 + j;
    for i in 0..6 {
        for j in 0..=i {
            let mut s = a[i][j] as f32;
            for p in 0..j {
                s -= l[tri(i, p)] * l[tri(j, p)];
            }
            if i == j {
                if !(s > 0.0) {
                    return false;
                }
                l[tri(i, i)] = s.sqrt();
            } else {
                l[tri(i, j)] = s / l[tri(j, j)];
            }
        }
    }
    true
}

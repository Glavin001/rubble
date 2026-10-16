//! Headless parity probe: same deterministic shot as the browser console check.
//! `cargo run -p engine-viewer --release --example fracture_compare -- house 0`
use engine_fracture::{build_scene, parse_fracture_glb};
use engine_viewer::fracture::FractureSim;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let file = match args.get(1).map_or("house", |s| s.as_str()) {
        "arch" => "masonry-arch.glb",
        _ => "veneer-house.glb",
    };
    let level: u8 = args.get(2).map_or(Ok(0), |s| s.parse())?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/fracture/assets")
        .join(file);
    let scene = build_scene(&parse_fracture_glb(&std::fs::read(path)?)?, level)?;
    let device = metal::Device::system_default().unwrap();
    let mut sim = FractureSim::new(&device, &device.new_command_queue(), &scene)?;
    let (n, ps) = (scene.bodies.len(), scene.projectile_start as usize);
    let before = sim.snapshot(n)?;
    let max_y = |a: &[f32]| (1..ps).map(|i| a[i * 28 + 1]).fold(f32::MIN, f32::max);
    sim.launch([0.0, scene.center[1], 8.0], [0.0, 0.0, -1.0])?;
    if args.get(3).is_some_and(|a| a == "clip") {
        for _ in 0..40 {
            sim.submit(1)?;
            sim.wait()?;
        }
        let a = sim.snapshot(n)?;
        let h = &scene.hull_data;
        let v3 = |r: [f32; 4]| glam::Vec3::new(r[0], r[1], r[2]);
        let hull = |c: usize| {
            let col = scene.colliders[c];
            let b = col.body as usize * 28;
            let p = glam::Vec3::new(a[b], a[b + 1], a[b + 2]);
            let q = glam::Quat::from_xyzw(a[b + 4], a[b + 5], a[b + 6], a[b + 7]);
            let base = col.params[0] as usize;
            let c0 = h[base];
            (base, c0[0] as usize, c0[1] as usize, p, q)
        };
        let gpu = unsafe {
            std::slice::from_raw_parts(sim.bufs[18].contents() as *const f32, h.len() * 3)
        };
        let bad: Vec<usize> = (0..h.len())
            .filter(|&i| (0..3).any(|k| gpu[i * 3 + k].to_bits() != h[i][k].to_bits()))
            .collect();
        println!(
            "hull records differing on GPU: {} first {:?}",
            bad.len(),
            &bad[..bad.len().min(10)]
        );
        let (ca, cb, rf, inf) = (1usize, 50usize, 2usize, 5usize);
        for (name, c, face) in [("ref", ca, rf), ("inc", cb, inf)] {
            let (base, nv, nf, p, q) = hull(c);
            let r = h[base + 1 + nv + nf + face];
            let normal = q * v3(h[base + 1 + nv + face]);
            let verts: Vec<String> = (0..r[1] as usize)
                .map(|i| format!("{:.3}", p + q * v3(h[r[0] as usize + i])))
                .collect();
            println!(
                "{name} c{c} body pos {p:.3} face {face} n {normal:.3} range {:?}: {}",
                (r[0], r[1]),
                verts.join(" ")
            );
        }
        return Ok(());
    }
    if args.get(3).is_some_and(|a| a == "bodies") {
        for _ in 0..40 {
            sim.submit(1)?;
            sim.wait()?;
        }
        let a = sim.snapshot(n)?;
        let moved = (1..ps)
            .filter(|&i| (0..8).any(|k| a[i * 28 + k] != before[i * 28 + k]))
            .count();
        println!("moved {moved}");
        for i in [1usize, 2, 20, 30, 40, 45, 46, 47, 48, 50, 60, 122] {
            let row: Vec<String> = a[i * 28..i * 28 + 8]
                .iter()
                .map(|x| format!("{x:.4}"))
                .collect();
            println!("{i}: {} f{}", row.join(","), a[i * 28 + 11].to_bits());
        }
        return Ok(());
    }
    if args.get(3).is_some_and(|a| a == "contacts") {
        for _ in 0..40 {
            sim.submit(1)?;
            sim.wait()?;
        }
        let mut rows: Vec<String> = sim
            .inspect_collisions()?
            .iter()
            .map(|c| {
                let f = |k: usize| f32::from_bits(c[k]);
                format!(
                    "{}-{} f{:x} n({:.3},{:.3},{:.3}) d{:.4} A({:.4},{:.4},{:.4}) B({:.4},{:.4},{:.4})",
                    c[0], c[1], c[2], f(4), f(5), f(6), f(7), f(8), f(9), f(10), f(12), f(13), f(14)
                )
            })
            .collect();
        rows.sort();
        println!("{} contacts\n{}", rows.len(), rows.join("\n"));
        return Ok(());
    }
    if args.get(3).is_some_and(|a| a == "trace") {
        let mut out = Vec::new();
        for k in 0..80 {
            sim.submit(1)?;
            sim.wait()?;
            let c = sim.last;
            out.push(format!(
                "{}/{}/{}/{}/{}",
                c.candidate, c.contact, c.manifold, c.awake, c.color
            ));
            if k == 39 || k == 79 {
                let a = sim.snapshot(n)?;
                println!(
                    "{}\nproj {:.3},{:.3},{:.3}",
                    out.join(" "),
                    a[ps * 28],
                    a[ps * 28 + 1],
                    a[ps * 28 + 2]
                );
                out.clear();
            }
        }
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    for k in 1..=6 {
        for _ in 0..60 {
            sim.submit(1)?;
            sim.wait()?;
        }
        let a = sim.snapshot(n)?;
        let standing = (1..ps)
            .filter(|&i| a[i * 28 + 1] > before[i * 28 + 1] - 0.3)
            .count();
        println!(
            "t={:.1} maxY={:.2} standing={standing} awake={} contacts={}",
            k as f32 * 0.5,
            max_y(&a),
            sim.last.awake,
            sim.last.contact
        );
    }
    println!(
        "{} ticks in {:.0} ms ({:.2} ms/tick incl. per-tick wait)",
        sim.steps,
        t0.elapsed().as_secs_f64() * 1e3,
        t0.elapsed().as_secs_f64() * 1e3 / sim.steps as f64
    );
    Ok(())
}

//! Deterministic parity probe, the native half of the browser comparison:
//! the same scene (shared Rust code), the same shot and tick count, the same
//! command lists; prints counters, body digests and a contact digest to diff
//! against the browser console snippet in engine-webgpu/fracture/README.md.
//!
//! `cargo run -p engine-viewer --release --example fracture_compare -- arch 0 240`
use engine_fracture::samples::{base_scene, initial_level, layout_scene, sample};
use engine_fracture::scene::Ray;
use engine_viewer::fracture::FractureSim;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let entry = sample(args.get(1).map_or("arch", String::as_str))?;
    let ticks: u32 = args.get(3).map_or(Ok(240), |s| s.parse())?;
    let asset = entry
        .file
        .map(|f| engine_fracture::parse_fracture_glb(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/fracture/assets").join(f))?))
        .transpose()?;
    let level = match args.get(2) {
        Some(l) => l.parse()?,
        None => asset.as_ref().and_then(|a| initial_level(a, entry.level)).unwrap_or(0),
    };
    let one = base_scene(asset.as_ref(), level, entry.max_hull_vertices)?.without_joints();
    // Deterministic budget: no SAT-cache early-out, exact sleep gate.
    let mut scene = layout_scene(one, entry.layout, asset.is_none());
    scene.capacity.deterministic = true;
    let device = metal::Device::system_default().unwrap();
    let mut sim = FractureSim::new(&device, &device.new_command_queue(), &scene)?;
    let shot = scene.row_shot.unwrap_or(Ray { origin: [0.0, scene.center[1], scene.extent * 1.5], direction: [0.0, 0.0, -1.0] });
    sim.launch(shot.origin, shot.direction)?;
    let n = scene.bodies.len();
    for k in 1..=ticks {
        sim.step(1)?;
        if k % 60 == 0 || k == ticks {
            let c = sim.last;
            println!("tick {k}: awake {} contacts {} candidates {} colors {} manifolds {}", c.awake, c.contact, c.candidate, c.color, c.manifold);
        }
    }
    let snap = sim.snapshot(n)?;
    let standing = (1..scene.projectile_start as usize).filter(|&i| (snap[i * 28 + 1] as f64 - scene.bodies[i].p[1]).abs() < 0.05).count();
    let sum: f64 = (1..n).map(|i| snap[i * 28] as f64 + snap[i * 28 + 1] as f64 + snap[i * 28 + 2] as f64).sum();
    println!("{} level {level}: {n} bodies · standing {standing} of {} · position sum {sum:.4}", entry.key, scene.projectile_start - 1);
    let (c, metas) = sim.inspect_collisions()?;
    let mut pairs: Vec<(u32, u32)> = metas.iter().map(|m| (m[0].min(m[1]), m[0].max(m[1]))).collect();
    pairs.sort_unstable();
    pairs.dedup();
    println!("inspect: {} contacts · {} candidates · {} manifolds · {} colliding pairs", c.contact, c.candidate, c.manifold, pairs.len());
    Ok(())
}

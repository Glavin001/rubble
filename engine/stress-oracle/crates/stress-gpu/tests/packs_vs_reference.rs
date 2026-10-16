//! The GPU world against stress-ref's `World` on the authored scene packs (convex hull
//! chunks): a box and a sphere thrown at small packs (hull contact with impactors, then
//! between the falling pieces and with the ground), and the villa cracking under its own
//! weight. Compared like the catalogue (`world_vs_reference.rs`), with the reference's
//! own spread beside each comparison. Skipped where the assets are not checked out.
//! `STRESS_GPU_PACKS=rig-portal,...` restricts the cases.

use stress_gpu::gpu::Gpu;
use stress_gpu::world::{unsupported, GpuWorld};
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape, ProbeDesc, ProbeKind, Scene};

mod common;
use common::{difference, gate, spread_difference};

/// The bounds of a scene's chunks (world).
fn bounds(scene: &Scene) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for c in scene.bodies.iter().flat_map(|b| b.chunks.iter()) {
        let p = Vec3::from_array(c.center);
        lo = lo.component_min(p);
        hi = hi.component_max(p);
    }
    (lo, hi)
}

/// An impactor aimed at the middle of the structure's -x face, from 0.2 m outside it,
/// with probes on its velocity and on a few chunks' displacement.
fn with_impactor(mut scene: Scene, shape: ImpactorShape, mass: f64, speed: f64) -> Scene {
    let (lo, hi) = bounds(&scene);
    let mid = (lo + hi) * 0.5;
    let reach = match shape {
        ImpactorShape::Sphere { radius } => radius,
        ImpactorShape::Box { half_extents } => half_extents[0],
    };
    let material = scene.bodies[0].chunks[0].material.clone();
    scene.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape,
        mass,
        position: [lo.x - 0.2 - reach, mid.y, mid.z],
        orientation: [0.0, 0.0, 0.0, 1.0],
        velocity: [speed, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material,
        crush: None,
    });
    scene.probes.push(ProbeDesc { name: "ram_velocity".into(), kind: ProbeKind::ImpactorVelocity { impactor: "ram".into(), axis: [1.0, 0.0, 0.0] } });
    with_chunk_probes(scene)
}

/// Displacement probes on the chunks nearest a few points of the structure.
fn with_chunk_probes(mut scene: Scene) -> Scene {
    let (lo, hi) = bounds(&scene);
    let body = scene.bodies[0].name.clone();
    for (i, f) in [[0.0, 0.5, 0.5], [0.5, 0.5, 1.0], [1.0, 0.5, 0.5]].iter().enumerate() {
        let target = lo + Vec3::new((hi.x - lo.x) * f[0], (hi.y - lo.y) * f[1], (hi.z - lo.z) * f[2]);
        let nearest = (0..scene.bodies[0].chunks.len())
            .min_by(|&a, &b| {
                let d = |k: usize| (Vec3::from_array(scene.bodies[0].chunks[k].center) - target).norm();
                d(a).total_cmp(&d(b))
            })
            .unwrap();
        for (axis, name) in [([1.0, 0.0, 0.0], "x"), ([0.0, 0.0, 1.0], "z")] {
            scene.probes.push(ProbeDesc { name: format!("chunk{i}_{name}"), kind: ProbeKind::ChunkDisplacement { body: body.clone(), chunk: nearest, axis } });
        }
    }
    scene
}

#[test]
fn scene_packs_match_the_reference_world() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs");
    let filter: Option<Vec<String>> = std::env::var("STRESS_GPU_PACKS").ok().map(|v| v.split(',').map(str::to_string).collect());
    let gpu = Gpu::new().expect("GPU");
    let cases: [(&str, &str, f64); 5] =
        [("rig-portal", "box", 0.05), ("comp-wall-bay", "sphere", 0.05), ("rig-toppled", "box", 0.05), ("house-1story", "box", 0.03), ("villa-savoye", "self-weight", 0.05)];
    let mut ran = 0;
    let mut failures = Vec::new();
    for (pack, load, duration) in cases {
        if filter.as_ref().is_some_and(|f| !f.iter().any(|p| pack.starts_with(p.as_str()))) {
            continue;
        }
        let path = dir.join(format!("{pack}.json"));
        if !path.exists() {
            eprintln!("{} not found; skipping", path.display());
            continue;
        }
        let base = stress_ref::scene_pack::import(&path, pack, "structure").expect("import");
        let mut scene = match load {
            "box" => with_impactor(base, ImpactorShape::Box { half_extents: [0.15, 0.3, 0.3] }, 400.0, 12.0),
            "sphere" => with_impactor(base, ImpactorShape::Sphere { radius: 0.25 }, 300.0, 15.0),
            _ => with_chunk_probes(base),
        };
        scene.sim.duration = duration;
        let name = format!("{pack} ({load})");
        if let Some(why) = unsupported(&scene) {
            eprintln!("{name:<28} skipped: {why}");
            continue;
        }
        let t = std::time::Instant::now();
        let ours = GpuWorld::new(&gpu, &scene).and_then(|mut w| w.run(&gpu)).expect("GPU world");
        let gpu_time = t.elapsed().as_secs_f64();
        // The reference's runs come from the cache (computed on a miss): see gate.rs.
        let refs = stress_gpu::gate::reference_set(&scene, true).expect("reference set");
        let (reference, spread, half, ref_time) = (refs.reference, refs.spreads, refs.half, refs.reference_seconds);
        let (gpu_err, gpu_notes) = difference(&reference, &ours);
        let (self_err, self_notes) = spread_difference(&reference, &spread);
        let (half_err, half_notes) = difference(&reference, &half);
        ran += 1;
        eprintln!("{name:<28} gpu {gpu_err:.1e} vs reference's spread {self_err:.1e}, half step {half_err:.1e} | time gpu {gpu_time:.1} s, reference {ref_time:.1} s");
        eprintln!("{:<28}   gpu: {}", "", gpu_notes.join("; "));
        eprintln!("{:<28}   ref spread: {}", "", self_notes.join("; "));
        eprintln!("{:<28}   ref half step: {}", "", half_notes.join("; "));
        let fails = gate(&reference, &ours, &half, &spread);
        if !fails.is_empty() {
            eprintln!("{:<28}   GATE: {}", "", fails.join("; "));
            failures.push(format!("{name}: {}", fails.join("; ")));
        }
    }
    if std::env::var("STRESS_GPU_GATE").as_deref() != Ok("report") {
        assert!(failures.is_empty(), "accuracy gate:\n{}", failures.join("\n"));
    }
    if ran == 0 {
        eprintln!("no scene pack found");
    }
}

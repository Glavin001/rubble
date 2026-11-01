//! The lab's GPU-resident threshold world (`ThresholdWorld`) on a scene pack: a 4 m block on
//! three columns, every column shot away. The freed block falls onto the footings and the
//! ground and rests there. A split is the GPU's alone (stress-gpu `fr_reparent`), so rubble's
//! host mirror never sees the freed pieces move: unless the broadphase grid is sized for the
//! largest chunk collider moving (`WgpuSim::reserve_movable_radius`), the block takes the
//! oversized-static list, which never queries, and falls through the ground.
//!
//!   cargo test --release -p stress-rubble --features gpu --test threshold_lands
#![cfg(feature = "gpu")]

use serde_json::{json, Value};
use stress_gpu::gpu::Gpu;
use stress_ref::capability::Profile;
use stress_ref::scene::{ImpactorDesc, ImpactorShape, Scene};
use stress_rubble::threshold::ThresholdWorld;

/// The pack (Y up): three footings (mass 0: supports) at y 0..0.5, three 0.4 x 2 x 0.4
/// columns of 400 kg on them, a 4 x 4 x 2.8 m block of 40 t on top (y 2.5..6.5). The
/// column bonds are weak in tension and shear: a shot frees its column.
fn pack() -> Vec<u8> {
    let columns = [[-1.0, -1.0], [-1.0, 1.0], [1.6, 0.0]];
    let (mut nodes, mut sizes, mut colliders, mut materials) = (vec![], vec![], vec![], vec![]);
    let mut node = |c: [f64; 3], h: [f64; 3], mass: f64| {
        nodes.push(json!({ "centroid": { "x": c[0], "y": c[1], "z": c[2] }, "mass": mass }));
        sizes.push(json!({ "x": 2.0 * h[0], "y": 2.0 * h[1], "z": 2.0 * h[2] }));
        colliders.push(json!({ "kind": "cuboid", "halfExtents": { "x": h[0], "y": h[1], "z": h[2] } }));
        materials.push("concrete");
        nodes.len() as u32 - 1
    };
    let footings: Vec<u32> = columns.iter().map(|&[x, z]| node([x, 0.25, z], [0.3, 0.25, 0.3], 0.0)).collect();
    let ids: Vec<u32> = columns.iter().map(|&[x, z]| node([x, 1.5, z], [0.2, 1.0, 0.2], 400.0)).collect();
    let top = node([0.0, 4.5, 0.0], [2.0, 2.0, 1.4], 4.0e4);
    let mut bonds = vec![];
    for (k, &[x, z]) in columns.iter().enumerate() {
        for (a, b, y) in [(footings[k], ids[k], 0.5), (ids[k], top, 2.5)] {
            bonds.push(json!({ "node0": a, "node1": b, "m": 1, "area": 0.16,
                "centroid": { "x": x, "y": y, "z": z }, "normal": { "x": 0, "y": 1, "z": 0 } }));
        }
    }
    let material = |name: &str, tension: f64| {
        json!({ "name": name, "elasticModulus": 30e9, "compressionElastic": 1e9, "compressionFatal": 1e9,
            "tensionElastic": tension, "tensionFatal": tension, "shearElastic": tension, "shearFatal": tension })
    };
    let pack: Value = json!({
        "version": 2, "key": "block_frame", "title": "Block on three columns",
        "defaults": { "solver": { "gravity": -9.81, "materialScale": 1,
            "materials": [material("concrete-member", 1e10), material("concrete-shot", 1e3)] } },
        "scenario": { "nodes": nodes, "nodeSizes": sizes, "nodeColliders": colliders, "nodeMaterials": materials, "bonds": bonds },
    });
    serde_json::to_vec(&pack).unwrap()
}

/// The pack as the lab's threshold mode loads it (engine-world `threshold_scene`).
fn scene() -> Scene {
    let json: Value = serde_json::from_slice(&pack()).unwrap();
    let (scene, _) = stress_ref::scene_pack::import_value_with(&json, "block_frame", "structure", "block_frame", Profile::Threshold.import_options()).expect("pack import");
    let mut scene = scene;
    scene.sim.frame_dt = 1.0 / 60.0;
    scene.validate().expect("scene");
    let mut scene = scene.with_derived();
    scene.sim.methods = stress_ref::scene::Methods::established();
    Profile::Threshold.apply_scene(&mut scene);
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    scene
}

#[test]
fn a_block_freed_on_the_gpu_lands_on_the_ground() {
    let gpu = Gpu::new().expect("gpu");
    let scene = scene();
    let up = {
        let g = scene.gravity;
        let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        [-g[0] / n, -g[1] / n, -g[2] / n]
    };
    let height = |p: [f64; 3]| p[0] * up[0] + p[1] * up[1] + p[2] * up[2];
    let ground = scene.ground.as_ref().map_or(0.0, |g| g.height);
    let (radius, mass) = (0.3, 400.0);
    let mut w = ThresholdWorld::new(&gpu, &scene, radius, mass).expect("threshold world");
    let start = w.chunk_centres(&gpu);
    // The block is the highest chunk, the columns the three at mid height.
    let top = (0..start.len()).max_by(|&a, &b| height(start[a].0).total_cmp(&height(start[b].0))).unwrap();
    let columns: Vec<usize> = (0..start.len()).filter(|&k| (height(start[k].0) - ground - 1.5).abs() < 0.1).collect();
    assert_eq!(columns.len(), 3, "three columns");
    // Each shot horizontal, 2.3 m off its column's centre, at 40 m/s.
    let side = if up[2].abs() > 0.5 { [0.0, 1.0, 0.0] } else { [0.0, 0.0, 1.0] };
    for &k in &columns {
        let c = start[k].0;
        let desc = ImpactorDesc {
            name: format!("shot{k}"),
            shape: ImpactorShape::Sphere { radius },
            mass,
            position: [c[0] - 2.3 * side[0], c[1] - 2.3 * side[1], c[2] - 2.3 * side[2]],
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: [40.0 * side[0], 40.0 * side[1], 40.0 * side[2]],
            angular_velocity: [0.0; 3],
            material: "projectile".into(),
            crush: None,
        };
        assert!(w.launch(&gpu, &desc), "shot launched");
    }
    let mut lowest = f64::INFINITY;
    for _ in 0..400 {
        w.step_frame(&gpu).expect("frame");
        let c = w.chunk_centres(&gpu);
        lowest = lowest.min(height(c[top].0));
    }
    let end = w.chunk_centres(&gpu);
    let (now, speed) = (height(end[top].0) - ground, end[top].1);
    eprintln!("the block's centre {} m above the ground at the end (lowest {}), speed {speed}", now, lowest - ground);
    // Down off its columns (its centre was 4.5 m up), onto the footings (0.5 m) and the
    // ground: its centre at least its smallest half extent (1.4 m) above the ground in every
    // frame, and at rest.
    assert!(now < 3.0, "the block came down ({now} m)");
    assert!(lowest - ground > 1.35, "the block went into the ground: its centre {} m above it", lowest - ground);
    assert!(speed < 0.5, "the block rests ({speed})");
}

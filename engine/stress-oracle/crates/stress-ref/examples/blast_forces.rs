//! Bond forces of a scene pack's gravity equilibrium (the static solve every frame of the
//! Blast-equivalent configuration does), fracture off, for the force-level comparison
//! with Blast's quasi-static solve (engine/docs/stress-fracture-blast-ladder.md,
//! `scripts/blast_force_agreement.py`). One JSON line: each bond's chunks, area, normal,
//! force and moment about its patch centroid (world, Z-up, tension-positive on `a`).
//!
//! `--blast-inputs`: the pack's own node masses and supports (nodes of zero mass) instead
//! of the import's calibrated densities and supports at grade, so the two solvers see the
//! same loads and boundary conditions and differ only in how they solve.
//!
//!   cargo run --release -p stress-ref --example blast_forces -- packs/house-1story.json [--blast-inputs]

fn main() {
    let path = std::env::args().nth(1).expect("pack path");
    let p = std::path::Path::new(&path);
    let name = p.file_stem().unwrap().to_string_lossy().to_string();
    let mut scene = stress_ref::scene_pack::import(p, &name, "structure").expect("pack");
    if std::env::args().any(|a| a == "--blast-inputs") {
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let nodes = json["scenario"]["nodes"].as_array().unwrap().clone();
        let solver = stress_ref::solver::ReferenceSolver::new(&scene);
        let volumes: Vec<f64> = solver.structures[0].chunks.iter().map(|c| c.volume()).collect();
        let body = &mut scene.bodies[0];
        for (i, c) in body.chunks.iter_mut().enumerate() {
            let mass = nodes[i]["mass"].as_f64().unwrap_or(0.0);
            c.support = if mass <= 0.0 { stress_ref::scene::Support::Fixed } else { stress_ref::scene::Support::None };
            if mass > 0.0 {
                // A chunk material of its own whose density gives the pack's mass.
                let mut m = scene.materials[&c.material].clone();
                m.density = mass / volumes[i];
                let key = format!("{}#{i}", c.material);
                scene.materials.insert(key.clone(), m);
                c.material = key;
            }
        }
    }
    scene.sim.fracture = false;
    scene.sim.gravity_prestress = true;
    let t = std::time::Instant::now();
    let world = stress_ref::world::World::new(&scene);
    let solve_s = t.elapsed().as_secs_f64();
    let m = &world.solver;
    let st = &m.structures[0];
    let bonds: Vec<_> = m.bonds[0]
        .iter()
        .map(|b| {
            let g = &b.geometry;
            let f = g.to_body(b.force.lin);
            let mo = g.to_body(b.force.ang);
            serde_json::json!([g.a, g.b, g.area, g.normal.to_array(), f.to_array(), mo.to_array()])
        })
        .collect();
    let supports = st.chunks.iter().filter(|c| c.support != stress_ref::scene::Support::None).count();
    let mass: f64 = st.chunks.iter().map(|c| c.mass).sum();
    println!("{}", serde_json::json!({"pack": name, "chunks": st.chunks.len(), "supports": supports, "mass": mass, "solve_s": solve_s, "bonds": bonds}));
}

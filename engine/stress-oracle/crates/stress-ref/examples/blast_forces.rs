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
//!   cargo run --release -p stress-ref --example blast_forces -- packs/house-1story.json [--blast-inputs] [--blast-model]

fn main() {
    let path = std::env::args().nth(1).expect("pack path");
    let p = std::path::Path::new(&path);
    let name = p.file_stem().unwrap().to_string_lossy().to_string();
    let has = |flag: &str| std::env::args().any(|a| a == flag);
    // `--blast-inputs`: the pack's masses and supports (mass-0 nodes); `--blast-model`:
    // Blast's isotropic stiffness and capped stress gain (`Methods::blast_model`) with the
    // pack's own moduli and strengths.
    let blast_model = has("--blast-model");
    let options = stress_ref::scene_pack::ImportOptions { authored_strengths: blast_model, authored_fatal: blast_model, pack_masses: has("--blast-inputs"), pack_modulus: blast_model, blast_authoring: false };
    let mut scene = stress_ref::scene_pack::import_with(p, &name, "structure", options).expect("pack");
    scene.sim.methods.blast_model = blast_model;
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

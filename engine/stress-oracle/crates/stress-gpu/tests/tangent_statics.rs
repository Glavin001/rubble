//! `sim.methods.tangent_statics` on the GPU: a static solve the secant iteration leaves
//! unconverged after `SECANT_STEPS` is solved again by the oracle's globalised tangent
//! re-solve (stress-ref statics.rs `equilibrate`). The GPU's static solve
//! (`GpuSolver::equilibrate`, `island_statics`) must reach the outcome of the oracle's
//! `equilibrate_or_keep` under the scene's methods: converged or not alike, and a
//! converged equilibrium within the gate's f32-against-f64 tolerance (1e-3 of the range,
//! gate.rs `difference`; as statics_vs_reference.rs).
//!
//! The clusters are the free fragments captured from house-1story rammed
//! (`tests/data/house_ram_settle_*.json`, see statics_vs_reference.rs): the secant
//! iteration diverges on both, in f32 on the GPU and in f64 in the oracle, while the
//! oracle's tangent re-solve converges. They run with the proven methods minus the layer
//! contact (the GPU's penalty contact), the method set `STRESS_METHODS=-layer_contact`
//! gives every scene.

use stress_gpu::gpu::Gpu;
use stress_gpu::solver::GpuSolver;
use stress_ref::math::Vec3;
use stress_ref::scene::{Methods, Scene, Support};
use stress_ref::solver::{ChunkLoads, ReferenceSolver};
use stress_ref::statics::StaticOptions;

fn fixture(house: &Scene, data: &serde_json::Value, methods: Methods) -> (ReferenceSolver, ChunkLoads) {
    let originals: Vec<usize> = data["chunks"].as_array().unwrap().iter().map(|c| c["index"].as_u64().unwrap() as usize).collect();
    let local = |c: usize| originals.iter().position(|&o| o == c).unwrap();
    let mut scene = house.clone();
    scene.impactors.clear();
    scene.sim.methods = methods;
    let mut body = house.bodies[0].clone();
    body.chunks = originals.iter().map(|&c| stress_ref::scene::ChunkDesc { support: Support::None, parent: None, ..house.bodies[0].chunks[c].clone() }).collect();
    let sources: Vec<usize> = data["bonds"].as_array().unwrap().iter().map(|b| b["source"].as_u64().unwrap() as usize).collect();
    body.bonds = sources
        .iter()
        .map(|&b| {
            let d = &house.bodies[0].bonds[b];
            stress_ref::scene::BondDesc { a: local(d.a), b: local(d.b), ..d.clone() }
        })
        .collect();
    scene.bodies = vec![body];
    let mut reference = ReferenceSolver::new(&scene);
    let v3 = |v: &serde_json::Value| Vec3::new(v[0].as_f64().unwrap(), v[1].as_f64().unwrap(), v[2].as_f64().unwrap());
    let p = &data["pose"];
    let q = |k: usize| p[1][k].as_f64().unwrap();
    reference.clusters[0].pose = stress_ref::math::Pose { position: v3(&p[0]), rotation: stress_ref::math::Quat { x: q(0), y: q(1), z: q(2), w: q(3) } };
    let mut loads = ChunkLoads::new(&reference);
    for (k, c) in data["chunks"].as_array().unwrap().iter().enumerate() {
        let st = &mut reference.chunks[0][k];
        st.u = v3(&c["u"]);
        st.th = v3(&c["th"]);
        st.inertia_scale = c["inertia_scale"].as_f64().unwrap();
        loads.force[0][k] = v3(&c["force"]);
        loads.torque[0][k] = v3(&c["torque"]);
    }
    for (i, b) in data["bonds"].as_array().unwrap().iter().enumerate() {
        let mut j = b["joint"].clone();
        let unreached: Vec<&str> = ["ductility", "ductility_c"].into_iter().filter(|k| j[*k].is_null()).collect();
        for k in &unreached {
            j[*k] = serde_json::json!(0.0);
        }
        let mut joint: stress_ref::joint::JointState = serde_json::from_value(j).expect("joint state");
        for k in &unreached {
            if *k == "ductility" {
                joint.ductility = f64::INFINITY;
            } else {
                joint.ductility_c = f64::INFINITY;
            }
        }
        let bi = reference.bonds[0].iter().position(|rb| rb.source == i).unwrap();
        reference.bonds[0][bi].joint = joint;
    }
    reference.refresh_bond_forces(0);
    (reference, loads)
}

/// The proven methods without the layer contact (`STRESS_METHODS=-layer_contact`).
fn proven_minus_layer() -> Methods {
    Methods { layer_contact: false, ..Methods::proven() }
}

#[test]
fn tangent_statics_settle_matches_the_oracle() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let pack = dir.join("../../packs/house-1story.json");
    if !pack.exists() {
        eprintln!("{} not found; skipping", pack.display());
        return;
    }
    let house = stress_ref::scene_pack::import(&pack, "house-1story", "structure").expect("house-1story");
    let mut failures = Vec::new();
    for name in ["house_ram_settle_frame62", "house_ram_settle_frame135"] {
        let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("tests/data/{name}.json"))).unwrap()).unwrap();
        // The case exercises the re-solve: the oracle's secant iteration alone (tangent
        // statics off) finds no equilibrium.
        let (mut secant_only, loads) = fixture(&house, &data, Methods { tangent_statics: false, ..proven_minus_layer() });
        let s = secant_only.equilibrate_or_keep(0, &loads, &StaticOptions::default());
        assert!(!s.converged, "{name}: the oracle's secant iteration converges (residual {:.3e}): no re-solve to test", s.residual);

        let (mut reference, loads) = fixture(&house, &data, proven_minus_layer());
        let mut solver = GpuSolver::new(&gpu, reference.clone(), None, Vec::new()).expect("GPU solver");
        let t = std::time::Instant::now();
        let g = solver.equilibrate(&gpu, &[0], &loads)[0];
        let gpu_time = t.elapsed().as_secs_f64();
        let r = reference.equilibrate_or_keep(0, &loads, &StaticOptions::default());
        eprintln!(
            "{name}: GPU converged {} residual {:.3e} ({} Newton, {} CG, {:.1} ms) | oracle converged {} residual {:.3e} ({} Newton, {} CG); secant alone: residual {:.3e}",
            g.converged,
            g.residual,
            g.newton_iterations,
            g.cg_iterations,
            1e3 * gpu_time,
            r.converged,
            r.residual,
            r.newton_iterations,
            r.cg_iterations,
            s.residual
        );
        if g.converged != r.converged {
            failures.push(format!("{name}: GPU converged {} (residual {:.3e}), oracle {} (residual {:.3e})", g.converged, g.residual, r.converged, r.residual));
            continue;
        }
        // The equilibrium as the host holds it, and as the GPU holds it (downloaded).
        let close = |label: &str, m: &ReferenceSolver| -> Option<String> {
            let range = reference.chunks[0].iter().map(|s| s.u.norm().max(s.th.norm())).fold(0.0, f64::max);
            let du = reference.chunks[0].iter().zip(&m.chunks[0]).map(|(a, b)| (a.u - b.u).norm().max((a.th - b.th).norm())).fold(0.0, f64::max);
            let frange = reference.bonds[0].iter().map(|b| b.force.lin.norm().max(b.force.ang.norm())).fold(0.0, f64::max);
            let df = reference.bonds[0].iter().zip(&m.bonds[0]).map(|(a, b)| (a.force.lin - b.force.lin).norm().max((a.force.ang - b.force.ang).norm())).fold(0.0, f64::max);
            eprintln!("  {label}: displacement error {:.1e} of {range:.3e}, bond force error {:.1e} of {frange:.3e}", du / range.max(1e-300), df / frange.max(1e-300));
            (du > 1e-3 * range || df > 1e-3 * frange).then(|| format!("{name} ({label}): displacement error {du:.3e} of {range:.3e}, bond force error {df:.3e} of {frange:.3e}"))
        };
        failures.extend(close("host", &solver.mirror));
        solver.download(&gpu);
        failures.extend(close("GPU", &solver.mirror));
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}


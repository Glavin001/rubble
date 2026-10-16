//! Static equilibrium on the GPU (`GpuSolver::equilibrate`) against stress-ref's
//! `equilibrate` in f64: anchored structures under gravity, and a free slab under a
//! point load (inertia relief). The GPU solves in f32 to a relative residual of 1e-5;
//! displacements are compared against their own range, reactions against the weight.

use stress_gpu::gpu::Gpu;
use stress_gpu::scenes;
use stress_gpu::solver::GpuSolver;
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, Support};
use stress_ref::solver::{ChunkLoads, ReferenceSolver};
use stress_ref::statics::StaticOptions;

fn catalogue(name: &str) -> Scene {
    stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()).find(|s| s.name == name).expect("scene")
}

/// Solve every cluster with bonds both ways from the unloaded state; worst displacement
/// error (relative to the largest displacement) and reaction error (relative to the
/// total weight).
fn compare(name: &str, scene: &Scene, loads_of: impl Fn(&ReferenceSolver) -> ChunkLoads) -> (f64, f64) {
    let gpu = Gpu::new().expect("GPU");
    let mut reference = ReferenceSolver::new(scene);
    let loads = loads_of(&reference);
    let mut solver = GpuSolver::new(&gpu, reference.clone(), None, Vec::new()).expect("GPU solver");
    let clusters: Vec<usize> = (0..reference.clusters.len()).filter(|&ci| !reference.clusters[ci].bonds.is_empty()).collect();
    let t = std::time::Instant::now();
    let results = solver.equilibrate(&gpu, &clusters, &loads);
    let gpu_time = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let mut ref_reports = Vec::new();
    for &ci in &clusters {
        ref_reports.push(reference.equilibrate(ci, &loads, &StaticOptions::default()));
    }
    let ref_time = t.elapsed().as_secs_f64();
    let m = &solver.mirror;
    let (mut scale, mut err) = (0.0f64, 0.0f64);
    let (mut weight, mut rerr) = (0.0f64, 0.0f64);
    for s in 0..reference.structures.len() {
        for c in 0..reference.structures[s].chunks.len() {
            let (a, b) = (&reference.chunks[s][c], &m.chunks[s][c]);
            scale = scale.max(a.u.norm());
            err = err.max((a.u - b.u).norm());
            weight += reference.structures[s].chunks[c].mass * reference.config.gravity.norm();
            rerr += (a.reaction.0 - b.reaction.0).norm();
        }
    }
    let (du, dr) = (err / scale.max(1e-30), rerr / weight.max(1e-30));
    for (k, (g, r)) in results.iter().zip(&ref_reports).enumerate() {
        eprintln!(
            "{name} cluster {}: gpu converged {} residual {:.1e} ({} Newton, {} CG) in {gpu_time:.3} s | reference converged {} residual {:.1e} ({} Newton, {} CG) in {ref_time:.3} s",
            clusters[k], g.converged, g.residual, g.newton_iterations, g.cg_iterations, r.converged, r.residual, r.newton_iterations, r.cg_iterations
        );
    }
    eprintln!("{name}: displacement error {du:.1e} of the largest ({scale:.3e} m); reaction error {dr:.1e} of the weight");
    for (g, r) in results.iter().zip(&ref_reports) {
        assert_eq!(g.converged, r.converged, "{name}: convergence differs");
    }
    (du, dr)
}

#[test]
fn equilibrium_matches_the_reference() {
    let gravity_only = |m: &ReferenceSolver| ChunkLoads::new(m);
    let mut cases: Vec<(&str, Scene)> = vec![
        ("slab 12x6x2", scenes::slab([12, 6, 2])),
        ("slab 40x20x4", scenes::slab([40, 20, 4])),
        ("b2_cantilever", catalogue("b2_cantilever")),
        ("b8_frame_gradual", catalogue("b8_frame_gradual")),
        ("s_floor_static", catalogue("s_floor_static")),
    ];
    for (name, scene) in cases.drain(..) {
        let (du, dr) = compare(name, &scene, gravity_only);
        assert!(du < 1e-3, "{name}: displacement error {du:.1e}");
        assert!(dr < 1e-4, "{name}: reaction error {dr:.1e}");
    }
    // A free slab (no supports) pushed at one corner: inertia relief.
    let mut free = scenes::slab([12, 6, 2]);
    for c in free.bodies[0].chunks.iter_mut() {
        c.support = Support::None;
    }
    let push = |m: &ReferenceSolver| {
        let mut l = ChunkLoads::new(m);
        l.force[0][0] = Vec3::new(0.0, 0.0, 5.0e4);
        l
    };
    let (du, _) = compare("free slab", &free, push);
    assert!(du < 1e-3, "free slab: displacement error {du:.1e}");
}

/// The settles that blew up house-1story rammed (GPU world coupled to PhysX GPU, the
/// realtime_peak settings, `STRESS_GPU_FIXED_HORIZON=1`): free fragments of 4 chunks
/// (550-640 kg; a 24 mm sliver bonded to three blocks, two of whose mutual bonds are fully
/// damaged, i.e. contact only) nearly at rest, under the loads `settle_quiet_clusters`
/// recorded (gravity and one-sided contact reactions: net force and moment nonzero). Each
/// one's state, loads and bond histories are a `tests/data/house_ram_settle_*.json`
/// (captured from that run: frame 62, where the GPU's residual overflowed, and frame 135,
/// reached once that was rejected, where it diverged to a finite 5e15); the chunks and
/// bonds are house-1story's.
///
/// The reference's static solve (statics.rs `equilibrate_or_keep`, f64) finds no
/// equilibrium for either (its secant iteration diverges) and keeps the state, so the
/// cluster stays in the explicit solve. The GPU's must agree: "converged" only with a
/// finite residual and the reference's equilibrium, else the state kept as it was; and a
/// free cluster's solution carries no rigid-mode component (statics.rs
/// `project_displacement`: zero mass-weighted translation and rotation) beyond f32
/// rounding of the sums defining it.
#[test]
fn free_cluster_settle_matches_the_reference() {
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
    for fixture in ["house_ram_settle_frame62", "house_ram_settle_frame135"] {
        let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("tests/data/{fixture}.json"))).unwrap()).unwrap();
        eprint!("{fixture}: ");
        if let Err(e) = settle_matches(&gpu, &house, &data) {
            failures.push(format!("{fixture}: {e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// One captured settle (`free_cluster_settle_matches_the_reference`).
fn settle_matches(gpu: &Gpu, house: &Scene, data: &serde_json::Value) -> Result<(), String> {
    // The fragment as a structure of its own: its chunks (free) and the bonds among them.
    let originals: Vec<usize> = data["chunks"].as_array().unwrap().iter().map(|c| c["index"].as_u64().unwrap() as usize).collect();
    let local = |c: usize| originals.iter().position(|&o| o == c).unwrap();
    let mut scene = house.clone();
    scene.impactors.clear();
    scene.sim.methods = stress_ref::scene::Methods::established();
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
    assert_eq!(reference.clusters.len(), 1, "the fragment is one cluster");
    assert!(!reference.clusters[0].anchored);
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
        // JSON has no infinity: serde_json wrote the ductilities not yet reached (inf)
        // as null.
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
    let start: Vec<(Vec3, Vec3)> = reference.chunks[0].iter().map(|s| (s.u, s.th)).collect();

    let mut solver = GpuSolver::new(gpu, reference.clone(), None, Vec::new()).expect("GPU solver");
    let t = std::time::Instant::now();
    let g = solver.equilibrate(gpu, &[0], &loads)[0];
    let gpu_time = t.elapsed().as_secs_f64();
    let r = reference.equilibrate_or_keep(0, &loads, &StaticOptions::default());
    let m = &solver.mirror;
    let max_u = m.chunks[0].iter().map(|s| s.u.norm()).fold(0.0, f64::max);
    let max_th = m.chunks[0].iter().map(|s| s.th.norm()).fold(0.0, f64::max);
    eprintln!(
        "free fragment: GPU converged {} residual {:.3e} ({} Newton, {} CG, {gpu_time:.3} s), max |u| {max_u:.3e} m, max |th| {max_th:.3e} rad | reference converged {} residual {:.3e} ({} Newton, {} CG)",
        g.converged, g.residual, g.newton_iterations, g.cg_iterations, r.converged, r.residual, r.newton_iterations, r.cg_iterations
    );
    // (c) Convergence as the reference's, and only with a finite residual.
    if g.converged && !g.residual.is_finite() {
        return Err(format!("GPU reports convergence with residual {} (max |u| {max_u:.3e} m, |th| {max_th:.3e} rad)", g.residual));
    }
    if g.converged != r.converged {
        return Err(format!("GPU converged {} (residual {:.3e}, max |u| {max_u:.3e} m, |th| {max_th:.3e} rad), reference {} (residual {:.3e})", g.converged, g.residual, r.converged, r.residual));
    }
    if !g.converged {
        // The state is kept (statics.rs `equilibrate_or_keep`).
        for (k, s) in m.chunks[0].iter().enumerate() {
            if (s.u, s.th) != start[k] {
                return Err(format!("chunk {k}: an unconverged solve changed the state"));
            }
        }
        return Ok(());
    }
    // (b) The equilibrium is the reference's (as `equilibrium_matches_the_reference`).
    let scale = reference.chunks[0].iter().map(|s| s.u.norm()).fold(0.0, f64::max);
    let err = reference.chunks[0].iter().zip(&m.chunks[0]).map(|(a, b)| (a.u - b.u).norm()).fold(0.0, f64::max);
    if err > 1e-3 * scale {
        return Err(format!("displacement error {err:.3e} of {scale:.3e}"));
    }
    // (a) No rigid-mode component: the mass-weighted translation and the inertia-weighted
    // rotation sums vanish to f32 rounding of their sums (gamma_n of the absolute sums).
    let cl = &m.clusters[0];
    let st = &m.structures[0];
    let (mut p, mut l, mut p_abs, mut l_abs) = (Vec3::ZERO, Vec3::ZERO, 0.0, 0.0);
    for &c in &cl.chunks {
        let ch = &st.chunks[c];
        let s = &m.chunks[0][c];
        let r = ch.center - cl.com;
        p += s.u * ch.mass;
        l += r.cross(s.u) * ch.mass + ch.inertia * s.th;
        p_abs += s.u.norm() * ch.mass;
        l_abs += r.norm() * s.u.norm() * ch.mass + ch.inertia.m.iter().map(|row| row.iter().map(|x| x.abs()).sum::<f64>()).fold(0.0, f64::max) * s.th.norm();
    }
    let n = cl.chunks.len() as f64 + 4.0;
    let u32_round = f32::EPSILON as f64 / 2.0;
    let gamma = n * u32_round / (1.0 - n * u32_round);
    if p.norm() > gamma * p_abs || l.norm() > gamma * l_abs {
        return Err(format!("rigid-mode component: translation {:.3e} (bound {:.3e}), rotation {:.3e} (bound {:.3e})", p.norm(), gamma * p_abs, l.norm(), gamma * l_abs));
    }
    Ok(())
}

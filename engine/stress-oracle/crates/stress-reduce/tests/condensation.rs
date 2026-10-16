//! The assembled network against the oracle's own static solve, and condensation
//! against the full solve (exact for statics), on scene packs that stand intact at rest.

use stress_reduce::{interior_of, pcg, rcb, top_eigen, Network, Reduced};
use stress_ref::math::Vec3;
use stress_ref::world::World;

fn pack(name: &str) -> Option<stress_ref::scene::Scene> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../packs/{name}.json"));
    if !path.exists() {
        eprintln!("{} not found; skipping", path.display());
        return None;
    }
    Some(stress_ref::scene_pack::import(&path, name, "structure").unwrap())
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let n: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    d / n
}

/// Packs whose bonds all stay below strength at rest (rest_utilization): their gravity
/// equilibrium is the linear one.
const INTACT_AT_REST: [&str; 3] = ["rig-portal", "comp-wall-bay", "house-1story"];

#[test]
fn assembled_network_reproduces_the_oracles_gravity_equilibrium() {
    for name in INTACT_AT_REST {
        let Some(mut scene) = pack(name) else { continue };
        scene.sim.gravity_prestress = true;
        let world = World::new(&scene);
        let m = &world.solver;
        let ci = (0..m.clusters.len()).find(|&c| m.clusters[c].anchored && !m.clusters[c].bonds.is_empty()).unwrap();
        let net = Network::of_cluster(m, ci, Vec3::from_array(scene.gravity));
        let (u, _) = pcg(&net.k, &net.gravity, 1e-13, 200_000);
        let s = m.clusters[ci].structure;
        let mut oracle = vec![0.0; net.n];
        for (i, &c) in net.chunks.iter().enumerate() {
            let st = &m.chunks[s][c];
            let v = [st.u.x, st.u.y, st.u.z, st.th.x, st.th.y, st.th.z];
            for d in 0..6 {
                if let Some(k) = net.dof[i][d] {
                    oracle[k] = v[d];
                }
            }
        }
        let e = rel(&u, &oracle);
        println!("{name}: {} dofs, relative difference to the oracle's prestress {e:.2e}", net.n);
        assert!(e < 1e-7, "{name}: {e:.3e}");
    }
}

#[test]
fn condensation_is_exact_for_statics() {
    for name in INTACT_AT_REST {
        let Some(scene) = pack(name) else { continue };
        let m = stress_ref::solver::ReferenceSolver::new(&scene);
        let ci = (0..m.clusters.len()).find(|&c| m.clusters[c].anchored && !m.clusters[c].bonds.is_empty()).unwrap();
        let net = Network::of_cluster(&m, ci, Vec3::from_array(scene.gravity));
        let (full, _) = pcg(&net.k, &net.gravity, 1e-14, 400_000);
        for target in [4, 16, 64] {
            let part = rcb(&net.centers, target);
            let red = Reduced::new(&net, &interior_of(&net, &part));
            let (ur, _) = pcg(&red.k, &red.load(&net.gravity), 1e-14, 400_000);
            let u = red.recover(&ur, &net.gravity);
            let e = rel(&u, &full);
            println!("{name} parts of {target}: {} of {} dofs kept, difference {e:.2e}", red.nr, net.n);
            assert!(e < 1e-9, "{name} parts of {target}: {e:.3e}");
            // A Ritz projection: the reduced top frequency cannot exceed the full one.
            // Checked where power iteration converges to round-off (the small packs; on
            // the house 300 iterations leave both estimates 1e-3 below their limits).
            if net.n <= 200 {
                let (full_top, _) = top_eigen(&net.k, &net.m, 20_000);
                let (red_top, _) = top_eigen(&red.k, &red.m, 20_000);
                assert!(red_top <= full_top * (1.0 + 1e-9), "{name}: reduced {red_top:.6e} > full {full_top:.6e}");
            }
        }
    }
}

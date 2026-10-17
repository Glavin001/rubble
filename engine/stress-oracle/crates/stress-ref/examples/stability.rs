//! Stability of the explicit step: every catalogue scene (or the scene files given),
//! without gravity, fracture or loads, its hidden displacements perturbed at random by
//! 1e-6 of the chunk size, run for `SUBSTEPS` substeps at Courant fractions 0.9 to 2.
//! A valid step bound is stable at 0.9; where it blows up shows how much the bound
//! leaves on the table (a tight bound blows up just above 1).
//!
//! ```text
//! cargo run --release -p stress-ref --example stability [scene.json ...]
//! STRESS_METHODS=scaled_step_bound cargo run --release -p stress-ref --example stability
//! ```

use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, SolveMode};
use stress_ref::world::World;

const SUBSTEPS: u64 = 20_000;
const SAFETIES: [f64; 5] = [0.9, 1.0, 1.1, 1.5, 2.0];

fn quiet(scene: &Scene, safety: f64) -> Scene {
    let mut s = scene.clone();
    s.gravity = [0.0; 3];
    s.sim.fracture = false;
    s.sim.gravity_prestress = false;
    s.sim.courant_safety = safety;
    s.sim.solve_mode = SolveMode::Explicit;
    s.loads.clear();
    s.events.clear();
    s.impactors.clear();
    s.probes.clear();
    s.ground = None;
    s
}

/// Largest ratio of the deformation energy (elastic + kinetic) to its initial value.
fn growth(scene: &Scene) -> (f64, u64) {
    let mut w = World::new(scene);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for (s, st) in w.solver.structures.iter().enumerate() {
        for (c, ch) in st.chunks.iter().enumerate() {
            let h = ch.half_extents.norm();
            let cs = &mut w.solver.chunks[s][c];
            cs.u += Vec3::new(rnd(), rnd(), rnd()) * (1e-6 * h);
            cs.th += Vec3::new(rnd(), rnd(), rnd()) * 1e-6;
        }
    }
    // The bonds' stored energy is what their last evaluation found: evaluated at the
    // perturbed state, it is the reference before any step. (Taken after the first
    // frame, a run that blew up within that frame read as stable.)
    for ci in 0..w.solver.clusters.len() {
        w.solver.refresh_bond_forces(ci);
    }
    let e0 = (w.solver.elastic_energy() + w.solver.kinetic_energy()).max(1e-300);
    let start = w.solver.substeps;
    let mut worst: f64 = 1.0;
    while w.solver.substeps - start < SUBSTEPS {
        w.step_frame();
        let e = w.solver.elastic_energy() + w.solver.kinetic_energy();
        if !e.is_finite() {
            return (f64::INFINITY, w.solver.substeps - start);
        }
        worst = worst.max(e / e0);
        if std::env::var("TRACE").is_ok() {
            println!("  substep {} e {e:.6e} ke {:.3e} el {:.3e} e0 {e0:.3e}", w.solver.substeps - start, w.solver.kinetic_energy(), w.solver.elastic_energy());
        }
        if worst > 1e12 {
            break;
        }
    }
    (worst, w.solver.substeps - start)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scenes: Vec<Scene> = if args.is_empty() {
        stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()).collect()
    } else {
        args.iter().map(|p| Scene::load(std::path::Path::new(p)).expect("scene")).collect()
    };
    println!("{:30} {}", "scene", SAFETIES.iter().map(|s| format!("{s:>10}")).collect::<String>());
    for scene in scenes.iter().filter(|s| s.bodies.iter().any(|b| !b.bonds.is_empty())) {
        let row: Vec<String> = SAFETIES
            .iter()
            .map(|&s| {
                let (g, n) = growth(&quiet(scene, s));
                if g > 1e6 {
                    format!("{:>10}", format!("BLOW@{n}"))
                } else {
                    format!("{g:>10.3}")
                }
            })
            .collect();
        println!("{:30} {}", scene.name, row.join(""));
    }
}

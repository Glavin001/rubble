//! Reproducibility and solver convergence: repeated runs are bit-identical, and the
//! static solve stops on an explicit residual criterion that holds for loads in any
//! direction (not a force tolerance that leaves sideways loads off by percents).

mod common;

use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::world::World;

/// The same fracturing scene twice, in the explicit and the implicit solve: every
/// probe sample, chunk state and value is bit-identical (single-threaded, ordered
/// iteration, seeded Weibull strengths).
#[test]
fn repeated_runs_are_bit_identical() {
    for mode in [SolveMode::Explicit, SolveMode::Implicit] {
        let mut scene = pressure_panel_weibull(300e3, "repeat");
        scene.sim.solve_mode = mode;
        scene.sim.duration = 0.02;
        let a = serde_json::to_string(&common::run(&scene)).unwrap();
        let b = serde_json::to_string(&common::run(&scene)).unwrap();
        assert!(a == b, "{mode:?}: runs differ");
        assert!(a.contains("\"any_bond_broken\":true"), "{mode:?}: the scene should fracture");
    }
}

/// A cantilever loaded at the tip in both transverse directions at once: the static
/// solve converges to its relative residual criterion and both deflections equal the
/// exact discrete solution `P (a^3/3 - a h^2/12) / EI + P a / (G A)` to 1e-6.
#[test]
fn static_solve_converges_for_loads_in_every_direction() {
    let n = 20;
    let mut s = cantilever(n, "two_way");
    s.loads.clear();
    let (pz, py) = (30.0, 45.0);
    for (dir, p) in [([0.0, 0.0, -1.0], pz), ([0.0, 1.0, 0.0], py)] {
        s.loads.push(LoadDesc::PointForce { body: "beam".into(), chunk: n, direction: dir, magnitude: TimeFunction::Constant { value: p }, point: None });
    }
    let w = World::new(&s);
    let tip = w.solver.chunk_position(0, n) - Vec3::from_array(s.bodies[0].chunks[n].center);
    let m = s.materials["beam"].clone();
    let (h, a, i, area) = (0.1, 2.0, 0.1f64.powi(4) / 12.0, 0.01);
    let exact = |p: f64| p * (a * a * a / 3.0 - a * h * h / 12.0) / (m.youngs_modulus * i) + p * a / (m.shear_modulus() * area);
    assert!(common::rel(-tip.z, exact(pz)) < 1e-6, "z {} vs {}", -tip.z, exact(pz));
    assert!(common::rel(tip.y, exact(py)) < 1e-6, "y {} vs {}", tip.y, exact(py));
    assert!(tip.x.abs() < 1e-6 * tip.norm(), "no axial drift: {}", tip.x);
}

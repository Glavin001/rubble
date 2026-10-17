//! Stability of the explicit contact step on the densest contact network: an `n^3`
//! lattice of equal unbonded cubes, every face in elastic-layer contact (pressed together
//! by a tiny overlap, the outer shell held), no gravity, no contact damping (restitution
//! 1), small random velocities on the inner cubes. Each run reports the largest ratio of
//! the kinetic energy to its first-frame value at each Courant safety: a valid contact
//! step bound stays bounded at 0.9.
//!
//! ```text
//! cargo run --release -p stress-ref --example contact_lattice [material] [n] [cube|layers|rotated|hull|hull_rotated]
//! ```

use stress_ref::builders::{body, chunk, new_scene};
use stress_ref::math::Vec3;
use stress_ref::scene::{Scene, SolveMode, Support};
use stress_ref::world::World;

const SUBSTEPS: u64 = 20_000;
const SAFETIES: [f64; 4] = [0.8, 0.9, 1.0, 1.1];

/// `variant`: `cube` (equal cubes, face to face), `layers` (cubes alternating with plates a
/// quarter as thick, so thin and thick chunks press face to face), `rotated` (cubes
/// turned +-10 degrees about z in a checkerboard, touching corner to face), `hull` and
/// `hull_rotated` (the same cubes as convex hulls, through the hull contact path).
fn lattice(material: &str, n: usize, safety: f64, variant: &str) -> Scene {
    let mut s = new_scene("contact_lattice", "dense lattice of unbonded cubes in face contact");
    s.gravity = [0.0; 3];
    s.materials.insert(material.into(), stress_ref::material::preset(material).expect("material"));
    let side = 0.25;
    let turn = 10f64.to_radians();
    let squeeze = 1.0 - 1e-5;
    let mut chunks = Vec::new();
    // Layer heights (z) for `layers`: side and side / 4 alternating.
    let thickness = |k: usize| if variant == "layers" && k % 2 == 1 { 0.25 * side } else { side };
    let mut z = 0.0;
    for k in 0..n {
        // Every contact the same overlap (an unequal one would push a plate harder on
        // one face than the other: no equilibrium to perturb).
        let tz = thickness(k);
        let zc = z + 0.5 * tz;
        z += tz - side * (1.0 - squeeze);
        for j in 0..n {
            for i in 0..n {
                let rotated = variant == "rotated" || variant == "hull_rotated";
                let pitch = if rotated { side * (turn.cos() + turn.sin()) * squeeze } else { side * squeeze };
                let mut c = chunk(Vec3::new(i as f64 * pitch, j as f64 * pitch, if rotated { k as f64 * side * squeeze } else { zc }), Vec3::new(0.5 * side, 0.5 * side, 0.5 * tz), material);
                if variant.starts_with("hull") {
                    let h = 0.5 * side;
                    c.hull = Some((0..8).map(|b| [if b & 1 == 0 { -h } else { h }, if b & 2 == 0 { -h } else { h }, if b & 4 == 0 { -0.5 * tz } else { 0.5 * tz }]).collect());
                }
                if rotated {
                    let a = if (i + j) % 2 == 0 { turn } else { -turn };
                    c.orientation = [(0.5 * a).cos(), 0.0, 0.0, (0.5 * a).sin()];
                }
                if [i, j, k].iter().any(|&x| x == 0 || x == n - 1) {
                    c.support = Support::Fixed;
                }
                chunks.push(c);
            }
        }
    }
    s.bodies.push(body("lattice", chunks, Vec::new()));
    s.sim.solve_mode = SolveMode::Explicit;
    s.sim.fracture = false;
    s.sim.gravity_prestress = false;
    s.sim.contact_restitution = 1.0;
    s.sim.courant_safety = safety;
    s.sim.duration = 1e9;
    s
}

fn growth(scene: &Scene) -> (f64, u64) {
    let mut w = World::new(scene);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for cl in w.solver.clusters.iter_mut() {
        if !cl.anchored {
            cl.velocity = Vec3::new(rnd(), rnd(), rnd()) * 1e-4;
            cl.angular_velocity = Vec3::new(rnd(), rnd(), rnd()) * 1e-4;
        }
    }
    // The reference is the kinetic energy given (not after a frame, which could already
    // have blown up).
    let e0 = w.solver.kinetic_energy().max(1e-300);
    let start = w.solver.substeps;
    let mut worst: f64 = 1.0;
    while w.solver.substeps - start < SUBSTEPS {
        w.step_frame();
        let e = w.solver.kinetic_energy();
        if !e.is_finite() {
            return (f64::INFINITY, w.solver.substeps - start);
        }
        worst = worst.max(e / e0);
        if worst > 1e12 {
            break;
        }
    }
    (worst, w.solver.substeps - start)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let material = args.first().map(String::as_str).unwrap_or("brick");
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let variant = args.get(2).map(String::as_str).unwrap_or("cube");
    let row: Vec<String> = SAFETIES
        .iter()
        .map(|&s| {
            let (g, k) = growth(&lattice(material, n, s, variant));
            if g > 1e6 {
                format!("{:>12}", format!("BLOW@{k}"))
            } else {
                format!("{g:>12.3}")
            }
        })
        .collect();
    println!("{material} n={n} {variant}: kinetic energy growth at safety {:?}: {}", SAFETIES, row.join(""));
}

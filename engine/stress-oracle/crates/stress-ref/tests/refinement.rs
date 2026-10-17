//! Two scales and multi-level pre-fracture: coarse chunks refine into finer ones on
//! demand, consistently with the coarse model and without changing momentum.

mod common;

use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::solver::{ChunkLoads, ReferenceSolver, SolverEvent};
use stress_ref::statics::StaticOptions;
use stress_ref::world::World;

/// A 0.4 x 0.4 x 2.0 m column of 0.2 m coarse chunks, each pre-fractured into eight
/// 0.1 m children, fixed at the base.
fn two_level_column() -> Scene {
    let mut s = new_scene("two_level", "two-level column");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let (mut chunks, bonds) = two_level_grid(Vec3::new(-0.2, -0.2, -0.2), [2, 2, 11], Vec3::splat(0.2), "concrete");
    for c in &mut chunks {
        if c.center[2] < 0.0 {
            c.support = Support::Fixed;
        }
    }
    s.bodies.push(body("column", chunks, bonds));
    s
}

fn tip_deflection(solver: &mut ReferenceSolver, s: &Scene) -> f64 {
    let mut loads = ChunkLoads::new(solver);
    // Same total lateral load on the top layer, whichever level represents it.
    let top: Vec<usize> = (0..s.bodies[0].chunks.len())
        .filter(|&c| solver.chunks[0][c].active && s.bodies[0].chunks[c].center[2] > 1.8 - 1e-9)
        .collect();
    for &c in &top {
        loads.force[0][c] = Vec3::new(1e4 / top.len() as f64, 0.0, 0.0);
    }
    let r = solver.solve_static_all(&loads, &StaticOptions::default(), 0.0);
    assert!(r.converged);
    top.iter().map(|&c| solver.chunks[0][c].u.x).sum::<f64>() / top.len() as f64
}

#[test]
fn refined_structure_responds_like_the_coarse_one() {
    let s = two_level_column();
    let mut coarse = ReferenceSolver::new(&s);
    let d_coarse = tip_deflection(&mut coarse, &s);
    let mut fine = ReferenceSolver::new(&s);
    let coarse_ids: Vec<usize> = (0..s.bodies[0].chunks.len()).filter(|&c| s.bodies[0].chunks[c].level == 0).collect();
    for c in coarse_ids {
        fine.refine_chunk(0, c);
    }
    assert!(fine.chunks[0].iter().zip(&s.bodies[0].chunks).all(|(st, d)| st.active == (d.level == 1)));
    let d_fine = tip_deflection(&mut fine, &s);
    // Both discretisations approximate the same cantilever beam; refinement changes the
    // response by less than the 10% of benchmark 10.
    let e = s.materials["concrete"].youngs_modulus;
    let i = 0.4f64.powi(4) / 12.0;
    let beam = 1e4 * 2.0f64.powi(3) / (3.0 * e * i);
    println!("coarse {d_coarse:.4e} fine {d_fine:.4e} beam {beam:.4e}");
    assert!(common::rel(d_fine, d_coarse) < 0.10, "{d_fine} vs {d_coarse}");
    assert!(common::rel(d_coarse, beam) < 0.06 && common::rel(d_fine, beam) < 0.06);
}

#[test]
fn refinement_conserves_momentum_and_kinetic_energy() {
    let mut s = two_level_column();
    for c in &mut s.bodies[0].chunks {
        c.support = Support::None;
    }
    s.bodies[0].linear_velocity = [1.0, -2.0, 0.5];
    s.bodies[0].angular_velocity = [0.3, 0.2, -1.0];
    let mut solver = ReferenceSolver::new(&s);
    // Some hidden motion on a coarse chunk too.
    solver.chunks[0][7].v = Vec3::new(0.1, 0.0, -0.05);
    solver.chunks[0][7].w = Vec3::new(0.0, 2.0, 0.0);
    let momentum = |sv: &ReferenceSolver| {
        let mut p = Vec3::ZERO;
        for (c, ch) in sv.structures[0].chunks.iter().enumerate() {
            if sv.chunks[0][c].active {
                p += sv.chunk_velocity(0, c).0 * ch.mass;
            }
        }
        p
    };
    let (p0, e0) = (momentum(&solver), solver.kinetic_energy());
    assert!(solver.refine_chunk(0, 7));
    assert!(solver.events.iter().any(|e| matches!(e, SolverEvent::Refined { chunk: 7, .. })));
    let (p1, e1) = (momentum(&solver), solver.kinetic_energy());
    assert!((p1 - p0).norm() < 1e-12 * p0.norm(), "{p1:?} vs {p0:?}");
    assert!(common::rel(e1, e0) < 1e-12, "{e1} vs {e0}");
}

#[test]
fn chunks_refine_on_demand_near_an_impact_only() {
    let mut s = new_scene("two_level_wall", "two-level wall hit by a sphere");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let (mut chunks, bonds) = two_level_grid(Vec3::new(-1.0, 0.0, -0.2), [10, 2, 11], Vec3::splat(0.2), "concrete");
    for c in &mut chunks {
        if c.center[2] < 0.0 {
            c.support = Support::Fixed;
        }
    }
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.1 },
        mass: 20.0,
        position: [0.0, -0.101, 1.0],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 2.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = 0.02;
    s.sim.frame_dt = 1e-3;
    s.sim.refine_utilization = Some(0.6);
    let mut w = World::new(&s);
    w.run();
    let refined: Vec<usize> = w
        .solver
        .events
        .iter()
        .filter_map(|e| if let SolverEvent::Refined { chunk, .. } = e { Some(*chunk) } else { None })
        .collect();
    let coarse = s.bodies[0].chunks.iter().filter(|c| c.level == 0).count();
    println!("refined {} of {coarse} coarse chunks", refined.len());
    assert!(!refined.is_empty(), "the impact should refine the struck chunks");
    assert!(refined.len() < coarse / 2, "refinement stays local");
    for c in refined {
        let x = Vec3::from_array(s.bodies[0].chunks[c].center);
        assert!((x - Vec3::new(0.0, 0.1, 1.0)).norm() < 1.2, "refined far from the impact: {x:?}");
    }
}

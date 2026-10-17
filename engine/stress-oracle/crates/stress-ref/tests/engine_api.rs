//! The engine-facing contract: contact impulses become a sustained load plus impact
//! pulses spread over a physical duration (never impulse / dt), crush-capped impactors, fracture
//! output with inherited velocities, and per-bond state exposed to other systems.

mod common;

use stress_ref::api::*;
use stress_ref::builders::*;
use stress_ref::math::{Pose, Vec3};
use stress_ref::scene::{CrushDesc, SolveMode};
use stress_ref::solver::{ChunkLoads, SolverEvent};

fn wall() -> stress_ref::scene::Scene {
    let mut s = wall_impact(0.0, "engine_wall");
    s.impactors.clear();
    s.probes.clear();
    s.metrics.clear();
    s
}

/// Integrate the filter's force on one chunk over time (trapezoid at fine resolution).
fn delivered_impulse(filter: &ContactLoadFilter, solver: &stress_ref::solver::ReferenceSolver, chunk: usize, t0: f64, t1: f64) -> (f64, f64) {
    let n = 20_000;
    let mut loads = ChunkLoads::new(solver);
    let (mut j, mut peak) = (0.0, 0.0f64);
    for i in 0..=n {
        let t = t0 + (t1 - t0) * i as f64 / n as f64;
        loads.clear();
        filter.loads_at(solver, t, &mut loads);
        let f = loads.force[0][chunk].norm();
        peak = peak.max(f);
        j += f * (t1 - t0) / n as f64 * if i == 0 || i == n { 0.5 } else { 1.0 };
    }
    (j, peak)
}

#[test]
fn impact_impulse_is_spread_over_the_hertz_duration() {
    let s = wall();
    let engine = EngineCoupledSolver::new(&s);
    let chunk = 500;
    let mut filter = ContactLoadFilter::new();
    let impulse = Vec3::new(0.0, 300.0, 0.0);
    let c = ContactImpulse {
        structure: 0,
        chunk,
        point: engine.solver.chunk_position(0, chunk),
        impulse,
        other_mass: 50.0,
        approach_speed: 6.0,
        other_modulus: 2e11,
        other_radius: 0.1,
        crush: None,
        other: None,
    };
    filter.ingest(&engine.solver, 0.0, 1.0 / 60.0, &[c]);
    // The wall is anchored, so the impact's effective mass is the impactor's.
    let ch = &engine.solver.structures[0].chunks[chunk];
    let e_star = 1.0 / ((1.0 - ch.poisson_ratio.powi(2)) / ch.youngs_modulus + 1.0 / 2e11);
    let td = hertz_duration(50.0, 0.1, e_star, 6.0);
    let (j, peak) = delivered_impulse(&filter, &engine.solver, chunk, 0.0, 2.0 * td);
    assert!(common::rel(j, impulse.norm()) < 1e-3, "impulse {j}");
    // Half-sine over the Hertz duration: peak = pi J / (2 td), not J / frame_dt.
    assert!(common::rel(peak, std::f64::consts::PI * j / (2.0 * td)) < 1e-2, "peak {peak}");
    assert!(td < 1.0 / 60.0, "a hard contact is shorter than a frame");
    assert!(common::rel(peak, j * 60.0) > 0.5, "never impulse/dt");
}

#[test]
fn crush_capped_impactor_holds_its_force_limit_and_delivers_the_impulse() {
    let s = wall();
    let engine = EngineCoupledSolver::new(&s);
    let chunk = 500;
    let mut filter = ContactLoadFilter::new();
    let max_force = 50e3;
    // 1500 kg car at 10 m/s, crushable energy 40 kJ < 75 kJ: crush, then bottom out.
    let c = ContactImpulse {
        structure: 0,
        chunk,
        point: engine.solver.chunk_position(0, chunk),
        impulse: Vec3::new(0.0, 15_000.0, 0.0),
        other_mass: 1500.0,
        approach_speed: 10.0,
        other_modulus: 2e11,
        other_radius: 0.3,
        crush: Some(CrushDesc { max_force, energy: 40e3 }),
        other: None,
    };
    filter.ingest(&engine.solver, 0.0, 1.0 / 60.0, &[c]);
    assert!(common::rel(filter.pulse_impulse(), 15_000.0) < 1e-9, "impulse {}", filter.pulse_impulse());
    // Crushing absorbs 40 kJ of the car's 75 kJ at the plateau force: the plateau
    // lasts m (v0 - v1) / F with v1 = sqrt(v0^2 - 2 E / m).
    let v1 = (100.0f64 - 2.0 * 40e3 / 1500.0).sqrt();
    let plateau = 1500.0 * (10.0 - v1) / max_force;
    let (j, _) = delivered_impulse(&filter, &engine.solver, chunk, 0.0, plateau + 0.05);
    assert!(common::rel(j, 15_000.0) < 1e-3, "impulse {j}");
    // During the plateau the force is exactly the crush force.
    let mut loads = ChunkLoads::new(&engine.solver);
    filter.loads_at(&engine.solver, 0.5 * plateau, &mut loads);
    assert!(common::rel(loads.force[0][chunk].norm(), max_force) < 1e-9);
}

#[test]
fn resting_contact_is_the_steady_load() {
    let s = wall();
    let engine = EngineCoupledSolver::new(&s);
    let chunk = 500;
    let mut filter = ContactLoadFilter::new();
    let dt = 1.0 / 60.0;
    let weight = 1000.0;
    for f in 0..120 {
        let c = ContactImpulse {
            structure: 0,
            chunk,
            point: engine.solver.chunk_position(0, chunk),
            impulse: Vec3::new(0.0, 0.0, -weight * dt),
            other_mass: 100.0,
            approach_speed: 0.0,
            other_modulus: 3e10,
            other_radius: 0.1,
            crush: None,
            other: None,
        };
        filter.ingest(&engine.solver, f as f64 * dt, dt, &[c]);
    }
    let mut loads = ChunkLoads::new(&engine.solver);
    filter.loads_at(&engine.solver, 2.0, &mut loads);
    assert!(common::rel(loads.force[0][chunk].z, -weight) < 1e-3, "{:?}", loads.force[0][chunk]);
}

/// Drive the solver like an engine would: a free beam, rigid motion owned by the
/// "engine", a hard hit in the middle. The solver reports the split, children carry
/// the parent's rigid velocity field plus what the impact gave them, and the bond
/// reports expose damage for decals, audio and VFX. Checked for the impact as a pulse
/// (explicit) and as a velocity condition (explicit and implicit at the frame step).
#[test]
fn engine_coupled_fracture_reports_children_and_bond_state() {
    for (model, mode) in [
        (ImpactModel::Pulse, SolveMode::Explicit),
        (ImpactModel::VelocityCondition, SolveMode::Explicit),
        (ImpactModel::VelocityCondition, SolveMode::Implicit),
    ] {
        let mut s = new_scene("free_beam", "");
        s.gravity = [0.0; 3];
        s.sim.solve_mode = mode;
        s.materials.insert("concrete".into(), oracle_concrete(None));
        let chunks = grid(Vec3::new(-0.5, -0.05, -0.05), [10, 1, 1], Vec3::splat(0.1), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        s.bodies.push(body("beam", chunks, bonds));
        let mut engine = EngineCoupledSolver::new(&s);
        engine.filter.impact_model = model;
        let id = engine.clusters()[0].id;
        let impulse = Vec3::new(0.0, 1500.0, 0.0);
        let mass0 = engine.clusters()[0].mass;
        // Like PhysX, the "engine" has already applied the contact impulse to the body.
        let v = Vec3::new(1.0, 0.0, 0.0) + impulse / mass0;
        let w = Vec3::new(0.0, 0.0, 0.5);
        let mut out_all = Vec::new();
        for f in 0..10 {
            let motion = vec![ClusterMotion { id, pose: Pose::default(), velocity: v, angular_velocity: w }];
            let contacts = if f == 0 {
                vec![ContactImpulse {
                    structure: 0,
                    chunk: 4,
                    point: engine.solver.chunk_position(0, 4),
                    impulse,
                    other_mass: 100.0,
                    approach_speed: 30.0,
                    other_modulus: 2e11,
                    other_radius: 0.05,
                    crush: None,
                    other: None,
                }]
            } else {
                vec![]
            };
            let out = engine.step(&FrameInput { dt: 1.0 / 60.0, motion: if f == 0 { motion } else { vec![] }, contacts });
            out_all.push(out);
        }
        let label = format!("{model:?} {mode:?}");
        let fractures: Vec<&Fracture> = out_all.iter().flat_map(|o| o.fractures.iter()).collect();
        assert!(!fractures.is_empty(), "{label}: the hit should break the beam");
        assert!(fractures[0].children.len() >= 2, "{label}: {fractures:?}");
        // The engine only knows the bodies it has: the beam is reported once, with every
        // piece it ended as (pieces that split again within the frame resolved), so the
        // pieces carry all of its mass (to the rounding of summing ten chunk masses).
        let beam: Vec<&Fracture> = fractures.iter().copied().filter(|f| f.parent == id).collect();
        assert_eq!(beam.len(), 1, "{label}: {fractures:?}");
        let pieces: f64 = beam[0].children.iter().map(|c| c.mass).sum();
        assert!((pieces - mass0).abs() <= 20.0 * f64::EPSILON * mass0, "{label}: pieces weigh {pieces}, the beam {mass0}");
        let reports = engine.bond_reports(0);
        assert!(reports.iter().any(|r| r.broken && r.damage >= 1.0), "{label}");
        assert!(reports.iter().all(|r| (0.0..=1.0).contains(&r.damage) && r.utilization >= 0.0), "{label}");
        let events: Vec<&SolverEvent> = out_all.iter().flat_map(|o| o.events.iter()).collect();
        assert!(events.iter().any(|e| matches!(e, SolverEvent::Cracked { .. })), "{label}");
        assert!(events.iter().any(|e| matches!(e, SolverEvent::Broken { .. })), "{label}");
        // The pieces carry exactly the momentum the engine gave the parent: the stress
        // solve only redistributes it (the impulse is not counted twice).
        let p: Vec3 = engine.clusters().iter().map(|c| Vec3::from_array(c.velocity) * c.mass).fold(Vec3::ZERO, |a, b| a + b);
        let expected = v * mass0;
        assert!((p - expected).norm() < 1e-6 * expected.norm(), "{label}: {p:?} vs {expected:?}");
        assert!(engine.clusters().iter().all(|c| !c.anchored), "{label}");
        println!("{label}: {} pieces", engine.clusters().len());
    }
}

//! Scenario runner. `run_native` drives the engine with the synchronous `step()`
//! against a software (or real) GPU adapter, records every tick, runs the per-tick
//! invariants and endpoint oracles, and returns a structured [`ScenarioReport`].
//!
//! The recording + checking helpers (`record_tick`, `build_spec`) are target-
//! agnostic so a future async wasm runner can reuse them verbatim with
//! `step_async().await`.

use glam::{DVec3, Mat3, Quat, Vec3};
use rubble3d::{RigidBodyDesc, SimConfig, World};
use rubble_math::BodyHandle;

use crate::invariants::{check_tick, energy_rel_tol, momentum_rel_tol, InvariantSpec};
use crate::metrics::{compute_metrics, BodyMeta, BodySample, TickRecord};
use crate::oracle::evaluate_endpoint;
use crate::report::{ScenarioReport, Violation};
use crate::scenario::{Scenario, ScenarioChecks};

/// Read one body's live state from the engine into a serializable sample.
fn record_tick(
    tick: usize,
    world: &World,
    handles: &[BodyHandle],
    metas: &[BodyMeta],
    gravity: Vec3,
) -> TickRecord {
    let mut bodies = Vec::with_capacity(handles.len());
    for (h, m) in handles.iter().zip(metas) {
        let pos = world.get_position(*h).unwrap_or(Vec3::ZERO);
        let rot = world.get_rotation(*h).unwrap_or(Quat::IDENTITY);
        let lin = world.get_velocity(*h).unwrap_or(Vec3::ZERO);
        let ang = world.get_angular_velocity(*h).unwrap_or(Vec3::ZERO);
        bodies.push(BodySample {
            label: m.label.clone(),
            pos: pos.to_array(),
            rot: rot.to_array(),
            lin: lin.to_array(),
            ang: ang.to_array(),
            mass: m.mass,
            is_dynamic: m.is_dynamic,
        });
    }
    let metrics = compute_metrics(&bodies, metas, gravity);
    TickRecord {
        tick,
        bodies,
        metrics,
    }
}

/// Build the per-tick invariant spec from the scenario and its recorded initial
/// (tick-0) state. All derived tolerances are assembled here in one place.
fn build_spec(
    cfg: &SimConfig,
    steps: usize,
    checks: &ScenarioChecks,
    r0: &TickRecord,
    shapes: Vec<rubble3d::ShapeDesc>,
    static_indices: Vec<usize>,
) -> InvariantSpec {
    let t = cfg.dt as f64 * steps as f64;

    let v0max = r0
        .bodies
        .iter()
        .map(|b| b.lin_vel().length() as f64)
        .fold(0.0, f64::max);
    let w0max = r0
        .bodies
        .iter()
        .map(|b| b.ang_vel().length() as f64)
        .fold(0.0, f64::max);

    let v_escape = checks
        .v_escape
        .unwrap_or_else(|| 4.0 * (v0max + cfg.gravity.length() as f64 * t) + 20.0);
    let omega_escape = checks.omega_escape.unwrap_or(4.0 * w0max + 50.0);

    let static_pos0 = static_indices
        .iter()
        .map(|&i| r0.bodies[i].position())
        .collect();

    let gmag = cfg.gravity.length() as f64;
    let lchar = r0
        .bodies
        .iter()
        .filter(|b| b.is_dynamic)
        .map(|b| b.position().length() as f64)
        .fold(1.0, f64::max);
    let total_mass = r0.metrics.total_mass.max(1e-3);
    // Absolute floor for energy non-increase: a small fraction of the available
    // potential *and* kinetic energy, so zero-gravity (purely kinetic) scenarios
    // still get a sensible floor instead of ~0.
    let energy_abs = 1e-3 * (total_mass * gmag * lchar + r0.metrics.kinetic_energy.abs()) + 1e-9;

    let momentum_scale: f64 = r0
        .bodies
        .iter()
        .filter(|b| b.is_dynamic)
        .map(|b| b.mass as f64 * b.lin_vel().length() as f64)
        .sum();
    // Derived f32 round-off floor for momentum conservation. The coefficient
    // counts the handful of round-off-accumulating ops per solver iteration; it is
    // order-of-magnitude (16 ≈ a few ops × a couple contacts), not fitted — a 5:1
    // mass-ratio collision lands near this floor at ~5e-5 relative.
    let momentum_tol = momentum_scale * momentum_rel_tol(cfg.solver_iterations) * 16.0 + 1e-6;

    InvariantSpec {
        dt: cfg.dt,
        gravity: cfg.gravity,
        quat_norm_tol: 1.0e-5,
        v_escape,
        omega_escape,
        teleport_extra: cfg.contact_offset as f64 + 4.0 * cfg.penetration_slop as f64,
        static_indices,
        static_pos0,
        static_tol: 1.0e-5,
        floor_y: checks.floor_y,
        penetration_bound: (cfg.penetration_slop + cfg.contact_offset) as f64,
        shapes,
        energy_non_increase: checks.energy_non_increase,
        baseline_energy: r0.metrics.total_energy,
        energy_rel: energy_rel_tol(cfg.solver_iterations),
        energy_abs,
        momentum_conserve: checks.momentum_conserve,
        baseline_momentum: DVec3::from_array(r0.metrics.linear_momentum),
        momentum_tol,
    }
}

/// Recover the local-frame inertia tensor the engine actually uses (it stores the
/// inverse). Zero for static bodies.
fn engine_inertia_local(world: &World, h: BodyHandle, is_dynamic: bool) -> Mat3 {
    if !is_dynamic {
        return Mat3::ZERO;
    }
    match world.get_inv_inertia(h) {
        Some(inv) if inv.determinant().abs() > 1e-20 => inv.inverse(),
        _ => Mat3::ZERO,
    }
}

/// Shared run core: create the world, add bodies, step + record + check, evaluate
/// oracles, and assemble the report. Parameterized by raw config/bodies/checks so
/// both the normal runner and the fault-injecting runner reuse identical logic
/// (the fault path differs only in the config/body descriptors it passes in).
fn run_core(
    name: &str,
    engine_cfg: &SimConfig,
    intended_cfg: &SimConfig,
    bodies: &[(&'static str, RigidBodyDesc)],
    steps: usize,
    checks: &ScenarioChecks,
) -> ScenarioReport {
    let mut world = match World::new(engine_cfg.clone()) {
        Ok(w) => w,
        Err(_) => return ScenarioReport::skipped(name),
    };

    let mut handles = Vec::with_capacity(bodies.len());
    let mut metas = Vec::with_capacity(bodies.len());
    let mut shapes = Vec::with_capacity(bodies.len());
    let mut static_indices = Vec::new();

    for (i, (label, desc)) in bodies.iter().enumerate() {
        let h = world.add_body(desc);
        let is_dynamic = desc.mass > 0.0;
        let inertia_local = engine_inertia_local(&world, h, is_dynamic);
        metas.push(BodyMeta {
            label: label.to_string(),
            mass: desc.mass,
            is_dynamic,
            inertia_local,
        });
        shapes.push(desc.shape.clone());
        if !is_dynamic {
            static_indices.push(i);
        }
        handles.push(h);
    }

    // The engine runs under `engine_cfg`, but every oracle/metric is evaluated
    // against `intended_cfg` (the *correct* physics). This is what lets a config
    // fault (e.g. negated gravity) be caught: the faulted trajectory is judged
    // against the intended expectation, not against the fault's own assumption.
    let mut traj: Vec<TickRecord> = Vec::with_capacity(steps + 1);
    traj.push(record_tick(
        0,
        &world,
        &handles,
        &metas,
        intended_cfg.gravity,
    ));
    let spec = build_spec(
        intended_cfg,
        steps,
        checks,
        &traj[0],
        shapes,
        static_indices,
    );

    let mut violations = Vec::new();
    for step in 1..=steps {
        world.step();
        let rec = record_tick(step, &world, &handles, &metas, intended_cfg.gravity);
        check_tick(&spec, traj.last(), &rec, &mut violations);
        traj.push(rec);
    }

    for ep in &checks.endpoints {
        evaluate_endpoint(ep, intended_cfg, steps, &traj, &mut violations);
    }
    let violations = collapse_violations(violations);

    let final_metrics = traj.last().map(|r| r.metrics.clone());
    let trajectory_on_failure = if violations.is_empty() {
        None
    } else {
        Some(traj)
    };

    ScenarioReport {
        scenario: name.to_string(),
        steps,
        body_count: bodies.len(),
        skipped_no_gpu: false,
        violations,
        final_metrics,
        trajectory_on_failure,
    }
}

/// Run a scenario natively (synchronous stepping). Returns a skipped report if no
/// GPU adapter is available, so the suite degrades gracefully on machines without
/// one rather than failing spuriously.
pub fn run_native(scn: &Scenario) -> ScenarioReport {
    let bodies: Vec<(&'static str, RigidBodyDesc)> = scn
        .bodies
        .iter()
        .map(|b| (b.label, b.desc.clone()))
        .collect();
    run_core(
        scn.name,
        &scn.config,
        &scn.config,
        &bodies,
        scn.steps,
        &scn.checks,
    )
}

/// Run a scenario with a deliberate fault injected into its config/bodies. Used by
/// the detection matrix to prove the suite actually catches bugs (and that its
/// tolerances aren't too loose). See [`crate::faults`].
#[cfg(feature = "faults")]
pub fn run_native_with_fault(scn: &Scenario, fault: crate::faults::FaultKind) -> ScenarioReport {
    let mut cfg = scn.config.clone();
    let mut descs: Vec<RigidBodyDesc> = scn.bodies.iter().map(|b| b.desc.clone()).collect();
    crate::faults::apply(fault, &mut cfg, &mut descs);
    let bodies: Vec<(&'static str, RigidBodyDesc)> = scn
        .bodies
        .iter()
        .zip(descs)
        .map(|(b, d)| (b.label, d))
        .collect();
    // Engine runs faulted (`cfg`); oracles judge against the intended `scn.config`.
    run_core(scn.name, &cfg, &scn.config, &bodies, scn.steps, &scn.checks)
}

/// Collapse repeated per-(kind, body) violations to their first occurrence,
/// annotating how many ticks they recurred on. The full trajectory is still
/// dumped for observability; this just keeps the violation list readable when a
/// per-tick invariant fails on every tick of a run.
fn collapse_violations(violations: Vec<Violation>) -> Vec<Violation> {
    use std::collections::{HashMap, HashSet};
    let mut counts: HashMap<(String, Option<usize>), usize> = HashMap::new();
    for v in &violations {
        *counts.entry((v.kind.clone(), v.body)).or_insert(0) += 1;
    }
    let mut seen: HashSet<(String, Option<usize>)> = HashSet::new();
    let mut out = Vec::new();
    for mut v in violations {
        let key = (v.kind.clone(), v.body);
        if seen.insert(key.clone()) {
            let c = counts[&key];
            if c > 1 {
                v.detail = format!("{} [recurred on {c} ticks]", v.detail);
            }
            out.push(v);
        }
    }
    out
}

/// Step a scene and return the full per-tick trajectory, with no checks applied.
/// Used by the metamorphic / determinism tests, which compare the trajectories of
/// deliberately-related scenes (same scene run twice, translated, reordered, or
/// re-massed). Returns `None` if no GPU adapter is available.
pub fn simulate_native(
    cfg: &SimConfig,
    bodies: &[(&'static str, RigidBodyDesc)],
    steps: usize,
) -> Option<Vec<TickRecord>> {
    let mut world = World::new(cfg.clone()).ok()?;
    let mut handles = Vec::with_capacity(bodies.len());
    let mut metas = Vec::with_capacity(bodies.len());
    for (label, desc) in bodies {
        let h = world.add_body(desc);
        let is_dynamic = desc.mass > 0.0;
        metas.push(BodyMeta {
            label: label.to_string(),
            mass: desc.mass,
            is_dynamic,
            inertia_local: engine_inertia_local(&world, h, is_dynamic),
        });
        handles.push(h);
    }
    let mut traj = Vec::with_capacity(steps + 1);
    traj.push(record_tick(0, &world, &handles, &metas, cfg.gravity));
    for step in 1..=steps {
        world.step();
        traj.push(record_tick(step, &world, &handles, &metas, cfg.gravity));
    }
    Some(traj)
}

/// Run every scenario in the ladder. Used by the native harness test and the gap
/// report generator.
pub fn run_all_native() -> Vec<ScenarioReport> {
    crate::scenario::scenarios()
        .iter()
        .map(run_native)
        .collect()
}

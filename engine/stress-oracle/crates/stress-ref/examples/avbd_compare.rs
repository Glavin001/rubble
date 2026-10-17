//! AVBD as the stress solver against the explicit truth (`findings/avbd-stress.md`).
//!
//! Each run prints one JSON line: the case, the solver variant, the outcome (broken
//! bonds, fragments, detached and fast chunks, impactor velocity, first crack, probes
//! summarised) and the cost (steps, sweeps, bond evaluations, wall time).
//!
//! ```text
//! cargo run --release -p stress-ref --example avbd_compare -- --case b5_wall_impact_v20 \
//!     --solver oracle|explicit|be|nm [--step 2e-5] [--iters 20] [--tol 0] \
//!     [--damage implicit|frozen] [--contact penalty|hard] [--end T] [--trace]
//! ```
//! `oracle` runs the oracle's own world (`World::run`); `explicit` is the chunk-level
//! replica of its integrator at its substep; `be` and `nm` are AVBD with backward Euler
//! and with Newmark's average acceleration. Methods are `established` (what the GPU
//! solver implements and what the rubble references were run with).

use stress_ref::avbd::{AvbdConfig, AvbdWorld, ContactMode, DamageMode, Integrator};
use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::world::World;

fn ram_into_free_wall() -> Scene {
    let mut s = new_scene(
        "ram_into_free_wall",
        "200 kg steel ram at 40 m/s into a free concrete wall",
    );
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let chunks = grid(
        Vec3::new(-0.5, 0.0, -0.1),
        [10, 2, 11],
        Vec3::splat(0.1),
        "concrete",
    );
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box {
            half_extents: [0.1, 0.1, 0.1],
        },
        mass: 200.0,
        position: [0.0, -0.15, 0.5],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 40.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = 0.5;
    s
}

/// A 400 kg box rammed at 12 m/s into the middle of a pack's -x face (as
/// `rubble_references`' `rammed`).
fn rammed(mut scene: Scene, name: &str) -> Scene {
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for c in scene.bodies.iter().flat_map(|b| b.chunks.iter()) {
        let p = Vec3::from_array(c.center);
        lo = lo.component_min(p);
        hi = hi.component_max(p);
    }
    let mid = (lo + hi) * 0.5;
    let material = scene.bodies[0].chunks[0].material.clone();
    scene.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box {
            half_extents: [0.15, 0.3, 0.3],
        },
        mass: 400.0,
        position: [lo.x - 0.35, mid.y, mid.z],
        orientation: [0.0, 0.0, 0.0, 1.0],
        velocity: [12.0, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material,
        crush: None,
    });
    scene.name = name.into();
    scene
}

fn scene(case: &str) -> Scene {
    if case == "ram_into_free_wall" {
        return ram_into_free_wall();
    }
    if case == "house_1story_ram" {
        let packs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs");
        let house = stress_ref::scene_pack::import(
            &packs.join("house-1story.json"),
            "house-1story",
            "structure",
        )
        .expect("house-1story pack");
        return rammed(house, "house_1story_ram");
    }
    if let Some(v) = case.strip_prefix("b5_wall_impact_v") {
        let speed: f64 = v.parse().expect("speed");
        return wall_impact(speed, case);
    }
    stress_ref::builders::catalog()
        .into_iter()
        .chain(stress_ref::showcases::catalog())
        .find(|s| s.name == case)
        .unwrap_or_else(|| panic!("no scene {case}"))
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let case = flag(&args, "--case").expect("--case NAME");
    let solver = flag(&args, "--solver").unwrap_or_else(|| "explicit".into());
    let mut sc = scene(&case);
    sc.sim.methods = Methods::established();
    if let Some(end) = flag(&args, "--end") {
        sc.sim.duration = end.parse().expect("--end T");
    }
    if args.iter().any(|a| a == "--true-step") {
        // The certified true stress step (the GPU solver's), with the contact limit.
        sc.sim.methods.true_step = true;
    }
    if args.iter().any(|a| a == "--no-prestress") {
        sc.sim.gravity_prestress = false;
    }
    if args.iter().any(|a| a == "--no-fracture") {
        sc.sim.fracture = false;
    }
    let derived = sc.with_derived();
    let start = std::time::Instant::now();
    let world = World::new(&derived);
    let setup = start.elapsed().as_secs_f64();
    if solver == "oracle" {
        let mut w = world;
        let t0 = std::time::Instant::now();
        let obs = w.run();
        let wall = t0.elapsed().as_secs_f64();
        let v0 = derived
            .impactors
            .iter()
            .map(|i| Vec3::from_array(i.velocity).norm())
            .fold(0.0, f64::max);
        let mut detached = 0;
        let mut fast = 0;
        for list in obs.bodies.values() {
            for c in list {
                if c.detached && !c.removed {
                    detached += 1;
                    if v0 > 0.0 && Vec3::from_array(c.velocity).norm() > 0.5 * v0 {
                        fast += 1;
                    }
                }
            }
        }
        let probes: serde_json::Map<String, serde_json::Value> = obs
            .probes
            .iter()
            .map(|(k, v)| (k.clone(), oracle_summary(v)))
            .collect();
        let out = serde_json::json!({
            "case": case, "solver": "oracle",
            "broken_bonds": obs.values.get("broken_bonds"), "fragments": obs.values.get("fragments"),
            "detached_chunks": detached, "fast_detached": fast,
            "first_crack_time": obs.values.get("first_crack_time"),
            "max_utilization": obs.values.get("max_failure_index"),
            "bond_dissipation": obs.values.get("bond_dissipation"),
            "impactor_velocity": w.impactors.iter().map(|i| i.velocity.to_array()).collect::<Vec<_>>(),
            "substeps": obs.values.get("substeps"), "substep": w.substep_dt(),
            "wall_seconds": wall, "setup_seconds": setup, "end_time": obs.end_time,
            "probes": probes,
        });
        if let Some(path) = flag(&args, "--dump") {
            std::fs::write(path, serde_json::to_string(&obs.probes).unwrap()).expect("dump");
        }
        println!("{out}");
        return;
    }
    let integrator = match solver.as_str() {
        "explicit" => Integrator::Explicit,
        "be" => Integrator::BackwardEuler,
        "nm" => Integrator::Newmark,
        other => panic!("--solver oracle|explicit|be|nm, not {other}"),
    };
    let config = AvbdConfig {
        integrator,
        step: flag(&args, "--step").map_or(1e-4, |v| v.parse().expect("--step")),
        iterations: flag(&args, "--iters").map_or(20, |v| v.parse().expect("--iters")),
        tolerance: flag(&args, "--tol").map_or(0.0, |v| v.parse().expect("--tol")),
        damage: match flag(&args, "--damage").as_deref() {
            None | Some("implicit") => DamageMode::Implicit,
            Some("frozen") => DamageMode::Frozen,
            Some(o) => panic!("--damage implicit|frozen, not {o}"),
        },
        contact: match flag(&args, "--contact").as_deref() {
            None | Some("penalty") => ContactMode::Penalty,
            Some("hard") => ContactMode::Hard,
            Some(o) => panic!("--contact penalty|hard, not {o}"),
        },
        trace_residuals: args.iter().any(|a| a == "--trace"),
        warm_previous: !args.iter().any(|a| a == "--warm-inertial"),
        margin: flag(&args, "--margin").map_or(0.25, |v| v.parse().expect("--margin")),
        stall: flag(&args, "--stall").map_or(40, |v| v.parse().expect("--stall")),
        hard_cap: flag(&args, "--hard-cap").map_or(1e6, |v| v.parse().expect("--hard-cap")),
        beta: flag(&args, "--beta").map_or(10.0, |v| v.parse().expect("--beta")),
        chebyshev: flag(&args, "--chebyshev").map_or(0.0, |v| v.parse().expect("--chebyshev")),
        predict_contacts: !args.iter().any(|a| a == "--start-contacts"),
        ..AvbdConfig::default()
    };
    let mut aw = AvbdWorld::from_world(&world, config.clone());
    let (bodies, bonds) = aw.size();
    if args.iter().any(|a| a == "--bench") {
        let reps = flag(&args, "--reps").map_or(200, |v| v.parse().expect("--reps"));
        let (explicit, sweep, colours) = aw.bench_kernels(reps, config.step);
        println!(
            "{}",
            serde_json::json!({"case": case, "bench": true, "free_bodies": bodies, "bonds": bonds,
            "explicit_pass_us": explicit * 1e6, "sweep_us": sweep * 1e6, "ratio": sweep / explicit, "colours": colours,
            "explicit_substep": aw.law_dt})
        );
        return;
    }
    let r = aw.run();
    let probes: serde_json::Map<String, serde_json::Value> = r
        .probes
        .iter()
        .map(|(k, v)| (k.clone(), summarise(v)))
        .collect();
    let steps = r.stats.steps as f64;
    // Mean sweeps of the steps that met the tolerance (neither capped nor stalled).
    let converged_mean = if config.tolerance > 0.0 {
        let n_bad = r.stats.unconverged_steps as usize;
        let mut c: Vec<u32> = r.stats.sweep_counts.clone();
        c.sort_unstable();
        let ok = &c[..c.len().saturating_sub(n_bad)];
        ok.iter().map(|&x| x as f64).sum::<f64>() / ok.len().max(1) as f64
    } else {
        0.0
    };
    let mut out = serde_json::json!({
        "case": case, "solver": solver,
        "step": if integrator == Integrator::Explicit { aw.law_dt } else { config.step },
        "iters": config.iterations, "tol": config.tolerance,
        "damage": format!("{:?}", config.damage), "warm_previous": config.warm_previous, "margin": config.margin, "chebyshev": config.chebyshev, "predict_contacts": config.predict_contacts, "contact": format!("{:?}", config.contact),
        "free_bodies": bodies, "bonds": bonds,
        "broken_bonds": r.broken_bonds, "fragments": r.fragments,
        "detached_chunks": r.detached_chunks, "fast_detached": r.fast_detached,
        "first_crack_time": r.first_crack_time, "max_utilization": r.max_utilization,
        "impactor_velocity": r.impactor_velocity, "bond_dissipation": r.bond_dissipation,
        "steps": r.stats.steps, "sweeps": r.stats.sweeps,
        "sweeps_per_step": r.stats.sweeps as f64 / steps.max(1.0),
        "bond_evaluations": r.stats.bond_evaluations, "block_solves": r.stats.block_solves,
        "mean_colours": r.stats.colour_sum as f64 / steps.max(1.0), "max_colours": r.stats.max_colours,
        "worst_residual": r.stats.worst_residual, "mean_residual": r.stats.residual_sum / steps.max(1.0),
        "unconverged_steps": r.stats.unconverged_steps,
        "stalled_steps": r.stats.stalled_steps, "worst_stall": r.stats.worst_stall,
        "sweeps_median": pct(&r.stats.sweep_counts, 0.5), "sweeps_p90": pct(&r.stats.sweep_counts, 0.9),
        "sweeps_max": r.stats.sweep_counts.iter().copied().max().unwrap_or(0),
        "sweeps_converged_mean": converged_mean,
        "wall_seconds": r.stats.wall_seconds, "setup_seconds": setup, "end_time": r.time,
        "probes": probes,
    });
    if config.trace_residuals {
        // Sweep residuals of the step with the worst final residual, and the mean per sweep.
        let traces = &r.stats.traces;
        if let Some(worst) = traces
            .iter()
            .max_by(|a, b| a.last().unwrap_or(&0.0).total_cmp(b.last().unwrap_or(&0.0)))
        {
            out["worst_trace"] = serde_json::json!(worst);
        }
        let n = traces.iter().map(|t| t.len()).max().unwrap_or(0);
        let mean: Vec<f64> = (0..n)
            .map(|i| {
                let v: Vec<f64> = traces.iter().filter_map(|t| t.get(i).copied()).collect();
                v.iter().sum::<f64>() / v.len().max(1) as f64
            })
            .collect();
        out["mean_trace"] = serde_json::json!(mean);
    }
    if let Some(path) = flag(&args, "--dump") {
        std::fs::write(path, serde_json::to_string(&r.probes).unwrap()).expect("dump");
    }
    println!("{out}");
}

fn pct(v: &[u32], q: f64) -> u32 {
    if v.is_empty() {
        return 0;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// The oracle's series with its between-sample envelope (`lo`, `hi`).
fn oracle_summary(p: &stress_ref::observation::ProbeSeries) -> serde_json::Value {
    let pts: Vec<[f64; 2]> = p.t.iter().zip(&p.v).map(|(&t, &v)| [t, v]).collect();
    let mut s = summarise(&pts);
    if !p.hi.is_empty() {
        s["max"] = serde_json::json!(p.hi.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        s["min"] = serde_json::json!(p.lo.iter().copied().fold(f64::INFINITY, f64::min));
    }
    s
}

/// A probe's series as: first, last, min, max, time of max, and for oscillations the
/// mean period from upward zero crossings about the final mean.
fn summarise(v: &[[f64; 2]]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    let first = v[0][1];
    let last = v[v.len() - 1][1];
    let (mut lo, mut hi, mut t_hi) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for p in v {
        if p[1] < lo {
            lo = p[1];
        }
        if p[1] > hi {
            hi = p[1];
            t_hi = p[0];
        }
    }
    serde_json::json!({"first": first, "last": last, "min": lo, "max": hi, "t_max": t_hi, "n": v.len()})
}

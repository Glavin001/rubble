//! Multi-time-step explicit-implicit coupling against the explicit truth
//! (`findings/multi-time-step.md`).
//!
//! Each run prints one JSON line: the case, the solver, the outcome (broken bonds,
//! fragments, detached and fast chunks with the detached set, ram velocity, first crack)
//! and the cost (fine and coarse steps, hot-zone and exterior work, wall time).
//!
//! ```text
//! cargo run --release --features mts -p stress-ref --example mts_compare -- \
//!     --case b5_wall_impact_v20 --solver oracle|explicit|mts [--true-step] [--end T] \
//!     [--ratio M] [--exterior newmark|cb] [--modes K] [--u-hot U] \
//!     [--horizon H] [--no-influence] [--verify rollback|count] [--drift] [--all-hot] \
//!     [--series FILE]
//! ```
//! `explicit` is the chunk-level replica of the oracle's integrator (`avbd.rs`); `mts`
//! runs its hot zone at the same step. Methods are `established`.
#![recursion_limit = "512"]

use stress_ref::avbd::mts::{detached_set, utilisation_map, Exterior, MtsConfig, MtsWorld, Verify};
use stress_ref::avbd::{AvbdConfig, AvbdWorld, Integrator};
use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::world::World;

fn ram_into_free_wall() -> Scene {
    let mut s = new_scene("ram_into_free_wall", "200 kg steel ram at 40 m/s into a free concrete wall");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let chunks = grid(Vec3::new(-0.5, 0.0, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.1, 0.1, 0.1] },
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
        shape: ImpactorShape::Box { half_extents: [0.15, 0.3, 0.3] },
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
        let house = stress_ref::scene_pack::import(&packs.join("house-1story.json"), "house-1story", "structure")
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
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let case = flag(&args, "--case").expect("--case NAME");
    let solver = flag(&args, "--solver").unwrap_or_else(|| "mts".into());
    let mut sc = scene(&case);
    sc.sim.methods = Methods::established();
    if let Some(end) = flag(&args, "--end") {
        sc.sim.duration = end.parse().expect("--end T");
    }
    if has(&args, "--true-step") {
        sc.sim.methods.true_step = true;
    }
    let derived = sc.with_derived();
    let world = World::new(&derived);
    let v0 = derived.impactors.iter().map(|i| Vec3::from_array(i.velocity).norm()).fold(0.0, f64::max);
    let label = flag(&args, "--label").unwrap_or_else(|| solver.clone());
    match solver.as_str() {
        "oracle" => {
            let mut w = world;
            let t0 = std::time::Instant::now();
            let obs = w.run();
            let wall = t0.elapsed().as_secs_f64();
            let mut detached = Vec::new();
            let mut fast = 0;
            let mut speed = 0.0;
            for (s, b) in derived.bodies.iter().enumerate() {
                if let Some(list) = obs.bodies.get(&b.name) {
                    for (c, ch) in list.iter().enumerate() {
                        if ch.detached && !ch.removed {
                            detached.push((s, c));
                            let sp = Vec3::from_array(ch.velocity).norm();
                            speed += sp;
                            if v0 > 0.0 && sp > 0.5 * v0 {
                                fast += 1;
                            }
                        }
                    }
                }
            }
            let n = detached.len().max(1) as f64;
            let out = serde_json::json!({
                "case": case, "solver": "oracle", "label": label,
                "broken_bonds": obs.values.get("broken_bonds"), "fragments": obs.values.get("fragments"),
                "detached_chunks": detached.len(), "fast_detached": fast, "detached": detached, "debris_speed": speed / n,
                "first_crack_time": obs.values.get("first_crack_time"),
                "impactor_velocity": w.impactors.iter().map(|i| i.velocity.to_array()).collect::<Vec<_>>(),
                "substeps": obs.values.get("substeps"), "substep": w.substep_dt(), "wall_seconds": wall,
            });
            println!("{out}");
        }
        "explicit" => {
            let mut aw = AvbdWorld::from_world(&world, AvbdConfig { integrator: Integrator::Explicit, ..AvbdConfig::default() });
            let r = aw.run();
            let (detached, debris) = detached_set(&aw);
            let out = serde_json::json!({
                "case": case, "solver": "explicit", "label": label, "step": aw.law_dt,
                "broken_bonds": r.broken_bonds, "fragments": r.fragments,
                "detached_chunks": r.detached_chunks, "fast_detached": r.fast_detached,
                "detached": detached, "debris_speed": debris,
                "first_crack_time": r.first_crack_time, "impactor_velocity": r.impactor_velocity,
                "steps": r.stats.steps, "bond_evaluations": r.stats.bond_evaluations,
                "contact_evaluations": r.stats.contact_evaluations,
                "wall_seconds": r.stats.wall_seconds, "end_time": r.time,
            });
            if let Some(path) = flag(&args, "--dump") {
                std::fs::write(path, serde_json::to_string(&r.probes).unwrap()).expect("dump");
            }
            println!("{out}");
        }
        "mts" => {
            let d = MtsConfig::default();
            let config = MtsConfig {
                ratio: flag(&args, "--ratio").map_or(d.ratio, |v| v.parse().expect("--ratio")),
                exterior: match flag(&args, "--exterior").as_deref() {
                    None | Some("newmark") => Exterior::Newmark,
                    Some("cb") => Exterior::CraigBampton,
                    Some(o) => panic!("--exterior newmark|cb, not {o}"),
                },
                cb_modes: flag(&args, "--modes").map_or(d.cb_modes, |v| v.parse().expect("--modes")),
                u_hot: flag(&args, "--u-hot").map_or(d.u_hot, |v| v.parse().expect("--u-hot")),
                horizon: flag(&args, "--horizon").map_or(d.horizon, |v| v.parse().expect("--horizon")),
                influence: !has(&args, "--no-influence"),
                influence_growth: flag(&args, "--growth").map_or(d.influence_growth, |v| v.parse().expect("--growth")),
                verify: match flag(&args, "--verify").as_deref() {
                    None | Some("rollback") => Verify::Rollback,
                    Some("count") => Verify::Count,
                    Some(o) => panic!("--verify rollback|count, not {o}"),
                },
                drift_correction: has(&args, "--drift"),
                max_rollbacks: flag(&args, "--max-rollbacks").map_or(d.max_rollbacks, |v| v.parse().expect("--max-rollbacks")),
                contact_margin: flag(&args, "--margin").map_or(d.contact_margin, |v| v.parse().expect("--margin")),
                newton_tol: flag(&args, "--newton-tol").map_or(d.newton_tol, |v| v.parse().expect("--newton-tol")),
                all_hot: has(&args, "--all-hot"),
                seed: flag(&args, "--seed").map_or(Vec::new(), |v| {
                    v.split(',').flat_map(|r| {
                        let (a, b) = r.split_once('-').unwrap_or((r, r));
                        a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()
                    }).collect()
                }),
                escalation: !has(&args, "--no-escalation"),
                energy: has(&args, "--energy"),
                fine_scale: flag(&args, "--fine-scale").map_or(1.0, |v| v.parse().expect("--fine-scale")),
                ..d
            };
            let mut mw = MtsWorld::new(&world, config.clone());
            let r = mw.run();
            let o = &r.outcome;
            let st = &r.stats;
            let recs = &st.records;
            let peak = |f: &dyn Fn(&stress_ref::avbd::mts::CoarseRecord) -> usize| recs.iter().map(f).max().unwrap_or(0);
            let mut out = serde_json::json!({
                "case": case, "solver": "mts", "label": label, "config": config, "step": mw.w.law_dt,
                "broken_bonds": o.broken_bonds, "fragments": o.fragments,
                "detached_chunks": o.detached_chunks, "fast_detached": o.fast_detached,
                "detached": r.detached, "debris_speed": r.debris_speed,
                "first_crack_time": o.first_crack_time, "impactor_velocity": o.impactor_velocity,
                "diverged": r.diverged, "end_time": o.time,
                "fine_steps": st.fine_steps, "coarse_steps": st.coarse_steps,
                "hot_bond_evals": st.hot_bond_evals, "full_bond_evals": st.full_bond_evals, "lookahead_bond_evals": st.lookahead_bond_evals,
                "explicit_work": st.hot_bond_evals as f64 / st.full_bond_evals.max(1) as f64,
                "hot_body_updates": st.hot_body_updates, "full_body_updates": st.full_body_updates,
                "hot_contact_evals": st.hot_contact_evals,
                "newton_iterations": st.newton_iterations, "ext_bond_evals": st.ext_bond_evals,
                "assemblies": st.assemblies, "factorizations": st.factorizations,
                "factor_flops": st.factor_flops, "solves": st.solves, "solve_flops": st.solve_flops,
                "eig_flops": st.eig_flops, "interface_dofs_max": st.interface_dofs_max,
                "unconverged_steps": st.unconverged_steps, "regularised_factorizations": st.regularised_factorizations, "worst_newton_residual": st.worst_newton_residual,
                "misses_utilization": st.misses_utilization, "misses_contact": st.misses_contact,
                "misses_damage": st.misses_damage, "rollbacks": st.rollbacks, "repartitions": st.repartitions,
                "exterior_cracks": st.exterior_cracks,
                "core_max": peak(&|r| r.core), "hot_max": peak(&|r| r.hot), "ring_max": peak(&|r| r.ring),
                "hot_final": recs.last().map(|r| r.hot), "drift_max": recs.iter().map(|r| r.drift).fold(0.0, f64::max),
                "velocity_gap_max": recs.iter().map(|r| r.velocity_gap).fold(0.0, f64::max),
                "ext_u_max": recs.iter().map(|r| r.ext_u_max).fold(0.0, f64::max),
                "t_hot": st.t_hot, "t_newton": st.t_newton, "t_factor": st.t_factor, "t_escalate": st.t_escalate,
                "wall_seconds": st.t_total,
            });
            if let Some(path) = flag(&args, "--dump") {
                std::fs::write(path, serde_json::to_string(&o.probes).unwrap()).expect("dump");
            }
            if let Some(path) = flag(&args, "--series") {
                std::fs::write(path, serde_json::to_string(recs).unwrap()).expect("series");
            }
            if has(&args, "--print-series") {
                out["series"] = serde_json::json!(recs);
            }
            println!("{out}");
        }
        "umap" => {
            let levels = [0.25, 0.5, 0.8, 1.0];
            let map = utilisation_map(&world, derived.sim.duration, &levels);
            println!("{}", serde_json::json!({"case": case, "levels": levels, "bonds": map}));
        }
        other => panic!("--solver oracle|explicit|mts|umap, not {other}"),
    }
}

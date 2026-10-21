//! P0 of the learned impact-outcome plan (`docs/learned-stress-solver/learned-stress-solver-plan.md`):
//! the oracle pin, the chaos floor and the perfect-model ceiling, one JSON line per run.
//!
//!   cargo run --release -p stress-rubble --features gpu --example learned_p0 -- \
//!       oracle  --scene free_wall [--member K --perturb kin|kin_weibull] [--seconds S] [--frame-dt DT]
//!       ceiling --scene free_wall --variant coupled|window_full|window_reduced|pre_filtered|pre_engine [--frame-dt DT]
//!       ceiling --scene free_wall --variant provider --provider oracle|files|punch [--dir D --timeout S --echo-oracle]
//!       hash    (every scene's canonical hash)
//!
//! `oracle`: the standalone stress-ref world (CPU f64), the truth of `findings/rewind-scaling.md`.
//! `ceiling`: the stress-ref world coupled to rubble's engine; at every frame the routing puts
//! an island in, the standalone oracle's own next frame is the "model" and its outcome is
//! applied as the variant says (see `stress_ref::world::handoff`):
//!   coupled         the coupled world as it is (islands simulated in place, full state)
//!   window_full     the oracle's frame-end state adopted whole
//!   window_reduced  the same, reduced to rigid bodies (hidden deformation dropped)
//!   pre_filtered    breaks and frame-end velocities applied at the frame-start poses, the
//!                   frame then ballistic (the call's contacts not given to the engine)
//!   pre_engine      the same, the frame then stepped by the engine with its contacts
//! A progress line goes to stderr every frame.

use std::collections::HashSet;
use std::time::Instant;

use serde_json::{json, Value};
use stress_ref::math::Vec3;
use stress_ref::scene::Scene;
use stress_ref::world::World;

#[path = "../tests/rewind_scenes/mod.rs"]
mod rewind_scenes;

const SECONDS: f64 = 0.15;

/// The P0 scenes, in the pinned configuration (`stress_rubble::scenes::realtime`).
fn scene(name: &str, seconds: f64) -> Scene {
    use stress_ref::builders::*;
    use stress_ref::scene::*;
    let rt = stress_rubble::scenes::realtime;
    match name {
        // The two M5 scenes as rewind-scaling.md's truth runs them (ram 5 cm short).
        "free_wall" => rewind_scenes::wall_row(1, 40.0, 0.1, seconds),
        "wall_row" => rewind_scenes::wall_row(2, 40.0, 0.1, seconds),
        // The same as stress_bench's GPU scenes run them (ram 1.15 m short, strikes in frame 2).
        "free_wall_late" => rt(stress_rubble::scenes::wall_ram_late(), seconds),
        "wall_row_late" => rt(stress_rubble::scenes::wall_row_late(2), seconds),
        // Two frames later still: the strike in frame 3, which the GPU punch needs (its check at
        // the end of a resident frame sees it coming; stress_bench's `*_late2`, `tests/punch_gpu.rs`).
        "free_wall_late2" => rt(stress_rubble::scenes::later(stress_rubble::scenes::wall_ram_late(), 2), seconds),
        "wall_row_late2" => rt(stress_rubble::scenes::later(stress_rubble::scenes::wall_row_late(2), 2), seconds),
        // Five more walls.
        "free_wall_v25" => rewind_scenes::wall_row(1, 25.0, 0.1, seconds),
        "free_wall_v60" => rewind_scenes::wall_row(1, 60.0, 0.1, seconds),
        "free_wall_edge" => {
            let mut s = rewind_scenes::wall_row(1, 40.0, 0.1, seconds);
            s.name = "free_wall_edge".into();
            s.impactors[0].position[0] = 0.35;
            s
        }
        "thick_wall" => {
            let mut s = rewind_scenes::wall_row(1, 40.0, 0.1, seconds);
            s.name = "thick_wall".into();
            let chunks = grid(stress_ref::math::Vec3::new(-0.5, 0.0, -0.1), [10, 3, 11], stress_ref::math::Vec3::splat(0.1), "concrete");
            let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
            s.bodies = vec![body("wall0", chunks, bonds)];
            s
        }
        "b5_v20" => {
            let mut s = rt(wall_impact(20.0, "b5_v20"), seconds);
            s.sim.frame_dt = 1.0 / 60.0;
            s
        }
        other => panic!("unknown scene {other}"),
    }
}

const SCENES: [&str; 11] = ["free_wall", "wall_row", "free_wall_late", "wall_row_late", "free_wall_late2", "wall_row_late2", "free_wall_v25", "free_wall_v60", "free_wall_edge", "thick_wall", "b5_v20"];

/// splitmix64: member `k`'s perturbation, reproducible.
fn uniform(seed: u64, i: u64) -> f64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Ensemble member `k`: the impact point moved by up to ±1 mm across the face, the speed
/// scaled by up to ±0.1 %, and (`kin_weibull`) Weibull strengths (modulus 8) with seed k.
fn perturb(s: &mut Scene, k: u64, mode: &str) -> Value {
    let seed = 0x5EED_0000 + k;
    // Diagnostic modes: `speed` (the speed only), `point` (the point only), `point_um`
    // (the point by up to ±1 µm), `x_only` / `z_only` (the point along one axis only).
    let base = mode.trim_end_matches("_weibull");
    let point = match base { "kin" | "point" => 1e-3, "point_um" => 1e-6, "x_only" | "z_only" => 1e-3, _ => 0.0 };
    let speed = if matches!(base, "kin" | "speed") { 1e-3 } else { 0.0 };
    let (mut dx, mut dz, fs) = (point * uniform(seed, 1), point * uniform(seed, 2), 1.0 + speed * uniform(seed, 3));
    if base == "x_only" {
        dz = 0.0;
    }
    if base == "z_only" {
        dx = 0.0;
    }
    let imp = &mut s.impactors[0];
    let v = Vec3::from_array(imp.velocity);
    let d = v * (1.0 / v.norm());
    // Two directions across the approach.
    let a = if d.z.abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let e1 = a.cross(d);
    let e1 = e1 * (1.0 / e1.norm());
    let e2 = d.cross(e1);
    let off = e1 * dx + e2 * dz;
    for i in 0..3 {
        imp.position[i] += off[i];
        imp.velocity[i] *= fs;
    }
    let mut weibull = None;
    if mode.ends_with("_weibull") {
        if let Some(m) = s.materials.get_mut("concrete") {
            m.weibull_modulus = Some(8.0);
        }
        s.sim.seed = k;
        weibull = Some(8.0);
    } else if !matches!(base, "kin" | "speed" | "point" | "point_um" | "x_only" | "z_only") {
        panic!("unknown --perturb {mode}");
    }
    json!({"member": k, "mode": mode, "offset_m": [off.x, off.y, off.z], "speed_factor": fs, "weibull_modulus": weibull, "seed": s.sim.seed})
}

fn scene_hash(s: &Scene) -> String {
    // FNV-1a over the canonical JSON (serde_json with float_roundtrip, sorted by the derive order).
    let text = serde_json::to_string(s).expect("scene json");
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// The run's outcome and ledger at the current state.
fn outcome(w: &World) -> Value {
    let mut v = outcome_of(&w.solver, &w.impactors);
    let e = v["energy"].as_object_mut().unwrap();
    e.insert("mechanical".into(), json!(w.mechanical_energy()));
    e.insert("dissipated".into(), json!(w.dissipated_energy()));
    e.insert("contact_dissipation".into(), json!(w.contact.dissipated));
    e.insert("crush".into(), json!(w.contact.crush));
    v
}

/// The outcome of a solver state (the oracle's, or a GPU world's f64 mirror).
fn outcome_of(sv: &stress_ref::solver::ReferenceSolver, impactors: &[stress_ref::world::Impactor]) -> Value {
    let imp = &impactors[0];
    let mut walls = Vec::new();
    let mut largest = 0.0f64;
    let mut total = 0.0;
    for (s, st) in sv.structures.iter().enumerate() {
        let mass: f64 = st.chunks.iter().map(|c| c.mass).sum();
        total += mass;
        let clusters: Vec<&stress_ref::solver::Cluster> = sv.clusters.iter().filter(|c| c.structure == s).collect();
        let big = clusters.iter().map(|c| c.mass).fold(0.0, f64::max);
        largest = largest.max(big);
        let broken: Vec<usize> = (0..sv.bonds[s].len()).filter(|&b| !sv.bonds[s][b].joint.connected(sv.bonds[s][b].rebar.is_some())).collect();
        // Chunk -> fragment (cluster id), for partition metrics.
        let part: Vec<i64> = (0..st.chunks.len()).map(|c| if sv.chunks[s][c].active { sv.clusters[sv.chunks[s][c].cluster].id as i64 } else { -1 }).collect();
        // Chunk world velocities (for velocity-field metrics).
        let vel: Vec<[f64; 3]> = (0..st.chunks.len()).map(|c| if sv.chunks[s][c].active { sv.chunk_velocity(s, c).0.to_array() } else { [0.0; 3] }).collect();
        let anchored: f64 = clusters.iter().filter(|c| c.anchored).map(|c| c.mass).sum();
        walls.push(json!({"name": st.name, "mass": mass, "fragments": clusters.len(), "largest_fraction": big / mass,
            "anchored_fraction": anchored / mass, "broken": broken, "bonds": sv.bonds[s].len(), "partition": part, "chunk_velocity": vel}));
    }
    let e = &sv.energy;
    let ke_imp: f64 = impactors.iter().map(|i| i.kinetic_energy()).sum();
    // The fracture-work floor the plan's energy bound would use: each broken bond's
    // smallest fracture energy times its area, against what the bonds booked.
    let (mut floor, mut booked) = (0.0, 0.0);
    for bonds in &sv.bonds {
        for b in bonds.iter().filter(|b| !b.joint.connected(b.rebar.is_some())) {
            floor += b.strength.g_tension.min(b.strength.g_shear) * b.geometry.area;
            booked += b.joint.dissipated;
        }
    }
    json!({
        "t": sv.time,
        "ram_velocity": imp.velocity.to_array(),
        "ram_vy": imp.velocity.y,
        "ram_position": imp.pose.position.to_array(),
        "fragments": sv.clusters.len(),
        "broken_bonds": sv.broken_bond_count(),
        "largest_fraction": largest / total,
        "walls": walls,
        "energy": {
            "ke_chunks": sv.kinetic_energy(),
            "ke_impactors": ke_imp,
            "elastic": sv.elastic_energy(),
            "bond_dissipation": e.bond_dissipation,
            "damping_dissipation": e.damping_dissipation,
            "external_work": e.external_work,
            "softening_overshoot": e.softening_overshoot,
            "softening_overshoot_negative": e.softening_overshoot_negative,
            "split_release": e.split_release,
            "static_discard": e.static_discard,
            "fracture_floor_gf_area": floor,
            "broken_bonds_booked": booked,
        }
    })
}

fn frame_line(w: &World, ms: f64, islands: usize) -> Value {
    let e = &w.solver.energy;
    json!({"t": w.solver.time, "ms": ms, "islands": islands, "ram_vy": w.impactors[0].velocity.y, "fragments": w.solver.clusters.len(),
        "broken": w.solver.broken_bond_count(), "mechanical": w.mechanical_energy(), "dissipated": w.dissipated_energy(),
        "ke_impactors": w.impactors.iter().map(|i| i.kinetic_energy()).sum::<f64>(), "ke_chunks": w.solver.kinetic_energy(),
        "elastic": w.solver.elastic_energy(), "bond_dissipation": e.bond_dissipation, "softening_overshoot": e.softening_overshoot,
        "softening_overshoot_negative": e.softening_overshoot_negative, "substeps": w.solver.substeps})
}

fn frames(s: &Scene) -> usize {
    (s.sim.duration / s.sim.frame_dt).round() as usize
}

fn run_oracle(s: &Scene, tag: &str) -> Value {
    let mut w = World::new(s);
    let n = frames(s);
    let e0 = w.mechanical_energy();
    let mut series = Vec::with_capacity(n);
    let t_all = Instant::now();
    for f in 0..n {
        let t = Instant::now();
        w.step_frame();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        series.push(frame_line(&w, ms, 0));
        eprintln!("[oracle {tag}] frame {}/{n} t {:.4} ram vy {:.3} fragments {} broken {} ({ms:.0} ms)", f + 1, w.solver.time, w.impactors[0].velocity.y, w.solver.clusters.len(), w.solver.broken_bond_count());
    }
    json!({"e0": e0, "frames": series, "end": outcome(&w), "total_s": t_all.elapsed().as_secs_f64()})
}

fn engine(s: &Scene) -> Box<stress_rubble::RubbleEngine> {
    Box::new(stress_rubble::RubbleEngine::new(Vec3::from_array(s.gravity)).expect("rubble engine"))
}

/// The oracle's frame from `w`'s state, applied to `w` as `variant` says.
fn model_frame(w: &mut World, variant: &str, fdt: f64) {
    match variant {
        "window_full" | "window_reduced" => {
            let mut model = w.standalone_copy();
            model.step_frame();
            w.adopt_keeping_engine(model);
            if variant == "window_reduced" {
                w.reduce_to_rigid();
                w.push_motion_to_engine();
            }
        }
        "pre_filtered" | "pre_engine" => {
            let mut model = w.standalone_copy();
            model.step_frame();
            // Breaks, then each bond's carried state (damage), then the rigid motion fitted
            // to the model's frame-end chunk velocities, then the impactors' velocities.
            let mut broken = Vec::new();
            for s in 0..w.solver.bonds.len() {
                for b in 0..w.solver.bonds[s].len() {
                    if w.solver.bonds[s][b].connected() && !model.solver.bonds[s][b].connected() {
                        broken.push((s, b));
                    }
                }
            }
            w.solver.break_bonds(&broken);
            for s in 0..w.solver.bonds.len() {
                for b in 0..w.solver.bonds[s].len() {
                    w.solver.bonds[s][b].joint = model.solver.bonds[s][b].joint.clone();
                }
            }
            let mv = &model.solver;
            let vel = |s: usize, c: usize| mv.chunks[s][c].active.then(|| mv.chunk_velocity(s, c));
            w.set_rigid_from_chunks(&vel);
            // The ledger of what the call dissipated, so the balance stays readable.
            w.solver.energy = model.solver.energy;
            w.contact.dissipated = model.contact.dissipated;
            w.contact.crush = model.contact.crush;
            for (a, b) in w.impactors.iter_mut().zip(&model.impactors) {
                a.velocity = b.velocity;
                a.angular_velocity = b.angular_velocity;
            }
            if variant == "pre_filtered" {
                w.advance_ballistic(fdt);
            } else {
                w.push_motion_to_engine();
                w.step_frame_engine_only();
            }
        }
        other => panic!("unknown --variant {other}"),
    }
}

fn run_ceiling(s: &Scene, variant: &str, tag: &str) -> Value {
    let mut w = World::with_engine(s, engine(s));
    let n = frames(s);
    let fdt = s.sim.frame_dt;
    let e0 = w.mechanical_energy();
    let mut series = Vec::with_capacity(n);
    let mut calls = 0;
    let t_all = Instant::now();
    for f in 0..n {
        let t = Instant::now();
        let islands = if variant == "coupled" { 0 } else { w.predict_islands(fdt, &HashSet::new()) };
        if islands > 0 {
            calls += 1;
            model_frame(&mut w, variant, fdt);
        } else {
            w.step_frame();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        series.push(frame_line(&w, ms, islands));
        eprintln!("[ceiling {tag}] frame {}/{n} islands {islands} ram vy {:.3} fragments {} broken {} ({ms:.0} ms)", f + 1, w.impactors[0].velocity.y, w.solver.clusters.len(), w.solver.broken_bond_count());
    }
    json!({"e0": e0, "calls": calls, "island_frames": w.island_frames, "redone_frames": w.redone_frames, "frames": series, "end": outcome(&w), "total_s": t_all.elapsed().as_secs_f64()})
}

/// Where the closed loop's outcomes come from (`ceiling --variant provider`).
struct ProviderOpts {
    /// `oracle` (in-process: the standalone oracle's frame, through the decoded-state path),
    /// `files` (an external process, `learned-impact/PROVIDER_PROTOCOL.md`), or `punch` (the
    /// host's predictive punch: its plug's bonds broken, then the engine takes the frame with
    /// its contacts, as the GPU punch does).
    kind: String,
    dir: std::path::PathBuf,
    timeout: f64,
    /// `files`: also write the oracle's own decoded state into each request (`oracle.json`),
    /// so a trivial provider (`scripts/echo_provider.py`) can check the loop end to end.
    echo_oracle: bool,
}

fn oracle_outcome(view: &World, keys: &stress_ref::world::IslandKeys, frame: i64) -> stress_ref::world::DecodedState {
    let mut model = view.standalone_copy();
    model.step_frame();
    view.decoded_from_oracle(&model, keys, frame, 0)
}

/// The frame-start state a file provider gets with its request until the call builder
/// (`stress-learned/src/outcome_calls.rs`) writes the `learned-impact-outcome/1` record:
/// each keyed body's world pose and motion.
fn request_state(view: &World, keys: &stress_ref::world::IslandKeys) -> Value {
    let sv = &view.solver;
    let chunks: Vec<Value> = keys
        .chunks
        .iter()
        .map(|&k| {
            let (s, c) = ((k >> 32) as usize, (k & 0xffff_ffff) as usize);
            let (v, w) = sv.chunk_velocity(s, c);
            let cl = &sv.clusters[sv.chunks[s][c].cluster];
            json!({"key": k, "position": sv.chunk_position(s, c).to_array(), "v": v.to_array(), "w": w.to_array(), "cluster": cl.id, "anchored": cl.anchored, "mass": sv.structures[s].chunks[c].mass})
        })
        .collect();
    let impactors: Vec<Value> = keys
        .impactors
        .iter()
        .map(|&k| {
            let i = &view.impactors[k as usize];
            json!({"key": k, "position": i.pose.position.to_array(), "orientation": [i.pose.rotation.w, i.pose.rotation.x, i.pose.rotation.y, i.pose.rotation.z], "v": i.velocity.to_array(), "w": i.angular_velocity.to_array(), "mass": i.mass})
        })
        .collect();
    json!({"chunks": chunks, "impactors": impactors, "live_bonds": keys.bonds})
}

/// One file-protocol call: write the request, wait for the response (polling, `timeout` s).
fn file_outcome(o: &ProviderOpts, view: &World, keys: &stress_ref::world::IslandKeys, frame: i64, scene: &str) -> Result<stress_ref::world::DecodedState, String> {
    let name = format!("{frame:06}_0");
    let req = o.dir.join("req").join(&name);
    std::fs::create_dir_all(&req).map_err(|e| e.to_string())?;
    let request = json!({"schema_version": "learned-impact-provider-request/0", "event_id": scene, "frame": frame, "island": 0, "time": view.solver.time,
        "frame_dt": view.scene.sim.frame_dt, "keys": keys, "state": request_state(view, keys), "call_record": null});
    std::fs::write(req.join("request.json"), serde_json::to_string(&request).unwrap()).map_err(|e| e.to_string())?;
    if o.echo_oracle {
        std::fs::write(req.join("oracle.json"), serde_json::to_string(&oracle_outcome(view, keys, frame)).unwrap()).map_err(|e| e.to_string())?;
    }
    std::fs::write(req.join("ready"), b"").map_err(|e| e.to_string())?;
    let resp = o.dir.join("resp").join(format!("{name}.json"));
    let t = Instant::now();
    loop {
        if let Ok(text) = std::fs::read_to_string(&resp) {
            return serde_json::from_str(&text).map_err(|e| format!("{}: {e}", resp.display()));
        }
        if t.elapsed().as_secs_f64() > o.timeout {
            return Err(format!("no response {} within {} s", resp.display(), o.timeout));
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// The closed loop: every routed frame's island outcome from the provider, applied at window
/// end (reduced state, the island's contacts filtered), to the end of the run.
fn run_provider(s: &Scene, o: &ProviderOpts, tag: &str) -> Value {
    let mut w = World::with_engine(s, engine(s));
    let n = frames(s);
    let fdt = s.sim.frame_dt;
    let e0 = w.mechanical_energy();
    let mut series = Vec::with_capacity(n);
    let mut log = Vec::new();
    let t_all = Instant::now();
    if o.kind == "files" {
        let _ = std::fs::remove_file(o.dir.join("done"));
        std::fs::create_dir_all(o.dir.join("resp")).expect("provider resp dir");
    }
    for f in 0..n {
        let t = Instant::now();
        // As the recorder does: each frame's peak utilisation starts afresh (the call
        // builder's `log1p_peak_util_prev` reads the previous frame's).
        w.solver.reset_peak_utilizations();
        let mut islands = 0;
        if o.kind == "punch" {
            islands = w.predict_islands(fdt, &HashSet::new());
            let bonds = if islands > 0 { w.predict_punch(fdt) } else { Vec::new() };
            if !bonds.is_empty() {
                w.solver.break_bonds(&bonds);
                w.push_motion_to_engine();
            }
            log.push(json!({"frame": f, "islands": islands, "punch_bonds": bonds.len()}));
            w.step_frame_engine_only();
        } else {
            let frame = f as i64;
            let mut call_ms = 0.0;
            let mut flags = Value::Null;
            let mut provider = |view: &World, keys: &stress_ref::world::IslandKeys| -> Result<stress_ref::world::DecodedState, String> {
                let tc = Instant::now();
                let d = match o.kind.as_str() {
                    "oracle" => Ok(oracle_outcome(view, keys, frame)),
                    "files" => file_outcome(o, view, keys, frame, &s.name),
                    other => Err(format!("unknown --provider {other}")),
                }?;
                call_ms = tc.elapsed().as_secs_f64() * 1e3;
                flags = d.flags.clone();
                Ok(d)
            };
            match w.step_frame_with_provider(&mut provider) {
                Ok(keys) => {
                    islands = keys.chunks.len() + keys.impactors.len();
                    if islands > 0 {
                        log.push(json!({"frame": f, "chunks": keys.chunks.len(), "bonds": keys.bonds.len(), "impactors": keys.impactors.len(), "call_ms": call_ms, "flags": flags}));
                    }
                }
                Err(e) => panic!("frame {f}: {e}"),
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        series.push(frame_line(&w, ms, islands));
        eprintln!("[provider {tag}] frame {}/{n} islands {islands} ram vy {:.3} fragments {} broken {} ({ms:.0} ms)", f + 1, w.impactors[0].velocity.y, w.solver.clusters.len(), w.solver.broken_bond_count());
    }
    if o.kind == "files" {
        let _ = std::fs::write(o.dir.join("done"), b"");
    }
    let calls = log.iter().filter(|l| l.get("chunks").is_some() || l.get("punch_bonds").is_some_and(|b| b.as_u64().unwrap_or(0) > 0)).count();
    json!({"e0": e0, "calls": calls, "provider": o.kind, "call_log": log, "island_frames": w.island_frames, "frames": series, "end": outcome(&w), "total_s": t_all.elapsed().as_secs_f64()})
}

/// The GPU world (rubble's engine and stress-gpu on one device), as stress_bench builds it:
/// `profile` a capability profile (`blast_parity`, `blast_parity_routed`) or `realtime`
/// (the scene's own configuration with impact islands, rewind-scaling.md's "rubble, no
/// rewind"). `STRESS_IMPACT_PUNCH=gpu` and `STRESS_IMPACT_WINDOW=s` select the punch and the
/// impact window as they do everywhere.
fn run_gpu(s: &Scene, profile: &str, tag: &str) -> Value {
    use stress_ref::capability::Profile;
    let gpu = stress_gpu::gpu::Gpu::new().expect("GPU");
    let mut scene = s.clone();
    let prof = Profile::parse(profile);
    let islands = match prof {
        Some(p) => {
            p.apply_scene(&mut scene);
            p.impact_islands()
        }
        None if profile == "realtime" => true,
        // Quasi-static everywhere with the full model's features, impacts as the engine's
        // contact loads, no islands (stress_bench's `--config qs`).
        None if profile == "qs" => {
            scene.sim.solve_mode = stress_ref::scene::SolveMode::QuasiStatic;
            false
        }
        None => panic!("unknown --profile {profile}"),
    };
    let cap = stress_rubble::capacity_for(&scene);
    let mut w = if prof.is_some_and(|p| p.resident()) {
        let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), cap).expect("rubble engine");
        let (iterations, substeps) = prof.unwrap().engine_policy();
        e.set_policy(iterations, substeps);
        stress_gpu::world::GpuWorld::with_resident_engine(&gpu, &scene, e).expect("GPU world")
    } else {
        let e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), cap).expect("rubble engine");
        stress_gpu::world::GpuWorld::with_engine(&gpu, &scene, Box::new(e)).expect("GPU world")
    };
    w.redo_fractured_frames = false;
    w.set_impact_islands(islands);
    if let Some(p) = prof {
        w.apply_profile(p);
    }
    if islands && scene.sim.solve_mode == stress_ref::scene::SolveMode::QuasiStatic {
        w.set_tiered(true).expect("tiered coupling");
    }
    let n = frames(&scene);
    let mut series = Vec::with_capacity(n);
    let t_all = Instant::now();
    for f in 0..n {
        let t = Instant::now();
        w.step_frame(&gpu).expect("frame");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let vy = w.solver.impactors[0].velocity.y;
        series.push(json!({"ms": ms, "ram_vy": vy}));
        eprintln!("[gpu {tag}] frame {}/{n} ram vy {vy:.3} ({ms:.0} ms)", f + 1);
    }
    w.sync(&gpu);
    let mut frames_out = series;
    // The frame series has no per-frame energy on the GPU path.
    for f in &mut frames_out {
        f.as_object_mut().unwrap().insert("fragments".into(), json!(null));
    }
    let end = outcome_of(&w.solver.mirror, &w.solver.impactors);
    json!({"e0": 0.5 * s.impactors[0].mass * Vec3::from_array(s.impactors[0].velocity).norm2(), "frames": frames_out, "end": end, "total_s": t_all.elapsed().as_secs_f64(),
        "profile": profile, "punch": std::env::var("STRESS_IMPACT_PUNCH").ok(), "window": std::env::var("STRESS_IMPACT_WINDOW").ok(), "coupled_frames": w.coupled_frames})
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().expect("oracle|ceiling|hash");
    let (mut name, mut member, mut mode, mut seconds, mut fdt, mut variant) = (String::from("free_wall"), None::<u64>, String::from("kin"), SECONDS, None::<f64>, String::from("coupled"));
    let mut provider = ProviderOpts { kind: "oracle".into(), dir: "provider".into(), timeout: 60.0, echo_oracle: false };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--scene" => name = args.next().expect("--scene NAME"),
            "--member" => member = args.next().and_then(|v| v.parse().ok()),
            "--perturb" => mode = args.next().expect("--perturb kin|kin_weibull"),
            "--seconds" => seconds = args.next().and_then(|v| v.parse().ok()).expect("--seconds S"),
            "--frame-dt" => fdt = args.next().map(|v| if let Some(d) = v.strip_prefix("1/") { 1.0 / d.parse::<f64>().unwrap() } else { v.parse().unwrap() }),
            "--variant" | "--profile" => variant = args.next().expect("--variant NAME"),
            "--provider" => provider.kind = args.next().expect("--provider oracle|files|punch"),
            "--dir" => provider.dir = args.next().expect("--dir DIR").into(),
            "--timeout" => provider.timeout = args.next().and_then(|v| v.parse().ok()).expect("--timeout S"),
            "--echo-oracle" => provider.echo_oracle = true,
            other => panic!("unknown argument {other}"),
        }
    }
    if cmd == "bonds" {
        // Each scene's bond and chunk geometry (body frame = world at rest) and its ram, for
        // the per-bond baseline's features.
        for n in SCENES {
            let s = scene(n, SECONDS);
            let w = World::new(&s);
            let sv = &w.solver;
            let walls: Vec<Value> = sv
                .structures
                .iter()
                .enumerate()
                .map(|(si, st)| {
                    let bonds: Vec<Value> = sv.bonds[si]
                        .iter()
                        .map(|b| {
                            let g = &b.geometry;
                            json!({"a": g.a, "b": g.b, "centroid": g.centroid.to_array(), "normal": g.normal.to_array(), "area": g.area,
                                "tensile": b.strength.tensile, "g_tension": b.strength.g_tension})
                        })
                        .collect();
                    let chunks: Vec<Value> = st.chunks.iter().map(|c| json!({"center": c.center.to_array(), "mass": c.mass, "half": c.half_extents.to_array(), "support": c.support != stress_ref::scene::Support::None})).collect();
                    json!({"name": st.name, "bonds": bonds, "chunks": chunks})
                })
                .collect();
            let i = &s.impactors[0];
            println!("{}", json!({"scene": n, "ram": {"position": i.position, "velocity": i.velocity, "mass": i.mass, "shape": format!("{:?}", i.shape)}, "gravity": s.gravity, "walls": walls}));
        }
        return;
    }
    if cmd == "hash" {
        for n in SCENES {
            let s = scene(n, SECONDS);
            println!("{}", json!({"scene": n, "hash": scene_hash(&s), "chunks": s.bodies.iter().map(|b| b.chunks.len()).sum::<usize>(), "bonds": s.bodies.iter().map(|b| b.bonds.len()).sum::<usize>()}));
        }
        return;
    }
    let mut s = scene(&name, seconds);
    if let Some(d) = fdt {
        s.sim.frame_dt = d;
    }
    let pert = member.map(|k| perturb(&mut s, k, &mode));
    let tag = format!("{name}{}", member.map(|k| format!(" m{k}")).unwrap_or_default());
    let mut out = match cmd.as_str() {
        "oracle" => run_oracle(&s, &tag),
        "ceiling" if variant == "provider" => run_provider(&s, &provider, &format!("{tag} {}", provider.kind)),
        "ceiling" => run_ceiling(&s, &variant, &format!("{tag} {variant}")),
        "gpu" => run_gpu(&s, &variant, &format!("{tag} {variant}")),
        "zero_break" => {
            // No fracture at all: the zero-break baseline (the oracle with fracture off).
            s.sim.fracture = false;
            run_oracle(&s, &format!("{tag} zero-break"))
        }
        other => panic!("unknown command {other}"),
    };
    let o = out.as_object_mut().unwrap();
    o.insert("cmd".into(), json!(cmd));
    o.insert("scene".into(), json!(name));
    o.insert("scene_hash".into(), json!(scene_hash(&s)));
    o.insert("frame_dt".into(), json!(s.sim.frame_dt));
    o.insert("seconds".into(), json!(s.sim.duration));
    o.insert("perturbation".into(), pert.unwrap_or(Value::Null));
    if cmd == "ceiling" || cmd == "gpu" {
        o.insert("variant".into(), json!(if variant == "provider" { format!("provider_{}", provider.kind) } else { variant.clone() }));
    }
    println!("{}", serde_json::to_string(&out).unwrap());
}

//! P0 of the learned impact-outcome plan (`docs/learned-stress-solver/learned-stress-solver-plan.md`):
//! the oracle pin, the chaos floor and the perfect-model ceiling, one JSON line per run.
//!
//!   cargo run --release -p stress-rubble --features gpu --example learned_p0 -- \
//!       oracle  --scene free_wall [--member K --perturb kin|kin_weibull] [--seconds S] [--frame-dt DT]
//!       ceiling --scene free_wall --variant coupled|window_full|window_reduced|pre_filtered|pre_engine [--frame-dt DT]
//!       ceiling --scene free_wall --variant provider --provider oracle|files|model|punch [--dir D --timeout S --echo-oracle --no-qs --checkpoint PT --device cpu|mps]
//!       hash    (every scene's canonical hash)
//!   Any run: --poses FILE writes every body's pose at every frame boundary (JSONL, `poses_open`).
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

/// `--poses FILE`: every body's pose at every frame boundary, for offline rendering (one
/// JSON line per frame, after a header line with the static shapes). A chunk's orientation is
/// its own (the cluster's rotation and the chunk's hidden rotation); `routed` lists the chunk
/// keys a provider's call covered in the frame that ended there (provider runs only; the
/// header's `routed_known` says whether it is filled).
struct Poses {
    out: std::io::BufWriter<std::fs::File>,
}

static POSES: std::sync::Mutex<Option<Poses>> = std::sync::Mutex::new(None);

fn poses_open(path: &str, s: &Scene, cmd: &str, variant: &str, routed_known: bool) {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).expect("--poses file"));
    let w = World::new(s);
    let sv = &w.solver;
    let chunks: Vec<Value> = sv
        .structures
        .iter()
        .enumerate()
        .flat_map(|(si, st)| st.chunks.iter().enumerate().map(move |(c, ch)| json!({"key": stress_ref::keys::chunk_key(si, c), "half_extents": ch.half_extents.to_array(), "mass": ch.mass})))
        .collect();
    let impactors: Vec<Value> = s
        .impactors
        .iter()
        .enumerate()
        .map(|(i, d)| json!({"key": stress_ref::keys::impactor_key(i), "name": d.name, "shape": match d.shape { stress_ref::scene::ImpactorShape::Sphere { radius } => json!({"sphere": {"radius": radius}}), stress_ref::scene::ImpactorShape::Box { half_extents } => json!({"box": {"half_extents": half_extents}}) }, "mass": d.mass}))
        .collect();
    let header = json!({"schema_version": "learned-p0-poses/1", "scene": s.name, "scene_hash": scene_hash(s), "cmd": cmd, "variant": variant, "frame_dt": s.sim.frame_dt,
        "routed_known": routed_known, "chunks": chunks, "impactors": impactors});
    writeln!(out, "{header}").unwrap();
    *POSES.lock().unwrap() = Some(Poses { out });
}

/// One frame boundary's poses (no-op without `--poses`).
fn poses_frame(frame: usize, sv: &stress_ref::solver::ReferenceSolver, impactors: &[stress_ref::world::Impactor], routed: &[i64]) {
    use std::io::Write;
    let mut g = POSES.lock().unwrap();
    let Some(p) = g.as_mut() else { return };
    let mut chunks = Vec::new();
    for (si, st) in sv.chunks.iter().enumerate() {
        for (c, cs) in st.iter().enumerate() {
            if !cs.active {
                continue;
            }
            let cl = &sv.clusters[cs.cluster];
            let q = (cl.pose.rotation * stress_ref::math::Quat::from_rotation_vector(cs.th)).normalized();
            let x = sv.chunk_position(si, c);
            chunks.push(json!([stress_ref::keys::chunk_key(si, c), [x.x, x.y, x.z], [q.w, q.x, q.y, q.z], cl.id]));
        }
    }
    let imps: Vec<Value> = impactors
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let (x, q, v) = (m.pose.position, m.pose.rotation, m.velocity);
            json!({"key": stress_ref::keys::impactor_key(i), "position": [x.x, x.y, x.z], "orientation": [q.w, q.x, q.y, q.z], "velocity": [v.x, v.y, v.z]})
        })
        .collect();
    let line = json!({"frame": frame, "t": sv.time, "chunks_columns": ["key", "position", "orientation_wxyz", "cluster"], "chunks": chunks, "impactors": imps, "routed": routed});
    writeln!(p.out, "{line}").unwrap();
}

fn poses_close() {
    use std::io::Write;
    if let Some(mut p) = POSES.lock().unwrap().take() {
        p.out.flush().unwrap();
    }
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
    poses_frame(0, &w.solver, &w.impactors, &[]);
    for f in 0..n {
        let t = Instant::now();
        w.step_frame();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        poses_frame(f + 1, &w.solver, &w.impactors, &[]);
        series.push(frame_line(&w, ms, 0));
        eprintln!("[oracle {tag}] frame {}/{n} t {:.4} ram vy {:.3} fragments {} broken {} ({ms:.0} ms)", f + 1, w.solver.time, w.impactors[0].velocity.y, w.solver.clusters.len(), w.solver.broken_bond_count());
    }
    json!({"e0": e0, "frames": series, "end": outcome(&w), "total_s": t_all.elapsed().as_secs_f64()})
}

/// The rubble engine of the coupled runs. `STRESS_P0_ENGINE_POLICY=production`: production's
/// solver policy (`Profile::engine_policy` of `blast_parity_routed`: 4 iterations, one step a
/// frame), the coupled reference's; unset: the engine's own defaults.
fn engine(s: &Scene) -> Box<stress_rubble::RubbleEngine> {
    let mut e = stress_rubble::RubbleEngine::new(Vec3::from_array(s.gravity)).expect("rubble engine");
    if std::env::var("STRESS_P0_ENGINE_POLICY").is_ok_and(|v| v == "production") {
        let (iterations, substeps) = stress_ref::capability::Profile::BlastParityRouted.engine_policy();
        e.set_policy(iterations, substeps);
    }
    Box::new(e)
}

/// What a coupled run's numbers depend on besides the scene: the engine policy, the routing,
/// and the GPU (rubble's results are deterministic on one adapter and driver, not across).
fn engine_context() -> Value {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let info = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
        .ok()
        .map(|a| a.get_info());
    let policy = if std::env::var("STRESS_P0_ENGINE_POLICY").is_ok_and(|v| v == "production") {
        let (i, n) = stress_ref::capability::Profile::BlastParityRouted.engine_policy();
        json!({"name": "production", "iterations": i, "max_substeps": n})
    } else {
        json!({"name": "engine_default"})
    };
    json!({
        "engine_policy": policy,
        "routing_tiered": std::env::var("STRESS_ROUTING_TIERED").is_ok_and(|v| v == "1"),
        "debris_routing": std::env::var("STRESS_DEBRIS_ROUTING").is_ok_and(|v| v == "1"),
        "handed_ground_overlaps": std::env::var("STRESS_HANDED_INDENTS").is_ok_and(|v| v == "1"),
        "adapter": info.as_ref().map(|i| json!({"name": i.name, "backend": format!("{:?}", i.backend), "driver": i.driver, "driver_info": i.driver_info, "vendor": i.vendor, "device": i.device})),
        // Metal reports no driver string: the OS build carries the GPU driver on macOS.
        "os": json!({"os": std::env::consts::OS, "version": std::process::Command::new("sw_vers").arg("-buildVersion").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())}),
    })
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
    poses_frame(0, &w.solver, &w.impactors, &[]);
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
        poses_frame(f + 1, &w.solver, &w.impactors, &[]);
        series.push(frame_line(&w, ms, islands));
        eprintln!("[ceiling {tag}] frame {}/{n} islands {islands} ram vy {:.3} fragments {} broken {} ({ms:.0} ms)", f + 1, w.impactors[0].velocity.y, w.solver.clusters.len(), w.solver.broken_bond_count());
    }
    json!({"e0": e0, "calls": calls, "island_frames": w.island_frames, "redone_frames": w.redone_frames, "frames": series, "end": outcome(&w), "total_s": t_all.elapsed().as_secs_f64()})
}

/// Where the closed loop's outcomes come from (`ceiling --variant provider`).
struct ProviderOpts {
    /// `oracle` (in-process: the standalone oracle's frame, through the decoded-state path),
    /// `files` (an external process, `learned-impact/PROVIDER_PROTOCOL.md`), `model` (P2's
    /// checkpoint, through `stress_learned::provider_client`, the same protocol), or `punch` (the
    /// host's predictive punch: its plug's bonds broken, then the engine takes the frame with
    /// its contacts, as the GPU punch does).
    kind: String,
    dir: std::path::PathBuf,
    timeout: f64,
    /// `files`: also write the oracle's own decoded state into each request (`oracle.json`),
    /// so a trivial provider (`scripts/echo_provider.py`) can check the loop end to end.
    echo_oracle: bool,
    /// `model`: the trained checkpoint, run by `stress_learned::provider_client` (it spawns
    /// `python -m learned_impact.outcome.provider` on `dir` and owns it).
    checkpoint: Option<std::path::PathBuf>,
    device: String,
    /// `model`: the Python package directory with its `.venv` (default: this checkout's
    /// `learned-impact/python`).
    python_dir: Option<std::path::PathBuf>,
    /// `--host standalone`: the closed loop on the oracle's own world (no rigid engine:
    /// `World::step_frame_with_provider_standalone`, diagnostic only: no contact outside the
    /// island in routed frames); default `engine` (rubble).
    standalone: bool,
    /// `--host a`: harness A, the standalone oracle with each routed island's frame through
    /// `World::step_frame_island_provider` (the island's full state across the boundary; the
    /// rest the oracle's native step). Provider `oracle`: the oracle's island step itself
    /// (the standalone oracle bit for bit); `oracle_decoded`: the oracle's outcome through the
    /// decoded state (the /3 boundary in harness A); `model`/`files`: their decoded state,
    /// applied to the island on a copy of the frame-start world.
    harness_a: bool,
    /// The call record's variant-B (QS frame) features (`live_call`'s `qs`): `--qs` / `--no-qs`
    /// force them. Unset, `files` builds them (as P1 records them) and `model` asks the
    /// provider (`model.json`'s `needs_qs`).
    qs: Option<bool>,
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
    // The call record (`learned-impact-outcome/2`, P1's call builder): the island's inputs
    // captured exactly as the recorder captures them, from a standalone copy of the
    // frame-start state.
    let mut standalone = view.standalone_copy();
    let ev = stress_learned::outcome_calls::live_call(&mut standalone, scene, &json!({}), &keys.chunks, &keys.impactors, frame as usize, o.qs.unwrap_or(true))?;
    stress_learned::outcome_calls::write_event(&req.join("call"), &ev, true)?;
    let request = json!({"schema_version": "learned-impact-provider-request/0", "event_id": scene, "frame": frame, "island": 0, "time": view.solver.time,
        "frame_dt": view.scene.sim.frame_dt, "keys": keys, "state": request_state(view, keys), "call_record": "call"});
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
    run_provider_world(World::with_engine(s, engine(s)), s, frames(s), o, tag, &mut |_, _| {})
}

/// `run_provider` on a given coupled world for `n` frames, `hook(world, frame)` after each
/// frame boundary (frame 0: the start).
fn run_provider_world(mut w: World, s: &Scene, n: usize, o: &ProviderOpts, tag: &str, hook: &mut dyn FnMut(&World, usize)) -> Value {
    let fdt = s.sim.frame_dt;
    let e0 = w.mechanical_energy();
    let mut series = Vec::with_capacity(n);
    let mut log = Vec::new();
    let t_all = Instant::now();
    let mut model = (o.kind == "model").then(|| {
        let mut spec = stress_learned::provider_client::ModelSpec::new(o.checkpoint.clone().expect("--provider model needs --checkpoint"));
        spec.device = o.device.clone();
        spec.timeout = o.timeout;
        spec.qs = o.qs;
        if let Some(d) = &o.python_dir {
            spec.python_dir = d.clone();
        }
        stress_learned::provider_client::ModelProvider::spawn(&spec, &o.dir, &s.name).expect("model provider")
    });
    if o.kind == "files" {
        let _ = std::fs::remove_file(o.dir.join("done"));
        std::fs::create_dir_all(o.dir.join("resp")).expect("provider resp dir");
    }
    poses_frame(0, &w.solver, &w.impactors, &[]);
    hook(&w, 0);
    for f in 0..n {
        let t = Instant::now();
        // (`step_frame_with_provider` resets each frame's peak utilisation after the
        // frame-start state the call reads; the punch path resets it here.)
        if o.kind == "punch" {
            w.solver.reset_peak_utilizations();
        }
        let mut islands = 0;
        let mut routed: Vec<i64> = Vec::new();
        if o.kind == "oracle_full" {
            // Diagnostic: the oracle's frame with its full state carried (hidden deformation
            // kept between calls, `window_full`), to separate the reduced state's effect.
            islands = w.predict_islands(fdt, &HashSet::new());
            if islands > 0 {
                model_frame(&mut w, "window_full", fdt);
                log.push(json!({"frame": f, "chunks": islands, "full_state": true}));
            } else {
                w.step_frame();
            }
        } else if o.kind == "punch" {
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
                    "oracle" | "oracle_decoded" => Ok(oracle_outcome(view, keys, frame)),
                    "files" => file_outcome(o, view, keys, frame, &s.name),
                    "model" => model.as_mut().expect("model provider").outcome(view, keys, frame),
                    other => Err(format!("unknown --provider {other}")),
                }?;
                call_ms = tc.elapsed().as_secs_f64() * 1e3;
                flags = d.flags.clone();
                Ok(d)
            };
            let stepped = if o.harness_a {
                let exact = o.kind == "oracle";
                let mut a_provider = |start: &World, m: &stress_ref::world::IslandMembers| -> Result<stress_ref::world::IslandState, String> {
                    if exact {
                        return stress_ref::world::oracle_island_step(start, m);
                    }
                    // A decoded state: applied to the island on a copy of the frame-start world
                    // (the island's other state carried as it was), the island's state taken.
                    let mut view = start.clone();
                    let island: HashSet<u64> = view.solver.clusters.iter().filter(|c| c.chunks.first().is_some_and(|&k| m.chunks.contains(&(c.structure, k)))).map(|c| c.id).collect();
                    for cl in &mut view.solver.clusters {
                        cl.driven = !island.contains(&cl.id);
                    }
                    for (i, imp) in view.impactors.iter_mut().enumerate() {
                        imp.driven = !m.impactors.contains(&i);
                    }
                    let keys = view.routed_island_keys();
                    let d = provider(&view, &keys)?;
                    let mut x = start.clone();
                    x.apply_decoded(&d)?;
                    Ok(x.extract_island(m))
                };
                w.step_frame_island_provider(&mut a_provider).map(|m| {
                    let mut chunks: Vec<i64> = m.chunks.iter().map(|&(s, c)| stress_ref::world::stable_key(s, c)).collect();
                    chunks.sort_unstable();
                    stress_ref::world::IslandKeys { chunks, impactors: m.impactors.iter().map(|&i| i as i64).collect(), ..Default::default() }
                })
            } else if o.standalone {
                w.step_frame_with_provider_standalone(&mut provider)
            } else {
                w.step_frame_with_provider(&mut provider)
            };
            match stepped {
                Ok(keys) => {
                    islands = keys.chunks.len() + keys.impactors.len();
                    routed = keys.chunks.clone();
                    if islands > 0 {
                        log.push(json!({"frame": f, "chunks": keys.chunks.len(), "bonds": keys.bonds.len(), "impactors": keys.impactors.len(), "call_ms": call_ms, "flags": flags}));
                    }
                }
                Err(e) => panic!("frame {f}: {e}"),
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        poses_frame(f + 1, &w.solver, &w.impactors, &routed);
        hook(&w, f + 1);
        series.push(frame_line(&w, ms, islands));
        eprintln!("[provider {tag}] frame {}/{n} islands {islands} ram vy {:.3} fragments {} broken {} ({ms:.0} ms)", f + 1, w.impactors[0].velocity.y, w.solver.clusters.len(), w.solver.broken_bond_count());
        if std::env::var("STRESS_P0_WATCH_RAM").is_ok_and(|v| v == "1") {
            let imp = &w.impactors[0];
            let gz = w.scene.ground.as_ref().map(|g| g.height).unwrap_or(f64::NEG_INFINITY);
            eprintln!("  ram v {:?} position {:?} ground {gz}", imp.velocity.to_array(), imp.pose.position.to_array());
        }
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
    let posing = POSES.lock().unwrap().is_some();
    if posing {
        w.sync(&gpu);
        poses_frame(0, &w.solver.mirror, &w.solver.impactors, &[]);
    }
    for f in 0..n {
        let t = Instant::now();
        w.step_frame(&gpu).expect("frame");
        if posing {
            w.sync(&gpu);
            poses_frame(f + 1, &w.solver.mirror, &w.solver.impactors, &[]);
        }
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

/// What the closed-loop evaluation needs of one run of a recorded event
/// (`learned_impact.closed_loop_eval`): the physics checks every frame, chunk states at
/// the sample frames, and the end state by stable key.
struct EventTrack {
    e0: f64,
    pe0: f64,
    p0: Vec3,
    free: bool,
    samples: Vec<usize>,
    sampled: Vec<Value>,
    /// Largest (KE + PE - PE0 - E0) / E0 over the run, and its frame.
    ke_excess: (f64, usize),
    /// Largest decrease of any bond's damage between frames, and non-finite frames.
    damage_drop: f64,
    nonfinite: usize,
    damage: Vec<Vec<f64>>,
    /// (KE + PE - PE0 - E0) / E0 at every frame boundary, and the impactors' KE / E0.
    energy: Vec<(f64, f64)>,
}

fn total_ke_pe(w: &World) -> (f64, f64, Vec3) {
    let g = w.solver.config.gravity;
    let sv = &w.solver;
    let (mut ke, mut pe, mut p) = (sv.kinetic_energy(), 0.0, Vec3::ZERO);
    for (si, st) in sv.structures.iter().enumerate() {
        for (c, ch) in st.chunks.iter().enumerate() {
            if sv.chunks[si][c].active {
                pe -= ch.mass * g.dot(sv.chunk_position(si, c));
                p += sv.chunk_velocity(si, c).0 * ch.mass;
            }
        }
    }
    for i in &w.impactors {
        ke += i.kinetic_energy();
        pe -= i.mass * g.dot(i.pose.position);
        p += i.velocity * i.mass;
    }
    (ke, pe, p)
}

fn chunk_rows(w: &World) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let sv = &w.solver;
    let (mut x, mut v) = (Vec::new(), Vec::new());
    for (si, st) in sv.chunks.iter().enumerate() {
        for c in 0..st.len() {
            let (p, vel) = if st[c].active { (sv.chunk_position(si, c), sv.chunk_velocity(si, c).0) } else { (Vec3::splat(f64::NAN), Vec3::splat(f64::NAN)) };
            x.push([p.x as f32, p.y as f32, p.z as f32]);
            v.push([vel.x as f32, vel.y as f32, vel.z as f32]);
        }
    }
    (x, v)
}

impl EventTrack {
    fn new(w: &World, e0: f64, samples: Vec<usize>) -> EventTrack {
        let (_, pe0, p0) = total_ke_pe(w);
        let anchored = w.solver.clusters.iter().any(|c| c.anchored);
        let free = !anchored && w.solver.config.gravity.norm() == 0.0;
        let damage = w.solver.bonds.iter().map(|b| b.iter().map(|x| x.joint.damage).collect()).collect();
        EventTrack { e0, pe0, p0, free, samples, sampled: Vec::new(), ke_excess: (f64::NEG_INFINITY, 0), damage_drop: 0.0, nonfinite: 0, damage, energy: Vec::new() }
    }

    fn frame(&mut self, w: &World, f: usize) {
        let (ke, pe, _) = total_ke_pe(w);
        let excess = (ke + pe - self.pe0 - self.e0) / self.e0;
        self.energy.push((excess, w.impactors.iter().map(|i| i.kinetic_energy()).sum::<f64>() / self.e0));
        if !(excess.is_finite()) {
            self.nonfinite += 1;
        } else if excess > self.ke_excess.0 {
            self.ke_excess = (excess, f);
        }
        for (s, bonds) in w.solver.bonds.iter().enumerate() {
            for (b, x) in bonds.iter().enumerate() {
                self.damage_drop = self.damage_drop.max(self.damage[s][b] - x.joint.damage);
                self.damage[s][b] = x.joint.damage;
            }
        }
        if self.samples.contains(&f) {
            let (x, v) = chunk_rows(w);
            if x.iter().chain(&v).flatten().any(|z| z.is_infinite()) {
                self.nonfinite += 1;
            }
            self.sampled.push(json!({"frame": f, "t": w.solver.time, "chunk_pos": x, "chunk_vel": v}));
        }
    }

    fn finish(self, w: &World, stop_speed: f64) -> Value {
        let sv = &w.solver;
        let (_, _, p1) = total_ke_pe(w);
        let mut cluster = Vec::new();
        let mut anchored = Vec::new();
        for st in &sv.chunks {
            for cs in st {
                cluster.push(if cs.active { sv.clusters[cs.cluster].id as i64 } else { -1 });
                anchored.push(cs.active && sv.clusters[cs.cluster].anchored);
            }
        }
        // Regimes as P1's recorder labels them (`outcome.rs`, relative to the remnant).
        let remnant = sv.clusters.iter().max_by(|a, b| a.mass.total_cmp(&b.mass)).expect("a cluster");
        let impactors: Vec<Value> = w
            .impactors
            .iter()
            .enumerate()
            .map(|(i, imp)| {
                let v0 = Vec3::from_array(w.scene.impactors[i].velocity);
                let d = v0.normalized();
                let c = remnant.com_world();
                let s = remnant.structure;
                let far = remnant.chunks.iter().map(|&k| (sv.chunk_position(s, k) - c).dot(d) + sv.structures[s].chunks[k].half_extents.norm()).fold(f64::NEG_INFINITY, f64::max);
                let behind = (imp.pose.position - c).dot(d);
                let rel = (imp.velocity - remnant.velocity).dot(d);
                let along = imp.velocity.dot(d);
                let regime = if behind > far && rel > -stop_speed && along > stop_speed {
                    "perforate"
                } else if rel < -stop_speed {
                    "rebound"
                } else if rel.abs() <= stop_speed.max(0.05 * v0.norm()) {
                    "arrest"
                } else {
                    "deflect"
                };
                json!({"key": stress_ref::keys::impactor_key(i), "v_end": imp.velocity.to_array(), "position_end": imp.pose.position.to_array(), "speed_along_v0": along, "regime": regime,
                    "finite": imp.velocity.to_array().iter().chain(imp.pose.position.to_array().iter()).all(|z| z.is_finite())})
            })
            .collect();
        let momentum = if self.free { Some((p1 - self.p0).norm() / self.p0.norm().max(1e-12)) } else { None };
        json!({"t_end": sv.time, "chunk_cluster": cluster, "chunk_anchored": anchored, "impactors": impactors, "fragments": sv.clusters.len(), "broken_bonds": sv.broken_bond_count(),
            "energy_series": self.energy.iter().map(|e| [e.0, e.1]).collect::<Vec<_>>(),
            "checks": {"ke_excess_over_e0": self.ke_excess.0, "ke_excess_frame": self.ke_excess.1, "momentum_rel": momentum, "damage_drop": self.damage_drop, "nonfinite": self.nonfinite},
            "samples": self.sampled})
    }
}

/// `learned_p0 event --event-dir D`: one recorded P1 event (its `scene.json`, the member as
/// recorded) replayed for the oracle's recorded frames under a candidate: `--variant oracle`
/// (the standalone oracle, which reproduces the record), `--variant qs` (CPU
/// `blast_parity`, P1's QS baseline), or `--variant provider --provider ...` (the closed
/// loop). Samples chunk states at the frames `--samples f,f,...`.
fn run_event(dir: &std::path::Path, variant: &str, o: &ProviderOpts, samples: Vec<usize>) -> Value {
    let read = |name: &str| -> Value { serde_json::from_str(&std::fs::read_to_string(dir.join(name)).unwrap_or_else(|e| panic!("{}/{name}: {e}", dir.display()))).expect("json") };
    let mut scene: Scene = serde_json::from_value(read("scene.json")).expect("scene.json");
    let outcome = read("outcome.json");
    let context = read("context.json");
    let n = outcome["frames"].as_u64().expect("outcome frames") as usize;
    let e0 = outcome["input_energy"].as_f64().expect("input energy");
    let stop_speed = context["stop_rule"]["stop_speed"].as_f64().unwrap_or(0.05);
    scene.sim.duration = n as f64 * scene.sim.frame_dt;
    let tag = scene.name.clone();
    let t_all = Instant::now();
    let (track, extra) = match variant {
        "oracle" | "qs" => {
            if variant == "qs" {
                stress_ref::capability::Profile::BlastParity.apply_scene(&mut scene);
            }
            let mut w = World::new(&scene);
            if variant == "qs" {
                stress_ref::capability::Profile::BlastParity.apply_config(&mut w.solver.config);
            }
            let mut track = EventTrack::new(&w, e0, samples);
            track.frame(&w, 0);
            for f in 0..n {
                w.step_frame();
                track.frame(&w, f + 1);
                if (f + 1) % 30 == 0 {
                    eprintln!("[event {tag} {variant}] frame {}/{n}", f + 1);
                }
            }
            let fp = stress_learned::outcome::fingerprint(&w);
            (track.finish(&w, stop_speed), json!({"fingerprint": fp, "recorded_fingerprint": outcome["fingerprint"]}))
        }
        "coupled" => {
            // The oracle's own coupled frame (impact islands simulated through the frame,
            // rubble stepping every other body): no call boundary at all, the coupling alone.
            let mut w = World::with_engine(&scene, engine(&scene));
            let mut track = EventTrack::new(&w, e0, samples);
            track.frame(&w, 0);
            for f in 0..n {
                w.step_frame();
                track.frame(&w, f + 1);
                if (f + 1) % 30 == 0 {
                    eprintln!("[event {tag} coupled] frame {}/{n} island frames {}", f + 1, w.island_frames);
                }
            }
            (track.finish(&w, stop_speed), json!({"island_frames": w.island_frames}))
        }
        "provider" => {
            let w = if o.standalone { World::new(&scene) } else { World::with_engine(&scene, engine(&scene)) };
            let mut track = Some(EventTrack::new(&w, e0, samples));
            let mut last: Option<World> = None;
            let out = run_provider_world(w, &scene, n, o, &tag, &mut |w, f| {
                track.as_mut().unwrap().frame(w, f);
                if f == n {
                    last = Some(w.clone());
                }
            });
            let end = track.take().unwrap().finish(last.as_ref().expect("last frame"), stop_speed);
            let flags: Vec<&Value> = out["call_log"].as_array().map(|l| l.iter().filter_map(|x| x.get("flags")).collect()).unwrap_or_default();
            let count = |k: &str| flags.iter().filter(|f| f.get(k).and_then(|v| v.as_bool()).unwrap_or(false)).count();
            let call_ms: Vec<f64> = out["call_log"].as_array().map(|l| l.iter().filter_map(|x| x["call_ms"].as_f64()).collect()).unwrap_or_default();
            (end, json!({"calls": out["calls"], "fallbacks": count("fallback"), "energy_bound_active": count("energy_bound_active"), "energy_bound_infeasible": count("energy_bound_infeasible"), "call_ms": call_ms}))
        }
        other => panic!("event: unknown --variant {other}"),
    };
    json!({"cmd": "event", "event_id": outcome["event_id"], "family": outcome["family"], "variant": if variant == "provider" { format!("provider_{}", o.kind) } else { variant.to_string() },
        "frames": n, "e0": e0, "run": track, "extra": extra, "total_s": t_all.elapsed().as_secs_f64(),
        "engine": if variant == "oracle" || variant == "qs" { Value::Null } else { engine_context() }})
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().expect("oracle|ceiling|gpu|event|hash");
    let mut event_dir: Option<String> = None;
    let mut samples: Vec<usize> = Vec::new();
    let (mut name, mut member, mut mode, mut seconds, mut fdt, mut variant) = (String::from("free_wall"), None::<u64>, String::from("kin"), SECONDS, None::<f64>, String::from("coupled"));
    let mut poses: Option<String> = None;
    let mut provider = ProviderOpts { kind: "oracle".into(), dir: "provider".into(), timeout: 60.0, echo_oracle: false, qs: None, checkpoint: None, device: "cpu".into(), python_dir: None, standalone: false, harness_a: false };
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
            "--no-qs" => provider.qs = Some(false),
            "--qs" => provider.qs = Some(true),
            "--checkpoint" => provider.checkpoint = Some(args.next().expect("--checkpoint PT").into()),
            "--device" => provider.device = args.next().expect("--device cpu|mps"),
            "--python-dir" => provider.python_dir = Some(args.next().expect("--python-dir DIR").into()),
            "--host" => match args.next().as_deref() {
                Some("standalone") => provider.standalone = true,
                Some("engine") => provider.standalone = false,
                Some("a") => (provider.standalone, provider.harness_a) = (true, true),
                other => panic!("--host standalone|engine|a, not {other:?}"),
            },
            "--poses" => poses = Some(args.next().expect("--poses FILE")),
            "--event-dir" => event_dir = Some(args.next().expect("--event-dir DIR")),
            "--samples" => samples = args.next().expect("--samples f,f").split(',').filter(|x| !x.is_empty()).map(|x| x.parse().expect("frame")).collect(),
            other => panic!("unknown argument {other}"),
        }
    }
    if cmd == "event" {
        let dir = std::path::PathBuf::from(event_dir.expect("event needs --event-dir"));
        println!("{}", serde_json::to_string(&run_event(&dir, &variant, &provider, samples)).unwrap());
        return;
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
    if let Some(p) = &poses {
        let v = match cmd.as_str() {
            "ceiling" if variant == "provider" => format!("provider_{}", provider.kind),
            "oracle" | "zero_break" => cmd.clone(),
            _ => variant.clone(),
        };
        poses_open(p, &s, &cmd, &v, cmd == "ceiling" && variant == "provider" && provider.kind != "punch");
    }
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
    poses_close();
    println!("{}", serde_json::to_string(&out).unwrap());
}

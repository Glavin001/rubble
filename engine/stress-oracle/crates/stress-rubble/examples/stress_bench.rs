//! Destruction benchmark: the same regimes through each stress pipeline, every frame
//! timed end to end, the outcome recorded. One JSON line per (solver, regime) on stdout.
//!
//! Solvers:
//!   oracle       the stress-ref standalone world (CPU f64, its own rigid bodies)
//!   rubble-gpu   rubble's integrated pipeline: rubble's AVBD engine and stress-gpu on
//!                one device, impact islands routed before the step, no frame rewind
//!   rubble-gpu-redo   the same with the reference's verify-and-redo
//!   rubble-gpu-2dev   rubble on its own device, with redo (the first coupling)
//! (PhysX GPU, the same coupled frame with PhysX, is `realtime_peak` in the PhysX
//! repository; `regimes` here are its scenes, so the outcomes compare one to one.)
//!
//! Regimes (the A12 `realtime_peak` set plus the anchored wall):
//!   house_idle  house-1story, fresh and intact
//!   wall_ram    200 kg ram at 40 m/s through a free concrete wall
//!   house_ram   house-1story struck by a 400 kg box at 12 m/s
//!   b5_v20      b5's wall anchored at its base, 1000 kg ram at 20 m/s (punch-through)
//! all adaptive, established methods, the certified true step, contact safety 0.35 and
//! stress safety 0.5 (A12's real-time configuration).
//!
//!   cargo run --release -p stress-rubble --features gpu --example stress_bench -- \
//!       [--solver oracle|rubble-gpu] [--only REGIME] [--seconds 0.25] [--repeats 1]
//!       [--pack PATH ...]   (any scene packs, idle and rammed, as the packs set)

use std::time::Instant;

use stress_ref::scene::{Scene, SolveMode};

use stress_rubble::scenes::{house, realtime as configure, wall_ram};

/// The fidelity ladder (`--config`): from the Blast-equivalent configuration to the full
/// real-time one, one switch per rung (each rung keeps every switch of the rungs before).
/// What each configuration changes on top of `configure`, and whether impact islands run.
///   blast         quasi-static equilibrium every frame with same-step cascade, threshold
///                 bonds (no softening, Weibull, crack contact, rate, fatigue, buckling,
///                 rebar, dashpots), the engine drives every body (contacts only as loads):
///                 what the Blast ExtStressSolver computes, solved to convergence
///   adaptive      + explicit dynamics while loaded (adaptive mode), still no islands
///   islands       + impact islands routed before the step
///   softening, weibull, crack_contact, rate_effects, static_fatigue, buckling, rebar,
///   damping       + that model feature, in this order
///   realtime      the full real-time configuration (= damping, every feature on)
///   qs            realtime's features, quasi-static everywhere, impacts included
///                 (no islands), budgeted static solves (`STRESS_QS_BUDGET`)
///   tiered        realtime's features, quasi-static outside impact islands (the
///                 tiered coupling, `GpuWorld::set_tiered`)
pub const LADDER: [&str; 12] = ["blast", "adaptive", "islands", "softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping", "realtime"];
const LADDER_FEATURES: [&str; 8] = ["softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping"];

/// `scene` (already `configure`d) under ladder configuration `config`; whether islands run.
fn ladder(mut s: Scene, config: &str) -> (Scene, bool) {
    if let Some(p) = stress_ref::capability::Profile::parse(config) {
        p.apply_scene(&mut s);
        return (s, p.impact_islands());
    }
    if config == "realtime" {
        return (s, true);
    }
    if config == "qs" {
        // Quasi-static impacts: realtime's features, every body the engine's, the frame's
        // static solve for every structure (Blast's model; STRESS_QS_BUDGET=newton,cg).
        s.sim.solve_mode = SolveMode::QuasiStatic;
        return (s, false);
    }
    if config == "tiered" {
        // The tiered coupling: impact islands explicit, everything else quasi-static.
        s.sim.solve_mode = SolveMode::QuasiStatic;
        return (s, true);
    }
    let rung = LADDER.iter().position(|&c| c == config).unwrap_or_else(|| panic!("unknown --config {config} (known: {})", LADDER.join(", ")));
    for (i, f) in LADDER_FEATURES.iter().enumerate() {
        // Feature i is on from rung 3 + i.
        s.sim.features.set(f, rung >= 3 + i).unwrap();
    }
    if rung == 0 {
        s.sim.solve_mode = SolveMode::QuasiStatic;
    }
    (s, rung >= 2)
}

/// The accuracy set: the realtime regimes plus the oracle's gate cases, each in the same
/// real-time configuration (so solvers differ only by pipeline).
pub fn accuracy(seconds: f64) -> Vec<(&'static str, Scene)> {
    let named = |n: &str| {
        stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()).find(|s| s.name == n).unwrap_or_else(|| panic!("no scene {n}"))
    };
    let mut out = regimes(seconds);
    for (label, v) in [("b5_v10", 10.0), ("b5_v15", 15.0), ("b5_v30", 30.0), ("b5_v40", 40.0)] {
        out.push((label, configure(stress_ref::builders::wall_impact(v, label), seconds)));
    }
    for (label, n) in [("b8_sudden", "b8_frame_sudden"), ("b4_sudden", "b4_support_loss_sudden"), ("house_car", "s_house_car"), ("floor_drop", "s_floor_drop")] {
        let mut s = configure(named(n), seconds);
        s.name = label.into();
        out.push((label, s));
    }
    out
}

/// The town set: copies of house-1story on a grid, intact (idle) or with one rammed —
/// how cost grows with buildings that are quiet.
pub fn towns(seconds: f64) -> Vec<(&'static str, Scene)> {
    let mut out = Vec::new();
    for (label, n, rams) in [("town16_idle", 16, 0), ("town64_idle", 64, 0), ("town16_one_ram", 16, 1)] {
        let town = stress_ref::scenes::pack_town(&house(), n, 2.0);
        let mut s = configure(if rams > 0 { stress_ref::scenes::bombarded(town, rams) } else { town }, seconds);
        s.name = label.into();
        out.push((label, s));
    }
    out
}

/// The x of a pack's -x surface (hull extents, not chunk centres: `scenes::rammed` places
/// the ram 0.2 m from the leftmost centre, inside packs whose chunks are wider than that).
fn surface_lo_x(scene: &Scene) -> f64 {
    scene
        .bodies
        .iter()
        .flat_map(|b| b.chunks.iter())
        .map(|c| c.center[0] + c.hull.as_ref().map_or(-c.half_extents[0], |h| h.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min)))
        .fold(f64::INFINITY, f64::min)
}

/// The Blast packs both solvers load, each intact (idle) and rammed: `scenes::rammed`'s
/// 400 kg box at 12 m/s into the middle of the -x face, its front 0.2 m from the surface
/// (`blast_ladder_bench bench PACK ram` fires its ball from the same surface).
pub fn packs(seconds: f64) -> Vec<(String, Scene)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs");
    let paths: Vec<_> = ["house-1story", "house-2story", "villa-savoye", "brick-building", "rig-portal", "rig-toppled", "comp-wall-bay"].iter().map(|n| dir.join(format!("{n}.json"))).collect();
    packs_at(&paths, seconds)
}

/// `--authored`: packs import with their own (game) bond strengths.
static AUTHORED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static AUTHORED_FATAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PACK_MASSES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// `--config PROFILE`: a capability profile (`stress_ref::capability`), set before the
/// scenes load.
static PROFILE: std::sync::OnceLock<stress_ref::capability::Profile> = std::sync::OnceLock::new();

/// Any scene packs (`--pack PATH`, repeatable), each intact (idle) and rammed, as `packs`.
pub fn packs_at(paths: &[std::path::PathBuf], seconds: f64) -> Vec<(String, Scene)> {
    let mut out = Vec::new();
    for path in paths {
        let name = path.file_stem().expect("pack file name").to_string_lossy().to_string();
        let name = name.as_str();
        let options = stress_ref::scene_pack::ImportOptions { authored_strengths: AUTHORED.load(std::sync::atomic::Ordering::Relaxed), authored_fatal: AUTHORED_FATAL.load(std::sync::atomic::Ordering::Relaxed), pack_masses: PACK_MASSES.load(std::sync::atomic::Ordering::Relaxed), pack_modulus: PACK_MASSES.load(std::sync::atomic::Ordering::Relaxed) };
        let options = PROFILE.get().map_or(options, |p| p.import_options());
        let scene = stress_ref::scene_pack::import_with(path, name, "structure", options).unwrap_or_else(|e| panic!("{name} pack: {e}"));
        let mut idle = configure(scene.clone(), seconds);
        idle.name = format!("{name}_idle");
        let (mut lo, hi) = stress_ref::scenes::bounds(&scene);
        lo.x = surface_lo_x(&scene);
        let mut ram = configure(stress_ref::scenes::ram_at(scene.clone(), "ram", lo, hi), seconds);
        ram.name = format!("{name}_ram");
        let mut bombard = idle.clone();
        bombard.name = format!("{name}_bombard");
        out.push((idle.name.clone(), idle));
        out.push((ram.name.clone(), ram));
        out.push((bombard.name.clone(), bombard));
        // vibe-land's perf-suite town bombard, exactly (`suite_shots`), when the pack has
        // its city layout beside it (`<pack>.meta.json`): 30 settle frames, then the
        // standard tier's 90 frames or the quick tier's 60 (its first two events).
        let meta = path.with_extension("meta.json");
        if let Ok(text) = std::fs::read_to_string(&meta) {
            let meta: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", meta.display()));
            for (tier, frames, events) in [("suite_bombard", 90, 5), ("suite_quick", 60, 2)] {
                let mut s = configure(scene.clone(), (SUITE_SETTLE + frames) as f64 / 60.0);
                s.name = format!("{name}_{tier}");
                let shots = suite_shots(&meta, &SUITE_BOMBARD[..events], -s.gravity[2]);
                SUITE_SHOTS.lock().unwrap().insert(s.name.clone(), shots);
                out.push((s.name.clone(), s));
            }
        }
    }
    out
}

/// Frames the town settles before vibe-land's bombard starts (its suite's `settle` phase).
const SUITE_SETTLE: usize = 30;

/// vibe-land `scripts/perf/suite.json` `town-bombard`: (tick, kind, place, height).
const SUITE_BOMBARD: [(usize, &str, &str, f64); 5] = [
    (0, "cannonball", "elm-park/house-16", 1.5),
    (10, "meteor", "elm-park/house-18", 2.0),
    (20, "cannonball", "elm-park/house-23", 1.5),
    (30, "meteor", "market-quarter/cinema-1", 2.0),
    (40, "cannonball", "elm-park/house-25", 1.5),
];

/// The suite scenes' shots, by scene name.
static SUITE_SHOTS: std::sync::Mutex<std::collections::BTreeMap<String, Vec<(usize, stress_ref::scene::ImpactorDesc)>>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The suite's shots as vibe-land fires them (`scripts/perf/suite.py` `shot_from_place`,
/// `server/src/perf_suite.rs` `Rig::fire`), in the pack's Y-up world mapped to the
/// scene's Z-up one (`scene_pack`: `(x, y, z) -> (x, -z, y)`):
/// - target: the place's pavement x, `height` up, on its street-side face;
/// - from the street: 30 m at slope 0.02 (cannonball), 140 m at slope 0.3 (meteor);
/// - aimed with a gravity lead over the flight time at 60 m/s (cannonball) or 140 m/s;
/// - cannonball 10,650 kg of 7,850 kg/m^3 steel (`city.rs`); meteor radius 2 m at
///   3,300 kg/m^3 (`meteor.rs`).
/// Each fires before its tick, after the settle frames. `(frame, impactor)`.
fn suite_shots(meta: &serde_json::Value, events: &[(usize, &str, &str, f64)], g: f64) -> Vec<(usize, stress_ref::scene::ImpactorDesc)> {
    use stress_ref::scene::{ImpactorDesc, ImpactorShape};
    let v3 = |v: &serde_json::Value| [v[0].as_f64().unwrap(), v[1].as_f64().unwrap(), v[2].as_f64().unwrap()];
    let zup = |p: [f64; 3]| [p[0], -p[2], p[1]];
    let places = meta["places"].as_array().expect("meta places");
    events
        .iter()
        .enumerate()
        .map(|(k, &(tick, kind, place, height))| {
            let p = places.iter().find(|p| p["id"] == place).unwrap_or_else(|| panic!("no place {place} in the pack's meta"));
            let (lo, hi, pave) = (v3(&p["min"]), v3(&p["max"]), v3(&p["pavement"]));
            let from_south = pave[2] < lo[2];
            let target = [pave[0], height, if from_south { lo[2] } else { hi[2] }];
            let meteor = kind == "meteor";
            let (bearing, slope, distance): (f64, f64, f64) = (if from_south { 180f64 } else { 0.0 }.to_radians(), if meteor { 0.3 } else { 0.02 }, if meteor { 140.0 } else { 30.0 });
            let origin = [target[0] + bearing.sin() * distance, target[1] + slope * distance, target[2] + bearing.cos() * distance];
            let speed = if meteor { 140.0 } else { 60.0 };
            let d = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
            let tt = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() / speed;
            let velocity = [d[0] / tt, d[1] / tt + 0.5 * g * tt, d[2] / tt];
            let (radius, mass) = if meteor { (2.0, 3300.0 * 4.0 / 3.0 * std::f64::consts::PI * 8.0) } else { ((10650.0 / 7850.0 * 3.0 / (4.0 * std::f64::consts::PI)).cbrt(), 10650.0) };
            (
                SUITE_SETTLE + tick,
                ImpactorDesc {
                    name: format!("{kind}{k}"),
                    shape: ImpactorShape::Sphere { radius },
                    mass,
                    position: zup(origin),
                    orientation: [1.0, 0.0, 0.0, 0.0],
                    velocity: zup(velocity),
                    angular_velocity: [0.0; 3],
                    material: "projectile".into(),
                    crush: None,
                },
            )
        })
        .collect()
}

/// vibe-land's `town-bombard` (scripts/perf/suite.json), as close as a pack allows:
/// cannonballs (10,650 kg steel, 60 m/s) and meteors (2 m, 3,300 kg/m^3, 140 m/s) into
/// four houses and the cinema, one every 10 frames, aimed `height` above each building's
/// base from 20 m off its -x side. The suite's places are its city layout's names; the
/// pack's buildings (`nodeGroups`) stand in for them. `(frame, impactor)`.
fn bombard_shots(scene: &Scene) -> Vec<(usize, stress_ref::scene::ImpactorDesc)> {
    use stress_ref::scene::{ImpactorDesc, ImpactorShape};
    let plan = [("house@house-14", "cannonball", 1.5), ("house@house-18", "meteor", 2.0), ("house@house-25", "cannonball", 1.5), ("cinema@cinema-150", "meteor", 2.0), ("house@house-32", "cannonball", 1.5)];
    let mut shots = Vec::new();
    for (k, (group, kind, height)) in plan.iter().enumerate() {
        let chunks: Vec<&stress_ref::scene::ChunkDesc> = scene.bodies.iter().flat_map(|b| &b.chunks).filter(|c| c.groups.iter().any(|g| g == group)).collect();
        if chunks.is_empty() {
            continue;
        }
        let n = chunks.len() as f64;
        let (cx, cy) = (chunks.iter().map(|c| c.center[0]).sum::<f64>() / n, chunks.iter().map(|c| c.center[1]).sum::<f64>() / n);
        let base = chunks.iter().map(|c| c.center[2]).fold(f64::INFINITY, f64::min);
        let lo_x = chunks.iter().map(|c| c.center[0]).fold(f64::INFINITY, f64::min);
        let target = [lo_x, cy, base + height];
        let _ = cx;
        let (speed, radius, mass) = if *kind == "meteor" { (140.0, 2.0, 3300.0 * 4.0 / 3.0 * std::f64::consts::PI * 8.0) } else { (60.0, (10650.0 / 7850.0 * 3.0 / (4.0 * std::f64::consts::PI)).cbrt(), 10650.0) };
        let origin = [target[0] - 20.0, target[1], target[2] + 6.0];
        let d = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let t = dist / speed;
        let g = -scene.gravity[2];
        let velocity = [d[0] / t, d[1] / t, d[2] / t + 0.5 * g * t];
        shots.push((
            10 * k,
            ImpactorDesc {
                name: format!("{kind}{k}"),
                shape: ImpactorShape::Sphere { radius },
                mass,
                position: origin,
                orientation: [1.0, 0.0, 0.0, 0.0],
                velocity,
                angular_velocity: [0.0; 3],
                material: "projectile".into(),
                crush: None,
            },
        ));
    }
    shots
}

/// The scale set: 256 copies of house-1story, intact or with four of them rammed (the
/// target of hundreds of houses).
pub fn scale(seconds: f64) -> Vec<(&'static str, Scene)> {
    let mut out = Vec::new();
    for (label, n, rams) in [("town256_idle", 256, 0), ("town256_four_rams", 256, 4)] {
        let town = stress_ref::scenes::pack_town(&house(), n, 2.0);
        let mut s = configure(if rams > 0 { stress_ref::scenes::bombarded_exact(town, rams) } else { town }, seconds);
        s.name = label.into();
        out.push((label, s));
    }
    out
}

pub fn regimes(seconds: f64) -> Vec<(&'static str, Scene)> {
    let mut idle = configure(house(), seconds);
    idle.name = "house_idle".into();
    let mut house_ram = configure(stress_ref::scenes::rammed(house()), seconds);
    house_ram.name = "house_ram".into();
    let b5 = configure(stress_ref::builders::wall_impact(20.0, "b5_v20"), seconds);
    vec![("house_idle", idle), ("wall_ram", configure(wall_ram(), seconds)), ("house_ram", house_ram), ("b5_v20", b5)]
}

struct Run {
    frames: Vec<f64>,
    ram: f64,
    fragments: f64,
    broken: f64,
    redone: u64,
    unverified: u64,
    profile: serde_json::Value,
    /// The scene's own metrics (b5: failure mode hole / push_over, hole area, speed lost).
    metrics: serde_json::Value,
    /// FNV-1a of the end state (cluster motion and membership, chunk hidden states,
    /// impactors), so a lossless change is checked bit for bit.
    digest: String,
}

fn digest(m: &stress_ref::solver::ReferenceSolver, impactors: &[stress_ref::world::Impactor]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |s: String| {
        for b in s.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    };
    for cl in &m.clusters {
        eat(format!("{} {:?} {:?} {:?} {:?}", cl.id, cl.chunks, cl.pose, cl.velocity, cl.angular_velocity));
    }
    for st in m.chunks.iter().flatten() {
        eat(format!("{:?} {:?} {:?} {:?} {}", st.u, st.th, st.v, st.w, st.removed));
    }
    for i in impactors {
        eat(format!("{:?} {:?} {:?}", i.pose, i.velocity, i.angular_velocity));
    }
    format!("{h:016x}")
}

fn metrics(scene: &Scene, o: &stress_ref::observation::Observation) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for m in &scene.metrics {
        if let Ok(v) = stress_ref::metrics::evaluate(scene, m, o) {
            out.insert(m.name.clone(), serde_json::to_value(v).unwrap_or(serde_json::Value::Null));
        }
    }
    serde_json::Value::Object(out)
}

fn run_oracle(scene: &Scene) -> Run {
    let mut w = stress_ref::world::World::new(scene);
    let n = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    let mut frames = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        w.step_frame();
        frames.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let o = w.observation();
    let ram = w.impactors.first().map_or(0.0, |i| i.velocity.norm());
    let digest = digest(&w.solver, &w.impactors);
    let metrics = metrics(scene, &o);
    Run { frames, ram, fragments: o.values["fragments"], broken: o.values["broken_bonds"], redone: w.redone_frames, unverified: 0, profile: serde_json::Value::Null, metrics, digest }
}

#[cfg(feature = "gpu")]
fn run_rubble_gpu(gpu: &stress_gpu::gpu::Gpu, scene: &Scene, two_devices: bool, no_redo: bool, islands: bool) -> Run {
    use stress_ref::math::Vec3;
    // One device for both (`rubble-gpu`), or rubble on its own device (`rubble-gpu-2dev`).
    let engine: Box<dyn stress_ref::engine::RigidEngine> = if two_devices {
        stress_rubble::RubbleEngine::boxed(Vec3::from_array(scene.gravity)).expect("rubble engine")
    } else {
        Box::new(stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(scene)).expect("rubble engine"))
    };
    // STRESS_RESIDENT=1: the resident quasi-static frame (engine and stress loads on the
    // GPU without read-backs; one device).
    let profile = PROFILE.get().copied();
    let mut w = if (std::env::var("STRESS_RESIDENT").is_ok_and(|v| v == "1") || profile.is_some_and(|p| p.resident())) && !two_devices {
        drop(engine);
        let e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(scene)).expect("rubble engine");
        stress_gpu::world::GpuWorld::with_resident_engine(gpu, scene, e).expect("GPU coupled world")
    } else {
        stress_gpu::world::GpuWorld::with_engine(gpu, scene, engine).expect("GPU coupled world")
    };
    // rubble's integrated pipeline: routing before the step, no frame rewind.
    w.redo_fractured_frames = !no_redo;
    // STRESS_BENCH_ORACLE_STEP=1: the oracle's own certified step instead of the GPU
    // world's sharper dashpot certificate (`own_dashpot_step`), for comparison.
    if std::env::var("STRESS_BENCH_ORACLE_STEP").is_ok_and(|v| v == "1") {
        w.own_dashpot_step = false;
    }
    w.set_impact_islands(islands);
    if let Some(p) = profile {
        w.apply_profile(p);
    }
    // STRESS_BENCH_STRICT_STEP=s: strict mode, fail when a structure's certified step is below s.
    if let Some(min) = std::env::var("STRESS_BENCH_STRICT_STEP").ok().and_then(|v| v.parse().ok()) {
        if let Err(e) = w.set_min_stress_step(min) {
            panic!("{e}");
        }
    }
    // STRESS_QS_UNCONVERGED=restore|keep|commit: what an unconverged static solve leaves
    // (Blast: commit). STRESS_QS_PASSES=n: the frame's cascade passes at most (Blast: 2).
    match std::env::var("STRESS_QS_UNCONVERGED").as_deref() {
        Ok("keep") => w.solver.mirror.config.static_unconverged = stress_ref::solver::UnconvergedSolve::KeepIterate,
        Ok("commit") => w.solver.mirror.config.static_unconverged = stress_ref::solver::UnconvergedSolve::CommitIterate,
        Ok("restore") | Err(_) => {}
        Ok(other) => panic!("STRESS_QS_UNCONVERGED={other}: restore, keep or commit"),
    }
    if let Some(n) = std::env::var("STRESS_QS_PASSES").ok().and_then(|v| v.parse().ok()) {
        w.solver.mirror.config.static_passes = Some(n);
    }
    // STRESS_QS_BUDGET=newton,cg: quasi-static solves within a budget after setup.
    if let Some((n, cg)) = std::env::var("STRESS_QS_BUDGET").ok().and_then(|v| v.split_once(',').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))) {
        w.solver.mirror.config.static_budget = Some((n, cg));
    }
    // Quasi-static with impact islands: the tiered coupling (`--config tiered`).
    if islands && scene.sim.solve_mode == SolveMode::QuasiStatic {
        w.set_tiered(true).expect("tiered coupling");
        // STRESS_TIERED_BUDGET=newton,cg: the static solves' budget after setup.
        if let Some((n, cg)) = std::env::var("STRESS_TIERED_BUDGET").ok().and_then(|v| v.split_once(',').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))) {
            w.solver.mirror.config.static_budget = Some((n, cg));
        }
    }
    let n = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    let mut frames = Vec::with_capacity(n);
    let trace = std::env::var("STRESS_BENCH_TRACE").is_ok();
    if trace {
        w.solver.sync(gpu);
        eprintln!("setup digest {}", digest(&w.solver.mirror, &w.solver.impactors));
        let mut r = stress_ref::solver::ReferenceSolver::new(scene);
        eprintln!("fresh digest {}", digest(&r, &[]));
        r.gravity_prestress();
        eprintln!("cpu prestress digest {}", digest(&r, &[]));
    }
    // STRESS_BENCH_FRAMES=1: each frame's profile deltas on stderr.
    let per_frame = std::env::var("STRESS_BENCH_FRAMES").is_ok();
    let suite = SUITE_SHOTS.lock().unwrap().get(&scene.name).cloned();
    let shots = match suite {
        Some(shots) => shots,
        None if scene.name.ends_with("_bombard") => bombard_shots(scene),
        None => Vec::new(),
    };
    if !shots.is_empty() {
        w.scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    }
    for f in 0..n {
        for (_, desc) in shots.iter().filter(|(at, _)| *at == f) {
            let imp = stress_ref::world::Impactor::from_desc(desc, &w.scene);
            w.add_impactor(gpu, imp).expect("projectile");
        }
        let before = w.solver.profile.clone();
        let (splits, replans, tail) = (w.solver.host_splits, w.solver.replans, w.tail_seconds);
        let (prep, refresh, back, seg) = (w.prepare_seconds, w.refresh_seconds, w.write_back_seconds, w.segment_setup_seconds);
        let t = Instant::now();
        w.step_frame(gpu).expect("frame");
        frames.push(t.elapsed().as_secs_f64() * 1e3);
        if per_frame {
            let p = &w.solver.profile;
            let ms = |a: f64, b: f64| (a - b) * 1e3;
            eprintln!(
                "frame {f}: {:.1} ms; prepare {:.1}, refresh {:.1}, write-back {:.1}, segment setup {:.1}, tail {:.1}, gpu {:.1}, encode {:.1}, readback {:.1}, download {:.1}, split {:.1}, build {:.1}; {} substeps, {} batches ({} of {} rounds used), {} host splits, {} replans; clusters {} ({} in islands), ram {:.2} m/s",
                frames[f],
                ms(w.prepare_seconds, prep),
                ms(w.refresh_seconds, refresh),
                ms(w.write_back_seconds, back),
                ms(w.segment_setup_seconds, seg),
                ms(w.tail_seconds, tail),
                ms(p.gpu, before.gpu),
                ms(p.encode, before.encode),
                ms(p.readback, before.readback),
                ms(p.download, before.download),
                ms(p.split, before.split),
                ms(p.build, before.build),
                p.substeps - before.substeps,
                p.batches - before.batches,
                p.rounds_used - before.rounds_used,
                p.rounds_encoded - before.rounds_encoded,
                w.solver.host_splits - splits,
                w.solver.replans - replans,
                w.solver.mirror.clusters.len(),
                w.solver.mirror.clusters.iter().filter(|c| !c.driven).count(),
                w.solver.impactors.first().map_or(0.0, |i| i.velocity.norm())
            );
        }
        if trace {
            w.sync(gpu);
            eprintln!("frame {f} digest {}", digest(&w.solver.mirror, &w.solver.impactors));
        }
    }
    w.sync(gpu);
    let o = w.observation();
    let ram = w.solver.impactors.first().map_or(0.0, |i| i.velocity.norm());
    let p = &w.solver.profile;
    let profile = serde_json::json!({"build_s": p.build, "gpu_s": p.gpu, "readback_s": p.readback, "download_s": p.download, "split_s": p.split, "encode_s": p.encode,
        "batches": p.batches, "substeps": p.substeps, "quiet_clusters": w.solver.quiet_count(), "host_splits": w.solver.host_splits, "replans": w.solver.replans, "resumed_halts": w.solver.resumed_halts, "tail_s": w.tail_seconds, "segment_retries": w.segment_retries, "plan_check": format!("{:?}", w.solver.plan_check), "split_check": format!("{:?}", w.solver.split_check),
        "kernels": stress_gpu::solver::KERNEL_NAMES.iter().zip(p.kernels.iter().zip(p.kernel_dispatches.iter())).filter(|(_, (t, _))| **t > 0.0).map(|(n, (t, c))| serde_json::json!([n, t, c])).collect::<Vec<_>>()});
    let digest = digest(&w.solver.mirror, &w.solver.impactors);
    // STRESS_BENCH_DUMP=path: the end state (chunk displacements, cluster and impactor
    // positions) as JSON, for comparing runs beyond their digests.
    if let Ok(path) = std::env::var("STRESS_BENCH_DUMP") {
        let m = &w.solver.mirror;
        let u: Vec<[f64; 3]> = m.chunks.iter().flatten().map(|st| st.u.to_array()).collect();
        let clusters: Vec<[f64; 3]> = m.clusters.iter().map(|c| c.pose.position.to_array()).collect();
        let imps: Vec<[f64; 3]> = w.solver.impactors.iter().map(|i| i.pose.position.to_array()).collect();
        let th: Vec<[f64; 3]> = m.chunks.iter().flatten().map(|st| st.th.to_array()).collect();
        let v: Vec<[f64; 3]> = m.chunks.iter().flatten().map(|st| st.v.to_array()).collect();
        let wv: Vec<[f64; 3]> = m.chunks.iter().flatten().map(|st| st.w.to_array()).collect();
        let cv: Vec<[f64; 3]> = m.clusters.iter().map(|c| c.velocity.to_array()).collect();
        let cw: Vec<[f64; 3]> = m.clusters.iter().map(|c| c.angular_velocity.to_array()).collect();
        let cr: Vec<[f64; 3]> = m.clusters.iter().map(|c| [c.pose.rotation.x, c.pose.rotation.y, c.pose.rotation.z]).collect();
        let iv: Vec<[f64; 3]> = w.solver.impactors.iter().map(|i| i.velocity.to_array()).collect();
        let iw: Vec<[f64; 3]> = w.solver.impactors.iter().map(|i| i.angular_velocity.to_array()).collect();
        std::fs::write(path, serde_json::json!({"u": u, "th": th, "v": v, "w": wv, "clusters": clusters, "cluster_v": cv, "cluster_w": cw, "cluster_rot": cr, "impactors": imps, "imp_v": iv, "imp_w": iw}).to_string()).expect("dump");
    }
    let metrics = metrics(scene, &o);
    Run { frames, ram, fragments: o.values["fragments"], broken: o.values["broken_bonds"], redone: w.redone_frames(), unverified: w.unverified_fractures, profile, metrics, digest }
}

fn main() {
    let mut solver = "rubble-gpu".to_string();
    let mut only: Option<String> = None;
    let mut seconds = 0.25;
    let mut repeats = 1usize;
    let mut set = "realtime".to_string();
    let mut pack_paths: Vec<std::path::PathBuf> = Vec::new();
    let mut config = "realtime".to_string();
    let mut overrides: Vec<(String, bool)> = Vec::new();
    let mut methods: Vec<(String, bool)> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--solver" => solver = args.next().expect("--solver NAME"),
            "--only" => only = args.next(),
            "--set" => set = args.next().expect("--set realtime|accuracy|towns|scale|packs"),
            "--config" => config = args.next().expect("--config LADDER-RUNG"),
            // A model feature on or off on top of the configuration (e.g. crushing=off).
            "--feature" => {
                let v = args.next().expect("--feature NAME=on|off");
                let (n, on) = v.split_once('=').expect("--feature NAME=on|off");
                overrides.push((n.to_string(), on == "on"));
            }
            // A method switch on top of the configuration (e.g. blast_stiffness=on).
            "--method" => {
                let v = args.next().expect("--method NAME=on|off");
                let (n, on) = v.split_once('=').expect("--method NAME=on|off");
                methods.push((n.to_string(), on == "on"));
            }
            "--authored" => AUTHORED.store(true, std::sync::atomic::Ordering::Relaxed),
            // The pack's fatal limits as strengths (Blast breaks there; `--authored` uses
            // the elastic ones).
            // The pack's node masses and mass-0 supports (Blast's inputs).
            "--pack-masses" => PACK_MASSES.store(true, std::sync::atomic::Ordering::Relaxed),
            "--authored-fatal" => {
                AUTHORED.store(true, std::sync::atomic::Ordering::Relaxed);
                AUTHORED_FATAL.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            "--pack" => {
                pack_paths.push(args.next().expect("--pack PATH").into());
                set = "packs".into();
            }
            "--seconds" => seconds = args.next().and_then(|v| v.parse().ok()).expect("--seconds S"),
            "--repeats" => repeats = args.next().and_then(|v| v.parse().ok()).expect("--repeats N"),
            other => panic!("unknown argument {other}"),
        }
    }
    if let Some(p) = stress_ref::capability::Profile::parse(&config) {
        let _ = PROFILE.set(p);
    }
    #[cfg(feature = "gpu")]
    let gpu = (solver != "oracle").then(|| stress_gpu::gpu::Gpu::new().expect("GPU"));
    let named = |l: Vec<(&'static str, Scene)>| l.into_iter().map(|(n, s)| (n.to_string(), s)).collect::<Vec<_>>();
    let list = match set.as_str() {
        "accuracy" => named(accuracy(seconds)),
        "towns" => named(towns(seconds)),
        "scale" => named(scale(seconds)),
        "packs" if !pack_paths.is_empty() => packs_at(&pack_paths, seconds),
        "packs" => packs(seconds),
        _ => named(regimes(seconds)),
    };
    for (name, scene) in list {
        if only.as_ref().is_some_and(|o| *o != name) {
            continue;
        }
        let (mut scene, islands) = ladder(scene, &config);
        // STRESS_BENCH_STRESS_SAFETY=x: the bond network's step safety (a measurement).
        if let Some(x) = std::env::var("STRESS_BENCH_STRESS_SAFETY").ok().and_then(|v| v.parse().ok()) {
            scene.sim.stress_courant_safety = Some(x);
        }
        for (n, on) in &overrides {
            scene.sim.features.set(n, *on).unwrap_or_else(|e| panic!("--feature: {e}"));
        }
        for (n, on) in &methods {
            scene.sim.methods.set(n, *on).unwrap_or_else(|e| panic!("--method: {e}"));
        }
        let _ = islands;
        for repeat in 0..repeats.max(1) {
            // A run that panics (an engine failure) is reported as failed and the next
            // scene runs, so one failing rung still reports the others.
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match solver.as_str() {
                "oracle" => run_oracle(&scene),
                #[cfg(feature = "gpu")]
                "rubble-gpu" => run_rubble_gpu(gpu.as_ref().unwrap(), &scene, false, true, islands),
                #[cfg(feature = "gpu")]
                "rubble-gpu-redo" => run_rubble_gpu(gpu.as_ref().unwrap(), &scene, false, false, islands),
                #[cfg(feature = "gpu")]
                "rubble-gpu-2dev" => run_rubble_gpu(gpu.as_ref().unwrap(), &scene, true, false, islands),
                other => panic!("unknown solver {other} (rubble-gpu needs --features gpu)"),
            }));
            let run = match run {
                Ok(run) => run,
                Err(e) => {
                    let why = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                    println!("{}", serde_json::json!({"solver": solver, "config": config, "regime": name, "repeat": repeat, "failed": why}));
                    continue;
                }
            };
            println!(
                "{}",
                serde_json::json!({"solver": solver, "config": config, "features": overrides.iter().map(|(n, on)| format!("{n}={}", if *on { "on" } else { "off" })).collect::<Vec<_>>(), "regime": name, "repeat": repeat, "seconds": seconds, "frame_dt": scene.sim.frame_dt,
                    "frame_ms": run.frames, "engine_ms": stress_rubble::profile::take(), "ram": run.ram, "fragments": run.fragments, "broken_bonds": run.broken, "redone_frames": run.redone, "unverified_fractures": run.unverified, "profile": run.profile, "metrics": run.metrics, "digest": run.digest})
            );
        }
    }
}

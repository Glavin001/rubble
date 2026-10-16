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

use std::time::Instant;

use stress_ref::scene::{Methods, Scene, SolveMode};

fn configure(mut s: Scene, seconds: f64) -> Scene {
    s.sim.methods = Methods::established();
    s.sim.methods.true_step = true;
    s.sim.courant_safety = 0.35;
    s.sim.stress_courant_safety = Some(0.5);
    s.sim.solve_mode = SolveMode::Adaptive;
    s.sim.duration = seconds;
    s
}

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
pub const LADDER: [&str; 12] = ["blast", "adaptive", "islands", "softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping", "realtime"];
const LADDER_FEATURES: [&str; 8] = ["softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping"];

/// `scene` (already `configure`d) under ladder configuration `config`; whether islands run.
fn ladder(mut s: Scene, config: &str) -> (Scene, bool) {
    if config == "realtime" {
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

fn wall_ram() -> Scene {
    use stress_ref::builders::*;
    use stress_ref::math::Vec3;
    use stress_ref::scene::*;
    let mut s = new_scene("wall_ram", "free concrete wall hit by a steel ram");
    s.gravity = [0.0; 3];
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
    s
}

fn house() -> Scene {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/house-1story.json");
    stress_ref::scene_pack::import(&path, "house-1story", "structure").expect("house-1story pack")
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

/// The Blast packs both solvers load, each intact (idle) and rammed as `house_ram` is.
pub fn packs(seconds: f64) -> Vec<(String, Scene)> {
    let mut out = Vec::new();
    for name in ["house-1story", "house-2story", "villa-savoye", "brick-building", "rig-portal", "rig-toppled", "comp-wall-bay"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../packs/{name}.json"));
        let scene = stress_ref::scene_pack::import(&path, name, "structure").unwrap_or_else(|e| panic!("{name} pack: {e}"));
        let mut idle = configure(scene.clone(), seconds);
        idle.name = format!("{name}_idle");
        let mut ram = configure(stress_ref::scenes::rammed(scene), seconds);
        ram.name = format!("{name}_ram");
        out.push((idle.name.clone(), idle));
        out.push((ram.name.clone(), ram));
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
        Box::new(stress_rubble::RubbleEngine::with_device(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone()).expect("rubble engine"))
    };
    let mut w = stress_gpu::world::GpuWorld::with_engine(gpu, scene, engine).expect("GPU coupled world");
    // rubble's integrated pipeline: routing before the step, no frame rewind.
    w.redo_fractured_frames = !no_redo;
    w.set_impact_islands(islands);
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
    for f in 0..n {
        let t = Instant::now();
        w.step_frame(gpu).expect("frame");
        frames.push(t.elapsed().as_secs_f64() * 1e3);
        if trace {
            eprintln!("frame {f} digest {}", digest(&w.solver.mirror, &w.solver.impactors));
        }
    }
    w.solver.sync(gpu);
    let o = w.observation();
    let ram = w.solver.impactors.first().map_or(0.0, |i| i.velocity.norm());
    let p = &w.solver.profile;
    let profile = serde_json::json!({"build_s": p.build, "gpu_s": p.gpu, "readback_s": p.readback, "download_s": p.download, "split_s": p.split,
        "batches": p.batches, "substeps": p.substeps, "host_splits": w.solver.host_splits, "tail_s": w.tail_seconds, "segment_retries": w.segment_retries,
        "kernels": stress_gpu::solver::KERNEL_NAMES.iter().zip(p.kernels.iter().zip(p.kernel_dispatches.iter())).filter(|(_, (t, _))| **t > 0.0).map(|(n, (t, c))| serde_json::json!([n, t, c])).collect::<Vec<_>>()});
    let digest = digest(&w.solver.mirror, &w.solver.impactors);
    let metrics = metrics(scene, &o);
    Run { frames, ram, fragments: o.values["fragments"], broken: o.values["broken_bonds"], redone: w.redone_frames(), unverified: w.unverified_fractures, profile, metrics, digest }
}

fn main() {
    let mut solver = "rubble-gpu".to_string();
    let mut only: Option<String> = None;
    let mut seconds = 0.25;
    let mut repeats = 1usize;
    let mut set = "realtime".to_string();
    let mut config = "realtime".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--solver" => solver = args.next().expect("--solver NAME"),
            "--only" => only = args.next(),
            "--set" => set = args.next().expect("--set realtime|accuracy|towns|packs"),
            "--config" => config = args.next().expect("--config LADDER-RUNG"),
            "--seconds" => seconds = args.next().and_then(|v| v.parse().ok()).expect("--seconds S"),
            "--repeats" => repeats = args.next().and_then(|v| v.parse().ok()).expect("--repeats N"),
            other => panic!("unknown argument {other}"),
        }
    }
    #[cfg(feature = "gpu")]
    let gpu = (solver != "oracle").then(|| stress_gpu::gpu::Gpu::new().expect("GPU"));
    let named = |l: Vec<(&'static str, Scene)>| l.into_iter().map(|(n, s)| (n.to_string(), s)).collect::<Vec<_>>();
    let list = match set.as_str() {
        "accuracy" => named(accuracy(seconds)),
        "towns" => named(towns(seconds)),
        "packs" => packs(seconds),
        _ => named(regimes(seconds)),
    };
    for (name, scene) in list {
        if only.as_ref().is_some_and(|o| *o != name) {
            continue;
        }
        let (scene, islands) = ladder(scene, &config);
        let _ = islands;
        for repeat in 0..repeats.max(1) {
            let run = match solver.as_str() {
                "oracle" => run_oracle(&scene),
                #[cfg(feature = "gpu")]
                "rubble-gpu" => run_rubble_gpu(gpu.as_ref().unwrap(), &scene, false, true, islands),
                #[cfg(feature = "gpu")]
                "rubble-gpu-redo" => run_rubble_gpu(gpu.as_ref().unwrap(), &scene, false, false, islands),
                #[cfg(feature = "gpu")]
                "rubble-gpu-2dev" => run_rubble_gpu(gpu.as_ref().unwrap(), &scene, true, false, islands),
                other => panic!("unknown solver {other} (rubble-gpu needs --features gpu)"),
            };
            println!(
                "{}",
                serde_json::json!({"solver": solver, "config": config, "regime": name, "repeat": repeat, "seconds": seconds, "frame_dt": scene.sim.frame_dt,
                    "frame_ms": run.frames, "engine_ms": stress_rubble::profile::take(), "ram": run.ram, "fragments": run.fragments, "broken_bonds": run.broken, "redone_frames": run.redone, "unverified_fractures": run.unverified, "profile": run.profile, "metrics": run.metrics, "digest": run.digest})
            );
        }
    }
}

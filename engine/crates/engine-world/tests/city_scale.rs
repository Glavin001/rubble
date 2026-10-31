//! Benchmark (ignored): what a step costs as a map fills with buildings
//! (engine/docs/large-maps.md, "Gates and benchmark").
//!
//! K threshold cuboid houses are laid on a grid over the same 2.2 km square, and the
//! per-step cost of four scenarios is measured for each K:
//! - `still`: nothing moves after settling;
//! - `drive`: one kinematic car drives a street, touching nothing;
//! - `demolish`: one house is struck;
//! - `war`: 20 houses are struck at once.
//! The goal is a cost that does not grow with K in any of them.
//!
//! CITY_K=16,64,256 cargo test -p engine-world --release --test city_scale -- --ignored --nocapture
//! CITY_SLEEP=1: the large-map sleep policy (`sleeping_solver_for`; the threshold solver's
//! own suggestion is never asleep).
//! CITY_JSON=path appends one JSON line per run. RUBBLE_GPU_TIMING=1: the GPU column is every
//! pass of the step (its timestamp marks; the waits they add inflate the wall column).
#![cfg(feature = "stress")]

use rubble::*;
use std::time::Instant;

fn map() -> f32 {
    std::env::var("CITY_MAP")
        .map(|s| s.parse().unwrap())
        .unwrap_or(2200.0)
}
const SETTLE: u32 = 120;
const MEASURE: u32 = 120;

struct Run {
    k: usize,
    scenario: &'static str,
    wall_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    /// `World::step` alone: the host's part of a step.
    host_ms: Vec<f64>,
    awake: u32,
    broken: u32,
}

fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p).round() as usize]
}

fn capacity(asset: &DestructibleAssetDesc, k: usize) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum::<usize>() * k;
    let one = stress_rubble::capacity_for(&scene);
    let mut c = one.clone();
    c.max_bodies = ((chunks + 64 + 4) as u32).next_power_of_two();
    c.max_colliders = c
        .max_colliders
        .max(((one.max_colliders as usize * k) as u32).next_power_of_two());
    c.max_hull_vertices = c
        .max_hull_vertices
        .max(((one.max_hull_vertices as usize * k) as u32).next_power_of_two());
    c.max_contacts =
        stress_rubble::threshold::contact_capacity(chunks, one.max_contacts * k as u32, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c.max_joints = c.max_joints.max(1 << 16);
    c
}

/// Lot centres: a ceil(sqrt(k))² grid over the map, centred on the origin.
fn lots(k: usize) -> (Vec<[f32; 2]>, f32) {
    let side = (k as f64).sqrt().ceil() as usize;
    let pitch = map() / side as f32;
    let origin = -map() / 2.0 + pitch / 2.0;
    let lots = (0..k)
        .map(|i| {
            [
                origin + (i % side) as f32 * pitch,
                origin + (i / side) as f32 * pitch,
            ]
        })
        .collect();
    (lots, pitch)
}

fn city(asset: &DestructibleAssetDesc, k: usize, sleep: bool) -> (World, Vec<[f32; 2]>, f32, f32) {
    // Sleep: the large-map policy (sleeping_solver_for); houses are placed awake, settle,
    // and fall asleep with their resting contacts archived.
    let solver = if sleep {
        asset.sleeping_solver_for(StrengthModel::Threshold)
    } else {
        asset.solver_for(StrengthModel::Threshold)
    }
    .expect("threshold solver");
    let def = WorldDef {
        capacity: capacity(asset, k),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: sleep,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let placement = stress.placement();
    let ground = stress.ground_height().unwrap_or(0.0) - placement[1] as f32;
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(
        g,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: ground,
        },
    )
    .unwrap();
    let a = w.create_destructible_asset(asset).unwrap();
    let (lots, pitch) = lots(k);
    for l in &lots {
        // The asset frame less its placement, moved to the lot.
        let p = [
            l[0] - placement[0] as f32,
            -placement[1] as f32,
            l[1] - placement[2] as f32,
        ];
        w.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform {
                position: p,
                ..Transform::default()
            },
            asleep: false,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    }
    (w, lots, pitch, ground)
}

fn frame(w: &mut World, broken: &mut u32) -> f64 {
    let t = Instant::now();
    w.step().unwrap();
    t.elapsed().as_secs_f64() * 1e3 + finish(w, broken)
}

/// Wait for the step's frame and count its broken bonds: ms.
fn finish(w: &mut World, broken: &mut u32) -> f64 {
    let t = Instant::now();
    for f in w.poll_blocking().unwrap() {
        for e in &f.destruction_events {
            if let DestructionEvent::BondsBroken {
                bonds,
                destructible,
            } = e
            {
                if std::env::var("CITY_DEBUG").is_ok() {
                    eprintln!("step {}: {destructible:?} broke {}", f.step, bonds.len());
                }
                *broken += bonds.len() as u32;
            }
        }
    }
    t.elapsed().as_secs_f64() * 1e3
}

fn shot(w: &mut World, at: [f32; 2], ground: f32) {
    let b = w
        .create_body(&BodyDef {
            linear_velocity: [0.0, -40.0, 0.0],
            is_bullet: true,
            ..BodyDef::dynamic([at[0], ground + 30.0, at[1]])
        })
        .unwrap();
    w.create_sphere_shape(
        b,
        &ShapeDef {
            density: 7800.0,
            ..ShapeDef::default()
        },
        [0.0; 3],
        0.6,
    )
    .unwrap();
    w.body_set_projectile(b, Some(Projectile::default()))
        .unwrap();
}

fn run(asset: &DestructibleAssetDesc, k: usize, scenario: &'static str, sleep: bool) -> Run {
    let (mut w, lots, pitch, ground) = city(asset, k, sleep);
    let mut broken = 0;
    for _ in 0..SETTLE {
        frame(&mut w, &mut broken);
    }
    match scenario {
        "drive" => {
            // A street halfway between the first two rows of lots.
            let z = lots[0][1] + pitch / 2.0;
            let car = w
                .create_body(&BodyDef {
                    body_type: BodyType::Kinematic,
                    linear_velocity: [15.0, 0.0, 0.0],
                    ..BodyDef::dynamic([-map() / 2.0, ground + 1.0, z])
                })
                .unwrap();
            w.create_box_shape(car, &ShapeDef::default(), [2.0, 0.75, 1.0])
                .unwrap();
        }
        "spawn" => {
            // A static box far above the map: touches nothing.
            let b = w
                .create_body(&BodyDef {
                    body_type: BodyType::Static,
                    ..BodyDef::dynamic([0.0, 500.0, 0.0])
                })
                .unwrap();
            w.create_box_shape(b, &ShapeDef::default(), [1.0, 1.0, 1.0])
                .unwrap();
        }
        "demolish" => shot(&mut w, lots[k / 2], ground),
        "war" => {
            for i in 0..20.min(k) {
                shot(&mut w, lots[i * k / 20.min(k)], ground);
            }
        }
        _ => {}
    }
    let broken_before = broken;
    let (mut wall_ms, mut gpu_ms, mut host_ms) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..MEASURE {
        let t = Instant::now();
        w.step().unwrap();
        host_ms.push(t.elapsed().as_secs_f64() * 1e3);
        wall_ms.push(host_ms.last().unwrap() + finish(&mut w, &mut broken));
        // RUBBLE_GPU_TIMING=1: every pass's GPU time (the step's marks), else the sim's.
        gpu_ms.push(w.last_step_gpu_ms().unwrap_or_else(|| w.profile_ms()));
    }
    let awake = w.counters().map(|c| c.awake_count).unwrap_or(0);
    Run {
        k,
        scenario,
        wall_ms,
        gpu_ms,
        host_ms,
        awake,
        broken: broken - broken_before,
    }
}

#[test]
#[ignore]
fn city_scale() {
    let ks: Vec<usize> = std::env::var("CITY_K")
        .unwrap_or_else(|_| "16,64,256".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let scenarios: Vec<String> = std::env::var("CITY_SCENARIOS")
        .unwrap_or_else(|_| "still,drive,demolish,war".into())
        .split(',')
        .map(String::from)
        .collect();
    let sleep = std::env::var("CITY_SLEEP").is_ok();
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    let chunks = asset.chunks.len();
    println!(
        "cuboid house: {chunks} chunks, {} bonds; sleep {sleep}",
        asset.bonds.len()
    );
    println!("| K | chunks | scenario | wall median ms | wall p95 ms | host median ms | gpu median ms | awake | bonds broken |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for &k in &ks {
        for s in &scenarios {
            let s: &'static str = Box::leak(s.clone().into_boxed_str());
            let r = run(&asset, k, s, sleep);
            let (wm, wp, hm, gm) = (
                percentile(&r.wall_ms, 0.5),
                percentile(&r.wall_ms, 0.95),
                percentile(&r.host_ms, 0.5),
                percentile(&r.gpu_ms, 0.5),
            );
            println!(
                "| {} | {} | {} | {wm:.2} | {wp:.2} | {hm:.2} | {gm:.2} | {} | {} |",
                r.k,
                r.k * chunks,
                r.scenario,
                r.awake,
                r.broken
            );
            if let Ok(path) = std::env::var("CITY_JSON") {
                use std::io::Write;
                let line = format!(
                    "{{\"k\":{},\"chunks\":{},\"scenario\":\"{}\",\"sleep\":{sleep},\"wall_median_ms\":{wm:.3},\"wall_p95_ms\":{wp:.3},\"host_median_ms\":{hm:.3},\"gpu_median_ms\":{gm:.3},\"awake\":{},\"broken\":{}}}\n",
                    r.k, r.k * chunks, r.scenario, r.awake, r.broken
                );
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .unwrap()
                    .write_all(line.as_bytes())
                    .unwrap();
            }
        }
    }
}

//! Equivalence harness (ignored; `EQUIV_OUT` or `EQUIV_COMPARE`): runs the same scenes, steps
//! and scripted strikes on one build and records them, then compares two recordings.
//!
//! Record (one scene and sleep mode per run, so each run stays bounded):
//!   EQUIV_OUT=dir EQUIV_SCENE=<scene> EQUIV_SLEEP=on|off \
//!     cargo test --release -p engine-world --test equivalence -- --ignored --nocapture record
//! Scenes: house, arch (61 lab shots each, nudged -30..30 mm, 120 frames), grid (the demo's
//! "100k Grid", a CCD ball fired at step 400, 900 steps), ring ("Brick Ring (28k)", 400
//! steps), gables (12 brick gables, two balls, 600 steps), city (16 threshold houses, the
//! war scenario after settling, 240 steps), vehicle (engine-vehicle's rigid sedan at full
//! throttle into a 160-brick wall, 600 steps).
//! Each run writes `dir/<scene>-<sleep>/`: `summary.txt` (one line a step or shot: contacts,
//! awake, bodies, bonds broken) and `bodies_<step>.bin` (every body's position, linear
//! velocity and flags, little-endian f32 / u32, 28 bytes a body).
//!
//! Compare: EQUIV_COMPARE=dirA,dirB cargo test ... -- --ignored --nocapture compare
//! prints, per recorded scene, the largest and mean position difference, the bodies whose
//! awake state differs, contact and bond differences, and the hard checks of each side
//! (bodies below the ground, NaNs, body counts).
//!
//! Only API present on origin/main is used, so the file runs unchanged on both builds.
#![cfg(feature = "stress")]

use rubble::*;
use std::io::Write;
use std::path::{Path, PathBuf};

fn out_dir(scene: &str, sleep: bool) -> PathBuf {
    let d = PathBuf::from(std::env::var("EQUIV_OUT").expect("EQUIV_OUT"))
        .join(format!("{scene}-{}", if sleep { "on" } else { "off" }));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `EQUIV_ENV=1`: the world reads the `ENGINE_*` benchmark variables (`WorldDef::env_overrides`),
/// so a recording can be made with one engine switch flipped.
fn env(mut def: WorldDef) -> WorldDef {
    def.env_overrides = std::env::var("EQUIV_ENV").is_ok_and(|v| v == "1");
    def
}

fn dump_bodies(w: &World, dir: &Path, tag: &str) {
    let mut bytes = Vec::new();
    for b in w.sim().download_bodies() {
        for v in b.position.iter().chain(b.linear_vel.iter()) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&b.flags.to_le_bytes());
    }
    std::fs::write(dir.join(format!("bodies_{tag}.bin")), bytes).unwrap();
}

fn summary_line(f: &mut std::fs::File, tag: &str, w: &World, broken: u32) {
    let c = w.counters();
    writeln!(
        f,
        "{tag} contacts {} awake {} bodies {} broken {broken}",
        c.map_or(0, |c| c.contact_count),
        c.map_or(0, |c| c.awake_count),
        c.map_or(0, |c| c.body_count),
    )
    .unwrap();
}

fn bonds_in(frames: &[Frame]) -> u32 {
    frames
        .iter()
        .flat_map(|f| f.destruction_events.iter())
        .map(|e| match e {
            DestructionEvent::BondsBroken { bonds, .. } => bonds.len() as u32,
            _ => 0,
        })
        .sum()
}

// ---- threshold shots (tests/threshold.rs's setup, as origin/main runs it) ----

fn capacity(asset: &DestructibleAssetDesc) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c
        .max_bodies
        .max(((chunks + 16 + 4) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c
}

fn target(asset: &DestructibleAssetDesc) -> Option<[f64; 3]> {
    if asset.chunks.iter().any(|c| c.node.is_some()) {
        let top = asset
            .chunks
            .iter()
            .max_by(|a, b| a.pose.position[1].total_cmp(&b.pose.position[1]))?;
        return Some(top.pose.position.map(|x| x as f64));
    }
    None
}

fn lab_shot(
    lab: &engine_fracture::Scene,
    target: Option<[f64; 3]>,
) -> ([f64; 3], [f64; 3], f64, f64) {
    let c = target.unwrap_or(lab.center);
    let origin = [c[0], c[1], c[2] + 3.0 * lab.extent.max(1.0)];
    let speed = engine_fracture::plan::default_shot_speed(lab.extent, lab.projectile_speed);
    let v = engine_fracture::plan::shot_velocity(
        c,
        lab.extent,
        lab.projectile_radius,
        origin,
        [0.0, 0.0, -1.0],
        Some(speed),
        lab.projectile_arc,
    );
    (origin, v, lab.projectile_radius, lab.projectile_mass)
}

fn shots(scene: &str, sleep: bool) {
    let (asset, lab) = if scene == "arch" {
        let glb = include_bytes!("../../engine-webgpu/fracture/assets/masonry-arch.glb");
        let asset = DestructibleAssetDesc::from_glb(glb, 1).unwrap();
        let parsed = engine_fracture::parse_fracture_glb(glb).unwrap();
        (
            asset,
            engine_fracture::build_scene(&parsed, 1, None).unwrap(),
        )
    } else {
        let asset = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
        (
            asset,
            engine_fracture::samples::cuboid_stress_scene(None)
                .unwrap()
                .0,
        )
    };
    let dir = out_dir(scene, sleep);
    let mut sum = std::fs::File::create(dir.join("summary.txt")).unwrap();
    let offset = asset
        .stress
        .as_ref()
        .unwrap()
        .placement()
        .map(|x| -x as f32);
    for nudge in -30..=30 {
        // The engine's sleeping threshold solver (wake on touch) or never asleep.
        let solver = if sleep {
            asset.sleeping_solver_for(StrengthModel::Threshold).unwrap()
        } else {
            asset.solver_for(StrengthModel::Threshold).unwrap()
        };
        let def = WorldDef {
            capacity: capacity(&asset),
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: sleep,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        };
        let mut w = World::new(GpuContext::Default, &env(def)).unwrap();
        if let Some(h) = asset.stress.as_ref().unwrap().ground_height() {
            let g = w.create_body(&BodyDef::default()).unwrap();
            w.create_shape(
                g,
                &ShapeDef::default(),
                Geometry::Plane {
                    normal: [0.0, 1.0, 0.0],
                    offset: h + offset[1],
                },
            )
            .unwrap();
        }
        let a = w.create_destructible_asset(&asset).unwrap();
        w.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })
        .unwrap();
        let (origin, v, r, m) = lab_shot(&lab, target(&asset));
        let origin = [origin[0] + nudge as f64 * 1e-3, origin[1], origin[2]];
        let at = [0, 1, 2].map(|k| (origin[k] + offset[k] as f64) as f32);
        let ball = w
            .create_body(&BodyDef {
                linear_velocity: v.map(|x| x as f32),
                ..BodyDef::dynamic(at)
            })
            .unwrap();
        w.create_sphere_shape(
            ball,
            &ShapeDef {
                update_body_mass: false,
                ..ShapeDef::default()
            },
            [0.0; 3],
            r as f32,
        )
        .unwrap();
        let i = (0.4 * m * r * r) as f32;
        w.body_set_mass_data(
            ball,
            MassData {
                mass: m as f32,
                center: [0.0; 3],
                rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
            },
        )
        .unwrap();
        w.body_set_projectile(ball, Some(Projectile::default()))
            .unwrap();
        let mut broken = 0;
        let mut timeline = Vec::new();
        for _ in 0..120 {
            w.step().unwrap();
            broken += bonds_in(&w.poll_blocking().unwrap());
            timeline.push(broken.to_string());
        }
        summary_line(
            &mut sum,
            &format!("shot {nudge} timeline {}", timeline.join(",")),
            &w,
            broken,
        );
        dump_bodies(&w, &dir, &format!("shot{nudge}"));
    }
}

// ---- engine-scenes snapshots (the demo's scenes) ----

fn snapshot_world(name: &str, sleep: bool, per_body: u32) -> (World, Vec<BodyId>) {
    let (host, _) = engine_scenes::build_host(name).unwrap();
    let snap = host.snapshot();
    let n = snap.bodies.len() as u32 + 8;
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: n.next_power_of_two(),
            max_colliders: n.next_power_of_two(),
            max_contacts: (per_body * n).next_power_of_two(),
            max_candidates: (per_body * n).next_power_of_two(),
            max_colors: 32,
            ..WorldCapacity::small()
        },
        enable_sleep: sleep,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &env(def)).unwrap();
    let imported = w.import_snapshot(&snap).unwrap();
    // The snapshot's solver replaces the world's: set sleep again (the demo's 0.5 s, or never).
    let mut solver = w.solver();
    solver.sleep_time = if sleep { 0.5 } else { 1.0e6 };
    w.set_solver(solver);
    (w, imported.bodies.clone())
}

fn ball(w: &mut World, at: [f32; 3], v: [f32; 3], r: f32, mass: f32, bullet: bool) {
    let b = w
        .create_body(&BodyDef {
            linear_velocity: v,
            is_bullet: bullet,
            ..BodyDef::dynamic(at)
        })
        .unwrap();
    let d = ShapeDef {
        density: mass / (4.0 / 3.0 * std::f32::consts::PI * r * r * r),
        ..ShapeDef::default()
    };
    w.create_sphere_shape(b, &d, [0.0; 3], r).unwrap();
}

fn stepped(
    mut w: World,
    dir: &Path,
    steps: u32,
    every: u32,
    mut script: impl FnMut(&mut World, u32),
) {
    let mut sum = std::fs::File::create(dir.join("summary.txt")).unwrap();
    let mut broken = 0;
    for s in 1..=steps {
        script(&mut w, s);
        w.step().unwrap();
        broken += bonds_in(&w.poll_blocking().unwrap());
        summary_line(&mut sum, &format!("step {s}"), &w, broken);
        if s % every == 0 {
            dump_bodies(&w, dir, &format!("{s:05}"));
        }
    }
}

fn gables(sleep: bool) -> World {
    // The demo's brick gables (scene-builders.mjs brickGables), 4 x 3 of them, and two balls.
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 8192,
            max_colliders: 8192,
            max_contacts: 1 << 17,
            max_candidates: 1 << 17,
            max_colors: 32,
            ..WorldCapacity::small()
        },
        enable_sleep: sleep,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &env(def)).unwrap();
    let mut solver = w.solver();
    solver.iterations = 3;
    solver.flags |= engine_schema::FLAG_REUSE_CONTACTS;
    solver.sleep_time = if sleep { 0.5 } else { 1.0e6 };
    w.set_solver(solver);
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [512.0, 0.5, 512.0])
        .unwrap();
    let brick = ShapeDef {
        density: 1.0,
        ..ShapeDef::default()
    };
    for cx in 0..4 {
        for cy in 0..3 {
            let x0 = (cx as f32 - 1.5) * 32.0;
            let z = (cy as f32 - 1.0) * 5.0;
            for k in 0..30 {
                let m = 30 - k;
                for i in 0..m {
                    let p = [
                        x0 + (i as f32 - (m - 1) as f32 / 2.0) * 1.01,
                        0.25 + 0.5 * k as f32,
                        z,
                    ];
                    let b = w.create_body(&BodyDef::dynamic(p)).unwrap();
                    w.create_box_shape(b, &brick, [0.5, 0.25, 0.25]).unwrap();
                }
            }
        }
    }
    w
}

fn vehicle(sleep: bool) -> World {
    // engine-vehicle's rigid sedan preset at full throttle (+Z) into a 16 x 10 wall of loose
    // bricks 25 m ahead, on a rough ground box.
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 4096,
            max_colliders: 4096,
            max_contacts: 1 << 16,
            max_candidates: 1 << 17,
            max_joints: 1024,
            max_colors: 32,
            ..WorldCapacity::small()
        },
        enable_sleep: sleep,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &env(def)).unwrap();
    w.set_gravity([0.0, -9.81, 0.0]).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -5.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    let rough = ShapeDef {
        material: SurfaceMaterial {
            friction: 1.0,
            ..SurfaceMaterial::default()
        },
        ..ShapeDef::default()
    };
    w.create_box_shape(g, &rough, [200.0, 5.0, 200.0]).unwrap();
    let half = [0.4, 0.2, 0.2];
    for row in 0..10 {
        for col in 0..16 {
            let x = (col as f32 - 7.5 + if row % 2 == 1 { 0.5 } else { 0.0 }) * 2.0 * half[0];
            let y = half[1] + row as f32 * 2.0 * half[1];
            let b = w.create_body(&BodyDef::dynamic([x, y, 25.0])).unwrap();
            w.create_box_shape(
                b,
                &ShapeDef {
                    density: 600.0,
                    ..ShapeDef::default()
                },
                half,
            )
            .unwrap();
        }
    }
    let built = w
        .create_vehicle(&VehicleDef {
            model: VehicleModel::Rigid(engine_vehicle::presets::sedan()),
            transform: Transform::new([0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]),
            parts: vec![],
            connections: vec![],
            user_data: 0,
        })
        .unwrap();
    w.set_vehicle_input(
        built.vehicle.unwrap(),
        VehicleInput {
            forward: 1.0,
            ..VehicleInput::default()
        },
    );
    w
}

fn record(scene: &str, sleep: bool) {
    let dir = out_dir(scene, sleep);
    match scene {
        "house" | "arch" => shots(scene, sleep),
        "grid" => {
            let (w, _) = snapshot_world("100k Grid", sleep, 16);
            let eye = glam::Vec3::new(46.0, 30.0, 58.0);
            let d = (glam::Vec3::new(0.0, 9.0, 0.0) - eye).normalize();
            stepped(w, &dir, 900, 50, |w, s| {
                if s == 400 {
                    ball(
                        w,
                        (eye + d * 2.0).to_array(),
                        (d * 30.0).to_array(),
                        0.35,
                        8.0,
                        true,
                    );
                }
            });
        }
        "ring" => {
            // 400 steps: from about step 440 the collapsed ring exhausts the record warm hash
            // (OVERFLOW_WARM_HASH, 64 probes) on main and on the PR alike.
            let (w, _) = snapshot_world("Brick Ring (28k)", sleep, 16);
            stepped(w, &dir, 400, 50, |w, s| {
                if s == 200 {
                    ball(w, [-80.0, 3.0, 0.0], [40.0, 0.0, 0.0], 2.0, 4000.0, true);
                }
            });
        }
        "gables" => {
            let w = gables(sleep);
            stepped(w, &dir, 600, 50, |w, s| {
                if s == 150 {
                    ball(w, [-16.0, 4.3, -25.0], [0.0, 0.0, 12.0], 4.0, 1.4e5, true);
                    ball(w, [16.0, 4.3, -25.0], [0.0, 0.0, 12.0], 4.0, 1.4e5, true);
                }
            });
        }
        "city" => {
            let house = rubble::city::city_house().unwrap();
            let def = rubble::city::CityDef {
                houses: 16,
                sleep,
                ..Default::default()
            };
            let mut w =
                World::new(GpuContext::Default, &env(def.world_def(&house).unwrap())).unwrap();
            let mut c = rubble::city::City::begin(&mut w, &house, def).unwrap();
            c.place(&mut w, 16).unwrap();
            for _ in 0..rubble::city::CITY_SETTLE {
                w.step().unwrap();
                w.poll_blocking().unwrap();
            }
            c.start(
                &mut w,
                rubble::city::CityScenario::from_name("war").unwrap(),
            )
            .unwrap();
            stepped(w, &dir, 240, 40, |_, _| {});
        }
        "vehicle" => stepped(vehicle(sleep), &dir, 600, 50, |_, _| {}),
        s => panic!("unknown EQUIV_SCENE {s}"),
    }
}

#[test]
#[ignore]
fn record_scene() {
    if std::env::var("EQUIV_OUT").is_err() {
        return;
    }
    let scene = std::env::var("EQUIV_SCENE").expect("EQUIV_SCENE");
    let sleep = std::env::var("EQUIV_SLEEP").map_or(true, |v| v != "off");
    record(&scene, sleep);
}

// ---- comparison ----

struct Bodies {
    pos: Vec<[f32; 3]>,
    vel: Vec<[f32; 3]>,
    flags: Vec<u32>,
}

fn load(p: &Path) -> Bodies {
    let b = std::fs::read(p).unwrap();
    let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let n = b.len() / 28;
    let mut out = Bodies {
        pos: Vec::new(),
        vel: Vec::new(),
        flags: Vec::new(),
    };
    for i in 0..n {
        let o = i * 28;
        out.pos.push([f(o), f(o + 4), f(o + 8)]);
        out.vel.push([f(o + 12), f(o + 16), f(o + 20)]);
        out.flags
            .push(u32::from_le_bytes(b[o + 24..o + 28].try_into().unwrap()));
    }
    out
}

fn summary(p: &Path) -> Vec<(String, [u64; 4])> {
    std::fs::read_to_string(p.join("summary.txt"))
        .unwrap_or_default()
        .lines()
        .map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            let num = |k: &str| {
                w.iter()
                    .position(|x| *x == k)
                    .map_or(0, |i| w[i + 1].parse().unwrap_or(0))
            };
            (
                w[..2].join(" "),
                [num("contacts"), num("awake"), num("bodies"), num("broken")],
            )
        })
        .collect()
}

const DYNAMIC: u32 = 1;
const SLEEPING: u32 = 4;

#[test]
#[ignore]
fn compare() {
    let Ok(dirs) = std::env::var("EQUIV_COMPARE") else {
        return;
    };
    let (a, b) = dirs.split_once(',').expect("EQUIV_COMPARE=dirA,dirB");
    let (a, b) = (PathBuf::from(a), PathBuf::from(b));
    let mut scenes: Vec<_> = std::fs::read_dir(&a)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .collect();
    scenes.sort();
    println!("| scene | dumps | max |dx| m | mean |dx| m | max |dv| m/s | awake differs (bodies) | contacts A/B (last) | bonds A/B | below ground A/B | NaN A/B | bodies A/B |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for s in scenes {
        let (da, db) = (a.join(&s), b.join(&s));
        if !db.exists() {
            continue;
        }
        let mut files: Vec<_> = std::fs::read_dir(&da)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|f| f.to_string_lossy().ends_with(".bin"))
            .collect();
        files.sort();
        let (mut dmax, mut dsum, mut dn, mut vmax, mut awake_diff) =
            (0f32, 0f64, 0u64, 0f32, 0usize);
        let (mut below, mut nan, mut count) = ([0usize; 2], [0usize; 2], [0usize; 2]);
        for f in &files {
            if !db.join(f).exists() {
                continue;
            }
            let (x, y) = (load(&da.join(f)), load(&db.join(f)));
            count = [x.pos.len(), y.pos.len()];
            for (k, side) in [&x, &y].iter().enumerate() {
                below[k] = below[k].max(
                    side.pos
                        .iter()
                        .zip(&side.flags)
                        .filter(|(p, &fl)| fl & DYNAMIC != 0 && p[1] < -1.0)
                        .count(),
                );
                nan[k] = nan[k].max(
                    side.pos
                        .iter()
                        .chain(side.vel.iter())
                        .filter(|p| p.iter().any(|v| !v.is_finite()))
                        .count(),
                );
            }
            for i in 0..x.pos.len().min(y.pos.len()) {
                // Parked slots (fracture piece pools, the demo's projectile) are not bodies.
                if x.pos[i][1] < -1000.0 || y.pos[i][1] < -1000.0 {
                    continue;
                }
                let d = (0..3)
                    .map(|k| (x.pos[i][k] - y.pos[i][k]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                let dv = (0..3)
                    .map(|k| (x.vel[i][k] - y.vel[i][k]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if d.is_finite() {
                    dmax = dmax.max(d);
                    dsum += d as f64;
                    dn += 1;
                }
                if dv.is_finite() {
                    vmax = vmax.max(dv);
                }
                let dyn_ = x.flags[i] & DYNAMIC != 0 && y.flags[i] & DYNAMIC != 0;
                if dyn_ && (x.flags[i] & SLEEPING) != (y.flags[i] & SLEEPING) {
                    awake_diff += 1;
                }
            }
        }
        let (sa, sb) = (summary(&da), summary(&db));
        let last = |v: &Vec<(String, [u64; 4])>, k: usize| v.last().map_or(0, |l| l.1[k]);
        let bonds = |v: &Vec<(String, [u64; 4])>| -> u64 {
            if s.to_string_lossy().starts_with("house") || s.to_string_lossy().starts_with("arch") {
                v.iter().map(|l| l.1[3]).sum()
            } else {
                last(v, 3)
            }
        };
        println!(
            "| {} | {} | {:.4} | {:.5} | {:.3} | {} | {}/{} | {}/{} | {}/{} | {}/{} | {}/{} |",
            s.to_string_lossy(),
            files.len(),
            dmax,
            dsum / dn.max(1) as f64,
            vmax,
            awake_diff,
            last(&sa, 0),
            last(&sb, 0),
            bonds(&sa),
            bonds(&sb),
            below[0],
            below[1],
            nan[0],
            nan[1],
            count[0],
            count[1]
        );
        if s.to_string_lossy().starts_with("house") || s.to_string_lossy().starts_with("arch") {
            let per: Vec<i64> = sa
                .iter()
                .zip(&sb)
                .map(|(x, y)| y.1[3] as i64 - x.1[3] as i64)
                .collect();
            let mean = per.iter().sum::<i64>() as f64 / per.len().max(1) as f64;
            let same = per.iter().filter(|d| **d == 0).count();
            println!("|   {} shots: bonds per shot A {:.1}, B {:.1}, paired B-A {mean:+.2}, identical shots {same}/{} | | | | | | | | | | |",
                s.to_string_lossy(), bonds(&sa) as f64 / sa.len().max(1) as f64, bonds(&sb) as f64 / sb.len().max(1) as f64, per.len());
        }
    }
}

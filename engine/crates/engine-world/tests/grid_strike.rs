//! A settled, fully asleep box grid struck by the demo's ball (`demo.html?scene=legacyGrid100k`,
//! F): the resting contacts' policies against each other. Probe (ignored):
//! `STRIKE_SCENE` (default "100k Grid"), `STRIKE_POLICY` = archive | carry | cold | never,
//! `STRIKE_FIRE` (step, default 400), `STRIKE_STEPS` (default 1000). Prints the awake count
//! every 50 steps and, at the end, how many boxes moved more than 0.1 m and 1 m.

use rubble::*;

struct Strike {
    awake: Vec<(u32, u32)>,
    moved: usize,
    moved_far: usize,
}

fn env_u(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn strike(scene: &str, policy: &str, fire: u32, steps: u32) -> Strike {
    let (host, _) = engine_scenes::build_host(scene).unwrap();
    let snap = host.snapshot();
    let n = snap.bodies.len() as u32 + 4;
    let mut def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: n.next_power_of_two(),
            max_colliders: n.next_power_of_two(),
            max_contacts: (16 * n).next_power_of_two(),
            max_candidates: (16 * n).next_power_of_two(),
            max_colors: 32,
            ..WorldCapacity::small()
        },
        enable_sleep: policy != "never",
        contacts: if std::env::var("STRIKE_CANONICAL").is_ok() {
            ContactStorage::Canonical
        } else {
            ContactStorage::Persistent
        },
        ..WorldDef::default()
    };

    if std::env::var("STRIKE_FEATURES_ENV").is_ok() {
        def.features = engine_webgpu::EngineFeatures::from_env();
    }
    match policy {
        "carry" => def.features.resting_contacts = RestingContacts::Carry,
        "cold" => def.features.archive_cold = true,
        "drop" => def.features.resting_contacts = RestingContacts::Drop,
        _ => {}
    }
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let imported = w.import_snapshot(&snap).unwrap();
    let mut solver = w.solver();
    // The demo's sleep (0.5 s) or never.
    solver.sleep_time = if policy == "never" { 1.0e6 } else { 0.5 };
    // STRIKE_DEMO_SOLVER=1: demo-config.mjs demoSolver (flags 0, alpha 0.99, betaLin 1e4,
    // sleep thresholds 0.15).
    if std::env::var("STRIKE_DEMO_SOLVER").is_ok() {
        solver.flags = 0;
        solver.alpha = 0.99;
        solver.beta_lin = 10000.0;
        solver.sleep_lin = 0.15;
        solver.sleep_ang = 0.15;
        solver.dt = 1.0 / 60.0;
    }
    w.set_solver(solver);
    let boxes: Vec<BodyId> = imported.bodies.iter().copied().skip(1).collect();
    // STRIKE_PARKED=1: the demo's way (scene_export.rs, joint-interaction.mjs): a static
    // sphere parked below the ground from the start, a bullet, launched at the fire step.
    let parked = std::env::var("STRIKE_PARKED").is_ok().then(|| {
        let b = w
            .create_body(&BodyDef { position: [0.0, -40.0, 0.0], ..BodyDef::default() })
            .unwrap();
        w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.35).unwrap();
        if std::env::var("STRIKE_NO_CCD").is_err() {
            w.body_set_bullet(b, true).unwrap();
        }
        b
    });
    let mut awake = Vec::new();
    let mut start = Vec::new();
    for s in 1..=steps {
        if s == fire {
            start = boxes.iter().map(|&b| w.body_transform(b).unwrap().unwrap().position).collect();
            // joint-interaction.mjs fire(): 8 kg, r 0.35, from the camera 2 m along the
            // view ray at 30 m/s (eye [46, 30, 58] towards look [0, 9, 0]).
            let eye = glam::Vec3::new(46.0, 30.0, 58.0);
            let dir = (glam::Vec3::new(0.0, 9.0, 0.0) - eye).normalize();
            if let Some(b) = parked {
                let t = Transform { position: (eye + dir * 2.0).to_array(), ..Transform::default() };
                w.body_set_type(b, BodyType::Dynamic, t, (dir * 30.0).to_array(), [0.0; 3]).unwrap();
                let i = 0.4 * 8.0 * 0.35 * 0.35;
                w.body_set_mass_data(
                    b,
                    MassData { mass: 8.0, center: [0.0; 3], rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]] },
                )
                .unwrap();
            } else {
            let b = w
                .create_body(&BodyDef {
                    linear_velocity: (dir * 30.0).to_array(),
                    // The demo's projectile is a bullet (CCD) unless ?ccd=0.
                    is_bullet: std::env::var("STRIKE_NO_CCD").is_err(),
                    ..BodyDef::dynamic((eye + dir * 2.0).to_array())
                })
                .unwrap();
            let r = 0.35f32;
            let m = 8.0f32;
            let d = ShapeDef { density: m / (4.0 / 3.0 * std::f32::consts::PI * r * r * r), ..ShapeDef::default() };
            w.create_sphere_shape(b, &d, [0.0; 3], r).unwrap();
            }
        }
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if s % 50 == 0 {
            awake.push((s, w.counters().map_or(0, |c| c.awake_count)));
            if std::env::var("STRIKE_BELOW").is_ok() && s >= fire {
                let below: Vec<usize> = boxes
                    .iter()
                    .enumerate()
                    .filter(|(_, &b)| w.body_transform(b).unwrap().unwrap().position[1] < 0.1)
                    .map(|(k, _)| k)
                    .collect();
                eprintln!("strike: step {s}: {} boxes below 0.1 m, first {:?}", below.len(), &below[..below.len().min(5)]);
            }
        }
    }
    let d: Vec<f32> = boxes
        .iter()
        .zip(&start)
        .map(|(&b, p)| glam::Vec3::from(w.body_transform(b).unwrap().unwrap().position).distance(glam::Vec3::from(*p)))
        .collect();
    let ys: Vec<f32> = boxes.iter().map(|&b| w.body_transform(b).unwrap().unwrap().position[1]).collect();
    let ymin = ys.iter().cloned().fold(f32::MAX, f32::min);
    eprintln!("strike: lowest box y {ymin:.2}, boxes below 0: {}", ys.iter().filter(|&&y| y < 0.0).count());
    Strike {
        awake,
        moved: d.iter().filter(|&&x| x > 0.1).count(),
        moved_far: d.iter().filter(|&&x| x > 1.0).count(),
    }
}

#[test]
#[ignore]
fn grid_strike_probe() {
    let scene = std::env::var("STRIKE_SCENE").unwrap_or_else(|_| "100k Grid".into());
    let policy = std::env::var("STRIKE_POLICY").unwrap_or_else(|_| "archive".into());
    let r = strike(&scene, &policy, env_u("STRIKE_FIRE", 400), env_u("STRIKE_STEPS", 1000));
    println!("strike {scene} {policy}: moved>0.1 {} moved>1 {} awake {:?}", r.moved, r.moved_far, r.awake);
}

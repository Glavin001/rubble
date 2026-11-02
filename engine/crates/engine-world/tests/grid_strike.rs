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
        ..WorldDef::default()
    };
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
    w.set_solver(solver);
    let boxes: Vec<BodyId> = imported.bodies.iter().copied().skip(1).collect();
    let mut awake = Vec::new();
    let mut start = Vec::new();
    for s in 1..=steps {
        if s == fire {
            start = boxes.iter().map(|&b| w.body_transform(b).unwrap().unwrap().position).collect();
            // joint-interaction.mjs fire(): 8 kg, r 0.35, from the camera 2 m along the
            // view ray at 30 m/s (eye [46, 30, 58] towards look [0, 9, 0]).
            let eye = glam::Vec3::new(46.0, 30.0, 58.0);
            let dir = (glam::Vec3::new(0.0, 9.0, 0.0) - eye).normalize();
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
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if s % 50 == 0 {
            awake.push((s, w.counters().map_or(0, |c| c.awake_count)));
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

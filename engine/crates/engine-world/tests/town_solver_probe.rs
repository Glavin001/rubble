//! Probe (ignored): Vibe Town's pack under solver settings, one variant per run: does the town
//! hold, and do a ball and a meteor land? Prints the shots' path and how far chunks went.
//! RUBBLE_TOWN_PACK=/path/vibe-town.json TOWN_SOLVER=lab cargo test -p engine-world --release \
//!   --test town_solver_probe -- --ignored --nocapture
//! TOWN_SOLVER: default | lab | a comma list of `field=value` edits on the lab's (or `default:`
//! first for the default's), e.g. `lab,flags=0` or `default:dt=0.008333`.
use rubble::*;

fn params(spec: &str, lab: AvbdParams) -> AvbdParams {
    let (mut p, rest) = match spec.split_once(':') {
        Some(("default", r)) => (AvbdParams::default(), r),
        _ if spec == "default" => (AvbdParams::default(), ""),
        _ => (lab, spec.strip_prefix("lab").unwrap_or(spec)),
    };
    for edit in rest.split(',').filter(|e| !e.is_empty()) {
        let (k, v) = edit.split_once('=').expect("field=value");
        let f: f32 = v.parse().unwrap();
        match k {
            "dt" => p.dt = f,
            "iterations" => p.iterations = f as u32,
            "flags" => p.flags = f as u32,
            "alpha" => p.alpha = f,
            "gamma" => p.gamma = f,
            "beta_lin" => p.beta_lin = f,
            "beta_ang" => p.beta_ang = f,
            "k_start" => p.k_start = f,
            "k_max_n" => p.k_max_n = f,
            "k_max_t" => p.k_max_t = f,
            "contact_offset" => p.contact_offset = f,
            "sleep_lin" => p.sleep_lin = f,
            "sleep_ang" => p.sleep_ang = f,
            "sleep_time" => p.sleep_time = f,
            "trust_lin" => p.trust_lin = f,
            "trust_ang" => p.trust_ang = f,
            "grid_cell_size" => p.grid_cell_size = f,
            other => panic!("unknown field {other}"),
        }
    }
    p
}

#[test]
#[ignore]
fn town_under_solver() {
    let Ok(path) = std::env::var("RUBBLE_TOWN_PACK") else { return };
    let spec = std::env::var("TOWN_SOLVER").unwrap_or_else(|_| "default".into());
    let desc = DestructibleAssetDesc::from_pack(&std::fs::read(&path).unwrap()).unwrap();
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 1 << 17,
            max_colliders: 1 << 18,
            max_contacts: 1 << 21,
            max_joints: 1 << 18,
            max_candidates: 1 << 21,
            max_colors: 64,
            max_commands: 4096,
            max_hull_vertices: 1 << 21,
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let p = params(&spec, desc.solver.expect("a pack carries the lab's solver"));
    println!("{spec}: {p:?}");
    w.set_solver(p);
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: 0.0 }).unwrap();
    let asset = w.create_destructible_asset(&desc).unwrap();
    let d = w.create_destructible(&DestructibleDef::new(asset)).unwrap();
    let n = desc.chunks.len() as u32;
    let bodies: Vec<BodyId> = (0..n).map(|c| w.destructible_chunk_body(d, c).unwrap()).collect();
    let poses: Vec<Transform> = (0..n).map(|c| w.destructible_chunk_pose(d, c).unwrap()).collect();
    let centres = |w: &World| -> Vec<glam::Vec3> {
        bodies
            .iter()
            .zip(&poses)
            .map(|(b, l)| {
                let t = w.body_transform(*b).unwrap().unwrap_or_else(|| w.body_initial_transform(*b).unwrap());
                glam::Quat::from_array(t.rotation) * glam::Vec3::from(l.position) + glam::Vec3::from(t.position)
            })
            .collect()
    };
    let steps_per_s = (1.0 / p.dt).round() as u32;
    let step = |w: &mut World, k: u32| -> u32 {
        let mut broken = 0;
        for _ in 0..k {
            w.step().unwrap();
            for f in w.poll_blocking().unwrap() {
                for e in &f.destruction_events {
                    if let DestructionEvent::BondsBroken { bonds, .. } = e {
                        broken += bonds.len() as u32;
                    }
                }
            }
        }
        broken
    };
    step(&mut w, steps_per_s);
    let start = centres(&w);
    // Unreal's Rubble.Video.Town hits: a 7 t iron ball at 45 m/s into house-85's wall, a 22 t
    // meteor straight down onto house-89 at 50 m/s.
    let shot = |w: &mut World, from: [f32; 3], v: [f32; 3], r: f32, density: f32| {
        let b = w.create_body(&BodyDef { linear_velocity: v, is_bullet: true, ..BodyDef::dynamic(from) }).unwrap();
        w.create_sphere_shape(b, &ShapeDef { density, ..ShapeDef::default() }, [0.0; 3], r).unwrap();
        b
    };
    let ball = shot(&mut w, [-51.2, 2.5, 40.0], [0.0, 0.0, -45.0], 0.6, 7800.0);
    let meteor = shot(&mut w, [-34.2, 50.0, 15.5], [0.0, -50.0, 0.0], 1.2, 3000.0);
    let mut total = 0;
    for tenth in 1..=60 {
        total += step(&mut w, steps_per_s / 10);
        let at = |b| glam::Vec3::from(w.body_transform(b).unwrap().unwrap().position);
        if tenth == 6 && std::env::var("TOWN_SHORT").is_ok() {
            println!("  t 0.6 s: meteor {:.2?} (18.2 with gravity, 20.0 without)", at(meteor));
            return;
        }
        if tenth <= 15 && tenth % 3 == 0 {
            println!("  t {:.1} s: ball {:.1?}, meteor {:.1?}, bonds broken {total}", tenth as f32 / 10.0, at(ball), at(meteor));
        }
        if tenth % 10 == 0 {
            let now = centres(&w);
            let (mut moved, mut far, mut top) = (0, 0, 0.0f32);
            for (a, b) in start.iter().zip(&now) {
                moved += ((*b - *a).length() > 0.1) as u32;
                far += ((*b - *a).length() > 5.0) as u32;
                top = top.max(b.y - a.y);
            }
            println!("  t {} s: bonds broken {total}; chunks moved >0.1 m {moved}, >5 m {far}; most risen {top:.1} m", tenth / 10);
        }
    }
}

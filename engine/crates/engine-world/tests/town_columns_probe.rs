//! Probe (ignored): Vibe Town under the threshold model, tower-187's ground-storey columns shot
//! from the side. Prints what still holds the tower up: the roof's actor (anchored or not) and
//! the ground-storey pieces still on it, by type.
//!   RUBBLE_TOWN_PACK=/path/vibe-town.json cargo test -p engine-world --release \
//!     --test town_columns_probe -- --ignored --nocapture
//! TOWN_SHOTS: `x` (each column row along +x, default), `xz` (also each column line along +z);
//! TOWN_SPEED (m/s, 45), TOWN_RADIUS (m, 0.6), TOWN_MASS (kg, 7000), TOWN_HEIGHT (m above the
//! ground, 1.3), TOWN_HEIGHTS (a comma list, a volley at each), TOWN_SUPPORT_LOSS (the
//! support-loss capability), TOWN_VOLLEYS (the shots again each second, this many in all).
#![cfg(feature = "stress")]

use rubble::*;
use std::collections::BTreeMap;

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore]
fn tower_columns_shot_from_the_side() {
    let Ok(path) = std::env::var("RUBBLE_TOWN_PACK") else { return };
    let bytes = std::fs::read(&path).unwrap();
    let pack: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let sc = &pack["scenario"];
    let group = |i: usize| sc["nodeGroups"][i].as_str().unwrap_or("").to_string();
    let kind = |i: usize| sc["nodeTypes"][i].as_str().unwrap_or("").to_string();
    let y_of = |i: usize| sc["nodes"][i]["centroid"]["y"].as_f64().unwrap();
    let asset = DestructibleAssetDesc::from_pack(&bytes).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 1 << 17,
            max_colliders: 1 << 18,
            max_contacts: 2_000_000,
            max_joints: 1 << 18,
            max_candidates: 1 << 21,
            max_colors: 64,
            max_commands: 4096,
            max_hull_vertices: 1 << 21,
        },
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    // The lab's ground at y = 0: the town lowered onto it.
    let ground = stress.ground_height().unwrap_or(0.0);
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: 0.0 }).unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let d = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new([0.0, -ground, 0.0], [0.0, 0.0, 0.0, 1.0]),
            support_loss: std::env::var("TOWN_SUPPORT_LOSS").is_ok(),
            ..DestructibleDef::new(a)
        })
        .unwrap();
    // TOWN_TOWER: the tower's group (tower@tower-187); TOWN_FROM: the side shot from (-x, +x,
    // -z, +z; default -x); TOWN_TARGET: a building it should fall on (its first chunk is watched).
    let tower_name = std::env::var("TOWN_TOWER").unwrap_or_else(|_| "tower@tower-187".into());
    let from = std::env::var("TOWN_FROM").unwrap_or_else(|_| "-x".into());
    let tower: Vec<usize> = (0..asset.chunks.len()).filter(|&i| group(i) == tower_name).collect();
    let x_of = |i: usize| sc["nodes"][i]["centroid"]["x"].as_f64().unwrap() as f32;
    let z_of = |i: usize| sc["nodes"][i]["centroid"]["z"].as_f64().unwrap() as f32;
    let target_name = std::env::var("TOWN_TARGET").ok();
    let target: Vec<usize> = target_name.as_ref().map_or(Vec::new(), |t| (0..asset.chunks.len()).filter(|&i| &group(i) == t).collect());
    let roof = *tower.iter().max_by(|&&a, &&b| y_of(a).total_cmp(&y_of(b))).unwrap() as u32;
    let ground_storey: Vec<usize> = tower.iter().copied().filter(|&i| y_of(i) < 3.4 && kind(i) != "foundation").collect();
    let (speed, r, m, h) = (env("TOWN_SPEED", 45.0), env("TOWN_RADIUS", 0.6), env("TOWN_MASS", 7000.0), env("TOWN_HEIGHT", 1.3));
    // The pack's footprint is centred (x shift +0.03, z shift -0.365) and lifted 0.5 m.
    let (sx, sz) = (0.03f32, -0.365f32);
    // TOWN_HEIGHTS: a comma list of heights in place of TOWN_HEIGHT (a volley at each).
    let heights: Vec<f32> = std::env::var("TOWN_HEIGHTS").map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_else(|_| vec![h]);
    // The column lines (pack frame) and the tower's footprint.
    let columns: Vec<usize> = tower.iter().copied().filter(|&i| kind(i) == "column" && y_of(i) < 3.4).collect();
    let mut xs: Vec<f32> = columns.iter().map(|&i| (x_of(i) * 10.0).round() / 10.0).collect();
    let mut zs: Vec<f32> = columns.iter().map(|&i| (z_of(i) * 10.0).round() / 10.0).collect();
    xs.sort_by(f32::total_cmp);
    xs.dedup();
    zs.sort_by(f32::total_cmp);
    zs.dedup();
    let (x0, x1) = (xs[0] - 7.0, xs[xs.len() - 1] + 7.0);
    let (z0, z1) = (zs[0] - 5.0, zs[zs.len() - 1] + 5.0);
    let (mut xs, mut zs) = (xs, zs);
    // TOWN_LINES: shoot only the first n lines (from the low end), e.g. one side's rows.
    if let Some(n) = std::env::var("TOWN_LINES").ok().and_then(|v| v.parse::<usize>().ok()) {
        xs.truncate(n);
        zs.truncate(n);
    }
    println!("{tower_name}: column lines x {xs:?}, z {zs:?}; shot from {from}");
    let mut shots: Vec<([f32; 3], [f32; 3])> = Vec::new();
    for &h in &heights {
        match from.as_str() {
            "-x" => shots.extend(zs.iter().map(|z| ([x0 + sx, h, z + sz], [speed, 0.0, 0.0]))),
            "+x" => shots.extend(zs.iter().map(|z| ([x1 + sx, h, z + sz], [-speed, 0.0, 0.0]))),
            "-z" => shots.extend(xs.iter().map(|x| ([x + sx, h, z0 + sz], [0.0, 0.0, speed]))),
            _ => shots.extend(xs.iter().map(|x| ([x + sx, h, z1 + sz], [0.0, 0.0, -speed]))),
        }
    }
    let fire = |w: &mut World, shots: &[([f32; 3], [f32; 3])]| {
        for (at, v) in shots {
            let b = w.create_body(&BodyDef { linear_velocity: *v, is_bullet: true, ..BodyDef::dynamic(*at) }).unwrap();
            w.create_sphere_shape(b, &ShapeDef { update_body_mass: false, ..ShapeDef::default() }, [0.0; 3], r).unwrap();
            let i = 0.4 * m * r * r;
            w.body_set_mass_data(b, MassData { mass: m, center: [0.0; 3], rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]] }).unwrap();
            w.body_set_projectile(b, Some(Projectile::default())).unwrap();
        }
    };
    fire(&mut w, &shots);
    // TOWN_VOLLEYS: the same shots again each second, this many in all (each reaches further).
    let volleys = env("TOWN_VOLLEYS", 1.0) as u32;
    println!("{} shots at {speed} m/s, r {r} m, {m} kg, {h} m up; tower-187: {} chunks, {} in the ground storey", shots.len(), tower.len(), ground_storey.len());
    let mut broken = 0;
    let hz = (1.0 / solver.dt).round() as u32;
    for second in 1..=8 {
        if second < volleys + 1 && second > 1 {
            fire(&mut w, &shots);
        }
        for _ in 0..hz {
            w.step().unwrap();
            for f in w.poll_blocking().unwrap() {
                for e in &f.destruction_events {
                    if let DestructionEvent::BondsBroken { bonds, .. } = e {
                        broken += bonds.len();
                    }
                }
            }
        }
        let actor = w.destructible_actor_of(d, roof).unwrap();
        let body = w.destructible_chunk_body(d, roof).unwrap();
        let y = w.body_transform(body).unwrap().map_or(0.0, |t| t.position[1]);
        // The roof chunk's centre (body x its pose), in the pack's frame.
        let roof_at = w.body_transform(body).unwrap().map(|t| {
            let pose = w.destructible_chunk_pose(d, roof).unwrap();
            let p = glam::Quat::from_array(t.rotation) * glam::Vec3::from(pose.position) + glam::Vec3::from(t.position);
            [p.x, p.y + ground, p.z]
        });
        println!("  roof centre {:.1?}", roof_at);
        let on: std::collections::HashSet<u32> = actor.chunks.iter().copied().collect();
        let mut holding: BTreeMap<String, usize> = BTreeMap::new();
        for &i in &ground_storey {
            if on.contains(&(i as u32)) {
                *holding.entry(kind(i)).or_default() += 1;
            }
        }
        if let Some(&c) = target.first() {
            let a = w.destructible_actor_of(d, c as u32).unwrap();
            println!("  {}: its first chunk's actor {} of {} chunks, anchored {}", target_name.as_deref().unwrap_or(""), a.chunks.len(), target.len(), a.anchored);
        }
        println!("t {second} s: bonds broken {broken}; roof's actor {} chunks, anchored {}, its body moved {:.2} m up; ground storey still on it: {holding:?}", actor.chunks.len(), actor.anchored, y - (-ground));
    }
}

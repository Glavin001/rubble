//! Probe (ignored): the glass tower felled onto its pavilion under the threshold model with support
//! loss, as the Unreal film shoots it (Rubble.Video.GlassTower), in seconds instead of a render.
//! Prints each second the tower's lean (its roof's body: degrees from upright and the direction its
//! top leans, +z north, -z south toward the pavilion) and how many pavilion chunks are knocked loose.
//!   RUBBLE_TOWN_PACK=/path/glass-tower.json cargo test -p engine-world --release --features stress \
//!     --test glass_tower_felling_probe -- --ignored --nocapture
//! GT_PLAN picks the shot plan (see `plan`); GT_SECONDS how long to run (20).
#![cfg(feature = "stress")]

use rubble::*;

/// One shot in the world's frame (Y up, m, the ground at y = 0): when, where from, its velocity.
struct Shot {
    at: f32,
    from: [f32; 3],
    v: [f32; 3],
}

/// The world frame (the pack's footprint centred by its scene's placement: the tower at z +9.99 m,
/// the pavilion at z -10.99) from the Unreal film's (X, Y, Z cm): x = Y, y = Z, z = -X.
fn unreal(at: f32, x: f32, y: f32, z: f32, dir: [f32; 3], speed: f32) -> Shot {
    // dir in Unreal axes: (dX, dY, dZ) -> (dY, dZ, -dX).
    Shot { at, from: [y / 100.0, z / 100.0, -x / 100.0], v: [dir[1] * speed, dir[2] * speed, -dir[0] * speed] }
}

fn plan(name: &str) -> Vec<Shot> {
    let mut shots = Vec::new();
    let west = |at: f32, x: f32, z: f32| unreal(at, x, -1406.0, z, [0.0, 1.0, 0.0], 45.0);
    match name {
        "none" => {}
        // The town film's: every storey up to the fifth along the three southern rows.
        "storeys" => {
            for at in [1.0, 2.0, 3.0] {
                for x in [-1174.0, -824.0, -474.0] {
                    for z in [100.0, 260.0, 530.0, 830.0, 1130.0] {
                        shots.push(west(at, x, z));
                    }
                }
            }
        }
        // The storeys plan, its volleys alternately from the west and the east faces, so their
        // momentum cancels instead of shoving the tower east past the pavilion.
        "both" => {
            for (k, at) in [1.0, 2.0, 3.0, 4.0].into_iter().enumerate() {
                for x in [-1174.0, -824.0, -474.0] {
                    for z in [100.0, 260.0, 530.0, 830.0, 1130.0] {
                        shots.push(if k % 2 == 0 { west(at, x, z) } else { unreal(at, x, 1406.0, z, [0.0, -1.0, 0.0], 45.0) });
                    }
                }
            }
        }
        // From the gap between the buildings (X +300 cm) north (-X) into the tower's south face, one
        // ball down each of its four column lines (Y -525, -175, 175, 525 cm) at its lowest storeys:
        // the cut is symmetric across its width, and the push at its foot is away from the pavilion.
        n if n.starts_with("south") => {
            let ups: &[f32] = if n == "south5" { &[100.0, 260.0, 530.0, 830.0, 1130.0] } else { &[100.0, 260.0, 530.0] };
            for at in [1.0, 2.0, 3.0] {
                for y in [-525.0, -175.0, 175.0, 525.0] {
                    for &z in ups {
                        shots.push(unreal(at, 300.0, y, z, [-1.0, 0.0, 0.0], 45.0));
                    }
                }
            }
        }
        // The ground storey of the three southern rows.
        "ground" => {
            for at in [1.0, 2.0, 3.0, 4.0] {
                for x in [-1174.0, -824.0, -474.0] {
                    for z in [80.0, 160.0, 240.0] {
                        shots.push(west(at, x, z));
                    }
                }
            }
        }
        // The storeys notch, then a back cut: the north row's ground storey once it leans.
        n if n.starts_with("felling") => {
            let back: f32 = n.trim_start_matches("felling").parse().unwrap_or(5.0);
            for at in [1.0, 2.0, 3.0] {
                for x in [-1174.0, -824.0, -474.0] {
                    for z in [100.0, 260.0, 530.0, 830.0, 1130.0] {
                        shots.push(west(at, x, z));
                    }
                }
            }
            for at in [back, back + 1.0] {
                for z in [80.0, 160.0, 240.0] {
                    shots.push(west(at, -1524.0, z));
                }
            }
            shots.sort_by(|a, b| a.at.total_cmp(&b.at));
        }
        // A felling wedge: deeper toward the pavilion.
        _ => {
            for at in [1.0, 2.0, 3.0, 4.0] {
                for (x, top) in [(-1174.0, 240.0), (-824.0, 530.0), (-474.0, 830.0)] {
                    for z in [80.0, 160.0, 240.0, 530.0, 830.0] {
                        if z <= top {
                            shots.push(west(at, x, z));
                        }
                    }
                }
            }
        }
    }
    shots
}

#[test]
#[ignore]
fn glass_tower_falls_onto_the_pavilion() {
    let Ok(path) = std::env::var("RUBBLE_TOWN_PACK") else { return };
    let bytes = std::fs::read(&path).unwrap();
    let pack: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let sc = &pack["scenario"];
    let group = |i: usize| sc["nodeGroups"][i].as_str().unwrap_or("").to_string();
    let y_of = |i: usize| sc["nodes"][i]["centroid"]["y"].as_f64().unwrap();
    let asset = DestructibleAssetDesc::from_pack(&bytes).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 1 << 15,
            max_colliders: 1 << 16,
            max_contacts: 2_000_000,
            max_joints: 1 << 14,
            max_candidates: 1 << 21,
            max_colors: 64,
            max_commands: 4096,
            max_hull_vertices: 1 << 20,
        },
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let ground = stress.ground_height().unwrap_or(0.0);
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: 0.0 }).unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let d = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new([0.0, -ground, 0.0], [0.0, 0.0, 0.0, 1.0]),
            support_loss: true,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    let n = asset.chunks.len();
    let tower: Vec<usize> = (0..n).filter(|&i| group(i).starts_with("tower@")).collect();
    let pavilion: Vec<usize> = (0..n).filter(|&i| group(i).starts_with("pavilion@")).collect();
    let roof = *tower.iter().max_by(|&&a, &&b| y_of(a).total_cmp(&y_of(b))).unwrap() as u32;
    let name = std::env::var("GT_PLAN").unwrap_or_else(|_| "wedge".into());
    let shots = plan(&name);
    let seconds = std::env::var("GT_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20u32);
    println!("plan {name}: {} shots; tower {} chunks, pavilion {}", shots.len(), tower.len(), pavilion.len());
    for s in shots.iter().take(3) {
        println!("  shot at {} s from {:?} at {:?} m/s", s.at, s.from, s.v);
    }
    let hz = (1.0 / solver.dt).round() as u32;
    let (r, m) = (0.6f32, 7000.0f32);
    let mut fired = 0;
    for step in 0..seconds * hz {
        let t = step as f32 / hz as f32;
        while fired < shots.len() && shots[fired].at <= t {
            let s = &shots[fired];
            let b = w.create_body(&BodyDef { linear_velocity: s.v, is_bullet: true, ..BodyDef::dynamic(s.from) }).unwrap();
            w.create_sphere_shape(b, &ShapeDef { update_body_mass: false, ..ShapeDef::default() }, [0.0; 3], r).unwrap();
            let i = 0.4 * m * r * r;
            w.body_set_mass_data(b, MassData { mass: m, center: [0.0; 3], rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]] }).unwrap();
            w.body_set_projectile(b, Some(Projectile::default())).unwrap();
            fired += 1;
        }
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if (step + 1) % hz == 0 {
            let body = w.destructible_chunk_body(d, roof).unwrap();
            let lean = w.body_transform(body).unwrap().map(|tf| {
                let up = glam::Quat::from_array(tf.rotation) * glam::Vec3::Y;
                (up.y.clamp(-1.0, 1.0).acos().to_degrees(), up.x, up.z)
            });
            let loose = pavilion.iter().filter(|&&i| !w.destructible_actor_of(d, i as u32).unwrap().anchored).count();
            // Where the roof is (world frame; the pavilion's footprint x -6..6, z -17..-5).
            let roof_at = w.body_transform(body).unwrap().map(|tf| {
                let pose = w.destructible_chunk_pose(d, roof).unwrap();
                glam::Quat::from_array(tf.rotation) * glam::Vec3::from(pose.position) + glam::Vec3::from(tf.position)
            });
            if let Some(p) = roof_at {
                println!("  roof at x {:+.1} y {:.1} z {:+.1}", p.x, p.y, p.z);
            }
            if std::env::var("GT_DEBUG").is_ok() {
                let tower_loose = tower.iter().filter(|&&i| !w.destructible_actor_of(d, i as u32).unwrap().anchored).count();
                let roof_actor = w.destructible_actor_of(d, roof).unwrap();
                println!("  tower chunks loose {tower_loose}; roof's actor {} chunks, anchored {}", roof_actor.chunks.len(), roof_actor.anchored);
                // What of the lowest two storeys is still on the roof's piece, by type and row (pack z).
                let on: std::collections::HashSet<u32> = roof_actor.chunks.iter().copied().collect();
                let mut low: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                for &i in &tower {
                    if y_of(i) < 6.5 && on.contains(&(i as u32)) {
                        let z = sc["nodes"][i]["centroid"]["z"].as_f64().unwrap();
                        *low.entry(format!("{} z{:+.0}", sc["nodeTypes"][i].as_str().unwrap_or(""), z)).or_default() += 1;
                    }
                }
                println!("  low storeys on the roof's piece: {low:?}");
                if let Some(&i) = pavilion.iter().find(|&&i| !w.destructible_actor_of(d, i as u32).unwrap().anchored) {
                    let b = w.destructible_chunk_body(d, i as u32).unwrap();
                    let c = &sc["nodes"][i]["centroid"];
                    println!("  loose pavilion chunk {i} ({}, node at {c}): body at {:?}", sc["nodeTypes"][i], w.body_transform(b).unwrap().map(|t| t.position));
                }
            }
            match lean {
                Some((deg, x, z)) => println!(
                    "t {:>2} s: tower leans {deg:5.1} deg (top toward x {x:+.2}, z {z:+.2}); pavilion chunks loose {loose}/{}",
                    (step + 1) / hz,
                    pavilion.len()
                ),
                None => println!("t {:>2} s: roof body gone; pavilion chunks loose {loose}/{}", (step + 1) / hz, pavilion.len()),
            }
        }
    }
}

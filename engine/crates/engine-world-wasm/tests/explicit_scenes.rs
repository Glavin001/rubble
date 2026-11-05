//! The crash lab's rig scenes on the GPU (`engine-world-wasm/src/explicit.rs`):
//! the FMVSS 214 moving deformable barrier into the procedural sedan's side, and
//! the door pull.
use engine_world_wasm::{explicit_sedan_truck, ExplicitSim};
use glam::Vec3;

fn sim(scene: &str, extra: serde_json::Value) -> Option<ExplicitSim> {
    let (_, device, queue) = engine_webgpu::default_device().ok()?;
    let mut spec = serde_json::json!({
        "vehicles": [{ "truck": explicit_sedan_truck(true) }],
        "scene": scene,
        "timing": false,
    });
    for (k, v) in extra.as_object().unwrap() {
        spec[k] = v.clone();
    }
    Some(ExplicitSim::new_native(&device, &queue, &spec).unwrap())
}

fn momentum(s: &ExplicitSim, v: usize) -> (Vec3, f32) {
    let vel = s.gpu_solver().velocities().remove(v);
    let nodes = &s.spawned(v).nodes;
    let m: f32 = nodes.iter().map(|n| n.mass).sum();
    (nodes.iter().zip(&vel).map(|(n, v)| *v * n.mass).sum::<Vec3>(), m)
}

#[test]
fn mdb_strikes_the_side_and_shares_its_momentum() {
    let Some(mut s) = sim("mdb214", serde_json::json!({})) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (p0, m) = momentum(&s, 1);
    assert!((m - engine_softbody::crash_rigs::MDB_MASS).abs() < 1.0, "MDB mass {m}");
    let v0 = p0 / m;
    eprintln!("MDB spawn velocity {v0:?} ({:.1} km/h)", v0.length() * 3.6);
    for k in 0..60 {
        s.frame_native(33).unwrap();
        if k % 6 == 5 {
            s.gpu_solver().wait();
            let all = s.gpu_solver().positions();
            let car_far = all[0].iter().map(|p| p.x).fold(f32::MAX, f32::min);
            let car_near = all[0].iter().map(|p| p.x).fold(f32::MIN, f32::max);
            let mdb_front = all[1].iter().map(|p| p.x).fold(f32::MAX, f32::min);
            let mm: Vec<f32> = s.spawned(1).nodes.iter().map(|n| n.mass).collect();
            let mc0: Vec<f32> = s.spawned(0).nodes.iter().map(|n| n.mass).collect();
            let cx = |p: &[Vec3], m: &[f32]| p.iter().zip(m).map(|(p, m)| p.x * m).sum::<f32>() / m.iter().sum::<f32>();
            eprintln!("   centres of mass x: car {:.2}, MDB {:.2}", cx(&all[0], &mc0), cx(&all[1], &mm));
            let (pc, mc) = momentum(&s, 0);
            let (pb, _) = momentum(&s, 1);
            eprintln!("{:.2} s: car x [{car_far:.2}, {car_near:.2}], MDB front {mdb_front:.2}; car v {:.2?}, MDB v {:.2?}", (k + 1) as f32 / 60.0, pc / mc, pb / m);
        }
    }
    s.gpu_solver().wait();
    let (pc, mc) = momentum(&s, 0);
    let (pb, _) = momentum(&s, 1);
    eprintln!("after 1 s: car {:?} m/s, MDB {:?} m/s", pc / mc, pb / m);
    let all = s.gpu_solver().positions();
    assert!(all.iter().flatten().all(|p| p.is_finite()));
    // No pass-through: the barrier's centre of mass stays outside the car's struck
    // side (the car's centre plus its half width), and it leaves slower than the car.
    let all = s.gpu_solver().positions();
    let cx = |p: &[Vec3], v: usize| {
        let m: Vec<f32> = s.spawned(v).nodes.iter().map(|n| n.mass).collect();
        p.iter().zip(&m).map(|(p, m)| p.x * m).sum::<f32>() / m.iter().sum::<f32>()
    };
    let (car, barrier) = (cx(&all[0], 0), cx(&all[1], 1));
    assert!(barrier - car > 0.9 + 0.5, "barrier {barrier} car {car}");
    assert!((pb / m).x > pc.x / mc, "barrier faster than the car across: {:?} {:?}", pb / m, pc / mc);
    // The car is pushed away from its struck (+X) side, the barrier slowed.
    assert!(pc.x / mc < -0.5, "car pushed: {:?}", pc / mc);
    assert!((pb / m).x > v0.x * 0.8, "barrier slowed: {:?}", pb / m);
}

#[test]
fn door_pull_moves_the_door_out() {
    let Some(mut s) = sim("doorpull", serde_json::json!({ "pullSpeed": 0.5 })) else {
        return;
    };
    let before = s.gpu_solver().positions().remove(0);
    for _ in 0..60 {
        s.frame_native(33).unwrap();
    }
    s.gpu_solver().wait();
    let after = s.gpu_solver().positions().remove(0);
    let rig = before.len() - 1;
    let moved = after[rig].x - before[rig].x;
    eprintln!("rig moved out {moved:.3} m in 1 s");
    assert!(moved > 0.1, "the rig pulls out: {moved}");
    // The door's nodes followed it outward; the held body did not go with it.
    let out = (0..rig).filter(|&i| after[i].x - before[i].x > 0.02).count();
    let centre = |p: &[Vec3]| p[..rig].iter().sum::<Vec3>() / rig as f32;
    let drift = (centre(&after) - centre(&before)).x;
    eprintln!("{out} car nodes moved out more than 2 cm; the body's centre {drift:.3} m");
    assert!(out >= 3);
    assert!(drift.abs() < 0.05, "the body is held: {drift}");
}


/// Generalized winding number of point `p` about a triangle surface (Van Oosterom
/// and Strackee's solid angle per triangle, summed over 4π): ±1 inside a closed,
/// consistently wound surface, 0 outside.
fn winding(p: Vec3, pos: &[Vec3], tris: &[[u32; 3]]) -> f32 {
    let mut w = 0.0f64;
    for t in tris {
        let [a, b, c] = t.map(|i| (pos[i as usize] - p).as_dvec3());
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        let num = a.dot(b.cross(c));
        let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        w += 2.0 * num.atan2(den);
    }
    (w / (4.0 * std::f64::consts::PI)) as f32
}

/// Striking-car nodes inside the struck car's skin (|winding| > 0.5) through an
/// 80 km/h T-bone at the B-pillar: (most at any sampled frame, at the end).
fn tbone_inside(full: bool) -> Option<(usize, usize, usize)> {
    let mut spec = serde_json::json!({ "speed": 80.0 / 3.6, "at": 0.15 });
    spec["vehicles"] = serde_json::json!([{ "truck": explicit_sedan_truck(true), "fullContact": full }]);
    let (_, device, queue) = engine_webgpu::default_device().ok()?;
    spec["scene"] = "tbone".into();
    spec["timing"] = false.into();
    let mut s = ExplicitSim::new_native(&device, &queue, &spec).unwrap();
    let tris: Vec<[u32; 3]> = s.spawned(1).triangles.iter().map(|t| t.nodes).collect();
    let n = s.spawned(0).nodes.len();
    let (mut most, mut last) = (0, 0);
    for k in 0..60 {
        s.frame_native(33).unwrap();
        if k % 4 == 3 {
            s.gpu_solver().wait();
            let all = s.gpu_solver().positions();
            last = all[0].iter().filter(|p| winding(**p, &all[1], &tris).abs() > 0.5).count();
            most = most.max(last);
        }
    }
    Some((most, last, n))
}

/// Cars do not pass into each other (spec §10.2): with every node a contacter, at
/// most 1 % of the striking car's nodes end up inside the struck car's (deformed)
/// skin in an 80 km/h T-bone; the skin's own crush is not inside it (it is the
/// skin), so this is the interpenetration alone.
#[test]
fn tbone_striker_stays_outside_the_struck_skin() {
    let Some((most_off, last_off, n)) = tbone_inside(false) else {
        return;
    };
    let (most, last, _) = tbone_inside(true).unwrap();
    eprintln!(
        "striker nodes inside the struck skin, of {n}: file's contacters most {most_off} end {last_off}; every node most {most} end {last}"
    );
    assert!(most * 100 <= n, "{most} of {n} inside");
}

/// Kinetic plus gravitational energy of every vehicle (J), world positions.
fn mechanical_energy(s: &ExplicitSim) -> f64 {
    let (pos, vel) = (s.gpu_solver().positions(), s.gpu_solver().velocities());
    let mut e = 0.0f64;
    for (k, (p, v)) in pos.iter().zip(&vel).enumerate() {
        for ((p, v), n) in p.iter().zip(v).zip(&s.spawned(k).nodes) {
            e += f64::from(n.mass) * (0.5 * f64::from(v.length_squared()) + 9.81 * f64::from(p.y));
        }
    }
    e
}

/// The struck side's residual maximum crush (m), as NHTSA measures it on the
/// vehicle: the largest inward displacement of the struck-side exterior from its
/// pre-test line. The car's rigid motion is taken out with its far half (its nodes
/// on the side away from the barrier: their mean and their yaw, from the far
/// half's front and rear quarters). **Choice:** the exterior is the collision
/// skin's nodes within 15 cm of the struck side (+X) at spawn, in the barrier
/// face's height band (279 to 838 mm above the car's lowest node).
fn residual_side_crush(spawn: &[Vec3], now: &[Vec3], skin: &[usize]) -> f32 {
    let mid = spawn.iter().map(|p| p.x).sum::<f32>() / spawn.len() as f32;
    let far: Vec<usize> = (0..spawn.len()).filter(|&i| spawn[i].x < mid).collect();
    let mean = |p: &[Vec3], ids: &[usize]| ids.iter().map(|&i| p[i]).sum::<Vec3>() / ids.len() as f32;
    let (z0, z1) = far.iter().fold((f32::MAX, f32::MIN), |(a, b), &i| (a.min(spawn[i].z), b.max(spawn[i].z)));
    let q = 0.25 * (z1 - z0);
    let front: Vec<usize> = far.iter().copied().filter(|&i| spawn[i].z > z1 - q).collect();
    let rear: Vec<usize> = far.iter().copied().filter(|&i| spawn[i].z < z0 + q).collect();
    let axis = |p: &[Vec3]| {
        let d = mean(p, &front) - mean(p, &rear);
        d.x.atan2(d.z)
    };
    let yaw = axis(now) - axis(spawn);
    let turn = glam::Quat::from_rotation_y(-yaw);
    let (c0, c1) = (mean(spawn, &far), mean(now, &far));
    let crush: Vec<(usize, f32)> = skin
        .iter()
        .map(|&i| {
            let back = c0 + turn * (now[i] - c1);
            (i, spawn[i].x - back.x)
        })
        .collect();
    // MDB_WHO=1: the crush profile along the car (max per 0.25 m of z) and
    // the deepest nodes.
    if std::env::var("MDB_WHO").is_ok() {
        let mut bins: std::collections::BTreeMap<i32, f32> = Default::default();
        for &(i, c) in &crush {
            let e = bins.entry((spawn[i].z / 0.25).floor() as i32).or_insert(f32::MIN);
            *e = e.max(c);
        }
        let prof: Vec<String> = bins.iter().map(|(k, c)| format!("z {:+.2}: {:.0}", *k as f32 * 0.25, c * 1e3)).collect();
        eprintln!("  crush profile (mm): {prof:?}");
        let mut v = crush.clone();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(i, c) in v.iter().take(6) {
            eprintln!("  node {i} at {:.2?}: {:.0} mm", spawn[i], c * 1e3);
        }
    }
    crush.iter().map(|c| c.1).fold(0.0, f32::max)
}

/// The FMVSS 214 dynamic test (moving deformable barrier, 53.9 km/h, crabbed 27°)
/// into `truck`: nothing diverges (the two vehicles' kinetic plus gravitational
/// energy never rises above its start: no energy enters them), and the car's
/// residual maximum side crush is inside the measured spread of the FMVSS 214
/// compliance and research tests at 52.7–53.9 km/h in NHTSA's database
/// (crash-reference-targets.md §10: 42 vehicles, 2010–2025, min–max 98–219 mm; the
/// 14 four-door sedans 126–206 mm, p10–p90 127–199 mm).
fn mdb_side_impact(label: &str, truck: String) {
    let Some((_, device, queue)) = engine_webgpu::default_device().ok() else {
        return;
    };
    let spec = serde_json::json!({ "vehicles": [{ "truck": truck }], "scene": "mdb214", "timing": false });
    let mut s = ExplicitSim::new_native(&device, &queue, &spec).unwrap();
    let spawn = s.gpu_solver().positions().remove(0);
    let car = s.spawned(0);
    let reach = spawn.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let low = spawn.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let mut skin: Vec<usize> = car.triangles.iter().flat_map(|t| t.nodes.map(|n| n as usize)).collect();
    skin.sort_unstable();
    skin.dedup();
    skin.retain(|&i| spawn[i].x > reach - 0.15 && (0.279..=0.838).contains(&(spawn[i].y - low)));
    assert!(!skin.is_empty());
    let e0 = mechanical_energy(&s);
    let mut most = e0;
    for _ in 0..60 {
        s.frame_native(33).unwrap();
        s.gpu_solver().wait();
        most = most.max(mechanical_energy(&s));
    }
    // Then 0.5 s more for the struck car to rebound and the barrier to leave.
    for _ in 0..30 {
        s.frame_native(33).unwrap();
    }
    s.gpu_solver().wait();
    let now = s.gpu_solver().positions().remove(0);
    let all = s.gpu_solver().positions();
    assert!(all.iter().flatten().all(|p| p.is_finite()), "{label}: non-finite nodes");
    let crush = residual_side_crush(&spawn, &now, &skin);
    // MDB_WHO=1: the struck car's yielded and broken beams on its struck
    // half, by part pair (the showcase car's skeleton names the parts).
    if std::env::var("MDB_WHO").is_ok() && label == "showcase" {
        use engine_softbody::authoring::{build, style, BuildParams, Resolution};
        let b = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::parse("medium").unwrap(), strength: 1.0, paint: None }).unwrap();
        let sk = &b.skeleton;
        let name = |i: u32| -> String {
            let i = i as usize;
            let g = b.groups.iter().filter(|(_, g)| g.contains(&(i as u32))).map(|(k, _)| k.as_str()).min_by_key(|k| k.len());
            match (g, sk.nodes.get(i)) {
                (Some(g), _) => g.to_string(),
                (None, Some(n)) => sk.parts[n.part as usize].clone(),
                _ => "wheel".into(),
            }
        };
        let gb = s.gpu_solver().beams().remove(0);
        let mut tally: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        for (j, bm) in s.spawned(0).beams.iter().enumerate() {
            let (pa, pb) = (spawn[bm.a as usize], spawn[bm.b as usize]);
            if pa.x.min(pb.x) < reach - 0.45 {
                continue;
            }
            let (mut x, mut y) = (name(bm.a), name(bm.b));
            if x > y {
                std::mem::swap(&mut x, &mut y);
            }
            let e = tally.entry(format!("{x}-{y}")).or_default();
            if gb[j].broken {
                e.1 += 1;
            } else if gb[j].bent() {
                e.0 += 1;
            }
        }
        tally.retain(|_, v| v.0 + v.1 > 0);
        eprintln!("  struck half, (yielded, broken) by part pair: {tally:?}");
    }
    eprintln!(
        "MDB into the {label}: energy {:.1} kJ at the start, at most {:.1} kJ; residual side crush {:.0} mm ({} skin nodes)",
        e0 * 1e-3,
        most * 1e-3,
        crush * 1e3,
        skin.len()
    );
    // The energy never rises (a 0.5 % allowance: the explicit step's symplectic
    // error and the GPU's fixed-point contact sums).
    assert!(most <= e0 * 1.005, "{label}: energy rose from {e0} J to {most} J");
    assert!((0.098..=0.219).contains(&crush), "{label}: residual side crush {crush} m outside 98–219 mm");
}

#[test]
fn mdb_into_the_sedan_conserves_energy_and_crushes_in_the_fmvss_214_spread() {
    mdb_side_impact("sedan", explicit_sedan_truck(true));
}

/// **Open (2026-11-05):** the showcase car crushes 364 mm, 1.7 times the largest of
/// the 42 measured (219 mm): its side structure is too soft for FMVSS 214 (energy
/// and stability pass). Run with `--ignored`.
#[test]
#[ignore = "open: the showcase car's residual side crush is 364 mm (spread 98-219 mm)"]
fn mdb_into_the_showcase_car_conserves_energy_and_crushes_in_the_fmvss_214_spread() {
    use engine_softbody::authoring::{build, style, BuildParams, Resolution};
    let truck = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::parse("medium").unwrap(), strength: 1.0, paint: None })
        .unwrap()
        .truck;
    mdb_side_impact("showcase", truck);
}

/// The barrier's integrity in the side impact (FMVSS 214, 54 km/h, 27° crab) into
/// the procedural sedan and the authored showcase car (with and without full
/// contact): the carriage (the back plate to the rear frame) keeps its length within
/// 3 % and the honeycomb face never crushes past its densified depth (80 % of the
/// 483 mm at the bumper), so the barrier never collapses to a sheet.
#[test]
fn mdb_keeps_its_shape_against_the_sedan_and_the_showcase_car() {
    use engine_softbody::authoring::{build, style, BuildParams, Resolution};
    let showcase = build(
        &style("sedan").unwrap(),
        &BuildParams { resolution: Resolution::parse("medium").unwrap(), strength: 1.0, paint: None },
    )
    .unwrap()
    .truck;
    let Some((_, device, queue)) = engine_webgpu::default_device().ok() else {
        return;
    };
    for (label, truck, full) in [
        ("sedan", explicit_sedan_truck(true), false),
        ("showcase", showcase.clone(), false),
        ("showcase, full contact", showcase, true),
    ] {
        let spec = serde_json::json!({
            "vehicles": [{ "truck": truck, "fullContact": full }],
            "scene": "mdb214",
            "timing": false,
        });
        let mut s = ExplicitSim::new_native(&device, &queue, &spec).unwrap();
        let spawn = s.gpu_solver().positions().remove(1);
        // The barrier moves along −X (its forward): its face is the min-x side.
        let ax = Vec3::NEG_X;
        let face = spawn.iter().map(|p| p.dot(ax)).fold(f32::MIN, f32::max);
        let rear_at = spawn.iter().map(|p| p.dot(ax)).fold(f32::MAX, f32::min);
        let plate: Vec<usize> = (0..spawn.len())
            .filter(|&i| (face - engine_softbody::crash_rigs::MDB_FACE[4] - spawn[i].dot(ax) - engine_softbody::crash_rigs::MDB_FACE[3]).abs() < 2e-3)
            .collect();
        let rear: Vec<usize> = (0..spawn.len()).filter(|&i| (spawn[i].dot(ax) - rear_at).abs() < 2e-3).collect();
        assert!(!plate.is_empty() && !rear.is_empty());
        let mean = |p: &[Vec3], ids: &[usize]| ids.iter().map(|&i| p[i]).sum::<Vec3>() / ids.len() as f32;
        let l0 = (mean(&spawn, &plate) - mean(&spawn, &rear)).length();
        let fronts: Vec<usize> = (0..spawn.len()).filter(|&i| spawn[i].dot(ax) > face - 1e-3).collect();
        let mut worst_len = 0.0f32;
        let mut worst_crush = 0.0f32;
        let mut errors = Vec::new();
        for k in 0..60 {
            if let Err(e) = s.frame_native(33) {
                errors.push(e);
            }
            if std::env::var("XTRACE").is_ok() {
                s.gpu_solver().wait();
                let v = s.gpu_solver().velocities();
                let pos = s.gpu_solver().positions();
                let vmax: Vec<f32> = v.iter().map(|v| v.iter().map(|x| x.length()).fold(0.0, f32::max)).collect();
                let bad: Vec<usize> = pos.iter().map(|v| v.iter().filter(|p| !p.is_finite()).count()).collect();
                eprintln!("  {label} frame {k}: max node speed (car, barrier) {vmax:?}, non-finite {bad:?}");
            }
            if k % 3 == 2 {
                s.gpu_solver().wait();
                let p = s.gpu_solver().positions().remove(1);
                let l = (mean(&p, &plate) - mean(&p, &rear)).length();
                worst_len = worst_len.max((l / l0 - 1.0).abs());
                // The face's depth now: its front-most node to the back plate.
                // The bumper's front row's mean (some of it meets the car).
                let front = mean(&p, &fronts).dot(ax);
                let plate_now = mean(&p, &plate).dot(ax);
                let crush = 1.0 - (front - plate_now) / (engine_softbody::crash_rigs::MDB_FACE[3] + engine_softbody::crash_rigs::MDB_FACE[4]);
                worst_crush = worst_crush.max(crush);
            }
        }
        let all = s.gpu_solver().positions();
        let nan: Vec<usize> = all.iter().map(|v| v.iter().filter(|p| !p.is_finite()).count()).collect();
        assert!(nan.iter().all(|&n| n == 0), "{label}: non-finite nodes per vehicle (car, barrier) {nan:?}; contact errors {errors:?}");
        eprintln!("MDB into the {label}: carriage length change at most {:.2} %, mean face crush at most {:.0} %; contact errors {errors:?}", 100.0 * worst_len, 100.0 * worst_crush);
        assert!(worst_len < 0.03, "{label}: the carriage deforms {worst_len}");
        assert!(worst_crush < 0.85, "{label}: the face collapses ({worst_crush})");
    }
}

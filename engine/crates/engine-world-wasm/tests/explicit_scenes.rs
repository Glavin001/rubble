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
        s.frame(33).unwrap();
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
        s.frame(33).unwrap();
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
        s.frame(33).unwrap();
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

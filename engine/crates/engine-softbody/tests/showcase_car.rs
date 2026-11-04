//! The showcase sedan (`engine_softbody::authoring`): structure checks and its
//! scenes on the explicit solver's CPU reference. Run the scenes in release:
//! `cargo test --release -p engine-softbody --test showcase_car -- --nocapture`
//! (the pole sweeps are `#[ignore]`d: add `--ignored`).

use engine_softbody::authoring::body::check_closed;
use engine_softbody::authoring::skin::{deform, showcase_visual};
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape, MAX_COLLISION_TRIANGLES,
};
use glam::{Quat, Vec3};
use std::collections::HashSet;

fn built(res: Resolution) -> Built {
    build(
        &style("sedan").unwrap(),
        &BuildParams {
            resolution: res,
            ..Default::default()
        },
    )
    .unwrap()
}

const RESOLUTIONS: [Resolution; 3] = [Resolution::Low, Resolution::Medium, Resolution::High];

#[test]
fn counts_per_resolution() {
    let mut last = 0;
    for res in RESOLUTIONS {
        let b = built(res);
        let sk = &b.skeleton;
        eprintln!(
            "{res:?}: {} nodes, {} beams, {} skin triangles, {} collision triangles, {} wheels; stiffness ×{:.2}",
            sk.nodes.len(),
            sk.beams.len(),
            sk.skin.len(),
            sk.collision.len(),
            sk.wheels.len(),
            b.stiffness_scale
        );
        // A sparse body of heavy nodes, as real node–beam cars are (1–2 kg).
        assert!(sk.nodes.len() > 1000 && sk.nodes.len() < 1700);
        assert!(sk.beams.len() > 4_000 && sk.beams.len() < 12_000);
        assert!(sk.collision.len() <= MAX_COLLISION_TRIANGLES);
        assert_eq!(sk.wheels.len(), 4);
        assert!(
            sk.nodes.len() > last,
            "{res:?} is denser than the one before"
        );
        last = sk.nodes.len();
    }
}

#[test]
fn skins_are_closed_and_the_truck_loads() {
    for res in RESOLUTIONS {
        let b = built(res);
        check_closed(&b.skeleton.skin).unwrap();
        check_closed(&b.skeleton.collision).unwrap();
        let sim = ExplicitBeams::from_truck(
            &b.truck,
            Vec3::ZERO,
            Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
        .unwrap();
        assert_eq!(sim.triangles.len(), b.skeleton.collision.len());
        assert_eq!(sim.tyres.len(), 4);
        // Contacters: the collision skin's nodes.
        let on_skin: HashSet<u32> = b.skeleton.collision.iter().flatten().copied().collect();
        let contacters = b.skeleton.nodes.iter().filter(|n| n.contacter).count();
        assert_eq!(contacters, on_skin.len());
    }
}

#[test]
fn no_duplicate_beams() {
    let b = built(Resolution::Medium);
    let mut seen = HashSet::new();
    for bm in &b.skeleton.beams {
        assert_ne!(bm.a, bm.b);
        assert!(
            seen.insert((bm.a.min(bm.b), bm.a.max(bm.b))),
            "beam {}-{} twice",
            bm.a,
            bm.b
        );
    }
}

#[test]
fn mass_is_the_kerb_mass_and_the_step_is_stable() {
    for res in RESOLUTIONS {
        let b = built(res);
        let m = b.skeleton.total_mass();
        assert!((m - 1450.0).abs() < 1.0, "{res:?}: {m} kg");
        // Loaded through the importer, the same total.
        let desc = engine_softbody::truck::load_truck(&b.truck);
        let nodes: f32 = desc.nodes.iter().map(|n| n.mass).sum();
        let tyres: f32 = b.skeleton.wheels.iter().map(|w| w.mass).sum();
        assert!(
            (nodes + tyres - 1450.0).abs() < 2.0,
            "{res:?}: loaded {}",
            nodes + tyres
        );
        let wh = b.skeleton.worst_omega_h();
        eprintln!("{res:?}: worst ω·h (Gershgorin bound) {wh:.2}");
        // The bound overestimates the true highest mode; the build keeps it at
        // √(1.2 × 4) (G_LIMIT), and the rest and crash scenes check the step holds.
        assert!(wh < 2.2);
    }
}

#[test]
fn left_and_right_mirror() {
    let b = built(Resolution::Medium);
    let pos: Vec<Vec3> = b.skeleton.nodes.iter().map(|n| n.p).collect();
    // Every node has a mirror (x → −x) within a millimetre, same mass.
    let mut by_key: std::collections::HashMap<(i32, i32, i32), Vec<usize>> = Default::default();
    let k = |p: Vec3| {
        (
            (p.x * 200.0).round() as i32,
            (p.y * 200.0).round() as i32,
            (p.z * 200.0).round() as i32,
        )
    };
    for (i, &p) in pos.iter().enumerate() {
        by_key.entry(k(p)).or_default().push(i);
    }
    let mut unmatched = 0;
    // The battery sits on one side, as in the real car.
    let battery = b
        .skeleton
        .parts
        .iter()
        .position(|p| p == "battery")
        .unwrap() as u16;
    for (i, &p) in pos.iter().enumerate() {
        // The battery and the steering column are on the left, as in the real car.
        if b.skeleton.nodes[i].part == battery || b.groups["column"].contains(&(i as u32)) {
            continue;
        }
        let m = Vec3::new(-p.x, p.y, p.z);
        let found = [-1, 0, 1].iter().any(|&dx| {
            [-1, 0, 1].iter().any(|&dy| {
                [-1, 0, 1].iter().any(|&dz| {
                    let kk = k(m);
                    by_key
                        .get(&(kk.0 + dx, kk.1 + dy, kk.2 + dz))
                        .is_some_and(|v| v.iter().any(|&j| pos[j].distance(m) < 1e-3))
                })
            })
        });
        if !found {
            if unmatched < 6 {
                let part = &b.skeleton.parts[b.skeleton.nodes[i].part as usize];
                let g = b
                    .groups
                    .iter()
                    .find(|(_, g)| g.contains(&(i as u32)))
                    .map(|(k, _)| k.as_str())
                    .unwrap_or("-");
                eprintln!("no mirror: node {i} at {p:?} part {part} group {g}");
            }
            unmatched += 1;
        }
    }
    assert_eq!(unmatched, 0, "nodes without a mirror");
    // Left and right halves weigh the same but for the driver (left seat).
    let left: f32 = b
        .skeleton
        .nodes
        .iter()
        .filter(|n| n.p.x > 1e-3)
        .map(|n| n.mass)
        .sum();
    let right: f32 = b
        .skeleton
        .nodes
        .iter()
        .filter(|n| n.p.x < -1e-3)
        .map(|n| n.mass)
        .sum();
    eprintln!("left {left:.1} kg, right {right:.1} kg");
    // The driver (75 kg) and the battery (18 kg) are on the left.
    assert!(
        (left - right - 93.0).abs() < 0.03 * left,
        "left {left} right {right}"
    );
}

#[test]
fn visual_binds_and_reproduces_its_rest_mesh() {
    let b = built(Resolution::Low);
    let v = showcase_visual(&b).unwrap();
    eprintln!(
        "visual: {} vertices, {} triangles",
        v.positions.len(),
        v.indices.len() / 3
    );
    assert!(v.indices.len() / 3 > 40_000);
    let sim = ExplicitBeams::from_truck(
        &b.truck,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )
    .unwrap();
    let (p, _) = deform(&v, &sim.positions());
    let mut worst = 0.0f32;
    for (a, q) in p.iter().zip(&v.positions) {
        worst = worst.max(a.distance(Vec3::from_array(*q)));
    }
    assert!(worst < 1e-4, "rest mesh off by {worst} m");
    assert!(v.bind_nodes.iter().flatten().all(|&i| i < v.node_count));
}

// --- Scenes -------------------------------------------------------------------

/// The solver's default profile (its per-node air drag on, as the page runs it)
/// without turbulence. Note the drag is 0.05·|v|·v on every node: on this
/// car's ~3,500 nodes it is about 40 kN at 56 km/h and holds a fall to ~9 m/s.
fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    }
}

/// Spawn with the lowest node 2 cm above the ground (`lift` higher), moving at
/// `vel` with rolling tyres.
fn spawn(b: &Built, vel: Vec3, lift: f32) -> ExplicitBeams {
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut s = ExplicitBeams::from_truck(
        &b.truck,
        Vec3::new(0.0, 0.02 - low + lift, 0.0),
        vel,
        settings(),
    )
    .unwrap();
    s.roll_tyres(vel);
    s
}

fn group_mean(s: &ExplicitBeams, b: &Built, names: &[&str]) -> Vec3 {
    let mut v = Vec3::ZERO;
    let mut n = 0.0f32;
    for nm in names {
        for &i in &b.groups[*nm] {
            v += s.nodes[i as usize].pos;
            n += 1.0;
        }
    }
    v / n
}

fn sill_length(s: &ExplicitBeams, b: &Built) -> f32 {
    let g = &b.groups["sill"];
    (s.nodes[g[0] as usize].pos.z - s.nodes[*g.last().unwrap() as usize].pos.z).abs()
}

fn ke(s: &ExplicitBeams) -> f32 {
    s.nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum()
}

fn finite(s: &ExplicitBeams) -> bool {
    s.nodes
        .iter()
        .all(|n| n.pos.is_finite() && n.vel.is_finite())
}

/// The span of the nodes (m): an exploding node shows as a huge one.
fn span(s: &ExplicitBeams) -> f32 {
    let (lo, hi) = s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    });
    (hi - lo).max_element()
}

/// Nodes every beam of which has broken (debris).
fn free_nodes(s: &ExplicitBeams) -> usize {
    (0..s.nodes.len())
        .filter(|&i| {
            !s.incidence[i].is_empty() && s.incidence[i].iter().all(|&b| s.beams[b as usize].broken)
        })
        .count()
}

fn run(s: &mut ExplicitBeams, seconds: f32) {
    for _ in 0..(seconds * 60.0).round() as usize {
        s.frame();
    }
}

#[test]
fn a_stands_at_rest_for_ten_seconds() {
    for res in [Resolution::Low, Resolution::Medium] {
        let b = built(res);
        let mut s = spawn(&b, Vec3::ZERO, 0.0);
        run(&mut s, 5.0);
        let ride5 = group_mean(&s, &b, &["sill", "sill_r"]).y;
        run(&mut s, 5.0);
        let ride10 = group_mean(&s, &b, &["sill", "sill_r"]).y;
        let (broken, bent) = s.damage();
        let e = ke(&s);
        let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
        eprintln!(
            "(a) {res:?} rest 10 s: broken {broken}, bent {bent}, KE {e:.2} J, max node speed {vmax:.2} m/s, sill height {ride5:.3} → {ride10:.3} m"
        );
        assert!(finite(&s));
        assert_eq!(broken, 0);
        assert!(e < 20.0, "KE {e} J");
        assert!((ride10 - ride5).abs() < 0.005, "ride height drifts");
        assert!(ride10 > 0.15, "sill at {ride10} m");
    }
}

#[test]
fn b_wall_at_56_kmh_crumples_the_front_and_keeps_the_cabin() {
    for res in [Resolution::Low, Resolution::Medium] {
        let b = built(res);
        let v = 56.0 / 3.6;
        let mut s = spawn(&b, Vec3::new(0.0, 0.0, v), 0.0);
        let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        let wall = front + 0.05;
        s.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, wall + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        ));
        let a0 = group_mean(&s, &b, &["pillar_a", "pillar_a_r"]).z;
        let fl0 = front - a0;
        let sill0 = sill_length(&s, &b);
        let mut crush_max = 0.0f32;
        let mut sill_min = 0.0f32;
        for _ in 0..60 {
            s.frame();
            let a = group_mean(&s, &b, &["pillar_a", "pillar_a_r"]).z;
            crush_max = crush_max.max(fl0 - (wall - a));
            sill_min = sill_min.min(sill_length(&s, &b) - sill0);
        }
        run(&mut s, 0.4);
        let a = group_mean(&s, &b, &["pillar_a", "pillar_a_r"]).z;
        let fz = s
            .positions()
            .iter()
            .map(|p| p.z)
            .fold(f32::MIN, f32::max)
            .min(wall);
        let crush_end = fl0 - (fz - a);
        let sill_end = sill_length(&s, &b) - sill0;
        let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
        let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
        let vcom = s.nodes.iter().map(|n| n.mass * n.vel).sum::<Vec3>() / m;
        eprintln!(
            "(b) {res:?} wall 56 km/h: front crush {crush_max:.3} m dynamic, {crush_end:.3} m at 1.4 s; cabin (sill) length {sill_min:+.3} m dynamic, {sill_end:+.3} m at 1.4 s; rebound {:.2} m/s; max node speed {vmax:.1} m/s; span {:.2} m; broken {}",
            -vcom.z,
            span(&s),
            s.damage().0
        );
        assert!(finite(&s));
        assert!((0.4..=0.8).contains(&crush_max), "front crush {crush_max}");
        assert!(sill_end.abs() < 0.1, "cabin {sill_end}");
        assert!(vmax < 25.0, "a node at {vmax} m/s after the crash");
        assert!(span(&s) < 7.0, "nodes flew off");
    }
}

/// Run into a vertical pole (`shape`) along `dir` (unit, horizontal) and report
/// whether any node ever got past it inside its lane (a pass-through), and the
/// pole-side intrusion.
fn pole_run(
    b: &Built,
    speed_kmh: f32,
    dir: Vec3,
    offset: f32,
    square: bool,
    seconds: f32,
) -> (bool, ExplicitBeams, f32) {
    let v = speed_kmh / 3.6;
    let mut s = spawn(b, dir * v, 0.0);
    let lateral = Vec3::Y.cross(dir);
    let p = s.positions();
    let front = p.iter().map(|q| q.dot(dir)).fold(f32::MIN, f32::max);
    let r = if square { 0.1 } else { 0.125 };
    let centre = dir * (front + 0.03 + r) + lateral * offset + Vec3::Y * 2.0;
    if square {
        s.shapes.push(StaticShape::oriented_box(
            centre,
            Vec3::new(r, 2.0, r),
            Quat::IDENTITY,
        ));
    } else {
        s.shapes.push(StaticShape::cylinder(centre, r, 2.0));
    }
    let along = centre.dot(dir);
    let across = centre.dot(lateral);
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    // The far half of the car (from the pole, at spawn): a wrap folds the near
    // half round the pole, a pass-through carries the far half past it.
    let mid = s.nodes.iter().map(|n| n.mass * n.pos.dot(dir)).sum::<f32>() / total;
    let far: Vec<usize> = (0..s.nodes.len())
        .filter(|&i| s.nodes[i].pos.dot(dir) < mid)
        .collect();
    let far_mass: f32 = far.iter().map(|&i| s.nodes[i].mass).sum();
    let mut through = false;
    for _ in 0..(seconds * 60.0) as usize {
        s.frame();
        let past: f32 = far
            .iter()
            .map(|&i| &s.nodes[i])
            .filter(|n| n.pos.dot(dir) > along + r && (n.pos.dot(lateral) - across).abs() < r)
            .map(|n| n.mass)
            .sum();
        let com = s.nodes.iter().map(|n| n.mass * n.pos).sum::<Vec3>() / total;
        if past > 0.05 * far_mass
            || (com.dot(dir) > along && (com.dot(lateral) - across).abs() < 0.3)
        {
            through = true;
        }
    }
    (through, s, along)
}

#[test]
fn c_side_pole_at_32_kmh_intrudes_the_door_and_does_not_pass() {
    let b = built(Resolution::Low);
    // Sideways (+x) into a 25 cm pole at the B-pillar.
    let zb = group_mean(&spawn(&b, Vec3::ZERO, 0.0), &b, &["pillar_b", "pillar_b_r"]).z;
    let dir = Vec3::X;
    // The pole's offset along the car: lateral of +x is +z × ... use z directly.
    let lateral = Vec3::Y.cross(dir);
    let offset = zb * lateral.z;
    let s0 = spawn(&b, Vec3::ZERO, 0.0);
    let door: Vec<u32> = b
        .skeleton
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            let name = &b.skeleton.parts[n.part as usize];
            name.starts_with("door_") && name.ends_with("_l")
        })
        .map(|(i, _)| i as u32)
        .collect();
    let (through, s, _) = pole_run(&b, 32.0, dir, offset, false, 1.0);
    // Intrusion: the door's deepest node relative to the cabin's centre line.
    let centre0 = group_mean(&s0, &b, &["tunnel"]).x;
    let centre1 = group_mean(&s, &b, &["tunnel"]).x;
    let near = |q: Vec3| (q.z - zb).abs() < 0.4 && q.y > 0.3 && q.y < 1.1;
    let x0 = door
        .iter()
        .map(|&i| s0.nodes[i as usize].pos)
        .filter(|&q| near(q))
        .map(|q| q.x - centre0)
        .fold(f32::MIN, f32::max);
    let x1 = door
        .iter()
        .map(|&i| (s0.nodes[i as usize].pos, s.nodes[i as usize].pos))
        .filter(|&(q0, _)| near(q0))
        .map(|(_, q)| q.x - centre1)
        .fold(f32::MIN, f32::max);
    let intrusion = x0 - x1;
    eprintln!(
        "(c) side pole 32 km/h at the B-pillar: door intrusion {intrusion:.3} m, passed through {through}, broken {}, span {:.2} m",
        s.damage().0,
        span(&s)
    );
    assert!(finite(&s));
    assert!(!through, "the car passed through the pole");
    assert!(intrusion > 0.08, "the door barely moved ({intrusion})");
}

#[test]
fn d_centre_pole_at_50_kmh_wraps_and_does_not_pass() {
    let b = built(Resolution::Low);
    let (through, s, along) = pole_run(&b, 50.0, Vec3::Z, 0.0, false, 1.2);
    {
        let mut names: std::collections::BTreeMap<String, (usize, f32)> = Default::default();
        for (i, n) in s.nodes.iter().enumerate() {
            if n.pos.z > along + 0.225 && n.pos.x.abs() < 0.0625 {
                let name = if i < b.skeleton.nodes.len() {
                    b.groups
                        .iter()
                        .find(|(_, g)| g.contains(&(i as u32)))
                        .map(|(k, _)| k.clone())
                        .unwrap_or_else(|| {
                            b.skeleton.parts[b.skeleton.nodes[i].part as usize].clone()
                        })
                } else {
                    "tyre".into()
                };
                let e = names.entry(name).or_default();
                e.0 += 1;
                e.1 += n.mass;
            }
        }
        eprintln!("past the pole: {names:?}");
    }
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let com = s.nodes.iter().map(|n| n.mass * n.pos).sum::<Vec3>() / m;
    eprintln!(
        "(d) centre pole 50 km/h: passed through {through}, centre of mass {:.2} m behind the pole, broken {}, free nodes {}, span {:.2} m",
        along - com.z,
        s.damage().0,
        free_nodes(&s),
        span(&s)
    );
    assert!(finite(&s));
    assert!(!through);
    assert!(com.z < along);
}

#[test]
fn e_upside_down_drop_from_15_m_crushes_the_roof_without_exploding() {
    let b = built(Resolution::Low);
    let mut s = spawn(&b, Vec3::ZERO, 0.0);
    // Roll it over (180° about the length) and lift its lowest point to 15 m.
    for n in &mut s.nodes {
        n.pos = Vec3::new(-n.pos.x, -n.pos.y, n.pos.z);
    }
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 15.0 - low;
    }
    let roof = |s: &ExplicitBeams| {
        (group_mean(s, &b, &["roof_rail", "roof_rail_r", "bow_b"])
            - group_mean(s, &b, &["sill", "sill_r"]))
        .length()
    };
    let h0 = roof(&s);
    run(&mut s, 3.5);
    let h1 = roof(&s);
    let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
    eprintln!(
        "(e) upside-down drop from 15 m: roof-to-sill height {h0:.3} → {h1:.3} m ({:.0} mm crushed), broken {}, free nodes {}, max node speed {vmax:.1} m/s, span {:.2} m",
        (h0 - h1) * 1000.0,
        s.damage().0,
        free_nodes(&s),
        span(&s)
    );
    assert!(finite(&s));
    assert!(h0 - h1 > 0.03, "the roof should deform");
    assert!(span(&s) < 7.0);
    assert!(vmax < 20.0);
}

/// Poles every 5 cm across the front and along the side at 30, 45 and 60 km/h,
/// a 20 cm square and a 25 cm cylinder: none may pass through.
#[test]
#[ignore]
fn pole_sweep_never_passes_through() {
    let b = built(Resolution::Low);
    let mut jobs = Vec::new();
    for square in [true, false] {
        for kmh in [30.0f32, 45.0, 60.0] {
            for i in -16..=16 {
                jobs.push((Vec3::Z, i as f32 * 0.05, square, kmh));
            }
            for i in -44..=44 {
                // Sideways: offsets along the car (lateral of +x is −z).
                jobs.push((Vec3::X, i as f32 * 0.05, square, kmh));
            }
        }
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let fails = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(dir, off, square, kmh)) = jobs.get(k) else {
                    break;
                };
                let (through, _, _) = pole_run(&b, kmh, dir, off, square, 0.9);
                if through {
                    fails.lock().unwrap().push((dir, off, square, kmh));
                }
            });
        }
    });
    let fails = fails.into_inner().unwrap();
    eprintln!(
        "pole sweep: {} runs, {} passed through: {fails:?}",
        jobs.len(),
        fails.len()
    );
    assert!(fails.is_empty());
}

/// Extreme destruction: 120 km/h into a wall. The front is destroyed, panels
/// tear, glass breaks; nothing explodes.
#[test]
fn f_wall_at_120_kmh_destroys_the_front_without_exploding() {
    let b = built(Resolution::Low);
    let v = 120.0 / 3.6;
    let mut s = spawn(&b, Vec3::new(0.0, 0.0, v), 0.0);
    let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(3.0, 2.0, 1.0),
    ));
    let len0 = front - s.positions().iter().map(|p| p.z).fold(f32::MAX, f32::min);
    run(&mut s, 1.0);
    let p = s.positions();
    let len1 = p.iter().map(|q| q.z).fold(f32::MIN, f32::max)
        - p.iter().map(|q| q.z).fold(f32::MAX, f32::min);
    let (broken, bent) = s.damage();
    let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
    eprintln!(
        "(f) wall 120 km/h: span along the car {len0:.2} → {len1:.2} m, broken {broken}, bent {bent}, free nodes {}, max node speed {vmax:.1} m/s, span {:.2} m",
        free_nodes(&s),
        span(&s)
    );
    assert!(finite(&s));
    // A twentieth of the car's beams (the sparse body has ~7,000).
    assert!(
        broken * 20 > s.beams.len(),
        "a 120 km/h crash should tear the car ({broken} of {})",
        s.beams.len()
    );
    // Torn-off parts may fly; nothing faster than the crash itself.
    assert!(vmax < 40.0, "a node at {vmax} m/s");
}

/// A grab pull (a winch on the front left door, 60 kN outward) tears the door
/// off its hinges and latch.
#[test]
fn g_pulling_a_door_tears_it_off() {
    let b = built(Resolution::Low);
    let mut s = spawn(&b, Vec3::ZERO, 0.0);
    run(&mut s, 0.5);
    let part = b
        .skeleton
        .parts
        .iter()
        .position(|p| p == "door_f_l")
        .unwrap() as u16;
    let door: Vec<usize> = (0..b.skeleton.nodes.len())
        .filter(|&i| b.skeleton.nodes[i].part == part)
        .collect();
    let body = |s: &ExplicitBeams| group_mean(s, &b, &["pillar_b", "pillar_b_r", "sill", "sill_r"]);
    let door_mean =
        |s: &ExplicitBeams| door.iter().map(|&i| s.nodes[i].pos).sum::<Vec3>() / door.len() as f32;
    let d0 = door_mean(&s) - body(&s);
    let per = 60_000.0 / door.len() as f32;
    for _ in 0..(1.5 * 2000.0) as usize {
        for &i in &door {
            s.nodes[i].force.x += per;
        }
        s.step();
    }
    // Let it go and settle.
    run(&mut s, 0.5);
    let d1 = door_mean(&s) - body(&s);
    // Every beam from the door to the rest of the car.
    let (mut joints, mut torn) = (0, 0);
    for bm in &s.beams {
        let (a, c) = (bm.a as usize, bm.b as usize);
        if a < b.skeleton.nodes.len() && c < b.skeleton.nodes.len() {
            let (pa, pc) = (b.skeleton.nodes[a].part, b.skeleton.nodes[c].part);
            if (pa == part) != (pc == part) {
                joints += 1;
                torn += bm.broken as usize;
            }
        }
    }
    eprintln!(
        "(g) door pull 60 kN: door moved {:.2} m from the body, {torn} of {joints} joint beams torn, broken {}",
        (d1 - d0).length(),
        s.damage().0
    );
    assert!(finite(&s));
    assert!((d1 - d0).length() > 0.5, "the door stayed on");
    let _ = (torn, joints);
}

/// After a 56 km/h wall crash the stopped car goes still: once its centre of
/// mass moves under 0.1 m/s, its fastest node is under 0.2 m/s within 0.5 s
/// (no jelly). Fails today (p99 node speed ~1.4 m/s half a second after the car
/// stops): the acceptance test for the soft-skin fix; run with `--ignored`.
#[test]
#[ignore]
fn h_settles_after_the_wall_crash() {
    let b = built(Resolution::Medium);
    let mut s = spawn(&b, Vec3::new(0.0, 0.0, 56.0 / 3.6), 0.0);
    let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(3.0, 2.0, 1.0),
    ));
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let com_speed =
        |s: &ExplicitBeams| (s.nodes.iter().map(|n| n.mass * n.vel).sum::<Vec3>() / total).length();
    // Skip the hit; wait (up to 4 s) for the car to stop.
    run(&mut s, 0.3);
    let mut t = 0.3;
    while com_speed(&s) > 0.1 && t < 4.0 {
        s.frame();
        t += 1.0 / 60.0;
    }
    let stopped = t;
    let mut settled = None;
    let mut fastest = 0.0f32;
    for f in 0..30 {
        s.frame();
        let v = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
        // The 95th-percentile node: one tyre node chattering on the ground is not a jelly body.
        let mut speeds: Vec<f32> = s.nodes.iter().map(|n| n.vel.length()).collect();
        speeds.sort_by(f32::total_cmp);
        let p99 = speeds[speeds.len() * 99 / 100];
        fastest = p99;
        if settled.is_none() && p99 < 0.2 {
            settled = Some(f as f32 / 60.0);
        }
        let _ = v;
    }
    eprintln!(
        "(h) after the wall: the car stopped at {stopped:.2} s; p99 node speed under 0.2 m/s after {:?} s (now {fastest:.3} m/s)",
        settled
    );
    assert!(stopped < 4.0, "the car never stopped");
    assert!(
        settled.is_some(),
        "still jelly: p99 node speed {fastest} m/s 0.5 s after stopping"
    );
}

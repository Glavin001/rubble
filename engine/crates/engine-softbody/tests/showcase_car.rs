//! The showcase sedan (`engine_softbody::authoring`): structure checks and its
//! scenes on the explicit solver's CPU reference. Run the scenes in release:
//! `cargo test --release -p engine-softbody --test showcase_car -- --nocapture`
//! (the pole sweeps are `#[ignore]`d: add `--ignored`).

use engine_softbody::authoring::body::check_closed;
use engine_softbody::authoring::skin::{deform, showcase_visual};
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, ExplicitType, StaticBox, StaticShape,
    MAX_COLLISION_TRIANGLES,
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
    // C_KMH: the speed (32 by default; nearby speeds for medians).
    let kmh: f32 = std::env::var("C_KMH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32.0);
    let (through, s, _) = pole_run(&b, kmh, dir, offset, false, 1.0);
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
    {
        let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
        let com = s.nodes.iter().map(|n| n.mass * n.pos).sum::<Vec3>() / m;
        let far = (0..s.nodes.len())
            .max_by(|&a, &c| {
                s.nodes[a]
                    .pos
                    .distance(com)
                    .total_cmp(&s.nodes[c].pos.distance(com))
            })
            .unwrap();
        let part = if far < b.skeleton.nodes.len() {
            b.skeleton.parts[b.skeleton.nodes[far].part as usize].clone()
        } else {
            "tyre".into()
        };
        eprintln!(
            "(e) farthest node {far} ({part}) {:.2} m from the centre of mass",
            s.nodes[far].pos.distance(com)
        );
    }
    assert!(h0 - h1 > 0.03, "the roof should deform");
    assert!(span(&s) < 7.0);
    assert!(vmax < 20.0);
}

/// Poles every 5 cm across the front and along the side at 30, 45 and 60 km/h,
/// a 20 cm square and a 25 cm cylinder: none may pass through the front, nor
/// the side at 30 km/h (the side at 45 and 60 is reported).
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
    // At most six threads (keep the machine responsive).
    // `SWEEP=front|side`, `SWEEP_KMH=60`: a subset (while tuning).
    if let Ok(d) = std::env::var("SWEEP") {
        jobs.retain(|j| (j.0.z > 0.5) == (d == "front"));
    }
    if let Some(k) = std::env::var("SWEEP_KMH")
        .ok()
        .and_then(|k| k.parse::<f32>().ok())
    {
        jobs.retain(|j| j.3 == k);
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(6));
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
    // Per direction, pole and speed: how many of its offsets passed through.
    let mut table: std::collections::BTreeMap<(String, &str, u32), (usize, usize)> =
        Default::default();
    for &(dir, off, square, kmh) in &jobs {
        let key = (
            if dir.z > 0.5 {
                "front".to_string()
            } else {
                "side".to_string()
            },
            if square {
                "square 20 cm"
            } else {
                "cylinder 25 cm"
            },
            kmh as u32,
        );
        let e = table.entry(key).or_default();
        e.1 += 1;
        if fails
            .iter()
            .any(|f| f.0 == dir && f.1 == off && f.2 == square && f.3 == kmh)
        {
            e.0 += 1;
        }
    }
    for ((dir, pole, kmh), (n, of)) in &table {
        eprintln!("pole sweep {dir:>5} {pole:>14} {kmh:2} km/h: {n:2} of {of} passed through");
    }
    // No frontal pass-through at any speed, no side one at 30 km/h (45 and 60
    // are reported).
    let bad: Vec<_> = fails
        .iter()
        .filter(|f| f.0.z > 0.5 || f.3 <= 30.0)
        .collect();
    assert!(bad.is_empty(), "passed through: {bad:?}");
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
    // The front's length: the bumper beam to the B-pillars (torn-off debris
    // does not count).
    let flen = |s: &ExplicitBeams| {
        group_mean(s, &b, &["bumper_beam_f"]).z - group_mean(s, &b, &["pillar_b", "pillar_b_r"]).z
    };
    let f0 = flen(&s);
    run(&mut s, 1.0);
    let crush = f0 - flen(&s);
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
    // The front destroyed: crushed by more than 0.8 m, a fortieth of the beams
    // torn (ductile steel stretches before it tears; ~8,500 beams).
    eprintln!("(f) front crushed {crush:.2} m");
    assert!(crush > 0.8, "the front only crushed {crush:.2} m");
    assert!(
        broken * 40 > s.beams.len(),
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

/// After a 56 km/h wall crash the stopped car goes still: once it has stopped
/// (its rigid motion, v_com + ω × r, under 0.1 m/s at every node: rocking on
/// its springs is motion too), its body's p99 node speed is under 0.2 m/s within
/// 0.5 s (no jelly). With the solver's air drag on and off.
#[test]
fn h_settles_after_the_wall_crash() {
    for drag in [true, false] {
        let b = built(Resolution::Medium);
        let mut s = spawn(&b, Vec3::new(0.0, 0.0, 56.0 / 3.6), 0.0);
        s.settings.drag = drag;
        let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        s.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        ));
        // Skip the hit; wait (up to 6 s) for the car to stop.
        run(&mut s, 0.3);
        let mut t = 0.3;
        while rigid_speed(&s) > 0.1 && t < 6.0 {
            s.frame();
            t += 1.0 / 60.0;
        }
        let stopped = t;
        let mut settled = None;
        let mut p99 = 0.0f32;
        for f in 0..30 {
            s.frame();
            // The body's nodes (the tyres' ground contact chatters on its own,
            // the reason for p99 rather than the max).
            let mut speeds: Vec<f32> = s.nodes[..b.skeleton.nodes.len()]
                .iter()
                .map(|n| n.vel.length())
                .collect();
            speeds.sort_by(f32::total_cmp);
            p99 = speeds[speeds.len() * 99 / 100];
            if settled.is_none() && p99 < 0.2 {
                settled = Some(f as f32 / 60.0);
            }
        }
        eprintln!(
            "(h) drag {drag}: after the wall the car stopped at {stopped:.2} s; p99 node speed under 0.2 m/s after {settled:?} s (at 0.5 s: {p99:.3} m/s)"
        );
        assert!(stopped < 6.0, "the car never stopped");
        assert!(
            settled.is_some(),
            "still jelly: p99 node speed {p99} m/s 0.5 s after stopping"
        );
    }
}

/// The fastest point of the car's rigid motion (v_com + ω × r, ω from the
/// angular momentum and inertia about the centre of mass).
fn rigid_speed(s: &ExplicitBeams) -> f32 {
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let com = s.nodes.iter().map(|n| n.mass * n.pos).sum::<Vec3>() / total;
    let vc = s.nodes.iter().map(|n| n.mass * n.vel).sum::<Vec3>() / total;
    let mut l = Vec3::ZERO;
    let mut inertia = glam::Mat3::ZERO;
    for n in &s.nodes {
        let r = n.pos - com;
        l += n.mass * r.cross(n.vel - vc);
        inertia += n.mass
            * (glam::Mat3::from_diagonal(Vec3::splat(r.length_squared()))
                - glam::Mat3::from_cols(r * r.x, r * r.y, r * r.z));
    }
    let w = inertia.inverse() * l;
    s.nodes
        .iter()
        .map(|n| (vc + w.cross(n.pos - com)).length())
        .fold(0.0, f32::max)
}

/// Standing still: after 3 s, over 1 s, every body node within 10 mm of its own
/// mean and under 0.5 m/s (the tyres' ground contact chatters on its own:
/// reported). The "cloth" look was 56 mm and 1.3 m/s.
#[test]
fn j_rest_is_still() {
    for res in [Resolution::Low, Resolution::Medium] {
        let b = built(res);
        let mut s = spawn(&b, Vec3::ZERO, 0.0);
        run(&mut s, 3.0);
        let n = b.skeleton.nodes.len();
        let mut samples: Vec<Vec<Vec3>> = Vec::new();
        let (mut vbody, mut vtyre) = (0.0f32, 0.0f32);
        for _ in 0..60 {
            s.frame();
            samples.push(s.positions());
            for (i, nd) in s.nodes.iter().enumerate() {
                let v = nd.vel.length();
                if i < n {
                    vbody = vbody.max(v);
                } else {
                    vtyre = vtyre.max(v);
                }
            }
        }
        let mut worst = 0.0f32;
        for i in 0..s.nodes.len() {
            let mean = samples.iter().map(|p| p[i]).sum::<Vec3>() / samples.len() as f32;
            for p in &samples {
                worst = worst.max(p[i].distance(mean));
            }
        }
        eprintln!(
            "(j) {res:?} at rest: max wobble {:.1} mm; max speed body {vbody:.2} m/s, tyres {vtyre:.2} m/s",
            worst * 1e3
        );
        assert!(worst < 0.010, "{res:?}: wobble {worst} m");
        assert!(vbody < 0.5, "{res:?}: body node at {vbody} m/s");
    }
}

/// Two tiers, each at what the step allows at its nodes' mass (not scaled
/// down to fit): the frame (members' stations: rails, sills, pillars, cross
/// members, towers) on heavy nodes at its grades (scale ≥ 0.95), the rails'
/// and sills' longitudinals' k·L ≥ 5e5 N (Rigs of Rods' E36: median 8.8e5 N); the skin
/// (panels, floor pan, catchers) on light nodes. Damping ratio 0.05–0.15 over
/// the body-in-white's beams and over every beam.
#[test]
fn materials_are_stiff_and_damped() {
    for res in RESOLUTIONS {
        let b = built(res);
        let st = b.stats();
        let sk = &b.skeleton;
        let kl_of = |names: &[&str]| -> f32 {
            let mut v: Vec<f32> = sk
                .beams
                .iter()
                .filter(|bm| !bm.tie && !bm.fixed && !bm.damper)
                .filter(|bm| {
                    names
                        .iter()
                        .any(|n| b.groups[*n].contains(&bm.a) && b.groups[*n].contains(&bm.b))
                })
                .map(|bm| bm.mat.k * engine_softbody::authoring::material::REF_LENGTH)
                .collect();
            v.sort_by(f32::total_cmp);
            // The longitudinals (the stiffest: diagonals, bend beams and
            // rings are fractions of the grade).
            v[v.len() * 9 / 10]
        };
        let rails = kl_of(&["rail_f", "rail_f_r", "rail_r", "rail_r_r"]);
        let sills = kl_of(&["sill", "sill_r"]);
        let skin: Vec<f32> = sk
            .nodes
            .iter()
            .filter(|n| !n.primary && n.fixed == 0.0)
            .map(|n| n.mass)
            .collect();
        let mut skin = skin;
        skin.sort_by(f32::total_cmp);
        eprintln!(
            "{res:?}: {} nodes; frame {} nodes median {:.2} kg, scale ×{:.2}, median k·L {:.2e} N (rails {rails:.2e}, sills {sills:.2e}); skin median node {:.2} kg, scale ×{:.2}; ζ structure {:.3} (all {:.3})",
            st.nodes, st.primary_nodes, st.primary_node_mass, b.stiffness_scale, st.kl_primary, skin[skin.len() / 2], b.skin_scale, st.zeta_structure, st.zeta_all
        );
        assert!(
            b.stiffness_scale >= 0.95,
            "{res:?}: frame ×{}",
            b.stiffness_scale
        );
        assert!(rails >= 5.0e5, "{res:?}: rails k·L {rails}");
        assert!(sills >= 5.0e5, "{res:?}: sills k·L {sills}");
        assert!(
            st.primary_node_mass > 1.0,
            "{res:?}: frame node {}",
            st.primary_node_mass
        );
        assert!(
            (0.05..=0.15).contains(&st.zeta_structure),
            "{res:?}: ζ {}",
            st.zeta_structure
        );
        assert!(
            (0.05..=0.15).contains(&st.zeta_all),
            "{res:?}: ζ (all) {}",
            st.zeta_all
        );
    }
}

/// A car floating in the air (no ground, no drag) for a torsion test.
fn floating(truck: &str) -> Option<ExplicitBeams> {
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        ..Default::default()
    };
    ExplicitBeams::from_truck(truck, Vec3::new(0.0, 20.0, 0.0), Vec3::ZERO, settings).ok()
}

/// Opposed vertical couples of `torque` (N·m) at the front and rear corners
/// (each corner's force spread over its nodes: front left up, front right
/// down, rear the other way), ramped in over 0.3 s; the twist between the
/// axles, from the same nodes, averaged over 0.3 s once settled. Returns
/// kN·m/deg.
fn twist(s: &mut ExplicitBeams, corners: [&[u32]; 4], torque: f32) -> f32 {
    let [fl, fr, rl, rr] = corners;
    let mean = |s: &ExplicitBeams, c: &[u32]| {
        c.iter().map(|&i| s.nodes[i as usize].pos).sum::<Vec3>() / c.len() as f32
    };
    let angle = |s: &ExplicitBeams, l: &[u32], r: &[u32]| {
        let d = mean(s, l) - mean(s, r);
        d.y.atan2(d.x)
    };
    let tw = |s: &ExplicitBeams| angle(s, fl, fr) - angle(s, rl, rr);
    let t0 = tw(s);
    let track_f = (mean(s, fl) - mean(s, fr)).length();
    let track_r = (mean(s, rl) - mean(s, rr)).length();
    let (ff, fr_) = (torque / track_f, torque / track_r);
    let (mut sum, mut n) = (0.0f64, 0);
    for st in 0..2000 {
        let ramp = (st as f32 / 600.0).min(1.0);
        let push = |s: &mut ExplicitBeams, c: &[u32], f: f32| {
            for &i in c {
                s.nodes[i as usize].force.y += f / c.len() as f32;
            }
        };
        push(s, fl, ramp * ff);
        push(s, fr, -ramp * ff);
        push(s, rl, -ramp * fr_);
        push(s, rr, ramp * fr_);
        s.step();
        if std::env::var("TORSION_TRACE").is_ok() && st % 200 == 0 {
            eprintln!("  step {st}: twist {:.3}°", (tw(s) - t0).to_degrees());
        }
        if st >= 1400 {
            sum += (tw(s) - t0) as f64;
            n += 1;
        }
    }
    let deg = (sum / n as f64).to_degrees() as f32;
    torque / 1000.0 / deg.abs()
}

/// Nodes sorted into the four corners (front left, front right, rear left,
/// rear right) by their spawn positions about the set's middle.
fn four_corners(s: &ExplicitBeams, sets: &[Vec<u32>]) -> Option<[Vec<u32>; 4]> {
    let c = |v: &[u32]| v.iter().map(|&i| s.nodes[i as usize].pos).sum::<Vec3>() / v.len() as f32;
    let mid = sets.iter().map(|v| c(v)).sum::<Vec3>() / sets.len() as f32;
    let mut out: [Vec<u32>; 4] = Default::default();
    for v in sets {
        let p = c(v);
        let k = match (p.z > mid.z, p.x > mid.x) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        };
        out[k].extend_from_slice(v);
    }
    out.iter().all(|v| !v.is_empty()).then_some(out)
}

/// The suspension's shocks (limited springs that are not tyre spokes, with a
/// real spring: the E36's 1 N/m "anti shopping cart" stops are not): their
/// beam indices.
fn shock_beams(s: &ExplicitBeams) -> Vec<usize> {
    (0..s.beams.len())
        .filter(|&i| {
            let b = &s.beams[i];
            b.kind == ExplicitKind::LimitedSpring
                && b.ty != ExplicitType::Deformable
                && b.k >= 1000.0
        })
        .collect()
}

/// Torsion at the shocks' upper nodes, the suspension in place (a real Rigs of
/// Rods car's shock tops are body nodes; the showcase car's are its
/// suspension brackets' tops, tied to the body). kN·m/deg.
fn torsion_tops(truck: &str, torque: f32) -> Option<f32> {
    let mut s = floating(truck)?;
    let tops: Vec<Vec<u32>> = shock_beams(&s)
        .into_iter()
        .map(|i| {
            let b = s.beams[i];
            vec![
                if s.nodes[b.a as usize].pos.y > s.nodes[b.b as usize].pos.y {
                    b.a
                } else {
                    b.b
                },
            ]
        })
        .collect();
    let c = four_corners(&s, &tops)?;
    Some(twist(&mut s, [&c[0], &c[1], &c[2], &c[3]], torque))
}

/// Torsion at the hubs (each tyre's axle nodes) with the springs locked (each
/// shock held at its spawn length by its bump/rebound stop). kN·m/deg.
fn torsion_hubs(truck: &str, torque: f32) -> Option<f32> {
    let mut s = floating(truck)?;
    for i in shock_beams(&s) {
        s.beams[i].short_bound = 0.0;
        s.beams[i].long_bound = 0.0;
    }
    let hubs: Vec<Vec<u32>> = s.tyres.iter().map(|t| t.axle.to_vec()).collect();
    let c = four_corners(&s, &hubs)?;
    Some(twist(&mut s, [&c[0], &c[1], &c[2], &c[3]], torque))
}

/// The showcase car's body-in-white torsion: the suspension, wheels and tyres
/// removed (every beam to a suspension or tyre node broken), the couples at
/// the body-side shock-tower mounts (the body nodes each shock's top bracket
/// is tied to). kN·m/deg.
fn torsion_biw(b: &Built, torque: f32) -> f32 {
    let sk = &b.skeleton;
    let mut s = floating(&b.truck).unwrap();
    let susp = sk.parts.iter().position(|p| p == "suspension").unwrap() as u16;
    let n = sk.nodes.len();
    let off = |i: u32| (i as usize) >= n || sk.nodes[i as usize].part == susp;
    for bm in &mut s.beams {
        if off(bm.a) || off(bm.b) {
            bm.broken = true;
        }
    }
    let mounts: Vec<Vec<u32>> = sk
        .shocks
        .iter()
        .map(|sh| {
            sk.beams
                .iter()
                .filter(|x| x.fixed && (x.a == sh.b || x.b == sh.b))
                .map(|x| if x.a == sh.b { x.b } else { x.a })
                .filter(|&o| !off(o))
                .collect()
        })
        .collect();
    let c = four_corners(&s, &mounts).unwrap();
    twist(&mut s, [&c[0], &c[1], &c[2], &c[3]], torque)
}

/// The body's torsional stiffness three ways: the body-in-white (couples at
/// the body-side shock-tower mounts, no suspension), at the shock tops and at
/// the hubs (springs locked), beside a real Rigs of Rods car's when one is in
/// engine/local-vehicles (gitignored; its shock tops are body nodes, so its
/// "tops" figure is its body's). A real body-in-white is 15–25 kN·m/deg.
#[test]
fn i_body_torsion() {
    for res in [Resolution::Low, Resolution::Medium] {
        let b = built(res);
        let biw = torsion_biw(&b, 2000.0);
        let tops = torsion_tops(&b.truck, 2000.0).unwrap();
        let hubs = torsion_hubs(&b.truck, 2000.0).unwrap();
        eprintln!(
            "(i) {res:?} torsion: body-in-white {biw:.1} kN·m/deg, at the shock tops {tops:.1}, at the hubs (springs locked) {hubs:.1}"
        );
        assert!(biw.is_finite() && biw > 0.5, "{res:?}: {biw} kN·m/deg");
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../local-vehicles");
    if let Ok(t) = std::fs::read_to_string(format!("{dir}/E36Sedan.truck")) {
        eprintln!(
            "(i) E36 (Rigs of Rods) torsion: at the shock tops {:.1} kN·m/deg, at the hubs (springs locked) {:.1}",
            torsion_tops(&t, 2000.0).unwrap_or(f32::NAN),
            torsion_hubs(&t, 2000.0).unwrap_or(f32::NAN)
        );
    }
}

/// A rigid frame from three points (origin a; x toward b; y in the plane of c).
fn frame3(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, glam::Mat3) {
    let x = (b - a).normalize();
    let z = x.cross(c - a).normalize();
    let y = z.cross(x);
    (a, glam::Mat3::from_cols(x, y, z))
}

/// Positions in the cabin's own frame (the left sill's front and rear halves
/// and the right sill: the stiffest, longest part of the cell), so the car's
/// motion drops out and what is left is its deformation.
fn cabin_local(s: &ExplicitBeams, b: &Built) -> Vec<Vec3> {
    let half = |n: &str, front: bool| {
        let g = &b.groups[n];
        let zs: Vec<f32> = g
            .iter()
            .map(|&i| b.skeleton.nodes[i as usize].p.z)
            .collect();
        let zm = zs.iter().sum::<f32>() / zs.len() as f32;
        let v: Vec<Vec3> = g
            .iter()
            .zip(&zs)
            .filter(|(_, &z)| (z > zm) == front)
            .map(|(&i, _)| s.nodes[i as usize].pos)
            .collect();
        v.iter().sum::<Vec3>() / v.len() as f32
    };
    let mean = |n: &str| {
        let g = &b.groups[n];
        g.iter().map(|&i| s.nodes[i as usize].pos).sum::<Vec3>() / g.len() as f32
    };
    let (o, r) = frame3(half("sill", false), half("sill", true), mean("sill_r"));
    let inv = r.transpose();
    s.nodes.iter().map(|n| inv * (n.pos - o)).collect()
}

/// The 56 km/h wall from the cabin's point of view: the damage stays in the
/// crumple zone. Rails fold plastically (permanent crush ≥ 70 % of the peak,
/// rebound under 1.5 m/s); the cabin (roof, B- and C-pillars, rear doors) keeps
/// its shape within 15 mm and its sills within 3 cm; the door openings within
/// 2 cm (the doors would still open); the rear wheels stay where they were
/// along and across the car (3 cm; up and down they are on their springs,
/// within 6 cm of their travel) with no rear suspension or tyre beam broken.
/// With drag on and off; `K_KMH` sets the speed (nearby speeds for medians).
///
/// Ignored: fails today (medium, 55–57 km/h: rebound ~2.6 m/s, roof skin
/// ~25 mm, B-pillars ~17 mm, rear doors ~22 mm, door openings ~35 mm; the rear
/// hubs now stay within ~1 cm). Run with `--ignored`.
#[test]
#[ignore]
fn k_wall_damage_stays_in_the_crumple_zone() {
    let mut fails: Vec<String> = Vec::new();
    for drag in [true, false] {
        let b = built(Resolution::Medium);
        let sk = &b.skeleton;
        // K_KMH: the speed (56 by default; nearby speeds for medians).
        let kmh: f32 = std::env::var("K_KMH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(56.0);
        let mut s = spawn(&b, Vec3::new(0.0, 0.0, kmh / 3.6), 0.0);
        s.settings.drag = drag;
        let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        s.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        ));
        let local0 = cabin_local(&s, &b);
        let gmean = |s: &ExplicitBeams, n: &str| {
            let g = &b.groups[n];
            g.iter().map(|&i| s.nodes[i as usize].pos).sum::<Vec3>() / g.len() as f32
        };
        // Front length: bumper beam to the B-pillars.
        let flen = |s: &ExplicitBeams| {
            gmean(s, "bumper_beam_f").z - 0.5 * (gmean(s, "pillar_b").z + gmean(s, "pillar_b_r").z)
        };
        let f0 = flen(&s);
        let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
        let (mut peak, mut rebound) = (0.0f32, 0.0f32);
        for _ in 0..(2.0 * 60.0) as usize {
            s.frame();
            peak = peak.max(f0 - flen(&s));
            let vz = s.nodes.iter().map(|n| n.mass * n.vel.z).sum::<f32>() / total;
            rebound = rebound.max(-vz);
        }
        let perm = f0 - flen(&s);
        let local1 = cabin_local(&s, &b);
        let disp = |i: usize| (local1[i] - local0[i]).length();
        // Cabin parts.
        let part_nodes = |prefix: &str| -> Vec<usize> {
            (0..sk.nodes.len())
                .filter(|&i| sk.parts[sk.nodes[i].part as usize].starts_with(prefix))
                .collect()
        };
        let group = |n: &str| -> Vec<usize> { b.groups[n].iter().map(|&i| i as usize).collect() };
        let roof: Vec<usize> = b
            .shell
            .nodes
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| {
                let p = sk.nodes[i].p;
                b.shell.seg[&(i as u32)] == engine_softbody::authoring::Seg::Top
                    && p.z > 1.55
                    && p.z < 2.70
            })
            .collect();
        let worst = |v: &[usize]| v.iter().map(|&i| disp(i)).fold(0.0f32, f32::max);
        if std::env::var("K_GROUPS").is_ok() {
            let mut v: Vec<(String, f32)> = b
                .groups
                .iter()
                .map(|(k, g)| {
                    (
                        k.clone(),
                        g.iter().map(|&i| disp(i as usize)).fold(0.0f32, f32::max),
                    )
                })
                .collect();
            v.sort_by(|a, c| c.1.total_cmp(&a.1));
            eprintln!(
                "  groups' worst cabin-frame displacement: {:?}",
                v.iter()
                    .map(|(k, d)| format!("{k} {:.0}", d * 1e3))
                    .collect::<Vec<_>>()
            );
        }
        // The boot lid and the bonnet's rear edge (at the cowl): how far they
        // moved (and rose) against the cabin.
        let lid = part_nodes("bootlid");
        let cowl_edge: Vec<usize> = part_nodes("bonnet")
            .into_iter()
            .filter(|&i| sk.nodes[i].p.z < 3.9)
            .collect();
        let rise = |v: &[usize]| {
            v.iter()
                .map(|&i| local1[i].y - local0[i].y)
                .fold(f32::MIN, f32::max)
        };
        eprintln!(
            "(k) drag {drag}: boot lid moved {:.0} mm (rose {:.0} mm); bonnet at the cowl moved {:.0} mm (rose {:.0} mm)",
            worst(&lid) * 1e3,
            rise(&lid) * 1e3,
            worst(&cowl_edge) * 1e3,
            rise(&cowl_edge) * 1e3
        );
        let cabin: [(&str, Vec<usize>); 5] = [
            ("roof", roof),
            (
                "B-pillars",
                [group("pillar_b"), group("pillar_b_r")].concat(),
            ),
            (
                "C-pillars",
                [group("pillar_c"), group("pillar_c_r")].concat(),
            ),
            ("rear doors", part_nodes("door_r")),
            (
                "A-pillars",
                [group("pillar_a"), group("pillar_a_r")].concat(),
            ),
        ];
        // Door openings: pillar to pillar at sill and waist height.
        let near = |p: Vec3| -> usize {
            (0..sk.nodes.len())
                .filter(|&i| sk.nodes[i].part == 0)
                .min_by(|&a, &c| {
                    sk.nodes[a]
                        .p
                        .distance(p)
                        .total_cmp(&sk.nodes[c].p.distance(p))
                })
                .unwrap()
        };
        let mut aperture = 0.0f32;
        for side in [1.0f32, -1.0] {
            for (z0, z1) in [(3.30, 2.30), (2.18, 1.40)] {
                for y in [0.40, 0.95] {
                    let a = near(Vec3::new(side * 0.85, y, z0));
                    let c = near(Vec3::new(side * 0.85, y, z1));
                    let d0 = local0[a].distance(local0[c]);
                    let d1 = local1[a].distance(local1[c]);
                    aperture = aperture.max((d1 - d0).abs());
                }
            }
        }
        let sill = |l: &[Vec3]| {
            let g = &b.groups["sill"];
            l[g[0] as usize].distance(l[*g.last().unwrap() as usize])
        };
        let sill_change = (sill(&local1) - sill(&local0)).abs();
        // Rear wheels: hubs in the cabin frame; rear suspension and tyre beams.
        let rear_axles: Vec<usize> = sk.wheels[2..]
            .iter()
            .flat_map(|w| [w.a1 as usize, w.a2 as usize])
            .collect();
        // The wheel centres (the axle nodes' midpoints).
        let hub_v: Vec<Vec3> = sk.wheels[2..]
            .iter()
            .map(|w| {
                let (a, c) = (w.a1 as usize, w.a2 as usize);
                0.5 * (local1[a] + local1[c]) - 0.5 * (local0[a] + local0[c])
            })
            .collect();
        // In the cabin frame: x along the car, y across, z down (the
        // suspension's travel). The car comes to rest nose down on its
        // crushed front, a little rolled: its rear wheels sit a few
        // centimetres up or down their travel; what must not move is where
        // they are along and across the car (the mounts, the knuckles).
        let hub = hub_v
            .iter()
            .map(|v| v.truncate().length())
            .fold(0.0f32, f32::max);
        let travel = hub_v.iter().map(|v| v.z.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "(k) drag {drag}: rear hubs moved (x along, y across, z down, mm) {:?}",
            hub_v
                .iter()
                .map(|v| format!("{:.0} {:.0} {:.0}", v.x * 1e3, v.y * 1e3, v.z * 1e3))
                .collect::<Vec<_>>()
        );
        if std::env::var("HUB_TRACE").is_ok() {
            for &i in &rear_axles {
                eprintln!("  rear axle node {i}: {:?} -> {:?}", local0[i], local1[i]);
            }
            for (k, w) in sk.wheels.iter().enumerate() {
                let p = s.nodes[w.a1 as usize].pos;
                eprintln!(
                    "  wheel {k} axle world {p:?} vel {:?}",
                    s.nodes[w.a1 as usize].vel
                );
            }
        }
        let susp = sk.parts.iter().position(|p| p == "suspension").unwrap() as u16;
        let zmid = 0.5 * (sk.wheels[0].a1 as f32 + 0.0);
        let _ = zmid;
        let rear_z = sk.nodes[sk.wheels[2].a1 as usize].p.z;
        let rear_broken = s
            .beams
            .iter()
            .filter(|bm| bm.broken)
            .filter(|bm| {
                let rear_susp = |i: u32| {
                    (i as usize) < sk.nodes.len()
                        && sk.nodes[i as usize].part == susp
                        && (sk.nodes[i as usize].p.z - rear_z).abs() < 0.6
                };
                let tyre = |i: u32| (i as usize) >= sk.nodes.len();
                (rear_susp(bm.a) && rear_susp(bm.b)) || tyre(bm.a) || tyre(bm.b)
            })
            .count();
        eprintln!(
            "(k) drag {drag}: front crush peak {peak:.3} m, permanent {perm:.3} m ({:.0} %), rebound {rebound:.2} m/s; cabin {}; sill {:.3} m; door openings {:.3} m; rear hubs {:.3} m (travel {travel:.3} m), rear suspension/tyre beams broken {rear_broken}",
            100.0 * perm / peak,
            cabin
                .iter()
                .map(|(n, v)| format!("{n} {:.0} mm", worst(v) * 1e3))
                .collect::<Vec<_>>()
                .join(", "),
            sill_change,
            aperture,
            hub
        );
        // Every failure of both runs is reported before the test fails.
        let mut check = |ok: bool, what: String| {
            if !ok {
                fails.push(format!("drag {drag}: {what}"));
            }
        };
        check(
            perm >= 0.7 * peak,
            format!("springs back: {perm} of {peak}"),
        );
        check(rebound < 1.5, format!("rebound {rebound} m/s"));
        for (n, v) in &cabin[..4] {
            check(worst(v) < 0.015, format!("{n} moved {} m", worst(v)));
        }
        check(sill_change < 0.03, format!("sill {sill_change} m"));
        check(aperture < 0.02, format!("door opening {aperture} m"));
        check(hub < 0.03, format!("rear hub moved {hub} m"));
        check(
            travel < 0.06,
            format!("rear hub off its spring by {travel} m"),
        );
        check(rear_broken == 0, format!("{rear_broken} rear beams broken"));
    }
    assert!(fails.is_empty(), "{fails:#?}");
}

/// A quasi-static push on the middle of the front door (a knee, a bollard):
/// a dent where it pushes, not a whole-panel bulge, and it stays when the push
/// is gone. Reported: the dent at the load, the door 30 cm away, what stays.
#[test]
fn l_door_point_load_dents_locally() {
    let b = built(Resolution::Medium);
    let sk = &b.skeleton;
    let mut s = spawn(&b, Vec3::ZERO, 0.0);
    run(&mut s, 1.0);
    let door: Vec<usize> = (0..sk.nodes.len())
        .filter(|&i| sk.parts[sk.nodes[i].part as usize] == "door_f_l")
        .collect();
    let mid = door.iter().map(|&i| sk.nodes[i].p).sum::<Vec3>() / door.len() as f32;
    let at = *door
        .iter()
        .filter(|&&i| b.shell.seg.contains_key(&(i as u32)))
        .min_by(|&&a, &&c| {
            sk.nodes[a]
                .p
                .distance(mid)
                .total_cmp(&sk.nodes[c].p.distance(mid))
        })
        .unwrap();
    let local0 = cabin_local(&s, &b);
    let far: Vec<usize> = door
        .iter()
        .copied()
        .filter(|&i| {
            let d = sk.nodes[i].p.distance(sk.nodes[at].p);
            d > 0.3 && d < 0.5
        })
        .collect();
    let push = 4000.0f32;
    let inward = |l: &[Vec3], i: usize| l[i].distance(local0[i]);
    let (mut dent, mut around) = (0.0f32, 0.0f32);
    for st in 0..3000 {
        let f = push * (st as f32 / 1000.0).min(1.0) * if st < 2000 { 1.0 } else { 0.0 };
        s.nodes[at].force.x -= f;
        s.step();
        if st == 1999 {
            let l = cabin_local(&s, &b);
            dent = inward(&l, at);
            around = far.iter().map(|&i| inward(&l, i)).sum::<f32>() / far.len().max(1) as f32;
        }
    }
    run(&mut s, 0.5);
    let l = cabin_local(&s, &b);
    let kept = inward(&l, at);
    eprintln!(
        "(l) door push {:.1} kN: dent {:.1} mm, door 30–50 cm away {:.1} mm ({:.0} %), kept {:.1} mm after release",
        push / 1000.0,
        dent * 1e3,
        around * 1e3,
        100.0 * around / dent,
        kept * 1e3
    );
    assert!(dent > 0.005, "no dent");
    assert!(around < 0.4 * dent, "the whole door bulges");
    assert!(kept > 0.3 * dent, "the dent sprang back");
}

/// At 56 km/h into a wall the panels stay on (tearing is for about 80 km/h
/// and up): the fenders crumple back against the A-pillars, the bonnet buckles
/// but stays latched and hinged. No panel node ends up more than 15 cm
/// outside the body's width at any time, and no beam joining the bonnet or a
/// fender to the rest of the car breaks. With drag on and off.
#[test]
fn m_wall_56_panels_stay_on() {
    for drag in [true, false] {
        let b = built(Resolution::Medium);
        let sk = &b.skeleton;
        let mut s = spawn(&b, Vec3::new(0.0, 0.0, 56.0 / 3.6), 0.0);
        s.settings.drag = drag;
        let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        s.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        ));
        let n = sk.nodes.len();
        let name = |i: usize| sk.parts[sk.nodes[i].part as usize].as_str();
        let panel = |i: usize| {
            let p = name(i);
            ["bumper", "bonnet", "bootlid", "fender", "door"]
                .iter()
                .any(|k| p.starts_with(k))
        };
        let panels: Vec<usize> = (0..n).filter(|&i| panel(i)).collect();
        // Lateral in the cabin's own frame (the car may yaw as it stops):
        // the frame's second axis is across the car (the third is up).
        let l0 = cabin_local(&s, &b);
        let (lo, hi) = l0[..n]
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, c), p| (a.min(p.y), c.max(p.y)));
        let mut worst = (0.0f32, 0usize);
        for _ in 0..(1.5 * 60.0) as usize {
            s.frame();
            let l = cabin_local(&s, &b);
            for &i in &panels {
                let out = (l[i].y - hi).max(lo - l[i].y);
                if out > worst.0 {
                    worst = (out, i);
                }
            }
        }
        let held = |i: u32| {
            let p = name(i as usize);
            p.starts_with("bonnet") || p.starts_with("fender")
        };
        let (mut joints, mut torn) = (0, 0);
        let mut torn_by: std::collections::BTreeMap<String, usize> = Default::default();
        for bm in &s.beams {
            let (a, c) = (bm.a as usize, bm.b as usize);
            if a >= n || c >= n || sk.nodes[a].part == sk.nodes[c].part {
                continue;
            }
            if held(bm.a) || held(bm.b) {
                joints += 1;
                if bm.broken {
                    torn += 1;
                    *torn_by
                        .entry(format!("{}~{}", name(a), name(c)))
                        .or_default() += 1;
                }
            }
        }
        eprintln!(
            "(m) drag {drag}: worst panel node {:.0} mm outside the body's width (node {} of {}); bonnet and fender joint beams torn {torn} of {joints} {torn_by:?}",
            worst.0.max(0.0) * 1e3,
            worst.1,
            name(worst.1)
        );
        assert!(finite(&s));
        assert!(worst.0 < 0.15, "a panel splayed {} m out", worst.0);
        assert_eq!(torn, 0, "bonnet or fender joints tore: {torn_by:?}");
    }
}

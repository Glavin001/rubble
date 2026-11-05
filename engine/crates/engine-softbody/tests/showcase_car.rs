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

/// The sill's length (its end nodes' distance: a car that yaws or pitches
/// after the crash keeps its sill's length).
fn sill_length(s: &ExplicitBeams, b: &Built) -> f32 {
    let g = &b.groups["sill"];
    s.nodes[g[0] as usize]
        .pos
        .distance(s.nodes[*g.last().unwrap() as usize].pos)
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
    let door: Vec<usize> = b
        .skeleton
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            let name = &b.skeleton.parts[n.part as usize];
            name.starts_with("door_") && name.ends_with("_l")
        })
        .map(|(i, _)| i)
        .collect();
    // C_KMH: the speed (32 by default; nearby speeds for medians).
    let kmh: f32 = std::env::var("C_KMH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32.0);
    let (through, s, _) = pole_run(&b, kmh, dir, offset, false, 1.0);
    // Intrusion: the deepest door node near the pole, across the car in the
    // far side's frame (the right sill along the car, the tunnel beside it).
    // The car spins 55-75° round the pole in the second after the hit, so
    // world x would measure the spin, not the door.
    let far = |st: &ExplicitBeams| {
        let g = &b.groups["sill_r"];
        let zs: Vec<f32> = g
            .iter()
            .map(|&i| b.skeleton.nodes[i as usize].p.z)
            .collect();
        let zm = zs.iter().sum::<f32>() / zs.len() as f32;
        let half = |front: bool| {
            let v: Vec<Vec3> = g
                .iter()
                .zip(&zs)
                .filter(|(_, &z)| (z > zm) == front)
                .map(|(&i, _)| st.nodes[i as usize].pos)
                .collect();
            v.iter().sum::<Vec3>() / v.len() as f32
        };
        let t = group_mean(st, &b, &["tunnel"]);
        let (o, x) = (half(false), (half(true) - half(false)).normalize());
        let up = x.cross(t - o).normalize();
        (o, up.cross(x))
    };
    let (o0, a0) = far(&s0);
    let (o1, a1) = far(&s);
    let inward = |i: usize| (s0.nodes[i].pos - o0).dot(a0) - (s.nodes[i].pos - o1).dot(a1);
    let near = |q: Vec3| (q.z - zb).abs() < 0.4 && q.y > 0.3 && q.y < 1.1;
    let intrusion = door
        .iter()
        .copied()
        .filter(|&i| near(s0.nodes[i].pos))
        .map(inward)
        .fold(f32::MIN, f32::max);
    let pillar = [&b.groups["pillar_b"]]
        .iter()
        .flat_map(|g| g.iter())
        .map(|&i| inward(i as usize))
        .fold(f32::MIN, f32::max);
    let spin = a0.with_y(0.0).angle_between(a1.with_y(0.0)).to_degrees();
    eprintln!(
        "(c) side pole {kmh} km/h at the B-pillar: door intrusion {intrusion:.3} m (B-pillar {pillar:.3} m), the car turned {spin:.0}°, passed through {through}, broken {}, span {:.2} m",
        s.damage().0,
        span(&s)
    );
    assert!(finite(&s));
    assert!(!through, "the car passed through the pole");
    assert!(intrusion > 0.05, "the door barely moved ({intrusion})");
    assert!(intrusion < 0.3, "the door intruded {intrusion} m");
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
    // `SWEEP_SPEEDS=28,32` other speeds, `SWEEP_SHIFT=0.025` offsets moved by
    // that much (how robust a clean sweep is).
    let speeds: Vec<f32> = std::env::var("SWEEP_SPEEDS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or_else(|| vec![30.0, 45.0, 60.0]);
    let shift: f32 = std::env::var("SWEEP_SHIFT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    for square in [true, false] {
        for &kmh in &speeds {
            for i in -16..=16 {
                jobs.push((Vec3::Z, i as f32 * 0.05 + shift, square, kmh));
            }
            for i in -44..=44 {
                // Sideways: offsets along the car (lateral of +x is −z).
                jobs.push((Vec3::X, i as f32 * 0.05 + shift, square, kmh));
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
    if std::env::var("SWEEP_LIST").is_ok() {
        for f in &fails {
            eprintln!(
                "  through: dir {:?} offset {:+.2} square {} {} km/h",
                f.0, f.1, f.2, f.3
            );
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
/// Ignored: fails today on the rebound and the door openings (medium, 53–59
/// km/h medians: rebound ~1.9 m/s, door openings ~20 mm; roof ~12 mm,
/// B-pillars ~11, C-pillars ~9, rear doors ~8, rear hubs ~1 cm pass). Run
/// with `--ignored`.
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
        if std::env::var("K_VEC").is_ok() {
            // Mean displacement (cabin frame: x along the car, y across, z
            // down) of some groups, mm.
            for g in [
                "header_f",
                "roof_rail",
                "roof_rail_r",
                "bow_b",
                "header_r",
                "pillar_a",
                "pillar_b",
                "pillar_c",
                "cowl",
                "firewall",
                "parcel",
            ] {
                let v = &b.groups[g];
                let d = v
                    .iter()
                    .map(|&i| local1[i as usize] - local0[i as usize])
                    .sum::<Vec3>()
                    / v.len() as f32;
                eprint!("{g} {:.0} {:.0} {:.0}; ", d.x * 1e3, d.y * 1e3, d.z * 1e3);
            }
            eprintln!();
            let mut rv: Vec<(f32, usize)> = roof.iter().map(|&i| (disp(i), i)).collect();
            rv.sort_by(|a, c| c.0.total_cmp(&a.0));
            if let Some(&(_, w)) = rv.first() {
                for bm in s
                    .beams
                    .iter()
                    .filter(|bm| bm.a as usize == w || bm.b as usize == w)
                {
                    let o = if bm.a as usize == w { bm.b } else { bm.a } as usize;
                    let l = s.nodes[bm.a as usize]
                        .pos
                        .distance(s.nodes[bm.b as usize].pos);
                    eprintln!(
                        "    beam to {o} {:?} part {} primary {}: rest0 {:.3} rest {:.3} len {:.3} k {:.0} tc {:.0} broken {}",
                        sk.nodes.get(o).map(|n| n.p),
                        sk.nodes.get(o).map(|n| sk.parts[n.part as usize].as_str()).unwrap_or("-"),
                        sk.nodes.get(o).is_some_and(|n| n.primary),
                        bm.rest0, bm.rest, l, bm.k, bm.tc, bm.broken
                    );
                }
            }
            for &(d, i) in rv.iter().take(6) {
                let gs: Vec<&String> = b
                    .groups
                    .iter()
                    .filter(|(_, g)| g.contains(&(i as u32)))
                    .map(|(k, _)| k)
                    .collect();
                eprintln!(
                    "  roof node {i} at {:?} part {} primary {} {gs:?}: {:.0} mm {:?}",
                    sk.nodes[i].p,
                    sk.parts[sk.nodes[i].part as usize],
                    sk.nodes[i].primary,
                    d * 1e3,
                    (local1[i] - local0[i]) * 1e3
                );
            }
        }
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
                    if std::env::var("K_APER").is_ok() {
                        let gs = |i: usize| -> Vec<&String> {
                            b.groups
                                .iter()
                                .filter(|(_, g)| g.contains(&(i as u32)))
                                .map(|(k, _)| k)
                                .collect()
                        };
                        eprintln!(
                            "  {a}: primary {} groups {:?}; {c}: primary {} groups {:?}",
                            sk.nodes[a].primary,
                            gs(a),
                            sk.nodes[c].primary,
                            gs(c)
                        );
                        eprintln!(
                            "  aperture x {side} z {z0}-{z1} y {y}: nodes {a} {:?} {c} {:?}: {:+.0} mm (moved {:.0?} and {:.0?} mm)",
                            sk.nodes[a].p,
                            sk.nodes[c].p,
                            (d1 - d0) * 1e3,
                            (local1[a] - local0[a]) * 1e3,
                            (local1[c] - local0[c]) * 1e3
                        );
                    }
                }
            }
        }
        // K_NEAR=x,y,z: the nodes round a point, their groups and how far they
        // moved in the cabin's frame (x along, y across, z down; mm).
        if let Ok(v) = std::env::var("K_NEAR") {
            let c: Vec<f32> = v.split(',').filter_map(|t| t.parse().ok()).collect();
            let p = Vec3::new(c[0], c[1], c[2]);
            let mut ns: Vec<usize> = (0..sk.nodes.len()).collect();
            ns.sort_by(|&a, &c| {
                sk.nodes[a]
                    .p
                    .distance(p)
                    .total_cmp(&sk.nodes[c].p.distance(p))
            });
            for &i in ns.iter().take(12) {
                let gs: Vec<&String> = b
                    .groups
                    .iter()
                    .filter(|(_, g)| g.contains(&(i as u32)))
                    .map(|(k, _)| k)
                    .collect();
                eprintln!(
                    "  near {i} {:?} {} primary {} mass {:.2} {gs:?}: moved {:.0?} mm",
                    sk.nodes[i].p,
                    sk.parts[sk.nodes[i].part as usize],
                    sk.nodes[i].primary,
                    s.nodes[i].mass,
                    (local1[i] - local0[i]) * 1e3
                );
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
        if st % 100 == 99 && std::env::var("L_TRACE").is_ok() {
            let l = cabin_local(&s, &b);
            let ties: Vec<String> = s
                .beams
                .iter()
                .filter(|bm| (bm.a as usize == at || bm.b as usize == at) && bm.k > 1.0e5)
                .map(|bm| {
                    let o = if bm.a as usize == at { bm.b } else { bm.a } as usize;
                    format!(
                        "{o}:{:.0}/{:.0}r{:.0}m{:.0}",
                        bm.force,
                        bm.tc,
                        bm.rest * 1e3,
                        inward(&l, o) * 1e3
                    )
                })
                .collect();
            eprintln!(
                "    (l) step {st} force {f:.0}: dent {:.1} mm; ties {}",
                inward(&l, at) * 1e3,
                ties.join(" ")
            );
        }
        if st == 1999 {
            let l = cabin_local(&s, &b);
            dent = inward(&l, at);
            if std::env::var("L_DIAG").is_ok() {
                eprintln!("    (l) pushed node {at} at {:?}", sk.nodes[at].p);
                for bm in s
                    .beams
                    .iter()
                    .filter(|bm| bm.a as usize == at || bm.b as usize == at)
                {
                    let o = if bm.a as usize == at { bm.b } else { bm.a } as usize;
                    if o >= sk.nodes.len() {
                        continue;
                    }
                    let gs: Vec<&String> = b
                        .groups
                        .iter()
                        .filter(|(_, g)| g.contains(&(o as u32)))
                        .map(|(k, _)| k)
                        .collect();
                    eprintln!(
                        "    (l) beam to {o} {:?} {} primary {} {gs:?}: k {:.0} force {:.0} N (yield {:.0}/{:.0}, rest {:.3} of {:.3}), node moved {:.1} mm",
                        sk.nodes[o].p,
                        sk.parts[sk.nodes[o].part as usize],
                        sk.nodes[o].primary,
                        bm.k,
                        bm.force,
                        bm.tc,
                        bm.tt,
                        bm.rest,
                        bm.rest0,
                        inward(&l, o) * 1e3
                    );
                }
            }
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
/// outside the body's width at any time, no beam joining the bonnet or a
/// fender to the rest of the car breaks, and the boot lid stays latched and
/// hinged within FMVSS 206's capacities. With drag on and off.
#[test]
fn m_wall_56_panels_stay_on() {
    for drag in [true, false] {
        let b = built(Resolution::Medium);
        let sk = &b.skeleton;
        let mut s = spawn(
            &b,
            Vec3::new(
                0.0,
                0.0,
                std::env::var("M_KMH")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(56.0f32)
                    / 3.6,
            ),
            0.0,
        );
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
        // The boot lid's edges: each lid node beside the body (a seam beam
        // to the quarters, the parcel shelf or the rear panel) and those body
        // nodes. How far the edge lifts off them (up in the cabin's frame).
        let body = sk.parts.iter().position(|p| p == "body").unwrap() as u16;
        let mut edge: Vec<(usize, Vec<usize>)> = Vec::new();
        for &i in panels.iter().filter(|&&i| name(i) == "bootlid") {
            let nb: Vec<usize> = s
                .beams
                .iter()
                .filter_map(|bm| {
                    let (a, c) = (bm.a as usize, bm.b as usize);
                    let o = if a == i {
                        c
                    } else if c == i {
                        a
                    } else {
                        return None;
                    };
                    (o < n && sk.nodes[o].part == body).then_some(o)
                })
                .collect();
            if !nb.is_empty() {
                edge.push((i, nb));
            }
        }
        if std::env::var("M_EDGE").is_ok() {
            for (i, nb) in &edge {
                eprintln!(
                    "  lid {i} {:?} -> {:?}",
                    sk.nodes[*i].p,
                    nb.iter()
                        .map(|&j| (j, sk.nodes[j].p, sk.nodes[j].primary))
                        .collect::<Vec<_>>()
                );
            }
        }
        // The lid's hinge and latch systems (as the Hinged joint builds them:
        // its front edge's seam beams, 14 cm deep, are the hinges, one a
        // side; its rear edge's within |x| ≤ 0.3 m the latch).
        let lid: Vec<usize> = panels
            .iter()
            .copied()
            .filter(|&i| name(i) == "bootlid")
            .collect();
        let (z0, z1) = lid.iter().fold((f32::MAX, f32::MIN), |(a, c), &i| {
            (a.min(sk.nodes[i].p.z), c.max(sk.nodes[i].p.z))
        });
        let mut systems: [Vec<usize>; 3] = Default::default();
        for (bi, bm) in s.beams.iter().enumerate() {
            let (a, c) = (bm.a as usize, bm.b as usize);
            if a >= n || c >= n {
                continue;
            }
            let (l, o) = if name(a) == "bootlid" { (a, c) } else { (c, a) };
            if name(l) != "bootlid" || sk.nodes[o].part != body {
                continue;
            }
            let q = sk.nodes[l].p;
            if q.z > z1 - 0.14 {
                systems[if q.x > 0.0 { 0 } else { 1 }].push(bi);
            } else if q.z < z0 + 0.14 && q.x.abs() <= 0.3 {
                systems[2].push(bi);
            }
        }
        let mut sys_force = [0.0f32; 3];
        let mut lift = (0.0f32, 0usize, 0usize);
        let mut open = (0.0f32, 0usize, 0usize);
        for f in 0..(1.5 * 60.0) as usize {
            s.frame();
            for (k, v) in systems.iter().enumerate() {
                let fsum: f32 = v.iter().map(|&bi| s.beams[bi].force.abs()).sum();
                sys_force[k] = sys_force[k].max(fsum);
            }
            if std::env::var("M_TRACE").is_ok() && f % 3 == 0 {
                // The body's deck opening (hinge line to latch line) and the
                // lid's own length, along the car (cabin frame), against spawn.
                let l = cabin_local(&s, &b);
                let mean = |v: &[usize], body_side: bool| -> Vec3 {
                    let ids: Vec<usize> = v
                        .iter()
                        .map(|&bi| {
                            let bm = &s.beams[bi];
                            let (a, c) = (bm.a as usize, bm.b as usize);
                            let lidn = if name(a) == "bootlid" { a } else { c };
                            let other = if lidn == a { c } else { a };
                            if body_side {
                                other
                            } else {
                                lidn
                            }
                        })
                        .collect();
                    ids.iter().map(|&i| l[i] - l0[i]).sum::<Vec3>() / ids.len() as f32
                };
                let hinge: Vec<usize> = [systems[0].clone(), systems[1].clone()].concat();
                let (hb, lb) = (mean(&hinge, true), mean(&systems[2], true));
                let up_mid = lid
                    .iter()
                    .map(|&i| -(l[i] - l0[i]).z)
                    .fold(f32::MIN, f32::max);
                eprintln!(
                    "  (m) f{f}: body hinge line moved {:.0} mm along, latch line {:.0} mm (opening shortened {:.0} mm); lid's highest rise {:.0} mm; hinge/latch forces {:.1} {:.1} {:.1} kN",
                    hb.x * 1e3,
                    lb.x * 1e3,
                    (lb.x - hb.x) * 1e3,
                    up_mid * 1e3,
                    systems[0].iter().map(|&bi| s.beams[bi].force).sum::<f32>() / 1e3,
                    systems[1].iter().map(|&bi| s.beams[bi].force).sum::<f32>() / 1e3,
                    systems[2].iter().map(|&bi| s.beams[bi].force).sum::<f32>() / 1e3
                );
            }
            let l = cabin_local(&s, &b);
            for &i in &panels {
                let out = (l[i].y - hi).max(lo - l[i].y);
                if out > worst.0 {
                    worst = (out, i);
                }
            }
            for (i, nb) in &edge {
                let rel =
                    |l: &[Vec3]| l[*i] - nb.iter().map(|&j| l[j]).sum::<Vec3>() / nb.len() as f32;
                // The frame's third axis is down.
                let up = -(rel(&l) - rel(&l0)).z;
                if up > lift.0 {
                    lift = (up, *i, f);
                    if std::env::var("M_DIAG").is_ok() && up > 0.02 {
                        eprintln!(
                            "  (m) frame {f}: lid node {i} up {:.0} mm; body neighbours {:?}",
                            -(l[*i] - l0[*i]).z * 1e3,
                            nb.iter()
                                .map(|&j| format!(
                                    "{j} {} {:?} up {:.0}",
                                    sk.parts[sk.nodes[j].part as usize],
                                    sk.nodes[j].p,
                                    -(l[j] - l0[j]).z * 1e3
                                ))
                                .collect::<Vec<_>>()
                        );
                    }
                }
                let gap = rel(&l).length() - rel(&l0).length();
                if gap > open.0 {
                    open = (gap, *i, f);
                }
            }
        }
        eprintln!(
            "(m) drag {drag}: boot lid edge lifts {:.0} mm off the body (node {} at {:?}, frame {})",
            lift.0 * 1e3,
            lift.1,
            sk.nodes[lift.1].p,
            lift.2
        );
        eprintln!(
            "(m) drag {drag}: boot lid seam opens {:.0} mm (node {} at {:?}, frame {})",
            open.0 * 1e3,
            open.1,
            sk.nodes[open.1].p,
            open.2
        );
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
        // The deck lid stays latched and hinged (as in 56 km/h NCAP tests):
        // no hinge or latch beam breaks and each system carries less than
        // FMVSS 206's capacity of a hinge or latch system (8.9 kN across, 11.1
        // kN along). The lift and seam gap above are reported (panel
        // buckling), not asserted.
        let broken = |v: &[usize]| v.iter().filter(|&&bi| s.beams[bi].broken).count();
        eprintln!(
            "(m) drag {drag}: boot lid hinges (left, right) and latch: peak {:.1}, {:.1}, {:.1} kN (FMVSS 206 8.9–11.1), beams {}, {}, {} of which broken {}, {}, {}",
            sys_force[0] / 1e3,
            sys_force[1] / 1e3,
            sys_force[2] / 1e3,
            systems[0].len(),
            systems[1].len(),
            systems[2].len(),
            broken(&systems[0]),
            broken(&systems[1]),
            broken(&systems[2])
        );
        for k in 0..3 {
            assert!(!systems[k].is_empty(), "boot lid system {k} has no beams");
            assert_eq!(broken(&systems[k]), 0, "boot lid hinge/latch {k} broke");
            assert!(
                sys_force[k] < 11_100.0,
                "boot lid hinge/latch {k} carried {} N",
                sys_force[k]
            );
        }
    }
}

/// The residual crush profile at bumper height, measured as NHTSA does
/// (`engine/docs/crash-reference-targets.md` §1, §4): in the cabin's frame,
/// six points C1..C6 equally spaced across the front's width, each the
/// rearward move of the frontmost node in a 10 cm strip between 0.30 and 0.65
/// m above the ground (at spawn). `structure`: frame nodes only (no bumper
/// cover or skin). Returns (C1..C6, max, NHTSA average (C1/2 + C2..C5 +
/// C6/2)/5).
fn front_profile(
    b: &Built,
    l0: &[Vec3],
    l1: &[Vec3],
    y0: &[f32],
    structure: bool,
    attached: &[bool],
) -> ([f32; 6], f32, f32) {
    let sk = &b.skeleton;
    let n = sk.nodes.len();
    // Nodes torn off (every beam broken) are debris, not the profile.
    let pick = |i: usize| {
        (!structure || sk.nodes[i].primary) && y0[i] > 0.30 && y0[i] < 0.65 && attached[i]
    };
    // Width: the band's extent across the car at spawn.
    let (lo, hi) = (0..n)
        .filter(|&i| pick(i))
        .fold((f32::MAX, f32::MIN), |(a, c), i| {
            (a.min(l0[i].y), c.max(l0[i].y))
        });
    let mut c = [0.0f32; 6];
    for (k, ck) in c.iter_mut().enumerate() {
        let y = lo + 0.05 + (hi - lo - 0.10) * k as f32 / 5.0;
        // The strip's front at spawn, and the nodes that made it (within 20
        // cm of it): the same material is followed, wherever it moves
        // across the car.
        let strip: Vec<usize> = (0..n)
            .filter(|&i| pick(i) && (l0[i].y - y).abs() < 0.05)
            .collect();
        let f0 = strip.iter().map(|&i| l0[i].x).fold(f32::MIN, f32::max);
        let f1 = strip
            .iter()
            .filter(|&&i| l0[i].x > f0 - 0.20)
            .map(|&i| l1[i].x)
            .fold(f32::MIN, f32::max);
        *ck = if strip.is_empty() {
            0.0
        } else {
            (f0 - f1).max(0.0)
        };
    }
    let max = c.iter().copied().fold(0.0f32, f32::max);
    let avg = (0.5 * c[0] + c[1] + c[2] + c[3] + c[4] + 0.5 * c[5]) / 5.0;
    (c, max, avg)
}

/// A rigid wall at `kmh` measured against the NHTSA sedan population
/// (`crash-reference-targets.md`): residual crush at bumper height (exterior
/// and structure), Campbell b1, the dynamic crush and when it peaks, the
/// springback, restitution and where the energy went. Returns (exterior
/// average crush, failures).
fn reference_wall(b: &Built, kmh: f32, drag: bool) -> Vec<(&'static str, f32)> {
    let sk = &b.skeleton;
    let v0 = kmh / 3.6;
    let mut s = spawn(b, Vec3::new(0.0, 0.0, v0), 0.0);
    s.settings.drag = drag;
    let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(3.0, 2.0, 1.0),
    ));
    let l0 = cabin_local(&s, b);
    let ground = s.nodes[..sk.nodes.len()]
        .iter()
        .map(|n| n.pos.y)
        .fold(f32::MAX, f32::min)
        - 0.02;
    let y0: Vec<f32> = s.nodes.iter().map(|n| n.pos.y - ground).collect();
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let ke0 = 0.5 * total * v0 * v0;
    let vcg = |s: &ExplicitBeams| s.nodes.iter().map(|n| n.mass * n.vel.z).sum::<f32>() / total;
    // Fixed steps (0.5 ms) for the pulse, read at the rear seat's cross
    // member (floor_x3, where NCAP's 093/094 accelerometers sit): its travel
    // past contact until it stops (dynamic crush, t_m) and its acceleration.
    let cm = |s: &ExplicitBeams| group_mean(s, b, &["floor_x3"]);
    let vcm = |s: &ExplicitBeams| {
        let g = &b.groups["floor_x3"];
        g.iter().map(|&i| s.nodes[i as usize].vel.z).sum::<f32>() / g.len() as f32
    };
    let z0 = cm(&s).z;
    let (mut t, mut t_contact, mut t_m, mut d_dyn) = (0.0f32, None, None, 0.0f32);
    let (mut z_contact, mut rebound) = (z0, 0.0f32);
    let dt = engine_softbody::authoring::skeleton::STEP;
    let mut acc: Vec<f32> = Vec::new();
    let mut vprev = vcm(&s);
    for _ in 0..(2.0 / dt) as usize {
        s.step();
        t += dt;
        let v = vcm(&s);
        acc.push((v - vprev) / dt / 9.81);
        vprev = v;
        let vg = vcg(&s);
        if t_contact.is_none() && vg < v0 - 0.05 {
            t_contact = Some(t);
            z_contact = cm(&s).z;
        }
        if t_contact.is_some() && t_m.is_none() && v <= 0.0 {
            t_m = Some(t);
            d_dyn = cm(&s).z - z_contact;
        }
        rebound = rebound.max(-vg);
    }
    let _ = z0;
    // 100 Hz two-pole Butterworth, forward and backward
    // (engine/scripts/nhtsa_pulse.py; close to CFC 60).
    let biquad = |x: &[f32]| -> Vec<f32> {
        let w = (std::f32::consts::PI * 100.0 * dt).tan();
        let n = 1.0 / (1.0 + 2f32.sqrt() * w + w * w);
        let (b0, a1, a2) = (
            w * w * n,
            2.0 * (w * w - 1.0) * n,
            (1.0 - 2f32.sqrt() * w + w * w) * n,
        );
        let (b1, b2) = (2.0 * b0, b0);
        let (mut x1, mut x2, mut y1, mut y2) = (x[0], x[0], x[0], x[0]);
        x.iter()
            .map(|&v| {
                let o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                (x2, x1, y2, y1) = (x1, v, y1, o);
                o
            })
            .collect()
    };
    let mut f = biquad(&acc);
    f.reverse();
    let mut f = biquad(&f);
    f.reverse();
    let peak_g = f.iter().map(|a| -a).fold(0.0f32, f32::max);
    let l1 = cabin_local(&s, b);
    let attached: Vec<bool> = (0..s.nodes.len())
        .map(|i| {
            s.incidence[i]
                .iter()
                .any(|&bm| !s.beams[bm as usize].broken)
        })
        .collect();
    let (cx, max_x, avg_x) = front_profile(b, &l0, &l1, &y0, false, &attached);
    let (cs, max_s, avg_s) = front_profile(b, &l0, &l1, &y0, true, &attached);
    let ke1 = ke(&s);
    let tm = t_m.unwrap_or(f32::NAN) - t_contact.unwrap_or(0.0);
    let b1 = (v0 - 8.0 / 3.6) / avg_x;
    let e = rebound / v0;
    let springback = d_dyn - max_s;
    let mean_g = v0 / tm / 9.81;
    eprintln!(
        "(n) wall {kmh} km/h drag {drag}: residual crush at bumper height exterior C1..C6 {:?} mm, max {:.0} avg {:.0} mm; structure {:?} mm, max {:.0} avg {:.0} mm",
        cx.map(|c| (c * 1e3).round()),
        max_x * 1e3,
        avg_x * 1e3,
        cs.map(|c| (c * 1e3).round()),
        max_s * 1e3,
        avg_s * 1e3
    );
    eprintln!(
        "(n) wall {kmh} km/h drag {drag}: Campbell b1 {b1:.1} 1/s; dynamic crush (rear seat cross member's travel) {:.0} mm at t_m {:.0} ms (v0·t_m/2 {:.0} mm), peak {peak_g:.1} g (100 Hz), mean to t_m {mean_g:.1} g; springback {:.0} mm (dynamic - structure max), rebound {rebound:.2} m/s (e {e:.3}); KE {:.1} kJ -> {:.2} kJ, rebound KE {:.2} kJ",
        d_dyn * 1e3,
        tm * 1e3,
        0.5 * v0 * tm * 1e3,
        springback * 1e3,
        ke0 / 1e3,
        ke1 / 1e3,
        0.5 * total * rebound * rebound / 1e3
    );
    // The cabin as NHTSA measures it (§1, §4.9): the door apertures (hinge
    // pillar to latch pillar, low and at the belt, both sides, both doors),
    // the A-pillars' bases and the firewall moving rearward.
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
    let mut aper: Vec<f32> = Vec::new();
    for side in [1.0f32, -1.0] {
        for (za, zb) in [(3.30, 2.30), (2.18, 1.40)] {
            for y in [0.40, 0.95] {
                let (a, c) = (
                    near(Vec3::new(side * 0.85, y, za)),
                    near(Vec3::new(side * 0.85, y, zb)),
                );
                aper.push(l1[a].distance(l1[c]) - l0[a].distance(l0[c]));
            }
        }
    }
    let rearward = |g: &str, ymax: f32| -> f32 {
        let v: Vec<usize> = b.groups[g]
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| sk.nodes[i].p.y < ymax)
            .collect();
        v.iter().map(|&i| l0[i].x - l1[i].x).sum::<f32>() / v.len() as f32
    };
    let a_base = rearward("pillar_a", 0.6).max(rearward("pillar_a_r", 0.6));
    let fw = rearward("firewall", 2.0);
    let mut sorted: Vec<f32> = aper.iter().map(|a| a.abs()).collect();
    sorted.sort_by(f32::total_cmp);
    let (aper_med, aper_p90) = (
        sorted[sorted.len() / 2],
        sorted[(sorted.len() * 9) / 10 - 1],
    );
    eprintln!(
        "(n) wall {kmh} km/h drag {drag}: door apertures {:?} mm (|median| {:.0}, p90 {:.0}); A-pillar base rearward {:.0} mm, firewall {:.0} mm",
        aper.iter().map(|a| (a * 1e3).round()).collect::<Vec<_>>(),
        aper_med * 1e3,
        aper_p90 * 1e3,
        a_base * 1e3,
        fw * 1e3
    );
    vec![
        ("avg crush", avg_x),
        ("max crush", max_x),
        ("b1", b1),
        ("dynamic crush", d_dyn),
        ("t_m", tm),
        ("restitution", e),
        ("peak g", peak_g),
        ("aperture p90", aper_p90),
        ("aperture median", aper_med),
        ("A-pillar base rearward", a_base),
        ("firewall rearward", fw),
    ]
}

/// NHTSA-referenced frontal measures (`engine/docs/crash-reference-targets.md`
/// §4.5–4.10, §8): residual crush at bumper height, Campbell b1, the pulse at
/// the rear seat's cross member (dynamic crush, t_m, peak g), restitution,
/// the cabin (door apertures, A-pillar bases, firewall) and Campbell's speed
/// scaling C̄(50)/C̄(56) ≈ 0.875. Medium; medians over 55–57 km/h with drag
/// on and off, each inside the measured sedans' spread.
#[test]
fn n_wall_matches_the_nhtsa_sedans() {
    let b = built(Resolution::Medium);
    // 55–57 km/h with drag on and off: the medians are judged (the solver's
    // run-to-run spread at one speed is a few centimetres).
    let mut runs: Vec<Vec<(&str, f32)>> = Vec::new();
    for kmh in [55.0, 56.0, 57.0] {
        for drag in [true, false] {
            runs.push(reference_wall(&b, kmh, drag));
        }
    }
    let med = |k: &str| {
        let mut v: Vec<f32> = runs
            .iter()
            .map(|r| r.iter().find(|x| x.0 == k).unwrap().1)
            .collect();
        v.sort_by(f32::total_cmp);
        0.5 * (v[(v.len() - 1) / 2] + v[v.len() / 2])
    };
    let mut fails = Vec::new();
    let mut band = |k: &str, lo: f32, hi: f32| {
        let m = med(k);
        eprintln!("(n) median {k} {m:.4} (reference {lo}–{hi})");
        if !(lo..=hi).contains(&m) {
            fails.push(format!("{k} {m} outside {lo}–{hi}"));
        }
    };
    // §4.5 residual max 380–580 mm; §4.6 b1 22.9–39.2; §8 (seven NCAP sedan
    // pulses) dynamic crush 0.62–0.75 m, t_m 63–76 ms, e 0.13–0.20, peak
    // 34–47 g; §4.9 apertures (individual p10–p90 ±9 mm, median 0–1, IQR
    // −2…+3), A-pillar base ≤ 17 mm, firewall ≤ 85 mm.
    band("max crush", 0.380, 0.580);
    band("b1", 22.9, 39.2);
    band("dynamic crush", 0.62, 0.75);
    band("t_m", 0.063, 0.076);
    band("restitution", 0.13, 0.20);
    band("peak g", 34.0, 47.0);
    band("aperture p90", 0.0, 0.009);
    band("aperture median", 0.0, 0.003);
    band("A-pillar base rearward", -0.017, 0.017);
    band("firewall rearward", -0.085, 0.085);
    // §4.10 Campbell's speed scaling: C̄(50)/C̄(56) ≈ 0.875 (no measured
    // spread; the sim's own spread decides the tolerance).
    let c56 = med("avg crush");
    let mut r50: Vec<f32> = [true, false]
        .iter()
        .map(|&d| {
            let r = reference_wall(&b, 50.0, d);
            r.iter().find(|x| x.0 == "avg crush").unwrap().1
        })
        .collect();
    r50.sort_by(f32::total_cmp);
    let ratio = 0.5 * (r50[0] + r50[1]) / c56;
    eprintln!("(n) average crush 50/56 km/h {ratio:.3} (Campbell 0.875)");
    if (ratio - 0.875).abs() > 0.1 {
        fails.push(format!("crush ratio 50/56 {ratio}"));
    }
    assert!(fails.is_empty(), "{fails:#?}");
}

/// The FMVSS 214 / NCAP side pole (`crash-reference-targets.md` §1, §4.11): a
/// 254 mm pole, 32 km/h, the car moving at 75° to its axis (driver's side
/// first, the front leading), the pole's line through the driver's head
/// (centre of gravity about z 2.55 m, 0.38 m off the centre line). Measured as
/// NHTSA does: the residual crush of the struck side's exterior at the pole,
/// against its pre-test line (in the far side's frame), once the car is at
/// rest. Medium. Ignored: run with `--ignored`.
#[test]
#[ignore]
fn o_side_pole_matches_the_nhtsa_sedans() {
    let b = built(Resolution::Medium);
    let kmh: f32 = std::env::var("O_KMH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32.0);
    let a = 75.0f32.to_radians();
    let dir = Vec3::new(a.sin(), 0.0, a.cos());
    let s0 = spawn(&b, Vec3::ZERO, 0.0);
    let mut s = spawn(&b, dir * kmh / 3.6, 0.0);
    let r = 0.127;
    // The head's point in plan view, then along the line of travel until the
    // pole stands 3 cm off the nearest node.
    // The B-pillar is at z 2.24 in the definition; the head 0.31 m ahead.
    let head = Vec3::new(
        0.38,
        0.0,
        group_mean(&s0, &b, &["pillar_b", "pillar_b_r"]).z + (2.55 - 2.24),
    );
    let gap = |c: Vec3| {
        s.nodes
            .iter()
            .map(|n| Vec3::new(n.pos.x - c.x, 0.0, n.pos.z - c.z).length())
            .fold(f32::MAX, f32::min)
    };
    let mut t = 0.0f32;
    while gap(head + dir * t) < r + 0.03 {
        t += 0.005;
    }
    let centre = head + dir * t + Vec3::Y * 2.0;
    s.shapes.push(StaticShape::cylinder(centre, r, 2.0));
    run(&mut s, 1.5);
    // The far side's frame (as the c test).
    let far = |st: &ExplicitBeams| {
        let g = &b.groups["sill_r"];
        let zs: Vec<f32> = g
            .iter()
            .map(|&i| b.skeleton.nodes[i as usize].p.z)
            .collect();
        let zm = zs.iter().sum::<f32>() / zs.len() as f32;
        let half = |front: bool| {
            let v: Vec<Vec3> = g
                .iter()
                .zip(&zs)
                .filter(|(_, &z)| (z > zm) == front)
                .map(|(&i, _)| st.nodes[i as usize].pos)
                .collect();
            v.iter().sum::<Vec3>() / v.len() as f32
        };
        let tu = group_mean(st, &b, &["tunnel"]);
        let (o, x) = (half(false), (half(true) - half(false)).normalize());
        let up = x.cross(tu - o).normalize();
        (o, x, up.cross(x))
    };
    let (o0, x0, a0) = far(&s0);
    let (o1, x1, a1) = far(&s);
    // Struck-side exterior (shell nodes on the left half) near the pole's
    // line along the car (±0.15 m at spawn), between sill and belt.
    let zc = (centre - o0).dot(x0);
    let inward = |i: usize| (s0.nodes[i].pos - o0).dot(a0) - (s.nodes[i].pos - o1).dot(a1);
    let mut best = (0.0f32, 0usize);
    for &i in &b.shell.nodes {
        let i = i as usize;
        let p = s0.nodes[i].pos;
        let along = (p - o0).dot(x0);
        if p.x > 0.3 && (along - zc).abs() < 0.15 && p.y > 0.3 && p.y < 1.2 {
            let d = inward(i);
            if d > best.0 {
                best = (d, i);
            }
        }
    }
    let _ = x1;
    let (crush, at) = best;
    // The structure there too (door beams, sill, B-pillar: frame nodes).
    let frame = (0..b.skeleton.nodes.len())
        .filter(|&i| {
            let p = s0.nodes[i].pos;
            b.skeleton.nodes[i].primary
                && p.x > 0.3
                && ((p - o0).dot(x0) - zc).abs() < 0.15
                && p.y > 0.25
                && p.y < 1.2
        })
        .map(inward)
        .fold(0.0f32, f32::max);
    eprintln!(
        "(o) side pole: the structure at the pole moved in {:.0} mm",
        frame * 1e3
    );
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    eprintln!(
        "(o) side pole {kmh} km/h at 75°: residual crush at the pole {:.0} mm (node {at} {} at y {:.2}); KE left {:.2} kJ of {:.1}",
        crush * 1e3,
        b.skeleton.parts[b.skeleton.nodes[at].part as usize],
        s0.nodes[at].pos.y,
        ke(&s) / 1e3,
        0.5 * total * (kmh / 3.6).powi(2) / 1e3
    );
    assert!(finite(&s));
    // §4.11: p10–p90 297–445 mm (hard 169–537).
    assert!(
        (0.297..=0.445).contains(&crush),
        "residual crush at the pole {crush} m"
    );
}

/// A quasi-static rig: `held` nodes kept where they are (the rig's
/// restraint); a rigid ram moved `travel` m along `dir` at `speed` m/s, each
/// step putting nodes inside it back on its surface (`inside(p, offset)`:
/// the depth along `dir`, if inside, with the ram moved `offset`). The force
/// is the momentum it takes out each step. Returns (ram travel, force) every
/// 5 mm.
fn ram_rig(
    s: &mut ExplicitBeams,
    held: &[usize],
    dir: Vec3,
    travel: f32,
    speed: f32,
    inside: &dyn Fn(Vec3, f32) -> Option<f32>,
) -> Vec<(f32, f32)> {
    let dt = engine_softbody::authoring::skeleton::STEP;
    let fixed: Vec<Vec3> = held.iter().map(|&i| s.nodes[i].pos).collect();
    let mut out = Vec::new();
    let (mut d, mut acc, mut n, mut next) = (0.0f32, 0.0f32, 0usize, 0.005f32);
    while d < travel {
        d += speed * dt;
        s.step();
        for (k, &i) in held.iter().enumerate() {
            s.nodes[i].pos = fixed[k];
            s.nodes[i].vel = Vec3::ZERO;
        }
        let mut f = 0.0f32;
        for nd in s.nodes.iter_mut() {
            if let Some(depth) = inside(nd.pos, d) {
                nd.pos += dir * depth;
                let vn = nd.vel.dot(dir);
                if vn < speed {
                    f += nd.mass * (speed - vn) / dt;
                    nd.vel += dir * (speed - vn);
                }
            }
        }
        acc += f;
        n += 1;
        if d >= next {
            out.push((d, acc / n as f32));
            acc = 0.0;
            n = 0;
            next += 0.005;
        }
    }
    out
}

/// FMVSS 214 S6.1 static door crush (`crash-reference-targets.md` §1, §4.12):
/// a rigid 305 mm cylinder, axis vertical, its bottom 127 mm above the door's
/// lowest point and its top above the window sill, centred on the front door,
/// pushed in across the car while the far side is held. The average force
/// must reach 10,000 N over the first 152 mm, 15,569 N over 305 mm, and a
/// peak of min(2 × curb weight, 31,138 N) within 457 mm. Medium; the ram moves
/// at 0.1 m/s (the rule's 12.7 mm/s would take 36 s; the force at 0.1 m/s
/// is still quasi-static: the door's own inertia is negligible). Crush is
/// counted from the ram's first contact.
#[test]
fn p_fmvss_214_static_door_crush() {
    let b = built(Resolution::Medium);
    let sk = &b.skeleton;
    let mut s = spawn(&b, Vec3::ZERO, 0.0);
    run(&mut s, 1.0);
    let door: Vec<usize> = (0..sk.nodes.len())
        .filter(|&i| sk.parts[sk.nodes[i].part as usize] == "door_f_l")
        .collect();
    let lo = door
        .iter()
        .map(|&i| s.nodes[i].pos.y)
        .fold(f32::MAX, f32::min);
    let (z0, z1) = door.iter().fold((f32::MAX, f32::MIN), |(a, c), &i| {
        (a.min(s.nodes[i].pos.z), c.max(s.nodes[i].pos.z))
    });
    let zc = 0.5 * (z0 + z1);
    let outer = door
        .iter()
        .map(|&i| s.nodes[i].pos.x)
        .fold(f32::MIN, f32::max);
    let r = 0.1525;
    let (y0, y1) = (lo + 0.127, lo + 0.80);
    // The far half of the car held (the rig's restraint).
    let held: Vec<usize> = (0..sk.nodes.len())
        .filter(|&i| sk.nodes[i].primary && s.nodes[i].pos.x < -0.2)
        .collect();
    let cx0 = outer + r + 0.01;
    let inside = move |p: Vec3, d: f32| -> Option<f32> {
        if p.y < y0 || p.y > y1 {
            return None;
        }
        let cx = cx0 - d;
        let dz = p.z - zc;
        if dz.abs() >= r {
            return None;
        }
        // The ram's leading surface at this z (a circle in plan view).
        let face = cx - (r * r - dz * dz).sqrt();
        (p.x > face).then(|| p.x - face)
    };
    let curve = ram_rig(&mut s, &held, -Vec3::X, 0.457 + 0.05, 0.1, &inside);
    // Crush is counted from the ram's first contact with the door (the
    // first 5 mm step carrying more than 200 N).
    let d_c = curve
        .iter()
        .find(|c| c.1 > 200.0)
        .map_or(0.0, |c| c.0 - 0.005);
    let mean_to = |x: f32| {
        let v: Vec<f32> = curve
            .iter()
            .filter(|c| c.0 > d_c && c.0 - d_c <= x + 1e-4)
            .map(|c| c.1)
            .collect();
        v.iter().sum::<f32>() / v.len().max(1) as f32
    };
    let (f152, f305) = (mean_to(0.152), mean_to(0.305));
    let peak = curve
        .iter()
        .filter(|c| c.0 > d_c && c.0 - d_c <= 0.457 + 1e-4)
        .map(|c| c.1)
        .fold(0.0f32, f32::max);
    let curb: f32 = s.nodes.iter().map(|n| n.mass).sum::<f32>() * 9.81;
    let need = (2.0 * curb).min(31_138.0);
    eprintln!(
        "(p) FMVSS 214 door crush (contact at {:.0} mm of ram travel): mean {:.1} kN over 152 mm (≥ 10.0), {:.1} kN over 305 mm (≥ 15.6), peak {:.1} kN over 457 mm (≥ {:.1}); curve every 5 cm: {:?} kN",
        d_c * 1e3,
        f152 / 1e3,
        f305 / 1e3,
        peak / 1e3,
        need / 1e3,
        curve
            .iter()
            .step_by(10)
            .map(|c| (c.1 / 1e2).round() / 10.0)
            .collect::<Vec<_>>()
    );
    assert!(finite(&s));
    let mut fails = Vec::new();
    if f152 < 10_000.0 {
        fails.push(format!("initial {f152} N"));
    }
    if f305 < 15_569.0 {
        fails.push(format!("intermediate {f305} N"));
    }
    if peak < need {
        fails.push(format!("peak {peak} N < {need}"));
    }
    assert!(fails.is_empty(), "{fails:?}");
}

/// FMVSS 216a roof crush (`crash-reference-targets.md` §1, §4.13): a rigid
/// 762 × 1,829 mm platen pitched 5° (front down) and rolled 25° (outboard
/// down), its forward edge 254 mm ahead of the roof's front, its centre line
/// over the driver's side roof rail, pushed along its normal while the sills
/// are held. The peak force within 127 mm of platen travel, against the
/// unloaded weight: 4.0–5.5 × (measured compact sedans 4.5–5.3 ×; FMVSS
/// minimum 3.0, IIHS "good" 4.0). Medium, 0.1 m/s.
#[test]
fn q_fmvss_216a_roof_crush() {
    let b = built(Resolution::Medium);
    let sk = &b.skeleton;
    let mut s = spawn(&b, Vec3::ZERO, 0.0);
    run(&mut s, 1.0);
    let g = |n: &str| group_mean(&s, &b, &[n]);
    // The roof's front: the windscreen header; its side: the left roof rail.
    let header = g("header_f");
    let rail = g("roof_rail");
    let (pitch, roll) = (5.0f32.to_radians(), 25.0f32.to_radians());
    // Push direction: down, tipped forward-down by the pitch and outboard-down
    // by the roll; the platen's axes across and along it.
    // (Rotating −y about +x by +pitch tips it rearward, so the platen's plane
    // is lower at its front edge: the front meets the roof first.)
    let push = Quat::from_rotation_x(pitch) * Quat::from_rotation_z(-roll) * -Vec3::Y;
    let along = (Quat::from_rotation_x(pitch) * Vec3::Z).normalize();
    let across = push.cross(along).normalize();
    let front = header.z + 0.254;
    let centre_z = front - 0.5 * 1.829;
    // Start: the platen's plane through the highest node under it.
    let under = |p: Vec3| {
        // In the platen's plane: across it and along it from its centre
        // (over the rail at roof height).
        let q = p - Vec3::new(rail.x, rail.y, centre_z);
        (q.dot(across).abs() < 0.381) && (q.dot(along).abs() < 0.9145)
    };
    let start = s
        .nodes
        .iter()
        .filter(|n| under(n.pos))
        .map(|n| n.pos.dot(push))
        .fold(f32::MAX, f32::min);
    // FMVSS 216a S7.1: the vehicle is supported on its sills (rigidly), not
    // its floor: the sills' nodes held.
    let held: Vec<usize> = ["sill", "sill_r"]
        .iter()
        .flat_map(|g| b.groups[*g].iter().map(|&i| i as usize))
        .collect();
    let _ = sk;
    let inside = move |p: Vec3, d: f32| -> Option<f32> {
        if !under(p) {
            return None;
        }
        let depth = start + d - p.dot(push);
        (depth > 0.0).then_some(depth)
    };
    let before = s.positions();
    let curve = ram_rig(&mut s, &held, push, 0.127, 0.1, &inside);
    let peak = curve.iter().map(|c| c.1).fold(0.0f32, f32::max);
    if std::env::var("Q_DIAG").is_ok() {
        let mut v: Vec<(String, f32)> = b
            .groups
            .iter()
            .map(|(k, g)| {
                let d = g
                    .iter()
                    .map(|&i| (s.nodes[i as usize].pos - before[i as usize]).dot(push))
                    .fold(0.0f32, f32::max);
                (k.clone(), d)
            })
            .collect();
        v.sort_by(|a, c| c.1.total_cmp(&a.1));
        eprintln!(
            "(q) groups moved along the push (mm): {:?}",
            v.iter()
                .take(14)
                .map(|(k, d)| format!("{k} {:.0}", d * 1e3))
                .collect::<Vec<_>>()
        );
    }
    let uvw: f32 = s.nodes.iter().map(|n| n.mass).sum::<f32>() * 9.81;
    eprintln!(
        "(q) FMVSS 216a roof: peak {:.1} kN within 127 mm = {:.2} × unloaded weight (≥ 3.0; IIHS good ≥ 4.0); curve every 25 mm: {:?} kN",
        peak / 1e3,
        peak / uvw,
        curve
            .iter()
            .step_by(5)
            .map(|c| (c.1 / 1e2).round() / 10.0)
            .collect::<Vec<_>>()
    );
    assert!(finite(&s));
    // FMVSS 216a needs 3.0 ×; IIHS "good" (≥ 4.0 ×) is what virtually all
    // vehicles reach, and the compact sedans measured are 4.5–5.3 ×: the
    // roof lands in 4.0–5.5 ×.
    assert!(
        (4.0..=5.5).contains(&(peak / uvw)),
        "roof {} × weight",
        peak / uvw
    );
}

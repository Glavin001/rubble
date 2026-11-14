//! A body dashpot (k = 0: the damping of the material between two nodes of a part,
//! `authoring::vehicle` "Body damping") must not outlive that material. Reported
//! 2026-11-14: torn-off pieces hover and circle near the car instead of flying off.
//! A node whose every spring has broken was still held by its dashpot (50 N·s/m to a
//! node of its part 1–2 m away, letting go only above 5 kN: 100 m/s of relative speed).
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, ExplicitKind};
use glam::Vec3;

/// A skin node of the showcase sedan with every spring on it broken, moved 3 m out
/// from the car and leaving at 5 m/s: nothing joins it to the car, so it keeps its
/// speed (no drag here; 0.2 s of gravity cannot change its horizontal speed) and its
/// dashpot has torn.
#[test]
fn a_torn_off_node_is_not_held_by_its_dashpot() {
    let b = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Medium, strength: 1.0, paint: None }).unwrap();
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings.clone()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut s = ExplicitBeams::from_truck(&b.truck, Vec3::new(0.0, probe.spawn_clearance() - low, 0.0), Vec3::ZERO, settings).unwrap();
    let damper = |bm: &engine_softbody::explicit_beams::Beam| bm.k == 0.0 && bm.c > 0.0 && bm.kind == ExplicitKind::Plain;
    let dampers = s.beams.iter().filter(|bm| damper(bm)).count();
    // The lightest node that has a dashpot of its own (a skin node).
    let i = (0..s.nodes.len())
        .filter(|&i| s.incidence[i].iter().any(|&j| damper(&s.beams[j as usize])))
        .min_by(|&a, &c| s.nodes[a].mass.total_cmp(&s.nodes[c].mass))
        .expect("the car has body dashpots");
    let held: Vec<u32> = s.incidence[i].iter().copied().filter(|&j| damper(&s.beams[j as usize])).collect();
    for &j in &s.incidence[i].clone() {
        if !damper(&s.beams[j as usize]) {
            s.beams[j as usize].broken = true;
        }
    }
    let centre = s.nodes.iter().map(|n| n.pos).sum::<Vec3>() / s.nodes.len() as f32;
    let out = ((s.nodes[i].pos - centre) * Vec3::new(1.0, 0.0, 1.0)).normalize();
    s.nodes[i].pos += out * 3.0;
    s.nodes[i].vel = out * 5.0;
    eprintln!("{dampers} dashpots of {} beams; node {i} ({:.3} kg) on {} dashpots, rest {:?} m", s.beams.len(), s.nodes[i].mass, held.len(), held.iter().map(|&j| s.beams[j as usize].rest).collect::<Vec<_>>());
    for _ in 0..400 {
        s.step();
    }
    let speed = s.nodes[i].vel.dot(out);
    eprintln!("after 0.2 s: {speed:.2} m/s outward; dashpots broken {:?}", held.iter().map(|&j| s.beams[j as usize].broken).collect::<Vec<_>>());
    assert!(held.iter().all(|&j| s.beams[j as usize].broken), "a dashpot still joins the torn-off node to the car");
    assert!(speed > 4.9, "the torn-off node was slowed to {speed:.2} m/s by the car");
}

/// Crushing never tears a dashpot (two points of a sheet pushed together), and a
/// dashpot at rest stays: the car standing for a second keeps every one.
#[test]
fn dashpots_stay_at_rest_and_under_compression() {
    let b = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Medium, strength: 1.0, paint: None }).unwrap();
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut s = ExplicitBeams::from_truck(&b.truck, Vec3::new(0.0, probe.spawn_clearance() - low, 0.0), Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap();
    for _ in 0..2000 {
        s.step();
    }
    let damper = |bm: &engine_softbody::explicit_beams::Beam| bm.k == 0.0 && bm.c > 0.0 && bm.kind == ExplicitKind::Plain;
    assert_eq!(s.beams.iter().filter(|bm| damper(bm) && bm.broken).count(), 0);
    // One dashpot's ends pushed to a tenth of its length: it stays.
    let j = s.beams.iter().position(|bm| damper(bm)).unwrap();
    let (a, c) = (s.beams[j].a as usize, s.beams[j].b as usize);
    let d = s.nodes[c].pos - s.nodes[a].pos;
    s.nodes[a].pos += d * 0.9;
    s.step();
    assert!(!s.beams[j].broken, "a compressed dashpot tore");
}

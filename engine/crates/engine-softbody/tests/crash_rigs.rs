//! The crash lab's rigs (`engine_softbody::crash_rigs`): the FMVSS 214 moving
//! deformable barrier's mass, centre of mass and crush force, and the door pull.
use engine_softbody::crash_rigs::{
    add_door_pull, door_nodes, mdb214_face_crush_force, mdb214_jbeam, MDB_FACE, MDB_MASS,
};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, Grab, JbeamWheels, StaticBox};
use engine_softbody::jbeam::load_jbeam;
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() }
}

fn mdb(v: Vec3) -> ExplicitBeams {
    let text = mdb214_jbeam();
    let desc = load_jbeam(&[("mdb.jbeam", &text)], None).unwrap();
    let tyres = engine_softbody::truck_export::TyreDefaults::for_vehicle(&desc);
    let (s, w) = ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Soft(tyres), Vec3::ZERO, v, settings()).unwrap();
    for w in &w {
        eprintln!("warning: {w}");
    }
    s
}

fn com(s: &ExplicitBeams) -> (Vec3, Vec3, f32) {
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let p = s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / m;
    let v = s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
    (p, v, m)
}

#[test]
fn mdb_has_the_standards_mass_and_centre_of_mass() {
    let s = mdb(Vec3::ZERO);
    let (p, _, m) = com(&s);
    assert!((m - MDB_MASS).abs() < 0.5, "mass {m}");
    // 19.7 in up (587.6(d)), within 1 cm.
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    assert!(((p.y - low + 0.02) - 19.7 * 0.0254).abs() < 0.01, "height {}", p.y - low + 0.02);
    // Face: 1,676 mm wide.
    let xs = s.nodes.iter().map(|n| n.pos.x);
    let w = xs.clone().fold(f32::MIN, f32::max) - xs.fold(f32::MAX, f32::min);
    assert!(w >= MDB_FACE[0] - 1e-3, "width {w}");
}

/// The model's crush force: its front skin held still (a perfectly flat, rigid wall
/// that its nodes cannot bounce off) while the carriage runs into it at 8 m/s. Exact
/// physics bounds it: every column yields at its crush stress times its area and
/// none hardens, so the face never carries more than the main block's stress on the
/// whole face (310 kPa × 1.676 m × 0.559 m = 290 kN; the bumper, stronger, is in
/// series in front of it), plus 5 % (the yield rule's half-step overshoot). It
/// carries less while columns unload: the carriage is not rigid at the 2 kHz step
/// (its back-plate nodes, a few kg each on stiff frame beams, ring against the frame), and
/// the mean over the crush is 101 % of σ·A (measured 2026-11-05 with the stiffened
/// carriage and its shear diagonals, 72 % before; the check keeps it above 85 %).
#[test]
fn mdb_crushes_at_the_honeycombs_stress() {
    let v = 8.0;
    let mut s = mdb(Vec3::ZERO);
    for n in &mut s.nodes {
        n.vel = Vec3::new(0.0, 0.0, v);
    }
    let spawn_pos: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
    let mut skin: Vec<u32> = s.triangles.iter().flat_map(|t| t.nodes).collect();
    skin.sort_unstable();
    skin.dedup();
    let held: Vec<(u32, Vec3)> = skin.iter().map(|&i| (i, s.nodes[i as usize].pos)).collect();
    let moving = |s: &ExplicitBeams| -> f32 {
        s.nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| !skin.contains(&(*i as u32)))
            .map(|(_, n)| n.mass * n.vel.z)
            .sum()
    };
    let p0 = moving(&s);
    let mut steps = 0;
    let mut peak = 0.0f32;
    let mut last = p0;
    while moving(&s) > 0.05 * p0 && steps < 400 {
        for &(i, p) in &held {
            s.nodes[i as usize].pos = p;
            s.nodes[i as usize].vel = Vec3::ZERO;
        }
        s.step();
        steps += 1;
        let p = moving(&s);
        if steps % 10 == 0 {
            peak = peak.max((last - p) / (10.0 * 0.0005));
            last = p;
        }
    }
    let f = mdb214_face_crush_force();
    let mean = (p0 - moving(&s)) / (steps as f32 * 0.0005);
    // Barrier integrity: the carriage (back plate to rear frame) stays rigid while
    // the face crushes.
    let (bend, len) = carriage_integrity(&s, &spawn_pos);
    eprintln!("carriage: length change {:.2} %, back plate out of plane {:.1} mm", 100.0 * len, 1e3 * bend);
    // Guards (the explicit carriage is not rigid at the step: its plate bends a few
    // cm under the 290 kN of the face, small against the 381 mm of honeycomb).
    assert!(len.abs() < 0.02 && bend < 0.05, "carriage deforms: {len} {bend}");
    eprintln!("crush over {steps} steps: mean {mean:.0} N, peak (10-step) {peak:.0} N; σ·A {f:.0} N");
    assert!(mean <= 1.05 * f && peak <= 1.2 * f, "above the crush stress: mean {mean}, peak {peak}");
    assert!(mean >= 0.85 * f, "mean crush force {mean} N of {f} N");
}

#[test]
fn door_pull_rig_pulls_several_nodes() {
    let text = mdb214_jbeam();
    let desc = load_jbeam(&[("mdb.jbeam", &text)], None).unwrap();
    let tyres = engine_softbody::truck_export::TyreDefaults::for_vehicle(&desc);
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Soft(tyres), Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let side = Vec3::new(1.0, 0.0, 0.0);
    let at = Vec3::new(1.0, 0.5, 0.0);
    let door = door_nodes(&s, side, at, 4);
    assert_eq!(door.len(), 4);
    let before: Vec<Vec3> = door.iter().map(|&d| s.nodes[d as usize].pos).collect();
    let rig = add_door_pull(&mut s, &door, at + side * 0.3);
    assert_eq!(s.incidence.len(), s.nodes.len());
    let target = s.nodes[rig as usize].pos + side * 0.2;
    s.grab = Some(Grab { node: rig, target, k: 50_000.0, c: 2.0 * (50_000.0f32 * 5.0).sqrt() });
    s.run(0.2);
    for (&d, b) in door.iter().zip(&before) {
        let moved = (s.nodes[d as usize].pos - *b).dot(side);
        assert!(moved > 0.0, "door node {d} pulled out: {moved}");
    }
    assert!(s.nodes.iter().all(|n| n.pos.is_finite()));
}

/// The MDB carriage's deformation against its spawn shape: (the back plate's
/// out-of-plane spread, m: the largest deviation of its nodes' displacement along
/// the barrier's axis from their mean; the change of the back plate to rear frame
/// distance, a fraction). The back plate's nodes are the 5th of every 4 + bumper
/// node rows… found by their spawn place: 0.381 m behind the face, and the rear
/// frame's the farthest back.
pub fn carriage_integrity(s: &ExplicitBeams, spawn: &[Vec3]) -> (f32, f32) {
    let axis = Vec3::Z;
    let depth = MDB_FACE[3];
    let front = spawn.iter().map(|p| p.dot(axis)).fold(f32::MIN, f32::max) - MDB_FACE[4];
    let plate: Vec<usize> = (0..spawn.len()).filter(|&i| (front - spawn[i].dot(axis) - depth).abs() < 1e-3).collect();
    let back = spawn.iter().map(|p| p.dot(axis)).fold(f32::MAX, f32::min);
    let rear: Vec<usize> = (0..spawn.len()).filter(|&i| (spawn[i].dot(axis) - back).abs() < 1e-3).collect();
    let mean = |ids: &[usize], p: &dyn Fn(usize) -> Vec3| ids.iter().map(|&i| p(i)).sum::<Vec3>() / ids.len() as f32;
    let now = |i: usize| s.nodes[i].pos;
    let then = |i: usize| spawn[i];
    let d: Vec<f32> = plate.iter().map(|&i| (now(i) - then(i)).dot(axis)).collect();
    let dm = d.iter().sum::<f32>() / d.len() as f32;
    let bend = d.iter().map(|x| (x - dm).abs()).fold(0.0, f32::max);
    let l0 = (mean(&plate, &then) - mean(&rear, &then)).length();
    let l1 = (mean(&plate, &now) - mean(&rear, &now)).length();
    (bend, l1 / l0 - 1.0)
}

/// A node flung to infinity never hangs the self-collision grid (spec §10): its
/// triangles' boxes count their cells in float and take every candidate, so the
/// step returns (the blow-up is the caller's to see in the positions).
#[test]
fn a_non_finite_node_does_not_hang_self_collision() {
    use engine_softbody::authoring::{build, style, BuildParams, Resolution};
    let truck = build(
        &style("sedan").unwrap(),
        &BuildParams { resolution: Resolution::parse("medium").unwrap(), strength: 1.0, paint: None },
    )
    .unwrap()
    .truck;
    let mut s = ExplicitBeams::from_truck(&truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    s.opt_in_full_contact();
    assert!(s.grid_cell > 0.0, "the self-collision grid is in use ({} candidates)", s.candidates.len());
    let t = s.triangles[0].nodes[0] as usize;
    s.nodes[t].pos = Vec3::new(f32::INFINITY, 0.0, f32::NEG_INFINITY);
    let start = std::time::Instant::now();
    for _ in 0..4 {
        s.step();
    }
    assert!(start.elapsed().as_secs_f32() < 10.0, "self-collision with a non-finite node took {:?}", start.elapsed());
}

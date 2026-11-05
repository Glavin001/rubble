//! The contact-force log and `rubble_contact_breaks` (jbeam-semantics-spec.md
//! §32), from their definitions: a resting node's contact force is its weight; a
//! drop's impact sets off its break group in the next step's beam pass (forward
//! only), and not below the line's force; the `.truck` lines are read and checked.

use engine_softbody::explicit_beams::*;
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
};

/// Two 1 kg nodes 1 m apart at height `y` over the ground (y = 0), joined by a
/// strong beam of break group 1.
fn pane(y: f32) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = 0.0;
    for x in [0.0, 1.0] {
        s.nodes.push(Node {
            pos: Vec3::new(x, y, 0.0),
            vel: Vec3::ZERO,
            force: Vec3::ZERO,
            mass: 1.0,
            friction: 1.0,
            flags: 0,
        });
    }
    let mut b = Beam::new(
        0,
        1,
        ExplicitKind::Plain,
        ExplicitType::Deformable,
        1.0e5,
        100.0,
        1.0,
        1.0e9,
        1.0e9,
        0.0,
    );
    b.group = 1;
    s.beams.push(b);
    s.finish();
    s
}

#[test]
fn a_resting_node_s_contact_force_is_its_weight() {
    let mut s = pane(0.01);
    s.enable_contact_force();
    for _ in 0..4000 {
        s.step();
    }
    // The contact chatters step to step; its mean over a second carries the weight.
    let mut mean = [0.0f64; 2];
    for _ in 0..2000 {
        s.step();
        for (m, f) in mean.iter_mut().zip(s.contact_forces().unwrap()) {
            *m += f64::from(f) / 2000.0;
        }
    }
    for (i, f) in mean.iter().enumerate() {
        assert!((f - 9.81).abs() < 0.05, "node {i}: {f} N");
    }
    let last = s.contact_forces().unwrap();
    let peak = s.take_contact_force_peak().unwrap();
    assert!(peak[0] >= last[0]);
    assert_eq!(s.take_contact_force_peak().unwrap(), vec![0.0, 0.0]);
}

/// The steps until the beam broke, and the first step whose contact force on node
/// 0 exceeded `limit`.
fn drop(limit: f32) -> (Option<u32>, Option<u32>) {
    let mut s = pane(0.5);
    s.set_contact_breaks(&[(1, limit, vec![0])]).unwrap();
    let (mut broke, mut over) = (None, None);
    for k in 1..=1000u32 {
        s.step();
        // The step's sum is complete after its own step (the next node pass moves
        // it to `last`): read it through a probe clone stepped once.
        if over.is_none() {
            let mut probe = s.clone();
            probe.step();
            if probe.contact_forces().unwrap()[0] > limit {
                over = Some(k);
            }
        }
        if broke.is_none() && s.beams[0].broken {
            broke = Some(k);
        }
    }
    (broke, over)
}

#[test]
fn an_impact_sets_off_its_group_in_the_next_step_and_not_below_its_force() {
    // A 0.5 m drop: about 3.1 m/s into the ground.
    let (broke, over) = drop(50.0);
    let (b, o) = (broke.expect("never broke"), over.expect("never over 50 N"));
    assert_eq!(b, o + 1, "broke at step {b}, the force passed at {o}");
    // Far above the impact: never.
    assert_eq!(drop(1.0e7), (None, None));
}

#[test]
fn truck_lines_are_read_and_checked() {
    let text = car_truck_bare(&CarParams::sedan());
    // (Before the closing `end`.)
    let body = text.trim_end().strip_suffix("end").unwrap_or(&text);
    let bad = format!("{body}\nrubble_contact_breaks\n999, 100, 0, 1\nend\n");
    let err = ExplicitBeams::from_truck(&bad, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap_err();
    assert!(err.contains("group 999"), "{err}");
    let plain = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    assert!(plain.contact_force.is_none());
    // A real group (the file's detacher group 7): read onto its nodes.
    let good = bad.replace("999, 100, 0, 1", "7, 100, 0, 1");
    let s = ExplicitBeams::from_truck(&good, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let log = s.contact_force.as_ref().unwrap();
    assert_eq!(&log.node_break[..3], &[(7, 100.0), (7, 100.0), (0, 0.0)]);
    let twice = bad.replace("999, 100, 0, 1", "7, 100, 0, 1\n8, 50, 1");
    let err = ExplicitBeams::from_truck(&twice, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap_err();
    assert!(err.contains("node 1"), "{err}");
}

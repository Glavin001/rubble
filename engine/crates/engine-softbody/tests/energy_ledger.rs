//! The energy ledger of `ExplicitBeamsWorld` (CPU): it closes (the kinetic
//! energy's change is the sum of every source's work less the step's
//! discretisation term), the vehicles' mechanical energy (kinetic, gravitational,
//! elastic) never rises, so no contact injects energy, and the beams' dissipation
//! is never negative and never more than the energy lost, in two standard crashes: two procedural sedans rear-ending
//! at 15 m/s, and the authored showcase car striking another 80 km/h off the door
//! middle (the T-bone whose per-beam tally exceeded the striker's energy).
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_contact::{EnergyLedger, ExplicitBeamsWorld};
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    }
}

fn report(label: &str, l: &EnergyLedger) {
    eprintln!(
        "{label}: KE {:.1} kJ, gravity PE {:.1} kJ, elastic {:.2} kJ (start {:.1} kJ mechanical); work: beams {:.1}, \
         self-contact {:.2}, vehicle contact normal {:.2} friction {:.2}, ground vertical {:.2} horizontal {:.2}, \
         air {:.2}, gravity {:.2}, slides {:.2} kJ; discretisation {:.3} kJ; beam dissipation {:.1} kJ",
        l.kinetic * 1e-3,
        l.gravity_pe * 1e-3,
        l.elastic * 1e-3,
        l.start * 1e-3,
        l.w_beams * 1e-3,
        l.w_self_contact * 1e-3,
        l.w_contact_normal * 1e-3,
        l.w_contact_friction * 1e-3,
        l.w_ground_vertical * 1e-3,
        l.w_ground_horizontal * 1e-3,
        l.w_air * 1e-3,
        l.w_gravity * 1e-3,
        l.w_slides * 1e-3,
        l.discretisation * 1e-3,
        l.beam_dissipation() * 1e-3,
    );
}

/// Runs `w` for `frames` 33-step frames with the ledger and checks it every frame.
/// Tolerances: rounding of f32 states summed in f64 (closure, 0.1 % of the start
/// energy) and the symplectic step's bounded energy error (0.5 %).
fn check(label: &str, mut w: ExplicitBeamsWorld, frames: usize) -> EnergyLedger {
    w.enable_ledger();
    let ke0 = w.ledger.as_ref().unwrap().kinetic;
    let e0 = w.ledger.as_ref().unwrap().start;
    let mut most = e0;
    for f in 0..frames {
        w.frame(33);
        let l = w.ledger.as_ref().unwrap();
        let closure = (l.kinetic - ke0) - l.kinetic_change();
        assert!(
            closure.abs() < 1e-3 * e0,
            "{label} frame {f}: ledger off by {closure} J"
        );
        most = most.max(l.mechanical());
        // (A contact's work alone may be positive: its projection cancels the
        // contacter's beam force, so it passes beam energy on; the sum is what
        // must not grow.)
    }
    let l = w.ledger.clone().unwrap();
    report(label, &l);
    assert!(
        most <= e0 * 1.005,
        "{label}: mechanical energy rose from {e0} J to {most} J"
    );
    assert!(
        l.beam_dissipation() > -5e-3 * e0,
        "{label}: beams released {} J",
        -l.beam_dissipation()
    );
    l
}

#[test]
fn ledger_closes_in_a_sedan_rear_end() {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let (z0, z1) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
        (a.min(n.pos.z), b.max(n.pos.z))
    });
    let v = Vec3::new(0.0, 0.0, 15.0);
    let mut rear =
        ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), v, settings()).unwrap();
    rear.roll_tyres(v);
    let front = ExplicitBeams::from_truck(
        &text,
        Vec3::new(0.0, 0.01 - low, z1 - z0 + 0.5),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    check(
        "sedan rear-end 15 m/s",
        ExplicitBeamsWorld::new(vec![front, rear]),
        60,
    );
}

/// The showcase car into another's front door, 80 km/h, 0.7 m forward of the
/// door's middle (`showcase_car.rs` test u, `TB_DZ=0.7`), with two-body contact
/// (jbeam-semantics-spec.md §10.3) on both cars. With RoR's rule (the `.truck`
/// default) the same crash's contacts inject energy: reported, not asserted (open:
/// the rules are to be unified once the T-bone leak is fixed).
#[test]
fn ledger_closes_in_the_showcase_t_bone_off_the_door_middle() {
    let b = build(
        &style("sedan").unwrap(),
        &BuildParams {
            resolution: Resolution::Medium,
            ..Default::default()
        },
    )
    .unwrap();
    let sk0 = &b.skeleton;
    let len = b.def.profile.length;
    let door_part = sk0.parts.iter().position(|p| p == "door_f_l").unwrap() as u16;
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let struck = ExplicitBeams::from_truck(
        &b.truck,
        Vec3::new(0.0, 0.02 - low, 0.0),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    let p0 = struck.positions();
    let door: Vec<usize> = (0..sk0.nodes.len())
        .filter(|&i| sk0.nodes[i].part == door_part)
        .collect();
    let zd = door.iter().map(|&i| p0[i].z).sum::<f32>() / door.len() as f32;
    let hw = p0.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let mut sk = sk0.clone();
    for nd in &mut sk.nodes {
        let q = nd.p - Vec3::new(0.0, 0.0, 0.5 * len);
        nd.p = Vec3::new(-q.z, q.y, q.x);
    }
    let truck = sk.to_truck();
    let pr = ExplicitBeams::from_truck(&truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let pp = pr.positions();
    let (lo, near) = pp
        .iter()
        .fold((f32::MAX, f32::MAX), |(l, x), p| (l.min(p.y), x.min(p.x)));
    let low0 = p0.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let v = Vec3::new(-80.0 / 3.6, 0.0, 0.0);
    let mut striker = ExplicitBeams::from_truck(
        &truck,
        Vec3::new(hw + 0.05 - near, low0 - lo, zd + 0.7),
        v,
        settings(),
    )
    .unwrap();
    striker.roll_tyres(v);
    let ke_striker: f64 = striker
        .nodes
        .iter()
        .map(|n| 0.5 * f64::from(n.mass) * f64::from(n.vel.length_squared()))
        .sum();
    eprintln!(
        "striker's kinetic energy at the start {:.1} kJ",
        ke_striker * 1e-3
    );
    // RoR's rule: reported.
    let mut w = ExplicitBeamsWorld::new(vec![struck.clone(), striker.clone()]);
    w.enable_ledger();
    for _ in 0..60 {
        w.frame(33);
    }
    report("OPEN, RoR's contact rule", w.ledger.as_ref().unwrap());
    let (mut struck, mut striker) = (struck, striker);
    struck.two_body_contact = true;
    striker.two_body_contact = true;
    let l = check(
        "two-body contact",
        ExplicitBeamsWorld::new(vec![struck, striker]),
        60,
    );
    // The beams can dissipate no more than the vehicles lost.
    assert!(
        l.beam_dissipation() <= l.start - l.mechanical() + 5e-3 * l.start,
        "beams {} J of {} J lost",
        l.beam_dissipation(),
        l.start - l.mechanical()
    );
}

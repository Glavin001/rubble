//! The pressure wheels' parking brake (`engine/docs/jbeam-semantics-spec.md`
//! §25.1), from its definition and physical invariants on the Part II car.
use engine_softbody::explicit_beams::{
    brake_apply, ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, STEP,
};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::PART2_CAR;
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    }
}

/// The Part II car with `parkingTorque` on its rear wheels (or none).
fn car(parking: Option<f32>, v: Vec3) -> ExplicitBeams {
    let text = match parking {
        Some(t) => PART2_CAR.replace(
            r#"["RL", "wheel_RL""#,
            &format!(r#"{{"parkingTorque": {t}}}, ["RL", "wheel_RL""#),
        ),
        None => PART2_CAR.to_string(),
    };
    let desc = load_jbeam(&[("t.jbeam", &text)], None).unwrap();
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
        &desc,
        JbeamWheels::Pressure,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    let low = s
        .nodes
        .iter()
        .map(|n| s.origin.y + n.pos.y)
        .fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.01 - low;
    }
    if v != Vec3::ZERO {
        for n in &mut s.nodes {
            n.vel = v;
        }
        s.roll_tyres(v);
    }
    s
}

fn kinetic(s: &ExplicitBeams) -> f32 {
    s.nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum()
}

#[test]
fn rows_with_parking_torque_get_brakes() {
    assert!(car(None, Vec3::ZERO).brakes.is_empty());
    let s = car(Some(800.0), Vec3::ZERO);
    // The modifier holds for the rows after it: RL and RR.
    assert_eq!(s.brakes.len(), 2);
    for b in &s.brakes {
        assert_eq!(b.torque, 800.0);
        assert_eq!(b.count, 24, "two hub rings of 12 rays");
    }
}

#[test]
fn released_brake_is_bit_identical_to_no_brake() {
    let mut a = car(None, Vec3::new(0.0, 0.0, 1.0));
    let mut b = car(Some(800.0), Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(b.parking_brake, 0.0, "released by default");
    for _ in 0..2000 {
        a.step();
        b.step();
    }
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!(x.pos.to_array(), y.pos.to_array());
        assert_eq!(x.vel.to_array(), y.vel.to_array());
    }
}

#[test]
fn brake_forces_are_a_pure_torque_pair() {
    let mut s = car(Some(800.0), Vec3::ZERO);
    // Spin the left rear disc about its axle.
    let b = s.brakes[0];
    let (p1, p2) = (
        s.nodes[b.axle[0] as usize].pos,
        s.nodes[b.axle[1] as usize].pos,
    );
    let a = (p2 - p1).normalize();
    let c = 0.5 * (p1 + p2);
    for i in b.first..b.first + b.count {
        let n = &mut s.nodes[i as usize];
        n.vel = a.cross(n.pos - c) * 3.0;
    }
    for n in &mut s.nodes {
        n.force = Vec3::ZERO;
    }
    brake_apply(&mut s.nodes, &b, 800.0, STEP);
    let net: Vec3 = s.nodes.iter().map(|n| n.force).sum();
    let torque = |ids: &mut dyn Iterator<Item = u32>| -> f32 {
        ids.map(|i| {
            let n = &s.nodes[i as usize];
            a.dot((n.pos - c).cross(n.force))
        })
        .sum()
    };
    let disc = torque(&mut (b.first..b.first + b.count));
    let arm = torque(&mut std::iter::once(b.arm));
    let axle = torque(&mut b.axle.into_iter());
    println!("net force {net:?}, disc torque {disc}, arm {arm}, axle {axle}");
    assert!(net.length() < 1e-2, "no net force: {net:?}");
    assert!(disc < 0.0, "opposes the spin");
    assert!(disc.abs() <= 800.0 * 1.0001, "at most the parking torque");
    assert!(
        (disc + arm).abs() < 1e-2 * disc.abs(),
        "the caliper takes the reaction"
    );
    assert!(
        axle.abs() < 1e-3 * disc.abs(),
        "the axle nodes take no torque"
    );
}

#[test]
fn applied_brake_stops_a_rolling_car() {
    let v = Vec3::new(0.0, 0.0, 1.0);
    let run = |park: f32| {
        let mut s = car(Some(800.0), v);
        s.parking_brake = park;
        let z0 = s.nodes[0].pos.z;
        s.run(3.0);
        (kinetic(&s), s.nodes[0].pos.z - z0)
    };
    let (ke_free, z_free) = run(0.0);
    let (ke_held, z_held) = run(1.0);
    println!("free: KE {ke_free:.3} J, travel {z_free:.3} m; held: KE {ke_held:.3} J, travel {z_held:.3} m");
    assert!(z_held < 0.5 * z_free, "the brake shortens the roll");
    assert!(ke_held < 0.1 * ke_free.max(1e-3), "and stops it");
}

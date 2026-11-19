//! Raycast wheels on rigid bodies (`DriveConfig::see_bodies`, known-bugs F7): the rays
//! meet the vehicle's rigid proxies and the wheels push back on them.
//!
//! * A car parked on a rigid box platform drives off it (with the setting off its wheels
//!   hang and it stays put).
//! * A car on a free-floating crate drives forward and pushes the crate back: the
//!   horizontal momentum of car and crate together is conserved (no ground, no other
//!   horizontal force) to the coupling's fixed-point rounding.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use engine_softbody::procedural::{car_truck, CarParams};
use glam::{Quat, Vec3};

fn sedan() -> String {
    car_truck(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    })
}

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    }
}

/// The raycast sedan with its lowest point `lift` m above y = 0 (its own ground far down
/// unless `ground` is given), wheels seeing bodies when `see`.
fn car(lift: f32, ground: Option<f32>, see: bool) -> ExplicitBeams {
    let (mut s, _) = ExplicitBeams::from_truck_raycast(&sedan(), None, Vec3::ZERO, 0.0, Vec3::ZERO, settings())
        .expect("raycast sedan");
    let lo = s.nodes.iter().map(|n| s.origin.y + n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += lift - lo;
    }
    s.ground = ground.unwrap_or(-1.0e4);
    s.drive.as_mut().unwrap().config.see_bodies = see;
    s.finish();
    s
}

fn forward(s: &ExplicitBeams) -> Vec3 {
    let [c, b, _] = s.drive.as_ref().unwrap().config.refs;
    let f = s.nodes[c as usize].pos - s.nodes[b as usize].pos;
    Vec3::new(f.x, 0.0, f.z).normalize()
}

fn com(s: &ExplicitBeams) -> (Vec3, Vec3, f32) {
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let x = s.nodes.iter().map(|n| (s.origin + n.pos) * n.mass).sum::<Vec3>() / m;
    let v = s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
    (x, v, m)
}

/// A box body of `half` extents centred at `c` (the car's footprint along `fw`), mass
/// `mass` (None: infinite, a static or anchored body).
fn platform(c: Vec3, half: Vec3, fw: Vec3, mass: Option<f32>) -> ProxySet {
    let yaw = fw.x.atan2(fw.z);
    let (inv_mass, inv_inertia) = match mass {
        Some(m) => {
            let i = Vec3::new(
                m / 3.0 * (half.y * half.y + half.z * half.z),
                m / 3.0 * (half.x * half.x + half.z * half.z),
                m / 3.0 * (half.x * half.x + half.y * half.y),
            );
            (1.0 / m, Vec3::ONE / i)
        }
        None => (0.0, Vec3::ZERO),
    };
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 1,
            pos: c,
            rot: Quat::from_rotation_y(yaw),
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass,
            inv_inertia,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(0, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Box { half }, 1, None)],
        frozen: false,
    };
    set.begin_window();
    set
}

const THROTTLE: DriveInput = DriveInput {
    forward: 1.0,
    steer: 0.0,
    brake: 0.0,
    handbrake: 0.0,
};

/// The car parked on a 6 m static platform 0.5 m high (its own ground at 0), settled
/// 1 s, then 4 s at full throttle: how far it went along its forward (m), and the
/// drop of its centre of mass from where it settled to its lowest (m).
fn drive_off(see: bool) -> (f32, f32) {
    let mut s = car(0.55, Some(0.0), see);
    let fw = forward(&s);
    let (x0, _, _) = com(&s);
    let c = Vec3::new(x0.x, 0.25, x0.z);
    s.rigid = Some(Box::new(platform(c, Vec3::new(1.2, 0.25, 3.0), fw, None)));
    s.run(1.0);
    let (x1, _, _) = com(&s);
    s.drive.as_mut().unwrap().input = THROTTLE;
    let mut low = f32::MAX;
    for _ in 0..240 {
        s.frame();
        low = low.min(com(&s).0.y);
    }
    let (x2, _, _) = com(&s);
    eprintln!("see {see}: settled at y {:.3}, went {:.2} m, lowest centre {:.3}", x1.y, (x2 - x1).dot(fw), low);
    ((x2 - x1).dot(fw), x1.y - low)
}

#[test]
fn a_car_on_a_rigid_platform_drives_off_it() {
    let (went, drop) = drive_off(true);
    // Off the 3 m half length and down the 0.5 m onto the ground below.
    assert!(went > 3.5, "drove {went} m");
    assert!(drop > 0.4, "never came down off the platform (dropped {drop} m)");
    let (stuck, _) = drive_off(false);
    assert!(stuck.abs() < 0.5, "without the setting the wheels hang: moved {stuck} m");
}

#[test]
fn a_wheel_on_a_crate_pushes_it_and_momentum_is_conserved() {
    // No ground: the car rests on a free-floating 2 t crate; both fall, nothing outside
    // acts horizontally.
    let mut s = car(0.55, None, true);
    let fw = forward(&s);
    let (x0, _, m) = com(&s);
    let crate_m = 2000.0;
    s.rigid = Some(Box::new(platform(Vec3::new(x0.x, 0.25, x0.z), Vec3::new(1.2, 0.25, 3.0), fw, Some(crate_m))));
    let horiz = |s: &ExplicitBeams| {
        let car: Vec3 = s.nodes.iter().map(|n| n.vel * n.mass).sum();
        let b = &s.rigid.as_ref().unwrap().bodies[0];
        let p = car + b.vel * crate_m;
        (Vec3::new(p.x, 0.0, p.z), Vec3::new(b.vel.x, 0.0, b.vel.z))
    };
    // Settle onto the crate (both falling), then throttle.
    for _ in 0..30 {
        s.frame();
    }
    let (p0, _) = horiz(&s);
    s.drive.as_mut().unwrap().input = THROTTLE;
    let mut contact = 0;
    for _ in 0..30 {
        s.frame();
        contact += s.drive.as_ref().unwrap().wheels.iter().filter(|w| w.contact).count();
    }
    let (p1, vc) = horiz(&s);
    let (_, vcar, _) = com(&s);
    let car_fw = Vec3::new(vcar.x, 0.0, vcar.z).dot(fw);
    let crate_fw = vc.dot(fw);
    eprintln!(
        "wheel contacts {contact}; car {car_fw:.3} m/s forward, crate {crate_fw:.3} m/s; horizontal momentum {:.3} -> {:.3} N s (car {:.0} N s)",
        p0.length(),
        p1.length(),
        car_fw * m
    );
    assert!(contact > 30 * 2, "the wheels stood on the crate ({contact} wheel-frames)");
    assert!(car_fw > 0.5, "the car drove forward ({car_fw} m/s)");
    assert!(crate_fw < -0.1, "the crate was pushed back ({crate_fw} m/s)");
    // Momentum: the exchange is car_fw * m; closed to well under 1 % of it.
    assert!((p1 - p0).length() < 0.005 * car_fw * m, "momentum {p0} -> {p1}");
}

/// A soft flatbed vehicle (no wheels): a 6 × 6 m two-layer node lattice 0.8 m deep,
/// 2 t, its top face collision triangles, lying on its own ground at y = 0 centred at
/// (`cx`, `cz`).
fn flatbed(cx: f32, cz: f32) -> ExplicitBeams {
    let n = 13;
    let id = |i: usize, j: usize, k: usize| (j * n + k) * n + i;
    let mut text = String::from("flatbed\n\nglobals\n2000, 0, tracks/semi\n\nminimass\n4\n\nset_beam_defaults 200000, 400, 400000, 4000000\n\nnodes\n");
    for j in 0..2 {
        for k in 0..n {
            for i in 0..n {
                text += &format!("{}, {}, {}, {}, n\n", id(i, j, k), cx - 3.0 + 0.5 * i as f32, 0.8 * j as f32, cz - 3.0 + 0.5 * k as f32);
            }
        }
    }
    text += "\nbeams\n";
    for j in 0..2usize {
        for k in 0..n {
            for i in 0..n {
                for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (0, 1, 1), (1, -1, 0), (0, -1, 1)] {
                    let (i2, j2, k2) = (i as i32 + di, j as i32 + dj, k as i32 + dk);
                    if (0..n as i32).contains(&i2) && (0..2).contains(&j2) && (0..n as i32).contains(&k2) {
                        text += &format!("{}, {}\n", id(i, j, k), id(i2 as usize, j2 as usize, k2 as usize));
                    }
                }
            }
        }
    }
    text += "\nsubmesh\ncab\n";
    for k in 0..n - 1 {
        for i in 0..n - 1 {
            let (a, b, c, d) = (id(i, 1, k), id(i + 1, 1, k), id(i + 1, 1, k + 1), id(i, 1, k + 1));
            text += &format!("{a}, {c}, {b}, c\n{a}, {d}, {c}, c\n");
        }
    }
    text += "\nend\n";
    let mut s = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).expect("flatbed");
    s.ground = 0.0;
    s.fixed_contact = true;
    s.finish();
    s
}

/// A raycast sedan (A) dropped onto a soft flatbed vehicle (B) in a world with vehicle
/// contact. Settled 1.5 s, then A at full throttle for 3 s. Returns (A's wheels in
/// contact after settling, A's wheel load over its weight, how far B's top sank under A
/// against B alone (m), A's travel along its forward relative to B (m)).
fn car_on_car(see: bool) -> (usize, f32, f32, f32) {
    use engine_softbody::explicit_contact::ExplicitBeamsWorld;
    let mut a = car(0.95, Some(0.0), see);
    a.fixed_contact = true;
    let fw = forward(&a);
    let (xa, _, ma) = com(&a);
    let b = flatbed(xa.x, xa.z);
    let top = |b: &ExplicitBeams| {
        let t: Vec<f32> = b.nodes.iter().map(|n| b.origin.y + n.pos.y).filter(|&y| y > 0.4).collect();
        t.iter().sum::<f32>() / t.len() as f32
    };
    let mut alone = b.clone();
    alone.run(1.5);
    let mut w = ExplicitBeamsWorld::new(vec![a, b]);
    for _ in 0..90 {
        w.frame_60();
    }
    let g = 9.807;
    let d = w.vehicles[0].drive.as_ref().unwrap();
    let contact = d.wheels.iter().filter(|s| s.contact).count();
    let la = d.wheels.iter().map(|s| s.load).sum::<f32>() / (ma * g);
    let sank = top(&alone) - top(&w.vehicles[1]);
    let rel = |w: &ExplicitBeamsWorld| com(&w.vehicles[0]).0 - com(&w.vehicles[1]).0;
    let r0 = rel(&w);
    w.vehicles[0].drive.as_mut().unwrap().input = THROTTLE;
    for _ in 0..180 {
        w.frame_60();
    }
    let went = (rel(&w) - r0).dot(fw);
    eprintln!("car on a vehicle, see {see}: A's wheels in contact {contact}, A load/weight {la:.3}, B's top sank {:.1} mm, A went {went:.2} m", sank * 1e3);
    (contact, la, sank, went)
}

#[test]
fn a_car_on_another_vehicle_drives_and_weighs_on_it() {
    let (contact, la, sank, went) = car_on_car(true);
    assert!(contact == 4, "A stands on its four wheels on B ({contact} in contact)");
    // A's weight goes through its wheels into B's triangles' nodes.
    assert!((la - 1.0).abs() < 0.1, "A's wheels carry its weight: {la}");
    assert!(sank > 1.0e-4, "B's top sank under A's weight: {sank} m");
    assert!(went > 2.5, "A drove on B: {went} m");
    let (c0, _, _, stuck) = car_on_car(false);
    assert_eq!(c0, 0, "without the setting A's wheels hang");
    assert!(stuck.abs() < 0.5, "without the setting A cannot drive: {stuck} m");
}

//! A raycast wheel is a solid: its hub never comes closer to the surface under it
//! than its rim's radius (the generated sedan; the press is
//! `engine-world/tests/wheel_rim_press.rs`).
//!
//! The raycast suspension (`explicit_drive`) acts between the hub and the ground.
//! Its bump stop is a finite spring at a length of 0.02 m against a rest length of
//! 0.203 m, so a hard landing or a kerb drives the hub, and the wheel and rim the
//! generated car draws on it, far under its rim's radius (measured before the fix,
//! `engine-world/examples/wheel_floor_probe`: the 5 m drop put the hubs at 0.155 m
//! over the ground against a 0.2 m rim and a 0.33 m tyre).
//!
//! Invariants (the wheel's geometry and energy, not recorded outputs):
//! - the hub (its axle nodes' middle) is never closer to the surface under it than
//!   the rim's radius less 1 cm (the stop is velocity-level, as the solver's ground:
//!   one step of approach at most): the rim's lowest point is never more than 1 cm
//!   into the surface, the tyre's undeformed outline never more than its section
//!   height (r − rim, a fully deflected tyre);
//! - nothing drives these scenes, so the car's kinetic + gravitational + elastic
//!   energy never rises above where it started (the ledger, as `sedan_energy.rs`).
use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox, STEP};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{Quat, Vec3};
use std::sync::OnceLock;

fn sedan() -> &'static SoftVehicleDesc {
    static D: OnceLock<SoftVehicleDesc> = OnceLock::new();
    D.get_or_init(|| {
        let inp = rig::DesignInput::find("sedan", None, None).expect("the sedan's design");
        let r = lattice::rig_with(&inp);
        let files = vg_input::physics_files(&inp, &r);
        let texts: Vec<(String, String)> = files
            .iter()
            .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
            .collect();
        let refs: Vec<(&str, &str)> = texts
            .iter()
            .filter(|(p, _)| p.ends_with(".jbeam"))
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let pc = texts
            .iter()
            .find(|(p, _)| p.ends_with(".pc"))
            .map(|(_, b)| b.as_str());
        engine_softbody::jbeam::load_jbeam(&refs, pc).expect("the sedan's jbeam")
    })
}

/// The sedan on raycast wheels facing +Z, its lowest wheel `lift` + 2 cm up, its
/// middle over the origin.
fn spawn(lift: f32) -> ExplicitBeams {
    let d = sedan();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(d, Vec3::ZERO, Vec3::ZERO, settings).expect("spawn");
    let f = engine_softbody::pack::forward_axis(d);
    let q = Quat::from_rotation_y(-f.x.atan2(f.z));
    let o = s.origin;
    s.origin = Vec3::ZERO;
    for n in &mut s.nodes {
        n.pos = q * (n.pos + o);
    }
    let (lo, hi) = s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    });
    let dr = s.drive.as_ref().unwrap();
    let wheel_low = dr
        .config
        .wheels
        .iter()
        .map(|w| {
            0.5 * (s.nodes[w.node1 as usize].pos.y + s.nodes[w.node2 as usize].pos.y) - w.radius
        })
        .fold(f32::MAX, f32::min);
    let shift = Vec3::new(
        -0.5 * (lo.x + hi.x),
        0.02 + lift - wheel_low,
        -0.5 * (lo.z + hi.z),
    );
    for n in &mut s.nodes {
        n.pos += shift;
    }
    s
}

/// Each raycast wheel's rim radius (the content's `hubRadius`) and tyre radius.
fn rims(s: &ExplicitBeams) -> Vec<(f32, f32)> {
    let d = sedan();
    s.drive
        .as_ref()
        .unwrap()
        .config
        .wheels
        .iter()
        .map(|w| {
            let pw = d
                .pressure_wheels
                .iter()
                .find(|p| {
                    (p.node1, p.node2) == (w.node1, w.node2)
                        || (p.node1, p.node2) == (w.node2, w.node1)
                })
                .expect("the wheel's pressureWheels row");
            (*pw.values.get("hubRadius").expect("hubRadius"), w.radius)
        })
        .collect()
}

/// The height of the surface under (x, z): the ground, or a box's top.
fn surface(s: &ExplicitBeams, x: f32, z: f32) -> f32 {
    let mut y = s.ground;
    for b in &s.boxes {
        if x >= b.lo.x && x <= b.hi.x && z >= b.lo.z && z <= b.hi.z {
            y = y.max(b.hi.y);
        }
    }
    y
}

struct Run {
    /// Lowest (hub height over the surface under it − rim radius) (m), the wheel's
    /// rim and tyre radius then, and when (s).
    worst: f32,
    rim: f32,
    radius: f32,
    at: f32,
    /// Largest rise of the ledger's energy above its start (J), the start (J).
    rise: f64,
    e0: f64,
    finite: bool,
}

fn run(s: ExplicitBeams, seconds: f32) -> Run {
    let rims = rims(&s);
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.enable_ledger();
    let e0 = w.ledger.as_ref().unwrap().mechanical();
    let mut out = Run {
        worst: f32::MAX,
        rim: 0.0,
        radius: 0.0,
        at: 0.0,
        rise: 0.0,
        e0,
        finite: true,
    };
    for k in 0..(seconds / STEP) as usize {
        w.step();
        let e = w.ledger.as_ref().unwrap().mechanical();
        if !e.is_finite() {
            out.finite = false;
            return out;
        }
        out.rise = out.rise.max(e - e0);
        let s = &w.vehicles[0];
        let dr = s.drive.as_ref().unwrap();
        for (wh, &(rim, radius)) in dr.config.wheels.iter().zip(&rims) {
            let hub =
                0.5 * (s.nodes[wh.node1 as usize].pos + s.nodes[wh.node2 as usize].pos) + s.origin;
            let gap = hub.y - surface(s, hub.x, hub.z) - rim;
            if gap < out.worst {
                out.worst = gap;
                out.rim = rim;
                out.radius = radius;
                out.at = k as f32 * STEP;
            }
        }
    }
    out
}

fn report(what: &str, r: &Run) {
    eprintln!(
        "{what}: hub lowest {:+.3} m against its rim ({:.3} m, tyre {:.3} m) at {:.3} s; energy rise {:.0} J of {:.0} J; finite {}",
        r.worst, r.rim, r.radius, r.at, r.rise, r.e0, r.finite
    );
}

/// Dropped flat from 5 m (9.9 m/s at the ground).
#[test]
fn a_5_m_drop_lands_on_the_rims_not_through_them() {
    let r = run(spawn(5.0), 2.5);
    report("5 m drop", &r);
    assert!(r.finite, "not finite");
    assert!(
        r.worst >= -0.01,
        "a hub came {:.3} m under its rim's radius ({:.3} m) over the ground at {:.3} s: the rim {:.3} m into the floor",
        -r.worst,
        r.rim,
        r.at,
        -r.worst
    );
    assert!(
        r.rise <= 1e-3 * r.e0,
        "energy rose {:.0} J above its start ({:.0} J)",
        r.rise,
        r.e0
    );
}

/// At 80 km/h square onto a 12 cm kerb (a box 6 m wide and long, its edge 2 m ahead
/// of the front wheels): the wheels ride up onto it. **Passes before the fix too**
/// (+7 mm over the rim at worst): the suspension carries a kerb before the stop is
/// reached. A guard that the stop changes nothing here for the worse. A higher kerb
/// cannot test the stop: a ray sees the kerb's top only once the hub is over it,
/// so a kerb taller than the tyre's section (0.13 m) puts the hub inside its rim's
/// radius of the top at that instant by geometry (known-bugs F7: rays see what is
/// under the hub only).
#[test]
fn a_kerb_strike_at_80_kmh_keeps_the_hubs_a_rim_over_the_kerb() {
    let mut s = spawn(0.0);
    let v = 80.0 / 3.6;
    for n in &mut s.nodes {
        n.vel = Vec3::Z * v;
    }
    if let Some(d) = s.drive.as_mut() {
        d.roll(v);
    }
    let front_hub = s
        .drive
        .as_ref()
        .unwrap()
        .config
        .wheels
        .iter()
        .map(|w| s.nodes[w.node1 as usize].pos.z)
        .fold(f32::MIN, f32::max);
    let edge = front_hub + 2.0;
    s.boxes.push(StaticBox {
        lo: Vec3::new(-3.0, s.ground - 1.0, edge),
        hi: Vec3::new(3.0, s.ground + 0.12, edge + 6.0),
    });
    let r = run(s, 1.2);
    report("12 cm kerb at 80 km/h", &r);
    assert!(r.finite, "not finite");
    assert!(
        r.worst >= -0.01,
        "a hub came {:.3} m under its rim's radius ({:.3} m) over the surface under it at {:.3} s",
        -r.worst,
        r.rim,
        r.at
    );
}

/// Steps `a` (the rim stop on) and `b` (off) in lockstep through `phases` (input,
/// seconds), comparing every node's position and velocity and every wheel's state
/// bit for bit while no hub of `a` has come within its rim's radius of the surface
/// under it. Returns (steps compared, whether all were identical, whether the stop
/// was ever reached).
fn lockstep(
    mut a: ExplicitBeams,
    mut b: ExplicitBeams,
    rims: &[f32],
    phases: &[(engine_softbody::explicit_drive::DriveInput, f32)],
) -> (usize, bool, bool) {
    a.drive.as_mut().unwrap().config.rim_stop = true;
    b.drive.as_mut().unwrap().config.rim_stop = false;
    let mut steps = 0;
    for &(input, secs) in phases {
        a.set_drive_input(input);
        b.set_drive_input(input);
        for _ in 0..(secs / STEP) as usize {
            a.step();
            b.step();
            let da = a.drive.as_ref().unwrap();
            let reached = da.config.wheels.iter().zip(rims).any(|(w, &rim)| {
                let hub = 0.5 * (a.nodes[w.node1 as usize].pos + a.nodes[w.node2 as usize].pos)
                    + a.origin;
                rim > 0.0 && hub.y - surface(&a, hub.x, hub.z) < rim
            });
            if reached {
                return (steps, true, true);
            }
            let same_nodes = a.origin.to_array().map(f32::to_bits)
                == b.origin.to_array().map(f32::to_bits)
                && a.nodes.iter().zip(&b.nodes).all(|(x, y)| {
                    x.pos.to_array().map(f32::to_bits) == y.pos.to_array().map(f32::to_bits)
                        && x.vel.to_array().map(f32::to_bits) == y.vel.to_array().map(f32::to_bits)
                });
            let db = b.drive.as_ref().unwrap();
            let same_wheels = da.wheels == db.wheels && da.smoothed == db.smoothed;
            if !(same_nodes && same_wheels) {
                return (steps, false, false);
            }
            steps += 1;
        }
    }
    (steps, true, false)
}

fn input(forward: f32, steer: f32, brake: f32) -> engine_softbody::explicit_drive::DriveInput {
    engine_softbody::explicit_drive::DriveInput {
        forward,
        steer,
        brake,
        handbrake: 0.0,
    }
}

/// Ordinary driving never reaches the stop, and runs bit for bit as without it: the
/// generated sedan (settle, full throttle, a steady turn, braking to a stop), the
/// JBeam fixture car in JBeam mode and the procedural sedan (its RoR `wheels` give no
/// rim: the stop has nothing to act on), as `explicit_drive.rs` drives them.
#[test]
fn ordinary_driving_is_bit_identical_with_the_rim_stop() {
    let drive = [
        (input(0.0, 0.0, 0.0), 1.0),
        (input(1.0, 0.0, 0.0), 3.0),
        (input(0.6, 0.5, 0.0), 2.0),
        (input(0.0, 0.0, 1.0), 2.0),
    ];
    let gen = |_: ()| spawn(0.0);
    let rims_gen: Vec<f32> = rims(&gen(())).iter().map(|r| r.0).collect();
    let (n, same, reached) = lockstep(gen(()), gen(()), &rims_gen, &drive);
    eprintln!("generated sedan: {n} steps compared, identical {same}, stop reached {reached}");
    assert!(
        !reached,
        "the generated sedan reached its rim stop in ordinary driving after {n} steps"
    );
    assert!(
        same,
        "the generated sedan's run differs after {n} steps with the stop never reached"
    );

    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let fixture = engine_softbody::jbeam::load_jbeam(
        &[("t.jbeam", engine_softbody::jbeam_fixture::PART2_CAR)],
        None,
    )
    .unwrap();
    let fix = || {
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(
            &fixture,
            Vec3::new(0.0, 0.8, 0.0),
            Vec3::ZERO,
            settings,
        )
        .unwrap();
        s.finish();
        s
    };
    let rims_fix: Vec<f32> = fix()
        .drive
        .as_ref()
        .unwrap()
        .config
        .wheels
        .iter()
        .map(|w| w.rim_radius)
        .collect();
    let (n, same, reached) = lockstep(
        fix(),
        fix(),
        &rims_fix,
        &[(input(0.0, 0.0, 0.0), 2.0), (input(1.0, 0.0, 0.0), 2.0)],
    );
    eprintln!("JBeam fixture (rims {rims_fix:?}): {n} steps compared, identical {same}, stop reached {reached}");
    assert!(
        same,
        "the JBeam fixture's run differs after {n} steps with the stop never reached"
    );

    let truck = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
        minimass: 10.0,
        ..engine_softbody::procedural::CarParams::sedan()
    });
    let proc_car = || {
        let (mut s, _) = ExplicitBeams::from_truck_raycast(
            &truck,
            None,
            Vec3::new(0.0, 0.6, 0.0),
            0.0,
            Vec3::ZERO,
            settings,
        )
        .unwrap();
        s.finish();
        s
    };
    let p = proc_car();
    let rims_proc: Vec<f32> = p
        .drive
        .as_ref()
        .unwrap()
        .config
        .wheels
        .iter()
        .map(|w| w.rim_radius)
        .collect();
    assert!(
        rims_proc.iter().all(|&r| r == 0.0),
        "RoR wheels gave a rim: {rims_proc:?}"
    );
    let (n, same, _) = lockstep(proc_car(), proc_car(), &rims_proc, &drive);
    eprintln!("procedural sedan: {n} steps compared, identical {same}");
    assert!(same, "the procedural sedan's run differs after {n} steps");
}

/// The stop is what changed the drop: off, the same drop puts a hub 4.9 cm into its
/// rim's radius (the failure the drop test guards).
#[test]
fn without_the_stop_the_drop_goes_through_the_rim() {
    let mut s = spawn(5.0);
    s.drive.as_mut().unwrap().config.rim_stop = false;
    let r = run(s, 2.5);
    report("5 m drop, stop off", &r);
    assert!(
        r.worst < -0.02,
        "the drop without the stop stayed {:+.3} m from the rim",
        r.worst
    );
}

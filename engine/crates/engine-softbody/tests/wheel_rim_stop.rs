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
        .map(|w| 0.5 * (s.nodes[w.node1 as usize].pos.y + s.nodes[w.node2 as usize].pos.y) - w.radius)
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
                .find(|p| (p.node1, p.node2) == (w.node1, w.node2) || (p.node1, p.node2) == (w.node2, w.node1))
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
            let hub = 0.5 * (s.nodes[wh.node1 as usize].pos + s.nodes[wh.node2 as usize].pos)
                + s.origin;
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

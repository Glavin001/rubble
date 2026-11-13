//! Parts of one car collide with each other: a door struck from the side does
//! not pass through the pillar, the sill and the seat behind it (the user's
//! report, 2026-11-13: a struck door ended inside the cabin, through the
//! B-pillar and the driver's seat).
//!
//! **The scene:** the rigged sedan, sideways on a carrier at 32 km/h (FMVSS
//! 214's pole speed) onto a rigid ram at the driver's door: 0.6 m along the
//! car at the seat, 0.45 to 0.85 m up, a striking car's bumper. It is over
//! the sill and between the pillars, so the door is driven into its opening.
//! (A pole to the ground at this speed crushes the sill and the floor with
//! the door: the seat moves 304 mm with the floor and the door never reaches
//! it.) CPU reference, surface contact on, the centred damping's objective
//! form on.
//!
//! **The properties** (every 2.5 ms for 0.3 s):
//!
//! - **No door node inside the seat.** The seat is the convex hull of its
//!   nodes. A node of the struck door is never deeper inside it than the
//!   contact band ([`BAND`]), and a door node beside the seat (within its
//!   length and height) is never further in than the seat's outboard face
//!   by more than the band.
//! - **The door's surface is not crossed by the body's frame or the seat.**
//!   The door opening's frame (the sill's top edge, the hinge pillar, the
//!   A-pillar, the roof rail, the B-pillar) and the seat's outboard edges
//!   never pierce a collision triangle of the door.
//! - **The seat is pushed.** If the door came to the seat at all, the seat
//!   has moved inboard.
//! - **The ledger closes:** kinetic + gravitational + elastic energy never
//!   rises above where it started.
use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox, STEP};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{Quat, Vec3};
use std::sync::OnceLock;

/// The contact band: the solver's collision range (2 cm) and as much again
/// for a contact that is being pushed in.
const BAND: f32 = 0.04;

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
        let pc = texts.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, b)| b.as_str());
        engine_softbody::jbeam::load_jbeam(&refs, pc).expect("the sedan's jbeam")
    })
}

/// The sedan on raycast wheels turned by `yaw`, its lowest wheel 2 cm up, its
/// middle over the origin (as `tests/sedan_energy.rs`).
fn spawn(yaw: f32) -> ExplicitBeams {
    let d = sedan();
    let settings = ExplicitBeamsSettings { turbulence: false, ..Default::default() };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(d, Vec3::ZERO, Vec3::ZERO, settings).expect("spawn");
    s.settings.centred_objective = true;
    let f = engine_softbody::pack::forward_axis(d);
    let q = Quat::from_rotation_y(yaw - f.x.atan2(f.z));
    let o = s.origin;
    s.origin = Vec3::ZERO;
    for n in &mut s.nodes {
        n.pos = q * (n.pos + o);
    }
    let (lo, hi) = s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)));
    let wheel_low = s
        .drive
        .as_ref()
        .map(|dr| {
            dr.config
                .wheels
                .iter()
                .map(|w| 0.5 * (s.nodes[w.node1 as usize].pos.y + s.nodes[w.node2 as usize].pos.y) - w.radius)
                .fold(f32::MAX, f32::min)
        })
        .expect("raycast wheels");
    let shift = Vec3::new(-0.5 * (lo.x + hi.x), 0.02 - wheel_low, -0.5 * (lo.z + hi.z));
    for n in &mut s.nodes {
        n.pos += shift;
    }
    s
}

/// The faces of the convex hull of `p` (outward normal, offset): a triple is
/// a face when every other point is on one side of its plane.
fn hull(p: &[Vec3]) -> Vec<(Vec3, f32)> {
    let mut out = Vec::new();
    for a in 0..p.len() {
        for b in a + 1..p.len() {
            for c in b + 1..p.len() {
                let n = (p[b] - p[a]).cross(p[c] - p[a]);
                if n.length() < 1e-6 {
                    continue;
                }
                let n = n.normalize();
                let d: Vec<f32> = p.iter().map(|q| n.dot(*q - p[a])).collect();
                if d.iter().all(|&x| x <= 1e-4) {
                    out.push((n, n.dot(p[a])));
                } else if d.iter().all(|&x| x >= -1e-4) {
                    out.push((-n, -n.dot(p[a])));
                }
            }
        }
    }
    out
}

/// How deep `q` is inside the hull (m; 0 outside).
fn depth(h: &[(Vec3, f32)], q: Vec3) -> f32 {
    h.iter().map(|(n, d)| d - n.dot(q)).fold(f32::MAX, f32::min).max(0.0)
}

/// Whether segment `a`-`b` crosses triangle `t`.
fn pierces(a: Vec3, b: Vec3, t: [Vec3; 3]) -> bool {
    let n = (t[1] - t[0]).cross(t[2] - t[0]);
    let (da, db) = (n.dot(a - t[0]), n.dot(b - t[0]));
    if da * db >= 0.0 {
        return false;
    }
    let x = a + (b - a) * (da / (da - db));
    (0..3).all(|k| n.dot((t[(k + 1) % 3] - t[k]).cross(x - t[k])) >= 0.0)
}

fn env(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

struct Outcome {
    /// Deepest a door node was inside the seat's hull (m), and when (s).
    in_seat: (f32, f32, String),
    /// Furthest a door node beside the seat was past the seat's outboard
    /// face (m), and when.
    past_face: (f32, f32, String),
    /// Most frame or seat edges through the door's triangles at one time.
    pierced: (usize, f32, String),
    /// The seat's inboard move at the end (m) and the door's deepest reach
    /// toward the seat (m inboard of where the seat's outboard face stood).
    seat_moved: f32,
    door_reach: f32,
    rise: f64,
    e0: f64,
}

/// The ram scene (`tbone` false) or the T-bone: a second sedan at 50 km/h
/// into the parked car's driver's door, its middle at the seat.
fn scene(tbone: bool) -> Outcome {
    {
        let names: Vec<String> = sedan().nodes.iter().map(|n| n.name.clone()).collect();
        let id = |n: &str| names.iter().position(|x| x == n).unwrap_or_else(|| panic!("no node {n}"));
        // The ram: the driver's side leads (+z). The T-bone: it faces −z.
        let quarter = std::f32::consts::FRAC_PI_2;
        let mut s = spawn(if tbone { quarter } else { -quarter });
        if !tbone {
            for n in &mut s.nodes {
                n.vel = Vec3::Z * (env("RAM_KMH", 32.0) / 3.6);
            }
            engine_softbody::crash_rigs::on_carrier(&mut s);
        }
        s.surface_contact = true;
        let door: Vec<usize> = (0..names.len()).filter(|&i| names[i].starts_with("door_FL_") && !names[i].ends_with("_strike")).collect();
        let seat: Vec<usize> = (0..names.len()).filter(|&i| names[i].starts_with("seafl_")).collect();
        assert!(door.len() > 10 && seat.len() >= 6, "the door's and the seat's nodes");
        let pos = |s: &ExplicitBeams, i: usize| s.origin + s.nodes[i].pos;
        let lead = door.iter().map(|&i| pos(&s, i).z).fold(0.0f32, |a, z| if z.abs() > a.abs() { z } else { a });
        assert!(if tbone { lead < -0.5 } else { lead > 0.5 }, "the driver's door faces the strike");
        let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        // The seat's middle along the car.
        let at = seat.iter().map(|&i| pos(&s, i).x).sum::<f32>() / seat.len() as f32;
        let mut striker = None;
        if tbone {
            let v = env("TBONE_KMH", 50.0) / 3.6;
            let mut p = spawn(0.0);
            p.surface_contact = true;
            let nose = p.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
            let side = s.nodes.iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
            for n in &mut s.nodes {
                n.pos += Vec3::new(-at, 0.0, nose - side + 0.3);
            }
            for n in &mut p.nodes {
                n.vel = Vec3::Z * v;
            }
            if let Some(d) = p.drive.as_mut() {
                d.roll(v);
            }
            striker = Some(p);
        } else {
            // The ram: 0.6 m along the car, 0.45 to 0.85 m up (over the sill,
            // under the belt), 5 cm ahead of the car's side.
            s.boxes.push(StaticBox::centered(Vec3::new(at, 0.65, front + 0.05 + 0.127), Vec3::new(env("RAM_HALF", 0.30), 0.20, 0.127)));
        }
        let door_tris: Vec<[u32; 3]> = s
            .triangles
            .iter()
            .map(|t| t.nodes)
            .filter(|t| t.iter().all(|&i| door.contains(&(i as usize))))
            .collect();
        assert!(!door_tris.is_empty(), "the door has collision triangles");
        // The frame under and behind the door, and the seat's outboard edges.
        let mut edges: Vec<(String, usize, usize)> = [
            ("sh_c2_3l", "sh_d2_bl"),
            ("sh_d2_bl", "sh_d2_tl"),
            ("sh_c2_1l", "sh_c2_2l"),
            ("sh_c2_2l", "sh_c2_3l"),
            ("sh_b2_ol", "sh_c2_1l"),
            ("sh_b2_ol", "sh_d1_ml"),
            ("sh_d1_ml", "sh_d1_tl"),
            ("sh_d1_tl", "sh_d2_tl"),
        ]
        .iter()
        .map(|(a, b)| (format!("{a}-{b}"), id(a), id(b)))
        .collect();
        for (a, b) in [("seafl_fo", "seafl_ro"), ("seafl_ro", "seafl_bo"), ("seafl_fo", "seafl_bo"), ("seafl_c", "seafl_bo")] {
            edges.push((format!("{a}-{b}"), id(a), id(b)));
        }
        // The seat's outboard face at the start, and the far sill as the
        // reference the seat's move is measured against.
        let far = id("sh_c2_3r");
        // (distances across the car to the far sill: smaller is further in)
        let across = |s: &ExplicitBeams, i: usize| (pos(s, i).z - pos(s, far).z).abs();
        let seat_c = |s: &ExplicitBeams| seat.iter().map(|&i| across(s, i)).sum::<f32>() / seat.len() as f32;
        let gap0 = seat_c(&s);
        let seat_face0 = seat.iter().map(|&i| across(&s, i)).fold(f32::MIN, f32::max);
        let mut w = ExplicitBeamsWorld::new(std::iter::once(s).chain(striker).collect());
        w.enable_ledger();
        let e0 = w.ledger.as_ref().unwrap().mechanical();
        let mut o = Outcome { in_seat: (0.0, 0.0, String::new()), past_face: (0.0, 0.0, String::new()), pierced: (0, 0.0, String::new()), seat_moved: 0.0, door_reach: f32::MIN, rise: 0.0, e0 };
        let steps = ((if tbone { 0.35 } else { 0.3 }) / STEP) as usize;
        for k in 0..steps {
            w.step();
            let e = w.ledger.as_ref().unwrap().mechanical();
            assert!(e.is_finite(), "not finite at {:.3} s", k as f32 * STEP);
            o.rise = o.rise.max(e - e0);
            if k % 5 != 4 {
                continue;
            }
            let s = &w.vehicles[0];
            let t = (k + 1) as f32 * STEP;
            let sp: Vec<Vec3> = seat.iter().map(|&i| pos(s, i)).collect();
            let h = hull(&sp);
            let (lo, hi) = sp.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), q| (a.min(*q), b.max(*q)));
            let face = seat.iter().map(|&i| across(s, i)).fold(f32::MIN, f32::max);
            for &i in &door {
                let d = depth(&h, pos(s, i));
                if d > o.in_seat.0 {
                    o.in_seat = (d, t, names[i].clone());
                }
                // The door's reach toward the seat, and past the seat's
                // outboard face as it stands now: its nodes beside the seat
                // (within its length and its height).
                let q = pos(s, i);
                if q.x > lo.x && q.x < hi.x && q.y > lo.y && q.y < hi.y {
                    o.door_reach = o.door_reach.max(seat_face0 - across(s, i));
                    let past = face - across(s, i);
                    if past > o.past_face.0 {
                        o.past_face = (past, t, names[i].clone());
                    }
                }
            }
            let through: Vec<&str> = edges
                .iter()
                .filter(|(_, a, b)| door_tris.iter().any(|tr| pierces(pos(s, *a), pos(s, *b), tr.map(|i| pos(s, i as usize)))))
                .map(|e| e.0.as_str())
                .collect();
            if through.len() > o.pierced.0 {
                o.pierced = (through.len(), t, through.join(", "));
            }
            if k + 5 >= steps {
                o.seat_moved = gap0 - seat_c(s);
            }
        }
        println!(
            "{}: \n ", if tbone { "T-bone into the driver's door, 50 km/h" } else { "ram at the driver's door, 32 km/h" });
        println!(
            "  door node deepest in the seat {:.0} mm ({} at {:.3} s), past its outboard face {:.0} mm ({} at {:.3} s); frame and seat edges through the door's triangles {} ({} at {:.3} s); door reach past the seat's face {:.0} mm; seat moved inboard {:.0} mm; energy rise {:.1} J of {:.0}",
            o.in_seat.0 * 1e3, o.in_seat.2, o.in_seat.1, o.past_face.0 * 1e3, o.past_face.2, o.past_face.1, o.pierced.0, o.pierced.2, o.pierced.1, o.door_reach * 1e3, o.seat_moved * 1e3, o.rise, o.e0
        );
        o
    }
}

fn ram() -> &'static Outcome {
    static O: OnceLock<Outcome> = OnceLock::new();
    O.get_or_init(|| scene(false))
}

fn tbone() -> &'static Outcome {
    static O: OnceLock<Outcome> = OnceLock::new();
    O.get_or_init(|| scene(true))
}

/// Both scenes' outcomes with their names.
fn both() -> [(&'static str, &'static Outcome); 2] {
    [("ram", ram()), ("T-bone", tbone())]
}

#[test]
fn a_struck_door_never_ends_inside_the_seat() {
    for (name, o) in both() {
    assert!(
        o.in_seat.0 <= BAND,
        "{name}: {} is {:.0} mm inside the seat at {:.3} s (band {:.0} mm)",
        o.in_seat.2,
        o.in_seat.0 * 1e3,
        o.in_seat.1,
        BAND * 1e3
    );
        assert!(
            o.past_face.0 <= BAND,
            "{name}: {} is {:.0} mm past the seat's outboard face, beside the seat, at {:.3} s",
            o.past_face.2,
            o.past_face.0 * 1e3,
            o.past_face.1
        );
    }
}

#[test]
fn the_pillar_the_sill_and_the_seat_never_pierce_the_door() {
    for (name, o) in both() {
        assert!(o.pierced.0 == 0, "{name}: through the door's triangles at {:.3} s: {}", o.pierced.1, o.pierced.2);
    }
}

#[test]
fn a_door_that_reaches_the_seat_pushes_it() {
    for (name, o) in both() {
        assert!(
            o.door_reach < BAND || o.seat_moved > 0.5 * (o.door_reach - BAND),
            "{name}: the door reached {:.0} mm past the seat's face and the seat moved {:.0} mm",
            o.door_reach * 1e3,
            o.seat_moved * 1e3
        );
    }
}

#[test]
fn the_side_poles_ledger_closes() {
    for (name, o) in both() {
        assert!(o.rise <= 1e-3 * o.e0, "{name}: energy rose {:.0} J above its start ({:.0} J)", o.rise, o.e0);
    }
}

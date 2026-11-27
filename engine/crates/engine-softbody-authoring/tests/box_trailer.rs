//! The generated 53 ft dry-van semi-trailer (`engine_vehiclegen::designs::box_trailer`,
//! rigged by `gen2::parts_rig`) on the CPU reference: evidence that it is physically sound
//! before the crash reference scenes use it.
//!
//! * rest: parked on its landing gear and its tandem it settles and stays (node speeds
//!   after 2 s, no beam yielded or broken, the deck's height steady, the lowest point on the
//!   ground, kinetic energy never rising);
//! * mass and the support split: the gross against the design's, and the ground's reaction
//!   under the tandem and under the landing gear against the lever rule on the design's own
//!   centre of mass (a statically determinate split: two supports);
//! * the rear impact guard against FMVSS No. 223 (49 CFR 571.223 S5.2, S6.4–S6.8): the
//!   trailer held, a rigid block pushed slowly into P1, P2 and across the supports (the
//!   uniform distributed load), force against deflection; the guard carries at least the
//!   regulation's 50 kN at P1 and P2 and 350 kN distributed before any attachment breaks.
//!
//! Run: `cargo test --release -p engine-softbody-authoring --test box_trailer -- --nocapture`

use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, NODE_IMMOVABLE};
use engine_softbody_authoring::authoring::gen2::vg_input;
use engine_softbody_authoring::vehicle_compare::{lowest_point, settings};
use engine_vehiclegen::designs::box_trailer::{self, BoxTrailer};
use glam::Vec3;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

const STEP: f32 = 0.0005;
/// The FMVSS 223 block's speed (m/s).
const PUSH_SPEED: f32 = 0.08;

fn desc() -> &'static SoftVehicleDesc {
    static D: OnceLock<SoftVehicleDesc> = OnceLock::new();
    D.get_or_init(|| {
        let inp = vg_input::VgInput::find("box_trailer", None, None).expect("box_trailer");
        let b = vg_input::build(inp).expect("rig");
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("box_trailer_test");
        for (p, bytes) in vg_input::files(&b) {
            let path = dir.join(&p);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
        }
        let files = engine_softbody::jbeam::read_vehicle_dir(&dir, "box_trailer").unwrap();
        let pc = std::fs::read_to_string(dir.join("box_trailer/box_trailer.pc")).ok();
        let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).unwrap()
    })
}

fn spawn() -> ExplicitBeams {
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(desc(), Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    s.surface_contact = true;
    let low = lowest_point(&s);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s
}

fn nd() -> usize {
    desc().nodes.len()
}

fn pos(s: &ExplicitBeams, i: usize) -> Vec3 {
    s.origin + s.nodes[i].pos
}

fn finite(s: &ExplicitBeams) -> bool {
    s.nodes.iter().all(|n| n.pos.is_finite() && n.vel.is_finite())
}

fn ids(prefix: &str) -> Vec<usize> {
    (0..nd()).filter(|&i| desc().nodes[i].name.starts_with(prefix)).collect()
}

fn centroid(s: &ExplicitBeams, ids: &[usize]) -> Vec3 {
    ids.iter().map(|&i| pos(s, i)).sum::<Vec3>() / ids.len() as f32
}

fn kinetic(s: &ExplicitBeams) -> f64 {
    s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum()
}

fn name(i: u32) -> String {
    desc().nodes.get(i as usize).map_or("wheel".to_string(), |n| n.name.clone())
}

/// Each node's part (the JBeam part that lists it).
fn part_of() -> Vec<String> {
    let mut out = vec![String::new(); nd()];
    for p in &desc().parts {
        for &i in &p.nodes {
            if (i as usize) < out.len() && out[i as usize].is_empty() {
                out[i as usize] = p.name.trim_start_matches("box_trailer_").to_string();
            }
        }
    }
    out
}

/// (bent, broken) beams by the part of their first node, and the bent and broken beams' names.
fn damage(s: &ExplicitBeams) -> (BTreeMap<String, (usize, usize)>, Vec<String>) {
    let part = part_of();
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut names = Vec::new();
    for b in &s.beams {
        let a = b.a as usize;
        let key = if a < part.len() { part[a].clone() } else { "wheels".into() };
        let e = out.entry(key).or_default();
        if b.broken {
            e.1 += 1;
            names.push(format!("{}-{} broken", name(b.a), name(b.b)));
        } else if b.bent() {
            e.0 += 1;
            names.push(format!("{}-{} set {:.1} mm", name(b.a), name(b.b), (b.rest - b.rest0) * 1e3));
        }
    }
    out.retain(|_, v| v.0 + v.1 > 0);
    (out, names)
}

/// Parked 10 s after the 2 cm drop of the spawn. (The guard bar rings longest: its first
/// modes, a 100 kg bar on stiffness the step's bound softens, take their damping from the
/// beams' relative damping, about 1 % modal, and ring down over about 5 s.)
#[test]
fn parked_it_settles_and_stays() {
    const T: f32 = 10.0;
    let mut s = spawn();
    let deck: Vec<usize> = ids("bx");
    let (mut heights, mut ke) = (Vec::new(), Vec::new());
    let (mut vmax_late, mut vmax_end, mut guard_end) = (0.0f32, (0.0f32, 0usize), 0.0f32);
    for k in 0..(T / STEP).round() as usize {
        s.step();
        let t = (k + 1) as f32 * STEP;
        if (k + 1) % 200 == 0 {
            heights.push((t, centroid(&s, &deck).y));
        }
        if (k + 1) % 100 == 0 {
            ke.push(kinetic(&s));
        }
        if t >= T - 4.0 {
            vmax_late = vmax_late.max((0..nd()).map(|i| s.nodes[i].vel.length()).fold(0.0, f32::max));
        }
        if t >= T - 0.5 {
            for i in 0..nd() {
                let v = s.nodes[i].vel.length();
                if desc().nodes[i].name.starts_with("gb") {
                    guard_end = guard_end.max(v);
                } else if v > vmax_end.0 {
                    vmax_end = (v, i);
                }
            }
        }
    }
    assert!(finite(&s), "state not finite");
    let h = |t0: f32| heights.iter().find(|x| x.0 >= t0 - 1e-4).unwrap().1;
    let (h2, h10) = (h(2.0), h(T));
    let (dmg, bent) = damage(&s);
    let low = lowest_point(&s);
    let guard = ids("gb").iter().map(|&i| pos(&s, i).y).fold(f32::MAX, f32::min);
    let peak = |a: usize, b: usize| ke[a..b].iter().cloned().fold(0.0, f64::max);
    let peaks: Vec<f64> = (1..T as usize).map(|w| peak(w * 20, (w + 1) * 20)).collect();
    println!(
        "parked {T} s: max node speed over the last 4 s {vmax_late:.4} m/s, the last 0.5 s {:.4} m/s ({}; the guard bar's {guard_end:.4}); box height at 2 s {h2:.4} m, at {T} s {h10:.4} m (change {:.2} mm); lowest point {:.1} mm; guard's lower edge {:.3} m over the ground; kinetic energy peak per second from 1 s (J) {:?}",
        vmax_end.0,
        desc().nodes[vmax_end.1].name,
        (h10 - h2) * 1e3,
        low * 1e3,
        guard - low,
        peaks.iter().map(|p| format!("{p:.3}")).collect::<Vec<_>>()
    );
    println!("  bent / broken by part: {dmg:?} {bent:?}");
    assert!(vmax_end.0 < 0.025, "a node still moves at {:.3} m/s at the end", vmax_end.0);
    // the guard bar still rings down (see above): slower than 0.1 m/s, its energy falling
    assert!(guard_end < 0.1, "the guard bar still moves at {guard_end:.3} m/s at the end");
    assert!((h10 - h2).abs() < 0.002, "the box's height drifts {:.1} mm from 2 to {T} s", (h10 - h2) * 1e3);
    assert!(dmg.is_empty(), "beams yielded or broke parked: {dmg:?} {bent:?}");
    assert!(low.abs() < 0.01, "the parked trailer's lowest point is {:.0} mm from the ground", low * 1e3);
    // (The guard's height parked is reported, not asserted: the raycast springs are sized
    // for the mean wheel load, total / 4, and the tandem carries 61 % of the trailer: it
    // stands high by the springs' unused sag. The design's own height is held to FMVSS 224
    // in `crash_zones`.)
    for w in 1..peaks.len() {
        assert!(peaks[w] <= peaks[w - 1] * 1.05 + 0.05, "kinetic energy grew at rest: {:.3} J after {:.3} J", peaks[w], peaks[w - 1]);
    }
}

#[test]
fn mass_and_support_split_match_the_design() {
    let mut s = spawn();
    for _ in 0..(3.0 / STEP).round() as usize {
        s.step();
    }
    let nodes: f32 = desc().nodes.iter().map(|n| n.mass).sum();
    let wheels: f32 = desc().wheels.iter().map(|w| w.mass).sum();
    let total = nodes + wheels;
    let g = 9.81;
    let d = s.drive.as_ref().expect("raycast wheels");
    let tandem: f32 = d.wheels.iter().map(|w| w.load).sum::<f32>() / g;
    // The lever rule on the design's centre of mass between the tandem's middle and the
    // landing gear (two supports: statically determinate).
    let t = BoxTrailer::new(Default::default());
    let (defs, _) = t.assembly();
    let (m, mz) = defs
        .iter()
        .flat_map(|p| p.nodes.iter())
        .fold((0.0f64, 0.0f64), |(m, mz), n| (m + n.mass, mz + n.mass * n.pos.z));
    let zg = (mz + 4.0 * 20.0 * 0.0) / (m + 80.0);
    let f = t.frame();
    let z_lg = f.stations()[engine_vehiclegen::heavy::station_near(&f, t.landing().z)];
    let legs_kg = (m + 80.0) * zg / z_lg;
    let tandem_kg = (m + 80.0) - legs_kg;
    println!(
        "mass {total:.0} kg (nodes {nodes:.0} + raycast wheels {wheels:.0}; design {:.0} = kerb {:.0} + load {:.0}); centre of mass {zg:.3} m ahead of the tandem's middle; the tandem carries {tandem:.0} kg parked (lever rule {tandem_kg:.0}), the landing gear {:.0} kg ({legs_kg:.0})",
        box_trailer::GROSS,
        box_trailer::KERB,
        box_trailer::GROSS - box_trailer::KERB,
        total - tandem
    );
    assert!((total / box_trailer::GROSS as f32 - 1.0).abs() < 0.01, "mass {total:.0} kg");
    assert!((tandem / tandem_kg as f32 - 1.0).abs() <= 0.05, "the tandem carries {tandem:.0} kg against {tandem_kg:.0}");
}

/// The force the guard's supports hold the bar back with along +Z (N): each unbroken
/// beam from a guard node to another part's, its last axial force on the guard node. At
/// a quasi-static push (the bar's inertia a few newtons, gravity across the push) it is
/// the block's push.
fn support_push(s: &ExplicitBeams) -> f32 {
    let mut f = 0.0f32;
    for b in s.beams.iter().filter(|b| !b.broken) {
        let (ga, gb) = (name(b.a).starts_with("gb"), name(b.b).starts_with("gb"));
        if ga == gb {
            continue;
        }
        let (g, o) = if ga { (b.a, b.b) } else { (b.b, b.a) };
        let d = (s.nodes[g as usize].pos - s.nodes[o as usize].pos).normalize_or_zero();
        // (force > 0: compression, pushing the guard node away from the other)
        f -= b.force * d.z;
    }
    f
}

/// One FMVSS 223 push: the block's x span and height, the guard's force (N) by the block's
/// travel past first contact (m), and the travel at which the first attachment broke.
struct Push {
    curve: Vec<(f32, f32)>,
    first_break: Option<(f32, String)>,
    /// The beams with the most permanent set at the end.
    set: Vec<String>,
}

/// The trailer held still (every node but the guard bar's immovable), and a rigid platen
/// 203 mm high centred 50 mm above the guard's bottom (S6.4, S6.7) driven forward at
/// [`PUSH_SPEED`] (quasi-static for a 100 kg bar: its inertia at that speed is a few
/// newtons; the regulation's 2-9 cm/min would be 270-1,200 s of 2 kHz steps) for `travel`
/// metres past first contact. The platen is kinematic: each rear-face node of the bar in
/// its footprint (x in `x0..x1`, its height band) is carried with it from the moment the
/// face reaches it (a rigid static box moved through the bar let a few nodes of a point
/// load slip out over its top). The force is the supports' reaction ([`support_push`]).
fn push(x0: f32, x1: f32, travel: f32) -> Push {
    let mut s = spawn();
    let guard: Vec<usize> = ids("gb");
    for i in 0..nd() {
        if !desc().nodes[i].name.starts_with("gb") {
            s.nodes[i].flags |= NODE_IMMOVABLE;
            s.nodes[i].vel = Vec3::ZERO;
        }
    }
    s.drive = None;
    // settle the bar on its supports
    for _ in 0..(0.5 / STEP).round() as usize {
        s.step();
    }
    let bottom = guard.iter().map(|&i| pos(&s, i).y).fold(f32::MAX, f32::min) - 0.005;
    let yc = bottom + 0.05;
    let rear_z = guard.iter().map(|&i| pos(&s, i).z).fold(f32::MAX, f32::min);
    // the bar's rear-face nodes in the platen's footprint
    let under: Vec<usize> = guard
        .iter()
        .copied()
        .filter(|&i| {
            let p = pos(&s, i);
            (x0..=x1).contains(&p.x) && (p.y - yc).abs() <= 0.1015 && p.z < rear_z + 0.02
        })
        .collect();
    assert!(!under.is_empty(), "no guard node under the platen at x {x0}..{x1}");
    let face0 = under.iter().map(|&i| pos(&s, i).z).fold(f32::MAX, f32::min);
    let attach = |s: &ExplicitBeams| -> Option<String> {
        s.beams.iter().find(|b| b.broken && (name(b.a).starts_with("gb") != name(b.b).starts_with("gb"))).map(|b| format!("{}-{}", name(b.a), name(b.b)))
    };
    let (mut curve, mut first_break) = (Vec::new(), None);
    let v = PUSH_SPEED;
    let steps = (travel / v / STEP).round() as usize;
    let mut held = vec![false; nd()];
    for k in 0..steps {
        let face = face0 + v * STEP * (k + 1) as f32;
        for &i in &under {
            if held[i] || pos(&s, i).z <= face {
                held[i] = true;
                s.nodes[i].flags |= NODE_IMMOVABLE;
                s.nodes[i].vel = Vec3::new(0.0, 0.0, v);
                s.nodes[i].pos.z = face - s.origin.z;
            }
        }
        s.step();
        let f = support_push(&s);
        if k % 20 == 0 {
            curve.push((face - face0, f));
        }
        if first_break.is_none() {
            if let Some(b) = attach(&s) {
                first_break = Some((face - face0, b));
            }
        }
    }
    assert!(finite(&s), "state not finite in the push");
    let mut set: Vec<(f32, String)> = s.beams.iter().filter(|b| b.bent() || b.broken).map(|b| ((b.rest - b.rest0).abs(), format!("{}-{} {:+.1} mm{}", name(b.a), name(b.b), (b.rest - b.rest0) * 1e3, if b.broken { " broken" } else { "" }))).collect();
    set.sort_by(|a, b| b.0.total_cmp(&a.0));
    Push { curve, first_break, set: set.into_iter().take(12).map(|x| x.1).collect() }
}

/// The largest force before the first attachment broke, the force at 125 mm, the energy
/// within the first 125 mm (J), and the curve's key points (every 25 mm to 400 mm).
fn summary(p: &Push) -> (f32, f32, f64, String) {
    let until = p.first_break.as_ref().map_or(f32::MAX, |b| b.0);
    let before = p.curve.iter().filter(|x| x.0 < until).map(|x| x.1).fold(0.0, f32::max);
    let at = |d: f32| p.curve.iter().find(|x| x.0 >= d).map_or(f32::NAN, |x| x.1);
    let mut e = 0.0f64;
    for w in p.curve.windows(2) {
        if w[1].0 <= 0.125 {
            e += 0.5 * f64::from(w[0].1 + w[1].1) * f64::from(w[1].0 - w[0].0);
        }
    }
    let pts: Vec<String> = (1..=16).map(|k| 0.025 * k as f32).map(|d| format!("{:.0} mm {:.1} kN", d * 1e3, at(d) * 1e-3)).collect();
    (before, at(0.125), e, pts.join(", "))
}

#[test]
fn the_guard_carries_fmvss_223s_loads_before_its_attachments_fail() {
    let g = BoxTrailer::new(Default::default()).guard();
    let (p1, half) = (g.p1_x() as f32, 0.1015f32);
    let udl = (g.support_x as f32) + 0.30;
    // (name, block's x span, the regulation's minimum: S5.2.1 50 kN at P1 and P2; S5.2(c)
    // the uniform distributed load 350 kN)
    let cases = [
        ("P1 (left, 3/8 of the width out)", p1 - half, p1 + half, 50e3f32),
        ("P2 (centre)", -half, half, 50e3),
        ("uniform distributed load", -udl, udl, 350e3),
    ];
    let mut bad = Vec::new();
    let pushes: Vec<Push> = std::thread::scope(|sc| {
        let hs: Vec<_> = cases.iter().map(|&(_, x0, x1, _)| sc.spawn(move || push(x0, x1, 0.40))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for ((case, _, _, need), p) in cases.into_iter().zip(pushes) {
        let (before, at125, e125, pts) = summary(&p);
        println!(
            "{case}: largest force before an attachment broke {:.1} kN (FMVSS 223 minimum {:.0} kN); at 125 mm {:.1} kN; energy within 125 mm {:.1} kJ; first attachment broke {}",
            before * 1e-3,
            need * 1e-3,
            at125 * 1e-3,
            e125 * 1e-3,
            p.first_break.as_ref().map_or("never (0.40 m pushed)".to_string(), |b| format!("at {:.0} mm ({})", b.0 * 1e3, b.1))
        );
        println!("  {pts}");
        println!("  most set: {}", p.set.join(", "));
        if before < need {
            bad.push(format!("{case}: {:.1} kN before an attachment broke, FMVSS 223 needs {:.0} kN", before * 1e-3, need * 1e-3));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("; "));
}

/// One run of a crash reference scene on the CPU reference, its report as one JSON line
/// (`CRASH_ID`, `CRASH_CONTACT`, `CRASH_RUN`): `crash_ref` runs 0..N in order; this runs
/// run k alone, so the eight runs of a slow setting fit separate bounded jobs.
#[test]
#[ignore]
fn crash_run() {
    use engine_softbody::crash_reference::{self as cr, Contact, CpuRunner, Jitter};
    let id = std::env::var("CRASH_ID").unwrap_or("S04".into());
    let contact = std::env::var("CRASH_CONTACT").ok().and_then(|c| Contact::from_name(&c)).unwrap_or(Contact::Edge);
    let run: usize = std::env::var("CRASH_RUN").ok().and_then(|r| r.parse().ok()).unwrap_or(0);
    let s = cr::get(&id).unwrap();
    let t = engine_softbody_authoring::crash_templates::templates_for(&s).unwrap();
    let built = s.build(&t, Jitter::run(run), contact).unwrap();
    let clock = std::time::Instant::now();
    let report = cr::run(&s, &t, &built, run, &mut CpuRunner::new(&built), &mut |_| {}).unwrap();
    println!("{}", serde_json::to_string(&report).unwrap());
    eprintln!("{id} {} run {run}: {:.1} s", contact.name(), clock.elapsed().as_secs_f32());
}

/// Diagnostics for the crash reference's trailer scenes (S04, S06): every 40 ms, along the
/// road (+Z), the car's body (the attached nodes of its heaviest unbroken part): its
/// foremost node, its greenhouse's (nodes over 1.0 m at rest) foremost node and lowest
/// height, against the trailer's rear face (its rear frame's rear nodes) and its guard
/// bar; each body's COM speed; the fastest node. `CRASH_ID` (S04), `CRASH_CONTACT` (edge).
#[test]
#[ignore]
fn probe_trailer_scene() {
    use engine_softbody::crash_reference::{self as cr, Contact, CpuRunner, Jitter, Runner};
    let id = std::env::var("CRASH_ID").unwrap_or("S04".into());
    let contact = std::env::var("CRASH_CONTACT").ok().and_then(|c| Contact::from_name(&c)).unwrap_or(Contact::Edge);
    let s = cr::get(&id).unwrap();
    let t = engine_softbody_authoring::crash_templates::templates_for(&s).unwrap();
    let built = s.build(&t, Jitter::run(0), contact).unwrap();
    let names: Vec<&str> = built.layout.bodies.iter().map(|b| b.name.as_str()).collect();
    let k_car = names.iter().position(|n| *n == "car").unwrap();
    let k_tr = names.iter().position(|n| *n == "trailer").unwrap();
    let car_desc = &t["sedan"].desc;
    let tr_desc = &t["box_trailer"].desc;
    let high: Vec<usize> = (0..car_desc.nodes.len()).filter(|&i| car_desc.nodes[i].position[1] > 1.0).collect();
    let rear: Vec<usize> = (0..tr_desc.nodes.len()).filter(|&i| tr_desc.nodes[i].name.starts_with("reaf") && tr_desc.nodes[i].name.ends_with('0')).collect();
    let guard: Vec<usize> = (0..tr_desc.nodes.len()).filter(|&i| tr_desc.nodes[i].name.starts_with("gb")).collect();
    let mut r = CpuRunner::new(&built);
    let per = (0.04 / cr::DT).round() as u32;
    let n = ((built.layout.impact_s + s.secs_after_impact) / 0.04).ceil() as usize;
    for k in 0..=n {
        let x = r.sample().unwrap();
        let (cp, tp) = (&x.pos[k_car], &x.pos[k_tr]);
        let front = (0..car_desc.nodes.len()).map(|i| cp[i].z).fold(f32::MIN, f32::max);
        let gh_front = high.iter().map(|&i| cp[i].z).fold(f32::MIN, f32::max);
        let gh_low = high.iter().map(|&i| cp[i].y).fold(f32::MAX, f32::min);
        let tr_rear = rear.iter().map(|&i| tp[i].z).sum::<f32>() / rear.len() as f32;
        let g = guard.iter().map(|&i| tp[i]).sum::<Vec3>() / guard.len() as f32;
        let com = |b: usize| {
            let m: Vec<f32> = built.fleet[b].nodes.iter().map(|n| n.mass).collect();
            let mm: f32 = m.iter().sum();
            x.vel[b].iter().zip(&m).map(|(v, w)| *v * *w).sum::<Vec3>() / mm
        };
        let fastest = x.vel.iter().enumerate().flat_map(|(b, v)| v.iter().enumerate().map(move |(i, v)| (v.length(), b, i))).fold((0.0, 0, 0), |a, b| if b.0 > a.0 { b } else { a });
        let vels: Vec<String> = (0..names.len()).map(|b| format!("{} {:.2}", names[b], com(b).z)).collect();
        println!(
            "t {:.2}: car front {:+.2}, greenhouse front {:+.2} (lowest {:.2} m) from the trailer's rear face; guard at {:+.2} (y {:.2}); vz {}; fastest {} node {} {:.1} m/s",
            k as f32 * 0.04 - built.layout.impact_s,
            front - tr_rear,
            gh_front - tr_rear,
            gh_low,
            g.z - tr_rear,
            g.y,
            vels.join(", "),
            names[fastest.1],
            fastest.2,
            fastest.0
        );
        if k < n {
            r.frame(per).unwrap();
        }
    }
}

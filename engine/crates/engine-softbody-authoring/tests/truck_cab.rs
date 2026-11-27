//! The generated 4x2 tractor unit (`engine_vehiclegen::designs::truck_cab`,
//! rigged by `gen2::parts_rig`) on the CPU reference: evidence that it is
//! physically sound before it is used in the crash reference scenes.
//!
//! * rest: parked on flat ground it settles and stays (node speeds after 2 s,
//!   no beam yielded or broken, ride height stable);
//! * mass and axle split against the published figures
//!   (`generated-vehicles-lessons.md` §22: 7,500 kg, 69.8 % front);
//! * a 30 km/h full-width rigid wall: finite, the frame's shortening and the
//!   beams it bent and broke, part by part;
//! * no energy created at rest.
//!
//! Run: `cargo test --release -p engine-softbody-authoring --test truck_cab -- --nocapture`

use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, StaticBox};
use engine_softbody_authoring::authoring::gen2::vg_input;
use engine_softbody_authoring::vehicle_compare::{lowest_point, settings};
use engine_vehiclegen::designs::truck_cab;
use glam::Vec3;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

const STEP: f32 = 0.0005;

fn desc() -> &'static SoftVehicleDesc {
    static D: OnceLock<SoftVehicleDesc> = OnceLock::new();
    D.get_or_init(|| {
        let inp = vg_input::VgInput::find("truck_cab", None, None).expect("truck_cab");
        let b = vg_input::build(inp).expect("rig");
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("truck_cab_test");
        for (p, bytes) in vg_input::files(&b) {
            let path = dir.join(&p);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
        }
        let files = engine_softbody::jbeam::read_vehicle_dir(&dir, "truck_cab").unwrap();
        let pc = std::fs::read_to_string(dir.join("truck_cab/truck_cab.pc")).ok();
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).unwrap()
    })
}

fn spawn(vel: Vec3) -> ExplicitBeams {
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(desc(), Vec3::ZERO, vel, settings()).unwrap();
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
    s.nodes
        .iter()
        .all(|n| n.pos.is_finite() && n.vel.is_finite())
}

fn ids(prefix: &str) -> Vec<usize> {
    (0..nd())
        .filter(|&i| desc().nodes[i].name.starts_with(prefix))
        .collect()
}

fn centroid(s: &ExplicitBeams, ids: &[usize]) -> Vec3 {
    ids.iter().map(|&i| pos(s, i)).sum::<Vec3>() / ids.len() as f32
}

/// The vehicle's forward direction (bumper ahead of the frame's rear end).
fn forward() -> Vec3 {
    let p = |ids: &[usize]| {
        ids.iter()
            .map(|&i| Vec3::from(desc().nodes[i].position))
            .sum::<Vec3>()
            / ids.len() as f32
    };
    let d = p(&ids("bf")) - p(&ids("frL6"));
    Vec3::new(d.x, 0.0, d.z).normalize()
}

/// Kinetic energy of the nodes (J).
fn kinetic(s: &ExplicitBeams) -> f64 {
    s.nodes
        .iter()
        .map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64)
        .sum()
}

/// Each node's part (the JBeam part that lists it).
fn part_of() -> Vec<String> {
    let mut out = vec![String::new(); nd()];
    for p in &desc().parts {
        for &i in &p.nodes {
            if (i as usize) < out.len() && out[i as usize].is_empty() {
                out[i as usize] = p.name.trim_start_matches("truck_cab_").to_string();
            }
        }
    }
    out
}

/// (bent, broken) beams by the part of their first node.
fn damage(s: &ExplicitBeams) -> BTreeMap<String, (usize, usize)> {
    let part = part_of();
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for b in &s.beams {
        let a = b.a as usize;
        let name = if a < part.len() {
            part[a].clone()
        } else {
            "wheels".into()
        };
        let e = out.entry(name).or_default();
        if b.broken {
            e.1 += 1;
        } else if b.bent() {
            e.0 += 1;
        }
    }
    out.retain(|_, v| v.0 + v.1 > 0);
    out
}

#[test]
fn parked_it_settles_and_stays() {
    let mut s = spawn(Vec3::ZERO);
    let body: Vec<usize> = ids("fr");
    let mut heights = Vec::new();
    let mut vmax_late = 0.0f32;
    let mut vmax_end = (0.0f32, 0usize);
    for k in 0..(4.0 / STEP).round() as usize {
        s.step();
        let t = (k + 1) as f32 * STEP;
        if (k + 1) % 200 == 0 {
            heights.push((t, centroid(&s, &body).y));
        }
        if t >= 2.0 {
            vmax_late = vmax_late.max(
                (0..nd())
                    .map(|i| s.nodes[i].vel.length())
                    .fold(0.0, f32::max),
            );
        }
        if t >= 3.5 {
            for i in 0..nd() {
                if s.nodes[i].vel.length() > vmax_end.0 {
                    vmax_end = (s.nodes[i].vel.length(), i);
                }
            }
        }
    }
    assert!(finite(&s), "state not finite");
    let h = |t0: f32| heights.iter().find(|x| x.0 >= t0 - 1e-4).unwrap().1;
    let (h2, h4) = (h(2.0), h(4.0));
    let dmg = damage(&s);
    println!("parked 4 s: max node speed after 2 s {vmax_late:.4} m/s, after 3.5 s {:.4} m/s ({}); frame height at 2 s {:.4} m, at 4 s {:.4} m (change {:.2} mm); bent/broken {dmg:?}", vmax_end.0, desc().nodes[vmax_end.1].name, h2, h4, (h4 - h2) * 1e3);
    assert!(
        vmax_end.0 < 0.025,
        "a node still moves at {:.3} m/s after 3.5 s",
        vmax_end.0
    );
    assert!(
        vmax_late < 0.05,
        "a node still moves at {vmax_late:.3} m/s after 2 s"
    );
    assert!(
        (h4 - h2).abs() < 0.002,
        "ride height drifts {:.1} mm from 2 to 4 s",
        (h4 - h2) * 1e3
    );
    assert!(dmg.is_empty(), "beams yielded or broke parked: {dmg:?}");
}

#[test]
fn mass_and_axle_split_match_the_published_figures() {
    let mut s = spawn(Vec3::ZERO);
    for _ in 0..(3.0 / STEP).round() as usize {
        s.step();
    }
    let nodes: f32 = desc().nodes.iter().map(|n| n.mass).sum();
    let wheels: f32 = desc().wheels.iter().map(|w| w.mass).sum();
    let total = nodes + wheels;
    // the ground's reaction under each axle (the raycast wheels' normal loads)
    let d = s.drive.as_ref().expect("raycast wheels");
    let fw = forward();
    let hub = |w: &engine_softbody::explicit_drive::DriveWheel| {
        0.5 * (pos(&s, w.node1 as usize) + pos(&s, w.node2 as usize))
    };
    let zf = d
        .config
        .wheels
        .iter()
        .map(|w| hub(w).dot(fw))
        .fold(f32::MIN, f32::max);
    let (mut front, mut rear) = (0.0f32, 0.0f32);
    for (w, st) in d.config.wheels.iter().zip(&d.wheels) {
        if hub(w).dot(fw) > zf - 0.5 {
            front += st.load;
        } else {
            rear += st.load;
        }
    }
    let g = 9.81;
    let (kf, kr) = (front / g, rear / g);
    let share = kf / (kf + kr);
    let target_f = truck_cab::KERB as f32 * truck_cab::FRONT_SHARE as f32;
    let target_r = truck_cab::KERB as f32 - target_f;
    println!(
        "mass {total:.0} kg (nodes {nodes:.0} + raycast wheels {wheels:.0}; target {:.0}); axle loads parked {kf:.0} / {kr:.0} kg, front {:.1} % (target {target_f:.0} / {target_r:.0} kg, {:.1} %)",
        truck_cab::KERB,
        100.0 * share,
        100.0 * truck_cab::FRONT_SHARE
    );
    assert!(
        (total / truck_cab::KERB as f32 - 1.0).abs() < 0.01,
        "mass {total:.0} kg"
    );
    assert!(total >= 5.0 * 1410.0, "not five sedans: {total:.0} kg");
    assert!(
        (kf / target_f - 1.0).abs() <= 0.05,
        "front axle {kf:.0} kg against {target_f:.0}"
    );
    assert!(
        (kr / target_r - 1.0).abs() <= 0.05,
        "rear axle {kr:.0} kg against {target_r:.0}"
    );
}

#[test]
fn a_30_kmh_full_width_wall_is_taken_by_the_frame() {
    let v = 30.0 / 3.6;
    let fw = forward();
    let mut s = spawn(fw * v);
    // settle-free spawn: the wall 0.25 m ahead of the bumper's front
    let front = (0..nd())
        .map(|i| pos(&s, i).dot(fw))
        .fold(f32::MIN, f32::max);
    let c = fw * (front + 0.25 + 0.5) + Vec3::new(0.0, 2.0, 0.0);
    let half = Vec3::new(
        if fw.x.abs() > 0.5 { 0.5 } else { 4.0 },
        2.0,
        if fw.z.abs() > 0.5 { 0.5 } else { 4.0 },
    );
    s.boxes.push(StaticBox::centered(c, half));
    let rails_f: Vec<usize> = ["frL0", "frR0"].iter().flat_map(|p| ids(p)).collect();
    let rails_r: Vec<usize> = ["frL6", "frR6"].iter().flat_map(|p| ids(p)).collect();
    let bumper = ids("bf");
    let cab_f: Vec<usize> = ids("cab0");
    let cab_r: Vec<usize> = ids("cab2");
    let len =
        |s: &ExplicitBeams, a: &[usize], b: &[usize]| (centroid(s, a) - centroid(s, b)).dot(fw);
    let (rail0, bump0, cab0) = (
        len(&s, &rails_f, &rails_r),
        len(&s, &bumper, &rails_r),
        len(&s, &cab_f, &cab_r),
    );
    let (mut rail_min, mut bump_min, mut cab_min) = (rail0, bump0, cab0);
    let mut ke_max_after = 0.0f64;
    let ke0 = kinetic(&s);
    for k in 0..(1.5 / STEP).round() as usize {
        s.step();
        if k % 20 == 0 {
            rail_min = rail_min.min(len(&s, &rails_f, &rails_r));
            bump_min = bump_min.min(len(&s, &bumper, &rails_r));
            cab_min = cab_min.min(len(&s, &cab_f, &cab_r));
        }
        if (k as f32) * STEP > 0.3 {
            ke_max_after = ke_max_after.max(kinetic(&s));
        }
    }
    assert!(finite(&s), "state not finite after the wall");
    let dmg = damage(&s);
    let speed = centroid(&s, &rails_r).dot(fw);
    let _ = speed;
    let (rail1, bump1, cab1) = (
        len(&s, &rails_f, &rails_r),
        len(&s, &bumper, &rails_r),
        len(&s, &cab_f, &cab_r),
    );
    println!(
        "wall 30 km/h ({:.0} kJ): rails shortened {:.1} mm dynamic, {:.1} mm residual; bumper to frame end {:.1} / {:.1} mm; cab {:.1} / {:.1} mm; KE after 0.3 s at most {:.1} kJ",
        ke0 * 1e-3,
        (rail0 - rail_min) * 1e3,
        (rail0 - rail1) * 1e3,
        (bump0 - bump_min) * 1e3,
        (bump0 - bump1) * 1e3,
        (cab0 - cab_min) * 1e3,
        (cab0 - cab1) * 1e3,
        ke_max_after * 1e-3
    );
    println!("  bent / broken by part: {dmg:?}");
    let name = |i: u32| {
        desc()
            .nodes
            .get(i as usize)
            .map_or("wheel", |n| n.name.as_str())
    };
    let broken: Vec<String> = s
        .beams
        .iter()
        .filter(|b| b.broken)
        .map(|b| format!("{}-{}", name(b.a), name(b.b)))
        .collect();
    println!("  broken: {}", broken.join(", "));
    // The frame: a car crushes 0.45-0.6 m at 56 km/h; the frame's own rails
    // shorten by a few centimetres at 30 km/h.
    assert!(
        rail0 - rail_min < 0.08,
        "the rails shortened {:.0} mm",
        (rail0 - rail_min) * 1e3
    );
    let frame_broken: usize = dmg
        .iter()
        .filter(|(k, _)| k.starts_with("frame_rail"))
        .map(|(_, v)| v.1)
        .sum();
    assert_eq!(frame_broken, 0, "frame beams broke: {dmg:?}");
    assert!(
        ke_max_after < ke0,
        "kinetic energy rose past the impact's: {ke_max_after:.0} J of {ke0:.0} J"
    );
}

#[test]
fn no_energy_is_created_at_rest() {
    let mut s = spawn(Vec3::ZERO);
    let mut ke = Vec::new();
    for k in 0..(6.0 / STEP).round() as usize {
        s.step();
        if (k + 1) % 100 == 0 {
            ke.push(kinetic(&s));
        }
    }
    // windows of one second from 1 s: each one's peak no higher than the one before
    let peak = |a: usize, b: usize| ke[a..b].iter().cloned().fold(0.0, f64::max);
    let peaks: Vec<f64> = (1..6).map(|w| peak(w * 20, (w + 1) * 20)).collect();
    println!(
        "parked: kinetic energy peak per second from 1 s (J): {:?}",
        peaks.iter().map(|p| format!("{p:.3}")).collect::<Vec<_>>()
    );
    for w in 1..peaks.len() {
        assert!(
            peaks[w] <= peaks[w - 1] * 1.05 + 0.05,
            "kinetic energy grew at rest: {:.3} J after {:.3} J",
            peaks[w],
            peaks[w - 1]
        );
    }
    assert!(finite(&s));
}

/// Diagnostics: the fastest nodes 3 s after spawn and the fastest one's
/// reversals in the next second (its mode's frequency × 2).
#[test]
#[ignore]
fn probe_ringing() {
    let mut s = spawn(Vec3::ZERO);
    for _ in 0..(3.0 / STEP).round() as usize {
        s.step();
    }
    let mut v: Vec<(f32, usize)> = (0..nd()).map(|i| (s.nodes[i].vel.length(), i)).collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (sp, i) in v.iter().take(12) {
        println!("{} {:.4} {:?}", desc().nodes[*i].name, sp, s.nodes[*i].vel);
    }
    let i = v[0].1;
    let mut last = s.nodes[i].vel;
    let mut cross = 0;
    for _ in 0..2000 {
        s.step();
        let now = s.nodes[i].vel;
        if now.dot(last) < 0.0 {
            cross += 1;
        }
        last = now;
    }
    println!("{}: {} reversals in 1 s", desc().nodes[i].name, cross);
}

/// A measurement, not adopted (an engine decision): the solver sizes every
/// raycast spring for the vehicle's mean wheel load (`pack::core_plan`), so a
/// front-heavy truck parks nose-down past its design (`every_design`'s lowest
/// point). The same truck with each wheel's spring and damper scaled to its
/// own static load (the same frequency and ratio on every wheel).
#[test]
#[ignore]
fn measure_suspension_sized_per_wheel_load() {
    let park = |scale: Option<Vec<f32>>| {
        let mut s = spawn(Vec3::ZERO);
        if let (Some(sc), Some(d)) = (scale, s.drive.as_mut()) {
            for (w, f) in d.config.wheels.iter_mut().zip(sc) {
                w.k *= f;
                w.c *= f;
            }
        }
        for _ in 0..(3.0 / STEP).round() as usize {
            s.step();
        }
        let loads: Vec<f32> = s
            .drive
            .as_ref()
            .unwrap()
            .wheels
            .iter()
            .map(|w| w.load)
            .collect();
        (lowest_point(&s), loads)
    };
    let (low0, loads) = park(None);
    let mean = loads.iter().sum::<f32>() / loads.len() as f32;
    let scale: Vec<f32> = loads.iter().map(|l| l / mean).collect();
    let (low1, _) = park(Some(scale.clone()));
    println!("lowest point parked: mean-load springs {:.0} mm; per-wheel-load springs (scale {scale:?}) {:.0} mm", low0 * 1e3, low1 * 1e3);
}

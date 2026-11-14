//! The generated sedan's hinges bend before they tear (the user's report,
//! 2026-11-14, on the lab's mountain: the hood and the boot lid "push in, break
//! the hinge, and fall into the car… what you'd expect is they get held there
//! and get crumpled").
//!
//! Measured cause (`examples/hinge_breaks`, `engine/docs/measurements/hinge-breaks.md`):
//! every hinge beam was NORMAL with no yield (`beamDeform` FLT_MAX), so the first
//! beam of a group to reach its strength (17–29 mm of stretch at k 1.5e6) broke
//! the whole group while the panel was barely bent (doors 2–6 %, boot 3 % largest
//! plastic strain), and the beam that broke first was usually a long one (0.5–0.77 m)
//! to a far body node, one of them on the rear rail's crush zone.
//!
//! The rules, from `engine/docs/beamng-authoring-census.md` §3 (BeamNG door hinges:
//! k 0.5–3e6 N/m, deform 10–77 kN, strength 30–98 kN a beam; bonnet 35–155 kN; boot
//! 10–35 kN):
//!
//! - **Every hinge beam yields before it breaks:** a finite `beamDeform` in the
//!   census's 10–77 kN, below its `beamStrength`, and a bounded bend
//!   (`deformLimitExpansion`).
//! - **Strengths in the census band** per closure.
//! - **Anchored near the hinge, on the body's structure:** each hinge beam's body
//!   node within [`HINGE_REACH`] of the hinge's marker and not on a crush member
//!   (rails, crush cans, bumper beam, aprons, rear rails), whose length changes
//!   when that zone folds whatever the lid does.
//! - **Behaviour (the solver's beam law, as spawned):** every hinge beam
//!   stretched by [`TRAVEL_MM`], and squeezed by as much (at most half its
//!   length), quasi-statically: it yields, and it does not break. (A whole-car
//!   check is the mountain run of `engine-world/examples/hinge_breaks.rs`.)
use engine_softbody::authoring::gen2::build::Role;
use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{
    jbeam_force_filtered, Beam, ExplicitBeams, ExplicitBeamsSettings, Node,
};
use glam::Vec3;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A hinge beam's body node is at most this far from the hinge's marker (m):
/// the nearest structural nodes of the body's lattice (spacing 0.2–0.45 m)
/// that still span three directions.
const HINGE_REACH: f32 = 0.45;

/// The travel (mm) a hinge beam bends through without breaking: the measured
/// hinges broke at 17–29 mm of one beam's stretch.
const TRAVEL_MM: f32 = 40.0;

struct Sedan {
    desc: SoftVehicleDesc,
    /// Hinge marker → its position in the loaded car's frame.
    markers: HashMap<String, Vec3>,
    /// Node name → its member role on the body (None: not a member node).
    member: HashMap<String, Option<Role>>,
}

fn sedan() -> &'static Sedan {
    static D: OnceLock<Sedan> = OnceLock::new();
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
        let desc = engine_softbody::jbeam::load_jbeam(&refs, pc).expect("the sedan's jbeam");
        use engine_softbody::authoring::gen2::rig::RigInput;
        let markers = inp
            .model()
            .markers
            .iter()
            .filter(|m| m.name.starts_with("hinge_"))
            .map(|m| {
                // JBeam (x, y, z) → the loaded car's (x, z, −y) (`load_jbeam`).
                let j = r.to_jbeam(m.pos);
                (m.name.clone(), Vec3::new(j.x, j.z, -j.y))
            })
            .collect();
        let member = r
            .car
            .nodes
            .iter()
            .zip(&r.member)
            .map(|(n, m)| (n.name.clone(), *m))
            .collect();
        Sedan {
            desc,
            markers,
            member,
        }
    })
}

/// A hinge group: its name, closure part, marker, beams (desc indices).
struct Hinge {
    name: String,
    part: String,
    marker: String,
    beams: Vec<usize>,
}

fn hinges(d: &SoftVehicleDesc) -> Vec<Hinge> {
    let mut out: Vec<Hinge> = Vec::new();
    for (g, name) in d.break_groups.iter().enumerate() {
        if !name.contains("hinge") {
            continue;
        }
        let beams: Vec<usize> = (0..d.beams.len())
            .filter(|&j| d.beams[j].break_group == g as u32 + 1)
            .collect();
        // hood_hinge_L → hinge_hood_L; door_FL_hinge_upper → hinge_door_FL_upper.
        let (closure, side) = name.split_once("_hinge_").expect("<closure>_hinge_<side>");
        let part = match closure {
            "hood" => "sedan_hood".to_string(),
            "trunk" => "sedan_trunk".to_string(),
            door => format!("sedan_{door}"),
        };
        out.push(Hinge {
            name: name.clone(),
            part,
            marker: format!("hinge_{closure}_{side}"),
            beams,
        });
    }
    assert_eq!(
        out.len(),
        12,
        "four doors × 2, hood × 2, boot × 2 hinge groups"
    );
    out
}

fn part_nodes(d: &SoftVehicleDesc, part: &str) -> Vec<u32> {
    d.part(part)
        .unwrap_or_else(|| panic!("no part {part}"))
        .nodes
        .clone()
}

/// (the beam's node in the closure, its node on the body)
fn ends(d: &SoftVehicleDesc, h: &Hinge, j: usize) -> (u32, u32) {
    let own = part_nodes(d, &h.part);
    let b = &d.beams[j];
    if own.contains(&b.a) {
        (b.a, b.b)
    } else {
        assert!(
            own.contains(&b.b),
            "{}: beam {j} has no end on {}",
            h.name,
            h.part
        );
        (b.b, b.a)
    }
}

#[test]
fn every_hinge_beam_yields_before_it_breaks() {
    let s = sedan();
    let mut bad = Vec::new();
    for h in hinges(&s.desc) {
        for &j in &h.beams {
            let b = &s.desc.beams[j];
            let lim = b.jbeam.map_or(0.0, |x| x.deform_limit_expansion);
            if !(b.deform.is_finite()
                && b.deform < b.strength
                && (10e3..=77e3).contains(&b.deform)
                && lim > 1.0
                && lim.is_finite())
            {
                bad.push(format!(
                    "{} beam {j}: deform {:.0} N, strength {:.0} N, deformLimitExpansion {lim}",
                    h.name, b.deform, b.strength
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "hinge beams that cannot yield before breaking (census: deform 10–77 kN below the strength, a bounded bend):\n{}",
        bad.join("\n")
    );
}

#[test]
fn hinge_strengths_in_the_census_band() {
    let s = sedan();
    let mut bad = Vec::new();
    for h in hinges(&s.desc) {
        let band = match h.part.as_str() {
            "sedan_hood" => 35e3..=155e3,
            "sedan_trunk" => 10e3..=35e3,
            _ => 30e3..=98e3,
        };
        for &j in &h.beams {
            let st = s.desc.beams[j].strength;
            if !band.contains(&st) {
                bad.push(format!(
                    "{} beam {j}: strength {st:.0} N, band {band:?}",
                    h.name
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn hinges_are_anchored_near_the_hinge_on_the_structure() {
    let s = sedan();
    let d = &s.desc;
    let crush = |r: Option<Role>| {
        matches!(
            r,
            Some(Role::Rail | Role::CrushCan | Role::BumperBeam | Role::Apron | Role::RearRail)
        )
    };
    let mut bad = Vec::new();
    for h in hinges(d) {
        let mk = s.markers[&h.marker];
        for &j in &h.beams {
            let (_, body) = ends(d, &h, j);
            let n = &d.nodes[body as usize];
            let dist = Vec3::from(n.position).distance(mk);
            let role = s.member.get(&n.name).copied().flatten();
            if dist > HINGE_REACH || crush(role) {
                bad.push(format!(
                    "{}: body node {} {:.2} m from the hinge, member {role:?}",
                    h.name, n.name, dist
                ));
            }
        }
    }
    bad.sort();
    bad.dedup();
    assert!(
        bad.is_empty(),
        "hinge anchors beyond {HINGE_REACH} m or on a crush member:\n{}",
        bad.join("\n")
    );
}

/// A hinge beam on its own, as the solver loads it, stretched (or squeezed)
/// quasi-statically by up to `mm` (`keep`: its strength does not fall as it
/// stretches, the JBeam rule): (the stretch it first yields at, the
/// stretch it breaks at), each in mm (NaN: never).
fn pull(beam: &Beam, mm: f32, sign: f32, keep: bool) -> (f32, f32) {
    let mut b = *beam;
    let node = |p: Vec3| Node {
        pos: p,
        vel: Vec3::ZERO,
        force: Vec3::ZERO,
        mass: 1.0,
        friction: 0.0,
        flags: 0,
    };
    let (mut yielded, mut broke) = (f32::NAN, f32::NAN);
    let n = (mm / 0.05).round() as usize;
    for k in 1..=n {
        let x = sign * k as f32 * 0.05e-3;
        let (na, nb) = (node(Vec3::ZERO), node(Vec3::X * (beam.rest0 + x)));
        let out = jbeam_force_filtered(&mut b, &na, &nb, false, None, keep);
        if yielded.is_nan() && b.rest != beam.rest0 {
            yielded = x.abs() * 1e3;
        }
        if out.check_break {
            broke = x.abs() * 1e3;
            break;
        }
    }
    (yielded, broke)
}

#[test]
fn hinge_beams_bend_through_the_travel_before_they_tear() {
    // Every hinge beam, as spawned, stretched by TRAVEL_MM (and squeezed by as
    // much, at most half its length): it yields, and it does not break.
    let s = sedan();
    let d = &s.desc;
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        ..Default::default()
    };
    let (sim, _) =
        ExplicitBeams::from_desc_jbeam_raycast(d, Vec3::ZERO, Vec3::ZERO, settings).expect("spawn");
    let mut bad = Vec::new();
    let mut n = 0;
    for h in hinges(d) {
        for &j in &h.beams {
            let b = &sim.beams[j];
            for (way, sign, mm) in [
                ("stretched", 1.0, TRAVEL_MM),
                ("squeezed", -1.0, TRAVEL_MM.min(500.0 * b.rest0)),
            ] {
                n += 1;
                let (y, br) = pull(b, mm, sign, !sim.plastic_strength_drop);
                if y.is_nan() || !br.is_nan() {
                    bad.push(format!(
                        "{} beam {j} (L0 {:.3} m) {way} {mm:.0} mm: yields at {y:.1} mm, breaks at {br:.1} mm",
                        h.name, b.rest0
                    ));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {n} hinge-beam pulls do not bend through {TRAVEL_MM} mm without tearing:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

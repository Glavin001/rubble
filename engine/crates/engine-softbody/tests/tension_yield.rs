//! `rubble_tension_yield` (jbeam-semantics-spec.md §18.4), from its definition:
//! a line sets the named deformable beam's tension threshold Tt and nothing
//! else; a file without the section loads as before; a line naming no
//! deformable plain beam fails the load. (The GPU's parity:
//! engine-webgpu `tension_yield_gpu`.)

use engine_softbody::explicit_beams::*;
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
};

fn load(text: &str) -> Result<ExplicitBeams, String> {
    ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, EXACT)
}

/// The file with `section` before its closing `end`.
fn with(text: &str, section: &str) -> String {
    let at = text.rfind("\nend").expect("end");
    format!("{}\n{section}{}", &text[..at], &text[at..])
}

#[test]
fn a_line_sets_only_its_beam_s_tension_yield() {
    let text = car_truck_bare(&CarParams::sedan());
    let before = load(&text).unwrap();
    let j = before
        .beams
        .iter()
        .position(|b| b.kind == ExplicitKind::Plain && b.ty == ExplicitType::Deformable)
        .unwrap();
    let (a, b) = (before.beams[j].a, before.beams[j].b);
    let tt = 4.0 * before.beams[j].tc;
    let after = load(&with(
        &text,
        &format!("rubble_tension_yield\n{b}, {a}, {tt}\n"),
    ))
    .unwrap();
    for (k, (x, y)) in before.beams.iter().zip(&after.beams).enumerate() {
        if k == j {
            assert_eq!(y.tt, tt);
            assert_eq!(y.tc, x.tc);
            assert_eq!(y.gate, x.tc.min(tt).min(x.strength));
        } else {
            assert_eq!(x, y, "beam {k} changed");
        }
    }
}

#[test]
fn a_line_naming_no_deformable_beam_fails() {
    let text = car_truck_bare(&CarParams::sedan());
    let n = load(&text).unwrap().nodes.len() as u32;
    let s = load(&text).unwrap();
    let joined = |a: u32, b: u32| {
        s.beams
            .iter()
            .any(|x| (x.a.min(x.b), x.a.max(x.b)) == (a.min(b), a.max(b)))
    };
    let b = (1..n).find(|&b| !joined(0, b)).unwrap();
    let err = load(&with(
        &text,
        &format!("rubble_tension_yield\n0, {b}, 1000\n"),
    ));
    assert!(err.is_err());
}

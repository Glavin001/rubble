//! `truck_export::desc_to_truck`: a desc written as `.truck` and read back by
//! Rubble's `.truck` loader is the same vehicle.

use engine_softbody::desc::BeamKind;
use engine_softbody::hires::{hires_car_jbeam, HiresParams};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::truck::load_truck;
use engine_softbody::truck_export::{desc_to_truck, TyreDefaults};

#[test]
fn jbeam_car_round_trips_through_truck() {
    let (jbeam, _) = hires_car_jbeam(&HiresParams::default());
    let desc = load_jbeam(&[("hires.jbeam", &jbeam)], None).unwrap();
    let (text, _) = desc_to_truck(&desc, TyreDefaults::for_vehicle(&desc));
    let back = load_truck(&text);

    // Node positions, masses, ground contact and friction; the tyres come after.
    assert!(back.nodes.len() >= desc.nodes.len());
    for (i, (a, b)) in desc.nodes.iter().zip(&back.nodes).enumerate() {
        for k in 0..3 {
            assert!(
                (a.position[k] - b.position[k]).abs() < 1e-5,
                "node {i} position"
            );
        }
        assert!(
            (a.mass - b.mass).abs() <= 1e-5 * a.mass,
            "node {i} mass {} / {}",
            a.mass,
            b.mass
        );
        assert_eq!(a.collision, b.collision, "node {i} ground contact");
        assert_eq!(a.friction, b.friction, "node {i} friction");
    }
    // Beams other than bounded and hydro keep their order and values.
    let plain: Vec<_> = desc
        .beams
        .iter()
        .filter(|b| !matches!(b.kind, BeamKind::Bounded | BeamKind::Hydro))
        .collect();
    assert!(back.beams.len() >= plain.len());
    for (i, (a, b)) in plain.iter().zip(&back.beams).enumerate() {
        assert_eq!((a.a, a.b), (b.a, b.b), "beam {i} nodes");
        let kind = if a.kind == BeamKind::Anisotropic {
            BeamKind::Normal
        } else {
            a.kind
        };
        assert_eq!(kind, b.kind, "beam {i} kind");
        assert_eq!(a.spring, b.spring, "beam {i} spring");
        assert_eq!(a.damp, b.damp, "beam {i} damping");
        let same = |x: f32, y: f32| (x.is_infinite() && y >= 1e29) || x == y;
        assert!(
            same(a.deform, b.deform),
            "beam {i} deform {} / {}",
            a.deform,
            b.deform
        );
        assert!(
            same(a.strength, b.strength),
            "beam {i} strength {} / {}",
            a.strength,
            b.strength
        );
        assert_eq!(a.plastic_coef, b.plastic_coef, "beam {i} plastic");
        assert_eq!(a.break_group, b.break_group, "beam {i} break group");
        if a.kind == BeamKind::Support {
            assert_eq!(a.long_bound, b.long_bound, "beam {i} support bound");
        }
    }
    // Triangles with their collision flag, contacters, the reference frame.
    assert_eq!(desc.triangles.len(), back.triangles.len());
    for (a, b) in desc.triangles.iter().zip(&back.triangles) {
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.collision, b.collision);
    }
    assert_eq!(
        desc.refs.as_ref().map(|r| (r.center, r.back, r.left)),
        back.refs.as_ref().map(|r| (r.center, r.back, r.left))
    );
    assert_eq!(desc.wheels.len(), back.wheels.len());
}

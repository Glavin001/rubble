//! LBEAMs, torsion bars and slidenodes (BeamNG-parity plan, phase 3) through the
//! JBeam importer and the explicit oracle, on small original fixtures.

use engine_softbody::explicit_ref::{ExplicitSettings, ExplicitSim};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::{BeamKind, ElementKind};
use glam::{DVec3, Quat, Vec3};

fn sim(d: &engine_softbody::SoftVehicleDesc) -> ExplicitSim {
    let mut s = ExplicitSim::new(ExplicitSettings {
        damping_scale: 1.0,
        ..Default::default()
    });
    s.add_vehicle(d, Vec3::ZERO, Quat::IDENTITY, Vec3::ZERO);
    s
}

/// A horizontal three-node arm, the root pinned by two fixed nodes: an LBEAM at
/// the middle node holds it straight against gravity, where plain beams alone
/// would let it hinge.
#[test]
fn lbeam_holds_an_arm_straight() {
    let jb = engine_softbody::elements::fixtures::lbeam_arm;
    let droop = |lbeam: bool| {
        let d = load_jbeam(&[("arm.jbeam", &jb(lbeam))], None).unwrap();
        assert_eq!(d.elements.len(), usize::from(lbeam));
        let mut s = sim(&d);
        s.run(1.5);
        let t = d.nodes.iter().position(|n| n.name == "t").unwrap();
        0.98 - s.x[t].y.max(-10.0) // below the rest height 1.0 (posZ), as a droop
    };
    let (with, without) = (droop(true), droop(false));
    println!("tip droop: with LBEAM {with:.3} m, without {without:.3} m");
    assert!(
        without > 0.1,
        "without the LBEAM the tip swings down ({without})"
    );
    assert!(with.abs() < 0.05, "the LBEAM holds it ({with})");
}

/// Two parallel arms joined by a torsion bar about their shared axis: twisting
/// one end needs the bar's stiffness k, and the angle returns when released.
#[test]
fn torsion_bar_resists_twist() {
    let jb = engine_softbody::elements::fixtures::TORSION_BAR;
    let d = load_jbeam(&[("tb.jbeam", jb)], None).unwrap();
    assert_eq!(d.elements.len(), 1);
    assert_eq!(d.elements[0].kind, ElementKind::Torsion);
    let mut s = sim(&d);
    // Gravity tries to swing node d about the b–c axis; the bar holds it.
    s.run(1.0);
    let dn = d.nodes.iter().position(|n| n.name == "d").unwrap();
    let sag = -s.x[dn].y;
    // Static: k·φ = m g · arm (1 m) ⇒ φ ≈ 9.81 / 500 rad ⇒ ~2 cm of drop.
    println!("torsion sag {sag:.4} m");
    assert!(sag > 0.01 && sag < 0.03, "sag {sag}");
}

/// A slidenode on a sloped capped rail slides down it under gravity and stops at
/// the cap; uncapped, it runs off the end and comes free.
#[test]
fn slidenode_slides_to_the_cap_or_off_the_end() {
    let jb = |capped: bool| engine_softbody::elements::fixtures::slide_rail(capped, 0.0);
    for capped in [true, false] {
        let d = load_jbeam(&[("r.jbeam", &jb(capped))], None).unwrap();
        assert_eq!(d.rails.len(), 1);
        assert_eq!(d.slidenodes.len(), 1);
        let mut s = sim(&d);
        let sn = d.nodes.iter().position(|n| n.name == "s").unwrap();
        let mut furthest: f64 = 0.0;
        for _ in 0..150 {
            s.run(0.01);
            furthest = furthest.max(s.x[sn].z);
        }
        let p = s.x[sn];
        println!(
            "capped {capped}: slidenode at {p:?}, furthest z {furthest:.3}, attached {}",
            s.slides[0].attached
        );
        if capped {
            // Down to the low end (Rubble: z 2), held there by the cap (it rebounds
            // off it: the cap spring has no damping here).
            assert!(s.slides[0].attached);
            assert!(furthest > 1.95 && furthest < 2.05, "furthest {furthest}");
            let _ = DVec3::ZERO;
        } else {
            assert!(!s.slides[0].attached, "ran off the uncapped end");
            assert!(p.y < -0.5, "and fell");
        }
    }
}

#[test]
fn anisotropic_beam_is_stiffer_in_tension() {
    let jb = r#"{"an": {"slotType": "main",
        "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1}, ["a", 0, 0, 0], ["b", 1, 0, 0]],
        "beams": [["id1:", "id2:"], {"beamType": "|ANISOTROPIC", "beamSpring": 1000, "springExpansion": 5000, "beamDamp": 0}, ["a", "b"]]
    }}"#;
    let d = load_jbeam(&[("an.jbeam", jb)], None).unwrap();
    assert_eq!(d.beams[0].kind, BeamKind::Anisotropic);
    assert_eq!(d.beams[0].spring_expansion, 5000.0);
    let mut b = engine_softbody::pack::pack_soft(
        &d,
        engine_softbody::pack::Placement {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        },
        &engine_softbody::SpawnOptions::default(),
        0,
        -1,
    )
    .beams[0];
    b.penalty = b.spring;
    let long = engine_softbody::model::elastic_terms(&b, 1.1, b.penalty).0;
    let short = engine_softbody::model::elastic_terms(&b, 0.9, b.penalty).0;
    assert!(
        (long - 500.0).abs() < 1e-3 && (short + 100.0).abs() < 1e-3,
        "{long} {short}"
    );
}

/// JBeam beam options: precompressionRange sets the spawn length in metres,
/// deformLimitStress caps hardening, and a breakGroupType 1 beam breaks with its
/// group without setting it off.
#[test]
fn beam_options_import_and_run() {
    let jb = r#"{"opt": {"slotType": "main",
        "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1},
            ["a", 0, 0, 0, {"fixed": true}], ["b", 1, 0, 0], ["c", 2, 0, 0]],
        "beams": [["id1:", "id2:"],
            {"beamSpring": 100000, "beamDamp": 0, "beamDeform": 5000, "beamStrength": 2000,
             "breakGroup": "g", "breakGroupType": 1, "precompressionRange": -0.1, "deformLimitStress": 6000},
            ["a", "b"],
            {"breakGroupType": 0, "precompressionRange": "", "beamStrength": "FLT_MAX", "deformLimitStress": ""},
            ["b", "c"]]
    }}"#;
    let d = load_jbeam(&[("opt.jbeam", jb)], None).unwrap();
    let (ab, bc) = (&d.beams[0], &d.beams[1]);
    assert!(
        (ab.precompression - 0.9).abs() < 1e-6,
        "{}",
        ab.precompression
    );
    assert_eq!(ab.deform_limit_stress, 6000.0);
    assert!(ab.break_group_passive && !bc.break_group_passive);
    assert_eq!(ab.break_group, bc.break_group);
    // The passive beam a-b, pulled 10 cm short at spawn with 2 kN strength, breaks
    // at once (10 kN). It must not set off its group: b-c (unbreakable) survives.
    let mut s = sim(&d);
    s.run(0.05);
    assert!(s.beams[0].broken(), "the precompressed beam breaks");
    assert!(
        !s.beams[1].broken(),
        "a passive member does not break its group"
    );
}

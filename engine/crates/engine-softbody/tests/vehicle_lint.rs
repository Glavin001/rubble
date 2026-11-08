//! `authoring::lint`'s hard rules on small synthetic cars: a sound one passes
//! every rule, and each broken variant fails exactly the rule it breaks.
use engine_softbody::authoring::lint::{lint, Expect, Finding, Level};
use engine_softbody::authoring::skin::{mat, ShowcaseVisual};
use engine_softbody::desc::{
    BeamKind, SoftBeam, SoftCoupler, SoftNode, SoftPart, SoftTriangle, SoftVehicleDesc,
};
use glam::Vec3;

/// A 1 m cube (8 nodes, 10 kg each, every pair braced), its six faces as
/// collision triangles, a break-group beam (a "hinge") and a coupler (a
/// "latch").
fn cube(k: f32) -> SoftVehicleDesc {
    let mut d = SoftVehicleDesc::default();
    for i in 0..8 {
        let p = [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32];
        d.nodes.push(SoftNode {
            name: format!("n{i}"),
            position: p,
            mass: 10.0,
            ..Default::default()
        });
    }
    for a in 0..8u32 {
        for b in a + 1..8 {
            d.beams.push(SoftBeam {
                a,
                b,
                kind: BeamKind::Normal,
                spring: k,
                damp: 10.0,
                deform: 1e4,
                strength: f32::INFINITY,
                ..Default::default()
            });
        }
    }
    d.break_groups = vec!["door_hinge".into()];
    d.beams[0].break_group = 1;
    d.couplers.push(SoftCoupler {
        a: 6,
        b: 7,
        strength: 5e4,
        ..Default::default()
    });
    // Faces: (axis, side) with their four corners.
    for (ax, s) in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)] {
        let c: Vec<u32> = (0..8u32).filter(|i| (i >> ax) & 1 == s).collect();
        // c is in bit order: 0,1,2,3 make the face's quad as 0-1-3-2.
        d.triangles.push(SoftTriangle {
            nodes: [c[0], c[1], c[3]],
            ..Default::default()
        });
        d.triangles.push(SoftTriangle {
            nodes: [c[0], c[3], c[2]],
            ..Default::default()
        });
    }
    d.parts.push(SoftPart {
        name: "test_body".into(),
        nodes: (0..8).collect(),
    });
    d
}

/// The cube's skin as a render mesh: a 5 cm grid of vertices over every face
/// (painted, normals out).
fn skin(d: &SoftVehicleDesc) -> ShowcaseVisual {
    let mut v = ShowcaseVisual {
        node_rest: d.nodes.iter().map(|n| n.position).collect(),
        ..Default::default()
    };
    for ax in 0..3 {
        for s in [0.0f32, 1.0] {
            for i in 0..=20 {
                for j in 0..=20 {
                    let mut p = [0.0f32; 3];
                    p[ax] = s;
                    p[(ax + 1) % 3] = i as f32 / 20.0;
                    p[(ax + 2) % 3] = j as f32 / 20.0;
                    let mut n = [0.0f32; 3];
                    n[ax] = if s > 0.5 { 1.0 } else { -1.0 };
                    v.positions.push(p);
                    v.normals.push(n);
                    v.material.push(mat::PAINT);
                }
            }
        }
    }
    v.pieces = vec![("body_shell".into(), 0, v.positions.len() as u32)];
    v
}

fn expect() -> Expect {
    Expect {
        mass: Some(80.0),
        markers: vec![
            ("hinge_door_FL_upper".into(), Vec3::new(0.5, 0.0, 0.0)),
            ("latch_door_FL".into(), Vec3::new(1.0, 1.0, 1.0)),
        ],
    }
}

fn level(f: &[Finding], rule: &str) -> Level {
    f.iter().find(|x| x.rule == rule && x.hard).unwrap().level
}

fn hard_fails(f: &[Finding]) -> Vec<&'static str> {
    f.iter()
        .filter(|x| x.hard && x.level == Level::Fail)
        .map(|x| x.rule)
        .collect()
}

#[test]
fn a_sound_car_passes_every_hard_rule() {
    let d = cube(1e5);
    let f = lint(&d, Some(&skin(&d)), &expect());
    assert!(hard_fails(&f).is_empty(), "{:?}", hard_fails(&f));
    for r in [
        "free_mechanism",
        "stability",
        "mass",
        "hardware",
        "skin_coverage",
    ] {
        assert_eq!(level(&f, r), Level::Pass, "{r}");
    }
}

#[test]
fn a_hole_in_the_skin_fails_coverage() {
    let mut d = cube(1e5);
    // Drop a side face's two triangles (x = 0; the lint's outer skin faces
    // away from the car's long axis, z here): its centre is 0.5 m from the
    // others.
    d.triangles.drain(0..2);
    let f = lint(&d, Some(&skin(&d)), &expect());
    assert_eq!(hard_fails(&f), vec!["skin_coverage"]);
}

#[test]
fn a_free_node_fails_mechanism() {
    let mut d = cube(1e5);
    // A node on two beams: free to swing about their axis.
    d.nodes.push(SoftNode {
        name: "loose".into(),
        position: [0.5, 0.5, 1.5],
        mass: 1.0,
        ..Default::default()
    });
    for b in [4u32, 5] {
        d.beams.push(SoftBeam {
            a: 8,
            b,
            kind: BeamKind::Normal,
            spring: 1e5,
            damp: 10.0,
            deform: 1e4,
            strength: f32::INFINITY,
            ..Default::default()
        });
    }
    let mut ex = expect();
    ex.mass = Some(81.0);
    let f = lint(&d, None, &ex);
    assert_eq!(level(&f, "free_mechanism"), Level::Fail);
    assert!(f
        .iter()
        .any(|x| x.rule == "free_mechanism" && x.offenders.iter().any(|o| o.starts_with("loose"))));
}

#[test]
fn a_loose_piece_fails_mechanism() {
    let mut d = cube(1e5);
    // A second braced triangle with no beam to the cube.
    for (i, p) in [
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [3.5, 1.0, 0.0],
        [3.5, 0.5, 1.0],
    ]
    .iter()
    .enumerate()
    {
        d.nodes.push(SoftNode {
            name: format!("p{i}"),
            position: *p,
            mass: 1.0,
            ..Default::default()
        });
    }
    for a in 8..12u32 {
        for b in a + 1..12 {
            d.beams.push(SoftBeam {
                a,
                b,
                kind: BeamKind::Normal,
                spring: 1e5,
                ..Default::default()
            });
        }
    }
    let f = lint(&d, None, &Expect::default());
    assert_eq!(level(&f, "free_mechanism"), Level::Fail);
}

#[test]
fn a_too_stiff_car_fails_stability() {
    // ω·h > 2: k far past the step's bound for 10 kg nodes.
    let d = cube(5e8);
    let f = lint(&d, Some(&skin(&d)), &expect());
    assert_eq!(hard_fails(&f), vec!["stability"]);
}

#[test]
fn a_car_off_its_mass_fails() {
    let d = cube(1e5);
    let mut ex = expect();
    ex.mass = Some(100.0); // 80 kg is 20 % light
    let f = lint(&d, Some(&skin(&d)), &ex);
    assert_eq!(hard_fails(&f), vec!["mass"]);
}

#[test]
fn a_marker_without_hardware_fails() {
    let d = cube(1e5);
    let mut ex = expect();
    ex.markers
        .push(("bolt_fender_FL_1".into(), Vec3::new(5.0, 5.0, 5.0)));
    let f = lint(&d, Some(&skin(&d)), &ex);
    assert_eq!(hard_fails(&f), vec!["hardware"]);
    assert!(f
        .iter()
        .any(|x| x.rule == "hardware" && x.offenders == vec!["bolt_fender_FL_1".to_string()]));
}

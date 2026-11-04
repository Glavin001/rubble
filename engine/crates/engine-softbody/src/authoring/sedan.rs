//! The showcase sedan: a 4.7 m mid-size four-door saloon (front engine, front
//! drive), 1,450 kg with its driver. Every number here is an editable design
//! input; `engine/docs/vehicle-authoring.md` walks through them.

use super::body::{BodyProfile, Well};
use super::curve::{Curve, Outline};
use super::material::Grades;
use super::members::{Corner, MemberSpec};
use super::vehicle::{
    BlockSpec, Edge, Joint, Look, NamedMass, Panel, PlateSpec, Region, VehicleDef, Zone,
};

/// A member (left side when `mirrored`).
#[allow(clippy::too_many_arguments)]
pub fn member(
    name: &str,
    path: &[(f32, f32, f32)],
    width: f32,
    height: f32,
    up: [f32; 3],
    grade: &str,
    scale: [f32; 2],
    beads: usize,
    mirrored: bool,
    attach: usize,
) -> MemberSpec {
    MemberSpec {
        name: name.into(),
        path: path.iter().map(|&(x, y, z)| [x, y, z]).collect(),
        width,
        height,
        up,
        grade: grade.into(),
        scale,
        beads,
        mirrored,
        attach,
        ride: false,
        section: 4,
        station: 0.0,
    }
}

/// A member with its own station spacing.
pub fn stations(mut m: MemberSpec, spacing: f32) -> MemberSpec {
    m.station = spacing;
    m
}

/// A two-node (ladder) member.
pub fn ladder(mut m: MemberSpec) -> MemberSpec {
    m.section = 2;
    m
}

/// A frame riding a panel (lid frames, door beams).
pub fn riding(mut m: MemberSpec) -> MemberSpec {
    m.ride = true;
    m
}

pub const UP: [f32; 3] = [0.0, 1.0, 0.0];
pub const OUT: [f32; 3] = [1.0, 0.0, 0.0];
pub const FWD: [f32; 3] = [0.0, 0.0, 1.0];

/// A cross member at station `z`, height `y`, from −x to x.
pub fn cross(name: &str, x: f32, y: f32, z: f32, grade: &str, scale: f32) -> MemberSpec {
    ladder(member(
        name,
        &[(-x, y, z), (x, y, z)],
        0.10,
        0.12,
        UP,
        grade,
        [scale, scale],
        0,
        false,
        2,
    ))
}

pub fn panel(name: &str, region: Region, grade: &str, joint: Joint, sided: bool) -> Panel {
    Panel {
        name: name.into(),
        region,
        grade: grade.into(),
        joint,
        sided,
    }
}

pub fn door_joint() -> Joint {
    Joint::Hinged {
        hinge: Edge::Front,
        hinge_scale: 1.5,
        latch_scale: 1.2,
        latch_y: Some([0.5, 1.0]),
        latch_x: None,
    }
}

pub fn sedan() -> VehicleDef {
    let length = 4.70;
    let profile = BodyProfile {
        length,
        top: Curve::new(&[
            (0.00, 0.99),
            (0.20, 1.02),
            (0.55, 1.045),
            (0.78, 1.06),
            (1.10, 1.25),
            (1.38, 1.395),
            (1.70, 1.435),
            (2.20, 1.445),
            (2.70, 1.415),
            (3.10, 1.20),
            (3.50, 0.985),
            (3.80, 0.935),
            (4.20, 0.885),
            (4.50, 0.845),
            (4.70, 0.80),
        ]),
        bottom: Curve::new(&[
            (0.00, 0.42),
            (0.25, 0.30),
            (0.55, 0.22),
            (0.90, 0.18),
            (1.20, 0.17),
            (3.60, 0.17),
            (4.15, 0.19),
            (4.45, 0.24),
            (4.70, 0.34),
        ]),
        shoulder: Curve::new(&[
            (0.00, 0.93),
            (0.40, 0.97),
            (0.78, 1.00),
            (1.40, 1.00),
            (2.30, 0.98),
            (3.00, 0.96),
            (3.50, 0.93),
            (3.90, 0.88),
            (4.30, 0.82),
            (4.70, 0.74),
        ]),
        half_width: Curve::new(&[
            (0.00, 0.80),
            (0.25, 0.86),
            (0.60, 0.895),
            (1.20, 0.905),
            (2.40, 0.91),
            (3.60, 0.905),
            (4.20, 0.88),
            (4.50, 0.84),
            (4.70, 0.78),
        ]),
        top_half_width: Curve::new(&[
            (0.00, 0.62),
            (0.40, 0.70),
            (0.78, 0.71),
            (1.38, 0.66),
            (2.00, 0.645),
            (2.70, 0.66),
            (3.50, 0.75),
            (4.00, 0.76),
            (4.50, 0.70),
            (4.70, 0.60),
        ]),
        crown: 0.04,
        side_bulge: 0.015,
        lower_radius: 0.07,
        top_radius: 0.05,
        front_sweep: 0.20,
        rear_sweep: 0.14,
        wells: vec![
            Well {
                z: 1.03,
                y: 0.32,
                radius: 0.37,
                inner_x: 0.58,
            },
            Well {
                z: 3.73,
                y: 0.32,
                radius: 0.37,
                inner_x: 0.58,
            },
        ],
    };
    let side_glass = vec![
        Outline::new(&[(3.24, 1.01), (2.80, 1.355), (2.36, 1.375), (2.36, 1.01)]),
        Outline::new(&[(2.12, 1.01), (2.12, 1.375), (1.64, 1.345), (1.47, 1.02)]),
    ];
    let panels = vec![
        panel(
            "bumper_f",
            Region::Any(vec![
                Region::Cap {
                    front: true,
                    y0: 0.0,
                    y1: 0.70,
                },
                Region::Box {
                    x: [0.0, 1.0],
                    y: [0.0, 0.60],
                    z: [4.18, 4.8],
                },
            ]),
            "plastic",
            Joint::Bolt {
                scale: 0.8,
                group: false,
            },
            false,
        ),
        panel(
            "bumper_r",
            Region::Any(vec![
                Region::Cap {
                    front: false,
                    y0: 0.0,
                    y1: 0.62,
                },
                Region::Box {
                    x: [0.0, 1.0],
                    y: [0.0, 0.58],
                    z: [-0.1, 0.55],
                },
            ]),
            "plastic",
            Joint::Bolt {
                scale: 0.8,
                group: false,
            },
            false,
        ),
        panel(
            "bonnet",
            Region::Any(vec![
                Region::Top { z0: 3.55, z1: 4.8 },
                Region::Cap {
                    front: true,
                    y0: 0.70,
                    y1: 2.0,
                },
            ]),
            "panel",
            Joint::Hinged {
                hinge: Edge::Rear,
                hinge_scale: 1.0,
                latch_scale: 0.8,
                latch_y: None,
                latch_x: Some(0.3),
            },
            false,
        ),
        panel(
            "bootlid",
            Region::Any(vec![
                Region::Top { z0: -0.1, z1: 0.76 },
                Region::Cap {
                    front: false,
                    y0: 0.80,
                    y1: 2.0,
                },
            ]),
            "panel",
            Joint::Hinged {
                hinge: Edge::Front,
                hinge_scale: 1.0,
                latch_scale: 0.8,
                latch_y: None,
                latch_x: Some(0.3),
            },
            false,
        ),
        panel(
            "fender",
            Region::Side(Outline::rect(3.36, 0.2, 4.8, 1.2)),
            "panel",
            Joint::Bolt {
                scale: 0.7,
                group: false,
            },
            true,
        ),
        panel(
            "door_f",
            Region::Side(Outline::new(&[
                (3.27, 0.29),
                (3.27, 0.99),
                (2.76, 1.40),
                (2.30, 1.42),
                (2.30, 0.29),
            ])),
            "panel",
            door_joint(),
            true,
        ),
        panel(
            "door_r",
            Region::Side(Outline::new(&[
                (2.18, 0.29),
                (2.18, 1.42),
                (1.62, 1.38),
                (1.40, 1.02),
                (1.40, 0.29),
            ])),
            "panel",
            door_joint(),
            true,
        ),
    ];
    let mut zones = vec![
        Zone {
            region: Region::Top { z0: 2.74, z1: 3.47 },
            grade: "glass".into(),
        },
        Zone {
            region: Region::Top { z0: 0.81, z1: 1.35 },
            grade: "glass".into(),
        },
    ];
    for o in &side_glass {
        zones.push(Zone {
            region: Region::Side(o.clone()),
            grade: "glass".into(),
        });
    }
    zones.extend([Zone {
        region: Region::Floor { z0: -1.0, z1: 6.0 },
        grade: "floor".into(),
    }]);
    let members = vec![
        // Front crash rails: under the floor from the cabin, up the firewall, to the
        // bumper beam; graded strong (cabin) to weak (crash boxes), beaded.
        member(
            "rail_f",
            &[
                (0.44, 0.24, 2.70),
                (0.44, 0.25, 3.20),
                (0.44, 0.40, 3.55),
                (0.44, 0.43, 4.56),
            ],
            0.14,
            0.24,
            UP,
            "crush",
            [2.0, 0.5],
            3,
            true,
            2,
        ),
        // Upper rails (aprons) along the inner fender tops.
        member(
            "apron",
            &[(0.70, 0.84, 3.40), (0.70, 0.78, 3.90), (0.64, 0.68, 4.42)],
            0.14,
            0.14,
            UP,
            "crush",
            [1.4, 0.5],
            3,
            true,
            2,
        ),
        member(
            "tower_f",
            &[(0.47, 0.44, 3.73), (0.62, 0.80, 3.73)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [1.5, 1.5],
            0,
            true,
            2,
        ),
        member(
            "tower_r",
            &[(0.47, 0.42, 1.03), (0.62, 0.80, 1.03)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [1.5, 1.5],
            0,
            true,
            2,
        ),
        stations(
            member(
                "sill",
                &[(0.80, 0.30, 1.34), (0.80, 0.30, 3.38)],
                0.12,
                0.24,
                UP,
                "uhss",
                [1.0, 1.0],
                0,
                true,
                3,
            ),
            0.4,
        ),
        stations(
            member(
                "pillar_a",
                &[
                    (0.80, 0.33, 3.34),
                    (0.80, 0.92, 3.38),
                    (0.71, 1.10, 3.10),
                    (0.62, 1.36, 2.74),
                ],
                0.10,
                0.10,
                OUT,
                "uhss",
                [1.2, 1.2],
                0,
                true,
                2,
            ),
            0.3,
        ),
        stations(
            member(
                "pillar_b",
                &[(0.80, 0.33, 2.24), (0.79, 0.98, 2.24), (0.63, 1.38, 2.24)],
                0.10,
                0.12,
                OUT,
                "boron",
                [1.0, 1.0],
                0,
                true,
                2,
            ),
            0.3,
        ),
        stations(
            member(
                "pillar_c",
                &[(0.80, 0.33, 1.34), (0.80, 0.98, 1.34), (0.62, 1.37, 1.50)],
                0.10,
                0.10,
                OUT,
                "uhss",
                [1.0, 1.0],
                0,
                true,
                2,
            ),
            0.3,
        ),
        stations(
            member(
                "roof_rail",
                &[(0.62, 1.37, 2.74), (0.62, 1.40, 2.24), (0.62, 1.38, 1.50)],
                0.10,
                0.10,
                UP,
                "uhss",
                [1.0, 1.0],
                0,
                true,
                2,
            ),
            0.4,
        ),
        cross("header_f", 0.60, 1.37, 2.76, "uhss", 1.0),
        cross("bow_b", 0.62, 1.39, 2.24, "uhss", 1.0),
        cross("header_r", 0.60, 1.36, 1.50, "uhss", 1.0),
        cross("floor_x1", 0.76, 0.25, 2.95, "uhss", 1.0),
        cross("floor_x2", 0.76, 0.25, 2.24, "boron", 1.0),
        cross("floor_x3", 0.76, 0.25, 1.55, "uhss", 1.0),
        cross("kickup", 0.80, 0.27, 1.38, "uhss", 1.0),
        stations(
            member(
                "tunnel",
                &[(0.0, 0.32, 1.50), (0.0, 0.34, 3.30)],
                0.16,
                0.20,
                UP,
                "boron",
                [1.0, 1.0],
                0,
                false,
                2,
            ),
            0.4,
        ),
        cross("firewall", 0.78, 0.30, 3.36, "uhss", 1.0),
        cross("dash", 0.76, 0.80, 3.20, "uhss", 1.0),
        cross("cowl", 0.74, 0.92, 3.44, "uhss", 1.0),
        member(
            "rad_post",
            &[(0.45, 0.34, 4.40), (0.45, 0.72, 4.40)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [0.6, 0.6],
            0,
            true,
            2,
        ),
        cross("rad_top", 0.55, 0.74, 4.40, "hsla", 0.6),
        ladder(member(
            "bumper_beam_f",
            &[
                (-0.72, 0.45, 4.49),
                (-0.40, 0.45, 4.585),
                (0.0, 0.45, 4.63),
                (0.40, 0.45, 4.585),
                (0.72, 0.45, 4.49),
            ],
            0.10,
            0.12,
            FWD,
            "alu",
            [1.0, 1.0],
            0,
            false,
            1,
        )),
        cross("subframe_f1", 0.42, 0.24, 3.55, "hsla", 1.5),
        cross("subframe_f2", 0.42, 0.25, 3.95, "hsla", 1.5),
        // Rear rails, graded the same way, to the rear bumper beam.
        member(
            "rail_r",
            &[
                (0.45, 0.21, 1.80),
                (0.45, 0.22, 1.40),
                (0.45, 0.40, 0.95),
                (0.45, 0.45, 0.17),
            ],
            0.10,
            0.13,
            UP,
            "hsla",
            [2.0, 0.5],
            3,
            true,
            2,
        ),
        ladder(member(
            "bumper_beam_r",
            &[
                (-0.70, 0.48, 0.21),
                (-0.40, 0.48, 0.14),
                (0.0, 0.48, 0.11),
                (0.40, 0.48, 0.14),
                (0.70, 0.48, 0.21),
            ],
            0.10,
            0.12,
            FWD,
            "alu",
            [1.0, 1.0],
            0,
            false,
            1,
        )),
        // Rear aprons (inner quarter panels) over the rear rails.
        member(
            "quarter_inner",
            &[(0.70, 0.98, 1.36), (0.70, 0.95, 0.80), (0.66, 0.90, 0.24)],
            0.12,
            0.12,
            UP,
            "hsla",
            [1.4, 0.6],
            0,
            true,
            2,
        ),
        cross("subframe_r1", 0.45, 0.25, 0.80, "hsla", 1.5),
        cross("subframe_r2", 0.45, 0.25, 1.26, "hsla", 1.5),
        cross("parcel", 0.72, 0.98, 1.30, "hsla", 1.0),
        // Door intrusion beams (they ride their door).
        riding(ladder(member(
            "door_beam_f",
            &[(0.83, 0.42, 3.18), (0.83, 0.62, 2.40)],
            0.10,
            0.10,
            OUT,
            "boron",
            [1.0, 1.0],
            0,
            true,
            3,
        ))),
        riding(ladder(member(
            "door_beam_r",
            &[(0.83, 0.42, 2.10), (0.83, 0.58, 1.52)],
            0.10,
            0.10,
            OUT,
            "boron",
            [1.0, 1.0],
            0,
            true,
            3,
        ))),
    ];
    let blocks = vec![
        BlockSpec {
            name: "powertrain".into(),
            centre: [0.0, 0.56, 3.98],
            half: [0.33, 0.24, 0.23],
            spacing: 0.2,
            kg: 175.0,
            mounts: 2,
            reach: 0.25,
            mount_grade: "hsla".into(),
            mount_scale: 1.0,
        },
        BlockSpec {
            name: "battery".into(),
            centre: [0.56, 0.62, 4.24],
            half: [0.09, 0.08, 0.11],
            spacing: 0.2,
            kg: 18.0,
            mounts: 2,
            reach: 0.25,
            mount_grade: "hsla".into(),
            mount_scale: 0.4,
        },
    ];
    let plates = vec![PlateSpec {
        name: "radiator".into(),
        corner: [-0.40, 0.34, 4.32],
        u: [0.80, 0.0, 0.0],
        v: [0.0, 0.34, 0.0],
        kg: 10.0,
        grade: "alu".into(),
        mount_scale: 0.4,
    }];
    let masses = vec![
        NamedMass {
            name: "driver".into(),
            kg: 75.0,
            lo: [0.10, 0.12, 2.35],
            hi: [0.80, 0.45, 3.0],
        },
        NamedMass {
            name: "front_seats".into(),
            kg: 50.0,
            lo: [-0.8, 0.12, 2.35],
            hi: [0.8, 0.45, 3.0],
        },
        NamedMass {
            name: "rear_seat".into(),
            kg: 30.0,
            lo: [-0.8, 0.12, 1.45],
            hi: [0.8, 0.45, 2.0],
        },
        NamedMass {
            name: "dashboard".into(),
            kg: 30.0,
            lo: [-0.8, 0.7, 3.05],
            hi: [0.8, 0.95, 3.5],
        },
        NamedMass {
            name: "fuel".into(),
            kg: 45.0,
            lo: [-0.6, 0.12, 1.15],
            hi: [0.6, 0.36, 1.7],
        },
        NamedMass {
            name: "interior".into(),
            kg: 50.0,
            lo: [-0.85, 0.12, 1.4],
            hi: [0.85, 0.32, 3.3],
        },
        NamedMass {
            name: "spare_wheel".into(),
            kg: 15.0,
            lo: [-0.5, 0.15, 0.3],
            hi: [0.5, 0.5, 0.8],
        },
    ];
    let wheel = |z: f32, driven: bool, spring: f32, damper: f32| Corner {
        x: 0.78,
        y: 0.32,
        z,
        tyre_radius: 0.32,
        tyre_width: 0.22,
        wheel_mass: 22.0,
        hub_mass: 14.0,
        arm_mass: 10.0,
        rays: 14,
        tyre_spring: 800_000.0,
        tyre_damp: 150.0,
        spring,
        damper,
        driven,
        braked: true,
    };
    let look = Look {
        paint: [0.42, 0.015, 0.025],
        side_glass,
        windscreen: [2.70, 3.50],
        backlight: [0.78, 1.38],
        frit: 0.05,
        headlight: Outline::new(&[
            (0.44, 0.66),
            (0.80, 0.70),
            (0.86, 0.76),
            (0.56, 0.78),
            (0.40, 0.74),
        ]),
        grille: Outline::new(&[(0.0, 0.47), (0.36, 0.48), (0.40, 0.66), (0.0, 0.68)]),
        intake: Outline::new(&[(0.0, 0.26), (0.52, 0.27), (0.48, 0.37), (0.0, 0.37)]),
        taillight: Outline::new(&[(0.42, 0.83), (0.86, 0.84), (0.86, 0.96), (0.50, 0.97)]),
        handles: vec![[2.42, 0.92], [1.56, 0.93]],
        gap: 0.005,
        plate_y: [0.40, 0.70],
        mirror: [3.22, 1.0],
    };
    VehicleDef {
        name: "Rubble Showcase Sedan".into(),
        profile,
        grades: Grades::default(),
        stations: vec![
            0.66, 0.78, 1.34, 1.42, 1.40, 2.16, 2.24, 2.32, 2.70, 3.32, 3.36, 3.50, 4.10,
        ],
        panels,
        shell_grade: "panel".into(),
        zones,
        crumple: Curve::new(&[
            (0.00, 0.45),
            (0.45, 0.6),
            (0.90, 0.9),
            (1.30, 1.6),
            (1.45, 2.2),
            (3.30, 2.2),
            (3.45, 1.2),
            (3.90, 0.8),
            (4.40, 0.55),
            (4.70, 0.45),
        ]),
        members,
        blocks,
        plates,
        masses,
        corners: vec![
            wheel(3.73, true, 30_000.0, 2_200.0),
            wheel(1.03, false, 26_000.0, 2_000.0),
        ],
        kerb_mass: 1450.0,
        look,
        frame_every: 5,
        frame_grade: "uhss".into(),
        webs: vec![
            ("apron".into(), "rail_f".into(), 0.6),
            ("quarter_inner".into(), "rail_r".into(), 0.6),
        ],
        frame_through: vec!["door".into()],
    }
}

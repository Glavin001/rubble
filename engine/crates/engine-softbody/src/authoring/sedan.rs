//! The showcase sedan: a 4.7 m mid-size four-door saloon (front engine, front
//! drive), 1,450 kg with its driver. Every number here is an editable design
//! input; `engine/docs/vehicle-authoring.md` walks through them.

use super::body::{BodyProfile, Well};
use super::curve::{Curve, Outline};
use super::material::Grades;
use super::members::{Corner, MemberSpec};
use super::vehicle::{
    BlockSpec, BraceSpec, Edge, Joint, Look, NamedMass, Panel, PlateSpec, Region, ShearSpec,
    VehicleDef, Zone,
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
        knee: None,
        beads,
        mirrored,
        attach,
        ride: false,
        section: 4,
        station: 0.0,
        catch: false,
        stiff: 1.0,
    }
}

/// A knee in a member's threshold gradient ([`MemberSpec::knee`]): scale `s`
/// at arc fraction `t`.
pub fn knee(mut m: MemberSpec, t: f32, s: f32) -> MemberSpec {
    m.knee = Some([t, s]);
    m
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

/// A box section `width` × `height` (cross members that carry mass: the dash
/// beam with the dashboard, the radiator support).
pub fn boxed(mut m: MemberSpec, width: f32, height: f32) -> MemberSpec {
    m.section = 4;
    m.width = width;
    m.height = height;
    m
}

/// A shear-panel diagonal (and its mirror).
pub fn brace(name: &str, from: [f32; 3], to: [f32; 3], grade: &str, kl: f32) -> BraceSpec {
    BraceSpec {
        name: name.into(),
        from,
        to,
        grade: grade.into(),
        kl,
        strength: 0.0,
    }
}

/// A brace that yields at `strength` N (into a crumple zone).
pub fn yielding(mut b: BraceSpec, strength: f32) -> BraceSpec {
    b.strength = strength;
    b
}

/// A heavier section (k·L × `s`).
pub fn stiff(mut m: MemberSpec, s: f32) -> MemberSpec {
    m.stiff = s;
    m
}

/// A passenger-cell member: its thresholds × [`CELL`] (the cell keeps its
/// shape in a 56 km/h crash; the crumple zones fold).
pub fn cell(mut m: MemberSpec) -> MemberSpec {
    m.scale = [m.scale[0] * CELL, m.scale[1] * CELL];
    m
}

/// A lumped screen's (glass shear panel's) yield force (N): it cracks.
pub const GLASS_YIELD: f32 = 8.0e3;

/// A roof member (roof rails, bows, headers, B-pillars): its
/// thresholds × [`ROOF`] (the roof crushes in a roll-over; the A- and C-pillars,
/// sills and floor are the cell's).
pub fn roof(mut m: MemberSpec) -> MemberSpec {
    m.scale = [m.scale[0] * ROOF, m.scale[1] * ROOF];
    m
}

/// The roof members' threshold scale.
pub const ROOF: f32 = 1.2;

/// Yield force of the roof's lumped panels (N).
pub const ROOF_YIELD: f32 = 1.2e4;

/// Yield force of the cabin's door-ring panels (N): strong in a
/// 56 km/h crash, crushed by a roll-over.
pub const CABIN_YIELD: f32 = 6.0e4;

/// The front subframe's cross members' threshold scale (a pole between the
/// engine and a rail meets them).
pub const SUBFRAME: f32 = 1.5;

/// The passenger cell's threshold scale.
pub const CELL: f32 = 4.0;

/// An impact member: pole catchers between its stations.
pub fn catching(mut m: MemberSpec) -> MemberSpec {
    m.catch = true;
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
        0.18,
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

/// A seat frame: a light rigid lattice on breakable floor mounts.
pub fn seat_frame(name: &str, centre: [f32; 3], half: [f32; 3], kg: f32) -> BlockSpec {
    BlockSpec {
        name: name.into(),
        centre,
        half,
        spacing: 0.5,
        kg,
        mounts: 3,
        reach: 0.45,
        mount_grade: "mount".into(),
        mount_scale: 3.0,
        bear: 0.0,
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

/// Yield force of the braces from the front strut towers into the cabin
/// (N): they fold with the crumple zone.
pub const END_YIELD: f32 = 2.0e4;

/// The front rails' threshold scale at their cabin end (under the floor at
/// the B-pillar), graded to 0.5 at the bumper.
pub const RAIL_CABIN: f32 = 1.0;

/// The crush grade's thresholds against `Grades::default`'s: the rails' and
/// aprons' plateau. What the front still carries when the car stops is what
/// the cell holds elastically and gives back as rebound (≈ F²/2k), so the
/// plateau is as low as the crush length allows (0.75 m at 56 km/h).
pub const CRUSH_PLATEAU: f32 = 0.65;

/// The main load paths' (rails, sills) section against their grade's k·L.
pub const RAIL_STIFF: f32 = 1.4;

/// The rear rails' section against the HSLA grade's k·L.
pub const REAR_RAIL_STIFF: f32 = 3.5;

/// The rear rails' threshold scale from the kick-up to the rear subframe
/// (they carry the suspension, the fuel and the spare wheel: in a frontal
/// crash at 40 g that is ~40 kN a side), graded to 0.5 at the bumper behind
/// [`REAR_KNEE`] (the arc fraction of the rear subframe's station, z 0.62).
pub const REAR_RAIL: f32 = 4.0;
pub const REAR_KNEE: f32 = 0.73;

/// The rear subframe's cross tubes' k·L (N): see [`Corner::cross`].
pub const REAR_CROSS: f32 = 1.5e6;

/// The sills' section against the UHSS grade's k·L.
pub const SILL_STIFF: f32 = 2.3;

/// A lumped shear panel's axial rigidity (N): the cabin box's faces and the
/// strut towers' braces into it, spread over the sections they bear on.
pub const KL_PANEL: f32 = 7.0e6;

/// A zigzag shear panel's diagonals between two members (N).
pub const KL_SHEAR: f32 = 1.0e6;

/// The cabin box's lumped X braces (the shear panels' first form).
#[allow(clippy::too_many_arguments)]
fn cabin_x(
    fb: [f32; 3],
    ft: [f32; 3],
    cw: [f32; 3],
    bb: [f32; 3],
    bt: [f32; 3],
    rb: [f32; 3],
    rt: [f32; 3],
    k: f32,
) -> Vec<BraceSpec> {
    let m = |p: [f32; 3]| [-p[0], p[1], p[2]];
    vec![
        brace("floor_f", fb, m(bb), "floor", k),
        brace("floor_r", bb, m(rb), "floor", k),
        yielding(brace("roof_f", ft, m(bt), "panel", k), ROOF_YIELD),
        yielding(brace("roof_r", bt, m(rt), "panel", k), ROOF_YIELD),
        brace("firewall_x", fb, m(cw), "hsla", k),
        yielding(brace("windscreen_x", cw, m(ft), "glass", k), GLASS_YIELD),
        brace("bulkhead_x", rb, m(rt), "hsla", k),
        yielding(brace("ring_f_x", fb, bt, "uhss", k), CABIN_YIELD),
        yielding(brace("ring_f_y", bb, ft, "uhss", k), CABIN_YIELD),
        yielding(brace("ring_r_x", bb, rt, "uhss", k), CABIN_YIELD),
        yielding(brace("ring_r_y", rb, bt, "uhss", k), CABIN_YIELD),
    ]
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
            (2.55, 1.425),
            (2.95, 1.22),
            (3.33, 1.00),
            (3.80, 0.945),
            (4.20, 0.875),
            (4.50, 0.825),
            (4.70, 0.765),
        ]),
        bottom: Curve::new(&[
            (0.00, 0.42),
            (0.25, 0.30),
            (0.55, 0.22),
            (0.90, 0.18),
            (1.20, 0.17),
            (3.60, 0.17),
            (4.15, 0.19),
            (4.45, 0.23),
            (4.70, 0.31),
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
            (4.30, 0.80),
            (4.70, 0.70),
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
            (2.55, 0.66),
            (3.35, 0.75),
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
                y: 0.335,
                radius: 0.385,
                inner_x: 0.58,
            },
            Well {
                z: 3.73,
                y: 0.335,
                radius: 0.385,
                inner_x: 0.58,
            },
        ],
    };
    let side_glass = vec![
        Outline::new(&[(3.09, 1.01), (2.65, 1.355), (2.36, 1.375), (2.36, 1.01)]),
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
                scale: 1.0,
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
            // Bolted along its top and to the A-pillar: holds at 56 km/h
            // (the fender crumples back), tears at about 80 and up.
            Joint::Bolt {
                scale: 2.0,
                group: false,
            },
            true,
        ),
        panel(
            "door_f",
            Region::Side(Outline::new(&[
                (3.27, 0.29),
                (3.27, 0.99),
                (2.61, 1.40),
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
            region: Region::Top { z0: 2.59, z1: 3.32 },
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
    // The frame: a space frame of members that meet at shared joint nodes
    // (every path vertex is a station; a station within `members::SNAP` of
    // another member's frame node is that node). The cabin's corners: the A-,
    // B- and C-pillar bases on the sills, their tops on the roof rails; the
    // floor's cross members meet the sills, the rails (x 0.44) and the tunnel.
    let rs = RAIL_STIFF;
    let floor_cross = |name: &str, z: f32, grade: &str| -> MemberSpec {
        ladder(member(
            name,
            &[
                (-0.79, 0.27, z),
                (-0.44, 0.26, z),
                (0.0, 0.30, z),
                (0.44, 0.26, z),
                (0.79, 0.27, z),
            ],
            0.10,
            0.16,
            UP,
            grade,
            [1.0, 1.0],
            0,
            false,
            2,
        ))
    };
    let members = vec![
        // Front crash rails: under the floor from the B-pillar's cross member,
        // up the firewall, under the strut tower, to the bumper beam; graded
        // strong (cabin) to weak (crash boxes), beaded.
        stiff(
            member(
                "rail_f",
                &[
                    (0.44, 0.23, 2.24),
                    (0.44, 0.24, 2.80),
                    (0.44, 0.31, 3.36),
                    (0.44, 0.41, 3.73),
                    (0.44, 0.42, 4.10),
                    (0.44, 0.43, 4.40),
                    (0.44, 0.43, 4.57),
                ],
                0.14,
                0.24,
                UP,
                "crush",
                [RAIL_CABIN, 0.5],
                3,
                true,
                1,
            ),
            rs,
        ),
        // Upper rails (aprons) from the A-pillar at the cowl over the strut
        // tower to the radiator support.
        ladder(member(
            "apron",
            &[(0.79, 0.92, 3.38), (0.66, 0.82, 3.73), (0.55, 0.74, 4.40)],
            0.14,
            0.14,
            UP,
            "crush",
            [1.4, 0.5],
            3,
            true,
            1,
        )),
        // Strut towers: the rail to the apron.
        ladder(member(
            "tower_f",
            &[(0.44, 0.41, 3.73), (0.66, 0.82, 3.73)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [1.5, 1.5],
            0,
            true,
            1,
        )),
        ladder(member(
            "tower_r",
            &[(0.45, 0.40, 1.03), (0.68, 0.86, 1.03)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [1.5, 1.5],
            0,
            true,
            1,
        )),
        catching(stiff(
            cell(member(
                "sill",
                &[
                    (0.79, 0.31, 1.36),
                    (0.79, 0.31, 1.80),
                    (0.79, 0.31, 2.24),
                    (0.79, 0.31, 2.80),
                    (0.79, 0.31, 3.36),
                ],
                0.18,
                0.24,
                UP,
                "uhss",
                [1.0, 1.0],
                0,
                true,
                2,
            )),
            SILL_STIFF,
        )),
        cell(member(
            "pillar_a",
            &[
                (0.79, 0.31, 3.36),
                (0.79, 0.92, 3.38),
                (0.71, 1.10, 2.95),
                (0.62, 1.37, 2.59),
            ],
            0.10,
            0.12,
            FWD,
            "uhss",
            [1.2, 1.2],
            0,
            true,
            1,
        )),
        catching(stations(
            roof(member(
                "pillar_b",
                &[(0.79, 0.31, 2.24), (0.79, 0.98, 2.24), (0.62, 1.39, 2.24)],
                0.14,
                0.14,
                OUT,
                "boron",
                [1.0, 1.0],
                0,
                true,
                1,
            )),
            0.4,
        )),
        ladder(cell(member(
            "pillar_c",
            &[(0.79, 0.31, 1.36), (0.79, 0.98, 1.36), (0.62, 1.37, 1.50)],
            0.10,
            0.12,
            FWD,
            "uhss",
            [1.0, 1.0],
            0,
            true,
            1,
        ))),
        ladder(roof(member(
            "roof_rail",
            &[
                (0.62, 1.37, 2.59),
                (0.62, 1.39, 2.24),
                (0.62, 1.39, 1.79),
                (0.62, 1.37, 1.50),
            ],
            0.10,
            0.10,
            UP,
            "uhss",
            [1.0, 1.0],
            0,
            true,
            1,
        ))),
        roof(cross("header_f", 0.62, 1.37, 2.59, "uhss", 1.0)),
        roof(cross("bow_b", 0.62, 1.39, 2.24, "uhss", 1.0)),
        // A rear roof bow (the roof skin's span between bows sags otherwise)
        // under a ring of the roof skin, with a station under each of its
        // nodes (x 0, ±0.31): the skin's flat centre otherwise snaps through
        // (it inverts without straining a beam) and stays 2 cm down after a
        // frontal crash.
        stations(roof(cross("bow_r", 0.62, 1.39, 1.79, "uhss", 1.0)), 0.31),
        roof(cross("header_r", 0.62, 1.37, 1.50, "uhss", 1.0)),
        catching(cell(floor_cross("floor_x1", 2.80, "uhss"))),
        catching(cell(floor_cross("floor_x2", 2.24, "boron"))),
        cell(floor_cross("floor_x3", 1.80, "uhss")),
        cell(floor_cross("kickup", 1.36, "uhss")),
        catching(cell(member(
            "tunnel",
            &[
                (0.0, 0.33, 1.36),
                (0.0, 0.33, 1.80),
                (0.0, 0.33, 2.24),
                (0.0, 0.33, 2.80),
                (0.0, 0.33, 3.36),
            ],
            0.16,
            0.20,
            UP,
            "boron",
            [1.0, 1.0],
            0,
            false,
            1,
        ))),
        catching(floor_cross("firewall", 3.36, "boron")),
        ladder(cell(member(
            "dash",
            &[
                (-0.79, 0.80, 3.15),
                (-0.38, 0.80, 3.09),
                (0.38, 0.80, 3.09),
                (0.79, 0.80, 3.15),
            ],
            0.10,
            0.18,
            UP,
            "uhss",
            [1.0, 1.0],
            0,
            false,
            2,
        ))),
        ladder(cell(member(
            "cowl",
            &[
                (-0.79, 0.92, 3.38),
                (-0.40, 0.92, 3.29),
                (0.40, 0.92, 3.29),
                (0.79, 0.92, 3.38),
            ],
            0.10,
            0.18,
            UP,
            "uhss",
            [1.0, 1.0],
            0,
            false,
            2,
        ))),
        ladder(member(
            "rad_post",
            &[(0.44, 0.43, 4.40), (0.55, 0.74, 4.40)],
            0.10,
            0.10,
            FWD,
            "hsla",
            [0.6, 0.6],
            0,
            true,
            1,
        )),
        cross("rad_top", 0.55, 0.74, 4.40, "hsla", 0.6),
        catching(ladder(member(
            "bumper_beam_f",
            &[
                (-0.72, 0.45, 4.49),
                (-0.44, 0.45, 4.57),
                (0.0, 0.45, 4.63),
                (0.44, 0.45, 4.57),
                (0.72, 0.45, 4.49),
            ],
            0.10,
            0.12,
            FWD,
            "uhss",
            [1.0, 1.0],
            0,
            false,
            1,
        ))),
        catching(ladder(member(
            "subframe_f1",
            &[(-0.44, 0.29, 3.73), (0.0, 0.27, 3.73), (0.44, 0.29, 3.73)],
            0.10,
            0.18,
            UP,
            "hsla",
            [SUBFRAME, SUBFRAME],
            0,
            false,
            2,
        ))),
        catching(ladder(member(
            "subframe_f2",
            &[(-0.44, 0.30, 4.10), (0.0, 0.28, 4.10), (0.44, 0.30, 4.10)],
            0.10,
            0.18,
            UP,
            "hsla",
            [SUBFRAME, SUBFRAME],
            0,
            false,
            2,
        ))),
        // The steering column, from the dash cross beam (a box: it is a
        // cantilever).
        member(
            "column",
            &[(0.38, 0.80, 3.09), (0.38, 0.93, 2.83)],
            0.10,
            0.10,
            UP,
            "hsla",
            [1.0, 1.0],
            0,
            false,
            0,
        ),
        // Rear rails from the rear seat's cross member over the kick-up,
        // under the rear strut tower, to the rear bumper beam.
        knee(
            stiff(
                member(
                    "rail_r",
                    &[
                        (0.44, 0.22, 1.80),
                        (0.44, 0.24, 1.36),
                        (0.44, 0.40, 1.03),
                        (0.44, 0.42, 0.62),
                        (0.44, 0.45, 0.17),
                    ],
                    0.10,
                    0.13,
                    UP,
                    "hsla",
                    [REAR_RAIL, 0.5],
                    3,
                    true,
                    1,
                ),
                REAR_RAIL_STIFF,
            ),
            REAR_KNEE,
            REAR_RAIL,
        ),
        catching(ladder(member(
            "bumper_beam_r",
            &[
                (-0.70, 0.48, 0.21),
                (-0.44, 0.48, 0.15),
                (0.0, 0.48, 0.11),
                (0.44, 0.48, 0.15),
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
        ))),
        // Rear aprons (inner quarter panels) from the C-pillar over the rear
        // strut tower.
        ladder(member(
            "quarter_inner",
            &[(0.79, 0.98, 1.36), (0.68, 0.86, 1.03), (0.66, 0.88, 0.24)],
            0.12,
            0.12,
            UP,
            "hsla",
            [1.4, 0.6],
            0,
            true,
            1,
        )),
        ladder(member(
            "subframe_r1",
            &[(-0.44, 0.31, 1.03), (0.0, 0.29, 1.03), (0.44, 0.31, 1.03)],
            0.10,
            0.18,
            UP,
            "hsla",
            [1.5, 1.5],
            0,
            false,
            2,
        )),
        ladder(member(
            "subframe_r2",
            &[(-0.44, 0.33, 0.62), (0.0, 0.31, 0.62), (0.44, 0.33, 0.62)],
            0.10,
            0.18,
            UP,
            "hsla",
            [1.5, 1.5],
            0,
            false,
            2,
        )),
        ladder(cell(member(
            "parcel",
            &[
                (-0.79, 0.98, 1.36),
                (-0.44, 1.00, 1.36),
                (0.44, 1.00, 1.36),
                (0.79, 0.98, 1.36),
            ],
            0.10,
            0.18,
            UP,
            "hsla",
            [1.0, 1.0],
            0,
            false,
            1,
        ))),
        // Door intrusion beams (they ride their door).
        catching(stations(
            riding(ladder(member(
                "door_beam_f",
                &[(0.83, 0.42, 3.18), (0.83, 0.62, 2.40)],
                0.10,
                0.14,
                OUT,
                "boron",
                [1.0, 1.0],
                0,
                true,
                3,
            ))),
            0.28,
        )),
        // Belt reinforcements under the windows (they ride their door).
        stations(
            riding(ladder(member(
                "door_belt_f",
                &[(0.84, 0.92, 3.20), (0.84, 0.94, 2.36)],
                0.10,
                0.14,
                OUT,
                "boron",
                [1.0, 1.0],
                0,
                true,
                3,
            ))),
            0.28,
        ),
        stations(
            riding(ladder(member(
                "door_belt_r",
                &[(0.84, 0.94, 2.12), (0.84, 0.94, 1.48)],
                0.10,
                0.14,
                OUT,
                "boron",
                [1.0, 1.0],
                0,
                true,
                3,
            ))),
            0.28,
        ),
        // Inner rails at mid height (they ride their door).
        stations(
            riding(ladder(member(
                "door_mid_f",
                &[(0.83, 0.74, 3.22), (0.83, 0.76, 2.36)],
                0.10,
                0.14,
                OUT,
                "boron",
                [1.0, 1.0],
                0,
                true,
                3,
            ))),
            0.28,
        ),
        stations(
            riding(ladder(member(
                "door_mid_r",
                &[(0.83, 0.76, 2.12), (0.83, 0.74, 1.50)],
                0.10,
                0.14,
                OUT,
                "boron",
                [1.0, 1.0],
                0,
                true,
                3,
            ))),
            0.28,
        ),
        catching(stations(
            riding(ladder(member(
                "door_beam_r",
                &[(0.83, 0.42, 2.10), (0.83, 0.58, 1.52)],
                0.10,
                0.14,
                OUT,
                "boron",
                [1.0, 1.0],
                0,
                true,
                3,
            ))),
            0.28,
        )),
    ];
    let blocks = vec![
        BlockSpec {
            name: "powertrain".into(),
            centre: [0.0, 0.56, 3.98],
            half: [0.33, 0.24, 0.23],
            // Under a pole's width (static contact is per node).
            spacing: 0.17,
            kg: 175.0,
            mounts: 2,
            reach: 0.25,
            mount_grade: "mount".into(),
            mount_scale: 1.0,
            bear: 0.5,
        },
        // Seat frames (the seats ride them): as heavy as their own lattice
        // needs; the rest of the seats' mass is a named mass on the floor
        // (`front_seats`, `rear_seat`), where it steadies the structure
        // instead of springing on its mounts in a crash.
        seat_frame("seat_fl", [0.38, 0.40, 2.62], [0.24, 0.07, 0.24], 7.5),
        seat_frame("seat_fr", [-0.38, 0.40, 2.62], [0.24, 0.07, 0.24], 7.5),
        seat_frame("seat_rear", [0.0, 0.40, 1.74], [0.62, 0.07, 0.22], 12.5),
        BlockSpec {
            name: "battery".into(),
            centre: [0.56, 0.62, 4.24],
            half: [0.09, 0.08, 0.11],
            spacing: 0.3,
            kg: 18.0,
            mounts: 2,
            reach: 0.25,
            mount_grade: "mount".into(),
            mount_scale: 3.0,
            bear: 0.0,
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
    // Each fills the stability floors of the frame nodes in its box first
    // (`Skeleton::add_mass_by_need`): the boxes span the structure that
    // carries it (seats and occupants: the floor members, sills, tunnel and
    // the pillars' feet, where the belts anchor; trim: the whole cabin).
    let masses = vec![
        NamedMass {
            name: "driver".into(),
            kg: 75.0,
            lo: [0.12, 0.12, 2.24],
            hi: [0.85, 1.0, 3.05],
        },
        NamedMass {
            name: "front_seats".into(),
            kg: 35.0,
            lo: [-0.85, 0.12, 2.24],
            hi: [0.85, 1.0, 3.05],
        },
        NamedMass {
            name: "rear_seat".into(),
            kg: 17.5,
            lo: [-0.85, 0.12, 1.30],
            hi: [0.85, 1.0, 2.05],
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
            lo: [-0.9, 0.12, 1.30],
            hi: [0.9, 1.45, 3.40],
        },
        NamedMass {
            name: "spare_wheel".into(),
            kg: 15.0,
            lo: [-0.5, 0.15, 0.3],
            hi: [0.5, 0.5, 0.8],
        },
    ];
    let wheel = |z: f32, driven: bool, spring: f32, damper: f32, cross: f32| Corner {
        x: 0.78,
        y: 0.335,
        z,
        tyre_radius: 0.335,
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
        cross,
    };
    let look = Look {
        paint: [0.42, 0.015, 0.025],
        side_glass,
        windscreen: [2.55, 3.33],
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
        grades: {
            let mut g = Grades::default();
            g.crush = g.crush.scaled(CRUSH_PLATEAU);
            g
        },
        // Door seams (each door's edge ring inside it, the pillar's between),
        // screens' ends, the bonnet's and boot lid's edges.
        stations: vec![
            0.78, 1.30, 1.52, 2.06, 2.24, 2.42, 2.59, 2.74, 3.16, 3.32, 3.48, 4.10,
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
            wheel(3.73, true, 30_000.0, 2_200.0, 0.0),
            wheel(1.03, false, 26_000.0, 2_000.0, REAR_CROSS),
        ],
        kerb_mass: 1450.0,
        look,
        frame_every: 0,
        frame_grade: "uhss".into(),
        webs: vec![
            ("apron".into(), "rail_f".into(), 0.6),
            ("quarter_inner".into(), "rail_r".into(), 0.6),
        ],
        // The cabin as a braced box: its shear panels (floor pan, roof,
        // firewall and screen, rear bulkhead, each door ring's frame) lumped
        // into X's that share the box's corners (sill and roof-rail ends at
        // the A-, B- and C-pillars), so the box closes: a torsion load meets
        // every face at once.
        braces: {
            let k = KL_PANEL;
            let (fb, ft, cw) = ([0.79, 0.31, 3.36], [0.62, 1.37, 2.59], [0.79, 0.92, 3.38]);
            let (bb, bt) = ([0.79, 0.31, 2.24], [0.62, 1.39, 2.24]);
            let (rb, rt) = ([0.79, 0.31, 1.36], [0.62, 1.37, 1.50]);
            // Front end: strut-tower tops and bases, the rails at the firewall.
            let (tf, tfb) = ([0.66, 0.82, 3.73], [0.44, 0.41, 3.73]);
            let rail_fw = [0.44, 0.31, 3.36];
            // Rear end: strut-tower tops and bases, the parcel shelf, the rear
            // rails.
            let (tr, trb) = ([0.68, 0.86, 1.03], [0.44, 0.40, 1.03]);
            let parcel = [0.79, 0.98, 1.36];
            let (rr1, rr2) = ([0.44, 0.24, 1.36], [0.44, 0.42, 0.62]);
            // The tunnel's ends.
            let (tun_r, tun_f) = ([0.0, 0.33, 1.36], [0.0, 0.33, 3.36]);
            let m = |p: [f32; 3]| [-p[0], p[1], p[2]];
            let _ = (bb, bt, ft, rt, tun_r, tun_f);
            cabin_x(fb, ft, cw, bb, bt, rb, rt, k)
                .into_iter()
                .chain({
                    [
                        // Front end: the cowl and strut towers as one box (strut
                        // brace, towers to the cowl and firewall, the bay's floor).
                        yielding(brace("strut_x", tf, m(cw), "uhss", k), END_YIELD),
                        brace("strut_bar", tf, m(tf), "uhss", k),
                        yielding(brace("strut_cowl", tf, cw, "uhss", k), END_YIELD),
                        yielding(brace("strut_fw", tf, fb, "uhss", k), END_YIELD),
                        yielding(brace("tower_cowl", tfb, cw, "uhss", k), END_YIELD),
                        yielding(brace("bay_floor", tfb, m(rail_fw), "uhss", k), END_YIELD),
                        // Rear end: towers, parcel shelf, C-pillars, rear floor.
                        brace("rtower_x", tr, m(parcel), "uhss", k),
                        brace("rtower_bar", tr, m(tr), "uhss", k),
                        brace("rtower_parcel", tr, parcel, "uhss", k),
                        brace("rtower_c", tr, rb, "uhss", k),
                        brace("rtower_base", trb, parcel, "uhss", k),
                        brace("rfloor_x", rr1, m(rr2), "floor", k),
                        brace("rbase_x", trb, m(rr1), "floor", k),
                    ]
                })
                .collect()
        },
        shear: {
            let k = KL_SHEAR;
            let sh = |a: &str, b: &str, axis: usize, grade: &str| ShearSpec {
                a: a.into(),
                b: b.into(),
                axis,
                grade: grade.into(),
                kl: k,
                // Screens crack.
                strength: if grade == "glass" { GLASS_YIELD } else { 0.0 },
            };
            vec![
                // The cabin box's faces.
                sh("roof_rail", "roof_rail_r", 2, "panel"),
                sh("sill", "tunnel", 2, "floor"),
                sh("sill", "roof_rail", 2, "uhss"),
                sh("firewall", "cowl", 0, "hsla"),
                sh("cowl", "header_f", 0, "glass"),
                sh("kickup", "parcel", 0, "hsla"),
                sh("parcel", "header_r", 0, "glass"),
                // Front end: inner fenders, the bay's floor.
                sh("rail_f", "apron", 2, "hsla"),
                sh("subframe_f1", "subframe_f2", 0, "hsla"),
                // Rear end: rear floor, inner quarters, subframe.
                sh("rail_r", "rail_r_r", 2, "floor"),
                sh("rail_r", "quarter_inner", 2, "hsla"),
                sh("subframe_r1", "subframe_r2", 0, "hsla"),
            ]
        },
        frame_through: vec!["door".into()],
        // The bonnet too (its inner panel bears on the aprons and towers
        // under its edges): crushed, it folds over the engine bay instead of
        // sinking through the wheelhouses' skin (which then rattles in
        // self-contact).
        backed: vec![
            "door".into(),
            "fender".into(),
            "bumper".into(),
            "bonnet".into(),
        ],
        shell_braces: false,
        body_damping: 50.0,
        tiered: true,
        body_damping_reach: 2.0,
        body_damping_strength: 5000.0,
    }
}

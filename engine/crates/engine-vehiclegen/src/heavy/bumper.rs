//! The front bumper and its underrun protection beam.

use super::*;

// -- the front bumper and its underrun beam -------------------------------------------------------

/// The front bumper: a steel front underrun protection beam (UN R93: its
/// lower edge at most 400 mm over the ground) under a plastic cover, on two
/// brackets to the rails' front ends.
#[derive(Clone, Debug)]
pub struct Underrun {
    pub z_front: f64,
    pub depth: f64,
    pub lower_y: f64,
    pub upper_y: f64,
    pub half_w: f64,
    pub kg: f64,
    /// the beam's own beams (yield, break) per beam (N)
    pub beam: (f64, f64),
    /// each bracket's beams (yield, break) per beam (N); `bracket_beams` a side
    pub bracket: (f64, f64),
    pub bracket_beams: usize,
}

pub fn front_underrun(
    u: &Underrun,
    f: &LadderFrame,
    paint: Mat,
) -> (MechPart, Vec<(String, Mesh)>) {
    let mut p = part("bumper_front", "bumper", Zone::FrontLower);
    let zb = u.z_front - u.depth;
    let xs = [-u.half_w, 0.0, u.half_w];
    for (i, x) in xs.iter().enumerate() {
        for (j, y) in [u.lower_y, u.upper_y].iter().enumerate() {
            for (k, z) in [zb, u.z_front].iter().enumerate() {
                // (inset 10 mm at the ends; the middle column on the centre plane: `signum(0.0)` is 1)
                let xi = if *x == 0.0 {
                    0.0
                } else {
                    x - x.signum() * 0.01
                };
                add_node(
                    &mut p,
                    format!("bf{i}{j}{k}"),
                    DVec3::new(
                        xi,
                        y + if j == 0 { 0.005 } else { -0.005 },
                        z + if k == 0 { 0.005 } else { -0.005 },
                    ),
                    u.kg / 12.0,
                    true,
                );
            }
        }
    }
    // each column's box complete, and each column to the next
    let mut pairs = Vec::new();
    for (a, b) in complete(12) {
        if (b / 4) <= (a / 4) + 1 {
            pairs.push((a, b));
        }
    }
    p.internal.push(Internal {
        pairs,
        v: vals(0.0, 0.0, u.beam.0, u.beam.1),
        deform_group: None,
    });
    for s in [1.0f64, -1.0] {
        let col = if s > 0.0 { 2 } else { 0 };
        let from: Vec<String> = (0..4)
            .map(|q| format!("bf{col}{}{}", q / 2, q % 2))
            .chain((0..4).map(|q| format!("bf1{}{}", q / 2, q % 2)))
            .collect();
        let to = rail_nodes(f, s, &CORNERS, f.z_front - 0.7, f.z_front);
        mount(
            &mut p,
            &format!("fup_bracket_{}", side_tag(s)),
            from,
            to,
            u.bracket_beams,
            vals(0.0, 0.0, u.bracket.0, u.bracket.1),
        );
    }
    // the cover over the beam; the beam's face shows under it
    let c = DVec3::new(
        0.0,
        0.5 * (u.lower_y + u.upper_y),
        u.z_front - 0.5 * u.depth,
    );
    let hh = DVec3::new(u.half_w, 0.5 * (u.upper_y - u.lower_y), 0.5 * u.depth);
    p.mesh = rbox(
        c + DVec3::new(0.0, 0.045, 0.0),
        hh - DVec3::new(0.0, 0.045, 0.0),
        0.05,
        4,
        paint,
    );
    let mut extra = Vec::new();
    extra.push((
        "underrun_beam".into(),
        box_(
            DVec3::new(0.0, u.lower_y + 0.06, u.z_front - 0.5 * u.depth),
            DVec3::new(u.half_w - 0.04, 0.06, 0.5 * u.depth - 0.01),
            Mat::MetalBare,
            None,
        ),
    ));
    for s in [1.0f64, -1.0] {
        let sd = side_tag(s);
        let x = s * (u.half_w - 0.30);
        let y = u.upper_y - 0.16;
        let mut housing = box_(
            DVec3::new(x, y, u.z_front + 0.004),
            DVec3::new(0.22, 0.07, 0.004),
            Mat::TrimBlack,
            None,
        );
        housing.add(&box_(
            DVec3::new(s * (u.half_w - 0.62), u.lower_y + 0.20, u.z_front + 0.004),
            DVec3::new(0.06, 0.04, 0.004),
            Mat::TrimBlack,
            None,
        ));
        extra.push((format!("headlight_housing_{sd}"), housing));
        let mut lens = box_(
            DVec3::new(x, y, u.z_front + 0.011),
            DVec3::new(0.20, 0.06, 0.003),
            Mat::LampLens,
            None,
        );
        lens.add(&box_(
            DVec3::new(s * (u.half_w - 0.62), u.lower_y + 0.20, u.z_front + 0.011),
            DVec3::new(0.05, 0.03, 0.003),
            Mat::LampLens,
            None,
        ));
        extra.push((format!("headlight_{sd}"), lens));
    }
    p.meshes = extra.iter().map(|e: &(String, Mesh)| e.0.clone()).collect();
    (p, extra)
}
